"""Checkpoint 保存/恢复（原子写入）。

约定：
- 写入采用「临时文件 + fsync + os.replace」，失败不留下半截文件；
- 保存 model / optimizer / scheduler / scaler / epoch / best metric / resolved config / seed / RNG 状态；
- resume 后 epoch 与学习率必须连续（epoch 从 `epoch + 1` 继续，scheduler 状态原样恢复）；
- last / best / 周期 checkpoint 分开命名，互不覆盖；
- **epoch 语义必须显式**：`epoch_completed=True` 表示该 epoch 已完整结束（`last/best/周期` 都是这种），
  `epoch_completed=False` 表示训练中途被打断（只有 `interrupt` 会是这种），此时额外记录
  `completed_epochs`（已完整完成的 epoch 数）与 `iterations_completed`（被打断 epoch 内已完成的 iteration 数），
  供上层决定"重跑被打断的 epoch"还是"改从上一个完整 epoch 继续"。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from torch import nn

CHECKPOINT_LAST = "checkpoint_last.pth"
CHECKPOINT_BEST = "checkpoint_best.pth"
CHECKPOINT_INTERRUPT = "checkpoint_interrupt.pth"


def checkpoint_epoch_name(epoch: int) -> str:
    return f"checkpoint_epoch_{int(epoch):04d}.pth"


def atomic_torch_save(obj: Any, path: str | Path) -> Path:
    """原子保存 torch 对象：写临时文件 → fsync → os.replace。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp"
    try:
        with open(tmp, "wb") as fh:
            torch.save(obj, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return path


def build_checkpoint_payload(
    *,
    model: nn.Module,
    epoch: int,
    best_metric: float | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    config: dict | None = None,
    seed: int | None = None,
    rng_states: dict | None = None,
    extra: dict | None = None,
    epoch_completed: bool = True,
    completed_epochs: int | None = None,
    iterations_completed: int | None = None,
    interrupted: bool = False,
) -> dict[str, Any]:
    """构造 checkpoint payload（唯一来源，避免 last / best / interrupt 三处字段漂移）。

    完整 epoch（`epoch_completed=True`）未显式给出 `completed_epochs` 时自动填 `epoch + 1`；
    训练中途保存（`epoch_completed=False`）时必须由调用方给出绝对计数。
    """
    if epoch_completed and completed_epochs is None:
        completed_epochs = int(epoch) + 1
    return {
        "epoch": int(epoch),
        "epoch_completed": bool(epoch_completed),
        "completed_epochs": None if completed_epochs is None else int(completed_epochs),
        "iterations_completed": None if iterations_completed is None else int(iterations_completed),
        "best_metric": None if best_metric is None else float(best_metric),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "config": config,
        "seed": None if seed is None else int(seed),
        "rng_states": rng_states,
        "extra": {**(extra or {}), **({"interrupted": True} if interrupted else {})},
    }


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    epoch: int,
    best_metric: float | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    config: dict | None = None,
    seed: int | None = None,
    rng_states: dict | None = None,
    extra: dict | None = None,
    epoch_completed: bool = True,
    completed_epochs: int | None = None,
    iterations_completed: int | None = None,
) -> Path:
    """保存 checkpoint（原子）。"""
    payload = build_checkpoint_payload(
        model=model,
        epoch=epoch,
        best_metric=best_metric,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        config=config,
        seed=seed,
        rng_states=rng_states,
        extra=extra,
        epoch_completed=epoch_completed,
        completed_epochs=completed_epochs,
        iterations_completed=iterations_completed,
    )
    return atomic_torch_save(payload, path)


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict:
    """加载 checkpoint 并可选地恢复各组件状态；返回元信息 dict。"""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint 不存在: {path}")
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if model is not None:
        model.load_state_dict(payload["model"], strict=strict)
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    return {
        "epoch": int(payload.get("epoch", -1)),
        "epoch_completed": bool(payload.get("epoch_completed", True)),
        "completed_epochs": payload.get("completed_epochs"),
        "iterations_completed": payload.get("iterations_completed"),
        "best_metric": payload.get("best_metric"),
        "config": payload.get("config"),
        "seed": payload.get("seed"),
        "rng_states": payload.get("rng_states"),
        "extra": payload.get("extra", {}),
        "path": str(path),
    }


def verify_checkpoint_architecture(
    path: str | Path,
    expected_identity: dict,
    *,
    map_location: str | torch.device = "cpu",
) -> dict:
    """在**任何 state_dict 加载之前**核对 checkpoint 的架构身份（v2.3 Residual-Encoder 与 legacy 严格隔离）。

    从 checkpoint 的 `config.architecture` 或 `extra.architecture` 读取已保存身份，与 `expected_identity`
    （name/version/sha256/encoder_type）比对：缺失或不一致抛 `ArchitectureIdentityError`（含 legacy 检测）。
    这样旧 PlainConv checkpoint 加载到新骨干时给出**清楚的拒绝错误**，而不是含糊的 key mismatch；
    且绝不通过 `strict=False` 绕过（本函数不加载权重，只读元信息）。
    """
    from ..config.architecture import verify_architecture_identity

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint 不存在: {path}")
    payload = torch.load(path, map_location=map_location, weights_only=False)
    config = payload.get("config") or {}
    extra = payload.get("extra") or {}
    found = config.get("architecture") or extra.get("architecture")
    verify_architecture_identity(expected_identity, found)
    return dict(found or {})


class CheckpointManager:
    """管理 run 目录下的 last / best / 周期 checkpoint（互不覆盖）。"""

    def __init__(self, run_dir: str | Path, *, save_every: int = 50) -> None:
        if int(save_every) < 1:
            raise ValueError(f"save_every 必须 >= 1，收到 {save_every}")
        self.checkpoint_dir = Path(run_dir) / "checkpoints"
        self.save_every = int(save_every)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    @property
    def last_path(self) -> Path:
        return self.checkpoint_dir / CHECKPOINT_LAST

    @property
    def best_path(self) -> Path:
        return self.checkpoint_dir / CHECKPOINT_BEST

    @property
    def interrupt_path(self) -> Path:
        return self.checkpoint_dir / CHECKPOINT_INTERRUPT

    def epoch_path(self, epoch: int) -> Path:
        return self.checkpoint_dir / checkpoint_epoch_name(epoch)

    def save(
        self,
        *,
        model: nn.Module,
        epoch: int,
        best_metric: float | None,
        optimizer=None,
        scheduler=None,
        scaler=None,
        config: dict | None = None,
        seed: int | None = None,
        rng_states: dict | None = None,
        extra: dict | None = None,
        is_best: bool = False,
        interruption: bool = False,
        epoch_completed: bool = True,
        completed_epochs: int | None = None,
        iterations_completed: int | None = None,
    ) -> dict[str, str]:
        """保存 last（总是）/ best（is_best）/ 周期（每 save_every epoch）/ interrupt。

        `interruption=True` 时只写 `checkpoint_interrupt.pth`（不覆盖 last/best），并强制
        `epoch_completed=False`、记录 `completed_epochs` 与 `iterations_completed`。
        """
        common = dict(
            model=model,
            epoch=epoch,
            best_metric=best_metric,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config=config,
            seed=seed,
            rng_states=rng_states,
            extra=extra,
        )
        written: dict[str, str] = {}
        if interruption:
            payload = build_checkpoint_payload(
                **common,
                epoch_completed=False,
                completed_epochs=completed_epochs,
                iterations_completed=iterations_completed,
                interrupted=True,
            )
            atomic_torch_save(payload, self.interrupt_path)
            written["interrupt"] = str(self.interrupt_path)
            return written

        # last / best / 周期 checkpoint 携带完全相同的 epoch 元数据
        done = dict(
            epoch_completed=epoch_completed,
            completed_epochs=completed_epochs,
            iterations_completed=iterations_completed,
        )
        written["last"] = str(save_checkpoint(self.last_path, **common, **done))
        if is_best:
            written["best"] = str(save_checkpoint(self.best_path, **common, **done))
        if (int(epoch) + 1) % self.save_every == 0:
            written["periodic"] = str(save_checkpoint(self.epoch_path(epoch), **common, **done))
        return written

    def list_checkpoints(self) -> list[str]:
        return sorted(str(p) for p in self.checkpoint_dir.glob("checkpoint_*.pth"))


__all__ = [
    "CHECKPOINT_BEST",
    "CHECKPOINT_INTERRUPT",
    "CHECKPOINT_LAST",
    "CheckpointManager",
    "atomic_torch_save",
    "build_checkpoint_payload",
    "checkpoint_epoch_name",
    "load_checkpoint",
    "save_checkpoint",
    "verify_checkpoint_architecture",
]
