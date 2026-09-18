"""M0 训练循环（自研；不依赖 nnUNetTrainer）。

设计要点：
- **device 显式传入**（`cpu` / `cuda:0`），import 与构造阶段**不初始化 CUDA**；构造时执行 `model.to(device)`；
- AMP 仅在 CUDA 上启用（CPU 请求 AMP 时给出明确提示而不是静默忽略）；
- epoch 与 batch 两级 tqdm 进度条；`--no-progress` 可关闭；
- 结构化日志：`metrics/epoch_metrics.csv` + `metrics/epoch_metrics.jsonl`；
- checkpoint：last（每 epoch）/ best（**checkpoint 指标**改善）/ 周期（每 N epoch）/ interrupt（Ctrl-C）；
- **两种验证模式**：
  - `diagnostic_patch`：随机 validation patch（仅用于 loader smoke / small-overfit 诊断），每个 epoch 都验证；
  - `formal_full_volume`：按 `validation_every_n_epochs`（正式 M0–M4 = 5）**低频**调用 `FullVolumeValidator`，
    遍历 fold 0 全部 validation study 做完整 3D 滑窗推理；**最后一个 epoch 即使不是 N 的倍数也必验证**；
    非验证 epoch 不调用 validator、不解析 checkpoint 指标、不更新 best/patience，日志标记 `validation_skipped=1`，
    但 `checkpoint_last` 与周期 checkpoint 仍正常保存（正式训练必须用这个模式）；
- checkpoint 指标按**显式字段名**从 epoch 指标中查找（缺失或非有限值立即报错，不做含糊回退）；
- **best checkpoint 只在有效 validation event 上产生**；
- early stopping（可选，正式 M0–M4 禁用）与 checkpoint 使用同一指标、同一 `min_delta`；
  **patience 单位 = validation events**（非 epoch），`min_epochs` 单位 = training epochs，只能在 validation event 后触发；
- resume 后 epoch / LR / 采样序列 / validation-event 计数 / best / early-stopping 状态连续；
- 不静默吞异常：数据耗尽、非有限 loss/指标、异常退出都会显式报错。

数据接口：`batch_source()` 每次调用返回一个新的迭代器，产出 `(data, seg)` CPU 张量。
训练器只负责 forward/backward/日志/checkpoint，不读取任何数据文件。
"""
from __future__ import annotations

import contextlib
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch
from torch import nn

from ..evaluation.full_volume_validator import FULL_VOLUME_METRIC_FIELDS
from ..utils.progress import make_progress
from ..utils.reproducibility import capture_rng_states, environment_info, restore_rng_states, set_seed
from .checkpointing import CheckpointManager, load_checkpoint
from .metrics import MetricAccumulator, counts_from_logits

BatchSource = Callable[[], Iterable[tuple[torch.Tensor, torch.Tensor]]]

# 规范名称（与配置层 `validation.mode` 一致）：`full_volume` / `diagnostic_patch`
VALIDATION_MODE_FULL_VOLUME = "full_volume"
VALIDATION_MODE_DIAGNOSTIC = "diagnostic_patch"
# 别名：文档与研究计划中使用的 `formal_full_volume` 等价于 `full_volume`
VALIDATION_MODE_FORMAL = "formal_full_volume"
VALIDATION_MODE_ALIASES = {
    VALIDATION_MODE_FULL_VOLUME: VALIDATION_MODE_FULL_VOLUME,
    VALIDATION_MODE_FORMAL: VALIDATION_MODE_FULL_VOLUME,
    VALIDATION_MODE_DIAGNOSTIC: VALIDATION_MODE_DIAGNOSTIC,
}
VALIDATION_MODES = (VALIDATION_MODE_FULL_VOLUME, VALIDATION_MODE_DIAGNOSTIC)


def _flatten_mapping(payload: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """把嵌套配置扁平成 `a.b.c → 值` 的映射（列表/标量按叶子处理）。"""
    flat: dict[str, Any] = {}
    for key, value in payload.items():
        path = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(_flatten_mapping(value, prefix=f"{path}."))
        else:
            flat[path] = value
    return flat


@dataclass(frozen=True)
class EarlyStoppingConfig:
    """early stopping 规则（与 checkpoint 共用同一指标与 `min_delta`）。

    单位约定（v2.2）：`patience` 以 **validation event** 计数（不是 epoch）；`min_epochs` 以
    **training epoch** 计数。正式 M0–M4 使用固定 200-epoch 预算，`enabled=False`（不启用“连续无改善”
    停止，以保证所有模型与 seed 获得相同训练预算）；本配置仅供工程调试或显式启用时使用。
    """

    enabled: bool = False
    min_epochs: int = 1
    patience: int = 10
    min_delta: float = 0.0

    def validate(self, max_epochs: int) -> None:
        if int(self.min_epochs) < 1:
            raise ValueError(f"early_stopping.min_epochs 必须 >= 1，收到 {self.min_epochs}")
        if int(self.patience) < 1:
            raise ValueError(f"early_stopping.patience 必须 >= 1，收到 {self.patience}")
        if float(self.min_delta) < 0:
            raise ValueError(f"early_stopping.min_delta 必须 >= 0，收到 {self.min_delta}")
        if int(self.min_epochs) > int(max_epochs):
            raise ValueError(
                f"early_stopping.min_epochs={self.min_epochs} 不能大于最大 epoch 数 {max_epochs}"
            )


@dataclass
class TrainerSummary:
    """训练结束摘要（写入日志与返回值）。"""

    status: str
    epochs_done: int
    best_metric: float | None
    best_epoch: int | None
    run_dir: str
    elapsed_sec: float
    interrupted: bool = False
    early_stopped: bool = False
    stop_reason: str | None = None
    bad_validation_events: int = 0
    validation_events_completed: int = 0
    checkpoint_metric: str = ""
    last_epoch_metrics: dict[str, float] | None = None


class M0Trainer:
    """自研 M0 训练器。"""

    def __init__(
        self,
        *,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any | None,
        device: str | torch.device,
        train_batches: BatchSource,
        val_batches: BatchSource | None = None,
        run_dir: str | Path,
        epochs: int,
        iterations_per_epoch: int,
        validation_iterations: int,
        grad_clip: float = 12.0,
        amp: bool = False,
        progress: bool = True,
        checkpoint_every: int = 50,
        checkpoint_metric: str = "val_loss",
        maximize: bool = False,
        min_delta: float = 0.0,
        validation_mode: str = VALIDATION_MODE_DIAGNOSTIC,
        validator: Any | None = None,
        validation_every_n_epochs: int = 1,
        early_stopping: EarlyStoppingConfig | None = None,
        config_snapshot: dict | None = None,
        seed: int | None = None,
        metrics_dir: str | Path | None = None,
        diagnostics_dir: str | Path | None = None,
        data_sources: Sequence[Any] = (),
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = torch.device(device)
        self.train_batches = train_batches
        self.val_batches = val_batches
        self.run_dir = Path(run_dir)
        self.epochs = int(epochs)
        self.iterations_per_epoch = int(iterations_per_epoch)
        self.validation_iterations = int(validation_iterations)
        self.grad_clip = float(grad_clip)
        self.progress = bool(progress)
        self.checkpoint_metric = str(checkpoint_metric)
        self.maximize = bool(maximize)
        self.min_delta = float(min_delta)
        requested_mode = str(validation_mode)
        if requested_mode not in VALIDATION_MODE_ALIASES:
            raise ValueError(
                f"validation_mode 必须是 {sorted(VALIDATION_MODE_ALIASES)} 之一"
                f"（`{VALIDATION_MODE_FORMAL}` 是 `{VALIDATION_MODE_FULL_VOLUME}` 的别名），收到 {requested_mode!r}"
            )
        self.validation_mode = VALIDATION_MODE_ALIASES[requested_mode]
        self.validator = validator
        self.validation_every_n_epochs = int(validation_every_n_epochs)
        self.early_stopping = early_stopping or EarlyStoppingConfig()
        self.config_snapshot = config_snapshot or {}
        self.seed = None if seed is None else int(seed)

        if self.epochs < 1 or self.iterations_per_epoch < 1 or self.validation_iterations < 1:
            raise ValueError("epochs / iterations_per_epoch / validation_iterations 必须 >= 1")
        if self.validation_every_n_epochs < 1:
            raise ValueError(
                f"validation_every_n_epochs 必须 >= 1（0 或负数非法；正式 M0–M4 使用 5），"
                f"收到 {self.validation_every_n_epochs}"
            )
        if not self.checkpoint_metric:
            raise ValueError("checkpoint_metric 不能为空")
        if self.validation_mode == VALIDATION_MODE_FULL_VOLUME:
            if self.validator is None:
                raise ValueError(
                    f"{VALIDATION_MODE_FULL_VOLUME}（正式）模式必须提供 validator（FullVolumeValidator 实例）"
                )
            if self.val_batches is not None:
                raise ValueError(
                    f"{VALIDATION_MODE_FULL_VOLUME} 模式不接受 val_batches："
                    "正式训练不得创建随机 validation patch provider"
                )
            if self.checkpoint_metric == "val_loss":
                raise ValueError("正式（full_volume）模式不得使用 val_loss 作为 checkpoint 指标")
            if not self.maximize:
                raise ValueError(
                    "正式（full_volume）模式的 checkpoint 指标是最大化指标（阳性 case-wise Dice），"
                    "maximize 必须为 True"
                )
        else:
            if self.val_batches is None:
                raise ValueError("diagnostic_patch 模式必须提供 val_batches")
        self.early_stopping.validate(self.epochs)
        if self.early_stopping.enabled and not math.isclose(
            float(self.early_stopping.min_delta), self.min_delta, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                "early stopping 与 checkpoint 必须使用同一个 min_delta："
                f"early_stopping.min_delta={self.early_stopping.min_delta} vs min_delta={self.min_delta}"
            )
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"device={self.device} 但当前进程看不到可用 CUDA；请检查 CUDA_VISIBLE_DEVICES 或改用 --device cpu"
            )
        # 关键：把模型搬到与输入相同的 device（否则 CUDA 训练会因 "CUDA 输入 + CPU 模型" 直接失败）
        self.model = self.model.to(self.device)
        self.data_sources = tuple(data_sources)
        for source in self.data_sources:
            if not hasattr(source, "set_epoch"):
                raise TypeError(f"data_sources 的元素必须实现 set_epoch(epoch)，收到 {type(source).__name__}")
        self.amp_requested = bool(amp)
        self.amp = bool(amp) and self.device.type == "cuda"
        if self.amp_requested and not self.amp:
            print("[trainer][WARN] 请求了 AMP，但 device 不是 CUDA，已按 FP32 运行（不做静默启用）")

        self.manager = CheckpointManager(self.run_dir, save_every=int(checkpoint_every))
        self.metrics_dir = Path(metrics_dir) if metrics_dir is not None else self.run_dir / "metrics"
        self.diagnostics_dir = Path(diagnostics_dir) if diagnostics_dir is not None else self.run_dir / "diagnostics"
        self.metrics_csv = self.metrics_dir / "epoch_metrics.csv"
        self.metrics_jsonl = self.metrics_dir / "epoch_metrics.jsonl"
        self.start_epoch = 0
        self.best_metric: float | None = None
        self.best_epoch: int | None = None
        self.bad_validation_events = 0
        self.validation_events_completed = 0
        self.last_validation_epoch: int | None = None
        self._history: list[dict[str, Any]] = []
        # 中断语义与续训安全
        self._rng_restored = False  # resume 恢复过 RNG 后，train() 不得再 set_seed 覆盖
        self._epoch_offset = 0  # 本次进程开始前已完整完成的 epoch 数（resume 时设为 start_epoch）
        self.current_epoch: int | None = None  # 正在执行的 epoch（指标/checkpoint 落盘后置 None）
        self.current_iteration: int | None = None  # 当前 epoch 内已开始的 iteration 索引
        self.current_phase: str | None = None  # "train" / "validation"

    # ------------------------------------------------------------------ 工具
    @staticmethod
    def _main_logits(outputs):
        """从（可能的）DS 输出中取最高分辨率 logits。"""
        if isinstance(outputs, (list, tuple)):
            if not outputs:
                raise ValueError("空输出列表")
            return outputs[0]
        return outputs

    def _autocast(self):
        if self.amp:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return contextlib.nullcontext()

    def _to_device(self, batch: tuple):
        """把 batch 搬到 device：``(data, seg)`` 或 ``(data, seg, zonal_prior)``（M3/M4）。"""
        if not isinstance(batch, (tuple, list)) or len(batch) not in (2, 3):
            raise RuntimeError(
                f"batch 必须是 (data, seg) 或 (data, seg, zonal_prior)，收到 {type(batch).__name__}"
                f"（长度 {len(batch) if isinstance(batch, (tuple, list)) else 'N/A'}）"
            )
        data, seg = batch[0], batch[1]
        prior = batch[2] if len(batch) == 3 else None
        if not torch.isfinite(data).all():
            raise RuntimeError("输入数据包含 NaN/Inf")
        if prior is None:
            return data.to(self.device), seg.to(self.device), None
        return data.to(self.device), seg.to(self.device), prior.to(self.device)

    def _forward_model(self, data: torch.Tensor, prior: torch.Tensor | None):
        """按模型声明的 `REQUIRED_INPUT_KEYS` 调用模型；缺 prior / 多余 prior 都立即报错。"""
        required = tuple(getattr(self.model, "REQUIRED_INPUT_KEYS", ("image",)))
        if "zonal_prior" in required:
            if prior is None:
                raise RuntimeError(
                    "模型要求 zonal_prior，但 batch 未提供（禁止用全零 prior 兜底）；"
                    "请检查数据管线是否配置 include_zonal_prior=True"
                )
            return self.model(data, zonal_prior=prior)
        if prior is not None:
            raise RuntimeError(
                "模型不读取 zonal_prior，但 batch 提供了 prior（禁止静默混用）；"
                "请检查实验配置的 model_id 与 prior 设置"
            )
        return self.model(data)

    def _backward(self, loss: torch.Tensor) -> None:
        if not torch.isfinite(loss):
            raise RuntimeError(f"loss 非有限值: {float(loss.detach())}（已中止，避免污染权重）")
        if self.amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

    # ------------------------------------------------------------- 目录/日志
    def _ensure_dirs(self) -> None:
        self.manager.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.diagnostics_dir.mkdir(parents=True, exist_ok=True)

    def _write_manifest(self) -> Path:
        self._ensure_dirs()
        manifest = {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "device": str(self.device),
            "model_parameter_device": str(next(self.model.parameters()).device),
            "amp_requested": self.amp_requested,
            "amp_active": self.amp,
            "max_epochs": self.epochs,
            "iterations_per_epoch": self.iterations_per_epoch,
            "validation_mode": self.validation_mode,
            "diagnostic_validation_iterations": self.validation_iterations,
            "checkpoint_metric": self.checkpoint_metric,
            "maximize_metric": self.maximize,
            "min_delta": self.min_delta,
            "early_stopping": asdict(self.early_stopping),
            "grad_clip": self.grad_clip,
            "seed": self.seed,
            "config": self.config_snapshot,
            "environment": environment_info(),
            "num_parameters": int(sum(p.numel() for p in self.model.parameters())),
        }
        target = self.diagnostics_dir / "run_manifest.json"
        if target.exists():
            target = self.diagnostics_dir / f"run_manifest_resume_{time.strftime('%Y%m%d_%H%M%S')}.json"
        target.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        return target

    def _append_metrics(self, row: Mapping[str, Any]) -> None:
        self._ensure_dirs()
        new_file = not self.metrics_csv.exists()
        with open(self.metrics_csv, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
            if new_file:
                writer.writeheader()
            writer.writerow(row)
        with open(self.metrics_jsonl, "a") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _checkpoint_extra(self) -> dict[str, Any]:
        es = self.early_stopping
        return {
            "checkpoint_metric": self.checkpoint_metric,
            "maximize_metric": self.maximize,
            "best_epoch": self.best_epoch,
            # v2.2：early stopping 按 validation event 计数；保留旧键名以兼容早期 checkpoint
            "bad_validation_events": self.bad_validation_events,
            "early_stopping_bad_epochs": self.bad_validation_events,
            "validation_events_completed": self.validation_events_completed,
            "last_validation_epoch": self.last_validation_epoch,
            "validation_every_n_epochs": self.validation_every_n_epochs,
            "early_stopping_enabled": bool(es.enabled),
            "min_epochs": int(es.min_epochs),
            "patience": int(es.patience),
            "min_delta": self.min_delta,
            "validation_protocol": self.config_snapshot.get("validation"),
        }

    def _resolve_metric(self, epoch_metrics: Mapping[str, Any]) -> float:
        """按显式字段名取 checkpoint 指标；缺失或非有限值立即报错。"""
        if self.checkpoint_metric not in epoch_metrics:
            raise KeyError(
                f"checkpoint 指标 {self.checkpoint_metric!r} 不在本轮指标中；"
                f"可用字段: {sorted(k for k in epoch_metrics)}"
            )
        raw = epoch_metrics[self.checkpoint_metric]
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"checkpoint 指标 {self.checkpoint_metric!r} 不是数值: {raw!r}") from exc
        if not math.isfinite(value):
            raise RuntimeError(f"checkpoint 指标 {self.checkpoint_metric!r} 非有限值: {value}")
        return value

    def _is_improvement(self, value: float) -> bool:
        if self.best_metric is None:
            return True
        if self.maximize:
            return value > self.best_metric + self.min_delta
        return value < self.best_metric - self.min_delta

    def _is_validation_epoch(self, epoch: int) -> bool:
        """判断该 epoch 结束后是否触发一次验证事件。

        - `full_volume`（正式）：每 `validation_every_n_epochs` 个 epoch 一次（1-based epoch 整除），
          且**最后一个 epoch 必验证**（即使不是 N 的倍数）；
        - `diagnostic_patch`（诊断）：每个 epoch 都验证（廉价随机 patch，作为“是否在拟合”信号）。
        """
        if self.validation_mode != VALIDATION_MODE_FULL_VOLUME:
            return True
        n = self.validation_every_n_epochs
        if n <= 1:
            return True
        is_last = epoch == self.epochs - 1
        return bool(is_last or ((epoch + 1) % n == 0))

    def _write_early_stop_summary(self, *, epoch: int, reason: str) -> Path:
        self._ensure_dirs()
        payload = {
            "early_stopped": True,
            "reason": reason,
            "stopped_after_epoch": int(epoch),
            "epochs_done": int(epoch) + 1,
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "checkpoint_metric": self.checkpoint_metric,
            "bad_validation_events": self.bad_validation_events,
            "validation_events_completed": self.validation_events_completed,
            "min_epochs": self.early_stopping.min_epochs,
            "patience": self.early_stopping.patience,
            "min_delta": self.min_delta,
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        path = self.diagnostics_dir / "early_stop_summary.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        return path

    # ------------------------------------------------------------------ 训练
    def train(self) -> TrainerSummary:
        """执行训练（含验证、日志、checkpoint、early stopping）。"""
        self._ensure_dirs()
        if self.seed is not None and not self._rng_restored:
            set_seed(self.seed, deterministic=False)
        elif self._rng_restored:
            print("[trainer] RNG 状态来自 checkpoint（不再 set_seed，保证续训精确一致）")
        manifest_path = self._write_manifest()
        print(f"[trainer] run_dir={self.run_dir}")
        print(
            f"[trainer] manifest={manifest_path}  device={self.device}  amp={self.amp}  "
            f"validation={self.validation_mode}  metric={self.checkpoint_metric} "
            f"(maximize={self.maximize}, min_delta={self.min_delta})"
        )

        t0 = time.time()
        epoch_bar = make_progress(
            range(self.start_epoch, self.epochs),
            total=self.epochs - self.start_epoch,
            desc="m0 epochs",
            disable=not self.progress,
        )
        status = "completed"
        stop_reason: str | None = None
        last_metrics: dict[str, float] | None = None
        try:
            for epoch in epoch_bar:
                if self.scheduler is not None:
                    self.scheduler.step(epoch)
                lr = float(self.optimizer.param_groups[0]["lr"])
                # 数据源按 epoch 切换采样序列（采样由 (seed, epoch, index) 决定，无需保存隐藏 RNG 状态）
                for source in self.data_sources:
                    source.set_epoch(epoch)
                self.current_epoch = epoch
                self.current_iteration = None

                epoch_t0 = time.time()
                train_metrics = self._run_epoch(epoch, train=True)
                row: dict[str, Any] = {
                    "epoch": epoch,
                    "lr": lr,
                    "train_loss": train_metrics["loss"],
                    "train_dice": train_metrics["dice"],
                    "train_tp": train_metrics["tp"],
                    "train_fp": train_metrics["fp"],
                    "train_fn": train_metrics["fn"],
                }
                # v2.2 低频验证调度：full_volume 每 every_n_epochs 一次，最后一个 epoch 必验证；
                # 非验证 epoch 不调用 validator、不解析 checkpoint 指标、不更新 best/patience。
                run_validation = self._is_validation_epoch(epoch)
                if self.validation_mode == VALIDATION_MODE_FULL_VOLUME:
                    row["validation_mode"] = VALIDATION_MODE_FULL_VOLUME
                    if run_validation:
                        self.current_phase = "validation"
                        val_metrics = self.validator.run(self.model, epoch=epoch)
                        # 复制 validator 返回的全部字段（保留“缺字段/非有限”由 _resolve_metric 报错的语义）
                        row.update({k: float(v) for k, v in val_metrics.items()})
                        row["validation_event"] = 1
                        row["validation_skipped"] = 0
                    else:
                        # 稳定 CSV schema：跳过的 epoch 用空串占位（与 validator 标准字段同集同序）
                        for key in FULL_VOLUME_METRIC_FIELDS:
                            row[key] = ""
                        row["validation_event"] = 0
                        row["validation_skipped"] = 1
                else:
                    # 诊断模式（随机 patch）：每个 epoch 都验证（廉价，作为“是否在拟合”信号）
                    diag_metrics = self._run_epoch(epoch, train=False)
                    row.update(
                        {
                            "val_loss": diag_metrics["loss"],
                            "val_dice": diag_metrics["dice"],
                            "val_tp": diag_metrics["tp"],
                            "val_fp": diag_metrics["fp"],
                            "val_fn": diag_metrics["fn"],
                            "validation_mode": VALIDATION_MODE_DIAGNOSTIC,
                            "validation_event": 1,
                            "validation_skipped": 0,
                        }
                    )
                epoch_time = time.time() - epoch_t0

                if run_validation:
                    metric_value = self._resolve_metric(row)
                    improved = self._is_improvement(metric_value)
                    self.validation_events_completed += 1
                    self.last_validation_epoch = epoch
                    if improved:
                        self.best_metric = metric_value
                        self.best_epoch = epoch
                        self.bad_validation_events = 0
                    else:
                        self.bad_validation_events += 1
                    # patience 单位 = validation events；min_epochs 单位 = training epochs
                    stop_now = bool(
                        self.early_stopping.enabled
                        and (epoch + 1) >= self.early_stopping.min_epochs
                        and self.bad_validation_events >= self.early_stopping.patience
                    )
                    metric_cell: Any = metric_value
                else:
                    improved = False
                    stop_now = False
                    metric_cell = ""

                row.update(
                    {
                        "epoch_time_s": round(epoch_time, 2),
                        "checkpoint_metric": self.checkpoint_metric,
                        "metric_value": metric_cell,
                        "is_best": int(improved),
                        "bad_validation_events": self.bad_validation_events,
                        "validation_events_completed": self.validation_events_completed,
                        "early_stopped": int(stop_now),
                    }
                )
                self._append_metrics(row)
                last_metrics = {
                    k: float(v) for k, v in row.items() if isinstance(v, (int, float)) and not isinstance(v, bool)
                }
                self._history.append(row)

                # best 只在有效 validation event 上产生（is_best 在非验证 epoch 恒为 False）；
                # last / 周期 checkpoint 无论是否验证都正常保存，保证可恢复。
                written = self.manager.save(
                    model=self.model,
                    epoch=epoch,
                    best_metric=self.best_metric,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    scaler=self.scaler if self.amp else None,
                    config=self.config_snapshot,
                    seed=self.seed,
                    rng_states=capture_rng_states(),
                    extra=self._checkpoint_extra(),
                    is_best=improved,
                    completed_epochs=epoch + 1,  # 该 epoch 已完整结束 → 绝对计数
                    iterations_completed=self.iterations_per_epoch,
                )
                postfix = {
                    "lr": f"{lr:.2e}",
                    "train_loss": f"{train_metrics['loss']:.4f}",
                }
                if run_validation:
                    postfix[self.checkpoint_metric[:12]] = f"{metric_value:.4f}"
                else:
                    postfix["val"] = "skipped"
                if hasattr(epoch_bar, "set_postfix"):
                    epoch_bar.set_postfix(postfix)
                if run_validation:
                    print(
                        f"[trainer] epoch {epoch + 1}/{self.epochs} train_loss={train_metrics['loss']:.4f} "
                        f"{self.checkpoint_metric}={metric_value:.4f} lr={lr:.3e} time={epoch_time:.1f}s "
                        f"validation_event={self.validation_events_completed} "
                        f"bad_validation_events={self.bad_validation_events} {'(best)' if improved else ''} "
                        f"ckpt={','.join(sorted(written))}"
                    )
                else:
                    next_val = min(
                        ((epoch + 1) // self.validation_every_n_epochs + 1) * self.validation_every_n_epochs,
                        self.epochs,
                    )
                    print(
                        f"[trainer] epoch {epoch + 1}/{self.epochs} train_loss={train_metrics['loss']:.4f} "
                        f"lr={lr:.3e} time={epoch_time:.1f}s validation_skipped=1 "
                        f"(下次全体积验证≈第 {next_val} epoch 或最终 epoch) ckpt={','.join(sorted(written))}"
                    )
                # 该 epoch 的指标/checkpoint 已落盘 → 视为完整完成
                self.current_epoch = None
                self.current_iteration = None
                if stop_now:
                    status = "early_stopped"
                    stop_reason = (
                        f"{self.checkpoint_metric} 连续 {self.bad_validation_events} 个 validation event 未改善"
                        f"（>= patience={self.early_stopping.patience}，patience 按 validation event 计数），"
                        f"且已完成 epoch {epoch + 1} >= min_epochs={self.early_stopping.min_epochs}"
                        f"（min_epochs 按 training epoch 计数）"
                    )
                    path = self._write_early_stop_summary(epoch=epoch, reason=stop_reason)
                    print(f"[trainer][EARLY-STOP] {stop_reason}；摘要: {path}")
                    break
        except KeyboardInterrupt:
            status = "interrupted"
            # 绝对计数：本进程完整完成的 epoch + resume 之前已完成的 epoch（_history 只含本进程）
            completed_epochs = self._epoch_offset + len(self._history)
            in_progress = self.current_epoch if self.current_epoch is not None else completed_epochs
            phase = getattr(self, "current_phase", None)
            if self.current_epoch is None:
                iterations_completed = 0  # 中断发生在两个 epoch 之间：模型等价于已完成 completed_epochs 个 epoch
            elif phase == "train":
                # 被打断的那个 iteration 尚未完成：已完成数为 current_iteration（0-based）
                iterations_completed = int(self.current_iteration or 0)
            else:
                iterations_completed = self.iterations_per_epoch  # 训练部分已完成，验证阶段被中断
            self.manager.save(
                model=self.model,
                epoch=in_progress,
                best_metric=self.best_metric,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler if self.amp else None,
                config=self.config_snapshot,
                seed=self.seed,
                rng_states=capture_rng_states(),
                extra={**self._checkpoint_extra(), "interrupted": True},
                interruption=True,
                completed_epochs=completed_epochs,
                iterations_completed=iterations_completed,
            )
            print(
                f"[trainer][INTERRUPT] 已保存 interrupt checkpoint: {self.manager.interrupt_path}"
                f"（epoch_completed=False：正在执行 epoch {in_progress}（阶段={phase}），"
                f"该 epoch 内已完成 {iterations_completed} 个 iteration，此前完整完成 {completed_epochs} 个 epoch）"
            )
            raise

        elapsed = time.time() - t0
        epochs_done = (self._history[-1]["epoch"] + 1) if self._history else self.start_epoch
        summary = TrainerSummary(
            status=status,
            epochs_done=epochs_done,
            best_metric=self.best_metric,
            best_epoch=self.best_epoch,
            run_dir=str(self.run_dir),
            elapsed_sec=elapsed,
            interrupted=status == "interrupted",
            early_stopped=status == "early_stopped",
            stop_reason=stop_reason,
            bad_validation_events=self.bad_validation_events,
            validation_events_completed=self.validation_events_completed,
            checkpoint_metric=self.checkpoint_metric,
            last_epoch_metrics=last_metrics,
        )
        self._print_summary(summary)
        return summary

    def _print_summary(self, summary: TrainerSummary) -> None:
        print(
            f"[trainer] 结束: status={summary.status} epochs_done={summary.epochs_done} "
            f"validation_events={summary.validation_events_completed} "
            f"best_epoch={summary.best_epoch} best_{summary.checkpoint_metric}={summary.best_metric} "
            f"bad_validation_events={summary.bad_validation_events} 耗时={summary.elapsed_sec:.1f}s"
        )
        if summary.stop_reason:
            print(f"[trainer] 停止原因: {summary.stop_reason}")
        print(
            f"[trainer] 输出: checkpoints={self.manager.checkpoint_dir} "
            f"metrics={self.metrics_csv} diagnostics={self.diagnostics_dir}"
        )

    def _run_epoch(self, epoch: int, *, train: bool) -> dict[str, float]:
        phase = "train" if train else "val(patch)"
        self.current_phase = "train" if train else "validation"
        source = self.train_batches if train else self.val_batches
        n_iter = self.iterations_per_epoch if train else self.validation_iterations
        assert source is not None
        self.model.train(train)
        iterator = iter(source())
        accumulator = MetricAccumulator()
        desc = f"{phase} e{epoch + 1}/{self.epochs}"
        bar = make_progress(range(n_iter), total=n_iter, desc=desc, disable=not self.progress)
        for i in bar:
            self.current_iteration = i
            try:
                batch = next(iterator)
            except StopIteration as exc:
                raise RuntimeError(
                    f"{phase} 数据源提前耗尽：第 {i + 1}/{n_iter} 次迭代无数据；请检查 batch source 的批次数量"
                ) from exc
            data, seg, prior = self._to_device(batch)
            if train:
                self.optimizer.zero_grad(set_to_none=True)
                with self._autocast():
                    outputs = self._forward_model(data, prior)
                    loss = self.loss_fn(outputs, seg)
                self._backward(loss)
            else:
                with torch.no_grad(), self._autocast():
                    outputs = self._forward_model(data, prior)
                    loss = self.loss_fn(outputs, seg)
            main = self._main_logits(outputs)
            accumulator.update(loss=float(loss.detach()), counts=counts_from_logits(main.detach(), seg))
            if hasattr(bar, "set_postfix"):
                bar.set_postfix({"loss": f"{accumulator.mean_loss:.4f}", "dice": f"{accumulator.counts.dice:.3f}"})
        return accumulator.summary()

    # ------------------------------------------------------------------ 续训
    #: 续训时必须与 checkpoint 一致的训练协议字段（选模 + 验证调度 + early stopping）
    PROTOCOL_KEYS = (
        "checkpoint_metric",
        "maximize_metric",
        "early_stopping_enabled",
        "min_epochs",
        "patience",
        "min_delta",
        "validation_every_n_epochs",
        "validation_protocol",
    )

    #: 续训时冻结的 resolved 配置前缀（含 seed/patch/模态顺序/增强/loss/optimizer/iterations/plan 与数据哈希）
    FROZEN_CONFIG_PREFIXES = (
        "experiment.",
        "data.",
        "model.",
        "loss.",
        "optimizer.",
        "scheduler.",
        "training.",
        "inference.",
        "validation.",
        "early_stopping.",
        "plan.",
        "train_pipeline.",
        "val_pipeline.",
        "provenance.plans_sha256",
        "provenance.splits_sha256",
        # v2.3：架构身份（name/version/sha256/encoder_type）与架构哈希必须在续训时冻结；
        # legacy PlainConv checkpoint 无这些键，与 residual run 的当前配置比对时会因缺失而被拒绝。
        "architecture.",
        "provenance.architecture_sha256",
        # PZ/TZ prior：数据集级 manifest 哈希必须一致，否则续训拒绝（M3/M4）
        "provenance.zonal_prior_manifest_sha256",
    )
    #: 冻结前缀内**允许显式变化**的项（epoch 预算是唯一允许延长的训练项）
    EXTENDABLE_CONFIG_PATHS = ("training.max_epochs",)
    #: 只影响吞吐/遥测的字段（允许变化，仅打印记录）
    EXTENDABLE_CONFIG_SUFFIXES = (".num_workers", ".pin_memory", ".checkpoint_every", ".progress")

    @classmethod
    def _is_frozen_config_key(cls, key: str) -> bool:
        """判断扁平化后的配置键在续训时是否必须冻结。"""
        if key in cls.EXTENDABLE_CONFIG_PATHS or key.endswith(cls.EXTENDABLE_CONFIG_SUFFIXES):
            return False
        return key.startswith(cls.FROZEN_CONFIG_PREFIXES)

    def _verify_resume_consistency(
        self,
        saved_config: Mapping[str, Any] | None,
        saved_extra: Mapping[str, Any],
        *,
        allow_change: bool,
    ) -> bool:
        """核对续训一致性：训练协议（选模/early stopping）+ 完整 resolved 训练配置。

        返回 `True` 表示存在差异且已显式放行，调用方需据此重置选模/early-stopping 状态。
        """
        current = self._checkpoint_extra()
        missing = [key for key in self.PROTOCOL_KEYS if key not in saved_extra]
        protocol_diff = {
            key: {"checkpoint": saved_extra[key], "current": current.get(key)}
            for key in self.PROTOCOL_KEYS
            if key in saved_extra and saved_extra[key] != current.get(key)
        }
        flat_saved = _flatten_mapping(saved_config or {})
        flat_current = _flatten_mapping(self.config_snapshot or {})
        changed = {
            key: {"checkpoint": flat_saved.get(key), "current": flat_current.get(key)}
            for key in sorted(set(flat_saved) | set(flat_current))
            if flat_saved.get(key) != flat_current.get(key)
        }
        config_diff = {key: value for key, value in changed.items() if self._is_frozen_config_key(key)}
        info_diff = {key: value for key, value in changed.items() if not self._is_frozen_config_key(key)}
        if info_diff:
            print(
                "[trainer] 续训时非冻结配置项存在差异（不阻塞，仅记录）: "
                + "；".join(
                    f"{key}: {value['checkpoint']} → {value['current']}" for key, value in info_diff.items()
                )
            )
        if not missing and not protocol_diff and not config_diff:
            return False
        detail = []
        if missing:
            detail.append(f"checkpoint 缺少协议字段 {missing}")
        if protocol_diff:
            detail.append(f"协议不一致 {protocol_diff}")
        if config_diff:
            detail.append(f"训练配置不一致 {config_diff}")
        message = "拒绝续训：" + "；".join(detail)
        if not allow_change:
            raise RuntimeError(
                message + "（确认要有意改变协议/配置时，请显式传入 allow_protocol_change=True）"
            )
        print(f"[trainer][WARN] {message}（已显式放行；正式实验不建议改变配置续训）")
        return True

    def last_logged_epoch(self) -> int | None:
        """读取 `metrics/epoch_metrics.csv` 中最后一个 epoch 索引（无文件/无数据行返回 None）。"""
        if not self.metrics_csv.is_file():
            return None
        last: int | None = None
        with open(self.metrics_csv, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                try:
                    last = int(float(row["epoch"]))  # type: ignore[index]
                except (KeyError, TypeError, ValueError):
                    continue
        return last

    def resume(
        self,
        checkpoint_path: str | Path,
        *,
        allow_mid_epoch: bool = False,
        allow_protocol_change: bool = False,
    ) -> dict:
        """恢复 model/optimizer/scheduler/scaler/RNG、epoch、best 指标、validation-event 计数与 early-stopping 状态。

        - **协议校验**：`checkpoint_metric / maximize / early_stopping / min_delta / validation_every_n_epochs /
          validation_protocol` 必须一致，否则默认拒绝续训；显式 `allow_protocol_change=True` 放行时会
          **重置选模与 event 计数状态**（旧 best/bad_validation_events/validation_events_completed 在新协议下
          不一定可比，必须重新开始记录）；
        - **配置冻结**：resolved 配置快照中 `experiment/data/model/loss/optimizer/scheduler/training/inference/
          validation/early_stopping/plan/*_pipeline` 前缀（含 seed、patch、模态顺序、增强、iterations、
          plan/split 哈希）必须一致，否则同样默认拒绝；`training.max_epochs` 是唯一允许显式延长的训练项，
          `num_workers / pin_memory / checkpoint_every / progress` 等仅吞吐/遥测字段允许变化（打印记录）；
        - **日志对齐**：checkpoint 的续训起点必须等于 `metrics/epoch_metrics.csv` 最后一个 epoch + 1，
          否则拒绝（防止从旧 checkpoint 恢复造成 epoch 行重复/错序与病例 CSV 覆盖）；
        - **epoch 语义**：`epoch_completed=True` 从 `epoch + 1` 继续；`epoch_completed=False`（训练中途被打断）
          默认**拒绝**，需显式 `allow_mid_epoch=True` —— 此时会**重跑被打断的那个 epoch**
          （模型权重是中断时刻的，该 epoch 的前若干 iteration 会被重复执行，checkpoint 里已记录具体数值）；
        - **RNG**：恢复后标记 `_rng_restored`，`train()` 不再 `set_seed` 覆盖。
        """
        meta = load_checkpoint(
            checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler if self.amp else None,
            map_location=self.device,
        )
        extra = dict(meta.get("extra") or {})
        protocol_changed = self._verify_resume_consistency(
            meta.get("config"), extra, allow_change=allow_protocol_change
        )
        restore_rng_states(meta.get("rng_states"))
        self._rng_restored = True
        if self.scheduler is not None and hasattr(self.scheduler, "max_steps"):
            # 续训以本次 run 的 epochs 为准（PolyLR 的 max_steps 决定 lr 曲线终点）
            self.scheduler.max_steps = int(self.epochs)

        epoch_completed = bool(meta.get("epoch_completed", True))
        completed_epochs = meta.get("completed_epochs")
        if epoch_completed:
            self.start_epoch = int(meta["epoch"]) + 1
            note = "完整 epoch"
            if completed_epochs is not None and int(completed_epochs) != int(meta["epoch"]) + 1:
                raise RuntimeError(
                    f"checkpoint 元数据不一致：epoch={meta['epoch']}（已完整结束）但 completed_epochs={completed_epochs}；"
                    "拒绝续训以免 epoch 计数漂移"
                )
        else:
            if not allow_mid_epoch:
                raise RuntimeError(
                    f"拒绝续训：{meta['path']} 是训练**中途**保存的（epoch_completed=False，"
                    f"epoch={meta['epoch']}，completed_epochs={completed_epochs}，"
                    f"iterations_completed={meta.get('iterations_completed')}）。"
                    "若要与最后一个完整 epoch 对齐，请改用 checkpoint_last.pth；"
                    "若确认要从中断状态继续（会重跑被打断的 epoch），请显式传入 allow_mid_epoch=True。"
                )
            self.start_epoch = int(completed_epochs) if completed_epochs is not None else int(meta["epoch"])
            note = f"中途中断（completed_epochs={completed_epochs}，将重跑 epoch {self.start_epoch}）"
        # 本次进程开始前已完整完成的 epoch 数（中断时用于计算绝对的 completed_epochs）
        self._epoch_offset = int(self.start_epoch)

        # 日志对齐：从较旧 checkpoint 恢复到同一 run 会造成 epoch 行重复/错序与同名病例 CSV 覆盖
        last_logged = self.last_logged_epoch()
        if last_logged is not None and self.start_epoch != last_logged + 1:
            raise RuntimeError(
                f"拒绝续训：该 checkpoint 将从 epoch {self.start_epoch} 开始，但 {self.metrics_csv} 已记录到 "
                f"epoch {last_logged}（相差 {self.start_epoch - last_logged - 1:+d} 个 epoch）。"
                "从较旧 checkpoint 恢复到同一 run 会产生重复/错序的 epoch 行并覆盖同名病例 CSV；"
                "请改用 checkpoint_last.pth，或换新的 --run-name 另开 run（--run-name 不同则日志目录不同）。"
            )

        if protocol_changed:
            # 指标口径/验证调度可能不可比 → 不继承旧选模与 event 计数，从续训点重新开始
            self.best_metric = None
            self.best_epoch = None
            self.bad_validation_events = 0
            self.validation_events_completed = 0
            self.last_validation_epoch = None
            print(
                "[trainer][WARN] 训练协议已变化：已重置 best_metric / best_epoch / bad_validation_events / "
                "validation_events_completed / last_validation_epoch，选模从续训点重新开始（正式实验不建议改变协议续训）"
            )
        else:
            self.best_metric = meta.get("best_metric")
            self.best_epoch = extra.get("best_epoch")
            # 兼容旧 checkpoint：优先读 bad_validation_events，回退到旧键 early_stopping_bad_epochs
            self.bad_validation_events = int(
                extra.get("bad_validation_events", extra.get("early_stopping_bad_epochs", 0))
            )
            self.validation_events_completed = int(extra.get("validation_events_completed", 0))
            last_val_epoch = extra.get("last_validation_epoch")
            self.last_validation_epoch = None if last_val_epoch is None else int(last_val_epoch)
        print(
            f"[trainer] resume: {meta['path']} → start_epoch={self.start_epoch}（{note}）"
            f"best_metric={self.best_metric} best_epoch={self.best_epoch} "
            f"bad_validation_events={self.bad_validation_events} "
            f"validation_events_completed={self.validation_events_completed} "
            f"lr={self.optimizer.param_groups[0]['lr']:.3e}"
        )
        return meta

    # --------------------------------------------------------------- scaler
    @property
    def scaler(self):
        if not hasattr(self, "_scaler"):
            self._scaler = torch.amp.GradScaler("cuda") if self.amp else None
        return self._scaler


__all__ = [
    "BatchSource",
    "EarlyStoppingConfig",
    "M0Trainer",
    "TrainerSummary",
    "VALIDATION_MODE_ALIASES",
    "VALIDATION_MODE_DIAGNOSTIC",
    "VALIDATION_MODE_FORMAL",
    "VALIDATION_MODE_FULL_VOLUME",
    "VALIDATION_MODES",
]
