"""优化器与学习率调度（与冻结 nnUNet v2.6.2 默认训练 recipe 对齐）。

对齐项：
- SGD，lr = 1e-2，momentum = 0.99，nesterov = True，weight_decay = 3e-5；
- PolyLR：lr = initial_lr * (1 - step / max_steps) ** power（power=0.9），按 **epoch** step；
- 每种取值都来自配置，不在 trainer 中散落硬编码；
- 断点续训时学习率必须连续（本调度器显式保存/恢复 step）。
"""
from __future__ import annotations

from typing import Iterable

import torch
from torch import nn

from ..config.experiment import OptimizerConfig, SchedulerConfig


class PolyLRScheduler:
    """按 epoch 递减的多项式学习率调度（行为与 nnUNet v2.6.2 PolyLRScheduler 一致）。"""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        initial_lr: float,
        max_steps: int,
        *,
        power: float = 0.9,
        current_step: int = 0,
    ) -> None:
        if max_steps < 1:
            raise ValueError(f"max_steps 必须 >= 1，收到 {max_steps}")
        self.optimizer = optimizer
        self.initial_lr = float(initial_lr)
        self.max_steps = int(max_steps)
        self.power = float(power)
        self.current_step = int(current_step)
        self._last_lr = [self.initial_lr]

    def step(self, current_step: int | None = None) -> float:
        """更新 lr；默认使用内部计数（每次 +1）。返回本次设置的 lr。"""
        if current_step is not None:
            self.current_step = int(current_step)
        new_lr = self.initial_lr * max(0.0, 1.0 - self.current_step / self.max_steps) ** self.power
        for group in self.optimizer.param_groups:
            group["lr"] = new_lr
        self._last_lr = [new_lr]
        self.current_step += 1
        return new_lr

    def get_last_lr(self) -> list[float]:
        return list(self._last_lr)

    def state_dict(self) -> dict:
        return {
            "initial_lr": self.initial_lr,
            "max_steps": self.max_steps,
            "power": self.power,
            "current_step": self.current_step,
            "_last_lr": list(self._last_lr),
        }

    def load_state_dict(self, state: dict) -> None:
        """恢复调度器状态。

        注意：这里**不直接改写 optimizer 的 lr**。学习率由 `step(epoch)` 在每个 epoch 开始时
        依据 epoch 索引重新计算（trainer 总以真实 epoch 索引调用 step），因此只要 epoch 连续，
        学习率必然连续；这样避免了与 optimizer state 的恢复顺序相互覆盖。
        """
        self.initial_lr = float(state["initial_lr"])
        self.max_steps = int(state["max_steps"])
        self.power = float(state["power"])
        self.current_step = int(state["current_step"])
        self._last_lr = list(state.get("_last_lr", [self.initial_lr]))


def build_optimizer(model: nn.Module, cfg: OptimizerConfig) -> torch.optim.Optimizer:
    """按配置构建优化器（当前仅支持 SGD，与 N0 对齐）。"""
    name = cfg.name.upper()
    if name != "SGD":
        raise ValueError(f"暂只支持 SGD（与 N0 对齐），收到 optimizer.name={cfg.name!r}")
    return torch.optim.SGD(
        model.parameters(),
        lr=float(cfg.lr),
        momentum=float(cfg.momentum),
        nesterov=bool(cfg.nesterov),
        weight_decay=float(cfg.weight_decay),
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer, cfg: SchedulerConfig, *, epochs: int, initial_lr: float
) -> PolyLRScheduler:
    """按配置构建 PolyLR（power 来自配置）。"""
    name = cfg.name.upper()
    if name not in ("POLYLR", "POLYLRSCHEDULER", "POLY"):
        raise ValueError(f"暂只支持 PolyLR（与 N0 对齐），收到 scheduler.name={cfg.name!r}")
    return PolyLRScheduler(optimizer, initial_lr=initial_lr, max_steps=int(epochs), power=float(cfg.power))


def build_param_groups(*modules: nn.Module) -> Iterable[nn.Parameter]:
    """把多个模块的参数合并（M1–M4 多分支时复用）。"""
    for module in modules:
        for param in module.parameters():
            yield param


__all__ = ["PolyLRScheduler", "build_optimizer", "build_param_groups", "build_scheduler"]
