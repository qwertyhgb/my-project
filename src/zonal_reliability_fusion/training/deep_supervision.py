"""Deep supervision 权重与多尺度损失包装。

与 nnUNet v2.6.2 行为对齐（`nnUNetTrainer` 的 deep supervision 权重构造）：
- 输出按**最高分辨率 → 最低分辨率**排列；
- 权重 w_i = 1 / 2^i；
- 最低分辨率输出权重置 0（不参与 loss）；
- 归一化使 sum(w) = 1；
- **只有一个输出时不能把唯一权重置 0**（此时返回 (1.0,)）。

目标下采样使用 nearest（不插值出新标签值）。
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .losses import prepare_target


def deep_supervision_weights(num_outputs: int) -> tuple[float, ...]:
    """返回 DS 各输出权重（高分辨率在前），sum = 1。"""
    if num_outputs < 1:
        raise ValueError(f"num_outputs 必须 >= 1，收到 {num_outputs}")
    if num_outputs == 1:
        return (1.0,)
    weights = [1.0 / (2**i) for i in range(num_outputs)]
    weights[-1] = 0.0  # 最低分辨率不参与 loss
    total = sum(weights)
    if total <= 0:  # pragma: no cover - 防御性
        raise ValueError("deep supervision 权重全为 0，无法归一化")
    return tuple(w / total for w in weights)


def downsample_target_nearest(target: torch.Tensor, size: Sequence[int]) -> torch.Tensor:
    """把 [B, D, H, W] long target 用 nearest 下采样到 size（保持 long）。"""
    if target.ndim != 4:
        raise ValueError(f"target 必须为 [B, D, H, W]，收到 {tuple(target.shape)}")
    if tuple(target.shape[-3:]) == tuple(int(s) for s in size):
        return target
    small = F.interpolate(target.float().unsqueeze(1), size=tuple(int(s) for s in size), mode="nearest")
    return small[:, 0].long()


def build_ds_targets(target: torch.Tensor, outputs: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    """按各输出尺度构建 target 列表（第 0 项为原始分辨率，不做下采样）。"""
    base = prepare_target(target, num_classes=2)
    targets = [base]
    for out in list(outputs)[1:]:
        targets.append(downsample_target_nearest(base, out.shape[-3:]))
    return targets


class DeepSupervisionLoss(nn.Module):
    """把单尺度损失应用到多尺度输出（按权重加权求和）。

    Args:
        loss_fn: 单尺度损失，签名 `loss_fn(logits, target)`。
        weights: 与输出数量一致的权重；None 时使用 `deep_supervision_weights`。
    """

    def __init__(self, loss_fn: nn.Module, weights: Sequence[float] | None = None) -> None:
        super().__init__()
        if loss_fn is None:
            raise ValueError("loss_fn 不能为 None")
        self.loss_fn = loss_fn
        self._weights = tuple(float(w) for w in weights) if weights is not None else None
        if self._weights is not None and not any(w != 0 for w in self._weights):
            raise ValueError("至少一个 DS 权重必须非 0")

    def resolved_weights(self, num_outputs: int) -> tuple[float, ...]:
        if self._weights is None:
            return deep_supervision_weights(num_outputs)
        if len(self._weights) != num_outputs:
            raise ValueError(f"权重数量 {len(self._weights)} 与输出数量 {num_outputs} 不一致")
        return self._weights

    def forward(self, outputs, target: torch.Tensor) -> torch.Tensor:
        """outputs: list[Tensor]（高→低）或单个 Tensor；target: [B,1,D,H,W] 或 [B,D,H,W]。"""
        if isinstance(outputs, torch.Tensor):
            return self.loss_fn(outputs, prepare_target(target, outputs.shape[1]))
        if not outputs:
            raise ValueError("空的 deep-supervision 输出列表")
        weights = self.resolved_weights(len(outputs))
        targets = build_ds_targets(target, outputs)
        total = None
        for weight, logits, tgt in zip(weights, outputs, targets):
            if weight == 0.0:
                continue
            term = self.loss_fn(logits, tgt)
            total = term * weight if total is None else total + term * weight
        if total is None:  # pragma: no cover - 由构造校验排除
            raise RuntimeError("所有 DS 权重均为 0")
        return total


__all__ = [
    "DeepSupervisionLoss",
    "build_ds_targets",
    "deep_supervision_weights",
    "downsample_target_nearest",
]
