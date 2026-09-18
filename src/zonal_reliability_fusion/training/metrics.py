"""训练期指标：前景 Dice 与 TP / FP / FN。

说明：
- 指标在 argmax 后的硬预测上统计，只针对前景类（label 1），不做后处理；
- 全阴性且预测全阴时 Dice 定义为 0.0，并标记 `empty=True`（避免 NaN 混入日志）；
- 正式 PI-CAI lesion AP 由后续评估阶段（picai_eval）承担，本模块不实现、不伪造。
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .losses import prepare_target


@dataclass(frozen=True)
class SegmentCounts:
    """TP / FP / FN 计数（前景类）。"""

    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def denominator(self) -> int:
        return 2 * self.tp + self.fp + self.fn

    @property
    def empty(self) -> bool:
        """参考与预测都没有前景。"""
        return self.tp == 0 and self.fp == 0 and self.fn == 0

    @property
    def dice(self) -> float:
        if self.denominator == 0:
            return 0.0
        return 2.0 * self.tp / self.denominator

    def as_dict(self) -> dict[str, float]:
        return {"tp": self.tp, "fp": self.fp, "fn": self.fn, "dice": self.dice}

    def __add__(self, other: "SegmentCounts") -> "SegmentCounts":
        return SegmentCounts(tp=self.tp + other.tp, fp=self.fp + other.fp, fn=self.fn + other.fn)


def counts_from_logits(logits: torch.Tensor, target: torch.Tensor) -> SegmentCounts:
    """由 logits 与 target 计算前景 TP/FP/FN（CPU/GPU 均可，返回 python int）。"""
    num_classes = logits.shape[1]
    tgt = prepare_target(target, num_classes)
    pred = logits.argmax(dim=1).long()
    fg_pred = pred == 1
    fg_true = tgt == 1
    tp = int((fg_pred & fg_true).sum().item())
    fp = int((fg_pred & ~fg_true).sum().item())
    fn = int((~fg_pred & fg_true).sum().item())
    return SegmentCounts(tp=tp, fp=fp, fn=fn)


class MetricAccumulator:
    """跨 batch 累加指标（训练/验证 epoch 统计）。"""

    def __init__(self) -> None:
        self.counts = SegmentCounts()
        self.loss_sum: float = 0.0
        self.samples: int = 0

    def update(self, *, loss: float, counts: SegmentCounts) -> None:
        self.counts = self.counts + counts
        self.loss_sum += float(loss)
        self.samples += 1

    @property
    def mean_loss(self) -> float:
        return self.loss_sum / self.samples if self.samples else 0.0

    def summary(self) -> dict[str, float]:
        return {"loss": self.mean_loss, **self.counts.as_dict()}


__all__ = ["MetricAccumulator", "SegmentCounts", "counts_from_logits"]
