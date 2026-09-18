"""损失函数：与正式 N0 对齐的 Focal + CrossEntropy（当前默认）与历史 Dice + CrossEntropy。

**默认口径（与正式 N0 一致，research_plan §9.2）**：`PiCAIFocalCELoss`
= `0.5 × Focal(γ=2, α=None) + 0.5 × CE`（PI-CAI official-style nnU-Net v2 Focal+CE NoFFT）。
`DiceCELoss` 是迁移前的自研实现，保留可用，但**不再与正式 N0 可比**
（见 `docs/P2_M0_Implementation.md` §18）。

Dice+CE 的历史对齐项（与冻结 plan / nnUNet v2.6.2 默认行为一致）：
- CrossEntropy 直接接收 **logits**（模型内部不做 softmax）；
- Dice 在 softmax 概率上计算，默认**排除 background**；
- smooth = 1e-5；
- batch_dice = False（plan 值）→ 逐样本计算后取平均；
- 全阴性 target 不产生 NaN；
- 支持 deep supervision（见 `deep_supervision.DeepSupervisionLoss`）。
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn


def prepare_target(target: torch.Tensor, num_classes: int) -> torch.Tensor:
    """把 target 规范为 [B, D, H, W] 的 long 张量（接受 [B,1,D,H,W]）。"""
    if target.ndim == 5 and target.shape[1] == 1:
        target = target[:, 0]
    if target.ndim != 4:
        raise ValueError(f"target 必须为 [B,1,D,H,W] 或 [B,D,H,W]，收到 {tuple(target.shape)}")
    return target.long()


def _one_hot(target: torch.Tensor, num_classes: int) -> torch.Tensor:
    """[B, D, H, W] -> [B, C, D, H, W]（float32）。"""
    return F.one_hot(target, num_classes=num_classes).movedim(-1, 1).float()


class SoftDiceLoss(nn.Module):
    """soft Dice（在 softmax 概率上计算），默认排除 background。

    **返回 `-Dice`（与 nnU-Net v2.6.2 的 `SoftDiceLoss.forward` 完全一致）**，
    因此 Dice 项在 [−1, 0] 之间，`Dice + CE` 的总 loss 可能为负。
    如需人类可读的 Dice，请使用 `dice_score()`（返回 [0, 1] 的正 Dice）。
    """

    def __init__(
        self,
        *,
        smooth: float = 1e-5,
        include_background: bool = False,
        batch_dice: bool = False,
    ) -> None:
        super().__init__()
        self.smooth = float(smooth)
        self.include_background = bool(include_background)
        self.batch_dice = bool(batch_dice)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return -self.dice_score(logits, target)

    def dice_score(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """前景平均 Dice（正数，[0,1]；与官方 `get_tp_fp_fn_tn` 的 DC 口径一致）。"""
        num_classes = logits.shape[1]
        target = prepare_target(target, num_classes)
        probs = torch.softmax(logits, dim=1)
        target_1h = _one_hot(target, num_classes)

        spatial_dims = tuple(range(2, logits.ndim))
        classes = list(range(num_classes)) if self.include_background else list(range(1, num_classes))
        if not classes:  # num_classes == 1（非本项目用法）
            return torch.zeros((), dtype=logits.dtype, device=logits.device)

        if self.batch_dice:
            intersection = (probs * target_1h).sum(dim=(0, *spatial_dims))[classes]
            denominator = (probs + target_1h).sum(dim=(0, *spatial_dims))[classes]
        else:
            intersection = (probs * target_1h).sum(dim=spatial_dims)[:, classes]  # [B, C_fg]
            denominator = (probs + target_1h).sum(dim=spatial_dims)[:, classes]
        dice = (2 * intersection + self.smooth) / torch.clip(denominator + self.smooth, min=1e-8)
        return dice.mean()


class DiceCELoss(nn.Module):
    """L = dice_weight * Dice + ce_weight * CrossEntropy。"""

    def __init__(
        self,
        *,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        smooth: float = 1e-5,
        include_background: bool = False,
        batch_dice: bool = False,
    ) -> None:
        super().__init__()
        self.dice_weight = float(dice_weight)
        self.ce_weight = float(ce_weight)
        self.dice = SoftDiceLoss(
            smooth=smooth, include_background=include_background, batch_dice=batch_dice
        )
        self.smooth = float(smooth)
        self.include_background = bool(include_background)
        self.batch_dice = bool(batch_dice)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_long = prepare_target(target, logits.shape[1])
        ce = F.cross_entropy(logits, target_long)
        dice_loss = self.dice(logits, target)
        return self.dice_weight * dice_loss + self.ce_weight * ce

    @torch.no_grad()
    def components(self, logits: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
        """拆分记录 dice/ce（用于 CSV 日志）。

        注意：`dice_loss` 为 **−Dice**（官方口径，≤0）；`dice_score` 为正 Dice（[0,1]）。
        """
        target_long = prepare_target(target, logits.shape[1])
        ce = F.cross_entropy(logits, target_long)
        dice_score = self.dice.dice_score(logits, target)
        return {
            "dice_loss": float(-dice_score.detach()),
            "ce_loss": float(ce.detach()),
            "dice_score": float(dice_score.detach()),
            "total": float((self.ce_weight * ce - self.dice_weight * dice_score).detach()),
        }


def _validate_target_strict(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Focal+CE 专用的严格 target 校验（与 N0 适配层同一语义）。

    与 `prepare_target` 的区别：显式拒绝非整数标签、负标签与越界标签。Dataset605 没有 ignore label，
    因此任何越界值都必须报错，而不是被静默当作背景。
    """
    if logits.ndim < 2:
        raise ValueError(f"logits 至少应为 [B,C]，收到 {tuple(logits.shape)}")
    if logits.shape[1] < 2:
        raise ValueError("PI-CAI Focal + CE 至少需要 background 与 lesion 两类")
    if target.ndim == logits.ndim:
        if target.shape[1] != 1:
            raise ValueError(
                "target 的类别维必须为 singleton；"
                f"收到 logits={tuple(logits.shape)}, target={tuple(target.shape)}"
            )
        target = target[:, 0]
    if tuple(target.shape) != tuple(logits.shape[:1] + logits.shape[2:]):
        raise ValueError(
            f"target 与 logits 的空间尺寸不匹配：logits={tuple(logits.shape)}, target={tuple(target.shape)}"
        )
    if torch.is_floating_point(target) and not torch.all(target == target.round()):
        raise ValueError("target 含非整数标签")
    target = target.long()
    if bool(torch.any(target < 0)) or bool(torch.any(target >= logits.shape[1])):
        raise ValueError("target 含超出 [0, num_classes) 的标签；Dataset605 不支持 ignore label")
    return target


class PiCAIFocalCELoss(nn.Module):
    """PI-CAI official-style Focal + CE（与正式 N0 对齐，research_plan §9.2）。

    ``L = focal_weight · Focal(gamma, alpha=None) + ce_weight · CE``

    对齐要点（与 `integrations.nnunet_picai_flce.PiCAIFocalCrossEntropyLoss` 逐值一致）：

    - Softmax → one-hot clamp(smooth, 1−smooth) → ``pt + smooth`` → ``−(1−pt)^γ · log(pt)``，再取 mean；
      ``alpha=None`` 即无类别权重（**不是** ``focalLossAlpha75`` 变体）；
    - CE 直接接收 **logits**（模型内部不做 softmax），与 Dice+CE 实现的口径一致；
    - 负标签 / 越界标签 / 非整数标签显式报错（Dataset605 无 ignore label）；
    - 全背景 target 不产生 NaN；
    - 本实现**不依赖 nnunetv2**（`training/` 的边界要求）；与 N0 的数值一致性由合成单元测试锁定。
    """

    def __init__(
        self,
        *,
        focal_weight: float = 0.5,
        ce_weight: float = 0.5,
        gamma: float = 2.0,
        smooth: float = 1e-5,
    ) -> None:
        super().__init__()
        if focal_weight < 0 or ce_weight < 0 or focal_weight + ce_weight <= 0:
            raise ValueError("focal_weight/ce_weight 必须非负且至少一个为正")
        if gamma < 0:
            raise ValueError("gamma 必须非负")
        if not 0.0 <= smooth < 1.0:
            raise ValueError("smooth 必须在 [0, 1) 内")
        self.focal_weight = float(focal_weight)
        self.ce_weight = float(ce_weight)
        self.gamma = float(gamma)
        self.smooth = float(smooth)

    def _focal_and_ce(
        self, logits: torch.Tensor, target_long: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        probabilities = torch.softmax(logits, dim=1)
        one_hot = _one_hot(target_long, logits.shape[1])
        if self.smooth:
            # 与 PI-CAI v1 公开实现一致：二分类使用 min=smooth/(C-1)，保留其多类语义。
            one_hot = one_hot.clamp(
                min=self.smooth / (logits.shape[1] - 1), max=1.0 - self.smooth
            )
        pt = (one_hot * probabilities).sum(dim=1) + self.smooth
        focal = (-torch.pow(1.0 - pt, self.gamma) * torch.log(pt)).mean()
        ce = F.cross_entropy(logits, target_long)
        return focal, ce, pt

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_long = _validate_target_strict(logits, target)
        focal, ce, _pt = self._focal_and_ce(logits, target_long)
        return self.focal_weight * focal + self.ce_weight * ce

    @torch.no_grad()
    def components(self, logits: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
        """拆分记录 focal/ce/pt（用于 CSV 日志）。"""
        target_long = _validate_target_strict(logits, target)
        focal, ce, pt = self._focal_and_ce(logits, target_long)
        return {
            "focal_loss": float(focal.detach()),
            "ce_loss": float(ce.detach()),
            "mean_pt": float(pt.detach().mean()),
            "total": float((self.focal_weight * focal + self.ce_weight * ce).detach()),
        }


__all__ = ["DiceCELoss", "PiCAIFocalCELoss", "SoftDiceLoss", "prepare_target"]
