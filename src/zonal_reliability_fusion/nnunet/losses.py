"""项目自有的损失组件（全部为对 nnU-Net 原生机制的最小扩展）。

这里只有**两**样东西，都是研究假设直接要求的：

1. :class:`PiCAIFocalCrossEntropyLoss`——PI-CAI 官方风格的 ``0.5*Focal(gamma=2) + 0.5*CE``
   （strong baseline 候选 A1）。deep supervision 加权沿用 nnU-Net 官方
   ``DeepSupervisionWrapper``，不在此重实现。
2. :class:`LesionnessAuxiliaryLoss`——在原生主损失之上加一项 coarse lesionness 监督（条件 C）。
   它**不替换**主损失，只做 ``main + weight * aux``。

**没有**第三样。Dice+CE、deep supervision 权重、optimizer、LR scheduler、checkpoint、
validation、滑窗推理全部是 nnU-Net 原生实现。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from torch import Tensor, nn

from zonal_reliability_fusion.lesion.lesionness import (
    LESIONNESS_DILATION_RADIUS_MM,
    lesionness_target,
)

#: coarse lesionness 辅助损失的权重。**预先冻结**：未看结果前固定，不得事后调参。
LESIONNESS_LOSS_WEIGHT = 0.5


class PiCAIFocalCrossEntropyLoss(nn.Module):
    """PI-CAI baseline 的等权 Focal + CE 二/多类体素损失。

    原始 trainer 将 ``FocalLoss(alpha=None, gamma=2, smooth=1e-5)`` 与 Cross-Entropy 以
    0.5/0.5 相加。``alpha=None`` 意味着没有额外的类别权重。输入为 nnU-Net 约定的 logits
    ``[B, C, ...]``，target 为 ``[B, 1, ...]`` 或 ``[B, ...]``。Dataset605/606 没有 ignore
    label，故遇到负标签或越界标签会显式失败而不是静默当作背景。
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

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
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
        if target.shape != logits.shape[:1] + logits.shape[2:]:
            raise ValueError(
                "target 与 logits 的空间尺寸不匹配："
                f"logits={tuple(logits.shape)}, target={tuple(target.shape)}"
            )
        if torch.is_floating_point(target) and not torch.all(target == target.round()):
            raise ValueError("target 含非整数标签")
        target = target.long()
        if bool(torch.any(target < 0)) or bool(torch.any(target >= logits.shape[1])):
            raise ValueError(
                "target 含超出 [0, num_classes) 的标签；Dataset605/606 不支持 ignore label"
            )

        # 复刻 PI-CAI v1 FocalLoss：Softmax -> one-hot clamp(smooth, 1-smooth)
        # -> pt + smooth -> -(1-pt)^gamma * log(pt)。alpha=None 即全 1。
        probabilities = torch.softmax(logits, dim=1)
        one_hot = F.one_hot(target, num_classes=logits.shape[1]).movedim(-1, 1)
        one_hot = one_hot.to(dtype=probabilities.dtype)
        if self.smooth:
            one_hot = one_hot.clamp(
                min=self.smooth / (logits.shape[1] - 1), max=1.0 - self.smooth
            )
        pt = (one_hot * probabilities).sum(dim=1) + self.smooth
        focal = -torch.pow(1.0 - pt, self.gamma) * torch.log(pt)
        focal = focal.mean()
        ce = F.cross_entropy(logits, target)
        return self.focal_weight * focal + self.ce_weight * ce


class PICAIFocalCrossEntropyLossMixin:
    """用 PI-CAI Focal+CE 替换 nnU-Net 默认损失；deep supervision 加权沿用官方规则。"""

    def _build_loss(self) -> nn.Module:
        if self.label_manager.has_regions:
            raise RuntimeError(
                "PI-CAI Focal+CE 仅支持互斥二/多类标签；Dataset605/606 应为 background=0, lesion=1。"
            )
        if self.label_manager.ignore_label is not None:
            raise RuntimeError(
                "PI-CAI Focal+CE 不支持 ignore label；dataset.json 不应声明 ignore。"
            )

        loss: nn.Module = PiCAIFocalCrossEntropyLoss(
            focal_weight=0.5, ce_weight=0.5, gamma=2.0, smooth=1e-5
        )
        if self.enable_deep_supervision:
            scales = self._get_deep_supervision_scales()
            weights = np.array(
                [1.0 / (2**i) for i in range(len(scales))], dtype=np.float64
            )
            # 与固定 nnU-Net v2.6.2 默认 trainer 一致：最低分辨率不监督；DDP 特殊分支沿用官方实现。
            weights[-1] = 1e-6 if self.is_ddp and not self._do_i_compile() else 0.0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
        return loss


class LesionnessAuxiliaryLoss(nn.Module):
    """``main_loss(seg_outputs, target) + weight * lesionness_loss(aux_logits, coarse_target)``。

    它回答的是条件 C 的核心问题「显式 lesion localization 是否减少完全漏检」。**它不替换**
    主损失，因此条件 C 相对条件 B 的差异可以清楚地归因到「多了一个 coarse 定位监督 + soft
    refinement」，而不是「换了损失」。

    监督来源与硬约束
    ----------------
    - coarse 目标**只**由 lesion GT 推导（物理半径膨胀 -> 最大池化到 coarse 网格）；
      validation / test 完全不使用 GT 推导的目标；
    - 主损失与辅助损失在**同一次** backward 中联合优化（不是两阶段）；
    - ``radius_mm`` 默认取 :data:`..lesion.lesionness.LESIONNESS_DILATION_RADIUS_MM`
      （预先冻结），可由消融显式覆盖。

    几何一致性
    ----------
    先在**全分辨率**上用真实 spacing 按物理半径膨胀，再最大池化到 coarse 网格。因此 coarse
    目标的语义恰好是「该 coarse 体素附近 r mm 内是否有真实病灶」，而不需要在 coarse 网格上
    重新解释 spacing。基线 spacing 由调用方通过 ``spacing_zyx`` 显式传入，
    **不**从数组 shape 反推。
    """

    def __init__(
        self,
        main_loss: nn.Module,
        *,
        network: nn.Module,
        spacing_zyx: tuple[float, float, float],
        weight: float = LESIONNESS_LOSS_WEIGHT,
        radius_mm: float = LESIONNESS_DILATION_RADIUS_MM,
    ) -> None:
        super().__init__()
        if weight < 0:
            raise ValueError(f"lesionness 损失权重必须非负，收到 {weight}")
        spacing = tuple(float(s) for s in spacing_zyx)
        if len(spacing) != 3 or any(not np.isfinite(s) or s <= 0 for s in spacing):
            raise ValueError(
                f"spacing_zyx 必须是三个有限正数（Z, Y, X 数组轴序），收到 {spacing_zyx!r}"
            )
        self.main_loss = main_loss
        #: 辅助 logits 的来源模块；loss 只读它的 ``auxiliary_outputs``
        self.network = network
        self.spacing_zyx = spacing
        self.weight = float(weight)
        self.radius_mm = float(radius_mm)

    @staticmethod
    def _full_resolution_target(target) -> Tensor:
        """取 deep supervision 列表的第 0 项（全分辨率）或单张量本身。"""
        if isinstance(target, (list, tuple)):
            if not target:
                raise ValueError("deep supervision target 列表为空")
            return target[0]
        return target

    def _coarse_target(self, full_resolution: Tensor, coarse_shape) -> Tensor:
        """由全分辨率 GT 生成 coarse lesionness 目标（膨胀 -> 最大池化到 coarse 网格）。"""
        # target 可能是 [B,1,D,H,W]（nnU-Net 默认）或 [B,D,H,W]
        squeezed = full_resolution
        if squeezed.ndim == 5:
            if squeezed.shape[1] != 1:
                raise ValueError(
                    f"期望单通道 target，收到 {tuple(squeezed.shape)}"
                )
            squeezed = squeezed[:, 0]
        if squeezed.ndim != 4:
            raise ValueError(f"target 必须为 [B,D,H,W]，收到 {tuple(squeezed.shape)}")
        outputs = []
        for batch_index in range(squeezed.shape[0]):
            array = squeezed[batch_index].detach().cpu().numpy()
            if not np.isin(array, [0, 1]).all():
                raise ValueError("lesion GT 必须是二值 0/1；lesionness 目标不接受其它取值")
            dilated = lesionness_target(array, self.spacing_zyx, radius_mm=self.radius_mm)
            reduced = _max_pool_to_shape(dilated, tuple(int(n) for n in coarse_shape))
            outputs.append(torch.from_numpy(reduced.astype(np.float32)))
        stacked = torch.stack(outputs, dim=0)
        return stacked.unsqueeze(1).to(full_resolution.device)

    def forward(self, output, target) -> Tensor:
        auxiliary = getattr(self.network, "auxiliary_outputs", None)
        if not isinstance(auxiliary, dict) or "lesionness_logits" not in auxiliary:
            raise RuntimeError(
                "network.auxiliary_outputs 缺少 lesionness_logits：lesionness 辅助损失要求"
                "网络在本步 forward 中已填充该字段（见 LesionAwareCoarseToFineNNUNet 的限制"
                "说明）。拒绝在缺失辅助输出的情况下静默退化为纯主损失。"
            )
        logits = auxiliary["lesionness_logits"]
        main = self.main_loss(output, target)
        if self.weight == 0:
            return main
        coarse_target = self._coarse_target(
            self._full_resolution_target(target), logits.shape[2:]
        )
        auxiliary_loss = F.binary_cross_entropy_with_logits(logits, coarse_target)
        return main + self.weight * auxiliary_loss


def _max_pool_to_shape(array: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    """把 3D 二值数组最大池化到 ``target_shape``；不整除时 fail-closed。

    与 nnU-Net deep supervision 的 2 倍降采样约定一致；不整除即报错，避免静默坐标错位。
    """
    current = np.asarray(array, dtype=bool)
    target = tuple(int(t) for t in target_shape)
    if current.ndim != 3:
        raise ValueError(f"期望 3D 数组，收到 shape={current.shape}")
    for axis in range(3):
        if current.shape[axis] == target[axis]:
            continue
        if current.shape[axis] < target[axis]:
            raise ValueError(
                f"coarse 目标是上采样方向：{current.shape} -> {target}；"
                "lesionness 只在 coarse 网格上监督，不做上采样"
            )
        if current.shape[axis] % target[axis] != 0:
            raise ValueError(
                f"降采样必须整除：轴 {axis} {current.shape[axis]} -> {target[axis]}"
            )
        factor = current.shape[axis] // target[axis]
        new_shape = (
            current.shape[:axis]
            + (target[axis], factor)
            + current.shape[axis + 1 :]
        )
        current = current.reshape(new_shape).max(axis=axis + 1)
    return current


__all__ = (
    "LESIONNESS_LOSS_WEIGHT",
    "LesionnessAuxiliaryLoss",
    "PICAIFocalCrossEntropyLossMixin",
    "PiCAIFocalCrossEntropyLoss",
)
