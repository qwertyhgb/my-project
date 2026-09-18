"""PI-CAI 官方 nnU-Net 基线损失的 nnU-Net v2 适配。

PI-CAI 主办方公开的 supervised nnU-Net 基线基于 nnU-Net v1，使用
``0.5 * FocalLoss(gamma=2) + 0.5 * CrossEntropyLoss``，而不是 v2 默认的
Dice + Cross-Entropy。此模块在项目代码中复刻该损失并提供 v2 trainer；
不修改 ``third_party/nnUNet``。

来源（访问日期：2026-09-15）：
- https://github.com/DIAGNijmegen/picai_baseline/blob/main/nnunet_baseline.md
- https://github.com/DIAGNijmegen/picai_nnunet_gc_algorithm/blob/main/nnUNetTrainerV2_Loss_FL_and_CE.py
- https://github.com/DIAGNijmegen/picai_nnunet_gc_algorithm/blob/main/nnUNetTrainerV2_focalLoss.py

这是一份 *v2 port*，而非逐位复现 v1：它保留本项目已经冻结的 v2.6.2
plans、split、数据物化和 deep supervision，仅替换每个监督尺度的基础损失。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class PiCAIFocalCrossEntropyLoss(nn.Module):
    """PI-CAI baseline 的等权 Focal + CE 二/多类体素损失。

    原始 trainer 将 ``FocalLoss(alpha=None, gamma=2, smooth=1e-5)`` 与
    Cross-Entropy 以 0.5/0.5 相加。``alpha=None`` 意味着没有额外的类别
    权重；请勿将仓库中另一个 ``focalLossAlpha75`` 变体误作官方主基线。

    输入为 nnU-Net 约定的 logits ``[B, C, ...]``，target 为 ``[B, 1, ...]``
    或 ``[B, ...]``。本项目 Dataset605 没有 ignore label，故遇到负标签会
    显式失败而不是静默把其视作背景。
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
            raise ValueError("target 含超出 [0, num_classes) 的标签；Dataset605 不支持 ignore label")

        # 复刻 PI-CAI v1 FocalLoss：Softmax → one-hot clamp(smooth, 1-smooth)
        # → pt + smooth → -(1-pt)^gamma * log(pt)。alpha=None 即全 1。
        probabilities = torch.softmax(logits, dim=1)
        one_hot = F.one_hot(target, num_classes=logits.shape[1]).movedim(-1, 1)
        one_hot = one_hot.to(dtype=probabilities.dtype)
        if self.smooth:
            # v1 对二分类使用 min=smooth/(C-1)。保留其多类语义。
            one_hot = one_hot.clamp(
                min=self.smooth / (logits.shape[1] - 1), max=1.0 - self.smooth
            )
        pt = (one_hot * probabilities).sum(dim=1) + self.smooth
        focal = -torch.pow(1.0 - pt, self.gamma) * torch.log(pt)
        focal = focal.mean()
        ce = F.cross_entropy(logits, target)
        return self.focal_weight * focal + self.ce_weight * ce


class nnUNetTrainerPICAI_FLCE(nnUNetTrainer):
    """nnU-Net v2.6.2 trainer using the PI-CAI official Focal + CE objective.

    其他训练行为均继承 v2.6.2 默认实现：network/plans、前景采样、增强、
    SGD+Nesterov、PolyLR、1000 epochs 和 checkpoint 规则不在此类中改动。

    唯一的运行时兼容性例外是 Gaussian blur 的 FFT *benchmark*：本环境的
    ``torch=2.10.0`` 与 ``fft-conv-pytorch=1.2.0`` 在后台增强 worker 中调用
    该 benchmark 时已发生 native ``free(): corrupted unsorted chunks``。因此
    保持 blur 的概率与 sigma 不变，只强制其使用 PyTorch 的常规 convolution
    backend，彻底不执行不稳定的 FFT 试跑。详见 docs/N0_PICAI_Official_Baseline.md。
    """

    @staticmethod
    def get_training_transforms(*args, **kwargs):
        """保留默认增强语义，仅关闭 GaussianBlurTransform 的 FFT benchmark。

        v2.6.2 默认构造 ``GaussianBlurTransform(..., benchmark=True)``。其
        benchmark 会在实际增强前分别试跑 ``conv`` 与 ``fft_conv``，后者在本
        环境的 worker 内引起原生内存损坏。设为 ``False`` 后同一 transform 仍
        以相同概率、相同 sigma 对相同通道做 Gaussian blur，只使用普通 convolution。
        """
        transforms = nnUNetTrainer.get_training_transforms(*args, **kwargs)
        disabled = 0
        for transform in transforms.transforms:
            candidate = getattr(transform, "transform", transform)
            if isinstance(candidate, GaussianBlurTransform):
                candidate.benchmark = False
                candidate.benchmark_use_fft.clear()
                disabled += 1
        if disabled != 1:
            raise RuntimeError(
                "nnU-Net v2.6.2 augmentation pipeline 未找到唯一的 GaussianBlurTransform；"
                "拒绝静默启动，以免重新调用不稳定的 fft_conv benchmark。"
            )
        return transforms

    def _build_loss(self) -> nn.Module:
        if self.label_manager.has_regions:
            raise RuntimeError(
                "nnUNetTrainerPICAI_FLCE 仅支持互斥二/多类标签；"
                "Dataset605_PICAI 应为 background=0, lesion=1。"
            )
        if self.label_manager.ignore_label is not None:
            raise RuntimeError(
                "nnUNetTrainerPICAI_FLCE 不支持 ignore label；"
                "Dataset605_PICAI 的 dataset.json 不应声明 ignore。"
            )

        loss: nn.Module = PiCAIFocalCrossEntropyLoss(
            focal_weight=0.5,
            ce_weight=0.5,
            gamma=2.0,
            smooth=1e-5,
        )
        if self.enable_deep_supervision:
            scales = self._get_deep_supervision_scales()
            weights = np.array([1.0 / (2**i) for i in range(len(scales))], dtype=np.float64)
            # 与本项目固定 nnU-Net v2.6.2 的默认 trainer 一致：最低分辨率不监督。
            # DDP 的特殊 1e-6 分支也沿用官方实现。
            weights[-1] = 1e-6 if self.is_ddp and not self._do_i_compile() else 0.0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
        return loss


class nnUNetTrainerPICAI_FLCE_NoFFT(nnUNetTrainerPICAI_FLCE):
    """No-FFT 修复版的独立运行身份。

    nnU-Net 用 trainer 类名构造 output path。保留父类以便完整保留首次
    ``nnUNetTrainerPICAI_FLCE`` 崩溃运行的日志；本子类不改变任何额外训练
    行为，只确保修复版写入一个全新的、不可覆盖的输出目录。
    """

    pass
