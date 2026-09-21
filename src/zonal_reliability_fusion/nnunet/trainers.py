"""项目对 nnU-Net v2.6.2 的全部 Trainer 扩展（单文件）。

三个 variant 共享同一套 nnU-Net 训练机制（optimizer=SGD+Nesterov、PolyLR、1000 epochs、
前景 patch 采样、增强主体、checkpoint/resume、validation、sliding-window inference），
差异只有两处，且都通过覆盖 nnU-Net 的钩子实现，绝不重实现训练循环：

1. 损失：PI-CAI 官方风格 ``0.5*Focal(gamma=2) + 0.5*CE``（替换默认 Dice+CE）；
2. 网络：baseline 用原生 backbone；image_gate / anatomy_gate 在原生 backbone 前加一个
   零初始化的 spatial modality reliability gate。

三个 Trainer 类名不同，nnU-Net 据此生成互不覆盖的 output folder：
    nnUNetTrainerPICAI_FLCE_NoFFT__nnUNetPlans__3d_fullres      (baseline，即已完成的 N0)
    nnUNetTrainerPICAI_ImageGate__nnUNetPlans__3d_fullres       (image_gate)
    nnUNetTrainerPICAI_AnatomyGate__nnUNetPlans__3d_fullres     (anatomy_gate)

NoFFT 修复：本环境 torch 与 fft-conv-pytorch 在增强 worker 中执行 GaussianBlurTransform 的
FFT *benchmark* 时会触发 native 内存损坏。修复保持 blur 的概率/sigma/通道不变，只关闭其
benchmark（强制走普通 convolution），彻底不执行不稳定的 FFT 试跑。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from torch import Tensor, nn

from zonal_reliability_fusion.nnunet.networks import (
    ANATOMY_INPUT_CHANNELS,
    MRI_CHANNELS,
    PRIOR_CHANNELS_ANATOMY,
    GatedNNUNet,
    SpatialModalityReliabilityGate,
)
from zonal_reliability_fusion.nnunet.transforms import (
    restrict_intensity_transforms_to_mri,
)


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

        # 复刻 PI-CAI v1 FocalLoss：Softmax → one-hot clamp(smooth, 1-smooth)
        # → pt + smooth → -(1-pt)^gamma * log(pt)。alpha=None 即全 1。
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


class NoFFTAugmentationMixin:
    """保留 nnU-Net 默认增强语义，仅关闭 GaussianBlurTransform 的 FFT benchmark。"""

    @staticmethod
    def get_training_transforms(*args, **kwargs):
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


class nnUNetTrainerPICAI_FLCE_NoFFT(
    NoFFTAugmentationMixin, PICAIFocalCrossEntropyLossMixin, nnUNetTrainer
):
    """baseline variant：原生 ``PlainConvUNet`` + 3 个 MRI 通道 + PI-CAI Focal+CE + NoFFT 修复。

    除损失与 blur benchmark 外，全部继承 nnU-Net v2.6.2 默认训练行为。类名与已完成的 N0
    运行一致，output folder 保持不变（可续训 / 只做 validation）。
    """


class _GatedTrainerBase(nnUNetTrainerPICAI_FLCE_NoFFT):
    """image_gate / anatomy_gate 的公共基类：只覆盖网络构建，其余全部继承 baseline。"""

    #: 子类声明其严格期望的输入通道数（image=3，anatomy=5）
    expected_input_channels: int = MRI_CHANNELS
    #: 子类声明 prior 通道数（image=0，anatomy=2）
    num_prior_channels: int = 0

    @staticmethod
    def _build_gated_network(
        architecture_class_name,
        arch_init_kwargs,
        arch_init_kwargs_req_import,
        num_input_channels,
        num_output_channels,
        enable_deep_supervision,
        *,
        expected_input_channels,
        num_prior_channels,
    ) -> nn.Module:
        if num_input_channels != expected_input_channels:
            raise ValueError(
                f"该 Trainer 严格要求 {expected_input_channels} 个输入通道"
                f"（{MRI_CHANNELS} MRI + {num_prior_channels} prior），收到 {num_input_channels}"
            )
        # backbone 永远只吃 3 个 MRI 通道；由 nnU-Net 官方工具按 plans 构建并初始化。
        backbone = get_network_from_plans(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            MRI_CHANNELS,
            num_output_channels,
            allow_init=True,
            deep_supervision=enable_deep_supervision,
        )
        gate = SpatialModalityReliabilityGate(input_channels=expected_input_channels)
        return GatedNNUNet(
            backbone,
            gate,
            mri_channels=MRI_CHANNELS,
            num_prior_channels=num_prior_channels,
        )


class nnUNetTrainerPICAI_ImageGate(_GatedTrainerBase):
    """image_gate variant：3 个 MRI 通道 -> gate 产生 3 通道 modality weights -> 原生 backbone。

    gate 只依据 T2W/ADC/HBV 本身，不引入任何解剖先验；不重实现 encoder/decoder。
    """

    expected_input_channels = MRI_CHANNELS
    num_prior_channels = 0

    @staticmethod
    def build_network_architecture(
        architecture_class_name,
        arch_init_kwargs,
        arch_init_kwargs_req_import,
        num_input_channels,
        num_output_channels,
        enable_deep_supervision=True,
    ) -> nn.Module:
        return _GatedTrainerBase._build_gated_network(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision,
            expected_input_channels=MRI_CHANNELS,
            num_prior_channels=0,
        )


class nnUNetTrainerPICAI_AnatomyGate(_GatedTrainerBase):
    """anatomy_gate variant：5 通道输入（T2W/ADC/HBV/PZ/TZ）。

    gate 依据 MRI + PZ/TZ 产生 3 个 MRI modality weights；只把加权后的前 3 个 MRI 通道送入
    原生 backbone，PZ/TZ 不进入分割 backbone（禁止 WG）。PZ/TZ 在进入 gate 前 clamp(0,1)。
    另外把强度增强限制在 MRI 通道，空间增强仍同步作用于 MRI/PZ/TZ。
    """

    expected_input_channels = ANATOMY_INPUT_CHANNELS
    num_prior_channels = PRIOR_CHANNELS_ANATOMY

    @staticmethod
    def build_network_architecture(
        architecture_class_name,
        arch_init_kwargs,
        arch_init_kwargs_req_import,
        num_input_channels,
        num_output_channels,
        enable_deep_supervision=True,
    ) -> nn.Module:
        return _GatedTrainerBase._build_gated_network(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision,
            expected_input_channels=ANATOMY_INPUT_CHANNELS,
            num_prior_channels=PRIOR_CHANNELS_ANATOMY,
        )

    @staticmethod
    def get_training_transforms(*args, **kwargs):
        # 先复用 baseline（含 NoFFT 修复）的默认增强，再把强度变换限制到 MRI 通道。
        transforms = nnUNetTrainerPICAI_FLCE_NoFFT.get_training_transforms(
            *args, **kwargs
        )
        transforms, n_wrapped = restrict_intensity_transforms_to_mri(
            transforms, MRI_CHANNELS
        )
        if n_wrapped == 0:
            raise RuntimeError(
                "anatomy_gate 未能在增强管线中找到任何强度变换来限制到 MRI 通道；"
                "拒绝静默启动，以免把 noise/blur/brightness/contrast/gamma 施加到 PZ/TZ 上。"
            )
        return transforms


#: 项目 Trainer 名 -> 类。训练入口与预测入口都用它做**进程内直接映射**，
#: 避免依赖 nnU-Net 的 ``recursive_find_python_class``（该函数只在 nnunetv2 包目录内递归扫描，
#: 找不到项目自定义类，且可能扫到环境中其他 nnunetv2 分支）。
PROJECT_TRAINERS: dict[str, type[nnUNetTrainer]] = {
    nnUNetTrainerPICAI_FLCE_NoFFT.__name__: nnUNetTrainerPICAI_FLCE_NoFFT,
    nnUNetTrainerPICAI_ImageGate.__name__: nnUNetTrainerPICAI_ImageGate,
    nnUNetTrainerPICAI_AnatomyGate.__name__: nnUNetTrainerPICAI_AnatomyGate,
}


def resolve_trainer_class(trainer_name: str) -> type[nnUNetTrainer] | None:
    """按类名解析项目 Trainer；不是项目 Trainer 时返回 ``None``（由调用方决定是否回退）。"""
    return PROJECT_TRAINERS.get(trainer_name)
