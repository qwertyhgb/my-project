"""自研模型层：plan-driven 主干 + 各模型（M0 基线与 M1–M4 融合链）。

三种模型身份并存、互不覆盖，且 checkpoint 严格隔离：

- **legacy（v2.2）**：`PlanDrivenUNet3D` + `M0ConcatModel`（PlainConv encoder），保留用于审计/基础设施回归；
- **v2.3 M0**：`PlanDrivenResidualEncoderUNet3D` + `M0ResidualConcatModel`（Residual-Encoder，三序列 concat）；
- **v2.3 M1–M4**：`MultiBranchFusionUNet3D` 及其子类（三路 modality-specific 浅层分支 + encoder stage 2 融合
  + 共享 Residual-Encoder 后半段）；融合结构由 `fusion_family` 架构配置解析，逐模型架构哈希不同。
"""
from .blocks import (
    ConvNormAct3d,
    StackedConvBlocks3d,
    apply_he_init,
    center_crop_or_pad_3d,
    init_weights_he,
    same_padding,
)
from .factory import (
    SUPPORTED_MODEL_IDS,
    build_m0_model,
    build_model,
    model_architecture_identity,
    model_identity,
    resolve_architecture_spec,
)
from .fusion_blocks import (
    ConditionalAffineHead,
    ImageDrivenGate,
    ModalityProjection,
    ShallowSkipAggregator,
    ZoneEncoder,
    ZoneFusionProjection,
)
from .fusion_multi_branch import (
    INPUT_KEY_IMAGE,
    INPUT_KEY_ZONAL,
    M1EqualFusionModel,
    M2ImageGateFusionModel,
    M3ZoneInputFusionModel,
    M4ConditionedGateFusionModel,
    MultiBranchFusionUNet3D,
)
from .m0_concat import MODALITY_ORDER, M0ConcatModel
from .m0_residual_concat import FORBIDDEN_INPUTS, M0ResidualConcatModel
from .residual_blocks import ResidualBlock3d, StackedResidualBlocks3d
from .residual_unet3d import (
    PlanDrivenResidualEncoderUNet3D,
    zero_init_residual_last_norms,
)
from .unet3d import PlanDrivenUNet3D

__all__ = [
    "FORBIDDEN_INPUTS",
    "INPUT_KEY_IMAGE",
    "INPUT_KEY_ZONAL",
    "MODALITY_ORDER",
    "SUPPORTED_MODEL_IDS",
    "ConditionalAffineHead",
    "ConvNormAct3d",
    "ImageDrivenGate",
    "M0ConcatModel",
    "M0ResidualConcatModel",
    "M1EqualFusionModel",
    "M2ImageGateFusionModel",
    "M3ZoneInputFusionModel",
    "M4ConditionedGateFusionModel",
    "ModalityProjection",
    "MultiBranchFusionUNet3D",
    "PlanDrivenResidualEncoderUNet3D",
    "PlanDrivenUNet3D",
    "ResidualBlock3d",
    "ShallowSkipAggregator",
    "StackedConvBlocks3d",
    "StackedResidualBlocks3d",
    "ZoneEncoder",
    "ZoneFusionProjection",
    "apply_he_init",
    "build_m0_model",
    "build_model",
    "center_crop_or_pad_3d",
    "init_weights_he",
    "model_architecture_identity",
    "model_identity",
    "resolve_architecture_spec",
    "same_padding",
    "zero_init_residual_last_norms",
]
