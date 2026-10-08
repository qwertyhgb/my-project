"""兼容转发层：``zonal_reliability_fusion.nnunet.networks``。

历史背景：本项目**曾经**把全部网络扩展都放在这个模块里（输入级 modality gate、浅层序列
特异特征融合、同区参照残差融合）。2026-10-08 起这些实现迁入
``zonal_reliability_fusion.legacy.fusion_networks``，因为它们属于**已归档的旧研究线**。

保留本模块的原因只有一个：**checkpoint 与历史调用点兼容**。既有 checkpoint 的
``state_dict`` 键名由那些类的结构决定，测试与 ``docs/Development_Log.md`` 的历史记录也按
``...nnunet.networks`` 引用它们；改名或删除会让这些引用失效并破坏历史事实的可核查性。

新主线请直接使用明确位置：

- 网络：``zonal_reliability_fusion.lesion.coarse_to_fine``（coarse-to-fine 包装）
- 区域条件化：``zonal_reliability_fusion.lesion.zone_conditioning``
- 旧门控 / 融合网络：``zonal_reliability_fusion.legacy.fusion_networks``（只读归档）
"""

from __future__ import annotations

from zonal_reliability_fusion.legacy.fusion_networks import (
    ANATOMY_INPUT_CHANNELS,
    FEATURE_STEM_CHANNELS,
    FEATURE_STEM_KERNEL_SIZE,
    GATE_HIDDEN_CHANNELS,
    GATE_OUTPUT_CHANNELS,
    MRI_CHANNELS,
    PRIOR_CHANNELS_ANATOMY,
    FeatureFusionNNUNet,
    GatedNNUNet,
    ShallowSequenceStem,
    SpatialModalityReliabilityGate,
    ZonalReferenceFusionNNUNet,
    ZonalReferenceResidualFusion,
    feature_gate_parameter_delta,
)
from zonal_reliability_fusion.lesion.coarse_to_fine import (
    CoarseLesionnessHead,
    LesionAwareCoarseToFineNNUNet,
)

__all__ = (
    "ANATOMY_INPUT_CHANNELS",
    "FEATURE_STEM_CHANNELS",
    "FEATURE_STEM_KERNEL_SIZE",
    "GATE_HIDDEN_CHANNELS",
    "GATE_OUTPUT_CHANNELS",
    "MRI_CHANNELS",
    "PRIOR_CHANNELS_ANATOMY",
    "CoarseLesionnessHead",
    "FeatureFusionNNUNet",
    "GatedNNUNet",
    "LesionAwareCoarseToFineNNUNet",
    "ShallowSequenceStem",
    "SpatialModalityReliabilityGate",
    "ZonalReferenceFusionNNUNet",
    "ZonalReferenceResidualFusion",
    "feature_gate_parameter_delta",
)
