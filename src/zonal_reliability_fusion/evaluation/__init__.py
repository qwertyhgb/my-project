"""评价体系包：新主线唯一的指标定义来源。

分层（与 ``docs/Evaluation_Protocol.md`` 一一对应）：

- :mod:`protocol`——冻结常量、异常、bootstrap、通用数值/JSON 工具；
- :mod:`case_metrics`——病例级：物理体积、连通域失败分析、HD95 / ASSD / NSD@τ；
- :mod:`lesion_metrics`——病灶实例级：匹配协议、大小分层、matched-lesion Dice；
- :mod:`anatomy_metrics`——按**预测**解剖先验的FP 负担分解与病灶定位分解。

本包与模型完全解耦：不 import nnU-Net、不 import torch、不读写文件（除
:func:`case_metrics.load_mask_pair` 这一显式的只读掩膜入口）。训练/推理代码可以安全地
import 它而不会引入循环依赖。
"""

from __future__ import annotations

from zonal_reliability_fusion.evaluation.anatomy_metrics import (
    ALLOWED_PRIOR_SOURCES,
    ANATOMY_DEFINITIONS,
    PRIOR_SOURCE_ORACLE_GT,
    PRIOR_SOURCE_PREDICTED,
    ZONE_KEYS,
    false_positive_burden,
    fp_component_counts_by_region,
    lesion_anatomy_localization,
    validate_anatomy_priors,
    zone_assignment,
)
from zonal_reliability_fusion.evaluation.case_metrics import (
    component_analysis,
    dice_from_counts,
    load_mask_pair,
    nsd_surface_area,
    pred_component_stats,
    recompute_case_counts,
    spacing_zyx,
    surface_distance_metrics,
    surface_metrics,
    verify_case_counts,
    voxel_volume_mm3,
)
from zonal_reliability_fusion.evaluation.lesion_metrics import (
    aggregate_lesion_instance_metrics,
    lesion_instance_metrics,
    lesion_size_stratum,
)
from zonal_reliability_fusion.evaluation.protocol import (
    CONNECTIVITY,
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_SEED,
    DICE_MATCH_TOLERANCE,
    ENDPOINT_TIERS,
    LESION_SIZE_MEDIUM_MAX_MM3,
    LESION_SIZE_SMALL_MAX_MM3,
    LESION_SIZE_STRATUM_KEYS,
    SCHEMA_VERSION,
    EvaluationError,
    bootstrap_ci_mean,
    bootstrap_ci_paired_mean,
    bootstrap_ci_paired_ratio,
    dist_stats,
    finite,
    json_safe,
    percentile_ci,
    ratio,
)

__all__ = (
    "ALLOWED_PRIOR_SOURCES",
    "ANATOMY_DEFINITIONS",
    "CONNECTIVITY",
    "DEFAULT_BOOTSTRAP_RESAMPLES",
    "DEFAULT_SEED",
    "DICE_MATCH_TOLERANCE",
    "ENDPOINT_TIERS",
    "LESION_SIZE_MEDIUM_MAX_MM3",
    "LESION_SIZE_SMALL_MAX_MM3",
    "LESION_SIZE_STRATUM_KEYS",
    "PRIOR_SOURCE_ORACLE_GT",
    "PRIOR_SOURCE_PREDICTED",
    "SCHEMA_VERSION",
    "ZONE_KEYS",
    "EvaluationError",
    "aggregate_lesion_instance_metrics",
    "bootstrap_ci_mean",
    "bootstrap_ci_paired_mean",
    "bootstrap_ci_paired_ratio",
    "component_analysis",
    "dice_from_counts",
    "dist_stats",
    "false_positive_burden",
    "finite",
    "fp_component_counts_by_region",
    "json_safe",
    "lesion_anatomy_localization",
    "lesion_instance_metrics",
    "lesion_size_stratum",
    "load_mask_pair",
    "nsd_surface_area",
    "percentile_ci",
    "pred_component_stats",
    "ratio",
    "recompute_case_counts",
    "spacing_zyx",
    "surface_distance_metrics",
    "surface_metrics",
    "validate_anatomy_priors",
    "verify_case_counts",
    "voxel_volume_mm3",
    "zone_assignment",
)
