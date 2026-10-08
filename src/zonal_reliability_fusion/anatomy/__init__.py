"""Stage 1：预测解剖先验（WG / PZ / TZ）的生成与校验。

Stage 1 **不是**论文的主要创新点。它的定位是 ``anatomical prior generator``：训练一个稳定、
独立、冻结的模型，对 lesion 数据集的每个病例生成**该病例自身 MRI 的** soft probability。

本包的分层
----------
- :mod:`contracts`——数据集/标签/split/概率导出的**冻结契约**（准备、训练、推理、评估四方共用）；
- :mod:`dataset`——先验的存放布局与 ``predicted prior`` / ``GT`` 的来源标记；
- :mod:`inference`——预测入口守卫与导出产物完整性校验（推理本身仍由 nnU-Net 官方实现完成）；
- :mod:`validation`——区域重叠、区域一致性、先验不确定度报告。

不可越过的边界（贯穿全包）
--------------------------
1. anatomy model **不读取 lesion GT**；
2. validation / test 的 anatomy prior **必须**来自该病例自身 MRI 的预测；
3. **禁止**把 GT WG/PZ/TZ 作为 lesion model 的推理输入（仅允许 ``ORACLE_GT`` 上界分析）；
4. anatomy 预测错误被视为**真实部署链条的一部分**，不做任何手工修正。
"""

from __future__ import annotations

from zonal_reliability_fusion.anatomy.contracts import (
    ANATOMY_CLASS_ORDER,
    ANATOMY_CONFIGURATION,
    ANATOMY_COUNTS,
    ANATOMY_DATASET,
    ANATOMY_GEOMETRY_ATOL,
    ANATOMY_KNOWN_MISSING,
    ANATOMY_LABELS,
    encode_anatomy,
    encode_anatomy_heads,
    validate_anatomy_dataset,
    validate_anatomy_probabilities,
    validate_anatomy_split,
    validate_label_array,
)
from zonal_reliability_fusion.anatomy.dataset import (
    ANATOMY_PROBABILITY_CHANNELS,
    DEFAULT_ANATOMY_PRIOR_DIR,
    PRIOR_LAYOUT,
    case_prior_paths,
    prior_directory,
)
from zonal_reliability_fusion.anatomy.validation import (
    anatomy_region_agreement,
    anatomy_region_metrics,
    prior_uncertainty_report,
)

__all__ = (
    "ANATOMY_CLASS_ORDER",
    "ANATOMY_CONFIGURATION",
    "ANATOMY_COUNTS",
    "ANATOMY_DATASET",
    "ANATOMY_GEOMETRY_ATOL",
    "ANATOMY_KNOWN_MISSING",
    "ANATOMY_LABELS",
    "ANATOMY_PROBABILITY_CHANNELS",
    "DEFAULT_ANATOMY_PRIOR_DIR",
    "PRIOR_LAYOUT",
    "anatomy_region_agreement",
    "anatomy_region_metrics",
    "case_prior_paths",
    "encode_anatomy",
    "encode_anatomy_heads",
    "prior_directory",
    "prior_uncertainty_report",
    "validate_anatomy_dataset",
    "validate_anatomy_probabilities",
    "validate_anatomy_split",
    "validate_label_array",
)
