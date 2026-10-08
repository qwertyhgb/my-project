"""Stage 2：lesion segmentation——论文真正的方法所在。

本包实现 Anatomy-Guided Lesion-Aware Coarse-to-Fine 框架的**方法侧**，只围绕三个核心失败
模式展开：

1. **完全漏检 lesion**（小病灶、困难病灶）——:mod:`lesionness` 的 coarse 定位监督；
2. **已检出病灶覆盖不足**——:mod:`coarse_to_fine` 的 soft lesion-guided refinement；
3. **提高 sensitivity 后假阳显著增加**——:mod:`roi` 的解剖约束搜索空间、
   :mod:`zone_conditioning` 的 soft 区域上下文与 :mod:`..sampling.hard_negative` 的困难负样本。

模块职责
--------
- :mod:`baseline`——strong baseline 的组成与单变量边界（所有消融的参照）；
- :mod:`roi`——Anatomy-Guided Prostate ROI（物理 margin，禁止 hard mask 乘法，不重采样）；
- :mod:`lesionness`——coarse lesionness 目标（物理半径膨胀的粗掩膜，唯一的研究性超参数已冻结）；
- :mod:`coarse_to_fine`——原生 backbone + coarse 头 + soft 残差 refinement；
- :mod:`zone_conditioning`——PZ/TZ 作为**解剖上下文**（不是 modality reweighting）；
- :mod:`prior_channels`——先验通道的增强边界（强度只作用于 MRI，空间同步作用于全部通道）。

设计顺序（任何时候都不允许颠倒）
--------------------------------
::

    strong baseline -> anatomy ROI -> lesion localization
    -> coarse-to-fine refinement -> zone anatomy context -> hard-negative control

如果某个模块无法回答「是否减少漏检 / 是否改善覆盖 / 是否减少假阳」中的任何一条，它原则上
不应该进入主模型。
"""

from __future__ import annotations

from zonal_reliability_fusion.lesion.baseline import (
    FAILURE_MODE_QUESTIONS,
    MAINLINE_CONDITIONS,
    STRONG_BASELINE_CANDIDATES,
    STRONG_BASELINE_COMPONENTS,
    TRAINING_STRATEGY_CONDITIONS,
    baseline_single_variable_boundary,
)
from zonal_reliability_fusion.lesion.coarse_to_fine import (
    CoarseLesionnessHead,
    LesionAwareCoarseToFineError,
    LesionAwareCoarseToFineNNUNet,
)
from zonal_reliability_fusion.lesion.lesionness import (
    LESIONNESS_DILATION_RADIUS_MM,
    LesionnessError,
    downsample_binary_max,
    lesionness_target,
    physical_radius_to_voxels,
)
from zonal_reliability_fusion.lesion.roi import (
    ProstateROI,
    ROIError,
    build_prostate_roi,
    expand_box_by_physical_margin,
    roi_retains_reference_lesions,
)
from zonal_reliability_fusion.lesion.zone_conditioning import (
    ZONE_MODES,
    ZoneAwareRefinement,
    zone_context,
    zone_mode_parameter_delta,
)

__all__ = (
    "FAILURE_MODE_QUESTIONS",
    "LESIONNESS_DILATION_RADIUS_MM",
    "MAINLINE_CONDITIONS",
    "STRONG_BASELINE_CANDIDATES",
    "STRONG_BASELINE_COMPONENTS",
    "TRAINING_STRATEGY_CONDITIONS",
    "ZONE_MODES",
    "CoarseLesionnessHead",
    "LesionAwareCoarseToFineError",
    "LesionAwareCoarseToFineNNUNet",
    "LesionnessError",
    "ProstateROI",
    "ROIError",
    "ZoneAwareRefinement",
    "baseline_single_variable_boundary",
    "build_prostate_roi",
    "downsample_binary_max",
    "expand_box_by_physical_margin",
    "lesionness_target",
    "physical_radius_to_voxels",
    "roi_retains_reference_lesions",
    "zone_context",
    "zone_mode_parameter_delta",
)
