"""兼容转发层：``zonal_reliability_fusion.nnunet.transforms``。

旧实现（为 Dataset606 的 PZ/TZ 输入通道把强度增强限制到 MRI）已迁入
``zonal_reliability_fusion.legacy.prior_transforms``。新主线的活跃实现在
``zonal_reliability_fusion.lesion.prior_channels``，行为契约相同。

保留本模块是为了历史调用点与测试不失效；**新代码请直接从新位置导入**。
"""

from __future__ import annotations

from zonal_reliability_fusion.legacy.prior_transforms import (
    INTENSITY_TRANSFORM_TYPES,
    MRIChannelRestrictedTransform,
    restrict_intensity_transforms_to_mri,
)

__all__ = (
    "INTENSITY_TRANSFORM_TYPES",
    "MRIChannelRestrictedTransform",
    "restrict_intensity_transforms_to_mri",
)
