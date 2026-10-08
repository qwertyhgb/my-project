"""兼容转发层：``zonal_reliability_fusion.nnunet.sampling`` -> ``...sampling.positive_sampling``。

阳性病例采样自 2026-10-08 起是新主线 **strong baseline 的组成部分**（不是创新点），其实现
移入 ``zonal_reliability_fusion.sampling`` 包。本模块仅为历史导入路径保留转发：

- ``scripts/``、``tests/`` 与已归档的 ``legacy.fusion_trainers`` 中的旧 import 继续可用；
- ``docs/Development_Log.md`` 记录的路径引用无需改写（历史事实不改）。

新代码请直接 ``from zonal_reliability_fusion.sampling import PositiveCaseSamplingMixin``。
"""

from __future__ import annotations

from zonal_reliability_fusion.sampling.positive_sampling import (
    PositiveCaseDataLoader,
    PositiveCaseSamplingMixin,
    identify_positive_cases,
)

__all__ = (
    "PositiveCaseDataLoader",
    "PositiveCaseSamplingMixin",
    "identify_positive_cases",
)
