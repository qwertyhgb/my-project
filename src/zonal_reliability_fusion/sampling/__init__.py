"""训练期采样策略（新主线的 training-strategy 层，不是方法创新）。

本包只包含**受控扩展**：继承或包装 nnU-Net 原生 ``nnUNetDataLoader``，复用其预处理
``class_locations`` 与 ``get_bbox``，绝不重实现 patch 裁剪、数据读取或增强管线。

- :mod:`positive_sampling`——阳性病例感知采样。**属于 strong baseline**，不是创新点。
  它回答的初步问题「增加病灶暴露是否减少漏检」已由既有实验回答（是），代价是假阳上升。
- :mod:`hard_negative`——解剖约束的困难负样本挖掘（新的 training 策略候选，条件 E）。
  直接针对「提高 sensitivity 后引入过多 false positives」这一失败模式。

两条硬约束（fail-closed）
------------------------
1. **训练/验证严格分离**：验证 loader 恒为原生 ``nnUNetDataLoader`` 与原生采样行为；
   任何阳性重采样或困难负样本注入都**只**作用于训练 split。
2. **挖掘只能在训练 split 内进行**：不得用 validation / test 的预测结果反向挑选训练样本。
   :func:`hard_negative.validate_mining_scope` 是这一约束的自动守卫。
"""

from __future__ import annotations

from zonal_reliability_fusion.sampling.hard_negative import (
    DEFAULT_HARD_NEGATIVE_DIR,
    HARD_NEGATIVE_SCHEMA_VERSION,
    HardNegativeDataLoader,
    HardNegativeMiningError,
    HardNegativeMiningMixin,
    build_hard_negative_set,
    load_hard_negative_set,
    select_hard_negative_locations,
    validate_mining_scope,
    write_hard_negative_set,
)
from zonal_reliability_fusion.sampling.positive_sampling import (
    PositiveCaseDataLoader,
    PositiveCaseSamplingMixin,
    identify_positive_cases,
)

__all__ = (
    "DEFAULT_HARD_NEGATIVE_DIR",
    "HARD_NEGATIVE_SCHEMA_VERSION",
    "HardNegativeDataLoader",
    "HardNegativeMiningError",
    "HardNegativeMiningMixin",
    "PositiveCaseDataLoader",
    "PositiveCaseSamplingMixin",
    "build_hard_negative_set",
    "identify_positive_cases",
    "load_hard_negative_set",
    "select_hard_negative_locations",
    "validate_mining_scope",
    "write_hard_negative_set",
)
