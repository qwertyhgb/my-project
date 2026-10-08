"""nnU-Net 集成层（新主线）。

本包是项目对固定版本 nnU-Net（``third_party/nnUNet``，v2.6.2，只读）的扩展点。分层：

=================  ==========================================================
模块                职责
=================  ==========================================================
:mod:`runtime`     运行时校验：``nnunetv2`` 必须解析到项目固定源码（不升级任何包）
:mod:`bases`       与 legacy 共用的 Trainer 基类（strong baseline A1/A2 + Stage 1）
:mod:`losses`      PI-CAI Focal+CE 与 lesionness 辅助损失（**只有这两样**）
:mod:`augmentation` NoFFT 兼容修复（只修 bug，不改增强语义）
:mod:`roi_sampling` Anatomy-Guided ROI 的训练采样层（条件 B）
:mod:`trainers`    ACTIVE（B/C/D/E）Trainer + legacy 兼容再导出 + 注册表
:mod:`networks`    兼容转发层（旧融合网络 + 新 lesion 网络），**不是**活跃实现位置
:mod:`transforms`  兼容转发层（旧先验增强边界）
:mod:`sampling`    兼容转发层 -> :mod:`..sampling`
=================  ==========================================================

planning/preprocessing、dataloader 主体、augmentation 主体、deep supervision、
optimizer/scheduler、training loop、checkpoint/resume、validation、sliding-window inference
与 prediction export 全部由 nnU-Net 负责，不在此重复实现。
"""

from __future__ import annotations

from zonal_reliability_fusion.nnunet.runtime import (
    DNA_REQUIRED_RANGE,
    FIXED_NNUNET_ROOT,
    PLAIN_CONV_UNET_DOTTED_PATH,
    PROJECT_ROOT,
    check_fixed_nnunet_runtime,
)

__all__ = (
    "DNA_REQUIRED_RANGE",
    "FIXED_NNUNET_ROOT",
    "PLAIN_CONV_UNET_DOTTED_PATH",
    "PROJECT_ROOT",
    "check_fixed_nnunet_runtime",
)
