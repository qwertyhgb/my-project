"""评估层：低频全体积验证（v2.2：正式 M0–M4 每 `every_n_epochs=5` 一次，最后一个 epoch 必验证）。

- `FullVolumeValidator`：对 fold 0 全部 validation study 做完整 3D 滑窗推理，产出 case-wise Dice 与
  阴性假阳性统计；
- checkpoint 指标固定为 `val_positive_casewise_dice_mean`（阴性空—空不记为 Dice=1）；
- 正式 picai_eval / 原始物理空间恢复放到最终评估阶段，不在此处实现。
"""
from .full_volume_validator import (
    CHECKPOINT_METRIC_FULL_VOLUME,
    FULL_VOLUME_METRIC_FIELDS,
    VALIDATION_MODES,
    CaseResult,
    FullVolumeValidator,
    ValidationProtocol,
    build_protocol_from_config,
)

__all__ = [
    "CHECKPOINT_METRIC_FULL_VOLUME",
    "FULL_VOLUME_METRIC_FIELDS",
    "VALIDATION_MODES",
    "CaseResult",
    "FullVolumeValidator",
    "ValidationProtocol",
    "build_protocol_from_config",
]
