"""训练层：损失、deep supervision、指标、优化器、checkpoint 与 M0 训练循环。

不依赖 nnUNetTrainer；数据由 `data.batch_provider` 提供 CPU 张量 batch。
"""
from .checkpointing import (
    CHECKPOINT_BEST,
    CHECKPOINT_INTERRUPT,
    CHECKPOINT_LAST,
    CheckpointManager,
    atomic_torch_save,
    load_checkpoint,
    save_checkpoint,
    verify_checkpoint_architecture,
)
from .deep_supervision import (
    DeepSupervisionLoss,
    build_ds_targets,
    deep_supervision_weights,
    downsample_target_nearest,
)
from .losses import DiceCELoss, PiCAIFocalCELoss, SoftDiceLoss, prepare_target
from .metrics import MetricAccumulator, SegmentCounts, counts_from_logits
from .optim import PolyLRScheduler, build_optimizer, build_scheduler
from .trainer import (
    VALIDATION_MODE_ALIASES,
    VALIDATION_MODE_DIAGNOSTIC,
    VALIDATION_MODE_FORMAL,
    VALIDATION_MODE_FULL_VOLUME,
    VALIDATION_MODES,
    EarlyStoppingConfig,
    M0Trainer,
    TrainerSummary,
)

__all__ = [
    "VALIDATION_MODE_ALIASES",
    "VALIDATION_MODE_DIAGNOSTIC",
    "VALIDATION_MODE_FORMAL",
    "VALIDATION_MODE_FULL_VOLUME",
    "VALIDATION_MODES",
    "EarlyStoppingConfig",
    "CHECKPOINT_BEST",
    "CHECKPOINT_INTERRUPT",
    "CHECKPOINT_LAST",
    "CheckpointManager",
    "DeepSupervisionLoss",
    "DiceCELoss",
    "PiCAIFocalCELoss",
    "M0Trainer",
    "MetricAccumulator",
    "PolyLRScheduler",
    "SegmentCounts",
    "SoftDiceLoss",
    "TrainerSummary",
    "atomic_torch_save",
    "build_ds_targets",
    "build_optimizer",
    "build_scheduler",
    "counts_from_logits",
    "deep_supervision_weights",
    "downsample_target_nearest",
    "load_checkpoint",
    "prepare_target",
    "save_checkpoint",
    "verify_checkpoint_architecture",
]
