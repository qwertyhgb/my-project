"""数据层：PI-CAI 元数据/划分（历史实现）+ 预处理数据适配、采样与增强（P2-A 新增）。

模块职责：
- `picai`：PI-CAI 原始数据清点与方向码（历史实现）；
- `preprocessed_store`：唯一允许 import nnunetv2 的只读适配层（.b2nd/.pkl）；
- `patch_sampler`：M0–M4 共用 patch 采样（前景过采样 + 阴性安全回退）；
- `transforms`：同步镜像 + 分序列强度接口（验证集空管道）；
- `batch_provider`：组合成 torch batch 迭代器。
"""
from .batch_provider import BatchProviderError, TorchBatchProvider
from .patch_sampler import DATA_PAD_VALUE, FOREGROUND_LABEL, SEG_PAD_VALUE, PatchSample, PatchSampler
from .preprocessed_store import (
    EXPECTED_CHANNELS,
    CaseProperties,
    PreprocessedCase,
    PreprocessedStore,
    PreprocessedStoreError,
    normalize_data,
    normalize_seg,
    verify_case_contract,
)
from .transforms import (
    SPATIAL_AXES,
    AugmentationPipeline,
    MirrorAugmentor,
    PerChannelIntensityAugmentor,
    UnavailableAugmentor,
    build_train_transforms,
    build_validation_transforms,
)

__all__ = [
    "DATA_PAD_VALUE",
    "EXPECTED_CHANNELS",
    "FOREGROUND_LABEL",
    "SPATIAL_AXES",
    "SEG_PAD_VALUE",
    "AugmentationPipeline",
    "BatchProviderError",
    "CaseProperties",
    "MirrorAugmentor",
    "PatchSample",
    "PatchSampler",
    "PerChannelIntensityAugmentor",
    "PreprocessedCase",
    "PreprocessedStore",
    "PreprocessedStoreError",
    "TorchBatchProvider",
    "UnavailableAugmentor",
    "build_train_transforms",
    "build_validation_transforms",
    "normalize_data",
    "normalize_seg",
    "verify_case_contract",
]
