"""增强管线的项目侧修补（只修 bug，不改增强语义）。

:class:`NoFFTAugmentationMixin` 保留 nnU-Net 默认增强的**全部语义**（概率、sigma、通道、
顺序都不变），只关闭 ``GaussianBlurTransform`` 的 FFT *benchmark*。

为什么必须这样做
----------------
本环境的 torch 与 ``fft-conv-pytorch`` 组合在 augmentation worker 中执行
``GaussianBlurTransform`` 的 FFT benchmark 时会触发 native 内存损坏（进程崩溃或非法内存
访问）。修复只把 ``benchmark`` 置为 ``False`` 并清空其缓存，使该变换恒定走普通
convolution 路径——**模糊本身的概率与 sigma 完全不变**，因此不构成对增强策略的修改，也不需要
在论文中作为变量报告。

fail-closed
-----------
若在 nnU-Net 升级后找不到唯一的 ``GaussianBlurTransform``，本 mixin 会直接报错而不是静默
启动：静默启动意味着重新执行不稳定的 FFT 试跑，而崩溃发生在 worker 进程里，极难定位。
"""

from __future__ import annotations

from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class NoFFTAugmentationMixin:
    """保留 nnU-Net 默认增强语义，仅关闭 GaussianBlurTransform 的 FFT benchmark。"""

    @staticmethod
    def get_training_transforms(*args, **kwargs):
        transforms = nnUNetTrainer.get_training_transforms(*args, **kwargs)
        disabled = 0
        for transform in transforms.transforms:
            candidate = getattr(transform, "transform", transform)
            if isinstance(candidate, GaussianBlurTransform):
                candidate.benchmark = False
                candidate.benchmark_use_fft.clear()
                disabled += 1
        if disabled != 1:
            raise RuntimeError(
                "nnU-Net v2.6.2 augmentation pipeline 未找到唯一的 GaussianBlurTransform；"
                "拒绝静默启动，以免重新调用不稳定的 fft_conv benchmark。"
            )
        return transforms


__all__ = ("NoFFTAugmentationMixin",)
