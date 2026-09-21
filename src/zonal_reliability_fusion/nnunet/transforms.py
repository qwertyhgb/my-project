"""anatomy variant 的 PZ/TZ 增强边界：强度增强只作用于 MRI，空间增强作用于全部通道。

nnU-Net 的默认增强管线（``nnUNetTrainer.get_training_transforms``）对 ``data_dict['image']``
的**所有通道**同时施加变换。Dataset606 有 5 个通道（T2W/ADC/HBV/PZ/TZ），其中：

- **空间变换**（SpatialTransform 的 rotation/scaling/elastic、MirrorTransform）必须同时作用于
  MRI 与 PZ/TZ，保证空间对齐 —— 这些变换**不包装**，原样作用于全部通道即可满足要求；
- **强度变换**（noise/blur/brightness/contrast/low-resolution/gamma）**禁止**作用于 PZ/TZ。

batchgeneratorsv2 0.3.0 的强度变换里只有 ``SimulateLowResolutionTransform`` 提供
``allowed_channels``，其余没有。为统一且不重实现管线，这里提供一个很小的通道包装 transform：
只把前 ``mri_channels`` 个通道交给内部强度变换，再与未被触碰的 prior 通道拼回原 tensor。
"""

from __future__ import annotations

import torch
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.intensity.brightness import (
    MultiplicativeBrightnessTransform,
)
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import (
    SimulateLowResolutionTransform,
)
from batchgeneratorsv2.transforms.utils.random import RandomTransform

from zonal_reliability_fusion.nnunet.networks import MRI_CHANNELS

#: nnU-Net 默认管线中会改动强度的变换类型（PZ/TZ 必须豁免）
INTENSITY_TRANSFORM_TYPES = (
    GaussianNoiseTransform,
    GaussianBlurTransform,
    MultiplicativeBrightnessTransform,
    ContrastTransform,
    SimulateLowResolutionTransform,
    GammaTransform,
)


class MRIChannelRestrictedTransform(BasicTransform):
    """把一个强度变换限制在前 ``mri_channels`` 个通道上，其余通道原样保留。

    内部变换仍是一个标准的 batchgeneratorsv2 ``BasicTransform``；本包装只在调用前把
    ``image`` 切成 MRI 部分交给它，调用后再与未修改的 prior 部分拼回。空间/镜像等变换
    不应被本类包装（它们需要同步作用于全部通道）。
    """

    def __init__(
        self, transform: BasicTransform, mri_channels: int = MRI_CHANNELS
    ) -> None:
        super().__init__()
        self.transform = transform
        self.mri_channels = int(mri_channels)

    def __call__(self, **data_dict) -> dict:
        image = data_dict.get("image")
        # 没有 image，或通道数不超过 MRI 通道数：无需限制，直接透传
        if image is None or image.shape[0] <= self.mri_channels:
            return self.transform(**data_dict)
        mri = image[: self.mri_channels]
        prior = image[self.mri_channels :]
        sub = dict(data_dict)
        sub["image"] = mri
        sub = self.transform(**sub)
        data_dict["image"] = torch.cat([sub["image"], prior], dim=0)
        return data_dict

    def __repr__(self) -> str:
        return f"{type(self).__name__}(mri_channels={self.mri_channels}, transform={self.transform})"


def restrict_intensity_transforms_to_mri(transforms, mri_channels: int = MRI_CHANNELS):
    """就地包装一个 ``ComposeTransforms`` 里的全部强度变换，使其只作用于 MRI 通道。

    - 直接出现的强度变换、以及被 ``RandomTransform`` 包裹的强度变换都会被包装；
    - 空间/镜像/deep-supervision/label 等非强度变换保持不变（继续同步作用于全部通道）；
    - 返回 ``(transforms, n_wrapped)``；调用方应校验 ``n_wrapped > 0``，以便在 nnU-Net
      改变管线结构时 fail-closed，而不是静默地把强度增强施加到 PZ/TZ 上。
    """
    n_wrapped = 0
    new_list = []
    for transform in transforms.transforms:
        if isinstance(transform, RandomTransform) and isinstance(
            transform.transform, INTENSITY_TRANSFORM_TYPES
        ):
            transform.transform = MRIChannelRestrictedTransform(
                transform.transform, mri_channels
            )
            n_wrapped += 1
        elif isinstance(transform, INTENSITY_TRANSFORM_TYPES):
            transform = MRIChannelRestrictedTransform(transform, mri_channels)
            n_wrapped += 1
        new_list.append(transform)
    transforms.transforms = new_list
    return transforms, n_wrapped
