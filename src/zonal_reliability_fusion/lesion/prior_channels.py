"""解剖先验作为**附加输入通道**时的增强边界（新主线的活跃实现）。

为什么需要这一层
----------------
nnU-Net 的默认增强管线对 ``data_dict['image']`` 的**所有通道**统一施加强度变换。但当输入
里带有 ``P(WG) / P(PZ) / P(TZ)`` 这类 soft probability 通道时，对它们施加
noise / blur / brightness / contrast / gamma 会把概率图扭曲成非法输入（越界、失去概率语义），
进而让「解剖先验错误是真实部署链条的一部分」这一前提被污染成「先验错误是增强伪影」。

因此：

- **空间变换**（rotation / scaling / elastic / mirror）必须**同步**作用于 MRI 与先验通道，
  保证空间对齐——它们不被包装；
- **强度变换**只作用于前 ``num_mri_channels`` 个通道，先验通道原样保留。

与 legacy 实现的关系
--------------------
旧研究线在 ``legacy.prior_transforms`` 里有一份同语义的实现（为旧 Dataset606 的 PZ/TZ
输入通道而写）。这里保留**同一行为契约**（强度只作用于 MRI、空间同步作用于全部通道），
但把活跃实现与调用点放在新主线的包里，使 «新方法不依赖 legacy 模块» 这一点在代码结构上成立。
legacy 模块继续存在，仅为 checkpoint 兼容与历史复现。
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

#: 新主线的冻结 MRI 通道数（T2W / ADC / HBV）
MRI_CHANNELS = 3

#: nnU-Net 默认管线中会改动**强度**的变换类型；先验通道必须豁免
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

    内部变换仍是标准 batchgeneratorsv2 ``BasicTransform``；本包装只在调用前把 ``image``
    切成 MRI 部分、调用后与未修改的先验部分拼回。空间/镜像变换不应被本类包装。
    """

    def __init__(
        self, transform: BasicTransform, mri_channels: int = MRI_CHANNELS
    ) -> None:
        super().__init__()
        self.transform = transform
        self.mri_channels = int(mri_channels)

    def __call__(self, **data_dict) -> dict:
        image = data_dict.get("image")
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
        return (
            f"{type(self).__name__}(mri_channels={self.mri_channels}, "
            f"transform={self.transform})"
        )


def restrict_intensity_transforms_to_mri(
    transforms, mri_channels: int = MRI_CHANNELS
) -> tuple[object, int]:
    """就地包装 ``ComposeTransforms`` 中的全部强度变换，使其只作用于 MRI 通道。

    返回 ``(transforms, n_wrapped)``；调用方**必须**校验 ``n_wrapped > 0``，否则在 nnU-Net
    改变管线结构时会静默地把强度增强施加到概率通道上（fail-closed 要求显式失败）。
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


__all__ = (
    "INTENSITY_TRANSFORM_TYPES",
    "MRI_CHANNELS",
    "MRIChannelRestrictedTransform",
    "restrict_intensity_transforms_to_mri",
)
