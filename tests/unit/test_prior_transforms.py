"""PZ/TZ 增强边界的纯合成 CPU 测试（不读取真实数据）。

覆盖：MRI 强度增强不会改变 PZ/TZ（#12）、spatial transform 同步作用于 MRI/PZ/TZ（#13），
以及 anatomy Trainer 的训练增强确实只把强度变换限制到 MRI 通道、空间/镜像变换不包装。
"""

from __future__ import annotations

import numpy as np
import torch
from batchgeneratorsv2.transforms.intensity.brightness import (
    MultiplicativeBrightnessTransform,
)
from batchgeneratorsv2.transforms.intensity.contrast import BGContrast
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.random import RandomTransform

from zonal_reliability_fusion.nnunet.trainers import nnUNetTrainerPICAI_AnatomyGate
from zonal_reliability_fusion.nnunet.transforms import (
    INTENSITY_TRANSFORM_TYPES,
    MRIChannelRestrictedTransform,
    restrict_intensity_transforms_to_mri,
)


def _five_channel_image() -> torch.Tensor:
    """5 通道合成图：前 3 为 MRI（随机），后 2 为 PZ/TZ（[0,1] 常量图案）。"""
    torch.manual_seed(11)
    img = torch.zeros(5, 8, 16, 16)
    img[:3] = torch.randn(3, 8, 16, 16)
    img[3] = 0.37  # PZ
    img[4] = 0.62  # TZ
    return img


# --------------------------------------------------------------------------- #12 强度增强不动 PZ/TZ
def test_intensity_restricted_to_mri_leaves_prior_untouched():
    torch.manual_seed(5)
    inner = GaussianNoiseTransform(
        noise_variance=(0.1, 0.1), p_per_channel=1.0, synchronize_channels=False
    )
    wrapped = MRIChannelRestrictedTransform(inner, mri_channels=3)

    img = _five_channel_image()
    img_before = img.clone()
    out = wrapped(image=img)["image"]

    # PZ/TZ 逐值不变
    assert torch.equal(out[3:], img_before[3:])
    # MRI 确实被强度增强改动
    assert not torch.equal(out[:3], img_before[:3])


def test_brightness_multiplier_only_scales_mri():
    inner = MultiplicativeBrightnessTransform(
        multiplier_range=BGContrast((2.0, 2.0)),
        synchronize_channels=True,
        p_per_channel=1.0,
    )
    wrapped = MRIChannelRestrictedTransform(inner, mri_channels=3)

    img = _five_channel_image()
    mri_before = img[:3].clone()
    prior_before = img[3:].clone()
    out = wrapped(image=img)["image"]

    assert torch.equal(out[3:], prior_before)  # prior 完全不变
    assert torch.allclose(out[:3], mri_before * 2.0, atol=1e-5)  # MRI 被乘以 2


def test_restrict_helper_wraps_every_intensity_transform():
    compose = ComposeTransforms(
        [
            SpatialTransform(
                (8, 16, 16),
                patch_center_dist_from_border=0,
                random_crop=False,
                p_rotation=0.0,
                p_scaling=0.0,
                p_elastic_deform=0.0,
            ),
            RandomTransform(
                GaussianNoiseTransform((0, 0.1), 1.0, True), apply_probability=1.0
            ),
            MultiplicativeBrightnessTransform(BGContrast((0.75, 1.25)), True, 1.0),
            MirrorTransform(allowed_axes=(0, 1, 2)),
        ]
    )
    _, n_wrapped = restrict_intensity_transforms_to_mri(compose, mri_channels=3)
    assert n_wrapped == 2  # noise（RandomTransform 内）+ brightness（裸变换）
    # 空间/镜像未被包装
    assert isinstance(compose.transforms[0], SpatialTransform)
    assert isinstance(compose.transforms[3], MirrorTransform)


# --------------------------------------------------------------------------- #13 空间变换同步作用于全部通道
def test_spatial_transform_moves_mri_and_prior_together():
    torch.manual_seed(3)
    spatial = SpatialTransform(
        (8, 16, 16),
        patch_center_dist_from_border=0,
        random_crop=False,
        p_elastic_deform=0.0,
        p_rotation=1.0,
        rotation=(0.5, 0.5),  # 固定非平凡角度，保证确实发生几何变换
        p_scaling=0.0,
        p_synchronize_scaling_across_axes=1,
        bg_style_seg_sampling=False,
    )
    base = torch.randn(8, 16, 16)
    img = base.unsqueeze(0).repeat(5, 1, 1, 1)  # 5 个完全相同的通道
    seg = torch.zeros(1, 8, 16, 16, dtype=torch.int16)

    out = spatial(image=img, segmentation=seg)["image"]

    # 相同的几何变换作用于所有通道：MRI 与 PZ/TZ 变换后仍逐值一致（空间对齐）
    for c in range(1, 5):
        assert torch.allclose(out[c], out[0], atol=1e-6)
    # 且确实发生了空间变换（不是恒等）
    assert not torch.allclose(out[0], img[0], atol=1e-4)


def test_mirror_transform_flips_all_channels_together():
    mirror = MirrorTransform(allowed_axes=(2,))
    img = _five_channel_image()
    img[:3] = (
        img[3:].sum(0, keepdim=True).repeat(3, 1, 1, 1)
    )  # 让所有通道相同，便于验证同步
    img[3] = img[0]
    img[4] = img[0]
    out = mirror(image=img, segmentation=torch.zeros(1, 8, 16, 16, dtype=torch.int16))[
        "image"
    ]
    for c in range(1, 5):
        assert torch.equal(out[c], out[0])


# --------------------------------------------------------------------------- anatomy Trainer 集成
def test_anatomy_training_transforms_restrict_intensity_but_not_spatial():
    transforms = nnUNetTrainerPICAI_AnatomyGate.get_training_transforms(
        patch_size=np.array([8, 16, 16]),
        rotation_for_DA=(-0.1, 0.1),
        deep_supervision_scales=[[1, 1, 1], [0.5, 0.5, 0.5]],
        mirror_axes=(0, 1, 2),
        do_dummy_2d_data_aug=False,
        use_mask_for_norm=[False, False, False, False, False],
        is_cascaded=False,
        foreground_labels=(1,),
    )
    n_wrapped = 0
    for t in transforms.transforms:
        inner = getattr(t, "transform", t)
        if isinstance(inner, MRIChannelRestrictedTransform):
            n_wrapped += 1
            assert isinstance(inner.transform, INTENSITY_TRANSFORM_TYPES)
        # 空间/镜像变换必须保持原样（不被通道限制包装），才能同步作用于 MRI/PZ/TZ
        if isinstance(inner, (SpatialTransform, MirrorTransform)):
            assert not isinstance(t, MRIChannelRestrictedTransform)
            assert not isinstance(inner, MRIChannelRestrictedTransform)
    # noise/blur/brightness/contrast/low-res/gamma(x2) 共 7 个强度变换应全部被限制到 MRI
    assert n_wrapped >= 6
