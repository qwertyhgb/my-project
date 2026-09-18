"""PI-CAI official-style Focal + CE v2 port 的纯合成 CPU 测试。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import numpy as np
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NNUNET_SOURCE = PROJECT_ROOT / "third_party" / "nnUNet"
if str(NNUNET_SOURCE) not in sys.path:
    sys.path.insert(0, str(NNUNET_SOURCE))

from zonal_reliability_fusion.integrations.nnunet_picai_flce import (  # noqa: E402
    PiCAIFocalCrossEntropyLoss,
    nnUNetTrainerPICAI_FLCE,
    nnUNetTrainerPICAI_FLCE_NoFFT,
)
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer  # noqa: E402


def _official_formula(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """逐项复算 PI-CAI v1 公开实现的 alpha=None / gamma=2 公式。"""
    probabilities = torch.softmax(logits, dim=1)
    labels = target[:, 0].long()
    one_hot = F.one_hot(labels, num_classes=logits.shape[1]).movedim(-1, 1).float()
    smooth = 1e-5
    one_hot = one_hot.clamp(min=smooth / (logits.shape[1] - 1), max=1.0 - smooth)
    pt = (one_hot * probabilities).sum(dim=1) + smooth
    focal = (-(1.0 - pt).pow(2.0) * pt.log()).mean()
    ce = F.cross_entropy(logits, labels)
    return 0.5 * focal + 0.5 * ce


def test_pi_cai_flce_matches_public_formula_and_backpropagates():
    torch.manual_seed(17)
    logits = torch.randn(2, 2, 3, 4, 5, requires_grad=True)
    target = torch.randint(0, 2, (2, 1, 3, 4, 5))
    loss = PiCAIFocalCrossEntropyLoss()(logits, target)
    assert torch.allclose(loss, _official_formula(logits, target), atol=1e-7, rtol=1e-6)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_pi_cai_flce_accepts_target_without_channel_dimension():
    logits = torch.tensor([[[[2.0, -1.0]], [[-2.0, 1.0]]]], requires_grad=True)
    target = torch.tensor([[[0, 1]]])
    loss = PiCAIFocalCrossEntropyLoss()(logits, target)
    assert torch.isfinite(loss)


@pytest.mark.parametrize(
    "bad_target",
    [torch.tensor([[[[-1]]]]), torch.tensor([[[[2]]]]), torch.tensor([[[[0.5]]]])],
)
def test_pi_cai_flce_rejects_invalid_or_ignore_targets(bad_target):
    logits = torch.zeros(1, 2, 1, 1)
    with pytest.raises(ValueError):
        PiCAIFocalCrossEntropyLoss()(logits, bad_target)


def test_picai_trainer_is_v2_subclass():
    assert issubclass(nnUNetTrainerPICAI_FLCE, nnUNetTrainer)
    assert issubclass(nnUNetTrainerPICAI_FLCE_NoFFT, nnUNetTrainerPICAI_FLCE)


def test_picai_trainer_keeps_blur_but_disables_unsafe_fft_benchmark():
    """构建默认训练增强不读真实数据；仅断言 Gaussian blur backend 被安全固定。"""
    transforms = nnUNetTrainerPICAI_FLCE.get_training_transforms(
        patch_size=np.array([8, 16, 16]),
        rotation_for_DA=(-0.1, 0.1),
        deep_supervision_scales=[[1, 1, 1], [0.5, 0.5, 0.5]],
        mirror_axes=(0, 1, 2),
        do_dummy_2d_data_aug=False,
        use_mask_for_norm=[False, False, False],
        is_cascaded=False,
        foreground_labels=(1,),
    )
    blur = [
        getattr(transform, "transform", transform)
        for transform in transforms.transforms
        if isinstance(getattr(transform, "transform", transform), GaussianBlurTransform)
    ]
    assert len(blur) == 1
    assert blur[0].benchmark is False
