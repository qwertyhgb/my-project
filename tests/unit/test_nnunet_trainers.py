"""Trainer 层的纯合成 CPU 测试（不读取真实数据、不实例化 Trainer、不写 outputs/）。

覆盖：PI-CAI Focal+CE 与公开公式逐值一致（#1）、baseline 是 nnUNetTrainer 子类（#2）、
NoFFT 保留 Gaussian blur 但 benchmark=False（#3）、网络 builder 可被 nnU-Net inference 使用
（#14）、三个 Trainer 类名互不相同（#15）、train_nnunet.py --help 成功（#16）。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

from zonal_reliability_fusion.nnunet.trainers import (
    PiCAIFocalCrossEntropyLoss,
    nnUNetTrainerPICAI_AnatomyGate,
    nnUNetTrainerPICAI_FLCE_NoFFT,
    nnUNetTrainerPICAI_ImageGate,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train" / "train_nnunet.py"


# --------------------------------------------------------------------------- #1 损失与公开公式一致
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
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


@pytest.mark.parametrize(
    "bad_target",
    [torch.tensor([[[[-1]]]]), torch.tensor([[[[2]]]]), torch.tensor([[[[0.5]]]])],
)
def test_pi_cai_flce_rejects_invalid_or_ignore_targets(bad_target):
    with pytest.raises(ValueError):
        PiCAIFocalCrossEntropyLoss()(torch.zeros(1, 2, 1, 1), bad_target)


# --------------------------------------------------------------------------- #2 继承关系
def test_all_trainers_are_nnunettrainer_subclasses():
    for trainer in (
        nnUNetTrainerPICAI_FLCE_NoFFT,
        nnUNetTrainerPICAI_ImageGate,
        nnUNetTrainerPICAI_AnatomyGate,
    ):
        assert issubclass(trainer, nnUNetTrainer)
    # gate variant 复用 baseline 的损失与 NoFFT 增强
    assert issubclass(nnUNetTrainerPICAI_ImageGate, nnUNetTrainerPICAI_FLCE_NoFFT)
    assert issubclass(nnUNetTrainerPICAI_AnatomyGate, nnUNetTrainerPICAI_FLCE_NoFFT)


# --------------------------------------------------------------------------- #3 NoFFT 修复
def test_nofft_keeps_blur_but_disables_fft_benchmark():
    transforms = nnUNetTrainerPICAI_FLCE_NoFFT.get_training_transforms(
        patch_size=np.array([8, 16, 16]),
        rotation_for_DA=(-0.1, 0.1),
        deep_supervision_scales=[[1, 1, 1], [0.5, 0.5, 0.5]],
        mirror_axes=(0, 1, 2),
        do_dummy_2d_data_aug=False,
        use_mask_for_norm=[False, False, False],
        is_cascaded=False,
        foreground_labels=(1,),
    )
    blurs = [
        getattr(t, "transform", t)
        for t in transforms.transforms
        if isinstance(getattr(t, "transform", t), GaussianBlurTransform)
    ]
    assert len(blurs) == 1  # blur 仍保留
    assert blurs[0].benchmark is False  # 但 FFT benchmark 被关闭


# --------------------------------------------------------------------------- #14 builder 可被 inference 使用
@pytest.mark.parametrize(
    "trainer,in_channels",
    [
        (nnUNetTrainerPICAI_FLCE_NoFFT, 3),
        (nnUNetTrainerPICAI_ImageGate, 3),
        (nnUNetTrainerPICAI_AnatomyGate, 5),
    ],
)
def test_build_network_architecture_inference_mode(
    synthetic_arch, trainer, in_channels
):
    """inference 时 deep_supervision=False，builder 必须返回 forward 输出单张量的网络。"""
    net = trainer.build_network_architecture(
        synthetic_arch["architecture_class_name"],
        synthetic_arch["arch_init_kwargs"],
        synthetic_arch["arch_init_kwargs_req_import"],
        in_channels,
        2,
        False,  # inference: deep supervision disabled
    )
    net.eval()
    with torch.no_grad():
        out = net(torch.randn(1, in_channels, 8, 16, 16))
    assert isinstance(out, torch.Tensor)
    assert out.shape[1] == 2


# --------------------------------------------------------------------------- #15 三个类名互不相同
def test_trainer_class_names_are_distinct():
    names = {
        nnUNetTrainerPICAI_FLCE_NoFFT.__name__,
        nnUNetTrainerPICAI_ImageGate.__name__,
        nnUNetTrainerPICAI_AnatomyGate.__name__,
    }
    assert len(names) == 3
    # baseline 类名必须与已完成的 N0 输出目录一致（不得改名导致无法续训/验证）
    assert nnUNetTrainerPICAI_FLCE_NoFFT.__name__ == "nnUNetTrainerPICAI_FLCE_NoFFT"


# --------------------------------------------------------------------------- #16 train_nnunet.py --help
def _load_train_entry():
    spec = importlib.util.spec_from_file_location("train_nnunet_entry", TRAIN_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_train_entry_help_succeeds():
    module = _load_train_entry()
    with pytest.raises(SystemExit) as exc:
        module.build_parser().parse_args(["--help"])
    assert exc.value.code == 0


def test_train_entry_variant_mapping_is_complete_and_distinct():
    module = _load_train_entry()
    assert set(module.VARIANT_TO_TRAINER) == {"baseline", "image_gate", "anatomy_gate"}
    assert len(set(module.VARIANT_TO_TRAINER.values())) == 3
    assert module.resolve_trainer_class("baseline") is nnUNetTrainerPICAI_FLCE_NoFFT
    assert module.resolve_trainer_class("image_gate") is nnUNetTrainerPICAI_ImageGate
    assert (
        module.resolve_trainer_class("anatomy_gate") is nnUNetTrainerPICAI_AnatomyGate
    )
