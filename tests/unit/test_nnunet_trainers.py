"""Trainer 层的纯合成 CPU 测试（不读取真实数据、不实例化 Trainer、不写 outputs/）。

覆盖：PI-CAI Focal+CE 与公开公式逐值一致、Trainer 是 nnUNetTrainer 子类、NoFFT 保留
Gaussian blur 但 benchmark=False、网络 builder 可被 nnU-Net inference 使用、五个 Trainer
类名互不相同、train_nnunet.py --help 成功。

`optimized_baseline`（``nnUNetTrainerPICAI_DiceCE_NoFFT``）另需验证三件事：
它不继承 Focal+CE mixin / FLCE baseline / gate 基类；它的 ``_build_loss`` **就是**原生
``nnUNetTrainer._build_loss``；用它构建的网络不含 gate。

`positive_sampling`（``nnUNetTrainerPICAI_FLCE_PositiveSampling_NoFFT``）的采样行为、
MRO 与 mixin 接线在 ``test_positive_case_sampling.py`` 单独覆盖。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from torch import nn

from zonal_reliability_fusion.nnunet.networks import (
    FEATURE_STEM_CHANNELS,
    FeatureFusionNNUNet,
    GatedNNUNet,
)
from zonal_reliability_fusion.nnunet.sampling import PositiveCaseSamplingMixin
from zonal_reliability_fusion.nnunet.trainers import (
    NoFFTAugmentationMixin,
    PiCAIFocalCrossEntropyLoss,
    PICAIFocalCrossEntropyLossMixin,
    _FeatureFusionTrainerBase,
    _GatedTrainerBase,
    nnUNetTrainerPICAI_AnatomyGate,
    nnUNetTrainerPICAI_AnatomyGate_PositiveSampling_NoFFT,
    nnUNetTrainerPICAI_DiceCE_NoFFT,
    nnUNetTrainerPICAI_FeatureAnatomyGate_PositiveSampling_NoFFT,
    nnUNetTrainerPICAI_FeatureImageGate_PositiveSampling_NoFFT,
    nnUNetTrainerPICAI_FeatureNoGate_PositiveSampling_NoFFT,
    nnUNetTrainerPICAI_FLCE_NoFFT,
    nnUNetTrainerPICAI_FLCE_PositiveSampling_NoFFT,
    nnUNetTrainerPICAI_ImageGate,
    nnUNetTrainerPICAI_ImageGate_PositiveSampling_NoFFT,
)
from zonal_reliability_fusion.nnunet.transforms import MRIChannelRestrictedTransform

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
_ALL_PROJECT_TRAINERS = (
    nnUNetTrainerPICAI_FLCE_NoFFT,
    nnUNetTrainerPICAI_DiceCE_NoFFT,
    nnUNetTrainerPICAI_ImageGate,
    nnUNetTrainerPICAI_AnatomyGate,
    nnUNetTrainerPICAI_FLCE_PositiveSampling_NoFFT,
    nnUNetTrainerPICAI_ImageGate_PositiveSampling_NoFFT,
    nnUNetTrainerPICAI_AnatomyGate_PositiveSampling_NoFFT,
    nnUNetTrainerPICAI_FeatureNoGate_PositiveSampling_NoFFT,
    nnUNetTrainerPICAI_FeatureImageGate_PositiveSampling_NoFFT,
    nnUNetTrainerPICAI_FeatureAnatomyGate_PositiveSampling_NoFFT,
)


def test_all_trainers_are_nnunettrainer_subclasses():
    for trainer in _ALL_PROJECT_TRAINERS:
        assert issubclass(trainer, nnUNetTrainer)
    # gate variant 复用 FLCE baseline 的损失与 NoFFT 增强
    assert issubclass(nnUNetTrainerPICAI_ImageGate, nnUNetTrainerPICAI_FLCE_NoFFT)
    assert issubclass(nnUNetTrainerPICAI_AnatomyGate, nnUNetTrainerPICAI_FLCE_NoFFT)
    # positive_sampling 在 FLCE baseline 之上只叠加采样 mixin
    assert issubclass(
        nnUNetTrainerPICAI_FLCE_PositiveSampling_NoFFT, nnUNetTrainerPICAI_FLCE_NoFFT
    )
    # 两个组合 Trainer：gate（结构）+ 阳性采样 mixin（公平匹配 RQ1/RQ2）
    assert issubclass(
        nnUNetTrainerPICAI_ImageGate_PositiveSampling_NoFFT,
        nnUNetTrainerPICAI_ImageGate,
    )
    assert issubclass(
        nnUNetTrainerPICAI_AnatomyGate_PositiveSampling_NoFFT,
        nnUNetTrainerPICAI_AnatomyGate,
    )
    for cls in (
        nnUNetTrainerPICAI_ImageGate_PositiveSampling_NoFFT,
        nnUNetTrainerPICAI_AnatomyGate_PositiveSampling_NoFFT,
    ):
        assert issubclass(cls, PositiveCaseSamplingMixin)
        assert issubclass(cls, nnUNetTrainerPICAI_FLCE_NoFFT)


def test_optimized_baseline_inherits_only_native_trainer_and_nofft():
    """DiceCE baseline 必须绕开 Focal+CE mixin 与 gate 基类，否则损失会被错误覆盖。"""
    assert issubclass(nnUNetTrainerPICAI_DiceCE_NoFFT, NoFFTAugmentationMixin)
    assert issubclass(nnUNetTrainerPICAI_DiceCE_NoFFT, nnUNetTrainer)

    for forbidden in (
        PICAIFocalCrossEntropyLossMixin,
        nnUNetTrainerPICAI_FLCE_NoFFT,
        _GatedTrainerBase,
        nnUNetTrainerPICAI_ImageGate,
        nnUNetTrainerPICAI_AnatomyGate,
    ):
        assert not issubclass(nnUNetTrainerPICAI_DiceCE_NoFFT, forbidden), (
            f"optimized_baseline 不得继承 {forbidden.__name__}"
        )

    # MRO 中也不得出现（issubclass 为真的更强表述）
    mro = nnUNetTrainerPICAI_DiceCE_NoFFT.__mro__
    assert PICAIFocalCrossEntropyLossMixin not in mro
    assert _GatedTrainerBase not in mro
    assert nnUNetTrainerPICAI_FLCE_NoFFT not in mro


# ------------------------------------------------------- 原生 Dice+CE 损失构建（optimized_baseline）
class _MinimalLossBuilderState:
    """只提供 ``nnUNetTrainer._build_loss`` 实际读取的属性，不实例化任何真实 Trainer。

    真实 ``nnUNetTrainer.__init__`` 会创建 output folder、初始化 logger 与 dataloader，
    测试中禁止触发。这里把原生 ``_build_loss`` 当作普通函数调用（显式传入本替身作为 self）。
    """

    def __init__(self, *, enable_deep_supervision: bool, num_scales: int = 6) -> None:
        self.label_manager = SimpleNamespace(has_regions=False, ignore_label=None)
        self.configuration_manager = SimpleNamespace(batch_dice=False)
        self.is_ddp = False
        self.enable_deep_supervision = enable_deep_supervision
        self._num_scales = num_scales

    def _do_i_compile(self) -> bool:
        return False

    def _get_deep_supervision_scales(self) -> list:
        return [[1.0, 1.0, 1.0] for _ in range(self._num_scales)]


def _native_base_loss(loss: nn.Module) -> nn.Module:
    """剥掉原生 ``DeepSupervisionWrapper``（若存在），返回基础损失。"""
    if isinstance(loss, DeepSupervisionWrapper):
        return loss.loss
    return loss


def test_optimized_baseline_build_loss_is_native_unbound_function():
    """不覆盖 ``_build_loss``：MRO 必须直接解析到 ``nnUNetTrainer._build_loss``。"""
    assert nnUNetTrainerPICAI_DiceCE_NoFFT._build_loss is nnUNetTrainer._build_loss
    # 对照：FLCE baseline 走的是 mixin 版本，二者不是同一个函数
    assert nnUNetTrainerPICAI_FLCE_NoFFT._build_loss is not nnUNetTrainer._build_loss


@pytest.mark.parametrize("enable_deep_supervision", [False, True])
def test_optimized_baseline_build_loss_is_native_dice_plus_ce(enable_deep_supervision):
    """实际调用 ``_build_loss()``：基础损失必须是原生 Dice+CE，绝不是 PiCAIFocalCrossEntropyLoss。"""
    builder = nnUNetTrainerPICAI_DiceCE_NoFFT._build_loss
    loss = builder(
        _MinimalLossBuilderState(enable_deep_supervision=enable_deep_supervision)
    )

    if enable_deep_supervision:
        # 外层允许是原生 DeepSupervisionWrapper（权重由原生代码算出）
        assert isinstance(loss, DeepSupervisionWrapper)
        expected = np.array([1.0 / (2**i) for i in range(6)])
        expected[-1] = 0.0  # 最低分辨率不监督（原生默认行为）
        expected = expected / expected.sum()
        assert np.allclose(loss.weight_factors, expected)

    base = _native_base_loss(loss)
    assert isinstance(base, DC_and_CE_loss)
    # 全部来自 nnunetv2 原生模块，而非项目自写实现
    assert type(base).__module__.startswith("nnunetv2.")
    assert isinstance(base.dc, MemoryEfficientSoftDiceLoss)
    assert isinstance(base.ce, RobustCrossEntropyLoss)
    assert isinstance(base.ce, nn.CrossEntropyLoss)
    assert base.weight_dice == 1 and base.weight_ce == 1
    assert base.dc.do_bg is False  # plans 的 do_bg=False（排除背景类）
    assert not isinstance(base, PiCAIFocalCrossEntropyLoss)
    assert not isinstance(loss, PiCAIFocalCrossEntropyLoss)


# --------------------------------------------------------------------------- #3 NoFFT 修复
@pytest.mark.parametrize(
    "trainer",
    [
        nnUNetTrainerPICAI_FLCE_NoFFT,
        nnUNetTrainerPICAI_DiceCE_NoFFT,
        nnUNetTrainerPICAI_FLCE_PositiveSampling_NoFFT,
        nnUNetTrainerPICAI_FeatureNoGate_PositiveSampling_NoFFT,
        nnUNetTrainerPICAI_FeatureImageGate_PositiveSampling_NoFFT,
    ],
)
def test_nofft_keeps_blur_but_disables_fft_benchmark(trainer):
    transforms = trainer.get_training_transforms(
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
    assert len(blurs) == 1  # blur 仍保留（没有被删除）
    assert blurs[0].benchmark is False  # 但 FFT benchmark 被关闭


@pytest.mark.parametrize(
    "trainer",
    [
        nnUNetTrainerPICAI_ImageGate_PositiveSampling_NoFFT,
        nnUNetTrainerPICAI_AnatomyGate_PositiveSampling_NoFFT,
        nnUNetTrainerPICAI_FeatureAnatomyGate_PositiveSampling_NoFFT,
    ],
)
def test_combined_trainers_keep_nofft_and_limit_intensity_to_mri(trainer):
    """组合 Trainer 的增强：NoFFT 仍生效；anatomy 的强度变换仍被包装到 MRI 通道。"""
    transforms = trainer.get_training_transforms(
        patch_size=np.array([8, 16, 16]),
        rotation_for_DA=(-0.1, 0.1),
        deep_supervision_scales=[[1, 1, 1], [0.5, 0.5, 0.5]],
        mirror_axes=(0, 1, 2),
        do_dummy_2d_data_aug=False,
        use_mask_for_norm=[False] * 5,
        is_cascaded=False,
        foreground_labels=(1,),
    )
    blurs = []
    n_restricted = 0
    for t in transforms.transforms:
        inner = getattr(t, "transform", t)
        if isinstance(
            inner, MRIChannelRestrictedTransform
        ):  # anatomy：强度变换被限制到 MRI
            n_restricted += 1
            inner = inner.transform
        if isinstance(inner, GaussianBlurTransform):
            blurs.append(inner)
    if trainer.expected_input_channels > 3:
        # anatomy 系（5 通道）：强度增强确实被限制到 MRI 通道（不是只声明不生效）
        assert n_restricted > 0
    else:
        # image 系（3 通道）没有 prior 通道需要保护，不应额外包装
        assert n_restricted == 0
    assert len(blurs) == 1  # blur 仍保留（没有被删除）
    assert blurs[0].benchmark is False  # 但 FFT benchmark 被关闭


# --------------------------------------------------------------------------- #14 builder 可被 inference 使用
@pytest.mark.parametrize(
    "trainer,in_channels",
    [
        (nnUNetTrainerPICAI_FLCE_NoFFT, 3),
        (nnUNetTrainerPICAI_DiceCE_NoFFT, 3),
        (nnUNetTrainerPICAI_ImageGate, 3),
        (nnUNetTrainerPICAI_AnatomyGate, 5),
        (nnUNetTrainerPICAI_FLCE_PositiveSampling_NoFFT, 3),
        (nnUNetTrainerPICAI_ImageGate_PositiveSampling_NoFFT, 3),
        (nnUNetTrainerPICAI_AnatomyGate_PositiveSampling_NoFFT, 5),
        (nnUNetTrainerPICAI_FeatureNoGate_PositiveSampling_NoFFT, 3),
        (nnUNetTrainerPICAI_FeatureImageGate_PositiveSampling_NoFFT, 3),
        (nnUNetTrainerPICAI_FeatureAnatomyGate_PositiveSampling_NoFFT, 5),
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


def test_optimized_baseline_builder_returns_plain_network_without_gate(synthetic_arch):
    """optimized_baseline 不覆盖 builder：必须得到原生 PlainConvUNet，且不含任何 gate。"""
    net = nnUNetTrainerPICAI_DiceCE_NoFFT.build_network_architecture(
        synthetic_arch["architecture_class_name"],
        synthetic_arch["arch_init_kwargs"],
        synthetic_arch["arch_init_kwargs_req_import"],
        3,
        2,
        False,
    )
    assert isinstance(net, PlainConvUNet)  # 原生类，未被包装
    assert not isinstance(net, GatedNNUNet)
    assert type(net).__module__.startswith("dynamic_network_architectures.")
    assert not hasattr(net, "gate")  # 无 gate 子模块
    assert not hasattr(net, "backbone")

    net.eval()
    with torch.no_grad():
        out = net(torch.randn(1, 3, 8, 16, 16))
    assert isinstance(out, torch.Tensor)
    assert out.shape[1] == 2


# --------------------------------------------------------------------------- #15 类名互不相同
def test_trainer_class_names_are_distinct():
    names = {t.__name__ for t in _ALL_PROJECT_TRAINERS}
    assert len(names) == 10  # nnU-Net 按类名隔离 output folder，重名会互相覆盖
    # 类名必须与各自已存在/预期的输出目录一致（不得改名导致无法续训/验证）
    assert nnUNetTrainerPICAI_FLCE_NoFFT.__name__ == "nnUNetTrainerPICAI_FLCE_NoFFT"
    assert nnUNetTrainerPICAI_DiceCE_NoFFT.__name__ == "nnUNetTrainerPICAI_DiceCE_NoFFT"
    assert (
        nnUNetTrainerPICAI_FLCE_PositiveSampling_NoFFT.__name__
        == "nnUNetTrainerPICAI_FLCE_PositiveSampling_NoFFT"
    )
    assert (
        nnUNetTrainerPICAI_ImageGate_PositiveSampling_NoFFT.__name__
        == "nnUNetTrainerPICAI_ImageGate_PositiveSampling_NoFFT"
    )
    assert (
        nnUNetTrainerPICAI_AnatomyGate_PositiveSampling_NoFFT.__name__
        == "nnUNetTrainerPICAI_AnatomyGate_PositiveSampling_NoFFT"
    )
    assert (
        nnUNetTrainerPICAI_FeatureNoGate_PositiveSampling_NoFFT.__name__
        == "nnUNetTrainerPICAI_FeatureNoGate_PositiveSampling_NoFFT"
    )
    assert (
        nnUNetTrainerPICAI_FeatureImageGate_PositiveSampling_NoFFT.__name__
        == "nnUNetTrainerPICAI_FeatureImageGate_PositiveSampling_NoFFT"
    )
    assert (
        nnUNetTrainerPICAI_FeatureAnatomyGate_PositiveSampling_NoFFT.__name__
        == "nnUNetTrainerPICAI_FeatureAnatomyGate_PositiveSampling_NoFFT"
    )
    # 输出目录由类名自然形成：十个类名互异 => 十个目录互异
    dirs = {f"{n}__nnUNetPlans__3d_fullres" for n in names}
    assert len(dirs) == 10


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
    # 有且仅有这十个 variant
    assert set(module.VARIANT_TO_TRAINER) == {
        "baseline",
        "optimized_baseline",
        "image_gate",
        "anatomy_gate",
        "positive_sampling",
        "image_gate_positive_sampling",
        "anatomy_gate_positive_sampling",
        "feature_no_gate_positive_sampling",
        "feature_image_gate_positive_sampling",
        "feature_anatomy_gate_positive_sampling",
    }
    assert len(set(module.VARIANT_TO_TRAINER.values())) == 10
    assert module.resolve_trainer_class("baseline") is nnUNetTrainerPICAI_FLCE_NoFFT
    assert (
        module.resolve_trainer_class("optimized_baseline")
        is nnUNetTrainerPICAI_DiceCE_NoFFT
    )
    assert module.resolve_trainer_class("image_gate") is nnUNetTrainerPICAI_ImageGate
    assert (
        module.resolve_trainer_class("anatomy_gate") is nnUNetTrainerPICAI_AnatomyGate
    )
    assert (
        module.resolve_trainer_class("positive_sampling")
        is nnUNetTrainerPICAI_FLCE_PositiveSampling_NoFFT
    )
    assert (
        module.resolve_trainer_class("image_gate_positive_sampling")
        is nnUNetTrainerPICAI_ImageGate_PositiveSampling_NoFFT
    )
    assert (
        module.resolve_trainer_class("anatomy_gate_positive_sampling")
        is nnUNetTrainerPICAI_AnatomyGate_PositiveSampling_NoFFT
    )
    assert (
        module.resolve_trainer_class("feature_no_gate_positive_sampling")
        is nnUNetTrainerPICAI_FeatureNoGate_PositiveSampling_NoFFT
    )
    assert (
        module.resolve_trainer_class("feature_image_gate_positive_sampling")
        is nnUNetTrainerPICAI_FeatureImageGate_PositiveSampling_NoFFT
    )
    assert (
        module.resolve_trainer_class("feature_anatomy_gate_positive_sampling")
        is nnUNetTrainerPICAI_FeatureAnatomyGate_PositiveSampling_NoFFT
    )


def test_project_trainers_registry_contains_new_trainer():
    from zonal_reliability_fusion.nnunet.trainers import (
        PROJECT_TRAINERS,
        resolve_trainer_class,
    )

    name = "nnUNetTrainerPICAI_FLCE_PositiveSampling_NoFFT"
    assert set(PROJECT_TRAINERS) == {
        "nnUNetTrainerPICAI_FLCE_NoFFT",
        "nnUNetTrainerPICAI_DiceCE_NoFFT",
        "nnUNetTrainerPICAI_ImageGate",
        "nnUNetTrainerPICAI_AnatomyGate",
        name,
        "nnUNetTrainerPICAI_ImageGate_PositiveSampling_NoFFT",
        "nnUNetTrainerPICAI_AnatomyGate_PositiveSampling_NoFFT",
        "nnUNetTrainerPICAI_FeatureNoGate_PositiveSampling_NoFFT",
        "nnUNetTrainerPICAI_FeatureImageGate_PositiveSampling_NoFFT",
        "nnUNetTrainerPICAI_FeatureAnatomyGate_PositiveSampling_NoFFT",
    }
    assert PROJECT_TRAINERS[name] is nnUNetTrainerPICAI_FLCE_PositiveSampling_NoFFT
    assert resolve_trainer_class(name) is nnUNetTrainerPICAI_FLCE_PositiveSampling_NoFFT
    for cls in _FEATURE_TRAINERS:
        assert resolve_trainer_class(cls.__name__) is cls
    for cls in (
        nnUNetTrainerPICAI_ImageGate_PositiveSampling_NoFFT,
        nnUNetTrainerPICAI_AnatomyGate_PositiveSampling_NoFFT,
    ):
        assert resolve_trainer_class(cls.__name__) is cls


# ------------------------------------ gate + positive_sampling 组合 Trainer（RQ1/RQ2 公平匹配）
@pytest.mark.parametrize(
    "cls,gate_cls",
    [
        (
            nnUNetTrainerPICAI_ImageGate_PositiveSampling_NoFFT,
            nnUNetTrainerPICAI_ImageGate,
        ),
        (
            nnUNetTrainerPICAI_AnatomyGate_PositiveSampling_NoFFT,
            nnUNetTrainerPICAI_AnatomyGate,
        ),
    ],
)
def test_combined_gate_trainers_exact_mro(cls, gate_cls):
    """MRO 必须精确等于设计值：采样 mixin 在最前，其余与对应旧 gate 一致。"""
    assert cls.__mro__ == (
        cls,
        PositiveCaseSamplingMixin,
        gate_cls,
        _GatedTrainerBase,
        nnUNetTrainerPICAI_FLCE_NoFFT,
        NoFFTAugmentationMixin,
        PICAIFocalCrossEntropyLossMixin,
        nnUNetTrainer,
        object,
    )


def test_combined_gate_trainers_reuse_gate_and_sampling_hooks():
    """组合 Trainer 必须分别复用对应 gate 的网络/增强钩子与采样 mixin。"""
    image_cls = nnUNetTrainerPICAI_ImageGate_PositiveSampling_NoFFT
    anatomy_cls = nnUNetTrainerPICAI_AnatomyGate_PositiveSampling_NoFFT

    for cls in (image_cls, anatomy_cls):
        assert cls.get_dataloaders is PositiveCaseSamplingMixin.get_dataloaders
        assert cls.positive_cases_per_batch == 1
        # optimizer / 训练步 / 验证步 / 验证 transform 继续走原生实现
        assert cls.configure_optimizers is nnUNetTrainer.configure_optimizers
        assert cls.train_step is nnUNetTrainer.train_step
        assert cls.validation_step is nnUNetTrainer.validation_step
        assert cls.get_validation_transforms is nnUNetTrainer.get_validation_transforms
    assert (
        image_cls.build_network_architecture
        is nnUNetTrainerPICAI_ImageGate.build_network_architecture
    )
    assert (
        anatomy_cls.build_network_architecture
        is nnUNetTrainerPICAI_AnatomyGate.build_network_architecture
    )
    assert (
        image_cls.get_training_transforms
        is NoFFTAugmentationMixin.get_training_transforms
    )
    assert (
        anatomy_cls.get_training_transforms
        is nnUNetTrainerPICAI_AnatomyGate.get_training_transforms
    )


@pytest.mark.parametrize(
    "cls",
    [
        nnUNetTrainerPICAI_ImageGate_PositiveSampling_NoFFT,
        nnUNetTrainerPICAI_AnatomyGate_PositiveSampling_NoFFT,
    ],
)
def test_combined_gate_trainers_build_loss_is_flce(cls):
    """损失必须仍是 PI-CAI FLCE（不是 Dice+CE），deep supervision 权重不变。"""
    assert cls._build_loss is PICAIFocalCrossEntropyLossMixin._build_loss
    assert cls._build_loss is nnUNetTrainerPICAI_FLCE_NoFFT._build_loss
    assert cls._build_loss is not nnUNetTrainer._build_loss

    loss = cls._build_loss(_MinimalLossBuilderState(enable_deep_supervision=True))
    assert isinstance(loss, DeepSupervisionWrapper)
    assert isinstance(loss.loss, PiCAIFocalCrossEntropyLoss)
    assert loss.loss.focal_weight == 0.5 and loss.loss.ce_weight == 0.5
    assert loss.loss.gamma == 2.0
    assert not isinstance(loss.loss, DC_and_CE_loss)


def _build_combined(cls, synthetic_arch, in_channels, deep_supervision=False):
    return cls.build_network_architecture(
        synthetic_arch["architecture_class_name"],
        synthetic_arch["arch_init_kwargs"],
        synthetic_arch["arch_init_kwargs_req_import"],
        in_channels,
        2,
        deep_supervision,
    )


def test_combined_image_gate_strictly_requires_three_channels(synthetic_arch):
    cls = nnUNetTrainerPICAI_ImageGate_PositiveSampling_NoFFT
    net = _build_combined(cls, synthetic_arch, 3)
    assert isinstance(net, GatedNNUNet)
    net.eval()
    with torch.no_grad():
        out = net(torch.randn(2, 3, 8, 16, 16))
    assert isinstance(out, torch.Tensor) and out.shape[1] == 2
    with pytest.raises(ValueError):
        net(torch.randn(1, 5, 8, 16, 16))
    with pytest.raises(ValueError):
        _build_combined(cls, synthetic_arch, 5)


def test_combined_anatomy_gate_strictly_requires_five_channels(synthetic_arch):
    cls = nnUNetTrainerPICAI_AnatomyGate_PositiveSampling_NoFFT
    net = _build_combined(cls, synthetic_arch, 5)
    assert isinstance(net, GatedNNUNet)
    assert net.mri_channels == 3 and net.num_prior_channels == 2
    net.eval()
    with torch.no_grad():
        out = net(torch.randn(2, 5, 8, 16, 16))
    assert isinstance(out, torch.Tensor) and out.shape[1] == 2
    for bad_channels in (3, 4, 6):
        with pytest.raises(ValueError):
            net(torch.randn(1, bad_channels, 8, 16, 16))
    with pytest.raises(ValueError):
        _build_combined(cls, synthetic_arch, 3)


def test_combined_anatomy_backbone_receives_only_mri(synthetic_arch):
    """anatomy backbone 仍只接收 3 个 MRI 通道：PZ/TZ 不进入分割骨干。"""
    cls = nnUNetTrainerPICAI_AnatomyGate_PositiveSampling_NoFFT
    net = _build_combined(cls, synthetic_arch, 5)
    net.eval()
    x = torch.randn(2, 5, 8, 16, 16)
    with torch.no_grad():
        # 零初始化：门控输出严格为 1，整体等价于 backbone 只吃前 3 个 MRI 通道
        out = net(x)
        reference = net.backbone(x[:, :3])
    assert torch.allclose(out, reference, atol=1e-6)
    with pytest.raises(RuntimeError):
        net.backbone(x)  # 原生 backbone 不接收 5 通道


def test_combined_anatomy_zero_init_scales_and_prior_clamp(synthetic_arch):
    """零初始化 scales≡1；compute_gate_scales 必须执行 prior clamp。

    注意：gate 内含 InstanceNorm3d，空间常数的 prior 偏移会被归一化消除，
    因此用空间变化的越界图案验证 clamp。
    """
    cls = nnUNetTrainerPICAI_AnatomyGate_PositiveSampling_NoFFT
    net = _build_combined(cls, synthetic_arch, 5)
    x = torch.randn(2, 5, 8, 16, 16)
    scales = net.compute_gate_scales(x)
    assert scales.shape[1] == 3
    assert torch.allclose(scales, torch.ones_like(scales), atol=1e-6)

    # 非平凡 gate：越界 prior 必须先 clamp 再进 gate（等价于显式 clamp 后的合法 prior）
    torch.manual_seed(123)
    with torch.no_grad():
        net.gate.conv2.weight.normal_(0.0, 0.3)
        net.gate.conv2.bias.normal_(0.0, 0.3)
    pattern = torch.linspace(-5.0, 7.0, 8 * 16 * 16).reshape(8, 16, 16)
    x_oob, x_valid = x.clone(), x.clone()
    x_oob[:, 3], x_oob[:, 4] = pattern, -pattern
    x_valid[:, 3] = pattern.clamp(0.0, 1.0)
    x_valid[:, 4] = (-pattern).clamp(0.0, 1.0)
    scales_oob = net.compute_gate_scales(x_oob)
    assert torch.equal(scales_oob, net.compute_gate_scales(x_valid))
    # 控制：交换 PZ/TZ 通道（仍合法 [0,1]）必须改变 scales，证明 gate 确实读取 prior
    x_swapped = x_valid.clone()
    x_swapped[:, 3], x_swapped[:, 4] = x_valid[:, 4].clone(), x_valid[:, 3].clone()
    assert not torch.allclose(net.compute_gate_scales(x_swapped), scales_oob)


# ===========================================================================
# 浅层序列特异特征融合（Research Plan §8.10）：三个 Trainer 的接线
# ===========================================================================
_FEATURE_TRAINERS = (
    nnUNetTrainerPICAI_FeatureNoGate_PositiveSampling_NoFFT,
    nnUNetTrainerPICAI_FeatureImageGate_PositiveSampling_NoFFT,
    nnUNetTrainerPICAI_FeatureAnatomyGate_PositiveSampling_NoFFT,
)


def _build_feature_trainer(cls, synthetic_arch, in_channels, deep_supervision=False):
    return cls.build_network_architecture(
        synthetic_arch["architecture_class_name"],
        synthetic_arch["arch_init_kwargs"],
        synthetic_arch["arch_init_kwargs_req_import"],
        in_channels,
        2,
        deep_supervision,
    )


@pytest.mark.parametrize("trainer", _FEATURE_TRAINERS)
def test_feature_trainers_reuse_flce_nofft_and_positive_sampling(trainer):
    """三个 feature Trainer 必须复用 FLCE 损失、NoFFT 修复与阳性病例采样。"""
    assert issubclass(trainer, nnUNetTrainer)
    assert issubclass(trainer, PositiveCaseSamplingMixin)
    assert issubclass(trainer, nnUNetTrainerPICAI_FLCE_PositiveSampling_NoFFT)
    assert issubclass(trainer, _FeatureFusionTrainerBase)
    # 损失经 MRO 解析到 PI-CAI FLCE，而不是 Dice+CE
    assert trainer._build_loss is PICAIFocalCrossEntropyLossMixin._build_loss
    assert trainer.get_training_transforms is not nnUNetTrainer.get_training_transforms
    # 训练 loader 必须由采样 mixin 提供（不被 feature 基类或网络层抢先解析）；
    # 该 mixin 内部把验证 loader 保持为原生 nnUNetDataLoader（见 test_positive_case_sampling.py）
    assert trainer.get_dataloaders is PositiveCaseSamplingMixin.get_dataloaders
    # 阳性病例采样的默认槽位数沿用同一常量（batch size 2 -> 每批 1 个阳性 patch）
    assert trainer.positive_cases_per_batch == 1


@pytest.mark.parametrize("trainer", _FEATURE_TRAINERS)
def test_feature_trainers_do_not_inherit_input_level_gate_trainers(trainer):
    """浅层版本不得继承输入级 gate / DiceCE 分支，避免混淆两个表征层级。"""
    for forbidden in (
        nnUNetTrainerPICAI_ImageGate,
        nnUNetTrainerPICAI_AnatomyGate,
        nnUNetTrainerPICAI_ImageGate_PositiveSampling_NoFFT,
        nnUNetTrainerPICAI_AnatomyGate_PositiveSampling_NoFFT,
        nnUNetTrainerPICAI_DiceCE_NoFFT,
        _GatedTrainerBase,
    ):
        assert not issubclass(trainer, forbidden), (
            f"{trainer.__name__} 不得继承 {forbidden.__name__}"
        )
        assert forbidden not in trainer.__mro__


def test_feature_trainers_declare_expected_channel_contracts():
    assert (
        nnUNetTrainerPICAI_FeatureNoGate_PositiveSampling_NoFFT.expected_input_channels
        == 3
    )
    assert (
        nnUNetTrainerPICAI_FeatureImageGate_PositiveSampling_NoFFT.expected_input_channels
        == 3
    )
    assert (
        nnUNetTrainerPICAI_FeatureAnatomyGate_PositiveSampling_NoFFT.expected_input_channels
        == 5
    )
    assert not nnUNetTrainerPICAI_FeatureNoGate_PositiveSampling_NoFFT.feature_uses_gate
    assert nnUNetTrainerPICAI_FeatureImageGate_PositiveSampling_NoFFT.feature_uses_gate
    assert (
        nnUNetTrainerPICAI_FeatureAnatomyGate_PositiveSampling_NoFFT.feature_uses_gate
    )
    assert (
        nnUNetTrainerPICAI_FeatureAnatomyGate_PositiveSampling_NoFFT.num_prior_channels
        == 2
    )


@pytest.mark.parametrize(
    "trainer,in_channels",
    [
        (nnUNetTrainerPICAI_FeatureNoGate_PositiveSampling_NoFFT, 3),
        (nnUNetTrainerPICAI_FeatureImageGate_PositiveSampling_NoFFT, 3),
        (nnUNetTrainerPICAI_FeatureAnatomyGate_PositiveSampling_NoFFT, 5),
    ],
)
def test_feature_trainer_network_uses_stems_and_native_backbone(
    synthetic_arch, trainer, in_channels
):
    net = _build_feature_trainer(trainer, synthetic_arch, in_channels)
    assert isinstance(net, FeatureFusionNNUNet)
    assert not isinstance(net, GatedNNUNet)
    assert type(net.backbone).__name__ == "PlainConvUNet"  # 由 plans 构建，未复制
    assert len(net.stems) == 3
    assert net.projection.out_channels == 3
    assert net.stems[0].channels == FEATURE_STEM_CHANNELS
    # gate 的存在性必须与 Trainer 声明一致
    assert (net.gate is not None) is trainer.feature_uses_gate


def test_feature_anatomy_trainer_keeps_prior_out_of_stems_and_backbone(synthetic_arch):
    """PZ/TZ 只进 gate：stem 与投影层/backbone 的输入都不含 prior。"""
    net = _build_feature_trainer(
        nnUNetTrainerPICAI_FeatureAnatomyGate_PositiveSampling_NoFFT,
        synthetic_arch,
        5,
    )
    net.eval()
    assert net.total_input_channels == 5
    assert net.gate.input_channels == FEATURE_STEM_CHANNELS * 3 + 2
    backbone_channels: list[int] = []
    handle = net.backbone.register_forward_pre_hook(
        lambda module, args: backbone_channels.append(args[0].shape[1])
    )
    try:
        with torch.no_grad():
            net(torch.randn(1, 5, 8, 16, 16))
    finally:
        handle.remove()
    assert backbone_channels == [3]
    # 固定 MRI、只改 PZ/TZ：stem 特征逐值相同（prior 没有进入 stem）
    mri = torch.randn(1, 3, 8, 16, 16)
    pattern = torch.linspace(0.0, 1.0, 8 * 16 * 16).reshape(8, 16, 16)
    x_a = torch.cat([mri, torch.stack([pattern, 1 - pattern])[None]], dim=1)
    x_b = torch.cat([mri, torch.stack([1 - pattern, pattern])[None]], dim=1)
    with torch.no_grad():
        feats_a, _ = net._stem_features(x_a)
        feats_b, _ = net._stem_features(x_b)
    assert all(torch.equal(a, b) for a, b in zip(feats_a, feats_b))


def test_feature_trainers_forward_in_inference_mode(synthetic_arch):
    """inference 模式（deep_supervision=False）下三条网络都能给出单张量 2 类输出。"""
    for trainer, in_channels in (
        (nnUNetTrainerPICAI_FeatureNoGate_PositiveSampling_NoFFT, 3),
        (nnUNetTrainerPICAI_FeatureImageGate_PositiveSampling_NoFFT, 3),
        (nnUNetTrainerPICAI_FeatureAnatomyGate_PositiveSampling_NoFFT, 5),
    ):
        net = _build_feature_trainer(trainer, synthetic_arch, in_channels)
        net.eval()
        with torch.no_grad():
            out = net(torch.randn(1, in_channels, 8, 16, 16))
        assert isinstance(out, torch.Tensor)
        assert out.shape[1] == 2


def test_feature_trainers_do_not_load_existing_checkpoints(synthetic_arch):
    """三个 feature 网络之间以及与原输入级 gate 网络之间都不得 strict 互认。"""
    no_gate = _build_feature_trainer(
        nnUNetTrainerPICAI_FeatureNoGate_PositiveSampling_NoFFT, synthetic_arch, 3
    )
    image = _build_feature_trainer(
        nnUNetTrainerPICAI_FeatureImageGate_PositiveSampling_NoFFT, synthetic_arch, 3
    )
    anatomy = _build_feature_trainer(
        nnUNetTrainerPICAI_FeatureAnatomyGate_PositiveSampling_NoFFT, synthetic_arch, 5
    )
    input_level = (
        nnUNetTrainerPICAI_ImageGate_PositiveSampling_NoFFT.build_network_architecture(
            synthetic_arch["architecture_class_name"],
            synthetic_arch["arch_init_kwargs"],
            synthetic_arch["arch_init_kwargs_req_import"],
            3,
            2,
            False,
        )
    )
    with pytest.raises(RuntimeError):
        no_gate.load_state_dict(image.state_dict(), strict=True)
    with pytest.raises(RuntimeError):
        anatomy.load_state_dict(image.state_dict(), strict=True)
    with pytest.raises(RuntimeError):
        image.load_state_dict(input_level.state_dict(), strict=True)


@pytest.mark.parametrize(
    "trainer,enable_ds",
    [
        (trainer, enable_ds)
        for trainer in _FEATURE_TRAINERS
        for enable_ds in (False, True)
    ],
)
def test_feature_trainer_loss_is_pi_cai_flce_not_dice_ce(trainer, enable_ds):
    """feature 三兄弟必须用 PI-CAI FLCE（不是 Dice+CE），deep supervision 权重沿用官方规则。"""
    assert trainer._build_loss is PICAIFocalCrossEntropyLossMixin._build_loss
    loss = trainer._build_loss(
        _MinimalLossBuilderState(enable_deep_supervision=enable_ds)
    )
    if enable_ds:
        assert isinstance(loss, DeepSupervisionWrapper)
        base = loss.loss
    else:
        base = loss
    assert isinstance(base, PiCAIFocalCrossEntropyLoss)
    assert (base.focal_weight, base.ce_weight, base.gamma) == (0.5, 0.5, 2.0)
    assert not isinstance(base, DC_and_CE_loss)
