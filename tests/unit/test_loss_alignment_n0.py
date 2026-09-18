"""M0 与正式 N0 的**损失口径对齐**测试（合成张量；不读真实数据、不使用 GPU）。

对齐对象：

- M0：`zonal_reliability_fusion.training.PiCAIFocalCELoss`（自研实现，不得 import nnunetv2）；
- N0：`zonal_reliability_fusion.integrations.nnunet_picai_flce.PiCAIFocalCrossEntropyLoss`
  （PI-CAI official-style Focal+CE 的 v2 适配层）。

验收（见 `docs/P2_M0_Implementation.md` §18）：同一 logits/target 下两者逐值一致（atol ≤ 1e-7），
deep-supervision 包装后的加权结果与 nnU-Net 官方 `DeepSupervisionWrapper` 一致。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "src"
NNUNET_SOURCE = PROJECT_ROOT / "third_party" / "nnUNet"
for path in (str(SRC), str(NNUNET_SOURCE)):
    if path not in sys.path:
        sys.path.insert(0, path)

from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper  # noqa: E402

from zonal_reliability_fusion.integrations.nnunet_picai_flce import (  # noqa: E402
    PiCAIFocalCrossEntropyLoss,
)
from zonal_reliability_fusion.config.experiment import (  # noqa: E402
    ExperimentConfigError,
    LossConfig,
    validate_loss_config,
)
from zonal_reliability_fusion.training import (  # noqa: E402
    DeepSupervisionLoss,
    PiCAIFocalCELoss,
    deep_supervision_weights,
    downsample_target_nearest,
)

ATOL = 1e-7


def _batch(batch: int = 1, shape=(8, 16, 16), *, with_fg: bool = True, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(batch, 2, *shape, generator=generator)
    target = torch.zeros(batch, 1, *shape, dtype=torch.long)
    if with_fg:
        target[..., 2:5, 4:9, 4:9] = 1
    return logits, target


# --------------------------------------------------------------------------- 单尺度数值对齐


@pytest.mark.parametrize("with_fg", [True, False])
def test_m0_focal_ce_matches_n0_adapter_value(with_fg):
    logits, target = _batch(batch=2, with_fg=with_fg, seed=3)
    m0 = PiCAIFocalCELoss(focal_weight=0.5, ce_weight=0.5, gamma=2.0, smooth=1e-5)
    n0 = PiCAIFocalCrossEntropyLoss(focal_weight=0.5, ce_weight=0.5, gamma=2.0, smooth=1e-5)
    assert m0(logits, target).item() == pytest.approx(n0(logits, target).item(), abs=ATOL, rel=0.0)


def test_m0_focal_ce_matches_n0_adapter_without_channel_dim():
    logits, target = _batch(with_fg=True)
    m0 = PiCAIFocalCELoss()
    n0 = PiCAIFocalCrossEntropyLoss()
    assert m0(logits, target[:, 0]).item() == pytest.approx(n0(logits, target[:, 0]).item(), abs=ATOL)


def test_m0_focal_ce_gradient_is_finite():
    logits, target = _batch(with_fg=True)
    logits.requires_grad_(True)
    loss = PiCAIFocalCELoss()(logits, target)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_m0_all_background_target_is_finite():
    logits, target = _batch(with_fg=False)
    loss = PiCAIFocalCELoss()(logits, target)
    assert torch.isfinite(loss)
    parts = PiCAIFocalCELoss().components(logits, target)
    assert parts["focal_loss"] >= 0.0 and parts["ce_loss"] >= 0.0
    assert 0.0 < parts["mean_pt"] <= 1.0


@pytest.mark.parametrize(
    "bad_target",
    [
        torch.tensor([[[[-1]]]]),
        torch.tensor([[[[2]]]]),
        torch.tensor([[[[0.5]]]]),
    ],
)
def test_m0_and_n0_reject_same_invalid_targets(bad_target):
    logits = torch.zeros(1, 2, 1, 1)
    with pytest.raises(ValueError):
        PiCAIFocalCELoss()(logits, bad_target)
    with pytest.raises(ValueError):
        PiCAIFocalCrossEntropyLoss()(logits, bad_target)


# --------------------------------------------------------------------------- deep supervision 对齐


def test_ds_weights_follow_n0_default_formula():
    """M0 的 DS 权重必须与 nnU-Net v2.6.2 默认公式一致：1/2^i、最低置 0、归一化。"""
    for n in (2, 6):
        expected = [1.0 / (2**i) for i in range(n)]
        expected[-1] = 0.0
        total = sum(expected)
        expected = [w / total for w in expected]
        got = deep_supervision_weights(n)
        assert got == pytest.approx(expected, abs=1e-12)
        assert got[0] == pytest.approx(0.516129, abs=1e-6) if n == 6 else True


def test_ds_wrapped_loss_matches_nnunet_official_wrapper():
    """同一多尺度输出/权重下，M0 `DeepSupervisionLoss` 与官方 `DeepSupervisionWrapper` 逐值一致。"""
    logits_main, target = _batch(with_fg=True, seed=5)
    logits_low = torch.randn(1, 2, 4, 8, 8, generator=torch.Generator().manual_seed(6))
    weights = (0.7, 0.3)

    m0_base = PiCAIFocalCELoss()
    m0_loss = DeepSupervisionLoss(m0_base, weights=weights)([logits_main, logits_low], target)

    n0_base = PiCAIFocalCrossEntropyLoss()
    # nnU-Net 的 wrapper 接收「已下采样好的 target 列表」；此处用与 M0 wrapper 相同的 nearest 结果，
    # 使比较只反映损失与权重语义，而不掺入双方的 target 下采样实现差异。
    low_target = downsample_target_nearest(target[:, 0], tuple(logits_low.shape[-3:]))
    n0_loss = DeepSupervisionWrapper(n0_base, torch.tensor(weights))(
        [logits_main, logits_low], [target, low_target]
    )

    assert m0_loss.item() == pytest.approx(float(n0_loss), abs=1e-6)


def test_ds_wrapped_loss_ignores_zero_weight_scale():
    logits_main, target = _batch(with_fg=True)
    outputs = [logits_main, torch.randn(1, 2, 4, 8, 8)]
    base = PiCAIFocalCELoss()
    loss = DeepSupervisionLoss(base, weights=(1.0, 0.0))(outputs, target)
    assert loss.item() == pytest.approx(base(logits_main, target).item(), rel=1e-6)


# --------------------------------------------------------------------------- 仓库配置


def test_repo_experiment_config_declares_focal_ce():
    """仓库的 M0 配置必须显式声明 focal_ce（与正式 N0 对齐），并给出全部必需参数。"""
    import yaml

    doc = yaml.safe_load((PROJECT_ROOT / "configs/experiments/m0_picai_3d_fullres.yaml").read_text(encoding="utf-8"))
    loss = doc["loss"]
    assert loss["name"] == "focal_ce"
    assert loss["focal_weight"] == pytest.approx(0.5)
    assert loss["ce_weight"] == pytest.approx(0.5)
    assert loss["gamma"] == pytest.approx(2.0)
    assert loss["smooth"] == pytest.approx(1e-5)
    assert "dice_weight" not in loss and "include_background" not in loss


# --------------------------------------------------------------------------- 配置层显式性校验


def _loss_cfg(**overrides):
    from types import SimpleNamespace

    base = dict(name="focal_ce", ce_weight=0.5, smooth=1e-5, batch_dice=False,
                focal_weight=0.5, gamma=2.0, dice_weight=None, include_background=None)
    base.update(overrides)
    return SimpleNamespace(loss=LossConfig(**base))


def test_validate_loss_config_accepts_both_documented_variants():
    validate_loss_config(_loss_cfg())  # focal_ce（与正式 N0 对齐）
    validate_loss_config(
        _loss_cfg(name="dice_ce", dice_weight=1.0, include_background=False, focal_weight=None, gamma=None)
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"name": "focal_ce", "focal_weight": None},       # 缺 focal_weight
        {"name": "focal_ce", "gamma": None},              # 缺 gamma
        {"name": "focal_ce", "dice_weight": 1.0},         # 混用 dice 字段
        {"name": "focal_ce", "include_background": False},  # 混用 dice 字段
        {"name": "dice_ce", "dice_weight": None},         # 缺 dice_weight
        {"name": "dice_ce", "include_background": None},  # 缺 include_background
        {"name": "dice_ce", "focal_weight": 0.5},         # 混用 focal 字段
        {"name": "nope"},                                 # 未知名称
        {"name": "focal_ce", "ce_weight": 0.0},           # ce_weight 非正
        {"name": "focal_ce", "smooth": 1.0},              # smooth 越界
    ],
)
def test_validate_loss_config_rejects_mixed_or_incomplete_configs(overrides):
    with pytest.raises(ExperimentConfigError):
        validate_loss_config(_loss_cfg(**overrides))
