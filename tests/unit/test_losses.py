"""损失与 deep supervision 测试（合成张量，强制 CPU）。"""
from __future__ import annotations

import pytest
import torch

from zonal_reliability_fusion.training import (
    DeepSupervisionLoss,
    DiceCELoss,
    PiCAIFocalCELoss,
    SoftDiceLoss,
    deep_supervision_weights,
    downsample_target_nearest,
)


def _batch(batch: int = 1, shape=(8, 16, 16), *, with_fg: bool = True, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(batch, 2, *shape, generator=generator)
    target = torch.zeros(batch, 1, *shape, dtype=torch.long)
    if with_fg:
        target[..., 2:5, 4:9, 4:9] = 1
    return logits, target


def test_loss_with_foreground_is_finite_and_backpropagates():
    logits, target = _batch(with_fg=True)
    logits.requires_grad_(True)
    loss = DiceCELoss()(logits, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_all_background_target_has_no_nan():
    logits, target = _batch(with_fg=False)
    loss = DiceCELoss()(logits, target)
    assert torch.isfinite(loss)
    assert not torch.isnan(loss)
    parts = DiceCELoss().components(logits, target)
    assert 0.0 <= parts["dice_score"] <= 1.0
    assert parts["dice_loss"] == pytest.approx(-parts["dice_score"])
    assert torch.isfinite(torch.tensor(parts["total"]))


def test_loss_matches_official_ce_minus_dice_formula():
    """Dice+CE 数值口径 = CE - Dice（官方 DC_and_CE_loss 用 -Dice，不含 +1 偏移）。"""
    logits, target = _batch(with_fg=True)
    loss_fn = DiceCELoss()
    manual_ce = torch.nn.functional.cross_entropy(logits, target[:, 0])
    manual_dice = -SoftDiceLoss().dice_score(logits, target)
    assert loss_fn(logits, target).item() == pytest.approx((manual_ce + manual_dice).item(), rel=1e-6)


def test_batch_dice_variant_finite():
    logits, target = _batch(batch=2, with_fg=True)
    loss = DiceCELoss(batch_dice=True)(logits, target)
    assert torch.isfinite(loss)


def test_soft_dice_returns_negative_dice_like_official():
    """自研 Dice 项返回 -Dice（与官方 SoftDiceLoss 数值一致，不含 +1 偏移）。"""
    target = torch.zeros(1, 1, 8, 8, 8, dtype=torch.long)
    target[..., 2:6, 2:6, 2:6] = 1
    background_pred = torch.full((1, 2, 8, 8, 8), -10.0)
    background_pred[:, 0] = 10.0  # 全背景预测 → 前景完全漏检
    assert SoftDiceLoss().dice_score(background_pred, target) < 0.01
    assert SoftDiceLoss()(background_pred, target) > -0.01

    mask = target[:, 0] == 1
    correct = torch.full((1, 2, 8, 8, 8), -10.0)
    correct[:, 0] = 10.0  # 默认预测背景
    correct[:, 1][mask] = 10.0  # 前景区域预测前景
    correct[:, 0][mask] = -10.0  # 前景区域压低背景 logit
    assert SoftDiceLoss().dice_score(correct, target) > 0.99
    assert SoftDiceLoss()(correct, target) < -0.99


def test_ds_weights_sum_to_one_and_lowest_is_zero():
    for n in (2, 3, 6):
        weights = deep_supervision_weights(n)
        assert len(weights) == n
        assert sum(weights) == pytest.approx(1.0)
        assert weights[-1] == 0.0
        assert weights[0] > weights[1] if n > 1 else True


def test_ds_weights_single_output_not_zero():
    assert deep_supervision_weights(1) == (1.0,)


def test_ds_weights_invalid():
    with pytest.raises(ValueError):
        deep_supervision_weights(0)


@pytest.mark.parametrize("weights", [(1.0, 0.0), (0.5, 0.5)])
def test_ds_loss_multi_scale(weights):
    logits_main, target = _batch(with_fg=True)
    logits_low = torch.randn(1, 2, 4, 8, 8)
    outputs = [logits_main, logits_low]
    base = DiceCELoss()
    loss_fn = DeepSupervisionLoss(base, weights=weights)
    loss = loss_fn(outputs, target)
    assert torch.isfinite(loss)
    # 低分辨率输出的 target 由 wrapper 自动 nearest 下采样
    low_target = downsample_target_nearest(target[:, 0], tuple(logits_low.shape[-3:]))
    expected = weights[0] * base(logits_main, target) + weights[1] * base(logits_low, low_target)
    assert loss.item() == pytest.approx(expected.item(), rel=1e-5, abs=1e-6)


def test_ds_loss_default_weights_ignore_lowest_resolution():
    logits_main, target = _batch(with_fg=True)
    outputs = [logits_main, torch.randn(1, 2, 4, 8, 8)]
    base = DiceCELoss()
    loss = DeepSupervisionLoss(base)(outputs, target)  # n=2 → [1.0, 0.0]
    assert loss.item() == pytest.approx(base(logits_main, target).item(), rel=1e-6)


def test_ds_loss_requires_nonzero_weight():
    with pytest.raises(ValueError):
        DeepSupervisionLoss(DiceCELoss(), weights=(0.0, 0.0))


def test_ds_loss_accepts_single_tensor():
    logits, target = _batch(with_fg=True)
    loss = DeepSupervisionLoss(DiceCELoss())(logits, target)
    assert torch.isfinite(loss)


def test_downsample_target_nearest_keeps_binary_values():
    _, target = _batch(with_fg=True)
    small = downsample_target_nearest(target[:, 0], (4, 8, 8))
    assert tuple(small.shape) == (1, 4, 8, 8)
    assert set(torch.unique(small).tolist()) <= {0, 1}
    assert small.dtype == torch.long
    same = downsample_target_nearest(target[:, 0], tuple(target.shape[-3:]))
    assert torch.equal(same, target[:, 0])
