"""M0 模型前向测试（合成小张量，强制 CPU）。"""
from __future__ import annotations

import pytest
import torch

from zonal_reliability_fusion.models import M0ConcatModel, PlanDrivenUNet3D
from zonal_reliability_fusion.models.blocks import center_crop_or_pad_3d, same_padding


def test_main_output_shape(mini_model: M0ConcatModel, mini_plan):
    x = torch.randn(2, 3, *mini_plan.patch_size)
    out = mini_model(x, deep_supervision=False)
    assert isinstance(out, torch.Tensor)
    assert tuple(out.shape) == (2, 2, *mini_plan.patch_size)
    assert torch.isfinite(out).all()


def test_deep_supervision_order_high_to_low(mini_model: M0ConcatModel, mini_plan):
    x = torch.randn(1, 3, *mini_plan.patch_size)
    outputs = mini_model(x)  # 默认 DS 打开
    assert isinstance(outputs, list)
    expected = list(mini_plan.decoder_output_shapes())
    assert len(outputs) == len(expected)
    assert [tuple(o.shape[-3:]) for o in outputs] == expected
    for o in outputs:
        assert o.shape[1] == 2  # 二分类 logits
        assert torch.isfinite(o).all()
    # 尺寸必须单调不增（高分辨率在前）
    voxels = [int(torch.tensor(o.shape[-3:]).prod()) for o in outputs]
    assert voxels == sorted(voxels, reverse=True)


def test_channel_mismatch_raises(mini_model: M0ConcatModel, mini_plan):
    with pytest.raises(ValueError, match="输入通道"):
        mini_model(torch.randn(1, 2, *mini_plan.patch_size))


def test_wrong_ndim_raises(mini_model: M0ConcatModel, mini_plan):
    with pytest.raises(ValueError):
        mini_model(torch.randn(1, 3, mini_plan.patch_size[0], mini_plan.patch_size[1]))


def test_ds_toggle(mini_model: M0ConcatModel, mini_plan):
    x = torch.randn(1, 3, *mini_plan.patch_size)
    mini_model.set_deep_supervision(False)
    assert isinstance(mini_model(x), torch.Tensor)
    mini_model.set_deep_supervision(True)
    assert isinstance(mini_model(x), list)


def test_main_logits_helper(mini_model: M0ConcatModel, mini_plan):
    x = torch.randn(1, 3, *mini_plan.patch_size)
    outputs = mini_model(x)
    main = M0ConcatModel.main_logits(outputs)
    assert torch.equal(main, outputs[0])
    assert torch.equal(M0ConcatModel.main_logits(main), main)


def test_mismatch_plan_forward_uses_center_crop(mismatch_plan):
    """patch 8x10x10 → 中间层 4x5x5，上采样后比 skip 多 1 体素，必须走中心裁剪。"""
    model = PlanDrivenUNet3D(mismatch_plan, in_channels=3, num_classes=2, deep_supervision=True)
    outputs = model(torch.randn(1, 3, *mismatch_plan.patch_size))
    assert [tuple(o.shape[-3:]) for o in outputs] == list(mismatch_plan.decoder_output_shapes())
    assert mismatch_plan.decoder_output_shapes() == ((8, 10, 10), (4, 5, 5))


def test_center_crop_or_pad_3d_crop_and_pad():
    x = torch.arange(4 * 5 * 5, dtype=torch.float32).reshape(1, 1, 4, 5, 5)
    cropped = center_crop_or_pad_3d(x, (4, 4, 4))
    assert tuple(cropped.shape) == (1, 1, 4, 4, 4)
    assert torch.equal(cropped, x[..., :4, :4])
    padded = center_crop_or_pad_3d(x, (6, 7, 7))
    assert tuple(padded.shape) == (1, 1, 6, 7, 7)
    assert torch.equal(padded[..., 1:5, 1:6, 1:6], x)
    assert float(padded[..., 0, :, :].abs().sum()) == 0.0


def test_center_crop_or_pad_rejects_extreme_diff():
    x = torch.zeros(1, 1, 4, 4, 4)
    with pytest.raises(ValueError):
        center_crop_or_pad_3d(x, (1, 1, 1))


def test_same_padding_rejects_even_kernel():
    assert same_padding((1, 3, 3)) == (0, 1, 1)
    with pytest.raises(ValueError):
        same_padding((2, 3, 3))


def test_summary_contains_plan_mapping(mini_model: M0ConcatModel, mini_plan):
    summary = mini_model.summary()
    assert summary["model_id"] == "M0"
    assert summary["num_classes"] == 2
    assert summary["patch_size"] == list(mini_plan.patch_size)
    assert summary["decoder_output_shapes"] == [list(s) for s in mini_plan.decoder_output_shapes()]
    assert summary["modality_order"] == ["T2W", "ADC", "HBV"]
    assert mini_model.num_parameters() > 0
    assert "M0" in mini_model.format_summary()


def test_model_has_no_softmax_in_forward(mini_model: M0ConcatModel, mini_plan):
    """输出必须是 logits（不满足概率和为 1 的约束），证明网络内没有 softmax。"""
    logits = mini_model(torch.randn(1, 3, *mini_plan.patch_size), deep_supervision=False)
    probs_sum = logits.softmax(dim=1).sum(dim=1)
    assert torch.allclose(probs_sum, torch.ones_like(probs_sum))
    raw_sum = logits.sum(dim=1)
    assert not torch.allclose(raw_sum, torch.ones_like(raw_sum), atol=1e-3)  # 原输出不是概率
