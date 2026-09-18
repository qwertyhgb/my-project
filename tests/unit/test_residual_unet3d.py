"""v2.3 PlanDrivenResidualEncoderUNet3D / M0ResidualConcatModel 合成测试（任务书九·7–13,21）。

全部使用缩小的合法 synthetic plan + CPU；不读取真实医学影像、不初始化 CUDA、不对真实 [16,320,320]
patch 做昂贵 forward。
"""
from __future__ import annotations

import pytest
import torch
from conftest import build_resenc_arch_doc

from zonal_reliability_fusion.config import architecture_spec_from_mapping
from zonal_reliability_fusion.inference.sliding_window import sliding_window_predict
from zonal_reliability_fusion.models import (
    M0ResidualConcatModel,
    PlanDrivenResidualEncoderUNet3D,
)
from zonal_reliability_fusion.models.residual_blocks import (
    ResidualBlock3d,
    StackedResidualBlocks3d,
)
from zonal_reliability_fusion.models.residual_unet3d import (
    zero_init_residual_last_norms,
)


# ------------------------------------------------------------------ 7. reduced plan 的 stage shape
def test_encoder_stage_shapes_match_plan(mini_resenc_model: M0ResidualConcatModel, mini_resenc_plan):
    net = mini_resenc_model.net
    assert net.encoder_stage_shapes() == mini_resenc_plan.stage_shapes()
    # stem + 每级 residual stage
    assert net.stem is not None
    assert len(net.encoder_stages) == mini_resenc_plan.n_stages
    for stage in net.encoder_stages:
        assert isinstance(stage, StackedResidualBlocks3d)


def test_blocks_per_stage_from_spec_not_plan(mini_resenc_model: M0ResidualConcatModel):
    net = mini_resenc_model.net
    assert net.blocks_per_stage == (1, 2, 2, 2)
    # 残差块总数 = sum(blocks_per_stage)
    assert net.num_residual_blocks() == sum(net.blocks_per_stage)
    assert all(isinstance(stage.blocks[0], ResidualBlock3d) for stage in net.encoder_stages)


def test_zero_init_last_norms_counted(mini_resenc_model: M0ResidualConcatModel):
    net = mini_resenc_model.net
    assert net._zero_init_blocks == sum(net.blocks_per_stage)
    # 再次调用返回相同数量（幂等）
    assert zero_init_residual_last_norms(net) == sum(net.blocks_per_stage)


# ------------------------------------------------------------------ 8. deep supervision 数量/顺序/shape
def test_deep_supervision_count_order_shape(mini_resenc_model: M0ResidualConcatModel, mini_resenc_plan):
    x = torch.randn(1, 3, *mini_resenc_plan.patch_size)
    outputs = mini_resenc_model(x)  # DS 默认打开
    assert isinstance(outputs, list)
    expected = list(mini_resenc_plan.decoder_output_shapes())
    assert len(outputs) == mini_resenc_plan.n_stages - 1 == len(expected)
    assert [tuple(o.shape[-3:]) for o in outputs] == expected
    for o in outputs:
        assert o.shape[1] == 2 and torch.isfinite(o).all()
    voxels = [int(torch.tensor(o.shape[-3:]).prod()) for o in outputs]
    assert voxels == sorted(voxels, reverse=True)  # 高分辨率在前


def test_ds_toggle_and_main_output(mini_resenc_model: M0ResidualConcatModel, mini_resenc_plan):
    x = torch.randn(2, 3, *mini_resenc_plan.patch_size)
    mini_resenc_model.set_deep_supervision(False)
    out = mini_resenc_model(x)
    assert isinstance(out, torch.Tensor) and tuple(out.shape) == (2, 2, *mini_resenc_plan.patch_size)
    mini_resenc_model.set_deep_supervision(True)
    assert isinstance(mini_resenc_model(x), list)


def test_odd_plan_uses_center_crop(odd_resenc_plan):
    doc = build_resenc_arch_doc(odd_resenc_plan, blocks_per_stage=[1, 2, 2, 2])
    spec = architecture_spec_from_mapping(doc, source_path="odd")
    model = M0ResidualConcatModel(odd_resenc_plan, spec, in_channels=3, num_classes=2, deep_supervision=True)
    outputs = model(torch.randn(1, 3, *odd_resenc_plan.patch_size))
    assert [tuple(o.shape[-3:]) for o in outputs] == list(odd_resenc_plan.decoder_output_shapes())


# ------------------------------------------------------------------ 9. forward / 无 softmax
def test_forward_finite_and_no_softmax(mini_resenc_model: M0ResidualConcatModel, mini_resenc_plan):
    logits = mini_resenc_model(torch.randn(1, 3, *mini_resenc_plan.patch_size), deep_supervision=False)
    assert torch.isfinite(logits).all()
    raw_sum = logits.sum(dim=1)
    assert not torch.allclose(raw_sum, torch.ones_like(raw_sum), atol=1e-3)  # 输出是 logits，不是概率


def test_channel_and_ndim_guards(mini_resenc_model: M0ResidualConcatModel, mini_resenc_plan):
    with pytest.raises(ValueError, match="通道"):
        mini_resenc_model(torch.randn(1, 2, *mini_resenc_plan.patch_size))
    with pytest.raises(ValueError):
        mini_resenc_model(torch.randn(1, 3, mini_resenc_plan.patch_size[0], mini_resenc_plan.patch_size[1]))


# ------------------------------------------------------------------ 10. backward
def test_backward_populates_all_gradients(mini_resenc_model: M0ResidualConcatModel, mini_resenc_plan):
    x = torch.randn(2, 3, *mini_resenc_plan.patch_size)
    outputs = mini_resenc_model(x)
    loss = sum(o.float().pow(2).mean() for o in outputs)  # 多尺度都能反传
    loss.backward()
    grads = [p.grad for p in mini_resenc_model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)
    assert any(torch.count_nonzero(g) > 0 for g in grads)
    assert torch.isfinite(loss).all()


# ------------------------------------------------------------------ 11. AMP / autocast 可用性
def test_autocast_forward_is_finite(mini_resenc_model: M0ResidualConcatModel, mini_resenc_plan):
    """CPU autocast(bf16) 下前向可用（AMP 兼容性；不使用 GPU）。"""
    x = torch.randn(1, 3, *mini_resenc_plan.patch_size)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = mini_resenc_model(x, deep_supervision=False)
    assert tuple(out.shape) == (1, 2, *mini_resenc_plan.patch_size)
    assert torch.isfinite(out.float()).all()


# ------------------------------------------------------------------ 12. reduced-volume sliding-window
def test_reduced_sliding_window_output_shape(mini_resenc_model: M0ResidualConcatModel, mini_resenc_plan):
    vol = torch.randn(1, 3, 12, 20, 20)  # >= patch，且非 patch 整数倍
    logits = sliding_window_predict(
        mini_resenc_model,
        vol,
        patch_size=mini_resenc_plan.patch_size,
        step_fraction=0.5,
        gaussian=True,
        progress=False,
    )
    assert tuple(logits.shape) == (1, 2, 12, 20, 20)
    assert torch.isfinite(logits).all()


# ------------------------------------------------------------------ 13. 参数统计
def test_parameter_counts(mini_resenc_model: M0ResidualConcatModel):
    total = mini_resenc_model.num_parameters()
    trainable = mini_resenc_model.num_trainable_parameters()
    assert total > 0 and trainable == total  # 全部可训练
    assert total == sum(p.numel() for p in mini_resenc_model.parameters())


# ------------------------------------------------------------------ 21. M0 不接受 PZ/TZ/WG/gate 输入
def test_m0_rejects_anatomy_and_gate_inputs(mini_resenc_model: M0ResidualConcatModel, mini_resenc_plan):
    x = torch.randn(1, 3, *mini_resenc_plan.patch_size)
    assert M0ResidualConcatModel.ACCEPTS_ANATOMY_INPUTS is False
    for forbidden in ("zonal", "pz_tz", "wg", "gate", "gate_weight"):
        with pytest.raises(TypeError):
            mini_resenc_model(x, **{forbidden: torch.randn(1, 2, *mini_resenc_plan.patch_size)})


def test_expected_output_shapes_guard(mini_resenc_model: M0ResidualConcatModel, mini_resenc_plan):
    assert mini_resenc_model.net.expected_output_shapes() == mini_resenc_plan.decoder_output_shapes()
    with pytest.raises(ValueError):
        mini_resenc_model.net.expected_output_shapes((1, 2, 3))


def test_spec_mismatched_features_rejected(mini_resenc_plan):
    """结构字段与冻结 plan 不一致（features）→ 拒绝构建（validate_architecture_against_plan）。"""
    from zonal_reliability_fusion.config.architecture import ArchitectureConfigError

    doc = build_resenc_arch_doc(mini_resenc_plan, blocks_per_stage=[1, 2, 2, 2])
    doc["features_per_stage"] = [4, 8, 16, 32]  # 与 plan 不符
    spec = architecture_spec_from_mapping(doc, source_path="bad")
    with pytest.raises(ArchitectureConfigError):
        PlanDrivenResidualEncoderUNet3D(mini_resenc_plan, spec, in_channels=3, num_classes=2)
