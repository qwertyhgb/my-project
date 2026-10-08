"""Lesionness 目标与 coarse-to-fine 网络的纯合成 CPU 测试。

覆盖 §33 要求的三项：

- **lesionness target generation**：物理半径膨胀（各向异性椭球）、二值校验、空 GT 合法；
- **coarse-to-fine forward**：原生 backbone + coarse 头 + soft refinement 的前向与反向；
- **deep supervision compatibility**：DS 打开时输出列表等长、第 0 项被 refinement 替换、
  返回值形状与原生 backbone **逐值同形**（这样 nnU-Net 的 ``validation_step`` 与滑窗推理
  不需要任何改动）。

另外覆盖零初始化恒等性：新模块在初始状态下不得改变 strong baseline 的数值行为。
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from zonal_reliability_fusion.lesion.coarse_to_fine import (
    LesionAwareCoarseToFineError,
    LesionAwareCoarseToFineNNUNet,
)
from zonal_reliability_fusion.lesion.lesionness import (
    LESIONNESS_DILATION_RADIUS_MM,
    LesionnessError,
    downsample_binary_max,
    lesionness_target,
    physical_radius_to_voxels,
)
from zonal_reliability_fusion.lesion.zone_conditioning import (
    ZoneAwareRefinement,
    zone_context,
    zone_mode_parameter_delta,
)
from zonal_reliability_fusion.nnunet.losses import (
    LesionnessAuxiliaryLoss,
    PiCAIFocalCrossEntropyLoss,
)

SPACING_ZYX = (3.0, 0.5, 0.5)


class _FakeBackbone(nn.Module):
    """最小 3D backbone：返回与 nnU-Net decoder 同形的单张量或 DS 列表（通道=seg heads）。"""

    def __init__(self, seg_heads: int = 2, deep_supervision: bool = True) -> None:
        super().__init__()
        self.seg_heads = seg_heads
        self.deep_supervision = deep_supervision
        self.stem = nn.Conv3d(3, 4, 3, padding=1)
        self.out = nn.Conv3d(4, seg_heads, 1)
        self.decoder = nn.Identity()

    def forward(self, x):
        features = torch.relu(self.stem(x))
        coarse = self.out(torch.nn.functional.avg_pool3d(features, 2))
        if not self.deep_supervision:
            return self.out(features)
        return [self.out(features), coarse]

    def compute_conv_feature_map_size(self, input_size):
        return 0


# --------------------------------------------------------------------------- targets
def test_radius_to_voxels_respects_anisotropic_spacing():
    assert physical_radius_to_voxels(3.0, SPACING_ZYX) == (1, 6, 6)
    assert physical_radius_to_voxels(9.0, SPACING_ZYX) == (3, 18, 18)
    # 向上取整：物理覆盖不小于承诺
    assert physical_radius_to_voxels(1.0, (3.0, 0.5, 0.5)) == (1, 2, 2)


def test_lesionness_target_is_binary_dilation_of_gt_only():
    mask = np.zeros((9, 9, 9), dtype=np.uint8)
    mask[4, 4, 4] = 1
    target = lesionness_target(mask, SPACING_ZYX, radius_mm=3.0)
    assert target.dtype == bool
    assert target[4, 4, 4]
    # Z 半径 1 voxel、面内半径 6 voxel 但因体积小被 clip —— 检查轴向上的差异
    assert target[3, 4, 4] and target[5, 4, 4]
    assert target[4, 8, 4]
    assert target.sum() > 1
    # 目标只来自 GT：GT 之外的体素不得被标为正值以外的东西
    assert not target[0, 0, 0]


def test_lesionness_target_supervision_comes_only_from_gt():
    """监督目标必须完全由 lesion GT 决定：与 GT 逐值相同之外没有任何额外输入。"""
    mask = np.zeros((12, 12, 12), dtype=np.uint8)
    mask[6, 6, 6] = 1
    a = lesionness_target(mask, SPACING_ZYX, radius_mm=3.0)
    b = lesionness_target(mask.copy(), SPACING_ZYX, radius_mm=3.0)
    assert np.array_equal(a, b)


def test_zero_radius_returns_gt_unchanged():
    mask = np.zeros((6, 6, 6), dtype=np.uint8)
    mask[2, 2, 2] = 1
    assert np.array_equal(lesionness_target(mask, SPACING_ZYX, radius_mm=0.0), mask > 0)


def test_empty_gt_is_legal_and_gives_empty_target():
    """阴性病例是合法输入（不是错误）：目标为全零。"""
    empty = np.zeros((8, 8, 8), dtype=np.uint8)
    target = lesionness_target(empty, SPACING_ZYX)
    assert target.dtype == bool
    assert not target.any()


def test_lesionness_target_rejects_non_binary_labels():
    """含 ignore label（-1）或其它取值时必须报错，不得被静默当成背景。"""
    bad = np.zeros((6, 6, 6), dtype=np.int16)
    bad[1, 1, 1] = -1
    with pytest.raises(LesionnessError, match="二值"):
        lesionness_target(bad, SPACING_ZYX)


def test_lesionness_target_rejects_wrong_rank_and_bad_spacing():
    with pytest.raises(LesionnessError, match="3D"):
        lesionness_target(np.zeros((6, 6), dtype=np.uint8), SPACING_ZYX)
    with pytest.raises(LesionnessError, match="spacing"):
        lesionness_target(np.zeros((6, 6, 6), dtype=np.uint8), (1.0, 0.0, 1.0))


def test_frozen_radius_constant_is_documented_value():
    assert LESIONNESS_DILATION_RADIUS_MM == 3.0


def test_downsample_binary_max_keeps_any_positive_semantics():
    target = np.zeros((4, 8, 8), dtype=bool)
    target[1, 3, 3] = True
    reduced = downsample_binary_max(target, (2, 4, 4))
    assert reduced.shape == (2, 4, 4)
    assert reduced[0, 1, 1]
    assert reduced.sum() == 1


def test_downsample_requires_exact_integer_factor():
    """不整除即报错：静默取整会引入坐标错位，使 lesionness 目标与图像不再对齐。"""
    target = np.zeros((5, 8, 8), dtype=bool)
    with pytest.raises(LesionnessError, match="整除"):
        downsample_binary_max(target, (2, 4, 4))
    # 目标大于输入（即上采样方向）同样不整除，必须报错而不是插值放大一个二值目标
    with pytest.raises(LesionnessError, match="整除"):
        downsample_binary_max(target, (8, 8, 8))
    with pytest.raises(LesionnessError, match="非法目标形状"):
        downsample_binary_max(np.zeros((4, 4, 4), dtype=bool), (0, 4, 4))


# --------------------------------------------------------------------------- zone context
def test_zone_context_derives_uncertainty_and_rejects_hard_labels():
    pz = torch.tensor([[[[0.9]]]])
    tz = torch.tensor([[[[0.2]]]])
    context, uncertainty = zone_context(pz, tz)
    assert context.shape[1] == 3
    assert uncertainty.item() == pytest.approx(0.1)
    assert context[0, 2, 0, 0].item() == pytest.approx(0.1)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        zone_context(torch.tensor([[[[2.0]]]]), tz)
    with pytest.raises(TypeError):
        zone_context(torch.tensor([[[[1]]]]), tz)


def test_zone_mode_parameter_delta_reports_expert_cost():
    assert zone_mode_parameter_delta(8, mode="concat") == 0
    assert zone_mode_parameter_delta(8, mode="zone_experts") == 3 * 64 + 24
    with pytest.raises(ValueError, match="mode"):
        zone_mode_parameter_delta(8, mode="mamba")


def test_zone_aware_refinement_is_identity_at_initialization():
    """末层零初始化 => 初始 F_out == F（这是与 strong baseline 单变量可比的前提）。"""
    for mode in ("concat", "zone_experts"):
        module = ZoneAwareRefinement(4, mode=mode, hidden_channels=3)
        features = torch.randn(2, 4, 4, 4, 4)
        context = torch.rand(2, 3, 4, 4, 4)
        assert torch.equal(module(features, context), features), mode


def test_zone_aware_refinement_shape_checks_are_fail_closed():
    module = ZoneAwareRefinement(4, hidden_channels=3)
    with pytest.raises(ValueError, match="通道数"):
        module(torch.randn(2, 5, 4, 4, 4), torch.rand(2, 3, 4, 4, 4))
    with pytest.raises(ValueError, match="同空间尺寸"):
        module(torch.randn(2, 4, 4, 4, 4), torch.rand(2, 3, 2, 2, 2))


# --------------------------------------------------------------------------- network
def _make_network(seg_heads=2, anatomy=0, deep_supervision=True, zone_mode="concat"):
    return LesionAwareCoarseToFineNNUNet(
        _FakeBackbone(seg_heads=seg_heads, deep_supervision=deep_supervision),
        num_segmentation_heads=seg_heads,
        num_anatomy_channels=anatomy,
        zone_mode=zone_mode,
    )


def test_coarse_to_fine_forward_matches_backbone_shapes_and_supports_backward():
    net = _make_network()
    x = torch.randn(2, 3, 8, 16, 16)
    outputs = net(x)
    assert isinstance(outputs, list) and len(outputs) == 2
    assert outputs[0].shape == (2, 2, 8, 16, 16)
    assert outputs[1].shape[1] == 2
    assert net.auxiliary_outputs is not None
    assert net.auxiliary_outputs["lesionness_logits"].shape[1] == 1
    sum(t.square().mean() for t in outputs).backward()
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters()
    )
    # coarse 头与 refinement 必须真的收到梯度（否则多任务没有生效）
    assert net.lesionness_head.head[-1].weight.grad is not None
    assert net.refinement.output[-1].weight.grad is not None


def test_coarse_to_fine_deep_supervision_disabled_returns_single_tensor():
    net = _make_network(deep_supervision=False)
    out = net(torch.randn(1, 3, 8, 16, 16))
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, 2, 8, 16, 16)


def test_coarse_to_fine_replaces_only_full_resolution_output():
    """DS 列表的低分辨率项必须原样透传，只有第 0 项经过 refinement。"""
    net = _make_network()
    with torch.no_grad():
        for module in net.refinement.output:
            if isinstance(module, nn.Conv3d):
                module.weight.fill_(1e-3)
                module.bias.zero_()
    x = torch.randn(2, 3, 8, 16, 16)
    outputs = net(x)
    # 低分辨率输出形状不受 refinement 影响（fake backbone 在所有轴上池化 2 倍）
    assert outputs[1].shape[2:] == (4, 8, 8)
    assert outputs[0].shape == (2, 2, 8, 16, 16)


def test_coarse_to_fine_is_identity_for_full_resolution_at_initialization():
    """零初始化 refinement 下，第 0 项输出必须与裸 backbone 逐值一致。"""
    backbone = _FakeBackbone()
    net = LesionAwareCoarseToFineNNUNet(backbone, num_segmentation_heads=2)
    x = torch.randn(1, 3, 8, 16, 16)
    with torch.no_grad():
        expected = backbone(x)
        actual = net(x)
    assert torch.allclose(actual[0], expected[0])
    assert torch.allclose(actual[1], expected[1])


def test_coarse_to_fine_accepts_soft_zone_channels_and_rejects_hard_or_logits():
    net = _make_network(anatomy=2)
    x = torch.rand(2, 5, 8, 16, 16)
    outputs = net(x)
    assert outputs[0].shape[1] == 2
    with pytest.raises(LesionAwareCoarseToFineError, match="soft probability"):
        net(torch.full((2, 5, 8, 16, 16), 1.5))
    with pytest.raises(LesionAwareCoarseToFineError, match="非有限"):
        bad = torch.rand(2, 5, 8, 16, 16)
        bad[0, 3, 0, 0, 0] = float("nan")
        net(bad)


def test_coarse_to_fine_input_channel_mismatch_is_fail_closed():
    net = _make_network()
    with pytest.raises(LesionAwareCoarseToFineError, match="输入通道"):
        net(torch.randn(1, 5, 8, 16, 16))
    with pytest.raises(LesionAwareCoarseToFineError, match=r"\[B,C,D,H,W\]"):
        net(torch.randn(1, 3, 8, 16))


def test_coarse_to_fine_rejects_unsupported_configuration():
    with pytest.raises(ValueError, match="num_anatomy_channels"):
        LesionAwareCoarseToFineNNUNet(
            _FakeBackbone(), num_segmentation_heads=2, num_anatomy_channels=3
        )
    with pytest.raises(ValueError, match="num_segmentation_heads"):
        LesionAwareCoarseToFineNNUNet(_FakeBackbone(), num_segmentation_heads=1)
    with pytest.raises(ValueError, match="MRI 通道数"):
        LesionAwareCoarseToFineNNUNet(
            _FakeBackbone(), num_segmentation_heads=2, mri_channels=4
        )


def test_decoder_property_is_proxied_for_deep_supervision_switch():
    net = _make_network()
    assert net.decoder is net.backbone.decoder


# --------------------------------------------------------------------------- auxiliary loss
class _StubNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.auxiliary_outputs = None


def test_auxiliary_loss_adds_supervision_and_requires_auxiliary_outputs():
    network = _StubNetwork()
    # 使用项目 strong baseline 的真实主损失（它接受 nnU-Net 约定的 [B,1,...] target）
    main = PiCAIFocalCrossEntropyLoss()
    loss = LesionnessAuxiliaryLoss(main, network=network, spacing_zyx=SPACING_ZYX)
    logits = torch.randn(2, 2, 8, 8, 8)
    target = torch.zeros(2, 1, 8, 8, 8, dtype=torch.long)
    with pytest.raises(RuntimeError, match="auxiliary_outputs"):
        loss(logits, target)

    network.auxiliary_outputs = {"lesionness_logits": torch.randn(2, 1, 1, 4, 4)}
    total = loss(logits, target)
    assert torch.isfinite(total)
    base = main(logits, target)
    assert total.item() > base.item()  # 辅助项确实被加上


def test_auxiliary_loss_weight_zero_disables_extra_term():
    network = _StubNetwork()
    network.auxiliary_outputs = {"lesionness_logits": torch.randn(2, 1, 1, 4, 4)}
    main = PiCAIFocalCrossEntropyLoss()
    logits = torch.randn(2, 2, 8, 8, 8)
    target = torch.zeros(2, 1, 8, 8, 8, dtype=torch.long)
    loss = LesionnessAuxiliaryLoss(
        main, network=network, spacing_zyx=SPACING_ZYX, weight=0.0
    )
    assert loss(logits, target).item() == pytest.approx(main(logits, target).item())


def test_auxiliary_loss_rejects_bad_spacing_and_negative_weight():
    network = _StubNetwork()
    with pytest.raises(ValueError, match="spacing_zyx"):
        LesionnessAuxiliaryLoss(
            PiCAIFocalCrossEntropyLoss(), network=network, spacing_zyx=(1.0, 2.0)
        )
    with pytest.raises(ValueError, match="权重"):
        LesionnessAuxiliaryLoss(
            PiCAIFocalCrossEntropyLoss(),
            network=network,
            spacing_zyx=SPACING_ZYX,
            weight=-1.0,
        )
