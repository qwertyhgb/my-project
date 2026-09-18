"""v2.3 残差块合成测试（对应任务书九·1–6；仅合成张量、强制 CPU、不读真实数据）。

覆盖：identity shortcut、通道 projection、stride projection、stride+通道同时变化、奇数尺寸、
He 初始化与「加法前最后一个 norm 置 0」的冻结初始化规则，以及 post-activation 语义。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from zonal_reliability_fusion.models.residual_blocks import (
    ResidualBlock3d,
    StackedResidualBlocks3d,
)


def _has(module: nn.Module, kind: type) -> bool:
    return any(isinstance(m, kind) for m in module.modules())


# ------------------------------------------------------------------ 1. identity shortcut
def test_identity_shortcut_when_same_channels_and_stride():
    blk = ResidualBlock3d(4, 4, (3, 3, 3), (1, 1, 1))
    assert isinstance(blk.skip, nn.Identity)
    assert blk.has_stride is False and blk.requires_projection is False
    x = torch.randn(2, 4, 6, 6, 6)
    assert tuple(blk(x).shape) == (2, 4, 6, 6, 6)


def test_post_activation_and_zero_init_make_block_identity_at_init():
    """zero_init_last_norm=True 时 conv2 的 IN 输出恒 0 → out = LeakyReLU(identity_skip(x))（post-activation 证据）。"""
    blk = ResidualBlock3d(3, 3, (3, 3, 3), (1, 1, 1), zero_init_last_norm=True)
    assert torch.count_nonzero(blk.conv2_norm.weight) == 0
    x = torch.randn(2, 3, 5, 5, 5)
    out = blk(x)
    assert torch.allclose(out, F.leaky_relu(x, negative_slope=0.01), atol=1e-6)


def test_zero_init_disabled_keeps_norm_weight_one():
    blk = ResidualBlock3d(3, 3, (3, 3, 3), (1, 1, 1), zero_init_last_norm=False)
    assert torch.allclose(blk.conv2_norm.weight, torch.ones_like(blk.conv2_norm.weight))


# ------------------------------------------------------------------ 2. 通道 projection shortcut
def test_channel_projection_shortcut_no_implicit_crop():
    blk = ResidualBlock3d(4, 8, (3, 3, 3), (1, 1, 1))  # stride 1，仅通道变化
    assert blk.requires_projection is True and blk.has_stride is False
    assert isinstance(blk.skip, nn.Sequential)
    convs = [m for m in blk.skip if isinstance(m, nn.Conv3d)]
    assert len(convs) == 1 and tuple(convs[0].kernel_size) == (1, 1, 1)
    assert convs[0].in_channels == 4 and convs[0].out_channels == 8  # 显式投影，非裁剪
    assert convs[0].bias is None  # projection_conv_bias=false
    assert _has(blk.skip, nn.InstanceNorm3d)  # projection_norm=true
    assert _has(blk.skip, nn.AvgPool3d) is False  # 无 stride → 不含 pool
    x = torch.randn(2, 4, 6, 6, 6)
    assert tuple(blk(x).shape) == (2, 8, 6, 6, 6)


# ------------------------------------------------------------------ 3. stride projection shortcut
def test_stride_shortcut_uses_avgpool_when_channels_equal():
    blk = ResidualBlock3d(8, 8, (3, 3, 3), (2, 2, 2))  # 仅 stride 变化
    assert blk.has_stride is True and blk.requires_projection is False
    assert isinstance(blk.skip, nn.Sequential)
    pools = [m for m in blk.skip if isinstance(m, nn.AvgPool3d)]
    assert len(pools) == 1 and pools[0].kernel_size == (2, 2, 2) and pools[0].stride == (2, 2, 2)
    assert not _has(blk.skip, nn.Conv3d)  # 通道相同 → 无 1×1×1 投影
    x = torch.randn(2, 8, 8, 8, 8)
    assert tuple(blk(x).shape) == (2, 8, 4, 4, 4)


# ------------------------------------------------------------------ 4. stride + 通道同时变化
def test_stride_and_channel_change_avgpool_then_projection():
    blk = ResidualBlock3d(4, 8, (3, 3, 3), (1, 2, 2))  # ResNet-D：先 avgpool 再 1×1×1 conv
    assert blk.has_stride is True and blk.requires_projection is True
    ops = list(blk.skip)
    assert isinstance(ops[0], nn.AvgPool3d)  # downsample_before_projection=true
    assert isinstance(ops[1], nn.Conv3d) and tuple(ops[1].kernel_size) == (1, 1, 1)
    assert isinstance(ops[2], nn.InstanceNorm3d)
    x = torch.randn(2, 4, 8, 8, 8)
    out = blk(x)
    assert tuple(out.shape) == (2, 8, 8, 4, 4)  # stride (1,2,2)
    assert torch.isfinite(out).all()


# ------------------------------------------------------------------ 5. 奇数尺寸
def test_odd_spatial_sizes_stride_downsample_matches_conv():
    blk = ResidualBlock3d(4, 4, (3, 3, 3), (2, 2, 2))
    x = torch.randn(1, 4, 7, 9, 5)  # 奇数尺寸
    out = blk(x)
    # 主路 conv1(stride2, same pad) 与 shortcut avgpool(stride2) 的输出尺寸必须一致，可相加
    expected = tuple((s - 1) // 2 + 1 for s in (7, 9, 5))  # (4,5,3)
    assert tuple(out.shape[-3:]) == expected
    assert torch.isfinite(out).all()


def test_stacked_blocks_first_has_stride_rest_identity():
    stack = StackedResidualBlocks3d(3, in_channels=4, out_channels=8, kernel_size=(3, 3, 3), stride=(2, 2, 2))
    blocks = list(stack.blocks)
    assert len(blocks) == 3
    assert blocks[0].has_stride is True and blocks[0].requires_projection is True
    for b in blocks[1:]:
        assert b.has_stride is False and b.requires_projection is False and isinstance(b.skip, nn.Identity)
    x = torch.randn(2, 4, 8, 8, 8)
    assert tuple(stack(x).shape) == (2, 8, 4, 4, 4)


# ------------------------------------------------------------------ 6. 初始化规则
def test_he_init_conv_and_bias_and_norm():
    blk = ResidualBlock3d(4, 8, (3, 3, 3), (1, 1, 1))
    # conv 权重非零（He/Kaiming），conv bias 置 0
    assert torch.count_nonzero(blk.conv1.conv.weight) > 0
    assert torch.count_nonzero(blk.conv2_conv.weight) > 0
    assert torch.count_nonzero(blk.conv1.conv.bias) == 0
    assert torch.count_nonzero(blk.conv2_conv.bias) == 0
    # conv1 的 IN（非加法前最后一个）weight=1、bias=0
    assert torch.allclose(blk.conv1.norm.weight, torch.ones_like(blk.conv1.norm.weight))
    assert torch.count_nonzero(blk.conv1.norm.bias) == 0
    # 加法前最后一个 norm（conv2 的 IN）被置 0
    assert torch.count_nonzero(blk.conv2_norm.weight) == 0
    assert torch.count_nonzero(blk.conv2_norm.bias) == 0


def test_projection_conv_bias_false_but_main_conv_bias_true():
    blk = ResidualBlock3d(4, 8, (3, 3, 3), (1, 2, 2), conv_bias=True)
    proj = next(m for m in blk.skip if isinstance(m, nn.Conv3d))
    assert proj.bias is None  # projection 1×1×1 conv 恒无 bias（对齐官方 BasicBlockD）
    assert blk.conv1.conv.bias is not None and blk.conv2_conv.bias is not None


def test_gradient_flows_through_main_and_shortcut():
    blk = ResidualBlock3d(4, 8, (3, 3, 3), (1, 2, 2), zero_init_last_norm=False)
    x = torch.randn(1, 4, 6, 6, 6, requires_grad=True)
    blk(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    proj = next(m for m in blk.skip if isinstance(m, nn.Conv3d))
    assert proj.weight.grad is not None  # shortcut 也参与反传
