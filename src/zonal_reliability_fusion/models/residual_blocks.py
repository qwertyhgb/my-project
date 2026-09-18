"""v2.3 残差基础块（plan-driven Residual-Encoder M0–M4 共用；自研实现，只依赖 torch）。

对齐 `dynamic_network_architectures.building_blocks.residual.BasicBlockD` 的**可验证行为**
（不复制第三方代码，也不 import 它），并按 research_plan §7.3 冻结：

- 基本算子：``Conv3d + InstanceNorm3d + LeakyReLU``（odd kernel + same padding，与冻结 plan 的 arch_kwargs 一致）；
- **post-activation** 残差块（ResNet Option-B 语义）::

      out = conv2(conv1(x))            # conv1: Conv→IN→LeakyReLU ; conv2: Conv→IN（无激活）
      out = out + shortcut(x)
      return LeakyReLU(out)            # 加法之后再激活

- shortcut 规则（ResNet-D 风格，禁止隐式裁剪通道）：
  - 空间尺寸与通道**均相同** → ``identity``；
  - 有 stride → 先 ``AvgPool3d(kernel=stride, stride=stride, ceil_mode=True)`` 再（若通道变化）``1×1×1 Conv``；
  - 通道变化 → ``1×1×1 Conv(bias=False) → InstanceNorm3d``（显式 projection，绝不裁剪通道）；

  **奇数尺寸适配（相对官方 BasicBlockD 的必要偏差）**：odd kernel + same padding 的 strided conv 输出为
  ``ceil(in/stride)``，而 ``AvgPool(kernel=stride, stride=stride)`` 默认（``ceil_mode=False``）输出 ``floor(in/stride)``，
  奇数尺寸下二者相差 1 导致残差无法相加。官方依赖 nnU-Net patch 可被 stride 整除来回避该问题；本项目按
  research_plan §7.3 要求**覆盖奇数空间尺寸**，故 shortcut 的 AvgPool 使用 ``ceil_mode=True``。对可整除尺寸
  （冻结 3d_fullres plan 的全部 stage）``ceil_mode`` 与默认**逐尺寸等价**，不改变真实 plan 的任何 shape。
- 初始化：He/Kaiming（fan_in, leaky_relu, a=negative_slope），conv bias=0，InstanceNorm 仿射 weight=1/bias=0；
  并把**每个残差块加法前的最后一个 InstanceNorm（conv2 的 norm）仿射参数置 0**
  （对齐官方 ``init_last_bn_before_add_to_0``，使残差块初始接近恒等，训练更稳定）。

覆盖奇数空间尺寸：same padding + AvgPool(stride) 与 ConvTranspose 的对齐由 decoder 侧
``center_crop_or_pad_3d`` 兜底（见 residual_unet3d）。
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from .blocks import (
    LEAKY_SLOPE,
    NORM_AFFINE,
    NORM_EPS,
    ConvNormAct3d,
    apply_he_init,
    same_padding,
)


class ResidualBlock3d(nn.Module):
    """post-activation 残差块（``BasicBlockD`` 等价行为）。

    Args:
        in_channels / out_channels: 主路输入/输出通道。
        kernel_size: 主路两个卷积的 odd kernel（same padding）。
        stride: **仅第一个卷积与 shortcut** 使用；第二个卷积恒 stride=1。
        conv_bias / norm_eps / norm_affine / negative_slope: 与冻结 plan 对齐的算子超参。
        zero_init_last_norm: 是否把加法前最后一个 norm（conv2 的 IN）仿射参数置 0（默认 True）。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int],
        stride: Sequence[int],
        *,
        conv_bias: bool = True,
        norm_eps: float = NORM_EPS,
        norm_affine: bool = NORM_AFFINE,
        negative_slope: float = LEAKY_SLOPE,
        zero_init_last_norm: bool = True,
    ) -> None:
        super().__init__()
        kernel_size = tuple(int(k) for k in kernel_size)
        stride = tuple(int(s) for s in stride)
        if len(kernel_size) != 3 or len(stride) != 3:
            raise ValueError(f"kernel_size/stride 必须长度 3，收到 {kernel_size} / {stride}")
        if in_channels < 1 or out_channels < 1:
            raise ValueError(f"通道数必须 >= 1，收到 in={in_channels} out={out_channels}")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.stride = stride
        self.negative_slope = float(negative_slope)
        self.zero_init_last_norm = bool(zero_init_last_norm)

        # conv1: Conv→IN→LeakyReLU（带 stride）；复用 blocks.ConvNormAct3d
        self.conv1 = ConvNormAct3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            conv_bias=conv_bias,
            norm_eps=norm_eps,
            norm_affine=norm_affine,
            negative_slope=negative_slope,
        )
        # conv2: Conv→IN（**无激活**），stride 恒 1
        self.conv2_conv = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size,
            stride=(1, 1, 1),
            padding=same_padding(kernel_size),
            bias=conv_bias,
        )
        self.conv2_norm = nn.InstanceNorm3d(out_channels, eps=norm_eps, affine=norm_affine)
        # 加法后的激活（post-activation）
        self.act = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)

        self.has_stride = any(s != 1 for s in stride)
        self.requires_projection = in_channels != out_channels

        # shortcut：identity 或（AvgPool if stride）+（1×1×1 Conv(bias=False)→IN if 通道变化）
        if self.has_stride or self.requires_projection:
            ops: list[nn.Module] = []
            if self.has_stride:
                # ceil_mode=True：使 shortcut 的空间尺寸与 odd-kernel + same-padding 的 strided conv 一致（覆盖奇数尺寸）
                ops.append(nn.AvgPool3d(stride, stride, ceil_mode=True))
            if self.requires_projection:
                ops.append(nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False))
                ops.append(nn.InstanceNorm3d(out_channels, eps=norm_eps, affine=norm_affine))
            self.skip: nn.Module = nn.Sequential(*ops)
        else:
            self.skip = nn.Identity()

        self._init_weights(norm_affine=norm_affine)

    def _init_weights(self, *, norm_affine: bool) -> None:
        apply_he_init(self, negative_slope=self.negative_slope)
        if self.zero_init_last_norm and norm_affine:
            # 加法前最后一个 norm（conv2 的 IN）置 0：残差块初始接近恒等（对齐官方 init_last_bn_before_add_to_0）
            if self.conv2_norm.weight is not None:
                nn.init.zeros_(self.conv2_norm.weight)
            if self.conv2_norm.bias is not None:
                nn.init.zeros_(self.conv2_norm.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = self.conv2_norm(self.conv2_conv(self.conv1(x)))
        out = out + residual  # 不用 inplace，避免与 residual 的 autograd 冲突
        return self.act(out)


class StackedResidualBlocks3d(nn.Module):
    """一个 encoder stage：``n_blocks`` 个 `ResidualBlock3d`。

    **第一个** block 承担该 stage 的 stride 与通道变化（in_channels→out_channels）；
    其余 block 恒 stride=1、in==out（identity shortcut）。与官方 `StackedResidualBlocks` 语义一致。
    """

    def __init__(
        self,
        n_blocks: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int],
        stride: Sequence[int],
        *,
        conv_bias: bool = True,
        norm_eps: float = NORM_EPS,
        norm_affine: bool = NORM_AFFINE,
        negative_slope: float = LEAKY_SLOPE,
        zero_init_last_norm: bool = True,
    ) -> None:
        super().__init__()
        if n_blocks < 1:
            raise ValueError(f"n_blocks 必须 >= 1，收到 {n_blocks}")
        common = {
            "conv_bias": conv_bias,
            "norm_eps": norm_eps,
            "norm_affine": norm_affine,
            "negative_slope": negative_slope,
            "zero_init_last_norm": zero_init_last_norm,
        }
        blocks = [ResidualBlock3d(in_channels, out_channels, kernel_size, stride, **common)]
        for _ in range(n_blocks - 1):
            blocks.append(ResidualBlock3d(out_channels, out_channels, kernel_size, (1, 1, 1), **common))
        self.blocks = nn.Sequential(*blocks)
        self.n_blocks = int(n_blocks)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.stride = tuple(int(s) for s in stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


__all__ = ["ResidualBlock3d", "StackedResidualBlocks3d"]
