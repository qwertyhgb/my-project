"""自研 3D 卷积基础块（M0–M4 共用）。

对齐冻结 plan 的 arch_kwargs：

    Conv3d(conv_bias=True) -> InstanceNorm3d(eps=1e-5, affine=True) -> LeakyReLU(0.01)

odd kernel 使用 same padding；权重初始化使用 Kaiming/He（`a = negative_slope`）。
所有模块的接口只依赖 `torch`，不依赖 nnU-Net / dynamic_network_architectures。
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

# 固定超参（与 nnUNetPlans.json 的 norm_op_kwargs / nonlin_kwargs 对齐）
LEAKY_SLOPE: float = 0.01
NORM_EPS: float = 1e-5
NORM_AFFINE: bool = True
CONV_BIAS: bool = True


def same_padding(kernel_size: Sequence[int]) -> tuple[int, ...]:
    """odd kernel 的 same padding：k // 2（偶数核直接报错，避免静默改变语义）。"""
    if any(k <= 0 for k in kernel_size):
        raise ValueError(f"kernel_size 必须为正，收到 {tuple(kernel_size)}")
    if any(k % 2 == 0 for k in kernel_size):
        raise ValueError(f"仅支持 odd kernel（same padding 需要），收到 {tuple(kernel_size)}")
    return tuple(k // 2 for k in kernel_size)


class ConvNormAct3d(nn.Module):
    """Conv3d -> InstanceNorm3d -> LeakyReLU（自研基础块）。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int] = (3, 3, 3),
        stride: Sequence[int] = (1, 1, 1),
        *,
        conv_bias: bool = CONV_BIAS,
        norm_eps: float = NORM_EPS,
        norm_affine: bool = NORM_AFFINE,
        negative_slope: float = LEAKY_SLOPE,
    ) -> None:
        super().__init__()
        kernel_size = tuple(int(k) for k in kernel_size)
        stride = tuple(int(s) for s in stride)
        if len(kernel_size) != 3 or len(stride) != 3:
            raise ValueError(f"kernel_size/stride 必须为长度 3，收到 {kernel_size} / {stride}")
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=same_padding(kernel_size), bias=conv_bias
        )
        self.norm = nn.InstanceNorm3d(out_channels, eps=norm_eps, affine=norm_affine)
        self.act = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class StackedConvBlocks3d(nn.Module):
    """n_convs 个 ConvNormAct3d；**第一个**卷积使用给定 stride 完成下采样/等尺寸，其余 stride=1。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int],
        n_convs: int,
        stride: Sequence[int],
        *,
        conv_bias: bool = CONV_BIAS,
        norm_eps: float = NORM_EPS,
        norm_affine: bool = NORM_AFFINE,
        negative_slope: float = LEAKY_SLOPE,
    ) -> None:
        super().__init__()
        if n_convs < 1:
            raise ValueError(f"n_convs 必须 >= 1，收到 {n_convs}")
        blocks = [
            ConvNormAct3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                conv_bias=conv_bias,
                norm_eps=norm_eps,
                norm_affine=norm_affine,
                negative_slope=negative_slope,
            )
        ]
        for _ in range(n_convs - 1):
            blocks.append(
                ConvNormAct3d(
                    out_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=(1, 1, 1),
                    conv_bias=conv_bias,
                    norm_eps=norm_eps,
                    norm_affine=norm_affine,
                    negative_slope=negative_slope,
                )
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


def center_crop_or_pad_3d(x: torch.Tensor, target_shape: Sequence[int], pad_value: float = 0.0) -> torch.Tensor:
    """把 [B, C, D, H, W] 用**中心裁剪或中心 padding** 对齐到 target_shape。

    该函数是 encoder skip 与 decoder 上采样结果之间的唯一对齐策略：
    ConvTranspose3d 在奇数尺寸上可能比 skip 多/少 1 个体素，这里显式处理，
    不使用插值掩盖结构性错误（尺寸差超过 x 的一半时直接报错，避免掩盖严重 bug）。
    """
    target = tuple(int(t) for t in target_shape)
    if len(target) != 3:
        raise ValueError(f"target_shape 必须为长度 3，收到 {target_shape}")
    cur = tuple(x.shape[-3:])
    if cur == target:
        return x
    for d in range(3):
        diff = target[d] - cur[d]
        if diff == 0:
            continue
        if abs(diff) > 2:
            raise ValueError(
                f"dim{d}: 需要对齐的尺寸差为 {diff}（当前 {cur[d]} → 目标 {target[d]}）；"
                "差异超过 2 体素说明 encoder/decoder 尺寸推导存在结构性错误，拒绝静默裁剪/填充"
            )
        if diff < 0:  # 裁剪
            start = (-diff) // 2
            x = x.narrow(-3 + d, start, target[d])
        else:  # padding
            before = diff // 2
            after = diff - before
            pad = [0, 0, 0, 0, 0, 0]  # (W_left, W_right, H_left, H_right, D_left, D_right)
            pad[2 * (2 - d)] = before
            pad[2 * (2 - d) + 1] = after
            x = F.pad(x, pad, mode="constant", value=pad_value)
    return x


def init_weights_he(module: nn.Module, negative_slope: float = LEAKY_SLOPE) -> None:
    """Kaiming/He 初始化（fan_in, leaky_relu）；conv bias 置零，InstanceNorm 仿射参数置 1/0。"""
    if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
        nn.init.kaiming_normal_(module.weight, a=negative_slope, mode="fan_in", nonlinearity="leaky_relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.InstanceNorm3d):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def apply_he_init(model: nn.Module, negative_slope: float = LEAKY_SLOPE) -> None:
    model.apply(lambda m: init_weights_he(m, negative_slope=negative_slope))


__all__ = [
    "CONV_BIAS",
    "LEAKY_SLOPE",
    "NORM_AFFINE",
    "NORM_EPS",
    "ConvNormAct3d",
    "StackedConvBlocks3d",
    "apply_he_init",
    "center_crop_or_pad_3d",
    "init_weights_he",
    "same_padding",
]
