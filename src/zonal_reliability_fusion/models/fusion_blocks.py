"""M1–M4 融合组件（浅层分支投影、浅层 skip 聚合器、image gate、zone encoder、条件仿射头、解剖注入）。

设计边界（任务书四/五/七）：

- 所有组件只依赖 ``torch``，不 import nnunetv2；算子超参（norm eps/affine、LeakyReLU 斜率、conv bias）
  一律由冻结 plan 传入，不在组件内写死；
- **浅层 skip 聚合器**只允许读取三路 modality-specific 特征，构造时即调用
  ``assert_shallow_skip_inputs_allowed`` 守卫（§8.6 契约），不得读取 PZ/TZ、WG、gate 权重/logits、decoder 特征；
- **image gate** 的 logits 只来自三路 stage 特征（不读取任何解剖输入）；最后一层零初始化 →
  初始 softmax 为严格等权 (1/3,1/3,1/3)；
- **zone encoder** 只接受 2 通道 (PZ, TZ) fractional occupancy；WG（3 通道）必须被拒绝；M3 与 M4 使用
  相同宽度/深度/融合尺度（由架构配置机器强制）；
- **条件仿射头**最后（产生 gamma/beta 的）一层零初始化 → 初始为恒等调制
  ``L = (1 + tanh(0)) * L_img + 0 = L_img``；
- gate 权重/条件调制量只是结构量，**不得**解释为临床或因果贡献。
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .blocks import LEAKY_SLOPE, NORM_AFFINE, NORM_EPS, same_padding
from .residual_blocks import ResidualBlock3d

#: 浅层 skip 聚合器声明的输入名（只能是三路 modality-specific 特征）
SHALLOW_SKIP_READS: tuple[str, ...] = ("t2w_stage_feature", "adc_stage_feature", "hbv_stage_feature")
#: PZ/TZ 先验的通道语义（channel 0 = PZ，channel 1 = TZ）；WG 不得作为输入
ZONE_PRIOR_CHANNELS: tuple[str, ...] = ("pz", "tz")
#: 融合模态数（T2W/ADC/HBV）
N_MODALITIES = 3


class ModalityProjection(nn.Module):
    """融合前逐模态投影：``Conv3d(C→C,1×1×1) → InstanceNorm3d → LeakyReLU``。"""

    def __init__(
        self,
        channels: int,
        *,
        conv_bias: bool = True,
        norm_eps: float = NORM_EPS,
        norm_affine: bool = NORM_AFFINE,
        negative_slope: float = LEAKY_SLOPE,
    ) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError(f"channels 必须 >= 1，收到 {channels}")
        self.channels = int(channels)
        self.conv = nn.Conv3d(channels, channels, kernel_size=1, stride=1, padding=0, bias=conv_bias)
        self.norm = nn.InstanceNorm3d(channels, eps=norm_eps, affine=norm_affine)
        self.act = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ShallowSkipAggregator(nn.Module):
    """§8.6 浅层 skip 聚合：``concat(三路) → 1×1×1 Conv → InstanceNorm3d → LeakyReLU``。

    输出通道 = 冻结 plan 的 ``features_per_stage[stage]``（由构造参数 target_channels 指定）。
    构造时执行契约守卫：声明的读取输入只能是三路 modality-specific 特征。
    """

    def __init__(
        self,
        in_channels_per_modality: int,
        target_channels: int,
        *,
        stage: int,
        reads: Sequence[str] = SHALLOW_SKIP_READS,
        conv_bias: bool = True,
        norm_eps: float = NORM_EPS,
        norm_affine: bool = NORM_AFFINE,
        negative_slope: float = LEAKY_SLOPE,
    ) -> None:
        super().__init__()
        # §8.6 契约运行时守卫：读入名命中 forbidden_inputs 立即报错
        from ..config.architecture import assert_shallow_skip_inputs_allowed

        assert_shallow_skip_inputs_allowed(reads)
        if tuple(reads) != SHALLOW_SKIP_READS:
            raise ValueError(
                f"浅层 skip 聚合器只允许读取 {list(SHALLOW_SKIP_READS)}，收到 {list(reads)}"
            )
        if stage not in (0, 1):
            raise ValueError(f"浅层 skip 聚合器只作用于 stage 0/1（§8.6），收到 stage={stage}")
        if in_channels_per_modality < 1 or target_channels < 1:
            raise ValueError("通道数必须 >= 1")
        self.stage = int(stage)
        self.reads = tuple(reads)
        self.in_channels_per_modality = int(in_channels_per_modality)
        self.target_channels = int(target_channels)
        self.conv = nn.Conv3d(
            N_MODALITIES * in_channels_per_modality,
            target_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=conv_bias,
        )
        self.norm = nn.InstanceNorm3d(target_channels, eps=norm_eps, affine=norm_affine)
        self.act = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)

    def forward(self, f_t2w: torch.Tensor, f_adc: torch.Tensor, f_hbv: torch.Tensor) -> torch.Tensor:
        if f_t2w.shape[1] != self.in_channels_per_modality:
            raise ValueError(
                f"浅层 skip 输入通道必须为 {self.in_channels_per_modality}，收到 {f_t2w.shape[1]}"
            )
        if f_adc.shape[1] != self.in_channels_per_modality or f_hbv.shape[1] != self.in_channels_per_modality:
            raise ValueError("三路浅层 skip 输入的通道数必须一致（结构相同、参数独立的三路分支）")
        return self.act(self.norm(self.conv(torch.cat((f_t2w, f_adc, f_hbv), dim=1))))


class ImageDrivenGate(nn.Module):
    """图像驱动空间序列 gate：**输入必须是三路 modality projection 输出 U_m**。

    拓扑（与版本化架构配置逐值一致）::

        concat([U_T2W, U_ADC, U_HBV]) → 1×1×1 Conv(3C→C) → InstanceNorm3d → LeakyReLU
                                      → 1×1×1 Conv(C→3) → logits（逐体素 3 模态）

    ``L_img`` 就是这里的 logits；权重 = softmax(L, dim=modality)。
    最后一层零初始化 → 初始 logits 全 0 → softmax 为严格等权 (1/3, 1/3, 1/3)，
    即 M2/M3/M4 的初始状态与 M1 的等权融合一致（便于归因与稳定性）。
    """

    def __init__(
        self,
        channels: int,
        *,
        hidden_channels: int | None = None,
        conv_bias: bool = True,
        norm_eps: float = NORM_EPS,
        norm_affine: bool = NORM_AFFINE,
        negative_slope: float = LEAKY_SLOPE,
        last_layer_zero_init: bool = True,
    ) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError(f"channels 必须 >= 1，收到 {channels}")
        hidden = int(hidden_channels or channels)
        self.channels = int(channels)
        self.hidden_channels = hidden
        self.proj = nn.Conv3d(
            N_MODALITIES * channels, hidden, kernel_size=1, stride=1, padding=0, bias=conv_bias
        )
        self.norm = nn.InstanceNorm3d(hidden, eps=norm_eps, affine=norm_affine)
        self.act = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        self.logits = nn.Conv3d(hidden, N_MODALITIES, kernel_size=1, stride=1, padding=0, bias=True)
        self.last_layer_zero_init = bool(last_layer_zero_init)
        if self.last_layer_zero_init:
            self.zero_init_last_layer()

    def zero_init_last_layer(self) -> None:
        """把 logits 层的权重与 bias 置 0（初始等权融合）。"""
        nn.init.zeros_(self.logits.weight)
        if self.logits.bias is not None:
            nn.init.zeros_(self.logits.bias)

    def forward(self, f_t2w: torch.Tensor, f_adc: torch.Tensor, f_hbv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 (gate_logits, gate_weights)；两者形状均为 ``[B, 3, *stage2_spatial]``。"""
        for name, f in (("T2W", f_t2w), ("ADC", f_adc), ("HBV", f_hbv)):
            if f.shape[1] != self.channels:
                raise ValueError(f"{name} stage 特征通道必须为 {self.channels}，收到 {f.shape[1]}")
        fused = torch.cat((f_t2w, f_adc, f_hbv), dim=1)
        logits = self.logits(self.act(self.norm(self.proj(fused))))
        weights = torch.softmax(logits, dim=1)
        return logits, weights


class ZoneEncoder(nn.Module):
    """PZ/TZ fractional occupancy → zone feature ``E_z``（M3 与 M4 共享同一宽度/深度/融合尺度）。

    输入为 **patch 空间**的 2 通道先验（channel 0 = PZ，channel 1 = TZ，取值 [0,1]）；
    内部先 ``adaptive_avg_pool3d`` 到融合 stage 空间尺寸（均值即 fractional occupancy），
    再经 conv stem + residual blocks 编码。WG（3 通道）在模型入口即被拒绝。

    拓扑（与版本化架构配置逐值一致）::

        prior[2] → adaptive_avg_pool3d(stage2 空间)
                 → 3×3×3 Conv(2→F, same padding) → InstanceNorm3d → LeakyReLU
                 → n_blocks × [1×1×1? 否] 标准 post-activation 残差块（Conv-IN-LeakyReLU → Conv-IN → +shortcut → LeakyReLU）

    残差块的 ``zero_init_last_norm`` 固定为 False（E_z 需要从初始就携带先验信息）。
    """

    def __init__(
        self,
        features: int,
        *,
        n_blocks: int = 2,
        kernel_size: Sequence[int] = (3, 3, 3),
        conv_bias: bool = True,
        norm_eps: float = NORM_EPS,
        norm_affine: bool = NORM_AFFINE,
        negative_slope: float = LEAKY_SLOPE,
    ) -> None:
        super().__init__()
        if features < 1 or n_blocks < 1:
            raise ValueError(f"features/n_blocks 必须 >= 1，收到 features={features} n_blocks={n_blocks}")
        kernel = tuple(int(k) for k in kernel_size)
        if any(k % 2 == 0 for k in kernel):
            raise ValueError(f"zone encoder kernel 必须为 odd（same padding），收到 {kernel}")
        self.features = int(features)
        self.n_blocks = int(n_blocks)
        self.kernel_size = kernel
        self.input_channels = len(ZONE_PRIOR_CHANNELS)
        self.stem = nn.Conv3d(
            self.input_channels, features, kernel_size=kernel, stride=1, padding=same_padding(kernel), bias=conv_bias
        )
        self.stem_norm = nn.InstanceNorm3d(features, eps=norm_eps, affine=norm_affine)
        self.stem_act = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        # zone encoder 的残差块不做「加法前 norm 置 0」：E_z 需要从初始就携带先验信息
        self.blocks = nn.Sequential(
            *[
                ResidualBlock3d(
                    features,
                    features,
                    kernel,
                    (1, 1, 1),
                    conv_bias=conv_bias,
                    norm_eps=norm_eps,
                    norm_affine=norm_affine,
                    negative_slope=negative_slope,
                    zero_init_last_norm=False,
                )
                for _ in range(n_blocks)
            ]
        )

    def adapt_prior(self, prior: torch.Tensor, target_shape: Sequence[int]) -> torch.Tensor:
        """把 patch 空间先验降到融合 stage 空间（均值池化 = fractional occupancy 语义）。"""
        target = tuple(int(t) for t in target_shape)
        if tuple(prior.shape[-3:]) == target:
            return prior
        # adaptive_avg_pool3d 要求 5D（ND 支持但显式 5D 更安全，且不掩盖 batch 错误）
        return F.adaptive_avg_pool3d(prior, target)

    def forward(self, prior: torch.Tensor, target_shape: Sequence[int]) -> torch.Tensor:
        x = self.adapt_prior(prior, target_shape)
        return self.blocks(self.stem_act(self.stem_norm(self.stem(x))))


class ConditionalAffineHead(nn.Module):
    """PZ/TZ 条件头：``E_z → gamma, beta``（逐体素，各 3 通道 = 每个模态一组）。

    拓扑（与版本化架构配置逐值一致）::

        E_z → 1×1×1 Conv(F→F) → InstanceNorm3d → LeakyReLU → 1×1×1 Conv(F→6) → split(gamma, beta)

    使用 ``L = (1 + tanh(gamma)) * L_img + beta``（有界调制：``1 + tanh`` ∈ (0, 2)）。
    产生 gamma/beta 的最后一层零初始化 → 初始 ``gamma = beta = 0`` → ``L = L_img``（恒等调制）。
    """

    def __init__(
        self,
        features: int,
        *,
        hidden_channels: int | None = None,
        conv_bias: bool = True,
        norm_eps: float = NORM_EPS,
        norm_affine: bool = NORM_AFFINE,
        negative_slope: float = LEAKY_SLOPE,
        head_last_layer_zero_init: bool = True,
    ) -> None:
        super().__init__()
        if features < 1:
            raise ValueError(f"features 必须 >= 1，收到 {features}")
        hidden = int(hidden_channels or features)
        self.features = int(features)
        self.hidden_channels = hidden
        self.proj = nn.Conv3d(features, hidden, kernel_size=1, stride=1, padding=0, bias=conv_bias)
        self.norm = nn.InstanceNorm3d(hidden, eps=norm_eps, affine=norm_affine)
        self.act = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        # 2 * 3 = gamma(3) + beta(3)
        self.head = nn.Conv3d(hidden, 2 * N_MODALITIES, kernel_size=1, stride=1, padding=0, bias=True)
        self.head_last_layer_zero_init = bool(head_last_layer_zero_init)
        if self.head_last_layer_zero_init:
            self.zero_init_head()

    def zero_init_head(self) -> None:
        nn.init.zeros_(self.head.weight)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, zone_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.head(self.act(self.norm(self.proj(zone_features))))
        gamma, beta = torch.split(out, N_MODALITIES, dim=1)
        return gamma, beta


class ZoneFusionProjection(nn.Module):
    """M3 的「普通解剖输入」投影：**严格** ``F_out = Q([F_fuse, E_z])``（无隐式残差相加）。

    拓扑：``concat(F_fuse, E_z) → 1×1×1 Conv(C+F → C) → InstanceNorm3d → LeakyReLU``。

    与 M4 的区别：解剖信息**不进入 gate**，只在融合完成后作为普通特征**替换**共享输入
    （用于区分「普通增加解剖信息」与「解剖直接控制序列 gate」）。M3/M4 使用相同 zone encoder。
    """

    def __init__(
        self,
        feature_channels: int,
        zone_features: int,
        *,
        conv_bias: bool = True,
        norm_eps: float = NORM_EPS,
        norm_affine: bool = NORM_AFFINE,
        negative_slope: float = LEAKY_SLOPE,
    ) -> None:
        super().__init__()
        if feature_channels < 1 or zone_features < 1:
            raise ValueError("feature_channels/zone_features 必须 >= 1")
        self.feature_channels = int(feature_channels)
        self.zone_features = int(zone_features)
        self.conv = nn.Conv3d(
            feature_channels + zone_features, feature_channels, kernel_size=1, stride=1, padding=0, bias=conv_bias
        )
        self.norm = nn.InstanceNorm3d(feature_channels, eps=norm_eps, affine=norm_affine)
        self.act = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)

    def forward(self, fused: torch.Tensor, zone_features: torch.Tensor) -> torch.Tensor:
        """返回 ``Q([F_fuse, E_z])``（不含 ``F_fuse +`` 残差项）。"""
        if fused.shape[1] != self.feature_channels:
            raise ValueError(f"F_fuse 通道必须为 {self.feature_channels}，收到 {fused.shape[1]}")
        if zone_features.shape[1] != self.zone_features:
            raise ValueError(f"zone feature 通道必须为 {self.zone_features}，收到 {zone_features.shape[1]}")
        if tuple(fused.shape[-3:]) != tuple(zone_features.shape[-3:]):
            raise ValueError(
                f"F_fuse 与 zone feature 的空间尺寸必须一致（融合尺度相同）："
                f"{tuple(fused.shape[-3:])} vs {tuple(zone_features.shape[-3:])}"
            )
        return self.act(self.norm(self.conv(torch.cat((fused, zone_features), dim=1))))


def validate_image_tensor(image: torch.Tensor, expected_channels: int, *, context: str = "") -> None:
    """影像张量的**结构性**检查（shape/通道；不做 finite/范围检查，供 CPU 入口或 debug 模式使用）。"""
    prefix = f"{context}: " if context else ""
    if image.ndim != 5:
        raise ValueError(f"{prefix}影像必须是 [B, C, D, H, W]，收到 shape={tuple(image.shape)}")
    if image.shape[1] != expected_channels:
        raise ValueError(
            f"{prefix}影像通道必须为 {expected_channels}（T2W/ADC/HBV），收到 {image.shape[1]}"
        )


def validate_zonal_prior(
    prior: torch.Tensor,
    *,
    image_shape: Sequence[int] | None = None,
    tol: float = 1e-4,
    check_finite: bool = True,
    context: str = "",
) -> None:
    """PZ/TZ prior 的**完整**契约检查（shape/有限性/逐通道 [0,1]/PZ+TZ<=1+tol/WG 拒绝）。

    这是 CPU 数据入口或显式 debug-validation 使用的检查；**训练热路径默认不调用**，
    避免每个 iteration 触发 GPU 同步（`bool(tensor.all())` / `float(tensor.min())` 等）。
    """
    prefix = f"{context}: " if context else ""
    if prior.ndim != 5:
        raise ValueError(f"{prefix}zonal prior 必须是 [B, 2, D, H, W]，收到 shape={tuple(prior.shape)}")
    if prior.shape[1] != len(ZONE_PRIOR_CHANNELS):
        raise ValueError(
            f"{prefix}zonal prior 通道必须为 {len(ZONE_PRIOR_CHANNELS)}（{list(ZONE_PRIOR_CHANNELS)}）；"
            f"收到 {prior.shape[1]}。WG（3 通道）或其他第三通道一律拒绝。"
        )
    if image_shape is not None:
        if tuple(prior.shape[0:1]) != tuple(image_shape[0:1]):
            raise ValueError(
                f"{prefix}prior batch={prior.shape[0]} 与 image batch={image_shape[0]} 不一致"
            )
        if tuple(prior.shape[-3:]) != tuple(image_shape[-3:]):
            raise ValueError(
                f"{prefix}prior 必须与 image 处于同一 patch 空间/网格：prior={tuple(prior.shape[-3:])} "
                f"image={tuple(image_shape[-3:])}（裁剪/翻转/旋转/缩放必须同步）"
            )
    if check_finite and not bool(torch.isfinite(prior).all()):
        raise ValueError(f"{prefix}zonal prior 含非有限值（NaN/Inf）")
    lo = float(prior.min())
    hi = float(prior.max())
    if lo < -tol or hi > 1.0 + tol:
        raise ValueError(
            f"{prefix}zonal prior 必须为 [0,1] fractional occupancy，收到范围 [{lo}, {hi}]；"
            "原始标签 0/1/2 必须先转换为 2 通道 one-hot/occupancy。"
        )
    pz = prior[:, 0]
    tz = prior[:, 1]
    if bool((pz < -tol).any()) or bool((tz < -tol).any()):
        raise ValueError(f"{prefix}PZ/TZ 通道存在负值（必须为 [0,1] occupancy）")
    overlap = float((pz + tz).max()) - 1.0
    if overlap > tol:
        raise ValueError(
            f"{prefix}PZ 与 TZ 重叠：max(PZ+TZ)={float((pz + tz).max()):.6f} > 1+{tol}；"
            "每个体素最多属于一个分区（background/PZ/TZ）。"
        )


def prior_occupancy_stats(prior: torch.Tensor) -> dict[str, float]:
    """prior 的少量标量统计（CPU/统计量诊断模式使用；不保留完整张量）。"""
    with torch.no_grad():
        pz = prior[:, 0]
        tz = prior[:, 1]
        return {
            "pz_mean": float(pz.mean()),
            "tz_mean": float(tz.mean()),
            "pz_max": float(pz.max()),
            "tz_max": float(tz.max()),
            "overlap_max": float((pz + tz).max()),
        }


__all__ = [
    "N_MODALITIES",
    "SHALLOW_SKIP_READS",
    "ZONE_PRIOR_CHANNELS",
    "ConditionalAffineHead",
    "ImageDrivenGate",
    "ModalityProjection",
    "ShallowSkipAggregator",
    "ZoneEncoder",
    "ZoneFusionProjection",
    "prior_occupancy_stats",
    "validate_image_tensor",
    "validate_zonal_prior",
]
