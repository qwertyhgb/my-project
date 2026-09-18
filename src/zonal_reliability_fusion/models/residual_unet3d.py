"""v2.3 plan-driven Residual-Encoder 3D U-Net（自研；M0–M4 共同骨干）。

结构由**冻结 plan**（`Plan3DConfig` = nnUNetPlans.json / 3d_fullres）与**版本化架构配置**
（`ArchitectureSpec`）共同驱动，二者职责严格分离（research_plan §7.2/§7.3/§9.4.2）：

- 来自冻结 plan：spacing、patch、stage 数、每级 kernel/stride、features_per_stage、
  decoder 的 ``n_conv_per_stage_decoder``、conv/norm/nonlin 超参、deep-supervision 输出尺度与顺序；
- 来自版本化架构配置：encoder=residual、每 stage 的 residual block 数、post-activation 顺序、
  shortcut/projection 规则、stem、初始化、decoder 身份、§8.6 浅层 skip 契约 id、融合 stage、架构哈希。

不变量（与 legacy `PlanDrivenUNet3D` 相同的对外契约）：

- encoder：default stem（1 conv，不下采样）→ 每级 `StackedResidualBlocks3d`（首块承担 stride/通道变化），保存 skip；
- decoder：**普通 U-Net**，`ConvTranspose3d(kernel=stride,stride=stride) → center_crop_or_pad → concat skip →
  stacked conv → 每级 seg head`；
- deep supervision：返回**最高分辨率 → 最低分辨率**的输出列表；关闭时只返回主输出；
- 网络内部**不做 softmax**；M0 **不读取** PZ/TZ、WG、gate；不 import nnunetv2；不复制第三方代码。
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from ..config.architecture import (
    ACTIVATION_ORDER_POST,
    BLOCK_TYPE_BASIC,
    ENCODER_TYPE_RESIDUAL,
    ArchitectureSpec,
    validate_architecture_against_plan,
)
from ..config.plans import Plan3DConfig
from .blocks import StackedConvBlocks3d, apply_he_init, center_crop_or_pad_3d
from .residual_blocks import ResidualBlock3d, StackedResidualBlocks3d


def zero_init_residual_last_norms(module: nn.Module) -> int:
    """把每个 `ResidualBlock3d` 加法前的最后一个 norm（conv2 的 IN）仿射参数置 0；返回处理的块数。

    对齐全官方 residual encoder 的 ``init_last_bn_before_add_to_0``（残差块初始接近恒等）。
    """
    count = 0
    for m in module.modules():
        if isinstance(m, ResidualBlock3d) and m.zero_init_last_norm:
            if m.conv2_norm.weight is not None:
                nn.init.zeros_(m.conv2_norm.weight)
            if m.conv2_norm.bias is not None:
                nn.init.zeros_(m.conv2_norm.bias)
            count += 1
    return count


class PlanDrivenResidualEncoderUNet3D(nn.Module):
    """plan-driven Residual-Encoder 3D U-Net 主干（v2.3 正式 M0–M4 骨干）。"""

    def __init__(
        self,
        plan: Plan3DConfig,
        spec: ArchitectureSpec,
        in_channels: int,
        num_classes: int,
        *,
        deep_supervision: bool = True,
    ) -> None:
        super().__init__()
        if in_channels < 1:
            raise ValueError(f"in_channels 必须 >= 1，收到 {in_channels}")
        if num_classes < 2:
            raise ValueError(f"num_classes 必须 >= 2，收到 {num_classes}")
        # 结构字段必须与冻结 plan 完全一致（features/kernel/stride/n_conv_decoder/算子），否则拒绝构建
        validate_architecture_against_plan(spec, plan)
        if spec.encoder_type != ENCODER_TYPE_RESIDUAL:
            raise ValueError(f"本类只构建 residual encoder，收到 encoder_type={spec.encoder_type!r}")
        if spec.block_type != BLOCK_TYPE_BASIC or spec.activation_order != ACTIVATION_ORDER_POST:
            raise ValueError(
                f"本类只实现 {BLOCK_TYPE_BASIC}/{ACTIVATION_ORDER_POST}，收到 "
                f"{spec.block_type!r}/{spec.activation_order!r}"
            )
        if spec.in_channels != in_channels or spec.num_classes != num_classes:
            raise ValueError(
                f"in_channels/num_classes 与架构配置不一致：模型=({in_channels},{num_classes}) "
                f"配置=({spec.in_channels},{spec.num_classes})"
            )

        self.plan = plan
        self.spec = spec
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.deep_supervision = bool(deep_supervision)
        self.blocks_per_stage = tuple(spec.blocks_per_stage)

        norm_eps = plan.norm_eps
        norm_affine = plan.norm_affine
        negative_slope = plan.negative_slope
        conv_bias = plan.conv_bias
        zero_init_last_norm = bool(spec.initialization.get("zero_init_last_norm_before_add", True))

        # ------------------------------------------------------------- stem（1 conv，不下采样）
        if spec.stem["n_convs"] != 1 or tuple(spec.stem["stride"]) != (1, 1, 1) or spec.stem["does_downsample"]:
            raise ValueError("stem 必须为 1 个卷积且不下采样（对齐官方 default stem）")
        self.stem = StackedConvBlocks3d(
            in_channels=self.in_channels,
            out_channels=plan.features_per_stage[0],
            kernel_size=plan.kernel_sizes[0],
            n_convs=1,
            stride=(1, 1, 1),
            conv_bias=conv_bias,
            norm_eps=norm_eps,
            norm_affine=norm_affine,
            negative_slope=negative_slope,
        )

        # ------------------------------------------------------------- residual encoder
        encoder_stages = []
        for i in range(plan.n_stages):
            in_ch = plan.features_per_stage[0] if i == 0 else plan.features_per_stage[i - 1]
            encoder_stages.append(
                StackedResidualBlocks3d(
                    n_blocks=self.blocks_per_stage[i],
                    in_channels=in_ch,
                    out_channels=plan.features_per_stage[i],
                    kernel_size=plan.kernel_sizes[i],
                    stride=plan.strides[i],
                    conv_bias=conv_bias,
                    norm_eps=norm_eps,
                    norm_affine=norm_affine,
                    negative_slope=negative_slope,
                    zero_init_last_norm=zero_init_last_norm,
                )
            )
        self.encoder_stages = nn.ModuleList(encoder_stages)

        # ------------------------------------------------------------- 普通 U-Net decoder（沿用冻结 plan）
        transpconvs, decoder_stages, seg_layers = [], [], []
        for s in range(1, plan.n_stages):
            skip_index = plan.n_stages - 1 - s  # s=1 → 最深一级 skip
            features_below = plan.features_per_stage[skip_index + 1]
            features_skip = plan.features_per_stage[skip_index]
            stride = plan.strides[skip_index + 1]
            transpconvs.append(
                nn.ConvTranspose3d(
                    features_below, features_skip, kernel_size=stride, stride=stride, bias=conv_bias
                )
            )
            decoder_stages.append(
                StackedConvBlocks3d(
                    in_channels=2 * features_skip,
                    out_channels=features_skip,
                    kernel_size=plan.kernel_sizes[skip_index],
                    n_convs=plan.n_conv_per_stage_decoder[s - 1],
                    stride=(1, 1, 1),
                    conv_bias=conv_bias,
                    norm_eps=norm_eps,
                    norm_affine=norm_affine,
                    negative_slope=negative_slope,
                )
            )
            # 每级都建 head：保证 DS 打开/关闭时权重完全一致（与官方/legacy 行为一致）
            seg_layers.append(nn.Conv3d(features_skip, self.num_classes, 1, 1, 0, bias=True))
        self.decoder_transpconvs = nn.ModuleList(transpconvs)
        self.decoder_stages = nn.ModuleList(decoder_stages)
        self.seg_layers = nn.ModuleList(seg_layers)

        # 全局 He 初始化后，重新把残差块加法前的最后一个 norm 置 0（apply_he_init 会把 IN 复位为 1）
        apply_he_init(self, negative_slope=negative_slope)
        self._zero_init_blocks = zero_init_residual_last_norms(self)

    # ------------------------------------------------------------------ 前向
    def set_deep_supervision(self, enabled: bool) -> None:
        self.deep_supervision = bool(enabled)

    def forward(self, x: torch.Tensor, deep_supervision: bool | None = None):
        """前向；DS 打开返回**高→低**分辨率输出列表，关闭返回单个主输出。"""
        if x.ndim != 5:
            raise ValueError(f"输入必须是 [B, C, D, H, W]，收到 shape={tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"输入通道必须为 {self.in_channels}，收到 {x.shape[1]}")
        ds = self.deep_supervision if deep_supervision is None else bool(deep_supervision)

        skips: list[torch.Tensor] = []
        features = self.stem(x)
        for stage in self.encoder_stages:
            features = stage(features)
            skips.append(features)

        outputs: list[torch.Tensor] = []
        for idx in range(self.plan.n_decoder_stages):
            skip_index = self.plan.n_stages - 2 - idx
            upsampled = self.decoder_transpconvs[idx](features)
            skip = skips[skip_index]
            upsampled = center_crop_or_pad_3d(upsampled, skip.shape[-3:])
            features = self.decoder_stages[idx](torch.cat((upsampled, skip), dim=1))
            if ds:
                outputs.append(self.seg_layers[idx](features))

        if ds:
            return outputs[::-1]  # 高分辨率在前
        return self.seg_layers[-1](features)

    # ------------------------------------------------------------------ 身份 / 工具
    def architecture_identity(self) -> dict:
        return self.spec.identity()

    def num_parameters(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))

    def num_trainable_parameters(self) -> int:
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def num_residual_blocks(self) -> int:
        return int(sum(1 for _ in self._residual_block_modules()))

    def _residual_block_modules(self):
        for m in self.modules():
            if isinstance(m, ResidualBlock3d):
                yield m

    def encoder_stage_shapes(self) -> tuple[tuple[int, int, int], ...]:
        """各 encoder stage 输出的空间尺寸（= plan.stage_shapes()；stem 不下采样）。"""
        return self.plan.stage_shapes()

    def configuration_summary(self) -> dict:
        ident = self.spec.identity()
        return {
            "dataset_name": self.plan.dataset_name,
            "configuration": self.plan.configuration,
            "encoder_type": self.spec.encoder_type,
            "block_type": self.spec.block_type,
            "activation_order": self.spec.activation_order,
            "architecture_name": ident["architecture_name"],
            "architecture_version": ident["architecture_version"],
            "architecture_sha256": ident["architecture_sha256"],
            "in_channels": self.in_channels,
            "num_classes": self.num_classes,
            "deep_supervision": self.deep_supervision,
            "n_stages": self.plan.n_stages,
            "patch_size": list(self.plan.patch_size),
            "spacing": list(self.plan.spacing),
            "features_per_stage": list(self.plan.features_per_stage),
            "strides": [list(s) for s in self.plan.strides],
            "kernel_sizes": [list(k) for k in self.plan.kernel_sizes],
            "blocks_per_stage": list(self.blocks_per_stage),
            "n_conv_per_stage_decoder": list(self.plan.n_conv_per_stage_decoder),
            "fusion_stage": self.spec.fusion_stage,
            "shallow_skip_contract_id": self.spec.shallow_skip_contract_id,
            "stage_shapes": [list(s) for s in self.plan.stage_shapes()],
            "decoder_output_shapes": [list(s) for s in self.plan.decoder_output_shapes()],
            "num_residual_blocks": self.num_residual_blocks(),
            "num_parameters": self.num_parameters(),
        }

    def expected_output_shapes(self, patch_size: Sequence[int] | None = None) -> tuple[tuple[int, int, int], ...]:
        """DS 输出的期望空间尺寸（高→低）。patch_size 只允许等于 plan 的 patch（保证可追溯）。"""
        if patch_size is not None and tuple(int(v) for v in patch_size) != tuple(self.plan.patch_size):
            raise ValueError(f"patch_size={tuple(patch_size)} 与 plan 的 {tuple(self.plan.patch_size)} 不一致")
        return self.plan.decoder_output_shapes()


__all__ = ["PlanDrivenResidualEncoderUNet3D", "zero_init_residual_last_norms"]
