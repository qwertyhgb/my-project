"""plan-driven 3D U-Net（自研实现，M0–M4 共用主干）。

结构完全由 `Plan3DConfig`（即冻结的 nnUNetPlans.json / 3d_fullres）驱动：
- encoder：每级第一次卷积用该级 stride 下采样，其余卷积 stride=1，保存 skip；
- decoder：ConvTranspose3d(kernel=stride, stride=stride) 上采样 → 与 skip 中心裁剪/填充对齐 →
  拼接 → `n_conv_per_stage_decoder` 个卷积 → 该级 segmentation head；
- deep supervision：返回**最高分辨率到最低分辨率**的输出列表；关闭时只返回主输出。

不变量：
- 网络内部**不做 softmax**；
- 不使用 PZ/TZ、WG、attention/gate、Transformer、预训练权重或病灶专用分支；
- 不复制第三方实现代码，仅对齐其可验证行为（层数、kernel/stride、DS 顺序）。
"""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from ..config.plans import Plan3DConfig
from .blocks import StackedConvBlocks3d, apply_he_init, center_crop_or_pad_3d


class PlanDrivenUNet3D(nn.Module):
    """由 plan 参数驱动的 3D U-Net 主干。"""

    def __init__(
        self,
        plan: Plan3DConfig,
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

        self.plan = plan
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.deep_supervision = bool(deep_supervision)

        # ------------------------------------------------------------- encoder
        encoder_stages = []
        for i in range(plan.n_stages):
            encoder_stages.append(
                StackedConvBlocks3d(
                    in_channels=self.in_channels if i == 0 else plan.features_per_stage[i - 1],
                    out_channels=plan.features_per_stage[i],
                    kernel_size=plan.kernel_sizes[i],
                    n_convs=plan.n_conv_per_stage[i],
                    stride=plan.strides[i],
                    conv_bias=plan.conv_bias,
                    norm_eps=plan.norm_eps,
                    norm_affine=plan.norm_affine,
                    negative_slope=plan.negative_slope,
                )
            )
        self.encoder_stages = nn.ModuleList(encoder_stages)

        # ------------------------------------------------------------- decoder
        transpconvs, decoder_stages, seg_layers = [], [], []
        for s in range(1, plan.n_stages):
            skip_index = plan.n_stages - 1 - s  # s=1 → 最深一级 skip
            features_below = plan.features_per_stage[skip_index + 1]
            features_skip = plan.features_per_stage[skip_index]
            stride = plan.strides[skip_index + 1]
            transpconvs.append(
                nn.ConvTranspose3d(
                    features_below, features_skip, kernel_size=stride, stride=stride, bias=plan.conv_bias
                )
            )
            decoder_stages.append(
                StackedConvBlocks3d(
                    in_channels=2 * features_skip,
                    out_channels=features_skip,
                    kernel_size=plan.kernel_sizes[skip_index],
                    n_convs=plan.n_conv_per_stage_decoder[s - 1],
                    stride=(1, 1, 1),
                    conv_bias=plan.conv_bias,
                    norm_eps=plan.norm_eps,
                    norm_affine=plan.norm_affine,
                    negative_slope=plan.negative_slope,
                )
            )
            # 每个 decoder level 都建 head：保证 DS 打开/关闭时权重完全一致（与官方行为一致）
            seg_layers.append(nn.Conv3d(features_skip, self.num_classes, 1, 1, 0, bias=True))
        self.decoder_transpconvs = nn.ModuleList(transpconvs)
        self.decoder_stages = nn.ModuleList(decoder_stages)
        self.seg_layers = nn.ModuleList(seg_layers)

        apply_he_init(self, negative_slope=plan.negative_slope)

    # ------------------------------------------------------------------ 前向
    def set_deep_supervision(self, enabled: bool) -> None:
        self.deep_supervision = bool(enabled)

    def forward(self, x: torch.Tensor, deep_supervision: bool | None = None):
        """前向。

        Args:
            x: [B, C, D, H, W]（C 必须等于构造时的 in_channels；本层不校验通道语义）。
            deep_supervision: None 时使用 `self.deep_supervision`。

        Returns:
            Tensor [B, num_classes, D, H, W]（DS 关闭）或 list[Tensor]（DS 打开，
            按**最高分辨率 → 最低分辨率**排列）。
        """
        if x.ndim != 5:
            raise ValueError(f"输入必须是 [B, C, D, H, W]，收到 shape={tuple(x.shape)}")
        ds = self.deep_supervision if deep_supervision is None else bool(deep_supervision)

        skips: list[torch.Tensor] = []
        features = x
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

    # ------------------------------------------------------------------ 工具
    def num_parameters(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))

    def num_trainable_parameters(self) -> int:
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def configuration_summary(self) -> dict:
        return {
            "dataset_name": self.plan.dataset_name,
            "configuration": self.plan.configuration,
            "in_channels": self.in_channels,
            "num_classes": self.num_classes,
            "deep_supervision": self.deep_supervision,
            "n_stages": self.plan.n_stages,
            "patch_size": list(self.plan.patch_size),
            "spacing": list(self.plan.spacing),
            "features_per_stage": list(self.plan.features_per_stage),
            "strides": [list(s) for s in self.plan.strides],
            "kernel_sizes": [list(k) for k in self.plan.kernel_sizes],
            "n_conv_per_stage": list(self.plan.n_conv_per_stage),
            "n_conv_per_stage_decoder": list(self.plan.n_conv_per_stage_decoder),
            "stage_shapes": [list(s) for s in self.plan.stage_shapes()],
            "decoder_output_shapes": [list(s) for s in self.plan.decoder_output_shapes()],
            "num_parameters": self.num_parameters(),
        }

    def expected_output_shapes(self, patch_size: Sequence[int] | None = None) -> tuple[tuple[int, int, int], ...]:
        """DS 输出的期望空间尺寸（高→低）。patch_size 只允许等于 plan 的 patch（保证可追溯）。"""
        if patch_size is not None and tuple(int(v) for v in patch_size) != tuple(self.plan.patch_size):
            raise ValueError(f"patch_size={tuple(patch_size)} 与 plan 的 {tuple(self.plan.patch_size)} 不一致")
        return self.plan.decoder_output_shapes()


__all__ = ["PlanDrivenUNet3D"]
