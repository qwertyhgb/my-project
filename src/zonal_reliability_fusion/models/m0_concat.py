"""M0：三序列普通 concat 基线（自研 nnU-Net-inspired 3D U-Net）。

M0 的边界（P2-A 明确要求）：
- 输入固定为 [B, 3, D, H, W]，channel 0/1/2 = T2W/ADC/HBV；
- 只做通道拼接后进入共享主干，**没有**序列特异性 stem、attention/gate、PZ/TZ、WG、
  Transformer/Mamba、预训练权重或病灶专用分支；
- 网络内部不做 softmax（logits 直接交给损失）。

M1–M4 将以同一 plan 驱动主干替换/包裹 input projection 层，因此本类只暴露最小接口：
`forward`、`main_logits`、`num_parameters`、`summary`。
"""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from ..config.plans import Plan3DConfig
from .unet3d import PlanDrivenUNet3D

#: 固定模态顺序（不得交换；与 nnU-Net 通道 0000/0001/0002 对应）
MODALITY_ORDER: tuple[str, str, str] = ("T2W", "ADC", "HBV")


class M0ConcatModel(nn.Module):
    """M0：T2W + ADC + HBV 三通道 concat 基线。"""

    def __init__(
        self,
        plan: Plan3DConfig,
        *,
        in_channels: int = 3,
        num_classes: int = 2,
        deep_supervision: bool = True,
        modality_order: Sequence[str] = MODALITY_ORDER,
    ) -> None:
        super().__init__()
        if tuple(modality_order) != MODALITY_ORDER:
            raise ValueError(
                f"M0 要求固定模态顺序 {list(MODALITY_ORDER)}（channel 0/1/2），收到 {list(modality_order)}"
            )
        if in_channels != len(MODALITY_ORDER):
            raise ValueError(f"M0 要求 in_channels={len(MODALITY_ORDER)}（T2W/ADC/HBV），收到 {in_channels}")

        self.model_id = "M0"
        self.modality_order = tuple(modality_order)
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.deep_supervision = bool(deep_supervision)
        self.plan = plan
        self.net = PlanDrivenUNet3D(
            plan, in_channels=self.in_channels, num_classes=self.num_classes, deep_supervision=self.deep_supervision
        )

    # ------------------------------------------------------------------ 前向
    def forward(self, x: torch.Tensor, deep_supervision: bool | None = None):
        """前向。训练/验证使用 DS 列表，推理调用方应传 `deep_supervision=False`（或使用 `main_logits`）。"""
        if x.ndim != 5:
            raise ValueError(f"M0 输入必须是 [B, 3, D, H, W]，收到 shape={tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"M0 输入通道必须为 {self.in_channels}（T2W/ADC/HBV），收到 {x.shape[1]}；"
                "模态顺序固定为 " + "/".join(self.modality_order)
            )
        return self.net(x, deep_supervision=deep_supervision)

    def set_deep_supervision(self, enabled: bool) -> None:
        self.deep_supervision = bool(enabled)
        self.net.set_deep_supervision(enabled)

    @staticmethod
    def main_logits(output: torch.Tensor | Sequence[torch.Tensor]) -> torch.Tensor:
        """从（可能的）DS 输出中取最高分辨率 logits [B, num_classes, D, H, W]。"""
        if isinstance(output, (list, tuple)):
            if not output:
                raise ValueError("空的 deep-supervision 输出列表")
            return output[0]
        return output

    # ------------------------------------------------------------------ 工具
    def num_parameters(self) -> int:
        return self.net.num_parameters()

    def summary(self) -> dict:
        info = self.net.configuration_summary()
        info.update({"model_id": self.model_id, "modality_order": list(self.modality_order)})
        return info

    def format_summary(self) -> str:
        """人类可读摘要（setup 检查脚本使用）。"""
        info = self.summary()
        lines = [
            f"model_id            : {info['model_id']}",
            f"dataset/configuration: {info['dataset_name']} / {info['configuration']}",
            f"input               : [B, {info['in_channels']}, *patch]  modality={info['modality_order']}",
            f"patch_size          : {info['patch_size']}  spacing={info['spacing']}",
            f"n_stages            : {info['n_stages']}",
            f"features_per_stage  : {info['features_per_stage']}",
            f"strides             : {info['strides']}",
            f"kernel_sizes        : {info['kernel_sizes']}",
            f"n_conv_per_stage    : {info['n_conv_per_stage']} / decoder {info['n_conv_per_stage_decoder']}",
            f"stage_shapes        : {info['stage_shapes']}",
            f"DS output shapes    : {info['decoder_output_shapes']}（高→低）",
            f"deep_supervision    : {info['deep_supervision']}",
            f"num_classes         : {info['num_classes']}（背景/病灶，网络内不做 softmax）",
            f"num_parameters      : {info['num_parameters']:,}",
        ]
        return "\n".join(lines)


__all__ = ["MODALITY_ORDER", "M0ConcatModel"]
