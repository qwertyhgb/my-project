"""v2.3 正式 M0：T2W + ADC + HBV 普通通道拼接 → plan-driven Residual-Encoder 3D U-Net。

M0 的边界（research_plan §7.2/§9.4.2、任务书三/六）：

- 输入固定为 ``[B, 3, D, H, W]``，channel 0/1/2 = T2W/ADC/HBV，直接 concat 单流输入；
- **没有**序列特异性 stem、attention/gate、PZ/TZ、WG、Transformer/Mamba、预训练权重或病灶专用分支；
- 网络内部不做 softmax（logits 直接交给损失）；
- 与 legacy `M0ConcatModel` 的唯一差异是**主干 encoder 从 PlainConv 升级为 residual**（decoder/DS/skip 对齐
  冻结 plan，保持一致），以便干净归因；
- 暴露稳定**架构身份**（name/version/sha256），供 checkpoint/resume 严格隔离（legacy↔residual 互相拒绝）。

M1–M4 将在同一骨干上于 encoder stage 2 插入 §8.6 的浅层 skip 聚合与融合模块；P2A **不实现** 它们。
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from ..config.architecture import ArchitectureSpec
from ..config.plans import Plan3DConfig
from .m0_concat import MODALITY_ORDER
from .residual_unet3d import PlanDrivenResidualEncoderUNet3D

#: M0 明确**不接受**的输入（PZ/TZ、WG、gate 权重/ logits、decoder 特征等）；仅用于测试与文档，
#: forward 不接收这些参数，传入即触发 TypeError。
FORBIDDEN_INPUTS: tuple[str, ...] = (
    "zonal",
    "pz",
    "tz",
    "pz_tz",
    "wg",
    "gate",
    "gate_weight",
    "sequence_logits",
    "anatomy",
)


class M0ResidualConcatModel(nn.Module):
    """v2.3 M0：三序列 concat + plan-driven Residual-Encoder 3D U-Net。"""

    #: M0 不读取任何解剖/gate 信息（§7.2）
    ACCEPTS_ANATOMY_INPUTS = False

    def __init__(
        self,
        plan: Plan3DConfig,
        spec: ArchitectureSpec,
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
        self.backbone_id = "residual_encoder_v23"
        self.modality_order = tuple(modality_order)
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.deep_supervision = bool(deep_supervision)
        self.plan = plan
        self.spec = spec
        self.net = PlanDrivenResidualEncoderUNet3D(
            plan,
            spec,
            in_channels=self.in_channels,
            num_classes=self.num_classes,
            deep_supervision=self.deep_supervision,
        )

    # ------------------------------------------------------------------ 前向
    def forward(self, x: torch.Tensor, deep_supervision: bool | None = None):
        """前向。只接受影像张量；传入 zonal/wg/gate 等关键字会触发 TypeError（M0 不读取解剖信息）。"""
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
        """从（可能的）DS 输出中取最高分辨率 logits ``[B, num_classes, D, H, W]``。"""
        if isinstance(output, (list, tuple)):
            if not output:
                raise ValueError("空的 deep-supervision 输出列表")
            return output[0]
        return output

    # ------------------------------------------------------------------ 身份 / 工具
    def architecture_identity(self) -> dict:
        """checkpoint/resume 架构身份（name/version/sha256/encoder_type）。"""
        return self.net.architecture_identity()

    def num_parameters(self) -> int:
        return self.net.num_parameters()

    def num_trainable_parameters(self) -> int:
        return self.net.num_trainable_parameters()

    def summary(self) -> dict:
        info = self.net.configuration_summary()
        info.update(
            {
                "model_id": self.model_id,
                "backbone_id": self.backbone_id,
                "modality_order": list(self.modality_order),
                "accepts_anatomy_inputs": self.ACCEPTS_ANATOMY_INPUTS,
            }
        )
        return info

    def format_summary(self) -> str:
        """人类可读摘要（setup / 架构摘要脚本使用）。"""
        info = self.summary()
        lines = [
            f"model_id            : {info['model_id']}（backbone={info['backbone_id']}）",
            f"architecture        : {info['architecture_name']} v{info['architecture_version']}",
            f"architecture_sha256 : {info['architecture_sha256']}",
            f"encoder_type        : {info['encoder_type']}（{info['block_type']} / {info['activation_order']}）",
            f"dataset/configuration: {info['dataset_name']} / {info['configuration']}",
            f"input               : [B, {info['in_channels']}, *patch]  modality={info['modality_order']}",
            f"patch_size          : {info['patch_size']}  spacing={info['spacing']}",
            f"n_stages            : {info['n_stages']}",
            f"features_per_stage  : {info['features_per_stage']}",
            f"blocks_per_stage    : {info['blocks_per_stage']}（residual blocks；来自版本化架构配置）",
            f"strides             : {info['strides']}",
            f"kernel_sizes        : {info['kernel_sizes']}",
            f"n_conv_per_stage_decoder: {info['n_conv_per_stage_decoder']}（普通 U-Net decoder，沿用 plan）",
            f"stage_shapes        : {info['stage_shapes']}",
            f"DS output shapes    : {info['decoder_output_shapes']}（高→低）",
            f"fusion_stage        : {info['fusion_stage']}  shallow_skip_contract={info['shallow_skip_contract_id']}",
            f"deep_supervision    : {info['deep_supervision']}",
            f"num_classes         : {info['num_classes']}（背景/病灶，网络内不做 softmax）",
            f"num_residual_blocks : {info['num_residual_blocks']}",
            f"num_parameters      : {info['num_parameters']:,}",
        ]
        return "\n".join(lines)


__all__ = ["FORBIDDEN_INPUTS", "M0ResidualConcatModel"]
