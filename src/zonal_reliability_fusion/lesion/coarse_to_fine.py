"""Lesion-Aware Coarse-to-Fine 网络（新主线条件 C 的方法主体）。

整体逻辑
--------
::

    T2W / ADC / HBV  +  predicted anatomy (soft)
              │
              ▼
        原生 nnU-Net backbone（共享 encoder + decoder，**不重写**）
              │
              ├──▶ coarse lesionness 头（最粗尺度）──▶ 高 sensitivity 的定位证据
              │
              └──▶ 全分辨率分割输出
                        │  + 上采样的 lesionness 概率（soft guidance）
                        │  + soft 解剖上下文 P(PZ)/P(TZ)/P(U)
                        ▼
                  轻量 refinement（末层零初始化）
                        ▼
                  final lesion segmentation

为什么可以「不重写 nnU-Net」
----------------------------
- **backbone 完全由** ``nnunetv2.utilities.get_network_from_plans`` 按 ``nnUNetPlans.json``
  构建。本模块只做**包装**：读它的输出、在其中一处加一个 1×1×1 头、在另一端加一个
  1×1×1 refinement。不复制 encoder/decoder，不复制训练循环、optimizer、LR scheduler、
  checkpoint、验证器或滑窗推理。
- coarse 头的输入取 backbone 的**最粗尺度输出**：deep supervision 打开时它是输出列表的最后
  一项；关闭时退化为单张量（此时 coarse 头作用在同一张量上，仍与最终输出共享全部权重）。

四条硬约束（逐条对应 Research Plan 的 coarse-to-fine 原则）
-----------------------------------------------------------
1. **coarse 预测不能硬裁掉任何区域**：本网络从不对输入或特征做空间裁剪，lesionness 只以
   ``sigmoid`` 概率形式进入 refinement（soft guidance）；
2. **lesionness 是 soft guidance**：它不是 attention mask，也不做阈值化；
3. **零 coarse score 区域仍保留 residual path**：refinement 是 ``F_out = F + delta``，
   ``delta`` 末层零初始化。即使 coarse 头给出 0，``F`` 仍然原样通过；
4. **防止 coarse 头错误导致无法恢复**：由于 (3) 的残差结构，初始状态与 strong baseline
   **逐值一致**；训练过程中即使 coarse 头误差较大，最终分割仍有完整的分割分支路径。

辅助输出的传递方式（需要如实说明的限制）
----------------------------------------
nnU-Net 的 ``validation_step`` 会对网络返回值直接 ``argmax``，因此网络**必须**只返回分割输出。
lesionness logits 通过模块属性 :attr:`auxiliary_outputs` 传递，由
:class:`~zonal_reliability_fusion.nnunet.trainers.LesionnessAuxiliaryLoss` 在同一个
``train_step`` 内立即读取。

**限制**：这条通道依赖「同一进程内 forward 后立即取用」，因此只适用于本项目当前的单进程、
单 GPU 配置（``num_gpus=1``）。若将来改用 DDP，必须改为显式返回元组并自定义
``train_step``/``validation_step``——本项目**不**采用那种做法，因为那会开始重写训练循环。
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from zonal_reliability_fusion.lesion.zone_conditioning import (
    ZONE_MODES,
    ZoneAwareRefinement,
    zone_context,
)

#: 后处理通道数：lesionness 概率 1 + 解剖上下文 3（P(PZ), P(TZ), P(U)）
LESIONNESS_GUIDANCE_CHANNELS = 1
ANATOMY_CONTEXT_CHANNELS = 3


class LesionAwareCoarseToFineError(RuntimeError):
    """coarse-to-fine 前向或配置中的可预期失败（fail-closed）。"""


class CoarseLesionnessHead(nn.Module):
    """最粗尺度上的 1×1×1 lesionness 头（极小；只多两个 1×1×1 卷积）。

    末层零初始化：初始时 lesionness logits 恒为 0，``sigmoid(0) = 0.5`` 的**常数**概率进入
    refinement，而 refinement 末层同样零初始化，因此初始整体行为与 strong baseline 一致。
    """

    def __init__(self, in_channels: int, hidden_channels: int = 8) -> None:
        super().__init__()
        in_channels = int(in_channels)
        hidden_channels = int(hidden_channels)
        if in_channels <= 0 or hidden_channels <= 0:
            raise ValueError("in_channels / hidden_channels 必须为正")
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.head = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, 1),
            nn.InstanceNorm3d(hidden_channels, affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(hidden_channels, 1, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 5 or features.shape[1] != self.in_channels:
            raise LesionAwareCoarseToFineError(
                f"coarse 头期望 [B,{self.in_channels},D,H,W]，收到 {tuple(features.shape)}"
            )
        return self.head(features)


class LesionAwareCoarseToFineNNUNet(nn.Module):
    """原生 backbone + coarse lesionness 头 + soft refinement 的薄包装。

    ``num_anatomy_channels``：

    - ``0``——不做解剖条件化（条件 C 的无区域对照；输入只有 3 个 MRI 通道）；
    - ``2``——额外输入 ``P(PZ) / P(TZ)`` 两个 soft 概率通道，派生 ``P(U)`` 作为第三通道
      上下文的来源（条件 D）。

    ``zone_mode`` 只在 ``num_anatomy_channels == 2`` 时生效，取值见
    :data:`..zone_conditioning.ZONE_MODES`。
    """

    def __init__(
        self,
        backbone: nn.Module,
        *,
        num_segmentation_heads: int,
        mri_channels: int = 3,
        num_anatomy_channels: int = 0,
        lesionness_hidden_channels: int = 8,
        refinement_hidden_channels: int = 8,
        zone_mode: str = "concat",
        use_lesionness_guidance: bool = True,
    ) -> None:
        super().__init__()
        mri_channels = int(mri_channels)
        num_anatomy_channels = int(num_anatomy_channels)
        num_segmentation_heads = int(num_segmentation_heads)
        if mri_channels != 3:
            raise ValueError(
                f"MRI 通道数固定为 3（T2W/ADC/HBV），收到 {mri_channels}"
            )
        if num_anatomy_channels not in (0, 2):
            raise ValueError(
                "num_anatomy_channels 只能是 0（无解剖条件化）或 2（P(PZ)/P(TZ)）；"
                f"收到 {num_anatomy_channels}"
            )
        if num_segmentation_heads < 2:
            raise ValueError(
                f"num_segmentation_heads 至少为 2（background + lesion），收到 {num_segmentation_heads}"
            )
        if zone_mode not in ZONE_MODES:
            raise ValueError(f"zone_mode 必须是 {ZONE_MODES} 之一，收到 {zone_mode!r}")
        self.mri_channels = mri_channels
        self.num_anatomy_channels = num_anatomy_channels
        self.total_input_channels = mri_channels + num_anatomy_channels
        self.num_segmentation_heads = num_segmentation_heads
        self.zone_mode = zone_mode
        self.use_lesionness_guidance = bool(use_lesionness_guidance)

        self.backbone = backbone
        # 原生 decoder 的返回张量已经是 ``seg_layers`` 的输出，因此
        # 最粗尺度输出与全分辨率输出的通道数**都**等于 num_segmentation_heads
        # （见 dynamic_network_architectures.architectures.unet.UNetDecoder.forward）。
        # 这里显式接收该通道数而不是从 backbone 里"猜"最后一个 Conv3d——猜测会在 backbone
        # 结构变化时静默产生错误形状。
        self.lesionness_head = CoarseLesionnessHead(
            num_segmentation_heads, hidden_channels=lesionness_hidden_channels
        )
        context_channels = (
            LESIONNESS_GUIDANCE_CHANNELS if self.use_lesionness_guidance else 0
        ) + (ANATOMY_CONTEXT_CHANNELS if self.num_anatomy_channels else 0)
        if context_channels == 0:
            raise ValueError(
                "refinement 的 context 为空：至少要启用 lesionness guidance 或解剖条件化之一"
            )
        self.context_channels = context_channels
        self.refinement = ZoneAwareRefinement(
            num_segmentation_heads,
            mode=zone_mode if self.num_anatomy_channels else "concat",
            hidden_channels=refinement_hidden_channels,
            context_channels=context_channels,
        )
        #: 辅助输出通道；只由训练期的 loss 读取（见模块 docstring 的「限制」）
        self.auxiliary_outputs: dict[str, Tensor] | None = None

    @property
    def decoder(self):
        """代理到 ``backbone.decoder``，使 ``set_deep_supervision_enabled`` 原样可用。"""
        return self.backbone.decoder

    # ------------------------------------------------------------------ 前向
    def _split_input(self, x: Tensor) -> tuple[Tensor, Tensor | None]:
        if x.ndim != 5:
            raise LesionAwareCoarseToFineError(
                f"输入必须为 [B,C,D,H,W]，收到 {tuple(x.shape)}"
            )
        if x.shape[1] != self.total_input_channels:
            raise LesionAwareCoarseToFineError(
                f"该网络要求 {self.total_input_channels} 个输入通道"
                f"（{self.mri_channels} MRI + {self.num_anatomy_channels} anatomy），"
                f"收到 {x.shape[1]}"
            )
        mri = x[:, : self.mri_channels]
        if self.num_anatomy_channels == 0:
            return mri, None
        anatomy = x[:, self.mri_channels :]
        if not bool(torch.isfinite(anatomy).all()):
            raise LesionAwareCoarseToFineError("anatomy 概率含非有限值")
        if bool((anatomy < 0).any()) or bool((anatomy > 1).any()):
            raise LesionAwareCoarseToFineError(
                "anatomy 通道必须是 soft probability（值域 [0,1]）；"
                "不接受 logits 或硬 one-hot"
            )
        return mri, anatomy

    def clear_auxiliary_outputs(self) -> None:
        """清空辅助输出；训练步开始时由 Trainer 调用，避免误用上一 batch 的张量。"""
        self.auxiliary_outputs = None

    def forward(self, x: Tensor):
        """返回**分割输出**（与原生 backbone 完全同形），并填充 :attr:`auxiliary_outputs`。

        返回值形状与原始 backbone 逐值一致：DS 关闭时为单张量，开启时为等长列表（第 0 项是
        全分辨率输出）。
        """
        mri, anatomy = self._split_input(x)
        outputs = self.backbone(mri)
        deep_supervision = isinstance(outputs, (list, tuple))
        full_resolution = outputs[0] if deep_supervision else outputs
        coarsest = outputs[-1] if deep_supervision else outputs

        for name, tensor in (("full-res output", full_resolution), ("coarsest output", coarsest)):
            if tensor.shape[1] != self.num_segmentation_heads:
                raise LesionAwareCoarseToFineError(
                    f"backbone 的 {name} 通道数为 {tensor.shape[1]}，"
                    f"与声明的 num_segmentation_heads={self.num_segmentation_heads} 不一致；"
                    "拒绝在错误形状上继续"
                )

        lesionness_logits = self.lesionness_head(coarsest)

        context_parts: list[Tensor] = []
        if self.use_lesionness_guidance:
            probability = torch.sigmoid(lesionness_logits)
            context_parts.append(
                F.interpolate(
                    probability,
                    size=full_resolution.shape[2:],
                    mode="trilinear",
                    align_corners=False,
                )
            )
        if anatomy is not None:
            zone, _uncertainty = zone_context(anatomy[:, 0:1], anatomy[:, 1:2])
            context_parts.append(
                F.interpolate(zone, size=full_resolution.shape[2:], mode="trilinear", align_corners=False)
                if zone.shape[2:] != full_resolution.shape[2:]
                else zone
            )
        context = torch.cat(context_parts, dim=1)
        if context.shape[1] != self.context_channels:
            raise LesionAwareCoarseToFineError(
                f"context 通道数 {context.shape[1]} 与声明 {self.context_channels} 不一致"
            )

        refined = self.refinement(full_resolution, context)
        if deep_supervision:
            outputs = list(outputs)
            outputs[0] = refined
        else:
            outputs = refined
        self.auxiliary_outputs = {"lesionness_logits": lesionness_logits}
        return outputs

    def compute_conv_feature_map_size(self, input_size):
        """代理到 backbone，供 nnU-Net 显存估算等使用（不改变其语义）。"""
        return self.backbone.compute_conv_feature_map_size(input_size)


__all__ = (
    "ANATOMY_CONTEXT_CHANNELS",
    "LESIONNESS_GUIDANCE_CHANNELS",
    "CoarseLesionnessHead",
    "LesionAwareCoarseToFineError",
    "LesionAwareCoarseToFineNNUNet",
)
