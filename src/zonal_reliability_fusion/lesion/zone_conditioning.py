"""PZ/TZ 的**新作用**：为 lesion localization / refinement 提供解剖上下文。

与旧研究线的关键区别（必须写清楚，避免又滑回门控）
--------------------------------------------------
旧的 ``anatomy_gate`` / ``feature_anatomy_gate`` 把 PZ/TZ 的作用写成「根据区域调整
T2W / ADC / HBV 的 modality weight」。**新主线不再这样解释**。新解释是：

    PZ/TZ 回答的是「这个候选病灶长在哪里、周围是什么组织、
    它落在 PZ / TZ / 分区边界 / 解剖不确定区？」

因此本模块的接口是 **context provider**，不是 modality reweighting：

- 输入是 lesion 的**共享特征** ``F`` 与 **soft** 区域概率 ``P(PZ) / P(TZ)``；
- 输出是同样形状、同样通道数的**解剖条件化特征**；
- 描述符与实现不得出现「modality weight / sequence weight / reliability」这类语义。

两种候选实现（默认先用简单的那一种）
------------------------------------
1. ``concat``（**默认**）：把 ``P(PZ) / P(TZ) / P(U)`` 直接拼到特征前的 context 里，交给一个
   1×1×1 卷积。参数最少、最易解释，是「直接 concatenation soft anatomy maps 是否已经足够」
   这一问题的直接检验。
2. ``zone_experts``：两个 expert 分支 + soft 融合

   .. math::

       F_{out} = P(PZ)\\,F_{PZ} + P(TZ)\\,F_{TZ} + P(U)\\,F_{shared}

   其中 ``P(U)`` 用于边界或 anatomy uncertainty 区域。它检验「是否真的需要两个 expert」。

:func:`zone_mode_parameter_delta` 给出两种模式的参数量之差，使选择可以**基于可控的参数量差异**
而不是直觉。默认不用 ``zone_experts``：如果 ``concat`` 已经足够（条件 D 的对照会直接回答），
就不应该引入 expert 分支——这正是旧的「为了 anatomy 而设计复杂 Gate」问题。

uncertain 的语义
----------------
``P(U)`` 不是 anatomy 模型直接输出的类别，而是由 soft 概率派生的**不确定度**：取
``1 - max(P(PZ), P(TZ))``，即两个区域证据都不强时取大值。这样它天然覆盖分区边界与
anatomy 预测不确定的区域，而不需要在训练中额外监督一个 uncertain 头。
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

#: 区域条件化的模式；``concat`` 是默认（最简、最少参数）
ZONE_MODES = ("concat", "zone_experts")


def zone_context(
    pz_probability: Tensor, tz_probability: Tensor
) -> tuple[Tensor, Tensor]:
    """由 soft ``P(PZ) / P(TZ)`` 派生区域上下文与不确定权重。

    返回 ``(context, uncertainty)``：

    - ``context``：``[B, 3, ...]``，通道顺序 ``(P(PZ), P(TZ), P(U))``；
    - ``uncertainty = 1 - max(P(PZ), P(TZ))``，落在 [0, 1]，区域证据不足时取大值。

    输入必须是 soft probability（值域 [0, 1]，有限）；硬 one-hot 会被显式拒绝，因为
    「soft 而不是硬 one-hot」是本设计的前置条件——硬标签会让边界体素失去 residual path。
    """
    for name, value in (("P(PZ)", pz_probability), ("P(TZ)", tz_probability)):
        if not torch.is_floating_point(value):
            raise TypeError(f"{name} 必须是浮点 soft probability，收到 {value.dtype}")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} 含非有限值")
        if bool((value < 0).any()) or bool((value > 1).any()):
            raise ValueError(
                f"{name} 超出 [0, 1]：region conditioning 要求 soft probability，"
                "不接受 logits 或硬 one-hot"
            )
    if pz_probability.shape != tz_probability.shape:
        raise ValueError(
            f"P(PZ) 与 P(TZ) 形状必须一致：{tuple(pz_probability.shape)} vs "
            f"{tuple(tz_probability.shape)}"
        )
    uncertainty = 1.0 - torch.maximum(pz_probability, tz_probability)
    return torch.cat([pz_probability, tz_probability, uncertainty], dim=1), uncertainty


class ZoneAwareRefinement(nn.Module):
    """把解剖区域上下文注入 lesion 特征的轻量 refinement。

    ``mode="concat"``（默认）::

        context = [P(PZ), P(TZ), P(U)]            (3 通道)
        F_out   = F + zero_init_conv([F, context])

    ``mode="zone_experts"``::

        F_PZ, F_TZ, F_shared = expert_pz(F), expert_tz(F), expert_shared(F)
        F_zone   = P(PZ)*F_PZ + P(TZ)*F_TZ + P(U)*F_shared
        F_out    = F + zero_init_conv([F_zone, context])

    两个模式的输出层都是**零初始化**，因此初始时 ``F_out == F``（恒等），新模块在初始状态
    下不会改变下游任何数值——这是「与 strong baseline 单变量可比」的前提。注意：恒等只保证
    初始等价，**不**保证训练后等价。

    ``zone_experts`` 的 expert 是**两个独立分支加一个共享分支**（不是每区域 CNN），参数量差异
    由 :func:`zone_mode_parameter_delta` 报告，便于判断「是否真的需要两个 expert」。
    """

    def __init__(
        self,
        feature_channels: int,
        *,
        mode: str = "concat",
        hidden_channels: int = 8,
        context_channels: int = 3,
    ) -> None:
        super().__init__()
        feature_channels = int(feature_channels)
        hidden_channels = int(hidden_channels)
        context_channels = int(context_channels)
        if feature_channels <= 0 or hidden_channels <= 0 or context_channels <= 0:
            raise ValueError("feature_channels / hidden_channels / context_channels 必须为正")
        if mode not in ZONE_MODES:
            raise ValueError(f"mode 必须是 {ZONE_MODES} 之一，收到 {mode!r}")
        self.feature_channels = feature_channels
        self.hidden_channels = hidden_channels
        self.context_channels = context_channels
        self.mode = mode

        if mode == "zone_experts":
            self.expert_pz = nn.Conv3d(feature_channels, feature_channels, 1)
            self.expert_tz = nn.Conv3d(feature_channels, feature_channels, 1)
            self.expert_shared = nn.Conv3d(feature_channels, feature_channels, 1)
        else:
            self.expert_pz = None
            self.expert_tz = None
            self.expert_shared = None

        self.output = nn.Sequential(
            nn.Conv3d(feature_channels + context_channels, hidden_channels, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(hidden_channels, feature_channels, 1),
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, features: Tensor, context: Tensor) -> Tensor:
        if features.ndim != 5:
            raise ValueError(f"features 必须为 [B,C,D,H,W]，收到 {tuple(features.shape)}")
        if features.shape[1] != self.feature_channels:
            raise ValueError(
                f"features 通道数应为 {self.feature_channels}，收到 {features.shape[1]}"
            )
        if context.shape[1] != self.context_channels:
            raise ValueError(
                f"context 通道数应为 {self.context_channels}，收到 {context.shape[1]}"
            )
        if context.shape[0] != features.shape[0] or context.shape[2:] != features.shape[2:]:
            raise ValueError(
                "context 与 features 必须同 batch、同空间尺寸："
                f"{tuple(context.shape)} vs {tuple(features.shape)}"
            )
        source = features
        if self.mode == "zone_experts":
            pz, tz, uncertainty = context[:, 0:1], context[:, 1:2], context[:, 2:3]
            source = (
                pz * self.expert_pz(features)
                + tz * self.expert_tz(features)
                + uncertainty * self.expert_shared(features)
            )
        return features + self.output(torch.cat([source, context], dim=1))


def zone_mode_parameter_delta(
    feature_channels: int, *, mode: str = "concat"
) -> int:
    """给出 ``mode`` 相对 ``concat`` 多出的参数量（用于「是否真的需要两个 expert」的判断）。

    ``zone_experts`` 额外引入三个 ``feature_channels x feature_channels x 1`` 卷积，
    因此差值是 ``3 * feature_channels**2 + 3 * feature_channels``（含 bias）。
    """
    feature_channels = int(feature_channels)
    if feature_channels <= 0:
        raise ValueError("feature_channels 必须为正")
    if mode == "concat":
        return 0
    if mode != "zone_experts":
        raise ValueError(f"mode 必须是 {ZONE_MODES} 之一，收到 {mode!r}")
    return 3 * feature_channels * feature_channels + 3 * feature_channels


__all__ = (
    "ZONE_MODES",
    "ZoneAwareRefinement",
    "zone_context",
    "zone_mode_parameter_delta",
)
