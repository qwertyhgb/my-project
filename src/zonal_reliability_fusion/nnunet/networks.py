"""轻量 spatial modality reliability gate 与原生 nnU-Net backbone 的薄包装。

设计原则（严格遵守）：

- **不复制 nnU-Net 的 U-Net**。backbone 一律由 ``nnunetv2.utilities.get_network_from_plans``
  按 plans.json 构建（Dataset605/606 的 3d_fullres 均为原生 ``PlainConvUNet``）。
- **gate 极小**：``Conv3d(in, hidden, 1) -> InstanceNorm3d -> LeakyReLU -> Conv3d(hidden, 3, 1)``，
  最后一个卷积 weight/bias 全零初始化。
- **零初始化恒等**：``weights = softmax(logits, dim=1)``、``scales = 3 * weights``、
  ``gated = mri * scales``。零初始化时 logits 全 0，softmax 得 1/3，scales 严格为 1，
  因此新模型初始行为与原生 baseline **逐值一致**（不会把输入缩小为 1/3）。
- **包装器透明**：代理 backbone 的 forward 与 deep supervision 输出格式；暴露 ``decoder``
  属性，使 ``nnUNetTrainer.set_deep_supervision_enabled`` 原样可用；state_dict 严格加载；
  forward 中不保存任何大张量 diagnostics。
- **anatomy variant**：严格要求 5 通道（T2W/ADC/HBV/PZ/TZ），PZ/TZ 进入 gate 前 clamp(0,1)，
  只把加权后的前 3 个 MRI 通道送入 backbone，PZ/TZ 不进入分割 backbone，禁止 WG。
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

#: 分割 backbone 永远只吃 3 个 MRI 通道（T2W、ADC、HBV）
MRI_CHANNELS = 3
#: gate 为每个 MRI 模态产生一个逐体素权重
GATE_OUTPUT_CHANNELS = 3
#: gate 的隐藏通道数（很小；网络差异由该常量与 Trainer 类名表达，不再维护大 YAML）
GATE_HIDDEN_CHANNELS = 8
#: anatomy variant 的 PZ/TZ 两个附加通道
PRIOR_CHANNELS_ANATOMY = 2
#: anatomy variant 的总输入通道数（3 MRI + PZ + TZ）
ANATOMY_INPUT_CHANNELS = MRI_CHANNELS + PRIOR_CHANNELS_ANATOMY


class SpatialModalityReliabilityGate(nn.Module):
    """逐体素 modality 可靠性 gate：输入若干通道，输出 3 个 modality scale。

    结构固定为两层 1x1x1 卷积 + InstanceNorm + LeakyReLU；末层零初始化，保证
    初始 ``scales == 1``。``forward`` 返回的是已经乘好系数的 ``scales``（``3 * softmax``），
    调用方直接 ``mri * scales`` 即可。
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int = GATE_HIDDEN_CHANNELS,
        num_modalities: int = GATE_OUTPUT_CHANNELS,
    ) -> None:
        super().__init__()
        input_channels = int(input_channels)
        hidden_channels = int(hidden_channels)
        num_modalities = int(num_modalities)
        if input_channels < num_modalities:
            raise ValueError(
                f"gate 输入通道 {input_channels} 少于要产生的 modality 数 {num_modalities}"
            )
        if hidden_channels <= 0 or num_modalities <= 0:
            raise ValueError("hidden_channels 与 num_modalities 必须为正")
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.num_modalities = num_modalities
        self.conv1 = nn.Conv3d(input_channels, hidden_channels, kernel_size=1)
        self.norm = nn.InstanceNorm3d(hidden_channels, affine=True)
        self.nonlin = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv3d(hidden_channels, num_modalities, kernel_size=1)
        # 末层零初始化：logits 恒为 0 -> softmax 均匀 1/num_modalities -> scales 恒为 1
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[1] != self.input_channels:
            raise ValueError(
                f"gate 期望 {self.input_channels} 个输入通道，收到 {tuple(x.shape)}"
            )
        logits = self.conv2(self.nonlin(self.norm(self.conv1(x))))
        weights = torch.softmax(logits, dim=1)
        return self.num_modalities * weights


class GatedNNUNet(nn.Module):
    """把原生 nnU-Net backbone 用一个 spatial modality reliability gate 包起来。

    - ``backbone``：由 ``get_network_from_plans`` 构建的原生网络（输入恒为 3 个 MRI 通道）。
    - ``gate``：产生 3 个逐体素 modality scale。
    - ``num_prior_channels``：0（image variant）或 2（anatomy variant 的 PZ/TZ）。

    forward 只做三件事：必要时 clamp prior -> gate 产生 scales -> ``mri * scales`` 送入
    backbone。deep supervision 输出格式由 backbone 决定，本包装原样返回，不做任何改动。
    """

    def __init__(
        self,
        backbone: nn.Module,
        gate: SpatialModalityReliabilityGate,
        *,
        mri_channels: int = MRI_CHANNELS,
        num_prior_channels: int = 0,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.gate = gate
        self.mri_channels = int(mri_channels)
        self.num_prior_channels = int(num_prior_channels)
        self.total_input_channels = self.mri_channels + self.num_prior_channels
        if self.num_prior_channels < 0:
            raise ValueError("num_prior_channels 不能为负")
        if self.gate.input_channels != self.total_input_channels:
            raise ValueError(
                f"gate 输入通道 {self.gate.input_channels} 与包装器总输入通道 "
                f"{self.total_input_channels} 不一致"
            )

    @property
    def decoder(self):
        """暴露原生 backbone 的 decoder，使 ``set_deep_supervision_enabled`` 正常工作。

        ``nnUNetTrainer.set_deep_supervision_enabled`` 执行 ``mod.decoder.deep_supervision = flag``；
        这里把 ``decoder`` 直接代理到 backbone.decoder（真正的 ``UNetDecoder``），无需重写该方法。
        """
        return self.backbone.decoder

    def _gate_input(self, x: Tensor) -> Tensor:
        """校验通道数；anatomy variant 对 PZ/TZ 执行 clamp(0,1) 后再交给 gate。"""
        if x.ndim < 2:
            raise ValueError(f"输入至少应为 [B,C,...]，收到 {tuple(x.shape)}")
        if x.shape[1] != self.total_input_channels:
            raise ValueError(
                f"该网络严格要求 {self.total_input_channels} 个输入通道"
                f"（{self.mri_channels} MRI + {self.num_prior_channels} prior），"
                f"收到 {x.shape[1]} 通道：{tuple(x.shape)}"
            )
        if self.num_prior_channels == 0:
            return x
        mri = x[:, : self.mri_channels]
        prior = x[:, self.mri_channels :].clamp(0.0, 1.0)
        return torch.cat([mri, prior], dim=1)

    def forward(self, x: Tensor):
        gate_input = self._gate_input(x)
        scales = self.gate(gate_input)
        gated_mri = gate_input[:, : self.mri_channels] * scales
        # backbone 的返回值（deep supervision 时为多分辨率 list，否则单张量）原样透传
        return self.backbone(gated_mri)

    def compute_conv_feature_map_size(self, input_size):
        """代理到 backbone，供 nnU-Net 显存估算等使用（不改变其语义）。"""
        return self.backbone.compute_conv_feature_map_size(input_size)
