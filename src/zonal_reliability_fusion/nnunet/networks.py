"""轻量 spatial modality gate 与原生 nnU-Net backbone 的薄包装。

术语说明：Python 类名/包名中的 ``Reliability``（``zonal_reliability_fusion``、
``SpatialModalityReliabilityGate``）是**历史内部标识**，保留它是为了 checkpoint、导入路径与
既有代码兼容；当前研究解释是模型学习到的 **learned sequence preference / scale**（序列相对
偏好与尺度），不是校准可靠性、真实图像质量或因果贡献。

设计原则（严格遵守）：

- **不复制 nnU-Net 的 U-Net**。backbone 一律由 ``nnunetv2.utilities.get_network_from_plans``
  按 plans.json 构建（Dataset605/606 的 3d_fullres 均为原生 ``PlainConvUNet``）。
- **gate 极小**：``Conv3d(in, hidden, 1) -> InstanceNorm3d -> LeakyReLU -> Conv3d(hidden, 3, 1)``，
  最后一个卷积 weight/bias 全零初始化。
- **零初始化恒等**：``weights = softmax(logits, dim=1)``（W，和为 1）、``scales = 3 * weights``
  （S，和为 3）、``gated = mri * scales``。零初始化时 logits 全 0，softmax 得 1/3，scales 严格
  为 1，因此新模型初始行为与原生 baseline **逐值一致**（不会把输入缩小为 1/3）。
  ``SpatialModalityReliabilityGate.compute_weights`` 与 ``GatedNNUNet.compute_gate_scales``
  是机制分析接口：纯函数式、不缓存任何 batch 大张量、不改变 state_dict。
- **包装器透明**：代理 backbone 的 forward 与 deep supervision 输出格式；暴露 ``decoder``
  属性，使 ``nnUNetTrainer.set_deep_supervision_enabled`` 原样可用；state_dict 严格加载；
  forward 中不保存任何大张量 diagnostics。
- **anatomy variant**：严格要求 5 通道（T2W/ADC/HBV/PZ/TZ），PZ/TZ 进入 gate 前 clamp(0,1)，
  只把加权后的前 3 个 MRI 通道送入 backbone，PZ/TZ 不进入分割 backbone，禁止 WG。
- **浅层序列特异特征融合（Research Plan §8.10）**：``FeatureFusionNNUNet`` 在原生 backbone 之前
  为每个 MRI 序列各放一个**参数互不共享、不下采样**的两层 3×3×3 stem（``ShallowSequenceStem``），
  把拼接后的 ``3*C_s`` 通道轻量投影回 3 通道再交给 backbone。feature gate 只读**拼接后的 stem
  特征**（anatomy 额外读 clamp 后的 PZ/TZ）并产生 3 个 logit；PZ/TZ 不进入任何 stem、投影层或
  backbone。feature gate 同样末层零初始化，因此初始 ``S≡1``，使 feature gate 版本**初始逐值等价于
  同 stem / 投影 / backbone 的 feature no-gate 版本**，但**不**等价于原生输入级 nnU-Net
  （stem 与投影本身仍然改变输入，见 Research Plan §8.10 末段）。
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
    """逐体素 modality gate：输入若干通道，输出 3 个 modality scale。

    结构固定为两层 1x1x1 卷积 + InstanceNorm + LeakyReLU；末层零初始化，保证
    初始 ``scales == 1``。数学约定（与 Research Plan 一致）：

    - :meth:`compute_weights` 返回归一化相对序列偏好 ``W = softmax(logits, dim=1)``，
      非负且通道和为 1；
    - :meth:`forward` 返回真正与 MRI 相乘的尺度 ``S = num_modalities * W``（非负，通道和为
      ``num_modalities``），调用方直接 ``mri * scales`` 即可。

    类名中的 "Reliability" 是历史内部标识（保持 checkpoint/导入兼容）；当前研究解释为
    learned sequence preference/scale，不是校准可靠性、真实图像质量或因果贡献。
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

    def compute_weights(self, x: Tensor) -> Tensor:
        """返回归一化相对偏好 ``W = softmax(logits, dim=1)``（非负，通道和为 1）。"""
        if x.shape[1] != self.input_channels:
            raise ValueError(
                f"gate 期望 {self.input_channels} 个输入通道，收到 {tuple(x.shape)}"
            )
        logits = self.conv2(self.nonlin(self.norm(self.conv1(x))))
        return torch.softmax(logits, dim=1)

    def forward(self, x: Tensor) -> Tensor:
        """返回与 MRI 相乘的尺度 ``S = num_modalities * W``（非负，通道和为 num_modalities）。"""
        return self.num_modalities * self.compute_weights(x)


class GatedNNUNet(nn.Module):
    """把原生 nnU-Net backbone 用一个 spatial modality reliability gate 包起来。

    - ``backbone``：由 ``get_network_from_plans`` 构建的原生网络（输入恒为 3 个 MRI 通道）。
    - ``gate``：产生 3 个逐体素 modality scale。
    - ``num_prior_channels``：0（image variant）或 2（anatomy variant 的 PZ/TZ）。

    forward 只做三件事：``compute_gate_scales``（内部 clamp prior 并产生 scales）->
    ``mri * scales`` -> 送入 backbone。deep supervision 输出格式由 backbone 决定，本包装
    原样返回，不做任何改动。
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

    def compute_gate_scales(self, x: Tensor) -> Tensor:
        """返回真正与 MRI 相乘的 3 通道尺度场 ``S``（非负、通道和为 3）。

        输入先经 :meth:`_gate_input`，因此通道数检查与 anatomy variant 的 PZ/TZ
        ``clamp(0,1)`` **不会**被绕过；纯函数式，不缓存、不保存任何 batch 大张量。
        """
        gate_input = self._gate_input(x)
        return self.gate(gate_input)

    def forward(self, x: Tensor):
        scales = self.compute_gate_scales(x)
        # _gate_input 只做通道检查与 prior clamp，从不改动 MRI 通道；因此直接取 x 的 MRI
        # 部分与修补前的 ``_gate_input(x)[:, :mri_channels] * scales`` 逐值一致。
        gated_mri = x[:, : self.mri_channels] * scales
        # backbone 的返回值（deep supervision 时为多分辨率 list，否则单张量）原样透传
        return self.backbone(gated_mri)

    def compute_conv_feature_map_size(self, input_size):
        """代理到 backbone，供 nnU-Net 显存估算等使用（不改变其语义）。"""
        return self.backbone.compute_conv_feature_map_size(input_size)


# ===========================================================================
# 浅层序列特异特征融合（Research Plan §8.10）
# ===========================================================================

#: 浅层序列 stem 的统一通道数 ``C_s``（三条序列共用同一取值；差异只来自各自的参数）。
#:
#: 取值依据（**解析估算，不是实测**）：patch ``16×320×320 = 1 638 400`` 体素、batch size 2、
#: fp32 时，每个 stem 在满分辨率上产生 2 层 × ``C_s`` 通道激活，三条序列合计 ``6*C_s`` 张满分辨率
#: 特征图（尚未计入 InstanceNorm 与 autograd 需要保存的中间量，两者约使该量级再翻一倍左右）：
#:
#: - ``C_s = 8``  -> 48 张 -> 约 0.63 GB 顶层激活，计入中间量约 1.7–2.0 GB；
#: - ``C_s = 16`` -> 96 张 -> 约 1.26 GB 顶层激活，计入中间量约 3.2–4.0 GB。
#:
#: 作为对照，原生 ``PlainConvUNet`` 的 stage 0 在满分辨率上已有 2 层 × 32 通道（batch 2 约
#: 0.84 GB）。``C_s = 16`` 会让骨干之前的开销超过原生 stage 0 的两倍，因此取保守值
#: **``C_s = 8``**；它与 gate 隐藏通道数（``GATE_HIDDEN_CHANNELS``）同量级，且**不改变**
#: ``nnUNetPlans.json``（骨干仍完全由 plans 构建）。
FEATURE_STEM_CHANNELS = 8
#: stem 卷积核尺寸：两层 3×3×3，stride=1、padding=1，因此空间尺寸逐值保持（不下采样）
FEATURE_STEM_KERNEL_SIZE = 3


def feature_gate_parameter_delta(
    gate_hidden_channels: int = GATE_HIDDEN_CHANNELS,
    num_prior_channels: int = PRIOR_CHANNELS_ANATOMY,
) -> int:
    """feature anatomy 与 feature image 的 gate 参数量之差。

    两者 stem、投影层、backbone 的形状与参数量**完全一致**；anatomy 的 gate 多读
    ``num_prior_channels`` 个输入通道（clamp 后的 PZ/TZ），只有 ``gate.conv1.weight`` 的输入维变大，
    差值恒为 ``gate_hidden_channels * num_prior_channels``（C_s=8、hidden=8、PZ/TZ=2 时为 **16**）。
    本函数的意义是把这个差值显式记录下来，而不是靠"两者结构相同"的说法掩盖它。

    这里**不**使用零值占位通道去把两者的 gate 参数量凑成相等：占位通道虽然不携带信息，但会使
    "anatomy 条件的唯一差异是 PZ/TZ"这一表述更难核查，收益不足以抵消复杂度。差值 16 个参数
    相对 backbone 的约 4.46×10⁷ 参数可以忽略，但结论中不得声称两者参数量严格相同。
    """
    return int(gate_hidden_channels) * int(num_prior_channels)


class ShallowSequenceStem(nn.Module):
    """单个 MRI 序列的浅层 stem：两层 3×3×3 Conv3d + 归一化 + 非线性，不下采样。

    映射为 ``X_m [B,1,...] -> H_m [B,C_s,...]``，空间尺寸因 ``padding = kernel_size // 2``、
    ``stride = 1`` 而逐值保持。两层卷积的**名义**直接感受野是 5×5×5 体素；由于使用 instance
    normalization，patch 统计会引入额外的整块依赖，因此不能把 5×5×5 当作完整有效感受野或毫米
    尺度不变的物理邻域（Research Plan §8.10）。

    三条序列各有**独立实例**：参数不共享由调用方（``FeatureFusionNNUNet``）用 ``nn.ModuleList``
    保证，并有测试守护。
    """

    def __init__(
        self,
        channels: int = FEATURE_STEM_CHANNELS,
        kernel_size: int = FEATURE_STEM_KERNEL_SIZE,
    ) -> None:
        super().__init__()
        channels = int(channels)
        kernel_size = int(kernel_size)
        if channels <= 0:
            raise ValueError(f"stem 通道数必须为正，收到 {channels}")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(
                f"stem 卷积核必须为正的奇数（保持空间尺寸），收到 {kernel_size}"
            )
        self.channels = channels
        self.kernel_size = kernel_size
        padding = kernel_size // 2
        self.conv1 = nn.Conv3d(1, channels, kernel_size, stride=1, padding=padding)
        self.norm1 = nn.InstanceNorm3d(channels, affine=True)
        self.conv2 = nn.Conv3d(
            channels, channels, kernel_size, stride=1, padding=padding
        )
        self.norm2 = nn.InstanceNorm3d(channels, affine=True)
        self.nonlin = nn.LeakyReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim < 2 or x.shape[1] != 1:
            raise ValueError(
                f"浅层 stem 只接受单序列输入 [B,1,...]，收到 {tuple(x.shape)}"
            )
        h = self.nonlin(self.norm1(self.conv1(x)))
        return self.nonlin(self.norm2(self.conv2(h)))


class FeatureFusionNNUNet(nn.Module):
    """浅层序列特异特征融合 + 原生 nnU-Net backbone（Research Plan §8.10）。

    数据流（``mri = x[:, :3]``，``z = x[:, 3:]``，仅 anatomy 存在 ``z``）：

    1. ``H_m = E_m(mri[:, m:m+1])``：三条序列各自的 stem，``H_m [B, C_s, ...]``；
    2. ``S = 3 * softmax(A_phi(Concat(H_T2W, H_ADC, H_HBV [, clamp(z,0,1)])))``：feature gate
       只读**拼接后的 stem 特征**（anatomy 额外读 clamp 到 [0,1] 的 PZ/TZ），输出恰好 3 个 logit；
    3. ``H~_m = S_m * H_m``：``S_m`` 在 ``H_m`` 的**全部 C_s 个通道上共享**（广播相乘），因此
       每个序列只有一个尺度场；
    4. ``X_feat = P_psi(Concat(H~_T2W, H~_ADC, H~_HBV))``：1×1×1 投影到**恰好 3 个通道**；
    5. ``Y_hat = F_theta(X_feat)``：原生 backbone（输入恒为 3 通道，结构与 plans 一致）。

    ``use_gate=False``（feature no-gate）使用**完全相同的** stem 与投影结构，只是跳过第 2、3 步
    直接投影未加权特征。此时 ``self.gate is None``，``state_dict`` 中不含任何 ``gate.*`` 键。

    关键结构约束：``z`` 只进入第 2 步的 gate —— 不进入任何 stem、投影层或 backbone；且
    ``z`` 先 ``clamp(0, 1)``。PZ/TZ 不参与分割特征提取，禁止 WG。

    feature gate 的末层零初始化使初始 ``S ≡ 1``，于是 feature gate 网络在**共享相同 stem / 投影 /
    backbone 权重**时与 feature no-gate 网络逐值一致；但这**不**意味着它等于原生输入级 nnU-Net：
    ``E_m`` 与 ``P_psi`` 仍然改变进入 backbone 的输入分布（Research Plan §8.10 末段）。

    参数量差异：``gate.conv1.weight`` 的输入维在 anatomy 下比 image 大 2
    （见 :func:`feature_gate_parameter_delta`，C_s=8 时为 16 个参数）；其余模块形状完全一致。
    """

    def __init__(
        self,
        backbone: nn.Module,
        *,
        stem_channels: int = FEATURE_STEM_CHANNELS,
        gate_hidden_channels: int = GATE_HIDDEN_CHANNELS,
        use_gate: bool = True,
        num_prior_channels: int = 0,
        mri_channels: int = MRI_CHANNELS,
    ) -> None:
        super().__init__()
        self.mri_channels = int(mri_channels)
        self.num_prior_channels = int(num_prior_channels)
        self.stem_channels = int(stem_channels)
        self.use_gate = bool(use_gate)
        if self.mri_channels != MRI_CHANNELS:
            raise ValueError(
                "浅层融合必须投影回恰好 3 个 MRI 通道（T2W/ADC/HBV），"
                f"收到 mri_channels={self.mri_channels}"
            )
        if self.num_prior_channels < 0:
            raise ValueError("num_prior_channels 不能为负")
        self.total_input_channels = self.mri_channels + self.num_prior_channels

        self.backbone = backbone
        # 三条序列各一个独立 stem 实例：参数不共享（ModuleList 不复制权重）
        self.stems = nn.ModuleList(
            ShallowSequenceStem(self.stem_channels) for _ in range(self.mri_channels)
        )
        # 拼接而非逐通道求和：三个独立 stem 的同序号通道没有语义对齐保证（§8.10）
        self.projection = nn.Conv3d(
            self.stem_channels * self.mri_channels, MRI_CHANNELS, kernel_size=1
        )
        if self.use_gate:
            self.gate = SpatialModalityReliabilityGate(
                input_channels=(
                    self.stem_channels * self.mri_channels + self.num_prior_channels
                ),
                hidden_channels=int(gate_hidden_channels),
                num_modalities=GATE_OUTPUT_CHANNELS,
            )
        else:
            # 普通属性（不是子模块）：no-gate 变体的 state_dict 不含任何 gate.* 键
            self.gate = None

    @property
    def decoder(self):
        """代理到 backbone.decoder，使 ``set_deep_supervision_enabled`` 原样可用。"""
        return self.backbone.decoder

    def _split_input(self, x: Tensor) -> tuple[Tensor, Tensor | None]:
        """校验通道数并拆出 MRI 与（anatomy 的）clamp(0,1) prior。"""
        if x.ndim < 2:
            raise ValueError(f"输入至少应为 [B,C,...]，收到 {tuple(x.shape)}")
        if x.shape[1] != self.total_input_channels:
            raise ValueError(
                f"该网络严格要求 {self.total_input_channels} 个输入通道"
                f"（{self.mri_channels} MRI + {self.num_prior_channels} prior），"
                f"收到 {x.shape[1]} 通道：{tuple(x.shape)}"
            )
        mri = x[:, : self.mri_channels]
        if self.num_prior_channels == 0:
            return mri, None
        return mri, x[:, self.mri_channels :].clamp(0.0, 1.0)

    def _stem_features(self, x: Tensor) -> tuple[list[Tensor], Tensor | None]:
        """返回 ``([H_T2W, H_ADC, H_HBV], clamped_prior_or_None)``。"""
        mri, prior = self._split_input(x)
        features = [
            stem(mri[:, index : index + 1]) for index, stem in enumerate(self.stems)
        ]
        return features, prior

    def _gate_input(self, features: list[Tensor], prior: Tensor | None) -> Tensor:
        """gate 的输入：拼接后的 stem 特征（anatomy 再拼上已 clamp 的 PZ/TZ）。

        这是 PZ/TZ 在网络中**唯一**的入口。
        """
        concatenated = torch.cat(features, dim=1)
        if prior is None:
            return concatenated
        return torch.cat([concatenated, prior], dim=1)

    def compute_gate_scales(self, x: Tensor) -> Tensor:
        """机制分析接口：返回真正与 ``H_m`` 相乘的 3 通道尺度场 ``S``（非负、通道和为 3）。

        纯函数式，不缓存任何 batch 大张量、不改变 ``state_dict``；no-gate 变体没有 gate，调用即报错。
        为便于独立调用，这里会重新计算三条 stem 特征（不经过投影层）。
        """
        if self.gate is None:
            raise RuntimeError(
                "feature_no_gate 没有 gate，无法提供尺度场 S；"
                "该条件只用于隔离浅层编码与投影本身的作用。"
            )
        features, prior = self._stem_features(x)
        return self.gate(self._gate_input(features, prior))

    def compute_weights(self, x: Tensor) -> Tensor:
        """机制分析接口：返回归一化相对偏好 ``W = softmax(logits, dim=1)``（非负、和为 1）。"""
        if self.gate is None:
            raise RuntimeError("feature_no_gate 没有 gate，无法提供相对偏好 W")
        features, prior = self._stem_features(x)
        return self.gate.compute_weights(self._gate_input(features, prior))

    def forward(self, x: Tensor):
        """返回 backbone 的原始输出（deep supervision 时为多分辨率 list，原样透传）。"""
        features, prior = self._stem_features(x)
        if self.gate is not None:
            scales = self.gate(self._gate_input(features, prior))
            # S_m 在该序列的全部 C_s 个通道上共享（广播相乘）：每个序列只有一个尺度场
            features = [
                feature * scales[:, index : index + 1]
                for index, feature in enumerate(features)
            ]
        projected = self.projection(torch.cat(features, dim=1))
        return self.backbone(projected)

    def compute_conv_feature_map_size(self, input_size):
        """代理到 backbone，供 nnU-Net 显存估算等使用（不改变其语义）。"""
        return self.backbone.compute_conv_feature_map_size(input_size)
