"""ARCHIVED：旧门控 / 特征融合 / 同区参照融合研究线的全部 Trainer 类。

归档时间：2026-10-08。**这些类不再是项目主线**，但仍然被完整保留，因为：

1. **checkpoint 兼容**：输出目录由 Trainer 类名决定，改名即让既有 checkpoint / validation
   产物失联。类名与实现必须逐字不变；
2. **历史实验可复现**：已完成训练的输入级门控与特征融合实验是论文的 preliminary /
   negative evidence；
3. **不篡改已记录的事实**：``Training_Log.md`` 与 ``docs/experiments/`` 里对这些类的引用
   继续有效。

**不得**在本文件里新增方法或修改任何超参数。新主线的方法一律进入
``zonal_reliability_fusion.lesion`` 与 ``zonal_reliability_fusion.nnunet.trainers``。

保留的具体条件：

- 输入级门控：``nnUNetTrainerPICAI_ImageGate`` / ``nnUNetTrainerPICAI_AnatomyGate`` 及其
  ``_PositiveSampling`` 组合；
- 特征级融合：``FeatureNoGate`` / ``FeatureImageGate`` / ``FeatureAnatomyGate``；
- 同区参照融合（100 epoch 短预算分支）：``ZonalReference`` / ``ZonalReferenceAdaptive``；
- 未启动成功的短预算普通融合：``FeatureNoGate`` / ``FeatureAnatomyGate`` 的 ``_100ep`` 版本。

它们共用的机制说明保留在各自的 docstring 中；类间比较边界见
``docs/archive/legacy_gate_research/README.md``。
"""

from __future__ import annotations

from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from torch import nn

from zonal_reliability_fusion.legacy.fusion_networks import (
    ANATOMY_INPUT_CHANNELS,
    FEATURE_STEM_CHANNELS,
    MRI_CHANNELS,
    PRIOR_CHANNELS_ANATOMY,
    FeatureFusionNNUNet,
    GatedNNUNet,
    SpatialModalityReliabilityGate,
    ZonalReferenceFusionNNUNet,
)
from zonal_reliability_fusion.legacy.prior_transforms import (
    restrict_intensity_transforms_to_mri,
)
from zonal_reliability_fusion.sampling.positive_sampling import (
    PositiveCaseSamplingMixin,
)
from zonal_reliability_fusion.nnunet.bases import (
    nnUNetTrainerPICAI_FLCE_NoFFT,
    nnUNetTrainerPICAI_FLCE_PositiveSampling_NoFFT,
)


class _GatedTrainerBase(nnUNetTrainerPICAI_FLCE_NoFFT):
    """image_gate / anatomy_gate 的公共基类：只覆盖网络构建，其余全部继承 baseline。"""

    #: 子类声明其严格期望的输入通道数（image=3，anatomy=5）
    expected_input_channels: int = MRI_CHANNELS
    #: 子类声明 prior 通道数（image=0，anatomy=2）
    num_prior_channels: int = 0

    @staticmethod
    def _build_gated_network(
        architecture_class_name,
        arch_init_kwargs,
        arch_init_kwargs_req_import,
        num_input_channels,
        num_output_channels,
        enable_deep_supervision,
        *,
        expected_input_channels,
        num_prior_channels,
    ) -> nn.Module:
        if num_input_channels != expected_input_channels:
            raise ValueError(
                f"该 Trainer 严格要求 {expected_input_channels} 个输入通道"
                f"（{MRI_CHANNELS} MRI + {num_prior_channels} prior），收到 {num_input_channels}"
            )
        # backbone 永远只吃 3 个 MRI 通道；由 nnU-Net 官方工具按 plans 构建并初始化。
        backbone = get_network_from_plans(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            MRI_CHANNELS,
            num_output_channels,
            allow_init=True,
            deep_supervision=enable_deep_supervision,
        )
        gate = SpatialModalityReliabilityGate(input_channels=expected_input_channels)
        return GatedNNUNet(
            backbone,
            gate,
            mri_channels=MRI_CHANNELS,
            num_prior_channels=num_prior_channels,
        )



class nnUNetTrainerPICAI_ImageGate(_GatedTrainerBase):
    """image_gate variant：3 个 MRI 通道 -> gate 产生 3 通道 modality weights -> 原生 backbone。

    gate 只依据 T2W/ADC/HBV 本身，不引入任何解剖先验；不重实现 encoder/decoder。
    """

    expected_input_channels = MRI_CHANNELS
    num_prior_channels = 0

    @staticmethod
    def build_network_architecture(
        architecture_class_name,
        arch_init_kwargs,
        arch_init_kwargs_req_import,
        num_input_channels,
        num_output_channels,
        enable_deep_supervision=True,
    ) -> nn.Module:
        return _GatedTrainerBase._build_gated_network(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision,
            expected_input_channels=MRI_CHANNELS,
            num_prior_channels=0,
        )



class nnUNetTrainerPICAI_AnatomyGate(_GatedTrainerBase):
    """anatomy_gate variant：5 通道输入（T2W/ADC/HBV/PZ/TZ）。

    gate 依据 MRI + PZ/TZ 产生 3 个 MRI modality weights；只把加权后的前 3 个 MRI 通道送入
    原生 backbone，PZ/TZ 不进入分割 backbone（禁止 WG）。PZ/TZ 在进入 gate 前 clamp(0,1)。
    另外把强度增强限制在 MRI 通道，空间增强仍同步作用于 MRI/PZ/TZ。
    """

    expected_input_channels = ANATOMY_INPUT_CHANNELS
    num_prior_channels = PRIOR_CHANNELS_ANATOMY

    @staticmethod
    def build_network_architecture(
        architecture_class_name,
        arch_init_kwargs,
        arch_init_kwargs_req_import,
        num_input_channels,
        num_output_channels,
        enable_deep_supervision=True,
    ) -> nn.Module:
        return _GatedTrainerBase._build_gated_network(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision,
            expected_input_channels=ANATOMY_INPUT_CHANNELS,
            num_prior_channels=PRIOR_CHANNELS_ANATOMY,
        )

    @staticmethod
    def get_training_transforms(*args, **kwargs):
        # 先复用 baseline（含 NoFFT 修复）的默认增强，再把强度变换限制到 MRI 通道。
        transforms = nnUNetTrainerPICAI_FLCE_NoFFT.get_training_transforms(
            *args, **kwargs
        )
        transforms, n_wrapped = restrict_intensity_transforms_to_mri(
            transforms, MRI_CHANNELS
        )
        if n_wrapped == 0:
            raise RuntimeError(
                "anatomy_gate 未能在增强管线中找到任何强度变换来限制到 MRI 通道；"
                "拒绝静默启动，以免把 noise/blur/brightness/contrast/gamma 施加到 PZ/TZ 上。"
            )
        return transforms



class nnUNetTrainerPICAI_ImageGate_PositiveSampling_NoFFT(
    PositiveCaseSamplingMixin,
    nnUNetTrainerPICAI_ImageGate,
):
    """image_gate_positive_sampling variant：image gate + 阳性病例采样（RQ1 公平匹配）。

    与旧 ``image_gate`` **唯一**的差异是训练集病例/patch 采样；与 ``positive_sampling``
    **唯一**的差异是网络（image gate）。因此除 gate 外，两者在损失、采样、增强、optimizer、
    LR scheduler、split 与随机种子策略上完全一致，可用于「只归因于 image gate」的 RQ1 比较。
    旧 ``baseline`` ↔ 旧 ``image_gate``（同为 FLCE + 原生采样）本身仍是**原生采样条件下**的受控
    RQ1 对照，只是不能与 ``positive_sampling`` 分支混合比较；其单次运行与 baseline 早期长期全
    背景的优化异常会限制结论强度。

    MRO（``PositiveCaseSamplingMixin`` 在前）保证：

    - loss 仍解析到 ``PICAIFocalCrossEntropyLossMixin._build_loss``
      （``0.5*Focal(gamma=2) + 0.5*CE``），不是 Dice+CE；
    - 网络仍解析到 ``nnUNetTrainerPICAI_ImageGate.build_network_architecture``：
      3 个 MRI 通道 -> gate 产生 3 通道尺度 -> 原生 backbone，末层零初始化（初始 scales≡1）；
    - 增强仍经 ``NoFFTAugmentationMixin.get_training_transforms``（NoFFT 修复生效）；
    - 训练 loader 换成 ``PositiveCaseDataLoader``（``positive_cases_per_batch=1``，槽位固定
      来自当前 fold 阳性训练病例并强制病灶中心裁剪）；验证 loader 保持原生
      ``nnUNetDataLoader``，无验证泄漏；
    - optimizer（SGD+Nesterov）、PolyLR、1000 epochs、deep supervision、checkpoint/resume、
      validation、inference 全部继承原生实现。

    本类**不**从任何旧 checkpoint 续训；输出目录由类名自然形成为独立新目录。
    """



class nnUNetTrainerPICAI_AnatomyGate_PositiveSampling_NoFFT(
    PositiveCaseSamplingMixin,
    nnUNetTrainerPICAI_AnatomyGate,
):
    """anatomy_gate_positive_sampling variant：anatomy gate + 阳性病例采样（RQ2 公平匹配）。

    与旧 ``anatomy_gate`` **唯一**的差异是训练集病例/patch 采样；与
    ``image_gate_positive_sampling`` 的**预期主要差异**是门控条件（多 PZ/TZ 两个通道）与数据集
    （Dataset606 vs Dataset605）。两者的损失、采样、增强、optimizer、LR scheduler 与随机种子
    策略一致，数据集配置（split、3d_fullres spacing、patch size、batch size、前三个 MRI 通道的
    normalization 与 per-channel fingerprint）也已核对一致；但在完成前三个 MRI 通道的**逐数组
    一致性审计**之前，不能宣称唯一差异是 PZ/TZ，RQ2 的归因结论以该审计完成为前提（见 README
    「Dataset605 与 Dataset606 的一致性边界」）。

    MRO（``PositiveCaseSamplingMixin`` 在前）保证：

    - loss 仍解析到 ``PICAIFocalCrossEntropyLossMixin._build_loss``（同一 PI-CAI FLCE）；
    - 网络仍解析到 ``nnUNetTrainerPICAI_AnatomyGate.build_network_architecture``：严格 5 通道
      （T2W/ADC/HBV + PZ/TZ），PZ/TZ 只在 gate 中被读取（进入前 clamp(0,1)），加权后的前 3 个
      MRI 通道进入原生 backbone，PZ/TZ **不进入**分割 backbone（禁止 WG）；
    - 增强仍经 ``nnUNetTrainerPICAI_AnatomyGate.get_training_transforms``：NoFFT 修复生效，
      且强度增强仍只作用于前 3 个 MRI 通道（PZ/TZ 豁免），空间/镜像变换仍同步作用于全部通道；
    - 训练 loader 换成 ``PositiveCaseDataLoader``（``positive_cases_per_batch=1``）；验证 loader
      保持原生 ``nnUNetDataLoader``（无验证泄漏）；
    - optimizer、PolyLR、1000 epochs、deep supervision、checkpoint/resume、validation、
      inference 全部继承原生实现。

    本类**不**从任何旧 checkpoint 续训；输出目录由类名自然形成为独立新目录。
    """



# ===========================================================================
# 浅层序列特异特征融合（Research Plan §8.10）：三个互相匹配的模型条件
# ===========================================================================


class _FeatureFusionTrainerBase(nnUNetTrainerPICAI_FLCE_PositiveSampling_NoFFT):
    """``feature_*`` 三兄弟的公共基类：只提供共享的静态网络构建辅助，不改其它行为。

    继承 ``nnUNetTrainerPICAI_FLCE_PositiveSampling_NoFFT`` 使三个 feature variant 自动获得
    与 ``positive_sampling`` **完全相同**的训练框架：PI-CAI FLCE 损失（``0.5*Focal(γ=2) + 0.5*CE``）、
    NoFFT 修复、每批固定一个阳性病灶 patch 的训练采样（验证 loader 保持原生）、以及 nnU-Net 原生的
    optimizer（SGD+Nesterov）、PolyLR、1000 epoch、deep supervision、checkpoint/resume、
    validation 与滑窗推理。

    三个子类**只**覆盖 ``build_network_architecture``；anatomy 子类额外覆盖
    ``get_training_transforms`` 以保持 PZ/TZ 豁免强度增强。三个条件因此构成同层级比较：

    - ``feature_no_gate`` ↔ ``feature_image_gate``：只归因于 feature gate（同一 stem/投影/backbone）；
    - ``feature_image_gate`` ↔ ``feature_anatomy_gate``：只归因于 PZ/TZ 条件（同 stem/投影/损失/采样），
      但两者的 ``gate.conv1.weight`` 相差 ``gate_hidden_channels * 2`` 个参数
      （见 ``feature_gate_parameter_delta``），因此**不得**声称参数量严格相同。
    """

    #: 子类声明其严格期望的输入通道数（no-gate / image = 3，anatomy = 5）
    expected_input_channels: int = MRI_CHANNELS
    #: 子类声明 prior 通道数（no-gate / image = 0，anatomy = 2）
    num_prior_channels: int = 0
    #: 子类声明是否使用 feature gate（no-gate 为 False）
    feature_uses_gate: bool = True

    @staticmethod
    def _build_feature_network(
        architecture_class_name,
        arch_init_kwargs,
        arch_init_kwargs_req_import,
        num_input_channels,
        num_output_channels,
        enable_deep_supervision,
        *,
        expected_input_channels,
        num_prior_channels,
        use_gate,
    ) -> nn.Module:
        if num_input_channels != expected_input_channels:
            raise ValueError(
                f"该 Trainer 严格要求 {expected_input_channels} 个输入通道"
                f"（{MRI_CHANNELS} MRI + {num_prior_channels} prior），收到 {num_input_channels}"
            )
        # backbone 永远只吃 3 个 MRI 通道；由 nnU-Net 官方工具按 plans 构建并初始化，
        # 不复制 encoder/decoder、不修改 plans（Dataset605/606 的 3d_fullres 均为 PlainConvUNet）。
        backbone = get_network_from_plans(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            MRI_CHANNELS,
            num_output_channels,
            allow_init=True,
            deep_supervision=enable_deep_supervision,
        )
        return FeatureFusionNNUNet(
            backbone,
            stem_channels=FEATURE_STEM_CHANNELS,
            use_gate=use_gate,
            num_prior_channels=num_prior_channels,
        )


class nnUNetTrainerPICAI_FeatureNoGate_PositiveSampling_NoFFT(
    _FeatureFusionTrainerBase
):
    """``feature_no_gate_positive_sampling``：三个独立浅层 stem + 投影，**无 gate**。

    对应 Research Plan §8.10 的 ``Feature no gate`` 条件
    ``Y_hat = F_θ(P_ψ([H_m]_m))``。它隔离**浅层编码与投影本身**的作用：与
    ``feature_image_gate`` 使用同一 stem、同一投影、同一 backbone 结构、同一损失与采样，唯一差异
    是跳过 ``S`` 的生成与相乘。

    网络：Dataset605（3 通道）→ 三条序列各自的 2 层 3×3×3 stem（``C_s = 8``，参数不共享、不下采样）
    → 拼接 24 通道 → 1×1×1 投影到 3 通道 → 原生 ``PlainConvUNet``。不含 gate，``state_dict`` 中
    不含任何 ``gate.*`` 键。本类不复用/不覆盖任何既有输出目录。
    """

    expected_input_channels = MRI_CHANNELS
    num_prior_channels = 0
    feature_uses_gate = False

    @staticmethod
    def build_network_architecture(
        architecture_class_name,
        arch_init_kwargs,
        arch_init_kwargs_req_import,
        num_input_channels,
        num_output_channels,
        enable_deep_supervision=True,
    ) -> nn.Module:
        return _FeatureFusionTrainerBase._build_feature_network(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision,
            expected_input_channels=MRI_CHANNELS,
            num_prior_channels=0,
            use_gate=False,
        )


class nnUNetTrainerPICAI_FeatureImageGate_PositiveSampling_NoFFT(
    _FeatureFusionTrainerBase
):
    """``feature_image_gate_positive_sampling``：feature gate 只依据三个 stem 特征。

    对应 Research Plan §8.10 的 ``Feature image gate`` 条件
    ``Y_hat = F_θ(P_ψ([S_φ(H)_m·H_m]_m))``。与 ``feature_no_gate`` 的唯一差异是 gate；
    与 ``feature_anatomy_gate`` 的唯一差异是 gate 条件（无 PZ/TZ）。

    gate 读**拼接后的 stem 特征**（``3*C_s`` 通道）并输出恰好 3 个 logit，``S = 3·softmax(A)``；
    ``S_m`` 在 ``H_m`` 的全部 ``C_s`` 个通道上共享（每个序列只有一个尺度场）。末层零初始化使
    初始 ``S ≡ 1``，因此它在共享相同 stem/投影/backbone 权重时**初始逐值等价于**
    ``feature_no_gate``，但**不**等价于原生输入级 nnU-Net（stem 与投影仍然改变输入）。
    """

    expected_input_channels = MRI_CHANNELS
    num_prior_channels = 0
    feature_uses_gate = True

    @staticmethod
    def build_network_architecture(
        architecture_class_name,
        arch_init_kwargs,
        arch_init_kwargs_req_import,
        num_input_channels,
        num_output_channels,
        enable_deep_supervision=True,
    ) -> nn.Module:
        return _FeatureFusionTrainerBase._build_feature_network(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision,
            expected_input_channels=MRI_CHANNELS,
            num_prior_channels=0,
            use_gate=True,
        )


class nnUNetTrainerPICAI_FeatureAnatomyGate_PositiveSampling_NoFFT(
    _FeatureFusionTrainerBase
):
    """``feature_anatomy_gate_positive_sampling``：feature gate 额外以 PZ/TZ 为条件。

    对应 Research Plan §8.10 的 ``Feature anatomy gate`` 条件
    ``Y_hat = F_θ(P_ψ([S_φ(H,Z)_m·H_m]_m))``，Dataset606（5 通道：T2W/ADC/HBV + PZ/TZ）。

    结构上与 ``feature_image_gate`` **只有 gate 条件不同**：同样的三个 stem（各 2 层 3×3×3，
    ``C_s = 8``，参数不共享、不下采样）、同样的 1×1×1 投影、同样的 FLCE 损失、同样的阳性病例采样、
    同样的 optimizer / 调度 / epoch / deep supervision。

    数据路径约束（与输入级 anatomy_gate 一致）：

    - PZ/TZ **只**进入 feature gate：不进入任何 stem、不进入投影层、不进入 backbone（禁止 WG）；
    - 进入 gate 前先 ``clamp(0, 1)``，避免把 spline 过冲当成概率；
    - 强度增强经 :meth:`get_training_transforms` 仍只作用于前 3 个 MRI 通道，空间/镜像增强仍同步
      作用于 MRI 与 PZ/TZ。

    与 ``feature_image_gate`` 的参数量差异是 gate 首层的输入维（``3*C_s+2`` vs ``3*C_s``），
    即 ``gate_hidden_channels * 2`` 个参数（C_s=8 时为 16 个）；其含义与边界见
    :func:`zonal_reliability_fusion.nnunet.networks.feature_gate_parameter_delta`。
    此外两者数据集不同（Dataset606 vs Dataset605），因此 RQ4 的归因仍受"前三个 MRI 通道逐数组
    一致性审计"这一前提约束（见 README）。
    """

    expected_input_channels = ANATOMY_INPUT_CHANNELS
    num_prior_channels = PRIOR_CHANNELS_ANATOMY
    feature_uses_gate = True

    @staticmethod
    def build_network_architecture(
        architecture_class_name,
        arch_init_kwargs,
        arch_init_kwargs_req_import,
        num_input_channels,
        num_output_channels,
        enable_deep_supervision=True,
    ) -> nn.Module:
        return _FeatureFusionTrainerBase._build_feature_network(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision,
            expected_input_channels=ANATOMY_INPUT_CHANNELS,
            num_prior_channels=PRIOR_CHANNELS_ANATOMY,
            use_gate=True,
        )

    @staticmethod
    def get_training_transforms(*args, **kwargs):
        # 直接复用输入级 anatomy_gate 已定义的增强边界（同一段代码，未修改其定义、未复制其实现）：
        # 先经 NoFFT 修复的默认管线，再把强度变换限制到前 3 个 MRI 通道。
        return nnUNetTrainerPICAI_AnatomyGate.get_training_transforms(*args, **kwargs)


class _ShortZonalFusionTrainer(_FeatureFusionTrainerBase):
    """独立 100-epoch 探索分支；四个条件均读 Dataset606 的五通道。

    只调整预算和融合方式。保持 FLCE、阳性采样、原生训练/验证/推理机制；100 epochs 在
    initialize/configure_optimizers 之前设置，使 PolyLR 总周期同步为 100。旧 Trainer 不变。
    """

    fusion_mode = "plain"
    expected_input_channels = ANATOMY_INPUT_CHANNELS
    num_prior_channels = PRIOR_CHANNELS_ANATOMY

    # 注意：不要覆写 __init__。nnUNetTrainer.__init__ 用
    #   for k in inspect.signature(self.__init__).parameters.keys(): self.my_init_kwargs[k] = locals()[k]
    # 从**子类**签名反射参数名，却从**父类**作用域的 locals() 取值；任何 *args/**kwargs 形式的
    # 子类 __init__ 都会让反射拿到 'args'/'kwargs' 这两个父类没有的名字，直接 KeyError: 'args'
    # （曾导致四个 *_100ep 变体在构造阶段 5 秒内全部失败）。预算与 spacing 校验因此放在 initialize()。

    def initialize(self):
        # initialize() 在 nnUNetTrainer.__init__ 末尾即被调用，早于 perform_actual_training 中的
        # configure_optimizers()（那里才用 self.num_epochs 构造 PolyLRScheduler），因此此处赋值
        # 既能同步 100 epoch 的训练循环上界，也能同步 PolyLR 总周期；resume 时同样经由
        # load_checkpoint -> initialize() 恢复，不依赖 __init__ 的 my_init_kwargs。
        spacing = tuple(float(s) for s in self.configuration_manager.spacing)
        if len(spacing) != 3 or any(abs(a - b) > 1e-6 for a, b in zip(spacing, (3.0, .5, .5))):
            raise ValueError("100ep 同区参照分支限定当前 3d_fullres spacing [3.0,0.5,0.5]")
        self.num_epochs = 100
        super().initialize()
        self.print_to_log_file(
            "Short zonal fusion:", f"mode={self.fusion_mode}",
            f"epochs={self.num_epochs}", f"iterations_per_epoch={self.num_iterations_per_epoch}",
            "input_channels=5 (T2W/ADC/HBV/PZ/TZ)",
            "priors=algorithmic_zonal_membership_weights_from_interpolated_binary_masks; "
            "not_calibrated_probability_or_measured_geometric_voxel_fraction; "
            "pooled_mean_and_support=algorithm_weighted_statistics; "
            "reference_includes_center_and_may_include_lesions_not_normal_tissue_truth",
        )

    @classmethod
    def build_network_architecture(
        cls, architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import,
        num_input_channels, num_output_channels, enable_deep_supervision=True,
    ):
        if num_input_channels != ANATOMY_INPUT_CHANNELS:
            raise ValueError("100ep 分支全部要求五通道 Dataset606；禁止把缺失分区当成零输入")
        if cls.fusion_mode in ("plain", "zone_gate"):
            return cls._build_feature_network(
                architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import,
                num_input_channels, num_output_channels, enable_deep_supervision,
                expected_input_channels=ANATOMY_INPUT_CHANNELS,
                num_prior_channels=PRIOR_CHANNELS_ANATOMY,
                use_gate=cls.fusion_mode == "zone_gate",
            )
        if cls.fusion_mode not in ("reference_fixed", "reference_adaptive"):
            raise ValueError(f"未知融合模式 {cls.fusion_mode}")
        backbone = get_network_from_plans(
            architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import,
            MRI_CHANNELS, num_output_channels, allow_init=True,
            deep_supervision=enable_deep_supervision,
        )
        return ZonalReferenceFusionNNUNet(
            backbone, adaptive=cls.fusion_mode == "reference_adaptive",
            stem_channels=FEATURE_STEM_CHANNELS,
        )

    @staticmethod
    def get_training_transforms(*args, **kwargs):
        return nnUNetTrainerPICAI_AnatomyGate.get_training_transforms(*args, **kwargs)


class nnUNetTrainerPICAI_FeatureNoGate_PositiveSampling_100ep_NoFFT(_ShortZonalFusionTrainer):
    """相同五通道数据的 MRI-only 普通特征融合对照；PZ/TZ 不参与预测。"""
    fusion_mode = "plain"


class nnUNetTrainerPICAI_FeatureAnatomyGate_PositiveSampling_100ep_NoFFT(_ShortZonalFusionTrainer):
    """普通分区条件 feature gate，100-epoch 匹配参照。"""
    fusion_mode = "zone_gate"


class nnUNetTrainerPICAI_ZonalReference_PositiveSampling_100ep_NoFFT(_ShortZonalFusionTrainer):
    """同区参照修正：有效参照区域使用固定强度，局部无支持时回退。"""
    fusion_mode = "reference_fixed"


class nnUNetTrainerPICAI_ZonalReferenceAdaptive_PositiveSampling_100ep_NoFFT(_ShortZonalFusionTrainer):
    """同区参照修正 + 学习式强度；强度不是校准质量/置信度。"""
    fusion_mode = "reference_adaptive"


