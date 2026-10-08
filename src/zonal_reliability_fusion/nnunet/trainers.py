"""项目对 nnU-Net v2.6.2 的 Trainer 扩展（新主线 + 兼容再导出）。

研究主线：**Anatomy-Guided Lesion-Aware Coarse-to-Fine Prostate Cancer Segmentation**。
所有 Trainer 共享同一套 nnU-Net 训练机制（optimizer=SGD+Nesterov、PolyLR、增强主体、
deep supervision、checkpoint/resume、validation、sliding-window inference）；差异只通过覆盖
nnU-Net 的钩子表达，**绝不重实现训练循环**。

**ACTIVE**（新主线，默认训练视图只应看到这些）
---------------------------------------------

==============================  ==========================================================  =========
variant（CLI 名）                做法                                                        条件
==============================  ==========================================================  =========
``positive_sampling``           strong baseline **A1**：FLCE + 阳性采样                     A
``dicece_positive_sampling``    strong baseline **A2**：原生 Dice+CE + 阳性采样（只差损失） A
``lesion_roi``                  A + Anatomy-Guided ROI（解剖约束的 patch 采样空间）         B
``lesion_coarse_to_fine``       B + coarse lesionness 头 + soft 残差 refinement             C
``lesion_zone_refine``          C + soft PZ/TZ 解剖上下文（zone-aware refinement）           D
``lesion_hard_negative``        best(C, D) + 解剖约束的困难负样本挖掘（可开关）              E
``anatomy_joint_100ep``         **Stage 1**：T2W -> WG/PZ/TZ 先验生成器                    Stage 1
==============================  ==========================================================  =========

**LEGACY**（归档，见 ``legacy/fusion_trainers.py``；仅通过 ``--legacy`` 可见）
------------------------------------------------------------------------------
``baseline`` / ``optimized_baseline`` / ``image_gate`` / ``anatomy_gate`` /
``image_gate_positive_sampling`` / ``anatomy_gate_positive_sampling`` /
``feature_no_gate_positive_sampling`` / ``feature_image_gate_positive_sampling`` /
``feature_anatomy_gate_positive_sampling`` / 四个 ``*_100ep`` 同区参照条件。
它们保留是为了 **checkpoint 兼容、历史实验可复现、以及作为新主线的 preliminary /
negative evidence**；不得进入新方法的默认训练流程。

单变量纪律
----------
每个 ACTIVE 条件只允许相对其**上一条**条件改变一件事：

- B 相对 A：只改训练 patch 采样空间（网络、损失、增强、optimizer 全不变）；
- C 相对 B：只加 coarse lesionness 头 + soft refinement + lesionness 辅助损失；
- D 相对 C：只加 soft PZ/TZ 解剖上下文（多两个输入通道）；
- E 相对 best(C, D)：只加困难负样本采样（``hard_negative_set_path`` 为 ``None`` 时完全退化）。

详见 ``docs/Experiment_Plan.md`` 与 :mod:`zonal_reliability_fusion.lesion.baseline`。
"""

from __future__ import annotations

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from torch import nn

from zonal_reliability_fusion.anatomy.contracts import (
    ANATOMY_CLASS_ORDER,
    ANATOMY_CONFIGURATION,
    ANATOMY_COUNTS,
    ANATOMY_DATASET,
    ANATOMY_GEOMETRY_ATOL,
    ANATOMY_KNOWN_MISSING,
    ANATOMY_LABELS,
    anatomy_encode,
    anatomy_encode_heads,
    anatomy_geometry,
    anatomy_read_array,
    anatomy_same_grid,
    anatomy_validate_array,
    anatomy_validate_probabilities,
    validate_anatomy_dataset,
    validate_anatomy_split,
)

# ===========================================================================
# ACTIVE：新方法模块（条件 B / C / D / E）
# ===========================================================================
from zonal_reliability_fusion.lesion.coarse_to_fine import (
    LesionAwareCoarseToFineNNUNet,
)
from zonal_reliability_fusion.lesion.prior_channels import (
    MRI_CHANNELS,
    restrict_intensity_transforms_to_mri,
)
from zonal_reliability_fusion.lesion.roi import ROIError

# ===========================================================================
# ACTIVE：strong baseline（A1 / A2）与 Stage 1
# ===========================================================================
from zonal_reliability_fusion.nnunet.augmentation import (
    NoFFTAugmentationMixin,
)
from zonal_reliability_fusion.nnunet.bases import (
    nnUNetTrainerPICAI_AnatomyJoint_100ep_NoFFT,
    nnUNetTrainerPICAI_DiceCE_NoFFT,
    nnUNetTrainerPICAI_DiceCE_PositiveSampling_NoFFT,
    nnUNetTrainerPICAI_FLCE_NoFFT,
    nnUNetTrainerPICAI_FLCE_PositiveSampling_NoFFT,
)
from zonal_reliability_fusion.nnunet.losses import (
    LESIONNESS_LOSS_WEIGHT,
    LesionnessAuxiliaryLoss,
    PiCAIFocalCrossEntropyLoss,
    PICAIFocalCrossEntropyLossMixin,
)
from zonal_reliability_fusion.nnunet.roi_sampling import (
    ROISamplingMixin,
    load_prostate_roi_set,
)
from zonal_reliability_fusion.sampling.hard_negative import HardNegativeMiningMixin

#: 条件 C/D/E 的骨干输入通道数（3 MRI，或 3 MRI + 2 soft zone 概率）
LESION_MRI_CHANNELS = MRI_CHANNELS
LESION_ZONE_CHANNELS = 2

#: soft 解剖上下文（P(PZ)/P(TZ)）的来源标记；正式结果只允许 predicted prior
ANATOMY_PRIOR_SOURCE = "predicted_prior"


class _LesionTrainerBase(nnUNetTrainerPICAI_FLCE_PositiveSampling_NoFFT):
    """条件 B/C/D/E 的公共基类：阳性采样 + ROI 约束 + 原生框架。

    继承 ``nnUNetTrainerPICAI_FLCE_PositiveSampling_NoFFT``（= strong baseline A1）使全部新
    条件自动获得与 A1 **完全相同**的训练框架：PI-CAI FLCE 损失、NoFFT 修复、每批固定一个阳性
    病灶 patch、原生 optimizer / PolyLR / deep supervision / checkpoint / validation / 滑窗
    推理。子类只覆盖 ``build_network_architecture``（必要时再覆盖 ``initialize`` /
    ``_build_loss``）。

    条件 B 是 C/D/E 的**组成部分**（每一项新方法都叠加在「解剖约束的搜索空间」之上），因此
    ``roi_set_path`` 的校验放在本基类，四个条件都会执行；缺失即报错，不允许静默退化。
    """

    #: 骨干的 MRI 通道数（恒为 3；PZ/TZ 只作为额外上下文，不改 MRI 部分）
    lesion_mri_channels: int = LESION_MRI_CHANNELS
    #: 额外的 soft 区域概率通道数：0（条件 B/C）或 2（条件 D）
    lesion_zone_channels: int = 0
    #: ROI 集合 JSON 路径（由预测 WG 生成）；``None`` 时训练入口拒绝启动
    roi_set_path: str | None = None
    #: ROI 内采样槽位比例（预先冻结的工程常量，见 docs/Experiment_Plan.md）
    roi_sampling_probability: float = 0.75

    def _expected_input_channels(self) -> int:
        return self.lesion_mri_channels + self.lesion_zone_channels

    def _assert_input_channels(self, num_input_channels: int) -> None:
        expected = self._expected_input_channels()
        if int(num_input_channels) != expected:
            raise ValueError(
                f"{type(self).__name__} 严格要求 {expected} 个输入通道"
                f"（{self.lesion_mri_channels} MRI + {self.lesion_zone_channels} zone），"
                f"收到 {num_input_channels}；拒绝把缺失的先验通道静默当成零输入"
            )

    def _roi_payload(self) -> dict:
        if self.roi_set_path is None:
            raise ValueError(
                f"{type(self).__name__} 必须显式提供 roi_set_path（条件 B 的 Anatomy-Guided "
                "ROI 是 C/D/E 的组成部分）；在路径缺失时拒绝静默退化回 strong baseline"
            )
        return load_prostate_roi_set(
            self.roi_set_path,
            expected_dataset_name=self.plans_manager.dataset_name,
            expected_fold=self.fold,
        )

    def initialize(self):
        payload = self._roi_payload()
        super().initialize()
        self.print_to_log_file(
            "Anatomy-guided ROI enabled:\n"
            f"source={self.roi_set_path}\n"
            f"cases={len(payload['cases'])}\n"
            f"margin_mm={payload['thresholds']['margin_mm']}\n"
            f"wg_threshold={payload['thresholds']['wg_threshold']}\n"
            f"prior_source={payload['provenance'].get('prior_source')}\n"
            f"roi_sampling_probability={self.roi_sampling_probability}"
        )

    @classmethod
    def get_training_transforms(cls, *args, **kwargs):
        """原生默认增强（含 NoFFT 修复）+ 在 zone 条件下把强度变换限制到 MRI 通道。

        条件 B/C（``lesion_zone_channels == 0``）下与 strong baseline 的增强**逐值一致**；
        只有条件 D 才需要先验通道豁免（先验是概率图，对其施加 noise/blur/brightness 会破坏
        概率语义）。空间/镜像变换始终同步作用于全部通道。
        """
        transforms = (
            nnUNetTrainerPICAI_FLCE_PositiveSampling_NoFFT.get_training_transforms(
                *args, **kwargs
            )
        )
        if cls.lesion_zone_channels == 0:
            return transforms
        transforms, n_wrapped = restrict_intensity_transforms_to_mri(
            transforms, cls.lesion_mri_channels
        )
        if n_wrapped == 0:
            raise RuntimeError(
                f"{cls.__name__} 未能在增强管线中找到任何强度变换来限制到 MRI 通道；"
                "拒绝静默启动，以免把 noise/blur/brightness/contrast/gamma 施加到 zone 概率上。"
            )
        return transforms

    def _build_lesion_network(
        self,
        architecture_class_name,
        arch_init_kwargs,
        arch_init_kwargs_req_import,
        num_input_channels,
        num_output_channels,
        enable_deep_supervision,
        *,
        zone_mode: str = "concat",
        use_lesionness_guidance: bool = True,
    ) -> nn.Module:
        """构建原生 backbone（3 MRI 通道）+ lesion-aware coarse-to-fine 包装。

        backbone 仍由 nnU-Net 官方工具按 plans 构建（``allow_init=True``），不复制
        encoder/decoder；本方法只负责把输入通道数校验与包装器的常量对齐。
        """
        from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

        self._assert_input_channels(num_input_channels)
        backbone = get_network_from_plans(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            self.lesion_mri_channels,
            num_output_channels,
            allow_init=True,
            deep_supervision=enable_deep_supervision,
        )
        return LesionAwareCoarseToFineNNUNet(
            backbone,
            num_segmentation_heads=num_output_channels,
            mri_channels=self.lesion_mri_channels,
            num_anatomy_channels=self.lesion_zone_channels,
            zone_mode=zone_mode,
            use_lesionness_guidance=use_lesionness_guidance,
        )

    def _make_lesionness_loss(self, main_loss: nn.Module) -> nn.Module:
        """把主损失包装成 ``main + weight * lesionness``；spacing 由 plans 显式提供。

        ``configuration_manager.spacing`` 与预处理数组的轴序一致（nnU-Net 的
        ``spacing_after_transpose``），因此可以直接作为物理半径膨胀所需的 ``spacing_zyx``，
        不需要额外的坐标变换。
        """
        spacing = tuple(float(s) for s in self.configuration_manager.spacing)
        if len(spacing) != 3:
            raise ValueError(
                f"条件 C/D/E 只支持 3D 配置；{self.configuration_name} 的 spacing 为 {spacing}"
            )
        return LesionnessAuxiliaryLoss(
            main_loss, network=self.network, spacing_zyx=spacing
        )


# --------------------------------------------------------------------------- 条件 B
class nnUNetTrainerPICAI_LesionROI_NoFFT(ROISamplingMixin, _LesionTrainerBase):
    """**条件 B**：A + Anatomy-Guided ROI（用预测 WG 约束训练 patch 采样空间）。

    相对 strong baseline（A）的**唯一**改变是训练 patch 的采样空间：ROI 内的槽位比例由
    :attr:`roi_sampling_probability` 控制，`roi_set_path` 指向由预测 WG 生成的前列腺 ROI。

    为什么是「采样空间约束」而不是「硬裁剪输入」：

    - ``image * wg_mask`` 会把 WG 边界错误直接变成「病灶被删除」，不可恢复；
    - 裁剪输入会在推理时引入 train/test 不一致，并需要额外的几何恢复步骤；
    - **约束采样空间**直接回答条件 B 的问题「单纯减少无关背景是否改善 lesion learning」，
      且**保留**一部分全视野采样槽位，使模型仍见到滑窗推理时会遇到的背景分布。

    预测 WG 来源、阈值与物理 margin 全部在 ROI 集合中显式记录（``prior_source`` /
    ``wg_threshold`` / ``margin_mm``），并在 ``initialize`` 时逐项校验。

    ``ROISamplingMixin`` 提供训练 loader 的 ROI 约束；验证 loader 保持原生
    ``nnUNetDataLoader``，无验证泄漏。阳性采样仍然生效（每批固定一个阳性病灶 patch），
    ROI 约束叠加在同一 loader 上（阳性槽位的病灶中心裁剪同样落在 ROI 内）。
    """

    #: zone 条件化模式（条件 D 才生效）；条件 B/C 固定 ``concat`` 且无 zone 通道
    lesion_zone_mode: str = "concat"
    #: 是否使用 lesionness soft guidance（条件 C 的消融开关；条件 B 不构建 lesionness 头）
    lesion_use_lesionness_guidance: bool = True

    @classmethod
    def build_network_architecture(
        cls,
        architecture_class_name,
        arch_init_kwargs,
        arch_init_kwargs_req_import,
        num_input_channels,
        num_output_channels,
        enable_deep_supervision=True,
    ) -> nn.Module:
        """条件 B 仍使用**原生 backbone**：ROI 只改采样空间，不改网络结构。

        这是「B 相对 A 只差采样空间」这一单变量边界在代码上的体现——本类不覆盖
        ``build_network_architecture`` 的实现会被子类 C 覆盖，因此这里显式保留原生路径。
        """
        from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

        cls._assert_input_channels(num_input_channels)
        return get_network_from_plans(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            allow_init=True,
            deep_supervision=enable_deep_supervision,
        )


# --------------------------------------------------------------------------- 条件 C
class nnUNetTrainerPICAI_LesionCoarseToFine_NoFFT(
    nnUNetTrainerPICAI_LesionROI_NoFFT
):
    """**条件 C**：B + lesion-aware coarse-to-fine（本论文的核心方法比较）。

    相对 B 的改动只有三处，构成一个不可拆分的整体（条件 C 就是「显式 lesion localization」）：

    1. 网络：原生 backbone 之外加一个 coarse lesionness 头（最粗尺度，1×1×1）与一个
       **末层零初始化**的 soft 残差 refinement（``F_out = F + delta``）；
    2. 损失：在原生主损失之上加一项 lesionness 辅助监督（``main + 0.5 * aux``），
       目标为由 lesion GT 按**物理半径**膨胀得到的粗掩膜；
    3. 采样：沿用 B 的 ROI 约束 + 阳性采样，不变。

    由于 refinement 末层零初始化，**初始状态与 B 逐值一致**，因此两者可以直接单变量比较。
    约束（逐条对应 Research Plan 的 coarse-to-fine 原则）：coarse 预测从不硬裁区域；
    lesionness 只作为 soft guidance；零 coarse score 区域仍有完整 residual path；不引入
    Transformer / Mamba。
    """

    lesion_zone_channels: int = 0

    @classmethod
    def build_network_architecture(
        cls,
        architecture_class_name,
        arch_init_kwargs,
        arch_init_kwargs_req_import,
        num_input_channels,
        num_output_channels,
        enable_deep_supervision=True,
    ) -> nn.Module:
        return cls._build_lesion_network(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision,
            zone_mode=cls.lesion_zone_mode,
            use_lesionness_guidance=cls.lesion_use_lesionness_guidance,
        )

    def _build_loss(self) -> nn.Module:
        """原生主损失 + lesionness 辅助损失（同一次 backward 联合训练）。"""
        return self._make_lesionness_loss(super()._build_loss())

# --------------------------------------------------------------------------- 条件 D
class nnUNetTrainerPICAI_LesionZoneRefine_NoFFT(
    nnUNetTrainerPICAI_LesionCoarseToFine_NoFFT
):
    """**条件 D**：C + zone-aware refinement（soft PZ/TZ 解剖上下文）。

    相对 C 的**唯一**改动是额外读入两个 soft 概率通道 ``P(PZ) / P(TZ)``（由冻结的 Stage-1
    anatomy 模型对该病例自身 MRI 预测得到），并用 ``P(U) = 1 - max(P(PZ), P(TZ))`` 作为
    分区边界 / 解剖不确定区的权重。它回答：

    > 在已经具有 lesion-aware coarse-to-fine learning 之后，PZ/TZ 是否能够进一步改善病灶
    > 完整分割或控制 FP？

    设计纪律（避免重蹈「为了 anatomy 而设计复杂 Gate」）：

    - PZ/TZ **只**进入 refinement 的 context，不进入 MRI 分支、不改变 modality 的任何权重；
    - 默认 :attr:`lesion_zone_mode` 是 ``"concat"``（最简、参数最少）。``"zone_experts"``
      是**候选**，只有在 ``concat`` 被证明不足、且参数量差异可控时才启用；
      :func:`..lesion.zone_conditioning.zone_mode_parameter_delta` 给出两者的参数差；
    - 若 D 相对 C 无稳定收益，按 stop rule 立即停止 anatomy 架构扩张（见
      ``docs/Experiment_Plan.md``），最终模型停在 C。
    """

    lesion_zone_channels: int = LESION_ZONE_CHANNELS
    lesion_zone_mode: str = "concat"

    def initialize(self):
        super().initialize()
        if self.num_input_channels != self._expected_input_channels():
            raise ValueError(
                f"条件 D 需要 {self._expected_input_channels()} 个输入通道"
                f"（3 MRI + 2 zone 概率）；Dataset 的 channel_names 数量为 "
                f"{self.num_input_channels}。请使用带 P(PZ)/P(TZ) 通道的数据集，"
                "禁止把缺失的 zone 概率静默当成零输入。"
            )
        self.print_to_log_file(
            "Zone-aware refinement enabled:\n"
            f"zone_mode={self.lesion_zone_mode}\n"
            f"zone_channels={self.lesion_zone_channels}\n"
            f"prior_source={ANATOMY_PRIOR_SOURCE}（必须来自该病例自身 MRI 的预测）"
        )


# --------------------------------------------------------------------------- 条件 E
class nnUNetTrainerPICAI_LesionHardNegative_NoFFT(
    HardNegativeMiningMixin, nnUNetTrainerPICAI_LesionCoarseToFine_NoFFT
):
    """**条件 E**：best(C, D) + 解剖约束的困难负样本挖掘（training strategy enhancement）。

    相对 best(C, D) 的**唯一**改动是训练采样：``hard_negative_set_path`` 指向 Round-2 在
    **训练 split 内**挖掘得到的困难负样本集合（高置信假阳、且落在预测 WG 内）。该位置来自
    第一轮模型对训练病例的推理，因此：

    - 挖掘只在 training split 内执行（:func:`..sampling.hard_negative.validate_mining_scope`）；
    - validation / test 绝不参与挖掘；
    - ``hard_negative_set_path is None`` 时本类**完全退化**为条件 C/D 的行为，因此「开 / 关」
      是同一个 Trainer 类的一个参数，可直接作为独立 ablation。

    定位：若效果稳定则进入最终方法；若效果有限，则作为 ablation / supplementary experiment，
    不强行塞进最终模型。
    """

    hard_negative_set_path: str | None = None
    hard_negative_cases_per_batch: int = 1

    def initialize(self):
        super().initialize()
        self.print_to_log_file(
            "Hard-negative condition:\n"
            f"enabled={self.hard_negative_set_path is not None}\n"
            f"source={self.hard_negative_set_path}\n"
            "scope=training split only（validation 绝不参与挖掘）"
        )


# ===========================================================================
# LEGACY：全部旧条件（兼容再导出；实现与类名逐字保留在 legacy/）
# ===========================================================================
from zonal_reliability_fusion.legacy.fusion_trainers import (
    _FeatureFusionTrainerBase,
    _GatedTrainerBase,
    nnUNetTrainerPICAI_AnatomyGate,
    nnUNetTrainerPICAI_AnatomyGate_PositiveSampling_NoFFT,
    nnUNetTrainerPICAI_FeatureAnatomyGate_PositiveSampling_100ep_NoFFT,
    nnUNetTrainerPICAI_FeatureAnatomyGate_PositiveSampling_NoFFT,
    nnUNetTrainerPICAI_FeatureImageGate_PositiveSampling_NoFFT,
    nnUNetTrainerPICAI_FeatureNoGate_PositiveSampling_100ep_NoFFT,
    nnUNetTrainerPICAI_FeatureNoGate_PositiveSampling_NoFFT,
    nnUNetTrainerPICAI_ImageGate,
    nnUNetTrainerPICAI_ImageGate_PositiveSampling_NoFFT,
    nnUNetTrainerPICAI_ZonalReference_PositiveSampling_100ep_NoFFT,
    nnUNetTrainerPICAI_ZonalReferenceAdaptive_PositiveSampling_100ep_NoFFT,
)

#: 新主线（ACTIVE）Trainer：默认训练视图只暴露这些
ACTIVE_TRAINERS: tuple[type[nnUNetTrainer], ...] = (
    nnUNetTrainerPICAI_FLCE_PositiveSampling_NoFFT,
    nnUNetTrainerPICAI_DiceCE_PositiveSampling_NoFFT,
    nnUNetTrainerPICAI_LesionROI_NoFFT,
    nnUNetTrainerPICAI_LesionCoarseToFine_NoFFT,
    nnUNetTrainerPICAI_LesionZoneRefine_NoFFT,
    nnUNetTrainerPICAI_LesionHardNegative_NoFFT,
    nnUNetTrainerPICAI_AnatomyJoint_100ep_NoFFT,
)

#: 归档（LEGACY）Trainer：需要 ``--legacy`` 才出现在训练入口的可见列表里
LEGACY_TRAINERS: tuple[type[nnUNetTrainer], ...] = (
    nnUNetTrainerPICAI_FLCE_NoFFT,
    nnUNetTrainerPICAI_DiceCE_NoFFT,
    nnUNetTrainerPICAI_ImageGate,
    nnUNetTrainerPICAI_AnatomyGate,
    nnUNetTrainerPICAI_ImageGate_PositiveSampling_NoFFT,
    nnUNetTrainerPICAI_AnatomyGate_PositiveSampling_NoFFT,
    nnUNetTrainerPICAI_FeatureNoGate_PositiveSampling_NoFFT,
    nnUNetTrainerPICAI_FeatureImageGate_PositiveSampling_NoFFT,
    nnUNetTrainerPICAI_FeatureAnatomyGate_PositiveSampling_NoFFT,
    nnUNetTrainerPICAI_FeatureNoGate_PositiveSampling_100ep_NoFFT,
    nnUNetTrainerPICAI_FeatureAnatomyGate_PositiveSampling_100ep_NoFFT,
    nnUNetTrainerPICAI_ZonalReference_PositiveSampling_100ep_NoFFT,
    nnUNetTrainerPICAI_ZonalReferenceAdaptive_PositiveSampling_100ep_NoFFT,
)

#: 项目 Trainer 名 -> 类。训练入口与预测入口都用它做**进程内直接映射**，
#: 避免依赖 nnU-Net 的 ``recursive_find_python_class``（该函数只在 nnunetv2 包目录内递归扫描，
#: 找不到项目自定义类，且可能扫到环境中其他 nnunetv2 分支）。
#:
#: 注册表必须包含**全部历史类名**：输出目录由 Trainer 类名决定，缺一个类名就会让对应的既有
#: checkpoint / validation 产物再也无法被本项目解析。
PROJECT_TRAINERS: dict[str, type[nnUNetTrainer]] = {
    trainer.__name__: trainer for trainer in (*ACTIVE_TRAINERS, *LEGACY_TRAINERS)
}


def resolve_trainer_class(trainer_name: str) -> type[nnUNetTrainer] | None:
    """按类名解析项目 Trainer；不是项目 Trainer 时返回 ``None``（由调用方决定是否回退）。"""
    return PROJECT_TRAINERS.get(trainer_name)


__all__ = (
    "ACTIVE_TRAINERS",
    "ANATOMY_DATASET",
    "ANATOMY_LABELS",
    "LEGACY_TRAINERS",
    "PROJECT_TRAINERS",
    "ROIError",
    "nnUNetTrainerPICAI_LesionCoarseToFine_NoFFT",
    "nnUNetTrainerPICAI_LesionHardNegative_NoFFT",
    "nnUNetTrainerPICAI_LesionROI_NoFFT",
    "nnUNetTrainerPICAI_LesionZoneRefine_NoFFT",
    "resolve_trainer_class",
)
