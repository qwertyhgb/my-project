"""新主线与历史训练线**共用**的 Trainer 基类（不含任何研究性模块）。

这一层刻意做得很薄：只放「strong baseline 本身」与「Stage-1 解剖先验模型」。把它们与
新方法（``lesion/``）和 legacy 融合线（``legacy/fusion_trainers.py``）分开，是为了让
「某个新模块到底改了 baseline 的什么」这个问题在代码结构上可以直接回答：

- 本文件里的每个类都**不含** gate / fusion / ROI / lesionness / zone 条件化；
- 新方法类只在 ``trainers.py`` 中、以**继承本文件**的方式叠加，一次只叠加一个模块。

包含的训练条件
--------------
======================================  ================================================================
类名                                    定位
======================================  ================================================================
``nnUNetTrainerPICAI_FLCE_NoFFT``       既有的 PI-CAI Focal+CE 基线（已完成的 N0；保留以支持续训/验证）
``nnUNetTrainerPICAI_DiceCE_NoFFT``     原生 Dice+CE 单变量基线（**用户已中止**，见类 docstring）
``nnUNetTrainerPICAI_FLCE_PositiveSampling_NoFFT``    strong baseline 候选 **A1**
``nnUNetTrainerPICAI_DiceCE_PositiveSampling_NoFFT``  strong baseline 候选 **A2**
``nnUNetTrainerPICAI_AnatomyJoint_100ep_NoFFT``       Stage-1 解剖先验生成器（WG/PZ/TZ region heads）
======================================  ================================================================

**A1 ↔ A2 的唯一差异必须是损失函数。** 其余一切（网络、数据集、split、patch/batch size、
增强、前景采样、deep supervision、optimizer、PolyLR、epoch、checkpoint、validation、
滑窗推理）都继承同一套 nnU-Net 原生实现，因此这一点在代码上是可以逐项核对的。
"""

from __future__ import annotations

import nnunetv2.paths
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

from zonal_reliability_fusion.anatomy.contracts import (
    ANATOMY_DATASET,
    validate_anatomy_dataset,
    validate_anatomy_split,
)
from zonal_reliability_fusion.nnunet.augmentation import NoFFTAugmentationMixin
from zonal_reliability_fusion.nnunet.losses import PICAIFocalCrossEntropyLossMixin
from zonal_reliability_fusion.sampling.positive_sampling import PositiveCaseSamplingMixin


class nnUNetTrainerPICAI_FLCE_NoFFT(
    NoFFTAugmentationMixin, PICAIFocalCrossEntropyLossMixin, nnUNetTrainer
):
    """PI-CAI Focal+CE 基线：原生 ``PlainConvUNet`` + 3 个 MRI 通道 + ``0.5*Focal(γ=2)+0.5*CE``。

    除损失与 blur benchmark 外，全部继承 nnU-Net v2.6.2 默认训练行为。类名与已完成的 N0
    运行一致，output folder 保持不变（可续训 / 只做 validation）。
    """


class nnUNetTrainerPICAI_DiceCE_NoFFT(NoFFTAugmentationMixin, nnUNetTrainer):
    """原生 nnU-Net Dice+CE 的**单变量**优化基线（**用户已中止**，仅保留以支持历史产物查询）。

    背景：已完成的 N0（``nnUNetTrainerPICAI_FLCE_NoFFT``）在 1000 epoch 中有约 700 epoch
    停留在「全部预测为背景」的状态，直到学习率衰减后才开始学习前景。

    唯一的改动是把损失换成 nnU-Net v2.6.2 原生 Dice+CE；其余一切与 FLCE 基线一致。实现上
    刻意保持「零自写损失」：基类列表里没有 ``PICAIFocalCrossEntropyLossMixin``，因此
    ``_build_loss`` 直接解析到 ``nnUNetTrainer._build_loss`` 的原生实现。

    **状态：训练已被用户中止**（最后完整 epoch 376，最大 pseudo Dice 约 0.0008，无
    ``validation/summary.json``）。**不得续训、不得引用为结果**。它在新主线的角色是
    「为什么需要先解决优化稳定性」的历史证据，而不是一个对照臂。
    """


class nnUNetTrainerPICAI_FLCE_PositiveSampling_NoFFT(
    PositiveCaseSamplingMixin,
    nnUNetTrainerPICAI_FLCE_NoFFT,
):
    """strong baseline 候选 **A1**：FLCE 基线 + 每批固定一个阳性病灶 patch。

    在 ``nnUNetTrainerPICAI_FLCE_NoFFT`` 之上**唯一**的改变是训练集病例/patch 采样：

    - 训练 loader 换成 ``PositiveCaseDataLoader``（最后 ``positive_cases_per_batch=1`` 个槽位
      固定来自当前 fold 的阳性训练病例并强制病灶中心裁剪）；
    - 验证 loader **保持原生** ``nnUNetDataLoader`` 与原生采样，无验证泄漏。

    MRO 保证其余一切与 FLCE 基线一致：损失仍为 ``0.5*Focal(γ=2)+0.5*CE``（deep supervision
    权重不变）；**不覆盖** ``build_network_architecture``（原生 ``PlainConvUNet``，3 个 MRI
    通道，不含 gate / PZ/TZ / 浅层 feature path）；optimizer、PolyLR、epoch、checkpoint、
    validation、推理全部继承原生实现。

    定位说明：阳性采样是进行 strong baseline 构建时**已经列入 baseline** 的 training
    strategy（见 :mod:`zonal_reliability_fusion.lesion.baseline`），不是论文的方法创新。
    """


class nnUNetTrainerPICAI_DiceCE_PositiveSampling_NoFFT(
    PositiveCaseSamplingMixin,
    nnUNetTrainerPICAI_DiceCE_NoFFT,
):
    """strong baseline 候选 **A2**：原生 Dice+CE + 阳性病例采样。

    相对 A1（``..._FLCE_PositiveSampling_NoFFT``）**只**把损失从
    ``0.5*Focal(γ=2)+0.5*CE`` 换成 nnU-Net v2.6.2 原生 Dice+CE；相对
    ``nnUNetTrainerPICAI_DiceCE_NoFFT`` **只**把训练采样换成 ``PositiveCaseDataLoader``。

    MRO 与钩子来源（``PositiveCaseSamplingMixin`` 在最前）：

    - ``_build_loss``：本类与 ``nnUNetTrainerPICAI_DiceCE_NoFFT`` 都不覆盖，解析到
      ``nnUNetTrainer._build_loss`` 的**原生实现**（``DC_and_CE_loss`` +
      ``MemoryEfficientSoftDiceLoss``，deep supervision 由原生 ``DeepSupervisionWrapper``
      按 ``1/2^i`` 加权）；``PICAIFocalCrossEntropyLossMixin`` **不在** MRO 中，项目没有自写
      Dice+CE；
    - ``get_dataloaders``：``PositiveCaseSamplingMixin``（训练 loader 为阳性采样 loader；
      验证 loader 保持原生）；
    - ``get_training_transforms``：``NoFFTAugmentationMixin``（同一 NoFFT 修复）；
    - ``build_network_architecture``：不覆盖，由 plans 构建原生 ``PlainConvUNet``。

    ``initialize`` 只在原生初始化之后追加一行可审计日志（实际构建出的 loss / 网络 / 通道数），
    不改变任何训练行为。输出目录由类名自然形成为独立目录，不从任何既有 checkpoint 续训。
    """

    def initialize(self):
        """原生 ``initialize()`` + 一行可审计的启动配置日志（不复制、不修改训练循环）。"""
        super().initialize()
        from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper

        base_loss = (
            self.loss.loss if isinstance(self.loss, DeepSupervisionWrapper) else self.loss
        )
        self.print_to_log_file(
            "A2 DiceCE + PositiveSampling audit:\n"
            f"trainer={type(self).__name__}\n"
            f"loss={type(self.loss).__name__} "
            f"(base={type(base_loss).__module__}.{type(base_loss).__name__})\n"
            f"network={type(self.network).__name__}\n"
            f"input_channels={self.num_input_channels}\n"
            f"output_channels={self.label_manager.num_segmentation_heads}"
        )


class nnUNetTrainerPICAI_AnatomyJoint_100ep_NoFFT(NoFFTAugmentationMixin, nnUNetTrainer):
    """**Stage 1**：T2W -> WG/PZ/TZ 的解剖先验生成器（共享原生网络，三个 region head）。

    定位：``anatomical prior generator``，**不是**论文的主要创新点。它只负责在推理时对每个
    病例生成该病例自身 MRI 的 ``P(WG)/P(PZ)/P(TZ)`` soft probability。

    100 epoch 是**工程性试跑预算**，不是研究性预算；Stage-1 的质量由
    :mod:`zonal_reliability_fusion.anatomy.validation` 的 region Dice 与
    :func:`..anatomy.validation.prior_uncertainty_report` 报告，不参与 lesion 主线的比较。

    硬约束（在 ``__init__`` 与 ``initialize`` 两处都校验，fail-closed）：

    - 只接受 ``Dataset607_PICAI_Anatomy / 3d_fullres``；
    - 单通道 T2W、三个 region head、**不允许** cascade 输入；
    - 使用**冻结的显式 split**（fold 0），无随机回退；
    - 逐病例与 provenance 契约核对，任何不一致直接报错。
    """

    def __init__(
        self,
        plans,
        configuration,
        fold,
        dataset_json,
        device=None,
    ):
        from pathlib import Path

        import torch

        if device is None:
            device = torch.device("cuda")
        validate_anatomy_dataset(dataset_json, plans.get("dataset_name"), configuration)
        # 在**调用时**读取 nnunetv2.paths.nlUNet_preprocessed，而不是 import 时快照：
        # 训练入口在设置 nnUNet_* 环境变量后运行，快照会读到过期的路径。
        preprocessed_root = nnunetv2.paths.nnUNet_preprocessed
        if not preprocessed_root:
            raise ValueError("nnUNet_preprocessed is required")
        validate_anatomy_split(
            Path(preprocessed_root) / ANATOMY_DATASET / "splits_final.json",
            dataset_json,
            fold,
        )
        super().__init__(plans, configuration, fold, dataset_json, device)
        # 必须在 initialize/configure_optimizers 构造 PolyLR 之前设置
        self.num_epochs = 100

    def initialize(self):
        from pathlib import Path

        validate_anatomy_dataset(
            self.dataset_json, self.plans_manager.dataset_name, self.configuration_name
        )
        validate_anatomy_split(
            Path(self.preprocessed_dataset_folder_base) / "splits_final.json",
            self.dataset_json,
            self.fold,
        )
        if self.label_manager.num_segmentation_heads != 3 or not self.label_manager.has_regions:
            raise ValueError("anatomy requires three native region heads")
        if self.configuration_manager.previous_stage_name is not None:
            raise ValueError("anatomy cannot use cascade inputs")
        super().initialize()
        if self.num_input_channels != 1:
            raise ValueError("anatomy requires one input channel")

    def do_split(self):
        from pathlib import Path

        tr, va = validate_anatomy_split(
            Path(self.preprocessed_dataset_folder_base) / "splits_final.json",
            self.dataset_json,
            self.fold,
        )
        if self.dataset_class is not None:
            ids = self.dataset_class.get_identifiers(self.preprocessed_dataset_folder)
            if set(ids) != set(tr) | set(va):
                raise ValueError("preprocessed anatomy case set differs from explicit split")
        return tr, va


__all__ = (
    "ANATOMY_DATASET",
    "nnUNetTrainerPICAI_AnatomyJoint_100ep_NoFFT",
    "nnUNetTrainerPICAI_DiceCE_NoFFT",
    "nnUNetTrainerPICAI_DiceCE_PositiveSampling_NoFFT",
    "nnUNetTrainerPICAI_FLCE_NoFFT",
    "nnUNetTrainerPICAI_FLCE_PositiveSampling_NoFFT",
)
