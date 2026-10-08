"""项目对 nnU-Net v2.6.2 的全部 Trainer 扩展（单文件）。

十五个 variant 共享同一套 nnU-Net 训练机制（optimizer=SGD+Nesterov、PolyLR、
增强主体、checkpoint/resume、validation、sliding-window inference），差异都通过覆盖
nnU-Net 的钩子实现，绝不重实现训练循环：

1. 损失：Focal+CE 分支使用 PI-CAI 官方风格 ``0.5*Focal(gamma=2) + 0.5*CE``；
   ``optimized_baseline`` 与 ``dicece_positive_sampling`` 则**不覆盖任何损失钩子**，
   直接用 nnU-Net 原生 Dice+CE；
2. 网络：baseline / optimized_baseline / positive_sampling 用原生 backbone；
   image_gate / anatomy_gate 及其 ``_positive_sampling`` 组合在原生 backbone 前加一个
   零初始化的 spatial modality gate；``feature_*_positive_sampling`` 三个条件在原生 backbone
   前加三个**参数不共享**的浅层序列 stem + 1×1×1 投影（可选 feature gate）——见 Research Plan
   §8.10 与 ``networks.FeatureFusionNNUNet``；
3. 采样：``positive_sampling``、两个 gate 组合、三个 ``feature_*_positive_sampling`` 与
   ``dicece_positive_sampling`` **只**把训练 loader 换成 ``PositiveCaseDataLoader``（每批固定一个
   阳性病灶 patch，验证 loader 保持原生）。新增四个短预算融合条件也复用此采样；
4. 短预算融合：旧十一个条件保持 1000 epochs；四个独立 ``*_100ep_NoFFT`` 条件使用
   Dataset606 五通道、100 epochs 与同步的原生 PolyLR 周期，分别为普通融合、分区 gate、
   同区参照残差修正和自适应参照强度。不覆盖旧目录，也不改变旧模型。

十五个 Trainer 类名不同，nnU-Net 据此隔离 output folder。旧十一个为：
    nnUNetTrainerPICAI_FLCE_NoFFT__nnUNetPlans__3d_fullres      (baseline，即已完成的 N0)
    nnUNetTrainerPICAI_DiceCE_NoFFT__nnUNetPlans__3d_fullres    (optimized_baseline，已中止)
    nnUNetTrainerPICAI_DiceCE_PositiveSampling_NoFFT__...       (dicece_positive_sampling，独立强参考基线)
    nnUNetTrainerPICAI_ImageGate__nnUNetPlans__3d_fullres       (image_gate)
    nnUNetTrainerPICAI_AnatomyGate__nnUNetPlans__3d_fullres     (anatomy_gate)
    nnUNetTrainerPICAI_FLCE_PositiveSampling_NoFFT__...         (positive_sampling)
    nnUNetTrainerPICAI_ImageGate_PositiveSampling_NoFFT__...    (image_gate_positive_sampling)
    nnUNetTrainerPICAI_AnatomyGate_PositiveSampling_NoFFT__...  (anatomy_gate_positive_sampling)
    nnUNetTrainerPICAI_FeatureNoGate_PositiveSampling_NoFFT__...      (feature_no_gate_positive_sampling)
    nnUNetTrainerPICAI_FeatureImageGate_PositiveSampling_NoFFT__...   (feature_image_gate_positive_sampling)
    nnUNetTrainerPICAI_FeatureAnatomyGate_PositiveSampling_NoFFT__... (feature_anatomy_gate_positive_sampling)

浅层特征融合三条件的比较边界（Research Plan §8.10、H6）：
``feature_no_gate`` ↔ ``feature_image_gate`` 只差 feature gate；``feature_image_gate`` ↔
``feature_anatomy_gate`` 只差 gate 条件（PZ/TZ）与数据集，但两者 ``gate.conv1.weight`` 相差
``gate_hidden_channels * 2`` 个参数（C_s=8 时为 16 个，见 ``feature_gate_parameter_delta``），
且浅层版本的零初始化**只**保证与 feature no-gate 初始等价，**不**保证与原生 nnU-Net 等价。

Focal+CE 分支（baseline / image_gate / anatomy_gate / positive_sampling）与 Dice+CE 分支
（optimized_baseline）是两个**独立的研究分支**：前者是既有的 PI-CAI 风格基线，后者是针对
「长期停留在背景预测」现象的单变量优化。两者除损失外一切训练机制相同，但**不能互相作为
归因门控收益的对照**。

``dicece_positive_sampling`` 是**独立于 A→B→C→D 主研究链的增强参考基线 / 损失消融**，不是新的
主研究阶段：它只回答「在 PositiveSampling 已稳定病灶暴露之后，nnU-Net 原生 Dice+CE 是否优于
PI-CAI Focal+CE（A）」。已中止的 ``optimized_baseline`` 使用原生采样且未完成训练，不能回答该
问题；本类相对 A 唯一改变损失，相对 ``optimized_baseline`` 唯一改变训练采样。

公平比较的边界（Research Plan §8.9 / §15.12）：核心门控比较必须固定损失、病例/patch 采样、
增强、optimizer、LR scheduler、split 与随机种子策略。

- 旧 ``baseline`` ↔ 旧 ``image_gate``（同为 FLCE + 原生采样）本身仍是**原生采样条件下**的受控
  RQ1 对照，但它们**不能与 ``positive_sampling`` 分支混合比较**（采样不同），且单次运行与
  baseline 早期长期全背景的优化异常会限制结论强度；
- 与 ``positive_sampling`` 分支的公平匹配比较：RQ1 用 ``positive_sampling`` ↔
  ``image_gate_positive_sampling``；RQ2 用 ``image_gate_positive_sampling`` ↔
  ``anatomy_gate_positive_sampling``（该比较的**预期主要差异**是门控条件与数据集
  Dataset605/606；在前三个 MRI 通道逐数组一致性审计完成前，不得宣称唯一差异是 PZ/TZ）。

NoFFT 修复：本环境 torch 与 fft-conv-pytorch 在增强 worker 中执行 GaussianBlurTransform 的
FFT *benchmark* 时会触发 native 内存损坏。修复保持 blur 的概率/sigma/通道不变，只关闭其
benchmark（强制走普通 convolution），彻底不执行不稳定的 FFT 试跑。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from torch import Tensor, nn

from zonal_reliability_fusion.nnunet.networks import (
    ANATOMY_INPUT_CHANNELS,
    FEATURE_STEM_CHANNELS,
    MRI_CHANNELS,
    PRIOR_CHANNELS_ANATOMY,
    FeatureFusionNNUNet,
    GatedNNUNet,
    SpatialModalityReliabilityGate,
    ZonalReferenceFusionNNUNet,
)
from zonal_reliability_fusion.nnunet.sampling import PositiveCaseSamplingMixin
from zonal_reliability_fusion.nnunet.transforms import (
    restrict_intensity_transforms_to_mri,
)


# One contract shared by preparation, training, prediction and independent evaluation.
ANATOMY_DATASET = "Dataset607_PICAI_Anatomy"
ANATOMY_LABELS = {"background": 0, "WG": [1, 3, 5, 7],
                  "PZ": [2, 3, 6, 7], "TZ": [4, 5, 6, 7]}
ANATOMY_CLASS_ORDER = [1, 2, 4]
ANATOMY_KNOWN_MISSING = "11050_1001070"
ANATOMY_COUNTS = (1499, 1276, 223)  # studies, not patients
ANATOMY_GEOMETRY_ATOL = 1e-4  # unchanged from the anatomy audit; rtol is always zero


def anatomy_read_array(path):
    # SimpleITK's NIfTI reader can replace stored NaN with zero. Inspect scaled raw
    # NIfTI values through nibabel before any integer conversion or membership test.
    import nibabel as nib
    array = np.asanyarray(nib.load(str(path)).dataobj)
    if array.ndim != 3 or not np.isfinite(array).all():
        raise ValueError(f"{path}: expected finite stored 3D NIfTI values")
    return array.transpose(2, 1, 0)  # XYZ -> native SimpleITK ZYX array order


def anatomy_validate_array(array, allowed, name):
    array = np.asarray(array)
    if array.ndim != 3 or not np.isfinite(array).all():
        raise ValueError(f"{name}: expected finite 3D labels")
    if not np.equal(array, np.rint(array)).all() or not np.isin(array, allowed).all():
        raise ValueError(f"{name}: non-integer or illegal labels; expected {allowed}")
    return array


def anatomy_encode(wg, yuan):
    wg = anatomy_validate_array(wg, [0, 1], "WG")
    yuan = anatomy_validate_array(yuan, [0, 1, 2], "Yuan")
    if wg.shape != yuan.shape:
        raise ValueError("WG/Yuan shape mismatch")
    return ((wg == 1).astype(np.uint8) + 2 * (yuan == 1).astype(np.uint8)
            + 4 * (yuan == 2).astype(np.uint8))


def anatomy_encode_heads(masks):
    masks = np.asarray(masks)
    if masks.ndim != 4 or masks.shape[0] != 3 or not np.isin(masks, [0, 1]).all():
        raise ValueError("expected three binary WG/PZ/TZ masks")
    return sum(masks[i].astype(np.uint8) * bit for i, bit in enumerate((1, 2, 4)))


def anatomy_geometry(image):
    if image.GetDimension() != 3:
        raise ValueError("expected 3D geometry")
    size, spacing = image.GetSize(), np.asarray(image.GetSpacing())
    origin, direction = np.asarray(image.GetOrigin()), np.asarray(image.GetDirection()).reshape(3, 3)
    if (any(n <= 0 for n in size) or not np.isfinite(spacing).all() or (spacing <= 0).any()
            or not np.isfinite(origin).all() or not np.isfinite(direction).all()
            or not np.allclose(direction @ direction.T, np.eye(3), rtol=0, atol=ANATOMY_GEOMETRY_ATOL)):
        raise ValueError("invalid physical geometry")
    return {"size": list(size), "spacing": spacing.tolist(), "origin": origin.tolist(),
            "direction": direction.flatten().tolist()}


def anatomy_same_grid(reference, other):
    a, b = anatomy_geometry(reference), anatomy_geometry(other)
    for key in a:
        same = a[key] == b[key] if key == "size" else np.allclose(
            a[key], b[key], rtol=0, atol=ANATOMY_GEOMETRY_ATOL)
        if not same:
            raise ValueError(f"physical grid mismatch: {key}")


def validate_anatomy_dataset(dataset_json, dataset_name=ANATOMY_DATASET, configuration="3d_fullres"):
    if dataset_name != ANATOMY_DATASET or configuration != "3d_fullres":
        raise ValueError("anatomy requires Dataset607_PICAI_Anatomy / 3d_fullres")
    if dataset_json.get("channel_names") != {"0000": "T2W"}:
        raise ValueError("anatomy requires single T2W channel 0000")
    if list(dataset_json.get("labels", {}).items()) != list(ANATOMY_LABELS.items()):
        raise ValueError("anatomy regions or region order mismatch")
    if dataset_json.get("regions_class_order") != ANATOMY_CLASS_ORDER:
        raise ValueError("anatomy regions_class_order mismatch")
    contract = dataset_json.get("anatomy_contract", {})
    if (contract.get("encoding") != "WG+2*PZ+4*TZ" or contract.get("zonal_source") != "zonal_yuan"
            or contract.get("wg_source") != "materialized_wg"
            or contract.get("explicit_exclusions") != [ANATOMY_KNOWN_MISSING]):
        raise ValueError("missing or invalid anatomy provenance contract")
    ids = contract.get("case_ids", [])
    if len(ids) != ANATOMY_COUNTS[0] or len(ids) != len(set(ids)) or dataset_json.get("numTraining") != len(ids):
        raise ValueError("anatomy study scope mismatch")
    if ANATOMY_KNOWN_MISSING in ids:
        raise ValueError("known missing WG cannot enter complete supervision")
    for cid in ids:
        if not isinstance(cid, str) or len(cid.split("_")) != 2 or not all(p.isdigit() for p in cid.split("_")):
            raise ValueError(f"invalid anatomy case ID: {cid}")
    return contract


def validate_anatomy_split(split_path, dataset_json, fold=0):
    import json
    from pathlib import Path
    contract = validate_anatomy_dataset(dataset_json)
    if str(fold) != "0":
        raise ValueError("anatomy frozen single split requires fold 0; no random fallback")
    path = Path(split_path)
    if not path.is_file():
        raise ValueError(f"explicit anatomy split missing: {path}; no random fallback")
    splits = json.loads(path.read_text())
    if not isinstance(splits, list) or len(splits) != 1:
        raise ValueError("expected one explicit frozen anatomy split")
    split = splits[0]
    tr, va = split.get("train", []), split.get("val", [])
    if (len(tr) != ANATOMY_COUNTS[1] or len(va) != ANATOMY_COUNTS[2]
            or len(tr) != len(set(tr)) or len(va) != len(set(va))
            or set(tr) & set(va) or set(tr) | set(va) != set(contract["case_ids"])):
        raise ValueError("anatomy split case scope/duplicates mismatch")
    if set(tr) != set(contract.get("train_cases", [])) or set(va) != set(contract.get("val_cases", [])):
        raise ValueError("anatomy split differs from frozen provenance")
    if {c.split("_")[0] for c in tr} & {c.split("_")[0] for c in va}:
        raise ValueError("anatomy train/validation patient overlap")
    return tr, va


def anatomy_validate_probabilities(probabilities, properties, reference):
    """Native restored (C,Z,Y,X) arrays plus trusted native pkl physical metadata."""
    probabilities = np.asarray(probabilities)
    geometry = anatomy_geometry(reference)
    shape = tuple(reversed(geometry["size"]))
    if probabilities.shape != (3, *shape):
        raise ValueError(f"anatomy probability channels/shape mismatch: {probabilities.shape} vs {(3, *shape)}")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError("anatomy probabilities must be finite within [0,1]")
    stuff = properties.get("sitk_stuff", {})
    for key in ("spacing", "origin", "direction"):
        value = np.asarray(stuff.get(key, []))
        if value.shape != np.asarray(geometry[key]).shape or not np.allclose(
                value, geometry[key], rtol=0, atol=ANATOMY_GEOMETRY_ATOL):
            raise ValueError(f"anatomy probability physical metadata mismatch: {key}")
    if not np.allclose(properties.get("spacing", []), list(reversed(geometry["spacing"])), rtol=0, atol=ANATOMY_GEOMETRY_ATOL):
        raise ValueError("anatomy probability array spacing mismatch")
    # shape_before_cropping is stored in preprocessing-transposed coordinates.
    before = properties.get("shape_before_cropping", [])
    forward = properties.get("anatomy_transpose_forward")
    if forward is None or sorted(forward) != [0, 1, 2] or tuple(before) != tuple(shape[i] for i in forward):
        raise ValueError("anatomy probability original shape metadata mismatch")
    return probabilities


class PiCAIFocalCrossEntropyLoss(nn.Module):
    """PI-CAI baseline 的等权 Focal + CE 二/多类体素损失。

    原始 trainer 将 ``FocalLoss(alpha=None, gamma=2, smooth=1e-5)`` 与 Cross-Entropy 以
    0.5/0.5 相加。``alpha=None`` 意味着没有额外的类别权重。输入为 nnU-Net 约定的 logits
    ``[B, C, ...]``，target 为 ``[B, 1, ...]`` 或 ``[B, ...]``。Dataset605/606 没有 ignore
    label，故遇到负标签或越界标签会显式失败而不是静默当作背景。
    """

    def __init__(
        self,
        *,
        focal_weight: float = 0.5,
        ce_weight: float = 0.5,
        gamma: float = 2.0,
        smooth: float = 1e-5,
    ) -> None:
        super().__init__()
        if focal_weight < 0 or ce_weight < 0 or focal_weight + ce_weight <= 0:
            raise ValueError("focal_weight/ce_weight 必须非负且至少一个为正")
        if gamma < 0:
            raise ValueError("gamma 必须非负")
        if not 0.0 <= smooth < 1.0:
            raise ValueError("smooth 必须在 [0, 1) 内")
        self.focal_weight = float(focal_weight)
        self.ce_weight = float(ce_weight)
        self.gamma = float(gamma)
        self.smooth = float(smooth)

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        if logits.ndim < 2:
            raise ValueError(f"logits 至少应为 [B,C]，收到 {tuple(logits.shape)}")
        if logits.shape[1] < 2:
            raise ValueError("PI-CAI Focal + CE 至少需要 background 与 lesion 两类")

        if target.ndim == logits.ndim:
            if target.shape[1] != 1:
                raise ValueError(
                    "target 的类别维必须为 singleton；"
                    f"收到 logits={tuple(logits.shape)}, target={tuple(target.shape)}"
                )
            target = target[:, 0]
        if target.shape != logits.shape[:1] + logits.shape[2:]:
            raise ValueError(
                "target 与 logits 的空间尺寸不匹配："
                f"logits={tuple(logits.shape)}, target={tuple(target.shape)}"
            )
        if torch.is_floating_point(target) and not torch.all(target == target.round()):
            raise ValueError("target 含非整数标签")
        target = target.long()
        if bool(torch.any(target < 0)) or bool(torch.any(target >= logits.shape[1])):
            raise ValueError(
                "target 含超出 [0, num_classes) 的标签；Dataset605/606 不支持 ignore label"
            )

        # 复刻 PI-CAI v1 FocalLoss：Softmax → one-hot clamp(smooth, 1-smooth)
        # → pt + smooth → -(1-pt)^gamma * log(pt)。alpha=None 即全 1。
        probabilities = torch.softmax(logits, dim=1)
        one_hot = F.one_hot(target, num_classes=logits.shape[1]).movedim(-1, 1)
        one_hot = one_hot.to(dtype=probabilities.dtype)
        if self.smooth:
            one_hot = one_hot.clamp(
                min=self.smooth / (logits.shape[1] - 1), max=1.0 - self.smooth
            )
        pt = (one_hot * probabilities).sum(dim=1) + self.smooth
        focal = -torch.pow(1.0 - pt, self.gamma) * torch.log(pt)
        focal = focal.mean()
        ce = F.cross_entropy(logits, target)
        return self.focal_weight * focal + self.ce_weight * ce


class PICAIFocalCrossEntropyLossMixin:
    """用 PI-CAI Focal+CE 替换 nnU-Net 默认损失；deep supervision 加权沿用官方规则。"""

    def _build_loss(self) -> nn.Module:
        if self.label_manager.has_regions:
            raise RuntimeError(
                "PI-CAI Focal+CE 仅支持互斥二/多类标签；Dataset605/606 应为 background=0, lesion=1。"
            )
        if self.label_manager.ignore_label is not None:
            raise RuntimeError(
                "PI-CAI Focal+CE 不支持 ignore label；dataset.json 不应声明 ignore。"
            )

        loss: nn.Module = PiCAIFocalCrossEntropyLoss(
            focal_weight=0.5, ce_weight=0.5, gamma=2.0, smooth=1e-5
        )
        if self.enable_deep_supervision:
            scales = self._get_deep_supervision_scales()
            weights = np.array(
                [1.0 / (2**i) for i in range(len(scales))], dtype=np.float64
            )
            # 与固定 nnU-Net v2.6.2 默认 trainer 一致：最低分辨率不监督；DDP 特殊分支沿用官方实现。
            weights[-1] = 1e-6 if self.is_ddp and not self._do_i_compile() else 0.0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
        return loss


class NoFFTAugmentationMixin:
    """保留 nnU-Net 默认增强语义，仅关闭 GaussianBlurTransform 的 FFT benchmark。"""

    @staticmethod
    def get_training_transforms(*args, **kwargs):
        transforms = nnUNetTrainer.get_training_transforms(*args, **kwargs)
        disabled = 0
        for transform in transforms.transforms:
            candidate = getattr(transform, "transform", transform)
            if isinstance(candidate, GaussianBlurTransform):
                candidate.benchmark = False
                candidate.benchmark_use_fft.clear()
                disabled += 1
        if disabled != 1:
            raise RuntimeError(
                "nnU-Net v2.6.2 augmentation pipeline 未找到唯一的 GaussianBlurTransform；"
                "拒绝静默启动，以免重新调用不稳定的 fft_conv benchmark。"
            )
        return transforms


class nnUNetTrainerPICAI_AnatomyJoint_100ep_NoFFT(NoFFTAugmentationMixin, nnUNetTrainer):
    """Shared native network, three region heads; 100 epochs is an engineering trial."""

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        from pathlib import Path
        from nnunetv2.paths import nnUNet_preprocessed
        validate_anatomy_dataset(dataset_json, plans.get("dataset_name"), configuration)
        if not nnUNet_preprocessed:
            raise ValueError("nnUNet_preprocessed is required")
        validate_anatomy_split(Path(nnUNet_preprocessed) / ANATOMY_DATASET / "splits_final.json", dataset_json, fold)
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100  # before initialize/configure_optimizers constructs PolyLR

    def initialize(self):
        from pathlib import Path
        validate_anatomy_dataset(self.dataset_json, self.plans_manager.dataset_name, self.configuration_name)
        validate_anatomy_split(Path(self.preprocessed_dataset_folder_base) / "splits_final.json", self.dataset_json, self.fold)
        if self.label_manager.num_segmentation_heads != 3 or not self.label_manager.has_regions:
            raise ValueError("anatomy requires three native region heads")
        if self.configuration_manager.previous_stage_name is not None:
            raise ValueError("anatomy cannot use cascade inputs")
        super().initialize()
        if self.num_input_channels != 1:
            raise ValueError("anatomy requires one input channel")

    def do_split(self):
        from pathlib import Path
        tr, va = validate_anatomy_split(Path(self.preprocessed_dataset_folder_base) / "splits_final.json", self.dataset_json, self.fold)
        if self.dataset_class is not None:
            ids = self.dataset_class.get_identifiers(self.preprocessed_dataset_folder)
            if set(ids) != set(tr) | set(va):
                raise ValueError("preprocessed anatomy case set differs from explicit split")
        return tr, va


class nnUNetTrainerPICAI_FLCE_NoFFT(
    NoFFTAugmentationMixin, PICAIFocalCrossEntropyLossMixin, nnUNetTrainer
):
    """baseline variant：原生 ``PlainConvUNet`` + 3 个 MRI 通道 + PI-CAI Focal+CE + NoFFT 修复。

    除损失与 blur benchmark 外，全部继承 nnU-Net v2.6.2 默认训练行为。类名与已完成的 N0
    运行一致，output folder 保持不变（可续训 / 只做 validation）。
    """


class nnUNetTrainerPICAI_DiceCE_NoFFT(NoFFTAugmentationMixin, nnUNetTrainer):
    """optimized_baseline variant：原生 nnU-Net Dice+CE 的**单变量**优化基线。

    背景：已完成的 N0（``nnUNetTrainerPICAI_FLCE_NoFFT``）在 1000 epoch 中有约 700 epoch
    停留在"全部预测为背景"的状态，直到学习率衰减后才开始学习前景。

    本轮**唯一**的改动是把损失换成 nnU-Net v2.6.2 原生 Dice+CE；其余一切——网络、数据集、
    split、patch size、batch size、增强、前景采样、deep supervision 权重、SGD、PolyLR、
    初始学习率、epoch 数、checkpoint、validation、inference——与 baseline 完全一致。

    实现上刻意保持"零自写损失"：

    - 基类列表里**没有** ``PICAIFocalCrossEntropyLossMixin``，也没有继承
      ``nnUNetTrainerPICAI_FLCE_NoFFT`` 或 ``_GatedTrainerBase``；
    - 因此本类**不覆盖** ``_build_loss``，MRO 直接解析到 ``nnUNetTrainer._build_loss``，
      使用的就是固定的 nnU-Net v2.6.2 原生实现：
      ``DC_and_CE_loss(batch_dice=plans.batch_dice, smooth=1e-5, do_bg=False,
      weight_ce=weight_dice=1, dice_class=MemoryEfficientSoftDiceLoss)``，
      deep supervision 时再由原生 ``DeepSupervisionWrapper`` 按 1/2^i 加权；
    - 也**不覆盖** ``build_network_architecture``，网络仍由原生
      ``get_network_from_plans`` 按 plans 构建（原生 ``PlainConvUNet``，3 个 MRI 通道，
      不含任何 gate）；
    - 只保留 ``NoFFTAugmentationMixin`` 的 blur benchmark 关闭，blur 概率与 sigma 不变。
    """


class nnUNetTrainerPICAI_DiceCE_PositiveSampling_NoFFT(
    PositiveCaseSamplingMixin,
    nnUNetTrainerPICAI_DiceCE_NoFFT,
):
    """``dicece_positive_sampling``：原生 nnU-Net Dice+CE + 阳性病例采样（独立强参考基线）。

    定位：**独立于 A→B→C→D 主研究链的增强参考基线 / 损失消融**，不是新的主研究阶段。
    它回答的问题是「在 PositiveSampling 已稳定病灶暴露之后，nnU-Net 原生 Dice+CE 是否优于
    PI-CAI Focal+CE（A = ``positive_sampling``）」。已中止的 ``optimized_baseline`` 使用
    原生采样且未完成训练，不能回答该问题。

    单变量边界（两个方向都只有一个变量）：

    - 相对 A（``nnUNetTrainerPICAI_FLCE_PositiveSampling_NoFFT``）：**只**把损失从
      ``0.5*Focal(gamma=2) + 0.5*CE`` 换成 nnU-Net v2.6.2 原生 Dice+CE；
    - 相对 ``nnUNetTrainerPICAI_DiceCE_NoFFT``：**只**把训练集病例/patch 采样换成
      ``PositiveCaseDataLoader``（每批固定一个阳性病灶 patch）。

    MRO 与钩子来源（``PositiveCaseSamplingMixin`` 在最前）：

    - ``_build_loss``：本类与 ``nnUNetTrainerPICAI_DiceCE_NoFFT`` 都不覆盖，解析到
      ``nnUNetTrainer._build_loss`` 的**原生实现**（``DC_and_CE_loss(batch_dice=
      plans.batch_dice, smooth=1e-5, do_bg=False, weight_ce=weight_dice=1,
      MemoryEfficientSoftDiceLoss)``，deep supervision 由原生 ``DeepSupervisionWrapper``
      按 1/2^i 加权）；``PICAIFocalCrossEntropyLossMixin`` **不在** MRO 中，项目没有自写
      Dice+CE，也不含 focal gamma / alpha / 类别权重；
    - ``get_dataloaders``：``PositiveCaseSamplingMixin``（训练 loader =
      ``PositiveCaseDataLoader``，``positive_cases_per_batch=1``；验证 loader 保持原生
      ``nnUNetDataLoader`` 与原生采样，无验证泄漏）；
    - ``get_training_transforms``：``NoFFTAugmentationMixin``（复用同一 NoFFT 修复）；
    - ``build_network_architecture``：不覆盖，解析到 ``nnUNetTrainer`` 的**原生实现**，由
      plans 构建原生 ``PlainConvUNet``（3 个 MRI 通道；不含 gate / PZ/TZ / 浅层 feature path）；
    - optimizer（SGD+Nesterov）、PolyLR、1000 epochs、deep supervision 权重、patch size、
      batch size、checkpoint/resume、validation、滑窗推理全部继承原生实现。

    ``initialize`` 只在原生初始化之后追加一行可审计日志（实际构建出的 loss / 网络 / 通道数），
    不改变任何训练行为。输出目录由类名自然形成为独立新目录，不从任何既有 checkpoint 续训、
    不复用任何既有产物。
    """

    def initialize(self):
        """原生 ``initialize()`` + 一行可审计的启动配置日志（不复制、不修改训练循环）。"""
        super().initialize()
        base_loss = (
            self.loss.loss
            if isinstance(self.loss, DeepSupervisionWrapper)
            else self.loss
        )
        self.print_to_log_file(
            "DiceCE + PositiveSampling audit:\n"
            f"trainer={type(self).__name__}\n"
            f"loss={type(self.loss).__name__} "
            f"(base={type(base_loss).__module__}.{type(base_loss).__name__})\n"
            f"network={type(self.network).__name__}\n"
            f"input_channels={self.num_input_channels}\n"
            f"output_channels={self.label_manager.num_segmentation_heads}"
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


class nnUNetTrainerPICAI_FLCE_PositiveSampling_NoFFT(
    PositiveCaseSamplingMixin,
    nnUNetTrainerPICAI_FLCE_NoFFT,
):
    """positive_sampling variant：FLCE baseline + 每批固定一个阳性病灶 patch。

    背景：nnU-Net 原生病例采样先均匀抽取病例、再决定槽位是否强制前景；对大量阴性
    病例的数据集（Dataset605 fold 0 阳性约 28%），batch size 2 时真正保证含病灶的
    patch 槽位仅约 14%，与已中止的 optimized_baseline（Dice+CE）全背景塌缩一致。

    在 ``nnUNetTrainerPICAI_FLCE_NoFFT``（baseline）之上**唯一**的改变是训练集
    病例/patch 采样（见 ``zonal_reliability_fusion.nnunet.sampling``）：

    - 训练 loader 换成 ``PositiveCaseDataLoader``：batch 最后
      ``positive_cases_per_batch=1`` 个槽位固定来自当前 fold 的阳性训练病例并强制
      病灶中心裁剪（50% patch 槽位保证含病灶、100% batch 至少一个病灶中心 patch）；
      其余槽位保持均匀病例采样，阴性病例监督不变；
    - 验证 loader 保持原生 ``nnUNetDataLoader`` 与原生采样，无验证泄漏。

    MRO 保证其余一切与 baseline 完全一致：

    - loss 仍解析到 ``PICAIFocalCrossEntropyLossMixin._build_loss``
      （``0.5*Focal(gamma=2) + 0.5*CE``，deep supervision 权重不变）；
    - 增强仍经 ``NoFFTAugmentationMixin.get_training_transforms``（blur benchmark
      关闭，概率/sigma 不变）；
    - **不覆盖** ``build_network_architecture``：网络仍由 plans 构建为原生
      ``PlainConvUNet``（3 个 MRI 通道，不含 image gate / anatomy gate / PZ/TZ）；
    - optimizer（SGD+Nesterov）、PolyLR、1000 epochs、batch_dice、patch size、
      batch size、checkpoint/resume、validation、inference 全部继承原生实现。

    本类**不**继承 ``nnUNetTrainerPICAI_DiceCE_NoFFT`` / ``nnUNetTrainerPICAI_ImageGate``
    / ``nnUNetTrainerPICAI_AnatomyGate``；输出目录由类名自然形成为独立新目录，
    不复用 baseline / DiceCE / gate 的任何已有产物，也不从其 checkpoint 续训。
    """


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


#: 项目 Trainer 名 -> 类。训练入口与预测入口都用它做**进程内直接映射**，
#: 避免依赖 nnU-Net 的 ``recursive_find_python_class``（该函数只在 nnunetv2 包目录内递归扫描，
#: 找不到项目自定义类，且可能扫到环境中其他 nnunetv2 分支）。
PROJECT_TRAINERS: dict[str, type[nnUNetTrainer]] = {
    nnUNetTrainerPICAI_AnatomyJoint_100ep_NoFFT.__name__: nnUNetTrainerPICAI_AnatomyJoint_100ep_NoFFT,
    nnUNetTrainerPICAI_FLCE_NoFFT.__name__: nnUNetTrainerPICAI_FLCE_NoFFT,
    nnUNetTrainerPICAI_DiceCE_NoFFT.__name__: nnUNetTrainerPICAI_DiceCE_NoFFT,
    nnUNetTrainerPICAI_DiceCE_PositiveSampling_NoFFT.__name__: (
        nnUNetTrainerPICAI_DiceCE_PositiveSampling_NoFFT
    ),
    nnUNetTrainerPICAI_ImageGate.__name__: nnUNetTrainerPICAI_ImageGate,
    nnUNetTrainerPICAI_AnatomyGate.__name__: nnUNetTrainerPICAI_AnatomyGate,
    nnUNetTrainerPICAI_FLCE_PositiveSampling_NoFFT.__name__: (
        nnUNetTrainerPICAI_FLCE_PositiveSampling_NoFFT
    ),
    nnUNetTrainerPICAI_ImageGate_PositiveSampling_NoFFT.__name__: (
        nnUNetTrainerPICAI_ImageGate_PositiveSampling_NoFFT
    ),
    nnUNetTrainerPICAI_AnatomyGate_PositiveSampling_NoFFT.__name__: (
        nnUNetTrainerPICAI_AnatomyGate_PositiveSampling_NoFFT
    ),
    nnUNetTrainerPICAI_FeatureNoGate_PositiveSampling_NoFFT.__name__: (
        nnUNetTrainerPICAI_FeatureNoGate_PositiveSampling_NoFFT
    ),
    nnUNetTrainerPICAI_FeatureImageGate_PositiveSampling_NoFFT.__name__: (
        nnUNetTrainerPICAI_FeatureImageGate_PositiveSampling_NoFFT
    ),
    nnUNetTrainerPICAI_FeatureAnatomyGate_PositiveSampling_NoFFT.__name__: (
        nnUNetTrainerPICAI_FeatureAnatomyGate_PositiveSampling_NoFFT
    ),
    nnUNetTrainerPICAI_FeatureNoGate_PositiveSampling_100ep_NoFFT.__name__: (
        nnUNetTrainerPICAI_FeatureNoGate_PositiveSampling_100ep_NoFFT
    ),
    nnUNetTrainerPICAI_FeatureAnatomyGate_PositiveSampling_100ep_NoFFT.__name__: (
        nnUNetTrainerPICAI_FeatureAnatomyGate_PositiveSampling_100ep_NoFFT
    ),
    nnUNetTrainerPICAI_ZonalReference_PositiveSampling_100ep_NoFFT.__name__: (
        nnUNetTrainerPICAI_ZonalReference_PositiveSampling_100ep_NoFFT
    ),
    nnUNetTrainerPICAI_ZonalReferenceAdaptive_PositiveSampling_100ep_NoFFT.__name__: (
        nnUNetTrainerPICAI_ZonalReferenceAdaptive_PositiveSampling_100ep_NoFFT
    ),
}


def resolve_trainer_class(trainer_name: str) -> type[nnUNetTrainer] | None:
    """按类名解析项目 Trainer；不是项目 Trainer 时返回 ``None``（由调用方决定是否回退）。"""
    return PROJECT_TRAINERS.get(trainer_name)
