"""阳性病例感知的训练 patch 采样（项目侧受控扩展，不修改 third_party）。

研究假设
--------
nnU-Net v2.6.2 的原生病例采样先从**全部**训练病例中均匀抽取，再按
``oversample_foreground_percent`` 决定某个 batch 槽位是否强制前景裁剪；若被选病例是
阴性病例，force_fg 也无法得到病灶中心 patch。Dataset605 fold 0 阳性病例仅约 28%、
batch size 2 时，真正保证含病灶的 patch 槽位约为 ``0.5 × 0.28 ≈ 14%``。已中止的
optimized_baseline（Dice+CE）出现严重全背景塌缩；另一个 PI-CAI 项目（338/1200 阳性、
同样 batch size 2、稀疏小病灶）的证据表明「Focal+CE + 每批固定一个阳性病灶 patch +
正常长周期 PolyLR」能较早脱离全背景。因此本模块只改变**训练集病例/patch 采样**，
其余 nnU-Net 机制（loss、网络、optimizer、调度、checkpoint、validation、推理）不变。

设计（最小侵入）
----------------
- ``PositiveCaseDataLoader`` 继承固定版本的原生 ``nnUNetDataLoader``，只覆盖两件事：
  ``get_indices``（把 batch 最后 ``positive_cases_per_batch`` 个位置替换为从阳性训练
  病例集合中有放回均匀抽取的病例）与 ``_oversample_last_XX_percent``（这些槽位始终
  force foreground）。bbox 计算、padding、数据读取、patch 裁剪全部沿用父类原生实现
  （继续使用预处理 ``class_locations``，不读取原始 NIfTI，不自行计算病灶坐标）。
- ``PositiveCaseSamplingMixin`` 只覆盖 ``get_dataloaders``：训练 loader 换成
  ``PositiveCaseDataLoader``，验证 loader 保持原生 ``nnUNetDataLoader`` 与原生采样
  行为；增强继续经 ``self.get_training_transforms`` 解析（NoFFT 等增强 mixin 仍生效）；
  SingleThreaded/NonDetMultiThreadedAugmenter 的接线与固定 v2.6.2 一致。
- 阳性病例只从**当前 fold 的训练 dataset identifiers** 识别：逐例读取预处理 ``.pkl``
  的 ``class_locations``，结合 ``label_manager.foreground_labels`` 判断；缺失/不可读/
  格式不合法/找不到阳性病例均 fail-closed，绝不把 metadata 错误静默当成阴性，也绝不
  把 validation 病例加入训练阳性集合。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import (
    NonDetMultiThreadedAugmenter,
)
from batchgenerators.dataloading.single_threaded_augmenter import (
    SingleThreadedAugmenter,
)
from batchgenerators.utilities.file_and_folder_operations import (
    isfile,
    join,
    load_pickle,
)
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from tqdm import tqdm


def _load_case_properties(source_folder: str, case_identifier: str) -> dict:
    """读取单个病例的预处理 ``.pkl`` properties；缺失/不可读/非 dict 均 fail-closed。"""
    pkl_file = join(source_folder, f"{case_identifier}.pkl")
    if not isfile(pkl_file):
        raise RuntimeError(
            f"缺少预处理 properties 文件：{pkl_file}。"
            "阳性病例识别 fail-closed：不会把缺失 metadata 的病例静默当成阴性。"
        )
    try:
        properties = load_pickle(pkl_file)
    except Exception as exc:
        raise RuntimeError(f"无法读取预处理 properties 文件 {pkl_file}: {exc}") from exc
    if not isinstance(properties, dict):
        raise TypeError(
            f"预处理 properties 文件 {pkl_file} 的内容不是 dict"
            f"（实际类型 {type(properties).__name__}），格式不合法，拒绝继续。"
        )
    return properties


def identify_positive_cases(
    dataset, foreground_labels: Sequence[int], *, progress: bool = True
) -> tuple[str, ...]:
    """从**当前 fold 训练 dataset** 的预处理 ``class_locations`` 识别阳性病例。

    - 只遍历 ``dataset.identifiers``（训练 split），不从完整病例列表注入、不读取
      validation 病例、不读临床标签、不按文件名猜测；
    - 逐例要求 ``.pkl`` 存在且可读、``class_locations`` 存在且为 dict、每个前景标签
      都有对应条目且条目可计数；任一不满足即抛错（fail-closed）；
    - 任一前景标签的 ``class_locations`` 条目非空即判为阳性；
    - 返回去重排序后的阳性 identifier 元组；一个阳性都没有时抛 RuntimeError；
    - 逐例扫描是启动期最耗时的循环，默认显示 tqdm 进度（``progress=False`` 可关闭）。
    """
    try:
        foreground_labels = tuple(int(label) for label in foreground_labels)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "阳性病例采样仅支持互斥整型前景标签（region 类标签不受支持），"
            f"收到 {foreground_labels!r}。"
        ) from exc
    if not foreground_labels:
        raise RuntimeError(
            "label_manager.foreground_labels 为空：数据集没有前景标签"
            "（Dataset605 应为 background=0 / lesion=1），无法定义阳性病例采样。"
        )
    positive_cases: list[str] = []
    for case_identifier in tqdm(
        dataset.identifiers,
        desc="positive-case scan",
        unit="case",
        leave=False,
        disable=not progress,
    ):
        properties = _load_case_properties(dataset.source_folder, case_identifier)
        class_locations = properties.get("class_locations")
        if class_locations is None:
            raise RuntimeError(
                f"病例 {case_identifier} 的预处理 properties 缺少 class_locations。"
                "阳性病例识别 fail-closed：不会把 metadata 错误静默当成阴性。"
            )
        if not isinstance(class_locations, dict):
            raise TypeError(
                f"病例 {case_identifier} 的 class_locations 不是 dict"
                f"（实际类型 {type(class_locations).__name__}），格式不合法，拒绝继续。"
            )
        for label in foreground_labels:
            if label not in class_locations:
                raise RuntimeError(
                    f"病例 {case_identifier} 的 class_locations 缺少前景标签 {label} 的条目；"
                    "预处理与 dataset.json 标签不一致，拒绝继续。"
                )
            locations = class_locations[label]
            if not hasattr(locations, "__len__"):
                raise RuntimeError(
                    f"病例 {case_identifier} 的 class_locations[{label}] 不是可计数容器"
                    f"（实际类型 {type(locations).__name__}），格式不合法，拒绝继续。"
                )
            if len(locations) > 0:
                positive_cases.append(case_identifier)
                break
    if not positive_cases:
        raise RuntimeError(
            "当前 fold 的训练集中没有找到任何阳性病例（class_locations 全为空）。"
            "请先核对预处理与标签，不要在此时启动 positive_sampling 实验。"
        )
    return tuple(sorted(set(positive_cases)))


class PositiveCaseDataLoader(nnUNetDataLoader):
    """每个 batch 的最后 ``positive_cases_per_batch`` 个槽位固定来自阳性训练病例。

    - ``get_indices``：先调用父类原生 case selection（均匀、有放回），保留前面的
      普通槽位（继续从完整训练集均匀抽样，阴性监督不变），再把最后
      ``positive_cases_per_batch`` 个位置替换为从阳性病例集合中均匀有放回抽取的病例；
    - ``_oversample_last_XX_percent``：最后 ``positive_cases_per_batch`` 个槽位始终
      force foreground；病灶中心 bbox 由父类 ``get_bbox`` 用预处理 ``class_locations``
      计算，本类不自行实现 ``get_bbox``；
    - 其余槽位保持父类的非强制行为。

    构造时 fail-closed：阳性 ID 去重排序后必须非空且全部属于本 loader 的训练
    dataset identifiers；``1 <= positive_cases_per_batch <= batch_size``；必须使用
    确定性槽位决策（``probabilistic_oversampling=False``）。
    """

    def __init__(
        self,
        *args,
        positive_case_identifiers: Sequence[str],
        positive_cases_per_batch: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        positive_case_identifiers = tuple(sorted(set(positive_case_identifiers)))
        if not positive_case_identifiers:
            raise ValueError(
                "positive_case_identifiers 不能为空：阳性病例采样至少需要一个阳性训练病例。"
            )
        unknown_identifiers = set(positive_case_identifiers) - set(self.indices)
        if unknown_identifiers:
            preview = sorted(unknown_identifiers)[:3]
            raise ValueError(
                "positive_case_identifiers 必须全部属于该 loader 的训练 dataset "
                f"identifiers；发现 {len(unknown_identifiers)} 个未知 ID"
                f"（例如 {preview}）。"
            )
        if not 1 <= positive_cases_per_batch <= self.batch_size:
            raise ValueError(
                "positive_cases_per_batch 必须满足 "
                f"1 <= positive_cases_per_batch <= batch_size；"
                f"收到 {positive_cases_per_batch}，batch_size={self.batch_size}。"
            )
        # 父类在 __init__ 里按 probabilistic_oversampling 选择槽位决策函数；
        # 概率式决策会绕过"阳性槽位始终 force_fg"的保证，必须拒绝。
        if self.get_do_oversample == self._probabilistic_oversampling:
            raise ValueError(
                "PositiveCaseDataLoader 要求 probabilistic_oversampling=False："
                "概率式 oversampling 无法保证阳性槽位始终执行病灶中心裁剪。"
            )
        self.positive_case_identifiers = positive_case_identifiers
        self.positive_cases_per_batch = int(positive_cases_per_batch)

    def get_indices(self):
        """父类原生选例后，把 batch 末尾固定数量的槽位替换为阳性训练病例。"""
        selected_keys = list(super().get_indices())
        if len(selected_keys) != self.batch_size:
            raise RuntimeError(
                f"父类 get_indices 返回 {len(selected_keys)} 个病例"
                f"（预期 {self.batch_size}）；槽位替换假设被破坏，拒绝继续。"
            )
        first_positive_position = self.batch_size - self.positive_cases_per_batch
        selected_keys[first_positive_position:] = np.random.choice(
            self.positive_case_identifiers,
            size=self.positive_cases_per_batch,
            replace=True,
        ).tolist()
        return selected_keys

    def _oversample_last_XX_percent(self, sample_idx: int) -> bool:
        """最后 ``positive_cases_per_batch`` 个 batch 槽位始终强制前景。"""
        return sample_idx >= self.batch_size - self.positive_cases_per_batch


class PositiveCaseSamplingMixin:
    """训练 loader 换成 ``PositiveCaseDataLoader``，验证 loader 保持原生（无泄漏）。

    只覆盖 ``get_dataloaders``；训练循环、optimizer、LR scheduler、checkpoint、
    validation、滑窗推理全部继承固定 nnU-Net v2.6.2。训练 transforms 继续由
    ``self.get_training_transforms`` 经 MRO 解析（NoFFT 等增强 mixin 依然生效）；
    patch size、initial patch size、deep-supervision scales、rotation、dummy 2D、
    mirror axes 等继续使用 nnU-Net 配置。augmenter 的类型、进程数、缓存与 pin memory
    与原生 ``nnUNetTrainer.get_dataloaders``（v2.6.2）逐行一致。
    """

    #: 每个 batch 固定的阳性病灶 patch 槽位数（Dataset605 batch size 2 时固定为 1）
    positive_cases_per_batch: int = 1

    def _positive_case_identifiers(self, dataset_train) -> tuple[str, ...]:
        """只从当前 fold 的训练 dataset 识别阳性病例（见 identify_positive_cases）。"""
        return identify_positive_cases(
            dataset_train, self.label_manager.foreground_labels
        )

    def get_dataloaders(self):
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

        # 与原生 v2.6.2 相同：patch size 决定 2D/3D loader、dummy 2D 与 initial patch size
        patch_size = self.configuration_manager.patch_size

        deep_supervision_scales = self._get_deep_supervision_scales()

        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        # training pipeline：self.get_training_transforms 经 MRO 解析（NoFFT 修复等仍生效）
        tr_transforms = self.get_training_transforms(
            patch_size,
            rotation_for_DA,
            deep_supervision_scales,
            mirror_axes,
            do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=(
                self.label_manager.foreground_regions
                if self.label_manager.has_regions
                else None
            ),
            ignore_label=self.label_manager.ignore_label,
        )

        # validation pipeline：原生
        val_transforms = self.get_validation_transforms(
            deep_supervision_scales,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=(
                self.label_manager.foreground_regions
                if self.label_manager.has_regions
                else None
            ),
            ignore_label=self.label_manager.ignore_label,
        )

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        # 阳性病例只从当前 fold 的训练 dataset 识别（fail-closed，见 identify_positive_cases）
        positive_case_identifiers = self._positive_case_identifiers(dataset_tr)

        n_training_cases = len(dataset_tr.identifiers)
        n_positive_cases = len(positive_case_identifiers)
        guaranteed_fraction = self.positive_cases_per_batch / self.batch_size
        self.print_to_log_file(
            "Positive-case sampling enabled:\n"
            f"training_cases={n_training_cases}\n"
            f"positive_cases={n_positive_cases}\n"
            f"negative_cases={n_training_cases - n_positive_cases}\n"
            f"positive_cases_per_batch={self.positive_cases_per_batch}\n"
            f"batch_size={self.batch_size}\n"
            f"guaranteed_positive_patch_fraction={guaranteed_fraction:g}"
        )

        # 训练 loader：唯一差异是病例/patch 采样；transforms、patch size、initial patch
        # size、sampling_probabilities=None 均与原生调用一致，probabilistic_oversampling
        # 固定为 False（确定性槽位决策才能保证阳性槽位始终病灶中心裁剪）。
        data_loader_train = PositiveCaseDataLoader(
            dataset_tr,
            self.batch_size,
            initial_patch_size,
            self.configuration_manager.patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None,
            pad_sides=None,
            transforms=tr_transforms,
            probabilistic_oversampling=False,
            positive_case_identifiers=positive_case_identifiers,
            positive_cases_per_batch=self.positive_cases_per_batch,
        )
        # 验证 loader：保持原生 nnUNetDataLoader 与原生采样行为，不按验证标签做任何
        # 阳性重采样；actual full-volume validation 完全交给 nnU-Net。
        data_loader_val = nnUNetDataLoader(
            dataset_val,
            self.batch_size,
            self.configuration_manager.patch_size,
            self.configuration_manager.patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None,
            pad_sides=None,
            transforms=val_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling,
        )

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(data_loader_train, None)
            mt_gen_val = SingleThreadedAugmenter(data_loader_val, None)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(
                data_loader=data_loader_train,
                transform=None,
                num_processes=allowed_num_processes,
                num_cached=max(6, allowed_num_processes // 2),
                seeds=None,
                pin_memory=self.device.type == "cuda",
                wait_time=0.002,
            )
            mt_gen_val = NonDetMultiThreadedAugmenter(
                data_loader=data_loader_val,
                transform=None,
                num_processes=max(1, allowed_num_processes // 2),
                num_cached=max(3, allowed_num_processes // 4),
                seeds=None,
                pin_memory=self.device.type == "cuda",
                wait_time=0.002,
            )
        # 与原生 v2.6.2 一致：立即各取一个 batch 预热
        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        return mt_gen_train, mt_gen_val
