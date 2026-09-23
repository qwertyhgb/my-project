"""阳性病例感知采样的纯合成 CPU 测试。

全部使用 tmp_path + 合成数组 / 合成 pkl：不读取真实医学数据、不访问
data/workdir/outputs 的真实病例、不启动训练、不写项目数据目录、不使用 GPU。

核心采样行为被**实际执行**验证（构造 loader、调用 get_indices /
generate_train_batch / get_dataloaders），而不是只比对类名或字符串。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from batchgenerators.utilities.file_and_folder_operations import (
    load_pickle,
    write_pickle,
)
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

from zonal_reliability_fusion.nnunet.networks import GatedNNUNet
from zonal_reliability_fusion.nnunet.sampling import (
    PositiveCaseDataLoader,
    PositiveCaseSamplingMixin,
    identify_positive_cases,
)
from zonal_reliability_fusion.nnunet.trainers import (
    NoFFTAugmentationMixin,
    PiCAIFocalCrossEntropyLoss,
    PICAIFocalCrossEntropyLossMixin,
    _GatedTrainerBase,
    nnUNetTrainerPICAI_AnatomyGate,
    nnUNetTrainerPICAI_AnatomyGate_PositiveSampling_NoFFT,
    nnUNetTrainerPICAI_DiceCE_NoFFT,
    nnUNetTrainerPICAI_FeatureAnatomyGate_PositiveSampling_NoFFT,
    nnUNetTrainerPICAI_FeatureImageGate_PositiveSampling_NoFFT,
    nnUNetTrainerPICAI_FeatureNoGate_PositiveSampling_NoFFT,
    nnUNetTrainerPICAI_FLCE_NoFFT,
    nnUNetTrainerPICAI_FLCE_PositiveSampling_NoFFT,
    nnUNetTrainerPICAI_ImageGate,
    nnUNetTrainerPICAI_ImageGate_PositiveSampling_NoFFT,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: 合成体数据：1 通道 20×24×24
CASE_SHAPE = (1, 20, 24, 24)
#: 合成病灶体素（z, y, x）；对应 class_locations 行 [0, z, y, x]
LESION_VOXEL = (10, 12, 12)


def _label_manager():
    """合成 LabelManager：覆盖 nnUNetDataLoader 与 mixin 实际访问的全部属性。"""
    return SimpleNamespace(
        all_labels=[0, 1],
        foreground_labels=[1],
        foreground_regions=[1],
        has_regions=False,
        has_ignore_label=False,
        ignore_label=None,
    )


class SyntheticPreprocessedDataset:
    """合成 nnU-Net dataset：identifiers + source_folder + load_case。

    - 阳性病例：seg 含单个病灶体素，pkl 的 ``class_locations`` 中标签 1 的条目非空
      （键使用 ``np.int64``，与真实 Dataset605 预处理 pkl 一致）；
    - 阴性病例：seg 全背景，``class_locations[1]`` 为空 list（同为真实结构）；
    - ``load_case`` 的 properties 直接从同一 pkl 读回，与识别逻辑同源。
    """

    def __init__(self, folder: Path, identifiers, positive_ids) -> None:
        self.source_folder = str(folder)
        self.identifiers = sorted(identifiers)
        self._positive_ids = set(positive_ids)
        folder.mkdir(parents=True, exist_ok=True)
        for cid in self.identifiers:
            if cid in self._positive_ids:
                class_locations = {np.int64(1): np.array([[0, *LESION_VOXEL]])}
            else:
                class_locations = {np.int64(1): []}
            write_pickle(
                {"class_locations": class_locations},
                str(folder / f"{cid}.pkl"),
            )

    def load_case(self, identifier):
        data = np.zeros(CASE_SHAPE, dtype=np.float32)
        seg = np.zeros(CASE_SHAPE, dtype=np.int16)
        if identifier in self._positive_ids:
            seg[0, LESION_VOXEL[0], LESION_VOXEL[1], LESION_VOXEL[2]] = 1
        properties = load_pickle(str(Path(self.source_folder) / f"{identifier}.pkl"))
        return data, seg, None, properties


@pytest.fixture()
def synthetic_split(tmp_path):
    """8 个训练病例（3 阳性）+ 2 个验证病例（其中 1 个阳性，绝不能进入训练阳性集合）。"""
    tr_ids = [f"case_{i:03d}" for i in range(8)]
    tr_positive = {"case_001", "case_005", "case_007"}
    val_ids = ["val_100", "val_101"]
    val_positive = {"val_101"}
    dataset_tr = SyntheticPreprocessedDataset(
        tmp_path / "nnUNetPlans_3d_fullres", tr_ids, tr_positive
    )
    dataset_val = SyntheticPreprocessedDataset(
        tmp_path / "nnUNetPlans_3d_fullres", val_ids, val_positive
    )
    return SimpleNamespace(
        dataset_tr=dataset_tr,
        dataset_val=dataset_val,
        tr_ids=tr_ids,
        tr_positive=tr_positive,
        val_ids=val_ids,
        val_positive=val_positive,
    )


def _make_loader(
    dataset,
    *,
    positive_ids,
    batch_size=2,
    positive_cases_per_batch=1,
    probabilistic_oversampling=False,
    patch_size=(8, 12, 12),
):
    return PositiveCaseDataLoader(
        dataset,
        batch_size,
        patch_size,
        patch_size,
        _label_manager(),
        oversample_foreground_percent=0.33,
        sampling_probabilities=None,
        pad_sides=None,
        probabilistic_oversampling=probabilistic_oversampling,
        transforms=None,
        positive_case_identifiers=positive_ids,
        positive_cases_per_batch=positive_cases_per_batch,
    )


# --------------------------------------------------------------- 阳性病例识别
def test_identify_positive_cases_from_synthetic_class_locations(synthetic_split):
    """通过合成 class_locations 正确识别阳性/阴性（含 np.int64 键，同真实 pkl）。"""
    identified = identify_positive_cases(
        synthetic_split.dataset_tr, _label_manager().foreground_labels
    )
    assert identified == tuple(sorted(synthetic_split.tr_positive))
    # 阴性病例不在结果里
    negatives = set(synthetic_split.tr_ids) - synthetic_split.tr_positive
    assert not (set(identified) & negatives)


def test_identify_positive_cases_only_uses_training_identifiers(synthetic_split):
    """阳性集合只能来自训练 identifiers：阳性 validation 病例绝不能被纳入。"""
    identified = identify_positive_cases(
        synthetic_split.dataset_tr, _label_manager().foreground_labels
    )
    assert not (set(identified) & synthetic_split.val_positive)
    assert not (set(identified) & set(synthetic_split.val_ids))


def test_identify_positive_cases_missing_pkl_fail_closed(tmp_path):
    ids = ["case_000", "case_001"]
    dataset = SyntheticPreprocessedDataset(tmp_path, ids, {"case_001"})
    (tmp_path / "case_000.pkl").unlink()
    with pytest.raises(RuntimeError, match="case_000"):
        identify_positive_cases(dataset, [1])


def test_identify_positive_cases_unreadable_pkl_fail_closed(tmp_path):
    ids = ["case_000", "case_001"]
    dataset = SyntheticPreprocessedDataset(tmp_path, ids, {"case_001"})
    (tmp_path / "case_000.pkl").write_bytes(b"not a pickle")
    with pytest.raises(RuntimeError, match="无法读取"):
        identify_positive_cases(dataset, [1])


def test_identify_positive_cases_missing_class_locations_fail_closed(tmp_path):
    ids = ["case_000"]
    write_pickle({"shape": [20, 24, 24]}, str(tmp_path / "case_000.pkl"))
    dataset = SimpleNamespace(source_folder=str(tmp_path), identifiers=ids)
    with pytest.raises(RuntimeError, match="缺少 class_locations"):
        identify_positive_cases(dataset, [1])


def test_identify_positive_cases_invalid_class_locations_fail_closed(tmp_path):
    ids = ["case_000"]
    write_pickle({"class_locations": [1, 2, 3]}, str(tmp_path / "case_000.pkl"))
    dataset = SimpleNamespace(source_folder=str(tmp_path), identifiers=ids)
    with pytest.raises(TypeError, match="不是 dict"):
        identify_positive_cases(dataset, [1])


def test_identify_positive_cases_non_dict_properties_fail_closed(tmp_path):
    ids = ["case_000"]
    write_pickle([1, 2, 3], str(tmp_path / "case_000.pkl"))
    dataset = SimpleNamespace(source_folder=str(tmp_path), identifiers=ids)
    with pytest.raises(TypeError, match="不是 dict"):
        identify_positive_cases(dataset, [1])


def test_identify_positive_cases_missing_label_key_fail_closed(tmp_path):
    ids = ["case_000"]
    write_pickle({"class_locations": {}}, str(tmp_path / "case_000.pkl"))
    dataset = SimpleNamespace(source_folder=str(tmp_path), identifiers=ids)
    with pytest.raises(RuntimeError, match="缺少前景标签 1"):
        identify_positive_cases(dataset, [1])


def test_identify_positive_cases_no_foreground_labels_fail_closed(tmp_path):
    dataset = SimpleNamespace(source_folder=str(tmp_path), identifiers=["case_000"])
    with pytest.raises(RuntimeError, match="没有前景标签"):
        identify_positive_cases(dataset, [])


def test_identify_positive_cases_no_positive_cases_fail_closed(tmp_path):
    ids = ["case_000", "case_001"]
    dataset = SyntheticPreprocessedDataset(tmp_path, ids, set())
    with pytest.raises(RuntimeError, match="没有找到任何阳性病例"):
        identify_positive_cases(dataset, [1])


# ------------------------------------------------------------- loader 构造校验
def test_positive_ids_are_deduplicated_and_sorted(synthetic_split):
    loader = _make_loader(
        synthetic_split.dataset_tr,
        positive_ids=["case_005", "case_001", "case_005"],
    )
    assert loader.positive_case_identifiers == ("case_001", "case_005")


def test_empty_positive_ids_fail_closed(synthetic_split):
    with pytest.raises(ValueError, match="不能为空"):
        _make_loader(synthetic_split.dataset_tr, positive_ids=[])


def test_unknown_positive_ids_fail_closed(synthetic_split):
    with pytest.raises(ValueError, match="未知 ID"):
        _make_loader(
            synthetic_split.dataset_tr,
            positive_ids=["case_001", "val_101"],
        )


def test_zero_positive_cases_per_batch_rejected(synthetic_split):
    with pytest.raises(ValueError, match="1 <= positive_cases_per_batch"):
        _make_loader(
            synthetic_split.dataset_tr,
            positive_ids=sorted(synthetic_split.tr_positive),
            positive_cases_per_batch=0,
        )


def test_too_many_positive_cases_per_batch_rejected(synthetic_split):
    with pytest.raises(ValueError, match="1 <= positive_cases_per_batch"):
        _make_loader(
            synthetic_split.dataset_tr,
            positive_ids=sorted(synthetic_split.tr_positive),
            positive_cases_per_batch=3,
        )


def test_probabilistic_oversampling_rejected(synthetic_split):
    """概率式槽位决策会绕过"阳性槽位始终 force_fg"的保证，必须拒绝。"""
    with pytest.raises(ValueError, match="probabilistic_oversampling=False"):
        _make_loader(
            synthetic_split.dataset_tr,
            positive_ids=sorted(synthetic_split.tr_positive),
            probabilistic_oversampling=True,
        )


# --------------------------------------------------------------- 采样槽位行为
def test_last_slot_always_from_positive_set(synthetic_split):
    np.random.seed(20260922)
    loader = _make_loader(
        synthetic_split.dataset_tr, positive_ids=sorted(synthetic_split.tr_positive)
    )
    for _ in range(200):
        keys = [str(k) for k in loader.get_indices()]
        assert keys[-1] in synthetic_split.tr_positive


def test_first_slot_covers_full_training_set_including_negatives(synthetic_split):
    np.random.seed(20260923)
    loader = _make_loader(
        synthetic_split.dataset_tr, positive_ids=sorted(synthetic_split.tr_positive)
    )
    slot0 = []
    for _ in range(200):
        keys = [str(k) for k in loader.get_indices()]
        slot0.append(keys[0])
    # 第一个槽位从完整训练集均匀抽样：阴性病例必然出现（保留阴性监督）
    assert set(slot0) >= (set(synthetic_split.tr_ids) - synthetic_split.tr_positive)
    # 阳性病例也允许出现在普通槽位（不排除、不固定成单一病例）
    assert set(slot0) & synthetic_split.tr_positive


def test_native_loader_slot1_can_be_negative_but_positive_sampler_never(
    synthetic_split,
):
    """对照：原生均匀采样下 force_fg 槽位可能是阴性病例；PositiveCaseDataLoader 不会。"""
    negatives = set(synthetic_split.tr_ids) - synthetic_split.tr_positive
    np.random.seed(20260924)
    native = nnUNetDataLoader(
        synthetic_split.dataset_tr,
        2,
        (8, 12, 12),
        (8, 12, 12),
        _label_manager(),
        oversample_foreground_percent=0.33,
        sampling_probabilities=None,
        pad_sides=None,
        transforms=None,
        probabilistic_oversampling=False,
    )
    native_slot1 = {str(list(native.get_indices())[1]) for _ in range(200)}
    # 原生 slot1（原生 oversample 下会被 force_fg 的槽位）确实可能抽到阴性病例
    assert native_slot1 & negatives

    np.random.seed(20260924)
    positive_loader = _make_loader(
        synthetic_split.dataset_tr, positive_ids=sorted(synthetic_split.tr_positive)
    )
    positive_slot1 = {str(positive_loader.get_indices()[-1]) for _ in range(200)}
    # PositiveCaseDataLoader 的最后一个槽位永远不出现阴性病例
    assert not (positive_slot1 & negatives)
    assert positive_slot1 <= synthetic_split.tr_positive


def test_force_fg_flags_for_slots(synthetic_split):
    loader = _make_loader(
        synthetic_split.dataset_tr, positive_ids=sorted(synthetic_split.tr_positive)
    )
    # 最后一个（阳性）槽位始终 force foreground
    assert loader.get_do_oversample(1) is True
    # 普通槽位不被强制为阳性
    assert loader.get_do_oversample(0) is False
    # slot 决策函数必须是我们覆盖的确定性版本
    assert loader.get_do_oversample == loader._oversample_last_XX_percent


def test_every_batch_has_one_positive_slot_with_foreground_target(synthetic_split):
    """连续合成采样多个 batch：每个 batch 恰有一个阳性病例槽，且其 target 含前景。"""
    np.random.seed(20260925)
    loader = _make_loader(
        synthetic_split.dataset_tr, positive_ids=sorted(synthetic_split.tr_positive)
    )
    for _ in range(25):
        batch = loader.generate_train_batch()
        keys = [str(k) for k in batch["keys"]]
        assert len(keys) == 2
        assert keys[-1] in synthetic_split.tr_positive
        # 被保证的阳性 patch target 确实含 foreground（病灶中心裁剪生效）
        assert batch["target"][-1].sum() > 0


# ------------------------------------------------- PositiveCaseSamplingMixin
class _MixinHost(PositiveCaseSamplingMixin):
    """提供 get_dataloaders 读取的全部属性/方法的合成宿主（不实例化 nnUNetTrainer）。

    transforms 返回 None（loader 不做增强），与原生 v2.6.2 的
    ``SingleThreadedAugmenter(dl, None)`` 路径一致；deep supervision scales 为 None
    表示不启用 DS 变换（与 get_validation_transforms 的原生默认一致）。
    """

    def __init__(self, dataset_tr, dataset_val) -> None:
        self.dataset_tr = dataset_tr
        self.dataset_val = dataset_val
        self.dataset_class = SyntheticPreprocessedDataset  # 非 None：跳过 infer
        self.preprocessed_dataset_folder = dataset_tr.source_folder
        self.configuration_manager = SimpleNamespace(
            patch_size=np.array([8, 12, 12]),
            use_mask_for_norm=[False, False, False],
        )
        self.label_manager = _label_manager()
        self.batch_size = 2
        self.oversample_foreground_percent = 0.33
        self.probabilistic_oversampling = False
        self.device = SimpleNamespace(type="cpu")
        self.is_cascaded = False
        self.log_calls: list[str] = []

    def _get_deep_supervision_scales(self):
        return None

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        return (-0.1, 0.1), False, np.array([8, 12, 12]), (0, 1, 2)

    def get_training_transforms(self, *args, **kwargs):
        return None

    def get_validation_transforms(self, *args, **kwargs):
        return None

    def get_tr_and_val_datasets(self):
        return self.dataset_tr, self.dataset_val

    def print_to_log_file(self, *args, **kwargs):
        self.log_calls.append(" ".join(str(a) for a in args))


@pytest.fixture()
def mixin_host(synthetic_split, monkeypatch):
    import zonal_reliability_fusion.nnunet.sampling as sampling_module

    # 单线程路径：避免在测试里启动增强子进程
    monkeypatch.setattr(sampling_module, "get_allowed_n_proc_DA", lambda: 0)
    return _MixinHost(synthetic_split.dataset_tr, synthetic_split.dataset_val)


def test_mixin_train_loader_is_positive_val_loader_is_native(
    mixin_host, synthetic_split
):
    mt_train, mt_val = mixin_host.get_dataloaders()
    # 训练 loader：PositiveCaseDataLoader（继承原生 nnUNetDataLoader）
    assert isinstance(mt_train.data_loader, PositiveCaseDataLoader)
    assert isinstance(mt_train.data_loader, nnUNetDataLoader)
    # 验证 loader：必须是原生 nnUNetDataLoader 本类，不是 PositiveCaseDataLoader
    assert type(mt_val.data_loader) is nnUNetDataLoader
    assert not isinstance(mt_val.data_loader, PositiveCaseDataLoader)
    # 阳性集合只含训练 identifiers
    assert (
        set(mt_train.data_loader.positive_case_identifiers)
        == synthetic_split.tr_positive
    )


def test_mixin_summary_uses_runtime_numbers(mixin_host, synthetic_split):
    mixin_host.get_dataloaders()
    assert mixin_host.log_calls, "必须打印一次结构化摘要"
    summary = mixin_host.log_calls[0]
    assert f"training_cases={len(synthetic_split.tr_ids)}" in summary
    assert f"positive_cases={len(synthetic_split.tr_positive)}" in summary
    assert (
        f"negative_cases={len(synthetic_split.tr_ids) - len(synthetic_split.tr_positive)}"
        in summary
    )
    assert "positive_cases_per_batch=1" in summary
    assert "batch_size=2" in summary
    assert "guaranteed_positive_patch_fraction=0.5" in summary


def test_mixin_batches_always_have_positive_slot_and_fg(mixin_host, synthetic_split):
    np.random.seed(20260926)
    mt_train, _ = mixin_host.get_dataloaders()
    for _ in range(10):
        batch = next(mt_train)
        keys = [str(k) for k in batch["keys"]]
        assert keys[-1] in synthetic_split.tr_positive
        assert batch["target"][-1].sum() > 0


def test_mixin_validation_batches_only_use_val_identifiers(mixin_host, synthetic_split):
    _, mt_val = mixin_host.get_dataloaders()
    for _ in range(10):
        batch = next(mt_val)
        keys = {str(k) for k in batch["keys"]}
        assert keys <= set(synthetic_split.val_ids)


# ---------------------------------------- gate + positive_sampling 组合 Trainer 的数据加载
def _make_combined_host(trainer_cls, dataset_tr, dataset_val):
    """基于指定 Trainer 的**真实 MRO** 构造宿主（不调用 nnUNetTrainer.__init__）。

    这里直接继承被测 Trainer 类（而非只继承 mixin），因此 ``get_dataloaders``
    沿真实类层次解析，能发现“新参数被 gate 类抢先解析”之类的问题。
    """

    class _Host(trainer_cls):
        def __init__(self) -> None:
            self.dataset_tr = dataset_tr
            self.dataset_val = dataset_val
            self.dataset_class = SyntheticPreprocessedDataset
            self.preprocessed_dataset_folder = dataset_tr.source_folder
            self.configuration_manager = SimpleNamespace(
                patch_size=np.array([8, 12, 12]),
                use_mask_for_norm=[False, False, False],
            )
            self.label_manager = _label_manager()
            self.batch_size = 2
            self.oversample_foreground_percent = 0.33
            self.probabilistic_oversampling = False
            self.device = SimpleNamespace(type="cpu")
            self.is_cascaded = False
            self.log_calls: list[str] = []

        def _get_deep_supervision_scales(self):
            return None

        def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
            return (-0.1, 0.1), False, np.array([8, 12, 12]), (0, 1, 2)

        def get_training_transforms(self, *args, **kwargs):
            return None

        def get_validation_transforms(self, *args, **kwargs):
            return None

        def get_tr_and_val_datasets(self):
            return self.dataset_tr, self.dataset_val

        def print_to_log_file(self, *args, **kwargs):
            self.log_calls.append(" ".join(str(a) for a in args))

    return _Host()


@pytest.mark.parametrize(
    "trainer_cls",
    [
        nnUNetTrainerPICAI_ImageGate_PositiveSampling_NoFFT,
        nnUNetTrainerPICAI_AnatomyGate_PositiveSampling_NoFFT,
        nnUNetTrainerPICAI_FeatureNoGate_PositiveSampling_NoFFT,
        nnUNetTrainerPICAI_FeatureImageGate_PositiveSampling_NoFFT,
        nnUNetTrainerPICAI_FeatureAnatomyGate_PositiveSampling_NoFFT,
    ],
)
def test_combined_trainers_dataloaders_via_real_mro(
    synthetic_split, monkeypatch, trainer_cls
):
    """组合 Trainer 的真实 MRO：训练 loader=PositiveCaseDataLoader，验证 loader=原生。"""
    import zonal_reliability_fusion.nnunet.sampling as sampling_module

    monkeypatch.setattr(sampling_module, "get_allowed_n_proc_DA", lambda: 0)
    host = _make_combined_host(
        trainer_cls, synthetic_split.dataset_tr, synthetic_split.dataset_val
    )
    # 真实 MRO 解析：get_dataloaders 必须来自采样 mixin，而不是被 gate 类或原生覆盖
    assert type(host).get_dataloaders is PositiveCaseSamplingMixin.get_dataloaders

    mt_train, mt_val = host.get_dataloaders()
    assert isinstance(mt_train.data_loader, PositiveCaseDataLoader)
    assert isinstance(mt_train.data_loader, nnUNetDataLoader)
    assert type(mt_val.data_loader) is nnUNetDataLoader
    assert not isinstance(mt_val.data_loader, PositiveCaseDataLoader)
    assert set(mt_train.data_loader.positive_case_identifiers) == (
        synthetic_split.tr_positive
    )
    # 每批最后一个槽位固定阳性病例并强制病灶中心裁剪
    np.random.seed(20260927)
    for _ in range(5):
        batch = next(mt_train)
        keys = [str(k) for k in batch["keys"]]
        assert keys[-1] in synthetic_split.tr_positive
        assert batch["target"][-1].sum() > 0
    assert "positive_cases_per_batch=1" in host.log_calls[0]
    # 验证 loader 只用验证 identifiers（无验证泄漏）
    for _ in range(5):
        batch = next(mt_val)
        assert {str(k) for k in batch["keys"]} <= set(synthetic_split.val_ids)


# ------------------------------------------------------------------ 新 Trainer
def test_positive_sampling_trainer_mro():
    cls = nnUNetTrainerPICAI_FLCE_PositiveSampling_NoFFT
    assert cls.__mro__ == (
        cls,
        PositiveCaseSamplingMixin,
        nnUNetTrainerPICAI_FLCE_NoFFT,
        NoFFTAugmentationMixin,
        PICAIFocalCrossEntropyLossMixin,
        nnUNetTrainer,
        object,
    )


def test_positive_sampling_trainer_forbidden_bases():
    cls = nnUNetTrainerPICAI_FLCE_PositiveSampling_NoFFT
    assert issubclass(cls, nnUNetTrainer)
    assert issubclass(cls, nnUNetTrainerPICAI_FLCE_NoFFT)
    assert issubclass(cls, PositiveCaseSamplingMixin)
    for forbidden in (
        nnUNetTrainerPICAI_DiceCE_NoFFT,
        nnUNetTrainerPICAI_ImageGate,
        nnUNetTrainerPICAI_AnatomyGate,
        _GatedTrainerBase,
    ):
        assert not issubclass(cls, forbidden), f"不得继承 {forbidden.__name__}"
        assert forbidden not in cls.__mro__


class _MinimalLossBuilderState:
    """只提供 _build_loss 实际读取的属性（与 test_nnunet_trainers 同一做法）。"""

    def __init__(self, *, enable_deep_supervision: bool, num_scales: int = 3) -> None:
        self.label_manager = SimpleNamespace(has_regions=False, ignore_label=None)
        self.configuration_manager = SimpleNamespace(batch_dice=False)
        self.is_ddp = False
        self.enable_deep_supervision = enable_deep_supervision
        self._num_scales = num_scales

    def _do_i_compile(self) -> bool:
        return False

    def _get_deep_supervision_scales(self) -> list:
        return [[1.0, 1.0, 1.0] for _ in range(self._num_scales)]


def test_positive_sampling_build_loss_is_flce():
    cls = nnUNetTrainerPICAI_FLCE_PositiveSampling_NoFFT
    assert cls._build_loss is PICAIFocalCrossEntropyLossMixin._build_loss
    assert cls._build_loss is nnUNetTrainerPICAI_FLCE_NoFFT._build_loss
    assert cls._build_loss is not nnUNetTrainer._build_loss

    loss = cls._build_loss(_MinimalLossBuilderState(enable_deep_supervision=True))
    assert isinstance(loss, DeepSupervisionWrapper)
    assert isinstance(loss.loss, PiCAIFocalCrossEntropyLoss)
    assert loss.loss.focal_weight == 0.5
    assert loss.loss.ce_weight == 0.5
    assert loss.loss.gamma == 2.0


def test_positive_sampling_network_is_plain_conv_unet_without_gate(synthetic_arch):
    cls = nnUNetTrainerPICAI_FLCE_PositiveSampling_NoFFT
    # 不覆盖 builder：MRO 直接解析到原生 nnUNetTrainer.build_network_architecture
    assert cls.build_network_architecture is nnUNetTrainer.build_network_architecture
    net = cls.build_network_architecture(
        synthetic_arch["architecture_class_name"],
        synthetic_arch["arch_init_kwargs"],
        synthetic_arch["arch_init_kwargs_req_import"],
        3,
        2,
        False,
    )
    assert isinstance(net, PlainConvUNet)
    assert not isinstance(net, GatedNNUNet)
    assert not hasattr(net, "gate")
    assert not hasattr(net, "backbone")
    assert type(net).__module__.startswith("dynamic_network_architectures.")
    # 仍为 3 个 MRI 输入通道（Dataset605 的 T2W/ADC/HBV）
    net.eval()
    with torch.no_grad():
        out = net(torch.randn(1, 3, 8, 16, 16))
    assert isinstance(out, torch.Tensor)
    assert out.shape[1] == 2


def test_positive_sampling_nofft_still_disables_fft_benchmark():
    transforms = nnUNetTrainerPICAI_FLCE_PositiveSampling_NoFFT.get_training_transforms(
        patch_size=np.array([8, 16, 16]),
        rotation_for_DA=(-0.1, 0.1),
        deep_supervision_scales=[[1, 1, 1], [0.5, 0.5, 0.5]],
        mirror_axes=(0, 1, 2),
        do_dummy_2d_data_aug=False,
        use_mask_for_norm=[False, False, False],
        is_cascaded=False,
        foreground_labels=(1,),
    )
    blurs = [
        getattr(t, "transform", t)
        for t in transforms.transforms
        if isinstance(getattr(t, "transform", t), GaussianBlurTransform)
    ]
    assert len(blurs) == 1
    assert blurs[0].benchmark is False


def test_five_trainer_names_and_output_dirs_are_distinct():
    classes = (
        nnUNetTrainerPICAI_FLCE_NoFFT,
        nnUNetTrainerPICAI_DiceCE_NoFFT,
        nnUNetTrainerPICAI_ImageGate,
        nnUNetTrainerPICAI_AnatomyGate,
        nnUNetTrainerPICAI_FLCE_PositiveSampling_NoFFT,
        nnUNetTrainerPICAI_ImageGate_PositiveSampling_NoFFT,
        nnUNetTrainerPICAI_AnatomyGate_PositiveSampling_NoFFT,
    )
    names = [c.__name__ for c in classes]
    assert len(set(names)) == 7
    # nnU-Net 输出目录由 Trainer 类名自然形成：类名互异 => 目录互异
    dirs = {f"{n}__nnUNetPlans__3d_fullres" for n in names}
    assert len(dirs) == 7
    assert (
        "nnUNetTrainerPICAI_FLCE_PositiveSampling_NoFFT__nnUNetPlans__3d_fullres"
        in dirs
    )
    assert (
        "nnUNetTrainerPICAI_ImageGate_PositiveSampling_NoFFT__nnUNetPlans__3d_fullres"
        in dirs
    )
    assert (
        "nnUNetTrainerPICAI_AnatomyGate_PositiveSampling_NoFFT__nnUNetPlans__3d_fullres"
        in dirs
    )


def test_project_trainers_resolves_new_trainer():
    from zonal_reliability_fusion.nnunet.trainers import (
        PROJECT_TRAINERS,
        resolve_trainer_class,
    )

    name = "nnUNetTrainerPICAI_FLCE_PositiveSampling_NoFFT"
    assert PROJECT_TRAINERS[name] is nnUNetTrainerPICAI_FLCE_PositiveSampling_NoFFT
    assert resolve_trainer_class(name) is nnUNetTrainerPICAI_FLCE_PositiveSampling_NoFFT
    for cls in (
        nnUNetTrainerPICAI_ImageGate_PositiveSampling_NoFFT,
        nnUNetTrainerPICAI_AnatomyGate_PositiveSampling_NoFFT,
    ):
        assert PROJECT_TRAINERS[cls.__name__] is cls
        assert resolve_trainer_class(cls.__name__) is cls


def test_train_entry_resolves_positive_sampling_variant():
    train_script = PROJECT_ROOT / "scripts" / "train" / "train_nnunet.py"
    spec = importlib.util.spec_from_file_location("train_nnunet_entry_ps", train_script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert set(module.VARIANT_TO_TRAINER) == {
        "baseline",
        "optimized_baseline",
        "image_gate",
        "anatomy_gate",
        "positive_sampling",
        "image_gate_positive_sampling",
        "anatomy_gate_positive_sampling",
        "feature_no_gate_positive_sampling",
        "feature_image_gate_positive_sampling",
        "feature_anatomy_gate_positive_sampling",
    }
    assert (
        module.resolve_trainer_class("positive_sampling")
        is nnUNetTrainerPICAI_FLCE_PositiveSampling_NoFFT
    )
    assert (
        module.resolve_trainer_class("image_gate_positive_sampling")
        is nnUNetTrainerPICAI_ImageGate_PositiveSampling_NoFFT
    )
    assert (
        module.resolve_trainer_class("anatomy_gate_positive_sampling")
        is nnUNetTrainerPICAI_AnatomyGate_PositiveSampling_NoFFT
    )


def test_predict_entry_resolves_new_trainer_name():
    predict_script = PROJECT_ROOT / "scripts" / "inference" / "predict_nnunet.py"
    spec = importlib.util.spec_from_file_location(
        "predict_nnunet_entry_ps", predict_script
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    name = "nnUNetTrainerPICAI_FLCE_PositiveSampling_NoFFT"
    assert name in module.PROJECT_TRAINER_NAMES
    assert (
        module.resolve_project_trainer(name)
        is nnUNetTrainerPICAI_FLCE_PositiveSampling_NoFFT
    )
    for cls in (
        nnUNetTrainerPICAI_ImageGate_PositiveSampling_NoFFT,
        nnUNetTrainerPICAI_AnatomyGate_PositiveSampling_NoFFT,
    ):
        assert cls.__name__ in module.PROJECT_TRAINER_NAMES
        assert module.resolve_project_trainer(cls.__name__) is cls
