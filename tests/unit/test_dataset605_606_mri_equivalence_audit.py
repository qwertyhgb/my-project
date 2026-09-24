"""Dataset605/606 前三 MRI 通道审计脚本的纯合成单元测试。

只使用 ``tmp_path`` 与内存中的合成数组（占位文件仅用于存在性检查），**不读取任何真实医学数据**，
也不触碰 ``workdir/`` 与 ``outputs/``。覆盖：完全一致→PASS、单 voxel/形状/缺病例/split/label/
NaN/Inf/dtype/标签越界→FAIL、PZ/TZ 不参与前三 MRI 数值判定、报告含 mismatch 明细、输入文件不被修改、
``--max-cases`` 只跑子集时永不判 PASS、CLI 的覆盖保护与环境变量 fail-closed。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUDIT_SCRIPT = (
    PROJECT_ROOT / "scripts" / "data" / "audit_dataset605_606_mri_equivalence.py"
)


def _load_audit():
    spec = importlib.util.spec_from_file_location(
        "audit_dataset605_606_mri_equivalence", AUDIT_SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = _load_audit()

CASE_IDS = ("case_a", "case_b", "case_c")


class FakePreprocessedCaseSource:
    """与 ``PreprocessedCaseSource`` 同接口的合成数据源（内存数组 + 占位文件）。"""

    def __init__(self, folder: Path, cases: dict[str, tuple[np.ndarray, np.ndarray]]):
        self.case_folder = Path(folder)
        self.case_folder.mkdir(parents=True, exist_ok=True)
        # 刻意持有调用方的 dict 本身（测试会在构造后替换病例，需与真实 source 的语义一致）
        self._cases = cases
        for case_identifier in self._cases:
            for name in (
                f"{case_identifier}.b2nd",
                f"{case_identifier}_seg.b2nd",
                f"{case_identifier}.pkl",
            ):
                (self.case_folder / name).write_bytes(b"placeholder")

    @property
    def identifiers(self) -> tuple[str, ...]:
        return tuple(self._cases)

    def load(self, case_identifier: str) -> tuple[np.ndarray, np.ndarray]:
        data, seg = self._cases[case_identifier]
        return data, seg


def _mri_data(seed: int, shape=(3, 2, 2, 2)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=shape).astype(np.float32)


def _prior_channels(seed: int, shape=(2, 2, 2, 2)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random(size=shape).astype(np.float32)


def _seg(shape=(2, 2, 2)) -> np.ndarray:
    seg = np.zeros(shape, dtype=np.uint8)
    seg[0, 0, 0] = 1
    return seg


def _cases_605() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {
        cid: (_mri_data(seed), _seg()) for seed, cid in enumerate(CASE_IDS, start=1)
    }


def _cases_606(
    *, mri_source: dict[str, tuple[np.ndarray, np.ndarray]] | None = None
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    base = _cases_605() if mri_source is None else mri_source
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for index, (cid, (mri, seg)) in enumerate(base.items()):
        data = np.concatenate([mri, _prior_channels(100 + index)], axis=0)
        out[cid] = (data, seg.copy())
    return out


def _dataset_json_605() -> dict:
    return {
        "channel_names": {"0": "T2W", "1": "ADC", "2": "HBV"},
        "labels": {"background": 0, "lesion": 1},
        "file_ending": ".nii.gz",
        "numTraining": len(CASE_IDS),
    }


def _dataset_json_606() -> dict:
    return {
        "channel_names": {
            "0": "T2W",
            "1": "ADC",
            "2": "HBV",
            "3": "noNorm",
            "4": "noNorm",
        },
        "labels": {"background": 0, "lesion": 1},
        "file_ending": ".nii.gz",
        "numTraining": len(CASE_IDS),
    }


def _splits() -> list[dict]:
    return [{"train": list(CASE_IDS[:2]), "val": list(CASE_IDS[2:])}]


def _run(
    tmp_path: Path,
    cases_605=None,
    cases_606=None,
    splits_605=None,
    splits_606=None,
    **kwargs,
):
    source_605 = FakePreprocessedCaseSource(
        tmp_path / "605", _cases_605() if cases_605 is None else cases_605
    )
    source_606 = FakePreprocessedCaseSource(
        tmp_path / "606", _cases_606() if cases_606 is None else cases_606
    )
    return audit.run_audit(
        source_605=source_605,
        source_606=source_606,
        splits_605=_splits() if splits_605 is None else splits_605,
        splits_606=_splits() if splits_606 is None else splits_606,
        dataset_json_605=_dataset_json_605(),
        dataset_json_606=_dataset_json_606(),
        metadata_605={
            "dataset_name": "Dataset605_PICAI",
            "configuration": "3d_fullres",
        },
        metadata_606={
            "dataset_name": "Dataset606_PICAI_Zonal",
            "configuration": "3d_fullres",
        },
        fold=0,
        progress=False,
        **kwargs,
    )


def _snapshot(folder: Path) -> dict[str, bytes]:
    return {
        path.name: path.read_bytes()
        for path in sorted(folder.iterdir())
        if path.is_file()
    }


# --------------------------------------------------------------------- PASS 路径


def test_identical_arrays_pass(tmp_path: Path) -> None:
    report = _run(tmp_path)
    assert report["status"] == audit.STATUS_PASS
    assert report["problems"] == []
    assert report["case_set_equal"] and report["split_equal"] and report["label_equal"]
    assert report["n_cases_checked"] == len(CASE_IDS)
    assert report["n_cases_exact_equal"] == len(CASE_IDS)
    assert report["n_cases_mismatch"] == 0
    assert report["mismatched_cases"] == []
    for name in audit.MRI_CHANNEL_NAMES:
        entry = report["per_channel"][name]
        assert entry["exact_equal_cases"] == len(CASE_IDS)
        assert entry["mismatch_cases"] == 0
        assert entry["global_max_abs_diff"] == 0.0
        assert entry["global_mean_abs_diff"] == 0.0
    assert report["channels"] == ["T2W", "ADC", "HBV"]


def test_prior_channels_are_excluded_from_mri_equivalence(tmp_path: Path) -> None:
    """PZ/TZ 通道完全不同也不影响前三 MRI 的判定。"""
    cases_606 = _cases_606()
    for index, cid in enumerate(CASE_IDS):
        data, seg = cases_606[cid]
        data = data.copy()
        data[3:] = (index + 1) * 7.5  # PZ/TZ 整块改成完全不同的值
        cases_606[cid] = (data, seg)
    report = _run(tmp_path, cases_606=cases_606)
    assert report["status"] == audit.STATUS_PASS
    assert report["n_cases_mismatch"] == 0
    assert report["per_channel"]["T2W"]["mismatch_cases"] == 0


# --------------------------------------------------------------------- FAIL 路径


def test_single_voxel_difference_fails_with_detail(tmp_path: Path) -> None:
    cases_606 = _cases_606()
    data, seg = cases_606["case_b"]
    data = data.copy()
    data[1, 0, 0, 0] += 1e-3  # 单个 ADC voxel 不同
    cases_606["case_b"] = (data, seg)
    report = _run(tmp_path, cases_606=cases_606)

    assert report["status"] == audit.STATUS_FAIL
    assert report["n_cases_checked"] == len(CASE_IDS)
    assert report["n_cases_exact_equal"] == len(CASE_IDS) - 1
    assert report["n_cases_mismatch"] == 1
    assert report["per_channel"]["ADC"]["mismatch_cases"] == 1
    assert report["per_channel"]["ADC"]["global_max_abs_diff"] == pytest.approx(
        1e-3, rel=1e-4
    )
    assert report["per_channel"]["T2W"]["mismatch_cases"] == 0

    details = [m for m in report["mismatched_cases"] if m["channel"] == "ADC"]
    assert len(details) == 1
    detail = details[0]
    assert detail["case_id"] == "case_b"
    assert detail["kind"] == "mri_values_differ"
    assert detail["unequal_voxels"] == 1
    assert detail["fraction_unequal"] == pytest.approx(1 / 8)
    assert detail["max_abs_diff"] == pytest.approx(1e-3, rel=1e-4)
    assert detail["mean_abs_diff"] == pytest.approx(1e-3 / 8, rel=1e-4)
    assert audit.STATUS_PASS not in report["status"]


def test_spatial_shape_difference_fails(tmp_path: Path) -> None:
    cases_605 = _cases_605()
    mri, seg = cases_605["case_a"]
    cases_605["case_a"] = (mri[:, :1], seg[:1])
    report = _run(tmp_path, cases_605=cases_605)
    assert report["status"] == audit.STATUS_FAIL
    kinds = {m["kind"] for m in report["mismatched_cases"]}
    assert "spatial_shape_mismatch" in kinds


def test_channel_count_difference_fails(tmp_path: Path) -> None:
    cases_606 = _cases_606()
    data, seg = cases_606["case_a"]
    cases_606["case_a"] = (data[:4], seg)  # 缺一个 PZ/TZ 通道
    report = _run(tmp_path, cases_606=cases_606)
    assert report["status"] == audit.STATUS_FAIL
    assert "channel_count_mismatch" in {m["kind"] for m in report["mismatched_cases"]}


def test_missing_case_fails(tmp_path: Path) -> None:
    cases_606 = _cases_606()
    cases_606.pop("case_c")
    report = _run(tmp_path, cases_606=cases_606)
    assert report["status"] == audit.STATUS_FAIL
    assert report["case_set_equal"] is False
    assert report["n_cases_605"] == len(CASE_IDS)
    assert report["n_cases_606"] == len(CASE_IDS) - 1
    missing = [
        m
        for m in report["mismatched_cases"]
        if m["kind"] == "case_missing_in_other_dataset"
    ]
    assert len(missing) == 1
    assert missing[0]["case_id"] == "case_c"
    assert missing[0]["present_in"] == "605"


def test_missing_preprocessed_file_fails(tmp_path: Path) -> None:
    source_605 = FakePreprocessedCaseSource(tmp_path / "605", _cases_605())
    source_606 = FakePreprocessedCaseSource(tmp_path / "606", _cases_606())
    (source_605.case_folder / "case_a.pkl").unlink()  # 预处理 metadata 缺失
    report = audit.run_audit(
        source_605=source_605,
        source_606=source_606,
        splits_605=_splits(),
        splits_606=_splits(),
        dataset_json_605=_dataset_json_605(),
        dataset_json_606=_dataset_json_606(),
        metadata_605={
            "dataset_name": "Dataset605_PICAI",
            "configuration": "3d_fullres",
        },
        metadata_606={
            "dataset_name": "Dataset606_PICAI_Zonal",
            "configuration": "3d_fullres",
        },
        fold=0,
        progress=False,
    )
    assert report["status"] == audit.STATUS_FAIL
    assert report["label_equal"] is False
    failures = report["load_failures"]
    assert len(failures) == 1
    assert failures[0]["kind"] == "missing_preprocessed_file"
    assert failures[0]["files"] == ["case_a.pkl"]


def test_split_difference_fails(tmp_path: Path) -> None:
    splits_606 = [{"train": ["case_a", "case_c"], "val": ["case_b"]}]
    report = _run(tmp_path, splits_606=splits_606)
    assert report["status"] == audit.STATUS_FAIL
    assert report["split_equal"] is False
    assert report["splits"]["all_folds_train_val_set_equal"] is False


def test_split_order_only_difference_does_not_fail(tmp_path: Path) -> None:
    """集合相同、仅顺序不同 ⇒ split_equal 仍为 True，但记录顺序差异。"""
    splits_606 = [{"train": list(reversed(CASE_IDS[:2])), "val": list(CASE_IDS[2:])}]
    report = _run(tmp_path, splits_606=splits_606)
    assert report["status"] == audit.STATUS_PASS
    assert report["split_equal"] is True
    assert report["splits"]["fold_0"]["train_set_equal"] is True
    assert report["splits"]["fold_0"]["train_order_equal"] is False


def test_fold_count_difference_fails(tmp_path: Path) -> None:
    splits_606 = _splits() + [{"train": list(CASE_IDS[:2]), "val": list(CASE_IDS[2:])}]
    report = _run(tmp_path, splits_606=splits_606)
    assert report["status"] == audit.STATUS_FAIL
    assert report["splits"]["n_folds_equal"] is False


def test_label_difference_fails(tmp_path: Path) -> None:
    cases_606 = _cases_606()
    data, seg = cases_606["case_a"]
    seg = seg.copy()
    seg[1, 1, 1] = 1  # 标签多出一个前景体素
    cases_606["case_a"] = (data, seg)
    report = _run(tmp_path, cases_606=cases_606)
    assert report["status"] == audit.STATUS_FAIL
    assert report["label_equal"] is False
    details = [m for m in report["mismatched_cases"] if m["channel"] == "SEG"]
    assert details and details[0]["kind"] == "label_values_differ"
    assert details[0]["unequal_voxels"] == 1


def test_label_out_of_range_fails(tmp_path: Path) -> None:
    cases_606 = _cases_606()
    data, seg = cases_606["case_a"]
    seg = seg.copy()
    seg[0, 0, 1] = 2  # 越界标签（Dataset605/606 只有 0/1）
    cases_606["case_a"] = (data, seg)
    report = _run(tmp_path, cases_606=cases_606)
    assert report["status"] == audit.STATUS_FAIL
    kinds = {m["kind"] for m in report["mismatched_cases"]}
    assert "unexpected_label_values" in kinds


def test_non_finite_values_fail(tmp_path: Path) -> None:
    for bad in (np.nan, np.inf):
        cases_606 = _cases_606()
        data, seg = cases_606["case_a"]
        data = data.copy()
        data[2, 0, 0, 0] = bad
        cases_606["case_a"] = (data, seg)
        report = _run(tmp_path, cases_606=cases_606)
        assert report["status"] == audit.STATUS_FAIL
        assert "non_finite_values" in {m["kind"] for m in report["mismatched_cases"]}


def test_dtype_difference_fails(tmp_path: Path) -> None:
    cases_606 = _cases_606()
    data, seg = cases_606["case_a"]
    cases_606["case_a"] = (data.astype(np.float16), seg)
    report = _run(tmp_path, cases_606=cases_606)
    assert report["status"] == audit.STATUS_FAIL
    assert "dtype_mismatch" in {m["kind"] for m in report["mismatched_cases"]}


def test_channel_name_mismatch_fails(tmp_path: Path) -> None:
    dataset_json_606 = _dataset_json_606()
    dataset_json_606["channel_names"]["1"] = "DWI"  # 第二通道不是 ADC
    source_605 = FakePreprocessedCaseSource(tmp_path / "605", _cases_605())
    source_606 = FakePreprocessedCaseSource(tmp_path / "606", _cases_606())
    report = audit.run_audit(
        source_605=source_605,
        source_606=source_606,
        splits_605=_splits(),
        splits_606=_splits(),
        dataset_json_605=_dataset_json_605(),
        dataset_json_606=dataset_json_606,
        metadata_605={
            "dataset_name": "Dataset605_PICAI",
            "configuration": "3d_fullres",
        },
        metadata_606={
            "dataset_name": "Dataset606_PICAI_Zonal",
            "configuration": "3d_fullres",
        },
        fold=0,
        progress=False,
    )
    assert report["status"] == audit.STATUS_FAIL
    assert report["metadata"]["mri_channel_names_equal"] is False


def test_zero_padded_channel_keys_pass(tmp_path: Path) -> None:
    """真实预处理 dataset.json 使用 "0000" 形式的通道键；键格式差异不得误判为通道名不一致。"""
    padded_605 = {
        "channel_names": {"0000": "T2W", "0001": "ADC", "0002": "HBV"},
        "labels": {"background": 0, "lesion": 1},
        "file_ending": ".nii.gz",
        "numTraining": len(CASE_IDS),
    }
    padded_606 = {
        "channel_names": {
            "0000": "T2W",
            "0001": "ADC",
            "0002": "HBV",
            "0003": "noNorm",
            "0004": "noNorm",
        },
        "labels": {"background": 0, "lesion": 1},
        "file_ending": ".nii.gz",
        "numTraining": len(CASE_IDS),
    }
    source_605 = FakePreprocessedCaseSource(tmp_path / "605", _cases_605())
    source_606 = FakePreprocessedCaseSource(tmp_path / "606", _cases_606())
    report = audit.run_audit(
        source_605=source_605,
        source_606=source_606,
        splits_605=_splits(),
        splits_606=_splits(),
        dataset_json_605=padded_605,
        dataset_json_606=padded_606,
        metadata_605={
            "dataset_name": "Dataset605_PICAI",
            "configuration": "3d_fullres",
        },
        metadata_606={
            "dataset_name": "Dataset606_PICAI_Zonal",
            "configuration": "3d_fullres",
        },
        fold=0,
        progress=False,
    )
    assert report["status"] == audit.STATUS_PASS
    metadata = report["metadata"]
    assert metadata["mri_channel_names_equal"] is True
    assert metadata["ordered_channel_names_605"] == ["T2W", "ADC", "HBV"]
    assert metadata["ordered_channel_names_606"][:3] == ["T2W", "ADC", "HBV"]
    assert metadata["n_channels_605"] == 3
    assert metadata["n_channels_606"] == 5


# --------------------------------------------------------------------- 报告与只读性


def test_report_is_json_serializable_and_contains_required_keys(tmp_path: Path) -> None:
    cases_606 = _cases_606()
    data, seg = cases_606["case_c"]
    data = data.copy()
    data[0, 1, 1, 1] += 0.5  # 制造一个 T2W mismatch，检查诊断字段
    cases_606["case_c"] = (data, seg)
    report = _run(tmp_path, cases_606=cases_606)
    payload = json.loads(json.dumps(report, ensure_ascii=False))

    for key in (
        "status",
        "dataset_605",
        "dataset_606",
        "n_cases_605",
        "n_cases_606",
        "case_set_equal",
        "split_equal",
        "label_equal",
        "n_cases_checked",
        "n_cases_exact_equal",
        "n_cases_mismatch",
        "channels",
        "per_channel",
        "mismatched_cases",
    ):
        assert key in payload, key
    for name in audit.MRI_CHANNEL_NAMES:
        for key in (
            "exact_equal_cases",
            "mismatch_cases",
            "global_max_abs_diff",
            "global_mean_abs_diff",
        ):
            assert key in payload["per_channel"][name], (name, key)
    mismatch = payload["mismatched_cases"][0]
    for key in (
        "case_id",
        "channel",
        "max_abs_diff",
        "mean_abs_diff",
        "unequal_voxels",
    ):
        assert key in mismatch, key
    assert payload["inputs_modified"] is False


def test_inputs_are_not_modified(tmp_path: Path) -> None:
    cases_605 = _cases_605()
    cases_606 = _cases_606()
    cases_605["case_a"] = (cases_605["case_a"][0].copy(), cases_605["case_a"][1].copy())
    cases_606["case_a"] = (cases_606["case_a"][0].copy(), cases_606["case_a"][1].copy())
    source_605 = FakePreprocessedCaseSource(tmp_path / "605", cases_605)
    source_606 = FakePreprocessedCaseSource(tmp_path / "606", cases_606)
    before_605 = _snapshot(source_605.case_folder)
    before_606 = _snapshot(source_606.case_folder)
    arrays_before = {
        "605": (cases_605["case_a"][0].copy(), cases_605["case_a"][1].copy()),
        "606": (cases_606["case_a"][0].copy(), cases_606["case_a"][1].copy()),
    }

    # 制造 mismatch，确保审计真的走完比较路径
    data, seg = cases_606["case_b"]
    data = data.copy()
    data[0, 0, 0, 0] += 1.0
    cases_606["case_b"] = (data, seg)

    report = audit.run_audit(
        source_605=source_605,
        source_606=source_606,
        splits_605=_splits(),
        splits_606=_splits(),
        dataset_json_605=_dataset_json_605(),
        dataset_json_606=_dataset_json_606(),
        metadata_605={
            "dataset_name": "Dataset605_PICAI",
            "configuration": "3d_fullres",
        },
        metadata_606={
            "dataset_name": "Dataset606_PICAI_Zonal",
            "configuration": "3d_fullres",
        },
        fold=0,
        progress=False,
    )
    assert report["status"] == audit.STATUS_FAIL
    assert _snapshot(source_605.case_folder) == before_605
    assert _snapshot(source_606.case_folder) == before_606
    assert np.array_equal(cases_605["case_a"][0], arrays_before["605"][0])
    assert np.array_equal(cases_605["case_a"][1], arrays_before["605"][1])
    assert np.array_equal(cases_606["case_a"][0], arrays_before["606"][0])
    assert np.array_equal(cases_606["case_a"][1], arrays_before["606"][1])
    assert not any(p.name.endswith(".json") for p in source_605.case_folder.iterdir())


def test_max_cases_subset_never_passes(tmp_path: Path) -> None:
    report = _run(tmp_path, max_cases=1)
    assert report["status"] == audit.STATUS_PARTIAL
    assert report["n_cases_checked"] == 1
    assert any("--max-cases" in problem for problem in report["problems"])


# --------------------------------------------------------------------- CLI 保护


def test_cli_requires_nnunet_preprocessed_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("nnUNet_preprocessed", raising=False)
    with pytest.raises(audit.AuditError):
        audit.main(["--output", str(tmp_path / "report.json")])


def test_cli_refuses_to_overwrite_existing_report(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("nnUNet_preprocessed", str(tmp_path))
    existing = tmp_path / "report.json"
    existing.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit) as excinfo:
        audit.main(["--output", str(existing)])
    assert "拒绝覆盖" in str(excinfo.value)


def test_cli_parser_defaults() -> None:
    args = audit.build_parser().parse_args([])
    assert args.configuration == "3d_fullres"
    assert args.fold == 0
    assert args.max_cases is None
    assert args.overwrite is False
    assert args.no_progress is False


# --------------------------------------------- 真实文件路径（磁盘上的合成预处理目录）


def test_load_json_accepts_top_level_array(tmp_path: Path) -> None:
    """``splits_final.json`` 顶层是数组：``_load_json`` 不得要求 JSON 对象。"""
    path = tmp_path / "splits_final.json"
    path.write_text(json.dumps([{"train": ["a"], "val": ["b"]}]), encoding="utf-8")
    assert isinstance(audit._load_json(path), list)
    object_path = tmp_path / "dataset.json"
    object_path.write_text(json.dumps({"a": 1}), encoding="utf-8")
    assert isinstance(audit._load_json(object_path), dict)
    with pytest.raises(audit.AuditError):
        audit._load_json_object(path)  # dataset.json 必须是对象


def _write_preprocessed_dataset(
    root: Path,
    name: str,
    cases: dict[str, tuple[np.ndarray, np.ndarray]],
) -> Path:
    """用 nnU-Net 自己的 ``save_case`` 写真实格式的 ``.b2nd`` / ``.pkl``（磁盘端到端测试用）。"""
    from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2

    dataset_dir = root / name
    case_folder = dataset_dir / "nnUNetPlans_3d_fullres"
    case_folder.mkdir(parents=True)
    for case_identifier, (data, seg) in cases.items():
        nnUNetDatasetBlosc2.save_case(
            data,
            seg,
            {"case_identifier": case_identifier},
            str(case_folder / case_identifier),
        )
    return dataset_dir


def _run_cli_on_disk(
    tmp_path: Path,
    monkeypatch,
    *,
    cases_605: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    cases_606: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
):
    """在磁盘上造出完整预处理目录 + dataset.json/splits_final.json，然后跑 ``main()``。"""
    root = tmp_path / "nnUNet_preprocessed"
    dataset605 = _write_preprocessed_dataset(
        root, "Dataset605_PICAI", _cases_605() if cases_605 is None else cases_605
    )
    dataset606 = _write_preprocessed_dataset(
        root, "Dataset606_PICAI_Zonal", _cases_606() if cases_606 is None else cases_606
    )
    for dataset_dir, dataset_json in (
        (dataset605, _dataset_json_605()),
        (dataset606, _dataset_json_606()),
    ):
        (dataset_dir / "dataset.json").write_text(
            json.dumps(dataset_json), encoding="utf-8"
        )
        (dataset_dir / "splits_final.json").write_text(
            json.dumps(_splits()), encoding="utf-8"
        )
    monkeypatch.setenv("nnUNet_preprocessed", str(root))
    output = tmp_path / "report.json"
    exit_code = audit.main(["--output", str(output), "--no-progress"])
    return exit_code, output, dataset605, dataset606


def test_cli_end_to_end_pass_on_real_b2nd_files(tmp_path: Path, monkeypatch) -> None:
    """端到端：磁盘上的合成预处理目录（真实 .b2nd/.pkl）→ PASS + 报告落盘。"""
    exit_code, output, dataset605, dataset606 = _run_cli_on_disk(tmp_path, monkeypatch)

    assert exit_code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == audit.STATUS_PASS
    assert report["n_cases_605"] == len(CASE_IDS)
    assert report["n_cases_606"] == len(CASE_IDS)
    assert report["n_cases_checked"] == len(CASE_IDS)
    assert report["n_cases_mismatch"] == 0
    assert report["inputs_modified"] is False
    for name in audit.MRI_CHANNEL_NAMES:
        assert report["per_channel"][name]["exact_equal_cases"] == len(CASE_IDS)
    # 真实读取路径走的是 nnU-Net 自己的 dataset 类，报告须记录实际目录
    assert report["case_folder_605"] == str(dataset605 / "nnUNetPlans_3d_fullres")
    assert report["case_folder_606"] == str(dataset606 / "nnUNetPlans_3d_fullres")
    assert report["splits"]["fold_0"]["train_set_equal"] is True


def test_cli_end_to_end_fail_still_writes_report(tmp_path: Path, monkeypatch) -> None:
    """端到端：单个 voxel 不同 → FAIL、退出码 2，报告仍写出并含 mismatch 明细。"""
    cases_606 = _cases_606()
    data, seg = cases_606["case_c"]
    data = data.copy()
    data[1, 0, 0, 0] += 0.25
    cases_606["case_c"] = (data, seg)

    exit_code, output, _, _ = _run_cli_on_disk(
        tmp_path, monkeypatch, cases_606=cases_606
    )

    assert exit_code == 2
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == audit.STATUS_FAIL
    assert report["per_channel"]["ADC"]["mismatch_cases"] == 1
    detail = [item for item in report["mismatched_cases"] if item["channel"] == "ADC"]
    assert detail and detail[0]["case_id"] == "case_c"
    assert detail[0]["unequal_voxels"] == 1


def test_cli_end_to_end_rejects_non_object_dataset_json(
    tmp_path: Path, monkeypatch
) -> None:
    """dataset.json 顶层不是对象时必须 fail-closed（真实报错路径）。"""
    root = tmp_path / "nnUNet_preprocessed"
    dataset_dir = _write_preprocessed_dataset(root, "Dataset605_PICAI", _cases_605())
    _write_preprocessed_dataset(root, "Dataset606_PICAI_Zonal", _cases_606())
    (dataset_dir / "dataset.json").write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    (dataset_dir / "splits_final.json").write_text(
        json.dumps(_splits()), encoding="utf-8"
    )
    (root / "Dataset606_PICAI_Zonal" / "dataset.json").write_text(
        json.dumps(_dataset_json_606()), encoding="utf-8"
    )
    (root / "Dataset606_PICAI_Zonal" / "splits_final.json").write_text(
        json.dumps(_splits()), encoding="utf-8"
    )
    monkeypatch.setenv("nnUNet_preprocessed", str(root))
    with pytest.raises(audit.AuditError):
        audit.main(["--output", str(tmp_path / "report.json"), "--no-progress"])
