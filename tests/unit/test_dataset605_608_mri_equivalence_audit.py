"""Dataset605/608 一致性审计脚本的纯合成单元测试。

只使用 ``tmp_path`` 与内存中的合成数组（占位 NIfTI 由 SimpleITK 写成合成影像），
**不读取任何真实医学数据**，也不触碰 ``workdir/`` 与 ``outputs/``。

覆盖的语义核心（与 Dataset605/606 审计同一套判定，本文件同时做防漂移核对）：

- 两数据集完全相同时 raw 层 PASS（但 preprocessed 产物缺失时 status 只能是 RAW_ONLY_PASS）；
- 单个 MRI 体素不同 → FAIL，并记录 ``max |Δ|`` / affected voxels；
- split 不同、有效标签不同（0↔1）、几何（size/spacing/origin/direction）不同、
  缺病例、prior 契约非法 → 一律 FAIL；
- 纯 ``-1↔0`` 原始 seg 差异 → 信息性，不影响 PASS；
- 608 的 prior 通道（第 4-6 通道）不参与前三 MRI 通道判等，只做信息性统计；
- ``--max-cases`` 子集永不判 PASS；报告已存在时 CLI 拒绝覆盖。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUDIT_SCRIPT = PROJECT_ROOT / "scripts" / "data" / "audit_dataset605_608_equivalence.py"
SIBLING_SCRIPT = PROJECT_ROOT / "scripts" / "data" / "audit_dataset605_606_mri_equivalence.py"


def _load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = _load_script(AUDIT_SCRIPT, "audit_dataset605_608")
sibling = _load_script(SIBLING_SCRIPT, "audit_dataset605_606")

DATASET_605 = audit.DEFAULT_DATASET_605
DATASET_608 = audit.DEFAULT_DATASET_608
CASE_TRAIN = "1_10"
CASE_VAL = "2_20"
CASE_IDS = (CASE_TRAIN, CASE_VAL)

# --------------------------------------------------------------------- 合成数据构造


def _mri(seed: int, shape=(2, 2, 2)) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=shape).astype(np.float32)


def _prior(seed: int, shape=(2, 2, 2)) -> np.ndarray:
    return np.random.default_rng(seed).random(size=shape).astype(np.float32)


def _label(shape=(2, 2, 2)) -> np.ndarray:
    label = np.zeros(shape, dtype=np.uint8)
    label[0, 0, 0] = 1
    return label


def _label_int8(shape=(2, 2, 2), *, minus_one_positions=((1, 1, 1),)) -> np.ndarray:
    label = np.zeros(shape, dtype=np.int8)
    for position in minus_one_positions:
        label[position] = -1
    label[0, 0, 0] = 1
    return label


def _cases_605(*, label_factory=None, mri_factory=None) -> dict:
    label_factory = label_factory or (lambda _seed: _label())
    mri_factory = mri_factory or (lambda seed: _mri(seed))
    return {
        cid: {"channels": [mri_factory(seed) for _ in range(3)], "label": label_factory(seed)}
        for seed, cid in enumerate(CASE_IDS, start=1)
    }


def _cases_608(base=None, *, prior_value=None) -> dict:
    base = _cases_605() if base is None else base
    out = {}
    for index, (cid, entry) in enumerate(base.items()):
        priors = [
            np.full(entry["channels"][0].shape, prior_value, np.float32)
            if prior_value is not None
            else _prior(50 + index * 3 + channel)
            for channel in range(3)
        ]
        out[cid] = {"channels": [*entry["channels"], *priors],
                    "label": entry["label"], "spacing": entry.get("spacing")}
    return out


def _dataset_json_605() -> dict:
    return {
        "channel_names": {"0000": "T2W", "0001": "ADC", "0002": "HBV"},
        "labels": {"background": 0, "lesion": 1},
        "file_ending": ".nii.gz",
        "numTraining": len(CASE_IDS),
    }


def _dataset_json_608(*, drop_train_record=False, channel_order=None) -> dict:
    trained = ["1"]

    def record(cid, mode):
        return {
            "case_id": cid,
            "prior_mode": mode,
            "checkpoint": "synthetic_stage1.pth",
            "trainer": "nnUNetTrainerPICAI_AnatomyJoint_100ep_NoFFT",
            "source_split": "fold_0",
            "geometry": {"synthetic": True},
            "anatomy_training_patient_ids": trained,
        }

    cases = {CASE_TRAIN: record(CASE_TRAIN, "IN_SAMPLE_PRED"),
             CASE_VAL: record(CASE_VAL, "HELD_OUT_PRED")}
    if drop_train_record:
        cases.pop(CASE_TRAIN)
    return {
        "channel_names": {
            "0000": "T2W", "0001": "ADC", "0002": "HBV",
            "0003": "noNorm", "0004": "noNorm", "0005": "noNorm",
        },
        "labels": {"background": 0, "lesion": 1},
        "file_ending": ".nii.gz",
        "numTraining": 2,
        "predicted_anatomy_contract": {
            "channel_order": channel_order or ["WG", "PZ", "TZ"],
            "source": "predicted_prior",
            "train_cases": [CASE_TRAIN],
            "val_cases": [CASE_VAL],
            "cases": cases,
            "source_lesion_split": "synthetic",
            "source_anatomy_model": "synthetic",
        },
    }


def _splits() -> list[dict]:
    return [{"train": [CASE_TRAIN], "val": [CASE_VAL]}]


def _write_raw_dataset(
    root: Path,
    name: str,
    cases: dict,
    *,
    dataset_json: dict,
    splits: list,
    spacing=(3.0, 0.5, 0.5),
    origin=(0.0, 0.0, 0.0),
) -> Path:
    """把合成病例写成真实 nnU-Net raw 布局（imagesTr / labelsTr / dataset.json / splits）。"""
    import SimpleITK as sitk

    folder = root / name
    (folder / "imagesTr").mkdir(parents=True, exist_ok=True)
    (folder / "labelsTr").mkdir(parents=True, exist_ok=True)
    for case_identifier, entry in cases.items():
        case_spacing = entry.get("spacing") or spacing
        arrays = [*entry["channels"], entry["label"]]
        for index, array in enumerate(arrays):
            image = sitk.GetImageFromArray(np.asarray(array))
            image.SetSpacing(case_spacing)
            image.SetOrigin(origin)
            if index == len(arrays) - 1:
                target = folder / "labelsTr" / f"{case_identifier}.nii.gz"
            else:
                target = folder / "imagesTr" / f"{case_identifier}_{index:04d}.nii.gz"
            sitk.WriteImage(image, str(target))
    (folder / "dataset.json").write_text(json.dumps(dataset_json), encoding="utf-8")
    (folder / "splits_final.json").write_text(json.dumps(splits), encoding="utf-8")
    return folder


def _write_both_raw(tmp_path: Path, *, cases_605=None, cases_608=None, splits_608=None,
                    dataset_json_608=None, spacing_608=(3.0, 0.5, 0.5)):
    cases_605 = _cases_605() if cases_605 is None else cases_605
    cases_608 = _cases_608(cases_605) if cases_608 is None else cases_608
    folder_605 = _write_raw_dataset(
        tmp_path, DATASET_605, cases_605, dataset_json=_dataset_json_605(), splits=_splits()
    )
    folder_608 = _write_raw_dataset(
        tmp_path, DATASET_608, cases_608,
        dataset_json=_dataset_json_608() if dataset_json_608 is None else dataset_json_608,
        splits=_splits() if splits_608 is None else splits_608, spacing=spacing_608,
    )
    return folder_605, folder_608


def _audit_existing(tmp_path: Path, *, include_preprocessed=False, max_cases=None,
                    preprocessed=None):
    """对 ``tmp_path`` 上**已经写好**的合成数据集跑审计（不重建任何文件）。"""
    folder_605, folder_608 = tmp_path / DATASET_605, tmp_path / DATASET_608
    return audit.run_audit(
        raw_605=audit.RawCaseSource(folder_605, expected_channels=3),
        raw_608=audit.RawCaseSource(folder_608, expected_channels=6),
        splits_605=json.loads((folder_605 / "splits_final.json").read_text()),
        splits_608=json.loads((folder_608 / "splits_final.json").read_text()),
        dataset_json_605=json.loads((folder_605 / "dataset.json").read_text()),
        dataset_json_608=json.loads((folder_608 / "dataset.json").read_text()),
        dataset_name_605=DATASET_605,
        dataset_name_608=DATASET_608,
        fold=0,
        progress=False,
        include_preprocessed=include_preprocessed,
        max_cases=max_cases,
        **(preprocessed or {}),
    )


def _run(tmp_path: Path, *, include_preprocessed=False, max_cases=None, **kwargs):
    _write_both_raw(tmp_path, **kwargs)
    return _audit_existing(
        tmp_path, include_preprocessed=include_preprocessed, max_cases=max_cases
    )


# --------------------------------------------------------------------- PASS 路径


def test_identical_raw_datasets_are_equivalent_but_raw_only_without_preprocessed(tmp_path):
    """raw 层完全一致 + 608 尚未预处理 ⇒ RAW_ONLY_PASS（不是 PASS，退出码 2）。"""
    report = _run(tmp_path, include_preprocessed=True)
    assert report["status"] == audit.STATUS_RAW_ONLY
    assert report["case_set_equal"] and report["case_order_equal"] and report["split_equal"]
    assert report["metadata_ok"] is True
    assert report["prior_contract_status"] == "VALID"
    assert report["n_cases_mri_mismatch"] == 0
    assert report["n_cases_label_mismatch"] == 0
    assert report["levels"]["preprocessed"] is None
    assert report["preprocessed_checked"] is False
    raw = report["levels"]["raw"]
    assert raw["status"] == audit.LEVEL_PASS
    assert raw["n_cases_checked"] == len(CASE_IDS)
    assert raw["n_cases_raw_identical"] == len(CASE_IDS)
    assert raw["mismatched_cases"] == []
    assert raw["informational_differences"] == []
    for name in audit.MRI_CHANNEL_NAMES:
        channel = raw["per_channel"][name]
        assert channel["exact_equal_cases"] == len(CASE_IDS)
        assert channel["mismatch_cases"] == 0
        assert channel["global_max_abs_diff"] == 0.0
    assert report["problems"] == []
    assert any("预处理产物尚不存在" in limitation for limitation in report["limitations"])
    assert report["inputs_modified"] is False


def test_prior_channels_are_excluded_from_mri_equivalence_and_reported_informationally(tmp_path):
    """608 的 prior 通道完全不同（含越界值）也不影响前三 MRI 通道判等。"""
    cases_605 = _cases_605()
    cases_608 = _cases_608(cases_605, prior_value=7.5)
    report = _run(tmp_path, cases_605=cases_605, cases_608=cases_608)
    assert report["status"] == audit.STATUS_RAW_ONLY
    assert report["n_cases_mri_mismatch"] == 0
    assert report["n_cases_label_mismatch"] == 0
    assert report["n_cases_prior_outside_unit_interval_info_only"] == len(CASE_IDS)
    raw = report["levels"]["raw"]
    assert raw["per_channel"]["T2W"]["mismatch_cases"] == 0
    prior = raw["prior_channels_info_only"][0]
    assert prior["any_outside_unit_interval"] is True
    assert prior["channels"][0]["min"] == pytest.approx(7.5)


def test_minus_one_versus_zero_is_informational_at_raw_level(tmp_path):
    """原始 seg 差异但有效标签相同（纯 -1↔0）⇒ 信息性，不影响一致性判定。"""
    cases_605 = _cases_605(label_factory=lambda _seed: _label_int8())
    cases_608 = _cases_608(cases_605)
    for entry in cases_608.values():
        entry["label"] = entry["label"].copy()
        entry["label"][entry["label"] == -1] = 0  # 608 把 -1 记为 0（映射后有效标签相同）
    report = _run(tmp_path, cases_605=cases_605, cases_608=cases_608)
    assert report["status"] == audit.STATUS_RAW_ONLY
    raw = report["levels"]["raw"]
    assert raw["n_cases_informational_raw_label_difference"] == len(CASE_IDS)
    assert raw["n_cases_label_mismatch"] == 0
    assert raw["informational_differences"][0]["raw_diff_pairs"] == {"-1→0": 1}
    assert all(item["kind"] != "raw_seg_difference_effectively_equal"
               for item in raw["mismatched_cases"])


# --------------------------------------------------------------------- FAIL 路径


def test_single_mri_voxel_difference_fails_with_detail(tmp_path):
    cases_605 = _cases_605()
    cases_608 = _cases_608(cases_605)
    cases_608[CASE_VAL]["channels"][1] = cases_608[CASE_VAL]["channels"][1].copy()
    cases_608[CASE_VAL]["channels"][1][0, 0, 0] += 1e-3  # 单个 ADC voxel 不同
    report = _run(tmp_path, cases_605=cases_605, cases_608=cases_608)

    assert report["status"] == audit.STATUS_FAIL
    assert report["n_cases_mri_mismatch"] == 1
    raw = report["levels"]["raw"]
    assert raw["per_channel"]["ADC"]["mismatch_cases"] == 1
    assert raw["per_channel"]["ADC"]["global_max_abs_diff"] == pytest.approx(1e-3, rel=1e-3)
    assert raw["per_channel"]["T2W"]["mismatch_cases"] == 0
    detail = [item for item in raw["mismatched_cases"] if item["channel"] == "ADC"]
    assert len(detail) == 1
    assert detail[0]["case_id"] == CASE_VAL
    assert detail[0]["kind"] == "mri_values_differ"
    assert detail[0]["unequal_voxels"] == 1
    assert detail[0]["max_abs_diff"] == pytest.approx(1e-3, rel=1e-3)
    assert detail[0]["fraction_unequal"] == pytest.approx(1 / 8)


def test_split_mismatch_fails(tmp_path):
    splits_608 = [{"train": [CASE_VAL], "val": [CASE_TRAIN]}]
    report = _run(tmp_path, splits_608=splits_608)
    assert report["status"] == audit.STATUS_FAIL
    assert report["split_equal"] is False
    assert report["splits"]["fold_0"]["train_set_equal"] is False
    assert any("split" in problem or "fold" in problem for problem in report["problems"])


def test_label_mismatch_fails(tmp_path):
    cases_605 = _cases_605()
    cases_608 = _cases_608(cases_605)
    cases_608[CASE_TRAIN]["label"] = cases_608[CASE_TRAIN]["label"].copy()
    cases_608[CASE_TRAIN]["label"][1, 1, 1] = 1  # 605 该位置是 0，608 是 1
    report = _run(tmp_path, cases_605=cases_605, cases_608=cases_608)
    assert report["status"] == audit.STATUS_FAIL
    assert report["n_cases_label_mismatch"] == 1
    raw = report["levels"]["raw"]
    detail = [item for item in raw["mismatched_cases"] if item["channel"] == "SEG"]
    assert detail and detail[0]["kind"] == "effective_label_values_differ"
    assert detail[0]["raw_diff_pairs"] == {"0→1": 1}


def test_unexpected_label_value_fails(tmp_path):
    cases_605 = _cases_605()
    cases_608 = _cases_608(cases_605)
    cases_608[CASE_TRAIN]["label"] = cases_608[CASE_TRAIN]["label"].copy()
    cases_608[CASE_TRAIN]["label"][0, 0, 1] = 2
    report = _run(tmp_path, cases_605=cases_605, cases_608=cases_608)
    assert report["status"] == audit.STATUS_FAIL
    detail = [item for item in report["levels"]["raw"]["mismatched_cases"]
              if item["kind"] == "unexpected_label_values"]
    assert detail and detail[0]["unexpected_values_608"] == [2]
    assert detail[0]["allowed_raw_labels"] == [-1, 0, 1]


def test_spacing_mismatch_fails_with_geometry_kind(tmp_path):
    cases_605 = _cases_605()
    report = _run(tmp_path, cases_605=cases_605, spacing_608=(4.0, 0.5, 0.5))
    assert report["status"] == audit.STATUS_FAIL
    raw = report["levels"]["raw"]
    kinds = {item["kind"] for item in raw["mismatched_cases"]}
    assert "geometry_spacing_equal" in kinds
    detail = next(item for item in raw["mismatched_cases"]
                  if item["kind"] == "geometry_spacing_equal")
    assert detail["geometry_605"]["spacing"] == [3.0, 0.5, 0.5]
    assert detail["geometry_608"]["spacing"] == [4.0, 0.5, 0.5]
    assert detail["differences"]["max_abs_spacing_delta"] == pytest.approx(1.0)
    label_geometry = {item["kind"] for item in raw["mismatched_cases"]
                      if item["channel"] == "SEG"}
    assert "label_geometry_spacing_equal" in label_geometry


def test_origin_mismatch_fails(tmp_path):
    """608 的 origin 被改动（例如手工重写 prior 通道）⇒ FAIL。"""
    cases_605 = _cases_605()
    cases_608 = _cases_608(cases_605)
    _write_both_raw(tmp_path, cases_605=cases_605, cases_608=cases_608)
    import SimpleITK as sitk

    target = tmp_path / DATASET_608 / "imagesTr" / f"{CASE_VAL}_0001.nii.gz"
    image = sitk.ReadImage(str(target))
    image.SetOrigin((5.0, 0.0, 0.0))
    sitk.WriteImage(image, str(target))
    report = _audit_existing(tmp_path)
    assert report["status"] == audit.STATUS_FAIL
    assert "geometry_origin_equal" in {
        item["kind"] for item in report["levels"]["raw"]["mismatched_cases"]
    }


def test_spatial_shape_mismatch_fails(tmp_path):
    cases_605 = _cases_605()
    cases_608 = _cases_608(cases_605)
    for entry in cases_608.values():
        entry["channels"] = [channel[:1] for channel in entry["channels"]]
        entry["label"] = entry["label"][:1]
    report = _run(tmp_path, cases_605=cases_605, cases_608=cases_608)
    assert report["status"] == audit.STATUS_FAIL
    kinds = {item["kind"] for item in report["levels"]["raw"]["mismatched_cases"]}
    assert "shape_or_dtype_mismatch" in kinds
    assert "geometry_size_equal" in kinds


def test_missing_case_fails(tmp_path):
    cases_605 = _cases_605()
    cases_608 = _cases_608(cases_605)
    cases_608.pop(CASE_VAL)
    report = _run(tmp_path, cases_605=cases_605, cases_608=cases_608)
    assert report["status"] == audit.STATUS_FAIL
    assert report["case_set_equal"] is False
    missing = [item for item in report["levels"]["raw"]["mismatched_cases"]
               if item["kind"] == "case_missing_in_other_dataset"]
    assert len(missing) == 1 and missing[0]["present_in"] == "605"


def test_missing_channel_file_fails(tmp_path):
    cases_605 = _cases_605()
    cases_608 = _cases_608(cases_605)
    _write_both_raw(tmp_path, cases_605=cases_605, cases_608=cases_608)
    (tmp_path / DATASET_608 / "imagesTr" / f"{CASE_VAL}_0002.nii.gz").unlink()
    report = _audit_existing(tmp_path)
    assert report["status"] == audit.STATUS_FAIL
    failures = report["levels"]["raw"]["load_failures"]
    assert failures and failures[0]["kind"] == "missing_file"
    assert failures[0]["files"] == [f"imagesTr/{CASE_VAL}_0002.nii.gz"]


def test_invalid_prior_contract_fails(tmp_path):
    report = _run(tmp_path, dataset_json_608=_dataset_json_608(drop_train_record=True))
    assert report["status"] == audit.STATUS_FAIL
    assert report["prior_contract_status"].startswith("INVALID")
    assert any("prior 契约非法" in problem for problem in report["problems"])


def test_max_cases_subset_never_passes(tmp_path):
    report = _run(tmp_path, include_preprocessed=True, max_cases=1)
    assert report["status"] == audit.STATUS_PARTIAL
    assert report["problems"] == []
    assert report["levels"]["raw"]["n_cases_checked"] == 1
    assert any("--max-cases" in limitation for limitation in report["limitations"])


def test_max_cases_subset_still_reports_a_real_mismatch_as_failure(tmp_path):
    """子集检查发现不一致时必须报 FAIL（不能因 --max-cases 降级成 PARTIAL 而掩盖问题）。"""
    cases_605 = _cases_605()
    cases_608 = _cases_608(cases_605)
    cases_608[CASE_TRAIN]["channels"][0] = cases_608[CASE_TRAIN]["channels"][0].copy()
    cases_608[CASE_TRAIN]["channels"][0][0, 0, 0] += 1.0
    report = _run(tmp_path, cases_605=cases_605, cases_608=cases_608, max_cases=1)
    assert report["status"] == audit.STATUS_FAIL
    assert report["n_cases_mri_mismatch"] == 1
    assert any("--max-cases" in limitation for limitation in report["limitations"])


# --------------------------------------------------------- preprocessed 层与语义防漂移


def _write_preprocessed_dataset(root: Path, name: str, cases: dict, *, channels: int) -> Path:
    """用 nnU-Net 自己的 ``save_case`` 写真实格式的 ``.b2nd`` / ``_seg.b2nd``。"""
    from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2

    dataset_dir = root / name
    case_folder = dataset_dir / "nnUNetPlans_3d_fullres"
    case_folder.mkdir(parents=True, exist_ok=True)
    for case_identifier, entry in cases.items():
        data = np.stack(entry["channels"][:channels]).astype(np.float32)
        seg = np.asarray(entry["label"])[None].astype(np.int8)
        nnUNetDatasetBlosc2.save_case(
            data, seg, {"case_identifier": case_identifier}, str(case_folder / case_identifier)
        )
    return dataset_dir


def _preprocessed_sources(tmp_path: Path, cases_605: dict, cases_608: dict):
    root = tmp_path / "preprocessed"
    return {
        "preprocessed_605": audit.PreprocessedCaseSource(
            _write_preprocessed_dataset(root, DATASET_605, cases_605, channels=3), "3d_fullres"
        ),
        "preprocessed_608": audit.PreprocessedCaseSource(
            _write_preprocessed_dataset(root, DATASET_608, cases_608, channels=6), "3d_fullres"
        ),
    }


def test_both_levels_identical_is_full_pass(tmp_path):
    cases_605 = _cases_605()
    cases_608 = _cases_608(cases_605)
    _write_both_raw(tmp_path, cases_605=cases_605, cases_608=cases_608)
    report = _audit_existing(
        tmp_path, include_preprocessed=True,
        preprocessed=_preprocessed_sources(tmp_path, cases_605, cases_608),
    )
    assert report["status"] == audit.STATUS_PASS
    assert report["problems"] == []
    assert report["preprocessed_available"] is True
    assert report["preprocessed_checked"] is True
    preprocessed = report["levels"]["preprocessed"]
    assert preprocessed["status"] == audit.LEVEL_PASS
    assert preprocessed["n_cases_checked"] == len(CASE_IDS)
    assert all(channel["exact_equal_cases"] == len(CASE_IDS)
               for channel in preprocessed["per_channel"].values())
    assert preprocessed["prior_channels_info_only"] == []  # prior 只在 raw 层统计


def test_preprocessed_mri_mismatch_fails(tmp_path):
    cases_605 = _cases_605()
    cases_608 = _cases_608(cases_605)
    _write_both_raw(tmp_path, cases_605=cases_605, cases_608=cases_608)
    broken = {cid: {"channels": [channel.copy() for channel in entry["channels"]],
                    "label": entry["label"]}
              for cid, entry in cases_608.items()}
    broken[CASE_TRAIN]["channels"][0] = broken[CASE_TRAIN]["channels"][0].copy()
    broken[CASE_TRAIN]["channels"][0][0, 0, 0] += 0.25
    report = _audit_existing(
        tmp_path, include_preprocessed=True,
        preprocessed=_preprocessed_sources(tmp_path, cases_605, broken),
    )
    assert report["status"] == audit.STATUS_FAIL
    preprocessed = report["levels"]["preprocessed"]
    assert preprocessed["per_channel"]["T2W"]["mismatch_cases"] == 1
    assert report["levels"]["raw"]["status"] == audit.LEVEL_PASS


def test_preprocessed_seg_label_mismatch_fails(tmp_path):
    cases_605 = _cases_605()
    cases_608 = _cases_608(cases_605)
    _write_both_raw(tmp_path, cases_605=cases_605, cases_608=cases_608)
    broken = {cid: {"channels": entry["channels"], "label": np.asarray(entry["label"]).copy()}
              for cid, entry in cases_608.items()}
    broken[CASE_VAL]["label"][1, 1, 1] = 1
    report = _audit_existing(
        tmp_path, include_preprocessed=True,
        preprocessed=_preprocessed_sources(tmp_path, cases_605, broken),
    )
    assert report["status"] == audit.STATUS_FAIL
    assert report["levels"]["preprocessed"]["n_cases_label_mismatch"] == 1


def test_effective_label_semantics_do_not_drift_from_sibling_audit():
    """本工具与 605/606 审计必须使用同一套「有效标签」语义（防止两处定义漂移）。"""
    assert audit.ALLOWED_RAW_SEG_LABELS == sibling.ALLOWED_RAW_SEG_LABELS == (-1, 0, 1)
    assert audit.EFFECTIVE_LABEL_REMAP == sibling.EFFECTIVE_LABEL_REMAP == {-1: 0}
    sample = np.array([[-1, 0], [1, 2]], dtype=np.int64)
    assert np.array_equal(audit._effective_labels(sample), sibling._effective_labels(sample))


def test_inputs_are_not_modified(tmp_path):
    cases_605 = _cases_605()
    cases_608 = _cases_608(cases_605)
    _write_both_raw(tmp_path, cases_605=cases_605, cases_608=cases_608)
    preprocessed = _preprocessed_sources(tmp_path, cases_605, cases_608)
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    report = _audit_existing(tmp_path, include_preprocessed=True, preprocessed=preprocessed)
    assert report["inputs_modified"] is False
    after = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    assert before == after
    assert report["status"] == audit.STATUS_PASS


def test_report_is_json_serializable(tmp_path):
    report = _run(tmp_path, dataset_json_608=_dataset_json_608(drop_train_record=True))
    payload = json.loads(json.dumps(report, ensure_ascii=False))
    for key in ("status", "problems", "case_set_equal", "split_equal", "metadata_ok",
                "prior_contract_status", "levels", "notes", "inputs_modified",
                "n_cases_mri_mismatch", "n_cases_label_mismatch"):
        assert key in payload, key
    assert payload["levels"]["raw"]["per_channel"]["T2W"]["exact_equal_cases"] == len(CASE_IDS)


# --------------------------------------------------------------------- CLI


def test_cli_requires_nnunet_raw_env(tmp_path, monkeypatch):
    monkeypatch.delenv("nnUNet_raw", raising=False)
    with pytest.raises(audit.AuditError):
        audit.main(["--output", str(tmp_path / "report.json")])


def test_cli_refuses_to_overwrite_existing_report(tmp_path, monkeypatch):
    monkeypatch.setenv("nnUNet_raw", str(tmp_path))
    existing = tmp_path / "report.json"
    existing.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit) as excinfo:
        audit.main(["--output", str(existing)])
    assert "拒绝覆盖" in str(excinfo.value)


def test_cli_parser_defaults():
    args = audit.build_parser().parse_args([])
    assert args.dataset_605 == "Dataset605_PICAI"
    assert args.dataset_608 == "Dataset608_PICAI_PredictedAnatomy"
    assert args.configuration == "3d_fullres"
    assert args.max_cases is None
    assert args.overwrite is False
    assert args.skip_preprocessed is False
    assert args.output == "outputs/reports/dataset605_608_equivalence_audit.json"


def _cli_fixture(tmp_path: Path, monkeypatch, *, both_levels: bool):
    cases_605 = _cases_605()
    cases_608 = _cases_608(cases_605)
    _write_both_raw(tmp_path, cases_605=cases_605, cases_608=cases_608)
    monkeypatch.setenv("nnUNet_raw", str(tmp_path))
    if both_levels:
        root = tmp_path / "preprocessed_root"
        monkeypatch.setenv("nnUNet_preprocessed", str(root))
        audit.PreprocessedCaseSource(
            _write_preprocessed_dataset(root, DATASET_605, cases_605, channels=3), "3d_fullres"
        )
        audit.PreprocessedCaseSource(
            _write_preprocessed_dataset(root, DATASET_608, cases_608, channels=6), "3d_fullres"
        )
    else:
        # 环境已配置但 608 尚未预处理：空 root 是合法状态，status 只能是 RAW_ONLY_PASS
        empty_root = tmp_path / "empty_preprocessed_root"
        empty_root.mkdir()
        monkeypatch.setenv("nnUNet_preprocessed", str(empty_root))
    return tmp_path / "report.json"


def test_cli_end_to_end_full_pass_on_both_levels(tmp_path, monkeypatch):
    output = _cli_fixture(tmp_path, monkeypatch, both_levels=True)
    exit_code = audit.main(["--output", str(output), "--no-progress"])
    report = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["status"] == audit.STATUS_PASS
    assert report["problems"] == []
    assert report["preprocessed_checked"] is True


def test_cli_end_to_end_raw_only_pass_when_preprocessed_missing(tmp_path, monkeypatch):
    output = _cli_fixture(tmp_path, monkeypatch, both_levels=False)
    exit_code = audit.main(["--output", str(output), "--no-progress"])
    report = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 2
    assert report["status"] == audit.STATUS_RAW_ONLY
    assert report["preprocessed_checked"] is False
    assert report["levels"]["raw"]["status"] == audit.LEVEL_PASS


def test_cli_end_to_end_skip_preprocessed_is_raw_only(tmp_path, monkeypatch):
    output = _cli_fixture(tmp_path, monkeypatch, both_levels=True)
    exit_code = audit.main(["--output", str(output), "--no-progress", "--skip-preprocessed"])
    report = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 2
    assert report["status"] == audit.STATUS_RAW_ONLY
    assert report["skip_preprocessed_requested"] is True
    assert report["preprocessed_skip_reason"] == "--skip-preprocessed"


def test_cli_end_to_end_fail_still_writes_report(tmp_path, monkeypatch):
    cases_605 = _cases_605()
    cases_608 = _cases_608(cases_605)
    cases_608[CASE_VAL]["channels"][2] = cases_608[CASE_VAL]["channels"][2].copy()
    cases_608[CASE_VAL]["channels"][2][0, 1, 0] += 0.5  # HBV 单个体素不同
    _write_both_raw(cases_605=cases_605, cases_608=cases_608, tmp_path=tmp_path)
    monkeypatch.setenv("nnUNet_raw", str(tmp_path))
    output = tmp_path / "report.json"
    exit_code = audit.main(["--output", str(output), "--no-progress", "--skip-preprocessed"])
    report = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 2
    assert report["status"] == audit.STATUS_FAIL
    assert report["n_cases_mri_mismatch"] == 1
    detail = [item for item in report["levels"]["raw"]["mismatched_cases"]
              if item["channel"] == "HBV"]
    assert detail and detail[0]["case_id"] == CASE_VAL
