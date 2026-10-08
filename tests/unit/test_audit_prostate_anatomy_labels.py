"""``scripts/data/audit_prostate_anatomy_labels.py`` 的纯合成单元测试。

只使用 ``tmp_path`` 与磁盘上的合成 NIfTI，**不读取任何真实医学数据**，
不触碰 ``data/``、``workdir/``、``outputs/``。

覆盖六个语义核心：

1. **标签契约（禁止整数截断）**：浮点存储但精确等于 0/1/2 合法；
   ``0.5`` / ``1.5`` / ``2.7`` / 负值 / NaN / ±Inf 必须被检查出来并逐例记录，
   且非法取值**绝不**经 ``arr > 0`` 进入 ROI。NaN 不得让整份报告丢失。
2. **总体状态覆盖所有被请求任务**：ROI 的 lesion 缺失/不可读/非法/几何错误、
   wg/zonal/prostate158 的失败都进入问题汇总；``inputs`` 恒被执行；
   未请求的 stage 明确标为未评估；已知缺失与未认可状态分开；``--max-cases`` 不作全量声明。
3. **分母完整性**：合法空病灶 ≠ 参考标签不可读；不可用时实例数为 ``null``、
   ``denominator_complete`` 为假、「全部参考病灶」口径置 ``null``；
   腺体来源不可用但 lesion 合法时仍算真实覆盖。
4. **统一 T2W 几何基准**：lesion 必须与 T2W 同网格；来源与 lesion 一致但与 T2W 不一致时
   两者都不可评估，且不得把 lesion FOV 称为完整 T2W FOV。
5. **保守 WG 来源判定**：竞争候选未完成比较时不得断言唯一。
6. **候选区域编码**：三种掩膜层级（原始头阈值 / 导出整数图 / 训练参考区域）必须分开核验。
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUDIT_SCRIPT = PROJECT_ROOT / "scripts" / "data" / "audit_prostate_anatomy_labels.py"
EVALUATOR_SCRIPT = PROJECT_ROOT / "scripts" / "evaluate_segmentation.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = _load_module("audit_prostate_anatomy_labels", AUDIT_SCRIPT)

SHAPE_ZYX = (4, 8, 8)
SPACING_XYZ = (0.5, 0.5, 3.0)  # → spacing_zyx = (3.0, 0.5, 0.5)
ORIGIN_XYZ = (0.0, 0.0, 0.0)


# --------------------------------------------------------------------------- 合成影像


def _write_nifti(
    path: Path,
    array_zyx: np.ndarray,
    *,
    spacing_xyz: tuple[float, float, float] = SPACING_XYZ,
    origin_xyz: tuple[float, float, float] = ORIGIN_XYZ,
) -> Path:
    import SimpleITK as sitk

    path.parent.mkdir(parents=True, exist_ok=True)
    image = sitk.GetImageFromArray(np.asarray(array_zyx))
    image.SetSpacing(tuple(float(s) for s in spacing_xyz))
    image.SetOrigin(tuple(float(o) for o in origin_xyz))
    sitk.WriteImage(image, str(path))
    return path


def _box(shape_zyx, zs, ys, xs, *, dtype=np.uint8) -> np.ndarray:
    array = np.zeros(shape_zyx, dtype=dtype)
    array[zs[0]:zs[1], ys[0]:ys[1], xs[0]:xs[1]] = 1
    return array


def _wg_default(shape=SHAPE_ZYX) -> np.ndarray:
    return _box(shape, (0, 3), (1, 7), (1, 7))


def _pz_default(shape=SHAPE_ZYX) -> np.ndarray:
    return _box(shape, (0, 3), (1, 4), (1, 7))


def _tz_default(shape=SHAPE_ZYX) -> np.ndarray:
    return _box(shape, (0, 3), (4, 7), (1, 7))


def _zonal_from(pz: np.ndarray, tz: np.ndarray, *, tz_label=2) -> np.ndarray:
    out = np.zeros(pz.shape, dtype=np.uint8)
    out[pz > 0] = 1
    out[tz > 0] = tz_label
    return out


def _rotated_hevi(shape=SHAPE_ZYX) -> np.ndarray:
    return _zonal_from(
        _box(shape, (0, 3), (1, 3), (1, 7)), _box(shape, (0, 3), (3, 7), (1, 7))
    )


def _lesion_two_instances(shape=SHAPE_ZYX) -> np.ndarray:
    """两个互不相连的 3D 6-邻域实例（一个在 PZ 侧、一个在 TZ 侧）。"""
    lesion = np.zeros(shape, dtype=np.uint8)
    lesion[1, 1:3, 1:3] = 1
    lesion[2, 5:7, 5:7] = 1
    return lesion


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return path


# --------------------------------------------------------------------------- 输入树


def _defaults(patient_id: str) -> dict:
    return {
        "split": "train",
        "patient_id": patient_id,
        "study_id": "s1",
        "t2w": _box(SHAPE_ZYX, (0, 4), (0, 8), (0, 8)),
        "lesion": _lesion_two_instances(),
        "wg": _wg_default(),
        "pz": _pz_default(),
        "tz": _tz_default(),
        "hevi": None,
        "bosma": None,
        "guerbet": None,
        "wg_status": "exact",
        "drop": set(),
        "corrupt": set(),
        "spacing_xyz": SPACING_XYZ,
        "t2w_spacing_xyz": None,
        "lesion_spacing_xyz": None,
        "wg_spacing_xyz": None,
        "zonal_spacing_xyz": None,
        "bosma_spacing_xyz": None,
        "guerbet_spacing_xyz": None,
        "bosma_ndim2d": False,
        "lesion_label_values": None,
        "wg_label_values": None,
        "yuan_label_values": None,
        "hevi_label_values": None,
        "case_id_override": None,
    }


class Tree:
    """一套合成输入树（manifest / split / materialization report / cases / labels / p158）。"""

    def __init__(self, root: Path):
        self.root = root
        self.manifest = root / "picai_manifest.csv"
        self.split_cases = root / "picai_nnunet_split_cases.csv"
        self.materialization_report = root / "materialization_report.csv"
        self.cases_root = root / "processed" / "picai" / "cases"
        self.labels_root = root / "PI-CAI" / "labels"
        self.prostate158_root = root / "Prostate158"
        self.case_ids: tuple[str, ...] = ()

    def case_dir(self, case_id: str) -> Path:
        return self.cases_root / case_id

    def argv(self, output: Path, *extra: str) -> list[str]:
        return [
            "--manifest",
            str(self.manifest),
            "--split-cases",
            str(self.split_cases),
            "--materialization-report",
            str(self.materialization_report),
            "--cases-root",
            str(self.cases_root),
            "--picai-labels-root",
            str(self.labels_root),
            "--prostate158-root",
            str(self.prostate158_root),
            "--output",
            str(output),
            "--no-progress",
            *extra,
        ]

    def all_input_files(self) -> list[Path]:
        return sorted(p for p in self.root.rglob("*") if p.is_file())


def _build_tree(root: Path, cases: list[dict]) -> Tree:
    tree = Tree(root)
    tree.cases_root.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict] = []
    split_rows: list[dict] = []
    report_rows: list[dict] = []

    for spec in cases:
        case_id = (
            spec["case_id_override"]
            if spec.get("case_id_override")
            else f"{spec['patient_id']}_{spec['study_id']}"
        )
        case_dir = tree.case_dir(case_id)
        case_dir.mkdir(parents=True, exist_ok=True)

        def _spacing(key, _spec=spec):
            value = _spec.get(key)
            return value if value is not None else _spec["spacing_xyz"]

        yuan_array = (
            spec["yuan_label_values"]
            if spec.get("yuan_label_values") is not None
            else _zonal_from(spec["pz"], spec["tz"])
        )
        hevi_array = (
            spec["hevi_label_values"]
            if spec.get("hevi_label_values") is not None
            else (
                spec["hevi"]
                if spec.get("hevi") is not None
                else _zonal_from(spec["pz"], spec["tz"])
            )
        )
        wg_array = (
            spec["wg_label_values"]
            if spec.get("wg_label_values") is not None
            else spec["wg"]
        )
        lesion_array = (
            spec["lesion_label_values"]
            if spec.get("lesion_label_values") is not None
            else spec["lesion"]
        )

        targets = {
            "t2w": (case_dir / "t2w.nii.gz", spec["t2w"], _spacing("t2w_spacing_xyz")),
            "lesion": (
                case_dir / "lesion.nii.gz",
                lesion_array,
                _spacing("lesion_spacing_xyz"),
            ),
            "wg": (case_dir / "wg.nii.gz", wg_array, _spacing("wg_spacing_xyz")),
            "yuan": (
                case_dir / "zonal_yuan.nii.gz",
                yuan_array,
                _spacing("zonal_spacing_xyz"),
            ),
            "hevi": (
                case_dir / "zonal_hevi.nii.gz",
                hevi_array,
                _spacing("zonal_spacing_xyz"),
            ),
        }
        dropped = spec.get("drop", set())
        corrupted = spec.get("corrupt", set())
        for key, (path, array, spacing) in targets.items():
            if key in dropped:
                continue
            if key in corrupted:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"not a nifti")
                continue
            if array is None:  # pragma: no cover
                continue
            _write_nifti(path, array, spacing_xyz=spacing)

        source_dir = tree.labels_root / "anatomical_delineations" / "whole_gland" / "AI"
        for source, key, spacing_key in (
            ("Bosma22b", "bosma", "bosma_spacing_xyz"),
            ("Guerbet23", "guerbet", "guerbet_spacing_xyz"),
        ):
            if key in dropped:
                continue
            path = source_dir / source / f"{case_id}.nii.gz"
            if key in corrupted:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"not a nifti")
                continue
            array = spec.get(key)
            if array is None:
                array = spec["wg"]
            if key == "bosma" and spec.get("bosma_ndim2d"):
                array = np.asarray(array)[0]  # 2D → 维度不合法
            _write_nifti(path, array, spacing_xyz=_spacing(spacing_key))

        zonal_dir = tree.labels_root / "anatomical_delineations" / "zonal_pz_tz" / "AI"
        for source, key in (("Yuan23", "yuan"), ("HeviAI23", "hevi")):
            if key in dropped:
                continue
            path = zonal_dir / source / f"{case_id}.nii.gz"
            if key in corrupted:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"not a nifti")
                continue
            array, spacing = targets[key][1], targets[key][2]
            if array is None:  # pragma: no cover
                continue
            _write_nifti(path, array, spacing_xyz=spacing)

        manifest_rows.append(
            {
                "case_id": case_id,
                "patient_id": spec["patient_id"],
                "study_id": spec["study_id"],
                "center": "TEST",
                "case_csPCa": spec.get("case_csPCa", "NO"),
            }
        )
        split_rows.append(
            {
                "case_id": case_id,
                "study_id": spec["study_id"],
                "patient_id": spec["patient_id"],
                "center": "TEST",
                "case_csPCa": spec.get("case_csPCa", "NO"),
                "split": spec["split"],
            }
        )
        report_rows.append(
            {
                "case_id": case_id,
                "patient_id": spec["patient_id"],
                "study_id": spec["study_id"],
                "status": "ok",
                "wg_status": spec.get("wg_status", "exact"),
            }
        )

    _write_csv(tree.manifest, list(manifest_rows[0].keys()), manifest_rows)
    _write_csv(tree.split_cases, list(split_rows[0].keys()), split_rows)
    _write_csv(tree.materialization_report, list(report_rows[0].keys()), report_rows)
    tree.case_ids = tuple(row["case_id"] for row in manifest_rows)
    return tree


def _make_cases(labels: list[str], **overrides) -> list[dict]:
    """``labels`` 当作 patient_id；studies 为 ``s1``/``s2``… → case_id = ``<label>_s<i>``。"""
    out = []
    for index, label in enumerate(labels):
        spec = _defaults(label)
        spec["study_id"] = f"s{index + 1}"
        spec.update(overrides)
        spec["patient_id"] = label
        out.append(spec)
    return out


def _run(tree: Tree, output: Path, *extra: str) -> int:
    return audit.main(tree.argv(output, *extra))


def _run_report(tree: Tree, output: Path, *extra: str) -> dict:
    _run(tree, output, *extra)
    assert output.is_file(), "报告必须被写出"
    return json.loads(output.read_text(encoding="utf-8"))


# ===========================================================================
# 1. 标签契约：禁止整数截断
# ===========================================================================


def test_float_stored_exact_zero_one_two_is_legal():
    """浮点存储但取值**精确**为 0/1/2 → 合法。"""
    array = _zonal_from(_pz_default(), _tz_default()).astype(np.float32)
    info = audit.inspect_label_array(array, audit.LABEL_CONTRACT_ZONAL)
    assert info["status"] == audit.LABEL_STATUS_OK
    assert info["observed_values"] == ["0", "1", "2"]
    assert info["observed_values_are_integer"] is True
    assert info["n_non_integer_voxels"] == 0
    assert info["n_illegal_voxels"] == 0
    assert audit.label_is_usable(info) is True


def test_value_repr_keeps_fractional_values_verbatim():
    assert audit._value_repr(0.0) == "0"
    assert audit._value_repr(2.0) == "2"
    assert audit._value_repr(-1.0) == "-1"
    assert audit._value_repr(1.5) == "1.5"
    assert audit._value_repr(0.5) == "0.5"
    assert audit._value_repr(np.nan) == "nan"
    assert audit._value_repr(np.inf) == "inf"
    assert audit._value_repr(-np.inf) == "-inf"


@pytest.mark.parametrize("value", [0.5, 1.5, 2.7, -1.0])
def test_non_integer_and_negative_values_are_flagged_without_truncation(value):
    """``1.5`` 必须被读成 ``1.5`` 而不是 ``1``；非整数与负值都非法。

    用只含背景 0 与注入值的数组，使「截断后的整数值是否出现」成为可判定的断言。
    """
    array = np.zeros(SHAPE_ZYX, dtype=np.float32)
    array[0, 0, 0] = value
    info = audit.inspect_label_array(array, audit.LABEL_CONTRACT_WG)
    observed = [float(v) for v in info["observed_values"]]
    injected = float(np.float32(value))
    assert any(abs(v - injected) < 1e-6 for v in observed), "必须保留实际观测值"
    assert info["status"] == audit.LABEL_STATUS_ILLEGAL_VALUES
    assert info["n_illegal_voxels"] >= 1
    assert audit.label_is_usable(info) is False
    if injected != float(int(injected)):
        # 非整数注入值必须被识别为非整数，且以**带小数点的文本**原样报告
        assert info["observed_values_are_integer"] is False
        assert info["n_non_integer_voxels"] == 1
        assert any(
            "." in text and abs(float(text) - injected) < 1e-6
            for text in info["observed_values"]
        ), "禁止把分数值截断成整数值"
    # 关键：非法值绝不进入允许前景
    mask = audit.label_foreground_mask(array, audit.LABEL_CONTRACT_WG)
    assert bool(mask[0, 0, 0]) is False


@pytest.mark.parametrize(
    "value,expected_repr",
    [(np.nan, "nan"), (np.inf, "inf"), (-np.inf, "-inf")],
)
def test_non_finite_values_are_flagged(value, expected_repr):
    array = _zonal_from(_pz_default(), _tz_default()).astype(np.float32)
    array[3, 7, 7] = value
    info = audit.inspect_label_array(array, audit.LABEL_CONTRACT_ZONAL)
    assert info["status"] == audit.LABEL_STATUS_NON_FINITE
    assert info["n_non_finite"] >= 1
    assert expected_repr not in info["observed_values"], "非有限值不进入取值集合"
    assert audit.label_is_usable(info) is False
    mask = audit.label_foreground_mask(array, audit.LABEL_CONTRACT_ZONAL)
    assert bool(mask[3, 7, 7]) is False


def test_wrong_ndim_is_flagged():
    info = audit.inspect_label_array(np.zeros((4, 4), dtype=np.uint8), audit.LABEL_CONTRACT_WG)
    assert info["status"] == audit.LABEL_STATUS_WRONG_NDIM
    assert info["ndim"] == 2
    assert audit.label_is_usable(info) is False


def test_empty_label_is_legal_and_usable():
    """合法空标签是真实阴性，**可用**且与「不可读」不同。"""
    info = audit.inspect_label_array(np.zeros(SHAPE_ZYX, dtype=np.uint8), audit.LABEL_CONTRACT_LESION)
    assert info["status"] == audit.LABEL_STATUS_EMPTY
    assert info["is_empty"] is True
    assert audit.label_is_usable(info) is True


def test_label_contracts_differ_by_allowed_values():
    assert audit.LABEL_CONTRACTS[audit.LABEL_CONTRACT_WG]["allowed_values"] == (0.0, 1.0)
    assert audit.LABEL_CONTRACTS[audit.LABEL_CONTRACT_LESION]["allowed_values"] == (0.0, 1.0)
    assert audit.LABEL_CONTRACTS[audit.LABEL_CONTRACT_ZONAL]["allowed_values"] == (
        0.0,
        1.0,
        2.0,
    )
    # Prostate158 的整数映射未核实 → 允许集合未知，不做取值声明
    p158 = audit.LABEL_CONTRACTS[audit.LABEL_CONTRACT_PROSTATE158_ANATOMY]
    assert p158["allowed_values"] is None
    info = audit.inspect_label_array(
        np.array([[[0, 5]]], dtype=np.int16), audit.LABEL_CONTRACT_PROSTATE158_ANATOMY
    )
    assert info["allowed_values"] is None
    assert info["allowed_values_unconstrained"] is True
    assert info["n_illegal_voxels"] == 0, "允许集合未知时不得声称取值非法"


def test_zonal_label_with_illegal_value_cannot_build_roi(tmp_path):
    """非法 zonal 取值不得经 ``arr > 0`` 进入 ROI 构造。"""
    cases = _make_cases(["c1"])
    illegal = _zonal_from(_pz_default(), _tz_default()).astype(np.float32)
    illegal[0, 0, 1] = 2.7
    cases[0]["yuan_label_values"] = illegal
    tree = _build_tree(tmp_path, cases)
    report = _run_report(tree, tmp_path / "r.json", "--stages", "roi")
    entry = report["stages"]["roi"]["per_case"]["yuan_pz_tz_union"][0]
    diagnostics = entry["source_diagnostics"]
    assert diagnostics["status"] == audit.ROI_SOURCE_LABEL_NOT_USABLE
    assert "2.7" in json.dumps(diagnostics["label_info"]["observed_values"])
    assert entry["fallback"] is True, "非法来源 → 回退，而不是用它构造 ROI"
    assert entry["fallback_fov"] == "full_t2w_fov"


def test_illegal_lesion_label_makes_reference_unavailable_not_zero(tmp_path):
    cases = _make_cases(["c1"])
    bad = _lesion_two_instances().astype(np.float32)
    bad[0, 0, 0] = 1.5
    cases[0]["lesion_label_values"] = bad
    tree = _build_tree(tmp_path, cases)
    report = _run_report(tree, tmp_path / "r.json", "--stages", "roi")
    entry = report["stages"]["roi"]["per_case"]["wg_materialized"][0]
    assert entry["roi_status"] == "illegal_lesion_values"
    assert entry["reference_state"] == audit.REFERENCE_UNAVAILABLE
    assert entry["n_reference_instances"] is None
    assert "1.5" in json.dumps(entry["lesion_unique_values"])


def test_nan_in_lesion_does_not_lose_the_report(tmp_path, monkeypatch):
    """NaN 只影响涉事病例的结论，报告仍完整写出（JSON 合法）。

    注意：SimpleITK 的 NIfTI 往返会把 NaN/±Inf 写成 0（已实测），
    因此这里在**读取层**注入 NaN，而不是指望磁盘保留它。
    """
    cases = _make_cases(["c1", "c2"])
    tree = _build_tree(tmp_path, cases)
    target = tree.case_dir("c1_s1") / "lesion.nii.gz"
    real_read_array = audit.read_array

    def patched_read_array(path):
        array = real_read_array(path)
        if Path(path) == target:
            array = np.asarray(array, dtype=np.float32)
            array[0, 0, 0] = np.nan
        return array

    monkeypatch.setattr(audit, "read_array", patched_read_array)
    report = _run_report(tree, tmp_path / "r.json", "--stages", "roi")

    assert report["schema"] == "prostate_anatomy_labels_audit"
    entries = report["stages"]["roi"]["per_case"]["wg_materialized"]
    assert len(entries) == 2
    nan_entry = [e for e in entries if e["case_id"] == "c1_s1"][0]
    ok_entry = [e for e in entries if e["case_id"] == "c2_s2"][0]
    assert nan_entry["reference_state"] == audit.REFERENCE_UNAVAILABLE
    assert nan_entry["n_reference_instances"] is None
    assert nan_entry["lesion_label_info"]["n_nan"] == 1
    assert ok_entry["reference_state"] == audit.REFERENCE_EVALUABLE_NONEMPTY
    # 报告仍是合法 JSON（NaN 被写为显式文本，而不是产出非法 JSON）
    assert "nan" in json.dumps(report)


def test_infinities_in_label_are_flagged_without_truncation():
    """±Inf 与 NaN 一样：逐例报错、不进入前景、不丢报告。"""
    for value in (np.inf, -np.inf):
        array = np.zeros(SHAPE_ZYX, dtype=np.float32)
        array[0, 0, 0] = 1.0
        array[1, 0, 0] = value
        info = audit.inspect_label_array(array, audit.LABEL_CONTRACT_LESION)
        assert info["status"] == audit.LABEL_STATUS_NON_FINITE
        assert info["n_non_finite"] == 1
        assert audit.label_is_usable(info) is False
        mask = audit.label_foreground_mask(array, audit.LABEL_CONTRACT_LESION)
        assert bool(mask[0, 0, 0]) is True
        assert bool(mask[1, 0, 0]) is False


def test_illegal_wg_label_is_rejected_as_comparison_baseline(tmp_path):
    cases = _make_cases(["c1"])
    bad = _wg_default().astype(np.float32)
    bad[0, 0, 0] = 3.0
    cases[0]["wg_label_values"] = bad
    tree = _build_tree(tmp_path, cases)
    report = _run_report(tree, tmp_path / "r.json", "--stages", "wg")
    entry = report["stages"]["wg"]["per_case"][0]
    assert entry["outcome"] == audit.WG_READ_ERROR
    assert entry["materialized_usable"] is False
    assert "3" in json.dumps(entry["materialized"]["label_info"]["observed_values"])


# ===========================================================================
# 2. WG 内容对应（保守判定）
# ===========================================================================


def test_wg_both_sources_match_is_source_ambiguity_not_failure(tmp_path):
    tree = _build_tree(tmp_path, _make_cases(["c1", "c2"]))
    report = _run_report(tree, tmp_path / "r.json", "--stages", "inputs", "wg")
    counts = report["stages"]["wg"]["counts"]
    assert counts[audit.WG_BOTH] == 2
    assert report["stages"]["wg"]["n_ambiguous"] == 2
    assert report["status"]["data_valid"] is True
    assert report["status"]["source_resolved"] is False
    assert report["status"]["source_audit_status"] == "unresolved"
    assert report["status"]["execution_complete"] is True


def test_wg_unique_source_only_when_competitor_completed_and_mismatched(tmp_path):
    cases = _make_cases(["c1"])
    cases[0]["guerbet"] = np.roll(_wg_default(), shift=1, axis=2)
    tree = _build_tree(tmp_path, cases)
    report = _run_report(tree, tmp_path / "r.json", "--stages", "inputs", "wg")
    entry = report["stages"]["wg"]["per_case"][0]
    assert entry["outcome"] == audit.WG_ONLY_BOSMA
    assert entry["unique_source_unresolved"] is False
    assert entry["match_summary"] == {
        "matched_candidates": ["bosma22b"],
        "mismatched_candidates": ["guerbet23"],
        "not_comparable_candidates": [],
        "all_candidates_completed_comparison": True,
    }
    assert report["status"]["source_resolved"] is True


def test_wg_missing_competitor_is_not_treated_as_mismatch(tmp_path):
    """竞争候选缺失 → 只能写「与某候选内容一致」，唯一性未解决。"""
    cases = _make_cases(["c1"])
    cases[0]["drop"] = {"guerbet"}
    tree = _build_tree(tmp_path, cases)
    report = _run_report(tree, tmp_path / "r.json", "--stages", "inputs", "wg")
    entry = report["stages"]["wg"]["per_case"][0]
    assert entry["outcome"] == audit.WG_MATCH_UNRESOLVED
    assert entry["unique_source_unresolved"] is True
    assert entry["match_summary"]["not_comparable_candidates"] == ["guerbet23"]
    assert entry["candidates"]["guerbet23"]["read_status"] == "missing"
    assert entry["candidates"]["guerbet23"]["match_result"] == audit.CANDIDATE_NOT_COMPARABLE
    assert entry["candidates"]["guerbet23"]["voxel_equal"] is None
    assert report["status"]["source_resolved"] is False


def test_wg_unreadable_competitor_is_not_treated_as_mismatch(tmp_path):
    cases = _make_cases(["c1"])
    cases[0]["corrupt"] = {"guerbet"}
    tree = _build_tree(tmp_path, cases)
    report = _run_report(tree, tmp_path / "r.json", "--stages", "inputs", "wg")
    entry = report["stages"]["wg"]["per_case"][0]
    assert entry["outcome"] == audit.WG_MATCH_UNRESOLVED
    assert entry["candidates"]["guerbet23"]["read_status"] == "read_error"


def test_wg_competitor_on_different_grid_is_not_treated_as_mismatch(tmp_path):
    cases = _make_cases(["c1"])
    cases[0]["guerbet_spacing_xyz"] = (0.5, 0.5, 6.0)
    tree = _build_tree(tmp_path, cases)
    report = _run_report(tree, tmp_path / "r.json", "--stages", "inputs", "wg")
    entry = report["stages"]["wg"]["per_case"][0]
    assert entry["outcome"] == audit.WG_MATCH_UNRESOLVED
    assert entry["candidates"]["guerbet23"]["grid_comparable"] is False
    assert any(
        "不自动重采样" in reason
        for reason in entry["candidates"]["guerbet23"]["not_comparable_reasons"]
    )
    assert report["status"]["source_resolved"] is False


def test_wg_competitor_with_invalid_geometry_is_not_comparable(tmp_path):
    cases = _make_cases(["c1"])
    cases[0]["bosma_ndim2d"] = True
    tree = _build_tree(tmp_path, cases)
    report = _run_report(tree, tmp_path / "r.json", "--stages", "inputs", "wg")
    entry = report["stages"]["wg"]["per_case"][0]
    assert entry["candidates"]["bosma22b"]["geometry_problems"]
    assert entry["candidates"]["bosma22b"]["match_result"] == audit.CANDIDATE_NOT_COMPARABLE
    assert entry["outcome"] == audit.WG_MATCH_UNRESOLVED


def test_wg_no_comparable_candidate(tmp_path):
    cases = _make_cases(["c1"])
    cases[0]["drop"] = {"bosma", "guerbet"}
    tree = _build_tree(tmp_path, cases)
    report = _run_report(tree, tmp_path / "r.json", "--stages", "inputs", "wg")
    entry = report["stages"]["wg"]["per_case"][0]
    assert entry["outcome"] == audit.WG_NO_COMPARABLE_CANDIDATE
    assert entry["unique_source_unresolved"] is True


def test_wg_neither_match_requires_both_candidates_compared(tmp_path):
    cases = _make_cases(["c1"])
    cases[0]["bosma"] = np.roll(_wg_default(), shift=1, axis=1)
    cases[0]["guerbet"] = np.roll(_wg_default(), shift=1, axis=2)
    tree = _build_tree(tmp_path, cases)
    report = _run_report(tree, tmp_path / "r.json", "--stages", "inputs", "wg")
    entry = report["stages"]["wg"]["per_case"][0]
    assert entry["outcome"] == audit.WG_NEITHER
    assert entry["match_summary"]["all_candidates_completed_comparison"] is True
    assert report["status"]["source_resolved"] is False


def test_wg_missing_materialized_file(tmp_path):
    cases = _make_cases(["c1"])
    cases[0]["drop"] = {"wg"}
    tree = _build_tree(tmp_path, cases)
    report = _run_report(tree, tmp_path / "r.json", "--stages", "inputs", "wg")
    assert report["stages"]["wg"]["counts"][audit.WG_MISSING] == 1
    assert report["stages"]["wg"]["n_source_resolved"] == 0


def test_wg_stage_declares_size_is_not_evidence_and_conservative_rule(tmp_path):
    tree = _build_tree(tmp_path, _make_cases(["c1"]))
    report = _run_report(tree, tmp_path / "r.json", "--stages", "inputs", "wg")
    stage = report["stages"]["wg"]
    assert "不使用文件大小" in stage["size_is_not_evidence_note"]
    assert "不能证明历史生成来源" in stage["size_is_not_evidence_note"]
    assert "not_comparable" in stage["conservative_rule_note"]


def test_wg_candidate_status_counts_are_reported(tmp_path):
    cases = _make_cases(["c1", "c2"])
    cases[1]["drop"] = {"guerbet"}
    tree = _build_tree(tmp_path, cases)
    report = _run_report(tree, tmp_path / "r.json", "--stages", "inputs", "wg")
    counts = report["stages"]["wg"]["candidate_status_counts"]
    assert counts["guerbet23"]["missing/not_comparable"] == 1
    assert counts["bosma22b"]["ok/match"] == 2


# ===========================================================================
# 3. 总体状态覆盖所有被请求任务
# ===========================================================================


def test_only_roi_stage_still_runs_forced_inputs_integrity(tmp_path):
    """即使只请求 roi，inputs 也必须执行，数据完整性检查不得被跳过。"""
    cases = _make_cases(["c1"])
    cases[0]["drop"] = {"lesion"}
    tree = _build_tree(tmp_path, cases)
    report = _run_report(tree, tmp_path / "r.json", "--stages", "roi")
    assert "inputs" in report["stages"], "inputs 是强制审计项"
    assert report["stages"]["inputs"]["n_cases_unexpected_missing"] == 1
    assert report["status"]["data_valid"] is False
    assert any("意外" in p for p in report["status"]["problems"])


def test_not_requested_stage_is_marked_not_evaluated_and_blocks_resolved(tmp_path):
    tree = _build_tree(tmp_path, _make_cases(["c1"]))
    report = _run_report(tree, tmp_path / "r.json", "--stages", "roi")
    status = report["status"]
    assert status["source_audit_status"] == "not_evaluated"
    assert status["source_resolved"] is None, "没检查不能当作已解决"
    assert "wg" in status["stages_not_evaluated"]
    assert "source_resolution" in status["stages_not_evaluated"]
    assert "不构成" in status["not_evaluated_note"]


def test_requested_wg_stage_is_evaluated(tmp_path):
    cases = _make_cases(["c1"])
    cases[0]["guerbet"] = np.roll(_wg_default(), shift=1, axis=2)
    tree = _build_tree(tmp_path, cases)
    report = _run_report(tree, tmp_path / "r.json", "--stages", "wg")
    assert report["status"]["source_audit_status"] == "resolved"
    assert report["status"]["source_resolved"] is True
    assert "source_resolution" not in report["status"]["stages_not_evaluated"]


def test_roi_lesion_missing_enters_problems(tmp_path):
    cases = _make_cases(["c1"])
    cases[0]["drop"] = {"lesion"}
    tree = _build_tree(tmp_path, cases)
    report = _run_report(tree, tmp_path / "r.json", "--stages", "roi")
    problems = report["status"]["problems"]
    assert report["status"]["data_valid"] is False
    assert any("lesion_missing" in p for p in problems)
    assert any("分母**不完整**" in p for p in problems)


def test_roi_lesion_read_error_enters_problems(tmp_path):
    cases = _make_cases(["c1"])
    cases[0]["corrupt"] = {"lesion"}
    tree = _build_tree(tmp_path, cases)
    report = _run_report(tree, tmp_path / "r.json", "--stages", "roi")
    assert any(
        "lesion_read_error" in p for p in report["status"]["problems"]
    )


def test_roi_illegal_lesion_enters_problems(tmp_path):
    cases = _make_cases(["c1"])
    bad = _lesion_two_instances().astype(np.float32)
    bad[0, 0, 0] = 1.5
    cases[0]["lesion_label_values"] = bad
    tree = _build_tree(tmp_path, cases)
    report = _run_report(tree, tmp_path / "r.json", "--stages", "roi")
    assert any(
        "illegal_lesion_values" in p for p in report["status"]["problems"]
    )


def test_roi_t2w_geometry_error_enters_problems(tmp_path):
    cases = _make_cases(["c1"])
    cases[0]["drop"] = {"t2w"}
    tree = _build_tree(tmp_path, cases)
    report = _run_report(tree, tmp_path / "r.json", "--stages", "roi")
    problems = " ".join(report["status"]["problems"])
    assert "t2w_missing" in problems
    assert report["status"]["data_valid"] is False


def test_zonal_failures_enter_problems(tmp_path):
    cases = _make_cases(["c1"])
    cases[0]["corrupt"] = {"yuan"}  # 物化 zonal 不可读
    tree = _build_tree(tmp_path, cases)
    report = _run_report(tree, tmp_path / "r.json", "--stages", "zonal")
    problems = " ".join(report["status"]["problems"])
    assert "zonal/yuan" in problems
    assert report["status"]["data_valid"] is False


def test_zonal_illegal_label_enters_problems(tmp_path):
    cases = _make_cases(["c1"])
    illegal = _zonal_from(_pz_default(), _tz_default()).astype(np.float32)
    illegal[0, 0, 1] = 2.7
    cases[0]["yuan_label_values"] = illegal
    tree = _build_tree(tmp_path, cases)
    report = _run_report(tree, tmp_path / "r.json", "--stages", "zonal")
    problems = " ".join(report["status"]["problems"])
    assert "未通过契约检查" in problems
    assert report["status"]["data_valid"] is False


def test_prostate158_failures_enter_problems(tmp_path):
    tree = _build_tree(tmp_path, _make_cases(["c1"]))
    _build_prostate158(tree.prostate158_root)
    (
        tree.prostate158_root
        / "test"
        / "prostate158_test"
        / "test"
        / "001"
        / "t2_anatomy_reader1.nii.gz"
    ).unlink()
    report = _run_report(tree, tmp_path / "r.json", "--stages", "prostate158")
    problems = " ".join(report["status"]["problems"])
    assert "Prostate158" in problems and "缺失" in problems
    assert report["status"]["data_valid"] is False


def test_wg_label_unusable_enters_problems(tmp_path):
    cases = _make_cases(["c1"])
    cases[0]["corrupt"] = {"wg"}
    tree = _build_tree(tmp_path, cases)
    report = _run_report(tree, tmp_path / "r.json", "--stages", "wg")
    assert any("wg stage" in p for p in report["status"]["problems"])


def test_recognized_known_missing_vs_unrecognized_wg_status(tmp_path):
    cases = _make_cases(["c1", "c2"])
    cases[0]["drop"] = {"wg"}
    cases[0]["wg_status"] = "excluded_known_faulty_bosma22b"
    cases[1]["drop"] = {"wg"}
    cases[1]["wg_status"] = "some_other_unknown_status"
    tree = _build_tree(tmp_path, cases)
    report = _run_report(tree, tmp_path / "r.json", "--stages", "inputs")
    stage = report["stages"]["inputs"]
    assert stage["n_cases_wg_missing_recognized_known"] == 1
    assert stage["n_cases_wg_status_unrecognized"] == 1
    assert stage["recognized_known_missing_wg_cases"] == ["c1_s1"]
    assert stage["unrecognized_wg_status_cases"] == ["c2_s2"]
    # 未被认可的状态不能被自动当成允许的已知缺失
    assert any(
        "未被认可为已知缺失" in p for p in report["status"]["problems"]
    )


def test_max_cases_subset_makes_no_full_dataset_claim(tmp_path):
    tree = _build_tree(tmp_path, _make_cases(["c1", "c2"]))
    output = tmp_path / "r.json"
    exit_code = _run(tree, output, "--max-cases", "1")
    report = json.loads(output.read_text(encoding="utf-8"))
    status = report["status"]
    assert report["scope"] == "subset"
    assert status["scope"] == "subset"
    assert status["execution_complete"] is False
    assert status["data_valid"] is None
    assert status["source_resolved"] is None
    assert status["subset_note"]
    assert exit_code == audit.EXIT_UNRESOLVED


def test_status_has_three_independent_booleans_and_no_all_pass(tmp_path):
    cases = _make_cases(["c1"])
    cases[0]["bosma"] = np.roll(_wg_default(), shift=1, axis=1)
    cases[0]["guerbet"] = np.roll(_wg_default(), shift=1, axis=2)
    tree = _build_tree(tmp_path, cases)
    report = _run_report(tree, tmp_path / "r.json", "--stages", "wg")
    status = report["status"]
    assert status["execution_complete"] is True
    assert status["data_valid"] is True
    assert status["source_resolved"] is False
    booleans = {k for k, v in status.items() if isinstance(v, bool)}
    assert booleans == {"execution_complete", "data_valid", "source_resolved"}


# ===========================================================================
# 4. 病灶分母完整性
# ===========================================================================


def test_read_failure_does_not_shrink_denominator_and_keeps_instance_unknown(tmp_path):
    """读取失败的 lesion 实例数**未知**（null），不是 0，也不是阴性病例。"""
    cases = _make_cases(["c1", "c2"])
    cases[1]["corrupt"] = {"lesion"}
    tree = _build_tree(tmp_path, cases)
    report = _run_report(tree, tmp_path / "r.json", "--stages", "roi")
    entries = report["stages"]["roi"]["per_case"]["wg_materialized"]
    assert len(entries) == 2, "读取失败的病例必须留在逐例结果里"
    failed = [e for e in entries if e["case_id"] == "c2_s2"][0]
    assert failed["roi_status"] == "lesion_read_error"
    assert failed["reference_state"] == audit.REFERENCE_UNAVAILABLE
    assert failed["n_reference_instances"] is None
    assert failed["voxel_coverage"] is None
    assert "未知" in failed["reference_unavailable_note"]

    overall = report["stages"]["roi"]["per_candidate_summary"]["wg_materialized"]["overall"]
    assert overall["n_cases"] == 2
    assert overall["n_cases_reference_evaluable"] == 1
    assert overall["n_cases_reference_unavailable"] == 1
    assert overall["denominator_complete"] is False
    assert overall["roi_status_counts"]["lesion_read_error"] == 1


def test_one_ok_one_unreadable_cannot_yield_full_coverage_claim(tmp_path):
    """关键场景：一例成功、一例 lesion 不可读时，不得产生代表全部病例的 100% 覆盖。"""
    cases = _make_cases(["c1", "c2"])
    cases[1]["corrupt"] = {"lesion"}
    tree = _build_tree(tmp_path, cases)
    report = _run_report(
        tree, tmp_path / "r.json", "--stages", "roi", "--roi-margin-mm", "5"
    )
    overall = report["stages"]["roi"]["per_candidate_summary"]["wg_materialized"]["overall"]
    # 可评估那一例确实 100% 覆盖，但它**不能**被写成全部参考病灶的结果
    for key in audit.ROI_ALL_REFERENCE_FIELDS:
        assert overall[key] is None, f"{key} 在分母不完整时必须为 null"
    assert overall["n_reference_instances_observed"] == 2
    assert overall["reference_scope"] == "all_cases_denominator_incomplete"
    subset = overall["evaluable_subset_only"]
    assert subset["n_cases"] == 1
    assert subset["overall_voxel_coverage"] == 1.0
    assert subset["n_reference_instances"] == 2
    assert "不代表全部参考病灶" in subset["scope"]
    assert "evaluable_subset_only" in overall["denominator_note"]
    assert "未知" in overall["denominator_note"]


def test_complete_denominator_reports_all_reference_scope(tmp_path):
    tree = _build_tree(tmp_path, _make_cases(["c1", "c2"]))
    report = _run_report(
        tree, tmp_path / "r.json", "--stages", "roi", "--roi-margin-mm", "5"
    )
    overall = report["stages"]["roi"]["per_candidate_summary"]["wg_materialized"]["overall"]
    assert overall["denominator_complete"] is True
    assert overall["reference_scope"] == "all_cases"
    assert overall["n_reference_instances"] == 4
    assert overall["n_reference_instances_observed"] == 4
    assert overall["overall_voxel_coverage"] == 1.0
    assert overall["n_cases_reference_unavailable"] == 0


def test_legal_empty_lesion_is_evaluable_with_zero_instances(tmp_path):
    """合法空病灶标签 = 真实阴性：可评估、实例数 0，**不是**未知。"""
    cases = _make_cases(["c1", "c2"])
    cases[0]["lesion"] = np.zeros(SHAPE_ZYX, dtype=np.uint8)
    tree = _build_tree(tmp_path, cases)
    report = _run_report(tree, tmp_path / "r.json", "--stages", "roi")
    entries = report["stages"]["roi"]["per_case"]["wg_materialized"]
    empty_entry = [e for e in entries if e["case_id"] == "c1_s1"][0]
    assert empty_entry["reference_state"] == audit.REFERENCE_EVALUABLE_EMPTY
    assert empty_entry["reference_evaluable"] is True
    assert empty_entry["n_reference_instances"] == 0
    overall = report["stages"]["roi"]["per_candidate_summary"]["wg_materialized"]["overall"]
    assert overall["denominator_complete"] is True
    assert overall["n_cases_reference_evaluable"] == 2
    assert overall["n_reference_instances"] == 2


def test_instance_counts_are_measured_from_reference_masks(tmp_path):
    cases = _make_cases(["c1"])
    lesion = np.zeros(SHAPE_ZYX, dtype=np.uint8)
    lesion[0, 0, 0] = 1
    lesion[1, 1, 1] = 1
    lesion[2, 2, 2] = 1
    cases[0]["lesion"] = lesion
    tree = _build_tree(tmp_path, cases)
    report = _run_report(tree, tmp_path / "r.json", "--stages", "roi")
    entry = report["stages"]["roi"]["per_case"]["wg_materialized"][0]
    assert entry["n_reference_instances"] == 3, "实例数由实际掩膜的 6-邻域连通域得出"
    assert entry["instance_coverages"] == [1.0, 1.0, 1.0]
    stage = report["stages"]["roi"]
    assert stage["per_candidate_summary"]["wg_materialized"]["overall"][
        "instance_counts_measured_from_reference_masks"
    ] is True
    assert "不存在任何预先固定" in stage["per_candidate_summary"]["wg_materialized"][
        "overall"
    ]["instance_count_note"]


def test_gland_unavailable_but_lesion_legal_still_computes_coverage(tmp_path):
    """腺体标签缺失 → 回退完整 T2W FOV，但仍计算**真实**病灶覆盖。"""
    cases = _make_cases(["c1"])
    cases[0]["drop"] = {"wg"}
    tree = _build_tree(tmp_path, cases)
    report = _run_report(tree, tmp_path / "r.json", "--stages", "roi")
    entry = report["stages"]["roi"]["per_case"]["wg_materialized"][0]
    assert entry["fallback"] is True
    assert entry["fallback_fov"] == "full_t2w_fov"
    assert entry["reference_evaluable"] is True
    assert entry["n_reference_instances"] == 2, "仍由实际 lesion 掩膜测得"
    assert entry["voxel_coverage"] == 1.0
    overall = report["stages"]["roi"]["per_candidate_summary"]["wg_materialized"]["overall"]
    assert overall["denominator_complete"] is True
    assert overall["n_cases_fallback"] == 1
    assert overall["n_instances_in_fallback_cases"] == 2


def test_per_split_denominator_completeness_is_judged_separately(tmp_path):
    cases = _make_cases(["c1", "c2", "c3"])
    cases[2]["split"] = "validation"
    cases[2]["corrupt"] = {"lesion"}
    tree = _build_tree(tmp_path, cases)
    report = _run_report(tree, tmp_path / "r.json", "--stages", "roi")
    summary = report["stages"]["roi"]["per_candidate_summary"]["wg_materialized"]
    assert summary["per_split_denominator_complete"] == {
        "train": True,
        "validation": False,
    }
    assert summary["per_split"]["train"]["denominator_complete"] is True
    assert summary["per_split"]["train"]["n_reference_instances"] == 4
    assert summary["per_split"]["validation"]["denominator_complete"] is False
    assert summary["per_split"]["validation"]["n_reference_instances"] is None


# ===========================================================================
# 5. 统一 T2W 几何基准
# ===========================================================================


def test_lesion_not_on_t2w_grid_is_unavailable_and_not_called_full_fov(tmp_path):
    cases = _make_cases(["c1"])
    cases[0]["lesion_spacing_xyz"] = (0.5, 0.5, 6.0)
    tree = _build_tree(tmp_path, cases)
    report = _run_report(tree, tmp_path / "r.json", "--stages", "roi")
    entry = report["stages"]["roi"]["per_case"]["wg_materialized"][0]
    assert entry["roi_status"] == "lesion_grid_mismatch_vs_t2w"
    assert entry["lesion_on_t2w_grid"] is False
    assert entry["reference_state"] == audit.REFERENCE_UNAVAILABLE
    assert entry["n_reference_instances"] is None
    assert entry["lesion_vs_t2w_differences"]
    assert "不得" in entry["reference_unavailable_reason"]
    assert "完整 T2W FOV" in entry["reference_unavailable_reason"]


def test_source_and_lesion_consistent_but_both_differ_from_t2w(tmp_path):
    """来源与 lesion 一致、但两者都不对应 T2W → 不可评估（不得按数组 bbox 硬算）。"""
    cases = _make_cases(["c1"])
    cases[0]["lesion_spacing_xyz"] = (0.5, 0.5, 6.0)
    cases[0]["wg_spacing_xyz"] = (0.5, 0.5, 6.0)
    tree = _build_tree(tmp_path, cases)
    report = _run_report(tree, tmp_path / "r.json", "--stages", "roi")
    entry = report["stages"]["roi"]["per_case"]["wg_materialized"][0]
    assert entry["roi_status"] == "lesion_grid_mismatch_vs_t2w"
    assert entry["reference_evaluable"] is False
    assert entry["n_reference_instances"] is None
    assert entry["lesion_geometry"]["spacing_xyz"] == pytest.approx([0.5, 0.5, 6.0])
    assert entry["t2w_geometry"]["spacing_xyz"] == pytest.approx([0.5, 0.5, 3.0])
    # 参考不可用时不进入 ROI 来源分支：不构造任何 ROI，也不声明来源可用性
    assert "roi_bbox_zyx" not in entry
    assert "source_diagnostics" not in entry
    # 来源与 lesion 在磁盘上确实互相一致（都是 6.0），但都与 T2W 不一致
    wg_geometry = audit.read_geometry(tree.case_dir("c1_s1") / "wg.nii.gz")
    lesion_geometry = audit.read_geometry(tree.case_dir("c1_s1") / "lesion.nii.gz")
    t2w_geometry = audit.read_geometry(tree.case_dir("c1_s1") / "t2w.nii.gz")
    assert wg_geometry.same_grid_as(lesion_geometry) is True
    assert wg_geometry.same_grid_as(t2w_geometry) is False
    assert lesion_geometry.same_grid_as(t2w_geometry) is False


def test_source_on_lesion_grid_but_differing_from_t2w_falls_back_to_t2w_fov(tmp_path):
    """来源与 lesion 一致但与 T2W 不一致 → 回退完整 **T2W** FOV（非 lesion FOV）。"""
    cases = _make_cases(["c1"])
    cases[0]["zonal_spacing_xyz"] = (0.5, 0.5, 6.0)
    tree = _build_tree(tmp_path, cases)
    report = _run_report(tree, tmp_path / "r.json", "--stages", "roi")
    entry = report["stages"]["roi"]["per_case"]["yuan_pz_tz_union"][0]
    assert entry["fallback"] is True
    assert entry["fallback_fov"] == "full_t2w_fov"
    assert entry["roi_bbox_zyx"] == [[0, 4], [0, 8], [0, 8]]
    assert entry["n_reference_instances"] == 2, "T2W 对齐的 lesion 仍可评估"
    assert "T2W 网格不一致" in entry["fallback_reason"]


def test_t2w_missing_makes_reference_unavailable(tmp_path):
    cases = _make_cases(["c1"])
    cases[0]["drop"] = {"t2w"}
    tree = _build_tree(tmp_path, cases)
    report = _run_report(tree, tmp_path / "r.json", "--stages", "roi")
    entry = report["stages"]["roi"]["per_case"]["wg_materialized"][0]
    assert entry["roi_status"] == "t2w_missing"
    assert entry["t2w_baseline_usable"] is False
    assert entry["n_reference_instances"] is None


def test_geometry_validity_detects_invalid_spacing_and_direction():
    import SimpleITK as sitk

    def _geom(size, spacing, origin, direction):
        return audit.Geometry(
            Path("x.nii.gz"),
            size,
            spacing,
            origin,
            direction,
        )

    assert _geom((4, 4, 4), (1.0, 1.0, 1.0), (0, 0, 0), tuple(np.eye(3).ravel())).validity_problems() == []
    assert any(
        "非全正" in p
        for p in _geom((4, 4, 4), (0.0, 1.0, 1.0), (0, 0, 0), tuple(np.eye(3).ravel())).validity_problems()
    )
    assert any(
        "非有限" in p
        for p in _geom((4, 4, 4), (float("nan"), 1.0, 1.0), (0, 0, 0), tuple(np.eye(3).ravel())).validity_problems()
    )
    assert any(
        "2D" in p or "维度" in p
        for p in _geom((4, 4), (1.0, 1.0), (0, 0), (1.0, 0.0, 0.0, 1.0)).validity_problems()
    )
    assert any(
        "正交归一" in p
        for p in _geom(
            (4, 4, 4), (1.0, 1.0, 1.0), (0, 0, 0), (2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        ).validity_problems()
    )
    assert any(
        "origin" in p
        for p in _geom((4, 4, 4), (1.0, 1.0, 1.0), (float("inf"), 0, 0), tuple(np.eye(3).ravel())).validity_problems()
    )


def test_geometry_comparison_uses_rtol_zero():
    """相对容差必须为 0：大 spacing 上的小相对差异也要被判为不同网格。"""
    import SimpleITK as sitk

    identity = tuple(np.eye(3).ravel())
    big = audit.Geometry(Path("a.nii.gz"), (4, 4, 4), (1000.0, 1.0, 1.0), (0, 0, 0), identity)
    slightly = audit.Geometry(
        Path("b.nii.gz"), (4, 4, 4), (1000.5, 1.0, 1.0), (0, 0, 0), identity
    )
    assert audit.GEOMETRY_RTOL == 0.0
    assert big.same_grid_as(slightly) is False
    comparison = audit.compare_geometry(big, slightly)
    assert comparison["same_grid"] is False
    assert comparison["comparable"] is True
    assert comparison["differences"]


def test_lesion_grid_unknown_reported_when_t2w_unreadable(tmp_path):
    cases = _make_cases(["c1"])
    cases[0]["corrupt"] = {"t2w"}
    tree = _build_tree(tmp_path, cases)
    report = _run_report(tree, tmp_path / "r.json", "--stages", "roi")
    stage = report["stages"]["inputs"]
    assert stage["n_cases_read_error"] == 1
    assert stage["n_cases_t2w_geometry_unusable"] == 1
    # T2W 基准不可用 → lesion 与 T2W 的对齐性**未知**（不是「对齐」）
    assert stage["n_cases_lesion_grid_unknown"] == 1
    assert stage["n_cases_lesion_not_on_t2w_grid"] == 0
    assert report["status"]["data_valid"] is False
    entry = report["stages"]["roi"]["per_case"]["wg_materialized"][0]
    assert entry["t2w_baseline_usable"] is False
    assert entry["roi_status"] == "t2w_read_error"
    assert entry["n_reference_instances"] is None


# ===========================================================================
# 6. 分区关系
# ===========================================================================


def test_zonal_union_equals_wg(tmp_path):
    tree = _build_tree(tmp_path, _make_cases(["c1", "c2"]))
    report = _run_report(tree, tmp_path / "r.json", "--stages", "zonal")
    for entry in report["stages"]["zonal"]["per_case_relations"]:
        assert entry["relation"] == audit.REL_EQUAL
        assert entry["union_vs_wg_dice"] == 1.0
        assert entry["wg_outside_union_voxels"] == 0
        assert entry["union_outside_wg_voxels"] == 0


def test_zonal_union_subset_of_wg_records_both_directions(tmp_path):
    cases = _make_cases(["c1"])
    cases[0]["tz"] = _box(SHAPE_ZYX, (0, 3), (4, 6), (1, 7))
    tree = _build_tree(tmp_path, cases)
    report = _run_report(tree, tmp_path / "r.json", "--stages", "zonal")
    entry = [
        e for e in report["stages"]["zonal"]["per_case_relations"] if e["source"] == "yuan"
    ][0]
    assert entry["relation"] == audit.REL_UNION_SUBSET
    assert entry["union_outside_wg_voxels"] == 0
    assert entry["wg_outside_union_voxels"] == 3 * 1 * 6
    assert entry["wg_outside_union_fraction_of_wg"] == pytest.approx(18 / 108)


def test_zonal_wg_subset_of_union_records_both_directions(tmp_path):
    cases = _make_cases(["c1"])
    cases[0]["pz"] = _box(SHAPE_ZYX, (0, 3), (1, 4), (0, 7))
    cases[0]["tz"] = _box(SHAPE_ZYX, (0, 3), (4, 7), (1, 7))
    tree = _build_tree(tmp_path, cases)
    report = _run_report(tree, tmp_path / "r.json", "--stages", "zonal")
    entry = [
        e for e in report["stages"]["zonal"]["per_case_relations"] if e["source"] == "yuan"
    ][0]
    assert entry["relation"] == audit.REL_WG_SUBSET
    assert entry["wg_outside_union_voxels"] == 0
    assert entry["union_outside_wg_voxels"] == 3 * 3 * 1


def test_zonal_empty_label_is_recorded_and_usable(tmp_path):
    cases = _make_cases(["c1"])
    cases[0]["pz"] = np.zeros(SHAPE_ZYX, dtype=np.uint8)
    cases[0]["tz"] = np.zeros(SHAPE_ZYX, dtype=np.uint8)
    tree = _build_tree(tmp_path, cases)
    report = _run_report(tree, tmp_path / "r.json", "--stages", "zonal")
    summary = report["stages"]["zonal"]["per_source_summary"]["yuan"]
    assert summary["n_empty_label"] == 1
    assert summary["n_label_not_usable"] == 0, "合法空标签不是「不可用」"
    entry = [
        e for e in report["stages"]["zonal"]["per_case_relations"] if e["source"] == "yuan"
    ][0]
    assert entry["union_voxels"] == 0


def test_zonal_grid_mismatch_is_recorded_with_skip_code(tmp_path):
    cases = _make_cases(["c1"])
    cases[0]["zonal_spacing_xyz"] = (0.5, 0.5, 6.0)
    tree = _build_tree(tmp_path, cases)
    report = _run_report(tree, tmp_path / "r.json", "--stages", "zonal")
    entry = [
        e for e in report["stages"]["zonal"]["per_case_relations"] if e["source"] == "yuan"
    ][0]
    assert entry["comparable_to_wg"] is False
    assert entry["skip_code"] == "grid_mismatch"
    assert "跨网格比较不执行" in entry["comparison_skipped_reason"]
    summary = report["stages"]["zonal"]["per_source_summary"]["yuan"]
    assert summary["skip_code_counts"]["grid_mismatch"] == 1
    # grid_mismatch 也进入问题汇总（不重采样 → 未做比较）
    assert any("不同网格" in p for p in report["status"]["problems"])


def test_zonal_cross_source_agreement_is_measured(tmp_path):
    cases = _make_cases(["c1"])
    cases[0]["hevi"] = _rotated_hevi()
    tree = _build_tree(tmp_path, cases)
    report = _run_report(tree, tmp_path / "r.json", "--stages", "zonal")
    x = report["stages"]["zonal"]["cross_source_agreement"]
    assert x["n_comparable"] == 1
    assert x["pz_dice"]["median"] == pytest.approx(2 * 36 / (54 + 36))
    assert x["union_dice"]["median"] == 1.0
    note = report["stages"]["zonal"]["interpretation_note"]
    assert "算法间一致性" in note
    assert "不是准确度" in note
    assert "不是标签噪声下限" in note


def test_zonal_cross_source_agreement_records_skip_code(tmp_path):
    cases = _make_cases(["c1"])
    cases[0]["corrupt"] = {"hevi"}
    tree = _build_tree(tmp_path, cases)
    report = _run_report(tree, tmp_path / "r.json", "--stages", "zonal")
    x = report["stages"]["zonal"]["cross_source_agreement"]
    assert x["n_comparable"] == 0
    assert x["skip_code_counts"]


# ===========================================================================
# 7. Prostate158
# ===========================================================================

#: 与真实 Prostate158 CSV 完全一致的列（reader2 只有 adc_tumor_reader2）
PROSTATE158_FIELDS = [
    "ID",
    "t2",
    "adc",
    "dwi",
    "t2_anatomy_reader1",
    "t2_tumor_reader1",
    "adc_tumor_reader1",
    "t2_anatomy_reader2",
    "adc_tumor_reader2",
]


def _build_prostate158(root: Path, *, anatomy_factory=None) -> None:
    factory = anatomy_factory or (
        lambda: _zonal_from(
            _box((3, 6, 6), (0, 3), (0, 3), (0, 4)),
            _box((3, 6, 6), (0, 3), (3, 6), (0, 4)),
        )
    )
    queues = {
        "train": (root / "train" / "prostate158_train", "train.csv", [24, 25], 1, 1),
        "valid": (root / "train" / "prostate158_train", "valid.csv", [20], 1, 1),
        "test": (root / "test" / "prostate158_test", "test.csv", [1], 2, 2),
    }
    for queue, (base, csv_name, ids, n_anatomy, n_tumor) in queues.items():
        rows = []
        for identifier in ids:
            rel_dir = f"{queue}/{identifier:03d}"
            base_dir = base / rel_dir
            anatomy = factory()
            row = dict.fromkeys(PROSTATE158_FIELDS, "")
            row["ID"] = str(identifier)
            row["t2"] = f"{rel_dir}/t2.nii.gz"
            row["adc"] = f"{rel_dir}/adc.nii.gz"
            row["dwi"] = f"{rel_dir}/dwi.nii.gz"
            _write_nifti(base_dir / "t2.nii.gz", _box((3, 6, 6), (0, 3), (0, 6), (0, 6)))
            for reader in range(1, n_anatomy + 1):
                column = f"t2_anatomy_reader{reader}"
                row[column] = f"{rel_dir}/{column}.nii.gz"
                _write_nifti(base_dir / f"{column}.nii.gz", anatomy)
            for reader in range(1, n_tumor + 1):
                for column in (f"t2_tumor_reader{reader}", f"adc_tumor_reader{reader}"):
                    if column in row:
                        row[column] = f"{rel_dir}/{column}.nii.gz"
            rows.append(row)
        _write_csv(base / csv_name, PROSTATE158_FIELDS, rows)


def test_prostate158_reports_values_by_queue_and_reader(tmp_path):
    tree = _build_tree(tmp_path, _make_cases(["c1"]))
    _build_prostate158(tree.prostate158_root)
    report = _run_report(tree, tmp_path / "r.json", "--stages", "prostate158")
    stage = report["stages"]["prostate158"]
    assert stage["n_ok"] == 5
    assert set(stage["by_queue_reader"]) == {
        "train/reader1",
        "valid/reader1",
        "test/reader1",
        "test/reader2",
    }
    assert stage["observed_value_sets"] == ["0;1;2"]
    assert stage["semantic_status"] == "semantic_unresolved"
    assert stage["allowed_values"] is None
    assert "不声明" in stage["allowed_values_note"]


def test_prostate158_semantic_basis_separates_literature_from_integers(tmp_path):
    tree = _build_tree(tmp_path, _make_cases(["c1"]))
    _build_prostate158(tree.prostate158_root)
    report = _run_report(tree, tmp_path / "r.json", "--stages", "prostate158")
    stage = report["stages"]["prostate158"]
    basis = stage["semantic_basis"]
    assert "central gland" in basis["primary_paper"]["statement"]
    assert basis["primary_paper"]["doi"] == "10.1016/j.compbiomed.2022.105817"
    assert "Transitional Zone" in basis["official_repository"]["statement"]
    assert "不得把论文的 central gland 改名为 TZ" in stage["semantic_rule"]
    assert "不据整数值推断" in stage["semantic_rule"]
    assert "不得把 CG 改名为 TZ" in basis["semantic_note"]


def test_prostate158_non_integer_values_are_flagged(tmp_path):
    tree = _build_tree(tmp_path, _make_cases(["c1"]))

    def factory():
        array = _zonal_from(
            _box((3, 6, 6), (0, 3), (0, 3), (0, 4)),
            _box((3, 6, 6), (0, 3), (3, 6), (0, 4)),
        ).astype(np.float32)
        array[0, 0, 0] = 1.5
        return array

    _build_prostate158(tree.prostate158_root, anatomy_factory=factory)
    report = _run_report(tree, tmp_path / "r.json", "--stages", "prostate158")
    stage = report["stages"]["prostate158"]
    assert stage["n_non_integer"] == stage["n_ok"]
    assert "1.5" in json.dumps(stage["observed_value_sets"][0])
    assert any("非整数值" in p for p in report["status"]["problems"])


def test_prostate158_missing_file_makes_data_invalid(tmp_path):
    tree = _build_tree(tmp_path, _make_cases(["c1"]))
    _build_prostate158(tree.prostate158_root)
    (
        tree.prostate158_root
        / "test"
        / "prostate158_test"
        / "test"
        / "001"
        / "t2_anatomy_reader1.nii.gz"
    ).unlink()
    report = _run_report(tree, tmp_path / "r.json", "--stages", "prostate158")
    assert report["stages"]["prostate158"]["n_missing"] == 1
    assert report["status"]["data_valid"] is False


# ===========================================================================
# 8. ROI 覆盖
# ===========================================================================


def test_expand_bbox_mm_uses_zyx_axis_order_and_ceils_outward():
    spacing_zyx = (3.0, 0.5, 0.7)
    bbox = ((2, 5), (1, 4), (1, 4))
    expanded = audit.expand_bbox_mm(bbox, 3.0, spacing_zyx, (20, 20, 20))
    # z: ceil(3/3.0)=1 ; y: ceil(3/0.5)=6 ; x: ceil(3/0.7)=5（向外取整）
    assert expanded == ((1, 6), (0, 10), (0, 9))


def test_expand_bbox_mm_clips_to_volume_and_zero_margin_is_identity():
    spacing_zyx = (3.0, 0.5, 0.5)
    bbox = ((0, 2), (0, 3), (0, 3))
    shape = (4, 8, 8)
    assert audit.expand_bbox_mm(bbox, 0.0, spacing_zyx, shape) == bbox
    assert audit.expand_bbox_mm(bbox, 5.0, spacing_zyx, shape) == (
        (0, 4),
        (0, 8),
        (0, 8),
    )


def test_expand_bbox_mm_rejects_negative_margin_and_bad_spacing():
    with pytest.raises(audit.AuditError):
        audit.expand_bbox_mm(((0, 1), (0, 1), (0, 1)), -1.0, (1.0, 1.0, 1.0), (4, 4, 4))
    with pytest.raises(audit.AuditError):
        audit.expand_bbox_mm(((0, 1), (0, 1), (0, 1)), 1.0, (0.0, 1.0, 1.0), (4, 4, 4))


def test_largest_component_bbox_takes_max_volume_component():
    mask = np.zeros(SHAPE_ZYX, dtype=bool)
    mask[0, 1:3, 1:3] = True
    mask[3, 5:8, 5:8] = True
    assert audit.largest_component_bbox(mask) == ((3, 4), (5, 8), (5, 8))
    assert audit.largest_component_bbox(np.zeros(SHAPE_ZYX, dtype=bool)) is None


def test_instance_coverages_uses_six_connectivity_for_diagonal_lesions():
    mask = np.zeros((4, 4, 4), dtype=bool)
    mask[1, 1, 1] = True
    mask[2, 2, 2] = True
    assert audit.instance_coverages(mask, ((0, 4), (0, 4), (0, 4)))["n_instances"] == 2


def test_roi_full_coverage(tmp_path):
    tree = _build_tree(tmp_path, _make_cases(["c1"]))
    report = _run_report(tree, tmp_path / "r.json", "--stages", "roi", "--roi-margin-mm", "5")
    for candidate in audit.ROI_CANDIDATES:
        overall = report["stages"]["roi"]["per_candidate_summary"][candidate]["overall"]
        assert overall["denominator_complete"] is True
        assert overall["n_reference_instances"] == 2
        assert overall["n_instances_fully_covered"] == 2
        assert overall["n_instances_fully_missed"] == 0
        assert overall["overall_voxel_coverage"] == 1.0
        assert overall["instance_coverage_macro_mean"] == 1.0
        assert overall["n_cases_fallback"] == 0


def test_roi_partial_coverage_and_fully_missed_instance(tmp_path):
    cases = _make_cases(["c1"])
    cases[0]["wg"] = _box(SHAPE_ZYX, (0, 3), (1, 4), (1, 7))
    cases[0]["bosma"] = cases[0]["wg"]
    cases[0]["guerbet"] = cases[0]["wg"]
    tree = _build_tree(tmp_path, cases)
    report = _run_report(tree, tmp_path / "r.json", "--stages", "roi", "--roi-margin-mm", "0")
    overall = report["stages"]["roi"]["per_candidate_summary"]["wg_materialized"]["overall"]
    assert overall["n_reference_instances"] == 2
    assert overall["n_instances_fully_covered"] == 1
    assert overall["n_instances_fully_missed"] == 1
    assert overall["overall_voxel_coverage"] == pytest.approx(0.5)
    assert overall["instance_coverage_macro_mean"] == pytest.approx(0.5)
    assert overall["instance_fully_missed_ratio"] == pytest.approx(0.5)


def test_roi_applied_margin_is_reported_in_voxels(tmp_path):
    cases = _make_cases(["c1"])
    cases[0]["wg"] = _box(SHAPE_ZYX, (1, 3), (3, 5), (3, 5))
    cases[0]["bosma"] = cases[0]["wg"]
    cases[0]["guerbet"] = cases[0]["wg"]
    tree = _build_tree(tmp_path, cases)
    report = _run_report(tree, tmp_path / "r.json", "--stages", "roi", "--roi-margin-mm", "1")
    entry = report["stages"]["roi"]["per_case"]["wg_materialized"][0]
    assert entry["roi_status"] == audit.ROI_FROM_LABEL
    assert entry["applied_margin_voxels_zyx"] == [[1, 1], [2, 2], [2, 2]]
    assert entry["largest_component_bbox_zyx"] == [[1, 3], [3, 5], [3, 5]]
    assert entry["roi_bbox_zyx"] == [[0, 4], [1, 7], [1, 7]]


def test_roi_margin_is_clipped_at_volume_boundary(tmp_path):
    cases = _make_cases(["c1"])
    cases[0]["wg"] = _box(SHAPE_ZYX, (1, 2), (3, 5), (3, 5))
    cases[0]["bosma"] = cases[0]["wg"]
    cases[0]["guerbet"] = cases[0]["wg"]
    tree = _build_tree(tmp_path, cases)
    report = _run_report(tree, tmp_path / "r.json", "--stages", "roi", "--roi-margin-mm", "3")
    entry = report["stages"]["roi"]["per_case"]["wg_materialized"][0]
    assert entry["roi_bbox_zyx"] == [[0, 3], [0, 8], [0, 8]]
    assert entry["applied_margin_voxels_zyx"] == [[1, 1], [3, 3], [3, 3]]


def test_roi_fallback_on_empty_label_reports_trigger(tmp_path):
    cases = _make_cases(["c1"])
    cases[0]["pz"] = np.zeros(SHAPE_ZYX, dtype=np.uint8)
    cases[0]["tz"] = np.zeros(SHAPE_ZYX, dtype=np.uint8)
    tree = _build_tree(tmp_path, cases)
    report = _run_report(tree, tmp_path / "r.json", "--stages", "roi")
    entry = report["stages"]["roi"]["per_case"]["yuan_pz_tz_union"][0]
    assert entry["roi_status"] == audit.ROI_FALLBACK_FULL_FOV
    assert entry["fallback"] is True
    assert entry["fallback_fov"] == "full_t2w_fov"
    assert entry["roi_bbox_zyx"] == [[0, 4], [0, 8], [0, 8]]
    assert entry["voxel_coverage"] == 1.0
    assert entry["applied_margin_voxels_zyx"] is None
    overall = report["stages"]["roi"]["per_candidate_summary"]["yuan_pz_tz_union"]["overall"]
    assert overall["roi_fallback_trigger_counts"][audit.ROI_SOURCE_EMPTY] == 1


def test_roi_stage_declares_scope_safety_and_no_writeback(tmp_path):
    tree = _build_tree(tmp_path, _make_cases(["c1"]))
    report = _run_report(tree, tmp_path / "r.json", "--stages", "roi")
    stage = report["stages"]["roi"]
    assert "不能称为阶段一预测模型的效果上界" in stage["scope_limit"]
    assert "只应使用外层 train 侧" in stage["scope_limit"]
    assert "不写回任何影像" in stage["roi_definition"]["no_writeback"]
    assert "T2W" in stage["roi_definition"]["geometry_baseline"]
    assert "不把 lesion 的 FOV 称为完整 T2W FOV" in stage["roi_definition"]["geometry_baseline"]
    assert "不存在预先给定的固定数量" in stage["reference_instances"]["denominator"]
    assert "null" in stage["reference_instances"]["unavailable_rule"]


# ===========================================================================
# 9. 候选阶段一区域编码
# ===========================================================================

REGION_SHAPE = (6, 6, 6)


def _wg_pz_tz_masks():
    """**非退化**几何：WG 严格大于 PZ ∪ TZ，``wg_only`` 为 16 体素。"""
    wg = np.zeros(REGION_SHAPE, dtype=bool)
    wg[1:5, 1:5, 1:5] = True
    pz = np.zeros(REGION_SHAPE, dtype=bool)
    pz[1:3, 1:5, 1:5] = True
    tz = np.zeros(REGION_SHAPE, dtype=bool)
    tz[3:4, 1:5, 1:5] = True
    return wg, pz, tz, wg & ~(pz | tz)


def _prob_block(masks):
    import torch

    probs = torch.zeros(len(masks), *REGION_SHAPE)
    for index, mask in enumerate(masks):
        probs[index][mask] = 0.9
    return probs


def _region_dice(seg: np.ndarray, reference: np.ndarray, region) -> float | None:
    from nnunetv2.evaluation.evaluate_predictions import region_or_label_to_mask

    mask_ref = region_or_label_to_mask(reference, region)
    mask_pred = region_or_label_to_mask(seg, region)
    denom = int(mask_ref.sum() + mask_pred.sum())
    if denom == 0:
        return None
    return 2.0 * float((mask_ref & mask_pred).sum()) / denom


def test_previous_encoding_wg_head_is_invisible_to_wg_region_metric():
    """回归证据：``{0,WG:[1,2],PZ:1,TZ:2}`` + ``regions_class_order=[3,1,2]`` 自相矛盾。"""
    from nnunetv2.utilities.label_handling.label_handling import LabelManager

    manager = LabelManager({"background": 0, "WG": [1, 2], "PZ": 1, "TZ": 2}, [3, 1, 2])
    assert manager.foreground_regions == [(1, 2), 1, 2]

    wg, pz, tz, _ = _wg_pz_tz_masks()
    reference = np.zeros(REGION_SHAPE, dtype=np.uint16)
    reference[pz] = 1
    reference[tz] = 2
    empty = np.zeros(REGION_SHAPE, dtype=bool)
    seg = manager.convert_probabilities_to_segmentation(
        _prob_block((wg, empty, empty))
    ).numpy()
    assert sorted(set(seg.ravel().tolist())) == [0, 3]
    assert _region_dice(seg, reference, (1, 2)) == 0.0


def test_pz_and_tz_heads_both_positive_later_region_overwrites_earlier():
    """同一体素 PZ、TZ 两头同时阳性：导出时**后写的区域覆盖先写的**。

    这是导出层的决策，不是「两头概率都被保留」。
    """
    from nnunetv2.evaluation.evaluate_predictions import region_or_label_to_mask
    from nnunetv2.utilities.label_handling.label_handling import LabelManager

    manager = LabelManager({"background": 0, "WG": [1, 2, 3], "PZ": 2, "TZ": 3}, [1, 2, 3])
    # 刻意构造真正重叠的 PZ / TZ 头掩膜
    pz_head = np.zeros(REGION_SHAPE, dtype=bool)
    pz_head[1:3, 1:5, 1:5] = True
    tz_head = np.zeros(REGION_SHAPE, dtype=bool)
    tz_head[2:4, 1:5, 1:5] = True
    overlap = pz_head & tz_head
    assert overlap.any(), "本测试需要真正的头重叠"
    wg_head = pz_head | tz_head

    probs = _prob_block((wg_head, pz_head, tz_head))
    assert bool(probs[1][overlap].min() > 0.5)
    assert bool(probs[2][overlap].min() > 0.5)

    seg = manager.convert_probabilities_to_segmentation(probs).numpy()
    # regions_class_order=[1,2,3]：后写的 TZ 覆盖先前写的 PZ
    assert sorted(set(seg[pz_head & ~tz_head].ravel().tolist())) == [2]
    assert sorted(set(seg[overlap].ravel().tolist())) == [3]

    # 层 1（原始头阈值）里 PZ、TZ 是**重叠**的；层 2（导出整数图）里互斥
    assert bool((pz_head & tz_head)[overlap].all())
    assert not bool(
        (region_or_label_to_mask(seg, 2) & region_or_label_to_mask(seg, 3)).any()
    )
    # 层 2 的 TZ 掩膜 = 两头阳性区域的并集（后写覆盖的结果），不是 TZ 头单独的形状
    assert np.array_equal(region_or_label_to_mask(seg, 3), tz_head | overlap)


def test_three_region_mask_layers_are_distinct(tmp_path):
    """必须区分：①原始头阈值掩膜 ②导出整数图重建的区域掩膜 ③训练参考区域掩膜。"""
    from nnunetv2.evaluation.evaluate_predictions import region_or_label_to_mask
    from nnunetv2.utilities.label_handling.label_handling import LabelManager

    manager = LabelManager({"background": 0, "WG": [1, 2, 3], "PZ": 2, "TZ": 3}, [1, 2, 3])
    wg, pz, tz, wg_only = _wg_pz_tz_masks()
    empty = np.zeros(REGION_SHAPE, dtype=bool)

    # 场景：WG 头全阴性，PZ/TZ 头阳性
    layer1 = (empty, pz, tz)
    seg = manager.convert_probabilities_to_segmentation(_prob_block(layer1)).numpy()
    layer2_wg = region_or_label_to_mask(seg, (1, 2, 3))
    layer2_pz = region_or_label_to_mask(seg, 2)
    layer2_tz = region_or_label_to_mask(seg, 3)

    reference = np.zeros(REGION_SHAPE, dtype=np.uint16)
    reference[wg_only] = 1
    reference[pz] = 2
    reference[tz] = 3
    layer3_wg = region_or_label_to_mask(reference, (1, 2, 3))
    layer3_pz = region_or_label_to_mask(reference, 2)
    layer3_tz = region_or_label_to_mask(reference, 3)

    # 层 1 WG 头全阴性；层 2 WG 区域却非空 —— 两者不等
    assert not bool(np.any(layer1[0]))
    assert int(layer2_wg.sum()) == int((pz | tz).sum())
    # 层 3（参考）里 WG ⊇ PZ ∪ TZ 由**构造**保证，层 2 里没有这个保证
    assert int(layer3_wg.sum()) == int(wg.sum())
    assert np.array_equal(layer3_pz, pz)
    assert np.array_equal(layer3_tz, tz)
    assert not np.array_equal(layer2_wg, layer3_wg)
    assert np.array_equal(layer2_pz, pz)  # PZ 头与参考 PZ 恰好一致（本合成构造）


def test_candidate_4class_encoding_is_self_consistent_but_not_the_only_one():
    """候选 A 自洽，但**不是唯一**合法编码：纯多类同样自洽。"""
    from nnunetv2.evaluation.evaluate_predictions import region_or_label_to_mask
    from nnunetv2.utilities.label_handling.label_handling import LabelManager

    wg, pz, tz, wg_only = _wg_pz_tz_masks()
    reference = np.zeros(REGION_SHAPE, dtype=np.uint16)
    reference[wg_only] = 1
    reference[pz] = 2
    reference[tz] = 3
    empty = np.zeros(REGION_SHAPE, dtype=bool)

    # 候选 A：区域编码 + regions_class_order=[1,2,3]
    manager_a = LabelManager({"background": 0, "WG": [1, 2, 3], "PZ": 2, "TZ": 3}, [1, 2, 3])
    assert manager_a.has_regions is True
    # 候选 B：纯多类（互斥类别，无 region 元组）
    manager_b = LabelManager({"background": 0, "WG_only": 1, "PZ": 2, "TZ": 3}, None)
    assert manager_b.has_regions is False

    for name, masks in {
        "wg_only": (wg_only, empty, empty),
        "pz_only": (empty, pz, empty),
        "tz_only": (empty, empty, tz),
        "wg_pz_tz": (wg, pz, tz),
    }.items():
        seg_a = manager_a.convert_probabilities_to_segmentation(
            _prob_block(masks)
        ).numpy()
        assert np.array_equal(
            region_or_label_to_mask(seg_a, (1, 2, 3)), masks[0] | masks[1] | masks[2]
        ), name
        assert np.array_equal(region_or_label_to_mask(seg_a, 2), masks[1]), name
        assert np.array_equal(region_or_label_to_mask(seg_a, 3), masks[2]), name
        # 候选 B 的导出图取值为 {0,1,2,3}（纯多类），区域掩膜同样自洽
        seg_b = np.zeros(REGION_SHAPE, dtype=np.uint16)
        seg_b[masks[0]] = 1
        seg_b[masks[1]] = 2
        seg_b[masks[2]] = 3
        assert np.array_equal(
            region_or_label_to_mask(seg_b, (1, 2, 3)), masks[0] | masks[1] | masks[2]
        ), name


def test_independent_sigmoids_do_not_guarantee_containment_in_either_direction():
    from nnunetv2.evaluation.evaluate_predictions import region_or_label_to_mask
    from nnunetv2.utilities.label_handling.label_handling import LabelManager

    manager = LabelManager({"background": 0, "WG": [1, 2, 3], "PZ": 2, "TZ": 3}, [1, 2, 3])
    wg, pz, tz, wg_only = _wg_pz_tz_masks()
    reference = np.zeros(REGION_SHAPE, dtype=np.uint16)
    reference[wg_only] = 1
    reference[pz] = 2
    reference[tz] = 3
    empty = np.zeros(REGION_SHAPE, dtype=bool)

    seg = manager.convert_probabilities_to_segmentation(
        _prob_block((empty, pz, tz))
    ).numpy()
    pred_wg = region_or_label_to_mask(seg, (1, 2, 3))
    assert pred_wg.sum() == int(pz.sum() + tz.sum())
    assert _region_dice(seg, reference, (1, 2, 3)) < 1.0

    seg2 = manager.convert_probabilities_to_segmentation(
        _prob_block((wg, empty, empty))
    ).numpy()
    assert region_or_label_to_mask(seg2, (1, 2, 3)).sum() > int(pz.sum() + tz.sum())


def test_multiclass_gland_proxy_probability_is_sum_not_max():
    import torch
    from nnunetv2.utilities.label_handling.label_handling import LabelManager

    manager = LabelManager({"background": 0, "PZ": 1, "TZ": 2}, None)
    assert manager.has_regions is False
    _wg, pz, tz, wg_only = _wg_pz_tz_masks()
    assert wg_only.any()

    logits = torch.zeros(3, *REGION_SHAPE)
    logits[0] = 1.0
    logits[1][pz] = 2.0
    logits[2][tz] = 0.5
    probs = manager.apply_inference_nonlin(logits).numpy()
    proxy_sum = probs[1] + probs[2]
    proxy_max = np.maximum(probs[1], probs[2])
    assert not np.allclose(proxy_sum[pz | tz], proxy_max[pz | tz])
    assert proxy_sum[wg_only].max() < 0.5
    assert proxy_max[wg_only].max() < 0.5


def test_audit_connectivity_matches_frozen_evaluator_constant():
    evaluator = _load_module("evaluate_segmentation", EVALUATOR_SCRIPT)
    assert audit.CONNECTIVITY == evaluator.CONNECTIVITY


# ===========================================================================
# 10. 输出与退出码
# ===========================================================================


def test_parser_has_no_overwrite_flag():
    args = audit.build_parser().parse_args([])
    assert not hasattr(args, "overwrite")
    assert Path(args.output) == audit.DEFAULT_OUTPUT
    assert args.stages == list(audit.STAGE_CHOICES)
    assert args.max_cases is None


def test_refuses_existing_report_and_keeps_it_untouched(tmp_path, capsys):
    tree = _build_tree(tmp_path, _make_cases(["c1"]))
    output = tmp_path / "r.json"
    output.write_text("{}", encoding="utf-8")
    exit_code = _run(tree, output, "--stages", "inputs")
    assert exit_code == audit.EXIT_PREFLIGHT
    assert output.read_text(encoding="utf-8") == "{}"
    assert "拒绝覆盖" in capsys.readouterr().err


def test_rejects_non_json_and_checkpoint_like_outputs(tmp_path):
    for name in ("r.txt", "pred.nii.gz", "checkpoint_final.pth", "report.npz", "x.csv"):
        with pytest.raises(audit.AuditError):
            audit.assert_output_allowed(tmp_path / name)
    audit.assert_output_allowed(tmp_path / "ok.json")


def test_rejects_output_colliding_with_input_paths(tmp_path):
    tree = _build_tree(tmp_path, _make_cases(["c1"]))
    # 输入是 CSV（后缀本身也被禁止），另造一个 .json 输入路径做碰撞测试
    other_input = tmp_path / "inputs.json"
    other_input.write_text("{}", encoding="utf-8")
    with pytest.raises(audit.AuditError) as excinfo:
        audit.assert_output_allowed(other_input, input_paths=(other_input,))
    assert "与输入文件重合" in str(excinfo.value)
    with pytest.raises(audit.AuditError):
        audit.assert_output_allowed(tree.manifest, input_paths=(tree.manifest,))


def test_rejects_output_inside_protected_roots():
    with pytest.raises(audit.AuditError) as excinfo:
        audit.assert_output_allowed(PROJECT_ROOT / "data" / "reports" / "x.json")
    assert "受保护目录" in str(excinfo.value)
    for forbidden in (
        Path("/opt/data/private/lm/data/Prostate/x.json"),
        PROJECT_ROOT / "workdir" / "x.json",
        PROJECT_ROOT / "third_party" / "x.json",
    ):
        with pytest.raises(audit.AuditError):
            audit.assert_output_allowed(forbidden)
    audit.assert_output_allowed(PROJECT_ROOT / "outputs" / "reports" / "x.json")


def test_atomic_write_leaves_no_temporary_file(tmp_path):
    tree = _build_tree(tmp_path, _make_cases(["c1"]))
    output = tmp_path / "r.json"
    _run(tree, output, "--stages", "inputs")
    leftovers = [p.name for p in tmp_path.rglob("*") if ".tmp" in p.name]
    assert leftovers == []
    assert json.loads(output.read_text(encoding="utf-8"))["status"][
        "execution_complete"
    ] is True


def test_input_files_are_not_modified_by_full_audit(tmp_path):
    tree = _build_tree(tmp_path / "inputs", _make_cases(["c1", "c2"]))
    _build_prostate158(tree.prostate158_root)
    before = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in tree.all_input_files()
    }
    assert len(before) > 10
    _run(tree, tmp_path / "report.json")
    after = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in tree.all_input_files()
    }
    assert before == after


def test_exit_code_zero_when_everything_resolves(tmp_path):
    cases = _make_cases(["c1"])
    cases[0]["guerbet"] = np.roll(_wg_default(), shift=1, axis=2)
    tree = _build_tree(tmp_path, cases)
    output = tmp_path / "r.json"
    assert _run(tree, output) == audit.EXIT_OK
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"]["execution_complete"] is True
    assert report["status"]["data_valid"] is True
    assert report["status"]["source_resolved"] is True
    assert report["status"]["problems"] == []
    assert report["status"]["source_problems"] == []


def test_exit_code_two_when_source_unresolved_but_report_written(tmp_path):
    cases = _make_cases(["c1"])
    cases[0]["bosma"] = np.roll(_wg_default(), shift=1, axis=1)
    cases[0]["guerbet"] = np.roll(_wg_default(), shift=1, axis=2)
    tree = _build_tree(tmp_path, cases)
    output = tmp_path / "r.json"
    assert _run(tree, output, "--stages", "inputs", "wg") == audit.EXIT_UNRESOLVED
    assert output.is_file()


def test_exit_code_three_when_preflight_fails(tmp_path):
    tree = _build_tree(tmp_path, _make_cases(["c1"]))
    missing = tmp_path / "nope.json"
    exit_code = audit.main(
        tree.argv(missing) + ["--cases-root", str(tmp_path / "ghost")]
    )
    assert exit_code == audit.EXIT_PREFLIGHT
    assert not missing.exists()


def test_report_records_input_paths_hashes_and_run_parameters(tmp_path):
    tree = _build_tree(tmp_path, _make_cases(["c1"]))
    report = _run_report(
        tree, tmp_path / "r.json", "--stages", "wg", "--roi-margin-mm", "7.5"
    )
    inputs = report["inputs"]
    assert inputs["manifest"] == str(tree.manifest)
    assert inputs["roi_margin_mm"] == 7.5
    assert inputs["requested_stages"] == ["wg"]
    assert inputs["stages_run"] == ["inputs", "wg"]
    assert inputs["inputs_stage_forced"] is True
    assert set(inputs["input_sha256"]) == {
        str(tree.manifest),
        str(tree.split_cases),
        str(tree.materialization_report),
    }
    assert all(len(v) == 64 for v in inputs["input_sha256"].values())
    assert report["inputs_modified"] is False
    assert set(report["evidence_classes"]) >= {
        "original",
        "materialized",
        "existing_report",
    }
    assert report["scope"] == "full_dataset"


def test_invalid_inputs_fail_closed(tmp_path):
    tree = _build_tree(tmp_path, _make_cases(["c1"]))
    output = tmp_path / "r.json"
    assert _run(tree, output) in (audit.EXIT_OK, audit.EXIT_UNRESOLVED)
    # 报告已存在 → 拒绝
    assert _run(tree, output) == audit.EXIT_PREFLIGHT



def test_membership_contract_native_targets_loss_export_and_metrics():
    import torch
    from nnunetv2.utilities.label_handling.label_handling import LabelManager
    from nnunetv2.training.loss.compound_losses import DC_and_BCE_loss
    from batchgeneratorsv2.transforms.utils.seg_to_regions import ConvertSegmentationToRegionsTransform
    # Every combination: WG only, zones inside/outside WG, zone overlap.
    encoded = np.arange(8, dtype=np.uint8).reshape(2, 2, 2)
    masks = audit.decode_anatomy_membership(encoded)
    assert np.array_equal(audit.encode_anatomy_membership(*masks), encoded)
    manager = LabelManager(audit.ANATOMY_REGIONS, [1, 2, 4])
    transform = ConvertSegmentationToRegionsTransform(manager.foreground_regions)
    targets = transform(segmentation=torch.from_numpy(encoded[None].astype(np.int16)))['segmentation']
    assert np.array_equal(targets.numpy(), masks)
    logits = torch.where(targets[None], 8., -8.).requires_grad_()
    loss = DC_and_BCE_loss({}, {'batch_dice': False, 'do_bg': True, 'smooth': 1e-5, 'ddp': False})
    correct = loss(logits, targets[None])
    wrong = loss(-logits, targets[None])
    assert torch.isfinite(correct) and correct < wrong
    correct.backward()
    assert torch.isfinite(logits.grad).all()
    probabilities = manager.apply_inference_nonlin(logits[0]).detach().numpy()
    threshold_masks = probabilities > .5
    assert np.array_equal(threshold_masks, masks)
    # Our lossless membership export, applied AFTER native probability restoration.
    independent_export = audit.encode_anatomy_membership(*threshold_masks)
    assert [_region_dice(independent_export, encoded, r) for r in manager.foreground_regions] == [1., 1., 1.]
    native = manager.convert_probabilities_to_segmentation(probabilities)
    # Native later-region assignment: TZ overrides PZ overrides WG.
    assert native.flatten().tolist() == [0, 1, 2, 2, 4, 4, 4, 4]
    native_dice = [_region_dice(native, encoded, r) for r in manager.foreground_regions]
    assert native_dice == pytest.approx([.4, 2/3, 1.])
    # Contradictory heads and PZ/TZ overlap are preserved by bit export.
    contradictory = threshold_masks.copy()
    contradictory[:, 0, 0, 1] = [False, True, True]
    restored = audit.decode_anatomy_membership(audit.encode_anatomy_membership(*contradictory))
    assert np.array_equal(restored, contradictory)
    assert audit._dice(restored[0], masks[0]) == pytest.approx(6/7)
    assert audit._dice(restored[1], masks[1]) == pytest.approx(8/9)
    assert audit._dice(restored[2], masks[2]) == pytest.approx(8/9)


def test_membership_contract_fail_closed():
    with pytest.raises(audit.AuditError):
        audit.encode_anatomy_membership(None, np.zeros(3), np.zeros(3))
    with pytest.raises(audit.AuditError):
        audit.encode_anatomy_membership(np.array([.5]), np.array([0]), np.array([0]))


def test_corner_displacement_same_indices_mm():
    a = audit.Geometry(Path('synthetic'), (2, 3, 4), (1, 2, 3), (0, 0, 0), np.eye(3).flatten())
    b = audit.Geometry(Path('synthetic'), (2, 3, 4), (1, 2, 3), (3, 4, 0), np.eye(3).flatten())
    result = audit.corner_displacements(a, b)
    assert len(result['distance_mm']) == 8
    assert result['distance_mm'] == [5.] * 8


def test_targeted_followup_only_baseline_unresolved_and_no_writeback(tmp_path, monkeypatch):
    tree = _build_tree(tmp_path / 'tree', _make_cases(['10000_1000000', '10001_1000001']))
    baseline_path = tmp_path / 'baseline.json'
    baseline = _run_report(tree, baseline_path, '--stages', 'wg')
    entry = baseline['stages']['wg']['per_case'][0]
    entry['unique_source_unresolved'] = True
    entry['outcome'] = 'no_comparable_candidate'
    baseline['stages']['wg']['per_case'][1]['unique_source_unresolved'] = False
    baseline_path.write_text(json.dumps(baseline))
    # Synthetic candidate: index array unchanged, physical origin shifted.
    candidate_path = Path(entry['candidates']['bosma22b']['path'])
    import SimpleITK as sitk
    image = sitk.ReadImage(str(candidate_path))
    image.SetOrigin((2., 0., 0.))
    sitk.WriteImage(image, str(candidate_path))
    before = {p: audit.sha256_file(p) for p in tree.all_input_files()}
    def no_full_audit(*args, **kwargs):
        raise AssertionError('full audit must not run')
    monkeypatch.setattr(audit, 'run_audit', no_full_audit)
    output = tmp_path / 'followup.json'
    assert audit.main(['--targeted-followup', str(baseline_path), '--output', str(output), '--no-progress', '--derived-resampling']) == 0
    report = json.loads(output.read_text())
    assert report['scope'] == 'targeted_followup'
    assert report['baseline']['sha256'] == audit.sha256_file(baseline_path)
    assert report['selected_case_ids'] == [entry['case_id']]
    comparison = report['per_case'][0]['comparisons']['materialized_vs_bosma22b']
    assert comparison['index_space']['equal'] is True
    assert comparison['same_grid'] is False
    assert comparison['original_physical_voxel_equal'] is None
    assert comparison['derived_comparison']['original_voxel_identity'] is False
    assert report['per_case'][0]['source_resolved'] is False
    image = report['per_case'][0]['images']['materialized']
    assert 'qform_code' in image['nifti_ras']
    assert all(audit.sha256_file(p) == digest for p, digest in before.items())
    assert audit.main(['--targeted-followup', str(baseline_path), '--output', str(output)]) == audit.EXIT_PREFLIGHT


def test_followup_rejects_duplicate_ids():
    with pytest.raises(audit.AuditError):
        audit.select_followup_entries({'stages': {'wg': {'per_case': [
            {'case_id': 'x', 'unique_source_unresolved': True},
            {'case_id': 'x', 'unique_source_unresolved': True}]}}})



def test_followup_known_missing_skipped_without_images(tmp_path, monkeypatch):
    baseline = {'stages': {'wg': {'per_case': [{
        'case_id': '11050_1001070', 'unique_source_unresolved': True,
        'outcome': audit.WG_MISSING,
        'known_missing_wg': {'category': 'recognized_known_missing'}}]},
        'inputs': {'per_case': []}}}
    source = tmp_path / 'baseline.json'
    source.write_text(json.dumps(baseline))
    def forbidden(*args):
        raise AssertionError('known missing must not read images')
    monkeypatch.setattr(audit, 'followup_image', forbidden)
    output = tmp_path / 'followup.json'
    assert audit.main(['--targeted-followup', str(source), '--output', str(output)]) == 0
    report = json.loads(output.read_text())
    assert report['summary']['skipped_known_missing'] == 1
    assert report['per_case'] == []


def test_native_probability_restore_crop_transpose_and_resampling():
    import torch
    from types import SimpleNamespace
    from nnunetv2.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape
    from nnunetv2.preprocessing.resampling.default_resampling import resample_data_or_seg_to_shape
    from nnunetv2.utilities.label_handling.label_handling import LabelManager
    manager = LabelManager(audit.ANATOMY_REGIONS, [1, 2, 4])
    def resample(data, shape, current_spacing, new_spacing):
        return resample_data_or_seg_to_shape(data, shape, current_spacing, new_spacing,
                                            is_seg=False, order=1, force_separate_z=False)
    plans = SimpleNamespace(transpose_forward=[2, 0, 1], transpose_backward=[1, 2, 0])
    config = SimpleNamespace(spacing=[2., 2., 2.], resampling_fn_probabilities=resample)
    properties = {'spacing': [1., 1., 1.],
                  'shape_after_cropping_and_before_resampling': [4, 4, 4],
                  'shape_before_cropping': [6, 7, 8],
                  'bbox_used_for_cropping': [[1, 5], [2, 6], [3, 7]]}
    logits = torch.ones((3, 2, 2, 2)) * 8
    segmentation, probabilities = convert_predicted_logits_to_segmentation_with_correct_shape(
        logits, plans, config, manager, properties, return_probabilities=True, num_threads_torch=1)
    assert probabilities.shape == (3, 7, 8, 6)
    assert segmentation.shape == (7, 8, 6)
    expected = np.zeros((6, 7, 8), dtype=bool)
    expected[1:5, 2:6, 3:7] = True
    expected = expected.transpose(1, 2, 0)
    assert np.array_equal(probabilities > .5, np.stack([expected]*3))
    assert np.isfinite(probabilities).all()
    exported = audit.encode_anatomy_membership(*(probabilities > .5))
    assert np.array_equal(exported == 7, expected)
