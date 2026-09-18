"""G0-R 图像 QC 的合成测试（临时目录 + 小型合成 NIfTI；不读真实病例、不使用 GPU）。

覆盖（对应任务验收点）：
- 物理坐标重采样（不同网格 → 重采样到 T2W 网格；同网格 → 不重采样）；
- checkerboard / edge-overlay 合成与切片选择；
- sampling manifest 契约、`--case-ids` 子集限制与非法病例拒绝；
- landmark 模板、mm 位移计算、FOV 范围校验；
- 无边界来源安全跳过；有来源时的 Hausdorff/平均表面距离；
- 输出隔离（拒绝覆盖非空目录）、dry-run 不写文件、`--no-progress`；
- 与既有 `alignment_metrics` schema 的兼容性（用 `validate_alignment_metrics_rows` 复核）。
"""
from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk
import yaml

from zonal_reliability_fusion.protocols import common as pc
from zonal_reliability_fusion.protocols import g0_r, g0_r_qc

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts/audit/render_picai_alignment_qc.py"


def load_script():
    spec = importlib.util.spec_from_file_location("render_g0r_qc_under_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script():
    return load_script()


# --------------------------------------------------------------------------- 合成数据
def write_3d_nifti(
    path: Path,
    *,
    size_xyz: tuple[int, int, int] = (32, 32, 6),
    spacing_xyz: tuple[float, float, float] = (0.5, 0.5, 3.0),
    origin_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    blob_xyz: tuple[int, int, int] = (16, 16, 3),
    blob_half: int = 4,
    value: float = 100.0,
) -> Path:
    """写一个含方形亮块的 3D NIfTI（合成；不是医学数据，仅为测试工具行为）。"""
    nx, ny, nz = size_xyz
    arr = np.zeros((nz, ny, nx), dtype=np.float32)
    cx, cy, cz = blob_xyz
    z0, z1 = max(cz - 1, 0), min(cz + 1, nz)
    arr[z0:z1, max(cy - blob_half, 0):cy + blob_half, max(cx - blob_half, 0):cx + blob_half] = value
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing([float(v) for v in spacing_xyz])
    img.SetOrigin([float(v) for v in origin_xyz])
    sitk.WriteImage(img, str(path))
    return path


def make_workspace(tmp_path: Path, *, cases=("c01_1", "c02_1")) -> dict:
    """构造临时工作区：物化病例（合成 3D）+ sampling manifest + 协议配置。"""
    materialized = tmp_path / "processed/picai"
    for case_id in cases:
        case_dir = materialized / "cases" / case_id
        case_dir.mkdir(parents=True)
        write_3d_nifti(case_dir / "t2w.nii.gz", spacing_xyz=(0.5, 0.5, 3.0), blob_xyz=(16, 16, 3))
        # ADC：更粗的平面内网格（16×16 @1.0mm，同物理范围）→ 需要物理坐标重采样
        write_3d_nifti(case_dir / "adc.nii.gz", size_xyz=(16, 16, 6), spacing_xyz=(1.0, 1.0, 3.0), blob_xyz=(8, 8, 3))
        # HBV：与 T2W 同网格 → 不做重采样
        write_3d_nifti(case_dir / "hbv.nii.gz", spacing_xyz=(0.5, 0.5, 3.0), blob_xyz=(16, 16, 3))
    manifest = {
        "protocol": {"id": "G0-R", "version": "draft-0.1", "status": "DRAFT"},
        "sampling": {
            "total_cases": len(cases),
            "cases": [
                {"case_id": cid, "strata": "unit_test", "selection_order": i} for i, cid in enumerate(cases)
            ],
        },
        "evaluation_plan": {"pairs": ["T2W-ADC", "T2W-HBV"], "units": "mm"},
        "status": "PENDING",
    }
    manifest_path = tmp_path / "sampling_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    config = {
        "protocol": {"id": "G0-R", "version": "draft-0.1", "status": "DRAFT", "sees_model_predictions": False},
        "inputs": {"manifest": str(tmp_path / "picai_manifest.csv"), "materialized_cases": str(materialized)},
        "sampling": {"n_cases": None, "strata": {}},
        "evaluation": {
            "pairs": ["T2W-ADC", "T2W-HBV"],
            "units": "mm",
            # 模拟“研究者已冻结最小数量规则”，使严格校验可通过（工具本身不编造阈值）
            "landmark_rules": {"coordinate_space": "native_index_per_series", "min_per_case_pair": 1},
        },
        "readers": {"record_fields": ["case_id", "pair", "reader_a_level", "reader_b_level", "final_level",
                                      "disagreement", "adjudication_note"]},
        "outputs": {"dir": str(tmp_path / "out")},
    }
    config_path = tmp_path / "g0r.yaml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    return {
        "materialized": materialized,
        "manifest": manifest_path,
        "config": config_path,
        "cases": list(cases),
    }


# --------------------------------------------------------------------------- 几何 / 重采样
def test_resample_only_to_t2w_grid_when_grids_differ(tmp_path: Path):
    t2w_path = write_3d_nifti(tmp_path / "t2w.nii.gz", spacing_xyz=(0.5, 0.5, 3.0), blob_xyz=(16, 16, 3))
    adc_path = write_3d_nifti(
        tmp_path / "adc.nii.gz", size_xyz=(16, 16, 6), spacing_xyz=(1.0, 1.0, 3.0), blob_xyz=(8, 8, 3)
    )
    t2w, moving, record = g0_r_qc.prepare_pair_arrays(t2w_path, adc_path, case_id="c01_1", pair="T2W-ADC")
    assert t2w.shape == (6, 32, 32) and moving.shape == (6, 32, 32)  # 已重采样到 T2W 网格
    assert record.same_grid_as_t2w is False and record.resampled_now is True
    assert record.as_dict()["note"] == g0_r_qc.RESAMPLE_ONLY_BANNER
    # 合成亮块在物理上等价位置（index 16,16 @0.5mm == index 8,8 @1.0mm）→ 重采样后应落回中心附近
    center_value = float(moving[3, 16, 16])
    assert center_value > 0.5 * float(np.max(moving))
    # 未做任何配准：几何记录里只有一个线性重采样步骤，且 origin 与 T2W 一致
    assert tuple(record.moving_origin) == (0.0, 0.0, 0.0)
    assert record.overlap_ratio  # 3D 下可计算物理包围盒重叠率


def test_no_resample_when_already_same_grid(tmp_path: Path):
    t2w_path = write_3d_nifti(tmp_path / "t2w.nii.gz")
    hbv_path = write_3d_nifti(tmp_path / "hbv.nii.gz")
    t2w, hbv, record = g0_r_qc.prepare_pair_arrays(t2w_path, hbv_path, case_id="c01_1", pair="T2W-HBV")
    assert record.same_grid_as_t2w is True and record.resampled_now is False
    assert np.array_equal(t2w, hbv)


# --------------------------------------------------------------------------- 显示合成
def test_checkerboard_and_edge_overlay_shapes():
    a = np.zeros((32, 32), dtype=np.float32)
    a[8:24, 8:24] = 1.0
    b = np.zeros((32, 32), dtype=np.float32)
    b[9:25, 9:25] = 2.0
    cb = g0_r_qc.checkerboard_image(a, b, block=4)
    assert cb.shape == (32, 32) and 0.0 <= float(cb.min()) and float(cb.max()) <= 1.0
    rgb = g0_r_qc.edge_overlay_image(a, b)
    assert rgb.shape == (32, 32, 3) and 0.0 <= float(rgb.min()) and float(rgb.max()) <= 1.0
    # 完全不同形状 → 报错
    with pytest.raises(pc.ProtocolConfigError, match="同形状"):
        g0_r_qc.checkerboard_image(a, b[:16])


def test_select_slice_indices():
    assert g0_r_qc.select_slice_indices(20) == [5, 10, 14]
    assert g0_r_qc.select_slice_indices(1) == [0]
    assert g0_r_qc.select_slice_indices(2) == [0, 1]
    with pytest.raises(pc.ProtocolConfigError, match="z 层数"):
        g0_r_qc.select_slice_indices(0)


# --------------------------------------------------------------------------- 抽样清单契约
def test_sampling_manifest_contract_and_subset(tmp_path: Path):
    ws = make_workspace(tmp_path)
    manifest = g0_r_qc.load_sampling_manifest(ws["manifest"])
    assert manifest["case_ids"] == ws["cases"]
    # 子集：允许
    assert g0_r_qc.select_cases(manifest, ["c02_1"]) == ["c02_1"]
    # 不在清单中：必须报错（不可替换/新增病例）
    with pytest.raises(pc.ProtocolConfigError, match="不在 sampling manifest"):
        g0_r_qc.select_cases(manifest, ["c99_9"])
    # protocol.id 错误：必须报错
    bad = json.loads(ws["manifest"].read_text(encoding="utf-8"))
    bad["protocol"]["id"] = "G0-E"
    bad_path = tmp_path / "bad_manifest.json"
    bad_path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(pc.ProtocolConfigError, match="G0-R"):
        g0_r_qc.load_sampling_manifest(bad_path)


def test_resolve_series_paths_uses_materialized_convention(tmp_path: Path):
    ws = make_workspace(tmp_path)
    paths = g0_r_qc.resolve_series_paths(ws["materialized"], "c01_1")
    assert set(paths) == {"t2w", "adc", "hbv"}
    assert all(p.is_file() for p in paths.values())
    assert g0_r_qc.missing_series(paths) == []
    # 兼容两种既有写法：物化根（<root>/cases/<id>）与直接指向 cases 目录（<root>/<id>）
    direct_root = ws["materialized"] / "cases"
    paths_direct = g0_r_qc.resolve_series_paths(direct_root, "c01_1")
    assert paths_direct["t2w"] == paths["t2w"]
    with pytest.raises(pc.ProtocolConfigError, match="物化病例目录不存在"):
        g0_r_qc.resolve_series_paths(ws["materialized"], "nope_0")


# --------------------------------------------------------------------------- landmark
def geometry_entry(
    *,
    size=(32, 32, 6),
    spacing=(0.5, 0.5, 3.0),
    origin=(0.0, 0.0, 0.0),
    direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
) -> dict:
    return {"size": list(size), "spacing": list(spacing), "origin": list(origin), "direction": list(direction)}


def geometry_case_entry(*, t2w: dict | None = None, pairs: dict | None = None) -> dict:
    """构造 `qc_geometry.json` 的 `cases[case_id]` 条目（t2w + 各 pair 的 moving 几何）。"""
    t2w = t2w or geometry_entry()
    entry = {
        "t2w_size": t2w["size"],
        "t2w_spacing": t2w["spacing"],
        "t2w_origin": t2w["origin"],
        "t2w_direction": t2w["direction"],
        "pairs": {},
    }
    for pair, cfg in (pairs or {}).items():
        moving = cfg or t2w
        entry["pairs"][pair] = {
            "moving_size": moving["size"],
            "moving_spacing": moving["spacing"],
            "moving_origin": moving["origin"],
            "moving_direction": moving["direction"],
        }
    return entry


def landmark_row(
    case_id: str = "c01_1",
    pair: str = "T2W-ADC",
    name: str = "lm1",
    *,
    t2w=(10, 10, 2),
    moving=(10, 10, 2),
    space: str = g0_r_qc.COORDINATE_SPACE_NATIVE,
    measured_by: str = "reader_a",
) -> dict:
    return {
        "case_id": case_id,
        "pair": pair,
        "landmark_name": name,
        "coordinate_space": space,
        "t2w_i": str(t2w[0]), "t2w_j": str(t2w[1]), "t2w_k": str(t2w[2]),
        "moving_i": str(moving[0]), "moving_j": str(moving[1]), "moving_k": str(moving[2]),
        "measured_by": measured_by,
        "notes": "",
    }


def test_parse_strict_index_rejects_non_integers():
    """小数坐标不得被静默截断（10.9 ↛ 10）；NaN/Inf/空必须拒绝。"""
    assert g0_r_qc.parse_strict_index("10") == (10, None)
    assert g0_r_qc.parse_strict_index("10.0") == (10, None)
    assert g0_r_qc.parse_strict_index(10) == (10, None)
    assert g0_r_qc.parse_strict_index("1e2") == (100, None)
    for bad in ("10.9", "", "   ", "abc", "nan", "inf", "-inf", None):
        value, err = g0_r_qc.parse_strict_index(bad)
        assert value is None and err, bad
    _, err = g0_r_qc.parse_strict_index("10.9")
    assert err is not None and "整数" in err


def test_validate_collects_all_errors_without_early_return():
    """必须收集全部行的全部错误（旧实现在第一个非法值处直接 return）。"""
    entries = geometry_case_entry(pairs={"T2W-ADC": None})
    rows = [
        landmark_row(t2w=(10.9, 10, 2)),                                # 非整数
        landmark_row(name="lm2", t2w=(-1, 10, 2)),                      # 负值
        landmark_row(name="lm3", space="unknown_space"),                # 坐标空间非法
        landmark_row(name="lm4", moving=(40, 10, 2)),                   # FOV：moving 超网格
        landmark_row(name="lm5", t2w=("", 10, 2)),                      # 空值
    ]
    errors = g0_r_qc.validate_landmark_rows(rows, allowed_cases=["c01_1"], grid_info_by_case={"c01_1": entries})
    assert any("t2w_i" in e and "10.9" in e and "不是整数索引" in e for e in errors)
    assert any("不能为负" in e for e in errors)
    assert any("coordinate_space" in e for e in errors)
    assert any("moving_i=40" in e for e in errors)
    assert any("t2w_i 为空" in e for e in errors)
    assert len(errors) >= 5  # 5 行问题全部报告（旧实现遇到第一个非法值即返回）


def test_validate_fov_uses_separate_t2w_and_moving_sizes():
    """T2W 坐标按 T2W 网格检查、moving 坐标按对应 ADC/HBV 网格检查（不得都用 T2W size）。"""
    entry = geometry_case_entry(
        t2w=geometry_entry(size=(32, 32, 6)),
        pairs={"T2W-ADC": geometry_entry(size=(64, 64, 6))},
    )
    ok = [landmark_row(t2w=(31, 31, 5), moving=(63, 63, 5))]
    assert g0_r_qc.validate_landmark_rows(ok, allowed_cases=["c01_1"], grid_info_by_case={"c01_1": entry}) == []
    # moving 坐标 > T2W size 但在 moving size 内 → 合法（旧实现会误报 t2w 越界）
    larger_only_in_moving = [landmark_row(t2w=(31, 31, 5), moving=(50, 50, 5))]
    assert (
        g0_r_qc.validate_landmark_rows(
            larger_only_in_moving, allowed_cases=["c01_1"], grid_info_by_case={"c01_1": entry}
        )
        == []
    )
    bad_t2w = [landmark_row(t2w=(50, 31, 5), moving=(10, 10, 5))]
    errors = g0_r_qc.validate_landmark_rows(bad_t2w, allowed_cases=["c01_1"], grid_info_by_case={"c01_1": entry})
    assert any("t2w_i=50" in e for e in errors)


def test_validate_rejects_duplicate_and_missing_columns():
    dup = [landmark_row(name="same"), landmark_row(name="same", t2w=(2, 2, 2), moving=(2, 2, 2))]
    errors = g0_r_qc.validate_landmark_rows(dup, allowed_cases=["c01_1"])
    assert any("重复 landmark" in e for e in errors)
    stripped = [{k: v for k, v in landmark_row().items() if k != "coordinate_space"}]
    errors = g0_r_qc.validate_landmark_rows(stripped, allowed_cases=["c01_1"])
    assert any("缺少必需列" in e and "coordinate_space" in e for e in errors)


def test_landmark_template_and_displacement():
    rows = g0_r_qc.build_landmark_template_rows(["c01_1"], ["T2W-ADC", "T2W-HBV"])
    assert len(rows) == 2
    assert all(row["t2w_i"] == "" for row in rows)  # 模板不预填坐标
    assert all(row["coordinate_space"] == g0_r_qc.COORDINATE_SPACE_NATIVE for row in rows)
    entry = geometry_case_entry(pairs={"T2W-ADC": None})
    filled = [landmark_row(t2w=(10, 10, 2), moving=(12, 14, 3))]
    result = g0_r_qc.compute_landmark_metrics(
        filled, geometry_by_case={"c01_1": entry}, allowed_cases=["c01_1"]
    )
    assert not result.errors and len(result.rows) == 1
    row = result.rows[0]
    # 同几何：Δindex=(2,4,1) × spacing (0.5,0.5,3.0) → sqrt(1+4+9)
    expected = math.sqrt((2 * 0.5) ** 2 + (4 * 0.5) ** 2 + (1 * 3.0) ** 2)
    assert row["metric"] == "landmark_displacement_mm"
    assert row["unit"] == "mm"
    assert float(row["value"]) == pytest.approx(expected, rel=1e-6)
    assert g0_r.validate_alignment_metrics_rows(result.rows) == []
    summary = g0_r_qc.summarize_landmark_metrics(result.rows)
    assert summary["c01_1"]["T2W-ADC"]["n_landmarks"] == 1


def test_physical_displacement_uses_origin_spacing_direction():
    """非零 origin / 不同 spacing / 交换轴 direction：位移必须按 LPS 物理坐标计算。"""
    t2w = geometry_entry(size=(64, 64, 8), spacing=(0.5, 0.5, 3.0), origin=(10.0, 20.0, 30.0))
    moving = geometry_entry(size=(64, 64, 8), spacing=(1.0, 1.0, 3.0), origin=(5.0, 5.0, 30.0))
    entry = geometry_case_entry(t2w=t2w, pairs={"T2W-ADC": moving})
    # 同一物理点：t2w (40,40,5) → (30,40,45)；moving (25,35,5) → (30,40,45)
    same = [landmark_row(t2w=(40, 40, 5), moving=(25, 35, 5))]
    result = g0_r_qc.compute_landmark_metrics(same, geometry_by_case={"c01_1": entry}, allowed_cases=["c01_1"])
    assert not result.errors and len(result.rows) == 1
    assert float(result.rows[0]["value"]) == pytest.approx(0.0, abs=1e-6)
    # moving 索引 x 方向 +2 → 物理位移 = 2 × 1.0 mm
    # （旧“Δindex × T2W spacing”算法会算成 |27-40|×0.5 = 6.5 mm）
    shifted = [landmark_row(t2w=(40, 40, 5), moving=(27, 35, 5))]
    result = g0_r_qc.compute_landmark_metrics(shifted, geometry_by_case={"c01_1": entry}, allowed_cases=["c01_1"])
    assert float(result.rows[0]["value"]) == pytest.approx(2.0, abs=1e-6)

    # 交换 x/y 轴的 direction：t2w (20,10,5) 与 moving (10,20,5) 是同一物理点
    swapped = geometry_entry(direction=(0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0))
    entry_swap = geometry_case_entry(t2w=geometry_entry(), pairs={"T2W-ADC": swapped})
    rotated = [landmark_row(t2w=(20, 10, 5), moving=(10, 20, 5))]
    result = g0_r_qc.compute_landmark_metrics(rotated, geometry_by_case={"c01_1": entry_swap}, allowed_cases=["c01_1"])
    assert float(result.rows[0]["value"]) == pytest.approx(0.0, abs=1e-6)


def test_completeness_flags_missing_coverage_and_unfrozen_rule():
    rows = [landmark_row(pair="T2W-ADC"), landmark_row(pair="T2W-HBV")]
    report = g0_r_qc.check_landmark_completeness(
        rows, allowed_cases=["c01_1", "c02_1"], allowed_pairs=["T2W-ADC", "T2W-HBV"]
    )
    assert report["status"] == "INCOMPLETE"
    assert report["n_cases_covered"] == 1
    assert "c02_1/T2W-ADC" in report["missing_case_pair"]
    assert any("landmark_min_count_rule_not_frozen" in b for b in report["blockers"])
    # 冻结最小数量规则后：数量不足成为 blocker
    report2 = g0_r_qc.check_landmark_completeness(
        rows, allowed_cases=["c01_1"], allowed_pairs=["T2W-ADC", "T2W-HBV"], min_per_case_pair=2
    )
    assert any("below_min_count" in b for b in report2["blockers"])
    # measured_by 缺失
    report3 = g0_r_qc.check_landmark_completeness(
        [landmark_row(measured_by="")], allowed_cases=["c01_1"], allowed_pairs=["T2W-ADC"], min_per_case_pair=1
    )
    assert any("measured_by_missing" in b for b in report3["blockers"])


# --------------------------------------------------------------------------- 边界距离
def test_boundary_metrics_skip_without_source():
    result = g0_r_qc.compute_boundary_metrics([])
    assert result.metrics == []
    assert any("no_boundary_source_configured" in item["reason"] for item in result.skipped)


def test_boundary_metrics_missing_entry_and_identical_masks(tmp_path: Path):
    # 缺 moving 条目 → UNKNOWN
    partial = [{"case_id": "c01_1", "pair": "T2W-ADC", "series": "t2w", "boundary_path": str(tmp_path / "x.nii.gz")}]
    result = g0_r_qc.compute_boundary_metrics(partial, allowed_cases=["c01_1"])
    assert result.metrics == []
    assert any("缺少 t2w 或 adc" in item["reason"] for item in result.skipped)

    # 完整条目：相同边界 → Hausdorff≈0；偏移 1 像素 → 距离 > 0
    ref = write_3d_nifti(tmp_path / "b_t2w.nii.gz", spacing_xyz=(1.0, 1.0, 3.0), blob_xyz=(12, 12, 3), blob_half=4, value=1.0)
    same = write_3d_nifti(tmp_path / "b_adc_same.nii.gz", spacing_xyz=(1.0, 1.0, 3.0), blob_xyz=(12, 12, 3), blob_half=4, value=1.0)
    shifted = write_3d_nifti(tmp_path / "b_adc_shift.nii.gz", spacing_xyz=(1.0, 1.0, 3.0), blob_xyz=(13, 12, 3), blob_half=4, value=1.0)
    entries_same = [
        {"case_id": "c01_1", "pair": "T2W-ADC", "series": "t2w", "boundary_path": str(ref)},
        {"case_id": "c01_1", "pair": "T2W-ADC", "series": "adc", "boundary_path": str(same)},
    ]
    result_same = g0_r_qc.compute_boundary_metrics(entries_same, allowed_cases=["c01_1"])
    hd_same = next(r for r in result_same.metrics if r["metric"] == "hausdorff_distance_mm")
    assert float(hd_same["value"]) == pytest.approx(0.0, abs=1e-6)

    entries_shift = [dict(entries_same[0]), dict(entries_same[1], boundary_path=str(shifted))]
    result_shift = g0_r_qc.compute_boundary_metrics(entries_shift, allowed_cases=["c01_1"])
    hd_shift = next(r for r in result_shift.metrics if r["metric"] == "hausdorff_distance_mm")
    ms_shift = next(r for r in result_shift.metrics if r["metric"] == "mean_surface_distance_mm")
    assert 0.5 <= float(hd_shift["value"]) <= 2.0  # 1 像素 × 1.0 mm
    assert 0.0 < float(ms_shift["value"]) <= float(hd_shift["value"]) + 1e-6


# --------------------------------------------------------------------------- 脚本：渲染模式
def test_script_help_and_dry_run(script, tmp_path, monkeypatch, capsys):
    ws = make_workspace(tmp_path)
    out_dir = tmp_path / "run"
    monkeypatch.setattr(pc, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        sys, "argv",
        ["x", "--config", str(ws["config"]), "--sampling-manifest", str(ws["manifest"]),
         "--out-dir", str(out_dir), "--dry-run"],
    )
    script.main()
    out = capsys.readouterr().out
    assert "resample ONLY" in out and "不判 PASS" in out
    assert not out_dir.exists()  # dry-run 不写文件


def test_script_render_end_to_end(script, tmp_path, monkeypatch, capsys):
    ws = make_workspace(tmp_path)
    out_dir = tmp_path / "run"
    monkeypatch.setattr(pc, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        sys, "argv",
        ["x", "--config", str(ws["config"]), "--sampling-manifest", str(ws["manifest"]),
         "--out-dir", str(out_dir), "--no-progress"],
    )
    script.main()
    # PNG：2 例 × 2 序列对 × 2 类型
    for case_id in ws["cases"]:
        for series in ("adc", "hbv"):
            for kind in ("checkerboard", "edge"):
                png = out_dir / "overlay" / f"{case_id}_{series}_{kind}.png"
                assert png.is_file() and png.stat().st_size > 0, png
    index_rows = list(csv.DictReader((out_dir / "qc_index.csv").open(encoding="utf-8")))
    assert len(index_rows) == 4
    assert index_rows[0]["pair"] in ("T2W-ADC", "T2W-HBV")
    template = list(csv.DictReader((out_dir / "landmark_template.csv").open(encoding="utf-8")))
    assert len(template) == 4
    geometry = json.loads((out_dir / "qc_geometry.json").read_text(encoding="utf-8"))
    assert geometry["source"].startswith("physical-space resample only")
    assert geometry["cases"]["c01_1"]["t2w_size"] == [32, 32, 6]
    assert geometry["cases"]["c01_1"]["pairs"]["T2W-ADC"]["resampled_now"] is True
    assert geometry["cases"]["c01_1"]["pairs"]["T2W-HBV"]["resampled_now"] is False
    metadata = json.loads((out_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["extra"]["n_png"] == 8
    assert metadata["extra"]["input_sha256"]  # 输入 SHA256 已记录
    assert "do NOT constitute G0-R PASS" in json.dumps(metadata, ensure_ascii=False)
    # 不包含病灶/模型信息（盲审材料纯净性）
    assert "lesion" not in (out_dir / "qc_index.csv").read_text(encoding="utf-8")
    assert "PASS" in capsys.readouterr().out


def test_script_rejects_out_dir_reuse_and_unknown_case(script, tmp_path, monkeypatch, capsys):
    ws = make_workspace(tmp_path)
    out_dir = tmp_path / "run"
    (out_dir).mkdir(parents=True)
    (out_dir / "existing.txt").write_text("keep", encoding="utf-8")
    monkeypatch.setattr(pc, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        sys, "argv",
        ["x", "--config", str(ws["config"]), "--sampling-manifest", str(ws["manifest"]),
         "--out-dir", str(out_dir), "--no-progress"],
    )
    with pytest.raises(SystemExit, match="拒绝覆盖"):
        script.main()
    assert (out_dir / "existing.txt").read_text(encoding="utf-8") == "keep"  # 既有内容未被触碰

    # 非法病例（不在抽样清单）→ 拒绝
    monkeypatch.setattr(
        sys, "argv",
        ["x", "--config", str(ws["config"]), "--sampling-manifest", str(ws["manifest"]),
         "--out-dir", str(tmp_path / "run2"), "--case-ids", "c99_9", "--no-progress"],
    )
    with pytest.raises(pc.ProtocolConfigError, match="不在 sampling manifest"):
        script.main()


def test_script_case_ids_subset(script, tmp_path, monkeypatch):
    ws = make_workspace(tmp_path)
    out_dir = tmp_path / "run_subset"
    monkeypatch.setattr(pc, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        sys, "argv",
        ["x", "--config", str(ws["config"]), "--sampling-manifest", str(ws["manifest"]),
         "--out-dir", str(out_dir), "--case-ids", "c01_1", "--no-progress"],
    )
    script.main()
    rows = list(csv.DictReader((out_dir / "qc_index.csv").open(encoding="utf-8")))
    assert {row["case_id"] for row in rows} == {"c01_1"}


# --------------------------------------------------------------------------- 脚本：指标与校验模式
def _fill_landmarks(run_dir: Path, filled_path: Path) -> Path:
    template = list(csv.DictReader((run_dir / "landmark_template.csv").open(encoding="utf-8")))
    for row in template:
        row["landmark_name"] = "unit_mark"
        row["t2w_i"] = "10"; row["t2w_j"] = "10"; row["t2w_k"] = "0"
        row["moving_i"] = "11"; row["moving_j"] = "10"; row["moving_k"] = "0"
        row["measured_by"] = "reader_a"
    with open(filled_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(template[0].keys()))
        writer.writeheader()
        writer.writerows(template)
    return filled_path


def test_script_metrics_and_validate_modes(script, tmp_path, monkeypatch, capsys):
    ws = make_workspace(tmp_path)
    out_dir = tmp_path / "run"
    monkeypatch.setattr(pc, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        sys, "argv",
        ["x", "--config", str(ws["config"]), "--sampling-manifest", str(ws["manifest"]),
         "--out-dir", str(out_dir), "--no-progress"],
    )
    script.main()
    filled = _fill_landmarks(out_dir, tmp_path / "filled.csv")

    # 校验模式：合法 → 退出码 0
    monkeypatch.setattr(
        sys, "argv",
        ["x", "--config", str(ws["config"]), "--sampling-manifest", str(ws["manifest"]),
         "--out-dir", str(out_dir), "--validate-landmarks", "--landmarks-csv", str(filled)],
    )
    with pytest.raises(SystemExit) as ok_exc:
        script.main()
    assert ok_exc.value.code == 0
    validation = json.loads((out_dir / "landmarks_validation.json").read_text(encoding="utf-8"))
    assert validation["status"] == "VALID" and validation["fov_check"] is True

    # 指标模式（无 boundary-source → 安全跳过）
    monkeypatch.setattr(
        sys, "argv",
        ["x", "--config", str(ws["config"]), "--sampling-manifest", str(ws["manifest"]),
         "--out-dir", str(out_dir), "--landmarks-csv", str(filled), "--no-progress"],
    )
    script.main()
    metrics_path = out_dir / "alignment_metrics.csv"
    assert metrics_path.is_file()
    rows = list(csv.DictReader(metrics_path.open(encoding="utf-8")))
    assert len(rows) == 4 and all(row["metric"] == "landmark_displacement_mm" for row in rows)
    # 与既有 schema 完全兼容（等价于 --validate-metrics 通过）
    assert g0_r.validate_alignment_metrics_rows(rows) == []
    skipped = json.loads((out_dir / "metrics_skipped.json").read_text(encoding="utf-8"))
    assert any("no_boundary_source_configured" in item["reason"] for item in skipped["boundary_skipped"])
    # 覆盖完整 + measured_by 已填 + 最小数量规则已冻结 → COMPLETE（仍不代表 G0-R 通过）
    summary = json.loads((out_dir / "landmark_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "COMPLETE"
    assert summary["coordinate_space"] == g0_r_qc.COORDINATE_SPACE_NATIVE
    report = json.loads((out_dir / "landmarks_validation.json").read_text(encoding="utf-8"))
    assert report["n_errors"] == 0 and report["landmarks_sha256"] and report["metrics_written"] is True
    assert report["covered_cases"] == report["expected_cases"] == 2
    # 重复运行不得覆盖已存在的 metrics
    with pytest.raises(SystemExit, match="拒绝覆盖"):
        script.main()


def test_script_validate_invalid_landmark(script, tmp_path, monkeypatch):
    ws = make_workspace(tmp_path)
    bad = tmp_path / "bad.csv"
    bad.write_text(
        "case_id,pair,landmark_name,coordinate_space,t2w_i,t2w_j,t2w_k,moving_i,moving_j,moving_k,"
        "measured_by,notes\n"
        "c99_9,T2W-ADC,x,native_index_per_series,1,1,1,1,1,1,reader_a,\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(pc, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        sys, "argv",
        ["x", "--config", str(ws["config"]), "--sampling-manifest", str(ws["manifest"]),
         "--validate-landmarks", "--landmarks-csv", str(bad)],
    )
    with pytest.raises(SystemExit) as exc:
        script.main()
    assert exc.value.code == 1


# --------------------------------------------------------------------------- 指标模式：严格拒绝 / 诊断放行
def _render_run(script, ws, out_dir: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        sys, "argv",
        ["x", "--config", str(ws["config"]), "--sampling-manifest", str(ws["manifest"]),
         "--out-dir", str(out_dir), "--no-progress"],
    )
    script.main()


def _inject_invalid(filled_path: Path, out_path: Path, *, mutate) -> Path:
    rows = list(csv.DictReader(filled_path.open(encoding="utf-8")))
    mutate(rows)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def test_metrics_mode_strictly_rejects_invalid_landmarks(script, tmp_path, monkeypatch):
    """默认严格模式：小数坐标 / FOV 越界 → 非零退出且不写 alignment_metrics.csv。"""
    ws = make_workspace(tmp_path)
    out_dir = tmp_path / "run"
    monkeypatch.setattr(pc, "PROJECT_ROOT", tmp_path)
    _render_run(script, ws, out_dir, monkeypatch)
    filled = _fill_landmarks(out_dir, tmp_path / "filled.csv")

    def mutate(rows):
        rows[0]["t2w_i"] = "10.9"      # 非整数（旧实现会静默截断为 10）
        rows[1]["moving_j"] = "999"    # moving FOV 越界
        rows[2]["t2w_k"] = "-1"        # 负坐标

    bad = _inject_invalid(filled, tmp_path / "bad_filled.csv", mutate=mutate)
    monkeypatch.setattr(
        sys, "argv",
        ["x", "--config", str(ws["config"]), "--sampling-manifest", str(ws["manifest"]),
         "--out-dir", str(out_dir), "--landmarks-csv", str(bad), "--no-progress"],
    )
    with pytest.raises(SystemExit) as exc:
        script.main()
    assert exc.value.code not in (0, None)
    assert not (out_dir / "alignment_metrics.csv").exists()
    report = json.loads((out_dir / "landmarks_validation.json").read_text(encoding="utf-8"))
    assert report["status"] == "INVALID" and report["metrics_written"] is False
    assert report["n_errors"] >= 3
    assert report["landmarks_sha256"] == pc.sha256_file(bad)
    assert "10.9" in " ".join(report["errors"])


def test_metrics_mode_allow_incomplete_before_rule_frozen(script, tmp_path, monkeypatch):
    """协议最小数量规则未冻结 → 默认拒绝（blocker）；--allow-incomplete 放行并标 INCOMPLETE。"""
    ws = make_workspace(tmp_path)
    # 去掉测试配置中的 landmark_rules 模拟“规则未冻结”
    doc = yaml.safe_load(ws["config"].read_text(encoding="utf-8"))
    doc["evaluation"].pop("landmark_rules", None)
    no_rule_config = tmp_path / "g0r_no_rule.yaml"
    no_rule_config.write_text(yaml.safe_dump(doc, allow_unicode=True), encoding="utf-8")

    out_dir = tmp_path / "run"
    monkeypatch.setattr(pc, "PROJECT_ROOT", tmp_path)
    _render_run(script, ws, out_dir, monkeypatch)
    filled = _fill_landmarks(out_dir, tmp_path / "filled.csv")

    argv_common = ["x", "--config", str(no_rule_config), "--sampling-manifest", str(ws["manifest"]),
                   "--out-dir", str(out_dir), "--landmarks-csv", str(filled), "--no-progress"]
    monkeypatch.setattr(sys, "argv", argv_common)
    with pytest.raises(SystemExit) as exc:
        script.main()
    assert exc.value.code not in (0, None)
    assert not (out_dir / "alignment_metrics.csv").exists()
    report = json.loads((out_dir / "landmarks_validation.json").read_text(encoding="utf-8"))
    assert any("landmark_min_count_rule_not_frozen" in b for b in report["blockers"])

    monkeypatch.setattr(sys, "argv", [*argv_common, "--allow-incomplete"])
    script.main()
    metrics_path = out_dir / "alignment_metrics.csv"
    assert metrics_path.is_file()
    written = list(csv.DictReader(metrics_path.open(encoding="utf-8")))
    assert len(written) == 4 and g0_r.validate_alignment_metrics_rows(written) == []
    summary = json.loads((out_dir / "landmark_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "INCOMPLETE"
    assert any("landmark_min_count_rule_not_frozen" in b for b in summary["completeness"]["blockers"])
    meta = json.loads((out_dir / "metrics_run_metadata.json").read_text(encoding="utf-8"))
    assert meta["extra"]["output_status"] == "INCOMPLETE"
    assert meta["extra"]["allow_incomplete"] is True
    assert meta["extra"]["incomplete_note"]


def test_metrics_mode_skips_out_of_fov_in_incomplete_mode(script, tmp_path, monkeypatch):
    """诊断放行时，越界行不得产出位移值（不得用未验证坐标填充指标）。"""
    ws = make_workspace(tmp_path)
    out_dir = tmp_path / "run"
    monkeypatch.setattr(pc, "PROJECT_ROOT", tmp_path)
    _render_run(script, ws, out_dir, monkeypatch)
    filled = _fill_landmarks(out_dir, tmp_path / "filled.csv")
    bad = _inject_invalid(filled, tmp_path / "fov_bad.csv", mutate=lambda rows: rows[0].update({"moving_i": "999"}))
    monkeypatch.setattr(
        sys, "argv",
        ["x", "--config", str(ws["config"]), "--sampling-manifest", str(ws["manifest"]),
         "--out-dir", str(out_dir), "--landmarks-csv", str(bad), "--allow-incomplete", "--no-progress"],
    )
    script.main()
    written = list(csv.DictReader((out_dir / "alignment_metrics.csv").open(encoding="utf-8")))
    assert len(written) == 3  # 越界行被跳过，未产出指标
    skipped = json.loads((out_dir / "metrics_skipped.json").read_text(encoding="utf-8"))
    assert any("超出 FOV" in item["reason"] for item in skipped["landmark_skipped"])
