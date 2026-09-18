"""G0-R Automated v0.3 的合成测试（临时目录 + 合成 3D NIfTI；**不读真实病例、不使用 GPU**）。

覆盖（对应任务验收点）：
- 0 mm / 已知 1-3 mm 平移 / 多方向 / 小角度扰动 / 不同 spacing-origin-direction / 奇数尺寸；
- 部分物理重叠与 FOV 不足标记；NaN/Inf/空/常量影像的 INVALID_INPUT；
- 诊断性刚体估计：真实 SimpleITK 合成对**必须 OK**（不再接受已知类型错误）、Euler3D 类型、有限性、
  固定 seed 可复现，以及 Composite 解包/多元素/错误类型/异常/非有限的 fail-closed；
- v0.3 校准：按 (pair, case_id) unit 分组、unit 内零位移参考、leave-one-out 噪声、pair 级阈值与跨 pair 隔离；
- 合成位移校准的单调性、检出率、零位移误报率与固定 seed 可复现性；
- 校准失败 → 必得 INSUFFICIENT_EVIDENCE；单例判定与指标冲突；
- 协议哈希绑定关键字段、输入策略强制、ADC/HBV 独立输出；
- CLI：`--dry-run` 不写文件、非空输出目录拒绝覆盖、输入影像不被修改；
- 静态扫描：无影像写回 API、无网络 import、无 lesion/WG/PZ-TZ/模型预测读取。
"""
from __future__ import annotations

import ast
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk
import yaml

from zonal_reliability_fusion.protocols import common as pc
from zonal_reliability_fusion.protocols import g0_r_automated_qc as aq

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLI_PATH = PROJECT_ROOT / "scripts/audit/run_picai_alignment_qc_automated.py"
AUTO_CONFIG = PROJECT_ROOT / "configs/protocols/g0_r_alignment_qc_automated.yaml"
LIB_PATH = PROJECT_ROOT / "src/zonal_reliability_fusion/protocols/g0_r_automated_qc.py"


@pytest.fixture(scope="module")
def cli():
    spec = importlib.util.spec_from_file_location("run_g0r_automated_under_test", CLI_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- 合成数据与配置
def _structure_zyx(size_xyz: tuple[int, int, int], seed: int = 7) -> np.ndarray:
    """多尺度合成结构（数组轴序 z, y, x）：大团块 + 平滑纹理。

    单一尺度的高斯团块会让跨模态边缘指标在 ~1 mm 位移后立即饱和（失去单调性），
    与真实影像不符；加入平滑纹理后可复现"位移越大、边缘一致性越差"的连续响应。
    """
    from scipy import ndimage

    nx, ny, nz = size_xyz
    zz, yy, xx = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
    rng = np.random.default_rng(seed)
    field = np.zeros((nz, ny, nx), dtype=np.float64)
    for _ in range(4):
        cz, cy, cx = (rng.uniform(0.3, 0.7) * s for s in (nz, ny, nx))
        rz, ry, rx = (rng.uniform(0.15, 0.25) * s for s in (nz, ny, nx))
        field += np.exp(-(((zz - cz) / rz) ** 2 + ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2))
    # 纹理特征尺度需明显大于被检验的位移（mm），否则 1 mm 位移即整体错位、指标饱和；
    # 这里 x/y 方向约 2.5–3 mm（spacing≈0.7mm），z 方向受 spacing 限制仍保持较细尺度。
    texture = ndimage.gaussian_filter(rng.normal(size=(nz, ny, nx)), sigma=(1.5, 4.0, 4.0))
    field += 0.8 * texture / max(float(np.abs(texture).max()), 1e-9)
    return field / max(float(field.max()), 1e-9)


def _to_image(arr: np.ndarray, spacing_xyz, origin_xyz=(0.0, 0.0, 0.0), direction=None) -> sitk.Image:
    img = sitk.GetImageFromArray(np.asarray(arr, dtype=np.float32))
    img.SetSpacing([float(v) for v in spacing_xyz])
    img.SetOrigin([float(v) for v in origin_xyz])
    if direction is not None:
        img.SetDirection([float(v) for v in direction])
    return img


def synthetic_pair(
    *,
    size_xyz=(32, 32, 8),
    spacing_xyz=(0.7, 0.7, 3.0),
    shift_mm: float = 0.0,
    shift_dir=(1.0, 0.0, 0.0),
    seed: int = 7,
    invert_moving: bool = True,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    """T2W 与 moving 共享同一解剖边界，但强度映射相反；moving 可注入已知位移。"""
    field = _structure_zyx(size_xyz, seed)
    t2w = field * 900.0 + 20.0
    moving = ((1.0 - field) * 700.0 + 30.0) if invert_moving else (field * 600.0 + 40.0)
    if shift_mm:
        moving, _ = aq.inject_translation(
            moving, spacing_xyz=spacing_xyz, direction=shift_dir, distance_mm=shift_mm
        )
    return t2w, moving, tuple(float(v) for v in spacing_xyz)


def prep_cfg() -> dict:
    return {
        "intensity": {
            "method": "robust_percentile_scaling",
            "low_percentile": 1.0,
            "high_percentile": 99.0,
            "background_masking": True,
        },
        "edge": {
            "scales_sigma_mm": [1.0],
            "gradient": "spacing_aware",
            "threshold_percentile": 90.0,
            "min_edge_voxels": 5,
        },
    }


def metrics_cfg() -> dict:
    return {
        "tolerance_radii_mm": [0.5, 1.0, 2.0, 3.0],
        "primary_metric": "sigma1.edge_f1_at_1.0mm",
        "primary_direction": "lower_is_worse",
        "consistency_metrics": [{"name": "sigma1.chamfer_mm", "direction": "higher_is_worse"}],
        "slicewise": False,
    }


def calibration_cfg(**overrides) -> dict:
    cfg = {
        "seed": 20260916,
        # v0.4：unit 完整性（与 canonical 配置同值）
        "min_complete_units_per_pair": 3,
        "require_complete_condition_grid": True,
        "displacements_mm": [0.0, 1.0, 2.0, 3.0],
        "directions": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "interpolation": "linear",
        "max_cases": 2,
        "stability_repeats": 3,
        "stability_noise_sigma": 0.02,
        "detection_multiplier": 3.0,
        "monotonicity_min": 0.8,
        "min_detectable_mm": 2.0,
        "require_detection_rate_at_min": 0.9,
        "max_zero_false_positive_rate": 0.05,
        "scales_used": "all",
    }
    cfg.update(overrides)
    return cfg


def decision_cfg() -> dict:
    return {
        "rule_version": "0.2",
        "all_pairs_must_be_acceptable_for_resample_only": True,
        "any_flagged_pair_triggers_rigid": True,
        "max_invalid_input_fraction": 0.10,
        "max_fov_insufficient_fraction": 0.10,
        "max_registration_failed_fraction": 0.10,
        "allowed_candidates": ["resample-only", "resample+rigid", "INSUFFICIENT_EVIDENCE"],
    }


def threshold_cfg() -> dict:
    return {
        "derivation": aq.THRESHOLD_DERIVATION_BY_PAIR,
        "aggregation_by_pair": "median",
        "direction": "from_metric",
    }


def _passed_calibration(*pairs: str) -> dict:
    """构造「已通过」的分对校准摘要（供 decide_overall 的判定逻辑测试使用）。"""
    names = pairs or ("T2W-ADC", "T2W-HBV")
    return {
        "passed": True,
        "reasons": [],
        "grouping": ["pair", "case_id"],
        "per_pair": {pair: {"pair": pair, "passed": True, "reasons": []} for pair in names},
    }


def _stub_registration(t2w, moving, spacing_xyz, cfg):
    """与真实 runner 同签名的确定性桩（仅合成测试使用）。"""
    del t2w, moving, spacing_xyz, cfg
    return {
        "translation_mm": [0.2, 0.0, 0.0],
        "rotation_deg": [0.0, 0.0, 0.0],
        "translation_magnitude_mm": 0.2,
        "rotation_magnitude_deg": 0.0,
        "metric_before": 1.0,
        "metric_after": 0.4,
    }


def make_workspace(tmp_path: Path, *, cases=("c01_1", "c02_1"), adc_shift_mm: float = 0.0) -> dict:
    """合成物化工作区：病例（t2w / adc 不同网格 / hbv 同网格）+ sampling manifest + 自动协议配置。"""
    materialized = tmp_path / "processed/picai/cases"
    for index, case_id in enumerate(cases):
        case_dir = materialized / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        t2w, moving, spacing = synthetic_pair(seed=7 + index, shift_mm=adc_shift_mm)
        sitk.WriteImage(_to_image(t2w, spacing, origin_xyz=(5.0, 6.0, 7.0)), str(case_dir / "t2w.nii.gz"))
        # ADC：平面内更粗的网格（16×16 @1.4mm ≈ 同物理范围）→ 触发物理坐标重采样
        adc_img = sitk.Resample(
            _to_image(moving, spacing),
            (16, 16, moving.shape[0]),
            sitk.Transform(),
            sitk.sitkLinear,
            (5.0, 6.0, 7.0),
            (1.4, 1.4, spacing[2]),
            (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            0.0,
            sitk.sitkFloat32,
        )
        sitk.WriteImage(adc_img, str(case_dir / "adc.nii.gz"))
        # HBV：与 T2W 完全同网格 → 不做重采样
        sitk.WriteImage(_to_image(moving, spacing, origin_xyz=(5.0, 6.0, 7.0)), str(case_dir / "hbv.nii.gz"))

    manifest = {
        "protocol": {"id": "G0-R", "version": "draft-0.1", "status": "DRAFT"},
        "sampling": {
            "total_cases": len(cases),
            "cases": [
                {"case_id": case_id, "strata": "unit_test", "selection_order": index}
                for index, case_id in enumerate(cases)
            ],
        },
        "evaluation_plan": {"pairs": ["T2W-ADC", "T2W-HBV"], "units": "mm"},
        "status": "PENDING",
    }
    manifest_path = tmp_path / "sampling_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    doc = yaml.safe_load(AUTO_CONFIG.read_text(encoding="utf-8"))
    doc["inputs"]["materialized_root"] = str(materialized)
    doc["inputs"]["sampling_manifest"] = str(manifest_path)
    doc["preprocessing"]["edge"]["scales_sigma_mm"] = [1.0]
    doc["metrics"]["primary_metric"] = "sigma1.edge_f1_at_1.0mm"
    doc["metrics"]["consistency_metrics"] = [{"name": "sigma1.chamfer_mm", "direction": "higher_is_worse"}]
    doc["fov_policy"]["min_axis_overlap_ratio"] = 0.5
    doc["calibration"]["max_cases"] = 1
    doc["calibration"]["displacements_mm"] = [0.0, 1.0, 2.0, 3.0]
    doc["calibration"]["directions"] = [[1, 0, 0], [0, 1, 0]]
    doc["calibration"]["stability_repeats"] = 2
    doc["calibration"]["min_detectable_mm"] = 2.0
    doc["calibration"]["require_detection_rate_at_min"] = 0.5
    config_path = tmp_path / "g0_r_automated_unit.yaml"
    config_path.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return {
        "doc": doc,
        "materialized_root": materialized,
        "manifest_path": manifest_path,
        "config_path": config_path,
        "case_ids": list(cases),
        "files": sorted(materialized.rglob("*.nii.gz")),
    }


# --------------------------------------------------------------------------- 协议/配置
def test_protocol_hash_binds_key_fields_but_not_paths():
    doc = yaml.safe_load(AUTO_CONFIG.read_text(encoding="utf-8"))
    base = aq.protocol_hash(doc)
    changed_fov = json.loads(json.dumps(doc))
    changed_fov["fov_policy"]["min_axis_overlap_ratio"] = 0.99
    assert aq.protocol_hash(changed_fov) != base
    changed_cal = json.loads(json.dumps(doc))
    changed_cal["calibration"]["min_detectable_mm"] = 3.0
    assert aq.protocol_hash(changed_cal) != base
    changed_decision = json.loads(json.dumps(doc))
    changed_decision["decision"]["max_invalid_input_fraction"] = 0.5
    assert aq.protocol_hash(changed_decision) != base
    # 运行参数（inputs/outputs/progress）不改变协议身份
    changed_path = json.loads(json.dumps(doc))
    changed_path["outputs"] = {"dir": "/tmp/elsewhere"}
    changed_path["inputs"] = {"sampling_manifest": "/tmp/x.json", "materialized_root": "/tmp/y"}
    changed_path["progress"] = False
    assert aq.protocol_hash(changed_path) == base


ARCHIVE_CONFIG = PROJECT_ROOT / "configs/protocols/archive/g0_r_alignment_qc_automated_draft_0_2.yaml"
#: draft-0.2 归档配置的原始 SHA256（与真实运行 20260918_074219 的 config_sha256 一致）
ARCHIVED_CONFIG_SHA256 = "67c479a055736693d290d017d2fc9bbe2f08558a593bc40cf76414920659c89a"
ARCHIVE_DRAFT_0_3 = PROJECT_ROOT / "configs/protocols/archive/g0_r_alignment_qc_automated_draft_0_3.yaml"
ARCHIVED_DRAFT_0_3_SHA256 = "beda4bbf1065c9c8493fe41a640f5e4a257076b256438b8dd493cbedfa678198"


def test_archived_draft_0_2_config_is_byte_identical_and_hash_matches():
    """draft-0.2 配置必须按字节归档（旧运行可追溯，哈希与 run_metadata 记录一致）。"""
    assert ARCHIVE_CONFIG.is_file(), ARCHIVE_CONFIG
    raw = ARCHIVE_CONFIG.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == ARCHIVED_CONFIG_SHA256
    run_meta = (
        PROJECT_ROOT / "outputs/diagnostics/g0_r_automated/20260918_074219/run_metadata.json"
    )
    if run_meta.is_file():
        recorded = json.loads(run_meta.read_text(encoding="utf-8"))["config_sha256"]
        assert recorded == ARCHIVED_CONFIG_SHA256


def test_archived_draft_0_3_config_is_byte_identical_and_hash_matches():
    """draft-0.3 配置必须按字节归档（draft-0.4 可追溯）。"""
    assert ARCHIVE_DRAFT_0_3.is_file(), ARCHIVE_DRAFT_0_3
    import hashlib as _hashlib

    assert _hashlib.sha256(ARCHIVE_DRAFT_0_3.read_bytes()).hexdigest() == ARCHIVED_DRAFT_0_3_SHA256


def test_v0_4_keeps_predeclared_fields_unchanged():
    """v0.4 是 fail-closed 修复，但**不得静默改变**抽样/序列对/位移/方向/门限/输入与隐私策略。"""
    new = yaml.safe_load(AUTO_CONFIG.read_text(encoding="utf-8"))
    old = yaml.safe_load(ARCHIVE_CONFIG.read_text(encoding="utf-8"))
    for path in (
        ("protocol", "automated_only"),
        ("protocol", "id"),
        ("protocol", "human_visual_review_required"),
        ("protocol", "human_landmarks_required"),
        ("protocol", "sees_model_predictions"),
        ("privacy",),
        ("input_policy",),
        ("inputs",),
        ("fov_policy",),
        ("preprocessing",),
        ("metrics",),
        ("registration_diagnostic",),
        ("calibration", "seed"),
        ("calibration", "displacements_mm"),
        ("calibration", "directions"),
        ("calibration", "interpolation"),
        ("calibration", "injection_border_mode"),
        ("calibration", "max_cases"),
        ("calibration", "stability_repeats"),
        ("calibration", "stability_noise_sigma"),
        ("calibration", "detection_multiplier"),
        ("calibration", "monotonicity_min"),
        ("calibration", "min_detectable_mm"),
        ("calibration", "require_detection_rate_at_min"),
        ("calibration", "max_zero_false_positive_rate"),
        ("calibration", "scales_used"),
        ("decision", "allowed_candidates"),
        ("decision", "insufficient_evidence_triggers"),
        ("decision", "max_invalid_input_fraction"),
        ("decision", "max_fov_insufficient_fraction"),
        ("decision", "max_registration_failed_fraction"),
    ):
        a, b = old, new
        for key in path:
            a, b = a[key], b[key]
        assert a == b, f"{'.'.join(path)} 不得在 v0.3 中静默改变"
    # 版本与方法性字段必须已升级
    assert new["protocol"]["version"] == "draft-0.4"
    assert new["protocol"]["output_schema_version"] == aq.OUTPUT_SCHEMA_VERSION
    assert new["decision"]["rule_version"] == "0.4"
    assert new["calibration"]["grouping"] == ["pair", "case_id"]
    assert new["calibration"]["zero_reference"] == "within_unit_mean"
    assert new["calibration"]["zero_noise_reference"] == "leave_one_out"
    assert new["calibration"]["require_each_pair_primary_pass"] is True
    assert new["thresholds"]["derivation"] == aq.THRESHOLD_DERIVATION_BY_PAIR
    assert new["thresholds"]["aggregation_by_pair"] == "median"
    # v0.4：unit 完整性冻结值（不得放松）
    assert int(new["calibration"]["min_complete_units_per_pair"]) == aq.CALIBRATION_MIN_COMPLETE_UNITS == 3
    assert bool(new["calibration"]["require_complete_condition_grid"]) is True
    assert aq.REQUIRE_COMPLETE_CONDITION_GRID is True


def test_automated_config_frozen_values():
    doc = yaml.safe_load(AUTO_CONFIG.read_text(encoding="utf-8"))
    aq.assert_automated_input_policy(doc)
    assert doc["protocol"]["id"] == "G0-R-AUTOMATED"
    assert doc["protocol"]["version"] == "draft-0.4"
    assert doc["protocol"]["status"] == "DRAFT"
    assert doc["protocol"]["automated_only"] is True
    assert doc["protocol"]["human_visual_review_required"] is False
    assert doc["protocol"]["human_landmarks_required"] is False
    assert doc["protocol"]["sees_model_predictions"] is False
    assert doc["privacy"]["external_upload_allowed"] is False
    assert doc["privacy"]["external_api_allowed"] is False
    assert doc["privacy"]["source_images_read_only"] is True
    assert doc["thresholds"]["derivation"] == aq.THRESHOLD_DERIVATION_BY_PAIR
    assert str(doc["decision"]["rule_version"]) == "0.4"
    assert int(doc["calibration"]["min_complete_units_per_pair"]) == 3
    assert doc["calibration"]["require_complete_condition_grid"] is True
    assert set(doc["decision"]["allowed_candidates"]) == set(aq.ALLOWED_CANDIDATES)
    scale = float(doc["preprocessing"]["edge"]["scales_sigma_mm"][-1])
    assert str(doc["metrics"]["primary_metric"]).startswith(f"sigma{scale:g}.")
    assert int(doc["calibration"]["seed"]) > 0 and int(doc["registration_diagnostic"]["seed"]) > 0


def test_input_policy_enforced():
    good = {
        "input_policy": {
            "lesion_allowed": False,
            "wg_allowed": False,
            "zonal_allowed": False,
            "model_predictions_allowed": False,
        },
        "privacy": {
            "external_upload_allowed": False,
            "external_api_allowed": False,
            "source_images_read_only": True,
        },
        "protocol": {"automated_only": True, "sees_model_predictions": False},
    }
    aq.assert_automated_input_policy(good)
    for key in ("lesion_allowed", "wg_allowed", "zonal_allowed", "model_predictions_allowed"):
        bad = json.loads(json.dumps(good))
        bad["input_policy"][key] = True
        with pytest.raises(ValueError, match=key):
            aq.assert_automated_input_policy(bad)
    bad_privacy = json.loads(json.dumps(good))
    bad_privacy["privacy"]["external_api_allowed"] = True
    with pytest.raises(ValueError, match="external_api_allowed"):
        aq.assert_automated_input_policy(bad_privacy)


# --------------------------------------------------------------------------- A. 数组/几何/FOV
def test_array_sanity_detects_empty_nan_inf_constant():
    assert aq.check_array_sanity(np.zeros((0, 0, 0)))["ok"] is False
    ok = np.random.default_rng(0).normal(size=(6, 6, 6)) + 10.0
    assert aq.check_array_sanity(ok)["ok"] is True
    with_nan = ok.copy()
    with_nan[0, 0, 0] = np.nan
    assert aq.check_array_sanity(with_nan)["ok"] is False
    with_inf = ok.copy()
    with_inf[0, 0, 0] = np.inf
    report = aq.check_array_sanity(with_inf)
    assert report["finite_fraction"] < 1.0 and report["ok"] is False
    assert aq.check_array_sanity(np.full((4, 4, 4), 5.0))["ok"] is False  # 常量


def test_geometry_sanity_flags_bad_geometry_and_accepts_rotated_direction():
    rotated = {
        "t2w_size": [32, 32, 8],
        "t2w_spacing": [0.7, 0.7, 3.0],
        "t2w_origin": [10.0, 20.0, 30.0],
        "t2w_direction": [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    }
    assert aq.check_geometry_sanity(rotated, role="t2w") == []
    bad_spacing = dict(rotated, t2w_spacing=[0.0, 0.7, 3.0])
    assert any("spacing" in err for err in aq.check_geometry_sanity(bad_spacing, role="t2w"))
    bad_direction = dict(rotated, t2w_direction=[1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    assert any("非正交" in err for err in aq.check_geometry_sanity(bad_direction, role="t2w"))
    missing = {}
    assert len(aq.check_geometry_sanity(missing, role="moving")) == 4


def test_fov_evidence_flags_insufficient_overlap():
    policy = {
        "min_axis_overlap_ratio": 0.5,
        "min_overlapping_axes": 3,
        "min_covered_slices_z": 4,
        "min_nonzero_voxel_fraction": 0.002,
    }
    good = {
        "overlap_ratio": [0.95, 0.95, 0.9],
        "t2w_size": [32, 32, 8],
        "t2w_spacing": [0.7, 0.7, 3.0],
        "moving_size": [32, 32, 8],
        "moving_spacing": [0.7, 0.7, 3.0],
    }
    ok = aq.fov_evidence(good, (8, 32, 32), policy=policy)
    assert ok["fov_ok"] is True and ok["fov_reasons"] == []
    assert ok["partial_overlap"] is True  # 0.9 < 1.0
    partial = dict(good, overlap_ratio=[0.2, 0.9, 0.9])
    flagged = aq.fov_evidence(partial, (8, 32, 32), policy=policy)
    assert flagged["fov_ok"] is False and any("重叠轴数" in r for r in flagged["fov_reasons"])
    thin = aq.fov_evidence(good, (2, 32, 32), policy=policy)
    assert thin["fov_ok"] is False and any("z 层数" in r for r in thin["fov_reasons"])
    unknown = aq.fov_evidence(dict(good, overlap_ratio=[]), (8, 32, 32), policy=policy)
    assert unknown["fov_ok"] is False


# --------------------------------------------------------------------------- B. 边缘一致性（mm / 轴序）
def test_distance_map_respects_axis_order_and_physical_units():
    """轴序验证：数组 axis0=z 必须使用 spacing_z，axis2=x 必须使用 spacing_x。"""
    spacing = (0.5, 0.5, 3.0)
    along_z = np.zeros((4, 4, 4), dtype=bool)
    along_z[0, 0, 0] = True
    along_z[2, 0, 0] = True
    dist_z = aq.distance_map_mm(along_z, spacing)
    assert dist_z is not None
    assert float(dist_z[1, 0, 0]) == pytest.approx(3.0, abs=1e-5)
    along_x = np.zeros((4, 4, 4), dtype=bool)
    along_x[0, 0, 0] = True
    along_x[0, 0, 2] = True
    dist_x = aq.distance_map_mm(along_x, spacing)
    assert dist_x is not None
    assert float(dist_x[0, 0, 1]) == pytest.approx(0.5, abs=1e-5)
    assert aq.distance_map_mm(np.zeros((4, 4, 4), dtype=bool), spacing) is None


def test_inject_translation_moves_content_by_exact_mm():
    arr = _structure_zyx((16, 16, 8), seed=3) * 1000.0
    spacing = (0.5, 0.5, 3.0)
    shifted, info = aq.inject_translation(arr, spacing_xyz=spacing, direction=(0, 0, 1), distance_mm=3.0)
    assert info["offset_voxels_zyx"][0] == pytest.approx(1.0)  # z 方向 3mm = 1 体素
    assert np.allclose(shifted[1:, :, :], arr[:-1, :, :], atol=1e-6)
    same, info0 = aq.inject_translation(arr, spacing_xyz=spacing, direction=(0, 0, 1), distance_mm=0.0)
    assert info0["offset_voxels_zyx"] == [0.0, 0.0, 0.0]
    assert np.array_equal(same, arr)
    with pytest.raises(ValueError, match="零向量"):
        aq.inject_translation(arr, spacing_xyz=spacing, direction=(0, 0, 0), distance_mm=1.0)


def test_zero_displacement_has_better_agreement_than_shifted():
    prep, mcfg = prep_cfg(), metrics_cfg()
    t2w, moving0, spacing = synthetic_pair()
    _, moving2, _ = synthetic_pair(shift_mm=2.0)
    _, moving3z, _ = synthetic_pair(shift_mm=3.0, shift_dir=(0, 0, 1))
    m0 = aq.pair_metrics_from_arrays(t2w, moving0, spacing, prep=prep, metrics_cfg=mcfg)
    m2 = aq.pair_metrics_from_arrays(t2w, moving2, spacing, prep=prep, metrics_cfg=mcfg)
    m3z = aq.pair_metrics_from_arrays(t2w, moving3z, spacing, prep=prep, metrics_cfg=mcfg)
    f1_key, chamfer_key = "sigma1.edge_f1_at_1.0mm", "sigma1.chamfer_mm"
    assert m0[f1_key] > m2[f1_key] >= 0.0
    assert m0[chamfer_key] < m2[chamfer_key]
    assert m0[f1_key] > m3z[f1_key]  # 跨方向同样敏感（z 向 spacing=3mm）


def test_shift_response_monotonic_across_magnitudes_and_directions():
    prep, mcfg = prep_cfg(), metrics_cfg()
    t2w, base, spacing = synthetic_pair()
    f1_key = "sigma1.edge_f1_at_1.0mm"
    for direction in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)):
        values = []
        for distance in (0.0, 1.0, 2.0, 3.0):
            _, moving, _ = synthetic_pair(shift_mm=distance, shift_dir=direction)
            values.append(aq.pair_metrics_from_arrays(t2w, moving, spacing, prep=prep, metrics_cfg=mcfg)[f1_key])
        assert values[0] >= values[1] >= values[2] >= values[3], (direction, values)
    # 未位移与位移差异必须显著
    _, shifted, _ = synthetic_pair(shift_mm=3.0)
    m_base = aq.pair_metrics_from_arrays(t2w, base, spacing, prep=prep, metrics_cfg=mcfg)
    m_shift = aq.pair_metrics_from_arrays(t2w, shifted, spacing, prep=prep, metrics_cfg=mcfg)
    assert m_base[f1_key] - m_shift[f1_key] > 0.1


def test_odd_size_anisotropic_spacing_and_rotated_direction_frames():
    prep, mcfg = prep_cfg(), metrics_cfg()
    t2w, moving, spacing = synthetic_pair(size_xyz=(33, 31, 7), spacing_xyz=(0.6, 0.9, 3.5))
    metrics = aq.pair_metrics_from_arrays(t2w, moving, spacing, prep=prep, metrics_cfg=mcfg)
    assert metrics["sigma1.status"] == "OK"
    assert metrics["sigma1.usable"] is True
    assert 0.0 <= metrics["sigma1.edge_f1_at_1.0mm"] <= 1.0


# --------------------------------------------------------------------------- C. 刚体诊断
def test_rigid_diagnostic_runner_success_and_failure_modes():
    t2w, moving, spacing = synthetic_pair()
    cfg = {"enabled": True, "seed": 1, "failure_status": aq.CASE_REGISTRATION_FAILED}
    ok = aq.estimate_rigid_diagnostic(t2w, moving, spacing, cfg=cfg, runner=_stub_registration)
    assert ok["status"] == "OK"
    assert ok["translation_magnitude_mm"] == pytest.approx(0.2)
    nonfinite = aq.estimate_rigid_diagnostic(
        t2w,
        moving,
        spacing,
        cfg=cfg,
        runner=lambda *a, **k: {"translation_mm": [float("nan"), 0.0, 0.0], "rotation_deg": [0, 0, 0],
                                "metric_before": 1.0, "metric_after": 1.0},
    )
    assert nonfinite["status"] == aq.CASE_REGISTRATION_FAILED
    assert "非有限" in nonfinite["reason"]

    def _boom(*args, **kwargs):
        raise RuntimeError("synthetic failure")

    failed = aq.estimate_rigid_diagnostic(t2w, moving, spacing, cfg=cfg, runner=_boom)
    assert failed["status"] == aq.CASE_REGISTRATION_FAILED and "synthetic failure" in failed["reason"]
    disabled = aq.estimate_rigid_diagnostic(
        t2w, moving, spacing, cfg={**cfg, "enabled": False}, runner=None
    )
    assert disabled["status"] == "DISABLED"


def _sitk_cfg(**overrides) -> dict:
    cfg = {
        "enabled": True,
        "seed": 5,
        "metric_bins": 16,
        "metric_sampling_percentage": 0.2,
        "learning_rate": 1.0,
        "min_step": 0.01,
        "iterations": 20,
        "shrink_factors": [2, 1],
        "smoothing_sigmas_mm": [1.0, 0.0],
        "failure_status": aq.CASE_REGISTRATION_FAILED,
    }
    cfg.update(overrides)
    return cfg


def test_rigid_diagnostic_sitk_returns_ok_on_structured_synthetic_pair():
    """结构化非退化合成对：真实 SimpleITK 路径**必须 OK**（不得再接受已知类型错误）。"""
    t2w, moving, spacing = synthetic_pair(size_xyz=(48, 48, 10), spacing_xyz=(1.0, 1.0, 3.0))
    cfg = _sitk_cfg()
    result = aq.estimate_rigid_diagnostic(t2w, moving, spacing, cfg=cfg)
    assert result["status"] == "OK", result
    assert result["final_transform_type"] == "Euler3DTransform"
    assert result["transform_unwrap"] in ("direct", "composite_single_euler")
    assert len(result["translation_mm"]) == 3 and len(result["rotation_deg"]) == 3
    for key in ("translation_magnitude_mm", "rotation_magnitude_deg", "metric_before", "metric_after"):
        assert np.isfinite(result[key]), key
    assert int(result["n_iterations"]) >= 0
    assert result["optimizer_stop_condition"]


def test_rigid_diagnostic_sitk_is_reproducible_with_fixed_seed():
    """相同输入 + 固定 seed：结果可复现（ITK 多线程归约只允许 ~1e-9 级浮点差异）。"""
    t2w, moving, spacing = synthetic_pair(size_xyz=(48, 48, 10), spacing_xyz=(1.0, 1.0, 3.0))
    cfg = _sitk_cfg()
    first = aq.estimate_rigid_diagnostic(t2w, moving, spacing, cfg=cfg)
    second = aq.estimate_rigid_diagnostic(t2w, moving, spacing, cfg=cfg)
    assert first["status"] == second["status"] == "OK"
    assert first["translation_mm"] == pytest.approx(second["translation_mm"], abs=1e-6)
    assert first["rotation_deg"] == pytest.approx(second["rotation_deg"], abs=1e-6)
    assert first["metric_after"] == pytest.approx(second["metric_after"], abs=1e-9)
    assert first["n_iterations"] == second["n_iterations"]
    assert first["optimizer_stop_condition"] == second["optimizer_stop_condition"]


def test_as_euler3d_transform_unwraps_only_single_euler_composite():
    """CompositeTransform 只允许「单元素 Euler3D」解包；多元素/错误内部类型必须拒绝。"""
    euler = sitk.Euler3DTransform()
    direct, mode, err = aq.as_euler3d_transform(euler)
    assert direct is euler and mode == "direct" and err is None

    single = sitk.CompositeTransform(3)
    single.AddTransform(sitk.Euler3DTransform())
    unwrapped, mode, err = aq.as_euler3d_transform(single)
    assert isinstance(unwrapped, sitk.Euler3DTransform) and mode == "composite_single_euler" and err is None

    many = sitk.CompositeTransform(3)
    many.AddTransform(sitk.Euler3DTransform())
    many.AddTransform(sitk.Euler3DTransform())
    assert aq.as_euler3d_transform(many)[0] is None
    assert "只允许单元素解包" in str(aq.as_euler3d_transform(many)[2])

    wrong_inner = sitk.CompositeTransform(3)
    wrong_inner.AddTransform(sitk.TranslationTransform(3))
    assert aq.as_euler3d_transform(wrong_inner)[0] is None
    assert "TranslationTransform" in str(aq.as_euler3d_transform(wrong_inner)[2])

    assert aq.as_euler3d_transform(sitk.TranslationTransform(3))[0] is None
    assert aq.as_euler3d_transform(None)[0] is None


class _FakeRegistration:
    """最小 SimpleITK 替身：用于注入「Execute 返回 CompositeTransform / 非 Euler」的失败路径。"""

    RANDOM = "RANDOM"

    def __init__(self, final_transform, *, metric_value=-0.5):
        self._final = final_transform
        self._metric = float(metric_value)

    def __getattr__(self, name):
        def _noop(*_args, **_kwargs):
            return None

        return _noop

    def MetricEvaluate(self, *_args):
        return -0.4

    def Execute(self, *_args):
        return self._final

    def GetMetricValue(self):
        return self._metric

    def GetOptimizerIteration(self):
        return 7

    def GetOptimizerStopConditionDescription(self):
        return "fake"


@pytest.mark.parametrize(
    "final_kind,expect_reason",
    [
        ("composite_two_euler", "只允许单元素解包"),
        ("composite_wrong_inner", "TranslationTransform"),
        ("translation", "要求 Euler3DTransform"),
    ],
)
def test_rigid_diagnostic_fails_closed_on_unexpected_final_transform(monkeypatch, final_kind, expect_reason):
    """最终 transform 类型不符 → REGISTRATION_DIAGNOSTIC_FAILED（不得伪造零位移/静默解包）。"""
    t2w, moving, spacing = synthetic_pair(size_xyz=(24, 24, 8), spacing_xyz=(1.0, 1.0, 3.0))
    if final_kind == "composite_two_euler":
        final = sitk.CompositeTransform(3)
        final.AddTransform(sitk.Euler3DTransform())
        final.AddTransform(sitk.Euler3DTransform())
    elif final_kind == "composite_wrong_inner":
        final = sitk.CompositeTransform(3)
        final.AddTransform(sitk.TranslationTransform(3))
    else:
        final = sitk.TranslationTransform(3)
    monkeypatch.setattr(aq.sitk, "ImageRegistrationMethod", lambda: _FakeRegistration(final))
    result = aq.estimate_rigid_diagnostic(t2w, moving, spacing, cfg=_sitk_cfg())
    assert result["status"] == aq.CASE_REGISTRATION_FAILED
    assert expect_reason in result["reason"]
    assert "translation_mm" not in result  # 不伪造零位移


def test_rigid_diagnostic_accepts_single_euler_composite(monkeypatch):
    """单元素 Euler3D Composite 可解包，且必须在输出中如实记录 unwrap 方式。"""
    t2w, moving, spacing = synthetic_pair(size_xyz=(24, 24, 8), spacing_xyz=(1.0, 1.0, 3.0))
    final = sitk.CompositeTransform(3)
    final.AddTransform(sitk.Euler3DTransform())
    monkeypatch.setattr(aq.sitk, "ImageRegistrationMethod", lambda: _FakeRegistration(final))
    result = aq.estimate_rigid_diagnostic(t2w, moving, spacing, cfg=_sitk_cfg())
    assert result["status"] == "OK", result
    assert result["transform_unwrap"] == "composite_single_euler"
    assert result["final_transform_type"] == "CompositeTransform"


# --------------------------------------------------------------------------- D. 校准与阈值
def _calibration_cases(count: int = 2, pairs: tuple[str, ...] = ("T2W-ADC", "T2W-HBV")) -> list[dict]:
    """v0.3：同一病例的两个序列对都要有各自 unit（校准按 (pair, case_id) 分组）。"""
    cases = []
    for index in range(count):
        t2w, moving, spacing = synthetic_pair(seed=7 + index)
        for pair in pairs:
            cases.append(
                {
                    "case_id": f"c{index:02d}",
                    "pair": pair,
                    "t2w": t2w,
                    "moving": moving,
                    "spacing_xyz": spacing,
                    "case_index": index,
                }
            )
    return cases


SYNTH_SCALE = "sigma1.5"
SYNTH_METRIC = f"{SYNTH_SCALE}.m"
SYNTH_USABLE_KEY = f"{SYNTH_SCALE}.usable"


def _synth_metric_cfg(metric: str | None = None) -> dict:
    """v0.4：合成指标名必须带 `sigma<scale>.` 前缀（否则 usable 守卫直接 fail-closed）。"""
    return {
        "primary_metric": metric or SYNTH_METRIC,
        "primary_direction": "lower_is_worse",
        "consistency_metrics": [],
    }


def _three_unit_spec(*, adc=("c1", "c2", "c3"), hbv=("c1", "c2", "c3")) -> dict:
    """三个 unit × 两个 pair 的合成谱（每 unit 有自身 baseline 与位移敏感度）。"""
    adc_slopes = {"c1": (0.35, 0.010), "c2": (0.30, 0.012), "c3": (0.40, 0.009)}
    hbv_slopes = {"c1": (0.12, 0.006), "c2": (0.16, 0.007), "c3": (0.10, 0.005)}
    return {
        "T2W-ADC": {case: adc_slopes[case] for case in adc},
        "T2W-HBV": {case: hbv_slopes[case] for case in hbv},
    }


def _synth_cal_cfg(**overrides) -> dict:
    cfg = {
        "min_detectable_mm": 2.0,
        "detection_multiplier": 3.0,
        "monotonicity_min": 0.8,
        "require_detection_rate_at_min": 0.9,
        "max_zero_false_positive_rate": 0.05,
        "displacements_mm": [0.0, 0.5, 1.0, 2.0, 3.0, 4.0],
        "directions": [[1, 0, 0], [0, 1, 0]],
        "stability_repeats": 3,
        "grouping": ["pair", "case_id"],
        "zero_reference": "within_unit_mean",
        "zero_noise_reference": "leave_one_out",
        "require_each_pair_primary_pass": True,
        # v0.4：每个 pair 至少 3 个 complete unit；条件网格必须完整
        "min_complete_units_per_pair": 3,
        "require_complete_condition_grid": True,
    }
    cfg.update(overrides)
    return cfg


def _synth_records(
    spec: dict,
    *,
    metric: str | None = None,
    distances: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 3.0, 4.0),
    directions: tuple[tuple[int, int, int], ...] = ((1, 0, 0), (0, 1, 0)),
    repeats: int = 3,
    jitter: float = 0.0002,
    drop_conditions: set | None = None,
    nan_conditions: set | None = None,
    unusable_units: set[str] | None = None,
) -> list[dict]:
    """构造合成校准记录：unit=(case_id, pair)，`value = baseline - slope × distance`（lower_is_worse）。

    v0.4 注入能力（用于 fail-closed 回归测试）：
    - `drop_conditions` / `nan_conditions`：键 `(case_id, distance, direction_index)`（零位移用 repeat index）；
    - `unusable_units`：该 case 的全部记录写 `sigma1.5.usable=false`。

    每条记录的 `metrics` 都带 `sigma1.5.usable=true`（除非被 `unusable_units` 关闭）。
    """
    name = metric or SYNTH_METRIC
    drop = set(drop_conditions or set())
    nan = set(nan_conditions or set())
    unusable = set(unusable_units or set())
    records: list[dict] = []

    def _emit(pair, case_id, distance, direction_index, direction, value, condition) -> None:
        key = (case_id, float(distance), direction_index)
        if key in drop:
            return
        if key in nan:
            value = float("nan")
        records.append(
            {
                "case_id": case_id,
                "pair": pair,
                "condition": condition,
                "distance_mm": float(distance),
                "direction": direction,
                "perturbation": {},
                "metrics": {name: float(value), SYNTH_USABLE_KEY: case_id not in unusable},
            }
        )

    for pair, units in sorted(spec.items()):
        for case_id, (baseline, slope) in sorted(units.items()):
            for index in range(repeats):
                _emit(pair, case_id, 0.0, index, None, baseline + jitter * index, f"d0.0|noise{index}")
            for distance in distances:
                if distance <= 0:
                    continue
                for direction_index, direction in enumerate(directions):
                    _emit(
                        pair,
                        case_id,
                        distance,
                        direction_index,
                        list(direction),
                        baseline - slope * float(distance),
                        f"d{distance:g}|dir{direction_index}",
                    )
    return records


def _synth_threshold_cfg(aggregation: str = "median") -> dict:
    return {
        "derivation": aq.THRESHOLD_DERIVATION_BY_PAIR,
        "aggregation_by_pair": aggregation,
        "direction": "from_metric",
    }


def _old_v0_2_mixed_stats(records: list[dict], *, metric: str = "m", multiplier: float = 3.0) -> dict:
    """复现 draft-0.2 的混合绝对 SD 方法（仅用于证明旧方法确实掩盖敏感性）。"""
    zero_values = [float(r["metrics"][metric]) for r in records if float(r["distance_mm"]) == 0.0]
    mean = float(np.mean(zero_values))
    sd = float(np.std(zero_values, ddof=1)) if len(zero_values) > 1 else 0.0
    threshold = min(mean - multiplier * sd, min(zero_values))

    def rate(distance: float) -> float:
        values = [float(r["metrics"][metric]) for r in records if abs(float(r["distance_mm"]) - distance) < 1e-12]
        return float(np.mean([1.0 if v < threshold else 0.0 for v in values]))

    return {"zero_mean": mean, "zero_sd": sd, "threshold": threshold, "detection_rate_2mm": rate(2.0)}


def test_calibration_is_reproducible_with_fixed_seed():
    prep, mcfg, ccfg = prep_cfg(), metrics_cfg(), calibration_cfg()
    cases = _calibration_cases(3)  # v0.4：每 pair 至少 3 个 complete unit
    first = aq.run_calibration(cases, cfg=ccfg, prep=prep, metrics_cfg=mcfg)
    second = aq.run_calibration(cases, cfg=ccfg, prep=prep, metrics_cfg=mcfg)
    assert set(first["per_pair"]) == {"T2W-ADC", "T2W-HBV"}
    for pair, info in first["per_pair"].items():
        for name, metric in info["per_metric"].items():
            other = second["per_pair"][pair]["per_metric"][name]
            assert metric["monotonicity"] == other["monotonicity"]
            assert metric["noise_threshold"] == other["noise_threshold"]
            assert metric["detection_rate_at_min_detectable"] == other["detection_rate_at_min_detectable"]
            assert metric["unit_midpoints"] == other["unit_midpoints"]
            assert [d["zero_values"] for d in metric["unit_details"]] == [
                d["zero_values"] for d in other["unit_details"]
            ]
    assert first["conditions"] == second["conditions"]
    # 条件集合包含 0mm 稳定性扰动与非零位移
    assert any("d0.0" in label for label in first["conditions"])
    assert sum(1 for label in first["conditions"] if label.startswith("d0.0")) == int(ccfg["stability_repeats"])


def test_calibration_passes_and_thresholds_derive_per_pair_midpoints():
    prep, mcfg, ccfg = prep_cfg(), metrics_cfg(), calibration_cfg()
    calibration = aq.run_calibration(_calibration_cases(3), cfg=ccfg, prep=prep, metrics_cfg=mcfg)
    assert calibration["passed"] is True, calibration["per_pair"]
    assert set(calibration["per_pair"]) == {"T2W-ADC", "T2W-HBV"}
    primary = mcfg["primary_metric"]
    for pair in ("T2W-ADC", "T2W-HBV"):
        info = calibration["per_pair"][pair]["per_metric"][primary]
        assert info["passed"] is True, (pair, info["reasons"])
        assert info["n_units_total"] == 3 and info["n_units_complete"] == 3
        assert info["n_units_at_min_detectable"] == 3 and info["incomplete_unit_ids"] == []
        assert info["monotonicity"] >= ccfg["monotonicity_min"]
        assert info["detection_rate_at_min_detectable"] >= ccfg["require_detection_rate_at_min"]
        assert info["zero_false_positive_rate"] <= ccfg["max_zero_false_positive_rate"]
        # 噪声阈值只来自 unit 内残差
        assert info["noise_threshold"] >= info["max_within_unit_zero_degradation"] - 1e-12
        assert info["noise_threshold"] >= ccfg["detection_multiplier"] * info["pooled_within_unit_zero_sd"] - 1e-12
    thresholds = aq.derive_thresholds(calibration, metrics_cfg=mcfg, thresholds_cfg=threshold_cfg())
    entry = thresholds["metrics"][primary]
    assert entry["available"] is True
    assert set(entry["by_pair"]) == {"T2W-ADC", "T2W-HBV"}
    for pair, pair_entry in entry["by_pair"].items():
        info = calibration["per_pair"][pair]["per_metric"][primary]
        expected = float(np.median(list(info["unit_midpoints"].values())))
        assert pair_entry["available"] is True
        assert pair_entry["threshold"] == pytest.approx(expected)
        assert pair_entry["unit_midpoints"] == info["unit_midpoints"]
        assert pair_entry["unit_zero_baselines"] == info["unit_zero_baselines"]
        assert pair_entry["unit_min_detectable_means"] == info["unit_min_detectable_means"]
        assert pair_entry["sensitivity_passed"] is True
        assert pair_entry["threshold"] < float(np.mean(list(info["unit_zero_baselines"].values())))
    with pytest.raises(ValueError, match="不支持的阈值推导公式"):
        aq.derive_thresholds(calibration, metrics_cfg=mcfg, thresholds_cfg={"derivation": "unknown_formula"})


def test_calibration_failure_yields_insufficient_evidence():
    """常量/无边缘影像 → 指标不可校准 → 必得 INSUFFICIENT_EVIDENCE（不得强行二选一）。"""
    prep, mcfg, ccfg = prep_cfg(), metrics_cfg(), calibration_cfg(displacements_mm=[0.0, 2.0])
    flat = np.full((16, 16, 6), 100.0)
    cases = [
        {
            "case_id": "flat",
            "pair": pair,
            "t2w": flat,
            "moving": flat * 0.8,
            "spacing_xyz": (0.7, 0.7, 3.0),
            "case_index": 0,
        }
        for pair in ("T2W-ADC", "T2W-HBV")
    ]
    calibration = aq.run_calibration(cases, cfg=ccfg, prep=prep, metrics_cfg=mcfg)
    assert calibration["passed"] is False
    assert calibration["reasons"]
    thresholds = aq.derive_thresholds(calibration, metrics_cfg=mcfg, thresholds_cfg=threshold_cfg())
    assert thresholds["metrics"][mcfg["primary_metric"]]["available"] is False
    overall = aq.decide_overall(
        [{"case_id": "flat", "pair": "T2W-ADC", "status": aq.CASE_ACCEPTABLE}],
        calibration,
        decision_cfg=decision_cfg(),
    )
    assert overall["candidate"] == aq.DECISION_INSUFFICIENT
    assert any("calibration_not_passed" in reason for reason in overall["reasons"])


def _usable_metrics(**values: float) -> dict:
    """构造带 `sigma*.usable=true` 的指标字典（键为该 sigma 前缀下的指标名）。"""
    out: dict = {}
    for name, value in values.items():
        out[name] = float(value)
        key = aq.metric_usable_key(name)
        if key:
            out[key] = True
    return out


def _pair_thresholds(pair_thresholds: dict, *, pair: str, metrics_cfg: dict, calibration_passed: dict) -> tuple[dict, dict]:
    """构造 v0.3 结构的 thresholds / calibration（只含指定 pair）。"""
    thresholds = {
        "derivation": aq.THRESHOLD_DERIVATION_BY_PAIR,
        "aggregation_by_pair": "median",
        "pairs": [pair],
        "metrics": {
            name: {
                "role": "primary" if name == metrics_cfg["primary_metric"] else "consistency",
                "direction": metrics_cfg["primary_direction"],
                "available": True,
                "by_pair": {pair: {"pair": pair, "available": True, "threshold": value, "sensitivity_passed": True}},
            }
            for name, value in pair_thresholds.items()
        },
    }
    calibration = {
        "per_pair": {
            pair: {
                "pair": pair,
                "passed": bool(calibration_passed.get(pair, True)),
                "per_metric": {name: {"passed": calibration_passed.get(pair, True)} for name in pair_thresholds},
            }
        },
        "passed": bool(calibration_passed.get(pair, True)),
        "reasons": [],
    }
    return thresholds, calibration


def test_decide_case_detects_conflict_and_unavailable_metrics():
    primary, other = f"{SYNTH_SCALE}.m_primary", f"{SYNTH_SCALE}.m_other"
    mcfg = {
        "primary_metric": primary,
        "primary_direction": "lower_is_worse",
        "consistency_metrics": [{"name": other, "direction": "higher_is_worse"}],
    }
    thresholds, calibration = _pair_thresholds(
        {primary: 0.5, other: 2.0}, pair="T2W-ADC", metrics_cfg=mcfg, calibration_passed={}
    )
    ok = aq.decide_case(
        _usable_metrics(**{primary: 0.6, other: 1.0}), thresholds, calibration, pair="T2W-ADC", metrics_cfg=mcfg
    )
    assert ok["status"] == aq.CASE_ACCEPTABLE and ok["pair"] == "T2W-ADC"
    conflict = aq.decide_case(
        _usable_metrics(**{primary: 0.4, other: 1.0}), thresholds, calibration, pair="T2W-ADC", metrics_cfg=mcfg
    )
    assert conflict["status"] == aq.CASE_INSUFFICIENT and "冲突" in conflict["reason"]
    flagged = aq.decide_case(
        _usable_metrics(**{primary: 0.4, other: 3.0}), thresholds, calibration, pair="T2W-ADC", metrics_cfg=mcfg
    )
    assert flagged["status"] == aq.CASE_FLAGGED
    unusable_value = _usable_metrics(**{primary: 0.6, other: 1.0})
    unusable_value[primary] = None
    unavailable = aq.decide_case(unusable_value, thresholds, calibration, pair="T2W-ADC", metrics_cfg=mcfg)
    assert unavailable["status"] == aq.CASE_INSUFFICIENT
    # 该 pair 未通过校准 → 主指标 UNAVAILABLE（不得据此判定）
    bad_thresholds, bad_calibration = _pair_thresholds(
        {primary: 0.5, other: 2.0}, pair="T2W-ADC", metrics_cfg=mcfg, calibration_passed={"T2W-ADC": False}
    )
    not_calibrated = aq.decide_case(
        _usable_metrics(**{primary: 0.9, other: 1.0}),
        bad_thresholds,
        bad_calibration,
        pair="T2W-ADC",
        metrics_cfg=mcfg,
    )
    assert not_calibrated["status"] == aq.CASE_INSUFFICIENT
    # 未知 pair 必须显式失败
    with pytest.raises(ValueError, match="未知 pair"):
        aq.decide_case(
            _usable_metrics(**{primary: 0.9}), thresholds, calibration, pair="T2W-FLAIR", metrics_cfg=mcfg
        )


def test_decide_overall_logic_and_fractions():
    cfg = decision_cfg()
    passed = _passed_calibration()
    only = aq.decide_overall(
        [{"case_id": "c", "pair": "T2W-ADC", "status": aq.CASE_ACCEPTABLE}], passed, decision_cfg=cfg
    )
    assert only["candidate"] == aq.DECISION_RESAMPLE_ONLY
    rigid = aq.decide_overall(
        [
            {"case_id": "c", "pair": "T2W-ADC", "status": aq.CASE_ACCEPTABLE},
            {"case_id": "c", "pair": "T2W-HBV", "status": aq.CASE_FLAGGED},
        ],
        passed,
        decision_cfg=cfg,
    )
    assert rigid["candidate"] == aq.DECISION_RESAMPLE_RIGID
    assert rigid["rigid_required_by"] == ["c/T2W-HBV"]
    fov = aq.decide_overall(
        [{"case_id": "c", "pair": "T2W-ADC", "status": aq.CASE_FOV_INSUFFICIENT}], passed, decision_cfg=cfg
    )
    assert fov["candidate"] == aq.DECISION_INSUFFICIENT
    empty = aq.decide_overall([], passed, decision_cfg=cfg)
    assert empty["candidate"] == aq.DECISION_INSUFFICIENT
    with pytest.raises(ValueError, match="any_flagged_pair_triggers_rigid"):
        aq.decide_overall([], passed, decision_cfg={**cfg, "any_flagged_pair_triggers_rigid": False})


# --------------------------------------------------------------------------- CLI / 端到端
def test_cli_dry_run_writes_nothing(cli, tmp_path, monkeypatch, capsys):
    ws = make_workspace(tmp_path)
    out_dir = tmp_path / "g0_r_automated_out"
    monkeypatch.setattr(pc, "PROJECT_ROOT", tmp_path)
    code = cli.main(
        [
            "--config", str(ws["config_path"]),
            "--sampling-manifest", str(ws["manifest_path"]),
            "--out-dir", str(out_dir),
            "--dry-run", "--no-progress",
        ]
    )
    assert code == 0
    assert not out_dir.exists()
    printed = capsys.readouterr().out
    assert "dry-run" in printed and "不读影像体素、不写任何文件" in printed


def test_cli_rejects_nonempty_out_dir(cli, tmp_path, monkeypatch):
    ws = make_workspace(tmp_path)
    out_dir = tmp_path / "g0_r_automated_out"
    out_dir.mkdir(parents=True)
    (out_dir / "existing.txt").write_text("keep me", encoding="utf-8")
    monkeypatch.setattr(pc, "PROJECT_ROOT", tmp_path)
    with pytest.raises(SystemExit, match="拒绝覆盖"):
        cli.main(
            [
                "--config", str(ws["config_path"]),
                "--sampling-manifest", str(ws["manifest_path"]),
                "--out-dir", str(out_dir),
                "--no-progress",
            ]
        )
    assert (out_dir / "existing.txt").read_text(encoding="utf-8") == "keep me"


def test_pipeline_end_to_end_pairs_independent_and_inputs_untouched(cli, tmp_path, monkeypatch):
    # v0.4：每 pair 需要 ≥3 个 complete unit → 3 例 × 2 对
    ws = make_workspace(tmp_path, cases=("c01_1", "c02_1", "c03_1"))
    ws["doc"]["calibration"]["max_cases"] = 3
    ws["doc"]["calibration"]["min_complete_units_per_pair"] = 3
    ws["doc"]["calibration"]["require_complete_condition_grid"] = True
    out_dir = tmp_path / "run"
    before = {path: pc.sha256_file(path) for path in ws["files"]}
    monkeypatch.setattr(pc, "PROJECT_ROOT", tmp_path)
    result = cli.run_pipeline(
        case_ids=ws["case_ids"],
        pairs=("T2W-ADC", "T2W-HBV"),
        materialized_root=ws["materialized_root"],
        doc=ws["doc"],
        manifest=cli.g0_r_qc.load_sampling_manifest(ws["manifest_path"]),
        manifest_path=ws["manifest_path"],
        out_dir=out_dir,
        progress=False,
        config_path=ws["config_path"],
        registration_runner=_stub_registration,
    )
    # 产物齐全
    for name in (
        "automated_metrics.csv",
        "per_case_decisions.csv",
        "calibration_summary.json",
        "threshold_derivation.json",
        "decision_draft.json",
        "qc_geometry.json",
        "input_hashes.json",
        "run_metadata.json",
        "skipped.csv",
        "qc_report.md",
    ):
        assert (out_dir / name).is_file(), name
    # ADC / HBV 独立结论
    assert set(result["conclusions"]) == {"T2W-ADC", "T2W-HBV"}
    assert len(result["decision_rows"]) == len(ws["case_ids"]) * 2
    decisions = list(csv.DictReader((out_dir / "per_case_decisions.csv").open(encoding="utf-8")))
    assert {row["pair"] for row in decisions} == {"T2W-ADC", "T2W-HBV"}
    assert all(row["status"] in aq.CASE_STATUSES for row in decisions)
    # 自动候选与 DRAFT 标记
    draft = json.loads((out_dir / "decision_draft.json").read_text(encoding="utf-8"))
    assert draft["candidate"] in aq.ALLOWED_CANDIDATES and draft["draft"] is True
    # 报告必须声明非人工阅片
    report = (out_dir / "qc_report.md").read_text(encoding="utf-8")
    assert "没有**人工看图" in report and "DRAFT" in report
    meta = json.loads((out_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert meta["extra"]["data_written_back"] is False
    assert meta["extra"]["protocol_hash"] == aq.protocol_hash(ws["doc"])
    # 输入影像未被修改
    after = {path: pc.sha256_file(path) for path in ws["files"]}
    assert before == after
    hashes = cli.load_output_json(out_dir / "input_hashes.json")
    assert hashes["input_sha256"] and all(
        value == after[Path(key)] for key, value in hashes["input_sha256"].items()
    )
    # 所有 JSON 输出都必须带 schema/协议身份（v0.2 输出不得被当成 v0.3）
    for name in (
        "calibration_summary.json",
        "threshold_derivation.json",
        "decision_draft.json",
        "qc_geometry.json",
        "input_hashes.json",
    ):
        payload = cli.load_output_json(out_dir / name)
        assert payload["schema_version"] == aq.OUTPUT_SCHEMA_VERSION, name
        assert payload["protocol_version"] == "draft-0.4", name
        assert payload["protocol_hash"] == aq.protocol_hash(ws["doc"]), name
        assert payload["draft"] is True, name
    calibration = cli.load_output_json(out_dir / "calibration_summary.json")
    assert set(calibration["per_pair"]) == {"T2W-ADC", "T2W-HBV"}
    assert calibration["grouping"] == ["pair", "case_id"]
    assert calibration["zero_reference"] == "within_unit_mean"
    assert calibration["zero_noise_reference"] == "leave_one_out"
    # v0.4：完整性计数与通过状态
    assert calibration["passed"] is True, calibration["reasons"]
    assert calibration["n_units_complete"] == calibration["n_units_total"] == 6
    assert calibration["incomplete_unit_ids"] == []
    assert calibration["min_complete_units_per_pair"] == 3
    assert calibration["require_complete_condition_grid"] is True
    for pair, info in calibration["per_pair"].items():
        assert info["passed"] is True, (pair, info["reasons"])
        assert info["n_units_complete"] == info["n_units_at_min_detectable"] == 3
        metric = info["per_metric"][ws["doc"]["metrics"]["primary_metric"]]
        assert metric["n_units_complete"] == 3 and metric["incomplete_unit_ids"] == []
        assert metric["metric_usable_key"].endswith(".usable")
    thresholds = cli.load_output_json(out_dir / "threshold_derivation.json")
    by_pair = thresholds["metrics"][ws["doc"]["metrics"]["primary_metric"]]["by_pair"]
    assert set(by_pair) == {"T2W-ADC", "T2W-HBV"}
    for entry in by_pair.values():
        assert entry["n_units_complete"] == 3 and entry["incomplete_unit_ids"] == []
        assert entry["require_complete_condition_grid"] is True


def test_pipeline_flags_invalid_input_and_fov_insufficient(cli, tmp_path, monkeypatch):
    """空/常量 moving → INVALID_INPUT；overlap 过低 → FOV_INSUFFICIENT；两者都不删病例。"""
    ws = make_workspace(tmp_path, cases=("c01_1", "c02_1"))
    # 把 c02_1 的 adc 替换为全零影像（空/常量 → INVALID_INPUT 触发，确定性）
    empty = np.zeros((6, 16, 16), dtype=np.float32)
    img = sitk.GetImageFromArray(empty)
    img.SetSpacing([1.4, 1.4, 3.0])
    img.SetOrigin([5.0, 6.0, 7.0])
    sitk.WriteImage(img, str(ws["materialized_root"] / "c02_1/adc.nii.gz"))
    out_dir = tmp_path / "run"
    monkeypatch.setattr(pc, "PROJECT_ROOT", tmp_path)
    result = cli.run_pipeline(
        case_ids=ws["case_ids"],
        pairs=("T2W-ADC",),
        materialized_root=ws["materialized_root"],
        doc=ws["doc"],
        manifest=cli.g0_r_qc.load_sampling_manifest(ws["manifest_path"]),
        manifest_path=ws["manifest_path"],
        out_dir=out_dir,
        progress=False,
        config_path=ws["config_path"],
        registration_runner=_stub_registration,
    )
    statuses = {(row["case_id"], row["pair"]): row["status"] for row in result["decision_rows"]}
    assert statuses[("c02_1", "T2W-ADC")] == aq.CASE_INVALID_INPUT
    assert statuses[("c01_1", "T2W-ADC")] in (aq.CASE_ACCEPTABLE, aq.CASE_FLAGGED, aq.CASE_INSUFFICIENT)
    assert result["overall"]["candidate"] == aq.DECISION_INSUFFICIENT  # 10% > 0% 上限（1/2 例）
    skipped = list(csv.DictReader((out_dir / "skipped.csv").open(encoding="utf-8")))
    decisions = list(csv.DictReader((out_dir / "per_case_decisions.csv").open(encoding="utf-8")))
    assert len(decisions) == 2  # 不删病例
    assert all(row["case_id"] for row in decisions) and isinstance(skipped, list)


def test_pipeline_marks_registration_failure(cli, tmp_path, monkeypatch):
    ws = make_workspace(tmp_path, cases=("c01_1",))
    out_dir = tmp_path / "run"
    monkeypatch.setattr(pc, "PROJECT_ROOT", tmp_path)

    def _boom(*args, **kwargs):
        raise RuntimeError("registration unavailable")

    result = cli.run_pipeline(
        case_ids=ws["case_ids"],
        pairs=("T2W-HBV",),
        materialized_root=ws["materialized_root"],
        doc=ws["doc"],
        manifest=cli.g0_r_qc.load_sampling_manifest(ws["manifest_path"]),
        manifest_path=ws["manifest_path"],
        out_dir=out_dir,
        progress=False,
        config_path=ws["config_path"],
        registration_runner=_boom,
    )
    assert result["decision_rows"][0]["status"] == aq.CASE_REGISTRATION_FAILED
    assert result["overall"]["candidate"] == aq.DECISION_INSUFFICIENT


# --------------------------------------------------------------------------- 静态边界扫描
def _imported_modules(tree: ast.AST) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".")[0])
    return modules


def test_no_network_no_writeback_no_label_apis():
    banned_modules = {"requests", "urllib", "urllib3", "httpx", "socket", "ftplib", "aiohttp", "boto3", "torch"}
    banned_source_tokens = (
        "sitk.WriteImage",
        "sitk.ImageFileWriter",
        "SaveImage",
        "np.save",
        "canonical_lesion",
        "load_marksheet",
        ".predict(",
        "wg_path",
        "lesion_path",
        "zonal_mask",
        "torch.load",
    )
    for path in (LIB_PATH, CLI_PATH):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        assert not (_imported_modules(tree) & banned_modules), path
        for token in banned_source_tokens:
            assert token not in source, f"{path.name} 含被禁止的调用: {token}"


def test_library_does_not_read_or_write_files():
    """核心算法库只接受数组与元数据：不直接读写影像/文件。"""
    source = LIB_PATH.read_text(encoding="utf-8")
    assert "open(" not in source
    assert "Path(" not in source
    assert "sitk.ReadImage" not in source


# --------------------------------------------------------------------------- v0.3/v0.4：校准分组、pair 隔离与 fail-closed
def _three_unit_calibration(**cfg_overrides):
    """3 unit × 2 pair 的完整合成校准（默认无缺陷）→ 返回 (calibration, thresholds, mcfg, ccfg)。"""
    mcfg = _synth_metric_cfg()
    ccfg = _synth_cal_cfg(**cfg_overrides)
    calibration = aq.summarize_calibration(_synth_records(_three_unit_spec()), cfg=ccfg, metrics_cfg=mcfg)
    thresholds = aq.derive_thresholds(calibration, metrics_cfg=mcfg, thresholds_cfg=_synth_threshold_cfg())
    return calibration, thresholds, mcfg, ccfg


def test_v0_4_three_complete_units_pass_and_defeat_mixed_baseline_sd():
    """B5-1/B5-7：3 个完整 unit × 完整距离方向网格 → 通过；旧混合绝对 SD 仍被证明会漏检。"""
    calibration, thresholds, _mcfg, ccfg = _three_unit_calibration()
    assert calibration["passed"] is True, calibration["reasons"]
    assert calibration["n_units_total"] == 6 and calibration["n_units_complete"] == 6
    assert calibration["incomplete_unit_ids"] == []
    assert calibration["min_complete_units_per_pair"] == 3
    assert calibration["require_complete_condition_grid"] is True
    for pair in ("T2W-ADC", "T2W-HBV"):
        info = calibration["per_pair"][pair]["per_metric"][SYNTH_METRIC]
        assert info["available"] is True and info["passed"] is True, (pair, info["reasons"])
        assert info["n_units_total"] == 3 and info["n_units_complete"] == 3
        assert info["n_units_at_min_detectable"] == 3 and info["n_units_participating"] == 3
        assert info["incomplete_unit_ids"] == []
        assert info["detection_rate_at_min_detectable"] >= ccfg["require_detection_rate_at_min"]
        assert info["zero_false_positive_rate"] <= ccfg["max_zero_false_positive_rate"]
        assert info["metric_usable_key"] == SYNTH_USABLE_KEY
        entry = thresholds["metrics"][SYNTH_METRIC]["by_pair"][pair]
        assert entry["available"] is True and entry["n_units_complete"] == 3
        assert entry["threshold"] == pytest.approx(
            float(np.median(list(info["unit_midpoints"].values())))
        )
    old = _old_v0_2_mixed_stats(_synth_records(_three_unit_spec()), metric=SYNTH_METRIC)
    assert old["detection_rate_2mm"] == 0.0  # 旧混合方法仍然掩盖敏感性
    adc = calibration["per_pair"]["T2W-ADC"]["per_metric"][SYNTH_METRIC]
    assert adc["noise_threshold"] < 0.05 * old["zero_sd"]
    assert adc["noise_threshold"] < 0.05 * (0.35 - 0.12)


def test_v0_4_noise_threshold_excludes_between_unit_baselines():
    """B5-2：病例间基线差异巨大但 unit 内一致退化 → 噪声阈值不被基线差异污染。"""
    spec = {
        "T2W-ADC": {"a": (0.05, 0.010), "b": (0.50, 0.010), "c": (0.95, 0.010)},
        "T2W-HBV": {"a": (0.05, 0.010), "b": (0.50, 0.010), "c": (0.95, 0.010)},
    }
    calibration = aq.summarize_calibration(_synth_records(spec), cfg=_synth_cal_cfg(), metrics_cfg=_synth_metric_cfg())
    info = calibration["per_pair"]["T2W-ADC"]["per_metric"][SYNTH_METRIC]
    assert info["n_units_complete"] == 3
    assert info["noise_threshold"] <= 0.02 * (0.95 - 0.05)
    assert info["detection_rate_at_min_detectable"] == 1.0
    assert info["zero_false_positive_rate"] == 0.0
    assert sorted(d["zero_baseline"] for d in info["unit_details"]) == pytest.approx([0.05, 0.50, 0.95], abs=1e-3)
    assert all(d["complete"] for d in info["unit_details"])


def test_v0_4_zero_displacement_false_positive_rate_within_limit():
    """B5-4：零位移噪声（明显抖动）下 zero FPR 仍在上限内，且噪声阈值随 unit 内抖动缩放。"""
    calibration = aq.summarize_calibration(
        _synth_records(_three_unit_spec(), jitter=0.004), cfg=_synth_cal_cfg(), metrics_cfg=_synth_metric_cfg()
    )
    for pair in ("T2W-ADC", "T2W-HBV"):
        info = calibration["per_pair"][pair]["per_metric"][SYNTH_METRIC]
        assert info["zero_false_positive_rate"] <= 0.05
        assert info["noise_threshold"] >= info["max_within_unit_zero_degradation"] - 1e-12
        assert info["noise_threshold"] >= 3.0 * info["pooled_within_unit_zero_sd"] - 1e-12
        assert info["noise_threshold"] > 0.001


def test_v0_4_insensitive_metric_cannot_pass_by_threshold_gaming():
    """B5-5：位移不敏感指标必须校准失败，且改阈值也不能让它参与判定。"""
    spec = {
        "T2W-ADC": {"c1": (0.35, 0.0), "c2": (0.30, 0.0), "c3": (0.40, 0.0)},
        "T2W-HBV": {"c1": (0.12, 0.006), "c2": (0.16, 0.007), "c3": (0.10, 0.005)},
    }
    mcfg = _synth_metric_cfg()
    calibration = aq.summarize_calibration(_synth_records(spec), cfg=_synth_cal_cfg(), metrics_cfg=mcfg)
    assert calibration["passed"] is False
    adc = calibration["per_pair"]["T2W-ADC"]["per_metric"][SYNTH_METRIC]
    assert adc["available"] is True and adc["passed"] is False
    assert adc["detection_rate_at_min_detectable"] == 0.0
    thresholds = aq.derive_thresholds(calibration, metrics_cfg=mcfg, thresholds_cfg=_synth_threshold_cfg())
    assert thresholds["metrics"][SYNTH_METRIC]["by_pair"]["T2W-ADC"]["sensitivity_passed"] is False
    hacked = json.loads(json.dumps(thresholds))
    hacked["metrics"][SYNTH_METRIC]["by_pair"]["T2W-ADC"]["threshold"] = 0.99
    decision = aq.decide_case(
        _usable_metrics(**{SYNTH_METRIC: 0.998}), hacked, calibration, pair="T2W-ADC", metrics_cfg=mcfg
    )
    assert decision["status"] == aq.CASE_INSUFFICIENT


def test_v0_4_one_insensitive_pair_fails_overall_calibration():
    """B5-3：一个 pair 可校准、另一个不敏感 → 总体 calibration 必须失败。"""
    spec = {
        "T2W-ADC": {"c1": (0.35, 0.010), "c2": (0.30, 0.012), "c3": (0.40, 0.009)},
        "T2W-HBV": {"c1": (0.12, 0.0), "c2": (0.16, 0.0), "c3": (0.10, 0.0)},
    }
    mcfg = _synth_metric_cfg()
    calibration = aq.summarize_calibration(_synth_records(spec), cfg=_synth_cal_cfg(), metrics_cfg=mcfg)
    assert calibration["passed"] is False
    assert calibration["per_pair"]["T2W-ADC"]["passed"] is True
    hbv = calibration["per_pair"]["T2W-HBV"]
    assert hbv["passed"] is False and hbv["n_units_complete"] == 3
    assert any("detection_rate" in reason for reason in hbv["per_metric"][SYNTH_METRIC]["reasons"])
    assert any("T2W-HBV" in reason for reason in calibration["reasons"])
    thresholds = aq.derive_thresholds(calibration, metrics_cfg=mcfg, thresholds_cfg=_synth_threshold_cfg())
    decision = aq.decide_case(
        _usable_metrics(**{SYNTH_METRIC: 0.20}), thresholds, calibration, pair="T2W-HBV", metrics_cfg=mcfg
    )
    assert decision["status"] == aq.CASE_INSUFFICIENT


def test_v0_4_pair_specific_thresholds_and_pair_aware_decide_case():
    """B5-6 + 要求 9：ADC/HBV 阈值分别生成；decide_case 只用对应 pair；未知 pair 显式失败。"""
    calibration, thresholds, mcfg, _ = _three_unit_calibration()
    by_pair = thresholds["metrics"][SYNTH_METRIC]["by_pair"]
    adc_threshold, hbv_threshold = by_pair["T2W-ADC"]["threshold"], by_pair["T2W-HBV"]["threshold"]
    assert adc_threshold != hbv_threshold and adc_threshold > hbv_threshold
    value = 0.25
    adc_decision = aq.decide_case(
        _usable_metrics(**{SYNTH_METRIC: value}), thresholds, calibration, pair="T2W-ADC", metrics_cfg=mcfg
    )
    hbv_decision = aq.decide_case(
        _usable_metrics(**{SYNTH_METRIC: value}), thresholds, calibration, pair="T2W-HBV", metrics_cfg=mcfg
    )
    assert adc_decision["status"] == aq.CASE_FLAGGED
    assert hbv_decision["status"] == aq.CASE_ACCEPTABLE
    assert adc_decision["checks"][0]["threshold"] == adc_threshold
    assert hbv_decision["checks"][0]["threshold"] == hbv_threshold
    with pytest.raises(ValueError, match="未知 pair"):
        aq.decide_case(
            _usable_metrics(**{SYNTH_METRIC: value}), thresholds, calibration, pair="T2W-FLAIR", metrics_cfg=mcfg
        )


def test_v0_4_consistency_metric_is_skipped_per_pair():
    """一致性指标在该 pair 不敏感 → SKIPPED（记录但不参与判定），主指标仍独立判定。"""
    mcfg = {
        "primary_metric": SYNTH_METRIC,
        "primary_direction": "lower_is_worse",
        "consistency_metrics": [{"name": f"{SYNTH_SCALE}.c", "direction": "higher_is_worse"}],
    }
    records = _synth_records(_three_unit_spec())
    for record in records:
        record["metrics"][f"{SYNTH_SCALE}.c"] = 1.0
    calibration = aq.summarize_calibration(records, cfg=_synth_cal_cfg(), metrics_cfg=mcfg)
    assert calibration["per_pair"]["T2W-ADC"]["per_metric"][f"{SYNTH_SCALE}.c"]["passed"] is False
    thresholds = aq.derive_thresholds(calibration, metrics_cfg=mcfg, thresholds_cfg=_synth_threshold_cfg())
    decision = aq.decide_case(
        _usable_metrics(**{SYNTH_METRIC: 0.34, f"{SYNTH_SCALE}.c": 5.0}),
        thresholds,
        calibration,
        pair="T2W-ADC",
        metrics_cfg=mcfg,
    )
    skipped = next(check for check in decision["checks"] if check["metric"] == f"{SYNTH_SCALE}.c")
    assert skipped["decision"] == "SKIPPED" and skipped["pair"] == "T2W-ADC"


def test_v0_4_order_invariance():
    """要求 10：输入顺序打乱后结果完全不变。"""
    mcfg, ccfg = _synth_metric_cfg(), _synth_cal_cfg()
    records = _synth_records(_three_unit_spec())
    base = aq.summarize_calibration(records, cfg=ccfg, metrics_cfg=mcfg)
    rng = np.random.default_rng(11)
    shuffled = [records[index] for index in rng.permutation(len(records))]
    other = aq.summarize_calibration(shuffled, cfg=ccfg, metrics_cfg=mcfg)
    assert other["passed"] == base["passed"]
    assert other["n_units_complete"] == base["n_units_complete"]
    assert other["incomplete_unit_ids"] == base["incomplete_unit_ids"]
    for pair in base["per_pair"]:
        first = base["per_pair"][pair]["per_metric"][SYNTH_METRIC]
        second = other["per_pair"][pair]["per_metric"][SYNTH_METRIC]
        for key in ("noise_threshold", "unit_midpoints", "unit_zero_baselines", "monotonicity",
                    "detection_rate_at_min_detectable", "incomplete_unit_ids", "n_units_complete"):
            assert first[key] == second[key], (pair, key)
    thresholds_first = aq.derive_thresholds(base, metrics_cfg=mcfg, thresholds_cfg=_synth_threshold_cfg())
    thresholds_second = aq.derive_thresholds(other, metrics_cfg=mcfg, thresholds_cfg=_synth_threshold_cfg())
    assert thresholds_first["metrics"][SYNTH_METRIC]["by_pair"] == thresholds_second["metrics"][SYNTH_METRIC]["by_pair"]


# --------------------------------------------------------------------------- v0.4：完整性 / usable fail-closed
def test_v0_4_nan_at_min_detectable_fails_pair_and_overall():
    """要求 1：某 unit 在 2 mm 的主指标为 NaN → 该 pair 与总体均失败。"""
    records = _synth_records(_three_unit_spec(), nan_conditions={("c2", 2.0, 0), ("c2", 2.0, 1)})
    calibration = aq.summarize_calibration(records, cfg=_synth_cal_cfg(), metrics_cfg=_synth_metric_cfg())
    assert calibration["passed"] is False
    adc = calibration["per_pair"]["T2W-ADC"]["per_metric"][SYNTH_METRIC]
    assert adc["passed"] is False
    assert adc["n_units_total"] == 3 and adc["n_units_complete"] == 2
    assert adc["n_units_at_min_detectable"] == 2
    assert "c2" in adc["incomplete_unit_ids"]
    assert any("incomplete_unit_condition_grid" in reason for reason in adc["reasons"])
    assert any("insufficient_complete_units" in reason for reason in adc["reasons"])
    detail = next(d for d in adc["unit_details"] if d["case_id"] == "c2")
    assert detail["complete"] is False
    assert any("非有限" in str(item.get("reason")) for item in detail["invalid_conditions"])
    # 总体必须失败；真实病例判定不得据此给出结论
    thresholds = aq.derive_thresholds(calibration, metrics_cfg=_synth_metric_cfg(), thresholds_cfg=_synth_threshold_cfg())
    decision = aq.decide_case(
        _usable_metrics(**{SYNTH_METRIC: 0.30}), thresholds, calibration, pair="T2W-ADC", metrics_cfg=_synth_metric_cfg()
    )
    assert decision["status"] == aq.CASE_INSUFFICIENT


def test_v0_4_missing_direction_at_min_detectable_fails():
    """要求 2：某 unit 在 2 mm 缺少一个方向 → 失败，且 missing_conditions 精确到该方向。"""
    records = _synth_records(_three_unit_spec(), drop_conditions={("c3", 2.0, 1)})
    calibration = aq.summarize_calibration(records, cfg=_synth_cal_cfg(), metrics_cfg=_synth_metric_cfg())
    assert calibration["passed"] is False
    adc = calibration["per_pair"]["T2W-ADC"]["per_metric"][SYNTH_METRIC]
    detail = next(d for d in adc["unit_details"] if d["case_id"] == "c3")
    assert detail["complete"] is False
    assert "2|dir1" in detail["missing_conditions"]
    entry = detail["by_distance"]["2"]
    assert entry["n_expected_directions"] == 2 and entry["n_observed_directions"] == 1
    assert entry["complete"] is False and entry["missing_directions"] == [[0.0, 1.0, 0.0]]
    assert adc["min_detectable_mm"] == 2.0
    assert "c3" in adc["incomplete_unit_ids"]
    assert any("min_detectable_mm_uncovered_units" in reason for reason in adc["reasons"])


def test_v0_4_unusable_scale_fails_primary_calibration():
    """要求 3：某 unit 的 sigma1.5.usable=false → 主指标校准失败（不得当作可用数值）。"""
    records = _synth_records(_three_unit_spec(), unusable_units={"c1"})
    calibration = aq.summarize_calibration(records, cfg=_synth_cal_cfg(), metrics_cfg=_synth_metric_cfg())
    assert calibration["passed"] is False
    adc = calibration["per_pair"]["T2W-ADC"]["per_metric"][SYNTH_METRIC]
    detail = next(d for d in adc["unit_details"] if d["case_id"] == "c1")
    assert detail["complete"] is False
    assert any("usable=false" in str(item.get("reason")) for item in detail["invalid_conditions"])
    assert "c1" in adc["incomplete_unit_ids"]
    assert detail["n_zero_observed"] == 0  # usable=false 的记录视为无效，不进入零位移参考


def test_v0_4_decide_case_unusable_metric_is_insufficient_evidence():
    """要求 4：真实病例指标数值有限但 usable=false → INSUFFICIENT_EVIDENCE（记录原因）。"""
    calibration, thresholds, mcfg, _ = _three_unit_calibration()
    metrics = _usable_metrics(**{SYNTH_METRIC: 0.30})
    metrics[SYNTH_USABLE_KEY] = False
    decision = aq.decide_case(metrics, thresholds, calibration, pair="T2W-ADC", metrics_cfg=mcfg)
    assert decision["status"] == aq.CASE_INSUFFICIENT
    check = decision["checks"][0]
    assert check["decision"] == "UNAVAILABLE"
    assert check["metric_usable"] is False
    assert "usable=false" in str(check["metric_unusable_reason"])
    assert "边缘体素不足" in str(check["metric_unusable_reason"])
    assert "metric_unusable" in check["reason"]
    # 一致性指标 unusable → SKIPPED（不得支持 ACCEPTABLE/FLAGGED）
    mcfg2 = {
        "primary_metric": SYNTH_METRIC,
        "primary_direction": "lower_is_worse",
        "consistency_metrics": [{"name": f"{SYNTH_SCALE}.c", "direction": "higher_is_worse"}],
    }
    calibration2 = aq.summarize_calibration(_synth_records(_three_unit_spec()), cfg=_synth_cal_cfg(), metrics_cfg=mcfg2)
    thresholds2 = aq.derive_thresholds(calibration2, metrics_cfg=mcfg2, thresholds_cfg=_synth_threshold_cfg())
    metrics2 = _usable_metrics(**{SYNTH_METRIC: 0.30, f"{SYNTH_SCALE}.c": 9.0})
    metrics2[f"{SYNTH_SCALE}.usable"] = False
    decision2 = aq.decide_case(metrics2, thresholds2, calibration2, pair="T2W-ADC", metrics_cfg=mcfg2)
    consistency_check = next(c for c in decision2["checks"] if c["metric"] == f"{SYNTH_SCALE}.c")
    assert consistency_check["decision"] == "SKIPPED"
    assert "不得用于支持" in consistency_check["reason"] or "仅记录" in consistency_check["reason"]


def test_v0_4_missing_usable_field_is_fail_closed():
    """要求 5：usable 字段缺失 / 非布尔 → fail-closed（不得默认为 true）。"""
    calibration, thresholds, mcfg, _ = _three_unit_calibration()
    for metrics in (
        {SYNTH_METRIC: 0.30},                       # 完全缺失
        {SYNTH_METRIC: 0.30, SYNTH_USABLE_KEY: 1},  # 非布尔
        {SYNTH_METRIC: 0.30, SYNTH_USABLE_KEY: None},
    ):
        decision = aq.decide_case(metrics, thresholds, calibration, pair="T2W-ADC", metrics_cfg=mcfg)
        assert decision["status"] == aq.CASE_INSUFFICIENT, metrics
        assert decision["checks"][0]["metric_usable"] is False
        assert decision["checks"][0]["metric_unusable_reason"]
    # 无 sigma 前缀的指标名同样 fail-closed
    assert aq.metric_usable({}, "m")[0] is False
    assert aq.metric_usable_key("sigma1.5.x") == "sigma1.5.usable"
    assert aq.metric_usable_key("sigma1.x") == "sigma1.usable"
    assert aq.metric_usable_key("m") is None


def test_v0_4_insufficient_complete_units_fails():
    """要求 6：每 pair 只有 1 或 2 个 complete unit（阈值 3）→ 失败。"""
    spec = _three_unit_spec()
    for n_units in (1, 2):
        subset = {
            "T2W-ADC": dict(list(spec["T2W-ADC"].items())[:n_units]),
            "T2W-HBV": dict(list(spec["T2W-HBV"].items())[:n_units]),
        }
        records = _synth_records(subset)
        calibration = aq.summarize_calibration(records, cfg=_synth_cal_cfg(), metrics_cfg=_synth_metric_cfg())
        assert calibration["passed"] is False, n_units
        adc = calibration["per_pair"]["T2W-ADC"]["per_metric"][SYNTH_METRIC]
        assert adc["n_units_complete"] == n_units
        assert any("insufficient_complete_units" in reason for reason in adc["reasons"])
        # 把门槛降到实际 unit 数后应当通过（证明失败来自数量门而非其它）
        relaxed = aq.summarize_calibration(
            records,
            cfg=_synth_cal_cfg(min_complete_units_per_pair=n_units),
            metrics_cfg=_synth_metric_cfg(),
        )
        assert relaxed["passed"] is True, n_units


def test_v0_4_counts_and_condition_audit_fields_are_reported():
    """要求 6/8：计数、incomplete_unit_ids、missing/invalid conditions 与逐距离覆盖正确。"""
    records = _synth_records(
        _three_unit_spec(),
        drop_conditions={("c1", 2.0, 1), ("c2", 0.0, 2)},
        unusable_units={"c3"},
    )
    mcfg = _synth_metric_cfg()
    calibration = aq.summarize_calibration(records, cfg=_synth_cal_cfg(), metrics_cfg=mcfg)
    adc = calibration["per_pair"]["T2W-ADC"]["per_metric"][SYNTH_METRIC]
    assert adc["n_units_total"] == 3 and adc["n_units_complete"] == 0
    assert adc["n_units_at_min_detectable"] == 0
    assert sorted(adc["incomplete_unit_ids"]) == ["c1", "c2", "c3"]
    assert calibration["incomplete_unit_ids"] == ["c1", "c2", "c3"]
    assert calibration["per_pair"]["T2W-ADC"]["passed"] is False
    details = {d["case_id"]: d for d in adc["unit_details"]}
    assert "2|dir1" in details["c1"]["missing_conditions"]
    assert "zero:1 条缺失（expected=3, observed=2）" in details["c2"]["missing_conditions"]
    assert any("usable=false" in str(i.get("reason")) for i in details["c3"]["invalid_conditions"])
    assert details["c1"]["by_distance"]["2"]["complete"] is False
    assert details["c2"]["by_distance"]["2"]["complete"] is True
    assert details["c2"]["n_zero_expected"] == 3 and details["c2"]["n_zero_observed"] == 2
    # 该 pair 失败但输出完整（不得静默丢弃）
    thresholds = aq.derive_thresholds(calibration, metrics_cfg=mcfg, thresholds_cfg=_synth_threshold_cfg())
    entry = thresholds["metrics"][SYNTH_METRIC]["by_pair"]["T2W-ADC"]
    assert entry["n_units_total"] == 3 and entry["n_units_complete"] == 0
    assert entry["incomplete_unit_ids"] == ["c1", "c2", "c3"]


def test_v0_4_pairs_cannot_borrow_units():
    """要求 9：ADC/HBV 完全独立，不得跨 pair 补足 unit。"""
    spec = {
        "T2W-ADC": dict(list(_three_unit_spec()["T2W-ADC"].items())[:3]),
        "T2W-HBV": dict(list(_three_unit_spec()["T2W-HBV"].items())[:2]),
    }
    calibration = aq.summarize_calibration(_synth_records(spec), cfg=_synth_cal_cfg(), metrics_cfg=_synth_metric_cfg())
    assert calibration["passed"] is False
    adc = calibration["per_pair"]["T2W-ADC"]
    hbv = calibration["per_pair"]["T2W-HBV"]
    assert adc["passed"] is True and adc["n_units_complete"] == 3 and adc["incomplete_unit_ids"] == []
    assert hbv["passed"] is False and hbv["n_units_complete"] == 2
    assert any("insufficient_complete_units" in reason for reason in hbv["reasons"])
    thresholds = aq.derive_thresholds(calibration, metrics_cfg=_synth_metric_cfg(), thresholds_cfg=_synth_threshold_cfg())
    assert thresholds["metrics"][SYNTH_METRIC]["by_pair"]["T2W-ADC"]["n_units_complete"] == 3
    assert thresholds["metrics"][SYNTH_METRIC]["by_pair"]["T2W-HBV"]["n_units_complete"] == 2


def test_v0_4_missing_distance_and_zero_repeats_and_nonfinite_fail_closed():
    """要求（fail-closed）：缺距离 / 缺零位移重复 / 非有限值 一律失败，不得静默缩小分母。"""
    mcfg, ccfg = _synth_metric_cfg(), _synth_cal_cfg()
    # (a) 全部 unit 的 2 mm 距离缺失（cfg 期望 2.0 在 displacements 中）
    records_a = _synth_records(_three_unit_spec(), distances=(0.0, 0.5, 1.0, 3.0))
    calibration_a = aq.summarize_calibration(records_a, cfg=ccfg, metrics_cfg=mcfg)
    assert calibration_a["passed"] is False
    adc_a = calibration_a["per_pair"]["T2W-ADC"]["per_metric"][SYNTH_METRIC]
    assert adc_a["n_units_complete"] == 0
    assert any("incomplete_unit_condition_grid" in reason for reason in adc_a["reasons"])
    detail_a = adc_a["unit_details"][0]
    assert "2|dir0" in detail_a["missing_conditions"] and "2|dir1" in detail_a["missing_conditions"]
    # (b) 只有零位移记录 → 无非零位移 unit
    zero_only = [r for r in _synth_records(_three_unit_spec()) if float(r["distance_mm"]) == 0.0]
    calibration_b = aq.summarize_calibration(zero_only, cfg=ccfg, metrics_cfg=mcfg)
    assert calibration_b["passed"] is False
    assert calibration_b["per_pair"]["T2W-ADC"]["per_metric"][SYNTH_METRIC]["n_units_complete"] == 0
    # (c) 零位移全为 NaN → 无效记录（不得当成 0）
    broken = _synth_records(
        {"T2W-ADC": {"c1": (0.35, 0.01), "c2": (0.30, 0.01), "c3": (0.40, 0.01)},
         "T2W-HBV": {"c1": (0.12, 0.01), "c2": (0.16, 0.01), "c3": (0.10, 0.01)}},
        nan_conditions={(c, 0.0, i) for c in ("c1", "c2", "c3") for i in range(3)},
    )
    calibration_c = aq.summarize_calibration(broken, cfg=ccfg, metrics_cfg=mcfg)
    assert calibration_c["passed"] is False
    assert calibration_c["per_pair"]["T2W-ADC"]["per_metric"][SYNTH_METRIC]["n_units_complete"] == 0


# --------------------------------------------------------------------------- schema / 兼容性守卫
def test_cli_schema_envelope_and_output_guard(cli, tmp_path):
    """要求 11：输出必须带 schema；读取 v0.2/v0.3 旧输出必须显式失败（禁止静默解释）。"""
    envelope = cli.schema_envelope(
        {"payload": 1},
        protocol={"id": "G0-R-AUTOMATED", "version": "draft-0.3", "status": "DRAFT"},
        protocol_hash="deadbeef",
    )
    assert envelope["schema_version"] == aq.OUTPUT_SCHEMA_VERSION
    assert envelope["payload"] == 1 and envelope["draft"] is True
    assert envelope["calibration_grouping"] == ["pair", "case_id"]
    good = tmp_path / "good.json"
    good.write_text(json.dumps(envelope), encoding="utf-8")
    assert cli.load_output_json(good)["payload"] == 1
    # 无 schema 的旧产物（v0.2 之前 / 手工 JSON）
    legacy_plain = tmp_path / "legacy_plain.json"
    legacy_plain.write_text(json.dumps({"passed": False, "per_metric": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="schema 不兼容"):
        cli.load_output_json(legacy_plain)
    # v0.2 / v0.3 输出必须被显式拒绝（不得按 v0.4 schema 解释）
    for old_schema in ("g0-r-automated/0.2", "g0-r-automated/0.3"):
        legacy = tmp_path / f"legacy_{old_schema.split('/')[1].replace('.', '_')}.json"
        legacy.write_text(
            json.dumps({"schema_version": old_schema, "passed": False, "per_metric": {}, "per_pair": {}}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="schema 不兼容"):
            cli.load_output_json(legacy)
        with pytest.raises(SystemExit, match="output_schema_version"):
            cli.assert_schema_version({"protocol": {"output_schema_version": old_schema}})
    assert (
        cli.assert_schema_version({"protocol": {"output_schema_version": aq.OUTPUT_SCHEMA_VERSION}})
        == aq.OUTPUT_SCHEMA_VERSION
    )
