"""G0-R 抽样与 schema 测试（合成 manifest 行；不读真实数据）。

覆盖：
- 确定性分层抽样（中心/阳阴/geometry_suspect 全纳入/partial/极端 FOV/间距跨度）；
- n_cases=null → 最小必选集合；n_cases 给定 → 确定性补齐；
- 盲审模板行数与字段；指标 schema 校验的错误分支。
"""
from __future__ import annotations

import copy

import pytest

from zonal_reliability_fusion.protocols import g0_r
from zonal_reliability_fusion.protocols.common import ProtocolConfigError

SAMPLING_CFG = {
    "deterministic": True,
    "n_cases": None,
    "strata": {
        "center": {"min_per_level": 1},
        "lesion_status": {"min_per_level": 1},
        "geometry_suspect": {"include_all": True},
        "partial_overlap": {"min_cases": 2},
        "extreme_fov": {"min_per_tail": 1},
        "spacing_span": {"min_per_tail": 1},
    },
}


def make_row(
    case_id: str,
    *,
    center: str,
    cspca: str,
    geometry: str = "same_physical_space_different_grid",
    t2w_volume: float = 1000.0,
    adc_planar: float = 1.0,
    t2w_planar: float = 0.5,
) -> dict:
    side = t2w_volume ** (1 / 3)
    return {
        "case_id": case_id,
        "patient_id": case_id.split("_")[0],
        "study_id": case_id.split("_")[1] if "_" in case_id else "1",
        "center": center,
        "case_csPCa": cspca,
        "geometry_status": geometry,
        "t2w_extent": f"{side:.3f};{side:.3f};{side:.3f}",
        "adc_extent": "10;10;10",
        "hbv_extent": "10;10;10",
        "t2w_spacing": f"{t2w_planar};{t2w_planar};3.0",
        "adc_spacing": f"{adc_planar};{adc_planar};3.0",
        "hbv_spacing": f"{adc_planar};{adc_planar};3.0",
    }


def make_manifest() -> list[dict]:
    rows = [
        make_row("c01_1", center="RUMC", cspca="YES", t2w_volume=100.0),          # 最小 FOV
        make_row("c02_1", center="RUMC", cspca="NO", adc_planar=0.8),             # ADC 间距最小
        make_row("c03_1", center="PCNN", cspca="YES", adc_planar=2.5),            # ADC 间距最大
        make_row("c04_1", center="PCNN", cspca="NO", t2w_volume=8000.0),          # 最大 FOV
        make_row("c05_1", center="ZGT", cspca="YES"),
        make_row("c06_1", center="ZGT", cspca="NO"),
        make_row("c07_1", center="RUMC", cspca="YES", geometry="geometry_suspect"),
        make_row("c08_1", center="PCNN", cspca="NO", geometry="geometry_suspect"),
        make_row("c09_1", center="ZGT", cspca="YES", geometry="partial_physical_overlap"),
        make_row("c10_1", center="RUMC", cspca="NO", geometry="partial_physical_overlap"),
        make_row("c11_1", center="PCNN", cspca="YES", geometry="partial_physical_overlap",
                 t2w_planar=0.3),                                                # T2W 间距最小
        make_row("c12_1", center="ZGT", cspca="NO", t2w_planar=0.63),            # T2W 间距最大
    ]
    return rows


def test_deterministic_minimum_required_sampling():
    manifest = make_manifest()
    result = g0_r.plan_sampling(copy.deepcopy(manifest), SAMPLING_CFG)
    assert result.n_cases_source == "minimum_required"
    assert result.n_cases_configured is None
    ids = [row["case_id"] for row in result.rows]
    # geometry_suspect 全部纳入
    assert "c07_1" in ids and "c08_1" in ids
    # 各分层均被命中
    for stratum in ("center", "lesion_status", "geometry_suspect", "partial_overlap", "extreme_fov", "spacing_span"):
        assert result.per_stratum.get(stratum, 0) >= 1, stratum
    # extreme_fov 两端：最小 FOV（c01）与最大 FOV（c04）都在集合里
    assert "c01_1" in ids and "c04_1" in ids
    # spacing 两端（ADC 最大 c03 / T2W 两端 c11、c12）
    assert "c03_1" in ids and "c11_1" in ids and "c12_1" in ids
    # 确定性：重复调用完全一致
    again = g0_r.plan_sampling(copy.deepcopy(manifest), SAMPLING_CFG)
    assert again.rows == result.rows and again.per_stratum == result.per_stratum


def test_configured_n_cases_fills_deterministically():
    manifest = make_manifest()
    cfg = copy.deepcopy(SAMPLING_CFG)
    cfg["n_cases"] = 12
    result = g0_r.plan_sampling(copy.deepcopy(manifest), cfg)
    assert result.n_cases_configured == 12
    assert result.n_cases_source == "configured"
    assert result.total == 12
    filled = [row for row in result.rows if "fill_deterministic" in row["strata"]]
    assert len(filled) == 1 and filled[0]["case_id"] == "c06_1"


def test_manifest_decision_stays_null():
    result = g0_r.plan_sampling(make_manifest(), SAMPLING_CFG)
    manifest = result.as_manifest()
    assert manifest["decision"] is None  # 工具不得填写配准决策
    assert manifest["total_cases"] == result.total
    assert all("strata" in case for case in manifest["cases"])


def test_blind_review_rows_shape():
    result = g0_r.plan_sampling(make_manifest(), SAMPLING_CFG)
    rows = g0_r.build_blind_review_rows(result)
    assert len(rows) == result.total * len(g0_r.PAIRS)
    for row in rows:
        assert row["reader_a_level"] == "" and row["final_level"] == ""  # 模板不预填
    pairs = {row["pair"] for row in rows}
    assert pairs == set(g0_r.PAIRS)


def test_parse_vector3_errors():
    assert g0_r.parse_vector3("1.0;2.0;3.0", field_name="x") == (1.0, 2.0, 3.0)
    assert g0_r.parse_vector3("", field_name="x") is None
    with pytest.raises(ProtocolConfigError, match="3 个分号"):
        g0_r.parse_vector3("1.0;2.0", field_name="x")
    with pytest.raises(ProtocolConfigError, match="格式不支持"):
        g0_r.parse_vector3(123, field_name="x")


def test_alignment_metrics_schema_validation():
    good = [
        {"case_id": "c01_1", "pair": "T2W-ADC", "metric": "landmark_displacement_mm",
         "value": "1.25", "unit": "mm", "method": "boundary", "measured_by": "reader_a", "notes": ""},
        {"case_id": "c01_1", "pair": "T2W-HBV", "metric": "mi", "value": "0.31", "unit": "au",
         "method": "aux", "measured_by": "tool", "notes": "auxiliary_only"},
    ]
    assert g0_r.validate_alignment_metrics_rows(good) == []

    bad = [
        {"case_id": "", "pair": "T2W-ADC", "metric": "landmark_displacement_mm",
         "value": "1.0", "unit": "mm"},
        {"case_id": "c02_1", "pair": "T2W-XXX", "metric": "nmi", "value": "0.5", "unit": "au"},
        {"case_id": "c02_1", "pair": "T2W-ADC", "metric": "unknown_metric", "value": "0.5", "unit": "mm"},
        {"case_id": "c02_1", "pair": "T2W-ADC", "metric": "landmark_displacement_mm",
         "value": "not_a_number", "unit": "mm"},
        {"case_id": "c02_1", "pair": "T2W-ADC", "metric": "landmark_displacement_mm",
         "value": "1.0", "unit": "au"},
    ]
    errors = g0_r.validate_alignment_metrics_rows(bad)
    assert len(errors) == 5
    assert any("case_id" in e for e in errors)
    assert any("pair" in e for e in errors)
    assert any("metric" in e for e in errors)
    assert any("不是数值" in e for e in errors)
    assert any("单位" in e for e in errors)
    assert g0_r.validate_alignment_metrics_rows([]) == ["指标文件没有数据行"]


def test_schema_document_marks_auxiliary_and_no_decision():
    doc = g0_r.metrics_schema_document()
    assert doc["decision"] is None
    assert set(doc["auxiliary_metrics"]) == {"mi", "nmi"}
    assert doc["metric_units"]["landmark_displacement_mm"] == "mm"
    assert "raw_cross_modality_NCC" not in doc["metric_allowed"]
