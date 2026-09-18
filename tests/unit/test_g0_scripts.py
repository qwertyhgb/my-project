"""G0 三个入口脚本的装配测试（合成配置/临时文件；不读真实医学数据、不初始化 CUDA）。

覆盖：
- `--help` 可用；
- G0-R：`--dry-run` 不写文件；正常运行生成抽样清单与盲审表；`--validate-metrics` 校验 schema；
- G0-E：`--dry-run` 不写文件；正常运行生成登记表与待补证据清单；
- G0-SAP：`--check-freeze` 报告阻塞、`--fail-if-not-ready` 退出码 3；正常生成冻结包模板。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

from zonal_reliability_fusion.protocols import common as pc

PROJECT_ROOT = Path(__file__).resolve().parents[2]
G0_R_SCRIPT = PROJECT_ROOT / "scripts/audit/audit_picai_alignment_qc.py"
G0_E_SCRIPT = PROJECT_ROOT / "scripts/audit/audit_g0e_candidates.py"
G0_SAP_SCRIPT = PROJECT_ROOT / "scripts/evaluate/prepare_g0_sap_freeze.py"


def load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def g0_r_script():
    return load_script(G0_R_SCRIPT, "g0_r_script_under_test")


@pytest.fixture(scope="module")
def g0_e_script():
    return load_script(G0_E_SCRIPT, "g0_e_script_under_test")


@pytest.fixture(scope="module")
def g0_sap_script():
    return load_script(G0_SAP_SCRIPT, "g0_sap_script_under_test")


# --------------------------------------------------------------------------- G0-R
MANIFEST_ROWS = [
    {
        "case_id": "c01_1", "patient_id": "c01", "study_id": "1", "center": "RUMC", "case_csPCa": "YES",
        "geometry_status": "geometry_suspect", "t2w_extent": "40;40;25", "adc_extent": "40;40;25",
        "hbv_extent": "40;40;25", "t2w_spacing": "0.5;0.5;3.0", "adc_spacing": "1.0;1.0;3.0",
        "hbv_spacing": "1.0;1.0;3.0",
    },
    {
        "case_id": "c02_1", "patient_id": "c02", "study_id": "1", "center": "PCNN", "case_csPCa": "NO",
        "geometry_status": "same_physical_space_different_grid", "t2w_extent": "60;60;30",
        "adc_extent": "60;60;30", "hbv_extent": "60;60;30", "t2w_spacing": "0.6;0.6;3.0",
        "adc_spacing": "0.9;0.9;3.0", "hbv_spacing": "0.9;0.9;3.0",
    },
]


def write_g0r_workspace(tmp_path: Path) -> tuple[Path, Path]:
    manifest = tmp_path / "manifest.csv"
    columns = list(MANIFEST_ROWS[0].keys())
    import csv as _csv

    with open(manifest, "w", newline="", encoding="utf-8") as fh:
        writer = _csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(MANIFEST_ROWS)
    config = tmp_path / "g0r.yaml"
    doc = {
        "protocol": {"id": "G0-R", "version": "draft-0.1", "status": "DRAFT", "sees_model_predictions": False},
        "inputs": {"manifest": str(manifest)},
        "sampling": {"n_cases": None, "strata": {"center": {"min_per_level": 1}, "lesion_status": {"min_per_level": 1},
                                                 "geometry_suspect": {"include_all": True}, "partial_overlap": {"min_cases": 1},
                                                 "extreme_fov": {"min_per_tail": 1}, "spacing_span": {"min_per_tail": 1}}},
        "evaluation": {"pairs": ["T2W-ADC", "T2W-HBV"], "units": "mm"},
        "readers": {"record_fields": ["case_id", "pair", "reader_a_level", "reader_b_level", "final_level",
                                      "disagreement", "adjudication_note"]},
        "outputs": {"dir": str(tmp_path / "out")},
    }
    config.write_text(yaml.safe_dump(doc, allow_unicode=True), encoding="utf-8")
    return config, manifest


def test_g0r_help(g0_r_script, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["audit_picai_alignment_qc.py", "--help"])
    with pytest.raises(SystemExit) as exc:
        g0_r_script.main()
    assert exc.value.code == 0


def test_g0r_dry_run_writes_nothing(g0_r_script, tmp_path, monkeypatch):
    config, _ = write_g0r_workspace(tmp_path)
    monkeypatch.setattr(pc, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["x", "--config", str(config), "--dry-run"])
    g0_r_script.main()
    assert not (tmp_path / "out").exists()


def test_g0r_generates_outputs(g0_r_script, tmp_path, monkeypatch):
    config, manifest = write_g0r_workspace(tmp_path)
    out_dir = tmp_path / "run"
    monkeypatch.setattr(pc, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["x", "--config", str(config), "--out-dir", str(out_dir), "--no-progress"])
    g0_r_script.main()
    for name in ("sampled_cases.csv", "sampling_manifest.json", "blind_review_sheet.csv",
                 "alignment_metrics_schema.json", "run_metadata.json"):
        assert (out_dir / name).is_file(), name
    import json

    manifest_doc = json.loads((out_dir / "sampling_manifest.json").read_text(encoding="utf-8"))
    assert manifest_doc["sampling"]["decision"] is None
    assert manifest_doc["status"] == "PENDING"
    assert manifest_doc["inputs"]["manifest_sha256"] == pc.sha256_file(manifest)
    rows = (out_dir / "blind_review_sheet.csv").read_text(encoding="utf-8").splitlines()
    assert rows[0].startswith("case_id,pair,reader_a_level")
    assert len(rows) == 1 + 2 * 2  # 2 病例 × 2 评价对


def test_g0r_validate_metrics(g0_r_script, tmp_path, monkeypatch, capsys):
    good = tmp_path / "metrics.csv"
    good.write_text(
        "case_id,pair,metric,value,unit,method,measured_by,notes\n"
        "c01_1,T2W-ADC,landmark_displacement_mm,1.2,mm,boundary,reader_a,\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["x", "--validate-metrics", str(good)])
    with pytest.raises(SystemExit) as ok_exc:  # 合法：退出码 0
        g0_r_script.main()
    assert ok_exc.value.code == 0
    assert "schema OK" in capsys.readouterr().out

    bad = tmp_path / "bad.csv"
    bad.write_text(
        "case_id,pair,metric,value,unit,method,measured_by,notes\n"
        "c01_1,T2W-ADC,landmark_displacement_mm,abc,mm,boundary,reader_a,\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["x", "--validate-metrics", str(bad)])
    with pytest.raises(SystemExit) as exc:
        g0_r_script.main()
    assert exc.value.code == 1


# --------------------------------------------------------------------------- G0-E
def build_fake_external_dataset(root: Path) -> Path:
    case_dir = root / "train/case_001"
    case_dir.mkdir(parents=True)
    for name in ("t2.nii.gz", "adc.nii.gz", "dwi.nii.gz", "t2_anatomy_reader1.nii.gz", "empty.nii.gz"):
        (case_dir / name).write_bytes(b"")
    (root / "train.csv").write_text("ID,t2,adc,dwi\n1,train/case_001/t2.nii.gz,"
                                    "train/case_001/adc.nii.gz,train/case_001/dwi.nii.gz\n", encoding="utf-8")
    return root


def write_g0e_workspace(tmp_path: Path) -> Path:
    external = build_fake_external_dataset(tmp_path / "ext")
    config = tmp_path / "g0e.yaml"
    doc = {
        "protocol": {"id": "G0-E", "version": "draft-0.1", "status": "DRAFT", "sees_model_predictions": False},
        "candidates": [
            {"name": "FakeExt", "priority": 1, "local_root": str(external), "license": "CC-BY"},
            {"name": "Excluded", "priority": 3, "local_root": None, "excluded_as_lesion_test": True},
        ],
        "outputs": {"dir": str(tmp_path / "out_e")},
    }
    config.write_text(yaml.safe_dump(doc, allow_unicode=True), encoding="utf-8")
    return config


def test_g0e_help(g0_e_script, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["audit_g0e_candidates.py", "--help"])
    with pytest.raises(SystemExit) as exc:
        g0_e_script.main()
    assert exc.value.code == 0


def test_g0e_dry_run_and_outputs(g0_e_script, tmp_path, monkeypatch, capsys):
    config = write_g0e_workspace(tmp_path)
    out_dir = tmp_path / "run_e"
    monkeypatch.setattr(pc, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["x", "--config", str(config), "--dry-run"])
    g0_e_script.main()
    assert "verdict / frozen_decision 保持 null" in capsys.readouterr().out
    assert not out_dir.exists()

    monkeypatch.setattr(sys, "argv", ["x", "--config", str(config), "--out-dir", str(out_dir), "--no-progress"])
    g0_e_script.main()
    for name in ("candidate_registry.csv", "evidence_status.json", "inventory_summary.json",
                 "missing_evidence.md", "audit_report.md", "run_metadata.json"):
        assert (out_dir / name).is_file(), name
    import json

    status = json.loads((out_dir / "evidence_status.json").read_text(encoding="utf-8"))
    assert status["frozen_decision"] is None
    fake = next(c for c in status["candidates"] if c["name"] == "FakeExt")
    assert fake["fields"]["license"]["status"] == "VERIFIED"
    assert fake["fields"]["patient_overlap_with_picai"]["status"] == "UNKNOWN"
    assert "patient_overlap_with_picai" in fake["missing_fields"]
    excluded = next(c for c in status["candidates"] if c["name"] == "Excluded")
    assert excluded["fields"]["verdict"]["status"] == "CONFIGURED_EXCLUDED"


# --------------------------------------------------------------------------- G0-SAP
def test_g0sap_help(g0_sap_script, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prepare_g0_sap_freeze.py", "--help"])
    with pytest.raises(SystemExit) as exc:
        g0_sap_script.main()
    assert exc.value.code == 0


def test_g0sap_check_freeze_not_ready(g0_sap_script, monkeypatch, capsys):
    """仓库配置当前为 DRAFT：必须报 NOT READY，且不得写文件、不得判 PASS。"""
    monkeypatch.setattr(sys, "argv", ["x", "--check-freeze", "--fail-if-not-ready"])
    with pytest.raises(SystemExit) as exc:
        g0_sap_script.main()
    assert exc.value.code == 3
    out = capsys.readouterr().out
    assert "NOT READY" in out and "PENDING" in out


def test_g0sap_generates_templates(g0_sap_script, tmp_path, monkeypatch):
    out_dir = tmp_path / "freeze"
    monkeypatch.setattr(pc, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(g0_sap_script, "REGISTERED_FILES", ())
    monkeypatch.setattr(
        sys, "argv",
        ["x", "--out-dir", str(out_dir), "--no-progress"],
    )
    g0_sap_script.main()
    for name in ("evaluation_config.sha256", "sap_a_method_hash.txt", "sap_a_snapshot.json",
                 "power_memo_template.md", "freeze_checklist.md",
                 "stats_config.template.json", "run_metadata.json"):
        assert (out_dir / name).is_file(), name
    assert "TODO" in (out_dir / "power_memo_template.md").read_text(encoding="utf-8")
    assert "sha256:" in (out_dir / "sap_a_method_hash.txt").read_text(encoding="utf-8")
    checklist = (out_dir / "freeze_checklist.md").read_text(encoding="utf-8")
    assert "冻结阻塞项" in checklist and "power_memo.status" in checklist
    assert "SAP-A 方法块哈希" in checklist and "sap_b_results" in checklist


def test_g0sap_frozen_with_blockers_refused(g0_sap_script, tmp_path, monkeypatch):
    """SAP-A 已声明冻结但阻塞项非空 → 拒绝生成冻结载体（不得把未完成配置标记为冻结）。"""
    from zonal_reliability_fusion.protocols import g0_sap

    config = tmp_path / "g0sap_frozen.yaml"
    doc = yaml.safe_load((PROJECT_ROOT / "configs/protocols/g0_sap.yaml").read_text(encoding="utf-8"))
    # 构造"自洽但未就绪"的 SAP-A 冻结声明：protocol.status 与 sap_a.status 一致，且记录方法块哈希；
    # 其余 SAP-A 必填字段仍为空 → 冻结就绪检查必须报阻塞并拒绝生成冻结载体。
    doc["protocol"]["status"] = "FROZEN"
    doc["sap_phases"]["sap_a"]["status"] = "FROZEN"
    doc["sap_phases"]["sap_a"]["method_sha256"] = g0_sap.sap_a_method_hash(doc)
    config.write_text(yaml.safe_dump(doc, allow_unicode=True), encoding="utf-8")
    monkeypatch.setattr(pc, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        sys, "argv",
        ["x", "--config", str(config), "--out-dir", str(tmp_path / "never"), "--no-progress"],
    )
    with pytest.raises(SystemExit) as exc:
        g0_sap_script.main()
    assert "拒绝生成冻结载体" in str(exc.value)
    assert not (tmp_path / "never").exists()
