"""G0-E 清点与证据状态测试（合成目录/CSV；不读真实医学数据）。

覆盖：
- 文件清单级清点（病例目录数、模态/标注文件计数、CSV 表头与行数、empty 占位）；
- 证据状态：配置已填 → VERIFIED；未填 → UNKNOWN；工具事实 → AUTO_ONLY；
- 候选排除配置 → CONFIGURED_EXCLUDED；
- 待补证据清单与报告渲染（不得出现推断出的 verdict）。
"""
from __future__ import annotations

from pathlib import Path

from zonal_reliability_fusion.protocols import g0_e


def build_fake_dataset(root: Path) -> Path:
    """构造一个“Prostate158 型”假数据集（空文件 + CSV 表头，无影像内容）。"""
    train = root / "train/prostate158_train"
    for case in ("001", "002", "003"):
        case_dir = train / "train" / case
        case_dir.mkdir(parents=True)
        for name in ("t2.nii.gz", "adc.nii.gz", "dwi.nii.gz", "t2_anatomy_reader1.nii.gz"):
            (case_dir / name).write_bytes(b"")
        (case_dir / "empty.nii.gz").write_bytes(b"")
        # 仅 001 例有肿瘤标注
    (train / "train" / "001" / "t2_tumor_reader1.nii.gz").write_bytes(b"")
    (train / "train.csv").write_text(
        "ID,t2,adc,dwi,t2_anatomy_reader1,t2_tumor_reader1\n"
        "1,train/001/t2.nii.gz,train/001/adc.nii.gz,train/001/dwi.nii.gz,"
        "train/001/t2_anatomy_reader1.nii.gz,train/001/t2_tumor_reader1.nii.gz\n"
        "2,train/002/t2.nii.gz,train/002/adc.nii.gz,train/002/dwi.nii.gz,"
        "train/002/t2_anatomy_reader1.nii.gz,\n",
        encoding="utf-8",
    )
    return root


def test_inventory_dataset_counts(tmp_path: Path):
    root = build_fake_dataset(tmp_path / "ds")
    inv = g0_e.inventory_dataset(root)
    assert inv["exists"] is True
    assert inv["n_case_dirs"] == 3
    assert inv["modality_files"]["t2"] >= 3
    assert inv["modality_files"]["adc"] >= 3
    assert inv["modality_files"]["dwi"] >= 3
    assert inv["annotation_files"]["anatomy"] >= 3
    assert inv["annotation_files"]["tumor"] >= 1
    assert inv["empty_placeholder_count"] == 3
    csv_entries = {Path(item["path"]).name: item for item in inv["csv_files"]}
    assert "train.csv" in csv_entries
    assert csv_entries["train.csv"]["header"][0] == "ID"
    assert csv_entries["train.csv"]["n_rows"] == 2
    assert "不作为判定依据" in inv["notes"]


def test_inventory_missing_root(tmp_path: Path):
    inv = g0_e.inventory_dataset(tmp_path / "nope")
    assert inv["exists"] is False
    assert inv["errors"]


def test_audit_candidate_unknown_not_inferred(tmp_path: Path):
    root = build_fake_dataset(tmp_path / "ds")
    inv = g0_e.inventory_dataset(root)
    candidate = {
        "name": "Fake158",
        "priority": 1,
        "local_root": str(root),
        "license": "CC-BY-4.0",           # 已填写 → VERIFIED
        "n_cases": None,                  # 未填写 → UNKNOWN
        "verdict": None,
    }
    audit = g0_e.audit_candidate(candidate, inv)
    assert audit["fields"]["license"]["status"] == "VERIFIED"
    assert audit["fields"]["n_cases"]["status"] == g0_e.UNKNOWN
    assert audit["fields"]["patient_overlap_with_picai"]["status"] == g0_e.UNKNOWN
    assert "patient_overlap_with_picai" in audit["missing"]
    # 自动事实只进 AUTO_ONLY，不覆盖人工字段
    assert audit["auto_only"]["n_case_dirs"] == 3
    assert "AUTO_ONLY" in audit["auto_only"]["note"]
    # 不得把 AUTO 事实写成 n_cases
    assert audit["fields"]["n_cases"]["value"] is None


def test_audit_candidate_excluded_is_config_not_pass():
    candidate = {"name": "ProstateX", "priority": 3, "excluded_as_lesion_test": True}
    audit = g0_e.audit_candidate(candidate, None)
    assert audit["fields"]["verdict"]["status"] == "CONFIGURED_EXCLUDED"
    rows = g0_e.build_missing_evidence_rows([audit])
    assert any(row["status"] == "CONFIGURED_EXCLUDED" for row in rows)
    # 被排除候选中仍未被推断为 eligible
    assert all(row["status"] != "VERIFIED" for row in rows)


def test_missing_evidence_rows_have_hints():
    audit = g0_e.audit_candidate({"name": "X"}, None)
    rows = g0_e.build_missing_evidence_rows([audit])
    fields = {row["missing_field"] for row in rows}
    assert "license" in fields
    assert all(row["required_evidence"] for row in rows)


def test_report_render_marks_unknown(tmp_path: Path):
    root = build_fake_dataset(tmp_path / "ds")
    inv = g0_e.inventory_dataset(root)
    audits = [
        g0_e.audit_candidate({"name": "A", "local_root": str(root), "license": "MIT"}, inv),
        g0_e.audit_candidate({"name": "B", "excluded_as_lesion_test": True}, None),
    ]
    report = g0_e.render_g0e_report_markdown(
        audits, protocol_version="draft-0.1", protocol_status="DRAFT", generated_at="2026-09-15T00:00:00"
    )
    assert "UNKNOWN" in report
    assert "CONFIGURED_EXCLUDED" in report
    # 工具不产生可用性结论：表格“结论”列不得出现任何 eligible 值
    assert "| eligible_external_test |" not in report
    assert "| exploratory_only |" not in report
    assert "本报告不产出 `frozen_decision`" in report
