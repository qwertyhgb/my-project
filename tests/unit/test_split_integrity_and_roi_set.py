"""split 泄漏检查与 ROI 集合校验的纯合成测试（不读真实数据、不创建 outputs）。

覆盖 §33 的 **split leakage guard** 与 §24 的 ``check_split_integrity.py``：
- ROI 集合只能覆盖训练 split，且 ``prior_source`` 不得是 ``ORACLE_GT``；
- 困难负样本集合的越界病例必须被检查脚本判为 FAIL（非零退出）；
- 缺失产物记为 SKIP，``--require`` 可把它升级为 FAIL（阶段关卡语义）。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from zonal_reliability_fusion.nnunet.roi_sampling import (
    ROI_SET_SCHEMA_VERSION,
    ROISetError,
    load_prostate_roi_set,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKER = PROJECT_ROOT / "scripts" / "data" / "check_split_integrity.py"


def _load_checker():
    spec = importlib.util.spec_from_file_location("check_split_integrity", CHECKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _roi_payload(train_cases, val_cases, cases, prior_source="predicted_prior", fold=0):
    return {
        "schema_version": ROI_SET_SCHEMA_VERSION,
        "dataset_name": "Dataset605_PICAI",
        "fold": fold,
        "definition": {"roi": "predicted WG bbox + physical margin"},
        "thresholds": {"margin_mm": 15.0, "wg_threshold": 0.5},
        "split": {"train_cases": train_cases, "val_cases": val_cases},
        "provenance": {"prior_source": prior_source},
        "cases": cases,
    }


# --------------------------------------------------------------------------- ROI set
def test_roi_set_accepts_train_only_payload(tmp_path: Path):
    path = tmp_path / "roi.json"
    path.write_text(
        json.dumps(
            _roi_payload(
                ["train_a", "train_b"],
                ["val_a"],
                {"train_a": {"lower_zyx": [0, 0, 0], "upper_zyx": [8, 8, 8]}},
            )
        ),
        encoding="utf-8",
    )
    payload = load_prostate_roi_set(
        path, expected_dataset_name="Dataset605_PICAI", expected_fold=0
    )
    assert "train_a" in payload["cases"]


def test_roi_set_rejects_validation_cases_and_illegal_boxes(tmp_path: Path):
    path = tmp_path / "roi.json"
    path.write_text(
        json.dumps(
            _roi_payload(
                ["train_a"],
                ["val_a"],
                {"val_a": {"lower_zyx": [0, 0, 0], "upper_zyx": [8, 8, 8]}},
            )
        ),
        encoding="utf-8",
    )
    with pytest.raises(ROISetError, match="非训练 split"):
        load_prostate_roi_set(
            path, expected_dataset_name="Dataset605_PICAI", expected_fold=0
        )

    path.write_text(
        json.dumps(
            _roi_payload(
                ["train_a"],
                ["val_a"],
                {"train_a": {"lower_zyx": [0, 0, 0], "upper_zyx": [0, 8, 8]}},
            )
        ),
        encoding="utf-8",
    )
    with pytest.raises(ROISetError, match="区间非法"):
        load_prostate_roi_set(
            path, expected_dataset_name="Dataset605_PICAI", expected_fold=0
        )


def test_roi_set_rejects_oracle_gt_prior_source(tmp_path: Path):
    """GT 解剖标签只允许用于上界分析；作为正式 ROI 来源必须被拒绝。"""
    path = tmp_path / "roi.json"
    path.write_text(
        json.dumps(
            _roi_payload(
                ["train_a"],
                ["val_a"],
                {"train_a": {"lower_zyx": [0, 0, 0], "upper_zyx": [8, 8, 8]}},
                prior_source="ORACLE_GT",
            )
        ),
        encoding="utf-8",
    )
    with pytest.raises(ROISetError, match="ORACLE_GT"):
        load_prostate_roi_set(
            path, expected_dataset_name="Dataset605_PICAI", expected_fold=0
        )


def test_roi_set_rejects_wrong_dataset_fold_and_schema(tmp_path: Path):
    path = tmp_path / "roi.json"
    base = _roi_payload(
        ["train_a"],
        ["val_a"],
        {"train_a": {"lower_zyx": [0, 0, 0], "upper_zyx": [8, 8, 8]}},
    )
    path.write_text(json.dumps(base), encoding="utf-8")
    with pytest.raises(ROISetError, match="数据集归属"):
        load_prostate_roi_set(
            path, expected_dataset_name="Dataset606_PICAI_Zonal", expected_fold=0
        )
    with pytest.raises(ROISetError, match="fold 归属"):
        load_prostate_roi_set(
            path, expected_dataset_name="Dataset605_PICAI", expected_fold=1
        )
    bad = dict(base, schema_version="9.9")
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ROISetError, match="schema_version"):
        load_prostate_roi_set(
            path, expected_dataset_name="Dataset605_PICAI", expected_fold=0
        )


# --------------------------------------------------------------------------- checker
def test_split_pair_check_flags_case_and_patient_overlap():
    module = _load_checker()
    overlap = module.check_split_pair("x", {"1_10", "2_20"}, {"1_10", "3_30"})
    assert overlap.status == module.FAIL
    patient_overlap = module.check_split_pair("x", {"1_10"}, {"1_11"})
    assert patient_overlap.status == module.FAIL
    ok = module.check_split_pair("x", {"1_10"}, {"2_20"}, expected_all={"1_10", "2_20"})
    assert ok.status == module.PASS


def test_checker_marks_missing_artifacts_as_skip_then_fail_with_require(tmp_path, monkeypatch, capsys):
    """缺产物 = SKIP（尚未生成）；--require 把它升级为 FAIL（阶段关卡）。"""
    import nnunetv2.paths

    monkeypatch.setattr(nnunetv2.paths, "nnUNet_preprocessed", str(tmp_path))
    monkeypatch.setattr(nnunetv2.paths, "nnUNet_results", str(tmp_path))
    code = _load_checker().main(["--require", "lesion_split"])
    output = capsys.readouterr().out
    assert code == 1
    assert "FAIL" in output
    assert "被 --require 升级为必须通过" in output


def _write_synthetic_lesion_split(tmp_path: Path) -> None:
    """写出一个最小可用的 lesion split，使检查脚本能走到「范围越界」这一分支。"""
    dataset_dir = tmp_path / "Dataset605_PICAI"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "dataset.json").write_text(
        json.dumps({"numTraining": 2}), encoding="utf-8"
    )
    (dataset_dir / "splits_final.json").write_text(
        json.dumps([{"train": ["train_a"], "val": ["val_a"]}]), encoding="utf-8"
    )


def test_checker_fails_on_hard_negative_scope_leak(tmp_path, monkeypatch, capsys):
    """困难负样本集合包含 validation 病例时，检查脚本必须 FAIL 并非零退出。"""
    import nnunetv2.paths

    from zonal_reliability_fusion.sampling.hard_negative import (
        HARD_NEGATIVE_SCHEMA_VERSION,
    )

    monkeypatch.setattr(nnunetv2.paths, "nnUNet_preprocessed", str(tmp_path))
    monkeypatch.setattr(nnunetv2.paths, "nnUNet_results", str(tmp_path))
    _write_synthetic_lesion_split(tmp_path)
    leak = tmp_path / "hn.json"
    leak.write_text(
        json.dumps(
            {
                "schema_version": HARD_NEGATIVE_SCHEMA_VERSION,
                "dataset_name": "Dataset605_PICAI",
                "fold": 0,
                "split": {"train_cases": ["train_a"], "val_cases": ["val_a"]},
                "cases": {"val_a": {"locations": [[1, 1, 1]]}},
            }
        ),
        encoding="utf-8",
    )
    code = _load_checker().main(["--hard-negative-set", str(leak)])
    output = capsys.readouterr().out
    assert code == 1
    assert "hard_negative_scope" in output
    assert "验证病例" in output
    assert "verdict=FAIL" in output


def test_checker_fails_on_oracle_roi_set(tmp_path, monkeypatch, capsys):
    import nnunetv2.paths

    monkeypatch.setattr(nnunetv2.paths, "nnUNet_preprocessed", str(tmp_path))
    monkeypatch.setattr(nnunetv2.paths, "nnUNet_results", str(tmp_path))
    _write_synthetic_lesion_split(tmp_path)
    roi = tmp_path / "roi.json"
    roi.write_text(
        json.dumps(
            _roi_payload(
                ["train_a"],
                ["val_a"],
                {"train_a": {"lower_zyx": [0, 0, 0], "upper_zyx": [4, 4, 4]}},
                prior_source="ORACLE_GT",
            )
        ),
        encoding="utf-8",
    )
    code = _load_checker().main(["--roi-set", str(roi)])
    output = capsys.readouterr().out
    assert code == 1
    assert "roi_set_scope" in output
    assert "ORACLE_GT" in output


def test_checker_passes_when_scope_is_clean_and_requires_satisfied(
    tmp_path, monkeypatch, capsys
):
    """干净的范围（ROI 只在训练 split）必须 PASS，且 --require 不再升级为 FAIL。"""
    import nnunetv2.paths

    monkeypatch.setattr(nnunetv2.paths, "nnUNet_preprocessed", str(tmp_path))
    monkeypatch.setattr(nnunetv2.paths, "nnUNet_results", str(tmp_path))
    _write_synthetic_lesion_split(tmp_path)
    roi = tmp_path / "roi.json"
    roi.write_text(
        json.dumps(
            _roi_payload(
                ["train_a"],
                ["val_a"],
                {"train_a": {"lower_zyx": [0, 0, 0], "upper_zyx": [4, 4, 4]}},
            )
        ),
        encoding="utf-8",
    )
    code = _load_checker().main(["--roi-set", str(roi), "--require", "lesion_split"])
    output = capsys.readouterr().out
    assert code == 0
    assert "verdict=PASS" in output
    assert "[PASS] lesion_split" in output


def test_anatomy_prior_directory_must_match_val_split_exactly(tmp_path: Path):
    """先验病例集合必须与 val split 精确对应：缺一例或多一例都是 FAIL。"""
    module = _load_checker()
    prior_dir = tmp_path / "priors"
    prior_dir.mkdir()
    for case_id in ("val_a", "val_b"):
        (prior_dir / f"{case_id}.npz").write_bytes(b"placeholder")

    exact = module.check_anatomy_prior_directory("x", prior_dir, {"val_a", "val_b"}, "D")
    assert exact.status == module.PASS

    missing = module.check_anatomy_prior_directory(
        "x", prior_dir, {"val_a", "val_b", "val_c"}, "D"
    )
    assert missing.status == module.FAIL
    assert "缺 1 例" in missing.detail

    extra = module.check_anatomy_prior_directory("x", prior_dir, {"val_a"}, "D")
    assert extra.status == module.FAIL
    assert "多 1 例" in extra.detail

    absent = module.check_anatomy_prior_directory("x", tmp_path / "nope", {"val_a"}, "D")
    assert absent.status == module.SKIP
