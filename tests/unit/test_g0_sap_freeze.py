"""G0-SAP 配置校验与冻结载体测试（合成配置；不运行评测、不读真实数据）。

覆盖：
- 硬冻结约束（主终点 / 层级比较 / bootstrap 定位 / seed-level 限制 / picai_eval 口径 / FROC 操作点）；
- 必需块与功效备忘录字段完整性；
- 冻结就绪检查（未填 → 阻塞；补齐 + sufficient → READY；FROZEN 但未完成 → 明确阻止）；
- 模板渲染（TODO、无编造数值）。
"""
from __future__ import annotations

import copy

import pytest

from zonal_reliability_fusion.protocols import g0_sap
from zonal_reliability_fusion.protocols.common import ProtocolConfigError


def base_config() -> dict:
    return {
        "protocol": {
            "id": "G0-SAP",
            "version": "draft-0.3",
            "status": "DRAFT",
            "sees_validation_predictions": False,
            "sees_final_test_predictions": False,
            "sees_model_predictions": False,
            "reviewer": None,
            "frozen_at": None,
        },
        "endpoints": {
            "primary": "val_positive_casewise_dice_mean",
            "key_secondary": ["lesion_average_precision"],
            "negative_exam": ["false_positive_case_rate"],
        },
        "checkpoint_selection": {"rule": "val_positive_casewise_dice_mean_max"},
        "case_level_metrics": {"positive_casewise_dice": {"definition": "..."}},
        "comparison_hierarchy": [
            {"id": "M4_vs_M2", "role": "confirmatory_primary"},
            {"id": "M4_vs_M3", "role": "confirmatory_secondary", "precondition": "M4_vs_M2_positive"},
        ],
        "strata": {"zonal": ["PZ", "TZ", "mixed", "outside"], "rule_ref": "research_plan §12.1"},
        "threshold_and_postprocessing": {
            "foreground_threshold": {"candidate_grid": None, "selection_metric": None, "tie_break_rule": None},
            "connectivity_3d": None,
            "min_lesion_volume": {"candidate_grid": None},
            "candidate_confidence": {"aggregation": None, "segmentation_to_detection_map": None},
        },
        "detection_evaluation": {
            "library": "picai_eval",
            "version": None,
            "min_overlap": 0.1,
            "overlap_func": None,
            "split_merge_rule": None,
            "case_confidence_function": None,
            "froc_operating_points": [0.5, 1.0, 2.0],
        },
        "statistics": {
            "analysis_unit": "patient_cluster",
            "ci_interpretation": "conditional_on_the_frozen_trained_seeds",
            "bootstrap": {"type": "paired_patient_cluster", "n_resamples": None, "seed": None, "ci": 0.95},
            "seed_level_bootstrap": {"allowed_as": "sensitivity_only"},
            "seed_reporting_required": ["all_seed_specific_effects_with_patient_ci"],
        },
        "seed_aggregation": {"seeds": None, "primary": "mean_paired_diff"},
        "power_memo": {
            "status": "INSUFFICIENT_DATA",
            "primary_endpoint": "val_positive_casewise_dice_mean",
            "n_positive_patients": None,
            "assumed_paired_effect": None,
            "assumed_paired_sd": None,
            "sd_source": None,
            "sd_sensitivity_range": None,
            "mde_95ci_width": None,
            "mcid": None,
            "mcid_source": None,
            "success_threshold": None,
        },
        "one_shot_evaluation": {"frozen_models": ["M2", "M3", "M4"]},
        "hash_binding": {"record_per_formal_run": ["splits_sha256"]},
        "outputs": {"dir": "outputs/metrics/evaluation", "files": []},
        "hashes": {},
        "sap_phases": {
            "sap_a": {"status": "NOT_STARTED", "method_sha256": None},
            "sap_b": {"status": "NOT_STARTED"},
            "sap_c": {"status": "NOT_STARTED"},
        },
        "sap_b_results": {
            "status": "NOT_STARTED",
            "threshold_policy": None,
            "selected_foreground_threshold": None,
            "selected_foreground_threshold_per_model": None,
            "selected_connectivity_3d": None,
            "selected_min_lesion_volume_mm3": None,
            "source_sap_a_config_sha256": None,
            "frozen_checkpoint_manifest_sha256": None,
            "validation_prediction_manifest_sha256": None,
            "selection_report_sha256": None,
            "sap_b_diff_audit_sha256": None,
            "completed_at": None,
            "reviewer": None,
        },
    }


def fully_filled_sap_a(doc: dict) -> dict:
    """把 SAP-A 方法字段全部填满并写入方法块哈希（仍保持 DRAFT / NOT_STARTED）。

    注意：方法块哈希必须在所有 SAP-A 方法字段填写**之后**计算。
    """
    doc["protocol"].update({"reviewer": "reviewer-A", "frozen_at": "2026-09-20"})
    doc["threshold_and_postprocessing"] = {
        "foreground_threshold": {
            "candidate_grid": [0.1, 0.5],
            "selection_metric": "val_dice",
            "tie_break_rule": "closest_0.5",
        },
        "connectivity_3d": 26,
        "min_lesion_volume": {"candidate_grid": [30.0]},
        "candidate_confidence": {"aggregation": "mean", "segmentation_to_detection_map": "cc"},
    }
    doc["detection_evaluation"].update(
        {"version": "1.2.2", "overlap_func": "iou", "split_merge_rule": "default", "case_confidence_function": "max"}
    )
    doc["statistics"]["bootstrap"].update({"n_resamples": 1000, "seed": 0})
    doc["power_memo"].update(
        {
            "status": "sufficient",
            "n_positive_patients": 63,
            "assumed_paired_effect": 0.05,
            "assumed_paired_sd": 0.12,
            "sd_source": "literature_range",
            "sd_sensitivity_range": [0.08, 0.18],
            "mde_95ci_width": 0.04,
            "mcid": 0.02,
            "mcid_source": "literature_range",
            "success_threshold": "CI 不跨 0 且 CI 下限 > -mcid",
        }
    )
    doc["hashes"] = {"configs/protocols/g0_sap.yaml": "0" * 64}
    doc["sap_phases"]["sap_a"]["method_sha256"] = g0_sap.sap_a_method_hash(doc)
    return doc


def completed_sap_b(doc: dict, *, policy: str = "shared") -> dict:
    """把 SAP-A 冻结 + SAP-B COMPLETED 的合法组合写入文档（用于正向/负向测试）。"""
    fully_filled_sap_a(doc)
    doc["protocol"]["status"] = "FROZEN"
    doc["protocol"]["sees_validation_predictions"] = True
    doc["sap_phases"]["sap_a"]["status"] = "FROZEN"
    doc["sap_phases"]["sap_b"]["status"] = "COMPLETED"
    doc["sap_b_results"].update(
        {
            "status": "COMPLETED",
            "threshold_policy": policy,
            "selected_foreground_threshold": 0.45 if policy == "shared" else None,
            "selected_foreground_threshold_per_model": {"M2": 0.45, "M3": 0.5, "M4": 0.4}
            if policy == "per_model"
            else None,
            "selected_connectivity_3d": 26,
            "selected_min_lesion_volume_mm3": 30.0,
            "source_sap_a_config_sha256": doc["sap_phases"]["sap_a"]["method_sha256"],
            "frozen_checkpoint_manifest_sha256": "a" * 64,
            "validation_prediction_manifest_sha256": "b" * 64,
            "selection_report_sha256": "c" * 64,
            "sap_b_diff_audit_sha256": "d" * 64,
            "completed_at": "2026-10-01T12:00:00",
            "reviewer": "reviewer-B",
        }
    )
    return doc


def test_valid_config_passes():
    assert g0_sap.validate_g0_sap_config(base_config()) == []
    g0_sap.assert_g0_sap_config(base_config())


@pytest.mark.parametrize(
    "mutate, needle",
    [
        (lambda d: d["endpoints"].update({"primary": "val_micro_dice"}), "endpoints.primary"),
        (lambda d: d.update({"statistics": {**d["statistics"], "ci_interpretation": "seed_level_ci"}}), "ci_interpretation"),
        (lambda d: d["statistics"].update({"bootstrap": {**d["statistics"]["bootstrap"], "type": "per_case"}}), "bootstrap.type"),
        (lambda d: d["statistics"].update({"analysis_unit": "study"}), "analysis_unit"),
        (lambda d: d["statistics"].update({"bootstrap": {**d["statistics"]["bootstrap"], "ci": 0.9}}), "bootstrap.ci"),
        (lambda d: d["statistics"].update({"seed_level_bootstrap": {"allowed_as": "confirmatory"}}), "sensitivity_only"),
        (lambda d: d["detection_evaluation"].update({"library": "custom"}), "detection_evaluation.library"),
        (lambda d: d["detection_evaluation"].update({"min_overlap": 0.2}), "min_overlap"),
        (lambda d: d["detection_evaluation"].update({"froc_operating_points": [1.0]}), "froc_operating_points"),
        (
            lambda d: d.update(
                {"comparison_hierarchy": [c for c in reversed(d["comparison_hierarchy"])]}
            ),
            "comparison_hierarchy",
        ),
        (lambda d: d.pop("strata"), "缺少必需配置块: strata"),
        (lambda d: d.update({"strata": {"zonal": []}}), "strata.zonal"),
    ],
)
def test_hard_constraints_rejected(mutate, needle):
    doc = base_config()
    mutate(doc)
    errors = g0_sap.validate_g0_sap_config(doc)
    assert any(needle in err for err in errors), (needle, errors)
    with pytest.raises(ProtocolConfigError):
        g0_sap.assert_g0_sap_config(doc)


def test_missing_comparison_precondition_rejected():
    doc = base_config()
    doc["comparison_hierarchy"][1].pop("precondition")
    errors = g0_sap.validate_g0_sap_config(doc)
    assert any("precondition" in err for err in errors)


def test_power_memo_field_missing_rejected():
    doc = base_config()
    doc["power_memo"].pop("mde_95ci_width")
    errors = g0_sap.validate_g0_sap_config(doc)
    assert any("mde_95ci_width" in err for err in errors)


def test_missing_sections_reported():
    doc = base_config()
    doc.pop("checkpoint_selection")
    doc.pop("case_level_metrics")
    errors = g0_sap.validate_g0_sap_config(doc)
    assert any("checkpoint_selection" in err for err in errors)
    assert any("case_level_metrics" in err for err in errors)


def test_freeze_readiness_not_ready_lists_fields():
    blockers = g0_sap.validate_freeze_readiness(base_config())
    assert blockers
    joined = " ".join(blockers)
    assert "threshold_and_postprocessing.foreground_threshold.candidate_grid" in joined
    assert "power_memo.status" in joined  # INSUFFICIENT_DATA 不算 sufficient


def test_freeze_readiness_ready_when_complete():
    doc = fully_filled_sap_a(base_config())
    assert g0_sap.validate_freeze_readiness(doc) == []
    # 即便就绪，也不得由工具自动改状态（仍为 DRAFT）
    assert doc["protocol"]["status"] == "DRAFT"


def test_mcid_missing_blocks_sap_a_freeze():
    """MCID 与来源缺失时不得 SAP-A FROZEN。"""
    doc = fully_filled_sap_a(base_config())
    doc["power_memo"]["mcid"] = None
    doc["power_memo"]["mcid_source"] = None
    blockers = g0_sap.validate_freeze_readiness(doc)
    joined = " ".join(blockers)
    assert "power_memo.mcid" in joined and "power_memo.mcid_source" in joined


def test_mcid_field_must_exist():
    doc = base_config()
    doc["power_memo"].pop("mcid")
    errors = g0_sap.validate_g0_sap_config(doc)
    assert any("mcid" in err for err in errors)


def test_sap_a_frozen_requires_protocol_frozen_and_hash():
    doc = fully_filled_sap_a(base_config())
    doc["sap_phases"]["sap_a"]["status"] = "FROZEN"
    errors = g0_sap.validate_g0_sap_config(doc)
    assert any("protocol.status 必须为 FROZEN" in err for err in errors)
    doc2 = fully_filled_sap_a(base_config())
    doc2["sap_phases"]["sap_a"]["method_sha256"] = None
    doc2["sap_phases"]["sap_a"]["status"] = "FROZEN"
    doc2["protocol"]["status"] = "FROZEN"
    errors2 = g0_sap.validate_g0_sap_config(doc2)
    assert any("method_sha256" in err for err in errors2)


def test_sap_a_not_started_forbids_visibility():
    doc = base_config()
    doc["protocol"]["sees_validation_predictions"] = True
    doc["protocol"]["sees_model_predictions"] = False
    errors = g0_sap.validate_g0_sap_config(doc)
    assert any("SAP-A 未冻结" in err for err in errors)


def test_legacy_visibility_must_match_final():
    doc = base_config()
    doc["protocol"]["sees_model_predictions"] = True  # final 仍为 false
    errors = g0_sap.validate_g0_sap_config(doc)
    assert any("sees_model_predictions" in err and "sees_final_test_predictions" in err for err in errors)


def test_sap_a_method_change_after_freeze_rejected():
    doc = fully_filled_sap_a(base_config())
    doc["protocol"]["status"] = "FROZEN"
    doc["sap_phases"]["sap_a"]["status"] = "FROZEN"
    assert g0_sap.validate_g0_sap_config(doc) == []
    doc["threshold_and_postprocessing"]["connectivity_3d"] = 6  # SAP-A 方法字段被改
    errors = g0_sap.validate_g0_sap_config(doc)
    assert any("方法块不一致" in err for err in errors)


def test_sap_b_not_started_must_not_prefill_results():
    doc = fully_filled_sap_a(base_config())
    doc["protocol"]["status"] = "FROZEN"
    doc["sap_phases"]["sap_a"]["status"] = "FROZEN"
    doc["sap_b_results"]["selected_foreground_threshold"] = 0.4
    errors = g0_sap.validate_g0_sap_config(doc)
    assert any("不得预填结果字段" in err for err in errors)


def test_sap_b_completed_valid_shared_policy_passes():
    doc = completed_sap_b(base_config(), policy="shared")
    assert g0_sap.validate_g0_sap_config(doc) == []


def test_sap_b_completed_valid_per_model_policy_passes():
    doc = completed_sap_b(base_config(), policy="per_model")
    assert g0_sap.validate_g0_sap_config(doc) == []


def test_sap_b_completed_requires_selected_values_and_hashes():
    doc = completed_sap_b(base_config(), policy="shared")
    doc["sap_b_results"]["selected_foreground_threshold"] = None
    doc["sap_b_results"]["selection_report_sha256"] = None
    errors = g0_sap.validate_g0_sap_config(doc)
    assert any("selected_foreground_threshold" in err for err in errors)
    assert any("selection_report_sha256" in err for err in errors)


def test_sap_b_completed_requires_validation_visibility_and_sap_a_frozen():
    doc = completed_sap_b(base_config())
    doc["protocol"]["sees_validation_predictions"] = False
    errors = g0_sap.validate_g0_sap_config(doc)
    assert any("sees_validation_predictions=true" in err for err in errors)
    doc2 = base_config()
    doc2["protocol"]["sees_validation_predictions"] = True
    doc2["sap_phases"]["sap_b"]["status"] = "COMPLETED"
    errors2 = g0_sap.validate_g0_sap_config(doc2)
    assert any("SAP-B COMPLETED 要求 SAP-A 已冻结" in err for err in errors2)


def test_sap_b_source_hash_must_match_sap_a_method_hash():
    doc = completed_sap_b(base_config())
    doc["sap_b_results"]["source_sap_a_config_sha256"] = "e" * 64
    errors = g0_sap.validate_g0_sap_config(doc)
    assert any("source_sap_a_config_sha256 必须等于" in err for err in errors)


def test_sap_b_requires_explicit_per_model_mapping():
    doc = completed_sap_b(base_config(), policy="per_model")
    doc["sap_b_results"]["selected_foreground_threshold_per_model"] = None
    errors = g0_sap.validate_g0_sap_config(doc)
    assert any("per_model" in err for err in errors)


def test_sap_c_before_completion_forbids_final_test_visibility():
    doc = completed_sap_b(base_config())
    doc["protocol"]["sees_final_test_predictions"] = True
    doc["protocol"]["sees_model_predictions"] = True
    errors = g0_sap.validate_g0_sap_config(doc)
    assert any("SAP-C 未完成" in err for err in errors)
    doc2 = completed_sap_b(base_config())
    doc2["sap_phases"]["sap_c"]["status"] = "COMPLETED"
    errors2 = g0_sap.validate_g0_sap_config(doc2)
    assert any("sees_final_test_predictions=true" in err for err in errors2)


def test_sap_c_completed_consistent_doc_passes():
    doc = completed_sap_b(base_config())
    doc["sap_phases"]["sap_c"]["status"] = "COMPLETED"
    doc["protocol"]["sees_final_test_predictions"] = True
    doc["protocol"]["sees_model_predictions"] = True
    assert g0_sap.validate_g0_sap_config(doc) == []


def test_sap_a_baseline_diff_audit_rejects_method_change():
    baseline = completed_sap_b(base_config())
    changed = copy.deepcopy(baseline)
    changed["threshold_and_postprocessing"]["foreground_threshold"]["candidate_grid"] = [0.2, 0.6]
    errors = g0_sap.validate_g0_sap_config(changed, sap_a_baseline=baseline)
    assert any("差异审计拒绝" in err for err in errors)
    assert g0_sap.validate_g0_sap_config(baseline, sap_a_baseline=baseline) == []


def test_diff_sap_a_methods_reports_paths():
    before = base_config()
    after = copy.deepcopy(before)
    after["threshold_and_postprocessing"]["connectivity_3d"] = 26
    diffs = g0_sap.diff_sap_a_methods(before, after)
    assert [item["path"] for item in diffs] == ["threshold_and_postprocessing.connectivity_3d"]
    assert diffs[0]["before"] is None and diffs[0]["after"] == 26


def test_sap_a_method_hash_is_stable_and_sensitive():
    doc = base_config()
    first = g0_sap.sap_a_method_hash(doc)
    assert first == g0_sap.sap_a_method_hash(copy.deepcopy(doc))
    doc["power_memo"]["mcid"] = 0.02
    assert g0_sap.sap_a_method_hash(doc) != first


def test_frozen_status_with_blockers_is_reported():
    doc = base_config()
    doc["protocol"]["status"] = "FROZEN"
    blockers = g0_sap.validate_freeze_readiness(doc)
    assert any("已写 FROZEN" in item for item in blockers)


def test_power_memo_template_has_todo_without_numbers():
    text = g0_sap.render_power_memo_template(base_config(), generated_at="2026-09-15T00:00:00")
    assert "TODO" in text
    assert "INSUFFICIENT DATA" in text
    assert "不得伪造 MDE" in text
    # 模板本身不得产生任何具体功效数值（表格里全部是 TODO）
    assert "0.05" not in text and "63" not in text


def test_freeze_checklist_lists_blockers():
    blockers = g0_sap.validate_freeze_readiness(base_config())
    text = g0_sap.render_freeze_checklist(base_config(), blockers, generated_at="2026-09-15T00:00:00")
    assert "冻结阻塞项" in text
    assert "power_memo.status" in text
    assert "status` 改为 `FROZEN`" in text


def test_stats_config_template_keeps_nulls():
    tmpl = g0_sap.stats_config_template(base_config())
    assert tmpl["bootstrap"]["n_resamples"] is None and tmpl["bootstrap"]["seed"] is None
    assert tmpl["ci_interpretation"] == "conditional_on_the_frozen_trained_seeds"
    assert "冻结" in tmpl["note"]


def test_repo_config_matches_validator():
    """仓库内 `configs/protocols/g0_sap.yaml` 必须通过同一套校验（防止 YAML/代码漂移）。"""
    from zonal_reliability_fusion.protocols import common as pc

    doc = pc.load_yaml_config(pc.PROJECT_ROOT / "configs/protocols/g0_sap.yaml")
    assert g0_sap.validate_g0_sap_config(doc) == []
    blockers = g0_sap.validate_freeze_readiness(doc)
    assert blockers  # 当前仍为 DRAFT：必须报出未填字段（不得被误判为可冻结）
    assert doc["protocol"]["status"] == "DRAFT"
    assert copy.deepcopy(doc)["protocol"]["status"] == "DRAFT"
