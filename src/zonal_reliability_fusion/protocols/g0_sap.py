"""G0-SAP：评测配置的静态校验、冻结就绪检查与冻结载体模板。

本模块**不运行评测**、不读取预测/影像数据；只回答以下问题：

1. 评测配置是否满足协议硬约束（主终点、层级比较、bootstrap 定位、picai_eval 口径）？
2. 距离“SAP-A 可冻结”还缺哪些字段（阈值/后处理候选、功效备忘录与 MCID、审阅/哈希）？
3. SAP-A / SAP-B / SAP-C 的阶段状态与预测可见性组合是否自洽？
4. SAP-B 是否只写入了独立的结果块（`sap_b_results`），而没有改动 SAP-A 的方法字段？

**阶段语义（与 docs/protocols/G0_SAP.md §0.1 一致）**：

- `protocol.status = FROZEN` **仅表示 SAP-A（方法与候选集）已冻结**，不代表 SAP-B/SAP-C；
- SAP-B 只能填写 `sap_b_results` 及审计字段；SAP-A 方法字段的任何改动都会使
  `sap_phases.sap_a.method_sha256` 与实际方法块不一致 → 校验失败（或由差异审计拒绝）；
- `power_memo`（含 MCID）属于 **SAP-A**，SAP-B 不得修改。

不得伪造任何数值：未填写字段以 `None` / `INSUFFICIENT_DATA` 表示，模板中写 `TODO`。
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

from .common import ProtocolConfigError, find_unfilled_fields, get_path

#: 主终点（硬冻结：与 research_plan §11.1 / §9.4 一致）
PRIMARY_ENDPOINT = "val_positive_casewise_dice_mean"

#: 主比较层级（顺序与角色不可交换）
REQUIRED_COMPARISONS = (
    {"id": "M4_vs_M2", "role": "confirmatory_primary"},
    {"id": "M4_vs_M3", "role": "confirmatory_secondary"},
)

#: 必须存在的顶层块（含 v2.2 新增：checkpoint 选择、病例级指标口径、分层、seed 汇总）
REQUIRED_SECTIONS: tuple[str, ...] = (
    "endpoints",
    "comparison_hierarchy",
    "checkpoint_selection",
    "case_level_metrics",
    "strata",
    "threshold_and_postprocessing",
    "detection_evaluation",
    "statistics",
    "seed_aggregation",
    "power_memo",
    "one_shot_evaluation",
    "hash_binding",
    "outputs",
    "sap_phases",       # v0.3：SAP-A/B/C 阶段状态
    "sap_b_results",    # v0.3：SAP-B 唯一允许写入的结果/审计块
)

#: 功效/精度备忘录字段（docs/protocols/G0_SAP.md §5）；全部属于 **SAP-A**，SAP-B 不得修改
POWER_MEMO_FIELDS: tuple[str, ...] = (
    "status",
    "primary_endpoint",
    "n_positive_patients",
    "assumed_paired_effect",
    "assumed_paired_sd",
    "sd_source",
    "sd_sensitivity_range",
    "mde_95ci_width",
    "mcid",
    "mcid_source",
    "success_threshold",
)

#: SAP-A 必须冻结的**方法字段**（SAP-B 只允许填写 `sap_b_results`，不得改动这些路径）
SAP_A_METHOD_PATHS: tuple[str, ...] = (
    "endpoints.primary",
    "endpoints.key_secondary",
    "endpoints.negative_exam",
    "comparison_hierarchy",
    "case_level_metrics",
    "strata",
    "checkpoint_selection",
    "threshold_and_postprocessing.foreground_threshold.candidate_grid",
    "threshold_and_postprocessing.foreground_threshold.selection_metric",
    "threshold_and_postprocessing.foreground_threshold.tie_break_rule",
    "threshold_and_postprocessing.connectivity_3d",
    "threshold_and_postprocessing.min_lesion_volume.candidate_grid",
    "threshold_and_postprocessing.candidate_confidence.aggregation",
    "threshold_and_postprocessing.candidate_confidence.segmentation_to_detection_map",
    "threshold_and_postprocessing.selection_scope",
    "threshold_and_postprocessing.independent_threshold_per_model_allowed_only_if",
    "detection_evaluation.library",
    "detection_evaluation.version",
    "detection_evaluation.min_overlap",
    "detection_evaluation.overlap_func",
    "detection_evaluation.split_merge_rule",
    "detection_evaluation.case_confidence_function",
    "detection_evaluation.ap_input",
    "detection_evaluation.froc_operating_points",
    "detection_evaluation.froc_other_curves",
    "statistics.analysis_unit",
    "statistics.multi_study_patient",
    "statistics.paired_diff_per_seed",
    "statistics.main_effect",
    "statistics.bootstrap.type",
    "statistics.bootstrap.n_resamples",
    "statistics.bootstrap.seed",
    "statistics.bootstrap.ci",
    "statistics.ci_interpretation",
    "statistics.seed_reporting_required",
    "statistics.seed_level_bootstrap",
    "statistics.ap_rule",
    "statistics.lesion_froc_rule",
    "statistics.multiplicity_discipline",
    "statistics.reproducibility_artifacts",
    "seed_aggregation.seeds",
    "seed_aggregation.primary",
    "seed_aggregation.report",
    "seed_aggregation.seed_level_ci",
    "seed_aggregation.disclaimer",
    "one_shot_evaluation.frozen_models",
    "one_shot_evaluation.rules",
    "hash_binding",
    "power_memo.status",
    "power_memo.primary_endpoint",
    "power_memo.n_positive_patients",
    "power_memo.assumed_paired_effect",
    "power_memo.assumed_paired_sd",
    "power_memo.sd_source",
    "power_memo.sd_sensitivity_range",
    "power_memo.mde_95ci_width",
    "power_memo.mcid",
    "power_memo.mcid_source",
    "power_memo.success_threshold",
    "power_memo.rules",
)

#: SAP-A 方法块哈希算法标识
SAP_A_METHOD_HASH_ALGORITHM = "sha256(canonical_json(SAP_A_METHOD_PATHS))"

#: 阶段允许的状态取值
SAP_A_STATUSES: tuple[str, ...] = ("NOT_STARTED", "FROZEN")
SAP_B_STATUSES: tuple[str, ...] = ("NOT_STARTED", "COMPLETED")
SAP_C_STATUSES: tuple[str, ...] = ("NOT_STARTED", "COMPLETED")

#: SAP-B 阈值口径
THRESHOLD_POLICIES: tuple[str, ...] = ("shared", "per_model")

#: SAP-B COMPLETED 时必须完整填写的结果字段（不含 `sap_b_results.status` 自身与策略相关字段）
SAP_B_REQUIRED_RESULT_FIELDS: tuple[str, ...] = (
    "threshold_policy",
    "selected_connectivity_3d",
    "selected_min_lesion_volume_mm3",
    "source_sap_a_config_sha256",
    "frozen_checkpoint_manifest_sha256",
    "validation_prediction_manifest_sha256",
    "selection_report_sha256",
    "sap_b_diff_audit_sha256",
    "completed_at",
    "reviewer",
)

#: 阈值策略相关字段（按 `threshold_policy` 二选一必填）
SAP_B_POLICY_FIELDS: tuple[str, ...] = (
    "selected_foreground_threshold",
    "selected_foreground_threshold_per_model",
)

#: SAP-B 结果块的**全部**结果字段（用于"SAP-B 未开始时不得预填"检查）
SAP_B_ALL_FIELDS: tuple[str, ...] = SAP_B_REQUIRED_RESULT_FIELDS + SAP_B_POLICY_FIELDS

#: SAP-B COMPLETED 时必须为合法 SHA256 的字段
SAP_B_HASH_FIELDS: tuple[str, ...] = (
    "source_sap_a_config_sha256",
    "frozen_checkpoint_manifest_sha256",
    "validation_prediction_manifest_sha256",
    "selection_report_sha256",
    "sap_b_diff_audit_sha256",
)

#: 冻结就绪所必需的字段（任一未填 → 不得置 FROZEN；由 `validate_freeze_readiness` 报告）
FREEZE_REQUIRED_FIELDS: tuple[str, ...] = (
    "protocol.reviewer",
    "protocol.frozen_at",
    "threshold_and_postprocessing.foreground_threshold.candidate_grid",
    "threshold_and_postprocessing.foreground_threshold.selection_metric",
    "threshold_and_postprocessing.foreground_threshold.tie_break_rule",
    "threshold_and_postprocessing.connectivity_3d",
    "threshold_and_postprocessing.min_lesion_volume.candidate_grid",
    "threshold_and_postprocessing.candidate_confidence.aggregation",
    "threshold_and_postprocessing.candidate_confidence.segmentation_to_detection_map",
    "detection_evaluation.version",
    "detection_evaluation.overlap_func",
    "detection_evaluation.split_merge_rule",
    "detection_evaluation.case_confidence_function",
    "statistics.bootstrap.n_resamples",
    "statistics.bootstrap.seed",
    "power_memo.status",
    "power_memo.n_positive_patients",
    "power_memo.assumed_paired_effect",
    "power_memo.assumed_paired_sd",
    "power_memo.sd_source",
    "power_memo.sd_sensitivity_range",
    "power_memo.mde_95ci_width",
    "power_memo.mcid",
    "power_memo.mcid_source",
    "power_memo.success_threshold",
    "sap_phases.sap_a.method_sha256",
    "hashes",
)


def _require_equal(actual: Any, expected: Any, name: str, errors: list[str]) -> None:
    if actual != expected:
        errors.append(f"{name} 必须为 {expected!r}（硬冻结），收到 {actual!r}")


# --------------------------------------------------------------------- SAP 阶段：方法块哈希与差异审计
def canonical_json(payload: Any) -> str:
    """稳定 JSON 序列化（sort_keys + 紧凑分隔符），用于方法块哈希与差异比对。"""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def sap_a_method_snapshot(doc: Mapping[str, Any]) -> dict[str, Any]:
    """抽取 SAP-A 方法块（缺失路径记为 `None`，不推断默认值）。"""
    return {path: get_path(doc, path, default=None) for path in SAP_A_METHOD_PATHS}


def sap_a_method_hash(doc: Mapping[str, Any]) -> str:
    """SAP-A 方法块哈希（SAP-A 冻结时写入 `sap_phases.sap_a.method_sha256`）。"""
    return hashlib.sha256(canonical_json(sap_a_method_snapshot(doc)).encode("utf-8")).hexdigest()


def diff_sap_a_methods(before: Mapping[str, Any], after: Mapping[str, Any]) -> list[dict[str, Any]]:
    """结构化差异审计：返回被改动的 SAP-A 方法字段（空列表 = 方法块未改动）。"""
    before_snap = sap_a_method_snapshot(before)
    after_snap = sap_a_method_snapshot(after)
    return [
        {"path": path, "before": before_snap[path], "after": after_snap[path]}
        for path in SAP_A_METHOD_PATHS
        if before_snap[path] != after_snap[path]
    ]


def _is_sha256_hex(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(char in "0123456789abcdef" for char in value.lower())


def validate_sap_phases(
    doc: Mapping[str, Any], *, sap_a_baseline: Mapping[str, Any] | None = None
) -> list[str]:
    """校验 SAP-A/B/C 阶段状态、预测可见性与 SAP-B 结果块的组合自洽性。

    `sap_a_baseline` 可选：传入 SAP-A 冻结时的配置快照（解析后的映射）时，额外执行
    **结构化差异审计**——SAP-A 方法字段的任何改动都会被拒绝。
    """
    errors: list[str] = []
    phases = doc.get("sap_phases")
    results = doc.get("sap_b_results")
    proto = doc.get("protocol")
    if not isinstance(phases, Mapping):
        return ["sap_phases 必须是映射（sap_a / sap_b / sap_c）"]
    if not isinstance(results, Mapping):
        return ["sap_b_results 必须是映射（SAP-B 的结果与审计字段）"]
    if not isinstance(proto, Mapping):
        return ["protocol 必须是映射"]

    for name, allowed in (("sap_a", SAP_A_STATUSES), ("sap_b", SAP_B_STATUSES), ("sap_c", SAP_C_STATUSES)):
        block = phases.get(name)
        if not isinstance(block, Mapping):
            errors.append(f"sap_phases.{name} 必须是映射")
            continue
        status = block.get("status")
        if status not in allowed:
            errors.append(f"sap_phases.{name}.status 必须是 {allowed} 之一，收到 {status!r}")
    a_block = phases.get("sap_a")
    b_block = phases.get("sap_b")
    c_block = phases.get("sap_c")
    a = a_block if isinstance(a_block, Mapping) else {}
    b = b_block if isinstance(b_block, Mapping) else {}
    c = c_block if isinstance(c_block, Mapping) else {}
    a_status = a.get("status")
    b_status = b.get("status")
    c_status = c.get("status")
    proto_status = proto.get("status")
    legacy_visible = bool(proto.get("sees_model_predictions", False))
    validation_visible = bool(proto.get("sees_validation_predictions", False))
    final_visible = bool(proto.get("sees_final_test_predictions", False))

    # 1) 兼容字段必须恒等于 final-test 可见性
    if legacy_visible != final_visible:
        errors.append(
            "protocol.sees_model_predictions（兼容字段）必须恒等于 "
            "protocol.sees_final_test_predictions（final-test 可见性）"
        )

    # 2) protocol.status 仅表示 SAP-A；不得用顶层 FROZEN 表示三个不同阶段
    if proto_status == "FROZEN" and a_status != "FROZEN":
        errors.append("protocol.status=FROZEN 仅表示 SAP-A 冻结，但 sap_phases.sap_a.status != FROZEN")
    if a_status == "FROZEN" and proto_status != "FROZEN":
        errors.append("sap_phases.sap_a.status=FROZEN 时 protocol.status 必须为 FROZEN（后续阶段用 sap_b/sap_c 表示）")

    # 3) SAP-A 未开始时不得看到任何预测
    if a_status == "NOT_STARTED" and (validation_visible or final_visible):
        errors.append("SAP-A 未冻结时 sees_validation_predictions / sees_final_test_predictions 必须为 false")

    # 4) SAP-A 方法块哈希：冻结时必须记录；记录后任何方法字段改动都使校验失败
    recorded_method_hash = a.get("method_sha256")
    computed_method_hash = sap_a_method_hash(doc)
    if a_status == "FROZEN" and not recorded_method_hash:
        errors.append("SAP-A 已冻结但缺少 sap_phases.sap_a.method_sha256（冻结时必须记录方法块哈希）")
    if recorded_method_hash and recorded_method_hash != computed_method_hash:
        errors.append(
            "sap_phases.sap_a.method_sha256 与当前 SAP-A 方法块不一致："
            "SAP-A 冻结后方法字段不得修改（差异审计拒绝）"
        )

    # 5) SAP-B：结果块只允许在 COMPLETED 时完整填写
    result_status = results.get("status")
    if result_status not in ("NOT_STARTED", "COMPLETED"):
        errors.append(f"sap_b_results.status 必须是 ('NOT_STARTED','COMPLETED') 之一，收到 {result_status!r}")
    if b_status == "NOT_STARTED":
        if validation_visible:
            errors.append("sap_phases.sap_b.status=NOT_STARTED 时 sees_validation_predictions 必须为 false")
        if result_status not in (None, "NOT_STARTED"):
            errors.append("SAP-B 未开始时 sap_b_results.status 必须为 NOT_STARTED")
        pre_filled = [
            path for path in SAP_B_ALL_FIELDS if get_path(results, path, default=None) not in (None, "", [], {})
        ]
        if pre_filled:
            errors.append("SAP-B 未开始时不得预填结果字段：" + ", ".join("sap_b_results." + p for p in pre_filled))
    if b_status == "COMPLETED":
        if a_status != "FROZEN":
            errors.append("SAP-B COMPLETED 要求 SAP-A 已冻结（方法未冻结时不得在 validation 上选阈值）")
        if not validation_visible:
            errors.append("SAP-B COMPLETED 要求 protocol.sees_validation_predictions=true")
        if result_status != "COMPLETED":
            errors.append("sap_phases.sap_b.status=COMPLETED 但 sap_b_results.status != COMPLETED")
        unfilled = find_unfilled_fields(results, SAP_B_REQUIRED_RESULT_FIELDS)
        if unfilled:
            errors.append("SAP-B COMPLETED 但结果字段缺失：" + ", ".join("sap_b_results." + p for p in unfilled))
        for field in SAP_B_HASH_FIELDS:
            value = get_path(results, field, default=None)
            if value not in (None, "") and not _is_sha256_hex(value):
                errors.append(f"sap_b_results.{field} 必须是 64 位十六进制 SHA256，收到 {value!r}")
        policy = get_path(results, "threshold_policy", default=None)
        if policy not in THRESHOLD_POLICIES and policy not in (None, ""):
            errors.append("sap_b_results.threshold_policy 必须是 'shared' 或 'per_model'")
        if policy == "shared" and get_path(results, "selected_foreground_threshold", default=None) in (None, ""):
            errors.append("threshold_policy=shared 时必须填写 sap_b_results.selected_foreground_threshold")
        if policy == "per_model":
            per_model = get_path(results, "selected_foreground_threshold_per_model", default=None)
            if not isinstance(per_model, Mapping) or not per_model:
                errors.append(
                    "threshold_policy=per_model 时必须填写显式 model_id → value 映射 "
                    "sap_b_results.selected_foreground_threshold_per_model"
                )
            if get_path(results, "selected_foreground_threshold", default=None) not in (None, ""):
                errors.append("threshold_policy=per_model 时不得同时填写共享阈值 selected_foreground_threshold")
        source_hash = get_path(results, "source_sap_a_config_sha256", default=None)
        if recorded_method_hash and source_hash and source_hash != recorded_method_hash:
            errors.append(
                "sap_b_results.source_sap_a_config_sha256 必须等于 sap_phases.sap_a.method_sha256"
                "（SAP-B 只能基于同一 SAP-A 方法块）"
            )
    if validation_visible and b_status != "COMPLETED":
        errors.append("sees_validation_predictions=true 时 sap_phases.sap_b.status 必须为 COMPLETED")

    # 6) SAP-C：最终 test 可见性只能在 SAP-C COMPLETED 之后
    if c_status == "NOT_STARTED" and final_visible:
        errors.append("SAP-C 未完成时 sees_final_test_predictions 必须为 false（不得查看最终 test 预测）")
    if c_status == "COMPLETED":
        if b_status != "COMPLETED":
            errors.append("SAP-C COMPLETED 要求 SAP-B 已 COMPLETED")
        if not final_visible:
            errors.append("SAP-C COMPLETED 要求 protocol.sees_final_test_predictions=true")

    # 7) 可选：与 SAP-A 基线快照做结构化差异审计
    if sap_a_baseline is not None:
        diffs = diff_sap_a_methods(sap_a_baseline, doc)
        if diffs:
            errors.append(
                "SAP-B 改动了 SAP-A 方法字段（差异审计拒绝）：" + ", ".join(item["path"] for item in diffs)
            )
        if sap_a_method_hash(sap_a_baseline) != computed_method_hash:
            errors.append("传入的 SAP-A 基线方法块与当前方法块不一致（差异审计拒绝）")
    return errors


def validate_g0_sap_config(
    doc: Mapping[str, Any], *, sap_a_baseline: Mapping[str, Any] | None = None
) -> list[str]:
    """校验 G0-SAP 配置；返回**错误列表**（空列表 = 通过）。不修改任何字段。"""
    errors: list[str] = []
    for section in REQUIRED_SECTIONS:
        value = doc.get(section)
        if value is None:
            errors.append(f"缺少必需配置块: {section}")

    _require_equal(get_path(doc, "endpoints.primary"), PRIMARY_ENDPOINT, "endpoints.primary", errors)

    hierarchy = get_path(doc, "comparison_hierarchy") or []
    if not isinstance(hierarchy, list):
        errors.append("comparison_hierarchy 必须是列表")
    else:
        ids = [item.get("id") for item in hierarchy if isinstance(item, Mapping)]
        roles = [item.get("role") for item in hierarchy if isinstance(item, Mapping)]
        _require_equal(ids, [c["id"] for c in REQUIRED_COMPARISONS], "comparison_hierarchy[].id", errors)
        _require_equal(roles, [c["role"] for c in REQUIRED_COMPARISONS], "comparison_hierarchy[].role", errors)
        if len(hierarchy) >= 2 and isinstance(hierarchy[1], Mapping):
            if hierarchy[1].get("precondition") != "M4_vs_M2_positive":
                errors.append("comparison_hierarchy[1].precondition 必须为 'M4_vs_M2_positive'")

    _require_equal(
        get_path(doc, "statistics.ci_interpretation"),
        "conditional_on_the_frozen_trained_seeds",
        "statistics.ci_interpretation",
        errors,
    )
    _require_equal(
        get_path(doc, "statistics.bootstrap.type"),
        "paired_patient_cluster",
        "statistics.bootstrap.type",
        errors,
    )
    _require_equal(get_path(doc, "statistics.analysis_unit"), "patient_cluster", "statistics.analysis_unit", errors)
    _require_equal(
        get_path(doc, "statistics.bootstrap.ci"), 0.95, "statistics.bootstrap.ci", errors
    )
    seed_level = get_path(doc, "statistics.seed_level_bootstrap.allowed_as")
    if seed_level not in ("sensitivity_only", None):
        errors.append("statistics.seed_level_bootstrap.allowed_as 只能是 'sensitivity_only'（不得作为确认性 seed-level CI）")

    _require_equal(
        get_path(doc, "detection_evaluation.library"), "picai_eval", "detection_evaluation.library", errors
    )
    _require_equal(
        get_path(doc, "detection_evaluation.min_overlap"), 0.1, "detection_evaluation.min_overlap", errors
    )
    froc = get_path(doc, "detection_evaluation.froc_operating_points")
    _require_equal(list(froc or []), [0.5, 1.0, 2.0], "detection_evaluation.froc_operating_points", errors)

    memo = doc.get("power_memo")
    if not isinstance(memo, Mapping):
        errors.append("power_memo 必须是映射")
    else:
        for field in POWER_MEMO_FIELDS:
            if field not in memo:
                errors.append(f"power_memo 缺少字段: {field}（值可为 null，但字段必须存在）")

    strata = doc.get("strata")
    if isinstance(strata, Mapping):
        zonal = strata.get("zonal")
        if not isinstance(zonal, (list, tuple)) or not zonal:
            errors.append("strata.zonal 必须为非空列表（PZ/TZ/mixed/outside）")

    errors.extend(validate_sap_phases(doc, sap_a_baseline=sap_a_baseline))
    return errors


def assert_g0_sap_config(doc: Mapping[str, Any], *, sap_a_baseline: Mapping[str, Any] | None = None) -> None:
    errors = validate_g0_sap_config(doc, sap_a_baseline=sap_a_baseline)
    if errors:
        raise ProtocolConfigError("G0-SAP 配置非法：\n- " + "\n- ".join(errors))


def validate_freeze_readiness(doc: Mapping[str, Any]) -> list[str]:
    """返回阻止“SAP-A 置为 FROZEN”的原因列表（空 = 可冻结）。

    规则（`protocol.status=FROZEN` **仅表示 SAP-A 方法冻结**）：

    - 所有 `FREEZE_REQUIRED_FIELDS` 必须已填写（非 null / 非 INSUFFICIENT_DATA），
      其中包含 **MCID 与 MCID 来源** 与 **SAP-A 方法块哈希**；
    - `power_memo.status` 必须为 `sufficient`；
    - SAP-C 状态与最终 test 可见性由 `validate_sap_phases` 独立校验（不参与 SAP-A 冻结就绪）。
    """
    blockers: list[str] = []
    unfilled = find_unfilled_fields(doc, FREEZE_REQUIRED_FIELDS)
    if unfilled:
        blockers.append("以下字段仍为空/未填（不得置 FROZEN）：" + ", ".join(unfilled))
    memo_status = str(get_path(doc, "power_memo.status", default="")).strip().lower()
    if memo_status != "sufficient":
        blockers.append(f"power_memo.status 必须是 'sufficient' 才能冻结，当前为 {memo_status or 'UNKNOWN'!r}")
    status = str(get_path(doc, "protocol.status", default="DRAFT"))
    if status == "FROZEN" and blockers:
        blockers.append("protocol.status 已写 FROZEN 但上述条件未满足：请回退为 DRAFT 或补齐证据")
    return blockers


def render_power_memo_template(doc: Mapping[str, Any], *, generated_at: str) -> str:
    """生成功效/精度备忘录模板（字段留 TODO；不编造任何数值）。"""
    memo = dict(doc.get("power_memo") or {})
    lines = [
        "# G0-SAP 功效/精度备忘录（模板）",
        "",
        f"- 生成时间：{generated_at}",
        f"- 当前 power_memo.status：`{memo.get('status', 'INSUFFICIENT_DATA')}`",
        "- **填写纪律**：只允许使用文献范围或**盲态** nuisance estimate；",
        "  不得用观察到的 M4 效果反向制定阈值；数据不足时保留 `INSUFFICIENT DATA`，不得伪造 MDE。",
        "",
        "| 字段 | 值 | 来源/说明 |",
        "|---|---|---|",
    ]
    for field_name in POWER_MEMO_FIELDS:
        value = memo.get(field_name)
        shown = "TODO" if value in (None, "", "INSUFFICIENT_DATA") else value
        if field_name == "assumed_paired_sd":
            source = memo.get("sd_source") or ""
        elif field_name == "mcid":
            source = memo.get("mcid_source") or ""
        else:
            source = ""
        lines.append(f"| `{field_name}` | {shown} | {source or ''} |")
    lines += [
        "",
        "## 规则（不可协商）",
        "",
        "- 主要终点固定为 `val_positive_casewise_dice_mean`（阳性患者 case-wise Dice）；",
        "- 阳性患者数必须来自最终 test（G0-E 路径 A/B/C）的实际计数；",
        "- **MCID、MCID 来源、成功阈值、效应/方差假设与敏感性范围全部属于 SAP-A**：必须在查看任何模型预测前写下；",
        "- **SAP-B 不得修改本备忘录**（SAP-B 只填写 `sap_b_results` 结果块；当前没有任何字段被列入 SAP-B 白名单）；",
        "- 若样本量/语义不足 → 看结果前降级为探索性验证（不得事后改判）。",
    ]
    return "\n".join(lines) + "\n"


def render_freeze_checklist(doc: Mapping[str, Any], blockers: Sequence[str], *, generated_at: str) -> str:
    """生成冻结清单（列出阻塞项与 SAP-A/B/C 步骤）。"""
    status = get_path(doc, "protocol.status", default="DRAFT")
    method_hash = sap_a_method_hash(doc)
    lines = [
        "# G0-SAP 冻结检查清单（自动生成）",
        "",
        f"- 生成时间：{generated_at}",
        f"- `protocol.status`：**{status}**（该状态**仅表示 SAP-A 方法冻结**；SAP-B/SAP-C 见 `sap_phases`）",
        f"- 当前 SAP-A 方法块哈希：`{method_hash}`",
        f"  （算法：`{SAP_A_METHOD_HASH_ALGORITHM}`；冻结时写入 `sap_phases.sap_a.method_sha256`）",
        "",
        "## 冻结阻塞项（SAP-A）",
        "",
    ]
    if blockers:
        lines += [f"- {item}" for item in blockers]
    else:
        lines.append("- （无：全部必填字段已就绪；仍须人工确认后手动把 status 改为 FROZEN 并重记哈希）")
    lines += [
        "",
        "## 冻结步骤（研究者执行）",
        "",
        "**SAP-A（看任何模型结果之前）**",
        "",
        "1. 确认 `evaluation_config.sha256` 与实测配置一致，并记录 `picai_eval` 版本与安装来源；",
        "2. 把本清单的阻塞项清零（阈值候选网格与选择指标、连通性、最小体积、候选置信度、",
        "   功效备忘录含 MCID 与来源、审阅与日期、`hashes`）；",
        "3. 用本清单打印的 SAP-A 方法块哈希填写 `sap_phases.sap_a.method_sha256`；",
        (
            "4. 手动将 `configs/protocols/g0_sap.yaml` 的 `protocol.status` 改为 `FROZEN`、"
            "`sap_phases.sap_a.status` 改为 `FROZEN`，并填写 `reviewer`/`frozen_at`；"
        ),
        "5. 重新运行 `scripts/evaluate/prepare_g0_sap_freeze.py` 生成冻结包（含更新后的哈希与 `sap_a_method_hash.txt`）。",
        "",
        "**SAP-B（冻结 checkpoint 之后、最终 test 之前）**",
        "",
        (
            "6. 在 PI-CAI validation 上按 SAP-A 的算法一次性选阈值与后处理，**只写入 `sap_b_results`**"
            "（含 `source_sap_a_config_sha256`、三个 manifest/report 哈希与差异审计哈希）；"
        ),
        (
            "7. 置 `sap_phases.sap_b.status=COMPLETED`、`protocol.sees_validation_predictions=true`；"
            "**不得改动 SAP-A 方法字段与 `power_memo`**（方法块哈希不一致会被校验器拒绝）。"
        ),
        "",
        "**SAP-C（封存 + 最终 test 一次性评估，属 G5）**",
        "",
        (
            "8. 封存全部配置与哈希、冻结模型集后一次性评估；置 `sap_phases.sap_c.status=COMPLETED` 与 "
            "`protocol.sees_final_test_predictions=true`；评估后任何修改都使结果失去确认性资格。"
        ),
    ]
    return "\n".join(lines) + "\n"


def stats_config_template(doc: Mapping[str, Any]) -> dict[str, Any]:
    """`stats_config.json` 模板：bootstrap 种子/次数等仍为 null（SAP-A 冻结时填写）。"""
    return {
        "generated_from": "configs/protocols/g0_sap.yaml",
        "protocol_version": get_path(doc, "protocol.version"),
        "sap_a_method_sha256": get_path(doc, "sap_phases.sap_a.method_sha256"),
        "sap_a_method_hash_algorithm": SAP_A_METHOD_HASH_ALGORITHM,
        "sap_phases": {
            "sap_a": get_path(doc, "sap_phases.sap_a.status"),
            "sap_b": get_path(doc, "sap_phases.sap_b.status"),
            "sap_c": get_path(doc, "sap_phases.sap_c.status"),
        },
        "analysis_unit": get_path(doc, "statistics.analysis_unit"),
        "main_effect": get_path(doc, "statistics.main_effect"),
        "bootstrap": {
            "type": get_path(doc, "statistics.bootstrap.type"),
            "n_resamples": get_path(doc, "statistics.bootstrap.n_resamples"),
            "seed": get_path(doc, "statistics.bootstrap.seed"),
            "ci": get_path(doc, "statistics.bootstrap.ci"),
        },
        "ci_interpretation": get_path(doc, "statistics.ci_interpretation"),
        "seed_reporting_required": get_path(doc, "statistics.seed_reporting_required"),
        "note": "模板：n_resamples/seed 必须由研究者在 SAP-A 冻结前填写；SAP-B 只填写 sap_b_results，不修改本块",
    }


__all__ = [
    "FREEZE_REQUIRED_FIELDS",
    "POWER_MEMO_FIELDS",
    "PRIMARY_ENDPOINT",
    "REQUIRED_COMPARISONS",
    "REQUIRED_SECTIONS",
    "SAP_A_METHOD_HASH_ALGORITHM",
    "SAP_A_METHOD_PATHS",
    "SAP_A_STATUSES",
    "SAP_B_ALL_FIELDS",
    "SAP_B_HASH_FIELDS",
    "SAP_B_POLICY_FIELDS",
    "SAP_B_REQUIRED_RESULT_FIELDS",
    "SAP_B_STATUSES",
    "SAP_C_STATUSES",
    "THRESHOLD_POLICIES",
    "assert_g0_sap_config",
    "canonical_json",
    "diff_sap_a_methods",
    "render_freeze_checklist",
    "render_power_memo_template",
    "sap_a_method_hash",
    "sap_a_method_snapshot",
    "stats_config_template",
    "validate_freeze_readiness",
    "validate_g0_sap_config",
    "validate_sap_phases",
]
