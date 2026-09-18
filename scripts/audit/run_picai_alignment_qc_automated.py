#!/usr/bin/env python
"""G0-R 全自动序列对齐 QC（Automated v0.3）：一键运行，无需人工阅片 / 人工 landmark / 双阅片 / 人工阈值。

用法（由**研究者**在本机运行；代理不得自行执行真实数据运行）：

    cd /opt/data/private/lm/my-projects
    conda activate lm
    source scripts/env_nnunet.sh

    G0R_MANIFEST="outputs/diagnostics/g0_r/20260915_093819/sampling_manifest.json"
    G0R_RUN="outputs/diagnostics/g0_r_automated/$(date -u +%Y%m%d_%H%M%S)"

    python scripts/audit/run_picai_alignment_qc_automated.py \
      --config configs/protocols/g0_r_alignment_qc_automated.yaml \
      --sampling-manifest "$G0R_MANIFEST" --out-dir "$G0R_RUN" --dry-run

    python scripts/audit/run_picai_alignment_qc_automated.py \
      --config configs/protocols/g0_r_alignment_qc_automated.yaml \
      --sampling-manifest "$G0R_MANIFEST" --out-dir "$G0R_RUN"

流程（全自动）：

    A 物理几何与 FOV  →  B 多尺度跨模态边缘一致性  →  C 诊断性刚体残余估计（内存中）
      →  D 合成已知位移校准（在内存中注入已知位移，自动推导候选阈值）
      →  单例判定（ACCEPTABLE / FLAGGED / FOV_INSUFFICIENT / INVALID_INPUT /
                 REGISTRATION_DIAGNOSTIC_FAILED / INSUFFICIENT_EVIDENCE）
      →  总体候选决策（resample-only / resample+rigid / INSUFFICIENT_EVIDENCE，DRAFT）

边界：只**读**物化影像；**不写回、不修改任何医学影像**；不联网、不调用外部 API/云服务；
不使用 lesion / WG / PZ-TZ 标签或任何模型预测；输出为 DRAFT 候选，**不等于 G0-R PASS**。

v0.3：输出 JSON 一律带 `schema_version = g0-r-automated/0.3` 与协议身份；校准按 `(pair, case_id)` unit 分组，
真实病例阈值按 pair 推导（`decide_case` 必须显式传 pair，未知 pair 直接失败）。
读取本工具输出必须用 `load_output_json()`（schema 不匹配即拒绝），**不得**把 v0.2 旧输出按 v0.3 schema 解释。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from zonal_reliability_fusion.protocols import common as pc
from zonal_reliability_fusion.protocols import g0_r_automated_qc as aq
from zonal_reliability_fusion.protocols import g0_r_qc
from zonal_reliability_fusion.utils import make_progress, resolve_progress

DEFAULT_CONFIG = PROJECT_ROOT / "configs/protocols/g0_r_alignment_qc_automated.yaml"

METRICS_CSV = "automated_metrics.csv"
DECISIONS_CSV = "per_case_decisions.csv"
SKIPPED_CSV = "skipped.csv"


def assert_schema_version(doc: Mapping[str, Any]) -> str:
    """配置声明的输出 schema 必须与代码常量一致（禁止 v0.2 / v0.3 输出混用）。"""
    declared = str(pc.get_path(doc, "protocol.output_schema_version", default=""))
    if declared != aq.OUTPUT_SCHEMA_VERSION:
        raise SystemExit(
            f"[g0-r-automated] 配置 protocol.output_schema_version={declared!r} 与代码常量 "
            f"{aq.OUTPUT_SCHEMA_VERSION!r} 不一致；不得把不同 schema 的输出混用（请先升/降版本或改配置）"
        )
    return declared


def schema_envelope(
    payload: Mapping[str, Any], *, protocol: Mapping[str, Any], protocol_hash: str
) -> dict[str, Any]:
    """给每个 JSON 输出加 schema 与协议身份（避免 v0.2 与 v0.3 结果被当成同一种）。"""
    return {
        "schema_version": aq.OUTPUT_SCHEMA_VERSION,
        "protocol_id": protocol.get("id"),
        "protocol_version": protocol.get("version"),
        "protocol_status": protocol.get("status"),
        "protocol_hash": protocol_hash,
        "draft": True,
        "calibration_grouping": list(aq.CALIBRATION_GROUPING),
        **dict(payload),
    }


def load_output_json(path: str | Path, *, expect_schema: str = aq.OUTPUT_SCHEMA_VERSION) -> dict[str, Any]:
    """读取本工具输出并**校验 schema**；缺失或不匹配一律拒绝（不做静默兼容解释）。"""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    found = payload.get("schema_version")
    if found != expect_schema:
        raise ValueError(
            f"输出 schema 不兼容: {path} 记录 schema_version={found!r}，期望 {expect_schema!r}"
            "（v0.2 输出属历史证据，不得按 v0.3 schema 解释）"
        )
    return payload


# --------------------------------------------------------------------------- CLI
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="G0-R 全自动序列对齐 QC（Automated v0.2；只读影像、不配准写回、不判 PASS）"
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="自动协议配置（默认 configs/protocols/g0_r_alignment_qc_automated.yaml）")
    parser.add_argument("--sampling-manifest", default="", help="冻结的抽样 manifest（默认取配置 inputs.sampling_manifest）")
    parser.add_argument("--out-dir", default="", help="输出目录（必须为空/不存在；默认 <outputs.dir>/<UTC 时间戳>）")
    parser.add_argument("--materialized-root", default="", help="物化影像根目录（默认取配置 inputs.materialized_root）")
    parser.add_argument("--case-ids", default="", help="仅处理指定病例（逗号分隔；默认按 manifest 顺序全部）")
    parser.add_argument("--max-cases", type=int, default=0, help="仅处理前 N 例（0 = 全部；调试用）")
    parser.add_argument("--dry-run", action="store_true", help="只打印计划：不读影像体素、不写任何文件")
    parser.add_argument("--no-progress", action="store_true", help="关闭进度条（日志环境）")
    parser.add_argument("--overlay", action="store_true", help="额外写每例每对 1 张 checkerboard PNG（仅审计用，无需人工查看）")
    return parser.parse_args(argv)


# --------------------------------------------------------------------------- 工具
def write_rows(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=list(fieldnames)).writeheader()
        return path
    columns = list(fieldnames) or sorted({key for row in rows for key in row})
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})
    return path


def json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def assert_out_dir_empty(out_dir: Path) -> Path:
    """输出目录必须为新建/空目录：非空即拒绝覆盖（不删除、不覆盖既有产物）。"""
    pc.assert_output_dir_isolated(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(
            f"[g0-r-automated] 输出目录非空，拒绝覆盖: {out_dir}（请改用新的时间戳目录，不删除既有产物）"
        )
    return out_dir


def limited(items: Sequence[Any], n: int = 5) -> list[Any]:
    return list(items[:n])


# --------------------------------------------------------------------------- 单例处理
def process_pair(
    *,
    case_id: str,
    pair: str,
    t2w_arr: np.ndarray,
    moving_arr: np.ndarray,
    record: Any,
    prep: Mapping[str, Any],
    metrics_cfg: Mapping[str, Any],
    fov_policy: Mapping[str, Any],
    reg_cfg: Mapping[str, Any],
    registration_runner: Callable[..., Mapping[str, Any]] | None,
) -> dict[str, Any]:
    """A/B/C 三组证据：几何+FOV、多尺度边缘一致性、诊断性刚体残余估计。"""
    spacing = tuple(float(v) for v in record.t2w_spacing)
    record_dict = record.as_dict()
    sanity_t2w = aq.check_array_sanity(t2w_arr)
    sanity_moving = aq.check_array_sanity(moving_arr)
    geometry_errors = aq.check_geometry_sanity(record_dict, role="t2w") + aq.check_geometry_sanity(
        record_dict, role="moving"
    )
    fov = aq.fov_evidence(record_dict, t2w_arr.shape, policy=fov_policy)
    min_nonzero = float(fov_policy["min_nonzero_voxel_fraction"])
    invalid_reasons: list[str] = []
    if not sanity_t2w.get("ok"):
        invalid_reasons.append(f"T2W 影像体检未通过: {sanity_t2w}")
    if not sanity_moving.get("ok"):
        invalid_reasons.append(f"moving 影像体检未通过: {sanity_moving}")
    if geometry_errors:
        invalid_reasons.append("几何异常: " + "; ".join(geometry_errors))
    if float(sanity_moving.get("nonzero_fraction", 0.0)) < min_nonzero:
        invalid_reasons.append(
            f"moving 非背景体素比例 {float(sanity_moving.get('nonzero_fraction', 0.0)):.5f} < {min_nonzero}"
        )
    if sanity_t2w.get("shape") != sanity_moving.get("shape"):
        invalid_reasons.append(f"形状不一致: {sanity_t2w.get('shape')} vs {sanity_moving.get('shape')}")

    slicewise_radius = float(metrics_cfg["tolerance_radii_mm"][1]) if metrics_cfg.get("slicewise") else None
    metrics = aq.pair_metrics_from_arrays(
        t2w_arr,
        moving_arr,
        spacing,
        prep=prep,
        metrics_cfg=metrics_cfg,
        slicewise_radius_mm=slicewise_radius,
    )
    registration = aq.estimate_rigid_diagnostic(
        t2w_arr, moving_arr, spacing, cfg=reg_cfg, runner=registration_runner
    )
    return {
        "case_id": case_id,
        "pair": pair,
        "series": g0_r_qc.PAIR_TO_SERIES[pair],
        "spacing_xyz": list(spacing),
        "shape_zyx": [int(v) for v in t2w_arr.shape],
        "sanity_t2w": sanity_t2w,
        "sanity_moving": sanity_moving,
        "geometry_errors": geometry_errors,
        "fov": fov,
        "invalid_reasons": invalid_reasons,
        "metrics": metrics,
        "registration": registration,
        "record": record_dict,
    }


def _registration_status(row: Mapping[str, Any]) -> str:
    return str((row.get("registration") or {}).get("status", "UNKNOWN"))


def decide_row_status(row: Mapping[str, Any], thresholds: Mapping[str, Any], calibration: Mapping[str, Any],
                      metrics_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """单例状态（顺序冻结）：INVALID_INPUT → FOV_INSUFFICIENT → REGISTRATION_FAILED → 指标判定。"""
    if row["invalid_reasons"]:
        return {
            "status": aq.CASE_INVALID_INPUT,
            "reason": "; ".join(row["invalid_reasons"]),
            "checks": [],
        }
    if not (row.get("fov") or {}).get("fov_ok"):
        return {
            "status": aq.CASE_FOV_INSUFFICIENT,
            "reason": "; ".join((row.get("fov") or {}).get("fov_reasons") or ["FOV 不足"]),
            "checks": [],
        }
    if _registration_status(row) != "OK":
        return {
            "status": aq.CASE_REGISTRATION_FAILED,
            "reason": str((row.get("registration") or {}).get("reason", "刚体诊断失败")),
            "checks": [],
        }
    return aq.decide_case(
        row["metrics"], thresholds, calibration, pair=str(row["pair"]), metrics_cfg=metrics_cfg
    )


# --------------------------------------------------------------------------- 输出
def build_metrics_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        flat = aq.flatten_metric_row(str(row["case_id"]), str(row["pair"]), row["metrics"])
        reg = row.get("registration") or {}
        flat["registration_status"] = reg.get("status", "UNKNOWN")
        flat["registration_translation_magnitude_mm"] = reg.get("translation_magnitude_mm")
        flat["registration_rotation_magnitude_deg"] = reg.get("rotation_magnitude_deg")
        flat["registration_metric_before"] = reg.get("metric_before")
        flat["registration_metric_after"] = reg.get("metric_after")
        flat["fov_ok"] = int(bool((row.get("fov") or {}).get("fov_ok")))
        flat["fov_min_overlap_ratio"] = (
            min((row.get("fov") or {}).get("overlap_ratio") or [float("nan")])
            if (row.get("fov") or {}).get("overlap_ratio")
            else None
        )
        flat["n_invalid_reasons"] = len(row.get("invalid_reasons") or [])
        out.append(flat)
    return out


def build_decision_rows(rows: Sequence[Mapping[str, Any]], statuses: Mapping[tuple[str, str], Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        key = (str(row["case_id"]), str(row["pair"]))
        status = statuses[key]
        checks = status.get("checks") or []
        primary = next((c for c in checks if c.get("role") == "primary"), {})
        reg = row.get("registration") or {}
        out.append(
            {
                "case_id": row["case_id"],
                "pair": row["pair"],
                "status": status["status"],
                "reason": status.get("reason", ""),
                "primary_metric": primary.get("metric", ""),
                "primary_value": primary.get("value", ""),
                "primary_threshold": primary.get("threshold", ""),
                "primary_decision": primary.get("decision", ""),
                "checks_json": json_cell(checks),
                "registration_status": reg.get("status", "UNKNOWN"),
                "registration_translation_mm": reg.get("translation_magnitude_mm", ""),
                "registration_rotation_deg": reg.get("rotation_magnitude_deg", ""),
                "fov_ok": int(bool((row.get("fov") or {}).get("fov_ok"))),
                "fov_reasons_json": json_cell((row.get("fov") or {}).get("fov_reasons") or []),
                "invalid_reasons_json": json_cell(row.get("invalid_reasons") or []),
            }
        )
    return out


def pair_conclusions(decision_rows: Sequence[Mapping[str, Any]], pairs: Sequence[str]) -> dict[str, Any]:
    """ADC / HBV 独立结论（不合并、不互相外推）。"""
    out: dict[str, Any] = {}
    for pair in pairs:
        subset = [r for r in decision_rows if r["pair"] == pair]
        counts = Counter(str(r["status"]) for r in subset)
        n = len(subset)
        flagged = counts.get(aq.CASE_FLAGGED, 0)
        acceptable = counts.get(aq.CASE_ACCEPTABLE, 0)
        insufficient_statuses = (
            aq.CASE_INSUFFICIENT,
            aq.CASE_INVALID_INPUT,
            aq.CASE_FOV_INSUFFICIENT,
            aq.CASE_REGISTRATION_FAILED,
        )
        if n == 0:
            verdict = "NO_DATA"
        elif any(counts.get(status, 0) for status in insufficient_statuses):
            verdict = "INSUFFICIENT_EVIDENCE"
        elif flagged > 0:
            verdict = "SYSTEMATIC_MISALIGNMENT_DETECTED"
        elif acceptable == n:
            verdict = "NO_SYSTEMATIC_MISALIGNMENT_DETECTED"
        else:
            verdict = "INSUFFICIENT_EVIDENCE"
        out[pair] = {
            "n": n,
            "counts": dict(sorted(counts.items())),
            "n_flagged": flagged,
            "n_acceptable": acceptable,
            "verdict": verdict,
            "flagged_cases": sorted({str(r["case_id"]) for r in subset if r["status"] == aq.CASE_FLAGGED}),
        }
    return out


def write_overlay_png(
    overlay_dir: Path, row: Mapping[str, Any], t2w: np.ndarray, moving: np.ndarray
) -> str | None:
    """可选审计材料：1 张中间层 checkerboard PNG（**仅审计用，无需人工查看**），失败不阻断流程。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001 - 可选依赖缺失不应阻断主流程  # pragma: no cover
        print(f"[g0-r-automated][WARN] overlay 跳过（matplotlib 不可用: {exc}）")
        return None
    overlay_dir.mkdir(parents=True, exist_ok=True)
    k = int(np.asarray(t2w).shape[0] // 2)
    panel = g0_r_qc.checkerboard_image(
        g0_r_qc.normalize_for_display(np.asarray(t2w)[k]), g0_r_qc.normalize_for_display(np.asarray(moving)[k])
    )
    fig = plt.figure(figsize=(4, 4), dpi=100)
    axis = fig.add_subplot(1, 1, 1)
    axis.imshow(panel, cmap="gray")
    axis.set_title(f"{row['case_id']} {row['pair']} k={k} resample-ONLY", fontsize=7)
    axis.axis("off")
    name = f"{row['case_id']}__{row['pair']}.png"
    fig.savefig(overlay_dir / name, bbox_inches="tight")
    plt.close(fig)
    return name


def write_report(
    path: Path,
    *,
    protocol: Mapping[str, Any],
    protocol_hash: str,
    manifest_path: Path,
    out_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    decision_rows: Sequence[Mapping[str, Any]],
    calibration: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    overall: Mapping[str, Any],
    conclusions: Mapping[str, Any],
    elapsed: float,
) -> None:
    """写出 `qc_report.md`（纯数值/文本，不需要任何人查看图像）。"""
    counts = Counter(str(r["status"]) for r in decision_rows)
    lines: list[str] = []
    lines.append("# G0-R 全自动序列对齐 QC 报告（Automated v0.3，DRAFT）\n")
    lines.append(
        f"- 协议：`{protocol.get('id')}` / `{protocol.get('version')}` / status=`{protocol.get('status')}`\n"
        f"- 协议哈希（protocol_hash）：`{protocol_hash}`\n"
        f"- 输出 schema：`{calibration.get('schema_version')}`；校准分组：{calibration.get('grouping')}"
        f"（unit = {calibration.get('unit_definition')}）\n"
        f"- 抽样 manifest：`{manifest_path}`\n"
        f"- 输出目录：`{out_dir}`\n"
        f"- 用时：{elapsed:.1f}s；case×pair 行数：{len(decision_rows)}\n"
        f"- **本报告为自动流程产物（DRAFT）；不是人工阅片、也不是双阅片证据，不能替代 G0-R 冻结。**\n"
    )
    lines.append("## 1. 输入完整性\n")
    lines.append(f"- 期望病例数：{overall.get('n_unique_cases')}；case×pair 行数：{len(decision_rows)}\n")
    lines.append(f"- 状态计数：{dict(sorted(counts.items()))}\n")
    lines.append(
        f"- 输入异常（INVALID_INPUT）：{[r['case_id'] + '/' + r['pair'] for r in decision_rows if r['status'] == aq.CASE_INVALID_INPUT] or '无'}\n"
    )
    lines.append("## 2. FOV / 几何异常\n")
    fov_bad = [r for r in decision_rows if r["status"] == aq.CASE_FOV_INSUFFICIENT]
    lines.append(f"- FOV 不足：{[r['case_id'] + '/' + r['pair'] for r in fov_bad] or '无'}\n")
    for row in rows:
        if row.get("geometry_errors"):
            lines.append(f"- 几何异常 {row['case_id']}/{row['pair']}：{'; '.join(row['geometry_errors'])}\n")
    lines.append("## 3. T2W-ADC 自动结论\n")
    lines.append(f"- {json.dumps(conclusions.get('T2W-ADC', {}), ensure_ascii=False)}\n")
    lines.append("## 4. T2W-HBV 自动结论\n")
    lines.append(f"- {json.dumps(conclusions.get('T2W-HBV', {}), ensure_ascii=False)}\n")
    lines.append("## 5. 合成已知位移校准（v0.3：按 (pair, case_id) unit 分组）\n")
    lines.append(
        f"- 是否通过（**每个 pair 的主指标都必须通过**）：**{calibration.get('passed')}**"
        f"（主指标 `{calibration.get('primary_metric')}`）\n"
        f"- 零位移参考：{calibration.get('zero_reference')}；噪声参考：{calibration.get('zero_noise_reference')}\n"
        f"- 噪声阈值公式：{calibration.get('noise_threshold_formula')}\n"
        f"- 退化定义：{calibration.get('degradation_definition')}\n"
        f"- 校准病例：{calibration.get('cases_used')}；unit 数：{calibration.get('n_units')}；"
        f"记录数：{calibration.get('n_records')}；缺失 pair：{calibration.get('missing_pairs')}\n"
        f"- 条件：{limited(calibration.get('conditions') or [], 8)} …（共 {len(calibration.get('conditions') or [])} 条）\n"
        f"- min_detectable_mm：{calibration.get('min_detectable_mm')}\n"
    )
    for pair, info in (calibration.get("per_pair") or {}).items():
        lines.append(
            f"- **pair `{pair}`**：passed={info.get('passed')} n_units={info.get('n_units')} "
            f"reasons={info.get('reasons')}\n"
        )
        for name, metric in (info.get("per_metric") or {}).items():
            if not metric.get("available"):
                lines.append(f"  - `{name}`：不可用（{metric.get('reason')}）\n")
                continue
            lines.append(
                f"  - `{name}`（{metric.get('direction')}）：passed={metric.get('passed')} "
                f"noise_threshold={metric.get('noise_threshold'):.6g} "
                f"monotonicity={metric.get('monotonicity'):.3f} "
                f"detect@min={metric.get('detection_rate_at_min_detectable'):.3f} "
                f"zero_FPR={metric.get('zero_false_positive_rate'):.3f} reasons={metric.get('reasons')}\n"
            )
            baselines = metric.get("unit_zero_baselines") or {}
            mins = metric.get("unit_min_detectable_means") or {}
            mids = metric.get("unit_midpoints") or {}
            lines.append(
                "    unit zero baseline / min-detectable mean / midpoint："
                + json.dumps(
                    {
                        str(case_id): [
                            baselines.get(case_id),
                            mins.get(case_id),
                            mids.get(case_id),
                        ]
                        for case_id in sorted(baselines)
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    lines.append("## 6. 自动候选决策（DRAFT）\n")
    lines.append(
        f"- 候选：**{overall.get('candidate')}**\n"
        f"- 触发原因：{overall.get('reasons') or '无'}\n"
        f"- 分状态占比：{ {k: round(float(v), 4) for k, v in (overall.get('fractions') or {}).items()} }\n"
        f"- 每 pair 真实病例阈值（自动推导）："
        f"{json.dumps({k: {p: e.get('threshold') for p, e in ((v.get('by_pair') or {}).items())} for k, v in (thresholds.get('metrics') or {}).items()}, ensure_ascii=False)}\n"
        f"- 分 pair 校准是否通过：{json.dumps(overall.get('pair_calibration_passed'), ensure_ascii=False)}\n"
        f"- 无校准的 pair：{overall.get('pairs_without_calibration') or '无'}\n"
        f"- 需要刚体校正的 case×pair：{overall.get('rigid_required_by') or '无'}\n"
    )
    lines.append("## 7. 不能冻结的原因\n")
    lines.append(
        "- 本流程只产出**候选**（DRAFT）。G0-R 冻结仍需研究者按 `docs/protocols/G0_R_ALIGNMENT_QC_AUTOMATED.md` §9\n"
        "  确认协议哈希、写入 `frozen_decision`、`reviewer`、`frozen_at`；在此之前 G0-R 仍为 Pending。\n"
        "- 即使 draft-0.3 工具运行成功（`calibration passed=true` 且候选非 `INSUFFICIENT_EVIDENCE`），\n"
        "  **也不自动使 G0-R PASS**；自动结论只是相对 QC 候选，0 mm 状态不等于解剖学对齐真值。\n"
    )
    if overall.get("candidate") == aq.DECISION_INSUFFICIENT:
        lines.append("- 本次结果为 `INSUFFICIENT_EVIDENCE`：**必须停止**，不得据此启动 P2B / 正式训练。\n")
    if overall.get("candidate") == aq.DECISION_RESAMPLE_RIGID:
        lines.append(
            "- 本次候选为 `resample+rigid`：若被研究者采纳，必须**新建派生数据、重新预处理并复检**，\n"
            "  不得在旧派生数据上直接训练；正式 M0/N0 仍须等待 G0-R 冻结。\n"
        )
    lines.append("## 8. 局限性\n")
    lines.append(
        "- 跨模态边缘一致性是**相对指标**：其绝对数值依赖强度标准化与边缘阈值，判定完全依赖合成位移校准\n"
        "  （0 mm vs min_detectable），不得解释为解剖学真值；\n"
        "- 诊断性刚体估计使用互信息 + 多分辨率梯度下降，对小 FOV、低对比或强各向异性数据可能失败（记为\n"
        "  `REGISTRATION_DIAGNOSTIC_FAILED`，不伪造零位移）；\n"
        "- 校准仅在部分病例（`calibration.max_cases`）上进行，用于推导阈值；不同中心/协议的对比度差异会带来\n"
        "  系统不确定性；\n"
        "- slicewise 统计仅供分布审计，不参与阈值判定；\n"
        "- 本流程**不读 lesion / WG / PZ-TZ，不使用任何模型预测**；不涉及任何人工阅片或双阅片结论。\n"
    )
    lines.append("## 9. 人工阅片声明\n")
    lines.append(
        "- 本报告由自动流程生成：**没有**人工看图、**没有**人工 landmark、**没有**双阅片、**没有**人工阈值选择。\n"
        "  报告中任何数值都不得被表述为人工阅片或双阅片证据。\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- 主流程
def run_pipeline(
    *,
    case_ids: Sequence[str],
    pairs: Sequence[str],
    materialized_root: Path,
    doc: Mapping[str, Any],
    manifest: Mapping[str, Any],
    manifest_path: Path,
    out_dir: Path,
    progress: bool,
    config_path: Path,
    registration_runner: Callable[..., Mapping[str, Any]] | None = None,
    overlay: bool = False,
) -> dict[str, Any]:
    """执行 A→D 全流程并写出全部产物（真实数据路径；由研究者启动）。"""
    started = time.time()
    prep = dict(doc["preprocessing"])
    metrics_cfg = dict(doc["metrics"])
    fov_policy = dict(doc["fov_policy"])
    cal_cfg = dict(doc["calibration"])
    reg_cfg = dict(doc["registration_diagnostic"])
    thresholds_cfg = dict(doc["thresholds"])
    decision_cfg = dict(doc["decision"])
    threshold_aggregation = str(
        thresholds_cfg.get("aggregation_by_pair", aq.PAIR_THRESHOLD_AGGREGATION)
    )
    protocol = dict(doc.get("protocol") or {})
    protocol_hash = aq.protocol_hash(doc)

    assert_out_dir_empty(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    geometry: dict[str, Any] = {}
    input_hashes: dict[str, str] = {}

    # ---------------- 阶段 1：读取（只读）→ A/B/C 证据 →（可选）校准与 overlay（逐例处理并释放数组）
    calibration_records: list[dict[str, Any]] = []
    cal_cases_used: list[str] = []
    overlay_written: list[str] = []
    overlay_dir = out_dir / "overlay"
    n_cal_cases = max(0, int(cal_cfg["max_cases"]))
    conditions = aq.build_calibration_conditions(cal_cfg)
    total_cal_steps = n_cal_cases * max(1, len(pairs)) * len(conditions)
    cal_bar = make_progress(
        list(range(total_cal_steps)), total=total_cal_steps, desc="calibration", disable=not progress
    )
    cal_iter = iter(cal_bar)

    def advance_calibration(_label: str) -> None:
        try:
            next(cal_iter)
        except StopIteration:  # pragma: no cover - 防御性
            pass

    for case_id in make_progress(list(case_ids), total=len(case_ids), desc="cases", disable=not progress):
        paths = g0_r_qc.resolve_series_paths(materialized_root, case_id)
        if not g0_r_qc.case_dir_for(materialized_root, case_id).is_dir():
            skipped.append({"case_id": case_id, "pair": "", "reason": "物化目录不存在", "stage": "read"})
            continue
        missing = g0_r_qc.missing_series(paths)
        case_entry: dict[str, Any] = {"pairs": {}}
        calibrate_this_case = len(cal_cases_used) < n_cal_cases
        case_calibrated = False
        for pair in pairs:
            series = g0_r_qc.PAIR_TO_SERIES[pair]
            if series in missing:
                skipped.append(
                    {"case_id": case_id, "pair": pair, "reason": f"缺少序列文件: {series}", "stage": "read"}
                )
                continue
            for key in ("t2w", series):
                path = paths[key]
                if path.is_file():
                    input_hashes[str(path)] = pc.sha256_file(path)
            try:
                t2w_arr, moving_arr, record = g0_r_qc.prepare_pair_arrays(
                    paths["t2w"], paths[series], case_id=case_id, pair=pair
                )
            except Exception as exc:  # noqa: BLE001 - 失败如实记录，不静默跳过
                skipped.append(
                    {"case_id": case_id, "pair": pair, "reason": f"读取/重采样失败: {exc}", "stage": "read"}
                )
                continue
            t2w_arr = np.asarray(t2w_arr)
            moving_arr = np.asarray(moving_arr)
            row = process_pair(
                case_id=case_id,
                pair=pair,
                t2w_arr=t2w_arr,
                moving_arr=moving_arr,
                record=record,
                prep=prep,
                metrics_cfg=metrics_cfg,
                fov_policy=fov_policy,
                reg_cfg=reg_cfg,
                registration_runner=registration_runner,
            )
            rows.append(row)
            case_entry["pairs"][pair] = record.as_dict()
            case_entry.setdefault(
                "t2w",
                {
                    "size": list(record.t2w_size),
                    "spacing": list(record.t2w_spacing),
                    "origin": list(record.t2w_origin),
                    "direction": list(record.t2w_direction),
                    "orientation": record.t2w_orientation,
                },
            )
            # D 组：合成位移校准（嵌入读取循环：仅在内存中扰动，用完即释放，不长期持有全部影像）
            if calibrate_this_case and not row["invalid_reasons"]:
                calibration_records.extend(
                    aq.calibration_records_for_case(
                        {
                            "case_id": case_id,
                            "pair": pair,
                            "t2w": t2w_arr,
                            "moving": moving_arr,
                            "spacing_xyz": row["spacing_xyz"],
                            "case_index": len(cal_cases_used),
                        },
                        cfg=cal_cfg,
                        prep=prep,
                        metrics_cfg=metrics_cfg,
                        progress_cb=advance_calibration,
                    )
                )
                case_calibrated = True
            if overlay:
                name = write_overlay_png(overlay_dir, row, t2w_arr, moving_arr)
                if name:
                    overlay_written.append(name)
            del t2w_arr, moving_arr
        if calibrate_this_case and case_calibrated:
            cal_cases_used.append(case_id)
        geometry[case_id] = case_entry

    # ---------------- 阶段 2：校准汇总 + 阈值推导
    calibration = aq.summarize_calibration(
        calibration_records,
        cfg=cal_cfg,
        metrics_cfg=metrics_cfg,
        threshold_aggregation=threshold_aggregation,
    )
    thresholds = aq.derive_thresholds(calibration, metrics_cfg=metrics_cfg, thresholds_cfg=thresholds_cfg)

    # ---------------- 阶段 3：单例判定 + 总体决策
    statuses: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        statuses[(str(row["case_id"]), str(row["pair"]))] = decide_row_status(
            row, thresholds, calibration, metrics_cfg
        )
    decision_rows = build_decision_rows(rows, statuses)
    overall = aq.decide_overall(
        [{"case_id": r["case_id"], "pair": r["pair"], "status": statuses[(r["case_id"], r["pair"])]["status"]} for r in rows],
        calibration,
        decision_cfg=decision_cfg,
    )
    conclusions = pair_conclusions(decision_rows, pairs)

    # ---------------- 输出（真实运行才写）
    metrics_rows = build_metrics_rows(rows)
    write_rows(out_dir / METRICS_CSV, metrics_rows, sorted({k for row in metrics_rows for k in row}))
    write_rows(
        out_dir / DECISIONS_CSV,
        decision_rows,
        [
            "case_id", "pair", "status", "reason", "primary_metric", "primary_value", "primary_threshold",
            "primary_decision", "checks_json", "registration_status", "registration_translation_mm",
            "registration_rotation_deg", "fov_ok", "fov_reasons_json", "invalid_reasons_json",
        ],
    )
    write_rows(out_dir / SKIPPED_CSV, skipped, ["case_id", "pair", "stage", "reason"])
    envelope = {"protocol": protocol, "protocol_hash": protocol_hash}
    pc.write_json(
        out_dir / "calibration_summary.json",
        schema_envelope(calibration, **envelope),
    )
    pc.write_json(
        out_dir / "threshold_derivation.json",
        schema_envelope(thresholds, **envelope),
    )
    pc.write_json(
        out_dir / "decision_draft.json",
        schema_envelope({**overall, "pair_conclusions": conclusions}, **envelope),
    )
    pc.write_json(
        out_dir / "qc_geometry.json",
        schema_envelope(
            {
                "source": "physical-space resample only (NO registration applied to data)",
                "cases": geometry,
            },
            **envelope,
        ),
    )
    pc.write_json(
        out_dir / "input_hashes.json",
        schema_envelope({"input_sha256": dict(input_hashes)}, **envelope),
    )
    write_report(
        out_dir / "qc_report.md",
        protocol=protocol,
        protocol_hash=protocol_hash,
        manifest_path=manifest_path,
        out_dir=out_dir,
        rows=rows,
        decision_rows=decision_rows,
        calibration=calibration,
        thresholds=thresholds,
        overall=overall,
        conclusions=conclusions,
        elapsed=time.time() - started,
    )
    pc.write_json(
        out_dir / "run_metadata.json",
        pc.build_run_metadata(
            tool="scripts/audit/run_picai_alignment_qc_automated.py",
            config_path=config_path,
            extra={
                "mode": "automated-qc",
                "protocol": protocol,
                "protocol_hash": protocol_hash,
                "sampling_manifest": str(manifest_path),
                "sampling_manifest_sha256": pc.sha256_file(manifest_path),
                "materialized_root": str(materialized_root),
                "case_ids_requested": list(case_ids),
                "pairs": list(pairs),
                "n_rows": len(rows),
                "n_skipped": len(skipped),
                "n_overlay_png": len(overlay_written),
                "schema_version": aq.OUTPUT_SCHEMA_VERSION,
                "calibration_grouping": list(calibration.get("grouping") or aq.CALIBRATION_GROUPING),
                "calibration_passed": bool(calibration.get("passed")),
                "calibration_passed_by_pair": {
                    str(pair): bool((info or {}).get("passed"))
                    for pair, info in sorted((calibration.get("per_pair") or {}).items())
                },
                "thresholds_by_pair": {
                    str(name): {
                        str(pair): entry.get("threshold")
                        for pair, entry in sorted(((info or {}).get("by_pair") or {}).items())
                    }
                    for name, info in sorted((thresholds.get("metrics") or {}).items())
                },
                "candidate": overall.get("candidate"),
                "insufficient_evidence": bool(overall.get("candidate") == aq.DECISION_INSUFFICIENT),
                "outputs": sorted(p.name for p in out_dir.iterdir()),
                "data_written_back": False,
                "note": "自动候选为 DRAFT；不等于 G0-R PASS；本工具不写回任何影像",
            },
        ),
    )

    return {
        "rows": rows,
        "decision_rows": decision_rows,
        "statuses": statuses,
        "calibration": calibration,
        "thresholds": thresholds,
        "overall": overall,
        "conclusions": conclusions,
        "skipped": skipped,
        "out_dir": out_dir,
        "elapsed": time.time() - started,
    }


def print_summary(result: Mapping[str, Any]) -> None:
    decision_rows = result["decision_rows"]
    counts = Counter(str(r["status"]) for r in decision_rows)
    overall = result["overall"]
    calibration = result["calibration"]
    print()
    print("=" * 78)
    print("[g0-r-automated] 运行结束")
    print(f"  成功行（case×pair）: {len(decision_rows)}；跳过: {len(result['skipped'])}")
    print(f"  状态明细: {dict(sorted(counts.items()))}")
    for pair, info in (result.get("conclusions") or {}).items():
        print(f"  {pair} 自动结论: {info['verdict']}（n={info['n']}, flagged={info['n_flagged']}）")
    print(
        f"  合成校准是否通过（每个 pair 的主指标都必须通过）: {bool(calibration.get('passed'))}"
        f"（主指标 {calibration.get('primary_metric')}）"
    )
    for pair, info in sorted((calibration.get("per_pair") or {}).items()):
        print(f"    校准[{pair}]: passed={info.get('passed')} units={info.get('n_units')} reasons={info.get('reasons')}")
    for name, entry in sorted(((result.get("thresholds") or {}).get("metrics") or {}).items()):
        by_pair = {p: (e.get("threshold") if e.get("available") else "NA") for p, e in (entry.get("by_pair") or {}).items()}
        print(f"    阈值[{name}]: {by_pair}")
    print(f"  输出路径: {result['out_dir']}")
    print(f"  候选决策: {overall.get('candidate')}")
    print(f"  是否 INSUFFICIENT_EVIDENCE: {overall.get('candidate') == aq.DECISION_INSUFFICIENT}")
    can_freeze = (
        overall.get("candidate") in (aq.DECISION_RESAMPLE_ONLY, aq.DECISION_RESAMPLE_RIGID)
        and bool(calibration.get("passed"))
    )
    print(f"  是否可提交自动冻结候选: {can_freeze}（仍需研究者写入 frozen_decision / reviewer / frozen_at）")
    if overall.get("candidate") == aq.DECISION_INSUFFICIENT:
        print("  [STOP] 结果为 INSUFFICIENT_EVIDENCE：必须停止，不得据此启动 P2B / 正式训练。")
    print("=" * 78)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = Path(args.config)
    doc = pc.load_yaml_config(config_path)
    aq.assert_automated_input_policy(doc)
    schema_version = assert_schema_version(doc)
    protocol_hash = aq.protocol_hash(doc)
    progress = resolve_progress(args.no_progress, bool(doc.get("progress", True)))

    manifest_path = (
        Path(args.sampling_manifest)
        if args.sampling_manifest
        else pc.resolve_project_path(pc.get_path(doc, "inputs.sampling_manifest"))
    )
    if not manifest_path.is_file():
        raise SystemExit(f"[g0-r-automated] 找不到抽样 manifest: {manifest_path}")
    manifest = g0_r_qc.load_sampling_manifest(manifest_path)
    pairs = tuple(str(p) for p in manifest["pairs"])
    materialized_root = (
        Path(args.materialized_root)
        if args.materialized_root
        else pc.resolve_project_path(pc.get_path(doc, "inputs.materialized_root"))
    )

    requested = [c.strip() for c in args.case_ids.split(",") if c.strip()] or None
    case_ids = g0_r_qc.select_cases(manifest, requested)
    if args.max_cases > 0:
        case_ids = case_ids[: args.max_cases]
    if not case_ids:
        raise SystemExit("[g0-r-automated] 没有可处理的病例（检查 manifest 与 --case-ids）")

    if args.dry_run:
        print(f"[g0-r-automated][dry-run] 协议 {doc['protocol']['id']} / {doc['protocol']['version']} "
              f"status={doc['protocol']['status']} schema={schema_version} hash={protocol_hash[:16]}…")
        print(f"[g0-r-automated][dry-run] manifest={manifest_path}（{len(manifest.get('case_ids') or [])} 例）")
        print(f"[g0-r-automated][dry-run] materialized_root={materialized_root}")
        print(f"[g0-r-automated][dry-run] 将处理 {len(case_ids)} 例 × {len(pairs)} 对 = {len(case_ids) * len(pairs)} 行")
        for case_id in case_ids:
            paths = g0_r_qc.resolve_series_paths(materialized_root, case_id)
            present = ", ".join(f"{k}={'Y' if v.is_file() else 'N'}" for k, v in paths.items())
            print(f"  - {case_id}: {present}")
        print(f"[g0-r-automated][dry-run] 校准病例上限={doc['calibration']['max_cases']}；"
              f"位移={doc['calibration']['displacements_mm']}；方向数={len(doc['calibration']['directions'])}")
        print("[g0-r-automated][dry-run] 不读影像体素、不写任何文件；真实运行请去掉 --dry-run")
        print(f"[g0-r-automated][dry-run] 预计写出: {METRICS_CSV}, {DECISIONS_CSV}, calibration_summary.json, "
              f"threshold_derivation.json, decision_draft.json, qc_geometry.json, input_hashes.json, "
              f"run_metadata.json, {SKIPPED_CSV}, qc_report.md"
              + ("（+overlay/）" if args.overlay else ""))
        return 0

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else pc.resolve_project_path(pc.get_path(doc, "outputs.dir", default="outputs/diagnostics/g0_r_automated"))
        / pc.timestamp_slug()
    )
    assert_out_dir_empty(out_dir)

    print(f"[g0-r-automated] 启动：{len(case_ids)} 例 × {len(pairs)} 对；协议哈希 {protocol_hash[:16]}…")
    print("[g0-r-automated] 边界：只读影像、不写回、不联网、不使用 lesion/WG/PZ-TZ/模型预测")
    result = run_pipeline(
        case_ids=case_ids,
        pairs=pairs,
        materialized_root=materialized_root,
        doc=doc,
        manifest=manifest,
        manifest_path=manifest_path,
        out_dir=out_dir,
        progress=progress,
        config_path=config_path,
        overlay=args.overlay,
    )
    print_summary(result)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
