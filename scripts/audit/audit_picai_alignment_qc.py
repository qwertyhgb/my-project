#!/usr/bin/env python
"""G0-R 序列对齐 QC：确定性分层抽样 + 盲审清单 + 指标 schema（只读元数据）。

本工具做四件事（**不读取影像体素**、**不决定配准方案、不写 G0-R 结论**）：

1. 按 `configs/protocols/g0_r_alignment_qc.yaml` 的 `sampling.strata` 做**确定性分层抽样**
   （中心 × 阳阴性 × 几何异常 × 极端 FOV × 间距跨度）；
2. 输出**机器可读抽样 manifest**（含 `decision=null`，绝不代替研究者决策）；
3. 输出**双阅片者盲审记录模板**（字段取自协议 `readers.record_fields`，全部留空）；
4. 生成/校验**定量对齐指标 schema**（`--validate-metrics` 可校验已有指标 CSV）。

真实数据（manifest / 几何审计）的读取由**研究者执行**；本工具自身不读影像、不用 GPU、不判 PASS。

运行示例（研究者）：

    cd /opt/data/private/lm/my-projects
    conda activate lm
    python scripts/audit/audit_picai_alignment_qc.py --dry-run          # 只看抽样计划
    python scripts/audit/audit_picai_alignment_qc.py                    # 生成抽样与盲审清单
    python scripts/audit/audit_picai_alignment_qc.py --validate-metrics outputs/diagnostics/g0_r/<run>/alignment_metrics.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from zonal_reliability_fusion.protocols import g0_r  # noqa: E402
from zonal_reliability_fusion.protocols import common as pc  # noqa: E402
from zonal_reliability_fusion.utils import make_progress, resolve_progress  # noqa: E402

DEFAULT_CONFIG = PROJECT_ROOT / "configs/protocols/g0_r_alignment_qc.yaml"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="G0-R 对齐 QC 抽样与清单（只读；不决定配准/不判 PASS）")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG), help="协议配置（默认 configs/protocols/g0_r_alignment_qc.yaml）")
    ap.add_argument("--manifest", default="", help="覆盖配置中的 manifest 路径（data/metadata/picai_manifest.csv）")
    ap.add_argument("--out-dir", default="", help="输出目录（默认 <config outputs.dir>/<timestamp>）")
    ap.add_argument("--n-cases", type=int, default=0, help="覆盖抽样总量（0=使用配置值；配置为 null 时只取最小必选集合）")
    ap.add_argument("--dry-run", action="store_true", help="只打印抽样计划，不写任何文件")
    ap.add_argument("--no-progress", action="store_true", help="关闭进度条（日志环境）")
    ap.add_argument(
        "--validate-metrics",
        default="",
        help="只校验已有定量指标 CSV 是否符合 schema（列/取值/单位），不写文件",
    )
    return ap.parse_args()


def load_manifest_rows(manifest_path: Path) -> list[dict]:
    if not manifest_path.is_file():
        raise SystemExit(
            f"[g0-r] manifest 不存在: {manifest_path}\n"
            "（该文件由 P0A 审计生成；若路径不同请用 --manifest 覆盖）"
        )
    with open(manifest_path, newline="", encoding="utf-8") as fh:
        rows = [dict(r) for r in csv.DictReader(fh)]
    if not rows:
        raise SystemExit(f"[g0-r] manifest 为空: {manifest_path}")
    return rows


def validate_metrics(path: Path) -> int:
    if not path.is_file():
        raise SystemExit(f"[g0-r] 指标文件不存在: {path}")
    with open(path, newline="", encoding="utf-8") as fh:
        rows = [dict(r) for r in csv.DictReader(fh)]
    errors = g0_r.validate_alignment_metrics_rows(rows)
    print(f"[g0-r][validate-metrics] file={path} rows={len(rows)} errors={len(errors)}")
    for err in errors[:50]:
        print(f"  - {err}")
    if errors:
        print("[g0-r][validate-metrics] 结果: INVALID（修正后再进入盲审统计；本工具不写入任何结论）")
        return 1
    print("[g0-r][validate-metrics] 结果: schema OK（仍不代表 G0-R 通过）")
    return 0


def main() -> None:
    args = parse_args()
    t0 = time.time()

    if args.validate_metrics:
        sys.exit(validate_metrics(Path(args.validate_metrics)))

    config_path = Path(args.config)
    doc = pc.load_yaml_config(config_path)
    protocol_meta = pc.validate_protocol_block(doc, expected_id="G0-R")
    progress = resolve_progress(args.no_progress, True)

    manifest_value = args.manifest or str(pc.get_path(doc, "inputs.manifest", default=""))
    if not manifest_value:
        raise SystemExit("[g0-r] 配置缺少 inputs.manifest，且未用 --manifest 覆盖")
    manifest_path = pc.resolve_project_path(manifest_value)
    rows = load_manifest_rows(manifest_path)

    sampling_cfg = dict(doc.get("sampling") or {})
    if args.n_cases:
        sampling_cfg["n_cases"] = int(args.n_cases)
    result = g0_r.plan_sampling(rows, sampling_cfg)

    print(
        f"[g0-r] manifest={manifest_path} rows={len(rows)} "
        f"selected={result.total} n_cases_source={result.n_cases_source}"
    )
    for name, count in sorted(result.per_stratum.items()):
        print(f"[g0-r]   stratum {name}: {count}")
    for warning in result.warnings:
        print(f"[g0-r][WARN] {warning}")

    readers_cfg = dict(doc.get("readers") or {})
    pairs = tuple(pc.get_path(doc, "evaluation.pairs", default=list(g0_r.PAIRS)))
    blind_rows = g0_r.build_blind_review_rows(
        result, pairs=pairs, record_fields=readers_cfg.get("record_fields")
    )

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else pc.resolve_project_path(pc.get_path(doc, "outputs.dir", default="outputs/diagnostics/g0_r"))
        / pc.timestamp_slug()
    )
    planned_files = [
        "sampled_cases.csv",
        "sampling_manifest.json",
        "blind_review_sheet.csv",
        "alignment_metrics_schema.json",
        "run_metadata.json",
    ]

    if args.dry_run:
        print(f"[g0-r][dry-run] 输出目录（不创建）: {out_dir}")
        print(f"[g0-r][dry-run] 计划写入: {', '.join(planned_files)}")
        print(f"[g0-r][dry-run] 盲审表行数: {len(blind_rows)}（{result.total} 例 × {len(pairs)} 评价对）")
        print("[g0-r][dry-run] decision 保持 null；本工具不决定 resample-only / resample+rigid")
        return

    out_dir = pc.ensure_output_dir(out_dir)
    manifest_sha = pc.sha256_file(manifest_path)

    # 1) 抽样结果 CSV
    sampling_columns = [
        "case_id",
        "patient_id",
        "study_id",
        "center",
        "case_csPCa",
        *g0_r.GEOMETRY_COLUMNS,
        "strata",
        "selection_order",
    ]
    pc.write_rows_csv(out_dir / "sampled_cases.csv", result.rows, fieldnames=sampling_columns)

    # 2) 机器可读 manifest
    sampling_manifest = {
        "protocol": protocol_meta,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "inputs": {
            "config": str(config_path),
            "config_sha256": pc.sha256_file(config_path),
            "manifest": str(manifest_path),
            "manifest_sha256": manifest_sha,
        },
        "sampling": result.as_manifest(),
        "evaluation_plan": {
            "pairs": list(pairs),
            "units": pc.get_path(doc, "evaluation.units", default="mm"),
            "visual_scale_levels": pc.get_path(doc, "evaluation.visual_scale.levels"),
            "auxiliary_only": pc.get_path(doc, "evaluation.auxiliary_only", default=[]),
            "prohibited_as_sole_criterion": pc.get_path(doc, "evaluation.prohibited_as_sole_criterion", default=[]),
        },
        "status": "PENDING",
        "note": "抽样已完成；阈值、配准决策与 G0-R 结论仍待研究者按协议 §5/§6 冻结（decision=null）",
    }
    pc.write_json(out_dir / "sampling_manifest.json", sampling_manifest)

    # 3) 盲审记录模板（空表；进度条覆盖逐行写入的规模）
    fieldnames = list(readers_cfg.get("record_fields") or blind_rows[0].keys())
    bar = make_progress(blind_rows, total=len(blind_rows), desc="write blind-review sheet", disable=not progress)
    pc.write_rows_csv(out_dir / "blind_review_sheet.csv", list(bar), fieldnames=fieldnames)

    # 4) 指标 schema
    pc.write_json(out_dir / "alignment_metrics_schema.json", g0_r.metrics_schema_document())

    # 5) 运行元数据
    metadata = pc.build_run_metadata(
        tool="scripts/audit/audit_picai_alignment_qc.py",
        config_path=config_path,
        extra={
            "manifest": str(manifest_path),
            "manifest_sha256": manifest_sha,
            "output_dir": str(out_dir),
            "selected_cases": result.total,
            "n_cases_source": result.n_cases_source,
        },
    )
    pc.write_json(out_dir / "run_metadata.json", metadata)

    elapsed = time.time() - t0
    print(
        f"[g0-r] wrote {len(planned_files)} files -> {out_dir} "
        f"(cases={result.total}, blind_rows={len(blind_rows)}, elapsed={elapsed:.1f}s)"
    )
    print(
        "[g0-r] 下一步（研究者）：① 按协议 §4 生成叠加图与定量指标（写入 alignment_metrics.csv 并用 "
        "--validate-metrics 校验）；② 双阅片者填写 blind_review_sheet.csv；③ 按 §5 盲态 pilot 确定阈值；"
        "④ 按 §6 冻结 decision（工具不代填）。G0-R 状态仍为 PENDING。"
    )


if __name__ == "__main__":
    main()
