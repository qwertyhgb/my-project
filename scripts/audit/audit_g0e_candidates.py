#!/usr/bin/env python
"""G0-E 候选数据集审计：文件清单级清点 + 证据清单（只读元数据）。

本工具做三件事（**不读取影像体素、不做任何独立性/可用性推断**）：

1. 对 `configs/protocols/g0_e_independent_test.yaml` 中的候选数据集做**文件清单级清点**
   （目录名/文件名/CSV 表头/行数）——结果一律标注 `AUTO_ONLY`，仅供参考；
2. 汇总证据状态：YAML 中已填字段 → `VERIFIED`；可自动得到的事实 → `AUTO_ONLY`；
   其余 → `UNKNOWN`（**不得**据此推断数据集可用或独立）；
3. 生成审计报告模板与**研究者待补证据清单**。

真实数据目录的遍历由**研究者执行**；`verdict` / `frozen_decision` 保持 `null`，工具不代填。

运行示例（研究者）：

    cd /opt/data/private/lm/my-projects
    conda activate lm
    python scripts/audit/audit_g0e_candidates.py --dry-run      # 只看清点计划
    python scripts/audit/audit_g0e_candidates.py                # 生成登记表与证据清单
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from zonal_reliability_fusion.protocols import g0_e  # noqa: E402
from zonal_reliability_fusion.protocols import common as pc  # noqa: E402
from zonal_reliability_fusion.utils import make_progress, resolve_progress  # noqa: E402

DEFAULT_CONFIG = PROJECT_ROOT / "configs/protocols/g0_e_independent_test.yaml"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="G0-E 候选审计（文件清单级；不推断独立性/可用性）")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG), help="协议配置（默认 configs/protocols/g0_e_independent_test.yaml）")
    ap.add_argument("--candidate", default="", help="只审计指定候选名（默认全部）")
    ap.add_argument("--out-dir", default="", help="输出目录（默认 <config outputs.dir>/<timestamp>）")
    ap.add_argument("--dry-run", action="store_true", help="只打印清点计划，不写任何文件")
    ap.add_argument("--no-progress", action="store_true", help="关闭进度条（日志环境）")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()

    config_path = Path(args.config)
    doc = pc.load_yaml_config(config_path)
    protocol_meta = pc.validate_protocol_block(doc, expected_id="G0-E")
    progress = resolve_progress(args.no_progress, True)

    candidates = [c for c in (doc.get("candidates") or []) if isinstance(c, dict)]
    if not candidates:
        raise SystemExit("[g0-e] 配置中没有 candidates")
    if args.candidate:
        candidates = [c for c in candidates if str(c.get("name")) == args.candidate]
        if not candidates:
            raise SystemExit(f"[g0-e] 未找到候选: {args.candidate}")

    audits: list[dict] = []
    inventories: dict[str, dict] = {}
    bar = make_progress(candidates, total=len(candidates), desc="inventory candidates", disable=not progress)
    for candidate in bar:
        inventory = None
        local_root = candidate.get("local_root")
        if local_root:
            root = Path(str(local_root))
            if root.is_dir():
                inventory = g0_e.inventory_dataset(root)
            else:
                inventory = {
                    "root": str(root),
                    "exists": False,
                    "errors": [f"根目录不存在: {root}"],
                    "notes": "AUTO_ONLY",
                }
        audit = g0_e.audit_candidate(candidate, inventory)
        audits.append(audit)
        inventories[str(candidate.get("name"))] = inventory or {"exists": None, "notes": "未配置 local_root"}
        if hasattr(bar, "set_postfix"):
            bar.set_postfix({"missing": len(audit["missing"])})

    missing_rows = g0_e.build_missing_evidence_rows(audits)
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else pc.resolve_project_path(pc.get_path(doc, "outputs.dir", default="outputs/diagnostics/g0_e"))
        / pc.timestamp_slug()
    )
    planned_files = [
        "candidate_registry.csv",
        "evidence_status.json",
        "inventory_summary.json",
        "missing_evidence.md",
        "audit_report.md",
        "run_metadata.json",
    ]

    if args.dry_run:
        print(f"[g0-e][dry-run] 输出目录（不创建）: {out_dir}")
        print(f"[g0-e][dry-run] 计划写入: {', '.join(planned_files)}")
        for audit in audits:
            auto = audit.get("auto_only") or {}
            print(
                f"[g0-e][dry-run] {audit['name']}: root_exists={auto.get('local_root_exists')} "
                f"case_dirs={auto.get('n_case_dirs')} missing_evidence={len(audit['missing'])} "
                f"excluded={audit['excluded_as_lesion_test']}"
            )
        print("[g0-e][dry-run] verdict / frozen_decision 保持 null；工具不做独立性或可用性推断")
        return

    out_dir = pc.ensure_output_dir(out_dir)

    # 1) 候选登记表（配置字段 + 自动清点事实，自动列显式标注 AUTO_ONLY）
    registry_columns = ["name", "priority", "local_root", "excluded_as_lesion_test", *g0_e.EVIDENCE_FIELDS]
    registry_rows = []
    for candidate, audit in zip(candidates, audits):
        row = {name: candidate.get(name, "") for name in ("name", "priority", "local_root", "excluded_as_lesion_test")}
        for field in g0_e.EVIDENCE_FIELDS:
            value = candidate.get(field)
            row[field] = "" if value is None else value
        registry_rows.append(row)
    pc.write_rows_csv(out_dir / "candidate_registry.csv", registry_rows, fieldnames=registry_columns)

    # 2) 证据状态 + 自动清点
    evidence_status = {
        "protocol": protocol_meta,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "candidates": [
            {
                "name": audit["name"],
                "excluded_as_lesion_test": audit["excluded_as_lesion_test"],
                "fields": audit["fields"],
                "missing_fields": audit["missing"],
            }
            for audit in audits
        ],
        "frozen_decision": None,  # 工具不填
        "note": "UNKNOWN = 证据缺失；AUTO_ONLY 值不作为判定依据；结论由研究者按 decision_tree 填写",
    }
    pc.write_json(out_dir / "evidence_status.json", evidence_status)
    pc.write_json(
        out_dir / "inventory_summary.json",
        {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "auto_only": True,
            "note": "文件清单级清点（目录/文件名/CSV 表头/行数）；不读影像体素、不作为判定依据",
            "inventories": inventories,
        },
    )

    # 3) 待补证据清单
    md_lines = ["# G0-E 待补证据清单（自动生成）", ""]
    md_lines.append(f"- 协议版本：`{protocol_meta['version']}`（状态：**{protocol_meta['status']}**）")
    md_lines.append(f"- 生成时间：{time.strftime('%Y-%m-%dT%H:%M:%S')}")
    md_lines.append("- 缺证据字段一律 `UNKNOWN`；补齐后重新运行本工具，**不代表 G0-E 通过**。")
    md_lines.append("")
    md_lines.append("| 候选 | 缺失字段 | 需要什么证据 |")
    md_lines.append("|---|---|---|")
    for row in missing_rows:
        md_lines.append(f"| {row['candidate']} | `{row['missing_field']}` | {row['required_evidence']} |")
    md_lines.append("")
    (out_dir / "missing_evidence.md").write_text("\n".join(md_lines), encoding="utf-8")

    # 4) 审计报告模板
    report = g0_e.render_g0e_report_markdown(
        audits,
        protocol_version=protocol_meta["version"],
        protocol_status=protocol_meta["status"],
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    (out_dir / "audit_report.md").write_text(report, encoding="utf-8")

    # 5) 运行元数据
    pc.write_json(
        out_dir / "run_metadata.json",
        pc.build_run_metadata(
            tool="scripts/audit/audit_g0e_candidates.py",
            config_path=config_path,
            extra={
                "candidates": [audit["name"] for audit in audits],
                "missing_evidence_items": len(missing_rows),
                "output_dir": str(out_dir),
            },
        ),
    )

    elapsed = time.time() - t0
    print(
        f"[g0-e] wrote {len(planned_files)} files -> {out_dir} "
        f"(candidates={len(audits)}, missing_items={len(missing_rows)}, elapsed={elapsed:.1f}s)"
    )
    print(
        "[g0-e] 下一步（研究者）：补齐 missing_evidence.md 中的证据（许可/语义/重叠/先验独立性/样本量），"
        "再按配置 decision_tree 冻结路径 A/B/C。工具不代填 verdict，G0-E 仍为 PENDING。"
    )


if __name__ == "__main__":
    main()
