#!/usr/bin/env python
"""G0-SAP 冻结载体准备：配置静态校验 + 冻结就绪检查 + 模板生成（不运行评测）。

本工具做五件事（**不读取预测/影像数据、不运行评测、不生成论文指标**）：

1. 静态校验 `configs/protocols/g0_sap.yaml` 的硬约束（主终点、层级比较、bootstrap 定位、
   `picai_eval` 口径、FROC 操作点、功效备忘录字段完整性含 MCID）；
2. 校验 SAP-A/B/C 阶段状态与预测可见性组合（`sap_b_results` 只能按阶段填写；
   `protocol.status=FROZEN` 仅表示 SAP-A 冻结）；
3. `--check-freeze`：报告 **SAP-A 冻结阻塞项**（仍为空的阈值/后处理/备忘录/MCID/审阅/方法块哈希字段），
   `--fail-if-not-ready` 时以退出码 3 表示"尚未就绪"（不是错误）；
4. 生成冻结包模板：`evaluation_config.sha256`、`sap_a_method_hash.txt`、`power_memo_template.md`、
   `freeze_checklist.md`、`stats_config.template.json`、`run_metadata.json`；
5. 输出路径隔离：默认写入 `outputs/metrics/evaluation/<version>/`，禁止落入训练产物目录。

**不得伪造数值**：模板中未定字段一律 `TODO` / `null`；`status=FROZEN` 但阻塞项非空时工具**拒绝**生成冻结载体。
`--sap-a-baseline` 可选：传入 SAP-A 冻结时的配置快照时执行**结构化差异审计**，SAP-A 方法字段的任何改动都会被拒绝。

运行示例（研究者）：

    cd /opt/data/private/lm/my-projects
    conda activate lm
    python scripts/evaluate/prepare_g0_sap_freeze.py --check-freeze       # 报告还缺什么（含 SAP-A 方法块哈希）
    python scripts/evaluate/prepare_g0_sap_freeze.py --dry-run            # 只看将写哪些文件
    python scripts/evaluate/prepare_g0_sap_freeze.py                      # 生成冻结包模板
    python scripts/evaluate/prepare_g0_sap_freeze.py \
        --sap-a-baseline outputs/metrics/evaluation/draft-0.3/sap_a_snapshot.json   # SAP-B 差异审计
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from zonal_reliability_fusion.protocols import g0_sap  # noqa: E402
from zonal_reliability_fusion.protocols import common as pc  # noqa: E402
from zonal_reliability_fusion.utils import resolve_progress  # noqa: E402

DEFAULT_CONFIG = PROJECT_ROOT / "configs/protocols/g0_sap.yaml"

#: 与 G0-SAP 冻结绑定的登记文件（存在才计算哈希，缺失记 MISSING；不伪造）
REGISTERED_FILES = (
    "configs/protocols/g0_sap.yaml",
    "docs/protocols/G0_SAP.md",
    "docs/research_plan.md",
    "data/splits/picai_train_val_split.json",
    "workdir/nnUNet_preprocessed/Dataset605_PICAI/splits_final.json",
    "workdir/nnUNet_preprocessed/Dataset605_PICAI/nnUNetPlans.json",
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="G0-SAP 冻结载体准备（静态校验 + 模板；不运行评测）")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG), help="评测配置（默认 configs/protocols/g0_sap.yaml）")
    ap.add_argument("--out-dir", default="", help="输出目录（默认 <config outputs.dir>/<protocol.version>）")
    ap.add_argument("--check-freeze", action="store_true", help="只做冻结就绪检查并打印阻塞项，不写文件")
    ap.add_argument("--fail-if-not-ready", action="store_true", help="配合 --check-freeze：未就绪时退出码 3（不是错误）")
    ap.add_argument("--dry-run", action="store_true", help="只打印将写入的文件，不写任何文件")
    ap.add_argument(
        "--sap-a-baseline",
        default="",
        help="可选：SAP-A 冻结时的配置快照；提供时执行结构化差异审计（SAP-A 方法字段改动将被拒绝）",
    )
    ap.add_argument("--no-progress", action="store_true", help="关闭进度条（本工具无长循环，保留参数以统一 CLI）")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()
    resolve_progress(args.no_progress, True)  # 本工具没有主要耗时循环；保持 CLI 一致性

    config_path = Path(args.config)
    doc = pc.load_yaml_config(config_path)
    pc.validate_protocol_block(doc, expected_id="G0-SAP")
    baseline = pc.load_yaml_config(Path(args.sap_a_baseline)) if args.sap_a_baseline else None
    g0_sap.assert_g0_sap_config(doc, sap_a_baseline=baseline)

    status = str(pc.get_path(doc, "protocol.status", default="DRAFT"))
    version = str(pc.get_path(doc, "protocol.version", default="unknown"))
    method_hash = g0_sap.sap_a_method_hash(doc)
    phases = {
        name: pc.get_path(doc, f"sap_phases.{name}.status", default=None)
        for name in ("sap_a", "sap_b", "sap_c")
    }
    blockers = g0_sap.validate_freeze_readiness(doc)

    print(f"[g0-sap] config={config_path} status={status} version={version}")
    print(
        f"[g0-sap] sap_phases: sap_a={phases['sap_a']} sap_b={phases['sap_b']} sap_c={phases['sap_c']}"
        "（protocol.status=FROZEN 仅表示 SAP-A 冻结）"
    )
    print(
        f"[g0-sap] SAP-A 方法块哈希: {method_hash}"
        f"（算法 {g0_sap.SAP_A_METHOD_HASH_ALGORITHM}；冻结时写入 sap_phases.sap_a.method_sha256）"
    )
    if args.sap_a_baseline:
        diffs = g0_sap.diff_sap_a_methods(baseline, doc)
        print(f"[g0-sap] SAP-A 基线差异审计：{'未改动方法字段' if not diffs else f'发现 {len(diffs)} 处改动'}")
    print(f"[g0-sap] freeze_readiness: {'READY' if not blockers else 'NOT READY'}（阻塞项 {len(blockers)}）")
    for blocker in blockers:
        print(f"[g0-sap]   - {blocker}")

    if args.check_freeze:
        if blockers and args.fail_if_not_ready:
            print("[g0-sap][check-freeze] 尚未就绪（exit 3）；补齐后可重跑。G0-SAP 仍为 PENDING。")
            sys.exit(3)
        print("[g0-sap][check-freeze] 检查完成；本工具不修改配置、不填字段、不判 PASS。")
        return

    if status == "FROZEN" and blockers:
        raise SystemExit(
            "[g0-sap] 拒绝生成冻结载体：protocol.status=FROZEN 但冻结就绪检查未通过。"
            "请回退为 DRAFT 或补齐阻塞项（不得把未完成配置标记为冻结）。"
        )

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else pc.resolve_project_path(pc.get_path(doc, "outputs.dir", default="outputs/metrics/evaluation")) / version
    )
    planned_files = [
        "evaluation_config.sha256",
        "sap_a_method_hash.txt",
        "sap_a_snapshot.json",
        "power_memo_template.md",
        "freeze_checklist.md",
        "stats_config.template.json",
        "run_metadata.json",
    ]
    generated_at = time.strftime("%Y-%m-%dT%H:%M:%S")

    if args.dry_run:
        print(f"[g0-sap][dry-run] 输出目录（不创建）: {out_dir}")
        print(f"[g0-sap][dry-run] 计划写入: {', '.join(planned_files)}")
        print("[g0-sap][dry-run] 模板中的数值字段一律 TODO/null；不伪造阈值或 MDE")
        return

    out_dir = pc.ensure_output_dir(out_dir)

    # 1) 配置与登记文件的内容哈希（sha256sum 格式；缺失文件记 MISSING，不伪造）
    hashes = pc.collect_hashes(REGISTERED_FILES, base=PROJECT_ROOT)
    sha_lines = [f"{digest}  {path}" for path, digest in sorted(hashes.items())]
    (out_dir / "evaluation_config.sha256").write_text("\n".join(sha_lines) + "\n", encoding="utf-8")

    # 2) SAP-A 方法块哈希 + 方法快照（SAP-B 差异审计基线）
    (out_dir / "sap_a_method_hash.txt").write_text(
        "\n".join(
            [
                f"algorithm: {g0_sap.SAP_A_METHOD_HASH_ALGORITHM}",
                f"sha256: {method_hash}",
                "填入位置: configs/protocols/g0_sap.yaml -> sap_phases.sap_a.method_sha256",
                "用途: SAP-A 冻结后任何方法字段改动都会使该哈希不一致并被校验器拒绝；",
                "      SAP-B 的 sap_b_results.source_sap_a_config_sha256 必须等于该值。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    pc.write_json(out_dir / "sap_a_snapshot.json", g0_sap.sap_a_method_snapshot(doc))

    # 3) 功效备忘录模板（字段留 TODO；不编造 MDE/MCID）
    (out_dir / "power_memo_template.md").write_text(
        g0_sap.render_power_memo_template(doc, generated_at=generated_at), encoding="utf-8"
    )

    # 4) 冻结清单（列出阻塞项与 SAP-A/B/C 步骤）
    (out_dir / "freeze_checklist.md").write_text(
        g0_sap.render_freeze_checklist(doc, blockers, generated_at=generated_at), encoding="utf-8"
    )

    # 5) stats_config 模板
    pc.write_json(out_dir / "stats_config.template.json", g0_sap.stats_config_template(doc))

    # 6) 运行元数据
    pc.write_json(
        out_dir / "run_metadata.json",
        pc.build_run_metadata(
            tool="scripts/evaluate/prepare_g0_sap_freeze.py",
            config_path=config_path,
            extra={
                "protocol_status": status,
                "sap_phases": phases,
                "sap_a_method_sha256": method_hash,
                "sap_a_method_hash_algorithm": g0_sap.SAP_A_METHOD_HASH_ALGORITHM,
                "freeze_ready": not blockers,
                "freeze_blockers": len(blockers),
                "registered_file_hashes": hashes,
                "output_dir": str(out_dir),
            },
        ),
    )

    elapsed = time.time() - t0
    print(f"[g0-sap] wrote {len(planned_files)} files -> {out_dir} (elapsed={elapsed:.1f}s)")
    print(
        "[g0-sap] 下一步（研究者）："
        "**SAP-A**：① 填写方法块（阈值候选网格/选择指标/tie-break/后处理候选/picai_eval 参数/统计与 bootstrap/"
        "FROC/功效备忘录含 MCID 与来源）；② 用上面的方法块哈希填写 sap_phases.sap_a.method_sha256；"
        "③ 手动置 protocol.status=FROZEN 与 sap_phases.sap_a.status=FROZEN 后重跑本工具。"
        "**SAP-B**（冻结 checkpoint 后）：只填写 sap_b_results（含 source_sap_a_config_sha256 与审计哈希），"
        "并以 --sap-a-baseline 传入本目录的 sap_a_snapshot.json 做差异审计。"
        "**SAP-C**：封存后对最终 test 一次性评估。G0-SAP 仍为 PENDING。"
    )


if __name__ == "__main__":
    main()
