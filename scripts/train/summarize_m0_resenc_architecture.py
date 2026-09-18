#!/usr/bin/env python
"""v2.3 Residual-Encoder M0 结构摘要 / 预算分析（只读、不读病例、不使用 GPU、不做真实 patch forward）。

做什么：
- 读取 v2.3 实验配置 + 冻结 plan + 版本化架构配置，校验三者一致；
- 在 CPU 上构建 `M0ResidualConcatModel`（仅初始化权重，**不 forward 真实 patch**）；
- 输出结构摘要：总/可训练参数量、每 stage block 数、每 stage 通道与空间 shape、deep-supervision 输出 shape、
  架构哈希、解析 MACs（含 encoder/decoder/DS heads）、激活规模与分析性显存估计（**显存标记为未实测**）；
- 可选 `--reduced-forward-check`：在**缩小的合成 plan** 上做一次 forward，用 hook 计数验证解析 MACs 公式
  （不触碰真实 `[16,320,320]` patch，不读任何病例）；
- 写 JSON 到 `outputs/diagnostics/m0_resenc/<name>/architecture_summary.json`（已存在则加时间戳，不覆盖）。

不做什么：不读取真实病例体素、不训练、不推理、不评测、不初始化 CUDA、不占用 GPU。

用法：
    cd /opt/data/private/lm/my-projects
    conda activate lm
    CUDA_VISIBLE_DEVICES="" python scripts/train/summarize_m0_resenc_architecture.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from zonal_reliability_fusion.config import (
    architecture_spec_from_mapping,
    load_experiment_config,
    load_plan,
)
from zonal_reliability_fusion.models import M0ResidualConcatModel
from zonal_reliability_fusion.models.factory import (
    resolve_architecture_spec,
)
from zonal_reliability_fusion.models.profiling import (
    analytical_macs,
    count_macs_with_hooks,
    structure_summary,
)
from zonal_reliability_fusion.utils import make_progress, resolve_progress

DEFAULT_CONFIG = PROJECT_ROOT / "configs/experiments/m0_resenc_picai_3d_fullres_v23.yaml"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="v2.3 Residual-Encoder M0 结构摘要 / 参数 / MACs / 显存状态（只读，不使用 GPU）"
    )
    ap.add_argument("--config", default=str(DEFAULT_CONFIG), help="v2.3 实验配置（含 architecture 引用）")
    ap.add_argument("--plans", default="", help="覆盖配置中的 plan 路径")
    ap.add_argument("--json-out", default="", help="摘要输出路径（默认 outputs/diagnostics/m0_resenc/<name>/）")
    ap.add_argument("--no-require-data", action="store_true", help="本工具默认不读取预处理数据；保留此开关以对齐其它入口")
    ap.add_argument(
        "--reduced-forward-check",
        action="store_true",
        help="在缩小的合成 plan 上做一次 forward，用 hook 计数验证解析 MACs 公式（不触碰真实 patch）",
    )
    ap.add_argument("--no-progress", action="store_true", help="关闭 tqdm 进度条（日志环境）")
    return ap.parse_args()


def _reduced_forward_check(real_spec) -> dict:
    """在**缩小的合成 plan** 上验证解析 MACs == hook MACs（廉价 forward；不使用真实 patch/数据/GPU）。"""
    import tempfile

    import torch

    mini_plan_doc = {
        "dataset_name": "Dataset605_PICAI",
        "plans_name": "nnUNetPlans",
        "configurations": {
            "3d_fullres": {
                "data_identifier": "nnUNetPlans_3d_fullres",
                "batch_size": 1,
                "patch_size": [4, 8, 8],
                "spacing": [3.0, 0.5, 0.5],
                "batch_dice": False,
                "architecture": {
                    "network_class_name": "x.PlainConvUNet",
                    "arch_kwargs": {
                        "n_stages": 3,
                        "features_per_stage": [2, 4, 4],
                        "conv_op": "torch.nn.modules.conv.Conv3d",
                        "kernel_sizes": [[1, 3, 3], [3, 3, 3], [3, 3, 3]],
                        "strides": [[1, 1, 1], [1, 2, 2], [2, 2, 2]],
                        "n_conv_per_stage": [2, 2, 2],
                        "n_conv_per_stage_decoder": [2, 2],
                        "conv_bias": True,
                        "norm_op": "torch.nn.modules.instancenorm.InstanceNorm3d",
                        "norm_op_kwargs": {"eps": 1e-5, "affine": True},
                        "dropout_op": None,
                        "nonlin": "torch.nn.LeakyReLU",
                        "nonlin_kwargs": {"inplace": True, "negative_slope": 0.01},
                    },
                },
            }
        },
    }
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "plans.json"
        p.write_text(json.dumps(mini_plan_doc))
        mini_plan = load_plan(p, "3d_fullres")
    # 用真实 spec 的规则字段，替换结构字段以匹配缩小 plan
    doc = real_spec.to_dict()
    doc.pop("architecture_sha256", None)
    doc.pop("source_path", None)
    doc["n_stages"] = mini_plan.n_stages
    doc["blocks_per_stage"] = [1, 2, 2]
    doc["features_per_stage"] = list(mini_plan.features_per_stage)
    doc["kernel_sizes"] = [list(k) for k in mini_plan.kernel_sizes]
    doc["strides"] = [list(s) for s in mini_plan.strides]
    doc["decoder"]["n_conv_per_stage_decoder"] = list(mini_plan.n_conv_per_stage_decoder)
    doc["fusion_stage"] = 2
    mini_spec = architecture_spec_from_mapping(doc, source_path="reduced")
    model = M0ResidualConcatModel(mini_plan, mini_spec, in_channels=3, num_classes=2, deep_supervision=True)
    hook = count_macs_with_hooks(model, torch.randn(1, 3, *mini_plan.patch_size))
    ana = analytical_macs(mini_plan, mini_spec, 3, 2)
    return {
        "reduced_patch_size": list(mini_plan.patch_size),
        "hook_macs": int(hook["total"]),
        "analytical_macs": int(ana["total"]),
        "match": bool(hook["total"] == ana["total"]),
    }


def main() -> None:
    args = parse_args()
    t0 = time.time()
    progress = resolve_progress(args.no_progress, True)

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    cfg = load_experiment_config(cfg_path)
    plans_path = Path(args.plans) if args.plans else cfg.resolve_path(cfg.paths.plans)
    plan = load_plan(plans_path, configuration=cfg.model.configuration, dataset_name="Dataset605_PICAI")
    spec = resolve_architecture_spec(cfg, plan)  # 校验 ref 身份 + arch↔plan 一致

    steps = ["build_model", "parameters", "macs_analytical", "activations", "summary"]
    if args.reduced_forward_check:
        steps.append("reduced_forward_check")
    succeeded, failed, skipped = 0, 0, 0

    model = None
    summary: dict = {}
    reduced: dict | None = None
    bar = make_progress(steps, total=len(steps), desc="arch summary", disable=not progress)
    for step in bar:
        try:
            if step == "build_model":
                model = M0ResidualConcatModel(
                    plan,
                    spec,
                    in_channels=cfg.model.in_channels,
                    num_classes=cfg.model.num_classes,
                    deep_supervision=cfg.model.deep_supervision,
                )
            elif step == "summary":
                summary = structure_summary(
                    model,
                    plan,
                    spec,
                    in_channels=cfg.model.in_channels,
                    num_classes=cfg.model.num_classes,
                    batch_size=plan.batch_size,
                )
            elif step == "reduced_forward_check":
                reduced = _reduced_forward_check(spec)
            # parameters / macs_analytical / activations 都包含在 structure_summary 内（此处仅计数进度）
            succeeded += 1
        except Exception as exc:  # noqa: BLE001 - 工具需汇总失败而非中断
            failed += 1
            print(f"[arch summary][ERROR] step={step} 失败: {exc}")
    if not args.reduced_forward_check:
        skipped += 1  # reduced_forward_check 未请求

    print("=" * 78)
    print(model.format_summary())
    print("-" * 78)
    params = summary["parameters"]
    macs = summary["macs"]
    acts = summary["activation_feature_map_elements"]
    mem = summary["peak_gpu_memory"]
    print(f"参数量              : total={params['total']:,}  trainable={params['trainable']:,}")
    print(f"MACs（解析，无 forward）: encoder={macs['encoder']:,}  decoder={macs['decoder']:,}  total={macs['total']:,}")
    print(f"  单位/口径          : {macs['unit']}")
    print(f"  方法              : {macs['method']}")
    print(f"  输入 shape         : {macs['input_shape']}（含 decoder 与 DS heads；不含 avgpool/norm/激活/加法）")
    print(f"激活元素（解析，逐样本）: total={acts['total']:,}")
    print(f"峰值显存            : status={mem['status']}（{mem['reason']}）")
    print(f"  分析性激活字节 fp32 : {mem['analytical_activation_bytes_fp32']:,} B（下界代理，非峰值显存）")
    if reduced is not None:
        print(f"reduced-forward 校验 : hook={reduced['hook_macs']:,} analytical={reduced['analytical_macs']:,} "
              f"match={reduced['match']}（缩小 patch {reduced['reduced_patch_size']}）")
    print("=" * 78)

    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config_path": str(cfg_path),
        "plans_path": str(plans_path),
        "architecture_config_path": str(spec.source_path),
        "architecture_identity": spec.identity(),
        "structure_summary": summary,
        "reduced_forward_check": reduced,
        "counts": {"succeeded": succeeded, "failed": failed, "skipped": skipped},
        "elapsed_sec": round(time.time() - t0, 2),
        "note": "只读结构分析：未读取真实病例、未使用 GPU、未做真实 patch forward、未训练/推理/评测",
    }
    out_path = Path(args.json_out) if args.json_out else (
        PROJECT_ROOT / "outputs/diagnostics/m0_resenc" / cfg.experiment.name / "architecture_summary.json"
    )
    if out_path.exists() and not args.json_out:
        out_path = out_path.with_name(f"architecture_summary_{time.strftime('%Y%m%d_%H%M%S')}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"[arch summary] 成功={succeeded} 失败={failed} 跳过={skipped} 耗时={payload['elapsed_sec']}s")
    print(f"[arch summary] 输出: {out_path}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
