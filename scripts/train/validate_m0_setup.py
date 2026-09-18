#!/usr/bin/env python
"""M0–M4 setup 检查（命令 A：只读、不读 MRI/lesion 内容、不做 forward、不使用 GPU）。

做什麼：
- 读取 M0 配置（YAML/JSON）与冻结 plan（nnUNetPlans.json，只读 JSON 字段）；
- 校验配置与 plan / 项目约定的冲突（patch、batch、模态顺序、fold=0、输出目录位置）；
- 读取 splits_final.json 结构摘要（fold 数、train/val 数量）；
- 构建完整 M0（在 CPU 上），打印参数量、stage shape、resolved 配置；
- 写出 setup_check.json 到 outputs/diagnostics/<model_id>/<name>/。

读什么（诚实边界）：
- 不读取 MRI/lesion 的 .b2nd/.pkl **内容**；
- 但 require-data 模式（默认）会**顺序读取 derived PZ/TZ `.npz` sidecar 并逐个校验 SHA256**
  （由 `PreprocessedStore.load_zonal_prior_manifest` 完成，`array_hash_checked=true`）；
  `--no-require-data` 跳过该读取，prior 状态记为 PRIOR_NOT_CHECKED。

不做什么：
- 不做 forward、不训练、不初始化 CUDA。

用法：
    cd /opt/data/private/lm/my-projects
    source scripts/env_nnunet.sh
    CUDA_VISIBLE_DEVICES="" python scripts/train/validate_m0_setup.py
    # 关闭 prior sidecar 校验进度条：
    CUDA_VISIBLE_DEVICES="" python scripts/train/validate_m0_setup.py --no-progress
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
    M0ExperimentConfig,
    load_experiment_config,
    load_plan,
    read_split_summary,
    validate_against_plan,
    validation_protocol_from_config,
)
from zonal_reliability_fusion.data.preprocessed_store import PreprocessedStore
from zonal_reliability_fusion.data.zonal_prior import file_sha256
from zonal_reliability_fusion.models.factory import (
    build_model,
    model_architecture_identity,
)
from zonal_reliability_fusion.training import deep_supervision_weights

DEFAULT_CONFIG = PROJECT_ROOT / "configs/experiments/m0_picai_3d_fullres.yaml"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="M0 setup 检查（只读，不使用 GPU）")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--no-progress", action="store_true", help="关闭 prior sidecar 校验进度条")
    ap.add_argument("--plans", default="", help="覆盖配置中的 plan 路径")
    ap.add_argument("--splits", default="", help="覆盖配置中的 split 路径")
    ap.add_argument("--no-require-data", action="store_true", help="允许预处理目录尚不存在（仅做配置校验）")
    ap.add_argument(
        "--json-out",
        default="",
        help="setup 检查结果输出路径（默认 outputs/diagnostics/<model_id 小写>/<name>/setup_check.json）",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    t0 = time.time()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    cfg: M0ExperimentConfig = load_experiment_config(cfg_path)

    plans_path = Path(args.plans) if args.plans else cfg.resolve_path(cfg.paths.plans)
    plan = load_plan(plans_path, configuration=cfg.model.configuration, dataset_name="Dataset605_PICAI")

    if args.splits:
        split_path = Path(args.splits)
    else:
        split_path = cfg.resolve_path(cfg.paths.splits)
    split_summary = read_split_summary(split_path)

    warnings = validate_against_plan(
        cfg, plan, project_root=PROJECT_ROOT, require_data=not args.no_require_data
    )

    model = build_model(cfg, plan)
    arch_identity = model_architecture_identity(model)

    ds_weights = deep_supervision_weights(len(plan.decoder_output_shapes()))

    print("=" * 78)
    print("[M0 setup] plan")
    print(f"  path            : {plan.source_path}")
    print(f"  dataset / config: {plan.dataset_name} / {plan.configuration}")
    print(f"  spacing / patch : {plan.spacing} / {plan.patch_size}  batch={plan.batch_size}")
    print(f"  stage_shapes    : {list(plan.stage_shapes())}")
    print(f"  DS 输出 shapes  : {list(plan.decoder_output_shapes())}（高→低）")
    print(f"  DS 权重         : {[round(w, 6) for w in ds_weights]}")
    print("[M0 setup] split")
    print(f"  path            : {split_summary['path']}")
    print(f"  folds={split_summary['n_folds']}  train={split_summary['n_train']}  val={split_summary['n_val']}")
    print("[M0 setup] 训练/验证协议")
    print(
        f"  max_epochs={cfg.training.max_epochs}  iters/epoch={cfg.training.iterations_per_epoch}  "
        f"diagnostic_val_iters={cfg.training.diagnostic_validation_iterations}"
    )
    print(
        f"  validation.mode={cfg.validation.mode}  metric={cfg.validation.checkpoint_metric} "
        f"(maximize={cfg.validation.maximize})  expected={cfg.validation.expected_cases} "
        f"(pos={cfg.validation.expected_positive_cases} / neg={cfg.validation.expected_negative_cases})"
    )
    print(
        f"  sliding-window: step_fraction={cfg.validation.step_fraction} gaussian={cfg.validation.gaussian} "
        f"mirror_tta={cfg.validation.mirror_tta} batch_size={cfg.validation.sliding_window_batch_size} "
        f"inner_patch_progress={cfg.validation.inner_patch_progress}"
    )
    print(
        f"  early_stopping: enabled={cfg.early_stopping.enabled} min_epochs={cfg.early_stopping.min_epochs} "
        f"patience={cfg.early_stopping.patience} min_delta={cfg.early_stopping.min_delta}"
    )
    print("[M0 setup] model")
    print(model.format_summary())
    if arch_identity is not None:
        print("[M0 setup] architecture identity (v2.3 Residual-Encoder)")
        print(f"  name/version    : {arch_identity['architecture_name']} / {arch_identity['architecture_version']}")
        print(f"  encoder_type    : {arch_identity['encoder_type']}")
        print(f"  architecture_sha256: {arch_identity['architecture_sha256']}")
    else:
        print("[M0 setup] architecture identity: legacy PlainConv M0（无版本化架构身份）")
    if warnings:
        print("[M0 setup] warnings")
        for item in warnings:
            print(f"  - {item}")
    print("=" * 78)

    # ---------------------------------------------------------------- prior readiness（M3/M4）
    needs_prior = str(cfg.experiment.model_id) in ("M3", "M4")
    prior_readiness: dict = {"required": needs_prior, "status": "N/A", "checks": {}}
    if needs_prior:
        if args.no_require_data:
            prior_readiness = {
                "required": True,
                "status": "PRIOR_NOT_CHECKED",
                "reason": "--no-require-data：未检查 prior root / manifest / sidecar",
                "training_ready": False,
                "checks": {},
            }
        else:
            checks: dict = {}
            try:
                prior_root = cfg.resolve_path(cfg.data.zonal_prior_root)
                checks["prior_root"] = str(prior_root)
                checks["prior_root_exists"] = bool(Path(prior_root).is_dir())
                if not checks["prior_root_exists"]:
                    raise FileNotFoundError(f"prior root 不存在: {prior_root}")
                checks["source"] = cfg.data.zonal_prior_source
                checks["configuration"] = cfg.model.configuration
                plans_sha = file_sha256(cfg.resolve_path(cfg.paths.plans))
                checks["plans_sha256"] = plans_sha
                prior_store = PreprocessedStore(
                    cfg.resolve_path(cfg.paths.preprocessed_dataset),
                    cfg.resolve_path(cfg.paths.splits),
                    fold=cfg.experiment.fold,
                    zonal_prior_root=prior_root,
                    zonal_prior_source=cfg.data.zonal_prior_source,
                    expected_plans_sha256=plans_sha,
                    expected_configuration=cfg.model.configuration,
                )
                manifest = prior_store.load_zonal_prior_manifest(
                    list(prior_store.train_ids) + list(prior_store.val_ids),
                    progress=not args.no_progress,
                    desc=f"prior sidecar 校验 ({cfg.experiment.model_id})",
                )
                stats = dict(getattr(prior_store, "zonal_prior_validation_stats", {}) or {})
                print(
                    "[M0 setup] prior 校验："
                    f"cases={stats.get('n_cases')} ok={stats.get('n_ok')} failed={stats.get('n_failed')} "
                    f"array_hash_checked={stats.get('array_hash_checked')} "
                    f"耗时={stats.get('elapsed_sec')}s manifest_sha256={str(stats.get('manifest_sha256'))[:12]}…"
                    f" frozen_records={len(getattr(prior_store, 'zonal_prior_frozen_records', {}) or {})}"
                )
                checks["manifest_sha256"] = manifest["manifest_sha256"]
                checks["manifest_n_cases"] = manifest["n_cases"]
                checks["coverage_equals_train_union_val"] = True
                checks["array_hash_checked"] = bool(
                    getattr(prior_store, "zonal_prior_array_hash_checked", False)
                )
                checks["validation_stats"] = dict(
                    getattr(prior_store, "zonal_prior_validation_stats", {}) or {}
                )
                prior_readiness = {
                    "required": True,
                    "status": "PRIOR_READY",
                    "training_ready": True,
                    "checks": checks,
                    "note": (
                        "已逐个校验 sidecar 元数据与数组 SHA256"
                        if checks.get("array_hash_checked")
                        else "未校验数组内容（array_hash_checked=false），不得声称全部 sidecar 已验证"
                    ),
                }
            except Exception as exc:  # noqa: BLE001 - 逐项检查失败必须转成 PRIOR_NOT_READY 而不是崩溃
                prior_readiness = {
                    "required": True,
                    "status": "PRIOR_NOT_READY",
                    "training_ready": False,
                    "reason": f"{type(exc).__name__}: {exc}",
                    "checks": checks,
                }
    print("[M0 setup] prior readiness")
    print(f"  status          : {prior_readiness['status']}（required={needs_prior}）")
    for key, value in prior_readiness.get("checks", {}).items():
        print(f"  {key:24s}: {value}")
    if prior_readiness.get("reason"):
        print(f"  reason          : {prior_readiness['reason']}")
    if needs_prior and not prior_readiness.get("training_ready", False):
        print("  NOT_TRAINING_READY: 该模型读取 PZ/TZ prior；prior 未验证通过前不得开始训练")

    payload = {
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config_path": str(cfg_path),
        "resolved_config": cfg.resolved_snapshot(),
        "validation_protocol": validation_protocol_from_config(cfg).to_dict(),
        "plan_snapshot": plan.resolved_snapshot(),
        "split_summary": split_summary,
        "warnings": warnings,
        "model_summary": model.summary(),
        "architecture_identity": arch_identity,
        "prior_readiness": prior_readiness,
        "deep_supervision_weights": list(ds_weights),
        "elapsed_sec": round(time.time() - t0, 2),
        "note": (
            "setup 检查不读取 MRI/lesion .b2nd、不做 forward、不初始化 CUDA；"
            "require-data 模式会顺序读取 derived PZ/TZ NPZ 并校验 SHA256"
            "（array_hash_checked 见 prior_readiness.checks），该顺序读取由本检查主动发起"
        ),
    }
    out_path = Path(args.json_out) if args.json_out else (
        PROJECT_ROOT
        / "outputs/diagnostics"
        / str(cfg.experiment.model_id).lower()
        / cfg.experiment.name
        / "setup_check.json"
    )
    if out_path.exists() and not args.json_out:
        # 默认不覆盖已有检查结果：追加时间戳
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_path = out_path.with_name(f"setup_check_{stamp}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"[M0 setup] 完成（{payload['elapsed_sec']}s）；结果写入: {out_path}")
    print(
        f"[M0 setup] 参数总量: {model.num_parameters():,}；"
        "未读取 MRI/lesion .b2nd、未做 forward、未使用 GPU"
        + (
            "；已顺序读取 derived PZ/TZ NPZ 做哈希校验"
            if needs_prior and prior_readiness.get("training_ready")
            else ""
        )
    )
    if needs_prior:
        print(
            f"[M0 setup] prior readiness: {prior_readiness['status']}"
            f"（training_ready={prior_readiness.get('training_ready', False)}）"
        )
        if prior_readiness.get("note"):
            print(f"[M0 setup] note: {prior_readiness['note']}")
    # 默认（require-data）模式下 prior 不就绪 → 诊断 JSON 已写出，但退出码必须非 0
    if needs_prior and not args.no_require_data and not prior_readiness.get("training_ready", False):
        print("[M0 setup] 退出码 1：prior 未就绪且处于 require-data 模式（NOT_TRAINING_READY）")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
