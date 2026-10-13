#!/usr/bin/env python3
"""Versioned training entry. V2 active; V1 frozen; legacy historical.

Only native nnU-Net run_training is invoked. All long tasks are user-run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

#: FROZEN V1 historical aliases; displayed only with --v1.
#: 前 9 项是 1000 epoch 正式臂；``*_100ep`` 是同一 A/B/C/D 的 100 epoch 方向筛选臂，
#: 独立类名 → 独立输出目录，只做 sanity 与方向筛选，不与历史 1000 epoch 结果比较优劣。
FROZEN_V1_VARIANTS = {
    "native_dicece": "nnUNetTrainerPICAI_DiceCE_NativeSampling_NoFFT",
    "anatomy_joint_100ep": "nnUNetTrainerPICAI_AnatomyJoint_100ep_NoFFT",
    "positive_sampling": "nnUNetTrainerPICAI_FLCE_PositiveSampling_NoFFT",
    "dicece_positive_sampling": "nnUNetTrainerPICAI_DiceCE_PositiveSampling_NoFFT",
    "positive_sampling_100ep": "nnUNetTrainerPICAI_FLCE_PositiveSampling_100ep_NoFFT",
    "dicece_positive_sampling_100ep": (
        "nnUNetTrainerPICAI_DiceCE_PositiveSampling_100ep_NoFFT"
    ),
    "neutral_fusion_flce": "nnUNetTrainerPICAI_NeutralFusion_FLCE_NoFFT",
    "neutral_fusion_dicece": "nnUNetTrainerPICAI_NeutralFusion_DiceCE_NoFFT",
    "neutral_fusion_flce_100ep": "nnUNetTrainerPICAI_NeutralFusion_FLCE_100ep_NoFFT",
    "neutral_fusion_dicece_100ep": "nnUNetTrainerPICAI_NeutralFusion_DiceCE_100ep_NoFFT",
    "lesion_fusion_flce": "nnUNetTrainerPICAI_LesionFusion_FLCE_NoFFT",
    "lesion_fusion_dicece": "nnUNetTrainerPICAI_LesionFusion_DiceCE_NoFFT",
    "lesion_fusion_flce_100ep": "nnUNetTrainerPICAI_LesionFusion_FLCE_100ep_NoFFT",
    "lesion_fusion_dicece_100ep": "nnUNetTrainerPICAI_LesionFusion_DiceCE_100ep_NoFFT",
    "anatomy_lesion_fusion_flce": "nnUNetTrainerPICAI_AnatomyLesionFusion_FLCE_NoFFT",
    "anatomy_lesion_fusion_dicece": "nnUNetTrainerPICAI_AnatomyLesionFusion_DiceCE_NoFFT",
    "anatomy_lesion_fusion_flce_100ep": (
        "nnUNetTrainerPICAI_AnatomyLesionFusion_FLCE_100ep_NoFFT"
    ),
    "anatomy_lesion_fusion_dicece_100ep": (
        "nnUNetTrainerPICAI_AnatomyLesionFusion_DiceCE_100ep_NoFFT"
    ),
}

#: LEGACY（归档）：需要 ``--legacy`` 才出现。类名与实现逐字保留在 legacy/fusion_trainers.py
LEGACY_VARIANTS = {
    "baseline": "nnUNetTrainerPICAI_FLCE_NoFFT",
    "optimized_baseline": "nnUNetTrainerPICAI_DiceCE_NoFFT",
    "image_gate": "nnUNetTrainerPICAI_ImageGate",
    "anatomy_gate": "nnUNetTrainerPICAI_AnatomyGate",
    "image_gate_positive_sampling": "nnUNetTrainerPICAI_ImageGate_PositiveSampling_NoFFT",
    "anatomy_gate_positive_sampling": (
        "nnUNetTrainerPICAI_AnatomyGate_PositiveSampling_NoFFT"
    ),
    "feature_no_gate_positive_sampling": (
        "nnUNetTrainerPICAI_FeatureNoGate_PositiveSampling_NoFFT"
    ),
    "feature_image_gate_positive_sampling": (
        "nnUNetTrainerPICAI_FeatureImageGate_PositiveSampling_NoFFT"
    ),
    "feature_anatomy_gate_positive_sampling": (
        "nnUNetTrainerPICAI_FeatureAnatomyGate_PositiveSampling_NoFFT"
    ),
    "feature_no_gate_positive_sampling_100ep": (
        "nnUNetTrainerPICAI_FeatureNoGate_PositiveSampling_100ep_NoFFT"
    ),
    "feature_anatomy_gate_positive_sampling_100ep": (
        "nnUNetTrainerPICAI_FeatureAnatomyGate_PositiveSampling_100ep_NoFFT"
    ),
    "zonal_reference_positive_sampling_100ep": (
        "nnUNetTrainerPICAI_ZonalReference_PositiveSampling_100ep_NoFFT"
    ),
    "zonal_reference_adaptive_positive_sampling_100ep": (
        "nnUNetTrainerPICAI_ZonalReferenceAdaptive_PositiveSampling_100ep_NoFFT"
    ),
}

SUPPORTING_VARIANTS = {
    "lesion_roi": "nnUNetTrainerPICAI_LesionROI_NoFFT",
    "lesion_coarse_to_fine": "nnUNetTrainerPICAI_LesionCoarseToFine_NoFFT",
    "lesion_zone_refine": "nnUNetTrainerPICAI_LesionZoneRefine_NoFFT",
    "lesion_hard_negative": "nnUNetTrainerPICAI_LesionHardNegative_NoFFT",
}

ACTIVE_VARIANTS = {
    "v2_early_fusion": "nnUNetTrainerPICAI_V2_EarlyFusion_DiceCE_NoFFT",
    "v2_multi_encoder_simple": "nnUNetTrainerPICAI_V2_MultiEncoder_DiceCE_NoFFT",
    "v2_adaptive_fusion": "nnUNetTrainerPICAI_V2_AdaptiveFusion_DiceCE_NoFFT",
    "v2_anatomy_fusion": "nnUNetTrainerPICAI_V2_AnatomyFusion_DiceCE_NoFFT",
    "v2_coarse": "nnUNetTrainerPICAI_V2_Coarse_DiceCE_NoFFT",
    "v2_refine": "nnUNetTrainerPICAI_V2_Refine_DiceCE_NoFFT",
    "v2_full_sched": "nnUNetTrainerPICAI_V2_FullScheduled_DiceCE_NoFFT",
    "v2_concat_fusion": "nnUNetTrainerPICAI_V2_ConcatFusion_DiceCE_NoFFT",
    "v2_concat_adaptive_fusion": "nnUNetTrainerPICAI_V2_ConcatAdaptiveFusion_DiceCE_NoFFT",
    "v2_concat_anatomy_fusion": "nnUNetTrainerPICAI_V2_ConcatAnatomyFusion_DiceCE_NoFFT",
}
V1_EXPLICIT_VARIANTS = {f"v1_{k}": v for k, v in {**FROZEN_V1_VARIANTS, **SUPPORTING_VARIANTS}.items()}
VARIANT_TO_TRAINER = {**ACTIVE_VARIANTS, **FROZEN_V1_VARIANTS, **V1_EXPLICIT_VARIANTS,
                      **SUPPORTING_VARIANTS, **LEGACY_VARIANTS}

#: nnU-Net 原生正式预算（Trainer 不声明短预算时使用）
DEFAULT_EPOCH_LIMIT = 1000


def expected_epoch_limit(trainer_class) -> int:
    """该 Trainer 的冻结 epoch 上界：短预算类显式声明，正式臂沿用 nnU-Net 原生 1000。

    正式 A/B/C/D **不**声明类属性（``nnUNetTrainer.__init__`` 里把实例 ``num_epochs``
    设为 1000），因此缺省回退值就是 1000。checkpoint 的 ``current_epoch`` 必须落在该上界
    之内，避免把 1000 epoch 的 checkpoint 续到 100 epoch 类或反向混用。
    """
    declared = getattr(trainer_class, "num_epochs", None)
    if isinstance(declared, bool) or not isinstance(declared, int) or declared <= 0:
        return DEFAULT_EPOCH_LIMIT
    return declared


def is_short_budget(trainer_class) -> bool:
    """该 Trainer 是否属于预先冻结的短预算（100 epoch）筛选臂（唯一真源在 trainers 模块）。"""
    from zonal_reliability_fusion.nnunet.trainers import SHORT_BUDGET_TRAINERS

    return trainer_class in SHORT_BUDGET_TRAINERS


#: 旧 100-epoch 同区参照分支：严格限定 Dataset606 / 3d_fullres
SHORT_FUSION_VARIANTS = frozenset(
    {
        "feature_no_gate_positive_sampling_100ep",
        "feature_anatomy_gate_positive_sampling_100ep",
        "zonal_reference_positive_sampling_100ep",
        "zonal_reference_adaptive_positive_sampling_100ep",
    }
)
#: Stage-1 解剖先验生成器
ANATOMY_VARIANT = "anatomy_joint_100ep"
#: 需要 anatomy-guided ROI 集合的条件（B/C/D/E）
ROI_VARIANTS = frozenset(
    {
        "lesion_roi",
        "lesion_coarse_to_fine",
        "lesion_zone_refine",
        "lesion_hard_negative",
    }
)
#: 需要 soft PZ/TZ 的 zone 概率通道的条件（D/E）
ZONE_VARIANTS = frozenset({"lesion_zone_refine", "lesion_hard_negative"})
#: 可选启用困难负样本的条件（E）
HARD_NEGATIVE_VARIANTS = frozenset({"lesion_hard_negative"})


def guard_anatomy_checkpoint(
    output_folder, continue_training, validation_only, expected_dataset_json=None,
    *, expected_trainer_name=None, expected_epoch_limit=100, expected_init_args=None
):
    """Reject writes to existing validation artifacts before creating a Trainer."""
    validation = output_folder / "validation"
    if validation.exists() or validation.is_symlink():
        if not validation.is_dir() or any(validation.iterdir()):
            raise SystemExit(
                f"validation 已有产物或路径不可用，拒绝再次写入：{validation}；"
                "原生训练结束和 validation-only 都会写此目录，本入口不提供验证重跑"
            )
    final = output_folder / "checkpoint_final.pth"
    if continue_training and (final.exists() or final.is_symlink()):
        raise SystemExit(
            f"checkpoint_final 已存在，训练已完成，拒绝 continue-training 再次保存：{final}；"
            "仅在 validation 无产物且最终 checkpoint 可用时允许 --validation-only"
        )
    import torch

    if not continue_training and not validation_only:
        if output_folder.exists() or output_folder.is_symlink():
            raise SystemExit(f"输出目录已存在，拒绝从头覆盖：{output_folder}")
        return
    names = ("checkpoint_final.pth",) if validation_only else (
        "checkpoint_final.pth", "checkpoint_latest.pth", "checkpoint_best.pth")
    checkpoint = next(
        (output_folder / name for name in names if (output_folder / name).is_file()), None
    )
    if checkpoint is None:
        raise SystemExit(f"没有可用 checkpoint，拒绝静默重训：{output_folder}")
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        required = {"network_weights", "optimizer_state", "current_epoch", "trainer_name", "init_args"}
        if not isinstance(state, dict) or not required <= state.keys():
            raise ValueError("checkpoint fields missing")
        expected_name = expected_trainer_name or VARIANT_TO_TRAINER[ANATOMY_VARIANT]
        if state["trainer_name"] != expected_name or not state["network_weights"]:
            raise ValueError("wrong trainer or empty network weights")
        if continue_training:
            optimizer = state["optimizer_state"]
            if (
                not isinstance(optimizer, dict)
                or not isinstance(optimizer.get("state"), dict)
                or not isinstance(optimizer.get("param_groups"), list)
                or not optimizer["param_groups"]
            ):
                raise ValueError("optimizer state unavailable")
        if not isinstance(state["current_epoch"], int) or not 0 <= state["current_epoch"] <= expected_epoch_limit:
            raise ValueError("checkpoint epoch outside engineering budget")
        from zonal_reliability_fusion.anatomy.contracts import validate_anatomy_dataset

        init = state["init_args"]
        if expected_dataset_json is not None and init["dataset_json"] != expected_dataset_json:
            raise ValueError("checkpoint dataset provenance differs from current dataset")
        if expected_trainer_name is None:
            validate_anatomy_dataset(init["dataset_json"], init["plans"]["dataset_name"], init["configuration"])
            if str(init["fold"]) != "0":
                raise ValueError("checkpoint fold mismatch")
        else:
            if expected_init_args is None or any(
                init.get(key) != value for key, value in expected_init_args.items()
            ):
                raise ValueError("checkpoint plans/configuration/fold/dataset mismatch")
    except Exception as exc:
        raise SystemExit(f"不可用 checkpoint {checkpoint}: {exc}") from exc


def build_parser(include_legacy: bool = False, include_supporting: bool = False,
                 include_v1: bool = False) -> argparse.ArgumentParser:
    variants = {**ACTIVE_VARIANTS,
                **(FROZEN_V1_VARIANTS if include_v1 else {}),
                **(V1_EXPLICIT_VARIANTS if include_v1 else {}),
                **(SUPPORTING_VARIANTS if include_supporting else {}),
                **(LEGACY_VARIANTS if include_legacy else {})}
    parser = argparse.ArgumentParser(
        description="VERSION=V2 ACTIVE MAINLINE: native multi-encoder anatomy-guided coarse-to-fine. "
                    "--v1 exposes frozen V1 aliases; --supporting exposes frozen V1 ablations; "
                    "--legacy exposes historical gate experiments.",
        epilog="All outputs remain isolated by Trainer class name. Long tasks are user-run only.",
    )
    parser.add_argument("variant", choices=sorted(variants),
                        help="Explicit versioned V2 variants; historical names need a version flag")
    parser.add_argument("--v1", action="store_true", help="Expose frozen V1 variants / old command aliases")
    parser.add_argument("dataset", help="数据集 ID 或名称，例如 605 / 607 / 608")
    parser.add_argument("configuration", help="nnU-Net 配置，例如 3d_fullres")
    parser.add_argument("fold", help="fold 编号（0-4）或 'all'")
    parser.add_argument(
        "--continue-training",
        action="store_true",
        help="从同一输出目录的 checkpoint 续训",
    )
    parser.add_argument(
        "--validation-only", action="store_true", help="只运行训练完成后的 nnU-Net validation"
    )
    parser.add_argument(
        "--export-validation-probabilities",
        action="store_true",
        help="validation 时导出原生恢复概率（npz）；解剖区域为 WG/PZ/TZ sigmoid，解剖分支必填",
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu", "mps"),
        default="cuda",
        help="训练设备（GPU 编号用 CUDA_VISIBLE_DEVICES 控制）",
    )
    parser.add_argument("--seeded-output", action="store_true",
                        help="Use a deterministic _Seed<seed> Trainer class and independent output directory")
    parser.add_argument("--supporting", action="store_true", help="Show supporting ROI/refinement ablations")
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="把归档的旧门控 / 融合条件加入可选的 variant 列表（默认不显示）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "显式随机种子（Python / NumPy / torch）。改善可重复性但不保证 bitwise 确定性"
            "（nnU-Net 的多进程增强顺序非确定）；最终候选必须跑 3 个 seed。"
        ),
    )
    parser.add_argument(
        "--roi-set",
        default=None,
        help="Anatomy-Guided ROI 集合 JSON（仅冻结 V1 supporting ROI/refinement 条件；V2 不使用）",
    )
    parser.add_argument(
        "--hard-negative-set",
        default=None,
        help="困难负样本集合 JSON（仅冻结 V1 supporting HN 条件；V2 不使用）",
    )
    parser.add_argument(
        "--run-config",
        default=None,
        help="把本次运行的冻结配置写到该 JSON 路径（dataset/fold/seed/variant/spacing 等）",
    )
    return parser


def resolve_trainer_class(variant: str):
    """variant -> 项目自定义 Trainer 类（延迟导入，避免 ``--help`` 触发重依赖）。"""
    from zonal_reliability_fusion.nnunet import trainers

    class_name = VARIANT_TO_TRAINER[variant]
    trainer_class = getattr(trainers, class_name, None)
    if trainer_class is None:  # pragma: no cover - 防御性
        raise RuntimeError(f"trainers 模块缺少 Trainer 类 {class_name}")
    if class_name not in trainers.PROJECT_TRAINERS:
        raise RuntimeError(
            f"Trainer {class_name} 未注册进 PROJECT_TRAINERS：预测入口将无法解析其 checkpoint"
        )
    return trainer_class


def build_run_config(args, trainer_class, plans: dict, dataset_json: dict) -> dict:
    """组装可审计的 ``run_config.json``（新主线要求每个实验都能打印自己的冻结配置）。

    ``fusion_constants``/``epochs`` 全部**从代码读取**，不在此手写第二份常量：常量一旦
    与实现漂移，run_config 就会变成误导性的 provenance。
    """
    if getattr(trainer_class, "research_version", None) == "V2":
        from zonal_reliability_fusion.nnunet.runtime import v2_output_root
        from zonal_reliability_fusion.nnunet.seeds import describe_seed_limitations
        from zonal_reliability_fusion.v2.fusion.simple_fusion import (
            CONCAT_KERNEL_SIZE,
            POLICY_FEATURE_SCALE,
        )
        from zonal_reliability_fusion.v2.losses.coarse_loss import COARSE_LOSS_WEIGHT

        configuration = plans["configurations"][args.configuration]
        return {
            "research_version": "V2", "architecture_family": trainer_class.architecture_family,
            "variant": args.variant, "trainer": trainer_class.__name__,
            "fusion_type": trainer_class.fusion_type,
            "concat_fusion": getattr(trainer_class, "concat_fusion", False),
            "fusion_projection_kernel": (list(CONCAT_KERNEL_SIZE)
                                         if getattr(trainer_class, "concat_fusion", False) else None),
            "fusion_projection_scales": (list(range(configuration["architecture"]["arch_kwargs"]["n_stages"]))
                                         if getattr(trainer_class, "concat_fusion", False) else []),
            "policy_feature_scale": (POLICY_FEATURE_SCALE
                                     if trainer_class.fusion_type in ("concat_adaptive", "concat_anatomy") else None),
            "policy_zero_initialized": trainer_class.fusion_type in ("concat_adaptive", "concat_anatomy"),
            "anatomy_conditioned": trainer_class.anatomy_conditioned,
            "coarse_head": trainer_class.coarse_head, "refiner": trainer_class.refiner,
            "training_phase": (trainer_class.phase_for_epoch(0)
                               if hasattr(trainer_class, "phase_for_epoch") else "V2-EARLY"),
            "phase_scheduled": trainer_class.phase_scheduled,
            "phase_schedule": (list(trainer_class.phase_boundaries) if trainer_class.phase_scheduled else None),
            "coarse_loss_weight": COARSE_LOSS_WEIGHT if trainer_class.coarse_head else 0,
            "dataset": plans["dataset_name"], "configuration": args.configuration, "fold": args.fold,
            "seed": args.seed, "seed_note": describe_seed_limitations(),
            "epochs": expected_epoch_limit(trainer_class),
            "results_root": str(v2_output_root("nnUNet_results")),
            "architecture": configuration["architecture"],
            "selected_fusion_scales": (list(range(max(1, configuration["architecture"]["arch_kwargs"]["n_stages"] - 3),
                                                    configuration["architecture"]["arch_kwargs"]["n_stages"]))
                                       if trainer_class.fusion_type in ("adaptive", "anatomy", "concat_adaptive", "concat_anatomy") else []),
            "patch_size": configuration["patch_size"], "batch_size": configuration["batch_size"],
            "spacing": configuration["spacing"], "input_channels": dataset_json["channel_names"],
            "loss": "native DiceCE + native deep supervision; coarse Dice+BCE only when enabled",
            "sampling": "project positive-case sampling; validation native",
            "optimizer": "native SGD / PolyLR; no freeze or LR groups in first version",
            "anatomy_contract": dataset_json.get("predicted_anatomy_contract"),
        }
    from zonal_reliability_fusion.multimodal.conditioned_fusion import (
        SOFTMAX_TEMPERATURE,
        STEM_CHANNELS,
    )
    from zonal_reliability_fusion.nnunet.seeds import describe_seed_limitations
    from zonal_reliability_fusion.sampling.positive_sampling import PositiveCaseSamplingMixin

    configuration = plans.get("configurations", {}).get(args.configuration, {})
    archive = configuration.get("architecture", {})
    channel_names = dataset_json.get("channel_names", {})
    roi_payload = None
    if args.roi_set:
        roi_payload = {
            "path": str(args.roi_set),
            "sampling_probability": trainer_class.roi_sampling_probability,
        }
    return {
        "research_version": "LEGACY" if args.variant in LEGACY_VARIANTS else "V1",
        "dataset": plans.get("dataset_name"),
        "configuration": args.configuration,
        "fold": args.fold,
        "variant": args.variant,
        "trainer": trainer_class.__name__,
        "seed": args.seed,
        "seed_note": describe_seed_limitations(),
        "network": archive.get("network_class_name"),
        "input_channels": channel_names,
        "num_input_channels": len(channel_names),
        "loss": (
            "PI-CAI 0.5*Focal(gamma=2)+0.5*CE"
            if any("FLCE" in base.__name__ for base in trainer_class.__mro__)
            else "nnU-Net native Dice+CE（由继承解析决定，见 Trainer docstring）"
        ),
        "sampling": (
            "positive_sampling; ROI only for explicit supporting variants"
            if issubclass(trainer_class, PositiveCaseSamplingMixin)
            else "nnU-Net native case selection and foreground oversampling; no project positive-case balancing"
        ),
        "fusion_condition": getattr(trainer_class, "fusion_condition", None),
        "fusion_constants": {
            "stem_channels": STEM_CHANNELS,
            "temperature": SOFTMAX_TEMPERATURE,
            "zero_init_residual": True,
        },
        "epochs": expected_epoch_limit(trainer_class),
        "budget": (
            "short_budget_screening（100 epoch：仅 sanity 与方向筛选，不与 1000 epoch 比较）"
            if is_short_budget(trainer_class)
            else "formal（nnU-Net 原生 1000 epoch）"
        ),
        "batch_size": configuration.get("batch_size"),
        "patch_size": configuration.get("patch_size"),
        "spacing": configuration.get("spacing"),
        "anatomy_prior_source": (
            "predicted_prior（冻结 Stage-1 模型对该病例自身 MRI 的预测）"
            if args.variant in ROI_VARIANTS or getattr(trainer_class, "fusion_condition", None) == "anatomy"
            else "not_applicable"
        ),
        "roi_setting": roi_payload,
        "lesionness_setting": (
            {
                "enabled": True,
                "target": "物理半径膨胀的 coarse lesion mask",
                "radius_mm": __import__(
                    "zonal_reliability_fusion.lesion.lesionness", fromlist=["x"]
                ).LESIONNESS_DILATION_RADIUS_MM,
                "loss_weight": __import__(
                    "zonal_reliability_fusion.nnunet.losses", fromlist=["x"]
                ).LESIONNESS_LOSS_WEIGHT,
                "zone_mode": getattr(trainer_class, "lesion_zone_mode", None),
            }
            if args.variant in ("lesion_coarse_to_fine", "lesion_zone_refine", "lesion_hard_negative")
            or getattr(trainer_class, "fusion_condition", None) in ("lesion", "anatomy")
            else {"enabled": False, "reason": "该条件不构建 lesionness 头"}
        ),
        "hard_negative_setting": (
            {
                "path": args.hard_negative_set,
                "enabled": args.hard_negative_set is not None,
                "cases_per_batch": trainer_class.hard_negative_cases_per_batch,
                "scope": "training split only（validation 绝不参与挖掘）",
            }
            if args.variant in HARD_NEGATIVE_VARIANTS
            else {"enabled": False, "reason": "该条件不属于困难负样本臂"}
        ),
    }


def write_run_config(path: str, payload: dict) -> Path:
    """原子写出 run 配置；已存在则拒绝覆盖（与评估器、ROI 集合的落盘策略一致）。"""
    target = Path(path)
    if target.exists():
        raise SystemExit(f"run 配置已存在，拒绝静默覆盖：{target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False, allow_nan=False)
            fh.write("\n")
        tmp.replace(target)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return target


def main(argv=None) -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--legacy", action="store_true")
    pre.add_argument("--supporting", action="store_true")
    pre.add_argument("--v1", action="store_true")
    known, _rest = pre.parse_known_args(argv)
    import sys
    arguments = list(sys.argv[1:] if argv is None else argv)
    explicit_v1 = any(arg in V1_EXPLICIT_VARIANTS for arg in arguments)
    args = build_parser(include_legacy=known.legacy, include_supporting=known.supporting,
                        include_v1=known.v1 or explicit_v1).parse_args(arguments)
    if args.variant.startswith("v1_"):
        args.variant = args.variant[3:]  # historical guards/semantics stay unchanged
    if args.variant in ACTIVE_VARIANTS:
        if args.seed is None or not args.seeded_output or not args.run_config:
            raise SystemExit("V2 requires --seed / --seeded-output / --run-config for isolated provenance")
        if not Path(args.run_config).stem.startswith("V2_"):
            raise SystemExit("V2 run-config filename must start with V2_")
        if args.roi_set or args.hard_negative_set:
            raise SystemExit("V2 first version does not mix ROI or hard-negative strategies")
        if args.configuration != "3d_fullres" or args.fold != "0":
            raise SystemExit("V2 first version requires 3d_fullres / fold 0")
        required_dataset = "608" if "anatomy" in args.variant or args.variant in ("v2_coarse", "v2_refine", "v2_full_sched") else "605"
        required_name = "Dataset608_PICAI_PredictedAnatomy" if required_dataset == "608" else "Dataset605_PICAI"
        if args.dataset not in (required_dataset, required_name):
            raise SystemExit(f"{args.variant} requires {required_name}")
        from zonal_reliability_fusion.nnunet.runtime import require_v2_output_path
        try:
            require_v2_output_path(args.run_config, "run_configs")
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

    if args.continue_training and args.validation_only:
        raise SystemExit("--continue-training 与 --validation-only 不能同时使用")
    if args.variant in SHORT_FUSION_VARIANTS:
        if args.dataset not in ("606", "Dataset606_PICAI_Zonal") or args.configuration != "3d_fullres":
            raise SystemExit("_100ep 融合分支严格使用 Dataset606_PICAI_Zonal / 3d_fullres")
    if args.variant in ROI_VARIANTS and not args.roi_set:
        raise SystemExit(
            f"{args.variant} 必须提供 --roi-set（Anatomy-Guided ROI 是条件 B/C/D/E 的组成"
            "部分）：拒绝在没有解剖先验的情况下静默退化"
        )
    if args.variant not in HARD_NEGATIVE_VARIANTS and args.hard_negative_set:
        raise SystemExit(
            f"--hard-negative-set 只对 {sorted(HARD_NEGATIVE_VARIANTS)} 生效；"
            f"{args.variant} 不使用困难负样本"
        )
    if args.variant == ANATOMY_VARIANT:
        if args.dataset not in ("607", "Dataset607_PICAI_Anatomy") or args.configuration != "3d_fullres" or args.fold != "0":
            raise SystemExit("anatomy requires Dataset607_PICAI_Anatomy / 3d_fullres / explicit fold 0")
        if not args.export_validation_probabilities:
            raise SystemExit("anatomy requires --export-validation-probabilities")

    import multiprocessing

    import torch
    from batchgenerators.utilities.file_and_folder_operations import join, load_json
    from nnunetv2.paths import nnUNet_preprocessed, nnUNet_results
    from nnunetv2.run import run_training as nnunet_run_training
    from nnunetv2.utilities.dataset_name_id_conversion import (
        maybe_convert_to_dataset_name,
    )

    from zonal_reliability_fusion.nnunet import check_fixed_nnunet_runtime
    from zonal_reliability_fusion.nnunet.seeds import apply_explicit_seed

    # 明确校验实际使用的 nnU-Net / DNA / PlainConvUNet 运行时（不升级、不覆盖任何包）
    check_fixed_nnunet_runtime()

    if args.variant in ACTIVE_VARIANTS:
        from zonal_reliability_fusion.nnunet.runtime import configure_v2_training_outputs
        nnUNet_results = str(configure_v2_training_outputs())
        print(f"[VERSION=V2] dedicated model root: {nnUNet_results}")

    trainer_class = resolve_trainer_class(args.variant)
    if args.seeded_output:
        from zonal_reliability_fusion.nnunet.trainers import seeded_trainer_class
        trainer_class = seeded_trainer_class(trainer_class, args.seed)

    # 新条件的集合路径必须在 Trainer 构造（会自动调用 initialize）之前落到类属性上；
    # 本进程只训练一个 Trainer，因此修改类属性不会影响其它实验。
    if args.variant in ROI_VARIANTS:
        if not hasattr(trainer_class, "roi_set_path"):
            raise SystemExit(f"{trainer_class.__name__} 不支持 --roi-set")
        trainer_class.roi_set_path = args.roi_set
    if args.variant in HARD_NEGATIVE_VARIANTS and hasattr(
        trainer_class, "hard_negative_set_path"
    ):
        trainer_class.hard_negative_set_path = args.hard_negative_set

    if args.variant in SHORT_FUSION_VARIANTS:
        output_folder = (
            Path(nnUNet_results) / "Dataset606_PICAI_Zonal"
            / f"{trainer_class.__name__}__nnUNetPlans__3d_fullres" / f"fold_{args.fold}"
        )
        if not args.continue_training and not args.validation_only and output_folder.exists():
            raise SystemExit(f"输出目录已存在，拒绝从头覆盖：{output_folder}")
        if args.continue_training and not any(
            (output_folder / filename).is_file()
            for filename in ("checkpoint_final.pth", "checkpoint_latest.pth", "checkpoint_best.pth")
        ):
            raise SystemExit(f"没有可续训的 checkpoint，拒绝静默重训：{output_folder}")

    anatomy_folder = None
    if args.variant == ANATOMY_VARIANT:
        from zonal_reliability_fusion.anatomy.contracts import (
            ANATOMY_DATASET,
            validate_anatomy_dataset,
            validate_anatomy_split,
        )

        folder = Path(nnUNet_preprocessed) / ANATOMY_DATASET
        try:
            anatomy_dataset_json = load_json(str(folder / "dataset.json"))
            validate_anatomy_dataset(anatomy_dataset_json)
            validate_anatomy_split(folder / "splits_final.json", anatomy_dataset_json, args.fold)
        except (ValueError, OSError, KeyError) as exc:
            raise SystemExit(f"anatomy preflight failed: {exc}") from exc
        output_folder = (
            Path(nnUNet_results) / ANATOMY_DATASET
            / f"{trainer_class.__name__}__nnUNetPlans__3d_fullres" / "fold_0"
        )
        guard_anatomy_checkpoint(
            output_folder, args.continue_training, args.validation_only, anatomy_dataset_json
        )
        anatomy_folder = folder

    # 显式种子：在构造 Trainer（进而构造模型与 dataloader）之前设置
    seed_info = apply_explicit_seed(args.seed, verbose=True)

    device = torch.device(args.device)
    if args.device == "cuda":
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    elif args.device == "cpu":
        torch.set_num_threads(multiprocessing.cpu_count())

    dataset_folder = (
        anatomy_folder
        if anatomy_folder is not None
        else Path(nnUNet_preprocessed)
        / maybe_convert_to_dataset_name(
            int(args.dataset) if args.dataset.isdigit() else args.dataset
        )
    )
    plans = load_json(str(dataset_folder / "nnUNetPlans.json"))
    dataset_json = load_json(str(dataset_folder / "dataset.json"))
    if args.variant in {**ACTIVE_VARIANTS, **FROZEN_V1_VARIANTS} and args.variant != "anatomy_joint_100ep":
        output_folder = (Path(nnUNet_results) / plans["dataset_name"]
                         / f"{trainer_class.__name__}__nnUNetPlans__{args.configuration}"
                         / f"fold_{args.fold}")
        guard_anatomy_checkpoint(
            output_folder, args.continue_training, args.validation_only,
            expected_dataset_json=dataset_json, expected_trainer_name=trainer_class.__name__,
            # 上界由 Trainer 类自身决定（短预算类声明 100；正式臂回退 1000），不在此硬编码
            expected_epoch_limit=expected_epoch_limit(trainer_class), expected_init_args={
                "plans": plans, "configuration": args.configuration,
                "fold": int(args.fold) if args.fold != "all" else "all", "dataset_json": dataset_json})
        if getattr(trainer_class, "anatomy_conditioned", False) or getattr(trainer_class, "fusion_condition", None) == "anatomy":
            from zonal_reliability_fusion.anatomy.dataset import validate_predicted_prior_dataset
            validate_predicted_prior_dataset(dataset_json, plans["dataset_name"])
        elif plans["dataset_name"] != "Dataset605_PICAI":
            raise SystemExit("neutral/lesion fusion requires Dataset605_PICAI")
    if args.run_config:
        payload = build_run_config(args, trainer_class, plans, dataset_json)
        payload["seed_applied"] = seed_info
        written = write_run_config(args.run_config, payload)
        print(f"[run-config] 已写出冻结配置：{written}")

    def trainer_factory(
        dataset_name_or_id,
        configuration,
        fold,
        trainer_name=trainer_class.__name__,
        plans_identifier="nnUNetPlans",
        device=device,
    ):
        """按 nnU-Net ``get_trainer_from_args`` 的签名直接实例化项目 Trainer。

        nnU-Net 的 ``recursive_find_python_class`` 只在 nnunetv2 包目录内搜索，找不到项目自定义
        Trainer；这里在本进程内替换该 factory（不修改 nnU-Net 源码），其余流程仍走官方实现。
        """
        ds = dataset_name_or_id
        if isinstance(ds, str) and not ds.startswith("Dataset"):
            ds = int(ds)
        dataset_name = maybe_convert_to_dataset_name(ds)
        folder = join(nnUNet_preprocessed, dataset_name)
        return trainer_class(
            plans=load_json(join(folder, plans_identifier + ".json")),
            configuration=configuration,
            fold=fold,
            dataset_json=load_json(join(folder, "dataset.json")),
            device=device,
        )

    nnunet_run_training.get_trainer_from_args = trainer_factory
    import time
    started = time.perf_counter()
    try:
        nnunet_run_training.run_training(
            dataset_name_or_id=args.dataset,
            configuration=args.configuration,
            fold=args.fold,
            trainer_class_name=trainer_class.__name__,
            plans_identifier="nnUNetPlans",
            num_gpus=1,
            export_validation_probabilities=args.export_validation_probabilities,
            continue_training=args.continue_training,
            only_run_validation=args.validation_only,
            device=device,
        )
    except Exception:
        if args.variant in ACTIVE_VARIANTS:
            print(json.dumps({"research_version": "V2", "success_runs": 0, "failed_runs": 1,
                              "skipped_runs": 0, "elapsed_seconds": time.perf_counter() - started,
                              "output_path": str(output_folder)}, ensure_ascii=False))
        raise
    if args.variant in ACTIVE_VARIANTS:
        print(json.dumps({"research_version": "V2", "success_runs": 1, "failed_runs": 0,
                          "skipped_runs": 0, "elapsed_seconds": time.perf_counter() - started,
                          "output_path": str(output_folder)}, ensure_ascii=False))


    if args.variant == ANATOMY_VARIANT:
        # Native validation already restored probabilities; only validate its artifacts here.
        from nnunetv2.paths import nnUNet_raw

        from zonal_reliability_fusion.anatomy.contracts import ANATOMY_DATASET
        from zonal_reliability_fusion.anatomy.inference import (
            check_anatomy_prediction_folder,
        )

        check_anatomy_prediction_folder(
            output_folder / "validation",
            Path(nnUNet_raw) / ANATOMY_DATASET / "imagesTr",
            anatomy_dataset_json["anatomy_contract"]["val_cases"],
            load_json(str(anatomy_folder / "nnUNetPlans.json"))["transpose_forward"],
        )


if __name__ == "__main__":
    import os

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
    main()
