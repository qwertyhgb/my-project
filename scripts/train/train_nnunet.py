#!/usr/bin/env python3
"""统一 nnU-Net 训练入口：variant -> Trainer 类 -> nnU-Net 官方 ``run_training``。

本脚本**只做映射与调用**：不实现任何训练循环、checkpoint、滑窗推理或验证逻辑，这些都交给
固定版本 nnU-Net（``third_party/nnUNet``，v2.6.2，只读）。

研究主线：**Anatomy-Guided Lesion-Aware Coarse-to-Fine Prostate Cancer Segmentation**

``--help`` 的 ``variant`` 列表默认只显示 **ACTIVE**（新主线），旧门控 / 融合条件在加上
``--legacy`` 后才出现。这是「训练入口分层」的落地：默认视图里不会再有十几个不知道哪个是主线
的 variant。

**ACTIVE**
    positive_sampling                  strong baseline A1：FLCE + 阳性采样
    dicece_positive_sampling           strong baseline A2：原生 Dice+CE + 阳性采样（只差损失）
    lesion_roi                         B：A + Anatomy-Guided ROI
    lesion_coarse_to_fine              C：B + lesionness 头 + soft refinement
    lesion_zone_refine                 D：C + soft PZ/TZ 解剖上下文
    lesion_hard_negative               E：best(C,D) + 解剖约束困难负样本
    anatomy_joint_100ep                Stage 1：T2W -> WG/PZ/TZ 先验生成器

**LEGACY**（需要 ``--legacy``）
    baseline / optimized_baseline / image_gate / anatomy_gate /
    image_gate_positive_sampling / anatomy_gate_positive_sampling /
    feature_no_gate_positive_sampling / feature_image_gate_positive_sampling /
    feature_anatomy_gate_positive_sampling / 四个 ``*_100ep`` 同区参照条件。
    保留是为了 checkpoint 兼容与历史实验可复现，**不是**主线。

用法（长任务由研究者本人运行）::

    cd /opt/data/private/lm/my-projects
    source /root/anaconda3/etc/profile.d/conda.sh && conda activate lm
    source scripts/env_nnunet.sh

    python scripts/train/train_nnunet.py --help          # 只看 ACTIVE
    python scripts/train/train_nnunet.py --help --legacy # 含 LEGACY

    # strong baseline A1 / A2（第一个要做的比较：只差损失）
    python scripts/train/train_nnunet.py positive_sampling 605 3d_fullres 0 --seed 20261008
    python scripts/train/train_nnunet.py dicece_positive_sampling 605 3d_fullres 0 --seed 20261008

    # 条件 B/C/D（需要先由数据脚本生成 ROI 集合）
    python scripts/train/train_nnunet.py lesion_roi 605 3d_fullres 0 \
        --seed 20261008 --roi-set workdir/anatomy_rois/Dataset605_PICAI_fold0.json
    python scripts/train/train_nnunet.py lesion_coarse_to_fine 605 3d_fullres 0 \
        --seed 20261008 --roi-set workdir/anatomy_rois/Dataset605_PICAI_fold0.json
    python scripts/train/train_nnunet.py lesion_zone_refine 606 3d_fullres 0 \
        --seed 20261008 --roi-set workdir/anatomy_rois/Dataset606_PICAI_Zonal_fold0.json

    # 条件 E（需要先由 scripts/data/mine_hard_negatives.py 生成困难负样本集合）
    python scripts/train/train_nnunet.py lesion_hard_negative 605 3d_fullres 0 \
        --seed 20261008 --roi-set <roi.json> --hard-negative-set <hn.json>

    # Stage 1
    python scripts/train/train_nnunet.py anatomy_joint_100ep 607 3d_fullres 0 \
        --export-validation-probabilities

可选：``--continue-training`` / ``--validation-only`` / ``--export-validation-probabilities`` /
``--device`` / ``--seed`` / ``--roi-set`` / ``--hard-negative-set`` / ``--run-config``。

**本轮不启动任何长训练。** 本文件只提供命令，命令由研究者本人执行。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

#: ACTIVE（研究主线）：默认 `--help` 只显示这些
ACTIVE_VARIANTS = {
    "positive_sampling": "nnUNetTrainerPICAI_FLCE_PositiveSampling_NoFFT",
    "dicece_positive_sampling": "nnUNetTrainerPICAI_DiceCE_PositiveSampling_NoFFT",
    "lesion_roi": "nnUNetTrainerPICAI_LesionROI_NoFFT",
    "lesion_coarse_to_fine": "nnUNetTrainerPICAI_LesionCoarseToFine_NoFFT",
    "lesion_zone_refine": "nnUNetTrainerPICAI_LesionZoneRefine_NoFFT",
    "lesion_hard_negative": "nnUNetTrainerPICAI_LesionHardNegative_NoFFT",
    "anatomy_joint_100ep": "nnUNetTrainerPICAI_AnatomyJoint_100ep_NoFFT",
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

VARIANT_TO_TRAINER = {**ACTIVE_VARIANTS, **LEGACY_VARIANTS}

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
    output_folder, continue_training, validation_only, expected_dataset_json=None
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
        if output_folder.exists():
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
        if state["trainer_name"] != VARIANT_TO_TRAINER[ANATOMY_VARIANT] or not state["network_weights"]:
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
        if not isinstance(state["current_epoch"], int) or not 0 <= state["current_epoch"] <= 100:
            raise ValueError("checkpoint epoch outside engineering budget")
        from zonal_reliability_fusion.anatomy.contracts import validate_anatomy_dataset

        init = state["init_args"]
        if expected_dataset_json is not None and init["dataset_json"] != expected_dataset_json:
            raise ValueError("checkpoint dataset provenance differs from current dataset")
        validate_anatomy_dataset(init["dataset_json"], init["plans"]["dataset_name"], init["configuration"])
        if str(init["fold"]) != "0":
            raise ValueError("checkpoint fold mismatch")
    except Exception as exc:
        raise SystemExit(f"不可用 anatomy checkpoint {checkpoint}: {exc}") from exc


def build_parser(include_legacy: bool = False) -> argparse.ArgumentParser:
    variants = {**ACTIVE_VARIANTS, **(LEGACY_VARIANTS if include_legacy else {})}
    parser = argparse.ArgumentParser(
        description=(
            "统一 nnU-Net-first 训练入口（新主线：Anatomy-Guided Lesion-Aware "
            "Coarse-to-Fine）。仅把 variant 映射到 Trainer 类并调用 nnU-Net 官方 "
            "run_training；输出目录由 Trainer 类名自动隔离，绝不复用/覆盖任何既有目录。"
            f"当前视图：{'ACTIVE + LEGACY' if include_legacy else '仅 ACTIVE（加 --legacy 查看归档条件）'}"
        ),
        epilog=(
            "变体分层：ACTIVE = "
            + ", ".join(sorted(ACTIVE_VARIANTS))
            + ("；LEGACY = " + ", ".join(sorted(LEGACY_VARIANTS)) if include_legacy else "")
        ),
    )
    parser.add_argument(
        "variant",
        choices=sorted(variants),
        help=(
            "[ACTIVE] positive_sampling / dicece_positive_sampling = strong baseline A1 / A2"
            "（唯一差异是损失）；lesion_roi = A + Anatomy-Guided ROI（条件 B）；"
            "lesion_coarse_to_fine = B + lesionness 头 + soft refinement（条件 C）；"
            "lesion_zone_refine = C + soft PZ/TZ 条件化（条件 D，需 zone 概率通道）；"
            "lesion_hard_negative = best(C,D) + 困难负样本（条件 E，可用 --hard-negative-set 开关）；"
            "anatomy_joint_100ep = Stage 1 解剖先验生成器。"
            + (" [LEGACY] 其余为归档的旧门控 / 融合条件。" if include_legacy else "")
        ),
    )
    parser.add_argument("dataset", help="数据集 ID 或名称，例如 605 / 606 / 607")
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
        help="Anatomy-Guided ROI 集合 JSON（条件 B/C/D/E 必填；由数据脚本用预测 WG 生成）",
    )
    parser.add_argument(
        "--hard-negative-set",
        default=None,
        help="困难负样本集合 JSON（条件 E 可选；不传则条件 E 完全退化为条件 C/D）",
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
    """组装可审计的 ``run_config.json``（新主线要求每个实验都能打印自己的冻结配置）。"""
    from zonal_reliability_fusion.nnunet.seeds import describe_seed_limitations

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
            if "Focal" in getattr(trainer_class, "__doc__", "") or "FLCE" in trainer_class.__name__
            else "nnU-Net native Dice+CE（由继承解析决定，见 Trainer docstring）"
        ),
        "sampling": "positive_sampling（每 batch 固定一个阳性病灶 patch）+ ROI 约束（若为 B/C/D/E）",
        "epochs": trainer_class.num_epochs
        if isinstance(getattr(trainer_class, "num_epochs", None), int)
        else "nnU-Net default（1000）",
        "batch_size": configuration.get("batch_size"),
        "patch_size": configuration.get("patch_size"),
        "spacing": configuration.get("spacing"),
        "anatomy_prior_source": (
            "predicted_prior（冻结 Stage-1 模型对该病例自身 MRI 的预测）"
            if args.variant in ROI_VARIANTS
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
    known, _rest = pre.parse_known_args(argv)
    args = build_parser(include_legacy=known.legacy).parse_args(argv)

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

    trainer_class = resolve_trainer_class(args.variant)

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
