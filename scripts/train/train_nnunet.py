#!/usr/bin/env python3
"""统一 nnU-Net 训练入口：variant -> Trainer 类 -> nnU-Net 官方 run_training。

本脚本**只做映射与调用**，不实现任何训练循环、checkpoint、滑窗推理或验证逻辑；这些都由
固定版本 nnU-Net（third_party/nnUNet，v2.6.2）负责。旧十一个 variant：

    baseline                     -> nnUNetTrainerPICAI_FLCE_NoFFT   原生 PlainConvUNet + 3 MRI + PI-CAI Focal+CE
    optimized_baseline           -> nnUNetTrainerPICAI_DiceCE_NoFFT 原生 PlainConvUNet + 3 MRI + 原生 Dice+CE（已中止）
    dicece_positive_sampling     -> nnUNetTrainerPICAI_DiceCE_PositiveSampling_NoFFT
                                    原生 PlainConvUNet + 原生 Dice+CE + 阳性病例采样
                                    （独立强参考基线 / 损失消融；不属于 A→B→C→D 主研究链）
    image_gate                   -> nnUNetTrainerPICAI_ImageGate    3 MRI + image gate（Focal+CE，原生采样）
    anatomy_gate                 -> nnUNetTrainerPICAI_AnatomyGate  5 通道（MRI + PZ/TZ）+ anatomy gate（Focal+CE，原生采样）
    positive_sampling            -> nnUNetTrainerPICAI_FLCE_PositiveSampling_NoFFT
                                    FLCE baseline + 每批固定一个阳性病灶 patch（验证 loader 保持原生）
    image_gate_positive_sampling -> nnUNetTrainerPICAI_ImageGate_PositiveSampling_NoFFT
                                    image gate + 阳性病例采样（与 positive_sampling 配对 = RQ1）
    anatomy_gate_positive_sampling -> nnUNetTrainerPICAI_AnatomyGate_PositiveSampling_NoFFT
                                    anatomy gate + 阳性病例采样（与 image_gate_positive_sampling 配对 = RQ2）
    feature_no_gate_positive_sampling -> nnUNetTrainerPICAI_FeatureNoGate_PositiveSampling_NoFFT
                                    三个独立浅层 stem + 投影，无 gate（隔离浅层编码/投影本身）
    feature_image_gate_positive_sampling -> nnUNetTrainerPICAI_FeatureImageGate_PositiveSampling_NoFFT
                                    feature gate（只读 stem 特征）+ 阳性病例采样（与 feature_no_gate 配对）
    feature_anatomy_gate_positive_sampling -> nnUNetTrainerPICAI_FeatureAnatomyGate_PositiveSampling_NoFFT
                                    feature gate + PZ/TZ 条件（与 feature_image_gate 配对 = RQ4）

`baseline` 与 `optimized_baseline` 是两个独立分支，唯一差异是损失；旧 `image_gate` /
`anatomy_gate` 使用原生采样，与 `positive_sampling` **不能**直接用于「只归因于 gate」的比较
（公平匹配见 `README.md`）。三个 `feature_*` 是同层级的一组条件（Research Plan §8.10），
同样使用阳性病例采样。`dicece_positive_sampling` 是**独立于 A→B→C→D 主研究链**的增强参考
基线：相对 `positive_sampling` 只换损失（FLCE → 原生 Dice+CE），相对 `optimized_baseline`
只换训练采样（原生 → 阳性病例采样）。另有四个独立 ``_100ep`` 别名，全部使用
Dataset606 / 3d_fullres：普通融合、普通分区 gate、同区参照修正、自适应参照强度。
十六个 Trainer 的类名互不相同，输出目录由 nnU-Net 按 Trainer 类名隔离；短预算分支
在调用原生训练前检查目录，拒绝从头覆盖与无 checkpoint 的静默续训回退。

用法（长任务由研究者本人运行）：
    python scripts/train/train_nnunet.py anatomy_joint_100ep 607 3d_fullres 0 --export-validation-probabilities
    python scripts/train/train_nnunet.py baseline 605 3d_fullres 0
    python scripts/train/train_nnunet.py optimized_baseline 605 3d_fullres 0
    python scripts/train/train_nnunet.py dicece_positive_sampling 605 3d_fullres 0
    python scripts/train/train_nnunet.py image_gate 605 3d_fullres 0
    python scripts/train/train_nnunet.py anatomy_gate 606 3d_fullres 0
    python scripts/train/train_nnunet.py positive_sampling 605 3d_fullres 0
    python scripts/train/train_nnunet.py image_gate_positive_sampling 605 3d_fullres 0
    python scripts/train/train_nnunet.py anatomy_gate_positive_sampling 606 3d_fullres 0
    python scripts/train/train_nnunet.py feature_no_gate_positive_sampling 605 3d_fullres 0
    python scripts/train/train_nnunet.py feature_image_gate_positive_sampling 605 3d_fullres 0
    python scripts/train/train_nnunet.py feature_anatomy_gate_positive_sampling 606 3d_fullres 0
    python scripts/train/train_nnunet.py feature_no_gate_positive_sampling_100ep 606 3d_fullres 0
    python scripts/train/train_nnunet.py feature_anatomy_gate_positive_sampling_100ep 606 3d_fullres 0
    python scripts/train/train_nnunet.py zonal_reference_positive_sampling_100ep 606 3d_fullres 0
    python scripts/train/train_nnunet.py zonal_reference_adaptive_positive_sampling_100ep 606 3d_fullres 0

可选：--continue-training / --validation-only / --export-validation-probabilities / --device
运行前须：cd 项目根目录 && conda activate lm && source scripts/env_nnunet.sh
"""

from __future__ import annotations

import argparse
from pathlib import Path

#: variant -> Trainer 类名（十六个类名互不相同，nnU-Net 据此隔离 output folder）
VARIANT_TO_TRAINER = {
    "anatomy_joint_100ep": "nnUNetTrainerPICAI_AnatomyJoint_100ep_NoFFT",
    "baseline": "nnUNetTrainerPICAI_FLCE_NoFFT",
    "optimized_baseline": "nnUNetTrainerPICAI_DiceCE_NoFFT",
    "dicece_positive_sampling": "nnUNetTrainerPICAI_DiceCE_PositiveSampling_NoFFT",
    "image_gate": "nnUNetTrainerPICAI_ImageGate",
    "anatomy_gate": "nnUNetTrainerPICAI_AnatomyGate",
    "positive_sampling": "nnUNetTrainerPICAI_FLCE_PositiveSampling_NoFFT",
    "image_gate_positive_sampling": (
        "nnUNetTrainerPICAI_ImageGate_PositiveSampling_NoFFT"
    ),
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


SHORT_FUSION_VARIANTS = frozenset({
    "feature_no_gate_positive_sampling_100ep", "feature_anatomy_gate_positive_sampling_100ep",
    "zonal_reference_positive_sampling_100ep", "zonal_reference_adaptive_positive_sampling_100ep"})
ANATOMY_VARIANT = "anatomy_joint_100ep"


def guard_anatomy_checkpoint(output_folder, continue_training, validation_only, expected_dataset_json=None):
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
    checkpoint = next((output_folder / name for name in names if (output_folder / name).is_file()), None)
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
            if (not isinstance(optimizer, dict) or not isinstance(optimizer.get("state"), dict)
                    or not isinstance(optimizer.get("param_groups"), list) or not optimizer["param_groups"]):
                raise ValueError("optimizer state unavailable")
        if not isinstance(state["current_epoch"], int) or not 0 <= state["current_epoch"] <= 100:
            raise ValueError("checkpoint epoch outside engineering budget")
        from zonal_reliability_fusion.nnunet.trainers import validate_anatomy_dataset
        init = state["init_args"]
        if expected_dataset_json is not None and init["dataset_json"] != expected_dataset_json:
            raise ValueError("checkpoint dataset provenance differs from current dataset")
        validate_anatomy_dataset(init["dataset_json"], init["plans"]["dataset_name"], init["configuration"])
        if str(init["fold"]) != "0":
            raise ValueError("checkpoint fold mismatch")
    except Exception as exc:
        raise SystemExit(f"不可用 anatomy checkpoint {checkpoint}: {exc}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "统一 nnU-Net-first 训练入口。仅把 variant 映射到 Trainer 类并调用 nnU-Net 官方 "
            "run_training；输出目录由 Trainer 类名自动隔离，绝不复用/覆盖 baseline 目录。"
        )
    )
    parser.add_argument(
        "variant",
        choices=sorted(VARIANT_TO_TRAINER),
        help=(
            "模型 variant："
            "baseline = Dataset605 + 原生 PlainConvUNet + PI-CAI Focal+CE + NoFFT（已完成的 N0）；"
            "optimized_baseline = Dataset605 + 原生 PlainConvUNet + 原生 Dice+CE + NoFFT"
            "（相对 baseline 的单变量改动：只换损失，用户已中止）；"
            "dicece_positive_sampling = Dataset605 + 原生 PlainConvUNet + 原生 Dice+CE + "
            "阳性病例采样（独立强参考基线 / 损失消融：相对 positive_sampling 只换损失，"
            "相对 optimized_baseline 只换训练采样；不属于 A→B→C→D 主研究链）；"
            "image_gate = Dataset605 + 3 MRI + image gate + PI-CAI Focal+CE（原生采样）；"
            "anatomy_gate = Dataset606 + 5 通道（MRI + PZ/TZ）+ anatomy gate + PI-CAI Focal+CE"
            "（原生采样）；"
            "positive_sampling = Dataset605 + FLCE baseline + 每批固定一个阳性病灶 patch"
            "（只改训练采样，验证 loader 保持原生）；"
            "image_gate_positive_sampling = Dataset605 + image gate + 阳性病例采样"
            "（与 positive_sampling 配对用于 RQ1 公平比较；已完成）；"
            "anatomy_gate_positive_sampling = Dataset606 + anatomy gate + 阳性病例采样"
            "（与 image_gate_positive_sampling 配对用于 RQ2 公平比较；已完成）；"
            "feature_no_gate_positive_sampling = Dataset605 + 三个独立浅层 3×3×3 stem + "
            "1×1×1 投影到 3 通道 + 原生 backbone（无 gate）+ 阳性病例采样；"
            "feature_image_gate_positive_sampling = Dataset605 + 同上，但 feature gate 只读"
            "拼接后的 stem 特征（末层零初始化，初始 S≡1）；"
            "feature_anatomy_gate_positive_sampling = Dataset606 + 同上，但 feature gate 额外"
            "以 clamp(0,1) 后的 PZ/TZ 为条件（PZ/TZ 不进入 stem/投影/backbone）"
            "；另有四个 _100ep 别名：同一 Dataset606、100 epochs，分别为普通拼接、分区门控、"
            "同区参照修正、同区参照自适应修正；仅限 3d_fullres，使用独立 Trainer 输出目录。"
        ),
    )
    parser.add_argument(
        "dataset", help="数据集 ID 或名称，例如 605 / 606 / Dataset606_PICAI_Zonal"
    )
    parser.add_argument("configuration", help="nnU-Net 配置，例如 3d_fullres")
    parser.add_argument("fold", help="fold 编号（0-4）或 'all'")
    parser.add_argument(
        "--continue-training",
        action="store_true",
        help="从同一输出目录的 checkpoint 续训",
    )
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="只运行训练完成后的 nnU-Net validation",
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
    return parser


def resolve_trainer_class(variant: str):
    """variant -> 项目自定义 Trainer 类（延迟导入，避免 --help 触发重依赖）。"""
    from zonal_reliability_fusion.nnunet import trainers

    class_name = VARIANT_TO_TRAINER[variant]
    trainer_class = getattr(trainers, class_name, None)
    if trainer_class is None:  # pragma: no cover - 防御性
        raise RuntimeError(f"trainers 模块缺少 Trainer 类 {class_name}")
    return trainer_class


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.continue_training and args.validation_only:
        raise SystemExit("--continue-training 与 --validation-only 不能同时使用")
    if args.variant in SHORT_FUSION_VARIANTS:
        if args.dataset not in ("606", "Dataset606_PICAI_Zonal") or args.configuration != "3d_fullres":
            raise SystemExit("_100ep 融合分支严格使用 Dataset606_PICAI_Zonal / 3d_fullres")

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

    # 明确校验实际使用的 nnU-Net / DNA / PlainConvUNet 运行时（不升级、不覆盖任何包）
    check_fixed_nnunet_runtime()

    trainer_class = resolve_trainer_class(args.variant)
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

    if args.variant == ANATOMY_VARIANT:
        from zonal_reliability_fusion.nnunet.trainers import validate_anatomy_dataset, validate_anatomy_split, ANATOMY_DATASET
        folder = Path(nnUNet_preprocessed) / ANATOMY_DATASET
        try:
            anatomy_dataset_json = load_json(str(folder / "dataset.json"))
            validate_anatomy_dataset(anatomy_dataset_json)
            validate_anatomy_split(folder / "splits_final.json", anatomy_dataset_json, args.fold)
        except (ValueError, OSError, KeyError) as exc:
            raise SystemExit(f"anatomy preflight failed: {exc}") from exc
        output_folder = Path(nnUNet_results) / ANATOMY_DATASET / f"{trainer_class.__name__}__nnUNetPlans__3d_fullres" / "fold_0"
        guard_anatomy_checkpoint(output_folder, args.continue_training, args.validation_only, anatomy_dataset_json)

    device = torch.device(args.device)
    if args.device == "cuda":
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    elif args.device == "cpu":
        torch.set_num_threads(multiprocessing.cpu_count())

    def trainer_factory(
        dataset_name_or_id,
        configuration,
        fold,
        trainer_name=trainer_class.__name__,
        plans_identifier="nnUNetPlans",
        device=device,
    ):
        """按 nnU-Net get_trainer_from_args 的签名直接实例化项目 Trainer。

        nnU-Net 的 recursive_find_python_class 只在 nnunetv2 包目录内搜索，找不到项目自定义
        Trainer；这里在本进程内替换该 factory（不修改 nnU-Net 源码），其余流程仍走官方实现。
        """
        ds = dataset_name_or_id
        if isinstance(ds, str) and not ds.startswith("Dataset"):
            ds = int(ds)
        dataset_name = maybe_convert_to_dataset_name(ds)
        folder = join(nnUNet_preprocessed, dataset_name)
        plans = load_json(join(folder, plans_identifier + ".json"))
        dataset_json = load_json(join(folder, "dataset.json"))
        return trainer_class(
            plans=plans,
            configuration=configuration,
            fold=fold,
            dataset_json=dataset_json,
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
        import importlib.util
        from nnunetv2.paths import nnUNet_raw
        spec = importlib.util.spec_from_file_location("anatomy_prediction_checks", Path(__file__).resolve().parents[1] / "inference/predict_nnunet.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.check_anatomy_prediction_folder(
            output_folder / "validation", Path(nnUNet_raw) / ANATOMY_DATASET / "imagesTr",
            anatomy_dataset_json["anatomy_contract"]["val_cases"],
            load_json(str(folder / "nnUNetPlans.json"))["transpose_forward"])


if __name__ == "__main__":
    import os

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
    main()
