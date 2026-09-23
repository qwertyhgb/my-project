#!/usr/bin/env python3
"""统一 nnU-Net 训练入口：variant -> Trainer 类 -> nnU-Net 官方 run_training。

本脚本**只做映射与调用**，不实现任何训练循环、checkpoint、滑窗推理或验证逻辑；这些都由
固定版本 nnU-Net（third_party/nnUNet，v2.6.2）负责。十个 variant：

    baseline                     -> nnUNetTrainerPICAI_FLCE_NoFFT   原生 PlainConvUNet + 3 MRI + PI-CAI Focal+CE
    optimized_baseline           -> nnUNetTrainerPICAI_DiceCE_NoFFT 原生 PlainConvUNet + 3 MRI + 原生 Dice+CE
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
同样使用阳性病例采样。十个 Trainer 的类名互不相同，输出目录由 nnU-Net 按 Trainer
类名自然隔离（不手工拼接路径）。

用法（长任务由研究者本人运行）：
    python scripts/train/train_nnunet.py baseline 605 3d_fullres 0
    python scripts/train/train_nnunet.py optimized_baseline 605 3d_fullres 0
    python scripts/train/train_nnunet.py image_gate 605 3d_fullres 0
    python scripts/train/train_nnunet.py anatomy_gate 606 3d_fullres 0
    python scripts/train/train_nnunet.py positive_sampling 605 3d_fullres 0
    python scripts/train/train_nnunet.py image_gate_positive_sampling 605 3d_fullres 0
    python scripts/train/train_nnunet.py anatomy_gate_positive_sampling 606 3d_fullres 0
    python scripts/train/train_nnunet.py feature_no_gate_positive_sampling 605 3d_fullres 0
    python scripts/train/train_nnunet.py feature_image_gate_positive_sampling 605 3d_fullres 0
    python scripts/train/train_nnunet.py feature_anatomy_gate_positive_sampling 606 3d_fullres 0

可选：--continue-training / --validation-only / --export-validation-probabilities / --device
运行前须：cd 项目根目录 && conda activate lm && source scripts/env_nnunet.sh
"""

from __future__ import annotations

import argparse

#: variant -> Trainer 类名（十个类名互不相同，nnU-Net 据此隔离 output folder）
VARIANT_TO_TRAINER = {
    "baseline": "nnUNetTrainerPICAI_FLCE_NoFFT",
    "optimized_baseline": "nnUNetTrainerPICAI_DiceCE_NoFFT",
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
}


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
            "image_gate = Dataset605 + 3 MRI + image gate + PI-CAI Focal+CE（原生采样）；"
            "anatomy_gate = Dataset606 + 5 通道（MRI + PZ/TZ）+ anatomy gate + PI-CAI Focal+CE"
            "（原生采样）；"
            "positive_sampling = Dataset605 + FLCE baseline + 每批固定一个阳性病灶 patch"
            "（只改训练采样，验证 loader 保持原生）；"
            "image_gate_positive_sampling = Dataset605 + image gate + 阳性病例采样"
            "（与 positive_sampling 配对用于 RQ1 公平比较；尚未训练）；"
            "anatomy_gate_positive_sampling = Dataset606 + anatomy gate + 阳性病例采样"
            "（与 image_gate_positive_sampling 配对用于 RQ2 公平比较；尚未训练）；"
            "feature_no_gate_positive_sampling = Dataset605 + 三个独立浅层 3×3×3 stem + "
            "1×1×1 投影到 3 通道 + 原生 backbone（无 gate）+ 阳性病例采样；"
            "feature_image_gate_positive_sampling = Dataset605 + 同上，但 feature gate 只读"
            "拼接后的 stem 特征（末层零初始化，初始 S≡1）；"
            "feature_anatomy_gate_positive_sampling = Dataset606 + 同上，但 feature gate 额外"
            "以 clamp(0,1) 后的 PZ/TZ 为条件（PZ/TZ 不进入 stem/投影/backbone）"
            "（三个 feature 条件尚未训练；README「浅层特征融合」有比较边界说明）"
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
        help="validation 时导出 softmax 概率（npz）",
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

    import multiprocessing

    import torch
    from batchgenerators.utilities.file_and_folder_operations import join, load_json
    from nnunetv2.paths import nnUNet_preprocessed
    from nnunetv2.run import run_training as nnunet_run_training
    from nnunetv2.utilities.dataset_name_id_conversion import (
        maybe_convert_to_dataset_name,
    )

    from zonal_reliability_fusion.nnunet import check_fixed_nnunet_runtime

    # 明确校验实际使用的 nnU-Net / DNA / PlainConvUNet 运行时（不升级、不覆盖任何包）
    check_fixed_nnunet_runtime()

    trainer_class = resolve_trainer_class(args.variant)

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


if __name__ == "__main__":
    import os

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
    main()
