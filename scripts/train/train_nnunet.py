#!/usr/bin/env python3
"""统一 nnU-Net 训练入口：variant -> Trainer 类 -> nnU-Net 官方 run_training。

本脚本**只做映射与调用**，不实现任何训练循环、checkpoint、滑窗推理或验证逻辑；这些都由
固定版本 nnU-Net（third_party/nnUNet，v2.6.2）负责。三个 variant：

    baseline      -> nnUNetTrainerPICAI_FLCE_NoFFT   原生 PlainConvUNet + 3 个 MRI 通道
    image_gate    -> nnUNetTrainerPICAI_ImageGate    3 个 MRI 通道 + image reliability gate
    anatomy_gate  -> nnUNetTrainerPICAI_AnatomyGate  5 通道（MRI + PZ/TZ）+ anatomy gate

用法（长任务由研究者本人运行）：
    python scripts/train/train_nnunet.py baseline 605 3d_fullres 0
    python scripts/train/train_nnunet.py image_gate 605 3d_fullres 0
    python scripts/train/train_nnunet.py anatomy_gate 606 3d_fullres 0

可选：--continue-training / --validation-only / --export-validation-probabilities / --device
运行前须：cd 项目根目录 && conda activate lm && source scripts/env_nnunet.sh
"""

from __future__ import annotations

import argparse

#: variant -> Trainer 类名（三个类名互不相同，nnU-Net 据此隔离 output folder）
VARIANT_TO_TRAINER = {
    "baseline": "nnUNetTrainerPICAI_FLCE_NoFFT",
    "image_gate": "nnUNetTrainerPICAI_ImageGate",
    "anatomy_gate": "nnUNetTrainerPICAI_AnatomyGate",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "统一 nnU-Net-first 训练入口。仅把 variant 映射到 Trainer 类并调用 nnU-Net 官方 "
            "run_training；输出目录由 Trainer 类名自动隔离，绝不复用/覆盖 baseline 目录。"
        )
    )
    parser.add_argument(
        "variant", choices=sorted(VARIANT_TO_TRAINER), help="模型 variant"
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
