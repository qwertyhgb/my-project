#!/usr/bin/env python3
"""启动 PI-CAI 官方 Focal + CE 基线的单 GPU nnU-Net v2 适配入口。

该脚本不会修改 third_party/nnUNet，也不会复用默认 ``nnUNetTrainer`` 的输出
目录。实际训练只能由研究者手动执行；用 ``--help`` 仅检查参数装配。
"""
from __future__ import annotations

import argparse
import multiprocessing
import os
from pathlib import Path
from typing import Union

import torch
from batchgenerators.utilities.file_and_folder_operations import join, load_json
from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.run import run_training as nnunet_run_training
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name

from zonal_reliability_fusion.integrations.nnunet_picai_flce import nnUNetTrainerPICAI_FLCE_NoFFT


def get_picai_trainer(
    dataset_name_or_id: Union[int, str],
    configuration: str,
    fold: int,
    trainer_name: str = "nnUNetTrainerPICAI_FLCE",
    plans_identifier: str = "nnUNetPlans",
    device: torch.device = torch.device("cuda"),
) -> nnUNetTrainerPICAI_FLCE_NoFFT:
    """构造适配 trainer；签名与 nnU-Net v2 的 ``get_trainer_from_args`` 对齐。"""
    if trainer_name != "nnUNetTrainerPICAI_FLCE_NoFFT":
        raise ValueError(f"该入口只支持 nnUNetTrainerPICAI_FLCE_NoFFT，收到 {trainer_name}")
    if isinstance(dataset_name_or_id, str) and not dataset_name_or_id.startswith("Dataset"):
        try:
            dataset_name_or_id = int(dataset_name_or_id)
        except ValueError as exc:
            raise ValueError("dataset 必须是数字 ID 或 DatasetXXX_Name") from exc
    dataset_name = maybe_convert_to_dataset_name(dataset_name_or_id)
    dataset_folder = join(nnUNet_preprocessed, dataset_name)
    plans = load_json(join(dataset_folder, plans_identifier + ".json"))
    dataset_json = load_json(join(dataset_folder, "dataset.json"))
    return nnUNetTrainerPICAI_FLCE_NoFFT(
        plans=plans,
        configuration=configuration,
        fold=fold,
        dataset_json=dataset_json,
        device=device,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "PI-CAI official-style Focal + CE v2 port（No-FFT 修复版）。单 GPU；输出目录由 "
            "nnUNetTrainerPICAI_FLCE_NoFFT 类名自动隔离。"
        )
    )
    parser.add_argument("dataset_name_or_id", help="例如 605 或 Dataset605_PICAI")
    parser.add_argument("configuration", help="例如 3d_fullres")
    parser.add_argument("fold", type=int, help="fold 编号")
    parser.add_argument("--continue-training", action="store_true", help="仅续训同一 PICAI_FLCE_NoFFT 输出目录")
    parser.add_argument("--validation-only", action="store_true", help="只做训练完成后的 nnU-Net validation")
    parser.add_argument("--export-validation-probabilities", action="store_true", help="validation 时保存 npz")
    parser.add_argument("--device", choices=("cuda", "cpu", "mps"), default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.continue_training and args.validation_only:
        raise SystemExit("--continue-training 与 --validation-only 不能同时使用")

    # 该入口是显式单 GPU：避免通过 monkeypatch 的 class factory 进入 DDP 子进程。
    # 多 GPU 适配需要独立的、经过 DDP 合成测试的启动器，不能静默假装可用。
    device = torch.device(args.device)
    if args.device == "cuda":
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    elif args.device == "cpu":
        torch.set_num_threads(multiprocessing.cpu_count())

    # nnU-Net 的 run_training 负责标准 checkpoint、训练与实际 validation 流程。
    # 仅替换其本次进程内的 trainer factory，不写入或修改 nnU-Net 源码。
    nnunet_run_training.get_trainer_from_args = get_picai_trainer
    nnunet_run_training.run_training(
        dataset_name_or_id=args.dataset_name_or_id,
        configuration=args.configuration,
        fold=args.fold,
        trainer_class_name="nnUNetTrainerPICAI_FLCE_NoFFT",
        plans_identifier="nnUNetPlans",
        num_gpus=1,
        export_validation_probabilities=args.export_validation_probabilities,
        continue_training=args.continue_training,
        only_run_validation=args.validation_only,
        device=device,
    )


if __name__ == "__main__":
    # 与官方 nnUNetv2_train entry point 一致，限制 torch.compile 等后台线程数。
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
    main()
