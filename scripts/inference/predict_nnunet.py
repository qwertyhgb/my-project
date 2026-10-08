#!/usr/bin/env python3
"""最小项目预测入口：把项目 Trainer 名映射到项目类后，调用 nnU-Net 官方预测入口。

**不自写推理器**：滑窗推理、预处理、导出全部由官方 ``nnUNetPredictor`` 与
``predict_entry_point_modelfolder`` 完成；本脚本只做两件事：

1. 在**进程内**替换 ``nnunetv2.inference.predict_from_raw_data.recursive_find_python_class``，
   使官方 ``initialize_from_trained_model_folder`` 在读取 checkpoint 的 ``trainer_name`` 后，
   能直接定位到项目自定义 Trainer 类（对已知项目 Trainer 名做直接映射，**不做递归扫描**；
   未知名字仍委托给 nnU-Net 原函数）。不修改 third_party/nnUNet。
2. 校验固定 nnU-Net 运行时（``check_fixed_nnunet_runtime``），然后调用官方 entry point。

用法（长任务由研究者运行；先 conda activate lm && source scripts/env_nnunet.sh）：
    python scripts/inference/predict_nnunet.py \
        -i <输入影像目录> -o <输出目录> \
        -m outputs/nnUNet_results/Dataset605_PICAI/nnUNetTrainerPICAI_FLCE_NoFFT__nnUNetPlans__3d_fullres \
        -f 0 -device cuda
其余参数（-chk/--save_probabilities/-npp/-nps/--disable_tta 等）与官方
``nnUNetv2_predict_from_modelfolder`` 完全一致（``--help`` 即官方帮助）。
"""

from __future__ import annotations

import sys

#: 项目全部 Trainer 的类名（与 checkpoint 中的 ``trainer_name`` 一致）；模块级常量，便于测试
PROJECT_TRAINER_NAMES = (
    "nnUNetTrainerPICAI_AnatomyJoint_100ep_NoFFT",
    "nnUNetTrainerPICAI_FLCE_NoFFT",
    "nnUNetTrainerPICAI_DiceCE_NoFFT",
    "nnUNetTrainerPICAI_DiceCE_PositiveSampling_NoFFT",
    "nnUNetTrainerPICAI_ImageGate",
    "nnUNetTrainerPICAI_AnatomyGate",
    "nnUNetTrainerPICAI_FLCE_PositiveSampling_NoFFT",
    "nnUNetTrainerPICAI_ImageGate_PositiveSampling_NoFFT",
    "nnUNetTrainerPICAI_AnatomyGate_PositiveSampling_NoFFT",
    "nnUNetTrainerPICAI_FeatureNoGate_PositiveSampling_NoFFT",
    "nnUNetTrainerPICAI_FeatureImageGate_PositiveSampling_NoFFT",
    "nnUNetTrainerPICAI_FeatureAnatomyGate_PositiveSampling_NoFFT",
    "nnUNetTrainerPICAI_FeatureNoGate_PositiveSampling_100ep_NoFFT",
    "nnUNetTrainerPICAI_FeatureAnatomyGate_PositiveSampling_100ep_NoFFT",
    "nnUNetTrainerPICAI_ZonalReference_PositiveSampling_100ep_NoFFT",
    "nnUNetTrainerPICAI_ZonalReferenceAdaptive_PositiveSampling_100ep_NoFFT",
)

_ORIGINAL_RECURSIVE_FIND = None


def resolve_project_trainer(trainer_name: str):
    """按类名解析项目 Trainer 类；不是项目 Trainer 时返回 ``None``。"""
    from zonal_reliability_fusion.nnunet.trainers import resolve_trainer_class

    return resolve_trainer_class(trainer_name)


def install_project_trainer_resolver():
    """在进程内为官方预测模块安装 trainer 名解析器（幂等；不修改 nnU-Net 源码文件）。

    返回被安装到 ``predict_from_raw_data.recursive_find_python_class`` 的解析函数。
    """
    global _ORIGINAL_RECURSIVE_FIND
    import nnunetv2.inference.predict_from_raw_data as predict_module

    if _ORIGINAL_RECURSIVE_FIND is None:
        _ORIGINAL_RECURSIVE_FIND = predict_module.recursive_find_python_class
    original = _ORIGINAL_RECURSIVE_FIND

    def resolver(folder, class_name, current_module):
        project_class = resolve_project_trainer(class_name)
        if project_class is not None:
            # 直接映射，避免在 nnunetv2 包目录内递归扫描（也避免扫到环境中其他 nnunetv2 分支）
            return project_class
        return original(folder, class_name, current_module)

    predict_module.recursive_find_python_class = resolver
    return resolver


def load_anatomy_probability_case(prediction_dir, cid, reference, transpose_forward):
    """Read trusted nnU-Net exports; npz has arrays, pkl has physical properties."""
    import pickle
    import numpy as np
    from pathlib import Path
    from zonal_reliability_fusion.nnunet.trainers import anatomy_validate_probabilities, anatomy_same_grid, anatomy_validate_array, anatomy_read_array
    import SimpleITK as sitk
    folder = Path(prediction_dir)
    for suffix in (".npz", ".pkl", ".nii.gz"):
        if not (folder / f"{cid}{suffix}").is_file():
            raise ValueError(f"{cid}: missing prediction artifact {suffix}")
    with np.load(folder / f"{cid}.npz", allow_pickle=False) as archive:
        if archive.files != ["probabilities"]:
            raise ValueError(f"{cid}: expected native probabilities key only")
        probabilities = archive["probabilities"]
    with (folder / f"{cid}.pkl").open("rb") as stream:
        properties = pickle.load(stream)  # trusted local native artifacts, never untrusted downloads
    if sorted(transpose_forward) != [0, 1, 2]:
        raise ValueError("invalid plans transpose_forward")
    properties = dict(properties, anatomy_transpose_forward=transpose_forward)
    probabilities = anatomy_validate_probabilities(probabilities, properties, reference)
    native = sitk.ReadImage(str(folder / f"{cid}.nii.gz"))
    anatomy_same_grid(reference, native)
    anatomy_validate_array(anatomy_read_array(folder / f"{cid}.nii.gz"), [0, 1, 2, 4], "native_ordered_export")
    return probabilities, properties, native


def check_anatomy_prediction_folder(prediction_dir, images_dir, case_ids, transpose_forward, no_progress=False):
    import time
    from pathlib import Path
    import SimpleITK as sitk
    from tqdm import tqdm
    started = time.monotonic()
    folder, images = Path(prediction_dir), Path(images_dir)
    expected = set(case_ids)
    if len(expected) != len(case_ids) or not expected:
        raise ValueError("invalid expected anatomy prediction case set")
    for suffix in (".npz", ".pkl", ".nii.gz"):
        actual = {p.name[:-len(suffix)] for p in folder.glob(f"*{suffix}")}
        if actual != expected:
            raise ValueError(f"anatomy prediction case set mismatch for {suffix}: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")
    errors = []
    for cid in tqdm(case_ids, desc="anatomy-probability-check", unit="study", disable=no_progress):
        try:
            reader = sitk.ImageFileReader()
            reader.SetFileName(str(images / f"{cid}_0000.nii.gz"))
            reader.ReadImageInformation()
            load_anatomy_probability_case(folder, cid, reader, transpose_forward)
        except Exception as exc:
            errors.append(f"{cid}: {exc}")
    print(f"[anatomy-probability-check] success={len(case_ids)-len(errors)} failed={len(errors)} skipped=0 elapsed={time.monotonic()-started:.2f}s output={folder}")
    if errors:
        raise ValueError("\n".join(errors))


def _parse_prediction_protection_arguments(arguments):
    """Parse the fixed native model-folder CLI, including '=' and native abbreviations.

    The option table only validates CLI syntax; original argv still goes to the
    native entry point. Unique actions reject repeated options instead of last-wins.
    """
    import argparse

    class ProtectionParser(argparse.ArgumentParser):
        def error(self, message):
            raise ValueError(f"prediction arguments: {message}")

    class UniqueArgument(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            supplied = getattr(namespace, "_supplied", set())
            if self.dest in supplied:
                parser.error(f"duplicate option for {self.dest}: {option_string}")
            supplied.add(self.dest)
            namespace._supplied = supplied
            setattr(namespace, self.dest, self.const if self.nargs == 0 else values)

    parser = ProtectionParser(add_help=False)
    for flag in ("-m", "-i", "-o"):
        parser.add_argument(flag, required=True, action=UniqueArgument)
    parser.add_argument("-f", nargs="+", action=UniqueArgument)
    for flag, kind in (("-step_size", float), ("-chk", str), ("-npp", int),
                       ("-nps", int), ("-prev_stage_predictions", str), ("-device", str),
                       ("-num_parts", int), ("-part_id", int)):
        parser.add_argument(flag, type=kind, action=UniqueArgument)
    for flags in (("--disable_tta",), ("--verbose",), ("--save_probabilities",),
                  ("--continue_prediction", "--c"), ("--disable_progress_bar",)):
        parser.add_argument(*flags, nargs=0, const=True, default=False, action=UniqueArgument)
    options = parser.parse_args(arguments)
    for name in ("m", "i", "o", "prev_stage_predictions", "chk", "device"):
        if getattr(options, name) == "":
            parser.error(f"empty value for {name}")
    return options


def _anatomy_prediction_preflight(arguments):
    """Only anatomy models get extra guards; all prediction remains native."""
    import json
    from pathlib import Path
    if "-h" in arguments or "--help" in arguments:
        return None
    options = _parse_prediction_protection_arguments(arguments)
    model_dir = Path(options.m)
    if not (model_dir / "dataset.json").is_file():
        return None  # native parser/runtime supplies its usual error
    dataset = json.loads((model_dir / "dataset.json").read_text())
    if "anatomy_contract" not in dataset and model_dir.name != "nnUNetTrainerPICAI_AnatomyJoint_100ep_NoFFT__nnUNetPlans__3d_fullres":
        return None
    from zonal_reliability_fusion.nnunet.trainers import validate_anatomy_dataset
    plans = json.loads((model_dir / "plans.json").read_text())
    validate_anatomy_dataset(dataset, plans["dataset_name"], "3d_fullres")
    if model_dir.name != "nnUNetTrainerPICAI_AnatomyJoint_100ep_NoFFT__nnUNetPlans__3d_fullres":
        raise ValueError("unexpected anatomy model folder/trainer/configuration")
    if not options.save_probabilities:
        raise ValueError("anatomy prediction requires --save_probabilities")
    if any(getattr(options, name) is not None for name in ("num_parts", "part_id", "prev_stage_predictions")):
        raise ValueError("anatomy minimal entry does not support partitioned/cascade prediction")
    source, output = Path(options.i), Path(options.o)
    if output.exists() and any(output.iterdir()):
        raise ValueError("anatomy prediction output already contains artifacts; refuses overwrite")
    files = list(source.glob("*.nii.gz"))
    if not files or any(not p.name.endswith("_0000.nii.gz") for p in files):
        raise ValueError("anatomy input must contain only single-channel T2W *_0000.nii.gz")
    ids = sorted(p.name[:-12] for p in files)
    return output, source, ids, plans["transpose_forward"]


def main(argv=None) -> None:
    # 允许 `--help` 直接透传到官方 parser；其余情况先校验运行时再委托官方 entry point。
    from zonal_reliability_fusion.nnunet import check_fixed_nnunet_runtime

    check_fixed_nnunet_runtime()
    install_project_trainer_resolver()

    from nnunetv2.inference.predict_from_raw_data import predict_entry_point_modelfolder

    arguments = list(sys.argv[1:] if argv is None else argv)
    anatomy = _anatomy_prediction_preflight(arguments)
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *arguments]
        predict_entry_point_modelfolder()
    finally:
        sys.argv = original_argv
    if anatomy is not None:
        check_anatomy_prediction_folder(*anatomy)


if __name__ == "__main__":
    import os

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
    sys.exit(main())
