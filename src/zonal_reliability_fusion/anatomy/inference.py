"""Stage 1 解剖先验的**预测入口契约**：CLI 守卫 + 导出产物完整性校验。

本模块把原先散落在 ``scripts/inference/predict_nnunet.py`` 里的解剖专用逻辑收拢过来，使
预测脚本只负责「安装项目 Trainer 解析器 -> 调用官方 entry point」。**推理本身仍然完全由
nnU-Net 官方实现完成**，这里不做任何滑窗、重采样或坐标变换。

守卫内容（全部 fail-closed）
----------------------------
- 解剖模型只能来自 ``Dataset607_PICAI_Anatomy / 3d_fullres``，且目录名必须是那唯一的 Trainer
  目录（防止把解剖预检套用到别的模型上）；
- 必须 ``--save_probabilities``（下游需要 soft probability，硬标签不够）；
- 不支持 partitioned / cascade 预测；
- 输出目录非空即拒绝覆盖（与训练入口同一策略）；
- 输入必须是**单通道** T2W（``*_0000.nii.gz``），保证 anatomy model 的输入契约与训练一致；
- 预测完成后逐个病例校验 ``npz / pkl / nii.gz`` 三件套齐全、概率形状与物理元数据正确。
"""

from __future__ import annotations

import json
from pathlib import Path

from zonal_reliability_fusion.anatomy.contracts import (
    ANATOMY_CONFIGURATION,
    ANATOMY_DATASET,
    ANATOMY_PREDICTION_SUFFIXES,
    anatomy_same_grid,
    read_nifti_stored_array,
    validate_anatomy_dataset,
    validate_anatomy_probabilities,
    validate_label_array,
)

#: 唯一合法的解剖模型目录名（Trainer 类名 + plans + 配置）
ANATOMY_MODEL_FOLDER_NAME = (
    "nnUNetTrainerPICAI_AnatomyJoint_100ep_NoFFT__nnUNetPlans__3d_fullres"
)


def parse_model_folder_arguments(arguments):
    """解析固定的原生 model-folder CLI，支持 ``=`` 写法与原生缩写。

    只用于**校验 CLI 语法**；原始 argv 仍然原样交给 nnU-Net 官方 entry point。重复选项会被
    拒绝（而不是 last-wins），避免「用户以为指定了 A，实际生效的是 B」。
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
    for flag, kind in (
        ("-step_size", float),
        ("-chk", str),
        ("-npp", int),
        ("-nps", int),
        ("-prev_stage_predictions", str),
        ("-device", str),
        ("-num_parts", int),
        ("-part_id", int),
    ):
        parser.add_argument(flag, type=kind, action=UniqueArgument)
    for flags in (
        ("--disable_tta",),
        ("--verbose",),
        ("--save_probabilities",),
        ("--continue_prediction", "--c"),
        ("--disable_progress_bar",),
    ):
        parser.add_argument(*flags, nargs=0, const=True, default=False, action=UniqueArgument)
    options = parser.parse_args(arguments)
    for name in ("m", "i", "o", "prev_stage_predictions", "chk", "device"):
        if getattr(options, name) == "":
            parser.error(f"empty value for {name}")
    return options


def load_anatomy_probability_case(prediction_dir, cid, reference, transpose_forward):
    """读取并校验一个病例的三件套导出；返回 ``(probabilities, properties, native_export)``。

    ``reference`` 是**该病例自身 T2W** 的图像信息读取器（不是 GT 标签）——校验的是输出几何
    与输入影像一致，不做任何坐标变换。
    """
    import pickle

    import numpy as np
    import SimpleITK as sitk

    folder = Path(prediction_dir)
    for suffix in ANATOMY_PREDICTION_SUFFIXES:
        if not (folder / f"{cid}{suffix}").is_file():
            raise ValueError(f"{cid}: missing prediction artifact {suffix}")
    with np.load(folder / f"{cid}.npz", allow_pickle=False) as archive:
        if archive.files != ["probabilities"]:
            raise ValueError(f"{cid}: expected native probabilities key only")
        probabilities = archive["probabilities"]
    with (folder / f"{cid}.pkl").open("rb") as stream:
        properties = pickle.load(stream)  # 仅本地原生产物，绝不加载不可信下载
    if sorted(transpose_forward) != [0, 1, 2]:
        raise ValueError("invalid plans transpose_forward")
    properties = dict(properties, anatomy_transpose_forward=transpose_forward)
    probabilities = validate_anatomy_probabilities(probabilities, properties, reference)
    native = sitk.ReadImage(str(folder / f"{cid}.nii.gz"))
    anatomy_same_grid(reference, native)
    validate_label_array(
        read_nifti_stored_array(folder / f"{cid}.nii.gz"),
        [0, 1, 2, 4],
        "native_ordered_export",
    )
    return probabilities, properties, native


def check_anatomy_prediction_folder(
    prediction_dir, images_dir, case_ids, transpose_forward, no_progress=False
) -> None:
    """逐个病例校验解剖预测目录：病例集合精确匹配 + 三件套完整 + 概率/几何合法。

    退出前输出结构化汇总（成功 / 失败 / 跳过 / 耗时 / 输出路径），进度条不替代该汇总。
    任一病例失败即抛错（fail-closed），并列出全部错误。
    """
    import time

    import SimpleITK as sitk
    from tqdm import tqdm

    started = time.monotonic()
    folder, images = Path(prediction_dir), Path(images_dir)
    expected = set(case_ids)
    if len(expected) != len(case_ids) or not expected:
        raise ValueError("invalid expected anatomy prediction case set")
    for suffix in ANATOMY_PREDICTION_SUFFIXES:
        actual = {p.name[: -len(suffix)] for p in folder.glob(f"*{suffix}")}
        if actual != expected:
            raise ValueError(
                f"anatomy prediction case set mismatch for {suffix}: "
                f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
            )
    errors = []
    for cid in tqdm(
        case_ids, desc="anatomy-probability-check", unit="study", disable=no_progress
    ):
        try:
            reader = sitk.ImageFileReader()
            reader.SetFileName(str(images / f"{cid}_0000.nii.gz"))
            reader.ReadImageInformation()
            load_anatomy_probability_case(folder, cid, reader, transpose_forward)
        except Exception as exc:
            errors.append(f"{cid}: {exc}")
    print(
        f"[anatomy-probability-check] success={len(case_ids)-len(errors)} "
        f"failed={len(errors)} skipped=0 elapsed={time.monotonic()-started:.2f}s "
        f"output={folder}"
    )
    if errors:
        raise ValueError("\n".join(errors))


def anatomy_prediction_preflight(arguments):
    """只在解剖模型上生效的额外守卫；其余模型的预测完全走原生路径。

    返回 ``None``（不是解剖模型）或 ``(output, source, case_ids, transpose_forward)``。
    """
    if "-h" in arguments or "--help" in arguments:
        return None
    options = parse_model_folder_arguments(arguments)
    model_dir = Path(options.m)
    if not (model_dir / "dataset.json").is_file():
        return None  # 原生 parser/runtime 会给出它自己的错误
    dataset = json.loads((model_dir / "dataset.json").read_text())
    if (
        "anatomy_contract" not in dataset
        and model_dir.name != ANATOMY_MODEL_FOLDER_NAME
    ):
        return None
    plans = json.loads((model_dir / "plans.json").read_text())
    validate_anatomy_dataset(dataset, plans["dataset_name"], ANATOMY_CONFIGURATION)
    if model_dir.name != ANATOMY_MODEL_FOLDER_NAME:
        raise ValueError("unexpected anatomy model folder/trainer/configuration")
    if not options.save_probabilities:
        raise ValueError("anatomy prediction requires --save_probabilities")
    if any(
        getattr(options, name) is not None
        for name in ("num_parts", "part_id", "prev_stage_predictions")
    ):
        raise ValueError("anatomy minimal entry does not support partitioned/cascade prediction")
    source, output = Path(options.i), Path(options.o)
    if output.exists() and any(output.iterdir()):
        raise ValueError(
            "anatomy prediction output already contains artifacts; refuses overwrite"
        )
    files = list(source.glob("*.nii.gz"))
    if not files or any(not p.name.endswith("_0000.nii.gz") for p in files):
        raise ValueError("anatomy input must contain only single-channel T2W *_0000.nii.gz")
    ids = sorted(p.name[: -len("_0000.nii.gz")] for p in files)
    return output, source, ids, plans["transpose_forward"]


__all__ = (
    "ANATOMY_DATASET",
    "ANATOMY_MODEL_FOLDER_NAME",
    "anatomy_prediction_preflight",
    "check_anatomy_prediction_folder",
    "load_anatomy_probability_case",
    "parse_model_folder_arguments",
)
