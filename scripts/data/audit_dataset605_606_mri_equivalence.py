#!/usr/bin/env python3
"""Dataset605_PICAI 与 Dataset606_PICAI_Zonal 前三 MRI 通道逐数组一致性审计。

为什么需要它
------------
RQ2 的核心比较是 ``image_gate_positive_sampling``（Dataset605，3 通道）↔
``anatomy_gate_positive_sampling``（Dataset606，5 通道）。两者除门控条件外还使用不同的数据集，
因此只有在**预处理后实际用于训练的前三个 MRI 通道逐数组一致**的前提下，才允许把性能差异主要
解释为 PZ/TZ 条件的作用。配置层面（split / spacing / patch size / normalization）一致**不等于**
数组一致，本工具只回答后者。

审计对象（只读）
----------------
- ``nnUNet_preprocessed/<Dataset605>/nnUNetPlans_3d_fullres/`` 与
  ``nnUNet_preprocessed/<Dataset606>/nnUNetPlans_3d_fullres/`` 的 ``.b2nd`` / ``_seg.b2nd``；
- 两个数据集的 ``splits_final.json`` 与 ``dataset.json``（元数据层面）。

逐例检查：病例身份/集合、array shape、通道数、dtype、NaN/Inf、前三个 MRI 通道
（``0000``/``0001``/``0002`` = T2W/ADC/HBV）的 ``np.array_equal`` 与浮点差值统计、lesion label
（shape / dtype / 唯一值 / 逐值相等）。

**PZ/TZ（Dataset606 的第 4/5 通道）不参与数值一致性比较**；只记录其取值范围作为解剖通道的
信息性诊断（不影响 status）。

fail-closed
-----------
以下任一情况都不判 PASS，且**绝不修改任何输入**（不修复、不重建、不重新预处理）：
病例集合不同、split 不同、lesion label 不同、MRI shape 不同、缺通道、文件缺失/损坏、
NaN/Inf、数值不一致、dtype 不同、标签取值越界。

status 取值：``MRI_ARRAY_AUDIT_PASS`` / ``MRI_ARRAY_AUDIT_FAIL`` /
``MRI_ARRAY_AUDIT_PARTIAL``（用了 ``--max-cases`` 只跑了子集，永不判 PASS）。
退出码：PASS 0；FAIL / PARTIAL 2。

用法（长任务，由研究者本人运行）
-------------------------------
    cd /opt/data/private/lm/my-projects
    conda activate lm
    source scripts/env_nnunet.sh
    python scripts/data/audit_dataset605_606_mri_equivalence.py \
        --output outputs/reports/dataset605_606_mri_equivalence_audit.json

成功判据：退出码 0、status=MRI_ARRAY_AUDIT_PASS、``n_cases_mismatch=0``、
``per_channel.*.exact_equal_cases = n_cases_checked``。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

#: 前三个 MRI 通道的名字与索引（Dataset605/606 的前三通道顺序固定）
MRI_CHANNEL_NAMES = ("T2W", "ADC", "HBV")
MRI_CHANNELS = 3
#: Dataset606 在三个 MRI 之后附加的 PZ/TZ 通道数（不参与数值一致性比较）
PRIOR_CHANNELS_ANATOMY = 2
#: Dataset605/606 在 3d_fullres 下的通道数
CHANNELS_605 = MRI_CHANNELS
CHANNELS_606 = MRI_CHANNELS + PRIOR_CHANNELS_ANATOMY

STATUS_PASS = "MRI_ARRAY_AUDIT_PASS"
STATUS_FAIL = "MRI_ARRAY_AUDIT_FAIL"
STATUS_PARTIAL = "MRI_ARRAY_AUDIT_PARTIAL"

DEFAULT_DATASET_605 = "Dataset605_PICAI"
DEFAULT_DATASET_606 = "Dataset606_PICAI_Zonal"
DEFAULT_CONFIGURATION = "3d_fullres"
DEFAULT_OUTPUT = "outputs/reports/dataset605_606_mri_equivalence_audit.json"


class AuditError(RuntimeError):
    """审计前置条件不满足（环境、路径、元数据缺失等），fail-closed。"""


# --------------------------------------------------------------------------- 数据源


class PreprocessedCaseSource:
    """一个数据集某个配置下的预处理病例读取器（只读，复用 nnU-Net 自己的 dataset 类）。

    刻意通过 ``infer_dataset_class`` + ``load_case`` 读取，使审计看到的数组与训练时**完全相同**；
    不自行解析 ``.b2nd``、不复制 nnU-Net 的读取逻辑。``nnunetv2`` 延迟导入，便于单元测试使用
    同接口的合成数据源。
    """

    def __init__(self, dataset_dir: Path, configuration: str) -> None:
        from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class

        self.dataset_dir = Path(dataset_dir)
        self.configuration = str(configuration)
        self.case_folder = self.dataset_dir / f"nnUNetPlans_{self.configuration}"
        if not self.case_folder.is_dir():
            raise AuditError(
                f"预处理目录不存在：{self.case_folder}。"
                "请确认已 source scripts/env_nnunet.sh 且该数据集已完成 preprocessing。"
            )
        dataset_class = infer_dataset_class(str(self.case_folder))
        self._dataset = dataset_class(str(self.case_folder))

    @property
    def identifiers(self) -> tuple[str, ...]:
        return tuple(self._dataset.identifiers)

    def load(self, case_identifier: str) -> tuple[np.ndarray, np.ndarray]:
        """返回 ``(data, seg)``；``load_case`` 的原生签名是 ``(data, seg, seg_prev, properties)``。"""
        data, seg, _seg_prev, _properties = self._dataset.load_case(case_identifier)
        return np.asarray(data), np.asarray(seg)


def missing_preprocessed_files(case_folder: Path, case_identifier: str) -> list[str]:
    """列出该病例缺失的预处理文件（只做存在性检查，不读内容）。"""
    return [
        name
        for name in (
            f"{case_identifier}.b2nd",
            f"{case_identifier}_seg.b2nd",
            f"{case_identifier}.pkl",
        )
        if not (case_folder / name).is_file()
    ]


# --------------------------------------------------------------------------- 元数据


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise AuditError(f"缺少必需文件：{path}")
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:  # 统一转为 AuditError，避免静默继续
        raise AuditError(f"无法解析 {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise AuditError(f"{path} 的内容不是 JSON 对象")
    return payload


def compare_metadata(
    dataset_json_605: dict,
    dataset_json_606: dict,
    *,
    dataset_name_605: str,
    dataset_name_606: str,
) -> dict:
    """比较 dataset.json 的通道与标签定义（元数据层面，不是数组比较）。"""
    channels_605 = dataset_json_605.get("channel_names")
    channels_606 = dataset_json_606.get("channel_names")
    if not isinstance(channels_605, dict) or not isinstance(channels_606, dict):
        raise AuditError("dataset.json 缺少 channel_names 字典")
    labels_605 = dataset_json_605.get("labels")
    labels_606 = dataset_json_606.get("labels")

    def _first_mri(channels: dict) -> list[str]:
        # 真实 dataset.json 的通道键可能是 "0000" 形式（预处理产物）或 "0" 形式；
        # 一律按通道序号排序后取前三个，避免把键格式差异误判成通道名不一致。
        def sort_key(item):
            key = str(item[0])
            try:
                return (0, int(key))
            except ValueError:
                return (1, key)

        ordered = [str(value) for _, value in sorted(channels.items(), key=sort_key)]
        return ordered[:MRI_CHANNELS]

    return {
        "dataset_605": dataset_name_605,
        "dataset_606": dataset_name_606,
        "channel_names_605": {str(k): str(v) for k, v in channels_605.items()},
        "channel_names_606": {str(k): str(v) for k, v in channels_606.items()},
        "ordered_channel_names_605": _first_mri(channels_605),
        "ordered_channel_names_606": _first_mri(channels_606),
        "n_channels_605": len(channels_605),
        "n_channels_606": len(channels_606),
        "mri_channel_names_equal": _first_mri(channels_605)
        == _first_mri(channels_606)
        == list(MRI_CHANNEL_NAMES),
        "labels_equal": labels_605 == labels_606,
        "labels_605": labels_605,
        "labels_606": labels_606,
        "file_ending_equal": dataset_json_605.get("file_ending")
        == dataset_json_606.get("file_ending"),
        "num_training_equal": dataset_json_605.get("numTraining")
        == dataset_json_606.get("numTraining"),
    }


def compare_splits(splits_605: list, splits_606: list, *, fold: int) -> dict:
    """比较两个数据集的 ``splits_final.json``（fold 数、逐 fold 的 train/val 集合与顺序）。"""
    for tag, splits in (("605", splits_605), ("606", splits_606)):
        if not isinstance(splits, list) or not splits:
            raise AuditError(f"splits_final.json({tag}) 不是非空列表")

    folds: list[dict] = []
    for index in range(max(len(splits_605), len(splits_606))):
        in_605 = index < len(splits_605)
        in_606 = index < len(splits_606)
        entry: dict = {"fold": index, "in_605": in_605, "in_606": in_606}
        if in_605 and in_606:
            train_605, train_606 = (
                splits_605[index]["train"],
                splits_606[index]["train"],
            )
            val_605, val_606 = splits_605[index]["val"], splits_606[index]["val"]
            entry.update(
                {
                    "n_train_605": len(train_605),
                    "n_train_606": len(train_606),
                    "n_val_605": len(val_605),
                    "n_val_606": len(val_606),
                    "train_set_equal": set(train_605) == set(train_606),
                    "train_order_equal": list(train_605) == list(train_606),
                    "val_set_equal": set(val_605) == set(val_606),
                    "val_order_equal": list(val_605) == list(val_606),
                    "train_val_overlap_605": len(set(train_605) & set(val_605)),
                    "train_val_overlap_606": len(set(train_606) & set(val_606)),
                }
            )
        folds.append(entry)

    if fold < 0 or fold >= len(splits_605) or fold >= len(splits_606):
        raise AuditError(
            f"fold {fold} 在 splits_final.json 中不存在"
            f"（605 有 {len(splits_605)} 个 fold，606 有 {len(splits_606)} 个 fold）"
        )

    n_folds_equal = len(splits_605) == len(splits_606)
    all_entries_equal = all(
        bool(entry.get("train_set_equal")) and bool(entry.get("val_set_equal"))
        for entry in folds
        if entry["in_605"] and entry["in_606"]
    )
    return {
        "n_folds_605": len(splits_605),
        "n_folds_606": len(splits_606),
        "n_folds_equal": n_folds_equal,
        "fold": fold,
        "fold_0": folds[fold] if fold < len(folds) else None,
        "per_fold": folds,
        "all_folds_train_val_set_equal": bool(n_folds_equal and all_entries_equal),
    }


# --------------------------------------------------------------------------- 逐例比较


def compare_case(
    case_identifier: str,
    data_605: np.ndarray,
    seg_605: np.ndarray,
    data_606: np.ndarray,
    seg_606: np.ndarray,
    *,
    allowed_labels: tuple[int, ...],
) -> dict:
    """比较单个病例；返回结构化结果，``mismatches`` 非空即该例失败。

    只比较前三个 MRI 通道；Dataset606 的 PZ/TZ 只做信息性范围统计，不参与一致性判定。
    """
    mismatches: list[dict] = []
    channels: dict[str, dict] = {}

    if data_605.ndim != 4 or data_606.ndim != 4:
        mismatches.append(
            {
                "case_id": case_identifier,
                "channel": "ALL",
                "kind": "ndim_mismatch",
                "shape": [list(data_605.shape), list(data_606.shape)],
            }
        )
        return {
            "case_id": case_identifier,
            "exact_equal": False,
            "mismatches": mismatches,
            "channels": channels,
            "prior_channel_stats_606": None,
            "seg": {
                "exact_equal": False,
                "shape_605": list(seg_605.shape),
                "shape_606": list(seg_606.shape),
            },
        }

    if data_605.shape[0] != CHANNELS_605 or data_606.shape[0] != CHANNELS_606:
        mismatches.append(
            {
                "case_id": case_identifier,
                "channel": "ALL",
                "kind": "channel_count_mismatch",
                "shape": [list(data_605.shape), list(data_606.shape)],
            }
        )
    if tuple(data_605.shape[1:]) != tuple(data_606.shape[1:]):
        mismatches.append(
            {
                "case_id": case_identifier,
                "channel": "ALL",
                "kind": "spatial_shape_mismatch",
                "shape": [list(data_605.shape), list(data_606.shape)],
            }
        )
    if data_605.dtype != data_606.dtype:
        mismatches.append(
            {
                "case_id": case_identifier,
                "channel": "ALL",
                "kind": "dtype_mismatch",
                "dtype": [str(data_605.dtype), str(data_606.dtype)],
            }
        )
    if not bool(np.isfinite(data_605).all()) or not bool(np.isfinite(data_606).all()):
        mismatches.append(
            {
                "case_id": case_identifier,
                "channel": "ALL",
                "kind": "non_finite_values",
            }
        )

    n_channels_common = min(data_605.shape[0], data_606.shape[0], MRI_CHANNELS)
    for index in range(n_channels_common):
        name = MRI_CHANNEL_NAMES[index]
        channel_a = data_605[index]
        channel_b = data_606[index]
        shapes_equal = tuple(channel_a.shape) == tuple(channel_b.shape)
        info: dict = {
            "shape_605": list(channel_a.shape),
            "shape_606": list(channel_b.shape),
            "dtype_605": str(channel_a.dtype),
            "dtype_606": str(channel_b.dtype),
            "total_voxels": int(channel_a.size),
            "exact_equal": False,
            "max_abs_diff": None,
            "mean_abs_diff": None,
            "unequal_voxels": None,
            "fraction_unequal": None,
        }
        if not shapes_equal:
            info["max_abs_diff"] = None
            mismatches.append(
                {
                    "case_id": case_identifier,
                    "channel": name,
                    "kind": "channel_shape_mismatch",
                    "shape": [list(channel_a.shape), list(channel_b.shape)],
                    "max_abs_diff": None,
                    "mean_abs_diff": None,
                    "unequal_voxels": None,
                }
            )
            channels[name] = info
            continue

        exact = bool(np.array_equal(channel_a, channel_b))
        info["exact_equal"] = exact
        if exact:
            info.update(
                {"max_abs_diff": 0.0, "mean_abs_diff": 0.0, "unequal_voxels": 0}
            )
            info["fraction_unequal"] = 0.0
        else:
            diff = np.abs(channel_a.astype(np.float64) - channel_b.astype(np.float64))
            unequal = int(np.count_nonzero(diff))
            info.update(
                {
                    "max_abs_diff": float(diff.max()) if diff.size else 0.0,
                    "mean_abs_diff": float(diff.mean()) if diff.size else 0.0,
                    "unequal_voxels": unequal,
                }
            )
            info["fraction_unequal"] = (
                float(unequal) / float(diff.size) if diff.size else 0.0
            )
            mismatches.append(
                {
                    "case_id": case_identifier,
                    "channel": name,
                    "kind": "mri_values_differ",
                    "shape": [list(channel_a.shape), list(channel_b.shape)],
                    "max_abs_diff": info["max_abs_diff"],
                    "mean_abs_diff": info["mean_abs_diff"],
                    "unequal_voxels": unequal,
                    "fraction_unequal": info["fraction_unequal"],
                }
            )
        channels[name] = info

    prior_stats = None
    if data_606.shape[0] > MRI_CHANNELS:
        prior = data_606[MRI_CHANNELS:]
        prior_stats = {
            "n_channels": int(prior.shape[0]),
            "min": float(np.min(prior)) if prior.size else None,
            "max": float(np.max(prior)) if prior.size else None,
        }

    seg_info = {
        "shape_605": list(seg_605.shape),
        "shape_606": list(seg_606.shape),
        "dtype_605": str(seg_605.dtype),
        "dtype_606": str(seg_606.dtype),
        "labels_605": sorted(int(v) for v in np.unique(seg_605)),
        "labels_606": sorted(int(v) for v in np.unique(seg_606)),
        "exact_equal": False,
    }
    seg_invalid = sorted(
        ({int(v) for v in np.unique(seg_605)} | {int(v) for v in np.unique(seg_606)})
        - {int(label) for label in allowed_labels}
    )
    if seg_invalid:
        mismatches.append(
            {
                "case_id": case_identifier,
                "channel": "SEG",
                "kind": "unexpected_label_values",
                "labels": seg_invalid,
            }
        )
    if tuple(seg_605.shape) != tuple(seg_606.shape):
        mismatches.append(
            {
                "case_id": case_identifier,
                "channel": "SEG",
                "kind": "label_shape_mismatch",
                "shape": [list(seg_605.shape), list(seg_606.shape)],
            }
        )
    elif seg_605.dtype != seg_606.dtype:
        mismatches.append(
            {
                "case_id": case_identifier,
                "channel": "SEG",
                "kind": "label_dtype_mismatch",
                "dtype": [str(seg_605.dtype), str(seg_606.dtype)],
            }
        )
    elif not bool(np.array_equal(seg_605, seg_606)):
        diff = np.abs(seg_605.astype(np.int64) - seg_606.astype(np.int64))
        mismatches.append(
            {
                "case_id": case_identifier,
                "channel": "SEG",
                "kind": "label_values_differ",
                "unequal_voxels": int(np.count_nonzero(diff)),
                "max_abs_diff": int(diff.max()) if diff.size else 0,
            }
        )
    else:
        seg_info["exact_equal"] = True

    return {
        "case_id": case_identifier,
        "exact_equal": not mismatches,
        "channels": channels,
        "prior_channel_stats_606": prior_stats,
        "seg": seg_info,
        "mismatches": mismatches,
    }


# --------------------------------------------------------------------------- 聚合


def _empty_channel_aggregate() -> dict:
    return {
        "exact_equal_cases": 0,
        "mismatch_cases": 0,
        "global_max_abs_diff": 0.0,
        "global_mean_abs_diff": None,
        "total_unequal_voxels": 0,
        "total_voxels": 0,
        "_sum_abs_diff": 0.0,
    }


def run_audit(
    *,
    source_605,
    source_606,
    splits_605: list,
    splits_606: list,
    dataset_json_605: dict,
    dataset_json_606: dict,
    metadata_605: dict,
    metadata_606: dict,
    fold: int,
    max_cases: int | None = None,
    progress: bool = True,
) -> dict:
    """执行完整审计并返回结构化报告（不写文件、不修改任何输入）。"""
    started = time.time()
    identifiers_605 = tuple(source_605.identifiers)
    identifiers_606 = tuple(source_606.identifiers)
    set_605, set_606 = set(identifiers_605), set(identifiers_606)

    per_channel = {name: _empty_channel_aggregate() for name in MRI_CHANNEL_NAMES}
    mismatched_cases: list[dict] = []
    prior_stats: list[dict] = []
    n_checked = 0
    n_exact = 0
    n_label_equal = 0
    load_failures: list[dict] = []

    common = sorted(set_605 & set_606)
    only_605 = sorted(set_605 - set_606)
    only_606 = sorted(set_606 - set_605)
    for case_identifier in only_605 + only_606:
        mismatched_cases.append(
            {
                "case_id": case_identifier,
                "channel": "CASE",
                "kind": "case_missing_in_other_dataset",
                "present_in": "605" if case_identifier in set_605 else "606",
            }
        )

    to_check = common if max_cases is None else common[: int(max_cases)]
    allowed_labels = tuple(int(label) for label in dataset_json_605["labels"].values())

    iterator = tqdm(
        to_check,
        desc="MRI array audit",
        unit="case",
        disable=not progress,
    )
    for case_identifier in iterator:
        missing_605 = missing_preprocessed_files(
            source_605.case_folder, case_identifier
        )
        missing_606 = missing_preprocessed_files(
            source_606.case_folder, case_identifier
        )
        missing_entries = [
            {
                "case_id": case_identifier,
                "dataset": tag,
                "kind": "missing_preprocessed_file",
                "files": missing,
            }
            for tag, missing in (("605", missing_605), ("606", missing_606))
            if missing
        ]
        if missing_entries:
            load_failures.extend(missing_entries)
            mismatched_cases.extend(missing_entries)
            continue

        try:
            data_605, seg_605 = source_605.load(case_identifier)
            data_606, seg_606 = source_606.load(case_identifier)
        except Exception as exc:  # noqa: BLE001 - 读失败必须 fail-closed，不静默当作通过
            failure = {
                "case_id": case_identifier,
                "kind": "load_error",
                "error": f"{type(exc).__name__}: {exc}",
            }
            load_failures.append(failure)
            mismatched_cases.append(failure)
            continue

        result = compare_case(
            case_identifier,
            data_605,
            seg_605,
            data_606,
            seg_606,
            allowed_labels=allowed_labels,
        )
        n_checked += 1
        if result["exact_equal"]:
            n_exact += 1
        if result["seg"].get("exact_equal"):
            n_label_equal += 1
        if result["prior_channel_stats_606"] is not None:
            prior_stats.append(result["prior_channel_stats_606"])
        for name, info in result["channels"].items():
            aggregate = per_channel[name]
            aggregate["total_voxels"] += int(info["total_voxels"])
            if info["exact_equal"]:
                aggregate["exact_equal_cases"] += 1
                continue
            aggregate["mismatch_cases"] += 1
            if info["max_abs_diff"] is not None:
                aggregate["global_max_abs_diff"] = max(
                    float(aggregate["global_max_abs_diff"]), float(info["max_abs_diff"])
                )
                aggregate["total_unequal_voxels"] += int(info["unequal_voxels"])
                aggregate["_sum_abs_diff"] += float(info["mean_abs_diff"]) * int(
                    info["total_voxels"]
                )
        mismatched_cases.extend(result["mismatches"])

    for aggregate in per_channel.values():
        if aggregate["total_voxels"]:
            aggregate["global_mean_abs_diff"] = (
                aggregate["_sum_abs_diff"] / aggregate["total_voxels"]
            )
        aggregate.pop("_sum_abs_diff", None)

    metadata_report = compare_metadata(
        dataset_json_605,
        dataset_json_606,
        dataset_name_605=metadata_605["dataset_name"],
        dataset_name_606=metadata_606["dataset_name"],
    )
    splits_report = compare_splits(splits_605, splits_606, fold=fold)

    case_set_equal = set_605 == set_606
    split_equal = bool(splits_report["all_folds_train_val_set_equal"])
    label_equal = n_checked > 0 and n_label_equal == n_checked and not load_failures
    problems: list[str] = []
    if not case_set_equal:
        problems.append(
            f"病例集合不一致：仅 605 有 {len(only_605)} 例，仅 606 有 {len(only_606)} 例"
        )
    if not split_equal:
        problems.append("splits_final.json 的 fold 划分不一致")
    if not label_equal:
        problems.append("lesion label 不一致或存在未检查/读取失败的病例")
    if not metadata_report["mri_channel_names_equal"]:
        problems.append("前三个 MRI 通道名不是 T2W/ADC/HBV 或两数据集不一致")
    if not metadata_report["labels_equal"]:
        problems.append("dataset.json 的 labels 定义不一致")
    if load_failures:
        problems.append(f"{len(load_failures)} 例预处理文件缺失或读取失败")
    n_structural_mismatch = n_checked - n_exact
    if n_structural_mismatch:
        problems.append(
            f"{n_structural_mismatch} 例存在结构性不一致"
            "（通道数 / 空间形状 / dtype / NaN·Inf / label / MRI 数值，逐例明细见 mismatched_cases）"
        )
    for name, aggregate in per_channel.items():
        if aggregate["mismatch_cases"]:
            problems.append(
                f"{name} 通道有 {aggregate['mismatch_cases']} 例数值不一致"
                f"（global max|Δ|={aggregate['global_max_abs_diff']:.6g}）"
            )
    if max_cases is not None and len(to_check) < len(common):
        problems.append(
            f"仅检查了 {len(to_check)}/{len(common)} 例（--max-cases），不能判 PASS"
        )

    if problems:
        status = (
            STATUS_PARTIAL
            if (max_cases is not None and len(to_check) < len(common))
            else STATUS_FAIL
        )
    else:
        status = STATUS_PASS

    report = {
        "status": status,
        "problems": problems,
        "generated_at_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "elapsed_seconds": round(time.time() - started, 3),
        "dataset_605": metadata_605["dataset_name"],
        "dataset_606": metadata_606["dataset_name"],
        "configuration": metadata_605["configuration"],
        "fold": int(fold),
        "n_cases_605": len(identifiers_605),
        "n_cases_606": len(identifiers_606),
        "case_set_equal": bool(case_set_equal),
        "split_equal": bool(split_equal),
        "label_equal": bool(label_equal),
        "n_cases_checked": int(n_checked),
        "n_cases_exact_equal": int(n_exact),
        "n_cases_mismatch": int(n_checked - n_exact + len(load_failures)),
        "channels": list(MRI_CHANNEL_NAMES),
        "per_channel": per_channel,
        "mismatched_cases": mismatched_cases,
        "load_failures": load_failures,
        "metadata": metadata_report,
        "splits": splits_report,
        "prior_channels_606_info_only": {
            "note": "PZ/TZ 不参与前三 MRI 通道的数值一致性判定；以下仅为信息性范围统计",
            "n_channel_entries": len(prior_stats),
            "global_min": min(
                (s["min"] for s in prior_stats if s["min"] is not None), default=None
            ),
            "global_max": max(
                (s["max"] for s in prior_stats if s["max"] is not None), default=None
            ),
        },
        "inputs_modified": False,
        "notes": [
            "本工具只读；不修复数据、不重建 Dataset606、不重新 preprocessing。",
            "判定依据是 np.array_equal（最严格）；浮点差值统计仅用于诊断。",
            (
                f"PASS 定义：status={STATUS_PASS}，n_cases_mismatch=0，"
                "各 MRI 通道 exact_equal_cases = n_cases_checked。"
            ),
        ],
    }
    return report


# --------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audit_dataset605_606_mri_equivalence.py",
        description=(
            "Dataset605 与 Dataset606 预处理后前三个 MRI 通道（T2W/ADC/HBV）的逐数组一致性审计"
            "（只读、fail-closed）。PZ/TZ 不参与数值一致性判定。"
        ),
    )
    parser.add_argument("--dataset-605", default=DEFAULT_DATASET_605)
    parser.add_argument("--dataset-606", default=DEFAULT_DATASET_606)
    parser.add_argument("--configuration", default=DEFAULT_CONFIGURATION)
    parser.add_argument(
        "--fold", type=int, default=0, help="仅用于 split 报告与说明（默认 0）"
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="只检查前 N 个共同病例（烟雾测试用）；此时 status 恒为 PARTIAL，永不判 PASS",
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖已存在的报告文件（默认拒绝，避免覆盖既有审计结果）",
    )
    parser.add_argument("--no-progress", action="store_true")
    return parser


def _resolve_preprocessed_root() -> Path:
    raw = os.environ.get("nnUNet_preprocessed")
    if not raw:
        raise AuditError(
            "环境变量 nnUNet_preprocessed 未设置。请先执行："
            "cd /opt/data/private/lm/my-projects && conda activate lm && source scripts/env_nnunet.sh"
        )
    root = Path(raw)
    if not root.is_dir():
        raise AuditError(f"nnUNet_preprocessed 指向的目录不存在：{root}")
    return root


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.time()
    root = _resolve_preprocessed_root()
    dir_605 = root / args.dataset_605
    dir_606 = root / args.dataset_606

    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        raise SystemExit(
            f"[audit] 报告文件已存在，拒绝覆盖：{output_path}\n"
            "如需重新生成请显式加 --overwrite（或换一个 --output 路径）。"
        )

    source_605 = PreprocessedCaseSource(dir_605, args.configuration)
    source_606 = PreprocessedCaseSource(dir_606, args.configuration)
    splits_605 = _load_json(dir_605 / "splits_final.json")
    splits_606 = _load_json(dir_606 / "splits_final.json")
    dataset_json_605 = _load_json(dir_605 / "dataset.json")
    dataset_json_606 = _load_json(dir_606 / "dataset.json")

    report = run_audit(
        source_605=source_605,
        source_606=source_606,
        splits_605=splits_605,
        splits_606=splits_606,
        dataset_json_605=dataset_json_605,
        dataset_json_606=dataset_json_606,
        metadata_605={
            "dataset_name": args.dataset_605,
            "configuration": args.configuration,
        },
        metadata_606={
            "dataset_name": args.dataset_606,
            "configuration": args.configuration,
        },
        fold=args.fold,
        max_cases=args.max_cases,
        progress=not args.no_progress,
    )
    report["preprocessed_root"] = str(root)
    report["case_folder_605"] = str(source_605.case_folder)
    report["case_folder_606"] = str(source_606.case_folder)
    report["elapsed_seconds"] = round(time.time() - started, 3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    _print_summary(report, output_path)
    return 0 if report["status"] == STATUS_PASS else 2


def _print_summary(report: dict, output_path: Path) -> None:
    print("\n=== Dataset605 / Dataset606 MRI 数组审计 ===")
    print(f"  status            : {report['status']}")
    print(
        f"  病例集合一致       : {report['case_set_equal']}（605={report['n_cases_605']} / 606={report['n_cases_606']}）"
    )
    print(f"  split 一致         : {report['split_equal']}")
    print(f"  lesion label 一致  : {report['label_equal']}")
    print(
        f"  已检查/完全一致/不一致 : {report['n_cases_checked']} / "
        f"{report['n_cases_exact_equal']} / {report['n_cases_mismatch']}"
    )
    for name in report["channels"]:
        entry = report["per_channel"][name]
        print(
            f"  {name:>3s}: exact={entry['exact_equal_cases']} mismatch={entry['mismatch_cases']} "
            f"global_max|Δ|={entry['global_max_abs_diff']!r} "
            f"global_mean|Δ|={entry['global_mean_abs_diff']!r}"
        )
    if report["problems"]:
        print("  问题：")
        for problem in report["problems"]:
            print(f"    - {problem}")
    print(f"  mismatched_cases : {len(report['mismatched_cases'])} 条（详见报告 JSON）")
    print(f"  输入是否被修改     : {report['inputs_modified']}")
    print(f"  报告：{output_path}")
    print("=== 审计结束 ===")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AuditError as exc:
        print(f"[audit] FAILED: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
