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
（``0000``/``0001``/``0002`` = T2W/ADC/HBV）的 ``np.array_equal`` 与浮点差值统计，以及 seg 的
两层比较（语义必须区分，不得混用）：

- **原始 seg 逐值相同**（``raw_seg_equal``）：预处理 ``_seg.b2nd`` 的字节/数值完全一致，
  含两侧都含 ``-1`` 的情况；
- **有效标签逐值相同**（``effective_labels_equal``）：把 ``-1`` 按本项目实际训练/验证变换
  （固定版本 nnU-Net 的 ``RemoveLabelTansform(-1, 0)``，见 ``nnUNetTrainer.py:800,855``）
  映射为 0 后的标签逐值一致。

预处理 seg 的合法原始取值是 ``{-1, 0, 1}``：``-1`` 来自 ``crop_to_nonzero`` 的 nonzero 裁剪填充
（``cropping.py:19,36``），不是任务标签。纯 ``-1↔0`` 的原始差异记为**信息性**（不影响 PASS）；
``0↔1``、``-1↔1``、``{-1,0,1}`` 之外的取值、MRI 任一体素差异、形状/dtype/非有限值问题一律 FAIL。
检查顺序固定为：shape → dtype → 有限性 → 取值合法性 →（全部通过后）才做整数转换与逐值比较；
NaN/Inf/非法值产生结构化失败条目，绝不异常崩溃或被截断成合法值。

**PZ/TZ（Dataset606 的第 4/5 通道）不参与前三 MRI 通道的逐值相等判定，也不影响 status/PASS**；
只记录其取值范围与 ``contains_non_finite`` 作为信息性诊断。

fail-closed
-----------
以下任一情况都不判 PASS，且**绝不修改任何输入**（不修复、不重建、不重新预处理）：
病例集合不同、split 不同、有效标签不同、非法 seg 取值、MRI shape 不同、缺通道、
文件缺失/损坏、NaN/Inf、MRI 数值不一致、dtype 不同。

status 取值：``MRI_ARRAY_AUDIT_PASS`` / ``MRI_ARRAY_AUDIT_FAIL`` /
``MRI_ARRAY_AUDIT_PARTIAL``（用了 ``--max-cases`` 只跑了子集，永不判 PASS）。
退出码：PASS 0；FAIL / PARTIAL 2。

用法（长任务，由研究者本人运行）
-------------------------------
    cd /opt/data/private/lm/my-projects
    conda activate lm
    source scripts/env_nnunet.sh
    python scripts/data/audit_dataset605_606_mri_equivalence.py \
        --output outputs/reports/dataset605_606_mri_equivalence_audit_v2.json

成功判据：退出码 0、status=MRI_ARRAY_AUDIT_PASS、``n_cases_effective_input_mismatch=0``、
``effective_labels_equal=true``、``mri_exact_equal_all=true``、
``per_channel.*.exact_equal_cases = n_cases_checked``；
``n_cases_raw_label_difference`` 允许 > 0（信息性，纯 -1↔0 原始差异）。
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
DEFAULT_OUTPUT = "outputs/reports/dataset605_606_mri_equivalence_audit_v2.json"


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


def _load_json(path: Path):
    """读取 JSON 文件，**不限定顶层类型**：``splits_final.json`` 顶层是数组，``dataset.json`` 是对象。

    顶层类型由各自的比较函数校验（``compare_splits`` 要求非空 list，``compare_metadata`` 由
    调用方保证拿到 dict），这里只负责「存在、可解析」。
    """
    if not path.is_file():
        raise AuditError(f"缺少必需文件：{path}")
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:  # 统一转为 AuditError，避免静默继续
        raise AuditError(f"无法解析 {path}: {exc}") from exc


def _load_json_object(path: Path) -> dict:
    """读取并要求顶层为 JSON 对象（``dataset.json`` 这类）。"""
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise AuditError(
            f"{path} 的内容不是 JSON 对象（顶层类型 {type(payload).__name__}），"
            "无法作为 dataset.json 使用。"
        )
    return payload


def _labels_within_allowed(labels) -> bool:
    """dataset.json 声明的任务标签必须落在合法原始取值 {-1, 0, 1} 内。

    本项目的任务标签是 {background: 0, lesion: 1}；-1 是 nnU-Net nonzero 裁剪的填充值，
    不应出现在 dataset.json 的 labels 里。若声明了其他取值，说明配置与本审计约定冲突，
    必须 fail-closed。
    """
    if not isinstance(labels, dict):
        return False
    try:
        declared = {int(value) for value in labels.values()}
    except (TypeError, ValueError):
        return False
    return declared <= set(ALLOWED_RAW_SEG_LABELS)


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
        "labels_within_allowed_raw": _labels_within_allowed(labels_605)
        and _labels_within_allowed(labels_606),
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


#: 预处理 seg 中合法的原始取值 {-1, 0, 1}（固定版本 nnU-Net v2.6.2，只读核对）：
#: - ``-1`` 的来源：``preprocessing/cropping/cropping.py:19`` 定义
#:   ``crop_to_nonzero(..., nonzero_label=-1)``，``:36`` 执行
#:   ``seg[(seg == 0) & (~nonzero_mask)] = nonzero_label``，即把 nonzero 裁剪框外的体素写成 -1；
#:   ``preprocessing/preprocessors/default_preprocessor.py:66`` 以默认参数调用它。
#: - ``-1 → 0`` 的映射：本项目实际训练/验证变换在损失计算前执行
#:   ``training/nnUNetTrainer/nnUNetTrainer.py:800``（get_training_transforms）与
#:   ``:855``（get_validation_transforms）中的 ``RemoveLabelTansform(-1, 0)``。
ALLOWED_RAW_SEG_LABELS = (-1, 0, 1)
#: 进入损失计算前的有效标签映射（对应上述 RemoveLabelTansform(-1, 0)）
EFFECTIVE_LABEL_REMAP = {-1: 0}


def _format_values(values) -> list:
    """安全格式化取值集合：先经过有限性检查才能安全转 int，这里不假设、不截断。"""
    out = []
    for value in np.asarray(values).ravel():
        if isinstance(value, (int, np.integer)):
            out.append(int(value))
        else:
            out.append(float(value))  # 含 NaN/Inf 时如实写成 nan/inf，不截断
    return sorted(out, key=lambda v: (isinstance(v, float), v))


def _effective_labels(seg: np.ndarray) -> np.ndarray:
    """按项目实际训练/验证变换（``RemoveLabelTansform(-1, 0)``）映射后的有效标签。

    仅在 seg 已通过有限性与取值合法性检查之后调用；映射只把 -1 改为 0，其余值原样保留。
    """
    effective = seg
    for source_value, target_value in EFFECTIVE_LABEL_REMAP.items():
        effective = np.where(effective == source_value, target_value, effective)
    return effective


def compare_case(
    case_identifier: str,
    data_605: np.ndarray,
    seg_605: np.ndarray,
    data_606: np.ndarray,
    seg_606: np.ndarray,
) -> dict:
    """比较单个病例；返回结构化结果。

    分类（``classification``）：

    - ``raw_identical``：MRI 三通道与原始 seg **字节/数值完全相同**（含两侧都含 -1 的情况）；
    - ``raw_label_difference_only``：原始 seg 存在差异，但差异全部是 ``-1↔0`` 这类经
      ``RemoveLabelTansform(-1, 0)`` 后**有效标签逐值相同**的差别（信息性，不影响 PASS）；
    - ``effective_mismatch``：存在任何影响 PASS 的不一致（MRI 任一体素差异、有效标签差异、
      非法标签值、形状/dtype/非有限值等结构问题），明细在 ``mismatches``；
    - ``unchecked``：连结构比较都无法进行（如 data 维度不对），只记录、不做任何截断转换。

    只比较前三个 MRI 通道；Dataset606 的 PZ/TZ 只做信息性统计，不参与一致性判定。
    """
    mismatches: list[dict] = []
    channels: dict[str, dict] = {}
    seg_unchecked_reason: str | None = None

    if data_605.ndim != 4 or data_606.ndim != 4:
        mismatches.append(
            {
                "case_id": case_identifier,
                "channel": "ALL",
                "kind": "ndim_mismatch",
                "shape": [list(data_605.shape), list(data_606.shape)],
            }
        )
        seg_unchecked_reason = "ndim_mismatch"

    if seg_unchecked_reason is None:
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
        if not bool(np.isfinite(data_605).all()) or not bool(
            np.isfinite(data_606).all()
        ):
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
                    {
                        "max_abs_diff": 0.0,
                        "mean_abs_diff": 0.0,
                        "unequal_voxels": 0,
                        "fraction_unequal": 0.0,
                    }
                )
            else:
                diff = np.abs(
                    channel_a.astype(np.float64) - channel_b.astype(np.float64)
                )
                unequal = int(np.count_nonzero(diff))
                info.update(
                    {
                        "max_abs_diff": float(diff.max()) if diff.size else 0.0,
                        "mean_abs_diff": float(diff.mean()) if diff.size else 0.0,
                        "unequal_voxels": unequal,
                        "fraction_unequal": (
                            float(unequal) / float(diff.size) if diff.size else 0.0
                        ),
                    }
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
        # PZ/TZ 仅信息性：min/max 用 nan 感知函数，非有限值如实标记；不影响 status/PASS。
        finite = bool(np.isfinite(prior).all()) if prior.size else True
        prior_stats = {
            "n_channels": int(prior.shape[0]),
            "min": float(np.nanmin(prior)) if prior.size else None,
            "max": float(np.nanmax(prior)) if prior.size else None,
            "contains_non_finite": not finite,
        }

    # ---- seg：先结构（shape/dtype/有限性），再取值合法性，最后才做任何转换/比较 ----
    seg_info: dict = {
        "shape_605": list(seg_605.shape),
        "shape_606": list(seg_606.shape),
        "dtype_605": str(seg_605.dtype),
        "dtype_606": str(seg_606.dtype),
        "raw_values_605": None,
        "raw_values_606": None,
        "unexpected_values_605": None,
        "unexpected_values_606": None,
        "raw_seg_equal": None,
        "raw_unequal_voxels": None,
        "raw_diff_pairs": None,
        "effective_labels_equal": None,
        "effective_unequal_voxels": None,
        "foreground_equal": None,
        "foreground_unequal_voxels": None,
        "unchecked_reason": None,
    }
    raw_label_differences: list[dict] = []

    if seg_unchecked_reason is not None:
        seg_info["unchecked_reason"] = seg_unchecked_reason
    elif not bool(np.isfinite(seg_605).all()) or not bool(np.isfinite(seg_606).all()):
        seg_info["unchecked_reason"] = "non_finite_values"
        mismatches.append(
            {
                "case_id": case_identifier,
                "channel": "SEG",
                "kind": "seg_non_finite_values",
                # 非有限值存在时禁止任何 int 转换；只报告 dtype 与是否含 NaN/Inf
                "dtype": [str(seg_605.dtype), str(seg_606.dtype)],
                "contains_nan_605": bool(np.isnan(seg_605).any())
                if np.issubdtype(seg_605.dtype, np.floating)
                else False,
                "contains_nan_606": bool(np.isnan(seg_606).any())
                if np.issubdtype(seg_606.dtype, np.floating)
                else False,
            }
        )
    else:
        # 有限性已保证，可安全转换到 float64 做取值合法性判断（不丢失、不截断）
        seg_605_f = seg_605.astype(np.float64)
        seg_606_f = seg_606.astype(np.float64)
        allowed = np.asarray(ALLOWED_RAW_SEG_LABELS, dtype=np.float64)
        legal_605 = np.isin(seg_605_f, allowed)
        legal_606 = np.isin(seg_606_f, allowed)
        # 合法性在 float64 上判定（避免无符号/负数表示陷阱），但报告的取值从原始数组提取，
        # 使整数 dtype 的 seg 报告整数值而不是转换后的浮点。
        unexpected_605 = np.unique(seg_605[~legal_605])
        unexpected_606 = np.unique(seg_606[~legal_606])
        seg_info["raw_values_605"] = _format_values(np.unique(seg_605))
        seg_info["raw_values_606"] = _format_values(np.unique(seg_606))
        seg_info["unexpected_values_605"] = _format_values(unexpected_605)
        seg_info["unexpected_values_606"] = _format_values(unexpected_606)
        if unexpected_605.size or unexpected_606.size:
            seg_info["unchecked_reason"] = "unexpected_label_values"
            mismatches.append(
                {
                    "case_id": case_identifier,
                    "channel": "SEG",
                    "kind": "unexpected_label_values",
                    "allowed_raw_labels": list(ALLOWED_RAW_SEG_LABELS),
                    "unexpected_values_605": seg_info["unexpected_values_605"],
                    "unexpected_values_606": seg_info["unexpected_values_606"],
                }
            )

    values_comparable = (
        seg_info["unchecked_reason"] is None
        and tuple(seg_605.shape) == tuple(seg_606.shape)
        and seg_605.dtype == seg_606.dtype
    )
    if seg_info["unchecked_reason"] is None and tuple(seg_605.shape) != tuple(
        seg_606.shape
    ):
        seg_info["unchecked_reason"] = "label_shape_mismatch"
        mismatches.append(
            {
                "case_id": case_identifier,
                "channel": "SEG",
                "kind": "label_shape_mismatch",
                "shape": [list(seg_605.shape), list(seg_606.shape)],
            }
        )
    elif seg_info["unchecked_reason"] is None and seg_605.dtype != seg_606.dtype:
        seg_info["unchecked_reason"] = "label_dtype_mismatch"
        mismatches.append(
            {
                "case_id": case_identifier,
                "channel": "SEG",
                "kind": "label_dtype_mismatch",
                "dtype": [str(seg_605.dtype), str(seg_606.dtype)],
            }
        )

    if values_comparable:
        # 有限性 + 合法性 + shape/dtype 均已通过，这里才允许整数转换与逐值比较
        seg_605_i = seg_605.astype(np.int64)
        seg_606_i = seg_606.astype(np.int64)
        seg_info["raw_seg_equal"] = bool(np.array_equal(seg_605_i, seg_606_i))
        foreground_605 = seg_605_i == 1
        foreground_606 = seg_606_i == 1
        seg_info["foreground_equal"] = bool(
            np.array_equal(foreground_605, foreground_606)
        )
        seg_info["foreground_unequal_voxels"] = int(
            np.count_nonzero(foreground_605 != foreground_606)
        )
        effective_605 = _effective_labels(seg_605_i)
        effective_606 = _effective_labels(seg_606_i)
        seg_info["effective_labels_equal"] = bool(
            np.array_equal(effective_605, effective_606)
        )
        effective_diff = effective_605 != effective_606
        seg_info["effective_unequal_voxels"] = int(np.count_nonzero(effective_diff))

        if seg_info["raw_seg_equal"]:
            seg_info["raw_unequal_voxels"] = 0
            seg_info["raw_diff_pairs"] = {}
        else:
            raw_diff = seg_605_i != seg_606_i
            raw_unequal = int(np.count_nonzero(raw_diff))
            pairs: dict[str, int] = {}
            for source_value, target_value in zip(
                seg_605_i[raw_diff].tolist(), seg_606_i[raw_diff].tolist()
            ):
                key = f"{source_value}→{target_value}"
                pairs[key] = pairs.get(key, 0) + 1
            seg_info["raw_unequal_voxels"] = raw_unequal
            seg_info["raw_diff_pairs"] = dict(sorted(pairs.items()))
            if seg_info["effective_labels_equal"]:
                # 信息性：原始 seg 有差异，但经 RemoveLabelTansform(-1, 0) 后有效标签逐值相同
                raw_label_differences.append(
                    {
                        "case_id": case_identifier,
                        "kind": "raw_seg_difference_effectively_equal",
                        "unequal_voxels": raw_unequal,
                        "raw_diff_pairs": seg_info["raw_diff_pairs"],
                        "effective_labels_equal": True,
                        "foreground_equal": seg_info["foreground_equal"],
                    }
                )
            else:
                mismatches.append(
                    {
                        "case_id": case_identifier,
                        "channel": "SEG",
                        "kind": "effective_label_values_differ",
                        "unequal_voxels": seg_info["effective_unequal_voxels"],
                        "raw_unequal_voxels": raw_unequal,
                        "raw_diff_pairs": seg_info["raw_diff_pairs"],
                    }
                )

    if not mismatches and seg_info["raw_seg_equal"] is True:
        classification = "raw_identical"
    elif not mismatches:
        classification = "raw_label_difference_only"
    elif seg_info["unchecked_reason"] is not None and not values_comparable:
        classification = "unchecked"
    else:
        classification = "effective_mismatch"

    return {
        "case_id": case_identifier,
        "classification": classification,
        "effective_input_consistent": not mismatches,
        "raw_identical": bool(mismatches == [] and seg_info["raw_seg_equal"] is True),
        "mismatches": mismatches,
        "raw_label_differences": raw_label_differences,
        "channels": channels,
        "prior_channel_stats_606": prior_stats,
        "seg": seg_info,
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
    """执行完整审计并返回结构化报告（不写文件、不修改任何输入）。

    字段语义（区分「原始逐值相同」与「进入训练损失的有效标签相同」）：

    - ``n_cases_raw_identical``：MRI 三通道与原始 seg **逐值全部相同**的例数（含两侧都含 -1）；
    - ``n_cases_effective_equal``：MRI 逐值相同且**有效标签**逐值相同的例数（有效标签 =
      原始 seg 经 ``RemoveLabelTansform(-1, 0)`` 映射后，即 -1→0、0/1 不变）；
    - ``n_cases_effective_input_mismatch``：**FAIL 驱动** —— MRI 任一体素差异、有效标签差异
      （含 0↔1、-1↔1）、非法标签值（{-1,0,1} 之外）、形状/dtype/非有限值等结构问题；
    - ``n_cases_raw_label_difference``：**信息性** —— 原始 seg 有差异但有效标签逐值相同
      （即纯 ``-1↔0`` 差异），不影响 PASS；
    - ``effective_labels_equal``：全部已检查例的有效标签逐值相同（bool）；
    - ``mismatched_cases`` 只含 FAIL 驱动条目；信息性差异在 ``raw_label_differences``。
    """
    started = time.time()
    identifiers_605 = tuple(source_605.identifiers)
    identifiers_606 = tuple(source_606.identifiers)
    set_605, set_606 = set(identifiers_605), set(identifiers_606)

    per_channel = {name: _empty_channel_aggregate() for name in MRI_CHANNEL_NAMES}
    mismatched_cases: list[dict] = []
    raw_label_differences: list[dict] = []
    per_case_seg_summary: list[dict] = []
    prior_stats: list[dict] = []
    n_checked = 0
    n_raw_identical = 0
    n_effective_equal = 0
    n_foreground_equal = 0
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
        )
        n_checked += 1
        seg = result["seg"]
        if result["raw_identical"]:
            n_raw_identical += 1
        if result["effective_input_consistent"]:
            n_effective_equal += 1
        if seg.get("foreground_equal"):
            n_foreground_equal += 1
        per_case_seg_summary.append(
            {
                "case_id": case_identifier,
                "classification": result["classification"],
                "mri_exact_equal": all(
                    info["exact_equal"] for info in result["channels"].values()
                )
                if result["channels"]
                else None,
                "raw_seg_equal": seg.get("raw_seg_equal"),
                "raw_unequal_voxels": seg.get("raw_unequal_voxels"),
                "effective_labels_equal": seg.get("effective_labels_equal"),
                "foreground_equal": seg.get("foreground_equal"),
            }
        )
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
        raw_label_differences.extend(result["raw_label_differences"])

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
    n_effective_input_mismatch = n_checked - n_effective_equal
    n_raw_label_difference = len(raw_label_differences)
    mri_exact_equal_all = n_checked > 0 and all(
        entry["mri_exact_equal"] for entry in per_case_seg_summary
    )
    effective_labels_equal = (
        n_checked > 0
        and n_effective_equal == n_checked
        and not load_failures
        and not only_605
        and not only_606
    )
    problems: list[str] = []
    if not case_set_equal:
        problems.append(
            f"病例集合不一致：仅 605 有 {len(only_605)} 例，仅 606 有 {len(only_606)} 例"
        )
    if not split_equal:
        problems.append("splits_final.json 的 fold 划分不一致")
    if not metadata_report["labels_within_allowed_raw"]:
        problems.append(
            "dataset.json 的 labels 含 {-1, 0, 1} 之外的取值，与本审计的合法原始标签约定冲突"
        )
    if not metadata_report["labels_equal"]:
        problems.append("dataset.json 的 labels 定义不一致")
    if not metadata_report["mri_channel_names_equal"]:
        problems.append("前三个 MRI 通道名不是 T2W/ADC/HBV 或两数据集不一致")
    if load_failures:
        problems.append(f"{len(load_failures)} 例预处理文件缺失或读取失败")
    if n_effective_input_mismatch:
        problems.append(
            f"{n_effective_input_mismatch} 例存在影响有效输入一致性的问题"
            "（MRI 任一体素差异 / 有效标签差异（含 0↔1、-1↔1）/ 非法标签值 / 形状、dtype、"
            "非有限值等结构问题，逐例明细见 mismatched_cases）"
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
        "effective_labels_equal": bool(effective_labels_equal),
        "mri_exact_equal_all": bool(mri_exact_equal_all),
        "n_cases_checked": int(n_checked),
        "n_cases_raw_identical": int(n_raw_identical),
        "n_cases_effective_equal": int(n_effective_equal),
        "n_cases_effective_input_mismatch": int(n_effective_input_mismatch),
        "n_cases_raw_label_difference": int(n_raw_label_difference),
        "n_cases_foreground_equal": int(n_foreground_equal),
        "allowed_raw_seg_labels": list(ALLOWED_RAW_SEG_LABELS),
        "effective_label_remap": {
            str(source): target for source, target in EFFECTIVE_LABEL_REMAP.items()
        },
        "channels": list(MRI_CHANNEL_NAMES),
        "per_channel": per_channel,
        "mismatched_cases": mismatched_cases,
        "raw_label_differences": raw_label_differences,
        "per_case_seg_summary": per_case_seg_summary,
        "load_failures": load_failures,
        "metadata": metadata_report,
        "splits": splits_report,
        "prior_channels_606_info_only": {
            "note": (
                "PZ/TZ 不参与前三 MRI 通道的逐值相等判定，也不影响 status/PASS；"
                "以下范围统计与 contains_non_finite 仅为信息性诊断"
            ),
            "n_channel_entries": len(prior_stats),
            "global_min": min(
                (s["min"] for s in prior_stats if s["min"] is not None), default=None
            ),
            "global_max": max(
                (s["max"] for s in prior_stats if s["max"] is not None), default=None
            ),
            "contains_non_finite": any(
                s.get("contains_non_finite") for s in prior_stats
            ),
        },
        "inputs_modified": False,
        "notes": [
            "本工具只读；不修复数据、不重建 Dataset606、不重新 preprocessing。",
            "MRI 判定依据是 np.array_equal（最严格，任一体素不同即 FAIL）；浮点差值统计仅用于诊断。",
            (
                "seg 合法原始取值为 {-1, 0, 1}：-1 来自固定版本 nnU-Net 的 crop_to_nonzero"
                "（preprocessing/cropping/cropping.py:19,36），并在训练/验证变换"
                " RemoveLabelTansform(-1, 0)（training/nnUNetTrainer/nnUNetTrainer.py:800,855）中"
                "于损失计算前映射为 0。"
            ),
            (
                "原始逐值相同（raw_seg_equal）≠ 有效标签相同（effective_labels_equal）：纯 -1↔0 "
                "差异记为信息性（n_cases_raw_label_difference），不影响 PASS；0↔1、-1↔1、"
                "{-1,0,1} 之外的取值、MRI 任一体素差异、形状/dtype/非有限值问题一律 FAIL。"
            ),
            (
                f"PASS 定义：status={STATUS_PASS} 要求 完整检查全部共同病例（无 --max-cases 子集、"
                "无读取失败）、病例集合与 split 一致、dataset.json 合法且两数据集一致、"
                "前三个 MRI 通道逐数组完全相同、所有 seg 合法且有效标签逐值相同；"
                "原始 -1↔0 差异不计入失败。"
            ),
            "PZ/TZ 的一切检查（min/max、contains_non_finite）均为信息性，不影响 status/PASS。",
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
    dataset_json_605 = _load_json_object(dir_605 / "dataset.json")
    dataset_json_606 = _load_json_object(dir_606 / "dataset.json")

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
    print(f"  status                : {report['status']}")
    print(
        f"  病例集合一致           : {report['case_set_equal']}"
        f"（605={report['n_cases_605']} / 606={report['n_cases_606']}）"
    )
    print(f"  split 一致             : {report['split_equal']}")
    print(
        f"  有效标签一致           : {report['effective_labels_equal']}"
        "（-1 按训练/验证变换 RemoveLabelTansform(-1, 0) 映射为 0 后逐值比较）"
    )
    print(
        f"  已检查例数             : {report['n_cases_checked']}"
        f"（MRI+原始seg 逐值全同 {report['n_cases_raw_identical']} / "
        f"有效输入一致 {report['n_cases_effective_equal']}）"
    )
    print(
        f"  有效输入不一致例数（FAIL 驱动） : {report['n_cases_effective_input_mismatch']}"
    )
    print(
        f"  原始 seg 差异例数（信息性，纯 -1↔0 不影响 PASS） : "
        f"{report['n_cases_raw_label_difference']}"
    )
    print(f"  病灶前景 (seg==1) 逐值相同例数   : {report['n_cases_foreground_equal']}")
    for name in report["channels"]:
        entry = report["per_channel"][name]
        print(
            f"  {name:>3s}: exact={entry['exact_equal_cases']} mismatch={entry['mismatch_cases']} "
            f"global_max|Δ|={entry['global_max_abs_diff']!r} "
            f"global_mean|Δ|={entry['global_mean_abs_diff']!r}"
        )
    prior = report["prior_channels_606_info_only"]
    print(
        f"  PZ/TZ（仅信息性，不影响 PASS）  : min={prior['global_min']!r} "
        f"max={prior['global_max']!r} contains_non_finite={prior['contains_non_finite']}"
    )
    if report["problems"]:
        print("  问题：")
        for problem in report["problems"]:
            print(f"    - {problem}")
    print(
        f"  mismatched_cases（FAIL 驱动）    : {len(report['mismatched_cases'])} 条"
        f"（信息性 raw_label_differences {len(report['raw_label_differences'])} 条，详见报告 JSON）"
    )
    print(f"  输入是否被修改         : {report['inputs_modified']}")
    print(f"  报告：{output_path}")
    print("=== 审计结束 ===")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AuditError as exc:
        print(f"[audit] FAILED: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
