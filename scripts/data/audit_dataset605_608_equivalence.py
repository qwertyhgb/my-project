#!/usr/bin/env python3
"""Dataset605_PICAI ↔ Dataset608_PICAI_PredictedAnatomy 一致性审计（只读、fail-closed）。

它回答的唯一问题
----------------
> 加入 predicted anatomy 先验后，原来的 MRI / lesion / split 是否**保持不变**？

因此它是未来 ``C vs D`` 解释「D 只增加了 predicted anatomy context」这一单变量边界的
**前置条件**。它**不**评价 anatomy prior 本身好不好（那属于 Stage-1 ``anatomy/validation.py``
的 soft-head 诊断，见 ``docs/experiments/anatomy_joint_100ep.md``）；职责不得混合。
prior 通道的数值范围只作**信息性**记录，不参与 PASS/FAIL。

审计层级
--------
1. **raw 层（始终执行）**：``nnUNet_raw/Dataset605_PICAI`` ↔
   ``nnUNet_raw/Dataset608_PICAI_PredictedAnatomy``：病例集合与顺序、6 通道结构、
   ``splits_final.json``（逐 fold train/val 集合与顺序）、``dataset.json``（前三维必须是
   T2W/ADC/HBV、标签一致、``numTraining``、``file_ending``、608 prior 契约），
   逐病例 T2W/ADC/HBV 的 ``size`` / ``spacing`` / ``origin`` / ``direction`` / shape / dtype /
   有限性 / ``np.array_equal`` / ``max |Δ|`` / affected voxels，以及 lesion label 的几何、
   合法取值、原始逐值相同与**有效标签**相同。
2. **preprocessed 层（存在则执行）**：``nnUNetPlans_3d_fullres`` 的 ``.b2nd`` / ``_seg.b2nd``
   前三个 MRI 通道与 seg；额外比较两边 ``nnUNetPlans.json`` 的 architecture / patch / batch /
   spacing / preprocessor。

「原始逐值相同」与「进入训练损失的有效标签相同」是两个不同的判定（与
``scripts/data/audit_dataset605_606_mri_equivalence.py`` 完全同一套语义；常量与有效标签映射
由 ``tests/unit/test_dataset605_608_mri_equivalence_audit.py`` 做防漂移核对）：合法原始 seg
取值为 ``{-1, 0, 1}``（``-1`` 来自 nnU-Net ``crop_to_nonzero`` 的裁剪填充，经
``RemoveLabelTansform(-1, 0)`` 在损失前映射为 0），纯 ``-1↔0`` 差异记为信息性；
``0↔1``、``-1↔1``、其余取值、MRI 任一体素差异、形状/dtype/非有限值/几何不一致一律 FAIL。

fail-closed：缺病例、缺文件、split 不同、数值/标签/几何不一致、prior 契约非法（含患者泄漏）、
读取异常都**不判 PASS**，且**绝不修改任何输入**（不修复、不重建、不重新预处理）。本工具不创建
Dataset608、不运行 preprocessing、不生成任何 prior。

status：``..._EQUIVALENCE_PASS``（raw + preprocessed 两层全通过，退出码 0）/
``..._RAW_ONLY_PASS``（raw 通过但 608 预处理产物尚不存在，或显式 ``--skip-preprocessed``
→ 尚不足以支撑 ``C vs D`` 单变量解释）/ ``..._EQUIVALENCE_PARTIAL``（``--max-cases`` 子集，
永不判 PASS）/ ``..._EQUIVALENCE_FAIL``；除 PASS 外一律退出码 2。
只要发现**任何**不一致即 ``FAIL``（优先于 PARTIAL/RAW_ONLY）。

用法（长任务，由研究者本人运行；需先 ``export PYTHONPATH="$PWD/src:$PYTHONPATH"``）::

    cd /opt/data/private/lm/my-projects
    conda activate lm
    source scripts/env_nnunet.sh
    export PYTHONPATH="$PWD/src:$PYTHONPATH"
    python scripts/data/audit_dataset605_608_equivalence.py \
        --output outputs/reports/dataset605_608_equivalence_audit.json

成功判据：退出码 0、``n_cases_mri_mismatch=0``、``n_cases_label_mismatch=0``、
``case_set_equal=true``、``split_equal=true``、``metadata_ok=true``、
``prior_contract_status=VALID``；``n_cases_prior_*_info_only`` 允许 > 0。

**本工具尚未在真实 Dataset608 上运行过**：Dataset608 目前 CODE READY / NOT MATERIALIZED。
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

# --------------------------------------------------------------------------- 常量

#: 前三个 MRI 通道的名字与索引（Dataset605/608 的前三通道顺序固定）
MRI_CHANNEL_NAMES = ("T2W", "ADC", "HBV")
MRI_CHANNELS = 3
#: Dataset608 在三个 MRI 之后附加的 predicted anatomy 通道数（不参与 MRI 判等）
PRIOR_CHANNELS_PREDICTED = 3
#: 两个数据集在 raw 层的通道数
CHANNELS_605_RAW = MRI_CHANNELS
CHANNELS_608_RAW = MRI_CHANNELS + PRIOR_CHANNELS_PREDICTED

#: 物理几何比较容差（size 用整数精确比较；spacing/origin/direction 用该容差）
GEOMETRY_ATOL = 1e-6

#: seg 中合法的取值 {-1, 0, 1}（-1 来自 nnU-Net ``crop_to_nonzero`` 的裁剪填充）
ALLOWED_RAW_SEG_LABELS = (-1, 0, 1)
#: 进入损失计算前的有效标签映射（对应 ``RemoveLabelTansform(-1, 0)``）
EFFECTIVE_LABEL_REMAP = {-1: 0}

#: 608 prior 通道的期望语义（channel_names 是 noNorm ×3，语义由 contract 声明）
EXPECTED_PRIOR_CHANNEL_ORDER = ["WG", "PZ", "TZ"]
EXPECTED_MRI_CHANNEL_NAMES = ["T2W", "ADC", "HBV"]
EXPECTED_LABELS = {"background": 0, "lesion": 1}

STATUS_PASS = "DATASET605_608_EQUIVALENCE_PASS"
STATUS_RAW_ONLY = "DATASET605_608_RAW_ONLY_PASS"
STATUS_PARTIAL = "DATASET605_608_EQUIVALENCE_PARTIAL"
STATUS_FAIL = "DATASET605_608_EQUIVALENCE_FAIL"

LEVEL_PASS = "PASS"
LEVEL_FAIL = "FAIL"
LEVEL_NOT_CHECKED = "NOT_CHECKED"

DEFAULT_DATASET_605 = "Dataset605_PICAI"
DEFAULT_DATASET_608 = "Dataset608_PICAI_PredictedAnatomy"
DEFAULT_CONFIGURATION = "3d_fullres"
DEFAULT_OUTPUT = "outputs/reports/dataset605_608_equivalence_audit.json"


class AuditError(RuntimeError):
    """审计前置条件不满足（环境、路径、元数据缺失等），fail-closed。"""


# --------------------------------------------------------------------------- 数据源


class RawCaseSource:
    """nnUNet_raw 数据集的只读读取器（通道文件枚举 + NIfTI 读取）。

    只通过文件名约定定位病例与通道（``<case>_<channel:04d>.nii.gz`` / ``<case>.nii.gz``），
    不猜测、不修补、不创建任何文件。
    """

    def __init__(self, dataset_dir: Path, *, expected_channels: int) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.images_folder = self.dataset_dir / "imagesTr"
        self.labels_folder = self.dataset_dir / "labelsTr"
        self.expected_channels = int(expected_channels)
        if not self.images_folder.is_dir() or not self.labels_folder.is_dir():
            raise AuditError(
                f"raw 目录结构不完整（需要 imagesTr/ 与 labelsTr/）：{self.dataset_dir}。"
                "请确认 nnUNet_raw 指向项目 workdir/nnUNet_raw。"
            )

    @property
    def identifiers(self) -> tuple[str, ...]:
        """病例集合：由 ``*_0000.nii.gz`` 派生（确定性排序，与文件系统枚举顺序无关）。"""
        found: set[str] = set()
        for path in self.images_folder.glob("*_0000.nii.gz"):
            found.add(_case_id_from_channel_file(path))
        return tuple(sorted(found))

    def channel_files(self, case_identifier: str) -> list[Path]:
        return [
            self.images_folder / f"{case_identifier}_{index:04d}.nii.gz"
            for index in range(self.expected_channels)
        ]

    def label_file(self, case_identifier: str) -> Path:
        return self.labels_folder / f"{case_identifier}.nii.gz"

    def missing_files(self, case_identifier: str) -> list[str]:
        names = [
            f"imagesTr/{path.name}"
            for path in self.channel_files(case_identifier)
            if not path.is_file()
        ]
        label = self.label_file(case_identifier)
        if not label.is_file():
            names.append(f"labelsTr/{label.name}")
        return names

    def read_image(self, path: Path):
        import SimpleITK as sitk  # 延迟导入：--help 与纯元数据检查不需要 ITK

        if not path.is_file():
            raise AuditError(f"缺少影像文件：{path}")
        try:
            return sitk.ReadImage(str(path))
        except Exception as exc:  # 读失败必须 fail-closed，不能静默当作通过
            raise AuditError(f"无法读取 {path}: {exc}") from exc


def _case_id_from_channel_file(path: Path) -> str:
    stem = path.name[: -len(".nii.gz")]
    case_identifier, separator, suffix = stem.rpartition("_")
    if not separator or not suffix.isdigit():
        raise AuditError(
            f"影像文件名不符合 nnU-Net 通道约定（<case>_<channel:04d>.nii.gz）：{path.name}"
        )
    return case_identifier


class PreprocessedCaseSource:
    """某数据集某配置下预处理病例的读取器（复用 nnU-Net 自己的 dataset 类）。

    刻意通过 ``infer_dataset_class`` + ``load_case`` 读取，使审计看到的数组与训练时**完全相同**；
    不自行解析 ``.b2nd``。``nnunetv2`` 延迟导入，便于单元测试使用同接口的合成数据源。
    """

    def __init__(self, dataset_dir: Path, configuration: str) -> None:
        from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class

        self.dataset_dir = Path(dataset_dir)
        self.configuration = str(configuration)
        self.case_folder = self.dataset_dir / f"nnUNetPlans_{self.configuration}"
        if not self.case_folder.is_dir():
            raise AuditError(f"预处理目录不存在：{self.case_folder}")
        dataset_class = infer_dataset_class(str(self.case_folder))
        self._dataset = dataset_class(str(self.case_folder))

    @property
    def identifiers(self) -> tuple[str, ...]:
        return tuple(self._dataset.identifiers)

    def load(self, case_identifier: str) -> tuple[np.ndarray, np.ndarray]:
        data, seg, _seg_prev, _properties = self._dataset.load_case(case_identifier)
        return np.asarray(data), np.asarray(seg)


def missing_preprocessed_files(case_folder: Path, case_identifier: str) -> list[str]:
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
    """读取 JSON，**不限定顶层类型**（``splits_final.json`` 是数组、``dataset.json`` 是对象）。"""
    if not path.is_file():
        raise AuditError(f"缺少必需文件：{path}")
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:
        raise AuditError(f"无法解析 {path}: {exc}") from exc


def _load_json_object(path: Path) -> dict:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise AuditError(
            f"{path} 的内容不是 JSON 对象（顶层类型 {type(payload).__name__}），"
            "无法作为 dataset.json 使用。"
        )
    return payload


def _ordered_channel_names(channels: dict) -> list[str]:
    """按通道序号排序后返回通道名；键格式（``"0"`` / ``"0000"``）差异不得误判为不一致。"""

    def sort_key(item):
        key = str(item[0])
        try:
            return (0, int(key))
        except ValueError:
            return (1, key)

    return [str(value) for _, value in sorted(channels.items(), key=sort_key)]


def compare_metadata(
    dataset_json_605: dict,
    dataset_json_608: dict,
    *,
    dataset_name_605: str,
    dataset_name_608: str,
) -> dict:
    """比较 dataset.json 的通道与标签定义（元数据层面，不是数组比较）。"""
    channels_605 = dataset_json_605.get("channel_names")
    channels_608 = dataset_json_608.get("channel_names")
    if not isinstance(channels_605, dict) or not isinstance(channels_608, dict):
        raise AuditError("dataset.json 缺少 channel_names 字典")
    ordered_605 = _ordered_channel_names(channels_605)
    ordered_608 = _ordered_channel_names(channels_608)
    labels_605 = dataset_json_605.get("labels")
    labels_608 = dataset_json_608.get("labels")
    return {
        "dataset_605": dataset_name_605,
        "dataset_608": dataset_name_608,
        "channel_names_605": {str(k): str(v) for k, v in channels_605.items()},
        "channel_names_608": {str(k): str(v) for k, v in channels_608.items()},
        "ordered_channel_names_605": ordered_605,
        "ordered_channel_names_608": ordered_608,
        "n_channels_605": len(channels_605),
        "n_channels_608": len(channels_608),
        "mri_channel_names_equal": ordered_605
        == ordered_608[:MRI_CHANNELS]
        == EXPECTED_MRI_CHANNEL_NAMES,
        "mri_channel_count_equal": len(channels_605) == CHANNELS_605_RAW,
        "prior_channel_slots_608": ordered_608[MRI_CHANNELS:],
        "labels_equal": labels_605 == labels_608 == EXPECTED_LABELS,
        "labels_605": labels_605,
        "labels_608": labels_608,
        "file_ending_equal": dataset_json_605.get("file_ending")
        == dataset_json_608.get("file_ending"),
        "num_training_equal": dataset_json_605.get("numTraining")
        == dataset_json_608.get("numTraining"),
    }


def validate_prior_contract(dataset_json_608: dict, dataset_name_608: str) -> str:
    """复用项目 608 契约校验；返回 ``VALID`` 或 ``INVALID: <reason>``。

    prior 契约非法（旧 Dataset606、非 predicted 来源、病例集合不完整、患者泄漏、
    validation 使用 in-sample prior）时返回 INVALID 并作为 FAIL 驱动。它校验的是
    **provenance，不是 anatomy 质量**。
    """
    try:
        from zonal_reliability_fusion.anatomy.dataset import validate_predicted_prior_dataset
    except ImportError as exc:  # pragma: no cover - 环境未按 README 设置 PYTHONPATH
        raise AuditError(
            "无法导入项目包 zonal_reliability_fusion；请先执行 "
            'export PYTHONPATH="$PWD/src:$PYTHONPATH"'
        ) from exc
    try:
        validate_predicted_prior_dataset(dataset_json_608, dataset_name_608)
    except Exception as exc:
        return f"INVALID: {type(exc).__name__}: {exc}"
    contract = dataset_json_608.get("predicted_anatomy_contract", {})
    if contract.get("channel_order") != EXPECTED_PRIOR_CHANNEL_ORDER:
        return "INVALID: prior channel order differs from WG/PZ/TZ"
    return "VALID"


def compare_splits(splits_605: list, splits_608: list, *, fold: int) -> dict:
    """比较两个数据集的 ``splits_final.json``（fold 数、逐 fold 的 train/val 集合与顺序）。"""
    for tag, splits in (("605", splits_605), ("608", splits_608)):
        if not isinstance(splits, list) or not splits:
            raise AuditError(f"splits_final.json({tag}) 不是非空列表")

    folds: list[dict] = []
    for index in range(max(len(splits_605), len(splits_608))):
        in_605 = index < len(splits_605)
        in_608 = index < len(splits_608)
        entry: dict = {"fold": index, "in_605": in_605, "in_608": in_608}
        if in_605 and in_608:
            train_605, train_608 = splits_605[index]["train"], splits_608[index]["train"]
            val_605, val_608 = splits_605[index]["val"], splits_608[index]["val"]
            entry.update(
                {
                    "n_train_605": len(train_605),
                    "n_train_608": len(train_608),
                    "n_val_605": len(val_605),
                    "n_val_608": len(val_608),
                    "train_set_equal": set(train_605) == set(train_608),
                    "train_order_equal": list(train_605) == list(train_608),
                    "val_set_equal": set(val_605) == set(val_608),
                    "val_order_equal": list(val_605) == list(val_608),
                    "train_val_overlap_605": len(set(train_605) & set(val_605)),
                    "train_val_overlap_608": len(set(train_608) & set(val_608)),
                }
            )
        folds.append(entry)

    if fold < 0 or fold >= len(splits_605) or fold >= len(splits_608):
        raise AuditError(
            f"fold {fold} 不存在（605 有 {len(splits_605)} 个 fold，608 有 {len(splits_608)} 个）"
        )

    n_folds_equal = len(splits_605) == len(splits_608)
    all_entries_equal = all(
        bool(entry.get("train_set_equal")) and bool(entry.get("val_set_equal"))
        for entry in folds
        if entry["in_605"] and entry["in_608"]
    )
    return {
        "n_folds_605": len(splits_605),
        "n_folds_608": len(splits_608),
        "n_folds_equal": n_folds_equal,
        "fold": fold,
        "fold_0": folds[fold] if fold < len(folds) else None,
        "per_fold": folds,
        "all_folds_train_val_set_equal": bool(n_folds_equal and all_entries_equal),
    }


# --------------------------------------------------------------------------- 数组/几何比较


def _format_values(values) -> list:
    """安全格式化取值集合：先经过有限性检查才能安全转 int，不截断、不哄骗。"""
    out = []
    for value in np.asarray(values).ravel():
        if isinstance(value, (int, np.integer)):
            out.append(int(value))
        else:
            out.append(float(value))
    return sorted(out, key=lambda v: (isinstance(v, float), v))


def _effective_labels(seg: np.ndarray) -> np.ndarray:
    """按 ``RemoveLabelTansform(-1, 0)`` 映射后的有效标签（仅在合法性检查通过后调用）。"""
    effective = seg
    for source_value, target_value in EFFECTIVE_LABEL_REMAP.items():
        effective = np.where(effective == source_value, target_value, effective)
    return effective


def _geometry(image) -> dict:
    return {
        "size": [int(v) for v in image.GetSize()],
        "spacing": [float(v) for v in image.GetSpacing()],
        "origin": [float(v) for v in image.GetOrigin()],
        "direction": [float(v) for v in image.GetDirection()],
    }


def _geometry_differences(geometry_605: dict, geometry_608: dict) -> dict:
    return {
        "size_equal": geometry_605["size"] == geometry_608["size"],
        "spacing_equal": bool(
            np.allclose(geometry_605["spacing"], geometry_608["spacing"], atol=GEOMETRY_ATOL, rtol=0)
        ),
        "origin_equal": bool(
            np.allclose(geometry_605["origin"], geometry_608["origin"], atol=GEOMETRY_ATOL, rtol=0)
        ),
        "direction_equal": bool(
            np.allclose(
                geometry_605["direction"], geometry_608["direction"], atol=GEOMETRY_ATOL, rtol=0
            )
        ),
        "max_abs_spacing_delta": float(
            np.max(np.abs(np.asarray(geometry_605["spacing"]) - np.asarray(geometry_608["spacing"])))
        ),
        "max_abs_origin_delta": float(
            np.max(np.abs(np.asarray(geometry_605["origin"]) - np.asarray(geometry_608["origin"])))
        ),
        "max_abs_direction_delta": float(
            np.max(
                np.abs(np.asarray(geometry_605["direction"]) - np.asarray(geometry_608["direction"]))
            )
        ),
    }


def compare_array_pair(array_605: np.ndarray, array_608: np.ndarray) -> dict:
    """比较一对数组：shape、dtype、有限性、逐值相等、``max |Δ|`` 与 affected voxels。

    非有限值或形状/dtype 不一致时**不做**逐值比较（``comparable=False``），避免把结构问题
    静默折成数值差异。
    """
    info: dict = {
        "shape_605": list(array_605.shape),
        "shape_608": list(array_608.shape),
        "dtype_605": str(array_605.dtype),
        "dtype_608": str(array_608.dtype),
        "total_voxels": int(array_605.size),
        "exact_equal": None,
        "max_abs_diff": None,
        "mean_abs_diff": None,
        "unequal_voxels": None,
        "fraction_unequal": None,
        "contains_non_finite_605": not bool(np.isfinite(array_605).all()),
        "contains_non_finite_608": not bool(np.isfinite(array_608).all()),
        "comparable": False,
    }
    if info["contains_non_finite_605"] or info["contains_non_finite_608"]:
        return info
    if info["shape_605"] != info["shape_608"] or info["dtype_605"] != info["dtype_608"]:
        return info
    info["comparable"] = True
    exact = bool(np.array_equal(array_605, array_608))
    info["exact_equal"] = exact
    if exact:
        info.update(
            {"max_abs_diff": 0.0, "mean_abs_diff": 0.0, "unequal_voxels": 0, "fraction_unequal": 0.0}
        )
        return info
    diff = np.abs(array_605.astype(np.float64) - array_608.astype(np.float64))
    unequal = int(np.count_nonzero(diff))
    info.update(
        {
            "max_abs_diff": float(diff.max()) if diff.size else 0.0,
            "mean_abs_diff": float(diff.mean()) if diff.size else 0.0,
            "unequal_voxels": unequal,
            "fraction_unequal": (float(unequal) / float(diff.size)) if diff.size else 0.0,
        }
    )
    return info


# --------------------------------------------------------------------------- 逐例比较


def _empty_case_record(case_identifier: str) -> dict:
    return {
        "case_id": case_identifier,
        "comparable": False,
        "effective_input_consistent": False,
        "raw_identical": None,
        "classification": "unchecked",
        "channels": {},
        "label": None,
        "prior_channels_info_only": None,
        "mismatches": [],
        "informational_differences": [],
    }


def compare_raw_case(
    case_identifier: str,
    *,
    source_605: RawCaseSource,
    source_608: RawCaseSource,
) -> dict:
    """raw 层单病例比较：MRI 三通道（含物理几何）+ lesion label + prior 通道信息性统计。"""
    record = _empty_case_record(case_identifier)
    for index, name in enumerate(MRI_CHANNEL_NAMES):
        image_605 = source_605.read_image(source_605.channel_files(case_identifier)[index])
        image_608 = source_608.read_image(source_608.channel_files(case_identifier)[index])
        record["channels"][name] = _compare_raw_channel(name, image_605, image_608, record)

    record["prior_channels_info_only"] = _compare_prior_channels(source_608, case_identifier)

    label_605 = source_605.read_image(source_605.label_file(case_identifier))
    label_608 = source_608.read_image(source_608.label_file(case_identifier))
    record["label"] = _compare_label_images(label_605, label_608, record, level="raw")
    _finalize(record)
    return record


def compare_preprocessed_case(
    case_identifier: str,
    *,
    source_605: PreprocessedCaseSource,
    source_608: PreprocessedCaseSource,
) -> dict:
    """preprocessed 层单病例比较：前三个 MRI 通道 + seg（几何由预处理链保证，此处不重复）。"""
    record = _empty_case_record(case_identifier)
    data_605, seg_605 = source_605.load(case_identifier)
    data_608, seg_608 = source_608.load(case_identifier)
    if data_605.ndim != 4 or data_608.ndim != 4:
        record["mismatches"].append(
            {"case_id": case_identifier, "channel": "ALL", "kind": "ndim_mismatch",
             "level": "preprocessed", "shape": [list(data_605.shape), list(data_608.shape)]}
        )
        _finalize(record)
        return record
    if data_605.shape[0] < MRI_CHANNELS or data_608.shape[0] < MRI_CHANNELS:
        record["mismatches"].append(
            {"case_id": case_identifier, "channel": "ALL", "kind": "missing_mri_channels",
             "level": "preprocessed", "shape": [list(data_605.shape), list(data_608.shape)]}
        )
    if tuple(data_605.shape[2:]) != tuple(data_608.shape[2:]):
        record["mismatches"].append(
            {"case_id": case_identifier, "channel": "ALL", "kind": "spatial_shape_mismatch",
             "level": "preprocessed", "shape": [list(data_605.shape), list(data_608.shape)]}
        )
    if data_605.dtype != data_608.dtype:
        record["mismatches"].append(
            {"case_id": case_identifier, "channel": "ALL", "kind": "dtype_mismatch",
             "level": "preprocessed", "dtype": [str(data_605.dtype), str(data_608.dtype)]}
        )
    for index, name in enumerate(MRI_CHANNEL_NAMES):
        if index >= data_605.shape[0] or index >= data_608.shape[0]:
            continue
        info = compare_array_pair(data_605[index], data_608[index])
        record["channels"][name] = info
        _record_mri_mismatch(info, record, name, level="preprocessed")
    record["label"] = _compare_label_arrays(seg_605, seg_608, record, level="preprocessed")
    _finalize(record)
    return record


def _compare_raw_channel(name, image_605, image_608, record: dict) -> dict:
    import SimpleITK as sitk

    geometry_605, geometry_608 = _geometry(image_605), _geometry(image_608)
    info = compare_array_pair(
        sitk.GetArrayFromImage(image_605), sitk.GetArrayFromImage(image_608)
    )
    differences = _geometry_differences(geometry_605, geometry_608)
    info["geometry_605"] = geometry_605
    info["geometry_608"] = geometry_608
    info["geometry_equality"] = differences
    for key in ("size_equal", "spacing_equal", "origin_equal", "direction_equal"):
        if not differences[key]:
            record["mismatches"].append({
                "case_id": record["case_id"], "channel": name, "kind": f"geometry_{key}",
                "level": "raw", "geometry_605": geometry_605, "geometry_608": geometry_608,
                "differences": differences,
            })
    _record_mri_mismatch(info, record, name, level="raw")
    return info


def _record_mri_mismatch(info: dict, record: dict, name: str, *, level: str) -> None:
    if not info["comparable"]:
        non_finite = info["contains_non_finite_605"] or info["contains_non_finite_608"]
        record["mismatches"].append({
            "case_id": record["case_id"], "channel": name, "level": level,
            "kind": "non_finite_values" if non_finite else "shape_or_dtype_mismatch",
            "shape": [info["shape_605"], info["shape_608"]],
            "dtype": [info["dtype_605"], info["dtype_608"]],
            "contains_non_finite_605": info["contains_non_finite_605"],
            "contains_non_finite_608": info["contains_non_finite_608"],
        })
        return
    if not info["exact_equal"]:
        record["mismatches"].append({
            "case_id": record["case_id"], "channel": name, "kind": "mri_values_differ",
            "level": level, "max_abs_diff": info["max_abs_diff"],
            "mean_abs_diff": info["mean_abs_diff"], "unequal_voxels": info["unequal_voxels"],
            "fraction_unequal": info["fraction_unequal"], "total_voxels": info["total_voxels"],
        })


def _compare_prior_channels(source_608: RawCaseSource, case_identifier: str) -> dict:
    """608 的 prior 通道：**仅信息性**统计（范围、有限性、是否全零），不影响 PASS/FAIL。"""
    import SimpleITK as sitk

    entries = []
    for offset in range(PRIOR_CHANNELS_PREDICTED):
        index = MRI_CHANNELS + offset
        array = sitk.GetArrayFromImage(source_608.read_image(source_608.channel_files(case_identifier)[index]))
        finite = bool(np.isfinite(array).all())
        entries.append({
            "channel_index": index,
            "name": EXPECTED_PRIOR_CHANNEL_ORDER[offset],
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "min": float(np.nanmin(array)) if array.size else None,
            "max": float(np.nanmax(array)) if array.size else None,
            "contains_non_finite": not finite,
            "outside_unit_interval": bool(
                finite and array.size and ((array < 0).any() or (array > 1).any())
            ),
            "all_zero": bool(finite and array.size and not array.any()),
        })
    return {
        "note": "prior 质量与合法性只在 Stage-1 诊断中评价；此处仅记录数值范围（信息性）",
        "channels": entries,
        "any_outside_unit_interval": any(e["outside_unit_interval"] for e in entries),
        "any_non_finite": any(e["contains_non_finite"] for e in entries),
        "any_all_zero": any(e["all_zero"] for e in entries),
    }


def _compare_label_images(image_605, image_608, record: dict, *, level: str) -> dict:
    import SimpleITK as sitk

    geometry_605, geometry_608 = _geometry(image_605), _geometry(image_608)
    differences = _geometry_differences(geometry_605, geometry_608)
    for key in ("size_equal", "spacing_equal", "origin_equal", "direction_equal"):
        if not differences[key]:
            record["mismatches"].append({
                "case_id": record["case_id"], "channel": "SEG", "kind": f"label_geometry_{key}",
                "level": level, "geometry_605": geometry_605, "geometry_608": geometry_608,
                "differences": differences,
            })
    result = _compare_label_arrays(
        sitk.GetArrayFromImage(image_605), sitk.GetArrayFromImage(image_608), record, level=level
    )
    result["geometry_605"] = geometry_605
    result["geometry_608"] = geometry_608
    result["geometry_equality"] = differences
    return result


def _compare_label_arrays(
    seg_605: np.ndarray, seg_608: np.ndarray, record: dict, *, level: str
) -> dict:
    """seg 两层比较；与 605/606 审计同一套语义（原始差异 vs 有效标签差异）。

    检查顺序固定为 shape → dtype → 有限性 → 取值合法性 →（全部通过后）逐值比较，
    绝不把非法值截断成合法值、也绝不把结构问题静默折成数值差异。
    """
    result: dict = {
        "raw_values_605": None, "raw_values_608": None,
        "unexpected_values_605": None, "unexpected_values_608": None,
        "raw_seg_equal": None, "raw_unequal_voxels": None, "raw_diff_pairs": None,
        "effective_labels_equal": None, "effective_unequal_voxels": None,
        "foreground_equal": None, "foreground_unequal_voxels": None,
        "unchecked_reason": None,
    }
    if tuple(seg_605.shape) != tuple(seg_608.shape):
        result["unchecked_reason"] = "label_shape_mismatch"
        record["mismatches"].append({
            "case_id": record["case_id"], "channel": "SEG", "kind": "label_shape_mismatch",
            "level": level, "shape": [list(seg_605.shape), list(seg_608.shape)],
        })
        return result
    if seg_605.dtype != seg_608.dtype:
        result["unchecked_reason"] = "label_dtype_mismatch"
        record["mismatches"].append({
            "case_id": record["case_id"], "channel": "SEG", "kind": "label_dtype_mismatch",
            "level": level, "dtype": [str(seg_605.dtype), str(seg_608.dtype)],
        })
        return result
    if not bool(np.isfinite(seg_605).all()) or not bool(np.isfinite(seg_608).all()):
        result["unchecked_reason"] = "non_finite_values"
        record["mismatches"].append({
            "case_id": record["case_id"], "channel": "SEG", "kind": "seg_non_finite_values",
            "level": level,
        })
        return result

    seg_605_f, seg_608_f = seg_605.astype(np.float64), seg_608.astype(np.float64)
    allowed = np.asarray(ALLOWED_RAW_SEG_LABELS, dtype=np.float64)
    legal_605, legal_608 = np.isin(seg_605_f, allowed), np.isin(seg_608_f, allowed)
    result["raw_values_605"] = _format_values(np.unique(seg_605))
    result["raw_values_608"] = _format_values(np.unique(seg_608))
    result["unexpected_values_605"] = _format_values(np.unique(seg_605[~legal_605]))
    result["unexpected_values_608"] = _format_values(np.unique(seg_608[~legal_608]))
    if result["unexpected_values_605"] or result["unexpected_values_608"]:
        result["unchecked_reason"] = "unexpected_label_values"
        record["mismatches"].append({
            "case_id": record["case_id"], "channel": "SEG", "kind": "unexpected_label_values",
            "level": level, "allowed_raw_labels": list(ALLOWED_RAW_SEG_LABELS),
            "unexpected_values_605": result["unexpected_values_605"],
            "unexpected_values_608": result["unexpected_values_608"],
        })
        return result

    seg_605_i, seg_608_i = seg_605.astype(np.int64), seg_608.astype(np.int64)
    result["raw_seg_equal"] = bool(np.array_equal(seg_605_i, seg_608_i))
    foreground_605, foreground_608 = seg_605_i == 1, seg_608_i == 1
    result["foreground_equal"] = bool(np.array_equal(foreground_605, foreground_608))
    result["foreground_unequal_voxels"] = int(np.count_nonzero(foreground_605 != foreground_608))
    effective_605, effective_608 = _effective_labels(seg_605_i), _effective_labels(seg_608_i)
    result["effective_labels_equal"] = bool(np.array_equal(effective_605, effective_608))
    result["effective_unequal_voxels"] = int(np.count_nonzero(effective_605 != effective_608))

    if result["raw_seg_equal"]:
        result["raw_unequal_voxels"] = 0
        result["raw_diff_pairs"] = {}
        return result
    raw_diff = seg_605_i != seg_608_i
    pairs: dict[str, int] = {}
    for source_value, target_value in zip(
        seg_605_i[raw_diff].tolist(), seg_608_i[raw_diff].tolist(), strict=True
    ):
        key = f"{source_value}→{target_value}"
        pairs[key] = pairs.get(key, 0) + 1
    result["raw_unequal_voxels"] = int(np.count_nonzero(raw_diff))
    result["raw_diff_pairs"] = dict(sorted(pairs.items()))
    if result["effective_labels_equal"]:
        # 信息性：原始 seg 有差异，但经 RemoveLabelTansform(-1, 0) 后有效标签逐值相同
        record["informational_differences"].append({
            "case_id": record["case_id"], "kind": "raw_seg_difference_effectively_equal",
            "level": level, "unequal_voxels": result["raw_unequal_voxels"],
            "raw_diff_pairs": result["raw_diff_pairs"],
            "foreground_equal": result["foreground_equal"],
        })
    else:
        record["mismatches"].append({
            "case_id": record["case_id"], "channel": "SEG",
            "kind": "effective_label_values_differ", "level": level,
            "unequal_voxels": result["effective_unequal_voxels"],
            "raw_unequal_voxels": result["raw_unequal_voxels"],
            "raw_diff_pairs": result["raw_diff_pairs"],
        })
    return result


def _finalize(record: dict) -> None:
    mismatches = record["mismatches"]
    record["comparable"] = not any(
        item["kind"] in ("ndim_mismatch", "missing_mri_channels") for item in mismatches
    )
    record["effective_input_consistent"] = not mismatches
    label = record["label"] or {}
    record["raw_identical"] = bool(
        not mismatches
        and label.get("raw_seg_equal") is True
        and all(channel.get("exact_equal") for channel in record["channels"].values())
    )
    if not mismatches and record["raw_identical"]:
        record["classification"] = "raw_identical"
    elif not mismatches:
        record["classification"] = "raw_label_difference_only"
    elif not record["comparable"]:
        record["classification"] = "unchecked"
    else:
        record["classification"] = "effective_mismatch"


# --------------------------------------------------------------------------- 聚合


def _empty_channel_aggregate() -> dict:
    return {
        "exact_equal_cases": 0,
        "mismatch_cases": 0,
        "global_max_abs_diff": 0.0,
        "global_mean_abs_diff": None,
        "total_unequal_voxels": 0,
        "total_voxels": 0,
        "_abs_sum": 0.0,
    }


def _run_level(
    *,
    identifiers_605: tuple[str, ...],
    identifiers_608: tuple[str, ...],
    load_and_compare,
    missing_605,
    missing_608,
    level: str,
    max_cases: int | None,
    progress: bool,
) -> dict:
    """跑完某一层级（raw / preprocessed）的病例循环并聚合；不写文件、不修改任何输入。"""
    set_605, set_608 = set(identifiers_605), set(identifiers_608)
    per_channel = {name: _empty_channel_aggregate() for name in MRI_CHANNEL_NAMES}
    mismatched_cases: list[dict] = []
    informational: list[dict] = []
    per_case_summary: list[dict] = []
    prior_entries: list[dict] = []
    load_failures: list[dict] = []
    n_checked = n_raw_identical = n_effective_equal = n_foreground_equal = 0
    n_prior_out_of_range = n_prior_non_finite = 0

    common = sorted(set_605 & set_608)
    only_605 = sorted(set_605 - set_608)
    only_608 = sorted(set_608 - set_605)
    for case_identifier in only_605 + only_608:
        mismatched_cases.append({
            "case_id": case_identifier, "channel": "CASE", "level": level,
            "kind": "case_missing_in_other_dataset",
            "present_in": "605" if case_identifier in set_605 else "608",
        })

    to_check = common if max_cases is None else common[: int(max_cases)]
    for case_identifier in tqdm(to_check, desc=f"{level} audit", unit="case", disable=not progress):
        missing_entries = [
            {"case_id": case_identifier, "dataset": tag, "level": level,
             "kind": "missing_file", "files": missing}
            for tag, missing in (
                ("605", missing_605(case_identifier)),
                ("608", missing_608(case_identifier)),
            )
            if missing
        ]
        if missing_entries:
            load_failures.extend(missing_entries)
            mismatched_cases.extend(missing_entries)
            continue
        try:
            record = load_and_compare(case_identifier)
        except Exception as exc:  # 读失败必须 fail-closed，不静默当作通过
            failure = {"case_id": case_identifier, "level": level, "kind": "load_error",
                       "error": f"{type(exc).__name__}: {exc}"}
            load_failures.append(failure)
            mismatched_cases.append(failure)
            continue

        n_checked += 1
        n_raw_identical += int(bool(record["raw_identical"]))
        n_effective_equal += int(bool(record["effective_input_consistent"]))
        label = record["label"] or {}
        n_foreground_equal += int(bool(label.get("foreground_equal")))
        per_case_summary.append({
            "case_id": case_identifier,
            "classification": record["classification"],
            "mri_exact_equal": all(info["exact_equal"] for info in record["channels"].values())
            if record["channels"] else None,
            "raw_seg_equal": label.get("raw_seg_equal"),
            "raw_unequal_voxels": label.get("raw_unequal_voxels"),
            "effective_labels_equal": label.get("effective_labels_equal"),
            "foreground_equal": label.get("foreground_equal"),
        })
        for name, info in record["channels"].items():
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
                aggregate["_abs_sum"] += float(info["mean_abs_diff"]) * int(info["total_voxels"])
        mismatched_cases.extend(record["mismatches"])
        informational.extend(record["informational_differences"])
        details = record["prior_channels_info_only"]
        if details is not None:
            prior_entries.append(details)
            n_prior_out_of_range += int(details["any_outside_unit_interval"])
            n_prior_non_finite += int(details["any_non_finite"])

    for aggregate in per_channel.values():
        if aggregate["total_voxels"]:
            aggregate["global_mean_abs_diff"] = aggregate["_abs_sum"] / aggregate["total_voxels"]
        aggregate.pop("_abs_sum", None)

    n_mri_mismatch = sum(1 for entry in per_case_summary if entry["mri_exact_equal"] is False)
    n_label_mismatch = sum(
        1 for entry in per_case_summary if entry["effective_labels_equal"] is False
    )
    return {
        "level": level,
        "status": LEVEL_PASS if (not mismatched_cases and n_checked > 0) else LEVEL_FAIL,
        "n_cases_605": len(identifiers_605),
        "n_cases_608": len(identifiers_608),
        "case_set_equal": bool(set_605 == set_608),
        "case_order_equal": bool(tuple(identifiers_605) == tuple(identifiers_608)),
        "n_cases_checked": int(n_checked),
        "n_cases_raw_identical": int(n_raw_identical),
        "n_cases_effective_equal": int(n_effective_equal),
        "n_cases_effective_input_mismatch": int(n_checked - n_effective_equal),
        "n_cases_mri_mismatch": int(n_mri_mismatch),
        "n_cases_label_mismatch": int(n_label_mismatch),
        "n_cases_foreground_equal": int(n_foreground_equal),
        "n_cases_informational_raw_label_difference": len(informational),
        "n_cases_prior_outside_unit_interval_info_only": int(n_prior_out_of_range),
        "n_cases_prior_non_finite_info_only": int(n_prior_non_finite),
        "mri_exact_equal_all": bool(n_checked > 0 and n_mri_mismatch == 0),
        "channels": list(MRI_CHANNEL_NAMES),
        "per_channel": per_channel,
        "mismatched_cases": mismatched_cases,
        "informational_differences": informational,
        "per_case_summary": per_case_summary,
        "load_failures": load_failures,
        "prior_channels_info_only": prior_entries,
        "case_folders": {},
    }


# --------------------------------------------------------------------------- 报告


def plans_match(preprocessed_605_dir, preprocessed_608_dir) -> dict | None:
    """比较两侧 ``nnUNetPlans.json`` 的架构/patch/batch/spacing（608 必须冻结继承 605）。"""
    if preprocessed_605_dir is None or preprocessed_608_dir is None:
        return None
    source = Path(preprocessed_605_dir) / "nnUNetPlans.json"
    target = Path(preprocessed_608_dir) / "nnUNetPlans.json"
    if not source.is_file() or not target.is_file():
        return {"available": False, "reason": "nnUNetPlans.json missing on one side"}
    plans_605 = _load_json_object(source)
    plans_608 = _load_json_object(target)
    keys = ("architecture", "patch_size", "batch_size", "spacing", "network_class_name")
    result: dict = {"available": True, "configurations": {}}
    for name in ("3d_fullres",):
        config_605 = plans_605.get("configurations", {}).get(name)
        config_608 = plans_608.get("configurations", {}).get(name)
        if config_605 is None or config_608 is None:
            result["configurations"][name] = {"comparable": False}
            continue
        entry = {
            key: {
                "605": config_605.get(key),
                "608": config_608.get(key),
                "equal": config_605.get(key) == config_608.get(key),
            }
            for key in keys
        }
        entry["preprocessor_name"] = {
            "605": config_605.get("preprocessor_name", "DefaultPreprocessor"),
            "608": config_608.get("preprocessor_name"),
            "equal": config_608.get("preprocessor_name") == "PredictedAnatomyPreprocessor",
        }
        entry["normalization_schemes_equal_first_three"] = (
            config_605.get("normalization_schemes", [])[:MRI_CHANNELS]
            == config_608.get("normalization_schemes", [])[:MRI_CHANNELS]
        )
        entry["comparable"] = True
        result["configurations"][name] = entry
    return result


def run_audit(
    *,
    raw_605,
    raw_608,
    splits_605: list,
    splits_608: list,
    dataset_json_605: dict,
    dataset_json_608: dict,
    dataset_name_605: str,
    dataset_name_608: str,
    fold: int,
    preprocessed_605=None,
    preprocessed_608=None,
    max_cases: int | None = None,
    progress: bool = True,
    include_preprocessed: bool = True,
) -> dict:
    """执行完整审计并返回结构化报告（不写文件、不修改任何输入）。"""
    started = time.time()
    metadata = compare_metadata(
        dataset_json_605,
        dataset_json_608,
        dataset_name_605=dataset_name_605,
        dataset_name_608=dataset_name_608,
    )
    prior_contract = validate_prior_contract(dataset_json_608, dataset_name_608)
    splits_report = compare_splits(splits_605, splits_608, fold=fold)

    raw_report = _run_level(
        identifiers_605=raw_605.identifiers,
        identifiers_608=raw_608.identifiers,
        load_and_compare=lambda cid: compare_raw_case(cid, source_605=raw_605, source_608=raw_608),
        missing_605=raw_605.missing_files,
        missing_608=raw_608.missing_files,
        level="raw",
        max_cases=max_cases,
        progress=progress,
    )
    raw_report["case_folders"] = {"605": str(raw_605.dataset_dir), "608": str(raw_608.dataset_dir)}

    preprocessed_available = preprocessed_605 is not None and preprocessed_608 is not None
    preprocessed_report: dict | None = None
    if include_preprocessed and preprocessed_available:
        preprocessed_report = _run_level(
            identifiers_605=preprocessed_605.identifiers,
            identifiers_608=preprocessed_608.identifiers,
            load_and_compare=lambda cid: compare_preprocessed_case(
                cid, source_605=preprocessed_605, source_608=preprocessed_608
            ),
            missing_605=lambda cid: missing_preprocessed_files(preprocessed_605.case_folder, cid),
            missing_608=lambda cid: missing_preprocessed_files(preprocessed_608.case_folder, cid),
            level="preprocessed",
            max_cases=max_cases,
            progress=progress,
        )
        preprocessed_report["case_folders"] = {
            "605": str(preprocessed_605.case_folder),
            "608": str(preprocessed_608.case_folder),
        }
        preprocessed_report["plans_match"] = plans_match(
            preprocessed_605.dataset_dir, preprocessed_608.dataset_dir
        )

    #: FAIL 驱动的问题（发现任何不一致）
    problems: list[str] = []
    #: 阻止 PASS 但不代表不一致的限制（子集检查、preprocessed 尚未可用）
    limitations: list[str] = []
    if not metadata["mri_channel_names_equal"]:
        problems.append("前三个 MRI 通道名不是 T2W/ADC/HBV 或两数据集不一致")
    if not metadata["mri_channel_count_equal"]:
        problems.append("Dataset605 的通道数不是 3（前三个 MRI 通道约定被破坏）")
    if not metadata["labels_equal"]:
        problems.append("dataset.json 的 labels 定义不一致或不是 {background:0, lesion:1}")
    if not metadata["file_ending_equal"]:
        problems.append("dataset.json 的 file_ending 不一致")
    if not metadata["num_training_equal"]:
        problems.append("dataset.json 的 numTraining 不一致")
    if prior_contract != "VALID":
        problems.append(f"Dataset608 prior 契约非法（{prior_contract}）")
    if not splits_report["all_folds_train_val_set_equal"]:
        problems.append("splits_final.json 的 fold 划分（train/val 集合）不一致")

    for level_report in (raw_report, preprocessed_report):
        if level_report is None:
            continue
        tag = level_report["level"]
        if not level_report["case_set_equal"]:
            problems.append(f"{tag}：病例集合不一致")
        if not level_report["case_order_equal"]:
            problems.append(f"{tag}：病例顺序不一致")
        if level_report["load_failures"]:
            problems.append(
                f"{tag}：{len(level_report['load_failures'])} 例文件缺失或读取失败"
            )
        if level_report["n_cases_effective_input_mismatch"]:
            problems.append(
                f"{tag}：{level_report['n_cases_effective_input_mismatch']} 例存在影响有效输入"
                "一致性的问题（MRI 体素/几何、有效标签、形状、dtype、非有限值；逐例明细见 "
                "mismatched_cases）"
            )
        for name, aggregate in level_report["per_channel"].items():
            if aggregate["mismatch_cases"]:
                problems.append(
                    f"{tag}/{name}：{aggregate['mismatch_cases']} 例数值不一致"
                    f"（global max|Δ|={aggregate['global_max_abs_diff']:.6g}）"
                )
    if max_cases is not None:
        limitations.append(
            f"仅检查了每个层级的前 {int(max_cases)} 个病例（--max-cases），不能判 PASS"
        )
    if not include_preprocessed:
        limitations.append("--skip-preprocessed：未检查 preprocessed 层，不能判 PASS")
    elif not preprocessed_available:
        limitations.append(
            "Dataset608 预处理产物尚不存在：只完成 raw 层审计，不足以支撑 C vs D 的单变量解释"
        )

    if problems:
        status = STATUS_FAIL
    elif max_cases is not None:
        status = STATUS_PARTIAL
    elif preprocessed_report is not None:
        status = STATUS_PASS
    else:
        status = STATUS_RAW_ONLY

    return {
        "status": status,
        "problems": problems,
        "limitations": limitations,
        "generated_at_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "elapsed_seconds": round(time.time() - started, 3),
        "dataset_605": dataset_name_605,
        "dataset_608": dataset_name_608,
        "fold": int(fold),
        "case_set_equal": bool(raw_report["case_set_equal"]),
        "case_order_equal": bool(raw_report["case_order_equal"]),
        "split_equal": bool(splits_report["all_folds_train_val_set_equal"]),
        "metadata_ok": bool(
            metadata["mri_channel_names_equal"]
            and metadata["mri_channel_count_equal"]
            and metadata["labels_equal"]
            and metadata["file_ending_equal"]
            and metadata["num_training_equal"]
        ),
        "prior_contract_status": prior_contract,
        "preprocessed_available": bool(preprocessed_available),
        "preprocessed_checked": preprocessed_report is not None,
        "preprocessed_skip_reason": (
            None
            if preprocessed_report is not None
            else ("--skip-preprocessed" if not include_preprocessed
                  else "Dataset608 preprocessed products not present yet")
        ),
        "n_cases_mri_mismatch": int(raw_report["n_cases_mri_mismatch"]),
        "n_cases_label_mismatch": int(raw_report["n_cases_label_mismatch"]),
        "n_cases_prior_outside_unit_interval_info_only": int(
            raw_report["n_cases_prior_outside_unit_interval_info_only"]
        ),
        "n_cases_prior_non_finite_info_only": int(
            raw_report["n_cases_prior_non_finite_info_only"]
        ),
        "allowed_raw_seg_labels": list(ALLOWED_RAW_SEG_LABELS),
        "effective_label_remap": {str(k): v for k, v in EFFECTIVE_LABEL_REMAP.items()},
        "geometry_atol": GEOMETRY_ATOL,
        "channels": list(MRI_CHANNEL_NAMES),
        "metadata": metadata,
        "splits": splits_report,
        "levels": {"raw": raw_report, "preprocessed": preprocessed_report},
        "inputs_modified": False,
        "notes": [
            "本工具只读：不物化 Dataset608、不运行 preprocessing、不生成/修正任何 prior。",
            "本工具只回答「MRI / lesion / split 是否保持不变」；anatomy prior 自身质量由 Stage-1 "
            "诊断负责，职责不混合。",
            "608 的 prior 通道（第 4-6 通道）只做信息性范围统计，不参与 PASS/FAIL，也不参与前三 "
            "MRI 通道的判等。",
            "MRI 判定依据是 np.array_equal（最严格，任一体素不同即 FAIL）；max|Δ|、affected "
            "voxels、均值差值仅用于诊断。",
            (
                "seg 合法原始取值为 {-1, 0, 1}：-1 来自固定版本 nnU-Net 的 crop_to_nonzero"
                "（preprocessing/cropping/cropping.py:19,36），并在训练/验证变换"
                " RemoveLabelTansform(-1, 0)（training/nnUNetTrainer/nnUNetTrainer.py:800,855）中"
                "于损失计算前映射为 0。"
            ),
            (
                "原始逐值相同（raw_seg_equal）≠ 有效标签相同（effective_labels_equal）：纯 -1↔0 "
                "差异记为信息性，不影响 PASS；0↔1、-1↔1、{-1,0,1} 之外的取值、MRI 任一体素差异、"
                "形状/dtype/非有限值/几何不一致一律 FAIL。"
            ),
            (
                f"PASS 定义：status={STATUS_PASS} 要求 raw 与 preprocessed 两层都完整通过"
                "（无 --max-cases 子集、无读取失败）、病例集合与顺序一致、split 一致、"
                "dataset.json 与 608 prior 契约合法、前三个 MRI 通道逐数组完全相同、"
                "所有 seg 合法且有效标签逐值相同。"
            ),
            (
                f"仅 raw 层通过时为 {STATUS_RAW_ONLY}（退出码仍为 2）：在真实 608 预处理产物存在"
                "并被审计前，不得把 C vs D 解释为「D 仅增加 predicted anatomy context」。"
            ),
        ],
    }


# --------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audit_dataset605_608_equivalence.py",
        description=(
            "Dataset605 与 Dataset608 的一致性审计（只读、fail-closed）：前三个 MRI 通道、"
            "lesion 标签、病例集合/顺序与 split 是否与 605 保持不变。"
            "prior 通道与 anatomy 质量不在本工具职责内。"
        ),
    )
    parser.add_argument("--dataset-605", default=DEFAULT_DATASET_605)
    parser.add_argument("--dataset-608", default=DEFAULT_DATASET_608)
    parser.add_argument("--configuration", default=DEFAULT_CONFIGURATION)
    parser.add_argument("--fold", type=int, default=0, help="仅用于 split 报告与说明（默认 0）")
    parser.add_argument(
        "--raw-root", default=None, help="nnUNet_raw 根目录（默认取环境变量 nnUNet_raw）"
    )
    parser.add_argument(
        "--preprocessed-root",
        default=None,
        help="nnUNet_preprocessed 根目录（默认取环境变量 nnUNet_preprocessed）",
    )
    parser.add_argument(
        "--skip-preprocessed",
        action="store_true",
        help="显式跳过 preprocessed 层（此时 status 只能是 RAW_ONLY_PASS，退出码 2）",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="每个层级只检查前 N 个共同病例（烟雾测试用）；此时永不判 PASS（不一致→FAIL，否则 PARTIAL）",
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖已存在的报告文件（默认拒绝，避免覆盖既有审计结果）",
    )
    parser.add_argument("--no-progress", action="store_true")
    return parser


def _resolve_root(argument: str | None, environment: str) -> Path:
    raw = argument or os.environ.get(environment)
    if not raw:
        raise AuditError(
            f"环境变量 {environment} 未设置且未显式给出对应参数。请先执行：cd "
            "/opt/data/private/lm/my-projects && conda activate lm && source scripts/env_nnunet.sh"
        )
    root = Path(raw)
    if not root.is_dir():
        raise AuditError(f"{environment} 指向的目录不存在：{root}")
    return root


def _try_preprocessed(root: Path, dataset_name: str, configuration: str):
    """尝试构造预处理数据源；目录不存在时返回 ``None``（由 status 语义显式表达，不静默通过）。"""
    folder = root / dataset_name
    if not (folder / f"nnUNetPlans_{configuration}").is_dir():
        return None
    return PreprocessedCaseSource(folder, configuration)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.time()
    raw_root = _resolve_root(args.raw_root, "nnUNet_raw")
    dir_605, dir_608 = raw_root / args.dataset_605, raw_root / args.dataset_608

    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        raise SystemExit(
            f"[audit] 报告文件已存在，拒绝覆盖：{output_path}\n"
            "如需重新生成请显式加 --overwrite（或换一个 --output 路径）。"
        )

    raw_605 = RawCaseSource(dir_605, expected_channels=CHANNELS_605_RAW)
    raw_608 = RawCaseSource(dir_608, expected_channels=CHANNELS_608_RAW)

    preprocessed_605 = preprocessed_608 = None
    preprocessed_root = None
    if not args.skip_preprocessed:
        preprocessed_root = _resolve_root(args.preprocessed_root, "nnUNet_preprocessed")
        preprocessed_605 = _try_preprocessed(
            preprocessed_root, args.dataset_605, args.configuration
        )
        preprocessed_608 = _try_preprocessed(
            preprocessed_root, args.dataset_608, args.configuration
        )

    report = run_audit(
        raw_605=raw_605,
        raw_608=raw_608,
        splits_605=_load_json(dir_605 / "splits_final.json"),
        splits_608=_load_json(dir_608 / "splits_final.json"),
        dataset_json_605=_load_json_object(dir_605 / "dataset.json"),
        dataset_json_608=_load_json_object(dir_608 / "dataset.json"),
        dataset_name_605=args.dataset_605,
        dataset_name_608=args.dataset_608,
        fold=args.fold,
        preprocessed_605=preprocessed_605,
        preprocessed_608=preprocessed_608,
        max_cases=args.max_cases,
        progress=not args.no_progress,
        include_preprocessed=not args.skip_preprocessed,
    )
    report["raw_root"] = str(raw_root)
    report["preprocessed_root"] = str(preprocessed_root) if preprocessed_root else None
    report["skip_preprocessed_requested"] = bool(args.skip_preprocessed)
    report["elapsed_seconds"] = round(time.time() - started, 3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    _print_summary(report, output_path)
    return 0 if report["status"] == STATUS_PASS else 2


def _print_summary(report: dict, output_path: Path) -> None:
    print("\n=== Dataset605 / Dataset608 一致性审计 ===")
    print(f"  status                  : {report['status']}")
    print(f"  病例集合一致             : {report['case_set_equal']}")
    print(f"  病例顺序一致             : {report['case_order_equal']}")
    print(f"  split 一致               : {report['split_equal']}")
    print(f"  dataset.json 元数据一致  : {report['metadata_ok']}")
    print(f"  608 prior 契约           : {report['prior_contract_status']}")
    print(
        f"  preprocessed 层已检查    : {report['preprocessed_checked']}"
        f"（skip_reason={report['preprocessed_skip_reason']}）"
    )
    print(f"  MRI 不一致例数           : {report['n_cases_mri_mismatch']}")
    print(f"  label 不一致例数         : {report['n_cases_label_mismatch']}")
    print(
        "  prior 超出 [0,1] 例数（仅信息性）: "
        f"{report['n_cases_prior_outside_unit_interval_info_only']}"
    )
    print(
        f"  prior 非有限例数（仅信息性）    : {report['n_cases_prior_non_finite_info_only']}"
    )
    for level in ("raw", "preprocessed"):
        entry = report["levels"][level]
        if entry is None:
            print(f"  [{level}] {LEVEL_NOT_CHECKED}")
            continue
        print(
            f"  [{level}] status={entry['status']} checked={entry['n_cases_checked']} "
            f"mri_mismatch={entry['n_cases_mri_mismatch']} "
            f"label_mismatch={entry['n_cases_label_mismatch']} "
            f"raw_identical={entry['n_cases_raw_identical']} "
            f"informational={entry['n_cases_informational_raw_label_difference']}"
        )
        for name in report["channels"]:
            channel = entry["per_channel"][name]
            print(
                f"      {name:>3s}: exact={channel['exact_equal_cases']} "
                f"mismatch={channel['mismatch_cases']} "
                f"global_max|Δ|={channel['global_max_abs_diff']!r} "
                f"global_mean|Δ|={channel['global_mean_abs_diff']!r}"
            )
    if report["limitations"]:
        print("  未满足 PASS 的限制：")
        for limitation in report["limitations"]:
            print(f"    - {limitation}")
    if report["problems"]:
        print("  问题（FAIL 驱动）：")
        for problem in report["problems"]:
            print(f"    - {problem}")
    print(f"  输入是否被修改           : {report['inputs_modified']}")
    print(f"  报告：{output_path}")
    print("=== 审计结束 ===")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AuditError as exc:
        print(f"[audit] FAILED: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
