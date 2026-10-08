"""Stage 1 解剖先验数据集契约（Dataset607_PICAI_Anatomy / 3d_fullres）。

这是 preparation、训练、预测与独立评估四方**共用**的唯一契约定义。任何一方单独放宽它都会
破坏「anatomy 先验只能来自该病例自身 MRI 的预测」这一前提。

数据集语义
----------
- 输入：**单个 T2W 通道**（``channel_names == {"0000": "T2W"}``）；
- 标签：三个**非互斥**区域的位编码（``WG+2*PZ+4*TZ``），因此使用 nnU-Net 的 region-based
  heads（``regions_class_order == [1, 2, 4]``，即 WG / PZ / TZ）；位编码的含义是「区域隶属
  组合」，**不是**新的解剖分类；
- 来源：``wg_source = materialized_wg``、``zonal_source = zonal_yuan``，即 WG 来自已物化的
  WG 掩膜，PZ/TZ 来自自动分区（Yuan）结果。二者是**算法伪监督**，不是手工标注；
- 已知缺失病例 ``11050_1001070`` 被**显式排除**：不得用空掩膜或 zonal union 顶替。

Stage-1 的角色边界（必须一直保持）
---------------------------------
- anatomy model **不读取 lesion GT**，任何 split 上都不允许；
- 本模型**不是论文的主要创新点**，它只是 ``anatomical prior generator``；
- validation / test 病例的 anatomy prior **必须**来自该病例自身 MRI 的预测；
- **禁止**把 GT WG/PZ/TZ 直接作为 lesion model 的推理输入（GT 只能用于 ``ORACLE_GT`` 上界
  分析，且必须显式标记，见 :mod:`..evaluation.anatomy_metrics`）。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

#: Stage-1 数据集的唯一合法名称与配置
ANATOMY_DATASET = "Dataset607_PICAI_Anatomy"
ANATOMY_CONFIGURATION = "3d_fullres"
#: region heads 的标签定义；顺序即 ``regions_class_order``
ANATOMY_LABELS = {
    "background": 0,
    "WG": [1, 3, 5, 7],
    "PZ": [2, 3, 6, 7],
    "TZ": [4, 5, 6, 7],
}
ANATOMY_CLASS_ORDER = [1, 2, 4]
#: 已知无法物化 WG 的病例：显式排除，绝不替代
ANATOMY_KNOWN_MISSING = "11050_1001070"
#: (studies, train, val)；**study 数**，不是患者数
ANATOMY_COUNTS = (1499, 1276, 223)
#: 物理几何比较的绝对容差；相对容差恒为 0
ANATOMY_GEOMETRY_ATOL = 1e-4
#: 本地导出的解剖预测文件名（由 nnU-Net 原生 validation / prediction 写出）
ANATOMY_PREDICTION_SUFFIXES = (".npz", ".pkl", ".nii.gz")


def read_nifti_stored_array(path) -> np.ndarray:
    """读取 NIfTI 的**原始存储值**并转成 SimpleITK 的 ZYX 数组轴序。

    SimpleITK 的 NIfTI reader 会把存储的 NaN 替换为 0；因此在做任何整数转换或隶属判断
    **之前**，必须通过 nibabel 检查缩放前的原始值。非 3D 或含非有限值即报错。
    """
    import nibabel as nib

    array = np.asanyarray(nib.load(str(path)).dataobj)
    if array.ndim != 3 or not np.isfinite(array).all():
        raise ValueError(f"{path}: expected finite stored 3D NIfTI values")
    return array.transpose(2, 1, 0)  # XYZ -> native SimpleITK ZYX array order


def validate_label_array(array, allowed, name: str) -> np.ndarray:
    """校验一个 3D 标签数组：整数、有限、且取值落在 ``allowed`` 内。"""
    array = np.asarray(array)
    if array.ndim != 3 or not np.isfinite(array).all():
        raise ValueError(f"{name}: expected finite 3D labels")
    if not np.equal(array, np.rint(array)).all() or not np.isin(array, allowed).all():
        raise ValueError(f"{name}: non-integer or illegal labels; expected {allowed}")
    return array


def encode_anatomy(wg: np.ndarray, yuan: np.ndarray) -> np.ndarray:
    """把 WG 与 Yuan 分区掩膜编码为位模式 ``WG + 2*PZ + 4*TZ``。"""
    wg = validate_label_array(wg, [0, 1], "WG")
    yuan = validate_label_array(yuan, [0, 1, 2], "Yuan")
    if wg.shape != yuan.shape:
        raise ValueError("WG/Yuan shape mismatch")
    return (
        (wg == 1).astype(np.uint8)
        + 2 * (yuan == 1).astype(np.uint8)
        + 4 * (yuan == 2).astype(np.uint8)
    )


def encode_anatomy_heads(masks) -> np.ndarray:
    """把三个**二值** WG/PZ/TZ 掩膜编码为同一套位模式（供评估侧反解）。"""
    masks = np.asarray(masks)
    if masks.ndim != 4 or masks.shape[0] != 3 or not np.isin(masks, [0, 1]).all():
        raise ValueError("expected three binary WG/PZ/TZ masks")
    return sum(masks[i].astype(np.uint8) * bit for i, bit in enumerate((1, 2, 4)))


def anatomy_geometry(image) -> dict:
    """抽取并校验 ``size / spacing / origin / direction``；非法几何直接报错。"""
    if image.GetDimension() != 3:
        raise ValueError("expected 3D geometry")
    size, spacing = image.GetSize(), np.asarray(image.GetSpacing())
    origin, direction = (
        np.asarray(image.GetOrigin()),
        np.asarray(image.GetDirection()).reshape(3, 3),
    )
    if (
        any(n <= 0 for n in size)
        or not np.isfinite(spacing).all()
        or (spacing <= 0).any()
        or not np.isfinite(origin).all()
        or not np.isfinite(direction).all()
        or not np.allclose(
            direction @ direction.T, np.eye(3), rtol=0, atol=ANATOMY_GEOMETRY_ATOL
        )
    ):
        raise ValueError("invalid physical geometry")
    return {
        "size": list(size),
        "spacing": spacing.tolist(),
        "origin": origin.tolist(),
        "direction": direction.flatten().tolist(),
    }


def anatomy_same_grid(reference, other) -> None:
    """要求两个图像处在**同一物理网格**上；不一致直接报错（不重采样）。"""
    a, b = anatomy_geometry(reference), anatomy_geometry(other)
    for key in a:
        same = (
            a[key] == b[key]
            if key == "size"
            else np.allclose(a[key], b[key], rtol=0, atol=ANATOMY_GEOMETRY_ATOL)
        )
        if not same:
            raise ValueError(f"physical grid mismatch: {key}")


def validate_anatomy_dataset(
    dataset_json,
    dataset_name=ANATOMY_DATASET,
    configuration=ANATOMY_CONFIGURATION,
) -> dict:
    """校验数据集身份、通道、region 定义、provenance 契约与病例范围。

    任何一项不符即报错（fail-closed）。返回 ``anatomy_contract``，调用方据此获得冻结的
    病例清单与 train/val 归属。
    """
    if dataset_name != ANATOMY_DATASET or configuration != ANATOMY_CONFIGURATION:
        raise ValueError(
            f"anatomy requires {ANATOMY_DATASET} / {ANATOMY_CONFIGURATION}"
        )
    if dataset_json.get("channel_names") != {"0000": "T2W"}:
        raise ValueError("anatomy requires single T2W channel 0000")
    if list(dataset_json.get("labels", {}).items()) != list(ANATOMY_LABELS.items()):
        raise ValueError("anatomy regions or region order mismatch")
    if dataset_json.get("regions_class_order") != ANATOMY_CLASS_ORDER:
        raise ValueError("anatomy regions_class_order mismatch")
    contract = dataset_json.get("anatomy_contract", {})
    if (
        contract.get("encoding") != "WG+2*PZ+4*TZ"
        or contract.get("zonal_source") != "zonal_yuan"
        or contract.get("wg_source") != "materialized_wg"
        or contract.get("explicit_exclusions") != [ANATOMY_KNOWN_MISSING]
    ):
        raise ValueError("missing or invalid anatomy provenance contract")
    ids = contract.get("case_ids", [])
    if (
        len(ids) != ANATOMY_COUNTS[0]
        or len(ids) != len(set(ids))
        or dataset_json.get("numTraining") != len(ids)
    ):
        raise ValueError("anatomy study scope mismatch")
    if ANATOMY_KNOWN_MISSING in ids:
        raise ValueError("known missing WG cannot enter complete supervision")
    for cid in ids:
        if (
            not isinstance(cid, str)
            or len(cid.split("_")) != 2
            or not all(p.isdigit() for p in cid.split("_"))
        ):
            raise ValueError(f"invalid anatomy case ID: {cid}")
    return contract


def validate_anatomy_split(split_path, dataset_json, fold=0) -> tuple[list, list]:
    """校验冻结的单一显式 split：fold 必须为 0、case 范围与 provenance 一致、无患者重叠。

    **不提供随机回退**：split 文件缺失即报错。
    """
    contract = validate_anatomy_dataset(dataset_json)
    if str(fold) != "0":
        raise ValueError("anatomy frozen single split requires fold 0; no random fallback")
    path = Path(split_path)
    if not path.is_file():
        raise ValueError(f"explicit anatomy split missing: {path}; no random fallback")
    splits = json.loads(path.read_text())
    if not isinstance(splits, list) or len(splits) != 1:
        raise ValueError("expected one explicit frozen anatomy split")
    split = splits[0]
    tr, va = split.get("train", []), split.get("val", [])
    if (
        len(tr) != ANATOMY_COUNTS[1]
        or len(va) != ANATOMY_COUNTS[2]
        or len(tr) != len(set(tr))
        or len(va) != len(set(va))
        or set(tr) & set(va)
        or set(tr) | set(va) != set(contract["case_ids"])
    ):
        raise ValueError("anatomy split case scope/duplicates mismatch")
    if set(tr) != set(contract.get("train_cases", [])) or set(va) != set(
        contract.get("val_cases", [])
    ):
        raise ValueError("anatomy split differs from frozen provenance")
    if {c.split("_")[0] for c in tr} & {c.split("_")[0] for c in va}:
        raise ValueError("anatomy train/validation patient overlap")
    return tr, va


def validate_anatomy_probabilities(probabilities, properties, reference) -> np.ndarray:
    """校验原生恢复的 ``(C,Z,Y,X)`` 概率数组与其可信物理元数据。

    仅接受**本地原生导出**（``npz`` 数组 + ``pkl`` 物理元数据）。任何形状、值域、物理元数据
    或 ``shape_before_cropping`` 不一致都直接报错；不做坐标变换、不重采样。
    """
    probabilities = np.asarray(probabilities)
    geometry = anatomy_geometry(reference)
    shape = tuple(reversed(geometry["size"]))
    if probabilities.shape != (3, *shape):
        raise ValueError(
            f"anatomy probability channels/shape mismatch: "
            f"{probabilities.shape} vs {(3, *shape)}"
        )
    if (
        not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
    ):
        raise ValueError("anatomy probabilities must be finite within [0,1]")
    stuff = properties.get("sitk_stuff", {})
    for key in ("spacing", "origin", "direction"):
        value = np.asarray(stuff.get(key, []))
        if value.shape != np.asarray(geometry[key]).shape or not np.allclose(
            value, geometry[key], rtol=0, atol=ANATOMY_GEOMETRY_ATOL
        ):
            raise ValueError(f"anatomy probability physical metadata mismatch: {key}")
    if not np.allclose(
        properties.get("spacing", []),
        list(reversed(geometry["spacing"])),
        rtol=0,
        atol=ANATOMY_GEOMETRY_ATOL,
    ):
        raise ValueError("anatomy probability array spacing mismatch")
    # shape_before_cropping 存储于 preprocessing 转置后的坐标系
    before = properties.get("shape_before_cropping", [])
    forward = properties.get("anatomy_transpose_forward")
    if (
        forward is None
        or sorted(forward) != [0, 1, 2]
        or tuple(before) != tuple(shape[i] for i in forward)
    ):
        raise ValueError("anatomy probability original shape metadata mismatch")
    return probabilities


# --------------------------------------------------------------------------- 历史私有别名
# trainers/tests 历史以 ``anatomy_*`` 命名引用这些函数；保留同一套名字使抽取到本模块
# **不改变**任何既有调用点。
anatomy_read_array = read_nifti_stored_array
anatomy_validate_array = validate_label_array
anatomy_encode = encode_anatomy
anatomy_encode_heads = encode_anatomy_heads
validate_anatomy_probabilities.__doc__ = validate_anatomy_probabilities.__doc__
anatomy_validate_probabilities = validate_anatomy_probabilities
