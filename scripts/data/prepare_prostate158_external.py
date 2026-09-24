#!/usr/bin/env python3
"""Prostate158 独立外测输入准备：CSV 驱动的只读 audit 与软链接 prepare。

本脚本**只**为已训练完成的 Dataset605 PI-CAI 模型整理独立外测输入，不建立新 nnU-Net
Dataset ID、不做 planning/preprocessing、不训练、不推理。所有真实 Prostate158 影像与标注
**只读**：默认路径下不写任何字节，``prepare`` 也只在**独立新目录**中建立相对符号链接。

通道映射（固定，与 Dataset605 的 ``dataset.json`` 一致）：

    _0000 -> CSV 的 t2  （T2W）
    _0001 -> CSV 的 adc （ADC）
    _0002 -> CSV 的 dwi （Prostate158 DWI）

**跨域声明（硬约束，必须写入审计与清单）**：Prostate158 的 ``dwi`` 只是作为原模型第三通道
（PI-CAI 训练时为 HBV / high-b-value DWI）的跨域输入，**不能宣称与训练用 HBV 完全等价**。

两个子命令
----------
``audit``（只读，不生成推理输入）：逐例核对 t2/adc/dwi 与主参考标签
（固定 ``adc_tumor_reader1``）是否存在、可读，核对 size / spacing / origin / direction /
物理空间范围、标签取值与阴阳性，汇总：

- ``linkable``       三通道网格逐值一致，可直接软链接；
- ``grid_mismatch``  三通道网格存在可解释差异，默认不处理；仅当物理坐标关系可信且源影像
                     覆盖 T2 参考网格时标记 ``resample_candidate=true``；
- ``failed``         缺通道 / 缺主参考 / 不可读 / 标签非法等，绝不静默跳过或当阴性。

``prepare``（只写独立新目录）：默认**仅**接受 ``linkable`` 病例并建立相对软链接；
``grid_mismatch`` 计入 skipped。**只有显式加** ``--allow-resample`` 才会在独立目录中对
``resample_candidate`` 的派生强度图做显式重采样（T2 为参考网格、线性连续值插值），
原始影像不动；空间关系不能确认时非零退出，绝不自行配准。

阴性病例的定义是 fail-closed 的：CSV 的 ``adc_tumor_reader1`` 字段非空、文件存在可读、
标签合法且**经验证确实为空掩膜**（如 ``empty.nii.gz``）才算阴性；CSV 空字段、文件缺失或
读取失败一律报错，不是阴性。

输出目录默认拒绝覆盖；先在同级临时暂存目录中完整构建，成功后原子改名，失败不留半成品。
``--help`` 不打印结束汇总（帮助不是一次运行）。
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

#: 外测输入的 nnU-Net 通道号 -> (CSV 列名, 领域名称)。顺序固定不可变。
CHANNELS = (
    ("0000", "t2", "T2W"),
    ("0001", "adc", "ADC"),
    ("0002", "dwi", "Prostate158 DWI（跨域：不等同于训练用 HBV）"),
)
#: 主参考标签列（预先固定，禁止逐例改选）
PRIMARY_READER_COLUMN = "adc_tumor_reader1"
#: 第二读者列：仅在实际存在且可核对的病例上做读者敏感性分析
READER2_COLUMN = "adc_tumor_reader2"
REQUIRED_COLUMNS = (
    "ID",
    "t2",
    "adc",
    "dwi",
    PRIMARY_READER_COLUMN,
)
QUEUES = ("test", "train", "valid")

SCHEMA_VERSION = "1.0"
MANIFEST_NAME = "external_input_manifest.json"
IMAGES_DIRNAME = "images"

#: 几何逐值比较容差：同一导出网格的 header 元数据应逐值相等；不允许「差不多就算同网格」
GEOMETRY_EXACT_TOL = 0.0
#: 方向余弦正交归一性检查容差
DIRECTION_ORTHONORMAL_TOL = 1e-5
#: 物理覆盖检查容差（mm）：目标网格角点允许略微越界
COVERAGE_TOL_MM = 1e-3
#: 标签掩膜允许的取值
ALLOWED_LABEL_VALUES = (0.0, 1.0)

CATEGORY_LINKABLE = "linkable"
CATEGORY_GRID_MISMATCH = "grid_mismatch"
CATEGORY_FAILED = "failed"


class Prostate158Error(RuntimeError):
    """可预期的输入准备错误（fail-closed；不生成半成品）。"""


# --------------------------------------------------------------------------- 通用小工具
def _now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def _publish_json(path: Path, payload: dict) -> None:
    """原子发布 JSON；目标已存在则拒绝覆盖（fail-closed）。"""
    if path.exists():
        raise Prostate158Error(f"输出文件已存在，拒绝静默覆盖: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False, allow_nan=False)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _real(path: Path) -> Path:
    return Path(os.path.realpath(str(path)))


def _is_within(child: Path, root: Path) -> bool:
    """realpath 之后判断 ``child`` 是否仍在 ``root`` 内（防止符号链接越界）。"""
    c, r = _real(child), _real(root)
    try:
        return c == r or r in c.parents
    except (OSError, ValueError):
        return False


def _case_id(queue: str, original_id: str) -> str:
    """稳定、唯一、可反查队列与原始 ID 的派生病例 ID。"""
    return f"P158_{queue}_{int(original_id):03d}"


# --------------------------------------------------------------------------- 几何（纯函数）
def geometry_of(img) -> dict:
    return {
        "size": tuple(int(v) for v in img.GetSize()),
        "spacing": tuple(float(v) for v in img.GetSpacing()),
        "origin": tuple(float(v) for v in img.GetOrigin()),
        "direction": tuple(float(v) for v in img.GetDirection()),
    }


def _same_tuple(a: tuple[float, ...], b: tuple[float, ...], tol: float) -> bool:
    return len(a) == len(b) and all(
        abs(float(x) - float(y)) <= tol for x, y in zip(a, b)
    )


def same_geometry(g1: dict, g2: dict, tol: float = GEOMETRY_EXACT_TOL) -> bool:
    return all(
        _same_tuple(g1[k], g2[k], tol)
        for k in ("size", "spacing", "origin", "direction")
    )


def direction_matrix(geom: dict) -> np.ndarray:
    return np.asarray(geom["direction"], dtype=float).reshape(3, 3)


def orthonormal_direction(geom: dict, tol: float = DIRECTION_ORTHONORMAL_TOL) -> bool:
    """方向矩阵必须是合法的三维正交归一基（det = ±1、MᵀM = I）。"""
    m = direction_matrix(geom)
    if m.shape != (3, 3) or not np.isfinite(m).all():
        return False
    if not all(math.isfinite(s) and s > 0.0 for s in geom["spacing"]):
        return False
    if not all(math.isfinite(o) for o in geom["origin"]):
        return False
    return bool(np.allclose(m.T @ m, np.eye(3), atol=tol)) and bool(
        abs(abs(np.linalg.det(m)) - 1.0) < tol
    )


def _fov_physical_corners(geom: dict) -> list[tuple[float, float, float]]:
    """网格 8 个角点在物理坐标系中的位置（mm）。"""
    spacing = np.asarray(geom["spacing"], dtype=float)
    n = np.asarray(geom["size"], dtype=float)
    origin = np.asarray(geom["origin"], dtype=float)
    axes = direction_matrix(geom) @ np.diag(spacing)  # 每列是一个体素轴向量
    corners = []
    for bits in range(8):
        k = np.array([(bits >> a) & 1 for a in range(3)], dtype=float) * (n - 1.0)
        corners.append(tuple(float(v) for v in (origin + axes @ k)))
    return corners


def fov_covers(
    source_geom: dict, target_geom: dict, tol_mm: float = COVERAGE_TOL_MM
) -> bool:
    """``source`` 网格的物理 FOV 是否完整覆盖 ``target`` 网格的 8 个物理角点。

    要求两者方向矩阵一致（同一物理坐标系），然后把 target 角点变换到 source 的连续索引，
    角点必须落在 ``[0, N-1]`` 内（允许 ``tol_mm`` 折算后的越界）。
    """
    if not (orthonormal_direction(source_geom) and orthonormal_direction(target_geom)):
        return False
    if not np.allclose(
        direction_matrix(source_geom),
        direction_matrix(target_geom),
        atol=DIRECTION_ORTHONORMAL_TOL,
    ):
        return False
    s_sp = np.asarray(source_geom["spacing"], dtype=float)
    tol_idx = tol_mm / np.maximum(s_sp, 1e-12)
    rot = direction_matrix(source_geom)
    origin = np.asarray(source_geom["origin"], dtype=float)
    n = np.asarray(source_geom["size"], dtype=float)
    for p in _fov_physical_corners(target_geom):
        idx = rot.T @ (np.asarray(p, dtype=float) - origin) / s_sp
        if np.any(~np.isfinite(idx)):
            return False
        if np.any(idx < -tol_idx) or np.any(idx > (n - 1) + tol_idx):
            return False
    return True


def geometry_mismatch_fields(g1: dict, g2: dict) -> list[str]:
    return [
        k
        for k in ("size", "spacing", "origin", "direction")
        if not _same_tuple(g1[k], g2[k], 0.0)
    ]


# --------------------------------------------------------------------------- CSV 解析
def _resolve_csv_path(
    csv_path: Path, raw: str, allow_root: Path, errors: list[str]
) -> Path | None:
    """把 CSV 单元格解析为 allow_root 内的真实文件路径；任何越界 / 绝对路径都 fail-closed。"""
    value = (raw or "").strip()
    if value == "":
        return None
    p = Path(value)
    if p.is_absolute():
        errors.append(f"CSV 路径不允许是绝对路径: {value!r}")
        return None
    if ".." in p.parts:
        errors.append(f"CSV 路径不允许包含 '..' 越界分量: {value!r}")
        return None
    resolved = (csv_path.parent / p).resolve(strict=False)
    if not _is_within(resolved, allow_root):
        errors.append(f"CSV 路径解析后越出允许根目录: {value!r} -> {resolved}")
        return None
    return resolved


def parse_csv(
    csv_path: Path, queue: str, allow_root: Path
) -> tuple[list[dict], list[str]]:
    """解析并做**字段级**校验（不读影像）。返回 (rows, errors)。"""
    errors: list[str] = []
    if queue not in QUEUES:
        raise Prostate158Error(f"未知队列 {queue!r}，允许：{QUEUES}")
    if not csv_path.is_file():
        raise Prostate158Error(f"CSV 不存在: {csv_path}")
    try:
        with open(csv_path, newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            fieldnames = reader.fieldnames or []
            raw_rows = list(reader)
    except (OSError, csv.Error) as exc:
        raise Prostate158Error(f"CSV 无法读取: {csv_path}: {exc}") from exc

    missing = [c for c in REQUIRED_COLUMNS if c not in fieldnames]
    if missing:
        raise Prostate158Error(f"CSV 缺少必需列: {missing}（实际列: {fieldnames}）")

    seen_original: set[str] = set()
    seen_case: set[str] = set()
    rows: list[dict] = []
    for i, raw in enumerate(raw_rows):
        row_no = i + 2  # 含表头的 1-based 行号
        row_errors: list[str] = []
        original = (raw.get("ID") or "").strip()
        if not original:
            row_errors.append("ID 为空")
            cid = None
        else:
            try:
                cid = _case_id(queue, original)
            except ValueError:
                row_errors.append(f"ID 不是整数: {original!r}")
                cid = None
            if original in seen_original:
                row_errors.append(f"原始 ID 重复: {original!r}")
            if cid is not None and cid in seen_case:
                row_errors.append(f"派生病例 ID 重复: {cid!r}")
            if cid is not None:
                seen_original.add(original)
                seen_case.add(cid)

        channels: dict[str, Any] = {}
        for _, column, _ in CHANNELS:
            resolved = _resolve_csv_path(
                csv_path, raw.get(column, ""), allow_root, row_errors
            )
            channels[column] = {
                "column": column,
                "raw": (raw.get(column) or "").strip(),
                "path": str(resolved) if resolved is not None else None,
                "field_present": (raw.get(column) or "").strip() != "",
            }
            if not (raw.get(column) or "").strip():
                row_errors.append(f"通道列 {column} 为空")

        primary_raw = (raw.get(PRIMARY_READER_COLUMN) or "").strip()
        if not primary_raw:
            # 空字段不是阴性；阴性必须指向经验证为空的真实掩膜文件
            row_errors.append(
                f"主参考列 {PRIMARY_READER_COLUMN} 为空（空字段不是阴性）"
            )
            primary_path = None
        else:
            primary_path = _resolve_csv_path(
                csv_path, primary_raw, allow_root, row_errors
            )

        reader2_raw = (raw.get(READER2_COLUMN) or "").strip()
        reader2_path = (
            _resolve_csv_path(csv_path, reader2_raw, allow_root, row_errors)
            if reader2_raw
            else None
        )

        if row_errors:
            errors.append(
                f"[CSV 第 {row_no} 行 / ID={original or '?'}] " + "; ".join(row_errors)
            )
        rows.append(
            {
                "row_number": row_no,
                "queue": queue,
                "original_id": original,
                "case_id": cid,
                "channels": channels,
                "primary_reference": {
                    "column": PRIMARY_READER_COLUMN,
                    "raw": primary_raw,
                    "path": str(primary_path) if primary_path is not None else None,
                    "field_present": primary_raw != "",
                },
                "reader2_reference": {
                    "column": READER2_COLUMN,
                    "raw": reader2_raw,
                    "path": str(reader2_path) if reader2_path is not None else None,
                    "field_present": reader2_raw != "",
                },
                "csv_errors": row_errors,
            }
        )
    return rows, errors


# --------------------------------------------------------------------------- 影像 / 标签读取
def _read_image(path: Path):
    import SimpleITK as sitk

    return sitk.ReadImage(str(path))


def read_label(path: Path) -> dict:
    """读取二值标签并校验；返回几何、唯一取值、阳性体素数。标签必须是 0/1 且全部有限。"""
    img = _read_image(path)
    arr = sitk_get_array(img)
    if not bool(np.isfinite(arr).all()):
        raise Prostate158Error("标签含非有限值")
    values = sorted(float(v) for v in np.unique(arr))
    bad = [v for v in values if v not in ALLOWED_LABEL_VALUES]
    if bad:
        raise Prostate158Error(f"标签含非 0/1 取值（前 5 个）: {bad[:5]}")
    return {
        "geometry": geometry_of(img),
        "label_values": values,
        "positive_voxels": int(np.count_nonzero(arr > 0.5)),
    }


def sitk_get_array(img):
    import SimpleITK as sitk

    return sitk.GetArrayFromImage(img)


def _label_status(path_str: str | None, field_present: bool) -> dict:
    """读取一个标签引用并给出 positive / negative / absent / invalid 状态。"""
    if not field_present or not path_str:
        return {"status": "absent", "path": path_str, "reason": "CSV 字段为空"}
    path = Path(path_str)
    if not path.is_file():
        return {"status": "invalid", "path": path_str, "reason": "文件不存在"}
    try:
        info = read_label(path)
    except Exception as exc:  # noqa: BLE001 - 第三方读取异常也要收敛成 invalid 并带原因
        return {
            "status": "invalid",
            "path": path_str,
            "reason": f"{type(exc).__name__}: {exc}",
        }
    status = "positive" if info["positive_voxels"] > 0 else "negative"
    return {
        "status": status,
        "path": str(path),
        "geometry": info["geometry"],
        "label_values": info["label_values"],
        "positive_voxels": info["positive_voxels"],
        "explicit_empty_mask": path.name == "empty.nii.gz",
    }


# --------------------------------------------------------------------------- audit 核心
def audit_case(row: dict) -> dict:
    """审计单例；任何异常都收敛到记录的 errors 中（不抛到逐例循环外）。"""
    cid = row["case_id"] or f"row{row['row_number']}"
    record: dict[str, Any] = {
        "case_id": cid,
        "queue": row["queue"],
        "original_id": row["original_id"],
        "csv_row_number": row["row_number"],
        "channels": {},
        "primary_reference": dict(row["primary_reference"]),
        "reader2_reference": dict(row["reader2_reference"]),
        "category": CATEGORY_FAILED,
        "resample_candidate": False,
        "resample_blockers": [],
        "errors": [],
    }
    # CSV 字段级错误（越界 / 重复 ID / 空字段等）直接判该例失败，不再继续读影像
    if row.get("csv_errors"):
        record["errors"].extend(row["csv_errors"])
        return record
    # 字段级错误在 parse 阶段无法带 case_id 落进记录，这里按路径缺失/越界再核一遍存在性
    channel_geoms: dict[str, dict] = {}
    for _, column, domain in CHANNELS:
        spec = row["channels"][column]
        entry = {
            "column": column,
            "domain": domain,
            "path": spec["path"],
            "field_present": spec["field_present"],
            "readable": False,
            "geometry": None,
        }
        if not spec["field_present"]:
            record["errors"].append(f"通道 {column} 字段为空")
        elif not spec["path"]:
            record["errors"].append(f"通道 {column} 路径越界或非法")
        elif not Path(spec["path"]).is_file():
            record["errors"].append(f"通道 {column} 文件不存在: {spec['path']}")
        else:
            try:
                img = _read_image(Path(spec["path"]))
                entry["readable"] = True
                entry["geometry"] = geometry_of(img)
                channel_geoms[column] = entry["geometry"]
            except Exception as exc:  # noqa: BLE001
                record["errors"].append(
                    f"通道 {column} 不可读: {type(exc).__name__}: {exc}"
                )
        record["channels"][column] = entry

    primary = _label_status(
        row["primary_reference"]["path"], row["primary_reference"]["field_present"]
    )
    record["primary_reference"].update(primary)
    if primary["status"] == "invalid":
        record["errors"].append(f"主参考标签无效: {primary.get('reason')}")
    elif primary["status"] == "absent":
        record["errors"].append("主参考标签缺失（空字段不得当作阴性）")

    reader2 = _label_status(
        row["reader2_reference"]["path"], row["reader2_reference"]["field_present"]
    )
    record["reader2_reference"].update(reader2)
    if reader2["status"] == "invalid":
        # reader2 是可选项：invalid 只记录，不让整例失败，但读者敏感性分析不得包含该例
        record["reader2_reference"]["note"] = reader2.get("reason")

    if record["errors"]:
        return record

    # 三通道网格关系
    t2_geom = channel_geoms["t2"]
    mismatched = {
        col: geometry_mismatch_fields(t2_geom, channel_geoms[col])
        for col in ("adc", "dwi")
    }
    mismatched = {col: fields for col, fields in mismatched.items() if fields}
    record["channels_mismatch_fields"] = mismatched
    label_geom = primary["geometry"]
    record["primary_label_matches_t2_grid"] = same_geometry(label_geom, t2_geom)
    record["primary_label_mismatch_fields"] = geometry_mismatch_fields(
        label_geom, t2_geom
    )
    if reader2["status"] in ("positive", "negative"):
        record["reader2_reference"]["label_matches_t2_grid"] = same_geometry(
            reader2["geometry"], t2_geom
        )
        record["reader2_reference"]["label_mismatch_fields"] = geometry_mismatch_fields(
            reader2["geometry"], t2_geom
        )

    if not mismatched:
        record["category"] = CATEGORY_LINKABLE
        return record

    record["category"] = CATEGORY_GRID_MISMATCH
    blockers: list[str] = []
    for col in ("adc", "dwi"):
        if col not in mismatched:
            continue
        g = channel_geoms[col]
        if not orthonormal_direction(g):
            blockers.append(f"{col} 方向矩阵不是合法正交归一基或 spacing/origin 非法")
        elif not np.allclose(
            direction_matrix(g),
            direction_matrix(t2_geom),
            atol=DIRECTION_ORTHONORMAL_TOL,
        ):
            blockers.append(
                f"{col} 与 T2 不在同一物理坐标系（direction 不同），不做配准"
            )
        elif not fov_covers(g, t2_geom):
            blockers.append(f"{col} 的物理 FOV 未完整覆盖 T2 参考网格")
    record["resample_blockers"] = blockers
    record["resample_candidate"] = not blockers
    return record


def run_audit(rows: list[dict], progress: bool) -> list[dict]:
    iterator = rows
    if progress:
        from tqdm import tqdm

        iterator = tqdm(
            rows, desc="audit", unit="case", mininterval=1.0, file=sys.stdout
        )
    return [audit_case(row) for row in iterator]


def _records_by_category(records: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {
        CATEGORY_LINKABLE: [],
        CATEGORY_GRID_MISMATCH: [],
        CATEGORY_FAILED: [],
    }
    for r in records:
        out[r["category"]].append(r)
    return out


def _audit_payload(
    csv_path: Path,
    queue: str,
    allow_root: Path,
    records: list[dict],
    parse_errors: list[str],
    elapsed_s: float,
) -> dict:
    by_cat = _records_by_category(records)
    primary_pos = [
        r for r in records if r["primary_reference"].get("status") == "positive"
    ]
    primary_neg = [
        r for r in records if r["primary_reference"].get("status") == "negative"
    ]
    reader2_available = [
        r
        for r in records
        if r["reader2_reference"].get("status") in ("positive", "negative")
    ]
    return _json_safe(
        {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": _now_utc(),
            "kind": "prostate158_external_audit",
            "queue": queue,
            "csv": str(csv_path),
            "allowed_root": str(allow_root),
            "domain_caveat": (
                "通道 0002 为 Prostate158 DWI，仅作为原 Dataset605 模型第三通道（训练时为 HBV）"
                "的跨域输入；不宣称二者完全等价。"
            ),
            "primary_reference_column": PRIMARY_READER_COLUMN,
            "reader2_reference_column": READER2_COLUMN,
            "counts": {
                "total": len(records),
                "linkable": len(by_cat[CATEGORY_LINKABLE]),
                "grid_mismatch": len(by_cat[CATEGORY_GRID_MISMATCH]),
                "grid_mismatch_resample_candidate": sum(
                    1 for r in by_cat[CATEGORY_GRID_MISMATCH] if r["resample_candidate"]
                ),
                "failed": len(by_cat[CATEGORY_FAILED]),
                "primary_positive": len(primary_pos),
                "primary_negative_verified_empty": len(primary_neg),
                "reader2_available": len(reader2_available),
                "csv_field_errors": len(parse_errors),
            },
            "elapsed_s": round(elapsed_s, 3),
            "csv_field_errors": parse_errors,
            "cases": records,
        }
    )


def _finish_summary(
    tag: str, t0: float, lines_counts: dict, output: str | None
) -> None:
    print("=== Prostate158 外测准备结束汇总 ===")
    print(f"  phase   : {tag}")
    for key, value in lines_counts.items():
        print(f"  {key:<9}: {value}")
    print(f"  elapsed : {time.monotonic() - t0:.2f} s")
    print(f"  output  : {output if output else '(未写出)'}")


# --------------------------------------------------------------------------- prepare
def _verify_symlink_target_within(target: Path, allow_root: Path) -> Path:
    """创建前校验：目标是 allow_root 内的现有文件，返回其 realpath。"""
    real_target = _real(target)
    if not real_target.is_file():
        raise Prostate158Error(f"软链接目标不是现有文件: {real_target}")
    if not _is_within(real_target, allow_root):
        raise Prostate158Error(f"软链接目标越出允许根目录: {real_target}")
    return real_target


def _verify_created_symlink(link: Path, real_target: Path) -> None:
    """创建后校验：链接必须是符号链接且解析到期望目标。"""
    if not link.is_symlink():
        raise Prostate158Error(f"预期为符号链接但不是: {link}")
    real_link = _real(link)
    if not real_link.is_file():
        raise Prostate158Error(f"符号链接悬空（目标不存在）: {link} -> {real_link}")
    if real_link != real_target:
        raise Prostate158Error(
            f"软链接校验失败: {link} -> {real_link}（期望 {real_target}）"
        )


def _make_symlink(link: Path, target: Path, allow_root: Path) -> None:
    real_target = _verify_symlink_target_within(target, allow_root)
    link.parent.mkdir(parents=True, exist_ok=True)
    rel = os.path.relpath(real_target, link.parent)
    os.symlink(rel, link)
    _verify_created_symlink(link, real_target)


def resample_intensity_to_t2(source_path: Path, t2_img) -> tuple[object, dict]:
    """把派生强度图以线性插值重采样到 T2 参考网格；返回 (image, 插值记录)。

    只允许在同一物理坐标系、源 FOV 覆盖目标网格时调用（调用方负责先用几何函数确认）。
    """
    import SimpleITK as sitk

    source = _read_image(source_path)
    flt = sitk.ResampleImageFilter()
    flt.SetReferenceImage(t2_img)
    flt.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    flt.SetInterpolator(sitk.sitkLinear)
    flt.SetDefaultPixelValue(0.0)
    flt.SetOutputPixelType(source.GetPixelID())
    out = flt.Execute(source)
    meta = {
        "interpolator": "SimpleITK.sitkLinear（连续值强度图）",
        "transform": "identity（仅几何重采样，不做任何图像配准）",
        "source_geometry": geometry_of(source),
        "target_geometry": geometry_of(t2_img),
        "output_geometry": geometry_of(out),
        "default_pixel_value": 0.0,
        "note": "派生影像写入独立外测目录；原始 Prostate158 影像未被修改",
    }
    return out, meta


def _manifest_id(manifest_core: dict) -> str:
    blob = json.dumps(manifest_core, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def _plan_cases(
    records: list[dict], allow_resample: bool
) -> tuple[list[dict], list[dict], list[dict]]:
    """把审计记录分为 (materialize, skipped, failures)。"""
    materialize, skipped, failures = [], [], []
    for r in records:
        if r["category"] == CATEGORY_LINKABLE:
            materialize.append({**r, "action": "symlink"})
        elif r["category"] == CATEGORY_GRID_MISMATCH:
            if allow_resample and r["resample_candidate"]:
                materialize.append({**r, "action": "resample"})
            elif allow_resample:
                failures.append(
                    {
                        **r,
                        "reason": "网格不同且不满足可信重采样条件: "
                        + "; ".join(r["resample_blockers"]),
                    }
                )
            else:
                skipped.append(
                    {
                        **r,
                        "reason": "三通道网格不同；默认不重采样（需要显式 --allow-resample）",
                    }
                )
        else:
            failures.append({**r, "reason": "审计失败: " + "; ".join(r["errors"])})
    return materialize, skipped, failures


def _materialize_case(plan: dict, images_dir: Path, allow_root: Path) -> dict:
    """为一例建立软链接 / 派生重采样文件；返回写入 manifest 的通道记录。"""
    import SimpleITK as sitk

    cid = plan["case_id"]
    out_channels: dict[str, dict] = {}
    t2_path = Path(plan["channels"]["t2"]["path"])
    t2_img = _read_image(t2_path)  # T2 恒为参考网格；其本身只软链接
    for channel_id, column, _ in CHANNELS:
        link = images_dir / f"{cid}_{channel_id}.nii.gz"
        src = Path(plan["channels"][column]["path"])
        if (
            column == "t2"
            or plan["action"] == "symlink"
            or column not in ("adc", "dwi")
        ):
            _make_symlink(link, src, allow_root)
            out_channels[column] = {
                "kind": "symlink",
                "source_path": str(_real(src)),
                "input_file": str(link),
                "geometry": plan["channels"][column]["geometry"],
            }
        else:
            # 仅 adc/dwi 在显式 --allow-resample 且该例 resample_candidate 时走到这里
            if not plan["resample_candidate"]:
                raise Prostate158Error(
                    f"{cid}: 内部错误：非 resample_candidate 病例进入重采样分支"
                )
            derived, meta = resample_intensity_to_t2(src, t2_img)
            if not same_geometry(meta["output_geometry"], meta["target_geometry"]):
                raise Prostate158Error(f"{cid}: 重采样后几何与 T2 参考网格不一致")
            sitk.WriteImage(derived, str(link), useCompression=True)
            if not link.is_file() or link.is_symlink():
                raise Prostate158Error(f"{cid}: 派生影像写入异常: {link}")
            out_channels[column] = {
                "kind": "derived_resampled",
                "source_path": str(_real(src)),
                "input_file": str(link),
                "resample": meta,
                "geometry": meta["output_geometry"],
            }
    return out_channels


def run_prepare(
    records: list[dict],
    parse_errors: list[str],
    args: argparse.Namespace,
    csv_path: Path,
    allow_root: Path,
    t0: float,
) -> None:
    if parse_errors:
        raise Prostate158Error(
            f"CSV 字段级错误 {len(parse_errors)} 项，拒绝 prepare：\n"
            + "\n".join(f"  {e}" for e in parse_errors[:50])
        )
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise Prostate158Error(
            f"输出目录已存在且非空，默认拒绝覆盖: {output_dir}（请使用新的独立目录）"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.staging-{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)

    materialize, skipped, failures = _plan_cases(
        records, allow_resample=args.allow_resample
    )
    if failures:
        _finish_summary(
            "prepare(preflight 失败，未写任何输入)",
            t0,
            {
                "total": len(records),
                "materialize": len(materialize),
                "skipped": len(skipped),
                "failed": len(failures),
            },
            str(output_dir),
        )
        raise Prostate158Error(
            f"存在 {len(failures)} 个不可处理病例，未生成任何输入：\n"
            + "\n".join(f"  {f['case_id']}: {f['reason']}" for f in failures[:50])
        )
    if not materialize:
        raise Prostate158Error(
            "没有可物化的病例（全部 skipped / failed），拒绝生成空目录"
        )

    images_dir = staging / IMAGES_DIRNAME
    images_dir.mkdir(parents=True)
    manifest_cases: list[dict] = []
    try:
        iterator = materialize
        if not args.no_progress:
            from tqdm import tqdm

            iterator = tqdm(
                materialize,
                desc="prepare",
                unit="case",
                mininterval=1.0,
                file=sys.stdout,
            )
        for plan in iterator:
            out_channels = _materialize_case(plan, images_dir, allow_root)
            p_ref = plan["primary_reference"]
            r2 = plan["reader2_reference"]
            manifest_cases.append(
                {
                    "case_id": plan["case_id"],
                    "queue": plan["queue"],
                    "original_id": plan["original_id"],
                    "csv_row_number": plan["csv_row_number"],
                    "action": plan["action"],
                    "channels": out_channels,
                    "primary_reference": {
                        "column": PRIMARY_READER_COLUMN,
                        "path": p_ref.get("path"),
                        "status": p_ref.get("status"),
                        "positive_voxels": p_ref.get("positive_voxels"),
                        "geometry": p_ref.get("geometry"),
                        "label_matches_t2_grid": plan.get(
                            "primary_label_matches_t2_grid"
                        ),
                        "label_mismatch_fields": plan.get(
                            "primary_label_mismatch_fields"
                        ),
                    },
                    "reader2_reference": {
                        "column": READER2_COLUMN,
                        "status": r2.get("status"),
                        "path": r2.get("path")
                        if r2.get("status") in ("positive", "negative")
                        else None,
                        "positive_voxels": r2.get("positive_voxels"),
                        "geometry": r2.get("geometry"),
                        "label_matches_t2_grid": r2.get("label_matches_t2_grid"),
                    },
                }
            )
        manifest_core = {
            "schema_version": SCHEMA_VERSION,
            "queue": args.queue,
            "csv": str(csv_path),
            "allowed_root": str(allow_root),
            "allow_resample": bool(args.allow_resample),
            "cases": manifest_cases,
        }
        manifest = {
            "schema_version": manifest_core["schema_version"],
            "manifest_id": _manifest_id(manifest_core),
            "created_at_utc": _now_utc(),
            "kind": "prostate158_external_input_manifest",
            "queue": manifest_core["queue"],
            "csv": manifest_core["csv"],
            "allowed_root": manifest_core["allowed_root"],
            "allow_resample": manifest_core["allow_resample"],
            "input_images_dir": IMAGES_DIRNAME,
            "channel_mapping": [
                {"nnunet_channel": ch, "csv_column": col, "domain": dom}
                for ch, col, dom in CHANNELS
            ],
            "domain_caveat": (
                "通道 0002 为 Prostate158 DWI，仅作为原 Dataset605 模型第三通道（训练时为 HBV）"
                "的跨域输入；不宣称二者完全等价。"
            ),
            "negative_definition": (
                "阴性 = adc_tumor_reader1 字段非空、文件可读、标签合法且经验证为空掩膜；"
                "CSV 空字段 / 缺文件 / 读取失败一律 fail-closed，不是阴性"
            ),
            "skipped_cases": [
                {"case_id": s["case_id"], "reason": s["reason"]} for s in skipped
            ],
            "cases": manifest_cases,
        }
        _publish_json(staging / MANIFEST_NAME, manifest)
        os.replace(staging, output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    _finish_summary(
        "prepare(完成)",
        t0,
        {
            "total": len(records),
            "materialized": len(materialize),
            "skipped": len(skipped),
            "failed": 0,
            "resampled": sum(1 for p in materialize if p["action"] == "resample"),
            "symlinked": sum(1 for p in materialize if p["action"] == "symlink"),
        },
        str(output_dir),
    )
    if skipped:
        print(
            f"  注意：{len(skipped)} 例因网格差异被跳过（明细见 manifest 的 skipped_cases 与 audit）"
        )
    print(f"  清单    : {output_dir / MANIFEST_NAME}")
    print(
        f"  输入目录: {output_dir / IMAGES_DIRNAME}（供 predict_nnunet.py -i 使用；标签不在其中）"
    )


# --------------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prostate158 独立外测输入准备（只读 audit / 软链接 prepare；原始影像绝不修改）",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--csv",
            required=True,
            type=Path,
            help="Prostate158 CSV（test.csv / train.csv / valid.csv）；单元格路径相对该 CSV 解析",
        )
        p.add_argument(
            "--queue",
            required=True,
            choices=QUEUES,
            help="队列名（official test=test；原作者训练/验证=train/valid），必须显式指定",
        )
        p.add_argument(
            "--allow-root",
            type=Path,
            default=None,
            help="允许链接/解析的根目录（默认 CSV 所在目录）；解析结果越界即拒绝",
        )
        p.add_argument("--no-progress", action="store_true", help="关闭 tqdm 进度条")

    p_audit = sub.add_parser("audit", help="只读审计，不生成推理输入")
    add_common(p_audit)
    p_audit.add_argument(
        "--output",
        type=Path,
        default=None,
        help="审计报告 JSON（可选；已存在则拒绝覆盖）",
    )

    p_prep = sub.add_parser(
        "prepare", help="在独立新目录生成 nnU-Net 推理输入（默认仅软链接）"
    )
    add_common(p_prep)
    p_prep.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="独立新输出目录（已存在且非空则拒绝；不含参考标签）",
    )
    p_prep.add_argument(
        "--allow-resample",
        action="store_true",
        help="默认关闭：仅对物理关系可信且覆盖 T2 网格的 adc/dwi 做显式派生重采样；"
        "绝不做图像配准",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    csv_path = Path(args.csv).resolve()
    allow_root = Path(args.allow_root).resolve() if args.allow_root else csv_path.parent
    t0 = time.monotonic()
    try:
        rows, parse_errors = parse_csv(csv_path, args.queue, allow_root)
        records = run_audit(rows, progress=not args.no_progress)
        if args.command == "audit":
            payload = _audit_payload(
                csv_path,
                args.queue,
                allow_root,
                records,
                parse_errors,
                time.monotonic() - t0,
            )
            by_cat = _records_by_category(records)
            output = str(args.output) if args.output else None
            if args.output is not None:
                _publish_json(args.output, payload)
            _finish_summary(
                "audit(只读)",
                t0,
                {
                    "total": len(records),
                    "linkable": len(by_cat[CATEGORY_LINKABLE]),
                    "grid_mismatch": len(by_cat[CATEGORY_GRID_MISMATCH]),
                    "resamp-cand": sum(
                        1
                        for r in by_cat[CATEGORY_GRID_MISMATCH]
                        if r["resample_candidate"]
                    ),
                    "failed": len(by_cat[CATEGORY_FAILED]),
                },
                output,
            )
            hard_failures = len(by_cat[CATEGORY_FAILED])
            if hard_failures:
                print(f"  错误：{hard_failures} 例/行未通过审计（明细见审计结果）")
                return 2
            return 0
        run_prepare(records, parse_errors, args, csv_path, allow_root, t0)
        return 0
    except Prostate158Error as exc:
        print(
            f"[prepare_prostate158_external] 失败（fail-closed）: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
