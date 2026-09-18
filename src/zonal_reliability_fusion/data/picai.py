"""PI-CAI 模型就绪数据审计的共享数据访问、几何与标签工具。

只读原始数据；所有派生结果由调用方写入既定目录。所有空间计算均基于
SimpleITK 的 index-to-physical-point 变换，不使用 size*spacing 简化。
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import SimpleITK as sitk

DEFAULT_DATA_ROOT = Path("/opt/data/private/lm/data/Prostate/PI-CAI")

# 第一阶段使用的三序列（不使用 cor/sag）
SEQUENCES: tuple[str, ...] = ("t2w", "adc", "hbv")

# 标签相对子目录
LESION_RESAMPLED = "labels/csPCa_lesion_delineations/human_expert/resampled"
LESION_POOCH25 = "labels/csPCa_lesion_delineations/human_expert/Pooch25"
WG_BOSMA22B = "labels/anatomical_delineations/whole_gland/AI/Bosma22b"
WG_GUERBET23 = "labels/anatomical_delineations/whole_gland/AI/Guerbet23"
ZONAL_YUAN23 = "labels/anatomical_delineations/zonal_pz_tz/AI/Yuan23"
ZONAL_HEVIAI23 = "labels/anatomical_delineations/zonal_pz_tz/AI/HeviAI23"
MARKSHEET_REL = "labels/clinical_information/marksheet.csv"

# canonical 病灶标签来源名
SRC_RESAMPLED = "human_expert/resampled"
SRC_POOCH25 = "human_expert/Pooch25"

GEOM_RTOL = 1e-3
GEOM_ATOL = 1e-3  # mm

# 几何状态分类
GRID_EXACT = "exact_same_grid"
GRID_DIFFERENT = "same_physical_space_different_grid"
GRID_PARTIAL = "partial_physical_overlap"
GRID_SUSPECT = "geometry_suspect"
GRID_MISSING = "missing_or_unreadable"


# --------------------------------------------------------------------------- #
# 路径与标识
# --------------------------------------------------------------------------- #
def case_id(patient_id, study_id) -> str:
    return f"{patient_id}_{study_id}"


def image_path(root: Path, patient_id, study_id, seq: str) -> Path:
    return Path(root) / "images" / str(patient_id) / f"{case_id(patient_id, study_id)}_{seq}.mha"


def label_path(root: Path, subdir: str, patient_id, study_id) -> Path:
    return Path(root) / subdir / f"{case_id(patient_id, study_id)}.nii.gz"


def load_marksheet(root: Path) -> list[dict]:
    """读取 marksheet.csv，返回按 (patient_id, study_id) 的字典列表。"""
    p = Path(root) / MARKSHEET_REL
    if not p.exists():
        raise FileNotFoundError(f"marksheet 不存在: {p}")
    with p.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    need = {"patient_id", "study_id", "center", "case_csPCa", "case_ISUP"}
    missing = need - set(rows[0].keys()) if rows else need
    if missing:
        raise ValueError(f"marksheet 缺少列: {missing}")
    return rows


# --------------------------------------------------------------------------- #
# 几何：size/spacing/origin/direction + 物理角点/bbox/extent/orientation
# --------------------------------------------------------------------------- #
@dataclass
class Geometry:
    size: tuple           # (x, y, z) 体素数
    spacing: tuple        # (sx, sy, sz) mm
    origin: tuple         # (ox, oy, oz) mm
    direction: tuple      # 9 元组（3x3 行优先）
    path: str = field(default="")

    @property
    def ndim(self) -> int:
        return len(self.size)


def read_geometry(path: Path | str) -> Geometry:
    """只读头信息，不加载体素；失败抛出异常（由调用方记录，不静默跳过）。"""
    path = str(path)
    r = sitk.ImageFileReader()
    r.SetFileName(path)
    r.ReadImageInformation()
    return Geometry(
        size=tuple(int(v) for v in r.GetSize()),
        spacing=tuple(float(v) for v in r.GetSpacing()),
        origin=tuple(float(v) for v in r.GetOrigin()),
        direction=tuple(float(v) for v in r.GetDirection()),
        path=path,
    )


def image_from_geometry(geo: Geometry, pixel_id: int = sitk.sitkFloat32) -> sitk.Image:
    """由几何信息构造空 SimpleITK 图像（不读体素），用于物理变换。"""
    img = sitk.Image([int(v) for v in geo.size], pixel_id, 1)
    img.SetSpacing([float(v) for v in geo.spacing])
    img.SetOrigin([float(v) for v in geo.origin])
    img.SetDirection([float(v) for v in geo.direction])
    return img


def physical_corners(geo: Geometry) -> np.ndarray:
    """8 个图像角点的物理坐标（SimpleITK index-to-physical-point），返回 (8,3)。"""
    img = image_from_geometry(geo)
    sx, sy, sz = geo.size
    corners = []
    for ix in (0, sx - 1):
        for iy in (0, sy - 1):
            for iz in (0, sz - 1):
                corners.append(img.TransformIndexToPhysicalPoint((ix, iy, iz)))
    return np.asarray(corners, dtype=float)  # (8,3)


def physical_bbox(geo: Geometry) -> tuple[np.ndarray, np.ndarray]:
    """返回 (mins, maxs)，即物理包围盒在 x/y/z 上的范围。"""
    c = physical_corners(geo)
    return c.min(axis=0), c.max(axis=0)


def physical_extent(geo: Geometry) -> np.ndarray:
    """物理包围盒各轴长度（max-min），非 size*spacing 简化。"""
    mins, maxs = physical_bbox(geo)
    return maxs - mins


def orientation_code(geo: Geometry) -> str:
    """由 direction 余弦返回 3 字符方向编码（SimpleITK/ITK 采用 DICOM LPS 物理坐标约定）。

    使用 SimpleITK 官方接口 DICOMOrientImageFilter_GetOrientationFromDirectionCosines：
    单位 direction 对应 "LPS"（旧实现按 RAS 约定手工解释单位阵，已修正）。
    仅用于方向字符串报告，不对影像做任何重定向或翻转。
    """
    d = np.asarray(geo.direction, dtype=float).ravel()
    if d.size != 9:
        raise ValueError(f"direction 应为 9 个分量，实际 {d.size}")
    return sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(d.tolist())


# --------------------------------------------------------------------------- #
# 网格关系
# --------------------------------------------------------------------------- #
def grids_equal(a: Geometry, b: Geometry,
                atol: float = GEOM_ATOL) -> bool:
    """是否完全同一网格：size 相等且 spacing/origin/direction 在容差内一致。"""
    if a.size != b.size:
        return False
    for fa, fb in (a.spacing, b.spacing), (a.origin, b.origin), (a.direction, b.direction):
        if not np.allclose(np.asarray(fa, float), np.asarray(fb, float), atol=atol):
            return False
    return True


def axes_overlap(a: Geometry, b: Geometry) -> tuple[np.ndarray, np.ndarray]:
    """每轴 (重叠长度, 重叠长度/a轴长)。基于物理包围盒。"""
    amin, amax = physical_bbox(a)
    bmin, bmax = physical_bbox(b)
    lo = np.maximum(amin, bmin)
    hi = np.minimum(amax, bmax)
    ov = np.maximum(0.0, hi - lo)
    ext = np.maximum(amax - amin, 1e-9)
    return ov, ov / ext


def classify_grid(ref: Geometry, other: Geometry,
                  full_atol: float = GEOM_ATOL,
                  overlap_thresh: float = 0.99) -> str:
    """相对参考网格 ref，对 other 做五类几何状态分类。"""
    if grids_equal(ref, other, atol=full_atol):
        return GRID_EXACT
    ov, ratio = axes_overlap(ref, other)
    if np.all(ratio >= overlap_thresh):
        # 物理空间高度重合，但网格参数不完全一致
        same_space = np.allclose(
            np.asarray(ref.direction, float), np.asarray(other.direction, float), atol=1e-2
        )
        return GRID_DIFFERENT if same_space else GRID_SUSPECT
    if np.all(ov > 0):
        return GRID_PARTIAL
    return GRID_SUSPECT


# --------------------------------------------------------------------------- #
# canonical 病灶标签
# --------------------------------------------------------------------------- #
def canonical_lesion(root: Path, patient_id, study_id) -> tuple[str, Optional[Path]]:
    """canonical 病灶标签来源：存在 Pooch25 用 Pooch25，否则用 resampled。"""
    pooch = label_path(root, LESION_POOCH25, patient_id, study_id)
    if pooch.exists():
        return SRC_POOCH25, pooch
    resmp = label_path(root, LESION_RESAMPLED, patient_id, study_id)
    if resmp.exists():
        return SRC_RESAMPLED, resmp
    return SRC_RESAMPLED, None  # 缺失，由调用方记录


def map_lesion_to_binary(arr: np.ndarray, source: str) -> np.ndarray:
    """将病灶标签映射为二值前景：resampled 的 2/3/4/5 → 1；Pooch25 的 1 → 1；其余→0。"""
    a = np.asarray(arr)
    if source == SRC_POOCH25:
        return (a == 1).astype(np.uint8)
    # human_expert/resampled: ISUP 2..5 为前景
    return np.isin(a, [2, 3, 4, 5]).astype(np.uint8)


# --------------------------------------------------------------------------- #
# 图像/标签 IO 与物理重采样
# --------------------------------------------------------------------------- #
def load_array(path: Path | str) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))


def resample_to_geometry(moving_path: Path | str, ref: Geometry, is_label: bool) -> np.ndarray:
    """将 moving 按物理坐标重采样到 ref 网格。标签用最近邻，影像用线性。"""
    moving = sitk.ReadImage(str(moving_path))
    ref_img = image_from_geometry(ref, sitk.sitkFloat32)
    filt = sitk.ResampleImageFilter()
    filt.SetReferenceImage(ref_img)
    filt.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
    filt.SetDefaultPixelValue(0)
    out = filt.Execute(moving)
    return sitk.GetArrayFromImage(out)


def physical_volume_ml(mask: np.ndarray, geo: Geometry) -> float:
    """按体素体积（spacing 乘积）统计前景物理体积 mL。mask 与 geo 同网格。"""
    voxel_ml = float(np.prod(geo.spacing)) / 1000.0
    return float(np.count_nonzero(mask)) * voxel_ml


def dice_score(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).astype(bool)
    b = np.asarray(b).astype(bool)
    inter = np.count_nonzero(a & b)
    denom = a.sum() + b.sum()
    return float(2.0 * inter / denom) if denom > 0 else 0.0


# --------------------------------------------------------------------------- #
# 患者级泄漏检查
# --------------------------------------------------------------------------- #
def find_patient_leaks(splits: dict[str, Sequence[str]]) -> list[str]:
    """splits: {split_name: [patient_id,...]}。返回跨集合重复的 patient_id。"""
    seen: dict[str, str] = {}
    leaks = []
    for name, pids in splits.items():
        for p in pids:
            if p in seen and seen[p] != name:
                leaks.append(p)
            seen[p] = name
    return sorted(set(leaks))


def validate_disjoint(splits: dict[str, Sequence[str]]) -> None:
    leaks = find_patient_leaks(splits)
    if leaks:
        raise AssertionError(f"患者级泄漏: {len(leaks)} 个 patient_id 跨集合, 例: {leaks[:5]}")
