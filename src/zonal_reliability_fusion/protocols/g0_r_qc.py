"""G0-R 图像 QC 的核心逻辑：物理空间叠加图合成、landmark 位移、边界距离与指标行构造。

边界（与 `docs/protocols/G0_R_ALIGNMENT_QC.md` 一致）：

- **只做“当前物理坐标重采样下的 QC”**：ADC/HBV 仅按物理坐标重采样到 T2W 参考网格，
  **不执行、不推荐任何刚体/非刚体配准**；
- **不显示**模型预测、病灶标签或任何模型结果（盲审材料不得含这些信息）；
- **不决定** `resample-only` / `resample+rigid`；阈值与决策由研究者按协议 §5/§6 填写；
- 无可靠边界来源时输出 `UNKNOWN` + 跳过原因，**绝不伪造**距离数值；
- 本模块不做 GPU / torch 操作，不写回任何原始或物化数据。

坐标约定：landmark 使用 **SimpleITK 索引序 `(i, j, k) = (x, y, z)`**（与 ITK-SNAP / SimpleITK
中看到的一致）；spacing 为 `(sx, sy, sz)`，位移 = `||(Δi·sx, Δj·sy, Δk·sz)||`。

物化数据的序列文件名取自仓库既有约定 `scripts/data/materialize_picai.py::SEQ_FILES`
（`cases/<case_id>/t2w.nii.gz`、`adc.nii.gz`、`hbv.nii.gz`），不另造命名规则。
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..data import picai as P
from .common import ProtocolConfigError

#: 序列对 → 物化数据的 moving 序列（T2W 恒为参考）
PAIR_TO_SERIES: dict[str, str] = {"T2W-ADC": "adc", "T2W-HBV": "hbv"}
PAIRS: tuple[str, ...] = ("T2W-ADC", "T2W-HBV")

#: 物化序列文件名（来源：`scripts/data/materialize_picai.py::SEQ_FILES`）
MATERIALIZED_SERIES_FILES: dict[str, str] = {
    "t2w": "t2w.nii.gz",
    "adc": "adc.nii.gz",
    "hbv": "hbv.nii.gz",
}

#: 默认切片比例（相对 k 深度；不含病灶/腺体信息，避免引入标签线索）
DEFAULT_SLICE_FRACTIONS: tuple[float, ...] = (0.25, 0.5, 0.75)

#: 图像叠加说明（必须出现在每张图上）
RESAMPLE_ONLY_BANNER = "physical-space resample ONLY (no registration)"


# --------------------------------------------------------------------------- 抽样清单
def load_sampling_manifest(path: str | Path) -> dict:
    """读取 `sampling_manifest.json` 并做契约校验（病例清单不可改）。"""
    p = Path(path)
    if not p.is_file():
        raise ProtocolConfigError(f"sampling manifest 不存在: {p}")
    doc = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(doc, Mapping):
        raise ProtocolConfigError(f"{p} 顶层必须是 JSON 对象")
    protocol = doc.get("protocol") or {}
    if protocol.get("id") != "G0-R":
        raise ProtocolConfigError(f"sampling manifest 的 protocol.id 应为 G0-R，收到 {protocol.get('id')!r}")
    cases = ((doc.get("sampling") or {}).get("cases")) or []
    case_ids = [str(item.get("case_id")) for item in cases if isinstance(item, Mapping) and item.get("case_id")]
    if not case_ids:
        raise ProtocolConfigError("sampling manifest 中没有 cases（先运行抽样工具）")
    if len(set(case_ids)) != len(case_ids):
        raise ProtocolConfigError("sampling manifest 的 case_id 存在重复，拒绝继续（请重跑抽样）")
    pairs = tuple((doc.get("evaluation_plan") or {}).get("pairs") or PAIRS)
    unknown_pairs = [pair for pair in pairs if pair not in PAIRS]
    if unknown_pairs:
        raise ProtocolConfigError(f"sampling manifest 含不支持的序列对: {unknown_pairs}（允许 {PAIRS}）")
    return {
        "path": str(p),
        "protocol": dict(protocol),
        "case_ids": case_ids,
        "strata_by_case": {str(item["case_id"]): item.get("strata", "") for item in cases},
        "order_by_case": {str(item["case_id"]): item.get("selection_order") for item in cases},
        "pairs": list(pairs),
        "raw": dict(doc),
    }


def select_cases(manifest: Mapping[str, Any], requested: Sequence[str] | None) -> list[str]:
    """`--case-ids` 只能取抽样清单的子集；不在清单中 → 报错（不静默替换/新增）。"""
    allowed = list(manifest["case_ids"])
    if not requested:
        return allowed
    unknown = [cid for cid in requested if cid not in set(allowed)]
    if unknown:
        raise ProtocolConfigError(
            f"--case-ids 含不在 sampling manifest 中的病例: {unknown[:5]}；"
            "抽样清单不可替换、增删或重抽样（G0-R 协议 §3）"
        )
    return [cid for cid in allowed if cid in set(requested)]


# --------------------------------------------------------------------------- 病例路径
def case_dir_for(materialized_root: str | Path, case_id: str) -> Path:
    """定位病例目录；兼容两种既有写法（不猜测其它命名）：

    - 物化根（如 `data/processed/picai`）→ `<root>/cases/<case_id>`；
    - 直接指向 cases 目录（如 `data/processed/picai/cases`）→ `<root>/<case_id>`。
    """
    root = Path(materialized_root)
    nested = root / "cases" / str(case_id)
    if nested.is_dir():
        return nested
    direct = root / str(case_id)
    if direct.is_dir():
        return direct
    return nested  # 默认返回标准布局，便于报错信息指向预期位置


def resolve_series_paths(materialized_root: str | Path, case_id: str) -> dict[str, Path]:
    """解析一例的 T2W/ADC/HBV 物化路径（既有命名约定 + 实际存在性校验）。"""
    case_dir = case_dir_for(materialized_root, case_id)
    if not case_dir.is_dir():
        raise ProtocolConfigError(f"物化病例目录不存在: {case_dir}（先确认 P0B 物化已完成）")
    return {series: case_dir / fname for series, fname in MATERIALIZED_SERIES_FILES.items()}


def missing_series(paths: Mapping[str, Path]) -> list[str]:
    return [series for series, path in paths.items() if not Path(path).is_file()]


# --------------------------------------------------------------------------- 显示合成（纯 numpy）
def normalize_for_display(array: np.ndarray, *, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    """百分位裁剪归一化到 [0,1]（仅用于显示，不改变任何数据）。"""
    a = np.asarray(array, dtype=np.float32)
    lo, hi = np.percentile(a, [low, high])
    if not math.isfinite(float(lo)) or not math.isfinite(float(hi)) or hi <= lo:
        hi = lo + 1.0
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0)


def checkerboard_image(a: np.ndarray, b: np.ndarray, *, block: int = 8) -> np.ndarray:
    """棋盘格融合（a/b 必须同形状）；用于目检两序列在同一网格上的错位台阶。"""
    a2, b2 = np.asarray(a), np.asarray(b)
    if a2.shape != b2.shape:
        raise ProtocolConfigError(f"checkerboard 需要同形状输入: {a2.shape} vs {b2.shape}")
    if block < 1:
        raise ProtocolConfigError(f"checkerboard block 必须 >= 1，收到 {block}")
    h, w = a2.shape
    yy, xx = np.indices((h, w))
    mask = ((yy // block) + (xx // block)) % 2 == 0
    out = normalize_for_display(a2).copy()
    out[mask] = normalize_for_display(b2)[mask]
    return out


def edge_map(array: np.ndarray, *, percentile: float = 90.0) -> np.ndarray:
    """梯度幅值边缘（纯 numpy）；用于 edge-overlay 的双轮廓判读。"""
    mag = np.abs(np.gradient(normalize_for_display(array)))
    magnitude = np.hypot(mag[0], mag[1])
    peak = float(magnitude.max())
    threshold = float(np.percentile(magnitude, percentile))
    if not math.isfinite(threshold) or threshold <= 0:
        threshold = peak * 0.5
    return magnitude >= max(threshold, 1e-9)


def edge_overlay_image(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """灰度底图（a）+ 边缘叠加：a 红、b 绿、重合黄 → 错位时出现红绿分离（双轮廓）。"""
    base = normalize_for_display(a)
    rgb = np.stack([base, base, base], axis=-1)
    ea, eb = edge_map(a), edge_map(b)
    rgb[ea] = [1.0, 0.25, 0.25]
    rgb[eb] = [0.25, 1.0, 0.25]
    rgb[ea & eb] = [1.0, 1.0, 0.25]
    return rgb


def select_slice_indices(n_slices: int, fractions: Sequence[float] = DEFAULT_SLICE_FRACTIONS) -> list[int]:
    """按深度比例选择切片（不含病灶/腺体信息；小体积自动去重）。"""
    if n_slices < 1:
        raise ProtocolConfigError(f"z 层数必须 >= 1，收到 {n_slices}")
    idx = {min(max(round(float(f) * (n_slices - 1)), 0), n_slices - 1) for f in fractions}
    return sorted(idx)


def hstack_panels(panels: Sequence[np.ndarray]) -> np.ndarray:
    """横向拼接多张面板（2D → 2D；3D RGB → RGB）。"""
    if not panels:
        raise ProtocolConfigError("hstack_panels 需要至少一个面板")
    shapes = {np.asarray(p).shape for p in panels}
    if len(shapes) != 1:
        raise ProtocolConfigError(f"hstack_panels 面板形状必须一致，收到 {sorted(shapes)}")
    return np.concatenate([np.asarray(p) for p in panels], axis=1)


# --------------------------------------------------------------------------- 几何与重采样记录
@dataclass
class PairGeometryRecord:
    """一例 × 一个序列对的重采样记录（只记录事实，不做任何配准决策）。"""

    case_id: str
    pair: str
    series: str
    t2w_path: str
    moving_path: str
    t2w_size: tuple
    t2w_spacing: tuple
    t2w_origin: tuple
    t2w_direction: tuple
    t2w_orientation: str
    moving_size: tuple
    moving_spacing: tuple
    moving_origin: tuple
    moving_direction: tuple
    same_grid_as_t2w: bool
    resampled_now: bool
    overlap_ratio: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "pair": self.pair,
            "series": self.series,
            "t2w_path": self.t2w_path,
            "moving_path": self.moving_path,
            "t2w_size": list(self.t2w_size),
            "t2w_spacing": list(self.t2w_spacing),
            "t2w_origin": list(self.t2w_origin),
            "t2w_direction": list(self.t2w_direction),
            "t2w_orientation": self.t2w_orientation,
            "moving_size": list(self.moving_size),
            "moving_spacing": list(self.moving_spacing),
            "moving_origin": list(self.moving_origin),
            "moving_direction": list(self.moving_direction),
            "same_grid_as_t2w": bool(self.same_grid_as_t2w),
            "resampled_now": bool(self.resampled_now),
            "overlap_ratio": list(self.overlap_ratio),
            "note": RESAMPLE_ONLY_BANNER,
        }


def prepare_pair_arrays(
    t2w_path: str | Path,
    moving_path: str | Path,
    *,
    case_id: str,
    pair: str,
) -> tuple[np.ndarray, np.ndarray, PairGeometryRecord]:
    """读入 T2W 与 moving，必要时**仅按物理坐标**重采样到 T2W 网格。

    - 同网格：直接使用（记录 `resampled_now=False`）；
    - 不同网格：用 `picai.resample_to_geometry`（线性插值、物理坐标）重采样到 T2W 网格；
    - **不做任何配准**（无刚体估计、无形变场、无互信息优化）。
    """
    t2w_geo = P.read_geometry(t2w_path)
    moving_geo = P.read_geometry(moving_path)
    if len(t2w_geo.size) not in (2, 3) or len(moving_geo.size) not in (2, 3):
        raise ProtocolConfigError(
            f"G0-R QC 只支持 2D/3D 数据（size={t2w_geo.size} / {moving_geo.size}）"
        )
    same_grid = P.grids_equal(t2w_geo, moving_geo)
    if same_grid:
        t2w = P.load_array(t2w_path)
        moving = P.load_array(moving_path)
    else:
        t2w = P.load_array(t2w_path)
        moving = P.resample_to_geometry(moving_path, t2w_geo, is_label=False)
    if np.asarray(t2w).shape != np.asarray(moving).shape:
        raise ProtocolConfigError(
            f"重采样后形状仍不一致（{np.asarray(t2w).shape} vs {np.asarray(moving).shape}）：{case_id}/{pair}"
        )
    # 物理包围盒重叠率：该几何维度不受支持时留空（不伪造数值）
    overlap_ratio: list[float] = []
    try:
        _, ratio = P.axes_overlap(t2w_geo, moving_geo)
        overlap_ratio = [float(v) for v in ratio]
    except Exception:  # noqa: BLE001 - 2D 下 physical_corners 不支持，留空并如实记录
        overlap_ratio = []
    record = PairGeometryRecord(
        case_id=case_id,
        pair=pair,
        series=PAIR_TO_SERIES[pair],
        t2w_path=str(t2w_path),
        moving_path=str(moving_path),
        t2w_size=tuple(t2w_geo.size),
        t2w_spacing=tuple(t2w_geo.spacing),
        t2w_origin=tuple(t2w_geo.origin),
        t2w_direction=tuple(t2w_geo.direction),
        t2w_orientation=P.orientation_code(t2w_geo),
        moving_size=tuple(moving_geo.size),
        moving_spacing=tuple(moving_geo.spacing),
        moving_origin=tuple(moving_geo.origin),
        moving_direction=tuple(moving_geo.direction),
        same_grid_as_t2w=bool(same_grid),
        resampled_now=bool(not same_grid),
        overlap_ratio=[float(v) for v in overlap_ratio],
    )
    return np.asarray(t2w), np.asarray(moving), record


# --------------------------------------------------------------------------- landmark 模板与位移
#: landmark CSV 的坐标空间（**冻结值**）：各序列的**原生体素索引**
#: `(i, j, k) = (x, y, z)` —— T2W 索引位于 T2W 物化网格，`moving` 索引位于对应 ADC/HBV 的物化网格；
#: 两个网格都可由研究者在 ITK-SNAP / 3D Slicer 中直接打开磁盘上的物化 `.nii.gz` 标注。
#: 物理坐标与毫米距离由 SimpleITK `TransformIndexToPhysicalPoint`（LPS）按**各自**几何计算，
#: 因此非零 origin、非单位 direction 与不同 spacing 都会被正确处理。
COORDINATE_SPACE_NATIVE = "native_index_per_series"
COORDINATE_SPACE_ALLOWED: tuple[str, ...] = (COORDINATE_SPACE_NATIVE,)

LANDMARK_TEMPLATE_COLUMNS: tuple[str, ...] = (
    "case_id",
    "pair",
    "landmark_name",
    "coordinate_space",
    "t2w_i",
    "t2w_j",
    "t2w_k",
    "moving_i",
    "moving_j",
    "moving_k",
    "measured_by",
    "notes",
)

#: 必备列（校验用；缺任一列时无法逐行校验）
LANDMARK_REQUIRED_COLUMNS: tuple[str, ...] = (
    "case_id",
    "pair",
    "landmark_name",
    "coordinate_space",
    "t2w_i",
    "t2w_j",
    "t2w_k",
    "moving_i",
    "moving_j",
    "moving_k",
)

LANDMARK_COORD_FIELDS: tuple[str, ...] = ("t2w_i", "t2w_j", "t2w_k", "moving_i", "moving_j", "moving_k")


def build_landmark_template_rows(
    case_ids: Sequence[str],
    pairs: Sequence[str],
    *,
    per_case_pairs: Mapping[str, Sequence[str]] | None = None,
) -> list[dict[str, Any]]:
    """生成人工 landmark 记录模板（坐标留空；`coordinate_space` 预填冻结值）。"""
    rows: list[dict[str, Any]] = []
    for case_id in case_ids:
        active_pairs = pairs if per_case_pairs is None else per_case_pairs.get(case_id, pairs)
        for pair in active_pairs:
            rows.append(
                {
                    "case_id": case_id,
                    "pair": pair,
                    "landmark_name": "",
                    "coordinate_space": COORDINATE_SPACE_NATIVE,
                    "t2w_i": "", "t2w_j": "", "t2w_k": "",
                    "moving_i": "", "moving_j": "", "moving_k": "",
                    "measured_by": "",
                    "notes": "",
                }
            )
    return rows


def parse_strict_index(value: Any) -> tuple[int | None, str | None]:
    """严格整数索引解析（返回 `(值, 错误)`；合法时错误为 `None`）。

    只接受有限、整数值（`10` / `10.0` / `1e2`）。**拒绝**：空、非数值、NaN/Inf、
    以及非整数小数（如 `10.9` —— 不允许静默截断到 `10`）。
    """
    if value is None:
        return None, "为空（必须填写整数体素索引）"
    if isinstance(value, str) and not value.strip():
        return None, "为空（必须填写整数体素索引）"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None, f"的值 {value!r} 不是数值"
    if not math.isfinite(number):
        return None, f"的值 {value!r} 非有限值（NaN/Inf 不允许）"
    if number != int(number):
        return None, f"的值 {value!r} 不是整数索引（不允许小数截断，请填写原始体素索引）"
    return int(number), None


def validate_landmark_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    allowed_cases: Sequence[str],
    allowed_pairs: Sequence[str] = PAIRS,
    grid_info_by_case: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[str]:
    """校验人工 landmark 行；**收集全部行的全部错误**后返回（不提前中止）。

    检查项：
    - 必需列（含 `coordinate_space`）；
    - `case_id` 必须在 sampling manifest 中、`pair` 合法、`landmark_name` 非空；
    - `coordinate_space` 必须是冻结值（见 `COORDINATE_SPACE_ALLOWED`）；
    - 严格整数坐标（拒绝小数截断 / NaN / Inf / 空）、非负；
    - **FOV 分别检查**：T2W 坐标按 T2W 网格 size，`moving` 坐标按对应 ADC/HBV 的 moving 网格 size；
    - 重复记录（同一 `case_id + pair + landmark_name`）。
    """
    errors: list[str] = []
    if not rows:
        return ["landmark 文件没有数据行"]
    header = set(rows[0].keys())
    missing_columns = [col for col in LANDMARK_REQUIRED_COLUMNS if col not in header]
    if missing_columns:
        errors.append(f"缺少必需列: {missing_columns}（模板见 landmark_template.csv）")
        return errors
    allowed_case_set = set(allowed_cases)
    seen: dict[tuple[str, str, str], int] = {}
    for idx, row in enumerate(rows):
        prefix = f"row {idx + 1}"
        case_id = str(row.get("case_id", "")).strip()
        pair = str(row.get("pair", "")).strip()
        name = str(row.get("landmark_name", "")).strip()
        row_ok = True
        if case_id not in allowed_case_set:
            errors.append(f"{prefix}: case_id={case_id!r} 不在 sampling manifest 中")
            row_ok = False
        if pair not in allowed_pairs:
            errors.append(f"{prefix}: pair={pair!r} 不在允许集合 {list(allowed_pairs)}")
            row_ok = False
        if not name:
            errors.append(f"{prefix}: landmark_name 为空（每个 landmark 必须命名以便复核）")
            row_ok = False
        space = str(row.get("coordinate_space", "")).strip()
        if space not in COORDINATE_SPACE_ALLOWED:
            errors.append(
                f"{prefix}: coordinate_space={space!r} 非法（冻结值为 {list(COORDINATE_SPACE_ALLOWED)}）"
            )
            row_ok = False
        coords: dict[str, int] = {}
        for field_name in LANDMARK_COORD_FIELDS:
            value, err = parse_strict_index(row.get(field_name))
            if err is not None:
                errors.append(f"{prefix}: {field_name} {err}")
                row_ok = False
                continue
            if value < 0:
                errors.append(f"{prefix}: {field_name}={value} 不能为负")
                row_ok = False
                continue
            coords[field_name] = value
        if row_ok and name:
            key = (case_id, pair, name)
            if key in seen:
                errors.append(
                    f"{prefix}: 重复 landmark（case_id={case_id}, pair={pair}, landmark_name={name}；"
                    f"首次出现在 row {seen[key]}）"
                )
            else:
                seen[key] = idx + 1
        if grid_info_by_case and case_id in grid_info_by_case and len(coords) == len(LANDMARK_COORD_FIELDS):
            entry = grid_info_by_case[case_id]
            t2w_size = entry.get("t2w_size")
            if t2w_size and len(t2w_size) == 3:
                t2w_limits = {"i": int(t2w_size[0]), "j": int(t2w_size[1]), "k": int(t2w_size[2])}
                for axis, limit in t2w_limits.items():
                    value = coords[f"t2w_{axis}"]
                    if value >= limit:
                        errors.append(f"{prefix}: t2w_{axis}={value} 超出 T2W 网格 {axis}<{limit}（{case_id}）")
            pair_entry = (entry.get("pairs") or {}).get(pair) or {}
            moving_size = pair_entry.get("moving_size")
            if moving_size and len(moving_size) == 3:
                moving_limits = {"i": int(moving_size[0]), "j": int(moving_size[1]), "k": int(moving_size[2])}
                for axis, limit in moving_limits.items():
                    value = coords[f"moving_{axis}"]
                    if value >= limit:
                        errors.append(
                            f"{prefix}: moving_{axis}={value} 超出 {pair} 的 moving 网格 {axis}<{limit}（{case_id}）"
                        )
    return errors


@dataclass
class LandmarkParseResult:
    rows: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    skipped: list[dict[str, Any]] = field(default_factory=list)


def geometry_from_record_fields(record: Mapping[str, Any], *, role: str) -> P.Geometry | None:
    """从 `qc_geometry.json` 的 `{role}_{size,spacing,origin,direction}` 字段构造 `picai.Geometry`。

    只读几何（不读体素）；字段不完整时返回 `None`（由调用方记为 blocker/skip，**不猜测默认值**）。
    """
    size = record.get(f"{role}_size")
    spacing = record.get(f"{role}_spacing")
    origin = record.get(f"{role}_origin")
    direction = record.get(f"{role}_direction")
    if not (size and spacing and origin and direction):
        return None
    return P.Geometry(
        size=tuple(int(v) for v in size),
        spacing=tuple(float(v) for v in spacing),
        origin=tuple(float(v) for v in origin),
        direction=tuple(float(v) for v in direction),
    )


def index_to_physical_lps(index_ijk: Sequence[int], geometry: P.Geometry) -> tuple[float, float, float]:
    """体素索引 → LPS 物理坐标（mm）：使用 SimpleITK 自身约定（含 origin / direction / spacing）。"""
    image = P.image_from_geometry(geometry)
    point = image.TransformIndexToPhysicalPoint([int(v) for v in index_ijk])
    return (float(point[0]), float(point[1]), float(point[2]))


def compute_landmark_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    geometry_by_case: Mapping[str, Mapping[str, Any]],
    allowed_cases: Sequence[str],
    allowed_pairs: Sequence[str] = PAIRS,
) -> LandmarkParseResult:
    """把人工 landmark 行换算为 `landmark_displacement_mm`（**LPS 物理坐标欧氏距离**）。

    坐标空间（冻结）：`native_index_per_series` —— T2W 索引在 T2W 物化网格、`moving` 索引在对应
    ADC/HBV 的物化网格；两侧分别经**各自** `size/spacing/origin/direction` 转物理坐标后计算毫米距离
    （非零 origin、非单位 direction 与不同 spacing 都会被正确纳入）。**不做任何配准**。

    非法/不完整行记入 `errors` / `skipped`（不静默截断、不伪造）；调用方（指标脚本）必须先通过
    `validate_landmark_rows` 的完整校验。
    """
    result = LandmarkParseResult()
    allowed_case_set = set(allowed_cases)
    for idx, row in enumerate(rows):
        prefix = f"row {idx + 1}"
        case_id = str(row.get("case_id", "")).strip()
        pair = str(row.get("pair", "")).strip()
        name = str(row.get("landmark_name", "")).strip()
        if case_id not in allowed_case_set:
            result.skipped.append({"row": prefix, "reason": f"case_id={case_id!r} 不在 sampling manifest 中"})
            continue
        if pair not in allowed_pairs:
            result.skipped.append({"row": prefix, "reason": f"pair={pair!r} 非法"})
            continue
        if not name:
            result.skipped.append({"row": prefix, "reason": "landmark_name 为空"})
            continue
        case_entry = geometry_by_case.get(case_id)
        if case_entry is None:
            result.skipped.append({"row": prefix, "reason": f"缺少 {case_id} 的几何条目（先运行渲染阶段）"})
            continue
        t2w_geo = geometry_from_record_fields(case_entry, role="t2w")
        pair_entry = (case_entry.get("pairs") or {}).get(pair) or {}
        moving_geo = geometry_from_record_fields(pair_entry, role="moving")
        if t2w_geo is None or moving_geo is None:
            result.skipped.append(
                {
                    "row": prefix,
                    "reason": f"缺少 {case_id}/{pair} 的完整几何（size/spacing/origin/direction）→ 无法计算物理位移",
                }
            )
            continue
        coords_t2w: dict[str, int] = {}
        coords_moving: dict[str, int] = {}
        bad = False
        for axis in ("i", "j", "k"):
            value, err = parse_strict_index(row.get(f"t2w_{axis}"))
            if err is not None:
                result.errors.append(f"{prefix}: t2w_{axis} {err}")
                bad = True
            else:
                coords_t2w[axis] = value
            value, err = parse_strict_index(row.get(f"moving_{axis}"))
            if err is not None:
                result.errors.append(f"{prefix}: moving_{axis} {err}")
                bad = True
            else:
                coords_moving[axis] = value
        if bad:
            continue
        # FOV：与 validate_landmark_rows 同一规则（T2W 按 T2W 网格、moving 按对应序列网格）；
        # 越界行不得产出位移值（即使处于诊断性放行模式，也不填充未验证坐标的结果）。
        out_of_fov: list[str] = []
        for role, geometry, coords in (("t2w", t2w_geo, coords_t2w), ("moving", moving_geo, coords_moving)):
            for axis_index, axis in enumerate(("i", "j", "k")):
                limit = int(geometry.size[axis_index])
                if coords[axis] >= limit:
                    out_of_fov.append(f"{role}_{axis}={coords[axis]}>={limit}")
        if out_of_fov:
            result.skipped.append({"row": prefix, "reason": "坐标超出 FOV（未产出指标）: " + "; ".join(out_of_fov)})
            continue
        p_t2w = index_to_physical_lps([coords_t2w["i"], coords_t2w["j"], coords_t2w["k"]], t2w_geo)
        p_moving = index_to_physical_lps([coords_moving["i"], coords_moving["j"], coords_moving["k"]], moving_geo)
        displacement = float(np.linalg.norm(np.asarray(p_t2w, dtype=float) - np.asarray(p_moving, dtype=float)))
        if not math.isfinite(displacement):
            result.errors.append(f"{prefix}: 位移非有限值")
            continue
        result.rows.append(
            {
                "case_id": case_id,
                "pair": pair,
                "metric": "landmark_displacement_mm",
                "value": f"{displacement:.6f}",
                "unit": "mm",
                "method": "manual_landmark",
                "measured_by": str(row.get("measured_by", "")).strip() or "UNKNOWN",
                "notes": f"landmark={name}; coordinate_space={COORDINATE_SPACE_NATIVE}; "
                f"moving_series={PAIR_TO_SERIES[pair]}"
                + (f"; {str(row.get('notes')).strip()}" if str(row.get("notes", "")).strip() else ""),
            }
        )
    return result


def check_landmark_completeness(
    rows: Sequence[Mapping[str, Any]],
    *,
    allowed_cases: Sequence[str],
    allowed_pairs: Sequence[str] = PAIRS,
    min_per_case_pair: int | None = None,
) -> dict[str, Any]:
    """完整性检查：覆盖病例、每 case×pair 计数、`measured_by` 填写与未知病例。

    **不编造最小 landmark 数量阈值**：`min_per_case_pair` 为 `None`（协议尚未冻结）时，`blockers` 中给出
    `landmark_min_count_rule_not_frozen`，要求研究者在真实运行前冻结
    （`configs/protocols/g0_r_alignment_qc.yaml` 的 `evaluation.landmark_rules.min_per_case_pair`）；
    给定该值后按它检查每个 case×pair 的计数。
    """
    allowed_case_set = set(allowed_cases)
    per_case_pair: dict[tuple[str, str], int] = {}
    covered_cases: set[str] = set()
    missing_measured_by: list[str] = []
    unknown_cases: list[str] = []
    for idx, row in enumerate(rows):
        case_id = str(row.get("case_id", "")).strip()
        pair = str(row.get("pair", "")).strip()
        if case_id not in allowed_case_set:
            unknown_cases.append(f"row {idx + 1}: {case_id!r}")
            continue
        covered_cases.add(case_id)
        per_case_pair[(case_id, pair)] = per_case_pair.get((case_id, pair), 0) + 1
        if not str(row.get("measured_by", "")).strip():
            missing_measured_by.append(f"row {idx + 1}")
    expected = [(case_id, pair) for case_id in allowed_cases for pair in allowed_pairs]
    missing_case_pair = [f"{case_id}/{pair}" for case_id, pair in expected if (case_id, pair) not in per_case_pair]
    missing_cases = [case_id for case_id in allowed_cases if case_id not in covered_cases]
    blockers: list[str] = []
    if missing_case_pair:
        blockers.append(
            f"coverage_incomplete: 缺少 {len(missing_case_pair)} 个 case×pair 的 landmark"
            f"（前 5: {missing_case_pair[:5]}）"
        )
    if missing_measured_by:
        blockers.append(
            f"measured_by_missing: {len(missing_measured_by)} 行的 measured_by 为空（双阅片追溯必需）"
        )
    if min_per_case_pair is None:
        blockers.append(
            "landmark_min_count_rule_not_frozen: 协议尚未规定每个 case×pair 的最小 landmark 数量"
            "（`configs/protocols/g0_r_alignment_qc.yaml` 的 `evaluation.landmark_rules.min_per_case_pair` 为 null）；"
            "研究者必须在真实运行前冻结该规则，本工具不得代为编造阈值"
        )
    else:
        below_min = [
            f"{case_id}/{pair}={count}"
            for (case_id, pair), count in sorted(per_case_pair.items())
            if count < int(min_per_case_pair)
        ]
        if below_min:
            blockers.append(f"below_min_count(min={int(min_per_case_pair)}): {below_min[:5]}")
    return {
        "n_expected_cases": len(allowed_cases),
        "n_cases_covered": len(covered_cases),
        "missing_cases": missing_cases,
        "missing_case_pair": missing_case_pair,
        "per_case_pair_counts": {f"{case_id}/{pair}": n for (case_id, pair), n in sorted(per_case_pair.items())},
        "missing_measured_by_rows": missing_measured_by,
        "unknown_cases": unknown_cases,
        "n_rows": len(rows),
        "min_per_case_pair_rule": min_per_case_pair,
        "blockers": blockers,
        "status": "COMPLETE" if not missing_case_pair and not missing_cases else "INCOMPLETE",
    }


def summarize_landmark_metrics(metric_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """按 case×pair 汇总 landmark 位移（n / median / max），供报告引用（不写进指标 CSV）。"""
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in metric_rows:
        if row.get("metric") != "landmark_displacement_mm":
            continue
        key = (str(row["case_id"]), str(row["pair"]))
        grouped.setdefault(key, []).append(float(row["value"]))
    summary: dict[str, Any] = {}
    for (case_id, pair), values in sorted(grouped.items()):
        arr = np.asarray(values, dtype=float)
        summary.setdefault(case_id, {})[pair] = {
            "n_landmarks": int(arr.size),
            "median_mm": float(np.median(arr)),
            "max_mm": float(arr.max()),
        }
    return summary


# --------------------------------------------------------------------------- 边界距离（可选来源）
BOUNDARY_TEMPLATE_COLUMNS: tuple[str, ...] = ("case_id", "pair", "series", "boundary_path", "notes")


@dataclass
class BoundaryResult:
    metrics: list[dict[str, Any]] = field(default_factory=list)
    skipped: list[dict[str, Any]] = field(default_factory=list)


def compute_boundary_metrics(
    entries: Sequence[Mapping[str, Any]],
    *,
    t2w_series: str = "t2w",
    allowed_cases: Sequence[str] = (),
    allowed_pairs: Sequence[str] = PAIRS,
) -> BoundaryResult:
    """按“可追溯来源”的边界 mask 计算 Hausdorff 与平均表面距离（mm）。

    `entries` 每行：`case_id, pair, series, boundary_path`（同一 case×pair 需 t2w 与 moving 两条）。
    - 未提供来源 / 缺条目 / mask 为空 / 网格不可对齐 → 记入 `skipped`（原因明确），**不产出任何伪造数值**；
    - 计算在物理坐标下进行（`useImageSpacing=True`）。
    """
    import SimpleITK as sitk

    result = BoundaryResult()
    if not entries:
        result.skipped.append(
            {"reason": "no_boundary_source_configured；边界距离记为 UNKNOWN（协议 §4.2.1，不得伪造）"}
        )
        return result
    by_key: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = {}
    for idx, row in enumerate(entries):
        case_id = str(row.get("case_id", "")).strip()
        pair = str(row.get("pair", "")).strip()
        series = str(row.get("series", "")).strip().lower()
        if allowed_cases and case_id not in set(allowed_cases):
            result.skipped.append({"row": idx + 1, "reason": f"case_id={case_id!r} 不在 sampling manifest 中"})
            continue
        if pair not in allowed_pairs:
            result.skipped.append({"row": idx + 1, "reason": f"pair={pair!r} 非法"})
            continue
        if series not in MATERIALIZED_SERIES_FILES:
            result.skipped.append({"row": idx + 1, "reason": f"series={series!r} 非 t2w/adc/hbv"})
            continue
        by_key.setdefault((case_id, pair), {})[series] = row

    for (case_id, pair), series_rows in sorted(by_key.items()):
        moving_series = PAIR_TO_SERIES[pair]
        if t2w_series not in series_rows or moving_series not in series_rows:
            result.skipped.append(
                {
                    "case_id": case_id,
                    "pair": pair,
                    "reason": f"缺少 {t2w_series} 或 {moving_series} 的边界条目 → UNKNOWN",
                }
            )
            continue
        ref_row = series_rows[t2w_series]
        moving_row = series_rows[moving_series]
        try:
            ref_geo = P.read_geometry(ref_row["boundary_path"])
            moving_geo = P.read_geometry(moving_row["boundary_path"])
            ref_img = sitk.ReadImage(str(ref_row["boundary_path"]), sitk.sitkUInt8)
            moving_img = sitk.ReadImage(str(moving_row["boundary_path"]), sitk.sitkUInt8)
        except Exception as exc:  # noqa: BLE001 - 逐例记录，不中断整体
            result.skipped.append({"case_id": case_id, "pair": pair, "reason": f"边界读取失败: {exc}"})
            continue
        if not P.grids_equal(ref_geo, moving_geo):
            moving_img = sitk.Resample(moving_img, ref_img, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
        ref_bin = sitk.Cast(ref_img > 0, sitk.sitkUInt8)
        moving_bin = sitk.Cast(moving_img > 0, sitk.sitkUInt8)
        if int(sitk.GetArrayViewFromImage(ref_bin).sum()) == 0 or int(sitk.GetArrayViewFromImage(moving_bin).sum()) == 0:
            result.skipped.append({"case_id": case_id, "pair": pair, "reason": "边界 mask 为空 → UNKNOWN"})
            continue
        hausdorff_filter = sitk.HausdorffDistanceImageFilter()
        hausdorff_filter.Execute(ref_bin, moving_bin)
        hausdorff_mm = float(hausdorff_filter.GetHausdorffDistance())
        distance_to_moving = sitk.SignedMaurerDistanceMap(
            moving_bin, insideIsPositive=False, squaredDistance=False, useImageSpacing=True
        )
        distance_to_ref = sitk.SignedMaurerDistanceMap(
            ref_bin, insideIsPositive=False, squaredDistance=False, useImageSpacing=True
        )
        ref_contour = sitk.LabelContour(ref_bin, fullyConnected=False)
        moving_contour = sitk.LabelContour(moving_bin, fullyConnected=False)
        ref_mask = sitk.GetArrayViewFromImage(ref_contour) > 0
        moving_mask = sitk.GetArrayViewFromImage(moving_contour) > 0
        d_to_moving = np.abs(sitk.GetArrayViewFromImage(distance_to_moving))[ref_mask]
        d_to_ref = np.abs(sitk.GetArrayViewFromImage(distance_to_ref))[moving_mask]
        mean_surface_mm = float((d_to_moving.mean() + d_to_ref.mean()) / 2.0)
        for metric, value in (("hausdorff_distance_mm", hausdorff_mm), ("mean_surface_distance_mm", mean_surface_mm)):
            if not math.isfinite(value):
                result.skipped.append({"case_id": case_id, "pair": pair, "reason": f"{metric} 非有限值"})
                continue
            result.metrics.append(
                {
                    "case_id": case_id,
                    "pair": pair,
                    "metric": metric,
                    "value": f"{value:.6f}",
                    "unit": "mm",
                    "method": "traceable_boundary_masks",
                    "measured_by": "tool",
                    "notes": "boundary source: " + str(ref_row.get("notes", "")).strip(),
                }
            )
    return result


def read_boundary_source(path: str | Path) -> list[dict[str, Any]]:
    """读取边界来源清单（CSV：`case_id,pair,series,boundary_path[,notes]` 或 JSON `{entries: [...]}`）。"""
    import csv

    p = Path(path)
    if not p.is_file():
        raise ProtocolConfigError(f"boundary source 不存在: {p}")
    if p.suffix.lower() == ".json":
        doc = json.loads(p.read_text(encoding="utf-8"))
        entries = doc.get("entries") if isinstance(doc, Mapping) else doc
        if not isinstance(entries, list):
            raise ProtocolConfigError("boundary source JSON 需要 {entries: [...]} 或顶层列表")
        return [dict(item) for item in entries]
    with open(p, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = [
            col for col in ("case_id", "pair", "series", "boundary_path") if col not in (reader.fieldnames or [])
        ]
        if missing:
            raise ProtocolConfigError(f"boundary source CSV 缺少列: {missing}（模板见 boundary_template.csv）")
        return [dict(row) for row in reader]


__all__ = [
    "BOUNDARY_TEMPLATE_COLUMNS",
    "COORDINATE_SPACE_ALLOWED",
    "COORDINATE_SPACE_NATIVE",
    "DEFAULT_SLICE_FRACTIONS",
    "LANDMARK_COORD_FIELDS",
    "LANDMARK_REQUIRED_COLUMNS",
    "LANDMARK_TEMPLATE_COLUMNS",
    "MATERIALIZED_SERIES_FILES",
    "PAIRS",
    "PAIR_TO_SERIES",
    "RESAMPLE_ONLY_BANNER",
    "BoundaryResult",
    "LandmarkParseResult",
    "PairGeometryRecord",
    "build_landmark_template_rows",
    "case_dir_for",
    "check_landmark_completeness",
    "checkerboard_image",
    "compute_boundary_metrics",
    "compute_landmark_metrics",
    "edge_map",
    "edge_overlay_image",
    "geometry_from_record_fields",
    "hstack_panels",
    "index_to_physical_lps",
    "load_sampling_manifest",
    "missing_series",
    "normalize_for_display",
    "parse_strict_index",
    "prepare_pair_arrays",
    "read_boundary_source",
    "resolve_series_paths",
    "select_cases",
    "select_slice_indices",
    "summarize_landmark_metrics",
    "validate_landmark_rows",
]
