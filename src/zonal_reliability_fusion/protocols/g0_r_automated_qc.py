"""G0-R Automated v0.3 核心算法库（全自动、自检、自决策的本地序列对齐 QC）。

**边界（与 `configs/protocols/g0_r_alignment_qc_automated.yaml` 及 `docs/protocols/G0_R_ALIGNMENT_QC_AUTOMATED.md` 一致）**：

- 只**读**物化影像（由 CLI 负责读取；本模块只接受 numpy 数组与几何元数据，纯计算、便于合成测试）；
- **不写回、不修改任何医学影像**；不联网、不调用任何外部 API / 云服务；
- 不使用 lesion / WG / PZ-TZ 标签、不使用任何模型预测或训练结果（见 `assert_automated_input_policy`）；
- 自动化路径只输出 `resample-only` / `resample+rigid` / `INSUFFICIENT_EVIDENCE` 三种**候选**（DRAFT）；
- 所有算法、参数、阈值推导公式与判定逻辑由配置冻结，并绑定 `protocol_hash`；
- 证据不足时必须输出 `INSUFFICIENT_EVIDENCE`，不得强行二选一。

**v0.3 方法性变更（相对 draft-0.2；必须整体升版本，不得混用）**：

1. 刚体诊断：SimpleITK `SetInitialTransform(..., inPlace=True)`（`inPlace=False` 返回 CompositeTransform，
   旧代码对其强转 Euler3D 会抛 `Transform is not of type Euler3DTransform!`）；initial/final 均做**显式类型校验**，
   只允许「本身是 Euler3D」或「仅含一个 Euler3D 的 CompositeTransform」解包，否则 fail-closed；
2. 校准分组：unit = `(case_id, pair)`，两个序列对**完全独立**；零位移参考为 unit 内均值，零位移噪声用
   leave-one-out 残差，禁止把病例间/pair 间的绝对基线差异混入同一 SD；
3. 真实病例阈值按 pair 推导（`median(unit_midpoints)`），与灵敏度门限分离命名；`decide_case` 必须显式传 `pair`，
   未知 pair 直接失败，禁止回退全局阈值。

**轴序约定（重要）**：numpy 数组来自 `sitk.GetArrayFromImage`，轴序为 `(z, y, x)`；
几何 `spacing/origin/direction` 为 SimpleITK 的 `(x, y, z)` 顺序。转换规则：

- 数组 → SimpleITK 图像：`GetImageFromArray` 生成的尺寸为 `(nx, ny, nz)`，spacing 直接按 `(sx, sy, sz)`
  设置（`array_to_image`）；
- 需要按数组轴给出物理尺度时（`np.gradient`、高斯 sigma、`distance_transform_edt`），
  必须用 `(sz, sy, sx)`（即 `physical_gradient_magnitude` / `binary_edges` / `distance_map_mm` 的写法）。

`test_distance_map_respects_axis_order_and_physical_units` 锁定该约定。
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

# --------------------------------------------------------------------------- 状态与决策常量
CASE_ACCEPTABLE = "ACCEPTABLE"
CASE_FLAGGED = "FLAGGED"
CASE_FOV_INSUFFICIENT = "FOV_INSUFFICIENT"
CASE_INVALID_INPUT = "INVALID_INPUT"
CASE_REGISTRATION_FAILED = "REGISTRATION_DIAGNOSTIC_FAILED"
CASE_INSUFFICIENT = "INSUFFICIENT_EVIDENCE"

CASE_STATUSES: tuple[str, ...] = (
    CASE_ACCEPTABLE,
    CASE_FLAGGED,
    CASE_FOV_INSUFFICIENT,
    CASE_INVALID_INPUT,
    CASE_REGISTRATION_FAILED,
    CASE_INSUFFICIENT,
)

DECISION_RESAMPLE_ONLY = "resample-only"
DECISION_RESAMPLE_RIGID = "resample+rigid"
DECISION_INSUFFICIENT = "INSUFFICIENT_EVIDENCE"

#: 三个自动候选（自动路径不得输出其他取值）
ALLOWED_CANDIDATES: tuple[str, ...] = (DECISION_RESAMPLE_ONLY, DECISION_RESAMPLE_RIGID, DECISION_INSUFFICIENT)

DIRECTION_LOWER_IS_WORSE = "lower_is_worse"
DIRECTION_HIGHER_IS_WORSE = "higher_is_worse"

#: 协议哈希覆盖的字段（运行参数 inputs/outputs/progress 不参与，避免路径变化影响协议身份）
PROTOCOL_HASH_EXCLUDED_TOP_LEVEL: tuple[str, ...] = ("inputs", "outputs", "progress")

PAIR_TO_SERIES: dict[str, str] = {"T2W-ADC": "adc", "T2W-HBV": "hbv"}

#: 所有自动输出共用的 schema 版本（禁止把 v0.2 旧输出按 v0.3 schema 静默解释）
OUTPUT_SCHEMA_VERSION = "g0-r-automated/0.4"

# ---- v0.4 指标可用性守卫与 unit 完整性（fail-closed） ----
#: 尺度可用性字段后缀：指标 `sigma1.5.<metric>` 对应 `sigma1.5.usable`
METRIC_USABLE_SUFFIX = "usable"
#: 尺度前缀（指标名前缀）
METRIC_SCALE_PREFIX = "sigma"
#: `sigma*.usable=false` 的原因（边缘体素不足，见 preprocessing.edge.min_edge_voxels）
METRIC_UNUSABLE_EDGE_VOXELS = "该尺度边缘体素不足（sigma*.usable=false）"
#: 每个 pair 至少需要的 **complete** unit 数（配置同名字段冻结）
CALIBRATION_MIN_COMPLETE_UNITS = 3
#: 是否要求被选中的校准 unit 条件网格完整（配置同名字段冻结）
REQUIRE_COMPLETE_CONDITION_GRID = True

# ---- v0.3 校准与阈值方法（方法性变更，随协议 draft-0.3 一起版本化） ----
#: 校准最小单元（pair 与 case_id 都必须分组，禁止跨 pair 合并分布）
CALIBRATION_GROUPING: tuple[str, ...] = ("pair", "case_id")
ZERO_REFERENCE_WITHIN_UNIT = "within_unit_mean"
ZERO_NOISE_LEAVE_ONE_OUT = "leave_one_out"
#: 零位移噪声阈值（只来自 unit 内残差；下限 0）
NOISE_THRESHOLD_FORMULA = (
    "max(detection_multiplier * pooled_within_unit_zero_sd, "
    "max_within_unit_zero_leave_one_out_degradation, 0.0)"
)
DEGRADATION_DEFINITION = (
    "lower_is_worse: degradation = unit_zero_reference - observed; "
    "higher_is_worse: degradation = observed - unit_zero_reference"
)
DETECTION_RATE_DEFINITION = (
    "detection_rate(d) = 该 pair 全部 (unit, direction) 样本中 degradation(u,d,dir) > noise_threshold 的比例"
)
MONOTONICITY_DEFINITION = (
    "按 unit 内 degradation 曲线先求跨 unit 均值（0 mm 点 degradation 恒为 0），"
    "再统计所有 (d_i < d_j) 配对中 agg(d_j) > agg(d_i) 的比例"
)
#: pair 级真实病例阈值的聚合方法（median / mean；由配置显式声明，不得藏在代码里）
PAIR_THRESHOLD_AGGREGATION = "median"
#: 阈值推导公式（v0.3）：median(unit_midpoint)，unit_midpoint = midpoint(unit 零位移基线, min_detectable 均值)
THRESHOLD_DERIVATION_BY_PAIR = "median_of_paired_unit_midpoints_by_pair"

#: pair → 强度扰动 seed 偏移（仅用于让同一病例的两个 pair 的噪声去相关；顺序稳定）
PAIR_NOISE_OFFSETS: dict[str, int] = {pair: index for index, pair in enumerate(sorted(PAIR_TO_SERIES))}


# --------------------------------------------------------------------------- 配置与协议身份
def protocol_hash(doc: Mapping[str, Any]) -> str:
    """协议内容哈希：去除 `inputs`/`outputs`/`progress` 后的 canonical JSON 的 SHA256。

    影响结论的任何字段（FOV 策略、预处理、指标、刚体诊断、校准、阈值、判定）变化都会改变哈希；
    仅路径/进度等运行参数变化不改变协议身份。
    """
    payload = {k: v for k, v in dict(doc).items() if k not in PROTOCOL_HASH_EXCLUDED_TOP_LEVEL}
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def assert_automated_input_policy(doc: Mapping[str, Any]) -> None:
    """强制自动路径的输入/隐私策略：任一被禁止项开启即拒绝运行（不静默降级）。"""
    policy = dict(doc.get("input_policy") or {})
    for key in ("lesion_allowed", "wg_allowed", "zonal_allowed", "model_predictions_allowed"):
        if policy.get(key) is not False:
            raise ValueError(f"input_policy.{key} 必须为 false（自动路径禁止 lesion/WG/PZ-TZ/模型预测输入）")
    privacy = dict(doc.get("privacy") or {})
    for key in ("external_upload_allowed", "external_api_allowed"):
        if privacy.get(key) is not False:
            raise ValueError(f"privacy.{key} 必须为 false（禁止任何外发/外部 API）")
    if privacy.get("source_images_read_only") is not True:
        raise ValueError("privacy.source_images_read_only 必须为 true（原始/物化影像只读）")
    protocol = dict(doc.get("protocol") or {})
    if protocol.get("sees_model_predictions") is not False:
        raise ValueError("protocol.sees_model_predictions 必须为 false")
    if protocol.get("automated_only") is not True:
        raise ValueError("protocol.automated_only 必须为 true")


# --------------------------------------------------------------------------- 数组/几何工具
def array_to_image(array: np.ndarray, spacing_xyz: Sequence[float]) -> sitk.Image:
    """numpy 数组（轴序 z,y,x）→ SimpleITK 图像。

    `sitk.GetImageFromArray` 生成的图像尺寸为 `(nx, ny, nz)`（numpy 轴逆序），
    因此 spacing 必须按 SimpleITK 自身的 `(sx, sy, sz)` 顺序设置，**不做任何反转**。
    """
    arr = np.asarray(array)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing([float(v) for v in spacing_xyz])
    return img


def check_array_sanity(array: np.ndarray) -> dict[str, Any]:
    """影像数组体检：空、非有限值、常量、非正体素占比。不抛错，如实报告。"""
    arr = np.asarray(array, dtype=np.float64)
    out: dict[str, Any] = {
        "shape": list(arr.shape),
        "n_voxels": int(arr.size),
        "empty": bool(arr.size == 0),
    }
    if arr.size == 0:
        out.update({"finite_fraction": 0.0, "nonzero_fraction": 0.0, "constant": True, "ok": False})
        return out
    finite = np.isfinite(arr)
    out["finite_fraction"] = float(finite.mean())
    out["n_nonfinite"] = int((~finite).sum())
    if finite.any():
        values = arr[finite]
        out["nonzero_fraction"] = float((values > 0).mean())
        out["min"] = float(values.min())
        out["max"] = float(values.max())
        span = float(values.max()) - float(values.min())
        # 常量判定使用相对容差：重采样/插值产生的浮点抖动不应把常量影像判为正常
        out["constant"] = bool(span <= max(1e-6, 1e-6 * abs(float(values.max()))))
    else:
        out["nonzero_fraction"] = 0.0
        out["constant"] = True
    out["ok"] = bool(out["finite_fraction"] == 1.0 and not out["constant"] and out["n_voxels"] > 0)
    return out


def check_geometry_sanity(entry: Mapping[str, Any], *, role: str) -> list[str]:
    """几何体检：size/spacing/origin/direction 完整性、正值 spacing、方向矩阵近似正交。"""
    errors: list[str] = []
    for field in ("size", "spacing", "origin", "direction"):
        if not entry.get(f"{role}_{field}"):
            errors.append(f"{role}_{field} 缺失")
    size = entry.get(f"{role}_size")
    spacing = entry.get(f"{role}_spacing")
    direction = entry.get(f"{role}_direction")
    if size and len(size) != 3:
        errors.append(f"{role}_size 非 3D（{size}）")
    if size and any(int(v) <= 0 for v in size):
        errors.append(f"{role}_size 存在非正尺寸（{size}）")
    if spacing and any((not math.isfinite(float(v))) or float(v) <= 0 for v in spacing):
        errors.append(f"{role}_spacing 存在非正/非有限值（{spacing}）")
    if direction and len(direction) == 9:
        m = np.asarray(direction, dtype=float).reshape(3, 3)
        if not np.all(np.isfinite(m)):
            errors.append(f"{role}_direction 存在非有限值")
        else:
            gram = m.T @ m
            if float(np.abs(gram - np.eye(3)).max()) > 1e-3:
                errors.append(f"{role}_direction 非正交（|MᵀM−I|max={float(np.abs(gram - np.eye(3)).max()):.2e}）")
    return errors


def fov_evidence(
    record: Mapping[str, Any],
    resampled_shape: Sequence[int],
    *,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    """A 组证据：物理重叠比例、z 覆盖、partial overlap 与 FOV 判定。

    `record` 为 `g0_r_qc.PairGeometryRecord.as_dict()`（含两侧完整几何与 `overlap_ratio`）。
    """
    overlap = [float(v) for v in (record.get("overlap_ratio") or [])]
    min_ratio = float(policy["min_axis_overlap_ratio"])
    n_overlapping = int(sum(1 for v in overlap if v >= min_ratio))
    shape = [int(v) for v in resampled_shape]
    z_slices = int(shape[0]) if shape else 0  # 数组轴序 (z, y, x) → z 为 axis 0
    min_z = int(policy["min_covered_slices_z"])
    reasons: list[str] = []
    if len(overlap) < 3:
        reasons.append(f"overlap_ratio 不可用（{overlap}）")
    if n_overlapping < int(policy["min_overlapping_axes"]):
        reasons.append(f"物理重叠轴数 {n_overlapping} < {policy['min_overlapping_axes']}（ratio={overlap}）")
    if z_slices < min_z:
        reasons.append(f"z 层数 {z_slices} < {min_z}")
    return {
        "size_resampled_zyx": shape,
        "z_slices": z_slices,
        "overlap_ratio": overlap,
        "n_overlapping_axes": n_overlapping,
        "partial_overlap": (bool(len(overlap) == 3 and min(overlap) < 1.0) if overlap else None),
        "moving_physical_extent_mm": _extent_mm(record, "moving"),
        "t2w_physical_extent_mm": _extent_mm(record, "t2w"),
        "extent_ratio_moving_over_t2w": _extent_ratio(record),
        "fov_ok": not reasons,
        "fov_reasons": reasons,
    }


def _extent_mm(record: Mapping[str, Any], role: str) -> list[float] | None:
    size = record.get(f"{role}_size")
    spacing = record.get(f"{role}_spacing")
    if not size or not spacing or len(size) != 3 or len(spacing) != 3:
        return None
    return [float(size[i]) * float(spacing[i]) for i in range(3)]


def _extent_ratio(record: Mapping[str, Any]) -> list[float] | None:
    a = _extent_mm(record, "moving")
    b = _extent_mm(record, "t2w")
    if not a or not b:
        return None
    return [float(a[i] / b[i]) if b[i] > 0 else float("nan") for i in range(3)]


# --------------------------------------------------------------------------- B. 跨模态边缘一致性
def robust_normalize(
    array: np.ndarray,
    *,
    method: str = "robust_percentile_scaling",
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
    background_masking: bool = True,
) -> np.ndarray:
    """强度标准化（冻结）：百分位裁剪 + 线性映射到 [0, 1]；可选排除背景（<=0）。

    只允许 `robust_percentile_scaling`（协议冻结）；其它方法名一律拒绝，避免静默换算法。
    """
    if method != "robust_percentile_scaling":
        raise ValueError(f"不支持的强度标准化方法: {method}（协议只允许 robust_percentile_scaling）")
    arr = np.asarray(array, dtype=np.float64)
    if arr.size == 0:
        return arr
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr)
    values = arr[finite]
    reference = values[values > 0] if (background_masking and (values > 0).any()) else values
    lo = float(np.percentile(reference, low_percentile))
    hi = float(np.percentile(reference, high_percentile))
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        return np.zeros_like(arr)
    out = (arr - lo) / (hi - lo)
    out[~finite] = 0.0
    return np.clip(out, 0.0, 1.0)


def physical_gradient_magnitude(array: np.ndarray, spacing_xyz: Sequence[float]) -> np.ndarray:
    """按物理 spacing 求导的梯度幅值（单位：标准化强度 / mm）。数组轴序 (z, y, x)。"""
    arr = np.asarray(array, dtype=np.float64)
    sp = np.asarray(spacing_xyz, dtype=np.float64)
    gz, gy, gx = np.gradient(arr, float(sp[2]), float(sp[1]), float(sp[0]))
    return np.sqrt(gx * gx + gy * gy + gz * gz)


def binary_edges(
    array: np.ndarray,
    spacing_xyz: Sequence[float],
    *,
    sigma_mm: float,
    threshold_percentile: float,
) -> np.ndarray:
    """多尺度边缘：物理尺度高斯平滑 → 物理梯度幅值 → 百分位阈值 → 二值边缘。"""
    arr = np.asarray(array, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"binary_edges 只支持 3D 数组，收到 ndim={arr.ndim}")
    if arr.size == 0:
        return np.zeros(arr.shape, dtype=bool)
    sp = np.asarray(spacing_xyz, dtype=np.float64)
    sigma_vox = [float(sigma_mm) / max(float(sp[2]), 1e-9), float(sigma_mm) / max(float(sp[1]), 1e-9),
                 float(sigma_mm) / max(float(sp[0]), 1e-9)]
    smooth = ndimage.gaussian_filter(arr, sigma=sigma_vox, mode="nearest")
    grad = physical_gradient_magnitude(smooth, spacing_xyz)
    thr = float(np.percentile(grad, threshold_percentile)) if grad.size else 0.0
    if not math.isfinite(thr) or thr <= 0:
        return np.zeros(arr.shape, dtype=bool)
    return grad >= thr


def distance_map_mm(mask: np.ndarray, spacing_xyz: Sequence[float]) -> np.ndarray | None:
    """每个体素到**最近掩膜前景体素**的欧氏距离（mm，物理单位）；空掩膜返回 None（不伪造数值）。

    使用 `scipy.ndimage.distance_transform_edt(~mask)`：`sampling` 按 numpy 数组轴序 `(z, y, x)`
    给出，即 `(spacing_z, spacing_y, spacing_x)`，因此 mm 距离与物理几何一致。
    """
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return None
    full_sampling = (float(spacing_xyz[2]), float(spacing_xyz[1]), float(spacing_xyz[0]))  # (z, y, x)
    sampling = full_sampling[-m.ndim :] if m.ndim <= 3 else full_sampling
    return np.asarray(ndimage.distance_transform_edt(~m, sampling=sampling), dtype=np.float64)


def edge_agreement_metrics(
    edge_a: np.ndarray,
    edge_b: np.ndarray,
    spacing_xyz: Sequence[float],
    *,
    radii_mm: Sequence[float],
) -> dict[str, Any]:
    """对称 edge-to-edge 距离与多容忍半径 edge F1（全部为 mm 单位，双向对称）。"""
    a = np.asarray(edge_a, dtype=bool)
    b = np.asarray(edge_b, dtype=bool)
    out: dict[str, Any] = {"n_edge_a": int(a.sum()), "n_edge_b": int(b.sum())}
    if not a.any() or not b.any():
        out["status"] = "NO_EDGE"
        return out
    dist_to_a = distance_map_mm(a, spacing_xyz)
    dist_to_b = distance_map_mm(b, spacing_xyz)
    if dist_to_a is None or dist_to_b is None:
        out["status"] = "NO_EDGE"
        return out
    d_b_to_a = dist_to_a[b]  # B 的边缘点 → 最近 A 边缘（mm）
    d_a_to_b = dist_to_b[a]
    out.update(
        {
            "status": "OK",
            "chamfer_mm": float((float(d_b_to_a.mean()) + float(d_a_to_b.mean())) / 2.0),
            "mean_a_to_b_mm": float(d_a_to_b.mean()),
            "mean_b_to_a_mm": float(d_b_to_a.mean()),
            "median_a_to_b_mm": float(np.median(d_a_to_b)),
            "p95_a_to_b_mm": float(np.percentile(d_a_to_b, 95)),
            "max_a_to_b_mm": float(d_a_to_b.max()),
        }
    )
    for radius in radii_mm:
        key = f"{float(radius):.1f}"
        precision = float(np.mean(d_b_to_a <= float(radius)))
        recall = float(np.mean(d_a_to_b <= float(radius)))
        f1 = 0.0 if precision + recall <= 0 else 2.0 * precision * recall / (precision + recall)
        out[f"edge_f1_at_{key}mm"] = f1
        out[f"edge_precision_at_{key}mm"] = precision
        out[f"edge_recall_at_{key}mm"] = recall
    return out


def slicewise_agreement(
    edge_a: np.ndarray,
    edge_b: np.ndarray,
    spacing_xyz: Sequence[float],
    *,
    radius_mm: float,
) -> dict[str, Any]:
    """分 slice（z 方向逐层）的边缘一致性：返回逐层 F1 列表与汇总。"""
    a = np.asarray(edge_a, dtype=bool)
    b = np.asarray(edge_b, dtype=bool)
    if a.ndim != 3 or b.ndim != 3 or a.shape != b.shape:
        return {"status": "SHAPE_MISMATCH"}
    per_slice: list[dict[str, Any]] = []
    for k in range(a.shape[0]):
        if not a[k].any() or not b[k].any():
            continue
        m = edge_agreement_metrics(a[k], b[k], (spacing_xyz[0], spacing_xyz[1], 1.0), radii_mm=[radius_mm])
        if m.get("status") == "OK":
            per_slice.append({"k": int(k), "edge_f1": float(m[f"edge_f1_at_{float(radius_mm):.1f}mm"])})
    if not per_slice:
        return {"status": "NO_SLICE", "n_slices": 0}
    values = np.asarray([row["edge_f1"] for row in per_slice], dtype=float)
    return {
        "status": "OK",
        "n_slices": int(values.size),
        "radius_mm": float(radius_mm),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
        "per_slice": per_slice,
    }


def pair_metrics_from_arrays(
    t2w: np.ndarray,
    moving: np.ndarray,
    spacing_xyz: Sequence[float],
    *,
    prep: Mapping[str, Any],
    metrics_cfg: Mapping[str, Any],
    slicewise_radius_mm: float | None = None,
) -> dict[str, Any]:
    """在 T2W 网格上计算一对序列的多尺度跨模态边缘一致性指标（不做任何配准）。

    指标键格式：`sigma<尺度mm>.<指标名>`；另附 `slicewise`（分 slice 统计，不参与阈值判定）。
    """
    intensity = dict(prep["intensity"])
    edge_cfg = dict(prep["edge"])
    gradient_mode = str(edge_cfg.get("gradient", "spacing_aware"))
    if gradient_mode != "spacing_aware":
        raise ValueError(f"不支持的梯度模式: {gradient_mode}（协议只允许 spacing_aware）")
    radii = [float(v) for v in metrics_cfg["tolerance_radii_mm"]]
    t2w_norm = robust_normalize(t2w, **intensity)
    moving_norm = robust_normalize(moving, **intensity)
    out: dict[str, Any] = {"status": "OK"}
    for sigma in edge_cfg["scales_sigma_mm"]:
        sigma_f = float(sigma)
        prefix = f"sigma{sigma_f:g}."
        edges_a = binary_edges(
            t2w_norm, spacing_xyz, sigma_mm=sigma_f, threshold_percentile=float(edge_cfg["threshold_percentile"])
        )
        edges_b = binary_edges(
            moving_norm, spacing_xyz, sigma_mm=sigma_f, threshold_percentile=float(edge_cfg["threshold_percentile"])
        )
        m = edge_agreement_metrics(edges_a, edges_b, spacing_xyz, radii_mm=radii)
        min_edge = int(edge_cfg.get("min_edge_voxels", 0) or 0)
        usable = bool(
            m.get("status") == "OK" and m["n_edge_a"] >= min_edge and m["n_edge_b"] >= min_edge
        )
        for key, value in m.items():
            out[f"{prefix}{key}" if key not in ("status",) else f"{prefix}status"] = value
        out[f"{prefix}usable"] = usable
        if slicewise_radius_mm is not None and usable:
            out[f"{prefix}slicewise"] = slicewise_agreement(
                edges_a, edges_b, spacing_xyz, radius_mm=float(slicewise_radius_mm)
            )
    return out


def primary_and_consistency_metrics(metrics_cfg: Mapping[str, Any]) -> list[dict[str, str]]:
    """返回参与判定的指标清单（主指标 + 一致性指标），顺序固定。"""
    out = [
        {
            "name": str(metrics_cfg["primary_metric"]),
            "direction": str(metrics_cfg["primary_direction"]),
            "role": "primary",
        }
    ]
    for item in metrics_cfg.get("consistency_metrics") or []:
        out.append({"name": str(item["name"]), "direction": str(item["direction"]), "role": "consistency"})
    return out


def flatten_metric_row(case_id: str, pair: str, metrics: Mapping[str, Any]) -> dict[str, Any]:
    """把指标 dict 展平为可写 CSV 的一行（只保留标量；嵌套 slicewise 由调用方单独输出）。"""
    row: dict[str, Any] = {"case_id": case_id, "pair": pair}
    for key, value in metrics.items():
        if isinstance(value, bool):
            row[key] = int(value)
        elif isinstance(value, (int, float)):
            row[key] = value
    return row


# --------------------------------------------------------------------------- C. 诊断性刚体残余估计
def _euler_summary(transform: sitk.Euler3DTransform) -> dict[str, Any]:
    translation = np.asarray(transform.GetTranslation(), dtype=float)
    angles = np.asarray(
        [transform.GetAngleX(), transform.GetAngleY(), transform.GetAngleZ()], dtype=float
    )
    return {
        "translation_mm": [float(v) for v in translation],
        "translation_magnitude_mm": float(np.linalg.norm(translation)),
        "rotation_deg": [float(v) for v in np.degrees(angles)],
        "rotation_magnitude_deg": float(np.degrees(np.linalg.norm(angles))),
        "center_mm": [float(v) for v in transform.GetCenter()],
    }


def as_euler3d_transform(transform: Any) -> tuple[sitk.Euler3DTransform | None, str, str | None]:
    """把 SimpleITK transform 严格解析为 `Euler3DTransform`（不允许静默接受未知类型）。

    返回 `(euler, mode, error)`：

    - `mode="direct"`：输入本身即 `Euler3DTransform`；
    - `mode="composite_single_euler"`：输入是**仅含一个** `Euler3DTransform` 的 `CompositeTransform`
      （SimpleITK `SetInitialTransform(inPlace=False)` 的返回形态；兼容解包，必须在输出中显式记录）；
    - `euler is None` 时 `error` 给出拒绝原因，调用方必须 fail-closed（**不得伪造零位移**）。
    """
    if isinstance(transform, sitk.Euler3DTransform):
        return transform, "direct", None
    if isinstance(transform, sitk.CompositeTransform):
        n_transforms = int(transform.GetNumberOfTransforms())
        if n_transforms != 1:
            return None, "rejected", f"CompositeTransform 含 {n_transforms} 个 transform（只允许单元素解包）"
        inner = transform.GetNthTransform(0)
        if isinstance(inner, sitk.Euler3DTransform):
            return inner, "composite_single_euler", None
        return None, "rejected", f"CompositeTransform 内部为 {type(inner).__name__}（要求 Euler3DTransform）"
    return None, "rejected", f"transform 类型为 {type(transform).__name__}（要求 Euler3DTransform）"


def _as_float_triplet(value: Any) -> tuple[list[float] | None, str | None]:
    """把 translation/rotation 归一化为 3 个有限 float；不合法时返回 (None, 原因)（不伪造 0）。"""
    if value is None or isinstance(value, (str, bytes)):
        return None, "缺失"
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None, "无法解析为数值"
    if arr.size != 3:
        return None, f"长度不为 3（{arr.size}）"
    if not bool(np.all(np.isfinite(arr))):
        return None, "含非有限值"
    return [float(v) for v in arr], None


def _finalize_registration(result: Mapping[str, Any], cfg: Mapping[str, Any]) -> dict[str, Any]:
    """结构 + 数值有限性校验；任何缺失/非法/非有限值一律判失败（**不伪造零位移**）。"""
    if result.get("status") != "OK":
        return dict(result)
    failure = str(cfg.get("failure_status", CASE_REGISTRATION_FAILED))
    out = dict(result)
    for key in ("translation_mm", "rotation_deg"):
        triplet, error = _as_float_triplet(result.get(key))
        if triplet is None:
            return {"status": failure, "reason": f"刚体估计的 {key} {error}（不伪造零位移）"}
        out[key] = triplet
    for key in ("metric_before", "metric_after"):
        value = result.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            return {"status": failure, "reason": f"刚体估计缺少合法的 {key}（不伪造零位移）"}
        if not math.isfinite(float(value)):
            return {"status": failure, "reason": f"刚体估计的 {key} 非有限（不伪造零位移）"}
        out[key] = float(value)
    magnitude = result.get("translation_magnitude_mm")
    if magnitude is not None and not math.isfinite(float(magnitude)):
        return {"status": failure, "reason": "刚体估计的 translation_magnitude_mm 非有限（不伪造零位移）"}
    n_iterations = result.get("n_iterations")
    if n_iterations is not None and (isinstance(n_iterations, bool) or not isinstance(n_iterations, (int, np.integer)) or int(n_iterations) < 0):
        return {"status": failure, "reason": f"刚体估计的 n_iterations 非法: {n_iterations!r}"}
    return out


def estimate_rigid_diagnostic(
    t2w: np.ndarray,
    moving: np.ndarray,
    spacing_xyz: Sequence[float],
    *,
    cfg: Mapping[str, Any],
    runner: Callable[..., Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """诊断性刚体残余估计（T2W 为 reference、moving 为 moving）。

    - **仅在内存中**估计 transform，输出平移（mm）、旋转（度）、优化前后 metric、迭代数与停止条件；
    - 不写盘、不修改物化数据、不得直接作为训练预处理；
    - **v0.3 修复**：`SetInitialTransform(..., inPlace=True)`（SimpleITK 2.5.3 下 `inPlace=False` 会让 `Execute`
      返回 `CompositeTransform`，旧代码对其强转会抛 `Transform is not of type Euler3DTransform!`）；
      initial/final 都做显式类型校验，只允许直接 Euler3D 或「单元素 Euler3D CompositeTransform」解包；
    - 失败（异常 / 类型不符 / 非有限 / 形状不一致）返回 `REGISTRATION_DIAGNOSTIC_FAILED`，**绝不伪造零位移**；
    - `runner` 可注入以便合成测试；默认使用 SimpleITK（算法/参数由配置冻结）。
    """
    failure_status = str(cfg.get("failure_status", CASE_REGISTRATION_FAILED))
    if runner is not None:
        try:
            result = dict(runner(t2w, moving, spacing_xyz, cfg))
        except Exception as exc:  # noqa: BLE001 - 诊断失败必须如实标记
            return {"status": failure_status, "reason": f"runner 异常: {exc}"}
        result.setdefault("status", "OK")
        return _finalize_registration(result, cfg)

    if not bool(cfg.get("enabled", True)):
        return {"status": "DISABLED", "reason": "registration_diagnostic.enabled=false"}

    arr_ref = np.asarray(t2w, dtype=np.float32)
    arr_mov = np.asarray(moving, dtype=np.float32)
    if arr_ref.size == 0 or arr_mov.size == 0 or arr_ref.shape != arr_mov.shape:
        return {"status": failure_status, "reason": "空数组或形状不一致（未做任何估计）"}

    ref = array_to_image(arr_ref, spacing_xyz)
    mov = array_to_image(arr_mov, spacing_xyz)
    seed = int(cfg["seed"])
    metric_before: float | None = None
    metric_after: float | None = None
    try:
        initial = sitk.CenteredTransformInitializer(
            ref, mov, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
        initial_euler, _, initial_error = as_euler3d_transform(initial)
        if initial_euler is None:
            return {"status": failure_status, "reason": f"initial transform 类型校验失败: {initial_error}"}
        registration = sitk.ImageRegistrationMethod()
        registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=int(cfg["metric_bins"]))
        registration.SetMetricSamplingStrategy(registration.RANDOM)
        registration.SetMetricSamplingPercentage(float(cfg["metric_sampling_percentage"]), seed)
        registration.SetInterpolator(sitk.sitkLinear)
        registration.SetOptimizerAsRegularStepGradientDescent(
            learningRate=float(cfg["learning_rate"]),
            minStep=float(cfg["min_step"]),
            numberOfIterations=int(cfg["iterations"]),
        )
        registration.SetOptimizerScalesFromPhysicalShift()
        registration.SetShrinkFactorsPerLevel([int(v) for v in cfg["shrink_factors"]])
        registration.SetSmoothingSigmasPerLevel([float(v) for v in cfg["smoothing_sigmas_mm"]])
        registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        # inPlace=True：Execute 返回被优化的 Euler3DTransform 本身（避免 CompositeTransform 强转错误）
        registration.SetInitialTransform(initial_euler, inPlace=True)
        metric_before = float(registration.MetricEvaluate(ref, mov))
        final = registration.Execute(ref, mov)
        metric_after = float(registration.GetMetricValue())
        transform, unwrap_mode, final_error = as_euler3d_transform(final)
        if transform is None:
            return {
                "status": failure_status,
                "reason": f"最终 transform 类型校验失败: {final_error}",
                "metric_before": metric_before,
                "metric_after": metric_after,
                "final_transform_type": type(final).__name__,
            }
        iteration = int(registration.GetOptimizerIteration())
        stop_condition = str(registration.GetOptimizerStopConditionDescription())
    except Exception as exc:  # noqa: BLE001 - SimpleITK 失败一律标记失败，不伪造结果
        payload: dict[str, Any] = {"status": failure_status, "reason": f"SimpleITK 异常: {exc}"}
        if metric_before is not None:
            payload["metric_before"] = metric_before
        if metric_after is not None:
            payload["metric_after"] = metric_after
        return payload

    result: dict[str, Any] = {
        "status": "OK",
        **_euler_summary(transform),
        "metric_before": metric_before,
        "metric_after": metric_after,
        "optimizer_stop_condition": stop_condition,
        "n_iterations": iteration,
        "final_transform_type": type(final).__name__,
        "transform_unwrap": unwrap_mode,
        "library": f"SimpleITK {sitk.Version_VersionString()}",
        "note": "仅为诊断性残余估计（内存中）；不写回影像、不改变物化数据、不单独决定结论",
    }
    return _finalize_registration(result, cfg)


# --------------------------------------------------------------------------- D. 合成已知位移校准
def inject_translation(
    array: np.ndarray,
    *,
    spacing_xyz: Sequence[float],
    direction: Sequence[float],
    distance_mm: float,
    interpolation: str = "linear",
    border_mode: str = "nearest",
) -> tuple[np.ndarray, dict[str, Any]]:
    """在**内存中**给数组注入已知物理位移（mm）；返回新数组与记录（不写盘、不改原数组）。

    约定：`direction` 为施加在图像内容上的位移方向（单位向量）；0 mm 时直接复制（不做插值）。

    `border_mode` 默认为 `nearest`（复制边缘值）：**不得用常量零填充**——那会在图像一侧制造
    人为强边界，污染边缘一致性指标并破坏"位移越大指标越差"的单调性。边界模式由协议配置冻结。
    """
    arr = np.asarray(array, dtype=np.float64)
    vec = np.asarray(direction, dtype=np.float64)
    norm = float(np.linalg.norm(vec))
    if norm <= 0:
        raise ValueError("direction 不能为零向量")
    vec = vec / norm
    if border_mode == "constant":
        raise ValueError("border_mode=constant 会引入人为边界并污染指标，协议禁止使用")
    sp = np.asarray(spacing_xyz, dtype=np.float64)
    info_base = {
        "distance_mm": float(distance_mm),
        "direction_unit": [float(v) for v in vec],
        "interpolation": interpolation,
        "border_mode": border_mode,
    }
    if abs(float(distance_mm)) < 1e-12:
        info_base["offset_voxels_zyx"] = [0.0, 0.0, 0.0]
        return arr.copy(), info_base
    # 物理位移 → 体素偏移（数组轴序 z, y, x）
    offset_vox = np.asarray(
        [
            vec[2] * float(distance_mm) / sp[2],
            vec[1] * float(distance_mm) / sp[1],
            vec[0] * float(distance_mm) / sp[0],
        ],
        dtype=float,
    )
    order = 1 if interpolation == "linear" else 0
    shifted = ndimage.affine_transform(
        arr, np.eye(3), offset=-offset_vox, order=order, mode=border_mode, cval=0.0
    )
    info_base["offset_voxels_zyx"] = [float(v) for v in offset_vox]
    return np.asarray(shifted, dtype=np.float64), info_base


def build_calibration_conditions(cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    """确定性构建校准条件：0 mm（含固定 seed 的稳定性扰动重复）+ 各方向 × 各非零位移。"""
    conditions: list[dict[str, Any]] = []
    for index in range(max(1, int(cfg["stability_repeats"]))):
        conditions.append({"label": f"d0.0|noise{index}", "distance_mm": 0.0, "direction": None, "noise_index": index})
    for distance in (float(v) for v in cfg["displacements_mm"]):
        if abs(distance) < 1e-12:
            continue
        for index, vec in enumerate(cfg["directions"]):
            conditions.append(
                {
                    "label": f"d{distance:g}|dir{index}",
                    "distance_mm": distance,
                    "direction": [float(v) for v in vec],
                    "noise_index": None,
                }
            )
    return conditions


def pair_noise_offset(pair: str) -> int:
    """该 pair 的强度扰动 seed 偏移（未知 pair 显式失败，不静默按 0 处理）。"""
    if pair not in PAIR_NOISE_OFFSETS:
        raise ValueError(f"未知 pair: {pair}（允许值: {sorted(PAIR_NOISE_OFFSETS)}）")
    return int(PAIR_NOISE_OFFSETS[pair])


def _perturb_intensity(
    array: np.ndarray,
    *,
    noise_index: int,
    seed: int,
    sigma_relative: float,
    case_index: int,
    pair_offset: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """固定 seed 的强度扰动（稳定性/零位移离散度用）；sigma 按 p99−p1 相对缩放。

    `pair_offset` 让同一病例的两个序列对使用**不同**噪声实现（seed 去相关），仍然完全确定可复现。
    """
    arr = np.asarray(array, dtype=np.float64)
    if sigma_relative <= 0 or arr.size == 0:
        return arr.copy(), {"noise_sigma_relative": 0.0, "noise_index": int(noise_index)}
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return arr.copy(), {"noise_sigma_relative": 0.0, "noise_index": int(noise_index)}
    scale = float(np.percentile(finite, 99) - np.percentile(finite, 1))
    rng = np.random.default_rng(
        int(seed) + 1000 * int(case_index) + 10 * int(noise_index) + int(pair_offset)
    )
    out = arr.copy()
    mask = np.isfinite(arr)
    out[mask] = arr[mask] + rng.normal(0.0, float(sigma_relative) * max(scale, 1e-12), int(mask.sum()))
    return out, {
        "noise_sigma_relative": float(sigma_relative),
        "noise_index": int(noise_index),
        "pair_noise_offset": int(pair_offset),
        "intensity_scale_p99_minus_p1": scale,
    }


def calibration_records_for_case(
    case: Mapping[str, Any],
    *,
    cfg: Mapping[str, Any],
    prep: Mapping[str, Any],
    metrics_cfg: Mapping[str, Any],
    progress_cb: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """单个病例（单序列对）的全部校准记录：在内存中扰动/平移 moving 后重算指标。

    `case` = `{case_id, pair, t2w, moving, spacing_xyz, case_index}`（数组已同一网格）。
    调用方可在读取后立即调用并释放数组，避免长期持有全部影像。

    v0.3：记录按 `(case_id, pair)` 归组（unit），零位移重复用于建立该 unit 的自身参考。
    """
    conditions = build_calibration_conditions(cfg)
    seed = int(cfg["seed"])
    pair_offset = pair_noise_offset(str(case["pair"]))
    records: list[dict[str, Any]] = []
    for cond in conditions:
        if cond["noise_index"] is not None:
            moving_pert, info = _perturb_intensity(
                case["moving"],
                noise_index=int(cond["noise_index"]),
                seed=seed,
                sigma_relative=float(cfg["stability_noise_sigma"]),
                case_index=int(case.get("case_index", 0)),
                pair_offset=pair_offset,
            )
        else:
            moving_pert, info = inject_translation(
                case["moving"],
                spacing_xyz=case["spacing_xyz"],
                direction=cond["direction"],
                distance_mm=float(cond["distance_mm"]),
                interpolation=str(cfg["interpolation"]),
                border_mode=str(cfg.get("injection_border_mode", "nearest")),
            )
        metrics = pair_metrics_from_arrays(
            case["t2w"], moving_pert, case["spacing_xyz"], prep=prep, metrics_cfg=metrics_cfg
        )
        records.append(
            {
                "case_id": str(case["case_id"]),
                "pair": str(case["pair"]),
                "condition": cond["label"],
                "distance_mm": float(cond["distance_mm"]),
                "direction": cond["direction"],
                "perturbation": info,
                "metrics": metrics,
            }
        )
        if progress_cb is not None:
            progress_cb(f"{case['case_id']}/{case['pair']} {cond['label']}")
    return records


def run_calibration(
    cases: Sequence[Mapping[str, Any]],
    *,
    cfg: Mapping[str, Any],
    prep: Mapping[str, Any],
    metrics_cfg: Mapping[str, Any],
    progress_cb: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """合成已知位移校准（全自动、无人工标签）：对每个病例累积记录后统一汇总。"""
    records: list[dict[str, Any]] = []
    for case in cases:
        records.extend(
            calibration_records_for_case(
                case, cfg=cfg, prep=prep, metrics_cfg=metrics_cfg, progress_cb=progress_cb
            )
        )
    return summarize_calibration(records, cfg=cfg, metrics_cfg=metrics_cfg)


def _degradation(direction: str, reference: float, observed: float) -> float:
    """退化量：正值一律表示「更差」（方向由指标定义决定）。"""
    if direction == DIRECTION_LOWER_IS_WORSE:
        return float(reference) - float(observed)
    if direction == DIRECTION_HIGHER_IS_WORSE:
        return float(observed) - float(reference)
    raise ValueError(f"未知指标方向: {direction}")


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        return None
    as_float = float(value)
    return as_float if math.isfinite(as_float) else None


# --------------------------------------------------------------------- 指标可用性守卫（v0.4）
#: 尺度前缀模式：`sigma<数字>[.<数字>].`（尺度本身可能含小数点，如 `sigma1.5.`）
_SCALE_PREFIX_RE = re.compile(rf"^({re.escape(METRIC_SCALE_PREFIX)}\d+(?:\.\d+)?)\.")


def metric_scale_prefix(name: str) -> str | None:
    """解析指标名的尺度前缀：`sigma1.5.edge_f1_at_1.0mm` → `sigma1.5`；无法解析返回 None。

    注意尺度本身可能含小数点（`sigma1.5`），因此不能用「第一个点」简单切分。
    """
    match = _SCALE_PREFIX_RE.match(str(name))
    return match.group(1) if match else None


def metric_usable_key(name: str) -> str | None:
    """指标对应的可用性字段名：`sigma1.5.<metric>` → `sigma1.5.usable`。"""
    prefix = metric_scale_prefix(name)
    return f"{prefix}.{METRIC_USABLE_SUFFIX}" if prefix else None


def metric_usable(metrics: Mapping[str, Any], name: str) -> tuple[bool, str | None]:
    """返回 `(usable, reason)`；缺失 / False / 非布尔 一律 fail-closed（**不得默认为 true**）。"""
    key = metric_usable_key(name)
    if key is None:
        return False, (
            f"指标名 {name!r} 缺少 `{METRIC_SCALE_PREFIX}<scale>.` 前缀，无法解析 {METRIC_USABLE_SUFFIX}"
        )
    if key not in metrics:
        return False, f"缺少 {key}（fail-closed，不得默认为 true）"
    value = metrics.get(key)
    if isinstance(value, bool):
        if value:
            return True, None
        return False, f"{key}=false（{METRIC_UNUSABLE_EDGE_VOXELS}）"
    return False, f"{key} 非布尔值（{value!r}；fail-closed）"


def metric_value_with_reason(metrics: Mapping[str, Any], name: str) -> tuple[float | None, str | None]:
    """先做 usable 守卫（fail-closed），再取有限数值；返回 `(值, 不可用原因)`。"""
    usable, reason = metric_usable(metrics, name)
    if not usable:
        return None, reason
    raw = metrics.get(name)
    numeric = _finite_number(raw)
    if numeric is None:
        return None, f"指标 {name} 缺失或非有限（{raw!r}）"
    return numeric, None


def _direction_key(direction: Any) -> tuple[float, ...] | None:
    """方向向量归一化为可比较 tuple；None / 空 / 非有限 一律返回 None（fail-closed）。"""
    if direction is None or isinstance(direction, (str, bytes)):
        return None
    try:
        arr = np.asarray(direction, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if arr.size == 0 or not bool(np.all(np.isfinite(arr))):
        return None
    return tuple(float(v) for v in arr)


def pair_threshold_from_unit_midpoints(unit_midpoints: Mapping[str, float], *, aggregation: str) -> float:
    """pair 级真实病例阈值：对同一 pair 的 unit midpoint 做**预先声明**的稳健聚合（默认 median）。"""
    values = [float(v) for v in unit_midpoints.values() if math.isfinite(float(v))]
    if not values:
        raise ValueError("没有可用的 unit midpoint（不得回退到全局阈值）")
    array = np.asarray(values, dtype=float)
    if aggregation == "median":
        return float(np.median(array))
    if aggregation == "mean":
        return float(np.mean(array))
    raise ValueError(f"不支持的 pair 阈值聚合方法: {aggregation}（只允许 median / mean）")


def _pair_units(records: Sequence[Mapping[str, Any]], pair: str) -> list[tuple[str, str]]:
    return sorted({(str(r["case_id"]), str(r["pair"])) for r in records if str(r["pair"]) == pair})


def _unit_metric_detail(
    rows: Sequence[Mapping[str, Any]],
    *,
    case_id: str,
    pair: str,
    name: str,
    direction: str,
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    """单个 unit 的条件网格完整性审计 + 零位移参考 + 逐距离/方向退化（v0.4，fail-closed）。

    - 只接受「数值有限 **且** `sigma*.usable === true`」的记录；
    - 期望网格来自**配置**：`stability_repeats` 条零位移 + 每个非零 `displacements_mm` × 全部 `directions`；
    - 缺条件、额外条件、非法方向、非有限值、usable 缺失/false 一律写入
      `missing_conditions` / `invalid_conditions`，**不得静默缩小分母**。
    """
    detail: dict[str, Any] = {
        "case_id": case_id,
        "pair": pair,
        "available": False,
        "complete": False,
        "zero_reference": ZERO_REFERENCE_WITHIN_UNIT,
        "missing_conditions": [],
        "invalid_conditions": [],
    }
    expected_zero = max(0, int(cfg.get("stability_repeats", 0) or 0))
    expected_directions: list[tuple[float, ...]] = []
    for raw in cfg.get("directions") or []:
        key = _direction_key(raw)
        if key is None:
            detail["invalid_conditions"].append(
                {"condition": "config.directions", "reason": f"配置方向非法: {raw!r}"}
            )
            continue
        if key not in expected_directions:
            expected_directions.append(key)
    direction_labels = {key: f"dir{index}" for index, key in enumerate(expected_directions)}
    expected_distances = sorted({float(d) for d in (cfg.get("displacements_mm") or []) if float(d) > 0})
    min_detect = float(cfg["min_detectable_mm"])
    detail["min_detectable_mm"] = min_detect
    # ---- 零位移（期望 stability_repeats 条）
    zero_values: list[float] = []
    for record in rows:
        if abs(float(record.get("distance_mm", 0.0))) >= 1e-12:
            continue
        value, reason = metric_value_with_reason(record.get("metrics") or {}, name)
        if value is None:
            detail["invalid_conditions"].append({"condition": str(record.get("condition")), "reason": reason})
            continue
        zero_values.append(value)
    detail["n_zero_expected"] = expected_zero
    detail["n_zero_observed"] = len(zero_values)
    detail["zero_values"] = [float(v) for v in zero_values]
    if len(zero_values) != expected_zero:
        detail["missing_conditions"].append(
            f"zero:{expected_zero - len(zero_values)} 条缺失（expected={expected_zero}, observed={len(zero_values)}）"
        )
    zero_baseline: float | None = None
    if zero_values:
        zero_baseline = float(np.mean(zero_values))
        detail["zero_baseline"] = zero_baseline
        detail["zero_sd_within_unit"] = float(np.std(zero_values, ddof=1)) if len(zero_values) > 1 else 0.0
        leave_one_out: list[float] = []
        for index, value in enumerate(zero_values):
            others = [zero_values[j] for j in range(len(zero_values)) if j != index]
            reference = float(np.mean(others)) if others else zero_baseline
            leave_one_out.append(_degradation(direction, reference, value))
        detail["zero_leave_one_out_degradations"] = [float(v) for v in leave_one_out]
        detail["max_zero_leave_one_out_degradation"] = float(max(leave_one_out)) if leave_one_out else 0.0
    else:
        detail["zero_leave_one_out_degradations"] = []
        detail["max_zero_leave_one_out_degradation"] = None
    # ---- 非零位移 × 方向（期望网格来自配置）
    for record in rows:
        record_distance = float(record.get("distance_mm", 0.0))
        if record_distance <= 0:
            continue
        if not any(abs(record_distance - d) < 1e-12 for d in expected_distances):
            detail["invalid_conditions"].append(
                {"condition": str(record.get("condition")), "reason": f"配置 displacements_mm 之外的距离 {record_distance:g} mm"}
            )
    by_distance: dict[str, Any] = {}
    for distance in expected_distances:
        key = f"{distance:g}"
        observed: dict[tuple[float, ...], float] = {}
        for record in rows:
            if abs(float(record.get("distance_mm", 0.0)) - distance) >= 1e-12:
                continue
            direction_key = _direction_key(record.get("direction"))
            if direction_key is None:
                detail["invalid_conditions"].append(
                    {"condition": str(record.get("condition")), "reason": "direction 缺失/非有限"}
                )
                continue
            if direction_key not in expected_directions:
                detail["invalid_conditions"].append(
                    {
                        "condition": str(record.get("condition")),
                        "reason": f"配置 directions 之外的方向 {list(direction_key)}",
                    }
                )
                continue
            value, reason = metric_value_with_reason(record.get("metrics") or {}, name)
            if value is None:
                detail["invalid_conditions"].append({"condition": str(record.get("condition")), "reason": reason})
                continue
            observed[direction_key] = value
        missing = [d for d in expected_directions if d not in observed]
        for missing_direction in missing:
            detail["missing_conditions"].append(f"{key}|{direction_labels[missing_direction]}")
        ordered_values = [observed[d] for d in expected_directions if d in observed]
        degradations = (
            [_degradation(direction, zero_baseline, value) for value in ordered_values]
            if zero_baseline is not None
            else []
        )
        entry: dict[str, Any] = {
            "distance_mm": float(distance),
            "n_expected_directions": len(expected_directions),
            "n_observed_directions": len(observed),
            "expected_directions": [list(d) for d in expected_directions],
            "observed_directions": [list(d) for d in sorted(observed)],
            "missing_directions": [list(d) for d in missing],
            "values": [float(v) for v in ordered_values],
            "degradations": [float(v) for v in degradations],
            "complete": bool(not missing),
        }
        if ordered_values:
            entry["mean_value"] = float(np.mean(ordered_values))
        if degradations:
            entry["mean_degradation"] = float(np.mean(degradations))
            entry["degradation_sd"] = float(np.std(degradations, ddof=1)) if len(degradations) > 1 else 0.0
        by_distance[key] = entry
    detail["by_distance"] = by_distance
    detail["min_detectable_complete"] = bool(by_distance.get(f"{min_detect:g}", {}).get("complete"))
    detail["available"] = bool(
        zero_values and any(entry["n_observed_directions"] for entry in by_distance.values())
    )
    detail["complete"] = bool(
        zero_values
        and by_distance
        and not detail["missing_conditions"]
        and not detail["invalid_conditions"]
    )
    if not detail["complete"]:
        if detail["missing_conditions"]:
            detail["reason"] = "incomplete_condition_grid"
        elif detail["invalid_conditions"]:
            detail["reason"] = "invalid_conditions"
        else:
            detail["reason"] = "no_records"
    return detail


def _metric_calibration_for_pair(
    records: Sequence[Mapping[str, Any]],
    *,
    pair: str,
    spec: Mapping[str, str],
    cfg: Mapping[str, Any],
    threshold_aggregation: str,
) -> dict[str, Any]:
    """单个 pair × 单个指标的校准（v0.4）。

    - 只有 **complete** unit 参与噪声/曲线/单调性/FPR/midpoint；
    - `require_complete_condition_grid=true` 时，存在任何不完整 unit → 该 pair 直接失败（不静默缩小分母）；
    - complete unit 数 < `min_complete_units_per_pair` → 失败；
    - `min_detectable_mm` 缺少任意 unit / 任意方向 → 失败。
    """
    name = str(spec["name"])
    direction = str(spec["direction"])
    min_detect = float(cfg["min_detectable_mm"])
    multiplier = float(cfg["detection_multiplier"])
    min_complete = int(cfg.get("min_complete_units_per_pair", CALIBRATION_MIN_COMPLETE_UNITS))
    require_complete = bool(cfg.get("require_complete_condition_grid", REQUIRE_COMPLETE_CONDITION_GRID))
    units = _pair_units(records, pair)
    details: list[dict[str, Any]] = []
    for case_id, unit_pair in units:
        rows = [r for r in records if str(r["case_id"]) == case_id and str(r["pair"]) == unit_pair]
        details.append(
            _unit_metric_detail(rows, case_id=case_id, pair=pair, name=name, direction=direction, cfg=cfg)
        )
    complete = [d for d in details if d["complete"]]
    incomplete_ids = [str(d["case_id"]) for d in details if not d["complete"]]
    n_at_min = sum(1 for d in complete if d["min_detectable_complete"])
    participants = complete if require_complete else [d for d in details if d["available"]]
    excluded = (
        []
        if require_complete
        else [str(d["case_id"]) for d in details if d["available"] and not d["complete"]]
    )
    base: dict[str, Any] = {
        "pair": pair,
        "direction": direction,
        "role": str(spec["role"]),
        "min_detectable_mm": min_detect,
        "unit_details": details,
        "n_units_total": len(details),
        "n_units_complete": len(complete),
        "n_units_at_min_detectable": n_at_min,
        "n_units_participating": len(participants),
        "unit_ids": [str(d["case_id"]) for d in participants],
        "complete_unit_ids": [str(d["case_id"]) for d in complete],
        "incomplete_unit_ids": incomplete_ids,
        "excluded_unit_ids": excluded,
        "min_complete_units_per_pair": min_complete,
        "require_complete_condition_grid": require_complete,
        "condition_grid": {
            "stability_repeats": max(0, int(cfg.get("stability_repeats", 0) or 0)),
            "displacements_mm": [float(d) for d in (cfg.get("displacements_mm") or [])],
            "directions": [list(_direction_key(d) or []) for d in (cfg.get("directions") or [])],
        },
        "zero_reference": ZERO_REFERENCE_WITHIN_UNIT,
        "zero_noise_reference": ZERO_NOISE_LEAVE_ONE_OUT,
    }
    min_key = f"{min_detect:g}"
    units_missing_min = [
        str(d["case_id"]) for d in details if not bool((d.get("by_distance") or {}).get(min_key, {}).get("complete"))
    ]
    gate_reasons: list[str] = []
    if not details:
        gate_reasons.append("no_units: 该 pair 没有任何校准 unit")
    if require_complete and incomplete_ids:
        gate_reasons.append(f"incomplete_unit_condition_grid: {incomplete_ids}")
    if len(complete) < min_complete:
        gate_reasons.append(f"insufficient_complete_units: {len(complete)} < {min_complete}")
    if units_missing_min:
        gate_reasons.append(f"min_detectable_mm_uncovered_units: {units_missing_min}")
    if not participants:
        return {
            **base,
            "available": False,
            "passed": False,
            "reason": "该 pair 没有可参与统计的 unit（条件网格不完整或记录缺失）",
            "reasons": gate_reasons or ["no_participating_units"],
        }
    dof = sum(max(0, int(d["n_zero_observed"]) - 1) for d in participants)
    pooled_within_unit_sd = (
        math.sqrt(
            sum(max(0, int(d["n_zero_observed"]) - 1) * float(d["zero_sd_within_unit"]) ** 2 for d in participants)
            / dof
        )
        if dof > 0
        else 0.0
    )
    max_zero_degradation = max(float(d["max_zero_leave_one_out_degradation"] or 0.0) for d in participants)
    noise_threshold = max(multiplier * pooled_within_unit_sd, max_zero_degradation, 0.0)

    curve: list[dict[str, Any]] = []
    for distance in sorted({float(d) for d in (cfg.get("displacements_mm") or []) if float(d) > 0}):
        key = f"{distance:g}"
        degradations: list[float] = []
        per_unit: dict[str, float] = {}
        for detail in participants:
            entry = (detail.get("by_distance") or {}).get(key)
            if not entry or not entry.get("degradations"):
                continue
            degradations.extend(float(v) for v in entry["degradations"])
            per_unit[str(detail["case_id"])] = float(entry["mean_degradation"])
        if not degradations:
            continue
        curve.append(
            {
                "distance_mm": float(distance),
                "n_samples": len(degradations),
                "degradation_mean": float(np.mean(degradations)),
                "degradation_sd": float(np.std(degradations, ddof=1)) if len(degradations) > 1 else 0.0,
                "degradation_min": float(min(degradations)),
                "degradation_max": float(max(degradations)),
                "detection_rate": float(np.mean([1.0 if v > noise_threshold else 0.0 for v in degradations])),
                "per_unit_degradation_mean": per_unit,
            }
        )

    series = [{"distance_mm": 0.0, "aggregate_degradation": 0.0}]
    for entry in curve:
        per_unit_values = list(entry["per_unit_degradation_mean"].values())
        series.append(
            {
                "distance_mm": float(entry["distance_mm"]),
                "aggregate_degradation": float(np.mean(per_unit_values)) if per_unit_values else 0.0,
            }
        )
    pairs_total = 0
    pairs_ok = 0
    for i in range(len(series)):
        for j in range(i + 1, len(series)):
            pairs_total += 1
            if float(series[j]["aggregate_degradation"]) > float(series[i]["aggregate_degradation"]):
                pairs_ok += 1
    monotonicity = float(pairs_ok / pairs_total) if pairs_total else 0.0

    at_min = next((c for c in curve if abs(float(c["distance_mm"]) - min_detect) < 1e-9), None)
    detection_rate_min = float(at_min["detection_rate"]) if at_min else 0.0
    per_unit_at_min = dict(at_min["per_unit_degradation_mean"]) if at_min else {}
    aggregate_at_min = float(np.mean(list(per_unit_at_min.values()))) if per_unit_at_min else None

    zero_samples = [float(v) for d in participants for v in d["zero_leave_one_out_degradations"]]
    zero_false_positive_rate = (
        float(np.mean([1.0 if v > noise_threshold else 0.0 for v in zero_samples])) if zero_samples else 0.0
    )

    unit_zero_baselines: dict[str, float] = {}
    unit_min_detectable_means: dict[str, float] = {}
    unit_midpoints: dict[str, float] = {}
    for detail in complete:  # v0.4：只有 complete unit 才能生成 midpoint
        entry = (detail.get("by_distance") or {}).get(min_key)
        if not entry or not entry.get("complete"):
            continue
        case_id = str(detail["case_id"])
        unit_zero_baselines[case_id] = float(detail["zero_baseline"])
        unit_min_detectable_means[case_id] = float(entry["mean_value"])
        unit_midpoints[case_id] = float((unit_zero_baselines[case_id] + unit_min_detectable_means[case_id]) / 2.0)
    threshold = None
    if unit_midpoints:
        try:
            threshold = pair_threshold_from_unit_midpoints(unit_midpoints, aggregation=str(threshold_aggregation))
        except ValueError:
            threshold = None

    reasons: list[str] = list(gate_reasons)
    if monotonicity < float(cfg["monotonicity_min"]):
        reasons.append(f"monotonicity={monotonicity:.3f} < {float(cfg['monotonicity_min'])}")
    if detection_rate_min < float(cfg["require_detection_rate_at_min"]):
        reasons.append(
            f"detection_rate@ {min_detect}mm = {detection_rate_min:.3f} < "
            f"{float(cfg['require_detection_rate_at_min'])}"
        )
    if zero_false_positive_rate > float(cfg["max_zero_false_positive_rate"]):
        reasons.append(
            f"zero_false_positive_rate={zero_false_positive_rate:.3f} > "
            f"{float(cfg['max_zero_false_positive_rate'])}"
        )
    if at_min is None:
        reasons.append(f"缺少 {min_detect} mm 的校准条件")
    elif aggregate_at_min is None or aggregate_at_min <= 0.0:
        reasons.append("min_detectable 处 unit 内退化未变差（方向不符）")
    if threshold is None:
        reasons.append(f"缺少 {min_detect} mm 的 unit midpoint（无法推导 pair 阈值）")

    return {
        **base,
        "available": True,
        "noise_threshold": float(noise_threshold),
        "noise_threshold_formula": NOISE_THRESHOLD_FORMULA,
        "pooled_within_unit_zero_sd": float(pooled_within_unit_sd),
        "max_within_unit_zero_degradation": float(max_zero_degradation),
        "detection_multiplier": multiplier,
        "degradation_definition": DEGRADATION_DEFINITION,
        "detection_rate_definition": DETECTION_RATE_DEFINITION,
        "monotonicity": monotonicity,
        "monotonicity_definition": MONOTONICITY_DEFINITION,
        "monotonicity_series": series,
        "curve": curve,
        "detection_rate_at_min_detectable": detection_rate_min,
        "aggregate_degradation_at_min_detectable": aggregate_at_min,
        "per_unit_degradation_at_min_detectable": per_unit_at_min,
        "zero_false_positive_rate": zero_false_positive_rate,
        "n_zero_samples": len(zero_samples),
        "unit_zero_baselines": unit_zero_baselines,
        "unit_min_detectable_means": unit_min_detectable_means,
        "unit_midpoints": unit_midpoints,
        "threshold": threshold,
        "pair_threshold_aggregation": str(threshold_aggregation),
        "metric_usable_key": metric_usable_key(name),
        "passed": not reasons,
        "reasons": reasons,
    }


def summarize_calibration(
    records: Sequence[Mapping[str, Any]],
    *,
    cfg: Mapping[str, Any],
    metrics_cfg: Mapping[str, Any],
    threshold_aggregation: str = PAIR_THRESHOLD_AGGREGATION,
) -> dict[str, Any]:
    """v0.3 校准汇总：**按 `(pair, case_id)` unit 分组**，噪声只来自 unit 内零位移残差。

    与 v0.2 的关键差异（方法性变更，见 `docs/protocols/G0_R_ALIGNMENT_QC_AUTOMATED.md` §7）：

    - v0.2 把所有病例、两个 pair 的零位移**绝对值**混进同一个分布，`zero_sd` 主要反映病例/pair 间的
      基线差异，从而淹没「同一 unit 内由位移引起的退化」；本版退化量一律相对**本 unit 的零位移均值**；
    - 零位移噪声阈值 = `max(k × pooled_within_unit_zero_sd, max(unit 内 leave-one-out 零位移退化), 0)`：
      pooled SD 只含 unit 内残差；下限保证 `zero_false_positive_rate = 0`（不把自身算进参考均值）；
    - 两个 pair 完全独立汇总；`passed` 要求**每个 pair** 的主指标各自通过（配置
      `calibration.require_each_pair_primary_pass`）。
    """
    specs = primary_and_consistency_metrics(metrics_cfg)
    min_detect = float(cfg["min_detectable_mm"])
    # 输入顺序归一化：统计量（含浮点求和顺序）不得依赖调用方给出记录的顺序
    records = sorted(
        records,
        key=lambda r: (
            str(r.get("pair")),
            str(r.get("case_id")),
            float(r.get("distance_mm", 0.0)),
            str(r.get("condition", "")),
            str(r.get("direction")),
        ),
    )
    pairs = sorted({str(r["pair"]) for r in records})
    per_pair: dict[str, Any] = {}
    for pair in pairs:
        units = _pair_units(records, pair)
        per_metric: dict[str, Any] = {}
        for spec in specs:
            per_metric[str(spec["name"])] = _metric_calibration_for_pair(
                records,
                pair=pair,
                spec=spec,
                cfg=cfg,
                threshold_aggregation=str(threshold_aggregation),
            )
        primary = str(metrics_cfg["primary_metric"])
        primary_info = per_metric.get(primary) or {}
        pair_ok = bool(primary_info.get("passed"))
        if pair_ok:
            pair_reasons: list[str] = []
        else:
            details = "; ".join(str(v) for v in (primary_info.get("reasons") or [])) or str(
                primary_info.get("reason", "unknown")
            )
            pair_reasons = [f"primary_metric_not_calibratable[{pair}]: {primary}（{details}）"]
        per_pair[pair] = {
            "pair": pair,
            # v0.4：计数以**主指标**的条件网格审计为准，明确区分 total / complete / at-min-detectable
            "counts_metric": primary,
            "n_units_total": int(primary_info.get("n_units_total", len(units))),
            "n_units_complete": int(primary_info.get("n_units_complete", 0)),
            "n_units_at_min_detectable": int(primary_info.get("n_units_at_min_detectable", 0)),
            "n_units_participating": int(primary_info.get("n_units_participating", 0)),
            "incomplete_unit_ids": list(primary_info.get("incomplete_unit_ids") or []),
            "min_complete_units_per_pair": int(
                primary_info.get("min_complete_units_per_pair", CALIBRATION_MIN_COMPLETE_UNITS)
            ),
            "require_complete_condition_grid": bool(
                primary_info.get("require_complete_condition_grid", REQUIRE_COMPLETE_CONDITION_GRID)
            ),
            "units": [f"{case_id}/{unit_pair}" for case_id, unit_pair in units],
            "primary_metric": primary,
            "primary_passed": pair_ok,
            "per_metric": per_metric,
            "passed": pair_ok,
            "reasons": pair_reasons,
        }
    required_pairs = sorted(PAIR_TO_SERIES)
    missing_pairs = [p for p in required_pairs if p not in per_pair]
    require_each = bool(cfg.get("require_each_pair_primary_pass", True))
    require_complete_grid = bool(cfg.get("require_complete_condition_grid", REQUIRE_COMPLETE_CONDITION_GRID))
    reasons: list[str] = []
    for pair in sorted(per_pair):
        if not per_pair[pair]["passed"]:
            reasons.extend(per_pair[pair]["reasons"])
    if missing_pairs and require_each:
        reasons.append("missing_pair_calibration: " + ", ".join(missing_pairs))
    if not records:
        reasons.append("no_calibration_cases: 没有任何可用于校准的合格病例")
    passed = bool(
        per_pair
        and all(per_pair[p]["passed"] for p in per_pair)
        and (not missing_pairs or not require_each)
    )
    return {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "calibration_kind": "synthetic_known_displacement",
        "grouping": list(CALIBRATION_GROUPING),
        "unit_definition": "(case_id, pair)",
        "zero_reference": ZERO_REFERENCE_WITHIN_UNIT,
        "zero_noise_reference": ZERO_NOISE_LEAVE_ONE_OUT,
        "noise_threshold_formula": NOISE_THRESHOLD_FORMULA,
        "degradation_definition": DEGRADATION_DEFINITION,
        "detection_rate_definition": DETECTION_RATE_DEFINITION,
        "monotonicity_definition": MONOTONICITY_DEFINITION,
        "pair_threshold_aggregation": str(threshold_aggregation),
        "conditions": [c["label"] for c in build_calibration_conditions(cfg)],
        "cases_used": sorted({str(r["case_id"]) for r in records}),
        "pairs_used": pairs,
        "required_pairs": required_pairs,
        "require_each_pair_primary_pass": require_each,
        "missing_pairs": missing_pairs,
        "n_records": len(records),
        "n_units_total": sum(int(per_pair[p]["n_units_total"]) for p in per_pair),
        "n_units_complete": sum(int(per_pair[p]["n_units_complete"]) for p in per_pair),
        "n_units_at_min_detectable": sum(int(per_pair[p]["n_units_at_min_detectable"]) for p in per_pair),
        "incomplete_unit_ids": sorted(
            {unit_id for p in per_pair for unit_id in per_pair[p]["incomplete_unit_ids"]}
        ),
        "min_complete_units_per_pair": int(cfg.get("min_complete_units_per_pair", CALIBRATION_MIN_COMPLETE_UNITS)),
        "require_complete_condition_grid": require_complete_grid,
        "min_detectable_mm": min_detect,
        "primary_metric": str(metrics_cfg["primary_metric"]),
        "per_pair": per_pair,
        "passed": passed,
        "reasons": reasons,
        "note": (
            "v0.4：两个 pair 完全独立；噪声阈值只来自 unit 内零位移残差（pooled within-unit SD）；"
            "只有 complete unit 参与统计与 midpoint，且存在不完整 unit（require_complete_condition_grid）"
            "或 complete unit 不足时该 pair 直接失败；指标必须 `sigma*.usable=true`（边缘体素不足一律 fail-closed）"
        ),
    }


def derive_thresholds(
    calibration: Mapping[str, Any], *, metrics_cfg: Mapping[str, Any], thresholds_cfg: Mapping[str, Any]
) -> dict[str, Any]:
    """v0.3 阈值推导：**按 pair** 对 unit midpoint 做稳健聚合（默认 median），不存在全局阈值。

    - `unit_midpoint(u) = (unit 零位移基线 + unit 在 min_detectable_mm 处各方向均值) / 2`；
    - `threshold(pair, metric) = median(unit_midpoint(u, metric))`（聚合方法由配置显式声明）；
    - 记录每个 unit 的 baseline / min-detectable 均值 / midpoint、聚合方法、指标方向与灵敏度校准状态；
    - 任一 pair 不可用即该 pair 不可用；**不得回退到全局阈值**（未知 pair 在 `decide_case` 直接失败）。
    """
    derivation = str(thresholds_cfg["derivation"])
    if derivation != THRESHOLD_DERIVATION_BY_PAIR:
        raise ValueError(
            f"不支持的阈值推导公式: {derivation}（协议只允许 {THRESHOLD_DERIVATION_BY_PAIR}）"
        )
    aggregation = str(thresholds_cfg.get("aggregation_by_pair", PAIR_THRESHOLD_AGGREGATION))
    if aggregation not in ("median", "mean"):
        raise ValueError(f"不支持的 pair 阈值聚合方法: {aggregation}（只允许 median / mean）")
    recorded_aggregation = str(calibration.get("pair_threshold_aggregation", "") or "")
    if recorded_aggregation and recorded_aggregation != aggregation:
        raise ValueError(
            f"校准记录与阈值配置的聚合方法不一致: {recorded_aggregation} vs {aggregation}（不得静默混用）"
        )
    pairs = [str(p) for p in (calibration.get("pairs_used") or [])]
    out: dict[str, Any] = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "derivation": derivation,
        "aggregation_by_pair": aggregation,
        "direction": "from_metric",
        "unit_midpoint_definition": "(unit 零位移基线 + unit 在 min_detectable_mm 处各方向均值) / 2",
        "sensitivity_gate_separate_from_case_threshold": True,
        "pairs": pairs,
        "metrics": {},
    }
    for spec in primary_and_consistency_metrics(metrics_cfg):
        name = str(spec["name"])
        by_pair: dict[str, Any] = {}
        for pair in pairs:
            info = (((calibration.get("per_pair") or {}).get(pair) or {}).get("per_metric") or {}).get(name) or {}
            entry: dict[str, Any] = {"pair": pair, "direction": str(info.get("direction", spec["direction"]))}
            completeness = {
                "n_units_total": int(info.get("n_units_total", 0)),
                "n_units_complete": int(info.get("n_units_complete", 0)),
                "n_units_at_min_detectable": int(info.get("n_units_at_min_detectable", 0)),
                "incomplete_unit_ids": list(info.get("incomplete_unit_ids") or []),
                "min_complete_units_per_pair": int(
                    info.get("min_complete_units_per_pair", CALIBRATION_MIN_COMPLETE_UNITS)
                ),
                "require_complete_condition_grid": bool(
                    info.get("require_complete_condition_grid", REQUIRE_COMPLETE_CONDITION_GRID)
                ),
                "metric_usable_key": info.get("metric_usable_key"),
            }
            if not info.get("available"):
                by_pair[pair] = {
                    **entry,
                    **completeness,
                    "available": False,
                    "sensitivity_passed": False,
                    "reasons": list(info.get("reasons") or [info.get("reason", "该 pair 指标不可用")]),
                }
                continue
            midpoints = {str(k): float(v) for k, v in (info.get("unit_midpoints") or {}).items()}
            if not midpoints:
                by_pair[pair] = {
                    **entry,
                    **completeness,
                    "available": False,
                    "sensitivity_passed": bool(info.get("passed")),
                    "reasons": ["缺少 min_detectable 处的 unit midpoint（无法推导 pair 阈值）"],
                }
                continue
            threshold = pair_threshold_from_unit_midpoints(midpoints, aggregation=aggregation)
            recorded = info.get("threshold")
            if recorded is None or abs(float(recorded) - float(threshold)) > 1e-9:
                raise ValueError(
                    f"校准记录的 threshold 与重新推导不一致（{name} / {pair}）: {recorded!r} vs {threshold!r}"
                )
            by_pair[pair] = {
                **entry,
                **completeness,
                "available": True,
                "threshold": float(threshold),
                "n_units": len(midpoints),
                "unit_zero_baselines": {k: float(v) for k, v in (info.get("unit_zero_baselines") or {}).items()},
                "unit_min_detectable_means": {
                    k: float(v) for k, v in (info.get("unit_min_detectable_means") or {}).items()
                },
                "unit_midpoints": midpoints,
                "min_detectable_mm": float(info.get("min_detectable_mm") or 0.0),
                "sensitivity_passed": bool(info.get("passed")),
                "sensitivity_reasons": list(info.get("reasons") or []),
            }
        available_for_all = bool(by_pair) and all(bool(item.get("available")) for item in by_pair.values())
        out["metrics"][name] = {
            "role": str(spec["role"]),
            "direction": str(spec["direction"]),
            "available": available_for_all,
            "by_pair": by_pair,
        }
    return out


# --------------------------------------------------------------------------- 自动判定
def decide_case(
    metrics: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    calibration: Mapping[str, Any],
    *,
    pair: str,
    metrics_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    """单例（**指定 pair**）自动判定：ACCEPTABLE / FLAGGED / INSUFFICIENT_EVIDENCE。

    v0.4（必须显式传 `pair`）：

    - **指标可用性守卫**：先查 `sigma*.usable`；缺失 / false / 非布尔一律 fail-closed，并记录
      `metric_unusable_reason`。主指标 unusable → `UNAVAILABLE` → `INSUFFICIENT_EVIDENCE`；
      一致性指标 unusable → `SKIPPED`（写明边缘体素不足），不得用于支持 ACCEPTABLE / FLAGGED；
    - 只使用**该 pair 自己**的校准单元与阈值；未知 pair 直接抛错（**禁止跨 pair 混用或回退全局阈值**）；
    - 主指标必须在该 pair 上可校准且阈值可用，否则该例为 INSUFFICIENT_EVIDENCE；
    - 一致性指标若未通过该 pair 的合成校准（对位移不敏感/饱和）→ 记为 SKIPPED 并排除出判定（如实记录）；
    - 通过校准的指标之间若结论冲突 → INSUFFICIENT_EVIDENCE。
    """
    pair = str(pair)
    pairs = [str(p) for p in (thresholds.get("pairs") or [])]
    if pair not in pairs:
        raise ValueError(
            f"未知 pair: {pair}（阈值按 pair 存储；允许值: {pairs or '（无）'}；"
            "禁止跨 pair 混用或回退到全局阈值）"
        )
    pair_calibration = (calibration.get("per_pair") or {}).get(pair) or {}
    checks: list[dict[str, Any]] = []
    for spec in primary_and_consistency_metrics(metrics_cfg):
        name = str(spec["name"])
        entry = (((thresholds.get("metrics") or {}).get(name) or {}).get("by_pair") or {}).get(pair) or {}
        cal = (pair_calibration.get("per_metric") or {}).get(name) or {}
        if entry.get("pair") not in (None, pair):
            raise ValueError(f"阈值 pair 标签不一致: {entry.get('pair')} != {pair}（拒绝静默混用）")
        numeric, unusable_reason = metric_value_with_reason(metrics, name)
        check_base: dict[str, Any] = {
            "pair": pair,
            "metric": name,
            "role": spec["role"],
            "metric_usable_key": metric_usable_key(name),
            "metric_usable": unusable_reason is None,
            "metric_unusable_reason": unusable_reason,
        }
        if unusable_reason is not None:
            # v0.4：usable 守卫优先（fail-closed）。主指标 → UNAVAILABLE；一致性指标 → SKIPPED
            primary_role = spec["role"] == "primary"
            checks.append(
                {
                    **check_base,
                    "decision": "UNAVAILABLE" if primary_role else "SKIPPED",
                    "reason": (
                        f"metric_unusable（{'主指标' if primary_role else '一致性指标'}）: {unusable_reason}"
                        + ("" if primary_role else "；该指标不得用于支持 ACCEPTABLE/FLAGGED，仅记录")
                    ),
                    "value": numeric,
                    "threshold": entry.get("threshold"),
                    "calibration_reasons": list(cal.get("reasons") or []),
                }
            )
            continue
        if spec["role"] != "primary" and not cal.get("passed"):
            checks.append(
                {
                    **check_base,
                    "decision": "SKIPPED",
                    "reason": "该一致性指标在该 pair 上未通过合成校准（对注入位移不敏感/饱和）→ 排除出判定，仅记录",
                    "value": numeric,
                    "threshold": entry.get("threshold"),
                    "calibration_reasons": list(cal.get("reasons") or []),
                }
            )
            continue
        if numeric is None or not entry.get("available") or not cal.get("passed"):
            checks.append(
                {
                    **check_base,
                    "metric_usable": True,
                    "metric_unusable_reason": None,
                    "decision": "UNAVAILABLE",
                    "reason": "指标缺失/非有限，或该 pair 未通过合成校准（不得据此判定）",
                    "value": numeric,
                    "threshold": entry.get("threshold"),
                }
            )
            continue
        threshold = float(entry["threshold"])
        if spec["direction"] == DIRECTION_LOWER_IS_WORSE:
            acceptable = numeric >= threshold
        else:
            acceptable = numeric <= threshold
        checks.append(
            {
                **check_base,
                "decision": CASE_ACCEPTABLE if acceptable else CASE_FLAGGED,
                "value": numeric,
                "threshold": threshold,
                "direction": spec["direction"],
                "sensitivity_passed": bool(entry.get("sensitivity_passed")),
            }
        )
    decisions = {c["decision"] for c in checks}
    judged = decisions & {CASE_ACCEPTABLE, CASE_FLAGGED}
    if "UNAVAILABLE" in decisions:
        unavailable = [c for c in checks if c["decision"] == "UNAVAILABLE"]
        detail = "; ".join(
            str(c.get("metric_unusable_reason") or c.get("metric") or "unknown") for c in unavailable[:3]
        )
        status, reason = (
            CASE_INSUFFICIENT,
            f"[{pair}] 主指标不可用（缺失/非有限/usable=false/该 pair 未通过校准）：{detail}",
        )
    elif CASE_ACCEPTABLE in judged and CASE_FLAGGED in judged:
        status, reason = CASE_INSUFFICIENT, f"[{pair}] 通过校准的指标间结论冲突（部分 ACCEPTABLE、部分 FLAGGED）"
    elif judged == {CASE_FLAGGED}:
        status, reason = CASE_FLAGGED, f"[{pair}] 参与判定的指标均超出该 pair 的自动阈值"
    elif judged == {CASE_ACCEPTABLE}:
        status, reason = CASE_ACCEPTABLE, f"[{pair}] 参与判定的指标均未超出该 pair 的自动阈值"
    else:  # pragma: no cover - 防御性：主指标异常时由 UNAVAILABLE 分支覆盖
        status, reason = CASE_INSUFFICIENT, f"[{pair}] 没有任何指标可参与判定（全部被跳过或不可用）"
    return {"status": status, "reason": reason, "pair": pair, "checks": checks}


def decide_overall(
    per_case: Sequence[Mapping[str, Any]],
    calibration: Mapping[str, Any],
    *,
    decision_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    """总体候选决策（DRAFT）：resample-only / resample+rigid / INSUFFICIENT_EVIDENCE。

    判定逻辑冻结（改变必须改代码版本）：校准未通过、异常比例超限、指标冲突 → INSUFFICIENT_EVIDENCE；
    任一 case×pair FLAGGED → resample+rigid；否则 resample-only。
    """
    if decision_cfg.get("all_pairs_must_be_acceptable_for_resample_only") is not True:
        raise ValueError("decision.all_pairs_must_be_acceptable_for_resample_only 必须为 true（判定逻辑需版本化变更）")
    if decision_cfg.get("any_flagged_pair_triggers_rigid") is not True:
        raise ValueError("decision.any_flagged_pair_triggers_rigid 必须为 true（判定逻辑需版本化变更）")
    reasons: list[str] = []
    if not calibration.get("passed"):
        reasons.append("calibration_not_passed: " + "; ".join(calibration.get("reasons") or ["unknown"]))
    per_pair_calibration = dict(calibration.get("per_pair") or {})
    missing_pair_calibration = sorted(
        {str(c.get("pair")) for c in per_case} - set(per_pair_calibration.keys())
    )
    if missing_pair_calibration:
        reasons.append("missing_pair_calibration: " + ", ".join(missing_pair_calibration))
    failed_pair_calibration = sorted(
        pair for pair, info in per_pair_calibration.items() if not bool((info or {}).get("passed"))
    )
    if failed_pair_calibration:
        reasons.append("pair_calibration_failed: " + ", ".join(failed_pair_calibration))
    total = len(per_case)

    def fraction(status: str) -> float:
        return (sum(1 for c in per_case if c.get("status") == status) / total) if total else 0.0

    fractions = {
        CASE_ACCEPTABLE: fraction(CASE_ACCEPTABLE),
        CASE_FLAGGED: fraction(CASE_FLAGGED),
        CASE_FOV_INSUFFICIENT: fraction(CASE_FOV_INSUFFICIENT),
        CASE_INVALID_INPUT: fraction(CASE_INVALID_INPUT),
        CASE_REGISTRATION_FAILED: fraction(CASE_REGISTRATION_FAILED),
        CASE_INSUFFICIENT: fraction(CASE_INSUFFICIENT),
    }
    limits = (
        ("invalid_input_fraction_above_max", CASE_INVALID_INPUT, float(decision_cfg["max_invalid_input_fraction"])),
        ("fov_insufficient_fraction_above_max", CASE_FOV_INSUFFICIENT, float(decision_cfg["max_fov_insufficient_fraction"])),
        (
            "registration_failed_fraction_above_max",
            CASE_REGISTRATION_FAILED,
            float(decision_cfg["max_registration_failed_fraction"]),
        ),
    )
    for label, status, limit in limits:
        if fractions[status] > limit:
            reasons.append(f"{label}: {fractions[status]:.3f} > {limit}")
    if any(c.get("status") == CASE_INSUFFICIENT for c in per_case):
        reasons.append("any_case_metric_conflict")
    if total == 0:
        reasons.append("no_cases")
    if reasons:
        candidate = DECISION_INSUFFICIENT
    elif fractions[CASE_FLAGGED] > 0:
        candidate = DECISION_RESAMPLE_RIGID
    else:
        candidate = DECISION_RESAMPLE_ONLY
    if candidate not in ALLOWED_CANDIDATES:  # pragma: no cover - 防御性
        candidate = DECISION_INSUFFICIENT
    return {
        "candidate": candidate,
        "draft": True,
        "rule_version": str(decision_cfg.get("rule_version", "unknown")),
        "calibration_grouping": list(calibration.get("grouping") or CALIBRATION_GROUPING),
        "pair_calibration_passed": {
            str(pair): bool((info or {}).get("passed")) for pair, info in sorted(per_pair_calibration.items())
        },
        "pairs_without_calibration": missing_pair_calibration,
        "reasons": reasons,
        "fractions": fractions,
        "n_rows": total,
        "n_unique_cases": len({str(c["case_id"]) for c in per_case}),
        "rigid_required_by": sorted(
            {f"{c['case_id']}/{c['pair']}" for c in per_case if c.get("status") == CASE_FLAGGED}
        ),
        "fov_insufficient_cases": sorted(
            {f"{c['case_id']}/{c['pair']}" for c in per_case if c.get("status") == CASE_FOV_INSUFFICIENT}
        ),
        "invalid_input_cases": sorted(
            {f"{c['case_id']}/{c['pair']}" for c in per_case if c.get("status") == CASE_INVALID_INPUT}
        ),
        "note": "自动候选为 DRAFT；冻结须由研究者按协议执行；本输出不等于 G0-R PASS",
    }


__all__ = [
    "ALLOWED_CANDIDATES",
    "CALIBRATION_GROUPING",
    "CALIBRATION_MIN_COMPLETE_UNITS",
    "CASE_ACCEPTABLE",
    "CASE_FLAGGED",
    "CASE_FOV_INSUFFICIENT",
    "CASE_INSUFFICIENT",
    "CASE_INVALID_INPUT",
    "CASE_REGISTRATION_FAILED",
    "CASE_STATUSES",
    "DECISION_INSUFFICIENT",
    "DECISION_RESAMPLE_ONLY",
    "DECISION_RESAMPLE_RIGID",
    "DEGRADATION_DEFINITION",
    "DETECTION_RATE_DEFINITION",
    "DIRECTION_HIGHER_IS_WORSE",
    "DIRECTION_LOWER_IS_WORSE",
    "METRIC_SCALE_PREFIX",
    "METRIC_UNUSABLE_EDGE_VOXELS",
    "METRIC_USABLE_SUFFIX",
    "MONOTONICITY_DEFINITION",
    "NOISE_THRESHOLD_FORMULA",
    "OUTPUT_SCHEMA_VERSION",
    "PAIR_NOISE_OFFSETS",
    "PAIR_THRESHOLD_AGGREGATION",
    "PAIR_TO_SERIES",
    "PROTOCOL_HASH_EXCLUDED_TOP_LEVEL",
    "REQUIRE_COMPLETE_CONDITION_GRID",
    "THRESHOLD_DERIVATION_BY_PAIR",
    "ZERO_NOISE_LEAVE_ONE_OUT",
    "ZERO_REFERENCE_WITHIN_UNIT",
    "array_to_image",
    "as_euler3d_transform",
    "assert_automated_input_policy",
    "binary_edges",
    "build_calibration_conditions",
    "calibration_records_for_case",
    "check_array_sanity",
    "check_geometry_sanity",
    "decide_case",
    "decide_overall",
    "derive_thresholds",
    "distance_map_mm",
    "edge_agreement_metrics",
    "estimate_rigid_diagnostic",
    "flatten_metric_row",
    "fov_evidence",
    "inject_translation",
    "metric_scale_prefix",
    "metric_usable",
    "metric_usable_key",
    "metric_value_with_reason",
    "pair_metrics_from_arrays",
    "pair_noise_offset",
    "pair_threshold_from_unit_midpoints",
    "physical_gradient_magnitude",
    "primary_and_consistency_metrics",
    "protocol_hash",
    "robust_normalize",
    "run_calibration",
    "slicewise_agreement",
    "summarize_calibration",
]
