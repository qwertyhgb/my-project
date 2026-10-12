"""病例级（case-level）指标：物理体积、连通域失败分析、表面距离指标。

对应论文新主线的 **primary endpoint**（positive-case macro Dice 的分子/分母来源）
与若干 **exploratory** 终点（HD95 / ASSD / NSD@τ）。全部为**纯函数**：不读文件、不写文件、
不依赖 nnU-Net。IO 与编排留在 ``scripts/evaluate_segmentation.py``。

空值与漏检语义（协议冻结）
---------------------------
- 阳性病例完全漏分（预测为空）时 Dice 记 **0**，不得排除该病例。
- ``n_ref == 0``（阴性病例）：真阴（空-空）Dice 为 ``None``，不是 0，也不是 1。
- GT 非空、预测为空：NSD = 0，HD95/ASSD = ``None``，该病例同时计入 missed。
- GT 与预测均为空：不纳入阳性病灶表面统计。
- 评估**不重采样掩膜**：真实几何不一致时直接 fail-closed（数值舍入由
  :func:`geometry_check` 的绝对容差与物理角点阈值吸收，见协议 §3）。
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np

from zonal_reliability_fusion.evaluation.protocol import (
    CONNECTIVITY,
    GEOMETRY_ATOL_DIRECTION,
    GEOMETRY_ATOL_MM,
    GEOMETRY_CORNER_ATOL_MM,
    EvaluationError,
    _finite,
    _ratio,
)


def voxel_volume_mm3(spacing_xyz: Sequence[float]) -> float:
    """单voxel 的物理体积（mm³）。``spacing_xyz`` 是 SimpleITK 的 (sx, sy, sz)。"""
    vol = 1.0
    for s in spacing_xyz:
        vol *= float(s)
    return vol


def spacing_zyx(spacing_xyz: Sequence[float]) -> tuple[float, ...]:
    """SimpleITK 的 spacing 是 (sx, sy, sz)，而数组轴序是 (z, y, x)，故需反转。"""
    return tuple(float(s) for s in reversed(list(spacing_xyz)))


def surface_distance_metrics():
    """延迟导入 DeepMind ``surface-distance``（NSD 的 surfel-area weighted 实现）。

    缺失时直接抛出 :class:`EvaluationError`（fail-closed）：**不会**静默退回旧的
    表面体素计数实现（那会改变已声明的论文指标口径）。
    """
    try:
        from surface_distance import metrics as surface_distance_metrics
    except ImportError as exc:  # pragma: no cover - 依赖缺失时才会走到
        raise EvaluationError(
            "缺少 surface-distance 包：NSD 必须按 surfel-area weighted 物理表面测度计算"
            "（DeepMind surface-distance==0.1），不得退回表面体素计数实现。"
            "请安装 surface-distance==0.1 后重试。"
        ) from exc
    return surface_distance_metrics


def nsd_surface_area(
    pred: np.ndarray, ref: np.ndarray, spacing_zyx: Sequence[float], tolerance_mm: float
) -> float | None:
    """NSD@τ：surfel-area weighted 物理表面 normalized surface Dice。

    使用 DeepMind ``surface-distance`` 0.1 官方实现，按物理表面测度 μ（surfel 面积，mm²）
    加权，而不是表面体素计数::

        NSD = (μ(S_pred ∩ B_τ(S_ref)) + μ(S_ref ∩ B_τ(S_pred))) / (μ(S_pred) + μ(S_ref))

    ``spacing_zyx`` 必须是**数组轴序** (z, y, x) 的真实 mm 间距（见 :func:`spacing_zyx`）。
    任一侧为空时本函数返回 ``None``（空掩膜口径由 :func:`surface_metrics` 决定）。
    缺少 surface-distance 时 fail-closed，不退回旧算法。
    """
    pred_mask = np.asarray(pred, dtype=bool)
    ref_mask = np.asarray(ref, dtype=bool)
    if pred_mask.shape != ref_mask.shape:
        raise EvaluationError(
            f"NSD 的 prediction 与 reference 形状不一致："
            f"{pred_mask.shape} vs {ref_mask.shape}"
        )
    if not pred_mask.any() or not ref_mask.any():
        return None
    metrics_module = surface_distance_metrics()
    surface_distances = metrics_module.compute_surface_distances(
        mask_gt=ref_mask,
        mask_pred=pred_mask,
        spacing_mm=tuple(float(s) for s in spacing_zyx),
    )
    return _finite(
        metrics_module.compute_surface_dice_at_tolerance(surface_distances, tolerance_mm)
    )


def surface_metrics(
    pred: np.ndarray, ref: np.ndarray, spacing_xyz: Sequence[float], tolerance_mm: float
) -> dict:
    """HD95 / ASSD（mm，复用 medpy）与 NSD@τ（surfel-area weighted 物理表面）。

    - GT 非空、预测为空        -> NSD = 0，HD95/ASSD = null（该病例同时计入 missed）
    - GT 与预测均为空          -> 全 null（不纳入阳性病灶表面统计）
    - GT 非空、预测非空但无重叠 -> 正常计算物理表面距离（Dice 为 0，计入 wrong-location）
    """
    spacing = spacing_zyx(spacing_xyz)
    ref_nonempty, pred_nonempty = bool(ref.any()), bool(pred.any())
    if not ref_nonempty:
        return {
            "applicable": False,
            "reason": "reference_empty",
            "hd95_mm": None,
            "assd_mm": None,
            "nsd": None,
        }
    if not pred_nonempty:
        return {
            "applicable": True,
            "reason": "prediction_empty",
            "hd95_mm": None,
            "assd_mm": None,
            "nsd": 0.0,
        }
    from medpy.metric import binary as medpy_binary

    return {
        "applicable": True,
        "reason": "ok",
        "hd95_mm": _finite(medpy_binary.hd95(pred, ref, voxelspacing=spacing)),
        "assd_mm": _finite(medpy_binary.assd(pred, ref, voxelspacing=spacing)),
        "nsd": _finite(nsd_surface_area(pred, ref, spacing, tolerance_mm)),
    }


def load_mask_pair(
    pred_path: str, ref_path: str
) -> tuple[np.ndarray, np.ndarray, tuple, dict]:
    """读取并校验一例 prediction/reference；几何不一致或标签非法时 fail-closed。

    这是评估侧唯一的掩膜IO 入口。评估**绝不**重采样、**绝不**修改掩膜。

    几何判定（2026-10-10 修订，见 :func:`geometry_check` 与协议 §3）：整型 ``size`` 必须严格
    相等；``spacing`` / ``origin`` / ``direction`` 使用绝对容差（``rtol=0``）吸收浮点存储
    舍入；并额外比较网格角点的物理坐标（mm）以拦住"逐元素在容差内、但随图像范围累积超限"的
    真实错位。返回 ``(pred_bool, ref_bool, ref_spacing_xyz, geometry_report)``；第四项是本次
    实际执行的几何检查记录（供结果 JSON 的 provenance 使用，调用方不得据此跳过检查）。
    """
    import SimpleITK as sitk

    if not os.path.isfile(pred_path):
        raise EvaluationError(f"prediction 不存在: {pred_path}")
    if not os.path.isfile(ref_path):
        raise EvaluationError(f"reference 不存在: {ref_path}")
    pred_img, ref_img = sitk.ReadImage(pred_path), sitk.ReadImage(ref_path)
    geometry = geometry_check(image_geometry(pred_img), image_geometry(ref_img))
    if not geometry["ok"]:
        raise EvaluationError(describe_geometry_violations(geometry))
    pred, ref = sitk.GetArrayFromImage(pred_img), sitk.GetArrayFromImage(ref_img)
    for tag, arr in (("prediction", pred), ("reference", ref)):
        if not bool(np.isfinite(arr).all()):
            raise EvaluationError(f"{tag} 含非有限值")
        bad = [float(v) for v in np.unique(arr) if float(v) not in (0.0, 1.0)]
        if bad:
            raise EvaluationError(f"{tag} 含非 0/1 标签值: {bad[:5]}")
    return (
        (pred > 0.5),
        (ref > 0.5),
        tuple(float(s) for s in ref_img.GetSpacing()),
        geometry,
    )


# --------------------------------------------------------------------------- 几何数值一致性


def image_geometry(image) -> dict:
    """从 SimpleITK 图像取出几何元数据（**不读取体素数据**）。

    ``size`` 为整型体素数 (nx, ny, nz)；``spacing`` / ``origin`` / ``direction`` 保持
    SimpleITK 的原始轴序与单位（spacing/origin：mm；direction：9 个方向余弦，行主序）。
    """
    return {
        "size": tuple(int(v) for v in image.GetSize()),
        "spacing": tuple(float(v) for v in image.GetSpacing()),
        "origin": tuple(float(v) for v in image.GetOrigin()),
        "direction": tuple(float(v) for v in image.GetDirection()),
    }


def grid_corner_points(
    size: Sequence[int],
    spacing_xyz: Sequence[float],
    origin: Sequence[float],
    direction: Sequence[float],
) -> np.ndarray:
    """体素**中心**角点的物理坐标（ITK 约定），返回 ``(N, 3)`` float64。

    映射（与 ``SimpleITK.Image.TransformIndexToPhysicalPoint`` 一致）::

        p(index) = origin + Σ_axis spacing[axis] · index[axis] · M[:, axis]

    其中 ``M = np.reshape(direction, (3, 3))`` 是 SimpleITK ``GetDirection()`` 的**行主序**
    展开，``M[:, axis]`` 为该轴的物理方向单位向量。每轴取 ``index ∈ {0, size-1}``；``size == 1``
    的退化轴只取 ``0``（避免同一角点被重复计数）。

    本规则**只检查体素中心角点**（``index ∈ {0, size-1}``），**不**检查外边界角点：外边界角点
    还要额外引入半个体素偏移，而两侧 ``spacing`` / ``direction`` 不同时，中心角点与外边界角点
    给出的偏差并不相同，因此不把两者当作等价判据。本函数只用于"两侧网格是否指向同一物理栅格"
    的数值一致性判断，**不是**解剖配准或对齐质量检查。

    纯函数：不读文件、不 import SimpleITK、不修改入参。
    """
    sizes = [int(v) for v in size]
    spacing = [float(v) for v in spacing_xyz]
    center = [float(v) for v in origin]
    matrix = np.asarray([float(v) for v in direction], dtype=np.float64).reshape(3, 3)
    if len(sizes) != 3 or len(spacing) != 3 or len(center) != 3:
        raise EvaluationError("几何元数据必须是三维（size / spacing / origin 各 3 项）")
    per_axis = [[0, s - 1] if s > 1 else [0] for s in sizes]
    points = []
    for ix in per_axis[0]:
        for iy in per_axis[1]:
            for iz in per_axis[2]:
                index = np.array([ix, iy, iz], dtype=np.float64)
                points.append(np.asarray(center) + matrix @ (np.asarray(spacing) * index))
    return np.asarray(points, dtype=np.float64)


def geometry_check_rule() -> dict:
    """本次几何数值一致性检查规则的**机器可读**描述（写入结果 JSON 的 provenance）。"""
    return {
        "name": "geometry_numeric_consistency",
        "revision": "2026-10-10",
        "integer_field": {"size": "strict equality; no tolerance"},
        "float_fields": {
            "spacing": {"abs_tolerance": GEOMETRY_ATOL_MM, "rtol": 0, "unit": "mm"},
            "origin": {"abs_tolerance": GEOMETRY_ATOL_MM, "rtol": 0, "unit": "mm"},
            "direction": {
                "abs_tolerance": GEOMETRY_ATOL_DIRECTION,
                "rtol": 0,
                "unit": "dimensionless",
            },
        },
        "physical_corner_check": {
            "points": "voxel-centre corner points; index in {0, size-1} per axis",
            "metric": "max euclidean physical distance",
            "abs_tolerance_mm": GEOMETRY_CORNER_ATOL_MM,
            "accept_if": "max_delta_mm <= abs_tolerance_mm",
        },
        "fail_closed": True,
        "resampling": "none",
        "interpretation": (
            "numeric consistency of float metadata only; not a registration/alignment "
            "quality check, and not a clinical tolerance"
        ),
    }


def geometry_check(pred_geometry: dict, ref_geometry: dict) -> dict:
    """比较两组几何元数据的数值一致性（纯函数，不读文件）。

    返回::

        {
          "rule": <geometry_check_rule()>,
          "ok": bool,                      # False 即调用方必须 fail-closed
          "exact_equal": bool,             # 四个字段逐值完全相等
          "max_abs_diff": {"spacing_mm": float|None, "origin_mm": float|None,
                           "direction": float|None},
          "nonzero_fields": [...],         # 在容差内但非零的字段（供 provenance 记录）
          "physical_corner": {"checked": bool, "n_points": int,
                              "max_delta_mm": float|None,
                              "abs_tolerance_mm": GEOMETRY_CORNER_ATOL_MM},
          "violations": [ {...} ],         # 结构化失败条目（含字段/最大差/容差/单位）
        }

    判定顺序：``size`` 严格相等 → 浮点字段有限性与 spacing 正值 → 逐字段绝对容差
    → （仅在两侧元数据可比且 size 相等时）角点物理偏差。任一环节不通过即
    ``ok = False``；**不**因几何问题修改任何输入。
    """
    violations: list[dict] = []
    max_abs: dict[str, float | None] = {
        "spacing_mm": None,
        "origin_mm": None,
        "direction": None,
    }
    pred_size = tuple(int(v) for v in pred_geometry["size"])
    ref_size = tuple(int(v) for v in ref_geometry["size"])
    size_equal = pred_size == ref_size
    if not size_equal:
        violations.append(
            {
                "kind": "size_mismatch",
                "field": "size",
                "detail": f"{pred_size} vs {ref_size}",
                "abs_tolerance": None,
                "unit": "voxels",
                "note": "整型体素数必须严格相等；不允许尺寸容差",
            }
        )

    fields = (
        ("spacing", "spacing_mm", GEOMETRY_ATOL_MM, "mm"),
        ("origin", "origin_mm", GEOMETRY_ATOL_MM, "mm"),
        ("direction", "direction", GEOMETRY_ATOL_DIRECTION, "dimensionless"),
    )
    values: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    comparable = True
    for key, metric_key, tolerance, unit in fields:
        pred_values = np.asarray([float(v) for v in pred_geometry[key]], dtype=np.float64)
        ref_values = np.asarray([float(v) for v in ref_geometry[key]], dtype=np.float64)
        if pred_values.shape != ref_values.shape:
            comparable = False
            violations.append(
                {
                    "kind": "malformed_metadata",
                    "field": key,
                    "detail": f"长度 {pred_values.size} vs {ref_values.size}",
                    "abs_tolerance": tolerance,
                    "unit": unit,
                }
            )
            continue
        if not bool(np.isfinite(pred_values).all()) or not bool(np.isfinite(ref_values).all()):
            comparable = False
            violations.append(
                {
                    "kind": "non_finite_metadata",
                    "field": key,
                    "detail": f"prediction={pred_values.tolist()} reference={ref_values.tolist()}",
                    "abs_tolerance": tolerance,
                    "unit": unit,
                }
            )
            continue
        if key == "spacing" and (bool((pred_values <= 0).any()) or bool((ref_values <= 0).any())):
            comparable = False
            violations.append(
                {
                    "kind": "non_positive_spacing",
                    "field": key,
                    "detail": f"prediction={pred_values.tolist()} reference={ref_values.tolist()}",
                    "abs_tolerance": tolerance,
                    "unit": unit,
                    "note": "spacing 必须为正；非正值无法构成物理网格",
                }
            )
            continue
        values[key] = (pred_values, ref_values)
        delta = float(np.max(np.abs(pred_values - ref_values)))
        max_abs[metric_key] = delta
        if not delta <= tolerance:
            violations.append(
                {
                    "kind": f"{key}_tolerance_exceeded",
                    "field": key,
                    "detail": f"max|Δ| = {delta:.6g} {unit}",
                    "max_abs_diff": delta,
                    "abs_tolerance": tolerance,
                    "unit": unit,
                }
            )

    corner = {
        "checked": False,
        "n_points": 0,
        "max_delta_mm": None,
        "abs_tolerance_mm": GEOMETRY_CORNER_ATOL_MM,
    }
    if size_equal and comparable and len(values) == len(fields):
        pred_points = grid_corner_points(
            pred_size, values["spacing"][0], values["origin"][0], values["direction"][0]
        )
        ref_points = grid_corner_points(
            ref_size, values["spacing"][1], values["origin"][1], values["direction"][1]
        )
        deltas = np.linalg.norm(pred_points - ref_points, axis=1)
        corner.update(
            checked=True, n_points=int(deltas.size), max_delta_mm=float(deltas.max())
        )
        if not corner["max_delta_mm"] <= GEOMETRY_CORNER_ATOL_MM:
            violations.append(
                {
                    "kind": "physical_corner_tolerance_exceeded",
                    "field": "physical_corner",
                    "detail": f"max Δ = {corner['max_delta_mm']:.6g} mm",
                    "max_abs_diff": corner["max_delta_mm"],
                    "abs_tolerance": GEOMETRY_CORNER_ATOL_MM,
                    "unit": "mm",
                    "note": (
                        "逐元素差异在容差内但随图像范围累积超出角点阈值；"
                        "拒绝该 prediction/reference 组合"
                    ),
                }
            )

    exact_equal = (
        size_equal
        and len(values) == len(fields)
        and all(bool(np.array_equal(values[key][0], values[key][1])) for key in values)
    )
    nonzero_fields = [
        key for key, metric_key, _tolerance, _unit in fields
        if max_abs[metric_key] not in (None, 0.0)
    ]
    return {
        "rule": geometry_check_rule(),
        "ok": not violations,
        "exact_equal": exact_equal,
        "max_abs_diff": max_abs,
        "nonzero_fields": nonzero_fields,
        "physical_corner": corner,
        "violations": violations,
    }


def describe_geometry_violations(geometry: dict) -> str:
    """把 :func:`geometry_check` 的结构化失败条目渲染成单行错误文本。

    每条都写出字段名、最大绝对差与其**单位**、以及本次采用的容差；角点条目额外给出 mm 单位。
    """
    lines = ["几何数值不一致（prediction vs reference）："]
    for item in geometry["violations"]:
        field = item["field"]
        unit = item.get("unit")
        tolerance = item.get("abs_tolerance")
        if item["kind"] == "size_mismatch":
            lines.append(
                f"  - size: {item['detail']}（整型体素数要求严格相等，无容差；单位 voxel）"
            )
            continue
        if item["kind"] in ("non_finite_metadata", "non_positive_spacing"):
            lines.append(f"  - {field}: {item['detail']}（{item['kind']}）")
            continue
        if item["kind"] == "malformed_metadata":
            lines.append(f"  - {field}: {item['detail']}（无法逐值比较）")
            continue
        if item["kind"] == "physical_corner_tolerance_exceeded":
            lines.append(
                f"  - 体素中心角点物理偏差: {item['detail']}"
                f"，超过容差 {tolerance:g} mm（单位 mm）"
            )
            continue
        lines.append(
            f"  - {field}: {item['detail']}，超过绝对容差 {tolerance:g}"
            f"（单位 {unit}；rtol=0）"
        )
    lines.append(
        "评估不重采样、不修改掩膜；真实几何不一致必须先行修复，不接受部分结果"
    )
    return "\n".join(lines)


def component_analysis(
    pred: np.ndarray, ref: np.ndarray, voxel_volume_mm3_value: float, case_id: str
) -> dict:
    """连通域失败分析（仅分割失败分析；**不**计算 AUROC / AP / FROC / detection score）。"""
    from scipy import ndimage

    structure = ndimage.generate_binary_structure(3, CONNECTIVITY)
    pred_lab, n_pred_comp = ndimage.label(pred, structure=structure)
    ref_lab, n_ref_comp = ndimage.label(ref, structure=structure)
    pred_sizes = np.bincount(pred_lab.ravel())[1:] if n_pred_comp else np.array([])
    ref_sizes = np.bincount(ref_lab.ravel())[1:] if n_ref_comp else np.array([])
    ref_uncovered = sum(
        1
        for comp_id in range(1, n_ref_comp + 1)
        if not np.any(pred[ref_lab == comp_id])
    )
    return {
        "case_id": case_id,
        "connectivity": CONNECTIVITY,
        "connectivity_note": "scipy.ndimage 结构元素阶数；1 = 6-邻域（面相邻）",
        "pred_component_count": int(n_pred_comp),
        "pred_component_max_volume_mm3": _finite(
            float(pred_sizes.max()) * voxel_volume_mm3_value if pred_sizes.size else 0.0
        ),
        "ref_component_count": int(n_ref_comp),
        "ref_component_total_volume_mm3": _finite(
            float(ref_sizes.sum()) * voxel_volume_mm3_value
        ),
        "ref_components_without_prediction_overlap": int(ref_uncovered),
        "ref_components_without_prediction_overlap_rate": _ratio(
            ref_uncovered, n_ref_comp
        ),
    }


def recompute_case_counts(pred: np.ndarray, ref: np.ndarray) -> dict:
    """由掩膜重新计算 TP/FP/FN/TN/n_pred/n_ref（整数计数，**不信任** summary.json）。"""
    return {
        "TP": int(np.count_nonzero(pred & ref)),
        "FP": int(np.count_nonzero(pred & ~ref)),
        "FN": int(np.count_nonzero(~pred & ref)),
        "TN": int(np.count_nonzero(~pred & ~ref)),
        "n_pred": int(np.count_nonzero(pred)),
        "n_ref": int(np.count_nonzero(ref)),
    }


def verify_case_counts(case: dict, recomputed: dict) -> str | None:
    """把掩膜重算的六项计数与 summary.json 逐项核对。

    返回差异描述（**不含** model / case 前缀，前缀由调用方统一添加）或 ``None``。
    """
    diffs = [
        f"{key}(mask={recomputed[key]}, summary={case[key]})"
        for key in ("TP", "FP", "FN", "TN", "n_pred", "n_ref")
        if recomputed[key] != case[key]
    ]
    if not diffs:
        return None
    return "掩膜与 summary.json 计数不一致 -> " + ", ".join(diffs)


def pred_component_stats(pred: np.ndarray, voxel_volume_mm3_value: float) -> dict:
    """预测掩膜的连通域统计（用于阴性病例的假阳团块；不评价检测性能）。"""
    from scipy import ndimage

    structure = ndimage.generate_binary_structure(3, CONNECTIVITY)
    lab, n_comp = ndimage.label(pred, structure=structure)
    sizes = np.bincount(lab.ravel())[1:] if n_comp else np.array([])
    return {
        "connectivity": CONNECTIVITY,
        "component_count": int(n_comp),
        "component_max_volume_mm3": _finite(
            float(sizes.max()) * voxel_volume_mm3_value if sizes.size else 0.0
        ),
    }


def dice_from_counts(tp: int, fp: int, fn: int, n_ref: int) -> float | None:
    """由 TP/FP/FN 重算 Dice。

    - ``n_ref == 0``（阴性病例，含真阴与假阳）-> ``0.0``（有预测）或 ``None``（空-空真阴）
    - ``n_ref > 0`` 且无重叠 -> ``0.0``（完全漏分不得排除，也不得记为无定义）
    """
    denom = 2 * tp + fp + fn
    if denom == 0:
        return None
    return 2 * tp / denom


# --------------------------------------------------------------------------- 历史私有别名
_voxel_volume_mm3 = voxel_volume_mm3
_spacing_zyx = spacing_zyx
_surface_distance_metrics = surface_distance_metrics
_verify_case_counts = verify_case_counts
_pred_component_stats = pred_component_stats
_recompute_dice = dice_from_counts
