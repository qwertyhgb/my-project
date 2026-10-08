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
- 评估**不重采样掩膜**：geometry 不一致时直接 fail-closed。
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np

from zonal_reliability_fusion.evaluation.protocol import (
    CONNECTIVITY,
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


def load_mask_pair(pred_path: str, ref_path: str) -> tuple[np.ndarray, np.ndarray, tuple]:
    """读取并校验一例 prediction/reference；几何不一致或标签非法时 fail-closed。

    这是评估侧唯一的掩膜IO 入口。评估**绝不**重采样、**绝不**修改掩膜：size / spacing /
    origin / direction 任一不一致即失败。
    """
    import SimpleITK as sitk

    if not os.path.isfile(pred_path):
        raise EvaluationError(f"prediction 不存在: {pred_path}")
    if not os.path.isfile(ref_path):
        raise EvaluationError(f"reference 不存在: {ref_path}")
    pred_img, ref_img = sitk.ReadImage(pred_path), sitk.ReadImage(ref_path)
    for tag, a, b in (
        ("size", pred_img.GetSize(), ref_img.GetSize()),
        ("spacing", pred_img.GetSpacing(), ref_img.GetSpacing()),
        ("origin", pred_img.GetOrigin(), ref_img.GetOrigin()),
        ("direction", pred_img.GetDirection(), ref_img.GetDirection()),
    ):
        if tuple(a) != tuple(b):
            raise EvaluationError(
                f"{tag} 不一致（prediction vs reference）：{tuple(a)} vs {tuple(b)}；"
                "评估不重采样，几何不一致必须先行修复"
            )
    pred, ref = sitk.GetArrayFromImage(pred_img), sitk.GetArrayFromImage(ref_img)
    for tag, arr in (("prediction", pred), ("reference", ref)):
        if not bool(np.isfinite(arr).all()):
            raise EvaluationError(f"{tag} 含非有限值")
        bad = [float(v) for v in np.unique(arr) if float(v) not in (0.0, 1.0)]
        if bad:
            raise EvaluationError(f"{tag} 含非 0/1 标签值: {bad[:5]}")
    return (pred > 0.5), (ref > 0.5), tuple(float(s) for s in ref_img.GetSpacing())


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
