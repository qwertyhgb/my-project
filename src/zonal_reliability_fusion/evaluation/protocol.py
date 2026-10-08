"""评价协议的**冻结常量**与通用工具（新主线唯一的指标定义来源）。

本模块只包含**纯定义**：协议常量、异常类型、数值/JSON 工具与bootstrap 置信区间。
它**不**读取任何文件、不做I/O、不 import nnU-Net，也不 import torch，因此可以被训练、
推理、评估与测试四方共同引用，而不会引入依赖环。

协议冻结说明
------------
以下常量在看到任何结果之前即已固定（见 ``docs/Evaluation_Protocol.md``），改动其中任何一个
都必须视为**协议变更**，需要在``Development_Log.md`` 中显式记录并说明是否需要重算历史结果：

- ``CONNECTIVITY``：病灶实例 = 3D 6-邻域连通域（面相邻）。不改成 18/26 邻域。
- 病灶大小分层边界：探索性分层（**不是**临床风险类别），< 500 / 500–1000 / > 1000 mm³。
- ``DICE_MATCH_TOLERANCE``：summary.json 的 Dice 与由 TP/FP/FN 重算值的最大允许偏差。
- 空值语义：分母为 0 时返回 ``None``（JSON ``null``），**不得**伪造为 0。

术语边界：``positive_voxel_precision`` 与 ``all_prediction_voxel_precision`` 分母不同，
不得都笼统称为 "precision"。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

#: 输出 schema 版本；改动指标集合时必须递增
SCHEMA_VERSION = "1.1"
DEFAULT_BOOTSTRAP_RESAMPLES = 10000
DEFAULT_SEED = 20260922
#: 判定「病例间 Dice 变化是否为平局」的容差（仅用于 improved/tied/worsened 计数）
TIE_TOLERANCE = 1e-12
#: 连通域分析的 connectivity（scipy.ndimage 结构元素阶数；1 = 6-邻域）
CONNECTIVITY = 1
#: 病灶实例级大小分层（探索性，预先冻结；单位 mm³；边界严格：<500 / 500–1000 / >1000）
LESION_SIZE_SMALL_MAX_MM3 = 500.0
LESION_SIZE_MEDIUM_MAX_MM3 = 1000.0
#: 分层键的顺序即报告顺序（small -> medium -> large），不得依赖 dict 迭代顺序
LESION_SIZE_STRATUM_KEYS = (
    "small_lt_500_mm3",
    "medium_500_to_1000_mm3",
    "large_gt_1000_mm3",
)
#: summary 中的 Dice 与由 TP/FP/FN 重算值的最大允许偏差（两者应逐位一致）
DICE_MATCH_TOLERANCE = 1e-6
#: 错误明细的显示上限；超过时汇总文本会明确写出「仅显示前 N 条，另有 M 条」
ERROR_DISPLAY_CAP = 50

#: 评价终点分级（新主线论文报告口径；详见 docs/Evaluation_Protocol.md）。
#: ``primary`` 只有一项；``key_secondary`` 是必须同时报告的；``exploratory`` 只作分层观察。
ENDPOINT_TIERS: dict[str, tuple[str, ...]] = {
    "primary": ("positive_case_macro_dice",),
    "key_secondary": (
        "lesion_sensitivity_any_overlap",
        "small_lesion_sensitivity_any_overlap",
        "matched_lesion_dice",
        "completely_missed_lesion_rate",
        "completely_missed_case_rate",
        "positive_voxel_recall",
        "positive_voxel_precision",
        "false_positive_burden",
    ),
    "exploratory": (
        "lesion_size_strata_sensitivity",
        "anatomy_region_fp_burden",
        "surface_metrics",
        "bootstrap_ci",
    ),
}


class EvaluationError(RuntimeError):
    """评估过程中的可预期失败（fail-closed，不发布输出）。"""


def finite(value: Any) -> float | None:
    """把 NaN / ±Inf / 不可转换值统一变为 ``None``（JSON 只允许 null）。

    模块内同时保留历史私有别名 ``_finite``，避免调用方（含既有测试）改名后失效。
    """
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def json_safe(obj: Any) -> Any:
    """递归把 numpy 标量/数组与 NaN/Inf 转成标准 JSON 可表示的值。"""
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return finite(float(obj))
    return obj


def ratio(numerator: float, denominator: float) -> float | None:
    """分母为 0 时返回 ``None``（不是 0）。这是协议级的空值语义。"""
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def dist_stats(values: Sequence[float]) -> dict:
    """分布统计；空输入返回 ``count=0`` 与全null（不写 NaN）。"""
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "q1": None,
            "q3": None,
            "min": None,
            "max": None,
        }
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "q1": float(np.percentile(arr, 25)),
        "q3": float(np.percentile(arr, 75)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def percentile_ci(samples: np.ndarray, alpha: float = 0.05) -> tuple[float, float] | None:
    if samples.size == 0:
        return None
    lo = float(np.percentile(samples, 100.0 * alpha / 2.0))
    hi = float(np.percentile(samples, 100.0 * (1.0 - alpha / 2.0)))
    return (lo, hi)


def bootstrap_ci_mean(
    values: Sequence[float], n_resamples: int, seed: int
) -> tuple[float, float] | None:
    """病例级重采样的均值 95% CI（单模型指标的不确定性）。"""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(int(n_resamples), arr.size))
    return percentile_ci(arr[idx].mean(axis=1))


def bootstrap_ci_paired_mean(
    deltas: Sequence[float], n_resamples: int, seed: int
) -> tuple[float, float] | None:
    """**配对** bootstrap：同一批病例索引同时重采样两个模型，禁止分别重采样。"""
    arr = np.asarray(list(deltas), dtype=float)
    if arr.size == 0:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(int(n_resamples), arr.size))
    return percentile_ci(arr[idx].mean(axis=1))


def bootstrap_ci_paired_ratio(
    numerator: Sequence[float],
    denominator: Sequence[float],
    n_resamples: int,
    seed: int,
) -> tuple[float, float] | None:
    """配对 bootstrap 的比值型指标 CI（micro Dice / recall / precision 等）。

    同一批病例索引同时重采样分子与分母；某次重采样分母为 0 时丢弃该次样本。
    """
    num = np.asarray(list(numerator), dtype=float)
    den = np.asarray(list(denominator), dtype=float)
    if num.size == 0 or num.size != den.size:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, num.size, size=(int(n_resamples), num.size))
    sn, sd = num[idx].sum(axis=1), den[idx].sum(axis=1)
    valid = sd != 0
    if not bool(valid.any()):
        return None
    return percentile_ci(sn[valid] / sd[valid])


# --------------------------------------------------------------------------- 历史私有别名
# 既有测试与脚本历史以 ``_finite`` / ``_json_safe`` / ``_ratio`` / ``_dist_stats`` /
# ``_percentile_ci`` 引用这些纯函数。保留别名，使抽取到本包**不改变**任何既有调用点，
# 也不需要重写已经记录在Training_Log / Development_Log 中的历史事实。
_finite = finite
_json_safe = json_safe
_ratio = ratio
_dist_stats = dist_stats
_percentile_ci = percentile_ci
