"""解剖区域层面的 FP 负担与病灶定位分解（新主线的第三个失败模式）。

为什么需要这个模块
------------------
新主线的第三个核心失败模式是「提高 sensitivity 后引入过多 false positives」。仅报告
``negative cases with FP`` 与 ``FP components / case`` 无法回答一个关键问题：**多出来的假阳
落在哪里？** 一个落在前列腺腺体外的假阳与一个落在 PZ 内的假阳，临床意义完全不同。

因此本模块把 FP 按**预测解剖先验**（predicted WG / PZ / TZ soft probability）分解，并给出
每个 reference lesion 的解剖定位。

先验来源的硬约束（与 Research Plan / train_nnunet 的口径一致）
------------------------------------------------------------
- 默认输入必须是**预测得到的 soft probability**（来自冻结 anatomy prior generator 对该病例
  自身 MRI 的预测），**不是** GT 标签。
- 传入 GT WG/PZ/TZ 时，调用方**必须**显式传 ``prior_source="ORACLE_GT"``；本模块会把该标记
  原样写进输出，使 oracle 分析**不可能**被误当成正式最终性能。
- 解剖先验错误是真实部署链条的一部分：不做任何「先验修正」或「按先验筛掉预测」的动作。

分区归属约定（避免歧义）
------------------------
给定 soft 概率 ``P(WG) / P(PZ) / P(TZ)``（各自 [0,1]，形状与掩膜一致）：

- ``inside_wg``：``P(WG) >= wg_threshold``；
- ``outside_wg``：``P(WG) < wg_threshold``；
- 在 ``inside_wg`` 内，区域归属由 ``P(PZ)`` 与 ``P(TZ)`` 的 **argmax** 决定（argmax 唯一，
  保证每个体素恰归一 zone，避免 PZ+TZ 双计）；两者都低于 ``zone_threshold`` 时记为
  ``uncertain``（WG 内部但分区证据不足，典型如 zone 边界与 WG 边界附近）。
- 阈值在调用时显式传入，本模块**不设**未经论证的默认值。
"""

from __future__ import annotations

import numpy as np

from zonal_reliability_fusion.evaluation.protocol import (
    CONNECTIVITY,
    EvaluationError,
    _dist_stats,
    _finite,
    _json_safe,
    _ratio,
)

#: 区域分解的固定报告顺序；不得依赖 dict 迭代顺序
ZONE_KEYS = ("PZ", "TZ", "uncertain")
#: 预测先验来源标记。正式结果只允许 ``predicted_prior``；``ORACLE_GT`` 仅为上界分析。
PRIOR_SOURCE_PREDICTED = "predicted_prior"
PRIOR_SOURCE_ORACLE_GT = "ORACLE_GT"
ALLOWED_PRIOR_SOURCES = (PRIOR_SOURCE_PREDICTED, PRIOR_SOURCE_ORACLE_GT)

#: 区域分解的定义文本；随 JSON 发布
ANATOMY_DEFINITIONS: dict = {
    "inside_wg": "P(WG) >= wg_threshold（预测腺体内）",
    "outside_wg": "P(WG) < wg_threshold（预测腺体外）",
    "zone_attribution": (
        "仅在 inside_wg 内按 argmax(P(PZ), P(TZ)) 唯一归属；两者都 < zone_threshold 时记为 "
        "uncertain；每个体素恰好归一 zone，PZ 与 TZ 不会双计"
    ),
    "prior_source": (
        "predicted_prior = 冻结 anatomy 模型对该病例自身 MRI 的预测（正式结果）；"
        "ORACLE_GT = 使用 GT 解剖标签的上界分析，不得作为正式最终性能"
    ),
    "anatomy_prediction_error": (
        "本分解不做任何先验修正或按先验筛除预测；anatomy 先验错误被视为真实部署链条的一部分"
    ),
    "not_a_clinical_stratification": "区域分解是 exploratory 分析，不是临床风险分层",
}


def validate_anatomy_priors(
    wg: np.ndarray, pz: np.ndarray, tz: np.ndarray, reference_shape: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """校验三个先验概率图：形状一致、有限、值域 [0,1]。

    ``reference_shape`` 是病灶掩膜的形状（Z, Y, X）。三者必须**逐值**同形——评估不重采样，
    几何不一致直接 fail-closed。
    """
    out = []
    for name, arr in (("WG", wg), ("PZ", pz), ("TZ", tz)):
        a = np.asarray(arr, dtype=np.float64)
        if a.shape != tuple(reference_shape):
            raise EvaluationError(
                f"预测先验 {name} 形状 {a.shape} 与掩膜形状 {tuple(reference_shape)} 不一致；"
                "评估不重采样"
            )
        if not bool(np.isfinite(a).all()):
            raise EvaluationError(f"预测先验 {name} 含非有限值")
        if bool((a < 0.0).any()) or bool((a > 1.0).any()):
            raise EvaluationError(f"预测先验 {name} 超出 [0, 1]（soft probability 要求）")
        out.append(a)
    return out[0], out[1], out[2]


def zone_assignment(pz: np.ndarray, tz: np.ndarray, zone_threshold: float) -> np.ndarray:
    """在给定区域内按 ``argmax(P(PZ), P(TZ))`` 给出唯一区域索引（0=PZ, 1=TZ, 2=uncertain）。"""
    if not 0.0 <= zone_threshold <= 1.0:
        raise EvaluationError(f"zone_threshold 必须落在 [0,1]，收到 {zone_threshold}")
    index = np.full(pz.shape, 2, dtype=np.int8)
    decidable = (pz >= zone_threshold) | (tz >= zone_threshold)
    index[decidable & (pz >= tz)] = 0
    index[decidable & (tz > pz)] = 1
    return index


def false_positive_burden(
    pred: np.ndarray,
    ref: np.ndarray,
    wg: np.ndarray,
    pz: np.ndarray,
    tz: np.ndarray,
    voxel_volume_mm3: float,
    *,
    wg_threshold: float,
    zone_threshold: float,
    prior_source: str = PRIOR_SOURCE_PREDICTED,
) -> dict:
    """按预测解剖先验分解 FP 负担（新主线第三失败模式的直接指标）。

    只统计 ``FP = pred & ~ref`` 的体素与连通域；TP/FP/FN 计数本身由
    :func:`..evaluation.case_metrics.recompute_case_counts` 负责，本函数不重复定义它们。

    ``prior_source`` 必须是 ``predicted_prior`` 或 ``ORACLE_GT``；后者原样写入输出，使上界
    分析无法被误当作正式结果。
    """
    if prior_source not in ALLOWED_PRIOR_SOURCES:
        raise EvaluationError(
            f"prior_source 必须是 {ALLOWED_PRIOR_SOURCES} 之一，收到 {prior_source!r}"
        )
    if not 0.0 <= wg_threshold <= 1.0:
        raise EvaluationError(f"wg_threshold 必须落在 [0,1]，收到 {wg_threshold}")
    pred = np.asarray(pred, dtype=bool)
    ref = np.asarray(ref, dtype=bool)
    wg, pz, tz = validate_anatomy_priors(wg, pz, tz, pred.shape)

    fp = pred & ~ref
    inside = fp & (wg >= wg_threshold)
    fp_counts: dict[str, int] = {
        "inside_wg": int(inside.sum()),
        "outside_wg": int(fp.sum()) - int(inside.sum()),
    }
    zone_index = zone_assignment(pz, tz, zone_threshold)
    for key, code in zip(ZONE_KEYS, (0, 1, 2), strict=True):
        fp_counts[key] = int((inside & (zone_index == code)).sum())
    unassigned = int(inside.sum()) - sum(fp_counts[key] for key in ZONE_KEYS)
    if unassigned:  # pragma: no cover - zone_assignment 覆盖全部体素，恒为 0
        raise EvaluationError(f"区域分解不完整，{unassigned} 个 inside_wg 体素未归类")

    return _json_safe(
        {
            "prior_source": prior_source,
            "thresholds": {
                "wg_threshold": float(wg_threshold),
                "zone_threshold": float(zone_threshold),
            },
            "voxel_volume_mm3": _finite(voxel_volume_mm3),
            "fp_voxels_total": int(fp.sum()),
            "fp_voxels": fp_counts,
            "fp_volume_mm3": {
                key: float(count) * float(voxel_volume_mm3)
                for key, count in fp_counts.items()
            },
            "fp_component_count": fp_component_counts_by_region(
                fp, wg, pz, tz, wg_threshold=wg_threshold, zone_threshold=zone_threshold
            ),
            "definitions": ANATOMY_DEFINITIONS,
        }
    )


def fp_component_counts_by_region(
    fp: np.ndarray,
    wg: np.ndarray,
    pz: np.ndarray,
    tz: np.ndarray,
    *,
    wg_threshold: float,
    zone_threshold: float,
) -> dict:
    """把 FP 连通域按其**质心所在区域**归类（体素口径与团块口径互补，不互相替代）。

    体素口径回答「多少 FP 体积落在腺体内」，团块口径回答「有多少个假阳团块落在腺体内」。
    两者都必须报告：一个巨大的腺体内 FP 团块在体素口径里很显眼，在团块口径里只有 1 个；
    反之亦然。
    """
    from scipy import ndimage

    if not 0.0 <= wg_threshold <= 1.0:
        raise EvaluationError(f"wg_threshold 必须落在 [0,1]，收到 {wg_threshold}")
    fp = np.asarray(fp, dtype=bool)
    wg, pz, tz = validate_anatomy_priors(wg, pz, tz, fp.shape)
    structure = ndimage.generate_binary_structure(3, CONNECTIVITY)
    labels, n_components = ndimage.label(fp, structure=structure)
    counts = {"inside_wg": 0, "outside_wg": 0}
    zone_counts = dict.fromkeys(ZONE_KEYS, 0)
    zone_index = zone_assignment(pz, tz, zone_threshold)
    for comp_id in range(1, int(n_components) + 1):
        component = labels == comp_id
        # 质心用 component 自身的体素坐标，避免被腺体外的其它 FP 拉偏
        coords = np.nonzero(component)
        centroid = tuple(int(round(float(c.mean()))) for c in coords)
        if wg[centroid] >= wg_threshold:
            counts["inside_wg"] += 1
            zone_counts[ZONE_KEYS[int(zone_index[centroid])]] += 1
        else:
            counts["outside_wg"] += 1
    return {
        "total": int(n_components),
        "by_gland": counts,
        "by_zone_inside_gland": zone_counts,
    }


def lesion_anatomy_localization(
    pred: np.ndarray,
    ref: np.ndarray,
    wg: np.ndarray,
    pz: np.ndarray,
    tz: np.ndarray,
    voxel_volume_mm3: float,
    *,
    wg_threshold: float,
    zone_threshold: float,
    prior_source: str = PRIOR_SOURCE_PREDICTED,
) -> dict:
    """按预测解剖先验给出 reference lesion 的定位分布，以及 lesion 级的漏检定位。

    这是「小病灶漏检是否集中在某个解剖位置」这一问题的直接证据来源：若 small-lesion 的
    漏检率在 PZ 明显高于 TZ，则 PZ/TZ 条件化（条件 D）才有明确的机制动机。

    漏检判定复用 :func:`..evaluation.lesion_metrics.lesion_instance_metrics` 的同一匹配协议
    （6-邻域、intersection >= 1、字典序最大基数匹配），因此这里的「missed」与主终点
    ``lesion_sensitivity_any_overlap`` **完全一致**，不会出现两套漏检定义。
    """
    from scipy import ndimage

    from zonal_reliability_fusion.evaluation.lesion_metrics import lesion_instance_metrics

    if prior_source not in ALLOWED_PRIOR_SOURCES:
        raise EvaluationError(
            f"prior_source 必须是 {ALLOWED_PRIOR_SOURCES} 之一，收到 {prior_source!r}"
        )
    if not 0.0 <= wg_threshold <= 1.0:
        raise EvaluationError(f"wg_threshold 必须落在 [0,1]，收到 {wg_threshold}")
    pred = np.asarray(pred, dtype=bool)
    ref = np.asarray(ref, dtype=bool)
    wg, pz, tz = validate_anatomy_priors(wg, pz, tz, pred.shape)

    structure = ndimage.generate_binary_structure(3, CONNECTIVITY)
    ref_lab, n_ref = ndimage.label(ref, structure=structure)
    zone_index = zone_assignment(pz, tz, zone_threshold)

    records: list[dict] = []
    for comp_id in range(1, int(n_ref) + 1):
        component = ref_lab == comp_id
        coords = np.nonzero(component)
        centroid = tuple(int(round(float(c.mean()))) for c in coords)
        if wg[centroid] < wg_threshold:
            region = "outside_wg"
        else:
            region = ZONE_KEYS[int(zone_index[centroid])]
        records.append(
            {
                "reference_component_id": comp_id,
                "voxels": int(component.sum()),
                "volume_mm3": _finite(float(component.sum()) * float(voxel_volume_mm3)),
                "anatomy_region": region,
            }
        )

    instance = lesion_instance_metrics(pred, ref, voxel_volume_mm3, "")
    dices_by_ref = {
        record["reference_component_id"]: record["matched_dice"]
        for record in instance["reference_lesion_records"]
    }
    by_region: dict[str, dict] = {}
    for record in records:
        bucket = by_region.setdefault(
            record["anatomy_region"],
            {
                "reference_lesion_count": 0,
                "matched_reference_lesion_count": 0,
                "dices": [],
            },
        )
        bucket["reference_lesion_count"] += 1
        dice = dices_by_ref.get(record["reference_component_id"])
        if dice is not None:
            bucket["matched_reference_lesion_count"] += 1
            bucket["dices"].append(float(dice))
    for bucket in by_region.values():
        bucket["lesion_sensitivity_any_overlap"] = _ratio(
            bucket["matched_reference_lesion_count"], bucket["reference_lesion_count"]
        )
        bucket["matched_lesion_dice"] = _dist_stats(bucket.pop("dices"))

    return _json_safe(
        {
            "prior_source": prior_source,
            "thresholds": {
                "wg_threshold": float(wg_threshold),
                "zone_threshold": float(zone_threshold),
            },
            "reference_lesions": int(n_ref),
            "by_anatomy_region": by_region,
            "reference_lesion_records": records,
            "definitions": ANATOMY_DEFINITIONS,
        }
    )
