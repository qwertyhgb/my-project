"""病灶实例级（lesion-level）指标：新主线评价体系的核心资产。

这一模块回答的是本项目三个核心失败模式中的前两个：

1. **完全漏检 lesion** -> :func:`lesion_instance_metrics` 的 ``lesion_sensitivity_any_overlap``
   与 ``small_lesion_sensitivity_any_overlap``（未匹配 lesion **不从分母删除**）；
2. **已检出病灶覆盖不足** -> ``matched_lesion_dice``（只对matched pair 计算，**必须**与
   lesion sensitivity 同时报告，否则单独报告会隐藏完全漏检）。

第三个失败模式（sensitivity 提升带来过多假阳）由 :mod:`..evaluation.anatomy_metrics`
的 FP burden 分解与 ``unmatched_predicted_lesions`` 共同承担。

协议冻结（改动任一项都视为协议变更）
------------------------------------
- 实例 = 3D 连通域，``CONNECTIVITY = 1``（6-邻域 / face connectivity）。仅角或边接触的体素
  不属于同一 lesion；不改成 18/26 邻域。
- 候选匹配 = intersection >= 1 voxel。
- 一对一 matching 的**字典序**目标：先最大化匹配数（maximum cardinality），再最大化总
  intersection 体素数，最后按 (reference id, prediction id) 升序确定性 tie-break。
  **不**以 Dice/IoU 为 matching 目标（避免 metric-optimizing-the-metric）。
- 大小分层为**探索性分层**（<500 / 500–1000 / >1000 mm³），**不是**临床大小分类或风险类别。
- 阴性病例（``n_ref == 0``）reference_lesions = 0：不进 sensitivity 分母，但其 prediction
  components 全部计入 ``unmatched_predicted_lesions``（假阳负担）。
- 不使用任何预测后处理：最小团块过滤 / 最大团块 / 形态学开闭 / 填洞 / 阈值优化一律禁止。
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from zonal_reliability_fusion.evaluation.protocol import (
    CONNECTIVITY,
    LESION_SIZE_MEDIUM_MAX_MM3,
    LESION_SIZE_SMALL_MAX_MM3,
    LESION_SIZE_STRATUM_KEYS,
    EvaluationError,
    _dist_stats,
    _finite,
    _json_safe,
    _ratio,
)

#: 实例级指标的冻结定义文本；随 JSON 一起发布，使报告自带口径而无需回查文档
MATCHING_DEFINITION: dict = {
    "candidate_rule": "intersection_voxels >= 1",
    "primary_objective": "maximum_cardinality",
    "secondary_objective": "maximum_total_intersection_voxels",
    "tie_break": (
        "reference_component_id_ascending_then_prediction_component_id_ascending"
    ),
    "objective_note": (
        "matching 不以 Dice / IoU 为目标：matched-lesion Dice 是报告终点之一，"
        "先最大化 Dice 会产生 metric-optimizing-the-metric 风险；"
        "Dice 只用于匹配后的质量评价"
    ),
}

SIZE_STRATA_DEFINITION: dict = {
    "small_lt_500_mm3": "V < 500 mm³（< 0.5 cc）",
    "medium_500_to_1000_mm3": "500 ≤ V ≤ 1000 mm³（0.5–1.0 cc）",
    "large_gt_1000_mm3": "V > 1000 mm³（> 1.0 cc）",
    "note": "exploratory size strata, not clinical risk categories",
}

INSTANCE_DEFINITIONS: dict = {
    "lesion_instance": (
        "3D connected component（6-邻域 / face connectivity，CONNECTIVITY=1）；"
        "仅角或边接触的体素不属于同一 lesion"
    ),
    "lesion_sensitivity_any_overlap": (
        "matched reference lesions / all reference lesions（any-overlap 匹配）；"
        "未匹配 lesion = missed，仍留在分母，不得从分母删除"
    ),
    "matched_lesion_dice": (
        "只作用于一对一 matched pairs：2|GT∩Pred| / (|GT|+|Pred|)；"
        "天然排除全部 missed lesions，不得描述为 overall lesion Dice"
    ),
    "predicted_lesions": (
        "prediction components 总数（含阴性病例的假阳组件；阴性病例不进 sensitivity 分母）"
    ),
    "unmatched_predicted_lesions": (
        "unmatched 的 prediction components / 全部病例；不改变现有 negative-case FP 主指标"
    ),
    "not_challenge_metric": (
        "segmentation failure analysis；不是 PI-CAI challenge detection metric"
        "（不计算 AUROC / average precision / FROC / detection score）"
    ),
}


def lesion_size_stratum(volume_mm3: float) -> str:
    """探索性病灶大小分层（预先冻结，**不是**临床风险类别）。

    边界严格固定：``499.999 -> small``，``500.000 -> medium``，``1000.000 -> medium``，
    ``1000.001 -> large``。分层依据是**单个 reference lesion** 的物理体积（mm³）。
    """
    if volume_mm3 < LESION_SIZE_SMALL_MAX_MM3:
        return "small_lt_500_mm3"
    if volume_mm3 <= LESION_SIZE_MEDIUM_MAX_MM3:
        return "medium_500_to_1000_mm3"
    return "large_gt_1000_mm3"


def _min_cost_flow_unit_matching(
    n_ref: int,
    n_pred: int,
    weights: dict[tuple[int, int], int],
    max_units: int,
) -> tuple[int, int]:
    """二部图 unit-capacity 最小费用流的增量增广（返回流量与最大总权重）。

    网络：source(0) -> reference(1..n_ref) -> prediction(n_ref+1..n_ref+n_pred) -> sink。
    所有容量为 1；``reference -> prediction`` 边费用 = ``-intersection_voxels``。每增广1
    单位恰好对应匹配中增加 1 条边。SSP（successive shortest paths）性质保证**每个流量值下
    费用最小**，即该基数下匹配总权重最大；增广不超过 ``max_units`` 次。

    图极小（病例内病灶数级别），用 Bellman-Ford 求最小费用增广路；候选边按
    ``sorted(weights)`` 固定顺序插入，不依赖 dict / set 迭代顺序。
    """
    if max_units <= 0 or n_ref <= 0 or n_pred <= 0:
        return 0, 0
    source = 0
    sink = n_ref + n_pred + 1
    # 每条边: [to, residual_capacity, cost, 反向边在其邻接表中的下标]
    graph: list[list[list[int]]] = [[] for _ in range(sink + 1)]

    def add_edge(u: int, v: int, cost: int) -> None:
        graph[u].append([v, 1, cost, len(graph[v])])
        graph[v].append([u, 0, -cost, len(graph[u]) - 1])

    for r in range(1, n_ref + 1):
        add_edge(source, r, 0)
    for p in range(1, n_pred + 1):
        add_edge(n_ref + p, sink, 0)
    for r, p in sorted(weights):
        add_edge(r, n_ref + p, -int(weights[(r, p)]))

    units = 0
    total_weight = 0
    while units < max_units:
        dist = [math.inf] * (sink + 1)
        in_edge: list[tuple[int, int] | None] = [None] * (sink + 1)
        dist[source] = 0.0
        for _ in range(sink):
            updated = False
            for u in range(sink + 1):
                if dist[u] == math.inf:
                    continue
                base = dist[u]
                for edge_index, edge in enumerate(graph[u]):
                    if edge[1] > 0 and base + edge[2] < dist[edge[0]]:
                        dist[edge[0]] = base + edge[2]
                        in_edge[edge[0]] = (u, edge_index)
                        updated = True
            if not updated:
                break
        if dist[sink] == math.inf or in_edge[sink] is None:
            break
        node = sink
        while node != source:
            u, edge_index = in_edge[node]  # type: ignore[misc]
            edge = graph[u][edge_index]
            edge[1] -= 1
            graph[node][edge[3]][1] += 1
            node = u
        units += 1
        total_weight += -int(dist[sink])
    return units, total_weight


def _max_weight_matching_exact_size(
    n_ref: int, n_pred: int, weights: dict[tuple[int, int], int], size: int
) -> int | None:
    """恰好 ``size`` 条边的匹配的最大总权重；不存在这样的匹配时返回 ``None``。"""
    units, weight = _min_cost_flow_unit_matching(n_ref, n_pred, weights, size)
    return weight if units == size else None


def _match_lesion_instances(
    intersections: dict[tuple[int, int], int], n_ref: int, n_pred: int
) -> list[tuple[int, int, int]]:
    """reference 与 prediction 病灶实例的一对一**字典序**匹配（协议预先冻结）。

    候选边：``intersection >= 1`` voxel。目标按字典序：

    1. 最大化匹配数（maximum cardinality）——尽可能多的 reference lesion 被匹配；
    2. 相同匹配数下最大化总 intersection 体素数；
    3. 仍相同则按 reference / prediction component id 升序给出确定性唯一解。

    **不**采用「匈牙利最大化 Dice / IoU」或「贪心 Dice / 重叠」作为 matching 目标：
    matched-lesion Dice 本身是报告终点之一，若先最大化 Dice 会产生
    metric-optimizing-the-metric 风险；Dice 只用于匹配后的质量评价。

    返回按 ref id 升序的 ``[(ref_id, pred_id, intersection_voxels), ...]``。
    实现：先求 (最大基数, 最大总权重)，再用「精确可行性判据 + id 升序贪心」确定唯一解——
    结果只由 (交点权重, 组件 id) 决定，与容器迭代顺序、随机性、文件/模型/病例顺序无关。
    """
    weights = {
        (int(r), int(p)): int(w) for (r, p), w in intersections.items() if int(w) >= 1
    }
    if n_ref <= 0 or n_pred <= 0 or not weights:
        return []
    total_units, best_weight = _min_cost_flow_unit_matching(
        n_ref, n_pred, weights, min(n_ref, n_pred)
    )
    if total_units <= 0:
        return []

    matched: list[tuple[int, int, int]] = []
    used_preds: set[int] = set()
    fixed_weight = 0
    for r in range(1, n_ref + 1):
        remaining = total_units - len(matched)
        if remaining <= 0:
            break
        need = best_weight - fixed_weight
        for p in sorted(pred for (ref, pred) in weights if ref == r):
            if p in used_preds:
                continue
            edge_weight = weights[(r, p)]
            if edge_weight > need:
                continue  # 单条边已超过剩余预算：不可能属于任何最优匹配
            sub = {
                (ref - r, pred): w
                for (ref, pred), w in weights.items()
                if ref > r and pred not in used_preds and pred != p
            }
            candidate = _max_weight_matching_exact_size(
                n_ref - r, n_pred, sub, remaining - 1
            )
            if candidate is not None and (
                candidate + fixed_weight + edge_weight == best_weight
            ):
                matched.append((r, p, edge_weight))
                used_preds.add(p)
                fixed_weight += edge_weight
                break
    if len(matched) != total_units or fixed_weight != best_weight:
        raise EvaluationError(
            f"matching inconsistency: 贪心结果 {len(matched)} 对 / 总 intersection "
            f"{fixed_weight} 与最大基数 {total_units} / 最大总 intersection "
            f"{best_weight} 不一致"
        )
    return matched


def _validate_lesion_matching(
    pairs: Sequence[tuple[int, int, int]], n_ref: int, n_pred: int
) -> None:
    """匹配后内部一致性断言（不一致时抛 ``EvaluationError``，整体 fail-closed）。

    保证：一对一（同一组件不被匹配两次）、每对 intersection >= 1、匹配数不超过任一侧
    实例数；``unmatched_prediction = prediction_count - matched_prediction`` 由计数恒等式保证。
    """
    refs = [r for r, _p, _w in pairs]
    preds = [p for _r, p, _w in pairs]
    if len(set(refs)) != len(refs):
        raise EvaluationError(
            f"matching inconsistency: reference component 重复匹配: {refs}"
        )
    if len(set(preds)) != len(preds):
        raise EvaluationError(
            f"matching inconsistency: prediction component 重复匹配: {preds}"
        )
    for r, p, w in pairs:
        if not (1 <= r <= n_ref) or not (1 <= p <= n_pred):
            raise EvaluationError(
                f"matching inconsistency: component id 越界: {(r, p)}"
            )
        if w < 1:
            raise EvaluationError(
                f"matching inconsistency: matched pair 的 intersection < 1 voxel: {(r, p, w)}"
            )
    if len(pairs) > n_ref or len(pairs) > n_pred:
        raise EvaluationError(
            f"matching inconsistency: 匹配数 {len(pairs)} 超过 reference {n_ref} "
            f"或 prediction {n_pred}"
        )


def lesion_instance_metrics(
    pred: np.ndarray, ref: np.ndarray, voxel_volume_mm3_value: float, case_id: str
) -> dict:
    """单病例的病灶实例级指标（full 模式专用；不使用任何预测后处理）。

    - 实例 = 3D 连通域（``CONNECTIVITY = 1``，6-邻域 / face connectivity）；
    - 候选匹配 = intersection >= 1 voxel；一对一字典序匹配见 :func:`_match_lesion_instances`；
    - ``lesion_sensitivity_any_overlap`` = matched reference lesions / all reference lesions，
      未匹配（missed）lesion **不从分母删除**；``n_ref == 0``（阴性病例）为 0/0 -> ``None``，
      不进入分母，但其 prediction components 全部计入 ``unmatched_predicted_lesions``；
    - 参考实例体积 = 体素数 x该病例真实 spacing 乘积（mm³），不使用数组 shape 猜 spacing；
    - 不做最小团块过滤 / 最大团块 / 形态学开闭 / 填洞 / 阈值优化等任何后处理。

    内部一致性（一对一、intersection >= 1、计数恒等式）不满足时抛 :class:`EvaluationError`。
    """
    from scipy import ndimage

    structure = ndimage.generate_binary_structure(3, CONNECTIVITY)
    ref_lab, n_ref_comp = ndimage.label(ref, structure=structure)
    pred_lab, n_pred_comp = ndimage.label(pred, structure=structure)
    ref_sizes = (
        np.bincount(ref_lab.ravel())[1:] if n_ref_comp else np.array([], dtype=np.int64)
    )
    pred_sizes = (
        np.bincount(pred_lab.ravel())[1:]
        if n_pred_comp
        else np.array([], dtype=np.int64)
    )

    intersections: dict[tuple[int, int], int] = {}
    if n_ref_comp and n_pred_comp:
        overlap = (ref_lab > 0) & (pred_lab > 0)
        if bool(overlap.any()):
            # 联合直方图一次算完全部 (ref component, pred component) 交集体素数
            codes = ref_lab[overlap].astype(np.int64) * (n_pred_comp + 1) + pred_lab[
                overlap
            ].astype(np.int64)
            counts = np.bincount(codes, minlength=(n_ref_comp + 1) * (n_pred_comp + 1))
            for code in np.nonzero(counts)[0]:
                r, p = divmod(int(code), n_pred_comp + 1)
                if r >= 1 and p >= 1:
                    intersections[(r, p)] = int(counts[code])

    pairs = _match_lesion_instances(intersections, n_ref_comp, n_pred_comp)
    _validate_lesion_matching(pairs, n_ref_comp, n_pred_comp)
    pair_by_ref = {r: (p, w) for r, p, w in pairs}
    pair_by_pred = {p: (r, w) for r, p, w in pairs}

    reference_records: list[dict] = []
    prediction_records: list[dict] = []
    matched_dices: list[float] = []
    for comp_id in range(1, n_ref_comp + 1):
        voxels = int(ref_sizes[comp_id - 1])
        volume = float(voxels) * float(voxel_volume_mm3_value)
        record: dict = {
            "reference_component_id": comp_id,
            "voxels": voxels,
            "volume_mm3": _finite(volume),
            "size_stratum": lesion_size_stratum(volume),
            "matched": comp_id in pair_by_ref,
            "matched_prediction_component_id": None,
            "intersection_voxels": None,
            "matched_dice": None,
        }
        if comp_id in pair_by_ref:
            pred_id, intersection = pair_by_ref[comp_id]
            pred_voxels = int(pred_sizes[pred_id - 1])
            dice = 2.0 * intersection / (voxels + pred_voxels)
            matched_dices.append(dice)
            record["matched_prediction_component_id"] = pred_id
            record["intersection_voxels"] = intersection
            record["matched_dice"] = _finite(dice)
        reference_records.append(record)
    for comp_id in range(1, n_pred_comp + 1):
        voxels = int(pred_sizes[comp_id - 1])
        record = {
            "prediction_component_id": comp_id,
            "voxels": voxels,
            "volume_mm3": _finite(float(voxels) * float(voxel_volume_mm3_value)),
            "matched": comp_id in pair_by_pred,
            "matched_reference_component_id": None,
            "intersection_voxels": None,
        }
        if comp_id in pair_by_pred:
            ref_id, intersection = pair_by_pred[comp_id]
            record["matched_reference_component_id"] = ref_id
            record["intersection_voxels"] = intersection
        prediction_records.append(record)

    matched_count = len(pairs)
    return {
        "case_id": case_id,
        "connectivity": CONNECTIVITY,
        "connectivity_name": "3D 6-neighborhood",
        "reference_lesions": int(n_ref_comp),
        "matched_reference_lesions": int(matched_count),
        "lesion_sensitivity_any_overlap": _ratio(matched_count, n_ref_comp),
        # prediction 侧：matched 与 reference 侧必然成对（一对一），unmatched = 全部假阳组件
        "predicted_lesions": int(n_pred_comp),
        "matched_predicted_lesions": int(matched_count),
        "unmatched_predicted_lesions": int(n_pred_comp - matched_count),
        "matched_lesion_dice": _dist_stats(matched_dices),
        "reference_lesion_records": reference_records,
        "prediction_lesion_records": prediction_records,
    }


def aggregate_lesion_instance_metrics(entries: dict[str, dict]) -> dict:
    """把逐病例实例级结果（``entries[cid]["lesion_instance_metrics"]``）聚合为全局块。

    只做计数与分布聚合，**不重新匹配**；病例顺序固定为 case id 升序。分母一律是**全部
    reference lesions**（阴性病例贡献 0 个分母，不参与 sensitivity）。

    ``entries`` 只要求每个 value 含``lesion_instance_metrics``；:func:`aggregate_overlay`
    的 FP burden 块与此**并列**输出，不是本函数的子集。
    """
    per_case = [entries[cid]["lesion_instance_metrics"] for cid in sorted(entries)]
    reference_total = sum(m["reference_lesions"] for m in per_case)
    matched_total = sum(m["matched_reference_lesions"] for m in per_case)
    predicted_total = sum(m["predicted_lesions"] for m in per_case)
    matched_pred_total = sum(m["matched_predicted_lesions"] for m in per_case)
    unmatched_pred_total = sum(m["unmatched_predicted_lesions"] for m in per_case)

    all_dices: list[float] = []
    strata: dict[str, dict] = {
        key: {
            "reference_lesion_count": 0,
            "matched_reference_lesion_count": 0,
            "dices": [],
        }
        for key in LESION_SIZE_STRATUM_KEYS
    }
    for metrics in per_case:
        for record in metrics["reference_lesion_records"]:
            key = record["size_stratum"]
            if key not in strata:
                raise EvaluationError(f"未知的病灶大小分层: {key!r}")
            strata[key]["reference_lesion_count"] += 1
            if record["matched"]:
                strata[key]["matched_reference_lesion_count"] += 1
            if record["matched_dice"] is not None:
                dice = float(record["matched_dice"])
                all_dices.append(dice)
                strata[key]["dices"].append(dice)

    size_strata = {
        key: {
            "reference_lesion_count": strata[key]["reference_lesion_count"],
            "matched_reference_lesion_count": strata[key]["matched_reference_lesion_count"],
            # 该层0 个 reference lesion 时sensitivity 为 null，不是 0
            "lesion_sensitivity_any_overlap": _ratio(
                strata[key]["matched_reference_lesion_count"],
                strata[key]["reference_lesion_count"],
            ),
            "matched_lesion_dice": _dist_stats(strata[key]["dices"]),
        }
        for key in LESION_SIZE_STRATUM_KEYS
    }

    return _json_safe(
        {
            "connectivity": CONNECTIVITY,
            "connectivity_name": "3D 6-neighborhood",
            "matching": MATCHING_DEFINITION,
            "size_strata_definition": SIZE_STRATA_DEFINITION,
            "n_cases": len(per_case),
            "reference_lesions": int(reference_total),
            "matched_reference_lesions": int(matched_total),
            "lesion_sensitivity_any_overlap": _ratio(matched_total, reference_total),
            # 与 lesion_sensitivity_any_overlap 按定义为同一量，单独列出以便直观阅读
            "matched_reference_fraction": _ratio(matched_total, reference_total),
            "small_lesion_sensitivity_any_overlap": _ratio(
                strata["small_lt_500_mm3"]["matched_reference_lesion_count"],
                strata["small_lt_500_mm3"]["reference_lesion_count"],
            ),
            # FP lesions / case 口径：unmatched predicted components / 全部病例（含阴性病例假阳）
            "predicted_lesions": int(predicted_total),
            "matched_predicted_lesions": int(matched_pred_total),
            "unmatched_predicted_lesions": int(unmatched_pred_total),
            "matched_lesion_dice": _dist_stats(all_dices),
            "size_strata": size_strata,
            "definitions": INSTANCE_DEFINITIONS,
        }
    )


# --------------------------------------------------------------------------- 历史私有别名
_lesion_size_stratum = lesion_size_stratum
_aggregate_lesion_instance_metrics = aggregate_lesion_instance_metrics
