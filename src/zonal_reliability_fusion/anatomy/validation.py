"""Stage 1 解剖先验的质量评价：区域重叠、区域一致性、先验可靠度。

用途（与论文的关系）
--------------------
Stage-1 模型**不是**主要创新点，但它的误差是下游 lesion model 的**真实输入分布**的一部分。
因此本模块的输出有两个用途：

1. 报告 Stage-1 自身的质量（WG / PZ / TZ 的 Dice、TP/FP/FN）；
2. 量化「解剖先验有多不确定」——下游的 uncertain 区域必须保留 residual path，取值依据来自
   这里的低置信体素比例，而不是拍脑袋。

口径（fail-closed）
-------------------
- 区域重叠复用 nnU-Net 原生 ``region_or_label_to_mask`` + ``compute_tp_fp_fn_tn``，因此与
  Stage-1 训练所用的 region 定义**逐位一致**；
- 参考与预测**都为空**时 Dice 定义为 ``None``（不是 1，也不是 0）；
- 形状不一致直接报错，不重采样。
"""

from __future__ import annotations

import numpy as np

from zonal_reliability_fusion.anatomy.contracts import ANATOMY_LABELS
from zonal_reliability_fusion.evaluation.protocol import EvaluationError, ratio


def anatomy_region_metrics(reference, prediction) -> dict:
    """按 nnU-Net 原生 region 定义计算 WG / PZ / TZ 的 Dice 与 TP/FP/FN/TN。

    ``reference`` 与 ``prediction`` 都是位编码数组（``WG+2*PZ+4*TZ``）。
    空-空（参考为空）时 Dice 为 ``None``，遵循 nnU-Net 原生规则。
    """
    from nnunetv2.evaluation.evaluate_predictions import (
        compute_tp_fp_fn_tn,
        region_or_label_to_mask,
    )

    from zonal_reliability_fusion.anatomy.contracts import validate_label_array

    validate_label_array(reference, range(8), "anatomy reference")
    validate_label_array(prediction, range(8), "anatomy prediction bitcode")
    if reference.shape != prediction.shape:
        raise EvaluationError("anatomy reference/prediction shape mismatch")
    metrics = {}
    for name, region in list(ANATOMY_LABELS.items())[1:]:
        ref = region_or_label_to_mask(reference, tuple(region))
        pred = region_or_label_to_mask(prediction, tuple(region))
        tp, fp, fn, tn = compute_tp_fp_fn_tn(ref, pred)
        denominator = 2 * int(tp) + int(fp) + int(fn)
        metrics[name] = {
            "Dice": 2 * int(tp) / denominator if denominator else None,
            "TP": int(tp),
            "FP": int(fp),
            "FN": int(fn),
            "TN": int(tn),
            "reference_voxels": int(ref.sum()),
            "prediction_voxels": int(pred.sum()),
        }
    return metrics


def anatomy_region_agreement(masks) -> dict:
    """区域一致性的 4 个比值：PZ/TZ 重叠、两者越出 WG 的体积占比。

    这些不是模型误差指标，而是**数据契约**的健全性检查：位编码下的 PZ/TZ 是否真的落在
    WG 内、是否存在双归属体素。比值分母为 0 时返回 ``None``。
    """
    wg, pz, tz = masks
    wg, pz, tz = (np.asarray(m, dtype=bool) for m in (wg, pz, tz))
    zone_union = pz | tz
    definitions = {
        "PZ_TZ_overlap_over_zone_union": (pz & tz, zone_union),
        "PZ_outside_WG_over_PZ": (pz & ~wg, pz),
        "TZ_outside_WG_over_TZ": (tz & ~wg, tz),
        "zones_outside_WG_over_zone_union": (zone_union & ~wg, zone_union),
    }
    return {
        name: {
            "numerator_voxels": int(numerator.sum()),
            "denominator_voxels": int(denominator.sum()),
            "ratio": ratio(int(numerator.sum()), int(denominator.sum())),
        }
        for name, (numerator, denominator) in definitions.items()
    }


def prior_uncertainty_report(
    wg_probability: np.ndarray,
    pz_probability: np.ndarray,
    tz_probability: np.ndarray,
    *,
    low_confidence_threshold: float,
) -> dict:
    """量化解剖先验的不确定程度（下游 residual path 的取值依据）。

    ``low_confidence_threshold`` 由调用方显式给出，本模块**不设**未经论证的默认值。

    报告三项：

    - ``wg_low_confidence_fraction``：WG 概率落在「既不明确在腺内、也不明确在腺外」的体素
      比例（``threshold <= P(WG) <= 1 - threshold``）——这正是 ROI 裁剪最容易出错的地方；
    - ``zone_undecided_fraction``：``P(PZ)`` 与 ``P(TZ)`` **都**低于 ``threshold`` 的体素比例
      （即 zone 归属不确定的区域）；
    - ``zone_margin``：``|P(PZ) - P(TZ)|`` 的分布，越小越不确定。
    """
    if not 0.0 <= low_confidence_threshold <= 0.5:
        raise EvaluationError(
            "low_confidence_threshold 必须落在 [0, 0.5]（超过 0.5 时低置信区间会与高置信"
            f"区间重叠）；收到 {low_confidence_threshold}"
        )
    wg = np.asarray(wg_probability, dtype=np.float64)
    pz = np.asarray(pz_probability, dtype=np.float64)
    tz = np.asarray(tz_probability, dtype=np.float64)
    if not (wg.shape == pz.shape == tz.shape):
        raise EvaluationError("WG/PZ/TZ 概率形状必须一致")
    total = int(wg.size)
    if total == 0:
        raise EvaluationError("解剖先验为空张量")
    low = float(low_confidence_threshold)
    undecided = (pz < low) & (tz < low)
    margin = np.abs(pz - tz)
    return {
        "low_confidence_threshold": low,
        "voxels": total,
        "wg_low_confidence_fraction": ratio(
            int(((wg >= low) & (wg <= 1.0 - low)).sum()), total
        ),
        "zone_undecided_fraction": ratio(int(undecided.sum()), total),
        "zone_margin": {
            "mean": float(margin.mean()),
            "q1": float(np.percentile(margin, 25)),
            "median": float(np.percentile(margin, 50)),
            "q3": float(np.percentile(margin, 75)),
            "min": float(margin.min()),
            "max": float(margin.max()),
        },
        "definitions": {
            "wg_low_confidence_fraction": (
                "threshold <= P(WG) <= 1 - threshold 的体素比例；ROI 物理边界最容易出错的位置"
            ),
            "zone_undecided_fraction": "P(PZ) 与 P(TZ) 都 < threshold 的体素比例",
            "not_a_calibration_claim": (
                "这是**soft 输出的数值分布**描述，不是校准（calibration）声明，"
                "也不代表真实图像质量"
            ),
        },
    }


def soft_head_metrics(reference, probabilities, *, threshold=0.5):
    """Independent sigmoid-head metrics; never reconstruct heads from ordered export.

    Threshold uses >, matching nnU-Net region export. Quantiles are per-case,
    over all FOV voxels and reference-positive voxels respectively.
    """
    from zonal_reliability_fusion.anatomy.contracts import validate_label_array

    reference = validate_label_array(reference, range(8), "anatomy reference")
    probabilities = np.asarray(probabilities)
    if probabilities.shape != (3, *reference.shape):
        raise EvaluationError("soft anatomy channels/shape mismatch")
    if not np.issubdtype(probabilities.dtype, np.floating):
        raise EvaluationError("soft anatomy must be floating point")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(probabilities > 1):
        raise EvaluationError("soft anatomy probabilities outside finite [0,1]")
    if not 0 < threshold < 1:
        raise EvaluationError("threshold must be in (0,1)")
    result = {}
    for index, name in enumerate(("WG", "PZ", "TZ")):
        ref = (reference & (1 << index)) != 0
        probability = probabilities[index]
        pred = probability > threshold
        tp = int(np.count_nonzero(ref & pred))
        fp = int(np.count_nonzero(~ref & pred))
        fn = int(np.count_nonzero(ref & ~pred))
        result[name] = {
            "Dice": ratio(2 * tp, 2 * tp + fp + fn),
            "recall": ratio(tp, tp + fn), "precision": ratio(tp, tp + fp),
            "TP": tp, "FP": fp, "FN": fn,
            "volume_ratio": ratio(tp + fp, tp + fn),
            "empty_prediction": not bool(pred.any()),
            "probability_quantiles_fov": np.quantile(probability, [0, .1, .5, .9, 1]).tolist(),
            "probability_quantiles_reference": (
                np.quantile(probability[ref], [0, .1, .5, .9, 1]).tolist() if ref.any() else None
            ),
        }
    return result


def diagnose_soft_heads(validation_dir, *, no_progress=False):
    """Read existing validation artifacts only. Return report; no files are written."""
    import json
    import time
    from pathlib import Path

    import SimpleITK as sitk
    from tqdm import tqdm

    from zonal_reliability_fusion.anatomy.contracts import anatomy_same_grid
    from zonal_reliability_fusion.anatomy.inference import load_anatomy_probability_case

    start = time.monotonic()
    folder = Path(validation_dir)
    summary = json.loads((folder / "summary.json").read_text())
    plans = json.loads((folder.parent.parent / "plans.json").read_text())
    records, errors = [], []
    seen = set()
    for entry in tqdm(summary["metric_per_case"], desc="soft anatomy heads", unit="case", disable=no_progress):
        try:
            reference_path = Path(entry["reference_file"])
            cid = reference_path.name.removesuffix(".nii.gz")
            if cid in seen:
                raise EvaluationError(f"duplicate case {cid}")
            seen.add(cid)
            reference = sitk.ReadImage(str(reference_path))
            probability, _, native = load_anatomy_probability_case(
                folder, cid, reference, plans["transpose_forward"]
            )
            anatomy_same_grid(reference, native)
            ref = sitk.GetArrayFromImage(reference)
            hard = sitk.GetArrayFromImage(native)
            reconstructed = np.zeros(ref.shape, dtype=np.uint8)
            for index, label in enumerate((1, 2, 4)):
                reconstructed[probability[index] > .5] = label
            records.append({
                "case_id": cid, "heads": soft_head_metrics(ref, probability),
                "hard_export_mismatch_voxels": int(np.count_nonzero(reconstructed != hard)),
                "hard_regions": anatomy_region_metrics(ref, hard),
            })
        except Exception as exc:
            errors.append(f"{entry.get('reference_file')}: {type(exc).__name__}: {exc}")
    print(f"[soft-head-diagnosis] success={len(records)} failed={len(errors)} skipped=0 "
          f"elapsed={time.monotonic()-start:.2f}s output=stdout (read-only)", flush=True)
    if errors:
        raise EvaluationError("\n".join(errors))
    def available_mean(items, axis=None):
        present = [item for item in items if item is not None]
        if not present:
            return None
        value = np.mean(present, axis=axis)
        return value.tolist() if axis is not None else float(value)

    aggregate = {}
    for name in ("WG", "PZ", "TZ"):
        values = [r["heads"][name] for r in records]
        totals = {key: sum(v[key] for v in values) for key in ("TP", "FP", "FN")}
        tp, fp, fn = (totals[k] for k in ("TP", "FP", "FN"))
        aggregate[name] = {
            **totals,
            **{f"macro_{key}": available_mean([v[key] for v in values])
               for key in ("Dice", "recall", "precision", "volume_ratio")},
            "micro_Dice": ratio(2 * tp, 2 * tp + fp + fn),
            "micro_recall": ratio(tp, tp + fn), "micro_precision": ratio(tp, tp + fp),
            "micro_volume_ratio": ratio(tp + fp, tp + fn),
            "empty_predictions": sum(v["empty_prediction"] for v in values),
            "hard_export_macro_Dice": available_mean([r["hard_regions"][name]["Dice"] for r in records]),
            "mean_per_case_probability_quantiles_fov": np.mean(
                [v["probability_quantiles_fov"] for v in values], axis=0).tolist(),
            "mean_per_case_probability_quantiles_reference": available_mean(
                [v["probability_quantiles_reference"] for v in values], axis=0),
        }
    return {
        "threshold": .5, "threshold_operator": ">", "cases": len(records),
        "quantile_levels": [0, .1, .5, .9, 1],
        "quantile_aggregation": "mean of case quantiles, not pooled voxel quantiles",
        "source": str(folder), "heads": aggregate,
        "hard_export_mismatch_voxels": sum(r["hard_export_mismatch_voxels"] for r in records),
        "per_case": records,
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Read-only independent soft anatomy diagnosis")
    parser.add_argument("validation_dir")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--per-case", action="store_true", help="Also print per-case metrics")
    args = parser.parse_args()
    report = diagnose_soft_heads(args.validation_dir, no_progress=args.no_progress)
    if not args.per_case:
        report.pop("per_case")
    print(json.dumps(report, indent=2, allow_nan=False))
