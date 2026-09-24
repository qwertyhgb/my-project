#!/usr/bin/env python3
"""Prostate158 独立外测病灶分割评估入口（清单驱动；不伪装成 nnU-Net validation 产物）。

与 ``scripts/evaluate_segmentation.py`` 的边界
---------------------------------------------
- 内部评估器读取 ``FOLD_DIR/validation/summary.json``；**外测没有也不允许伪造这种产物**。
  本入口读取 prepare 脚本生成的 ``external_input_manifest.json``（病例清单、原始参考标签路径、
  输入准备版本）与独立预测目录中的 ``<case_id>.nii.gz``。
- 指标口径与内部评估器**完全一致**（通过 importlib 复用其纯函数，不复制实现、不改其 CLI 行为）：
  阳性宏平均 Dice（完全漏分记 0）、median/micro Dice、体素召回、仅阳性 precision、
  含阴性假阳的 overall precision、完全漏分、阴性假阳病例数与假阳体积。
- **只评估病灶分割**：不计算 AUROC / average precision / FROC / PI-CAI challenge score。

固定规则（fail-closed）
-----------------------
- 主参考预先固定为 ``adc_tumor_reader1``；``adc_tumor_reader2`` 只能用 ``--reader reader2`` 在
  **实际存在且可核对的病例子集**上做单独的读者敏感性分析，不逐例选标签、不与 reader1 混组。
- 队列（official test / 原作者 train / valid）由清单决定，分开报告，不合并冒充官方测试集。
- 模型与 checkpoint 必须在查看 Prostate158 结果之前固定；多模型比较要求同一清单版本、
  同一读者、同一病例集合与同一批参考标签路径，否则拒绝比较（不做交集静默退化）。
- 预测与参考不在同一网格时：仅当物理坐标关系可信（方向正交归一、同一坐标系、预测 FOV
  覆盖参考网格）才把**派生预测掩膜**以最近邻插值映射到参考网格；**原始参考标签绝不重采样**，
  不做数组索引强比，不估计未知配准；评估对齐事实写入报告。
- 任一病例失败 / 标签非 0/1 / 几何不可信时，收集全部错误后非零退出，**不发布**结果 JSON。
- 队列无阴性病例时，阴性假阳相关指标（含 overall precision）记 ``not_applicable``，绝不填 0。
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

SCHEMA_VERSION = "1.0"
DEFAULT_BOOTSTRAP_RESAMPLES = 10000
DEFAULT_SEED = 20260923
DIRECTION_ORTHONORMAL_TOL = 1e-5
COVERAGE_TOL_MM = 1e-3
ALLOWED_LABEL_VALUES = (0.0, 1.0)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERNAL_EVALUATOR = PROJECT_ROOT / "scripts" / "evaluate_segmentation.py"

READER_PRIMARY = "reader1"
READER_SENSITIVITY = "reader2"


def _load_internal_evaluator():
    """复用内部评估器的纯函数（不修改其文件、不改变其 CLI 与测试行为）。"""
    spec = importlib.util.spec_from_file_location(
        "picsnunu_internal_evaluate_segmentation", INTERNAL_EVALUATOR
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载内部评估器: {INTERNAL_EVALUATOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_INTERNAL = _load_internal_evaluator()
EvaluationError = _INTERNAL.EvaluationError
_ratio = _INTERNAL._ratio
_json_safe = _INTERNAL._json_safe
_publish_json = _INTERNAL._publish_json
_dist_stats = _INTERNAL._dist_stats
_recompute_case_counts = _INTERNAL.recompute_case_counts
_bootstrap_ci_mean = _INTERNAL.bootstrap_ci_mean
_bootstrap_ci_paired_mean = _INTERNAL.bootstrap_ci_paired_mean
TIE_TOLERANCE = _INTERNAL.TIE_TOLERANCE


def _now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- 几何（与准备脚本同口径）
def _geom(img) -> dict:
    return {
        "size": tuple(int(v) for v in img.GetSize()),
        "spacing": tuple(float(v) for v in img.GetSpacing()),
        "origin": tuple(float(v) for v in img.GetOrigin()),
        "direction": tuple(float(v) for v in img.GetDirection()),
    }


def _same_tuple(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    return len(a) == len(b) and all(float(x) == float(y) for x, y in zip(a, b))


def same_geometry(g1: dict, g2: dict) -> bool:
    return all(
        _same_tuple(g1[k], g2[k]) for k in ("size", "spacing", "origin", "direction")
    )


def _direction_matrix(g: dict) -> np.ndarray:
    return np.asarray(g["direction"], dtype=float).reshape(3, 3)


def _orthonormal(g: dict) -> bool:
    m = _direction_matrix(g)
    if m.shape != (3, 3) or not np.isfinite(m).all():
        return False
    if not all(math.isfinite(s) and s > 0 for s in g["spacing"]):
        return False
    if not all(math.isfinite(o) for o in g["origin"]):
        return False
    return bool(
        np.allclose(m.T @ m, np.eye(3), atol=DIRECTION_ORTHONORMAL_TOL)
    ) and bool(abs(abs(np.linalg.det(m)) - 1.0) < DIRECTION_ORTHONORMAL_TOL)


def _fov_corners(g: dict) -> np.ndarray:
    spacing = np.asarray(g["spacing"], dtype=float)
    n = np.asarray(g["size"], dtype=float)
    origin = np.asarray(g["origin"], dtype=float)
    axes = _direction_matrix(g) @ np.diag(spacing)
    out = []
    for bits in range(8):
        k = np.array([(bits >> a) & 1 for a in range(3)], dtype=float) * (n - 1.0)
        out.append(origin + axes @ k)
    return np.asarray(out, dtype=float)


def fov_covers(source: dict, target: dict, tol_mm: float = COVERAGE_TOL_MM) -> bool:
    """source 网格物理 FOV 是否覆盖 target 的 8 个物理角点（同一物理坐标系前提下）。"""
    if not (_orthonormal(source) and _orthonormal(target)):
        return False
    if not np.allclose(
        _direction_matrix(source),
        _direction_matrix(target),
        atol=DIRECTION_ORTHONORMAL_TOL,
    ):
        return False
    sp = np.asarray(source["spacing"], dtype=float)
    tol_idx = tol_mm / np.maximum(sp, 1e-12)
    rot = _direction_matrix(source)
    origin = np.asarray(source["origin"], dtype=float)
    n = np.asarray(source["size"], dtype=float)
    for p in _fov_corners(target):
        idx = rot.T @ (p - origin) / sp
        if (
            np.any(~np.isfinite(idx))
            or np.any(idx < -tol_idx)
            or np.any(idx > (n - 1) + tol_idx)
        ):
            return False
    return True


# --------------------------------------------------------------------------- 清单与预测
def load_manifest(path: Path) -> dict:
    if not path.is_file():
        raise EvaluationError(f"清单不存在: {path}")
    try:
        with open(path, encoding="utf-8") as fh:
            manifest = json.load(fh)
    except json.JSONDecodeError as exc:
        raise EvaluationError(f"清单不是合法 JSON: {exc}") from exc
    if manifest.get("kind") != "prostate158_external_input_manifest":
        raise EvaluationError(
            f"清单 kind 不是 prostate158_external_input_manifest: {manifest.get('kind')!r}"
        )
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise EvaluationError("清单缺少非空 cases")
    ids = [c.get("case_id") for c in cases]
    if len(set(ids)) != len(ids):
        raise EvaluationError("清单病例 ID 有重复")
    return manifest


def _ref_for_reader(case: dict, reader: str) -> tuple[str, str]:
    """返回 (参考标签路径, 清单中的状态 positive/negative)；reader2 缺失即 fail-closed。"""
    if reader == READER_PRIMARY:
        ref = case.get("primary_reference", {})
        path, status = ref.get("path"), ref.get("status")
        if not path or status not in ("positive", "negative"):
            raise EvaluationError(
                f"{case.get('case_id')}: 主参考 {ref.get('column')} 状态非法（{status!r}）"
            )
        return path, status
    ref = case.get("reader2_reference", {})
    path, status = ref.get("path"), ref.get("status")
    if not path or status not in ("positive", "negative"):
        raise EvaluationError(
            f"{case.get('case_id')}: reader2 敏感性分析要求该例存在可核对的 "
            f"adc_tumor_reader2（实际状态 {status!r}）；不得回退到 reader1"
        )
    return path, status


def select_cases(manifest: dict, reader: str) -> list[dict]:
    cases = manifest["cases"]
    if reader == READER_PRIMARY:
        return list(cases)
    subset = [
        c
        for c in cases
        if c.get("reader2_reference", {}).get("status") in ("positive", "negative")
    ]
    if not subset:
        raise EvaluationError("清单中没有任何病例具备可核对的 reader2 标签")
    return subset


def _binary_array(img, tag: str) -> np.ndarray:
    import SimpleITK as sitk

    arr = sitk.GetArrayFromImage(img)
    if not bool(np.isfinite(arr).all()):
        raise EvaluationError(f"{tag} 含非有限值")
    bad = [float(v) for v in np.unique(arr) if float(v) not in ALLOWED_LABEL_VALUES]
    if bad:
        raise EvaluationError(f"{tag} 含非 0/1 标签值（前 5 个）: {bad[:5]}")
    return arr > 0.5


def evaluate_case(case: dict, reader: str, prediction_dir: Path) -> dict:
    """评估一例；几何不可信时抛 EvaluationError。返回逐例计数与对齐记录。"""
    import SimpleITK as sitk

    cid = case["case_id"]
    ref_path, manifest_status = _ref_for_reader(case, reader)
    pred_path = prediction_dir / f"{cid}.nii.gz"
    if not pred_path.is_file():
        raise EvaluationError(f"{cid}: 缺少预测文件 {pred_path}")
    if not Path(ref_path).is_file():
        raise EvaluationError(f"{cid}: 参考标签不存在 {ref_path}")

    pred_img = sitk.ReadImage(str(pred_path))
    ref_img = sitk.ReadImage(str(ref_path))
    pred_geom, ref_geom = _geom(pred_img), _geom(ref_img)

    aligned = same_geometry(pred_geom, ref_geom)
    alignment = {
        "evaluation_alignment": "none_same_grid"
        if aligned
        else "nearest_neighbor_prediction_to_reference",
        "prediction_geometry": pred_geom,
        "reference_geometry": ref_geom,
        "reference_resampled": False,
    }
    if not aligned:
        if not (_orthonormal(pred_geom) and _orthonormal(ref_geom)):
            raise EvaluationError(
                f"{cid}: 预测/参考方向矩阵不合法，不能在物理空间对齐（不做配准）"
            )
        if not np.allclose(
            _direction_matrix(pred_geom),
            _direction_matrix(ref_geom),
            atol=DIRECTION_ORTHONORMAL_TOL,
        ):
            raise EvaluationError(
                f"{cid}: 预测与参考不在同一物理坐标系（direction 不同），不做配准"
            )
        if not fov_covers(pred_geom, ref_geom):
            raise EvaluationError(
                f"{cid}: 预测网格物理 FOV 未覆盖参考网格，最近邻映射会静默丢区域；拒绝评估"
            )
        flt = sitk.ResampleImageFilter()
        flt.SetReferenceImage(ref_img)
        flt.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
        flt.SetInterpolator(sitk.sitkNearestNeighbor)
        flt.SetDefaultPixelValue(0)
        flt.SetOutputPixelType(sitk.sitkUInt8)
        pred_img = flt.Execute(pred_img)
        alignment["interpolator"] = (
            "SimpleITK.sitkNearestNeighbor（仅作用于派生预测掩膜）"
        )

    pred = _binary_array(pred_img, f"{cid} prediction")
    ref = _binary_array(ref_img, f"{cid} reference")

    counts = _recompute_case_counts(pred, ref)
    if manifest_status == "negative" and counts["n_ref"] != 0:
        raise EvaluationError(
            f"{cid}: 清单/标签状态为阴性，但参考掩膜实际非空（n_ref={counts['n_ref']}）"
        )
    if manifest_status == "positive" and counts["n_ref"] == 0:
        raise EvaluationError(f"{cid}: 清单状态为阳性，但参考掩膜实际为空")

    denom = 2 * counts["TP"] + counts["FP"] + counts["FN"]
    dice = None if denom == 0 else 2 * counts["TP"] / denom
    return {
        "case_id": cid,
        "reader": reader,
        "reference_file": ref_path,
        "prediction_file": str(pred_path),
        "manifest_reference_status": manifest_status,
        "has_reference": counts["n_ref"] > 0,
        **counts,
        "dice": dice,
        "alignment": alignment,
    }


# --------------------------------------------------------------------------- 汇总指标
def summarize_model(
    model_name: str,
    prediction_dir: Path,
    cases: list[dict],
    reader: str,
    bootstrap_resamples: int,
    seed: int,
    progress: bool,
) -> tuple[dict, dict[str, dict]]:
    work = cases
    if progress:
        from tqdm import tqdm

        work = tqdm(
            cases,
            desc=f"evaluate[{model_name}]",
            unit="case",
            mininterval=1.0,
            file=sys.stdout,
        )

    errors: list[str] = []
    evaluated: list[dict] = []
    aligned_cases = 0
    for case in work:
        try:
            entry = evaluate_case(case, reader, prediction_dir)
        except EvaluationError as exc:
            errors.append(f"[{model_name}] {case.get('case_id')}: {exc}")
            continue
        except Exception as exc:  # noqa: BLE001
            errors.append(
                f"[{model_name}] {case.get('case_id')}: {type(exc).__name__}: {exc}"
            )
            continue
        evaluated.append(entry)
        if entry["alignment"]["evaluation_alignment"] != "none_same_grid":
            aligned_cases += 1
    if errors:
        raise EvaluationError(
            "逐例评估失败，不发布结果：\n"
            + "\n".join(f"  {e}" for e in errors[:50])
            + (f"\n  ... 另有 {len(errors) - 50} 条" if len(errors) > 50 else "")
        )

    by_id = {e["case_id"]: e for e in evaluated}
    pos = sorted(
        (e for e in evaluated if e["has_reference"]), key=lambda e: e["case_id"]
    )
    neg = sorted(
        (e for e in evaluated if not e["has_reference"]), key=lambda e: e["case_id"]
    )
    pos_dice = [float(e["dice"]) for e in pos]
    tp = sum(e["TP"] for e in pos)
    fp = sum(e["FP"] for e in pos)
    fn = sum(e["FN"] for e in pos)
    neg_fp_voxels = sum(e["FP"] for e in neg)
    neg_fp_cases = [e for e in neg if e["n_pred"] > 0]
    missed = [e for e in pos if e["TP"] == 0]
    empty_pred = [e for e in pos if e["n_pred"] == 0]
    wrong_loc = [e for e in pos if e["TP"] == 0 and e["n_pred"] > 0]
    overlap = [e for e in pos if e["TP"] > 0]

    has_neg = len(neg) > 0
    fp_stats = _dist_stats([float(e["FP"]) for e in neg_fp_cases])
    report = {
        "model": model_name,
        "predictions_dir": str(prediction_dir),
        "case_counts": {
            "num_cases": len(evaluated),
            "num_positive_cases": len(pos),
            "num_negative_cases": len(neg),
            "case_ids_positive": [e["case_id"] for e in pos],
            "case_ids_negative": [e["case_id"] for e in neg],
        },
        "reader": reader,
        "reader_scope": (
            "full_queue"
            if reader == READER_PRIMARY
            else "reader2_available_subset_only（不与 reader1 混组）"
        ),
        "evaluation_grid_alignment": {
            "cases_with_prediction_to_reference_nn_resample": aligned_cases,
            "cases_on_same_grid": len(evaluated) - aligned_cases,
            "rule": (
                "仅在同一物理坐标系、方向正交归一且预测 FOV 覆盖参考网格时，对派生预测掩膜做"
                "最近邻映射；原始参考标签从不重采样"
            ),
        },
        "positive_segmentation": {
            "positive_macro_dice_mean": _json_safe(float(np.mean(pos_dice)))
            if pos_dice
            else None,
            "positive_macro_dice_median": _json_safe(float(np.median(pos_dice)))
            if pos_dice
            else None,
            "positive_macro_dice_q1": _json_safe(float(np.percentile(pos_dice, 25)))
            if pos_dice
            else None,
            "positive_macro_dice_q3": _json_safe(float(np.percentile(pos_dice, 75)))
            if pos_dice
            else None,
            "positive_macro_dice_bootstrap_ci95": _bootstrap_ci_mean(
                pos_dice, bootstrap_resamples, seed
            ),
            "positive_micro_dice": _ratio(2 * tp, 2 * tp + fp + fn),
            "definition": "逐例 Dice 宏平均，完全漏分病例记 0；micro 为阳性病例 pooled Dice",
        },
        "voxel_metrics": {
            "positive_voxel_recall": _ratio(tp, tp + fn),
            "positive_voxel_precision": _ratio(tp, tp + fp),
            "all_prediction_voxel_precision": (
                _ratio(tp, tp + fp + neg_fp_voxels) if has_neg else None
            ),
            "all_prediction_voxel_precision_applicability": (
                "applicable"
                if has_neg
                else "not_applicable：该队列/子集没有阴性病例，含阴性假阳的 overall precision 不填 0"
            ),
            "tp": int(tp),
            "fp_positive_cases": int(fp),
            "fn": int(fn),
            "fp_negative_cases": int(neg_fp_voxels),
        },
        "miss_structure": {
            "positive_overlap_cases": len(overlap),
            "positive_overlap_case_rate": _ratio(len(overlap), len(pos)),
            "positive_missed_cases": len(missed),
            "positive_missed_case_rate": _ratio(len(missed), len(pos)),
            "positive_empty_prediction_cases": len(empty_pred),
            "positive_wrong_location_cases": len(wrong_loc),
            "case_ids_missed": [e["case_id"] for e in missed],
            "case_ids_wrong_location": [e["case_id"] for e in wrong_loc],
        },
        "negative_false_positive": {
            "applicable": has_neg,
            "negative_fp_cases": len(neg_fp_cases) if has_neg else None,
            "negative_fp_case_rate": _ratio(len(neg_fp_cases), len(neg))
            if has_neg
            else None,
            "negative_clean_cases": (len(neg) - len(neg_fp_cases)) if has_neg else None,
            "negative_fp_voxels_total": int(neg_fp_voxels) if has_neg else None,
            "negative_fp_voxels_median": fp_stats["median"] if has_neg else None,
            "negative_fp_voxels_mean": fp_stats["mean"] if has_neg else None,
            "negative_fp_voxels_max": fp_stats["max"] if has_neg else None,
            "case_ids_false_positive": [e["case_id"] for e in neg_fp_cases]
            if has_neg
            else [],
            "not_applicable_reason": None if has_neg else "队列/子集无阴性病例",
        },
        "per_case": [
            _json_safe(
                {
                    "case_id": e["case_id"],
                    "has_reference": e["has_reference"],
                    "n_ref": e["n_ref"],
                    "n_pred": e["n_pred"],
                    "TP": e["TP"],
                    "FP": e["FP"],
                    "FN": e["FN"],
                    "TN": e["TN"],
                    "dice": e["dice"],
                    "evaluation_alignment": e["alignment"]["evaluation_alignment"],
                }
            )
            for e in sorted(evaluated, key=lambda e: e["case_id"])
        ],
    }
    return report, by_id


# --------------------------------------------------------------------------- 多模型可比性与配对
def require_comparable(
    manifest: dict, models: list[tuple[str, Path]], reports: dict, cases_by_model: dict
) -> None:
    """同一清单版本、同一读者、同一病例集合与同一参考标签路径；不一致即拒绝比较。"""
    problems: list[str] = []
    ref_name = models[0][0]
    ref_ids = set(cases_by_model[ref_name])
    ref_cases = cases_by_model[ref_name]
    for name, _ in models[1:]:
        ids = set(cases_by_model[name])
        if ids != ref_ids:
            only_a = sorted(ref_ids - ids)
            only_b = sorted(ids - ref_ids)
            problems.append(
                f"[{name}] 与 [{ref_name}] 病例集合不一致：缺少 {len(only_a)} 例"
                f"（{only_a[:5]}）、多出 {len(only_b)} 例（{only_b[:5]}）；禁止只在交集上比较"
            )
            continue
        for cid in sorted(ref_ids):
            a, b = ref_cases[cid], cases_by_model[name][cid]
            if a["reference_file"] != b["reference_file"]:
                problems.append(
                    f"case {cid}: 参考标签路径不一致 {a['reference_file']} vs {b['reference_file']}"
                )
            if a["has_reference"] != b["has_reference"]:
                problems.append(f"case {cid}: 阴阳性状态不一致")
    if problems:
        raise EvaluationError(
            "模型间不可比，拒绝配对比较：\n" + "\n".join(f"  {p}" for p in problems)
        )


def paired_comparison(
    a_name: str,
    b_name: str,
    a_cases: dict,
    b_cases: dict,
    bootstrap_resamples: int,
    seed: int,
) -> dict:
    pos_ids = sorted(cid for cid, c in a_cases.items() if c["has_reference"])
    neg_ids = sorted(cid for cid, c in a_cases.items() if not c["has_reference"])
    deltas = []
    for cid in pos_ids:
        da = float(a_cases[cid]["dice"] or 0.0)
        db = float(b_cases[cid]["dice"] or 0.0)
        deltas.append(db - da)
    improved = sum(1 for d in deltas if d > TIE_TOLERANCE)
    worsened = sum(1 for d in deltas if d < -TIE_TOLERANCE)
    tied = len(deltas) - improved - worsened
    new_overlap = [
        cid for cid in pos_ids if a_cases[cid]["TP"] == 0 and b_cases[cid]["TP"] > 0
    ]
    lost_overlap = [
        cid for cid in pos_ids if a_cases[cid]["TP"] > 0 and b_cases[cid]["TP"] == 0
    ]

    def sums(cases, ids):
        return (
            sum(cases[i]["TP"] for i in ids),
            sum(cases[i]["FP"] for i in ids),
            sum(cases[i]["FN"] for i in ids),
        )

    tpa, fpa, fna = sums(a_cases, pos_ids)
    tpb, fpb, fnb = sums(b_cases, pos_ids)
    nva = sum(1 for i in neg_ids if a_cases[i]["n_pred"] > 0)
    nvb = sum(1 for i in neg_ids if b_cases[i]["n_pred"] > 0)
    micro_a, micro_b = (
        _ratio(2 * tpa, 2 * tpa + fpa + fna),
        _ratio(2 * tpb, 2 * tpb + fpb + fnb),
    )
    rec_a, rec_b = _ratio(tpa, tpa + fna), _ratio(tpb, tpb + fnb)
    prec_a, prec_b = _ratio(tpa, tpa + fpa), _ratio(tpb, tpb + fpb)
    ci = _bootstrap_ci_paired_mean(deltas, bootstrap_resamples, seed)

    def delta(x, y):
        return None if x is None or y is None else float(y) - float(x)

    return _json_safe(
        {
            "model_a": a_name,
            "model_b": b_name,
            "delta_definition": "model_b - model_a",
            "mean_paired_dice_delta": float(np.mean(deltas)) if deltas else None,
            "median_paired_dice_delta": float(np.median(deltas)) if deltas else None,
            "paired_bootstrap_ci95": ci,
            "ci_excludes_zero": None if ci is None else bool(ci[0] > 0 or ci[1] < 0),
            "improved_cases": improved,
            "tied_cases": tied,
            "worsened_cases": worsened,
            "new_overlap_cases": len(new_overlap),
            "new_overlap_case_ids": new_overlap,
            "lost_overlap_cases": len(lost_overlap),
            "lost_overlap_case_ids": lost_overlap,
            "complete_miss_delta": (
                sum(1 for i in pos_ids if b_cases[i]["TP"] == 0)
                - sum(1 for i in pos_ids if a_cases[i]["TP"] == 0)
            ),
            "negative_fp_case_delta": nvb - nva,
            "positive_micro_dice_delta": delta(micro_a, micro_b),
            "positive_voxel_recall_delta": delta(rec_a, rec_b),
            "positive_voxel_precision_delta": delta(prec_a, prec_b),
            "negatives": {
                "positive_micro_dice": {"a": micro_a, "b": micro_b},
                "positive_voxel_recall": {"a": rec_a, "b": rec_b},
                "positive_voxel_precision": {"a": prec_a, "b": prec_b},
                "negative_fp_cases": {"a": nva, "b": nvb},
            },
        }
    )


def print_table(reports: dict) -> None:
    def fmt(v):
        return "n/a" if v is None else f"{float(v):.4f}"

    print()
    print("=== Prostate158 外测病灶分割评估 ===")
    header = (
        f"{'model':<22}{'n':>5}{'pos':>5}{'neg':>5}{'macroD':>9}{'medD':>8}"
        f"{'microD':>9}{'recall':>8}{'posPrec':>9}{'allPrec':>9}{'missed':>8}{'negFP':>7}"
    )
    print(header)
    print("-" * len(header))
    for name, rep in reports.items():
        cc, ps, vm, ms, nf = (
            rep["case_counts"],
            rep["positive_segmentation"],
            rep["voxel_metrics"],
            rep["miss_structure"],
            rep["negative_false_positive"],
        )
        print(
            f"{name:<22}{cc['num_cases']:>5}{cc['num_positive_cases']:>5}{cc['num_negative_cases']:>5}"
            f"{fmt(ps['positive_macro_dice_mean']):>9}{fmt(ps['positive_macro_dice_median']):>8}"
            f"{fmt(ps['positive_micro_dice']):>9}{fmt(vm['positive_voxel_recall']):>8}"
            f"{fmt(vm['positive_voxel_precision']):>9}{fmt(vm['all_prediction_voxel_precision']):>9}"
            f"{ms['positive_missed_cases']:>8}{fmt(nf['negative_fp_cases']):>7}"
        )
    print()


# --------------------------------------------------------------------------- CLI
def _parse_model(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--model 格式必须是 name=预测目录")
    name, path = value.split("=", 1)
    name, path = name.strip(), path.strip()
    if not name or not path:
        raise argparse.ArgumentTypeError("--model 的 name 与目录都不能为空")
    return name, Path(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prostate158 独立外测病灶分割评估（清单驱动；不重采样原始参考标签；只评分割）",
    )
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="prepare 脚本生成的 external_input_manifest.json",
    )
    parser.add_argument(
        "--model",
        required=True,
        action="append",
        type=_parse_model,
        help="name=预测目录（可重复；多模型要求同清单/同读者/同病例集合）",
    )
    parser.add_argument(
        "--reader",
        choices=(READER_PRIMARY, READER_SENSITIVITY),
        default=READER_PRIMARY,
        help="reader1=adc_tumor_reader1（默认，主参考）；reader2 仅在可核对子集上做敏感性分析",
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="结果 JSON（已存在则拒绝覆盖）"
    )
    parser.add_argument(
        "--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--note", default=None, help="可选溯源备注（如固定的 checkpoint 名），原样记录"
    )
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    t0 = time.monotonic()
    try:
        if args.bootstrap_resamples < 1:
            raise EvaluationError("--bootstrap-resamples 必须 >= 1")
        manifest = load_manifest(args.manifest)
        selected = select_cases(manifest, args.reader)
        selected_ids = [c["case_id"] for c in selected]
        if len({name for name, _ in args.model}) != len(args.model):
            raise EvaluationError("--model 的 name 有重复")

        reports, cases_by_model = {}, {}
        for name, pred_dir in args.model:
            if not pred_dir.is_dir():
                raise EvaluationError(f"[{name}] 预测目录不存在: {pred_dir}")
            report, by_id = summarize_model(
                name,
                pred_dir,
                selected,
                args.reader,
                args.bootstrap_resamples,
                args.seed,
                progress=not args.no_progress,
            )
            reports[name] = report
            cases_by_model[name] = by_id

        require_comparable(manifest, args.model, reports, cases_by_model)
        comparisons = []
        for i in range(len(args.model)):
            for j in range(i + 1, len(args.model)):
                a, b = args.model[i][0], args.model[j][0]
                comparisons.append(
                    paired_comparison(
                        a,
                        b,
                        cases_by_model[a],
                        cases_by_model[b],
                        args.bootstrap_resamples,
                        args.seed,
                    )
                )

        payload = _json_safe(
            {
                "schema_version": SCHEMA_VERSION,
                "created_at_utc": _now_utc(),
                "kind": "prostate158_external_segmentation_metrics",
                "manifest": str(args.manifest),
                "manifest_id": manifest.get("manifest_id"),
                "queue": manifest.get("queue"),
                "reader": args.reader,
                "selected_case_ids": selected_ids,
                "num_selected_cases": len(selected),
                "bootstrap": {
                    "resamples": args.bootstrap_resamples,
                    "seed": args.seed,
                    "scope": "病例级重采样，不覆盖 run-to-run / 跨 seed 方差",
                },
                "domain_caveat": manifest.get("domain_caveat"),
                "detection_metrics": "不计算 AUROC / average precision / FROC / PI-CAI challenge score",
                "note": args.note,
                "models": reports,
                "paired_comparisons": comparisons,
                "elapsed_s": round(time.monotonic() - t0, 3),
            }
        )
        _publish_json(args.output, payload)
        print_table(reports)
        for cmp_ in comparisons:
            ci = cmp_["paired_bootstrap_ci95"]
            ci_txt = "n/a" if ci is None else f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"
            print(
                f"配对 {cmp_['model_a']} -> {cmp_['model_b']}: "
                f"meanDelta={cmp_['mean_paired_dice_delta']:.4f} CI95={ci_txt} "
                f"improved/tied/worsened={cmp_['improved_cases']}/{cmp_['tied_cases']}/"
                f"{cmp_['worsened_cases']} newOverlap={cmp_['new_overlap_cases']} "
                f"lostOverlap={cmp_['lost_overlap_cases']} negFPCaseDelta="
                f"{cmp_['negative_fp_case_delta']:+d}"
            )
        print("\n=== 外测评估结束汇总 ===")
        print("  status  : completed")
        print(
            f"  queue   : {manifest.get('queue')}  reader: {args.reader}  "
            f"cases: {len(selected)}"
        )
        print(f"  models  : {', '.join(n for n, _ in args.model)}")
        print(f"  elapsed : {time.monotonic() - t0:.2f} s")
        print(f"  output  : {args.output}")
        return 0
    except EvaluationError as exc:
        print(
            f"[evaluate_external_segmentation] 失败（fail-closed，未发布结果）:\n{exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
