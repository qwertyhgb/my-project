"""统一病灶分割评估入口的纯合成 CPU 测试。

全部使用 ``tmp_path`` + 合成 ``summary.json`` + 合成小型 NIfTI/数组；CPU 运行；不读取任何真实
医学数据，不写 ``data/`` ``workdir/`` ``outputs/``，不实例化 nnU-Net Trainer，不启动 inference，
不使用 GPU。测试不依赖当前真实 baseline / image_gate 的 summary.json。
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "evaluate_segmentation.py"
#: SimpleITK 的 spacing 轴序（sx, sy, sz）。注意数组轴序是 (z, y, x)，二者相反：
#: 这里的 (0.5, 0.5, 3.0) 表示**数组轴 0（z / 层方向）厚 3.0 mm**，
#: 与 Dataset605 3d_fullres 的 plans ``spacing [3.0, 0.5, 0.5]``（nnU-Net 内部数组轴）
#: 经 SimpleITK 读回后的取值一致。体素体积 = 0.5*0.5*3.0 = 0.75 mm³。
SPACING_XYZ = (0.5, 0.5, 3.0)


def _load_module():
    spec = importlib.util.spec_from_file_location("evaluate_segmentation_entry", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- 合成产物
def _case(case_id: str, n_ref: int, n_pred: int, tp: int) -> dict:
    """由 (n_ref, n_pred, TP) 派生一致的 FP/FN/TN/Dice（与 nnU-Net 口径一致）。"""
    fp, fn = n_pred - tp, n_ref - tp
    denom = 2 * tp + fp + fn
    dice = (2 * tp / denom) if denom > 0 else float("nan")
    return {
        "case_id": case_id,
        "n_ref": n_ref,
        "n_pred": n_pred,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": 1000,
        "dice": dice,
    }


def _write_fold(root: Path, name: str, cases: list[dict]) -> Path:
    """把合成病例写成 ``<root>/<name>/validation/summary.json``。"""
    fold = root / name
    validation = fold / "validation"
    validation.mkdir(parents=True, exist_ok=True)
    entries = []
    for c in cases:
        entries.append(
            {
                "metrics": {
                    "1": {
                        "Dice": c["dice"],
                        "FN": c["FN"],
                        "FP": c["FP"],
                        "IoU": float("nan"),
                        "TN": c["TN"],
                        "TP": c["TP"],
                        "n_pred": c["n_pred"],
                        "n_ref": c["n_ref"],
                    }
                },
                "reference_file": str(
                    fold / "gt_segmentations" / f"{c['case_id']}.nii.gz"
                ),
                "prediction_file": str(validation / f"{c['case_id']}.nii.gz"),
            }
        )
    tp = sum(c["TP"] for c in cases)
    fp = sum(c["FP"] for c in cases)
    fn = sum(c["FN"] for c in cases)
    finite_dice = [c["dice"] for c in cases if np.isfinite(c["dice"])]
    doc = {
        "foreground_mean": {
            # 复刻 nnU-Net 的 nanmean 语义（全为 NaN 时给 NaN，并避免 numpy 空切片告警）
            "Dice": float(np.mean(finite_dice)) if finite_dice else float("nan"),
            "IoU": float("nan"),
            "FN": fn,
            "FP": fp,
            "TN": sum(c["TN"] for c in cases),
            "TP": tp,
            "n_pred": sum(c["n_pred"] for c in cases),
            "n_ref": sum(c["n_ref"] for c in cases),
        },
        "mean": {"1": {}},
        "metric_per_case": entries,
    }
    # allow_nan=True 复刻 nnU-Net 写出的 NaN（Python json 可原样读回）
    (validation / "summary.json").write_text(
        json.dumps(doc, allow_nan=True), encoding="utf-8"
    )
    return fold


def _write_nifti(path: Path, arr: np.ndarray, spacing=SPACING_XYZ) -> None:
    import SimpleITK as sitk

    path.parent.mkdir(parents=True, exist_ok=True)
    img = sitk.GetImageFromArray(arr.astype(np.uint8))
    img.SetSpacing(tuple(float(s) for s in spacing))
    sitk.WriteImage(img, str(path))


def _cube(shape=(12, 12, 12), lo=(2, 2, 2), hi=(6, 6, 6)) -> np.ndarray:
    arr = np.zeros(shape, dtype=np.uint8)
    arr[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]] = 1
    return arr


def _mask_counts(ref_arr: np.ndarray, pred_arr: np.ndarray) -> dict:
    """由掩膜得到与 full 模式完全一致的六项计数与 Dice（便于构造自洽的 summary）。"""
    ref_b, pred_b = ref_arr > 0, pred_arr > 0
    tp = int(np.count_nonzero(ref_b & pred_b))
    fp = int(np.count_nonzero(~ref_b & pred_b))
    fn = int(np.count_nonzero(ref_b & ~pred_b))
    tn = int(np.count_nonzero(~ref_b & ~pred_b))
    denom = 2 * tp + fp + fn
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "n_pred": int(np.count_nonzero(pred_b)),
        "n_ref": int(np.count_nonzero(ref_b)),
        "Dice": (2 * tp / denom) if denom > 0 else float("nan"),
    }


def _write_fold_with_masks(
    root: Path,
    name: str,
    specs: list[tuple[str, np.ndarray, np.ndarray]],
    *,
    spacing=SPACING_XYZ,
) -> Path:
    """写出一份带真实（合成）NIfTI 的 fold；六项计数全部由掩膜算出，与 summary 严格自洽。"""
    fold = root / name
    validation = fold / "validation"
    validation.mkdir(parents=True, exist_ok=True)
    entries, dices = [], []
    for case_id, ref_arr, pred_arr in specs:
        ref_path = fold / "gt_segmentations" / f"{case_id}.nii.gz"
        pred_path = validation / f"{case_id}.nii.gz"
        _write_nifti(ref_path, ref_arr, spacing)
        _write_nifti(pred_path, pred_arr, spacing)
        metrics = {**_mask_counts(ref_arr, pred_arr), "IoU": float("nan")}
        dices.append(metrics["Dice"])
        entries.append(
            {
                "metrics": {"1": metrics},
                "reference_file": str(ref_path),
                "prediction_file": str(pred_path),
            }
        )
    finite = [d for d in dices if np.isfinite(d)]
    doc = {
        "foreground_mean": {"Dice": float(np.mean(finite)) if finite else float("nan")},
        "mean": {"1": {}},
        "metric_per_case": entries,
    }
    (validation / "summary.json").write_text(
        json.dumps(doc, allow_nan=True), encoding="utf-8"
    )
    return fold


# --------------------------------------------------------------------------- 1-6 基本病例形态
def test_perfect_positive_case(tmp_path):
    mod = _load_module()
    fold = _write_fold(tmp_path, "m", [_case("a", 100, 100, 100)])
    rep, _ = mod.summarize_model("m", fold, 200, 1)
    assert rep["case_counts"] == {
        "num_cases": 1,
        "num_positive_cases": 1,
        "num_negative_cases": 0,
        "case_ids_positive": ["a"],
        "case_ids_negative": [],
    }
    assert rep["positive_segmentation"]["positive_macro_dice_mean"] == pytest.approx(
        1.0
    )
    assert rep["positive_segmentation"]["positive_micro_dice"] == pytest.approx(1.0)
    assert rep["voxel_metrics"]["positive_voxel_recall"] == pytest.approx(1.0)
    assert rep["voxel_metrics"]["positive_voxel_precision"] == pytest.approx(1.0)
    assert rep["miss_structure"]["positive_missed_cases"] == 0


def test_partial_overlap_case(tmp_path):
    mod = _load_module()
    # n_ref=100, n_pred=60, TP=50 -> Dice = 100/110
    fold = _write_fold(tmp_path, "m", [_case("a", 100, 60, 50)])
    rep, _ = mod.summarize_model("m", fold, 200, 1)
    expected = 2 * 50 / (2 * 50 + 10 + 50)
    assert rep["positive_segmentation"]["positive_macro_dice_mean"] == pytest.approx(
        expected
    )
    assert rep["voxel_metrics"]["positive_voxel_recall"] == pytest.approx(50 / 100)
    assert rep["voxel_metrics"]["positive_voxel_precision"] == pytest.approx(50 / 60)


def test_positive_case_with_empty_prediction_is_counted_as_zero(tmp_path):
    mod = _load_module()
    fold = _write_fold(tmp_path, "m", [_case("a", 100, 0, 0)])
    rep, _ = mod.summarize_model("m", fold, 200, 1)
    assert rep["positive_segmentation"]["positive_macro_dice_mean"] == pytest.approx(
        0.0
    )
    assert rep["miss_structure"]["positive_empty_prediction_cases"] == 1
    assert rep["miss_structure"]["positive_wrong_location_cases"] == 0


def test_positive_case_wrong_location_is_not_empty_prediction(tmp_path):
    mod = _load_module()
    fold = _write_fold(tmp_path, "m", [_case("a", 100, 40, 0)])
    rep, _ = mod.summarize_model("m", fold, 200, 1)
    assert rep["miss_structure"]["positive_wrong_location_cases"] == 1
    assert rep["miss_structure"]["positive_empty_prediction_cases"] == 0
    assert rep["positive_segmentation"]["positive_macro_dice_mean"] == pytest.approx(
        0.0
    )


def test_negative_case_with_empty_prediction(tmp_path):
    mod = _load_module()
    fold = _write_fold(tmp_path, "m", [_case("n", 0, 0, 0)])
    rep, _ = mod.summarize_model("m", fold, 200, 1)
    assert rep["case_counts"]["num_negative_cases"] == 1
    assert rep["negative_false_positive"]["negative_clean_cases"] == 1
    assert rep["negative_false_positive"]["negative_fp_cases"] == 0
    # 真阴病例不得把 Dice 人为记为 1 混入宏平均
    assert rep["positive_segmentation"]["positive_macro_dice_mean"] is None


def test_negative_case_with_false_positive(tmp_path):
    mod = _load_module()
    fold = _write_fold(tmp_path, "m", [_case("n", 0, 30, 0)])
    rep, _ = mod.summarize_model("m", fold, 200, 1)
    nf = rep["negative_false_positive"]
    assert nf["negative_fp_cases"] == 1
    assert nf["negative_fp_case_rate"] == pytest.approx(1.0)
    assert nf["negative_fp_voxels_total"] == 30
    assert nf["negative_fp_voxels_max"] == pytest.approx(30.0)


# --------------------------------------------------------------------------- 7-10 宏/微/精度
def test_macro_dice_includes_missed_case_as_zero(tmp_path):
    mod = _load_module()
    fold = _write_fold(
        tmp_path,
        "m",
        [_case("hit", 100, 100, 100), _case("miss", 200, 0, 0)],  # Dice 1.0 与 0.0
    )
    rep, _ = mod.summarize_model("m", fold, 200, 1)
    assert rep["positive_segmentation"]["positive_macro_dice_mean"] == pytest.approx(
        0.5
    )
    assert rep["positive_segmentation"]["positive_macro_dice_median"] == pytest.approx(
        0.5
    )
    assert rep["miss_structure"]["positive_missed_cases"] == 1


def test_micro_dice_only_aggregates_positive_cases(tmp_path):
    mod = _load_module()
    # 阳性：n_ref=100, n_pred=60, TP=50 -> FP=10, FN=50 -> micro = 100/160
    # 阴性另有 FP=1000，必须完全不影响 micro Dice
    fold = _write_fold(tmp_path, "m", [_case("p", 100, 60, 50), _case("n", 0, 1000, 0)])
    rep, _ = mod.summarize_model("m", fold, 200, 1)
    assert rep["positive_segmentation"]["positive_micro_dice"] == pytest.approx(
        100 / 160
    )
    assert rep["voxel_metrics"]["numerator_denominator"]["fp_negative"] == 1000
    # 与含阴性假阳的 all-prediction precision 口径必须不同
    assert rep["voxel_metrics"]["positive_voxel_precision"] == pytest.approx(50 / 60)
    assert rep["voxel_metrics"]["all_prediction_voxel_precision"] == pytest.approx(
        50 / (60 + 1000)
    )


def test_positive_and_all_prediction_precision_are_distinguished(tmp_path):
    mod = _load_module()
    fold = _write_fold(tmp_path, "m", [_case("p", 100, 60, 50), _case("n", 0, 40, 0)])
    rep, _ = mod.summarize_model("m", fold, 200, 1)
    vm = rep["voxel_metrics"]
    assert vm["positive_voxel_precision"] == pytest.approx(50 / 60)
    assert vm["all_prediction_voxel_precision"] == pytest.approx(50 / 100)
    assert vm["positive_voxel_precision"] != vm["all_prediction_voxel_precision"]
    # 两者都必须显式存在，不得互相替代
    assert "positive_voxel_precision" in vm and "all_prediction_voxel_precision" in vm


def test_negative_fp_case_rate_and_volume_stats(tmp_path):
    mod = _load_module()
    fold = _write_fold(
        tmp_path,
        "m",
        [
            _case("n1", 0, 10, 0),
            _case("n2", 0, 30, 0),
            _case("n3", 0, 0, 0),
            _case("n4", 0, 0, 0),
        ],
    )
    rep, _ = mod.summarize_model("m", fold, 200, 1)
    nf = rep["negative_false_positive"]
    assert nf["negative_fp_cases"] == 2
    assert nf["negative_fp_case_rate"] == pytest.approx(0.5)
    assert nf["negative_clean_cases"] == 2
    assert nf["negative_fp_voxels_total"] == 40
    assert nf["negative_fp_voxels_n"] == 2
    assert nf["negative_fp_voxels_median"] == pytest.approx(20.0)
    assert nf["negative_fp_voxels_max"] == pytest.approx(30.0)
    assert nf["negative_fp_voxels_q1"] == pytest.approx(15.0)
    assert nf["negative_fp_voxels_q3"] == pytest.approx(25.0)


# --------------------------------------------------------------------------- 11-13 可比性与 bootstrap
def test_incomparable_case_sets_must_fail(tmp_path):
    mod = _load_module()
    a = _write_fold(tmp_path, "a", [_case("x", 10, 10, 10)])
    b = _write_fold(tmp_path, "b", [_case("y", 10, 10, 10)])
    rep_a, cases_a = mod.summarize_model("a", a, 100, 1)
    rep_b, cases_b = mod.summarize_model("b", b, 100, 1)
    with pytest.raises(mod.EvaluationError, match="case id 集合不一致"):
        mod._require_comparable({"a": rep_a, "b": rep_b}, {"a": cases_a, "b": cases_b})


def test_mismatched_n_ref_must_fail(tmp_path):
    mod = _load_module()
    a = _write_fold(tmp_path, "a", [_case("x", 10, 10, 10)])
    b = _write_fold(tmp_path, "b", [_case("x", 11, 11, 11)])
    rep_a, cases_a = mod.summarize_model("a", a, 100, 1)
    rep_b, cases_b = mod.summarize_model("b", b, 100, 1)
    with pytest.raises(mod.EvaluationError, match="n_ref 不一致"):
        mod._require_comparable({"a": rep_a, "b": rep_b}, {"a": cases_a, "b": cases_b})


def test_mismatched_reference_identity_must_fail(tmp_path):
    mod = _load_module()
    a = _write_fold(tmp_path, "a", [_case("x", 10, 10, 10)])
    b = _write_fold(tmp_path, "b", [_case("x", 0, 0, 0)])
    rep_a, cases_a = mod.summarize_model("a", a, 100, 1)
    rep_b, cases_b = mod.summarize_model("b", b, 100, 1)
    with pytest.raises(mod.EvaluationError, match="阳性/阴性身份不一致"):
        mod._require_comparable({"a": rep_a, "b": rep_b}, {"a": cases_a, "b": cases_b})


def _varied_case_pairs(n: int = 24) -> tuple[list[dict], list[dict]]:
    """构造 n 对同 case、不同 Dice 的合成病例，使配对差值为连续变化。"""
    cases_a, cases_b = [], []
    for i in range(n):
        n_ref = 100 + 11 * i
        tp_a = (n_ref * (20 + (i % 9) * 7)) // 100
        tp_b = (n_ref * (25 + (i % 7) * 9)) // 100
        cases_a.append(_case(f"c{i:02d}", n_ref, tp_a + 3 + i, tp_a))
        cases_b.append(_case(f"c{i:02d}", n_ref, tp_b + 2 + i, tp_b))
    return cases_a, cases_b


def test_paired_bootstrap_is_reproducible_with_fixed_seed(tmp_path):
    mod = _load_module()
    cases_a, cases_b = _varied_case_pairs()
    fold_a = _write_fold(tmp_path, "a", cases_a)
    fold_b = _write_fold(tmp_path, "b", cases_b)
    rep_a, ca = mod.summarize_model("a", fold_a, 2000, 7)
    rep_b, cb = mod.summarize_model("b", fold_b, 2000, 7)

    def run(seed: int) -> dict:
        return mod.paired_comparisons(
            {"a": rep_a, "b": rep_b}, {"a": ca, "b": cb}, 2000, seed
        )[0]

    first, second = run(12345), run(12345)
    assert first["paired_bootstrap_ci95"] == second["paired_bootstrap_ci95"]
    assert first["paired_bootstrap_ci95"][0] <= first["paired_bootstrap_ci95"][1]

    # 不同 seed 必须给出不同 CI（证明 seed 真的控制重采样，而非被忽略）
    deltas = [r["delta"] for r in first["positive_case_dice_deltas"]]
    assert len(set(deltas)) > 10, "合成数据不足以让 bootstrap 表现出随机性"
    assert run(999)["paired_bootstrap_ci95"] != first["paired_bootstrap_ci95"]

    assert first["delta_definition"] == "model_b - model_a"
    assert first["improved_cases"] + first["tied_cases"] + first[
        "worsened_cases"
    ] == len(cases_a)


def test_paired_comparison_reports_overlap_and_miss_changes(tmp_path):
    mod = _load_module()
    fold_a = _write_fold(
        tmp_path, "a", [_case("fixed", 100, 0, 0), _case("broken", 100, 60, 50)]
    )
    fold_b = _write_fold(
        tmp_path, "b", [_case("fixed", 100, 60, 50), _case("broken", 100, 0, 0)]
    )
    rep_a, ca = mod.summarize_model("a", fold_a, 200, 1)
    rep_b, cb = mod.summarize_model("b", fold_b, 200, 1)
    cmp_ = mod.paired_comparisons({"a": rep_a, "b": rep_b}, {"a": ca, "b": cb}, 200, 1)[
        0
    ]
    assert cmp_["new_overlap_case_ids"] == ["fixed"]
    assert cmp_["lost_overlap_case_ids"] == ["broken"]
    assert cmp_["complete_miss_delta"] == 0
    assert cmp_["mean_paired_dice_delta"] == pytest.approx(0.0)


# --------------------------------------------------------------------------- 14-17 表面/物理指标
def test_identical_masks_give_perfect_surface_metrics():
    mod = _load_module()
    mask = _cube((16, 16, 16)) > 0
    out = mod.surface_metrics(mask, mask, SPACING_XYZ, 1.0)
    assert out["reason"] == "ok"
    assert out["hd95_mm"] == pytest.approx(0.0)
    assert out["assd_mm"] == pytest.approx(0.0)
    assert out["nsd"] == pytest.approx(1.0)


def test_single_voxel_translation_respects_anisotropic_spacing():
    """单体素掩膜平移可给出解析解，用于验证 mm 距离确实尊重各轴 spacing。

    SPACING_XYZ = (0.5, 0.5, 3.0)（SimpleITK 轴序）对应数组轴序 (3.0, 0.5, 0.5)：
    沿数组轴 0（z）平移 3 体素 = 9.0 mm；沿数组轴 1（y）平移 3 体素 = 1.5 mm。
    """
    mod = _load_module()
    ref = np.zeros((16, 16, 16), dtype=bool)
    ref[2, 2, 2] = True
    along_axis0 = np.zeros_like(ref)
    along_axis0[5, 2, 2] = True
    along_axis1 = np.zeros_like(ref)
    along_axis1[2, 5, 2] = True

    out0 = mod.surface_metrics(along_axis0, ref, SPACING_XYZ, 1.0)
    assert out0["hd95_mm"] == pytest.approx(9.0)
    assert out0["assd_mm"] == pytest.approx(9.0)

    out1 = mod.surface_metrics(along_axis1, ref, SPACING_XYZ, 1.0)
    assert out1["hd95_mm"] == pytest.approx(1.5)
    assert out1["assd_mm"] == pytest.approx(1.5)

    # 同一批掩膜在等向 spacing 下必须给出 3.0 mm（证明没有把体素距离当作 mm）
    out_iso = mod.surface_metrics(along_axis0, ref, (1.0, 1.0, 1.0), 1.0)
    assert out_iso["hd95_mm"] == pytest.approx(3.0)
    assert out_iso["hd95_mm"] != pytest.approx(out0["hd95_mm"])


def test_nsd_tolerance_is_used_verbatim_in_mm():
    """NSD@τ 严格按物理 mm 使用给定容差，且各轴使用自己的真实 spacing。

    构造：单个相邻体素对（pred 相对 ref 沿某个数组轴平移 1 体素）。surfel 结构已核实：
    共享角点距离 0，非共享角点距离 = 该轴真实 spacing，两侧 surfel 面积相同，因此
        τ < 该轴 spacing -> 命中一半 surfel（NSD = 0.5）
        τ ≥ 该轴 spacing -> 全部命中（NSD = 1.0）
    """
    mod = _load_module()
    spacing_xyz = (0.5, 1.5, 4.0)  # SimpleITK (sx, sy, sz)
    spacing_zyx = mod._spacing_zyx(spacing_xyz)  # 数组轴序 (z, y, x)
    assert spacing_zyx == (4.0, 1.5, 0.5)

    def translated_pair(axis: int):
        ref = np.zeros((12, 12, 12), dtype=bool)
        ref[4, 4, 4] = True
        pred = ref.copy()
        pred[4, 4, 4] = False
        pos = [4, 4, 4]
        pos[axis] += 1
        pred[tuple(pos)] = True
        return pred, ref

    # 数组轴 0（z，spacing 4.0 mm）
    pred_z, ref = translated_pair(0)
    assert mod.nsd_surface_area(pred_z, ref, spacing_zyx, 3.0) == pytest.approx(0.5)
    assert mod.nsd_surface_area(pred_z, ref, spacing_zyx, 4.5) == pytest.approx(1.0)
    # 数组轴 1（y，spacing 1.5 mm）
    pred_y, _ = translated_pair(1)
    assert mod.nsd_surface_area(pred_y, ref, spacing_zyx, 1.0) == pytest.approx(0.5)
    assert mod.nsd_surface_area(pred_y, ref, spacing_zyx, 2.0) == pytest.approx(1.0)
    # 数组轴 2（x，spacing 0.5 mm）
    pred_x, _ = translated_pair(2)
    assert mod.nsd_surface_area(pred_x, ref, spacing_zyx, 0.4) == pytest.approx(0.5)
    assert mod.nsd_surface_area(pred_x, ref, spacing_zyx, 0.6) == pytest.approx(1.0)

    # 同一对掩膜在等向 spacing 下距离变成 1.0 mm：若实现忽略 spacing（把体素步长当 mm）
    # 或错轴，上面的 0.5 / 1.0 组合不可能同时成立
    assert mod.nsd_surface_area(pred_z, ref, (1.0, 1.0, 1.0), 3.0) == pytest.approx(1.0)
    assert mod.nsd_surface_area(pred_z, ref, (1.0, 1.0, 1.0), 0.5) == pytest.approx(0.5)
    # surface_metrics 走 SimpleITK 轴序（自动反转）；与直接调用（数组轴序）必须一致
    assert mod.surface_metrics(pred_x, ref, spacing_xyz, 0.6)["nsd"] == pytest.approx(
        1.0
    )


def test_nsd_surface_area_weights_by_surfel_area_not_voxel_count():
    """非对称构造必须区分 surfel-area weighting 与 surface-voxel counting。

    构造：5×5×1 平板（pred 与 ref 完全重合）+ 远离的单个孤立体素（沿 x 平移 3 体素 = 1.5 mm
    > τ）。同一批表面元素有两种口径：按物理面积（surfel 面积）加权，或每个表面元素计 1。
    两种口径必须给出不同结果；新实现必须等于前者且不等于后者（不是旧算法的换名字）。
    """
    mod = _load_module()
    from surface_distance import metrics as surface_distance_metrics

    spacing_zyx = (3.0, 0.5, 0.5)  # SPACING_XYZ 的数组轴序
    tolerance_mm = 0.6
    ref = np.zeros((24, 24, 24), dtype=bool)
    ref[4, 4:9, 4:9] = True  # 5×5×1 平板
    ref[20, 20, 20] = True  # 远离的单个孤立体素
    pred = ref.copy()
    pred[20, 20, 20] = False
    pred[20, 20, 23] = True  # 沿 x 平移 3 体素 = 1.5 mm > τ

    # 用官方包提取原始 surfel 表示，在测试内独立构造两种口径的期望值（不调用被测函数）
    surface_distances = surface_distance_metrics.compute_surface_distances(
        mask_gt=ref, mask_pred=pred, spacing_mm=spacing_zyx
    )
    d_gt = surface_distances["distances_gt_to_pred"]
    d_pred = surface_distances["distances_pred_to_gt"]
    a_gt = surface_distances["surfel_areas_gt"]
    a_pred = surface_distances["surfel_areas_pred"]
    area_weighted = (
        np.sum(a_gt[d_gt <= tolerance_mm]) + np.sum(a_pred[d_pred <= tolerance_mm])
    ) / (np.sum(a_gt) + np.sum(a_pred))
    count_weighted = (
        np.count_nonzero(d_gt <= tolerance_mm)
        + np.count_nonzero(d_pred <= tolerance_mm)
    ) / (d_gt.size + d_pred.size)

    # 两种口径在这组非对称表面上必须可区分（否则本测试无法证明 weighting 生效）
    assert abs(float(area_weighted) - float(count_weighted)) > 0.01

    got = mod.nsd_surface_area(pred, ref, spacing_zyx, tolerance_mm)
    assert got == pytest.approx(float(area_weighted), abs=1e-9)  # surfel 面积加权
    assert got != pytest.approx(float(count_weighted), abs=0.01)  # 不是表面体素计数


def test_missing_surface_distance_fails_closed(tmp_path, monkeypatch):
    """缺 surface-distance 时 NSD 必须 fail-closed，不得静默退回旧算法。"""
    mod = _load_module()
    monkeypatch.setitem(sys.modules, "surface_distance", None)
    monkeypatch.setitem(sys.modules, "surface_distance.metrics", None)
    with pytest.raises(mod.EvaluationError, match="surface-distance"):
        mod._surface_distance_metrics()

    # full 模式必须在读取任何 NIfTI 之前直接失败，而不是逐例收集上千条导入错误
    ref = _cube((16, 16, 16))
    fold = _write_fold_with_masks(tmp_path, "m", [("a", ref, ref)])
    with pytest.raises(mod.EvaluationError, match="surface-distance"):
        mod.main(
            ["--model", f"m={fold}", "--mode", "full", "--nsd-tolerance-mm", "1.0"]
        )


def test_full_report_nsd_method_is_surfel_area_based(tmp_path):
    """full 报告必须声明 surfel-area weighted，且不再出现旧的 surface-voxel based。"""
    mod = _load_module()
    ref = _cube((16, 16, 16))
    fold = _write_fold_with_masks(tmp_path, "m", [("a", ref, ref)])
    rep, cases = mod.summarize_model("m", fold, 100, 1)
    mod.run_full_mode({"m": rep}, {"m": cases}, 1.0, None, progress=False)
    method = rep["surface_metrics"]["nsd_method"]
    assert "surfel-area weighted" in method
    assert "surface-distance" in method
    assert "surface-voxel" not in method
    assert rep["surface_metrics"]["nsd_tolerance_mm"] == pytest.approx(1.0)


def test_empty_prediction_surface_handling():
    mod = _load_module()
    ref = _cube((16, 16, 16)) > 0
    empty = np.zeros_like(ref, dtype=bool)

    missed = mod.surface_metrics(empty, ref, SPACING_XYZ, 1.0)
    assert missed["reason"] == "prediction_empty"
    assert missed["nsd"] == pytest.approx(0.0)
    assert missed["hd95_mm"] is None and missed["assd_mm"] is None

    both_empty = mod.surface_metrics(empty, empty, SPACING_XYZ, 1.0)
    assert both_empty["applicable"] is False
    assert both_empty["nsd"] is None and both_empty["hd95_mm"] is None


def test_anisotropic_spacing_gives_correct_mm3_volume(tmp_path):
    mod = _load_module()
    ref = _cube((16, 16, 16), (2, 2, 2), (6, 6, 6))
    n_ref = int(ref.sum())
    fold = _write_fold_with_masks(tmp_path, "m", [("a", ref, ref)])
    rep, cases = mod.summarize_model("m", fold, 100, 1)
    mod.run_full_mode({"m": rep}, {"m": cases}, 1.0, None, progress=False)
    expect_vox_mm3 = 3.0 * 0.5 * 0.5
    assert rep["volume_metrics_mm3"]["reference_volume_mm3"]["median"] == pytest.approx(
        n_ref * expect_vox_mm3
    )
    row = next(r for r in rep["per_case"] if r["case_id"] == "a")
    assert row["voxel_volume_mm3"] == pytest.approx(expect_vox_mm3)
    assert row["reference_volume_mm3"] == pytest.approx(n_ref * expect_vox_mm3)
    assert row["rve"] == pytest.approx(0.0)
    assert row["arve"] == pytest.approx(0.0)


def test_size_strata_and_threshold_validation(tmp_path):
    mod = _load_module()
    small = _cube((16, 16, 16), (2, 2, 2), (4, 4, 4))  # 8 体素 = 6 mm³
    big = _cube((16, 16, 16), (4, 4, 4), (12, 12, 12))  # 512 体素 = 384 mm³
    fold = _write_fold_with_masks(
        tmp_path, "m", [("small", small, small), ("big", big, big)]
    )
    rep, cases = mod.summarize_model("m", fold, 100, 1)
    mod.run_full_mode({"m": rep}, {"m": cases}, 1.0, [100.0], progress=False)
    assert rep["size_strata"]["volume_thresholds_mm3"] == [100.0]
    assert [s["n"] for s in rep["size_strata"]["strata"]] == [1, 1]
    assert "单病灶" in rep["size_strata"]["burden_note"]

    # 参数校验：乱序 / 重复 / 非正 / 非数值 都必须 fail
    assert mod.parse_thresholds("100,200") == [100.0, 200.0]
    assert mod.parse_thresholds(None) is None
    for bad in ("200,100", "100,100", "0", "-5", "abc", "100,,200"):
        with pytest.raises(mod.EvaluationError):
            mod.parse_thresholds(bad)


def test_full_mode_geometry_mismatch_is_fail_closed(tmp_path):
    mod = _load_module()
    ref = _cube((16, 16, 16))
    fold = _write_fold_with_masks(tmp_path, "m", [("a", ref, ref)])
    # 把 prediction 改成不同 spacing
    _write_nifti(fold / "validation" / "a.nii.gz", ref, spacing=(2.0, 0.5, 0.5))
    rep, cases = mod.summarize_model("m", fold, 100, 1)
    with pytest.raises(mod.EvaluationError, match="spacing 不一致"):
        mod.run_full_mode({"m": rep}, {"m": cases}, 1.0, None, progress=False)


def test_full_mode_rejects_non_binary_labels(tmp_path):
    mod = _load_module()
    ref = _cube((16, 16, 16))
    fold = _write_fold_with_masks(tmp_path, "m", [("a", ref, ref)])
    weird = ref.copy()
    weird[0, 0, 0] = 7
    _write_nifti(fold / "validation" / "a.nii.gz", weird)
    rep, cases = mod.summarize_model("m", fold, 100, 1)
    with pytest.raises(mod.EvaluationError, match="含非 0/1 标签值"):
        mod.run_full_mode({"m": rep}, {"m": cases}, 1.0, None, progress=False)


def test_full_mode_rejects_mask_count_mismatch_with_summary(tmp_path):
    """掩膜必须与 summary.json 的体素计数一致，防止评估对象与计数来源不一致。"""
    mod = _load_module()
    ref = _cube((16, 16, 16))  # 64 体素
    fold = _write_fold_with_masks(tmp_path, "m", [("a", ref, ref)])
    # 用另一种合法 0/1 掩膜（125 体素）覆盖 prediction
    _write_nifti(
        fold / "validation" / "a.nii.gz", _cube((16, 16, 16), (0, 0, 0), (5, 5, 5))
    )
    rep, cases = mod.summarize_model("m", fold, 100, 1)
    with pytest.raises(mod.EvaluationError, match="掩膜与 summary.json 计数不一致"):
        mod.run_full_mode({"m": rep}, {"m": cases}, 1.0, None, progress=False)


def test_component_analysis_counts_components():
    mod = _load_module()
    ref = np.zeros((16, 16, 16), dtype=bool)
    ref[1, 1, 1] = True
    ref[9, 9, 9] = True
    pred = np.zeros_like(ref)
    pred[1, 1, 1] = True
    out = mod.component_analysis(pred, ref, 0.75, "a")
    assert out["connectivity"] == 1
    assert out["ref_component_count"] == 2
    assert out["pred_component_count"] == 1
    assert out["ref_components_without_prediction_overlap"] == 1
    assert out["pred_component_max_volume_mm3"] == pytest.approx(0.75)


# --------------------------------------------------------------------------- 19-22 CLI 与输出
def test_bootstrap_resamples_must_be_positive(tmp_path):
    mod = _load_module()
    fold = _write_fold(tmp_path, "m", [_case("a", 10, 10, 10)])
    with pytest.raises(mod.EvaluationError, match="必须为正整数"):
        mod.main(["--model", f"m={fold}", "--bootstrap-resamples", "0"])


def test_full_mode_requires_explicit_nsd_tolerance(tmp_path):
    mod = _load_module()
    fold = _write_fold(tmp_path, "m", [_case("a", 10, 10, 10)])
    with pytest.raises(mod.EvaluationError, match="nsd-tolerance-mm"):
        mod.main(["--model", f"m={fold}", "--mode", "full"])


def test_duplicate_model_names_rejected(tmp_path):
    mod = _load_module()
    fold = _write_fold(tmp_path, "m", [_case("a", 10, 10, 10)])
    with pytest.raises(mod.EvaluationError, match="模型名重复"):
        mod.parse_models([f"m={fold}", f"m={fold}"])


def test_output_is_published_once_and_never_overwritten(tmp_path, capsys):
    mod = _load_module()
    fold = _write_fold(tmp_path, "m", [_case("a", 100, 100, 100)])
    out = tmp_path / "metrics.json"
    assert mod.main(["--model", f"m={fold}", "--output", str(out)]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema_version"] == mod.SCHEMA_VERSION
    assert payload["mode"] == "summary"
    assert payload["models"]["m"]["positive_segmentation"][
        "positive_macro_dice_mean"
    ] == (pytest.approx(1.0))
    assert "paired_comparisons" in payload
    assert payload["metrics_scope"].startswith("lesion segmentation only")

    # 已存在时必须拒绝覆盖，且原内容不变
    before = out.read_text(encoding="utf-8")
    with pytest.raises(mod.EvaluationError, match="拒绝覆盖"):
        mod.main(["--model", f"m={fold}", "--output", str(out)])
    assert out.read_text(encoding="utf-8") == before
    capsys.readouterr()


def test_json_output_is_standard_and_contains_no_nan(tmp_path):
    mod = _load_module()
    # 含真阴病例（Dice = NaN）与完全漏分病例，强制触发 NaN 与 null 路径
    fold = _write_fold(tmp_path, "m", [_case("p", 10, 0, 0), _case("n", 0, 0, 0)])
    out = tmp_path / "metrics.json"
    assert mod.main(["--model", f"m={fold}", "--output", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text
    payload = json.loads(text)
    row = next(r for r in payload["models"]["m"]["per_case"] if r["case_id"] == "n")
    assert row["dice"] is None  # 真阴病例 Dice 无定义 -> null，而不是 1 或 NaN


def test_failure_publishes_no_partial_json_and_no_temp_leftover(tmp_path):
    mod = _load_module()
    ref = _cube((16, 16, 16))
    fold = _write_fold_with_masks(tmp_path, "m", [("a", ref, ref)])
    _write_nifti(fold / "validation" / "a.nii.gz", ref, spacing=(2.0, 0.5, 0.5))
    out = tmp_path / "metrics.json"
    with pytest.raises(mod.EvaluationError):
        mod.main(
            [
                "--model",
                f"m={fold}",
                "--mode",
                "full",
                "--nsd-tolerance-mm",
                "1.0",
                "--output",
                str(out),
            ]
        )
    assert not out.exists()
    assert list(tmp_path.glob(".*tmp")) == []


def test_cli_help_and_failure_exit_codes(tmp_path):
    cmds = [
        (0, [sys.executable, str(SCRIPT), "--help"]),
        (2, [sys.executable, str(SCRIPT)]),
        (2, [sys.executable, str(SCRIPT), "--model", "bad-spec"]),
        (
            2,
            [
                sys.executable,
                str(SCRIPT),
                "--model",
                "m=/nonexistent/fold",
                "--mode",
                "full",
            ],
        ),
    ]
    for expected, cmd in cmds:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        assert proc.returncode == expected, (cmd, proc.stderr[-400:])
        if expected == 0:
            assert "--model NAME=FOLD_DIR" in proc.stdout
    # 输出已存在时 CLI 必须非零退出
    mod = _load_module()
    fold = _write_fold(tmp_path, "m", [_case("a", 10, 10, 10)])
    out = tmp_path / "metrics.json"
    out.write_text("{}", encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--model", f"m={fold}", "--output", str(out)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "拒绝覆盖" in proc.stderr
    assert mod is not None


def test_module_import_pulls_neither_torch_nor_nnunet():
    """评估入口必须保持轻量：不得导入 torch / nnunetv2（也就不可能使用 GPU 或推理）。"""
    code = (
        "import importlib.util, sys;"
        f"spec = importlib.util.spec_from_file_location('ev', r'{SCRIPT}');"
        "mod = importlib.util.module_from_spec(spec);"
        "spec.loader.exec_module(mod);"
        "bad = sorted({k.split('.')[0] for k in sys.modules "
        "if k.split('.')[0] in ('torch', 'nnunetv2', 'medpy', 'SimpleITK', 'scipy')});"
        "print(','.join(bad))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr[-400:]
    # 重量级依赖只在 full 模式按需导入；模块级导入不得拉起它们
    assert proc.stdout.strip() == ""


# ------------------------------------------------------- 修补轮：summary 计数严格校验
def _patch_metrics(fold: Path, case_id: str, **updates) -> None:
    """就地改写某例 metrics（用于构造不一致的 summary.json）。"""
    path = fold / "validation" / "summary.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    for entry in doc["metric_per_case"]:
        stem = entry["reference_file"].split("/")[-1].removesuffix(".nii.gz")
        if stem == case_id:
            entry["metrics"]["1"].update(updates)
    path.write_text(json.dumps(doc, allow_nan=True), encoding="utf-8")


@pytest.mark.parametrize(
    ("updates", "expected"),
    [
        ({"n_pred": 999}, r"TP\+FP"),
        ({"n_ref": 999}, r"TP\+FN"),
        ({"TP": 5.5}, "必须是整数"),
        ({"FP": -1}, "不能为负"),
        ({"n_ref": float("nan")}, "不是有限数"),
        ({"n_ref": float("inf")}, "不是有限数"),
        ({"Dice": 0.9}, "不一致"),
    ],
)
def test_inconsistent_summary_metrics_must_fail(tmp_path, updates, expected):
    """summary 中的计数/ Dice 不允许不自洽：小数、负数、NaN、恒等式违反、Dice 不符都要失败。"""
    mod = _load_module()
    fold = _write_fold(
        tmp_path, "m", [_case("a", 100, 60, 50)]
    )  # Dice = 100/160 = 0.625
    _patch_metrics(fold, "a", **updates)
    with pytest.raises(mod.EvaluationError, match=expected):
        mod.summarize_model("m", fold, 100, 1)


def test_summary_dice_is_recomputed_not_trusted(tmp_path):
    """宏平均必须使用由 TP/FP/FN 重算的 Dice，而不是 summary 里写的值。"""
    mod = _load_module()
    fold = _write_fold(tmp_path, "m", [_case("a", 100, 60, 50)])  # 重算 Dice = 0.625
    rep, _ = mod.summarize_model("m", fold, 100, 1)
    assert rep["positive_segmentation"]["positive_macro_dice_mean"] == pytest.approx(
        100 / 160
    )


def test_truenegative_with_finite_dice_must_fail(tmp_path):
    """真阴（空-空）病例的 Dice 必须为空/NaN；给出有限值即视为被污染。"""
    mod = _load_module()
    fold = _write_fold(tmp_path, "m", [_case("n", 0, 0, 0)])
    _patch_metrics(fold, "n", Dice=1.0)
    with pytest.raises(mod.EvaluationError, match="真阴病例"):
        mod.summarize_model("m", fold, 100, 1)


def test_positive_case_without_dice_must_fail(tmp_path):
    """阳性病例必须能重算 Dice；summary 若用 NaN 顶替应失败。"""
    mod = _load_module()
    fold = _write_fold(tmp_path, "m", [_case("a", 100, 60, 50)])
    _patch_metrics(fold, "a", Dice=float("nan"))
    with pytest.raises(mod.EvaluationError, match="未给出 Dice"):
        mod.summarize_model("m", fold, 100, 1)


# ------------------------------------------------------- 修补轮：full 模式覆盖全部病例
def _empty(shape=(16, 16, 16)) -> np.ndarray:
    return np.zeros(shape, dtype=np.uint8)


def test_negative_case_spacing_mismatch_must_fail(tmp_path):
    mod = _load_module()
    pred = _cube((16, 16, 16))
    fold = _write_fold_with_masks(tmp_path, "m", [("n", _empty(), pred)])
    _write_nifti(fold / "validation" / "n.nii.gz", pred, spacing=(2.0, 0.5, 3.0))
    rep, cases = mod.summarize_model("m", fold, 100, 1)
    with pytest.raises(mod.EvaluationError, match="spacing 不一致"):
        mod.run_full_mode({"m": rep}, {"m": cases}, 1.0, None, progress=False)


def test_true_negative_missing_file_must_fail(tmp_path):
    """真阴病例（n_pred=0）同样必须存在文件，不得跳过读取。"""
    mod = _load_module()
    fold = _write_fold_with_masks(tmp_path, "m", [("tn", _empty(), _empty())])
    (fold / "validation" / "tn.nii.gz").unlink()
    rep, cases = mod.summarize_model("m", fold, 100, 1)
    with pytest.raises(mod.EvaluationError, match="prediction 不存在"):
        mod.run_full_mode({"m": rep}, {"m": cases}, 1.0, None, progress=False)


def test_true_negative_missing_reference_must_fail(tmp_path):
    mod = _load_module()
    fold = _write_fold_with_masks(tmp_path, "m", [("tn", _empty(), _empty())])
    (fold / "gt_segmentations" / "tn.nii.gz").unlink()
    rep, cases = mod.summarize_model("m", fold, 100, 1)
    with pytest.raises(mod.EvaluationError, match="reference 不存在"):
        mod.run_full_mode({"m": rep}, {"m": cases}, 1.0, None, progress=False)


def test_negative_case_non_binary_prediction_must_fail(tmp_path):
    mod = _load_module()
    pred = _cube((16, 16, 16))
    fold = _write_fold_with_masks(tmp_path, "m", [("n", _empty(), pred)])
    weird = pred.copy()
    weird[0, 0, 0] = 3
    _write_nifti(fold / "validation" / "n.nii.gz", weird)
    rep, cases = mod.summarize_model("m", fold, 100, 1)
    with pytest.raises(mod.EvaluationError, match="含非 0/1 标签值"):
        mod.run_full_mode({"m": rep}, {"m": cases}, 1.0, None, progress=False)


def test_negative_case_with_nonempty_reference_must_fail(tmp_path):
    """summary 声明 n_ref=0，但 reference 掩膜实际非空时必须失败。"""
    mod = _load_module()
    fold = _write_fold_with_masks(tmp_path, "m", [("n", _empty(), _empty())])
    _write_nifti(fold / "gt_segmentations" / "n.nii.gz", _cube((16, 16, 16)))
    rep, cases = mod.summarize_model("m", fold, 100, 1)
    with pytest.raises(mod.EvaluationError, match="reference 掩膜非空"):
        mod.run_full_mode({"m": rep}, {"m": cases}, 1.0, None, progress=False)


def test_negative_case_prediction_count_mismatch_must_fail(tmp_path):
    """阴性病例的 n_pred / FP 与掩膜不一致时必须失败。"""
    mod = _load_module()
    pred = _cube((16, 16, 16))  # 64 体素
    fold = _write_fold_with_masks(tmp_path, "m", [("n", _empty(), pred)])
    _write_nifti(
        fold / "validation" / "n.nii.gz", _cube((16, 16, 16), (0, 0, 0), (5, 5, 5))
    )
    rep, cases = mod.summarize_model("m", fold, 100, 1)
    with pytest.raises(mod.EvaluationError, match="掩膜与 summary.json 计数不一致"):
        mod.run_full_mode({"m": rep}, {"m": cases}, 1.0, None, progress=False)


def test_full_mode_passes_for_mixed_case_types(tmp_path):
    """阳性 + 假阳阴性 + 真阴 的混合病例集必须完整通过，且真阴也被读取校验。"""
    mod = _load_module()
    ref_pos = _cube((16, 16, 16), (2, 2, 2), (6, 6, 6))
    pred_pos = _cube((16, 16, 16), (2, 2, 2), (5, 5, 5))
    fold = _write_fold_with_masks(
        tmp_path,
        "m",
        [
            ("pos", ref_pos, pred_pos),
            ("fpneg", _empty(), _cube((16, 16, 16), (9, 9, 9), (11, 11, 11))),
            ("tn", _empty(), _empty()),
        ],
    )
    rep, cases = mod.summarize_model("m", fold, 200, 1)
    mod.run_full_mode({"m": rep}, {"m": cases}, 1.0, [50.0], progress=False)

    mv = rep["mask_verification"]
    assert mv["n_cases_in_summary"] == 3
    assert mv["n_cases_read_and_verified"] == 3
    assert mv["all_cases_verified"] is True

    tn_row = next(r for r in rep["per_case"] if r["case_id"] == "tn")
    assert tn_row["mask_verified"] is True
    assert tn_row["reference_volume_mm3"] == pytest.approx(0.0)
    assert "hd95_mm" not in tn_row  # 真阴不参与正表面统计

    fp_row = next(r for r in rep["per_case"] if r["case_id"] == "fpneg")
    assert fp_row["fp_component_count"] == 1
    assert fp_row["reference_volume_mm3"] == pytest.approx(0.0)

    neg = rep["component_analysis"]["negative_case_fp_components"]
    assert neg["fp_component_count_total"] == 1
    assert neg["n_negative_cases_with_prediction"] == 1

    pos_row = next(r for r in rep["per_case"] if r["case_id"] == "pos")
    # pred_pos 比 ref_pos 小一圈（27 vs 64 体素），故 RVE 为负、HD95 为正
    assert pos_row["rve"] == pytest.approx((27 - 64) / 64)
    assert pos_row["arve"] == pytest.approx((64 - 27) / 64)
    assert pos_row["hd95_mm"] is not None and pos_row["hd95_mm"] > 0
    assert pos_row["nsd"] is not None
    assert rep["size_strata"]["volume_thresholds_mm3"] == [50.0]


def test_full_mode_reads_each_case_exactly_once(tmp_path, monkeypatch):
    """full 模式对**每个病例**只读一次，不重复读取（含真阴与假阳阴性）。"""
    mod = _load_module()
    fold = _write_fold_with_masks(
        tmp_path,
        "m",
        [
            ("p", _cube((16, 16, 16)), _cube((16, 16, 16))),
            ("fpneg", _empty(), _cube((16, 16, 16), (9, 9, 9), (11, 11, 11))),
            ("tn1", _empty(), _empty()),
            ("tn2", _empty(), _empty()),
        ],
    )
    rep, cases = mod.summarize_model("m", fold, 100, 1)

    calls: list[tuple[str, str]] = []
    original = mod.load_mask_pair

    def counting(pred_path: str, ref_path: str):
        calls.append((pred_path, ref_path))
        return original(pred_path, ref_path)

    monkeypatch.setattr(mod, "load_mask_pair", counting)
    mod.run_full_mode({"m": rep}, {"m": cases}, 1.0, None, progress=False)

    assert len(calls) == 4  # 4 例各一次
    assert len(set(calls)) == 4  # 没有任何重复读取
    assert rep["mask_verification"]["all_cases_verified"] is True


@pytest.mark.parametrize("bad", ["nan", "inf", "-inf", "0", "-1.0"])
def test_invalid_nsd_tolerance_must_fail(tmp_path, bad):
    """nsd tolerance 必须是正的有限数：nan / inf / -inf / 非正 一律非零退出。

    用 ``=`` 形式传入，以走本工具的参数校验（空格形式下 argparse 会先把 ``-1.0`` 当作选项）。
    """
    mod = _load_module()
    fold = _write_fold(tmp_path, "m", [_case("a", 10, 10, 10)])
    with pytest.raises(mod.EvaluationError, match="nsd-tolerance-mm"):
        mod.main(
            ["--model", f"m={fold}", "--mode", "full", f"--nsd-tolerance-mm={bad}"]
        )


def test_space_separated_negative_option_also_exits_nonzero(tmp_path):
    """``--nsd-tolerance-mm -1.0``（空格形式）被 argparse 视为选项，CLI 仍必须非零退出。"""
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--model",
            f"m={tmp_path / 'x'}",
            "--mode",
            "full",
            "--nsd-tolerance-mm",
            "-1.0",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0


def test_invalid_nsd_tolerance_rejected_in_summary_mode(tmp_path):
    mod = _load_module()
    fold = _write_fold(tmp_path, "m", [_case("a", 10, 10, 10)])
    with pytest.raises(mod.EvaluationError, match="nsd-tolerance-mm"):
        mod.main(["--model", f"m={fold}", "--nsd-tolerance-mm", "inf"])


def test_multi_model_load_errors_are_reported_together(tmp_path):
    """多个模型同时缺产物时必须一次性汇总，而不是只报第一个。"""
    mod = _load_module()
    good = _write_fold(tmp_path, "good", [_case("a", 10, 10, 10)])
    with pytest.raises(mod.EvaluationError) as excinfo:
        mod.main(
            [
                "--model",
                f"a={tmp_path / 'missing_a'}",
                "--model",
                f"b={tmp_path / 'missing_b'}",
                "--model",
                f"good={good}",
            ]
        )
    message = str(excinfo.value)
    assert "共 2 项" in message
    assert "missing_a" in message and "missing_b" in message


# --------------------------------------- 最小修补轮：逐例异常收集范围与结构化结束汇总
def _identical_case_fold(tmp_path, ids=("a", "b", "c")):
    """构造若干"完美分割"的阳性病例，使流程能走到 surface / component 阶段。"""
    ref = _cube((16, 16, 16))
    return _write_fold_with_masks(tmp_path, "m", [(cid, ref, ref) for cid in ids])


_SUMMARY_KEYS = (
    "status",
    "mode",
    "phase",
    "unit",
    "ok",
    "failed",
    "skipped",
    "elapsed",
    "output",
)


def _summary_blocks(text: str) -> list[str]:
    """按分隔符切出全部"评估结束汇总"块，用于断言"有且只有一份"。"""
    return text.split("=== 评估结束汇总 ===")[1:]


def _parse_summary(block: str) -> dict[str, str]:
    """把汇总块解析成 {字段: 值}；只取已知字段，忽略其余行。"""
    fields: dict[str, str] = {}
    for line in block.splitlines():
        head, sep, tail = line.partition(":")
        if sep and head.strip() in _SUMMARY_KEYS:
            fields[head.strip()] = tail.strip()
    return fields


def test_full_mode_collects_surface_failures_across_all_cases(
    tmp_path, monkeypatch, capsys
):
    """surface_metrics 逐例抛异常时，所有病例都要被执行，错误一次性汇总且写明总数。"""
    mod = _load_module()
    fold = _identical_case_fold(tmp_path, ids=("a", "b", "c"))
    rep, cases = mod.summarize_model("m", fold, 100, 1)

    calls: list[str] = []

    def boom(pred, ref_arr, spacing, tolerance_mm):
        calls.append("surface")
        raise RuntimeError("medpy exploded")

    monkeypatch.setattr(mod, "surface_metrics", boom)
    with pytest.raises(mod.EvaluationError) as excinfo:
        mod.run_full_mode({"m": rep}, {"m": cases}, 1.0, None, progress=False)

    message = str(excinfo.value)
    assert "共 3 项" in message  # 正确总数
    for cid in ("a", "b", "c"):
        assert f"[m] {cid}:" in message  # 三个 case id 同时出现
    assert "RuntimeError" in message  # 异常类型
    assert "medpy exploded" in message  # 异常信息
    assert len(calls) == 3  # 第一例失败后没有中止，其余病例仍被执行
    capsys.readouterr()


def test_full_mode_collects_component_failures_across_all_cases(
    tmp_path, monkeypatch, capsys
):
    """component_analysis 逐例抛异常时同样按病例收集，不中止其余病例。"""
    mod = _load_module()
    fold = _identical_case_fold(tmp_path, ids=("a", "b"))
    rep, cases = mod.summarize_model("m", fold, 100, 1)

    def boom(pred, ref_arr, voxel_volume_mm3, case_id):
        raise ValueError("scipy label failed")

    monkeypatch.setattr(mod, "component_analysis", boom)
    with pytest.raises(mod.EvaluationError) as excinfo:
        mod.run_full_mode({"m": rep}, {"m": cases}, 1.0, None, progress=False)

    message = str(excinfo.value)
    assert "共 2 项" in message
    assert "[m] a:" in message and "[m] b:" in message
    assert "ValueError" in message and "scipy label failed" in message
    capsys.readouterr()


def test_full_mode_failure_leaves_no_partial_or_fake_results(tmp_path, monkeypatch):
    """失败病例不得留下伪造的 mask_verified，任何聚合结果也不得写入 report。"""
    mod = _load_module()
    fold = _identical_case_fold(tmp_path, ids=("a", "b"))
    rep, cases = mod.summarize_model("m", fold, 100, 1)

    real_component = mod.component_analysis

    def boom_for_a(pred, ref_arr, voxel_volume_mm3, case_id):
        if case_id == "a":
            raise RuntimeError("volume/component step failed")
        return real_component(pred, ref_arr, voxel_volume_mm3, case_id)

    monkeypatch.setattr(mod, "component_analysis", boom_for_a)
    with pytest.raises(mod.EvaluationError) as excinfo:
        mod.run_full_mode({"m": rep}, {"m": cases}, 1.0, None, progress=False)

    message = str(excinfo.value)
    assert "[m] a:" in message
    assert "[m] b:" not in message  # b 成功，不应出现在错误汇总中

    # report 必须保持"完全未被写入"的状态：既没有 mask_verification 聚合块，
    # 也没有任何逐例 mask_verified 标记
    assert "mask_verification" not in rep
    assert rep["surface_metrics"] is None
    assert rep["component_analysis"] is None
    assert all("mask_verified" not in row for row in rep["per_case"])


def test_full_mode_metric_failure_publishes_nothing(tmp_path, monkeypatch):
    """指定 --output 时，逐例指标失败不得产生 JSON，也不得留下临时文件。"""
    mod = _load_module()
    fold = _identical_case_fold(tmp_path, ids=("a", "b"))
    out = tmp_path / "segmentation_metrics.json"

    def boom(pred, ref_arr, spacing, tolerance_mm):
        raise RuntimeError("metric backend failure")

    monkeypatch.setattr(mod, "surface_metrics", boom)
    with pytest.raises(mod.EvaluationError):
        mod.main(
            [
                "--model",
                f"m={fold}",
                "--mode",
                "full",
                "--nsd-tolerance-mm",
                "1.0",
                "--output",
                str(out),
            ]
        )
    assert not out.exists()
    assert list(tmp_path.glob(".*tmp")) == []


def test_full_mode_does_not_swallow_keyboard_interrupt(tmp_path, monkeypatch):
    """KeyboardInterrupt / SystemExit 不属于 Exception，必须原样向上抛出。"""
    mod = _load_module()
    fold = _identical_case_fold(tmp_path, ids=("a", "b"))
    rep, cases = mod.summarize_model("m", fold, 100, 1)

    def interrupt(pred_path, ref_path):
        raise KeyboardInterrupt

    monkeypatch.setattr(mod, "load_mask_pair", interrupt)
    with pytest.raises(KeyboardInterrupt):
        mod.run_full_mode({"m": rep}, {"m": cases}, 1.0, None, progress=False)


def test_full_mode_run_stats_count_model_case_pairs(tmp_path, monkeypatch):
    """full 逐例阶段：ok / failed 以 model-case 对为单位，且不影响 fail-closed 行为。"""
    mod = _load_module()
    fold = _identical_case_fold(tmp_path, ids=("a", "b", "c"))
    rep, cases = mod.summarize_model("m", fold, 100, 1)

    # 成功路径：3 例全部通过
    ok_stats = mod.RunStats(mode="full", output=None)
    mod.run_full_mode({"m": rep}, {"m": cases}, 1.0, None, progress=False, run=ok_stats)
    assert (ok_stats.ok, ok_stats.failed, ok_stats.skipped) == (3, 0, 0)
    assert ok_stats.phase == mod.PHASE_FULL_CASES
    assert ok_stats.unit == mod.UNIT_MODEL_CASE

    # 失败路径：2 例失败、1 例成功；计数不改变"整体仍然 fail-closed"这一事实
    real_component = mod.component_analysis

    def boom_for_two(pred, ref_arr, voxel_volume_mm3, case_id):
        if case_id in ("a", "b"):
            raise RuntimeError("boom")
        return real_component(pred, ref_arr, voxel_volume_mm3, case_id)

    monkeypatch.setattr(mod, "component_analysis", boom_for_two)
    bad_stats = mod.RunStats(mode="full", output=None)
    with pytest.raises(mod.EvaluationError):
        mod.run_full_mode(
            {"m": rep}, {"m": cases}, 1.0, None, progress=False, run=bad_stats
        )
    assert (bad_stats.ok, bad_stats.failed, bad_stats.skipped) == (1, 2, 0)
    assert bad_stats.phase == mod.PHASE_FULL_CASES
    assert bad_stats.unit == mod.UNIT_MODEL_CASE


def test_run_stats_fields_units_and_elapsed():
    """结束汇总必须包含 phase / unit / ok / failed / skipped / elapsed / output。"""
    mod = _load_module()
    stats = mod.RunStats(mode="summary", output=None)

    # preflight：尚未开始计数，unit 与三个计数都必须是 not_applicable，不得伪造成 0
    assert stats.phase == mod.PHASE_PREFLIGHT
    assert stats.unit is None
    assert (stats.ok, stats.failed, stats.skipped) == (None, None, None)
    preflight = stats.render("failed")
    for field in (
        "status",
        "mode",
        "phase",
        "unit",
        "ok",
        "failed",
        "skipped",
        "elapsed",
        "output",
    ):
        assert field in preflight
    assert "phase   : preflight" in preflight
    assert preflight.count(mod.NOT_APPLICABLE) == 4  # unit + ok + failed + skipped
    assert stats.elapsed_s >= 0.0

    # model_loading：unit=model，ok/failed 表示模型数，skipped 不适用
    stats.begin_model_loading()
    stats.model_loaded()
    stats.model_loaded()
    stats.model_failed()
    loading = stats.render("failed")
    assert "phase   : model_loading" in loading
    assert f"unit    : {mod.UNIT_MODEL}" in loading
    assert "ok      : 2" in loading
    assert "failed  : 1" in loading
    assert f"skipped : {mod.NOT_APPLICABLE}" in loading

    # full_cases：unit=model-case，三个计数共享同一单位且都是真实数字
    full = mod.RunStats(mode="full", output="out.json")
    full.begin_full_cases(total_pairs=5)
    full.full_case_ok()
    full.full_case_failed()
    full.end_full_cases()
    full_text = full.render("succeeded")
    assert "phase   : full_cases" in full_text
    assert f"unit    : {mod.UNIT_MODEL_CASE}" in full_text
    assert "ok      : 1" in full_text
    assert "failed  : 1" in full_text
    assert "skipped : 3" in full_text
    assert "out.json" in full_text
    assert mod.NOT_APPLICABLE not in full_text

    # 未请求写出时必须明示，而不是留空
    no_output_stats = mod.RunStats(mode="full", output=None)
    assert "(未请求写出 JSON)" in no_output_stats.render("succeeded")


def test_run_summary_is_printed_on_success_and_failure(tmp_path, capsys):
    """成功与失败两条路径都打印**恰好一份**结构化结束汇总（进度条不能替代它）。"""
    mod = _load_module()
    good = _write_fold(tmp_path, "m", [_case("a", 100, 100, 100)])

    assert mod.main(["--model", f"m={good}"]) == 0
    out = capsys.readouterr().out
    assert len(_summary_blocks(out)) == 1
    fields = _parse_summary(_summary_blocks(out)[0])
    assert fields["status"] == "succeeded"
    assert fields["mode"] == "summary"
    assert fields["phase"] == mod.PHASE_COMPLETED
    assert fields["unit"] == mod.UNIT_MODEL  # summary 模式永远是 model
    assert fields["ok"] == "1"
    assert fields["failed"] == "0"
    # summary 模式不读 NIfTI，不存在 model-case 对：必须 not_applicable，不是 0
    assert fields["skipped"] == mod.NOT_APPLICABLE
    assert fields["elapsed"].endswith("s")

    with pytest.raises(mod.EvaluationError):
        mod.main(["--model", f"m={tmp_path / 'missing'}"])
    err = capsys.readouterr().err
    assert len(_summary_blocks(err)) == 1
    fields = _parse_summary(_summary_blocks(err)[0])
    assert fields["status"] == "failed"
    assert fields["phase"] == mod.PHASE_MODEL_LOADING
    assert fields["unit"] == mod.UNIT_MODEL
    assert fields["failed"] == "1"


# ------------------------------------------- 最小修补轮：phase / unit 与普通异常汇总
def test_full_with_two_missing_folds_reports_model_loading_phase(tmp_path, capsys):
    """full 在模型加载阶段失败：phase=model_loading、unit=model、failed=2。"""
    mod = _load_module()
    with pytest.raises(mod.EvaluationError):
        mod.main(
            [
                "--model",
                f"a={tmp_path / 'missing_a'}",
                "--model",
                f"b={tmp_path / 'missing_b'}",
                "--mode",
                "full",
                "--nsd-tolerance-mm",
                "1.0",
            ]
        )
    err = capsys.readouterr().err
    assert len(_summary_blocks(err)) == 1
    fields = _parse_summary(_summary_blocks(err)[0])
    assert fields["status"] == "failed"
    assert fields["phase"] == mod.PHASE_MODEL_LOADING
    assert fields["unit"] == mod.UNIT_MODEL
    assert fields["failed"] == "2"
    assert fields["ok"] == "0"
    assert fields["skipped"] == mod.NOT_APPLICABLE


def test_full_partial_model_load_failure_keeps_model_unit(tmp_path, capsys):
    """一个成功加载、一个加载失败：不得把"1 个模型"标成 1 个 model-case 对。"""
    mod = _load_module()
    good = _write_fold(tmp_path, "good", [_case("a", 10, 10, 10)])
    with pytest.raises(mod.EvaluationError):
        mod.main(
            [
                "--model",
                f"good={good}",
                "--model",
                f"bad={tmp_path / 'missing'}",
                "--mode",
                "full",
                "--nsd-tolerance-mm",
                "1.0",
            ]
        )
    fields = _parse_summary(_summary_blocks(capsys.readouterr().err)[0])
    assert fields["phase"] == mod.PHASE_MODEL_LOADING
    assert fields["unit"] == mod.UNIT_MODEL
    assert fields["unit"] != mod.UNIT_MODEL_CASE
    assert fields["ok"] == "1"
    assert fields["failed"] == "1"


def test_comparability_failure_reports_comparability_phase(tmp_path, capsys):
    """可比性检查失败：phase=comparability，unit 仍为 model，不得改标为 model-case。"""
    mod = _load_module()
    a = _write_fold(tmp_path, "a", [_case("x", 10, 10, 10)])
    b = _write_fold(tmp_path, "b", [_case("y", 10, 10, 10)])
    with pytest.raises(mod.EvaluationError, match="case id 集合不一致"):
        mod.main(["--model", f"a={a}", "--model", f"b={b}"])
    fields = _parse_summary(_summary_blocks(capsys.readouterr().err)[0])
    assert fields["status"] == "failed"
    assert fields["phase"] == mod.PHASE_COMPARABILITY
    assert fields["unit"] == mod.UNIT_MODEL
    assert fields["ok"] == "2"  # 两个模型都加载成功，语义是"模型数"
    assert fields["failed"] == "0"
    assert fields["skipped"] == mod.NOT_APPLICABLE


def test_full_success_summary_keeps_model_case_unit(tmp_path, capsys):
    """正常跑完 full 病例循环：单位必须是 model-case，且计数是 model-case 对。"""
    mod = _load_module()
    fold = _identical_case_fold(tmp_path, ids=("a", "b"))
    assert (
        mod.main(
            [
                "--model",
                f"m={fold}",
                "--mode",
                "full",
                "--nsd-tolerance-mm",
                "0.5",
                "--no-progress",
            ]
        )
        == 0
    )
    fields = _parse_summary(_summary_blocks(capsys.readouterr().out)[0])
    assert fields["status"] == "succeeded"
    assert fields["phase"] == mod.PHASE_COMPLETED
    # 单位必须从上一步延续下来，而不是回落到 model
    assert fields["unit"] == mod.UNIT_MODEL_CASE
    assert fields["ok"] == "2"
    assert fields["failed"] == "0"
    assert fields["skipped"] == "0"


def test_full_cases_phase_is_entered_before_the_loop(tmp_path, monkeypatch):
    """进入逐例循环时立刻切到 phase=full_cases / unit=model-case，并已重置计数。"""
    mod = _load_module()
    fold = _identical_case_fold(tmp_path, ids=("a", "b"))
    rep, cases = mod.summarize_model("m", fold, 100, 1)
    stats = mod.RunStats(mode="full", output=None)
    assert stats.phase == mod.PHASE_PREFLIGHT

    real_surface = mod.surface_metrics
    calls: list[int] = []

    def boom(pred, ref_arr, spacing, tolerance_mm):
        # 在第一例内部检查：此时应已进入 full_cases 且单位已是 model-case
        assert stats.phase == mod.PHASE_FULL_CASES
        assert stats.unit == mod.UNIT_MODEL_CASE
        calls.append(1)
        if len(calls) == 1:
            assert (stats.ok, stats.failed) == (0, 0)
            raise RuntimeError("stop at first case")
        return real_surface(pred, ref_arr, spacing, tolerance_mm)

    monkeypatch.setattr(mod, "surface_metrics", boom)
    with pytest.raises(mod.EvaluationError):
        mod.run_full_mode(
            {"m": rep}, {"m": cases}, 1.0, None, progress=False, run=stats
        )
    assert stats.phase == mod.PHASE_FULL_CASES
    assert stats.unit == mod.UNIT_MODEL_CASE
    assert (stats.ok, stats.failed, stats.skipped) == (1, 1, 0)


def test_publish_failure_prints_one_summary_and_reraises(tmp_path, monkeypatch, capsys):
    """普通 Exception（非 EvaluationError）也要打印汇总、不吞掉异常、且只有一份。"""
    mod = _load_module()
    fold = _write_fold(tmp_path, "m", [_case("a", 10, 10, 10)])
    out = tmp_path / "metrics.json"

    def boom(path, payload):
        raise OSError("disk full")

    monkeypatch.setattr(mod, "_publish_json", boom)
    with pytest.raises(OSError, match="disk full"):
        mod.main(["--model", f"m={fold}", "--output", str(out)])

    err = capsys.readouterr().err
    assert len(_summary_blocks(err)) == 1  # 有且只有一份失败汇总，不重复打印
    fields = _parse_summary(_summary_blocks(err)[0])
    assert fields["status"] == "failed"
    assert fields["phase"] == mod.PHASE_PUBLISHING
    assert fields["unit"] == mod.UNIT_MODEL  # summary 模式
    assert fields["ok"] == "1"
    assert fields["failed"] == "0"
    # 异常没有被吞掉（pytest.raises 已确认），且没有落盘任何东西
    assert not out.exists()
    assert list(tmp_path.glob(".*tmp")) == []
