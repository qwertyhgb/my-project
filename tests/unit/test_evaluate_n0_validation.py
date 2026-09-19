"""`scripts/evaluate/evaluate_n0_validation.py` 的合成测试。

只使用合成掩膜与合成概率图：**不读真实医学影像、不使用 GPU、不训练、不推理**。
夹具目录建在项目内（`outputs/metrics/_pytest_eval_*`），因为评测输出目录必须位于项目内；
测试结束会清理。
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import uuid
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_PATH = PROJECT_ROOT / "scripts/evaluate/evaluate_n0_validation.py"


@pytest.fixture(scope="module")
def ev():
    spec = importlib.util.spec_from_file_location("evaluate_n0_validation_under_test", EVAL_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _write_mask(path: Path, array: np.ndarray, spacing=(1.0, 1.0, 3.0)) -> None:
    image = sitk.GetImageFromArray(np.asarray(array, dtype=np.uint8))
    image.SetSpacing(list(spacing))
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(path))


def _block(size=(8, 16, 16), z=(2, 5), y=(4, 9), x=(4, 9)) -> np.ndarray:
    arr = np.zeros(size, dtype=np.uint8)
    arr[z[0]:z[1], y[0]:y[1], x[0]:x[1]] = 1
    return arr


@pytest.fixture()
def workspace():
    """项目内临时夹具目录（评测输出目录必须在项目内，因此不用 tmp_path）。"""
    root = PROJECT_ROOT / "outputs/metrics" / f"_pytest_eval_{uuid.uuid4().hex[:8]}"
    (root / "results/fold_0/validation").mkdir(parents=True, exist_ok=True)
    (root / "gt").mkdir(parents=True, exist_ok=True)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _write_npz(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        probabilities=np.stack([1 - mask, mask]).astype(np.float32),
    )


def _build_case_set(root: Path, *, with_probabilities: bool) -> dict[str, Path]:
    """A=完美、B=部分重叠、C=GT 空但预测非空（FP 判据）、D=完全缺失。"""
    gt = _block()
    shifted = _block(x=(6, 11))
    empty = np.zeros_like(gt)
    validation = root / "results/fold_0/validation"

    _write_mask(root / "gt/A.nii.gz", gt)
    _write_mask(validation / "A.nii.gz", gt)
    _write_mask(root / "gt/B.nii.gz", gt)
    _write_mask(validation / "B.nii.gz", shifted)
    _write_mask(root / "gt/C.nii.gz", empty)
    _write_mask(validation / "C.nii.gz", shifted)
    if with_probabilities:
        _write_npz(validation / "A.npz", gt)
        _write_npz(validation / "B.npz", shifted)
        _write_npz(validation / "C.npz", shifted)

    (root / "splits.json").write_text(
        json.dumps([{"train": ["X"], "val": ["A", "B", "C", "D"]}]), encoding="utf-8"
    )
    (root / "labels.csv").write_text(
        "case_id,case_csPCa\nA,YES\nB,YES\nC,NO\nD,NO\n", encoding="utf-8"
    )
    return {
        "results": root / "results/fold_0",
        "splits": root / "splits.json",
        "gt": root / "gt",
        "labels": root / "labels.csv",
        "out": root / "out",
    }


def _run(ev, paths: dict[str, Path], *extra: str) -> int:
    return ev.main(
        [
            "--results-dir", str(paths["results"]),
            "--splits", str(paths["splits"]),
            "--gt-dir", str(paths["gt"]),
            "--labels", str(paths["labels"]),
            "--out-dir", str(paths["out"]),
            *extra,
        ]
    )


def _metrics(paths: dict[str, Path]) -> dict:
    return json.loads((paths["out"] / "metrics.json").read_text(encoding="utf-8"))["metrics"]


def test_dice_is_exact_for_perfect_and_partial_cases(ev, workspace):
    paths = _build_case_set(workspace, with_probabilities=False)
    assert _run(ev, paths) == 1  # D 缺失 → 不完整
    metrics = _metrics(paths)
    # A：完全重合 → 1.0；B：TP=45 FP=30 FN=30 → 2*45/150 = 0.6
    assert metrics["dice"]["n_gt_positive"] == 2
    assert metrics["dice"]["mean_gt_positive"] == pytest.approx(0.8)
    assert metrics["detection"]["detection_rate_gt_positive"] == pytest.approx(1.0)


def test_predicted_positive_definition_catches_false_positive(ev, workspace):
    """关键判据：GT 为空但预测非空必须计为假阳性（不能用"与 GT 重叠"定义）。"""
    paths = _build_case_set(workspace, with_probabilities=False)
    _run(ev, paths)
    patient = _metrics(paths)["patient_level"]
    assert patient["tp"] == 2
    assert patient["fp"] == 1
    assert patient["fn"] == 0
    assert patient["tn"] == 0
    assert patient["specificity"] == pytest.approx(0.0)
    assert patient["ppv"] == pytest.approx(2 / 3)
    # 重叠口径下 C 只能算"未检出"，两个定义必须分开输出
    assert _metrics(paths)["detection"]["n_detected"] == 2
    assert _metrics(paths)["detection"]["n_predicted_positive"] == 3


def test_missing_case_is_fail_closed_and_listed(ev, workspace):
    paths = _build_case_set(workspace, with_probabilities=False)
    code = _run(ev, paths)
    metrics = _metrics(paths)
    assert code == 1
    assert metrics["complete"] is False
    assert metrics["n_expected"] == 4
    assert metrics["n_evaluated"] == 3
    assert [item["case_id"] for item in metrics["missing"]] == ["D"]


def test_score_metrics_require_probability_maps(ev, workspace):
    without = _build_case_set(workspace, with_probabilities=False)
    _run(ev, without)
    unavailable = _metrics(without)["score_based"]
    assert unavailable["available"] is False
    assert "概率图" in unavailable["reason"]
    assert _metrics(without)["probability_maps_available"] is False


def test_score_metrics_computed_when_probabilities_present(ev, workspace):
    paths = _build_case_set(workspace, with_probabilities=True)
    _run(ev, paths)
    metrics = _metrics(paths)
    score = metrics["score_based"]
    assert metrics["probability_maps_available"] is True
    assert score["available"] is True
    assert score["n"] == 3
    assert 0.0 <= score["average_precision"] <= 1.0
    assert 0.0 <= score["roc_auc"] <= 1.0
    assert metrics["official_metrics_computed"] is False


def test_shape_mismatch_is_refused(ev, workspace):
    paths = _build_case_set(workspace, with_probabilities=False)
    _write_mask(workspace / "results/fold_0/validation/B.nii.gz", _block(size=(8, 16, 12)))
    with pytest.raises(SystemExit) as excinfo:
        _run(ev, paths)
    assert "尺寸不一致" in str(excinfo.value)


def test_spacing_mismatch_is_refused(ev, workspace):
    paths = _build_case_set(workspace, with_probabilities=False)
    _write_mask(
        workspace / "results/fold_0/validation/B.nii.gz",
        _block(x=(6, 11)),
        spacing=(2.0, 1.0, 3.0),
    )
    with pytest.raises(SystemExit) as excinfo:
        _run(ev, paths)
    assert "spacing 不一致" in str(excinfo.value)


def test_limit_option_and_outputs_written(ev, workspace):
    paths = _build_case_set(workspace, with_probabilities=False)
    _run(ev, paths, "--limit", "3")
    assert _metrics(paths)["n_expected"] == 3
    assert (paths["out"] / "per_case.csv").is_file()
    assert (paths["out"] / "summary.md").is_file()
    header = (paths["out"] / "per_case.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "predicted_positive" in header
    assert "score" in header
