"""Anatomy-Guided ROI 的纯合成单元测试（不读取真实医学数据、不需要 GPU）。

覆盖 §33 要求的三项：

- **ROI physical margin**：margin 必须以 mm 定义，在各向异性 spacing 下每轴 voxel 数不同，
  且实际物理边界**不小于**承诺（向上取整）；
- **ROI restore geometry**：crop -> restore 必须逐值还原，ROI 外填指定值；
- **empty WG prediction**：空 WG 预测**不得**导致病灶被裁掉，默认回退全视野并记录原因；
  另外覆盖 small WG prediction（ROI 小于 patch size 时仍保持物理 margin 语义）。
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from zonal_reliability_fusion.lesion.roi import (
    ProstateROI,
    ROIError,
    build_prostate_roi,
    expand_box_by_physical_margin,
    roi_retains_reference_lesions,
)

#: Dataset605/606 的 3d_fullres 数组轴序 spacing（Z, Y, X）
SPACING_ZYX = (3.0, 0.5, 0.5)


def _wg_with_box(shape, lower, upper):
    wg = np.zeros(shape, dtype=np.float64)
    wg[lower[0] : upper[0], lower[1] : upper[1], lower[2] : upper[2]] = 1.0
    return wg


# --------------------------------------------------------------------------- margin
def test_physical_margin_uses_per_axis_spacing():
    """同一 mm margin 在不同 spacing 下的 voxel 扩张必须不同（不是固定 voxel 常数）。"""
    lower, upper = (4, 8, 8), (6, 12, 12)
    new_lower, new_upper = expand_box_by_physical_margin(
        lower, upper, SPACING_ZYX, margin_mm=9.0
    )
    # 9 mm / 3.0 mm = 3 voxel（Z）；9 mm / 0.5 mm = 18 voxel（Y, X）
    assert new_lower == (1, -10, -10)
    assert new_upper == (9, 30, 30)


def test_margin_rounds_up_so_physical_border_is_never_tighter():
    """9 mm 的 margin 在 2 mm spacing 上取 ceil(4.5)=5 voxel（物理 10 mm >= 9 mm）。"""
    lower, upper = (5, 5, 5), (7, 7, 7)
    new_lower, _ = expand_box_by_physical_margin(lower, upper, (2.0, 2.0, 2.0), 9.0)
    assert 5 - new_lower[0] == 5
    assert (5 - new_lower[0]) * 2.0 >= 9.0


def test_zero_margin_is_identity():
    lower, upper = (2, 3, 4), (5, 6, 7)
    assert expand_box_by_physical_margin(lower, upper, SPACING_ZYX, 0.0) == (lower, upper)


def test_negative_margin_and_bad_spacing_fail_closed():
    with pytest.raises(ROIError, match="margin_mm"):
        expand_box_by_physical_margin((1, 1, 1), (2, 2, 2), SPACING_ZYX, -1.0)
    with pytest.raises(ROIError, match="spacing"):
        expand_box_by_physical_margin((1, 1, 1), (2, 2, 2), (1.0, 0.0, 1.0), 5.0)


def test_roi_clips_to_volume_and_reports_margin_voxels():
    wg = _wg_with_box((12, 16, 16), (4, 6, 6), (8, 10, 10))
    roi = build_prostate_roi(wg, SPACING_ZYX, margin_mm=9.0, wg_threshold=0.5)
    assert roi.lower == (1, 0, 0)
    assert roi.upper == (11, 16, 16)
    assert roi.margin_voxels == (3, 18, 18)
    assert roi.spacing_zyx == SPACING_ZYX
    assert roi.is_full_fov is False
    assert roi.fallback_reason is None


# --------------------------------------------------------------------------- geometry
def test_crop_restore_is_exact_and_fills_outside():
    rng = np.random.default_rng(0)
    volume = rng.random((12, 16, 16))
    wg = _wg_with_box((12, 16, 16), (4, 6, 6), (8, 10, 10))
    roi = build_prostate_roi(wg, SPACING_ZYX, margin_mm=6.0, wg_threshold=0.5)

    cropped = roi.crop(volume)
    assert cropped.shape == roi.shape
    restored = roi.restore(cropped, fill=-1.0)
    assert restored.shape == volume.shape
    inside = tuple(slice(roi.lower[i], roi.upper[i]) for i in range(3))
    assert np.array_equal(restored[inside], volume[inside])
    outside = np.ones(volume.shape, dtype=bool)
    outside[inside] = False
    assert np.all(restored[outside] == -1.0)


def test_restore_rejects_wrong_shape_instead_of_misplacing_prediction():
    wg = _wg_with_box((12, 16, 16), (4, 6, 6), (8, 10, 10))
    roi = build_prostate_roi(wg, SPACING_ZYX, margin_mm=6.0, wg_threshold=0.5)
    with pytest.raises(ROIError, match="ROI 形状"):
        roi.restore(np.zeros((3, 3, 3)))


def test_crop_rejects_wrong_original_shape_instead_of_resampling():
    wg = _wg_with_box((12, 16, 16), (4, 6, 6), (8, 10, 10))
    roi = build_prostate_roi(wg, SPACING_ZYX, margin_mm=6.0, wg_threshold=0.5)
    with pytest.raises(ROIError, match="原空间形状"):
        roi.crop(np.zeros((12, 16, 15)))


def test_pad_plan_never_resamples_and_never_crops_the_roi():
    wg = _wg_with_box((12, 16, 16), (4, 6, 6), (8, 10, 10))
    roi = build_prostate_roi(wg, SPACING_ZYX, margin_mm=6.0, wg_threshold=0.5)
    plan = roi.pad_plan((16, 16, 16))
    assert plan["overhang"] == (0, 0, 0)
    assert plan["resampling"].startswith("none")
    # ROI 大于目标尺寸时只报告 overhang（= 超出量），不擅自裁剪（裁剪可能切掉病灶）
    small_plan = roi.pad_plan((2, 2, 2))
    assert small_plan["pad_before"] == (0, 0, 0)
    assert small_plan["pad_after"] == (0, 0, 0)
    assert small_plan["overhang"] == tuple(size - 2 for size in roi.shape)


def test_to_json_is_json_serializable_and_records_no_resampling():
    import json

    wg = _wg_with_box((12, 16, 16), (4, 6, 6), (8, 10, 10))
    roi = build_prostate_roi(wg, SPACING_ZYX, margin_mm=6.0, wg_threshold=0.5)
    payload = json.loads(json.dumps(roi.to_json()))
    assert payload["resampling"] == "none"
    assert payload["margin_mm"] == 6.0
    assert payload["prior_source"] == "predicted_prior"
    assert payload["roi_shape_zyx"] == list(roi.shape)


# --------------------------------------------------------------------------- empty/small WG
def test_empty_wg_prediction_falls_back_to_full_fov_and_never_deletes_lesion():
    """空 WG 预测**不得**产生空 ROI（那会把病灶裁掉），必须回退全视野并记录原因。"""
    empty = np.zeros((12, 16, 16))
    roi = build_prostate_roi(empty, SPACING_ZYX, margin_mm=9.0, wg_threshold=0.5)
    assert roi.is_full_fov is True
    assert roi.fallback_reason == "empty_wg_prediction"
    assert roi.lower == (0, 0, 0)
    assert roi.upper == (12, 16, 16)


def test_empty_wg_prediction_can_be_strictly_rejected():
    empty = np.zeros((12, 16, 16))
    with pytest.raises(ROIError, match="empty WG prediction"):
        build_prostate_roi(
            empty, SPACING_ZYX, margin_mm=9.0, wg_threshold=0.5, empty_fallback="error"
        )


def test_small_wg_prediction_still_keeps_requested_physical_margin():
    """小 WG（3x3x3 voxel）也必须得到完整 margin，而不是被 clip 成退化区间。"""
    wg = _wg_with_box((20, 40, 40), (10, 20, 20), (11, 21, 21))
    roi = build_prostate_roi(wg, SPACING_ZYX, margin_mm=6.0, wg_threshold=0.5)
    # Z: ceil(6/3)=2 voxel -> [8, 13)；Y/X: ceil(6/0.5)=12 voxel -> [8, 33)
    assert roi.lower == (8, 8, 8)
    assert roi.upper == (13, 33, 33)
    # 每轴实际物理 margin 不小于承诺的 6 mm（box 上界 Z=11、Y=X=21）
    box_lower, box_upper = (10, 20, 20), (11, 21, 21)
    for axis, spacing in enumerate(SPACING_ZYX):
        assert (box_lower[axis] - roi.lower[axis]) * spacing >= 6.0
        assert (roi.upper[axis] - box_upper[axis]) * spacing >= 6.0


def test_wg_covering_whole_volume_is_recorded_as_fallback():
    full = np.ones((8, 8, 8))
    roi = build_prostate_roi(full, SPACING_ZYX, margin_mm=3.0, wg_threshold=0.5)
    assert roi.is_full_fov is True
    assert roi.fallback_reason == "wg_covers_full_fov"


def test_high_threshold_can_empty_the_mask_and_uses_same_safe_fallback():
    wg = np.full((8, 8, 8), 0.4)
    roi = build_prostate_roi(wg, SPACING_ZYX, margin_mm=3.0, wg_threshold=0.5)
    assert roi.fallback_reason == "empty_wg_prediction"


# --------------------------------------------------------------------------- validation
def test_roi_input_validation_is_fail_closed():
    with pytest.raises(ROIError, match="3D"):
        build_prostate_roi(np.zeros((8, 8)), SPACING_ZYX, margin_mm=3.0, wg_threshold=0.5)
    with pytest.raises(ROIError, match="有限"):
        bad = np.zeros((8, 8, 8))
        bad[0, 0, 0] = np.nan
        build_prostate_roi(bad, SPACING_ZYX, margin_mm=3.0, wg_threshold=0.5)
    with pytest.raises(ROIError, match=r"\[0, 1\]"):
        build_prostate_roi(
            np.full((8, 8, 8), 2.0), SPACING_ZYX, margin_mm=3.0, wg_threshold=0.5
        )
    with pytest.raises(ROIError, match="wg_threshold"):
        build_prostate_roi(
            np.ones((8, 8, 8)), SPACING_ZYX, margin_mm=3.0, wg_threshold=1.5
        )
    with pytest.raises(ROIError, match="empty_fallback"):
        build_prostate_roi(
            np.ones((8, 8, 8)), SPACING_ZYX, margin_mm=3.0, wg_threshold=0.5,
            empty_fallback="delete_everything",
        )


def test_roi_retains_reference_lesions_reports_structural_loss():
    """ROI 裁掉 GT 时**必须**被显式报告：这是结构性上限降低，不是模型漏检。"""
    wg = _wg_with_box((20, 20, 20), (4, 4, 4), (8, 8, 8))
    lesion = np.zeros((20, 20, 20), dtype=bool)
    lesion[5, 5, 5] = True  # 腺体内
    lesion[18, 18, 18] = True  # 腺体外远处（会被很小的 ROI 裁掉）
    roi = build_prostate_roi(wg, SPACING_ZYX, margin_mm=0.0, wg_threshold=0.5)
    report = roi_retains_reference_lesions(roi, lesion)
    assert report["reference_voxels"] == 2
    assert report["retained_voxels"] == 1
    assert report["fully_retained"] is False
    assert report["retained_voxel_fraction"] == pytest.approx(0.5)


def test_roi_retains_reference_lesions_is_exact_when_full_fov():
    empty = np.zeros((10, 10, 10))
    lesion = np.zeros((10, 10, 10), dtype=bool)
    lesion[3, 3, 3] = True
    roi = build_prostate_roi(empty, SPACING_ZYX, margin_mm=5.0, wg_threshold=0.5)
    report = roi_retains_reference_lesions(roi, lesion)
    assert report["fully_retained"] is True
    assert report["retained_voxel_fraction"] == pytest.approx(1.0)


def test_roi_is_immutable_dataclass():
    wg = _wg_with_box((10, 10, 10), (3, 3, 3), (5, 5, 5))
    roi = build_prostate_roi(wg, SPACING_ZYX, margin_mm=3.0, wg_threshold=0.5)
    assert isinstance(roi, ProstateROI)
    with pytest.raises(dataclasses.FrozenInstanceError):
        roi.lower = (0, 0, 0)  # type: ignore[misc]
