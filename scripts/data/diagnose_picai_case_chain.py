#!/usr/bin/env python3
"""Read-only PI-CAI case-chain diagnostics; real-image runs belong to the user.

Reads original MRI/canonical GT, materialized MRI/GT, Dataset605 links,
existing nnU-Net preprocessed arrays and final predictions. No preprocessing,
inference, registration, repair or input writes. Virtual resampling is used
only for diagnostic comparisons/plots, never exported as training data.
Outputs are exclusively created in a NEW outputs/diagnostics subdirectory.
Use --self-test for in-memory synthetic checks (no medical data or file writes).
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from tqdm import tqdm

PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_CASES = "10842_1000858,10121_1000121,11283_1001306"
DEFAULT_RUN = PROJECT / (
    "outputs/nnUNet_results/Dataset605_PICAI/"
    "nnUNetTrainerPICAI_FLCE_PositiveSampling_NoFFT_Seed20261008"
    "__nnUNetPlans__3d_fullres/fold_0"
)
CHANNELS = ("t2w", "adc", "hbv")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def geometry(image):
    return {"size_xyz": list(image.GetSize()), "spacing_xyz_mm": list(image.GetSpacing()),
            "origin_lps_mm": list(image.GetOrigin()), "direction": list(image.GetDirection())}


def same_grid(a, b):
    return (a.GetSize() == b.GetSize() and all(
        np.allclose(x, y, rtol=0, atol=1e-4) for x, y in (
            (a.GetSpacing(), b.GetSpacing()), (a.GetOrigin(), b.GetOrigin()),
            (a.GetDirection(), b.GetDirection()))))


def image_like(array, reference):
    image = sitk.GetImageFromArray(array)
    image.CopyInformation(reference)
    return image


def aligned(image, reference, label=False):
    # Identity physical transform: does NOT estimate/correct anatomical registration.
    return sitk.Resample(image, reference, sitk.Transform(3, sitk.sitkIdentity),
                         sitk.sitkNearestNeighbor if label else sitk.sitkLinear,
                         0.0, sitk.sitkUInt8 if label else sitk.sitkFloat32)


def historical_mri_conversion(image, reference):
    """Reproduce archived materializer, including ITK's input-type output cast.

    A float reference image does not make ResampleImageFilter output float.
    Without SetOutputPixelType, uint16 inputs yield uint16 interpolation output
    BEFORE NumPy's rounding. Keep the float virtual image as a separate diagnostic.
    """
    if same_grid(image, reference):
        array = sitk.GetArrayFromImage(image)
    else:
        sampler = sitk.ResampleImageFilter()
        sampler.SetReferenceImage(reference)
        sampler.SetInterpolator(sitk.sitkLinear)
        sampler.SetDefaultPixelValue(0)
        array = sitk.GetArrayFromImage(sampler.Execute(image)).astype(np.float32)
    require(np.isfinite(array).all(), "Nonfinite historical MRI conversion")
    return np.clip(np.rint(array), 0, 65535).astype(np.uint16)


def binary_label(array, source):
    values = set(np.unique(array).tolist())
    allowed = {0, 1} if source == "human_expert/Pooch25" else {0, 2, 3, 4, 5}
    require(source in {"human_expert/Pooch25", "human_expert/resampled"},
            f"Unsupported canonical source: {source}")
    require(values <= allowed, f"Illegal canonical labels {values} for {source}")
    return (array > 0).astype(np.uint8)


def stats(array, mask=None):
    values = np.asarray(array if mask is None else array[mask]).reshape(-1)
    finite = np.isfinite(values)
    result = {"n": int(values.size), "nonfinite": int((~finite).sum())}
    values = values[finite]
    if values.size:
        result.update(zip(("min", "p01", "p50", "p99", "max"),
                          map(float, np.percentile(values, [0, 1, 50, 99, 100]))))
        result.update(mean=float(values.mean()), std=float(values.std()),
                      zero_fraction=float(np.mean(values == 0)))
    return result


def difference(a, b):
    require(a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}")
    delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return {"equal": bool(np.array_equal(a, b)), "different_voxels": int(np.count_nonzero(delta)),
            "max_abs_difference": float(delta.max()), "mean_abs_difference": float(delta.mean())}


def lesion_stats(mask, spacing_zyx):
    mask = np.asarray(mask, dtype=bool)
    labels, count = ndimage.label(mask, structure=ndimage.generate_binary_structure(3, 1))
    sizes = np.bincount(labels.ravel())[1:]
    indices = np.where(mask)
    return {"voxels": int(mask.sum()), "volume_ml": float(mask.sum() * np.prod(spacing_zyx) / 1000),
            "occupied_z_slices": int(np.count_nonzero(mask.any(axis=(1, 2)))),
            "bbox_zyx_inclusive": [[int(v.min()), int(v.max())] for v in indices] if mask.any() else None,
            "centroid_zyx": [float(v.mean()) for v in indices] if mask.any() else None,
            "components_6_connected": int(count), "component_voxels_desc": sorted(map(int, sizes), reverse=True)}


def dice(a, b):
    denominator = int(a.sum() + b.sum())
    return float(2 * np.count_nonzero(a & b) / denominator) if denominator else None


def load_preprocessed(folder, case):
    # Open existing nnU-Net outputs directly in read-only mode, without unpacking/caches.
    import blosc2

    if (folder / f"{case}.b2nd").exists():
        blosc2.set_nthreads(1)
        data = np.asarray(blosc2.open(str(folder / f"{case}.b2nd"), mode="r")[:])
        seg = np.asarray(blosc2.open(str(folder / f"{case}_seg.b2nd"), mode="r")[:])
    elif (folder / f"{case}.npy").exists() and (folder / f"{case}_seg.npy").exists():
        data = np.load(folder / f"{case}.npy", mmap_mode="r", allow_pickle=False)
        seg = np.load(folder / f"{case}_seg.npy", mmap_mode="r", allow_pickle=False)
    else:
        with np.load(folder / f"{case}.npz", allow_pickle=False) as archive:
            data, seg = archive["data"], archive["seg"]
    with (folder / f"{case}.pkl").open("rb") as stream:
        properties = pickle.load(stream)  # Only trusted, local nnU-Net metadata.
    require(data.ndim == 4 and data.shape[0] == 3, "Expected three preprocessed MRI channels")
    require(seg.shape == (1, *data.shape[1:]), "Preprocessed segmentation shape mismatch")
    require(set(np.unique(seg).tolist()) <= {-1, 0, 1}, "Illegal preprocessed labels")
    require(np.isfinite(data).all(), "Nonfinite preprocessed MRI")
    return data, seg[0], properties


def preprocessed_centroid_lps(mask, properties, transpose, reference):
    # Pixel-centre approximation, using actual old/new shapes rather than nominal spacing.
    if not mask.any():
        return None
    centre = np.array([v.mean() for v in np.where(mask)])
    old_shape = np.array(properties["shape_after_cropping_and_before_resampling"])
    starts = np.array([v[0] for v in properties["bbox_used_for_cropping"]])
    transposed = (centre + .5) * old_shape / np.array(mask.shape) - .5 + starts
    original_zyx = transposed[np.argsort(transpose)]
    return list(reference.TransformContinuousIndexToPhysicalPoint(tuple(original_zyx[::-1])))


def plot_chain(path, raw, material, expected_gt, gt, predicted, predata, pregt, slices):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def show(ax, array, label, pred, title):
        finite = array[np.isfinite(array)]
        lo, hi = np.percentile(finite, [1, 99]) if finite.size else (0, 1)
        ax.imshow(array, cmap="gray", vmin=lo, vmax=hi if hi > lo else lo + 1, origin="upper")
        if label.any():
            ax.contour(label, levels=[.5], colors=["lime"], linewidths=.8)
        if pred is not None and pred.any():
            ax.contour(pred, levels=[.5], colors=["red"], linewidths=.8)
        ax.set_title(title, fontsize=8)
        ax.axis("off")

    present = np.flatnonzero(gt.any(axis=(1, 2)))
    selected = present[np.unique(np.linspace(0, len(present) - 1, min(slices, len(present))).astype(int))]
    def scaled(array):
        lo, hi = np.percentile(array, [1, 99])
        return np.clip((array - lo) / (hi - lo if hi > lo else 1), 0, 1)

    for z in selected:
        fig, axes = plt.subplots(3, 3, figsize=(13, 12))
        for ch, name in enumerate(CHANNELS):
            show(axes[0, ch], raw[ch][z], expected_gt[z], None, f"Original {name.upper()} -> T2 grid (virtual)")
            show(axes[1, ch], material[ch][z], gt[z], predicted[z], f"Materialized {name.upper()}")
        show(axes[2, 0], material[0][z], expected_gt[z], gt[z], "Label check: green=canonical; red=materialized")
        yy, xx = np.indices(gt[z].shape)
        checker = (yy // 24 + xx // 24) % 2 == 0
        for ch in (1, 2):
            board = np.where(checker, scaled(material[0][z]), scaled(material[ch][z]))
            show(axes[2, ch], board, gt[z], None, f"T2/{CHANNELS[ch].upper()} checkerboard (visual alignment)")
        fig.suptitle(f"{path.name} | T2 z={z} | top: green=canonical GT; middle: green=materialized GT, red=prediction\n"
                     "Independent display scaling; visual check required; not a registration certificate")
        fig.tight_layout()
        fig.savefig(path / f"t2_grid_z{z:03d}.png", dpi=140)
        plt.close(fig)
    present = np.flatnonzero(pregt.any(axis=(1, 2)))
    for z in present[np.unique(np.linspace(0, len(present) - 1, min(slices, len(present))).astype(int))] if len(present) else []:
        fig, axes = plt.subplots(1, 3, figsize=(13, 5))
        for ch, name in enumerate(CHANNELS):
            show(axes[ch], predata[ch, z], pregt[z], None, f"Preprocessed {name.upper()}")
        fig.suptitle(f"{path.name} | preprocessed z={z} | green=preprocessed GT\n"
                     "Native preprocessed axes; slice numbers are not T2 slice numbers")
        fig.tight_layout()
        fig.savefig(path / f"preprocessed_z{z:03d}.png", dpi=140)
        plt.close(fig)


def inspect_case(case, row, args, plans, configuration, progress):
    result = {"case_id": case, "center": row["center"], "case_ISUP": row["case_ISUP"],
              "canonical_source": row["canonical_source"], "warnings": [], "checks": {}}
    require(Path(row["canonical_path"]).name == f"{case}.nii.gz", "Canonical label filename/case mismatch")
    for name in CHANNELS:
        require(Path(row[f"{name}_path"]).name == f"{case}_{name}.mha", "Original MRI filename/case mismatch")
    raw = [sitk.ReadImage(row[f"{name}_path"]) for name in CHANNELS]
    require(all(image.GetDimension() == 3 for image in raw), "Expected 3D MRI")
    canonical = sitk.ReadImage(row["canonical_path"])
    canon_array = binary_label(sitk.GetArrayFromImage(canonical), row["canonical_source"])
    require(canon_array.any(), "Selected diagnostic case has empty canonical GT")
    result["original_geometry"] = {name: geometry(image) for name, image in zip(CHANNELS, raw)}
    result["canonical_geometry"] = geometry(canonical)
    result["original_intensity"] = {name: {"dtype": str(sitk.GetArrayFromImage(image).dtype),
                                          **stats(sitk.GetArrayFromImage(image))}
                                    for name, image in zip(CHANNELS, raw)}
    progress.update(1)

    case_dir = args.materialized / "cases" / case
    material = [sitk.ReadImage(str(case_dir / f"{name}.nii.gz")) for name in CHANNELS]
    label_image = sitk.ReadImage(str(case_dir / "lesion.nii.gz"))
    require(all(same_grid(raw[0], image) for image in [*material, label_image]), "Materialized grid differs from original T2")
    gt_array = sitk.GetArrayFromImage(label_image)
    require(set(np.unique(gt_array).tolist()) <= {0, 1}, "Illegal materialized labels")
    gt = gt_array == 1
    expected_gt = sitk.GetArrayFromImage(aligned(image_like(canon_array, canonical), raw[0], True)) == 1
    result["checks"]["canonical_to_materialized_label_equal"] = bool(np.array_equal(expected_gt, gt))
    progress.update(1)

    raw_aligned, mat_arrays = [], []
    result["materialization"] = {}
    for name, original, image in zip(CHANNELS, raw, material):
        original_array = sitk.GetArrayFromImage(original)
        virtual = original_array if same_grid(original, raw[0]) else sitk.GetArrayFromImage(aligned(original, raw[0]))
        actual = sitk.GetArrayFromImage(image)
        require(np.isfinite(virtual).all() and np.isfinite(actual).all(), f"Nonfinite {name}")
        expected = historical_mri_conversion(original, raw[0])
        ones = image_like(np.ones(original_array.shape, dtype=np.uint8), original)
        coverage = sitk.GetArrayFromImage(aligned(ones, raw[0], True)) > 0
        result["materialization"][name] = {
            "expected_historical_uint16_conversion": difference(expected, actual),
            "float_virtual_to_materialized": difference(virtual, actual),
            "virtual_values_below_0": int(np.count_nonzero(virtual < 0)),
            "virtual_values_above_65535": int(np.count_nonzero(virtual > 65535)),
            "geometric_coverage_fraction": float(coverage.mean()),
            "gt_outside_geometric_coverage_fraction": float(np.mean(~coverage[gt])) if gt.any() else None,
            "whole_image_intensity": stats(actual), "gt_intensity": stats(actual, gt),
            "geometric_coverage_note": "Nearest-neighbour support of original voxel grid, not anatomical alignment or valid signal."}
        result["checks"][f"{name}_historical_conversion_equal"] = bool(np.array_equal(expected, actual))
        raw_aligned.append(virtual)
        mat_arrays.append(actual)
        progress.update(1)

    for index, (name, image) in enumerate(zip(CHANNELS, material)):
        linked = args.raw_dataset / "imagesTr" / f"{case}_{index:04d}.nii.gz"
        reference = case_dir / f"{name}.nii.gz"
        linked_image = sitk.ReadImage(str(linked))
        result["checks"][f"nnunet_raw_{name}_same_grid_and_array"] = bool(
            same_grid(image, linked_image) and np.array_equal(mat_arrays[index], sitk.GetArrayFromImage(linked_image)))
        result.setdefault("nnunet_raw_resolved_paths", {})[name] = {"path": str(linked.resolve()),
                                                                  "expected": str(reference.resolve())}
    linked_label = sitk.ReadImage(str(args.raw_dataset / "labelsTr" / f"{case}.nii.gz"))
    exported_gt = sitk.ReadImage(str(args.preprocessed / "gt_segmentations" / f"{case}.nii.gz"))
    for name, image in [("nnunet_raw_label", linked_label), ("nnunet_exported_reference", exported_gt)]:
        result["checks"][f"{name}_same_grid_and_array"] = bool(
            same_grid(label_image, image) and np.array_equal(gt_array, sitk.GetArrayFromImage(image)))

    predata, preseg, props = load_preprocessed(args.preprocessed / configuration["data_identifier"], case)
    pregt = preseg == 1
    transpose = plans["transpose_forward"]
    require(sorted(transpose) == [0, 1, 2], "Invalid transpose_forward")
    require(np.allclose(props["spacing"], label_image.GetSpacing()[::-1]), "Preprocessed source spacing mismatch")
    require(list(props["shape_before_cropping"]) == list(np.array(gt.shape)[transpose]), "Preprocessed source shape mismatch")
    stuff = props["sitk_stuff"]
    require(np.allclose(stuff["origin"], label_image.GetOrigin()) and
            np.allclose(stuff["direction"], label_image.GetDirection()), "Preprocessed source geometry mismatch")
    locations = np.asarray(props["class_locations"][1])
    valid_locations = (locations.ndim == 2 and locations.shape[1] == 4 and len(locations) > 0)
    if valid_locations:
        valid_locations = bool(np.isfinite(locations).all() and np.all(locations == np.rint(locations)) and
                               np.all(locations[:, 0] == 0) and np.all(locations[:, 1:] >= 0) and
                               np.all(locations[:, 1:] < np.array(pregt.shape)))
    if valid_locations:
        valid_locations = bool(np.all(pregt[tuple(locations[:, 1:].astype(int).T)]))
    result["checks"]["preprocessed_foreground_preserved"] = bool(pregt.any())
    result["checks"]["class_locations_valid"] = valid_locations
    # Diagnostic label resampling only, using the frozen nnU-Net implementation.
    # Never writes training arrays or reruns the preprocessing pipeline.
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    config_manager = PlansManager(plans).get_configuration(args.configuration)
    transposed_gt = gt.transpose(transpose)
    bbox = props["bbox_used_for_cropping"]
    require(len(bbox) == 3 and all(0 <= bounds[0] < bounds[1] <= size
                                 for bounds, size in zip(bbox, transposed_gt.shape)), "Invalid crop bbox")
    cropped_gt = transposed_gt[tuple(slice(*bounds) for bounds in bbox)]
    require(list(cropped_gt.shape) == list(props["shape_after_cropping_and_before_resampling"]), "Crop shape mismatch")
    result["checks"]["crop_preserves_all_gt_voxels"] = int(cropped_gt.sum()) == int(gt.sum())
    expected_preseg = config_manager.resampling_fn_seg(
        cropped_gt[None].astype(np.int8), pregt.shape,
        np.array(props["spacing"])[transpose], configuration["spacing"])[0] == 1
    result["checks"]["preprocessed_foreground_matches_native_label_resampling"] = bool(np.array_equal(expected_preseg, pregt))
    result["preprocessed_label_comparison"] = {
        "expected_vs_stored_dice": dice(expected_preseg, pregt),
        "different_foreground_voxels": int(np.count_nonzero(expected_preseg != pregt)),
        "crop_lost_foreground_voxels": int(gt.sum() - cropped_gt.sum()),
        "note": "Diagnostic foreground comparison using frozen native segmentation resampler; -1 padding treated separately."}
    progress.update(1)
    result["labels"] = {
        "canonical": lesion_stats(canon_array > 0, canonical.GetSpacing()[::-1]),
        "materialized": lesion_stats(gt, label_image.GetSpacing()[::-1]),
        "preprocessed_nominal_spacing": lesion_stats(pregt, configuration["spacing"])}
    old_shape = np.array(props["shape_after_cropping_and_before_resampling"])
    effective_spacing = np.array(props["spacing"])[transpose] * old_shape / np.array(pregt.shape)
    result["labels"]["preprocessed_extent_based_spacing"] = lesion_stats(pregt, effective_spacing)
    result["labels"]["preprocessed_to_materialized_volume_ratio"] = (
        result["labels"]["preprocessed_extent_based_spacing"]["volume_ml"] /
        result["labels"]["materialized"]["volume_ml"] if gt.any() else None)
    result["labels"]["volume_note"] = "Voxel-count estimates; nominal and extent-based spacing reported separately; no clinical lesion-instance matching."
    for label in result["labels"].values():
        if isinstance(label, dict) and label.get("centroid_zyx") is not None:
            label["centroid_note"] = "Indices use each stage's own axes/grid."
    centre = result["labels"]["materialized"]["centroid_zyx"]
    result["materialized_centroid_lps_mm"] = list(label_image.TransformContinuousIndexToPhysicalPoint(tuple(centre[::-1]))) if centre else None
    result["preprocessed_centroid_lps_mm_approx"] = preprocessed_centroid_lps(pregt, props, transpose, label_image)
    result["preprocessed"] = {"shape_czyx": list(predata.shape), "nominal_spacing_transposed": configuration["spacing"],
                               "extent_based_spacing_transposed": effective_spacing.tolist(),
                               "bbox_used_for_cropping": props["bbox_used_for_cropping"],
                               "class_locations_count": int(len(locations)),
                               "intensity": {name: {"whole_image": stats(predata[i]), "gt": stats(predata[i], pregt)}
                                             for i, name in enumerate(CHANNELS)}}
    prediction_image = sitk.ReadImage(str(args.run / "validation" / f"{case}.nii.gz"))
    require(same_grid(label_image, prediction_image), "Prediction geometry mismatch")
    prediction_array = sitk.GetArrayFromImage(prediction_image)
    require(set(np.unique(prediction_array).tolist()) <= {0, 1}, "Illegal prediction labels")
    predicted = prediction_array == 1
    result["prediction"] = {"dice": dice(gt, predicted), "tp_voxels": int((gt & predicted).sum()),
                            "fp_voxels": int((~gt & predicted).sum()), "fn_voxels": int((gt & ~predicted).sum()),
                            "predicted_voxels": int(predicted.sum()),
                            "probabilities_checked": False}
    result["warnings"].append("Hard predictions cannot distinguish subthreshold response from absent response.")
    for name, values in result["materialization"].items():
        if values["gt_outside_geometric_coverage_fraction"]:
            result["warnings"].append(f"{name}: some GT lies outside original grid support")
    progress.update(1)
    output = args.output / case
    output.mkdir(exist_ok=False)
    plot_chain(output, raw_aligned, mat_arrays, expected_gt, gt, predicted, predata, pregt, args.slices)
    progress.update(1)
    result["status"] = "COMPLETED" if all(result["checks"].values()) else "CHECK_MISMATCH"
    return result


def write_json(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False,
                  default=lambda value: value.tolist() if isinstance(value, np.ndarray) else value.item())


def probability_diagnostics(probabilities, gt):
    require(probabilities.shape == (2, *gt.shape), "Expected restored two-class probabilities on GT grid")
    require(np.isfinite(probabilities).all(), "Nonfinite probabilities")
    require(float(probabilities.min()) >= 0 and float(probabilities.max()) <= 1, "Probabilities outside [0,1]")
    require(np.allclose(probabilities.sum(axis=0), 1, rtol=0, atol=.002), "Probabilities do not sum to one")
    lesion = probabilities[1]
    thresholds = []
    for threshold in (.01, .05, .1, .2, .3, .5):
        mask = lesion > threshold
        tp, fp, fn = int((mask & gt).sum()), int((mask & ~gt).sum()), int((~mask & gt).sum())
        thresholds.append({"threshold_strictly_greater_than": threshold,
                           "tp_voxels": tp, "fp_voxels": fp, "fn_voxels": fn,
                           "gt_coverage_fraction": tp / int(gt.sum()) if gt.any() else None,
                           "precision": tp / (tp + fp) if tp + fp else None,
                           "dice": dice(mask, gt)})
    return {"gt_positive": bool(gt.any()), "gt_voxels": int(gt.sum()),
            "gt_probability": stats(lesion, gt), "outside_gt_probability": stats(lesion, ~gt),
            "threshold_diagnostics": thresholds,
            "threshold_note": "Exploratory fixed thresholds, no optimal threshold selection or formal performance claim; outside-GT FP is voxel count, not lesion/case FPR."}


def inspect_probabilities(case, row, args, progress):
    folder = args.probabilities
    with np.load(folder / f"{case}.npz", allow_pickle=False) as archive:
        probabilities = archive["probabilities"].astype(np.float32)
    with (folder / f"{case}.pkl").open("rb") as stream:
        props = pickle.load(stream)
    ref = sitk.ReadImage(str(args.preprocessed / "gt_segmentations" / f"{case}.nii.gz"))
    values = sitk.GetArrayFromImage(ref)
    require(set(np.unique(values).tolist()) <= {0, 1}, "Illegal GT values")
    gt = values == 1
    require(np.allclose(props["sitk_stuff"]["spacing"], ref.GetSpacing(), rtol=0, atol=1e-4)
            and np.allclose(props["sitk_stuff"]["origin"], ref.GetOrigin(), rtol=0, atol=1e-4)
            and np.allclose(props["sitk_stuff"]["direction"], ref.GetDirection(), rtol=0, atol=1e-4),
            "Restored probability geometry metadata differs from GT")
    native = sitk.ReadImage(str(folder / f"{case}.nii.gz"))
    previous = sitk.ReadImage(str(args.run / "validation" / f"{case}.nii.gz"))
    require(same_grid(native, ref) and same_grid(previous, ref), "Hard prediction geometry differs from GT")
    native_array = sitk.GetArrayFromImage(native)
    previous_array = sitk.GetArrayFromImage(previous)
    require(set(np.unique(native_array).tolist()) <= {0, 1}, "Illegal new hard prediction")
    result = {"case_id": case, "status": "COMPLETED", "checks": {},
              "probability_folder": str(folder.resolve()), **probability_diagnostics(probabilities, gt)}
    progress.update(1)
    result["checks"]["argmax_equals_new_hard_prediction"] = bool(np.array_equal(probabilities.argmax(axis=0), native_array))
    result["new_vs_original_hard_prediction"] = difference(native_array, previous_array)
    result["hard_prediction_dice"] = dice(native_array == 1, gt)
    lesion = probabilities[1]
    peak = tuple(int(v) for v in np.unravel_index(lesion.argmax(), lesion.shape))
    gt_peak = None
    nearest_distance = None
    if gt.any():
        gt_peak_index = np.flatnonzero(gt.ravel())[int(np.argmax(lesion[gt]))]
        gt_peak = tuple(int(v) for v in np.unravel_index(gt_peak_index, gt.shape))
        coordinates = np.argwhere(gt)
        delta_mm = (coordinates - np.asarray(peak)) * np.asarray(ref.GetSpacing()[::-1])
        nearest_distance = float(np.sqrt((delta_mm ** 2).sum(axis=1).min()))
    result["global_peak"] = {"probability": float(lesion[peak]), "index_zyx": list(peak),
                             "inside_gt": bool(gt[peak]),
                             "nearest_gt_voxel_centre_distance_mm": nearest_distance,
                             "point_lps_mm": list(ref.TransformIndexToPhysicalPoint(peak[::-1]))}
    result["gt_peak"] = {"probability": float(lesion[gt_peak]), "index_zyx": list(gt_peak)} if gt_peak else None
    result["voxel_volume_mm3"] = float(np.prod(ref.GetSpacing()))
    result["case_lesion_burden_mm3"] = float(gt.sum() * np.prod(ref.GetSpacing()))
    progress.update(1)
    if args.no_plots or not gt.any():
        result["plots_generated"] = False
        progress.update(1)
        if not all(result["checks"].values()):
            result["status"] = "CHECK_MISMATCH"
        return result
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t2 = sitk.ReadImage(str(args.materialized / "cases" / case / "t2w.nii.gz"))
    require(same_grid(t2, ref), "T2 geometry differs from GT")
    t2array = sitk.GetArrayFromImage(t2)
    out = args.output / case
    out.mkdir(exist_ok=False)
    slices = np.flatnonzero(gt.any(axis=(1, 2)))
    selected = sorted(set(slices[np.linspace(0, len(slices)-1, min(args.slices, len(slices))).astype(int)].tolist() + [gt_peak[0]]))
    for z in selected:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        lo, hi = np.percentile(t2array[z], [1, 99])
        axes[0].imshow(t2array[z], cmap="gray", vmin=lo, vmax=hi if hi > lo else lo + 1)
        if (native_array[z] == 1).any():
            axes[0].contour(native_array[z] == 1, levels=[.5], colors=["red"])
        axes[0].set_title("T2: green=GT; red=native hard prediction")
        for axis, limit, title in [(axes[1], 1., "Lesion probability: fixed 0-1"),
                                    (axes[2], max(float(lesion.max()), 1e-6), "Lesion probability: scaled to case maximum")]:
            artist = axis.imshow(lesion[z], cmap="magma", vmin=0, vmax=limit)
            fig.colorbar(artist, ax=axis, fraction=.046)
            axis.set_title(title)
        for axis in axes:
            axis.contour(gt[z], levels=[.5], colors=["lime"])
            axis.axis("off")
        fig.suptitle(f"{case} | original T2 grid z={z} | green=GT | diagnostic only")
        fig.tight_layout()
        fig.savefig(out / f"probability_z{z:03d}.png", dpi=140)
        plt.close(fig)
    progress.update(1)
    result["plots_generated"] = True
    if not all(result["checks"].values()):
        result["status"] = "CHECK_MISMATCH"
    return result


def aggregate_probabilities(cases, expected_total):
    complete = [case for case in cases if case["status"] == "COMPLETED"]
    positives = [case for case in complete if case["gt_positive"]]
    negatives = [case for case in complete if not case["gt_positive"]]
    result = {"scope": "exploratory segmentation probability diagnostics",
              "complete_cohort": len(complete) == expected_total,
              "included": len(complete), "excluded_error_or_mismatch": len(cases) - len(complete),
              "positive_cases": len(positives), "negative_cases": len(negatives),
              "new_hard_prediction_differs_from_original_cases": [case["case_id"] for case in complete
                                                                    if not case["new_vs_original_hard_prediction"]["equal"]],
              "thresholds": [],
              "interpretation": "No optimal threshold selection, CIs, AUROC/AP/FROC or formal endpoint replacement. Positive any-overlap rate is not lesion-instance sensitivity. Empty-negative Dice is not included in positive macro Dice."}
    for index, threshold in enumerate((.01, .05, .1, .2, .3, .5)):
        pos = [case["threshold_diagnostics"][index] for case in positives]
        neg = [case["threshold_diagnostics"][index] for case in negatives]
        tp, fp, fn = (sum(item[key] for item in pos) for key in ("tp_voxels", "fp_voxels", "fn_voxels"))
        neg_fp = [item["fp_voxels"] for item in neg]
        result["thresholds"].append({
            "threshold_strictly_greater_than": threshold,
            "positive_macro_dice": float(np.mean([item["dice"] for item in pos])) if pos else None,
            "positive_pooled_dice": 2 * tp / (2 * tp + fp + fn) if pos else None,
            "positive_pooled_voxel_recall": tp / (tp + fn) if pos else None,
            "positive_cases_with_zero_gt_overlap": sum(item["tp_voxels"] == 0 for item in pos),
            "positive_cases_with_any_gt_overlap": sum(item["tp_voxels"] > 0 for item in pos),
            "positive_cases_with_empty_prediction": sum(item["tp_voxels"] + item["fp_voxels"] == 0 for item in pos),
            "negative_cases_with_fp": sum(value > 0 for value in neg_fp),
            "negative_case_fp_rate": sum(value > 0 for value in neg_fp) / len(neg) if neg else None,
            "negative_fp_voxels_total": sum(neg_fp),
            "negative_fp_voxels_mean_all_negative_cases": float(np.mean(neg_fp)) if neg else None,
            "negative_fp_volume_mm3_mean_all_negative_cases": float(np.mean([
                item["fp_voxels"] * case["voxel_volume_mm3"] for item, case in zip(neg, negatives)])) if neg else None})
    result["positive_zero_overlap_at_0_5_cases"] = [case["case_id"] for case in positives
                                                   if case["threshold_diagnostics"][-1]["tp_voxels"] == 0]
    result["positive_zero_overlap_at_0_5_but_overlap_at_0_1_cases"] = [case["case_id"] for case in positives
        if case["threshold_diagnostics"][-1]["tp_voxels"] == 0 and case["threshold_diagnostics"][2]["tp_voxels"] > 0]
    return result


def safe_output(path):
    resolved = path.resolve()
    allowed = (PROJECT / "outputs/diagnostics").resolve()
    require(resolved.is_relative_to(allowed) and resolved != allowed,
            "Output must be a NEW subdirectory of project outputs/diagnostics")
    require(not path.exists() and not path.is_symlink(), f"Refusing existing output: {path}")
    return resolved


def self_test():
    # No files, original images, nnU-Net dataset reads, or real-data processing.
    mask = np.zeros((5, 7, 9), dtype=np.uint8)
    mask[1:3, 2:4, 3:5] = 1
    image = sitk.GetImageFromArray(mask)
    image.SetSpacing((.5, .8, 3.))
    image.SetOrigin((11., -7., 2.))
    image.SetDirection((0., -1., 0., 1., 0., 0., 0., 0., 1.))
    assert same_grid(image, aligned(image, image, True))
    assert np.array_equal(sitk.GetArrayFromImage(aligned(image, image, True)), mask)
    shifted = sitk.Image(image)
    shifted.SetOrigin((11., -7., 200.))
    assert not sitk.GetArrayFromImage(aligned(shifted, image, True)).any()
    assert not same_grid(image, shifted)
    ramp = sitk.GetImageFromArray(np.broadcast_to(np.arange(9, dtype=np.uint16), (5, 7, 9)).copy())
    offset = sitk.Image(ramp)
    offset.SetOrigin((.75, 0., 0.))
    historical = historical_mri_conversion(ramp, offset)
    floating = sitk.GetArrayFromImage(aligned(ramp, offset))
    assert historical[2, 3, 2] == 2 and np.isclose(floating[2, 3, 2], 2.75)
    assert np.rint(floating[2, 3, 2]) == 3
    assert np.array_equal(historical_mri_conversion(ramp, ramp), sitk.GetArrayFromImage(ramp))
    assert np.array_equal(binary_label(mask, "human_expert/Pooch25"), mask)
    assert np.array_equal(binary_label(mask * 2, "human_expert/resampled"), mask)
    try:
        binary_label(mask, "human_expert/resampled")
    except ValueError:
        pass
    else:
        raise AssertionError("Illegal label 1 in resampled source was accepted")
    measured = lesion_stats(mask, image.GetSpacing()[::-1])
    assert measured["voxels"] == 8 and measured["components_6_connected"] == 1
    assert np.isclose(measured["volume_ml"], .0096)
    assert dice(mask.astype(bool), mask.astype(bool)) == 1
    assert dice(mask.astype(bool), np.zeros_like(mask, dtype=bool)) == 0
    assert difference(mask, mask)["equal"]
    props = {"shape_after_cropping_and_before_resampling": mask.shape,
             "bbox_used_for_cropping": [[0, v] for v in mask.shape]}
    point = preprocessed_centroid_lps(mask, props, [0, 1, 2], image)
    assert np.allclose(point, image.TransformContinuousIndexToPhysicalPoint((3.5, 2.5, 1.5)))
    transposed = mask.transpose(1, 2, 0)
    props["shape_after_cropping_and_before_resampling"] = transposed.shape
    props["bbox_used_for_cropping"] = [[0, v] for v in transposed.shape]
    assert np.allclose(preprocessed_centroid_lps(transposed, props, [1, 2, 0], image), point)
    assert stats(np.array([0., 1., np.nan]))["nonfinite"] == 1
    soft = np.full(mask.shape, .2, dtype=np.float32)
    report = probability_diagnostics(np.stack((1-soft, soft)), mask.astype(bool))
    assert report["gt_probability"]["max"] < .5
    assert report["threshold_diagnostics"][-1]["gt_coverage_fraction"] == 0
    assert report["threshold_diagnostics"][2]["gt_coverage_fraction"] == 1
    negative = probability_diagnostics(np.stack((1-soft, soft)), np.zeros_like(mask, dtype=bool))
    assert negative["gt_probability"]["n"] == 0 and negative["threshold_diagnostics"][2]["gt_coverage_fraction"] is None
    assert negative["threshold_diagnostics"][-1]["dice"] is None
    synthetic_cases = [{"case_id": str(i), "status": "COMPLETED", "voxel_volume_mm3": 1.,
                        "new_vs_original_hard_prediction": {"equal": True}, **entry}
                       for i, entry in enumerate((report, negative))]
    aggregate = aggregate_probabilities(synthetic_cases, 2)
    assert aggregate["complete_cohort"] and aggregate["positive_cases"] == aggregate["negative_cases"] == 1
    assert aggregate["thresholds"][2]["negative_case_fp_rate"] == 1
    assert aggregate["thresholds"][-1]["negative_case_fp_rate"] == 0
    assert aggregate["thresholds"][-1]["positive_cases_with_zero_gt_overlap"] == 1
    assert not aggregate_probabilities(synthetic_cases, 3)["complete_cohort"]
    for forbidden in [PROJECT / "data/new", PROJECT / "outputs/nnUNet_results/new", PROJECT / "outputs/diagnostics"]:
        try:
            safe_output(forbidden)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Unsafe output accepted: {forbidden}")
    print("Synthetic checks passed: geometry, label mapping, physical resampling, volumes, transpose, metrics, output confinement. No files written.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default=DEFAULT_CASES, help="Comma-separated positive validation case IDs")
    parser.add_argument("--controls", type=int, default=2, help="Add top-Dice positive validation cases as visual controls (0 disables)")
    parser.add_argument("--manifest", type=Path, default=PROJECT / "data/metadata/picai_manifest.csv")
    parser.add_argument("--materialized", type=Path, default=PROJECT / "data/processed/picai")
    parser.add_argument("--raw-dataset", type=Path, default=PROJECT / "workdir/nnUNet_raw/Dataset605_PICAI")
    parser.add_argument("--preprocessed", type=Path, default=PROJECT / "workdir/nnUNet_preprocessed/Dataset605_PICAI")
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--probabilities", type=Path, default=None,
                        help="Analyze existing native restored lesion probabilities in this predictions folder; no inference")
    parser.add_argument("--all-validation", action="store_true", help="Analyze ALL validation cases, including negatives; requires --probabilities; ignores --cases/--controls")
    parser.add_argument("--no-plots", action="store_true", help="Probability statistics only; do not generate heatmaps")
    parser.add_argument("--configuration", default="3d_fullres")
    parser.add_argument("--output", type=Path, default=None, help="NEW outputs/diagnostics subdirectory; default timestamped")
    parser.add_argument("--slices", type=int, default=3, help="GT-bearing slices per grid, evenly spaced")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    require(Path(sys.executable).absolute() == Path("/root/anaconda3/envs/lm/bin/python"), "Activate conda lm")
    import nnunetv2
    require(Path(nnunetv2.__file__).resolve().is_relative_to(PROJECT / "third_party/nnUNet"), "Source scripts/env_nnunet.sh first")
    require(args.controls >= 0 and args.slices > 0, "controls must be >=0; slices must be >0")
    require(not args.all_validation or args.probabilities is not None, "--all-validation requires --probabilities")
    cases = [item.strip() for item in args.cases.split(",") if item.strip()]
    require(cases and len(cases) == len(set(cases)), "Empty or duplicate case IDs")
    require(all(re.fullmatch(r"\d+_\d+", case) for case in cases), "Illegal case ID")
    with args.manifest.open(newline="", encoding="utf-8") as stream:
        manifest_rows = list(csv.DictReader(stream))
    rows = {row["case_id"]: row for row in manifest_rows}
    require(len(rows) == len(manifest_rows), "Duplicate manifest case IDs")
    summary = json.loads((args.run / "validation/summary.json").read_text())
    metrics = {Path(item["reference_file"]).name.removesuffix(".nii.gz"): item["metrics"]["1"]
               for item in summary["metric_per_case"]}
    require(len(metrics) == len(summary["metric_per_case"]), "Duplicate validation summary cases")
    if args.all_validation:
        cases, controls = sorted(metrics), []
    else:
        for case in cases:
            require(case in rows and case in metrics and metrics[case]["n_ref"] > 0,
                    f"Not a known positive validation case: {case}")
        eligible = sorted((case for case, metric in metrics.items()
                           if metric["n_ref"] > 0 and metric["Dice"] >= .5 and case not in cases),
                          key=lambda case: (-metrics[case]["Dice"], case))
        require(len(eligible) >= args.controls, "Not enough positive controls with Dice >=0.5")
        controls = eligible[:args.controls]
        cases += controls
    plans = json.loads((args.preprocessed / "nnUNetPlans.json").read_text())
    configuration = plans["configurations"][args.configuration]
    require(len(configuration["spacing"]) == 3, "Only 3D configurations are supported")
    dataset = json.loads((args.raw_dataset / "dataset.json").read_text())
    require(len(dataset["channel_names"]) == 3 and
            {int(key): value for key, value in dataset["channel_names"].items()} == {0: "T2W", 1: "ADC", 2: "HBV"}
            and dataset["labels"] == {"background": 0, "lesion": 1}, "Unexpected dataset channel/label contract")
    fold_match = re.fullmatch(r"fold_(\d+)", args.run.name)
    require(fold_match is not None, "Run must point to a fold_N directory")
    splits = json.loads((args.preprocessed / "splits_final.json").read_text())
    split = splits[int(fold_match.group(1))]
    train, validation = set(split["train"]), set(split["val"])
    require(not train & validation, "Train/validation case overlap")
    require(not {case.split("_")[0] for case in train} & {case.split("_")[0] for case in validation},
            "Train/validation patient overlap")
    require(set(metrics) == validation, "Validation summary differs from selected fold split")
    require(all(case in rows and case in validation for case in cases), "Selected cases missing from manifest/validation")
    if args.probabilities:
        require(all((args.probabilities / f"{case}{suffix}").is_file()
                    for case in cases for suffix in (".npz", ".pkl", ".nii.gz")), "Missing probability/hard prediction/metadata files")
        if args.all_validation:
            require({path.name.removesuffix(".npz") for path in args.probabilities.glob("*.npz")} == validation,
                    "Probability case set differs from validation split")
    prefix = "picai_probability_diagnostics" if args.probabilities else "picai_case_chain"
    args.output = safe_output(args.output or PROJECT / "outputs/diagnostics" /
                             f"{prefix}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S_%fZ}")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"status": "RUNNING", "created_utc": datetime.now(timezone.utc).isoformat(),
              "command": sys.argv, "run": str(args.run.resolve()), "output": str(args.output),
              "manifest": str(args.manifest.resolve()), "preprocessed": str(args.preprocessed.resolve()),
              "mode": "probability_diagnostics" if args.probabilities else "case_chain",
              "plans": plans, "selected_cases": cases, "controls": controls,
              "control_selection": ("All cases from selected validation split; no success-case selection." if args.all_validation else
                                    "Top-Dice positive validation cases (Dice >=0.5); illustrative, not matched or unbiased."),
              "limitations": ["Subset diagnostics; no dataset-wide PASS or causal conclusion.",
                              "Identity physical resampling is not anatomical registration.",
                              "Intensity ranges across native grids are descriptive, not directly matched samples.",
                              "6-connected components are diagnostic, not official lesion evaluation.",
                              "Centroid mapping is approximate; preprocessing is not rerun."], "cases": []}
    print(f"Read-only diagnostic cases: {cases}\nOutput: {args.output}", flush=True)
    stages = 3 if args.probabilities else 8
    with tqdm(total=len(cases) * stages, desc=prefix, unit="stage", disable=args.no_progress, file=sys.stdout) as progress:
        for case in cases:
            progress.set_postfix_str(case)
            before = progress.n
            try:
                result = (inspect_probabilities(case, rows[case], args, progress) if args.probabilities else
                          inspect_case(case, rows[case], args, plans, configuration, progress))
                if args.probabilities:
                    result["checks"]["gt_voxels_match_original_summary"] = result["gt_voxels"] == metrics[case]["n_ref"]
                    if not all(result["checks"].values()):
                        result["status"] = "CHECK_MISMATCH"
                result["selection_role"] = ("full_validation_cohort" if args.all_validation else
                                            "visual_success_control" if case in controls else "requested_case")
            except Exception as exc:
                result = {"case_id": case, "status": "ERROR", "error": f"{type(exc).__name__}: {exc}"}
                tqdm.write(f"ERROR {case}: {result['error']}")
            progress.update(before + stages - progress.n)
            write_json(args.output / f"{case}.json", result)
            report["cases"].append(result)
    good = sum(case["status"] == "COMPLETED" for case in report["cases"])
    report.update(status="DIAGNOSTICS_COMPLETED_REVIEW_REQUIRED" if good == len(cases) else "DIAGNOSTICS_INCOMPLETE_OR_MISMATCH",
                  successful=good, failed=len(cases) - good, skipped=0, elapsed_seconds=time.monotonic() - started)
    if args.probabilities:
        report["probability_aggregate"] = aggregate_probabilities(report["cases"], len(cases))
        report["full_validation_requested"] = args.all_validation
    write_json(args.output / "summary.json", report)
    print(json.dumps({key: report[key] for key in ("status", "successful", "failed", "skipped", "elapsed_seconds", "output")}, ensure_ascii=False))
    return 0 if good == len(cases) else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f"Diagnostic setup failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
