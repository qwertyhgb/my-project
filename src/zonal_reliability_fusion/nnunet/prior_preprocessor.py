"""Wrap native MRI preprocessing; align soft anatomy on exactly the same grid.

MRI crop, normalization, resampling and class_locations remain native. Priors
use native linear probability interpolation after the identical transpose/crop.
No raw-image resampling implementation or second training framework is added.
"""
from __future__ import annotations

import copy

import numpy as np
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager

from zonal_reliability_fusion.anatomy.dataset import (
    PREDICTED_DATASET,
    validate_predicted_prior_dataset,
)


class PredictedAnatomyPreprocessor(DefaultPreprocessor):
    def run_case_npy(self, data, seg, properties, plans_manager, configuration_manager, dataset_json):
        if data.ndim != 4 or data.shape[0] != 6:
            raise ValueError("expected MRI + predicted WG/PZ/TZ")
        prior = data[3:]
        if not np.isfinite(prior).all() or (prior < 0).any() or (prior > 1).any():
            raise ValueError("invalid soft anatomy input")
        config = copy.deepcopy(configuration_manager.configuration)
        config["normalization_schemes"] = config["normalization_schemes"][:3]
        config["use_mask_for_norm"] = config["use_mask_for_norm"][:3]
        mri, segmentation, props = super().run_case_npy(
            data[:3], seg, properties, plans_manager, ConfigurationManager(config), dataset_json)
        forward = plans_manager.transpose_forward
        prior = prior.transpose([0, *[i+1 for i in forward]])
        bbox = props["bbox_used_for_cropping"]
        prior = prior[(slice(None), *[slice(lo, hi) for lo, hi in bbox])]
        spacing = [props["spacing"][i] for i in forward]
        prior = configuration_manager.resampling_fn_probabilities(
            prior, mri.shape[1:], spacing, configuration_manager.spacing)
        if not np.isfinite(prior).all() or (prior < 0).any() or (prior > 1).any():
            raise ValueError("probability interpolation violated [0,1]")
        return np.concatenate((mri, prior.astype(np.float32)), axis=0), segmentation, props

    def run(self, dataset_name_or_id, configuration_name, plans_identifier, num_processes):
        import json
        import time
        from pathlib import Path

        from nnunetv2.paths import nnUNet_preprocessed

        if dataset_name_or_id not in (608, "608", PREDICTED_DATASET):
            raise ValueError("only independent Dataset608 is supported")
        folder = Path(nnUNet_preprocessed) / PREDICTED_DATASET
        document = json.loads((folder / "dataset.json").read_text())
        validate_predicted_prior_dataset(document, PREDICTED_DATASET)
        plans = json.loads((folder / f"{plans_identifier}.json").read_text())
        target = folder / plans["configurations"][configuration_name]["data_identifier"]
        if target.exists() or target.is_symlink():
            raise ValueError(f"refusing native preprocessing overwrite: {target}")
        started = time.monotonic()
        try:
            super().run(PREDICTED_DATASET, configuration_name, plans_identifier, num_processes)
        except Exception:
            print(f"[prior-preprocess] failed=1 success=unknown skipped=0 "
                  f"elapsed={time.monotonic()-started:.2f}s output={target} INCOMPLETE; preserved")
            raise
        print(f"[prior-preprocess] success={document['numTraining']} failed=0 skipped=0 "
              f"elapsed={time.monotonic()-started:.2f}s output={target}")


def install_prior_preprocessor_resolver():
    """Idempotent in-process resolver, including native prediction workers."""
    from nnunetv2.utilities.plans_handling import plans_handler

    if getattr(plans_handler.recursive_find_python_class, "_predicted_anatomy", False):
        return
    original = plans_handler.recursive_find_python_class

    def resolver(folder, class_name, current_module):
        if class_name == "PredictedAnatomyPreprocessor":
            return PredictedAnatomyPreprocessor
        return original(folder, class_name, current_module)

    resolver._predicted_anatomy = True
    plans_handler.recursive_find_python_class = resolver


def prepare_frozen_plans(source_preprocessed, predicted_raw, target_preprocessed):
    """User-run preparation: clone baseline architecture/spacing/budget unchanged."""
    import json
    from pathlib import Path

    source, raw, target = map(Path, (source_preprocessed, predicted_raw, target_preprocessed))
    if source.name != "Dataset605_PICAI" or target.name != PREDICTED_DATASET or target.exists() or target.is_symlink():
        raise ValueError("require Dataset605 source and a new Dataset608 output")
    doc = json.loads((raw / "dataset.json").read_text())
    validate_predicted_prior_dataset(doc, target.name)
    splits = json.loads((source / "splits_final.json").read_text())
    contract = doc["predicted_anatomy_contract"]
    if splits[0] != {"train": contract["train_cases"], "val": contract["val_cases"]}:
        raise ValueError("baseline split differs from predicted prior split")
    plans = json.loads((source / "nnUNetPlans.json").read_text())
    plans["dataset_name"] = PREDICTED_DATASET
    config = plans["configurations"]["3d_fullres"]
    config["normalization_schemes"] += ["NoNormalization"] * 3
    config["use_mask_for_norm"] += [False] * 3
    config["preprocessor_name"] = "PredictedAnatomyPreprocessor"
    plans["configurations"] = {"3d_fullres": config}
    plans["predicted_anatomy_plans_source"] = str((source / "nnUNetPlans.json").resolve())
    target.mkdir(parents=True)
    for filename, value in (("nnUNetPlans.json", plans), ("dataset.json", doc),
                            ("splits_final.json", splits)):
        (target / filename).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    # Fingerprint remains explicitly MRI-derived, never a claim of a new fingerprint audit.
    fingerprint = json.loads((source / "dataset_fingerprint.json").read_text())
    fingerprint["predicted_anatomy_note"] = "MRI fingerprint reused from frozen Dataset605; priors noNorm"
    (target / "dataset_fingerprint.json").write_text(json.dumps(fingerprint, indent=2) + "\n")
    print(f"[prior-plans] success=1 failed=0 skipped=0 output={target}; preprocessing not started")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="User-run frozen Dataset608 plans or native preprocessing")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    for key in ("source-preprocessed", "predicted-raw", "target-preprocessed"):
        prepare.add_argument(f"--{key}", required=True)
    preprocess = sub.add_parser("preprocess")
    preprocess.add_argument("--processes", type=int, default=2)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "prepare":
        prepare_frozen_plans(**args)
    else:
        PredictedAnatomyPreprocessor(verbose=False).run(608, "3d_fullres", "nnUNetPlans", args["processes"])
