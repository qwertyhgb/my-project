"""Synthetic data only, including real native Trainer construction and plans."""
from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from zonal_reliability_fusion.anatomy.dataset import (
    PREDICTED_DATASET,
    validate_predicted_prior_dataset,
)
from zonal_reliability_fusion.anatomy.validation import soft_head_metrics
from zonal_reliability_fusion.multimodal.conditioned_fusion import (
    ConditionedMultimodalNNUNet,
    fusion_weight_statistics,
    parameter_counts,
)
from zonal_reliability_fusion.nnunet import trainers

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def predicted_document():
    return {
        "channel_names": {f"{i:04d}": name for i, name in enumerate(
            ["T2W", "ADC", "HBV", "noNorm", "noNorm", "noNorm"])},
        "labels": {"background": 0, "lesion": 1}, "numTraining": 2,
        "file_ending": ".nii.gz",
        "predicted_anatomy_contract": {
            "source": "predicted_prior", "channel_order": ["WG", "PZ", "TZ"],
            "train_cases": ["1_10"], "val_cases": ["2_20"],
            "cases": {
                cid: {"case_id": cid, "prior_mode": mode, "checkpoint": "synthetic.pth",
                      "trainer": "Stage1", "source_split": "fold_0", "geometry": {"synthetic": True},
                      "anatomy_training_patient_ids": ["1"]}
                for cid, mode in (("1_10", "IN_SAMPLE_PRED"), ("2_20", "HELD_OUT_PRED"))
            },
        },
    }


@pytest.mark.parametrize("condition", ["neutral", "lesion", "anatomy"])
def test_fusion_paths_gradients_and_identity(condition):
    torch.manual_seed(17)
    model = ConditionedMultimodalNNUNet(nn.Identity(), condition=condition)
    x = torch.randn(1, 3, 8, 8, 8)
    if condition == "anatomy":
        x = torch.cat((x, torch.rand_like(x)), dim=1)
    x.requires_grad_()
    fused, weights, neutral = model.fuse(x)
    assert fused.shape == (1, 3, 8, 8, 8)
    assert len({id(stem[0].weight) for stem in model.stems}) == 3
    if condition != "neutral":
        assert torch.equal(fused, neutral)
        assert weights.shape == (1, 3, 8, 8, 8)
        assert torch.all(weights >= 0) and torch.all(weights <= 1)
        assert torch.allclose(weights.sum(1), torch.ones_like(weights[:, 0]))
        # Zero residual blocks controller gradients at initialization by design.
        # Verify gradient paths once residual weights are nonzero, not at step zero.
        nn.init.normal_(model.residual_projection.weight, std=.05)
        fused, weights, _ = model.fuse(x)
        aux = model.auxiliary_outputs["lesionness_logits"]
        loss = fused.square().mean() + aux.square().mean()
    else:
        loss = fused.square().mean()
    loss.backward()
    for stem in model.stems:
        assert stem[0].weight.grad.abs().sum() > 0
    if condition != "neutral":
        assert model.controller[0].weight.grad.abs().sum() > 0
        assert model.lesionness_head[0].weight.grad.abs().sum() > 0
    if condition == "anatomy":
        assert x.grad[:, 3:].abs().sum() > 0
        with torch.no_grad():
            _, first, _ = model.fuse(x)
            changed = x.clone(); changed[:, 3:] = 1 - changed[:, 3:]
            _, second, _ = model.fuse(changed)
        assert not torch.equal(first, second)
    model.export_fusion_weights = True
    model(x)
    if condition != "neutral":
        artifact = model.fusion_weight_artifact(
            case_id="synthetic", checkpoint="synthetic", geometry={"synthetic": True})
        stats = fusion_weight_statistics(artifact["weights"][0].reshape(3, -1))
        assert len(stats["mean"]) == 3 and stats["mean_entropy"] >= 0


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -.1, 1.1])
def test_soft_prior_fail_closed(value):
    model = ConditionedMultimodalNNUNet(nn.Identity(), condition="anatomy")
    x = torch.rand(1, 6, 4, 4, 4); x[:, 3, 0, 0, 0] = value
    with pytest.raises(ValueError):
        model(x)
    with pytest.raises(ValueError):
        model(torch.rand(1, 5, 4, 4, 4))


def test_soft_heads_are_not_ordered_hard_export():
    reference = np.full((2, 2, 2), 3, dtype=np.uint8)  # WG + PZ
    probability = np.stack([np.full(reference.shape, .9), np.full(reference.shape, .9),
                            np.full(reference.shape, .1)]).astype(np.float32)
    result = soft_head_metrics(reference, probability)
    assert result["WG"]["Dice"] == result["PZ"]["Dice"] == 1
    from nnunetv2.utilities.label_handling.label_handling import LabelManager
    hard = LabelManager(trainers.ANATOMY_LABELS, [1, 2, 4]).convert_probabilities_to_segmentation(probability)
    assert np.all(hard == 2) and not np.isin(hard, [1, 3, 5, 7]).any()
    probability[0] = .5
    assert soft_head_metrics(reference, probability)["WG"]["Dice"] == 0


def test_predicted_provenance_rejects_leakage_old_dataset_and_missing_cases():
    document = predicted_document()
    validate_predicted_prior_dataset(document, PREDICTED_DATASET)
    with pytest.raises(ValueError):
        validate_predicted_prior_dataset(document, "Dataset606_PICAI_Zonal")
    for mode in ("IN_SAMPLE_PRED", "OOF_PRED", "ORACLE_GT"):
        changed = copy.deepcopy(document)
        changed["predicted_anatomy_contract"]["cases"]["2_20"]["prior_mode"] = mode
        with pytest.raises(ValueError):
            validate_predicted_prior_dataset(changed, PREDICTED_DATASET)
    del document["predicted_anatomy_contract"]["cases"]["1_10"]
    with pytest.raises(ValueError):
        validate_predicted_prior_dataset(document, PREDICTED_DATASET)


CLASS_NAMES = [
    "nnUNetTrainerPICAI_FLCE_PositiveSampling_NoFFT",
    "nnUNetTrainerPICAI_DiceCE_PositiveSampling_NoFFT",
    *[f"nnUNetTrainerPICAI_{variant}_{loss}_NoFFT"
      for variant in ("NeutralFusion", "LesionFusion", "AnatomyLesionFusion")
      for loss in ("FLCE", "DiceCE")],
]


@pytest.mark.parametrize("class_name", CLASS_NAMES)
@pytest.mark.parametrize("real_architecture", [False, True], ids=["minimal-plans", "real-605-plans"])
def test_true_native_trainer_construct_initialize_forward_loss_backward(
    tmp_path, monkeypatch, synthetic_arch, class_name, real_architecture
):
    # Reading a JSON plan is permitted; every image/target and output is synthetic in tmp_path.
    plans_path = ROOT / "workdir/nnUNet_preprocessed/Dataset605_PICAI/nnUNetPlans.json"
    if not plans_path.is_file():
        pytest.skip("real Dataset605 plan unavailable")
    plans = json.loads(plans_path.read_text())
    config = plans["configurations"]["3d_fullres"]
    condition = getattr(trainers.PROJECT_TRAINERS[class_name], "fusion_condition", None)
    dataset = PREDICTED_DATASET if condition == "anatomy" else "Dataset605_PICAI"
    plans["dataset_name"] = dataset
    if condition == "anatomy":
        config["preprocessor_name"] = "PredictedAnatomyPreprocessor"
        config["normalization_schemes"] += ["NoNormalization"] * 3
        config["use_mask_for_norm"] += [False] * 3
    if not real_architecture:
        config["architecture"] = {
            "network_class_name": synthetic_arch["architecture_class_name"],
            "arch_kwargs": synthetic_arch["arch_init_kwargs"],
            "_kw_requires_import": synthetic_arch["arch_init_kwargs_req_import"],
        }
    doc = predicted_document() if condition == "anatomy" else {
        "channel_names": {"0000": "T2W", "0001": "ADC", "0002": "HBV"},
        "labels": {"background": 0, "lesion": 1}, "numTraining": 2,
    }
    native = importlib.import_module("nnunetv2.training.nnUNetTrainer.nnUNetTrainer")
    monkeypatch.setattr(native, "nnUNet_preprocessed", str(tmp_path / "preprocessed"))
    monkeypatch.setattr(native, "nnUNet_results", str(tmp_path / "results"))
    base = tmp_path / "preprocessed" / dataset
    folder = base / config["data_identifier"]; folder.mkdir(parents=True)
    (base / "splits_final.json").write_text(json.dumps([{"train": ["1_10"], "val": ["2_20"]}]))
    shape = (16, 64, 64) if real_architecture else (8, 16, 16)
    channels = len(doc["channel_names"])
    np.savez(folder / "1_10.npz", data=np.zeros((channels, 2, 2, 2), np.float32),
             seg=np.zeros((1, 2, 2, 2), np.int8))
    cls = trainers.resolve_trainer_class(class_name)
    trainer = cls(plans, "3d_fullres", 0, doc, device=torch.device("cpu"))
    trainer.initialize()  # Real nnU-Net __init__, optimizer, scheduler, loss and network.
    x = torch.randn(1, channels, *shape)
    if condition == "anatomy":
        x[:, 3:] = torch.rand_like(x[:, 3:])
    output = trainer.network(x)
    full = torch.zeros((1, 1, *shape), dtype=torch.long)
    full[:, :, 2:4, 4:6, 4:6] = 1
    targets = [F.interpolate(full.float(), size=o.shape[2:], mode="nearest").long() for o in output]
    loss = trainer.loss(output, targets)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in trainer.network.parameters())
    assert trainer.optimizer.__class__.__name__ == "SGD"
    assert trainer.lr_scheduler.__class__.__name__ == "PolyLRScheduler"
    assert trainer.dataset_class.__name__ == "nnUNetDatasetNumpy"
    # Native DS toggling and inference (no GT or auxiliary loss) remain valid.
    trainer.set_deep_supervision_enabled(False)
    with torch.no_grad():
        assert trainer.network(x).shape == (1, 2, *shape)


def test_supporting_classmethods_construct_real_network(synthetic_arch):
    for cls in trainers.SUPPORTING_TRAINERS:
        channels = 3 + cls.lesion_zone_channels
        network = cls.build_network_architecture(
            synthetic_arch["architecture_class_name"], synthetic_arch["arch_init_kwargs"],
            synthetic_arch["arch_init_kwargs_req_import"], channels, 2)
        x = torch.rand(1, channels, 8, 16, 16)
        outputs = network(x)
        sum(o.square().mean() for o in outputs).backward()


def test_parameter_report():
    models = {"A": nn.Conv3d(3, 2, 1)}
    for key, mode in (("B", "neutral"), ("C", "lesion"), ("D", "anatomy")):
        models[key] = ConditionedMultimodalNNUNet(nn.Conv3d(3, 2, 1), condition=mode)
    report = parameter_counts(models)
    assert report["A"]["delta"] == 0
    assert report["D"]["total"] > report["C"]["total"] > report["B"]["total"]


@pytest.mark.parametrize("transpose", [[0, 1, 2], [2, 0, 1]])
def test_prior_preprocessing_preserves_native_mri_and_geometry(synthetic_arch, transpose):
    from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
    from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager

    from zonal_reliability_fusion.nnunet.prior_preprocessor import PredictedAnatomyPreprocessor

    config = {"spacing": [3., .5, .5], "normalization_schemes": ["ZScoreNormalization"]*3,
              "use_mask_for_norm": [False]*3, "resampling_fn_data": "resample_data_or_seg_to_shape",
              "resampling_fn_data_kwargs": {"is_seg": False, "order": 3, "order_z": 0},
              "resampling_fn_seg": "resample_data_or_seg_to_shape",
              "resampling_fn_seg_kwargs": {"is_seg": True, "order": 1, "order_z": 0},
              "resampling_fn_probabilities": "resample_data_or_seg_to_shape",
              "resampling_fn_probabilities_kwargs": {"is_seg": False, "order": 1, "order_z": 0}}
    config["architecture"] = {}
    plans = PlansManager({"transpose_forward": transpose, "transpose_backward": np.argsort(transpose).tolist(),
                          "foreground_intensity_properties_per_channel": {str(i): {} for i in range(3)}})
    rng = np.random.default_rng(1)
    mri = rng.normal(size=(3, 4, 8, 8)).astype(np.float32)
    mri[:, :, :2] = 0
    prior = rng.uniform(size=mri.shape).astype(np.float32)
    data = np.concatenate((mri, prior))
    seg = np.zeros((1, 4, 8, 8), np.int8); seg[:, 1:3, 3:5, 3:5] = 1
    props = {"spacing": [3., .75, .75]}
    doc = {"labels": {"background": 0, "lesion": 1}}
    expected, expected_seg, expected_props = DefaultPreprocessor(False).run_case_npy(
        mri, seg, copy.deepcopy(props), plans, ConfigurationManager(config), doc)
    config["normalization_schemes"] += ["NoNormalization"]*3
    config["use_mask_for_norm"] += [False]*3
    actual, actual_seg, actual_props = PredictedAnatomyPreprocessor(False).run_case_npy(
        data, seg, copy.deepcopy(props), plans, ConfigurationManager(config), doc)
    assert np.array_equal(actual[:3], expected)
    assert np.array_equal(actual_seg, expected_seg)
    assert actual_props["bbox_used_for_cropping"] == expected_props["bbox_used_for_cropping"]
    assert actual[3:].shape[1:] == expected.shape[1:]
    assert np.isfinite(actual[3:]).all() and actual[3:].min() >= 0 and actual[3:].max() <= 1


def test_predicted_dataset_materialization_synthetic_geometry_and_no_overwrite(tmp_path):
    import pickle

    import SimpleITK as sitk

    from zonal_reliability_fusion.anatomy.contracts import ANATOMY_LABELS
    from zonal_reliability_fusion.anatomy.dataset import materialize_predicted_dataset

    source = tmp_path / "Dataset605_PICAI"; source.mkdir()
    (source / "imagesTr").mkdir(); (source / "labelsTr").mkdir()
    priors = tmp_path / "priors"; priors.mkdir()
    model = tmp_path / "model"; model.mkdir(); (model / "fold_0").mkdir()
    (model / "fold_0/checkpoint_final.pth").write_bytes(b"synthetic-test-not-model")
    (model / "dataset.json").write_text(json.dumps({"labels": ANATOMY_LABELS,
        "anatomy_contract": {"train_cases": ["1_10"]}}))
    (model / "plans.json").write_text(json.dumps({"dataset_name": "Dataset607_PICAI_Anatomy",
                                                "transpose_forward": [0, 1, 2]}))
    split = tmp_path / "splits.json"
    split.write_text(json.dumps([{"train": ["1_10"], "val": ["2_20"]}]))
    for cid in ("1_10", "2_20"):
        image = sitk.GetImageFromArray(np.ones((4, 4, 4), np.float32))
        for channel in range(3):
            sitk.WriteImage(image, str(source / "imagesTr" / f"{cid}_{channel:04d}.nii.gz"))
        sitk.WriteImage(sitk.Cast(image, sitk.sitkUInt8), str(source / "labelsTr" / f"{cid}.nii.gz"))
        probability = np.full((3, 4, 4, 4), .2, np.float32)
        np.savez(priors / f"{cid}.npz", probabilities=probability)
        sitk.WriteImage(sitk.GetImageFromArray(np.zeros((4, 4, 4), np.uint8)),
                        str(priors / f"{cid}.nii.gz"))
        props = {"spacing": [1., 1., 1.], "shape_before_cropping": [4, 4, 4],
                 "sitk_stuff": {"spacing": image.GetSpacing(), "origin": image.GetOrigin(),
                                "direction": image.GetDirection()}}
        (priors / f"{cid}.pkl").write_bytes(pickle.dumps(props))
    output = tmp_path / "raw" / PREDICTED_DATASET
    doc = materialize_predicted_dataset(source, priors, model, output, split, no_progress=True)
    assert doc["predicted_anatomy_contract"]["cases"]["2_20"]["prior_mode"] == "HELD_OUT_PRED"
    assert len(list((output / "imagesTr").glob("*.nii.gz"))) == 12
    with pytest.raises(ValueError, match="new Dataset608"):
        materialize_predicted_dataset(source, priors, model, output, split, no_progress=True)
    # Change prior physical grid: it must fail without modifying any original input.
    image.SetOrigin((5., 0., 0.))
    sitk.WriteImage(image, str(priors / "1_10.nii.gz"))
    with pytest.raises(ValueError, match="grid"):
        materialize_predicted_dataset(source, priors, model, tmp_path / "other" / PREDICTED_DATASET,
                                      split, no_progress=True)


def test_seeded_trainer_names_preserve_history_and_checkpoint_resolution():
    historical = dict(trainers.PROJECT_TRAINERS)
    base = trainers.nnUNetTrainerPICAI_FLCE_PositiveSampling_NoFFT
    seeded = trainers.seeded_trainer_class(base, 20261008)
    assert seeded.__name__ == base.__name__ + "_Seed20261008"
    assert seeded is trainers.resolve_trainer_class(seeded.__name__)
    assert seeded is trainers.seeded_trainer_class(base, 20261008)
    assert issubclass(seeded, base)
    assert trainers.seeded_trainer_class(base, 20261009) is not seeded
    assert historical == trainers.PROJECT_TRAINERS
    for invalid in (None, True, -1, 2**32):
        with pytest.raises(ValueError):
            trainers.seeded_trainer_class(base, invalid)
    with pytest.raises(ValueError):
        trainers.seeded_trainer_class(trainers.SUPPORTING_TRAINERS[0], 1)


def test_empty_weight_region_has_null_statistics():
    assert all(value is None for value in fusion_weight_statistics(torch.empty(3, 0)).values())
