"""V2 synthetic construction, mechanism, backward, registry and inference contracts."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from zonal_reliability_fusion.nnunet import trainers
from zonal_reliability_fusion.v2.anatomy.pyramid import AnatomyPyramid
from zonal_reliability_fusion.v2.fusion.adaptive_fusion import AdaptiveFusion
from zonal_reliability_fusion.v2.fusion.anatomy_conditioned_fusion import AnatomyConditionedFusion
from zonal_reliability_fusion.v2.fusion.simple_fusion import ConcatFusion, SimpleFusion

ROOT = Path(__file__).resolve().parents[3]
V2 = trainers.ACTIVE_V2_TRAINERS


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def build(cls, synthetic_arch, deep=True):
    return cls.build_network_architecture(**synthetic_arch,
                                         num_input_channels=6 if cls.anatomy_conditioned else 3,
                                         num_output_channels=2, enable_deep_supervision=deep)


def image(cls, shape=(8, 16, 16)):
    x = torch.randn(1, 6 if cls.anatomy_conditioned else 3, *shape)
    if cls.anatomy_conditioned:
        x[:, 3:] = torch.rand_like(x[:, 3:])
    return x


@pytest.mark.parametrize("cls", V2, ids=lambda c: c.__name__)
def test_every_step_shapes_gradients_counts_and_checkpoint_reconstruction(cls, synthetic_arch):
    net = build(cls, synthetic_arch)
    x = image(cls)
    if cls.architecture_family == "multi_encoder":
        assert len(net.encoders) == 3
        params = [set(map(id, e.parameters())) for e in net.encoders]
        assert not params[0] & params[1] and not params[1] & params[2]
        for encoder in net.encoders:
            assert [tuple(f.shape) for f in encoder(x[:, :1])] == [
                (1, 4, 8, 16, 16), (1, 8, 4, 8, 8), (1, 8, 2, 4, 4)]
        net.export_fusion_weights = True
        assert net.compute_conv_feature_map_size(x.shape[2:]) > 0
    output = net(x)
    assert [tuple(o.shape) for o in output] == [(1, 2, 8, 16, 16), (1, 2, 4, 8, 8)]
    loss = sum(o.square().mean() for o in output)
    if cls.coarse_head:
        assert net.coarse_logits.shape == (1, 1, 2, 4, 4)
        loss = loss + net.coarse_logits.square().mean()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
    if cls.architecture_family == "multi_encoder":
        for e in net.encoders:
            assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in e.parameters())
        for weights in net.last_fusion_weights.values():
            assert torch.allclose(weights.sum(1), torch.ones_like(weights[:, 0]), atol=1e-6)
    clone = build(cls, synthetic_arch)
    clone.load_state_dict(net.state_dict(), strict=True)
    assert sum(p.numel() for p in net.parameters()) == sum(p.numel() for p in clone.parameters())
    net.decoder.deep_supervision = clone.decoder.deep_supervision = False
    net.eval(); clone.eval()
    with torch.no_grad():
        assert net(x).shape == (1, 2, 8, 16, 16)
        torch.testing.assert_close(net(x), clone(x))


def test_fusion_anatomy_pyramid_and_context_influence():
    features = [torch.randn(1, 4, 4, 4, 4) for _ in range(3)]
    torch.testing.assert_close(SimpleFusion()(features), sum(features) / 3)
    pyramid = AnatomyPyramid(2)
    contexts = pyramid(torch.rand(1, 3, 8, 8, 8), [(4, 4, 4), (2, 2, 2)])
    assert [tuple(c.shape) for c in contexts] == [(1, 8, 4, 4, 4), (1, 8, 2, 2, 2)]
    for fusion, context in [(AdaptiveFusion(4), None), (AnatomyConditionedFusion(4), contexts[0])]:
        fused, weights = fusion(features, context)
        assert fused.shape == features[0].shape and torch.isfinite(fused).all()
        torch.testing.assert_close(weights.sum(1), torch.ones_like(weights[:, 0]))
    context = contexts[0].detach().requires_grad_()
    _, weights = AnatomyConditionedFusion(4)(features, context)
    weights[:, 0].sum().backward()
    assert context.grad.abs().sum() > 0


@pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan"), float("inf")])
def test_prior_fail_closed(bad, synthetic_arch):
    cls = V2[3]; net = build(cls, synthetic_arch)
    x = image(cls); x[:, 3, 0, 0, 0] = bad
    with pytest.raises(ValueError): net(x)
    with pytest.raises(ValueError): AnatomyPyramid(1)(x[:, 3:], [(2, 2, 2)])


def test_input_and_decoder_fail_closed(synthetic_arch):
    net = build(V2[1], synthetic_arch)
    for x in [torch.rand(1, 6, 8, 16, 16), torch.rand(1, 3, 7, 16, 16),
              torch.zeros(1, 3, 8, 16, 16, dtype=torch.long)]:
        with pytest.raises(ValueError): net(x)
    bad = copy.deepcopy(synthetic_arch); bad["arch_init_kwargs"]["conv_op"] = "torch.nn.Conv2d"
    with pytest.raises(ValueError): build(V2[1], bad)
    with pytest.raises(ValueError): SimpleFusion()([torch.rand(1, 4, 4, 4, 4)])


def test_zero_init_refiner_equivalence_and_phase_gradients(synthetic_arch):
    cls = trainers.nnUNetTrainerPICAI_V2_FullScheduled_DiceCE_NoFFT
    net = build(cls, synthetic_arch, deep=False)
    x = image(cls)
    net.set_phase("V2-P2"); base = net(x)
    net.set_phase("V2-P3"); refined = net(x)
    torch.testing.assert_close(refined, base, rtol=0, atol=0)
    refined.square().mean().backward()
    assert net.refiner.output.weight.grad.abs().sum() > 0
    with torch.no_grad(): net.refiner.output.bias.fill_(0.1)
    assert not torch.equal(net(x), base)
    net.set_phase("V2-P1"); net.zero_grad(set_to_none=True)
    net(x).square().mean().backward()
    assert net.coarse_logits is None
    assert all(p.grad is None for p in net.fusions.parameters())
    assert all(p.grad is None for p in net.refiner.parameters())
    assert [cls.phase_for_epoch(e) for e in (0, 199, 200, 599, 600, 999)] == [
        "V2-P1", "V2-P1", "V2-P2", "V2-P2", "V2-P3", "V2-P3"]


def load_entry():
    spec = importlib.util.spec_from_file_location("train_v2_test", ROOT / "scripts/train/train_nnunet.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_registry_cli_isolation_and_v1_legacy_compatibility():
    entry = load_entry()
    assert set(entry.ACTIVE_VARIANTS.values()) == {c.__name__ for c in V2}
    assert not set(V2) & set(trainers.FROZEN_V1_TRAINERS)
    assert set(entry.VARIANT_TO_TRAINER.values()) == set(trainers.PROJECT_TRAINERS)
    for alias in ("neutral_fusion_dicece", "fusion", "B", "D", "image_gate"):
        with pytest.raises(SystemExit): entry.build_parser().parse_args([alias, "605", "3d_fullres", "0"])
    assert entry.build_parser(include_v1=True).parse_args(["neutral_fusion_dicece", "605", "3d_fullres", "0"]).variant == "neutral_fusion_dicece"
    from zonal_reliability_fusion.multimodal.conditioned_fusion import (
        ConditionedMultimodalNNUNet as old,
    )
    from zonal_reliability_fusion.v1.multimodal.shallow_conditioned_fusion import (
        ConditionedMultimodalNNUNet as frozen,
    )
    assert old is frozen and old.__module__ == "zonal_reliability_fusion.multimodal.conditioned_fusion"
    for cls in (*V2, *trainers.FROZEN_V1_TRAINERS, *trainers.LEGACY_TRAINERS):
        assert trainers.resolve_trainer_class(cls.__name__) is cls
    for cls in (*V2, *trainers.FROZEN_V1_TRAINERS):
        if "AnatomyJoint" not in cls.__name__:
            seeded = trainers.seeded_trainer_class(cls, 20261008)
            assert trainers.resolve_trainer_class(seeded.__name__) is seeded
    import nnunetv2
    assert Path(nnunetv2.__file__).resolve().is_relative_to(ROOT / "third_party/nnUNet")
    import sys
    assert not any("Prostate-Lesion-Segmentation" in p for p in sys.path)


@pytest.mark.parametrize("cls", V2, ids=lambda c: c.__name__)
def test_native_constructor_optimizer_loss_and_resume_phase(tmp_path, monkeypatch, synthetic_arch, cls):
    from zonal_reliability_fusion.nnunet.v2.trainers import _V2Mixin

    native = __import__("nnunetv2.training.nnUNetTrainer.nnUNetTrainer", fromlist=["nnUNetTrainer"])
    monkeypatch.setattr(native, "nnUNet_preprocessed", str(tmp_path / "preprocessed"))
    monkeypatch.setattr(native, "nnUNet_results", str(tmp_path / "results"))
    # Metadata JSON allowed; all arrays and paths here are synthetic.
    path = ROOT / "workdir/nnUNet_preprocessed/Dataset605_PICAI/nnUNetPlans.json"
    if not path.is_file(): pytest.skip("local plan unavailable")
    plans = json.loads(path.read_text())
    config = plans["configurations"]["3d_fullres"]
    config["architecture"] = {"network_class_name": synthetic_arch["architecture_class_name"],
                              "arch_kwargs": synthetic_arch["arch_init_kwargs"],
                              "_kw_requires_import": synthetic_arch["arch_init_kwargs_req_import"]}
    dataset = "Dataset608_PICAI_PredictedAnatomy" if cls.anatomy_conditioned else "Dataset605_PICAI"
    plans["dataset_name"] = dataset
    doc = {"channel_names": {"0": "T2W", "1": "ADC", "2": "HBV"},
           "labels": {"background": 0, "lesion": 1}, "numTraining": 2}
    if cls.anatomy_conditioned:
        config["preprocessor_name"] = "PredictedAnatomyPreprocessor"
        config["normalization_schemes"] += ["NoNormalization"] * 3
        config["use_mask_for_norm"] += [False] * 3
        doc["channel_names"].update({"3": "noNorm", "4": "noNorm", "5": "noNorm"})
        records = {}
        for cid, mode in [("1_10", "IN_SAMPLE_PRED"), ("2_20", "HELD_OUT_PRED")]:
            records[cid] = {"case_id": cid, "prior_mode": mode, "checkpoint": "synthetic.pth",
                            "trainer": "Stage1", "source_split": "synthetic", "geometry": {"size": [1, 1, 1]},
                            "anatomy_training_patient_ids": ["1"]}
        doc["predicted_anatomy_contract"] = {"source": "predicted_prior", "channel_order": ["WG", "PZ", "TZ"],
                                            "train_cases": ["1_10"], "val_cases": ["2_20"], "cases": records}
    base = tmp_path / "preprocessed" / dataset
    (base / config["data_identifier"]).mkdir(parents=True)
    (base / "splits_final.json").write_text(json.dumps([{"train": ["1_10"], "val": ["2_20"]}]))
    # Native initialize only probes synthetic file suffixes, no dataloaders run.
    import numpy as np
    np.savez(base / config["data_identifier"] / "1_10.npz", data=np.zeros((3, 2, 2, 2)), seg=np.zeros((1, 2, 2, 2)))
    trainer = cls(plans, "3d_fullres", 0, doc, device=torch.device("cpu"))
    assert _V2Mixin._validate_v2_contract(trainer) == (["1_10"], ["2_20"])
    trainer.initialize()
    if cls.architecture_family == "multi_encoder": trainer.network.set_phase("V2-P3" if cls.refiner else "V2-P2")
    output = trainer.network(image(cls))
    target = torch.zeros((1, 1, 8, 16, 16), dtype=torch.long)
    target[:, :, 2, 3, 4] = 1
    targets = [F.interpolate(target.float(), size=o.shape[2:], mode="nearest").long() for o in output]
    loss = trainer.loss(output, targets); assert torch.isfinite(loss); loss.backward()
    assert trainer.optimizer.__class__.__name__ == "SGD"
    assert trainer.lr_scheduler.__class__.__name__ == "PolyLRScheduler"
    if cls.coarse_head: assert trainer.network.coarse_head.weight.grad.abs().sum() > 0
    if cls.phase_scheduled:
        trainer.current_epoch = 650; trainer.on_epoch_start(); assert trainer.network.phase == "V2-P3"
        checkpoint = tmp_path / "synthetic_scheduled.pth"
        trainer.save_checkpoint(str(checkpoint))
        trainer.network.set_phase("V2-P1")
        trainer.load_checkpoint(str(checkpoint))
        assert trainer.current_epoch == 651 and trainer.network.phase == "V2-P3"
    trainer.set_deep_supervision_enabled(False)
    assert trainer.network(image(cls)).shape == (1, 2, 8, 16, 16)


@pytest.mark.parametrize("cls", V2, ids=lambda c: c.__name__)
def test_actual_plan_shape_contract(cls):
    dataset = "Dataset608_PICAI_PredictedAnatomy" if cls.anatomy_conditioned else "Dataset605_PICAI"
    path = ROOT / "workdir/nnUNet_preprocessed" / dataset / "nnUNetPlans.json"
    if not path.is_file(): pytest.skip("local plan unavailable")
    arch = json.loads(path.read_text())["configurations"]["3d_fullres"]["architecture"]
    net = cls.build_network_architecture(arch["network_class_name"], arch["arch_kwargs"],
                                        arch["_kw_requires_import"], 6 if cls.anatomy_conditioned else 3, 2, False)
    x = image(cls, (16, 64, 64))
    output = net(x); assert output.shape == (1, 2, 16, 64, 64)
    output.square().mean().backward()
    assert sum(p.numel() for p in net.parameters()) > 0
    if cls.architecture_family == "multi_encoder":
        assert net.selected_scales == (4, 5, 6)


@pytest.mark.parametrize("variant", list(load_entry().ACTIVE_VARIANTS))
def test_v2_run_config_records_actual_version_mechanisms_and_schedule(variant, synthetic_arch):
    from types import SimpleNamespace

    entry = load_entry(); cls = entry.resolve_trainer_class(variant)
    configuration = {"architecture": {"network_class_name": synthetic_arch["architecture_class_name"],
                                      "arch_kwargs": synthetic_arch["arch_init_kwargs"]},
                     "patch_size": [8, 16, 16], "batch_size": 2, "spacing": [3., .5, .5]}
    plans = {"dataset_name": "synthetic", "configurations": {"3d_fullres": configuration}}
    args = SimpleNamespace(variant=variant, configuration="3d_fullres", fold="0", seed=20261008)
    payload = entry.build_run_config(args, cls, plans, {"channel_names": {"0": "synthetic"}})
    assert payload["research_version"] == "V2"
    assert payload["architecture_family"] == cls.architecture_family
    for field in ("fusion_type", "anatomy_conditioned", "coarse_head", "refiner"):
        assert payload[field] == getattr(cls, field)
    assert payload["phase_schedule"] == ([200, 600] if cls.phase_scheduled else None)
    assert payload["concat_fusion"] == getattr(cls, "concat_fusion", False)
    if payload["concat_fusion"]:
        assert payload["fusion_projection_kernel"] == [3, 3, 3]
        assert payload["fusion_projection_scales"] == [0, 1, 2]
        assert payload["policy_zero_initialized"] == (cls.fusion_type != "concat")
        assert payload["policy_feature_scale"] == (None if cls.fusion_type == "concat" else 3.0)
    assert "stem_channels" not in str(payload)


@pytest.mark.parametrize("flags, error", [
    ([], "requires --seed"),
    (["--seed", "1", "--seeded-output", "--run-config", "old.json"], "filename"),
    (["--seed", "1", "--seeded-output", "--run-config", "V2_test.json", "--roi-set", "old.json"], "does not mix"),
])
def test_v2_cli_rejects_missing_provenance_and_unplanned_strategies_before_training(flags, error):
    with pytest.raises(SystemExit, match=error):
        load_entry().main(["v2_multi_encoder_simple", "605", "3d_fullres", "0", *flags])


def test_coarse_supervision_preserves_tiny_positive_and_rejects_invalid_target(synthetic_arch):
    from zonal_reliability_fusion.v2.losses.coarse_loss import CoarseAuxiliaryLoss

    net = build(V2[4], synthetic_arch, deep=False)
    output = net(image(V2[4]))
    criterion = CoarseAuxiliaryLoss(lambda o, t: o.square().mean(), net)
    target = torch.zeros(1, 1, 8, 16, 16)
    target[:, :, 1, 1, 1] = 1
    coarse = F.adaptive_max_pool3d(target, net.coarse_logits.shape[2:])
    assert coarse.sum() == 1
    criterion(output, target).backward()
    assert net.coarse_head.weight.grad.abs().sum() > 0
    target[:, :, 1, 1, 1] = -1
    with pytest.raises(ValueError, match="binary lesion GT"):
        criterion(output, target)


@pytest.mark.parametrize("cls", V2, ids=lambda c: c.__name__)
def test_native_sliding_window_on_synthetic_tensor_only(cls, synthetic_arch):
    from types import SimpleNamespace

    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    net = build(cls, synthetic_arch, deep=False)
    predictor = nnUNetPredictor(use_gaussian=False, use_mirroring=False,
                               perform_everything_on_device=False, device=torch.device("cpu"),
                               allow_tqdm=False)
    predictor.network = net
    predictor.configuration_manager = SimpleNamespace(patch_size=[8, 16, 16])
    predictor.label_manager = SimpleNamespace(num_segmentation_heads=2)
    predictor.allowed_mirroring_axes = None
    # Multiple tiles exercise native aggregation without any image/file I/O.
    x = image(cls, shape=(8, 24, 24))[0]
    output = predictor.predict_sliding_window_return_logits(x)
    assert output.shape == (2, 8, 24, 24) and torch.isfinite(output).all()
    assert torch.isfinite(output.softmax(0)).all()


@pytest.mark.parametrize("category", ["run_configs", "predictions", "metrics", "reports"])
def test_v2_output_subtrees_reject_historical_paths_traversal_and_symlinks(tmp_path, monkeypatch, category):
    from zonal_reliability_fusion.nnunet import runtime

    monkeypatch.setattr(runtime, "PROJECT_ROOT", tmp_path)
    root = runtime.v2_output_root(category)
    assert runtime.require_v2_output_path(root / "V2_example", category) == root / "V2_example"
    historical = tmp_path / "outputs" / category
    historical.mkdir(parents=True)
    for path in (historical / "V2_example", root / "../../old", root):
        with pytest.raises(ValueError): runtime.require_v2_output_path(path, category)
    root.mkdir(parents=True)
    (root / "escape").symlink_to(historical, target_is_directory=True)
    with pytest.raises(ValueError): runtime.require_v2_output_path(root / "escape" / "V2_example", category)


def test_v2_training_sets_both_native_result_globals_without_writing(tmp_path, monkeypatch):
    import importlib

    import nnunetv2.paths as paths

    from zonal_reliability_fusion.nnunet import runtime

    native = importlib.import_module("nnunetv2.training.nnUNetTrainer.nnUNetTrainer")
    monkeypatch.setattr(runtime, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(paths, "nnUNet_results", str(tmp_path / "historical"))
    monkeypatch.setattr(native, "nnUNet_results", str(tmp_path / "historical"))
    monkeypatch.setenv("nnUNet_results", str(tmp_path / "historical"))
    result = runtime.configure_v2_training_outputs()
    assert result == tmp_path / "outputs/v2/nnUNet_results"
    assert paths.nnUNet_results == native.nnUNet_results == str(result)
    assert not result.exists()  # Preflight is path/global configuration only.


def test_v2_training_rejects_flat_run_config_before_any_native_call():
    with pytest.raises(SystemExit, match="child of"):
        load_entry().main(["v2_multi_encoder_simple", "605", "3d_fullres", "0",
                           "--seed", "1", "--seeded-output", "--run-config",
                           str(ROOT / "outputs/run_configs/V2_wrong.json")])


def load_script(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_v2_prediction_destination_protection_preserves_existing_files(tmp_path, monkeypatch):
    from zonal_reliability_fusion.nnunet import runtime

    monkeypatch.setattr(runtime, "PROJECT_ROOT", tmp_path)
    module = load_script("scripts/inference/predict_nnunet.py", "predict_v2_paths")
    model = tmp_path / "nnUNetTrainerPICAI_V2_MultiEncoder_DiceCE_NoFFT__nnUNetPlans__3d_fullres"
    arguments = ["-m", str(model), "-i", str(tmp_path / "synthetic_inputs"), "-o"]
    with pytest.raises(ValueError, match="child of"):
        module.v2_prediction_output_preflight([*arguments, str(tmp_path / "outputs/predictions/V2_bad")])
    target = runtime.v2_output_root("predictions") / "V2_run"
    module.v2_prediction_output_preflight([*arguments, str(target)])
    target.mkdir(parents=True)
    protected = target / "synthetic.txt"; protected.write_text("preserve")
    with pytest.raises(ValueError, match="already exists"):
        module.v2_prediction_output_preflight([*arguments, str(target)])
    assert protected.read_text() == "preserve"
    # Historical prediction behavior remains under its existing shared/native contract.
    module.v2_prediction_output_preflight(["-m", str(tmp_path / "old_model"), "-i", "synthetic", "-o", str(target)])


def test_evaluation_cannot_publish_v2_to_historical_area_or_mislabel_v1(tmp_path, monkeypatch):
    from zonal_reliability_fusion.nnunet import runtime

    monkeypatch.setattr(runtime, "PROJECT_ROOT", tmp_path)
    module = load_script("scripts/evaluate_segmentation.py", "evaluate_v2_paths")
    model = runtime.v2_output_root("nnUNet_results") / "synthetic/fold_0"
    models = [("V2_B", model)]
    with pytest.raises(module.EvaluationError, match="child of"):
        module.check_versioned_evaluation_output(models, tmp_path / "outputs/metrics/V2_bad.json")
    for category in ("metrics", "reports"):
        assert module.check_versioned_evaluation_output(models, runtime.v2_output_root(category) / "V2_good.json") == {"V2_B": True}
    with pytest.raises(module.EvaluationError, match="V2 label"):
        module.check_versioned_evaluation_output([("V2_B", tmp_path / "old_model")], None)
    with pytest.raises(module.EvaluationError, match="explicit version"):
        module.check_versioned_evaluation_output([("B", model)], None)


CONCAT_TRAINERS = (
    trainers.nnUNetTrainerPICAI_V2_ConcatFusion_DiceCE_NoFFT,
    trainers.nnUNetTrainerPICAI_V2_ConcatAdaptiveFusion_DiceCE_NoFFT,
    trainers.nnUNetTrainerPICAI_V2_ConcatAnatomyFusion_DiceCE_NoFFT,
)


def test_concat_family_uses_identical_projection_and_uniform_initial_predictions(synthetic_arch):
    base = build(CONCAT_TRAINERS[0], synthetic_arch)
    x = image(CONCAT_TRAINERS[0])
    for cls in CONCAT_TRAINERS:
        net = build(cls, synthetic_arch)
        loaded = net.load_state_dict(base.state_dict(), strict=False)
        assert not loaded.unexpected_keys
        assert all(".policy." in key or key.startswith("anatomy_pyramid.") for key in loaded.missing_keys)
        assert list(net.fusions) == ["0", "1", "2"]
        for stage, fusion in net.fusions.items():
            assert isinstance(fusion, ConcatFusion)
            assert fusion.block[0].kernel_size == (3, 3, 3)
            assert fusion.block[0].in_channels == 3 * fusion.block[0].out_channels
            assert (fusion.policy is not None) == (int(stage) in net.selected_scales and cls.fusion_type != "concat")
        xx = torch.cat([x, torch.rand_like(x)], 1) if cls.anatomy_conditioned else x
        net.export_fusion_weights = True
        for actual, expected in zip(net(xx), base(x), strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for weights in net.last_fusion_weights.values():
            torch.testing.assert_close(weights, torch.full_like(weights, 1/3), rtol=0, atol=0)
        with pytest.raises(ValueError, match="without phase scheduling"):
            net.set_phase("V2-P1")


def test_concat_matched_seed_keeps_common_parameters_identical(synthetic_arch):
    common = None
    for cls in CONCAT_TRAINERS:
        torch.manual_seed(20261008)
        net = build(cls, synthetic_arch)
        state = {k: v for k, v in net.state_dict().items()
                 if ".policy." not in k and not k.startswith("anatomy_pyramid.")}
        if common is None:
            common = state
        else:
            assert state.keys() == common.keys()
            for key in state:
                torch.testing.assert_close(state[key], common[key], rtol=0, atol=0)


@pytest.mark.parametrize("cls", CONCAT_TRAINERS[1:], ids=lambda c: c.__name__)
def test_concat_policy_last_layer_learns_from_zero_initialization(cls, synthetic_arch):
    net = build(cls, synthetic_arch)
    loss = sum(o.square().mean() for o in net(image(cls)))
    loss.backward()
    for fusion in net.fusions.values():
        assert fusion.block[0].weight.grad.abs().sum() > 0
        if fusion.policy is not None:
            assert fusion.policy.controller[-1].weight.grad.abs().sum() > 0


def test_concat_anatomy_controller_can_use_context_after_initialization():
    policy = AnatomyConditionedFusion(4)
    fusion = ConcatFusion(4, norm_op=torch.nn.InstanceNorm3d,
                          norm_op_kwargs={"affine": True}, nonlin=torch.nn.LeakyReLU,
                          nonlin_kwargs={"inplace": True}, conv_bias=True, policy=policy)
    torch.nn.init.normal_(policy.controller[-1].weight, std=0.1)
    context = torch.rand(1, 8, 4, 4, 4, requires_grad=True)
    features = [torch.randn(1, 4, 4, 4, 4) for _ in range(3)]
    _, weights = fusion(features, context)
    weights[:, 0].sum().backward()
    assert context.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="missing or invalid"):
        fusion(features)


@pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan"), float("inf")])
def test_concat_anatomy_rejects_bad_prior(bad, synthetic_arch):
    net = build(CONCAT_TRAINERS[2], synthetic_arch)
    x = image(CONCAT_TRAINERS[2]); x[:, 3, 0, 0, 0] = bad
    with pytest.raises(ValueError): net(x)


def test_concat_baseline_rejects_wrong_channels_and_preserves_mean_variant(synthetic_arch):
    net = build(CONCAT_TRAINERS[0], synthetic_arch)
    fusion = net.fusions["0"]
    with pytest.raises(ValueError, match="plans-defined channels"):
        fusion([torch.randn(1, 3, 4, 4, 4) for _ in range(3)])
    with pytest.raises(ValueError, match="must not receive anatomy"):
        fusion([torch.randn(1, 4, 4, 4, 4) for _ in range(3)], torch.rand(1, 8, 4, 4, 4))
    old = build(trainers.nnUNetTrainerPICAI_V2_MultiEncoder_DiceCE_NoFFT, synthetic_arch)
    assert not old.fusions and old.fusion_type == "simple"
    assert not any("fusions." in key for key in old.state_dict())
