"""Plan3DConfig 与实验配置校验测试（纯 CPU，不读取病例数据）。"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from zonal_reliability_fusion.config import (
    ExperimentConfigError,
    load_experiment_config,
    load_plan,
    read_split_summary,
    validate_against_plan,
)
from zonal_reliability_fusion.config.plans import PlanConfigError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REAL_PLANS = PROJECT_ROOT / "workdir/nnUNet_preprocessed/Dataset605_PICAI/nnUNetPlans.json"


def write_json(path: Path, doc) -> Path:
    path.write_text(json.dumps(doc, indent=2))
    return path


# ------------------------------------------------------------------ plan 解析
def test_mini_plan_basic_fields(mini_plan):
    assert mini_plan.n_stages == 3
    assert mini_plan.patch_size == (8, 16, 16)
    assert mini_plan.spacing == (3.0, 0.5, 0.5)
    assert mini_plan.batch_size == 2
    assert mini_plan.features_per_stage == (4, 8, 8)
    assert mini_plan.kernel_sizes[0] == (1, 3, 3)
    assert mini_plan.strides[1] == (2, 2, 2)
    assert mini_plan.n_conv_per_stage_decoder == (1, 1)
    assert mini_plan.conv_bias is True
    assert mini_plan.norm_eps == pytest.approx(1e-5)
    assert mini_plan.negative_slope == pytest.approx(0.01)


def test_mini_plan_stage_shapes_and_ds_order(mini_plan):
    assert mini_plan.stage_shapes() == ((8, 16, 16), (4, 8, 8), (2, 4, 4))
    ds_shapes = mini_plan.decoder_output_shapes()  # 高分辨率 → 低分辨率
    assert ds_shapes == ((8, 16, 16), (4, 8, 8))
    assert all(a[0] >= b[0] for a, b in zip(ds_shapes, ds_shapes[1:]))
    assert len(ds_shapes) == mini_plan.n_stages - 1


def test_all_stage_shapes_are_positive(mini_plan):
    assert all(min(shape) >= 1 for shape in mini_plan.stage_shapes())


@pytest.mark.parametrize(
    "mutate",
    [
        lambda doc: doc["configurations"]["3d_fullres"]["architecture"]["arch_kwargs"].update({"features_per_stage": [4, 8]}),
        lambda doc: doc["configurations"]["3d_fullres"]["architecture"]["arch_kwargs"]["kernel_sizes"].__setitem__(1, [2, 3, 3]),
        lambda doc: doc["configurations"]["3d_fullres"]["architecture"]["arch_kwargs"].update({"conv_op": "torch.nn.modules.conv.Conv2d"}),
        lambda doc: doc["configurations"]["3d_fullres"].update({"patch_size": [0, 16, 16]}),
        lambda doc: doc["configurations"]["3d_fullres"]["architecture"]["arch_kwargs"].update({"n_conv_per_stage_decoder": [1]}),
        lambda doc: doc["configurations"]["3d_fullres"]["architecture"]["arch_kwargs"].update({"nonlin": "torch.nn.ReLU"}),
    ],
)
def test_illegal_plans_rejected(tmp_path: Path, mini_plan_doc: dict, mutate):
    bad = copy.deepcopy(mini_plan_doc)
    mutate(bad)
    with pytest.raises(PlanConfigError):
        load_plan(write_json(tmp_path / "bad.json", bad), "3d_fullres")


def test_missing_configuration_rejected(tmp_path: Path, mini_plan_doc: dict):
    bad = copy.deepcopy(mini_plan_doc)
    bad["configurations"]["2d"] = bad["configurations"].pop("3d_fullres")
    with pytest.raises(PlanConfigError):
        load_plan(write_json(tmp_path / "bad.json", bad), "3d_fullres")


@pytest.mark.skipif(not REAL_PLANS.is_file(), reason="真实 nnUNetPlans.json 不存在")
def test_real_plans_fields_read_only():
    """对真实 plan 只做 JSON 字段读取与断言（不读取任何 B2ND / 病例）。"""
    plan = load_plan(REAL_PLANS, "3d_fullres", dataset_name="Dataset605_PICAI")
    assert plan.spacing == (3.0, 0.5, 0.5)
    assert plan.patch_size == (16, 320, 320)
    assert plan.batch_size == 2
    assert plan.n_stages == 7
    assert plan.features_per_stage == (32, 64, 128, 256, 320, 320, 320)
    assert plan.kernel_sizes == ((1, 3, 3), (1, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3))
    assert plan.strides == ((1, 1, 1), (1, 2, 2), (1, 2, 2), (2, 2, 2), (2, 2, 2), (1, 2, 2), (1, 2, 2))
    assert plan.n_conv_per_stage == (2,) * 7
    assert plan.n_conv_per_stage_decoder == (2,) * 6
    assert plan.batch_dice is False
    assert plan.stage_shapes()[0] == (16, 320, 320)
    assert plan.stage_shapes()[-1] == (4, 5, 5)
    ds_shapes = plan.decoder_output_shapes()
    assert len(ds_shapes) == 6
    assert ds_shapes[0] == (16, 320, 320)
    assert ds_shapes[-1] == (4, 10, 10)


# ------------------------------------------------------------------ 实验配置校验
def _config_doc(
    plan_path: Path, split_path: Path, pre_dir: Path, output_root: Path, *, n_val: int = 2, n_pos: int = 1, n_neg: int = 1
) -> dict:
    return {
        "experiment": {"name": "unit_test", "model_id": "M0", "seed": 0, "fold": 0},
        "paths": {
            "plans": str(plan_path),
            "preprocessed_dataset": str(pre_dir),
            "splits": str(split_path),
            "output_root": str(output_root),
        },
        "model": {"in_channels": 3, "num_classes": 2, "deep_supervision": True, "configuration": "3d_fullres"},
        "data": {
            "modality_order": ["T2W", "ADC", "HBV"],
            "batch_size": 2,
            "patch_size": [8, 16, 16],
            "oversample_foreground": 0.33,
            "num_workers": 0,
            "pin_memory": False,
        },
        "loss": {"name": "focal_ce", "focal_weight": 0.5, "ce_weight": 0.5, "gamma": 2.0, "smooth": 1e-5, "batch_dice": False},
        "optimizer": {"name": "SGD", "lr": 0.01, "momentum": 0.99, "nesterov": True, "weight_decay": 3e-5},
        "scheduler": {"name": "PolyLR", "power": 0.9},
        "training": {
            "max_epochs": 2,
            "iterations_per_epoch": 2,
            "diagnostic_validation_iterations": 2,
            "amp": False,
            "grad_clip": 12.0,
            "checkpoint_every": 1,
            "progress": False,
        },
        "validation": {
            "mode": "full_volume",
            "every_n_epochs": 1,
            "checkpoint_metric": "val_positive_casewise_dice_mean",
            "maximize": True,
            "expected_cases": n_val,
            "expected_positive_cases": n_pos,
            "expected_negative_cases": n_neg,
            "step_fraction": 0.5,
            "gaussian": True,
            "mirror_tta": False,
            "sliding_window_batch_size": 1,
            "inner_patch_progress": False,
            "save_case_metrics": True,
        },
        "early_stopping": {"enabled": True, "min_epochs": 1, "patience": 1, "min_delta": 1e-4},
        "inference": {"step_fraction": 0.5, "gaussian": True, "mirror_tta": False},
    }


def _load_cfg(tmp_path: Path, doc: dict):
    return load_experiment_config(write_json(tmp_path / "cfg.json", doc))


def test_experiment_config_ok(tmp_path: Path, mini_plan):
    split_path = write_json(tmp_path / "splits_final.json", [{"train": ["a", "b"], "val": ["c", "d"]}])
    cfg = _load_cfg(tmp_path, _config_doc(Path(mini_plan.source_path), split_path, tmp_path, tmp_path / "out"))
    warnings = validate_against_plan(cfg, mini_plan, project_root=tmp_path, require_data=False)
    assert isinstance(warnings, list)
    assert any("checkpoint 指标" in w for w in warnings)
    assert read_split_summary(split_path) == {
        "path": str(split_path),
        "n_folds": 1,
        "n_train": 2,
        "n_val": 2,
    }


@pytest.mark.parametrize(
    "mutate",
    [
        lambda doc: doc["data"].update({"modality_order": ["T2W", "HBV", "ADC"]}),
        lambda doc: doc["data"].update({"patch_size": [4, 8, 8]}),
        lambda doc: doc["data"].update({"batch_size": 4}),
        lambda doc: doc["experiment"].update({"fold": 1}),
        lambda doc: doc["model"].update({"in_channels": 4}),
        lambda doc: doc["loss"].update({"batch_dice": True}),
        lambda doc: doc["model"].update({"configuration": "2d"}),
    ],
)
def test_experiment_config_conflicts_rejected(tmp_path: Path, mini_plan, mutate):
    split_path = write_json(tmp_path / "splits_final.json", [{"train": ["a", "b"], "val": ["c", "d"]}])
    doc = _config_doc(Path(mini_plan.source_path), split_path, tmp_path, tmp_path / "out")
    mutate(doc)
    cfg = _load_cfg(tmp_path, doc)
    with pytest.raises(ExperimentConfigError):
        validate_against_plan(cfg, mini_plan, project_root=tmp_path, require_data=False)


def test_multifold_split_rejected(tmp_path: Path, mini_plan):
    split_path = write_json(
        tmp_path / "splits_final.json",
        [{"train": ["a"], "val": ["c"]}, {"train": ["c"], "val": ["a"]}],
    )
    cfg = _load_cfg(tmp_path, _config_doc(Path(mini_plan.source_path), split_path, tmp_path, tmp_path / "out"))
    with pytest.raises(ExperimentConfigError):
        validate_against_plan(cfg, mini_plan, project_root=tmp_path, require_data=False)


def test_output_root_outside_project_rejected(tmp_path: Path, mini_plan):
    split_path = write_json(tmp_path / "splits_final.json", [{"train": ["a", "b"], "val": ["c", "d"]}])
    cfg = _load_cfg(tmp_path, _config_doc(Path(mini_plan.source_path), split_path, tmp_path, Path("/tmp/outside_project")))
    with pytest.raises(ExperimentConfigError):
        validate_against_plan(cfg, mini_plan, project_root=tmp_path, require_data=False)


def test_missing_preprocessed_dir_rejected(tmp_path: Path, mini_plan):
    split_path = write_json(tmp_path / "splits_final.json", [{"train": ["a", "b"], "val": ["c", "d"]}])
    cfg = _load_cfg(tmp_path, _config_doc(Path(mini_plan.source_path), split_path, tmp_path / "nope", tmp_path / "out"))
    with pytest.raises(ExperimentConfigError):
        validate_against_plan(cfg, mini_plan, project_root=tmp_path, require_data=True)


# --------------------------------------------------------- P2-A3：验证协议配置校验
def _doc_and_split(tmp_path: Path, mini_plan, **kwargs) -> tuple[dict, Path]:
    split_path = write_json(tmp_path / "splits_final.json", [{"train": ["a", "b"], "val": ["c", "d"]}])
    doc = _config_doc(Path(mini_plan.source_path), split_path, tmp_path, tmp_path / "out", **kwargs)
    return doc, split_path


def test_full_volume_allows_disabled_early_stopping(tmp_path: Path, mini_plan):
    """v2.2：正式 full_volume 允许禁用 early stopping（固定 200-epoch 预算）；启用时只告警不报错。"""
    doc, _ = _doc_and_split(tmp_path, mini_plan)
    doc["early_stopping"]["enabled"] = False
    cfg = _load_cfg(tmp_path, doc)
    warnings = validate_against_plan(cfg, mini_plan, project_root=tmp_path, require_data=False)
    assert isinstance(warnings, list)  # 不再因禁用 early stopping 报错

    # 启用 early stopping 时给出告警（patience 按 validation event 计数），但同样不报错
    doc2, _ = _doc_and_split(tmp_path, mini_plan)
    doc2["early_stopping"]["enabled"] = True
    cfg2 = _load_cfg(tmp_path, doc2)
    warnings2 = validate_against_plan(cfg2, mini_plan, project_root=tmp_path, require_data=False)
    assert any("early_stopping" in w for w in warnings2)


def test_every_n_epochs_five_is_legal(tmp_path: Path, mini_plan):
    """v2.2：正式低频验证 every_n_epochs=5 合法（与禁用的 early stopping 共存）。"""
    doc, _ = _doc_and_split(tmp_path, mini_plan)
    doc["validation"]["every_n_epochs"] = 5
    doc["early_stopping"]["enabled"] = False
    cfg = _load_cfg(tmp_path, doc)
    assert cfg.validation.every_n_epochs == 5
    validate_against_plan(cfg, mini_plan, project_root=tmp_path, require_data=False)


def test_expected_cases_must_match_split(tmp_path: Path, mini_plan):
    doc, _ = _doc_and_split(tmp_path, mini_plan, n_val=3, n_pos=1, n_neg=2)
    cfg = _load_cfg(tmp_path, doc)
    with pytest.raises(ExperimentConfigError, match="expected_cases"):
        validate_against_plan(cfg, mini_plan, project_root=tmp_path, require_data=False)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda doc: doc["validation"].update({"checkpoint_metric": "val_loss"}),
        lambda doc: doc["validation"].update({"every_n_epochs": 0}),
        lambda doc: doc["validation"].update({"expected_positive_cases": 2}),
        lambda doc: doc["validation"].update({"step_fraction": 0.0}),
        lambda doc: doc["validation"].update({"mode": "nope"}),
        lambda doc: doc["validation"].update({"maximize": False}),
        lambda doc: doc["early_stopping"].update({"min_epochs": 5}),
    ],
)
def test_invalid_validation_protocol_rejected_at_load(tmp_path: Path, mini_plan, mutate):
    doc, _ = _doc_and_split(tmp_path, mini_plan)
    mutate(doc)
    with pytest.raises(ExperimentConfigError):
        _load_cfg(tmp_path, doc)


def test_diagnostic_validation_mode_warns(tmp_path: Path, mini_plan):
    doc, _ = _doc_and_split(tmp_path, mini_plan)
    doc["validation"]["mode"] = "diagnostic_patch"
    doc["early_stopping"]["enabled"] = False  # 诊断模式不要求 early stopping
    cfg = _load_cfg(tmp_path, doc)
    warnings = validate_against_plan(cfg, mini_plan, project_root=tmp_path, require_data=False)
    assert any("诊断模式" in w for w in warnings)


def test_training_config_exposes_max_epochs_alias(tmp_path: Path, mini_plan):
    doc, _ = _doc_and_split(tmp_path, mini_plan)
    doc["training"].pop("max_epochs")
    doc["training"]["epochs"] = 7  # 旧字段名仍需兼容
    cfg = _load_cfg(tmp_path, doc)
    assert cfg.training.max_epochs == 7 and cfg.training.epochs == 7
    assert cfg.training.diagnostic_validation_iterations == 2
