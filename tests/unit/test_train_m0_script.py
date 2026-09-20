"""训练入口脚本的装配测试（不读真实病例、不初始化 CUDA、不起训练）。

覆盖：
- run 目录安全检查（非空且未 resume → 拒绝启动）；
- 验证组件装配：正式模式只建 FullVolumeValidator，**不创建**随机 validation patch provider；
- 仓库内 M0 配置确实声明了 P2-A3 的训练/验证/early-stopping 协议。
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from zonal_reliability_fusion.config import load_experiment_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts/train/train_m0.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location("train_m0_script_under_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script():
    return load_script_module()


class FakeStore:
    """只提供 val_ids / available_ids 的假 store：任何读取病例的调用都会失败。"""

    def __init__(self, val_ids: tuple[str, ...]) -> None:
        self.val_ids = tuple(val_ids)

    @property
    def available_ids(self) -> tuple[str, ...]:
        return self.val_ids

    def get_case(self, case_id: str):  # pragma: no cover - 装配测试不应触发
        raise AssertionError(f"装配测试不应读取病例: {case_id}")


def write_config(tmp_path: Path, mini_plan, *, n_val: int = 2) -> Path:
    doc = {
        "experiment": {"name": "script_unit", "model_id": "M0", "seed": 0, "fold": 0},
        "paths": {
            "plans": str(mini_plan.source_path),
            "preprocessed_dataset": str(tmp_path / "pre"),
            "splits": str(tmp_path / "splits_final.json"),
            "output_root": str(tmp_path / "out"),
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
        "augmentation": {"mirror": True, "mirror_p_per_axis": 0.5, "per_channel_intensity": False},
        "loss": {"name": "dice_ce", "dice_weight": 1.0, "ce_weight": 1.0, "smooth": 1e-5, "include_background": False, "batch_dice": False},
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
            "expected_positive_cases": 1,
            "expected_negative_cases": n_val - 1,
            "step_fraction": 0.5,
            "gaussian": True,
            "mirror_tta": False,
            "sliding_window_batch_size": 1,
            "inner_patch_progress": False,
            "save_case_metrics": True,
        },
        "early_stopping": {"enabled": True, "min_epochs": 1, "patience": 2, "min_delta": 1e-4},
        "inference": {"step_fraction": 0.5, "gaussian": True, "mirror_tta": False},
    }
    (tmp_path / "splits_final.json").write_text(json.dumps([{"train": ["a", "b"], "val": ["c", "d"][:n_val]}]))
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(doc))
    return cfg_path


# ------------------------------------------------------------------ 目录安全
def test_ensure_fresh_run_dirs_rejects_non_empty(tmp_path: Path, script):
    run_dir = tmp_path / "run"
    metrics_dir = tmp_path / "metrics"
    diagnostics_dir = tmp_path / "diagnostics"
    script.ensure_fresh_run_dirs(run_dir, metrics_dir, diagnostics_dir)  # 都不存在 → 允许

    metrics_dir.mkdir()
    script.ensure_fresh_run_dirs(run_dir, metrics_dir, diagnostics_dir)  # 空目录 → 允许
    (metrics_dir / "epoch_metrics.csv").write_text("epoch\n0\n")
    with pytest.raises(SystemExit, match="拒绝启动"):
        script.ensure_fresh_run_dirs(run_dir, metrics_dir, diagnostics_dir)
    # 显式 resume → 允许（继续既有 run）
    script.ensure_fresh_run_dirs(run_dir, metrics_dir, diagnostics_dir, resume_from=str(metrics_dir / "x.pth"))


# ------------------------------------------------------------ 验证组件装配
def test_formal_mode_creates_only_full_volume_validator(tmp_path: Path, mini_plan, script):
    from zonal_reliability_fusion.evaluation import FullVolumeValidator

    cfg = load_experiment_config(write_config(tmp_path, mini_plan, n_val=2))
    store = FakeStore(("c", "d"))
    validator, val_provider = script.build_validation_components(
        cfg, mini_plan, store,
        mode="full_volume", device="cpu", progress=False, metrics_dir=tmp_path / "metrics",
    )
    assert isinstance(validator, FullVolumeValidator)
    assert val_provider is None  # 正式模式不再创建随机 validation patch provider
    assert validator.case_ids == store.val_ids
    assert validator.protocol.checkpoint_metric == "val_positive_casewise_dice_mean"
    assert (validator.protocol.expected_cases, validator.protocol.expected_positive_cases,
            validator.protocol.expected_negative_cases) == (2, 1, 1)


def test_diagnostic_mode_creates_patch_provider_only(tmp_path: Path, mini_plan, script):
    from zonal_reliability_fusion.data import TorchBatchProvider

    cfg = load_experiment_config(write_config(tmp_path, mini_plan, n_val=2))
    store = FakeStore(("c", "d"))
    validator, val_provider = script.build_validation_components(
        cfg, mini_plan, store,
        mode="diagnostic_patch", device="cpu", progress=False, metrics_dir=tmp_path / "metrics",
    )
    assert validator is None
    assert isinstance(val_provider, TorchBatchProvider)
    assert val_provider.force_foreground is False  # 诊断验证同样不做前景过采样
    assert val_provider.num_batches == cfg.training.diagnostic_validation_iterations


def test_unknown_mode_rejected(tmp_path: Path, mini_plan, script):
    cfg = load_experiment_config(write_config(tmp_path, mini_plan, n_val=2))
    with pytest.raises(SystemExit, match="未知验证模式"):
        script.build_validation_components(
            cfg, mini_plan, FakeStore(("c", "d")), mode="nope", device="cpu", progress=False
        )


# ------------------------------------------------------------ 仓库配置自检
def test_repo_config_declares_formal_protocol():
    """仓库正式 M0 配置必须声明 v2.2 协议：200-epoch 固定预算、每 5 epoch 低频验证、禁用 early stopping。"""
    cfg = load_experiment_config(PROJECT_ROOT / "configs/experiments/m0_picai_3d_fullres.yaml")
    assert cfg.training.max_epochs == 200  # 固定预算（PolyLR max_steps 对齐 200）
    assert cfg.training.iterations_per_epoch == 250
    assert cfg.validation.mode == "full_volume"
    assert cfg.validation.every_n_epochs == 5  # v2.2：低频全体积验证
    assert cfg.validation.checkpoint_metric == "val_positive_casewise_dice_mean"
    assert cfg.validation.maximize is True
    assert (cfg.validation.expected_cases, cfg.validation.expected_positive_cases, cfg.validation.expected_negative_cases) == (223, 63, 160)
    assert cfg.validation.step_fraction == 0.5
    assert cfg.validation.gaussian is True and cfg.validation.mirror_tta is False
    assert cfg.validation.sliding_window_batch_size == 1
    assert cfg.validation.inner_patch_progress is False
    # v2.2：正式 M0–M4 禁用“连续无改善”early stopping（固定预算）；patience 单位为 validation event
    assert cfg.early_stopping.enabled is False
    assert (cfg.early_stopping.min_epochs, cfg.early_stopping.patience) == (50, 8)
    assert cfg.early_stopping.min_delta == pytest.approx(1e-4)
    assert cfg.inference.mirror_tta is False


def test_formal_resenc_family_declares_1000_epoch_protocol():
    """正式消融族（M0-resenc + M1–M4）必须声明 1000-epoch 固定预算 + 每 50 epoch 低频验证。

    2026-09-20 协议变更（见 `docs/protocol_changelog.md`）：预算 200 → 1000（与 N0 的 epoch 预算对齐）、
    验证调度 5 → 50。legacy `m0_picai_3d_fullres.yaml` 保持 v2.2 冻结值（200/5）不动，
    因此 `test_repo_config_declares_formal_protocol` 仍针对 legacy 文件断言 200/5。
    """
    family = (
        "m0_resenc_picai_3d_fullres_v23.yaml",
        "m1_equal_picai_3d_fullres_v24.yaml",
        "m2_image_gate_picai_3d_fullres_v24.yaml",
        "m3_zone_input_picai_3d_fullres_v24.yaml",
        "m4_conditioned_gate_picai_3d_fullres_v24.yaml",
    )
    for name in family:
        cfg = load_experiment_config(PROJECT_ROOT / "configs/experiments" / name)
        assert cfg.training.max_epochs == 1000, name
        assert cfg.training.iterations_per_epoch == 250, name
        assert cfg.validation.mode == "full_volume", name
        assert cfg.validation.every_n_epochs == 50, name
        assert (
            cfg.validation.expected_cases,
            cfg.validation.expected_positive_cases,
            cfg.validation.expected_negative_cases,
        ) == (223, 63, 160), name
        assert cfg.validation.checkpoint_metric == "val_positive_casewise_dice_mean", name
        assert cfg.validation.maximize is True, name
        assert cfg.early_stopping.enabled is False, name
