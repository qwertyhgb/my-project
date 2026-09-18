"""v2.3 训练/摘要入口脚本装配测试（CLI --help、配置分派、输出隔离；不读病例、不用 GPU、不起训练）。"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from zonal_reliability_fusion.config import load_experiment_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN = PROJECT_ROOT / "scripts/train/train_m0.py"
SETUP = PROJECT_ROOT / "scripts/train/validate_m0_setup.py"
SUMMARY = PROJECT_ROOT / "scripts/train/summarize_m0_resenc_architecture.py"
EXP_V23 = PROJECT_ROOT / "configs/experiments/m0_resenc_picai_3d_fullres_v23.yaml"
EXP_LEGACY = PROJECT_ROOT / "configs/experiments/m0_picai_3d_fullres.yaml"
REAL_PLANS = PROJECT_ROOT / "workdir/nnUNet_preprocessed/Dataset605_PICAI/nnUNetPlans.json"
REAL_SPLITS = PROJECT_ROOT / "workdir/nnUNet_preprocessed/Dataset605_PICAI/splits_final.json"


def _env() -> dict:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONPATH"] = os.pathsep.join(
        [str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "third_party/nnUNet"), env.get("PYTHONPATH", "")]
    )
    return env


@pytest.mark.parametrize("script", [TRAIN, SETUP, SUMMARY])
def test_cli_help_ok(script: Path):
    """三个入口的 --help 必须正常（argparse 装配，不读数据、不建模型、不用 GPU）。"""
    assert script.is_file()
    proc = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        env=_env(),
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "usage" in proc.stdout.lower()


def test_summary_script_has_no_progress_flag():
    text = SUMMARY.read_text()
    assert "--no-progress" in text and "--reduced-forward-check" in text


# ------------------------------------------------------------------ 配置分派（parse 级，不建模型）
def test_v23_config_declares_residual_architecture():
    cfg = load_experiment_config(EXP_V23)
    assert cfg.architecture is not None
    assert cfg.architecture.encoder_type == "residual_encoder"
    assert cfg.architecture.architecture_name == "m0_resenc_picai_3d_fullres"
    assert cfg.architecture.config_path.endswith("configs/architectures/m0_resenc_v23.yaml")
    # 训练/验证协议与 legacy 一致（复用同一基础设施）
    assert cfg.training.max_epochs == 200 and cfg.validation.every_n_epochs == 5
    assert cfg.loss.name == "focal_ce" and cfg.early_stopping.enabled is False


def test_legacy_config_has_no_architecture_block():
    cfg = load_experiment_config(EXP_LEGACY)
    assert cfg.architecture is None  # → 工厂只会构建 legacy PlainConv M0


def test_output_identity_isolated_from_legacy():
    v23 = load_experiment_config(EXP_V23)
    legacy = load_experiment_config(EXP_LEGACY)
    assert v23.experiment.name != legacy.experiment.name
    # 输出根目录不得共用
    assert v23.paths.output_root.rstrip("/").endswith("m0_resenc")
    assert legacy.paths.output_root.rstrip("/").endswith("/m0")
    assert v23.paths.output_root != legacy.paths.output_root


@pytest.mark.skipif(not (REAL_PLANS.is_file() and REAL_SPLITS.is_file()), reason="真实 plan/split 不存在")
def test_v23_config_validates_against_real_plan_and_split():
    from zonal_reliability_fusion.config import load_plan, validate_against_plan
    from zonal_reliability_fusion.models.factory import resolve_architecture_spec

    cfg = load_experiment_config(EXP_V23)
    plan = load_plan(cfg.resolve_path(cfg.paths.plans), "3d_fullres", dataset_name="Dataset605_PICAI")
    warnings = validate_against_plan(cfg, plan, project_root=PROJECT_ROOT, require_data=False)
    assert isinstance(warnings, list)
    # 架构 spec 与冻结 plan 一致（features/kernel/stride/decoder n_conv）
    spec = resolve_architecture_spec(cfg, plan)
    assert spec.blocks_per_stage == (1, 3, 4, 6, 6, 6, 6)
    assert spec.features_per_stage == plan.features_per_stage
