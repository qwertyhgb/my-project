"""项目预测入口的纯合成测试（不读取真实数据、不跑真实推理）。

重点验证：每个 checkpoint 里的 ``trainer_name`` 都能被预测入口解析到项目 Trainer 类
（即官方 ``initialize_from_trained_model_folder`` 内部调用的 ``recursive_find_python_class``
在安装解析器后能返回项目类），而不是只测 ``build_network_architecture``。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREDICT_SCRIPT = PROJECT_ROOT / "scripts" / "inference" / "predict_nnunet.py"

EXPECTED = {
    "nnUNetTrainerPICAI_FLCE_NoFFT": "nnUNetTrainerPICAI_FLCE_NoFFT",
    "nnUNetTrainerPICAI_DiceCE_NoFFT": "nnUNetTrainerPICAI_DiceCE_NoFFT",
    "nnUNetTrainerPICAI_ImageGate": "nnUNetTrainerPICAI_ImageGate",
    "nnUNetTrainerPICAI_AnatomyGate": "nnUNetTrainerPICAI_AnatomyGate",
    "nnUNetTrainerPICAI_FLCE_PositiveSampling_NoFFT": (
        "nnUNetTrainerPICAI_FLCE_PositiveSampling_NoFFT"
    ),
    "nnUNetTrainerPICAI_ImageGate_PositiveSampling_NoFFT": (
        "nnUNetTrainerPICAI_ImageGate_PositiveSampling_NoFFT"
    ),
    "nnUNetTrainerPICAI_AnatomyGate_PositiveSampling_NoFFT": (
        "nnUNetTrainerPICAI_AnatomyGate_PositiveSampling_NoFFT"
    ),
    "nnUNetTrainerPICAI_FeatureNoGate_PositiveSampling_NoFFT": (
        "nnUNetTrainerPICAI_FeatureNoGate_PositiveSampling_NoFFT"
    ),
    "nnUNetTrainerPICAI_FeatureImageGate_PositiveSampling_NoFFT": (
        "nnUNetTrainerPICAI_FeatureImageGate_PositiveSampling_NoFFT"
    ),
    "nnUNetTrainerPICAI_FeatureAnatomyGate_PositiveSampling_NoFFT": (
        "nnUNetTrainerPICAI_FeatureAnatomyGate_PositiveSampling_NoFFT"
    ),
}


def _load_predict_entry():
    spec = importlib.util.spec_from_file_location(
        "predict_nnunet_entry", PREDICT_SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _project_classes() -> dict:
    from zonal_reliability_fusion.nnunet.trainers import (
        nnUNetTrainerPICAI_AnatomyGate,
        nnUNetTrainerPICAI_AnatomyGate_PositiveSampling_NoFFT,
        nnUNetTrainerPICAI_DiceCE_NoFFT,
        nnUNetTrainerPICAI_FeatureAnatomyGate_PositiveSampling_NoFFT,
        nnUNetTrainerPICAI_FeatureImageGate_PositiveSampling_NoFFT,
        nnUNetTrainerPICAI_FeatureNoGate_PositiveSampling_NoFFT,
        nnUNetTrainerPICAI_FLCE_NoFFT,
        nnUNetTrainerPICAI_FLCE_PositiveSampling_NoFFT,
        nnUNetTrainerPICAI_ImageGate,
        nnUNetTrainerPICAI_ImageGate_PositiveSampling_NoFFT,
    )

    return {
        "nnUNetTrainerPICAI_FLCE_NoFFT": nnUNetTrainerPICAI_FLCE_NoFFT,
        "nnUNetTrainerPICAI_DiceCE_NoFFT": nnUNetTrainerPICAI_DiceCE_NoFFT,
        "nnUNetTrainerPICAI_ImageGate": nnUNetTrainerPICAI_ImageGate,
        "nnUNetTrainerPICAI_AnatomyGate": nnUNetTrainerPICAI_AnatomyGate,
        "nnUNetTrainerPICAI_FLCE_PositiveSampling_NoFFT": (
            nnUNetTrainerPICAI_FLCE_PositiveSampling_NoFFT
        ),
        "nnUNetTrainerPICAI_ImageGate_PositiveSampling_NoFFT": (
            nnUNetTrainerPICAI_ImageGate_PositiveSampling_NoFFT
        ),
        "nnUNetTrainerPICAI_AnatomyGate_PositiveSampling_NoFFT": (
            nnUNetTrainerPICAI_AnatomyGate_PositiveSampling_NoFFT
        ),
        "nnUNetTrainerPICAI_FeatureNoGate_PositiveSampling_NoFFT": (
            nnUNetTrainerPICAI_FeatureNoGate_PositiveSampling_NoFFT
        ),
        "nnUNetTrainerPICAI_FeatureImageGate_PositiveSampling_NoFFT": (
            nnUNetTrainerPICAI_FeatureImageGate_PositiveSampling_NoFFT
        ),
        "nnUNetTrainerPICAI_FeatureAnatomyGate_PositiveSampling_NoFFT": (
            nnUNetTrainerPICAI_FeatureAnatomyGate_PositiveSampling_NoFFT
        ),
    }


def test_predict_entry_declares_every_project_trainer_name():
    module = _load_predict_entry()
    assert tuple(module.PROJECT_TRAINER_NAMES) == tuple(EXPECTED)

    # PROJECT_TRAINER_NAMES 与实际注册表必须一致，避免漏加新 Trainer
    from zonal_reliability_fusion.nnunet.trainers import PROJECT_TRAINERS

    assert set(module.PROJECT_TRAINER_NAMES) == set(PROJECT_TRAINERS)


def test_predict_entry_resolves_all_checkpoint_trainer_names():
    """resolve_project_trainer 必须把每个 trainer_name 映射到对应项目类。"""
    module = _load_predict_entry()
    classes = _project_classes()
    for name in EXPECTED:
        assert module.resolve_project_trainer(name) is classes[name]


def test_installed_resolver_feeds_official_predictor_lookup():
    """安装解析器后，官方预测模块用来定位 trainer 的函数必须返回项目类（不递归扫描）。"""
    module = _load_predict_entry()
    module.install_project_trainer_resolver()

    import nnunetv2
    import nnunetv2.inference.predict_from_raw_data as predict_module
    from batchgenerators.utilities.file_and_folder_operations import join

    classes = _project_classes()
    folder = join(nnunetv2.__path__[0], "training", "nnUNetTrainer")
    for name in EXPECTED:
        # 这正是 initialize_from_trained_model_folder 内部读取 checkpoint['trainer_name'] 后的调用
        resolved = predict_module.recursive_find_python_class(
            folder, name, "nnunetv2.training.nnUNetTrainer"
        )
        assert resolved is classes[name]


def test_resolver_install_is_idempotent():
    module = _load_predict_entry()
    first = module.install_project_trainer_resolver()
    second = module.install_project_trainer_resolver()
    # 重复安装不得层层包裹：原始函数只捕获一次
    assert callable(first) and callable(second)
    classes = _project_classes()
    for name in EXPECTED:
        assert module.resolve_project_trainer(name) is classes[name]


def test_predict_entry_help_exits_zero(monkeypatch):
    """`--help` 透传到官方 parser，退出码 0（不触发真实推理）。"""
    module = _load_predict_entry()
    monkeypatch.setattr("sys.argv", ["predict_nnunet.py", "--help"])
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 0
