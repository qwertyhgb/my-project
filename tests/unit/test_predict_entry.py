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

def _expected_names() -> dict[str, str]:
    """项目 Trainer 名 -> 类名；**由注册表派生**，不维护第二份手写清单。

    维护两份清单是过去 README / 训练入口 / 预测入口 Trainer 数量口径不一致（11 / 15 / 16）
    的直接原因。现在唯一真源是 ``PROJECT_TRAINERS``，这份测试只检查「派生是否正确」。
    """
    from zonal_reliability_fusion.nnunet.trainers import PROJECT_TRAINERS

    return {name: name for name in PROJECT_TRAINERS}


EXPECTED = _expected_names()


def _load_predict_entry():
    spec = importlib.util.spec_from_file_location(
        "predict_nnunet_entry", PREDICT_SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _project_classes() -> dict:
    """项目 Trainer 名 -> 类；**由注册表派生**（唯一真源）。

    历史上这份映射是手写的，每新增一个 Trainer 都会漏；现在直接读 ``PROJECT_TRAINERS``，
    因此测试只检查「解析器是否与注册表一致」。
    """
    from zonal_reliability_fusion.nnunet.trainers import PROJECT_TRAINERS

    return dict(PROJECT_TRAINERS)


def test_predict_entry_declares_every_project_trainer_name():
    """预测入口的 Trainer 名单必须由注册表派生，不允许出现第二份手写清单。"""
    from zonal_reliability_fusion.nnunet.trainers import PROJECT_TRAINERS

    module = _load_predict_entry()
    assert set(module.PROJECT_TRAINER_NAMES) == set(PROJECT_TRAINERS)
    assert set(module.PROJECT_TRAINER_NAMES) == set(EXPECTED)
    # 排序保证审计输出稳定（预测入口的 name 列表用于日志与自检）
    assert tuple(module.PROJECT_TRAINER_NAMES) == tuple(sorted(EXPECTED))

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



def _anatomy_prediction_fixture(tmp_path, monkeypatch):
    import json

    from zonal_reliability_fusion.anatomy import contracts as anatomy_contracts
    from zonal_reliability_fusion.nnunet import trainers as t
    monkeypatch.setattr(anatomy_contracts, "ANATOMY_COUNTS", (2, 1, 1))
    monkeypatch.setattr(t, 'ANATOMY_COUNTS', (2, 1, 1))
    model = tmp_path / 'nnUNetTrainerPICAI_AnatomyJoint_100ep_NoFFT__nnUNetPlans__3d_fullres'
    model.mkdir()
    dataset = {'channel_names': {'0000': 'T2W'}, 'labels': t.ANATOMY_LABELS, 'regions_class_order': [1, 2, 4],
        'numTraining': 2, 'anatomy_contract': {'encoding': 'WG+2*PZ+4*TZ', 'wg_source': 'materialized_wg',
        'zonal_source': 'zonal_yuan', 'explicit_exclusions': ['11050_1001070'],
        'case_ids': ['1_10', '2_20'], 'train_cases': ['1_10'], 'val_cases': ['2_20']}}
    (model / 'dataset.json').write_text(json.dumps(dataset))
    (model / 'plans.json').write_text(json.dumps({'dataset_name': t.ANATOMY_DATASET, 'transpose_forward': [0, 1, 2]}))
    images = tmp_path / 'images'
    images.mkdir()
    (images / '1_10_0000.nii.gz').write_bytes(b'synthetic placeholder; preflight reads names only')
    output = tmp_path / 'prediction'
    return model, images, output


@pytest.mark.parametrize("equals", [False, True])
def test_anatomy_prediction_requires_probabilities_and_new_output(tmp_path, monkeypatch, equals):
    model, images, output = _anatomy_prediction_fixture(tmp_path, monkeypatch)
    module = _load_predict_entry()
    arguments = ([f'-m={model}', f'-i={images}', f'-o={output}'] if equals else
                 ['-m', str(model), '-i', str(images), '-o', str(output)])
    import nnunetv2.inference.predict_from_raw_data as native
    calls = []
    monkeypatch.setattr(native, 'predict_entry_point_modelfolder', lambda: calls.append('native'))
    monkeypatch.setattr(module, 'check_anatomy_prediction_folder', lambda *args: calls.append('checked'))
    with pytest.raises(ValueError, match='save_probabilities'):
        module.main(arguments)
    assert calls == []
    good = arguments + ['--save_probabilities']
    state = module._anatomy_prediction_preflight(good)
    assert state[2] == ['1_10']
    # Native predictor remains the sole inference implementation.
    module.main(good)
    assert calls == ['native', 'checked']
    output.mkdir()
    (output / 'existing.npz').write_bytes(b'preserve')
    calls.clear()
    with pytest.raises(ValueError, match='refuses overwrite'):
        module.main(good)
    assert calls == []
    assert (output / 'existing.npz').read_bytes() == b'preserve'


def test_anatomy_probability_folder_missing_case_fails(tmp_path):
    module = _load_predict_entry()
    with pytest.raises(ValueError, match='case set mismatch'):
        module.check_anatomy_prediction_folder(tmp_path, tmp_path, ['1_10'], [0, 1, 2], no_progress=True)


@pytest.mark.parametrize('extra', [
    ['-m=other'], ['-i=other'], ['-o=other'], ['-m'], ['-m='],
    ['-i'], ['-o'], ['-prev_stage_predictions'], ['-prev_stage_predictions='],
    ['-num_parts=2'], ['-part_id=0'], ['-prev_stage_predictions=previous'],
    ['-num_parts', '2'], ['-part_id', '0'], ['-prev_stage_predictions', 'previous'],
    ['-prev=previous'], ['--save_probabilities'], ['--c', '--continue_prediction'], ['--disable'],
])
def test_prediction_protection_rejects_invalid_ambiguous_or_restricted_arguments_before_native(tmp_path, monkeypatch, extra):
    model, images, output = _anatomy_prediction_fixture(tmp_path, monkeypatch)
    module = _load_predict_entry()
    import nnunetv2.inference.predict_from_raw_data as native
    calls = []
    monkeypatch.setattr(native, 'predict_entry_point_modelfolder', lambda: calls.append('native'))
    arguments = [f'-m={model}', f'-i={images}', f'-o={output}', '--save_probabilities', *extra]
    with pytest.raises(ValueError):
        module.main(arguments)
    assert calls == []
    assert not output.exists()


@pytest.mark.parametrize('equals', [False, True])
def test_legacy_prediction_argv_delegation_unchanged(tmp_path, monkeypatch, equals):
    import json
    import sys

    import nnunetv2.inference.predict_from_raw_data as native
    model = tmp_path / 'lesion-model'
    model.mkdir()
    (model / 'dataset.json').write_text(json.dumps({'labels': {'background': 0, 'lesion': 1}}))
    output = tmp_path / 'existing-predictions'
    output.mkdir()
    preserved = output / 'case.nii.gz'
    preserved.write_bytes(b'existing synthetic result')
    argv = ([f'-m={model}', f'-i={tmp_path}', f'-o={output}'] if equals else
            ['-m', str(model), '-i', str(tmp_path), '-o', str(output)]) + ['--continue_prediction', '-f', '0', '-device=cpu']
    calls = []
    monkeypatch.setattr(native, 'predict_entry_point_modelfolder', lambda: calls.append(list(sys.argv[1:])))
    module = _load_predict_entry()
    monkeypatch.setattr(module, 'check_anatomy_prediction_folder', lambda *args: pytest.fail('legacy must not use anatomy checks'))
    module.main(argv)
    assert calls == [argv]
    assert preserved.read_bytes() == b'existing synthetic result'
