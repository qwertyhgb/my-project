"""§8.6 浅层 skip 接口契约测试（任务书九·22；P2A 只冻结接口，不实现 Q_l / F_fuse / M1–M4）。"""
from __future__ import annotations

import copy

import pytest

from zonal_reliability_fusion.config.architecture import (
    SHALLOW_SKIP_CONTRACT,
    SHALLOW_SKIP_CONTRACT_ID,
    ArchitectureConfigError,
    assert_shallow_skip_inputs_allowed,
    validate_shallow_skip_contract,
)


def test_frozen_contract_validates_against_itself():
    assert validate_shallow_skip_contract(copy.deepcopy(SHALLOW_SKIP_CONTRACT))["contract_id"] == SHALLOW_SKIP_CONTRACT_ID


def test_contract_freezes_interface_semantics():
    c = SHALLOW_SKIP_CONTRACT
    assert c["applies_to_stages"] == [0, 1]
    assert c["inputs"]["count"] == 3 and c["inputs"]["modality_order"] == ["T2W", "ADC", "HBV"]
    assert c["target_channels_source"] == "plan.features_per_stage[stage]"
    assert c["topology"] == ["concat", "conv1x1x1", "instance_norm3d", "leaky_relu"]
    assert c["produces_second_modality_logits"] is False
    assert c["stage2_skip"] == "F_fuse"
    assert c["stage3_and_deeper_skip"] == "shared_encoder_skip"
    assert c["identical_across_m1_m4"] is True


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update(contract_id="shallow_skip.vX.other"),
        lambda d: d.update(applies_to_stages=[0, 1, 2]),
        lambda d: d["inputs"].update(count=2),
        lambda d: d.update(topology=["concat", "conv3x3x3", "instance_norm3d", "leaky_relu"]),
        lambda d: d.update(produces_second_modality_logits=True),
        lambda d: d.update(stage2_skip="shared_encoder_skip"),
        lambda d: d.update(identical_across_m1_m4=False),
        lambda d: d.update(forbidden_inputs=["pz_tz"]),  # 削弱禁止项
    ],
)
def test_contract_drift_rejected(mutate):
    doc = copy.deepcopy(SHALLOW_SKIP_CONTRACT)
    mutate(doc)
    with pytest.raises(ArchitectureConfigError):
        validate_shallow_skip_contract(doc)


def test_contract_missing_field_rejected():
    doc = copy.deepcopy(SHALLOW_SKIP_CONTRACT)
    doc.pop("stage2_skip")
    with pytest.raises(ArchitectureConfigError, match="缺少字段"):
        validate_shallow_skip_contract(doc)


# ------------------------------------------------------------------ 运行时输入边界守卫
def test_modality_features_allowed():
    assert_shallow_skip_inputs_allowed(["t2w_feature", "adc_feature", "hbv_feature"])  # 不抛


@pytest.mark.parametrize(
    "forbidden",
    ["pz_tz", "wg", "sequence_gate_weight", "sequence_logits", "decoder_feature"],
)
def test_anatomy_and_gate_inputs_forbidden(forbidden):
    with pytest.raises(ArchitectureConfigError, match="禁止读取"):
        assert_shallow_skip_inputs_allowed(["t2w_feature", forbidden])


def test_guard_reports_all_hits():
    with pytest.raises(ArchitectureConfigError) as exc:
        assert_shallow_skip_inputs_allowed(["pz_tz", "wg"])
    assert "pz_tz" in str(exc.value) and "wg" in str(exc.value)
