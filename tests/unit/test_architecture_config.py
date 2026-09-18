"""v2.3 版本化架构配置 / 稳定哈希 / 身份校验测试（任务书九·14,15,17,18；纯 CPU，不读真实数据）。"""
from __future__ import annotations

import copy
from pathlib import Path

import pytest
from conftest import build_resenc_arch_doc

from zonal_reliability_fusion.config import (
    FUSION_STAGE,
    SHALLOW_SKIP_CONTRACT,
    SHALLOW_SKIP_CONTRACT_ID,
    ArchitectureConfigError,
    ArchitectureIdentityError,
    architecture_spec_from_mapping,
    canonical_json,
    load_architecture_config,
    validate_architecture_against_plan,
    verify_architecture_identity,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REAL_PLANS = PROJECT_ROOT / "workdir/nnUNet_preprocessed/Dataset605_PICAI/nnUNetPlans.json"
REAL_ARCH = PROJECT_ROOT / "configs/architectures/m0_resenc_v23.yaml"


def _spec(mini_resenc_plan, **over):
    doc = build_resenc_arch_doc(mini_resenc_plan, blocks_per_stage=[1, 2, 2, 2])
    doc.update(over)
    return architecture_spec_from_mapping(doc, source_path="synthetic")


# ------------------------------------------------------------------ 14. 哈希稳定性
def test_hash_is_stable_and_hex(mini_resenc_plan):
    s1 = _spec(mini_resenc_plan)
    s2 = _spec(mini_resenc_plan)
    h = s1.architecture_sha256()
    assert h == s2.architecture_sha256()
    assert isinstance(h, str) and len(h) == 64 and all(c in "0123456789abcdef" for c in h)


def test_hash_independent_of_key_order(mini_resenc_plan):
    doc = build_resenc_arch_doc(mini_resenc_plan, blocks_per_stage=[1, 2, 2, 2])
    forward = architecture_spec_from_mapping(doc, source_path="a")
    # 反转顶层键插入顺序；canonical_json(sort_keys) 应给出相同哈希
    reversed_doc = {k: doc[k] for k in reversed(list(doc.keys()))}
    backward = architecture_spec_from_mapping(reversed_doc, source_path="b")
    assert forward.architecture_sha256() == backward.architecture_sha256()


def test_non_structural_fields_do_not_change_hash(mini_resenc_plan):
    base = _spec(mini_resenc_plan)
    doc = build_resenc_arch_doc(mini_resenc_plan, blocks_per_stage=[1, 2, 2, 2])
    doc["provenance"] = {"configuration": "3d_fullres", "plan_source": "/totally/different/path.json", "note": "x"}
    with_prov = architecture_spec_from_mapping(doc, source_path="different_source_path")
    assert base.architecture_sha256() == with_prov.architecture_sha256()  # provenance/source_path 不进入哈希
    assert "provenance" not in canonical_json(base.structural_dict())


# ------------------------------------------------------------------ 15. 任一（可自由取值的）结构字段改变 → 哈希改变
@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update(architecture_name="other_name"),
        lambda d: d.update(architecture_version="v9.9"),
        lambda d: d.update(blocks_per_stage=[1, 2, 2, 3]),
        lambda d: d.update(features_per_stage=[4, 8, 16, 32]),
        lambda d: d["kernel_sizes"].__setitem__(1, [1, 3, 3]),
        lambda d: d["strides"].__setitem__(2, [1, 2, 2]),
        lambda d: d.update(conv_bias=False),
        lambda d: d["normalization"].update(eps=1e-4),
        lambda d: d["initialization"].update(zero_init_last_norm_before_add=False),
        lambda d: d["decoder"].update(n_conv_per_stage_decoder=[2, 2, 1]),
        lambda d: d.update(in_channels=4),
        lambda d: d.update(num_classes=3),
    ],
)
def test_any_structural_field_change_changes_hash(mini_resenc_plan, mutate):
    base = _spec(mini_resenc_plan).architecture_sha256()
    doc = build_resenc_arch_doc(mini_resenc_plan, blocks_per_stage=[1, 2, 2, 2])
    mutate(doc)
    changed = architecture_spec_from_mapping(doc, source_path="synthetic").architecture_sha256()
    assert changed != base


@pytest.mark.parametrize(
    "mutate,err",
    [
        # 枚举/契约非法值
        (lambda d: d.update(activation_order="pre_activation"), ArchitectureConfigError),
        (lambda d: d.update(block_type="bottleneck"), ArchitectureConfigError),
        (lambda d: d.update(encoder_type="plain_conv"), ArchitectureConfigError),
        (lambda d: d.update(fusion_stage=1), ArchitectureConfigError),
        (lambda d: d["shallow_skip_contract"].update(topology=["concat", "conv3x3x3"]), ArchitectureConfigError),
        (lambda d: d.update(blocks_per_stage=[1, 2, 2]), ArchitectureConfigError),  # 长度 != n_stages
        # 审阅者指出的 4 例：声明模型未实现的值必须拒绝（“改了但不生效”漏洞）
        (lambda d: d["shortcut"].update(projection_kernel=[3, 3, 3]), ArchitectureConfigError),
        (lambda d: d["initialization"].update(method="xavier_uniform"), ArchitectureConfigError),
        (lambda d: d["decoder"].update(skip_merge="sum"), ArchitectureConfigError),
        (lambda d: d["deep_supervision"].update(output_order="low_to_high_resolution"), ArchitectureConfigError),
        # 白名单：shortcut / stem / decoder / DS / norm / nonlin / init 的其余冻结字段
        (lambda d: d["shortcut"].update(projection_when="stride_or_channel_change"), ArchitectureConfigError),
        (lambda d: d["shortcut"].update(downsample_when="always"), ArchitectureConfigError),
        (lambda d: d["shortcut"].update(downsample_before_projection=False), ArchitectureConfigError),
        (lambda d: d["shortcut"].update(projection_conv_bias=True), ArchitectureConfigError),
        (lambda d: d["shortcut"].update(implicit_channel_crop_forbidden=False), ArchitectureConfigError),
        (lambda d: d["stem"].update(n_convs=2), ArchitectureConfigError),
        (lambda d: d["stem"].update(type="custom_stem"), ArchitectureConfigError),
        (lambda d: d["stem"].update(does_downsample=True), ArchitectureConfigError),
        (lambda d: d["decoder"].update(residual=True), ArchitectureConfigError),
        (lambda d: d["decoder"].update(skip_align="interpolate"), ArchitectureConfigError),
        (lambda d: d["decoder"].update(upsample="interpolate"), ArchitectureConfigError),
        (lambda d: d["decoder"].update(type="residual_decoder"), ArchitectureConfigError),
        (lambda d: d["decoder"].update(transpconv_kernel_equals_stride=False), ArchitectureConfigError),
        (lambda d: d["decoder"].update(per_level_seg_heads=False), ArchitectureConfigError),
        (lambda d: d["deep_supervision"].update(weight_scheme="uniform"), ArchitectureConfigError),
        (lambda d: d["deep_supervision"].update(lowest_resolution_weight_zero=False), ArchitectureConfigError),
        (lambda d: d["deep_supervision"].update(weights_normalized_to_one=False), ArchitectureConfigError),
        (lambda d: d["normalization"].update(op="BatchNorm3d"), ArchitectureConfigError),
        (lambda d: d["nonlinearity"].update(op="ReLU"), ArchitectureConfigError),
        (lambda d: d["nonlinearity"].update(inplace=False), ArchitectureConfigError),
        (lambda d: d["initialization"].update(mode="fan_out"), ArchitectureConfigError),
        (lambda d: d["initialization"].update(nonlinearity="relu"), ArchitectureConfigError),
        (lambda d: d["initialization"].update(conv_bias_init="ones"), ArchitectureConfigError),
        (lambda d: d["initialization"].update(norm_weight_init="zeros"), ArchitectureConfigError),
        # 一致性：init 与 nonlinearity 的 LeakyReLU 斜率必须相等
        (lambda d: d["nonlinearity"].update(negative_slope=0.02), ArchitectureConfigError),
    ],
)
def test_illegal_or_unsupported_value_rejected(mini_resenc_plan, mutate, err):
    doc = build_resenc_arch_doc(mini_resenc_plan, blocks_per_stage=[1, 2, 2, 2])
    mutate(doc)
    with pytest.raises(err):
        architecture_spec_from_mapping(doc, source_path="synthetic")


# ------------------------------------------------------------------ 契约 id / fusion stage 冻结
def test_contract_id_and_fusion_stage_frozen(mini_resenc_plan):
    spec = _spec(mini_resenc_plan)
    assert spec.shallow_skip_contract_id == SHALLOW_SKIP_CONTRACT_ID
    assert spec.fusion_stage == FUSION_STAGE == 2
    assert SHALLOW_SKIP_CONTRACT["produces_second_modality_logits"] is False
    assert "pz_tz" in SHALLOW_SKIP_CONTRACT["forbidden_inputs"]


def test_shortcut_semantics_frozen_downsample_vs_projection(mini_resenc_plan):
    """配置声明的 shortcut 语义必须与实现一致：下采样仅由 stride 触发，projection 仅由通道变化触发。"""
    sc = _spec(mini_resenc_plan).shortcut
    assert sc["identity_when"] == "same_spatial_and_channels"
    assert sc["downsample_when"] == "stride_change"
    assert sc["projection_when"] == "channel_change"  # 不是 stride_or_channel_change
    assert sc["projection_kernel"] == (1, 1, 1)
    assert sc["projection_conv_bias"] is False and sc["projection_norm"] is True


# ------------------------------------------------------------------ arch ↔ plan 一致性
def test_validate_against_plan_ok(mini_resenc_plan):
    spec = _spec(mini_resenc_plan)
    warns = validate_architecture_against_plan(spec, mini_resenc_plan)
    assert isinstance(warns, list) and warns


@pytest.mark.parametrize(
    "field_name,value",
    [
        ("features_per_stage", [4, 8, 16, 32]),
        ("strides", [[1, 1, 1], [1, 2, 2], [2, 2, 2], [1, 2, 2]]),
        ("kernel_sizes", [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]),
    ],
)
def test_validate_against_plan_rejects_drift(mini_resenc_plan, field_name, value):
    doc = build_resenc_arch_doc(mini_resenc_plan, blocks_per_stage=[1, 2, 2, 2])
    doc[field_name] = value
    spec = architecture_spec_from_mapping(doc, source_path="synthetic")
    with pytest.raises(ArchitectureConfigError):
        validate_architecture_against_plan(spec, mini_resenc_plan)


def test_decoder_n_conv_must_match_plan(mini_resenc_plan):
    doc = build_resenc_arch_doc(mini_resenc_plan, blocks_per_stage=[1, 2, 2, 2])
    doc["decoder"]["n_conv_per_stage_decoder"] = [1, 1, 1]  # 与 plan 的 [2,2,2] 不符
    spec = architecture_spec_from_mapping(doc, source_path="synthetic")
    with pytest.raises(ArchitectureConfigError):
        validate_architecture_against_plan(spec, mini_resenc_plan)


# ------------------------------------------------------------------ 17/18. 身份校验
def test_verify_identity_ok_and_mismatch(mini_resenc_plan):
    ident = _spec(mini_resenc_plan).identity()
    verify_architecture_identity(ident, dict(ident))  # 一致 → 不抛
    bad = dict(ident)
    bad["architecture_sha256"] = "0" * 64
    with pytest.raises(ArchitectureIdentityError, match="架构身份不一致"):
        verify_architecture_identity(ident, bad)


def test_verify_identity_rejects_missing_and_legacy(mini_resenc_plan):
    ident = _spec(mini_resenc_plan).identity()
    with pytest.raises(ArchitectureIdentityError, match="legacy"):
        verify_architecture_identity(ident, None)  # checkpoint 无架构身份
    with pytest.raises(ArchitectureIdentityError, match="architecture_name"):
        verify_architecture_identity(ident, {"encoder_type": "plain_conv"})  # 缺 name（疑似 legacy）


# ------------------------------------------------------------------ 真实配置（只读 JSON/YAML，不读病例）
@pytest.mark.skipif(not (REAL_ARCH.is_file() and REAL_PLANS.is_file()), reason="真实 arch/plan 不存在")
def test_real_arch_config_loads_and_matches_real_plan():
    from zonal_reliability_fusion.config import load_plan

    spec = load_architecture_config(REAL_ARCH)
    plan = load_plan(REAL_PLANS, "3d_fullres", dataset_name="Dataset605_PICAI")
    validate_architecture_against_plan(spec, plan)
    assert spec.blocks_per_stage == (1, 3, 4, 6, 6, 6, 6)  # 官方 ResEnc-M encoder 拓扑
    assert spec.features_per_stage == plan.features_per_stage
    assert list(spec.decoder["n_conv_per_stage_decoder"]) == list(plan.n_conv_per_stage_decoder) == [2, 2, 2, 2, 2, 2]
    assert spec.architecture_name == "m0_resenc_picai_3d_fullres"
    assert spec.architecture_version == "v2.3"
    assert spec.shortcut["downsample_when"] == "stride_change"
    assert spec.shortcut["projection_when"] == "channel_change"
    assert len(spec.architecture_sha256()) == 64


def test_copy_of_spec_has_identical_hash(mini_resenc_plan):
    spec = _spec(mini_resenc_plan)
    clone = copy.deepcopy(spec)
    assert clone.architecture_sha256() == spec.architecture_sha256()
    assert clone.to_dict()["architecture_sha256"] == spec.architecture_sha256()


def test_mutating_returned_dict_does_not_change_hash(mini_resenc_plan):
    """frozen spec 不得因调用方修改 structural_dict()/to_dict() 返回的嵌套 dict 而改变哈希。

    回归防护：早期版本返回内部 dict 的引用，导致外部修改（如 reduced-forward 检查）污染 spec 与哈希。
    """
    spec = _spec(mini_resenc_plan)
    before = spec.architecture_sha256()
    structural = spec.structural_dict()
    structural["decoder"]["n_conv_per_stage_decoder"] = [9, 9, 9]  # 篡改返回的嵌套 dict
    structural["shortcut"]["projection_conv_bias"] = True
    full = spec.to_dict()
    full["deep_supervision"]["weight_scheme"] = "uniform"
    assert spec.architecture_sha256() == before  # 哈希不变
    assert spec.structural_dict()["decoder"]["n_conv_per_stage_decoder"] != [9, 9, 9]
    assert spec.identity()["architecture_sha256"] == before
