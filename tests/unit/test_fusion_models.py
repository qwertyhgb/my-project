"""M1–M4 融合模型合成测试（纯合成张量，不读取任何真实医学数据）。

架构文档来源：**直接读取真实 `configs/architectures/fusion_resenc_v24.yaml` 的 fusion_family**，
只用 `conftest.build_resenc_arch_doc(plan)` 替换 plan 相关字段（features/kernel/stride/decoder/blocks），
避免在测试里复制一份可能与真实配置漂移的 FUSION_FAMILY。

覆盖（任务书「必须新增的合成测试」的模型层部分）：
- gate 精确读取 modality projection 输出（hook spy）；
- M3 精确执行 Q([F_fuse,E_Z])，无隐式 residual；
- 配置 topology 与真实模块一致（含 InstanceNorm3d）；
- 非法 PZ/TZ overlap 被拒绝；
- M1 严格 1/3、M2/M3/M4 逐体素权重和为 1、M4 零初始化恒等；
- M1/M2 拒绝 zonal；M3/M4 缺 zonal 报错；WG 拒绝；
- 浅层 skip 契约守卫；stage-2 融合同时进入共享 encoder 与 decoder skip；
- 两个 optimizer step 后 gate/conditioning/zone encoder 关键参数获得有限且非零梯度；
- 解析 MACs 与 hook 计数逐值一致（缩小合成 plan）；
- 参数量分解、架构哈希稳定、跨模型身份隔离、固定 seed 可复现；
- 非有限输入、错误通道数、非法 prior 明确失败。
"""
from __future__ import annotations

import copy
import types
from pathlib import Path

import pytest
import torch
import yaml
from conftest import build_resenc_arch_doc

from zonal_reliability_fusion.config.architecture import (
    FUSION_VARIANT_TO_MODEL_ID,
    FUSION_VARIANTS,
    ArchitectureConfigError,
    ArchitectureIdentityError,
    ArchitectureRef,
    architecture_spec_from_mapping,
    load_architecture_config,
    resolve_fusion_spec,
    verify_architecture_identity,
)
from zonal_reliability_fusion.models import (
    INPUT_KEY_ZONAL,
    M1EqualFusionModel,
    M2ImageGateFusionModel,
    M3ZoneInputFusionModel,
    M4ConditionedGateFusionModel,
    build_model,
)
from zonal_reliability_fusion.models.fusion_blocks import (
    SHALLOW_SKIP_READS,
    ShallowSkipAggregator,
)
from zonal_reliability_fusion.models.profiling import FUSION_MAC_GROUPS

REAL_FUSION_CONFIG = Path("configs/architectures/fusion_resenc_v24.yaml")

VARIANT_TO_CLASS = {
    "m1_equal": M1EqualFusionModel,
    "m2_image_gate": M2ImageGateFusionModel,
    "m3_zone_input": M3ZoneInputFusionModel,
    "m4_conditioned_gate": M4ConditionedGateFusionModel,
}


def _fusion_arch_doc(plan) -> dict:
    """mini plan 的骨干 + 真实配置的 fusion_family（不复制一份可能漂移的融合定义）。"""
    real = yaml.safe_load(REAL_FUSION_CONFIG.read_text())
    doc = build_resenc_arch_doc(plan)
    doc["architecture_name"] = "fusion_resenc_test"
    doc["architecture_version"] = real["architecture_version"]
    doc["fusion_family"] = copy.deepcopy(real["fusion_family"])
    return doc


@pytest.fixture
def fusion_doc(mini_resenc_plan):
    return _fusion_arch_doc(mini_resenc_plan)


@pytest.fixture
def fusion_specs(fusion_doc):
    base = architecture_spec_from_mapping(fusion_doc, source_path="synthetic")
    return {variant: resolve_fusion_spec(base, variant) for variant in FUSION_VARIANTS}


def _make(variant: str, plan, spec, *, ds: bool = True):
    return VARIANT_TO_CLASS[variant](plan, spec, in_channels=3, num_classes=2, deep_supervision=ds)


def _inputs(plan, *, batch: int = 1, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    patch = tuple(plan.patch_size)
    image = torch.randn(batch, 3, *patch, generator=g)
    pz = (torch.rand(batch, 1, *patch, generator=g) > 0.5).float()
    tz = 1.0 - pz
    return image, torch.cat((pz, tz), dim=1)


def _kwargs(model, prior):
    return {} if model.zone_encoder is None else {"zonal_prior": prior}


# --------------------------------------------------------------------------- 输出 / DS
@pytest.mark.parametrize("variant", FUSION_VARIANTS)
def test_output_shapes_and_ds_order(variant, mini_resenc_plan, fusion_specs):
    model = _make(variant, mini_resenc_plan, fusion_specs[variant]).eval()
    image, prior = _inputs(mini_resenc_plan)
    with torch.no_grad():
        out = model(image, **_kwargs(model, prior))
    expected = mini_resenc_plan.decoder_output_shapes()
    assert isinstance(out, list) and len(out) == len(expected)
    for got, want in zip(out, expected):
        assert tuple(got.shape[-3:]) == tuple(want)
        assert got.shape[:2] == (image.shape[0], 2)
    depths = [t.shape[-3] for t in out]
    assert depths == sorted(depths, reverse=True)


@pytest.mark.parametrize("variant", FUSION_VARIANTS)
def test_forward_backward_grads_finite_and_nonzero(variant, mini_resenc_plan, fusion_specs):
    torch.manual_seed(0)
    model = _make(variant, mini_resenc_plan, fusion_specs[variant])
    image, prior = _inputs(mini_resenc_plan)
    out = model(image, **_kwargs(model, prior))
    sum(t.float().pow(2).mean() for t in out).backward()
    for param in model.parameters():
        if param.requires_grad and param.grad is not None:
            assert bool(torch.isfinite(param.grad).all())
    # 所有参数梯度必须有限；gate 必须获得非零信号。
    # 注意：M4 的 conditioning head 是零初始化，第一步反传时 zone encoder 侧梯度为 0（数学上必然），
    # 「两步之后 zone encoder / conditioning 都获得非零信号」由下面的两 step 训练测试覆盖。
    probes = [("gate", model.gate), ("conditioning", model.conditioning), ("zone_encoder", model.zone_encoder)]
    for name, module in probes:
        if module is None:
            continue
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads, f"{name} 没有梯度"
        for g in grads:
            assert bool(torch.isfinite(g).all()), f"{name} 存在非有限梯度"
    if model.gate is not None:
        gate_grads = [p.grad for p in model.gate.parameters() if p.grad is not None]
        assert any(float(g.abs().sum()) > 0 for g in gate_grads), "gate 梯度全零"


@pytest.mark.parametrize("variant", FUSION_VARIANTS)
def test_amp_synthetic_forward(variant, mini_resenc_plan, fusion_specs):
    model = _make(variant, mini_resenc_plan, fusion_specs[variant]).eval()
    image, prior = _inputs(mini_resenc_plan)
    with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = model(image, **_kwargs(model, prior))
    for tensor in out if isinstance(out, list) else [out]:
        assert tensor.dtype == torch.bfloat16 and bool(torch.isfinite(tensor.float()).all())


def test_two_optimizer_steps_train_gate_and_conditioning(mini_resenc_plan, fusion_specs):
    """两个 optimizer step 后 gate / conditioning / zone encoder 的参数必须被更新（非零信号）。"""
    torch.manual_seed(0)
    model = _make("m4_conditioned_gate", mini_resenc_plan, fusion_specs["m4_conditioned_gate"])
    image, prior = _inputs(mini_resenc_plan)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    watched = {
        "zone_encoder.stem.weight": model.zone_encoder.stem.weight,
        "gate.logits.weight": model.gate.logits.weight,
        "conditioning.head.weight": model.conditioning.head.weight,
    }
    before = {k: v.detach().clone() for k, v in watched.items()}
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        out = model(image, zonal_prior=prior)
        sum(t.float().pow(2).mean() for t in out).backward()
        optimizer.step()
    for key, param in watched.items():
        assert bool(torch.isfinite(param).all())
        assert not torch.allclose(param.detach(), before[key]), f"{key} 未被训练信号更新"


# --------------------------------------------------------------------------- gate 语义
def test_gate_reads_projection_outputs(mini_resenc_plan, fusion_specs):
    """A 修复验证：gate 收到的张量必须就是三个 modality projection 的输出（不是投影前 stage2 特征）。"""
    model = _make("m2_image_gate", mini_resenc_plan, fusion_specs["m2_image_gate"]).eval()
    captured: dict[str, list[torch.Tensor]] = {"projections": [], "gate_inputs": []}

    def _proj_hook(_m, _i, output):
        captured["projections"].append(output.detach())

    def _gate_pre_hook(_m, inputs):
        captured["gate_inputs"] = [t.detach() for t in inputs]

    handles = [p.register_forward_hook(_proj_hook) for p in model.modality_projections]
    handles.append(model.gate.register_forward_pre_hook(_gate_pre_hook))
    try:
        image, _ = _inputs(mini_resenc_plan)
        with torch.no_grad():
            model(image)
    finally:
        for h in handles:
            h.remove()
    assert len(captured["projections"]) == 3 and len(captured["gate_inputs"]) == 3
    for idx, (gate_in, projection_out) in enumerate(zip(captured["gate_inputs"], captured["projections"])):
        assert torch.allclose(gate_in, projection_out, atol=0, rtol=0), f"gate 输入 {idx} 不是 projection 输出"


def test_m1_weights_strictly_one_third(mini_resenc_plan, fusion_specs):
    model = _make("m1_equal", mini_resenc_plan, fusion_specs["m1_equal"]).eval()
    model.set_diagnostics(True)
    image, _ = _inputs(mini_resenc_plan)
    with torch.no_grad():
        model(image)
    diag = model.gate_diagnostics()
    assert "gate_logits" not in diag  # M1 没有 gate
    weights = diag["final_weights"]
    assert torch.allclose(weights, torch.full_like(weights, 1.0 / 3.0), atol=0, rtol=0)
    assert torch.allclose(weights.sum(dim=1), torch.ones_like(weights[:, 0]))


@pytest.mark.parametrize("variant", ("m2_image_gate", "m3_zone_input", "m4_conditioned_gate"))
def test_weight_sum_to_one_and_zero_init_equality(variant, mini_resenc_plan, fusion_specs):
    model = _make(variant, mini_resenc_plan, fusion_specs[variant]).eval()
    model.set_diagnostics(True)
    image, prior = _inputs(mini_resenc_plan)
    with torch.no_grad():
        model(image, **_kwargs(model, prior))
    weights = model.gate_diagnostics()["final_weights"]
    assert weights.shape[1] == 3 and bool((weights > 0).all())
    assert torch.allclose(weights.sum(dim=1), torch.ones_like(weights[:, 0]), atol=1e-5)
    assert torch.allclose(weights, torch.full_like(weights, 1.0 / 3.0), atol=1e-5)


def test_m4_zero_init_conditioning_is_identity(mini_resenc_plan, fusion_specs):
    model = _make("m4_conditioned_gate", mini_resenc_plan, fusion_specs["m4_conditioned_gate"]).eval()
    model.set_diagnostics(True)
    image, prior = _inputs(mini_resenc_plan)
    with torch.no_grad():
        model(image, zonal_prior=prior)
    diag = model.gate_diagnostics()
    assert torch.allclose(diag["gamma"], torch.zeros_like(diag["gamma"]))
    assert torch.allclose(diag["beta"], torch.zeros_like(diag["beta"]))
    assert torch.allclose(diag["conditioned_logits"], diag["image_logits"], atol=0, rtol=0)


def test_m4_prior_changes_gate_when_conditioning_active(mini_resenc_plan, fusion_specs):
    torch.manual_seed(1)
    model = _make("m4_conditioned_gate", mini_resenc_plan, fusion_specs["m4_conditioned_gate"]).eval()
    with torch.no_grad():
        model.conditioning.head.weight.normal_(0.0, 0.05)
        model.conditioning.head.bias.normal_(0.0, 0.05)
    model.set_diagnostics(True)
    image, prior = _inputs(mini_resenc_plan)
    with torch.no_grad():
        model(image, zonal_prior=prior)
    ref = model.gate_diagnostics()
    with torch.no_grad():
        model(image, zonal_prior=prior.flip(-1))
    got = model.gate_diagnostics()
    assert not torch.allclose(ref["gamma"], got["gamma"])
    assert not torch.allclose(ref["final_weights"], got["final_weights"])


def test_m3_prior_does_not_enter_gate(mini_resenc_plan, fusion_specs):
    model = _make("m3_zone_input", mini_resenc_plan, fusion_specs["m3_zone_input"]).eval()
    model.set_diagnostics(True)
    image, prior = _inputs(mini_resenc_plan)
    with torch.no_grad():
        model(image, zonal_prior=prior)
    ref = model.gate_diagnostics()
    with torch.no_grad():
        model(image, zonal_prior=prior.flip(-1))
    got = model.gate_diagnostics()
    assert torch.allclose(ref["gate_logits"], got["gate_logits"], atol=0, rtol=0)
    assert not torch.allclose(ref["fused_stage2"], got["fused_stage2"])


# --------------------------------------------------------------------------- M3 公式
def test_m3_is_exact_q_without_implicit_residual(mini_resenc_plan, fusion_specs):
    """B 修复验证：M3 输出严格等于 Q([F_fuse, E_z])，不含 F_fuse 残差相加。"""
    model = _make("m3_zone_input", mini_resenc_plan, fusion_specs["m3_zone_input"]).eval()
    model.set_diagnostics(True)
    captured: dict[str, torch.Tensor] = {}

    def _pre_hook(_m, inputs):
        captured["pre"] = inputs[0].detach()

    handle = model.zone_injection.register_forward_pre_hook(_pre_hook)
    image, prior = _inputs(mini_resenc_plan)
    with torch.no_grad():
        model(image, zonal_prior=prior)
    handle.remove()
    diag = model.gate_diagnostics()
    e_z = diag["zone_features"]
    expected = model.zone_injection.act(
        model.zone_injection.norm(model.zone_injection.conv(torch.cat((captured["pre"], e_z), dim=1)))
    )
    assert torch.allclose(diag["fused_stage2"], expected, atol=0, rtol=0)  # 严格相等
    assert not torch.allclose(diag["fused_stage2"], captured["pre"] + expected)  # 无隐式残差
    assert model.spec.fusion["zone_injection"]["implicit_residual"] is False


# --------------------------------------------------------------------------- 输入边界
@pytest.mark.parametrize("variant", ("m3_zone_input", "m4_conditioned_gate"))
def test_illegal_prior_overlap_rejected(variant, mini_resenc_plan, fusion_specs):
    """非法 overlap（PZ=TZ=1）与越界必须在严格校验下失败。"""
    model = _make(variant, mini_resenc_plan, fusion_specs[variant]).eval()
    image, prior = _inputs(mini_resenc_plan)
    bad = prior.clone()
    bad[:, 0] = 1.0
    bad[:, 1] = 1.0  # PZ + TZ = 2 > 1
    with pytest.raises(ValueError, match="重叠"):
        model.validate_inputs(image, bad)
    model.strict_input_validation = True
    with pytest.raises(ValueError, match="重叠"):
        model(image, zonal_prior=bad)
    out_of_range = prior.clone()
    out_of_range[:, 0] *= 3.0
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        model.validate_inputs(image, out_of_range)


@pytest.mark.parametrize("variant", ("m1_equal", "m2_image_gate"))
def test_m1_m2_reject_zonal_input(variant, mini_resenc_plan, fusion_specs):
    model = _make(variant, mini_resenc_plan, fusion_specs[variant]).eval()
    image, prior = _inputs(mini_resenc_plan)
    with pytest.raises(ValueError, match="不读取 PZ/TZ"):
        model(image, zonal_prior=prior)


@pytest.mark.parametrize("variant", ("m3_zone_input", "m4_conditioned_gate"))
def test_m3_m4_require_zonal_and_reject_wg(variant, mini_resenc_plan, fusion_specs):
    model = _make(variant, mini_resenc_plan, fusion_specs[variant]).eval()
    image, prior = _inputs(mini_resenc_plan)
    with pytest.raises(ValueError, match="需要 zonal_prior"):
        model(image)
    wg = torch.cat((prior, torch.zeros_like(prior[:, :1])), dim=1)
    with pytest.raises(ValueError, match="zonal prior 必须是"):
        model(image, zonal_prior=wg)


def test_wrong_grid_and_nonfinite_rejected(mini_resenc_plan, fusion_specs):
    model = _make("m4_conditioned_gate", mini_resenc_plan, fusion_specs["m4_conditioned_gate"]).eval()
    image, prior = _inputs(mini_resenc_plan)
    with pytest.raises(ValueError, match="同一 batch/网格"):
        model(image, zonal_prior=prior[..., : prior.shape[-1] // 2])
    nan = prior.clone()
    nan[:, 1, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="非有限值"):
        model.validate_inputs(image, nan)
    bad_image = image.clone()
    bad_image[0, 0, 0, 0, 0] = float("inf")
    model.strict_input_validation = True
    with pytest.raises(ValueError, match="非有限值"):
        model(bad_image, zonal_prior=prior)
    with pytest.raises(ValueError, match="输入通道"):
        model(image[:, :2], zonal_prior=prior)


# --------------------------------------------------------------------------- 契约 / 结构
def test_shallow_skip_aggregator_rejects_forbidden_inputs(mini_resenc_plan):
    from zonal_reliability_fusion.config.architecture import SHALLOW_SKIP_CONTRACT

    assert set(SHALLOW_SKIP_READS).isdisjoint(set(SHALLOW_SKIP_CONTRACT["forbidden_inputs"]))
    with pytest.raises(ArchitectureConfigError, match="禁止读取"):
        ShallowSkipAggregator(4, 4, stage=0, reads=("t2w_stage_feature", "pz_tz", "hbv_stage_feature"))
    with pytest.raises(ValueError, match="只允许读取"):
        ShallowSkipAggregator(4, 4, stage=0, reads=("t2w_stage_feature", "adc_stage_feature", "wg_feature"))
    with pytest.raises(ValueError, match="stage 0/1"):
        ShallowSkipAggregator(4, 4, stage=2)


def test_stage2_fusion_enters_shared_encoder_and_decoder_skip(mini_resenc_plan, fusion_specs):
    model = _make("m2_image_gate", mini_resenc_plan, fusion_specs["m2_image_gate"]).eval()
    model.set_diagnostics(True)
    captured: dict[str, torch.Tensor] = {}

    def _capture(name):
        def hook(_module, inputs):
            captured[name] = inputs[0].detach()

        return hook

    model.shared_encoder_stages[0].register_forward_pre_hook(_capture("shared_in"))
    model.decoder_stages[0].register_forward_pre_hook(_capture("decoder_in"))
    image, _ = _inputs(mini_resenc_plan)
    with torch.no_grad():
        model(image)
    fused = model.gate_diagnostics()["fused_stage2"]
    assert torch.allclose(captured["shared_in"], fused, atol=0, rtol=0)
    features_skip = mini_resenc_plan.features_per_stage[mini_resenc_plan.n_stages - 2]
    assert torch.allclose(captured["decoder_in"][:, features_skip:], fused, atol=0, rtol=0)


def test_config_topology_matches_real_modules(fusion_specs):
    """C 修复验证：配置声明的 topology 必须与真实模块一致（含 InstanceNorm3d）。"""
    gate_cfg = fusion_specs["m2_image_gate"].fusion["gate"]
    assert gate_cfg["logits_topology"] == ("conv1x1x1", "instance_norm3d", "leaky_relu", "conv1x1x1")
    assert gate_cfg["logits_source"] == "modality_projection_outputs"
    zone_cfg = fusion_specs["m4_conditioned_gate"].fusion["zone_encoder"]
    assert zone_cfg["stem_topology"] == ("conv3d_same_padding", "instance_norm3d", "leaky_relu")
    assert "instance_norm3d" in zone_cfg["block_topology"]
    assert zone_cfg["block_zero_init_last_norm"] is False
    cond_cfg = fusion_specs["m4_conditioned_gate"].fusion["conditioning"]
    assert "instance_norm3d" in cond_cfg["head_topology"]
    inj_cfg = fusion_specs["m3_zone_input"].fusion["zone_injection"]
    assert inj_cfg["type"] == "post_fusion_projection" and inj_cfg["implicit_residual"] is False


# --------------------------------------------------------------------------- MACs
@pytest.mark.parametrize("variant", FUSION_VARIANTS)
def test_analytic_macs_match_hook_counting(variant, mini_resenc_plan, fusion_specs):
    model = _make(variant, mini_resenc_plan, fusion_specs[variant]).eval()
    analytic = model.macs_breakdown()
    hooked = model.macs_breakdown_with_hooks()
    assert set(FUSION_MAC_GROUPS).issubset(analytic.keys())
    assert hooked["unassigned"] == 0
    for group in FUSION_MAC_GROUPS:
        assert analytic[group] == hooked[group], f"{variant}.{group}: analytic={analytic[group]} hook={hooked[group]}"
    assert analytic["total"] == hooked["total"] > 0
    assert analytic["total"] > analytic["shared_encoder_and_decoder"] > 0


def test_macs_breakdown_is_model_specific(mini_resenc_plan, fusion_specs):
    """M0 的 MACs 不得被直接当作 M1–M4 的数字；各模型分组必须反映实际结构。"""
    values = {v: _make(v, mini_resenc_plan, fusion_specs[v]).macs_breakdown() for v in FUSION_VARIANTS}
    assert values["m1_equal"]["gate"] == 0
    assert values["m2_image_gate"]["gate"] > 0 and values["m2_image_gate"]["zone_encoder"] == 0
    assert values["m3_zone_input"]["zone_encoder"] == values["m4_conditioned_gate"]["zone_encoder"] > 0
    assert values["m3_zone_input"]["zone_injection"] > 0 and values["m4_conditioned_gate"]["conditioning"] > 0
    assert values["m2_image_gate"]["total"] != values["m1_equal"]["total"]


# --------------------------------------------------------------------------- 诊断模式
def test_diagnostics_modes(mini_resenc_plan, fusion_specs):
    model = _make("m2_image_gate", mini_resenc_plan, fusion_specs["m2_image_gate"]).eval()
    image, _ = _inputs(mini_resenc_plan)
    with torch.no_grad():
        model(image)
    assert model.gate_diagnostics() == {} and model.diagnostics_stats() == {}  # 默认 off
    model.set_diagnostics_mode("stats")
    with torch.no_grad():
        model(image)
    stats = model.diagnostics_stats()
    assert stats and all(isinstance(v, float) for v in stats.values())
    assert model.gate_diagnostics() == {}  # stats 模式不保留完整张量
    model.set_diagnostics_mode("cpu")
    with torch.no_grad():
        model(image)
    tensors = model.gate_diagnostics()
    assert tensors and all(t.device.type == "cpu" and not t.requires_grad for t in tensors.values())
    with pytest.raises(ValueError, match="off/stats/cpu"):
        model.set_diagnostics_mode("bogus")


# --------------------------------------------------------------------------- 身份 / 参数量
def test_architecture_hashes_distinct_and_m0_unchanged(fusion_specs):
    hashes = {v: s.architecture_sha256() for v, s in fusion_specs.items()}
    assert len(set(hashes.values())) == len(FUSION_VARIANTS)
    m0 = load_architecture_config("configs/architectures/m0_resenc_v23.yaml")
    assert m0.fusion is None
    assert m0.architecture_sha256() == "8bb5127e5c9a99ca1a0631cb365cdcf0b866441ac225911e17faa6bc995bbee5"
    assert all(h != m0.architecture_sha256() for h in hashes.values())
    for variant, spec in fusion_specs.items():
        assert spec.identity()["fusion_variant"] == variant
        assert spec.identity()["model_id"] == FUSION_VARIANT_TO_MODEL_ID[variant]


def test_cross_model_identity_rejected(mini_resenc_plan, fusion_specs):
    m1 = fusion_specs["m1_equal"].identity()
    m2 = fusion_specs["m2_image_gate"].identity()
    verify_architecture_identity(m1, m1)
    with pytest.raises(ArchitectureIdentityError):
        verify_architecture_identity(m1, m2)
    with pytest.raises(ArchitectureIdentityError):
        verify_architecture_identity(m1, None)


def test_parameter_breakdown_and_capacity_report(mini_resenc_plan, fusion_specs):
    breakdowns = {}
    for variant, spec in fusion_specs.items():
        model = _make(variant, mini_resenc_plan, spec)
        bd = model.parameter_breakdown()
        assert bd["total"] == model.num_parameters() == sum(v for k, v in bd.items() if k != "total")
        breakdowns[variant] = bd
    assert breakdowns["m1_equal"]["gate"] == breakdowns["m1_equal"]["zone_encoder"] == 0
    assert breakdowns["m2_image_gate"]["gate"] > 0 and breakdowns["m2_image_gate"]["zone_encoder"] == 0
    assert breakdowns["m3_zone_input"]["zone_encoder"] == breakdowns["m4_conditioned_gate"]["zone_encoder"] > 0
    assert breakdowns["m3_zone_input"]["zone_injection"] > 0 and breakdowns["m3_zone_input"]["conditioning"] == 0
    assert breakdowns["m4_conditioned_gate"]["conditioning"] > 0


def test_model_build_reproducible_with_fixed_seed(mini_resenc_plan, fusion_specs):
    def _checksum(model) -> float:
        return float(sum(p.detach().double().sum() for p in model.parameters()))

    torch.manual_seed(7)
    a = _make("m4_conditioned_gate", mini_resenc_plan, fusion_specs["m4_conditioned_gate"])
    torch.manual_seed(7)
    b = _make("m4_conditioned_gate", mini_resenc_plan, fusion_specs["m4_conditioned_gate"])
    assert _checksum(a) == _checksum(b)
    assert a.architecture_identity() == b.architecture_identity()
    assert a.macs_breakdown() == b.macs_breakdown()


def test_model_identity_fields(mini_resenc_plan, fusion_specs):
    model = _make("m4_conditioned_gate", mini_resenc_plan, fusion_specs["m4_conditioned_gate"])
    identity = model.model_identity()
    assert identity["model_id"] == "M4" and identity["backbone_id"] == "residual_encoder_v23"
    assert identity["fusion_variant"] == "m4_conditioned_gate" and identity["fusion_stage"] == 2
    assert identity["gate_type"] == "image_driven_softmax_over_modalities"
    assert identity["conditioning_type"] == "bounded_conditional_affine"
    assert identity["zone_encoder"]["features"] == 32
    assert model.REQUIRED_INPUT_KEYS == ("image", INPUT_KEY_ZONAL)


# --------------------------------------------------------------------------- 工厂分派
def _factory_cfg(tmp_path, plan, *, model_id, architecture, fusion_variant):
    doc = _fusion_arch_doc(plan)
    arch_path = tmp_path / "fusion_arch.yaml"
    arch_path.write_text(yaml.safe_dump(doc, sort_keys=False))
    ref = ArchitectureRef(
        config_path=str(arch_path),
        architecture_name=doc["architecture_name"],
        architecture_version=doc["architecture_version"],
        encoder_type=doc["encoder_type"],
        fusion_variant=fusion_variant,
    )
    return types.SimpleNamespace(
        experiment=types.SimpleNamespace(model_id=model_id),
        architecture=None if architecture is False else ref,
        model=types.SimpleNamespace(in_channels=3, num_classes=2, deep_supervision=True),
        resolve_path=lambda p: p,
    )


@pytest.mark.parametrize("model_id", ("M1", "M2", "M3", "M4"))
def test_build_model_dispatch(model_id, tmp_path, mini_resenc_plan):
    variant = {"M1": "m1_equal", "M2": "m2_image_gate", "M3": "m3_zone_input", "M4": "m4_conditioned_gate"}[model_id]
    cfg = _factory_cfg(tmp_path, mini_resenc_plan, model_id=model_id, architecture=True, fusion_variant=variant)
    model = build_model(cfg, mini_resenc_plan)
    assert type(model) is VARIANT_TO_CLASS[variant] and model.MODEL_ID == model_id


def test_build_model_rejects_bad_ids(tmp_path, mini_resenc_plan):
    cfg = _factory_cfg(tmp_path, mini_resenc_plan, model_id="M9", architecture=True, fusion_variant="m1_equal")
    with pytest.raises(ArchitectureConfigError, match="未知 experiment.model_id"):
        build_model(cfg, mini_resenc_plan)
    cfg2 = _factory_cfg(tmp_path, mini_resenc_plan, model_id="M2", architecture=True, fusion_variant="m1_equal")
    with pytest.raises(ArchitectureConfigError, match="不一致"):
        build_model(cfg2, mini_resenc_plan)
    cfg3 = _factory_cfg(tmp_path, mini_resenc_plan, model_id="M3", architecture=True, fusion_variant=None)
    with pytest.raises(ArchitectureConfigError, match="必须显式声明 fusion_variant"):
        build_model(cfg3, mini_resenc_plan)
    cfg4 = _factory_cfg(tmp_path, mini_resenc_plan, model_id="M4", architecture=False, fusion_variant=None)
    with pytest.raises(ArchitectureConfigError, match="需要 architecture 块"):
        build_model(cfg4, mini_resenc_plan)
    cfg5 = _factory_cfg(tmp_path, mini_resenc_plan, model_id="M0", architecture=True, fusion_variant="m1_equal")
    with pytest.raises(ArchitectureConfigError, match="M0 不得声明"):
        build_model(cfg5, mini_resenc_plan)


# --------------------------------------------------------------------------- 真实配置
def test_real_fusion_config_all_variants_resolve():
    spec = load_architecture_config(REAL_FUSION_CONFIG)
    assert spec.fusion is None and spec.architecture_version == "v2.4"
    resolved = {v: resolve_fusion_spec(spec, v) for v in FUSION_VARIANTS}
    assert len({s.architecture_sha256() for s in resolved.values()}) == 4
    assert resolved["m3_zone_input"].fusion["zone_encoder"] == resolved["m4_conditioned_gate"].fusion["zone_encoder"]
    assert resolved["m2_image_gate"].fusion["gate"] == resolved["m4_conditioned_gate"].fusion["gate"]
    with pytest.raises(ArchitectureConfigError):
        resolve_fusion_spec(load_architecture_config("configs/architectures/m0_resenc_v23.yaml"), "m1_equal")


def test_fusion_family_declarations_enforced(fusion_doc):
    bad_presence = copy.deepcopy(fusion_doc)
    bad_presence["fusion_family"]["variants"]["m3_zone_input"]["zone_injection"] = None
    with pytest.raises(ArchitectureConfigError, match="组件存在性"):
        architecture_spec_from_mapping(bad_presence)

    bad_topology = copy.deepcopy(fusion_doc)
    bad_topology["fusion_family"]["variants"]["m2_image_gate"]["gate"]["logits_topology"] = ["conv1x1x1", "conv1x1x1"]
    with pytest.raises(ArchitectureConfigError, match="未实现/不支持"):
        architecture_spec_from_mapping(bad_topology)

    bad_model_id = copy.deepcopy(fusion_doc)
    bad_model_id["fusion_family"]["variants"]["m2_image_gate"]["model_id"] = "M3"
    with pytest.raises(ArchitectureConfigError, match="model_id"):
        architecture_spec_from_mapping(bad_model_id)

    bad_encoder = copy.deepcopy(fusion_doc)
    bad_encoder["fusion_family"]["variants"]["m4_conditioned_gate"]["zone_encoder"]["features"] = 16
    with pytest.raises(ArchitectureConfigError, match="未实现/不支持"):
        architecture_spec_from_mapping(bad_encoder)

    bad_m3 = copy.deepcopy(fusion_doc)
    bad_m3["fusion_family"]["variants"]["m3_zone_input"]["zone_injection"]["implicit_residual"] = True
    with pytest.raises(ArchitectureConfigError, match="未实现/不支持"):
        architecture_spec_from_mapping(bad_m3)
