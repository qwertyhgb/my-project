"""gate 网络与原生 nnU-Net backbone 包装器的纯合成 CPU 测试（不读取真实数据）。

覆盖：image_gate 3 通道（#4）、anatomy_gate 严格 5 通道（#5）、零初始化 scales==1（#6）、
零初始化 gated MRI 与原 MRI 逐值一致（#7）、deep supervision 输出格式（#8）、
set_deep_supervision_enabled 切换（#9）、strict state_dict save/load（#10）、
anatomy prior clamp 到 [0,1]（#11）。

浅层序列特异特征融合（Research Plan §8.10，``FeatureFusionNNUNet``）：三个 stem 参数不共享且
保持空间尺寸、backbone 只接收三通道、零初始化 feature gate 与同 stem/投影/backbone 的
feature no-gate 逐值一致（但**不**等价于原生 nnU-Net）、``W``/``S`` 的非负与和约束、每个序列只有
一个共享尺度场、image gate 响应 MRI 变化、anatomy gate 响应具有空间结构的 PZ/TZ（且 stem 不受其
影响）、PZ/TZ clamp、deep supervision、strict state_dict 与 image/anatomy 变体隔离、
gate 参数差被显式记录。
"""

from __future__ import annotations

import pytest
import torch

from zonal_reliability_fusion.nnunet.networks import (
    FEATURE_STEM_CHANNELS,
    FeatureFusionNNUNet,
    GatedNNUNet,
    ShallowSequenceStem,
    SpatialModalityReliabilityGate,
    feature_gate_parameter_delta,
)
from zonal_reliability_fusion.nnunet.trainers import (
    nnUNetTrainerPICAI_AnatomyGate,
    nnUNetTrainerPICAI_FeatureAnatomyGate_PositiveSampling_NoFFT,
    nnUNetTrainerPICAI_FeatureImageGate_PositiveSampling_NoFFT,
    nnUNetTrainerPICAI_FeatureNoGate_PositiveSampling_NoFFT,
    nnUNetTrainerPICAI_FLCE_NoFFT,
    nnUNetTrainerPICAI_ImageGate,
)


def _build_image(synthetic_arch, deep_supervision=True):
    return nnUNetTrainerPICAI_ImageGate.build_network_architecture(
        synthetic_arch["architecture_class_name"],
        synthetic_arch["arch_init_kwargs"],
        synthetic_arch["arch_init_kwargs_req_import"],
        3,
        2,
        deep_supervision,
    )


def _build_anatomy(synthetic_arch, deep_supervision=True):
    return nnUNetTrainerPICAI_AnatomyGate.build_network_architecture(
        synthetic_arch["architecture_class_name"],
        synthetic_arch["arch_init_kwargs"],
        synthetic_arch["arch_init_kwargs_req_import"],
        5,
        2,
        deep_supervision,
    )


# --------------------------------------------------------------------------- #4 image_gate 通道
def test_image_gate_accepts_three_channels(synthetic_arch):
    net = _build_image(synthetic_arch, deep_supervision=False)
    net.eval()
    with torch.no_grad():
        out = net(torch.randn(2, 3, 8, 16, 16))
    assert isinstance(out, torch.Tensor)
    assert out.shape[1] == 2


def test_image_gate_rejects_non_three_channels(synthetic_arch):
    net = _build_image(synthetic_arch, deep_supervision=False)
    with pytest.raises(ValueError):
        net(torch.randn(1, 5, 8, 16, 16))


# --------------------------------------------------------------------------- #5 anatomy_gate 严格 5 通道
@pytest.mark.parametrize("bad_channels", [3, 4, 6])
def test_anatomy_gate_strictly_requires_five_channels(synthetic_arch, bad_channels):
    net = _build_anatomy(synthetic_arch, deep_supervision=False)
    with pytest.raises(ValueError):
        net(torch.randn(1, bad_channels, 8, 16, 16))


def test_anatomy_gate_accepts_five_channels(synthetic_arch):
    net = _build_anatomy(synthetic_arch, deep_supervision=False)
    net.eval()
    with torch.no_grad():
        out = net(torch.randn(2, 5, 8, 16, 16))
    assert isinstance(out, torch.Tensor)
    assert out.shape[1] == 2


def test_anatomy_builder_rejects_wrong_dataset_channel_count(synthetic_arch):
    """build_network_architecture 在通道数不符时必须 fail-closed（image=3 / anatomy=5）。"""
    with pytest.raises(ValueError):
        nnUNetTrainerPICAI_AnatomyGate.build_network_architecture(
            synthetic_arch["architecture_class_name"],
            synthetic_arch["arch_init_kwargs"],
            synthetic_arch["arch_init_kwargs_req_import"],
            3,
            2,
            True,
        )
    with pytest.raises(ValueError):
        nnUNetTrainerPICAI_ImageGate.build_network_architecture(
            synthetic_arch["architecture_class_name"],
            synthetic_arch["arch_init_kwargs"],
            synthetic_arch["arch_init_kwargs_req_import"],
            5,
            2,
            True,
        )


# --------------------------------------------------------------------------- #6 零初始化 scales == 1
def test_gate_zero_init_scales_are_one(synthetic_arch):
    net = _build_image(synthetic_arch, deep_supervision=False)
    x = torch.randn(2, 3, 8, 16, 16)
    scales = net.gate(x)
    assert scales.shape[1] == 3
    assert torch.allclose(scales, torch.ones_like(scales), atol=1e-6)
    # 零初始化时 W = 1/3（S = 3W = 1 的等价表述）
    weights = net.gate.compute_weights(x)
    assert torch.allclose(weights, torch.full_like(weights, 1 / 3), atol=1e-6)


def test_gate_last_conv_is_zero_initialized():
    gate = SpatialModalityReliabilityGate(input_channels=5)
    assert torch.count_nonzero(gate.conv2.weight) == 0
    assert torch.count_nonzero(gate.conv2.bias) == 0


def test_gate_compute_weights_nonnegative_and_sums_to_one(synthetic_arch):
    """``compute_weights`` 返回的 W 必须非负且通道和为 1（非平凡权重下也成立）。"""
    net = _build_image(synthetic_arch, deep_supervision=False)
    torch.manual_seed(7)
    with torch.no_grad():
        # 扰动末层：让 logits 非零，softmax 不再恒为 1/3
        net.gate.conv2.weight.normal_(0.0, 0.3)
        net.gate.conv2.bias.normal_(0.0, 0.3)
    x = torch.randn(2, 3, 8, 16, 16)
    weights = net.gate.compute_weights(x)
    assert weights.shape[1] == 3
    assert bool((weights >= 0).all())
    assert torch.allclose(weights.sum(dim=1), torch.ones_like(weights[:, 0]))
    # 非平凡的 softmax：不是被扰动前的常数 1/3
    assert not torch.allclose(weights, torch.full_like(weights, 1 / 3), atol=1e-3)
    # forward 返回的就是 S = 3 * W（真正与 MRI 相乘的尺度）
    assert torch.allclose(net.gate(x), 3.0 * weights)


def test_gate_compute_scales_nonnegative_and_sums_to_three(synthetic_arch):
    """尺度 S 必须非负且通道和为 3（零初始化时严格为 1）。"""
    net = _build_image(synthetic_arch, deep_supervision=False)
    torch.manual_seed(11)
    with torch.no_grad():
        net.gate.conv2.weight.normal_(0.0, 0.5)
        net.gate.conv2.bias.normal_(0.0, 0.5)
    x = torch.randn(2, 3, 8, 16, 16)
    scales = net.gate(x)
    assert bool((scales >= 0).all())
    assert torch.allclose(scales.sum(dim=1), torch.full_like(scales[:, 0], 3.0))


# --------------------------------------------------------------------------- #7 零初始化恒等
def test_zero_init_gated_mri_equals_original(synthetic_arch):
    net = _build_image(synthetic_arch, deep_supervision=False)
    net.eval()
    x = torch.randn(2, 3, 8, 16, 16)
    with torch.no_grad():
        gated = net(x)
        reference = net.backbone(x)  # 原生 backbone 直接吃未加权 MRI
    assert torch.allclose(gated, reference, atol=1e-6)


def test_zero_init_anatomy_equals_backbone_on_mri(synthetic_arch):
    net = _build_anatomy(synthetic_arch, deep_supervision=False)
    net.eval()
    x = torch.randn(2, 5, 8, 16, 16)
    with torch.no_grad():
        gated = net(x)
        reference = net.backbone(x[:, :3])
    assert torch.allclose(gated, reference, atol=1e-6)


# --------------------------------------------------------------------------- #8 deep supervision 输出格式
def test_wrapper_preserves_deep_supervision_output_format(synthetic_arch):
    net = _build_image(synthetic_arch, deep_supervision=True)
    net.eval()
    x = torch.randn(2, 3, 8, 16, 16)
    with torch.no_grad():
        out = net(x)
        reference = net.backbone(x)
    assert isinstance(out, list) and isinstance(reference, list)
    assert len(out) == len(reference)
    for a, b in zip(out, reference):
        assert isinstance(a, torch.Tensor)
        assert a.shape == b.shape


# --------------------------------------------------------------------------- #9 set_deep_supervision_enabled 切换
def test_set_deep_supervision_enabled_toggles_output(synthetic_arch):
    net = _build_image(synthetic_arch, deep_supervision=True)
    net.eval()
    x = torch.randn(1, 3, 8, 16, 16)
    # 复刻 nnUNetTrainer.set_deep_supervision_enabled 的行为：mod.decoder.deep_supervision = flag
    assert net.decoder is net.backbone.decoder
    net.decoder.deep_supervision = True
    with torch.no_grad():
        assert isinstance(net(x), list)
    net.decoder.deep_supervision = False
    with torch.no_grad():
        assert isinstance(net(x), torch.Tensor)


# --------------------------------------------------------------------------- #10 strict state_dict
def test_strict_state_dict_roundtrip(synthetic_arch, tmp_path):
    net = _build_image(synthetic_arch, deep_supervision=True)
    ckpt = tmp_path / "ckpt.pth"
    torch.save({"network_weights": net.state_dict()}, ckpt)

    net2 = _build_image(synthetic_arch, deep_supervision=True)
    state = torch.load(ckpt, weights_only=False)["network_weights"]
    net2.load_state_dict(state, strict=True)  # 不应抛异常

    net.eval()
    net2.eval()
    x = torch.randn(1, 3, 8, 16, 16)
    with torch.no_grad():
        o1, o2 = net(x), net2(x)
    assert all(torch.allclose(a, b, atol=1e-6) for a, b in zip(o1, o2))


def test_strict_state_dict_rejects_mismatched_variant(synthetic_arch):
    image_net = _build_image(synthetic_arch, deep_supervision=False)
    anatomy_net = _build_anatomy(synthetic_arch, deep_supervision=False)
    with pytest.raises(RuntimeError):
        anatomy_net.load_state_dict(image_net.state_dict(), strict=True)


# --------------------------------------------------------------------------- #11 anatomy prior clamp
def test_anatomy_prior_clamped_to_unit_interval(synthetic_arch):
    net = _build_anatomy(synthetic_arch, deep_supervision=False)
    x = torch.randn(1, 5, 8, 16, 16)
    # 人为把 PZ/TZ 拉到 [0,1] 之外（模拟 preprocessing 后 spline 过冲）
    x[:, 3] = x[:, 3] * 5.0 + 3.0
    x[:, 4] = x[:, 4] * 5.0 - 3.0
    gate_input = net._gate_input(x)
    prior = gate_input[:, 3:]
    assert float(prior.min()) >= 0.0
    assert float(prior.max()) <= 1.0
    # MRI 通道不被 clamp 改动
    assert torch.equal(gate_input[:, :3], x[:, :3])


def test_compute_gate_scales_reuses_gate_input_and_clamps_prior(synthetic_arch):
    """compute_gate_scales 必须先经 _gate_input：prior clamp 与通道检查不可绕过。

    注意：gate 内含 InstanceNorm3d，**空间常数**的 prior 偏移会被整块归一化消除，
    因此这里用**空间变化**的越界图案才能观察到 clamp。
    """
    net = _build_anatomy(synthetic_arch, deep_supervision=False)
    torch.manual_seed(21)
    with torch.no_grad():
        # 非平凡 gate：否则 zero-init 恒为 1，clamp 前后无法区分
        net.gate.conv2.weight.normal_(0.0, 0.3)
        net.gate.conv2.bias.normal_(0.0, 0.3)
    x = torch.randn(1, 5, 8, 16, 16)
    pattern = torch.linspace(-5.0, 7.0, 8 * 16 * 16).reshape(8, 16, 16)

    x_oob = x.clone()
    x_oob[0, 3] = pattern  # 含越界值 [-5, 7]
    x_oob[0, 4] = -pattern
    x_valid = x.clone()
    x_valid[0, 3] = pattern.clamp(0.0, 1.0)  # 显式 clamp 后的合法 prior
    x_valid[0, 4] = (-pattern).clamp(0.0, 1.0)

    scales_oob = net.compute_gate_scales(x_oob)
    assert torch.equal(
        scales_oob, net.compute_gate_scales(x_valid)
    )  # 先 clamp 再进 gate
    # 控制：交换 PZ/TZ 两个通道（仍是合法 [0,1] 值）必须改变 scales，
    # 证明 gate 确实读取 prior（否则上面的相等断言毫无意义）
    x_swapped = x_valid.clone()
    x_swapped[0, 3], x_swapped[0, 4] = x_valid[0, 4].clone(), x_valid[0, 3].clone()
    assert not torch.allclose(net.compute_gate_scales(x_swapped), scales_oob)
    # 通道检查不会因新方法被绕过
    with pytest.raises(ValueError):
        net.compute_gate_scales(torch.randn(1, 3, 8, 16, 16))


# --------------------------------------------------------------------------- forward 数学行为
@pytest.mark.parametrize("builder", [_build_image, _build_anatomy])
def test_forward_matches_reference_gating(synthetic_arch, builder):
    """forward 与修补前数学行为一致：backbone(x_前3通道 * S)，且 S 确实生效。"""
    in_channels = 3 if builder is _build_image else 5
    net = builder(synthetic_arch, deep_supervision=False)
    torch.manual_seed(99)
    with torch.no_grad():
        net.gate.conv2.weight.normal_(0.0, 0.3)
        net.gate.conv2.bias.normal_(0.0, 0.3)
    net.eval()
    x = torch.randn(2, in_channels, 8, 16, 16)
    with torch.no_grad():
        scales = net.compute_gate_scales(x)
        reference = net.backbone(x[:, :3] * scales)
        out = net(x)
    assert torch.allclose(out, reference, atol=1e-6)
    # 控制：非恒等 scales 时结果必须不同于直接 backbone(x)（证明 scales 真的被乘上）
    assert not torch.allclose(scales, torch.ones_like(scales), atol=1e-3)
    with torch.no_grad():
        plain = net.backbone(x[:, :3])
    assert not torch.allclose(out, plain)


# --------------------------------------------------------------------------- 包装器契约
def test_wrapper_does_not_copy_unet_and_proxies_backbone(synthetic_arch):
    net = _build_image(synthetic_arch, deep_supervision=False)
    assert isinstance(net, GatedNNUNet)
    # backbone 是原生 PlainConvUNet（不是项目自写的第二套 U-Net）
    assert type(net.backbone).__name__ == "PlainConvUNet"
    assert hasattr(net.backbone, "encoder") and hasattr(net.backbone, "decoder")
    # compute_conv_feature_map_size 代理到 backbone
    assert net.compute_conv_feature_map_size(
        [8, 16, 16]
    ) == net.backbone.compute_conv_feature_map_size([8, 16, 16])


# ===========================================================================
# 浅层序列特异特征融合（Research Plan §8.10）
# ===========================================================================
def _build_feature(synthetic_arch, trainer, in_channels, deep_supervision=False):
    return trainer.build_network_architecture(
        synthetic_arch["architecture_class_name"],
        synthetic_arch["arch_init_kwargs"],
        synthetic_arch["arch_init_kwargs_req_import"],
        in_channels,
        2,
        deep_supervision,
    )


def _feature_no_gate(synthetic_arch, deep_supervision=False):
    return _build_feature(
        synthetic_arch,
        nnUNetTrainerPICAI_FeatureNoGate_PositiveSampling_NoFFT,
        3,
        deep_supervision,
    )


def _feature_image_gate(synthetic_arch, deep_supervision=False):
    return _build_feature(
        synthetic_arch,
        nnUNetTrainerPICAI_FeatureImageGate_PositiveSampling_NoFFT,
        3,
        deep_supervision,
    )


def _feature_anatomy_gate(synthetic_arch, deep_supervision=False):
    return _build_feature(
        synthetic_arch,
        nnUNetTrainerPICAI_FeatureAnatomyGate_PositiveSampling_NoFFT,
        5,
        deep_supervision,
    )


def _copy_shared_weights(source, target):
    """把 no-gate 的 stem/投影/backbone 权重拷到 gate 版本（gate 保持零初始化）。

    返回缺失键；断言缺失键**只有** gate（即两份网络的其余结构逐键同形）。
    """
    result = target.load_state_dict(source.state_dict(), strict=False)
    assert list(result.unexpected_keys) == []
    assert all(key.startswith("gate.") for key in result.missing_keys)
    assert result.missing_keys  # gate 版本必须有新增的 gate 键
    return list(result.missing_keys)


def _perturb_feature_gate(net, seed=7, std=0.3):
    """把 feature gate 的末层从零初始化扰动成非平凡，使 S 不再恒为 1。"""
    torch.manual_seed(seed)
    with torch.no_grad():
        net.gate.conv2.weight.normal_(0.0, std)
        net.gate.conv2.bias.normal_(0.0, std)


# ------------------------------------------------- 三个 stem：不共享、保持尺寸
def test_feature_has_three_independent_stems(synthetic_arch):
    net = _feature_no_gate(synthetic_arch)
    assert isinstance(net.stems, torch.nn.ModuleList)
    assert len(net.stems) == 3
    assert all(isinstance(stem, ShallowSequenceStem) for stem in net.stems)
    # 参数不共享：不同 tensor 对象 + 不同存储 + 权重不相等
    for attr in ("conv1", "conv2"):
        params = [getattr(stem, attr).weight for stem in net.stems]
        assert len({id(p) for p in params}) == 3
        assert len({p.data_ptr() for p in params}) == 3
    assert not torch.equal(net.stems[0].conv1.weight, net.stems[1].conv1.weight)
    assert not torch.equal(net.stems[1].conv2.weight, net.stems[2].conv2.weight)


def test_feature_stem_preserves_spatial_shape(synthetic_arch):
    net = _feature_no_gate(synthetic_arch)
    # 奇数尺寸也要逐值保持（padding = kernel // 2，stride = 1）
    x = torch.randn(2, 3, 9, 16, 17)
    for index, stem in enumerate(net.stems):
        with torch.no_grad():
            h = stem(x[:, index : index + 1])
        assert h.shape == (2, FEATURE_STEM_CHANNELS, 9, 16, 17)
    # 两层 3×3×3 卷积的名义感受野 = 5×5×5，且确实没有下采样
    assert net.stems[0].conv1.kernel_size == (3, 3, 3)
    assert net.stems[0].conv1.stride == (1, 1, 1)
    assert net.stems[0].conv2.stride == (1, 1, 1)


def test_feature_stem_rejects_multi_channel_input(synthetic_arch):
    net = _feature_no_gate(synthetic_arch)
    with pytest.raises(ValueError):
        net.stems[0](torch.randn(1, 3, 8, 16, 16))


# ------------------------------------------------- backbone 只接收三通道
@pytest.mark.parametrize(
    "builder,in_channels",
    [
        (_feature_no_gate, 3),
        (_feature_image_gate, 3),
        (_feature_anatomy_gate, 5),
    ],
)
def test_feature_backbone_receives_exactly_three_channels(
    synthetic_arch, builder, in_channels
):
    net = builder(synthetic_arch)
    net.eval()
    seen: list[int] = []
    handle = net.backbone.register_forward_pre_hook(
        lambda module, args: seen.append(args[0].shape[1])
    )
    try:
        with torch.no_grad():
            net(torch.randn(1, in_channels, 8, 16, 16))
    finally:
        handle.remove()
    assert seen == [3]  # 骨干只看到三通道融合特征
    # 投影层是 1×1×1，且输出通道恰好为 3
    assert net.projection.out_channels == 3
    assert net.projection.kernel_size == (1, 1, 1)


def test_feature_projection_input_is_concatenated_stems(synthetic_arch):
    net = _feature_no_gate(synthetic_arch)
    assert net.projection.in_channels == FEATURE_STEM_CHANNELS * 3


# ------------------------------------------------- 零初始化：等价于 feature no-gate
@pytest.mark.parametrize(
    "builder,in_channels",
    [(_feature_image_gate, 3), (_feature_anatomy_gate, 5)],
)
def test_zero_init_feature_gate_equals_feature_no_gate(
    synthetic_arch, builder, in_channels
):
    """共享 stem/投影/backbone 权重时，零初始化 feature gate 与 no-gate 逐值一致。"""
    plain = _feature_no_gate(synthetic_arch)
    gated = builder(synthetic_arch)
    _copy_shared_weights(plain, gated)
    # 零初始化成立：S 恒为 1
    x = torch.randn(2, in_channels, 8, 16, 16)
    scales = gated.compute_gate_scales(x)
    assert torch.allclose(scales, torch.ones_like(scales), atol=1e-6)

    plain.eval()
    gated.eval()
    with torch.no_grad():
        out_plain = plain(x[:, :3])
        out_gated = gated(x)
    assert torch.allclose(out_plain, out_gated, atol=1e-6)
    # anatomy 额外检查：完全不同的 PZ/TZ 不改变输出（零初始化下 prior 无法影响结果）
    x_other = x.clone()
    x_other[:, 3:] = 1.0 - x_other[:, 3:]
    gated.eval()
    with torch.no_grad():
        assert torch.allclose(gated(x_other), out_plain, atol=1e-6)


def test_zero_init_feature_net_is_not_equivalent_to_native_nnunet(synthetic_arch):
    """零初始化只保证与 feature no-gate 等价，**不**等价于原生输入级 nnU-Net。

    为排除"权重不同"这一平凡解释，这里让两者 backbone 权重**逐值相同**：残余差异只可能来自
    stem 与投影层本身。
    """
    native = nnUNetTrainerPICAI_FLCE_NoFFT.build_network_architecture(
        synthetic_arch["architecture_class_name"],
        synthetic_arch["arch_init_kwargs"],
        synthetic_arch["arch_init_kwargs_req_import"],
        3,
        2,
        False,
    )
    feature = _feature_no_gate(synthetic_arch)
    # 原生网络的键没有包装前缀，这里显式映射到 feature 网络的 backbone 子模块
    native_state = native.state_dict()
    result = feature.load_state_dict(
        {f"backbone.{key}": value for key, value in native_state.items()}, strict=False
    )
    assert list(result.unexpected_keys) == []
    # 缺失键只有 stem 与投影（feature no-gate 没有 gate）
    assert all(key.startswith(("stems.", "projection.")) for key in result.missing_keys)
    # backbone 权重逐值相同，残余差异只可能来自 stem + 投影
    for key, value in native_state.items():
        assert torch.equal(feature.state_dict()[f"backbone.{key}"], value)

    native.eval()
    feature.eval()
    x = torch.randn(2, 3, 8, 16, 16)
    with torch.no_grad():
        out_native = native(x)
        out_feature = feature(x)
    assert not torch.allclose(out_native, out_feature, atol=1e-4)


# ------------------------------------------------- W / S 的数学性质
@pytest.mark.parametrize(
    "builder,in_channels", [(_feature_image_gate, 3), (_feature_anatomy_gate, 5)]
)
def test_feature_gate_weights_and_scales_constraints(
    synthetic_arch, builder, in_channels
):
    net = builder(synthetic_arch)
    _perturb_feature_gate(net, seed=5)
    x = torch.randn(2, in_channels, 8, 16, 16)
    weights = net.compute_weights(x)
    scales = net.compute_gate_scales(x)
    assert weights.shape[1] == 3  # 恰好三个 logit
    assert scales.shape[1] == 3
    assert bool((weights >= 0).all())
    assert bool((scales >= 0).all())
    assert torch.allclose(weights.sum(dim=1), torch.ones_like(weights[:, 0]))
    assert torch.allclose(scales.sum(dim=1), torch.full_like(scales[:, 0], 3.0))
    assert torch.allclose(scales, 3.0 * weights)  # S = 3W
    # 非平凡：不再是被扰动前的常数 1/3
    assert not torch.allclose(weights, torch.full_like(weights, 1 / 3), atol=1e-3)


def test_feature_gate_zero_initialized_last_layer(synthetic_arch):
    for builder in (_feature_image_gate, _feature_anatomy_gate):
        net = builder(synthetic_arch)
        assert torch.count_nonzero(net.gate.conv2.weight) == 0
        assert torch.count_nonzero(net.gate.conv2.bias) == 0


@pytest.mark.parametrize(
    "builder,in_channels", [(_feature_image_gate, 3), (_feature_anatomy_gate, 5)]
)
def test_each_sequence_uses_exactly_one_scale_field(
    synthetic_arch, builder, in_channels
):
    """``S_m`` 在 ``H_m`` 的全部 ``C_s`` 个通道上共享：逐通道比值必须完全相同。"""
    net = builder(synthetic_arch)
    _perturb_feature_gate(net, seed=9, std=0.4)
    x = torch.randn(2, in_channels, 8, 16, 16)
    with torch.no_grad():
        features, _prior = net._stem_features(x)
        scales = net.compute_gate_scales(x)
    assert len(features) == 3
    # 尺度场的通道数是 3（每序列一个），而不是 3*C_s
    assert scales.shape[1] == 3
    assert features[0].shape[1] == FEATURE_STEM_CHANNELS
    for index, feature in enumerate(features):
        weighted = feature * scales[:, index : index + 1]
        ratio = weighted / feature
        assert torch.allclose(ratio, ratio[:, :1].expand_as(ratio), atol=1e-5), (
            f"第 {index} 条序列的特征通道没有共享同一个尺度场"
        )


def test_feature_forward_takes_scales_only_from_gate(synthetic_arch):
    """forward 与手工重建逐值一致：投影(S_m·H_m 的拼接)。"""
    net = _feature_image_gate(synthetic_arch)
    _perturb_feature_gate(net, seed=31)
    net.eval()
    x = torch.randn(2, 3, 8, 16, 16)
    with torch.no_grad():
        features, _ = net._stem_features(x)
        scales = net.compute_gate_scales(x)
        manual = net.projection(
            torch.cat([features[i] * scales[:, i : i + 1] for i in range(3)], dim=1)
        )
        reference = net.backbone(manual)
        out = net(x)
    assert torch.allclose(out, reference, atol=1e-6)
    # 控制：S 不是恒等，且结果不同于未加权路径（证明 S 真的生效）
    assert not torch.allclose(scales, torch.ones_like(scales), atol=1e-3)
    with torch.no_grad():
        unweighted = net.backbone(net.projection(torch.cat(features, dim=1)))
    assert not torch.allclose(out, unweighted)


# ------------------------------------------------- 条件响应
def test_feature_image_gate_responds_to_mri_features(synthetic_arch):
    net = _feature_image_gate(synthetic_arch)
    _perturb_feature_gate(net, seed=13, std=0.5)
    x = torch.randn(2, 3, 8, 16, 16)
    x_changed = x.clone()
    x_changed[:, 1] = x_changed[:, 1] * 2.0 + 0.5  # 只改 ADC
    with torch.no_grad():
        assert not torch.allclose(
            net.compute_gate_scales(x), net.compute_gate_scales(x_changed)
        )


def test_feature_anatomy_gate_responds_to_structured_pz_tz(synthetic_arch):
    """固定 MRI、只改具有空间结构的 PZ/TZ：gate 与输出变化，stem 完全不变。"""
    net = _feature_anatomy_gate(synthetic_arch)
    _perturb_feature_gate(net, seed=17, std=0.5)
    mri = torch.randn(1, 3, 8, 16, 16)
    pattern = torch.linspace(0.0, 1.0, 8 * 16 * 16).reshape(8, 16, 16)
    z_a = torch.stack([pattern, 1.0 - pattern])[None]
    z_b = torch.stack([1.0 - pattern, pattern])[None]  # 交换 PZ/TZ，仍是合法 [0,1]
    x_a = torch.cat([mri, z_a], dim=1)
    x_b = torch.cat([mri, z_b], dim=1)

    with torch.no_grad():
        feats_a, prior_a = net._stem_features(x_a)
        feats_b, _ = net._stem_features(x_b)
        # 1) PZ/TZ 不进入任何 stem
        assert all(torch.equal(a, b) for a, b in zip(feats_a, feats_b))
        # 2) 只影响 gate
        scales_a = net.compute_gate_scales(x_a)
        scales_b = net.compute_gate_scales(x_b)
        assert not torch.allclose(scales_a, scales_b)
        # 3) 并传播到输出
        net.eval()
        assert not torch.allclose(net(x_a), net(x_b))
    # prior 通道确实被读进 gate 输入（唯一入口）
    gate_input = net._gate_input(feats_a, prior_a)
    assert gate_input.shape[1] == FEATURE_STEM_CHANNELS * 3 + 2


def test_feature_no_gate_ignores_prior_channels_entirely(synthetic_arch):
    net = _feature_no_gate(synthetic_arch)
    assert net.gate is None
    assert not any(key.startswith("gate.") for key in net.state_dict())
    with pytest.raises(RuntimeError):
        net.compute_gate_scales(torch.randn(1, 3, 8, 16, 16))
    with pytest.raises(RuntimeError):
        net.compute_weights(torch.randn(1, 3, 8, 16, 16))


# ------------------------------------------------- PZ/TZ clamp 与通道严格性
def test_feature_anatomy_prior_is_clamped_to_unit_interval(synthetic_arch):
    """越界 PZ/TZ 与显式 clamp 后的 PZ/TZ 必须给出逐值相同的尺度与输出。"""
    net = _feature_anatomy_gate(synthetic_arch)
    _perturb_feature_gate(net, seed=23)
    mri = torch.randn(1, 3, 8, 16, 16)
    pattern = torch.linspace(-5.0, 7.0, 8 * 16 * 16).reshape(8, 16, 16)
    x_oob = torch.cat([mri, torch.stack([pattern, -pattern])[None]], dim=1)
    x_valid = torch.cat(
        [
            mri,
            torch.stack([pattern.clamp(0.0, 1.0), (-pattern).clamp(0.0, 1.0)])[None],
        ],
        dim=1,
    )
    with torch.no_grad():
        split_mri, prior_oob = net._split_input(x_oob)
        _, prior_valid = net._split_input(x_valid)
        assert torch.equal(prior_oob, prior_valid)
        assert float(prior_oob.min()) >= 0.0 and float(prior_oob.max()) <= 1.0
        # MRI 通道不被 clamp 改动
        assert torch.equal(split_mri, mri)
        feats_oob, _ = net._stem_features(x_oob)
        feats_valid, _ = net._stem_features(x_valid)
        assert torch.equal(
            net._gate_input(feats_oob, prior_oob),
            net._gate_input(feats_valid, prior_valid),
        )
        assert torch.equal(
            net.compute_gate_scales(x_oob), net.compute_gate_scales(x_valid)
        )
        net.eval()
        assert torch.equal(net(x_oob), net(x_valid))


@pytest.mark.parametrize("bad_channels", [1, 2, 4, 6])
def test_feature_image_rejects_non_three_channels(synthetic_arch, bad_channels):
    net = _feature_image_gate(synthetic_arch)
    with pytest.raises(ValueError):
        net(torch.randn(1, bad_channels, 8, 16, 16))


@pytest.mark.parametrize("bad_channels", [3, 4, 6])
def test_feature_anatomy_strictly_requires_five_channels(synthetic_arch, bad_channels):
    net = _feature_anatomy_gate(synthetic_arch)
    with pytest.raises(ValueError):
        net(torch.randn(1, bad_channels, 8, 16, 16))


def test_feature_builder_rejects_wrong_dataset_channel_count(synthetic_arch):
    """build_network_architecture 必须在通道数不符时 fail-closed。"""
    for trainer, wrong_channels in (
        (nnUNetTrainerPICAI_FeatureNoGate_PositiveSampling_NoFFT, 5),
        (nnUNetTrainerPICAI_FeatureImageGate_PositiveSampling_NoFFT, 5),
        (nnUNetTrainerPICAI_FeatureAnatomyGate_PositiveSampling_NoFFT, 3),
    ):
        with pytest.raises(ValueError):
            _build_feature(synthetic_arch, trainer, wrong_channels)


# ------------------------------------------------- deep supervision / state_dict
def test_feature_preserves_deep_supervision_output_format(synthetic_arch):
    net = _feature_image_gate(synthetic_arch, deep_supervision=True)
    net.eval()
    with torch.no_grad():
        out = net(torch.randn(2, 3, 8, 16, 16))
    assert isinstance(out, list)
    assert len(out) == 2
    assert all(isinstance(item, torch.Tensor) for item in out)


def test_feature_decoder_proxy_toggles_deep_supervision(synthetic_arch):
    net = _feature_image_gate(synthetic_arch, deep_supervision=True)
    assert net.decoder is net.backbone.decoder
    net.eval()
    x = torch.randn(1, 3, 8, 16, 16)
    net.decoder.deep_supervision = True
    with torch.no_grad():
        assert isinstance(net(x), list)
    net.decoder.deep_supervision = False
    with torch.no_grad():
        assert isinstance(net(x), torch.Tensor)


def test_feature_strict_state_dict_roundtrip(synthetic_arch, tmp_path):
    net = _feature_image_gate(synthetic_arch, deep_supervision=True)
    ckpt = tmp_path / "feature_ckpt.pth"
    torch.save({"network_weights": net.state_dict()}, ckpt)

    net2 = _feature_image_gate(synthetic_arch, deep_supervision=True)
    net2.load_state_dict(
        torch.load(ckpt, weights_only=False)["network_weights"], strict=True
    )
    net.eval()
    net2.eval()
    x = torch.randn(1, 3, 8, 16, 16)
    with torch.no_grad():
        for a, b in zip(net(x), net2(x)):
            assert torch.allclose(a, b, atol=1e-6)


def test_feature_variants_are_not_state_dict_compatible(synthetic_arch):
    """image ↔ anatomy 的 gate.conv1.weight 形状不同，strict 加载必须失败。"""
    image = _feature_image_gate(synthetic_arch)
    anatomy = _feature_anatomy_gate(synthetic_arch)
    no_gate = _feature_no_gate(synthetic_arch)
    with pytest.raises(RuntimeError):
        anatomy.load_state_dict(image.state_dict(), strict=True)
    with pytest.raises(RuntimeError):
        no_gate.load_state_dict(image.state_dict(), strict=True)
    # 输入级 gate 网络与本族也不互认（避免从旧 checkpoint 误加载）
    with pytest.raises(RuntimeError):
        image.load_state_dict(_build_image(synthetic_arch).state_dict(), strict=True)


# ------------------------------------------------- 参数量差异被显式记录
def test_feature_gate_parameter_delta_is_recorded(synthetic_arch):
    image = _feature_image_gate(synthetic_arch)
    anatomy = _feature_anatomy_gate(synthetic_arch)
    image_gate_params = sum(p.numel() for p in image.gate.parameters())
    anatomy_gate_params = sum(p.numel() for p in anatomy.gate.parameters())
    # 差值是 gate 首层多出的输入通道：hidden * 2
    assert anatomy_gate_params - image_gate_params == feature_gate_parameter_delta()
    assert feature_gate_parameter_delta() == 16
    # 差额确实来自 conv1.weight 的输入维
    assert (
        anatomy.gate.conv1.weight.shape[0] == image.gate.conv1.weight.shape[0]
    )  # 输出仍是 hidden
    assert anatomy.gate.conv1.weight.shape[1] == image.gate.conv1.weight.shape[1] + 2
    # 其余模块形状完全一致
    assert [tuple(p.shape) for p in image.stems.parameters()] == [
        tuple(p.shape) for p in anatomy.stems.parameters()
    ]
    assert image.projection.weight.shape == anatomy.projection.weight.shape
    assert [tuple(p.shape) for p in image.backbone.parameters()] == [
        tuple(p.shape) for p in anatomy.backbone.parameters()
    ]


def test_feature_module_does_not_copy_unet_and_proxies_backbone(synthetic_arch):
    net = _feature_image_gate(synthetic_arch)
    assert isinstance(net, FeatureFusionNNUNet)
    assert type(net.backbone).__name__ == "PlainConvUNet"
    assert hasattr(net.backbone, "encoder") and hasattr(net.backbone, "decoder")
    assert net.compute_conv_feature_map_size([8, 16, 16]) == (
        net.backbone.compute_conv_feature_map_size([8, 16, 16])
    )
    # 不是输入级 gate 包装器
    assert not isinstance(net, GatedNNUNet)
