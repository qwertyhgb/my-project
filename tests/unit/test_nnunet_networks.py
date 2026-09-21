"""gate 网络与原生 nnU-Net backbone 包装器的纯合成 CPU 测试（不读取真实数据）。

覆盖：image_gate 3 通道（#4）、anatomy_gate 严格 5 通道（#5）、零初始化 scales==1（#6）、
零初始化 gated MRI 与原 MRI 逐值一致（#7）、deep supervision 输出格式（#8）、
set_deep_supervision_enabled 切换（#9）、strict state_dict save/load（#10）、
anatomy prior clamp 到 [0,1]（#11）。
"""

from __future__ import annotations

import pytest
import torch

from zonal_reliability_fusion.nnunet.networks import (
    GatedNNUNet,
    SpatialModalityReliabilityGate,
)
from zonal_reliability_fusion.nnunet.trainers import (
    nnUNetTrainerPICAI_AnatomyGate,
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


def test_gate_last_conv_is_zero_initialized():
    gate = SpatialModalityReliabilityGate(input_channels=5)
    assert torch.count_nonzero(gate.conv2.weight) == 0
    assert torch.count_nonzero(gate.conv2.bias) == 0


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
