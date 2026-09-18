"""滑窗推理测试（合成张量与逐点模型，强制 CPU）。"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from zonal_reliability_fusion.inference import (
    compute_gaussian_importance_map,
    compute_sliding_window_steps,
    logits_to_probabilities,
    sliding_window_predict,
)

PATCH = (8, 16, 16)


class IdentityLogits(torch.nn.Module):
    """逐点模型：class0 = 输入第一通道，class1 = -输入第一通道。"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x[:, :1], -x[:, :1]], dim=1)


class OnesLogits(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = (x.shape[0], 2, *x.shape[-3:])
        return torch.ones(shape)


class DeepSupervisionOnes(torch.nn.Module):
    """返回 DS 列表（主输出为 1，低分辨率为 3），用于验证只使用主输出。"""

    def forward(self, x: torch.Tensor):
        main = torch.ones((x.shape[0], 2, *x.shape[-3:]))
        low = torch.full((x.shape[0], 2, *(s // 2 for s in x.shape[-3:])), 3.0)
        return [main, low]


def test_gaussian_map_center_above_edges_and_positive():
    weight = compute_gaussian_importance_map(PATCH)
    assert weight.shape == PATCH
    assert float(weight.min()) > 0.0
    assert float(weight.max()) == pytest.approx(1.0)
    center = weight[PATCH[0] // 2, PATCH[1] // 2, PATCH[2] // 2]
    assert float(center) > float(weight[0, 0, 0])
    assert float(center) > float(weight[-1, -1, -1])


def test_sliding_window_steps_cover_last_position():
    steps = compute_sliding_window_steps((10, 17, 17), PATCH, 0.5)
    assert steps[0][-1] == 10 - PATCH[0]
    assert steps[1][-1] == 17 - PATCH[1]
    assert steps[2][-1] == 17 - PATCH[2]
    for dim_steps in steps:
        assert dim_steps[0] == 0
        assert dim_steps == sorted(dim_steps)


def test_sliding_window_steps_argument_checks():
    with pytest.raises(ValueError):
        compute_sliding_window_steps((4, 4, 4), PATCH, 0.5)  # image < patch
    with pytest.raises(ValueError):
        compute_sliding_window_steps((10, 17, 17), PATCH, 0.0)
    with pytest.raises(ValueError):
        compute_sliding_window_steps((10, 17, 17), PATCH, 1.5)


def test_identity_model_restores_values_exactly():
    image = torch.randn(1, 3, 10, 17, 17)
    out = sliding_window_predict(IdentityLogits(), image, patch_size=PATCH, step_fraction=0.5, progress=False)
    # 逐点模型：输出 2 类 logits，空间尺寸与输入一致
    assert tuple(out.shape) == (1, 2, 10, 17, 17)
    assert torch.allclose(out[:, 0], image[:, 0], atol=1e-5)
    assert torch.allclose(out[:, 1], -image[:, 0], atol=1e-5)


def test_small_image_padded_then_cropped():
    image = torch.randn(1, 3, 5, 5, 5)
    out = sliding_window_predict(IdentityLogits(), image, patch_size=PATCH, progress=False)
    assert tuple(out.shape) == (1, 2, 5, 5, 5)
    assert torch.allclose(out[:, 0], image[:, 0], atol=1e-5)


def test_no_uncovered_voxels_with_ones_model():
    image = torch.randn(1, 3, 9, 17, 20)
    out = sliding_window_predict(OnesLogits(), image, patch_size=PATCH, step_fraction=0.5, progress=False)
    assert tuple(out.shape) == (1, 2, 9, 17, 20)
    assert torch.allclose(out, torch.ones_like(out))


def test_deep_supervision_model_uses_main_output_only():
    image = torch.randn(1, 3, 9, 17, 17)
    ds_out = sliding_window_predict(DeepSupervisionOnes(), image, patch_size=PATCH, progress=False)
    assert torch.allclose(ds_out, torch.ones_like(ds_out))  # 主输出为 1，未混入低分辨率 3


def test_batching_matches_single_patch_forward():
    image = torch.randn(1, 3, 10, 17, 17)
    one = sliding_window_predict(IdentityLogits(), image, patch_size=PATCH, batch_size=1, progress=False)
    many = sliding_window_predict(IdentityLogits(), image, patch_size=PATCH, batch_size=4, progress=False)
    assert torch.allclose(one, many, atol=1e-6)


def test_mirror_tta_interface_is_off_by_default_and_runs():
    image = torch.randn(1, 3, 9, 17, 17)
    plain = sliding_window_predict(IdentityLogits(), image, patch_size=PATCH, progress=False)
    tta = sliding_window_predict(IdentityLogits(), image, patch_size=PATCH, mirror_tta=True, progress=False)
    assert torch.allclose(plain, tta, atol=1e-5)  # 逐点模型的镜像平均等价


def test_uniform_weight_option_and_probabilities():
    image = torch.randn(1, 3, 9, 17, 17)
    out = sliding_window_predict(OnesLogits(), image, patch_size=PATCH, gaussian=False, progress=False)
    assert torch.allclose(out, torch.ones_like(out))
    probs = logits_to_probabilities(torch.randn(1, 2, 4, 4, 4))
    assert torch.allclose(probs.sum(dim=1), torch.ones_like(probs[:, 0]))


def test_progress_flag_is_wired(capsys):
    """progress=True 必须真正走 tqdm（耗时循环有进度条）；progress=False 时无输出。"""
    image = torch.randn(1, 3, 9, 17, 17)
    silent = sliding_window_predict(OnesLogits(), image, patch_size=PATCH, progress=False)
    out = capsys.readouterr()
    assert out.out == "" and out.err == ""

    shown = sliding_window_predict(OnesLogits(), image, patch_size=PATCH, progress=True)
    captured = capsys.readouterr()
    assert "sliding-window" in (captured.err + captured.out)  # tqdm 默认写 stderr
    assert torch.allclose(silent, shown)


def test_invalid_arguments_raise():
    with pytest.raises(ValueError):
        sliding_window_predict(OnesLogits(), torch.randn(1, 3, 4, 4, 4), patch_size=(0, 4, 4), progress=False)
    with pytest.raises(ValueError):
        sliding_window_predict(OnesLogits(), torch.randn(1, 3, 4, 4, 4), patch_size=(4, 4, 4), batch_size=0)
