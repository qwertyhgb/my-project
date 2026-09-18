"""Patch 采样器测试（纯 numpy 合成数据，不读取真实病例）。"""
from __future__ import annotations

import numpy as np
import pytest

from zonal_reliability_fusion.data import PatchSampler

PATCH = (8, 16, 16)


def coord_field(shape: tuple[int, int, int]) -> np.ndarray:
    """每个体素编码其唯一坐标，用于验证影像/标签空间同步。"""
    d, h, w = shape
    return (
        np.arange(d)[:, None, None] * 10000 + np.arange(h)[None, :, None] * 100 + np.arange(w)[None, None, :]
    ).astype(np.int32)


def make_case(shape: tuple[int, int, int], *, fg: tuple[int, int, int] | None = None):
    field = coord_field(shape)
    data = np.stack([field.astype(np.float32)] * 3, axis=0)  # [3, D, H, W]
    seg = np.zeros((1, *shape), dtype=np.uint8)
    if fg is not None:
        seg[(0, *fg)] = 1
    return data, seg


def test_random_crop_shape_and_data_seg_sync():
    """影像与标签必须来自同一个 bbox：让两者都编码坐标场即可逐体素验证。"""
    shape = (12, 20, 20)
    field = coord_field(shape)
    data = np.stack([field.astype(np.float32)] * 3, axis=0)
    seg = field[None].astype(np.int32)  # 仅测试同步性，不要求 {0,1}
    sampler = PatchSampler(PATCH, oversample_foreground=0.0, seed=0)
    sample = sampler.sample(data, seg, case_id="c0", force_fg=False)
    assert sample.data.shape == (3, *PATCH)
    assert sample.seg.shape == (1, *PATCH)
    assert np.array_equal(sample.data[0], sample.seg[0].astype(np.float32))
    d0, d1 = sample.bbox[0]
    h0, h1 = sample.bbox[1]
    w0, w1 = sample.bbox[2]
    assert np.array_equal(sample.data[0], field[d0:d1, h0:h1, w0:w1].astype(np.float32))
    assert sample.used_foreground is False and sample.foreground_fallback is False


def test_foreground_crop_contains_foreground_voxel():
    shape = (12, 20, 20)
    fg = (3, 15, 15)
    data, seg = make_case(shape, fg=fg)
    sampler = PatchSampler(PATCH, oversample_foreground=1.0, seed=1)
    sample = sampler.sample(data, seg, case_id="c1", force_fg=True)
    assert sample.used_foreground is True
    assert sample.foreground_fallback is False
    assert int(sample.seg.sum()) >= 1
    for axis, (lo, hi) in enumerate(sample.bbox):
        assert lo <= fg[axis] < hi  # 指定的前景体素落在 patch 内


def test_negative_case_force_fg_falls_back_safely():
    shape = (12, 20, 20)
    data, seg = make_case(shape, fg=None)
    sampler = PatchSampler(PATCH, oversample_foreground=1.0, seed=2)
    sample = sampler.sample(data, seg, case_id="neg", force_fg=True)
    assert sample.foreground_fallback is True
    assert sample.used_foreground is False
    assert sample.data.shape == (3, *PATCH)  # 不报错、不放前景


def test_smaller_than_patch_is_padded():
    shape = (4, 5, 6)
    data, seg = make_case(shape, fg=(1, 2, 3))
    sampler = PatchSampler(PATCH, oversample_foreground=0.0, seed=3)
    sample = sampler.sample(data, seg, case_id="small", force_fg=False)
    assert sample.data.shape == (3, *PATCH)
    assert sample.seg.shape == (1, *PATCH)
    # padding 值显式：data=0.0、seg=0，且 padding 在坐标高端
    assert sample.data[0, 4:, :, :].max() == 0.0
    assert sample.data[0, :, 5:, :].max() == 0.0
    assert sample.seg[0, 4:, :, :].max() == 0
    # 原始区域必须被完整保留
    assert np.array_equal(sample.data[0, :4, :5, :6], coord_field(shape).astype(np.float32))


def test_validation_never_uses_foreground_oversampling():
    shape = (12, 20, 20)
    data, seg = make_case(shape, fg=(5, 5, 5))
    sampler = PatchSampler(PATCH, oversample_foreground=0.99, seed=4)
    for _ in range(5):
        sample = sampler.sample(data, seg, force_fg=False)
        assert sample.used_foreground is False


def test_should_force_foreground_probabilities():
    always = PatchSampler(PATCH, oversample_foreground=1.0, seed=5)
    never = PatchSampler(PATCH, oversample_foreground=0.0, seed=6)
    assert all(always.should_force_foreground() for _ in range(10))
    assert not any(never.should_force_foreground() for _ in range(10))


def test_class_locations_are_used_when_present():
    shape = (12, 20, 20)
    data, seg = make_case(shape, fg=None)  # seg 里没有前景
    voxel = (6, 9, 10)
    properties = {"class_locations": {1: np.array([list(voxel)], dtype=np.int64)}}
    sampler = PatchSampler(PATCH, oversample_foreground=1.0, seed=7)
    sample = sampler.sample(data, seg, case_id="cl", properties=properties, force_fg=True)
    assert sample.used_foreground is True
    assert sample.foreground_fallback is False
    for axis, (lo, hi) in enumerate(sample.bbox):
        assert lo <= voxel[axis] < hi


def test_class_locations_official_nx4_format():
    """官方 preprocessor 对 [1,D,H,W] 做 argwhere，保存的是 (n,4)：第 0 列是标签，后 3 列是坐标。"""
    shape = (12, 20, 20)
    data, seg = make_case(shape, fg=None)
    voxel = (6, 9, 10)
    official = np.array([[1, *voxel]], dtype=np.int64)  # (1,4)：首列为标签索引
    assert official.shape == (1, 4)
    properties = {"class_locations": {1: official}}
    sampler = PatchSampler(PATCH, oversample_foreground=1.0, seed=11)
    sample = sampler.sample(data, seg, case_id="cl4", properties=properties, force_fg=True)
    assert sample.used_foreground is True
    for axis, (lo, hi) in enumerate(sample.bbox):
        assert lo <= voxel[axis] < hi
    # 规范化函数同时接受 (n,3) 与 (n,4)
    assert PatchSampler.normalize_class_locations(official).tolist() == [list(voxel)]
    assert PatchSampler.normalize_class_locations(np.array([list(voxel)])).tolist() == [list(voxel)]
    with pytest.raises(ValueError):
        PatchSampler.normalize_class_locations(np.array([[1, 2]]))


def test_foreground_patch_is_centered_on_voxel():
    """与官方一致：bbox 起点 = clamp(voxel - patch//2, 0, shape-patch)。"""
    shape = (20, 30, 30)
    voxel = (10, 15, 15)
    data, seg = make_case(shape, fg=voxel)
    sampler = PatchSampler(PATCH, oversample_foreground=1.0, seed=12)
    sample = sampler.sample(data, seg, force_fg=True)
    expected = [voxel[d] - PATCH[d] // 2 for d in range(3)]
    assert [lo for lo, _ in sample.bbox] == expected


def test_invalid_inputs_raise():
    sampler = PatchSampler(PATCH)
    data = np.zeros((3, 8, 16, 16), dtype=np.float32)
    seg = np.zeros((1, 8, 16, 16), dtype=np.uint8)
    with pytest.raises(ValueError):
        sampler.sample(data[0], seg)  # data 维度错误
    with pytest.raises(ValueError):
        sampler.sample(data, np.zeros((1, 4, 16, 16), dtype=np.uint8))  # shape 不一致
    with pytest.raises(ValueError):
        PatchSampler((0, 16, 16))
    with pytest.raises(ValueError):
        PatchSampler(PATCH, oversample_foreground=1.5)


def test_output_shapes_match_patch_strictly():
    sampler = PatchSampler(PATCH, oversample_foreground=0.5, seed=8)
    data, seg = make_case((30, 19, 17), fg=(20, 4, 4))
    for force in (None, True, False):
        sample = sampler.sample(data, seg, force_fg=force)
        assert sample.data.shape[-3:] == PATCH
        assert sample.seg.shape[-3:] == PATCH
