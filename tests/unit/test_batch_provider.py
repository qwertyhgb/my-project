"""TorchBatchProvider / PatchDataset 测试（假的内存 store，不读取真实数据）。"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from zonal_reliability_fusion.data import (
    BatchProviderError,
    CaseProperties,
    PatchSampler,
    PreprocessedCase,
    TorchBatchProvider,
    build_train_transforms,
    build_validation_transforms,
)

PATCH = (8, 16, 16)
SHAPE = (12, 24, 24)


class FakeStore:
    """最小假 store：只需 available_ids 与 get_case（鸭子类型）。"""

    def __init__(self, n_cases: int = 3, *, with_fg: bool = True):
        self.cases: dict[str, PreprocessedCase] = {}
        rng = np.random.default_rng(0)
        for i in range(n_cases):
            data = rng.standard_normal((3, *SHAPE)).astype(np.float32)
            seg = np.zeros((1, *SHAPE), dtype=np.uint8)
            if with_fg:
                seg[0, 4:6, 8:12, 8:12] = 1
            self.cases[f"case_{i}"] = PreprocessedCase(
                case_id=f"case_{i}",
                data=data,
                seg=seg,
                properties=CaseProperties(spacing=(3.0, 0.5, 0.5), shape_before_cropping=None, class_locations=None),
            )
        self.available_ids = tuple(sorted(self.cases))

    def get_case(self, case_id: str) -> PreprocessedCase:
        return self.cases[case_id]


def _provider(store, *, force_foreground, transform, num_batches=2, batch_size=2, seed=0, num_workers=0, progress=False):
    sampler = PatchSampler(PATCH, oversample_foreground=0.33, seed=seed)
    return TorchBatchProvider(
        store,
        sampler,
        batch_size=batch_size,
        num_batches=num_batches,
        case_ids=store.available_ids,
        transform=transform,
        seed=seed,
        force_foreground=force_foreground,
        num_workers=num_workers,
        progress=progress,
    )


def test_batch_shapes_and_labels():
    store = FakeStore(with_fg=True)
    provider = _provider(store, force_foreground=None, transform=build_train_transforms())
    batches = list(provider)
    assert len(batches) == 2
    for data, seg in batches:
        assert tuple(data.shape) == (2, 3, *PATCH)
        assert tuple(seg.shape) == (2, 1, *PATCH)
        assert data.dtype == torch.float32
        assert torch.isfinite(data).all()
        assert set(torch.unique(seg).tolist()) <= {0.0, 1.0}
    assert provider.stats["batches"] == 2
    assert provider.stats["samples"] == 4


def test_validation_provider_never_requests_foreground():
    store = FakeStore(with_fg=True)
    provider = _provider(store, force_foreground=False, transform=build_validation_transforms())
    list(provider)
    assert provider.stats["foreground_requested"] == 0
    assert provider.foreground_per_batch() == 0


def test_foreground_per_batch_matches_official_semantics():
    """官方：每批固定 bs - round(bs*(1-oversample)) 个样本强制前景。"""
    store = FakeStore(with_fg=True)
    provider = _provider(store, force_foreground=None, transform=build_train_transforms(), batch_size=2)
    assert provider.foreground_per_batch() == 1  # bs=2, oversample=0.33 → 恰好 1/2

    provider4 = _provider(store, force_foreground=None, transform=build_train_transforms(), batch_size=4)
    assert provider4.foreground_per_batch() == 4 - round(4 * (1 - 0.33))  # = 2

    provider_all = _provider(store, force_foreground=None, transform=build_train_transforms())
    provider_all.sampler.oversample_foreground = 1.0
    assert provider_all.foreground_per_batch() == 2  # 全部强制前景

    list(provider)
    assert provider.stats["foreground_requested"] == 2 * provider.foreground_per_batch()
    assert provider.stats["foreground_fallbacks_is_exact"] is True


def test_negative_cases_are_handled_without_error():
    store = FakeStore(with_fg=False)
    sampler = PatchSampler(PATCH, oversample_foreground=1.0, seed=2)
    provider = TorchBatchProvider(
        store, sampler, batch_size=2, num_batches=1, case_ids=store.available_ids, seed=2, force_foreground=None
    )
    batches = list(provider)
    assert len(batches) == 1
    assert provider.stats["foreground_fallbacks"] == 2  # 全部安全回退（单进程统计精确）


def test_deterministic_with_same_seed_and_epoch():
    store = FakeStore()
    a = list(_provider(store, force_foreground=None, transform=build_train_transforms(), seed=7))
    b = list(_provider(store, force_foreground=None, transform=build_train_transforms(), seed=7))
    for (da, sa), (db, sb) in zip(a, b):
        assert torch.equal(da, db)
        assert torch.equal(sa, sb)


def test_epoch_changes_sampling_sequence_and_is_reproducible():
    store = FakeStore()
    provider = _provider(store, force_foreground=None, transform=build_train_transforms(), seed=7, num_batches=1)
    provider.set_epoch(0)
    first = [b[0].clone() for b in provider]
    provider.set_epoch(0)
    again = [b[0].clone() for b in provider]
    provider.set_epoch(1)
    second = [b[0].clone() for b in provider]
    assert torch.equal(first[0], again[0])  # 同 epoch 完全可复现（无隐藏 RNG 状态）
    assert not torch.equal(first[0], second[0])  # 换 epoch 采样不同


def test_multiprocess_loader_yields_same_shapes():
    """num_workers>0 走 torch DataLoader（fork），批次数与形状必须一致。"""
    store = FakeStore()
    provider = _provider(
        store, force_foreground=None, transform=build_train_transforms(), num_workers=2, seed=3
    )
    batches = list(provider)
    assert len(batches) == 2
    assert tuple(batches[0][0].shape) == (2, 3, *PATCH)
    assert provider.stats["foreground_fallbacks_is_exact"] is False  # 多进程下不做不准确统计


def test_empty_case_list_raises():
    store = FakeStore()
    with pytest.raises(BatchProviderError):
        TorchBatchProvider(store, PatchSampler(PATCH), batch_size=1, num_batches=1, case_ids=[])


def test_describe_reports_pipeline():
    store = FakeStore()
    provider = _provider(store, force_foreground=None, transform=build_train_transforms())
    info = provider.describe()
    assert info["patch_size"] == list(PATCH)
    assert info["augmentation"]["spatial"] == ["MirrorAugmentor"]
    assert info["force_foreground"] is None
    assert info["foreground_per_batch"] == 1
    assert "deterministic" in info["sampling"]
    assert info["num_workers"] == 0 and info["pin_memory"] is False
