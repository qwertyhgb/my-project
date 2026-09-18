"""nnU-Net seg 填充哨兵（-1）兼容性的合成回归测试（纯合成数组；不读取真实病例）。

上游语义（静态核对，见 third_party/nnUNet）：
- `nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py:800`（训练）与 `:855`（验证）都执行
  `RemoveLabelTansform(-1, 0)`；该 transform 的实现是**纯值映射**：
  `segmentation[segmentation == label_value] = set_to`（不裁剪、不截断）。
- Dataset605 `dataset.json` 只定义 background=0 / lesion=1，无语义 ignore 类。
因此本项目的读取适配层必须把存储态 -1 映射为 0，且对 {-1,0,1} 以外的任何值 fail-closed。
"""
from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest
import torch

from zonal_reliability_fusion.data import (
    PatchSampler,
    PreprocessedStore,
    PreprocessedStoreError,
    TorchBatchProvider,
    build_train_transforms,
    normalize_seg,
    verify_case_contract,
)
from zonal_reliability_fusion.data.preprocessed_store import (
    SEG_SENTINEL_LABEL,
    SEG_STORAGE_LABELS,
)

SHAPE = (6, 8, 8)
PATCH = (4, 4, 4)


# --------------------------------------------------------------------------- A–F 转换契约
def test_a_negative_case_all_sentinel_becomes_all_zero_uint8():
    raw = np.full(SHAPE, SEG_SENTINEL_LABEL, dtype=np.int16)
    raw[0, 0, 0] = 0
    out = normalize_seg(raw, "case_neg")
    assert out.shape == (1, *SHAPE)
    assert out.dtype == np.uint8
    assert np.count_nonzero(out) == 0
    assert set(np.unique(out).tolist()) == {0}


def test_b_positive_case_foreground_count_unchanged():
    raw = np.full(SHAPE, SEG_SENTINEL_LABEL, dtype=np.int8)
    raw[2:4, 2:4, 2:4] = 1
    raw[0, 0, 0] = 0
    out = normalize_seg(raw, "case_pos")
    assert np.array_equal(out, (raw == 1).astype(np.uint8)[None])
    assert int(out.sum()) == int(np.count_nonzero(raw == 1)) == 8
    assert set(np.unique(out).tolist()) == {0, 1}


def test_c_input_array_is_not_modified_in_place():
    raw = np.full(SHAPE, SEG_SENTINEL_LABEL, dtype=np.int16)
    raw[1, 1, 1] = 1
    before = raw.copy()
    out = normalize_seg(raw, "case_x")
    assert raw.dtype == np.int16 and np.array_equal(raw, before), "输入不得被原地修改"
    assert not np.shares_memory(out, raw), "必须返回新数组"
    out[...] = 0
    assert np.array_equal(raw, before), "写返回值不得影响调用者输入"


def test_d_returned_array_is_c_contiguous():
    raw = np.asfortranarray(np.full(SHAPE, SEG_SENTINEL_LABEL, dtype=np.int16))
    raw[0, 1, 2] = 1
    assert not raw.flags["C_CONTIGUOUS"]
    out = normalize_seg(raw, "case_f")
    assert out.flags["C_CONTIGUOUS"] and out.dtype == np.uint8


@pytest.mark.parametrize(
    "dtype", [np.uint8, np.int8, np.int16, np.int32, np.int64, np.bool_, np.float32, np.float64]
)
def test_e_pure_binary_input_has_no_regression(dtype):
    ref = np.zeros(SHAPE, dtype=np.uint8)
    ref[1:3, 1:3, 1:3] = 1
    out = normalize_seg(ref.astype(dtype), "case_bin")
    assert out.dtype == np.uint8 and out.shape == (1, *SHAPE)
    assert np.array_equal(out[0], ref)


def test_f_shape_adaptation_3d_and_4d_singleton():
    assert normalize_seg(np.zeros(SHAPE, np.uint8)).shape == (1, *SHAPE)
    assert normalize_seg(np.zeros((1, *SHAPE), np.uint8)).shape == (1, *SHAPE)
    with pytest.raises(PreprocessedStoreError, match="形状"):
        normalize_seg(np.zeros((2, *SHAPE), np.uint8), "case_two_channels")
    with pytest.raises(PreprocessedStoreError, match="形状"):
        normalize_seg(np.zeros((1, 1, *SHAPE), np.uint8), "case_five_dim")


# --------------------------------------------------------------------------- G 拒绝路径
REJECT_CASES = [
    ("minus2", np.full(SHAPE, -2, np.int16), "取值"),
    ("plus2", np.full(SHAPE, 2, np.int16), "取值"),
    ("uint255", np.full(SHAPE, 255, np.uint8), "取值"),
    ("half", np.full(SHAPE, 0.5, np.float32), "非整数"),
    ("one_and_half", np.full(SHAPE, 1.5, np.float32), "非整数"),
    ("nan", np.full(SHAPE, np.nan, np.float32), "非有限"),
    ("pos_inf", np.full(SHAPE, np.inf, np.float32), "非有限"),
    ("neg_inf", np.full(SHAPE, -np.inf, np.float32), "非有限"),
    ("complex", np.full(SHAPE, 1 + 0j, np.complex64), "不受支持"),
    ("string", np.full(SHAPE, "1"), "不受支持"),
    ("object", np.full(SHAPE, {"label": 1}, dtype=object), "不受支持"),
    ("datetime", np.zeros(SHAPE, dtype="datetime64[s]"), "不受支持"),
]


@pytest.mark.parametrize("name, raw, match", REJECT_CASES, ids=[c[0] for c in REJECT_CASES])
def test_g_illegal_labels_and_dtypes_are_rejected(name, raw, match):
    with pytest.raises(PreprocessedStoreError, match=match) as excinfo:
        normalize_seg(raw, f"case_{name}")
    assert not isinstance(excinfo.value, TypeError), "必须抛 PreprocessedStoreError，不得泄漏 TypeError"


def test_g_two_and_minus_two_are_not_silently_mapped():
    """-2 / 2 绝不能被当成背景或病灶（禁止 clip / 截断 / astype(uint8)）。"""
    for value in (-2, 2):
        raw = np.zeros(SHAPE, np.int16)
        raw[0, 0, 0] = value
        with pytest.raises(PreprocessedStoreError, match="取值"):
            normalize_seg(raw, f"case_{value}")


def test_g_input_with_value_255_never_becomes_foreground():
    """直接 astype(uint8) 会把 -1 变 255；本实现必须先验证（255 属于非法取值）。"""
    raw = np.full(SHAPE, SEG_SENTINEL_LABEL, dtype=np.int16)
    raw[0, 0, 0] = 255
    with pytest.raises(PreprocessedStoreError, match="取值"):
        normalize_seg(raw, "case_255")


# --------------------------------------------------------------------------- H get_case 路径
class _FakeDataset:
    """minimal 假 dataset：只提供 `__getitem__`（get_case 的四元组协议）。"""

    def __init__(self, entries: dict[str, tuple]):
        self._entries = entries

    def __getitem__(self, case_id: str):
        return self._entries[case_id]


def _raw_seg(kind: str) -> np.ndarray:
    if kind == "pos":
        raw = np.full(SHAPE, SEG_SENTINEL_LABEL, dtype=np.int16)
        raw[2:4, 2:4, 2:4] = 1
        return raw
    if kind == "neg":
        raw = np.full(SHAPE, SEG_SENTINEL_LABEL, dtype=np.int16)
        raw[0, 0, 0] = 0
        return raw
    return np.zeros(SHAPE, dtype=np.uint8)


def _entry(kind: str) -> tuple:
    data = np.zeros((3, *SHAPE), dtype=np.float32)
    props = {"spacing": (3.0, 0.5, 0.5), "shape_before_cropping": SHAPE, "class_locations": None}
    return (data, _raw_seg(kind), None, props)


def _store_with(cases: dict[str, tuple]) -> PreprocessedStore:
    store = PreprocessedStore.__new__(PreprocessedStore)
    store._dataset = _FakeDataset(cases)
    store.dataset_folder = Path("/nonexistent-unused")
    store.dataset_identifier = "nnUNetPlans_3d_fullres"
    store.fold = 0
    store.train_ids = tuple(cases)
    store.val_ids = ()
    store.zonal_prior_root = None
    store.zonal_prior_source = None
    store.expected_plans_sha256 = None
    store.expected_configuration = None
    store.zonal_prior_manifest_sha256 = None
    store.zonal_prior_array_hash_checked = False
    store.zonal_prior_validation_stats = {}
    store.zonal_prior_frozen_records = MappingProxyType({})
    store.require_frozen_manifest = False
    return store


def test_h_get_case_accepts_sentinel_and_contract_reports_binary():
    store = _store_with({"case_pos": _entry("pos"), "case_neg": _entry("neg")})
    case = store.get_case("case_pos")
    assert case.seg.dtype == np.uint8 and case.seg.shape == (1, *SHAPE)
    info = verify_case_contract(case)
    assert info["seg_dtype"] == "uint8"
    assert set(info["seg_labels"]) <= {0, 1}
    assert info["labels_binary"] is True
    assert SEG_SENTINEL_LABEL not in info["seg_labels"]

    neg = store.get_case("case_neg")
    info_neg = verify_case_contract(neg)
    assert info_neg["seg_labels"] == [0] and info_neg["labels_binary"] is True

    # 纯 {0,1} 的既有行为不回归
    plain = verify_case_contract(_store_with({"case_plain": _entry("plain")}).get_case("case_plain"))
    assert plain["seg_labels"] == [0] and plain["labels_binary"] is True


# --------------------------------------------------------------------------- I/J sampler & provider
def _provider(store: PreprocessedStore, *, force_foreground=None, num_batches=2, batch_size=2):
    return TorchBatchProvider(
        store,
        PatchSampler(PATCH, oversample_foreground=0.5, seed=0),
        batch_size=batch_size,
        num_batches=num_batches,
        case_ids=store.train_ids,
        transform=build_train_transforms(),
        seed=0,
        force_foreground=force_foreground,
        num_workers=0,
        progress=False,
    )


def test_i_provider_batches_never_contain_sentinel_or_255():
    store = _store_with({"case_pos": _entry("pos"), "case_neg": _entry("neg")})
    provider = _provider(store, force_foreground=True)
    seen_pairs = 0
    for data, seg in provider:
        uniq = set(torch.unique(seg).tolist())
        assert uniq <= {0.0, 1.0}, f"batch seg 出现非法取值 {uniq}"
        assert -1.0 not in uniq and 255.0 not in uniq
        assert torch.isfinite(data).all()
        seen_pairs += 1
    assert seen_pairs == provider.num_batches


def test_j_foreground_and_fallback_semantics_unchanged_by_mapping():
    sampler = PatchSampler(PATCH, oversample_foreground=1.0, seed=0)
    data = np.zeros((3, *SHAPE), dtype=np.float32)

    # 阳性：raw {-1,0,1} 映射后必须与参考 {0,1} seg 行为完全一致（真前景，无回退）
    ref_pos = np.zeros((1, *SHAPE), np.uint8)
    ref_pos[0, 2:4, 2:4, 2:4] = 1
    mapped_pos = normalize_seg(_raw_seg("pos"))
    p1 = sampler.sample(data, mapped_pos, force_fg=True, rng=np.random.default_rng(0))
    p2 = sampler.sample(data, ref_pos, force_fg=True, rng=np.random.default_rng(0))
    assert p1.bbox == p2.bbox
    assert p1.used_foreground is True and p2.used_foreground is True
    assert p1.foreground_fallback is False and p2.foreground_fallback is False
    assert np.array_equal(p1.seg, p2.seg)

    # 阴性：raw 全为 {-1,0}（哨兵填充）时等价于全 0 seg → 安全回退，语义不变
    ref_neg = np.zeros((1, *SHAPE), np.uint8)
    mapped_neg = normalize_seg(_raw_seg("neg"))
    n1 = sampler.sample(data, mapped_neg, force_fg=True, rng=np.random.default_rng(1))
    n2 = sampler.sample(data, ref_neg, force_fg=True, rng=np.random.default_rng(1))
    assert n1.bbox == n2.bbox
    assert n1.used_foreground is False and n2.used_foreground is False
    assert n1.foreground_fallback is True and n2.foreground_fallback is True
    assert np.array_equal(n1.seg, n2.seg)
    assert set(np.unique(n1.seg).tolist()) == {0}, "回退 patch 里不得出现 -1"


def test_j_negative_case_fallback_stats_match_binary_negative_case():
    neg_store = _store_with({"case_neg": _entry("neg")})
    plain_store = _store_with({"case_plain": _entry("plain")})
    neg_provider = _provider(neg_store, force_foreground=True, num_batches=1, batch_size=2)
    plain_provider = _provider(plain_store, force_foreground=True, num_batches=1, batch_size=2)
    list(neg_provider)
    list(plain_provider)
    assert neg_provider.stats["foreground_fallbacks"] == plain_provider.stats["foreground_fallbacks"]
    assert neg_provider.stats["foreground_requested"] == plain_provider.stats["foreground_requested"] == 2


# --------------------------------------------------------------------------- 官方语义等价性
def test_k_equivalent_to_official_remove_label_transform_semantics():
    """等价于 `RemoveLabelTansform(-1, 0)` 之后 Dataset605 {0,1} 二值标签的逐体素语义。"""
    rng = np.random.default_rng(0)
    raw = rng.choice(np.asarray(SEG_STORAGE_LABELS, dtype=np.int16), size=SHAPE)
    # 官方 transform 是纯值映射（不裁剪、不截断）：seg[seg == -1] = 0
    official = raw.copy()
    official[official == SEG_SENTINEL_LABEL] = 0
    out = normalize_seg(raw, "case_random")
    assert out.shape == (1, *SHAPE)  # 读取适配边界统一补通道
    assert np.array_equal(out, official.astype(np.uint8)[None])
    # 映射后不得残留哨兵，也不得出现 255（astype(uint8) 的典型副作用）
    uniq = set(np.unique(out).tolist())
    assert uniq <= {0, 1} and SEG_SENTINEL_LABEL not in uniq and 255 not in uniq
