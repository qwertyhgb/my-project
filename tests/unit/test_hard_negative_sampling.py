"""解剖约束困难负样本挖掘与采样的纯合成测试。

覆盖 §33 要求的 **hard negative sampler** 与 **split leakage guard**：

- 候选定义（高置信假阳 AND GT 背景 AND 在预测 WG 内）与确定性排序；
- 挖掘范围守卫：validation 病例绝不能出现在挖掘结果中（fail closed）；
- 采样器槽位分配：困难负样本槽位使用存储位置裁剪，且**不**被标记为 ``force_fg``；
- 开 / 关：``hard_negative_set_path is None`` 时完全退化为普通阳性采样。

全部使用合成数组与合成 dataset，不读取真实医学数据、不实例化真实 Trainer。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from zonal_reliability_fusion.sampling.hard_negative import (
    HARD_NEGATIVE_SCHEMA_VERSION,
    HardNegativeMiningError,
    build_hard_negative_set,
    load_hard_negative_set,
    select_hard_negative_locations,
    validate_mining_scope,
    write_hard_negative_set,
)

SHAPE = (8, 12, 12)


def _probability_with_blobs(blobs):
    array = np.zeros(SHAPE, dtype=np.float64)
    for (z, y, x), value in blobs.items():
        array[z, y, x] = value
    return array


# --------------------------------------------------------------------------- candidate rule
def test_selects_only_high_confidence_background_inside_wg():
    """三个条件必须**同时**满足：高置信、GT 背景、在预测 WG 内。"""
    probability = _probability_with_blobs(
        {
            (1, 1, 1): 0.9,  # 高置信 + GT 背景 + 腺体内 -> 选中
            (2, 2, 2): 0.9,  # 高置信 + GT 病灶 -> 不是假阳，排除
            (3, 3, 3): 0.9,  # 高置信 + GT 背景 + 腺体外 -> 排除
            (4, 4, 4): 0.2,  # 低置信 -> 排除
        }
    )
    reference = np.zeros(SHAPE, dtype=bool)
    reference[2, 2, 2] = True
    wg = np.zeros(SHAPE, dtype=np.float64)
    wg[0:3, 0:3, 0:3] = 1.0  # 只覆盖前 3x3x3

    locations = select_hard_negative_locations(
        probability,
        reference,
        wg,
        confidence_threshold=0.5,
        wg_threshold=0.5,
        max_locations_per_case=10,
    )
    assert locations == [[1, 1, 1]]


def test_selection_order_is_deterministic_by_component_size_then_coordinates():
    probability = np.zeros(SHAPE, dtype=np.float64)
    probability[1, 1, 1] = 0.9  # 单体素团块
    probability[5, 5, 5] = probability[5, 5, 6] = 0.9  # 两体素团块（应排在前）
    reference = np.zeros(SHAPE, dtype=bool)
    wg = np.ones(SHAPE, dtype=np.float64)

    first = select_hard_negative_locations(
        probability, reference, wg,
        confidence_threshold=0.5, wg_threshold=0.5, max_locations_per_case=10,
    )
    second = select_hard_negative_locations(
        probability.copy(), reference.copy(), wg.copy(),
        confidence_threshold=0.5, wg_threshold=0.5, max_locations_per_case=10,
    )
    assert first == second
    assert first[0] == [5, 5, 5] or first[0] == [5, 5, 6]
    assert len(first) == 2


def test_max_locations_truncates_by_volume_descending():
    probability = np.zeros(SHAPE, dtype=np.float64)
    probability[1, 1, 1] = 0.9
    probability[5, 5, 5] = probability[5, 5, 6] = probability[5, 5, 7] = 0.9
    locations = select_hard_negative_locations(
        probability, np.zeros(SHAPE, dtype=bool), np.ones(SHAPE),
        confidence_threshold=0.5, wg_threshold=0.5, max_locations_per_case=1,
    )
    assert len(locations) == 1
    assert locations[0] == [5, 5, 6]  # 3 体素团块的质心


def test_no_candidate_returns_empty_and_is_not_an_error():
    empty = select_hard_negative_locations(
        np.zeros(SHAPE), np.zeros(SHAPE, dtype=bool), np.ones(SHAPE),
        confidence_threshold=0.5, wg_threshold=0.5, max_locations_per_case=5,
    )
    assert empty == []


def test_selection_is_fail_closed_on_shape_and_thresholds():
    probability = np.zeros(SHAPE, dtype=np.float64)
    with pytest.raises(HardNegativeMiningError, match="形状必须一致"):
        select_hard_negative_locations(
            probability, np.zeros((4, 4, 4), dtype=bool), np.ones(SHAPE),
            confidence_threshold=0.5, wg_threshold=0.5, max_locations_per_case=5,
        )
    with pytest.raises(HardNegativeMiningError, match="confidence_threshold"):
        select_hard_negative_locations(
            probability, np.zeros(SHAPE, dtype=bool), np.ones(SHAPE),
            confidence_threshold=1.5, wg_threshold=0.5, max_locations_per_case=5,
        )
    bad = np.zeros(SHAPE, dtype=np.float64)
    bad[0, 0, 0] = np.inf
    with pytest.raises(HardNegativeMiningError, match="非有限"):
        select_hard_negative_locations(
            bad, np.zeros(SHAPE, dtype=bool), np.ones(SHAPE),
            confidence_threshold=0.5, wg_threshold=0.5, max_locations_per_case=5,
        )


# --------------------------------------------------------------------------- scope guard
def test_mining_scope_guard_rejects_validation_cases():
    payload = {"cases": {"train_1": {"locations": [[1, 1, 1]]}, "val_1": {"locations": []}}}
    with pytest.raises(HardNegativeMiningError, match="验证病例"):
        validate_mining_scope(payload, train_cases=["train_1"], val_cases=["val_1"])


def test_mining_scope_guard_rejects_cases_outside_train_split():
    payload = {"cases": {"other_1": {"locations": [[1, 1, 1]]}}}
    with pytest.raises(HardNegativeMiningError, match="不在训练 split"):
        validate_mining_scope(payload, train_cases=["train_1"], val_cases=["val_1"])


def test_mining_scope_guard_rejects_illegal_split_and_empty_train():
    payload = {"cases": {"a": {"locations": []}}}
    with pytest.raises(HardNegativeMiningError, match="相交"):
        validate_mining_scope(payload, train_cases=["a"], val_cases=["a"])
    with pytest.raises(HardNegativeMiningError, match="train_cases 为空"):
        validate_mining_scope(payload, train_cases=[], val_cases=["b"])


def test_mining_scope_guard_accepts_train_only_payload():
    payload = {"cases": {"train_1": {"locations": [[1, 1, 1]]}}}
    validate_mining_scope(payload, train_cases=["train_1", "train_2"], val_cases=["val_1"])


# --------------------------------------------------------------------------- build/IO
def _synthetic_arrays(case_id):
    probability = np.zeros(SHAPE, dtype=np.float64)
    probability[2, 2, 2] = 0.9
    probability[6, 6, 6] = 0.95
    reference = np.zeros(SHAPE, dtype=bool)
    reference[6, 6, 6] = True  # 第二个点其实是病灶 -> 不应被选
    wg = np.ones(SHAPE, dtype=np.float64)
    return probability, reference, wg


def test_build_hard_negative_set_records_provenance_and_summary():
    payload = build_hard_negative_set(
        ["case_a", "case_b"],
        _synthetic_arrays,
        dataset_name="Dataset605_PICAI",
        fold=0,
        train_cases=["case_a", "case_b"],
        val_cases=["val_1"],
        provenance={"prior_source": "predicted_prior"},
        confidence_threshold=0.5,
        wg_threshold=0.5,
        max_locations_per_case=5,
        progress=False,
    )
    assert payload["schema_version"] == HARD_NEGATIVE_SCHEMA_VERSION
    assert payload["cases"]["case_a"]["locations"] == [[2, 2, 2]]
    assert payload["summary"]["total_locations"] == 2
    assert payload["summary"]["cases_with_hard_negatives"] == 2
    assert payload["definition"]["location"].startswith("候选假阳连通域")


def test_build_hard_negative_set_refuses_val_cases_in_input():
    with pytest.raises(HardNegativeMiningError, match="验证病例"):
        build_hard_negative_set(
            ["case_a", "val_1"],
            _synthetic_arrays,
            dataset_name="Dataset605_PICAI",
            fold=0,
            train_cases=["case_a"],
            val_cases=["val_1"],
            provenance={"prior_source": "predicted_prior"},
            confidence_threshold=0.5,
            wg_threshold=0.5,
            max_locations_per_case=5,
            progress=False,
        )


def test_write_and_load_roundtrip_validates_dataset_and_fold(tmp_path: Path):
    payload = build_hard_negative_set(
        ["case_a"],
        _synthetic_arrays,
        dataset_name="Dataset605_PICAI",
        fold=0,
        train_cases=["case_a"],
        val_cases=["val_1"],
        provenance={"prior_source": "predicted_prior"},
        confidence_threshold=0.5,
        wg_threshold=0.5,
        max_locations_per_case=5,
        progress=False,
    )
    target = tmp_path / "hn.json"
    write_hard_negative_set(target, payload)
    loaded = load_hard_negative_set(
        target, expected_dataset_name="Dataset605_PICAI", expected_fold=0
    )
    assert loaded["cases"]["case_a"]["locations"] == [[2, 2, 2]]

    with pytest.raises(HardNegativeMiningError, match="数据集归属"):
        load_hard_negative_set(
            target, expected_dataset_name="Dataset606_PICAI_Zonal", expected_fold=0
        )
    with pytest.raises(HardNegativeMiningError, match="fold 归属"):
        load_hard_negative_set(
            target, expected_dataset_name="Dataset605_PICAI", expected_fold=1
        )
    with pytest.raises(HardNegativeMiningError, match="拒绝静默覆盖"):
        write_hard_negative_set(target, payload)


def test_load_rejects_illegal_coordinates_and_schema(tmp_path: Path):
    path = tmp_path / "hn.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": HARD_NEGATIVE_SCHEMA_VERSION,
                "dataset_name": "Dataset605_PICAI",
                "fold": 0,
                "split": {"train_cases": ["case_a"], "val_cases": ["val_1"]},
                "cases": {"case_a": {"locations": [[1, 1]]}},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(HardNegativeMiningError, match="非法困难负样本坐标"):
        load_hard_negative_set(
            path, expected_dataset_name="Dataset605_PICAI", expected_fold=0
        )

    path.write_text(json.dumps({"schema_version": "0.0"}), encoding="utf-8")
    with pytest.raises(HardNegativeMiningError, match="schema_version"):
        load_hard_negative_set(
            path, expected_dataset_name="Dataset605_PICAI", expected_fold=0
        )


# --------------------------------------------------------------------------- sampler slots
class _MinimalLabelManager:
    """满足 ``nnUNetDataLoader`` 构造所需的最小 LabelManager 替身（纯合成）。"""

    # nnU-Net 原生构造 ``tuple([-1] + label_manager.all_labels)``，因此必须是 list
    all_labels: ClassVar[list[int]] = [0, 1]
    foreground_labels = (1,)
    foreground_regions = None
    ignore_label = None
    has_ignore_label = False


class _MinimalDataset:
    """满足 ``nnUNetDataLoader.__init__``（determine_shapes + 选例）的最小 dataset 替身。"""

    def __init__(self, identifiers, shape=(6, 8, 8)) -> None:
        self.identifiers = tuple(identifiers)
        self.source_folder = "."
        self._shape = shape

    def load_case(self, identifier):
        channels, *spatial = self._shape
        data = np.zeros((channels, *spatial), dtype=np.float32)
        seg = np.zeros((1, *spatial), dtype=np.int16)
        properties = {"class_locations": {1: np.array([[0, 1, 1, 1]])}}
        return data, seg, None, properties


def _make_loader(
    batch_size=4,
    identifiers=("case_a", "case_b", "case_c"),
    hard_negative_locations=None,
    hard_negative_cases_per_batch=1,
    positive_cases_per_batch=1,
):
    """用真实 ``HardNegativeDataLoader`` 构造器 + 最小合成 dataset 装配 loader。"""
    from zonal_reliability_fusion.sampling.hard_negative import HardNegativeDataLoader as _H

    dataset = _MinimalDataset(identifiers)
    return _H(
        dataset,
        batch_size,
        (4, 4, 4),
        (4, 4, 4),
        _MinimalLabelManager(),
        oversample_foreground_percent=0.5,
        sampling_probabilities=None,
        pad_sides=None,
        transforms=None,
        probabilistic_oversampling=False,
        positive_case_identifiers=("case_a", "case_b"),
        positive_cases_per_batch=positive_cases_per_batch,
        hard_negative_locations=hard_negative_locations
        if hard_negative_locations is not None
        else {"case_b": ((1, 2, 3), (4, 5, 6))},
        hard_negative_cases_per_batch=hard_negative_cases_per_batch,
    )


def test_slot_layout_is_positive_last_and_hard_negative_before_it():
    loader = _make_loader(batch_size=4)
    assert loader._first_hard_negative_position() == 2
    np.random.seed(0)
    keys = loader.get_indices()
    assert len(keys) == 4
    assert keys[3] in loader.positive_case_identifiers
    assert keys[2] in loader.hard_negative_locations


def test_hard_negative_slot_uses_stored_location_and_is_not_force_fg():
    loader = _make_loader(batch_size=4, hard_negative_locations={"case_b": ((5, 6, 7),)})
    np.random.seed(1)
    loader.get_indices()
    class_locations = {1: np.array([[0, 1, 1, 1]])}
    # 槽位 0、1：普通槽位，force_fg=False 时走 nnU-Net 原生随机裁剪
    for _ in range(2):
        lbs, ubs = loader.get_bbox((6, 8, 8), False, class_locations, None, False)
        assert len(lbs) == 3 and len(ubs) == 3
    # 槽位 2：困难负样本槽位，bbox 以存储位置为中心（patch 4 -> 下界 [3,4,5]）
    lbs, ubs = loader.get_bbox((6, 8, 8), False, class_locations, None, False)
    assert lbs == [3, 4, 5]
    assert ubs == [7, 8, 9]
    # 槽位 3 为阳性槽位：走原生病灶中心裁剪，消费掉最后一个决策后队列必须清空
    loader.get_bbox((6, 8, 8), True, class_locations, None, False)
    assert not loader._slot_decisions


def test_hard_negative_slot_rejects_force_fg():
    """存储位置是背景；被标记为 force_fg 时必须报错而不是静默错位。"""
    loader = _make_loader(batch_size=4, hard_negative_locations={"case_b": ((5, 6, 7),)})
    np.random.seed(2)
    loader.get_indices()
    class_locations = {1: np.array([[0, 1, 1, 1]])}
    loader.get_bbox((6, 8, 8), False, class_locations, None, False)
    loader.get_bbox((6, 8, 8), False, class_locations, None, False)
    with pytest.raises(RuntimeError, match="force_fg"):
        loader.get_bbox((6, 8, 8), True, class_locations, None, False)


def test_slot_decision_queue_is_reset_and_guarded():
    loader = _make_loader(batch_size=4)
    np.random.seed(3)
    loader.get_indices()
    with pytest.raises(RuntimeError, match="未全部消费"):
        loader.get_indices()


def test_get_bbox_without_get_indices_is_fail_closed():
    loader = _make_loader(batch_size=4)
    loader._slot_decisions = None
    with pytest.raises(RuntimeError, match="未经过 get_indices"):
        loader.get_bbox((6, 8, 8), False, None, None, False)


def test_oversample_marks_only_positive_slots():
    loader = _make_loader(batch_size=4)
    assert [loader._oversample_last_XX_percent(i) for i in range(4)] == [
        False, False, False, True,
    ]


def test_loader_constructor_rejects_slot_overflow_and_unknown_cases():
    from zonal_reliability_fusion.sampling.hard_negative import HardNegativeDataLoader as _H

    dataset = _MinimalDataset(("case_a",))
    common = {
        "oversample_foreground_percent": 0.5,
        "sampling_probabilities": None,
        "pad_sides": None,
        "transforms": None,
        "probabilistic_oversampling": False,
        "positive_case_identifiers": ("case_a",),
        "positive_cases_per_batch": 1,
    }
    with pytest.raises(ValueError, match="不能超过"):
        _H(
            dataset, 2, (4, 4, 4), (4, 4, 4), _MinimalLabelManager(),
            hard_negative_locations={"case_a": [[1, 1, 1]]},
            hard_negative_cases_per_batch=2, **common,
        )
    with pytest.raises(ValueError, match="未知 ID"):
        _H(
            dataset, 4, (4, 4, 4), (4, 4, 4), _MinimalLabelManager(),
            hard_negative_locations={"case_zzz": [[1, 1, 1]]},
            hard_negative_cases_per_batch=1, **common,
        )
    with pytest.raises(ValueError, match="不能为空"):
        _H(
            dataset, 4, (4, 4, 4), (4, 4, 4), _MinimalLabelManager(),
            hard_negative_locations={}, hard_negative_cases_per_batch=1, **common,
        )


def test_composed_roi_and_hard_negative_loader_consumes_both_slot_queues():
    from zonal_reliability_fusion.sampling.hard_negative import ROIHardNegativeDataLoader

    identifiers = ("case_a", "case_b", "case_c")
    loader = ROIHardNegativeDataLoader(
        _MinimalDataset(identifiers, shape=(3, 12, 12, 12)), 4, (4, 4, 4), (4, 4, 4),
        _MinimalLabelManager(), oversample_foreground_percent=.5,
        probabilistic_oversampling=False, transforms=None,
        positive_case_identifiers=("case_a", "case_b"), positive_cases_per_batch=1,
        hard_negative_locations={"case_b": ((6, 6, 6),)}, hard_negative_cases_per_batch=1,
        roi_boxes=dict.fromkeys(identifiers, ((1, 1, 1), (8, 8, 8))),
        roi_sampling_probability=1.,
    )
    np.random.seed(7)
    for _ in range(3):
        keys = loader.get_indices()
        assert keys[2] == "case_b"
        for slot in range(4):
            lower, upper = loader.get_bbox((12, 12, 12), slot == 3,
                                           {1: np.array([[0, 2, 2, 2]])})
            if slot == 2:
                assert list(lower) == [4, 4, 4]
            elif slot < 2:
                assert all(1 <= v <= 4 for v in lower)
            assert all(upper[i] - lower[i] == 4 for i in range(3))
        assert not loader._slot_decisions and not loader._pending_keys
