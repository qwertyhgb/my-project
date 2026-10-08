"""Stage-1 解剖先验的契约、不确定度与区域指标的纯合成测试。

覆盖 §33 要求的 **anatomy probability shape** 与 ORACLE 标记纪律：

- WG/PZ/TZ soft probability 的形状/值域/有限性必须被显式校验（评估不重采样）；
- 先验不确定度报告输出低置信比例与 zone margin；
- 区域重叠指标与 nnU-Net 原生 region 定义一致，空-空 Dice 为 ``None``；
- FP 负担分解**不接受**未标记的 GT 解剖标签（``ORACLE_GT`` 必须显式标注）。
"""

from __future__ import annotations

import numpy as np
import pytest

from zonal_reliability_fusion.anatomy.contracts import (
    ANATOMY_CLASS_ORDER,
    ANATOMY_COUNTS,
    ANATOMY_DATASET,
    ANATOMY_KNOWN_MISSING,
    ANATOMY_LABELS,
    encode_anatomy,
    encode_anatomy_heads,
    validate_anatomy_dataset,
    validate_label_array,
)
from zonal_reliability_fusion.anatomy.validation import (
    anatomy_region_agreement,
    anatomy_region_metrics,
    prior_uncertainty_report,
)
from zonal_reliability_fusion.evaluation.anatomy_metrics import (
    PRIOR_SOURCE_ORACLE_GT,
    PRIOR_SOURCE_PREDICTED,
    ZONE_KEYS,
    false_positive_burden,
    lesion_anatomy_localization,
    validate_anatomy_priors,
    zone_assignment,
)
from zonal_reliability_fusion.evaluation.protocol import EvaluationError


# --------------------------------------------------------------------------- contract
def _valid_dataset_document(num_training=None):
    case_ids = [f"{10000 + i}_{1000000 + i}" for i in range(3)]
    return {
        "channel_names": {"0000": "T2W"},
        "labels": ANATOMY_LABELS,
        "regions_class_order": ANATOMY_CLASS_ORDER,
        "numTraining": len(case_ids) if num_training is None else num_training,
        "anatomy_contract": {
            "encoding": "WG+2*PZ+4*TZ",
            "wg_source": "materialized_wg",
            "zonal_source": "zonal_yuan",
            "explicit_exclusions": [ANATOMY_KNOWN_MISSING],
            "case_ids": case_ids,
            "train_cases": case_ids[:2],
            "val_cases": case_ids[2:],
        },
    }


def test_dataset_contract_requires_single_t2w_and_three_region_heads():
    doc = _valid_dataset_document()
    assert ANATOMY_DATASET == "Dataset607_PICAI_Anatomy"
    # 数据集身份 / 配置必须精确匹配（不接受别名或其它配置）
    with pytest.raises(ValueError, match="Dataset607"):
        validate_anatomy_dataset(doc, dataset_name="Dataset605_PICAI")
    with pytest.raises(ValueError, match="Dataset607"):
        validate_anatomy_dataset(doc, configuration="2d")
    # 必须是单通道 T2W（多通道意味着 anatomy 模型不再是「只看 T2W」）
    with pytest.raises(ValueError, match="single T2W channel"):
        validate_anatomy_dataset({**doc, "channel_names": {"0000": "T2W", "0001": "ADC"}})
    # region 定义与顺序必须与契约一致
    with pytest.raises(ValueError, match="regions_class_order"):
        validate_anatomy_dataset({**doc, "regions_class_order": [1, 2, 3]})
    with pytest.raises(ValueError, match="provenance"):
        bad = _valid_dataset_document()
        bad["anatomy_contract"]["wg_source"] = "guessed"
        validate_anatomy_dataset(bad)


def test_dataset_contract_rejects_known_missing_case_in_scope():
    """已知缺失病例不得以空掩膜顶替：出现在 case_ids 里就必须拒绝。

    构造一个**规模正确**（1499 例）的合成契约，使校验能走到 known-missing 分支，而不是先在
    study scope 检查处失败——否则这条守卫实际上没有被测试覆盖。
    """
    doc = _valid_dataset_document()
    case_ids = [f"{10000 + i}_{1000000 + i}" for i in range(ANATOMY_COUNTS[0] - 1)]
    case_ids.append(ANATOMY_KNOWN_MISSING)
    doc["anatomy_contract"]["case_ids"] = case_ids
    doc["numTraining"] = len(case_ids)
    with pytest.raises(ValueError, match="known missing WG"):
        validate_anatomy_dataset(doc)


def test_dataset_contract_checks_study_scope_against_frozen_counts():
    doc = _valid_dataset_document()
    # 与冻结的 ANATOMY_COUNTS 不同 -> 必须报错（防止静默缩小/扩大研究范围）
    with pytest.raises(ValueError, match="study scope"):
        validate_anatomy_dataset(doc)
    assert ANATOMY_COUNTS == (1499, 1276, 223)


def test_encode_anatomy_uses_bit_pattern_and_rejects_illegal_labels():
    wg = np.zeros((2, 2, 2), dtype=np.uint8)
    wg[0, 0, 0] = 1
    yuan = np.zeros((2, 2, 2), dtype=np.uint8)
    yuan[0, 0, 1] = 1
    yuan[0, 0, 0] = 2
    encoded = encode_anatomy(wg, yuan)
    assert encoded[0, 0, 0] == 1 + 4  # WG + TZ
    assert encoded[0, 0, 1] == 2  # PZ
    with pytest.raises(ValueError, match="illegal labels"):
        encode_anatomy(wg, np.full((2, 2, 2), 3, dtype=np.uint8))
    with pytest.raises(ValueError, match="shape mismatch"):
        encode_anatomy(wg, np.zeros((3, 3, 3), dtype=np.uint8))


def test_encode_heads_roundtrips_with_contract_bit_pattern():
    wg = np.zeros((6, 6, 6), dtype=np.uint8)
    pz = np.zeros((6, 6, 6), dtype=np.uint8)
    tz = np.zeros((6, 6, 6), dtype=np.uint8)
    wg[1, 1, 1] = pz[2, 2, 2] = tz[3, 3, 3] = 1
    encoded = encode_anatomy_heads(np.stack([wg, pz, tz]))
    assert encoded[1, 1, 1] == 1
    assert encoded[2, 2, 2] == 2
    assert encoded[3, 3, 3] == 4
    with pytest.raises(ValueError, match="three binary"):
        encode_anatomy_heads(np.zeros((2, 4, 4, 4), dtype=np.uint8))


def test_label_array_validation_is_fail_closed():
    with pytest.raises(ValueError, match="3D"):
        validate_label_array(np.zeros((4, 4)), range(8), "x")
    with pytest.raises(ValueError, match="illegal labels"):
        validate_label_array(np.zeros((4, 4, 4)) + 9, range(8), "x")
    with pytest.raises(ValueError, match="finite"):
        bad = np.zeros((4, 4, 4))
        bad[0, 0, 0] = np.nan
        validate_label_array(bad, range(8), "x")


# --------------------------------------------------------------------------- region metrics
def test_region_metrics_follow_native_region_definition():
    wg = np.zeros((8, 8, 8), dtype=np.uint8)
    pz = np.zeros((8, 8, 8), dtype=np.uint8)
    wg[0:4, 0:4, 0:4] = 1
    pz[0:2, 0:2, 0:2] = 1
    reference = encode_anatomy_heads(np.stack([wg, pz, np.zeros_like(pz)]))
    metrics = anatomy_region_metrics(reference, reference)
    for region in ("WG", "PZ", "TZ"):
        assert metrics[region]["TP"] > 0 or region == "TZ"
    assert metrics["WG"]["Dice"] == pytest.approx(1.0)
    assert metrics["PZ"]["Dice"] == pytest.approx(1.0)
    # 参考为空的区域：Dice 未定义（None），不是 0 也不是 1
    assert metrics["TZ"]["Dice"] is None


def test_region_metrics_reject_shape_and_label_mismatch():
    reference = np.zeros((4, 4, 4), dtype=np.uint8)
    with pytest.raises(EvaluationError, match="shape mismatch"):
        anatomy_region_metrics(reference, np.zeros((5, 4, 4), dtype=np.uint8))
    with pytest.raises(ValueError, match="illegal labels"):
        anatomy_region_metrics(reference, np.full((4, 4, 4), 9, dtype=np.uint8))


def test_region_agreement_reports_none_for_empty_denominators():
    masks = (
        np.zeros((4, 4, 4), dtype=bool),
        np.zeros((4, 4, 4), dtype=bool),
        np.zeros((4, 4, 4), dtype=bool),
    )
    report = anatomy_region_agreement(masks)
    assert report["PZ_outside_WG_over_PZ"]["ratio"] is None
    assert report["PZ_outside_WG_over_PZ"]["denominator_voxels"] == 0


def test_region_agreement_detects_zones_outside_wg():
    wg = np.zeros((4, 4, 4), dtype=bool)
    wg[0, 0, 0] = True
    pz = np.zeros((4, 4, 4), dtype=bool)
    pz[0, 0, 0] = True
    pz[1, 1, 1] = True  # 越出 WG
    tz = np.zeros((4, 4, 4), dtype=bool)
    report = anatomy_region_agreement((wg, pz, tz))
    assert report["PZ_outside_WG_over_PZ"]["ratio"] == pytest.approx(0.5)


# --------------------------------------------------------------------------- uncertainty
def test_prior_uncertainty_report_quantifies_undecided_regions():
    wg = np.full((4, 4, 4), 0.5)  # 全部落在低置信带
    pz = np.full((4, 4, 4), 0.1)
    tz = np.full((4, 4, 4), 0.05)
    report = prior_uncertainty_report(
        wg, pz, tz, low_confidence_threshold=0.2
    )
    assert report["wg_low_confidence_fraction"] == pytest.approx(1.0)
    assert report["zone_undecided_fraction"] == pytest.approx(1.0)
    assert report["zone_margin"]["mean"] == pytest.approx(0.05)
    assert "not_a_calibration_claim" in report["definitions"]


def test_prior_uncertainty_report_validates_threshold_and_shapes():
    with pytest.raises(EvaluationError, match="low_confidence_threshold"):
        prior_uncertainty_report(
            np.zeros((4, 4, 4)), np.zeros((4, 4, 4)), np.zeros((4, 4, 4)),
            low_confidence_threshold=0.9,
        )
    with pytest.raises(EvaluationError, match="形状"):
        prior_uncertainty_report(
            np.zeros((4, 4, 4)), np.zeros((5, 4, 4)), np.zeros((4, 4, 4)),
            low_confidence_threshold=0.2,
        )


# --------------------------------------------------------------------------- FP burden
def test_validate_anatomy_priors_enforces_shape_finite_and_range():
    reference = np.zeros((4, 4, 4), dtype=bool)
    with pytest.raises(EvaluationError, match="形状"):
        validate_anatomy_priors(np.zeros((5, 4, 4)), np.zeros((4, 4, 4)), np.zeros((4, 4, 4)), reference.shape)
    with pytest.raises(EvaluationError, match=r"\[0, 1\]"):
        validate_anatomy_priors(np.full((4, 4, 4), 1.5), np.zeros((4, 4, 4)), np.zeros((4, 4, 4)), reference.shape)
    with pytest.raises(EvaluationError, match="非有限"):
        bad = np.zeros((4, 4, 4))
        bad[0, 0, 0] = np.inf
        validate_anatomy_priors(bad, np.zeros((4, 4, 4)), np.zeros((4, 4, 4)), reference.shape)


def test_zone_assignment_is_unique_and_covers_all_voxels():
    pz = np.array([[[0.9]]], dtype=np.float64)
    tz = np.array([[[0.8]]], dtype=np.float64)
    index = zone_assignment(pz, tz, 0.5)
    assert index.shape == pz.shape
    assert index.item() == 0  # argmax -> PZ，不与 TZ 双计
    both_low = zone_assignment(np.array([[[0.1]]]), np.array([[[0.2]]]), 0.5)
    assert both_low.item() == 2  # uncertain
    with pytest.raises(EvaluationError, match="zone_threshold"):
        zone_assignment(pz, tz, 1.5)


def test_false_positive_burden_decomposes_by_predicted_anatomy():
    shape = (4, 4, 4)
    pred = np.zeros(shape, dtype=bool)
    ref = np.zeros(shape, dtype=bool)
    pred[0, 0, 0] = True  # 腺体内
    pred[3, 3, 3] = True  # 腺体外
    wg = np.zeros(shape, dtype=np.float64)
    wg[0:2, 0:2, 0:2] = 1.0
    pz = np.zeros(shape, dtype=np.float64)
    pz[0, 0, 0] = 0.9
    tz = np.zeros(shape, dtype=np.float64)

    report = false_positive_burden(
        pred, ref, wg, pz, tz, voxel_volume_mm3=1.5,
        wg_threshold=0.5, zone_threshold=0.5,
    )
    assert report["prior_source"] == PRIOR_SOURCE_PREDICTED
    assert report["fp_voxels_total"] == 2
    assert report["fp_voxels"]["inside_wg"] == 1
    assert report["fp_voxels"]["outside_wg"] == 1
    assert report["fp_voxels"]["PZ"] == 1
    assert report["fp_voxels"]["TZ"] == 0
    assert sum(report["fp_voxels"][key] for key in ZONE_KEYS) == 1
    assert report["fp_volume_mm3"]["outside_wg"] == pytest.approx(1.5)
    assert report["fp_component_count"]["total"] == 2
    assert report["fp_component_count"]["by_gland"] == {"inside_wg": 1, "outside_wg": 1}


def test_oracle_gt_must_be_declared_explicitly_and_is_recorded_verbatim():
    """GT 解剖标签只能用于上界分析；未标注时等同于把 oracle 当正式结果，必须拒绝。"""
    shape = (4, 4, 4)
    with pytest.raises(EvaluationError, match="prior_source"):
        false_positive_burden(
            np.zeros(shape, dtype=bool), np.zeros(shape, dtype=bool),
            np.zeros(shape), np.zeros(shape), np.zeros(shape), 1.0,
            wg_threshold=0.5, zone_threshold=0.5, prior_source="gt_labels",
        )
    report = false_positive_burden(
        np.zeros(shape, dtype=bool), np.zeros(shape, dtype=bool),
        np.zeros(shape), np.zeros(shape), np.zeros(shape), 1.0,
        wg_threshold=0.5, zone_threshold=0.5, prior_source=PRIOR_SOURCE_ORACLE_GT,
    )
    assert report["prior_source"] == PRIOR_SOURCE_ORACLE_GT
    assert "ORACLE_GT" in report["definitions"]["prior_source"]


def test_lesion_anatomy_localization_reuses_the_same_missed_definition():
    """区域分解的「missed」必须与主终点使用同一匹配协议（6-邻域 + any-overlap）。"""
    shape = (6, 6, 6)
    ref = np.zeros(shape, dtype=bool)
    ref[1, 1, 1] = True  # PZ 内，被检出
    ref[4, 4, 4] = True  # TZ 内，完全漏检
    pred = np.zeros(shape, dtype=bool)
    pred[1, 1, 1] = True  # 与第一个病灶重叠 -> matched
    pred[3, 3, 3] = True  # 额外假阳团块（不进 lesion 级匹配）
    wg = np.ones(shape)
    pz = np.zeros(shape)
    pz[1, 1, 1] = 0.9
    tz = np.zeros(shape)
    tz[4, 4, 4] = 0.9

    report = lesion_anatomy_localization(
        pred, ref, wg, pz, tz, voxel_volume_mm3=1.0,
        wg_threshold=0.5, zone_threshold=0.5,
    )
    assert report["reference_lesions"] == 2
    by_region = report["by_anatomy_region"]
    assert by_region["PZ"]["reference_lesion_count"] == 1
    assert by_region["PZ"]["lesion_sensitivity_any_overlap"] == pytest.approx(1.0)
    assert by_region["TZ"]["lesion_sensitivity_any_overlap"] == pytest.approx(0.0)
    assert by_region["TZ"]["matched_reference_lesion_count"] == 0
