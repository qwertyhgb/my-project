"""`diagnose_n0_zero_dice` 与 `nnunet_dataloader_probe` 的合成单元测试。

约束（AGENTS.md §2）：
- **不读取任何真实医学数据**：全部使用合成数组与 tmp_path 下的合成 JSON/CSV 元数据；
- **不 import nnunetv2**：适配层只在函数内部惰性 import，测试同时断言这一点；
- 不初始化 CUDA，不启动训练或推理。
"""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zonal_reliability_fusion.integrations import nnunet_dataloader_probe as probe  # noqa: E402


def _load_script_module():
    path = PROJECT_ROOT / "scripts/train/diagnose_n0_zero_dice.py"
    spec = importlib.util.spec_from_file_location("diagnose_n0_zero_dice", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


diagnose = _load_script_module()


# --------------------------------------------------------------------------- 依赖边界


def test_adapter_import_does_not_import_nnunetv2():
    """在独立子进程中验证：导入适配层本身不触发 nnunetv2（惰性 import 边界）。

    不能用 `"nnunetv2" not in sys.modules` 直接断言：pytest 会在同一进程内收集其它测试文件，
    任何一处 import 都会污染全局状态（本测试先前因此不稳定）。
    """
    import subprocess

    code = (
        "import sys; sys.path.insert(0, r'"
        + str(SRC)
        + "'); import zonal_reliability_fusion.integrations.nnunet_dataloader_probe as m; "
        + "print('NNUNETV2_IMPORTED=' + str('nnunetv2' in sys.modules))"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert "NNUNETV2_IMPORTED=False" in proc.stdout, proc.stdout + proc.stderr


# --------------------------------------------------------------------------- 纯函数：patch 统计


def test_patch_foreground_stats_detects_foreground_and_bbox():
    seg = np.zeros((4, 5, 6), dtype=np.int16)
    seg[1:3, 2:4, 3:5] = 1
    stats = probe.patch_foreground_stats(seg)
    assert stats["fg_voxels"] == 2 * 2 * 2
    assert stats["labels"] == [0, 1]
    assert stats["binary_ok"] is True
    assert stats["bbox"]["z"] == [1, 2]
    assert stats["bbox"]["y"] == [2, 3]
    assert stats["bbox"]["x"] == [3, 4]
    assert 0 < stats["fg_ratio"] < 1


def test_patch_foreground_stats_accepts_singleton_dims_and_flags_empty():
    seg = np.zeros((1, 1, 3, 3, 3), dtype=np.int16)
    stats = probe.patch_foreground_stats(seg)
    assert stats["fg_voxels"] == 0
    assert stats["bbox"] is None
    assert stats["binary_ok"] is True


def test_patch_foreground_stats_flags_non_binary_labels():
    seg = np.zeros((3, 3, 3), dtype=np.int16)
    seg[0, 0, 0] = 2
    stats = probe.patch_foreground_stats(seg)
    assert stats["binary_ok"] is False
    assert stats["labels"] == [0, 2]


def test_patch_foreground_stats_rejects_multi_patch_input():
    with pytest.raises(ValueError):
        probe.patch_foreground_stats(np.zeros((2, 1, 3, 3, 3), dtype=np.int16))


# --------------------------------------------------------------------------- 纯函数：FG 过采样语义


@pytest.mark.parametrize(
    "batch_size,percent,expected",
    [
        (2, 0.33, [False, True]),          # round(2*0.67)=1 → 每批 1 个 FG 强制 patch
        (4, 0.33, [False, False, False, True]),  # round(4*0.67)=3 → 每批 1 个
        (2, 0.0, [False, False]),          # 关闭过采样
        (1, 0.33, [False]),                # 官方语义：bs=1 时 round(0.67)=1 → 无 FG 强制
        (2, 1.0, [True, True]),            # round(0)=0 → 全部强制
    ],
)
def test_foreground_oversample_flags_match_official_semantics(batch_size, percent, expected):
    assert probe.foreground_oversample_flags(batch_size, percent) == expected


# --------------------------------------------------------------------------- 纯函数：class_locations


def test_class_locations_report_valid_nx4():
    cl = {1: np.array([[1, 2, 3, 4], [1, 5, 6, 7]], dtype=np.int64)}
    rep = probe.class_locations_report(cl, spatial_shape=(10, 10, 10))
    assert rep["present"] and rep["in_bounds"]
    assert rep["n_samples_label"] == 2
    assert rep["label_column_unique"] == [1]
    assert rep["issues"] == []


def test_class_locations_report_missing_and_empty_and_out_of_bounds():
    assert probe.class_locations_report(None, (5, 5, 5))["issues"], "缺失应报问题"
    empty = {1: np.zeros((0, 4), dtype=np.int64)}
    assert "为空" in " ".join(probe.class_locations_report(empty, (5, 5, 5))["issues"])
    oob = {1: np.array([[1, 0, 0, 99]], dtype=np.int64)}
    rep = probe.class_locations_report(oob, (5, 5, 5))
    assert not rep["in_bounds"] and rep["issues"]


def test_class_locations_report_wrong_label_column():
    cl = {1: np.array([[2, 1, 1, 1]], dtype=np.int64)}
    rep = probe.class_locations_report(cl, (5, 5, 5))
    assert any("首列标签索引" in msg for msg in rep["issues"])


def test_class_locations_report_missing_label_key():
    rep = probe.class_locations_report({2: np.array([[2, 1, 1, 1]])}, (5, 5, 5))
    assert any("缺少 label=1" in msg for msg in rep["issues"])


# --------------------------------------------------------------------------- 纯函数：病例异常判定与聚合


def _case_row(**kw):
    row = {
        "case_id": "c1", "split": "val", "expected_positive": True,
        "seg_exists": True, "data_exists": True, "labels_binary": True,
        "n_fg_voxels": 100, "seg_labels": [0, 1], "class_locations_issues": [],
    }
    row.update(kw)
    return row


def test_evaluate_case_row_flags_positive_with_empty_seg():
    issues = probe.evaluate_case_row(_case_row(n_fg_voxels=0, expected_positive=True))
    assert any("阳性病例" in m for m in issues)


def test_evaluate_case_row_flags_negative_with_foreground_and_non_binary():
    issues = probe.evaluate_case_row(_case_row(expected_positive=False, seg_labels=[0, 1]))
    assert any("阴性病例" in m for m in issues)
    issues2 = probe.evaluate_case_row(_case_row(labels_binary=False, seg_labels=[0, 1, 2]))
    assert any("非 {0,1}" in m for m in issues2)


def test_evaluate_case_row_flags_missing_files():
    issues = probe.evaluate_case_row(_case_row(seg_exists=False, data_exists=False))
    assert len([m for m in issues if "缺失" in m]) == 2


def test_aggregate_case_rows_counts_and_percentiles():
    rows = [
        _case_row(case_id="a", split="val", expected_positive=True, n_fg_voxels=100, issues=[]),
        _case_row(case_id="b", split="val", expected_positive=True, n_fg_voxels=300, issues=[]),
        _case_row(case_id="c", split="val", expected_positive=False, n_fg_voxels=0, issues=[]),
        _case_row(case_id="d", split="val", expected_positive=True, n_fg_voxels=0,
                  issues=["阳性病例（case_csPCa=YES）预处理 seg 全空 → 标签链路可能丢失前景"]),
        _case_row(case_id="e", split="train", expected_positive=True, n_fg_voxels=50, issues=[]),
    ]
    agg = probe.aggregate_case_rows(rows)
    assert set(agg) == {"train", "val"}
    assert agg["val"]["n_cases"] == 4
    assert agg["val"]["n_expected_positive"] == 3
    assert agg["val"]["n_expected_negative"] == 1
    assert agg["val"]["n_positive_with_empty_seg"] == 1
    assert agg["val"]["n_anomalies"] == 1
    assert agg["val"]["anomalies"][0]["case_id"] == "d"
    assert agg["val"]["positive_fg_voxels_percentiles"]["n"] == 3


def test_aggregate_batch_rows_phase_aware_lost_rate():
    rows = []
    for i in range(10):
        rows.append({"split": "val", "phase": "pre_aug", "batch_idx": 0, "patch_idx": i,
                     "case_id": f"c{i}", "case_expected_positive": True,
                     "forced_foreground": i % 2 == 1, "fg_voxels": 10, "fg_ratio": 0.1,
                     "labels": [0, 1], "has_foreground": True})
    for i in range(10):
        rows.append({"split": "val", "phase": "post_aug", "batch_idx": 0, "patch_idx": i,
                     "case_id": f"c{i}", "case_expected_positive": True,
                     "forced_foreground": i % 2 == 1, "fg_voxels": 5 if i < 5 else 0,
                     "fg_ratio": 0.05, "labels": [0, 1], "has_foreground": i < 5})
    agg = probe.aggregate_batch_rows(rows)["val"]
    # n_patches / fg_patch_rate 统计 pre_aug（采样）那一遍；post_aug（增强后）单独统计
    assert agg["n_patches"] == 10
    assert agg["fg_patch_rate"] == pytest.approx(1.0)
    assert agg["n_forced_foreground_patches"] == 5  # pre_aug 中 i 为奇数的 patch 被强制取前景
    assert agg["n_forced_foreground_patches_with_fg"] == 5
    assert agg["forced_foreground_hit_rate"] == pytest.approx(1.0)
    assert agg["n_random_patches"] == 5
    assert agg["n_patches_checked_after_aug"] == 10
    assert agg["fg_patch_rate_after_aug"] == pytest.approx(0.5)
    assert agg["lost_foreground_after_aug_rate"] == pytest.approx(0.5, abs=1e-6)


def test_aggregate_batch_rows_without_post_aug_has_no_lost_rate():
    rows = [{"split": "val", "phase": "pre_aug", "forced_foreground": True,
             "fg_voxels": 7, "has_foreground": True, "patch_idx": 0}]
    agg = probe.aggregate_batch_rows(rows)["val"]
    assert agg["lost_foreground_after_aug_rate"] is None
    assert agg["fg_patch_rate"] == 1.0


# --------------------------------------------------------------------------- 纯函数：决策树


def test_build_verdict_static_failure_wins():
    v = probe.build_verdict(False, None, None, static_issues=["train=1277 / val=223 失败"])
    assert v["verdict"] == probe.VERDICT_MAPPING_OR_CONFIG_BROKEN
    assert v["reasons"]


def test_build_verdict_data_broken_before_sampling():
    case_summary = {"val": {"n_missing_files": 0, "n_non_binary_labels": 0,
                            "n_positive_with_empty_seg": 3, "n_class_locations_problems": 0}}
    v = probe.build_verdict(True, case_summary, None)
    assert v["verdict"] == probe.VERDICT_DATA_OR_LABEL_BROKEN


def test_build_verdict_class_locations_problem():
    case_summary = {"val": {"n_missing_files": 0, "n_non_binary_labels": 0,
                            "n_positive_with_empty_seg": 0, "n_class_locations_problems": 5}}
    v = probe.build_verdict(True, case_summary, None)
    assert v["verdict"] == probe.VERDICT_FG_SAMPLING_AT_RISK


def test_build_verdict_fg_sampling_ineffective():
    batch_summary = {"val": {"n_forced_foreground_patches": 20,
                             "n_forced_foreground_patches_with_fg": 0,
                             "fg_patch_rate": 0.0, "lost_foreground_after_aug_rate": None}}
    v = probe.build_verdict(True, None, batch_summary)
    assert v["verdict"] == probe.VERDICT_FG_SAMPLING_INEFFECTIVE


def test_build_verdict_augmentation_loses_foreground():
    batch_summary = {"val": {"n_forced_foreground_patches": 20,
                             "n_forced_foreground_patches_with_fg": 10,
                             "fg_patch_rate": 0.3, "lost_foreground_after_aug_rate": 0.6}}
    v = probe.build_verdict(True, None, batch_summary)
    assert v["verdict"] == probe.VERDICT_AUGMENTATION_LOSES_FOREGROUND


def test_build_verdict_pipeline_ok_and_low_rate():
    ok = {"val": {"n_forced_foreground_patches": 20, "n_forced_foreground_patches_with_fg": 6,
                  "fg_patch_rate": 0.28, "lost_foreground_after_aug_rate": 0.0}}
    assert probe.build_verdict(True, None, ok)["verdict"] == probe.VERDICT_PIPELINE_OK_METRIC_INFORMATIVE
    low = {"val": {"n_forced_foreground_patches": 20, "n_forced_foreground_patches_with_fg": 1,
                   "fg_patch_rate": 0.01, "lost_foreground_after_aug_rate": 0.0}}
    assert probe.build_verdict(True, None, low)["verdict"] == probe.VERDICT_FG_PATCH_RATE_LOW


def test_build_verdict_needs_batch_probe_when_no_sampling():
    assert probe.build_verdict(True, {"val": {"n_missing_files": 0, "n_non_binary_labels": 0,
                                              "n_positive_with_empty_seg": 0,
                                              "n_class_locations_problems": 0}}, None)["verdict"] \
        == probe.VERDICT_NEEDS_BATCH_PROBE


def test_every_verdict_code_has_a_next_step():
    codes = {getattr(probe, n) for n in dir(probe) if n.startswith("VERDICT_")}
    assert len(codes) >= 8, f"判定码数量偏少：{sorted(codes)}"
    for code in codes:
        assert probe._NEXT_STEPS.get(code), f"判定码 {code} 缺少下一步说明"


# --------------------------------------------------------------------------- 脚本：CLI / IO / 日志解析 / 静态审计


def _write_min_metadata(tmp: Path) -> dict:
    """合成一套自洽的元数据（无真实数据）。"""
    cases = [f"{10000 + i}_{1000000 + i}" for i in range(4)]
    dataset_json = {
        "name": "Dataset605_PICAI",
        "channel_names": {"0000": "T2W", "0001": "ADC", "0002": "HBV"},
        "labels": {"background": 0, "lesion": 1},
        "numTraining": 4,
        "file_ending": ".nii.gz",
    }
    plans = {"configurations": {"3d_fullres": {
        "patch_size": [16, 320, 320], "spacing": [3.0, 0.5, 0.5], "batch_size": 2,
        "data_identifier": "nnUNetPlans_3d_fullres", "batch_dice": False,
        "use_mask_for_norm": [False, False, False],
        "normalization_schemes": ["ZScoreNormalization"] * 3,
        "architecture": {"arch_kwargs": {"n_stages": 7}}}}}
    splits = [{"train": cases[:3], "val": cases[3:]}]
    (tmp / "dataset.json").write_text(json.dumps(dataset_json), encoding="utf-8")
    (tmp / "nnUNetPlans.json").write_text(json.dumps(plans), encoding="utf-8")
    (tmp / "splits_final.json").write_text(json.dumps(splits), encoding="utf-8")

    with open(tmp / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv_writer = __import__("csv").DictWriter(
            f, fieldnames=["patient_id", "study_id", "case_id", "case_csPCa"])
        csv_writer.writeheader()
        for i, case in enumerate(cases):
            pid, sid = case.split("_")
            w.writerow({"patient_id": pid, "study_id": sid, "case_id": case,
                        "case_csPCa": "YES" if i % 2 == 0 else "NO"})
    with open(tmp / "split_cases.csv", "w", newline="", encoding="utf-8") as f:
        w = __import__("csv").DictWriter(
            f, fieldnames=["case_id", "study_id", "patient_id", "center", "case_csPCa", "split"])
        w.writeheader()
        for i, case in enumerate(cases):
            pid, sid = case.split("_")
            w.writerow({"case_id": case, "study_id": sid, "patient_id": pid, "center": "RUMC",
                        "case_csPCa": "YES" if i % 2 == 0 else "NO",
                        "split": "train" if i < 3 else "validation"})
    return {"cases": cases, "dataset_json": dataset_json, "plans": plans}


def _fake_args(tmp: Path, **kw) -> types.SimpleNamespace:
    base = dict(
        dataset_json=tmp / "dataset.json",
        plans=tmp / "nnUNetPlans.json",
        splits=tmp / "splits_final.json",
        split_csv=tmp / "split_cases.csv",
        manifest=tmp / "manifest.csv",
        configuration="3d_fullres",
        fold=0,
    )
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_static_audit_passes_on_consistent_metadata(tmp_path):
    _write_min_metadata(tmp_path)
    res = diagnose.static_audit(_fake_args(tmp_path), expected_counts={"train": 3, "val": 1})
    assert res["ok"], res["failed_checks"]
    assert res["positive_cases_by_split"]["train"] == 2


def test_static_audit_detects_wrong_label_mapping(tmp_path):
    meta = _write_min_metadata(tmp_path)
    bad = dict(meta["dataset_json"])
    bad["labels"] = {"background": 0, "lesion": 2}
    (tmp_path / "dataset.json").write_text(json.dumps(bad), encoding="utf-8")
    res = diagnose.static_audit(_fake_args(tmp_path), expected_counts={"train": 3, "val": 1})
    assert not res["ok"]
    assert any("标签" in c for c in res["failed_checks"])


def test_static_audit_detects_split_csv_mismatch(tmp_path):
    _write_min_metadata(tmp_path)
    with open(tmp_path / "split_cases.csv", "a", encoding="utf-8") as f:
        f.write("9999_1999999,1999999,9999,RUMC,NO,validation\n")
    res = diagnose.static_audit(_fake_args(tmp_path), expected_counts={"train": 3, "val": 1})
    assert not res["ok"]


def test_training_log_audit_parses_pseudo_dice_and_loss(tmp_path):
    log = tmp_path / "training_log_2026_9_15_01_17_28.txt"
    log.write_text(
        "2026-09-15 01:20:07.337400: train_loss -0.0667 \n"
        "2026-09-15 01:20:07.337937: val_loss -0.077 \n"
        "2026-09-15 01:20:07.337937: Pseudo dice [np.float32(0.0)] \n"
        "2026-09-15 01:20:07.339328: Epoch time: 150.86 s \n"
        "2026-09-15 01:20:07.343686: Yayy! New best EMA pseudo Dice: 0.0 \n"
        "2026-09-15 01:40:27.044327: Pseudo dice [np.float32(0.0002)] \n",
        encoding="utf-8",
    )
    res = diagnose.training_log_audit(tmp_path)
    assert res["found"]
    assert res["n_epochs_logged"] == 2
    assert res["n_epochs_pseudo_dice_nonzero"] == 1
    assert res["first_nonzero_pseudo_dice_epoch"] == 1
    assert res["train_loss_last"] == pytest.approx(-0.0667)
    assert res["epoch_time_last_sec"] == pytest.approx(150.86)
    assert res["n_yayy_new_best_events"] == 1


def test_training_log_audit_without_log(tmp_path):
    res = diagnose.training_log_audit(tmp_path)
    assert res["found"] is False and res["log_file"] is None


def test_parse_args_defaults_and_overrides():
    args = diagnose.parse_args([])
    assert args.split == "both" and args.max_batches == 0 and args.fold == 0
    # parse_args 末尾会填充默认输出目录 outputs/diagnostics/n0_zero_dice/<timestamp>
    assert "outputs/diagnostics/n0_zero_dice" in str(args.output)
    assert args.splits.name == "splits_final.json" and args.plans.name == "nnUNetPlans.json"
    args2 = diagnose.parse_args(["--split", "val", "--max-batches", "5", "--with-augmentation",
                                "--no-progress", "--max-cases", "20"])
    assert args2.split == "val" and args2.max_batches == 5 and args2.with_augmentation
    assert args2.no_progress and args2.max_cases == 20 and args2.output is not None


def test_write_csv_and_json_roundtrip(tmp_path):
    rows = [{"a": 1, "b": [0, 1], "c": None}, {"a": 2, "b": [], "c": "x"}]
    diagnose._write_csv(rows, tmp_path / "x.csv", ["a", "b", "c"])
    text = (tmp_path / "x.csv").read_text(encoding="utf-8")
    assert "a,b,c" in text and "[0, 1]" in text
    diagnose._write_json({"k": [1, 2]}, tmp_path / "x.json")
    assert json.loads((tmp_path / "x.json").read_text(encoding="utf-8")) == {"k": [1, 2]}


def test_augmentation_config_replication_is_anisotropic_for_plan_patch():
    cfg = probe.derive_official_aug_config([16, 320, 320])
    # max/patch[0] = 20 > ANISO_THRESHOLD(3) → dummy 2D 增强；与训练日志 "do_dummy_2d_data_aug: True" 一致
    assert cfg["do_dummy_2d_data_aug"] is True
    assert cfg["mirror_axes"] == [0, 1, 2]
    assert cfg["rotation_for_DA"][0] == pytest.approx(-np.pi)
