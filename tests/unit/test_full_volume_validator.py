"""FullVolumeValidator 测试（假内存 store + 合成 3D 张量 + CPU；不读真实数据、不初始化 CUDA）。

合成模型是逐点模型：`logits = [-x0, +x0]`，因此「预测为正前景」等价于输入 channel0 > 0。
滑窗在 patch == 体积尺寸时是恒等恢复（已有专门测试），所以可以精确构造 Dice。
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import torch

from zonal_reliability_fusion.data import CaseProperties, PreprocessedCase
from zonal_reliability_fusion.evaluation import CaseResult, FullVolumeValidator, ValidationProtocol

SHAPE = (4, 8, 8)
PATCH = (4, 8, 8)


class PointwiseLogits(torch.nn.Module):
    """logits = [-x0, +x0]：channel0 > 0 的体素被判为前景。"""

    def __init__(self, *, deep_supervision: bool = False, nonfinite: bool = False) -> None:
        super().__init__()
        self.deep_supervision = bool(deep_supervision)
        self.nonfinite = bool(nonfinite)
        self.observed_ds: list[bool] = []
        self.observed_training: list[bool] = []

    def set_deep_supervision(self, enabled: bool) -> None:
        self.deep_supervision = bool(enabled)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.observed_ds.append(self.deep_supervision)
        self.observed_training.append(self.training)
        logits = torch.cat([-x[:, :1], x[:, :1]], dim=1)
        if self.nonfinite:
            logits = logits * float("inf") * 0.0 + float("nan")  # 注入 NaN
        return logits


def box(shape=SHAPE, z=(1, 2), y=(2, 4), x=(2, 4)) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[z[0] : z[1], y[0] : y[1], x[0] : x[1]] = True
    return mask


def make_case(case_id: str, *, gt_mask=None, pred_mask=None) -> PreprocessedCase:
    data = np.full((3, *SHAPE), -1.0, dtype=np.float32)
    if pred_mask is not None:
        data[0][pred_mask] = 1.0
    seg = np.zeros((1, *SHAPE), dtype=np.uint8)
    if gt_mask is not None:
        seg[0][gt_mask] = 1
    return PreprocessedCase(
        case_id=case_id,
        data=data,
        seg=seg,
        properties=CaseProperties(spacing=None, shape_before_cropping=None, class_locations=None),
    )


class RecordingStore:
    """假 store：记录每个 case 被读取的次数与顺序。"""

    def __init__(self, cases: dict[str, PreprocessedCase]) -> None:
        self.cases = cases
        self.calls: list[str] = []

    @property
    def val_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.cases))

    @property
    def available_ids(self) -> tuple[str, ...]:
        return self.val_ids

    def get_case(self, case_id: str) -> PreprocessedCase:
        self.calls.append(case_id)
        return self.cases[case_id]


def make_protocol(n_cases: int, n_positive: int, n_negative: int) -> ValidationProtocol:
    return ValidationProtocol(
        mode="full_volume",
        expected_cases=n_cases,
        expected_positive_cases=n_positive,
        expected_negative_cases=n_negative,
    )


# --------------------------------------------------------------------------- 覆盖
def test_all_ids_visited_exactly_once(tmp_path: Path):
    store = RecordingStore(
        {
            "c0": make_case("c0", gt_mask=box(), pred_mask=box()),
            "c1": make_case("c1", gt_mask=box(y=(5, 6))),
            "c2": make_case("c2"),
        }
    )
    validator = FullVolumeValidator(
        store, store.val_ids, patch_size=PATCH, protocol=make_protocol(3, 2, 1), metrics_dir=tmp_path, progress=False
    )
    validator.run(PointwiseLogits(), epoch=0)
    assert store.calls == list(store.val_ids)  # 顺序与数量完全一致：无随机抽样、无遗漏、无重复
    assert len(store.calls) == len(set(store.calls)) == 3


def test_duplicate_or_mismatched_ids_rejected(tmp_path: Path):
    store = RecordingStore({"c0": make_case("c0"), "c1": make_case("c1"), "c2": make_case("c2")})
    with pytest.raises(ValueError, match="重复"):
        FullVolumeValidator(store, ("c0", "c0", "c1"), patch_size=PATCH, protocol=make_protocol(3, 1, 2), metrics_dir=tmp_path)
    with pytest.raises(RuntimeError, match="病例数"):
        FullVolumeValidator(store, ("c0", "c1"), patch_size=PATCH, protocol=make_protocol(3, 1, 2), metrics_dir=tmp_path)


def test_positive_negative_count_mismatch_raises(tmp_path: Path):
    store = RecordingStore(
        {
            "c0": make_case("c0", gt_mask=box(), pred_mask=box()),
            "c1": make_case("c1", gt_mask=box(y=(5, 6))),  # 也是阳性
            "c2": make_case("c2"),
        }
    )
    # 协议声称只有 1 个阳性，实际 2 个 → 必须报错且不返回 best
    validator = FullVolumeValidator(
        store, store.val_ids, patch_size=PATCH, protocol=make_protocol(3, 1, 2), metrics_dir=tmp_path, progress=False
    )
    with pytest.raises(RuntimeError, match="阳性/阴性覆盖检查失败"):
        validator.run(PointwiseLogits(), epoch=0)


# --------------------------------------------------------------------------- Dice 语义
def test_positive_perfect_prediction_dice_is_one(tmp_path: Path):
    store = RecordingStore(
        {"c0": make_case("c0", gt_mask=box(), pred_mask=box()), "c1": make_case("c1"), "c2": make_case("c2")}
    )
    validator = FullVolumeValidator(
        store, store.val_ids, patch_size=PATCH, protocol=make_protocol(3, 1, 2), metrics_dir=tmp_path, progress=False
    )
    metrics = validator.run(PointwiseLogits(), epoch=0)
    assert metrics["val_positive_casewise_dice_mean"] == pytest.approx(1.0)
    assert metrics["val_positive_casewise_dice_median"] == pytest.approx(1.0)
    assert metrics["val_positive_cases"] == 1 and metrics["val_negative_cases"] == 2


def test_positive_empty_prediction_dice_is_zero(tmp_path: Path):
    store = RecordingStore(
        {"c0": make_case("c0", gt_mask=box()), "c1": make_case("c1"), "c2": make_case("c2")}
    )
    validator = FullVolumeValidator(
        store, store.val_ids, patch_size=PATCH, protocol=make_protocol(3, 1, 2), metrics_dir=tmp_path, progress=False
    )
    metrics = validator.run(PointwiseLogits(), epoch=0)
    assert metrics["val_positive_casewise_dice_mean"] == pytest.approx(0.0)
    assert metrics["val_micro_dice"] == pytest.approx(0.0)


def test_negative_empty_empty_not_counted_as_dice_one(tmp_path: Path):
    """阴性空—空不得进入主要 Dice 均值（否则会虚高）。"""
    store = RecordingStore(
        {
            "c0": make_case("c0", gt_mask=box(), pred_mask=box()),  # 阳性，Dice=1
            "c1": make_case("c1"),  # 阴性，空—空
            "c2": make_case("c2"),  # 阴性，空—空
        }
    )
    validator = FullVolumeValidator(
        store, store.val_ids, patch_size=PATCH, protocol=make_protocol(3, 1, 2), metrics_dir=tmp_path, progress=False
    )
    metrics = validator.run(PointwiseLogits(), epoch=0)
    assert metrics["val_positive_cases"] == 1
    assert metrics["val_positive_casewise_dice_mean"] == pytest.approx(1.0)  # 不是 (1+1+1)/3
    assert metrics["val_negative_fp_case_rate"] == pytest.approx(0.0)
    assert metrics["val_fp_voxels_per_negative_exam"] == pytest.approx(0.0)


def test_negative_with_false_positives_counted(tmp_path: Path):
    store = RecordingStore(
        {
            "c0": make_case("c0", gt_mask=box(), pred_mask=box()),
            "c1": make_case("c1", pred_mask=box(y=(1, 2))),  # 阴性但预测阳性 8 体素
            "c2": make_case("c2"),
        }
    )
    validator = FullVolumeValidator(
        store, store.val_ids, patch_size=PATCH, protocol=make_protocol(3, 1, 2), metrics_dir=tmp_path, progress=False
    )
    metrics = validator.run(PointwiseLogits(), epoch=0)
    assert metrics["val_negative_fp_case_rate"] == pytest.approx(0.5)  # 2 个阴性里 1 个有 FP
    assert metrics["val_fp_voxels_per_negative_exam"] == pytest.approx(1.0)  # 2 体素 / 2 个阴性例
    assert metrics["val_positive_casewise_dice_mean"] == pytest.approx(1.0)  # 阴性不进入主指标


def test_micro_dice_computation(tmp_path: Path):
    gt0, pred0 = box(), box()  # 8 体素全对
    gt1 = box(y=(5, 7), x=(5, 7))  # 8 体素 GT
    pred1 = box(y=(5, 6), x=(5, 7))  # 只预测中一半 → tp=4, fn=4
    store = RecordingStore(
        {
            "c0": make_case("c0", gt_mask=gt0, pred_mask=pred0),
            "c1": make_case("c1", gt_mask=gt1, pred_mask=pred1),
            "c2": make_case("c2"),
        }
    )
    validator = FullVolumeValidator(
        store, store.val_ids, patch_size=PATCH, protocol=make_protocol(3, 2, 1), metrics_dir=tmp_path, progress=False
    )
    metrics = validator.run(PointwiseLogits(), epoch=0)
    tp_total, fp_total, fn_total = 8 + 4, 0, 0 + 4
    expected_micro = 2 * tp_total / (2 * tp_total + fp_total + fn_total)
    assert metrics["val_micro_dice"] == pytest.approx(expected_micro)
    assert metrics["val_positive_casewise_dice_mean"] == pytest.approx((1.0 + 2 * 4 / (2 * 4 + 4)) / 2)


def test_case_result_properties():
    empty_empty = CaseResult("neg", gt_fg_voxels=0, pred_fg_voxels=0, tp=0, fp=0, fn=0)
    assert empty_empty.dice == 0.0  # 空—空不记为 1
    assert empty_empty.is_positive is False
    positive_missed = CaseResult("pos", gt_fg_voxels=10, pred_fg_voxels=0, tp=0, fp=0, fn=10)
    assert positive_missed.dice == 0.0
    assert positive_missed.is_positive is True


# --------------------------------------------------------------------------- 输出与状态
def test_case_metrics_csv_written(tmp_path: Path):
    store = RecordingStore(
        {"c0": make_case("c0", gt_mask=box(), pred_mask=box()), "c1": make_case("c1"), "c2": make_case("c2")}
    )
    validator = FullVolumeValidator(
        store, store.val_ids, patch_size=PATCH, protocol=make_protocol(3, 1, 2), metrics_dir=tmp_path, progress=False
    )
    metrics = validator.run(PointwiseLogits(), epoch=3)
    path = tmp_path / "validation_cases" / "epoch_0003.csv"
    assert path.is_file() and validator.last_case_csv_path == path
    rows = list(csv.DictReader(path.open()))
    assert [r["case_id"] for r in rows] == list(store.val_ids)  # 全部病例、恰好一次
    assert {"case_id", "is_positive", "tp", "fp", "fn", "dice"} <= set(rows[0].keys())
    assert metrics["validation_elapsed_sec"] >= 0.0


def test_no_case_csv_when_disabled(tmp_path: Path):
    store = RecordingStore({"c0": make_case("c0"), "c1": make_case("c1"), "c2": make_case("c2", gt_mask=box())})
    protocol = ValidationProtocol(
        mode="full_volume", expected_cases=3, expected_positive_cases=1, expected_negative_cases=2,
        save_case_metrics=False,
    )
    validator = FullVolumeValidator(store, store.val_ids, patch_size=PATCH, protocol=protocol, metrics_dir=tmp_path, progress=False)
    validator.run(PointwiseLogits(), epoch=0)
    assert validator.last_case_csv_path is None
    assert not (tmp_path / "validation_cases").exists()


def test_deep_supervision_and_train_state_restored(tmp_path: Path):
    store = RecordingStore(
        {"c0": make_case("c0", gt_mask=box(), pred_mask=box()), "c1": make_case("c1"), "c2": make_case("c2")}
    )
    model = PointwiseLogits(deep_supervision=True)
    model.train()
    validator = FullVolumeValidator(
        store, store.val_ids, patch_size=PATCH, protocol=make_protocol(3, 1, 2), metrics_dir=tmp_path, progress=False
    )
    validator.run(model, epoch=0)
    assert model.deep_supervision is True  # DS 状态已恢复
    assert model.training is True  # train/eval 状态已恢复
    assert model.observed_ds and not any(model.observed_ds)  # 验证期间 DS 一直是关闭的
    assert model.observed_training and not any(model.observed_training)  # 验证期间为 eval


def test_nonfinite_prediction_raises(tmp_path: Path):
    store = RecordingStore({"c0": make_case("c0"), "c1": make_case("c1"), "c2": make_case("c2", gt_mask=box())})
    validator = FullVolumeValidator(
        store, store.val_ids, patch_size=PATCH, protocol=make_protocol(3, 1, 2), metrics_dir=tmp_path, progress=False
    )
    with pytest.raises(RuntimeError, match="NaN/Inf"):
        validator.run(PointwiseLogits(nonfinite=True), epoch=0)


def test_progress_flag_controls_tqdm(tmp_path: Path, capsys):
    store = RecordingStore(
        {"c0": make_case("c0", gt_mask=box(), pred_mask=box()), "c1": make_case("c1"), "c2": make_case("c2")}
    )
    protocol = make_protocol(3, 1, 2)
    silent = FullVolumeValidator(
        store, store.val_ids, patch_size=PATCH, protocol=protocol, metrics_dir=tmp_path, progress=False
    )
    silent.run(PointwiseLogits(), epoch=0)
    out = capsys.readouterr()
    assert out.out == "" and out.err == ""

    shown = FullVolumeValidator(
        store, store.val_ids, patch_size=PATCH, protocol=protocol, metrics_dir=tmp_path, progress=True
    )
    shown.run(PointwiseLogits(), epoch=0)
    captured = capsys.readouterr()
    assert "val full-volume" in (captured.err + captured.out)


def test_no_progress_disables_inner_patch_progress_too(tmp_path: Path, capsys):
    """内层 patch 进度开关打开时，progress=False 仍必须关闭全部进度输出。"""
    store = RecordingStore(
        {"c0": make_case("c0", gt_mask=box(), pred_mask=box()), "c1": make_case("c1"), "c2": make_case("c2")}
    )
    protocol = ValidationProtocol(
        mode="full_volume", expected_cases=3, expected_positive_cases=1, expected_negative_cases=2,
        inner_patch_progress=True,
    )
    validator = FullVolumeValidator(
        store, store.val_ids, patch_size=PATCH, protocol=protocol, metrics_dir=tmp_path, progress=False
    )
    validator.run(PointwiseLogits(), epoch=0)
    out = capsys.readouterr()
    assert out.out == "" and out.err == ""


def test_protocol_validation_rules():
    with pytest.raises(ValueError, match="mode"):
        ValidationProtocol(mode="nope").validate()
    with pytest.raises(ValueError, match="expected_positive_cases"):
        ValidationProtocol(expected_cases=10, expected_positive_cases=3, expected_negative_cases=3).validate()
    with pytest.raises(ValueError, match="step_fraction"):
        ValidationProtocol(step_fraction=1.5).validate()
    # v2.2：every_n_epochs >= 1 合法（5=正式低频，1=每 epoch）；0/负数非法
    ValidationProtocol(every_n_epochs=5).validate()
    ValidationProtocol(every_n_epochs=1).validate()
    with pytest.raises(ValueError, match="every_n_epochs"):
        ValidationProtocol(every_n_epochs=0).validate()
    with pytest.raises(ValueError, match="every_n_epochs"):
        ValidationProtocol(every_n_epochs=-1).validate()
    with pytest.raises(ValueError, match="val_loss"):
        ValidationProtocol(checkpoint_metric="val_loss").validate()
    # 指标与方向硬冻结：不允许换成其他记录字段或最小化方向
    with pytest.raises(ValueError, match="已冻结"):
        ValidationProtocol(checkpoint_metric="val_micro_dice").validate()
    with pytest.raises(ValueError, match="最大化"):
        ValidationProtocol(maximize=False).validate()
    ValidationProtocol().validate()  # 默认协议合法（223/63/160）
