"""M0 训练循环测试（**纯合成随机张量、CPU、不读取任何真实数据**）。

说明：这里的 2 epoch x 2 iteration 只是训练循环的单元测试（验证日志/checkpoint/续训/进度
与异常路径），不是 small-overfit 验收，也不代表 M0 有效。
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch

from zonal_reliability_fusion.models import M0ConcatModel
from zonal_reliability_fusion.training import (
    DeepSupervisionLoss,
    DiceCELoss,
    EarlyStoppingConfig,
    M0Trainer,
    deep_supervision_weights,
    load_checkpoint,
)
from zonal_reliability_fusion.training.optim import PolyLRScheduler


def synthetic_source(shape, *, batch: int = 2, n_batches: int = 2, seed: int = 0, with_fg: bool = True):
    """返回 batch_source：每次调用生成新的随机张量（可直接喂给 trainer）。"""

    def source():
        generator = torch.Generator().manual_seed(seed)
        for _ in range(n_batches):
            data = torch.randn(batch, 3, *shape, generator=generator)
            seg = torch.zeros(batch, 1, *shape)
            if with_fg:
                seg[:, :, 2:5, 4:9, 4:9] = 1
            yield data, seg

    return source


def build_trainer(tmp_path: Path, mini_plan, *, epochs: int = 2, amp: bool = False, progress: bool = False, train_n: int = 2):
    model = M0ConcatModel(mini_plan, deep_supervision=True)
    weights = deep_supervision_weights(len(mini_plan.decoder_output_shapes()))
    loss_fn = DeepSupervisionLoss(DiceCELoss(), weights=weights)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.99, nesterov=True, weight_decay=3e-5)
    scheduler = PolyLRScheduler(optimizer, initial_lr=0.01, max_steps=epochs, power=0.9)
    return M0Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device="cpu",
        train_batches=synthetic_source(mini_plan.patch_size, n_batches=train_n, seed=0),
        val_batches=synthetic_source(mini_plan.patch_size, n_batches=2, seed=1),
        run_dir=tmp_path / "run",
        epochs=epochs,
        iterations_per_epoch=train_n,
        validation_iterations=2,
        progress=progress,
        checkpoint_every=1,
        amp=amp,
        seed=0,
        config_snapshot={"unit_test": True},
    )


def test_trainer_runs_epochs_and_writes_logs_and_checkpoints(tmp_path: Path, mini_plan):
    trainer = build_trainer(tmp_path, mini_plan)
    summary = trainer.train()
    assert summary.status == "completed"
    assert summary.epochs_done == 2
    assert summary.best_metric is not None

    csv_path = tmp_path / "run" / "metrics" / "epoch_metrics.csv"
    assert csv_path.is_file()
    rows = list(csv.DictReader(csv_path.open()))
    assert len(rows) == 2
    assert float(rows[0]["train_loss"]) >= 0.0
    assert float(rows[0]["lr"]) == pytest.approx(0.01)
    assert float(rows[1]["lr"]) == pytest.approx(0.01 * (1 - 1 / 2) ** 0.9)
    assert (tmp_path / "run" / "checkpoints" / "checkpoint_last.pth").is_file()
    assert (tmp_path / "run" / "checkpoints" / "checkpoint_best.pth").is_file()
    assert (tmp_path / "run" / "diagnostics" / "run_manifest.json").is_file()


def test_trainer_resume_continues_epoch_and_lr(tmp_path: Path, mini_plan):
    trainer = build_trainer(tmp_path, mini_plan, epochs=2)
    trainer.train()
    resumed = build_trainer(tmp_path, mini_plan, epochs=3)
    resumed.resume(tmp_path / "run" / "checkpoints" / "checkpoint_last.pth")
    assert resumed.start_epoch == 2
    summary = resumed.train()
    assert summary.status == "completed"
    assert summary.epochs_done == 3
    rows = list(csv.DictReader((tmp_path / "run" / "metrics" / "epoch_metrics.csv").open()))
    assert len(rows) == 3  # resume 后追加而不是覆盖
    assert float(rows[-1]["lr"]) == pytest.approx(0.01 * (1 - 2 / 3) ** 0.9)


def test_trainer_raises_when_source_exhausted(tmp_path: Path, mini_plan):
    trainer = build_trainer(tmp_path, mini_plan, epochs=1, train_n=1)
    trainer.iterations_per_epoch = 3  # 声明 3 个 batch 但数据源只有 1 个
    with pytest.raises(RuntimeError, match="提前耗尽"):
        trainer.train()


def test_trainer_warns_when_amp_requested_on_cpu(tmp_path: Path, mini_plan, capsys):
    trainer = build_trainer(tmp_path, mini_plan, epochs=1, amp=True)
    assert trainer.amp is False
    trainer.train()
    captured = capsys.readouterr()
    assert "AMP" in captured.out and "WARN" in captured.out


def test_trainer_moves_model_to_device_and_syncs_data_epoch(tmp_path: Path, mini_plan):
    """模型必须与输入同 device；数据源每轮收到 set_epoch(epoch)。"""
    epochs_seen: list[int] = []

    class FakeSource:
        def set_epoch(self, epoch: int) -> None:
            epochs_seen.append(int(epoch))

    trainer = build_trainer(tmp_path, mini_plan, epochs=2)
    trainer.data_sources = (FakeSource(),)
    trainer.train()
    assert epochs_seen == [0, 1]

    # 模型参数必须已在 trainer.device 上（CPU 场景为 cpu；CUDA 场景由同一调用迁移）
    assert next(trainer.model.parameters()).device.type == trainer.device.type

    with pytest.raises(TypeError):
        M0Trainer(
            model=trainer.model,
            loss_fn=trainer.loss_fn,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
            device="cpu",
            train_batches=trainer.train_batches,
            val_batches=trainer.val_batches,
            run_dir=tmp_path / "bad_sources",
            epochs=1,
            iterations_per_epoch=1,
            validation_iterations=1,
            progress=False,
            data_sources=[object()],  # 缺 set_epoch
        )


def test_trainer_manifest_records_model_device(tmp_path: Path, mini_plan):
    trainer = build_trainer(tmp_path, mini_plan, epochs=1)
    trainer.train()
    manifest = json.loads((tmp_path / "run" / "diagnostics" / "run_manifest.json").read_text())
    assert manifest["device"] == "cpu"
    assert manifest["model_parameter_device"] == "cpu"


def test_trainer_rejects_invalid_arguments(tmp_path: Path, mini_plan):
    trainer = build_trainer(tmp_path, mini_plan)
    with pytest.raises(ValueError):
        M0Trainer(
            model=trainer.model,
            loss_fn=trainer.loss_fn,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
            device="cpu",
            train_batches=trainer.train_batches,
            val_batches=trainer.val_batches,
            run_dir=tmp_path / "bad",
            epochs=0,
            iterations_per_epoch=1,
            validation_iterations=1,
            progress=False,
        )


# =========================================================== P2-A3：正式全体积验证模式


class ScriptedValidator:
    """假验证器：按 epoch 返回预设 Dice（只测训练循环逻辑，不做真实推理）。"""

    def __init__(self, metrics: dict[int, float], metric: str = "val_positive_casewise_dice_mean") -> None:
        self.metrics = dict(metrics)
        self.metric = metric
        self.calls: list[int] = []

    def run(self, model, *, epoch: int) -> dict[str, float]:
        self.calls.append(int(epoch))
        if epoch not in self.metrics:
            raise AssertionError(f"epoch {epoch} 没有预设指标（测试应当在此停止）")
        return {
            self.metric: float(self.metrics[epoch]),
            "val_positive_casewise_dice_median": float(self.metrics[epoch]),
            "val_micro_dice": float(self.metrics[epoch]) * 0.9,
            "val_positive_cases": 63.0,
            "val_negative_cases": 160.0,
            "val_cases_evaluated": 223.0,
            "val_negative_fp_case_rate": 0.1,
            "val_fp_voxels_per_negative_exam": 5.0,
            "validation_elapsed_sec": 0.01,
        }


def build_formal_trainer(
    tmp_path: Path,
    mini_plan,
    *,
    dice_metrics: dict[int, float],
    epochs: int,
    early_stopping,
    min_delta: float,
    validator: ScriptedValidator | None = None,
    run_name: str = "run_formal",
    train_source=None,
    seed: int | None = None,
    config_snapshot: dict | None = None,
    validation_every_n_epochs: int = 1,
):
    model = M0ConcatModel(mini_plan, deep_supervision=True)
    weights = deep_supervision_weights(len(mini_plan.decoder_output_shapes()))
    loss_fn = DeepSupervisionLoss(DiceCELoss(), weights=weights)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.99, nesterov=True, weight_decay=3e-5)
    scheduler = PolyLRScheduler(optimizer, initial_lr=0.01, max_steps=epochs, power=0.9)
    validator = validator or ScriptedValidator(dice_metrics)
    trainer = M0Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device="cpu",
        train_batches=train_source or synthetic_source(mini_plan.patch_size, n_batches=2, seed=0),
        val_batches=None,  # 正式模式不得创建随机 validation patch provider
        run_dir=tmp_path / run_name,
        epochs=epochs,
        iterations_per_epoch=2,
        validation_iterations=1,
        validation_mode="formal_full_volume",
        validator=validator,
        validation_every_n_epochs=validation_every_n_epochs,
        checkpoint_metric="val_positive_casewise_dice_mean",
        maximize=True,
        min_delta=min_delta,
        early_stopping=early_stopping,
        progress=False,
        checkpoint_every=1,
        config_snapshot=config_snapshot if config_snapshot is not None else {"validation": {"mode": "full_volume"}},
        seed=seed,
    )
    return trainer, validator


def test_formal_mode_uses_full_volume_dice_metric_and_saves_best(tmp_path: Path, mini_plan):
    trainer, validator = build_formal_trainer(
        tmp_path, mini_plan,
        dice_metrics={0: 0.10, 1: 0.20, 2: 0.15}, epochs=3,
        early_stopping=EarlyStoppingConfig(enabled=False, min_epochs=1, patience=10, min_delta=1e-4),
        min_delta=1e-4,
    )
    summary = trainer.train()
    assert summary.status == "completed" and summary.epochs_done == 3
    assert summary.checkpoint_metric == "val_positive_casewise_dice_mean"
    assert (summary.best_metric, summary.best_epoch) == (0.20, 1)
    assert validator.calls == [0, 1, 2]  # 每个 epoch 恰好验证一次
    assert (tmp_path / "run_formal" / "checkpoints" / "checkpoint_best.pth").is_file()
    rows = list(csv.DictReader((tmp_path / "run_formal" / "metrics" / "epoch_metrics.csv").open()))
    assert "val_positive_casewise_dice_mean" in rows[0]
    assert "val_cases_evaluated" in rows[0]
    assert "val_loss" not in rows[0]  # 正式模式不计算 patch validation loss
    assert [int(r["is_best"]) for r in rows] == [1, 1, 0]


def test_best_not_replaced_when_improvement_below_min_delta(tmp_path: Path, mini_plan):
    trainer, _ = build_formal_trainer(
        tmp_path, mini_plan,
        dice_metrics={0: 0.500, 1: 0.50005}, epochs=2,
        early_stopping=EarlyStoppingConfig(enabled=False, min_epochs=1, patience=10, min_delta=1e-4),
        min_delta=1e-4,
    )
    summary = trainer.train()
    assert summary.best_metric == pytest.approx(0.500)  # 改善小于 min_delta → 不更新 best
    assert summary.best_epoch == 0
    assert summary.bad_validation_events == 1
    rows = list(csv.DictReader((tmp_path / "run_formal" / "metrics" / "epoch_metrics.csv").open()))
    assert [int(r["is_best"]) for r in rows] == [1, 0]


def test_early_stopping_never_before_min_epochs(tmp_path: Path, mini_plan):
    trainer, validator = build_formal_trainer(
        tmp_path, mini_plan,
        dice_metrics={0: 0.5, 1: 0.4, 2: 0.3, 3: 0.2}, epochs=10,
        early_stopping=EarlyStoppingConfig(enabled=True, min_epochs=3, patience=1, min_delta=1e-4),
        min_delta=1e-4,
    )
    summary = trainer.train()
    assert summary.epochs_done == 3  # 不早于 min_epochs
    assert summary.early_stopped and summary.status == "early_stopped"
    assert validator.calls == [0, 1, 2]
    assert (tmp_path / "run_formal" / "diagnostics" / "early_stop_summary.json").is_file()


def test_early_stopping_stops_after_patience(tmp_path: Path, mini_plan):
    trainer, validator = build_formal_trainer(
        tmp_path, mini_plan,
        dice_metrics={0: 0.5, 1: 0.4, 2: 0.3}, epochs=10,
        early_stopping=EarlyStoppingConfig(enabled=True, min_epochs=1, patience=2, min_delta=1e-4),
        min_delta=1e-4,
    )
    summary = trainer.train()
    assert summary.epochs_done == 3  # 连续 2 个 validation event 未改善（epoch 1、2）后停止
    assert summary.bad_validation_events == 2
    assert validator.calls == [0, 1, 2]
    payload = json.loads((tmp_path / "run_formal" / "diagnostics" / "early_stop_summary.json").read_text())
    assert payload["early_stopped"] is True and payload["patience"] == 2 and payload["bad_validation_events"] == 2


def test_resume_restores_early_stopping_state_and_best(tmp_path: Path, mini_plan):
    """协议完全一致时续训：bad_validation_events / best / epoch / LR 连续。"""
    protocol_stop = EarlyStoppingConfig(enabled=True, min_epochs=1, patience=2, min_delta=1e-4)
    first, _ = build_formal_trainer(
        tmp_path, mini_plan,
        dice_metrics={0: 0.5, 1: 0.4}, epochs=2,
        early_stopping=protocol_stop, min_delta=1e-4,
        run_name="run_resume",
    )
    summary_a = first.train()
    assert summary_a.bad_validation_events == 1 and summary_a.best_metric == pytest.approx(0.5)
    assert not summary_a.early_stopped

    second, validator = build_formal_trainer(
        tmp_path, mini_plan,
        dice_metrics={0: 0.5, 1: 0.4, 2: 0.3, 3: 0.2}, epochs=5,
        early_stopping=protocol_stop, min_delta=1e-4,
        run_name="run_resume",
    )
    second.resume(tmp_path / "run_resume" / "checkpoints" / "checkpoint_last.pth")
    assert second.bad_validation_events == 1  # early-stopping counter 已恢复
    assert second.best_metric == pytest.approx(0.5) and second.best_epoch == 0
    summary_b = second.train()
    assert validator.calls == [2]  # 从 epoch 2 继续
    assert summary_b.epochs_done == 3 and summary_b.bad_validation_events == 2
    assert summary_b.early_stopped


def test_resume_rejects_protocol_change(tmp_path: Path, mini_plan):
    """改变 early stopping / 指标方向等训练协议后，默认拒绝续训。"""
    first, _ = build_formal_trainer(
        tmp_path, mini_plan,
        dice_metrics={0: 0.5, 1: 0.4}, epochs=2,
        early_stopping=EarlyStoppingConfig(enabled=True, min_epochs=1, patience=2, min_delta=1e-4),
        min_delta=1e-4,
        run_name="run_proto",
    )
    first.train()

    changed, _ = build_formal_trainer(
        tmp_path, mini_plan,
        dice_metrics={0: 0.5, 1: 0.4, 2: 0.3}, epochs=3,
        early_stopping=EarlyStoppingConfig(enabled=True, min_epochs=1, patience=5, min_delta=1e-4),  # patience 变了
        min_delta=1e-4,
        run_name="run_proto",
    )
    with pytest.raises(RuntimeError, match="协议不一致"):
        changed.resume(tmp_path / "run_proto" / "checkpoints" / "checkpoint_last.pth")

    allowed, _ = build_formal_trainer(
        tmp_path, mini_plan,
        dice_metrics={0: 0.5, 1: 0.4, 2: 0.3}, epochs=3,
        early_stopping=EarlyStoppingConfig(enabled=True, min_epochs=1, patience=5, min_delta=1e-4),
        min_delta=1e-4,
        run_name="run_proto",
    )
    allowed, validator = build_formal_trainer(
        tmp_path, mini_plan,
        dice_metrics={2: 0.3}, epochs=3,
        early_stopping=EarlyStoppingConfig(enabled=True, min_epochs=1, patience=5, min_delta=1e-4),
        min_delta=1e-4,
        run_name="run_proto",
    )
    allowed.resume(
        tmp_path / "run_proto" / "checkpoints" / "checkpoint_last.pth", allow_protocol_change=True
    )
    assert allowed.start_epoch == 2
    # 协议变化 → 不继承旧 best / early-stopping 状态（旧指标在新协议下不可比）
    assert allowed.best_metric is None and allowed.best_epoch is None and allowed.bad_validation_events == 0
    summary = allowed.train()
    assert validator.calls == [2]
    assert summary.best_epoch == 2  # 选模从续训点重新开始


def test_resume_does_not_reseed_rng(tmp_path: Path, mini_plan, monkeypatch):
    """问题 3 回归：resume 恢复 RNG 后，train() 不得再用 set_seed 覆盖。"""
    import zonal_reliability_fusion.training.trainer as trainer_module

    calls: list[int] = []
    monkeypatch.setattr(trainer_module, "set_seed", lambda seed, **kw: calls.append(int(seed)))

    fresh, _ = build_formal_trainer(
        tmp_path, mini_plan,
        dice_metrics={0: 0.3, 1: 0.4}, epochs=2,
        early_stopping=EarlyStoppingConfig(enabled=False, min_epochs=1, patience=1, min_delta=1e-4),
        min_delta=1e-4,
        run_name="run_rng",
        seed=0,
    )
    fresh.train()
    assert calls == [0]  # 全新 run 会 set_seed 一次

    calls.clear()
    resumed, _ = build_formal_trainer(
        tmp_path, mini_plan,
        dice_metrics={0: 0.3, 1: 0.4}, epochs=2,
        early_stopping=EarlyStoppingConfig(enabled=False, min_epochs=1, patience=1, min_delta=1e-4),
        min_delta=1e-4,
        run_name="run_rng",
        seed=0,
    )
    resumed.resume(tmp_path / "run_rng" / "checkpoints" / "checkpoint_last.pth")
    assert resumed._rng_restored is True
    resumed.train()
    assert calls == []  # 续训不再 set_seed（RNG 来自 checkpoint）


def test_resume_rejects_run_config_change_and_allows_max_epochs_extension(tmp_path: Path, mini_plan):
    """问题 2 回归：续训必须冻结完整训练配置；仅 max_epochs 允许显式延长。"""
    protocol_stop = EarlyStoppingConfig(enabled=False, min_epochs=1, patience=1, min_delta=1e-4)
    base_snapshot = {
        "validation": {"mode": "full_volume"},
        "data": {"patch_size": [16, 32, 32], "modality_order": ["T2W", "ADC", "HBV"]},
        "optimizer": {"lr": 0.01, "momentum": 0.99},
        "training": {"max_epochs": 2, "iterations_per_epoch": 2},
        "provenance": {"plans_sha256": "aaa", "splits_sha256": "bbb"},
    }
    run_name = "run_cfg"
    first, _ = build_formal_trainer(
        tmp_path, mini_plan,
        dice_metrics={0: 0.3, 1: 0.4}, epochs=2,
        early_stopping=protocol_stop, min_delta=1e-4, run_name=run_name,
        config_snapshot=base_snapshot,
    )
    first.train()

    # (a) 冻结项变化（optimizer.lr）→ 默认拒绝
    changed = {**base_snapshot, "optimizer": {"lr": 0.02, "momentum": 0.99}}
    trainer_lr, _ = build_formal_trainer(
        tmp_path, mini_plan,
        dice_metrics={2: 0.5}, epochs=3,
        early_stopping=protocol_stop, min_delta=1e-4, run_name=run_name,
        config_snapshot=changed,
    )
    with pytest.raises(RuntimeError, match="训练配置不一致"):
        trainer_lr.resume(tmp_path / run_name / "checkpoints" / "checkpoint_last.pth")

    # (b) 冻结项变化（数据哈希）→ 同样拒绝
    changed_hash = {**base_snapshot, "provenance": {"plans_sha256": "aaa", "splits_sha256": "CHANGED"}}
    trainer_hash, _ = build_formal_trainer(
        tmp_path, mini_plan,
        dice_metrics={2: 0.5}, epochs=3,
        early_stopping=protocol_stop, min_delta=1e-4, run_name=run_name,
        config_snapshot=changed_hash,
    )
    with pytest.raises(RuntimeError, match="训练配置不一致"):
        trainer_hash.resume(tmp_path / run_name / "checkpoints" / "checkpoint_last.pth")

    # (c) 非冻结项变化（num_workers/pin_memory）→ 放行（仅打印记录）
    relaxed = {
        **base_snapshot,
        "train_pipeline": {"num_workers": 8, "pin_memory": True},
    }
    trainer_relaxed, _ = build_formal_trainer(
        tmp_path, mini_plan,
        dice_metrics={2: 0.5}, epochs=3,
        early_stopping=protocol_stop, min_delta=1e-4, run_name=run_name,
        config_snapshot=relaxed,
    )
    trainer_relaxed.resume(tmp_path / run_name / "checkpoints" / "checkpoint_last.pth")
    assert trainer_relaxed.start_epoch == 2

    # (d) max_epochs 是唯一允许显式延长的训练项 → 放行
    extended = {
        **base_snapshot,
        "training": {"max_epochs": 6, "iterations_per_epoch": 2},
    }
    trainer_ext, _ = build_formal_trainer(
        tmp_path, mini_plan,
        dice_metrics={2: 0.5}, epochs=6,
        early_stopping=protocol_stop, min_delta=1e-4, run_name=run_name,
        config_snapshot=extended,
    )
    trainer_ext.resume(tmp_path / run_name / "checkpoints" / "checkpoint_last.pth")
    assert trainer_ext.start_epoch == 2 and trainer_ext.epochs == 6


def test_resume_rejects_stale_checkpoint_log_mismatch(tmp_path: Path, mini_plan):
    """问题 3 回归：从较旧 checkpoint 恢复到同一 run 会被拒绝（日志不对齐）。"""
    protocol_stop = EarlyStoppingConfig(enabled=False, min_epochs=1, patience=1, min_delta=1e-4)
    run_name = "run_align"
    ckpt_dir = tmp_path / run_name / "checkpoints"
    first, _ = build_formal_trainer(
        tmp_path, mini_plan,
        dice_metrics={0: 0.3, 1: 0.4, 2: 0.5}, epochs=3,
        early_stopping=protocol_stop, min_delta=1e-4, run_name=run_name,
    )
    first.train()
    assert first.last_logged_epoch() == 2

    stale, _ = build_formal_trainer(
        tmp_path, mini_plan,
        dice_metrics={1: 0.4}, epochs=3,
        early_stopping=protocol_stop, min_delta=1e-4, run_name=run_name,
    )
    with pytest.raises(RuntimeError, match="epoch_metrics.csv"):
        stale.resume(ckpt_dir / "checkpoint_epoch_0000.pth")  # 期望从 epoch 1 开始，日志已到 2

    aligned, _ = build_formal_trainer(
        tmp_path, mini_plan,
        dice_metrics={3: 0.6}, epochs=4,
        early_stopping=protocol_stop, min_delta=1e-4, run_name=run_name,
    )
    aligned.resume(ckpt_dir / "checkpoint_last.pth")  # epoch 2 → 从 3 继续，与日志对齐
    assert aligned.start_epoch == 3


def _make_interrupting_source(mini_plan, *, interrupt_at_call: int, iterations: int = 2):
    """构造一个训练数据源：在第 `interrupt_at_call` 次被调用（即该进程内的第 N 个 epoch）的第 2 个 iteration 打断。"""

    state = {"calls": 0}

    def source():
        state["calls"] += 1
        call = state["calls"]

        def generator():
            for i in range(iterations):
                if call == interrupt_at_call and i == 1:
                    raise KeyboardInterrupt
                g = torch.Generator().manual_seed(i)
                data = torch.randn(2, 3, *mini_plan.patch_size, generator=g)
                seg = torch.zeros(2, 1, *mini_plan.patch_size)
                seg[:, :, 2:5, 4:9, 4:9] = 1
                yield data, seg

        return generator()

    return source


def test_interrupt_after_resume_records_absolute_completed_epochs(tmp_path: Path, mini_plan):
    """问题 1 回归：resume 之后再次中断，completed_epochs 必须是绝对计数（含 resume 前完成的 epoch）。"""
    protocol = EarlyStoppingConfig(enabled=False, min_epochs=1, patience=1, min_delta=1e-4)
    run_name = "run_resume_interrupt"
    ckpt_dir = tmp_path / run_name / "checkpoints"

    # 第一次运行：epochs=6，在第 5 个 epoch（索引 4）中途被打断 → 此前完整完成 4 个 epoch
    first, _ = build_formal_trainer(
        tmp_path, mini_plan,
        dice_metrics={0: 0.30, 1: 0.31, 2: 0.32, 3: 0.33}, epochs=6,
        early_stopping=protocol, min_delta=1e-4, run_name=run_name,
        train_source=_make_interrupting_source(mini_plan, interrupt_at_call=5),
    )
    with pytest.raises(KeyboardInterrupt):
        first.train()
    meta1 = load_checkpoint(ckpt_dir / "checkpoint_interrupt.pth")
    assert meta1["epoch"] == 4 and meta1["completed_epochs"] == 4 and meta1["iterations_completed"] == 1

    # 从 last（epoch 3 完整结束）续训，在本次进程的第 2 个 epoch（绝对索引 5）再次被打断
    second, _ = build_formal_trainer(
        tmp_path, mini_plan,
        dice_metrics={4: 0.34}, epochs=6,
        early_stopping=protocol, min_delta=1e-4, run_name=run_name,
        train_source=_make_interrupting_source(mini_plan, interrupt_at_call=2),
    )
    second.resume(ckpt_dir / "checkpoint_last.pth")
    assert second.start_epoch == 4 and second._epoch_offset == 4
    with pytest.raises(KeyboardInterrupt):
        second.train()
    meta2 = load_checkpoint(ckpt_dir / "checkpoint_interrupt.pth")
    assert meta2["epoch"] == 5
    assert meta2["completed_epochs"] == 5  # 4（resume 前）+ 1（本次完整完成）——修复前会错记为 1
    assert meta2["iterations_completed"] == 1

    # 显式允许从中断处续训：start_epoch 必须等于绝对 completed_epochs
    third, _ = build_formal_trainer(
        tmp_path, mini_plan,
        dice_metrics={5: 0.35}, epochs=6,
        early_stopping=protocol, min_delta=1e-4, run_name=run_name,
    )
    third.resume(ckpt_dir / "checkpoint_interrupt.pth", allow_mid_epoch=True)
    assert third.start_epoch == 5 and third._epoch_offset == 5


def test_interrupt_checkpoint_records_mid_epoch_semantics(tmp_path: Path, mini_plan):
    """问题 1 回归：中断 checkpoint 必须标记 epoch_completed=False，并记录已完成 epoch/iteration。"""
    state = {"epoch_calls": 0}

    def interrupting_source():
        state["epoch_calls"] += 1
        epoch_calls = state["epoch_calls"]

        def generator():
            for i in range(2):
                if epoch_calls == 2 and i == 1:  # 第二个 epoch 的第 2 个 iteration 被打断
                    raise KeyboardInterrupt
                generator_seed = torch.Generator().manual_seed(i)
                data = torch.randn(2, 3, *mini_plan.patch_size, generator=generator_seed)
                seg = torch.zeros(2, 1, *mini_plan.patch_size)
                seg[:, :, 2:5, 4:9, 4:9] = 1
                yield data, seg

        return generator()

    trainer, _ = build_formal_trainer(
        tmp_path, mini_plan,
        dice_metrics={0: 0.3, 1: 0.4, 2: 0.5}, epochs=3,
        early_stopping=EarlyStoppingConfig(enabled=False, min_epochs=1, patience=1, min_delta=1e-4),
        min_delta=1e-4,
        run_name="run_interrupt",
        train_source=interrupting_source,
    )
    with pytest.raises(KeyboardInterrupt):
        trainer.train()

    interrupt_meta = load_checkpoint(tmp_path / "run_interrupt" / "checkpoints" / "checkpoint_interrupt.pth")
    assert interrupt_meta["epoch_completed"] is False
    assert interrupt_meta["epoch"] == 1  # 被打断的 epoch 索引
    assert interrupt_meta["completed_epochs"] == 1  # epoch 0 已完整完成
    assert interrupt_meta["iterations_completed"] == 1
    assert interrupt_meta["extra"]["interrupted"] is True

    last_meta = load_checkpoint(tmp_path / "run_interrupt" / "checkpoints" / "checkpoint_last.pth")
    assert last_meta["epoch"] == 0 and last_meta["epoch_completed"] is True  # last 未被中断状态覆盖

    # 默认拒绝从中断 checkpoint 续训，并提示改用 checkpoint_last
    with pytest.raises(RuntimeError, match="中途"):
        trainer.resume(tmp_path / "run_interrupt" / "checkpoints" / "checkpoint_interrupt.pth")

    # 从 last（最后一个完整 epoch）续训：权重与 epoch 0 末尾对齐，从 epoch 1 开始
    from_last, _ = build_formal_trainer(
        tmp_path, mini_plan,
        dice_metrics={1: 0.4, 2: 0.5}, epochs=3,
        early_stopping=EarlyStoppingConfig(enabled=False, min_epochs=1, patience=1, min_delta=1e-4),
        min_delta=1e-4,
        run_name="run_interrupt",
    )
    from_last.resume(tmp_path / "run_interrupt" / "checkpoints" / "checkpoint_last.pth")
    assert from_last.start_epoch == 1

    # 显式允许 → 重跑被打断的 epoch（start_epoch == completed_epochs），可正常跑完
    resumed, validator = build_formal_trainer(
        tmp_path, mini_plan,
        dice_metrics={1: 0.4, 2: 0.5}, epochs=3,
        early_stopping=EarlyStoppingConfig(enabled=False, min_epochs=1, patience=1, min_delta=1e-4),
        min_delta=1e-4,
        run_name="run_interrupt",
    )
    resumed.resume(
        tmp_path / "run_interrupt" / "checkpoints" / "checkpoint_interrupt.pth", allow_mid_epoch=True
    )
    assert resumed.start_epoch == 1 and resumed.best_metric == pytest.approx(0.3)
    summary = resumed.train()
    assert summary.epochs_done == 3 and validator.calls == [1, 2]


def test_missing_or_nonfinite_checkpoint_metric_raises(tmp_path: Path, mini_plan):
    bad_field, _ = build_formal_trainer(
        tmp_path, mini_plan,
        dice_metrics={0: 0.1}, epochs=1,
        early_stopping=EarlyStoppingConfig(enabled=False, min_epochs=1, patience=1, min_delta=0.0),
        min_delta=0.0,
        validator=ScriptedValidator({0: 0.1}, metric="some_other_metric"),
        run_name="run_missing",
    )
    with pytest.raises(KeyError, match="checkpoint 指标"):
        bad_field.train()

    nan_metric, _ = build_formal_trainer(
        tmp_path, mini_plan,
        dice_metrics={0: float("nan")}, epochs=1,
        early_stopping=EarlyStoppingConfig(enabled=False, min_epochs=1, patience=1, min_delta=0.0),
        min_delta=0.0,
        run_name="run_nan",
    )
    with pytest.raises(RuntimeError, match="非有限值"):
        nan_metric.train()


def test_validation_mode_argument_checks(tmp_path: Path, mini_plan):
    model = M0ConcatModel(mini_plan, deep_supervision=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    common = dict(
        model=model,
        loss_fn=DeepSupervisionLoss(DiceCELoss(), weights=deep_supervision_weights(len(mini_plan.decoder_output_shapes()))),
        optimizer=optimizer,
        scheduler=None,
        device="cpu",
        train_batches=synthetic_source(mini_plan.patch_size, n_batches=1, seed=0),
        run_dir=tmp_path / "argcheck",
        epochs=1,
        iterations_per_epoch=1,
        validation_iterations=1,
        progress=False,
    )
    with pytest.raises(ValueError, match="validator"):
        M0Trainer(**common, validation_mode="formal_full_volume")  # 缺 validator
    with pytest.raises(ValueError, match="val_batches"):
        M0Trainer(**common, validation_mode="diagnostic_patch")  # 诊断模式缺 val_batches
    with pytest.raises(ValueError, match="不接受 val_batches"):
        M0Trainer(
            **common,
            validation_mode="formal_full_volume",
            validator=ScriptedValidator({0: 0.1}),
            val_batches=synthetic_source(mini_plan.patch_size, n_batches=1, seed=1),
        )
    with pytest.raises(ValueError, match="val_loss"):
        M0Trainer(
            **common,
            validation_mode="formal_full_volume",
            validator=ScriptedValidator({0: 0.1}),
            checkpoint_metric="val_loss",
        )
    with pytest.raises(ValueError, match="min_delta"):
        M0Trainer(
            **common,
            validation_mode="formal_full_volume",
            validator=ScriptedValidator({0: 0.1}),
            checkpoint_metric="val_positive_casewise_dice_mean",
            maximize=True,
            min_delta=1e-3,  # 与 early_stopping.min_delta 不一致 → 必须报错
            early_stopping=EarlyStoppingConfig(enabled=True, min_epochs=1, patience=1, min_delta=1e-4),
        )


# =========================================================== v2.2：低频全体积验证调度


def test_lowfreq_skips_non_multiple_epochs_and_validates_on_schedule(tmp_path: Path, mini_plan):
    """every_n_epochs=5：第 1–4 epoch 跳过验证，第 5 epoch 验证；跳过 epoch 不更新 best/patience。"""
    trainer, validator = build_formal_trainer(
        tmp_path, mini_plan,
        dice_metrics={4: 0.42}, epochs=5,
        early_stopping=EarlyStoppingConfig(enabled=False, min_epochs=1, patience=8, min_delta=1e-4),
        min_delta=1e-4,
        validation_every_n_epochs=5,
        run_name="run_lowfreq",
    )
    summary = trainer.train()
    assert validator.calls == [4]  # 只有第 5 个 epoch（0-based 4）触发全体积验证
    assert summary.validation_events_completed == 1
    assert summary.best_epoch == 4 and summary.best_metric == pytest.approx(0.42)
    assert summary.bad_validation_events == 0
    rows = list(csv.DictReader((tmp_path / "run_lowfreq" / "metrics" / "epoch_metrics.csv").open()))
    assert len(rows) == 5
    assert [int(r["validation_skipped"]) for r in rows] == [1, 1, 1, 1, 0]
    assert [int(r["validation_event"]) for r in rows] == [0, 0, 0, 0, 1]
    assert [int(r["is_best"]) for r in rows] == [0, 0, 0, 0, 1]  # best 只在 validation event 产生
    # 跳过 epoch 的 dice 字段为空串（稳定 CSV schema），验证 epoch 有值
    assert rows[0]["val_positive_casewise_dice_mean"] == ""
    assert float(rows[4]["val_positive_casewise_dice_mean"]) == pytest.approx(0.42)
    # 跳过 epoch 仍保存 checkpoint_last（可恢复）；best 在验证 epoch 产生
    assert (tmp_path / "run_lowfreq" / "checkpoints" / "checkpoint_last.pth").is_file()
    assert (tmp_path / "run_lowfreq" / "checkpoints" / "checkpoint_best.pth").is_file()


def test_lowfreq_final_epoch_validates_even_if_not_multiple(tmp_path: Path, mini_plan):
    """epochs=7、every_n_epochs=5：验证发生在第 5 与第 7（最终）epoch，即使 7 不是 5 的倍数。"""
    trainer, validator = build_formal_trainer(
        tmp_path, mini_plan,
        dice_metrics={4: 0.30, 6: 0.50}, epochs=7,
        early_stopping=EarlyStoppingConfig(enabled=False, min_epochs=1, patience=8, min_delta=1e-4),
        min_delta=1e-4,
        validation_every_n_epochs=5,
        run_name="run_lowfreq_final",
    )
    summary = trainer.train()
    assert validator.calls == [4, 6]  # 0-based：第 5 与第 7（最终）epoch
    assert summary.validation_events_completed == 2
    assert summary.best_epoch == 6 and summary.best_metric == pytest.approx(0.50)


def test_lowfreq_non_validation_epoch_does_not_update_best(tmp_path: Path, mini_plan):
    """非验证 epoch 不解析指标、不更新 best；best checkpoint 只落在 validation event 上。"""
    trainer, validator = build_formal_trainer(
        tmp_path, mini_plan,
        dice_metrics={1: 0.30, 3: 0.20}, epochs=4,
        early_stopping=EarlyStoppingConfig(enabled=False, min_epochs=1, patience=8, min_delta=1e-4),
        min_delta=1e-4,
        validation_every_n_epochs=2,
        run_name="run_lowfreq_best",
    )
    summary = trainer.train()
    assert validator.calls == [1, 3]  # 1-based epoch 2、4
    assert summary.best_epoch == 1 and summary.best_metric == pytest.approx(0.30)
    assert summary.bad_validation_events == 1
    best_meta = load_checkpoint(tmp_path / "run_lowfreq_best" / "checkpoints" / "checkpoint_best.pth")
    assert best_meta["epoch"] == 1  # best checkpoint 落在 validation event（epoch1）上


def test_lowfreq_early_stopping_counts_validation_events(tmp_path: Path, mini_plan):
    """可选 early stopping：patience 按 validation event 计数（非 epoch）。"""
    trainer, validator = build_formal_trainer(
        tmp_path, mini_plan,
        dice_metrics={1: 0.5, 3: 0.4, 5: 0.3}, epochs=10,
        early_stopping=EarlyStoppingConfig(enabled=True, min_epochs=1, patience=2, min_delta=1e-4),
        min_delta=1e-4,
        validation_every_n_epochs=2,
        run_name="run_lowfreq_es",
    )
    summary = trainer.train()
    assert validator.calls == [1, 3, 5]  # 验证事件在 0-based epoch 1、3、5
    assert summary.bad_validation_events == 2
    assert summary.epochs_done == 6  # 停在 0-based epoch 5（第 6 个 epoch）
    assert summary.early_stopped
    payload = json.loads((tmp_path / "run_lowfreq_es" / "diagnostics" / "early_stop_summary.json").read_text())
    assert payload["bad_validation_events"] == 2 and payload["patience"] == 2
    assert "validation event" in payload["reason"] and "epoch 未改善" not in payload["reason"]


def test_lowfreq_early_stopping_min_epochs_is_training_epochs(tmp_path: Path, mini_plan):
    """min_epochs 按 training epoch 解释：1-based epoch 4 的 event 已满足 patience，但 4 < min_epochs=5 不停。"""
    trainer, validator = build_formal_trainer(
        tmp_path, mini_plan,
        dice_metrics={1: 0.5, 3: 0.4, 5: 0.3}, epochs=10,
        early_stopping=EarlyStoppingConfig(enabled=True, min_epochs=5, patience=1, min_delta=1e-4),
        min_delta=1e-4,
        validation_every_n_epochs=2,
        run_name="run_lowfreq_minepochs",
    )
    summary = trainer.train()
    assert validator.calls == [1, 3, 5]
    assert summary.epochs_done == 6  # epoch3（1-based 4<5）不停；epoch5（1-based 6>=5）停


def test_lowfreq_resume_restores_event_counters(tmp_path: Path, mini_plan):
    """resume 恢复 validation_events_completed / bad_validation_events / best / last_validation_epoch。"""
    protocol = EarlyStoppingConfig(enabled=False, min_epochs=1, patience=8, min_delta=1e-4)
    run_name = "run_lowfreq_resume"
    first, _ = build_formal_trainer(
        tmp_path, mini_plan,
        dice_metrics={1: 0.4}, epochs=2,
        early_stopping=protocol, min_delta=1e-4, run_name=run_name, validation_every_n_epochs=2,
    )
    s1 = first.train()
    assert s1.validation_events_completed == 1 and first.last_validation_epoch == 1

    second, validator = build_formal_trainer(
        tmp_path, mini_plan,
        dice_metrics={3: 0.5}, epochs=4,
        early_stopping=protocol, min_delta=1e-4, run_name=run_name, validation_every_n_epochs=2,
    )
    second.resume(tmp_path / run_name / "checkpoints" / "checkpoint_last.pth")
    assert second.validation_events_completed == 1
    assert second.bad_validation_events == 0
    assert second.best_metric == pytest.approx(0.4) and second.best_epoch == 1
    assert second.last_validation_epoch == 1
    s2 = second.train()
    assert validator.calls == [3]  # epoch2 跳过，epoch3（1-based 4，最终）验证
    assert s2.validation_events_completed == 2


def test_lowfreq_resume_rejects_schedule_change_and_resets_on_allow(tmp_path: Path, mini_plan):
    """改变 every_n_epochs 属于协议变化：默认拒绝 resume；显式放行时重置 event/best 状态。"""
    run_name = "run_lowfreq_sched"
    first, _ = build_formal_trainer(
        tmp_path, mini_plan,
        dice_metrics={1: 0.4}, epochs=2,
        early_stopping=EarlyStoppingConfig(enabled=False, min_epochs=1, patience=8, min_delta=1e-4),
        min_delta=1e-4, run_name=run_name, validation_every_n_epochs=2,
    )
    first.train()

    changed, _ = build_formal_trainer(
        tmp_path, mini_plan,
        dice_metrics={2: 0.5}, epochs=3,
        early_stopping=EarlyStoppingConfig(enabled=False, min_epochs=1, patience=8, min_delta=1e-4),
        min_delta=1e-4, run_name=run_name, validation_every_n_epochs=1,  # 调度改变
    )
    with pytest.raises(RuntimeError, match="协议不一致"):
        changed.resume(tmp_path / run_name / "checkpoints" / "checkpoint_last.pth")

    allowed, validator = build_formal_trainer(
        tmp_path, mini_plan,
        dice_metrics={2: 0.5}, epochs=3,
        early_stopping=EarlyStoppingConfig(enabled=False, min_epochs=1, patience=8, min_delta=1e-4),
        min_delta=1e-4, run_name=run_name, validation_every_n_epochs=1,
    )
    allowed.resume(
        tmp_path / run_name / "checkpoints" / "checkpoint_last.pth", allow_protocol_change=True
    )
    assert allowed.best_metric is None and allowed.validation_events_completed == 0
    assert allowed.bad_validation_events == 0 and allowed.last_validation_epoch is None
    summary = allowed.train()
    assert validator.calls == [2]  # every_n_epochs=1，从 epoch2 开始每 epoch 验证（epoch2 也是最终）
    assert summary.best_epoch == 2


def test_trainer_rejects_invalid_every_n_epochs(tmp_path: Path, mini_plan):
    """validation_every_n_epochs 必须 >= 1：0/负数非法；1 与 5 合法。"""
    for bad in (0, -1):
        with pytest.raises(ValueError, match="validation_every_n_epochs"):
            build_formal_trainer(
                tmp_path, mini_plan,
                dice_metrics={0: 0.1}, epochs=1,
                early_stopping=EarlyStoppingConfig(enabled=False, min_epochs=1, patience=8, min_delta=1e-4),
                min_delta=1e-4, validation_every_n_epochs=bad, run_name=f"run_bad_{abs(bad)}",
            )
    for good in (1, 5):
        trainer, _ = build_formal_trainer(
            tmp_path, mini_plan,
            dice_metrics={0: 0.1}, epochs=1,
            early_stopping=EarlyStoppingConfig(enabled=False, min_epochs=1, patience=8, min_delta=1e-4),
            min_delta=1e-4, validation_every_n_epochs=good, run_name=f"run_ok_{good}",
        )
        assert trainer.validation_every_n_epochs == good
