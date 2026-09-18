"""Checkpoint 与 PolyLR 测试（临时目录，强制 CPU）。"""
from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from zonal_reliability_fusion.training import (
    CheckpointManager,
    atomic_torch_save,
    load_checkpoint,
    save_checkpoint,
)
from zonal_reliability_fusion.training.optim import PolyLRScheduler


def make_setup():
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.LeakyReLU(0.01), torch.nn.Linear(4, 2))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.99, nesterov=True, weight_decay=3e-5)
    scheduler = PolyLRScheduler(optimizer, initial_lr=0.01, max_steps=10, power=0.9)
    return model, optimizer, scheduler


def test_polylr_semantics_and_state_round_trip():
    model, optimizer, scheduler = make_setup()
    lr0 = scheduler.step(0)
    assert lr0 == pytest.approx(0.01)
    lr5 = scheduler.step(5)
    assert lr5 == pytest.approx(0.01 * (1 - 5 / 10) ** 0.9)
    state = scheduler.state_dict()
    other = PolyLRScheduler(optimizer, 0.01, 10, power=0.9)
    other.load_state_dict(copy.deepcopy(state))
    assert other.current_step == 6
    assert other.get_last_lr()[0] == pytest.approx(lr5)
    # 恢复不改变 optimizer 的当前 lr（lr 由 epoch 索引在每轮 step(epoch) 时确定）
    optimizer.param_groups[0]["lr"] = 0.0001
    other.load_state_dict(copy.deepcopy(state))
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0001)
    assert other.step(5) == pytest.approx(lr5)  # 以显式 epoch 调用 → LR 连续


def test_checkpoint_round_trip(tmp_path: Path):
    model, optimizer, scheduler = make_setup()
    ckpt = tmp_path / "ckpt.pth"
    save_checkpoint(
        ckpt, model=model, epoch=3, best_metric=0.125, optimizer=optimizer, scheduler=scheduler,
        config={"lr": 0.01}, seed=7,
    )
    scheduler.step(4)
    optimizer.param_groups[0]["lr"] = 0.003

    model2, optimizer2, scheduler2 = make_setup()
    with torch.no_grad():
        for p in model2.parameters():
            p.add_(1.0)
    meta = load_checkpoint(ckpt, model=model2, optimizer=optimizer2, scheduler=scheduler2, map_location="cpu")
    assert meta["epoch"] == 3
    assert meta["best_metric"] == pytest.approx(0.125)
    assert meta["config"] == {"lr": 0.01}
    assert meta["seed"] == 7
    for a, b in zip(model.parameters(), model2.parameters()):
        assert torch.equal(a, b)
    assert optimizer2.param_groups[0]["lr"] == pytest.approx(0.01)  # 恢复的是 checkpoint 里的 optimizer 状态
    assert scheduler2.current_step == 0  # 保存时的调度器状态（未与后续 step 混淆）
    assert scheduler2.initial_lr == pytest.approx(0.01)


def test_atomic_save_leaves_no_tmp_files(tmp_path: Path):
    path = tmp_path / "obj.pth"
    atomic_torch_save({"a": 1}, path)
    assert path.is_file()
    assert list(tmp_path.glob(".*tmp*")) == []
    assert list(tmp_path.glob("*.tmp")) == []


def test_checkpoint_manager_naming(tmp_path: Path):
    model, optimizer, scheduler = make_setup()
    manager = CheckpointManager(tmp_path / "run", save_every=2)
    written = manager.save(
        model=model, epoch=0, best_metric=1.0, optimizer=optimizer, scheduler=scheduler, is_best=True
    )
    assert set(written) == {"last", "best"}
    assert manager.last_path.is_file() and manager.best_path.is_file()

    written = manager.save(model=model, epoch=1, best_metric=1.0, optimizer=optimizer, scheduler=scheduler)
    # save_every=2 → (epoch + 1) % 2 == 0 时写出周期 checkpoint
    assert set(written) == {"last", "periodic"}
    assert manager.epoch_path(1).is_file()

    written = manager.save(
        model=model, epoch=99, best_metric=None, optimizer=optimizer, scheduler=scheduler, interruption=True
    )
    assert "interrupt" in written
    assert manager.interrupt_path.is_file()
    assert len(manager.list_checkpoints()) >= 3


def test_manager_writes_epoch_metadata_to_all_checkpoints(tmp_path: Path):
    """第三轮复审回归：last / best / 周期 checkpoint 必须携带一致的 epoch 元数据。"""
    model, optimizer, scheduler = make_setup()
    manager = CheckpointManager(tmp_path / "run", save_every=1)
    written = manager.save(
        model=model,
        epoch=4,
        best_metric=0.5,
        optimizer=optimizer,
        scheduler=scheduler,
        is_best=True,
        completed_epochs=5,
        iterations_completed=250,
    )
    assert set(written) == {"last", "best", "periodic"}
    for name, path in (
        ("last", manager.last_path),
        ("best", manager.best_path),
        ("periodic", manager.epoch_path(4)),
    ):
        meta = load_checkpoint(path)
        assert meta["epoch"] == 4, name
        assert meta["epoch_completed"] is True, name
        assert meta["completed_epochs"] == 5, name
        assert meta["iterations_completed"] == 250, name

    # 未显式给出 completed_epochs 时，完整 epoch 自动填 epoch + 1
    manager.save(model=model, epoch=6, best_metric=0.6, optimizer=optimizer, scheduler=scheduler)
    meta = load_checkpoint(manager.last_path)
    assert meta["epoch_completed"] is True and meta["completed_epochs"] == 7

    # interrupt 一律标记为未完成，并保留调用方给出的绝对计数
    manager.save(
        model=model,
        epoch=7,
        best_metric=0.6,
        optimizer=optimizer,
        scheduler=scheduler,
        interruption=True,
        completed_epochs=7,
        iterations_completed=3,
    )
    meta = load_checkpoint(manager.interrupt_path)
    assert meta["epoch_completed"] is False
    assert meta["completed_epochs"] == 7 and meta["iterations_completed"] == 3
    assert meta["extra"]["interrupted"] is True


def test_load_missing_checkpoint_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "nope.pth")


def test_checkpoint_carries_rng_states(tmp_path: Path):
    from zonal_reliability_fusion.utils import capture_rng_states, restore_rng_states

    model, optimizer, scheduler = make_setup()
    states = capture_rng_states()
    path = tmp_path / "rng.pth"
    save_checkpoint(path, model=model, epoch=1, optimizer=optimizer, scheduler=scheduler, rng_states=states)
    meta = load_checkpoint(path, model=model)
    assert meta["rng_states"] is not None
    restore_rng_states(meta["rng_states"])  # 不应抛异常
