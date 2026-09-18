"""checkpoint / resume 架构身份隔离测试（任务书九·16,17,18；合成 CPU，不读真实数据、不使用 GPU）。

覆盖：相同架构 checkpoint/resume 成功、架构哈希不一致 resume 被拒绝、legacy PlainConv checkpoint 被明确拒绝、
legacy 权重严格加载到新骨干失败（禁止 strict=False 绕过），以及 trainer 冻结前缀包含架构身份。
"""
from __future__ import annotations

import pytest
import torch

from zonal_reliability_fusion.config.architecture import ArchitectureIdentityError
from zonal_reliability_fusion.models import M0ConcatModel, M0ResidualConcatModel
from zonal_reliability_fusion.training import (
    DeepSupervisionLoss,
    PiCAIFocalCELoss,
    deep_supervision_weights,
    load_checkpoint,
    save_checkpoint,
    verify_checkpoint_architecture,
)
from zonal_reliability_fusion.training.trainer import M0Trainer
from zonal_reliability_fusion.utils import capture_rng_states


def _batch_source(plan, n=2, iters=2):
    data = torch.randn(n, 3, *plan.patch_size)
    seg = torch.randint(0, 2, (n, 1, *plan.patch_size))
    batches = [(data, seg) for _ in range(iters)]
    return lambda: iter(batches)


def _make_trainer(model, plan, run_dir, config_snapshot):
    loss = DeepSupervisionLoss(
        PiCAIFocalCELoss(focal_weight=0.5, ce_weight=0.5, gamma=2.0, smooth=1e-5),
        weights=deep_supervision_weights(len(plan.decoder_output_shapes())),
    )
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.99, nesterov=True)
    return M0Trainer(
        model=model,
        loss_fn=loss,
        optimizer=opt,
        scheduler=None,
        device="cpu",
        train_batches=_batch_source(plan),
        val_batches=_batch_source(plan),
        run_dir=run_dir,
        epochs=1,
        iterations_per_epoch=2,
        validation_iterations=1,
        validation_mode="diagnostic_patch",
        checkpoint_metric="val_loss",
        maximize=False,
        progress=False,
        config_snapshot=config_snapshot,
        seed=0,
    )


# ------------------------------------------------------------------ 16. 相同架构 resume 成功
def test_same_architecture_checkpoint_roundtrip(tmp_path, mini_resenc_plan, resenc_spec):
    model = M0ResidualConcatModel(mini_resenc_plan, resenc_spec, in_channels=3, num_classes=2)
    ident = model.architecture_identity()
    ckpt = tmp_path / "ckpt.pth"
    save_checkpoint(ckpt, model=model, epoch=0, best_metric=0.5, config={"architecture": ident}, seed=0,
                    rng_states=capture_rng_states())

    # 身份守卫通过（不抛）
    found = verify_checkpoint_architecture(ckpt, ident)
    assert found["architecture_sha256"] == ident["architecture_sha256"]

    # 全新同架构模型 strict 加载成功，且权重被覆盖为 checkpoint 的值
    fresh = M0ResidualConcatModel(mini_resenc_plan, resenc_spec, in_channels=3, num_classes=2)
    meta = load_checkpoint(ckpt, model=fresh, strict=True)
    assert meta["epoch"] == 0
    ref = model.state_dict()
    got = fresh.state_dict()
    assert set(ref) == set(got)
    assert all(torch.equal(ref[k], got[k]) for k in ref)  # strict 加载后逐权重一致


# ------------------------------------------------------------------ 17. 哈希不一致 → 拒绝
def test_verify_rejects_hash_mismatch(tmp_path, mini_resenc_plan, resenc_spec):
    model = M0ResidualConcatModel(mini_resenc_plan, resenc_spec, in_channels=3, num_classes=2)
    ident = model.architecture_identity()
    ckpt = tmp_path / "ckpt.pth"
    save_checkpoint(ckpt, model=model, epoch=0, config={"architecture": ident})
    tampered = dict(ident)
    tampered["architecture_sha256"] = "0" * 64
    with pytest.raises(ArchitectureIdentityError, match="架构身份不一致"):
        verify_checkpoint_architecture(ckpt, tampered)


# ------------------------------------------------------------------ 18. legacy checkpoint 明确拒绝
def test_verify_rejects_legacy_checkpoint(tmp_path, mini_resenc_plan, resenc_spec):
    legacy = M0ConcatModel(mini_resenc_plan, in_channels=3, num_classes=2)  # 无架构身份
    residual = M0ResidualConcatModel(mini_resenc_plan, resenc_spec, in_channels=3, num_classes=2)
    ckpt = tmp_path / "legacy.pth"
    save_checkpoint(ckpt, model=legacy, epoch=0, config={"model_id": "M0"})  # 无 architecture 键
    with pytest.raises(ArchitectureIdentityError, match="legacy"):
        verify_checkpoint_architecture(ckpt, residual.architecture_identity())


def test_legacy_state_dict_strict_load_into_residual_rejected(mini_resenc_plan, resenc_spec):
    legacy = M0ConcatModel(mini_resenc_plan, in_channels=3, num_classes=2)
    residual = M0ResidualConcatModel(mini_resenc_plan, resenc_spec, in_channels=3, num_classes=2)
    # 严格加载 legacy 权重到 residual → key 结构不同 → RuntimeError（不得用 strict=False 绕过）
    with pytest.raises(RuntimeError):
        residual.load_state_dict(legacy.state_dict(), strict=True)


def test_load_checkpoint_defaults_to_strict_true(tmp_path, mini_resenc_plan, resenc_spec):
    import inspect

    sig = inspect.signature(load_checkpoint)
    assert sig.parameters["strict"].default is True  # 默认严格，禁止静默 strict=False 迁移


# ------------------------------------------------------------------ trainer 冻结前缀 + resume 拒绝
def test_trainer_frozen_prefix_includes_architecture():
    assert "architecture." in M0Trainer.FROZEN_CONFIG_PREFIXES
    assert "provenance.architecture_sha256" in M0Trainer.FROZEN_CONFIG_PREFIXES


def test_trainer_resume_rejects_architecture_hash_mismatch(tmp_path, mini_resenc_plan, resenc_spec):
    model = M0ResidualConcatModel(mini_resenc_plan, resenc_spec, in_channels=3, num_classes=2)
    ident = model.architecture_identity()
    base_snapshot = {"architecture": dict(ident), "provenance": {"architecture_sha256": ident["architecture_sha256"]}}

    run_dir = tmp_path / "run"
    t1 = _make_trainer(model, mini_resenc_plan, run_dir, base_snapshot)
    t1.train()  # 写 checkpoint_last.pth（含 architecture 身份）
    ckpt = run_dir / "checkpoints" / "checkpoint_last.pth"
    assert ckpt.is_file()

    # 相同架构 → resume 成功（start_epoch=1）
    ok = _make_trainer(
        M0ResidualConcatModel(mini_resenc_plan, resenc_spec, in_channels=3, num_classes=2),
        mini_resenc_plan, run_dir, {"architecture": dict(ident),
                                    "provenance": {"architecture_sha256": ident["architecture_sha256"]}},
    )
    ok.resume(ckpt)
    assert ok.start_epoch == 1

    # 架构哈希不一致 → resume 被拒绝（frozen 前缀命中）
    bad = dict(ident)
    bad["architecture_sha256"] = "f" * 64
    bad_trainer = _make_trainer(
        M0ResidualConcatModel(mini_resenc_plan, resenc_spec, in_channels=3, num_classes=2),
        mini_resenc_plan, tmp_path / "run_bad",
        {"architecture": bad, "provenance": {"architecture_sha256": bad["architecture_sha256"]}},
    )
    with pytest.raises(RuntimeError, match="拒绝续训"):
        bad_trainer.resume(ckpt)
