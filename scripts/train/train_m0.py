#!/usr/bin/env python
"""M0 训练入口（自研实现；所有真实数据/训练命令均由研究者手动运行）。

三种模式：
1. `--loader-smoke`  （命令 B）读取少量真实病例，检查 data/seg 契约、模态顺序、采样器与增强；
                      **不训练、不做 forward、不使用 GPU**；
2. `--overfit`       （命令 C）用少量阳性病例做固定 iteration 的小样本过拟合（**诊断**：随机 patch 验证），
                      输出独立 diagnostic 目录（checkpoint + loss 曲线），不与正式训练目录混用；
3. 默认模式           **正式训练**（max_epochs 见配置，正式 1000），按 `validation.every_n_epochs`（正式 50）
                      低频对 fold 0 的全部 validation study（223 例）做**完整 3D 体积**滑窗推理（最后一个 epoch
                      即使不是 5 的倍数也必验证），按 `val_positive_casewise_dice_mean` 最大化保存 best checkpoint；
                      正式 M0–M4 使用固定 200-epoch 预算，**禁用**“连续无改善”early stopping（只保留 NaN/Inf/
                      覆盖病例数错误等安全中止）；输出
                      `outputs/checkpoints/m0/<run>/`、`outputs/metrics/m0/<run>/`、`outputs/diagnostics/m0/<run>/`。

运行前提：
    cd /opt/data/private/lm/my-projects
    source scripts/env_nnunet.sh        # 固定 third_party/nnUNet v2.6.2 + 项目内 nnUNet_* 路径
    /root/anaconda3/envs/lm/bin/python scripts/train/train_m0.py ...

安全约定：
- 显式 `--device`（默认 cuda；CPU 场景传 cpu），import 阶段不初始化 CUDA；
- 不修改任何原始/预处理数据（只读）；
- 正式 run 若目标目录已存在内容且未显式 `--resume-from`，拒绝启动（不追加旧 CSV、不覆盖 checkpoint）；
- 训练中断（Ctrl-C）会保存 `checkpoint_interrupt.pth` 后继续抛出异常；该 checkpoint 标记为
  `epoch_completed=False`（记录 `completed_epochs` 与 `iterations_completed`），默认**不允许**直接续训——
  建议改用与最后一个完整 epoch 对齐的 `checkpoint_last.pth`；确需从中断状态继续时显式传
  `--allow-mid-epoch-resume`（会重跑被打断的 epoch）；
- 续训会核对完整训练协议（checkpoint 指标 / maximize / min_epochs / patience / min_delta / validation 协议），
  不一致默认拒绝；确需改变时显式传 `--allow-protocol-change`。
"""
from __future__ import annotations

import argparse
import csv as _csv
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import torch

from zonal_reliability_fusion.config import (
    M0ExperimentConfig,
    load_experiment_config,
    load_plan,
    validate_against_plan,
    validation_protocol_from_config,
)
from zonal_reliability_fusion.data import (
    PatchSampler,
    PreprocessedStore,
    TorchBatchProvider,
    build_train_transforms,
    build_validation_transforms,
    verify_case_contract,
)
from zonal_reliability_fusion.evaluation import FullVolumeValidator
from zonal_reliability_fusion.models.factory import (
    build_model,
    model_architecture_identity,
    model_identity,
)
from zonal_reliability_fusion.training import (
    VALIDATION_MODE_DIAGNOSTIC,
    VALIDATION_MODE_FULL_VOLUME,
    DeepSupervisionLoss,
    DiceCELoss,
    EarlyStoppingConfig,
    M0Trainer,
    PiCAIFocalCELoss,
    build_optimizer,
    build_scheduler,
    deep_supervision_weights,
    verify_checkpoint_architecture,
)
from zonal_reliability_fusion.training.gpu_memory_profile import (
    GPU_PROFILE_FILENAME,
    PHASE_BUILD,
    PHASE_TRAIN,
    PHASE_TRAINER_RUN,
    PHASE_VALIDATION,
    STATUS_COMPLETED,
    STATUS_CUDA_OOM,
    STATUS_ERROR,
    STATUS_NOT_APPLICABLE,
    GpuMemoryProfileError,
    GpuMemoryProfiler,
    is_cuda_oom,
)
from zonal_reliability_fusion.utils import (
    code_version_info,
    file_sha256,
    make_progress,
    resolve_progress,
    set_seed,
)

DEFAULT_CONFIG = PROJECT_ROOT / "configs/experiments/m0_picai_3d_fullres.yaml"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="M0 训练 / loader smoke / 小样本过拟合（自研实现）")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--device", default="cuda", help="cpu / cuda / cuda:0（显式指定，不做隐式选择）")
    ap.add_argument("--no-progress", action="store_true", help="关闭 tqdm 进度条（日志环境）")
    ap.add_argument("--run-name", default="", help="输出子目录名（默认 <experiment.name>）")

    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--loader-smoke", action="store_true", help="只做真实数据 loader 检查（不训练、不用 GPU）")
    mode.add_argument("--overfit", action="store_true", help="小样本过拟合（独立的诊断验证，输出到 diagnostic 目录）")

    ap.add_argument("--cases", default="", help="逗号分隔 case_id（loader-smoke/overfit 使用）")
    ap.add_argument("--limit-cases", type=int, default=2, help="loader-smoke 未指定 --cases 时默认取 train 前 N 例")
    ap.add_argument("--iterations", type=int, default=60, help="overfit 的 train iteration 数")
    ap.add_argument("--val-iterations", type=int, default=5, help="overfit 的 diagnostic validation iteration 数")
    ap.add_argument("--epochs", type=int, default=1, help="overfit 的 epoch 数")
    ap.add_argument(
        "--resume-from",
        default="",
        help="从 checkpoint 继续正式训练（跳过 run 目录非空检查；会核对训练协议/配置冻结项与 epoch 日志对齐，"
        "不对齐时请改用 checkpoint_last.pth 或换 --run-name）",
    )
    ap.add_argument(
        "--allow-mid-epoch-resume",
        action="store_true",
        help="允许从训练中途的 interrupt checkpoint 续训（会重跑被打断的那个 epoch；默认拒绝，建议改用 checkpoint_last.pth）",
    )
    ap.add_argument(
        "--allow-protocol-change",
        action="store_true",
        help="允许在训练协议（checkpoint 指标/maximize/early stopping/验证调度）变化后续训（默认拒绝；"
        "放行时会重置 best/bad_validation_events/validation_events_completed，选模从续训点重新开始）",
    )
    ap.add_argument("--no-plot", action="store_true", help="overfit 不生成 loss 曲线 PNG")
    return ap.parse_args()


# --------------------------------------------------------------------------- 公共
def _all_finite(*arrays) -> bool:
    """有限性检查：torch.Tensor 用 torch.isfinite，numpy 用 np.isfinite（避免 numpy↔torch 桥接）。"""
    for arr in arrays:
        if isinstance(arr, torch.Tensor):
            if not bool(torch.isfinite(arr).all()):
                return False
        elif not bool(np.isfinite(arr).all()):
            return False
    return True


def load_cfg(args) -> M0ExperimentConfig:
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    return load_experiment_config(cfg_path)


def requires_zonal_prior(cfg: M0ExperimentConfig) -> bool:
    """M3/M4 需要 PZ/TZ prior（由 model_id 决定；配置层已强制来源与根目录存在）。"""
    return str(cfg.experiment.model_id) in ("M3", "M4")


def load_prior_manifest_sha256(cfg: M0ExperimentConfig, store, *, progress: bool = True) -> str | None:
    """M3/M4：**深度校验**数据集级 prior manifest（覆盖 train∪val）并返回 manifest_sha256；M0–M2 返回 None。

    由本函数（脚本层）打印校验统计；`store.load_zonal_prior_manifest` 内部缓存，
    同一次启动内重复调用不会再次扫描全部 prior。
    """
    if not requires_zonal_prior(cfg):
        return None
    manifest = store.load_zonal_prior_manifest(
        list(store.train_ids) + list(store.val_ids),
        progress=progress,
        desc=f"prior sidecar 校验 ({cfg.experiment.model_id})",
    )
    stats = dict(getattr(store, "zonal_prior_validation_stats", {}) or {})
    print(
        "[m0 train] prior 校验："
        f"cases={stats.get('n_cases')} ok={stats.get('n_ok')} failed={stats.get('n_failed')} "
        f"array_hash_checked={stats.get('array_hash_checked')} 耗时={stats.get('elapsed_sec')}s "
        f"manifest_sha256={str(stats.get('manifest_sha256'))[:12]}…"
        f" frozen_records={len(getattr(store, 'zonal_prior_frozen_records', {}) or {})}"
    )
    return str(manifest["manifest_sha256"])


def assert_prior_snapshot_consistency(snapshot: dict) -> None:
    """M3/M4 快照硬约束：pipeline 里记录的 prior 身份必须与 provenance 完全一致且已校验数组。

    要求 `train/val_pipeline.zonal_prior.manifest_sha256 == provenance.zonal_prior_manifest_sha256`
    且 `array_hash_checked is True`（否则 resume/审计会把「未校验的 prior」当作已验证事实）。
    """
    provenance_sha = (snapshot.get("provenance") or {}).get("zonal_prior_manifest_sha256")
    for key in ("train_pipeline", "val_pipeline"):
        pipeline = snapshot.get(key)
        if not isinstance(pipeline, dict):
            continue
        zonal = pipeline.get("zonal_prior") or {}
        if not zonal.get("enabled"):
            continue
        if zonal.get("manifest_sha256") != provenance_sha:
            raise SystemExit(
                f"[m0 train] 快照不一致：{key}.zonal_prior.manifest_sha256="
                f"{zonal.get('manifest_sha256')!r} != provenance.zonal_prior_manifest_sha256={provenance_sha!r}"
            )
        if zonal.get("array_hash_checked") is not True:
            raise SystemExit(
                f"[m0 train] 快照不一致：{key}.zonal_prior.array_hash_checked 必须为 True"
                "（prior 数组未完成逐个 SHA256 校验，不得启动正式训练）"
            )


def build_store(cfg: M0ExperimentConfig) -> PreprocessedStore:
    if requires_zonal_prior(cfg):
        if cfg.data.zonal_prior_root is None or cfg.data.zonal_prior_source is None:
            raise SystemExit(
                f"[m0 train] model_id={cfg.experiment.model_id} 需要 PZ/TZ prior，但配置缺少 "
                "data.zonal_prior_source / data.zonal_prior_root"
            )
        return PreprocessedStore(
            cfg.resolve_path(cfg.paths.preprocessed_dataset),
            cfg.resolve_path(cfg.paths.splits),
            fold=cfg.experiment.fold,
            zonal_prior_root=cfg.resolve_path(cfg.data.zonal_prior_root),
            zonal_prior_source=cfg.data.zonal_prior_source,
            # sidecar 必须与当前 plan 文件哈希 + configuration 绑定（只比 shape 不够）
            expected_plans_sha256=file_sha256(cfg.resolve_path(cfg.paths.plans)),
            expected_configuration=cfg.model.configuration,
            # 正式训练 fail-closed：未成功加载并冻结 manifest 记录前，禁止逐例读取 prior
            require_frozen_manifest=True,
        )
    return PreprocessedStore(
        cfg.resolve_path(cfg.paths.preprocessed_dataset),
        cfg.resolve_path(cfg.paths.splits),
        fold=cfg.experiment.fold,
    )


#: 预检产物文件名（`validate_m0_setup.py` 写出）——不视为"已存在的训练 run"
PREFLIGHT_DIAGNOSTIC_FILES = ("setup_check.json",)
#: 预检产物文件名前缀（`--loader-smoke` 的按时间戳落盘）——同上
PREFLIGHT_DIAGNOSTIC_PREFIXES = ("loader_smoke_",)


def preflight_only_diagnostics(directory: Path) -> bool:
    """`diagnostics` 目录是否**只**包含预检产物（不含任何训练 run 产物）。

    预检产物由研究者按手册在正式训练**之前**执行：
    `validate_m0_setup.py`（`setup_check.json`）与 `--loader-smoke`（`loader_smoke_<时间戳>.json`）。
    它们不构成"已存在的训练 run"，因此不得阻止正式训练启动（否则照手册顺序先跑预检就无法开训）。
    """
    if not directory.exists():
        return True
    for entry in directory.iterdir():
        name = entry.name
        if name in PREFLIGHT_DIAGNOSTIC_FILES or any(
            name.startswith(prefix) for prefix in PREFLIGHT_DIAGNOSTIC_PREFIXES
        ):
            continue
        return False
    return True


def ensure_fresh_run_dirs(
    *dirs: Path,
    resume_from: str = "",
    preflight_tolerant: Sequence[Path] = (),
) -> None:
    """正式 run 的安全检查：目标目录非空且未显式 resume → 拒绝启动。

    避免向旧 CSV 追加、覆盖旧 checkpoint 或污染历史实验。

    `preflight_tolerant` 中的目录允许**只包含预检产物**（见 `preflight_only_diagnostics`）；
    任何训练产物（`run_manifest.json`、checkpoint、epoch CSV、loss 曲线等）仍一律拒绝。
    """
    if resume_from:
        return
    tolerant = {Path(p) for p in preflight_tolerant}
    non_empty = [
        str(d)
        for d in dirs
        if d.exists() and any(d.iterdir()) and not (Path(d) in tolerant and preflight_only_diagnostics(Path(d)))
    ]
    if non_empty:
        raise SystemExit(
            "[m0 train] 以下输出目录已存在内容且未指定 --resume-from，拒绝启动（不覆盖/不追加旧实验）:\n  "
            + "\n  ".join(non_empty)
            + "\n如需继续既有 run，请显式传入 --resume-from <checkpoint_last.pth>；"
            "如需全新 run，请使用新的 --run-name。"
        )


def check_raw_dataset_json() -> dict:
    """只读 nnUNet raw 的 dataset.json，交叉核对通道顺序（不读取体素）。"""
    dataset_json = PROJECT_ROOT / "workdir/nnUNet_raw/Dataset605_PICAI/dataset.json"
    if not dataset_json.is_file():
        return {"available": False, "path": str(dataset_json)}
    doc = json.loads(dataset_json.read_text())
    return {
        "available": True,
        "path": str(dataset_json),
        "channel_names": doc.get("channel_names"),
        "labels": doc.get("labels"),
        "numTraining": doc.get("numTraining"),
        "file_ending": doc.get("file_ending"),
    }


def build_model_and_loss(cfg: M0ExperimentConfig, plan):
    # 按 experiment.model_id 明确构建 M0–M4（未知 model_id / 身份不符一律报错，禁止回退；见 models/factory.py）
    model = build_model(cfg, plan)
    if cfg.loss.name == "focal_ce":
        # 与正式 N0 对齐：0.5 × Focal(γ=2, α=None) + 0.5 × CE（research_plan §9.2）
        base_loss = PiCAIFocalCELoss(
            focal_weight=cfg.loss.focal_weight,
            ce_weight=cfg.loss.ce_weight,
            gamma=cfg.loss.gamma,
            smooth=cfg.loss.smooth,
        )
    elif cfg.loss.name == "dice_ce":
        # 迁移前的自研口径（Dice + CE）；与正式 N0 不可做正式性能比较
        base_loss = DiceCELoss(
            dice_weight=cfg.loss.dice_weight,
            ce_weight=cfg.loss.ce_weight,
            smooth=cfg.loss.smooth,
            include_background=cfg.loss.include_background,
            batch_dice=cfg.loss.batch_dice,
        )
    else:  # pragma: no cover - 配置层已校验
        raise ValueError(f"未知 loss.name={cfg.loss.name!r}；支持 'focal_ce' 与 'dice_ce'")
    if cfg.model.deep_supervision:
        weights = deep_supervision_weights(len(plan.decoder_output_shapes()))
        loss_fn = DeepSupervisionLoss(base_loss, weights=weights)
    else:
        loss_fn = base_loss
    return model, loss_fn


def build_validation_components(
    cfg: M0ExperimentConfig,
    plan,
    store: PreprocessedStore,
    *,
    mode: str,
    device: str,
    progress: bool,
    metrics_dir: Path | None = None,
    diagnostic_case_ids=None,
    diagnostic_num_batches: int | None = None,
    seed_offset: int = 1,
):
    """按验证模式构造 `(validator, val_provider)`。

    - `full_volume`（正式）：返回 `(FullVolumeValidator, None)`，**不创建**随机 validation patch provider；
    - `diagnostic_patch`（loader smoke / small-overfit）：返回 `(None, TorchBatchProvider)`。
    """
    if mode == VALIDATION_MODE_FULL_VOLUME:
        protocol = validation_protocol_from_config(cfg)
        validator = FullVolumeValidator(
            store,
            store.val_ids,
            patch_size=cfg.data.patch_size,
            device=device,
            protocol=protocol,
            metrics_dir=metrics_dir,
            progress=progress,
            desc="val full-volume",
            amp=cfg.training.amp,
            include_zonal_prior=requires_zonal_prior(cfg),
        )
        return validator, None
    if mode == VALIDATION_MODE_DIAGNOSTIC:
        provider = TorchBatchProvider(
            store,
            PatchSampler(
                tuple(cfg.data.patch_size),
                oversample_foreground=cfg.data.oversample_foreground,
                seed=cfg.experiment.seed,
            ),
            batch_size=cfg.data.batch_size,
            num_batches=int(diagnostic_num_batches or cfg.training.diagnostic_validation_iterations),
            case_ids=tuple(diagnostic_case_ids) if diagnostic_case_ids is not None else store.val_ids,
            transform=build_validation_transforms(),
            seed=cfg.experiment.seed + seed_offset,
            force_foreground=False,  # 诊断验证同样不做前景过采样
            include_zonal_prior=requires_zonal_prior(cfg),
            num_workers=cfg.data.num_workers,
            pin_memory=False,
        )
        return None, provider
    raise SystemExit(
        f"[m0] 未知验证模式: {mode!r}（允许 {VALIDATION_MODE_FULL_VOLUME} / {VALIDATION_MODE_DIAGNOSTIC}；"
        "配置层使用 validation.mode）"
    )


def build_trainer(
    cfg,
    plan,
    *,
    device,
    progress,
    run_dir,
    metrics_dir,
    diagnostics_dir,
    validation_mode,
    validator=None,
    store=None,
    iterations=None,
    val_iterations=None,
    epochs=None,
    train_cases=None,
):
    store = store or build_store(cfg)
    seed = cfg.experiment.seed
    sampler = PatchSampler(
        tuple(cfg.data.patch_size), oversample_foreground=cfg.data.oversample_foreground, seed=seed
    )
    pin_memory = bool(cfg.data.pin_memory) and str(device).startswith("cuda")
    train_transform = build_train_transforms(
        mirror=cfg.augmentation.mirror,
        mirror_p_per_axis=cfg.augmentation.mirror_p_per_axis,
        per_channel_intensity=cfg.augmentation.per_channel_intensity,
    )
    train_provider = TorchBatchProvider(
        store,
        sampler,
        batch_size=cfg.data.batch_size,
        num_batches=int(iterations or cfg.training.iterations_per_epoch),
        case_ids=train_cases or store.train_ids,
        transform=train_transform,
        seed=seed,
        force_foreground=None,  # 每个 batch 固定 round(bs*oversample) 个样本强制前景（官方语义）
        include_zonal_prior=requires_zonal_prior(cfg),
        num_workers=cfg.data.num_workers,
        pin_memory=pin_memory,
    )
    val_provider = None
    if validation_mode == VALIDATION_MODE_DIAGNOSTIC:
        val_provider = TorchBatchProvider(
            store,
            sampler,
            batch_size=cfg.data.batch_size,
            num_batches=int(val_iterations or cfg.training.diagnostic_validation_iterations),
            case_ids=store.val_ids if train_cases is None else list(train_cases),
            transform=build_validation_transforms(),
            seed=seed + 1,
            force_foreground=False,
            include_zonal_prior=requires_zonal_prior(cfg),
            num_workers=cfg.data.num_workers,
            pin_memory=pin_memory,
        )
    elif validator is None:
        raise SystemExit("[m0] formal_full_volume 模式必须提供 validator（见 build_validation_components）")

    model, loss_fn = build_model_and_loss(cfg, plan)
    optimizer = build_optimizer(model, cfg.optimizer)
    total_epochs = int(epochs or cfg.training.max_epochs)
    scheduler = build_scheduler(optimizer, cfg.scheduler, epochs=total_epochs, initial_lr=cfg.optimizer.lr)

    if validation_mode == VALIDATION_MODE_FULL_VOLUME:
        checkpoint_metric = cfg.validation.checkpoint_metric
        maximize = bool(cfg.validation.maximize)
        early_stopping = EarlyStoppingConfig(
            enabled=cfg.early_stopping.enabled,
            min_epochs=cfg.early_stopping.min_epochs,
            patience=cfg.early_stopping.patience,
            min_delta=cfg.early_stopping.min_delta,
        )
        min_delta = float(cfg.early_stopping.min_delta)
    else:
        # 诊断模式（small-overfit）：按 val_loss 取最小，作为「是否在拟合」的信号，不用于正式选择
        checkpoint_metric = "val_loss"
        maximize = False
        early_stopping = EarlyStoppingConfig(enabled=False, min_epochs=1, patience=1, min_delta=0.0)
        min_delta = 0.0

    protocol_snapshot = validation_protocol_from_config(cfg).to_dict()
    # M3/M4：**先**完成 prior manifest 深度校验（一次完整扫描），再生成任何 pipeline 快照，
    # 使 train/val_pipeline.zonal_prior 能记录 manifest_sha256 与 array_hash_checked。
    prior_manifest_sha256 = load_prior_manifest_sha256(cfg, store, progress=progress)
    config_snapshot = {
        **cfg.resolved_snapshot(),
        "plan": plan.resolved_snapshot(),
        "validation": protocol_snapshot,
        "early_stopping": asdict(early_stopping),
        "train_pipeline": train_provider.describe(),
        "val_pipeline": None if val_provider is None else val_provider.describe(),
        "provenance": {
            "plans_sha256": file_sha256(cfg.resolve_path(cfg.paths.plans)),
            "splits_sha256": file_sha256(cfg.resolve_path(cfg.paths.splits)),
            "code_version": code_version_info(PROJECT_ROOT),
            # M3/M4：prior 数据集级 manifest 哈希进入冻结快照（resume 时不一致即拒绝）
            "zonal_prior_manifest_sha256": prior_manifest_sha256,
        },
    }
    # 快照自洽性硬约束（M3/M4）：pipeline 与 provenance 的 prior 身份必须逐值一致
    assert_prior_snapshot_consistency(config_snapshot)
    # v2.3：把架构身份（name/version/sha256/encoder_type）写入快照，供 checkpoint/resume 严格隔离；
    # legacy PlainConv（identity=None）保持原快照不变，不影响其既有 resume 行为。
    arch_identity = model_architecture_identity(model)
    if arch_identity is not None:
        config_snapshot["architecture"] = arch_identity
        config_snapshot["architecture_ref"] = cfg.architecture.to_dict() if cfg.architecture is not None else None
        config_snapshot["provenance"]["architecture_sha256"] = arch_identity["architecture_sha256"]
    # M1–M4：把完整模型身份（model_id/backbone/融合结构/哈希）写入快照，供 checkpoint/resume 与日志追溯
    full_identity = model_identity(model)
    if full_identity is not None:
        config_snapshot["model_identity"] = full_identity
    trainer = M0Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        train_batches=lambda: train_provider,
        # trainer 契约：batch source 必须是**可调用**工厂（内部 `iter(source())`）；直接传 provider
        # 实例会得到 `TypeError: 'TorchBatchProvider' object is not callable`（overfit 曾在 validation 阶段失败）。
        val_batches=None if val_provider is None else (lambda: val_provider),
        run_dir=run_dir,
        metrics_dir=metrics_dir,
        diagnostics_dir=diagnostics_dir,
        epochs=total_epochs,
        iterations_per_epoch=train_provider.num_batches,
        validation_iterations=int(val_iterations or cfg.training.diagnostic_validation_iterations),
        validation_mode=validation_mode,
        validator=validator,
        validation_every_n_epochs=cfg.validation.every_n_epochs,
        checkpoint_metric=checkpoint_metric,
        maximize=maximize,
        min_delta=min_delta,
        early_stopping=early_stopping,
        grad_clip=cfg.training.grad_clip,
        amp=cfg.training.amp,
        progress=progress,
        checkpoint_every=cfg.training.checkpoint_every,
        config_snapshot=config_snapshot,
        seed=seed,
        data_sources=[train_provider],  # 每轮 epoch 同步采样序列（保证 resume 后可复现）
    )
    return trainer, model, store


# --------------------------------------------------------------------------- 模式
def run_loader_smoke(args, cfg: M0ExperimentConfig, plan, *, store=None) -> None:
    """命令 B：真实数据 loader 检查（不训练 / 不 forward / 不用 GPU）。

    按 model_id 分派：M0–M2 只读 image（2-tuple）；M3/M4 必须读取 PZ/TZ prior（3-tuple），
    缺 sidecar / 来源不符 / manifest 不一致 / 契约违规一律失败，**不允许 prior 未读却报告成功**。
    """
    t0 = time.time()
    progress = resolve_progress(args.no_progress, cfg.training.progress)
    needs_prior = requires_zonal_prior(cfg)
    store = store if store is not None else build_store(cfg)
    manifest_info: dict = {"checked": False, "required": needs_prior}
    if needs_prior:
        try:
            manifest = store.load_zonal_prior_manifest(
                list(store.train_ids) + list(store.val_ids),
                progress=progress,
                desc=f"prior sidecar 校验 ({cfg.experiment.model_id})",
            )
        except Exception as exc:
            raise SystemExit(f"[m0 loader-smoke] prior manifest 校验失败: {exc}") from exc
        stats = dict(getattr(store, "zonal_prior_validation_stats", {}) or {})
        print(
            "[m0 loader-smoke] prior 校验："
            f"cases={stats.get('n_cases')} ok={stats.get('n_ok')} failed={stats.get('n_failed')} "
            f"array_hash_checked={stats.get('array_hash_checked')} 耗时={stats.get('elapsed_sec')}s "
            f"manifest_sha256={str(stats.get('manifest_sha256'))[:12]}…"
            f" frozen_records={len(getattr(store, 'zonal_prior_frozen_records', {}) or {})}"
        )
        manifest_info = {
            "checked": True,
            "required": True,
            "manifest_sha256": manifest["manifest_sha256"],
            "n_cases": manifest["n_cases"],
            "source": manifest.get("source"),
            "configuration": manifest.get("configuration"),
            "plans_sha256": manifest.get("plans_sha256"),
            "validation_stats": stats,
        }
    case_ids = [c.strip() for c in args.cases.split(",") if c.strip()] if args.cases else list(store.train_ids[: args.limit_cases])
    sampler = PatchSampler(tuple(cfg.data.patch_size), oversample_foreground=cfg.data.oversample_foreground, seed=cfg.experiment.seed)

    payload = {
        "checked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model_id": cfg.experiment.model_id,
        "requires_zonal_prior": needs_prior,
        "zonal_prior_manifest": manifest_info,
        "split": {
            "n_train": len(store.train_ids),
            "n_val": len(store.val_ids),
            "fold": store.fold,
            "splits_sha256": file_sha256(cfg.resolve_path(cfg.paths.splits)),
        },
        "plans_sha256": file_sha256(cfg.resolve_path(cfg.paths.plans)),
        "code_version": code_version_info(PROJECT_ROOT),
        "dataset_json": check_raw_dataset_json(),
        "cases": [],
        "batches": [],
        "elapsed_sec": None,
        "note": "只读检查：不训练、不做 forward、不初始化 CUDA",
    }
    case_bar = make_progress(case_ids, total=len(case_ids), desc="loader-smoke cases", disable=not progress)
    for case_id in case_bar:
        case = (
            store.get_case(case_id, include_zonal_prior=True)
            if needs_prior
            else store.get_case(case_id)
        )
        info = verify_case_contract(case)
        if needs_prior:
            prior = case.zonal_prior
            if prior is None:
                raise SystemExit(f"[m0 loader-smoke] {case_id}: 需要 prior 但 store 未返回（禁止零兜底）")
            overlap = float((prior[0] + prior[1]).max())
            info["prior"] = {
                "source": cfg.data.zonal_prior_source,
                "shape": list(prior.shape),
                "dtype": str(prior.dtype),
                "finite": _all_finite(prior),
                "min": float(prior.min()),
                "max": float(prior.max()),
                "max_pz_plus_tz": overlap,
                "channels": ["PZ", "TZ"],
            }
        samples = {}
        for force_fg in (False, True):
            sample = (
                sampler.sample(
                    case.data,
                    case.seg,
                    zonal_prior=case.zonal_prior,
                    case_id=case_id,
                    properties=case.properties,
                    force_fg=force_fg,
                )
                if needs_prior
                else sampler.sample(
                    case.data, case.seg, case_id=case_id, properties=case.properties, force_fg=force_fg
                )
            )
            entry = {
                "data_shape": list(sample.data.shape),
                "seg_shape": list(sample.seg.shape),
                "data_dtype": str(sample.data.dtype),
                "seg_dtype": str(sample.seg.dtype),
                "seg_labels": sorted(int(v) for v in set(sample.seg.reshape(-1).tolist())),
                "fg_voxels": int((sample.seg == 1).sum()),
                "used_foreground": sample.used_foreground,
                "foreground_fallback": sample.foreground_fallback,
                "patch_matches_plan": tuple(sample.data.shape[-3:]) == tuple(plan.patch_size),
            }
            if needs_prior:
                if sample.zonal_prior is None:
                    raise SystemExit(f"[m0 loader-smoke] {case_id}: sampler 未返回 prior")
                if sample.prior_bbox != sample.bbox:
                    raise SystemExit(
                        f"[m0 loader-smoke] {case_id}: prior bbox {sample.prior_bbox} != image bbox {sample.bbox}"
                    )
                entry["prior_shape"] = list(sample.zonal_prior.shape)
                entry["prior_bbox_identical"] = True
                entry["prior_max_pz_plus_tz"] = float((sample.zonal_prior[0] + sample.zonal_prior[1]).max())
            samples[f"force_fg={force_fg}"] = entry
        info["sampler"] = samples
        payload["cases"].append(info)
        suffix = ""
        if needs_prior:
            suffix = (
                f" prior={info['prior']['shape']} max(PZ+TZ)={info['prior']['max_pz_plus_tz']:.4f}"
                f" bbox_sync={samples['force_fg=True']['prior_bbox_identical']}"
            )
        print(f"[m0 loader-smoke] {case_id}: data={info['data_shape']} seg={info['seg_shape']} "
              f"labels={info['seg_labels']} fg_sample={samples['force_fg=True']['fg_voxels']}{suffix}")

    provider = TorchBatchProvider(
        store, sampler, batch_size=cfg.data.batch_size, num_batches=2,
        case_ids=case_ids,
        transform=build_train_transforms(
            mirror=cfg.augmentation.mirror,
            mirror_p_per_axis=cfg.augmentation.mirror_p_per_axis,
            per_channel_intensity=cfg.augmentation.per_channel_intensity,
        ),
        seed=cfg.experiment.seed, force_foreground=None,
        num_workers=0, pin_memory=False,  # smoke 用单进程，保证统计精确
        progress=progress, desc="loader-smoke batches",
        include_zonal_prior=needs_prior,
    )
    batch_info = []
    for step, batch in enumerate(provider):
        if needs_prior:
            if len(batch) != 3:
                raise SystemExit(f"[m0 loader-smoke] batch {step}: 期望 (data, seg, prior)，收到 {len(batch)} 元组")
            data, seg, prior = batch
        else:
            if len(batch) != 2:
                raise SystemExit(f"[m0 loader-smoke] batch {step}: 期望 (data, seg)，收到 {len(batch)} 元组")
            data, seg = batch
            prior = None
        entry = {
            "step": step,
            "data_shape": list(data.shape),
            "data_dtype": str(data.dtype),
            "seg_shape": list(seg.shape),
            "seg_dtype": str(seg.dtype),
            "seg_unique": sorted(int(v) for v in set(seg.reshape(-1).tolist())),
            "finite": _all_finite(data, seg),
        }
        if prior is not None:
            if prior.shape[1:] != (2, *tuple(data.shape[-3:])):
                raise SystemExit(
                    f"[m0 loader-smoke] batch {step}: prior 形状 {tuple(prior.shape)} 与 image patch 不匹配"
                )
            entry["prior"] = {
                "shape": list(prior.shape),
                "dtype": str(prior.dtype),
                "finite": _all_finite(prior),
                "min": float(prior.min()),
                "max": float(prior.max()),
                "max_pz_plus_tz": float((prior[:, 0] + prior[:, 1]).max()),
            }
        batch_info.append(entry)
        print(f"[m0 loader-smoke] batch {step}: data={list(data.shape)} seg={list(seg.shape)} "
              f"unique={entry['seg_unique']}"
              + (f" prior={entry['prior']['shape']} max(PZ+TZ)={entry['prior']['max_pz_plus_tz']:.4f}"
                 if prior is not None else ""))
    payload["batches"] = batch_info
    payload["provider_stats"] = provider.stats
    payload["elapsed_sec"] = round(time.time() - t0, 2)

    out_dir = PROJECT_ROOT / "outputs/diagnostics" / str(cfg.experiment.model_id).lower() / (
        args.run_name or cfg.experiment.name
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"loader_smoke_{datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"[m0 loader-smoke] 完成：{len(case_ids)} 例、{len(batch_info)} 个 batch，耗时 {payload['elapsed_sec']}s")
    print(f"[m0 loader-smoke] 结果: {out_path}")
    criteria = "seg_labels ⊆ {0,1}、patch 形状 == plan.patch_size、无 NaN/Inf、通道=3"
    if needs_prior:
        criteria += "、prior=[2,*patch]、max(PZ+TZ)<=1、bbox 与 image 完全一致、manifest 覆盖 train∪val"
    print(f"[m0 loader-smoke] 成功判据：{criteria}")


def run_overfit(args, cfg: M0ExperimentConfig, plan) -> None:
    """命令 C：小样本过拟合（诊断模式；独立输出目录，不与正式训练混用）。"""
    if not args.cases:
        raise SystemExit(
            "[m0 overfit] 必须用 --cases 指定少量（阳性）病例，"
            "例如 --cases 10005_1000005,10021_1000021"
        )
    if args.device == "cpu" and cfg.training.amp:
        print("[m0 overfit][WARN] 请求 AMP 但 device=cpu：trainer 会按 FP32 运行")
    case_ids = [c.strip() for c in args.cases.split(",") if c.strip()]
    run_name = args.run_name or f"overfit_{datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')}"
    base = PROJECT_ROOT / "outputs/diagnostics" / str(cfg.experiment.model_id).lower() / run_name
    if base.exists() and any(base.iterdir()):
        raise SystemExit(f"[m0 overfit] 目标目录已存在内容，拒绝覆盖: {base}（请换 --run-name）")
    progress = resolve_progress(args.no_progress, cfg.training.progress)
    store = build_store(cfg)
    missing = [c for c in case_ids if c not in set(store.available_ids)]
    if missing:
        raise SystemExit(f"[m0 overfit] 病例不在 fold0 的 train/val 中: {missing}")
    # 运行时 GPU 峰值显存遥测：**必须在 build_trainer/model.to(device) 之前** reset 峰值，
    # 否则模型驻留显存会被漏掉；成功、CUDA OOM 与普通异常都落盘同一份 JSON。
    profiler = GpuMemoryProfiler(
        args.device,
        output_path=base / "diagnostics" / GPU_PROFILE_FILENAME,
        metadata={
            "model": {
                "model_id": cfg.experiment.model_id,
                "architecture_version": str(
                    getattr(cfg.architecture, "architecture_version", None)
                ),
                "architecture_sha256": None,
                "num_parameters": None,
            },
            "run": {
                "run_name": run_name,
                "output_dir": str(base),
                "batch_size": cfg.data.batch_size,
                "patch_size": [int(v) for v in cfg.data.patch_size],
                "cases": list(case_ids),
                "epochs": int(args.epochs or cfg.training.max_epochs),
                "train_iterations": int(args.iterations or cfg.training.iterations_per_epoch),
                "validation_iterations": int(
                    args.val_iterations or cfg.training.diagnostic_validation_iterations
                ),
                "validation_mode": VALIDATION_MODE_DIAGNOSTIC,
            },
            "amp_enabled": None,  # 构建后按 trainer 实际开关回填
            "zonal_prior_manifest_sha256": None,
        },
    )
    def _write_report(status: str, *, error: str, exception_type: str) -> Path | None:
        """先读取峰值统计并落盘，再允许释放缓存；返回报告路径（落盘失败返回 None）。"""
        try:
            path = profiler.finalize(status, error=error, exception_type=exception_type)
            print(f"[m0 overfit] GPU 显存报告（{status}, phase={profiler.phase}）: {path}")
        except GpuMemoryProfileError as write_exc:
            path = None
            print(f"[m0 overfit] GPU 显存报告落盘失败（原始异常仍保留）: {write_exc}")
        if path is not None:
            profiler.release_cache()  # 只在报告**成功落盘**之后释放缓存
        return path

    def _finalize_failure(exc: BaseException) -> Path | None:
        status = STATUS_CUDA_OOM if is_cuda_oom(exc) else STATUS_ERROR
        return _write_report(status, error=f"{type(exc).__name__}: {exc}", exception_type=type(exc).__name__)

    trainer = None
    try:
        # initialize_cuda 阶段也在保护范围内：start/reset/mem_get_info/get_device_properties 任一步失败
        # 都会走下面的异常路径，落盘 status=ERROR（phase=initialize_cuda）并以非 0 退出。
        profiler.start()
        if profiler.status == STATUS_ERROR:
            # CUDA requested 但 is_available=False：不得继续 build_trainer
            report_path = _write_report(
                STATUS_ERROR, error=profiler.reason, exception_type="CudaUnavailable"
            )
            raise SystemExit(
                f"[m0 overfit] CUDA 不可用（phase={profiler.phase}）: {profiler.reason}"
                + (f"；显存报告: {report_path}" if report_path else "；显存报告落盘失败")
            )
        profiler.mark_phase(PHASE_BUILD)
        trainer, model, _ = build_trainer(
            cfg, plan,
            device=args.device, progress=progress,
            run_dir=base / "checkpoints", metrics_dir=base / "metrics", diagnostics_dir=base / "diagnostics",
            validation_mode=VALIDATION_MODE_DIAGNOSTIC,
            iterations=args.iterations, val_iterations=args.val_iterations, epochs=args.epochs,
            train_cases=case_ids,
            store=store,
        )
        arch_identity = model_architecture_identity(model)
        profiler.update(
            model={
                "model_id": str(getattr(model, "MODEL_ID", cfg.experiment.model_id)),
                "architecture_sha256": None if arch_identity is None else arch_identity["architecture_sha256"],
                "num_parameters": int(model.num_parameters()),
            },
            amp_enabled=bool(getattr(trainer, "amp", cfg.training.amp)),
            zonal_prior_manifest_sha256=getattr(store, "zonal_prior_manifest_sha256", None),
        )
        profiler.snapshot("after_build")  # 模型/优化器已驻留（model.to(device) 在此前完成）
        print(
            f"[m0 overfit] cases={case_ids} iterations={args.iterations} epochs={args.epochs} "
            f"device={args.device} 验证=诊断随机 patch（不是全体积）"
        )
        # trainer.train() 内部先 _run_epoch(train=True) 再 _run_epoch(train=False)：
        # 二者不可分割，因此只标记一个 trainer_run 阶段（不伪造 train/validation 分界）。
        profiler.mark_phase(PHASE_TRAINER_RUN)
        summary = trainer.train()
        # 峰值是 high-water mark，覆盖 build + train + validation 的全部 kernel 与优化器状态
        profiler.snapshot("after_trainer_run")
    except SystemExit:
        raise
    except KeyboardInterrupt as exc:
        _finalize_failure(exc)
        raise
    except BaseException as exc:
        # 只有真实 trainer 暴露的 current_phase 才允许细分失败阶段；缺失/未知一律记 trainer_run
        if profiler.phase == PHASE_TRAINER_RUN:
            hint = getattr(trainer, "current_phase", None)
            if hint in (PHASE_TRAIN, PHASE_VALIDATION):
                profiler.phase = str(hint)
            else:
                profiler.phase = PHASE_TRAINER_RUN
        report_path = _finalize_failure(exc)
        raise SystemExit(
            f"[m0 overfit] {'CUDA_OOM' if is_cuda_oom(exc) else 'ERROR'}"
            f"（phase={profiler.phase}）: {type(exc).__name__}: {exc}"
            + (f"；显存报告: {report_path}" if report_path else "；显存报告落盘失败")
        ) from exc

    if not args.no_plot:
        plot_loss_curve(base / "metrics" / "epoch_metrics.csv", base / "diagnostics" / "loss_curve.png")
    # CPU 路径保持 NOT_APPLICABLE（不得在收尾时被改写成 COMPLETED）
    final_status = (
        STATUS_NOT_APPLICABLE if profiler.status == STATUS_NOT_APPLICABLE else STATUS_COMPLETED
    )
    report_path = profiler.finalize(final_status)
    report = profiler.to_report()
    if report["process_memory"] is not None:
        peak = report["process_memory"]
        gpu = report["gpu"] or {}
        print(
            f"[m0 overfit] GPU 峰值（当前进程，覆盖整个 trainer_run = build+train+validation；"
            f"不存在独立 train/validation 峰值）: peak_allocated={peak['peak_allocated_gib']} GiB / "
            f"peak_reserved={peak['peak_reserved_gib']} GiB（非整卡占用、非 nvidia-smi）；"
            f"整卡 free {gpu.get('free_gib_at_start')} → {gpu.get('free_gib_at_end')} GiB；报告={report_path}"
        )
    else:
        print(f"[m0 overfit] GPU 遥测: {report['status']}（{report['reason']}）；报告={report_path}")
    print(
        f"[m0 overfit] 结束: status={summary.status} best_{summary.checkpoint_metric}={summary.best_metric} "
        f"输出={base}（诊断结果，不代表模型有效）"
    )


def run_training(args, cfg: M0ExperimentConfig, plan) -> None:
    """默认模式：正式训练（每 every_n_epochs 低频全体积验证 + Dice 选 best；正式协议禁用 early stopping）。"""
    run_name = args.run_name or cfg.experiment.name
    run_dir = cfg.resolve_path(cfg.paths.output_root) / run_name
    model_key = str(cfg.experiment.model_id).lower()  # m0 / m1 / m2 / m3 / m4（M0 目录保持不变）
    metrics_dir = PROJECT_ROOT / "outputs/metrics" / model_key / run_name
    diagnostics_dir = PROJECT_ROOT / "outputs/diagnostics" / model_key / run_name
    # diagnostics 目录同时承载预检产物（setup_check.json / loader_smoke_*.json）→ 仅对"只含预检产物"
    # 的情况放行；任何训练产物仍会拒绝启动。
    ensure_fresh_run_dirs(
        run_dir,
        metrics_dir,
        diagnostics_dir,
        resume_from=args.resume_from,
        preflight_tolerant=(diagnostics_dir,),
    )

    progress = resolve_progress(args.no_progress, cfg.training.progress)
    store = build_store(cfg)
    validator, _ = build_validation_components(
        cfg, plan, store,
        mode=cfg.validation.mode,
        device=args.device,
        progress=progress,
        metrics_dir=metrics_dir,
    )
    trainer, model, _ = build_trainer(
        cfg, plan,
        device=args.device, progress=progress,
        run_dir=run_dir, metrics_dir=metrics_dir, diagnostics_dir=diagnostics_dir,
        validation_mode=cfg.validation.mode,
        validator=validator,
        store=store,
    )
    if args.resume_from:
        arch_identity = model_architecture_identity(model)
        if arch_identity is not None:
            # 加载 state_dict 之前先核对架构身份：legacy PlainConv 或哈希不匹配 → 清楚的拒绝错误
            verify_checkpoint_architecture(args.resume_from, arch_identity)
        trainer.resume(
            args.resume_from,
            allow_mid_epoch=bool(args.allow_mid_epoch_resume),
            allow_protocol_change=bool(args.allow_protocol_change),
        )
    print(
        f"[m0 train] run={run_name} device={args.device} max_epochs={trainer.epochs} "
        f"iters/epoch={trainer.iterations_per_epoch}"
    )
    print(
        f"[m0 train] validation={trainer.validation_mode} checkpoint_metric={trainer.checkpoint_metric} "
        f"(maximize={trainer.maximize}, min_delta={trainer.min_delta}) "
        f"early_stopping={{enabled={trainer.early_stopping.enabled}, min_epochs={trainer.early_stopping.min_epochs}, "
        f"patience={trainer.early_stopping.patience}}}"
    )
    if trainer.validation_mode == VALIDATION_MODE_FULL_VOLUME:
        print(
            f"[m0 train] 全体积验证频率=每 {trainer.validation_every_n_epochs} epoch（最后一个 epoch 必验证）"
            f" case 数={len(store.val_ids)}（冻结 fold 0；阳性/阴性期望见配置）"
        )
    print(f"[m0 train] train_cases={len(store.train_ids)} params={model.num_parameters():,}")
    summary = trainer.train()
    print(
        f"[m0 train] 结束: status={summary.status} epochs={summary.epochs_done} "
        f"validation_events={summary.validation_events_completed} best_epoch={summary.best_epoch} "
        f"best_{summary.checkpoint_metric}={summary.best_metric} "
        f"bad_validation_events={summary.bad_validation_events} early_stopped={summary.early_stopped}"
    )
    if summary.early_stopped:
        print(f"[m0 train] early stopping 原因: {summary.stop_reason}")


def plot_loss_curve(csv_path: Path, png_path: Path) -> None:
    """从 epoch_metrics.csv 画 train loss 与验证曲线（matplotlib 可选）。

    正式全体积 run 的验证字段是 case-wise Dice（无 val_loss），诊断 run 是 val_loss/val_dice；两者都能画。
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[m0 plot] 未安装 matplotlib，跳过曲线（CSV 仍可用于绘图）")
        return
    if not csv_path.is_file():
        print(f"[m0 plot] 找不到 {csv_path}，跳过绘图")
        return
    with open(csv_path) as fh:
        rows = [_r for _r in _csv.DictReader(fh)]
    if not rows:
        return

    def series(key: str, transform=str) -> tuple[list[int], list[float]]:
        values = [(int(r["epoch"]), float(r[key])) for r in rows if r.get(key) not in (None, "", "nan")]
        return [v[0] for v in values], [v[1] for v in values]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x, y = series("train_loss")
    axes[0].plot(x, y, label="train_loss")
    if any(r.get("val_loss") not in (None, "") for r in rows):
        xv, yv = series("val_loss")
        axes[0].plot(xv, yv, label="val_loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()

    val_key = None
    for candidate in ("val_positive_casewise_dice_mean", "val_micro_dice", "val_dice"):
        if any(r.get(candidate) not in (None, "") for r in rows):
            val_key = candidate
            break
    if val_key is not None:
        xd, yd = series(val_key)
        axes[1].plot(xd, yd, label=val_key, color="tab:green")
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("dice")
        axes[1].legend()
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    print(f"[m0 plot] 曲线: {png_path}")


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args)
    plan = load_plan(cfg.resolve_path(cfg.paths.plans), configuration=cfg.model.configuration, dataset_name="Dataset605_PICAI")
    validate_against_plan(cfg, plan, project_root=PROJECT_ROOT, require_data=True)
    set_seed(cfg.experiment.seed, deterministic=False)

    if args.loader_smoke:
        run_loader_smoke(args, cfg, plan)
    elif args.overfit:
        run_overfit(args, cfg, plan)
    else:
        run_training(args, cfg, plan)


if __name__ == "__main__":
    main()
