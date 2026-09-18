"""把「预处理数据 + 采样器 + 增强」组合成 torch batch 迭代器（训练/验证共用）。

设计要点（P2-A 复审后修订）：

1. **多进程与 pinned memory**：真正的数据加载走 `torch.utils.data.DataLoader`
   （`num_workers` / `pin_memory` 来自配置，不再是被忽略的摆设）；
2. **前景请求与官方一致**：每个 batch 内**固定** `bs - round(bs * (1 - oversample))` 个样本强制前景
   （官方 `nnUNetDataLoader._oversample_last_XX_percent` 语义；bs=2、oversample=0.33 → 每批恰好 1/2），
   而不是"每个样本独立 33% 概率"；验证集固定 `force_foreground=False`；
3. **可复现采样且无隐藏状态**：每个样本的随机种子由 `(seed, epoch, index)` 稳定混合得到，
   因此采样序列与 worker 数量/顺序无关，resume 时只要 epoch 正确即完全可复现（不需要保存 RNG 状态）；
   训练器每轮调用 `set_epoch(epoch)`；
4. 输出 `(data [B, C, *patch] float32, seg [B, 1, *patch] float32)`；不做 device 迁移（trainer 负责）。

线程/进程注意：`num_workers > 0` 依赖 Linux 默认的 fork 启动方式；数据只读，worker 各自打开 `.b2nd`。
调试或需要精确统计回退次数时请用 `num_workers=0`（多进程下 `foreground_fallbacks` 无法跨进程汇总，记为 None）。
"""
from __future__ import annotations

from typing import Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..utils.progress import make_progress
from .patch_sampler import PatchSampler
from .preprocessed_store import PreprocessedStore
from .transforms import AugmentationPipeline

DEFAULT_TRAIN_DESC = "train batches"
DEFAULT_VAL_DESC = "val batches"


class BatchProviderError(RuntimeError):
    """数据源不可用或采样失败（不静默跳过）。"""


def _validate_prior_tensor(prior: torch.Tensor, *, case_id: str = "") -> None:
    """CPU 数据入口的 prior 契约校验（热路径之外，一次性、无 GPU 同步）。

    检查：shape [2,D,H,W]、有限、逐通道 [0,1]、PZ+TZ <= 1+tol。
    """
    from .preprocessed_store import ZONAL_PRIOR_CHANNELS, ZONAL_PRIOR_TOL

    if prior.ndim != 4 or prior.shape[0] != ZONAL_PRIOR_CHANNELS:
        raise BatchProviderError(
            f"{case_id}: prior patch 形状必须为 [{ZONAL_PRIOR_CHANNELS},D,H,W]，收到 {tuple(prior.shape)}"
        )
    if not bool(torch.isfinite(prior).all()):
        raise BatchProviderError(f"{case_id}: prior patch 含非有限值")
    lo = float(prior.min())
    hi = float(prior.max())
    if lo < -ZONAL_PRIOR_TOL or hi > 1.0 + ZONAL_PRIOR_TOL:
        raise BatchProviderError(f"{case_id}: prior patch 取值越界 [{lo}, {hi}]（必须为 [0,1]）")
    overlap = float((prior[0] + prior[1]).max())
    if overlap > 1.0 + ZONAL_PRIOR_TOL:
        raise BatchProviderError(
            f"{case_id}: prior patch 存在 PZ/TZ 重叠（max={overlap:.6f} > 1）"
        )


def mix_seed(*values: int) -> int:
    """把若干整数稳定混合为 32 位种子（不依赖 Python `hash()`，跨进程/跨机器一致）。"""
    mixed = 0x9E3779B9
    for value in values:
        mixed = (mixed ^ (int(value) + 0x9E3779B9 + (mixed << 6) + (mixed >> 2))) & 0xFFFFFFFF
    return mixed % (2**32)


class PatchDataset(Dataset):
    """map-style 数据集：index → (data patch, seg patch)。

    index 的语义：`batch_index, position = divmod(index, batch_size)`，
    `position` 用于决定该样本是否强制前景（与官方按 batch 内位置取样的行为一致）。
    """

    def __init__(
        self,
        store: PreprocessedStore,
        sampler: PatchSampler,
        *,
        case_ids: Sequence[str],
        transform: AugmentationPipeline | None,
        seed: int,
        batch_size: int,
        force_foreground: bool | None,
        length: int,
        include_zonal_prior: bool = False,
    ) -> None:
        self.store = store
        self.sampler = sampler
        self.case_ids = tuple(case_ids)
        self.transform = transform
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.force_foreground = force_foreground
        self.length = int(length)
        self.include_zonal_prior = bool(include_zonal_prior)
        self.epoch = 0
        self.foreground_fallbacks = 0  # 仅 num_workers=0 时准确
        if not self.case_ids:
            raise BatchProviderError("case_ids 为空，无法构建数据集")
        if self.include_zonal_prior and getattr(store, "zonal_prior_root", None) is None:
            raise BatchProviderError(
                "include_zonal_prior=True 但 store 未配置 zonal_prior_root/zonal_prior_source；"
                "禁止用全零 prior 兜底"
            )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.length

    def foreground_per_batch(self) -> int:
        """每个 batch 强制前景的样本数（官方语义）。"""
        if self.force_foreground is True:
            return self.batch_size
        if self.force_foreground is False:
            return 0
        return self.batch_size - int(round(self.batch_size * (1 - self.sampler.oversample_foreground)))

    def __getitem__(self, index: int) -> tuple:
        """返回 `(data, seg)`（M0–M2）或 `(data, seg, zonal_prior)`（M3/M4）。"""
        _, position = divmod(int(index), self.batch_size)
        rng = np.random.default_rng(mix_seed(self.seed, self.epoch, index))
        force_fg = self.force_foreground
        if force_fg is None:
            # 官方语义：batch 内位置 >= round(bs * (1 - oversample)) 的样本强制前景
            force_fg = position >= (self.batch_size - self.foreground_per_batch())
        case_id = str(rng.choice(self.case_ids))
        # 仅在需要 prior 时传 kwarg：保持与只实现 get_case(case_id) 的 store（含测试假 store）兼容
        case = (
            self.store.get_case(case_id, include_zonal_prior=True)
            if self.include_zonal_prior
            else self.store.get_case(case_id)
        )
        sample = self.sampler.sample(
            case.data,
            case.seg,
            zonal_prior=case.zonal_prior,
            case_id=case_id,
            properties=case.properties,
            force_fg=bool(force_fg),
            rng=rng,
        )
        if sample.foreground_fallback:
            self.foreground_fallbacks += 1
        if self.include_zonal_prior and sample.zonal_prior is None:
            raise BatchProviderError(
                f"{case_id}: include_zonal_prior=True 但 store 未返回 prior（禁止用全零 prior 兜底）"
            )
        data, seg, prior = sample.data, sample.seg, sample.zonal_prior
        if self.transform is not None and not self.transform.is_noop:
            if prior is None:
                data, seg = self.transform.apply(data, seg, rng)
            else:
                data, seg, prior = self.transform.apply(data, seg, rng, prior)
        data_t = torch.from_numpy(np.ascontiguousarray(data)).to(torch.float32)
        seg_t = torch.from_numpy(np.ascontiguousarray(seg)).to(torch.float32)
        if prior is None:
            return data_t, seg_t
        prior_t = torch.from_numpy(np.ascontiguousarray(prior)).to(torch.float32)
        _validate_prior_tensor(prior_t, case_id=case_id)
        return data_t, seg_t, prior_t


class TorchBatchProvider:
    """生成固定数量的 batch（训练/验证共用；每次 `__iter__` 为一轮新的随机采样）。"""

    def __init__(
        self,
        store: PreprocessedStore,
        sampler: PatchSampler,
        *,
        batch_size: int,
        num_batches: int,
        case_ids: Sequence[str] | None = None,
        transform: AugmentationPipeline | None = None,
        seed: int = 0,
        force_foreground: bool | None = None,
        num_workers: int = 0,
        pin_memory: bool = False,
        progress: bool = False,
        desc: str = DEFAULT_TRAIN_DESC,
        include_zonal_prior: bool = False,
    ) -> None:
        if int(batch_size) < 1 or int(num_batches) < 1:
            raise ValueError(f"batch_size / num_batches 必须 >= 1，收到 {batch_size} / {num_batches}")
        if int(num_workers) < 0:
            raise ValueError(f"num_workers 必须 >= 0，收到 {num_workers}")
        self.store = store
        self.sampler = sampler
        self.batch_size = int(batch_size)
        self.num_batches = int(num_batches)
        self.case_ids = tuple(case_ids) if case_ids is not None else tuple(store.available_ids)
        if not self.case_ids:
            raise BatchProviderError("case_ids 为空，无法构建 batch")
        self.transform = transform
        self.force_foreground = force_foreground
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.progress = bool(progress)
        self.desc = desc
        self.seed = int(seed)
        self.include_zonal_prior = bool(include_zonal_prior)
        if self.include_zonal_prior and getattr(store, "zonal_prior_root", None) is None:
            raise BatchProviderError(
                "include_zonal_prior=True 但 store 未配置 zonal_prior_root/zonal_prior_source"
            )
        self.epoch = 0
        self.stats = {
            "batches": 0,
            "samples": 0,
            "foreground_requested": 0,
            "foreground_fallbacks": 0,
            "num_workers": self.num_workers,
        }

    def __len__(self) -> int:
        return self.num_batches

    def set_epoch(self, epoch: int) -> None:
        """训练器在每轮 epoch 开始时调用，保证采样序列随 epoch 变化且可复现。"""
        self.epoch = int(epoch)

    # ------------------------------------------------------------------ 内部
    def foreground_per_batch(self) -> int:
        return self._make_dataset().foreground_per_batch()

    def _make_dataset(self) -> PatchDataset:
        return PatchDataset(
            self.store,
            self.sampler,
            case_ids=self.case_ids,
            transform=self.transform,
            seed=self.seed,
            batch_size=self.batch_size,
            force_foreground=self.force_foreground,
            length=self.num_batches * self.batch_size,
            include_zonal_prior=self.include_zonal_prior,
        )

    # ------------------------------------------------------------------ 迭代
    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        dataset = self._make_dataset()
        dataset.set_epoch(self.epoch)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            persistent_workers=False,
        )
        bar = make_progress(loader, total=len(loader), desc=f"{self.desc} (e{self.epoch})", disable=not self.progress)
        n_batches = 0
        n_samples = 0
        for batch in bar:
            data = batch[0]
            if not torch.isfinite(data).all():
                raise BatchProviderError("采样结果包含 NaN/Inf")
            n_batches += 1
            n_samples += int(data.shape[0])
            yield batch

        self.stats["batches"] += n_batches
        self.stats["samples"] += n_samples
        self.stats["foreground_requested"] += n_batches * dataset.foreground_per_batch()
        if self.num_workers == 0:
            self.stats["foreground_fallbacks"] += int(dataset.foreground_fallbacks)
            self.stats["foreground_fallbacks_is_exact"] = True
        else:
            self.stats.setdefault("foreground_fallbacks_is_exact", False)
        if n_batches != self.num_batches:
            raise BatchProviderError(
                f"batch 数量不足：期望 {self.num_batches}，实际 {n_batches}；请检查 num_batches / drop_last 设置"
            )

    def describe(self) -> dict:
        """写入 run manifest 的数据管线描述。"""
        return {
            "num_batches": self.num_batches,
            "batch_size": self.batch_size,
            "n_cases": len(self.case_ids),
            "patch_size": list(self.sampler.patch_size),
            "oversample_foreground": self.sampler.oversample_foreground,
            "force_foreground": self.force_foreground,
            "foreground_per_batch": self.foreground_per_batch(),
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "sampling": "deterministic (seed, epoch, index)",
            "zonal_prior": (
                getattr(self.store, "describe_zonal_prior", lambda: {"enabled": False})()
                if self.include_zonal_prior
                else {"enabled": False}
            ),
            "augmentation": None
            if self.transform is None
            else {
                "spatial": [type(t).__name__ for t in self.transform.spatial],
                "intensity": [type(t).__name__ for t in self.transform.intensity],
            },
        }


__all__ = ["BatchProviderError", "PatchDataset", "TorchBatchProvider", "mix_seed"]
