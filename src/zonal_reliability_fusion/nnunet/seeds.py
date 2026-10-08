"""显式随机种子控制（新主线要求；只控制项目自身能控制的部分）。

为什么需要
----------
项目过去的大量实验是**单次 run**，无法报告 ``mean ± std``，也无法做病例级配对分析。新主线
要求最终候选做 3 个独立 seed 的完整训练，因此必须先把「种子」变成一个显式、可审计、可传入的
参数，而不是依赖进程默认状态。

能控制什么（诚实边界）
----------------------
:func:`apply_explicit_seed` 设置项目自身能控制的部分：

- ``Python random``（含 ``np.random`` 之外的采样辅助）；
- ``NumPy legacy 全局随机状态``——nnU-Net 的 dataloader 与 batchgenerators 使用
  ``np.random`` 全局状态，因此这是**最关键**的一项；
- ``torch`` CPU/GPU 随机状态；
- 模型初始化：由 ``torch`` 种子覆盖（``get_network_from_plans`` 用 torch 的初始化器）。

**不能**保证什么
----------------
nnU-Net v2.6.2 的增强由多进程 ``NonDetMultiThreadedAugmenter`` 驱动，数据读取与增强的完成
顺序在进程间非确定；即使所有随机源都被播种，pipeline 的**非确定性调度**仍然存在。因此：

    explicit seed improves repeatability but does not guarantee bitwise determinism.

本项目**不**声称 bitwise 可复现，也**不**把 ``torch.use_deterministic_algorithms(True)``
当作默认设置：它会显著降低吞吐，而且无法消除多进程 augmenter 的调度顺序差异。真正可复现的
粒度是「完整训练 + validation 流程的统计可比性」，由多 seed 的 ``mean ± std`` 表达。
"""

from __future__ import annotations

import os
import random

#: 新主线默认 seed 列表（用于最终候选的 3 个独立 run）
DEFAULT_SEED_LIST = (20261008, 20261009, 20261010)
#: 未指定 seed 时使用的占位值；``None`` 表示**不碰**任何随机源（保持 nnU-Net 默认行为）
NO_EXPLICIT_SEED = None


def apply_explicit_seed(seed: int, *, verbose: bool = False) -> dict:
    """把显式种子写入 Python / NumPy / torch 的随机源；返回实际设置的内容（可审计）。

    传 ``seed=None`` 时**什么都不做**（返回 ``{"applied": False}``）。这一分支存在的意义是让
    「不加种子」本身也是一个显式、可记录的选择，而不是代码里没有这条路径。
    """
    if seed is None:
        return {"applied": False, "seed": None}
    seed = int(seed)
    if seed < 0:
        raise ValueError(f"seed 必须是非负整数，收到 {seed}")

    random.seed(seed)
    applied: dict = {"applied": True, "seed": seed}
    try:
        import numpy as np

        np.random.seed(seed % (2**32))
        applied["numpy"] = True
    except ImportError:  # pragma: no cover - numpy 是本项目硬依赖
        applied["numpy"] = False
    try:
        import torch

        torch.manual_seed(seed)
        applied["torch"] = True
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            applied["torch_cuda"] = True
        else:
            applied["torch_cuda"] = False
    except ImportError:  # pragma: no cover - torch 是本项目硬依赖
        applied["torch"] = False
    # 让子进程（多进程 augmenter 的 worker）继承同一组环境提示；NumPy 之外的工具链会读它们。
    os.environ["PYTHONHASHSEED"] = str(seed)
    if verbose:
        print(f"[explicit-seed] {applied}")
    return applied


def describe_seed_limitations() -> str:
    """写给 run 配置与论文附录的诚实边界说明（必须原样保留语义）。"""
    return (
        "explicit seed improves repeatability but does not guarantee bitwise determinism: "
        "nnU-Net v2.6.2 drives augmentation with NonDetMultiThreadedAugmenter, whose "
        "inter-process completion order is non-deterministic."
    )


__all__ = (
    "DEFAULT_SEED_LIST",
    "NO_EXPLICIT_SEED",
    "apply_explicit_seed",
    "describe_seed_limitations",
)
