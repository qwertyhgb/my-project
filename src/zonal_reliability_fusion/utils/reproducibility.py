"""可复现性工具（seed、RNG、环境快照）。

注意：
- 本模块**不初始化 CUDA**（不调用 torch.cuda.*）；
- `environment_info()` 默认不查询 CUDA 可用性，避免在被 GPU 占用的机器上产生额外副作用。
"""
from __future__ import annotations

import platform
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np


def set_seed(seed: int, *, deterministic: bool = False) -> dict[str, Any]:
    """设置 python / numpy / torch 随机种子，返回状态快照（写入日志用）。

    Args:
        seed: 全局随机种子。
        deterministic: 为 True 时尝试设置 cudnn deterministic（仅在使用 CUDA 时生效）。
    """
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    snapshot: dict[str, Any] = {"seed": int(seed), "deterministic": bool(deterministic), "torch": None}
    try:
        import torch

        torch.manual_seed(int(seed))
        snapshot["torch"] = torch.__version__
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:  # pragma: no cover - torch 为项目标准依赖
        pass
    return snapshot


def get_rng(seed: int | None = None) -> np.random.Generator:
    """独立 numpy Generator（不污染全局随机状态，便于数据管线复现）。"""
    return np.random.default_rng(None if seed is None else int(seed))


def seed_worker(worker_id: int, base_seed: int = 12345) -> None:  # pragma: no cover - 供 DataLoader 使用
    """torch DataLoader 的 worker_init_fn：让每个 worker 的 numpy / python 种子可复现。"""
    worker_seed = (int(base_seed) + int(worker_id)) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def capture_rng_states() -> dict[str, Any]:
    """捕获 python / numpy / torch RNG 状态（checkpoint 断点续训用）。"""
    states: dict[str, Any] = {"python": random.getstate(), "numpy": np.random.get_state()}
    try:
        import torch

        states["torch"] = torch.get_rng_state()
    except ImportError:  # pragma: no cover
        states["torch"] = None
    return states


def restore_rng_states(states: dict[str, Any]) -> None:
    """恢复 RNG 状态；键缺失时静默跳过（旧 checkpoint 兼容）。"""
    if not states:
        return
    if states.get("python") is not None:
        random.setstate(states["python"])
    if states.get("numpy") is not None:
        np.random.set_state(states["numpy"])
    if states.get("torch") is not None:
        try:
            import torch

            torch.set_rng_state(states["torch"])
        except ImportError:  # pragma: no cover
            pass


def file_sha256(path: str | Path) -> str:
    """文件 SHA256（用于 plan / split 的可追溯校验值）。"""
    import hashlib
    from pathlib import Path as _Path

    p = _Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"文件不存在: {p}")
    digest = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_version_info(project_root: str | Path | None = None) -> dict[str, Any]:
    """代码版本信息（git commit / dirty 状态）；无 git 仓库时明确记录 unavailable。"""
    import subprocess
    from pathlib import Path as _Path

    root = str(project_root) if project_root is not None else None
    info: dict[str, Any] = {"git": "unavailable", "project_root": root}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=10, check=False
        )
        if commit.returncode != 0:
            return info
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, timeout=10, check=False
        )
        info["git"] = commit.stdout.strip()
        info["dirty"] = bool(status.stdout.strip())
    except (FileNotFoundError, subprocess.SubprocessError):  # pragma: no cover - 环境相关
        info["git"] = "unavailable"
    return info


def environment_info(*, query_cuda: bool = False) -> dict[str, Any]:
    """环境快照（写入 run 目录，保证可追溯）。默认不查询 CUDA。"""
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    try:
        import torch

        info["torch"] = torch.__version__
        if query_cuda:
            info["cuda_available"] = bool(torch.cuda.is_available())
            info["cuda_version"] = getattr(torch.version, "cuda", None)
    except ImportError:  # pragma: no cover
        info["torch"] = None
    return info


__all__ = [
    "capture_rng_states",
    "code_version_info",
    "environment_info",
    "file_sha256",
    "get_rng",
    "restore_rng_states",
    "seed_worker",
    "set_seed",
]
