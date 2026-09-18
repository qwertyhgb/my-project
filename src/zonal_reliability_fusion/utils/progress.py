"""进度显示工具（AGENTS.md §1）。

约定：
- 训练/验证/推理/数据处理默认显示进度条，可通过 `--no-progress` 关闭；
- 分布式时只有 rank 0 显示，避免重复输出；
- 进度条不替代结构化日志：调用方仍需输出成功/失败计数、耗时与输出路径。
"""
from __future__ import annotations

from typing import Iterable, Iterator, TypeVar

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore[assignment]

T = TypeVar("T")


class NullProgress(Iterator[T]):
    """`--no-progress` / 非主进程时的空进度条（接口与 tqdm 兼容）。"""

    def __init__(self, iterable: Iterable[T] | None = None) -> None:
        self._iterator = iter(iterable) if iterable is not None else iter(())

    def __iter__(self) -> "NullProgress[T]":
        return self

    def __next__(self) -> T:
        return next(self._iterator)

    def update(self, n: int = 1) -> None:  # noqa: D401 - 兼容接口
        return None

    def set_postfix(self, *args, **kwargs) -> None:  # noqa: D401
        return None

    def set_description(self, *args, **kwargs) -> None:  # noqa: D401
        return None

    def close(self) -> None:
        return None


def is_main_process(rank: int | None = None) -> bool:
    """是否为主进程（rank 0）。

    `rank` 显式给定时直接使用（便于单测）；否则查询 torch.distributed 状态。
    """
    if rank is not None:
        return rank == 0
    try:  # 仅在真正使用分布式时才有意义
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
    except Exception:  # pragma: no cover - 防御性
        pass
    return True


def make_progress(
    iterable: Iterable[T] | None = None,
    *,
    total: int | None = None,
    desc: str = "",
    disable: bool = False,
    rank: int | None = None,
    unit: str = "it",
    leave: bool = False,
):
    """创建进度条（不可用时退化为空进度条）。"""
    show = (not disable) and is_main_process(rank)
    if not show or tqdm is None:
        return NullProgress(iterable)
    bar = tqdm(iterable, total=total, desc=desc, disable=False, mininterval=1.0, unit=unit, leave=leave)
    return bar


def resolve_progress(cli_no_progress: bool, config_progress: bool = True) -> bool:
    """合并 CLI `--no-progress` 与配置项：任一要求关闭则关闭。"""
    return bool(config_progress) and not bool(cli_no_progress)


__all__ = ["NullProgress", "is_main_process", "make_progress", "resolve_progress"]
