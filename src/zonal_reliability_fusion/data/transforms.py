"""数据增强（P2-A 阶段：同步镜像 + 明确的接口占位）。

research_plan §9.1 要求空间增强同步作用于序列与标签；M0 不得加入 modality corruption /
modality dropout / 域随机化。

本文件当前状态（诚实边界，详见 docs/P2_M0_Implementation.md）：
- **已实现并测试**：随机镜像（三轴独立、影像/标签同步）；
- **已实现（默认关闭）**：分序列强度增强的最小安全实现（乘法 scale + 加法 offset）；
- **仅保留接口（调用即报错）**：rotation / scaling / noise / blur / gamma —— 本轮不实现
  未经测试的 3D 空间变换，避免与 nnU-Net 增强行为不可控地漂移。

验证集使用 `build_validation_transforms()`（显式空管道，不做任何随机增强）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np

SPATIAL_AXES = (0, 1, 2)  # D, H, W


class SpatialTransform(Protocol):
    """空间变换：必须**同步**作用于 data、seg 与可选 prior。

    返回 `(data, seg)`（未提供 prior）或 `(data, seg, prior)`（提供 prior）。
    实现必须在同一随机决策下对三者施加完全相同的空间操作（含镜像/翻转等）。
    """

    def __call__(
        self,
        data: np.ndarray,
        seg: np.ndarray,
        rng: np.random.Generator,
        prior: np.ndarray | None = None,
    ) -> tuple:
        ...


class IntensityTransform(Protocol):
    """强度变换：只作用于 data（[C, D, H, W]），分序列（逐通道）独立应用。"""

    def __call__(self, data: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        ...


@dataclass
class MirrorAugmentor:
    """随机镜像（空间同步）。

    Args:
        axes: 作用的空间轴（0=D, 1=H, 2=W）。
        p_per_axis: 每个轴独立翻转的概率。
    """

    axes: Sequence[int] = SPATIAL_AXES
    p_per_axis: float = 0.5

    def __post_init__(self) -> None:
        self.axes = tuple(int(a) for a in self.axes)
        if any(a not in (0, 1, 2) for a in self.axes):
            raise ValueError(f"axes 只能取 0/1/2，收到 {self.axes}")
        if not (0.0 <= float(self.p_per_axis) <= 1.0):
            raise ValueError(f"p_per_axis 必须在 [0,1]，收到 {self.p_per_axis}")

    def __call__(
        self,
        data: np.ndarray,
        seg: np.ndarray,
        rng: np.random.Generator,
        prior: np.ndarray | None = None,
    ) -> tuple:
        for axis in self.axes:
            if rng.random() < self.p_per_axis:
                data = np.flip(data, axis=axis + 1)  # +1: 跳过通道维
                seg = np.flip(seg, axis=axis + 1)
                if prior is not None:
                    prior = np.flip(prior, axis=axis + 1)
        out = (np.ascontiguousarray(data), np.ascontiguousarray(seg))
        if prior is None:
            return out
        return (*out, np.ascontiguousarray(prior))


@dataclass
class PerChannelIntensityAugmentor:
    """分序列强度增强（逐通道独立；z-score 空间下的安全变换）。

    注意：这是**最小实现 + 接口占位**，不等同于 nnU-Net 的 intensity augmentation
    （gamma / noise / blur / bias field）。默认关闭（enabled=False）。
    """

    enabled: bool = False
    p_per_channel: float = 0.1
    scale_range: tuple[float, float] = (0.9, 1.1)
    offset_range: tuple[float, float] = (-0.1, 0.1)

    def __post_init__(self) -> None:
        if not (0.0 <= float(self.p_per_channel) <= 1.0):
            raise ValueError(f"p_per_channel 必须在 [0,1]，收到 {self.p_per_channel}")
        for name, rng_range in (("scale_range", self.scale_range), ("offset_range", self.offset_range)):
            if len(tuple(rng_range)) != 2 or float(rng_range[0]) > float(rng_range[1]):
                raise ValueError(f"{name} 必须为 (low, high) 且 low <= high，收到 {rng_range}")

    def __call__(self, data: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if not self.enabled:
            return data
        out = data.astype(np.float32, copy=True)
        for c in range(out.shape[0]):
            if rng.random() < self.p_per_channel:
                scale = float(rng.uniform(*self.scale_range))
                offset = float(rng.uniform(*self.offset_range))
                out[c] = out[c] * scale + offset
        return out


class UnavailableAugmentor:
    """未实现的增强接口占位：调用即抛出，明确指出本轮不实现（避免假象对齐）。"""

    def __init__(self, name: str, reason: str = "") -> None:
        self.name = name
        self.reason = reason or "P2-A 不实现未经测试的 3D 空间/强度变换"

    def __call__(self, *_args, **_kwargs):  # pragma: no cover - 仅作为显式占位
        raise NotImplementedError(
            f"{self.name} 尚未实现：{self.reason}；"
            "见 docs/P2_M0_Implementation.md「augmentation 现状与暂存差异」"
        )


@dataclass
class AugmentationPipeline:
    """空间增强（同步）+ 强度增强（逐通道）的组合。"""

    spatial: list = field(default_factory=list)
    intensity: list = field(default_factory=list)

    def apply(
        self,
        data: np.ndarray,
        seg: np.ndarray,
        rng: np.random.Generator,
        prior: np.ndarray | None = None,
    ) -> tuple:
        """空间增强同步作用于 data/seg/prior；**强度增强只作用于 data**（不碰 seg/prior）。

        返回 `(data, seg)`（未提供 prior）或 `(data, seg, prior)`（提供 prior）。
        """
        for transform in self.spatial:
            if prior is None:
                data, seg = transform(data, seg, rng)
            else:
                data, seg, prior = transform(data, seg, rng, prior)
        for transform in self.intensity:
            data = transform(data, rng)
        if prior is None:
            return data, seg
        return data, seg, prior

    @property
    def is_noop(self) -> bool:
        return not self.spatial and not self.intensity


def build_train_transforms(
    *,
    mirror: bool = True,
    mirror_p_per_axis: float = 0.5,
    per_channel_intensity: bool = False,
) -> AugmentationPipeline:
    """训练增强：当前 = 同步镜像（+ 可选的分序列强度接口，默认关闭）。

    未实现项（rotation/scaling/noise/blur/gamma）不在管道内，配置层会以
    `pending_parity` 显式列出，防止被误认为"已与 nnU-Net 对齐"。
    """
    spatial = [MirrorAugmentor(axes=SPATIAL_AXES, p_per_axis=mirror_p_per_axis)] if mirror else []
    intensity = [PerChannelIntensityAugmentor(enabled=bool(per_channel_intensity))]
    return AugmentationPipeline(spatial=spatial, intensity=intensity)


def build_validation_transforms() -> AugmentationPipeline:
    """验证增强：显式空管道（无任何随机增强）。"""
    return AugmentationPipeline(spatial=[], intensity=[])


__all__ = [
    "SPATIAL_AXES",
    "AugmentationPipeline",
    "MirrorAugmentor",
    "PerChannelIntensityAugmentor",
    "UnavailableAugmentor",
    "build_train_transforms",
    "build_validation_transforms",
]
