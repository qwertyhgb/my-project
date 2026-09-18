"""M0–M4 共用的 patch 采样器（训练/验证同一实现）。

约定（research_plan §9.3）：
- patch size 来自冻结 plan（3d_fullres: (16, 320, 320)）；
- 训练时按 `oversample_foreground` 概率强制包含前景，前景位置优先取
  properties['class_locations']（preprocessor 已写入 .pkl），否则回退为从 seg 现算；
- 阴性病例（无前景）必须安全回退为随机 crop，绝不报错、绝不伪造前景；
- **验证集不启用**随机前景过采样（force_fg=False）；
- 影像与标签空间变换完全同步（同一 bbox）；
- 影像小于 patch 时显式 padding（data=0.0，seg=0），并保持两者同步。

本模块不读取文件、不依赖第三方库，全部为纯 numpy 逻辑，便于合成单元测试。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

DATA_PAD_VALUE = 0.0
SEG_PAD_VALUE = 0
FOREGROUND_LABEL = 1


@dataclass(frozen=True)
class PatchSample:
    """一次采样的结果（数组形状严格等于 patch_size）。"""

    case_id: str
    data: np.ndarray  # [C, *patch]
    seg: np.ndarray  # [1, *patch]
    bbox: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    used_foreground: bool
    foreground_fallback: bool  # True = 请求了前景但该病例没有前景，已安全回退为随机 crop
    zonal_prior: np.ndarray | None = None  # [2, *patch]（PZ/TZ）；仅 M3/M4 使用
    prior_bbox: tuple[tuple[int, int], tuple[int, int], tuple[int, int]] | None = None  # 必须与 bbox 相同


class PatchSampler:
    """随机 / 前景过采样 patch 采样器。"""

    def __init__(
        self,
        patch_size: tuple[int, int, int] = (16, 320, 320),
        *,
        oversample_foreground: float = 0.33,
        seed: int = 0,
        foreground_label: int = FOREGROUND_LABEL,
        data_pad_value: float = DATA_PAD_VALUE,
        seg_pad_value: int = SEG_PAD_VALUE,
    ) -> None:
        if len(patch_size) != 3 or any(int(v) < 1 for v in patch_size):
            raise ValueError(f"patch_size 必须为长度 3 的正整数，收到 {patch_size}")
        if not (0.0 <= float(oversample_foreground) <= 1.0):
            raise ValueError(f"oversample_foreground 必须在 [0,1]，收到 {oversample_foreground}")
        self.patch_size = tuple(int(v) for v in patch_size)
        self.oversample_foreground = float(oversample_foreground)
        self.foreground_label = int(foreground_label)
        self.data_pad_value = float(data_pad_value)
        self.seg_pad_value = int(seg_pad_value)
        self.rng = np.random.default_rng(int(seed))

    # ------------------------------------------------------------ 采样决策
    def should_force_foreground(self, rng: np.random.Generator | None = None) -> bool:
        rng = rng or self.rng
        return bool(rng.random() < self.oversample_foreground)

    # ------------------------------------------------------------ 主体
    def sample(
        self,
        data: np.ndarray,
        seg: np.ndarray,
        *,
        zonal_prior: np.ndarray | None = None,
        case_id: str = "",
        properties: Mapping[str, object] | None = None,
        force_fg: bool | None = None,
        rng: np.random.Generator | None = None,
    ) -> PatchSample:
        """采样一个 patch。

        Args:
            data: [C, D, H, W]
            seg: [1, D, H, W] 或 [D, H, W]，取值 {0,1}
            zonal_prior: 可选 [2, D, H, W]（PZ/TZ）；与 data/seg 使用**同一 bbox 与 padding**
            properties: 可含 'class_locations'（{label: array(n,3)}）
            force_fg: None → 按 oversample_foreground 概率随机决定；False → 永远随机 crop。
        """
        rng = rng or self.rng
        data = np.asarray(data)
        seg = np.asarray(seg)
        if data.ndim != 4:
            raise ValueError(f"data 必须为 [C,D,H,W]，收到 {data.shape}")
        if seg.ndim == 3:
            seg = seg[None]
        if seg.ndim != 4 or seg.shape[0] != 1:
            raise ValueError(f"seg 必须为 [1,D,H,W] 或 [D,H,W]，收到 {seg.shape}")
        if data.shape[-3:] != seg.shape[-3:]:
            raise ValueError(f"data {data.shape[-3:]} 与 seg {seg.shape[-3:]} 空间尺寸不一致")

        prior = None if zonal_prior is None else np.asarray(zonal_prior)
        if prior is not None:
            if prior.ndim != 4 or prior.shape[0] != 2:
                raise ValueError(f"zonal_prior 必须为 [2,D,H,W]（PZ/TZ），收到 {prior.shape}")
            if prior.shape[-3:] != data.shape[-3:]:
                raise ValueError(
                    f"zonal_prior 空间尺寸 {prior.shape[-3:]} 与 data {data.shape[-3:]} 不一致；"
                    "裁剪与增强必须同步"
                )
            prior = np.ascontiguousarray(prior.astype(np.float32, copy=False))

        if force_fg is None:
            force_fg = self.should_force_foreground(rng)

        # data / seg / prior 使用完全相同的 padding（见 _pad_to_at_least_patch）
        data, seg, prior = self._pad_to_at_least_patch(data, seg, prior)
        spatial = data.shape[-3:]

        used_foreground = False
        fallback = False
        fg_voxel = self._sample_foreground_voxel(seg, properties, rng) if force_fg else None
        if force_fg and fg_voxel is None:
            fallback = True  # 阴性病例：安全回退为随机 crop

        starts = []
        for d in range(3):
            span = spatial[d] - self.patch_size[d]
            if span < 0:  # 不应发生（已 padding）
                raise RuntimeError(f"dim{d}: 空间尺寸 {spatial[d]} < patch {self.patch_size[d]}")
            if fg_voxel is not None:
                # 与官方一致：以前景体素为中心取 patch（`voxel - patch//2`），再夹到合法范围内。
                # 官方在越界时用 padding 补齐；我们改为平移 bbox（仍包含该体素，且不引入 padding 伪影）。
                start = int(fg_voxel[d]) - self.patch_size[d] // 2
                start = max(0, min(start, span))
            else:
                start = int(rng.integers(0, span + 1))
            starts.append(start)
        used_foreground = fg_voxel is not None

        # image / seg / prior 使用**完全相同**的 bbox
        slices = tuple(slice(starts[d], starts[d] + self.patch_size[d]) for d in range(3))
        data_patch = data[(slice(None), *slices)]
        seg_patch = seg[(slice(None), *slices)]
        prior_patch = None if prior is None else prior[(slice(None), *slices)]
        bbox = tuple((starts[d], starts[d] + self.patch_size[d]) for d in range(3))
        return PatchSample(
            case_id=case_id,
            data=np.ascontiguousarray(data_patch),
            seg=np.ascontiguousarray(seg_patch),
            bbox=bbox,
            used_foreground=used_foreground,
            foreground_fallback=fallback,
            zonal_prior=None if prior_patch is None else np.ascontiguousarray(prior_patch),
            prior_bbox=None if prior_patch is None else bbox,
        )

    # ------------------------------------------------------------ 内部工具
    def _pad_to_at_least_patch(
        self, data: np.ndarray, seg: np.ndarray, prior: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """（内部）返回三元组；prior 为 None 时第三个元素为 None。"""
        """在高端（末尾）padding，使各空间维 >= patch（影像/标签/prior 同步、显式 pad 值）。

        prior 的 padding 值为 0.0（等价于 background occupancy，不引入虚假 PZ/TZ）。
        """
        pads = [(0, 0)]
        for d in range(3):
            deficit = self.patch_size[d] - data.shape[-3 + d]
            pads.append((0, deficit) if deficit > 0 else (0, 0))
        if any(p[1] > 0 for p in pads[1:]):
            data = np.pad(data, pads, mode="constant", constant_values=self.data_pad_value)
            seg = np.pad(seg, pads, mode="constant", constant_values=self.seg_pad_value)
            if prior is not None:
                prior = np.pad(prior, pads, mode="constant", constant_values=0.0)
        return data, seg, prior

    def _sample_foreground_voxel(
        self,
        seg: np.ndarray,
        properties: Mapping[str, object] | None,
        rng: np.random.Generator,
    ) -> tuple[int, int, int] | None:
        """返回一个前景体素坐标；没有前景时返回 None（调用方负责安全回退）。

        `properties` 可以是 dict（含 'class_locations'）或 `CaseProperties` 等带
        `class_locations` 属性的对象。
        """
        class_locations = None
        if properties is not None:
            if isinstance(properties, Mapping):
                class_locations = properties.get("class_locations")
            else:
                class_locations = getattr(properties, "class_locations", None)
        if isinstance(class_locations, Mapping):
            locations = class_locations.get(self.foreground_label)
            arr = np.asarray(locations) if locations is not None else None
            if arr is not None and np.asarray(arr).size:
                coords = self.normalize_class_locations(arr)
                idx = int(rng.integers(0, coords.shape[0]))
                return tuple(int(v) for v in coords[idx])  # type: ignore[return-value]
        fg = np.argwhere(seg[0] == self.foreground_label)
        if fg.size == 0:
            return None
        idx = int(rng.integers(0, fg.shape[0]))
        return tuple(int(v) for v in fg[idx])  # type: ignore[return-value]

    @staticmethod
    def normalize_class_locations(arr: np.ndarray) -> np.ndarray:
        """把 `properties['class_locations'][label]` 规范为 (n, 3) 空间坐标。

        官方 preprocessor 对 `[1, D, H, W]` 标签做 `np.argwhere`，保存的是 **(n, 4)**：
        第 0 列是标签/通道索引，后 3 列才是空间坐标（官方 DataLoader 用 `selected_voxel[i + 1]`）。
        同时接受 (n, 3)（纯空间坐标）以兼容其他来源。
        """
        arr = np.asarray(arr)
        if arr.ndim != 2 or arr.shape[1] not in (3, 4):
            raise ValueError(f"class_locations 形状应为 (n,3)（空间坐标）或 (n,4)（官方格式），收到 {arr.shape}")
        return arr[:, -3:].astype(np.int64, copy=False)


__all__ = ["DATA_PAD_VALUE", "FOREGROUND_LABEL", "SEG_PAD_VALUE", "PatchSample", "PatchSampler"]
