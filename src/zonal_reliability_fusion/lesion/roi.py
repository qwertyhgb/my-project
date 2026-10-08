"""Anatomy-Guided Prostate ROI（新主线条件 B：ROI 消融）。

它在回答一个很朴素的问题：**单纯减少无关背景，是否已经能改善 lesion learning？**

设计约束（为什么不是 ``image * wg_mask``）
------------------------------------------
1. **禁止 hard mask 乘法**。``image = image * wg_mask`` 会把 WG 预测的边界错误直接变成
   「病灶被删除」，这是不可恢复的失败：一个跨越腺体边界的病灶会在进入网络之前就消失。
   因此本模块只做 **bounding box + 物理 margin**，保留腺体周围的完整安全边界。
2. **margin 必须以物理距离 mm 定义**，不是「随便固定 voxel 数」：Dataset605/606 的
   ``3d_fullres`` spacing 是 ``[3.0, 0.5, 0.5] mm``，同一 mm 边界在 z 轴与面内对应不同的
   voxel 数；用 voxel 常数会让 z 方向的物理边界比面内小 6 倍。
3. **不重采样**。裁剪后通过 **padding**（不是 resize）满足网络输入尺寸要求，因此体素物理
   尺寸、spacing、origin 全部保持不变，预测可以**无损**恢复到原空间。重采样会改变
   lesion 的物理体积并破坏与评估协议的 mm³ 口径的一致性，本项目明确拒绝。
4. **空 WG 预测不回退为「删除病灶」**：回退到全视野并**显式记录**原因，绝不静默。
   这使 anatomy 先验错误成为真实部署链条的一部分，而不是被隐藏掉。

crop transform 的保存与恢复
---------------------------
:class:`ProstateROI` 完整记录 ``lower`` / ``upper`` / ``original_shape`` / ``spacing`` /
``margin_mm`` / 来源标记，并提供 :meth:`crop` 与 :meth:`restore`：

- :meth:`restore` 把裁剪空间的任何数组（概率图、掩膜、logits）放回原空间，ROI 外填 ``fill``；
- :meth:`to_json` 给出可审计的 provenance，写入 run 配置或评估报告。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


class ROIError(RuntimeError):
    """ROI 构造或几何恢复中的可预期失败（fail-closed）。"""


def _validate_probability(array: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(array, dtype=np.float64)
    if array.ndim != 3:
        raise ROIError(f"{name} 必须是 3D 数组（Z, Y, X），收到 shape={array.shape}")
    if array.size == 0:
        raise ROIError(f"{name} 为空数组")
    if not bool(np.isfinite(array).all()):
        raise ROIError(f"{name} 含非有限值")
    if bool((array < 0.0).any()) or bool((array > 1.0).any()):
        raise ROIError(f"{name} 超出 [0, 1]（要求 soft probability，不是 logits）")
    return array


def _validate_spacing(spacing_zyx: tuple[float, float, float]) -> tuple[float, float, float]:
    spacing = tuple(float(s) for s in spacing_zyx)
    if len(spacing) != 3:
        raise ROIError(f"spacing 必须是三个值（Z, Y, X 数组轴序），收到 {spacing_zyx!r}")
    if any(not math.isfinite(s) or s <= 0 for s in spacing):
        raise ROIError(f"spacing 必须是有限正数，收到 {spacing_zyx!r}")
    return spacing  # type: ignore[return-value]


def expand_box_by_physical_margin(
    lower: tuple[int, int, int],
    upper: tuple[int, int, int],
    spacing_zyx: tuple[float, float, float],
    margin_mm: float,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """按**物理距离**扩张半开区间 ``[lower, upper)``；每轴 voxel 数向上取整。

    向上取整保证实际物理边界**不小于**请求的 ``margin_mm``（不会因为取整而比承诺更紧）。
    """
    if not math.isfinite(margin_mm) or margin_mm < 0:
        raise ROIError(f"margin_mm 必须是非负有限数，收到 {margin_mm!r}")
    spacing = _validate_spacing(spacing_zyx)
    if upper <= lower:
        raise ROIError(f"非法区间：[{lower}, {upper})")
    expansion = tuple(int(math.ceil(margin_mm / s)) for s in spacing)
    return (
        tuple(lower[i] - expansion[i] for i in range(3)),  # type: ignore[return-value]
        tuple(upper[i] + expansion[i] for i in range(3)),  # type: ignore[return-value]
    )


@dataclass(frozen=True)
class ProstateROI:
    """一个前列腺 ROI 的完整 crop transform（可序列化、可恢复几何）。

    坐标一律是**数组轴序** ``(Z, Y, X)``，与 nnU-Net 预处理数组、评估器的 ``spacing_zyx``
    完全一致。``[lower, upper)`` 是半开区间。
    """

    lower: tuple[int, int, int]
    upper: tuple[int, int, int]
    original_shape: tuple[int, int, int]
    spacing_zyx: tuple[float, float, float]
    margin_mm: float
    wg_threshold: float
    prior_source: str
    fallback_reason: str | None = None

    # ---------------------------------------------------------------- 属性
    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(self.upper[i] - self.lower[i] for i in range(3))  # type: ignore[return-value]

    @property
    def is_full_fov(self) -> bool:
        """是否退化为全视野（空 WG 预测或 WG 覆盖整幅图像时）。"""
        return self.lower == (0, 0, 0) and self.upper == self.original_shape

    @property
    def margin_voxels(self) -> tuple[int, int, int]:
        """实际使用的每轴 margin（voxel 数，向上取整）。"""
        return tuple(  # type: ignore[return-value]
            int(math.ceil(self.margin_mm / s)) for s in self.spacing_zyx
        )

    # ---------------------------------------------------------------- 几何
    def crop(self, array: np.ndarray, *, fill: float = 0.0) -> np.ndarray:
        """把原空间数组裁到 ROI；形状不符时报错（不静默广播）。"""
        array = np.asarray(array)
        if tuple(array.shape) != self.original_shape:
            raise ROIError(
                f"crop 期望原空间形状 {self.original_shape}，收到 {tuple(array.shape)}；"
                "ROI 不做重采样"
            )
        slices = tuple(slice(self.lower[i], self.upper[i]) for i in range(3))
        return array[slices]

    def restore(
        self, cropped: np.ndarray, *, original_shape: tuple[int, int, int] | None = None,
        fill: float = 0.0,
    ) -> np.ndarray:
        """把 ROI 空间的数组放回原空间（ROI 外填 ``fill``）。

        ``cropped`` 必须与 :attr:`shape` 逐值一致；形状不符即报错，避免把错位的预测静默
        贴回原空间——那会污染评估结果。
        """
        cropped = np.asarray(cropped)
        if tuple(cropped.shape) != self.shape:
            raise ROIError(
                f"restore 期望 ROI 形状 {self.shape}，收到 {tuple(cropped.shape)}"
            )
        target = self.original_shape if original_shape is None else tuple(original_shape)
        if len(target) != 3:
            raise ROIError(f"original_shape 必须是 3 维，收到 {target}")
        if any(target[i] < self.upper[i] for i in range(3)):
            raise ROIError(
                f"目标形状 {target} 小于 ROI 上界 {self.upper}：无法放入，拒绝静默裁剪"
            )
        out = np.full(target, fill, dtype=cropped.dtype)
        out[tuple(slice(self.lower[i], self.upper[i]) for i in range(3))] = cropped
        return out

    def pad_plan(self, target_shape: tuple[int, int, int]) -> dict:
        """给出把 ROI 空间补齐到 ``target_shape`` 的 padding 方案（**不**重采样）。

        返回 ``{"pad_before": ..., "pad_after": ..., "overhang": ...}``：

        - ``pad_before`` / ``pad_after`` 是每轴前后各补多少 voxel；
        - ``overhang`` 是 ROI 在某轴**超过**目标尺寸的量（网络需要的是至少 patch size；
          超过时由 nnU-Net 的滑窗自行处理，本模块不擅自裁剪 ROI，以免切掉病灶）。

        与 ``resize`` 的关键区别：padding 不改变 spacing，因此物理体积、mm³ 口径与
        ``docs/Evaluation_Protocol.md`` 完全一致。
        """
        target = tuple(int(t) for t in target_shape)
        if len(target) != 3:
            raise ROIError(f"target_shape 必须是 3 维，收到 {target_shape}")
        pad_before = tuple(max(0, (target[i] - self.shape[i]) // 2) for i in range(3))
        pad_after = tuple(
            max(0, target[i] - self.shape[i] - pad_before[i]) for i in range(3)
        )
        overhang = tuple(max(0, self.shape[i] - target[i]) for i in range(3))
        return {
            "pad_before": pad_before,
            "pad_after": pad_after,
            "overhang": overhang,
            "resampling": "none; padding only, spacing preserved",
        }

    # ---------------------------------------------------------------- 审计
    def to_json(self) -> dict[str, Any]:
        """可审计的 ROI provenance（写入 run 配置 / 评估报告）。"""
        return {
            "lower_zyx": list(self.lower),
            "upper_zyx": list(self.upper),
            "roi_shape_zyx": list(self.shape),
            "original_shape_zyx": list(self.original_shape),
            "spacing_zyx_mm": list(self.spacing_zyx),
            "margin_mm": float(self.margin_mm),
            "margin_voxels": list(self.margin_voxels),
            "wg_threshold": float(self.wg_threshold),
            "prior_source": self.prior_source,
            "fallback_reason": self.fallback_reason,
            "is_full_fov": self.is_full_fov,
            "resampling": "none",
        }


def build_prostate_roi(
    wg_probability: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    *,
    margin_mm: float,
    wg_threshold: float,
    prior_source: str = "predicted_prior",
    empty_fallback: str = "full_fov",
) -> ProstateROI:
    """由**预测** WG 概率图构造物理 margin 的前列腺 ROI。

    流程（与 Research Plan §Anatomy-Guided ROI 一致）::

        predicted WG -> bounding box -> physical margin expansion -> prostate-centred ROI

    ``empty_fallback`` 决定 WG 阈值化后为空（或覆盖整幅图像）时的行为：

    - ``"full_fov"``（默认）：回退到全视野，并在 :attr:`ProstateROI.fallback_reason` 中**显式
      记录**原因。这是**不删除**任何可能病灶的安全行为；
    - ``"error"``：直接抛 :class:`ROIError`（严格模式；用于需要确认先验确实非空的分析）。

  两种模式都不接受「把病灶裁掉」这一结果：选择永远是「保留更多」。
    """
    wg = _validate_probability(wg_probability, "wg_probability")
    spacing = _validate_spacing(spacing_zyx)
    if not 0.0 <= wg_threshold <= 1.0:
        raise ROIError(f"wg_threshold 必须落在 [0, 1]，收到 {wg_threshold}")
    if empty_fallback not in ("full_fov", "error"):
        raise ROIError(f"empty_fallback 必须是 'full_fov' 或 'error'，收到 {empty_fallback!r}")

    original_shape = tuple(int(n) for n in wg.shape)  # type: ignore[assignment]
    mask = wg >= wg_threshold
    if not bool(mask.any()):
        if empty_fallback == "error":
            raise ROIError(
                "WG 预测在阈值化后为空（empty WG prediction）；strict 模式下拒绝构造 ROI"
            )
        return ProstateROI(
            lower=(0, 0, 0),
            upper=original_shape,
            original_shape=original_shape,
            spacing_zyx=spacing,
            margin_mm=float(margin_mm),
            wg_threshold=float(wg_threshold),
            prior_source=prior_source,
            fallback_reason="empty_wg_prediction",
        )

    occupied = np.nonzero(mask)
    lower = tuple(int(axis.min()) for axis in occupied)  # type: ignore[assignment]
    upper = tuple(int(axis.max()) + 1 for axis in occupied)  # type: ignore[assignment]
    lower, upper = expand_box_by_physical_margin(lower, upper, spacing, margin_mm)
    lower = tuple(max(0, value) for value in lower)  # type: ignore[assignment]
    upper = tuple(min(original_shape[i], upper[i]) for i in range(3))  # type: ignore[assignment]
    if any(upper[i] <= lower[i] for i in range(3)):  # pragma: no cover - 防退化
        raise ROIError(f"扩张后的 ROI 退化：{[lower, upper]}")
    if lower == (0, 0, 0) and upper == original_shape:
        reason = "wg_covers_full_fov"
    else:
        reason = None
    return ProstateROI(
        lower=lower,
        upper=upper,
        original_shape=original_shape,
        spacing_zyx=spacing,
        margin_mm=float(margin_mm),
        wg_threshold=float(wg_threshold),
        prior_source=prior_source,
        fallback_reason=reason,
    )


def roi_retains_reference_lesions(
    roi: ProstateROI, reference_lesion: np.ndarray
) -> dict:
    """ROI 的**安全边界**检查：reference lesion 是否被 ROI 完整保留。

    这不是模型指标计算，而是 ROI 设计的健全性检查：如果 ROI 把 GT 病灶裁掉，该病例的 Dice
    上限已经被结构性压低，**必须**在报告里说明，不得当作模型漏检。

    返回每个 lesion 连通域的保留情况（完整保留 / 部分保留 / 完全被裁）。
    """
    from scipy import ndimage

    ref = np.asarray(reference_lesion, dtype=bool)
    if tuple(ref.shape) != roi.original_shape:
        raise ROIError(
            f"reference 形状 {tuple(ref.shape)} 与 ROI 原空间 {roi.original_shape} 不一致"
        )
    cropped = roi.crop(ref)
    structure = ndimage.generate_binary_structure(3, 1)
    _orig_labels, n_components = ndimage.label(ref, structure=structure)
    retained_inside = int(cropped.sum())
    total = int(ref.sum())
    return {
        "reference_lesions": int(n_components),
        "reference_voxels": total,
        "retained_voxels": retained_inside,
        "retained_voxel_fraction": (retained_inside / total) if total else None,
        "fully_retained": total == 0 or retained_inside == total,
        "roi_json": roi.to_json(),
    }


__all__ = (
    "ProstateROI",
    "ROIError",
    "build_prostate_roi",
    "expand_box_by_physical_margin",
    "roi_retains_reference_lesions",
)
