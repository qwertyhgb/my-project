"""Lesionness / 局部化辅助目标（新主线条件 C 的 supervision 侧）。

它把模型的学习目标显式拆成两件事
--------------------------------
- **Where is the lesion?**——coarse lesionness 头，追求高 sensitivity，尽量不出现
  ``completely missed lesions``；
- **What is the exact lesion boundary?**——最终的 segmentation 头，追求完整覆盖。

目标形式的选择（只选一种，不一次性实现多个版本）
------------------------------------------------
候选：Gaussian lesion center heatmap / dilated lesion mask / coarse lesion mask /
distance-transform target。本项目选择 **物理半径膨胀的 coarse lesion mask**：

1. **与现有数据标签形式一致**：Dataset605/606 的 label 是二值 lesion，膨胀后仍是二值掩膜，
   因此可以直接沿用 nnU-Net 原生的 Focal+CE / Dice+CE 与 ``DeepSupervisionWrapper``，
   不需要引入新的损失族、也不需要新的 target 编码；
2. **与现有 3D spacing 自洽**：膨胀半径以 **mm** 定义，再按每个尺度的真实 spacing 折算成
   voxel 数。当前 ``3d_fullres`` 是 ``[3.0, 0.5, 0.5] mm``，用 voxel 常数会让 z 方向的物理
   半径比面内大 6 倍；
3. **最容易解释**：目标值语义就是「该体素在某个真实病灶 r mm 邻域内」；
4. **避免 heatmap 的校准问题**：Gaussian center heatmap 的峰值出现在单个体素上，3D 稀疏病灶
   下极端类不平衡，且 sigma 需要额外调参——这会引入一个不可归因的超参数。

唯一的研究性超参数是 ``LESIONNESS_DILATION_RADIUS_MM``，它在**看结果之前**固定，必须写进
run 配置，不得在看到结果后调整。

监督来源的硬约束
----------------
- lesionness supervision **只来自 lesion GT**（经 :func:`lesionness_target` 变换）；
- validation / test 时**不使用**任何 GT 推导的目标；推理只看 coarse head 的预测；
- coarse 头与 final segmentation 头**完全联合训练**（同一个 optimizer、同一次 backward）。
"""

from __future__ import annotations

import math

import numpy as np

#: coarse lesionness 的膨胀半径（mm）。**预先冻结**：未看结果前固定，不得事后调参。
#:
#: 取值理由：Dataset605/606 的 ``3d_fullres`` 面内 spacing 是 0.5 mm，z 轴 3.0 mm。3 mm 的
#: 物理半径在面内约 6 voxel、z 轴约 1 voxel，对应「病灶及其紧邻腺体组织」这一个尺度量级；
#: 它比典型小病灶（< 500 mm³，约 10 mm 直径）的半径略小，因此不会把病灶两两连成一片。
LESIONNESS_DILATION_RADIUS_MM = 3.0


class LesionnessError(RuntimeError):
    """lesionness 目标生成中的可预期失败（fail-closed）。"""


def physical_radius_to_voxels(
    radius_mm: float, spacing_zyx: tuple[float, float, float]
) -> tuple[int, int, int]:
    """把物理半径（mm）折算为每轴 voxel 半径，向上取整（物理覆盖不小于承诺）。"""
    if not math.isfinite(radius_mm) or radius_mm < 0:
        raise LesionnessError(f"radius_mm 必须是非负有限数，收到 {radius_mm!r}")
    spacing = tuple(float(s) for s in spacing_zyx)
    if len(spacing) != 3 or any(not math.isfinite(s) or s <= 0 for s in spacing):
        raise LesionnessError(f"spacing 必须是三个有限正数（Z, Y, X），收到 {spacing_zyx!r}")
    return tuple(int(math.ceil(radius_mm / s)) for s in spacing)  # type: ignore[return-value]


def _ellipsoid_structure(radii: tuple[int, int, int]) -> np.ndarray:
    """按物理半径构造**椭球**结构元素。

    各向异性 spacing 下用立方体结构元素会把 z 方向的物理半径放大 6 倍；椭球保证「在物理空间
    近似球形」，这是把半径以 mm 定义的唯一自洽实现。
    """
    if any(r < 0 for r in radii):
        raise LesionnessError(f"voxel 半径不能为负，收到 {radii}")
    grids = np.ogrid[
        tuple(slice(-r, r + 1) for r in radii)  # type: ignore[arg-type]
    ]
    denominator = [(r * r) if r > 0 else None for r in radii]
    squared = np.zeros(
        tuple(2 * r + 1 for r in radii), dtype=np.float64
    )
    for axis, (grid, limit) in enumerate(zip(grids, denominator, strict=True)):
        # 某轴半径为 0 时该轴不参与距离（退化为一层）
        if limit is None:
            continue
        shape = [1, 1, 1]
        shape[axis] = 2 * radii[axis] + 1
        squared = squared + (np.asarray(grid).reshape(shape) ** 2) / limit
    return squared <= 1.0


def lesionness_target(
    lesion_mask: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    *,
    radius_mm: float = LESIONNESS_DILATION_RADIUS_MM,
) -> np.ndarray:
    """由 lesion GT 生成 coarse lesionness 目标（物理半径膨胀的二值掩膜）。

    - 输入必须是二值 lesion GT（0/1）；含其它值即报错，避免把 ignore label 当成背景；
    - 膨胀使用按 spacing 折算的**椭球**结构元素，因此物理半径在各向异性下仍然正确；
    - 半径 ``0`` 时返回与输入逐值相同的掩膜（便于消融）；
    - 空 GT（阴性病例）返回全零目标，这是合法输入，**不是**错误。
    """
    mask = np.asarray(lesion_mask)
    if mask.ndim != 3:
        raise LesionnessError(f"lesion_mask 必须是 3D，收到 shape={mask.shape}")
    if not np.isin(mask, [0, 1, 0.0, 1.0, True, False]).all():
        raise LesionnessError(
            "lesion_mask 必须是二值 0/1（Dataset605/606 无 ignore label）；"
            f"发现其它取值 {sorted(set(np.unique(mask).tolist()))[:5]}"
        )
    binary = mask.astype(bool)
    radii = physical_radius_to_voxels(radius_mm, spacing_zyx)
    if all(r == 0 for r in radii) or not bool(binary.any()):
        return binary.copy()
    from scipy import ndimage

    return ndimage.binary_dilation(binary, structure=_ellipsoid_structure(radii))


def downsample_binary_max(
    array: np.ndarray, target_shape: tuple[int, int, int]
) -> np.ndarray:
    """用 **max-pooling** 把二值目标降到目标形状（保持「附近有病灶」的语义）。

    只在整数倍降采样且能整除时工作；不整除即报错，避免静默引入坐标错位。这与 nnU-Net
    deep supervision 的降采样约定一致（``_get_deep_supervision_scales`` 的 2 倍序列）。
    """
    arr = np.asarray(array)
    target = tuple(int(t) for t in target_shape)
    if arr.ndim != 3 or len(target) != 3:
        raise LesionnessError(f"必须是 3D：array={arr.shape}, target={target_shape}")
    if tuple(arr.shape) == target:
        return arr.astype(bool).copy()
    if any(t <= 0 for t in target):
        raise LesionnessError(f"非法目标形状 {target}")
    if any(arr.shape[i] % target[i] != 0 for i in range(3)):
        raise LesionnessError(
            f"降采样必须整除：{tuple(arr.shape)} -> {target}；不整除会引入坐标错位"
        )
    factors = tuple(arr.shape[i] // target[i] for i in range(3))
    reduced = arr.astype(bool)
    for axis, factor in enumerate(factors):
        if factor == 1:
            continue
        # 逐轴 max-pool：先 reshape 再 max，保证与 nnU-Net 的 DS 下采样一致
        new_shape = (
            reduced.shape[:axis] + (reduced.shape[axis] // factor, factor)
            + reduced.shape[axis + 1 :]
        )
        reduced = reduced.reshape(new_shape).max(axis=axis + 1)
    return reduced


__all__ = (
    "LESIONNESS_DILATION_RADIUS_MM",
    "LesionnessError",
    "downsample_binary_max",
    "lesionness_target",
    "physical_radius_to_voxels",
)
