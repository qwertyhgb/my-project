"""3D 滑窗推理（Gaussian 加权融合）。

research_plan §9.5 要求：overlap sliding-window + Gaussian 权重融合 + 必要且统一的镜像 TTA。
本模块独立于训练代码，可对任意（含 DS 的）模型做推理；**不读取真实病例**，输入为张量。

关键约定：
- patch size 来自 plan（调用方传入）；
- 默认 `step_fraction = 0.5`；
- 输入小于 patch 时先 padding，输出裁回原尺寸；
- Gaussian importance map 中心权重大于边缘，且处处 > 0；
- 所有体素的累计权重必须 > 0（显式断言，不留未覆盖体素）；
- 模型返回 deep-supervision 列表时只取最高分辨率输出；
- 在 **float32** 上累计 logits，融合后再 softmax（本函数只返回 logits，softmax 交给下游）；
- mirror TTA 默认关闭；开启时对 8 种镜像翻转的 logits 求平均（与 nnU-Net 的概率平均略有差异，
  正式评估前需与 N0 对齐确认，详见 docs/P2_M0_Implementation.md）。
"""
from __future__ import annotations

import itertools
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ..utils.progress import make_progress

PAD_VALUE = 0.0


def compute_gaussian_importance_map(
    patch_size: Sequence[int], *, sigma_scale: float = 1.0 / 8, dtype: type = np.float32
) -> np.ndarray:
    """Gaussian importance map（与 nnUNet 相同的构造：中心置 1 → 高斯滤波 → 归一到 max=1 → 保证 > 0）。"""
    patch_size = tuple(int(p) for p in patch_size)
    if len(patch_size) != 3 or any(p < 1 for p in patch_size):
        raise ValueError(f"patch_size 必须为长度 3 的正整数，收到 {patch_size}")
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError as exc:  # pragma: no cover - scipy 为项目依赖
        raise ImportError("需要 scipy 计算 Gaussian importance map") from exc

    tmp = np.zeros(patch_size, dtype=np.float64)
    tmp[tuple(p // 2 for p in patch_size)] = 1.0
    sigmas = [p * sigma_scale for p in patch_size]
    smoothed = gaussian_filter(tmp, sigmas, 0, mode="constant", cval=0)
    smoothed /= smoothed.max()
    smoothed[smoothed == 0] = smoothed[smoothed > 0].min()  # 不允许出现 0 权重
    return smoothed.astype(dtype)


def compute_sliding_window_steps(
    image_size: Sequence[int], patch_size: Sequence[int], step_fraction: float = 0.5
) -> list[list[int]]:
    """计算各维的起点列表（与 nnUNet 的 compute_steps_for_sliding_window 语义一致）。"""
    image_size = tuple(int(i) for i in image_size)
    patch_size = tuple(int(p) for p in patch_size)
    if any(i < p for i, p in zip(image_size, patch_size)):
        raise ValueError(f"image_size={image_size} 必须 >= patch_size={patch_size}（调用方需先 padding）")
    if not (0.0 < float(step_fraction) <= 1.0):
        raise ValueError(f"step_fraction 必须在 (0,1]，收到 {step_fraction}")

    target_step = [p * float(step_fraction) for p in patch_size]
    num_steps = [int(np.ceil((i - p) / t)) + 1 for i, p, t in zip(image_size, patch_size, target_step)]
    steps: list[list[int]] = []
    for d in range(len(patch_size)):
        max_step_value = image_size[d] - patch_size[d]
        if num_steps[d] > 1:
            actual_step = max_step_value / (num_steps[d] - 1)
            steps.append([int(np.round(actual_step * i)) for i in range(num_steps[d])])
        else:
            steps.append([0])
    return steps


def pad_image_to_patch(image: torch.Tensor, patch_size: Sequence[int]) -> tuple[torch.Tensor, tuple[int, int, int]]:
    """把 [B, C, D, H, W] padding 到各维 >= patch；返回 (padded, 原始空间尺寸)。"""
    spatial = tuple(int(v) for v in image.shape[-3:])
    pads = []
    for d in range(3):
        deficit = int(patch_size[d]) - spatial[d]
        pads.append((0, max(0, deficit)))
    if any(p[1] > 0 for p in pads):
        # F.pad 的顺序是最后一维在前
        pad_arg = []
        for d in reversed(range(3)):
            pad_arg.extend(list(pads[d]))
        image = F.pad(image, pad_arg, mode="constant", value=PAD_VALUE)
    return image, spatial


def _main_logits(output) -> torch.Tensor:
    if isinstance(output, (list, tuple)):
        if not output:
            raise ValueError("模型返回空输出列表")
        return output[0]
    return output


def _pad_pair_to_patch(
    image: torch.Tensor, prior: torch.Tensor | None, patch_size: Sequence[int]
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """用**完全相同**的 padding 把 image 与 prior 补齐到 >= patch（prior 缺失时为 None）。"""
    spatial = tuple(int(v) for v in image.shape[-3:])
    pads = [(0, max(0, int(patch_size[d]) - spatial[d])) for d in range(3)]
    if not any(p[1] > 0 for p in pads):
        return image, prior
    pad_arg: list[int] = []
    for d in reversed(range(3)):
        pad_arg.extend(list(pads[d]))
    image = F.pad(image, pad_arg, mode="constant", value=PAD_VALUE)
    if prior is not None:
        prior = F.pad(prior, pad_arg, mode="constant", value=PAD_VALUE)
    return image, prior


@torch.no_grad()
def sliding_window_predict(
    model: torch.nn.Module,
    image: torch.Tensor,
    *,
    zonal_prior: torch.Tensor | None = None,
    patch_size: Sequence[int],
    step_fraction: float = 0.5,
    gaussian: bool = True,
    mirror_tta: bool = False,
    device: str | torch.device = "cpu",
    batch_size: int = 1,
    progress: bool = True,
    desc: str = "sliding-window",
) -> torch.Tensor:
    """滑窗推理并返回融合后的 logits [B, C, D, H, W]（与输入同尺寸）。

    Args:
        model: 任意 nn.Module；内部使用 `model.eval()`，返回 DS 列表时只取最高分辨率。
        image: [B, C, D, H, W] 或 [C, D, H, W]。
        zonal_prior: 可选 [B, 2, D, H, W]（PZ/TZ）；与 image **使用相同 padding、窗口坐标、
            batch 组合与镜像翻转**。模型要求 prior 时必填（不得用全零兜底）；模型不读取 prior 时禁止提供。
        patch_size: 来自 plan 的 patch。
        step_fraction: 滑窗步长比例（默认 0.5）。
        gaussian: 使用 Gaussian importance map（False 时用均匀权重）。
        mirror_tta: 是否启用 8 向镜像 TTA（默认关闭）。
        batch_size: 每个 forward 的 patch 数（>1 时按 batch 前向）。
        progress: 是否显示 tqdm 进度。
    """
    if image.ndim == 4:
        image = image[None]
    if image.ndim != 5:
        raise ValueError(f"image 必须是 [B, C, D, H, W]，收到 {tuple(image.shape)}")
    if int(batch_size) < 1:
        raise ValueError(f"batch_size 必须 >= 1，收到 {batch_size}")
    patch_size = tuple(int(p) for p in patch_size)
    if len(patch_size) != 3 or any(p < 1 for p in patch_size):
        raise ValueError(f"patch_size 必须为长度 3 的正整数，收到 {patch_size}")

    requires_prior = "zonal_prior" in tuple(getattr(model, "REQUIRED_INPUT_KEYS", ("image",)))
    if requires_prior and zonal_prior is None:
        raise ValueError(
            "模型要求 zonal_prior，但滑窗推理未提供（禁止用全零 prior 兜底）"
        )
    if zonal_prior is not None and not requires_prior:
        raise ValueError("模型不读取 zonal_prior，但滑窗推理提供了 prior（禁止静默混用）")
    if zonal_prior is not None:
        if zonal_prior.ndim == 4:
            zonal_prior = zonal_prior[None]
        if zonal_prior.ndim != 5 or zonal_prior.shape[1] != 2:
            raise ValueError(f"zonal_prior 必须是 [B, 2, D, H, W]，收到 {tuple(zonal_prior.shape)}")
        if int(zonal_prior.shape[0]) != int(image.shape[0]) or tuple(zonal_prior.shape[-3:]) != tuple(
            image.shape[-3:]
        ):
            raise ValueError(
                f"zonal_prior 必须与 image 同 batch/空间尺寸：prior={tuple(zonal_prior.shape)} "
                f"image={tuple(image.shape)}（网格不同立即失败，不得在推理时临时重采样）"
            )

    if int(image.shape[0]) != 1:
        raise ValueError(
            f"sliding_window_predict 每次只处理单例（B=1），收到 B={int(image.shape[0])}；请对病例循环调用"
        )
    model.eval()
    original_shape = tuple(int(v) for v in image.shape[-3:])
    padded, padded_prior = _pad_pair_to_patch(image, zonal_prior, patch_size)
    padded_shape = tuple(int(v) for v in padded.shape[-3:])
    starts = compute_sliding_window_steps(padded_shape, patch_size, step_fraction)
    positions = list(itertools.product(*starts))

    def _single_pass(images: torch.Tensor, priors: torch.Tensor | None, desc: str = "") -> torch.Tensor:
        num_classes = None
        acc: torch.Tensor | None = None
        weight_map = _weight_map(patch_size, gaussian)
        offsets = range(0, len(positions), int(batch_size))
        bar = make_progress(offsets, total=len(offsets), desc=desc, disable=not progress, unit="patch")
        for start in bar:
            chunk = positions[start : start + int(batch_size)]
            single = images[0]  # [C, D, H, W]（B=1，已在上方校验）
            patches = torch.stack(
                [
                    single[..., s0 : s0 + patch_size[0], s1 : s1 + patch_size[1], s2 : s2 + patch_size[2]].contiguous()
                    for s0, s1, s2 in chunk
                ],
                dim=0,
            ).to(device)  # [P, C, *patch]
            if priors is None:
                logits = _main_logits(model(patches)).float().cpu()
            else:
                # 与 image patch **逐个坐标一致**的 prior patch（同一 chunk、同一切片）
                single_prior = priors[0]
                prior_patches = torch.stack(
                    [
                        single_prior[
                            ..., s0 : s0 + patch_size[0], s1 : s1 + patch_size[1], s2 : s2 + patch_size[2]
                        ].contiguous()
                        for s0, s1, s2 in chunk
                    ],
                    dim=0,
                ).to(device)
                logits = _main_logits(model(patches, zonal_prior=prior_patches)).float().cpu()
            if num_classes is None:
                num_classes = int(logits.shape[1])
                acc = torch.zeros((images.shape[0], num_classes, *padded_shape), dtype=torch.float32)
            assert acc is not None
            for k, (s0, s1, s2) in enumerate(chunk):
                acc[..., s0 : s0 + patch_size[0], s1 : s1 + patch_size[1], s2 : s2 + patch_size[2]] += (
                    logits[k] * weight_map
                )
        assert acc is not None, "滑窗位置为空"
        total_weight = _accumulated_weight(padded_shape, positions, patch_size, weight_map)
        if bool((total_weight <= 0).any()):
            raise RuntimeError("存在未被任何 patch 覆盖的体素（累计权重 <= 0），拒绝返回不完整结果")
        return acc / total_weight

    if mirror_tta:
        flips = list(itertools.product([False, True], repeat=3))
        total = None
        for flip_index, flip in enumerate(flips):
            augmented = padded
            augmented_prior = padded_prior
            for axis, do_flip in enumerate(flip):
                if do_flip:
                    augmented = torch.flip(augmented, dims=[axis + 2])
                    if augmented_prior is not None:
                        augmented_prior = torch.flip(augmented_prior, dims=[axis + 2])
            logits = _single_pass(
                augmented, augmented_prior, desc=f"{desc} tta {flip_index + 1}/{len(flips)}"
            )
            for axis, do_flip in enumerate(flip):
                if do_flip:
                    logits = torch.flip(logits, dims=[axis + 2])
            total = logits if total is None else total + logits
        assert total is not None
        merged = total / len(flips)
    else:
        merged = _single_pass(padded, padded_prior, desc=desc)

    return merged[..., : original_shape[0], : original_shape[1], : original_shape[2]]


def _weight_map(patch_size: Sequence[int], gaussian: bool) -> torch.Tensor:
    if gaussian:
        weight = torch.from_numpy(compute_gaussian_importance_map(patch_size)).float()
    else:
        weight = torch.ones(tuple(int(p) for p in patch_size), dtype=torch.float32)
    return weight[None, None]


def _accumulated_weight(
    padded_shape: Sequence[int], positions: Sequence[Sequence[int]], patch_size: Sequence[int], weight_map: torch.Tensor
) -> torch.Tensor:
    total = torch.zeros((1, 1, *padded_shape), dtype=torch.float32)
    for s0, s1, s2 in positions:
        total[..., s0 : s0 + patch_size[0], s1 : s1 + patch_size[1], s2 : s2 + patch_size[2]] += weight_map
    return total


def logits_to_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """softmax（推理后的最终一步；网络上/下游均可复用）。"""
    return torch.softmax(logits, dim=1)


__all__ = [
    "compute_gaussian_importance_map",
    "compute_sliding_window_steps",
    "logits_to_probabilities",
    "pad_image_to_patch",
    "sliding_window_predict",
]
