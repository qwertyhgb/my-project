"""N0「Pseudo dice 长期为 0」排查专用的 nnU-Net 只读适配层。

边界与约定（与 `docs/project_structure.md` 的边界一致）
--------------------------------------------------------
- `models/`、`training/`、`inference/` 不得 import `nnunetv2`；本模块位于 `integrations/`，
  是**专门的第三方适配层**，允许在函数内部**惰性** import `nnunetv2` / `batchgenerators` / `blosc2`。
  模块导入本身不触发任何第三方依赖，因此可以在没有 nnU-Net 环境时被测试与 `--help` 使用。
- 全部函数只读：不写回 `.b2nd`/`.pkl`，不创建或修改 nnU-Net 的 raw/preprocessed/results 目录，
  也不会以任何方式干扰正在运行的训练进程。
- 纯函数（patch 统计、class_locations 校验、聚合与判定）不依赖第三方包，可直接被合成单元测试覆盖。

为什么需要这个模块
------------------
v2.6.2 的训练 loop 与「Pseudo dice」指标存在若干容易误解的细节（FG 强制采样、验证集也做 FG 过采样、
指标只在 patch 上累计），这些细节决定了「pseudo dice 恒为 0」到底是指标问题还是模型问题。
本模块把这些行为**以官方实现为准**暴露出来，供 `scripts/train/diagnose_n0_zero_dice.py` 直接测量。
"""
from __future__ import annotations

import math
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

FOREGROUND_LABEL = 1


# --------------------------------------------------------------------------- 纯函数


def as_single_patch(seg: np.ndarray, case_id: str = "") -> np.ndarray:
    """把 seg 规范为单张 3D patch ``[D, H, W]``；多于一张 patch 时报错。"""
    arr = np.asarray(seg)
    while arr.ndim > 3:
        if arr.shape[0] != 1:
            raise ValueError(f"{case_id}: 期望单张 patch，收到 shape={arr.shape}")
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"{case_id}: patch 必须是 3D，收到 shape={arr.shape}")
    return arr


def patch_foreground_stats(seg: np.ndarray, label: int = FOREGROUND_LABEL, case_id: str = "") -> dict:
    """单张 patch 的前景统计（纯函数）。

    返回 ``fg_voxels / total_voxels / fg_ratio / labels / binary_ok / bbox_z|y|x``。
    """
    arr = as_single_patch(seg, case_id)
    labels = np.unique(arr)
    fg = arr == label
    n_fg = int(fg.sum())
    bbox = None
    if n_fg > 0:
        idx = np.argwhere(fg)
        lo, hi = idx.min(0), idx.max(0)
        bbox = {
            "z": [int(lo[0]), int(hi[0])],
            "y": [int(lo[1]), int(hi[1])],
            "x": [int(lo[2]), int(hi[2])],
        }
    return {
        "fg_voxels": n_fg,
        "total_voxels": int(arr.size),
        "fg_ratio": float(n_fg / arr.size) if arr.size else 0.0,
        "labels": [int(v) for v in labels[:20]],
        "binary_ok": bool(np.all(np.isin(labels, [0, label]))),
        "bbox": bbox,
    }


def foreground_oversample_flags(batch_size: int, oversample_foreground_percent: float) -> list[bool]:
    """复刻官方 ``nnUNetDataLoader._oversample_last_XX_percent``（v2.6.2）。

    官方实现：``not sample_idx < round(batch_size * (1 - oversample_foreground_percent))``。
    即「一个 batch 内固定后 N 个样本被强制取在前景体素上」，不是逐样本概率。
    """
    if batch_size <= 0:
        raise ValueError("batch_size 必须为正")
    n_random = round(batch_size * (1.0 - float(oversample_foreground_percent)))
    return [not (i < n_random) for i in range(int(batch_size))]


def class_locations_report(
    class_locations: Mapping | None,
    spatial_shape: Sequence[int],
    label: int = FOREGROUND_LABEL,
) -> dict:
    """校验官方 ``properties['class_locations']``（纯函数）。

    官方 preprocessor 存的是 ``{label: ndarray (n, 4)}``，首列为标签索引、后三列为**空间坐标**
    （坐标已相对裁剪后的数组；官方 dataloader 用 ``selected_voxel[i + 1]`` 取值，故必须带首列）。
    """
    report = {
        "present": class_locations is not None,
        "keys": [],
        "shape": None,
        "n_samples_label": 0,
        "label_column_unique": [],
        "coords_min": None,
        "coords_max": None,
        "in_bounds": False,
        "issues": [],
    }
    if class_locations is None:
        report["issues"].append("class_locations 缺失（.pkl 无该字段）")
        return report

    try:
        keys = list(class_locations.keys())
    except AttributeError:
        report["issues"].append("class_locations 不是 mapping")
        return report
    report["keys"] = [str(k) for k in keys]

    arr = None
    for k in keys:
        if k == label or str(k) == str(label):
            arr = class_locations[k]
            break
    if arr is None:
        report["issues"].append(f"缺少 label={label} 的 class_locations（现有 keys={report['keys']}）")
        return report

    arr = np.asarray(arr)
    report["shape"] = list(arr.shape)
    if arr.size == 0:
        report["issues"].append(f"label={label} 的 class_locations 为空（该病例无前景体素）")
        return report
    if arr.ndim != 2 or arr.shape[1] < 3:
        report["issues"].append(f"class_locations 形状异常：{arr.shape}（期望 (n,3) 或 (n,4)）")
        return report

    coords = arr[:, -3:]
    report["n_samples_label"] = int(arr.shape[0])
    report["coords_min"] = [int(v) for v in coords.min(0)]
    report["coords_max"] = [int(v) for v in coords.max(0)]
    if arr.shape[1] == 4:
        report["label_column_unique"] = [int(v) for v in np.unique(arr[:, 0])[:10]]

    ok = True
    for d in range(3):
        if coords[:, d].min() < 0 or coords[:, d].max() >= int(spatial_shape[d]):
            ok = False
    report["in_bounds"] = ok
    if not ok:
        report["issues"].append(
            f"class_locations 坐标越界：min={report['coords_min']} max={report['coords_max']} "
            f"shape={list(spatial_shape)}"
        )
    if arr.shape[1] == 4 and not bool(np.all(arr[:, 0] == label)):
        report["issues"].append("class_locations 首列标签索引与期望 label 不一致")
    return report


def evaluate_case_row(row: Mapping) -> list[str]:
    """对单例审计行做异常判定（纯函数），返回问题列表。"""
    issues: list[str] = []
    if not row.get("seg_exists", True):
        issues.append("预处理 seg 缺失")
    if not row.get("data_exists", True):
        issues.append("预处理 data 缺失")
    if not row.get("labels_binary", True):
        issues.append(f"seg 取值非 {{0,1}}：{row.get('seg_labels')}")
    if row.get("data_seg_shape_mismatch"):
        issues.append("data 与 seg 空间尺寸不一致")
    if row.get("data_not_finite"):
        issues.append("data 含 NaN/Inf")
    expected = row.get("expected_positive")
    n_fg = int(row.get("n_fg_voxels") or 0)
    if expected is True and n_fg == 0:
        issues.append("阳性病例（case_csPCa=YES）预处理 seg 全空 → 标签链路可能丢失前景")
    if expected is False and n_fg > 0:
        issues.append("阴性病例（case_csPCa=NO）预处理 seg 出现前景体素")
    for msg in row.get("class_locations_issues") or []:
        issues.append(f"class_locations: {msg}")
    return issues


def _percentiles(values: Sequence[float], ps: Sequence[float] = (0, 25, 50, 75, 100)) -> dict | None:
    vals = [float(v) for v in values]
    if not vals:
        return None
    vals.sort()
    out = {}
    n = len(vals)
    for p in ps:
        idx = min(n - 1, max(0, int(round(p / 100.0 * (n - 1)))))
        out[f"p{int(p)}"] = vals[idx]
    out["n"] = n
    return out


def aggregate_case_rows(rows: Sequence[Mapping]) -> dict:
    """按 split 聚合逐例审计结果（纯函数）。"""
    out: dict = {}
    for split in sorted({str(r.get("split")) for r in rows}):
        sub = [r for r in rows if str(r.get("split")) == split]
        pos = [r for r in sub if r.get("expected_positive") is True]
        neg = [r for r in sub if r.get("expected_positive") is False]
        fg_vals = [int(r.get("n_fg_voxels") or 0) for r in pos]
        anomalies = [
            {"case_id": r.get("case_id"), "issues": list(r.get("issues") or [])}
            for r in sub
            if r.get("issues")
        ]
        out[split] = {
            "n_cases": len(sub),
            "n_expected_positive": len(pos),
            "n_expected_negative": len(neg),
            "n_missing_files": sum(1 for r in sub if not r.get("seg_exists", True) or not r.get("data_exists", True)),
            "n_non_binary_labels": sum(1 for r in sub if not r.get("labels_binary", True)),
            "n_positive_with_empty_seg": sum(1 for r in pos if int(r.get("n_fg_voxels") or 0) == 0),
            "n_negative_with_foreground": sum(1 for r in neg if int(r.get("n_fg_voxels") or 0) > 0),
            "n_class_locations_problems": sum(
                1 for r in sub if (r.get("class_locations_issues") or [])
            ),
            "n_anomalies": len(anomalies),
            "anomalies": anomalies[:200],
            "positive_fg_voxels_percentiles": _percentiles(fg_vals),
            "labels_observed": sorted({lab for r in sub for lab in (r.get("seg_labels") or [])}),
        }
    return out


def aggregate_batch_rows(rows: Sequence[Mapping]) -> dict:
    """按 split 聚合 patch 采样探针结果（纯函数）。

    行分为两个 phase：

    - ``pre_aug``：官方 dataloader（``transforms=None``）采样出的原始 patch，
      用于判断「FG 强制采样是否真的命中前景」；
    - ``post_aug``：接**官方训练增强**后的 patch，用于判断增强是否把前景抹掉。

    ``lost_foreground_after_aug_rate`` 是**聚合估计** ``1 - rate_post / rate_pre``，不是逐 patch 配对比较：
    增强内部会消耗 RNG，两次采样的 case/bbox 不保证一一对应。该估计为保守指标（只反映整体比例变化）。
    """
    out: dict = {}
    for split in sorted({str(r.get("split")) for r in rows}):
        sub = [r for r in rows if str(r.get("split")) == split]
        pre = [r for r in sub if str(r.get("phase", "pre_aug")) == "pre_aug"]
        post = [r for r in sub if str(r.get("phase")) == "post_aug"]
        base = pre if pre else sub
        n_pre, n_pre_fg = len(pre), sum(1 for r in pre if r.get("has_foreground"))
        n_post, n_post_fg = len(post), sum(1 for r in post if r.get("has_foreground"))
        rate_pre = (n_pre_fg / n_pre) if n_pre else 0.0
        rate_post = (n_post_fg / n_post) if n_post else 0.0
        lost = None
        if n_pre and n_post and rate_pre > 0:
            lost = max(0.0, 1.0 - rate_post / rate_pre)

        forced = [r for r in base if r.get("forced_foreground")]
        n_fg = sum(1 for r in base if r.get("has_foreground"))
        out[split] = {
            "n_patches": len(base),
            "n_patches_with_foreground": n_fg,
            "fg_patch_rate": (n_fg / len(base)) if base else 0.0,
            "n_forced_foreground_patches": len(forced),
            "n_forced_foreground_patches_with_fg": sum(1 for r in forced if r.get("has_foreground")),
            "forced_foreground_hit_rate": (
                sum(1 for r in forced if r.get("has_foreground")) / len(forced) if forced else 0.0
            ),
            "n_random_patches": len(base) - len(forced),
            "n_random_patches_with_foreground": sum(
                1 for r in base if not r.get("forced_foreground") and r.get("has_foreground")
            ),
            "foreground_voxels_in_fg_patches": _percentiles(
                [int(r.get("fg_voxels") or 0) for r in base if r.get("has_foreground")]
            ),
            "n_patches_checked_after_aug": n_post,
            "n_patches_with_foreground_after_aug": n_post_fg,
            "fg_patch_rate_after_aug": rate_post,
            "lost_foreground_after_aug_rate": lost,
            "augmentation_config_note": "聚合估计（1 - rate_post/rate_pre）；非逐 patch 配对",
        }
    return out


# 判定码 → 决策树分支（对应排查方案 D 的 7 个分支）
VERDICT_MAPPING_OR_CONFIG_BROKEN = "MAPPING_OR_CONFIG_BROKEN"          # 分支 2
VERDICT_DATA_OR_LABEL_BROKEN = "DATA_OR_LABEL_BROKEN"                  # 分支 1
VERDICT_FG_SAMPLING_AT_RISK = "FG_SAMPLING_AT_RISK"                    # 分支 3（先兆）
VERDICT_FG_SAMPLING_INEFFECTIVE = "FG_SAMPLING_INEFFECTIVE"            # 分支 3
VERDICT_AUGMENTATION_LOSES_FOREGROUND = "AUGMENTATION_LOSES_FOREGROUND"  # 分支 4
VERDICT_PIPELINE_OK_METRIC_INFORMATIVE = "PIPELINE_OK_METRIC_INFORMATIVE"  # 分支 6/7
VERDICT_FG_PATCH_RATE_LOW = "FG_PATCH_RATE_LOW"                        # 分支 3/5 之间
VERDICT_NEEDS_BATCH_PROBE = "NEEDS_BATCH_PROBE"                        # 需继续采样探针

_NEXT_STEPS = {
    VERDICT_MAPPING_OR_CONFIG_BROKEN:
        "先修 metadata 链路（dataset.json / plans / splits / case 映射），不要改模型或训练超参。",
    VERDICT_DATA_OR_LABEL_BROKEN:
        "定位到具体 case_id 后核对该例原始标签与重采样决定；必要时用新的派生目录重做该例子集，禁止覆盖既有 processed。",
    VERDICT_FG_SAMPLING_AT_RISK:
        "class_locations 不完整会使 FG 强制采样静默回退为随机 crop；先用 --split val --max-batches 20 确认实际命中率，再决定是否重建该子集的 class_locations（新目录、新 dataset id）。",
    VERDICT_FG_SAMPLING_INEFFECTIVE:
        "FG 强制 patch 完全没命中前景 → 采样/属性链路异常；跑 --with-augmentation 与 train split 对照，并检查负例占比与 class_locations 完整性。",
    VERDICT_AUGMENTATION_LOSES_FOREGROUND:
        "增强后前景消失比例过高 → 用官方 transforms 做单因素对照（先关 spatial 增强），保持 N0 原样、另开 N0-debug run。",
    VERDICT_PIPELINE_OK_METRIC_INFORMATIVE:
        "数据/采样/标签闭环正常且 pseudo dice 有统计信号 → 下一步做(1)冻结 checkpoint 的全体积验证 与(2)2–4 阳性例受控 overfit，判断是“慢”还是“坏”。",
    VERDICT_FG_PATCH_RATE_LOW:
        "含前景 patch 比例过低 → 先跑受控 overfit 验证梯度闭环，再考虑（新 run、单因素）提高诊断采样覆盖。",
    VERDICT_NEEDS_BATCH_PROBE:
        "逐例审计已通过；请加 --max-batches（建议 20）运行采样探针以确认 FG 强制采样是否真正命中前景。",
}


def build_verdict(
    static_ok: bool,
    case_summary: Mapping | None = None,
    batch_summary: Mapping | None = None,
    *,
    static_issues: Sequence[str] = (),
    min_fg_patch_rate: float = 0.05,
    max_lost_after_aug: float = 0.2,
) -> dict:
    """把审计/探针结果映射到决策树分支（纯函数，确定性）。"""
    reasons: list[str] = []

    if not static_ok:
        reasons.append("静态审计未通过：" + "; ".join(static_issues or ["见 static_audit.json"]))
        code = VERDICT_MAPPING_OR_CONFIG_BROKEN
        return {"verdict": code, "reasons": reasons, "next_step": _NEXT_STEPS[code]}

    if case_summary is not None:
        for split, s in case_summary.items():
            if s.get("n_missing_files"):
                reasons.append(f"{split}: {s['n_missing_files']} 例预处理文件缺失")
            if s.get("n_non_binary_labels"):
                reasons.append(f"{split}: {s['n_non_binary_labels']} 例 seg 取值非 {{0,1}}")
            if s.get("n_positive_with_empty_seg"):
                reasons.append(
                    f"{split}: {s['n_positive_with_empty_seg']} 例阳性病例预处理 seg 全空"
                )
        if reasons:
            code = VERDICT_DATA_OR_LABEL_BROKEN
            return {"verdict": code, "reasons": reasons, "next_step": _NEXT_STEPS[code]}
        for split, s in case_summary.items():
            if s.get("n_class_locations_problems"):
                reasons.append(f"{split}: {s['n_class_locations_problems']} 例 class_locations 异常/缺失")
        if reasons:
            code = VERDICT_FG_SAMPLING_AT_RISK
            return {"verdict": code, "reasons": reasons, "next_step": _NEXT_STEPS[code]}

    if batch_summary:
        for split, s in batch_summary.items():
            if s.get("n_forced_foreground_patches", 0) > 0 and s.get(
                "n_forced_foreground_patches_with_fg", 0
            ) == 0:
                reasons.append(
                    f"{split}: {s['n_forced_foreground_patches']} 个 FG 强制 patch 全部不含前景"
                )
                code = VERDICT_FG_SAMPLING_INEFFECTIVE
                return {"verdict": code, "reasons": reasons, "next_step": _NEXT_STEPS[code]}
        for split, s in batch_summary.items():
            lost = s.get("lost_foreground_after_aug_rate")
            if lost is not None and lost > max_lost_after_aug:
                reasons.append(f"{split}: 增强后前景消失比例 {lost:.2%} > {max_lost_after_aug:.0%}")
                code = VERDICT_AUGMENTATION_LOSES_FOREGROUND
                return {"verdict": code, "reasons": reasons, "next_step": _NEXT_STEPS[code]}
        for split, s in batch_summary.items():
            rate = float(s.get("fg_patch_rate") or 0.0)
            if rate >= min_fg_patch_rate:
                reasons.append(f"{split}: 含前景 patch 比例 {rate:.2%} ≥ {min_fg_patch_rate:.0%}")
                code = VERDICT_PIPELINE_OK_METRIC_INFORMATIVE
                return {"verdict": code, "reasons": reasons, "next_step": _NEXT_STEPS[code]}
        for split, s in batch_summary.items():
            reasons.append(f"{split}: 含前景 patch 比例 {float(s.get('fg_patch_rate') or 0.0):.2%} 偏低")
        code = VERDICT_FG_PATCH_RATE_LOW
        return {"verdict": code, "reasons": reasons, "next_step": _NEXT_STEPS[code]}

    code = VERDICT_NEEDS_BATCH_PROBE
    return {"verdict": code, "reasons": reasons, "next_step": _NEXT_STEPS[code]}


# --------------------------------------------------------------------------- 官方适配（惰性 import）


def open_official_dataset(
    preprocessed_folder: str,
    case_ids: Sequence[str],
    *,
    data_folder_name: str = "nnUNetPlans_3d_fullres",
):
    """打开官方 ``nnUNetDatasetBlosc2``（只读、惰性）。

    返回对象支持 ``load_case(case_id) -> (data, seg, seg_prev, properties)``；其中 ``data``/``seg``
    是 blosc2 惰性数组，只有显式 ``np.asarray(...)`` 才会真正读取体素。
    """
    from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2

    folder = f"{str(preprocessed_folder).rstrip('/')}/{data_folder_name}"
    return nnUNetDatasetBlosc2(folder=folder, identifiers=[str(c) for c in case_ids])


def load_case_light(dataset, case_id: str, *, read_data: bool = False, read_seg: bool = True):
    """只读物化：properties 必读；data/seg 按需物化，默认只读 seg（省 I/O 与内存）。"""
    data, seg, _seg_prev, properties = dataset.load_case(str(case_id))
    seg_np = np.asarray(seg) if read_seg else None
    data_np = np.asarray(data) if read_data else None
    return data_np, seg_np, dict(properties)


def derive_official_aug_config(patch_size: Sequence[int], *, aniso_threshold: float | None = None) -> dict:
    """复刻 ``nnUNetTrainer.configure_rotation_dummyDA_mirroring_and_inital_patch_size``（3D 分支）。

    v2.6.2 中该函数是实例方法，只能通过 trainer 实例调用；为**不实例化 trainer**（避免写入 results 目录、
    不影响正在运行的训练），此处按源码逐行复刻。返回值会写入诊断报告，便于与训练日志中打印的
    ``do_dummy_2d_data_aug: ...`` 交叉核对。
    """
    ps = tuple(int(v) for v in patch_size)
    if len(ps) != 3:
        raise ValueError(f"仅支持 3D patch，收到 {ps}")
    if aniso_threshold is None:
        try:  # 默认与官方一致；取不到时回退官方常量 3.0
            from nnunetv2.configuration import ANISO_THRESHOLD as _thr

            aniso_threshold = float(_thr)
        except Exception:
            aniso_threshold = 3.0
    do_dummy_2d = (max(ps) / ps[0]) > float(aniso_threshold)
    if do_dummy_2d:
        rotation_for_da = (-180.0 / 360 * 2.0 * math.pi, 180.0 / 360 * 2.0 * math.pi)
    else:
        rotation_for_da = (-30.0 / 360 * 2.0 * math.pi, 30.0 / 360 * 2.0 * math.pi)
    return {
        "patch_size": list(ps),
        "aniso_threshold": float(aniso_threshold),
        "do_dummy_2d_data_aug": bool(do_dummy_2d),
        "rotation_for_DA": list(rotation_for_da),
        "mirror_axes": [0, 1, 2],
        "replicated_from": "nnUNetTrainer.configure_rotation_dummyDA_mirroring_and_inital_patch_size (v2.6.2)",
    }


def build_official_training_transforms(
    patch_size: Sequence[int],
    *,
    use_mask_for_norm: Sequence[bool] | None = None,
    foreground_labels: Sequence[int] = (FOREGROUND_LABEL,),
):
    """用**官方** ``nnUNetTrainer.get_training_transforms`` 构造训练增强管道（v2.6.2 中为 staticmethod）。

    ``deep_supervision_scales=None``：不追加 DS 下采样，便于直接检查「增强后标签是否仍含前景」。
    """
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

    cfg = derive_official_aug_config(patch_size)
    transforms = nnUNetTrainer.get_training_transforms(
        np.array(tuple(int(v) for v in patch_size)),
        tuple(cfg["rotation_for_DA"]),
        None,  # deep_supervision_scales=None → 不做 DS 下采样
        tuple(cfg["mirror_axes"]),
        cfg["do_dummy_2d_data_aug"],
        use_mask_for_norm=list(use_mask_for_norm) if use_mask_for_norm is not None else [False] * 3,
        is_cascaded=False,
        foreground_labels=tuple(int(v) for v in foreground_labels),
        regions=None,
        ignore_label=None,
    )
    return transforms, cfg


def apply_official_transform(transform, image: np.ndarray, seg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """按官方 dataloader 的调用约定应用增强（``tr(**{'image':..., 'segmentation':...})``）。

    官方 dataloader 逐样本调用（见 ``nnUNetDataLoader.generate_train_batch``），此处保持完全一致。
    """
    import torch

    with torch.no_grad():
        out = transform(
            **{
                "image": torch.from_numpy(np.asarray(image)).float(),
                "segmentation": torch.from_numpy(np.asarray(seg)).to(torch.int16),
            }
        )
    img = out["image"].detach().cpu().numpy()
    seg_out = out["segmentation"]
    if isinstance(seg_out, (list, tuple)):
        raise ValueError("transform 返回了 DS 多尺度标签；请在构造时传入 deep_supervision_scales=None")
    return img, seg_out.detach().cpu().numpy()


def build_official_dataloader(
    preprocessed_folder: str,
    case_ids: Sequence[str],
    *,
    patch_size: Sequence[int],
    batch_size: int,
    dataset_json: Mapping,
    oversample_foreground_percent: float = 0.33,
    transforms=None,
    data_folder_name: str = "nnUNetPlans_3d_fullres",
):
    """构造与官方训练**同参数**的 ``nnUNetDataLoader``（transforms=None 时为纯采样，不做增强）。"""
    from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
    from nnunetv2.utilities.label_handling.label_handling import LabelManager

    dataset = open_official_dataset(
        preprocessed_folder, case_ids, data_folder_name=data_folder_name
    )
    label_manager = LabelManager(dataset_json["labels"], regions_class_order=None)
    patch = tuple(int(v) for v in patch_size)
    loader = nnUNetDataLoader(
        dataset,
        int(batch_size),
        patch,  # patch_size
        patch,  # final_patch_size（与官方训练一致：验证/训练都不做低分辨率模拟的尺寸变换）
        label_manager,
        oversample_foreground_percent=float(oversample_foreground_percent),
        sampling_probabilities=None,
        pad_sides=None,
        transforms=transforms,
        probabilistic_oversampling=False,
    )
    return loader


__all__ = [
    "FOREGROUND_LABEL",
    "as_single_patch",
    "patch_foreground_stats",
    "foreground_oversample_flags",
    "class_locations_report",
    "evaluate_case_row",
    "aggregate_case_rows",
    "aggregate_batch_rows",
    "build_verdict",
    "open_official_dataset",
    "load_case_light",
    "derive_official_aug_config",
    "build_official_training_transforms",
    "apply_official_transform",
    "build_official_dataloader",
    "VERDICT_MAPPING_OR_CONFIG_BROKEN",
    "VERDICT_DATA_OR_LABEL_BROKEN",
    "VERDICT_FG_SAMPLING_AT_RISK",
    "VERDICT_FG_SAMPLING_INEFFECTIVE",
    "VERDICT_AUGMENTATION_LOSES_FOREGROUND",
    "VERDICT_PIPELINE_OK_METRIC_INFORMATIVE",
    "VERDICT_FG_PATCH_RATE_LOW",
    "VERDICT_NEEDS_BATCH_PROBE",
]
