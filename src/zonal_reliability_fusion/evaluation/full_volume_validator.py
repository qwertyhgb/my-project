"""低频全体积验证（P2-A3；v2.2 迁移为每 N epoch 一次）。

正式训练的验证协议（research_plan §9.4 / §11）：
- 每 `every_n_epochs`（正式 M0–M4 = 50，2026-09-20 起）个完整 epoch 结束后，对 fold 0 的**全部** validation
  study（冻结划分：223 例）逐个做**完整 3D 体积** sliding-window 推理，不使用随机 validation patch 代替；
  无论是否恰逢 N 的倍数，正常训练的**最后一个 epoch** 必须执行一次全体积验证（调度在 trainer 侧）；
- 验证不做任何随机数据增强；
- 推理期间关闭 deep supervision（只使用最高分辨率输出），结束后恢复模型的 DS 状态与 train/eval 状态；
- 只在 nnU-Net 预处理空间计算 Dice（恢复到原始物理空间 + 正式 picai_eval 放在最终评估阶段）；
- 不保存每个 epoch 的完整体积预测（避免磁盘爆炸），只保存病例级指标 CSV 与 epoch 汇总。

checkpoint 指标：`val_positive_casewise_dice_mean`
- 只对**阳性病例**（GT 含前景）逐例计算完整体积 Dice，再取算术平均；
- 阳性病例预测为空 → Dice = 0；
- **阴性病例不加入主要 Dice 均值，且"GT 与预测均为空"不得记为 Dice = 1**；
- 阴性表现通过 `val_negative_fp_case_rate` 与 `val_fp_voxels_per_negative_exam` 记录。

本模块不读取文件、不初始化 CUDA；数据经只读的 `PreprocessedStore`，推理经 `sliding_window_predict`。
"""
from __future__ import annotations

import csv
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from ..data.preprocessed_store import PreprocessedStore
from ..inference.sliding_window import sliding_window_predict
from ..utils.progress import make_progress

#: 允许的验证模式
VALIDATION_MODES: tuple[str, str] = ("full_volume", "diagnostic_patch")

#: 正式 checkpoint 指标（唯一允许用于选模的字段；解析出的其余字段只作记录）
CHECKPOINT_METRIC_FULL_VOLUME = "val_positive_casewise_dice_mean"

#: validator 产出的全部字段（其中只有上面那一个参与选模）
FULL_VOLUME_METRIC_FIELDS: tuple[str, ...] = (
    "val_positive_casewise_dice_mean",
    "val_positive_casewise_dice_median",
    "val_micro_dice",
    "val_positive_cases",
    "val_negative_cases",
    "val_cases_evaluated",
    "val_negative_fp_case_rate",
    "val_fp_voxels_per_negative_exam",
    "validation_elapsed_sec",
)


@dataclass(frozen=True)
class ValidationProtocol:
    """验证协议（必须与冻结划分一致；M0–M4 共用同一协议）。"""

    mode: str = "full_volume"
    every_n_epochs: int = 1
    checkpoint_metric: str = CHECKPOINT_METRIC_FULL_VOLUME
    maximize: bool = True
    expected_cases: int = 223
    expected_positive_cases: int = 63
    expected_negative_cases: int = 160
    step_fraction: float = 0.5
    gaussian: bool = True
    mirror_tta: bool = False
    sliding_window_batch_size: int = 1
    inner_patch_progress: bool = False
    save_case_metrics: bool = True

    def validate(self) -> None:
        """合法性检查（配置层与 validator 共用同一套规则）。"""
        if self.mode not in VALIDATION_MODES:
            raise ValueError(f"validation.mode 必须是 {VALIDATION_MODES} 之一，收到 {self.mode!r}")
        if int(self.every_n_epochs) < 1:
            raise ValueError(
                f"validation.every_n_epochs 必须 >= 1（0 或负数非法；正式 M0–M4 使用 50），"
                f"收到 {self.every_n_epochs}"
            )
        if any(
            int(v) < 1
            for v in (self.expected_cases, self.expected_positive_cases, self.expected_negative_cases)
        ):
            raise ValueError(
                "expected_cases / expected_positive_cases / expected_negative_cases 必须为正，"
                f"收到 {self.expected_cases}/{self.expected_positive_cases}/{self.expected_negative_cases}"
            )
        if int(self.expected_positive_cases) + int(self.expected_negative_cases) != int(self.expected_cases):
            raise ValueError(
                "expected_positive_cases + expected_negative_cases 必须等于 expected_cases，收到 "
                f"{self.expected_positive_cases}+{self.expected_negative_cases} != {self.expected_cases}"
            )
        if not (0.0 < float(self.step_fraction) <= 1.0):
            raise ValueError(f"step_fraction 必须在 (0,1]，收到 {self.step_fraction}")
        if int(self.sliding_window_batch_size) < 1:
            raise ValueError(f"sliding_window_batch_size 必须 >= 1，收到 {self.sliding_window_batch_size}")
        if not self.checkpoint_metric:
            raise ValueError("checkpoint_metric 不能为空")
        if self.checkpoint_metric == "val_loss":
            raise ValueError("正式 checkpoint 指标不得为 val_loss（固定为 val_positive_casewise_dice_mean）")
        if self.mode == "full_volume":
            # 指标与方向**硬冻结**（research_plan §11.1）：其他字段只作记录，不参与选模
            if self.checkpoint_metric != CHECKPOINT_METRIC_FULL_VOLUME:
                raise ValueError(
                    f"full_volume 模式的 checkpoint 指标已冻结为 {CHECKPOINT_METRIC_FULL_VOLUME}，"
                    f"不得改用 {self.checkpoint_metric!r}（记录的字段见 {FULL_VOLUME_METRIC_FIELDS}）"
                )
            if not self.maximize:
                raise ValueError(
                    f"full_volume 模式的 checkpoint 指标 {CHECKPOINT_METRIC_FULL_VOLUME} 为最大化指标，"
                    "maximize 必须为 True"
                )

    @property
    def maximize_metric(self) -> bool:
        return bool(self.maximize)

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "every_n_epochs": int(self.every_n_epochs),
            "checkpoint_metric": self.checkpoint_metric,
            "maximize": bool(self.maximize),
            "expected_cases": int(self.expected_cases),
            "expected_positive_cases": int(self.expected_positive_cases),
            "expected_negative_cases": int(self.expected_negative_cases),
            "step_fraction": float(self.step_fraction),
            "gaussian": bool(self.gaussian),
            "mirror_tta": bool(self.mirror_tta),
            "sliding_window_batch_size": int(self.sliding_window_batch_size),
            "inner_patch_progress": bool(self.inner_patch_progress),
            "save_case_metrics": bool(self.save_case_metrics),
        }


@dataclass(frozen=True)
class CaseResult:
    """单个 validation 病例的体素级统计与 Dice（预处理空间）。"""

    case_id: str
    gt_fg_voxels: int
    pred_fg_voxels: int
    tp: int
    fp: int
    fn: int

    @property
    def is_positive(self) -> bool:
        return self.gt_fg_voxels > 0

    @property
    def dice(self) -> float:
        """完整体积 Dice；阳性病例预测为空 → 0.0；空—空 → 0.0（不记为 1）。"""
        denominator = 2 * self.tp + self.fp + self.fn
        if denominator == 0:
            return 0.0
        return 2.0 * self.tp / denominator

    def as_row(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "is_positive": int(self.is_positive),
            "gt_fg_voxels": self.gt_fg_voxels,
            "pred_fg_voxels": self.pred_fg_voxels,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "dice": self.dice,
        }


class FullVolumeValidator:
    """遍历全部 validation 病例做完整 3D 滑窗推理，返回 epoch 级汇总指标。

    接口约定（trainer 侧只依赖 `.run(model, epoch=...)`）：duck-typing 即可替换为测试用假验证器。
    """

    def __init__(
        self,
        store: PreprocessedStore,
        case_ids: Sequence[str],
        *,
        patch_size: Sequence[int],
        device: str | torch.device = "cpu",
        protocol: ValidationProtocol | None = None,
        metrics_dir: str | Path | None = None,
        progress: bool = True,
        desc: str = "val full-volume",
        amp: bool = False,
        foreground_label: int = 1,
        include_zonal_prior: bool | None = None,
    ) -> None:
        self.protocol = protocol or ValidationProtocol()
        self.protocol.validate()
        self.store = store
        self.case_ids = tuple(str(c) for c in case_ids)
        if len(set(self.case_ids)) != len(self.case_ids):
            duplicates = sorted({c for c in self.case_ids if self.case_ids.count(c) > 1})
            raise ValueError(f"validation case_ids 存在重复: {duplicates[:5]}（共 {len(duplicates)} 个）")
        if len(self.case_ids) != self.protocol.expected_cases:
            raise RuntimeError(
                f"validation 病例数 {len(self.case_ids)} != 冻结期望 {self.protocol.expected_cases}；"
                "拒绝开始全体积验证（请检查 fold 0 划分）"
            )
        if len(patch_size) != 3 or any(int(p) < 1 for p in patch_size):
            raise ValueError(f"patch_size 必须为长度 3 的正整数，收到 {patch_size}")
        self.patch_size = tuple(int(p) for p in patch_size)
        self.device = torch.device(device)
        self.metrics_dir = Path(metrics_dir) if metrics_dir is not None else None
        self.progress = bool(progress)
        self.desc = str(desc)
        self.amp_requested = bool(amp)
        self.amp = bool(amp) and self.device.type == "cuda"
        self.foreground_label = int(foreground_label)
        #: PZ/TZ prior：None = 由模型声明自动判定（run 时会与模型的 REQUIRED_INPUT_KEYS 核对）
        if include_zonal_prior is not None and include_zonal_prior and getattr(store, "zonal_prior_root", None) is None:
            raise ValueError(
                "include_zonal_prior=True 但 store 未配置 zonal_prior_root/zonal_prior_source"
            )
        self.include_zonal_prior = include_zonal_prior
        #: 最近一次 run 的病例级 CSV 路径（save_case_metrics=False 时为 None）
        self.last_case_csv_path: Path | None = None

    # ------------------------------------------------------------------ 工具
    def case_csv_path(self, epoch: int) -> Path | None:
        if self.metrics_dir is None:
            return None
        return self.metrics_dir / "validation_cases" / f"epoch_{int(epoch):04d}.csv"

    @staticmethod
    def _autocast(amp: bool):
        import contextlib

        if amp:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return contextlib.nullcontext()

    def _aggregate(self, results: Sequence[CaseResult]) -> dict[str, float]:
        positives = [r for r in results if r.is_positive]
        negatives = [r for r in results if not r.is_positive]
        if len(results) != self.protocol.expected_cases:
            raise RuntimeError(
                f"病例覆盖检查失败：实际评估 {len(results)} 例 != 期望 {self.protocol.expected_cases} 例"
            )
        if len(positives) != self.protocol.expected_positive_cases or len(negatives) != self.protocol.expected_negative_cases:
            raise RuntimeError(
                f"阳性/阴性覆盖检查失败：实际 {len(positives)} 阳性 / {len(negatives)} 阴性 != 期望 "
                f"{self.protocol.expected_positive_cases} / {self.protocol.expected_negative_cases}；"
                "拒绝保存 best（请检查划分与标签）"
            )
        dice_values = [r.dice for r in positives]
        if not dice_values:
            raise RuntimeError("没有阳性病例，无法计算 val_positive_casewise_dice_mean")
        tp_total = sum(r.tp for r in results)
        fp_total = sum(r.fp for r in results)
        fn_total = sum(r.fn for r in results)
        micro_denominator = 2 * tp_total + fp_total + fn_total
        micro_dice = 0.0 if micro_denominator == 0 else 2.0 * tp_total / micro_denominator
        negative_fp_voxels = sum(r.fp for r in negatives)
        negative_with_fp = sum(1 for r in negatives if r.fp > 0)
        metrics = {
            "val_positive_casewise_dice_mean": float(statistics.fmean(dice_values)),
            "val_positive_casewise_dice_median": float(statistics.median(dice_values)),
            "val_micro_dice": float(micro_dice),
            "val_positive_cases": float(len(positives)),
            "val_negative_cases": float(len(negatives)),
            "val_cases_evaluated": float(len(results)),
            "val_negative_fp_case_rate": float(negative_with_fp / len(negatives)) if negatives else 0.0,
            "val_fp_voxels_per_negative_exam": float(negative_fp_voxels / len(negatives)) if negatives else 0.0,
        }
        for key, value in metrics.items():
            if not math.isfinite(float(value)):
                raise RuntimeError(f"验证指标 {key} 非有限值: {value}")
        return metrics

    def _write_case_metrics(self, epoch: int, results: Sequence[CaseResult]) -> Path | None:
        path = self.case_csv_path(epoch)
        if path is None or not self.protocol.save_case_metrics:
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [r.as_row() for r in results]
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        return path

    # ------------------------------------------------------------------ 主体
    def run(self, model: torch.nn.Module, *, epoch: int = 0) -> dict[str, float]:
        """执行一次完整 validation（全部 223 例，每例恰好一次）。"""
        t0 = time.time()
        was_training = bool(model.training)
        ds_supported = hasattr(model, "set_deep_supervision")
        ds_state = bool(getattr(model, "deep_supervision", True)) if ds_supported else None

        # PZ/TZ prior 一致性：模型声明与 validator 配置必须一致（不一致立即失败，不得静默兜底）
        requires_prior = "zonal_prior" in tuple(getattr(model, "REQUIRED_INPUT_KEYS", ("image",)))
        if self.include_zonal_prior is None:
            self.include_zonal_prior = requires_prior
        elif bool(self.include_zonal_prior) != requires_prior:
            raise RuntimeError(
                f"validator.include_zonal_prior={self.include_zonal_prior} 与模型 "
                f"REQUIRED_INPUT_KEYS={'zonal_prior' in tuple(getattr(model, 'REQUIRED_INPUT_KEYS', ('image',)))} 不一致；"
                "请在装配阶段修正（禁止用全零 prior 或忽略 prior）"
            )
        if self.include_zonal_prior and getattr(self.store, "zonal_prior_root", None) is None:
            raise RuntimeError("需要 PZ/TZ prior，但 store 未配置 zonal_prior_root/zonal_prior_source")

        results: list[CaseResult] = []
        visited: list[str] = []
        model.eval()
        if ds_supported:
            model.set_deep_supervision(False)  # 验证只用最高分辨率输出
        try:
            with torch.inference_mode(), self._autocast(self.amp):
                bar = make_progress(
                    self.case_ids, total=len(self.case_ids), desc=f"{self.desc} e{epoch + 1}", disable=not self.progress
                )
                for case_id in bar:
                    # 仅需要 prior 时传 kwarg，保持与 duck-typing 假 store（只实现 get_case(case_id)）兼容
                    case = (
                        self.store.get_case(case_id, include_zonal_prior=True)
                        if self.include_zonal_prior
                        else self.store.get_case(case_id)
                    )
                    if case.case_id != case_id:
                        raise RuntimeError(f"store 返回的 case_id {case.case_id!r} 与请求 {case_id!r} 不一致")
                    image = torch.from_numpy(np.ascontiguousarray(case.data))[None]
                    prior = None
                    if self.include_zonal_prior:
                        if case.zonal_prior is None:
                            raise RuntimeError(
                                f"{case_id}: 缺 PZ/TZ prior（禁止用全零 prior 兜底），请检查 sidecar 物化"
                            )
                        prior = torch.from_numpy(np.ascontiguousarray(case.zonal_prior))[None]
                        if tuple(prior.shape[-3:]) != tuple(image.shape[-3:]):
                            raise RuntimeError(
                                f"{case_id}: prior {tuple(prior.shape[-3:])} 与 image {tuple(image.shape[-3:])} "
                                "空间尺寸不一致（网格不同立即失败）"
                            )
                    logits = sliding_window_predict(
                        model,
                        image,
                        zonal_prior=prior,
                        patch_size=self.patch_size,
                        step_fraction=self.protocol.step_fraction,
                        gaussian=self.protocol.gaussian,
                        mirror_tta=self.protocol.mirror_tta,
                        device=self.device,
                        batch_size=self.protocol.sliding_window_batch_size,
                        # 内层 patch 进度默认关闭；`--no-progress` 时必须一并关闭（不产生任何进度输出）
                        progress=bool(self.protocol.inner_patch_progress) and self.progress,
                        desc=f"sw {case_id}",
                    )
                    if not torch.isfinite(logits).all():
                        raise RuntimeError(f"{case_id}: 滑窗推理输出包含 NaN/Inf，拒绝继续（不保存 best）")
                    gt = torch.from_numpy(case.seg[0]).to(torch.bool)
                    pred = logits.argmax(dim=1)[0].cpu() == self.foreground_label
                    tp = int((pred & gt).sum().item())
                    fp = int((pred & ~gt).sum().item())
                    fn = int((~pred & gt).sum().item())
                    result = CaseResult(
                        case_id=case_id,
                        gt_fg_voxels=int(gt.sum().item()),
                        pred_fg_voxels=int(pred.sum().item()),
                        tp=tp,
                        fp=fp,
                        fn=fn,
                    )
                    results.append(result)
                    visited.append(case_id)
                    if hasattr(bar, "set_postfix"):
                        positives = [r.dice for r in results if r.is_positive]
                        running_mean = float(statistics.fmean(positives)) if positives else float("nan")
                        bar.set_postfix({"pos_dice": f"{running_mean:.4f}", "case": case_id})
        finally:
            if ds_supported and ds_state is not None:
                model.set_deep_supervision(ds_state)
            model.train(was_training)

        if visited != list(self.case_ids):
            missing = sorted(set(self.case_ids) - set(visited))
            duplicated = sorted({c for c in visited if visited.count(c) > 1})
            raise RuntimeError(
                f"validation 覆盖检查失败：missing={missing[:5]} duplicated={duplicated[:5]} "
                f"(visited={len(visited)} expected={len(self.case_ids)})"
            )
        metrics = self._aggregate(results)
        metrics["validation_elapsed_sec"] = float(time.time() - t0)
        if not math.isfinite(metrics["validation_elapsed_sec"]):
            raise RuntimeError("validation_elapsed_sec 非有限值")
        self.last_case_csv_path = self._write_case_metrics(epoch, results)
        return metrics


def build_protocol_from_config(config: Mapping[str, Any]) -> ValidationProtocol:
    """从配置字典构造并校验 `ValidationProtocol`（供配置层复用）。"""
    protocol = ValidationProtocol(
        mode=str(config.get("mode", "full_volume")),
        every_n_epochs=int(config.get("every_n_epochs", 1)),
        checkpoint_metric=str(config.get("checkpoint_metric", CHECKPOINT_METRIC_FULL_VOLUME)),
        maximize=bool(config.get("maximize", True)),
        expected_cases=int(config.get("expected_cases", 223)),
        expected_positive_cases=int(config.get("expected_positive_cases", 63)),
        expected_negative_cases=int(config.get("expected_negative_cases", 160)),
        step_fraction=float(config.get("step_fraction", 0.5)),
        gaussian=bool(config.get("gaussian", True)),
        mirror_tta=bool(config.get("mirror_tta", False)),
        sliding_window_batch_size=int(config.get("sliding_window_batch_size", 1)),
        inner_patch_progress=bool(config.get("inner_patch_progress", False)),
        save_case_metrics=bool(config.get("save_case_metrics", True)),
    )
    protocol.validate()
    return protocol


__all__ = [
    "CHECKPOINT_METRIC_FULL_VOLUME",
    "FULL_VOLUME_METRIC_FIELDS",
    "VALIDATION_MODES",
    "CaseResult",
    "FullVolumeValidator",
    "ValidationProtocol",
    "build_protocol_from_config",
]
