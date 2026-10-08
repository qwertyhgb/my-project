#!/usr/bin/env python3
"""统一论文级病灶分割评估入口（离线读取 nnU-Net validation 产物）。

设计边界
--------
- **只读**：summary 模式只读 ``FOLD_DIR/validation/summary.json``；full 模式在此之上读取该文件
  ``metric_per_case`` 指向的 prediction / reference NIfTI。绝不重写 nnU-Net 的 validation、
  inference 或 prediction，绝不训练模型，绝不在评估中重采样掩膜，绝不静默修改掩膜。
- **统一口径**：多个模型使用完全相同的指标定义、病例集合与 bootstrap 设置比较。
- **只评估病灶分割**：**AUROC / average precision / FROC / PI-CAI challenge score 不是本工具的
  核心指标，本工具不计算它们。**
- **fail-closed**：任一模型失败、任一病例几何不一致、病例集合/身份/``n_ref`` 不一致时，
  先打印完整错误汇总，再非零退出，且**不发布**输出 JSON。

两种模式
--------
``summary``（默认）只读 ``summary.json``，输出体素计数派生的全部指标，不读 NIfTI。
``full`` 在其上读取 NIfTI，追加几何验证、物理体积（mm³）、RVE/ARVE、表面指标
（HD95 / ASSD / NSD@τ，单位 mm）、病灶体积分层、连通域失败分析与病灶实例级指标；必须**显式**提供
``--nsd-tolerance-mm``，本工具不设置未经论证的默认值。

病灶实例级指标（``lesion_instance_metrics``，仅 full 模式，协议在看结果前冻结）
----------------------------------------------------------------------------
- 实例 = 3D 连通域（``CONNECTIVITY = 1``，6-邻域 / face connectivity），不改成 18/26 邻域；
- 候选匹配 = 至少 1 个真实重叠体素（intersection ≥ 1）；
- 一对一 matching 按字典序目标：先最大化匹配数（maximum cardinality），再最大化总
  intersection 体素数，最后按 (reference id, prediction id) 升序确定性 tie-break；
  **不**以 Dice 为 matching 目标（matched-lesion Dice 是报告终点，见 Development Log）；
- 主指标命名为 ``lesion_sensitivity_any_overlap``，未匹配 reference lesion 不从分母删除；
- 探索性大小分层：< 500 / 500–1000 / > 1000 mm³（不是临床风险类别）；
- 不使用任何预测后处理（最小团块过滤 / 最大团块 / 形态学 / 阈值优化一律禁止）。

关键口径（避免歧义）
--------------------
- ``positive_voxel_precision`` = ΣTP / (ΣTP + ΣFP)，**仅阳性病例**（论文核心口径）。
- ``all_prediction_voxel_precision`` = ΣTP / (ΣTP + ΣFP + ΣFP_阴性)，分母含阴性病例假阳体素。
  两者**不得**都笼统称为 "precision"。
- 阳性病例完全漏分时 Dice 记为 **0**，不得排除。
- GT 非空、预测为空：NSD = 0，HD95/ASSD = null，同时计入 missed case。
- GT 与预测均为空：不纳入阳性病灶表面统计。
- NSD@τ 使用 DeepMind ``surface-distance`` 0.1 官方实现：按**物理表面测度 μ（surfel 面积，
  mm²）加权**，不是表面体素计数；spacing 为数组轴序 (z, y, x) 的真实 mm；缺少该包时
  fail-closed（不静默退回旧算法）。
- 体素数量一律标注 **voxel-based**，不得冒充 mm³。
- ``case_lesion_burden_mm3`` 是**病例总阳性体积**，不是单病灶大小。

用法
----
见 ``README.md`` 的「病灶分割评估」小节：那里是命令的**唯一**来源，包含 summary / full 两种模式的
完整可复制命令、输出目录约定与运行前提。此处不再维护第二份命令，也**不提供、不推荐**任何 NSD
容差或体积分层阈值取值。

``--nsd-tolerance-mm`` 与 ``--volume-thresholds-mm3`` 都必须由研究者在运行前自行确定；本脚本只
校验它们的形式（正有限数 / 严格递增且无重复），不设任何默认值。``full`` 模式是长任务，只能由
研究者在训练结束后运行。
"""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import math
import os
import sys
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

#项目包路径引导：``scripts/env_nnunet.sh`` 已经把 ``<root>/src`` 放进 PYTHONPATH；这里再补一次，
# 使本脚本在未 source 环境、直接被 ``python scripts/evaluate_segmentation.py`` 调用时同样可用
# （评估脚本必须能在纯 CPU 环境下独立运行，不依赖 nnU-Net 运行时校验）。
_PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(_PROJECT_SRC))

#指标协议常量、纯函数与病灶实例级匹配全部从 ``zonal_reliability_fusion.evaluation`` 导入。
# 该包是论文评价体系的**唯一**定义来源（docs/Evaluation_Protocol.md）；本脚本只保留
# CLI、产物读取、模型间可比性检查、编排与落盘，不重复定义任何指标。
from zonal_reliability_fusion.evaluation import (
    CONNECTIVITY,
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_SEED,
    DICE_MATCH_TOLERANCE,
    LESION_SIZE_MEDIUM_MAX_MM3,
    LESION_SIZE_SMALL_MAX_MM3,
    LESION_SIZE_STRATUM_KEYS,
    SCHEMA_VERSION,
    EvaluationError,
    aggregate_lesion_instance_metrics as _aggregate_lesion_instance_metrics,
    bootstrap_ci_mean,
    bootstrap_ci_paired_mean,
    bootstrap_ci_paired_ratio,
    component_analysis,
    dice_from_counts as _recompute_dice,
    dist_stats as _dist_stats,
    finite as _finite,
    json_safe as _json_safe,
    lesion_instance_metrics,
    lesion_size_stratum as _lesion_size_stratum,
    load_mask_pair,
    nsd_surface_area,
    pred_component_stats as _pred_component_stats,
    ratio as _ratio,
    recompute_case_counts,
    spacing_zyx as _spacing_zyx,
    surface_distance_metrics as _surface_distance_metrics,
    surface_metrics,
    verify_case_counts as _verify_case_counts,
    voxel_volume_mm3 as _voxel_volume_mm3,
)
from zonal_reliability_fusion.evaluation.lesion_metrics import (
    _match_lesion_instances,
    _max_weight_matching_exact_size,
    _min_cost_flow_unit_matching,
    _validate_lesion_matching,
)

#: 判定「病例间 Dice 变化是否为平局」的容差（仅用于 improved/tied/worsened 计数）
TIE_TOLERANCE = 1e-12
#: 错误明细的显示上限；超过时汇总文本会明确写出「仅显示前 N 条，另有 M 条」
ERROR_DISPLAY_CAP = 50

#: ``RunStats`` 的**阶段**：与 ``unit`` 一起决定 ok / failed / skipped 的语义
PHASE_PREFLIGHT = "preflight"
PHASE_MODEL_LOADING = "model_loading"
PHASE_COMPARABILITY = "comparability"
PHASE_FULL_CASES = "full_cases"
PHASE_PUBLISHING = "publishing"
PHASE_COMPLETED = "completed"

#: 计数的**单位**；同一次汇总里的 ok / failed / skipped 必须共享同一个 unit
UNIT_MODEL = "model"
UNIT_MODEL_CASE = "model-case"

#: 计数在"该阶段不存在这个量"时的渲染文本（不得伪造成 0）
NOT_APPLICABLE = "not_applicable"


def _bump(value: int | None) -> int:
    """把 ``None``（尚未启用）当作 0 再加一；仅用于记账。"""
    return (0 if value is None else value) + 1


def _fmt_count(value: int | None) -> str:
    """计数渲染：``None`` 表示该阶段不存在这个量，显式写 not_applicable 而不是 0。"""
    return NOT_APPLICABLE if value is None else str(value)


class RunStats:
    """一次运行的**阶段与计数事实**，只用于结束汇总。

    严格约束：这些字段**不参与**指标计算，也**不影响** fail-closed 判定（计数本身不会改变
    是否发布 JSON，也不会改变是否抛错）。成功与失败两条路径输出同一组字段。

    单位**不能由 ``mode`` 推断**：``full`` 可能在模型加载或可比性检查阶段就失败，此时
    ``ok`` / ``failed`` 仍然是**模型**计数，把它标成 model-case 是错的。因此单位由 ``phase``
    强制决定：

    ==================  ================  ===========================================
    phase               unit              ok / failed / skipped
    ==================  ================  ===========================================
    ``preflight``       not_applicable    尚未开始计数（参数校验阶段）
    ``model_loading``   ``model``         已加载 / 加载失败的**模型**数；skipped 不适用
    ``comparability``   ``model``         沿用模型计数（**不得**改标为 model-case）
    ``full_cases``      ``model-case``    逐例通过 / 失败 / 未进入检查的**对**数
    ``publishing``      沿用上一阶段      沿用上一阶段
    ``completed``       沿用上一阶段      沿用上一阶段
    ==================  ================  ===========================================

    只有 :meth:`begin_full_cases` 才切到 ``model-case`` 并**重置**计数。``skipped`` 在不可知时
    保持 ``None`` 并渲染为 ``not_applicable``，绝不伪造成 0。
    """

    def __init__(self, mode: str, output: str | None) -> None:
        self.mode = mode
        self.output = output
        self.phase = PHASE_PREFLIGHT
        self.unit: str | None = None
        self.ok: int | None = None
        self.failed: int | None = None
        self.skipped: int | None = None
        self._total_pairs: int | None = None
        # 单调时钟：耗时统计不受系统时间调整影响
        self._t0 = time.monotonic()

    # ------------------------------------------------ 阶段切换（纯记账，无任何判定作用）
    def begin_model_loading(self) -> None:
        """进入模型加载阶段：``unit=model``，ok / failed 都表示模型数。"""
        self.phase = PHASE_MODEL_LOADING
        self.unit = UNIT_MODEL
        self.ok = 0
        self.failed = 0
        self.skipped = None  # 该阶段不存在"被跳过的 model"这一概念

    def model_loaded(self) -> None:
        self.ok = _bump(self.ok)

    def model_failed(self) -> None:
        self.failed = _bump(self.failed)

    def begin_comparability(self) -> None:
        """可比性检查阶段：**沿用 model 单位**与已加载的模型计数。"""
        self.phase = PHASE_COMPARABILITY
        self.unit = UNIT_MODEL

    def begin_full_cases(self, total_pairs: int) -> None:
        """真正进入逐例循环之前，才切到 ``unit=model-case`` 并重置逐例计数。"""
        self.phase = PHASE_FULL_CASES
        self.unit = UNIT_MODEL_CASE
        self.ok = 0
        self.failed = 0
        self.skipped = None
        self._total_pairs = total_pairs

    def full_case_ok(self) -> None:
        self.ok = _bump(self.ok)

    def full_case_failed(self) -> None:
        self.failed = _bump(self.failed)

    def end_full_cases(self) -> None:
        """逐例循环结束后结算 skipped；总对数是已知的，不是猜测。"""
        if self._total_pairs is None:
            return
        ok = 0 if self.ok is None else self.ok
        failed = 0 if self.failed is None else self.failed
        self.skipped = self._total_pairs - ok - failed

    def begin_publishing(self) -> None:
        """进入结果汇总与落盘阶段；单位与计数沿用上一阶段。"""
        self.phase = PHASE_PUBLISHING

    def mark_completed(self) -> None:
        """整轮执行完毕（含落盘）；单位与计数沿用上一阶段。"""
        self.phase = PHASE_COMPLETED

    @property
    def elapsed_s(self) -> float:
        """自 ``main()`` 建立统计起经过的秒数（``time.monotonic()`` 差值）。"""
        return time.monotonic() - self._t0

    def render(self, status: str) -> str:
        """结构化结束汇总（独立于 tqdm 进度条，进度条不能替代它）。"""
        output = self.output if self.output else "(未请求写出 JSON)"
        unit = self.unit if self.unit is not None else NOT_APPLICABLE
        return "\n".join(
            [
                "=== 评估结束汇总 ===",
                f"  status  : {status}",
                f"  mode    : {self.mode}",
                f"  phase   : {self.phase}",
                f"  unit    : {unit}",
                f"  ok      : {_fmt_count(self.ok)}",
                f"  failed  : {_fmt_count(self.failed)}",
                f"  skipped : {_fmt_count(self.skipped)}",
                f"  elapsed : {self.elapsed_s:.2f} s",
                f"  output  : {output}",
            ]
        )


# --------------------------------------------------------------------------- 通用工具
def _now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _publish_json(path: Path, payload: dict) -> None:
    """原子发布：临时文件 + ``os.replace``；已存在则拒绝覆盖（fail-closed）。"""
    if path.exists():
        raise EvaluationError(f"输出文件已存在，拒绝静默覆盖: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            # allow_nan=False 是硬约束：NaN/Infinity 泄漏时直接失败，不写出非法 JSON
            json.dump(payload, fh, indent=2, ensure_ascii=False, allow_nan=False)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _case_id_from_path(path_str: str) -> str:
    """从 prediction/reference 文件名稳定提取 case id（去掉目录与 .nii/.nii.gz 后缀）。"""
    name = os.path.basename(str(path_str))
    for suffix in (".nii.gz", ".nii"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return os.path.splitext(name)[0]


def _foreground_metrics(metrics: dict) -> dict:
    """取 ``metric_per_case[*]["metrics"]`` 里唯一的前景标签条目（background 记 0）。"""
    keys = [k for k in metrics if str(k) not in ("0",)]
    if len(keys) != 1:
        raise EvaluationError(
            f"metrics 中前景标签不唯一（现有键 {list(metrics)}）；"
            "本工具只支持单一前景标签的分割评估"
        )
    return metrics[keys[0]]


def _format_error_report(
    title: str, errors: Sequence[str], cap: int = ERROR_DISPLAY_CAP
) -> str:
    """统一的错误汇总文本：写明总条数；超过显示上限时明确标注未显示条数。"""
    lines = [f"{title}（共 {len(errors)} 项）"]
    if len(errors) <= cap:
        lines.extend(f"  {e}" for e in errors)
    else:
        lines.extend(f"  {e}" for e in errors[:cap])
        lines.append(f"  ... 仅显示前 {cap} 条，另有 {len(errors) - cap} 条未显示")
    return "\n".join(lines)


def _strict_non_negative_int(metrics: dict, key: str, model_name: str, cid: str) -> int:
    """从 summary 读取一个**非负整数**计数；禁止 int() 静默截断小数、NaN、负数。"""
    if key not in metrics:
        raise EvaluationError(f"[{model_name}] {cid}: metrics 缺少 {key}")
    raw = metrics[key]
    if isinstance(raw, bool) or not isinstance(
        raw, (int, float, np.integer, np.floating)
    ):
        raise EvaluationError(f"[{model_name}] {cid}: {key} 不是数值（{raw!r}）")
    value = float(raw)
    if not math.isfinite(value):
        raise EvaluationError(f"[{model_name}] {cid}: {key} 不是有限数（{raw!r}）")
    if value != int(value):
        raise EvaluationError(
            f"[{model_name}] {cid}: {key} 必须是整数，实际为 {raw!r}（禁止静默截断小数）"
        )
    if value < 0:
        raise EvaluationError(f"[{model_name}] {cid}: {key} 不能为负（{raw!r}）")
    return int(value)


def _check_reported_dice(
    reported: Any, recomputed: float | None, model_name: str, cid: str
) -> None:
    """summary 中若给出 Dice，必须与重算值一致；真阴（空-空）必须为空/NaN。"""
    reported_finite = _finite(reported)
    if recomputed is None:
        if reported_finite is not None:
            raise EvaluationError(
                f"[{model_name}] {cid}: 真阴病例（n_ref=0 且 n_pred=0）的 Dice 应为空/NaN，"
                f"实际为 {reported!r}（疑似被人为记为 1 之类）"
            )
        return
    if reported_finite is None:
        raise EvaluationError(
            f"[{model_name}] {cid}: summary 未给出 Dice，但由 TP/FP/FN 重算得到 "
            f"{recomputed:.10f}"
        )
    if abs(reported_finite - recomputed) > DICE_MATCH_TOLERANCE:
        raise EvaluationError(
            f"[{model_name}] {cid}: summary Dice={reported_finite:.10f} 与由 TP/FP/FN "
            f"重算的 {recomputed:.10f} 不一致（容差 {DICE_MATCH_TOLERANCE:g}）"
        )


# --------------------------------------------------------------------------- 读取产物
def load_model_cases(model_name: str, fold_dir: Path) -> tuple[dict, dict]:
    """读取单个模型的 ``validation/summary.json``，返回 (case id -> 体素计数, 原始 doc)。"""
    summary_path = fold_dir / "validation" / "summary.json"
    if not summary_path.is_file():
        raise EvaluationError(
            f"[{model_name}] 缺少 validation/summary.json: {summary_path}"
        )
    try:
        with open(summary_path, encoding="utf-8") as fh:
            doc = json.load(fh)
    except json.JSONDecodeError as exc:
        raise EvaluationError(
            f"[{model_name}] summary.json 不是合法 JSON: {exc}"
        ) from exc

    entries = doc.get("metric_per_case")
    if not isinstance(entries, list) or not entries:
        raise EvaluationError(f"[{model_name}] summary.json 缺少非空 metric_per_case")

    cases: dict[str, dict] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise EvaluationError(f"[{model_name}] metric_per_case 元素不是对象")
        ref_path, pred_path = entry.get("reference_file"), entry.get("prediction_file")
        if not ref_path or not pred_path:
            raise EvaluationError(
                f"[{model_name}] metric_per_case 缺少 prediction_file/reference_file"
            )
        cid, pred_cid = _case_id_from_path(ref_path), _case_id_from_path(pred_path)
        if cid != pred_cid:
            raise EvaluationError(
                f"[{model_name}] case id 不一致: reference={cid!r} prediction={pred_cid!r}"
            )
        if cid in cases:
            raise EvaluationError(f"[{model_name}] case id 重复: {cid!r}")
        metrics = entry.get("metrics")
        if not isinstance(metrics, dict):
            raise EvaluationError(f"[{model_name}] {cid}: 缺少 metrics")
        try:
            m = _foreground_metrics(metrics)
        except EvaluationError as exc:
            raise EvaluationError(f"[{model_name}] {cid}: {exc}") from exc

        counts = {
            key: _strict_non_negative_int(m, key, model_name, cid)
            for key in ("TP", "FP", "FN", "TN", "n_pred", "n_ref")
        }
        if counts["TP"] + counts["FP"] != counts["n_pred"]:
            raise EvaluationError(
                f"[{model_name}] {cid}: TP+FP ({counts['TP']}+{counts['FP']}) "
                f"!= n_pred ({counts['n_pred']})"
            )
        if counts["TP"] + counts["FN"] != counts["n_ref"]:
            raise EvaluationError(
                f"[{model_name}] {cid}: TP+FN ({counts['TP']}+{counts['FN']}) "
                f"!= n_ref ({counts['n_ref']})"
            )
        # Dice 一律由 TP/FP/FN 重算；summary 中的 Dice 只用于一致性核对，不作为宏平均输入
        dice = _recompute_dice(
            counts["TP"], counts["FP"], counts["FN"], counts["n_ref"]
        )
        _check_reported_dice(m.get("Dice"), dice, model_name, cid)
        cases[cid] = {
            "case_id": cid,
            "reference_file": str(ref_path),
            "prediction_file": str(pred_path),
            **counts,
            "dice": dice,
            "has_reference": counts["n_ref"] > 0,
        }
    return cases, doc


def _positive(cases: dict) -> list[dict]:
    return sorted(
        (c for c in cases.values() if c["n_ref"] > 0), key=lambda c: c["case_id"]
    )


def _negative(cases: dict) -> list[dict]:
    return sorted(
        (c for c in cases.values() if c["n_ref"] == 0), key=lambda c: c["case_id"]
    )


# --------------------------------------------------------------------------- summary 指标
def summarize_model(
    model_name: str, fold_dir: Path, bootstrap_resamples: int, seed: int
) -> tuple[dict, dict]:
    """计算单个模型的 summary 级指标。返回 (报告 dict, 病例 dict)。"""
    cases, doc = load_model_cases(model_name, fold_dir)
    pos, neg = _positive(cases), _negative(cases)

    # --- 4.1 病例集合 ---
    case_counts = {
        "num_cases": len(cases),
        "num_positive_cases": len(pos),
        "num_negative_cases": len(neg),
        "case_ids_positive": [c["case_id"] for c in pos],
        "case_ids_negative": [c["case_id"] for c in neg],
    }
    reported = doc.get("foreground_mean") or {}
    nnunet_reported = {
        "foreground_mean_Dice": _finite(reported.get("Dice")),
        "foreground_mean_IoU": _finite(reported.get("IoU")),
        "foreground_mean_TP": _finite(reported.get("TP")),
        "foreground_mean_FP": _finite(reported.get("FP")),
        "foreground_mean_FN": _finite(reported.get("FN")),
        "foreground_mean_n_pred": _finite(reported.get("n_pred")),
        "foreground_mean_n_ref": _finite(reported.get("n_ref")),
        "note": (
            "nnU-Net 原生 foreground_mean 为逐病例 np.nanmean；分母包含「假阳阴性病例」记的 0，"
            "真阴病例被剔除。与本工具的阳性病例口径不同，仅供交叉核对。"
        ),
    }

    # --- 4.2 阳性主指标 ---
    pos_dice = [float(c["dice"]) for c in pos]
    tp = sum(c["TP"] for c in pos)
    fp = sum(c["FP"] for c in pos)
    fn = sum(c["FN"] for c in pos)
    n_pred_pos = sum(c["n_pred"] for c in pos)
    n_ref_pos = sum(c["n_ref"] for c in pos)
    positive_segmentation = {
        "positive_macro_dice_mean": _finite(np.mean(pos_dice)) if pos_dice else None,
        "positive_macro_dice_median": _finite(np.median(pos_dice))
        if pos_dice
        else None,
        "positive_macro_dice_q1": _finite(np.percentile(pos_dice, 25))
        if pos_dice
        else None,
        "positive_macro_dice_q3": _finite(np.percentile(pos_dice, 75))
        if pos_dice
        else None,
        "positive_macro_dice_bootstrap_ci95": bootstrap_ci_mean(
            pos_dice, bootstrap_resamples, seed
        ),
        "positive_micro_dice": _ratio(2 * tp, 2 * tp + fp + fn),
        "positive_micro_dice_ci95": bootstrap_ci_paired_ratio(
            [2 * c["TP"] for c in pos],
            [2 * c["TP"] + c["FP"] + c["FN"] for c in pos],
            bootstrap_resamples,
            seed,
        ),
        "numerator_denominator": {
            "macro_dice_n": len(pos_dice),
            "macro_dice_sum": _finite(np.sum(pos_dice)) if pos_dice else None,
            "micro_dice_numerator": 2 * tp,
            "micro_dice_denominator": 2 * tp + fp + fn,
        },
        "definitions": {
            "positive_macro_dice": "逐例 Dice 的宏平均；完全漏分的阳性病例记 0",
            "positive_micro_dice": "2*ΣTP / (2*ΣTP + ΣFP + ΣFN)，只汇总 GT 阳性病例",
        },
    }

    # --- 4.3 Recall / Precision ---
    neg_fp_voxels = sum(c["FP"] for c in neg)
    voxel_metrics = {
        "positive_voxel_recall": _ratio(tp, tp + fn),
        "positive_voxel_precision": _ratio(tp, tp + fp),
        "all_prediction_voxel_precision": _ratio(tp, tp + fp + neg_fp_voxels),
        "numerator_denominator": {
            "tp": tp,
            "fp_positive": fp,
            "fp_negative": neg_fp_voxels,
            "fn": fn,
            "recall_denominator": tp + fn,
            "positive_precision_denominator": tp + fp,
            "all_prediction_precision_denominator": tp + fp + neg_fp_voxels,
        },
        "definitions": {
            "positive_voxel_recall": "ΣTP / (ΣTP + ΣFN)，仅 GT 阳性病例",
            "positive_voxel_precision": "ΣTP / (ΣTP + ΣFP)，仅 GT 阳性病例（论文核心口径）",
            "all_prediction_voxel_precision": (
                "ΣTP / (ΣTP + ΣFP_阳性 + ΣFP_阴性)，分母含阴性病例假阳体素；"
                "阴性病例假阳单独在 negative_false_positive 报告中给出"
            ),
        },
    }

    # --- 4.4 完全漏分结构 ---
    overlap = [c for c in pos if c["TP"] > 0]
    missed = [c for c in pos if c["TP"] == 0]
    empty_pred = [c for c in pos if c["n_pred"] == 0]
    wrong_loc = [c for c in pos if c["TP"] == 0 and c["n_pred"] > 0]
    n_pos = len(pos)
    miss_structure = {
        "positive_overlap_cases": len(overlap),
        "positive_overlap_case_rate": _ratio(len(overlap), n_pos),
        "positive_missed_cases": len(missed),
        "positive_missed_case_rate": _ratio(len(missed), n_pos),
        "positive_empty_prediction_cases": len(empty_pred),
        "positive_empty_prediction_case_rate": _ratio(len(empty_pred), n_pos),
        "positive_wrong_location_cases": len(wrong_loc),
        "positive_wrong_location_case_rate": _ratio(len(wrong_loc), n_pos),
        "case_ids_missed": [c["case_id"] for c in missed],
        "case_ids_wrong_location": [c["case_id"] for c in wrong_loc],
        "definition": "完全漏分 = TP == 0；不因 Dice 或表面指标无定义而被排除",
    }

    # --- 4.5 阴性病例假阳性 ---
    neg_fp_cases = [c for c in neg if c["n_pred"] > 0]
    stats = _dist_stats([float(c["FP"]) for c in neg_fp_cases])
    negative_false_positive = {
        "negative_fp_cases": len(neg_fp_cases),
        "negative_fp_case_rate": _ratio(len(neg_fp_cases), len(neg)),
        "negative_clean_cases": len(neg) - len(neg_fp_cases),
        "negative_fp_voxels_total": int(neg_fp_voxels),
        "negative_fp_voxels_n": stats["count"],
        "negative_fp_voxels_mean": stats["mean"],
        "negative_fp_voxels_median": stats["median"],
        "negative_fp_voxels_q1": stats["q1"],
        "negative_fp_voxels_q3": stats["q3"],
        "negative_fp_voxels_max": stats["max"],
        "case_ids_false_positive": [c["case_id"] for c in neg_fp_cases],
        "definition": (
            "阴性病例 = n_ref == 0。体素分布统计作用于**出现假阳的阴性病例**"
            "（分母见 negative_fp_voxels_n）；真阴病例不参与 Dice 宏平均。"
        ),
    }

    # --- 4.6 体积关系（voxel-based） ---
    detected = overlap
    ratios = [c["n_pred"] / c["n_ref"] for c in detected if c["n_ref"] > 0]
    volume_metrics = {
        "units": "voxel-based",
        "note": "以下均为体素数量关系，不是 mm³；物理体积只在 full 模式由 spacing 计算",
        "positive_predicted_reference_voxel_ratio": _ratio(n_pred_pos, n_ref_pos),
        "detected_positive_predicted_reference_ratio_median": (
            _finite(np.median(ratios)) if ratios else None
        ),
        "detected_positive_predicted_reference_ratio_q1": (
            _finite(np.percentile(ratios, 25)) if ratios else None
        ),
        "detected_positive_predicted_reference_ratio_q3": (
            _finite(np.percentile(ratios, 75)) if ratios else None
        ),
        "detected_positive_n": len(detected),
        "totals_positive_cases": {
            "TP": int(tp),
            "FP": int(fp),
            "FN": int(fn),
            "n_pred": int(n_pred_pos),
            "n_ref": int(n_ref_pos),
        },
        "totals_negative_cases": {
            "FP": int(neg_fp_voxels),
            "n_pred": int(neg_fp_voxels),
        },
        "totals_all_cases": {
            "TP": int(tp),
            "FP": int(fp + neg_fp_voxels),
            "FN": int(fn),
            "n_pred": int(n_pred_pos + neg_fp_voxels),
            "n_ref": int(n_ref_pos),
        },
    }

    per_case = [
        {
            "case_id": c["case_id"],
            "has_reference": c["has_reference"],
            "n_ref": c["n_ref"],
            "n_pred": c["n_pred"],
            "TP": c["TP"],
            "FP": c["FP"],
            "FN": c["FN"],
            "TN": c["TN"],
            "dice": _finite(c["dice"]),
            "tp_over_n_ref": _ratio(c["TP"], c["n_ref"]),
            "tp_over_n_pred": _ratio(c["TP"], c["n_pred"]),
            "n_pred_over_n_ref": _ratio(c["n_pred"], c["n_ref"])
            if c["n_ref"] > 0
            else None,
        }
        for c in sorted(cases.values(), key=lambda c: c["case_id"])
    ]

    report = {
        "source": str(fold_dir),
        "summary_json": str(fold_dir / "validation" / "summary.json"),
        "case_counts": case_counts,
        "nnunet_reported": nnunet_reported,
        "positive_segmentation": positive_segmentation,
        "voxel_metrics": voxel_metrics,
        "miss_structure": miss_structure,
        "negative_false_positive": negative_false_positive,
        "volume_metrics": volume_metrics,
        "surface_metrics": None,
        "size_strata": None,
        "component_analysis": None,
        # 病灶实例级指标只在 full 模式填充（summary 模式不读 NIfTI，无法定义连通域实例）
        "lesion_instance_metrics": None,
        "per_case": per_case,
    }
    return report, cases


# --------------------------------------------------------------------------- 配对比较
def _require_comparable(
    reports: dict[str, dict], cases_by_model: dict[str, dict]
) -> None:
    """多模型可比性校验（fail-closed；不一致时禁止继续，也不允许退化为求交集）。"""
    names = list(reports)
    if len(names) < 2:
        return
    ref = names[0]
    ref_ids = set(cases_by_model[ref])
    problems: list[str] = []
    for name in names[1:]:
        ids = set(cases_by_model[name])
        if ids != ref_ids:
            only_ref, only_other = sorted(ref_ids - ids), sorted(ids - ref_ids)
            problems.append(
                f"[{name}] 相对 [{ref}] 的 case id 集合不一致："
                f"缺少 {len(only_ref)} 例、多出 {len(only_other)} 例；"
                f"缺少示例 {only_ref[:10]}；多出示例 {only_other[:10]}"
            )
            continue
        for cid in sorted(ref_ids):
            a, b = cases_by_model[ref][cid], cases_by_model[name][cid]
            if (a["n_ref"] > 0) != (b["n_ref"] > 0):
                problems.append(f"case {cid}: GT 阳性/阴性身份不一致")
            elif a["n_ref"] != b["n_ref"]:
                problems.append(
                    f"case {cid}: n_ref 不一致（{a['n_ref']} vs {b['n_ref']}）"
                )
    if problems:
        raise EvaluationError(
            "多模型不可比，禁止只比较交集后继续：\n"
            + _format_error_report("可比性冲突", problems)
        )


def paired_comparisons(
    reports: dict[str, dict],
    cases_by_model: dict[str, dict],
    bootstrap_resamples: int,
    seed: int,
) -> list[dict]:
    """对每一对模型给出配对比较（方向统一为 ``model_b - model_a``）。"""
    out: list[dict] = []
    names = list(reports)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a_name, b_name = names[i], names[j]
            a_cases, b_cases = cases_by_model[a_name], cases_by_model[b_name]
            pos_ids = sorted(cid for cid, c in a_cases.items() if c["n_ref"] > 0)
            neg_ids = sorted(cid for cid, c in a_cases.items() if c["n_ref"] == 0)

            dice_deltas, rows = [], []
            for cid in pos_ids:
                da = float(a_cases[cid]["dice"] or 0.0)
                db = float(b_cases[cid]["dice"] or 0.0)
                dice_deltas.append(db - da)
                rows.append(
                    {"case_id": cid, "dice_a": da, "dice_b": db, "delta": db - da}
                )

            improved = sum(1 for d in dice_deltas if d > TIE_TOLERANCE)
            worsened = sum(1 for d in dice_deltas if d < -TIE_TOLERANCE)
            tied = len(dice_deltas) - improved - worsened

            tp_a = sum(a_cases[c]["TP"] for c in pos_ids)
            fp_a = sum(a_cases[c]["FP"] for c in pos_ids)
            fn_a = sum(a_cases[c]["FN"] for c in pos_ids)
            tp_b = sum(b_cases[c]["TP"] for c in pos_ids)
            fp_b = sum(b_cases[c]["FP"] for c in pos_ids)
            fn_b = sum(b_cases[c]["FN"] for c in pos_ids)
            negvox_a = sum(a_cases[c]["FP"] for c in neg_ids)
            negvox_b = sum(b_cases[c]["FP"] for c in neg_ids)
            negfp_a = sum(1 for c in neg_ids if a_cases[c]["n_pred"] > 0)
            negfp_b = sum(1 for c in neg_ids if b_cases[c]["n_pred"] > 0)

            micro_a = _ratio(2 * tp_a, 2 * tp_a + fp_a + fn_a)
            micro_b = _ratio(2 * tp_b, 2 * tp_b + fp_b + fn_b)
            rec_a, rec_b = _ratio(tp_a, tp_a + fn_a), _ratio(tp_b, tp_b + fn_b)
            prec_a, prec_b = _ratio(tp_a, tp_a + fp_a), _ratio(tp_b, tp_b + fp_b)
            aprec_a = _ratio(tp_a, tp_a + fp_a + negvox_a)
            aprec_b = _ratio(tp_b, tp_b + fp_b + negvox_b)

            new_overlap = [
                cid
                for cid in pos_ids
                if a_cases[cid]["TP"] == 0 and b_cases[cid]["TP"] > 0
            ]
            lost_overlap = [
                cid
                for cid in pos_ids
                if a_cases[cid]["TP"] > 0 and b_cases[cid]["TP"] == 0
            ]
            missed_a = sum(1 for cid in pos_ids if a_cases[cid]["TP"] == 0)
            missed_b = sum(1 for cid in pos_ids if b_cases[cid]["TP"] == 0)

            ci = bootstrap_ci_paired_mean(dice_deltas, bootstrap_resamples, seed)

            def _delta(x, y):
                return None if x is None or y is None else float(y) - float(x)

            entry = {
                "model_a": a_name,
                "model_b": b_name,
                "delta_definition": "model_b - model_a",
                "positive_case_dice_deltas": rows,
                "mean_paired_dice_delta": _finite(np.mean(dice_deltas))
                if dice_deltas
                else None,
                "median_paired_dice_delta": (
                    _finite(np.median(dice_deltas)) if dice_deltas else None
                ),
                "paired_bootstrap_ci95": ci,
                "ci_excludes_zero": None
                if ci is None
                else bool(ci[0] > 0.0 or ci[1] < 0.0),
                "improved_cases": improved,
                "tied_cases": tied,
                "worsened_cases": worsened,
                "new_overlap_cases": len(new_overlap),
                "new_overlap_case_ids": new_overlap,
                "lost_overlap_cases": len(lost_overlap),
                "lost_overlap_case_ids": lost_overlap,
                "complete_miss_delta": missed_b - missed_a,
                "negative_fp_case_delta": negfp_b - negfp_a,
                "positive_micro_dice_delta": _delta(micro_a, micro_b),
                "positive_voxel_recall_delta": _delta(rec_a, rec_b),
                "positive_voxel_precision_delta": _delta(prec_a, prec_b),
                "all_prediction_voxel_precision_delta": _delta(aprec_a, aprec_b),
                "negatives": {
                    "positive_micro_dice": {"a": micro_a, "b": micro_b},
                    "positive_voxel_recall": {"a": rec_a, "b": rec_b},
                    "positive_voxel_precision": {"a": prec_a, "b": prec_b},
                    "all_prediction_voxel_precision": {"a": aprec_a, "b": aprec_b},
                    "negative_fp_cases": {"a": negfp_a, "b": negfp_b},
                    "complete_miss_cases": {"a": missed_a, "b": missed_b},
                },
                "interpretation": (
                    "no_clear_paired_improvement"
                    if (ci is not None and ci[0] <= 0.0 <= ci[1])
                    else "paired_bootstrap_ci_excludes_zero"
                ),
                "interpretation_rule": (
                    "置信区间跨 0 时只陈述「未观察到明确配对改善」，不宣称显著提升；"
                    "存在平局计数的 fixed tolerance = " + repr(TIE_TOLERANCE)
                ),
            }
            out.append(_json_safe(entry))
    return out


# --------------------------------------------------------------------------- 终端输出
def print_table(reports: dict[str, dict], comparisons: list[dict], mode: str) -> None:
    def fmt(v: Any, nd: int = 4) -> str:
        f = _finite(v)
        return "n/a" if f is None else f"{f:.{nd}f}"

    print()
    print(f"=== 病灶分割评估（mode={mode}）===")
    header = (
        f"{'model':<20}{'cases':>7}{'pos':>6}{'neg':>6}"
        f"{'macroDice':>11}{'medDice':>9}{'microDice':>11}"
        f"{'recall':>9}{'posPrec':>9}{'allPrec':>9}{'missed':>8}{'negFP':>7}"
    )
    print(header)
    print("-" * len(header))
    for name, rep in reports.items():
        cc, ps = rep["case_counts"], rep["positive_segmentation"]
        vm, ms = rep["voxel_metrics"], rep["miss_structure"]
        nf = rep["negative_false_positive"]
        print(
            f"{name:<20}{cc['num_cases']:>7}{cc['num_positive_cases']:>6}"
            f"{cc['num_negative_cases']:>6}"
            f"{fmt(ps['positive_macro_dice_mean']):>11}"
            f"{fmt(ps['positive_macro_dice_median']):>9}"
            f"{fmt(ps['positive_micro_dice']):>11}"
            f"{fmt(vm['positive_voxel_recall']):>9}"
            f"{fmt(vm['positive_voxel_precision']):>9}"
            f"{fmt(vm['all_prediction_voxel_precision']):>9}"
            f"{ms['positive_missed_cases']:>8}{nf['negative_fp_cases']:>7}"
        )
    if comparisons:
        print()
        print("--- 配对比较（delta = model_b - model_a，仅在阳性病例上）---")
        for cmp_ in comparisons:
            ci = cmp_["paired_bootstrap_ci95"]
            ci_txt = "n/a" if ci is None else f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"
            print(
                f"{cmp_['model_a']} -> {cmp_['model_b']}: "
                f"meanDeltaDice={fmt(cmp_['mean_paired_dice_delta'])} CI95={ci_txt} "
                f"(improved/tied/worsened={cmp_['improved_cases']}/"
                f"{cmp_['tied_cases']}/{cmp_['worsened_cases']}), "
                f"newOverlap={cmp_['new_overlap_cases']} lostOverlap={cmp_['lost_overlap_cases']}, "
                f"missDelta={cmp_['complete_miss_delta']:+d}, "
                f"negFPCaseDelta={cmp_['negative_fp_case_delta']:+d} "
                f"[{cmp_['interpretation']}]"
            )
    print()


# --------------------------------------------------------------------------- full 模式
# 几何 / 体积 / 连通域 / 表面距离 / 病灶实例级匹配的**纯函数**全部来自
# ``zonal_reliability_fusion.evaluation``（见文件顶部 import）。本节只保留
# full 模式的逐例编排 ``_build_full_case_entry`` 及其后续流程。


def _build_full_case_entry(cid: str, case: dict, tolerance_mm: float) -> dict:
    """完整处理一例并返回其 full entry；**不在此处捕获异常**。

    流程：读取 prediction/reference 掩膜 → 几何与 0/1 标签校验 → 阴性 reference 必须为空
    → 由掩膜重算六项计数并与 summary.json 逐项核对 → 物理体积 → ``surface_metrics``
    → ``component_analysis``（阳性）或 ``_pred_component_stats``（假阳阴性）
    → ``lesion_instance_metrics``（**全部**病例：阳性做完整实例匹配；阴性病例
    reference_lesions=0、其 prediction components 全部记为 unmatched 假阳）。

    函数要么返回**完整构造**的 entry，要么抛出异常。``mask_verified=True`` 只在全部步骤
    成功后写入，因此失败病例绝不会留下伪造的"已验证"结果。
    """
    pred, ref, spacing = load_mask_pair(case["prediction_file"], case["reference_file"])

    # 阴性 reference 必须实际为空
    if case["n_ref"] == 0 and bool(ref.any()):
        raise EvaluationError(
            "summary 声明 n_ref=0，但 reference 掩膜非空"
            f"（实际 {int(np.count_nonzero(ref))} 体素）"
        )

    recomputed = recompute_case_counts(pred, ref)
    mismatch = _verify_case_counts(case, recomputed)
    if mismatch is not None:
        raise EvaluationError(mismatch)

    vox_vol = _voxel_volume_mm3(spacing)
    ref_vol = recomputed["n_ref"] * vox_vol
    pred_vol = recomputed["n_pred"] * vox_vol
    entry: dict = {
        "case_id": cid,
        "has_reference": case["n_ref"] > 0,
        "spacing_xyz": list(spacing),
        "voxel_volume_mm3": vox_vol,
        "reference_volume_mm3": ref_vol,
        "prediction_volume_mm3": pred_vol,
        "mask_verified": True,
    }
    if case["n_ref"] > 0:
        entry.update(
            rve=_ratio(pred_vol - ref_vol, ref_vol),
            arve=_ratio(abs(pred_vol - ref_vol), ref_vol),
            surface=surface_metrics(pred, ref, spacing, tolerance_mm),
            components=component_analysis(pred, ref, vox_vol, cid),
        )
    elif recomputed["n_pred"] > 0:
        # 假阳阴性病例：掩膜已在上方完成全部校验，这里只额外统计预测团块
        entry["fp_components"] = _pred_component_stats(pred, vox_vol)
    # 病灶实例级指标：全部病例都算（阴性病例 reference_lesions=0，其 prediction components
    # 全部为 unmatched 假阳；不进入 sensitivity 分母）。异常同样由逐例异常边界收集。
    entry["lesion_instance_metrics"] = lesion_instance_metrics(pred, ref, vox_vol, cid)
    return entry


def run_full_mode(
    reports: dict[str, dict],
    cases_by_model: dict[str, dict],
    tolerance_mm: float,
    volume_thresholds_mm3: Sequence[float] | None,
    progress: bool,
    run: RunStats | None = None,
) -> None:
    """full 模式：**遍历并校验全部病例**（阳性 / 假阳阴性 / 真阴），再补充物理与表面指标。

    每一例都必须：读取 prediction 与 reference、校验几何（size / spacing / origin /
    direction）与 0/1 标签、由掩膜重算 TP/FP/FN/TN/n_pred/n_ref 并与 summary.json **逐项**
    核对。真阴病例（n_pred=0）同样不得跳过文件与几何检查。

    逐例异常收集范围覆盖该病例的**全部步骤**（读掩膜、计数核对、体积、``surface_metrics``、
    ``component_analysis`` / ``_pred_component_stats`` / ``lesion_instance_metrics``、entry 构造）：
    任一例失败只记录
    ``[model] case: 异常类型: 信息`` 并继续处理其余病例，全部病例跑完后再统一报错、不发布
    JSON。``KeyboardInterrupt`` / ``SystemExit`` 不属于 ``Exception``，不会被捕获。

    仅计数用的 ``run.ok`` / ``run.failed`` 在此更新；``run=None`` 时使用一次性统计，
    计数本身不影响上述 fail-closed 判定。
    """
    stats = run if run is not None else RunStats(mode="full", output=None)
    work = [
        (model_name, cid)
        for model_name, cases in cases_by_model.items()
        for cid in sorted(cases)
    ]
    iterator: Iterable = work
    if progress:
        from tqdm import tqdm

        # 总量 = 所有模型病例数之和，进度条覆盖逐病例主循环
        iterator = tqdm(
            work,
            desc="full-mode",
            total=len(work),
            unit="case",
            mininterval=1.0,
            file=sys.stdout,
        )

    errors: list[str] = []
    full_by_model: dict[str, dict[str, dict]] = {name: {} for name in cases_by_model}

    # 只有真正进入逐例循环之前才切到 unit=model-case，并重置逐例计数
    stats.begin_full_cases(total_pairs=len(work))

    for model_name, cid in iterator:
        case = cases_by_model[model_name][cid]
        try:
            entry = _build_full_case_entry(cid, case, tolerance_mm)
        except EvaluationError as exc:
            errors.append(f"[{model_name}] {cid}: {exc}")
            stats.full_case_failed()
            continue
        except Exception as exc:
            # 逐例失败收集（含 MedPy / SciPy / 体积 / 连通域等第三方异常）。
            # KeyboardInterrupt 与 SystemExit 派生自 BaseException 而非 Exception，不会被吞掉。
            errors.append(f"[{model_name}] {cid}: {type(exc).__name__}: {exc}")
            stats.full_case_failed()
            continue
        full_by_model[model_name][cid] = entry
        stats.full_case_ok()

    # 循环覆盖了全部 work，skipped 是已知量（在抛错之前结算，失败路径也有真实计数）
    stats.end_full_cases()

    if errors:
        raise EvaluationError(
            "full 模式未通过逐例校验，不发布结果：\n"
            + _format_error_report("逐例校验失败", errors)
        )

    for model_name, report in reports.items():
        cases = cases_by_model[model_name]
        full = full_by_model[model_name]
        pos_ids = [c["case_id"] for c in _positive(cases)]

        def get(cid: str, key: str, table: dict = full) -> Any:
            """本模型内按 case id 取 full 模式结果（避免重复下标链）。"""
            return table[cid][key]

        ref_volumes = [get(c, "reference_volume_mm3") for c in pos_ids]
        hd95s = [full[c]["surface"]["hd95_mm"] for c in pos_ids]
        nsds = [full[c]["surface"]["nsd"] for c in pos_ids]

        report["surface_metrics"] = {
            "nsd_tolerance_mm": tolerance_mm,
            "nsd_method": (
                "surfel-area weighted physical-surface NSD "
                "(surface-distance 0.1; spacing in mm)"
            ),
            "hd95_method": "medpy.metric.binary.hd95（尊重各轴 spacing，单位 mm）",
            "assd_method": "medpy.metric.binary.assd（尊重各轴 spacing，单位 mm）",
            "n_positive_cases": len(pos_ids),
            "n_cases_with_defined_hd95": sum(1 for v in hd95s if v is not None),
            "hd95_mm": _dist_stats(hd95s),
            "assd_mm": _dist_stats([full[c]["surface"]["assd_mm"] for c in pos_ids]),
            "nsd": _dist_stats(nsds),
            "missed_case_handling": (
                "GT 非空、预测为空：NSD = 0，HD95/ASSD = null，同时计入 missed case"
            ),
        }
        report["volume_metrics_mm3"] = {
            "voxel_volume_mm3_note": "由各例图像 spacing 算出；GT 与 prediction 几何必须一致",
            "reference_volume_mm3": _dist_stats(ref_volumes),
            "rve": {
                "definition": "RVE = (Vpred - Vref) / Vref，仅 GT 阳性病例",
                "median": _finite(np.median([get(c, "rve") for c in pos_ids]))
                if pos_ids
                else None,
                "q1": _finite(np.percentile([get(c, "rve") for c in pos_ids], 25))
                if pos_ids
                else None,
                "q3": _finite(np.percentile([get(c, "rve") for c in pos_ids], 75))
                if pos_ids
                else None,
            },
            "arve": {
                "definition": "ARVE = |Vpred - Vref| / Vref，仅 GT 阳性病例",
                "median": _finite(np.median([get(c, "arve") for c in pos_ids]))
                if pos_ids
                else None,
                "q1": _finite(np.percentile([get(c, "arve") for c in pos_ids], 25))
                if pos_ids
                else None,
                "q3": _finite(np.percentile([get(c, "arve") for c in pos_ids], 75))
                if pos_ids
                else None,
            },
        }

        if volume_thresholds_mm3:
            bounds = [0.0, *[float(t) for t in volume_thresholds_mm3], float("inf")]
            strata = []
            for lo, hi in itertools.pairwise(bounds):
                members = [
                    c for c in pos_ids if lo <= get(c, "reference_volume_mm3") < hi
                ]
                if not members:
                    strata.append(
                        {
                            "volume_min_mm3": lo,
                            "volume_max_mm3": hi,
                            "n": 0,
                            "positive_macro_dice": None,
                            "miss_rate": None,
                            "positive_voxel_recall": None,
                            "positive_voxel_precision": None,
                            "nsd_median": None,
                        }
                    )
                    continue
                tp = sum(cases[c]["TP"] for c in members)
                fp = sum(cases[c]["FP"] for c in members)
                fn = sum(cases[c]["FN"] for c in members)
                stratum_nsds = [
                    full[c]["surface"]["nsd"]
                    for c in members
                    if full[c]["surface"]["nsd"] is not None
                ]
                strata.append(
                    {
                        "volume_min_mm3": lo,
                        "volume_max_mm3": hi,
                        "n": len(members),
                        "positive_macro_dice": _finite(
                            np.mean([float(cases[c]["dice"] or 0.0) for c in members])
                        ),
                        "miss_rate": _ratio(
                            sum(1 for c in members if cases[c]["TP"] == 0), len(members)
                        ),
                        "positive_voxel_recall": _ratio(tp, tp + fn),
                        "positive_voxel_precision": _ratio(tp, tp + fp),
                        "nsd_median": _finite(np.median(stratum_nsds))
                        if stratum_nsds
                        else None,
                    }
                )
            report["size_strata"] = _json_safe(
                {
                    "volume_thresholds_mm3": [float(t) for t in volume_thresholds_mm3],
                    "stratified_by": "reference physical volume（reference_volume_mm3）",
                    "burden_note": (
                        "分层依据是整例总阳性体积 case_lesion_burden_mm3，不是单病灶大小；"
                        "单病灶大小需连通域级分析（见 component_analysis）"
                    ),
                    "strata": strata,
                }
            )

        # 阴性病例假阳团块：直接复用 full 模式已加载并校验过的掩膜，不重复读盘
        neg_fp = {
            cid: e["fp_components"]
            for cid, e in full.items()
            if not e["has_reference"] and "fp_components" in e
        }
        report["component_analysis"] = _json_safe(
            {
                "connectivity": CONNECTIVITY,
                "note": (
                    "仅分割失败分析；不计算 AUROC / average precision / FROC / "
                    "PI-CAI detection score"
                ),
                "n_positive_cases": len(pos_ids),
                "pred_component_count_total": sum(
                    full[c]["components"]["pred_component_count"] for c in pos_ids
                ),
                "pred_component_count_per_case": {
                    c: full[c]["components"]["pred_component_count"] for c in pos_ids
                },
                "pred_component_max_volume_mm3": _finite(
                    max(
                        full[c]["components"]["pred_component_max_volume_mm3"] or 0.0
                        for c in pos_ids
                    )
                )
                if pos_ids
                else None,
                "ref_component_count_total": sum(
                    full[c]["components"]["ref_component_count"] for c in pos_ids
                ),
                "ref_components_without_prediction_overlap_total": sum(
                    full[c]["components"]["ref_components_without_prediction_overlap"]
                    for c in pos_ids
                ),
                "per_case": [full[c]["components"] for c in pos_ids],
                "negative_case_fp_components": {
                    "connectivity": CONNECTIVITY,
                    "fp_component_count_total": sum(
                        v["component_count"] for v in neg_fp.values()
                    ),
                    "fp_component_count_per_case": {
                        cid: v["component_count"] for cid, v in neg_fp.items()
                    },
                    "fp_component_max_volume_mm3": _finite(
                        max(
                            (
                                v["component_max_volume_mm3"] or 0.0
                                for v in neg_fp.values()
                            ),
                            default=0.0,
                        )
                    ),
                    "n_negative_cases_with_prediction": len(neg_fp),
                    "source": "full 模式已加载并校验过的掩膜（不重复读盘）",
                },
            }
        )

        # 病灶实例级指标（仅 full 模式）：聚合逐病例结果，不重新匹配。
        # 这是与现有 component_analysis 独立的字段，避免改变既有字段语义。
        report["lesion_instance_metrics"] = _aggregate_lesion_instance_metrics(full)

        # 逐例掩膜校验的覆盖情况：full 模式必须覆盖**全部**病例（含真阴）
        report["mask_verification"] = {
            "n_cases_in_summary": len(cases),
            "n_cases_read_and_verified": len(full),
            "all_cases_verified": len(full) == len(cases),
            "per_case_checks": [
                "prediction 与 reference 文件存在且可读",
                "size / spacing / origin / direction 与 reference 一致（不重采样）",
                "标签仅允许 0/1，且不含非有限值",
                "由掩膜重算 TP/FP/FN/TN/n_pred/n_ref，并与 summary.json 逐项核对",
                "n_ref=0 的病例其 reference 掩膜必须实际为空",
            ],
            "truenegative_note": (
                "真阴病例（n_pred=0）同样执行完整文件与几何检查，不被跳过"
            ),
        }

        by_id = {row["case_id"]: row for row in report["per_case"]}
        for cid, entry in full.items():
            row = by_id.get(cid)
            if row is None:
                continue
            row["mask_verified"] = entry["mask_verified"]
            row["voxel_volume_mm3"] = entry["voxel_volume_mm3"]
            row["reference_volume_mm3"] = entry["reference_volume_mm3"]
            row["prediction_volume_mm3"] = entry["prediction_volume_mm3"]
            if "surface" in entry:
                row["rve"] = entry["rve"]
                row["arve"] = entry["arve"]
                row["hd95_mm"] = entry["surface"]["hd95_mm"]
                row["assd_mm"] = entry["surface"]["assd_mm"]
                row["nsd"] = entry["surface"]["nsd"]
                row["surface_status"] = entry["surface"]["reason"]
            if "fp_components" in entry:
                row["fp_component_count"] = entry["fp_components"]["component_count"]
                row["fp_component_max_volume_mm3"] = entry["fp_components"][
                    "component_max_volume_mm3"
                ]
            # 逐病例实例级结果（含可审计的 reference / prediction lesion records）
            row["lesion_instance_metrics"] = entry["lesion_instance_metrics"]


# --------------------------------------------------------------------------- CLI
def anatomy_region_metrics(reference, prediction):
    """Native region masks/counts, with native empty-both Dice=undefined rule."""
    from nnunetv2.evaluation.evaluate_predictions import (
        compute_tp_fp_fn_tn,
        region_or_label_to_mask,
    )

    from zonal_reliability_fusion.anatomy.contracts import (
        ANATOMY_LABELS,
        anatomy_validate_array,
    )

    anatomy_validate_array(reference, range(8), "anatomy reference")
    anatomy_validate_array(prediction, range(8), "anatomy prediction bitcode")
    if reference.shape != prediction.shape:
        raise EvaluationError("anatomy reference/prediction shape mismatch")
    metrics = {}
    for name, region in list(ANATOMY_LABELS.items())[1:]:
        ref = region_or_label_to_mask(reference, tuple(region))
        pred = region_or_label_to_mask(prediction, tuple(region))
        tp, fp, fn, tn = compute_tp_fp_fn_tn(ref, pred)
        denominator = 2 * int(tp) + int(fp) + int(fn)
        metrics[name] = {"Dice": 2 * int(tp) / denominator if denominator else None,
                         "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
                         "reference_voxels": int(ref.sum()), "prediction_voxels": int(pred.sum())}
    return metrics


def anatomy_disagreement(masks):
    wg, pz, tz = masks
    zone_union = pz | tz
    definitions = {"PZ_TZ_overlap_over_zone_union": (pz & tz, zone_union),
                   "PZ_outside_WG_over_PZ": (pz & ~wg, pz),
                   "TZ_outside_WG_over_TZ": (tz & ~wg, tz),
                   "zones_outside_WG_over_zone_union": (zone_union & ~wg, zone_union)}
    return {name: {"numerator_voxels": int(numerator.sum()), "denominator_voxels": int(denominator.sum()),
                   "ratio": _ratio(int(numerator.sum()), int(denominator.sum()))}
            for name, (numerator, denominator) in definitions.items()}


def build_anatomy_parser():
    parser = argparse.ArgumentParser(description="Independent WG/PZ/TZ probability evaluation on original T2W grid; pseudo-label agreement")
    for flag in ("prediction-dir", "reference-dir", "images-dir", "dataset-json", "plans", "split-file", "output-dir"):
        parser.add_argument(f"--{flag}", required=True)
    parser.add_argument("--native-ordered-export", action="store_true", help="Additional separate ordered integer export metrics")
    parser.add_argument("--no-progress", action="store_true")
    return parser


def run_anatomy_evaluation(args):
    import hashlib
    import importlib.util
    import time

    import SimpleITK as sitk
    from tqdm import tqdm

    from zonal_reliability_fusion.anatomy.contracts import (
        anatomy_encode_heads,
        anatomy_read_array,
        anatomy_same_grid,
        anatomy_validate_array,
        validate_anatomy_dataset,
        validate_anatomy_split,
    )
    started = time.monotonic()
    output = Path(args.output_dir)
    prediction, reference_dir, images = map(Path, (args.prediction_dir, args.reference_dir, args.images_dir))
    if output.exists() or output.is_symlink():
        raise EvaluationError(f"anatomy output exists; refuses overwrite: {output}")
    for protected in (prediction, reference_dir, images, Path(__file__).resolve().parents[1] / "data",
                      Path(__file__).resolve().parents[1] / "third_party"):
        if output.resolve() == protected.resolve() or protected.resolve() in output.resolve().parents:
            raise EvaluationError(f"anatomy evaluation output inside input/protected path: {output}")
    try:
        dataset = json.loads(Path(args.dataset_json).read_text())
        plans = json.loads(Path(args.plans).read_text())
        contract = validate_anatomy_dataset(dataset, plans["dataset_name"], "3d_fullres")
        _, case_ids = validate_anatomy_split(args.split_file, dataset, 0)
        expected = set(case_ids)
        references = {p.name[:-7] for p in reference_dir.glob("*.nii.gz")}
        if references != set(contract["case_ids"]):
            raise ValueError("reference case set differs from full raw anatomy contract")
        image_ids = {p.name[:-12] for p in images.glob("*_0000.nii.gz")}
        if image_ids != set(contract["case_ids"]):
            raise ValueError("T2W image case set differs from full raw anatomy contract")
        for suffix in (".npz", ".pkl", ".nii.gz"):
            actual = {p.name[:-len(suffix)] for p in prediction.glob(f"*{suffix}")}
            if actual != expected:
                raise ValueError(f"prediction case set mismatch {suffix}: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")
    except (ValueError, KeyError, OSError) as exc:
        raise EvaluationError(f"anatomy preflight: {exc}") from exc
    spec = importlib.util.spec_from_file_location("anatomy_native_probability_checks", Path(__file__).resolve().parent / "inference/predict_nnunet.py")
    checks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checks)
    entries, failures = [], []
    # No derived files are published until every expected study is evaluable.
    for cid in tqdm(case_ids, desc="anatomy-independent-evaluation", unit="study", disable=args.no_progress):
        try:
            reference = sitk.ReadImage(str(reference_dir / f"{cid}.nii.gz"))
            t2w_reader = sitk.ImageFileReader()
            t2w_reader.SetFileName(str(images / f"{cid}_0000.nii.gz"))
            t2w_reader.ReadImageInformation()
            anatomy_same_grid(t2w_reader, reference)
            ref_array = anatomy_validate_array(anatomy_read_array(reference_dir / f"{cid}.nii.gz"), range(8), "reference membership")
            probabilities, _, native = checks.load_anatomy_probability_case(prediction, cid, reference, plans["transpose_forward"])
            masks = probabilities > .5  # independent; no containment correction or exclusivity
            code = anatomy_encode_heads(masks)
            entry = {"case_id": cid, "independent_probability_threshold": anatomy_region_metrics(ref_array, code),
                     "prediction_disagreement": anatomy_disagreement(masks)}
            if args.native_ordered_export:
                entry["native_ordered_export"] = anatomy_region_metrics(ref_array, sitk.GetArrayFromImage(native))
            entries.append(entry)
        except Exception as exc:
            failures.append({"case_id": cid, "reason": f"{type(exc).__name__}: {exc}"})
            tqdm.write(f"[anatomy] {cid}: {failures[-1]['reason']}")
    print(f"[anatomy] success={len(entries)} failed={len(failures)} skipped=0 elapsed={time.monotonic()-started:.2f}s output={output}")
    if failures:
        raise EvaluationError(f"anatomy incomplete evaluation; no output published: {failures}")
    summary = {}
    for kind in ["independent_probability_threshold"] + (["native_ordered_export"] if args.native_ordered_export else []):
        summary[kind] = {}
        for region in ("WG", "PZ", "TZ"):
            values = [e[kind][region]["Dice"] for e in entries if e[kind][region]["Dice"] is not None]
            summary[kind][region] = {"mean_Dice": float(np.mean(values)) if values else None,
                                      "n_valid": len(values), "n_empty_both": len(entries)-len(values), "n_expected": len(case_ids)}
    pooled = {}
    for key in entries[0]["prediction_disagreement"]:
        numerator = sum(e["prediction_disagreement"][key]["numerator_voxels"] for e in entries)
        denominator = sum(e["prediction_disagreement"][key]["denominator_voxels"] for e in entries)
        pooled[key] = {"numerator_voxels": numerator, "denominator_voxels": denominator,
                       "pooled_voxel_ratio": _ratio(numerator, denominator)}
    report = {"schema": "picai_anatomy_independent_regions_v1", "threshold": ">0.5",
              "channel_order": ["WG", "PZ", "TZ"], "encoding": "WG+2*PZ+4*TZ",
              "interpretation": "Agreement with algorithm pseudo-labels, not human anatomical accuracy",
              "empty_rule": "both empty: Dice null, excluded from mean with n_empty_both reported; only one empty: Dice 0",
              "probability_grid": "native correct-shape restored original T2W array; pkl sitk_stuff supplies physical geometry; preprocessing crop outside filled with zero",
              "native_ordered_export_note": "TZ overwrites PZ overwrites WG; never the primary independent-head metric",
              "expected_case_ids": case_ids, "per_case": entries, "summary": summary,
              "prediction_disagreement": pooled, "success": len(entries), "failed": 0, "skipped": 0,
              "elapsed_seconds": time.monotonic()-started, "output_path": str(output),
              "metadata_sha256": {str(path): hashlib.sha256(Path(path).read_bytes()).hexdigest()
                                  for path in (args.dataset_json, args.plans, args.split_file)}}
    output.mkdir(parents=True, exist_ok=False)
    derived = output / "independent_threshold_bitcode"
    derived.mkdir()
    for cid in tqdm(case_ids, desc="anatomy-derived-export", unit="study", disable=args.no_progress):
        # Second pass limits memory to one case; copies geometry, never native predictions.
        ref = sitk.ReadImage(str(reference_dir / f"{cid}.nii.gz"))
        probabilities, _, _ = checks.load_anatomy_probability_case(prediction, cid, ref, plans["transpose_forward"])
        image = sitk.GetImageFromArray(anatomy_encode_heads(probabilities > .5))
        image.CopyInformation(ref)
        sitk.WriteImage(image, str(derived / f"{cid}.nii.gz"))
    _publish_json(output / "anatomy_regions.json", report)
    print(json.dumps(summary, ensure_ascii=False))
    print(f"[anatomy] published success={len(entries)} failed=0 skipped=0 elapsed={time.monotonic()-started:.2f}s output={output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate_segmentation.py",
        description=(
            "统一论文级病灶分割评估：离线读取已存在的 nnU-Net validation 产物，"
            "以完全相同的口径比较多个模型。不重写 validation/inference/prediction，"
            "不计算 AUROC / average precision / FROC / PI-CAI challenge score。"
        ),
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="NAME=FOLD_DIR",
        help="模型名与 fold 目录（可重复）。目录下必须存在 validation/summary.json；模型名必须唯一",
    )
    parser.add_argument(
        "--mode",
        choices=("summary", "full"),
        default="summary",
        help="summary（默认，只读 summary.json）/ full（额外读 NIfTI，需 --nsd-tolerance-mm）",
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=DEFAULT_BOOTSTRAP_RESAMPLES,
        help=f"bootstrap 重采样次数（正整数，默认 {DEFAULT_BOOTSTRAP_RESAMPLES}）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"bootstrap 随机种子，保证可复现（默认 {DEFAULT_SEED}）",
    )
    parser.add_argument(
        "--nsd-tolerance-mm",
        type=float,
        default=None,
        help="NSD 容差（mm）。full 模式必须显式提供；本工具不设置未经论证的默认值",
    )
    parser.add_argument(
        "--volume-thresholds-mm3",
        default=None,
        metavar="T1,T2,...",
        help="按参考物理体积（mm³）分层；不提供时不擅自创建小/中/大定义",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="可选输出 JSON 路径；不提供时只打印终端汇总；路径已存在时非零退出",
    )
    parser.add_argument(
        "--no-progress", action="store_true", help="关闭 full 模式的 tqdm 进度"
    )
    return parser


def parse_models(specs: Iterable[str]) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for spec in specs:
        if "=" not in spec:
            raise EvaluationError(f"--model 需要 NAME=FOLD_DIR 形式，收到 {spec!r}")
        name, _, path_str = spec.partition("=")
        name, path_str = name.strip(), path_str.strip()
        if not name or not path_str:
            raise EvaluationError(f"--model 的 name 与 path 都不能为空: {spec!r}")
        if name in seen:
            raise EvaluationError(f"--model 模型名重复: {name!r}")
        seen.add(name)
        out.append((name, Path(path_str)))
    if not out:
        raise EvaluationError("至少需要提供一个 --model")
    return out


def parse_thresholds(raw: str | None) -> list[float] | None:
    if raw is None or not raw.strip():
        return None
    values: list[float] = []
    for part in (p.strip() for p in raw.split(",")):
        if not part:
            raise EvaluationError("--volume-thresholds-mm3 含空项")
        try:
            value = float(part)
        except ValueError as exc:
            raise EvaluationError(
                f"--volume-thresholds-mm3 含非数值项: {part!r}"
            ) from exc
        if not math.isfinite(value) or value <= 0:
            raise EvaluationError(f"--volume-thresholds-mm3 必须为正的有限数: {part!r}")
        values.append(value)
    if len(set(values)) != len(values):
        raise EvaluationError(f"--volume-thresholds-mm3 含重复阈值: {values}")
    if values != sorted(values):
        raise EvaluationError(f"--volume-thresholds-mm3 必须单调递增: {values}")
    return values


def _report_failure(run: RunStats) -> None:
    """打印失败汇总；只在唯一一处调用，确保同一异常只产生一份汇总。"""
    print(run.render("failed"), file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI 入口：成功与失败两条路径都输出**恰好一份**结构化结束汇总。

    - ``EvaluationError`` 与任何其他 ``Exception`` 都打印 ``status=failed`` 的汇总，然后
      **原样重抛**（不吞掉异常，退出码仍交回 ``__main__`` 决定）；
    - 两个 ``except`` 分支互斥，因此同一异常不会打印两份汇总；
    - ``KeyboardInterrupt`` / ``SystemExit`` 派生自 ``BaseException`` 而非 ``Exception``，
      不会被捕获；
    - ``--help`` 由 argparse 在建立统计之前直接 ``SystemExit(0)``，因此不打印汇总
      （帮助不是一次运行）。
    """
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "anatomy":
        started = time.time()
        anatomy_args = build_anatomy_parser().parse_args(arguments[1:])
        try:
            return run_anatomy_evaluation(anatomy_args)
        except Exception as exc:
            print(f"[anatomy] status=failed unit=task success=0 failed=1 skipped=0 elapsed={time.time()-started:.2f}s output={anatomy_args.output_dir} reason={exc}")
            raise
    args = build_parser().parse_args(argv)
    run = RunStats(mode=args.mode, output=args.output)
    try:
        exit_code = _run(args, run)
    except EvaluationError:
        _report_failure(run)
        raise
    except Exception:
        _report_failure(run)
        raise
    run.mark_completed()
    print(run.render("succeeded"))
    return exit_code


def _run(args: argparse.Namespace, run: RunStats) -> int:
    """实际执行评估。统计只累加到 ``run``，不参与指标计算与 fail-closed 判定。"""
    if args.bootstrap_resamples <= 0:
        raise EvaluationError("--bootstrap-resamples 必须为正整数")
    models = parse_models(args.model)
    thresholds = parse_thresholds(args.volume_thresholds_mm3)

    # --nsd-tolerance-mm 必须是正的有限数（nan / inf / -inf 一律拒绝）
    if args.nsd_tolerance_mm is not None and (
        not math.isfinite(args.nsd_tolerance_mm) or args.nsd_tolerance_mm <= 0
    ):
        raise EvaluationError(
            f"--nsd-tolerance-mm 必须是正的有限数，实际为 {args.nsd_tolerance_mm!r}"
        )
    if args.mode == "full" and args.nsd_tolerance_mm is None:
        raise EvaluationError(
            "full 模式必须显式提供 --nsd-tolerance-mm（不设置默认值）"
        )
    if args.mode == "full":
        # NSD 依赖 DeepMind surface-distance（surfel-area weighted）；在读取任何 NIfTI 之前
        # 先做一次导入检查，缺包时立即 fail-closed，而不是逐例收集上千条导入错误。
        _surface_distance_metrics()
    if args.output is not None and Path(args.output).exists():
        raise EvaluationError(f"输出文件已存在，拒绝覆盖: {args.output}")

    # 先遍历全部模型再统一报告加载错误：不因第一个模型失败就提前退出
    # 进入加载阶段：unit=model，ok/failed 都表示模型数（与 mode 无关）
    run.begin_model_loading()
    reports: dict[str, dict] = {}
    cases_by_model: dict[str, dict] = {}
    load_errors: list[str] = []
    for name, fold_dir in models:
        try:
            if not fold_dir.is_dir():
                raise EvaluationError(f"fold 目录不存在: {fold_dir}")
            report, cases = summarize_model(
                name, fold_dir, args.bootstrap_resamples, args.seed
            )
        except EvaluationError as exc:
            message = str(exc)
            load_errors.append(
                message if message.startswith(f"[{name}]") else f"[{name}] {message}"
            )
            run.model_failed()
            continue
        except Exception as exc:
            load_errors.append(f"[{name}] {type(exc).__name__}: {exc}")
            run.model_failed()
            continue
        reports[name] = report
        cases_by_model[name] = cases
        run.model_loaded()
    if load_errors:
        raise EvaluationError(
            "模型加载失败，不发布结果：\n"
            + _format_error_report("加载失败", load_errors)
        )

    # 可比性检查阶段：沿用 model 单位，即便失败也不得改标成 model-case
    run.begin_comparability()
    _require_comparable(reports, cases_by_model)
    comparisons = paired_comparisons(
        reports, cases_by_model, args.bootstrap_resamples, args.seed
    )

    if args.mode == "full":
        # full 模式下 ok/failed 的含义切换为「model-case 对」，由 run_full_mode 覆盖
        run_full_mode(
            reports,
            cases_by_model,
            float(args.nsd_tolerance_mm),
            thresholds,
            progress=not args.no_progress,
            run=run,
        )

    # 结果汇总与落盘阶段；单位与计数沿用上一阶段
    run.begin_publishing()
    payload = _json_safe(
        {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": _now_utc(),
            "mode": args.mode,
            "seed": args.seed,
            "bootstrap_resamples": args.bootstrap_resamples,
            "nsd_tolerance_mm": (
                float(args.nsd_tolerance_mm)
                if args.nsd_tolerance_mm is not None
                else None
            ),
            "volume_thresholds_mm3": thresholds,
            "metrics_scope": (
                "lesion segmentation only; AUROC / average precision / FROC / "
                "PI-CAI challenge score are intentionally not computed"
            ),
            "models": reports,
            "paired_comparisons": comparisons,
        }
    )

    print_table(reports, comparisons, args.mode)

    if args.output is not None:
        _publish_json(Path(args.output), payload)
        print(f"[evaluate] 已写出: {args.output}")
    else:
        print("[evaluate] 未提供 --output，仅打印终端汇总")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except EvaluationError as exc:
        print(f"[evaluate][ERROR] {exc}", file=sys.stderr)
        sys.exit(2)
