#!/usr/bin/env python3
"""N0 / M0 训练结果评测（只读、CPU）：nnU-Net validation 输出 vs nnU-Net 格式 GT。

用法（训练结束后执行）：

```bash
cd /opt/data/private/lm/my-projects
conda activate lm
source scripts/env_nnunet.sh
python scripts/evaluate/evaluate_n0_validation.py
```

默认评测 `nnUNetTrainerPICAI_FLCE_NoFFT` 的 fold 0 validation（223 例），输出写入
`outputs/metrics/n0_validation_<UTC 时间戳>/`。

**诚实声明（务必与结果一起读）**：

1. 本脚本**不是** G0-SAP 冻结口径的评测。官方 PI-CAI 指标（lesion-level AP、PI-CAUC）需要
   `picai_eval` 与逐病灶标注；本环境**未安装** `picai_eval`，因此报告只给出**代理指标**并在
   `metrics.json` 中显式标注 `official_metrics_computed=false` 及原因；
2. 未导出概率图（`validation/<case>.npz`）时，**不计算**任何基于分数的指标（AP / AUC），
   且**禁止**用二值掩膜伪装成分数（fail-closed）；
3. 只用 nnU-Net 自带 `summary.json` 的 `nanmean` 不能作为论文数字；同理本脚本的 Dice 也不等于
   官方 `picai_eval` 口径；
4. 只读：不写回任何影像、不修改任何既有产物；CPU、不初始化 CUDA、不训练、不推理；
5. 输出目录与训练产物隔离（`outputs/metrics/`，不落在 `outputs/nnUNet_results/` 或 `workdir/`）。

**两个"检出"定义必须区分（本脚本显式分开输出）**：

- `lesion_hit`（= `detected`）：预测与 GT 病灶**至少 1 个体素重叠** —— 只在 GT 有病灶时有意义，
  用于"GT 阳性例检出率"；
- `predicted_positive`：预测**非空**（体积 > 0）—— 用于患者级混淆矩阵，因为 GT 阴性的病例
  不可能与病灶重叠，若用重叠定义则假阳性恒为 0、特异度虚高。

退出码：`0` = 全部期望病例都已评测；`1` = 有缺失（缺预测或缺 GT）或无可评测病例（报告仍会写出）。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from zonal_reliability_fusion.protocols import common as pc

DEFAULT_RESULTS_DIR = (
    "outputs/nnUNet_results/Dataset605_PICAI/"
    "nnUNetTrainerPICAI_FLCE_NoFFT__nnUNetPlans__3d_fullres/fold_0"
)
DEFAULT_SPLITS = "workdir/nnUNet_preprocessed/Dataset605_PICAI/splits_final.json"
DEFAULT_GT_DIR = "workdir/nnUNet_preprocessed/Dataset605_PICAI/gt_segmentations"
DEFAULT_LABELS = "data/metadata/picai_manifest.csv"
DEFAULT_OUT_ROOT = "outputs/metrics"

POSITIVE_LABELS = {"YES", "TRUE", "1", "POS", "POSITIVE"}
NEGATIVE_LABELS = {"NO", "FALSE", "0", "NEG", "NEGATIVE"}


# --------------------------------------------------------------------------- 输入
def read_splits_val_ids(splits_path: Path, fold: int) -> list[str]:
    """读取指定 fold 的 validation case_id（升序，确定性）。"""
    payload = json.loads(Path(splits_path).read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"splits 文件格式异常（应为非空 list）: {splits_path}")
    if fold < 0 or fold >= len(payload):
        raise ValueError(f"fold 越界：{fold}（splits 共 {len(payload)} 折）")
    val = payload[fold].get("val")
    if not isinstance(val, list) or not val:
        raise ValueError(f"fold {fold} 的 val 为空: {splits_path}")
    ids = sorted({str(c) for c in val})
    if len(ids) != len(val):
        raise ValueError(f"fold {fold} 的 val 存在重复 case_id（拒绝静默去重）: {splits_path}")
    return ids


def read_case_labels(labels_path: Path) -> dict[str, int]:
    """读取病例级 csPCa 标签（`YES`/`NO`）→ 1/0；无法判定的病例不进入患者级统计。"""
    out: dict[str, int] = {}
    with Path(labels_path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or ()
        if "case_id" not in header or "case_csPCa" not in header:
            raise ValueError(f"标签文件缺少 case_id / case_csPCa 列: {labels_path}")
        for row in reader:
            case_id = str(row["case_id"]).strip()
            raw = str(row.get("case_csPCa", "")).strip().upper()
            if not case_id:
                continue
            if raw in POSITIVE_LABELS:
                out[case_id] = 1
            elif raw in NEGATIVE_LABELS:
                out[case_id] = 0
    if not out:
        raise ValueError(f"标签文件未解析出任何可用标签: {labels_path}")
    return out


def read_mask(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    """读取掩膜为 bool 数组（轴序 z,y,x）+ 物理 spacing（x,y,z）。"""
    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image)
    spacing = tuple(float(v) for v in image.GetSpacing())
    if len(spacing) != 3:
        raise ValueError(f"期望 3D 图像，实际 spacing={spacing}: {path}")
    return np.asarray(array > 0.5), spacing  # type: ignore[return-value]


def read_probability_score(npz_path: Path, *, lesion_channels: slice = slice(1, None)) -> float | None:
    """从 nnU-Net 概率图 `.npz` 中取病灶通道的最大概率（用于代理 AP / AUC）。

    找不到可用数组或数值非有限时返回 `None`（调用方据此 fail-closed，不得伪造分数）。
    """
    try:
        with np.load(str(npz_path)) as handle:
            key = "probabilities" if "probabilities" in handle else (handle.files[0] if handle.files else None)
            if key is None:
                return None
            probs = np.asarray(handle[key])
    except Exception:  # noqa: BLE001 - 概率图可选；读取失败按"不可用"处理并如实记录
        return None
    if probs.ndim < 2 or probs.shape[0] < 2:
        return None
    lesion = probs[lesion_channels]
    if lesion.size == 0:
        return None
    value = float(np.max(lesion))
    return value if np.isfinite(value) else None


# --------------------------------------------------------------------------- 指标
def binary_counts(pred: np.ndarray, gt: np.ndarray) -> tuple[int, int, int]:
    tp = int(np.count_nonzero(pred & gt))
    fp = int(np.count_nonzero(pred & ~gt))
    fn = int(np.count_nonzero(~pred & gt))
    return tp, fp, fn


def dice_from_counts(tp: int, fp: int, fn: int) -> float:
    """Dice；两侧均为空时定义为 1.0（显式约定，避免 0/0）。"""
    denominator = 2 * tp + fp + fn
    return 1.0 if denominator == 0 else float(2 * tp / denominator)


def iou_from_counts(tp: int, fp: int, fn: int) -> float:
    union = tp + fp + fn
    return 1.0 if union == 0 else float(tp / union)


def mean_or_none(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def confusion_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """患者级混淆矩阵。

    - 预测阳性 = `predicted_positive`（预测掩膜非空）—— **不能**用"与 GT 重叠"，否则 GT 阴性病例
      永远算作阴性、假阳性恒为 0、特异度虚高；
    - 真值 = `case_csPCa`（病例级 csPCa 标注）。
    """
    usable = [r for r in rows if r.get("case_csPCa") in (0, 1) and r.get("evaluated")]
    if not usable:
        return {"available": False, "reason": "no_labeled_evaluated_cases", "n": 0}
    tp = sum(1 for r in usable if r["case_csPCa"] == 1 and r["predicted_positive"])
    fp = sum(1 for r in usable if r["case_csPCa"] == 0 and r["predicted_positive"])
    fn = sum(1 for r in usable if r["case_csPCa"] == 1 and not r["predicted_positive"])
    tn = sum(1 for r in usable if r["case_csPCa"] == 0 and not r["predicted_positive"])

    def ratio(num: int, den: int) -> float | None:
        return float(num / den) if den else None

    return {
        "available": True,
        "n": len(usable),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "sensitivity": ratio(tp, tp + fn),
        "specificity": ratio(tn, tn + fp),
        "ppv": ratio(tp, tp + fp),
        "npv": ratio(tn, tn + fn),
        "accuracy": ratio(tp + tn, len(usable)),
        "definition": "预测阳性 = 预测掩膜非空（体积 > 0）；真值 = case_csPCa",
        "note": "代理口径：官方 PI-CAI 用 lesion-level 检出（含最小病灶体积规则）计算，需 picai_eval",
    }


def score_based_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """基于概率分数的代理指标（AP / 患者级 ROC-AUC）。

    **代理指标**，不等于官方 lesion-level AP / PI-CAUC（官方口径需 `picai_eval` 与逐病灶标注）。
    """
    scored = [r for r in rows if r.get("case_csPCa") in (0, 1) and r.get("score") is not None]
    if not scored:
        return {
            "available": False,
            "reason": (
                "no_probability_scores（未导出概率图 `validation/<case>.npz`；"
                "禁止用二值掩膜伪装成分数）"
            ),
            "n": 0,
        }
    y_true = np.array([int(r["case_csPCa"]) for r in scored], dtype=np.int64)
    y_score = np.array([float(r["score"]) for r in scored], dtype=np.float64)
    if y_true.min() == y_true.max():
        return {"available": False, "reason": "single_class_labels", "n": len(scored)}
    from sklearn.metrics import average_precision_score, roc_auc_score

    return {
        "available": True,
        "n": len(scored),
        "average_precision": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "score_definition": "病灶通道（channel>=1）预测概率的最大值",
        "note": "**代理指标**：不是官方 lesion-level AP / PI-CAUC（官方口径需 picai_eval + 逐病灶标注）",
    }


# --------------------------------------------------------------------------- CLI
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="评测 nnU-Net validation 输出（只读、CPU；不训练、不推理、不写回影像）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR, help="fold 目录（含 validation/）")
    parser.add_argument("--splits", default=DEFAULT_SPLITS, help="splits_final.json（取 val 病例）")
    parser.add_argument("--fold", type=int, default=0, help="fold 编号（默认 0）")
    parser.add_argument("--gt-dir", default=DEFAULT_GT_DIR, help="nnU-Net 格式 GT 目录（gt_segmentations）")
    parser.add_argument("--labels", default=DEFAULT_LABELS, help="病例级标签 CSV（case_csPCa）")
    parser.add_argument("--no-labels", action="store_true", help="跳过患者级统计（不使用标签文件）")
    parser.add_argument("--out-dir", default="", help=f"输出目录（默认 {DEFAULT_OUT_ROOT}/<name>_<时间戳>）")
    parser.add_argument("--name", default="n0_validation", help="默认输出目录前缀")
    parser.add_argument("--limit", type=int, default=0, help="仅评测前 N 例（0 = 全部；调试用）")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    results_dir = pc.resolve_project_path(args.results_dir)
    splits_path = pc.resolve_project_path(args.splits)
    gt_dir = pc.resolve_project_path(args.gt_dir)
    labels_path = pc.resolve_project_path(args.labels)

    if not results_dir.is_dir():
        raise SystemExit(f"[eval] 找不到 fold 目录: {results_dir}")
    validation_dir = results_dir / "validation"
    if not validation_dir.is_dir():
        raise SystemExit(
            f"[eval] 找不到 validation 输出目录: {validation_dir}\n"
            "       训练结束会自动生成；若只想补跑验证并导出概率图：\n"
            "       python scripts/train/train_n0_picai_flce.py 605 3d_fullres 0 --validation-only "
            "--export-validation-probabilities"
        )
    for path, label in ((splits_path, "splits"), (gt_dir, "gt_segmentations")):
        if not path.exists():
            raise SystemExit(f"[eval] 找不到 {label}: {path}")

    val_ids = read_splits_val_ids(splits_path, int(args.fold))
    if args.limit > 0:
        val_ids = val_ids[: args.limit]
    labels = {} if args.no_labels else read_case_labels(labels_path)

    rows: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []
    for case_id in val_ids:
        pred_path = validation_dir / f"{case_id}.nii.gz"
        gt_path = gt_dir / f"{case_id}.nii.gz"
        npz_path = validation_dir / f"{case_id}.npz"
        record: dict[str, Any] = {
            "case_id": case_id,
            "has_prediction": pred_path.is_file(),
            "has_ground_truth": gt_path.is_file(),
            "case_csPCa": labels.get(case_id),
            "score": None,
            "evaluated": False,
        }
        if not pred_path.is_file() or not gt_path.is_file():
            missing.append(
                {
                    "case_id": case_id,
                    "missing_prediction": str(not pred_path.is_file()),
                    "missing_ground_truth": str(not gt_path.is_file()),
                }
            )
            rows.append(record)
            continue
        pred, spacing = read_mask(pred_path)
        gt, gt_spacing = read_mask(gt_path)
        if pred.shape != gt.shape:
            raise SystemExit(
                f"[eval] 预测与 GT 尺寸不一致（拒绝静默重采样）: {case_id} pred={pred.shape} gt={gt.shape}"
            )
        if any(abs(a - b) > 1e-6 for a, b in zip(spacing, gt_spacing)):
            raise SystemExit(f"[eval] 预测与 GT spacing 不一致: {case_id} pred={spacing} gt={gt_spacing}")
        tp, fp, fn = binary_counts(pred, gt)
        voxel_mm3 = float(spacing[0] * spacing[1] * spacing[2])
        pred_voxels = int(np.count_nonzero(pred))
        record.update(
            {
                "evaluated": True,
                "tp_voxels": tp,
                "fp_voxels": fp,
                "fn_voxels": fn,
                "dice": dice_from_counts(tp, fp, fn),
                "iou": iou_from_counts(tp, fp, fn),
                "detected": bool(tp > 0),
                "predicted_positive": bool(pred_voxels > 0),
                "gt_volume_mm3": float(np.count_nonzero(gt) * voxel_mm3),
                "pred_volume_mm3": float(pred_voxels * voxel_mm3),
                "voxel_volume_mm3": voxel_mm3,
                "score": read_probability_score(npz_path) if npz_path.is_file() else None,
            }
        )
        rows.append(record)

    evaluated = [r for r in rows if r["evaluated"]]
    gt_positive = [r for r in evaluated if r["gt_volume_mm3"] > 0]
    probabilities_present = any(r.get("score") is not None for r in evaluated)

    try:
        import picai_eval  # noqa: F401

        picai_eval_available = True
    except ImportError:
        picai_eval_available = False

    metrics = {
        "n_expected": len(val_ids),
        "n_evaluated": len(evaluated),
        "n_missing": len(missing),
        "complete": len(missing) == 0 and len(evaluated) == len(val_ids),
        "missing": missing,
        "dice": {
            "mean_all_evaluated": mean_or_none([r["dice"] for r in evaluated]),
            "mean_gt_positive": mean_or_none([r["dice"] for r in gt_positive]),
            "n_gt_positive": len(gt_positive),
            "definition": "Dice（二值掩膜，pred>0.5 vs GT>0.5）；两侧均为空记为 1.0",
        },
        "detection": {
            "n_detected": sum(1 for r in evaluated if r["detected"]),
            "detection_rate_gt_positive": (
                float(sum(1 for r in gt_positive if r["detected"]) / len(gt_positive)) if gt_positive else None
            ),
            "n_predicted_positive": sum(1 for r in evaluated if r["predicted_positive"]),
            "empty_prediction_cases": sorted(r["case_id"] for r in evaluated if not r["predicted_positive"]),
            "definition": "detected = 预测与 GT 病灶至少 1 个体素重叠（仅对 GT 阳性例有意义）",
        },
        "patient_level": confusion_metrics(rows),
        "score_based": score_based_metrics(rows),
        "probability_maps_available": probabilities_present,
        "picai_eval_available": picai_eval_available,
        "official_metrics_computed": False,
        "official_metrics_reason": (
            "未安装 picai_eval" if not picai_eval_available else "未按 G0-SAP 冻结口径接入"
        ),
        "scope_note": (
            "本报告为**自动评测的代理指标**；官方 PI-CAI 指标（lesion-level AP / PI-CAUC）需 picai_eval "
            "与逐病灶标注，且评测口径须按 G0-SAP 先冻结。"
        ),
    }

    out_dir = (
        pc.resolve_project_path(args.out_dir)
        if args.out_dir
        else pc.resolve_project_path(DEFAULT_OUT_ROOT) / f"{args.name}_{pc.timestamp_slug()}"
    )
    out_dir = pc.ensure_output_dir(out_dir)
    fieldnames = [
        "case_id",
        "evaluated",
        "has_prediction",
        "has_ground_truth",
        "case_csPCa",
        "detected",
        "predicted_positive",
        "dice",
        "iou",
        "tp_voxels",
        "fp_voxels",
        "fn_voxels",
        "gt_volume_mm3",
        "pred_volume_mm3",
        "score",
    ]
    pc.write_rows_csv(out_dir / "per_case.csv", rows, fieldnames=fieldnames)
    metadata = pc.build_run_metadata(
        tool="scripts/evaluate/evaluate_n0_validation.py",
        extra={
            "results_dir": str(results_dir),
            "validation_dir": str(validation_dir),
            "splits": str(splits_path),
            "fold": int(args.fold),
            "gt_dir": str(gt_dir),
            "labels": None if args.no_labels else str(labels_path),
            "checkpoint_present": (results_dir / "checkpoint_final.pth").is_file(),
            "case_id_source": "splits_final.json",
            "data_written_back": False,
        },
    )
    pc.write_json(out_dir / "metrics.json", {"run_metadata": metadata, "metrics": metrics})
    (out_dir / "summary.md").write_text(render_summary(metrics, results_dir), encoding="utf-8")

    print(f"[eval] 输出目录: {out_dir}")
    print(f"[eval] 期望 {metrics['n_expected']} 例 / 已评测 {metrics['n_evaluated']} 例 / 缺失 {metrics['n_missing']} 例")
    print(f"[eval] Dice（有病灶例）= {metrics['dice']['mean_gt_positive']}")
    print(f"[eval] 病灶检出率（GT 阳性例）= {metrics['detection']['detection_rate_gt_positive']}")
    patient = metrics["patient_level"]
    if patient.get("available"):
        print(
            f"[eval] 患者级: n={patient['n']} TP={patient['tp']} FP={patient['fp']} "
            f"FN={patient['fn']} TN={patient['tn']} sens={patient['sensitivity']} "
            f"spec={patient['specificity']} PPV={patient['ppv']}"
        )
    print(
        f"[eval] 基于分数的代理指标可用: {metrics['score_based'].get('available')}"
        f"（{metrics['score_based'].get('reason', 'ok')}）"
    )
    print(f"[eval] 官方口径指标: 未计算（{metrics['official_metrics_reason']}）")
    if not metrics["complete"]:
        print("[eval][WARN] 存在缺失病例，结果不完整（详见 metrics.json 的 missing 列表）")
        return 1
    return 0


def render_summary(metrics: dict[str, Any], results_dir: Path) -> str:
    """人工可读报告（纯数值；不含影像）。"""
    patient = metrics["patient_level"]
    score = metrics["score_based"]
    lines = [
        "# N0 validation 评测报告（自动、代理指标）\n\n",
        f"- 评测对象：`{results_dir}`\n",
        f"- 期望病例：{metrics['n_expected']}；已评测：{metrics['n_evaluated']}；缺失：{metrics['n_missing']}\n",
        f"- 完整性：**{metrics['complete']}**\n",
        (
            f"- 概率图可用：{metrics['probability_maps_available']}；"
            f"`picai_eval` 可用：{metrics['picai_eval_available']}\n\n"
        ),
        "## 1. Dice（二值掩膜）\n\n",
        f"- 全部已评测病例均值：{metrics['dice']['mean_all_evaluated']}\n",
        f"- 有病灶病例（n={metrics['dice']['n_gt_positive']}）均值：{metrics['dice']['mean_gt_positive']}\n\n",
        "## 2. 检出（两个定义必须区分）\n\n",
        f"- `detected`（与 GT 病灶有重叠）：{metrics['detection']['n_detected']} 例\n",
        f"- GT 阳性例检出率：{metrics['detection']['detection_rate_gt_positive']}\n",
        f"- `predicted_positive`（预测非空）：{metrics['detection']['n_predicted_positive']} 例\n",
        f"- 预测为空的病例数：{len(metrics['detection']['empty_prediction_cases'])}\n\n",
        "## 3. 患者级（预测阳性 = 预测非空；真值 = case_csPCa）\n\n",
    ]
    if patient.get("available"):
        lines.append(
            f"- n={patient['n']} TP={patient['tp']} FP={patient['fp']} FN={patient['fn']} TN={patient['tn']}\n"
        )
        lines.append(
            f"- 敏感度={patient['sensitivity']} 特异度={patient['specificity']} "
            f"PPV={patient['ppv']} NPV={patient['npv']} 准确率={patient['accuracy']}\n\n"
        )
    else:
        lines.append(f"- 不可用：{patient.get('reason')}\n\n")
    lines.append("## 4. 基于分数的代理指标\n\n")
    if score.get("available"):
        lines.append(
            f"- 平均精度 AP={score['average_precision']:.6f}｜"
            f"患者级 ROC-AUC={score['roc_auc']:.6f}（n={score['n']}）\n"
        )
    else:
        lines.append(f"- 不可用：{score.get('reason')}\n")
    lines += [
        "\n## 5. 不能作为论文指标的原因\n\n",
        f"- {metrics['scope_note']}\n",
        f"- 官方口径未计算原因：{metrics['official_metrics_reason']}\n",
        "\n## 6. 缺失病例\n\n",
    ]
    if metrics["missing"]:
        for item in metrics["missing"]:
            lines.append(
                f"- `{item['case_id']}`：缺预测={item['missing_prediction']} "
                f"缺GT={item['missing_ground_truth']}\n"
            )
    else:
        lines.append("- 无\n")
    return "".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
