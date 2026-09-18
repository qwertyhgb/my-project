#!/usr/bin/env python
"""N0 官方基线「Pseudo dice 长期为 0」只读诊断工具（PI-CAI Dataset605 / fold 0）。

回答三个问题
------------
1. **数据与标签**：预处理 `.b2nd`/`.pkl` 的标签值、阳性例前景体素、`class_locations` 是否完好；
2. **采样**：用**官方 nnUNetDataLoader**（与训练完全同参数：batch=2、oversample=0.33、plan patch）
   实际采样若干 batch，统计每个 patch 的前景体素、哪些 patch 是 FG 强制采样、是否真的命中前景；
3. **增强**：用**官方** `nnUNetTrainer.get_training_transforms` 对同一批 patch 做增强，
   统计增强后前景是否仍然存在（含前后对比的聚合估计）。

安全性（严格遵守 AGENTS.md §1–§4）
----------------------------------
- **只读**：只以只读方式打开 `.b2nd`（blosc2 + mmap）与 `.pkl`；不写回、不改名、不删除任何数据；
- **不干扰训练**：不实例化 `nnUNetTrainer`，不创建/触碰 `nnUNet_results/` 下的任何文件；
- **输出隔离**：所有产物写入 `outputs/diagnostics/n0_zero_dice/<timestamp>/`；
- **进度**：默认显示 tqdm 进度条（病例级 + patch 级），`--no-progress` 可关闭；结束时打印成功/失败/跳过计数、
  耗时与输出路径。

用法（必须由研究者本人执行）
----------------------------
    conda activate lm
    cd /opt/data/private/lm/my-projects
    source scripts/env_nnunet.sh
    python scripts/train/diagnose_n0_zero_dice.py --split val --max-batches 20
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:  # 允许未设置 PYTHONPATH 时也能 import 项目包
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from zonal_reliability_fusion.integrations import nnunet_dataloader_probe as probe  # noqa: E402
from zonal_reliability_fusion.utils.progress import make_progress, resolve_progress  # noqa: E402

# --------------------------------------------------------------------------- 常量

DEFAULT_DATASET_NAME = "Dataset605_PICAI"
DEFAULT_TRAINER_DIR = "nnUNetTrainer__nnUNetPlans__3d_fullres"
EXPECTED_CHANNEL_NAMES = {"0000": "T2W", "0001": "ADC", "0002": "HBV"}
EXPECTED_LABELS = {"background": 0, "lesion": 1}
EXPECTED_OVERALL_SPLIT = {"train": 1277, "val": 223}

CASE_FIELDS = [
    "case_id", "split", "expected_positive",
    "data_exists", "seg_exists", "props_exists",
    "data_shape", "data_dtype", "seg_shape", "seg_labels", "labels_binary",
    "n_fg_voxels", "fg_ratio", "fg_bbox_z", "fg_bbox_y", "fg_bbox_x",
    "spacing", "shape_before_cropping", "bbox_used_for_cropping",
    "class_locations_present", "class_locations_n_samples", "class_locations_in_bounds",
    "class_locations_issues", "n_issues", "issues",
]

BATCH_FIELDS = [
    "split", "phase", "batch_idx", "patch_idx", "case_id", "case_expected_positive",
    "forced_foreground", "fg_voxels", "fg_ratio", "labels", "has_foreground",
]


# --------------------------------------------------------------------------- 工具


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _write_csv(rows: Sequence[Mapping], path: Path, fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v)
                             for k, v in row.items() if k in fields})


def _load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- 静态审计（纯元数据）


def static_audit(args, *, expected_counts: Mapping[str, int] | None = None) -> dict:
    """dataset.json / plans / splits_final / 划分 CSV 的一致性审计（只读 JSON/CSV，不读体素）。

    `expected_counts` 默认使用 PI-CAI 冻结划分（train=1277 / val=223）；合成单元测试可覆盖。
    """
    expected_counts = dict(expected_counts or EXPECTED_OVERALL_SPLIT)
    checks: list[dict] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "pass": bool(ok), "detail": detail})

    # --- dataset.json ---
    dataset_json = {}
    try:
        dataset_json = _load_json(args.dataset_json)
        ch = {str(k): str(v) for k, v in (dataset_json.get("channel_names") or {}).items()}
        check("dataset.json 通道顺序为 0000=T2W/0001=ADC/0002=HBV", ch == EXPECTED_CHANNEL_NAMES, f"{ch}")
        labels = {str(k): int(v) for k, v in (dataset_json.get("labels") or {}).items()}
        check("dataset.json 标签为 background=0 / lesion=1", labels == EXPECTED_LABELS, f"{labels}")
        check("dataset.json 无 ignore 标签（不会把前景当忽略）", "ignore" not in labels, f"keys={list(labels)}")
        check("dataset.json file_ending=.nii.gz", dataset_json.get("file_ending") == ".nii.gz",
              f"{dataset_json.get('file_ending')}")
    except Exception as exc:  # pragma: no cover - 环境相关
        check("dataset.json 可读", False, f"{type(exc).__name__}: {exc}")

    # --- plans ---
    plans_summary: dict = {}
    try:
        plans = _load_json(args.plans)
        configs = list((plans.get("configurations") or {}).keys())
        check(f"plans 含 configuration={args.configuration}", args.configuration in configs, f"{configs}")
        cfg = (plans.get("configurations") or {}).get(args.configuration, {})
        plans_summary = {
            "patch_size": cfg.get("patch_size"),
            "spacing": cfg.get("spacing"),
            "batch_size": cfg.get("batch_size"),
            "data_identifier": cfg.get("data_identifier"),
            "batch_dice": cfg.get("batch_dice"),
            "use_mask_for_norm": cfg.get("use_mask_for_norm"),
            "normalization_schemes": cfg.get("normalization_schemes"),
            "n_stages": (cfg.get("architecture") or {}).get("arch_kwargs", {}).get("n_stages"),
        }
        check("plans patch_size 元素均 >0",
              bool(plans_summary["patch_size"]) and all(int(v) > 0 for v in plans_summary["patch_size"]),
              f"{plans_summary['patch_size']}")
        check("plans batch_size > 0", int(plans_summary["batch_size"] or 0) > 0, f"{plans_summary['batch_size']}")
    except Exception as exc:  # pragma: no cover
        check("plans 可读", False, f"{type(exc).__name__}: {exc}")

    # --- splits_final ---
    splits = []
    tr, va = [], []
    try:
        splits = _load_json(args.splits)
        fold = splits[int(args.fold)]
        tr, va = list(fold["train"]), list(fold["val"])
        counts = {"train": len(tr), "val": len(va)}
        check("splits_final fold 数与声明一致", len(splits) == 1, f"n_folds={len(splits)}")
        check("train/val 无重复", len(set(tr)) == len(tr) and len(set(va)) == len(va),
              f"train={len(set(tr))}/{len(tr)} val={len(set(va))}/{len(va)}")
        check("train/val 无交集", not (set(tr) & set(va)), f"overlap={len(set(tr) & set(va))}")
        check(f"train={expected_counts['train']} / val={expected_counts['val']}",
              counts == expected_counts, f"{counts}")
        check("train ∪ val 覆盖全部 study", len(set(tr) | set(va)) == sum(expected_counts.values()),
              f"{len(set(tr) | set(va))}")
    except Exception as exc:  # pragma: no cover
        check("splits_final 可读", False, f"{type(exc).__name__}: {exc}")

    # --- 划分 CSV 与 manifest 元数据链路 ---
    split_csv_rows: list[dict] = []
    try:
        with open(args.split_csv, encoding="utf-8") as f:
            split_csv_rows = list(csv.DictReader(f))
        bad_fmt = [r["case_id"] for r in split_csv_rows
                   if r["case_id"] != f"{r['patient_id']}_{r['study_id']}"]
        check("CSV: case_id == patient_id_study_id", not bad_fmt, f"不符 {len(bad_fmt)} 例")
        with open(args.manifest, encoding="utf-8") as f:
            manifest = list(csv.DictReader(f))
        pair2case = {(r["patient_id"], r["study_id"]): r["case_id"] for r in manifest}
        pair2pos = {(r["patient_id"], r["study_id"]): r["case_csPCa"] for r in manifest}
        bad_map = [r["case_id"] for r in split_csv_rows
                   if pair2case.get((r["patient_id"], r["study_id"])) != r["case_id"]]
        check("CSV ↔ manifest: (patient,study)→case_id 一致", not bad_map, f"不一致 {len(bad_map)} 例")
        bad_pos = [r["case_id"] for r in split_csv_rows
                   if pair2pos.get((r["patient_id"], r["study_id"])) != r["case_csPCa"]]
        check("CSV ↔ manifest: csPCa 标签一致", not bad_pos, f"不一致 {len(bad_pos)} 例")
        if tr:
            by_split = {s: {r["case_id"] for r in split_csv_rows if r["split"] == s}
                        for s in ("train", "validation")}
            check("CSV train 集合 == splits_final train 集合", by_split.get("train") == set(tr),
                  f"CSV={len(by_split.get('train') or [])} splits={len(set(tr))}")
            check("CSV validation 集合 == splits_final val 集合", by_split.get("validation") == set(va),
                  f"CSV={len(by_split.get('validation') or [])} splits={len(set(va))}")
    except Exception as exc:  # pragma: no cover
        check("划分 CSV / manifest 可读", False, f"{type(exc).__name__}: {exc}")

    positives = {
        "train": sum(1 for r in split_csv_rows if r.get("split") == "train" and r.get("case_csPCa") == "YES"),
        "validation": sum(1 for r in split_csv_rows if r.get("split") == "validation" and r.get("case_csPCa") == "YES"),
    }
    failed = [c["check"] for c in checks if not c["pass"]]
    return {
        "checks": checks,
        "n_checks": len(checks),
        "n_failed": len(failed),
        "failed_checks": failed,
        "ok": not failed,
        "positive_cases_by_split": positives,
        "plans_summary": plans_summary,
        "dataset_json": dataset_json,
    }


def training_log_audit(results_dir: Path) -> dict:
    """解析当前/最近一次训练的 training_log，提取 pseudo dice / loss / epoch 耗时（只读文本）。"""
    logs = sorted(Path(results_dir).glob("training_log_*.txt"), key=lambda p: p.stat().st_mtime)
    if not logs:
        return {"log_file": None, "found": False}
    log_file = logs[-1]
    text = log_file.read_text(encoding="utf-8", errors="replace")

    pseudo = [float(m) for m in re.findall(r"Pseudo dice \[np\.float32\(([-0-9.eE+]+)\)\]", text)]
    train_loss = [float(m) for m in re.findall(r"train_loss ([-0-9.eE+]+)", text)]
    val_loss = [float(m) for m in re.findall(r"val_loss ([-0-9.eE+]+)", text)]
    epoch_time = [float(m) for m in re.findall(r"Epoch time: ([-0-9.eE+]+) s", text)]
    nonzero = [i for i, v in enumerate(pseudo) if v > 0]
    return {
        "found": True,
        "log_file": str(log_file),
        "n_epochs_logged": len(pseudo),
        "pseudo_dice": pseudo,
        "pseudo_dice_last": pseudo[-1] if pseudo else None,
        "n_epochs_pseudo_dice_nonzero": len(nonzero),
        "first_nonzero_pseudo_dice_epoch": nonzero[0] if nonzero else None,
        "train_loss_first": train_loss[0] if train_loss else None,
        "train_loss_last": train_loss[-1] if train_loss else None,
        "train_loss_min": min(train_loss) if train_loss else None,
        "train_loss_max": max(train_loss) if train_loss else None,
        "val_loss_last": val_loss[-1] if val_loss else None,
        "epoch_time_last_sec": epoch_time[-1] if epoch_time else None,
        "n_yayy_new_best_events": text.count("Yayy! New best EMA pseudo Dice"),
    }


# --------------------------------------------------------------------------- 病例审计


def audit_cases(
    *,
    preprocessed_root: Path,
    case_ids: Sequence[str],
    split_name: str,
    expected_positive: Mapping[str, bool | None],
    data_folder_name: str,
    progress: bool,
    check_data_integrity: bool,
) -> list[dict]:
    """逐例只读审计（一次一例，不把全量数据读进内存）。"""
    rows: list[dict] = []
    dataset = probe.open_official_dataset(preprocessed_root, case_ids, data_folder_name=data_folder_name)
    bar = make_progress(case_ids, total=len(case_ids), desc=f"病例审计[{split_name}]", disable=not progress)
    for case_id in bar:
        row = {k: None for k in CASE_FIELDS}
        row.update({"case_id": case_id, "split": split_name,
                    "expected_positive": expected_positive.get(case_id)})
        try:
            data, seg, props = probe.load_case_light(
                dataset, case_id, read_data=check_data_integrity, read_seg=True
            )
            # load_case 成功即意味着 data.<b2nd> 已被 blosc2 以只读方式打开（文件存在且可解析）；
            # 只有 --check-data-integrity 时才会真正物化 data 体素做数值检查。
            row["data_exists"] = True
            row["seg_exists"] = seg is not None
            row["props_exists"] = bool(props)
            seg_stats = probe.patch_foreground_stats(seg, case_id=case_id)
            row.update({
                "seg_shape": list(np.shape(seg)),
                "seg_labels": seg_stats["labels"],
                "labels_binary": seg_stats["binary_ok"],
                "n_fg_voxels": seg_stats["fg_voxels"],
                "fg_ratio": round(seg_stats["fg_ratio"], 8),
            })
            bbox = seg_stats["bbox"]
            if bbox:
                row["fg_bbox_z"] = f"{bbox['z'][0]}-{bbox['z'][1]}"
                row["fg_bbox_y"] = f"{bbox['y'][0]}-{bbox['y'][1]}"
                row["fg_bbox_x"] = f"{bbox['x'][0]}-{bbox['x'][1]}"
            if check_data_integrity and data is not None:
                row["data_shape"] = list(np.shape(data))
                row["data_dtype"] = str(np.asarray(data).dtype)
                row["data_not_finite"] = bool(not np.isfinite(np.asarray(data, dtype=np.float32)).all())
                if row["data_shape"][-3:] != row["seg_shape"]:
                    row["data_seg_shape_mismatch"] = True
            row["spacing"] = list(props.get("spacing")) if props.get("spacing") is not None else None
            row["shape_before_cropping"] = (
                list(props["shape_before_cropping"]) if props.get("shape_before_cropping") is not None else None
            )
            row["bbox_used_for_cropping"] = (
                [list(b) for b in props["bbox_used_for_cropping"]]
                if props.get("bbox_used_for_cropping") is not None else None
            )
            spatial_shape = row["seg_shape"]
            cl_report = probe.class_locations_report(props.get("class_locations"), spatial_shape)
            row["class_locations_present"] = cl_report["present"]
            row["class_locations_n_samples"] = cl_report["n_samples_label"]
            row["class_locations_in_bounds"] = cl_report["in_bounds"]
            row["class_locations_issues"] = cl_report["issues"]
            if not row["labels_binary"]:
                pass  # evaluate_case_row 会记录
        except Exception as exc:  # 单例失败不终止整体审计
            row["seg_exists"] = False
            row.setdefault("issues", [])
            row["_error"] = f"{type(exc).__name__}: {exc}"
        issues = probe.evaluate_case_row(row)
        if row.get("_error"):
            issues.append(f"读取失败：{row['_error']}")
        row["issues"] = issues
        row["n_issues"] = len(issues)
        rows.append({k: row.get(k) for k in CASE_FIELDS})
    bar.close()
    return rows


# --------------------------------------------------------------------------- 采样 / 增强探针


def probe_batches(
    *,
    loader,
    split_name: str,
    phase: str,
    max_batches: int,
    batch_size: int,
    oversample_foreground_percent: float,
    expected_positive: Mapping[str, bool | None],
    progress: bool,
) -> list[dict]:
    """用官方 dataloader 采样 max_batches 个 batch，逐 patch 记录前景统计。"""
    rows: list[dict] = []
    flags = probe.foreground_oversample_flags(batch_size, oversample_foreground_percent)
    bar = make_progress(None, total=max_batches, desc=f"patch 采样[{split_name}/{phase}]",
                        disable=not progress, unit="batch")
    for b in range(max_batches):
        batch = loader.generate_train_batch()
        keys = list(batch["keys"])
        target = batch["target"]
        for j in range(batch_size):
            seg_j = target[j]
            if hasattr(seg_j, "detach"):
                seg_j = seg_j.detach().cpu().numpy()
            seg_j = np.asarray(seg_j)
            stats = probe.patch_foreground_stats(seg_j, case_id=str(keys[j]) if j < len(keys) else "")
            case_id = str(keys[j]) if j < len(keys) else ""
            rows.append({
                "split": split_name,
                "phase": phase,
                "batch_idx": b,
                "patch_idx": j,
                "case_id": case_id,
                "case_expected_positive": expected_positive.get(case_id),
                "forced_foreground": bool(flags[j]),
                "fg_voxels": stats["fg_voxels"],
                "fg_ratio": round(stats["fg_ratio"], 8),
                "labels": stats["labels"],
                "has_foreground": stats["fg_voxels"] > 0,
            })
        bar.update(1)
    bar.close()
    return rows


# --------------------------------------------------------------------------- 主流程


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="N0 官方基线 Pseudo dice 长期为 0 的只读诊断（不训练、不推理、不改数据）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    p.add_argument("--preprocessed-root", type=Path,
                   default=PROJECT_ROOT / "workdir/nnUNet_preprocessed" / DEFAULT_DATASET_NAME)
    p.add_argument("--data-folder-name", default="nnUNetPlans_3d_fullres")
    p.add_argument("--plans", type=Path, default=None, help="默认 <preprocessed-root>/nnUNetPlans.json")
    p.add_argument("--dataset-json", type=Path, default=None, help="默认 <preprocessed-root>/dataset.json")
    p.add_argument("--splits", type=Path, default=None, help="默认 <preprocessed-root>/splits_final.json")
    p.add_argument("--split-csv", type=Path,
                   default=PROJECT_ROOT / "data/splits/picai_nnunet_split_cases.csv")
    p.add_argument("--manifest", type=Path,
                   default=PROJECT_ROOT / "data/metadata/picai_manifest.csv")
    p.add_argument("--results-dir", type=Path,
                   default=PROJECT_ROOT / "outputs/nnUNet_results" / DEFAULT_DATASET_NAME / DEFAULT_TRAINER_DIR / "fold_0")
    p.add_argument("--configuration", default="3d_fullres")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--split", choices=["train", "val", "both"], default="both")
    p.add_argument("--max-cases", type=int, default=None, help="每个 split 最多审计多少例（默认全部）")
    p.add_argument("--max-batches", type=int, default=0, help="采样探针的 batch 数（0=跳过）")
    p.add_argument("--batch-size", type=int, default=None, help="默认取 plan 的 batch_size")
    p.add_argument("--oversample-foreground-percent", type=float, default=0.33)
    p.add_argument("--with-augmentation", action="store_true", help="额外跑一遍官方训练增强并统计前景存活")
    p.add_argument("--check-data-integrity", action="store_true",
                   help="额外物化 data 体素做 NaN/Inf 与 shape 检查（I/O 明显更重）")
    p.add_argument("--case-ids", nargs="*", default=None, help="只审计指定 case_id（调试用）")
    p.add_argument("--skip-static-audit", action="store_true")
    p.add_argument("--skip-case-audit", action="store_true")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--fail-on-anomaly", action="store_true", help="发现异常时以退出码 2 结束")
    args = p.parse_args(argv)

    args.plans = args.plans or (args.preprocessed_root / "nnUNetPlans.json")
    args.dataset_json = args.dataset_json or (args.preprocessed_root / "dataset.json")
    args.splits = args.splits or (args.preprocessed_root / "splits_final.json")
    if args.output is None:
        args.output = args.project_root / "outputs/diagnostics/n0_zero_dice" / _now_stamp()
    return args


def _case_split_map(args) -> dict[str, str]:
    """case_id → 'train'/'validation'（来自冻结划分 CSV 元数据）。"""
    with open(args.split_csv, encoding="utf-8") as f:
        return {r["case_id"]: r["split"] for r in csv.DictReader(f)}


def _expected_positive_map(args) -> dict[str, bool]:
    with open(args.split_csv, encoding="utf-8") as f:
        return {r["case_id"]: (r["case_csPCa"] == "YES") for r in csv.DictReader(f)}


def _selected_ids(args, splits_doc: list, split_name: str) -> list[str]:
    if args.case_ids:
        return [str(c) for c in args.case_ids]
    fold = splits_doc[int(args.fold)]
    ids = list(fold["train"] if split_name == "train" else fold["val"])
    ids = [str(i) for i in ids]
    if args.max_cases:
        ids = ids[: int(args.max_cases)]
    return ids


def main(argv: Sequence[str] | None = None) -> int:
    t0 = time.time()
    args = parse_args(argv)
    progress = resolve_progress(args.no_progress, True)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[诊断] 输出目录：{out_dir}")
    print(f"[诊断] 环境：python={sys.executable}")
    print(f"[诊断] 警告：本工具只读，不会修改任何数据或训练产物")

    plan_doc = {}
    try:
        plan_doc = _load_json(args.plans)
    except Exception:
        pass
    plan_cfg = (plan_doc.get("configurations") or {}).get(args.configuration, {})
    patch_size = plan_cfg.get("patch_size") or [16, 320, 320]
    batch_size = int(args.batch_size or plan_cfg.get("batch_size") or 2)

    summary: dict = {
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "python": sys.executable,
        "plan_patch_size": list(patch_size),
        "plan_batch_size": batch_size,
        "oversample_foreground_percent": args.oversample_foreground_percent,
        "fg_flags_per_batch": probe.foreground_oversample_flags(batch_size, args.oversample_foreground_percent),
        "outputs": {},
    }

    np.random.seed(int(args.seed))

    # 1) 静态审计
    static = None
    if not args.skip_static_audit:
        static = static_audit(args)
        _write_json(static, out_dir / "static_audit.json")
        summary["outputs"]["static_audit"] = str(out_dir / "static_audit.json")
        print(f"[静态审计] {static['n_checks'] - static['n_failed']}/{static['n_checks']} 通过"
              f"（失败 {static['n_failed']}）")

    # 2) 训练日志审计
    log_audit = training_log_audit(Path(args.results_dir))
    _write_json(log_audit, out_dir / "training_log_audit.json")
    summary["outputs"]["training_log_audit"] = str(out_dir / "training_log_audit.json")
    if log_audit.get("found"):
        print(f"[训练日志] {Path(log_audit['log_file']).name}：已记录 {log_audit['n_epochs_logged']} epoch，"
              f"pseudo dice 非零 epochs={log_audit['n_epochs_pseudo_dice_nonzero']}"
              f"（首次非零 epoch={log_audit['first_nonzero_pseudo_dice_epoch']}），"
              f"train_loss 末值={log_audit['train_loss_last']}")
    else:
        print("[训练日志] 未找到 training_log_*.txt")

    # 3) 病例审计
    case_rows: list[dict] = []
    case_summary = None
    if not args.skip_case_audit:
        splits_doc = _load_json(args.splits)
        expected_positive = _expected_positive_map(args)
        targets = {
            "train": _selected_ids(args, splits_doc, "train"),
            "val": _selected_ids(args, splits_doc, "val"),
        }
        wanted = ["train", "val"] if args.split == "both" else [args.split]
        for split_name in wanted:
            case_rows.extend(audit_cases(
                preprocessed_root=Path(args.preprocessed_root),
                case_ids=targets[split_name],
                split_name=split_name,
                expected_positive=expected_positive,
                data_folder_name=args.data_folder_name,
                progress=progress,
                check_data_integrity=args.check_data_integrity,
            ))
        case_summary = probe.aggregate_case_rows(case_rows)
        _write_csv(case_rows, out_dir / "case_audit.csv", CASE_FIELDS)
        _write_json(case_summary, out_dir / "case_audit_summary.json")
        summary["outputs"]["case_audit"] = str(out_dir / "case_audit.csv")
        for split_name, s in case_summary.items():
            print(f"[病例审计] {split_name}: {s['n_cases']} 例（阳性 {s['n_expected_positive']}）；"
                  f"阳性空 seg={s['n_positive_with_empty_seg']}；class_locations 异常="
                  f"{s['n_class_locations_problems']}；异常总数={s['n_anomalies']}")

    # 4) 采样 / 增强探针
    batch_rows: list[dict] = []
    batch_summary = None
    if args.max_batches > 0:
        splits_doc = _load_json(args.splits)
        expected_positive = _expected_positive_map(args)
        dataset_json = _load_json(args.dataset_json)
        wanted = ["train", "val"] if args.split == "both" else [args.split]
        augment_config: dict | None = None
        for split_name in wanted:
            ids = _selected_ids(args, splits_doc, split_name)
            loader_pre = probe.build_official_dataloader(
                str(args.preprocessed_root), ids,
                patch_size=patch_size, batch_size=batch_size, dataset_json=dataset_json,
                oversample_foreground_percent=args.oversample_foreground_percent, transforms=None,
                data_folder_name=args.data_folder_name,
            )
            batch_rows.extend(probe_batches(
                loader=loader_pre, split_name=split_name, phase="pre_aug", max_batches=args.max_batches,
                batch_size=batch_size, oversample_foreground_percent=args.oversample_foreground_percent,
                expected_positive=expected_positive, progress=progress,
            ))
            if args.with_augmentation:
                transforms, augment_config = probe.build_official_training_transforms(
                    patch_size, use_mask_for_norm=plan_cfg.get("use_mask_for_norm"),
                )
                loader_aug = probe.build_official_dataloader(
                    str(args.preprocessed_root), ids,
                    patch_size=patch_size, batch_size=batch_size, dataset_json=dataset_json,
                    oversample_foreground_percent=args.oversample_foreground_percent, transforms=transforms,
                    data_folder_name=args.data_folder_name,
                )
                batch_rows.extend(probe_batches(
                    loader=loader_aug, split_name=split_name, phase="post_aug", max_batches=args.max_batches,
                    batch_size=batch_size, oversample_foreground_percent=args.oversample_foreground_percent,
                    expected_positive=expected_positive, progress=progress,
                ))
        batch_summary = probe.aggregate_batch_rows(batch_rows)
        _write_csv(batch_rows, out_dir / "batch_probe.csv", BATCH_FIELDS)
        _write_json(batch_summary, out_dir / "batch_probe_summary.json")
        if augment_config is not None:
            _write_json(augment_config, out_dir / "augmentation_config.json")
            summary["outputs"]["augmentation_config"] = str(out_dir / "augmentation_config.json")
            summary["augmentation_config"] = augment_config
        summary["outputs"]["batch_probe"] = str(out_dir / "batch_probe.csv")
        for split_name, s in batch_summary.items():
            print(f"[采样探针] {split_name}: patch={s['n_patches']}，含前景 {s['n_patches_with_foreground']} "
                  f"({s['fg_patch_rate']:.1%})；FG 强制 patch={s['n_forced_foreground_patches']}，"
                  f"命中前景={s['n_forced_foreground_patches_with_fg']}"
                  f"（{s['forced_foreground_hit_rate']:.1%}）")
            if s.get("lost_foreground_after_aug_rate") is not None:
                print(f"          增强后前景消失（聚合估计）：{s['lost_foreground_after_aug_rate']:.1%}")

    # 5) 判定
    verdict = probe.build_verdict(
        static_ok=bool(static["ok"]) if static is not None else True,
        case_summary=case_summary,
        batch_summary=batch_summary,
        static_issues=(static or {}).get("failed_checks", []),
    )
    _write_json(verdict, out_dir / "verdict.json")
    summary["outputs"]["verdict"] = str(out_dir / "verdict.json")
    summary["verdict"] = verdict

    # 6) 运行摘要
    summary["elapsed_sec"] = round(time.time() - t0, 3)
    summary["counts"] = {
        "case_rows": len(case_rows),
        "batch_rows": len(batch_rows),
        "cases_with_issues": sum(1 for r in case_rows if r.get("n_issues")),
    }
    summary["seed"] = int(args.seed)
    _write_json(summary, out_dir / "run_summary.json")

    print()
    print("=" * 78)
    print(f"判定：{verdict['verdict']}")
    for reason in verdict["reasons"]:
        print(f"  - {reason}")
    print(f"建议下一步：{verdict['next_step']}")
    print(f"输出：{out_dir}")
    print(f"耗时：{summary['elapsed_sec']} s；成功/完成计数：{summary['counts']}；失败：0（逐例失败会写入 CSV）")
    print("=" * 78)

    if args.fail_on_anomaly and (summary["counts"]["cases_with_issues"] or not (static or {}).get("ok", True)):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
