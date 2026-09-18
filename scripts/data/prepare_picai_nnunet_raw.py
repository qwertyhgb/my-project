#!/usr/bin/env python
"""P0B/N0 准备：把物化结果组织为 nnU-Net v2 raw 数据集（符号链接，不复制体素）。

约定（third_party/nnUNet v2.6.2，documentation/dataset_format.md）：
    <nnUNet_raw>/Dataset605_PICAI/
        imagesTr/<case>_0000.nii.gz  -> cases/<case>/t2w.nii.gz
        imagesTr/<case>_0001.nii.gz  -> cases/<case>/adc.nii.gz
        imagesTr/<case>_0002.nii.gz  -> cases/<case>/hbv.nii.gz
        labelsTr/<case>.nii.gz       -> cases/<case>/lesion.nii.gz
        dataset.json                 （channel_names 0/1/2 = T2W/ADC/HBV；labels: background/lesion）

模态顺序固定为 T2W、ADC、HBV（research_plan §16）。链接使用相对路径，
避免移动目录后失效；原始物化文件仍留在 data/processed/picai/cases/。

安全：默认不覆盖已有完整结果；数据集目录已存在且未声明 --resume/--overwrite 时中止；
--resume 跳过已存在且指向正确目标的链接并重建缺失链接；--overwrite 强制重建。

用法：
    python scripts/data/prepare_picai_nnunet_raw.py --dry-run
    python scripts/data/prepare_picai_nnunet_raw.py --cases 10005_1000005
    python scripts/data/prepare_picai_nnunet_raw.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_MATERIALIZED = PROJECT / "data/processed/picai"
DEFAULT_MANIFEST = PROJECT / "data/metadata/picai_manifest.csv"
DEFAULT_MAPPING = PROJECT / "data/metadata/picai_nnunet_case_mapping.csv"
DEFAULT_SUMMARY = PROJECT / "data/metadata/picai_nnunet_raw_prep_summary.json"

# 通道定义（顺序即 nnU-Net 通道号，固定不可变）
CHANNELS = (("0000", "T2W", "t2w.nii.gz"), ("0001", "ADC", "adc.nii.gz"), ("0002", "HBV", "hbv.nii.gz"))


def link_ok(link: Path, target: Path) -> bool:
    return link.is_symlink() and Path(os.path.realpath(link)) == Path(os.path.realpath(target))


def main() -> None:
    ap = argparse.ArgumentParser(description="准备 nnU-Net raw 数据集（symlink）")
    ap.add_argument("--materialized-root", default=str(DEFAULT_MATERIALIZED))
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--nnunet-raw-root", default=os.environ.get("nnUNet_raw", ""),
                    help="nnUNet_raw 根目录（默认取环境变量 nnUNet_raw）")
    ap.add_argument("--dataset-id", type=int, default=605)
    ap.add_argument("--dataset-name", default="PICAI")
    ap.add_argument("--cases", default="", help="逗号分隔 case_id 子集")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--mapping-out", default=str(DEFAULT_MAPPING))
    ap.add_argument("--summary-out", default=str(DEFAULT_SUMMARY))
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args()

    if not args.nnunet_raw_root:
        raise SystemExit("[nnunet-raw] 未找到 nnUNet_raw：请设置环境变量或显式传 --nnunet-raw-root")
    raw_root = Path(args.nnunet_raw_root)
    ds_dir = raw_root / f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    mat_root = Path(args.materialized_root)

    df = pd.read_csv(args.manifest)
    if args.cases:
        sel = [s.strip() for s in args.cases.split(",") if s.strip()]
        unknown = [c for c in sel if c not in set(df["case_id"])]
        if unknown:
            raise SystemExit(f"[nnunet-raw] --cases 含未知 case_id: {unknown[:5]}")
    else:
        sel = df["case_id"].tolist()

    # 冲突检查：同 ID 不同名的其他数据集
    if raw_root.exists():
        others = [p.name for p in raw_root.glob(f"Dataset{args.dataset_id:03d}_*") if p.name != ds_dir.name]
        if others:
            raise SystemExit(f"[nnunet-raw] Dataset ID {args.dataset_id:03d} 已被占用: {others}")
    # 目标数据集目录状态
    if ds_dir.exists() and any(ds_dir.rglob("*")):
        if not (args.resume or args.overwrite):
            raise SystemExit(f"[nnunet-raw] 目标已存在内容: {ds_dir}。加 --resume 跳过/补齐，或 --overwrite 重建。")
    print(f"[nnunet-raw] dataset={ds_dir.name} cases_selected={len(sel)} raw_root={raw_root}")

    rows, missing = [], []
    for cid in tqdm(sel, desc="nnunet-raw", mininterval=1.0,
                    disable=args.no_progress, file=sys.stdout):
        case_dir = mat_root / "cases" / cid
        srcs = {fname: case_dir / fname for _, _, fname in CHANNELS}
        srcs["lesion.nii.gz"] = case_dir / "lesion.nii.gz"
        miss = [str(p) for p in srcs.values() if not p.exists() or p.stat().st_size == 0]
        if miss:
            missing.append({"case_id": cid, "missing": miss})
            continue
        if args.dry_run:
            rows.append({"case_id": cid, "img_links": 3, "label_link": 1, "dry_run": True})
            continue
        img_dir = ds_dir / "imagesTr"
        lab_dir = ds_dir / "labelsTr"
        img_dir.mkdir(parents=True, exist_ok=True)
        lab_dir.mkdir(parents=True, exist_ok=True)
        ok = True
        for suffix, _, fname in CHANNELS:
            link = img_dir / f"{cid}_{suffix}.nii.gz"
            target = case_dir / fname
            rel = Path(os.path.relpath(target, link.parent))
            if link.exists() or link.is_symlink():
                if link_ok(link, target):
                    continue
                if not args.overwrite:
                    print(f"[nnunet-raw][WARN] 链接已存在但目标不符: {link} -> {os.readlink(link)}")
                    ok = False
                    break
                link.unlink()
            os.symlink(rel, link)
        if not ok:
            missing.append({"case_id": cid, "missing": ["link_conflict"]})
            continue
        lab_link = lab_dir / f"{cid}.nii.gz"
        lab_target = case_dir / "lesion.nii.gz"
        if not (lab_link.exists() or lab_link.is_symlink()) or args.overwrite:
            if lab_link.is_symlink() or lab_link.exists():
                lab_link.unlink()
            os.symlink(Path(os.path.relpath(lab_target, lab_link.parent)), lab_link)
        rows.append({"case_id": cid, "img_links": 3, "label_link": 1, "dry_run": False})

    n_ok = sum(1 for r in rows if not r.get("dry_run"))
    if args.dry_run:
        print(f"[nnunet-raw] dry-run: 可链接 {len(rows)} 例，缺失 {len(missing)} 例")
        return

    # dataset.json
    n_total = len(sel)
    dataset_json = {
        "name": ds_dir.name,
        "description": "PI-CAI csPCa lesion segmentation (T2W/ADC/HBV resampled to per-case T2W grid; canonical binary lesion)",
        "channel_names": {suffix: name for suffix, name, _ in CHANNELS},
        "labels": {"background": 0, "lesion": 1},
        "numTraining": n_total,
        "file_ending": ".nii.gz",
    }
    (ds_dir / "dataset.json").write_text(json.dumps(dataset_json, indent=2, ensure_ascii=False) + "\n")

    # case mapping（含中心/阳性与源路径）
    meta = df.set_index("case_id")
    mapping_rows = []
    for cid in sel:
        row = meta.loc[cid] if cid in meta.index else None
        mapping_rows.append({
            "case_id": cid,
            "center": row["center"] if row is not None else "",
            "case_csPCa": row["case_csPCa"] if row is not None else "",
            "imagesTr_0000": str(ds_dir / "imagesTr" / f"{cid}_0000.nii.gz"),
            "imagesTr_0001": str(ds_dir / "imagesTr" / f"{cid}_0001.nii.gz"),
            "imagesTr_0002": str(ds_dir / "imagesTr" / f"{cid}_0002.nii.gz"),
            "labelsTr": str(ds_dir / "labelsTr" / f"{cid}.nii.gz"),
            "source_case_dir": str(mat_root / "cases" / cid),
        })
    mapping_path = Path(args.mapping_out)
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(mapping_rows).to_csv(mapping_path, index=False)

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_dir": str(ds_dir), "dataset_name": ds_dir.name,
        "n_selected": len(sel), "n_linked": n_ok, "n_missing": len(missing),
        "missing_cases": missing,
        "channel_order": {suffix: name for suffix, name, _ in CHANNELS},
        "labels": {"background": 0, "lesion": 1},
        "n_imagesTr": len(list((ds_dir / "imagesTr").glob("*.nii.gz"))),
        "n_labelsTr": len(list((ds_dir / "labelsTr").glob("*.nii.gz"))),
        "mapping_csv": str(mapping_path), "dataset_json": str(ds_dir / "dataset.json"),
    }
    Path(args.summary_out).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[nnunet-raw] done: linked={n_ok} missing={len(missing)} imagesTr={summary['n_imagesTr']} "
          f"labelsTr={summary['n_labelsTr']}")
    print(f"[nnunet-raw] dataset.json={ds_dir / 'dataset.json'} mapping={mapping_path} summary={args.summary_out}")
    if missing:
        print(f"[nnunet-raw][WARN] missing({len(missing)}): {[m['case_id'] for m in missing[:10]]}")


if __name__ == "__main__":
    main()
