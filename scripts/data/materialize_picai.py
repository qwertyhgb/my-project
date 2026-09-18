#!/usr/bin/env python
"""P0B PI-CAI 训练数据物化（病例内 T2W 参考网格对齐）。

规则（与 research_plan v2.0 / G0-P 条件一致）：
- 原始数据严格只读；输出写入 data/processed/picai/（不写原始数据目录）；
- 以每例轴位 T2W 的完整物理网格（size/spacing/origin/direction）为病例内唯一参考；
- ADC/HBV 按 SimpleITK 物理坐标重采样到 T2W 网格（线性插值），round+clip 存 uint16；
- canonical 病灶标签（Pooch25 优先，否则 human_expert/resampled）先二值化（2–5/1→1），
  再最近邻重采样到 T2W 网格；阴性保持全 0；
- Yuan23 主分区先验 / HeviAI23 敏感性先验：最近邻重采样，保持 {0,1,2}；
- Bosma22b WG 仅用于后续采样/QC；`11050_1001070` 的异常 WG 禁用且不替换（显式记录）。

输出布局（out_root = data/processed/picai）：
    cases/<case_id>/t2w.nii.gz, adc.nii.gz, hbv.nii.gz,
                     lesion.nii.gz, zonal_yuan.nii.gz, zonal_hevi.nii.gz[, wg.nii.gz]
    materialization_report.csv          每例一行（状态/校验值/大小/耗时）
    materialization_summary.json        结构化汇总（成功/失败/跳过/耗时/输出路径）
    materialization_validation.json/csv --verify 模式：1500 例完整性校验

安全与复现：
- 默认不覆盖已有完整结果；无 --resume/--overwrite 时若检出已有输出会直接中止；
- 每个文件先写 <case>/_tmp/<name>.nii.gz（保留 .nii.gz 后缀），读回校验后 os.replace 原子重命名；
- 单例失败逐例记录到 report CSV 并继续；结束后写结构化 summary；
- 相同参数可重复运行（--resume 跳过完整例、重做不完整例）。

用法示例：
    python scripts/data/materialize_picai.py --dry-run
    python scripts/data/materialize_picai.py --cases 10057_1000057,10000_1000000
    python scripts/data/materialize_picai.py --resume
    python scripts/data/materialize_picai.py --verify
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))
from zonal_reliability_fusion.data import picai as P  # noqa: E402

PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PROJECT / "data/metadata/picai_manifest.csv"
DEFAULT_OUT_ROOT = PROJECT / "data/processed/picai"

# 病例内固定文件名（模态顺序固定为 T2W、ADC、HBV）
SEQ_FILES = (("t2w", "t2w.nii.gz"), ("adc", "adc.nii.gz"), ("hbv", "hbv.nii.gz"))
LABEL_FILES = (("lesion", "lesion.nii.gz"), ("zonal_yuan", "zonal_yuan.nii.gz"),
               ("zonal_hevi", "zonal_hevi.nii.gz"), ("wg", "wg.nii.gz"))

# 已确认异常：Bosma22b WG 与 PZ∪TZ 无交集（官方 README 标注 faulty）。WG 禁用且不替换。
WG_EXCLUDED = {"11050_1001070"}

LESION_KNOWN = {0, 1}
ZONAL_KNOWN = {0, 1, 2}
WG_KNOWN = {0, 1}

# 磁盘估算用压缩率（保守上限，最终以实测为准）
COMPRESS_EST = 0.55


def expected_files(case_id: str) -> list[str]:
    """该例应输出的文件清单（异常例不含 WG）。"""
    names = [f for _, f in SEQ_FILES] + [f for k, f in LABEL_FILES if k != "wg"]
    if case_id not in WG_EXCLUDED:
        names.append("wg.nii.gz")
    return names


def free_space(p: Path) -> int:
    p = Path(p)
    while not p.exists() and p != p.parent:
        p = p.parent
    return shutil.disk_usage(p).free


def gb(n: float) -> str:
    return f"{n / 1e9:.1f} GB"


# --------------------------------------------------------------------------- #
# 读取与重采样（全部基于物理坐标，禁止数组索引对齐）
# --------------------------------------------------------------------------- #
def resample_label_array(arr: np.ndarray, src_geo: P.Geometry, ref: P.Geometry) -> np.ndarray:
    """把已读入的标签数组（携带 src_geo 几何）最近邻重采样到 ref 网格。"""
    img = sitk.GetImageFromArray(arr.astype(np.uint8))
    img.SetSpacing([float(v) for v in src_geo.spacing])
    img.SetOrigin([float(v) for v in src_geo.origin])
    img.SetDirection([float(v) for v in src_geo.direction])
    ref_img = P.image_from_geometry(ref, sitk.sitkFloat32)
    f = sitk.ResampleImageFilter()
    f.SetReferenceImage(ref_img)
    f.SetInterpolator(sitk.sitkNearestNeighbor)
    f.SetOutputPixelType(sitk.sitkUInt8)
    f.SetDefaultPixelValue(0)
    out = sitk.GetArrayFromImage(f.Execute(img))
    return np.asarray(out, dtype=np.uint8)


def load_image_u16(path: Path, ref: P.Geometry) -> tuple[np.ndarray, dict]:
    """读取影像并统一为 uint16：同网格直接取值，否则物理坐标线性重采样。"""
    src_geo = P.read_geometry(path)
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)
    info = {"src_dtype": str(arr.dtype), "resampled": False, "neg_clip": 0, "over_clip": 0}
    if P.grids_equal(ref, src_geo):
        if np.issubdtype(arr.dtype, np.integer):
            info["neg_clip"] = int((arr < 0).sum())
            info["over_clip"] = int((arr > 65535).sum())
            return np.clip(arr, 0, 65535).astype(np.uint16), info
        a = arr.astype(np.float32)
        if not np.all(np.isfinite(a)):
            raise ValueError(f"影像含非有限值: {path}")
        return np.clip(np.rint(a), 0, 65535).astype(np.uint16), info
    ref_img = P.image_from_geometry(ref, sitk.sitkFloat32)
    f = sitk.ResampleImageFilter()
    f.SetReferenceImage(ref_img)
    f.SetInterpolator(sitk.sitkLinear)
    f.SetDefaultPixelValue(0)
    a = sitk.GetArrayFromImage(f.Execute(img)).astype(np.float32)
    info["resampled"] = True
    if not np.all(np.isfinite(a)):
        raise ValueError(f"重采样产生非有限值: {path}")
    return np.clip(np.rint(a), 0, 65535).astype(np.uint16), info


def load_label_checked(path: Path, ref: P.Geometry, known: set) -> tuple[np.ndarray, bool]:
    """读取标签并校验唯一值，必要时最近邻重采样到 ref 网格。"""
    if not path.exists():
        raise FileNotFoundError(f"标签缺失: {path}")
    src_geo = P.read_geometry(path)
    arr = P.load_array(path)
    uv = {int(v) for v in np.unique(arr)}
    illegal = uv - known
    if illegal:
        raise ValueError(f"非法标签值 {sorted(illegal)}: {path}")
    if P.grids_equal(ref, src_geo):
        return arr.astype(np.uint8), False
    out = resample_label_array(arr, src_geo, ref)
    uv2 = {int(v) for v in np.unique(out)}
    illegal2 = uv2 - known
    if illegal2:
        raise ValueError(f"重采样后出现非法标签值 {sorted(illegal2)}: {path}")
    return out, True


def load_lesion_binary(root: Path, pid, sid, ref: P.Geometry) -> tuple[np.ndarray, str, int, str]:
    """canonical 病灶标签 → 二值 {0,1}（含物理重采样）；返回 (array, source, 前景体素, 网格关系)。"""
    src, path = P.canonical_lesion(root, pid, sid)
    if path is None or not Path(path).exists():
        raise FileNotFoundError("canonical 病灶标签缺失")
    geo = P.read_geometry(path)
    arr = P.load_array(path)
    binary = P.map_lesion_to_binary(arr, src)
    fg_pre = int(binary.sum())
    if P.grids_equal(ref, geo):
        return binary, src, fg_pre, P.GRID_EXACT
    out = resample_label_array(binary, geo, ref)
    uv = {int(v) for v in np.unique(out)}
    if not uv.issubset(LESION_KNOWN):
        raise ValueError(f"病灶重采样后出现非法值 {sorted(uv - LESION_KNOWN)}")
    return out, src, fg_pre, P.classify_grid(ref, geo)


# --------------------------------------------------------------------------- #
# 写出（临时文件 + 校验 + 原子重命名）
# --------------------------------------------------------------------------- #
def write_nifti(arr: np.ndarray, ref: P.Geometry, out_path: Path) -> None:
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing([float(v) for v in ref.spacing])
    img.SetOrigin([float(v) for v in ref.origin])
    img.SetDirection([float(v) for v in ref.direction])
    sitk.WriteImage(img, str(out_path), useCompression=True)


def validate_written(tmp_dir: Path, ref: P.Geometry, names: list[str],
                     lesion_fg_pre: int) -> dict:
    """读回临时文件做结构校验：几何一致 / 唯一值 / 阳性不丢失 / 阴性不凭空出现。"""
    v = {"validation_ok": True, "validation_detail": ""}
    detail = []
    ref_sz = tuple(int(x) for x in ref.size)
    for name in names:
        p = tmp_dir / name
        if not p.exists() or p.stat().st_size == 0:
            v["validation_ok"] = False
            detail.append(f"{name}:missing")
            continue
        g = P.read_geometry(p)
        if (tuple(int(x) for x in g.size) != ref_sz
                or not np.allclose(g.spacing, ref.spacing, atol=1e-4)
                or not np.allclose(g.origin, ref.origin, atol=1e-4)
                or not np.allclose(g.direction, ref.direction, atol=1e-4)):
            v["validation_ok"] = False
            detail.append(f"{name}:geometry")
        uv = {int(x) for x in np.unique(P.load_array(p))}
        if name == "lesion.nii.gz":
            v["lesion_unique"] = ",".join(map(str, sorted(uv)))
            if not uv.issubset(LESION_KNOWN):
                v["validation_ok"] = False
                detail.append(f"lesion:illegal{ sorted(uv - LESION_KNOWN) }")
        elif name.startswith("zonal_"):
            key = name.replace("zonal_", "").replace(".nii.gz", "")
            v[f"{key}_unique"] = ",".join(map(str, sorted(uv)))
            if not uv.issubset(ZONAL_KNOWN):
                v["validation_ok"] = False
                detail.append(f"{name}:illegal{ sorted(uv - ZONAL_KNOWN) }")
        elif name == "wg.nii.gz":
            v["wg_unique"] = ",".join(map(str, sorted(uv)))
            if not uv.issubset(WG_KNOWN):
                v["validation_ok"] = False
                detail.append(f"wg:illegal{ sorted(uv - WG_KNOWN) }")
    if "lesion_unique" in v:
        # 阳性不得因重采样消失；阴性不得凭空出现
        lesion_arr = P.load_array(tmp_dir / "lesion.nii.gz")
        fg_post = int((lesion_arr > 0).sum())
        v["lesion_fg_post"] = fg_post
        if lesion_fg_pre > 0 and fg_post == 0:
            v["validation_ok"] = False
            detail.append("lesion:positive_lost")
        if lesion_fg_pre == 0 and fg_post > 0:
            v["validation_ok"] = False
            detail.append("lesion:negative_became_positive")
    v["validation_detail"] = ";".join(detail)
    return v


# --------------------------------------------------------------------------- #
# 单例处理
# --------------------------------------------------------------------------- #
def process_case(root: Path, out_root: Path, row) -> dict:
    pid, sid = row["patient_id"], row["study_id"]
    cid = P.case_id(pid, sid)
    t0 = time.time()
    rec = {"case_id": cid, "patient_id": pid, "study_id": sid,
           "center": row.get("center", ""), "case_csPCa": row.get("case_csPCa", ""),
           "status": "", "canonical_source": "", "lesion_grid": "", "lesion_fg_pre": "",
           "lesion_fg_post": "", "lesion_unique": "", "yuan_unique": "", "hevi_unique": "",
           "wg_unique": "", "wg_status": "", "validation_ok": "", "validation_detail": "",
           "bytes_total": "", "elapsed_sec": "", "error": ""}
    tmp_dir = None
    try:
        names = expected_files(cid)
        t2w_path = P.image_path(root, pid, sid, "t2w")
        ref = P.read_geometry(t2w_path)

        arrays: dict[str, np.ndarray] = {}
        for seq, fname in SEQ_FILES:
            p = P.image_path(root, pid, sid, seq)
            arr, info = load_image_u16(p, ref)
            arrays[fname] = arr
            rec[f"{seq}_src_dtype"] = info["src_dtype"]
            rec[f"{seq}_resampled"] = info["resampled"]
            if info["neg_clip"] or info["over_clip"]:
                rec[f"{seq}_clip"] = f"neg{info['neg_clip']}/over{info['over_clip']}"

        binary, src, fg_pre, grid_rel = load_lesion_binary(root, pid, sid, ref)
        arrays["lesion.nii.gz"] = binary
        rec["canonical_source"] = src
        rec["lesion_grid"] = grid_rel
        rec["lesion_fg_pre"] = fg_pre

        for key, sub, fname in (("yuan", P.ZONAL_YUAN23, "zonal_yuan.nii.gz"),
                                ("hevi", P.ZONAL_HEVIAI23, "zonal_hevi.nii.gz")):
            arr, was_res = load_label_checked(P.label_path(root, sub, pid, sid), ref, ZONAL_KNOWN)
            arrays[fname] = arr
            rec[f"{key}_resampled"] = was_res

        if cid in WG_EXCLUDED:
            # 已确认 faulty WG：禁用且不替换（不静默）
            rec["wg_status"] = "excluded_known_faulty_bosma22b"
        else:
            arr, was_res = load_label_checked(P.label_path(root, P.WG_BOSMA22B, pid, sid), ref, WG_KNOWN)
            arrays["wg.nii.gz"] = arr
            rec["wg_status"] = "resampled" if was_res else "exact"

        # 临时目录（只清除本脚本自己的临时目录）
        case_dir = out_root / "cases" / cid
        tmp_dir = case_dir / "_tmp"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        for fname, arr in arrays.items():
            write_nifti(arr, ref, tmp_dir / fname)

        v = validate_written(tmp_dir, ref, names, fg_pre)
        rec.update(v)
        if not v["validation_ok"]:
            raise ValueError(f"写出校验失败: {v['validation_detail']}")

        for fname in names:
            os.replace(tmp_dir / fname, case_dir / fname)
        tmp_dir.rmdir()

        rec["status"] = "ok"
        rec["bytes_total"] = int(sum((case_dir / f).stat().st_size for f in names))
    except Exception as e:  # noqa: BLE001
        rec["status"] = "failed"
        rec["error"] = f"{type(e).__name__}: {e}"
    rec["elapsed_sec"] = round(time.time() - t0, 2)
    return rec


# --------------------------------------------------------------------------- #
# 选择 / 估算 / 报告
# --------------------------------------------------------------------------- #
def select_cases(args, df: pd.DataFrame) -> list[str]:
    if args.cases:
        ids = [s.strip() for s in args.cases.split(",") if s.strip()]
        known = set(df["case_id"])
        unknown = [c for c in ids if c not in known]
        if unknown:
            raise SystemExit(f"[materialize] --cases 中 {len(unknown)} 例不在 manifest: {unknown[:5]}")
        return ids
    if args.limit:
        return df["case_id"].head(args.limit).tolist()
    return df["case_id"].tolist()


def estimate_disk(df: pd.DataFrame, sel: list[str], out_root: Path) -> dict:
    sub = df[df["case_id"].isin(sel)]
    sizes = []
    for s in sub["t2w_size"].fillna(""):
        sizes.append(float(np.prod([int(x) for x in str(s).split(";")])) if str(s) else 0.0)
    n_vox = float(np.sum(sizes))
    raw = n_vox * (3 * 2 + 4 * 1)  # 3 影像 uint16 + 4 标签 uint8
    est = raw * COMPRESS_EST
    free = free_space(out_root)
    return {"n_cases": len(sub), "n_vox_total": n_vox, "bytes_raw": raw,
            "bytes_est_compressed": est, "free_bytes": free,
            "ok": free > est * 1.5}


def write_report(path: Path, report: dict) -> None:
    """写入完整 report（含本次未处理的既有行，避免子集重跑丢失历史）。"""
    if not report:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(report.values())).to_csv(path, index=False)


def write_summary(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))


# --------------------------------------------------------------------------- #
# --verify：1500 例完整性校验（仅读头部，不重读体素）
# --------------------------------------------------------------------------- #
def run_verify(args, df: pd.DataFrame, sel: list[str], out_root: Path, no_progress: bool) -> None:
    rows = []
    for cid in tqdm(sel, desc="verify", mininterval=1.0, disable=no_progress, file=sys.stdout):
        case_dir = out_root / "cases" / cid
        names = expected_files(cid)
        missing = [n for n in names if not (case_dir / n).exists() or (case_dir / n).stat().st_size == 0]
        extra_wg = (cid in WG_EXCLUDED) and (case_dir / "wg.nii.gz").exists()
        geo_ok, geo_err, sizes = "", "", {}
        if not missing:
            try:
                geos = {n: P.read_geometry(case_dir / n) for n in names}
                ref = geos["t2w.nii.gz"]
                bad = [n for n, g in geos.items() if
                       tuple(int(x) for x in g.size) != tuple(int(x) for x in ref.size)
                       or not np.allclose(g.spacing, ref.spacing, atol=1e-4)
                       or not np.allclose(g.origin, ref.origin, atol=1e-4)
                       or not np.allclose(g.direction, ref.direction, atol=1e-4)]
                geo_ok = len(bad) == 0
                geo_err = ",".join(bad)
                sizes = {n: int((case_dir / n).stat().st_size) for n in names}
            except Exception as e:  # noqa: BLE001
                geo_ok = False
                geo_err = f"{type(e).__name__}: {e}"
        if extra_wg:
            status = "unexpected_wg_for_excluded_case"
        elif missing:
            status = "missing_files"
        elif not geo_ok:
            status = "geometry_mismatch"
        else:
            status = "complete"
        rows.append({"case_id": cid, "status": status, "missing": ",".join(missing),
                     "extra_wg": bool(extra_wg), "geometry_ok": geo_ok, "geometry_err": geo_err,
                     "bytes_total": int(sum(sizes.values())) if sizes else "",
                     "n_files": len(names) - len(missing)})
    out = pd.DataFrame(rows)
    csv_path = out_root / "materialization_validation.csv"
    json_path = out_root / "materialization_validation.json"
    out.to_csv(csv_path, index=False)
    counts = out["status"].value_counts().to_dict()
    summary = {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "out_root": str(out_root), "n_expected": len(sel),
        "counts": {k: int(v) for k, v in counts.items()},
        "n_complete": int((out["status"] == "complete").sum()),
        "all_complete": bool((out["status"] == "complete").all()),
        "problem_cases": out[out["status"] != "complete"]["case_id"].tolist(),
        "bytes_total": int(pd.to_numeric(out["bytes_total"], errors="coerce").fillna(0).sum()),
        "csv": str(csv_path),
    }
    write_summary(json_path, summary)
    print(f"[verify] done: {summary['counts']} all_complete={summary['all_complete']} "
          f"bytes_total={gb(summary['bytes_total'])} out={csv_path}")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="P0B PI-CAI 训练数据物化（T2W 参考网格对齐）")
    ap.add_argument("--data-root", default=str(P.DEFAULT_DATA_ROOT), help="PI-CAI 原始数据根（只读）")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--cases", default="", help="逗号分隔的 case_id 子集（smoke test）")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="只做磁盘估算，不写任何数据")
    ap.add_argument("--resume", action="store_true", help="跳过已有完整结果，重做不完整例")
    ap.add_argument("--overwrite", action="store_true", help="强制重做所选病例")
    ap.add_argument("--verify", action="store_true", help="只校验已有物化结果（1500 例完整性）")
    ap.add_argument("--flush-every", type=int, default=25, help="report CSV 增量落盘间隔")
    ap.add_argument("--no-progress", action="store_true", help="关闭进度条（日志环境）")
    args = ap.parse_args()

    root = Path(args.data_root)
    out_root = Path(args.out_root)
    df = pd.read_csv(args.manifest)
    if "case_id" not in df.columns:
        raise SystemExit(f"[materialize] manifest 缺少 case_id 列: {args.manifest}")
    sel = select_cases(args, df)
    print(f"[materialize] manifest={args.manifest} cases_selected={len(sel)} out_root={out_root}")

    if args.verify:
        run_verify(args, df, sel, out_root, args.no_progress)
        return

    est = estimate_disk(df, sel, out_root)
    print(f"[estimate] cases={est['n_cases']} 参考网格总预算体素={est['n_vox_total']:.3g}")
    print(f"[estimate] 未压缩≈{gb(est['bytes_raw'])} 预计压缩后≈{gb(est['bytes_est_compressed'])} "
          f"剩余空间={gb(est['free_bytes'])} headroom_ok={est['ok']}")
    if args.dry_run:
        print("[dry-run] 未写入任何数据。")
        return
    if not est["ok"]:
        raise SystemExit("[materialize] 空间不足（需 free > 1.5×预计压缩体积），已中止。")

    report_path = out_root / "materialization_report.csv"
    summary_path = out_root / "materialization_summary.json"
    report = {}
    if report_path.exists():
        report = {r["case_id"]: r for r in pd.read_csv(report_path).to_dict("records")}
        print(f"[materialize] 载入既有 report {len(report)} 行（仅更新所处理病例，历史行保留）")

    # 安全门：已有输出且未声明 --resume/--overwrite 时中止
    existing = []
    for cid in sel:
        case_dir = out_root / "cases" / cid
        if any((case_dir / n).exists() for n in expected_files(cid)):
            existing.append(cid)
    if existing and not (args.resume or args.overwrite):
        raise SystemExit(
            f"[materialize] 检出 {len(existing)} 例已有输出（如 {existing[:3]}）。"
            f"如需跳过完整结果加 --resume；如需强制重做加 --overwrite。已中止。")

    t0 = time.time()
    n_ok = n_failed = n_skipped = 0
    try:
        for i, cid in enumerate(tqdm(sel, desc="materialize", mininterval=1.0,
                                     disable=args.no_progress, file=sys.stdout)):
            case_dir = out_root / "cases" / cid
            complete = all((case_dir / n).exists() and (case_dir / n).stat().st_size > 0
                           for n in expected_files(cid))
            if complete and not args.overwrite:
                n_skipped += 1
                if args.resume:
                    row_dict = report.get(cid)
                    if row_dict is None:
                        row = df[df["case_id"] == cid].iloc[0]
                        row_dict = {"case_id": cid, "patient_id": row["patient_id"],
                                    "study_id": row["study_id"], "center": row.get("center", ""),
                                    "case_csPCa": row.get("case_csPCa", ""), "status": "skipped_complete"}
                        report[cid] = row_dict
                continue
            row = df[df["case_id"] == cid].iloc[0]
            rec = process_case(root, out_root, row)
            report[cid] = rec
            if rec["status"] == "ok":
                n_ok += 1
            else:
                n_failed += 1
            if (i + 1) % args.flush_every == 0:
                write_report(report_path, report)
    finally:
        write_report(report_path, report)
        statuses = [report[c]["status"] for c in sel if c in report]
        summary = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "data_root": str(root), "out_root": str(out_root), "manifest": str(args.manifest),
            "n_selected": len(sel), "n_ok": statuses.count("ok"),
            "n_failed": statuses.count("failed"), "n_skipped_complete": statuses.count("skipped_complete"),
            "failed_cases": [c for c in sel if report.get(c, {}).get("status") == "failed"],
            "wg_excluded": sorted(WG_EXCLUDED & set(sel)),
            "bytes_total": int(sum(int(report[c]["bytes_total"]) for c in sel
                                   if report.get(c, {}).get("status") == "ok"
                                   and str(report[c].get("bytes_total", "")).isdigit())),
            "elapsed_sec": round(time.time() - t0, 1),
            "report_csv": str(report_path),
        }
        write_summary(summary_path, summary)
        print(f"[materialize] done: ok={summary['n_ok']} failed={summary['n_failed']} "
              f"skipped={summary['n_skipped_complete']} elapsed={summary['elapsed_sec']}s "
              f"bytes_total={gb(summary['bytes_total'])}")
        print(f"[materialize] out: report={report_path} summary={summary_path}")
        if summary["failed_cases"]:
            print(f"[materialize][WARN] failed_cases({len(summary['failed_cases'])}): "
                  f"{summary['failed_cases'][:10]}")


if __name__ == "__main__":
    main()
