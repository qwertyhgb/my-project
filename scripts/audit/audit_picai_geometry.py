#!/usr/bin/env python
"""G0-3 全量物理几何审计。

对每 study 的 T2W/ADC/HBV/canonical-lesion/WG/Yuan23/HeviAI23 计算完整几何：
size/spacing/origin/direction/orientation、8 角点物理坐标(经 SimpleITK
index-to-physical-point)、物理包围盒与 extent、相对 T2W 的网格关系五分类
(exact_same_grid / same_physical_space_different_grid / partial_physical_overlap /
geometry_suspect / missing_or_unreadable)、重叠率与是否需重采样。

输出长表 geometry_audit.csv 与异常表 geometry_anomalies.csv。原始数据只读。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))
from zonal_reliability_fusion.data import picai as P  # noqa: E402

ROLES = ["t2w", "adc", "hbv", "lesion", "wg", "zonal_yuan", "zonal_hevi"]
ANOM_SEV = {P.GRID_MISSING: 3, P.GRID_PARTIAL: 2, P.GRID_SUSPECT: 2, P.GRID_DIFFERENT: 0, P.GRID_EXACT: 0}


def role_path(root, pid, sid, role):
    if role == "t2w":
        return P.image_path(root, pid, sid, "t2w")
    if role == "adc":
        return P.image_path(root, pid, sid, "adc")
    if role == "hbv":
        return P.image_path(root, pid, sid, "hbv")
    if role == "lesion":
        return P.canonical_lesion(root, pid, sid)[1]
    if role == "wg":
        return P.label_path(root, P.WG_BOSMA22B, pid, sid)
    if role == "zonal_yuan":
        return P.label_path(root, P.ZONAL_YUAN23, pid, sid)
    if role == "zonal_hevi":
        return P.label_path(root, P.ZONAL_HEVIAI23, pid, sid)
    raise ValueError(role)


def audit_one(root, pid, sid, center, cspca):
    ref = None
    try:
        ref = P.read_geometry(P.image_path(root, pid, sid, "t2w"))
    except Exception:  # noqa: BLE001
        ref = None
    ref_orient = P.orientation_code(ref) if ref else ""
    rows = []
    for role in ROLES:
        path = role_path(root, pid, sid, role)
        rec = {"case_id": P.case_id(pid, sid), "patient_id": pid, "study_id": sid,
               "center": center, "case_csPCa": cspca, "role": role,
               "path": str(path) if path else "", "exists": bool(path and Path(path).exists())}
        geo = None
        err = ""
        if path and Path(path).exists():
            try:
                geo = P.read_geometry(path)
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {e}"
        if geo is None:
            rec.update({"readable": False, "size": "", "spacing": "", "origin": "", "direction": "",
                        "orientation": "", "corner_min_x": "", "corner_min_y": "", "corner_min_z": "",
                        "corner_max_x": "", "corner_max_y": "", "corner_max_z": "",
                        "extent_x": "", "extent_y": "", "extent_z": "",
                        "same_grid": "", "grid_status": P.GRID_MISSING,
                        "overlap_x": "", "overlap_y": "", "overlap_z": "", "needs_resample": ""})
            rec["anomaly_reason"] = ("文件缺失或不可读: " + err) if err else "文件缺失"
            rec["severity"] = ANOM_SEV[P.GRID_MISSING]
        else:
            mn, mx = P.physical_bbox(geo)
            ext = P.physical_extent(geo)
            orient = P.orientation_code(geo)
            if ref is None:
                status = P.GRID_MISSING
            elif role == "t2w":
                status = P.GRID_EXACT
            else:
                status = P.classify_grid(ref, geo)
            ov, ratio = P.axes_overlap(ref, geo) if ref else (np.full(3, np.nan), np.full(3, np.nan))
            reasons = []
            if status == P.GRID_PARTIAL:
                reasons.append("与T2W仅部分物理重叠")
            elif status == P.GRID_SUSPECT:
                reasons.append("几何可疑(方向或覆盖范围异常)")
            if ref and orient != ref_orient:
                reasons.append(f"orientation不一致({orient}vs{ref_orient})")
            if np.any(ext <= 0):
                reasons.append("extent非正")
            rec.update({"readable": True, "size": ";".join(map(str, geo.size)),
                        "spacing": ";".join(f"{v:.4f}" for v in geo.spacing),
                        "origin": ";".join(f"{v:.3f}" for v in geo.origin),
                        "direction": ";".join(f"{v:.4f}" for v in geo.direction),
                        "orientation": orient,
                        "corner_min_x": f"{mn[0]:.3f}", "corner_min_y": f"{mn[1]:.3f}", "corner_min_z": f"{mn[2]:.3f}",
                        "corner_max_x": f"{mx[0]:.3f}", "corner_max_y": f"{mx[1]:.3f}", "corner_max_z": f"{mx[2]:.3f}",
                        "extent_x": f"{ext[0]:.2f}", "extent_y": f"{ext[1]:.2f}", "extent_z": f"{ext[2]:.2f}",
                        "same_grid": bool(ref and P.grids_equal(ref, geo)),
                        "grid_status": status,
                        "overlap_x": f"{ratio[0]:.4f}", "overlap_y": f"{ratio[1]:.4f}", "overlap_z": f"{ratio[2]:.4f}",
                        "needs_resample": role != "t2w" and status in (P.GRID_DIFFERENT, P.GRID_PARTIAL, P.GRID_SUSPECT)})
            rec["anomaly_reason"] = "; ".join(reasons)
            rec["severity"] = max([ANOM_SEV[status]] + ([2] if reasons and status in (P.GRID_EXACT, P.GRID_DIFFERENT) else [0]))
        rows.append(rec)
    return rows


def main():
    ap = argparse.ArgumentParser(description="PI-CAI 全量物理几何审计（G0-3）")
    ap.add_argument("--data-root", default=str(P.DEFAULT_DATA_ROOT))
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parents[2] / "outputs/diagnostics/picai_model_readiness"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-progress", action="store_true", help="关闭进度条（日志环境）")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    np.random.seed(args.seed)

    root = Path(args.data_root)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ms = P.load_marksheet(root)
    if args.limit:
        ms = ms[:args.limit]
    print(f"[geometry] studies={len(ms)} out={out_dir}")

    t0 = time.time()
    all_rows, errors = [], []
    for row in tqdm(ms, desc="geometry", mininterval=1.0, disable=args.no_progress, file=sys.stdout):
        try:
            all_rows.extend(audit_one(root, row["patient_id"], row["study_id"], row.get("center", ""), row.get("case_csPCa", "")))
        except Exception as e:  # noqa: BLE001
            errors.append({"case_id": P.case_id(row["patient_id"], row["study_id"]), "error": f"{type(e).__name__}: {e}"})

    df = pd.DataFrame(all_rows)
    audit_path = out_dir / "geometry_audit.csv"
    df.to_csv(audit_path, index=False)
    anom = df[(df["grid_status"].isin([P.GRID_MISSING, P.GRID_PARTIAL, P.GRID_SUSPECT])) | (df["anomaly_reason"] != "")].copy()
    anom = anom.sort_values(["severity", "case_id"], ascending=[False, True])
    anom_path = out_dir / "geometry_anomalies.csv"
    anom.to_csv(anom_path, index=False)

    print(f"[geometry] wrote {audit_path} rows={len(df)} elapsed={time.time()-t0:.1f}s errors={len(errors)}")
    print(f"[geometry] wrote {anom_path} anomaly_rows={len(anom)}")
    print(f"[geometry] done: rows={len(df)} anomalies={len(anom)} errors={len(errors)} elapsed={time.time()-t0:.1f}s out_dir={out_dir}")
    print("[geometry] grid_status 分布:")
    print(df.groupby(["role", "grid_status"]).size().unstack(fill_value=0).to_string())
    if errors:
        pd.DataFrame(errors).to_csv(out_dir / "geometry_build_errors.csv", index=False)
        print(f"[geometry] build_errors={len(errors)} -> geometry_build_errors.csv")


if __name__ == "__main__":
    main()
