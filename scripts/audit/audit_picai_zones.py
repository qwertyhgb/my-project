#!/usr/bin/env python
"""G0-4 腺体与分区先验全量审计。

对每 study 将 WG(Bosma22b)/Yuan23/HeviAI23 按物理坐标最近邻重采样到 T2W 网格，
统计：空 mask、非法标签、前景物理体积、3D 连通域数、最大连通域比例、PZ/TZ 并存
与异常重叠、PZ∪TZ 与 WG 的 Dice、PZ∪TZ 越界(WG 外)比例、WG 未覆盖比例、
Yuan vs Hevi 的 PZ/TZ Dice 与体积差异，并按中心分层、异常病例排名。

支持 --resume 断点续跑（按 case_id 跳过已完成）。检查官方提示异常 WG 病例 11050_1001070。
原始数据只读；输出写入 outputs/diagnostics/picai_model_readiness/。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))
from zonal_reliability_fusion.data import picai as P  # noqa: E402

KNOWN_WG = {0, 1}
KNOWN_ZONAL = {0, 1, 2}
CHECK_CASE = "11050_1001070"


def load_label(path, t2w_geo):
    """读取标签并(如需)NN重采样到 T2W 网格；返回 (array, was_resampled)。"""
    geo = P.read_geometry(path)
    arr = P.load_array(path)
    if P.grids_equal(t2w_geo, geo):
        return arr, False
    return P.resample_to_geometry(path, t2w_geo, is_label=True), True


def n_components(mask):
    if not mask.any():
        return 0, 0.0
    lab, n = ndimage.label(mask)
    if n == 0:
        return 0, 0.0
    sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
    return int(n), float(sizes.max() / max(mask.sum(), 1))


def audit_case(root, pid, sid, center, cspca):
    cid = P.case_id(pid, sid)
    t2w_geo = P.read_geometry(P.image_path(root, pid, sid, "t2w"))
    vox_ml = float(np.prod(t2w_geo.spacing)) / 1000.0
    rec = {"case_id": cid, "patient_id": pid, "study_id": sid, "center": center, "case_csPCa": cspca}

    paths = {"wg": P.label_path(root, P.WG_BOSMA22B, pid, sid),
             "yuan": P.label_path(root, P.ZONAL_YUAN23, pid, sid),
             "hevi": P.label_path(root, P.ZONAL_HEVIAI23, pid, sid)}
    masks = {}
    for k, p in paths.items():
        if not p.exists():
            rec[f"{k}_exists"] = False
            continue
        rec[f"{k}_exists"] = True
        arr, was_res = load_label(p, t2w_geo)
        masks[k] = arr
        rec[f"{k}_resampled"] = was_res
        uv = set(int(x) for x in np.unique(arr).tolist())
        rec[f"{k}_unique"] = ",".join(map(str, sorted(uv)))
        rec[f"{k}_illegal"] = not uv.issubset(KNOWN_WG if k == "wg" else KNOWN_ZONAL)
        rec[f"{k}_empty"] = not arr.any()
        rec[f"{k}_volume_ml"] = float(arr.astype(bool).sum()) * vox_ml
        n, maxr = n_components(arr.astype(bool))
        rec[f"{k}_n_components"] = n
        rec[f"{k}_largest_cc_ratio"] = maxr

    # WG 与分区关系
    if "wg" in masks and "yuan" in masks:
        rec.update(relations(masks["wg"].astype(bool), masks["yuan"], vox_ml, "yuan"))
    if "wg" in masks and "hevi" in masks:
        rec.update(relations(masks["wg"].astype(bool), masks["hevi"], vox_ml, "hevi"))
    # Yuan vs Hevi
    if "yuan" in masks and "hevi" in masks:
        py, ty = (masks["yuan"] == 1), (masks["yuan"] == 2)
        ph, th = (masks["hevi"] == 1), (masks["hevi"] == 2)
        rec["pz_dice_yuan_hevi"] = P.dice_score(py, ph)
        rec["tz_dice_yuan_hevi"] = P.dice_score(ty, th)
        rec["pz_vol_diff_ml"] = float(abs(py.sum() - ph.sum())) * vox_ml
        rec["tz_vol_diff_ml"] = float(abs(ty.sum() - th.sum())) * vox_ml
        # PZ/TZ 同时存在 & 重叠（对每个分区来源）
        for nm, pz, tz in (("yuan", py, ty), ("hevi", ph, th)):
            rec[f"{nm}_pztz_both"] = bool(pz.any() and tz.any())
            ov = int((pz & tz).sum())
            rec[f"{nm}_pztz_overlap_vox"] = ov
            rec[f"{nm}_pztz_overlap_flag"] = ov > 0

    rec["n_anomaly_flags"] = sum(1 for kk, vv in rec.items() if kk.endswith(("_empty", "_illegal", "_overlap_flag")) and vv is True)
    rec["low_wg_zonal_dice"] = bool(min([rec.get("yuan_pztz_wg_dice", 1), rec.get("hevi_pztz_wg_dice", 1)]) < 0.9)
    return rec


def relations(wg, zonal, vox_ml, name):
    pz = (zonal == 1)
    tz = (zonal == 2)
    pztz = pz | tz
    out = {}
    out[f"{name}_pztz_volume_ml"] = float(pztz.sum()) * vox_ml
    out[f"{name}_pz_volume_ml"] = float(pz.sum()) * vox_ml
    out[f"{name}_tz_volume_ml"] = float(tz.sum()) * vox_ml
    out[f"{name}_pztz_wg_dice"] = P.dice_score(pztz, wg)
    pztz_n, pztz_cc = n_components(pztz)
    out[f"{name}_pztz_n_components"] = pztz_n
    out[f"{name}_pztz_largest_cc_ratio"] = pztz_cc
    if pztz.any():
        out[f"{name}_pztz_outside_wg_ratio"] = float((pztz & ~wg).sum() / pztz.sum())
    else:
        out[f"{name}_pztz_outside_wg_ratio"] = ""
    if wg.any():
        out[f"{name}_wg_uncovered_ratio"] = float((wg & ~pztz).sum() / wg.sum())
    else:
        out[f"{name}_wg_uncovered_ratio"] = ""
    return out


def main():
    ap = argparse.ArgumentParser(description="PI-CAI 腺体/分区先验全量审计（G0-4）")
    ap.add_argument("--data-root", default=str(P.DEFAULT_DATA_ROOT))
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parents[2] / "outputs/diagnostics/picai_model_readiness"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="跳过 zonal_quality.csv 中已完成的 case_id")
    ap.add_argument("--flush-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    np.random.seed(args.seed)

    root = Path(args.data_root)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    q_path = out_dir / "zonal_quality.csv"
    ms = P.load_marksheet(root)
    if args.limit:
        ms = ms[:args.limit]

    done = set()
    cols = None
    if args.resume and q_path.exists():
        done = set(pd.read_csv(q_path)["case_id"].tolist())
        print(f"[zones] resume: 已完成 {len(done)} 例")

    t0 = time.time()
    pending, errors = [], []
    for i, row in enumerate(ms):
        cid = P.case_id(row["patient_id"], row["study_id"])
        if cid in done:
            continue
        try:
            pending.append(audit_case(root, row["patient_id"], row["study_id"], row.get("center", ""), row.get("case_csPCa", "")))
        except Exception as e:  # noqa: BLE001
            errors.append({"case_id": cid, "error": f"{type(e).__name__}: {e}"})
        if len(pending) >= args.flush_every:
            flush(pending, q_path); pending = []
        if (i + 1) % 100 == 0:
            print(f"  scanned {i+1}/{len(ms)} elapsed={time.time()-t0:.1f}s pending_errors={len(errors)}")
    if pending:
        flush(pending, q_path)

    df = pd.read_csv(q_path)
    write_summaries(df, out_dir, errors)
    check_flagged(root, out_dir)
    print(f"[zones] wrote {q_path} rows={len(df)} elapsed={time.time()-t0:.1f}s errors={len(errors)}")


def flush(recs, q_path):
    df = pd.DataFrame(recs)
    header = not q_path.exists()
    df.to_csv(q_path, mode="a", index=False, header=header)


def write_summaries(df, out_dir, errors):
    # 异常病例排名：按异常信号数 + 低 Dice 排序
    flag_cols = [c for c in df.columns if c.endswith(("_empty", "_illegal", "_overlap_flag", "low_wg_zonal_dice"))]
    df["anomaly_score"] = df[flag_cols].apply(lambda r: sum(bool(x) for x in r), axis=1)
    ranked = df.sort_values(["anomaly_score", "hevi_pztz_wg_dice"], ascending=[False, True])
    ranked.to_csv(out_dir / "zonal_quality_ranked.csv", index=False)

    # 中心分层 summary
    agg_cols = ["wg_volume_ml", "yuan_pztz_volume_ml", "hevi_pztz_volume_ml",
                "yuan_pztz_wg_dice", "hevi_pztz_wg_dice", "pz_dice_yuan_hevi", "tz_dice_yuan_hevi",
                "yuan_pztz_outside_wg_ratio", "yuan_wg_uncovered_ratio"]
    rows = []
    for (center, cspca), g in df.groupby(["center", "case_csPCa"]):
        r = {"center": center, "case_csPCa": cspca, "n_cases": len(g),
             "n_wg_empty": int(g["wg_empty"].sum()), "n_yuan_empty": int(g["yuan_empty"].sum()),
             "n_hevi_empty": int(g["hevi_empty"].sum()),
             "n_yuan_illegal": int(g["yuan_illegal"].sum()), "n_hevi_illegal": int(g["hevi_illegal"].sum()),
             "n_pztz_overlap": int((g["yuan_pztz_overlap_flag"].astype(bool) | g["hevi_pztz_overlap_flag"].astype(bool)).sum()),
             "n_low_dice": int(g["low_wg_zonal_dice"].sum())}
        for c in agg_cols:
            if c in g:
                r[f"{c}_median"] = float(pd.to_numeric(g[c], errors="coerce").median())
        rows.append(r)
    pd.DataFrame(rows).to_csv(out_dir / "center_summary.csv", index=False)
    if errors:
        pd.DataFrame(errors).to_csv(out_dir / "zonal_build_errors.csv", index=False)
    print(f"[zones] center_summary + ranked written; errors={len(errors)}")


def check_flagged(root, out_dir):
    """检查官方提示的 Bosma22b 异常病例 11050_1001070。"""
    pid, sid = "11050", "1001070"
    p = P.label_path(root, P.WG_BOSMA22B, pid, sid)
    rep = {"case_id": CHECK_CASE, "wg_path": str(p), "wg_exists": p.exists()}
    if p.exists():
        try:
            t2w_geo = P.read_geometry(P.image_path(root, pid, sid, "t2w"))
            wg, was_res = load_label(p, t2w_geo)
            rep.update({"wg_unique": ",".join(map(str, sorted(int(x) for x in np.unique(wg).tolist()))),
                        "wg_empty": bool(not wg.any()), "wg_fg_voxels": int(wg.astype(bool).sum()),
                        "wg_volume_ml": float(wg.astype(bool).sum()) * float(np.prod(t2w_geo.spacing)) / 1000.0,
                        "resampled_to_t2w": was_res,
                        "t2w_size": ";".join(map(str, t2w_geo.size))})
        except Exception as e:  # noqa: BLE001
            rep["error"] = f"{type(e).__name__}: {e}"
    (out_dir / "flagged_case_11050_1001070.json").write_text(
        pd.Series(rep).to_json(force_ascii=False, indent=2))
    print(f"[zones] flagged case {CHECK_CASE}: {rep}")


if __name__ == "__main__":
    main()
