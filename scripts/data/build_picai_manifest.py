#!/usr/bin/env python
"""G0-1 建立统一 PI-CAI manifest：一行一个 study（1500）。

包含：身份/临床域、T2W/ADC/HBV 与各标签路径与存在性、canonical 病灶标签来源、
每个文件的 size/spacing/origin/extent/orientation、canonical 标签唯一值与是否空、
相对 T2W 的网格状态与异常原因。同时对照预期数字做验证并输出 validation JSON。
原始数据只读；输出写入 data/metadata/。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))

from zonal_reliability_fusion.data import picai as P  # noqa: E402

EXPECTED = {
    "n_studies": 1500, "n_patients": 1476, "n_negative": 1075, "n_positive": 425,
    "n_expert_pos_original": 220, "n_expert_pos_pooch25": 205,
    "centers": {"RUMC": {"studies": 800, "pos": 236},
                "PCNN": {"studies": 350, "pos": 109},
                "ZGT": {"studies": 350, "pos": 80}},
}

SEQS = P.SEQUENCES
GEO_COLS = ["exists", "size", "spacing", "origin", "extent", "orient", "grid"]


def fmt3(v):
    return ";".join(str(int(x)) for x in v) if v is not None else ""


def fmtf(v):
    return ";".join(f"{x:.3f}" for x in v) if v is not None else ""


def read_geo_safe(path):
    try:
        return P.read_geometry(path), ""
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


def label_values_safe(path):
    """读取标签体素的唯一值与是否空；失败返回 None 与原因。"""
    try:
        arr = P.load_array(path)
        uv = np.unique(arr)
        return [int(x) for x in uv.tolist()], int(np.count_nonzero(arr)), ""
    except Exception as e:  # noqa: BLE001
        return None, None, f"{type(e).__name__}: {e}"


def build_row(root, row):
    pid, sid = row["patient_id"], row["study_id"]
    cid = P.case_id(pid, sid)
    r = {"patient_id": pid, "study_id": sid, "case_id": cid,
         "center": row.get("center", ""), "case_csPCa": row.get("case_csPCa", ""),
         "case_ISUP": row.get("case_ISUP", ""), "patient_age": row.get("patient_age", ""),
         "psa": row.get("psa", ""), "psad": row.get("psad", ""),
         "prostate_volume": row.get("prostate_volume", ""),
         "histopath_type": row.get("histopath_type", ""), "mri_date": row.get("mri_date", ""),
         "lesion_ISUP": row.get("lesion_ISUP", ""), "lesion_GS": row.get("lesion_GS", "")}

    # 影像三序列几何
    geos = {}
    for s in SEQS:
        p = P.image_path(root, pid, sid, s)
        geo, err = read_geo_safe(p)
        geos[s] = geo
        r[f"{s}_path"] = str(p)
        r[f"{s}_exists"] = bool(geo is not None)
        if geo:
            r[f"{s}_size"] = fmt3(geo.size)
            r[f"{s}_spacing"] = fmtf(geo.spacing)
            r[f"{s}_origin"] = fmtf(geo.origin)
            r[f"{s}_extent"] = fmtf(P.physical_extent(geo))
            r[f"{s}_orient"] = P.orientation_code(geo)
        else:
            for c in ["size", "spacing", "origin", "extent", "orient"]:
                r[f"{s}_{c}"] = ""
        if err:
            r[f"{s}_err"] = err

    ref = geos["t2w"]
    # 影像序列真实网格分类（相对 T2W），不硬编码
    for s in SEQS:
        g = geos[s]
        r[f"{s}_grid"] = (P.GRID_MISSING if g is None else
                          (P.GRID_EXACT if (s == "t2w" or ref is None)
                           else P.classify_grid(ref, g)))
    # canonical lesion
    src_name, cpath = P.canonical_lesion(root, pid, sid)
    r["lesion_resampled_path"] = str(P.label_path(root, P.LESION_RESAMPLED, pid, sid))
    r["lesion_resampled_exists"] = P.label_path(root, P.LESION_RESAMPLED, pid, sid).exists()
    r["lesion_pooch25_path"] = str(P.label_path(root, P.LESION_POOCH25, pid, sid))
    r["lesion_pooch25_exists"] = P.label_path(root, P.LESION_POOCH25, pid, sid).exists()
    r["canonical_source"] = src_name
    r["canonical_path"] = str(cpath) if cpath else ""
    r["canonical_exists"] = bool(cpath and cpath.exists())
    vals, fg, lerr = (None, None, "missing")
    if cpath and cpath.exists():
        vals, fg, lerr = label_values_safe(cpath)
    r["canonical_label_values"] = ",".join(map(str, vals)) if vals is not None else ""
    r["canonical_is_empty"] = (fg == 0) if fg is not None else ""
    r["canonical_fg_voxels"] = fg if fg is not None else ""

    # canonical lesion 网格状态（相对 T2W）
    cgeo, cerr = read_geo_safe(cpath) if (cpath and cpath.exists()) else (None, "missing")
    r["canonical_grid"] = P.classify_grid(ref, cgeo) if (ref and cgeo) else P.GRID_MISSING

    # WG / zonal 标签：路径 + 存在 + 几何 + 网格状态
    label_sets = {
        "wg_bosma22b": P.WG_BOSMA22B, "zonal_yuan23": P.ZONAL_YUAN23,
        "zonal_heviai23": P.ZONAL_HEVIAI23, "wg_guerbet23": P.WG_GUERBET23,
    }
    for key, sub in label_sets.items():
        p = P.label_path(root, sub, pid, sid)
        geo, _ = read_geo_safe(p)
        r[f"{key}_path"] = str(p)
        r[f"{key}_exists"] = bool(geo is not None)
        if geo:
            r[f"{key}_size"] = fmt3(geo.size)
            r[f"{key}_spacing"] = fmtf(geo.spacing)
            r[f"{key}_orient"] = P.orientation_code(geo)
            r[f"{key}_grid"] = P.classify_grid(ref, geo) if ref else P.GRID_MISSING
        else:
            for c in ["size", "spacing", "orient"]:
                r[f"{key}_{c}"] = ""
            r[f"{key}_grid"] = P.GRID_MISSING

    # 汇总
    r["n_sequences_present"] = sum(1 for s in SEQS if geos[s] is not None)
    r["all_sequences_present"] = r["n_sequences_present"] == len(SEQS)
    grids = [r[f"{s}_grid"] for s in SEQS] + [r["canonical_grid"],
             r["wg_bosma22b_grid"], r["zonal_yuan23_grid"], r["zonal_heviai23_grid"]]
    order = [P.GRID_EXACT, P.GRID_DIFFERENT, P.GRID_PARTIAL, P.GRID_SUSPECT, P.GRID_MISSING]
    r["geometry_status"] = max(grids, key=lambda g: order.index(g) if g in order else 99)
    r["anomaly_reason"] = {
        P.GRID_EXACT: "", P.GRID_DIFFERENT: "物理空间对应但采样网格不同(需重采样)",
        P.GRID_PARTIAL: "与T2W仅部分物理重叠", P.GRID_SUSPECT: "几何可疑(方向或覆盖范围异常)",
        P.GRID_MISSING: "关键文件缺失或不可读",
    }[r["geometry_status"]]
    if not r["canonical_exists"]:
        r["anomaly_reason"] = (r["anomaly_reason"] + "; " if r["anomaly_reason"] else "") + "canonical病灶标签缺失"
    return r


def validate(df):
    import pandas as pd
    rep = {}
    rep["n_studies"] = len(df)
    rep["n_patients"] = df["patient_id"].nunique()
    rep["n_negative"] = int((df["case_csPCa"] == "NO").sum())
    rep["n_positive"] = int((df["case_csPCa"] == "YES").sum())
    rep["n_expert_pos_original"] = int(((df["canonical_source"] == P.SRC_RESAMPLED) & (~df["canonical_is_empty"].astype(str).isin(["True"]))).sum())
    rep["n_expert_pos_pooch25"] = int((df["canonical_source"] == P.SRC_POOCH25).sum())
    rep["centers"] = {}
    for c, g in df.groupby("center"):
        rep["centers"][c] = {"studies": len(g), "pos": int((g["case_csPCa"] == "YES").sum())}
    rep["checks"] = {}
    for k in ["n_studies", "n_patients", "n_negative", "n_positive", "n_expert_pos_original", "n_expert_pos_pooch25"]:
        rep["checks"][k] = {"expected": EXPECTED[k], "actual": rep[k], "pass": rep[k] == EXPECTED[k]}
    for c, exp in EXPECTED["centers"].items():
        act = rep["centers"].get(c, {"studies": 0, "pos": 0})
        rep["checks"][f"center_{c}"] = {"expected": exp, "actual": act,
                                        "pass": act["studies"] == exp["studies"] and act["pos"] == exp["pos"]}
    rep["all_pass"] = all(v["pass"] for v in rep["checks"].values())
    return rep


def main():
    ap = argparse.ArgumentParser(description="建立统一 PI-CAI manifest（G0-1）")
    ap.add_argument("--data-root", default=str(P.DEFAULT_DATA_ROOT))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[2] / "data/metadata/picai_manifest.csv"))
    ap.add_argument("--validation-out", default=str(Path(__file__).resolve().parents[2] / "data/metadata/picai_manifest_validation.json"))
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 个 study（smoke test）")
    ap.add_argument("--no-progress", action="store_true", help="关闭进度条（日志环境）")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    np.random.seed(args.seed)

    import pandas as pd
    root = Path(args.data_root)
    rows = P.load_marksheet(root)
    if args.limit:
        rows = rows[:args.limit]
    print(f"[manifest] marksheet rows={len(rows)} data_root={root}")

    t0 = time.time()
    recs, errors = [], []
    for row in tqdm(rows, desc="manifest", mininterval=1.0, disable=args.no_progress, file=sys.stdout):
        try:
            recs.append(build_row(root, row))
        except Exception as e:  # noqa: BLE001
            errors.append({"case_id": P.case_id(row["patient_id"], row["study_id"]),
                           "error": f"{type(e).__name__}: {e}"})

    df = pd.DataFrame(recs)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    rep = validate(df)
    rep["build_errors"] = errors
    rep["n_build_errors"] = len(errors)
    rep["elapsed_sec"] = round(time.time() - t0, 1)
    vout = Path(args.validation_out)
    vout.parent.mkdir(parents=True, exist_ok=True)
    vout.write_text(json.dumps(rep, indent=2, ensure_ascii=False))

    print(f"[manifest] wrote {out} rows={len(df)} elapsed={rep['elapsed_sec']}s build_errors={len(errors)}")
    print("[validate] " + json.dumps({k: v for k, v in rep["checks"].items()}, ensure_ascii=False))
    print(f"[validate] all_pass={rep['all_pass']}  (validation json: {vout})")
    print(f"[manifest] done: ok={len(df)} failed={len(errors)} skipped=0 elapsed={rep['elapsed_sec']}s out={out} validation={vout}")


if __name__ == "__main__":
    main()
