#!/usr/bin/env python
"""G0-5 物理空间 QC 图生成。

所有叠加均先将 ADC/HBV(线性)、lesion/WG/PZ/TZ(最近邻) 按物理坐标重采样到
T2W 参考空间，严禁直接按相同数组索引叠加不同网格的影像。

每类选例：各中心阳性/阴性/小病灶、多病灶、两套分区差异大、WG/几何异常；
另生成 index-space vs physical-space 对照图解释头信息异常。
图上标注 patient_id/study_id/center/spacing/标签来源/异常类型。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage

SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))
from zonal_reliability_fusion.data import picai as P  # noqa: E402

CENTERS = ["RUMC", "PCNN", "ZGT"]
PZ_COLOR = (0.0, 0.85, 0.0, 0.35)
TZ_COLOR = (0.1, 0.4, 1.0, 0.35)


def resample(path, t2w_geo, is_label):
    geo = P.read_geometry(path)
    if P.grids_equal(t2w_geo, geo):
        return P.load_array(path).astype(np.float32 if not is_label else np.int16)
    return P.resample_to_geometry(path, t2w_geo, is_label=is_label)


def fg_slice(arr):
    if arr is None or not np.any(arr):
        return arr.shape[0] // 2 if arr is not None else 0
    return int(np.argmax((arr > 0).sum(axis=(1, 2))))


def load_case(root, pid, sid):
    t2w_geo = P.read_geometry(P.image_path(root, pid, sid, "t2w"))
    d = {"t2w_geo": t2w_geo,
         "t2w": P.load_array(P.image_path(root, pid, sid, "t2w")).astype(np.float32),
         "adc": resample(P.image_path(root, pid, sid, "adc"), t2w_geo, False),
         "hbv": resample(P.image_path(root, pid, sid, "hbv"), t2w_geo, False)}
    src, cpath = P.canonical_lesion(root, pid, sid)
    d["lesion_src"] = src
    d["lesion"] = P.map_lesion_to_binary(resample(cpath, t2w_geo, True), src) if cpath else None
    d["wg"] = resample(P.label_path(root, P.WG_BOSMA22B, pid, sid), t2w_geo, True)
    d["yuan"] = resample(P.label_path(root, P.ZONAL_YUAN23, pid, sid), t2w_geo, True)
    d["hevi"] = resample(P.label_path(root, P.ZONAL_HEVIAI23, pid, sid), t2w_geo, True)
    return d


def overlay_zones(ax, base, pz, tz, title):
    ax.imshow(base, cmap="gray")
    ov = np.zeros(base.shape + (4,))
    ov[pz] = PZ_COLOR
    ov[tz] = TZ_COLOR
    ax.imshow(ov)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def overlay_contour(ax, base, mask, title, color="red"):
    ax.imshow(base, cmap="gray")
    if mask is not None and np.any(mask):
        ax.contour(mask.astype(float), levels=[0.5], colors=color, linewidths=1.0)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def case_figure(root, pid, sid, center, tag, anomaly, out_path):
    d = load_case(root, pid, sid)
    z = fg_slice(d["lesion"])
    sp = ";".join(f"{v:.2f}" for v in d["t2w_geo"].spacing)
    idtxt = f"{pid}_{sid} | {center} | sp=[{sp}] | lesion={d['lesion_src']}"
    if anomaly:
        idtxt += f"\nANOMALY: {anomaly}"

    fig, axes = plt.subplots(3, 3, figsize=(13, 12))
    t2 = d["t2w"][z]
    axes[0, 0].imshow(t2, cmap="gray"); axes[0, 0].set_title(idtxt, fontsize=8); axes[0, 0].axis("off")
    overlay_contour(axes[0, 1], t2, d["lesion"][z] if d["lesion"] is not None else None, "T2W + lesion", "red")
    overlay_contour(axes[0, 2], t2, d["wg"][z] > 0, "T2W + WG(Bosma22b)", "yellow")
    overlay_zones(axes[1, 0], t2, d["yuan"][z] == 1, d["yuan"][z] == 2, "T2W + Yuan23 (PZ=green,TZ=blue)")
    overlay_zones(axes[1, 1], t2, d["hevi"][z] == 1, d["hevi"][z] == 2, "T2W + HeviAI23 (PZ=green,TZ=blue)")
    dice_yh = P.dice_score(d["yuan"][z] == 1, d["hevi"][z] == 1)
    axes[1, 2].imshow(t2, cmap="gray"); axes[1, 2].set_title(f"PZ Dice(Yuan|Hevi)={dice_yh:.3f}", fontsize=8); axes[1, 2].axis("off")
    axes[2, 0].imshow(t2, cmap="gray"); axes[2, 0].set_title("T2W", fontsize=8); axes[2, 0].axis("off")
    axes[2, 1].imshow(d["adc"][z], cmap="gray"); axes[2, 1].set_title("ADC (resampled)", fontsize=8); axes[2, 1].axis("off")
    axes[2, 2].imshow(d["hbv"][z], cmap="gray"); axes[2, 2].set_title("HBV (resampled)", fontsize=8); axes[2, 2].axis("off")
    fig.suptitle(f"QC[{tag}] physical-space overlay (resampled to T2W)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=95, bbox_inches="tight")
    plt.close(fig)


def index_vs_physical(root, pid, sid, out_path):
    """对照：直接数组索引叠加(错误) vs 物理重采样叠加(正确)，用于解释头信息异常。"""
    t2w_geo = P.read_geometry(P.image_path(root, pid, sid, "t2w"))
    t2w = P.load_array(P.image_path(root, pid, sid, "t2w")).astype(np.float32)
    adc_native = P.load_array(P.image_path(root, pid, sid, "adc")).astype(np.float32)
    adc_phys = resample(P.image_path(root, pid, sid, "adc"), t2w_geo, False)
    z = t2w.shape[0] // 2
    zn = min(z, adc_native.shape[0] - 1)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    axes[0].imshow(t2w[z], cmap="gray"); axes[0].set_title(f"T2W z={z}\nsp={t2w_geo.spacing}", fontsize=8); axes[0].axis("off")
    axes[1].imshow(adc_native[zn], cmap="gray"); axes[1].set_title(f"ADC native index z={zn} (own grid)\n= index-space overlay", fontsize=8); axes[1].axis("off")
    axes[2].imshow(adc_phys[z], cmap="gray"); axes[2].set_title(f"ADC resampled to T2W z={z}\n= physical-space overlay (correct)", fontsize=8); axes[2].axis("off")
    fig.suptitle(f"index-space vs physical-space: {pid}_{sid} (grids differ)", fontsize=9)
    fig.tight_layout(); fig.savefig(out_path, dpi=95, bbox_inches="tight"); plt.close(fig)


def pick_multi_and_zonediff(root, manifest, rng):
    """抽样阳性例：选多病灶(连通域最多) 与 分区差异大(Yuan|Hevi PZ Dice 最低)。"""
    pos = manifest[manifest.case_csPCa == "YES"].sample(n=min(120, (manifest.case_csPCa == "YES").sum()), random_state=rng)
    best_multi, best_diff, n_cc, best_dice = None, None, -1, 2.0
    for _, r in pos.iterrows():
        try:
            d = load_case(root, r.patient_id, r.study_id)
            if d["lesion"] is not None and np.any(d["lesion"]):
                _, nc = ndimage.label(d["lesion"] > 0)
                if nc > n_cc:
                    n_cc, best_multi = nc, r
            dyh = P.dice_score(d["yuan"] == 1, d["hevi"] == 1)
            if dyh < best_dice:
                best_dice, best_diff = dyh, r
        except Exception:  # noqa: BLE001
            continue
    return best_multi, n_cc, best_diff, best_dice


def main():
    ap = argparse.ArgumentParser(description="生成 PI-CAI 物理空间 QC 图（G0-5）")
    ap.add_argument("--data-root", default=str(P.DEFAULT_DATA_ROOT))
    ap.add_argument("--manifest", default=str(Path(__file__).resolve().parents[2] / "data/metadata/picai_manifest.csv"))
    ap.add_argument("--geom-anom", default=str(Path(__file__).resolve().parents[2] / "outputs/diagnostics/picai_model_readiness/geometry_anomalies.csv"))
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parents[2] / "outputs/diagnostics/picai_model_readiness/qc_figures"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.data_root)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest)
    anom = pd.read_csv(args.geom_anom) if Path(args.geom_anom).exists() else pd.DataFrame(columns=["case_id", "grid_status", "anomaly_reason"])
    anom_by_case = {r.case_id: f"{r.role}:{r.grid_status}" for r in anom.itertuples()} if len(anom) else {}

    def anom_txt(cid):
        return anom_by_case.get(cid, "")

    selected = {}
    for c in CENTERS:
        sub = manifest[manifest.center == c]
        pos = sub[sub.case_csPCa == "YES"]
        neg = sub[sub.case_csPCa == "NO"]
        if len(pos):
            selected[f"{c}_positive"] = pos.iloc[0]
            small = pos.assign(fg=pd.to_numeric(pos["canonical_fg_voxels"], errors="coerce")).sort_values("fg")
            selected[f"{c}_smalllesion"] = small.iloc[0]
        if len(neg):
            selected[f"{c}_negative"] = neg.iloc[0]
    for cid in anom[anom.grid_status == "geometry_suspect"]["case_id"].unique():
        row = manifest[manifest.case_id == cid]
        if len(row):
            selected[f"anomaly_{cid}"] = row.iloc[0]

    print(f"[qc] 基础选例 {len(selected)} 例; 追加多病灶/分区差异选例...")
    bm, ncc, bd, bdice = pick_multi_and_zonediff(root, manifest, args.seed)
    if bm is not None:
        selected[f"multi_ncc{ncc}"] = bm
    if bd is not None:
        selected[f"zonediff_dice{bdice:.3f}"] = bd

    for tag, r in selected.items():
        cid = r.case_id
        out = out_dir / f"qc_{tag}_{cid}.png"
        try:
            case_figure(root, r.patient_id, r.study_id, r.center, tag, anom_txt(cid), out)
            print(f"[qc] wrote {out.name}")
        except Exception as e:  # noqa: BLE001
            print(f"[qc][ERROR] {tag} {cid}: {type(e).__name__}: {e}")

    # index vs physical 对照：取一个 adc 为 different/suspect 的病例
    diff_case = manifest[(manifest.adc_grid != "exact_same_grid")].iloc[0]
    ivp = out_dir / f"index_vs_physical_{diff_case.case_id}.png"
    index_vs_physical(root, diff_case.patient_id, diff_case.study_id, ivp)
    print(f"[qc] wrote {ivp.name}")
    print(f"[qc] done -> {out_dir}")


if __name__ == "__main__":
    main()
