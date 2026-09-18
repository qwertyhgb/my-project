#!/usr/bin/env python
"""P0B 物化后 QC 图：检查重采样后多序列空间一致性（针对 geometry_suspect 等病例）。

读取 data/processed/picai/cases/<case_id>/ 下已物化的
T2W/ADC/HBV/lesion/WG/Yuan23/HeviAI23（不触碰原始数据），生成物理空间叠加图：
- T2W 底图 + lesion/WG 轮廓；
- T2W + Yuan23 / HeviAI23 分区色块（PZ=绿，TZ=蓝）；
- ADC / HBV 独立显示 + T2W|ADC、T2W|HBV 棋盘格（用于目检重采样后对齐）；
并输出每例 ADC/HBV 零值体素占比（视野覆盖度）与结构化 summary。

用法：
    python scripts/audit/generate_picai_materialized_qc.py
    python scripts/audit/generate_picai_materialized_qc.py --cases 10057_1000057
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))
from zonal_reliability_fusion.data import picai as P  # noqa: E402

PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_ROOT = PROJECT / "data/processed/picai"
DEFAULT_QC_DIR = PROJECT / "outputs/diagnostics/picai_model_readiness/qc_figures_post_p0b"
DEFAULT_SUSPECTS = "10057_1000057,10161_1000164,10489_1000497,10805_1000821,11414_1001438"
PZ_COLOR = (0.0, 0.85, 0.0, 0.35)
TZ_COLOR = (0.1, 0.4, 1.0, 0.35)


def load_case_arrays(case_dir: Path):
    d = {}
    for key, fname in (("t2w", "t2w.nii.gz"), ("adc", "adc.nii.gz"), ("hbv", "hbv.nii.gz"),
                       ("lesion", "lesion.nii.gz"), ("yuan", "zonal_yuan.nii.gz"),
                       ("hevi", "zonal_hevi.nii.gz"), ("wg", "wg.nii.gz")):
        p = case_dir / fname
        d[key] = P.load_array(p) if p.exists() else None
    return d


def norm2d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    lo, hi = np.percentile(a, [1, 99])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((a - lo) / (hi - lo), 0, 1)


def checkerboard(a: np.ndarray, b: np.ndarray, block: int = 8) -> np.ndarray:
    h, w = a.shape
    yy, xx = np.indices((h, w))
    m = (((yy // block) + (xx // block)) % 2 == 0)
    out = norm2d(a).copy()
    out[m] = norm2d(b)[m]
    return out


def pick_slice(wg, lesion, nz: int) -> int:
    # 阳性病例优先显示病灶层；否则取 WG 层；均无前景则取中间层
    for arr in (lesion, wg):
        if arr is not None and arr.any():
            return int(np.argmax((arr > 0).sum(axis=(1, 2))))
    return nz // 2


def overlay_contour(ax, base, mask, title, color="red"):
    ax.imshow(base, cmap="gray")
    if mask is not None and np.any(mask):
        ax.contour(mask.astype(float), levels=[0.5], colors=color, linewidths=1.0)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def overlay_zones(ax, base, pz, tz, title):
    ax.imshow(base, cmap="gray")
    ov = np.zeros(base.shape + (4,))
    if pz is not None:
        ov[pz] = PZ_COLOR
    if tz is not None:
        ov[tz] = TZ_COLOR
    ax.imshow(ov)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def main() -> None:
    ap = argparse.ArgumentParser(description="P0B 物化后 QC 图（多序列空间一致性）")
    ap.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--manifest", default=str(PROJECT / "data/metadata/picai_manifest.csv"))
    ap.add_argument("--cases", default=DEFAULT_SUSPECTS, help="逗号分隔 case_id")
    ap.add_argument("--out-dir", default=str(DEFAULT_QC_DIR))
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.manifest).set_index("case_id")
    cids = [s.strip() for s in args.cases.split(",") if s.strip()]

    records = []
    for cid in tqdm(cids, desc="qc", disable=args.no_progress, file=sys.stdout):
        case_dir = out_root / "cases" / cid
        d = load_case_arrays(case_dir)
        nz = d["t2w"].shape[0]
        z = pick_slice(d["wg"], d["lesion"], nz)
        t2 = d["t2w"][z]
        row = df.loc[cid] if cid in df.index else None
        meta = (f" | {row['center']} | {row['case_csPCa']} | adc:{row['adc_grid'][:18]}"
                f" | hbv:{row['hbv_grid'][:18]} | lesion:{row['canonical_grid'][:22]}") if row is not None else ""
        sp = ";".join(f"{v:.2f}" for v in P.read_geometry(case_dir / "t2w.nii.gz").spacing)
        idtxt = f"{cid}{meta}\nT2W spacing=[{sp}] z={z}/{nz}"

        fig, axes = plt.subplots(3, 3, figsize=(13, 12))
        axes[0, 0].imshow(t2, cmap="gray"); axes[0, 0].set_title(idtxt, fontsize=7); axes[0, 0].axis("off")
        overlay_contour(axes[0, 1], t2, d["lesion"] [z] if d["lesion"] is not None else None, "T2W + lesion (resampled)", "red")
        overlay_contour(axes[0, 2], t2, (d["wg"][z] > 0) if d["wg"] is not None else None, "T2W + WG", "yellow")
        overlay_zones(axes[1, 0], t2, (d["yuan"][z] == 1) if d["yuan"] is not None else None,
                      (d["yuan"][z] == 2) if d["yuan"] is not None else None, "T2W + Yuan23 (PZ=green,TZ=blue)")
        overlay_zones(axes[1, 1], t2, (d["hevi"][z] == 1) if d["hevi"] is not None else None,
                      (d["hevi"][z] == 2) if d["hevi"] is not None else None, "T2W + HeviAI23 (PZ=green,TZ=blue)")
        axes[1, 2].imshow(checkerboard(t2, d["adc"][z]), cmap="gray")
        axes[1, 2].set_title("checkerboard T2W|ADC (block=8)", fontsize=8); axes[1, 2].axis("off")
        axes[2, 0].imshow(d["adc"][z], cmap="gray"); axes[2, 0].set_title("ADC (resampled to T2W grid)", fontsize=8); axes[2, 0].axis("off")
        axes[2, 1].imshow(d["hbv"][z], cmap="gray"); axes[2, 1].set_title("HBV (resampled to T2W grid)", fontsize=8); axes[2, 1].axis("off")
        axes[2, 2].imshow(checkerboard(t2, d["hbv"][z]), cmap="gray")
        axes[2, 2].set_title("checkerboard T2W|HBV (block=8)", fontsize=8); axes[2, 2].axis("off")
        fig.suptitle(f"P0B post-materialization QC (physical-space, T2W grid) — {cid}", fontsize=10)
        fig.tight_layout()
        out_png = out_dir / f"qc_post_{cid}.png"
        fig.savefig(out_png, dpi=95, bbox_inches="tight")
        plt.close(fig)

        rec = {"case_id": cid, "png": str(out_png), "slice": z,
               "t2w_shape": list(map(int, d["t2w"].shape)),
               "adc_zero_frac": float((d["adc"] == 0).mean()),
               "hbv_zero_frac": float((d["hbv"] == 0).mean()),
               "lesion_fg_vox": int((d["lesion"] > 0).sum()) if d["lesion"] is not None else -1,
               "has_wg": d["wg"] is not None}
        records.append(rec)
        print(f"[qc] wrote {out_png.name} adc_zero={rec['adc_zero_frac']:.3f} hbv_zero={rec['hbv_zero_frac']:.3f}")

    summary_path = out_dir / "qc_post_materialization_summary.json"
    merged = {}
    if summary_path.exists():
        for c in json.loads(summary_path.read_text(encoding="utf-8")).get("cases", []):
            merged[c["case_id"]] = c
    for c in records:
        merged[c["case_id"]] = c
    summary = {"cases": [merged[k] for k in sorted(merged)], "n_cases": len(merged), "out_dir": str(out_dir)}
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[qc] done: n={len(records)} (累积 {len(merged)}) -> {out_dir}")


if __name__ == "__main__":
    main()
