#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""全量读取 BPH / PCA 的 ROI.nii，明确 ROI 语义（病灶 vs 腺体）：
统计前景体素、前景体积(mL)、连通域数量、空ROI(阴性)比例，并核对模态dtype一致性。"""
import os, glob
import numpy as np
import SimpleITK as sitk
from scipy import ndimage

BASE = "/opt/data/private/lm/data/Prostate"
MODS = ["ADC", "DWI", "T2_not_fs", "ROI"]


def vol_mm3(spacing):
    return float(np.prod(spacing))


for name in ["BPH", "PCA"]:
    cases = sorted([c for c in glob.glob(os.path.join(BASE, name, "*")) if os.path.isdir(c)])
    fg_vox, fg_vol, n_cc, empty, dtype_mix = [], [], [], 0, {}
    dir_example = None
    for c in cases:
        # dtype 一致性核对（每模态）
        for m in MODS:
            p = os.path.join(c, m + ".nii")
            if not os.path.exists(p):
                continue
            r = sitk.ImageFileReader(); r.SetFileName(p); r.ReadImageInformation()
            dt = sitk.GetPixelIDValueAsString(r.GetPixelID())
            dtype_mix.setdefault(m, {}).setdefault(dt, 0)
            dtype_mix[m][dt] += 1
            if dir_example is None:
                dir_example = tuple(round(x, 2) for x in r.GetDirection())
        # ROI 体素分析
        rp = os.path.join(c, "ROI.nii")
        img = sitk.ReadImage(rp)
        arr = sitk.GetArrayFromImage(img)
        sp = img.GetSpacing()
        fg = int((arr > 0).sum())
        if fg == 0:
            empty += 1
        fg_vox.append(fg)
        fg_vol.append(fg * vol_mm3(sp) / 1000.0)  # mL
        lab, n = ndimage.label(arr > 0)
        n_cc.append(n)

    fg_vox = np.array(fg_vox); fg_vol = np.array(fg_vol); n_cc = np.array(n_cc)
    print("=" * 70)
    print(f"{name}: {len(cases)} 例  ROI 全量分析")
    print("=" * 70)
    print(f"direction 示例: {dir_example}")
    print(f"空ROI(前景=0)病例数: {empty}")
    print(f"前景体素数  min={fg_vox.min()} median={int(np.median(fg_vox))} "
          f"mean={fg_vox.mean():.0f} max={fg_vox.max()}")
    print(f"前景体积mL  min={fg_vol.min():.2f} median={np.median(fg_vol):.2f} "
          f"q75={np.quantile(fg_vol,.75):.2f} q90={np.quantile(fg_vol,.90):.2f} max={fg_vol.max():.2f}")
    print(f"连通域数    分布: {dict(zip(*[x.tolist() for x in np.unique(n_cc, return_counts=True)]))}")
    print(f"单连通域(=1)病例: {(n_cc==1).sum()}  多连通域(>1): {(n_cc>1).sum()}  空(=0): {(n_cc==0).sum()}")
    print(f"各模态 dtype 分布:")
    for m in MODS:
        print(f"    {m}: {dtype_mix.get(m, {})}")
    print()
