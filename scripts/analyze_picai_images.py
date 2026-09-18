#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PI-CAI 影像(.mha)与多层级标签仓库的属性和语义分析(抽样)。"""
import os, glob, random
import numpy as np
import SimpleITK as sitk

random.seed(0)
BASE = "/opt/data/private/lm/data/Prostate/PI-CAI"
IMG = os.path.join(BASE, "images")
LAB = os.path.join(BASE, "labels")


def head(path):
    r = sitk.ImageFileReader(); r.SetFileName(path); r.ReadImageInformation()
    return r.GetSize(), tuple(round(s, 3) for s in r.GetSpacing()), sitk.GetPixelIDValueAsString(r.GetPixelID())


def uniq_nz(path):
    a = sitk.GetArrayFromImage(sitk.ReadImage(path))
    u, c = np.unique(a, return_counts=True)
    d = dict(zip(u.tolist(), c.tolist()))
    fg = int(sum(v for k, v in d.items() if k != 0))
    return d, fg


# 收集所有 study 的 t2w 文件 -> study 前缀
t2ws = sorted(glob.glob(os.path.join(IMG, "*", "*_t2w.mha")))
prefixes = [os.path.basename(p).replace("_t2w.mha", "") for p in t2ws]
print("=" * 70)
print(f"PI-CAI 影像: 患者目录={len(glob.glob(os.path.join(IMG,'*')))} 含t2w的study={len(t2ws)}")
print("=" * 70)

# 影像模态头信息抽样
print("\n--- 影像模态 size/spacing/dtype (抽样8个study) ---")
sample_pf = random.sample(prefixes, 8)
for pf in sample_pf:
    pd = pf.split("_")[0]
    line = f"  [{pf}] "
    for mod in ["t2w", "adc", "hbv", "cor", "sag"]:
        p = os.path.join(IMG, pd, f"{pf}_{mod}.mha")
        if os.path.exists(p):
            s, sp, dt = head(p)
            line += f"{mod}:size={s},sp={sp},{dt.split()[0]}bit "
    print(line)

# 各标签来源唯一值
def sample_labels(sub, n=8, ext=".nii.gz"):
    files = sorted(glob.glob(os.path.join(LAB, sub, f"*{ext}")))
    if not files:
        print(f"  {sub}: 无文件"); return
    allvals = {}
    pos = 0
    samp = random.sample(files, min(n, len(files)))
    for f in samp:
        d, fg = uniq_nz(f)
        for k in d:
            allvals[k] = allvals.get(k, 0) + 1
        if fg > 0:
            pos += 1
    print(f"  {sub}: 共{len(files)}文件; 抽样{len(samp)}; 出现标签值={sorted(allvals.keys())}; 抽样中阳性(前景>0)={pos}/{len(samp)}")

print("\n--- csPCa 病灶标签 (lesion) ---")
sample_labels("csPCa_lesion_delineations/human_expert/original", 12)
sample_labels("csPCa_lesion_delineations/human_expert/resampled", 6)
sample_labels("csPCa_lesion_delineations/human_expert/Pooch25", 6)
sample_labels("csPCa_lesion_delineations/AI/Bosma22a", 6)
print("\n--- 解剖标签 (anatomical) ---")
sample_labels("anatomical_delineations/whole_gland/AI/Bosma22b", 6)
sample_labels("anatomical_delineations/zonal_pz_tz/AI/Yuan23", 4)
sample_labels("anatomical_delineations/zonal_pz_tz/AI/HeviAI23", 4)

# human_expert/original 阳性率全量估计(读较多)
print("\n--- human_expert/original 阳性率(抽样60) ---")
files = sorted(glob.glob(os.path.join(LAB, "csPCa_lesion_delineations/human_expert/original/*.nii.gz")))
samp = random.sample(files, min(60, len(files)))
pos, valcnt = 0, {}
for f in samp:
    d, fg = uniq_nz(f)
    if fg > 0:
        pos += 1
    for k in d:
        if k != 0:
            valcnt[k] = valcnt.get(k, 0) + 1
print(f"  阳性={pos}/{len(samp)}; 各非零标签值出现的文件数={dict(sorted(valcnt.items()))}")

# additional_resources
print("\n--- additional_resources ---")
print("  ", os.listdir(os.path.join(LAB, "additional_resources")))
