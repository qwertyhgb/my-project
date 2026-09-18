#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""配准 QC：1) 定位 BPH 中 origin 不一致的病例；2) 生成 T2/ADC+ROI 叠加图。
叠加分物理空间（ADC 先重采样到 ROI 网格）与 index 空间（仅数组空间，仅当
同 origin 时等价于物理对齐）；严禁只按相同数组索引叠加不同网格的影像。"""
import os, glob
import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/opt/data/private/lm/data/Prostate"
OUT = "/opt/data/private/lm/my-projects/docs/figures"
os.makedirs(OUT, exist_ok=True)
MODS = ["ADC", "DWI", "T2_not_fs", "ROI"]
TOL = 1e-4

# ---- 1) 找 BPH 中 origin 不一致的病例 ----
def geo(path):
    r = sitk.ImageFileReader(); r.SetFileName(path); r.ReadImageInformation()
    return (np.array(r.GetSize(), float), np.array(r.GetSpacing(), float),
            np.array(r.GetOrigin(), float), np.array(r.GetDirection(), float))

bph = sorted([c for c in glob.glob(os.path.join(BASE, "BPH", "*")) if os.path.isdir(c)])
anomalies = []
for c in bph:
    g = {m: geo(os.path.join(c, m + ".nii")) for m in MODS if os.path.exists(os.path.join(c, m + ".nii"))}
    ref = g["ADC"]
    for m in g:
        if not np.allclose(g[m][2], ref[2], atol=TOL):  # origin
            anomalies.append((os.path.basename(c), m, np.round(g[m][2] - ref[2], 3).tolist()))
print("BPH origin 不一致的病例:")
for a in anomalies:
    print("   ", a)

# ---- 2) 找 PCA 中 ROI 体积最大/中位 的病例 ----
pca = sorted([c for c in glob.glob(os.path.join(BASE, "PCA", "*")) if os.path.isdir(c)])
vols = []
for c in pca:
    img = sitk.ReadImage(os.path.join(c, "ROI.nii"))
    arr = sitk.GetArrayFromImage(img)
    vols.append(((arr > 0).sum() * float(np.prod(img.GetSpacing())) / 1000.0, os.path.basename(c)))
vols.sort()
pca_max = vols[-1][1]; pca_med = vols[len(vols)//2][1]
print(f"\nPCA ROI 体积最大例: {pca_max} = {vols[-1][0]:.1f} mL")
print(f"PCA ROI 体积中位例: {pca_med} = {vols[len(vols)//2][0]:.1f} mL")

# BPH 中位体积例
bph_vols = []
for c in bph:
    img = sitk.ReadImage(os.path.join(c, "ROI.nii"))
    arr = sitk.GetArrayFromImage(img)
    bph_vols.append(((arr > 0).sum() * float(np.prod(img.GetSpacing())) / 1000.0, os.path.basename(c)))
bph_vols.sort()
bph_med = bph_vols[len(bph_vols)//2][1]
bph_anom = anomalies[0][0] if anomalies else bph_vols[0][1]
print(f"BPH ROI 体积中位例: {bph_med} = {bph_vols[len(bph_vols)//2][0]:.1f} mL")
print(f"BPH 叠加图将含异常例: {bph_anom}")

# ---- 3) 叠加可视化：物理空间重采样 vs index 空间对照 ----
def resample_to(src_img, ref_img, is_label):
    f = sitk.ResampleImageFilter()
    f.SetReferenceImage(ref_img)
    f.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
    f.SetDefaultPixelValue(0)
    return sitk.GetArrayFromImage(f.Execute(src_img))

selected = [("BPH", bph_med), ("BPH", bph_anom), ("PCA", pca_med), ("PCA", pca_max)]
fig, axes = plt.subplots(len(selected), 3, figsize=(13, 4 * len(selected)))
for i, (ds, cid) in enumerate(selected):
    d = os.path.join(BASE, ds, cid)
    t2_img = sitk.ReadImage(os.path.join(d, "T2_not_fs.nii"))
    adc_img = sitk.ReadImage(os.path.join(d, "ADC.nii"))
    roi_img = sitk.ReadImage(os.path.join(d, "ROI.nii"))
    t2 = sitk.GetArrayFromImage(t2_img).astype(float)
    adc_phys = resample_to(adc_img, roi_img, False).astype(float)  # ADC 重采样到 ROI(T2) 网格
    adc_index = sitk.GetArrayFromImage(adc_img).astype(float)       # ADC 原生网格(index)
    roi = sitk.GetArrayFromImage(roi_img)
    same_origin = np.allclose(adc_img.GetOrigin(), roi_img.GetOrigin(), atol=TOL)
    z = int(np.argmax(roi.sum(axis=(1, 2))))
    zn = min(z, adc_index.shape[0] - 1)
    vol = (roi > 0).sum() * float(np.prod(roi_img.GetSpacing())) / 1000.0
    axes[i][0].imshow(t2[z], cmap="gray")
    axes[i][0].contour(roi[z], levels=[0.5], colors="red", linewidths=1.2)
    axes[i][0].set_title(f"{ds} {cid} | T2+ROI | z={z} ROI={vol:.1f}mL", fontsize=8)
    axes[i][1].imshow(adc_index[zn], cmap="gray")
    axes[i][1].contour(roi[z], levels=[0.5], colors="red", linewidths=1.2)
    axes[i][1].set_title(f"ADC+ROI INDEX-space | same_origin={same_origin}\n(仅数组空间，非同网格时非物理)", fontsize=8)
    axes[i][2].imshow(adc_phys[z], cmap="gray")
    axes[i][2].contour(roi[z], levels=[0.5], colors="red", linewidths=1.2)
    axes[i][2].set_title("ADC+ROI PHYSICAL-space (resampled)", fontsize=8)
    for j in range(3):
        axes[i][j].axis("off")
plt.tight_layout()
outp = os.path.join(OUT, "registration_overlay_qc.png")
plt.savefig(outp, dpi=100, bbox_inches="tight")
print(f"\n叠加图已保存(含物理空间重采样列): {outp}")
