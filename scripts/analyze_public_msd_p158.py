#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""公共数据集 MSD_Prostate 与 Prostate158 的影像属性与标签语义分析。"""
import os, glob, random
import numpy as np
import SimpleITK as sitk

random.seed(0)
BASE = "/opt/data/private/lm/data/Prostate"


def head(path):
    r = sitk.ImageFileReader(); r.SetFileName(path); r.ReadImageInformation()
    return r.GetSize(), tuple(round(s, 3) for s in r.GetSpacing()), sitk.GetPixelIDValueAsString(r.GetPixelID())


def uniq(path):
    a = sitk.GetArrayFromImage(sitk.ReadImage(path))
    u, c = np.unique(a, return_counts=True)
    return dict(zip(u.tolist(), c.tolist())), a


# ============================ MSD_Prostate ============================
print("=" * 70)
print("MSD_Prostate (Task05_Prostate)")
print("=" * 70)
msd = os.path.join(BASE, "MSD_Prostate/Task05_Prostate")
imgtr = sorted(glob.glob(os.path.join(msd, "imagesTr/*.nii.gz")))
labtr = sorted(glob.glob(os.path.join(msd, "labelsTr/*.nii.gz")))
imgts = sorted(glob.glob(os.path.join(msd, "imagesTs/*.nii.gz")))
print(f"imagesTr={len(imgtr)} labelsTr={len(labtr)} imagesTs={len(imgts)}")
print("imagesTr 尺寸/spacing/dtype (全量, 前若干):")
sizes = {}
for p in imgtr + imgts:
    s, sp, dt = head(p)
    sizes.setdefault((len(s), dt), []).append((s, sp))
for k, v in sizes.items():
    print(f"  维度数={k[0]} dtype={k[1]}: {len(v)} 例; 示例 size={v[0][0]} spacing={v[0][1]}")
# 抽样一个 image 看 4D 通道 & 强度
p0 = imgtr[0]
img = sitk.ReadImage(p0)
print(f"\n样例影像 {os.path.basename(p0)}: size={img.GetSize()} (最后维=通道数) spacing={img.GetSpacing()}")
arr = sitk.GetArrayFromImage(img)  # (C, D, H, W)
print(f"  array shape={arr.shape} -> 通道数={arr.shape[0]} (0:T2, 1:ADC)")
for ch in range(arr.shape[0]):
    a = arr[ch].astype(np.float32)
    print(f"  通道{ch}: min={a.min():.0f} max={a.max():.0f} mean={a.mean():.0f}")
# 标签唯一值
print("labelsTr 标签唯一值分布(抽样5):")
for p in random.sample(labtr, min(5, len(labtr))):
    d, a = uniq(p)
    print(f"  {os.path.basename(p)}: unique(count)={d}")

# ============================ Prostate158 ============================
print("\n" + "=" * 70)
print("Prostate158")
print("=" * 70)
p158 = os.path.join(BASE, "Prostate158")
tr_cases = sorted(glob.glob(os.path.join(p158, "train/prostate158_train/train/*")))
te_cases = sorted(glob.glob(os.path.join(p158, "test/prostate158_test/test/*")))
print(f"train 病例目录={len(tr_cases)}  test 病例目录={len(te_cases)}")
# 每例文件类型统计
filetypes = {}
for c in tr_cases + te_cases:
    for f in os.listdir(c):
        if f.endswith(".nii.gz"):
            key = f[:-7]
            filetypes[key] = filetypes.get(key, 0) + 1
print("各类型标注文件出现次数:")
for k, v in sorted(filetypes.items()):
    print(f"  {k}: {v}")
# 影像模态 spacing/size 抽样
print("\n影像模态 t2/adc/dwi 头信息(抽样6例):")
for c in random.sample(tr_cases, 6):
    line = f"  [{os.path.basename(c)}] "
    for m in ["t2", "adc", "dwi"]:
        p = os.path.join(c, m + ".nii.gz")
        if os.path.exists(p):
            s, sp, dt = head(p)
            line += f"{m}:size={s},sp_z={sp[2]},dt={dt.split()[0]} "
    print(line)
# 标签语义
print("\n标注唯一值(抽样):")
for tag in ["t2_anatomy_reader1", "t2_tumor_reader1", "adc_tumor_reader1", "adc_tumor_reader2"]:
    files = glob.glob(os.path.join(p158, f"**/{tag}.nii.gz"), recursive=True)
    files = [f for f in files if "empty" not in os.path.basename(f)]
    if not files:
        print(f"  {tag}: 无文件"); continue
    d, a = uniq(random.choice(files))
    nz = {k: v for k, v in d.items() if k != 0}
    print(f"  {tag}: unique(count)={d} (共{len(files)}个此类文件)")
