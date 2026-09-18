#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""核对 BPH/PCA 同例内 4 模态是否处于同一物理网格：
逐项比较 size / spacing / origin / direction / physical extent，而非仅看数组尺寸。
只读头信息，速度快。"""
import os, glob, random
import numpy as np
import SimpleITK as sitk

random.seed(0)
BASE = "/opt/data/private/lm/data/Prostate"
MODS = ["ADC", "DWI", "T2_not_fs", "ROI"]
TOL = 1e-4


def geo(path):
    """返回 size/spacing/origin/direction 及基于 SimpleITK index-to-physical-point 的
    8 角点物理包围盒 extent（非 size*spacing 简化）。"""
    r = sitk.ImageFileReader(); r.SetFileName(path); r.ReadImageInformation()
    size = tuple(int(v) for v in r.GetSize())
    sp = np.array(r.GetSpacing(), dtype=float)
    org = np.array(r.GetOrigin(), dtype=float)
    dirn = np.array(r.GetDirection(), dtype=float)
    img = sitk.Image(list(size), sitk.sitkFloat32, 1)
    img.SetSpacing(list(sp)); img.SetOrigin(list(org)); img.SetDirection(list(dirn))
    sx, sy, sz = size
    corners = np.array([img.TransformIndexToPhysicalPoint((ix, iy, iz))
                        for ix in (0, sx - 1) for iy in (0, sy - 1) for iz in (0, sz - 1)])
    mn, mx = corners.min(axis=0), corners.max(axis=0)
    extent = mx - mn  # 角点物理包围盒 extent
    return np.array(size, dtype=float), sp, org, dirn, extent


for name in ["BPH", "PCA"]:
    cases = sorted([c for c in glob.glob(os.path.join(BASE, name, "*")) if os.path.isdir(c)])
    all_same = {"size": 0, "spacing": 0, "origin": 0, "direction": 0, "extent": 0}
    fully_same = 0
    examples = []
    for c in cases:
        gs = {}
        ok = True
        for m in MODS:
            p = os.path.join(c, m + ".nii")
            if not os.path.exists(p):
                ok = False; break
            gs[m] = geo(p)
        if not ok:
            continue
        ref = gs["ADC"]
        keys = ["size", "spacing", "origin", "direction", "extent"]
        idx = {k: i for i, k in enumerate(keys)}
        same = {}
        for k in keys:
            i = idx[k]
            same[k] = all(np.allclose(gs[m][i], ref[i], atol=TOL) for m in gs)
            if same[k]:
                all_same[k] += 1
        if all(same.values()):
            fully_same += 1
        if len(examples) < 3:
            examples.append((os.path.basename(c), ref))

    n = len(cases)
    print("=" * 70)
    print(f"{name}: {n} 例  同例内4模态几何一致性核对")
    print("=" * 70)
    for k in ["size", "spacing", "origin", "direction", "extent"]:
        print(f"  {k:10s} 一致的病例数: {all_same[k]}/{n}")
    print(f"  >>> 五项(size/spacing/origin/direction/extent)全部一致(=同一物理网格): {fully_same}/{n}")
    print("  示例(首个病例的几何参数):")
    for cid, ref in examples[:2]:
        size, sp, org, dirn, ext = ref
        print(f"    [{cid}] size={size.astype(int).tolist()} spacing={np.round(sp,4).tolist()}")
        print(f"            origin={np.round(org,3).tolist()} extent(mm)={np.round(ext,1).tolist()}")
        print(f"            direction={np.round(dirn,3).tolist()}")
    print()
