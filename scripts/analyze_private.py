#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""私有数据集 BPH / PCA 影像属性与 ROI 标签语义分析。
- 全量读取头信息(size/spacing/dtype/direction)，只读header不读体素，速度快。
- 抽样读取体素，统计 ADC/DWI/T2 强度范围 与 ROI 标签唯一值分布。
"""
import os, glob, random
import numpy as np
import SimpleITK as sitk

random.seed(0)
BASE = "/opt/data/private/lm/data/Prostate"
MODS = ["ADC", "DWI", "T2_not_fs", "ROI"]


def header_only(path):
    r = sitk.ImageFileReader()
    r.SetFileName(path)
    r.ReadImageInformation()
    return {
        "size": r.GetSize(),
        "spacing": tuple(round(s, 4) for s in r.GetSpacing()),
        "dtype": sitk.GetPixelIDValueAsString(r.GetPixelID()),
        "dim": r.GetDimension(),
        "direction": r.GetDirection(),
    }


def is_identity(dir_tuple, tol=1e-3):
    d = np.array(dir_tuple).reshape(3, 3)
    return bool(np.allclose(d, np.eye(3), atol=tol))


for name in ["BPH", "PCA"]:
    cases = sorted(glob.glob(os.path.join(BASE, name, "*")))
    cases = [c for c in cases if os.path.isdir(c)]
    print("=" * 70)
    print(f"私有数据集 {name}: 共 {len(cases)} 例")
    print("=" * 70)

    # --- 全量头信息 ---
    shapes, spacings, dtypes, dirs_ok = {}, {}, {}, 0
    per_case_consistent = 0
    n_head = 0
    for c in cases:
        hs = {}
        for m in MODS:
            p = os.path.join(c, m + ".nii")
            if not os.path.exists(p):
                continue
            h = header_only(p)
            hs[m] = h
            shapes.setdefault(m, {}).setdefault(h["size"], 0)
            shapes[m][h["size"]] += 1
            spacings.setdefault(m, []).append(h["spacing"])
            dtypes.setdefault(m, {}).setdefault(h["dtype"], 0)
            dtypes[m][h["dtype"]] += 1
            n_head += 1
        # 同一例内 4 个模态是否同尺寸(共配准)
        szs = {hs[m]["size"] for m in hs}
        if len(szs) == 1:
            per_case_consistent += 1
        # 方向是否单位阵(取一个模态代表)
        any_m = next(iter(hs.values()))
        if is_identity(any_m["direction"]):
            dirs_ok += 1

    print(f"读取头文件数: {n_head}")
    print(f"同例内4模态尺寸完全一致(共配准)的病例数: {per_case_consistent}/{len(cases)}")
    print(f"方向矩阵≈单位阵(轴位/无旋转)的病例数: {dirs_ok}/{len(cases)}")
    for m in MODS:
        print(f"\n  [{m}] 尺寸(size)取值分布:")
        for s, cnt in sorted(shapes.get(m, {}).items(), key=lambda x: -x[1])[:6]:
            print(f"      {s}: {cnt} 例")
        print(f"  [{m}] 数据类型: {dtypes.get(m, {})}")
        sp = np.array(spacings.get(m, []))
        if len(sp):
            print(f"  [{m}] 体素间距 spacing (mm) 统计: "
                  f"x[{sp[:,0].min():.3f},{sp[:,0].max():.3f}] "
                  f"y[{sp[:,1].min():.3f},{sp[:,1].max():.3f}] "
                  f"z[{sp[:,2].min():.3f},{sp[:,2].max():.3f}] "
                  f"| z中位={np.median(sp[:,2]):.3f}")

    # --- 抽样体素统计 ---
    sample = random.sample(cases, min(8, len(cases)))
    print(f"\n  抽样 {len(sample)} 例读取体素 -> 强度/ROI标签统计:")
    roi_vals_all = {}
    for c in sample:
        cid = os.path.basename(c)
        line = f"    [{cid}] "
        for m in ["ADC", "DWI", "T2_not_fs"]:
            p = os.path.join(c, m + ".nii")
            if os.path.exists(p):
                a = sitk.GetArrayFromImage(sitk.ReadImage(p)).astype(np.float32)
                line += f"{m}:min={a.min():.0f}/max={a.max():.0f}/mean={a.mean():.0f} "
        rp = os.path.join(c, "ROI.nii")
        if os.path.exists(rp):
            r = sitk.GetArrayFromImage(sitk.ReadImage(rp))
            uv, ct = np.unique(r, return_counts=True)
            roi_vals_all[cid] = dict(zip(uv.tolist(), ct.tolist()))
            fg = ct[uv != 0].sum() if (uv != 0).any() else 0
            line += f"| ROI unique={dict(zip(uv.tolist(), ct.tolist()))} 前景体素={fg}"
        print(line)

    print(f"\n  [{name}] ROI 标签唯一值汇总(抽样): "
          f"所有出现过的标签值 = {sorted(set(v for d in roi_vals_all.values() for v in d.keys()))}")
    print()
