#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PI-CAI 临床信息 marksheet.csv 统计分析（正确处理引号内逗号）。"""
import pandas as pd
import numpy as np

CSV = "/opt/data/private/lm/data/Prostate/PI-CAI/labels/clinical_information/marksheet.csv"
df = pd.read_csv(CSV)

print("=" * 60)
print("PI-CAI marksheet.csv 临床统计")
print("=" * 60)
print("列名:", list(df.columns))
print("总行数(study数):", len(df))
print("唯一 patient_id 数:", df["patient_id"].nunique())
print("唯一 study_id 数:", df["study_id"].nunique())

print("\n--- case_csPCa 分布 (YES=临床显著前列腺癌) ---")
print(df["case_csPCa"].value_counts(dropna=False))

print("\n--- center 分布 (采集中心) ---")
print(df["center"].value_counts(dropna=False))

print("\n--- histopath_type 分布 (病理采样方式) ---")
print(df["histopath_type"].value_counts(dropna=False))

print("\n--- case_ISUP 分布 (病例级 ISUP 分级) ---")
print(df["case_ISUP"].value_counts(dropna=False).sort_index())

# 数值型临床指标
for col in ["patient_age", "psa", "psad", "prostate_volume"]:
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    print(f"\n--- {col} ---")
    print(f"  有效值数: {len(s)} / {len(df)}  缺失: {df[col].isna().sum()}")
    if len(s):
        print(f"  min={s.min():.2f} q25={s.quantile(.25):.2f} median={s.median():.2f} "
              f"q75={s.quantile(.75):.2f} max={s.max():.2f} mean={s.mean():.2f}")

# 按 csPCa 分组比较 PSA / 体积 / 年龄
print("\n--- 按 case_csPCa 分组 (中位数) ---")
g = df.copy()
for col in ["patient_age", "psa", "prostate_volume", "psad"]:
    g[col] = pd.to_numeric(g[col], errors="coerce")
print(g.groupby("case_csPCa")[["patient_age", "psa", "prostate_volume", "psad"]].median())
print("\n各组样本量:")
print(g["case_csPCa"].value_counts())

# lesion_ISUP: 展开每个病灶(引号内逗号分隔)
print("\n--- lesion_ISUP 病灶级统计 (展开多病灶) ---")
les = []
for v in df["lesion_ISUP"].dropna():
    for part in str(v).split(","):
        part = part.strip()
        if part and part != "N/A":
            try:
                les.append(int(float(part)))
            except ValueError:
                pass
les = pd.Series(les)
print("病灶总数(可解析ISUP):", len(les))
print(les.value_counts().sort_index())
