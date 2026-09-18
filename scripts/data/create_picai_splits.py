#!/usr/bin/env python
"""PI-CAI 患者级 train/validation 划分（research_plan v2.0 协议）。

协议：
- PI-CAI 三个中心共同进入 train/validation；中心仅作为分层因素，不作为未见域；
- 划分单位 patient_id，按 center × patient_csPCa 分层；seed=0；val_ratio=0.15；
- PI-CAI 内部不设 test；最终测试由通过 G0-E 语义与几何审计后冻结的外部数据集承担；
- 旧文件 data/splits/picai_cross_center_splits.json 保留用于历史追踪，不得覆盖；
- 划分算法与 v1 相同（seed/分层键/比例不变），新文件患者列表应与旧文件一致，
  可用 --compare-legacy 做编程校验（输出一致性检查 JSON）。

输出：data/splits/picai_train_val_split.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))
from zonal_reliability_fusion.data import picai as P  # noqa: E402

PROJECT = Path(__file__).resolve().parents[2]


def build_patient_index(ms):
    p_studies = defaultdict(list)
    p_center = {}
    p_csPCa = {}
    center_conflicts = []
    for row in ms:
        p = row["patient_id"]
        p_studies[p].append(row["study_id"])
        c = row.get("center", "")
        if p in p_center and p_center[p] != c:
            center_conflicts.append(p)
        p_center[p] = c
        p_csPCa[p] = p_csPCa.get(p, False) or (row.get("case_csPCa", "") == "YES")
    n_multi = sum(1 for v in p_studies.values() if len(v) > 1)
    return p_studies, p_center, p_csPCa, center_conflicts, n_multi


def stratified_val_split(patients, p_center, p_csPCa, val_ratio, seed):
    """全部 patient 按 (center, patient_csPCa) 分层划分 train/val。"""
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for p in patients:
        buckets[(p_center[p], p_csPCa[p])].append(p)
    train, val = [], []
    for key in sorted(buckets):
        ps = sorted(buckets[key])
        rng.shuffle(ps)
        n = len(ps)
        if n <= 1:
            n_val = 0
        elif n < 6:
            n_val = 1
        else:
            n_val = max(1, int(round(n * val_ratio)))
        val += ps[:n_val]
        train += ps[n_val:]
    return sorted(train), sorted(val)


def summarize(pids, p_center, p_csPCa, p_studies):
    studies = [s for p in pids for s in p_studies[p]]
    pos = sum(1 for p in pids if p_csPCa[p])
    by_center = defaultdict(lambda: [0, 0])  # center -> [n, pos]
    for p in pids:
        by_center[p_center[p]][0] += 1
        by_center[p_center[p]][1] += 1 if p_csPCa[p] else 0
    return {"n_patients": len(pids), "n_studies": len(studies),
            "n_positive_patients": pos, "n_negative_patients": len(pids) - pos,
            "by_center": {c: {"n": v[0], "pos": v[1]} for c, v in sorted(by_center.items())},
            "patients": sorted(pids), "studies": sorted(studies)}


def compare_with_legacy(doc, legacy_path: Path) -> dict:
    """编程校验新划分与旧文件的 patient/study 列表一致性。"""
    old = json.loads(legacy_path.read_text(encoding="utf-8"))
    out = {"legacy": str(legacy_path), "checks": {}, "pass": True}
    for name in ("train", "validation"):
        for field in ("patients", "studies"):
            a = sorted(old["splits"][name][field])
            b = sorted(doc["splits"][name][field])
            ok = a == b
            out["checks"][f"{name}_{field}_equal"] = {"n_old": len(a), "n_new": len(b), "equal": ok}
            out["pass"] = out["pass"] and ok
    return out


def main():
    ap = argparse.ArgumentParser(description="PI-CAI 患者级 train/val 划分（v2.0；测试由外部数据集承担）")
    ap.add_argument("--data-root", default=str(P.DEFAULT_DATA_ROOT))
    ap.add_argument("--out", default=str(PROJECT / "data/splits/picai_train_val_split.json"))
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compare-legacy", default="", help="旧划分文件路径（患者/study 列表一致性校验）")
    ap.add_argument("--comparison-out", default="", help="一致性校验 JSON 输出（默认写在新划分同目录）")
    args = ap.parse_args()

    ms = P.load_marksheet(Path(args.data_root))
    p_studies, p_center, p_csPCa, center_conflicts, n_multi = build_patient_index(ms)
    print(f"[splits] patients={len(p_studies)} studies={len(ms)} "
          f"多study患者={n_multi} center冲突患者={len(set(center_conflicts))}")

    all_patients = sorted(p_studies.keys())
    tr_p, va_p = stratified_val_split(all_patients, p_center, p_csPCa, args.val_ratio, args.seed)

    leaks = P.find_patient_leaks({"train": tr_p, "validation": va_p})
    leak_check = {"n_leaked_patients": len(leaks), "leaked": leaks, "pass": len(leaks) == 0}

    # 多 study 患者：所有 study 必须落在同一集合（通过 study -> split 映射独立复核）
    study_split = {}
    for name, pids in (("train", tr_p), ("validation", va_p)):
        for p in pids:
            for s in p_studies[p]:
                study_split[s] = name
    multi_violations = [p for p, studies in p_studies.items()
                        if len(studies) > 1 and len({study_split[s] for s in studies}) != 1]
    multi_check = {"n_multi_study_patients": n_multi,
                   "all_multi_study_patients_single_split": len(multi_violations) == 0,
                   "violations": multi_violations[:5]}

    doc = {
        "seed": args.seed, "val_ratio": args.val_ratio, "unit": "patient_id",
        "scope": "PI-CAI 仅 train/validation；最终测试由通过 G0-E 语义与几何审计后冻结的外部数据集承担，PI-CAI 内部不设 test。",
        "val_stratification": "center × patient_csPCa",
        "note": "同一患者多次 study 同集合；validation 用于参数选择，不得用外部测试集调参。",
        "center_conflicts": sorted(set(center_conflicts)),
        "multi_study_check": multi_check,
        "splits": {
            "train": summarize(tr_p, p_center, p_csPCa, p_studies),
            "validation": summarize(va_p, p_center, p_csPCa, p_studies),
        },
        "leak_check": leak_check,
        "all_splits_pass_leak_check": leak_check["pass"],
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2, ensure_ascii=False))

    tr, va = doc["splits"]["train"], doc["splits"]["validation"]
    print(f"[splits] all_pass={leak_check['pass']} -> {out}")
    print(f"  train: p={tr['n_patients']}(+{tr['n_positive_patients']}) studies={tr['n_studies']} by_center={tr['by_center']}")
    print(f"  val:   p={va['n_patients']}(+{va['n_positive_patients']}) studies={va['n_studies']} by_center={va['by_center']}")
    print(f"  multi_study: {multi_check}")
    if not leak_check["pass"]:
        print(f"[splits][WARN] 患者泄漏: {leaks[:5]}")

    if args.compare_legacy:
        cmp = compare_with_legacy(doc, Path(args.compare_legacy))
        cmp_path = Path(args.comparison_out) if args.comparison_out else \
            out.with_name(out.stem + "_vs_legacy_check.json")
        cmp_path.write_text(json.dumps(cmp, indent=2, ensure_ascii=False))
        print(f"[splits] legacy 一致性校验 pass={cmp['pass']} -> {cmp_path}")
        for k, v in cmp["checks"].items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
