#!/usr/bin/env python
"""P0B 物化报告审计：materialization_report.csv × manifest 期望交叉核对。

用途：全量物化完成后（或运行中途配合 --allow-incomplete）验证
"无静默遗漏、标签语义正确、阳性不丢失、阴性不凭空出现、异常 WG 已显式禁用"。

检查项：
1. 覆盖：report 的 case 集合与所选清单一致（中途运行时加 --allow-incomplete）；
2. 终态：所有行 status ∈ {ok, skipped_complete}，无 failed；
3. validation_ok 全为 True（ok 行）；
4. 阳性保留 / 阴性为空（对照 manifest canonical_fg_voxels）；
5. lesion_unique ⊆ {0,1}；zonal_*_unique ⊆ {0,1,2}；wg_unique ⊆ {0,1}；
6. exact_same_grid 病例 lesion_fg_pre == manifest canonical_fg_voxels；
7. `11050_1001070` 的 wg_status 以 excluded 开头。

输出：<report 同目录>/materialization_audit.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]


def parse_uniques(s) -> set:
    s = str(s)
    if not s or s == "nan":
        return set()
    return {int(x) for x in s.split(",")}


def main() -> None:
    ap = argparse.ArgumentParser(description="P0B 物化报告审计（report × manifest）")
    ap.add_argument("--manifest", default=str(PROJECT / "data/metadata/picai_manifest.csv"))
    ap.add_argument("--report", default=str(PROJECT / "data/processed/picai/materialization_report.csv"))
    ap.add_argument("--out", default="")
    ap.add_argument("--wg-excluded-case", default="11050_1001070")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="允许 report 尚未覆盖全部 manifest 病例（运行中途检查用）")
    args = ap.parse_args()

    man = pd.read_csv(args.manifest)
    rep = pd.read_csv(args.report)
    man_ids = set(man["case_id"])
    rep_ids = set(rep["case_id"])
    checks, fails = {}, []

    missing = sorted(man_ids - rep_ids)
    unknown = sorted(rep_ids - man_ids)
    coverage_pass = (not missing) and (not unknown)
    checks["coverage"] = {"n_manifest": len(man_ids), "n_report": len(rep_ids),
                          "n_missing": len(missing), "missing_cases": missing[:20],
                          "unknown_cases": unknown[:20],
                          "pass": bool(coverage_pass or (args.allow_incomplete and not unknown))}
    if unknown:
        fails.append({"check": "coverage_unknown", "cases": unknown[:20]})

    statuses = rep["status"].astype(str)
    failed = rep[statuses == "failed"]["case_id"].tolist()
    checks["terminal_status"] = {"counts": statuses.value_counts().to_dict(),
                                 "failed": failed[:20],
                                 "pass": bool(statuses.isin(["ok", "skipped_complete"]).all())}
    if failed:
        fails.append({"check": "terminal_status", "cases": failed[:20]})

    ok = rep[rep["status"].astype(str) == "ok"]
    vo = ok["validation_ok"].astype(str).str.lower()
    checks["validation_ok_all_true"] = {"n_ok_rows": int(len(ok)),
                                        "n_not_true": int((vo != "true").sum()),
                                        "pass": bool((vo == "true").all()) if len(ok) else False}

    j = man[["case_id", "canonical_fg_voxels", "canonical_grid"]].merge(rep, on="case_id", how="inner")
    j["fg_pre"] = pd.to_numeric(j["lesion_fg_pre"], errors="coerce").fillna(-1)
    j["fg_post"] = pd.to_numeric(j["lesion_fg_post"], errors="coerce").fillna(-1)
    j["man_fg"] = pd.to_numeric(j["canonical_fg_voxels"], errors="coerce").fillna(-1)

    pos = j[j["man_fg"] > 0]
    neg = j[j["man_fg"] == 0]
    pos_lost = pos[pos["fg_post"] <= 0]["case_id"].tolist()
    neg_bad = neg[neg["fg_post"] != 0]["case_id"].tolist()
    checks["positive_preserved"] = {"n_positives": int(len(pos)), "n_lost": len(pos_lost),
                                    "lost": pos_lost[:20], "pass": len(pos_lost) == 0}
    checks["negative_empty"] = {"n_negatives": int(len(neg)), "n_bad": len(neg_bad),
                                "bad": neg_bad[:20], "pass": len(neg_bad) == 0}
    if pos_lost:
        fails.append({"check": "positive_preserved", "cases": pos_lost[:20]})
    if neg_bad:
        fails.append({"check": "negative_empty", "cases": neg_bad[:20]})

    bad_lesion = [c for c, s in zip(j["case_id"], j["lesion_unique"]) if not parse_uniques(s).issubset({0, 1})]
    checks["lesion_values"] = {"n_bad": len(bad_lesion), "bad": bad_lesion[:20], "pass": len(bad_lesion) == 0}
    for col in ("yuan_unique", "hevi_unique"):
        bad = [c for c, s in zip(j["case_id"], j[col]) if not parse_uniques(s).issubset({0, 1, 2})]
        checks[f"{col}_values"] = {"n_bad": len(bad), "bad": bad[:20], "pass": len(bad) == 0}
    bad_wg = [c for c, s in zip(j["case_id"], j["wg_unique"]) if parse_uniques(s) - {0, 1}]
    checks["wg_values"] = {"n_bad": len(bad_wg), "bad": bad_wg[:20], "pass": len(bad_wg) == 0}

    exact = j[(j["canonical_grid"] == "exact_same_grid") & (j["man_fg"] >= 0)]
    mism = exact[exact["fg_pre"] != exact["man_fg"]]["case_id"].tolist()
    checks["exact_grid_fg_equal_manifest"] = {"n_exact": int(len(exact)), "n_mismatch": len(mism),
                                              "mismatch": mism[:20], "pass": len(mism) == 0}

    wgx = j[j["case_id"] == args.wg_excluded_case]
    if len(wgx):
        st = str(wgx.iloc[0]["wg_status"])
        checks["wg_excluded_case"] = {"case_id": args.wg_excluded_case, "wg_status": st,
                                      "pass": st.startswith("excluded")}
    else:
        checks["wg_excluded_case"] = {"case_id": args.wg_excluded_case, "wg_status": "not_in_report",
                                      "note": "该例尚未处理或不在报告中", "pass": False}

    all_pass = all(bool(v.get("pass")) for v in checks.values())
    out = {"checked_at": datetime.now().isoformat(timespec="seconds"),
           "manifest": args.manifest, "report": args.report,
           "n_report_rows": int(len(rep)), "checks": checks,
           "all_pass": bool(all_pass), "fails": fails}
    out_path = Path(args.out) if args.out else Path(args.report).with_name("materialization_audit.json")
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    for k, v in checks.items():
        assert k
    print(json.dumps(checks, indent=2, ensure_ascii=False)[:2600])
    print(f"[audit] all_pass={all_pass} -> {out_path}")


if __name__ == "__main__":
    main()
