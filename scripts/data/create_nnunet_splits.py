#!/usr/bin/env python
"""N0 准备：把冻结划分转换为 nnU-Net v2 单 fold splits_final.json（Dataset605_PICAI）。

为什么必须显式提供（third_party/nnUNet v2.6.2，
nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py:551 do_split）：
    训练时若 <nnUNet_preprocessed>/<dataset>/splits_final.json 不存在，nnU-Net 会自行生成
    默认 5-fold（seed=12345），既不使用本研究冻结的 1256/220（患者）与 1277/223（study）
    划分，也可能把同一患者的不同 study 分到 train / val 两侧。

输入（只读）：
    data/splits/picai_train_val_split.json   冻结划分（含 patients 与 studies 两级列表）
    data/metadata/picai_manifest.csv         study_id -> case_id 映射、中心、阳性标签

输出（默认不覆盖：内容一致则跳过，不一致且未加 --overwrite 则中止）：
    <preprocessed>/Dataset605_PICAI/splits_final.json   训练实际读取的文件（fold 0 为唯一 fold）
    <raw>/Dataset605_PICAI/splits_final.json            副本；nnU-Net 的 plan 阶段会把 raw 中的
                                                        该文件复制到 preprocessed，可防止将来
                                                        重新规划时丢失
    data/splits/picai_nnunet_splits_validation.json     结构化校验结果
    data/splits/picai_nnunet_split_cases.csv            case -> split 明细表

校验（任一失败即中止，不写出任何文件）：
    1. 两侧 study 列表无重复，且数量与划分文件声明的 n_studies / n_patients 一致
    2. train / validation 的 study 与患者均无交集；全部 study 均已分配且无跨侧
    3. train ∪ validation == manifest 全部 study == nnU-Net 预处理 case 集合
    4. 多 study 患者（23 名）的全部 study 都在同一侧
    5. manifest 的 case_id 等于 f"{patient_id}_{study_id}"（nnU-Net 命名假设）

说明：
    - 只做格式转换，不重新划分、不引入随机性；不读取任何影像体素。
    - 写入采用临时文件 fsync 后原子重命名；结束输出成功/失败计数与耗时。
    - 本脚本只负责数据格式；启动训练仍由研究者另行明确执行。

用法：
    python scripts/data/create_nnunet_splits.py --dry-run
    python scripts/data/create_nnunet_splits.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_SPLIT_JSON = PROJECT / "data/splits/picai_train_val_split.json"
DEFAULT_MANIFEST = PROJECT / "data/metadata/picai_manifest.csv"
DEFAULT_PREPROCESSED_ROOT = PROJECT / "workdir/nnUNet_preprocessed"
DEFAULT_RAW_ROOT = PROJECT / "workdir/nnUNet_raw"
DEFAULT_SUMMARY = PROJECT / "data/splits/picai_nnunet_splits_validation.json"
DEFAULT_MAPPING = PROJECT / "data/splits/picai_nnunet_split_cases.csv"

GT_SUFFIX = ".nii.gz"
B2ND_SUFFIX = ".b2nd"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def norm_splits(obj) -> list[dict]:
    """规范化 splits 结构用于内容比较（键名/排序统一）。"""
    out = []
    for fold in obj:
        out.append({"train": sorted(map(str, fold.get("train", []))),
                    "val": sorted(map(str, fold.get("val", [])))})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="生成 nnU-Net 单 fold splits_final.json（Dataset605_PICAI）")
    ap.add_argument("--split-json", default=str(DEFAULT_SPLIT_JSON))
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--preprocessed-root", default=str(DEFAULT_PREPROCESSED_ROOT),
                    help="nnUNet_preprocessed 根目录（默认项目内 workdir/nnUNet_preprocessed）")
    ap.add_argument("--raw-root", default=str(DEFAULT_RAW_ROOT),
                    help="nnUNet_raw 根目录（默认项目内 workdir/nnUNet_raw，用于写副本）")
    ap.add_argument("--dataset-id", type=int, default=605)
    ap.add_argument("--dataset-name", default="PICAI")
    ap.add_argument("--summary-out", default=str(DEFAULT_SUMMARY))
    ap.add_argument("--mapping-out", default=str(DEFAULT_MAPPING))
    ap.add_argument("--no-raw-copy", action="store_true", help="不向 nnUNet_raw 写副本")
    ap.add_argument("--dry-run", action="store_true", help="只校验与打印计划，不写任何文件")
    ap.add_argument("--overwrite", action="store_true", help="目标内容不一致时强制重建")
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    ds_name = f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    pre_dir = Path(args.preprocessed_root) / ds_name
    raw_dir = Path(args.raw_root) / ds_name
    print(f"[nnunet-splits] dataset={ds_name}")
    print(f"[nnunet-splits] preprocessed={pre_dir}")
    print(f"[nnunet-splits] raw={raw_dir}{'（跳过副本）' if args.no_raw_copy else ''}")
    print(f"[nnunet-splits] split_json={args.split_json}")
    print(f"[nnunet-splits] manifest={args.manifest}")
    for env_name, chosen in (("nnUNet_preprocessed", args.preprocessed_root), ("nnUNet_raw", args.raw_root)):
        env_val = os.environ.get(env_name, "")
        if env_val and Path(env_val).resolve() != Path(chosen).resolve():
            print(f"[nnunet-splits][WARN] 环境变量 {env_name}={env_val} 与本脚本使用的 {chosen} 不一致；"
                  f"训练时请显式设置 {env_name}={chosen}")

    checks: list[dict] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"check": name, "pass": bool(ok), "detail": detail})
        print(f"[nnunet-splits] {'PASS' if ok else 'FAIL'} {name}" + (f" | {detail}" if detail else ""))

    # ---------- 读取输入 ----------
    split_path = Path(args.split_json)
    manifest_path = Path(args.manifest)
    doc = json.loads(split_path.read_text())
    sp = doc["splits"]
    for key in ("train", "validation"):
        if key not in sp:
            raise SystemExit(f"[nnunet-splits] 划分文件缺少 splits.{key}（现有键: {list(sp.keys())}）")
    df = pd.read_csv(manifest_path)

    # 结构性前置条件：不满足则无法建立映射
    if not df["study_id"].is_unique:
        raise SystemExit("[nnunet-splits] manifest 中 study_id 不唯一，无法建立 study->case 映射")
    if not df["case_id"].is_unique:
        raise SystemExit("[nnunet-splits] manifest 中 case_id 不唯一")
    m = df.assign(sid=df["study_id"].astype(str), cid=df["case_id"].astype(str), pid=df["patient_id"].astype(str))
    if not ((m["pid"] + "_" + m["sid"]) == m["cid"]).all():
        raise SystemExit("[nnunet-splits] manifest 的 case_id 不等于 f'{patient_id}_{study_id}'，与 nnU-Net 命名假设不符")
    study_all = set(m["sid"])
    case_all = set(m["cid"])

    # ---------- 预处理 case 集合 ----------
    gt_dir = pre_dir / "gt_segmentations"
    b2nd_dir = pre_dir / "nnUNetPlans_3d_fullres"
    if not gt_dir.is_dir() or not b2nd_dir.is_dir():
        raise SystemExit(f"[nnunet-splits] 预处理目录不完整: {pre_dir}（需要 gt_segmentations/ 与 nnUNetPlans_3d_fullres/）")
    gt_ids = {p.name[: -len(GT_SUFFIX)] for p in gt_dir.glob(f"*{GT_SUFFIX}")}
    b2_ids = {p.name[: -len(B2ND_SUFFIX)] for p in b2nd_dir.glob(f"*{B2ND_SUFFIX}")
              if not p.name.endswith("_seg" + B2ND_SUFFIX)}

    sides = {k: [str(s) for s in sp[k]["studies"]] for k in ("train", "validation")}
    train_studies, val_studies = set(sides["train"]), set(sides["validation"])
    train_pats = {str(p) for p in sp["train"]["patients"]}
    val_pats = {str(p) for p in sp["validation"]["patients"]}

    # ---------- 逐病例校验（进度可见） ----------
    rows: list[dict] = []
    missing_gt: list[str] = []
    missing_b2nd: list[str] = []
    no_side: list[str] = []
    bar = tqdm(m.itertuples(index=False), total=len(m), desc="nnunet-splits",
               mininterval=1.0, disable=args.no_progress, file=sys.stdout)
    for r in bar:
        sid, cid, pid = str(r.study_id), str(r.case_id), str(r.patient_id)
        if cid not in gt_ids:
            missing_gt.append(cid)
        if cid not in b2_ids:
            missing_b2nd.append(cid)
        if sid in train_studies:
            side = "train"
        elif sid in val_studies:
            side = "validation"
        else:
            side = ""
            no_side.append(sid)
        rows.append({"case_id": cid, "study_id": sid, "patient_id": pid,
                     "center": r.center, "case_csPCa": r.case_csPCa, "split": side})
    side_of_case = {r["study_id"]: r["split"] for r in rows}

    train_cases = sorted(c for s, c in zip(m["sid"], m["cid"]) if s in train_studies)
    val_cases = sorted(c for s, c in zip(m["sid"], m["cid"]) if s in val_studies)

    # ---------- 校验 ----------
    multi_pids = {p for p, n in m["pid"].value_counts().items() if n > 1}
    multi_viol = [p for p in sorted(multi_pids)
                  if len({side_of_case.get(c, "") for c in set(m.loc[m["pid"] == p, "sid"])}) != 1]

    check("两侧 study 列表无重复",
          all(len(sides[k]) == len(set(sides[k])) for k in sides),
          f"train={len(sides['train'])}/{len(set(sides['train']))} "
          f"validation={len(sides['validation'])}/{len(set(sides['validation']))}")
    check("train / validation 的 study 无交集", not (train_studies & val_studies),
          f"overlap={len(train_studies & val_studies)}")
    check("train / validation 的患者无交集", not (train_pats & val_pats),
          f"overlap={len(train_pats & val_pats)}")
    check("全部 study 均已分配（无遗漏）", not no_side, f"未分配={len(no_side)}")
    check("每个 case 恰好归属一侧", set(side_of_case.values()) == {"train", "validation"},
          f"取值={sorted(set(side_of_case.values()))}")
    check("study 并集覆盖 manifest 全部", (train_studies | val_studies) == study_all,
          f"union={len(train_studies | val_studies)} manifest={len(study_all)}")
    check("与划分文件声明的 n_studies 一致",
          all(len(sides[k]) == int(sp[k].get("n_studies", -1)) for k in sides),
          f"train={len(sides['train'])}/{sp['train'].get('n_studies')} "
          f"validation={len(sides['validation'])}/{sp['validation'].get('n_studies')}")
    check("与划分文件声明的 n_patients 一致",
          all(len(set(map(str, sp[k]["patients"]))) == int(sp[k].get("n_patients", -1)) for k in sides),
          f"train={len(train_pats)}/{sp['train'].get('n_patients')} "
          f"validation={len(val_pats)}/{sp['validation'].get('n_patients')}")
    check("预处理 case 集合一致（gt_segmentations）", gt_ids == case_all,
          f"gt={len(gt_ids)} manifest={len(case_all)} 差={sorted((gt_ids ^ case_all))[:5]}")
    check("预处理 case 集合一致（nnUNetPlans_3d_fullres）", b2_ids == case_all,
          f"b2nd={len(b2_ids)} manifest={len(case_all)} 差={sorted((b2_ids ^ case_all))[:5]}")
    check("train ∪ validation == 预处理 case 集合", (set(train_cases) | set(val_cases)) == gt_ids,
          f"split={len(set(train_cases) | set(val_cases))} pre={len(gt_ids)}")
    check("逐病例存在性（gt_seg / b2nd）", not missing_gt and not missing_b2nd,
          f"缺 gt_seg={len(missing_gt)} 缺 b2nd={len(missing_b2nd)}")
    check(f"多 study 患者（{len(multi_pids)} 名）全部 study 在同一侧", not multi_viol,
          f"违规={multi_viol[:5]}")

    # ---------- 计划写出 ----------
    splits_obj = [{"train": train_cases, "val": val_cases}]
    payload = (json.dumps(splits_obj, indent=2, ensure_ascii=False) + "\n").encode()
    targets = [("preprocessed", pre_dir / "splits_final.json")]
    if not args.no_raw_copy:
        if raw_dir.is_dir():
            targets.append(("raw", raw_dir / "splits_final.json"))
        else:
            print(f"[nnunet-splits][WARN] raw 数据集目录不存在，跳过副本: {raw_dir}")
    kb = len(payload) / 1024.0
    print(f"[nnunet-splits] 体积估算: splits_final.json ≈ {kb:.1f} KB × {len(targets)} 个目标；"
          f"summary/mapping 合计 < 1 MB（磁盘空间可忽略）")

    def target_state(path: Path) -> str:
        if not path.exists():
            return "absent"
        try:
            old = json.loads(path.read_text())
        except Exception:
            return "unreadable"
        return "identical" if norm_splits(old) == norm_splits(splits_obj) else "different"

    states = {label: target_state(path) for label, path in targets}
    print(f"[nnunet-splits] 目标状态: " + "  ".join(f"{label}={states[label]}({path.name})" for label, path in targets))

    n_fail = sum(1 for c in checks if not c["pass"])
    if n_fail:
        print(f"[nnunet-splits] 校验失败 {n_fail}/{len(checks)} 项，未写出任何文件")
        raise SystemExit(2)

    if args.dry_run:
        print(f"[nnunet-splits][dry-run] 全部校验通过；将写出:")
        for label, path in targets:
            print(f"    {label}: {path}  (fold 0: train={len(train_cases)} val={len(val_cases)})")
        print(f"    summary: {args.summary_out}")
        print(f"    mapping: {args.mapping_out}（{len(rows)} 行）")
        print(f"[nnunet-splits][dry-run] 未写出任何文件；耗时 {time.time() - t0:.1f}s")
        return

    blocked = [(label, path) for label, path in targets
               if states[label] == "different" and not args.overwrite]
    if blocked:
        for label, path in blocked:
            print(f"[nnunet-splits] 目标已存在且内容不一致: {path}。加 --overwrite 强制重建。")
        raise SystemExit(3)
    unreadable = [(label, path) for label, path in targets if states[label] == "unreadable"]
    if unreadable and not args.overwrite:
        for label, path in unreadable:
            print(f"[nnunet-splits] 目标存在但无法解析: {path}。加 --overwrite 重建。")
        raise SystemExit(3)

    written = []
    for label, path in targets:
        if states[label] == "identical":
            print(f"[nnunet-splits] 跳过（内容一致）: {path}")
            written.append({"label": label, "path": str(path), "action": "skipped_identical"})
            continue
        atomic_write_bytes(path, payload)
        written.append({"label": label, "path": str(path), "action": "written",
                        "sha256": sha256_bytes(payload)})
        print(f"[nnunet-splits] 写出: {path} (sha256={sha256_bytes(payload)[:16]}...)")

    # 回读一致性（以训练实际读取的文件为准）
    readback = json.loads((pre_dir / "splits_final.json").read_text())
    check("回读校验：fold 数=1 且内容一致",
          len(readback) == 1 and norm_splits(readback) == norm_splits(splits_obj),
          f"folds={len(readback)} train={len(readback[0]['train'])} val={len(readback[0]['val'])}")

    # 映射表
    pd.DataFrame(rows).to_csv(args.mapping_out, index=False)

    ctr = {}
    for key, ss in (("train", train_studies), ("validation", val_studies)):
        sub = m[m["sid"].isin(ss)]
        ctr[key] = {k: int(v) for k, v in sorted(sub["center"].value_counts().to_dict().items())}

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": ds_name,
        "fold_count": 1,
        "split_unit": doc.get("unit"),
        "scope": doc.get("scope"),
        "seed": doc.get("seed"),
        "val_ratio": doc.get("val_ratio"),
        "counts": {
            "train_studies": len(train_cases), "validation_studies": len(val_cases),
            "total_studies": len(train_cases) + len(val_cases),
            "train_patients": len(train_pats), "validation_patients": len(val_pats),
        },
        "by_center": ctr,
        "checks": checks,
        "inputs": {
            "split_json": str(split_path), "split_json_sha256": sha256_file(split_path),
            "manifest": str(manifest_path), "manifest_sha256": sha256_file(manifest_path),
            "preprocessed_dir": str(pre_dir),
        },
        "outputs": {
            "splits_final_targets": written,
            "splits_payload_sha256": sha256_bytes(payload),
            "summary": str(args.summary_out), "mapping_csv": str(args.mapping_out),
            "mapping_rows": len(rows),
        },
        "elapsed_sec": round(time.time() - t0, 1),
    }
    Path(args.summary_out).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"[nnunet-splits] summary={args.summary_out}  mapping={args.mapping_out}({len(rows)} 行)")
    print(f"[nnunet-splits] 完成: 校验 {len(checks)}/{len(checks)} PASS，"
          f"fold0 train={len(train_cases)} validation={len(val_cases)}，耗时 {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
