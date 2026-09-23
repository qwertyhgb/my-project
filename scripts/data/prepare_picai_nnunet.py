#!/usr/bin/env python3
"""统一的 PI-CAI -> nnU-Net 数据准备入口（三个 subcommand）。

设计边界：**只做最少量的数据组织**，不重实现 nnU-Net 的 planning/preprocessing。所有派生
结果写入 nnU-Net 约定目录（``$nnUNet_raw`` / ``$nnUNet_preprocessed``，默认在项目 ``workdir/``），
原始物化数据（``data/processed/picai/cases``）只读。长命令由研究者本人运行。

subcommand：

    baseline  组织 Dataset605_PICAI（3 通道 T2W/ADC/HBV）的 nnU-Net raw 数据集（符号链接，不复制体素）。
    zonal     组织 Dataset606_PICAI_Zonal（5 通道：T2W/ADC/HBV + PZ/TZ）。MRI 与标签沿用符号链接；
              PZ/TZ 由分区标签 zonal_<source>.nii.gz 派生为两个算法生成的 [0,1] 分区隶属通道
              （zonal membership；非校准概率、非几何体素占比；noNorm），与对应病例 T2W 网格严格一致。
    splits    把冻结划分 data/splits/picai_train_val_split.json 转换为某数据集的 splits_final.json（单 fold）。

安全：默认不覆盖已有内容；``--resume`` 跳过/补齐，``--overwrite`` 强制重建；派生文件原子写入；
每个 subcommand 结束都打印 成功/失败/跳过/耗时/输出目录。所有耗时循环带 tqdm（``--no-progress`` 关闭）。

用法（示例，长任务请由研究者运行）：
    python scripts/data/prepare_picai_nnunet.py baseline --dry-run
    python scripts/data/prepare_picai_nnunet.py baseline
    python scripts/data/prepare_picai_nnunet.py zonal --zonal-source yuan --dry-run
    python scripts/data/prepare_picai_nnunet.py zonal --zonal-source yuan
    python scripts/data/prepare_picai_nnunet.py splits --dataset-id 605 --dataset-name PICAI
    python scripts/data/prepare_picai_nnunet.py splits --dataset-id 606 --dataset-name PICAI_Zonal
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_MATERIALIZED = PROJECT / "data/processed/picai"
DEFAULT_MANIFEST = PROJECT / "data/metadata/picai_manifest.csv"
DEFAULT_SPLIT_JSON = PROJECT / "data/splits/picai_train_val_split.json"
DEFAULT_PREPROCESSED_ROOT = PROJECT / "workdir/nnUNet_preprocessed"
DEFAULT_RAW_ROOT = PROJECT / "workdir/nnUNet_raw"

# MRI 三通道 + 标签（顺序即 nnU-Net 通道号，固定不可变）
MRI_CHANNELS = (
    ("0000", "T2W", "t2w.nii.gz"),
    ("0001", "ADC", "adc.nii.gz"),
    ("0002", "HBV", "hbv.nii.gz"),
)
LABEL_FILE = "lesion.nii.gz"
# Dataset606 附加的 PZ/TZ 两个通道（noNorm；nnU-Net v2.6.2 的 'nonorm' -> NoNormalization）
PRIOR_CHANNELS = (("0003", "PZ"), ("0004", "TZ"))
ZONAL_LABEL_PZ = 1
ZONAL_LABEL_TZ = 2
ZONAL_SOURCES = ("yuan", "hevi")


# --------------------------------------------------------------------------- 通用工具
def _link_ok(link: Path, target: Path) -> bool:
    return link.is_symlink() and Path(os.path.realpath(link)) == Path(
        os.path.realpath(target)
    )


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _symlink(link: Path, target: Path, *, dry_run: bool, overwrite: bool) -> str:
    """建立相对符号链接；返回状态 ok/skipped/conflict/dry_run。绝不删除非链接的既有文件。"""
    if dry_run:
        return "dry_run"
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        if _link_ok(link, target):
            return "skipped"
        if not overwrite:
            return "conflict"
        link.unlink()
    rel = Path(os.path.relpath(target, link.parent))
    os.symlink(rel, link)
    return "ok"


def _select_cases(manifest_path: Path, cases_arg: str) -> list[str]:
    df = pd.read_csv(manifest_path)
    if cases_arg:
        sel = [s.strip() for s in cases_arg.split(",") if s.strip()]
        unknown = [c for c in sel if c not in set(df["case_id"])]
        if unknown:
            raise SystemExit(f"[prepare] --cases 含未知 case_id: {unknown[:5]}")
        return sel
    return df["case_id"].tolist()


def _guard_existing(ds_dir: Path, resume: bool, overwrite: bool, dry_run: bool) -> None:
    if (
        ds_dir.exists()
        and any(ds_dir.rglob("*"))
        and not (resume or overwrite or dry_run)
    ):
        raise SystemExit(
            f"[prepare] 目标已存在内容: {ds_dir}。加 --resume 跳过/补齐，或 --overwrite 重建。"
        )


def _raw_dataset_dir(raw_root: str, dataset_id: int, dataset_name: str) -> Path:
    if not raw_root:
        raise SystemExit(
            "[prepare] 未找到 nnUNet_raw：请设置环境变量或显式传 --nnunet-raw-root"
        )
    root = Path(raw_root)
    ds_dir = root / f"Dataset{dataset_id:03d}_{dataset_name}"
    if root.exists():
        others = [
            p.name
            for p in root.glob(f"Dataset{dataset_id:03d}_*")
            if p.name != ds_dir.name
        ]
        if others:
            raise SystemExit(
                f"[prepare] Dataset ID {dataset_id:03d} 已被其他名称占用: {others}"
            )
    return ds_dir


def _finish_summary(
    tag: str, t0: float, counts: dict, out_dir: Path, extra: dict | None = None
) -> None:
    elapsed = round(time.time() - t0, 2)
    print(
        f"[{tag}] done: ok={counts.get('ok', 0)} skipped={counts.get('skipped', 0)} "
        f"failed={counts.get('failed', 0)} conflict={counts.get('conflict', 0)} elapsed={elapsed}s"
    )
    print(f"[{tag}] output_dir={out_dir}")
    if extra:
        for k, v in extra.items():
            print(f"[{tag}] {k}={v}")


# --------------------------------------------------------------------------- baseline / zonal 共用
def _link_mri_and_label(
    case_dir: Path, ds_dir: Path, cid: str, *, dry_run: bool, overwrite: bool
) -> tuple[str, list[str]]:
    """为一个病例建立 MRI 三通道 + 标签的符号链接；返回 (状态, 缺失文件列表)。"""
    srcs = {fname: case_dir / fname for _, _, fname in MRI_CHANNELS}
    srcs[LABEL_FILE] = case_dir / LABEL_FILE
    missing = [str(p) for p in srcs.values() if not p.exists() or p.stat().st_size == 0]
    if missing:
        return "missing", missing
    statuses = []
    img_dir = ds_dir / "imagesTr"
    lab_dir = ds_dir / "labelsTr"
    for suffix, _, fname in MRI_CHANNELS:
        statuses.append(
            _symlink(
                img_dir / f"{cid}_{suffix}.nii.gz",
                case_dir / fname,
                dry_run=dry_run,
                overwrite=overwrite,
            )
        )
    statuses.append(
        _symlink(
            lab_dir / f"{cid}.nii.gz",
            case_dir / LABEL_FILE,
            dry_run=dry_run,
            overwrite=overwrite,
        )
    )
    if "conflict" in statuses:
        return "conflict", []
    if dry_run:
        return "dry_run", []
    if all(s == "skipped" for s in statuses):
        return "skipped", []
    return "ok", []


def _write_dataset_json(
    ds_dir: Path,
    channel_names: dict,
    num_training: int,
    description: str,
    *,
    dry_run: bool,
) -> None:
    doc = {
        "name": ds_dir.name,
        "description": description,
        "channel_names": channel_names,
        "labels": {"background": 0, "lesion": 1},
        "numTraining": num_training,
        "file_ending": ".nii.gz",
    }
    if dry_run:
        return
    _atomic_write_bytes(
        ds_dir / "dataset.json",
        (json.dumps(doc, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
    )


def _cases_on_disk(ds_dir: Path, required_suffixes: tuple[str, ...]) -> set[str]:
    """扫描 imagesTr，返回拥有**全部** required_suffixes 通道文件的 case_id 集合。"""
    img_dir = ds_dir / "imagesTr"
    if not img_dir.is_dir() or not required_suffixes:
        return set()
    sets: list[set[str]] = []
    for suf in required_suffixes:
        strip = len(suf) + len(".nii.gz") + 1  # 去掉 "_<suf>.nii.gz"
        ids = {
            p.name[:-strip] for p in img_dir.glob(f"*_{suf}.nii.gz") if p.name[:-strip]
        }
        sets.append(ids)
    return set.intersection(*sets) if sets else set()


def _finalize_raw_dataset(
    tag: str,
    *,
    ds_dir: Path,
    sel: list[str],
    counts: dict,
    failures: list[dict],
    channel_names: dict,
    description: str,
    required_suffixes: tuple[str, ...],
    subset_used: bool,
    dry_run: bool,
    t0: float,
    extra: dict | None = None,
) -> None:
    """fail-closed 收尾：先打印完整汇总，再决定写 dataset.json 还是非零退出。

    - numTraining = imagesTr 中拥有全部通道的**实际完整病例数**（不是本次选择数），与磁盘一致；
    - 任一 failed/conflict，或（完整运行时）磁盘病例集合 != 选择集合 -> 打印汇总后 SystemExit(2)，
      且**不发布**新的 dataset.json；
    - --cases 子集运行后若数据集仍含选择外的旧病例 -> 明确 WARNING（绝不静默遗留）。
    """
    sel_set = set(sel)
    summary_extra = dict(extra or {})
    summary_extra["dry_run"] = dry_run
    summary_extra["failures"] = failures[:10]

    if dry_run:
        summary_extra["numTraining_planned"] = len(sel)
        _finish_summary(tag, t0, counts, ds_dir, summary_extra)
        print(f"[{tag}][dry-run] 未写 dataset.json")
        if counts.get("failed", 0) or counts.get("conflict", 0):
            raise SystemExit(2)
        return

    on_disk = _cases_on_disk(ds_dir, required_suffixes)
    missing_selected = sorted(sel_set - on_disk)
    leftovers = sorted(on_disk - sel_set)
    num_training = len(on_disk)
    if subset_used and leftovers:
        print(
            f"[{tag}][WARN] --cases 子集运行后，数据集仍含 {len(leftovers)} 个不在本次选择内的旧病例"
            f"（未静默删除；numTraining 按磁盘实际完整病例数计）: {leftovers[:5]}"
        )
    inconsistent = (not subset_used) and (on_disk != sel_set)
    summary_extra.update(
        {
            "numTraining": num_training,
            "cases_on_disk": len(on_disk),
            "missing_selected": missing_selected[:5],
            "leftover_cases": leftovers[:5],
        }
    )
    _finish_summary(tag, t0, counts, ds_dir, summary_extra)

    if counts.get("failed", 0) or counts.get("conflict", 0) or inconsistent:
        reason = (
            "存在 failed/conflict"
            if (counts.get("failed") or counts.get("conflict"))
            else "磁盘完整病例集合与选择集合不一致"
        )
        print(f"[{tag}] fail-closed（{reason}）：未发布新的 dataset.json")
        raise SystemExit(2)
    _write_dataset_json(ds_dir, channel_names, num_training, description, dry_run=False)
    print(
        f"[{tag}] dataset.json 已写出: numTraining={num_training} path={ds_dir / 'dataset.json'}"
    )


# --------------------------------------------------------------------------- zonal 派生
class ZonalCheckError(RuntimeError):
    """PZ/TZ 派生/校验失败（几何、标签或来源不匹配）。"""


def _check_source_geometry(zonal_path: Path, t2w_path: Path, zonal_source: str):
    """读取 t2w 与 zonal_<source>，校验二者网格一致；返回 (t2w_sitk, label_arr)。"""
    import SimpleITK as sitk

    t2w = sitk.ReadImage(str(t2w_path))
    zonal = sitk.ReadImage(str(zonal_path))
    if (
        t2w.GetSize() != zonal.GetSize()
        or t2w.GetSpacing() != zonal.GetSpacing()
        or t2w.GetOrigin() != zonal.GetOrigin()
        or t2w.GetDirection() != zonal.GetDirection()
    ):
        raise ZonalCheckError(
            f"zonal_{zonal_source} 与 t2w 网格不一致（size/spacing/origin/direction）"
        )
    return t2w, sitk.GetArrayFromImage(zonal)


def _expected_pz_tz(label_arr, zonal_source: str):
    """由 0/1/2 分区标签派生 PZ/TZ（float32, [0,1]）；非法标签报错。"""
    uniq = set(np.unique(label_arr).tolist())
    if not uniq.issubset({0, ZONAL_LABEL_PZ, ZONAL_LABEL_TZ}):
        raise ZonalCheckError(
            f"zonal_{zonal_source} 含非法标签 {sorted(uniq - {0, 1, 2})[:5]}（只允许 0/1/2）"
        )
    pz = (label_arr == ZONAL_LABEL_PZ).astype(np.float32)
    tz = (label_arr == ZONAL_LABEL_TZ).astype(np.float32)
    return pz, tz


def _verify_existing_prior(
    out_pz: Path, out_tz: Path, t2w, exp_pz, exp_tz
) -> tuple[str, str]:
    """校验既有 PZ/TZ 是否与**本次指定 source** 派生结果完全一致；只读，不修改任何文件。

    完全一致 -> skipped；缺失/为空/几何不一致/形状不一致/数值越界/内容不一致 -> conflict；
    不可读（损坏）-> failed。
    """
    import SimpleITK as sitk

    for path, exp in ((out_pz, exp_pz), (out_tz, exp_tz)):
        if not path.exists():
            return "conflict", f"{path.name} 缺失"
        if path.stat().st_size == 0:
            return "conflict", f"{path.name} 为空"
        try:
            img = sitk.ReadImage(str(path))
        except Exception as exc:  # noqa: BLE001
            return "failed", f"{path.name} 不可读: {type(exc).__name__}: {exc}"
        if (
            img.GetSize() != t2w.GetSize()
            or img.GetSpacing() != t2w.GetSpacing()
            or img.GetOrigin() != t2w.GetOrigin()
            or img.GetDirection() != t2w.GetDirection()
        ):
            return "conflict", f"{path.name} 几何与 T2W 不一致"
        got = sitk.GetArrayFromImage(img)
        if got.shape != exp.shape:
            return (
                "conflict",
                f"{path.name} 形状 {tuple(got.shape)} != 预期 {tuple(exp.shape)}",
            )
        if (
            not np.isfinite(got).all()
            or float(got.min()) < 0.0
            or float(got.max()) > 1.0
        ):
            return "conflict", f"{path.name} 数值越界（要求 [0,1]）"
        if not np.array_equal(got.astype(np.float32), exp):
            return "conflict", f"{path.name} 内容与当前指定 zonal source 派生结果不一致"
    return "skipped", ""


def _write_prior_pair(out_dir: Path, out_pz: Path, out_tz: Path, t2w, pz, tz) -> None:
    """原子写出 PZ/TZ（严格复用 T2W 几何；tmp -> os.replace）。"""
    import SimpleITK as sitk

    out_dir.mkdir(parents=True, exist_ok=True)
    for out_path, channel in ((out_pz, pz), (out_tz, tz)):
        img = sitk.GetImageFromArray(channel)
        img.CopyInformation(t2w)
        tmp = out_path.with_name(f".{out_path.name}.{os.getpid()}.tmp.nii.gz")
        sitk.WriteImage(img, str(tmp))
        os.replace(tmp, out_path)


def _derive_prior_niftis(
    case_dir: Path,
    ds_dir: Path,
    cid: str,
    zonal_source: str,
    *,
    dry_run: bool,
    overwrite: bool,
) -> tuple[str, str]:
    """从 zonal_<source>.nii.gz 派生 PZ/TZ 两个 [0,1] 通道 NIfTI（与 T2W 网格严格一致）。

    返回 (状态, 原因)。状态：ok/skipped/dry_run/conflict/failed。**fail-closed**：
    - 只存在 PZ/TZ 之一且未 --overwrite -> conflict（不覆盖、不补写）；
    - 两者都存在且未 --overwrite -> 读取当前指定 source，校验既有 PZ/TZ 的几何/形状/数值/内容
      完全一致才 skipped；来源/几何/内容不一致 -> conflict；不得修改已有 PZ/TZ；
    - 仅 --overwrite 才重新原子生成（可用于来源切换或修复）。
    """
    zonal_path = case_dir / f"zonal_{zonal_source}.nii.gz"
    t2w_path = case_dir / "t2w.nii.gz"
    out_dir = ds_dir / "imagesTr"
    out_pz = out_dir / f"{cid}_0003.nii.gz"
    out_tz = out_dir / f"{cid}_0004.nii.gz"
    if not zonal_path.exists() or not t2w_path.exists():
        return "failed", f"缺少 {zonal_path.name} 或 {t2w_path.name}"

    pz_exists, tz_exists = out_pz.exists(), out_tz.exists()
    # 只有一个先验输出且未 --overwrite：结构冲突，既不覆盖也不补写
    if (pz_exists != tz_exists) and not overwrite:
        return "conflict", "PZ/TZ 只存在其一，且未指定 --overwrite（拒绝覆盖/补写）"
    if dry_run:
        return "dry_run", ""

    try:
        t2w, label_arr = _check_source_geometry(zonal_path, t2w_path, zonal_source)
        exp_pz, exp_tz = _expected_pz_tz(label_arr, zonal_source)
        # 两者都存在且未 --overwrite：先验证是否确由本次 source 生成，完全一致才 skipped
        if pz_exists and tz_exists and not overwrite:
            return _verify_existing_prior(out_pz, out_tz, t2w, exp_pz, exp_tz)
        _write_prior_pair(out_dir, out_pz, out_tz, t2w, exp_pz, exp_tz)
        return "ok", ""
    except ZonalCheckError as exc:
        return "failed", str(exc)
    except Exception as exc:  # noqa: BLE001 - 逐例失败必须记录并继续
        return "failed", f"{type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------- subcommand: baseline
def cmd_baseline(args: argparse.Namespace) -> None:
    t0 = time.time()
    mat_root = Path(args.materialized_root)
    ds_dir = _raw_dataset_dir(args.nnunet_raw_root, args.dataset_id, args.dataset_name)
    _guard_existing(ds_dir, args.resume, args.overwrite, args.dry_run)
    sel = _select_cases(Path(args.manifest), args.cases)
    print(
        f"[baseline] dataset={ds_dir.name} cases={len(sel)} raw_root={Path(args.nnunet_raw_root)}"
    )
    counts = {"ok": 0, "skipped": 0, "failed": 0, "conflict": 0}
    failures: list[dict] = []
    for cid in tqdm(
        sel,
        desc="baseline-raw",
        mininterval=1.0,
        disable=args.no_progress,
        file=sys.stdout,
    ):
        status, detail = _link_mri_and_label(
            mat_root / "cases" / cid,
            ds_dir,
            cid,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )
        if status == "missing":
            counts["failed"] += 1
            failures.append({"case_id": cid, "reason": f"missing={detail}"})
        elif status == "conflict":
            counts["conflict"] += 1
            failures.append({"case_id": cid, "reason": "link_conflict"})
        else:
            counts[status] = counts.get(status, 0) + 1
    channel_names = {suffix: name for suffix, name, _ in MRI_CHANNELS}
    _finalize_raw_dataset(
        "baseline",
        ds_dir=ds_dir,
        sel=sel,
        counts=counts,
        failures=failures,
        channel_names=channel_names,
        description="PI-CAI csPCa lesion segmentation (T2W/ADC/HBV on per-case T2W grid)",
        required_suffixes=tuple(s for s, _, _ in MRI_CHANNELS),
        subset_used=bool(args.cases),
        dry_run=args.dry_run,
        t0=t0,
    )


# --------------------------------------------------------------------------- subcommand: zonal
def cmd_zonal(args: argparse.Namespace) -> None:
    t0 = time.time()
    if args.zonal_source not in ZONAL_SOURCES:
        raise SystemExit(
            f"[zonal] --zonal-source 必须是 {list(ZONAL_SOURCES)}（不得静默默认）"
        )
    mat_root = Path(args.materialized_root)
    ds_dir = _raw_dataset_dir(args.nnunet_raw_root, args.dataset_id, args.dataset_name)
    _guard_existing(ds_dir, args.resume, args.overwrite, args.dry_run)
    sel = _select_cases(Path(args.manifest), args.cases)
    print(
        f"[zonal] dataset={ds_dir.name} cases={len(sel)} source=zonal_{args.zonal_source} raw_root={Path(args.nnunet_raw_root)}"
    )
    counts = {"ok": 0, "skipped": 0, "failed": 0, "conflict": 0}
    failures: list[dict] = []
    for cid in tqdm(
        sel,
        desc="zonal-raw",
        mininterval=1.0,
        disable=args.no_progress,
        file=sys.stdout,
    ):
        case_dir = mat_root / "cases" / cid
        link_status, detail = _link_mri_and_label(
            case_dir, ds_dir, cid, dry_run=args.dry_run, overwrite=args.overwrite
        )
        if link_status in ("missing", "conflict"):
            counts["failed" if link_status == "missing" else "conflict"] += 1
            failures.append(
                {"case_id": cid, "reason": f"mri_label_{link_status}={detail}"}
            )
            continue
        prior_status, reason = _derive_prior_niftis(
            case_dir,
            ds_dir,
            cid,
            args.zonal_source,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )
        if prior_status in ("failed", "conflict"):
            counts[prior_status] += 1
            failures.append({"case_id": cid, "reason": reason})
        elif prior_status == "dry_run":
            counts["dry_run"] = counts.get("dry_run", 0) + 1
        elif prior_status == "skipped":
            counts["skipped"] += 1
        else:
            counts["ok"] += 1
    channel_names = {suffix: name for suffix, name, _ in MRI_CHANNELS}
    channel_names.update({suffix: "noNorm" for suffix, _ in PRIOR_CHANNELS})
    _finalize_raw_dataset(
        "zonal",
        ds_dir=ds_dir,
        sel=sel,
        counts=counts,
        failures=failures,
        channel_names=channel_names,
        description=(
            "PI-CAI csPCa (T2W/ADC/HBV) + algorithm-derived PZ/TZ zonal membership channels "
            "([0,1], not calibrated probabilities or geometric occupancy; noNorm; "
            f"prior_source=zonal_{args.zonal_source})"
        ),
        required_suffixes=tuple(s for s, _, _ in MRI_CHANNELS)
        + tuple(s for s, _ in PRIOR_CHANNELS),
        subset_used=bool(args.cases),
        dry_run=args.dry_run,
        t0=t0,
        extra={"zonal_source": args.zonal_source},
    )


# --------------------------------------------------------------------------- subcommand: splits
def cmd_splits(args: argparse.Namespace) -> None:
    t0 = time.time()
    ds_name = f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    pre_dir = Path(args.preprocessed_root) / ds_name
    raw_dir = (
        Path(Path(args.nnunet_raw_root) / ds_name) if args.nnunet_raw_root else None
    )
    split_path = Path(args.split_json)
    manifest_path = Path(args.manifest)
    print(
        f"[splits] dataset={ds_name} split_json={split_path} manifest={manifest_path}"
    )

    doc = json.loads(split_path.read_text())
    sp = doc["splits"]
    for key in ("train", "validation"):
        if key not in sp:
            raise SystemExit(
                f"[splits] 划分文件缺少 splits.{key}（现有键: {list(sp.keys())}）"
            )
    df = pd.read_csv(manifest_path)
    if not df["study_id"].is_unique or not df["case_id"].is_unique:
        raise SystemExit("[splits] manifest 中 study_id / case_id 不唯一")
    m = df.assign(
        sid=df["study_id"].astype(str),
        cid=df["case_id"].astype(str),
        pid=df["patient_id"].astype(str),
    )
    if not ((m["pid"] + "_" + m["sid"]) == m["cid"]).all():
        raise SystemExit(
            "[splits] manifest 的 case_id 不等于 f'{patient_id}_{study_id}'"
        )

    train_studies = {str(s) for s in sp["train"]["studies"]}
    val_studies = {str(s) for s in sp["validation"]["studies"]}
    if train_studies & val_studies:
        raise SystemExit(
            f"[splits] train/validation study 有交集: {sorted(train_studies & val_studies)[:5]}"
        )
    train_cases = sorted(c for s, c in zip(m["sid"], m["cid"]) if s in train_studies)
    val_cases = sorted(c for s, c in zip(m["sid"], m["cid"]) if s in val_studies)
    all_cases = set(m["cid"])
    if (set(train_cases) | set(val_cases)) != all_cases:
        raise SystemExit("[splits] train ∪ validation 未覆盖 manifest 全部 case")
    # 多 study 患者不得跨侧
    side_of_study = {
        **{s: "train" for s in train_studies},
        **{s: "val" for s in val_studies},
    }
    for pid, grp in m.groupby("pid"):
        sides = {side_of_study.get(s) for s in set(grp["sid"])}
        if len(sides) != 1:
            raise SystemExit(
                f"[splits] 患者 {pid} 的多个 study 跨 train/val 两侧: {sides}"
            )

    splits_obj = [{"train": train_cases, "val": val_cases}]
    payload = (json.dumps(splits_obj, indent=2, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )

    targets: list[tuple[str, Path]] = []
    if pre_dir.is_dir():
        targets.append(("preprocessed", pre_dir / "splits_final.json"))
    else:
        print(
            f"[splits][WARN] preprocessed 数据集目录不存在，跳过（请先运行 nnU-Net plan/preprocess）: {pre_dir}"
        )
    if raw_dir is not None and raw_dir.is_dir() and not args.no_raw_copy:
        targets.append(("raw", raw_dir / "splits_final.json"))

    if args.dry_run:
        print(
            f"[splits][dry-run] fold0 train={len(train_cases)} val={len(val_cases)}；将写出 {len(targets)} 个目标"
        )
        for label, path in targets:
            print(f"    {label}: {path}")
        _finish_summary(
            "splits",
            t0,
            {"ok": 0, "skipped": 0, "failed": 0, "conflict": 0},
            pre_dir,
            {"dry_run": True},
        )
        return
    if not targets:
        raise SystemExit("[splits] 没有可写目标（preprocessed/raw 数据集目录均不存在）")

    counts = {"ok": 0, "skipped": 0, "failed": 0, "conflict": 0}
    for label, path in targets:
        if path.exists() and not args.overwrite:
            try:
                old = json.loads(path.read_text())
            except Exception:  # noqa: BLE001
                old = None
            if old == splits_obj:
                counts["skipped"] += 1
                print(f"[splits] 跳过（内容一致）: {path}")
                continue
            counts["conflict"] += 1
            print(f"[splits] 目标已存在且内容不一致: {path}（加 --overwrite 强制重建）")
            continue
        _atomic_write_bytes(path, payload)
        counts["ok"] += 1
        print(f"[splits] 写出: {path}")
    _finish_summary(
        "splits",
        t0,
        counts,
        pre_dir,
        {"fold0_train": len(train_cases), "fold0_val": len(val_cases)},
    )
    if counts["conflict"] or counts["failed"]:
        # fail-closed：splits_final.json 冲突必须非零退出（不静默沿用不一致的既有文件）
        print("[splits] fail-closed：存在冲突/失败，未静默沿用既有 splits_final.json")
        raise SystemExit(2)


# --------------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="统一的 PI-CAI -> nnU-Net 数据准备入口（baseline / zonal / splits）"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--materialized-root", default=str(DEFAULT_MATERIALIZED))
        p.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
        p.add_argument(
            "--nnunet-raw-root",
            default=os.environ.get("nnUNet_raw", str(DEFAULT_RAW_ROOT)),
        )
        p.add_argument("--cases", default="", help="逗号分隔 case_id 子集（默认全部）")
        p.add_argument("--dry-run", action="store_true")
        p.add_argument("--resume", action="store_true")
        p.add_argument("--overwrite", action="store_true")
        p.add_argument("--no-progress", action="store_true")

    p_base = sub.add_parser(
        "baseline", help="组织 Dataset605_PICAI（3 通道）raw 数据集"
    )
    add_common(p_base)
    p_base.add_argument("--dataset-id", type=int, default=605)
    p_base.add_argument("--dataset-name", default="PICAI")
    p_base.set_defaults(func=cmd_baseline)

    p_zonal = sub.add_parser(
        "zonal", help="组织 Dataset606_PICAI_Zonal（5 通道，含 PZ/TZ）raw 数据集"
    )
    add_common(p_zonal)
    p_zonal.add_argument("--dataset-id", type=int, default=606)
    p_zonal.add_argument("--dataset-name", default="PICAI_Zonal")
    p_zonal.add_argument(
        "--zonal-source",
        required=True,
        choices=list(ZONAL_SOURCES),
        help="分区标签来源（yuan/hevi，必须显式提供；不同来源不得静默混用）",
    )
    p_zonal.set_defaults(func=cmd_zonal)

    p_split = sub.add_parser(
        "splits", help="把冻结划分转换为某数据集的 splits_final.json（单 fold）"
    )
    p_split.add_argument("--split-json", default=str(DEFAULT_SPLIT_JSON))
    p_split.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    p_split.add_argument(
        "--preprocessed-root",
        default=os.environ.get("nnUNet_preprocessed", str(DEFAULT_PREPROCESSED_ROOT)),
    )
    p_split.add_argument(
        "--nnunet-raw-root", default=os.environ.get("nnUNet_raw", str(DEFAULT_RAW_ROOT))
    )
    p_split.add_argument("--dataset-id", type=int, required=True)
    p_split.add_argument("--dataset-name", required=True)
    p_split.add_argument("--no-raw-copy", action="store_true")
    p_split.add_argument("--dry-run", action="store_true")
    p_split.add_argument("--overwrite", action="store_true")
    p_split.set_defaults(func=cmd_splits)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if getattr(args, "resume", False) and getattr(args, "overwrite", False):
        raise SystemExit("[prepare] --resume 与 --overwrite 互斥")
    args.func(args)


if __name__ == "__main__":
    main()
