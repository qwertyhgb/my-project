#!/usr/bin/env python3
"""由**预测** WG 先验为 lesion 数据集构建 Anatomy-Guided ROI 集合（条件 B/C/D/E 的前置）。

它做什么
--------
对每个训练 split 病例：读取该病例自身 T2W 的 predicted ``P(WG)``（来自冻结的 Stage-1
anatomy 模型），按 ``wg_threshold`` 阈值化、取 bounding box，再按 ``margin_mm`` 物理扩张，
写出 ``surgery`` 无关的纯几何 ROI 集合 JSON。

它**不**做什么（硬约束）
------------------------
- **不读** lesion GT（ROI 只能来自预测解剖先验；用 GT 会构成泄漏）；
- **不读**原始医学影像（只读先验目录里的 ``.npz`` 概率与 ``.pkl`` 物理元数据）；
- **不重采样**任何图像；
- **不生成** GT 解剖 ROI（``--prior-source`` 只允许 ``predicted_prior``）。

输入布局
--------
先验目录需含 nnU-Net 原生导出的 ``<case>.npz``（键 ``probabilities``，形状 ``(3,Z,Y,X)``，
通道顺序 ``WG / PZ / TZ``）与 ``<case>.pkl``（物理元数据）。见
:mod:`zonal_reliability_fusion.anatomy.dataset`。

用法（长任务；由研究者本人运行）::

    cd /opt/data/private/lm/my-projects
    conda activate lm && source scripts/env_nnunet.sh
    python scripts/data/build_anatomy_roi_set.py \
        --prior-dir workdir/anatomy_priors/Dataset607_PICAI_Anatomy/validation \
        --lesion-dataset Dataset605_PICAI --fold 0 \
        --margin-mm 15 --wg-threshold 0.5 \
        --output workdir/anatomy_rois/Dataset605_PICAI_fold0.json

进度：逐例 tqdm，并在结束打印 成功/失败/跳过/耗时/输出路径。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_PROJECT_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(_PROJECT_SRC))


def _load_probabilities(npz_path: Path):
    import numpy as np

    with np.load(npz_path, allow_pickle=False) as archive:
        if archive.files != ["probabilities"]:
            raise ValueError(f"{npz_path}: 期望唯一键 'probabilities'，实际 {archive.files}")
        return archive["probabilities"]


def _load_spacing_zyx(pkl_path: Path) -> tuple[float, float, float]:
    import pickle

    with pkl_path.open("rb") as stream:
        properties = pickle.load(stream)  # 仅本地原生产物
    spacing = properties.get("spacing")
    if not isinstance(spacing, (list, tuple)) or len(spacing) != 3:
        raise ValueError(f"{pkl_path}: 缺少合法的 spacing 元数据")
    return tuple(float(s) for s in spacing)  # type: ignore[return-value]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "由预测 WG 先验构建 Anatomy-Guided ROI 集合（不读 lesion GT、不读原始影像、不重采样）"
        )
    )
    parser.add_argument(
        "--prior-dir", required=True, help="预测 WG 先验目录（含 <case>.npz 与 <case>.pkl）"
    )
    parser.add_argument(
        "--lesion-dataset", required=True, help="目标 lesion 数据集名称，例如 Dataset605_PICAI"
    )
    parser.add_argument("--fold", type=int, default=0, help="fold（默认 0）")
    parser.add_argument(
        "--split-file",
        default=None,
        help="lesion 数据集 splits_final.json；默认取 <nnUNet_preprocessed>/<dataset>/splits_final.json",
    )
    parser.add_argument("--margin-mm", type=float, default=15.0, help="ROI 物理安全边界（mm）")
    parser.add_argument(
        "--wg-threshold", type=float, default=0.5, help="P(WG) 阈值（soft -> 二值，仅用于取 bbox）"
    )
    parser.add_argument(
        "--empty-fallback",
        choices=("full_fov", "skip"),
        default="full_fov",
        help=(
            "WG 预测为空时的行为：full_fov = 回退全视野并记录原因（安全，默认）；"
            "skip = 跳过该病例（会出现缺失 ROI，训练入口会拒绝启动）"
        ),
    )
    parser.add_argument("--output", required=True, help="输出 ROI 集合 JSON（已存在则拒绝覆盖）")
    parser.add_argument("--no-progress", action="store_true", help="关闭逐例进度条")
    return parser


def main(argv=None) -> int:
    # 先解析参数：``--help`` 必须能在**没有** source env_nnunet.sh 的 shell 里工作，
    # 因此 nnunetv2 / 项目包的导入放在参数解析之后。
    args = build_parser().parse_args(argv)

    from nnunetv2.paths import nnUNet_preprocessed
    from tqdm import tqdm

    from zonal_reliability_fusion.lesion.roi import build_prostate_roi
    from zonal_reliability_fusion.nnunet.roi_sampling import (
        ROI_SET_SCHEMA_VERSION,
        load_prostate_roi_set,
    )

    started = time.monotonic()

    dataset_dir = Path(nnUNet_preprocessed) / args.lesion_dataset
    split_file = (
        Path(args.split_file) if args.split_file else dataset_dir / "splits_final.json"
    )
    if not split_file.is_file():
        raise SystemExit(f"缺少 lesion split 文件：{split_file}")
    splits = json.loads(split_file.read_text(encoding="utf-8"))
    if not isinstance(splits, list) or not 0 <= args.fold < len(splits):
        raise SystemExit(f"fold {args.fold} 超出范围（split 文件含 {len(splits)} 个 fold）")
    train_cases = [str(c) for c in splits[args.fold].get("train", [])]
    val_cases = [str(c) for c in splits[args.fold].get("val", [])]
    if not train_cases or not val_cases:
        raise SystemExit("split 的 train 或 val 为空，拒绝继续")

    prior_dir = Path(args.prior_dir)
    if not prior_dir.is_dir():
        raise SystemExit(f"先验目录不存在：{prior_dir}")

    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"输出已存在，拒绝静默覆盖：{output}")

    cases: dict[str, dict] = {}
    errors: list[str] = []
    skipped: list[str] = []
    fallbacks: list[str] = []
    for case_id in tqdm(
        train_cases, desc="build anatomy ROI", unit="case", disable=args.no_progress
    ):
        npz_path, pkl_path = prior_dir / f"{case_id}.npz", prior_dir / f"{case_id}.pkl"
        if not npz_path.is_file() or not pkl_path.is_file():
            errors.append(f"{case_id}: 先验缺失（{npz_path.name} / {pkl_path.name}）")
            continue
        try:
            probabilities = _load_probabilities(npz_path)
            if probabilities.ndim != 4 or probabilities.shape[0] != 3:
                raise ValueError(
                    f"期望 (3,Z,Y,X) 概率数组，实际 {probabilities.shape}"
                )
            spacing = _load_spacing_zyx(pkl_path)
            roi = build_prostate_roi(
                probabilities[0],  # 通道顺序 WG / PZ / TZ
                spacing,
                margin_mm=args.margin_mm,
                wg_threshold=args.wg_threshold,
                prior_source="predicted_prior",
                empty_fallback="error" if args.empty_fallback == "skip" else "full_fov",
            )
        except Exception as exc:  # noqa: BLE001 - 逐例收集后统一汇总
            if args.empty_fallback == "skip" and "empty WG prediction" in str(exc):
                skipped.append(case_id)
                continue
            errors.append(f"{case_id}: {type(exc).__name__}: {exc}")
            continue
        if roi.fallback_reason is not None:
            fallbacks.append(f"{case_id}: {roi.fallback_reason}")
        entry: dict = {
            "lower_zyx": list(roi.lower),
            "upper_zyx": list(roi.upper),
            "original_shape_zyx": list(roi.original_shape),
            "spacing_zyx_mm": list(roi.spacing_zyx),
            "was_full_fov_fallback": roi.fallback_reason is not None,
        }
        if roi.fallback_reason is not None:
            entry["fallback_reason"] = roi.fallback_reason
        cases[case_id] = entry

    payload = {
        "schema_version": ROI_SET_SCHEMA_VERSION,
        "dataset_name": args.lesion_dataset,
        "fold": args.fold,
        "definition": {
            "roi": "predicted WG 的二值 bbox + 物理 margin（不重采样、不做 hard mask 乘法）",
            "thresholds": {
                "margin_mm": args.margin_mm,
                "wg_threshold": args.wg_threshold,
            },
            "coordinate_space": "预处理数组轴序 (Z, Y, X)，与 nnU-Net 的 patch 采样一致",
            "margin_rounding": "ceil(margin_mm / spacing)，保证物理边界不小于承诺",
            "no_lesion_gt": "ROI 只由 predicted anatomy 推导；不读取 lesion GT",
        },
        "thresholds": {"margin_mm": args.margin_mm, "wg_threshold": args.wg_threshold},
        "split": {"train_cases": train_cases, "val_cases": val_cases},
        "provenance": {
            "prior_source": "predicted_prior",
            "prior_dir": str(prior_dir),
            "anatomy_model": (
                "Dataset607_PICAI_Anatomy / nnUNetTrainerPICAI_AnatomyJoint_100ep_NoFFT"
            ),
            "note": (
                "anatomy 预测错误是真实部署链条的一部分；本集合不做任何手工修正，"
                "也不允许把 GT 解剖标签作为 ROI 来源"
            ),
        },
        "cases": cases,
        "summary": {
            "requested_cases": len(train_cases),
            "roi_cases": len(cases),
            "failed_cases": len(errors),
            "skipped_cases": len(skipped),
            "full_fov_fallbacks": len(fallbacks),
        },
    }

    if errors:
        print("致命：以下病例无法生成 ROI（fail closed，不写出集合）：")
        for line in errors[:50]:
            print(f"  {line}")
        if len(errors) > 50:
            print(f"  ... 另有 {len(errors) - 50} 条未显示")
        print(
            f"[build-anatomy-roi] success={len(cases)} failed={len(errors)} "
            f"skipped={len(skipped)} elapsed={time.monotonic()-started:.2f}s output=(未写出)"
        )
        return 1

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(f".{output.name}.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, allow_nan=False)
        fh.write("\n")
    tmp.replace(output)

    # 自检：写出的文件必须能通过训练入口同一套校验
    load_prostate_roi_set(
        output, expected_dataset_name=args.lesion_dataset, expected_fold=args.fold
    )
    if fallbacks:
        print(f"提示：{len(fallbacks)} 例因 WG 预测为空/覆盖全幅而回退为全视野 ROI：")
        for line in fallbacks[:10]:
            print(f"  {line}")
        if len(fallbacks) > 10:
            print(f"  ... 另有 {len(fallbacks) - 10} 条未显示")
    print(
        f"[build-anatomy-roi] success={len(cases)} failed=0 skipped={len(skipped)} "
        f"elapsed={time.monotonic()-started:.2f}s output={output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())