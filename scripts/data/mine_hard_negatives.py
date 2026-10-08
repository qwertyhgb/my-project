#!/usr/bin/env python3
"""Round-2 困难负样本挖掘（新主线条件 E 的前置；**只在训练 split 内**执行）。

分轮流程中的位置
----------------
::

    Round 1  训练 strong baseline / coarse-to-fine 模型（得到 checkpoint）
    Round 2  只对**训练 split**做推理，挖掘高置信假阳作为困难负样本  <- 本脚本
    Round 3  训练时提高这些位置的采样概率（条件 E）

候选定义（预先冻结，见 :mod:`zonal_reliability_fusion.sampling.hard_negative`）
------------------------------------------------------------------------------
``prediction_probability >= confidence_threshold`` **AND** ``GT == background``
**AND** ``WG_probability >= wg_threshold``，取满足条件的假阳连通域质心（6-邻域），
按体积降序。

安全红线（fail closed）
-----------------------
- ``--split`` 只允许 ``train``：本脚本**拒绝**对 validation / test 病例挖掘；
- 训练 split 的病例集合必须与 lesion split 文件的 ``train`` 完全一致；
- 第一轮预测的病例集合必须**恰好等于**训练 split（多一例少一例都报错）；
- 不读取原始医学影像；WG 概率来自冻结 anatomy 模型的预测（``--wg-dir``）。

用法（长任务；由研究者本人运行）::

    cd /opt/data/private/lm/my-projects
    conda activate lm && source scripts/env_nnunet.sh

    # 1) 对训练 split 病例做第一轮推理（官方预测器，输出现概率）
    #    见 README「预测」小节：--save_probabilities，-o outputs/predictions/<run>_train/
    # 2) 由冻结 anatomy 模型对同一批训练病例生成 WG 先验
    # 3) 挖掘
    python scripts/data/mine_hard_negatives.py \
        --prediction-dir outputs/predictions/round1_train_t2wadchbv \
        --reference-dir  workdir/nnUNet_raw/<lesion dataset>/labelsTr \
        --wg-dir         workdir/anatomy_priors/<lesion dataset>/train \
        --lesion-dataset Dataset605_PICAI --fold 0 \
        --confidence-threshold 0.5 --wg-threshold 0.5 --max-locations-per-case 8 \
        --output workdir/hard_negatives/Dataset605_PICAI_fold0_round1.json

进度：逐例 tqdm，并在结束打印 成功/失败/跳过/耗时/输出路径与总位置数。
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


def _load_binary_mask(path: Path, expected_shape=None):
    """读取二值 NIfTI 掩膜（数组轴序 ZYX）；非 0/1 取值或几何无关的异常即报错。"""
    import numpy as np
    import SimpleITK as sitk

    if not path.is_file():
        raise FileNotFoundError(f"掩膜不存在：{path}")
    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image)
    if not bool(np.isfinite(array).all()):
        raise ValueError(f"{path}: 含非有限值")
    bad = [float(v) for v in np.unique(array) if float(v) not in (0.0, 1.0)]
    if bad:
        raise ValueError(f"{path}: 含非 0/1 取值 {bad[:5]}")
    mask = array > 0.5
    if expected_shape is not None and mask.shape != tuple(expected_shape):
        raise ValueError(f"{path}: 形状 {mask.shape} 与第一轮预测 {tuple(expected_shape)} 不一致")
    return mask


def _load_probability(path: Path):
    import numpy as np

    if not path.is_file():
        raise FileNotFoundError(f"概率文件不存在：{path}")
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            keys = list(archive.files)
            if "probabilities" in keys:
                array = archive["probabilities"]
            elif len(keys) == 1:
                array = archive[keys[0]]
            else:
                raise ValueError(f"{path}: 无法确定概率数组（键 {keys}）")
    else:
        import pickle

        with path.open("rb") as stream:
            array = pickle.load(stream)
    array = np.asarray(array)
    if array.ndim == 4 and array.shape[0] == 2:
        return array[1]  # 取前景通道
    if array.ndim == 3:
        return array
    raise ValueError(
        f"{path}: 概率数组形状 {array.shape} 不被支持（期望 (2,Z,Y,X) 或 (Z,Y,X)）"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Round-2 困难负样本挖掘（只在 training split 内；validation 绝不参与）"
    )
    parser.add_argument(
        "--prediction-dir",
        required=True,
        help="第一轮模型对**训练 split**病例的预测（含概率导出，例如 <case>.npz）",
    )
    parser.add_argument(
        "--reference-dir",
        required=True,
        help="训练 split 的 lesion GT 目录（labelsTr；只用训练病例的标签）",
    )
    parser.add_argument(
        "--wg-dir",
        required=True,
        help="训练 split 的 predicted P(WG) 目录（冻结 anatomy 模型的输出；禁止用 GT WG）",
    )
    parser.add_argument("--lesion-dataset", required=True, help="lesion 数据集名称")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument(
        "--split-file",
        default=None,
        help="lesion splits_final.json；默认 <nnUNet_preprocessed>/<dataset>/splits_final.json",
    )
    parser.add_argument(
        "--split",
        choices=("train",),
        default="train",
        help="只允许 train：validation / test 绝不参与挖掘",
    )
    parser.add_argument(
        "--confidence-threshold", type=float, default=0.5, help="第一轮阳性概率阈值"
    )
    parser.add_argument(
        "--wg-threshold", type=float, default=0.5, help="P(WG) 阈值（腺体内/近腺体判据）"
    )
    parser.add_argument(
        "--max-locations-per-case", type=int, default=8, help="每病例最多保留的位置数（按体积降序）"
    )
    parser.add_argument(
        "--min-component-voxels", type=int, default=1, help="候选连通域的最小体素数"
    )
    parser.add_argument("--output", required=True, help="输出困难负样本集合 JSON（拒绝覆盖）")
    parser.add_argument(
        "--reference-suffix", default="", help="GT 文件名后缀（例如 '' 或 '.nii.gz'）"
    )
    parser.add_argument("--no-progress", action="store_true", help="关闭逐例进度条")
    return parser


def main(argv=None) -> int:
    # 先解析参数：``--help`` 必须能在**没有** source env_nnunet.sh 的 shell 里工作。
    args = build_parser().parse_args(argv)

    import nnunetv2.paths  # noqa: F401  （在调用时读取 nnUNet_preprocessed，避免快照过期）
    from nnunetv2.paths import nnUNet_preprocessed

    from zonal_reliability_fusion.sampling.hard_negative import (
        build_hard_negative_set,
        load_hard_negative_set,
        write_hard_negative_set,
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

    # 第一轮预测必须恰好覆盖训练 split（多一例少一例都报错，避免把 val 病例卷进来）
    prediction_dir = Path(args.prediction_dir)
    if not prediction_dir.is_dir():
        raise SystemExit(f"第一轮预测目录不存在：{prediction_dir}")
    predicted = {p.name.split(".")[0] for p in prediction_dir.iterdir() if p.name.endswith(".npz")}
    if not predicted:
        predicted = {
            p.name[: -len(".nii.gz")]
            for p in prediction_dir.iterdir()
            if p.name.endswith(".nii.gz")
        }
    missing = set(train_cases) - predicted
    extra = predicted - set(train_cases)
    if missing or extra:
        raise SystemExit(
            "第一轮预测的病例集合必须恰好等于 training split："
            f"缺 {len(missing)} 例（例如 {sorted(missing)[:3]}），"
            f"多 {len(extra)} 例（例如 {sorted(extra)[:3]}）。"
            "多出的病例极可能来自 validation，拒绝继续。"
        )

    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"输出已存在，拒绝静默覆盖：{output}")

    reference_dir, wg_dir = Path(args.reference_dir), Path(args.wg_dir)
    for name, folder in (("reference-dir", reference_dir), ("wg-dir", wg_dir)):
        if not folder.is_dir():
            raise SystemExit(f"{name} 不存在：{folder}")

    def load_case_arrays(case_id: str):
        probability = _load_probability(prediction_dir / f"{case_id}.npz")
        reference = _load_binary_mask(
            reference_dir / f"{case_id}{args.reference_suffix or '.nii.gz'}", probability.shape
        )
        wg = _load_probability(wg_dir / f"{case_id}.npz")
        if wg.shape != probability.shape:
            raise ValueError(
                f"{case_id}: WG 先验形状 {wg.shape} 与第一轮预测 {probability.shape} 不一致"
            )
        return probability, reference, wg

    payload = build_hard_negative_set(
        train_cases,
        load_case_arrays,
        dataset_name=args.lesion_dataset,
        fold=args.fold,
        train_cases=train_cases,
        val_cases=val_cases,
        provenance={
            "prediction_dir": str(prediction_dir),
            "reference_dir": str(reference_dir),
            "wg_dir": str(wg_dir),
            "prior_source": "predicted_prior",
            "split_used": "train",
            "note": (
                "挖掘只在 training split 内执行；validation 的预测与标签从未被读取；"
                "WG 来自冻结 anatomy 模型的预测，不是 GT 解剖标签"
            ),
        },
        confidence_threshold=args.confidence_threshold,
        wg_threshold=args.wg_threshold,
        max_locations_per_case=args.max_locations_per_case,
        min_component_voxels=args.min_component_voxels,
        progress=not args.no_progress,
    )
    written = write_hard_negative_set(output, payload)
    # 自检：写出的文件必须能通过训练入口同一套校验
    loaded = load_hard_negative_set(
        written, expected_dataset_name=args.lesion_dataset, expected_fold=args.fold
    )
    summary = loaded["summary"]
    print(
        f"[mine-hard-negatives] success={summary['cases_with_hard_negatives']} "
        f"failed=0 skipped={summary['mined_cases'] - summary['cases_with_hard_negatives']} "
        f"locations={summary['total_locations']} elapsed={time.monotonic()-started:.2f}s "
        f"output={written}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())