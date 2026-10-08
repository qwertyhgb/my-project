#!/usr/bin/env python3
"""训练 / 验证 split 泄漏自动检查（新主线要求；遇到任何泄漏 fail closed）。

检查对象
--------
跨 Stage 1（解剖先验）与 Stage 2（lesion）的全部 split 相关产物：

1. anatomy model 的 train IDs（``Dataset607_PICAI_Anatomy/splits_final.json`` + provenance 契约）；
2. anatomy model 的 validation IDs；
3. lesion model 的 train IDs（``Dataset605/606`` 的 ``splits_final.json``）；
4. lesion model 的 validation IDs；
5. 生成的 anatomy 预测（先验目录里的病例集合必须**恰好等于**该 split 的 val 病例）；
6. 困难负样本挖掘集合（病例必须**全部来自** lesion train split）。

判定与退出码
------------
- 输出 ``PASS`` / ``FAIL`` 与逐项明细；
- 任一项 FAIL 即非零退出（fail closed）；
- 缺文件（尚未生成的产物）记为 ``SKIP``，**不**算通过也不是失败；但 ``--require``
  可以把它升级为 FAIL，用于「必须已有先验才能开始下一阶段」的关卡。

用法（只读检查，秒级）::

    cd /opt/data/private/lm/my-projects
    conda activate lm && source scripts/env_nnunet.sh
    python scripts/data/check_split_integrity.py
    python scripts/data/check_split_integrity.py --anatomy-prior-dir workdir/anatomy_priors/Dataset605_PICAI/prototype_prior
    python scripts/data/check_split_integrity.py --hard-negative-set workdir/hard_negatives/hn_fold0.json

本脚本**只读**，不创建、不移动、不删除任何文件，也不读取原始医学图像。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(_PROJECT_SRC))

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"


class CheckResult:
    """单项检查的结果（名称 + 判定 + 明细）。"""

    def __init__(self, name: str, status: str, detail: str) -> None:
        self.name = name
        self.status = status
        self.detail = detail

    def render(self) -> str:
        return f"[{self.status:4s}] {self.name}: {self.detail}"


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _patient_of(case_id: str) -> str:
    return str(case_id).split("_")[0]


def check_split_pair(
    name: str, train: set[str], val: set[str], expected_all: set[str] | None = None
) -> CheckResult:
    """一对 train/val 病例集合自身的完整性（重叠 / 患者级重叠 / 覆盖）。"""
    if not train:
        return CheckResult(name, FAIL, "train 集合为空")
    if not val:
        return CheckResult(name, FAIL, "val 集合为空")
    overlap = train & val
    if overlap:
        return CheckResult(
            name,
            FAIL,
            f"train/val 病例重叠 {len(overlap)} 例（例如 {sorted(overlap)[:3]}）",
        )
    patient_overlap = {_patient_of(c) for c in train} & {_patient_of(c) for c in val}
    if patient_overlap:
        return CheckResult(
            name,
            FAIL,
            f"train/val 患者级重叠 {len(patient_overlap)} 个患者"
            f"（例如 {sorted(patient_overlap)[:3]}）",
        )
    if expected_all is not None:
        missing = expected_all - (train | val)
        extra = (train | val) - expected_all
        if missing or extra:
            return CheckResult(
                name,
                FAIL,
                f"与预期病例集合不一致：缺 {len(missing)} 例，多 {len(extra)} 例",
            )
    return CheckResult(
        name, PASS, f"train={len(train)} / val={len(val)}，无病例级或患者级重叠"
    )


def check_nnunet_split_file(
    name: str, path: Path, expected_total: int | None, fold: int
) -> tuple[CheckResult, set[str], set[str]]:
    """读取 nnU-Net ``splits_final.json`` 并检查该 fold 的 train/val。"""
    if not path.is_file():
        return CheckResult(name, SKIP, f"split 文件不存在：{path}"), set(), set()
    try:
        splits = _load_json(path)
    except json.JSONDecodeError as exc:
        return CheckResult(name, FAIL, f"split 文件不是合法 JSON：{exc}"), set(), set()
    if not isinstance(splits, list) or not splits:
        return CheckResult(name, FAIL, "splits_final.json 不是非空数组"), set(), set()
    if not 0 <= fold < len(splits):
        return CheckResult(name, FAIL, f"fold {fold} 超出范围（共 {len(splits)} fold）"), set(), set()
    train = {str(c) for c in splits[fold].get("train", [])}
    val = {str(c) for c in splits[fold].get("val", [])}
    expected = None
    if expected_total is not None:
        if len(train) + len(val) != expected_total:
            return (
                CheckResult(
                    name,
                    FAIL,
                    f"train+val={len(train) + len(val)} 与 dataset.json 的 numTraining="
                    f"{expected_total} 不一致",
                ),
                train,
                val,
            )
    result = check_split_pair(name, train, val, expected)
    return result, train, val


def check_anatomy_split_contract(
    name: str, anatomy_dir: Path, fold: int
) -> tuple[CheckResult, set[str], set[str]]:
    """anatomy 数据集：provenance 契约与 ``splits_final.json`` 必须逐例一致。"""
    dataset_json_path = anatomy_dir / "dataset.json"
    if not dataset_json_path.is_file():
        return CheckResult(name, SKIP, f"anatomy dataset.json 不存在：{dataset_json_path}"), set(), set()
    try:
        dataset_json = _load_json(dataset_json_path)
    except json.JSONDecodeError as exc:
        return CheckResult(name, FAIL, f"anatomy dataset.json 非法：{exc}"), set(), set()
    result, train, val = check_nnunet_split_file(
        name, anatomy_dir / "splits_final.json", dataset_json.get("numTraining"), fold
    )
    if result.status != PASS:
        return result, train, val
    contract = dataset_json.get("anatomy_contract", {})
    contract_train = {str(c) for c in contract.get("train_cases", [])}
    contract_val = {str(c) for c in contract.get("val_cases", [])}
    if not contract_train or not contract_val:
        return (
            CheckResult(name, FAIL, "anatomy_contract 缺少 train_cases / val_cases"),
            train,
            val,
        )
    if train != contract_train or val != contract_val:
        return (
            CheckResult(
                name,
                FAIL,
                "splits_final.json 与 anatomy_contract 的 train/val 归属不一致："
                "provenance 契约是唯一真源",
            ),
            train,
            val,
        )
    known_missing = contract.get("explicit_exclusions", [])
    if set(known_missing) & (train | val):
        return (
            CheckResult(
                name,
                FAIL,
                f"已知缺失病例出现在 split 中：{known_missing}（不得以空掩膜顶替）",
            ),
            train,
            val,
        )
    return (
        CheckResult(
            name,
            PASS,
            f"train={len(train)} / val={len(val)}，与 anatomy_contract 逐例一致",
        ),
        train,
        val,
    )


def check_anatomy_prior_directory(
    name: str, prior_dir: Path, val_cases: set[str], lesion_dataset_name: str | None
) -> CheckResult:
    """生成解剖先验的病例集合必须**恰好等于**该 split 的 val 病例。

    少于 val：说明还有病例没有先验，后续 lesion 实验会缺输入；
    多于 val：说明先验目录混入了 train 病例（或别的 split / 别的数据集），是泄漏风险。
    """
    if not prior_dir.exists():
        return CheckResult(name, SKIP, f"先验目录不存在：{prior_dir}")
    npz = {p.name[: -len(".npz")] for p in prior_dir.glob("*.npz")}
    if not npz:
        return CheckResult(name, SKIP, f"先验目录中没有 .npz：{prior_dir}")
    if not val_cases:
        return CheckResult(name, FAIL, "缺少可比较的 val 病例集合（anatomy split 未就绪）")
    missing = val_cases - npz
    extra = npz - val_cases
    if missing or extra:
        return CheckResult(
            name,
            FAIL,
            f"先验病例集合与 val split 不一致：缺 {len(missing)} 例"
            f"（例如 {sorted(missing)[:3]}），多 {len(extra)} 例（例如 {sorted(extra)[:3]}）；"
            "先验只能来自该病例自身 MRI 的预测，且集合必须与 split 精确对应",
        )
    return CheckResult(
        name,
        PASS,
        f"{len(npz)} 例先验，与 val split 精确一致（lesion dataset={lesion_dataset_name}）",
    )


def check_hard_negative_set(
    name: str, path: Path, lesion_train: set[str], lesion_val: set[str]
) -> CheckResult:
    """困难负样本挖掘病例必须全部来自 lesion train split。"""
    if not path.is_file():
        return CheckResult(name, SKIP, f"困难负样本集合不存在：{path}")
    try:
        payload = _load_json(path)
        from zonal_reliability_fusion.sampling.hard_negative import validate_mining_scope

        validate_mining_scope(payload, train_cases=lesion_train, val_cases=lesion_val)
    except Exception as exc:  # noqa: BLE001 - 任何违规都是 FAIL
        return CheckResult(name, FAIL, f"{type(exc).__name__}: {exc}")
    cases = payload.get("cases", {})
    return CheckResult(
        name,
        PASS,
        f"{len(cases)} 例挖掘病例全部属于 training split，"
        f"与 validation 无交集（locations={payload.get('summary', {}).get('total_locations')}）",
    )


def check_roi_set(
    name: str, path: Path, lesion_train: set[str], lesion_val: set[str]
) -> CheckResult:
    """ROI 集合必须只覆盖 lesion train split，且先验来源不得是 ORACLE_GT。"""
    if not path.is_file():
        return CheckResult(name, SKIP, f"ROI 集合不存在：{path}")
    try:
        payload = _load_json(path)
    except json.JSONDecodeError as exc:
        return CheckResult(name, FAIL, f"ROI 集合非法 JSON：{exc}")
    cases = {str(c) for c in payload.get("cases", {})}
    leaked = cases & lesion_val
    if leaked:
        return CheckResult(
            name,
            FAIL,
            f"ROI 集合包含 {len(leaked)} 个 validation 病例（例如 {sorted(leaked)[:3]}）",
        )
    outside = cases - lesion_train
    if outside:
        return CheckResult(
            name,
            FAIL,
            f"ROI 集合包含 {len(outside)} 个不在 training split 的病例"
            f"（例如 {sorted(outside)[:3]}）",
        )
    prior_source = payload.get("provenance", {}).get("prior_source")
    if prior_source == "ORACLE_GT":
        return CheckResult(
            name,
            FAIL,
            "provenance.prior_source == ORACLE_GT：GT 解剖标签只允许用于上界分析，"
            "禁止作为正式 lesion 实验的 ROI 来源",
        )
    if not cases:
        return CheckResult(name, FAIL, "ROI 集合为空")
    return CheckResult(
        name, PASS, f"{len(cases)} 例 ROI，全部属于 training split；prior_source={prior_source!r}"
    )


def main(argv=None) -> int:
    from nnunetv2.paths import nnUNet_preprocessed, nnUNet_results

    parser = argparse.ArgumentParser(
        description="训练/验证 split 泄漏自动检查（只读；任何泄漏 fail closed）",
    )
    parser.add_argument("--fold", type=int, default=0, help="要检查的 fold（默认 0）")
    parser.add_argument(
        "--lesion-dataset",
        default="Dataset605_PICAI",
        help="lesion 数据集名称（默认 Dataset605_PICAI）",
    )
    parser.add_argument(
        "--anatomy-dataset",
        default="Dataset607_PICAI_Anatomy",
        help="Stage-1 解剖数据集名称",
    )
    parser.add_argument(
        "--anatomy-prior-dir",
        default=None,
        help="生成的解剖先验目录（含 *.npz）；不传则 SKIP 该项",
    )
    parser.add_argument(
        "--hard-negative-set", default=None, help="困难负样本集合 JSON；不传则 SKIP"
    )
    parser.add_argument("--roi-set", default=None, help="ROI 集合 JSON；不传则 SKIP")
    parser.add_argument(
        "--require",
        default="",
        help=(
            "把指定检查项名称（逗号分隔）的 SKIP 升级为 FAIL，用于阶段关卡；"
            "例如 --require anatomy_split,lesion_split"
        ),
    )
    args = parser.parse_args(argv)

    preprocessed = Path(nnUNet_preprocessed)
    required = {name.strip() for name in args.require.split(",") if name.strip()}
    results: list[CheckResult] = []

    anatomy_dir = preprocessed / args.anatomy_dataset
    anatomy_result, anatomy_train, anatomy_val = check_anatomy_split_contract(
        "anatomy_split", anatomy_dir, args.fold
    )
    results.append(anatomy_result)

    lesion_dir = preprocessed / args.lesion_dataset
    lesion_total = None
    lesion_dataset_json = lesion_dir / "dataset.json"
    if lesion_dataset_json.is_file():
        lesion_total = _load_json(lesion_dataset_json).get("numTraining")
    lesion_result, lesion_train, lesion_val = check_nnunet_split_file(
        "lesion_split", lesion_dir / "splits_final.json", lesion_total, args.fold
    )
    results.append(lesion_result)

    # 两个 stage 之间的交叉检查：anatomy val 与 lesion split 的关系必须明确
    if anatomy_train or anatomy_val:
        if lesion_train or lesion_val:
            cross = (anatomy_train | anatomy_val) & (lesion_train | lesion_val)
            results.append(
                CheckResult(
                    "stage_cross_scope",
                    PASS if cross else SKIP,
                    f"anatomy 与 lesion 病例集合的交集大小 = {len(cross)}"
                    "（两个 stage 覆盖同一批 study 是预期的；本项只记录事实，不做 pass/fail 判定）",
                )
            )

    if args.anatomy_prior_dir:
        results.append(
            check_anatomy_prior_directory(
                "anatomy_prior_cases",
                Path(args.anatomy_prior_dir),
                anatomy_val,
                args.lesion_dataset,
            )
        )
    else:
        results.append(
            CheckResult(
                "anatomy_prior_cases",
                SKIP,
                "未提供 --anatomy-prior-dir；无法检查先验病例集合",
            )
        )

    results.append(
        check_hard_negative_set(
            "hard_negative_scope",
            Path(args.hard_negative_set) if args.hard_negative_set else Path("__absent__"),
            lesion_train,
            lesion_val,
        )
    )
    results.append(
        check_roi_set(
            "roi_set_scope",
            Path(args.roi_set) if args.roi_set else Path("__absent__"),
            lesion_train,
            lesion_val,
        )
    )

    for result in results:
        if result.name in required and result.status == SKIP:
            result.status = FAIL
            result.detail = f"该检查项被 --require 升级为必须通过；{result.detail}"

    print("=== split integrity check ===")
    for result in results:
        print(result.render())
    failed = [r for r in results if r.status == FAIL]
    passed = [r for r in results if r.status == PASS]
    skipped = [r for r in results if r.status == SKIP]
    print(
        f"[split-integrity] verdict={'FAIL' if failed else 'PASS'} "
        f"passed={len(passed)} failed={len(failed)} skipped={len(skipped)} "
        f"preprocessed={preprocessed} results={nnUNet_results}"
    )
    if failed:
        print("任何泄漏都必须先修复再启动训练（fail closed）。")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())