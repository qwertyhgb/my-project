#!/usr/bin/env python
"""把 PZ/TZ 分区标签物化为 **plan-space** sidecar（M3/M4 的 prior 输入）。

用法（必须由研究者执行；需先 `source scripts/env_nnunet.sh`）：

    python scripts/data/materialize_zonal_prior.py --source yuan --dry-run --limit-cases 3
    python scripts/data/materialize_zonal_prior.py --source yuan --resume

关键保证：

1. **几何完全复用 nnU-Net v2.6.2**：transpose → crop(properties 中记录的图像 nonzero bbox) →
   `configuration_manager.resampling_fn_seg`（最近邻）；不重新发明任何几何逻辑。
2. **病灶 oracle fail-closed**：同一转换应用于原始 lesion，必须与 `.b2nd` 中 seg 逐体素一致；
   materializer 在写盘前**再强制校验一次** oracle 字段，不一致绝不写文件。
3. **不修改既有数据**：只写 `<out_root>/<case_id>.npz` + `.json` + `manifest.json`（+ summary），
   绝不触碰 `.b2nd` / raw / split / plan / third_party。
4. **原子写入 + 安全 resume**：所有产物先写临时文件再 `os.replace`；`--resume` 先做轻量身份检查
   （source/plan/config/输入哈希/数组哈希），匹配才跳过昂贵转换；损坏或半成品必须失败或显式重算。
5. **数据集级 manifest**：排序记录全部 case_id + 输入/输出哈希 + metadata hash，给出 canonical
   `manifest_sha256`；完整运行（未限定 case 子集）时必须覆盖 train∪val，否则退出码非 0。
6. **性能**：`nnUNetDatasetBlosc2` 只在首次需要时构造一次并复用；seg 按需读取（不物化三通道影像）。
7. `--dry-run` **名实相符**：对选中病例执行完整只读转换 + oracle + prior 契约 + 元数据构造，
   但零文件写入；任一失败退出码非 0；未指定子集时要求显式 `--case-ids` 或正数 `--limit-cases`。
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from zonal_reliability_fusion.data.zonal_prior import (
    ZONAL_SOURCES,
    ZonalPriorError,
    ZonalPriorMaterializer,
    array_sha256,
    convert_label_to_plan_space,
    file_sha256,
    geometry_from_properties,
    label_to_pz_tz,
    oracle_check_lesion,
    validate_prior_array,
)

DEFAULT_CASES_ROOT = PROJECT_ROOT / "data/processed/picai/cases"
DEFAULT_PREPROCESSED = PROJECT_ROOT / "workdir/nnUNet_preprocessed/Dataset605_PICAI"
DEFAULT_PLANS = DEFAULT_PREPROCESSED / "nnUNetPlans.json"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="物化 PZ/TZ plan-space prior sidecar（只读原始数据 + 只写新派生目录）")
    ap.add_argument("--source", required=True, choices=list(ZONAL_SOURCES), help="分区标签来源（必填，不得默认）")
    ap.add_argument("--plans", default=str(DEFAULT_PLANS), help="冻结 nnUNetPlans.json")
    ap.add_argument("--preprocessed-dataset", default=str(DEFAULT_PREPROCESSED), help="Dataset605_PICAI 预处理目录")
    ap.add_argument("--configuration", default="3d_fullres", help="plan 配置名")
    ap.add_argument("--splits", default="", help="splits_final.json（默认取预处理目录内同名文件；用于覆盖范围校验）")
    ap.add_argument("--fold", type=int, default=0, help="覆盖范围校验使用的 fold")
    ap.add_argument("--cases-root", default=str(DEFAULT_CASES_ROOT), help="物化病例目录（含 zonal_*/lesion.nii.gz）")
    ap.add_argument("--out-root", default="", help="输出派生目录（默认 data/processed/picai_zonal_<source>_v1）")
    ap.add_argument("--case-ids", default="", help="逗号分隔的 case_id 子集（默认预处理目录下的全部病例）")
    ap.add_argument("--limit-cases", type=int, default=0, help="只处理前 N 例（0 = 全部）")
    ap.add_argument("--resume", action="store_true", help="跳过已完成且身份一致的病例（不做昂贵转换）")
    ap.add_argument("--overwrite", action="store_true", help="允许覆盖已存在的 sidecar（默认拒绝）")
    ap.add_argument("--no-progress", action="store_true", help="关闭 tqdm 进度条")
    ap.add_argument("--dry-run", action="store_true", help="完整只读转换 + oracle 检查，但不写任何文件")
    return ap.parse_args()


def build_context(args: argparse.Namespace) -> dict[str, Any]:
    """加载 plan / 读取器（惰性 import nnunetv2；仅在脚本运行时需要）。"""
    try:
        from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "无法 import nnunetv2：请先 `source scripts/env_nnunet.sh`（固定 v2.6.2 源码树）"
        ) from exc

    plans_path = Path(args.plans)
    if not plans_path.is_file():
        raise SystemExit(f"plans 文件不存在: {plans_path}")
    doc = json.loads(plans_path.read_text())
    plans_manager = PlansManager(doc)
    configuration_manager = plans_manager.get_configuration(args.configuration)
    # nnU-Net 2.6.2 的 image_reader_writer_class 是 @property（返回 reader **类**）：
    # 调用一次得到 SimpleITKIO 实例；写成 ()() 会在第二次调用实例时
    # 抛 TypeError: 'SimpleITKIO' object is not callable
    reader = plans_manager.image_reader_writer_class()
    return {
        "plans_manager": plans_manager,
        "configuration_manager": configuration_manager,
        "reader": reader,
        "plans_path": plans_path,
        "plans_sha256": file_sha256(plans_path),
        "transpose_forward": tuple(int(i) for i in plans_manager.transpose_forward),
        "target_spacing": tuple(float(v) for v in configuration_manager.spacing),
        "resample_fn_seg": configuration_manager.resampling_fn_seg,
    }


def list_case_ids(args: argparse.Namespace, context: dict[str, Any]) -> list[str]:
    if args.case_ids:
        return [c.strip() for c in args.case_ids.split(",") if c.strip()]
    data_dir = Path(args.preprocessed_dataset) / context["configuration_manager"].data_identifier
    if not data_dir.is_dir():
        raise SystemExit(f"预处理目录不存在: {data_dir}")
    ids = sorted(p.name[: -len("_seg.b2nd")] for p in data_dir.glob("*_seg.b2nd"))
    if not ids:
        raise SystemExit(f"{data_dir} 下没有 *_seg.b2nd")
    if args.limit_cases:
        ids = ids[: int(args.limit_cases)]
    return ids


def read_split_case_ids(args: argparse.Namespace) -> list[str]:
    """train∪val（fold 0）——完整运行的覆盖范围基准。"""
    splits_path = Path(args.splits) if args.splits else Path(args.preprocessed_dataset) / "splits_final.json"
    if not splits_path.is_file():
        raise SystemExit(f"split 文件不存在（无法校验覆盖范围）: {splits_path}")
    folds = json.loads(splits_path.read_text())
    if not isinstance(folds, list) or len(folds) <= int(args.fold):
        raise SystemExit(f"{splits_path} 不是合法的 fold 列表（fold={args.fold}）")
    fold_doc = folds[int(args.fold)]
    return sorted({str(c) for c in fold_doc["train"]} | {str(c) for c in fold_doc["val"]})


class CaseConverter:
    """单病例转换器：复用同一个 nnUNetDatasetBlosc2 实例，seg 按需读取。"""

    def __init__(self, args: argparse.Namespace, context: dict[str, Any]) -> None:
        self.args = args
        self.context = context
        self.cases_root = Path(args.cases_root)
        self.data_dir = Path(args.preprocessed_dataset) / context["configuration_manager"].data_identifier
        self._dataset: Any = None
        self.dataset_constructed = 0  # 供测试断言「N 例只构造一次」

    # ---------------------------------------------------------------- 资源
    def dataset(self):
        """惰性构造并复用（避免每例重建 + 重复扫描 1500 个 identifier）。"""
        if self._dataset is None:
            from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2

            self._dataset = nnUNetDatasetBlosc2(folder=str(self.data_dir))
            self.dataset_constructed += 1
        return self._dataset

    def _read_seg_only(self, case_id: str):
        """只解压 seg；`data` 是官方 lazy blosc2 句柄，**不物化三通道影像**。"""
        data, seg, seg_prev, _props = self.dataset()[case_id]
        try:
            seg_arr = __import__("numpy").asarray(seg)
        finally:
            del data, seg_prev
        return seg_arr

    def _read_volume(self, path: Path):
        try:
            arr, _ = self.context["reader"].read_seg(str(path))
        except Exception as exc:
            raise ZonalPriorError(f"读取 {path.name} 失败: {type(exc).__name__}: {exc}") from exc
        return arr

    # ---------------------------------------------------------------- 身份 / 转换
    def case_input_hashes(self, case_id: str) -> dict[str, str]:
        case_dir = self.cases_root / case_id
        paths = {
            "zonal": case_dir / f"zonal_{self.args.source}.nii.gz",
            "lesion": case_dir / "lesion.nii.gz",
            "properties": self.data_dir / f"{case_id}.pkl",
        }
        for path in paths.values():
            if not path.exists():
                raise ZonalPriorError(f"缺少输入: {path}")
        return {name: file_sha256(path) for name, path in paths.items()}

    def identity_probe(self, case_id: str) -> dict[str, Any]:
        """resume 用轻量身份（只做文件哈希，不做 resample、不读 .b2nd）。"""
        return {
            "source": self.args.source,
            "configuration": self.args.configuration,
            "plans_sha256": self.context["plans_sha256"],
            "input_sha256": self.case_input_hashes(case_id),
        }

    def convert(self, case_id: str) -> tuple[Any, dict[str, Any]]:
        case_dir = self.cases_root / case_id
        zonal_path = case_dir / f"zonal_{self.args.source}.nii.gz"
        lesion_path = case_dir / "lesion.nii.gz"
        props_path = self.data_dir / f"{case_id}.pkl"
        seg_path = self.data_dir / f"{case_id}_seg.b2nd"
        for path in (zonal_path, lesion_path, props_path, seg_path):
            if not path.exists():
                raise ZonalPriorError(f"缺少输入: {path}")

        properties = pickle.loads(props_path.read_bytes())
        geometry = geometry_from_properties(
            properties,
            transpose_forward=self.context["transpose_forward"],
            target_spacing=self.context["target_spacing"],
        )

        # --- oracle：同一转换应用于原始 lesion，必须与 .b2nd 的 seg 逐体素一致
        lesion_plan = convert_label_to_plan_space(
            self._read_volume(lesion_path), geometry, resample_fn=self.context["resample_fn_seg"]
        )
        seg_ref = self._read_seg_only(case_id)
        oracle = oracle_check_lesion(lesion_plan, seg_ref)
        if not oracle.get("exact_match", False):
            raise ZonalPriorError(f"lesion oracle 校验失败：{oracle}；该例拒绝生成 prior（几何不可信）")

        # --- PZ/TZ：同一几何转换为 plan 空间 → 2 通道 one-hot
        zonal_plan = convert_label_to_plan_space(
            self._read_volume(zonal_path), geometry, resample_fn=self.context["resample_fn_seg"]
        )
        if tuple(zonal_plan.shape) != tuple(seg_ref.shape[-3:]):
            raise ZonalPriorError(
                f"zonal 转换后形状 {tuple(zonal_plan.shape)} != seg {tuple(seg_ref.shape[-3:])}"
            )
        prior = label_to_pz_tz(zonal_plan)
        validate_prior_array(prior)

        metadata = {
            "source": self.args.source,
            "configuration": self.args.configuration,
            "case_id": case_id,
            "plans_path": str(self.context["plans_path"]),
            "plans_sha256": self.context["plans_sha256"],
            "label_mapping": {"0": "background", "1": "PZ", "2": "TZ"},
            "geometry": geometry.to_dict(),
            "target_spacing": list(self.context["target_spacing"]),
            "input_paths": {
                "zonal": str(zonal_path),
                "lesion": str(lesion_path),
                "properties": str(props_path),
                "seg_b2nd": str(seg_path),
            },
            "input_sha256": self.case_input_hashes(case_id),
            "oracle": {**oracle, "method": "apply_same_transform_to_raw_lesion_and_compare_to_b2nd_seg"},
            "converted_label_sha256": array_sha256(zonal_plan.astype("int16")),
        }
        return prior, metadata


def main() -> int:
    args = parse_args()
    out_root = Path(args.out_root) if args.out_root else PROJECT_ROOT / f"data/processed/picai_zonal_{args.source}_v1"
    context = build_context(args)
    case_ids = list_case_ids(args, context)
    subset_run = bool(args.case_ids) or int(args.limit_cases) > 0
    if args.dry_run and not subset_run:
        raise SystemExit(
            "[zonal] --dry-run 必须显式限定子集（--case-ids 或正的 --limit-cases），"
            "避免误触全量只读转换"
        )
    # 运行**之前**确定 canonical 期望集合（train∪val）：manifest 发布门控与覆盖检查都基于它
    expected_ids: list[str] | None = None
    if subset_run:
        print("[zonal] 子集运行：不会创建或覆盖 canonical manifest.json")
    else:
        expected_ids = read_split_case_ids(args)
        if sorted(str(c) for c in case_ids) != sorted(str(c) for c in expected_ids):
            print(
                f"[zonal] 覆盖范围预先检查失败：将处理 {len(case_ids)} 例，"
                f"但 fold {args.fold} 的 train∪val 为 {len(expected_ids)} 例"
            )
            return 2
    print(f"[zonal] source={args.source}  cases={len(case_ids)}  out_root={out_root}  "
          f"dry_run={args.dry_run}  full_run={expected_ids is not None}")
    print(f"[zonal] plans={context['plans_path']}  sha256={context['plans_sha256'][:12]}…")
    print(f"[zonal] transpose_forward={context['transpose_forward']}  target_spacing={context['target_spacing']}")

    converter = CaseConverter(args, context)
    materializer = ZonalPriorMaterializer(
        out_root,
        source=args.source,
        configuration=args.configuration,
        plans_sha256=context["plans_sha256"],
        resume=args.resume,
        overwrite=args.overwrite,
    )
    summary = materializer.run(
        case_ids,
        converter.convert,
        identity_probe=converter.identity_probe if args.resume else None,
        expected_case_ids=expected_ids,
        dry_run=args.dry_run,
        progress=not args.no_progress,
        desc=f"zonal {args.source}",
    )
    print(
        f"[zonal] {'dry-run 完成' if args.dry_run else '完成'}：ok={summary['n_ok']} "
        f"skipped={summary['n_skipped']} failed={summary['n_failed']} 耗时={summary['elapsed_sec']}s"
    )
    failure = summary["n_failed"] > 0

    if args.dry_run:
        print("[zonal] --dry-run：未写入任何文件（npz/json/manifest/summary 全部跳过）")
        if summary["oracle_mismatch"]:
            print(f"[zonal] oracle 未通过（前 5）: {summary['oracle_mismatch'][:5]}")
        if summary["failed"]:
            print(f"[zonal] 失败清单（前 10）: {summary['failed'][:10]}")
        return 1 if failure else 0

    if summary.get("manifest_published"):
        print(f"[zonal] canonical manifest 已发布: {summary['manifest_path']}  "
              f"sha256={summary['manifest_sha256'][:12]}…  n_cases={summary.get('manifest_n_cases')}")
    else:
        reason = summary.get("manifest_not_published_reason", "(unknown)")
        print(f"[zonal] canonical manifest 未发布：{reason}")
        if not subset_run:
            # 完整运行未能发布 manifest 属于硬失败（既有 manifest 未被触碰）
            failure = True
    path = materializer.write_summary(summary)
    print(f"[zonal] 输出目录: {out_root}")
    print(f"[zonal] 统计: {path}")
    if summary["failed"]:
        print(f"[zonal] 失败清单（前 10 条，完整清单见统计文件）: {summary['failed'][:10]}")
    return 1 if failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
