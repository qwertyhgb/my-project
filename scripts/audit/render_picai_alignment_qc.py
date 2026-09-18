#!/usr/bin/env python
"""G0-R 图像对齐 QC 执行工具：物理空间叠加图 + landmark 位移 + 边界距离（只读）。

在**已抽样**的 PI-CAI 病例上（`sampling_manifest.json` 的病例清单，不可替换/增删/重抽样）：

1. **渲染模式（默认）**：把 ADC/HBV **仅按物理坐标**重采样到 T2W 参考网格（**不做任何配准**），
   为每个 case × 序列对生成 `checkerboard` 与 `edge-overlay` PNG（盲审材料：**不含病灶标签、
   不含模型预测**），并输出 landmark / boundary 记录模板、几何记录与运行元数据；
2. **指标模式（`--landmarks-csv`）**：**先强制完整校验**（严格整数坐标、非负、T2W/moving 各自的 FOV、
   病例子集与 case×pair 覆盖、重复记录、坐标空间），校验未通过时**非零退出且不写** `alignment_metrics.csv`；
   通过后把人工坐标换算为 `landmark_displacement_mm`（**LPS 物理坐标欧氏距离**），并在提供
   `--boundary-source` 时计算 `hausdorff_distance_mm` / `mean_surface_distance_mm`；
   `--allow-incomplete` 仅用于诊断性放行，产物标记 **INCOMPLETE（不得用于冻结）**；
3. **校验模式（`--validate-landmarks`）**：只做上述校验并输出报告（含 SHA256、错误数、覆盖病例与 case×pair），
   不写图像、不写指标。

**坐标空间（冻结）**：`native_index_per_series` —— `t2w_*` 为 T2W 物化网格上的体素索引，
`moving_*` 为对应 ADC/HBV 物化网格上的体素索引（`(i, j, k) = (x, y, z)`）；两个网格都可直接由研究者在
ITK-SNAP / 3D Slicer 中打开磁盘上的物化 `.nii.gz` 标注。位移由 SimpleITK
`TransformIndexToPhysicalPoint`（含各自 origin/direction/spacing）计算，**不做任何配准**。

安全与边界：

- 只读原始/物化数据，**不修改**任何数据、split、plan 或训练产物；
- 输出只能写入 `outputs/diagnostics/g0_r/<timestamp>/`（或显式 `--out-dir`，且拒绝覆盖非空目录）；
- **不执行、不推荐配准**；不显示病灶/模型信息；阈值与 `resample-only`/`resample+rigid` 决策由研究者填写；
- 无可靠边界来源时，边界距离记为 `UNKNOWN` 并写出跳过原因（不伪造数值）。

用法（研究者）：

    cd /opt/data/private/lm/my-projects
    conda activate lm

    # 1) 渲染（先 --dry-run 看计划）
    python scripts/audit/render_picai_alignment_qc.py --dry-run
    python scripts/audit/render_picai_alignment_qc.py

    # 2) 人工填写 <run>/landmark_template.csv → landmarks_filled.csv（双阅片者各填一份）

    # 3) 校验填写
    python scripts/audit/render_picai_alignment_qc.py --validate-landmarks \
        --landmarks-csv <run>/landmarks_filled.csv

    # 4) 计算指标（可选 --boundary-source）
    python scripts/audit/render_picai_alignment_qc.py --landmarks-csv <run>/landmarks_filled.csv

    # 5) 用既有 schema 复核
    python scripts/audit/audit_picai_alignment_qc.py --validate-metrics <run>/alignment_metrics.csv

**重要**：本工具产出的图与指标**不等于 G0-R PASS**；仍需双阅片者判读、盲态 pilot 阈值与研究者手动冻结
`resample-only` / `resample+rigid`（协议 §5/§6）。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from zonal_reliability_fusion.protocols import common as pc
from zonal_reliability_fusion.protocols import g0_r, g0_r_qc
from zonal_reliability_fusion.utils import make_progress, resolve_progress

DEFAULT_CONFIG = PROJECT_ROOT / "configs/protocols/g0_r_alignment_qc.yaml"
DEFAULT_MATERIALIZED_ROOT = "data/processed/picai"

METRIC_COLUMNS = list(g0_r.ALIGNMENT_METRICS_COLUMNS)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="G0-R 图像对齐 QC（物理空间叠加图 + landmark 位移；只读、不配准、不判 PASS）"
    )
    ap.add_argument("--config", default=str(DEFAULT_CONFIG), help="协议配置（默认 configs/protocols/g0_r_alignment_qc.yaml）")
    ap.add_argument("--sampling-manifest", default="", help="抽样 manifest（默认取 outputs/diagnostics/g0_r/ 下最新）")
    ap.add_argument("--out-dir", default="", help="输出目录（默认 <config outputs.dir>/<timestamp>）")
    ap.add_argument("--materialized-root", default="", help=f"物化数据根（默认 {DEFAULT_MATERIALIZED_ROOT}）")
    ap.add_argument("--case-ids", default="", help="仅处理抽样清单的子集（逗号分隔；不在清单中会报错）")
    ap.add_argument("--block-size", type=int, default=8, help="checkerboard 块大小（像素）")
    ap.add_argument("--landmarks-csv", default="", help="人工填写的 landmark CSV（指标模式；配合 --validate-landmarks 则只校验）")
    ap.add_argument("--validate-landmarks", action="store_true", help="只校验 landmark CSV（不写图像/指标）")
    ap.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="诊断性放行不完整/含错误的 landmark 输入（默认严格拒绝：非零退出且不写 alignment_metrics.csv）；"
        "放行产物标记 INCOMPLETE，不得用于冻结",
    )
    ap.add_argument("--boundary-source", default="", help="可选：可追溯边界 mask 清单（CSV/JSON）；未提供则边界距离记 UNKNOWN")
    ap.add_argument("--dry-run", action="store_true", help="只打印计划，不写任何文件")
    ap.add_argument("--no-progress", action="store_true", help="关闭进度条（日志环境）")
    return ap.parse_args()


# --------------------------------------------------------------------------- 小工具
def latest_run_dir(root: Path, marker: str) -> Path | None:
    """返回包含 marker 文件的最新子目录（按修改时间）。"""
    if not root.is_dir():
        return None
    candidates = [p for p in root.iterdir() if p.is_dir() and (p / marker).is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def prepare_fresh_dir(path: Path) -> Path:
    p = pc.assert_output_dir_isolated(path)
    if p.exists() and any(p.iterdir()):
        raise SystemExit(
            f"[g0-r-qc] 目标目录已存在且非空，拒绝覆盖: {p}\n"
            "（现有抽样/QC 产物必须保留；请使用新的时间戳目录或显式 --out-dir 指向新目录）"
        )
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_rows(path: Path, rows: list[dict], fieldnames: list[str]) -> Path:
    return pc.write_rows_csv(path, rows, fieldnames=fieldnames)


def import_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


# --------------------------------------------------------------------------- 渲染
def save_panels(
    panels: list,
    path: Path,
    *,
    kind: str,
    case_id: str,
    pair: str,
    slices: list[int],
    n_slices: int,
    orientation: str,
    resample_note: str,
) -> Path:
    plt = import_matplotlib()
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.9 * n, 4.4))
    axes_list = list(getattr(axes, "flat", [axes]))
    for ax, panel, z in zip(axes_list, panels, slices):
        import numpy as np

        array = np.asarray(panel)
        ax.imshow(array, cmap=None if array.ndim == 3 else "gray")
        ax.set_title(f"{kind} | k={z}/{n_slices - 1}", fontsize=8)
        ax.axis("off")
    fig.suptitle(f"{case_id} | {pair} | orientation={orientation}", fontsize=10)
    fig.text(
        0.5,
        0.02,
        f"{resample_note} | blind-review material: no lesion labels / no model outputs",
        ha="center",
        fontsize=7,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return path


def render_mode(args: argparse.Namespace, doc: dict, config_path: Path, progress: bool) -> None:
    import numpy as np

    manifest_path = Path(args.sampling_manifest) if args.sampling_manifest else None
    if manifest_path is None:
        root = pc.resolve_project_path(pc.get_path(doc, "outputs.dir", default="outputs/diagnostics/g0_r"))
        latest = latest_run_dir(root, "sampling_manifest.json")
        if latest is None:
            raise SystemExit(
                f"[g0-r-qc] 未找到 sampling manifest（{root} 下无 sampling_manifest.json）；"
                "请先运行 scripts/audit/audit_picai_alignment_qc.py，或用 --sampling-manifest 指定"
            )
        manifest_path = latest / "sampling_manifest.json"
        print(f"[g0-r-qc] 使用最新 sampling manifest: {manifest_path}")
    manifest = g0_r_qc.load_sampling_manifest(manifest_path)
    requested = [c.strip() for c in args.case_ids.split(",") if c.strip()] if args.case_ids else None
    case_ids = g0_r_qc.select_cases(manifest, requested)
    pairs = list(manifest["pairs"])

    materialized_root = pc.resolve_project_path(
        args.materialized_root or pc.get_path(doc, "inputs.materialized_cases", default=DEFAULT_MATERIALIZED_ROOT)
    )
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else pc.resolve_project_path(pc.get_path(doc, "outputs.dir", default="outputs/diagnostics/g0_r"))
        / pc.timestamp_slug()
    )

    if args.dry_run:
        print(f"[g0-r-qc][dry-run] sampling manifest: {manifest_path}（cases={len(case_ids)}）")
        print(f"[g0-r-qc][dry-run] materialized root: {materialized_root}")
        print(f"[g0-r-qc][dry-run] 输出目录（不创建）: {out_dir}")
        print(f"[g0-r-qc][dry-run] 计划: {len(case_ids)} 例 × {len(pairs)} 序列对 × 2 图（checkerboard/edge）")
        print(f"[g0-r-qc][dry-run] 切片: {g0_r_qc.DEFAULT_SLICE_FRACTIONS}（相对 k 深度，含病灶信息=否）")
        # 只检查目录存在性（不读影像）：尽早暴露路径配置错误
        missing_dirs = [cid for cid in case_ids if not g0_r_qc.case_dir_for(materialized_root, cid).is_dir()]
        if missing_dirs:
            print(
                f"[g0-r-qc][dry-run][WARN] {len(missing_dirs)} 例物化目录不存在（前 3: {missing_dirs[:3]}）；"
                "请检查 --materialized-root / config inputs.materialized_cases（仅检查目录，不读影像）"
            )
        else:
            print("[g0-r-qc][dry-run] 物化目录检查: 全部病例目录存在（仅检查目录，不读影像）")
        print("[g0-r-qc][dry-run] 声明: physical-space resample ONLY；不执行/不推荐配准；不判 PASS")
        return

    out_dir = prepare_fresh_dir(out_dir)
    overlay_dir = out_dir / "overlay"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    index_rows: list[dict] = []
    geometry: dict[str, dict] = {}
    skipped: list[dict] = []
    input_hashes: dict[str, str] = {}

    bar = make_progress(case_ids, total=len(case_ids), desc="g0-r render cases", disable=not progress)
    for case_id in bar:
        try:
            paths = g0_r_qc.resolve_series_paths(materialized_root, case_id)
        except pc.ProtocolConfigError as exc:
            skipped.append({"case_id": case_id, "pair": "", "reason": str(exc)})
            continue
        missing = g0_r_qc.missing_series(paths)
        if missing:
            skipped.append({"case_id": case_id, "pair": "", "reason": f"缺少物化序列文件: {missing}"})
            continue
        t2w_geo = None
        case_entry: dict = {"t2w_size": None, "t2w_spacing": None, "t2w_orientation": None, "pairs": {}}
        for pair in pairs:
            moving_series = g0_r_qc.PAIR_TO_SERIES[pair]
            try:
                t2w, moving, record = g0_r_qc.prepare_pair_arrays(
                    paths["t2w"], paths[moving_series], case_id=case_id, pair=pair
                )
            except Exception as exc:  # noqa: BLE001 - 逐例记录，不中断整体
                skipped.append({"case_id": case_id, "pair": pair, "reason": f"读取/重采样失败: {exc}"})
                continue
            n_slices = int(np.asarray(t2w).shape[0])
            slices = g0_r_qc.select_slice_indices(n_slices)
            panels_cb = [g0_r_qc.checkerboard_image(t2w[z], moving[z], block=args.block_size) for z in slices]
            panels_ed = [g0_r_qc.edge_overlay_image(t2w[z], moving[z]) for z in slices]
            png_cb = overlay_dir / f"{case_id}_{moving_series}_checkerboard.png"
            png_ed = overlay_dir / f"{case_id}_{moving_series}_edge.png"
            try:
                save_panels(
                    panels_cb, png_cb, kind="checkerboard", case_id=case_id, pair=pair, slices=slices,
                    n_slices=n_slices, orientation=record.t2w_orientation, resample_note=g0_r_qc.RESAMPLE_ONLY_BANNER,
                )
                save_panels(
                    panels_ed, png_ed, kind="edge-overlay(T2W=red/moving=green)", case_id=case_id, pair=pair,
                    slices=slices, n_slices=n_slices, orientation=record.t2w_orientation,
                    resample_note=g0_r_qc.RESAMPLE_ONLY_BANNER,
                )
            except Exception as exc:  # noqa: BLE001
                skipped.append({"case_id": case_id, "pair": pair, "reason": f"PNG 渲染失败: {exc}"})
                continue
            index_rows.append(
                {
                    "case_id": case_id,
                    "pair": pair,
                    "series": moving_series,
                    "checkerboard_png": str(png_cb),
                    "edge_overlay_png": str(png_ed),
                    "slices_k": ";".join(str(z) for z in slices),
                    "n_slices": n_slices,
                    "same_grid_as_t2w": int(record.same_grid_as_t2w),
                    "resampled_now": int(record.resampled_now),
                    "t2w_orientation": record.t2w_orientation,
                    "strata": manifest["strata_by_case"].get(case_id, ""),
                }
            )
            case_entry["pairs"][pair] = record.as_dict()
            if t2w_geo is None:
                t2w_geo = record
                # 计算物理坐标（LPS）所需的完整 T2W 几何：size/spacing/origin/direction 缺一不可
                case_entry["t2w_size"] = list(record.t2w_size)
                case_entry["t2w_spacing"] = list(record.t2w_spacing)
                case_entry["t2w_origin"] = list(record.t2w_origin)
                case_entry["t2w_direction"] = list(record.t2w_direction)
                case_entry["t2w_orientation"] = record.t2w_orientation
        geometry[case_id] = case_entry

    # 输入影像 SHA256（只读；带进度）
    hash_targets = [
        str(g0_r_qc.resolve_series_paths(materialized_root, cid)[s])
        for cid in case_ids
        if g0_r_qc.case_dir_for(materialized_root, cid).is_dir()
        for s in ("t2w", "adc", "hbv")
        if g0_r_qc.resolve_series_paths(materialized_root, cid)[s].is_file()
    ]
    hash_bar = make_progress(hash_targets, total=len(hash_targets), desc="input sha256", disable=not progress)
    for path_str in hash_bar:
        input_hashes[path_str] = pc.sha256_file(path_str)

    # 模板与清单（坐标空间为冻结值；研究者按各自序列的原生体素索引标注）
    template_rows = [
        {
            "case_id": row["case_id"],
            "pair": row["pair"],
            "landmark_name": "",
            "coordinate_space": g0_r_qc.COORDINATE_SPACE_NATIVE,
            "t2w_i": "", "t2w_j": "", "t2w_k": "",
            "moving_i": "", "moving_j": "", "moving_k": "",
            "measured_by": "",
            "notes": "",
        }
        for row in index_rows
    ]
    write_rows(out_dir / "landmark_template.csv", template_rows, list(g0_r_qc.LANDMARK_TEMPLATE_COLUMNS))
    boundary_rows = [
        {"case_id": row["case_id"], "pair": row["pair"], "series": "t2w", "boundary_path": "", "notes": ""}
        for row in index_rows
    ] + [
        {"case_id": row["case_id"], "pair": row["pair"], "series": row["series"], "boundary_path": "", "notes": ""}
        for row in index_rows
    ]
    write_rows(out_dir / "boundary_template.csv", boundary_rows, list(g0_r_qc.BOUNDARY_TEMPLATE_COLUMNS))
    write_rows(
        out_dir / "qc_index.csv",
        index_rows,
        ["case_id", "pair", "series", "checkerboard_png", "edge_overlay_png", "slices_k", "n_slices",
         "same_grid_as_t2w", "resampled_now", "t2w_orientation", "strata"],
    )
    write_rows(out_dir / "skipped.csv", skipped, ["case_id", "pair", "reason"])

    pc.write_json(
        out_dir / "qc_geometry.json",
        {
            "source": "physical-space resample only (no registration)",
            "sampling_manifest": str(manifest_path),
            "config_sha256": pc.sha256_file(config_path),
            "cases": geometry,
        },
    )
    pc.write_json(
        out_dir / "run_metadata.json",
        pc.build_run_metadata(
            tool="scripts/audit/render_picai_alignment_qc.py",
            config_path=config_path,
            extra={
                "mode": "render",
                "sampling_manifest": str(manifest_path),
                "sampling_manifest_sha256": pc.sha256_file(manifest_path),
                "materialized_root": str(materialized_root),
                "case_ids": case_ids,
                "pairs": pairs,
                "n_png": 2 * len(index_rows),
                "n_skipped": len(skipped),
                "input_sha256": input_hashes,
                "outputs": sorted(p.name for p in out_dir.iterdir()),
                "note": "blind-review material only; outputs do NOT constitute G0-R PASS",
            },
        ),
    )

    print(
        f"[g0-r-qc] 渲染完成: cases={len(case_ids)} 例×{len(pairs)} 对, PNG={2 * len(index_rows)}, "
        f"skipped={len(skipped)}, 输出={out_dir}"
    )
    if skipped:
        print("[g0-r-qc][WARN] 存在跳过项，详见 skipped.csv（原因逐条记录，未伪造任何结果）")
    print(
        "[g0-r-qc] 下一步（研究者）：① 双阅片者用 ITK-SNAP/3D Slicer 打开物化序列、"
        "填写 landmark_template.csv（坐标序 i,j,k = x,y,z）；② 盲态判读视觉等级（blind_review_sheet.csv）；"
        "③ --validate-landmarks 校验后运行指标模式生成 alignment_metrics.csv。"
        "所有产物仅为 QC 证据，G0-R 仍为 PENDING。"
    )


# --------------------------------------------------------------------------- 指标计算
def metrics_mode(args: argparse.Namespace, doc: dict, config_path: Path, progress: bool) -> None:
    manifest_path = Path(args.sampling_manifest) if args.sampling_manifest else None
    if manifest_path is None:
        root = pc.resolve_project_path(pc.get_path(doc, "outputs.dir", default="outputs/diagnostics/g0_r"))
        latest = latest_run_dir(root, "sampling_manifest.json")
        if latest is None:
            raise SystemExit("[g0-r-qc] 未找到 sampling manifest（请用 --sampling-manifest 指定）")
        manifest_path = latest / "sampling_manifest.json"
    manifest = g0_r_qc.load_sampling_manifest(manifest_path)

    run_dir = Path(args.out_dir) if args.out_dir else None
    if run_dir is None:
        root = pc.resolve_project_path(pc.get_path(doc, "outputs.dir", default="outputs/diagnostics/g0_r"))
        run_dir = latest_run_dir(root, "qc_geometry.json")
        if run_dir is None:
            raise SystemExit(
                "[g0-r-qc] 未找到含 qc_geometry.json 的渲染目录；请先运行渲染模式或用 --out-dir 指定"
            )
    geometry_path = run_dir / "qc_geometry.json"
    if not geometry_path.is_file():
        raise SystemExit(f"[g0-r-qc] 缺少 {geometry_path}（先运行渲染模式生成几何记录）")
    geometry_doc = json.loads(geometry_path.read_text(encoding="utf-8"))
    geometry_by_case = dict(geometry_doc.get("cases") or {})
    if not geometry_by_case:
        raise SystemExit(f"[g0-r-qc] {geometry_path} 不含 cases 几何记录（请先重跑渲染模式）")

    landmarks_path = Path(args.landmarks_csv)
    if not landmarks_path.is_file():
        raise SystemExit(f"[g0-r-qc] landmark 文件不存在: {landmarks_path}")
    with open(landmarks_path, newline="", encoding="utf-8") as fh:
        landmark_rows = [dict(row) for row in csv.DictReader(fh)]

    # ---- 强制完整校验（不可绕过：非整数 / 负坐标 / FOV 越界 / 病例-pair 完整性 / 重复 / 坐标空间）----
    min_per_case_pair = pc.get_path(doc, "evaluation.landmark_rules.min_per_case_pair")
    validation_errors = g0_r_qc.validate_landmark_rows(
        landmark_rows,
        allowed_cases=manifest["case_ids"],
        allowed_pairs=tuple(manifest["pairs"]),
        grid_info_by_case=geometry_by_case,
    )
    completeness = g0_r_qc.check_landmark_completeness(
        landmark_rows,
        allowed_cases=manifest["case_ids"],
        allowed_pairs=tuple(manifest["pairs"]),
        min_per_case_pair=None if min_per_case_pair is None else int(min_per_case_pair),
    )
    validation_report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "landmarks_csv": str(landmarks_path),
        "landmarks_sha256": pc.sha256_file(landmarks_path),
        "n_rows": len(landmark_rows),
        "n_errors": len(validation_errors),
        "errors": validation_errors[:200],
        "covered_cases": completeness["n_cases_covered"],
        "expected_cases": completeness["n_expected_cases"],
        "covered_case_pairs": sorted(completeness["per_case_pair_counts"]),
        "missing_case_pair": completeness["missing_case_pair"],
        "per_case_pair_counts": completeness["per_case_pair_counts"],
        "missing_measured_by_rows": completeness["missing_measured_by_rows"],
        "blockers": completeness["blockers"],
        "min_per_case_pair_rule": completeness["min_per_case_pair_rule"],
        "coordinate_space": g0_r_qc.COORDINATE_SPACE_NATIVE,
        "status": "VALID" if not validation_errors and not completeness["blockers"] else "INVALID",
    }
    strict_ok = not validation_errors and not completeness["blockers"]
    n_expected_case_pairs = len(manifest["case_ids"]) * len(manifest["pairs"])
    print(
        f"[g0-r-qc][landmark-check] rows={len(landmark_rows)} errors={len(validation_errors)} "
        f"coverage={completeness['n_cases_covered']}/{completeness['n_expected_cases']} cases, "
        f"{len(completeness['per_case_pair_counts'])}/{n_expected_case_pairs} case×pair, "
        f"status={'OK' if strict_ok else 'INVALID'}"
    )
    for err in validation_errors[:20]:
        print(f"  - error: {err}")
    if len(validation_errors) > 20:
        print(f"  - ... 其余 {len(validation_errors) - 20} 条见 landmarks_validation.json")
    for blocker in completeness["blockers"]:
        print(f"  - blocker: {blocker}")

    if not strict_ok and not args.allow_incomplete:
        if run_dir.is_dir():
            pc.write_json(
                run_dir / "landmarks_validation.json",
                {**validation_report, "mode": "metrics-strict-reject", "metrics_written": False},
            )
        raise SystemExit(
            "[g0-r-qc] landmark 校验未通过（默认严格模式）：拒绝生成 alignment_metrics.csv。\n"
            f"  errors={len(validation_errors)} blockers={len(completeness['blockers'])}；"
            "请修正填写后重跑；仅在诊断用途下可用 --allow-incomplete 放行"
            "（产物标记 INCOMPLETE，不得用于冻结或论文结果）"
        )
    if not strict_ok and args.allow_incomplete:
        print(
            "[g0-r-qc][WARN] --allow-incomplete 已启用：本次产物标记 INCOMPLETE，"
            "**不得用于冻结或论文结果**（先修正输入或先冻结协议规则）"
        )

    landmark_result = g0_r_qc.compute_landmark_metrics(
        landmark_rows,
        geometry_by_case=geometry_by_case,
        allowed_cases=manifest["case_ids"],
        allowed_pairs=tuple(manifest["pairs"]),
    )
    if landmark_result.errors:
        # 校验通过后不应出现；防御性分支（严格模式拒绝，诊断模式降级为告警并记录）
        if not args.allow_incomplete:
            raise SystemExit(
                "[g0-r-qc] 指标计算阶段仍存在坐标错误（不应发生在已通过校验的输入上）：\n- "
                + "\n- ".join(landmark_result.errors[:20])
            )
        print(
            f"[g0-r-qc][WARN] {len(landmark_result.errors)} 行坐标错误已被跳过（INCOMPLETE 诊断模式），"
            "未产出这些行的指标"
        )

    metric_rows: list[dict] = list(landmark_result.rows)
    boundary_result = g0_r_qc.BoundaryResult()
    if args.boundary_source:
        entries = g0_r_qc.read_boundary_source(args.boundary_source)
        boundary_result = g0_r_qc.compute_boundary_metrics(
            entries, allowed_cases=manifest["case_ids"], allowed_pairs=tuple(manifest["pairs"])
        )
    else:
        boundary_result.skipped.append(
            {
                "reason": "no_boundary_source_configured：--boundary-source 未配置；"
                "边界距离记为 UNKNOWN（不伪造数值，协议 §4.2.1）"
            }
        )
    metric_rows.extend(boundary_result.metrics)

    schema_errors = g0_r.validate_alignment_metrics_rows(metric_rows) if metric_rows else ["没有可写出的指标行"]
    if metric_rows and schema_errors:
        raise SystemExit("[g0-r-qc] 生成的指标行不符合既有 schema（拒绝写出）：\n- " + "\n- ".join(schema_errors))

    if args.dry_run:
        print(f"[g0-r-qc][dry-run] landmark 行={len(landmark_rows)} → 指标行={len(landmark_result.rows)}")
        print(f"[g0-r-qc][dry-run] 边界指标行={len(boundary_result.metrics)}；跳过={len(boundary_result.skipped)}")
        print(
            f"[g0-r-qc][dry-run] landmark 校验: errors={len(validation_errors)} "
            f"blockers={len(completeness['blockers'])} → 产物状态 "
            f"{'COMPLETE' if strict_ok else 'INCOMPLETE（不得用于冻结）'}"
        )
        print(f"[g0-r-qc][dry-run] 将写入 {run_dir / 'alignment_metrics.csv'}（不创建）")
        return

    pc.assert_output_dir_isolated(run_dir)
    if not metric_rows:
        raise SystemExit(
            "[g0-r-qc] 没有可写出的指标行（landmark 可能为空或全部被跳过）；"
            "未写出 alignment_metrics.csv，避免产生空/误导结果"
        )
    metrics_path = run_dir / "alignment_metrics.csv"
    if metrics_path.exists():
        raise SystemExit(f"[g0-r-qc] {metrics_path} 已存在，拒绝覆盖（请换新 run 目录）")
    write_rows(metrics_path, metric_rows, METRIC_COLUMNS)
    output_status = "COMPLETE" if strict_ok else "INCOMPLETE"
    pc.write_json(
        run_dir / "landmarks_validation.json",
        {**validation_report, "mode": "metrics", "metrics_written": True, "output_status": output_status},
    )
    pc.write_json(
        run_dir / "landmark_summary.json",
        {
            "source_landmarks": str(landmarks_path),
            "landmarks_sha256": validation_report["landmarks_sha256"],
            "n_landmark_rows": len(landmark_rows),
            "n_metric_rows": len(landmark_result.rows),
            "per_case_pair": g0_r_qc.summarize_landmark_metrics(metric_rows),
            "units": "mm",
            "coordinate_space": g0_r_qc.COORDINATE_SPACE_NATIVE,
            "completeness": {
                "covered_cases": completeness["n_cases_covered"],
                "expected_cases": completeness["n_expected_cases"],
                "missing_case_pair": completeness["missing_case_pair"],
                "missing_measured_by_rows": completeness["missing_measured_by_rows"],
                "blockers": completeness["blockers"],
            },
            "status": output_status,
            "note": "INCOMPLETE 产物（--allow-incomplete 放行）不得用于冻结或论文结果"
            if output_status == "INCOMPLETE"
            else "校验与覆盖完整；仍不代表 G0-R 通过（阈值与决策待研究者冻结）",
        },
    )
    pc.write_json(
        run_dir / "metrics_skipped.json",
        {
            "landmark_skipped": landmark_result.skipped,
            "landmark_errors": landmark_result.errors,
            "boundary_skipped": boundary_result.skipped,
        },
    )
    pc.write_json(
        run_dir / "metrics_run_metadata.json",
        pc.build_run_metadata(
            tool="scripts/audit/render_picai_alignment_qc.py",
            config_path=config_path,
            extra={
                "mode": "metrics",
                "landmarks_csv": str(landmarks_path),
                "landmarks_sha256": pc.sha256_file(landmarks_path),
                "boundary_source": args.boundary_source or None,
                "metrics_csv": str(metrics_path),
                "n_metrics": len(metric_rows),
                "n_boundary_metrics": len(boundary_result.metrics),
                "coordinate_space": g0_r_qc.COORDINATE_SPACE_NATIVE,
                "output_status": output_status,
                "allow_incomplete": bool(args.allow_incomplete),
                "n_validation_errors": len(validation_errors),
                "n_blockers": len(completeness["blockers"]),
                "note": "不给配准建议、不判 G0-R PASS；阈值与决策由研究者按协议 §5/§6 填写",
                "incomplete_note": "INCOMPLETE 产物不得用于冻结或论文结果"
                if output_status == "INCOMPLETE"
                else None,
            },
        ),
    )
    print(
        f"[g0-r-qc] 指标完成: landmark={len(landmark_result.rows)} 行, 边界={len(boundary_result.metrics)} 行, "
        f"跳过={len(landmark_result.skipped) + len(boundary_result.skipped)} → {metrics_path}"
    )
    print(
        "[g0-r-qc] 请用既有 schema 复核：python scripts/audit/audit_picai_alignment_qc.py "
        f"--validate-metrics {metrics_path}"
    )


# --------------------------------------------------------------------------- landmark 校验
def validate_mode(args: argparse.Namespace, doc: dict) -> int:
    manifest_path = Path(args.sampling_manifest) if args.sampling_manifest else None
    if manifest_path is None:
        root = pc.resolve_project_path(pc.get_path(doc, "outputs.dir", default="outputs/diagnostics/g0_r"))
        latest = latest_run_dir(root, "sampling_manifest.json")
        if latest is None:
            raise SystemExit("[g0-r-qc] 未找到 sampling manifest（请用 --sampling-manifest 指定）")
        manifest_path = latest / "sampling_manifest.json"
    manifest = g0_r_qc.load_sampling_manifest(manifest_path)

    landmarks_path = Path(args.landmarks_csv)
    if not landmarks_path.is_file():
        raise SystemExit(f"[g0-r-qc] landmark 文件不存在: {landmarks_path}")
    with open(landmarks_path, newline="", encoding="utf-8") as fh:
        rows = [dict(row) for row in csv.DictReader(fh)]

    grid_info = None
    run_dir = Path(args.out_dir) if args.out_dir else None
    if run_dir is None:
        root = pc.resolve_project_path(pc.get_path(doc, "outputs.dir", default="outputs/diagnostics/g0_r"))
        run_dir = latest_run_dir(root, "qc_geometry.json")
    if run_dir is not None and (Path(run_dir) / "qc_geometry.json").is_file():
        grid_info = json.loads((Path(run_dir) / "qc_geometry.json").read_text(encoding="utf-8")).get("cases")

    min_per_case_pair = pc.get_path(doc, "evaluation.landmark_rules.min_per_case_pair")
    errors = g0_r_qc.validate_landmark_rows(
        rows,
        allowed_cases=manifest["case_ids"],
        allowed_pairs=tuple(manifest["pairs"]),
        grid_info_by_case=grid_info,
    )
    completeness = g0_r_qc.check_landmark_completeness(
        rows,
        allowed_cases=manifest["case_ids"],
        allowed_pairs=tuple(manifest["pairs"]),
        min_per_case_pair=None if min_per_case_pair is None else int(min_per_case_pair),
    )
    n_expected_case_pairs = len(manifest["case_ids"]) * len(manifest["pairs"])
    print(
        f"[g0-r-qc][validate-landmarks] file={landmarks_path} rows={len(rows)} errors={len(errors)} "
        f"coverage={completeness['n_cases_covered']}/{completeness['n_expected_cases']} cases, "
        f"{len(completeness['per_case_pair_counts'])}/{n_expected_case_pairs} case×pair"
    )
    for err in errors[:50]:
        print(f"  - error: {err}")
    if len(errors) > 50:
        print(f"  - ... 其余 {len(errors) - 50} 条见 landmarks_validation.json")
    for blocker in completeness["blockers"]:
        print(f"  - blocker: {blocker}")
    if not grid_info:
        print(
            "[g0-r-qc][validate-landmarks] 提示: 未找到 qc_geometry.json，"
            "FOV 范围检查已跳过（仅完成列/病例/pair/坐标格式校验）"
        )
    status = "VALID" if not errors and not completeness["blockers"] else "INVALID"
    if run_dir and Path(run_dir).is_dir():
        pc.write_json(
            Path(run_dir) / "landmarks_validation.json",
            {
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "landmarks_csv": str(landmarks_path),
                "landmarks_sha256": pc.sha256_file(landmarks_path),
                "n_rows": len(rows),
                "n_errors": len(errors),
                "errors": errors[:200],
                "covered_cases": completeness["n_cases_covered"],
                "expected_cases": completeness["n_expected_cases"],
                "covered_case_pairs": sorted(completeness["per_case_pair_counts"]),
                "missing_case_pair": completeness["missing_case_pair"],
                "per_case_pair_counts": completeness["per_case_pair_counts"],
                "missing_measured_by_rows": completeness["missing_measured_by_rows"],
                "blockers": completeness["blockers"],
                "min_per_case_pair_rule": completeness["min_per_case_pair_rule"],
                "coordinate_space": g0_r_qc.COORDINATE_SPACE_NATIVE,
                "fov_check": bool(grid_info),
                "status": status,
                "note": "校验通过不代表 G0-R 通过；仍需盲态阈值与研究者决策",
            },
        )
    return 1 if status == "INVALID" else 0


def main() -> None:
    args = parse_args()
    t0 = time.time()
    config_path = Path(args.config)
    doc = pc.load_yaml_config(config_path)
    pc.validate_protocol_block(doc, expected_id="G0-R")
    progress = resolve_progress(args.no_progress, True)

    if args.validate_landmarks:
        if not args.landmarks_csv:
            raise SystemExit("[g0-r-qc] --validate-landmarks 需要同时提供 --landmarks-csv")
        sys.exit(validate_mode(args, doc))
    if args.landmarks_csv:
        metrics_mode(args, doc, config_path, progress)
    else:
        render_mode(args, doc, config_path, progress)
    print(f"[g0-r-qc] elapsed={time.time() - t0:.1f}s；G0-R 状态仍为 PENDING/DRAFT（不因本工具产物变为 PASS）")


if __name__ == "__main__":
    main()
