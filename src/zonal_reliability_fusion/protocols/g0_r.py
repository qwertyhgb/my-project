"""G0-R：确定性分层抽样、盲审清单与定量对齐指标 schema。

本模块只处理 manifest / 几何审计**元数据行**（dict 列表），不读取影像体素；
不决定“采用配准”或 G0-R 的通过状态——`decision` 相关字段一律由研究者按
`docs/protocols/G0_R_ALIGNMENT_QC.md` §5/§6 填写，本模块保持 `None`。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .common import ProtocolConfigError

#: 必须分别评价的序列对（docs/protocols/G0_R_ALIGNMENT_QC.md §4.1）
PAIRS: tuple[str, ...] = ("T2W-ADC", "T2W-HBV")

#: 定量指标枚举与允许单位（mm 为位移/距离；au = 任意单位，仅作辅助）
METRIC_UNITS: dict[str, str] = {
    "landmark_displacement_mm": "mm",
    "boundary_distance_mm": "mm",
    "mean_surface_distance_mm": "mm",
    "hausdorff_distance_mm": "mm",
    "mi": "au",
    "nmi": "au",
}
#: 只能作辅助、不得单独决策的指标（protocol §4.2.4）
AUXILIARY_METRICS: tuple[str, ...] = ("mi", "nmi")

#: 指标 CSV 的列（机器可读 schema 的字段来源）
ALIGNMENT_METRICS_COLUMNS: tuple[str, ...] = (
    "case_id",
    "pair",
    "metric",
    "value",
    "unit",
    "method",
    "measured_by",
    "notes",
)

#: sampling_cfg 中使用的几何字段（来自 picai_manifest.csv）
GEOMETRY_COLUMNS: tuple[str, ...] = (
    "geometry_status",
    "t2w_extent",
    "adc_extent",
    "hbv_extent",
    "t2w_spacing",
    "adc_spacing",
    "hbv_spacing",
)


def parse_vector3(value: Any, *, field_name: str) -> tuple[float, float, float] | None:
    """解析 `"1.0;2.0;3.0"` 形式的三元组；空值 → None；非法 → 报错。"""
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return tuple(float(v) for v in value)  # type: ignore[return-value]
    if isinstance(value, str):
        token = value.strip()
        if not token:
            return None
        parts = token.split(";")
        if len(parts) != 3:
            raise ProtocolConfigError(f"{field_name} 需要 3 个分号分隔的数值，收到 {value!r}")
        return tuple(float(p) for p in parts)  # type: ignore[return-value]
    raise ProtocolConfigError(f"{field_name} 格式不支持: {value!r}")


def extent_volume(extent: Sequence[float]) -> float:
    """物理包围盒体积（mm³），用于极端 FOV 分层排序。"""
    return float(extent[0]) * float(extent[1]) * float(extent[2])


def planar_spacing_mean(spacing: Sequence[float]) -> float:
    """平面内间距均值（前两个分量的算术平均），用于 spacing 跨度分层。"""
    return (float(spacing[0]) + float(spacing[1])) / 2.0


@dataclass
class SamplingResult:
    """确定性抽样结果（不包含任何阈值/决策）。"""

    rows: list[dict[str, Any]] = field(default_factory=list)
    per_stratum: dict[str, int] = field(default_factory=dict)
    n_cases_configured: int | None = None
    n_cases_source: str = "minimum_required"
    warnings: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.rows)

    def as_manifest(self) -> dict[str, Any]:
        return {
            "total_cases": self.total,
            "n_cases_configured": self.n_cases_configured,
            "n_cases_source": self.n_cases_source,
            "per_stratum_counts": dict(self.per_stratum),
            "warnings": list(self.warnings),
            "decision": None,  # G0-R 决策由研究者按 §6 冻结；工具不得填写
            "cases": [
                {"case_id": row["case_id"], "strata": row["strata"], "selection_order": row["selection_order"]}
                for row in self.rows
            ],
        }


def _sorted_case_ids(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return sorted(str(r.get("case_id", "")) for r in rows)


def plan_sampling(
    manifest_rows: Sequence[Mapping[str, Any]],
    sampling_cfg: Mapping[str, Any],
) -> SamplingResult:
    """按 `configs/protocols/g0_r_alignment_qc.yaml` 的 `sampling.strata` 做确定性分层抽样。

    规则（与协议草案 §3.1 一一对应）：

    1. `geometry_suspect` 全部纳入；
    2. `center`：每个中心（按名称排序）至少 `min_per_level` 例；
    3. `lesion_status`：`case_csPCa` 每个取值至少 `min_per_level` 例；
    4. `partial_overlap`：`geometry_status == partial_physical_overlap` 至少 `min_cases` 例；
    5. `extreme_fov`：按 T2W extent 体积排序，最大/最小各 `min_per_tail` 例；
    6. `spacing_span`：按 ADC 与 T2W 平面内间距排序，两端各 `min_per_tail` 例；
    7. 若 `sampling.n_cases` 已填写，剩余名额按 `case_id` 升序补齐（`fill_deterministic`）；
       若为 `null`，只保留最小必选集合，并在结果中标记 `n_cases_source="minimum_required"`。

    全部步骤均为确定性（排序 + 固定优先级），不使用随机数。
    """
    if not manifest_rows:
        raise ProtocolConfigError("manifest_rows 为空：无法抽样（先确认 manifest 已生成）")
    rows: list[Mapping[str, Any]] = []
    for row in manifest_rows:
        cid = str(row.get("case_id", "")).strip()
        if not cid:
            raise ProtocolConfigError("manifest 行缺少 case_id")
        rows.append(row)

    strata_cfg = dict(sampling_cfg.get("strata") or {})
    warnings: list[str] = []
    selected: dict[str, dict[str, Any]] = {}
    hit_counts: dict[str, int] = {}

    def _add(row: Mapping[str, Any], stratum: str) -> None:
        cid = str(row["case_id"])
        if cid not in selected:
            selected[cid] = {"row": dict(row), "strata": [], "selection_order": len(selected)}
        if stratum not in selected[cid]["strata"]:
            selected[cid]["strata"].append(stratum)
            hit_counts[stratum] = hit_counts.get(stratum, 0) + 1

    # 1. geometry_suspect 全纳入
    suspect_cfg = dict(strata_cfg.get("geometry_suspect") or {})
    if suspect_cfg.get("include_all"):
        for row in sorted(rows, key=lambda r: str(r["case_id"])):
            if str(row.get("geometry_status", "")) == "geometry_suspect":
                _add(row, "geometry_suspect")

    # 2. center 分层
    center_cfg = dict(strata_cfg.get("center") or {})
    min_per_level = int(center_cfg.get("min_per_level", 1))
    centers = sorted({str(r.get("center", "")).strip() for r in rows if str(r.get("center", "")).strip()})
    for center in centers:
        taken = 0
        for row in sorted(rows, key=lambda r: str(r["case_id"])):
            if taken >= min_per_level:
                break
            if str(row.get("center", "")).strip() == center and str(row["case_id"]) not in selected:
                _add(row, "center")
                taken += 1
        if taken < min_per_level:
            warnings.append(f"center={center!r} 仅取到 {taken}/{min_per_level} 例")

    # 3. lesion_status 分层（case_csPCa 取值）
    lesion_cfg = dict(strata_cfg.get("lesion_status") or {})
    min_per_level = int(lesion_cfg.get("min_per_level", 1))
    statuses = sorted({str(r.get("case_csPCa", "")).strip() for r in rows if str(r.get("case_csPCa", "")).strip()})
    for status in statuses:
        taken = 0
        for row in sorted(rows, key=lambda r: str(r["case_id"])):
            if taken >= min_per_level:
                break
            if str(row.get("case_csPCa", "")).strip() == status and str(row["case_id"]) not in selected:
                _add(row, "lesion_status")
                taken += 1
        if taken < min_per_level:
            warnings.append(f"case_csPCa={status!r} 仅取到 {taken}/{min_per_level} 例")

    # 4. partial_overlap
    partial_cfg = dict(strata_cfg.get("partial_overlap") or {})
    min_cases = int(partial_cfg.get("min_cases", 0))
    taken = 0
    for row in sorted(rows, key=lambda r: str(r["case_id"])):
        if taken >= min_cases:
            break
        if str(row.get("geometry_status", "")) == "partial_physical_overlap":
            _add(row, "partial_overlap")
            taken += 1
    if taken < min_cases:
        warnings.append(f"partial_overlap 仅取到 {taken}/{min_cases} 例")

    # 5. extreme_fov（按 T2W extent 体积）
    fov_cfg = dict(strata_cfg.get("extreme_fov") or {})
    min_per_tail = int(fov_cfg.get("min_per_tail", 0))
    if min_per_tail > 0:
        with_extent: list[tuple[float, Mapping[str, Any]]] = []
        for row in rows:
            extent = parse_vector3(row.get("t2w_extent"), field_name=f"t2w_extent[{row.get('case_id')}]")
            if extent is None:
                continue
            with_extent.append((extent_volume(extent), row))
        with_extent.sort(key=lambda item: (item[0], str(item[1]["case_id"])))
        missing_extent = len(rows) - len(with_extent)
        if missing_extent:
            warnings.append(f"extreme_fov: {missing_extent} 例缺 t2w_extent，已跳过")
        for _, row in with_extent[:min_per_tail]:
            _add(row, "extreme_fov")
        for _, row in with_extent[-min_per_tail:]:
            _add(row, "extreme_fov")

    # 6. spacing_span（ADC / T2W 平面内间距两端）
    spacing_cfg = dict(strata_cfg.get("spacing_span") or {})
    min_per_tail = int(spacing_cfg.get("min_per_tail", 0))
    if min_per_tail > 0:
        for column in ("adc_spacing", "t2w_spacing"):
            with_spacing: list[tuple[float, Mapping[str, Any]]] = []
            for row in rows:
                spacing = parse_vector3(row.get(column), field_name=f"{column}[{row.get('case_id')}]")
                if spacing is None:
                    continue
                with_spacing.append((planar_spacing_mean(spacing), row))
            with_spacing.sort(key=lambda item: (item[0], str(item[1]["case_id"])))
            for _, row in with_spacing[:min_per_tail]:
                _add(row, "spacing_span")
            for _, row in with_spacing[-min_per_tail:]:
                _add(row, "spacing_span")

    # 7. 按配置补齐到 n_cases（仅当已填写）
    n_cases_cfg = sampling_cfg.get("n_cases")
    n_cases: int | None = None if n_cases_cfg is None else int(n_cases_cfg)
    n_cases_source = "minimum_required"
    if n_cases is not None:
        n_cases_source = "configured"
        for row in sorted(rows, key=lambda r: str(r["case_id"])):
            if len(selected) >= n_cases:
                break
            if str(row["case_id"]) not in selected:
                _add(row, "fill_deterministic")
        if len(selected) < n_cases:
            warnings.append(f"目标 n_cases={n_cases} 未满足（可用病例仅 {len(selected)}）")
        elif len(selected) > n_cases:
            warnings.append(
                f"最小必选集合（{len(selected)}）已超过配置 n_cases={n_cases}；不裁剪硬分层（需人工调整配额）"
            )

    out_rows = [
        {
            "case_id": cid,
            "strata": ";".join(info["strata"]),
            "selection_order": info["selection_order"],
            **{k: info["row"].get(k, "") for k in ("patient_id", "study_id", "center", "case_csPCa", *GEOMETRY_COLUMNS)},
        }
        for cid, info in sorted(selected.items(), key=lambda kv: kv[1]["selection_order"])
    ]
    return SamplingResult(
        rows=out_rows,
        per_stratum=hit_counts,
        n_cases_configured=n_cases,
        n_cases_source=n_cases_source,
        warnings=warnings,
    )


def build_blind_review_rows(
    sampling: SamplingResult,
    *,
    pairs: Sequence[str] = PAIRS,
    record_fields: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """构造双阅片者盲审记录模板：每个入选病例 × 每个评价对一行，字段留空。

    模板列默认取协议 `readers.record_fields`；不预填任何等级/结论。
    """
    fields = list(record_fields or ("case_id", "pair", "reader_a_level", "reader_b_level", "final_level", "disagreement", "adjudication_note"))
    rows: list[dict[str, Any]] = []
    for row in sampling.rows:
        for pair in pairs:
            entry = {name: "" for name in fields}
            entry["case_id"] = row["case_id"]
            entry["pair"] = pair
            rows.append(entry)
    return rows


def metrics_schema_document() -> dict[str, Any]:
    """定量对齐指标的输入/输出 schema（写入 `alignment_metrics_schema.json`）。"""
    return {
        "columns": list(ALIGNMENT_METRICS_COLUMNS),
        "pair_allowed": list(PAIRS),
        "metric_allowed": sorted(METRIC_UNITS),
        "metric_units": dict(METRIC_UNITS),
        "auxiliary_metrics": list(AUXILIARY_METRICS),
        "auxiliary_rule": "MI/NMI 只能作为辅助量，不得单独决定是否需要配准（protocol §4.2.4）",
        "value_rule": "必须为有限数值；单位 mm 的指标代表毫米位移/距离",
        "required_columns": ["case_id", "pair", "metric", "value", "unit"],
        "decision": None,  # 指标本身不产生决策字段
    }


def validate_alignment_metrics_rows(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """校验定量指标行；返回错误信息列表（不抛错，便于一次性报告全部问题）。"""
    errors: list[str] = []
    if not rows:
        return ["指标文件没有数据行"]
    for idx, row in enumerate(rows):
        prefix = f"row {idx + 1}"
        case_id = str(row.get("case_id", "")).strip()
        if not case_id:
            errors.append(f"{prefix}: case_id 为空")
        pair = str(row.get("pair", "")).strip()
        if pair not in PAIRS:
            errors.append(f"{prefix}: pair={pair!r} 不在允许集合 {list(PAIRS)}")
        metric = str(row.get("metric", "")).strip()
        if metric not in METRIC_UNITS:
            errors.append(f"{prefix}: metric={metric!r} 不在允许集合 {sorted(METRIC_UNITS)}")
        raw_value = row.get("value", "")
        try:
            value: float | None = float(raw_value)
        except (TypeError, ValueError):
            errors.append(f"{prefix}: value={raw_value!r} 不是数值")
            value = None
        if value is not None and not math.isfinite(value):
            errors.append(f"{prefix}: value={raw_value!r} 非有限值")
        if metric in METRIC_UNITS:
            unit = str(row.get("unit", "")).strip()
            if unit != METRIC_UNITS[metric]:
                errors.append(f"{prefix}: unit={unit!r} 与 metric={metric!r} 的期望单位 {METRIC_UNITS[metric]!r} 不一致")
    return errors


__all__ = [
    "ALIGNMENT_METRICS_COLUMNS",
    "AUXILIARY_METRICS",
    "GEOMETRY_COLUMNS",
    "METRIC_UNITS",
    "PAIRS",
    "SamplingResult",
    "build_blind_review_rows",
    "extent_volume",
    "metrics_schema_document",
    "parse_vector3",
    "plan_sampling",
    "planar_spacing_mean",
    "validate_alignment_metrics_rows",
]
