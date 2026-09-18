"""G0-E：外部候选清点与证据清单（只读元数据；缺证据一律 UNKNOWN）。

本模块只做两类事情：

1. **文件清单级清点**（`inventory_dataset`）：目录名、文件名、CSV 表头、行数——
   不打开 `.nii.gz` / `.mha` 体素，不读取影像强度或标签内容；
2. **证据状态汇总**（`audit_candidate`）：把 YAML 中已填写的字段标为 `VERIFIED`，
   把工具可自动得到的事实标为 `AUTO_ONLY`（仅供参考、**不作为判定依据**），
   其余一律 `UNKNOWN` / `PENDING`。

**绝不推断**候选数据集的“独立性”“可用性”或“等同任务”结论；
`verdict` 只允许来自配置（人工）或保持 `None`。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .common import UNKNOWN

#: 候选登记表中必须由研究者提供证据的字段（缺一即 UNKNOWN）
EVIDENCE_FIELDS: tuple[str, ...] = (
    "version",
    "source",
    "license",
    "usage_scope",
    "n_cases",
    "n_lesions",
    "modalities",
    "patient_overlap_with_picai",
    "label_semantics",
    "zonal_labels",
    "prior_independence",
    "metrics_supported",
    "verdict",
    "verdict_reason",
)

#: 文件名关键词 → 归类（仅文件名匹配，不代表语义已核实）
MODALITY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "t2": ("t2", "t2w"),
    "adc": ("adc",),
    "dwi": ("dwi", "hbv", "bval", "b_value"),
}
ANNOTATION_KEYWORDS: dict[str, tuple[str, ...]] = {
    "anatomy": ("anatomy",),
    "tumor": ("tumor", "lesion", "seg", "mask"),
}

_CASE_DIR_RE = re.compile(r"^\d+$")

#: 清点时最多展示的异常样例数（避免报告膨胀）
MAX_EXAMPLES = 10


def _read_csv_header(path: Path, *, count_rows: bool = True) -> dict[str, Any]:
    """读取 CSV 表头与数据行数（仅元数据；不解析医学数据）。"""
    import csv

    info: dict[str, Any] = {"path": str(path), "header": [], "n_rows": None, "error": None}
    try:
        with open(path, newline="", encoding="utf-8", errors="replace") as fh:
            reader = csv.reader(fh)
            header = next(reader, None)
            info["header"] = list(header) if header else []
            if count_rows:
                info["n_rows"] = sum(1 for _ in reader)
    except OSError as exc:  # pragma: no cover - 权限/IO 异常
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def inventory_dataset(root: str | Path, *, max_depth: int = 6, count_csv_rows: bool = True) -> dict:
    """对候选数据集做文件清单级清点（只读目录与文件名 + CSV 表头/行数）。

    Returns:
        字典包含：`root`、`exists`、`n_case_dirs`、`case_dir_examples`、`modality_files`、
        `annotation_files`、`empty_placeholder_count`、`csv_files`、`errors`、`notes`。
    """
    root = Path(root)
    inventory: dict[str, Any] = {
        "root": str(root),
        "exists": root.is_dir(),
        "n_case_dirs": 0,
        "case_dir_examples": [],
        "n_files_total": 0,
        "modality_files": {key: 0 for key in MODALITY_KEYWORDS},
        "annotation_files": {key: 0 for key in ANNOTATION_KEYWORDS},
        "empty_placeholder_count": 0,
        "csv_files": [],
        "errors": [],
        "notes": "文件清单级清点：不读取影像体素、不解析标签内容；AUTO 值不作为判定依据",
    }
    if not root.is_dir():
        inventory["errors"].append(f"候选根目录不存在: {root}")
        return inventory

    case_dirs: list[Path] = []
    for path in sorted(root.rglob("*")):
        try:
            depth = len(path.relative_to(root).parts)
        except ValueError:  # pragma: no cover - 防御性
            continue
        if depth > max_depth:
            continue
        if not path.is_dir():
            inventory["n_files_total"] += 1
            name = path.name.lower()
            if name.endswith((".nii", ".nii.gz", ".mha", ".nrrd")):
                for key, keywords in MODALITY_KEYWORDS.items():
                    if any(kw in name for kw in keywords):
                        inventory["modality_files"][key] += 1
                for key, keywords in ANNOTATION_KEYWORDS.items():
                    if any(kw in name for kw in keywords):
                        inventory["annotation_files"][key] += 1
                if name.startswith("empty"):
                    inventory["empty_placeholder_count"] += 1
            if name.endswith(".csv"):
                inventory["csv_files"].append(_read_csv_header(path, count_rows=count_csv_rows))
            continue
        if _CASE_DIR_RE.match(path.name):
            case_dirs.append(path)

    inventory["n_case_dirs"] = len(case_dirs)
    inventory["case_dir_examples"] = [str(p.relative_to(root)) for p in case_dirs[:MAX_EXAMPLES]]
    return inventory


def _field_status(value: Any) -> str:
    if value is None:
        return UNKNOWN
    if isinstance(value, str):
        token = value.strip().upper()
        if token in ("", "UNKNOWN", "TODO", "TBD", "PENDING", "INSUFFICIENT_DATA"):
            return UNKNOWN
    if isinstance(value, (list, dict)) and len(value) == 0:
        return UNKNOWN
    return "VERIFIED"


def audit_candidate(candidate: Mapping[str, Any], inventory: Mapping[str, Any] | None) -> dict:
    """汇总一个候选的证据状态；不做任何独立性/可用性推断。"""
    name = str(candidate.get("name", "")).strip() or UNKNOWN
    audit: dict[str, Any] = {
        "name": name,
        "excluded_as_lesion_test": bool(candidate.get("excluded_as_lesion_test", False)),
        "local_root": candidate.get("local_root"),
        "fields": {},
        "auto_only": {},
        "missing": [],
    }
    for field in EVIDENCE_FIELDS:
        value = candidate.get(field)
        status = _field_status(value)
        if audit["excluded_as_lesion_test"] and field in ("verdict", "verdict_reason"):
            status = "CONFIGURED_EXCLUDED"  # 配置已声明排除（仍不是通过）
        audit["fields"][field] = {"value": value, "status": status}
        if status == UNKNOWN:
            audit["missing"].append(field)

    if inventory:
        audit["auto_only"] = {
            "local_root_exists": bool(inventory.get("exists")),
            "n_case_dirs": inventory.get("n_case_dirs"),
            "n_files_total": inventory.get("n_files_total"),
            "modality_files": dict(inventory.get("modality_files") or {}),
            "annotation_files": dict(inventory.get("annotation_files") or {}),
            "empty_placeholder_count": inventory.get("empty_placeholder_count"),
            "csv_files": [item.get("path") for item in (inventory.get("csv_files") or [])],
            "note": "AUTO_ONLY：来自文件/表头清单，不作为独立性、任务等价性或可用性的判定依据",
        }
    return audit


def build_missing_evidence_rows(audits: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """生成“研究者待补证据清单”（每候选每缺失字段一行）。"""
    rows: list[dict[str, Any]] = []
    for audit in audits:
        for field in audit.get("missing", []):
            rows.append(
                {
                    "candidate": audit.get("name"),
                    "missing_field": field,
                    "required_evidence": _evidence_hint(str(field)),
                    "status": UNKNOWN,
                }
            )
        if audit.get("excluded_as_lesion_test"):
            rows.append(
                {
                    "candidate": audit.get("name"),
                    "missing_field": "（已配置排除）",
                    "required_evidence": "无需补充；仅作为 G0-E 决策树中的 rejected 引用",
                    "status": "CONFIGURED_EXCLUDED",
                }
            )
    return rows


def _evidence_hint(field: str) -> str:
    hints = {
        "version": "数据集版本/发布日期与获取渠道",
        "source": "来源（机构/论文/DOI）",
        "license": "许可文件或链接 + 允许的使用范围",
        "usage_scope": "研究/再分发/商用边界",
        "n_cases": "病例数（附统计口径）",
        "n_lesions": "病灶数（按 §4.1 病灶定义）",
        "modalities": "实际可用模态清单与文件对应关系",
        "patient_overlap_with_picai": "与 PI-CAI 的患者重叠排查方法与结果（必须为 0 才可作 test）",
        "label_semantics": "病灶定义/阳性阴性定义/病理标准逐条对照（见 label_semantics_required_fields）",
        "zonal_labels": "PZ/TZ 是否存在、编码与权威来源",
        "prior_independence": "分区器训练数据是否覆盖目标病例的溯源证据（见 prior_independence_required_fields）",
        "metrics_supported": "该数据支持的指标清单（超出部分禁止报告）",
        "verdict": "依据 decision_tree 得出的结论（eligible_external_test / exploratory_only / rejected）",
        "verdict_reason": "结论理由（引用证据文件）",
    }
    return hints.get(field, "补充可溯源证据")


def render_g0e_report_markdown(
    audits: Sequence[Mapping[str, Any]],
    *,
    protocol_version: str,
    protocol_status: str,
    generated_at: str,
) -> str:
    """渲染审计报告模板（Markdown）；所有未证实项标注 UNKNOWN/PENDING。"""
    lines: list[str] = [
        "# G0-E 候选审计报告（自动生成模板）",
        "",
        f"- 协议版本：`{protocol_version}`（状态：**{protocol_status}**）",
        f"- 生成时间：{generated_at}",
        "- 本报告由文件清单级审计生成：**不读取影像体素**；`AUTO_ONLY` 值不作为判定依据；",
        "  未填写字段一律 `UNKNOWN`，**不得**据此推断独立性或可用性。",
        "",
        "## 候选汇总",
        "",
        "| 候选 | 根目录 | 目录存在 | case 目录数 | CSV 数 | 缺失证据字段数 | 结论 |",
        "|---|---|:---:|---:|---:|---:|---|",
    ]
    for audit in audits:
        auto = audit.get("auto_only") or {}
        verdict = (audit.get("fields", {}).get("verdict") or {}).get("value")
        if audit.get("excluded_as_lesion_test"):
            verdict = "CONFIGURED_EXCLUDED"
        lines.append(
            "| {name} | {root} | {exists} | {n_dirs} | {n_csv} | {n_missing} | {verdict} |".format(
                name=audit.get("name"),
                root=audit.get("local_root") or "-",
                exists="是" if auto.get("local_root_exists") else "否",
                n_dirs=auto.get("n_case_dirs", "-"),
                n_csv=len(auto.get("csv_files") or []),
                n_missing=len(audit.get("missing", [])),
                verdict=verdict if verdict else UNKNOWN,
            )
        )
    lines += ["", "## 待补证据（研究者）", ""]
    for audit in audits:
        if not audit.get("missing"):
            continue
        lines.append(f"### {audit.get('name')}")
        for field in audit["missing"]:
            lines.append(f"- `{field}`：{_evidence_hint(str(field))}")
        lines.append("")
    lines += [
        "## 决策树（逐候选执行，冻结前不得跳过）",
        "",
        "见 `configs/protocols/g0_e_independent_test.yaml` 的 `decision_tree`；",
        "本报告不产出 `frozen_decision`（保持 `null`）。",
    ]
    return "\n".join(lines) + "\n"


__all__ = [
    "ANNOTATION_KEYWORDS",
    "EVIDENCE_FIELDS",
    "MODALITY_KEYWORDS",
    "audit_candidate",
    "build_missing_evidence_rows",
    "inventory_dataset",
    "render_g0e_report_markdown",
]
