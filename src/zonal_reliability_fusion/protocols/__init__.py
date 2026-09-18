"""G0 协议工具子包（G0-R / G0-E / G0-SAP 的机器可读载体、校验逻辑与图像 QC 核心）。

**模块边界（按模块区分，勿混用表述）**：

- `common` / `g0_r` / `g0_e` / `g0_sap`：只处理配置与元数据（YAML / JSON / CSV 清单），
  **不读取医学影像体素**、不导入 torch / nnunetv2；
- `g0_r_qc`：G0-R 图像 QC 核心逻辑。**研究者运行时只读物化影像体素**（T2W/ADC/HBV），
  仅用于对齐 QC（物理坐标重采样、棋盘格/边缘叠加、landmark 物理位移），
  **不修改任何数据、不训练、不推理、不执行任何配准**；模块被导入时不会读影像、不占 GPU。

共同边界：

- **不决定任何门的通过**：不写 `PASS`、不写 `frozen_decision`、不填写阈值；
  未知/无证据一律保持 `null` / `UNKNOWN` / `PENDING`；
- 输出一律写入项目内目录，并附带配置内容哈希与运行元数据，便于冻结与审计。

入口脚本：
- `scripts/audit/audit_picai_alignment_qc.py`（G0-R 抽样与盲审清单；只读元数据）；
- `scripts/audit/render_picai_alignment_qc.py`（G0-R 图像 QC **人工路径（历史草案）**：渲染 / 指标 / 校验三模式；
  研究者运行时只读物化影像体素；盲审材料**不含病灶标签与模型输出**；不配准、不判 PASS）；
- `scripts/audit/run_picai_alignment_qc_automated.py`（**G0-R Automated v0.2 当前主路径**：零人工阅片/landmark/
  双阅片/人工阈值；四组自动证据 + 合成位移校准 + 候选决策；只读物化影像、不写回、不联网、不判 PASS）；
- `scripts/audit/audit_g0e_candidates.py`（G0-E 候选清点与证据清单；只读文件清单/表头）；
- `scripts/evaluate/prepare_g0_sap_freeze.py`（G0-SAP 冻结载体与就绪检查；不运行评测）。

自动 QC 算法库：`g0_r_automated_qc`（纯计算；不读文件、不写文件、不联网；轴序与 mm 单位由合成测试锁定）。
"""
from . import g0_r_automated_qc
from .common import (
    PENDING,
    UNKNOWN,
    ProtocolConfigError,
    assert_output_dir_isolated,
    build_run_metadata,
    load_yaml_config,
    resolve_project_path,
    sha256_file,
    write_json,
    write_rows_csv,
)
from .g0_e import (
    audit_candidate,
    build_missing_evidence_rows,
    inventory_dataset,
    render_g0e_report_markdown,
)
from .g0_r import (
    ALIGNMENT_METRICS_COLUMNS,
    METRIC_UNITS,
    PAIRS,
    SamplingResult,
    build_blind_review_rows,
    plan_sampling,
    validate_alignment_metrics_rows,
)
from .g0_r_qc import (
    COORDINATE_SPACE_ALLOWED,
    COORDINATE_SPACE_NATIVE,
    RESAMPLE_ONLY_BANNER,
    BoundaryResult,
    LandmarkParseResult,
    PairGeometryRecord,
    build_landmark_template_rows,
    check_landmark_completeness,
    checkerboard_image,
    compute_boundary_metrics,
    compute_landmark_metrics,
    edge_overlay_image,
    geometry_from_record_fields,
    index_to_physical_lps,
    load_sampling_manifest,
    parse_strict_index,
    prepare_pair_arrays,
    read_boundary_source,
    resolve_series_paths,
    select_cases,
    select_slice_indices,
    summarize_landmark_metrics,
    validate_landmark_rows,
)
from .g0_sap import (
    POWER_MEMO_FIELDS,
    validate_freeze_readiness,
    validate_g0_sap_config,
)

# 导出清单（按 isort 规则排序；每个名字来自上方对应子模块的导入）
__all__ = [
    "ALIGNMENT_METRICS_COLUMNS",
    "COORDINATE_SPACE_ALLOWED",
    "COORDINATE_SPACE_NATIVE",
    "METRIC_UNITS",
    "PAIRS",
    "PENDING",
    "POWER_MEMO_FIELDS",
    "RESAMPLE_ONLY_BANNER",
    "UNKNOWN",
    "BoundaryResult",
    "LandmarkParseResult",
    "PairGeometryRecord",
    "ProtocolConfigError",
    "SamplingResult",
    "assert_output_dir_isolated",
    "audit_candidate",
    "build_blind_review_rows",
    "build_landmark_template_rows",
    "build_missing_evidence_rows",
    "build_run_metadata",
    "check_landmark_completeness",
    "checkerboard_image",
    "compute_boundary_metrics",
    "compute_landmark_metrics",
    "edge_overlay_image",
    "g0_r_automated_qc",
    "geometry_from_record_fields",
    "index_to_physical_lps",
    "inventory_dataset",
    "load_sampling_manifest",
    "load_yaml_config",
    "parse_strict_index",
    "plan_sampling",
    "prepare_pair_arrays",
    "read_boundary_source",
    "render_g0e_report_markdown",
    "resolve_project_path",
    "resolve_series_paths",
    "select_cases",
    "select_slice_indices",
    "sha256_file",
    "summarize_landmark_metrics",
    "validate_alignment_metrics_rows",
    "validate_freeze_readiness",
    "validate_g0_sap_config",
    "validate_landmark_rows",
    "write_json",
    "write_rows_csv",
]
