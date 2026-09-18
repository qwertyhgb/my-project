"""G0 协议工具的共享逻辑：配置加载、字段校验、内容哈希、运行元数据、输出目录隔离。

设计约束：

- **不导入 torch / nnunetv2**，不读取医学影像体素（只处理 YAML/JSON/CSV 元数据）；
- **不决定任何门的通过**：本模块不产生 `PASS` / `FROZEN` / 阈值，只做校验、哈希与记录；
- 缺失信息一律保持 `null` / `UNKNOWN` / `PENDING`，不得以默认值填充。
"""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

#: 项目根目录（src/zonal_reliability_fusion/protocols/common.py → parents[3]）
PROJECT_ROOT = Path(__file__).resolve().parents[3]

#: 缺证据 / 未决定的规范标记（不得用默认值替代）
UNKNOWN = "UNKNOWN"
PENDING = "PENDING"

#: 允许的协议状态；DRAFT 表示"工具已实现但门未冻结"
ALLOWED_PROTOCOL_STATUS = ("DRAFT", "FROZEN")

#: 工具输出禁止落入的目录前缀（训练产物与 nnU-Net 工作区），保证输出路径隔离
FORBIDDEN_OUTPUT_PREFIXES = (
    "outputs/checkpoints",
    "outputs/nnUNet_results",
    "workdir/nnUNet_raw",
    "workdir/nnUNet_preprocessed",
    "workdir/nnUNet_pred",
)


class ProtocolConfigError(ValueError):
    """协议配置缺失、类型非法或与协议约定冲突。"""


# --------------------------------------------------------------------------- 配置加载
def load_yaml_config(path: str | Path) -> dict:
    """读取协议 YAML 配置（顶层必须是映射）。"""
    path = Path(path)
    if not path.is_file():
        raise ProtocolConfigError(f"协议配置不存在: {path}")
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - 环境已确认可用
        raise ProtocolConfigError(
            f"缺少 PyYAML，无法读取 {path}；请先安装 pyyaml（conda activate lm）"
        ) from exc
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(doc, Mapping):
        raise ProtocolConfigError(f"{path} 顶层必须是映射，收到 {type(doc).__name__}")
    return dict(doc)


def require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProtocolConfigError(f"{name} 必须是映射，收到 {type(value).__name__}")
    return value


def require_str(value: Any, name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not value and not allow_empty):
        raise ProtocolConfigError(f"{name} 必须为非空字符串，收到 {value!r}")
    return value


def require_int(value: Any, name: str, *, minimum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ProtocolConfigError(f"{name} 必须为整数，收到 {value!r}")
    if minimum is not None and value < minimum:
        raise ProtocolConfigError(f"{name} 必须 >= {minimum}，收到 {value}")
    return value


def require_sequence(value: Any, name: str, *, minimum_len: int | None = None) -> list:
    if not isinstance(value, (list, tuple)):
        raise ProtocolConfigError(f"{name} 必须是列表，收到 {type(value).__name__}")
    if minimum_len is not None and len(value) < minimum_len:
        raise ProtocolConfigError(f"{name} 至少需要 {minimum_len} 项，收到 {len(value)}")
    return list(value)


def get_path(doc: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    """按 `a.b.c` 路径取嵌套字段（缺失返回 default，不报错）。"""
    node: Any = doc
    for key in dotted.split("."):
        if not isinstance(node, Mapping) or key not in node:
            return default
        node = node[key]
    return node


def validate_protocol_block(doc: Mapping[str, Any], *, expected_id: str) -> dict:
    """校验 `protocol` 块（id / version / status / 是否已见模型预测）。

    注意：本函数**不会**把状态提升为 FROZEN；只拒绝非法状态与错误的 id。
    """
    block = require_mapping(doc.get("protocol"), "protocol")
    protocol_id = require_str(block.get("id"), "protocol.id")
    if protocol_id != expected_id:
        raise ProtocolConfigError(f"protocol.id 应为 {expected_id!r}，收到 {protocol_id!r}")
    version = require_str(block.get("version"), "protocol.version")
    status = require_str(block.get("status"), "protocol.status")
    if status not in ALLOWED_PROTOCOL_STATUS:
        raise ProtocolConfigError(
            f"protocol.status 必须是 {ALLOWED_PROTOCOL_STATUS} 之一，收到 {status!r}"
        )
    if "sees_final_test_predictions" in block:
        # 显式拆分预测可见性的协议（G0-SAP v0.3+）：兼容字段必须恒等于 final-test 可见性；
        # 阶段状态与可见性的组合自洽性由 g0_sap.validate_sap_phases 强制（SAP-A/B/C）。
        if bool(block.get("sees_model_predictions", False)) != bool(
            block.get("sees_final_test_predictions", False)
        ):
            raise ProtocolConfigError(
                "protocol.sees_model_predictions（兼容字段）必须恒等于 protocol.sees_final_test_predictions"
                "；阶段一致性由 G0-SAP 校验器强制"
            )
    elif bool(block.get("sees_model_predictions", False)):
        raise ProtocolConfigError(
            "protocol.sees_model_predictions 为 true：协议要求「不得在任何模型结果可见后修改/执行本协议」"
            "（G0-R/G0-E/G0-SAP 均禁止查看模型预测）"
        )
    return {"id": protocol_id, "version": version, "status": status}


def find_unfilled_fields(doc: Mapping[str, Any], dotted_paths: Iterable[str]) -> list[str]:
    """返回仍为 `None` / 空 / `TODO` / `INSUFFICIENT DATA` 的字段路径清单（冻结就绪检查用）。"""
    unfilled: list[str] = []
    for dotted in dotted_paths:
        value = get_path(doc, dotted, default=None)
        if value is None:
            unfilled.append(dotted)
            continue
        if isinstance(value, str):
            token = value.strip().upper()
            if token in ("", "TODO", "TBD", "INSUFFICIENT_DATA", "INSUFFICIENT DATA"):
                unfilled.append(dotted)
        elif isinstance(value, (list, dict)) and len(value) == 0:
            unfilled.append(dotted)
    return unfilled


# --------------------------------------------------------------------------- 哈希 / 元数据
def sha256_file(path: str | Path) -> str:
    """文件 SHA256（只读；用于配置与元数据文件的可追溯校验值）。"""
    p = Path(path)
    if not p.is_file():
        raise ProtocolConfigError(f"文件不存在，无法计算哈希: {p}")
    digest = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def collect_hashes(paths: Sequence[str | Path], *, base: Path | None = None) -> dict[str, str]:
    """对存在文件计算 SHA256（缺失文件按 `MISSING` 记录，不伪造值）。"""
    base = Path(base) if base is not None else PROJECT_ROOT
    out: dict[str, str] = {}
    for raw in paths:
        p = Path(raw)
        p = p if p.is_absolute() else base / p
        key = str(p)
        out[key] = sha256_file(p) if p.is_file() else "MISSING"
    return out


def build_run_metadata(
    *,
    tool: str,
    config_path: str | Path | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict:
    """构造运行元数据（时间、解释器、平台、配置哈希、工具名）。

    刻意不导入 torch：G0 工具不得初始化 CUDA，也不应因环境探测产生 GPU 副作用。
    """
    metadata: dict[str, Any] = {
        "tool": tool,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "project_root": str(PROJECT_ROOT),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    try:
        import numpy  # noqa: PLC0415 - 轻量依赖，仅记录版本

        metadata["numpy"] = numpy.__version__
    except ImportError:  # pragma: no cover - numpy 为项目标准依赖
        metadata["numpy"] = None
    if config_path is not None:
        cfg = Path(config_path)
        cfg = cfg if cfg.is_absolute() else PROJECT_ROOT / cfg
        metadata["config_path"] = str(cfg)
        metadata["config_sha256"] = sha256_file(cfg) if cfg.is_file() else "MISSING"
    if extra:
        metadata["extra"] = dict(extra)
    return metadata


# --------------------------------------------------------------------------- 输出
def resolve_project_path(value: str | Path) -> Path:
    """把配置中的相对路径解析为项目内绝对路径。"""
    p = Path(value)
    return p if p.is_absolute() else PROJECT_ROOT / p


def assert_output_dir_isolated(path: str | Path) -> Path:
    """校验输出目录位于项目内、且不与训练产物 / nnU-Net 工作区重叠（输出路径隔离）。"""
    p = resolve_project_path(path)
    try:
        rel = p.resolve().relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise ProtocolConfigError(f"输出目录必须在项目内: {p}") from exc
    rel_posix = rel.as_posix()
    for prefix in FORBIDDEN_OUTPUT_PREFIXES:
        if rel_posix == prefix or rel_posix.startswith(prefix + "/"):
            raise ProtocolConfigError(
                f"输出目录与训练产物/nnU-Net 工作区重叠（禁止）: {rel_posix}（前缀 {prefix}）"
            )
    return p


def ensure_output_dir(path: str | Path) -> Path:
    """创建输出目录（先做隔离校验）。"""
    p = assert_output_dir_isolated(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False) + "\n", encoding="utf-8")
    return p


def write_rows_csv(path: str | Path, rows: Sequence[Mapping[str, Any]], *, fieldnames: Sequence[str] | None = None) -> Path:
    """写 CSV（行可以为空；字段名显式给定或取首行键）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    names = list(fieldnames) if fieldnames is not None else (list(rows[0].keys()) if rows else [])
    if not names:
        raise ProtocolConfigError(f"写 CSV 需要至少一个字段名: {p}")
    with open(p, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in names})
    return p


def timestamp_slug() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


__all__ = [
    "ALLOWED_PROTOCOL_STATUS",
    "FORBIDDEN_OUTPUT_PREFIXES",
    "PENDING",
    "PROJECT_ROOT",
    "ProtocolConfigError",
    "UNKNOWN",
    "assert_output_dir_isolated",
    "build_run_metadata",
    "collect_hashes",
    "ensure_output_dir",
    "find_unfilled_fields",
    "get_path",
    "load_yaml_config",
    "require_int",
    "require_mapping",
    "require_sequence",
    "require_str",
    "resolve_project_path",
    "sha256_file",
    "sha256_text",
    "timestamp_slug",
    "validate_protocol_block",
    "write_json",
    "write_rows_csv",
]
