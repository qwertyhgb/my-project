"""PZ/TZ prior 的 plan-space 物化逻辑（与固定 nnU-Net v2.6.2 预处理几何完全一致）。

设计边界：

- **不重新发明几何**：transpose → crop(properties 中记录的图像 nonzero bbox) → resample，
  顺序与参数照抄 nnU-Net `DefaultPreprocessor.run_case_npy`；重采样函数由调用方注入
  （脚本注入固定的 `configuration_manager.resampling_fn_seg`，最近邻）。
- **病灶 oracle fail-closed**：把同一转换应用到原始 lesion 标签，结果必须与 `.b2nd` 中 seg 完全一致；
  materializer 在**写文件之前**再强制校验一次 oracle 字段（即使 converter 忘记抛异常也不写盘）。
- **严格离散标签**：`label_to_pz_tz` 只接受数值精确属于 {0,1,2} 的数组（finite + 精确成员判断），
  拒绝 1.5 / 2.1 / -0.2 / NaN / Inf，并在错误信息中保留**原始非法值**。
- **原子写入**：所有产物（npz / json / manifest / summary）先写同目录临时文件 → flush + fsync →
  `os.replace`；任一步失败不留看似完整的最终文件；默认不覆盖。
- **sidecar 完整性**：每个 sidecar 都带 source / configuration / plans_sha256 / 输入哈希 /
  output_sha256 / oracle 结果；读取端逐项校验，数组被篡改必然失败。
- 本模块只做纯计算与文件编排，不 import nnunetv2。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ZONAL_BACKGROUND = 0
ZONAL_PZ = 1
ZONAL_TZ = 2
#: 合法标签集合（0/1/2）
ZONAL_LABELS: tuple[int, ...] = (ZONAL_BACKGROUND, ZONAL_PZ, ZONAL_TZ)
#: sidecar 通道语义（channel 0 = PZ，channel 1 = TZ）；WG 不得进入
ZONE_CHANNEL_NAMES: tuple[str, ...] = ("PZ", "TZ")
#: 合法 prior 来源（必须显式选择，不得静默默认）
ZONAL_SOURCES: tuple[str, ...] = ("yuan", "hevi")
ORACLE_MISMATCH_TOLERANCE = 0
MANIFEST_FILENAME = "manifest.json"
SUMMARY_FILENAME = "materialization_summary.json"


class ZonalPriorError(RuntimeError):
    """prior 几何、标签、oracle、完整性校验或读写失败。"""


# --------------------------------------------------------------------------- 几何
@dataclass(frozen=True)
class PlanSpaceGeometry:
    """从病例 properties + plan 提取的 plan-space 几何（transpose → crop → resample）。"""

    transpose_forward: tuple[int, ...]
    original_spacing: tuple[float, ...]  # 已按 transpose_forward 重排
    target_spacing: tuple[float, ...]
    bbox_used_for_cropping: tuple[tuple[int, int], ...]
    shape_before_cropping: tuple[int, ...]
    target_shape: tuple[int, ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "transpose_forward": list(self.transpose_forward),
            "original_spacing": list(self.original_spacing),
            "target_spacing": list(self.target_spacing),
            "bbox_used_for_cropping": [list(b) for b in self.bbox_used_for_cropping],
            "shape_before_cropping": list(self.shape_before_cropping),
            "target_shape": None if self.target_shape is None else list(self.target_shape),
        }


def geometry_from_properties(
    properties: Mapping[str, Any],
    *,
    transpose_forward: Sequence[int],
    target_spacing: Sequence[float],
) -> PlanSpaceGeometry:
    """按 nnU-Net 规则从 properties 提取几何（缺失必需键即报错，不猜默认值）。"""
    transpose = tuple(int(i) for i in transpose_forward)
    if sorted(transpose) != [0, 1, 2]:
        raise ZonalPriorError(f"transpose_forward 必须是 0/1/2 的排列，收到 {transpose}")
    if "spacing" not in properties:
        raise ZonalPriorError("properties 缺少 'spacing'（无法重建 plan-space 几何）")
    if "bbox_used_for_cropping" not in properties:
        raise ZonalPriorError(
            "properties 缺少 'bbox_used_for_cropping'：拒绝重建裁剪框（必须复用 nnU-Net 记录）"
        )
    if "shape_before_cropping" not in properties:
        raise ZonalPriorError("properties 缺少 'shape_before_cropping'")
    spacing = tuple(float(v) for v in properties["spacing"])
    bbox_raw = properties["bbox_used_for_cropping"]
    bbox = tuple((int(b[0]), int(b[1])) for b in bbox_raw)
    if len(bbox) != 3:
        raise ZonalPriorError(f"bbox_used_for_cropping 必须是 3 个 (start, stop)，收到 {bbox_raw}")
    return PlanSpaceGeometry(
        transpose_forward=transpose,
        original_spacing=tuple(spacing[i] for i in transpose),
        target_spacing=tuple(float(v) for v in target_spacing),
        bbox_used_for_cropping=bbox,
        shape_before_cropping=tuple(int(v) for v in properties["shape_before_cropping"]),
    )


def convert_label_to_plan_space(
    label: np.ndarray,
    geometry: PlanSpaceGeometry,
    *,
    resample_fn: Callable[[np.ndarray, Sequence[int], Sequence[float], Sequence[float]], np.ndarray],
) -> np.ndarray:
    """把**原始网格**的标签（3D 或 [1,...]）转换到 plan 空间（transpose → crop → resample）。"""
    arr = np.asarray(label)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        raise ZonalPriorError(f"标签必须是 3D（或 [1,D,H,W]），收到 {arr.shape}")
    arr = arr.transpose(geometry.transpose_forward)
    if tuple(arr.shape) != tuple(geometry.shape_before_cropping):
        raise ZonalPriorError(
            f"transpose 后形状 {tuple(arr.shape)} != properties.shape_before_cropping "
            f"{tuple(geometry.shape_before_cropping)}；原始标签与 .pkl 不是同一网格（拒绝继续）"
        )
    slicer = tuple(slice(b[0], b[1]) for b in geometry.bbox_used_for_cropping)
    arr = arr[slicer]
    cropped_shape = tuple(int(v) for v in arr.shape)
    new_shape = _target_shape(cropped_shape, geometry)
    out = resample_fn(arr[None], new_shape, geometry.original_spacing, geometry.target_spacing)
    out = np.asarray(out)
    if out.ndim == 4:
        if out.shape[0] != 1:
            raise ZonalPriorError(f"resample 返回了 {out.shape[0]} 个通道，期望 1")
        out = out[0]
    if tuple(out.shape) != tuple(new_shape):
        raise ZonalPriorError(f"resample 后形状 {tuple(out.shape)} != 目标 {tuple(new_shape)}")
    return out


def _target_shape(cropped_shape: Sequence[int], geometry: PlanSpaceGeometry) -> tuple[int, ...]:
    """目标形状 = 按 spacing 比例缩放后的形状（与 nnU-Net compute_new_shape 同规则，四舍五入）。"""
    cropped = tuple(int(v) for v in cropped_shape)
    if geometry.target_shape is not None:
        return tuple(int(v) for v in geometry.target_shape)
    new_shape = []
    for size, cur, tgt in zip(cropped, geometry.original_spacing, geometry.target_spacing):
        new_shape.append(int(np.round(size * (cur / tgt))))
    return tuple(new_shape)


def label_to_pz_tz(label: np.ndarray) -> np.ndarray:
    """0/1/2 标签 → `[2, D, H, W]` float32（channel 0 = PZ，channel 1 = TZ）。

    **严格离散**：只接受数值精确属于 {0,1,2} 的数组；先检查 finite，再用精确成员判断，
    不做 `int(v)` 截断（1.5 / 2.1 / -0.2 / NaN / Inf 一律拒绝，并在报错中保留原始值）。
    """
    arr = np.asarray(label)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        raise ZonalPriorError(f"标签必须是 3D，收到 {arr.shape}")
    if not np.isfinite(arr).all():
        bad = arr[~np.isfinite(arr)]
        raise ZonalPriorError(f"标签含非有限值（NaN/Inf），示例：{bad.reshape(-1)[:5].tolist()}")
    member = np.isin(arr, np.asarray(ZONAL_LABELS))
    if not bool(member.all()):
        invalid = np.unique(arr[~member])
        raise ZonalPriorError(
            f"标签含非法取值 {invalid.tolist()}（只允许精确取值 {list(ZONAL_LABELS)}；"
            "禁止小数/截断后的伪合法值）"
        )
    labels = arr.astype(np.int64, copy=False)
    out = np.zeros((len(ZONE_CHANNEL_NAMES), *labels.shape), dtype=np.float32)
    out[0] = (labels == ZONAL_PZ).astype(np.float32)
    out[1] = (labels == ZONAL_TZ).astype(np.float32)
    return np.ascontiguousarray(out)


def validate_prior_array(prior: np.ndarray, *, tol: float = 1e-4) -> None:
    """prior 契约：shape [2,D,H,W]、有限、逐通道 [0,1]、PZ+TZ <= 1+tol、无 WG 通道。"""
    arr = np.asarray(prior)
    if arr.ndim != 4 or arr.shape[0] != len(ZONE_CHANNEL_NAMES):
        raise ZonalPriorError(
            f"prior 形状必须为 [{len(ZONE_CHANNEL_NAMES)}, D, H, W]（{list(ZONE_CHANNEL_NAMES)}），收到 {arr.shape}"
        )
    if not np.isfinite(arr).all():
        raise ZonalPriorError("prior 含非有限值")
    if arr.min() < -tol or arr.max() > 1.0 + tol:
        raise ZonalPriorError(f"prior 取值必须在 [0,1]，实际 [{float(arr.min())}, {float(arr.max())}]")
    overlap = float((arr[0] + arr[1]).max())
    if overlap > 1.0 + tol:
        raise ZonalPriorError(f"PZ 与 TZ 重叠：max(PZ+TZ)={overlap:.6f} > 1+{tol}")


# --------------------------------------------------------------------------- oracle
def oracle_check_lesion(converted_binary: np.ndarray, reference_seg: np.ndarray) -> dict[str, Any]:
    """把转换后的 lesion 与 `.b2nd` 中的 seg 逐体素比较（oracle）。"""
    conv = np.asarray(converted_binary)
    ref = np.asarray(reference_seg)
    if ref.ndim == 4 and ref.shape[0] == 1:
        ref = ref[0]
    report: dict[str, Any] = {
        "reference_shape": list(ref.shape),
        "converted_shape": list(conv.shape),
        "shape_match": tuple(conv.shape) == tuple(ref.shape),
    }
    if not report["shape_match"]:
        report.update({"exact_match": False, "mismatch_voxels": None, "reason": "shape_mismatch"})
        return report
    conv_bin = (conv > 0).astype(np.uint8)
    ref_bin = (ref > 0).astype(np.uint8)
    mismatch = int(np.count_nonzero(conv_bin != ref_bin))
    report.update(
        {
            "mismatch_voxels": mismatch,
            "converted_fg_voxels": int(conv_bin.sum()),
            "reference_fg_voxels": int(ref_bin.sum()),
            "exact_match": mismatch <= ORACLE_MISMATCH_TOLERANCE,
        }
    )
    return report


def require_oracle_pass(metadata: Mapping[str, Any], *, case_id: str = "") -> dict[str, Any]:
    """materializer 侧的 fail-closed 校验：oracle 必须存在且严格通过，否则拒绝写盘。"""
    oracle = metadata.get("oracle")
    if not isinstance(oracle, Mapping):
        raise ZonalPriorError(f"{case_id}: oracle rejection — metadata 缺少 oracle 结果（拒绝写入）")
    problems: list[str] = []
    if oracle.get("exact_match") is not True:
        problems.append(f"exact_match={oracle.get('exact_match')!r}")
    if oracle.get("shape_match") is not True:
        problems.append(f"shape_match={oracle.get('shape_match')!r}")
    mismatch = oracle.get("mismatch_voxels")
    if not isinstance(mismatch, int) or isinstance(mismatch, bool) or mismatch != 0:
        problems.append(f"mismatch_voxels={mismatch!r}")
    if problems:
        raise ZonalPriorError(f"{case_id}: oracle rejection — {', '.join(problems)}（拒绝写入 sidecar）")
    return dict(oracle)


# --------------------------------------------------------------------------- 哈希 / 原子写
def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(array))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def json_sha256(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


#: SHA256 十六进制格式（小写 64 位）；空值/大写/非 hex 一律视为非法
SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


def is_sha256_hex(value: Any) -> bool:
    """严格判断是否为非空、小写、64 位十六进制 SHA256。"""
    return bool(isinstance(value, str) and SHA256_HEX_RE.match(value))


def normalize_prior(prior: np.ndarray) -> np.ndarray:
    """把 prior 统一规范为 **contiguous float32**（哈希、写盘、校验共用同一个对象语义）。

    防止 float64 输入出现「哈希按 float64 计算、NPZ 按 float32 落盘」的不一致。
    """
    return np.ascontiguousarray(np.asarray(prior, dtype=np.float32))


def metadata_sha256_of(meta: Mapping[str, Any]) -> str:
    """sidecar 元数据的唯一哈希定义：**实际 JSON 对象**的 canonical JSON SHA256。"""
    return json_sha256(dict(meta))


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    """同目录临时文件 → flush + fsync → os.replace；失败不留最终文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        tmp = Path(handle.name)
        try:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            with contextlib_suppress():
                tmp.unlink()
            raise
    try:
        os.replace(tmp, path)
    except BaseException:
        with contextlib_suppress():
            tmp.unlink()
        raise


def contextlib_suppress():
    import contextlib

    return contextlib.suppress(FileNotFoundError)


def write_json_atomic(path: str | Path, doc: Any) -> Path:
    path = Path(path)
    _atomic_write_bytes(path, (json.dumps(doc, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))
    return path


def _npz_bytes(prior: np.ndarray) -> bytes:
    import io

    buffer = io.BytesIO()
    np.savez_compressed(buffer, pz_tz=np.asarray(prior, dtype=np.float32))
    return buffer.getvalue()


def write_npz_atomic(path: str | Path, prior: np.ndarray) -> Path:
    path = Path(path)
    _atomic_write_bytes(path, _npz_bytes(prior))
    return path


def read_sidecar_metadata(meta_path: str | Path) -> dict[str, Any]:
    path = Path(meta_path)
    if not path.is_file():
        raise ZonalPriorError(f"sidecar 元数据不存在: {path}")
    try:
        doc = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ZonalPriorError(f"sidecar 元数据损坏（JSON 解析失败）: {path}: {exc}") from exc
    if not isinstance(doc, dict):
        raise ZonalPriorError(f"sidecar 元数据必须是 JSON 对象: {path}")
    return doc


def read_sidecar_prior(npz_path: str | Path) -> np.ndarray:
    """读取 NPZ 中的 `pz_tz`，**保留真实存储 dtype**（不做 float32 转换）。

    调用方必须先自行校验 `array.dtype`（`validate_sidecar` 要求真实 dtype == float32），
    再按需用 `normalize_prior()` 转成 contiguous float32 参与哈希/契约校验。
    无条件转换会让「float16 存储 + float32 元数据」这类不一致悄悄通过。
    """
    path = Path(npz_path)
    if not path.is_file():
        raise ZonalPriorError(f"sidecar 数组不存在: {path}")
    try:
        with np.load(path) as npz:
            if "pz_tz" not in npz:
                raise ZonalPriorError(f"sidecar 缺少 'pz_tz' 数组: {path}")
            return np.ascontiguousarray(npz["pz_tz"])
    except ZonalPriorError:
        raise
    except Exception as exc:  # 损坏文件必须显式失败
        raise ZonalPriorError(f"sidecar 数组无法读取（可能损坏）: {path}: {type(exc).__name__}: {exc}") from exc


# --------------------------------------------------------------------------- sidecar
_REQUIRED_META_KEYS = (
    "source",
    "configuration",
    "plans_sha256",
    "input_sha256",
    "output_shape",
    "output_dtype",
    "output_channel_semantics",
    "output_sha256",
    "oracle",
)


def sidecar_paths(out_root: str | Path, case_id: str) -> tuple[Path, Path]:
    root = Path(out_root)
    return root / f"{case_id}.npz", root / f"{case_id}.json"


def finalize_sidecar_metadata(
    case_id: str,
    prior: np.ndarray,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """补齐写盘必需字段并**统一定义** output_sha256（基于规范化后的 float32 数组）。"""
    arr = normalize_prior(prior)
    meta = dict(metadata)
    meta["case_id"] = case_id
    meta["output_shape"] = list(arr.shape)
    meta["output_dtype"] = "float32"
    meta["output_channel_semantics"] = list(ZONE_CHANNEL_NAMES)
    meta["output_sha256"] = array_sha256(arr)
    missing = [key for key in _REQUIRED_META_KEYS if key not in meta]
    if missing:
        raise ZonalPriorError(f"{case_id}: sidecar 元数据缺少必需字段 {missing}（拒绝写入）")
    if not is_sha256_hex(meta["output_sha256"]):
        raise ZonalPriorError(f"{case_id}: output_sha256 必须为非空 64 位小写 hex（拒绝写入）")
    return meta


def _stage_bytes(path: Path, payload: bytes) -> Path:
    """把 payload 写入同目录临时文件并 flush+fsync；返回临时文件路径（尚未提交）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        tmp = Path(handle.name)
        try:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            with contextlib_suppress():
                tmp.unlink()
            raise
    return tmp


def _cleanup(paths: Sequence[Path]) -> None:
    for path in paths:
        with contextlib_suppress():
            Path(path).unlink()


class SidecarCommitError(ZonalPriorError):
    """sidecar 成对提交失败：同时携带原始错误与回滚（恢复/清理）错误。"""

    def __init__(
        self,
        case_id: str,
        stage: str,
        original: BaseException,
        recovery_errors: Sequence[str],
    ) -> None:
        self.case_id = case_id
        self.stage = stage
        self.original = original
        self.recovery_errors = list(recovery_errors)
        detail = f"{case_id}: sidecar 提交在 {stage} 阶段失败：{type(original).__name__}: {original}"
        if self.recovery_errors:
            detail += "；回滚未能完全完成（需要人工检查）：" + " | ".join(self.recovery_errors)
        else:
            detail += "；回滚已完成（旧文件已恢复、本轮新文件/临时文件已清除）"
        super().__init__(detail)


def _require_writable_metadata(case_id: str, metadata: Mapping[str, Any]) -> dict[str, Any]:
    """fail-closed：写盘前强制校验身份字段（不依赖上游 materializer 的检查）。"""
    meta = dict(metadata)
    source = meta.get("source")
    if source not in ZONAL_SOURCES:
        raise ZonalPriorError(
            f"{case_id}: metadata.source={source!r} 非法（必须是 {list(ZONAL_SOURCES)}），拒绝写入"
        )
    if not isinstance(meta.get("configuration"), str) or not meta["configuration"]:
        raise ZonalPriorError(f"{case_id}: metadata.configuration 必须为非空字符串，拒绝写入")
    if not is_sha256_hex(meta.get("plans_sha256")):
        raise ZonalPriorError(f"{case_id}: metadata.plans_sha256 必须为 64 位小写 hex，拒绝写入")
    input_hashes = meta.get("input_sha256")
    if not isinstance(input_hashes, Mapping) or not input_hashes:
        raise ZonalPriorError(f"{case_id}: metadata.input_sha256 必须为非空映射，拒绝写入")
    if any(not is_sha256_hex(v) for v in input_hashes.values()):
        raise ZonalPriorError(f"{case_id}: metadata.input_sha256 含非法哈希（必须为非空 64 位小写 hex）")
    if not is_sha256_hex(meta.get("oracle_sha256_placeholder", "0" * 64)):  # pragma: no cover - 防御性
        raise ZonalPriorError(f"{case_id}: metadata 哈希字段格式非法，拒绝写入")
    require_oracle_pass(meta, case_id=case_id)  # 写盘侧 fail-closed（oracle 必须严格通过）
    return meta


def _stage_json_bytes(meta: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(meta), indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")


def _unique_backup_path(final: Path) -> Path:
    """本轮唯一的备份文件名（绝不删除/复用固定名历史 .bak）。"""
    import uuid

    return final.with_name(f".{final.name}.{os.getpid()}.{uuid.uuid4().hex[:10]}.bak")


def _rollback_pair(
    *,
    committed: Sequence[Path],
    staged: Sequence[Path],
    backups: Sequence[tuple[Path, Path]],
) -> list[str]:
    """尽最大可能回滚：删除本轮新文件与临时文件 → 恢复全部旧文件。返回未能完成的错误描述。"""
    errors: list[str] = []
    for path in [*committed, *staged]:
        try:
            Path(path).unlink(missing_ok=True)
        except OSError as exc:  # pragma: no cover - 极端 IO 失败
            errors.append(f"删除 {path} 失败: {type(exc).__name__}: {exc}")
    for final, backup in backups:
        try:
            os.replace(backup, final)
        except OSError as exc:
            errors.append(
                f"恢复旧文件 {final} 失败（备份仍在 {backup}）: {type(exc).__name__}: {exc}"
            )
    # 备份若已被消费（replace 成功）会自动消失；仍存在的孤立备份按「恢复失败」处理，不静默删除
    for _final, backup in backups:
        if Path(backup).exists():
            errors.append(f"备份未消费且未能恢复: {backup}")
    return errors


def commit_sidecar_pair(
    out_root: str | Path,
    case_id: str,
    prior: np.ndarray,
    metadata: Mapping[str, Any],
) -> tuple[Path, Path, dict[str, Any]]:
    """**异常安全**地成对提交 `<case>.npz` + `<case>.json`（staging → 备份 → 提交 → 落盘复核）。

    说明（不夸大）：两个独立路径的 `os.replace` 不是崩溃级事务原子；这里保证的是
    **异常安全 + 失败回滚 + 读取端 fail-closed**：
    - 任一步异常都会删除本轮新文件、恢复全部旧文件、清理本轮临时文件与备份；
    - 回滚不彻底时抛出 `SidecarCommitError`，其中同时包含原始错误与恢复错误（绝不静默）；
    - 读取端（`validate_sidecar` / store / materializer）在配对不完整时一律拒绝。
    """
    out_root = Path(out_root)
    npz_path, meta_path = sidecar_paths(out_root, case_id)
    arr = normalize_prior(prior)
    validate_prior_array(arr)  # fail-closed：写盘前强制数值契约
    meta = _require_writable_metadata(case_id, finalize_sidecar_metadata(case_id, arr, metadata))

    staged: list[Path] = []
    backups: list[tuple[Path, Path]] = []
    committed: list[Path] = []
    stage = "staging"
    try:
        npz_tmp = _stage_bytes(npz_path, _npz_bytes(arr))
        staged.append(npz_tmp)
        meta_tmp = _stage_bytes(meta_path, _stage_json_bytes(meta))
        staged.append(meta_tmp)

        stage = "backup"
        for final in (npz_path, meta_path):
            if final.exists():
                backup = _unique_backup_path(final)
                os.replace(final, backup)
                backups.append((final, backup))

        stage = "commit"
        os.replace(npz_tmp, npz_path)
        committed.append(npz_path)
        os.replace(meta_tmp, meta_path)
        committed.append(meta_path)

        # 落盘复核：以**实际落盘 JSON** 与**实际落盘数组**为准
        stage = "verify"
        final_meta = read_sidecar_metadata(meta_path)
        if canonical_json(final_meta) != canonical_json(meta):
            raise ZonalPriorError(f"{case_id}: 落盘元数据与预期不一致（拒绝认定为完成）")
        if not is_sha256_hex(final_meta.get("output_sha256")):
            raise ZonalPriorError(f"{case_id}: 落盘 output_sha256 非法")
        stored = read_sidecar_prior(npz_path)
        if stored.dtype != np.float32:
            raise ZonalPriorError(f"{case_id}: 落盘数组 dtype={stored.dtype} 必须为 float32")
        if array_sha256(stored) != final_meta["output_sha256"]:
            raise ZonalPriorError(f"{case_id}: 落盘数组 SHA256 与元数据不一致（拒绝认定为完成）")
        if list(stored.shape) != list(final_meta.get("output_shape") or []):
            raise ZonalPriorError(f"{case_id}: 落盘数组 shape 与元数据不一致（拒绝认定为完成）")
    except BaseException as exc:
        recovery_errors = _rollback_pair(committed=committed, staged=staged, backups=backups)
        raise SidecarCommitError(case_id, stage, exc, recovery_errors) from exc
    _cleanup([backup for _final, backup in backups])
    return npz_path, meta_path, final_meta


# --------------------------------------------------------------------------- 统一深度校验
@dataclass(frozen=True)
class ValidatedSidecar:
    """统一深度校验的结果（元数据 + 已读取数组 + 两种哈希 + 数组是否真校验过）。"""

    case_id: str
    metadata: dict[str, Any]
    prior: np.ndarray | None
    metadata_sha256: str
    output_sha256: str
    array_hash_checked: bool


def validate_sidecar(
    out_root: str | Path,
    case_id: str,
    *,
    expected_source: str | None = None,
    expected_configuration: str | None = None,
    expected_plans_sha256: str | None = None,
    expected_input_sha256: Mapping[str, Any] | None = None,
    expected_metadata_sha256: str | None = None,
    expected_output_sha256: str | None = None,
    expected_array_shape: Sequence[int] | None = None,
    verify_array_hash: bool = True,
    read_array: bool = True,
) -> ValidatedSidecar:
    """**唯一的严格 sidecar 校验器**（materializer / store / resume 共用同一套规则）。

    校验：文件存在 → metadata.case_id 一致 → source / configuration / plans_sha256 / input_sha256
    与期望一致且格式合法 → metadata_sha256（若给出期望）→ output_sha256 格式与（若给出）期望 →
    oracle 严格通过 → 数组（`read_array`）：**真实存储 dtype == float32**、shape 与元数据及
    （若给出）`expected_array_shape` 一致、channel semantics == [PZ,TZ]、数组 SHA256 == output_sha256、
    数值契约（finite / [0,1] / PZ+TZ<=1）。任一不符即抛 `ZonalPriorError`。
    返回的 `prior` 已经过 `normalize_prior`（contiguous float32）。
    """
    npz_path, meta_path = sidecar_paths(out_root, case_id)
    if not meta_path.is_file():
        raise ZonalPriorError(f"{case_id}: sidecar 元数据不存在: {meta_path}")
    if read_array and not npz_path.is_file():
        raise ZonalPriorError(f"{case_id}: sidecar 数组不存在: {npz_path}")
    meta = read_sidecar_metadata(meta_path)
    if meta.get("case_id") != case_id:
        raise ZonalPriorError(
            f"{case_id}: sidecar metadata.case_id={meta.get('case_id')!r} 与请求的病例号不一致"
        )
    if expected_source is not None and meta.get("source") != expected_source:
        raise ZonalPriorError(
            f"{case_id}: sidecar 来源 {meta.get('source')!r} 与期望 {expected_source!r} 不一致"
            "（Yuan/Hevi 不得混用）"
        )
    if expected_configuration is not None and meta.get("configuration") != expected_configuration:
        raise ZonalPriorError(
            f"{case_id}: sidecar configuration={meta.get('configuration')!r} 与当前配置 "
            f"{expected_configuration!r} 不一致"
        )
    if expected_plans_sha256 is not None and meta.get("plans_sha256") != expected_plans_sha256:
        raise ZonalPriorError(f"{case_id}: sidecar plans_sha256 与当前 nnUNetPlans.json 哈希不一致")
    input_hashes = meta.get("input_sha256")
    if not isinstance(input_hashes, Mapping) or not input_hashes:
        raise ZonalPriorError(f"{case_id}: sidecar input_sha256 必须为非空映射")
    if any(not is_sha256_hex(v) for v in input_hashes.values()):
        raise ZonalPriorError(f"{case_id}: sidecar input_sha256 含非法哈希值")
    if expected_input_sha256 is not None and dict(input_hashes) != dict(expected_input_sha256):
        raise ZonalPriorError(f"{case_id}: sidecar input_sha256 与期望不一致（输入文件已变化）")
    actual_meta_sha = metadata_sha256_of(meta)
    if expected_metadata_sha256 is not None and actual_meta_sha != expected_metadata_sha256:
        raise ZonalPriorError(
            f"{case_id}: sidecar metadata_sha256 与 manifest 不一致（元数据被修改或 manifest 伪造）"
        )
    output_sha256 = meta.get("output_sha256")
    if not is_sha256_hex(output_sha256):
        raise ZonalPriorError(f"{case_id}: sidecar output_sha256 必须为非空 64 位小写 hex")
    if expected_output_sha256 is not None and output_sha256 != expected_output_sha256:
        raise ZonalPriorError(f"{case_id}: sidecar output_sha256 与 manifest 不一致")
    require_oracle_pass(meta, case_id=case_id)

    prior: np.ndarray | None = None
    if read_array:
        stored = read_sidecar_prior(npz_path)
        # 先看**真实存储 dtype**（read_sidecar_prior 不做转换），通过后才 normalize
        if stored.dtype != np.float32:
            raise ZonalPriorError(
                f"{case_id}: sidecar 数组真实存储 dtype={stored.dtype} 必须为 float32"
                "（元数据声称 float32 不足以代表实际存储类型）"
            )
        prior = normalize_prior(stored)
        recorded = meta.get("output_shape")
        if not isinstance(recorded, Sequence) or list(recorded) != list(prior.shape):
            raise ZonalPriorError(
                f"{case_id}: sidecar output_shape={recorded} 与数组实际 {list(prior.shape)} 不一致"
            )
        if expected_array_shape is not None and tuple(int(v) for v in expected_array_shape) != tuple(prior.shape[-3:]):
            raise ZonalPriorError(
                f"{case_id}: prior 空间尺寸 {tuple(prior.shape[-3:])} != 期望 {tuple(int(v) for v in expected_array_shape)}"
            )
        if meta.get("output_dtype") != "float32":
            raise ZonalPriorError(f"{case_id}: sidecar output_dtype={meta.get('output_dtype')!r} != 'float32'")
        if list(meta.get("output_channel_semantics") or []) != list(ZONE_CHANNEL_NAMES):
            raise ZonalPriorError(
                f"{case_id}: sidecar channel_semantics={meta.get('output_channel_semantics')!r} "
                f"必须为 {list(ZONE_CHANNEL_NAMES)}"
            )
        if verify_array_hash and array_sha256(prior) != output_sha256:
            raise ZonalPriorError(f"{case_id}: sidecar 数组 SHA256 与元数据不一致（文件被篡改或损坏）")
        validate_prior_array(prior)
    elif verify_array_hash:
        raise ZonalPriorError(
            f"{case_id}: read_array=False 时无法校验数组哈希（不得声称已校验数组内容）"
        )
    return ValidatedSidecar(
        case_id=case_id,
        metadata=meta,
        prior=prior,
        metadata_sha256=actual_meta_sha,
        output_sha256=str(output_sha256),
        array_hash_checked=bool(read_array and verify_array_hash),
    )


def sidecar_identity_ok(
    out_root: str | Path,
    case_id: str,
    expected: Mapping[str, Any],
    *,
    verify_array_hash: bool = True,
) -> tuple[bool, str]:
    """`--resume` 的轻量跳过判据：**复用统一校验器**；任一不符返回 (False, 原因)。"""
    try:
        validate_sidecar(
            out_root,
            case_id,
            expected_source=expected.get("source"),
            expected_configuration=expected.get("configuration"),
            expected_plans_sha256=expected.get("plans_sha256"),
            expected_input_sha256=expected.get("input_sha256"),
            verify_array_hash=verify_array_hash,
        )
    except ZonalPriorError as exc:
        return False, str(exc)
    return True, "ok"


def write_prior_sidecar(
    out_root: str | Path,
    case_id: str,
    prior: np.ndarray,
    metadata: Mapping[str, Any],
    *,
    overwrite: bool = False,
    resume: bool = False,
) -> tuple[Path, Path, str, dict[str, Any]]:
    """写入 `<out_root>/<case_id>.npz` + `.json`；返回 `(npz, json, status, 最终落盘 metadata)`。

    公开路径**一律 fail-closed**：normalize → validate_prior_array → 身份字段校验 → oracle 严格通过
    → 成对提交 → 落盘复核。`resume=True` 时用统一严格校验器判断能否 `skipped`，
    弱比较（只看 source/plans/shape）已移除：损坏数组、错误 configuration、
    错误 channel semantics、oracle 失败都**不会**返回 skipped。
    """
    out_root = Path(out_root)
    npz_path, meta_path = sidecar_paths(out_root, case_id)
    arr = normalize_prior(prior)
    validate_prior_array(arr)
    meta = _require_writable_metadata(case_id, finalize_sidecar_metadata(case_id, arr, metadata))

    exists = npz_path.exists() or meta_path.exists()
    if exists and resume and not overwrite:
        try:
            validate_sidecar(
                out_root,
                case_id,
                expected_source=meta.get("source"),
                expected_configuration=meta.get("configuration"),
                expected_plans_sha256=meta.get("plans_sha256"),
                expected_input_sha256=meta.get("input_sha256"),
                verify_array_hash=True,
            )
        except ZonalPriorError:
            pass  # 既有 sidecar 不严格一致 → 重写该例（resume 允许）
        else:
            return npz_path, meta_path, "skipped", read_sidecar_metadata(meta_path)
    elif exists and not overwrite:
        raise ZonalPriorError(
            f"{case_id}: 输出已存在（{npz_path.name} / {meta_path.name}）且未允许覆盖；"
            "如需续跑请用 --resume，如需重算请显式用 --overwrite"
        )
    npz_path, meta_path, final_meta = commit_sidecar_pair(out_root, case_id, arr, meta)
    return npz_path, meta_path, "ok", final_meta


# --------------------------------------------------------------------------- 数据集级 manifest
def build_manifest(
    records: Sequence[Mapping[str, Any]],
    *,
    source: str,
    plans_sha256: str,
    configuration: str,
) -> dict[str, Any]:
    """构造数据集级 manifest；记录字段与哈希格式在此强制（空值/伪造格式直接报错）。"""
    cases = []
    seen: set[str] = set()
    duplicates: list[str] = []
    for record in records:
        case_id = str(record.get("case_id", ""))
        if not case_id:
            raise ZonalPriorError("manifest 记录缺少 case_id")
        if case_id in seen:
            duplicates.append(case_id)
        seen.add(case_id)
        input_hashes = dict(record.get("input_sha256") or {})
        if not input_hashes or any(not is_sha256_hex(v) for v in input_hashes.values()):
            raise ZonalPriorError(f"{case_id}: input_sha256 必须为非空 64 位小写 hex 映射")
        output_sha256 = str(record.get("output_sha256", ""))
        if not is_sha256_hex(output_sha256):
            raise ZonalPriorError(f"{case_id}: output_sha256 必须为非空 64 位小写 hex")
        metadata_sha256 = str(record.get("metadata_sha256", ""))
        if not is_sha256_hex(metadata_sha256):
            raise ZonalPriorError(f"{case_id}: metadata_sha256 必须为非空 64 位小写 hex")
        cases.append(
            {
                "case_id": case_id,
                "input_sha256": input_hashes,
                "output_sha256": output_sha256,
                "metadata_sha256": metadata_sha256,
            }
        )
    if duplicates:
        raise ZonalPriorError(f"manifest 记录存在重复 case_id: {sorted(set(duplicates))[:5]}")
    cases.sort(key=lambda item: item["case_id"])
    if not cases:
        raise ZonalPriorError("manifest 记录为空（拒绝构造空 manifest）")
    payload = {
        "source": source,
        "plans_sha256": plans_sha256,
        "configuration": configuration,
        "channel_semantics": list(ZONE_CHANNEL_NAMES),
        "n_cases": len(cases),
        "case_ids": [item["case_id"] for item in cases],
        "cases": cases,
    }
    payload["manifest_sha256"] = json_sha256({k: v for k, v in payload.items() if k != "manifest_sha256"})
    return payload


def verify_manifest(
    manifest: Mapping[str, Any],
    expected_case_ids: Sequence[str],
    *,
    source: str | None = None,
    plans_sha256: str | None = None,
    configuration: str | None = None,
) -> dict[str, Any]:
    """校验 manifest 完整性：字段、哈希格式、排序、唯一、ids 一致、覆盖范围、来源与自校验。"""
    if not isinstance(manifest, Mapping):
        raise ZonalPriorError("manifest 必须是 JSON 对象")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ZonalPriorError("manifest 缺少 'cases' 或为空")
    ids: list[str] = []
    for index, item in enumerate(cases):
        if not isinstance(item, Mapping):
            raise ZonalPriorError(f"manifest.cases[{index}] 必须是对象")
        case_id = item.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ZonalPriorError(f"manifest.cases[{index}].case_id 必须为非空字符串")
        ids.append(case_id)
        input_hashes = item.get("input_sha256")
        if not isinstance(input_hashes, Mapping) or not input_hashes:
            raise ZonalPriorError(f"{case_id}: input_sha256 必须为非空映射")
        if any(not is_sha256_hex(v) for v in input_hashes.values()):
            raise ZonalPriorError(f"{case_id}: input_sha256 含非法哈希值")
        if not is_sha256_hex(item.get("output_sha256")):
            raise ZonalPriorError(f"{case_id}: output_sha256 必须为非空 64 位小写 hex")
        if not is_sha256_hex(item.get("metadata_sha256")):
            raise ZonalPriorError(f"{case_id}: metadata_sha256 必须为非空 64 位小写 hex")
    if len(set(ids)) != len(ids):
        raise ZonalPriorError("manifest 存在重复 case_id")
    if ids != sorted(ids):
        raise ZonalPriorError("manifest.cases 必须按 case_id 升序排列")
    case_ids = manifest.get("case_ids")
    if not isinstance(case_ids, list) or list(case_ids) != ids:
        raise ZonalPriorError("manifest.case_ids 必须与 cases 的 case_id 顺序完全一致")
    if int(manifest.get("n_cases", -1)) != len(cases):
        raise ZonalPriorError("manifest.n_cases 与 cases 长度不一致")
    expected = {str(c) for c in expected_case_ids}
    missing = sorted(expected - set(ids))
    extra = sorted(set(ids) - expected)
    if missing or extra:
        raise ZonalPriorError(
            f"manifest 覆盖范围不等于期望病例集合：缺少 {missing[:5]}（共 {len(missing)}），"
            f"多余 {extra[:5]}（共 {len(extra)}）"
        )
    if source is not None and manifest.get("source") != source:
        raise ZonalPriorError(f"manifest.source={manifest.get('source')!r} != {source!r}")
    if plans_sha256 is not None and manifest.get("plans_sha256") != plans_sha256:
        raise ZonalPriorError("manifest.plans_sha256 与当前 plan 文件哈希不一致")
    if configuration is not None and manifest.get("configuration") != configuration:
        raise ZonalPriorError(f"manifest.configuration={manifest.get('configuration')!r} != {configuration!r}")
    if list(manifest.get("channel_semantics") or []) != list(ZONE_CHANNEL_NAMES):
        raise ZonalPriorError("manifest.channel_semantics 必须为 ['PZ','TZ']")
    recomputed = json_sha256({k: v for k, v in manifest.items() if k != "manifest_sha256"})
    if recomputed != manifest.get("manifest_sha256"):
        raise ZonalPriorError("manifest_sha256 自校验失败（文件被修改或损坏）")
    if not is_sha256_hex(manifest.get("manifest_sha256")):
        raise ZonalPriorError("manifest.manifest_sha256 必须为非空 64 位小写 hex")
    return dict(manifest)


def write_manifest(out_root: str | Path, manifest: Mapping[str, Any], *, overwrite: bool = True) -> Path:
    """原子发布 canonical manifest（调用方必须先完成覆盖范围与自校验）。"""
    path = Path(out_root) / MANIFEST_FILENAME
    if path.exists() and not overwrite:
        raise ZonalPriorError(f"manifest 已存在且未允许覆盖: {path}")
    return write_json_atomic(path, dict(manifest))


# --------------------------------------------------------------------------- 物化编排
@dataclass
class CaseOutcome:
    """单个病例的物化结果（用于失败清单与统计）。"""

    case_id: str
    status: str  # ok / skipped / failed
    reason: str = ""
    oracle: dict[str, Any] = field(default_factory=dict)
    npz_path: str = ""
    json_path: str = ""
    #: **最终落盘 metadata**（含 output_sha256；skipped 时为磁盘上的实测元数据）
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest_record(self) -> dict[str, Any]:
        """构造 manifest 记录（metadata_sha256 由**最终落盘 metadata** 计算）。"""
        if not self.metadata:
            raise ZonalPriorError(f"{self.case_id}: 缺少最终落盘 metadata，无法构造 manifest 记录")
        return {
            "case_id": self.case_id,
            "input_sha256": dict(self.metadata.get("input_sha256") or {}),
            "output_sha256": str(self.metadata.get("output_sha256", "")),
            "metadata_sha256": metadata_sha256_of(self.metadata),
        }


class ZonalPriorMaterializer:
    """逐病例物化 PZ/TZ sidecar 的编排（几何/nnU-Net 交互由 `convert` 注入，便于合成测试）。

    `convert(case_id) -> (prior[2,D,H,W], metadata)` 必须读取原始分区标签、转换到 plan 空间、
    执行 lesion oracle 并返回可直接写入的数组与元数据。**oracle 由本类二次强制**（fail-closed）。

    canonical manifest 的发布规则（严格）：
    - 仅当 `expected_case_ids` 给出且本轮 `case_ids` **完整覆盖**它、`n_failed == 0` 时发布；
    - 发布前用 `verify_manifest` 校验 cases/case_ids/n_cases/排序/唯一/覆盖范围；
    - 子集运行、dry-run、任何失败都不创建也不覆盖既有 manifest；
    - 全量 resume 时从 `out_root` 中**全部已验证的实际 sidecar** 重建 manifest。
    """

    def __init__(
        self,
        out_root: str | Path,
        *,
        source: str,
        configuration: str,
        plans_sha256: str,
        resume: bool = False,
        overwrite: bool = False,
    ) -> None:
        if source not in ZONAL_SOURCES:
            raise ZonalPriorError(f"source={source!r} 非法；必须是 {list(ZONAL_SOURCES)}（不得静默选择）")
        if resume and overwrite:
            raise ZonalPriorError("--resume 与 --overwrite 互斥：请明确是续跑还是重算")
        self.out_root = Path(out_root)
        self.source = source
        self.configuration = configuration
        self.plans_sha256 = plans_sha256
        self.resume = bool(resume)
        self.overwrite = bool(overwrite)

    def expected_identity(self, input_sha256: Mapping[str, Any]) -> dict[str, Any]:
        """构造 resume 期望身份（供调用方与 sidecar_identity_ok 对齐）。"""
        return {
            "source": self.source,
            "configuration": self.configuration,
            "plans_sha256": self.plans_sha256,
            "input_sha256": dict(input_sha256),
        }

    # ------------------------------------------------------------------ 内部
    def _skipped_outcome(self, case_id: str) -> CaseOutcome:
        """从磁盘读取实际 sidecar 元数据构造 skipped outcome（不使用内存对象）。"""
        meta = read_sidecar_metadata(sidecar_paths(self.out_root, case_id)[1])
        return CaseOutcome(
            case_id=case_id,
            status="skipped",
            oracle=dict(meta.get("oracle") or {}),
            npz_path=str(sidecar_paths(self.out_root, case_id)[0]),
            json_path=str(sidecar_paths(self.out_root, case_id)[1]),
            metadata=meta,
        )

    def collect_verified_records(
        self,
        case_ids: Sequence[str],
        *,
        identity_probe: Callable[[str], Mapping[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """从磁盘重建 manifest 记录：逐个 sidecar 做完整性/身份校验后才纳入。"""
        records: list[dict[str, Any]] = []
        for case_id in case_ids:
            case_id = str(case_id)
            meta = read_sidecar_metadata(sidecar_paths(self.out_root, case_id)[1])
            expected: dict[str, Any]
            if identity_probe is not None:
                expected = dict(identity_probe(case_id))
                for key, value in (
                    ("source", self.source),
                    ("configuration", self.configuration),
                    ("plans_sha256", self.plans_sha256),
                ):
                    if expected.get(key) != value:
                        raise ZonalPriorError(
                            f"{case_id}: identity_probe.{key}={expected.get(key)!r} 与当前运行 {value!r} 不一致"
                        )
            else:
                expected = self.expected_identity(meta.get("input_sha256") or {})
            ok, why = sidecar_identity_ok(self.out_root, case_id, expected)
            if not ok:
                raise ZonalPriorError(f"{case_id}: sidecar 未通过完整性校验（{why}）")
            outcome = CaseOutcome(case_id=case_id, status="ok", metadata=meta)
            records.append(outcome.manifest_record())
        return records

    def run(
        self,
        case_ids: Sequence[str],
        convert: Callable[[str], tuple[np.ndarray, Mapping[str, Any]]],
        *,
        identity_probe: Callable[[str], Mapping[str, Any]] | None = None,
        expected_case_ids: Sequence[str] | None = None,
        dry_run: bool = False,
        progress: bool = True,
        desc: str = "zonal prior",
    ) -> dict[str, Any]:
        """执行物化（或 dry-run）；仅在满足严格条件时发布 canonical manifest。"""
        import time

        requested = [str(c) for c in case_ids]
        expected_set = None if expected_case_ids is None else {str(c) for c in expected_case_ids}
        is_full_run = expected_set is not None and set(requested) == expected_set
        outcomes: list[CaseOutcome] = []
        iterator: Sequence[str] = list(requested)
        if progress:  # dry-run 同样逐例显示进度（转换 + oracle 仍然逐例执行）
            try:
                from tqdm import tqdm

                iterator = tqdm(iterator, desc=desc, mininterval=1.0)  # type: ignore[assignment]
            except ImportError:  # pragma: no cover
                iterator = list(requested)
        t0 = time.time()
        for case_id in iterator:
            case_id = str(case_id)
            try:
                if self.resume and not self.overwrite and identity_probe is not None:
                    try:
                        expected = dict(identity_probe(case_id))
                    except Exception:  # noqa: BLE001 - probe 只用于「能否跳过」；失败即走完整转换
                        expected = None
                    for key, value in (
                        ("source", self.source),
                        ("configuration", self.configuration),
                        ("plans_sha256", self.plans_sha256),
                    ):
                        if expected is not None and expected.get(key) != value:
                            raise ZonalPriorError(
                                f"{case_id}: identity_probe.{key}={expected.get(key)!r} 与当前运行 "
                                f"{value!r} 不一致（拒绝跳过，必须重新物化）"
                            )
                    if expected is not None:
                        ok, _why = sidecar_identity_ok(self.out_root, case_id, expected)
                        if ok:
                            outcomes.append(self._skipped_outcome(case_id))
                            continue
                prior, metadata = convert(case_id)
                meta = dict(metadata)
                meta.setdefault("source", self.source)
                meta.setdefault("configuration", self.configuration)
                meta.setdefault("plans_sha256", self.plans_sha256)
                oracle = require_oracle_pass(meta, case_id=case_id)
                meta["oracle"] = oracle
                p = normalize_prior(prior)
                validate_prior_array(p)
                if dry_run:
                    outcomes.append(CaseOutcome(case_id=case_id, status="ok", oracle=oracle, metadata=meta))
                    continue
                if self.resume and not self.overwrite:
                    ok, _why = sidecar_identity_ok(
                        self.out_root, case_id, self.expected_identity(meta.get("input_sha256") or {})
                    )
                    if ok:
                        outcomes.append(self._skipped_outcome(case_id))
                        continue
                npz_path, meta_path, status, final_meta = write_prior_sidecar(
                    self.out_root,
                    case_id,
                    p,
                    meta,
                    overwrite=self.overwrite or self.resume,
                    resume=False,
                )
                if status == "skipped":
                    outcomes.append(self._skipped_outcome(case_id))
                    continue
                outcomes.append(
                    CaseOutcome(
                        case_id=case_id,
                        status="ok",
                        oracle=oracle,
                        npz_path=str(npz_path),
                        json_path=str(meta_path),
                        metadata=final_meta,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - 逐例失败必须记录并继续
                outcomes.append(CaseOutcome(case_id=case_id, status="failed", reason=f"{type(exc).__name__}: {exc}"))
        elapsed = round(time.time() - t0, 2)

        n_ok = sum(1 for o in outcomes if o.status == "ok")
        n_skipped = sum(1 for o in outcomes if o.status == "skipped")
        n_failed = sum(1 for o in outcomes if o.status == "failed")
        summary: dict[str, Any] = {
            "source": self.source,
            "configuration": self.configuration,
            "plans_sha256": self.plans_sha256,
            "out_root": str(self.out_root),
            "dry_run": bool(dry_run),
            "full_run": bool(is_full_run),
            "n_total": len(outcomes),
            "n_ok": n_ok,
            "n_skipped": n_skipped,
            "n_failed": n_failed,
            "elapsed_sec": elapsed,
            "failed": [{"case_id": o.case_id, "reason": o.reason} for o in outcomes if o.status == "failed"],
            "oracle_mismatch": [
                {"case_id": o.case_id, "oracle": o.oracle}
                for o in outcomes
                if o.oracle and not o.oracle.get("exact_match", True)
            ],
            "manifest_published": False,
            "manifest_not_published_reason": "",
        }

        # ---- canonical manifest 发布（严格门控；发布前校验，失败不触碰既有 manifest）
        if dry_run:
            summary["manifest_not_published_reason"] = "dry_run"
        elif not is_full_run:
            summary["manifest_not_published_reason"] = (
                "subset_run"
                if expected_set is not None
                else "expected_case_ids_not_provided"
            )
        elif n_failed > 0:
            summary["manifest_not_published_reason"] = f"n_failed={n_failed}"
        elif n_ok + n_skipped != len(expected_set or ()):
            summary["manifest_not_published_reason"] = (
                f"incomplete_coverage={n_ok + n_skipped}/{len(expected_set or ())}"
            )
        else:
            try:
                records = self.collect_verified_records(sorted(expected_set), identity_probe=identity_probe)
                manifest = build_manifest(
                    records,
                    source=self.source,
                    plans_sha256=self.plans_sha256,
                    configuration=self.configuration,
                )
                verify_manifest(
                    manifest,
                    sorted(expected_set),
                    source=self.source,
                    plans_sha256=self.plans_sha256,
                    configuration=self.configuration,
                )
                manifest_path = write_manifest(self.out_root, manifest)
                summary["manifest_published"] = True
                summary["manifest_path"] = str(manifest_path)
                summary["manifest_sha256"] = manifest["manifest_sha256"]
                summary["manifest_n_cases"] = manifest["n_cases"]
            except Exception as exc:  # noqa: BLE001
                summary["manifest_not_published_reason"] = f"{type(exc).__name__}: {exc}"
        return summary

    def write_summary(self, summary: Mapping[str, Any]) -> Path:
        """原子写入 `<out_root>/materialization_summary.json`（不覆盖历史：加时间戳）。"""
        import time

        path = self.out_root / SUMMARY_FILENAME
        if path.exists():
            stamp = time.strftime("%Y%m%d_%H%M%S")
            path = self.out_root / f"materialization_summary_{stamp}.json"
        return write_json_atomic(path, dict(summary))
