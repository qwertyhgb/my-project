"""`--overfit` 诊断模式的**运行时 GPU 峰值显存遥测**（只读 `torch.cuda` API；不改训练语义）。

测量语义（写进报告的 `measurement_scope`，避免把整卡占用误算成本进程分配）：
- `process_memory.*`：来自 `torch.cuda.memory_allocated / memory_reserved / max_memory_*`，
  是**当前 PyTorch 进程**的分配器统计（peak = 本进程 high-water mark）；
- `gpu.total_bytes / gpu.free_bytes_*`：来自 `torch.cuda.mem_get_info`，是**整张 GPU** 的
  可用状态（含 nvidia-smi 可见的外部进程占用）；因此 free 少并不等于本进程占用多；
- 不使用任何 `nvidia-smi` 子进程作为生产数据来源；
- `models/profiling.py` 的静态报告仍标记 `peak_gpu_memory=NOT_MEASURED`（未被本模块改写），
  运行时实测是本模块产出的**独立产物**。

生命周期（`reset_peak_memory_stats` 必须早于 `build_trainer` / `model.to(device)`）：
    start()                          # 规范化 device → reset peaks → 记录 start 采样
    mark_phase("build_trainer")      # 进入模型构建
    snapshot("after_build")          # 构建完成后（模型已 to(device)）
    mark_phase("trainer_run")        # trainer.train() 内部会依次跑 train 与 validation
    trainer.train()
    snapshot("after_trainer_run")    # 覆盖 build+train+validation 的整体峰值
    finalize(STATUS_COMPLETED)       # 读 end 采样 → 原子写 JSON

诚实边界（不得夸大）：
- `trainer.train()` 是不可分割的一段（内部先 `_run_epoch(train=True)` 再 `_run_epoch(train=False)`），
  因此**没有独立的 train / validation 峰值**；`peak_*` 是整个 `trainer_run` 的 high-water mark；
- 异常时才用 `trainer.current_phase`（"train" / "validation"）细分**失败阶段**；缺失/未知记 `trainer_run`；
- 失败路径同样写盘：`finalize(STATUS_CUDA_OOM / STATUS_ERROR, ...)` 先取峰值再写；
- end 采样失败不丢报告：记录 `final_sample_error` 并用已有样本落盘；
- 只有在峰值统计落盘之后才允许 `release_cache()`（empty_cache）。
"""
from __future__ import annotations

import contextlib
import json
import os
import tempfile
import time
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

#: 报告文件名（固定落在 `outputs/diagnostics/<model_id>/<run_name>/diagnostics/`）
GPU_PROFILE_FILENAME = "gpu_memory_profile.json"

STATUS_COMPLETED = "COMPLETED"
STATUS_CUDA_OOM = "CUDA_OOM"
STATUS_ERROR = "ERROR"
STATUS_NOT_APPLICABLE = "NOT_APPLICABLE"
STATUSES = (STATUS_COMPLETED, STATUS_CUDA_OOM, STATUS_ERROR, STATUS_NOT_APPLICABLE)

PHASE_INITIALIZE = "initialize_cuda"
PHASE_BUILD = "build_trainer"
#: `trainer.train()` 整体（内部先 train 再 validation，二者不可分割）
PHASE_TRAINER_RUN = "trainer_run"
#: 仅用于**异常时**按 `trainer.current_phase` 细分失败阶段（不存在独立的 train/validation 峰值）
PHASE_TRAIN = "train"
PHASE_VALIDATION = "validation"
PHASE_FINALIZE = "finalize"
PHASES = (
    PHASE_INITIALIZE,
    PHASE_BUILD,
    PHASE_TRAINER_RUN,
    PHASE_TRAIN,
    PHASE_VALIDATION,
    PHASE_FINALIZE,
)

#: CPU 路径的固定 reason（`status=NOT_APPLICABLE`）
NOT_APPLICABLE_REASON_CPU = "device is not CUDA"

MEASUREMENT_SCOPE = (
    "process_memory.* = 当前 PyTorch 进程（torch.cuda.memory_allocated/memory_reserved/"
    "max_memory_allocated/max_memory_reserved）；peak = current PyTorch process high-water mark，"
    "不是整卡占用、也不是 nvidia-smi 读数（not nvidia-smi, not whole-GPU occupancy）。"
    "gpu.total_bytes/free_bytes_* = torch.cuda.mem_get_info，为整张 GPU 状态（含外部进程占用）。"
)

class GpuMemoryProfileError(RuntimeError):
    """显存报告无法落盘（调用方仍保留原始异常）。"""


def bytes_to_mib(value: float | None) -> float | None:
    """字节 → MiB（1 MiB = 1024² B）；None 透传（CPU 路径不伪造数字）。"""
    return None if value is None else round(float(value) / (1024**2), 3)


def bytes_to_gib(value: float | None) -> float | None:
    """字节 → GiB（1 GiB = 1024³ B）；None 透传。"""
    return None if value is None else round(float(value) / (1024**3), 3)


def is_cuda_oom(exc: BaseException) -> bool:
    """识别 CUDA OOM：torch 的 `OutOfMemoryError`、同名异常类，或典型 OOM 消息。"""
    oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_type is not None and isinstance(exc, oom_type):
        return True
    if type(exc).__name__ == "OutOfMemoryError":
        return True
    message = str(exc).lower()
    return "out of memory" in message and "cuda" in message


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_write_json(path: str | Path, doc: Mapping[str, Any]) -> Path:
    """临时文件 + fsync + `os.replace` 原子发布（中断不会留下半截报告）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(dict(doc), indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    with tempfile.NamedTemporaryFile(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        tmp = Path(handle.name)
        try:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                tmp.unlink()
            raise
    try:
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()
        raise
    return path


def _with_derived(sample: Mapping[str, Any]) -> dict[str, Any]:
    """为**每个** `*_bytes` 字段补 MiB/GiB 派生值。

    同时支持带后缀的键（`allocated_bytes_at_start` → `allocated_mib_at_start` /
    `allocated_gib_at_start`）与朴素键（`total_bytes` → `total_mib` / `total_gib`），
    避免派生字段名不匹配而变成 null（那会让「MiB/GiB 派生值」形同虚设）。
    """
    marker = "_bytes"
    out = dict(sample)
    for key, value in sample.items():
        if not isinstance(key, str) or marker not in key:
            continue
        head, _, tail = key.partition(marker)  # head="allocated", tail="_at_start" / ""
        out[f"{head}_mib{tail}"] = bytes_to_mib(value)
        out[f"{head}_gib{tail}"] = bytes_to_gib(value)
    return out


class GpuMemoryProfiler:
    """当前 PyTorch 进程的 GPU 显存遥测（CUDA 路径读 API；CPU 路径完全不触碰 CUDA）。"""

    def __init__(
        self,
        device: str,
        *,
        output_path: str | Path,
        metadata: Mapping[str, Any] | None = None,
        cuda_api: Any = None,
    ) -> None:
        self.device_argument = str(device)
        self.output_path = Path(output_path)
        self.metadata: dict[str, Any] = dict(metadata or {})
        self._cuda_api = cuda_api  # 注入点（合成测试用）；None → 惰性解析 torch.cuda
        self.status: str | None = None
        self.phase: str = PHASE_INITIALIZE
        self.reason = ""
        self.cuda_device: str | None = None
        self.device_name: str | None = None
        self.compute_capability: str | None = None
        self.started_at: str | None = None
        self.completed_at: str | None = None
        self._t0: float | None = None
        self._error = ""
        self._exception_type = ""
        self._final_sample_error = ""
        self._cuda_active = False
        self._samples: dict[str, dict[str, Any]] = {}
        self._phases: dict[str, dict[str, Any]] = {}
        self._empty_cache_calls = 0

    # ------------------------------------------------------------------ 内部
    def _api(self):
        return self._cuda_api if self._cuda_api is not None else torch.cuda

    def _is_cuda_requested(self) -> bool:
        return self.device_argument.startswith("cuda")

    def _device_index(self) -> int:
        if ":" not in self.device_argument:
            return 0
        try:
            return int(self.device_argument.split(":", 1)[1])
        except ValueError as exc:  # pragma: no cover - 非法 device 字符串
            raise GpuMemoryProfileError(f"无法解析 device={self.device_argument!r}") from exc

    def _sample(self) -> dict[str, Any]:
        api = self._api()
        device = self.cuda_device
        free, total = api.mem_get_info(device)
        raw = {
            "allocated_bytes": int(api.memory_allocated(device)),
            "reserved_bytes": int(api.memory_reserved(device)),
            "peak_allocated_bytes": int(api.max_memory_allocated(device)),
            "peak_reserved_bytes": int(api.max_memory_reserved(device)),
            "free_bytes": int(free),
            "total_bytes": int(total),
        }
        return _with_derived(raw)

    # ------------------------------------------------------------------ 生命周期
    def start(self) -> None:
        """初始化目标 CUDA device 并复位峰值统计（必须在模型构建之前调用）。"""
        self.started_at = _now_iso()
        self._t0 = time.time()
        self.phase = PHASE_INITIALIZE
        if not self._is_cuda_requested():
            self.status = STATUS_NOT_APPLICABLE
            self.reason = NOT_APPLICABLE_REASON_CPU
            return
        api = self._api()
        if not bool(api.is_available()):  # pragma: no cover - 依赖真实环境
            self.status = STATUS_ERROR
            self.reason = "请求了 CUDA device，但 torch.cuda.is_available() 为 False"
            self._error = self.reason
            self._exception_type = "CudaUnavailable"
            return
        self.cuda_device = f"cuda:{self._device_index()}"
        self.device_name = str(api.get_device_name(self.cuda_device))
        props = api.get_device_properties(self.cuda_device)
        self.compute_capability = f"{int(props.major)}.{int(props.minor)}"
        # 关键顺序：先 reset 峰值，再做任何 model.to(device)/构建，否则模型驻留会被漏掉
        api.reset_peak_memory_stats(self.cuda_device)
        self._cuda_active = True
        self._samples["start"] = self._sample()
        self._phases[PHASE_INITIALIZE] = self._samples["start"]

    def mark_phase(self, phase: str) -> None:
        """记录当前阶段；CUDA 时在该阶段入口采样一次（首次进入时的驻留/峰值）。"""
        self.phase = phase
        if not self._cuda_active:
            return
        sample = self._sample()
        self._phases.setdefault(phase, sample)

    def snapshot(self, name: str) -> None:
        """记录命名采样点（如 after_build / after_train / after_validation）。"""
        if not self._cuda_active:
            return
        self._samples[name] = self._sample()

    def update(self, **fields: Any) -> None:
        """合并元数据（模型身份、参数量、AMP 实际开关、manifest 哈希等）。"""
        for key, value in fields.items():
            if isinstance(value, Mapping) and isinstance(self.metadata.get(key), Mapping):
                merged = dict(self.metadata[key])  # type: ignore[arg-type]
                merged.update(value)  # type: ignore[arg-type]
                self.metadata[key] = merged
            else:
                self.metadata[key] = value

    def release_cache(self) -> None:
        """**仅在峰值统计已落盘后**调用；CPU/未激活路径为 no-op。"""
        if not self._cuda_active:
            return
        self._empty_cache_calls += 1
        with contextlib.suppress(Exception):  # pragma: no cover - 释放缓存失败不影响主流程
            self._api().empty_cache()

    # ------------------------------------------------------------------ 报告
    def elapsed_sec(self) -> float:
        return 0.0 if self._t0 is None else round(time.time() - self._t0, 3)

    def to_report(self, *, status: str | None = None) -> dict[str, Any]:
        """构造结构化报告（CPU 路径的 CUDA 字段一律 None，不伪造数字）。"""
        effective_status = status or self.status or STATUS_ERROR
        start = self._samples.get("start")
        build = self._samples.get("after_build")
        end = self._samples.get("end")
        model_meta = dict(self.metadata.get("model") or {})
        run_meta = dict(self.metadata.get("run") or {})
        gpu = None
        if start is not None:
            free_start = start.get("free_bytes")
            free_end = None if end is None else end.get("free_bytes")
            total = start.get("total_bytes")
            gpu = {
                "total_bytes": total,
                "total_mib": bytes_to_mib(total),
                "total_gib": bytes_to_gib(total),
                "free_bytes_at_start": free_start,
                "free_mib_at_start": bytes_to_mib(free_start),
                "free_gib_at_start": bytes_to_gib(free_start),
                "free_bytes_at_end": free_end,
                "free_mib_at_end": bytes_to_mib(free_end),
                "free_gib_at_end": bytes_to_gib(free_end),
                "free_bytes_delta_start_to_end": (
                    None if free_start is None or free_end is None else int(free_end) - int(free_start)
                ),
            }
        process_memory = None
        if start is not None or end is not None:
            def _field(sample: Mapping[str, Any] | None, key: str) -> Any:
                return None if sample is None else sample.get(key)

            # peak 是单调 high-water mark：end 采样失败时回退到最新的可用样本（after_build / start），
            # 保证报告不因第二次采样失败而丢失峰值信息；出处写入 peak_sample_source。
            if end is not None:
                peak_sample, peak_source = end, "end"
            elif build is not None:
                peak_sample, peak_source = build, "after_build"
            else:
                peak_sample, peak_source = start, "start"
            process_memory = _with_derived(
                {
                    "allocated_bytes_at_start": _field(start, "allocated_bytes"),
                    "reserved_bytes_at_start": _field(start, "reserved_bytes"),
                    "allocated_bytes_after_build": _field(build, "allocated_bytes"),
                    "reserved_bytes_after_build": _field(build, "reserved_bytes"),
                    "allocated_bytes_at_end": _field(end, "allocated_bytes"),
                    "reserved_bytes_at_end": _field(end, "reserved_bytes"),
                    "peak_allocated_bytes": _field(peak_sample, "peak_allocated_bytes"),
                    "peak_reserved_bytes": _field(peak_sample, "peak_reserved_bytes"),
                }
            )
            process_memory["peak_sample_source"] = peak_source
        return {
            "status": effective_status,
            "phase": self.phase,
            "reason": self.reason or None,
            "device_argument": self.device_argument,
            "cuda_device": self.cuda_device,
            "device_name": self.device_name,
            "compute_capability": self.compute_capability,
            "amp_enabled": self.metadata.get("amp_enabled"),
            "measurement_scope": MEASUREMENT_SCOPE,
            "peak_is_process_peak_not_nvidia_smi": True,
            "cuda_api_used": bool(self._cuda_active),
            "gpu": gpu,
            "process_memory": process_memory,
            "phases": {phase: dict(sample) for phase, sample in self._phases.items()},
            "samples": {name: dict(sample) for name, sample in self._samples.items()},
            "model": model_meta,
            "run": run_meta,
            "zonal_prior_manifest_sha256": self.metadata.get("zonal_prior_manifest_sha256"),
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "elapsed_sec": self.elapsed_sec(),
            "exception_type": self._exception_type or None,
            "error": self._error or None,
            "final_sample_error": self._final_sample_error or None,
            "peak_scope": (
                "整个 trainer_run（build_trainer + train + validation）的当前进程 high-water mark；"
                "不存在独立的 train/validation 峰值"
            ),
        }

    def finalize(
        self,
        status: str,
        *,
        error: str = "",
        exception_type: str = "",
        reason: str = "",
    ) -> Path:
        """读取最终采样（**早于任何 empty_cache**）并原子写盘。失败也走本方法。"""
        if status not in STATUSES:
            raise GpuMemoryProfileError(f"未知 status={status!r}；允许 {list(STATUSES)}")
        self.status = status
        if reason:
            self.reason = reason
        if error:
            self._error = error
        if exception_type:
            self._exception_type = exception_type
        if status in (STATUS_COMPLETED, STATUS_NOT_APPLICABLE):
            self.phase = PHASE_FINALIZE  # CPU 的 NOT_APPLICABLE 同样归到 finalize
        if self._cuda_active:
            # end 采样失败不丢报告：保留已有 start/after_build/峰值样本，并记录原因
            try:
                self._samples["end"] = self._sample()
            except Exception as exc:  # noqa: BLE001 - 任何采样异常都不应导致报告丢失
                self._final_sample_error = f"{type(exc).__name__}: {exc}"
        self.completed_at = _now_iso()
        try:
            return atomic_write_json(self.output_path, self.to_report())
        except OSError as exc:
            raise GpuMemoryProfileError(
                f"GPU 显存报告无法落盘: {self.output_path}: {type(exc).__name__}: {exc}"
            ) from exc
