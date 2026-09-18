"""`--overfit` 运行时 GPU 峰值显存遥测的合成测试（mock CUDA 与 trainer；不初始化 GPU、不读真实病例）。

覆盖：reset 早于 build_trainer、after_build/peak 采集、trainer_run 真实分界（不伪造 train/validation
分界）、按 trainer.current_phase 细分失败阶段（train / validation）、build 阶段 OOM、initialize_cuda
阶段（reset / mem_get_info / CUDA 不可用）异常仍落盘、end 采样失败不丢报告、普通异常、
CPU 路径不触碰 CUDA API、JSON 原子写、MiB/GiB 换算、既有 overfit 参数与输出目录不回归。
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from torch import nn

import zonal_reliability_fusion.training.gpu_memory_profile as gmp
from zonal_reliability_fusion.config import (
    load_architecture_config,
    load_experiment_config,
    load_plan,
)
from zonal_reliability_fusion.models.profiling import structure_summary

REAL_ROOT = Path("/opt/data/private/lm/my-projects")
M0_CONFIG = REAL_ROOT / "configs/experiments/m0_resenc_picai_3d_fullres_v23.yaml"
M0_ARCH = REAL_ROOT / "configs/architectures/m0_resenc_v23.yaml"
PLANS = REAL_ROOT / "workdir/nnUNet_preprocessed/Dataset605_PICAI/nnUNetPlans.json"
GIB = 1024**3


# =========================================================================== 假 CUDA / 假 trainer
class FakeCudaApi:
    """记录所有 CUDA API 调用；`allocate()` 模拟本进程分配。"""

    def __init__(
        self,
        *,
        total: int = 24 * GIB,
        free: int = 17_800_000_000,
        events: list | None = None,
        available: bool = True,
        reset_error: Exception | None = None,
        mem_get_info_error_after: int | None = None,
    ):
        self.available = available
        self.reset_error = reset_error
        self.mem_get_info_error_after = mem_get_info_error_after
        self._mem_get_info_calls = 0
        self.calls: list = []
        self.events = events if events is not None else []
        self.total = total
        self.free = free
        self.allocated = 0
        self.reserved = 0
        self.peak_allocated = 0
        self.peak_reserved = 0
        self.json_existed_at_empty_cache: bool | None = None
        self.json_path: Path | None = None

    # --- CUDA API
    def is_available(self):
        self.calls.append("is_available")
        return self.available

    def get_device_name(self, device):
        self.calls.append(("get_device_name", device))
        return "Fake RTX 3090"

    def get_device_properties(self, device):
        self.calls.append(("get_device_properties", device))
        return SimpleNamespace(major=8, minor=6)

    def reset_peak_memory_stats(self, device):
        self.calls.append(("reset_peak_memory_stats", device))
        if self.reset_error is not None:
            raise self.reset_error
        self.events.append("reset_peak_memory_stats")
        self.peak_allocated = self.allocated
        self.peak_reserved = self.reserved

    def mem_get_info(self, device):
        self.calls.append(("mem_get_info", device))
        self._mem_get_info_calls += 1
        if (
            self.mem_get_info_error_after is not None
            and self._mem_get_info_calls > self.mem_get_info_error_after
        ):
            raise RuntimeError(f"simulated mem_get_info failure #{self._mem_get_info_calls}")
        return (self.free, self.total)

    def memory_allocated(self, device):
        return self.allocated

    def memory_reserved(self, device):
        return self.reserved

    def max_memory_allocated(self, device):
        return self.peak_allocated

    def max_memory_reserved(self, device):
        return self.peak_reserved

    def empty_cache(self):
        self.calls.append("empty_cache")
        if self.json_path is not None:
            self.json_existed_at_empty_cache = self.json_path.exists()
        self.reserved = 0

    # --- 模拟本进程显存分配
    def allocate(self, nbytes: int) -> None:
        self.allocated += nbytes
        self.reserved += nbytes
        self.peak_allocated = max(self.peak_allocated, self.allocated)
        self.peak_reserved = max(self.peak_reserved, self.reserved)
        self.free = max(0, self.free - nbytes)

    def release(self, nbytes: int) -> None:
        self.allocated = max(0, self.allocated - nbytes)


class ExplodingCudaApi:
    """任何属性访问都失败：用于证明 CPU 路径完全不触碰 CUDA API。"""

    def __getattr__(self, name):  # pragma: no cover - 触发即测试失败
        raise AssertionError(f"CPU 路径不得访问 torch.cuda.{name}")


class _FakeCudaOOM(RuntimeError):
    """与 torch.cuda.OutOfMemoryError 同类名/同消息形态的合成 OOM。"""


def _fake_oom() -> RuntimeError:
    exc = _FakeCudaOOM("CUDA out of memory. Tried to allocate 2.00 GiB (GPU 0; 24.00 GiB total)")
    return exc


class _FakeModel:
    MODEL_ID = "M0"

    def num_parameters(self) -> int:
        return 148_193_644

    def architecture_identity(self) -> dict:
        return {"architecture_name": "m0_resenc", "architecture_version": "v2.3", "architecture_sha256": "f" * 64}


class _FakeTrainer:
    """模拟真实 M0Trainer：train() 内部先 _run_epoch(train=True) 再 _run_epoch(train=False)，

    并且**真实存在** `current_phase`（trainer.py:250/490/689 设置 "train" / "validation"）。
    """

    def __init__(self, *, amp: bool = True, on_train=None, phases=("train", "validation")):
        self.amp = amp
        self._on_train = on_train
        self._phases = tuple(phases)
        self.current_phase: str | None = None

    def train(self):
        for phase in self._phases:
            self.current_phase = phase  # 与真实 trainer 一致：进入每个 epoch 阶段时设置
            if self._on_train is not None:
                self._on_train(phase)
        return SimpleNamespace(status="completed", checkpoint_metric="val_loss", best_metric=0.42)


class _FakeStore:
    available_ids = ("case_a", "case_b")
    train_ids = ("case_a", "case_b")
    val_ids = ()
    zonal_prior_manifest_sha256 = None


def _load_train_module():
    spec = importlib.util.spec_from_file_location("train_m0_gpu_mod", REAL_ROOT / "scripts/train/train_m0.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _args(**overrides):
    base = {
        "cases": "case_a,case_b",
        "device": "cuda",
        "run_name": "unit_gpu_profile",
        "iterations": 1,
        "val_iterations": 1,
        "epochs": 1,
        "no_plot": True,
        "no_progress": True,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _flow(tmp_path, monkeypatch, *, api, build=None, trainer=None, args=None, events=None, cfg=None):
    """运行真实的 run_overfit（build_store/build_trainer 被 mock），返回 (report_path, events, kwargs)。"""
    mod = _load_train_module()
    monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(gmp.torch, "cuda", api)
    captured: dict = {"events": events if events is not None else [], "kwargs": None}
    monkeypatch.setattr(mod, "build_store", lambda cfg_: _FakeStore())

    def _default_build(cfg_, plan_, **kwargs):
        captured["events"].append("build_trainer")
        captured["kwargs"] = kwargs
        api.allocate(2 * GIB)  # model.to(device) + 优化器驻留
        return trainer if trainer is not None else _FakeTrainer(), _FakeModel(), kwargs["store"]

    monkeypatch.setattr(mod, "build_trainer", build or _default_build)
    cfg_obj = cfg if cfg is not None else load_experiment_config(M0_CONFIG)
    run_args = args if args is not None else _args()
    mod.run_overfit(run_args, cfg_obj, object())
    base = tmp_path / "outputs/diagnostics" / str(cfg_obj.experiment.model_id).lower() / run_args.run_name
    report_path = base / "diagnostics" / gmp.GPU_PROFILE_FILENAME
    if api is not None and isinstance(api, FakeCudaApi):
        api.json_path = report_path
        if trainer is not None:
            pass
    return report_path, captured["events"], captured["kwargs"]


# =========================================================================== 1) CUDA 成功路径
def test_cuda_success_reset_before_build_and_peak_captured(tmp_path, monkeypatch):
    events: list = []
    api = FakeCudaApi(events=events)
    # 模拟 forward/backward/optimizer.step(Adam 状态首配) + 诊断验证 forward
    trainer = _FakeTrainer(
        on_train=lambda phase: api.allocate(3 * GIB if phase == "train" else GIB)
    )
    seen: dict = {}

    def _build(cfg_, plan_, **kwargs):
        seen["kwargs"] = kwargs
        events.append("build_trainer")  # reset 必须已经发生
        api.allocate(2 * GIB)
        return trainer, _FakeModel(), kwargs["store"]

    report_path, _, _ = _flow(tmp_path, monkeypatch, api=api, build=_build, events=events)

    assert events.index("reset_peak_memory_stats") < events.index("build_trainer"), (
        f"reset_peak_memory_stats 必须早于 build_trainer：{events}"
    )
    doc = json.loads(report_path.read_text())
    assert doc["status"] == "COMPLETED"
    assert doc["phase"] == gmp.PHASE_FINALIZE
    assert doc["cuda_device"] == "cuda:0" and doc["device_name"] == "Fake RTX 3090"
    assert doc["compute_capability"] == "8.6"
    assert doc["amp_enabled"] is True
    assert doc["cuda_api_used"] is True
    assert doc["peak_is_process_peak_not_nvidia_smi"] is True
    scope = doc["measurement_scope"]
    assert "current PyTorch process" in scope and "not nvidia-smi" in scope
    assert "当前 PyTorch 进程" in scope and "整张 GPU 状态" in scope

    # after_build / peak 字段
    pm = doc["process_memory"]
    assert pm["allocated_bytes_after_build"] == 2 * GIB
    assert pm["peak_allocated_bytes"] == 6 * GIB  # 2 GiB 驻留 + 3 GiB train + 1 GiB validation
    assert pm["peak_reserved_bytes"] == 6 * GIB
    assert pm["peak_allocated_gib"] == 6.0
    assert pm["peak_sample_source"] == "end"
    assert pm["allocated_bytes_at_end"] == 6 * GIB
    # 整卡信息单独成块，且与进程统计区分
    assert doc["gpu"]["total_bytes"] == 24 * GIB
    assert doc["gpu"]["free_bytes_at_start"] == 17_800_000_000
    assert doc["gpu"]["free_bytes_at_end"] == 17_800_000_000 - 6 * GIB
    assert doc["gpu"]["free_gib_at_start"] == round(17_800_000_000 / GIB, 3)
    # 阶段/采样点：只有真实的 trainer_run，不伪造 train/validation 分界
    assert {"initialize_cuda", "build_trainer", "trainer_run"} <= set(doc["phases"])
    assert "train" not in doc["phases"] and "validation" not in doc["phases"]
    assert {"start", "after_build", "after_trainer_run", "end"} <= set(doc["samples"])
    assert "after_train" not in doc["samples"] and "after_validation" not in doc["samples"]
    assert "trainer_run" in doc["peak_scope"] and "不存在独立的 train/validation 峰值" in doc["peak_scope"]
    assert doc["final_sample_error"] is None
    # 身份与运行元数据
    assert doc["model"]["model_id"] == "M0"
    assert doc["model"]["num_parameters"] == 148_193_644
    assert doc["model"]["architecture_sha256"] == "f" * 64
    assert doc["run"]["batch_size"] == 2
    assert doc["run"]["train_iterations"] == 1 and doc["run"]["validation_iterations"] == 1
    assert doc["run"]["cases"] == ["case_a", "case_b"]
    assert doc["started_at"] and doc["completed_at"] and doc["elapsed_sec"] >= 0
    assert doc["exception_type"] is None and doc["error"] is None


# =========================================================================== 2) OOM / ERROR 落盘
def test_build_phase_oom_writes_cuda_oom_and_exits_nonzero(tmp_path, monkeypatch):
    events: list = []
    api = FakeCudaApi(events=events)
    seen: dict = {}

    def _build(cfg_, plan_, **kwargs):
        seen["kwargs"] = kwargs
        events.append("build_trainer")
        api.allocate(1 * GIB)  # build 中途已占用部分显存
        raise _fake_oom()

    with pytest.raises(SystemExit) as excinfo:
        _flow(tmp_path, monkeypatch, api=api, build=_build, events=events)
    assert excinfo.value.code not in (0, None), "OOM 必须以非 0 退出"

    base = tmp_path / "outputs/diagnostics/m0/unit_gpu_profile"
    report_path = base / "diagnostics" / gmp.GPU_PROFILE_FILENAME
    doc = json.loads(report_path.read_text())  # OOM 也必须落盘
    assert doc["status"] == "CUDA_OOM"
    assert doc["phase"] == gmp.PHASE_BUILD
    assert doc["exception_type"] == "_FakeCudaOOM"
    assert "out of memory" in doc["error"].lower()
    assert doc["process_memory"]["peak_allocated_bytes"] == 1 * GIB
    assert "CUDA_OOM" in str(excinfo.value.code) and "build_trainer" in str(excinfo.value.code)


def test_train_phase_oom_writes_cuda_oom_after_peak_and_then_empty_cache(tmp_path, monkeypatch):
    api = FakeCudaApi()
    def _train_oom(phase):
        api.allocate(3 * GIB)
        raise _fake_oom()

    trainer = _FakeTrainer(on_train=_train_oom)
    with pytest.raises(SystemExit) as excinfo:
        _flow(tmp_path, monkeypatch, api=api, trainer=trainer)
    assert excinfo.value.code not in (0, None)

    report_path = tmp_path / "outputs/diagnostics/m0/unit_gpu_profile/diagnostics" / gmp.GPU_PROFILE_FILENAME
    api.json_path = report_path
    doc = json.loads(report_path.read_text())
    assert doc["status"] == "CUDA_OOM"
    assert doc["phase"] == gmp.PHASE_TRAIN
    assert doc["process_memory"]["peak_allocated_bytes"] == 5 * GIB
    assert doc["process_memory"]["peak_reserved_bytes"] == 5 * GIB


def _raise_oom():
    raise _fake_oom()


def _raise_oom_accepting_phase(phase):  # pragma: no cover - 供钩子签名兼容
    raise _fake_oom()


def test_train_phase_oom_empty_cache_only_after_report_written(tmp_path, monkeypatch):
    """`empty_cache` 必须发生在峰值统计落盘之后（本测试在 empty_cache 内检查文件已存在）。"""
    api = FakeCudaApi()
    report_path = tmp_path / "outputs/diagnostics/m0/unit_gpu_profile/diagnostics" / gmp.GPU_PROFILE_FILENAME
    api.json_path = report_path

    def _train_oom(phase):
        api.allocate(3 * GIB)
        assert not report_path.exists(), "落盘前不得调用 empty_cache（此处仅防御性检查）"
        raise _fake_oom()

    with pytest.raises(SystemExit):
        _flow(tmp_path, monkeypatch, api=api, trainer=_FakeTrainer(on_train=_train_oom))
    assert api.json_existed_at_empty_cache is True, "empty_cache 只能在报告落盘后调用"
    doc = json.loads(report_path.read_text())
    assert doc["status"] == "CUDA_OOM" and doc["phase"] == gmp.PHASE_TRAIN


def test_generic_exception_writes_error_json_and_exits_nonzero(tmp_path, monkeypatch):
    api = FakeCudaApi()

    def _train_boom(phase):
        api.allocate(1 * GIB)
        raise ValueError("synthetic failure")

    with pytest.raises(SystemExit) as excinfo:
        _flow(tmp_path, monkeypatch, api=api, trainer=_FakeTrainer(on_train=_train_boom))
    assert excinfo.value.code not in (0, None)

    report_path = tmp_path / "outputs/diagnostics/m0/unit_gpu_profile/diagnostics" / gmp.GPU_PROFILE_FILENAME
    doc = json.loads(report_path.read_text())
    assert doc["status"] == "ERROR"
    assert doc["phase"] == gmp.PHASE_TRAIN
    assert doc["exception_type"] == "ValueError" and "synthetic failure" in doc["error"]
    assert doc["status"] != "COMPLETED"


def test_missing_report_path_is_reported_when_write_fails(tmp_path, monkeypatch):
    """报告落盘失败也要非 0 退出，并保留原始异常摘要（不静默）。"""
    api = FakeCudaApi()
    monkeypatch.setattr(gmp.os, "replace", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(SystemExit):
        _flow(tmp_path, monkeypatch, api=api, trainer=_FakeTrainer(on_train=lambda phase: _raise_oom()))
    assert "empty_cache" not in api.calls, "报告未落盘时不得释放缓存"


# =========================================================================== 3) CPU 路径
def test_cpu_path_never_touches_cuda_and_reports_not_applicable(tmp_path, monkeypatch):
    api = ExplodingCudaApi()  # 任何 torch.cuda 访问都会 AssertionError
    monkeypatch.setattr(gmp.torch, "cuda", api)
    trainer = _FakeTrainer(amp=False)  # CPU 上 AMP 实际未启用

    def _build(cfg_, plan_, **kwargs):
        return trainer, _FakeModel(), kwargs["store"]

    mod = _load_train_module()
    monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(mod, "build_store", lambda cfg_: _FakeStore())
    monkeypatch.setattr(mod, "build_trainer", _build)
    cfg = load_experiment_config(M0_CONFIG)
    args = _args(device="cpu")
    mod.run_overfit(args, cfg, object())

    report_path = tmp_path / "outputs/diagnostics/m0/unit_gpu_profile/diagnostics" / gmp.GPU_PROFILE_FILENAME
    doc = json.loads(report_path.read_text())
    assert doc["status"] == "NOT_APPLICABLE"
    assert doc["reason"] == gmp.NOT_APPLICABLE_REASON_CPU
    assert doc["phase"] == gmp.PHASE_FINALIZE, "CPU 路径收尾后 phase 必须是 finalize"
    assert doc["cuda_api_used"] is False and doc["final_sample_error"] is None
    assert doc["cuda_device"] is None and doc["device_name"] is None
    # 不伪造数字：CPU 路径的 CUDA 专属区块整体为 null（不是一堆 0）
    assert doc["process_memory"] is None and doc["gpu"] is None
    assert doc["phases"] == {} and doc["samples"] == {}
    assert doc["amp_enabled"] is False


# =========================================================================== 4) 原子写 / 换算 / 分类
def test_json_write_is_atomic_and_leaves_no_partial_file(tmp_path, monkeypatch):
    target = tmp_path / "d" / gmp.GPU_PROFILE_FILENAME
    gmp.atomic_write_json(target, {"status": "COMPLETED"})
    assert json.loads(target.read_text())["status"] == "COMPLETED"
    assert [p.name for p in target.parent.iterdir() if p.name.endswith(".tmp")] == []

    monkeypatch.setattr(gmp.os, "replace", lambda *a, **k: (_ for _ in ()).throw(OSError("simulated replace failure")))
    with pytest.raises(OSError, match="simulated replace failure"):
        gmp.atomic_write_json(target, {"status": "ERROR"})
    monkeypatch.undo()
    assert json.loads(target.read_text())["status"] == "COMPLETED", "失败写盘不得破坏既有报告"
    assert [p.name for p in target.parent.iterdir() if p.name.endswith(".tmp")] == []


def test_profiler_write_failure_raises_profile_error(tmp_path, monkeypatch):
    profiler = gmp.GpuMemoryProfiler("cpu", output_path=tmp_path / gmp.GPU_PROFILE_FILENAME)
    profiler.start()
    monkeypatch.setattr(gmp.os, "replace", lambda *a, **k: (_ for _ in ()).throw(OSError("readonly")))
    with pytest.raises(gmp.GpuMemoryProfileError, match="无法落盘"):
        profiler.finalize(gmp.STATUS_ERROR, error="boom", exception_type="RuntimeError")


def test_derived_mib_gib_fields_cover_suffixed_keys(tmp_path, monkeypatch):
    """带后缀的字节键也必须派生 MiB/GiB（否则派生值会静默变成 null）。"""
    api = FakeCudaApi()
    report_path, _, _ = _flow(tmp_path, monkeypatch, api=api)
    doc = json.loads(report_path.read_text())
    pm = doc["process_memory"]
    assert pm["allocated_bytes_at_start"] == 0
    assert pm["allocated_gib_at_start"] == 0.0 and pm["allocated_mib_at_start"] == 0.0
    assert pm["allocated_gib_after_build"] == 2.0
    assert pm["peak_allocated_gib"] == 2.0 and pm["peak_reserved_mib"] == 2048.0
    assert doc["gpu"]["total_gib"] == 24.0 and doc["gpu"]["free_mib_at_start"] is not None
    # 任何 *_bytes 字段都必须有配套的 MiB/GiB 派生值，不允许 null
    for block in (pm, doc["gpu"]):
        for key, value in block.items():
            if key.endswith("_bytes") and value is not None:
                prefix = key[: -len("_bytes")]
                assert block[f"{prefix}_mib"] is not None, f"{prefix}_mib 缺失"
                assert block[f"{prefix}_gib"] is not None, f"{prefix}_gib 缺失"


def test_mib_gib_conversions():
    assert gmp.bytes_to_mib(1024**2) == 1.0
    assert gmp.bytes_to_gib(1024**3) == 1.0
    assert gmp.bytes_to_gib(1536 * 1024**2) == 1.5
    assert gmp.bytes_to_mib(None) is None and gmp.bytes_to_gib(None) is None
    assert gmp.bytes_to_mib(1024) == 0.001


def test_is_cuda_oom_classification():
    oom_type = type("OutOfMemoryError", (RuntimeError,), {})
    assert gmp.is_cuda_oom(oom_type("boom")) is True
    assert gmp.is_cuda_oom(_FakeCudaOOM("CUDA out of memory. Tried to allocate 1 GiB")) is True
    assert gmp.is_cuda_oom(RuntimeError("CUDA out of memory")) is True
    assert gmp.is_cuda_oom(ValueError("synthetic failure")) is False
    assert gmp.is_cuda_oom(RuntimeError("out of memory")) is False  # 非 CUDA 消息


# =========================================================================== 5) 既有语义不回归
def test_existing_overfit_args_and_output_dirs_unchanged(tmp_path, monkeypatch):
    api = FakeCudaApi()
    _, _events, kwargs = _flow(tmp_path, monkeypatch, api=api)
    base = tmp_path / "outputs/diagnostics/m0/unit_gpu_profile"
    assert kwargs["run_dir"] == base / "checkpoints"
    assert kwargs["metrics_dir"] == base / "metrics"
    assert kwargs["diagnostics_dir"] == base / "diagnostics"
    assert kwargs["validation_mode"] == "diagnostic_patch"
    assert kwargs["iterations"] == 1 and kwargs["val_iterations"] == 1 and kwargs["epochs"] == 1
    assert kwargs["train_cases"] == ["case_a", "case_b"]
    assert kwargs["device"] == "cuda"
    # 报告落在 diagnostics 目录内，且未新增其他顶层产物
    profile = base / "diagnostics" / gmp.GPU_PROFILE_FILENAME
    assert profile.is_file()
    assert sorted(p.name for p in base.iterdir()) == ["diagnostics"]
    assert sorted(p.name for p in (base / "diagnostics").iterdir()) == [gmp.GPU_PROFILE_FILENAME]


def test_static_profiling_report_still_not_measured(tmp_path):
    """`models/profiling.py` 的静态报告仍标记 NOT_MEASURED（运行时实测是独立产物）。"""
    class _TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv3d(3, 4, kernel_size=3, padding=1)

    plan = load_plan(str(PLANS), "3d_fullres", "Dataset605_PICAI")
    spec = load_architecture_config(M0_ARCH)
    report = structure_summary(
        _TinyModel(), plan, spec, in_channels=3, num_classes=2, batch_size=2
    )
    assert report["peak_gpu_memory"]["status"] == "NOT_MEASURED"
    assert "analytical_activation_bytes_fp32" in report["peak_gpu_memory"]
    assert report["parameters"]["total"] > 0


def test_profiler_default_cpu_is_silent_about_cuda(tmp_path, monkeypatch):
    """库层默认不建目录、不打印；CPU 实例化不触发任何 CUDA 调用（注入的 api 一次性校验）。"""
    api = ExplodingCudaApi()
    profiler = gmp.GpuMemoryProfiler("cpu", output_path=tmp_path / "sub" / gmp.GPU_PROFILE_FILENAME, cuda_api=api)
    profiler.start()
    profiler.mark_phase(gmp.PHASE_BUILD)
    profiler.snapshot("after_build")
    profiler.update(model={"model_id": "M0"})
    path = profiler.finalize(gmp.STATUS_NOT_APPLICABLE)
    assert path.is_file()
    assert profiler.release_cache() is None  # CPU no-op，且不得触碰 api
    doc = json.loads(path.read_text())
    assert doc["model"]["model_id"] == "M0"
    assert np.isfinite(doc["elapsed_sec"])


# =========================================================================== 第 9 轮：真实分界与初始化保护
def test_validation_phase_oom_reports_phase_validation(tmp_path, monkeypatch):
    """B：trainer 先进入 train、再置 validation 后 OOM → phase=validation（不再伪装成 train）。"""
    api = FakeCudaApi()

    def _hook(phase):
        api.allocate(2 * GIB if phase == "train" else GIB)
        if phase == "validation":
            raise _fake_oom()

    trainer = _FakeTrainer(on_train=_hook)
    with pytest.raises(SystemExit) as excinfo:
        _flow(tmp_path, monkeypatch, api=api, trainer=trainer)
    assert excinfo.value.code not in (0, None)

    report_path = tmp_path / "outputs/diagnostics/m0/unit_gpu_profile/diagnostics" / gmp.GPU_PROFILE_FILENAME
    doc = json.loads(report_path.read_text())
    assert doc["status"] == "CUDA_OOM"
    assert doc["phase"] == gmp.PHASE_VALIDATION, f"validation OOM 必须记 phase=validation，实际 {doc['phase']}"
    assert "validation" in str(excinfo.value.code)
    assert doc["process_memory"]["peak_allocated_bytes"] == 5 * GIB  # 2 驻留 + 2 train + 1 validation


def test_unknown_current_phase_falls_back_to_trainer_run(tmp_path, monkeypatch):
    """C 补充：trainer 未暴露 current_phase（或取值未知）→ phase=trainer_run。"""
    api = FakeCudaApi()

    trainer = _FakeTrainer(phases=("train",))

    def _hook(phase):
        trainer.current_phase = "warmup"  # 未知取值 → 回退为 trainer_run
        raise ValueError("boom without known phase")

    trainer._on_train = _hook
    with pytest.raises(SystemExit):
        _flow(tmp_path, monkeypatch, api=api, trainer=trainer)
    report_path = tmp_path / "outputs/diagnostics/m0/unit_gpu_profile/diagnostics" / gmp.GPU_PROFILE_FILENAME
    doc = json.loads(report_path.read_text())
    assert doc["phase"] == gmp.PHASE_TRAINER_RUN and doc["status"] == "ERROR"


def test_trainer_run_peak_covers_build_train_and_validation(tmp_path, monkeypatch):
    """D：trainer_run 峰值覆盖 build + train + validation 的全部模拟分配。"""
    api = FakeCudaApi()

    def _hook(phase):
        api.allocate(int(3.5 * GIB) if phase == "train" else int(1.0 * GIB))

    trainer = _FakeTrainer(on_train=_hook)
    report_path, _, _ = _flow(tmp_path, monkeypatch, api=api, trainer=trainer)
    doc = json.loads(report_path.read_text())
    pm = doc["process_memory"]
    assert pm["allocated_bytes_after_build"] == 2 * GIB
    assert pm["peak_allocated_gib"] == round((2 + 3.5 + 1.0), 3)  # 6.5 GiB
    assert pm["peak_reserved_bytes"] == pm["peak_allocated_bytes"]
    assert set(doc["samples"]) == {"start", "after_build", "after_trainer_run", "end"}


def test_initialize_cuda_reset_failure_still_writes_error_json(tmp_path, monkeypatch):
    """E：reset_peak_memory_stats 抛错 → ERROR JSON（phase=initialize_cuda）+ 非 0 退出。"""
    api = FakeCudaApi(reset_error=RuntimeError("simulated reset failure"))
    events: list = []
    with pytest.raises(SystemExit) as excinfo:
        _flow(tmp_path, monkeypatch, api=api, events=events)
    assert excinfo.value.code not in (0, None)
    assert "build_trainer" not in events, "initialize_cuda 失败后不得继续 build_trainer"

    report_path = tmp_path / "outputs/diagnostics/m0/unit_gpu_profile/diagnostics" / gmp.GPU_PROFILE_FILENAME
    doc = json.loads(report_path.read_text())
    assert doc["status"] == "ERROR"
    assert doc["phase"] == gmp.PHASE_INITIALIZE
    assert "simulated reset failure" in doc["error"]
    assert doc["process_memory"] is None and doc["gpu"] is None  # 无任何样本可伪造


def test_initialize_cuda_mem_get_info_failure_still_writes_error_json(tmp_path, monkeypatch):
    """F：start 阶段第一次 mem_get_info 抛错 → 仍写 ERROR JSON，并记录 final_sample_error。"""
    api = FakeCudaApi(mem_get_info_error_after=0)  # 第一次调用即失败
    with pytest.raises(SystemExit):
        _flow(tmp_path, monkeypatch, api=api)
    report_path = tmp_path / "outputs/diagnostics/m0/unit_gpu_profile/diagnostics" / gmp.GPU_PROFILE_FILENAME
    doc = json.loads(report_path.read_text())
    assert doc["status"] == "ERROR" and doc["phase"] == gmp.PHASE_INITIALIZE
    assert doc["process_memory"] is None
    assert "simulated mem_get_info failure" in (doc["final_sample_error"] or "")
    assert isinstance(doc["final_sample_error"], str)


def test_cuda_unavailable_skips_build_and_exits_nonzero(tmp_path, monkeypatch):
    """G：CUDA requested 但 is_available=False → 不调用 build_trainer、ERROR JSON、非 0。"""
    api = FakeCudaApi(available=False)
    events: list = []
    with pytest.raises(SystemExit) as excinfo:
        _flow(tmp_path, monkeypatch, api=api, events=events)
    assert excinfo.value.code not in (0, None)
    assert "build_trainer" not in events, "CUDA 不可用时不得继续 build_trainer"
    report_path = tmp_path / "outputs/diagnostics/m0/unit_gpu_profile/diagnostics" / gmp.GPU_PROFILE_FILENAME
    doc = json.loads(report_path.read_text())
    assert doc["status"] == "ERROR"
    assert doc["phase"] == gmp.PHASE_INITIALIZE
    assert "is_available" in doc["error"] and doc["cuda_api_used"] is False


def test_end_sample_failure_keeps_report_with_final_sample_error(tmp_path, monkeypatch):
    """H：finalize 的 end 采样失败 → 用已有 start/峰值样本落盘，并记录 final_sample_error。"""
    # 调用次序：start(1) → snapshot after_build(2) → finalize 的 end(3) 失败
    api = FakeCudaApi(mem_get_info_error_after=2)
    out = tmp_path / "diagnostics" / gmp.GPU_PROFILE_FILENAME
    profiler = gmp.GpuMemoryProfiler(
        "cuda",
        output_path=out,
        cuda_api=api,
        metadata={"model": {"model_id": "M4"}, "run": {"run_name": "end_sample_failure"}},
    )
    profiler.start()
    api.allocate(2 * GIB)
    profiler.snapshot("after_build")
    path = profiler.finalize(gmp.STATUS_CUDA_OOM, error="CUDA out of memory", exception_type="_FakeCudaOOM")
    doc = json.loads(Path(path).read_text())
    assert doc["status"] == "CUDA_OOM"
    assert "simulated mem_get_info failure" in doc["final_sample_error"]
    assert "end" not in doc["samples"], "end 采样失败时不得伪造 end 样本"
    assert {"start", "after_build"} <= set(doc["samples"])
    pm = doc["process_memory"]
    assert pm["peak_allocated_bytes"] == 2 * GIB, "end 采样失败时峰值必须回退到已有样本"
    assert pm["peak_allocated_gib"] == 2.0
    assert pm["allocated_bytes_at_start"] == 0
    assert doc["error"] == "CUDA out of memory"


def test_overfit_case_hint_and_delivery_use_positive_cases(tmp_path, monkeypatch):
    """J：错误消息/示例不得再出现 fold-0 阴性病例 10006_1000006。"""
    import inspect

    mod = _load_train_module()
    monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path)
    cfg = load_experiment_config(M0_CONFIG)
    with pytest.raises(SystemExit) as excinfo:
        mod.run_overfit(_args(cases=""), cfg, object())
    message = str(excinfo.value)
    assert "10005_1000005,10021_1000021" in message
    assert "10006_1000006" not in message

    src = inspect.getsource(mod.run_overfit)
    assert "10021_1000021" in src and "10006_1000006" not in src
