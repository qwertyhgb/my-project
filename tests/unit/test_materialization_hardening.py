"""物化前加固的合成测试（纯合成数据；不读取真实医学影像、不运行物化/训练）。

覆盖：
1. v2.4 实验身份一致性（文件名 / architecture_version / experiment.name / output_root / 来源标记）；
2. `run_loader_smoke` 的 M3/M4 3-tuple 分支与 M0–M2 只读 image 分支（含 JSON prior 字段）；
3. 严格离散标签（1.5 / 2.1 / -0.2 / NaN / Inf 全部拒绝）；
4. oracle 不通过时物化器拒绝写盘（fail-closed）；
5. sidecar 数组被篡改 → 哈希校验失败；plan/config/source/channel semantics 不符 → 失败；
6. manifest 缺病例 / 重复 / 哈希变化 → 失败；
7. checkpoint 续训时 prior manifest 哈希不同 → 拒绝；
8. 原子写失败不留最终文件；
9. 有效 resume 不调用昂贵 converter；N 例只构造一次 dataset；
10. dry-run 真正执行转换但零文件写入。
"""
from __future__ import annotations

import json
import os
import types
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from zonal_reliability_fusion.config.architecture import load_architecture_config
from zonal_reliability_fusion.config.experiment import load_experiment_config
from zonal_reliability_fusion.data.preprocessed_store import (
    PreprocessedCase,
    PreprocessedStore,
    PreprocessedStoreError,
)
from zonal_reliability_fusion.data.zonal_prior import (
    SidecarCommitError,
    ZonalPriorError,
    ZonalPriorMaterializer,
    array_sha256,
    build_manifest,
    is_sha256_hex,
    label_to_pz_tz,
    metadata_sha256_of,
    sidecar_identity_ok,
    validate_sidecar,
    verify_manifest,
    write_prior_sidecar,
)
from zonal_reliability_fusion.data.zonal_prior import (
    json_sha256 as zonal_prior_json_sha256,
)
from zonal_reliability_fusion.training.trainer import M0Trainer

PROJECT = Path("/opt/data/private/lm/my-projects")
V24_CONFIGS = {
    "m1_equal": ("M1", None),
    "m2_image_gate": ("M2", None),
    "m3_zone_input": ("M3", "yuan"),
    "m4_conditioned_gate": ("M4", "yuan"),
}


def _bare_store(out_root):
    """构造只测 sidecar 的 store（绕过目录校验），补齐 store 方法触达的全部属性。"""
    from types import MappingProxyType

    store = PreprocessedStore.__new__(PreprocessedStore)
    store.zonal_prior_root = out_root
    store.zonal_prior_source = "yuan"
    store.expected_configuration = "3d_fullres"
    store.expected_plans_sha256 = "a" * 64
    store.zonal_prior_manifest_sha256 = None
    store.zonal_prior_array_hash_checked = False
    store.zonal_prior_validation_stats = {}
    store.zonal_prior_frozen_records = MappingProxyType({})
    store.require_frozen_manifest = False
    return store


def _valid_prior(shape=(2, 4, 4, 4)) -> np.ndarray:
    prior = np.zeros(shape, dtype=np.float32)
    prior[0, 0, 0, 0] = 1.0
    return prior


def _hex(seed: str) -> str:
    """确定性 64 位小写 hex（用于构造合法 manifest 记录）。"""
    import hashlib

    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _metadata(prior: np.ndarray, *, source: str = "yuan", plans: str = "a" * 64, config: str = "3d_fullres",
              input_hashes: dict | None = None, oracle_ok: bool = True) -> dict:
    return {
        "source": source,
        "configuration": config,
        "plans_sha256": plans,
        "input_sha256": input_hashes
        or {"zonal": _hex("h1"), "lesion": _hex("h2"), "properties": _hex("h3")},
        "oracle": {
            "exact_match": oracle_ok,
            "shape_match": True,
            "mismatch_voxels": 0 if oracle_ok else 9,
        },
    }


def _identity_ok(name: str, output_root: str, source: str | None) -> bool:
    """版本一致性判据：experiment.name 与 output_root 必须含 v24（M3/M4 还须含来源）。"""
    if "v24" not in name or "v24" not in output_root:
        return False
    if "v23" in name or "v23" in output_root:
        return False
    return not (source is not None and (source not in name or source not in output_root))


# --------------------------------------------------------------------------- 1. 配置身份
@pytest.mark.parametrize("tag", list(V24_CONFIGS))
def test_v24_config_identity_is_consistent(tag):
    model_id, source = V24_CONFIGS[tag]
    path = PROJECT / f"configs/experiments/{tag}_picai_3d_fullres_v24.yaml"
    cfg = load_experiment_config(path)
    spec = load_architecture_config(cfg.resolve_path(cfg.architecture.config_path))
    assert cfg.experiment.model_id == model_id
    assert path.name.endswith("_v24.yaml")
    # 文件名 / architecture_version / experiment.name / output_root 的版本必须一致
    assert cfg.architecture.architecture_version == spec.architecture_version == "v2.4"
    # 正向断言：文件名 / experiment.name / output_root 的**路径组件**必须含 v24（v23 不得出现）
    assert "v24" in cfg.experiment.name
    assert "v23" not in cfg.experiment.name
    assert _identity_ok(cfg.experiment.name, cfg.paths.output_root, source)
    assert "v24" in path.name
    # 负向对照：去掉 v24 的同一路径必须**不能**通过该判据（证明断言有效）
    assert not _identity_ok(f"{tag}_v23", f"outputs/checkpoints/{tag}", None)
    if source:
        assert source in cfg.experiment.name and source in cfg.paths.output_root
        assert cfg.data.zonal_prior_source == source
    else:
        assert cfg.data.zonal_prior_source is None and cfg.data.zonal_prior_root is None


def test_v24_output_roots_are_distinct():
    roots = []
    for tag in V24_CONFIGS:
        cfg = load_experiment_config(PROJECT / f"configs/experiments/{tag}_picai_3d_fullres_v24.yaml")
        roots.append(cfg.paths.output_root)
    assert len(set(roots)) == len(roots)


# --------------------------------------------------------------------------- 3. 严格离散标签
@pytest.mark.parametrize("bad", [1.5, 2.1, -0.2, np.nan, np.inf, -np.inf])
def test_label_to_pz_tz_rejects_non_integer_and_non_finite(bad):
    label = np.zeros((2, 2, 2), dtype=np.float32)
    label[0, 0, 0] = bad
    with pytest.raises(ZonalPriorError) as excinfo:
        label_to_pz_tz(label)
    message = str(excinfo.value)
    if np.isfinite(bad):
        # 保留原始非法值（不是截断后的 1/2）
        assert repr(float(np.float32(bad))) in message
        assert "只允许精确取值" in message
    else:
        assert "非有限值" in message


def test_label_to_pz_tz_accepts_exact_integers_only():
    label = np.zeros((2, 2, 2), dtype=np.int16)
    label[0, 0, 0], label[1, 1, 1] = 1, 2
    prior = label_to_pz_tz(label)
    assert prior.shape == (2, 2, 2, 2) and prior[0, 0, 0, 0] == 1.0 and prior[1, 1, 1, 1] == 1.0


# --------------------------------------------------------------------------- 4. oracle fail-closed
def test_materializer_rejects_oracle_failure_without_writing(tmp_path):
    out_root = tmp_path / "out"

    def convert(_cid: str):
        return _valid_prior(), _metadata(_valid_prior(), oracle_ok=False)

    materializer = ZonalPriorMaterializer(out_root, source="yuan", configuration="3d_fullres", plans_sha256="a" * 64)
    summary = materializer.run(["c1"], convert, progress=False)
    assert summary["n_ok"] == 0 and summary["n_failed"] == 1
    assert "oracle rejection" in summary["failed"][0]["reason"]
    assert not (out_root / "c1.npz").exists() and not (out_root / "c1.json").exists()


def test_materializer_rejects_missing_oracle_metadata(tmp_path):
    out_root = tmp_path / "out2"

    def convert(_cid: str):
        meta = _metadata(_valid_prior())
        meta.pop("oracle")
        return _valid_prior(), meta

    materializer = ZonalPriorMaterializer(out_root, source="yuan", configuration="3d_fullres", plans_sha256="a" * 64)
    summary = materializer.run(["c1"], convert, progress=False)
    assert summary["n_failed"] == 1 and not (out_root / "c1.json").exists()


# --------------------------------------------------------------------------- 5. sidecar 完整性
def test_sidecar_array_tampering_is_detected(tmp_path):
    out_root = tmp_path / "sidecar"
    prior = _valid_prior()
    write_prior_sidecar(out_root, "c1", prior, _metadata(prior))
    # 篡改数组（保持形状），元数据 hash 不变
    tampered = prior.copy()
    tampered[0, 1, 1, 1] = 1.0
    with np.load(out_root / "c1.npz") as npz:
        stored = dict(npz)
    stored["pz_tz"] = tampered
    np.savez_compressed(out_root / "c1.npz", **stored)
    ok, why = sidecar_identity_ok(out_root, "c1", _metadata(tampered))
    assert not ok and "SHA256" in why
    store = _bare_store(out_root)
    with pytest.raises(PreprocessedStoreError, match="SHA256"):
        store.load_zonal_prior("c1", seg_shape=(4, 4, 4))


@pytest.mark.parametrize(
    "mutate, match",
    [
        ({"source": "hevi"}, "来源"),
        ({"configuration": "2d"}, "configuration"),
        ({"plans_sha256": "b" * 64}, "plans_sha256"),
        ({"output_channel_semantics": ["WG"]}, "channel_semantics"),
        ({"output_dtype": "float16"}, "output_dtype"),
    ],
)
def test_sidecar_field_mismatch_fails(tmp_path, mutate, match):
    out_root = tmp_path / f"sidecar_{abs(hash(match))}"
    prior = _valid_prior()
    write_prior_sidecar(out_root, "c1", prior, _metadata(prior))
    meta_path = out_root / "c1.json"
    meta = json.loads(meta_path.read_text())
    meta.update(mutate)
    meta_path.write_text(json.dumps(meta))
    store = _bare_store(out_root)
    with pytest.raises(PreprocessedStoreError, match=match):
        store.load_zonal_prior("c1", seg_shape=(4, 4, 4))


# --------------------------------------------------------------------------- 6. manifest 校验
def _records(ids):
    return [
        {
            "case_id": cid,
            "input_sha256": {"zonal": _hex(f"z-{cid}")},
            "output_sha256": _hex(f"o-{cid}"),
            "metadata_sha256": _hex(f"m-{cid}"),
        }
        for cid in ids
    ]


def test_manifest_missing_extra_duplicate_and_hash_change():
    manifest = build_manifest(_records(["a", "b", "c"]), source="yuan", plans_sha256="a" * 64, configuration="3d_fullres")
    verify_manifest(manifest, ["a", "b", "c"], source="yuan", plans_sha256="a" * 64, configuration="3d_fullres")
    with pytest.raises(ZonalPriorError, match="覆盖范围"):
        verify_manifest(manifest, ["a", "b"])
    with pytest.raises(ZonalPriorError, match="覆盖范围"):
        verify_manifest(manifest, ["a", "b", "c", "d"])
    with pytest.raises(ZonalPriorError, match="重复"):
        build_manifest(_records(["a", "a"]), source="yuan", plans_sha256="a" * 64, configuration="3d_fullres")
    tampered = dict(manifest)
    tampered["cases"] = manifest["cases"][:-1]
    tampered["case_ids"] = manifest["case_ids"][:-1]
    tampered["n_cases"] = 2
    with pytest.raises(ZonalPriorError, match="manifest_sha256"):
        verify_manifest(tampered, ["a", "b"])
    with pytest.raises(ZonalPriorError, match="plans_sha256"):
        verify_manifest(manifest, ["a", "b", "c"], plans_sha256="b" * 64)


# --------------------------------------------------------------------------- 7. resume 冻结键
def test_resume_rejects_changed_prior_manifest_hash():
    assert M0Trainer._is_frozen_config_key("provenance.zonal_prior_manifest_sha256")
    current_extra = {key: 0 for key in M0Trainer.PROTOCOL_KEYS}
    fake_self = types.SimpleNamespace(
        config_snapshot={"provenance": {"zonal_prior_manifest_sha256": "AAA"}},
        PROTOCOL_KEYS=M0Trainer.PROTOCOL_KEYS,
        _is_frozen_config_key=M0Trainer._is_frozen_config_key,
    )

    def _checkpoint_extra():
        return dict(current_extra)

    fake_self._checkpoint_extra = _checkpoint_extra
    with pytest.raises(RuntimeError, match="zonal_prior_manifest_sha256"):
        M0Trainer._verify_resume_consistency(
            fake_self,
            {"provenance": {"zonal_prior_manifest_sha256": "BBB"}},
            current_extra,
            allow_change=False,
        )
    # 相同哈希 → 无差异
    assert (
        M0Trainer._verify_resume_consistency(
            fake_self,
            {"provenance": {"zonal_prior_manifest_sha256": "AAA"}},
            current_extra,
            allow_change=False,
        )
        is False
    )


# --------------------------------------------------------------------------- 8. 原子写
def test_atomic_write_failure_leaves_no_final_file(tmp_path, monkeypatch):
    out_root = tmp_path / "atomic"
    prior = _valid_prior()

    import zonal_reliability_fusion.data.zonal_prior as zp

    def _boom(src, dst):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(zp.os, "replace", _boom)
    with pytest.raises(SidecarCommitError, match="simulated replace failure") as excinfo:
        write_prior_sidecar(out_root, "c1", prior, _metadata(prior))
    assert isinstance(excinfo.value.original, OSError)
    assert not (out_root / "c1.json").exists() and not (out_root / "c1.npz").exists()
    assert _residue(out_root) == []


# --------------------------------------------------------------------------- 9. resume 不调用 converter
def test_valid_resume_skips_expensive_converter(tmp_path):
    out_root = tmp_path / "resume"
    prior = _valid_prior()
    meta = _metadata(prior)
    write_prior_sidecar(out_root, "c1", prior, meta)
    calls = {"convert": 0, "probe": 0}

    def convert(_cid):
        calls["convert"] += 1
        raise AssertionError("resume 命中时不得调用昂贵 converter")

    def probe(_cid):
        calls["probe"] += 1
        return {"source": "yuan", "configuration": "3d_fullres", "plans_sha256": "a" * 64,
                "input_sha256": meta["input_sha256"]}

    materializer = ZonalPriorMaterializer(
        out_root, source="yuan", configuration="3d_fullres", plans_sha256="a" * 64, resume=True
    )
    summary = materializer.run(["c1"], convert, identity_probe=probe, progress=False)
    assert summary["n_skipped"] == 1 and calls == {"convert": 0, "probe": 1}
    # 身份不匹配时**不得跳过**：pre-check 失败 → 必须真正调用 converter
    def bad_probe(_cid):
        return {"source": "yuan", "configuration": "3d_fullres", "plans_sha256": "a" * 64,
                "input_sha256": {"zonal": "changed"}}

    def convert_counting(_cid):
        calls["convert"] += 1
        return prior, meta

    calls["convert"] = 0
    materializer2 = ZonalPriorMaterializer(
        out_root, source="yuan", configuration="3d_fullres", plans_sha256="a" * 64, resume=True
    )
    materializer2.run(["c1"], convert_counting, identity_probe=bad_probe, progress=False)
    assert calls["convert"] == 1, "身份不匹配时不得标记 skipped"

    # 半成品/损坏 sidecar（缺 .npz）必须失败或显式重算，绝不静默跳过
    (out_root / "c1.npz").unlink()
    calls["convert"] = 0
    materializer3 = ZonalPriorMaterializer(
        out_root, source="yuan", configuration="3d_fullres", plans_sha256="a" * 64, resume=True
    )
    summary3 = materializer3.run(["c1"], convert_counting, identity_probe=probe, progress=False)
    assert calls["convert"] == 1 and summary3["n_skipped"] == 0 and summary3["n_ok"] == 1
    assert (out_root / "c1.npz").is_file()


# --------------------------------------------------------------------------- 10. dry-run
def test_dry_run_converts_but_writes_nothing(tmp_path):
    out_root = tmp_path / "dry"
    calls = {"n": 0}

    def convert(_cid):
        calls["n"] += 1
        return _valid_prior(), _metadata(_valid_prior())

    materializer = ZonalPriorMaterializer(out_root, source="yuan", configuration="3d_fullres", plans_sha256="a" * 64)
    summary = materializer.run(["c1", "c2"], convert, dry_run=True, progress=False)
    assert calls["n"] == 2 and summary["n_ok"] == 2 and summary["dry_run"] is True
    assert list(out_root.glob("*")) == [] if out_root.exists() else True
    assert "manifest_sha256" not in summary


# --------------------------------------------------------------------------- 11. dataset 只构造一次
class _FakeBlosc2Dataset:
    instances = 0

    def __init__(self, folder: str) -> None:
        type(self).instances += 1
        self.folder = folder

    def __getitem__(self, case_id: str):
        seg = np.zeros((1, 4, 4, 4), dtype=np.int16)
        return np.zeros((3, 4, 4, 4), dtype=np.float32), seg, None, {}


def test_case_converter_builds_dataset_once(tmp_path, monkeypatch):
    import importlib
    import sys

    from scripts.data.materialize_zonal_prior import CaseConverter

    fake_module = types.ModuleType("nnunetv2.training.dataloading.nnunet_dataset")
    fake_module.nnUNetDatasetBlosc2 = _FakeBlosc2Dataset
    monkeypatch.setitem(sys.modules, "nnunetv2", types.ModuleType("nnunetv2"))
    monkeypatch.setitem(sys.modules, "nnunetv2.training", types.ModuleType("nnunetv2.training"))
    monkeypatch.setitem(sys.modules, "nnunetv2.training.dataloading", types.ModuleType("nnunetv2.training.dataloading"))
    monkeypatch.setitem(sys.modules, "nnunetv2.training.dataloading.nnunet_dataset", fake_module)
    _FakeBlosc2Dataset.instances = 0

    args = types.SimpleNamespace(source="yuan", cases_root=str(tmp_path), configuration="3d_fullres",
                                 preprocessed_dataset=str(tmp_path), plans="")
    context = {"configuration_manager": types.SimpleNamespace(data_identifier="nnUNetPlans_3d_fullres")}
    converter = CaseConverter(args, context)
    for _ in range(5):
        converter._read_seg_only("case_x")
    assert _FakeBlosc2Dataset.instances == 1
    assert converter.dataset_constructed == 1
    _ = importlib  # 保持 import 使用


# --------------------------------------------------------------------------- 2. loader-smoke 分派
class _FakePriorStore:
    """duck-typing store：train/val ids + prior sidecar + manifest。"""

    def __init__(self, *, with_prior: bool = True, manifest_hash: str = "M" * 64):
        self.train_ids = ("case_a",)
        self.val_ids = ("case_b",)
        self.fold = 0
        self.zonal_prior_root = "fake-root" if with_prior else None
        self.zonal_prior_source = "yuan" if with_prior else None
        self.expected_configuration = "3d_fullres"
        self.expected_plans_sha256 = "a" * 64
        self.zonal_prior_manifest_sha256 = manifest_hash if with_prior else None
        self._with_prior = with_prior
        self.get_case_calls: list[dict] = []

    def get_case(self, case_id: str, *, include_zonal_prior: bool = False):
        self.get_case_calls.append({"case_id": case_id, "include_zonal_prior": include_zonal_prior})
        if include_zonal_prior and not self._with_prior:
            raise PreprocessedStoreError(f"{case_id}: 缺 PZ/TZ prior sidecar")
        prior = None
        if include_zonal_prior:
            prior = np.zeros((2, 4, 5, 6), dtype=np.float32)
            prior[0, 0, 0, 0] = 1.0
            prior[1] = 1.0 - prior[0]
        shape = (4, 5, 6)
        grid = np.indices(shape)
        return PreprocessedCase(
            case_id=case_id,
            data=np.stack([grid[0], grid[1], grid[2]]).astype(np.float32),
            seg=(grid[1] == 2).astype(np.uint8)[None],
            properties=types.SimpleNamespace(spacing=None, shape_before_cropping=None, class_locations=None),
            zonal_prior=prior,
        )

    def load_zonal_prior_manifest(self, expected_case_ids, **kwargs):
        if not self._with_prior:
            raise PreprocessedStoreError("prior root 未配置")
        return {
            "manifest_sha256": self.zonal_prior_manifest_sha256,
            "n_cases": len(expected_case_ids),
            "source": "yuan",
            "configuration": "3d_fullres",
            "plans_sha256": "a" * 64,
        }

    def describe_zonal_prior(self) -> dict:
        return {"enabled": self._with_prior, "source": self.zonal_prior_source}


def _smoke_args(tmp_path):
    return types.SimpleNamespace(
        cases="", limit_cases=1, no_progress=True, run_name=f"synthetic_smoke_{tmp_path.name}"
    )


def _patch_smoke_module(monkeypatch, tmp_path):
    from scripts.train import train_m0

    monkeypatch.setattr(train_m0, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(train_m0, "check_raw_dataset_json", lambda: {"available": False})
    monkeypatch.setattr(train_m0, "code_version_info", lambda _root: {})
    return train_m0


def test_loader_smoke_m4_reads_prior(tmp_path, monkeypatch):
    train_m0 = _patch_smoke_module(monkeypatch, tmp_path)
    cfg = load_experiment_config(PROJECT / "configs/experiments/m4_conditioned_gate_picai_3d_fullres_v24.yaml")
    plan = types.SimpleNamespace(patch_size=(5, 5, 5))
    store = _FakePriorStore()
    train_m0.run_loader_smoke(_smoke_args(tmp_path), cfg, plan, store=store)
    assert all(call["include_zonal_prior"] for call in store.get_case_calls)
    payload_path = next((tmp_path / "outputs/diagnostics/m4").rglob("loader_smoke_*.json"))
    payload = json.loads(payload_path.read_text())
    assert payload["requires_zonal_prior"] is True
    assert payload["zonal_prior_manifest"]["checked"] is True
    assert payload["zonal_prior_manifest"]["manifest_sha256"] == "M" * 64
    case_prior = payload["cases"][0]["prior"]
    assert case_prior["shape"] == [2, 4, 5, 6]
    assert case_prior["channels"] == ["PZ", "TZ"]
    assert case_prior["max_pz_plus_tz"] <= 1.0 + 1e-6
    assert payload["cases"][0]["sampler"]["force_fg=True"]["prior_bbox_identical"] is True
    assert payload["batches"][0]["prior"]["shape"][1] == 2


def test_loader_smoke_m4_fails_without_sidecar(tmp_path, monkeypatch):
    train_m0 = _patch_smoke_module(monkeypatch, tmp_path)
    cfg = load_experiment_config(PROJECT / "configs/experiments/m4_conditioned_gate_picai_3d_fullres_v24.yaml")
    plan = types.SimpleNamespace(patch_size=(5, 5, 5))
    store = _FakePriorStore(with_prior=False)
    with pytest.raises(SystemExit, match="prior manifest 校验失败"):
        train_m0.run_loader_smoke(_smoke_args(tmp_path), cfg, plan, store=store)


def test_loader_smoke_m2_never_reads_prior(tmp_path, monkeypatch):
    train_m0 = _patch_smoke_module(monkeypatch, tmp_path)
    cfg = load_experiment_config(PROJECT / "configs/experiments/m2_image_gate_picai_3d_fullres_v24.yaml")
    plan = types.SimpleNamespace(patch_size=(5, 5, 5))
    store = _FakePriorStore(with_prior=False)
    train_m0.run_loader_smoke(_smoke_args(tmp_path), cfg, plan, store=store)
    assert all(call["include_zonal_prior"] is False for call in store.get_case_calls)
    payload_path = next((tmp_path / "outputs/diagnostics/m2").rglob("loader_smoke_*.json"))
    payload = json.loads(payload_path.read_text())
    assert payload["requires_zonal_prior"] is False
    assert "prior" not in payload["cases"][0]
    assert "prior" not in payload["batches"][0]


# --------------------------------------------------------------------------- 12. store manifest 读取
def test_store_manifest_roundtrip_and_coverage(tmp_path):
    out_root = tmp_path / "manifest_root"
    records, metas = [], []
    for cid in ("case_a", "case_b"):
        prior = _valid_prior()
        meta = _metadata(prior, input_hashes={"zonal": _hex(f"z-{cid}")})
        write_prior_sidecar(out_root, cid, prior, meta)
        written = json.loads((out_root / f"{cid}.json").read_text())
        metas.append(written)
        records.append(
            {
                "case_id": cid,
                "input_sha256": written["input_sha256"],
                "output_sha256": written["output_sha256"],
                "metadata_sha256": metadata_sha256_of(written),
            }
        )
    manifest = build_manifest(records, source="yuan", plans_sha256="a" * 64, configuration="3d_fullres")
    (out_root / "manifest.json").write_text(json.dumps(manifest))
    store = _bare_store(out_root)
    loaded = store.load_zonal_prior_manifest(["case_a", "case_b"])
    assert store.zonal_prior_array_hash_checked is True
    assert loaded["manifest_sha256"] == manifest["manifest_sha256"]
    assert store.zonal_prior_manifest_sha256 == manifest["manifest_sha256"]
    # 覆盖范围不符 → 失败
    with pytest.raises(PreprocessedStoreError, match="覆盖范围"):
        store.load_zonal_prior_manifest(["case_a"])
    # 单文件被替换（input hash 变）→ 失败
    tampered = json.loads((out_root / "case_a.json").read_text())
    tampered["input_sha256"] = {"zonal": _hex("hacked")}
    (out_root / "case_a.json").write_text(json.dumps(tampered))
    with pytest.raises(PreprocessedStoreError, match="metadata_sha256|input_sha256"):
        store.load_zonal_prior_manifest(["case_a", "case_b"])
    assert array_sha256(_valid_prior()) == metas[0]["output_sha256"]
    assert torch is not None  # 保持 torch 导入（供其他用例使用）
    assert yaml is not None


# ===========================================================================
# 本轮 P0/P1 回归测试（首轮物化即可加载、成对提交、manifest 发布门控、深度校验）
# ===========================================================================
PLANS = "a" * 64


def _materializer(out_root, **kw):
    return ZonalPriorMaterializer(
        out_root, source="yuan", configuration="3d_fullres", plans_sha256=PLANS, **kw
    )


def _convert_ok(case_id: str, *, dtype=np.float32):
    def convert(cid: str):
        prior = np.zeros((2, 4, 4, 4), dtype=dtype)
        prior[0, 0, 0, 0] = 1.0
        meta = {
            "source": "yuan",
            "configuration": "3d_fullres",
            "plans_sha256": PLANS,
            "input_sha256": {"zonal": _hex(f"z-{cid}"), "lesion": _hex(f"l-{cid}"), "properties": _hex(f"p-{cid}")},
            "oracle": {"exact_match": True, "shape_match": True, "mismatch_voxels": 0},
        }
        return prior, meta

    return convert


def _store_for(out_root):
    return _bare_store(out_root)


def _fail_nth_replace(monkeypatch, nth: int):
    """让第 nth 次 os.replace 抛错（其余正常），用于故障注入。"""
    import zonal_reliability_fusion.data.zonal_prior as zp

    real_replace = zp.os.replace
    counter = {"n": 0}

    def _patched(src, dst):
        counter["n"] += 1
        if counter["n"] == nth:
            raise OSError(f"simulated replace failure #{nth}")
        return real_replace(src, dst)

    monkeypatch.setattr(zp.os, "replace", _patched)
    return counter


# --- R1：首轮两病例物化后 manifest 两个 hash 非空，store 可立即加载
def test_r1_first_run_manifest_is_immediately_loadable(tmp_path):
    out_root = tmp_path / "r1"
    summary = _materializer(out_root).run(
        ["case_a", "case_b"], _convert_ok("a"), expected_case_ids=["case_a", "case_b"], progress=False
    )
    assert summary["manifest_published"] is True and summary["n_failed"] == 0
    manifest = json.loads((out_root / "manifest.json").read_text())
    assert is_sha256_hex(manifest["manifest_sha256"])
    for record in manifest["cases"]:
        assert is_sha256_hex(record["output_sha256"]), "output_sha256 不得为空"
        assert is_sha256_hex(record["metadata_sha256"]), "metadata_sha256 不得为空"
    loaded = _store_for(out_root).load_zonal_prior_manifest(["case_a", "case_b"])
    assert loaded["manifest_sha256"] == manifest["manifest_sha256"]


# --- R2：metadata_sha256 == 实际落盘 JSON 的 canonical hash
def test_r2_metadata_sha256_matches_on_disk_json(tmp_path):
    out_root = tmp_path / "r2"
    materializer = _materializer(out_root)
    materializer.run(["case_a"], _convert_ok("a"), expected_case_ids=["case_a"], progress=False)
    on_disk = json.loads((out_root / "case_a.json").read_text())
    manifest = json.loads((out_root / "manifest.json").read_text())
    assert manifest["cases"][0]["metadata_sha256"] == metadata_sha256_of(on_disk)
    assert manifest["cases"][0]["output_sha256"] == on_disk["output_sha256"]
    # 篡改文件内容 → manifest 校验必须失败
    on_disk["label_mapping"] = {"0": "hacked"}
    (out_root / "case_a.json").write_text(json.dumps(on_disk))
    with pytest.raises(PreprocessedStoreError, match="metadata_sha256"):
        _store_for(out_root).load_zonal_prior_manifest(["case_a"])


# --- R3：第二次 os.replace 失败 → 新建不留残缺对；覆盖时旧对仍可读
def test_r3_second_replace_failure_leaves_no_broken_pair(tmp_path, monkeypatch):
    out_root = tmp_path / "r3"
    # 新建：commit 调序 = npz(1)、json(2) → 第 2 次失败
    _fail_nth_replace(monkeypatch, 2)
    with pytest.raises(SidecarCommitError, match="simulated replace failure") as excinfo_new:
        write_prior_sidecar(out_root, "case_new", _valid_prior(), _metadata(_valid_prior()))
    assert isinstance(excinfo_new.value.original, OSError)
    assert not (out_root / "case_new.npz").exists(), "不应留下孤立 NPZ"
    assert not (out_root / "case_new.json").exists()
    assert _residue(out_root) == []
    monkeypatch.undo()

    # 覆盖：先备份(1,2)、再提交 npz(3)、json(4) → 第 4 次失败，旧对必须完整可读
    write_prior_sidecar(out_root, "case_old", _valid_prior(), _metadata(_valid_prior()))
    old_meta = json.loads((out_root / "case_old.json").read_text())
    old_array = np.load(out_root / "case_old.npz")["pz_tz"]
    new_prior = _valid_prior()
    new_prior[0, 1, 1, 1] = 1.0
    _fail_nth_replace(monkeypatch, 4)
    with pytest.raises(SidecarCommitError, match="simulated replace failure"):
        write_prior_sidecar(out_root, "case_old", new_prior, _metadata(new_prior), overwrite=True)
    monkeypatch.undo()
    assert _residue(out_root) == []
    assert json.loads((out_root / "case_old.json").read_text()) == old_meta
    assert np.array_equal(np.load(out_root / "case_old.npz")["pz_tz"], old_array)
    ok, why = sidecar_identity_ok(
        out_root,
        "case_old",
        {"source": "yuan", "configuration": "3d_fullres", "plans_sha256": PLANS,
         "input_sha256": old_meta["input_sha256"]},
    )
    assert ok, why


# --- R4：一例成功一例失败 → 不发布 manifest
def test_r4_partial_failure_publishes_no_manifest(tmp_path):
    out_root = tmp_path / "r4"

    def convert(cid: str):
        if cid == "case_bad":
            raise ZonalPriorError("oracle rejection: simulated")
        return _convert_ok(cid)(cid)

    summary = _materializer(out_root).run(
        ["case_ok", "case_bad"], convert, expected_case_ids=["case_ok", "case_bad"], progress=False
    )
    assert summary["n_ok"] == 1 and summary["n_failed"] == 1
    assert summary["manifest_published"] is False
    assert "n_failed" in summary["manifest_not_published_reason"]
    assert not (out_root / "manifest.json").exists()


# --- R5：子集运行不创建/覆盖 canonical manifest
def test_r5_subset_run_never_touches_canonical_manifest(tmp_path):
    out_root = tmp_path / "r5"
    _materializer(out_root).run(
        ["case_a", "case_b"], _convert_ok("a"), expected_case_ids=["case_a", "case_b"], progress=False
    )
    manifest_path = out_root / "manifest.json"
    original = manifest_path.read_bytes()
    subset = _materializer(out_root).run(["case_a"], _convert_ok("a"), progress=False)  # 无 expected → 子集
    assert subset["manifest_published"] is False
    assert subset["manifest_not_published_reason"] == "expected_case_ids_not_provided"
    assert manifest_path.read_bytes() == original, "子集运行不得覆盖既有 canonical manifest"
    # 空目录的子集运行也不得创建 manifest
    empty_root = tmp_path / "r5_empty"
    _materializer(empty_root).run(["case_a"], _convert_ok("a"), progress=False)
    assert not (empty_root / "manifest.json").exists()
    # dry-run 同样不得创建
    dry_root = tmp_path / "r5_dry"
    _materializer(dry_root).run(
        ["case_a", "case_b"], _convert_ok("a"), expected_case_ids=["case_a", "case_b"],
        dry_run=True, progress=False,
    )
    assert not (dry_root / "manifest.json").exists()


# --- R6：全量 resume 混合 skipped/new → manifest 覆盖完整集合
def test_r6_full_resume_rebuilds_manifest_from_disk(tmp_path):
    out_root = tmp_path / "r6"
    _materializer(out_root).run(
        ["case_a", "case_b"], _convert_ok("a"), expected_case_ids=["case_a", "case_b"], progress=False
    )

    def probe(case_id: str):
        meta = json.loads((out_root / f"{case_id}.json").read_text())
        return {
            "source": "yuan",
            "configuration": "3d_fullres",
            "plans_sha256": PLANS,
            "input_sha256": meta["input_sha256"],
        }

    calls = {"n": 0}

    def convert(case_id: str):
        calls["n"] += 1
        return _convert_ok(case_id)(case_id)

    summary = _materializer(out_root, resume=True).run(
        ["case_a", "case_b", "case_c"],
        convert,
        identity_probe=probe,
        expected_case_ids=["case_a", "case_b", "case_c"],
        progress=False,
    )
    assert calls["n"] == 1, "已存在的 sidecar 不应重新转换"
    assert summary["n_skipped"] == 2 and summary["n_ok"] == 1
    assert summary["manifest_published"] is True
    manifest = json.loads((out_root / "manifest.json").read_text())
    assert manifest["case_ids"] == ["case_a", "case_b", "case_c"]
    assert all(is_sha256_hex(r["metadata_sha256"]) for r in manifest["cases"])
    _store_for(out_root).load_zonal_prior_manifest(["case_a", "case_b", "case_c"])


# --- R7：伪造 metadata_sha256 / 空 output_sha256 / 缺 NPZ / case_ids 不一致 全部拒绝
def test_r7_manifest_tampering_is_rejected(tmp_path):
    out_root = tmp_path / "r7"
    _materializer(out_root).run(
        ["case_a", "case_b"], _convert_ok("a"), expected_case_ids=["case_a", "case_b"], progress=False
    )
    manifest_path = out_root / "manifest.json"
    good = json.loads(manifest_path.read_text())

    forged = json.loads(json.dumps(good))
    forged["cases"][0]["metadata_sha256"] = _hex("forged")
    # 伪造后必须先过 manifest 自校验 → 重算 hash 模拟「攻击者同步改了自校验值」
    forged["manifest_sha256"] = zonal_prior_json_sha256(
        {k: v for k, v in forged.items() if k != "manifest_sha256"}
    )
    manifest_path.write_text(json.dumps(forged))
    with pytest.raises(PreprocessedStoreError, match="metadata_sha256"):
        _store_for(out_root).load_zonal_prior_manifest(["case_a", "case_b"])

    empty = json.loads(json.dumps(good))
    empty["cases"][0]["output_sha256"] = ""
    empty["manifest_sha256"] = zonal_prior_json_sha256(
        {k: v for k, v in empty.items() if k != "manifest_sha256"}
    )
    manifest_path.write_text(json.dumps(empty))
    with pytest.raises((PreprocessedStoreError, ZonalPriorError), match="output_sha256|hex"):
        _store_for(out_root).load_zonal_prior_manifest(["case_a", "case_b"])

    manifest_path.write_text(json.dumps(good))
    (out_root / "case_b.npz").unlink()
    with pytest.raises(PreprocessedStoreError, match="数组不存在"):
        _store_for(out_root).load_zonal_prior_manifest(["case_a", "case_b"])

    (out_root / "case_b.npz").write_bytes((out_root / "case_a.npz").read_bytes())  # 恢复一个可读文件
    bad_ids = json.loads(json.dumps(good))
    bad_ids["case_ids"] = list(reversed(bad_ids["case_ids"]))
    manifest_path.write_text(json.dumps(bad_ids))
    with pytest.raises((PreprocessedStoreError, ZonalPriorError), match="case_ids|manifest_sha256"):
        _store_for(out_root).load_zonal_prior_manifest(["case_a", "case_b"])


# --- R8：float64 prior 写盘后哈希与实际 float32 数组一致
def test_r8_float64_prior_hash_matches_stored_float32(tmp_path):
    out_root = tmp_path / "r8"
    prior64 = _valid_prior().astype(np.float64)
    npz_path, meta_path, status, meta = write_prior_sidecar(out_root, "case_f64", prior64, _metadata(prior64))
    assert status == "ok"
    stored = np.load(npz_path)["pz_tz"]
    assert stored.dtype == np.float32
    assert meta["output_dtype"] == "float32"
    assert meta["output_sha256"] == array_sha256(stored.astype(np.float32))
    assert meta["output_sha256"] == array_sha256(prior64.astype(np.float32))
    assert is_sha256_hex(meta["output_sha256"])
    # 与磁盘 JSON 一致
    assert json.loads(meta_path.read_text())["output_sha256"] == meta["output_sha256"]


# --- R9：v24 判据有效（见 test_v24_config_identity_is_consistent 的负向对照）


# --- R10：setup 默认模式 prior 不就绪 → 退出码非 0
def test_r10_setup_exit_code_when_prior_not_ready(tmp_path, monkeypatch):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "validate_m0_setup_mod", PROJECT / "scripts/train/validate_m0_setup.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    cfg_doc = yaml.safe_load(
        (PROJECT / "configs/experiments/m4_conditioned_gate_picai_3d_fullres_v24.yaml").read_text()
    )
    cfg_doc["data"]["zonal_prior_root"] = str(tmp_path / "not_materialized")
    cfg_path = tmp_path / "m4_prior_missing.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg_doc, sort_keys=False))

    base_argv = ["validate_m0_setup.py", "--config", str(cfg_path), "--json-out", str(tmp_path / "out.json")]
    monkeypatch.setattr("sys.argv", base_argv)
    code_require = module.main()
    assert code_require != 0, "require-data 模式 prior 未就绪必须退出码非 0"
    payload = json.loads((tmp_path / "out.json").read_text())
    assert payload["prior_readiness"]["status"] == "PRIOR_NOT_READY"
    assert payload["prior_readiness"]["training_ready"] is False

    monkeypatch.setattr("sys.argv", [*base_argv, "--no-require-data"])
    assert module.main() == 0
    payload2 = json.loads((tmp_path / "out.json").read_text())
    assert payload2["prior_readiness"]["status"] == "PRIOR_NOT_CHECKED"
    assert payload2["prior_readiness"]["training_ready"] is False

# ===========================================================================
# 第 3 轮：提交状态机回滚 / fail-closed 写盘 / 恶意自洽 sidecar / 快照一致性 / 进度
# ===========================================================================
def _residue(out_root) -> list[str]:
    """本轮不应留下的临时文件与备份文件。"""
    out = Path(out_root)
    if not out.exists():
        return []
    return sorted(p.name for p in out.iterdir() if p.name.endswith((".tmp", ".bak")) or p.name.startswith("."))


def _snapshot_pair(out_root, case_id):
    meta = json.loads((out_root / f"{case_id}.json").read_text())
    with np.load(out_root / f"{case_id}.npz") as npz:
        arr = np.ascontiguousarray(npz["pz_tz"])
    return meta, arr


def _assert_old_pair_intact(out_root, case_id, old_meta, old_arr):
    meta, arr = _snapshot_pair(out_root, case_id)
    assert meta == old_meta, "旧元数据必须原样恢复"
    assert np.array_equal(arr, old_arr), "旧数组必须原样恢复"
    validate_sidecar(
        out_root,
        case_id,
        expected_source=old_meta["source"],
        expected_configuration=old_meta["configuration"],
        expected_plans_sha256=old_meta["plans_sha256"],
        expected_input_sha256=old_meta["input_sha256"],
        expected_metadata_sha256=metadata_sha256_of(old_meta),
        expected_output_sha256=old_meta["output_sha256"],
    )


# --- A. 覆盖既有有效对时，5 条失败路径都必须完整回滚
@pytest.mark.parametrize(
    "fail_call, match",
    [
        (1, "第 1 次备份"),
        (2, "第 2 次备份"),
        (3, "第 1 次提交"),
        (4, "第 2 次提交"),
    ],
)
def test_a_overwrite_fault_injection_restores_old_pair(tmp_path, monkeypatch, fail_call, match):
    out_root = tmp_path / f"a{fail_call}"
    old_prior = _valid_prior()
    write_prior_sidecar(out_root, "case_x", old_prior, _metadata(old_prior))
    old_meta, old_arr = _snapshot_pair(out_root, "case_x")

    new_prior = _valid_prior()
    new_prior[0, 2, 2, 2] = 1.0

    import zonal_reliability_fusion.data.zonal_prior as zp

    real_replace = zp.os.replace
    counter = {"n": 0}

    def _patched(src, dst):
        counter["n"] += 1
        if counter["n"] == fail_call:
            raise OSError(f"simulated failure at {match}")
        return real_replace(src, dst)

    monkeypatch.setattr(zp.os, "replace", _patched)
    with pytest.raises(SidecarCommitError, match=match) as excinfo:
        write_prior_sidecar(out_root, "case_x", new_prior, _metadata(new_prior), overwrite=True)
    monkeypatch.undo()
    assert isinstance(excinfo.value.original, OSError)
    assert excinfo.value.recovery_errors == [], f"回滚必须完全成功: {excinfo.value.recovery_errors}"
    _assert_old_pair_intact(out_root, "case_x", old_meta, old_arr)
    assert _residue(out_root) == []


def test_a5_post_commit_verification_failure_restores_old_pair(tmp_path, monkeypatch):
    """提交后落盘 JSON 复核失败 → 同样回滚（旧对完好、无残留）。"""
    out_root = tmp_path / "a5"
    old_prior = _valid_prior()
    write_prior_sidecar(out_root, "case_x", old_prior, _metadata(old_prior))
    old_meta, old_arr = _snapshot_pair(out_root, "case_x")

    import zonal_reliability_fusion.data.zonal_prior as zp

    real_read = zp.read_sidecar_metadata
    calls = {"n": 0}

    def _patched_read(path):
        doc = real_read(path)
        calls["n"] += 1
        if calls["n"] >= 1:  # overwrite 路径下第一次读取就是提交后的落盘复核
            doc = dict(doc)
            doc["tampered_by_test"] = True
        return doc

    monkeypatch.setattr(zp, "read_sidecar_metadata", _patched_read)
    work = _valid_prior()
    work[0, 3, 3, 3] = 1.0
    with pytest.raises(SidecarCommitError, match="落盘元数据与预期不一致") as excinfo:
        write_prior_sidecar(out_root, "case_x", work, _metadata(work), overwrite=True)
    monkeypatch.undo()
    assert excinfo.value.stage == "verify"
    assert excinfo.value.recovery_errors == []
    _assert_old_pair_intact(out_root, "case_x", old_meta, old_arr)
    assert _residue(out_root) == []


def test_a6_unique_backups_do_not_touch_historical_bak(tmp_path):
    """备份文件使用本轮唯一名称；历史 .bak 不得被删除或覆盖。"""
    out_root = tmp_path / "a6"
    prior = _valid_prior()
    write_prior_sidecar(out_root, "case_x", prior, _metadata(prior))
    legacy = out_root / ".case_x.json.legacy.bak"
    legacy.write_text("历史备份（不得被删除）")
    work = _valid_prior()
    work[0, 1, 2, 3] = 1.0
    write_prior_sidecar(out_root, "case_x", work, _metadata(work), overwrite=True)
    assert legacy.read_text() == "历史备份（不得被删除）"
    assert _residue(out_root) == [legacy.name]  # 只应剩历史文件，本轮备份已清理


# --- B. 写盘侧 fail-closed（不依赖 materializer 的上游检查）
@pytest.mark.parametrize(
    "mutate, match",
    [
        ({"source": "unknown"}, "source"),
        ({"source": None}, "source"),
        ({"configuration": ""}, "configuration"),
        ({"plans_sha256": "not-a-hash"}, "plans_sha256"),
        ({"input_sha256": {}}, "input_sha256"),
        ({"input_sha256": {"zonal": "xyz"}}, "input_sha256"),
        ({"oracle": {"exact_match": False, "shape_match": True, "mismatch_voxels": 0}}, "oracle"),
        ({"oracle": None}, "oracle"),
    ],
)
def test_b_writer_is_fail_closed_on_bad_metadata(tmp_path, mutate, match):
    out_root = tmp_path / "b"
    prior = _valid_prior()
    meta = _metadata(prior)
    meta.update(mutate)
    with pytest.raises(ZonalPriorError, match=match):
        write_prior_sidecar(out_root, "case_b", prior, meta)
    assert not (out_root / "case_b.npz").exists() and not (out_root / "case_b.json").exists()
    assert _residue(out_root) == []


@pytest.mark.parametrize(
    "bad_prior, match",
    [
        ({"nan": True}, "非有限值"),
        ({"value": 2.0}, r"\[0,1\]"),
        ({"overlap": True}, "重叠"),
    ],
)
def test_b_writer_rejects_invalid_array(tmp_path, bad_prior, match):
    out_root = tmp_path / "b_arr"
    arr = _valid_prior()
    if bad_prior.get("nan"):
        arr[0, 0, 0, 0] = np.nan
    if "value" in bad_prior:
        arr[0, 0, 0, 0] = bad_prior["value"]
    if bad_prior.get("overlap"):
        arr[0, 0, 0, 0] = 1.0
        arr[1, 0, 0, 0] = 1.0
    with pytest.raises(ZonalPriorError, match=match):
        write_prior_sidecar(out_root, "case_b", arr, _metadata(arr))
    assert not (out_root / "case_b.npz").exists()


def test_b_writer_rejects_wrong_channel_count(tmp_path):
    out_root = tmp_path / "b_ch"
    arr = np.zeros((3, 4, 4, 4), dtype=np.float32)
    arr[2, 0, 0, 0] = 1.0  # WG 通道：必须拒绝
    with pytest.raises(ZonalPriorError, match="形状"):
        write_prior_sidecar(out_root, "case_b", arr, _metadata(arr))


@pytest.mark.parametrize(
    "breakage, match",
    [
        ("array", "SHA256"),
        ("configuration", "configuration"),
        ("channel_semantics", "channel_semantics"),
        ("oracle", "oracle"),
    ],
)
def test_b_resume_never_skips_broken_sidecar(tmp_path, breakage, match):
    """resume 的弱比较已移除：损坏/不匹配的既有 sidecar 不得返回 skipped。"""
    out_root = tmp_path / f"b_resume_{breakage}"
    prior = _valid_prior()
    write_prior_sidecar(out_root, "case_r", prior, _metadata(prior))
    meta_path = out_root / "case_r.json"
    meta = json.loads(meta_path.read_text())
    if breakage == "array":
        tampered = prior.copy()
        tampered[0, 1, 1, 1] = 1.0
        with np.load(out_root / "case_r.npz") as npz:
            stored = dict(npz)
        stored["pz_tz"] = tampered
        np.savez_compressed(out_root / "case_r.npz", **stored)
    elif breakage == "configuration":
        meta["configuration"] = "2d"
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    elif breakage == "channel_semantics":
        meta["output_channel_semantics"] = ["WG"]
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    else:
        meta["oracle"] = {"exact_match": False, "shape_match": True, "mismatch_voxels": 3}
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    status = write_prior_sidecar(out_root, "case_r", prior, _metadata(prior), resume=True)[2]
    assert status != "skipped", f"{breakage} 损坏的 sidecar 不得返回 skipped（必须重写修好）"
    assert status == "ok"
    validate_sidecar(
        out_root,
        "case_r",
        expected_source="yuan",
        expected_configuration="3d_fullres",
        expected_plans_sha256="a" * 64,
        expected_input_sha256=_metadata(prior)["input_sha256"],
    )


# --- C. 恶意但「哈希自洽」的 sidecar 必须在 PRIOR_READY 之前被拒绝
def _forge(out_root, case_id, *, meta_mutate=None, array_mutate=None):
    """写出一份哈希自洽（metadata/output/manifest 全部同步重算）的 sidecar，并保持 manifest 合法。"""
    base = _valid_prior()
    if array_mutate is None:
        arr = base
    else:
        arr = base.copy()
        mutated = array_mutate(arr)  # 就地修改返回 None 也支持
        if mutated is not None:
            arr = mutated
    arr = np.ascontiguousarray(arr)
    np.savez_compressed(out_root / f"{case_id}.npz", pz_tz=arr)
    meta = json.loads((out_root / f"{case_id}.json").read_text())
    meta["output_shape"] = list(arr.shape)
    meta["output_sha256"] = array_sha256(arr)
    if meta_mutate is not None:
        if callable(meta_mutate):
            meta_mutate(meta)
        else:
            meta.update(meta_mutate)
    (out_root / f"{case_id}.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    man_path = out_root / "manifest.json"
    man = json.loads(man_path.read_text())
    for rec in man["cases"]:
        if rec["case_id"] == case_id:
            rec["output_sha256"] = meta["output_sha256"]
            rec["metadata_sha256"] = metadata_sha256_of(meta)
    man["manifest_sha256"] = zonal_prior_json_sha256({k: v for k, v in man.items() if k != "manifest_sha256"})
    man_path.write_text(json.dumps(man, indent=2, sort_keys=True) + "\n")
    # 先证明「manifest 层面自洽」——攻击者已同步重算所有哈希
    verify_manifest(
        man,
        [rec["case_id"] for rec in man["cases"]],
        source="yuan",
        plans_sha256="a" * 64,
        configuration="3d_fullres",
    )
    return man


def _force_overlap(arr):
    """制造 PZ/TZ 重叠（两通道在同体素都为 1）。"""
    arr[0, 0, 0, 0] = 1.0
    arr[1, 0, 0, 0] = 1.0


FORGERY_CASES = [
    ("source", {"source": "hevi"}, None, "来源"),
    ("configuration", {"configuration": "2d"}, None, "configuration"),
    ("plans", {"plans_sha256": "b" * 64}, None, "plans_sha256"),
    ("case_id", {"case_id": "other_case"}, None, "case_id"),
    ("output_shape", {"output_shape": [2, 4, 4, 5]}, None, "output_shape"),
    ("output_dtype", {"output_dtype": "float16"}, None, "output_dtype"),
    ("channel_semantics", {"output_channel_semantics": ["WG"]}, None, "channel_semantics"),
    ("oracle", {"oracle": {"exact_match": False, "shape_match": True, "mismatch_voxels": 1}}, None, "oracle"),
    ("nan", None, lambda a: a.__setitem__((0, 0, 0, 0), np.nan), "非有限值"),
    ("out_of_range", None, lambda a: a.__setitem__((0, 0, 0, 0), 2.0), r"\[0,1\]"),
    ("overlap", None, _force_overlap, "重叠"),
]


@pytest.mark.parametrize("name, meta_mutate, array_mutate, match", FORGERY_CASES)
def test_c_malicious_but_self_consistent_sidecar_is_rejected(
    tmp_path, name, meta_mutate, array_mutate, match
):
    out_root = tmp_path / f"forge_{name}"
    _materializer(out_root).run(["case_a"], _convert_ok("a"), expected_case_ids=["case_a"], progress=False)
    _forge(out_root, "case_a", meta_mutate=meta_mutate, array_mutate=array_mutate)
    store = _store_for(out_root)
    with pytest.raises(PreprocessedStoreError, match=match):
        store.load_zonal_prior_manifest(["case_a"])
    assert store.zonal_prior_manifest_sha256 is None, "拒绝后不得留下 PRIOR_READY 状态"
    with pytest.raises(PreprocessedStoreError, match=match):
        store.load_zonal_prior("case_a", seg_shape=(4, 4, 4))


# --- D. 训练快照一致性（train_pipeline.zonal_prior ⟷ provenance）
def _load_train_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "train_m0_mod", PROJECT / "scripts/train/train_m0.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _snapshot(pipeline_sha, checked, *, provenance_sha=None, enabled=True):
    """构造合成快照；`provenance_sha` 缺省与 pipeline 相同（用于制造不一致）。"""
    if provenance_sha is None:
        provenance_sha = pipeline_sha
    return {
        "train_pipeline": {
            "zonal_prior": {"enabled": enabled, "manifest_sha256": pipeline_sha, "array_hash_checked": checked}
        },
        "val_pipeline": {
            "zonal_prior": {"enabled": enabled, "manifest_sha256": pipeline_sha, "array_hash_checked": checked}
        },
        "provenance": {"zonal_prior_manifest_sha256": provenance_sha},
    }


def test_d_train_snapshot_consistency():
    mod = _load_train_module()
    sha = _hex("manifest")
    mod.assert_prior_snapshot_consistency(_snapshot(sha, True))  # 一致 → 通过

    with pytest.raises(SystemExit, match="manifest_sha256"):
        mod.assert_prior_snapshot_consistency(_snapshot(_hex("other"), True, provenance_sha=sha))
    with pytest.raises(SystemExit, match="array_hash_checked"):
        mod.assert_prior_snapshot_consistency(_snapshot(sha, False))
    # M0–M2：prior 未启用 → 不做该约束
    mod.assert_prior_snapshot_consistency(_snapshot(None, False, enabled=False))


def test_d_manifest_validation_runs_before_pipeline_snapshot():
    """build_trainer 必须先完成 manifest 深度校验，再生成 pipeline 快照（顺序回归）。"""
    import inspect

    src = inspect.getsource(_load_train_module().build_trainer)
    assert src.index("load_prior_manifest_sha256(") < src.index("train_provider.describe()"), (
        "manifest 深度校验必须早于 train_provider.describe() 快照"
    )
    assert "assert_prior_snapshot_consistency(config_snapshot)" in src


def test_d_train_passes_progress_flag_to_manifest_load():
    mod = _load_train_module()

    class _Cfg:
        class experiment:
            model_id = "M4"

    class _Store:
        train_ids = ("c1",)
        val_ids = ("c2",)

        def __init__(self):
            self.kwargs = None

        def load_zonal_prior_manifest(self, ids, **kwargs):
            self.kwargs = kwargs
            return {"manifest_sha256": "c" * 64}

    store = _Store()
    assert mod.load_prior_manifest_sha256(_Cfg, store, progress=False) == "c" * 64
    assert store.kwargs["progress"] is False
    assert "prior sidecar 校验" in store.kwargs["desc"]
    # M0–M2 不触发 prior 扫描
    class _Cfg0:
        class experiment:
            model_id = "M0"

    assert mod.load_prior_manifest_sha256(_Cfg0, _Store(), progress=True) is None


def test_d_loader_smoke_passes_progress_and_reports_stats():
    import inspect

    src = inspect.getsource(_load_train_module().run_loader_smoke)
    assert "progress=progress" in src and "array_hash_checked" in src


# --- E. 全量校验的进度与统计（脚本层控制，库默认静默）
def test_e_manifest_validation_progress_and_stats(tmp_path, monkeypatch):
    import zonal_reliability_fusion.data.preprocessed_store as store_mod
    from zonal_reliability_fusion.data import zonal_prior as zp

    out_root = tmp_path / "e"
    _materializer(out_root).run(
        ["case_a", "case_b"], _convert_ok("a"), expected_case_ids=["case_a", "case_b"], progress=False
    )

    bars = []

    class _FakeTqdm:
        def __init__(self, iterable, **kwargs):
            bars.append(kwargs)
            self._items = list(iterable)

        def __iter__(self):
            return iter(self._items)

        def __len__(self):
            return len(self._items)

    monkeypatch.setattr(store_mod, "tqdm", _FakeTqdm)

    # 静默路径：默认 progress=False，不创建进度条，但统计完整
    quiet = _store_for(out_root)
    quiet.load_zonal_prior_manifest(["case_a", "case_b"])
    assert bars == [], "库函数默认不得创建进度条"
    stats = quiet.zonal_prior_validation_stats
    assert stats["n_cases"] == 2 and stats["n_ok"] == 2 and stats["n_failed"] == 0
    assert stats["array_hash_checked"] is True
    assert stats["elapsed_sec"] >= 0 and is_sha256_hex(stats["manifest_sha256"])

    # 进度路径：显式 progress=True → 创建进度条（含 total），并逐个真正校验
    calls = {"n": 0}
    real_validate = zp.validate_sidecar

    def _counting_validate(*args, **kwargs):
        calls["n"] += 1
        return real_validate(*args, **kwargs)

    monkeypatch.setattr(zp, "validate_sidecar", _counting_validate)
    loud = _store_for(out_root)
    loud.load_zonal_prior_manifest(["case_a", "case_b"], progress=True, desc="单元测试校验")
    assert calls["n"] == 2, "校验器必须逐例调用"
    assert len(bars) == 1 and bars[0]["total"] == 2 and bars[0]["desc"] == "单元测试校验"

    # 已移除跨调用缓存：同进程内再次调用必须重新逐例校验（不依赖弱文件指纹）
    calls["n"] = 0
    again = loud.load_zonal_prior_manifest(["case_a", "case_b"])
    assert calls["n"] == 2, "必须重新扫描，不得复用旧结果"
    assert again["manifest_sha256"] == stats["manifest_sha256"]
    assert "cached" not in loud.zonal_prior_validation_stats


def test_e_failed_validation_reports_counts_and_never_ready(tmp_path):
    out_root = tmp_path / "e2"
    _materializer(out_root).run(
        ["case_a", "case_b"], _convert_ok("a"), expected_case_ids=["case_a", "case_b"], progress=False
    )
    _forge(out_root, "case_b", meta_mutate=lambda m: m.update({"configuration": "2d"}))
    store = _store_for(out_root)
    with pytest.raises(PreprocessedStoreError, match="深度校验失败"):
        store.load_zonal_prior_manifest(["case_a", "case_b"])
    assert store.zonal_prior_validation_stats["n_ok"] == 1
    assert store.zonal_prior_validation_stats["n_failed"] == 1
    assert store.zonal_prior_manifest_sha256 is None
    assert store.zonal_prior_array_hash_checked is False


def test_e_setup_cli_has_no_progress_flag():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "validate_m0_setup_mod", PROJECT / "scripts/train/validate_m0_setup.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    import sys as _sys

    old = _sys.argv
    try:
        _sys.argv = ["validate_m0_setup.py", "--no-progress"]
        assert module.parse_args().no_progress is True
        _sys.argv = ["validate_m0_setup.py"]
        assert module.parse_args().no_progress is False
    finally:
        _sys.argv = old

def test_a7_unrecoverable_rollback_raises_with_both_errors(tmp_path, monkeypatch):
    """回滚本身失败时：必须抛同时包含原始错误与恢复错误的 SidecarCommitError（绝不静默）。"""
    out_root = tmp_path / "a7"
    old_prior = _valid_prior()
    write_prior_sidecar(out_root, "case_x", old_prior, _metadata(old_prior))

    import zonal_reliability_fusion.data.zonal_prior as zp

    real_replace = zp.os.replace
    counter = {"n": 0}

    def _patched(src, dst):
        counter["n"] += 1
        if counter["n"] == 4:  # JSON 提交失败
            raise OSError("simulated commit failure")
        if counter["n"] >= 5:  # 之后的恢复 replace 全部失败
            raise OSError("simulated rollback failure")
        return real_replace(src, dst)

    monkeypatch.setattr(zp.os, "replace", _patched)
    work = _valid_prior()
    work[0, 2, 2, 2] = 1.0
    with pytest.raises(SidecarCommitError, match="回滚未能完全完成") as excinfo:
        write_prior_sidecar(out_root, "case_x", work, _metadata(work), overwrite=True)
    monkeypatch.undo()
    message = str(excinfo.value)
    assert "simulated commit failure" in message, "原始错误必须在异常中"
    assert "simulated rollback failure" in message, "恢复错误必须在异常中"
    assert excinfo.value.recovery_errors, "恢复错误列表不得为空"
    assert excinfo.value.stage == "commit"
    # 备份仍在，可人工恢复（不静默删除）
    backups = [p for p in out_root.iterdir() if p.name.endswith(".bak")]
    assert backups, "回滚失败时备份必须保留，便于人工恢复"


# ===========================================================================
# 第 4 轮验收：惰性进度条 / 真实 dtype / 冻结 manifest 记录 / READY 复位 / 无弱缓存
# ===========================================================================
def test_r11_progress_events_interleave_with_validation(tmp_path, monkeypatch):
    """两例：事件顺序必须是 progress/a → validate/a → progress/b → validate/b（惰性迭代）。"""
    import zonal_reliability_fusion.data.preprocessed_store as store_mod
    from zonal_reliability_fusion.data import zonal_prior as zp

    out_root = tmp_path / "r11"
    _materializer(out_root).run(
        ["case_a", "case_b"], _convert_ok("a"), expected_case_ids=["case_a", "case_b"], progress=False
    )

    events: list[str] = []

    class _TracingTqdm:
        def __init__(self, iterable, **kwargs):
            self._items = list(iterable)
            self.total = kwargs.get("total")

        def __iter__(self):
            for item in self._items:
                events.append(f"progress/{item['case_id']}")
                yield item

        def __len__(self):
            return len(self._items)

    real_validate = zp.validate_sidecar

    def _tracing_validate(root_arg, case_id, **kwargs):
        events.append(f"validate/{case_id}")
        return real_validate(root_arg, case_id, **kwargs)

    monkeypatch.setattr(store_mod, "tqdm", _TracingTqdm)
    monkeypatch.setattr(zp, "validate_sidecar", _tracing_validate)
    _store_for(out_root).load_zonal_prior_manifest(["case_a", "case_b"], progress=True)

    assert events == [
        "progress/case_a",
        "validate/case_a",
        "progress/case_b",
        "validate/case_b",
    ], f"进度与校验必须交错（当前事件序列：{events}）"


def test_r11b_progress_bar_total_matches_case_count(tmp_path, monkeypatch):
    """进度条必须携带 total（比例/ETA 才有意义）。"""
    import zonal_reliability_fusion.data.preprocessed_store as store_mod

    out_root = tmp_path / "r11b"
    _materializer(out_root).run(
        ["case_a", "case_b", "case_c"], _convert_ok("a"),
        expected_case_ids=["case_a", "case_b", "case_c"], progress=False,
    )
    seen = {}

    class _TracingTqdm:
        def __init__(self, iterable, **kwargs):
            seen.update(kwargs)
            self._items = list(iterable)

        def __iter__(self):
            return iter(self._items)

        def __len__(self):
            return len(self._items)

    monkeypatch.setattr(store_mod, "tqdm", _TracingTqdm)
    _store_for(out_root).load_zonal_prior_manifest(
        ["case_a", "case_b", "case_c"], progress=True, desc="三例"
    )
    assert seen["total"] == 3 and seen["desc"] == "三例"


def test_r12_float16_npz_is_rejected_even_when_hashes_are_self_consistent(tmp_path):
    """float16 存储 + metadata 谎报 float32 + 全部哈希同步重算 → 仍必须被拒绝。"""
    out_root = tmp_path / "r12"
    _materializer(out_root).run(["case_a"], _convert_ok("a"), expected_case_ids=["case_a"], progress=False)

    arr16 = _valid_prior().astype(np.float16)
    np.savez_compressed(out_root / "case_a.npz", pz_tz=arr16)
    meta = json.loads((out_root / "case_a.json").read_text())
    meta["output_dtype"] = "float32"  # 谎报
    meta["output_shape"] = list(arr16.shape)
    meta["output_sha256"] = array_sha256(arr16)
    (out_root / "case_a.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")

    man_path = out_root / "manifest.json"
    man = json.loads(man_path.read_text())
    man["cases"][0]["output_sha256"] = meta["output_sha256"]
    man["cases"][0]["metadata_sha256"] = metadata_sha256_of(meta)
    man["manifest_sha256"] = zonal_prior_json_sha256({k: v for k, v in man.items() if k != "manifest_sha256"})
    man_path.write_text(json.dumps(man, indent=2, sort_keys=True) + "\n")

    # 攻击者版本在 manifest 层完全自洽
    verify_manifest(man, ["case_a"], source="yuan", plans_sha256="a" * 64, configuration="3d_fullres")
    with pytest.raises(PreprocessedStoreError, match="dtype"):
        _store_for(out_root).load_zonal_prior_manifest(["case_a"])
    with pytest.raises(PreprocessedStoreError, match="dtype"):
        _store_for(out_root).load_zonal_prior("case_a", seg_shape=(4, 4, 4))


def test_r13_frozen_manifest_blocks_swapped_but_valid_prior(tmp_path):
    """深度预检后替换成另一份合法 prior（sidecar 自洽）→ 训练逐例读取必须被冻结记录拒绝。"""
    out_root = tmp_path / "r13"
    _materializer(out_root).run(
        ["case_a", "case_b"], _convert_ok("a"), expected_case_ids=["case_a", "case_b"], progress=False
    )
    store = _store_for(out_root)
    store.load_zonal_prior_manifest(["case_a", "case_b"])
    assert len(store.zonal_prior_frozen_records) == 2

    # 另一份仍满足 finite/[0,1]/PZ+TZ<=1 的数组；同步更新该例 JSON 的 output_sha256
    swapped = _valid_prior()
    swapped[0, 3, 3, 3] = 1.0
    np.savez_compressed(out_root / "case_a.npz", pz_tz=swapped)
    meta = json.loads((out_root / "case_a.json").read_text())
    meta["output_sha256"] = array_sha256(swapped)
    (out_root / "case_a.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")

    # 1) sidecar 自身完全自洽（不带冻结期望时通过）
    validate_sidecar(
        out_root,
        "case_a",
        expected_source="yuan",
        expected_configuration="3d_fullres",
        expected_plans_sha256="a" * 64,
    )
    # 2) 但已冻结的 manifest 记录不允许它通过
    with pytest.raises(PreprocessedStoreError, match="manifest|不一致"):
        store.load_zonal_prior("case_a", seg_shape=(4, 4, 4))
    # 3) 未受影响的病例仍可正常读取
    assert store.load_zonal_prior("case_b", seg_shape=(4, 4, 4)).shape == (2, 4, 4, 4)


def test_r13b_training_requires_frozen_manifest_before_reading_prior(tmp_path):
    """正式 M3/M4：未成功冻结 manifest 记录前，逐例读取必须失败。"""
    out_root = tmp_path / "r13b"
    _materializer(out_root).run(["case_a"], _convert_ok("a"), expected_case_ids=["case_a"], progress=False)
    store = _store_for(out_root)
    store.require_frozen_manifest = True
    with pytest.raises(PreprocessedStoreError, match="尚未成功冻结"):
        store.load_zonal_prior("case_a", seg_shape=(4, 4, 4))
    store.load_zonal_prior_manifest(["case_a"])
    assert store.load_zonal_prior("case_a", seg_shape=(4, 4, 4)).shape == (2, 4, 4, 4)


def test_r14_failure_clears_ready_state(tmp_path):
    """先成功（READY），再制造失败：READY 状态与冻结记录必须被清空。"""
    out_root = tmp_path / "r14"
    _materializer(out_root).run(
        ["case_a", "case_b"], _convert_ok("a"), expected_case_ids=["case_a", "case_b"], progress=False
    )
    store = _store_for(out_root)
    store.load_zonal_prior_manifest(["case_a", "case_b"])
    assert store.zonal_prior_manifest_sha256 is not None
    assert store.zonal_prior_array_hash_checked is True
    assert len(store.zonal_prior_frozen_records) == 2

    meta_b = out_root / "case_b.json"
    doc = json.loads(meta_b.read_text())
    doc["configuration"] = "2d"
    meta_b.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")

    with pytest.raises(PreprocessedStoreError, match="深度校验失败"):
        store.load_zonal_prior_manifest(["case_a", "case_b"])
    assert store.zonal_prior_manifest_sha256 is None
    assert store.zonal_prior_array_hash_checked is False
    assert dict(store.zonal_prior_frozen_records) == {}
    # 失败后逐例读取同样不得通过（无冻结记录可用，且 sidecar 已损坏）
    with pytest.raises(PreprocessedStoreError):
        store.load_zonal_prior("case_b", seg_shape=(4, 4, 4))


def test_r15_same_mtime_and_size_edit_is_not_reused(tmp_path):
    """同尺寸改写 + 恢复 mtime_ns/size：不得复用任何旧结果（缓存已删除）。"""
    out_root = tmp_path / "r15"
    _materializer(out_root).run(["case_a"], _convert_ok("a"), expected_case_ids=["case_a"], progress=False)
    store = _store_for(out_root)
    store.load_zonal_prior_manifest(["case_a"])

    json_path = out_root / "case_a.json"
    text = json_path.read_text()
    stat = json_path.stat()
    old_hash = json.loads(text)["output_sha256"]
    new_hash = ("b" if old_hash[0] != "b" else "c") + old_hash[1:]
    tampered = text.replace(old_hash, new_hash, 1)
    assert len(tampered) == len(text), "必须保持同尺寸"
    json_path.write_text(tampered)
    os.utime(json_path, ns=(stat.st_atime_ns, stat.st_mtime_ns))

    after = json_path.stat()
    assert after.st_size == stat.st_size and after.st_mtime_ns == stat.st_mtime_ns, "指纹必须保持不变"
    with pytest.raises(PreprocessedStoreError, match="metadata_sha256|output_sha256"):
        store.load_zonal_prior_manifest(["case_a"])


def test_r16_all_finite_free_of_numpy_tensor_bridge_warning():
    """batch prior 检查不得对 Torch tensor 调用 np.isfinite（不再产生 __array_wrap__ 警告）。"""
    import inspect
    import warnings

    mod = _load_train_module()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert mod._all_finite(torch.ones(2, 3)) is True
        assert mod._all_finite(torch.tensor([1.0, float("nan")])) is False
        assert mod._all_finite(np.array([0.0, 1.0]), torch.ones(2)) is True
    bridge = [w for w in caught if "array_wrap" in str(w.message)]
    assert bridge == [], f"不得触发 numpy↔torch 桥接警告: {[str(w.message) for w in bridge]}"

    src = inspect.getsource(mod.run_loader_smoke)
    assert "np.isfinite(prior)" not in src and "np.isfinite(data)" not in src
    assert src.count("_all_finite(") >= 2, "两处 finite 检查都必须走 _all_finite"


def test_r17_dry_run_with_progress_steps_per_case_and_writes_nothing(tmp_path, monkeypatch):
    """dry_run=True + progress=True：tqdm 逐例推进、converter 被调用、零文件写入。"""
    import tqdm as tqdm_module

    out_root = tmp_path / "r17"
    events: list[str] = []
    converted: list[str] = []

    class _TracingTqdm:
        def __init__(self, iterable, **kwargs):
            self._items = list(iterable)
            events.append(f"tqdm(n={len(self._items)},desc={kwargs.get('desc')})")

        def __iter__(self):
            for item in self._items:
                events.append(f"progress/{item}")
                yield item

        def __len__(self):
            return len(self._items)

    def convert(case_id: str):
        converted.append(case_id)
        events.append(f"convert/{case_id}")  # 与进度事件记录在同一序列以证明交错
        return _convert_ok(case_id)(case_id)

    # run() 内部是 `from tqdm import tqdm`（调用时查找）→ patch 真实模块属性即可生效
    monkeypatch.setattr(tqdm_module, "tqdm", _TracingTqdm)
    summary = _materializer(out_root).run(
        ["case_a", "case_b"],
        convert,
        expected_case_ids=["case_a", "case_b"],
        dry_run=True,
        progress=True,
    )

    assert events == [
        "tqdm(n=2,desc=zonal prior)",
        "progress/case_a",
        "convert/case_a",
        "progress/case_b",
        "convert/case_b",
    ], f"dry-run 的进度必须与 converter 逐例交错（当前事件：{events}）"
    assert converted == ["case_a", "case_b"], "dry-run 仍必须逐例执行转换"
    assert summary["dry_run"] is True and summary["n_ok"] == 2 and summary["n_failed"] == 0
    assert summary["manifest_published"] is False

    # 零写入：目录不存在或为空；npz/json/manifest/summary 一律没有
    assert not out_root.exists() or not any(out_root.iterdir()), "dry-run 不得写入任何文件"
    for name in ("case_a.npz", "case_a.json", "case_b.npz", "case_b.json",
                 "manifest.json", "materialization_summary.json"):
        assert not (out_root / name).exists(), f"dry-run 不得产生 {name}"


def test_r17b_dry_run_progress_can_be_disabled(tmp_path, monkeypatch):
    """progress=False 时 dry-run 不创建进度条（保持脚本层可控）。"""
    import tqdm as tqdm_module

    out_root = tmp_path / "r17b"
    created = {"n": 0}

    class _TracingTqdm:
        def __init__(self, iterable, **kwargs):
            created["n"] += 1
            self._items = list(iterable)

        def __iter__(self):
            return iter(self._items)

        def __len__(self):
            return len(self._items)

    monkeypatch.setattr(tqdm_module, "tqdm", _TracingTqdm)
    summary = _materializer(out_root).run(
        ["case_a"], _convert_ok("a"), expected_case_ids=["case_a"], dry_run=True, progress=False
    )
    assert created["n"] == 0 and summary["n_ok"] == 1
    assert not out_root.exists() or not any(out_root.iterdir())


def test_r18_frozen_records_are_deeply_immutable(tmp_path):
    """冻结记录外层与嵌套 input_sha256 都必须不可变（修改触发 TypeError）。"""
    out_root = tmp_path / "r18"
    _materializer(out_root).run(["case_a"], _convert_ok("a"), expected_case_ids=["case_a"], progress=False)
    store = _store_for(out_root)
    store.load_zonal_prior_manifest(["case_a"])

    record = store.zonal_prior_frozen_records["case_a"]
    with pytest.raises(TypeError):
        record["output_sha256"] = "0" * 64  # 外层不可变
    with pytest.raises(TypeError):
        record["input_sha256"]["zonal"] = "0" * 64  # 嵌套不可变
    with pytest.raises((TypeError, AttributeError)):
        record["input_sha256"].clear()  # mappingproxy 无 clear（同样是「不支持修改」）
    with pytest.raises(TypeError):
        store.zonal_prior_frozen_records["case_b"] = record  # 顶层映射不可变

    # 记录本身仍可用于逐例校验（只读语义不受影响）
    assert store.load_zonal_prior("case_a", seg_shape=(4, 4, 4)).shape == (2, 4, 4, 4)
    assert set(record["input_sha256"]) == {"zonal", "lesion", "properties"}


def _load_materialize_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "materialize_zonal_prior_mod", PROJECT / "scripts/data/materialize_zonal_prior.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_r19_build_context_instantiates_reader_exactly_once(tmp_path, monkeypatch):
    """nnU-Net 2.6.2 的 image_reader_writer_class 是 @property：builder 必须只实例化一次。

    旧写法 `image_reader_writer_class()()` 会对 SimpleITKIO **实例**再调用一次 →
    `TypeError: 'SimpleITKIO' object is not callable`；本测试在旧代码上必然失败。
    """
    import sys
    import types as pytypes

    mod = _load_materialize_module()
    reads = {"n": 0}

    class _FakeReader:
        def read_seg(self, seg_fname):  # 与真实 SimpleITKIO.read_seg 同名
            raise AssertionError("synthetic reader 不应被真正调用")

    class _FakePlansManager:
        def __init__(self, doc):
            self.doc = doc

        @property
        def image_reader_writer_class(self):  # 语义与 nnU-Net 2.6.2 一致：property → 类
            reads["n"] += 1
            return _FakeReader

        @property
        def transpose_forward(self):
            return (0, 1, 2)

        def get_configuration(self, name):
            return pytypes.SimpleNamespace(
                spacing=(3.0, 0.5, 0.5), resampling_fn_seg=lambda *a, **k: None
            )

    fake = pytypes.ModuleType("nnunetv2.utilities.plans_handling.plans_handler")
    fake.PlansManager = _FakePlansManager
    for parent in ("nnunetv2", "nnunetv2.utilities", "nnunetv2.utilities.plans_handling"):
        monkeypatch.setitem(sys.modules, parent, pytypes.ModuleType(parent))
    monkeypatch.setitem(sys.modules, "nnunetv2.utilities.plans_handling.plans_handler", fake)

    plans_path = tmp_path / "nnUNetPlans.json"
    plans_path.write_text(json.dumps({"image_reader_writer": "SimpleITKIO"}))
    args = pytypes.SimpleNamespace(plans=str(plans_path), configuration="3d_fullres")

    context = mod.build_context(args)

    assert reads["n"] == 1, f"reader 只能实例化一次（property 被读取 {reads['n']} 次）"
    reader = context["reader"]
    assert isinstance(reader, _FakeReader), f"必须是 reader 实例，收到 {type(reader).__name__}"
    assert callable(getattr(reader, "read_seg", None)), "返回对象必须具备 read_seg"
    with pytest.raises(TypeError):
        reader()  # 旧写法会在这里失败（实例不可调用）
    assert context["transpose_forward"] == (0, 1, 2)
    assert context["target_spacing"] == (3.0, 0.5, 0.5)
    assert is_sha256_hex(context["plans_sha256"])
    assert context["plans_path"] == plans_path
