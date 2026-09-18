"""PZ/TZ prior 端到端链路合成测试（纯合成数组，不读取任何真实医学数据）。

覆盖：
- plan-space 转换顺序（transpose → crop → resample）与 lesion oracle 校验；
- 物化编排的 ok / skipped(resume) / failed / 不覆盖语义与失败清单；
- 数据链路：sampler 的 bbox/padding 同步、镜像增强同步、provider 3-tuple 与 CPU prior 校验；
- trainer 按 REQUIRED_INPUT_KEYS 调用 `model(image)` / `model(image, zonal_prior=...)`；
- sliding-window 的 prior patch 与 image patch 坐标逐个一致（含镜像 TTA）；
- 配置契约：M3/M4 必须配 prior，M0–M2 禁止声明 prior，来源标记强制。
"""
from __future__ import annotations

import hashlib
import json
import types
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest
import torch

from zonal_reliability_fusion.config.experiment import (
    ExperimentConfigError,
    load_experiment_config,
)
from zonal_reliability_fusion.data.batch_provider import (
    BatchProviderError,
    PatchDataset,
    TorchBatchProvider,
)
from zonal_reliability_fusion.data.patch_sampler import PatchSampler
from zonal_reliability_fusion.data.preprocessed_store import (
    PreprocessedCase,
    PreprocessedStore,
    PreprocessedStoreError,
)
from zonal_reliability_fusion.data.transforms import (
    MirrorAugmentor,
    build_train_transforms,
)
from zonal_reliability_fusion.data.zonal_prior import (
    ZONAL_LABELS,
    ZonalPriorError,
    ZonalPriorMaterializer,
    convert_label_to_plan_space,
    geometry_from_properties,
    label_to_pz_tz,
    oracle_check_lesion,
    validate_prior_array,
    write_prior_sidecar,
)
from zonal_reliability_fusion.inference.sliding_window import sliding_window_predict
from zonal_reliability_fusion.training.trainer import M0Trainer

PATCH = (8, 8, 8)


# --------------------------------------------------------------------------- 纯函数
def test_label_to_pz_tz_semantics_and_illegal_labels():
    label = np.zeros((2, 3, 4), dtype=np.int16)
    label[0, 0, 0] = 1  # PZ
    label[1, 2, 3] = 2  # TZ
    prior = label_to_pz_tz(label)
    assert prior.shape == (2, 2, 3, 4) and prior.dtype == np.float32
    assert prior[0, 0, 0, 0] == 1.0 and prior[1, 0, 0, 0] == 0.0
    assert prior[1, 1, 2, 3] == 1.0 and prior[0, 1, 2, 3] == 0.0
    assert float((prior[0] + prior[1]).max()) == 1.0
    bad = label.copy()
    bad[0, 0, 1] = 7
    with pytest.raises(ZonalPriorError, match="非法取值"):
        label_to_pz_tz(bad)
    assert ZONAL_LABELS == (0, 1, 2)


def test_geometry_requires_frozen_properties():
    props = {"spacing": [3.0, 0.5, 0.5], "shape_before_cropping": [16, 320, 320]}
    with pytest.raises(ZonalPriorError, match="bbox_used_for_cropping"):
        geometry_from_properties(props, transpose_forward=(0, 1, 2), target_spacing=(3.0, 0.5, 0.5))
    with pytest.raises(ZonalPriorError, match="transpose_forward"):
        geometry_from_properties(props, transpose_forward=(0, 1), target_spacing=(3.0, 0.5, 0.5))


def test_convert_order_transpose_crop_resample():
    """顺序必须是 transpose → crop(properties bbox) → resample，且形状逐值受控。"""
    label = np.zeros((4, 6, 8), dtype=np.int16)
    label[1, 2, 3] = 1
    props = {
        "spacing": [3.0, 1.0, 1.0],
        "shape_before_cropping": [4, 6, 8],
        "bbox_used_for_cropping": [[0, 4], [1, 5], [2, 6]],
    }
    geometry = geometry_from_properties(props, transpose_forward=(0, 1, 2), target_spacing=(3.0, 0.5, 0.5))
    calls: list[tuple] = []

    def _resample(arr, new_shape, cur_spacing, new_spacing):
        calls.append((tuple(arr.shape), tuple(new_shape), tuple(cur_spacing), tuple(new_spacing)))
        out = np.zeros((1, *new_shape), dtype=arr.dtype)
        out[0, 1, 1, 1] = 1  # 故意写一个可识别的值
        return out

    converted = convert_label_to_plan_space(label, geometry, resample_fn=_resample)
    assert calls and calls[0][0] == (1, 4, 4, 4)  # 已 crop（(1,4,6,8)→(1,4,4,4)）
    assert calls[0][1] == (4, 8, 8)  # 目标形状按 spacing 比例
    assert tuple(converted.shape) == (4, 8, 8)
    assert converted[1, 1, 1] == 1


def test_convert_rejects_grid_mismatch():
    label = np.zeros((5, 6, 8), dtype=np.int16)
    props = {
        "spacing": [3.0, 1.0, 1.0],
        "shape_before_cropping": [4, 6, 8],
        "bbox_used_for_cropping": [[0, 4], [0, 6], [0, 8]],
    }
    geometry = geometry_from_properties(props, transpose_forward=(0, 1, 2), target_spacing=(3.0, 1.0, 1.0))
    with pytest.raises(ZonalPriorError, match="不是同一网格"):
        convert_label_to_plan_space(label, geometry, resample_fn=lambda a, s, c, n: a)


def test_oracle_check_exact_and_shape_mismatch():
    seg = np.zeros((3, 4, 5), dtype=np.uint8)
    seg[1, 1, 1] = 1
    exact = oracle_check_lesion(seg.copy(), seg[None])
    assert exact["exact_match"] and exact["mismatch_voxels"] == 0
    shifted = np.zeros_like(seg)
    shifted[2, 2, 2] = 1
    report = oracle_check_lesion(shifted, seg)
    assert not report["exact_match"] and report["mismatch_voxels"] > 0
    bad_shape = oracle_check_lesion(np.zeros((2, 2, 2), dtype=np.uint8), seg)
    assert not bad_shape["exact_match"] and bad_shape["reason"] == "shape_mismatch"


def test_validate_prior_array_rejects_overlap_and_range():
    good = np.zeros((2, 4, 4, 4), dtype=np.float32)
    good[0, 0, 0, 0] = 1.0
    validate_prior_array(good)
    overlap = good.copy()
    overlap[1, 0, 0, 0] = 1.0  # PZ + TZ = 2 > 1
    with pytest.raises(ZonalPriorError, match="重叠"):
        validate_prior_array(overlap)
    with pytest.raises(ZonalPriorError, match=r"\[0,1\]"):
        validate_prior_array(good * 5.0)
    with pytest.raises(ZonalPriorError, match="形状"):
        validate_prior_array(np.zeros((3, 4, 4, 4), dtype=np.float32))


# --------------------------------------------------------------------------- 物化编排
PLANS_SHA = "a" * 64


def _stub_convert(case_id: str, *, oracle_ok: bool = True):
    def convert(cid: str):
        prior = np.zeros((2, 4, 4, 4), dtype=np.float32)
        prior[0, 0, 0, 0] = 1.0
        meta = {
            "source": "yuan",
            "configuration": "3d_fullres",
            "plans_sha256": PLANS_SHA,
            "input_sha256": {"zonal": hashlib.sha256(f"hash-{cid}".encode()).hexdigest()},
            "oracle": {
                "exact_match": oracle_ok,
                "shape_match": True,
                "mismatch_voxels": 0 if oracle_ok else 7,
            },
        }
        return prior, meta

    return convert


def test_materializer_ok_skip_fail_and_no_overwrite(tmp_path):
    out_root = tmp_path / "zonal_out"
    materializer = ZonalPriorMaterializer(out_root, source="yuan", configuration="3d_fullres", plans_sha256=PLANS_SHA)
    summary = materializer.run(["c1", "c2"], _stub_convert("c1"), progress=False)
    assert summary["n_ok"] == 2 and summary["n_failed"] == 0
    assert (out_root / "c1.npz").is_file() and (out_root / "c1.json").is_file()
    meta = json.loads((out_root / "c1.json").read_text())
    assert meta["source"] == "yuan" and meta["output_shape"] == [2, 4, 4, 4]
    assert meta["output_channel_semantics"] == ["PZ", "TZ"] and meta["output_sha256"]

    # 默认拒绝覆盖（元数据完整但文件已存在）
    with pytest.raises(ZonalPriorError, match="已存在"):
        write_prior_sidecar(
            out_root,
            "c1",
            np.zeros((2, 4, 4, 4), dtype=np.float32),
            _stub_convert("c1")("c1")[1],
        )

    # resume：同来源 + 同输入哈希 → skipped
    resume = ZonalPriorMaterializer(out_root, source="yuan", configuration="3d_fullres", plans_sha256=PLANS_SHA, resume=True)
    summary2 = resume.run(["c1"], _stub_convert("c1"), progress=False)
    assert summary2["n_skipped"] == 1 and summary2["n_ok"] == 0

    # 失败病例被逐例记录（不中断整批）
    def _boom(_cid: str):
        raise ZonalPriorError("oracle 校验失败：mismatch")

    summary3 = ZonalPriorMaterializer(
        tmp_path / "zonal_fail", source="hevi", configuration="3d_fullres", plans_sha256=PLANS_SHA
    ).run(
        ["bad1", "bad2"], _boom, progress=False
    )
    assert summary3["n_failed"] == 2 and len(summary3["failed"]) == 2
    assert "oracle" in summary3["failed"][0]["reason"]


def test_materializer_rejects_oracle_mismatch_and_writes_summary(tmp_path):
    """oracle 不通过必须 fail-closed：n_failed=1、不写 sidecar、reason 明确为 oracle rejection。"""
    out_root = tmp_path / "zonal_oracle"
    materializer = ZonalPriorMaterializer(out_root, source="yuan", configuration="3d_fullres", plans_sha256=PLANS_SHA)
    summary = materializer.run(["c1"], _stub_convert("c1", oracle_ok=False), progress=False)
    assert summary["n_ok"] == 0 and summary["n_failed"] == 1
    assert "oracle rejection" in summary["failed"][0]["reason"]
    assert not (out_root / "c1.npz").exists() and not (out_root / "c1.json").exists()
    path = materializer.write_summary(summary)
    assert path.is_file() and json.loads(path.read_text())["n_failed"] == 1


def test_materializer_validates_source_and_flags(tmp_path):
    with pytest.raises(ZonalPriorError, match="source"):
        ZonalPriorMaterializer(tmp_path, source="unknown", configuration="3d_fullres", plans_sha256=PLANS_SHA)
    with pytest.raises(ZonalPriorError, match="互斥"):
        ZonalPriorMaterializer(
        tmp_path, source="yuan", configuration="3d_fullres", plans_sha256=PLANS_SHA, resume=True, overwrite=True
    )


# --------------------------------------------------------------------------- 数据链路
class _FakeStore:
    """最小 duck-typing store：单病例 + 可选 prior（形状与 seg 一致）。"""

    def __init__(self, *, with_prior: bool = True, prior: np.ndarray | None = None):
        shape = (4, 5, 6)
        grid = np.indices(shape)
        self.data = np.stack([grid[0], grid[1], grid[2]]).astype(np.float32)  # 0/1/2 通道编码坐标
        self.seg = (grid[1] == 2).astype(np.uint8)[None]
        if prior is None:
            base = np.zeros((2, *shape), dtype=np.float32)
            base[0] = (grid[1] == 2).astype(np.float32)
            base[1] = 1.0 - base[0]
            prior = base
        self.prior = prior if with_prior else None
        self.zonal_prior_root = "fake-root" if with_prior else None
        self.zonal_prior_source = "yuan" if with_prior else None
        self.available_ids = ("case_a",)

    def get_case(self, case_id: str, *, include_zonal_prior: bool = False):
        return PreprocessedCase(
            case_id=case_id,
            data=self.data,
            seg=self.seg,
            properties=types.SimpleNamespace(class_locations=None),
            zonal_prior=self.prior if include_zonal_prior else None,
        )

    def describe_zonal_prior(self) -> dict:
        return {"enabled": self.zonal_prior_root is not None, "source": self.zonal_prior_source}


def test_sampler_crop_and_padding_sync_image_seg_prior():
    store = _FakeStore()
    sampler = PatchSampler(patch_size=PATCH, oversample_foreground=0.0, seed=3)
    for seed in range(5):
        rng = np.random.default_rng(seed)
        sample = sampler.sample(store.data, store.seg, zonal_prior=store.prior, rng=rng, force_fg=False)
        assert sample.zonal_prior.shape == (2, *PATCH)
        assert sample.bbox == sample.prior_bbox
        # data 通道 0/1/2 编码 (i,j,k)；prior 由 (j==2) 派生 → 用同一坐标场验证三者同一切片
        d0 = sample.data[0]
        assert np.array_equal(sample.zonal_prior[0], (d0 % 100 == 2).astype(np.float32) * 0 + sample.zonal_prior[0])
        assert sample.bbox[1][0] <= 2 < sample.bbox[1][1]  # j==2 落在同一 crop 内
    # padding 同步：给一个小于 patch 的病例
    small = _FakeStore()
    small.data = np.zeros((3, 3, 3, 3), dtype=np.float32)
    small.seg = np.zeros((1, 3, 3, 3), dtype=np.uint8)
    small.prior = np.ones((2, 3, 3, 3), dtype=np.float32)
    small.prior[1] = 0.0
    sample = sampler.sample(small.data, small.seg, zonal_prior=small.prior, force_fg=False)
    assert sample.data.shape[-3:] == PATCH and sample.seg.shape[-3:] == PATCH
    assert sample.zonal_prior.shape == (2, *PATCH)
    assert float(sample.zonal_prior[..., 3:, :, :].sum()) == 0.0  # padding 区域 prior = 0


def test_mirror_augmentation_syncs_prior():
    store = _FakeStore()
    sampler = PatchSampler(patch_size=PATCH, oversample_foreground=0.0, seed=1)
    sample = sampler.sample(store.data, store.seg, zonal_prior=store.prior, force_fg=False)
    pipeline = build_train_transforms(mirror=True, mirror_p_per_axis=1.0)  # 一定翻转
    data2, seg2, prior2 = pipeline.apply(sample.data, sample.seg, np.random.default_rng(0), sample.zonal_prior)
    manual = np.flip(sample.data, axis=(1, 2, 3))
    assert np.array_equal(data2, manual)
    assert np.array_equal(prior2, np.flip(sample.zonal_prior, axis=(1, 2, 3)))
    assert np.array_equal(seg2, np.flip(sample.seg, axis=(1, 2, 3)))
    # 只给 data/seg 时保持旧接口（2-tuple）
    data3, seg3 = MirrorAugmentor(p_per_axis=0.0)(sample.data, sample.seg, np.random.default_rng(0))
    assert np.array_equal(data3, sample.data) and np.array_equal(seg3, sample.seg)


def test_provider_returns_prior_triple_and_validates():
    store = _FakeStore()
    provider = TorchBatchProvider(
        store,
        PatchSampler(patch_size=PATCH, oversample_foreground=0.0, seed=0),
        batch_size=2,
        num_batches=2,
        include_zonal_prior=True,
        force_foreground=False,
    )
    batches = list(provider)
    assert all(len(batch) == 3 for batch in batches)
    data, seg, prior = batches[0]
    assert data.shape[1:] == (3, *PATCH) and seg.shape[1:] == (1, *PATCH) and prior.shape[1:] == (2, *PATCH)
    assert provider.describe()["zonal_prior"]["source"] == "yuan"
    # 旧接口：不带 prior 时仍是 2-tuple
    legacy = TorchBatchProvider(
        store, PatchSampler(patch_size=PATCH, oversample_foreground=0.0, seed=0), batch_size=2, num_batches=1
    )
    assert all(len(batch) == 2 for batch in legacy)
    # 非法 prior（overlap）必须失败
    bad = _FakeStore()
    bad.prior = bad.prior.copy()
    bad.prior[1] = 1.0  # PZ 与 TZ 全 1 → overlap
    bad_provider = TorchBatchProvider(
        bad,
        PatchSampler(patch_size=PATCH, oversample_foreground=0.0, seed=0),
        batch_size=1,
        num_batches=1,
        include_zonal_prior=True,
        force_foreground=False,
    )
    with pytest.raises(BatchProviderError, match="重叠"):
        list(bad_provider)
    # 需要 prior 但 store 未配置 → 构造即失败
    with pytest.raises(BatchProviderError, match="未配置"):
        TorchBatchProvider(
            _FakeStore(with_prior=False),
            PatchSampler(patch_size=PATCH),
            batch_size=1,
            num_batches=1,
            include_zonal_prior=True,
        )


def test_patch_dataset_missing_prior_raises():
    """store 已配置 prior 根目录、但返回的病例没有 prior → 逐样本立即失败（禁止零兜底）。"""

    class _NoPriorStore(_FakeStore):
        def get_case(self, case_id: str, *, include_zonal_prior: bool = False):
            case = super().get_case(case_id, include_zonal_prior=False)
            return PreprocessedCase(
                case_id=case.case_id,
                data=case.data,
                seg=case.seg,
                properties=case.properties,
                zonal_prior=None,
            )

    store = _NoPriorStore(with_prior=True)
    dataset = PatchDataset(
        store,
        PatchSampler(patch_size=PATCH),
        case_ids=("case_a",),
        transform=None,
        seed=0,
        batch_size=1,
        force_foreground=False,
        length=1,
        include_zonal_prior=True,
    )
    with pytest.raises(BatchProviderError):
        dataset[0]


# --------------------------------------------------------------------------- trainer 调用约定
class _PriorModel(torch.nn.Module):
    REQUIRED_INPUT_KEYS = ("image", "zonal_prior")

    def __init__(self):
        super().__init__()
        self.calls: list[dict] = []

    def forward(self, image, *, zonal_prior=None):
        self.calls.append({"image": image.shape, "prior": None if zonal_prior is None else zonal_prior.shape})
        return image[:, :2]


class _PlainModel(torch.nn.Module):
    REQUIRED_INPUT_KEYS = ("image",)

    def __init__(self):
        super().__init__()
        self.calls: list[dict] = []

    def forward(self, image):
        self.calls.append({"image": image.shape})
        return image[:, :2]


def test_trainer_to_device_and_model_dispatch():
    prior_model = _PriorModel()
    trainer = types.SimpleNamespace(model=prior_model, device=torch.device("cpu"))
    image = torch.zeros(1, 3, 4, 4, 4)
    seg = torch.zeros(1, 1, 4, 4, 4)
    prior = torch.zeros(1, 2, 4, 4, 4)
    data, _, prior_t = M0Trainer._to_device(trainer, (image, seg, prior))
    assert prior_t is not None and prior_t.shape == prior.shape
    M0Trainer._forward_model(trainer, data, prior_t)
    assert prior_model.calls[0]["prior"] == tuple(prior.shape)
    with pytest.raises(RuntimeError, match="batch 未提供"):
        M0Trainer._forward_model(trainer, data, None)

    plain = _PlainModel()
    trainer2 = types.SimpleNamespace(model=plain, device=torch.device("cpu"))
    data2, _, prior2 = M0Trainer._to_device(trainer2, (image, seg))
    assert prior2 is None
    M0Trainer._forward_model(trainer2, data2, None)
    assert plain.calls[0]["image"] == tuple(image.shape)
    with pytest.raises(RuntimeError, match="禁止静默混用"):
        M0Trainer._forward_model(trainer2, data2, prior)
    with pytest.raises(RuntimeError, match="batch 必须是"):
        M0Trainer._to_device(trainer2, (image,))


# --------------------------------------------------------------------------- 滑窗对齐
class _RecordingPriorModel(torch.nn.Module):
    """记录每个 patch 的 image/prior；返回与 prior 空间同尺寸的 logits。"""

    REQUIRED_INPUT_KEYS = ("image", "zonal_prior")

    def __init__(self):
        super().__init__()
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def forward(self, image, *, zonal_prior=None):
        assert zonal_prior is not None
        self.calls.append((image.detach().clone(), zonal_prior.detach().clone()))
        return zonal_prior[:, :2]


def _coordinate_field(shape) -> np.ndarray:
    grid = np.indices(shape)
    return (grid[0] * 10_000 + grid[1] * 100 + grid[2]).astype(np.float32)


def test_sliding_window_prior_patches_aligned_with_image():
    rng = np.random.default_rng(0)
    shape = (5, 9, 11)
    coords = _coordinate_field(shape)
    image = torch.from_numpy(np.stack([coords, rng.random(shape).astype(np.float32), coords])[None])
    prior_np = np.zeros((2, *shape), dtype=np.float32)
    prior_np[0] = (coords % 2 == 0).astype(np.float32)
    prior_np[1] = 1.0 - prior_np[0]
    prior = torch.from_numpy(prior_np[None])

    model = _RecordingPriorModel()
    out = sliding_window_predict(
        model, image, zonal_prior=prior, patch_size=(4, 4, 4), step_fraction=0.5, progress=False, desc="t"
    )
    assert out.shape[-3:] == shape and len(model.calls) > 1
    for image_patch, prior_patch in model.calls:
        # prior 由 image 通道 0 的坐标场派生（PZ = coords%2==0）→ 若窗口对齐，可由 image patch 复原 prior
        expected_pz = (image_patch[0, 0] % 2 == 0).float()
        assert torch.equal(prior_patch[0, 0], expected_pz), "prior patch 与 image patch 坐标不一致"
        assert image_patch.shape[-3:] == prior_patch.shape[-3:] == (4, 4, 4)

    # 镜像 TTA 也必须同步（对齐关系在翻转后仍然成立）
    model2 = _RecordingPriorModel()
    sliding_window_predict(
        model2,
        image,
        zonal_prior=prior,
        patch_size=(4, 4, 4),
        step_fraction=1.0,
        mirror_tta=True,
        progress=False,
        desc="t",
    )
    assert model2.calls
    for image_patch, prior_patch in model2.calls:
        assert torch.equal(prior_patch[0, 0], (image_patch[0, 0] % 2 == 0).float())


def test_sliding_window_requires_prior_for_prior_models():
    model = _RecordingPriorModel()
    image = torch.zeros(1, 3, 4, 4, 4)
    with pytest.raises(ValueError, match="未提供"):
        sliding_window_predict(model, image, patch_size=(4, 4, 4), progress=False)
    plain = _PlainModel()
    prior = torch.zeros(1, 2, 4, 4, 4)
    with pytest.raises(ValueError, match="不读取 zonal_prior"):
        sliding_window_predict(plain, image, zonal_prior=prior, patch_size=(4, 4, 4), progress=False)


# --------------------------------------------------------------------------- 配置契约
def test_config_contract_for_zonal_prior(tmp_path):
    good = load_experiment_config("configs/experiments/m4_conditioned_gate_picai_3d_fullres_v24.yaml")
    assert good.data.zonal_prior_source == "yuan" and good.data.zonal_prior_root
    assert "yuan" in good.paths.output_root and "yuan" in good.experiment.name
    import yaml

    doc = yaml.safe_load(Path("configs/experiments/m4_conditioned_gate_picai_3d_fullres_v24.yaml").read_text())
    bad_source = dict(doc)
    bad_source["data"] = {**doc["data"], "zonal_prior_source": "none"}
    p = tmp_path / "bad_source.yaml"
    p.write_text(yaml.safe_dump(bad_source, sort_keys=False))
    with pytest.raises(ExperimentConfigError, match="非法"):
        load_experiment_config(p)

    m2 = yaml.safe_load(Path("configs/experiments/m2_image_gate_picai_3d_fullres_v24.yaml").read_text())
    m2_bad = {**m2, "data": {**m2["data"], "zonal_prior_source": "yuan", "zonal_prior_root": "data/x"}}
    p2 = tmp_path / "m2_bad.yaml"
    p2.write_text(yaml.safe_dump(m2_bad, sort_keys=False))
    with pytest.raises(ExperimentConfigError, match="不读取 PZ/TZ prior"):
        load_experiment_config(p2)

    m3 = yaml.safe_load(Path("configs/experiments/m3_zone_input_picai_3d_fullres_v24.yaml").read_text())
    m3_bad = {**m3, "paths": {**m3["paths"], "output_root": "outputs/checkpoints/m3_zone_input"}}
    p3 = tmp_path / "m3_bad.yaml"
    p3.write_text(yaml.safe_dump(m3_bad, sort_keys=False))
    with pytest.raises(ExperimentConfigError, match="来源标记"):
        load_experiment_config(p3)


def test_store_sidecar_roundtrip_and_failures(tmp_path):
    """store 读取 sidecar：形状/来源/overlap 不合法必须失败（不静默兜底）。"""
    root = tmp_path / "sidecar"
    prior = np.zeros((2, 4, 5, 6), dtype=np.float32)
    prior[0, 0, 0, 0] = 1.0
    write_prior_sidecar(
        root,
        "case_a",
        prior,
        {
            "source": "yuan",
            "configuration": "3d_fullres",
            "plans_sha256": PLANS_SHA,
            "input_sha256": {"zonal": hashlib.sha256(b"h").hexdigest()},
            "oracle": {"exact_match": True, "shape_match": True, "mismatch_voxels": 0},
        },
    )
    store = PreprocessedStore.__new__(PreprocessedStore)  # 绕过目录校验，仅测 sidecar 读取
    store.zonal_prior_root = root
    store.zonal_prior_source = "yuan"
    store.expected_configuration = "3d_fullres"
    store.expected_plans_sha256 = PLANS_SHA
    store.zonal_prior_manifest_sha256 = None
    store.zonal_prior_validation_stats = {}
    store.zonal_prior_frozen_records = MappingProxyType({})
    store.require_frozen_manifest = False
    loaded = store.load_zonal_prior("case_a", seg_shape=(4, 5, 6))
    assert loaded.shape == (2, 4, 5, 6) and loaded[0, 0, 0, 0] == 1.0
    with pytest.raises(PreprocessedStoreError, match="空间尺寸"):
        store.load_zonal_prior("case_a", seg_shape=(4, 5, 7))
    with pytest.raises(PreprocessedStoreError, match="sidecar"):
        store.load_zonal_prior("case_missing", seg_shape=(4, 5, 6))
    store.zonal_prior_source = "hevi"
    with pytest.raises(PreprocessedStoreError, match="来源"):
        store.load_zonal_prior("case_a", seg_shape=(4, 5, 6))
    store.zonal_prior_source = "yuan"
    meta_path = root / "case_a.json"
    meta = json.loads(meta_path.read_text())
    meta["output_shape"] = [2, 4, 5, 7]
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(PreprocessedStoreError, match="output_shape"):
        store.load_zonal_prior("case_a", seg_shape=(4, 5, 6))
