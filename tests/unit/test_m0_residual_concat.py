"""M0ResidualConcatModel 与工厂分派测试（任务书九·19,20,21；纯 CPU，不读真实数据）。

覆盖：新配置只能构造 residual M0、旧配置只能构造 legacy PlainConv、身份不混淆、不能静默回退、
main_logits / 模态顺序守卫 / summary / architecture_identity。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from conftest import build_resenc_arch_doc

from zonal_reliability_fusion.config import ArchitectureRef
from zonal_reliability_fusion.config.architecture import ArchitectureConfigError
from zonal_reliability_fusion.models import M0ConcatModel, M0ResidualConcatModel
from zonal_reliability_fusion.models.factory import (
    build_m0_model,
    model_architecture_identity,
    resolve_architecture_spec,
    verify_ref_matches_spec,
)


class _ModelCfg:
    in_channels = 3
    num_classes = 2
    deep_supervision = True


class _FakeCfg:
    def __init__(self, architecture, root: Path) -> None:
        self.architecture = architecture
        self.model = _ModelCfg()
        self._root = Path(root)

    def resolve_path(self, value):
        p = Path(value)
        return p if p.is_absolute() else self._root / p


def _write_arch(tmp_path: Path, mini_resenc_plan, **over) -> ArchitectureRef:
    doc = build_resenc_arch_doc(mini_resenc_plan, blocks_per_stage=[1, 2, 2, 2])
    doc.update(over)
    (tmp_path / "arch.json").write_text(json.dumps(doc))
    return ArchitectureRef(
        config_path="arch.json",
        architecture_name=doc["architecture_name"],
        architecture_version=doc["architecture_version"],
        encoder_type=doc["encoder_type"],
    )


# ------------------------------------------------------------------ 19. 新配置 → 只能 residual
def test_new_config_builds_residual_m0(tmp_path: Path, mini_resenc_plan):
    ref = _write_arch(tmp_path, mini_resenc_plan)
    model = build_m0_model(_FakeCfg(ref, tmp_path), mini_resenc_plan)
    assert isinstance(model, M0ResidualConcatModel)
    assert not isinstance(model, M0ConcatModel)
    ident = model_architecture_identity(model)
    assert ident is not None and ident["encoder_type"] == "residual_encoder"
    assert ident["architecture_name"] == "test_resenc" and len(ident["architecture_sha256"]) == 64


# ------------------------------------------------------------------ 20. 旧配置 → 只能 legacy PlainConv
def test_legacy_config_builds_plain_conv_m0(tmp_path: Path, mini_resenc_plan):
    model = build_m0_model(_FakeCfg(None, tmp_path), mini_resenc_plan)  # architecture=None
    assert isinstance(model, M0ConcatModel)
    assert not isinstance(model, M0ResidualConcatModel)
    assert model_architecture_identity(model) is None  # legacy 无版本化架构身份


def test_new_config_cannot_fall_back_to_plain(tmp_path: Path, mini_resenc_plan):
    """架构文件声明 encoder_type=plain_conv → 加载即拒绝（新配置不会静默回退到 PlainConv）。"""
    ref = _write_arch(tmp_path, mini_resenc_plan, encoder_type="plain_conv")
    with pytest.raises(ArchitectureConfigError):
        build_m0_model(_FakeCfg(ref, tmp_path), mini_resenc_plan)


def test_ref_identity_must_match_arch_config(tmp_path: Path, mini_resenc_plan):
    ref = _write_arch(tmp_path, mini_resenc_plan)
    bad_ref = ArchitectureRef(
        config_path="arch.json",
        architecture_name="WRONG_NAME",  # 与架构文件不一致
        architecture_version=ref.architecture_version,
        encoder_type=ref.encoder_type,
    )
    with pytest.raises(ArchitectureConfigError, match="不一致"):
        build_m0_model(_FakeCfg(bad_ref, tmp_path), mini_resenc_plan)


def test_resolve_spec_validates_against_plan(tmp_path: Path, mini_resenc_plan):
    ref = _write_arch(tmp_path, mini_resenc_plan)
    spec = resolve_architecture_spec(_FakeCfg(ref, tmp_path), mini_resenc_plan)
    assert spec.blocks_per_stage == (1, 2, 2, 2)
    with pytest.raises(ArchitectureConfigError):
        verify_ref_matches_spec(
            ArchitectureRef("arch.json", "x", "v2.3", "residual_encoder"), spec
        )


# ------------------------------------------------------------------ 模型接口
def test_main_logits_and_modality_guard(mini_resenc_model: M0ResidualConcatModel, mini_resenc_plan):
    outputs = mini_resenc_model(torch.randn(1, 3, *mini_resenc_plan.patch_size))
    main = M0ResidualConcatModel.main_logits(outputs)
    assert torch.equal(main, outputs[0])
    assert tuple(main.shape)[:2] == (1, 2)
    with pytest.raises(ValueError, match="模态顺序"):
        M0ResidualConcatModel(mini_resenc_plan, mini_resenc_model.spec, modality_order=["ADC", "T2W", "HBV"])
    with pytest.raises(ValueError, match="in_channels"):
        M0ResidualConcatModel(mini_resenc_plan, mini_resenc_model.spec, in_channels=4)


def test_summary_contains_identity_and_backbone(mini_resenc_model: M0ResidualConcatModel):
    s = mini_resenc_model.summary()
    assert s["model_id"] == "M0"
    assert s["backbone_id"] == "residual_encoder_v23"
    assert s["encoder_type"] == "residual_encoder"
    assert s["accepts_anatomy_inputs"] is False
    assert s["blocks_per_stage"] == [1, 2, 2, 2]
    assert s["fusion_stage"] == 2
    assert len(s["architecture_sha256"]) == 64
    text = mini_resenc_model.format_summary()
    assert "residual_encoder_v23" in text and "architecture_sha256" in text


def test_identity_matches_spec_hash(mini_resenc_model: M0ResidualConcatModel, resenc_spec):
    ident = mini_resenc_model.architecture_identity()
    assert ident["architecture_sha256"] == resenc_spec.architecture_sha256()
    assert ident["architecture_version"] == "v2.3"
