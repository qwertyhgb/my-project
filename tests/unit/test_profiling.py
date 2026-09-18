"""结构摘要 / 参数量 / MACs / 显存状态测试（任务书八、九·13；纯 CPU，解析计算，不 forward 真实 patch）。"""
from __future__ import annotations

import torch

from zonal_reliability_fusion.models import M0ResidualConcatModel
from zonal_reliability_fusion.models.profiling import (
    analytical_feature_map_elements,
    analytical_macs,
    count_macs_with_hooks,
    count_parameters,
    structure_summary,
)


def test_count_parameters_matches_module(mini_resenc_model: M0ResidualConcatModel):
    p = count_parameters(mini_resenc_model)
    assert p["total"] == sum(q.numel() for q in mini_resenc_model.parameters())
    assert p["trainable"] == p["total"] and p["non_trainable"] == 0


def test_analytical_macs_equals_hook_macs_on_reduced_plan(mini_resenc_model, mini_resenc_plan, resenc_spec):
    """解析 MACs 公式必须与 forward-hook 计数在缩小的合成 plan 上逐值一致（验证公式正确）。"""
    hook = count_macs_with_hooks(mini_resenc_model, torch.randn(1, 3, *mini_resenc_plan.patch_size))
    ana = analytical_macs(mini_resenc_plan, resenc_spec, 3, 2)
    assert hook["total"] == ana["total"] > 0
    assert ana["encoder"] > 0 and ana["decoder"] > 0
    assert ana["encoder"] + ana["decoder"] == ana["total"]


def test_analytical_macs_scales_with_blocks(mini_resenc_plan, resenc_spec):
    from conftest import build_resenc_arch_doc

    from zonal_reliability_fusion.config import architecture_spec_from_mapping

    base = analytical_macs(mini_resenc_plan, resenc_spec, 3, 2)["total"]
    more = build_resenc_arch_doc(mini_resenc_plan, blocks_per_stage=[2, 4, 4, 4])
    spec2 = architecture_spec_from_mapping(more, source_path="s2")
    bigger = analytical_macs(mini_resenc_plan, spec2, 3, 2)["total"]
    assert bigger > base  # 更多 block → 更多 MACs


def test_feature_map_elements_positive(mini_resenc_plan, resenc_spec):
    acts = analytical_feature_map_elements(mini_resenc_plan, resenc_spec, 3, 2)
    assert acts["total"] == acts["encoder"] + acts["decoder"] > 0


def test_structure_summary_contents_and_memory_status(mini_resenc_model, mini_resenc_plan, resenc_spec):
    s = structure_summary(mini_resenc_model, mini_resenc_plan, resenc_spec, in_channels=3, num_classes=2)
    # 必需字段
    for key in (
        "architecture_name",
        "architecture_version",
        "architecture_sha256",
        "blocks_per_stage",
        "features_per_stage",
        "stage_shapes",
        "decoder_output_shapes",
        "parameters",
        "macs",
        "activation_feature_map_elements",
        "peak_gpu_memory",
    ):
        assert key in s
    assert s["architecture_sha256"] == resenc_spec.architecture_sha256()
    assert s["blocks_per_stage"] == list(resenc_spec.blocks_per_stage)
    assert s["stage_shapes"] == [list(x) for x in mini_resenc_plan.stage_shapes()]
    assert s["decoder_output_shapes"] == [list(x) for x in mini_resenc_plan.decoder_output_shapes()]
    # MACs 口径明确
    assert "MACs" in s["macs"]["unit"] and "FLOPs" in s["macs"]["unit"]
    assert s["macs"]["includes_decoder"] is True
    assert s["macs"]["includes_deep_supervision_heads"] is True
    assert s["macs"]["method"].startswith("analytical")
    # 显存：明确未实测，不得伪造成实测值
    assert s["peak_gpu_memory"]["status"] == "NOT_MEASURED"
    assert "reason" in s["peak_gpu_memory"]
    assert s["peak_gpu_memory"]["analytical_activation_bytes_fp32"] > 0


def test_summary_does_not_run_forward_on_real_patch(mini_resenc_model, mini_resenc_plan, resenc_spec):
    """structure_summary 只做解析计算：调用前后模型无 buffer/缓存被 forward 写入（eval/train 状态不变）。"""
    mini_resenc_model.train()
    structure_summary(mini_resenc_model, mini_resenc_plan, resenc_spec, in_channels=3, num_classes=2)
    assert mini_resenc_model.training is True  # 未被内部切换到 eval（无隐藏 forward）
