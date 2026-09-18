"""pytest 公共夹具：使 src/ 可导入，并提供合成 mini plan / 模型（不读取真实医学数据）。"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# 合成 mini plan：结构合法但与真实 plan 尺寸无关，供 CPU 单元测试使用（patch 8x16x16，3 级）
MINI_PLAN_DOC: dict[str, Any] = {
    "dataset_name": "Dataset605_PICAI",
    "plans_name": "nnUNetPlans",
    "configurations": {
        "3d_fullres": {
            "data_identifier": "nnUNetPlans_3d_fullres",
            "batch_size": 2,
            "patch_size": [8, 16, 16],
            "spacing": [3.0, 0.5, 0.5],
            "batch_dice": False,
            "architecture": {
                "network_class_name": "dynamic_network_architectures.architectures.unet.PlainConvUNet",
                "arch_kwargs": {
                    "n_stages": 3,
                    "features_per_stage": [4, 8, 8],
                    "conv_op": "torch.nn.modules.conv.Conv3d",
                    "kernel_sizes": [[1, 3, 3], [3, 3, 3], [3, 3, 3]],
                    "strides": [[1, 1, 1], [2, 2, 2], [2, 2, 2]],
                    "n_conv_per_stage": [1, 1, 1],
                    "n_conv_per_stage_decoder": [1, 1],
                    "conv_bias": True,
                    "norm_op": "torch.nn.modules.instancenorm.InstanceNorm3d",
                    "norm_op_kwargs": {"eps": 1e-5, "affine": True},
                    "dropout_op": None,
                    "nonlin": "torch.nn.LeakyReLU",
                    "nonlin_kwargs": {"inplace": True, "negative_slope": 0.01},
                },
            },
        }
    },
}


def write_plan(tmp_path: Path, doc: dict) -> Path:
    path = tmp_path / "nnUNetPlans.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2))
    return path


@pytest.fixture()
def mini_plan_doc() -> dict:
    return copy.deepcopy(MINI_PLAN_DOC)


@pytest.fixture()
def mini_plan(tmp_path: Path, mini_plan_doc: dict):
    """小型但结构合法的 Plan3DConfig（patch 8x16x16，3 级，DS 输出 2 个）。"""
    from zonal_reliability_fusion.config import load_plan

    return load_plan(write_plan(tmp_path, mini_plan_doc), "3d_fullres")


@pytest.fixture()
def mismatch_plan(tmp_path: Path, mini_plan_doc: dict):
    """奇数尺寸会触发 decoder 中心裁剪的 plan（patch 8x10x10）。"""
    from zonal_reliability_fusion.config import load_plan

    doc = copy.deepcopy(mini_plan_doc)
    doc["configurations"]["3d_fullres"]["patch_size"] = [8, 10, 10]
    return load_plan(write_plan(tmp_path, doc), "3d_fullres")


@pytest.fixture()
def mini_model(mini_plan):
    from zonal_reliability_fusion.models import M0ConcatModel

    return M0ConcatModel(mini_plan, in_channels=3, num_classes=2, deep_supervision=True)


@pytest.fixture(autouse=True)
def _force_cpu(monkeypatch: pytest.MonkeyPatch):
    """测试全程禁止 GPU：设置 CUDA_VISIBLE_DEVICES 为空（不初始化 CUDA）。"""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")


# --------------------------------------------------------------------------- v2.3 Residual-Encoder M0 夹具
# 4 级合成 plan：足以覆盖 stride/通道变化与 deep supervision；patch 8x16x16（CPU 廉价 forward）
MINI_RESENC_PLAN_DOC: dict[str, Any] = {
    "dataset_name": "Dataset605_PICAI",
    "plans_name": "nnUNetPlans",
    "configurations": {
        "3d_fullres": {
            "data_identifier": "nnUNetPlans_3d_fullres",
            "batch_size": 2,
            "patch_size": [8, 16, 16],
            "spacing": [3.0, 0.5, 0.5],
            "batch_dice": False,
            "architecture": {
                "network_class_name": "dynamic_network_architectures.architectures.unet.PlainConvUNet",
                "arch_kwargs": {
                    "n_stages": 4,
                    "features_per_stage": [4, 8, 16, 16],
                    "conv_op": "torch.nn.modules.conv.Conv3d",
                    "kernel_sizes": [[1, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
                    "strides": [[1, 1, 1], [1, 2, 2], [2, 2, 2], [2, 2, 2]],
                    "n_conv_per_stage": [2, 2, 2, 2],
                    "n_conv_per_stage_decoder": [2, 2, 2],
                    "conv_bias": True,
                    "norm_op": "torch.nn.modules.instancenorm.InstanceNorm3d",
                    "norm_op_kwargs": {"eps": 1e-5, "affine": True},
                    "dropout_op": None,
                    "nonlin": "torch.nn.LeakyReLU",
                    "nonlin_kwargs": {"inplace": True, "negative_slope": 0.01},
                },
            },
        }
    },
}


def build_resenc_arch_doc(plan, blocks_per_stage=None) -> dict[str, Any]:
    """按给定 plan 派生一份**结构一致**的 v2.3 架构配置 doc（供合成测试构建 ArchitectureSpec）。

    非 plan 派生的规则字段（stem/shortcut/norm/nonlin/init/decoder/DS/§8.6 契约）与仓库
    `configs/architectures/m0_resenc_v23.yaml` 保持同一冻结取值。
    """
    from zonal_reliability_fusion.config.architecture import SHALLOW_SKIP_CONTRACT

    n = plan.n_stages
    blocks = list(blocks_per_stage) if blocks_per_stage is not None else [2] * n
    return {
        "architecture_name": "test_resenc",
        "architecture_version": "v2.3",
        "encoder_type": "residual_encoder",
        "block_type": "basic_post_activation",
        "activation_order": "post_activation",
        "n_stages": n,
        "blocks_per_stage": blocks,
        "features_per_stage": list(plan.features_per_stage),
        "kernel_sizes": [list(k) for k in plan.kernel_sizes],
        "strides": [list(s) for s in plan.strides],
        "stem": {"type": "stacked_conv_blocks", "n_convs": 1, "stride": [1, 1, 1], "does_downsample": False},
        "shortcut": {
            "identity_when": "same_spatial_and_channels",
            "downsample_when": "stride_change",
            "projection_when": "channel_change",
            "downsample_op": "avgpool3d_kernel_eq_stride",
            "downsample_before_projection": True,
            "projection_kernel": [1, 1, 1],
            "projection_conv_bias": False,
            "projection_norm": True,
            "implicit_channel_crop_forbidden": True,
        },
        "normalization": {"op": "InstanceNorm3d", "eps": 1e-5, "affine": True},
        "nonlinearity": {"op": "LeakyReLU", "negative_slope": 0.01, "inplace": True},
        "conv_bias": True,
        "initialization": {
            "method": "he_kaiming_normal",
            "mode": "fan_in",
            "nonlinearity": "leaky_relu",
            "negative_slope": 0.01,
            "conv_bias_init": "zeros",
            "norm_weight_init": "ones",
            "norm_bias_init": "zeros",
            "zero_init_last_norm_before_add": True,
        },
        "decoder": {
            "type": "plain_unet_decoder",
            "upsample": "convtranspose3d",
            "transpconv_kernel_equals_stride": True,
            "skip_align": "center_crop_or_pad",
            "skip_merge": "concat",
            "stage_block": "stacked_conv_blocks",
            "n_conv_per_stage_decoder": list(plan.n_conv_per_stage_decoder),
            "per_level_seg_heads": True,
            "residual": False,
        },
        "deep_supervision": {
            "output_order": "high_to_low_resolution",
            "weight_scheme": "1_over_2_pow_i",
            "lowest_resolution_weight_zero": True,
            "weights_normalized_to_one": True,
            "single_output_weight": 1.0,
        },
        "shallow_skip_contract": copy.deepcopy(SHALLOW_SKIP_CONTRACT),
        "fusion_stage": 2 if n > 2 else n - 1,
        "in_channels": 3,
        "num_classes": 2,
        "provenance": {"configuration": "3d_fullres", "plan_source": str(plan.source_path)},
    }


@pytest.fixture()
def mini_resenc_plan_doc() -> dict:
    return copy.deepcopy(MINI_RESENC_PLAN_DOC)


@pytest.fixture()
def mini_resenc_plan(tmp_path: Path, mini_resenc_plan_doc: dict):
    """4 级合成 plan（patch 8x16x16；DS 输出 3 个）。"""
    from zonal_reliability_fusion.config import load_plan

    return load_plan(write_plan(tmp_path / "resenc", mini_resenc_plan_doc), "3d_fullres")


@pytest.fixture()
def odd_resenc_plan(tmp_path: Path, mini_resenc_plan_doc: dict):
    """奇数尺寸 plan（patch 8x10x10）：触发 decoder center_crop_or_pad 对齐。"""
    from zonal_reliability_fusion.config import load_plan

    doc = copy.deepcopy(mini_resenc_plan_doc)
    doc["configurations"]["3d_fullres"]["patch_size"] = [8, 10, 10]
    return load_plan(write_plan(tmp_path / "resenc_odd", doc), "3d_fullres")


@pytest.fixture()
def resenc_spec(mini_resenc_plan):
    """与 mini_resenc_plan 一致的 ArchitectureSpec（blocks_per_stage=(1,2,2,2)）。"""
    from zonal_reliability_fusion.config import architecture_spec_from_mapping

    doc = build_resenc_arch_doc(mini_resenc_plan, blocks_per_stage=[1, 2, 2, 2])
    return architecture_spec_from_mapping(doc, source_path="synthetic")


@pytest.fixture()
def mini_resenc_model(mini_resenc_plan, resenc_spec):
    from zonal_reliability_fusion.models import M0ResidualConcatModel

    return M0ResidualConcatModel(mini_resenc_plan, resenc_spec, in_channels=3, num_classes=2, deep_supervision=True)
