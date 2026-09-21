"""pytest 公共夹具：纯合成、CPU-only，不读取任何真实医学数据。

只负责：把 ``src/`` 与固定版本 ``third_party/nnUNet`` 加入 import 路径、强制 CPU、
并提供一份结构合法的合成 nnU-Net 架构参数（供 ``build_network_architecture`` /
``get_network_from_plans`` 在 CPU 上构建极小网络）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = PROJECT_ROOT / "src"
_NNUNET = PROJECT_ROOT / "third_party" / "nnUNet"
# third_party/nnUNet 必须在最前，确保使用固定 v2.6.2 而非 lm 环境中可编辑安装的其他分支
for _p in (str(_NNUNET), str(_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


@pytest.fixture(autouse=True)
def _force_cpu(monkeypatch: pytest.MonkeyPatch):
    """测试全程禁止 GPU：清空 CUDA_VISIBLE_DEVICES（不初始化 CUDA）。"""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")


#: 合成 3 级 PlainConvUNet 架构参数（patch 8x16x16，CPU 廉价 forward；DS 输出 2 个）
SYNTHETIC_ARCH_CLASS = "dynamic_network_architectures.architectures.unet.PlainConvUNet"
SYNTHETIC_ARCH_KWARGS = {
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
    "dropout_op_kwargs": None,
    "nonlin": "torch.nn.LeakyReLU",
    "nonlin_kwargs": {"inplace": True},
}
SYNTHETIC_ARCH_REQ_IMPORT = ["conv_op", "norm_op", "dropout_op", "nonlin"]


@pytest.fixture()
def synthetic_arch() -> dict:
    """返回构建原生 backbone 所需的 (class_name, kwargs, req_import) 三元组字典。"""
    import copy

    return {
        "architecture_class_name": SYNTHETIC_ARCH_CLASS,
        "arch_init_kwargs": copy.deepcopy(SYNTHETIC_ARCH_KWARGS),
        "arch_init_kwargs_req_import": list(SYNTHETIC_ARCH_REQ_IMPORT),
    }
