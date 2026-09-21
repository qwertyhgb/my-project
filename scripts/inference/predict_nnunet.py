#!/usr/bin/env python3
"""最小项目预测入口：把三个项目 Trainer 名映射到项目类后，调用 nnU-Net 官方预测入口。

**不自写推理器**：滑窗推理、预处理、导出全部由官方 ``nnUNetPredictor`` 与
``predict_entry_point_modelfolder`` 完成；本脚本只做两件事：

1. 在**进程内**替换 ``nnunetv2.inference.predict_from_raw_data.recursive_find_python_class``，
   使官方 ``initialize_from_trained_model_folder`` 在读取 checkpoint 的 ``trainer_name`` 后，
   能直接定位到项目自定义 Trainer 类（对三个已知名做直接映射，**不做递归扫描**；未知名字
   仍委托给 nnU-Net 原函数）。不修改 third_party/nnUNet。
2. 校验固定 nnU-Net 运行时（``check_fixed_nnunet_runtime``），然后调用官方 entry point。

用法（长任务由研究者运行；先 conda activate lm && source scripts/env_nnunet.sh）：
    python scripts/inference/predict_nnunet.py \
        -i <输入影像目录> -o <输出目录> \
        -m outputs/nnUNet_results/Dataset605_PICAI/nnUNetTrainerPICAI_FLCE_NoFFT__nnUNetPlans__3d_fullres \
        -f 0 -device cuda
其余参数（-chk/--save_probabilities/-npp/-nps/--disable_tta 等）与官方
``nnUNetv2_predict_from_modelfolder`` 完全一致（``--help`` 即官方帮助）。
"""

from __future__ import annotations

import sys

#: 项目三个 Trainer 的类名（与 checkpoint 中的 ``trainer_name`` 一致）；模块级常量，便于测试
PROJECT_TRAINER_NAMES = (
    "nnUNetTrainerPICAI_FLCE_NoFFT",
    "nnUNetTrainerPICAI_ImageGate",
    "nnUNetTrainerPICAI_AnatomyGate",
)

_ORIGINAL_RECURSIVE_FIND = None


def resolve_project_trainer(trainer_name: str):
    """按类名解析项目 Trainer 类；不是项目 Trainer 时返回 ``None``。"""
    from zonal_reliability_fusion.nnunet.trainers import resolve_trainer_class

    return resolve_trainer_class(trainer_name)


def install_project_trainer_resolver():
    """在进程内为官方预测模块安装 trainer 名解析器（幂等；不修改 nnU-Net 源码文件）。

    返回被安装到 ``predict_from_raw_data.recursive_find_python_class`` 的解析函数。
    """
    global _ORIGINAL_RECURSIVE_FIND
    import nnunetv2.inference.predict_from_raw_data as predict_module

    if _ORIGINAL_RECURSIVE_FIND is None:
        _ORIGINAL_RECURSIVE_FIND = predict_module.recursive_find_python_class
    original = _ORIGINAL_RECURSIVE_FIND

    def resolver(folder, class_name, current_module):
        project_class = resolve_project_trainer(class_name)
        if project_class is not None:
            # 直接映射，避免在 nnunetv2 包目录内递归扫描（也避免扫到环境中其他 nnunetv2 分支）
            return project_class
        return original(folder, class_name, current_module)

    predict_module.recursive_find_python_class = resolver
    return resolver


def main(argv=None) -> None:
    # 允许 `--help` 直接透传到官方 parser；其余情况先校验运行时再委托官方 entry point。
    from zonal_reliability_fusion.nnunet import check_fixed_nnunet_runtime

    check_fixed_nnunet_runtime()
    install_project_trainer_resolver()

    from nnunetv2.inference.predict_from_raw_data import predict_entry_point_modelfolder

    predict_entry_point_modelfolder()


if __name__ == "__main__":
    import os

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
    sys.exit(main())
