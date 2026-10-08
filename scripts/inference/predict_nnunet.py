#!/usr/bin/env python3
"""最小项目预测入口：把项目 Trainer 名映射到项目类后，调用 nnU-Net 官方预测入口。

**不自写推理器**：滑窗推理、预处理、导出全部由官方 ``nnUNetPredictor`` 与
``predict_entry_point_modelfolder`` 完成；本脚本只做三件事：

1. 在**进程内**替换 ``nnunetv2.inference.predict_from_raw_data.recursive_find_python_class``，
   使官方 ``initialize_from_trained_model_folder`` 在读取 checkpoint 的 ``trainer_name`` 后，
   能直接定位到项目自定义 Trainer 类（对已知项目 Trainer 名做直接映射，**不做递归扫描**；
   未知名字仍委托给 nnU-Net 原函数）。不修改 ``third_party/nnUNet``。
2. 校验固定 nnU-Net 运行时（``check_fixed_nnunet_runtime``）。
3. 对 **Stage-1 解剖先验模型**执行额外守卫与产物完整性校验
   （:mod:`zonal_reliability_fusion.anatomy.inference`）；其余模型的预测完全走原生路径。

用法（长任务由研究者运行；先 ``conda activate lm && source scripts/env_nnunet.sh``）::

    python scripts/inference/predict_nnunet.py \
        -i <输入影像目录> -o <输出目录> \
        -m outputs/nnUNet_results/Dataset605_PICAI/nnUNetTrainerPICAI_FLCE_NoFFT__nnUNetPlans__3d_fullres \
        -f 0 -device cuda

其余参数（``-chk`` / ``--save_probabilities`` / ``-npp`` / ``-nps`` / ``--disable_tta`` 等）
与官方 ``nnUNetv2_predict_from_modelfolder`` 完全一致（``--help`` 即官方帮助）。

**Stage-1 解剖先验**必须导出 soft probability（下游只读 npz 的 soft 输出，不读硬标签）::

    python scripts/inference/predict_nnunet.py \
        -i <T2W 单通道目录> -o <先验输出目录> \
        -m outputs/nnUNet_results/Dataset607_PICAI_Anatomy/nnUNetTrainerPICAI_AnatomyJoint_100ep_NoFFT__nnUNetPlans__3d_fullres \
        -f 0 --save_probabilities -device cuda
"""

from __future__ import annotations

import sys

# 项目包路径引导：``scripts/env_nnunet.sh`` 已把 ``<root>/src`` 放进 PYTHONPATH；这里再补一次，
# 使脚本在未 source 环境时同样可用。
from pathlib import Path as _Path

_PROJECT_SRC = _Path(__file__).resolve().parents[2] / "src"
if str(_PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(_PROJECT_SRC))

#: 项目全部 Trainer 的类名（与 checkpoint 中的 ``trainer_name`` 一致）。
#: **由注册表派生**，因此不可能与 ``PROJECT_TRAINERS`` 漂移——新增 Trainer 忘记注册时
#: 预测入口会在同一处立刻失败，而不是在推理时才报「找不到类」。
def _project_trainer_names() -> tuple[str, ...]:
    from zonal_reliability_fusion.nnunet.trainers import PROJECT_TRAINERS

    return tuple(sorted(PROJECT_TRAINERS))


PROJECT_TRAINER_NAMES: tuple[str, ...] = _project_trainer_names()

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


# --------------------------------------------------------------------------- 兼容再导出
# 解剖先验相关实现已迁入 ``zonal_reliability_fusion.anatomy.inference``；这里保留同名转发，
# 使既有测试与历史命令的导入路径继续有效。
from zonal_reliability_fusion.anatomy.inference import (  # noqa: F401
    anatomy_prediction_preflight,
    check_anatomy_prediction_folder,
    load_anatomy_probability_case,
)

# 历史私有名：既有测试与审计脚本按带下划线的名字引用该守卫
_anatomy_prediction_preflight = anatomy_prediction_preflight


def main(argv=None) -> None:
    # 允许 ``--help`` 直接透传到官方 parser；其余情况先校验运行时再委托官方 entry point。
    from zonal_reliability_fusion.nnunet import check_fixed_nnunet_runtime

    check_fixed_nnunet_runtime()
    install_project_trainer_resolver()

    from nnunetv2.inference.predict_from_raw_data import predict_entry_point_modelfolder

    arguments = list(sys.argv[1:] if argv is None else argv)
    anatomy = anatomy_prediction_preflight(arguments)
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *arguments]
        predict_entry_point_modelfolder()
    finally:
        sys.argv = original_argv
    if anatomy is not None:
        check_anatomy_prediction_folder(*anatomy)


if __name__ == "__main__":
    import os

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
    main()
