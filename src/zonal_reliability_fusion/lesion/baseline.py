"""Strong Baseline 的定义与单变量边界（新主线所有消融的唯一参照）。

新主线**只**允许在 strong baseline 上逐步增加模块。任何新模块进入最终模型之前，必须能与
strong baseline 做**单变量**比较：一次只改一个东西。

新主线启动时的第一个比较（A1 vs A2）
------------------------------------
======================  ====================================================  =============
条件                    组成                                                  回答的问题
======================  ====================================================  =============
``A1``                  nnU-Net + PI-CAI Focal+CE + PositiveSampling          Focal+CE 能达到什么水平
``A2``                  nnU-Net + 原生 Dice+CE + PositiveSampling             原生 Dice+CE 是否更好
======================  ====================================================  =============

两者的**唯一**差异必须是损失函数。因此下列各项必须逐项相同：网络（原生 ``PlainConvUNet``）、
数据集与 split、fold、patch size、batch size、增强（含 NoFFT 修复）、前景采样比例、deep
supervision 权重、optimizer（SGD+Nesterov）、初始学习率、PolyLR 总周期、epoch 数、
checkpoint 策略、validation 与滑窗推理。

:func:`baseline_single_variable_boundary` 把这张表变成可审计的数据结构，写进 run 配置；
:data:`STRONG_BASELINE_COMPONENTS` 声明 strong baseline 的**组成部分**，用于在代码审阅时
快速判断「某个新模块是否偷偷改了 baseline」。

阳性采样的定位（必须一直写对）
------------------------------
``positive_sampling`` 是 **foreground-aware training stabilization strategy**，**不是**论文的
方法创新。它属于 strong baseline 的一部分。因此它在论文中出现在「训练设置」，不出现在
「Method contribution」。
"""

from __future__ import annotations

#: A1 / A2 的名称 -> Trainer 类名（与 ``nnunet/trainers.py`` 的注册表一致）
STRONG_BASELINE_CANDIDATES: dict[str, str] = {
    "A1_flce_positive_sampling": "nnUNetTrainerPICAI_FLCE_PositiveSampling_NoFFT",
    "A2_dicece_positive_sampling": "nnUNetTrainerPICAI_DiceCE_PositiveSampling_NoFFT",
}

#: strong baseline 的组成部分；任何新方法都只能在**完整保留**这些组成的前提下叠加
STRONG_BASELINE_COMPONENTS: tuple[str, ...] = (
    "输入：T2W + ADC + HBV（Dataset605/606 的前三个 MRI 通道）",
    "网络：由 nnUNetPlans.json 构建的原生 PlainConvUNet（不复制 encoder/decoder）",
    "损失：A1 = PI-CAI 0.5*Focal(gamma=2)+0.5*CE；A2 = nnU-Net 原生 Dice+CE",
    "采样：PositiveCaseSampling（每 batch 固定一个阳性病灶 patch；验证 loader 保持原生）",
    "增强：nnU-Net 默认管线 + NoFFT 兼容修复（blur 概率/sigma 不变）",
    "优化器 / 调度：原生 SGD+Nesterov 与 PolyLR",
    "deep supervision：原生 DeepSupervisionWrapper 与 1/2^i 权重",
    "checkpoint / validation / 滑窗推理：全部原生",
)

#: 新主线四条件的顺序（论文主线，不允许并行扩张）
MAINLINE_CONDITIONS: tuple[tuple[str, str], ...] = (
    ("A", "Strong Baseline（best loss + positive sampling）"),
    ("B", "A + Anatomy-Guided ROI（predicted WG -> prostate-centred ROI）"),
    ("C", "B + Lesion-Aware Coarse-to-Fine（coarse lesionness + soft refinement）"),
    ("D", "C + Zone-Aware Refinement（predicted PZ/TZ soft context）"),
)

#: 条件 E：训练策略增强，可能进最终方法，也可能只作为 ablation
TRAINING_STRATEGY_CONDITIONS: tuple[tuple[str, str], ...] = (
    ("E", "best(C, D) + Anatomy-Constrained Hard Negative Mining"),
)

#: 每个条件必须能回答的三个问题；回答不了就不该进主模型
FAILURE_MODE_QUESTIONS: tuple[str, ...] = (
    "Does it reduce missed lesions?",
    "Does it improve lesion coverage?",
    "Does it reduce false positives?",
)


def baseline_single_variable_boundary() -> dict:
    """A1 ↔ A2 的单变量边界（可审计，写进 run 配置与论文附录）。

    只有 ``loss`` 允许不同；其余各项由 ``docs/Experiment_Plan.md`` 与
    ``scripts/train/train_nnunet.py`` 共同保证。
    """
    return {
        "compared": sorted(STRONG_BASELINE_CANDIDATES),
        "single_variable": "loss",
        "loss_variants": {
            "A1_flce_positive_sampling": "0.5*Focal(gamma=2, alpha=None) + 0.5*CrossEntropy",
            "A2_dicece_positive_sampling": (
                "nnU-Net v2.6.2 原生 DC_and_CE_loss（MemoryEfficientSoftDiceLoss，"
                "do_bg=False，weight_ce=weight_dice=1）"
            ),
        },
        "must_be_identical": STRONG_BASELINE_COMPONENTS,
        "not_a_comparison": (
            "不得把原生采样的 gate / DiceCE 臂（如 optimized_baseline）与本比较混用："
            "采样不同则不是单变量"
        ),
    }


__all__ = (
    "FAILURE_MODE_QUESTIONS",
    "MAINLINE_CONDITIONS",
    "STRONG_BASELINE_CANDIDATES",
    "STRONG_BASELINE_COMPONENTS",
    "TRAINING_STRATEGY_CONDITIONS",
    "baseline_single_variable_boundary",
)
