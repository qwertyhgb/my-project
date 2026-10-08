"""Anatomy-Guided Lesion-Aware Coarse-to-Fine Prostate Cancer Segmentation。

> 本项目研究预测前列腺解剖先验与病灶感知粗到细学习能否提高小型和困难前列腺癌病灶的检出与
> 完整分割，同时控制假阳性。

包结构（研究逻辑优先，import 路径保持稳定以兼容既有 checkpoint 与历史记录）
--------------------------------------------------------------------------
===================  ============================================================
包                    角色
===================  ============================================================
:mod:`anatomy`        Stage 1：predicted WG / PZ / TZ soft prior 的契约、布局与校验
:mod:`lesion`         Stage 2：论文方法本体（ROI、lesionness、coarse-to-fine、zone 条件化）
:mod:`sampling`       训练采样策略（阳性采样 = baseline 组成；困难负样本 = 条件 E）
:mod:`evaluation`     统一评价体系（case / lesion / anatomy 三层，协议冻结）
:mod:`nnunet`         nnU-Net 集成层（Trainer、损失、运行时校验）
:mod:`legacy`         归档的旧 gate / fusion 研究线（只读）
===================  ============================================================

术语边界：包名与历史类名里的 ``zonal_reliability_fusion`` / ``Reliability`` 是**历史内部
标识**，保留是为了 checkpoint 与既有导入路径兼容。当前研究解释是 learned sequence
preference / scale，**不是**校准可靠性、真实图像质量或因果贡献。
"""

from __future__ import annotations

__all__: tuple[str, ...] = ()
