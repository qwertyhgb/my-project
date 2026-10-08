# 旧门控 / 特征融合研究线（已归档）

> 状态：**ARCHIVED 2026-10-08**。本文件描述已停止扩张的研究线，用于 checkpoint 追溯与论文
> motivation 引用。当前主线见 `docs/Research_Plan.md`。

## 这条线在问什么

> T2W / ADC / HBV 应该以什么权重进行融合？PZ/TZ 是否应当作为融合的条件？

三个层次的问题被依次实现：

1. **输入级 modality gate**（`ImageGate` / `AnatomyGate`）——在原生 backbone 之前对 3 个 MRI
   通道乘一个逐体素尺度场；
2. **浅层序列特异特征融合**（`FeatureNoGate` / `FeatureImageGate` / `FeatureAnatomyGate`）——
   三条序列各有独立的两层 3×3×3 stem，1×1×1 投影回 3 通道，可选的 feature gate 只读拼接后的
   stem 特征；
3. **同区参照残差融合**（`ZonalReference` / `ZonalReferenceAdaptive`）——在面内降采样 4 倍的特征
   统计上计算"同区局部参照"的标准化对比，作为残差注入。这一支的 4 个 100-epoch 条件
   **全部在构造阶段失败**（`KeyError: 'args'`），修复后**从未重跑**，因此**没有任何结果**。

## 关键实现事实（供 checkpoint 追溯）

- gate 结构：`Conv3d(in, 8, 1) → InstanceNorm3d → LeakyReLU → Conv3d(8, 3, 1)`，末层
  weight/bias 零初始化 ⇒ 初始 `scales ≡ 1`，因此新模型初始行为与原生输入级 nnU-Net 逐值一致。
- anatomy 条件：5 通道（T2W/ADC/HBV + PZ/TZ），PZ/TZ **只**进入 gate，**不**进入分割 backbone；
  进入 gate 前 `clamp(0,1)`；强度增强只作用于前 3 个 MRI 通道，空间增强同步作用于全部通道。
- feature 融合：三个 stem 参数**不共享**、不下采样；投影到**恰好 3 通道**；`anatomy` 与 `image`
  的 gate 参数量相差 `gate_hidden_channels * 2`（C_s=8 时为 16），因此**不得**声称两者参数量严格
  相同。
- 类名中的 `Reliability` 是**历史内部标识**（为 checkpoint 与导入路径兼容而保留），当前解释是
  **learned sequence preference / scale**，**不是**校准可靠性、真实图像质量或因果贡献。

## 为什么停止

1. **输入级重加权没有稳定收益**：两次配对比较（image gate、anatomy 条件）的病例级 Dice 变化
   95% CI 均跨 0；
2. **收益与代价同时出现**：两个 gate 臂都把工作点推向更保守一侧（precision 上升、召回下降、
   漏检数上升），这在"首要问题是漏检"的项目里方向不对；
3. **更基本的事实**：`positive_sampling`（增加病灶暴露）带来 **+0.1093** 的配对 Dice 提升，
   而 modality 重加权在同一层级上是 0 附近 —— 说明瓶颈在**病灶暴露与定位**，不在通道加权。

## 可以怎样被引用（表述纪律）

可以说：

> Direct input-level modality reweighting did not provide a consistent segmentation improvement
> in the tested runs, motivating a shift toward lesion-aware anatomical modeling.

不可以说：

- "gate is proven useless" / "gate 无效" / "证明 gate 没有作用" / "gate 降低性能"；
- "PZ/TZ 无用" / "解剖信息导致性能下降"；
- 把 CI 跨 0 解读为"两者等效"。

理由：CI 跨 0 既不支持提升也不支持下降；全部对比均为**单次运行**（nnU-Net v2.6.2 不设随机
种子）；`Dataset605 ↔ 606` 的逐值一致性审计只覆盖前三个 MRI 通道，PZ/TZ 不参与判等，因此
anatomy 条件的比较并非严格单变量。

## 相关代码（只读，禁止扩张）

- `src/zonal_reliability_fusion/legacy/fusion_networks.py`——网络实现；
- `src/zonal_reliability_fusion/legacy/fusion_trainers.py`——Trainer 实现（类名逐字保留）；
- `src/zonal_reliability_fusion/legacy/prior_transforms.py`——先验通道的增强边界（行为契约已被
  新主线在 `lesion/prior_channels.py` 继承）；
- 兼容转发层：`nnunet/networks.py`、`nnunet/transforms.py`（仅供历史导入路径使用）。

训练入口默认**不显示**这些条件，需要 `--legacy` 才出现在 variant 列表里。

## 参考研究

1. Perez-Garcia et al., *Deep learning-based segmentation of prostatic zones*, 2021.
2. Sanyal et al., *Learning to balance multi-modal MRI for prostate segmentation*.
3. PI-CAI challenge report（Focal + CE 损失的来源）。
