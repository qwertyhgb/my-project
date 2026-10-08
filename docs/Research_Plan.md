# Research Plan

**Anatomy-Guided Lesion-Aware Multimodal Coarse-to-Fine Fusion for Prostate Cancer Segmentation**

Working title; subject to change after experimental validation. 不预先决定 acronym。
本文只记录研究问题、假设、机制、边界与预期贡献；实验矩阵见 Experiment_Plan，
实现契约见 Method，指标唯一来源为 Evaluation_Protocol，真实运行见 Training_Log。

## 1. Research Question

**预测的前列腺解剖先验和粗病灶定位信息，能否共同指导 T2W、ADC、HBV 的局部多模态特征融合，
从而减少小型和困难病灶的漏检、改善完整覆盖，并控制假阳性？**

Can predicted prostate anatomy and coarse lesion localization jointly condition local multimodal
fusion of T2W, ADC and HBV to reduce missed lesions, improve lesion coverage, and control false positives?

## 2. Failure Modes

三类失败保持不变：完全漏检（尤其小型/困难病灶）、已检出但覆盖不足、敏感度提高后的假阳负担。
三者必须同时评价，不能通过扩大预测换 recall 而不报告 precision 与 FP。
多模态融合服务于定位和解剖感知分割；不把“哪个 MRI 权重更大”作为独立研究目标。

## 3. Preliminary Evidence and Its Boundary

既有实验提供方向性动机，证据与限制见 Findings 的 Preliminary Findings。
历史运行不是受控重复实验；病例 bootstrap 不包含训练随机性，CI 跨 0 不证明等效或无效。
旧输入级/特征级 gate 与同区参照路线不恢复、不扩张，历史说明见 docs/archive/。
既有证据不能推出新条件化融合有效，也不能推出其优于其它已发表机制。

## 4. Hypotheses

| 假设 | 待检验机制 | 对应问题 |
|---|---|---|
| H1 | sequence-specific shallow representation 提供比原始通道 early fusion 更合适的表征底座 | 是否值得先独立编码再进入共享 backbone？ |
| H2 | coarse lesion localization 条件化局部融合，减少完全漏检 | lesion-aware fusion 是否改善 sensitivity？ |
| H3 | predicted anatomy 在 lesion-aware fusion 上进一步改善覆盖或控制 FP | 解剖环境是否提供增量？ |
| H4 | 困难负样本挖掘降低 FP 且不损害 sensitivity | 训练策略是否收回敏感度代价？ |

全部是待验证假设。H1 是机制性前置，H4 是训练策略，不强行包装为网络创新。

## 5. Stage 1: Predicted Anatomy

冻结的单 T2W anatomy model 输出 soft P(WG)、P(PZ)、P(TZ)，只作为先验生成器。
它不是主要创新；监督来自算法伪标签，不等于人工解剖真值。
anatomy model 不读取 lesion GT。验证和外部病例的 anatomy 必须来自其自身 MRI 的预测，
且不得参与该 anatomy checkpoint 的训练。GT anatomy 只用于显式 ORACLE_GT 上界分析。

训练病例若被 anatomy model 见过，预测标记为 IN_SAMPLE_PRED；这不是 lesion-label leakage，
但可能产生先验质量的 train/test shift。最终优先评估 cross-fitted / OOF 方案，先核定成本，
不因计划变化自动训练多个 anatomy model。OOF、HELD_OUT、EXTERNAL 模式必须可追踪。

重叠 sigmoid heads 的独立预测与按顺序覆盖的硬导出不同，质量必须直接评估 soft heads；
不能从硬导出低 WG Dice 单独推断 WG 头塌缩。实际诊断记录见 Training_Log 与 Stage-1 实验文档。

## 6. Neutral Multimodal Representation

对 m∈{T2W, ADC, HBV}，H_m=E_m(X_m)，三个浅层 stem 不共享参数。
F0=P0(concat(H_T2,H_ADC,H_HBV))，投影到原生 backbone 接受的三通道。
只使用轻量卷积，不建立三个完整 encoder，不复制 nnU-Net。
该表征是机制对照；其价值须由实验确认。

## 7. Lesion-Aware Coarse Localization

在 fusion controller 前，由中性表征产生 coarse logits，L=sigmoid(h_L(F0))。
辅助目标只从训练 lesion GT 的物理半径膨胀生成；推理只使用模型预测。
coarse 分支与最终分割联合训练，L 真正进入 controller，不只是最终分割之后的辅助头。
L 是 soft context；禁止按阈值硬裁或硬乘，L=0 处仍保留完整 neutral path。

## 8. Anatomy- and Lesion-Conditioned Local Fusion

A=[P(WG),P(PZ),P(TZ),P(U)]，P(U)=1-max(P(PZ),P(TZ))。
P(U) 只是 uncertainty-like context，不是概率校准结论。

z=g(concat(H_T2,H_ADC,H_HBV,L,A))；w=softmax(z)，每位置三个系数和为 1。
Hadaptive=Σ_m w_m H_m；Ffinal=F0+R(concat(Hadaptive,L,A))。
R 的末层零初始化，在相同 neutral/backbone 权重下，初始 Ffinal=F0。
随后由原生 nnU-Net 输出 fine lesion segmentation。

C 只使用 L，D 再加入 A。anatomy 不进行 hard gating，不把不同区域送入独立完整 encoder。
这与输出 logits 侧的 residual prediction refinement 是不同数据流；既有实现保留为 supporting ablation。

## 9. Supporting Strategies

Positive sampling 属于 strong baseline。ROI 是可选 search-space / sampling ablation；
它不再占用核心条件 B 的名称，是否进入最终模型由独立证据决定。
Hard-negative mining 只在最终 C/D 候选确定后研究，只用 training split，不用 validation error 挑样本。
不把 ROI、采样、融合、loss、优化器同时改变后作单变量解释。

## 10. Mechanism Analysis

融合系数只表示模型行为，不代表 MRI 的临床重要性或医学因果贡献。
可分析 lesion/background、预测 WG/PZ/TZ、解剖不确定区及真实物理病灶大小分层中的系数分布和熵。
GT lesion 只允许进入离线分析；不能作为推理条件。不得强求系数符合预期临床规则。
必须区分导出的 patch 系数与经过审计的全体积系数。

## 11. Controlled Comparisons

A 是最佳 baseline loss 下的 early-fusion nnU-Net；B 为 neutral representation；
C 增加 lesion-conditioned residual fusion；D 再增加 anatomy context；E 增加困难负样本采样。
每个条件的预算与对照见 Experiment_Plan。C 的增量包含辅助监督与条件残差这一整体，
仅 C vs B 不足以把收益单独归因到 softmax 系数；必要时加 auxiliary-only 机制消融。
损失家族由 A1/A2 结果选择，不能永久绑定 FLCE。

## 12. Evaluation

Primary：所有 GT-positive 病例的 macro Dice，完全漏检记 0。
Key secondary：病灶级/小病灶 sensitivity、matched-lesion Dice、complete miss、positive voxel
recall/precision 与 FP burden。Matched-lesion Dice 必须与 sensitivity 同报。
大小分层是 exploratory physical-volume strata，不是临床分级。
评价仍为 segmentation failure analysis，不计算挑战赛 detection score。
指标、匹配、空值语义与统计常量全部引用 Evaluation_Protocol，不在本文重新定义。

## 13. Novelty Boundary

不声称首次 adaptive fusion、lesion-guided MRI fusion、coarse-to-fine prostate segmentation 或
anatomy-aware PCa analysis。待验证差异是 predicted zonal anatomy 与 coarse lesion localization
共同条件化局部多模态融合。正式 novelty claim 必须另行文献调研。

## 14. Reproducibility

历史无 seed baseline 只保留为历史证据。正式 loss 选择优先 matched-seed 对照。
最终 baseline 与 proposed model 使用同 fold、同预算、同评估、三个显式 seed，报告各 run、
mean±std 与病例级配对统计。Explicit seed improves repeatability but does not guarantee bitwise determinism.
短预算只筛选方向，不与历史 full-budget 模型直接声明优劣。

## 15. Stop Rules

H1 无可重复收益：先判断是否保留 independent stems，不立即加深 encoder。
H2 未减少 complete miss 且未提高 sensitivity：记录假设未获支持，不立即增加 attention depth。
H3 未显示稳定增量：停止 anatomy-conditioned fusion 扩张，最终模型可以停在 C。
H4 未降低 FP 或损害 sensitivity：只作为 negative/ablation evidence。
CI 跨 0 表示未观察到明确配对改善，不表示等效。具体运行判据见 Experiment_Plan。

## 16. External Stress Test

架构与选型冻结后才使用 Prostate158。使用同一冻结 anatomy pipeline，不用外部 GT anatomy，
不根据 test 表现改参数。第三序列与 HBV 不完全同质时称 distribution-shift stress test。

## 17. Expected Contributions

1. Lesion-Aware Coarse Localization：显式学习可疑位置，针对小型/困难病灶完全漏检。
2. Anatomy- and Lesion-Conditioned Multimodal Fusion：结合预测解剖与 lesionness 指导局部特征整合。

两条都是待验证贡献。Lesion-Centric Evaluation and Fusion Behaviour Analysis 是实验分析支撑，
不必作为第三个方法创新。

## 18. Limitations

算法解剖伪监督、单 fold、训练随机性、标签噪声、prior quality shift、额外参数与计算量均须报告。
增加模型容量本身可能解释收益，须报告参数增量，必要时采用容量对照。
零初始化只保证初始 neutral 等价，不保证训练后效果，也不保证不同独立初始化模型逐值相同。
不把单次改善写成稳定贡献，不把诊断阈值扫描当正式阈值选型。
