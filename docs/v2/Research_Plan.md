# VERSION=V2 — Research Plan

**WORKING TITLE: Anatomy-Guided Lesion-Aware Coarse-to-Fine Multimodal Prostate Cancer Segmentation**

中文：基于预测解剖先验与病灶感知粗到细多模态融合的三维前列腺癌病灶分割。
备选工作标题：Anatomy-Guided Adaptive Multimodal Fusion for Coarse-to-Fine 3D Prostate Cancer Segmentation。
不提前冻结 acronym，也不因代码存在声明方法有效。

本文只维护科学问题、待验证假设和贡献边界。具体预算、split、命令与运行 stop rules 见
[Experiment Plan](Experiment_Plan.md)；实现契约见 [Method](Method.md)；已有证据见
[Findings](Findings.md)。指标唯一来源为共享 [Evaluation Protocol](../Evaluation_Protocol.md)。

## 1. Clinical / technical motivation

T2W、ADC、HBV/DWI 提供不同的结构与扩散相关信息。病灶可表现为小体积、异质、
边界不清或困难背景；本研究把这些作为需要检验的失败模式，不预先声明某模态或解剖分区决定诊断。
固定 early fusion 直接将 MRI 通道送入共享 encoder，不能显式保留三个独立 modality-specific hierarchy。
这并不证明 early fusion 较差，而是提出一个可控制的 representation 对照。

预测的患者自身 WG/PZ/TZ 概率提供空间与分区 context。研究其是否帮助学习融合策略，
而非手工规定 PZ 用 ADC/DWI、TZ 用 T2W。粗病灶证据则用于末端局部修正的位置权重，
希望在控制假阳性的条件下减少漏检、改善覆盖；这些目标均须未来受控实验支持。

## 2. Research Question

**预测的前列腺解剖先验能否指导 T2W、ADC 和 HBV/DWI 的层次化自适应多模态融合，
从而改善三维前列腺癌病灶分割，
尤其减少小病灶和困难病灶漏检，同时控制假阳性？**

Can predicted prostate zonal anatomy guide hierarchical adaptive multimodal fusion of T2W,
ADC and HBV/DWI to improve accurate 3D prostate cancer lesion segmentation?

当前按用户要求收敛为规模可控的硕士研究：以融合为核心，H1–H3为必要假设；
H4/H5和完整coarse-to-fine框架保留为已实现候选，不作为当前必要贡献或必须启用的主模型。
不因四组结果不理想同时加入新损失、采样、backbone或多个轻量模块。

## 3. Central Hypotheses — all unvalidated

| 假设 | 待验证机制 | 解释边界 |
|---|---|---|
| H1 | modality-specific hierarchical encoding 相对 early/shared encoding 可能提供更合适的 representation | 完整 encoder 同时改变 capacity/compute；不能仅归因于模态语义 |
| H2 | 在相同可学习concat投影上，image-adaptive调节可能改善病灶表示 | 比较包含controller参数，不声称无容量混杂；均值控制保留但非当前强对照 |
| H3 | predicted WG/PZ/TZ context 在 adaptive fusion 上可能改善 coverage/sensitivity/FP trade-off | 须保持 MRI/标签/split 与预处理一致；先验误差可能传导 |
| H4 | representation-level coarse supervision 可能改善 lesion evidence 并为 refinement 提供位置 context | coarse head 不是另一个完整网络；单独监督收益与 refinement 收益分开 |
| H5 | late zero-init residual refinement 可能改善 delineation；稳定后启用可能比全程启用更稳定 | phase scheduling 是独立训练策略，不能从一次 run 估计训练方差 |

## 4. V1 evidence boundary

[V1 Findings](../v1/Findings_V1.md) 只约束其 shallow stems → 24→3 → pre-backbone conditioned residual
结构及对应预算。更早 gate 属于 LEGACY。V1 short-budget negative evidence 不能推出
multi-encoder 无效、anatomy 无效、adaptive fusion 无效，也不能推出 V2 必然有效。
V1 与 V2 处于不同 hypothesis space；版本、capacity、训练预算与研究机制须明确区分。

## 5. V2 Architecture

对模态 m∈{T2W, ADC, HBV}、encoder 尺度 s：

\[
F_m^s=E_m^s(X_m),\qquad A^s=E_A^s(P_{WG},P_{PZ},P_{TZ}).
\]

三个 encoder 参数独立，全部取自相同 nnU-Net plans 的原生实现。
当前concat家族所有尺度使用相同可学习投影Φ_s；selected deeper scales的C/D使用soft policy：

\[
z^s=G_s(F_{T2}^s,F_{ADC}^s,F_{HBV}^s,A^s),\quad
\alpha^s=\operatorname{softmax}_m(z^s),\quad
F_{fused}^s=\Phi_s(\operatorname{concat}_m(3\alpha_m^s F_m^s)),\quad\sum_m\alpha_m^s=1.
\]

Anatomy tells the network **HOW to fuse**。WG 倾向于 spatial context，PZ/TZ 倾向于 zonal context；
其具体系数由模型学习。Anatomy 仅进入 controller，不是第四个 MRI encoder，也不硬遮挡影像。
浅层直接concat投影；B所有尺度无policy。共享原生decoder产生base lesion logits q_base。
下面为保留的coarse-to-fine候选扩展，当前concat家族不启用。原候选fused bottleneck产生：

\[
P_{coarse}=\sigma(H_c(F_{fused}^{L})).
\]

首版实际采用等价的 logit correction（不改写原生 decoder）：

\[
c=\operatorname{stopgrad}\{\max(P_{base},\operatorname{up}(P_{coarse}))\},\quad
\Delta q=R(F_{T2}^0,F_{ADC}^0,F_{HBV}^0,q_{base},A,c),\quad
q_{final}=q_{base}+c\Delta q.
\]

这里 P_base 为 binary softmax 的 lesion channel。R 是局部卷积，末层零初始化，
初始 q_final=q_base；c 是 soft spatial update weight，不是 hard ROI。
Lesion evidence tells the network **WHERE to refine**。漏检区域若 base/coarse 均低，
修正可能受抑制，这是需要验证的失败模式。

## 6. Innovation boundary

不声称首次 multi-encoder、adaptive fusion、anatomy-aware PCa segmentation、coarse-to-fine 或 refinement。
拟研究差异是 predicted zonal anatomy + hierarchical modality-specific representation +
anatomy-conditioned adaptive调节是否超过相同可学习concat融合；late local refinement仅为保留扩展。
novelty claim 必须在后续系统文献调研后冻结。

参考 [Align then Refine](https://arxiv.org/abs/2604.18713) 的多编码器、辅助监督与末端修正思想；
作者使用 text semantic alignment / confidence-gated text cross-attention。
本项目独立组合原生网络，当前以anatomy控制concat前的融合调节；lesion probability修正仅为保留候选。
未复制作者网络，不导入其 fork，不声称复现或把作者结果作为本项目 baseline。

## 7. Predicted Anatomy

Stage-1 是共享辅助先验生成器，不是主要创新。正式 lesion 模型只使用 soft predicted WG/PZ/TZ，
解剖模型不读 lesion GT；validation/test prior 必须来自该病例自身 MRI 的预测并核对训练范围。
GT anatomy 只允许显式 ORACLE_GT upper-bound analysis，不能充当正式模型结果。

保留 IN_SAMPLE_PRED / OOF_PRED / HELD_OUT_PRED / EXTERNAL_PRED 与 checkpoint、split、geometry。
in-sample 不是 OOF，可能存在先验质量 train/test shift；支持 OOF metadata 不等于已完成 OOF 实验。
Dataset608、equivalence audit 和 Stage-1 历史产物继续共享且只读复用；硬 export 的 WG 失败不能替代 soft head 诊断。

## 8. Proposed contributions — pending evidence

1. **Anatomy-Conditioned Hierarchical Multimodal Fusion**：患者自身预测分区 anatomy 控制深层局部模态融合。
保留候选扩展：**Lesion-Aware Coarse-to-Fine Local Refinement**；当前核心研究不启用，
也不将其计入必须验证的贡献。

phase scheduling、评价、权重可视化、采样、资源审计属于 training / analysis / support，
不包装成额外方法创新。不把上述候选贡献写成已验证贡献。

## 9. Failure modes

重点检查 complete miss、small lesion miss、incomplete lesion coverage、FP、over-segmentation、
boundary quality。提高 sensitivity 时必须同时报告 FP 与 precision，不能用预测膨胀替代质量改善。
只在 matched lesions 上改善 Dice 可能来自匹配子集变化，必须同时看 sensitivity。

## 10. Mechanism analysis

分析 alpha_T2 / alpha_ADC / alpha_HBV 在 lesion/background、WG/PZ/TZ、small/large lesion
中的分布。预测 anatomy 可重叠，区域口径引用共享协议并明确离线标注来源。
fusion coefficients ≠ causal clinical importance；不强求符合 PI-RADS，也不据权重作医学因果解释。
初版导出只保证 patch-grid coefficients；全体积滑窗权重汇总尚未实现。

## 11. Training strategy

当前concat三组从头全程启用，共享原生优化策略；policy初始等权，预设机制本身的增量为待检验问题。
以下阶段训练仅为原候选扩展，不作为当前核心研究必要策略：

V2-P1 先稳定独立 encoders + deterministic fusion + shared decoder；V2-P2 开启 anatomy-conditioned
fusion 与 coarse supervision；V2-P3 开启 refiner。精确边界在 Method 与 Experiment Plan 同步记录，
不从 validation 表现自适应选择 phase。首版全部沿用原生 optimizer/LR，不宣称 freeze/unfreeze 已被验证。
全程启用与 scheduled 是不同 variant，可通过受控对照检验 H5。

## 12. Evaluation hierarchy

Primary：positive-case macro Dice。Key secondary：lesion sensitivity、small-lesion sensitivity、
matched-lesion Dice、complete miss、voxel recall/precision、FP burden、HD95、ASSD、NSD。
主表展示的Dice_all为单独的全病例兼容口径，不替代研究Primary；定义与边界只见共享Evaluation Protocol。
大小分层与 NSD tolerance 是否正式/探索性，严格沿用共享 [Evaluation Protocol](../Evaluation_Protocol.md)。
不重复指标定义、匹配规则或空值语义；CI 跨 0 不代表等效或证明无效。

## 13. Controlled comparisons

当前为共享early-fusion → multi-encoder concat → image-adaptive调节 → anatomy-conditioned调节。
三者保持相同可学习投影，以区分普通融合、图像自适应与解剖条件的作用。
均值及coarse/refine/scheduled仅保留候选，不扩大当前核心实验；具体variant/trainer与预算只见Experiment Plan。
每一条边只改变一个明确机制；容量变化如实记录，不把 architecture 对照称为参数匹配实验。

## 14. Reproducibility

冻结 seed/fold/paired case set 与训练预算；最终候选重复多 seed，病例 bootstrap 不覆盖训练间变异。
保留 plans、split、先验 provenance、环境与版本，checkpoint 通过明确 V2 Trainer 重建。
显式 seed 改善可重复性，但多进程增强不保证 bitwise deterministic。

## 15. Stop rules and limitations

若concat multi-encoder相对strong baseline明显变差，先诊断representation、capacity、optimization，
停止堆 anatomy/refiner；若 anatomy 无稳定增量，停止增加 anatomy complexity；
若 refiner 无改善或恶化，保留 negative evidence，不追加 attention depth。
运行判据见 Experiment Plan；进入主模型至少回答减少漏检、改善覆盖、减少假阳之一并报告代价。
三 encoder 显存、prior shift、小样本重复预算、coarse suppression 都是明确限制。
