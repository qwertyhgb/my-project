# Research Plan

## 面向前列腺癌病灶分割的序列特异浅层特征融合与解剖条件建模

**英文工作标题：** *Sequence-Specific Shallow Feature Fusion with Anatomical Conditioning for Prostate Cancer Lesion Segmentation*

## 1. 研究范围

本研究以 nnU-Net 为统一分割基础，考察 T2W、ADC、HBV 三序列 MRI 在病灶分割前如何形成和融合局部表征，以及 PZ/TZ 是否能为融合提供有用的解剖条件。研究对象是**病灶分割**，不是 PI-CAI challenge 的病灶检测任务。标题是工作标题，不预设解剖条件一定有效。

项目已有的输入级门控研究提供了方法选择的初步证据；下一阶段的论文主线是**序列特异浅层表征 → 特征级自适应融合 → 解剖条件融合**。已发生的训练、验证、配对比较和具体数值以 [Findings](Findings.md)、[Training Log](Training_Log.md) 及各实验记录为准，本文不重写运行日志。

研究检验可测量的收益，而不保证正结果。无增益、收益伴随过高假阳代价，或解剖条件未获支持，都是需要如实报告的研究结果。

新增候选研究方向是**同区参照引导的多模态残差融合**：在序列特异浅层表示之上，显式构造
患者自身的分区条件局部背景对比，再检验它对融合的增量作用。该候选不从历史输入级结果推导
正结论，保留既有研究链作为相关假设和历史记录。实现与运行状态分别记录在 Development Log
与 Training Log，本文仅描述科学问题、机制和评价边界。

## 2. 医学与成像学动机

T2W 描述腺体形态和组织背景；ADC 与高 b 值扩散加权像 HBV 从不同角度呈现扩散受限。三者互补，但病灶常较小、边界模糊，信号表现与周围组织随病例和序列而变。仅把三幅图作为三个原始通道交给分割骨干，并未明确控制融合发生在何种表征层级。

PZ 与 TZ 的正常背景和常见良性改变不同。[PI-RADS v2.1](https://cs.acr.org/-/media/ACR/Files/RADS/PI-RADS/PIRADS-V2-1.pdf) 提供了区域相关判读的临床动机，但其规则不能直接改写成体素级固定序列权重。分区可能跨病灶、配准不准，自动生成的 [PI-CAI 分区资源](https://doi.org/10.5281/zenodo.7615350)也不是病灶真值。若分区来自同一 MRI，可记为 $Z=f(X)$；它不必增加 MRI 之外的信息，却可能在有限样本和有限模型容量下提供**结构化归纳偏置**。

本研究的优先失败模式是完全漏分、小病灶覆盖不足与检出后只分出病灶核心。分析必须同时检查阴性病例假阳，避免通过扩大预测范围换取表面上的召回改善。

方法定位也受到已有研究约束：[nnU-Net](https://doi.org/10.1038/s41592-020-01008-z)及
[nnU-Net Revisited](https://papers.miccai.org/miccai-2024/562-Paper2847.html)强调充分配置的
分割基础与严谨对照；[Jiang 等的多流融合编码器](https://doi.org/10.1002/mp.16374)和
[NaMa](https://doi.org/10.1609/aaai.v38i5.28215)已经研究多序列特征融合或邻域感知机制。
因此本研究不声称首次提出序列自适应融合，而是在统一 nnU-Net 和前景感知训练条件下检验一个
**更轻量、信息路径受限**的浅层特征方案及 PZ/TZ 增量。

## 3. 初步实验的证据与研究缺口

既有 nnU-Net 实验包含普通三序列融合、输入级 MRI 门控和 PZ/TZ 条件门控。阳性病例感知采样缓解了稀疏病灶训练时的前景学习延迟，使后续模型处于可学习的共同训练条件。两组在该条件下匹配的输入级门控比较，均**未观察到明确的主指标增量**；其配对区间不能证明无效或等效，也没有覆盖训练随机性。具体指标和边界见 Findings。

输入级门控在每个位置直接执行 $S_m(p)X_m(p)$。1×1×1 卷积直接读取当前位置；归一化虽引入 patch 统计依赖，仍不能被解释为显式建模局部纹理、边界、周围组织和序列特异病灶模式。上述结果**提示**原始强度重加权可能受表征能力限制，因此有理由检验浅层局部表征；它没有证明这一限制是负结果的唯一原因，更没有证明特征级方法一定有效。

旧输入级条件保留为 **preliminary representation-level baselines**：它们回答原始强度重加权在已测试训练条件下能否提供可辨认的收益，并构成升级到特征级融合的研究动机。历史实验事实和顺序不因计划更新而改写。

## 4. 方法框架

### 4.1 序列特异浅层局部表征

设 $X=(X_1,X_2,X_3)$ 分别对应 T2W、ADC、HBV。三个参数互不共享的浅层 stem 产生

$$
H_m=E_{\phi_m}(X_m),\qquad m\in\{1,2,3\}.
$$

每个 stem 以局部 3×3×3 卷积形成序列特异表示，保持空间分辨率。名义卷积感受野不等于完整有效感受野；instance normalization 还引入 patch 级统计依赖。无门控条件拼接三个 $H_m$，经轻量投影 $P_\psi$ 回到骨干所需通道：

$$
U_0=P_\psi\bigl(\operatorname{Concat}(H_1,H_2,H_3)\bigr),\qquad
\widehat Y_0=F_\theta(U_0).
$$

分割骨干 $F_\theta$ 继续由 nnU-Net plans 构建；不重新实现训练循环、验证器或滑窗推理。

### 4.2 特征级自适应融合

影像条件门控读取拼接后的浅层特征，输出三个空间 logit。令

$$
W_m(p)=\operatorname{softmax}_m\!\left(A_{\eta}(H)(p)\right),\qquad
S_m(p)=3W_m(p),\qquad
U_1=P_\psi\bigl(\operatorname{Concat}(S_1H_1,S_2H_2,S_3H_3)\bigr).
$$

每个序列的一个尺度场作用于该 stem 的全部特征通道。末层零初始化时 $S_m\equiv1$，因此在**共享相同 stem、投影和骨干权重**时，门控路径初始输出与无门控路径一致；这不意味着它与直接输入原始 MRI 的 nnU-Net 一致。$W$ 是内部相对尺度，不是校准可靠性、临床重要性或因果贡献。

### 4.3 解剖条件特征融合

令 $Z=(Z_{PZ},Z_{TZ})$。解剖条件门控把 $\operatorname{Clip}(Z,0,1)$ 与 $H$ 拼接后生成 $S_m(H,Z)$。PZ/TZ **只进入 feature gate**，不进入各 stem、投影层或分割骨干：

$$
U_2=P_\psi\bigl(\operatorname{Concat}(S_1(H,Z)H_1,S_2(H,Z)H_2,S_3(H,Z)H_3)\bigr),
\qquad \widehat Y_2=F_\theta(U_2).
$$

该路径限制了直接输入渠道，但尺度场仍可能间接传递位置信息，因此“只进门控”不等于消除了捷径。MRI-only 与 zonal 数据版本的前三 MRI 通道和有效标签必须先完成逐数组一致性审计；PZ/TZ 本身以及附加参数量的差异仍要在解释中披露。

### 4.4 同区参照引导的残差融合（新增候选）

对每个 MRI 序列，在预测分区支持的局部物理邻域内汇聚特征一阶、二阶矩，形成分区条件
中心特征与背景参照的标准化差异。降采样前分别汇聚加权矩，避免不同区域先混合再估计背景。
区域权重是自动标注的占据率；它不等价于后验置信度。参照可以含病灶和良性改变，不能称为
健康组织参照，也不能通过训练病灶真值剔除病灶后再作为网络输入。

令普通融合为 $F_0$，基于原始特征、同区对比与分区支持量学习修正 $\Delta F$，则
$F=F_0+a\Delta F$。固定条件中的 $a$ 由有效参照与分区占据率决定，自适应条件再引入
学习式修正强度。coarse 统计位置无足够局部参照支持时，修正为零；上采样后再通过原分辨率
占据率约束分区外修正。这不推断病例或病灶身份。
零初始化仅保证共享基础权重时的初始行为一致，不能证明优化轨迹或最终效果相同。

检验两个假设：同区对比是否超越普通分区条件门控；学习修正强度是否改善分区误差下的
任务表现。若有候选收益，进一步使用相同邻域、但不区分解剖分区的参照来识别普通局部对比
与区域条件各自的作用。容量、邻域范围和新增算子也是需要报告的比较边界。

此历史融合候选初版只改变融合，保持共同训练条件和原生骨干；完整腺体/分区预测属于
后续两阶段候选的支撑模块，ROI 定位仍是独立受控问题。修正强度不是校准可靠性；采用局部参照也不保证解决序列错位。
仍沿用本计划的阳性病例主终点、病灶实例与阴性假阳指标，并补充实际计算开销和预先固定的
分区扰动下性能变化。滑窗边界、区域参照污染和小区域支持不足均为潜在失败模式。

## 5. 新的核心研究问题

**RQ1：序列特异浅层表征是否有用？** 在统一阳性采样条件下，比较普通三序列 nnU-Net 融合与无门控浅层特征融合。假设 H1：先学习局部序列特异表示可能改善病灶分割，尤其是完全漏分与覆盖不足。这个比较同时改变 stems、投影、额外参数及骨干输入分布；若有收益，只能归因于**整条浅层表征路径**，不能归因于单个 stem。

**RQ2：特征级自适应融合是否带来增量？** 在共享相同 stem、投影、骨干和训练条件下，比较无门控与 MRI 条件 feature gate。假设 H2：自适应特征尺度可能改善病灶覆盖及小病灶敏感度。只有同层级比较支持增量，才能把门控列为方法贡献。

**RQ3：PZ/TZ 条件是否进一步有用？** 在相同特征级门控形式下，比较只读 MRI 的 gate 与额外读取 PZ/TZ 的 gate。假设 H3：分区可能作为结构化归纳偏置改善序列融合，但无需包含 MRI 外的新病灶信息，也不规定 PZ/TZ 固定偏好哪一序列。若无明确增益，应报告“在所测试形式下未获支持”。

这三个问题只针对**尚待测试的特征级主线**。既有输入级 image gate 与 anatomy gate 的问题和结论保留为 preliminary findings，不再与新 RQ1–RQ3 共用编号。

## 6. 最小主线与受控比较

| 条件 | 模型 | 该比较检验什么 |
|---|---|---|
| A | `positive_sampling` | 统一训练条件下的普通三序列参照 |
| B | `feature_no_gate_positive_sampling` | A→B：浅层表征路径整体效用 |
| C | `feature_image_gate_positive_sampling` | B→C：同一特征层级的自适应门控增量 |
| D | `feature_anatomy_gate_positive_sampling` | C→D：PZ/TZ 作为门控条件的增量 |

Positive-case-aware patch sampling 是所有主条件共享的 **foreground-aware training stabilization strategy**，用于缓解病灶稀疏造成的前景梯度稀释；它不是本论文的主要方法创新。Focal+CE、split、增强、优化器、调度、训练预算与完整病例验证口径应保持一致。A→B 改变整个前端路径；B→C 主要改变 feature gate；C→D 除解剖条件外还涉及数据集配置与少量门控参数，不能声称严格只差 PZ/TZ。

原生采样 baseline、旧 image gate，以及阳性采样下的 input image/anatomy gate，均保留为 preliminary/supporting experiments。它们不能替代 A→B→C→D 的特征级匹配比较。任何历史数值、checkpoint 与实验顺序均以原始记录为准。

## 7. 评价终点与小病灶分析

### 7.1 主终点和病例级配套指标

主终点仍是**参考病灶阳性病例的宏平均 Dice**。所有阳性病例纳入；完全无重叠者记 0；阴性真空预测不赋 1 混入均值。设阳性病例集合为 $\mathcal I_+$，则

$$
\operatorname{Dice}_{\mathrm{macro}+}
=\frac1{|\mathcal I_+|}\sum_{i\in\mathcal I_+}
\frac{2TP_i}{2TP_i+FP_i+FN_i}.
$$

主要配套指标为阳性病例 micro Dice、阳性体素召回和 precision、完全漏分率、阴性病例假阳率与假阳体素负担。报告逐病例配对变化及置信区间，并列展示收益和代价。训练时 patch pseudo Dice 只用于优化监控。

### 7.2 病灶实例级关键次要终点

预先把 **lesion-level sensitivity、小病灶敏感度及 matched-lesion Dice** 纳入关键次要分析。参考病灶实例由三维连通域定义。具体协议已在看到新模型结果之前冻结（冻结时点与实现见 Development Log）：

- **实例定义**：reference 与 prediction lesion 实例均为 3D 连通域，按 6-邻域（face connectivity，`CONNECTIVITY = 1`）判定；仅通过角或边接触的体素不属于同一 lesion。
- **候选匹配**：reference 与 prediction 实例至少有一个真实重叠体素（intersection ≥ 1）才构成候选边。
- **一对一分配**：一个实例至多被匹配一次，按字典序目标求解——先最大化被匹配的 reference 实例数（maximum cardinality），再最大化全部匹配对的总 intersection 体素数，仍完全相同时按 reference component id、prediction component id 升序给出确定性唯一解。配对不对 Dice 最大化：matched-lesion Dice 是报告终点本身，先据其选配会引入 metric-optimizing-the-metric；Dice 只用于匹配后的质量评价。
- **主指标命名**：实例敏感度一律写成 **any-overlap lesion sensitivity**（`lesion_sensitivity_any_overlap` = matched reference lesions / all reference lesions）。未匹配（missed）的 reference lesion 不从分母删除；reference 实例数为 0 的病例不进入分母，其 prediction components 仍计入假阳分析。
- **分层**：探索性区间按**每个参考实例**的物理体积（逐例 spacing 换算 mm³）：`< 500 mm³`、`500 ≤ V ≤ 1000 mm³`、`> 1000 mm³`，边界严格按此判定；不能用整例阳性体素总数代替。这些是**探索性分层，不是临床类别**，若据文献或数据质量调整须记录原因和时点。多病灶病例逐实例处理；小组样本稀少时只作描述性解释。
- **质量指标**：matched-lesion Dice 只在 matched pairs 上报告，天然排除全部 missed lesions，不得描述为 overall lesion Dice；报告中与 any-overlap sensitivity / matched reference fraction 并列。

病灶实例敏感度衡量分割失败模式，**不构成 PI-CAI challenge 的病灶检测 benchmark**；现有按病例体素数分箱的结果不能冒充该分析。

HD95、NSD 等边界指标可作少量补充，需预先固定容差和空预测约定。AUROC、AP、FROC 与 PI-CAI challenge score 不属于本研究主终点。

## 8. 重复性与确认策略

现有输入级实验均为单次独立训练。病例 bootstrap 只刻画**固定 checkpoint 下病例重采样不确定性**，不包含初始化、增强、采样和优化的 run-to-run 波动。

探索阶段对 A/B/C/D 各观察一次，依据主终点、实例级失败模式与假阳代价筛选方向；这只能产生候选结论。若出现有意义的候选增益，确认阶段只对**最终候选与其匹配参照**做 3 次独立显式种子训练，保持相同 split、预算和评价，报告各 run、均值±标准差以及逐病例配对分析。设置种子前须检查现有入口能否完整控制初始化、增强和采样随机性；不能把“传入了 seed”写成完全确定性保证。

所有模型、checkpoint、阈值和后处理选择应在内部验证中完成并冻结。病例 bootstrap 与跨训练 run 方差分开报告。若资源不足以完成确认，论文结论明确限定为探索性。

## 9. 外部验证与分布变化

对内部验证已选定并冻结的最终模型，Prostate158 可以作为**不微调、不重训**的外部压力测试。训练第三序列为 HBV，Prostate158 第三序列为 DWI；两者不能默认为同质输入，结果应解释为跨域压力测试，不能简单归因于融合机制。

若最终模型依赖 PZ/TZ，外测前还必须具备独立、冻结、可复现且不使用测试病灶标签的分区生成流程。若无法可靠取得 PZ/TZ，anatomy 模型与 MRI-only 模型就不能进行完全等价的外部对照；须将此写为限制。

Prostate158 不参与模型、checkpoint、概率阈值或后处理选择。看见其结果后改变任一选择，会使该集合变为开发集，原外测结论失效。

## 10. 停止规则与范围控制

1. 若 B 相对 A 在主终点和关键失败模式上没有有意义的收益，不以加深 stem 或扩大容量作为自动补救；先报告浅层路径在测试形式下的结果。
2. 若 C 相对 B 无明确收益，feature gate 不进入最终方法贡献；不通过更换 Transformer、Mamba 或复杂注意力追逐单次正结果。
3. 若 D 相对 C 无明确收益，停止扩张 anatomy-conditioned architecture；报告 PZ/TZ 条件在测试形式下未获支持，除非出现新的独立证据或研究问题。

“有意义”需要结合宏平均 Dice、完全漏分、实例敏感度和阴性假阳的共同方向判断；CI 跨 0 不能证明等效。停止规则控制**模型扩张**，不阻止完成必要的误差分析和如实报告负结果。

历史特征融合路线将多任务分区分割排除在其单变量主线之外；这一边界不再覆盖当前两阶段
候选方向。当前阶段一的 T2W 联合 WG/PZ/TZ 解剖伪监督是提供预测软信息的支撑模块，
仅联合三个区域不能构成创新声明。阶段二候选以完整视野与预测分区软信息研究方法收益，
患者隔离的预测是必要边界；解剖伪监督一致性不能代替人工解剖准确率或病灶终点。
WG hard masking、zone-only、复杂反事实、全面门控解释、缺失序列研究和完整多流编码器
仍不作为本次候选扩展。

## 11. 限制与解释边界

MRI 配准、自动分区质量及不同病例的成像差异会影响融合。输入或特征门控产生的图仅表示模型内部相对尺度；它们不是序列真实质量、校准概率、临床可靠性、PI-RADS 判读权重或对预测的唯一因果贡献。滑窗窗口和归一化统计也可能改变展示图的形态。

病例级“有任意重叠”不是病灶实例敏感度；实例级敏感度也不是完整检测挑战分数。一次训练、一个内部划分与跨域外测均限制泛化表述。参数离开零初始化不能替代任务终点改善。

## 12. 预期贡献

1. 在统一 nnU-Net 与 foreground-aware training protocol 下，检验原始输入重加权和浅层特征融合的差异。
2. 构建并检验轻量序列特异浅层表示与特征级自适应融合对稀疏病灶分割的价值。
3. 既有分区门控将 PZ/TZ 路径限定于 feature gate；新增候选将其用于同区参照与融合修正，
   检验区域背景对比和可调修正各自的增量价值。
4. 同时刻画完全漏分、病灶实例及小病灶敏感度、覆盖不足与假阳代价。
5. 对有希望的最终候选进行独立种子确认，并在条件允许时开展冻结模型的跨域压力测试。

上述都是**待检验的预期贡献**。即使最终解剖条件未获支持，强训练参照、特征融合对照、负结果与小病灶误差分析仍构成完整的研究证据。

## 13. 参考研究

1. Turkbey B, Rosenkrantz AB, Haider MA, et al. **PI-RADS v2.1.** *European Urology*, 2019. [ACR PDF](https://cs.acr.org/-/media/ACR/Files/RADS/PI-RADS/PIRADS-V2-1.pdf)
2. Isensee F, Jaeger PF, Kohl SAA, Petersen J, Maier-Hein KH. **nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation.** *Nature Methods*, 2021. [DOI](https://doi.org/10.1038/s41592-020-01008-z)
3. Isensee F, Wald T, Ulrich C, et al. **nnU-Net Revisited: A Call for Rigorous Validation in 3D Medical Image Segmentation.** 2024. [arXiv](https://arxiv.org/abs/2404.09556)
4. Saha A, Bosma JS, Twilt JJ, et al. **Artificial intelligence and radiologists in prostate cancer detection on MRI (PI-CAI).** *The Lancet Oncology*, 2024. [DOI](https://doi.org/10.1016/S1470-2045(24)00220-1)
5. Jiang M, Yuan B, Kou W, et al. **Prostate cancer segmentation from MRI by a multistream fusion encoder.** *Medical Physics*, 2023. [DOI](https://doi.org/10.1002/mp.16374)
6. Meng R, Zhang X, Huang S, et al. **NaMa: Neighbor-Aware Multi-Modal Adaptive Learning for Prostate Tumor Segmentation on Anisotropic MR Images.** *AAAI*, 2024. [DOI](https://doi.org/10.1609/aaai.v38i5.28215)
7. Karagoz A, Alis D, Seker ME, et al. **Anatomically guided self-adapting deep neural network for clinically significant prostate cancer detection on bi-parametric MRI.** 2023. [DOI](https://doi.org/10.1186/s13244-023-01439-0)
8. Karagöz A, Şeker ME, Yergin M, et al. **The PI-CAI Challenge: Zonal Segmentation for Prostate MRI.** Dataset, 2023. [DOI](https://doi.org/10.5281/zenodo.7615350)

## 14. 一句话总结

> 本研究在统一的 foreground-aware nnU-Net 训练条件下，检验三序列 MRI 是否应先形成序列特异的浅层局部表征再自适应融合，并进一步研究 PZ/TZ 能否作为结构化归纳偏置改善特征级融合，重点关注完全漏分、小病灶敏感度与病灶覆盖不足。


## 联合解剖伪监督与完整视野研究边界

阶段一研究共享 T2W 分割网络同时学习独立 WG、PZ、TZ 软预测的可行性。
WG 为独立算法伪标签；PZ/TZ 来自 zonal_yuan。Yuan 是工程复用的首版选择，
不代表较 Hevi 更准确。不同算法的监督不可预设严格包含关系；不将分歧组合命名为
新的正常解剖类别，不用分区并集替换 WG。位组合仅是无损监督存储机制，
原生区域目标与三个 sigmoid 通道保持独立，不施加互斥或包含约束。

首版阶段二采用完整 FOV 与阶段一预测分区软信息。两阶段研究中，算法标签覆盖
与阶段一模型预测覆盖是不同证据。5 mm ROI 硬裁剪暂不实施；ROI 留作独立受控
消融，参数选择仅使用训练侧，不能依据外层验证结果反复调整。
阶段一缺失区域不能作为空掩膜监督；折外软信息必须来自未见过该患者的模型。

预定义解剖报告口径：原始 T2W 网格上三个独立概率 >0.5 的区域 Dice，分别对
原始算法监督计算；报告区域重叠与相互包含违例比例。原生顺序覆盖整数导出的
区域指标单独命名，不能代替独立头指标。软信息不自动等于校准置信度。
