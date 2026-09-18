# 研究计划：基于解剖分区条件自适应多序列融合的前列腺癌病灶分割

> 英文工作标题：**Anatomy-Conditioned Adaptive Multi-sequence Fusion for Prostate Cancer Lesion Segmentation**
> 版本：**v2.3.1** · 2026-09-17
> 主项目目录：`/opt/data/private/lm/my-projects`；主数据目录：`/opt/data/private/lm/data/Prostate/PI-CAI`
>
> **本文档是研究协议（计划）**：定义研究问题、数据与独立测试设计、模型与消融、冻结的训练/评价/统计协议、
> 阶段门的定义与失败转向。**本文档不维护任何实时状态，也不记录运行事实。**

## 0. 版本、文档边界与阅读约定

### 0.1 版本摘要（当前 v2.3.1）

**v2.3.1（2026-09-17，amendment，发生在任何正式模型结果之前）**：本轮文档边界重构过程中**新增了四项
实质性协议要求**，按 SPIRIT 2025 的 amendment 纪律单独升版登记（完整记录见 `docs/protocol_changelog.md`）。
**因此本轮不是「仅文档迁移」**——下列条款会改变实验设计与结论强度，必须按 v2.3.1 执行：

1. **G0-SAP 三阶段生命周期（SAP-A / SAP-B / SAP-C）**：把「阈值与后处理选择」从单一的「冻结前」拆分为
   ① 看模型结果前冻结**方法与候选集**（SAP-A）→ ② checkpoint 冻结后在 PI-CAI validation 上按既定算法
   **一次性填数**（SAP-B，允许验证预测可见）→ ③ 封存全部配置与哈希后对最终 test **一次性评估**（SAP-C）；
   `sees_validation_predictions` 与 `sees_final_test_predictions` 分开记录（§11.4、`docs/protocols/G0_SAP.md`）；
2. **最小实际/临床相关差异（MCID）与主判定规则**：成功标准不得只写「效应为正且 CI 不跨 0」，必须给出
   CI 相对 MCID 的位置；无法确立 MCID 时只能声明「分割精度增益」（§3.1）；
3. **M1/M2 容量混杂控制**：必须预先声明同容量对照，或把 H2/G3 显式降级为探索性证据（§8.8）；
4. **G0-R 证据充分性决策与系统文献检索要求**：16 例自动相对指标不足以单独支撑配准决策，须由研究者显式决定
   是否追加人工复核或扩大抽样（§4.2.1）；新颖性表述必须以系统检索与截止日期为前提，不得使用「首次」（§2.2）。

**v2.3（2026-09-16）**：不扩张研究主线，也不把成熟 U-Net 变体包装成创新；只做两件事：

1. **统一 M0–M4 的骨干**：正式 M0–M4 的共同骨干由 PlainConv 3D U-Net 升级为
   **plan-driven Residual-Encoder 3D U-Net**（§7.2/§7.3），继续沿用冻结 `3d_fullres` 的 spacing、patch、
   kernel、stride、通道上限、深度监督与推理几何；目标复杂度以 nnU-Net ResEnc-M 级别为起点，不采用 L/XL；
2. **冻结浅层 skip 与主模型边界**：stage 0/1 的三路 modality-specific skip 固定为
   `concat → 1×1×1 projection → InstanceNorm → LeakyReLU` 聚合（§8.6），stage 2 skip 使用 `F_fuse`，
   stage 3 及以后使用共享 encoder skip；主模型不加入 anatomy-conditioned skip、Attention U-Net gate、
   UNet++/UNet3+ 密集 skip、Transformer/Mamba bottleneck 或多尺度序列 gate。

研究边界是：**用成熟残差 U-Net 保证后端能力，用单一、可归因的 PZ/TZ 条件序列 gate 承担创新**。
完整版本变更史（v2.0 → v2.1 → v2.2 → v2.3 → v2.3.1，含修改原因与对既有结果的影响）见
`docs/protocol_changelog.md`；当前实现与门的实时状态见 `docs/STATUS.md`。

### 0.2 文档边界：每类事实的唯一权威来源

| 事实类型 | 唯一权威来源 | 本文档的职责 |
|---|---|---|
| 研究协议条款、假设、设计、冻结规则、门的定义 | **`docs/research_plan.md`（本文档）** | 只写“计划做什么、如何判定” |
| 当前状态（门状态、实现状态、blocker、下一步） | **`docs/STATUS.md`** | 不重复维护任何实时状态 |
| 编号与步骤的释义（`G0-*`、`G1`–`G5`、`P0`–`P10`、`M0`–`M4`、`N0`、`H0`–`H4`、`D`、`B`、`SAP-A/B/C`） | **`docs/GLOSSARY.md`** | 出现编号时直接引用，不重复解释（注意 `H0` 是模型对照、`H4` 才是假设） |
| 协议版本变更史（含 amendment 与影响） | **`docs/protocol_changelog.md`** | 只保留简短版本摘要 |
| 实际运行过的命令、环境、耗时、结果、失败、产物 | **`docs/experiment_log.md`** | 不记录任何运行事实 |
| 代码/配置/协议实现变更 | **`docs/Development_Log.md`** | 不记录工程实现过程 |
| 用户执行的完整命令、输出路径、成功判据 | **`docs/runbooks/`** | 只引用入口，不复述 shell 命令 |
| 代理运行权限、conda 环境、进度条规则 | **`AGENTS.md`** | 不复述代理规则 |
| 具体实现与验收细节 | `docs/P2_M0_Implementation.md`（legacy PlainConv，含附录 A）、`docs/P2_M0_ResidualEncoder_v23.md`（v2.3 正式 M0） | 只引用 |
| G0-R / G0-E / G0-SAP 的详细协议 | `docs/protocols/` + `configs/protocols/` | 只定义与这些协议的关系 |

**维护纪律**：本文档中的表格若与 `docs/STATUS.md` 冲突，以 `docs/STATUS.md` 为准；
若与 `docs/protocols/` 的冻结版本冲突，以冻结协议为准并立即修订本文档。

### 0.3 研究主线

本研究只回答一个问题：

> **PZ/TZ 解剖分区能否显式指导 T2W、ADC、HBV 的空间自适应融合，从而改善前列腺癌病灶分割？**

本研究只设计一个核心创新模块：**解剖分区条件融合模块**。论文厚度主要来自严谨的融合对照、分区分层和序列消融，
而不是堆叠多个复杂模块。

数据协议：

- PI-CAI：训练与模型选择；
- 外部数据集（或重划的 PI-CAI internal test）：冻结后的最终测试；
- 不再设置 held-out center，不做 leave-one-center-out（LOCO）；
- 不引入域随机化、中心不变特征、可靠性损失或序列退化训练；
- 外部验证不等于提出了域泛化方法，论文不作 domain generalization 声明。

此前关于残余错位、Patch2Space 式局部对应、可靠性自适应和跨中心域泛化的内容均不再属于本研究主线；
几何差异仅作为数据预处理与质量控制问题（§4.2.1）。

### 0.4 阅读与证据约定（不得违反）

1. **工程完成 ≠ 科学阶段门 PASS**：代码迁移、合成测试通过、工具实现或工具运行成功，都不使任何门变为 PASS；
2. **validation ≠ 独立确认**：PI-CAI validation 结果只用于开发与方向性支持，不得包装为 H3 的独立确认（§13.6）；
3. **gate 权重 ≠ 因果贡献**：融合系数只是结构量，不得解释为临床重要性或对网络信息流的贡献百分比
   （§1.3、§8.3、§12.5）；
4. **待冻结项保持 Pending**：尚未由研究者冻结的关键选择（阈值、seed、预算差异、先验独立性等）在本文档中
   一律标注“待冻结”，不得用假设值填充；集中清单见 §20.2；
5. **不伪造文献结论**：任何“首次/创新”表述必须附来源，且不得仅凭关键词检索成立（§2.2）。

---

## 1. 研究定位与贡献边界

### 1.1 研究任务

本研究使用前列腺 bpMRI 的轴位 T2W、ADC、HBV 三序列，输出临床显著前列腺癌病灶的三维二值概率图。模型重点研究不同 MRI 序列在不同前列腺解剖区域中的互补关系。

### 1.2 核心思想

常规多序列模型通常采用通道拼接或仅根据影像特征进行动态加权。本研究将冻结的 PZ/TZ 解剖先验直接用于条件化融合决策，使网络学习空间位置相关的序列权重：

$$
W(p)=G\big(F_{T2}(p),F_{ADC}(p),F_{HBV}(p),Z_{PZ}(p),Z_{TZ}(p)\big).
$$

这里的“自适应”指权重随影像特征与空间位置变化，不表示模型能够估计真实图像质量，也不表示已经具备域泛化能力。

### 1.3 术语约定

- T2W、ADC、HBV 均属于 MRI，正文统一称为“多序列融合”，不宣称是三个独立数据集；
- HBV 是高 b 值弥散加权图像，外部数据中的 DWI 只有在采集定义核实后才能与 HBV 对应；
- PZ 表示外周带，TZ 表示移行带，WG 表示全腺体；
- “soft fusion”指连续学习的序列权重，不等于输入的是 softmax 解剖概率；
- 本地 PZ/TZ 当前为离散分区标签，只能称为 hard/one-hot anatomical prior；
- gate 权重是网络内部融合系数，不直接等价于临床重要性或因果贡献。

### 1.4 明确不主张的内容

本研究不宣称：

- 首次使用前列腺分区；
- 首次进行动态多序列加权；
- 提出了全新的 U-Net、decoder、skip connection 或 residual encoder；
- 模型学习到的权重严格复现 PI-RADS；
- 模型能够判断真实 MRI 质量；
- 外部测试结果证明了通用域泛化能力。

---

## 2. 科学依据与研究缺口

PI-RADS v2.1 在 PZ 与 TZ 中采用不同的主导序列：PZ 评价主要依赖 DWI，TZ 评价主要依赖 T2W。这为“序列贡献可能具有解剖区域依赖性”提供了医学动机，但 PI-RADS 是诊断评分规则，不能被直接硬编码为病灶分割规则。

相关研究已经表明：

- T2W、ADC、DWI/HBV 具有互补信息；
- 多流编码器能够改善多序列特征交互；
- NaMa 的 MIAL 已基于跨序列相关性进行 subject-specific adaptive fusion，并在 PI-CAI
  上评估；NesMFle 继续发展 modality-informativeness flexible learning；
- LeSMI 已使用 Dynamic Modality Weighting 动态整合 T2W、DWI 与 ADC；因此“学习三个序列权重”
  本身不足以构成创新；
- AtPCa-Net 等方法已将分区知识用于附加输入、分区损失、检测约束或分区专属模型；
- 2025 年已有工作将 zonal awareness 与 Attention U-Net 用于 bpMRI 前列腺癌分割；在更广泛的医学分割中，
  Attention U-Net、UNet++ 和 anatomy-guided decoder/skip 也已充分说明 encoder–decoder 与 skip 可被改造。因此，
  “给 skip 加 attention”或“让解剖先验进入 decoder”本身不作为本研究的新意；
- nnU-Net Revisited 与官方 residual-encoder presets 表明，成熟的 CNN residual U-Net 仍是有竞争力的 3D 医学分割
  骨干。本研究据此把 Residual-Encoder 作为所有 M0–M4 的共同工程底座，而不是新的方法贡献；
- 已有研究比较了按 PZ/TZ 选择不同序列的 hard-zone 策略，这类硬性选序列不能作为本研究的新意；
- 硬性规定 PZ 使用 ADC/HBV、TZ 使用 T2W 可能过度简化真实病灶表现。

本研究的具体缺口限定为：

> **如何在不采用硬路由规则的前提下，让 PZ/TZ 解剖先验显式调节 T2W、ADC、HBV 的空间融合，并通过受控消融证明收益确实来自“解剖条件融合”而非单纯增加输入或模型容量。**

### 2.1 最近邻工作与差异矩阵

| 工作类型 | 影像驱动自适应融合 | PZ/TZ 显式先验 | 先验直接调制序列 gate | 本研究中的角色 |
|---|:---:|:---:|:---:|---|
| 多流/多编码器融合 | 部分 | 否 | 否 | 证明多分支与特征交互已有；由 M1 控制容量 |
| NaMa / NesMFle / LeSMI | 是 | 否 | 否 | M2 的最近邻基线，不作创新主张 |
| anatomy concat / zonal loss | 不是核心 | 是 | 否 | 由 M3 控制“只是增加解剖信息” |
| zonal-aware Attention U-Net / anatomy-guided decoder | 不是序列融合核心 | 是 | 否 | 证明解剖引导分割/skip 已有；主模型不重复包装为创新 |
| hard-zone sequence routing | 否 | 是 | 硬路由 | H0 可选对照，不作最终方法 |
| **本研究 M4** | **是** | **是** | **是，连续条件调制** | 核心待验证缺口 |

论文不主张“首次动态多序列融合”或“首次使用前列腺分区”。方法定位只聚焦于：
**在单个 3D 网络内，使用冻结 PZ/TZ 先验直接条件化空间序列 gate**。

### 2.2 新颖性表述纪律与文献检索状态（2026-09-17 新增）

**前提**：系统性文献检索（数据库、检索式、检索截止日期、逐项机制对照表）属于**待完成事项**——其状态与
阻塞项见 `docs/STATUS.md`（待冻结决策 D10）；已完成的关键词级定点核查记录在
`docs/literature/novelty_and_standards_check_20260917.md`。

因此，在完成系统检索并由研究者确认之前：

- **禁止**出现“首次”“世界首个”“绝对创新”“首次使用分区”等表述；
- 可行表述上限为：**“据我们所知（to our knowledge），尚未有工作以受控消融证明分区条件化空间序列 gate
  相对图像驱动空间 gate 的增益”**，且必须在论文方法/引言中注明检索范围与截止日期；
- 关键邻接工作（NaMa / NesMFle / LeSMI / Z-SSMNet / zonal-aware attention U-Net / 以及模态门控类工作）
  必须逐项列入机制对照表（输入先验类型、融合粒度、骨干、数据、评测口径）；
- 特别提示：邻接的模态门控研究报告了「在卷积 backbone 上门控可能塌缩为近静态模态先验」的现象
  （arXiv:2604.10702 摘要，需人工确认正文），这对 H2 构成直接相关风险，已并入 §19 风险表要求 gate 诊断必报。

检索协议建议与待冻结项见 §20.2 D10。

---

## 3. 研究问题与可证伪假设

### RQ1：三序列是否具有互补性及分区差异？

**H1（探索性）**：T2W、ADC、HBV 单序列与三序列模型的性能不同；在 PZ、TZ 和 mixed 病灶中移除不同序列所造成的性能变化具有不同模式。

H1 不要求结果严格符合 PI-RADS。若未观察到明确的分区差异，应如实报告。

### RQ2：空间动态融合是否优于简单融合？

**H2（基础）**：在相同训练协议下，图像驱动的空间 gate（M2）优于等权多分支融合（M1）和普通通道拼接（M0）。

若 M2 不优于 M1/M0，说明动态融合本身尚未建立，不能继续把 M4 的变化解释为解剖条件融合收益。

### RQ3：分区先验是否应直接条件化融合？

**H3（核心）**：分区条件 gate（M4）优于无分区空间 gate（M2），并优于将 PZ/TZ 仅作为普通网络输入的模型（M3）。

核心证据是 M4−M2 与 M4−M3 的配对性能差异及置信区间，而不是单张权重热图。

### RQ4：软条件融合是否优于硬分区规则？

**H4（次要）**：M4 学习到的连续序列权重优于预设的 hard-zone routing（H0）。

H0 是医学规则启发的附加对照，不是主消融链的必要组成。若实现会引入明显容量或优化差异，则只作为补充实验，不用于核心归因。

### 3.1 最小成功标准

论文的**主要终点**预定为阳性患者的 case-wise Dice；主要比较按层级顺序执行：

1. M4 vs M2；
2. 仅在第 1 项成立后，确认 M4 vs M3。

上述确认性判定必须基于不参与 checkpoint、阈值或超参选择的冻结最终 test。对两个层级比较，要求
预先声明的配对效应估计为正，且双侧 95% CI 不跨 0；PI-CAI validation 上的同类结果只用于开发与方向性支持。
若 G0-E 证明外部数据与 csPCa 任务不等价，则必须在任何正式 M0–M4 结果产生前重新冻结 PI-CAI 内部 test。

**成功标准的当前缺口（待冻结，不得由代理填数）**：现行标准只要求“效应为正且 95% CI 不跨 0”，**尚未声明
最小实际/临床相关差异（MCID / minimal practically relevant difference）**。仅凭“CI 不跨 0”会把统计显著
但实际可忽略的增益写成成功。冻结要求：

1. 在 G0-SAP 中写入预定的 MCID 及其来源（文献范围 / 专家共识 / 盲态 pilot），且必须在查看正式结果前写下；
2. 主判定同时报告点估计、95% CI 与 CI 相对 MCID 的位置（而非只报“是否跨 0”）；
3. 若 MCID 无法确立，结论只能表述为“分割精度增益”，**不得**声称临床相关性；
4. 该项属 §20.2 待冻结决策 D4；未冻结前不得宣布“达到主要成功标准”。

**v2.2 统计口径说明（与 §13.2 一致，不得在结果产生后改写）**：主 CI 是 **conditional on the frozen trained
seeds** 的患者簇配对 bootstrap，只刻画"在已冻结的这 3 个训练 seed 下"的患者间不确定度；3 个 seed 不用于
估计 seed 总体分布，因此**不得**把该 CI 包装成强确认性的 seed-level 区间。全部 seed-specific 效应与跨 seed
均值/SD/范围必须同时报告（见 §13.2）。

论文核心方向成立还需要：

1. 三个冻结种子的效应方向不得由单一极端种子主导，并须报告全部 seed-specific 结果；
2. 增益不能只来自单个病例、单一阈值或事后选择的后处理；
3. PZ/TZ 分层或序列移除实验能提供与总体结果相容的机制证据；
4. 使用另一套分区先验的预定敏感性实验后，结论不能完全崩溃。

若只满足总体性能提升而缺乏机制证据，论文应降低“区域相关序列贡献”的解释强度。

---

## 4. 数据、输入与标签口径

### 4.1 PI-CAI 主开发数据

| 中心 | Study 数 | csPCa 阳性 | 阴性 |
|---|---:|---:|---:|
| RUMC | 800 | 236 | 564 |
| PCNN | 350 | 109 | 241 |
| ZGT | 350 | 80 | 270 |
| 合计 | 1500 | 425 | 1075 |

1500 个 study 来自 1476 名患者。所有划分以 `patient_id` 为单位，同一患者的多次检查不得跨集合。

### 4.2 输入与输出

- 输入：轴位 T2W、ADC、HBV；暂不使用冠状位或矢状位 T2；
- 输出：三维二值病灶概率图；
- PI-CAI 监督目标：csPCa，ISUP ≥ 2 为前景；
- 阴性病例保留为空标签，用于控制假阳性；
- 三序列在物理坐标下统一到冻结 T2W 参考网格后方可融合；
- **重采样只统一网格，不等于完成序列配准**。PI-CAI 公开开发数据的 T2W、ADC、HBV 原则上未配准，
  逐位置融合不得默认三者具有体素级完美对应。

**官方依据（2026-09-17 核对，来源：PI-CAI 官方数据页 `https://pi-cai.grand-challenge.org/DATA/`）**：公开训练与
开发集（1500）**原则上不共配准**，仅 54/9107 训练病例由组织方手动刚性配准（ITK-SNAP，6 DoF）；隐藏调优（100）
与隐藏测试（1000）队列由组织方全部配准（其中 85/1000 为手动）。因此本项目不能把公开训练集的错位水平外推到
PI-CAI 官方排行榜口径，也不能假设「仅物理重采样」等价于官方评测所用的配准数据；该差异必须在论文限制中声明。
其余官方事实核对（HBV 的 b 值定义、ADC 绝对强度不可跨中心比较、许可与 ProstateX 重叠）见
`docs/literature/novelty_and_standards_check_20260917.md` §4。

### 4.2.1 序列错位协议

在任何 run 被视为正式 N0/M0–M4 结果前，必须通过 G0-R：

1. 在不查看模型结果的前提下，对预先选定、覆盖中心/阳阴性/几何异常的病例做序列对齐 QC；
2. 冻结主协议是“仅物理坐标重采样”还是“额外固定刚体配准”，不得根据 M4 表现事后改变；
3. 冻结配准失败、显著错位与 FOV 不足的标记、保留/排除原则；
4. 所有 N0、M0–M4 使用同一主协议；对位移敏感性只作补充实验，不用于事后挑选主结果。

**证据充分性的已知边界（必须与结论一起报告，待冻结项见 §20.2 D2）**：

- 当前自动路径（G0-R-AUTOMATED draft-0.2）使用**跨模态边缘一致性等相对指标**，其绝对数值依赖强度标准化与
  边缘阈值，判定完全依赖**合成已知位移校准**；
- 抽样为 **16 例**（含全部 5 例 `geometry_suspect` 与中心/阳阴性/极端 FOV 分层），不等于全体 1500 例的错位分布；
- 候选决策只有 `resample-only` / `resample+rigid` / `INSUFFICIENT_EVIDENCE` 三种，且**工具不写决策**；
- 因此：**不得**把自动 QC 结果表述为解剖学真值或人工阅片结论；若决策为 `INSUFFICIENT_EVIDENCE` 必须停止，
  不得强行开训；是否追加人工 landmark 复核（历史草案路径仍保留）或扩大抽样，属研究者决定。

### 4.3 canonical 病灶标签

- 1295 例使用 `human_expert/resampled`；
- 另外 205 个阳性例使用 `human_expert/Pooch25`；
- resampled 标签值 2–5 映射为前景 1；
- Pooch25 标签值 1 保持为前景 1；
- 阴性病例保持全 0；
- 原始标签只读，统一二值标签写入派生数据目录。

Pooch25 为补充的人类专家标注，使全部 1500 study 都有专家来源的 csPCa 标签；但它与原 1295 例的生成批次、
标签编码和轮廓风格可能不同。冻结 split 不为此重划，但正式报告必须给出 train/validation/test 中的标注来源计数，
并对阳性病例补充报告 `resampled` 与 `Pooch25` 分层敏感性。

**官方来源核对（2026-09-17，见 `docs/literature/novelty_and_standards_check_20260917.md`）**：
官方标注仓库 `github.com/DIAGNijmegen/picai_labels`（README）确认 Pooch25 位于
`csPCa_lesion_delineations/human_expert/Pooch25/`，是对**此前仅有 AI 标注的 205 例阳性**补充的人类专家标注，
其标签编码为**二分类 0/1**（与 `human_expert/` 的 0/2/3/4/5 多分类不同）；使用该标注须引用
Pooch et al., 2025。本项目「Pooch25 值 1 → 前景 1」的映射与该二分类口径一致，**不得**把两套编码在同一
映射函数中静默混用；混用必须在实现中显式分支并由单元测试锁定。

### 4.4 解剖先验

- WG：Bosma22b AI whole-gland mask，仅用于数据质控、统一评测分层、假阳性分析与单列消融；不得参与 M0–M4 的裁剪、patch 采样或网络输入；
- PZ/TZ 主来源：Yuan23，标签为 `0=背景、1=PZ、2=TZ`；
- PZ/TZ 敏感性来源：HeviAI23；
- 所有病灶模型训练期间解剖先验保持冻结；
- 第一阶段不训练新的 WG/PZ/TZ 网络，不进行病灶—分区多任务联合训练；
- 本地当前只有离散分区标签，主实验使用 one-hot hard prior；
- 若以后取得真实 softmax 预测，只能作为独立敏感性实验，不能与主实验静默混用；
- `11050_1001070` 的 Bosma22b WG 已确认异常，禁止用于任何 WG 依赖环节；其分区标签仍可按审计结论使用。

### 4.5 其他数据集的角色

- **Prostate158**：外部测试首选候选；具有 T2/ADC/DWI、病灶与分区标注，但需先核对肿瘤定义、DWI/HBV对应关系、PZ/TZ标签映射和先验公平性；
- **私有 PCA**：只有在 ROI 语义、病理标准、病例级诊断和与 PI-CAI 的任务映射得到书面确认后，才能作为外部测试；当前不能称为 csPCa 金标准；
- **私有 BPH**：当前 ROI 语义待确认，不作为病灶分割测试集；
- **MSD_Prostate**：主要是解剖分区数据，不作为 PCa 病灶分割测试集；
- **ProstateX**：PI-CAI 已包含部分病例映射，禁止作为独立外部集造成患者重叠。

---

## 5. 数据划分与最终测试协议

### 5.1 PI-CAI train/validation

PI-CAI 三个中心共同进入训练与验证，中心仅作为分层因素，不再作为未见域：

| 集合 | Patients | Studies | 阳性患者 | 用途 |
|---|---:|---:|---:|---|
| Train | 1256 | 1277 | 362 | 参数学习 |
| Validation | 220 | 223 | 63 | checkpoint、阈值及有限超参数选择 |

划分固定为：

- patient-level；
- 按 `center × patient_csPCa` 分层；
- 随机种子 0；
- validation ratio = 0.15；
- 所有 N0、M0–M4 使用完全相同的患者列表。

现有文件 `data/splits/picai_cross_center_splits.json` 的内容可复用，但文件名已不符合新研究定位。正式训练前应生成或迁移为：

`data/splits/picai_train_val_split.json`

旧文件保留用于审计追踪，不静默覆盖。

### 5.2 外部测试

本研究不要求从 PI-CAI 再划内部 test，但必须有一套完全不参与开发的最终测试数据。外部测试遵循：

1. 在首次查看外部模型结果前冻结网络结构、checkpoint 规则、后处理与阈值；
2. 不使用外部数据选择模型、调整归一化、微调参数或修改解剖 gate；
3. 不根据多个外部候选集的结果挑选最有利者；
4. 外部数据与 PI-CAI 的患者必须无重叠；
5. 标签语义、阳性定义和可报告指标必须事先写入审计文件；
6. 外部数据若没有 csPCa/ISUP定义，只能称为 external PCa lesion validation；
7. 外部测试若只有阳性病例，只能评价适用的分割指标，不能据此报告完整 AUROC 或阴性 FP/exam；
8. 外部结果失败时不得回头使用该结果调参，否则该数据集失去测试资格；
9. 外部确认不只评价 M4，而是一次性评价预先冻结的 M2、M3、M4 模型集，否则无法在外部数据上回答
   M4−M2 与 M4−M3；
10. 三个冻结随机种子全部进入外部评价，不根据 validation 或 external test 事后挑选单一种子；
    主分析使用 seed-specific 指标的预定聚合，seed ensemble 只作次要分析。

Prostate158 主要面向专家标注的 PI-RADS≥4 可疑 PCa 病灶，不得在 G0-E 完成前假定它与 PI-CAI
`ISUP≥2` csPCa 目标等价。若最终只能作 `external PCa lesion validation`，则该结果是跨数据敏感性证据，
不替代同任务独立 test 上的 H3 确认。

### 5.3 外部解剖先验公平性

M4 依赖 PZ/TZ 输入，因此外部测试前必须冻结先验协议：

- **主要外部分析必须**在 PI-CAI 与外部集上使用同一可执行的冻结自动分区模型，或使用经预先验证、
  两域一致的自动先验协议；
- 若外部集只能使用人工分区真值，该结果必须标记为 `oracle-prior external analysis`，不能作为主要部署证据；
- 不允许只给 M4 使用人工真值、而让其他 anatomy baseline 使用更差先验；
- 外部集的 PZ/TZ 编码与空间几何必须单独审计；
- 记录 Yuan23、HeviAI23 或替代分区器的训练数据、是否接触 PI-CAI/外部数据、是否使用病灶标签、
  权重/代码可获性与许可证；无法溯源时必须降级相关结论。

### 5.4 独立测试路径冻结门 G0-E

外部数据正式成为 test 前必须完成：

- 数据许可和使用范围确认；
- 病例、序列、病灶标签与分区标签 manifest；
- 病灶语义与阴性病例定义；
- T2/ADC/DWI或HBV的几何和强度审计；
- PZ/TZ标签映射与来源审计；
- **分区器训练数据独立性审计（v2.2）**：核实 Yuan23、HeviAI23 或其他分区器的训练数据是否覆盖 PI-CAI
  （尤其 validation/test 病例）或外部测试病例；**无重叠时才可作为独立自动先验**，有重叠时主分析必须改用
  out-of-fold 预测或未接触目标病例的独立分区器，无法证明独立时只能标记 `oracle/upper-bound` /
  `prior-friendly` 并降级结论（细则见 §12.4）；
- 与 PI-CAI 的患者重叠排查；
- 可用指标和排除规则冻结。

G0-E 不阻塞 N0 工程调试，但必须在任何结果被计为可报告的正式 N0/M0–M4 run 前解决。若没有外部候选能承担
同任务确认，必须在此时从 PI-CAI 冻结内部 test 并重建 train/validation；已有 1277/223 split 作废但保留审计，
不允许在观察正式模型结果后再做这一决定。

---

## 6. 数据可用性与正式 run 的启动条件

> 本节只定义**条件**；当前状态与证据链接见 `docs/STATUS.md`。

**G0-P（PI-CAI 数据可用）的通过条件** = 以下 8 项全部完成，且判定与逐项证据写入
`docs/PICAI_Model_Readiness_Audit.md`：

1. 修正 SimpleITK 方向码约定：单位 direction 对应 LPS，而不是 RAS；
2. 修正相应单元测试，重新生成含 orientation 字段的 manifest 与 geometry audit；
3. 将 ADC/HBV 按物理坐标重采样至冻结参考网格；
4. 生成覆盖 1500 study 的 canonical 二值病灶标签；
5. 对 5 例 `geometry_suspect` 完成重采样后人工 QC 并记录处理决定；
6. 在所有 WG 依赖环节屏蔽 `11050_1001070` 的异常 WG；
7. 生成新的 `picai_train_val_split.json` 并复核患者泄漏；
8. 完成 nnU-Net/raw 与自研数据索引所需的模态顺序、标签和校验值冻结。

数据物化与审计修正不改变分区质量分析的口径；G0-P 的判定细节以审计报告为准，本文档不复述状态。

**N0 与 M0–M4 正式 run 的通用启动条件**：

1. G0-R 已冻结序列错位主协议，且该决策**未改变**本次 run 使用的预处理身份；
2. G0-E 已冻结独立测试路径（路径 A 或 C）；
3. 训练预算、checkpoint 规则与增强对齐差异已显式冻结（见 §9.4 与 `docs/runbooks/n0_training.md`）；
4. 查看任何论文指标前，G0-SAP 已冻结。

凡上述条件未满足而执行的 run，**一律只能标记为 feasibility run**，不得并入正式比较，也不得事后补标。

**N0 启动前必须冻结的两项**：

1. **训练预算与骨干差异**：N0 使用官方 PlainConv 网络与 1000 epoch；自研 M0–M4 使用 v2.3 Residual-Encoder
   骨干与 `max_epochs=200`。两者必须在启动前冻结（可将 N0 对齐到官方自带变体 `nnUNetTrainer_250epochs`，
   官方无 200 epoch 变体）并在论文中显式声明，否则 N0 vs M0 的差距无法完全归因于实现差异；
2. **N0 checkpoint 使用规则**：正式推理与评测固定使用 `checkpoint_final.pth`（`nnUNetv2_predict` 默认即如此），
   禁用 `--val_best`；即使出现 `Pseudo dice` 长期贴地，checkpoint 选择同样不得依赖 `checkpoint_best.pth`。

**已排除预实验的处置约束（数据安全，不随状态变化）**：nnU-Net v2 默认 Dice+CE 的预实验（未完成）与首次
FFT 崩溃 run 的目录、日志与 checkpoint **必须保留、不得删除或覆盖**；前者**不得**写成正式失败性能实验，
**不得**补写任何全体积 Dice / AP / AUROC 指标，也**不再作为正式 N0 baseline**。运行事实见
`docs/experiment_log.md`，当前状态见 `docs/STATUS.md`。

**模型身份隔离要求**：旧 PlainConv `PlanDrivenUNet3D` 的任何运行只能标记为 legacy/feasibility，
不得补标为 v2.3 正式 M0；v2.3 正式 M0 = `PlanDrivenResidualEncoderUNet3D`（§7.2、§10.1）。

---

## 7. 技术路线：官方强基线与自研框架

### 7.1 官方强基线 N0

**定义（术语统一）**：`N0 = PI-CAI official-style nnU-Net v2 Focal+CE NoFFT`。

- 框架：项目内固定 `third_party/nnUNet`（tag `v2.6.2`，commit `74ceb6803d10dcee29b2cc481678d3a3d069f281`），
  **不修改第三方代码**；
- 数据与计划：`Dataset605_PICAI`、冻结 `3d_fullres` plan、冻结 fold 0 split（train 1277 / val 223）；
- 输入：T2W+ADC+HBV 通道拼接；单 fold、单 GPU；
- **唯一方法学变化**：把 v2 默认的、每个 deep-supervision 尺度的 Dice + CE 换为 PI-CAI 官方公开基线的
  `0.5 × FocalLoss(gamma=2, alpha=None) + 0.5 × CrossEntropyLoss`；
- 其余保持 v2.6.2 默认：网络结构、前景过采样、增强、SGD + Nesterov、PolyLR、1000 epoch、checkpoint 规则；
- 唯一运行时兼容性例外：关闭 Gaussian blur 的 FFT benchmark（`benchmark=False`），blur 概率（0.2）、
  sigma（0.5–1.0）、适用通道与其余增强不变——本环境（torch 2.10 + fft-conv-pytorch 1.2.0）的稳定性修复，
  不是新的数据增强策略；
- 适配代码：`src/zonal_reliability_fusion/integrations/nnunet_picai_flce.py`；启动器：
  `scripts/train/train_n0_picai_flce.py`；输出目录：
  `outputs/nnUNet_results/Dataset605_PICAI/nnUNetTrainerPICAI_FLCE_NoFFT__nnUNetPlans__3d_fullres/fold_0/`。

**术语与表述边界**：本项目称其为 **PI-CAI official-style Focal+CE v2 port**，**不得**称为「逐位官方复现」——
官方基线基于 nnU-Net v1、1295 个带 human-expert ISUP≥2 标注的 PI-CAI 扫描，并使用患者无重叠的 5-fold CV 与集成；
本项目为 nnU-Net v2、1500 study、当前单 fold。详见 `docs/N0_PICAI_Official_Baseline.md`。

**当前状态**：见 `docs/STATUS.md`；执行入口与成功判据见 `docs/runbooks/n0_training.md`。

N0 用于提供成熟训练框架的强参考结果，不承载核心融合模块，也不是 M0–M4 的因果消融基线。在训练预算、
checkpoint 选择和增强协议未配平时，N0 只能作 contextual reference，不对 N0−M0 差异作单一因果解释。

### 7.2 自研 plan-driven Residual-Encoder 3D U-Net

自研代码位于 `src/zonal_reliability_fusion/`。该路径是历史工程命名，论文与新文档不再使用“reliability”作为方法主张；后续若重命名包，必须单独迁移并保持可追溯性。

论文中将该网络称为 **plan-driven Residual-Encoder 3D U-Net** 或 **custom residual-encoder 3D U-Net**，不得称为
官方 nnU-Net，也不得把 residual encoder 本身作为研究创新。

自研框架读取或复用固定 `3d_fullres` plan 的数据与空间规划结果：

- target spacing；
- patch size 与 batch size；
- 每级 kernel 与 stride；
- 通道数及上限；
- 分序列归一化；
- 前景过采样；
- deep supervision；
- overlap sliding-window inference、Gaussian融合和必要的镜像TTA；
- 预测恢复至原始物理空间。

v2.3 只改变自研 M0–M4 的网络 block 身份：将原 PlainConv encoder 升级为 residual encoder。目标复杂度以
nnU-Net ResEnc-M 级别为起点，但本项目仍由自身版本化配置显式冻结 block 数量、参数量、FLOPs 与显存，不得仅凭
“ResEnc-M-like”名称推断实现等价。若残差结构要求修改通道或 block 数，必须在任何正式 M0–M4 run 前升版并记录
新配置哈希；不得静默改变已物化数据、split、spacing、patch 或推理几何。

所有自研模型 M0–M4 共享同一 residual block 规范、stage/通道计划、stage 2 之后的共享 encoder、decoder、数据、
采样、损失、训练预算、checkpoint 和推理协议。M0 的三序列直接 concat 单流输入，与 M1–M4 的 stage 0–2
modality-specific 三分支，是主消融预先定义的输入路径差异；它不授权为 M4 单独增加 block、通道、decoder 或 skip。

### 7.3 自研基础网络边界

- encoder 使用 `Conv3d + InstanceNorm3d + LeakyReLU` 构成的残差 block；当 stride 或通道变化时，shortcut 使用
  明确的投影，不允许隐式裁剪通道；
- residual encoder 的 stage 数、kernel、stride、通道上限继续来自冻结 plan；block 数与 shortcut 规则写入版本化
  架构配置，不在代码中隐式推断；
- decoder 使用成熟的普通 U-Net 路径：`ConvTranspose3d → concat skip → stacked conv`，保留 deep supervision；
- 主模型不加入 anatomy-conditioned skip、Attention U-Net gate、UNet++/UNet3+、Transformer、Mamba、扩散模型、
  对比学习或额外小病灶分支；
- 若今后单列比较其他 decoder/skip 变体，该变体必须对 M0–M4 同等适用，不得只增强 M4，也不得并入核心 H3 归因。

残差 block 的概念形式为：

$$
F_{l+1}=\sigma\big(R_l(F_l)+P_l(F_l)\big),
$$

其中 $R_l$ 是卷积—归一化—激活堆叠，$P_l$ 是恒等或投影 shortcut，$\sigma$ 是非线性激活。确切的 pre/post-activation
顺序、每 stage block 数和初始化规则必须由实现配置与单元测试冻结，而不能只依赖该概念公式。

只有当新的 Residual-Encoder M0 完成 shape、残差 shortcut、deep supervision、过拟合、滑窗恢复和主要指标检查，
且参数/FLOPs/显存已记录后，才开始融合创新实验。旧 PlainConv M0 的测试通过不能替代新的 G2。

---

## 8. 核心方法

### 8.0 总体概念数据流（M4）

```text
 T2W/ADC/HBV ─► S_m ─► P_m ─► U_m ─┐
                                   ├─► concat ─► G_img ─► L_img ─┐
 PZ/TZ one-hot ─► AvgPool ─► G_Z ─► E_Z ─► H_Z ─► (γ_Z,β_Z) ────┴─► L=(1+γ_Z)⊙L_img+β_Z
                                                                       │
                                                        W=softmax(L, sequence)
                                                                       │
             F_fuse = Σ_m W_m⊙U_m（encoder stage 2）◄───────────────────┘
                                   │
                    共享 residual encoder ─► 普通 decoder + 冻结 skip 聚合
                                   │
                        deep supervision ─► 3D 病灶概率图
```

图未展开各 stage 的分辨率、残差 block、全部 decoder skip 与 DS 输出；浅层 skip 的唯一主方案由 §8.6 冻结，
不允许 M4 独占额外的 gated/anatomy skip 路径。

该图同时定义主消融链的退化关系：

- **M1**：移除 `G_img` 与解剖分支，在三个序列间固定使用 `W=(1/3,1/3,1/3)`；
- **M2**：保留 `G_img`，移除解剖分支，即 `gamma_Z=beta_Z=0`；
- **M3**：gate 与 M2 完全相同，`E_Z` 不进入 gate，而是在融合后通过 `Q([F_fuse,E_Z])` 注入；
- **M4**：使用图中的完整 conditional-affine anatomy-conditioned gate。

为保持图的可读性，未展开各 encoder stage 的分辨率、残差 block、全部 decoder skip 与 deep-supervision 输出。
浅层 skip 的唯一主方案由 §8.6 冻结，不允许 M4 独占额外的 gated/anatomy skip 路径。

### 8.1 序列特异性浅层 residual stem

T2W、ADC、HBV分别经过结构相同、参数独立的浅层 residual stem：

$$
F_m=S_m(X_m),\qquad m\in\{T2,ADC,HBV\}.
$$

每个分支再映射到共同特征空间：

$$
U_m=P_m(F_m).
$$

必须先投影和归一化再加权，减少由特征幅值差异造成的伪权重。三个分支只从 stage 0 延伸至 stage 2，融合后共享
余下 residual encoder 与完整 decoder，不使用三个完整 encoder。stem 的 residual block 规则与共享 encoder 一致，
但三条序列分支参数独立。

### 8.2 图像驱动空间gate

无解剖条件的gate根据三序列特征生成每个空间位置的序列logits：

$$
L^{img}=G_{img}([U_{T2},U_{ADC},U_{HBV}]).
$$

M2用于判断普通空间动态融合是否已经足够。

### 8.3 解剖分区条件gate

原始 PZ/TZ one-hot prior 在融合尺度使用 average/adaptive-average pooling 降采样，保留边界体素的区域占比，经轻量
zone encoder 得到：

$$
\widetilde Z=\operatorname{AvgPool}(Z),\qquad E_Z=G_Z(\widetilde Z).
$$

这里的 $\widetilde Z\in[0,1]$ 只表示 **fractional zonal occupancy**，不得称为自动分区模型的 softmax 概率。

由解剖特征生成与三个序列 logit 对应的局部仿射调制参数：

$$
(\widehat\gamma_Z,\beta_Z)=H_Z(E_Z),\qquad
\gamma_Z=\tanh(\widehat\gamma_Z),\qquad
\gamma_Z,\beta_Z\in\mathbb{R}^{3\times D_k\times H_k\times W_k}.
$$

最终权重为：

$$
L=(1+\gamma_Z)\odot L^{img}+\beta_Z,\qquad
W=\operatorname{softmax}(L,\text{sequence}).
$$

融合特征为：

$$
F_{fuse}=\sum_m W_m\odot U_m,\qquad \sum_m W_m=1.
$$

调制头的最后一层使用零初始化，使 $\gamma_Z=0,\beta_Z=0$ 时 M4 从 M2 的 image-driven gate 开始；
`tanh` 有界参数化用于避免早期极端放大。$\gamma_Z$、$\beta_Z$ 和最终 $W$ 都只是结构量，
不具备唯一可辨识的因果或临床贡献语义。论文使用 `learned sequence preference` 或 `fusion coefficient`，不使用“贡献百分比”。

### 8.4 M3 的普通解剖输入路径

M3 与 M4 使用相同的 $\widetilde Z$ 和 $G_Z$，但 $E_Z$ **不进入 gate**。M3 依然先用与 M2 相同的
$L^{img}$ 完成序列融合，再将 $[F_{fuse},E_Z]$ 经轻量 $1\times1\times1$ 投影后送入后续共享 encoder：

$$
F^{M3}_{shared}=Q([F_{fuse},E_Z]).
$$

这样 M3 不能通过上游共享特征间接改写 image gate。M3/M4 的 zone encoder 宽度、深度与融合尺度保持一致；
实现时必须报告参数量和 FLOPs。若总参数差异无法控制在 1% 内，则增加与 M4 同结构但输入 constant occupancy 的容量对照，
不将 M4−M3 的全部差异归因于解剖条件。

### 8.5 融合尺度

主实验只在冻结 `3d_fullres` plan 的 **encoder stage 2 输出**（stage 从 0 计数）融合。该位置相对 plan 输入的
累积 stride 为 `(1, 4, 4)`，即仅经过两次平面内下采样，不用“1/4 体素分辨率”暗示 z/xy 各向同性。M1–M4 必须使用
完全相同的融合 stage，不再使用 validation 搜索融合位置。

只有单尺度M4成立后，才允许把多尺度融合列为补充扩展；多尺度不能在第一版与核心模块同时引入。

### 8.6 浅层 skip 聚合与标准 decoder

M1–M4 在 stage 0/1 有三路 modality-specific 特征。为保留小病灶所需的高分辨率细节，同时避免建立第二套解剖 gate，
浅层 skip 固定为：

$$
S_l=Q_l([F^l_{T2},F^l_{ADC},F^l_{HBV}]),\qquad l\in\{0,1\},
$$

其中 $Q_l$ 固定采用 `1×1×1 Conv → InstanceNorm3d → LeakyReLU`，把三路拼接特征压缩到冻结 plan 对应 stage 的
skip 通道数。主协议同时冻结：

- $Q_l$ 不读取 PZ/TZ、WG、最终序列权重 $W$ 或 decoder 特征；
- $Q_l$ 不输出新的 modality logits，不构成第二个序列 gate；
- M1、M2、M3、M4 使用完全相同的 $Q_l$ 拓扑、初始化规则和通道数；
- stage 2 skip 直接使用 $S_2=F_{fuse}$；stage 3 及更深 skip 来自共享 residual encoder；
- decoder 统一使用普通 concat skip，不加入 attention、dense/full-scale skip 或 anatomy modulation；
- 必须报告 $Q_l$ 与整个模型的参数量/FLOPs，并检查 M3/M4 的参数差异规则（§8.4）。

$Q_l$ 是共同骨干中的浅层信息整形器，不作创新。其存在会给 decoder 提供不经过核心 gate 的低层细节，因此不得将
gate 权重解释为网络全部信息流的“贡献百分比”；核心归因仍以 M4−M2、M4−M3 的受控性能差异为准。

### 8.7 WG使用边界

WG默认不进入核心gate，避免把腺体定位收益与分区条件融合收益混为一谈。**v2.2 进一步收紧：WG 退出所有
主模型（M0–M4）的 patch 采样与网络输入。**

- **M0–M4 的采样协议必须完全相同**：均使用冻结 plan 的 patch/batch 与基于病灶前景的过采样（§9.3），
  不得为任何模型（尤其 M4）单独引入 WG 驱动的采样、裁剪或前景定义；
- **WG 不得参与任何主模型的 patch 采样或输入**：不作为 M0–M4 的输入通道，也不用于定位/限制采样区域；
- WG 只允许用于：数据 QC、**统一的评测分层与假阳性分析**（对 M0–M4 使用同一套 WG 派生分层）、
  PZ/TZ 质量审计，以及**单列消融**；
- **不允许 M4 独占 WG 信息**：若将 WG 作为网络输入或采样先验，必须作为对所有模型同等适用的单列消融，
  不得并入 M4 默认配置，也不得只给 M4 使用；
- `11050_1001070` 的异常 WG 在所有 WG 依赖环节（QC/分层/审计）继续屏蔽（§4.4）。

### 8.8 容量与混杂控制（H2/H3 归因的前提）

**已识别的混杂**：M2 相对 M1 不只引入「动态融合」，还额外引入 gate 子网络（`G_img`）的参数与非线性；
若直接把 M2−M1 读成「动态融合收益」，会把容量增益误读为机制增益。

**规范要求（必须在运行 M1/M2 之前冻结其一；属 §20.2 待冻结决策 D6）**：

1. **同容量对照（首选）**：为 M1 增加参数规模相当、但**不读取跨序列比较信息**的对照（例如逐序列独立仿射
   或常量 gate），使 M2−M1 只反映「跨序列比较 / 空间动态性」；对照必须对 M1 与 M2 同等适用，不得只增强一方；
2. **显式降级（备选）**：若不设同容量对照，则 H2 只能作为**探索性**证据，不得用于宣布「动态融合成立」，
   也不得据此把 M4 的变化解释为解剖条件融合收益；
3. **M3 vs M4**：按 §8.4 约束（同 zone encoder 宽度/深度/融合尺度；总参数差异无法控制在 1% 内时，
   必须增加同结构、输入 constant occupancy 的容量对照），不得把 M4−M3 的全部差异归因于解剖条件；
4. **报告要求**：M0–M4 逐一报告总参数量、可训练参数量、MACs（含 gate 与浅层 skip 聚合器）与峰值显存，
   其中 gate 相关参数量必须单独列出；缺失这些数字时不得作 H2/H3 的确认性陈述。

---

## 9. 训练与推理协议

### 9.1 增强

空间增强必须同步应用于T2W、ADC、HBV、病灶和解剖先验：

- rotation；
- scaling；
- mirroring；
- crop。

常规强度增强可按序列独立应用，但只作为标准正则化，不作为可靠性创新：

- gamma/contrast；
- Gaussian noise；
- blur；
- bias field（适度）。

第一版不使用专门的modality corruption、modality dropout、域随机化、consistency loss或ranking loss。

**增强对齐（冻结要求）**：M0–M4 与正式 N0 的增强差异清单固定为 `augmentation.pending_parity`
（rotation / scaling / 低分辨率模拟 / noise / blur / gamma，共 6 项）。在启动任何正式 run 前必须二选一并记录：
「对齐到 N0」或「接受差异并在论文中显式声明」；该项属 §20.2 待冻结决策 D7，**不得在结果产生后调整**。

### 9.2 损失

**正式 N0**（`PI-CAI official-style nnU-Net v2 Focal+CE NoFFT`）在每个 deep-supervision 尺度使用：

$$
L_{N0}=0.5\,L_{\mathrm{Focal}}(\gamma=2,\ \alpha=\mathrm{None})+0.5\,L_{\mathrm{CE}},
$$

深度监督权重仍为 v2.6.2 默认（`1/2^i`，最低分辨率不监督）。

**自研 M0–M4 的目标损失协议**：

$$
L_{seg}=0.5\,L_{\mathrm{Focal}}(\gamma=2,\ \alpha=\mathrm{None})+0.5\,L_{\mathrm{CE}}.
$$

约束：

- M0–M4 必须复用与正式 N0 **逐值对齐**的同一实现，并保持 deep-supervision 权重与官方
  `DeepSupervisionWrapper` 一致；**不得**引入第三种损失口径；
- `DiceCELoss` 仅作为迁移前的 legacy 口径保留，**与正式 N0 不可做正式性能比较**；
- 配置层必须对「缺参数 / 两口径混用 / 未知名称」显式报错，禁止静默回退默认值；
- 损失口径的任何变更都必须在查看结果前完成并记录（实现与对齐证据见 `docs/Development_Log.md`
  与 `tests/unit/test_loss_alignment_n0.py`）。

当前实现与验收状态见 `docs/STATUS.md`；**实现状态不改变本节的协议要求**。

Focal/Tversky 不作为创新变量：除「与 N0 对齐」这一唯一动机外，不得为了刷分单独调整损失；也不得引入
modality corruption、域随机化或一致性损失。

### 9.3 Patch采样

- 初始参考nnU-Net的前景过采样比例；
- 保留阴性病例和非病灶前列腺patch；
- M0–M4使用同一sampler；
- 不允许为M4单独提高阳性patch比例。

### 9.4 优化与checkpoint

- 初始对齐固定nnUNetv2版本的SGD+Nesterov+PolyLR；
- 使用AMP、断点续训、固定随机种子和完整配置快照；
- checkpoint只依据PI-CAI validation的预定指标选择；
- threshold与后处理只在PI-CAI validation确定；
- 外部测试不得参与任何选择。

**训练预算（v2.2 固定，不声称最优）**：正式 M0–M4 统一使用 **`max_epochs=200` 固定预算**、每 epoch 250
 iteration。**PolyLR 终点（`max_steps`）固定为 200 epoch**，与 `max_epochs` 一致。固定预算的目的：使所有模型与
所有 seed 获得**完全相同的训练预算**，避免“连续无改善”停止在不同模型/seed 上引入不同的有效 epoch 数从而污染
配对比较。若今后确需延长（如 300/400），只允许依据 M0 的训练曲线与计算耗时决定，并在实现或运行 M1–M4 前
记入 `experiment_log.md` 并冻结；**不得依据 M1–M4 或外部 test 表现调整**，且调整后必须对全部模型与 seed 统一适用。

**正式验证协议（M0–M4 必须完全一致）**：`validation.every_n_epochs=5`。在第 5、10、15…个完整 epoch
结束后，遍历 fold 0 的**全部 223 个 validation study**（阳性 63 / 阴性 160）做**完整 3D 体积** sliding-window
推理——patch 来自冻结 plan、`step_fraction=0.5`、Gaussian 融合、mirror TTA 关闭、不做任何随机增强、推理期间关闭
deep supervision；不使用随机 validation patch 代替完整体积。无论是否恰逢 5 的倍数，正常训练的**最后一个 epoch**
必须执行一次全体积验证。非 full-volume epoch 只记录训练期 online 指标，**不参与 checkpoint 选择、不更新 best、
不计入 patience**。

每个 full-volume validation event 只保存病例级指标与汇总，不落盘预测体积；best checkpoint 冻结后才单独做一次
完整预测以保存结果。低频验证的**规范性要求**见 §9.4.1；v2.3 架构必须把架构身份纳入配置快照与 resume 校验，
并在新骨干上重跑合成回归（§9.4.2）。**G2 未通过前不得启动 M0 正式训练**；实现与验收状态见 `docs/STATUS.md`。

**checkpoint 指标**：`val_positive_casewise_dice_mean`（阳性病例 case-wise Dice 的算术平均，最大化）。
阳性病例预测为空记 `Dice = 0`；**阴性病例不进入主要 Dice 均值，且"GT 与预测均为空"不得记为 `Dice = 1`**，
阴性表现以 `val_negative_fp_case_rate` 与 `val_fp_voxels_per_negative_exam` 记录。每轮结束必须校验实际覆盖数量
（223 / 63 / 160），不一致时**明确报错且不保存 best**。

**early stopping（v2.2：正式主实验禁用）**：正式 M0–M4 **禁用“连续无改善”early stopping**
（`early_stopping.enabled=false`），使每个模型与 seed 都跑满固定 200-epoch 预算；**只保留 NaN/Inf、覆盖病例数
（223/63/160）不一致、预测非有限值等安全中止**（由 validator/trainer 硬校验触发，与 early stopping 无关）。
选模仍按 validation best checkpoint（`val_positive_casewise_dice_mean` 最大化）。

通用 early-stopping 代码可保留用于工程调试，但**其 `patience` 单位必须是 validation event（非 epoch）、
`min_epochs` 单位是 training epoch，且只能在 validation event 后触发**；resume 必须连续恢复 event 计数与 best 状态，
不得把非验证 epoch 计入 patience。**该开关不得在正式 M0–M4 配置中启用。**

**每个 run 必须报告实际完成的 epoch 数与 validation event 数量**（训练结束摘要与 `epoch_metrics.csv` 均记录
`epochs_done`、`validation_events_completed`），以便核对固定预算是否被完整执行、是否发生意外安全中止。

### 9.4.1 低频全体积验证：规范性要求

“每 5 epoch 验证”不是改一个 YAML 数字，而是一组**必须实现并被回归测试覆盖**的行为。以下为规范性要求
（§9.4 的实现必须全部满足；实现与验收记录见 `docs/P2_M0_Implementation.md` 附录 A，v2.3 侧见
`docs/P2_M0_ResidualEncoder_v23.md`）：

1. **validation-event 调度**：`full_volume` 模式按 `every_n_epochs` 触发；1-based epoch 整除 N 时为验证事件，
   否则跳过；`every_n_epochs >= 1` 合法，`0/负数` 必须在配置层、协议层与 trainer 层均报错；
2. **最终 epoch 必验证**：`epoch == max_epochs-1` 即使不是 N 的倍数也强制触发一次全体积验证；
3. **非验证 epoch 不更新选模状态**：不调用全体积验证器、不解析 checkpoint 指标、不更新 `best_metric`/`best_epoch`、
   不递增 patience；日志与 CSV 必须标记跳过状态，且不得让"跳过"看起来像"验证失败"；
4. **best 只在 validation event 产生**：`checkpoint_best.pth` 只可能落在验证 epoch；
5. **非验证 epoch 仍正常保存 last/周期 checkpoint**：跳过验证不影响可恢复性（含绝对 epoch 计数）；
6. **resume 必须连续恢复选模与 event 状态**：`validation_events_completed`、`last_validation_epoch`、
   `best_metric`、`best_epoch`、patience 计数与 epoch/LR；
7. **协议一致性**：验证频率纳入 resume 协议校验；协议变化默认**拒绝 resume**，显式放行时必须**重置**
   不可比的 best/event 状态并留痕；
8. **CSV schema 稳定**：跳过 epoch 对全体积指标字段写空串占位，保证每行字段集合与顺序一致；
9. **真实 223 例全体积推理的速度、显存与数值必须在真实 GPU 上核实**（属 G2/P2B，由研究者执行）。

### 9.4.2 v2.3 Residual-Encoder：G2 验收要求

为避免「G2 通过前不得实现 M1–M4」与「G2 又要求 M1–M4 已实现」的循环依赖，本清单明确拆成
**G2 前置（P2A/P2B）** 与 **G2 后置（P3–P5）** 两组。**G2 的通过条件只引用前置 1–8 项**；
后置 9–11 项由 P3–P5 分阶段完成，不得反向成为 G2 的前置条件。
逐项完成证据与验收细节见 `docs/P2_M0_ResidualEncoder_v23.md` §9；当前状态见 `docs/STATUS.md`。

| 组别 | # | 要求 |
|---|---|---|
| **G2 前置**（P2A 实现/合成验证；P2B 真实数据验证） | 1 | residual block 必须覆盖：恒等 shortcut、通道投影、stride 下采样、**奇数尺寸**与 He 初始化 |
| | 2 | 必须有**版本化架构配置**冻结 block 数、激活顺序、shortcut 规则、初始化、参数预算与结构哈希；声明未实现的值必须在加载时报错 |
| | 3 | encoder/decoder 全部 stage shape 与 deep-supervision 输出顺序必须与冻结 `3d_fullres` plan 一致 |
| | 4 | §8.6 浅层 skip **接口契约**必须冻结并通过校验；不要求在 G2 阶段实现 $Q_l$ / $F_{fuse}$ |
| | 5 | 必须输出参数量、可训练参数量、FLOPs/MACs 与结构摘要；**峰值显存必须实测**（估计值只能作下界代理） |
| | 6 | 架构名/block 配置/哈希纳入 checkpoint/resume 校验；旧 PlainConv checkpoint 必须被拒绝；**禁止 `strict=False`** |
| | 7 | 必须在**不读真实医学数据**的合成输入上完成 forward/backward、AMP、DS、滑窗 shape 与 resume 回归 |
| | 8 | 必须由研究者在 **G0-R 冻结预处理身份后**执行真实数据三步验证：loader smoke → small-overfit → 诊断性首次全体积验证 |
| **G2 后置**（P3–P5，各模型正式运行前完成） | 9 | 实现 M1–M4 统一 $Q_l$ skip 聚合器，并测试其不接受 PZ/TZ/WG/gate 权重 |
| | 10 | 测试 stage 2 的 $F_{fuse}$ 同时进入共享 residual encoder 与对应 skip；stage 3 以后只用共享 skip |
| | 11 | 为 M1–M4 输出参数量/FLOPs/峰值显存/结构摘要，并把 skip/fusion 协议身份纳入 checkpoint/resume 校验 |

### 9.5 推理

- overlap sliding-window；
- Gaussian权重融合；
- 必要且统一的镜像TTA；
- 恢复至原始物理空间；
- 所有模型使用完全相同的后处理协议。

---

## 10. 基线、核心消融与训练优先级

### 10.1 主模型链

| 编号 | 模型 | 目的 |
|---|---|---|
| **N0** | **PI-CAI official-style nnU-Net v2 Focal+CE NoFFT**（T2W+ADC+HBV concat；v2.6.2 框架，单 fold） | 协议不完全配平的成熟强参考 |
| **M0** | 三序列普通 concat + plan-driven Residual-Encoder 3D U-Net（v2.3 目标架构；须先通过 §9.4.2 / G2） | 验证新自研骨干与管线 |
| **M1** | 三个浅层 residual stem + 等权融合 + 同一 residual backbone | 控制多分支结构与容量影响 |
| **M2** | M1 + 图像驱动空间gate | 测试普通动态融合 |
| **M3** | M2 gate 不受解剖影响；融合后注入与 M4 同源的 zone feature | 控制“只是增加解剖信息” |
| **M4** | M2 + PZ/TZ 条件仿射调制 gate | 核心方法 |
| **H0（可选）** | 同一 M1 特征上的预设 deterministic hard-zone routing | 比较硬规则与学习式软融合 |

**表格使用约束**：

- 已排除的预实验是「nnU-Net v2 默认 Dice+CE」（`nnUNetTrainer`），**不是** N0，不得作为基线出现在论文表格中；
  其产物处置约束见 §6，运行记录见 `docs/experiment_log.md`。
- M0–M4 进入正式比较前必须统一到 Focal+CE（同一公式、同一 DS 权重），并冻结与 N0 的训练预算差异（§9.4）。
- M0–M4 必须共享同一 Residual-Encoder/decoder/block 配置；M1–M4 还须共享 §8.6 的浅层 skip 聚合器。Residual
  backbone、普通 decoder 与 skip 不作为创新，只在 M0–M4 内提供共同容量基础。
- 旧 `PlanDrivenUNet3D` PlainConv M0 只能标记为 legacy/feasibility，不得与 v2.3 M1–M4 混入同一正式消融表。

### 10.2 预先固定的主要比较

- M1 vs M0：多分支与容量影响；
- M2 vs M1：空间动态融合收益（**容量混杂控制见 §8.8**：未设同容量对照时，H2 只能作为探索性证据，
  且不得据此把 M4 的变化解释为解剖条件融合收益）；
- M3 vs M2：普通解剖输入收益；
- **M4 vs M2：分区条件的总增益；**
- **M4 vs M3：解剖用于gate是否优于普通输入；**
- M4 vs H0：学习式软融合与硬规则的差异（可选）。

未经M1–M3控制，不将M4相对M0的全部提升归因于核心创新。

M3 与 M4 必须共享同一 zone encoder 设计与融合 stage，并按 §8.4 控制容量。H0 不训练独立 PZ/TZ 网络；其确定性权重
在实现前冻结，例如 PZ 内 `(T2W, ADC, HBV)=(0, 0.5, 0.5)`、TZ 内 `(1, 0, 0)`，mixed/背景使用等权；
在 fractional occupancy 上，只有当某一分区占比 `>=0.75` 时应用对应硬路由，其余位置一律等权。H0 只是 PI-RADS-inspired 压力对照，
不声称精确复刻临床阅片或某篇既有工作。

### 10.3 输入序列实验

第一优先级：

| ID | 输入 | 目的 |
|---|---|---|
| S0 | T2W | 单序列基线 |
| S1 | ADC | 单序列基线 |
| S2 | HBV | 单序列基线 |
| S3 | T2W+ADC+HBV | 三序列互补性 |

第二优先级（核心结果成立且资源允许）：

- T2W+ADC；
- T2W+HBV；
- ADC+HBV。

输入组合实验优先使用同一concat backbone，不为每种组合重新设计专属网络。

### 10.4 资源控制

- feasibility阶段：每个模型先运行1个固定种子，只用于检查数值稳定性、显存、速度和实现错误；
- 不得根据单种子效果好坏只复验“赢家”；任何基于性能的 G3/G4 通过或失败判定都必须先完成预定的多种子运行；
- 正式结论至少对 M0/M2/M3/M4 运行相同的 3 个冻结种子；M1 若用于判定 H2/G3，也须使用相同 3 个种子；
- 输入组合和H0可先单种子筛查，避免无效计算扩张。

---

## 11. 评价指标

### 11.1 主要指标

- **primary endpoint**：阳性患者 case-wise Dice；
- 关键次要指标：lesion-level average precision（AP）；
- 关键次要 FROC 指标：预定 FP/exam 操作点下的 lesion sensitivity；
- 阴性检查指标：false-positive case rate 与 FP/exam。

AP和检测指标优先使用官方`picai_eval`。Dice不把“预测与真值均为空”的阴性病例简单记为1；阴性表现通过FP/exam和病例级指标报告。
主 FROC 操作点预定为 `0.5、1.0、2.0 FP/exam`；其余曲线仅作补充展示。

**训练期 checkpoint 指标**（实现字段名，硬冻结）：`val_positive_casewise_dice_mean`——阳性病例 case-wise Dice 的
算术平均，最大化；定义、覆盖校验（223 / 63 / 160）与 early stopping 规则见 §9.4。

**最终 test 的语义终点（estimand）**：`positive_casewise_dice`——与上式**同一数学定义**，但在冻结最终 test 上
一次性评估，不参与任何选择。**两者不得混用命名**：

- `val_` 前缀字段只表示「训练期 validation 事件上由自研 validator 产出的选模指标」，**不是**最终终点名称；
- 论文表格与摘要中报告的是最终 test 的 `positive_casewise_dice`；引用训练曲线时必须显式写出 `val_` 前缀；
- 若要把实现字段重命名为与语义终点一致（去掉 `val_` 前缀或改用 `test_` 前缀），属待冻结决策 §20.2 D5，
  **不得由实现者单方面改名**（改名会使既有 checkpoint、CSV、配置与内容哈希失配）；
- 机器可读映射见 `docs/protocols/G0_SAP.md` §1.1 与 `configs/protocols/g0_sap.yaml` 的 `endpoints` 块。

论文指标（case-wise Dice、AP、lesion sensitivity、FP/exam）在 best checkpoint 冻结后单独计算，训练过程不得据此选择模型。

### 11.2 次要指标

- patient-level AUROC（外部数据具备可靠病例标签时）；
- lesion-wise Dice/recall；
- NSD（仅在预先冻结物理容差后）；
- 小病灶分层；
- PZ/TZ/mixed/outside分层；
- 参数量、显存、训练与推理时间。

### 11.3 外部指标限制

外部数据缺少ISUP、阴性病例或统一病灶定义时，只报告该数据实际支持的指标，并将其与PI-CAI validation分表呈现，不混合汇总。

PI-CAI 阴性参考可来自组织学 `ISUP≤1`，也可来自无随访的 MRI `PI-RADS≤2`。因此阴性 FP 结果必须承认参考标准不确定性；
在元数据允许时，补充报告组织学阴性与 MRI-only 阴性子组。

### 11.4 评测与后处理冻结

`scripts/evaluate/` 的**实现**与合成回归测试必须在 SAP-A 冻结前完成（否则无法定义候选网格与选择算法）；
但在真实 validation / 最终 test 上的**运行**只能发生在对应的生命周期阶段（见下）。协议至少冻结：

1. 用于 Dice 的前景阈值候选网格、选择方法与 tie-break rule；
2. 3D 连通性、最小病灶体积候选网格、候选病灶置信度聚合和分割图到 detection map 的转换；
3. `picai_eval` 版本、`min_overlap`、`overlap_func`、split/merge 规则和 case-confidence 函数；主协议默认
   `IoU >= 0.1`，任何改动都必须在观察模型结果前写入版本化评测配置；
4. sliding-window、Gaussian、TTA、原始物理空间恢复与插值的唯一实现；
5. threshold 和后处理只在 PI-CAI validation 上选择一次，然后对所有冻结 test 和模型保持一致；
   若特定模型使用独立阈值，必须事先声明并对每个模型完全同规则执行。

**尚未冻结的关键选择（必须在 SAP-A 冻结时写明并记录理由；属 §20.2 D3）**：

- 主比较使用**共同阈值**（默认口径：对全部模型使用同一套在 validation 上选定的阈值）还是
  **每模型独立阈值**（须事先声明、对每个模型完全同规则执行）——两者语义不同，**不得事后切换**；
- 后处理参数（连通性、最小体积、候选置信度聚合与 detection map 转换）、bootstrap 次数与随机种子、
  `picai_eval` 版本与 `overlap_func` / `split_merge_rule` / `case_confidence_function`。

**G0-SAP 生命周期（v2.3.1 amendment）**：把「冻结」拆成三个时点，消除「必须先实现/必须先看 validation 才能填数」
与「冻结前不得执行」之间的循环：

| 阶段 | 时点 | 允许看到 | 必须完成 | 允许写入 | 禁止 |
|---|---|---|---|---|---|
| **SAP-A** | 任何模型预测之前 | 无模型预测 | 冻结**方法与候选集**（上表 1–5 的网格、选择指标、tie-break、阈值口径、统计与 bootstrap 参数、FROC 操作点、**功效备忘录含 MCID 与来源**），记录 SAP-A 方法块哈希 | 方法块 + `power_memo` + 审阅/哈希字段 | 不得用任何 validation/test 数值做选择 |
| **SAP-B** | 冻结 checkpoint 之后、最终 test 之前 | PI-CAI validation 预测 | 按 SAP-A 的算法与网格在 validation 上**一次性填数**，记录来源哈希与差异审计 | **仅 `sap_b_results`**（阈值/后处理数值 + 5 个 SHA256 + 审计信息）+ 阶段与可见性字段 | 不得改动 SAP-A 方法字段与 `power_memo`（方法块哈希不一致即失败）；不得查看最终 test |
| **SAP-C** | 评估阶段 | 最终 test 预测 | 封存全部配置与哈希与冻结模型集后，对最终 test **一次性评估**，按 §3.1/§13 完整报告 | `sap_phases.sap_c` + 可见性字段 | 评估后任何修改都使该结果失去确认性资格 |

- **SAP-A 方法块哈希**：方法块 = 冻结的全部方法路径（含 `power_memo` 与 `endpoints`/`comparison_hierarchy`）；
  哈希算法 `sha256(canonical_json(method_paths))`，由冻结载体工具打印并写入冻结包；
  SAP-A 冻结后任何方法字段改动都会使 `sap_phases.sap_a.method_sha256` 不一致 → 校验失败，
  且 `--sap-a-baseline` 差异审计会逐路径报出改动。
- **时点纪律（保守口径）**：`power_memo`（含 MCID、MCID 来源、`success_threshold`、效应/方差假设与敏感性范围）
  **全部属于 SAP-A**；SAP-B **不得修改** `power_memo`；当前**没有**任何允许在 SAP-B 填写的盲态 nuisance
  quantity 白名单字段——若确需新增，必须先提出理由与精确字段白名单并升版登记。
- 预测可见性必须**分字段**记录：`sees_validation_predictions`（SAP-B COMPLETED 后置 true）、
  `sees_final_test_predictions`（SAP-C COMPLETED 后置 true）；兼容字段 `sees_model_predictions` 必须**恒等于**
  后者（由 `zonal_reliability_fusion.protocols.g0_sap` 的校验器强制）。
- `protocol.status=FROZEN` **仅表示 SAP-A 冻结**；SAP-B/SAP-C 的状态只由 `sap_phases.*.status` 表示。
- 若 SAP-B 中发现必须**改变方法**（而非只是填数），该变更必须升版并说明原因，且相关结果只能作为探索性证据；
- 在 SAP-A 冻结之前，任何「同一阈值下比较」或「为各模型分别调阈值」的做法都不得写入论文主结果。

FROC 主操作点已冻结为 `0.5 / 1.0 / 2.0 FP/exam`（§11.1），其余曲线仅作补充展示。

AP 使用候选病灶连续置信度排序，不把某个 Dice 二值阈值后的固定预测当作完整 AP 输入。

**配置载体**：上述评测配置必须落盘为版本化文件（`configs/protocols/g0_sap.yaml`，协议说明
`docs/protocols/G0_SAP.md`），并记录内容哈希；功效/精度备忘录字段一并纳入 G0-SAP 通过条件（§14）。
冻结载体由 `scripts/evaluate/prepare_g0_sap_freeze.py` 生成/校验（静态硬约束、`--check-freeze` 就绪检查、
功效备忘录模板、内容哈希、输出路径隔离）。**该工具与评测逻辑的实现/执行状态见 `docs/STATUS.md`；
本协议只规定它们必须满足的要求。**

---

## 12. 分区—序列机制实验

### 12.1 病灶分区规则

参考病灶默认使用二值标签的 3D 26-连通域定义。对每个参考病灶 $L$ 计算：

$$
c_Z=|L\cap(PZ\cup TZ)|/|L|,
$$

$$
q_{PZ}=\frac{|L\cap PZ|}{|L\cap(PZ\cup TZ)|},\qquad
q_{TZ}=\frac{|L\cap TZ|}{|L\cap(PZ\cup TZ)|}.
$$

- $c_Z<0.5$：outside/uncertain；
- $c_Z\ge 0.5$ 且 $q_{PZ}\ge 0.75$：PZ lesion；
- $c_Z\ge 0.5$ 且 $q_{TZ}\ge 0.75$：TZ lesion；
- $c_Z\ge 0.5$ 且两者均未达 0.75：mixed/boundary。

分层标签来自冻结的 AI 分区，不表述为人工解剖金标准。在不查看模型预测的情况下，必须先报告各集合的病灶分区数、
体积分布、ISUP 分布与标注来源；若 mixed/outside 样本过少，仅作描述性结果。对 `0.67/0.80` 的区域优势阈值做补充
敏感性分析，不事后选择最有利阈值。

### 12.2 Zone × Sequence Ablation

**v2.2 重写（消除“按分区分别训练”的误读）**：序列消融是 **4 个全局输入配置**，每个配置在全体训练集上
**只训练一次**，得到一份全视野预测；随后对同一份预测按 PZ/TZ/mixed/outside 分层**评价**（而不是在每个分区
分别训练一个模型）。

4 个全局输入配置（均为同一 backbone，仅输入序列集不同）：

- `all`：T2W + ADC + HBV；
- `−T2W`：ADC + HBV；
- `−ADC`：T2W + HBV；
- `−HBV`：T2W + ADC。

执行与评价规则：

1. **训练**：4 个配置各训练一次（同一冻结 split、同一采样/损失/预算/推理协议，遵守 §9.3/§9.4），得到 4 份全体积
   预测；**不得为 PZ/TZ/mixed 各训练一套模型**，也不得按分区切换不同网络；
2. **分层评价**：用 §12.1 的病灶分区规则把参考病灶划为 PZ/TZ/mixed/outside，在**同一份预测**上分别汇总指标
   （case-wise Dice/AP/sensitivity 等）；**分层是评价维度，不是训练维度**；
3. **机制读取**：比较 `all` 与 `−T2W/−ADC/−HBV` 在各分层的性能变化模式；该比较是探索性机制证据，
   不单独作为 H1 的确认性终点；
4. **序列移除方式预先固定**：主方案优先**重新训练对应输入组合**；**推理期置零**只能称为 `stress test`，
   不能替代重新训练，也不得混入主消融链；
5. 输入组合实验优先复用同一 concat backbone（§10.3），单种子筛查后再对用于结论的配置补足预定 seed（§10.4）。

### 12.3 Anatomy使用方式对照

- no anatomy（M2）；
- anatomy concat（M3）；
- zone-conditioned gate（M4）；
- hard-zone routing（H0，可选）。

该组实验直接回答“解剖信息应当如何进入网络”。

### 12.4 先验来源敏感性与先验独立性

- Yuan23（主，训练与推理均用 Yuan23）；
- HeviAI23（来源敏感性：使用同一预定 seed 从头训练并推理 M4-Hevi，与同 seed M4-Yuan 对照）；
- no anatomy；
- 可选真实softmax prior；
- 可选轻度prior perturbation（腐蚀/膨胀/patch-drop）。

将 Yuan 训练的 M4 在推理时直接换成 Hevi 只能称为 `cross-prior stress test`，它衡量推理期先验偏移，不与重新训练的
来源敏感性等价。

**先验独立性规则（v2.2 新增，与 G0-E 联动）**：M4 的收益只有在分区先验与目标病例独立时才能归因为“解剖条件
融合”，否则可能只是 memorization/泄漏：

1. **必须审计分区器训练数据重叠**：G0-E 需核实 Yuan23、HeviAI23 或其他分区器的训练数据是否覆盖 PI-CAI
   （尤其 validation/test 病例）或外部测试病例，并记录可溯源证据（论文/权重/代码/许可证）；
2. **无重叠时才可作独立自动先验**：只有当分区器未接触目标病例时，其输出才能作为主分析的独立自动先验；
3. **有重叠时的主分析降级**：若分区器训练集与目标病例重叠，主分析必须改用 **out-of-fold 预测**或**未接触
   目标病例的独立分区器**；无法证明独立时，相关结果只能标记为 `oracle/upper-bound` 或 `prior-friendly
   analysis`，**不得作为确认性结果**（不能用于 H3 的独立确认）；
4. **形态学扰动的语义边界**：腐蚀、膨胀、patch-drop 等扰动**只能评价先验鲁棒性**（模型对先验质量下降的
   敏感性），**不得被描述为消除或量化训练集重叠泄漏**；两者是不同问题，不得混用；
5. 无法溯源分区器训练数据时，必须在正式报告与外部结论中显式降级（与 §5.3 一致）。

若更换先验来源后性能显著崩溃，应降低方法稳定性结论，并优先改进先验处理而不是继续增加融合模块。

**审计要求与待冻结项（§20.2 D8）**：Yuan23 / HeviAI23 的训练数据覆盖情况必须完成审计并写入 G0-E 记录
（审计结论与当前证据状态见 `docs/STATUS.md` 与 `configs/protocols/g0_e_independent_test.yaml` 的状态字段）。因此：

- 在任何 M4 结果被解释为「解剖条件融合收益」之前，必须先完成该审计并写入 G0-E 记录；
- 若无法证明独立，M4 相关结果只能标记为 `prior-friendly` 或 oracle 分析，**不得**用于 H3 的确认性判定；
- 审计的最小证据要求：分区器论文/官方说明中的训练集描述、权重与代码来源、许可证，以及是否包含
  PI-CAI 或外部测试病例的比对记录（可由 `docs/runbooks/g0_e_candidate_audit.md` 的清单承载）。

### 12.5 Gate诊断

至少记录：

- 三个序列的空间权重分布；
- PZ、TZ、病灶与背景内的权重统计；
- 权重熵和单序列塌缩；
- 序列移除后的性能变化是否与权重趋势一致；
- 不同分区先验来源下的权重稳定性。

权重图只能作为机制诊断，不能单独证明模型有效或符合临床因果关系。

---

## 13. 统计分析与结果冻结

1. **分析单位**：拆分一律以患者为 cluster；同一患者的多个 study 随患者一起重采样，不当作独立观测。
2. **Dice**：多 study 患者先在患者内对阳性 study Dice 取算术平均，其阴性 study 只进入 FP 指标；对每个冻结 seed
   再计算患者级配对差。主效应是三个预定 seed 的平均配对差，然后以患者为单位做 paired cluster bootstrap 95% CI。
   **该 CI 的准确定位是 `conditional on the frozen trained seeds`（v2.2）**：它刻画的是"在已冻结的这 3 个训练
   seed 下"的患者间不确定度，不覆盖 seed 采样不确定度。因此必须同时完整报告：
   ① 全部 seed-specific 效应（3 个配对差与各自的患者级 CI）；
   ② 跨 seed 均值、SD 与取值范围；
   ③ 效应方向是否由单一极端 seed 主导（§3.1 成立条件的核查项）。
   若增加 patient×seed two-way/multi-bootstrap，只能作为**敏感性分析**，且必须声明 3 个 seed 不足以稳定估计
   seed 分布（尤其尾部），**不得**把 3-seed bootstrap 包装成强确认性的 seed-level CI。
3. **AP**：AP 不是病例级可加指标。每个 bootstrap replicate 必须重采样患者及其全部 study，然后在重构的整个 cohort
   上分别重算 M4 与对照模型的 AP，最后取配对差；不构造虚假的 per-case AP。
4. **lesion/FROC**：同样按患者 cluster bootstrap，不把同一患者的多个病灶当作完全独立样本。
5. **主比较**：按 §3.1 的 M4−M2 → M4−M3 层级顺序进行；PZ/TZ、输入组合、其他指标和阈值敏感性属于预先声明的
   次级/探索性分析，不事后抬升为主终点。
6. **数据角色**：validation 用于 checkpoint、阈值与有限超参选择，因此不包装成独立 test；最终确认只在通过 G0-E 的
   同任务 external test 或事先重划并冻结的 PI-CAI internal test 上进行。
7. **多阅片者/多标签**：各来源分别报告，不事后选择更有利者；主阅片者或聚合规则须在观察预测前冻结。
8. **可复现性**：保存每个 seed 的病例级预测、候选病灶、指标、bootstrap 索引/种子与完整统计配置；不只报告单次最好 Dice。

**功效/精度备忘录（v2.2：G0-SAP 通过条件的一部分）**：在 G0-E/G0-SAP 完成时，**不查看任何最终 test 预测**，
利用病例数、阳性数、病灶数与分区数估算主效应 95% CI 的可达精度。备忘录必须写明：主要终点、阳性患者数、
假定的配对效应、假设的 paired-difference SD 及其来源、由此得到的最小可检测效应（MDE）与预期 95% CI 宽度。
规则：

- **不得使用观察到的 M4 效果反向制定成功阈值**；阈值必须在看到正式模型结果前写下；
- 缺少可靠方差来源时，只能使用文献范围或**盲态** nuisance estimate，并做参数范围敏感性（给出乐观/悲观区间）；
- 若外部样本量不足（或语义不等价）以至无法支持确认性判定，必须在**看结果前**降级为探索性外部验证，
  而不是在结果不显著后才更改定位；
- 数据不足时，备忘录只保留模板与字段、显式写 `INSUFFICIENT DATA`，**不得伪造具体 MDE 数值**。

备忘录字段与模板见 `docs/protocols/G0_SAP.md`（草案）；落盘配置见 `configs/protocols/g0_sap.yaml`。

---

## 14. 阶段门定义与失败转向

> **本节只定义门**（目的 / 通过条件 / 失败转向）。**状态、证据与 blocker 见 `docs/STATUS.md`**；
> 触发条件与命令见 `docs/runbooks/`。

| 门 | 目的 | 通过条件（摘要） | 失败转向 |
|---|---|---|---|
| **G0-P** | PI-CAI 数据可用 | §6 的 8 项完成，判定写入 `docs/PICAI_Model_Readiness_Audit.md` | 回退修正数据/审计 |
| **G0-R** | 序列错位协议冻结 | §4.2.1 对齐 QC 完成 + `frozen_decision` 冻结（含预处理身份） | 修正预处理、重建派生数据/plan；旧 run 只作 feasibility |
| **G0-E** | 独立测试路径冻结 | 候选审计完成 + 路径 A/B/C **预先**冻结 + 先验独立性审计 | 更换候选；全部不可用则重划 internal test（不得用 validation 冒充） |
| **G0-SAP** | 评测与统计冻结 | SAP-A 冻结方法与候选集（§11.4）+ SAP-B 在 validation 上一次性填数并完成差异审计 + §3.1/§13 判定规则与功效/精度备忘录 | 看结果前降级为探索性分析并改写成功标准 |
| **G1** | 官方基线可复现 | G0-R/G0-E 已冻结；N0 稳定训练、推理并输出预定指标 | 排查数据/增强/推理链路，**不加入创新模块** |
| **G2** | 自研基线可信 | §9.4.2 前置 1–8（含峰值显存**实测**） | 排查 shortcut/shape/旁路/显存；必要时统一缩减 block 并升版 |
| **G3** | 动态融合成立 | 3 个冻结 seed 上 M2 优于 M1/M0，且无 gate 塌缩与容量混杂 | 退回 concat/等权，排查归一化、旁路与融合尺度 |
| **G4** | 解剖条件开发证据成立 | 3 个 seed 上 M4 优于 M2/M3；换源不过度敏感；容量受控 | 按失败模式收缩论文主张（见 §14.1） |
| **G5** | 独立确认完成 | G0-SAP 通过 + 冻结模型集在最终 test 一次性评估 | 区分任务/序列/先验差异；**不得**用 test 返调模型 |

### 14.1 逐门细目（完整通过条件与补充约束）

- **G0-P**：manifest、标签规则、几何与分区审计、患者级 train/validation 全部完成；方向码与训练数据物化完成
  （细目见 §6）；判定与证据写入 `docs/PICAI_Model_Readiness_Audit.md`。
- **G0-R**：执行路径 = G0-R Automated v0.2（协议 `docs/protocols/G0_R_ALIGNMENT_QC_AUTOMATED.md`，配置
  `configs/protocols/g0_r_alignment_qc_automated.yaml`；四组证据 A–D，零人工判读，只输出候选）。人工双阅片协议
  `docs/protocols/G0_R_ALIGNMENT_QC.md` 保留为历史草案与后备。候选为 `resample+rigid` 时必须新建派生数据并重新
  预处理；为 `INSUFFICIENT_EVIDENCE` 时必须停止。证据边界见 §4.2.1。**冻结前不得启动 P2B、正式 M0 或 N0。**
- **G0-E**：完成外部候选的数据身份、标签语义、三序列映射、解剖先验协议、指标范围与排除规则审计，并在
  「合格同任务 external test」与「重划并冻结 PI-CAI internal test」之间**预先冻结其一**；患者重叠排查（必须为 0）
  与分区器训练数据独立性审计（§5.4 / §12.4）是判定的必要组成。工具缺证据一律 `UNKNOWN`，不代填 `verdict`；
  载体：`docs/protocols/G0_E_INDEPENDENT_TEST.md` + `configs/protocols/g0_e_independent_test.yaml`。
- **G0-SAP**：按 §11.4 的三阶段生命周期执行——**SAP-A**（看模型结果前）冻结方法与候选集（阈值网格/选择指标/
  tie-break、共同 vs 每模型独立阈值、后处理候选、`picai_eval` 参数、统计与 bootstrap 规则、FROC 操作点）、
  §3.1/§13 的主终点与判定规则（含 MCID 与来源）、§11.1 的实现字段/语义终点命名映射，并通过功效/精度备忘录
  （§13），随后记录 **SAP-A 方法块哈希**（`sap_phases.sap_a.method_sha256`）并置
  `protocol.status=sap_phases.sap_a.status=FROZEN`（顶层 FROZEN 仅表示 SAP-A）；
  **SAP-B**（checkpoint 冻结后）在 PI-CAI validation 上按 SAP-A 的算法一次性填数到**独立的 `sap_b_results` 块**
  （含阈值/后处理数值、来源哈希与差异审计哈希），并完成"仅数值变化"差异审计（`--sap-a-baseline`）；
  **SAP-C** 属 G5（封存 + 最终 test 一次性评估）。冻结载体 `scripts/evaluate/prepare_g0_sap_freeze.py`；
  SAP-A 冻结前**不得**在真实 validation/test 上运行评测，也不得用 `scripts/evaluate/`、nnU-Net `summary.json`
  或任何临时脚本数值作为论文主结果。
- **G1**：G0-R 已通过、G0-E 已冻结独立测试路径，且 N0 能够稳定训练、推理并输出预定分割/检测指标；完成后以
  `checkpoint_final.pth` 做全量验证（禁用 `--val_best`），按 §11.1/§11.4 用冻结 `picai_eval` 口径计算论文指标。
  nnU-Net 自带 `validation/summary.json`（223 例 `nanmean`）与自研阳性病例 Dice **不可直接比较**；N0 只作
  contextual reference，不作因果基线（§7.1、§10.2）。
- **G2**：完成 §9.4.2 的 **G2 前置 1–8 项**（含 M0 峰值显存**实测**），并冻结 M0 骨干/训练配置哈希与 M1–M4 共用
  skip 接口契约；M1–M4 的完整模型哈希在各自实现后、首次运行前逐一冻结，**不得**为尚未实现的模型伪造哈希；
  未做配平实验时不得设置「M0 必须达到 N0 的某个百分比」作为硬阈值。
- **G3**：在预定 3 个种子上，M2 相比 M1/M0 出现稳定正向开发证据，且**不存在 gate 塌缩与容量混杂**（§8.8）；
  单种子 feasibility 结果不能单独宣布通过或失败。
- **G4**：在预定 3 个种子与 PI-CAI validation 上，M4 相比 M2、M3 出现稳定正向证据；分区换源后结论不过度敏感
  （§12.4 / G0-E）；容量按 §8.4 / §8.8 控制。**G4 只决定是否把冻结模型送入最终 test，不把 validation 结果包装成
  H3 的独立确认。** 失败转向（按模式）：
  - M3 有效但 M4 无效 → 收缩为简单解剖辅助分割；
  - M2 有效但 M3/M4 无效 → 保留图像驱动多序列融合，分区仅用于结果分析；
  - M1/M2 均无效 → 回到 concat，转向小病灶采样、损失或数据质量问题；
  - M4 只在单一先验有效 → 降低主张，优先研究先验稳定性；
  - **不得**把阴性结果包装成解剖条件融合创新。
- **G5**：G0-SAP 已通过；冻结的 M2/M3/M4 模型集（全部预定 seed）在通过 G0-E 的最终 test 上完成**一次性**评估；
  按 §3.1/§13 报告全部预定指标、置信区间与失败病例。**只有 G5** 才可将 H3 表述为「得到独立确认」。
  外部结果较弱不意味着 PI-CAI 数据工作作废；须区分任务标签差异、序列定义差异、分区先验质量与真实模型泛化不足；
  **不得**利用 test 结果返调模型。

---

## 15. 依赖顺序与实施阶段

以下编号表示**门控依赖**，不是要求所有工作机械串行。P0C、P0D、P0E 与 P2A 均不依赖模型结果，可以并行推进；
模型训练、性能比较和后续模块实现仍严格服从各自前置门。不得用“可并行开发”解释为可以绕过 G0-R/G0-E/G0-SAP
或 G2 查看正式结果。

关键硬依赖统一如下：

| 动作 | 必须先满足 |
|---|---|
| P2A 新骨干代码与合成测试 | G0-P PASS；不依赖 G0-R/G0-E/G0-SAP |
| P2B 新 M0 真实数据诊断（不作为论文结果） | P2A 1–7 完成 + G0-R 已冻结预处理身份 |
| 正式 N0 训练 | G0-R PASS + G0-E 路径冻结 + N0 预算/checkpoint 规则冻结 |
| 正式 M0 训练 | G2 PASS + G0-R PASS + G0-E 路径冻结 + 增强/预算协议冻结 |
| M1–M4 实现 | G2 PASS；各模型运行前完成 §9.4.2 后置对应项 |
| 查看论文指标或进入最终 test | G0-SAP PASS；最终 test 另须 G0-E 路径冻结 |

阶段定义（**执行状态与阻塞项见 `docs/STATUS.md`；命令见 `docs/runbooks/`**）：

1. **P0A 审计修正**：修正 LPS 方向码与测试，更新 manifest/geometry audit；
2. **P0B 数据物化**：重采样、canonical 二值标签、异常病例 QC、train/val 新文件；
3. **P0C 序列错位决策**（对应 G0-R）：运行 G0-R Automated v0.2 自动 QC → 冻结 `frozen_decision`
   （`resample-only` / `resample+rigid`）；人工双阅片路径降级为历史草案与后备；
4. **P0D 外部审计**（对应 G0-E）：审计 Prostate158 等候选，核实私有 PCA 语义，决定是否需要 PI-CAI internal test；
5. **P0E 评测与统计冻结**（对应 G0-SAP）：实现版本化评测配置/脚本与合成测试，填写功效备忘录并冻结；
6. **P1 官方基线**（对应 G1）：完成 N0 = PI-CAI official-style nnU-Net v2 Focal+CE NoFFT；启动条件见 §6；
7. **P2A v2.3 骨干实现**：实现 plan-driven Residual-Encoder M0、版本化架构配置、结构摘要、参数/FLOPs、
   checkpoint 架构身份与 §9.4.2 的 G2 前置 1–7 项；旧 PlainConv M0 保留为 legacy，不覆盖；
8. **P2B 自研基线验证**（对应 G2 第 8 项）：G0-R 冻结预处理身份后，由研究者对新 M0 执行 loader smoke、
   small-overfit、诊断性首次全体积验证并**实测峰值显存**；随后冻结骨干、skip 接口、训练预算与 M0 配置哈希；
9. **P3 容量控制**：实现三个浅层 residual stem、§8.6 统一 skip 聚合与 stage 2 等权融合 M1
   （含 §8.8 要求的容量对照决策）；
10. **P4 动态融合**：在同一骨干上实现 M2；
11. **P5 解剖条件**：实现共享 zone encoder 的 M3 与 conditional-affine M4；
12. **P6 机制实验**：序列组合、PZ/TZ 分层、序列移除、先验换源、gate 诊断；
13. **P7 可选对照**：同网络 deterministic H0 hard-zone routing；
14. **P8 正式复验**：预定关键模型 3 个种子与配对统计；
15. **P9 独立测试**：冻结模型集后一次性评估（G5）；
16. **P10 论文整理**：图表、置信区间、失败病例、局限性与贡献边界。

每阶段必须保存配置、日志、结构化指标与退出条件。未经阶段门验证，不并行堆叠后续模型模块。

---

## 16. 工程与复现要求

- 原始数据只读，派生数据写入`data/interim/`或`data/processed/`；
- manifest与split写入`data/metadata/`和`data/splits/`；
- 模型、训练、评测与第三方适配分别放入既定`src/`子目录；
- 第三方仓库保持不修改，适配层写入项目代码；
- 每次运行保存配置、代码版本、随机种子、环境、manifest校验值和checkpoint；
  **当前主项目目录不是 Git 仓库**（见本节末尾 blocker），因此在版本控制方案确定前，「代码版本」只能以人工记录的
  代码指纹代替，且该 run 不得并入正式确认性比较；
- 模态顺序固定为T2W、ADC、HBV，并由单元测试保护；
- 单元测试至少覆盖几何同步、标签映射、residual shortcut/stride/channel projection、各 stage 输出 shape、§8.6
  浅层 skip 聚合输入边界、deep supervision、gate 权重和为1、患者泄漏与滑窗恢复；
- `experiment_log.md`只记录实际运行内容，不把计划结果写成事实；
- `Development_Log.md`记录代码和协议变更；
- 失败实验保留配置和结论，避免重复试错和选择性报告；
- split、预处理身份、阈值、后处理、统计分析计划和外部测试协议冻结后，任何修改必须升版本并说明原因；
- 每个正式 run 保存 split/manifest/plan、训练配置、G0-R 预处理身份与 G0-SAP 评测配置的内容哈希；任一哈希不一致时，
  不得把结果并入同一确认性比较。
- 每个 M0–M4 checkpoint 必须保存 residual block 配置、skip 聚合协议、融合 stage、参数量与架构哈希；PlainConv 与
  Residual-Encoder checkpoint 互相拒绝加载，不允许通过非严格加载掩盖架构不一致。
- **代码版本记录（正式实验前 blocker）**：主项目目录当前**不是 Git 仓库**，正式 run 无法可靠保存代码 commit，
  复现性不足。处理要求：① 在任何正式 run 之前，由**用户**决定版本控制方案（初始化 Git 或采用等价的代码指纹/
  快照记录机制）；② 该方案确定前执行的 run 只能标记为 feasibility；③ **代理不得擅自 `git init` 或改动版本控制
  配置**。该项状态与证据见 `docs/STATUS.md` B1。

### 16.1 环境、命令归属与进度要求

**本节不复述协作规则**。代理可执行范围、命令归属（数据处理、训练、验证、推理、评测与批量分析一律交由研究者
运行）、conda 环境 `lm`、缺失依赖的安装规则与进度条要求，**一律以项目根目录 `AGENTS.md` 为唯一权威来源**。

本文档只补充两条协议侧要求：

1. 任何交付给研究者运行的命令，必须附工作目录、环境激活方式、预期输出、进度观察方式与成功判据；
   命令模板统一维护在 `docs/runbooks/`，本文档不再内嵌长命令；
2. 长任务必须以结构化日志记录成功数、失败数、跳过数、耗时与输出路径；进度条不能替代结构化日志。

---

## 17. 明确不做的内容

第一版不加入：

- 跨中心训练目标、LOCO或域泛化模块；
- domain adversarial learning、feature randomization或consistency training；
- 专门的modality corruption与可靠性损失；
- 新的分区分割网络或病灶—分区联合训练；
- Patch2Space式局部对应、可变形注意力或配准网络；
- 硬编码PZ→DWI、TZ→T2W作为最终方法；
- Transformer、Mamba、扩散模型或foundation model；
- anatomy-conditioned/attention skip、UNet++嵌套skip、UNet3+全尺度skip或复杂注意力decoder；
- 三个完整独立encoder和多层cross-attention；
- 强制gate权重符合人工设定；
- 同时改变backbone、loss、sampling、augmentation和fusion；
- 未验证的临床解释、“首次”声明和域泛化结论。

---

## 18. 预期贡献与论文表述

若H3成立，预期贡献可表述为：

1. 提出一种解剖分区条件的空间自适应多序列融合模块，使冻结的PZ/TZ先验直接参与T2W、ADC、HBV的融合决策，而不是仅作为附加输入或硬路由规则；
2. 通过结构容量控制、普通解剖输入对照、分区—序列消融、先验换源和外部数据评估，系统验证解剖条件融合的有效性及适用边界。

Residual-Encoder、普通 decoder、标准 skip，以及 M1–M4 共用的 §8.6 浅层 skip 聚合，均为工程骨干，不列为论文
创新，不使用“提出全新 U-Net”或“创新 decoder/skip”表述。推荐方法描述为：

> 在成熟的 plan-driven Residual-Encoder 3D U-Net 中，于 encoder stage 2 插入 PZ/TZ 条件的空间自适应多序列
> 融合模块；后端使用共享 residual encoder、普通 decoder 与标准 skip 完成病灶分割。

只有H2成立时，论文收缩为图像驱动的多序列空间融合，PZ/TZ仅用于分层分析。

只有M3成立而M4不成立时，论文收缩为解剖先验辅助病灶分割，不声称“解剖条件融合”。

最终题目、摘要和贡献必须由结果决定。

---

## 19. 主要风险

| 风险 | 影响 | 预防/处理 |
|---|---|---|
| T2W/ADC/HBV 重采样后仍有错位 | 逐位置 gate 融合不同组织，破坏机制解释 | G0-R 对齐 QC，冻结是否固定刚体配准，补充位移敏感性 |
| 外部测试标签与PI-CAI不一致 | 无法直接比较或结论失真 | G0-E语义审计，限制指标和表述 |
| 外部样本量/语义不足以确认 H3 | validation 变成事实上的 test | 正式 M0–M4 前冻结 PI-CAI internal test，或事先降级外部分析 |
| 外部集缺少公平的自动PZ/TZ先验 | M4外部评估不公平 | 冻结统一先验模型；人工分区仅作oracle分析 |
| AI分区先验质量不稳定 | M4依赖特定先验 | 双来源审计、换源实验、先验扰动 |
| Pooch25 与旧标注轮廓风格、编码批次不同 | 模型或指标受标注批次影响 | Pooch25 标签为官方二分类 0/1（与旧标注 0/2/3/4/5 不同），混用必须显式映射并由测试锁定；报告来源计数与阳性子组敏感性 |
| Residual-Encoder 在 `[16,320,320]` patch 下显存超限 | 无法保持 batch/patch 与模型公平性 | 先按 ResEnc-M 级预算做合成显存审计；只统一缩减所有 M0–M4 的 block 数，不单独缩减 M4，不直接上 L/XL |
| 浅层 skip 聚合形成 gate 旁路 | M4 的条件 gate 影响被低层特征绕过或难以解释 | §8.6 固定低容量 `concat→1×1×1` 聚合；所有模型一致；报告 gate 诊断且不把权重称为全网络贡献 |
| 旧 PlainConv 与新 Residual checkpoint 混用 | 结果不可复现或错误恢复 | 架构哈希与严格 resume 校验；旧输出目录只读保留，新架构使用独立 model/config identity |
| 自研M0明显弱于N0 | 创新增益无法可信归因 | 先通过G2，必要时接入nnUNetTrainer |
| N0 与 M 系列预算/checkpoint 不匹配 | N0−M0 排名不具因果可比性 | N0 仅作 contextual reference；若需排名则增加 matched protocol |
| Gate单序列塌缩 / 权重近似静态 | 权重失去区分力，H2 结论不成立 | 投影归一化、熵与分布诊断、序列移除；**必报**跨病例 gate 权重分布（邻接的模态门控工作报告过卷积 backbone 上门控塌缩为近静态先验，arXiv:2604.10702 摘要，需人工确认正文） |
| M2 vs M1 的容量混杂 | 把容量增益误读为动态融合增益 | §8.8：预先声明同容量对照，或把 H2/G3 降级为探索性证据（§20.2 D6） |
| 成功标准缺少 MCID | 统计显著但实际可忽略的增益被写成成功 | §3.1 / §13：在看结果前冻结 MCID 并把 CI 相对 MCID 的位置写进主判定（§20.2 D4） |
| 阈值口径未冻结（共同 vs 每模型独立） | 主比较口径不一致，结果可被质疑 | §11.4 / §20.2 D3：G0-SAP 冻结时必须写明并记录理由，不得事后切换 |
| G0-R 证据不足（16 例、相对指标、合成校准） | 错位决策可能不代表全体数据 | 结论必须附证据边界；必要时追加人工 landmark 复核或扩大抽样（§20.2 D2）；`INSUFFICIENT_EVIDENCE` 时必须停止 |
| Anatomy concat已经足够 | M4创新必要性不足 | M3严格控制，若M4无增益则执行失败转向 |
| 病灶小、Dice方差大 | 总体结果不稳定 | AP、敏感度、FP/exam、分层与bootstrap |
| 外部test被反复用于调参 | 结果失去独立性 | 一次性冻结评估，所有选择仅用PI-CAI validation |
| 全体积验证频率过高 | 训练时间被验证主导 | 每 5 epoch 验证；正式主实验禁用“连续无改善”early stopping、固定 200-epoch 预算，只保留安全中止（§9.4） |
| 固定 200-epoch 预算下模型未收敛 | M0–M4 结果偏弱但可比 | 预算调整只允许依据 M0 曲线与耗时，在 M1–M4 前冻结并对所有模型/seed 统一适用（§9.4） |
| 分区先验训练集与目标病例重叠 | M4 收益实为记忆/泄漏，不可归因 | G0-E 先验独立性审计；有重叠时改用 out-of-fold/独立分区器，无法证明独立时降级为 prior-friendly（§12.4 / §20.2 D8） |
| 主项目目录无版本控制 | 正式 run 无法记录代码 commit，复现性不足 | 正式实验前由用户决定版本控制/代码指纹方案（§16、`docs/STATUS.md` B1）；代理不得擅自 `git init` |
| 新颖性表述缺乏系统检索支撑 | 审稿人质疑“首次”类声明 | §2.2：冻结检索协议与截止日期；只用“据我们所知”级别表述；提交前更新检索（§20.2 D10） |
| 训练组合过多 | 时间与算力失控 | 阶段门；单种子只做工程 feasibility；确认性模型预先冻结 |

---

## 20. 运行入口与待冻结决策

### 20.1 运行入口（命令与成功判据）

本文档**不内嵌 shell 命令**。所有由研究者执行的命令、输出路径、进度观察与成功判据统一维护在 `docs/runbooks/`：

| 入口 | 对应门 / 阶段 |
|---|---|
| `docs/runbooks/g0_r_alignment_qc.md` | G0-R 自动 QC 与 `frozen_decision` 冻结 |
| `docs/runbooks/g0_e_candidate_audit.md` | G0-E 候选审计与路径 A/B/C 冻结 |
| `docs/runbooks/g0_sap_freeze.md` | G0-SAP 评测/统计冻结 |
| `docs/runbooks/p2b_m0_validation.md` | P2B / G2 真实数据三步验证（含峰值显存实测） |
| `docs/runbooks/n0_training.md` | G1 正式 N0 训练与续训 |

当前状态、blocker 与下一步优先级**唯一权威来源**：`docs/STATUS.md`。

### 20.2 待冻结决策表（必须由研究者决定，代理不得填数）

| ID | 决策 | 为什么必须冻结 | 冻结位置（建议） | 可选方案（**不预设结论**） |
|---|---|---|---|---|
| D1 | G0-E 独立测试路径：A（合格 external test）/ B（事先降级为探索性）/ C（重划 PI-CAI internal test） | 决定 H3 确认性证据来源，以及现行 1277/223 split 是否作废 | `configs/protocols/g0_e_independent_test.yaml::decision_tree.frozen_decision` | 见 `docs/protocols/G0_E_INDEPENDENT_TEST.md` §9 |
| D2 | G0-R 证据是否充分；是否追加人工 landmark 复核或扩大抽样 | 16 例自动相对指标不足以代表全体 1500 例 | G0-R 协议载体（自动 + 人工草案） | 接受自动结论并声明边界 / 追加人工复核 / 扩大抽样 |
| D3 | 阈值口径：共同阈值 vs 每模型独立阈值；后处理、bootstrap 次数与种子、`picai_eval` 参数 | 主比较口径必须在看结果前固定 | `configs/protocols/g0_sap.yaml` | 见 §11.4；两种方案都必须「事先声明 + 同规则执行」 |
| D4 | 最小实际/临床相关差异（MCID）与主判定规则 | 仅「CI 不跨 0」会把可忽略增益写成成功 | `configs/protocols/g0_sap.yaml::power_memo.success_threshold` | 文献范围 / 专家共识 / 盲态 pilot；无法确立时降级为精度声明 |
| D5 | 实现字段是否重命名以对齐语义终点（`val_positive_casewise_dice_mean` → `positive_casewise_dice` / `test_*`） | 改名将使既有 checkpoint、CSV、配置与哈希失配 | `docs/protocols/G0_SAP.md` §1.1 + 相关配置 | 保持现名只做语义映射 / 升版重命名并重跑回归 |
| D6 | M1/M2 容量混杂控制：同容量对照 or 降级 H2 为探索性 | 决定 H2/G3 能否作为确认性证据 | §8.8 + M1/M2 配置 | 同容量 constant-gate 对照 / 显式降级 |
| D7 | N0 与自研预算差异的冻结方式；`augmentation.pending_parity` 6 项是「对齐」还是「接受差异并声明」 | 决定 N0−M0 的可解释性与配平声明 | N0 配置 + §9.1 | 保持 1000 epoch 并声明 / 改用官方 250-epoch 变体 / 增加 matched-protocol run |
| D8 | Yuan23 / HeviAI23 先验独立性审计结论与降级策略 | 先验泄漏会使 M4 收益不可归因 | `configs/protocols/g0_e_independent_test.yaml::candidate_priors` | 独立则用 / 有重叠改 out-of-fold / 无法证明降级 `prior-friendly` |
| D9 | 峰值显存超限时的统一缩减方案（`blocks_per_stage`）与新架构哈希 | 影响全部 M0–M4 的可训练性；不得只缩减 M4 | `configs/architectures/m0_resenc_v23.yaml`（升版） | 统一缩减并升版重跑合成回归 / 其他对所有模型一致的方案 |
| D10 | 文献系统检索协议与检索截止日期 | 决定新颖性表述的可辩护程度 | `docs/literature/` + §2.2 | 见 `docs/literature/novelty_and_standards_check_20260917.md` §6 |

### 20.3 纪律提醒

- 上述任一决策若在查看对应结果之前未冻结，相关结果只能作为**探索性证据**；
- 本表只列「待决定」，**不预设答案**；代理不得代为填写数值或替研究者选择；
- 决策完成后：更新对应协议载体与内容哈希 → 记入 `docs/protocol_changelog.md` → 更新 `docs/STATUS.md`；
- 本文档旧版中的「近期下一步」逐条清单已迁出：执行顺序见 §15，实时状态与优先级见 `docs/STATUS.md`。

---

## 21. 关键参考资料

**完整清单（含逐条注释、报告规范与邻接工作）见 `docs/literature/references.md`。**
核心来源：

| 类别 | 来源 |
|---|---|
| 任务与临床 | PI-RADS v2.1（ACR）；PI-CAI 主论文（Lancet Oncol 2024, DOI 10.1016/S1470-2045(24)00220-1） |
| **PI-CAI 官方事实（配准/HBV/ADC/许可/标签）** | 官方数据页 <https://pi-cai.grand-challenge.org/DATA/>；标注仓库 <https://github.com/DIAGNijmegen/picai_labels>（含 Pooch25） |
| 基线与评测 | `picai_baseline`、`picai_eval`；nnU-Net（Nature Methods 2020）；nnU-Net Revisited；官方 ResEnc presets |
| 融合基线 | NaMa（AAAI 2024）、NesMFle（TBME 2025）、LeSMI（Med Phys 2025）；多流融合编码器 |
| 分区先验邻接工作 | Z-SSMNet（2025）、zonal-aware attention U-Net（2025）、AtPCa-Net（2024）；模态门控 MIGF（arXiv:2604.10702，需人工确认正文） |
| 文档与报告规范 | SPIRIT 2025（BMJ 389:e081477 / Nat Med 31:1784–1792）；CLAIM 2024（DOI 10.1148/ryai.240300，条目需人工确认） |
| 外部候选 | Prostate158（DOI 10.1016/j.compbiomed.2022.105817） |
| 本次核查记录 | `docs/literature/novelty_and_standards_check_20260917.md` |
