# 文献与规范核查备忘（2026-09-17）

> **本文的性质：一次关键词级、非系统的定点核查记录**，用于支撑 `docs/research_plan.md` 的文档边界重构与
> §2.2 / §19 的风险陈述。**它不构成系统综述，也不构成新颖性检索结论**：
> 未使用数据库检索式（PubMed/Scopus/Embase/IEEE/arXiv API）、未登记检索日期截止、未做 PRISMA 式筛选、
> 未逐篇核读方法正文。任何「首次」「绝对创新」的表述都**不得**基于本文成立。
>
> 核查人：协作代理（只读联网检索）；**结论的采信与进一步核查责任在研究者**。
> 未逐条确认的内容一律标注「需人工确认」。

## 1. 核查问题与结论速览

| # | 核查问题 | 结论（谨慎） | 证据强度 |
|---|---|---|---|
| Q1 | 研究协议应包含什么，运行日志/结果应写在哪里 | 权威指南一致要求：协议 = **计划**（含 SAP 的可获取位置与版本审计留痕）；已完成的实施细节与结果属于报告/日志。本项目据此把运行事实迁至 `experiment_log.md`、工程变更迁至 `Development_Log.md`、版本史迁至 `protocol_changelog.md` | 强（SPIRIT 2025 正文可读取） |
| Q2 | 协议版本与修订如何留痕 | 每一版方案都应包含**透明审计留痕**（日期 + 变更描述）；重要修订应即时上报并在完成报告中描述；方案与结果报告比对可发现选择性报告与未披露修订 | 强（SPIRIT 2025 原文引述） |
| Q3 | AI 医学影像研究的报告规范是否要求计划/结果分离 | CLAIM 2024 是 AI 医学影像研究的**报告清单**（面向已完成研究），与协议文件互补；具体条目编号与措辞**需人工确认官方清单** | 中（官方页面存在；正文未取回，见 §3.2） |
| Q4 | PI-CAI 公开数据的序列是否配准 | **公开训练/开发集（1500）原则上不共配准**；仅 54/9107 训练病例由组织方手动刚体配准；隐藏调优（100）与隐藏测试（1000）全部由组织方配准（其中 85/1000 手动）。→ 本项目的 G0-R 前提成立 | 强（PI-CAI 官方 Data 页） |
| Q5 | PI-CAI 的 HBV / ADC / 标签口径 | HBV = 轴位高 b 值 DWI（**b ≥ 1000 s/mm²**）；**ADC 绝对强度跨中心不可直接解读**；标注为 human_expert（original/resampled）+ Pooch25 补齐；许可 CC BY-NC 4.0；数据含 ProstateX 328 例 | 强（PI-CAI 官方 Data 页 + `picai_labels` 官方仓库 README） |
| Q6 | 现有工作中，PZ/TZ 条件化**空间序列 gate** 是否已被报告 | 关键词级检索**未在摘要层面发现**完全相同的组合；但存在高度相关的邻接工作（见 §4），且**本项目尚未做系统检索与正文级核查**，因此**只能表述为「据本次非系统核查未见同组合报告」，不得称「首次」** | 弱—中（需系统检索与人工确认） |

## 2. 本项目文档边界重构的依据（Q1 / Q2）

来源：Chan AW et al. *SPIRIT 2025 statement: updated guideline for protocols of randomised trials.*
BMJ 2025;389:e081477（DOI 10.1136/bmj-2024-081477；同步发表于 Nature Medicine 2025;31:1784–1792，
DOI 10.1038/s41591-025-03668-w）。以下为核查中可读取的正文要点（原文措辞）：

- 协议是**活文件（living document）**，试验期间常被正式修订；
  「**every protocol version should contain a transparent audit trail documenting the dates and descriptions of changes**」
  （每一版方案都应包含透明审计留痕，记录修改日期与内容描述）→ 对应本项目的 `docs/protocol_changelog.md`；
- 重要修订应及时上报监管机构与注册库，并在完成报告中描述 → 对应本项目在修订时必须写明
  「是否发生在查看结果之前、对既有结果的影响」；
- SPIRIT 关注「**计划**（what is planned）」，CONSORT 关注已完成研究的报告；两者比对可识别
  **选择性报告与未披露的方案修改（如主终点或分析的变更）** → 对应本项目把计划与运行日志分离、
  并把主终点/主比较冻结在 G0-SAP 的做法；
- 「There are often related documents (for example, full statistical analysis plan, data management plan) …
  Any such documents should be referenced in the protocol and made available for review」
  （SAP 等文件应在方案中被**引用**并可获取；SPIRIT 2025 将此写入条目 5）→ 对应本项目
  `configs/protocols/g0_sap.yaml` + `docs/protocols/G0_SAP.md` 与 research_plan §11.4 / §13 的引用关系。

**适用性限制（诚实声明）**：SPIRIT 面向随机对照试验，本项目是回顾性回顾 + 外部测试的算法研究，
CLEAR/CLAIM 等影像报告规范的适用性更直接。本项目只借用其**协议—结果分离、版本留痕、SAP 预先冻结**
这三条原则，不声称遵循 SPIRIT 2025 的正式合规性。

## 3. 报告规范核查

### 3.1 CLAIM 2024（Q3）

- 官方页面：RSNA《Checklist for Artificial Intelligence in Medical Imaging (CLAIM)》，
  https://pubs.rsna.org/page/ai/claim ；2024 更新版文章
  https://pubs.rsna.org/doi/10.1148/ryai.240300 （*Radiology: Artificial Intelligence* 2024）。
- 核查限制：本次抓取该文页面返回 **403 Forbidden**，官方 PDF 无法解析正文，因此**未能逐条核对**条目编号与措辞。
- 谨慎结论：CLAIM 2024 是针对**已完成 AI 医学影像研究**的报告清单，用于保证数据划分、参考标准、
  统计分析、结果与计划的对应关系被完整报告；它与「研究协议」是互补关系，不能替代协议与 SAP 的预先冻结。
- **需人工确认**：条目编号、条目数、以及是否包含「方案注册/可得性」的专门条目。

### 3.2 其他规范

- TRIPOD+AI 等预测模型报告规范在 CLAIM 材料中被提及为配套规范，本次**未核查原文**，不作引用。

## 4. PI-CAI 官方事实核查（Q4 / Q5）

来源 1：PI-CAI Challenge 官方数据页 https://pi-cai.grand-challenge.org/DATA/（组织方维护）。
来源 2：官方标注仓库 `https://github.com/DIAGNijmegen/picai_labels`（README，raw 版本已读取）。

| 事实 | 官方表述要点 | 对本项目的意义 |
|---|---|---|
| 序列配准 | 公开训练与开发集（1500）**原则上不共配准**；绝大多数病例对齐尚可，但存在相当一部分明显偏移；仅 54/9107 训练病例手动配准（ITK-SNAP v3.80，刚性 6 DoF） | 本项目 §4.2.1 的 G0-R 是必要前提；「仅物理重采样」的默认假设必须由 QC 证据支撑 |
| 隐藏队列 | 隐藏调优（100）与隐藏测试（1000）由组织方全部配准（85/1000 手动） | 本项目不能把公开训练集的错位水平外推到 PI-CAI 官方排行榜口径 |
| HBV | 轴位高 b 值 DWI，**b ≥ 1000 s/mm²** | 支持本项目 §1.3 对 HBV 的定义；外部数据的 DWI 需核实 b 值后才能对应 |
| ADC | **绝对强度值跨中心不可直接解读**（采集协议不标准化、缩放不一致；PI-RADS v2 亦提示谨慎） | 支持本项目「分序列归一化、不跨中心比较 ADC 绝对值」的处理；外部数据 ADC 映射要单独审计 |
| 标签 | `csPCa_lesion_delineations/human_expert/original`（标注者/中心不同，部分在 T2W、部分在 DWI/ADC 分辨率）与 `resampled`（统一到轴位 T2W 分辨率），覆盖 **1295/1500**；1500 例中 1075 例良性/惰性、425 例 csPCa，其中 220/1295 应含病灶标注 | 与本项目 §4.3 的「1295 用 resampled」一致；**必须报告标注来源计数** |
| Pooch25 | 官方仓库 2025-07-01 更新：依据 Pooch et al., 2025（medRxiv 预印本）为**此前仅 AI 标注的 205 例阳性**补充人类专家标注；目录 `human_expert/Pooch25/`；标签为**二分类 0/1**（与 `human_expert/` 的 0/2/3/4/5 多分类不同）；使用时须引用该文献 | 与本项目 §4.3 一致；**补充关键细节**：Pooch25 标签编码为二分类，混用两种标注时必须先统一映射（本项目映射为前景 1，符合该口径）；正式报告须给出 train/validation/test 的标注来源计数与分层敏感性 |
| 全腺体 AI 标注 | `anatomical_delineations/whole_gland/AI/Bosma22b`；官方 README 明确提示 AI 分割可能出错，并**举出 `11050_1001070`** 作为示例 | 与本项目 §4.4 对该例的屏蔽一致 |
| 许可与重叠 | 影像 **CC BY-NC 4.0**（非商业）；公开训练集**包含 ProstateX 的 328 例**，不得与 ProstateX 叠加使用 | 支持本项目 §4.5「ProstateX 禁止作为独立外部集」 |
| 队列 | 1476 名患者 / 1500 次检查；中心 RUMC/PCNN/ZGT（公开集）；参考标准：阳性 = 组织学 ISUP ≥ 2，阴性 = ISUP ≤ 1 或 MRI PI-RADS ≤ 2 | 与本项目 §4.1 的 1500/1476 与中心计数一致 |
| 影像获取 | Zenodo，DOI 10.5281/zenodo.6624726 | 引用与复现用 |

**需人工确认**：
1. `picai_labels` README 于 2026-07-13 新增了「常规临床实践中分配的 PI-RADS 评分」字段，README 明确警告其
   为便利抽样、**不代表诊疗标准、不可用于 AI 系统基准测试**——本项目是否引用该字段需由研究者决定；
2. Pooch25 的具体标注质量/阅片流程应以 Pooch et al. 原文（medRxiv 预印本，可能已正式发表）为准，
   本次未核读原文。

## 5. 邻接工作核查（Q6）——新颖性只能谨慎表述

检索方式：关键词级 web 检索（2026-09-17），**非系统检索**；以下只记录标题/摘要层面可核实的信息。

| 工作 | 与本项目的关系 | 证据 |
|---|---|---|
| NaMa（AAAI 2024）、NesMFle（IEEE TBME 2025） | 影像驱动的邻域自适应多模态融合；构成 M2 的最近邻基线，**M2 不作为创新** | DOI 10.1609/aaai.v38i5.28215；10.1109/TBME.2025.3562766（本项目 §21 已引用） |
| LeSMI（Med Phys 2025） | 动态模态加权（Dynamic Modality Weighting）整合 T2W/DWI/ADC；「学习三个序列权重」本身不足以构成创新 | DOI 10.1002/mp.70019 |
| Z-SSMNet（Comput Med Imaging Graph 2025） | zonal-aware 分割网络；证明「使用分区信息」已有工作 | DOI 10.1016/j.compmedimag.2025.102510 |
| 分区 awareness + attention-guided U-Net（DIGITAL HEALTH 2025） | 将分区 awareness 集成到注意力 U-Net；证明「解剖先验进入 decoder/skip」已有工作 | DOI 10.1177/20552076251314546 |
| MIGF / 「Backbone-Conditional Behavior of Modality Gating」（arXiv:2604.10702，2026） | 针对 PI-CAI(1500) + Prostate158 的**模态级门控**融合；5 折、180 模型、门控机制分析。摘要层面**未见** PZ/TZ 或空间分区条件化；其报告的关键现象是：在卷积 backbone 上门控会**塌缩为近静态模态先验**（跨样本权重 SD 0.0033），门控收益依赖 backbone 固有 modality awareness | arXiv:2604.10702（v4, 2026-06-24）摘要页；**方法正文未核读，需人工确认** |

**谨慎结论（必须按此表述）**：

1. 据本次**非系统**核查，**未发现**已报告工作采用「在单个 3D 网络中、用冻结 PZ/TZ 先验直接条件化
   空间序列 gate（conditional affine modulation）」这一精确组合；
2. 但该组合的每个**组成部分**（分区先验作为输入/约束、动态序列加权、解剖引导 decoder/skip、
   解剖条件注意力）都已有公开工作，因此**不得**表述为「首次使用分区」「首次动态融合」或「绝对创新」；
3. 论文中可行的表述上限是：**「据我们所知（to our knowledge），尚未有工作以受控消融证明
   分区条件化空间序列 gate 相对图像驱动空间 gate 的增益」**，且该表述须在完成系统检索后由研究者确认；
4. **MIGF 的 gate collapse 现象对本项目 H2 构成直接相关的风险**：M2 的 image-driven spatial gate 也
   可能出现权重近似静态或跨病例塌缩；本项目已有的 gate 诊断（§12.5：权重分布、熵、单序列塌缩）
   必须在 H2 判定中作为必报证据（已并入 §19 风险表）。

## 6. 建议的系统检索协议（尚未冻结；属待用户决定事项 D10）

| 项 | 建议内容（**未冻结**） |
|---|---|
| 数据库 | PubMed / Embase / IEEE Xplore / Scopus / arXiv（cs.CV）+ DBLP |
| 检索式 | (prostate OR prostatic) AND (zonal OR zone OR "peripheral zone" OR "transition zone") AND (fusion OR gating OR "attention" OR "modality weighting") AND (segmentation OR detection)；配合 PI-CAI / bpMRI / multi-parametric 同义词 |
| 时间范围与截止 | 建议 2018-01-01 至检索执行日；**截止日期须由研究者冻结**（GPT/LLM 类工作与预印本更新快） |
| 筛选 | 双人独立标题/摘要筛选 + 分歧解决；记录排除原因 |
| 提取 | 逐项机制对照表：输入先验类型（无/拼接/分区损失/门控条件/硬路由）、融合粒度（模态级/空间级）、骨干、数据、评测口径 |
| 产物 | `docs/literature/` 下的检索记录与对照表；用于支撑（或推翻）「未见同组合报告」的表述 |
| 纪律 | 检索截止后新增的相关工作只能以「后续工作」形式补记，不得据此回改已冻结的方法主张 |

## 7. 本次核查未做的事（诚实声明）

- 未做数据库级系统检索、未做 PRISMA 流程、未登记检索截止日期；
- 未核读任何论文的方法正文（仅标题/摘要/官方 README）；
- 未核读 CLAIM 2024 官方清单正文（页面 403、PDF 无法解析）；
- 未核读 Pooch et al. 原文；
- 未使用任何医学数据、未运行任何模型或评测。
