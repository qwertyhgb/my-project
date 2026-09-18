# 标号与步骤总览（G0 / SAP / P / M / H / D / B）

> **本文件回答两个问题**：① 项目里的这些编号（G0-P、G0-SAP、P2B、M4、H3、D7、B4……）各代表什么？
> ② 整个项目按什么步骤推进、谁卡住谁？
>
> **本文件不记录任何当前状态，也不定义协议条款。**
> - 当前状态 / 卡在哪 / 下一步 → `docs/STATUS.md`
> - 协议条款（通过条件、失败转向、统计规则） → `docs/research_plan.md`、`docs/protocols/`
> - 实际跑过什么 → `docs/experiment_log.md`
>
> 三者若与本文件冲突，以 `docs/research_plan.md` 与冻结协议为准。

## 0. 先看哪份文件（一句话路由）

| 我想知道…… | 去哪里看 |
|---|---|
| 现在到哪一步？卡在哪？下一步是什么？ | `docs/STATUS.md`（唯一状态来源） |
| 为什么要这样做？怎么算成功？失败怎么办？ | `docs/research_plan.md` |
| 具体要敲什么命令？成功判据？ | `docs/runbooks/` |
| G0-R / G0-E / G0-SAP 的详细协议与机器可读配置 | `docs/protocols/` + `configs/protocols/` |
| 协议改过什么？为什么改？ | `docs/protocol_changelog.md` |
| 实际运行过什么命令、结果如何？ | `docs/experiment_log.md` |
| 代码 / 配置 / 测试改了什么？ | `docs/Development_Log.md` |
| 代理能做什么、环境怎么配、进度条要求 | `AGENTS.md` |
| 每个编号什么意思、步骤顺序 | **本文件** |

## 1. 前缀速查

| 前缀 | 类别 | 例子 | 说明 |
|---|---|---|---|
| `G0-*` | **数据与协议层面的准入/冻结门** | `G0-P`、`G0-R`、`G0-E`、`G0-SAP` | 决定「数据能不能用、协议能不能开跑」；未通过时正式实验不得启动 |
| `G1`–`G5` | **科学阶段门** | `G2 = 自研基线可信` | 决定「研究结论能不能成立」，与模型/证据强度绑定 |
| `P0`–`P10` | **实施阶段**（工程与执行顺序） | `P0B`、`P2A`、`P2B` | 「先做什么、后做什么」；阶段与门是多对一关系 |
| `M0`–`M4` | **自研模型链** | `M4` | 消融链上的模型编号，见 §5 |
| `N0` | **官方风格基线** | `N0` | PI-CAI official-style nnU-Net v2 Focal+CE NoFFT；仅作 contextual reference |
| `H0` | **硬分区路由对照（模型编号）** | `H0` | ⚠ **命名易混**：`H0` 是一个**模型/对照**（P7、§10.1），**不是**假设 |
| `H1`–`H4` | **可证伪假设** | `H3` | `H4` 专门检验「M4 软融合 vs H0 硬路由」；见 §6 |
| `RQ1`–`RQ4` | **研究问题** | `RQ3` | 与 H1–H4 一一对应（RQ4 ↔ H4） |
| `SAP-A/B/C` | **G0-SAP 的三个时点** | `SAP-B` | 见 §2.4 |
| `D1`–`D10` | **待冻结决策**（必须由研究者决定） | `D4 = MCID` | 见 §7 |
| `B1`–`B9` | **正式实验前的 blocker** | `B1 = 无 Git`（**2026-09-18 已关闭**：Git 已启用） | 见 §8 |

> 记忆口诀：**G 是门（gate），P 是阶段（phase），M 是模型（model），H 通常是假设（hypothesis；
> 例外的 `H0` 是模型对照），D 是决定（decision），B 是阻塞（blocker）。**

## 2. G0 系列：数据与协议门（4 个）

| 门 | 它回答的问题 | 通过条件（摘要） | 通过后解锁 | 未通过 / 失败时 |
|---|---|---|---|---|
| **G0-P** | PI-CAI 数据能不能用？ | manifest、标签规则、几何与分区审计、患者级 train/validation、方向码与数据物化共 8 项全部完成（§6） | 可以进入后续所有数据侧工作（P2A 等） | 回退修正数据与审计，不得开训 |
| **G0-R** | 三序列（T2W/ADC/HBV）错位问题怎么处理？ | 不查看模型结果的对齐 QC + 冻结「仅重采样」或「重采样+固定刚体配准」，并冻结失败处理与预处理身份 | 解锁 P2B、正式 M0 与 N0 训练 | 修正预处理、重建派生数据；旧数据上的 run 只能算 feasibility |
| **G0-E** | 最终测试用什么数据？ | 候选外部集审计（许可/语义/序列映射/患者重叠/先验独立性）完成，并**预先**冻结路径 A、B 或 C | 解锁确认性最终测试（G5） | 更换候选；都不合格则在看结果前重划 PI-CAI internal test（**不得**用 validation 冒充 test） |
| **G0-SAP** | 评测与统计口径怎么定？ | 按 SAP-A/B/C 三阶段冻结方法、在 validation 上一次性填数值、封存后一次性评估（§2.4 与 §11.4） | 解锁「查看论文指标」；**最终 test 另须 G0-E 路径冻结**（评估本身属 G5） | 看结果前降级为探索性分析并改写成功标准 |

### 2.1 G0-P 的 8 项（§6）

方向码修正 → 单元测试与 manifest/geometry audit 重建 → ADC/HBV 物理重采样 → canonical 二值标签 →
`geometry_suspect` 人工 QC → 异常 WG 屏蔽 → 新 split 与泄漏复核 → 模态顺序/校验值冻结。

### 2.2 G0-R 的三种候选决策

| 候选 | 含义 |
|---|---|
| `resample-only` | 只做物理坐标重采样，不额外配准 |
| `resample+rigid` | 额外做固定刚体配准（必须新建派生数据并重新预处理） |
| `INSUFFICIENT_EVIDENCE` | 证据不足 → **必须停止**，不得强行开训 |

工具只输出**候选**，不写决策；决策由研究者冻结并记录哈希。
（自动路径 = `G0-R-AUTOMATED`，当前载体版本 **draft-0.3**：按 `(pair, case_id)` unit 校准、真实病例阈值按 pair 推导；draft-0.2 配置与失败运行产物保留为历史证据。人工双阅片路径保留为历史草案与后备。）

### 2.3 G0-E 的三条路径

| 路径 | 含义 |
|---|---|
| **A** | 存在合格的同任务外部测试集（可作确认性最终测试） |
| **B** | 外部数据语义/样本量不足 → **事先**降级为探索性验证（不得声称独立确认） |
| **C** | 无合格外部集 → 在任何正式模型结果前重划并冻结 **PI-CAI internal test** |

### 2.4 G0-SAP 的三个时点（SAP-A / SAP-B / SAP-C）

| 阶段 | 时点 | 允许看到 | 允许写入 | 禁止 |
|---|---|---|---|---|
| **SAP-A** | 任何模型预测之前 | 无模型预测 | **方法块**（阈值候选网格/选择指标/tie-break/后处理候选/`picai_eval` 参数/统计与 bootstrap 规则/FROC 操作点）+ **功效备忘录与 MCID** + 方法块哈希 | 不得用任何 validation/test 数值做选择 |
| **SAP-B** | 冻结 checkpoint 之后、最终 test 之前 | PI-CAI validation 预测 | **只写 `sap_b_results`**（选定阈值/后处理数值 + 来源哈希与差异审计） | 不得改动 SAP-A 方法字段与 `power_memo` |
| **SAP-C** | 评估阶段（属 G5） | 最终 test 预测 | 阶段与可见性字段 | 评估后任何修改都使结果失去确认性资格 |

**常见误读**：`protocol.status=FROZEN` **只表示 SAP-A 冻结**，不代表 SAP-B/SAP-C 完成。

## 3. G1–G5：科学阶段门

| 门 | 名称 | 通过条件（摘要） | 失败转向（摘要） |
|---|---|---|---|
| **G1** | 官方基线可复现 | G0-R/G0-E 已冻结；N0 能稳定训练、推理并输出预定指标（`checkpoint_final.pth` + 冻结 `picai_eval` 口径） | 排查数据/增强/推理链路，**不加入创新模块** |
| **G2** | 自研基线可信 | §9.4.2 的 G2 前置 1–8 项（含**峰值显存实测**）+ 冻结骨干/配置哈希与 skip 接口契约 | 排查 shortcut/shape/旁路/显存；必要时统一缩减 block 并升版 |
| **G3** | 动态融合成立 | 3 个冻结 seed 上 M2 优于 M1/M0，且无 gate 塌缩与容量混杂 | 退回 concat/等权分支，排查归一化、旁路与融合尺度 |
| **G4** | 解剖条件开发证据成立 | 3 个 seed 上 M4 优于 M2/M3；先验换源不过度敏感；容量受控 | 按失败模式收缩论文主张（M3-only / M2-only / 回到 concat） |
| **G5** | 独立确认完成 | G0-SAP 已冻结 + 冻结模型集（全部 seed）在最终 test 上**一次性**评估 | 区分任务/序列/先验差异；**不得**用 test 返调模型 |

**只有 G5** 才允许把 H3 表述为「得到独立确认」；validation（G3/G4）不是独立确认。

## 4. P0–P10：实施阶段

| 阶段 | 内容 | 对应门 |
|---|---|---|
| **P0A** | 审计修正（方向码、manifest/geometry audit） | G0-P |
| **P0B** | 数据物化（重采样、canonical 标签、异常 QC、train/val） | G0-P |
| **P0C** | 序列错位决策（对齐 QC → `frozen_decision`） | **G0-R** |
| **P0D** | 外部候选审计；决定是否需要 PI-CAI internal test | **G0-E** |
| **P0E** | 评测/统计配置与脚本、功效备忘录、冻结 | **G0-SAP** |
| **P1** | 官方基线 N0 正式训练与全量验证 | **G1** |
| **P2A** | v2.3 骨干实现（架构配置/哈希、参数与 MACs、skip 契约、合成回归） | G2（前置 1–7） |
| **P2A′** | PZ/TZ prior 物化（plan-space sidecar + canonical manifest + 深度校验链）与 M1–M4 融合代码 | 不属任何门；M3/M4 的**数据前置**（runbook：`p2a_zonal_prior_materialization.md`） |
| **P2B** | 新 M0 真实数据三步验证 + 峰值显存实测（`--overfit` 1 iteration → `gpu_memory_profile.json`） | **G2**（第 8 项） |
| **P3** | M1：浅层 stem + 统一 skip 聚合 + 等权融合（含容量对照决策） | G3 |
| **P4** | M2：图像驱动空间 gate | G3 |
| **P5** | M3 / M4：共享 zone encoder；conditional-affine gate | G4 |
| **P6** | 机制实验（序列组合、PZ/TZ 分层、序列移除、先验换源、gate 诊断） | G4 |
| **P7** | H0：同网络 deterministic hard-zone routing（可选对照） | G4 |
| **P8** | 正式复验（3 个冻结 seed 与配对统计） | G4 |
| **P9** | 独立测试（冻结模型集一次性评估） | **G5** |
| **P10** | 论文整理（图表、置信区间、失败病例、局限性与贡献边界） | — |

**关键硬依赖**：

| 动作 | 必须先满足 |
|---|---|
| P2A 新骨干代码与合成测试 | G0-P |
| P2B 新 M0 真实数据诊断（不作为论文结果） | P2A 1–7 完成 + G0-R 已冻结预处理身份 |
| 正式 N0 训练 | G0-R + G0-E 路径冻结 + 预算/checkpoint 规则冻结 |
| 正式 M0 训练 | G2 + G0-R + G0-E 路径冻结 + 增强/预算协议冻结 |
| M1–M4 实现 | G2 通过（协议条款）；研究者工程指令已**提前实现代码**（只写代码+合成测试，不读真实数据、不训练）——偏离已登记于 `docs/protocol_changelog.md`；**正式训练仍受 G2 约束** |
| M3/M4 读取真实 prior | 该例 sidecar 通过深度校验 + 已冻结逐例 manifest 记录（`zonal_prior_frozen_records`）；全量物化完成（`manifest.json` 覆盖 fold 0 的 train∪val） |
| 查看论文指标 / 进入最终 test | G0-SAP（SAP-A/B 完成）+ G0-E 路径冻结；最终评估属 G5 |

> P0C、P0D、P0E 与 P2A **不依赖模型结果**，可并行推进；但「可并行开发」不等于可以绕过任何门。

## 5. M0–M4 / N0 / H0：模型与对照

| 名称 | 结构（自研链） | 用来回答什么 |
|---|---|---|
| **M0** | 三序列普通 concat + plan-driven **Residual-Encoder** 3D U-Net | 自研骨干与管线是否可信（G2） |
| **M1** | 三序列**独立**浅层分支 + 固定等权融合 `W=(1/3,1/3,1/3)`（无 gate、无解剖输入） | 多分支与容量的影响（M1 vs M0） |
| **M2** | 在 M1 上加入**图像驱动**空间 gate（`G_img`，无解剖分支） | 空间动态融合是否优于简单融合（**H2**，注意容量混杂 §8.8） |
| **M3** | gate 同 M2；PZ/TZ 特征 `E_Z` 通过 `Q([F_fuse,E_Z])` 作为**普通输入**注入 | 解剖信息「作为输入」是否有用 |
| **M4** | **核心创新**：`L=(1+γ_Z)⊙L_img+β_Z`，PZ/TZ 条件仿射调制序列 gate | 解剖**条件化 gate** 是否优于普通输入（**H3**，M4 vs M3） |

| 对照 | 说明 |
|---|---|
| **N0** | PI-CAI official-style nnU-Net v2 Focal+CE NoFFT。骨干/epoch/增强与 M 系列不完全配平 → **只作 contextual reference**，不作因果消融基线 |
| **H0** | 同网络 + deterministic hard-zone routing（P7，可选）：当某分区占比 ≥ 0.75 时套用该分区规则，其余位置等权。用来对比「学习式软融合」与「硬分区规则」（对应假设 **H4**）。⚠ **H0 是模型编号，不是零假设** |

**主比较链（预先固定，不得事后更换）**：M4 vs M2（总增益）→ 仅当成立后，M4 vs M3（条件化 gate 是否优于普通输入）。

## 6. H1–H4 与 RQ1–RQ4

| 假设 | 内容 | 对应研究问题 | 主要证据 | 地位 |
|---|---|---|---|---|
| **H1**（探索性） | 三序列具有互补性，且互补程度随 PZ/TZ 分层而不同 | RQ1 | P6 序列组合 + PZ/TZ 分层分析 | 探索性 |
| **H2**（基础） | 空间动态融合优于简单融合（等权/concat） | RQ2 | M2 vs M1 / M0（需容量混杂控制） | 主消融链 |
| **H3**（核心） | 用 PZ/TZ 先验**直接条件化** gate 优于把解剖信息当普通输入 | RQ3 | M4 vs M2、M4 vs M3 | 主消融链，G5 独立确认 |
| **H4**（次要） | 学习到的连续序列权重优于预设 hard-zone routing | RQ4 | M4 vs **H0**（P7，可选） | 附属对照，**不作核心归因** |

> - G3 对应 **H2** 的开发证据，G4 对应 **H3** 的开发证据，G5 才是 **H3** 的独立确认；H1/H4 只作探索性/补充。
> - 再次提醒：**`H0` 是模型编号**（hard-zone routing 对照），**`H4` 才是假设**；两者字面相近但类别不同。

## 7. D1–D10：待冻结决策（必须由研究者决定，代理不得填数）

| 编号 | 决策内容 | 冻结载体（建议） |
|---|---|---|
| **D1** | G0-E 走路径 A / B / C | `configs/protocols/g0_e_independent_test.yaml` |
| **D2** | G0-R 证据是否充分；是否追加人工 landmark 复核或扩大抽样 | G0-R 协议载体 |
| **D3** | 共同阈值 vs 每模型独立阈值；后处理、bootstrap 次数与种子、`picai_eval` 参数 | `configs/protocols/g0_sap.yaml` |
| **D4** | **MCID 与主判定规则**（成功标准是否只要求 CI 不跨 0） | `g0_sap.yaml` 的 `power_memo` |
| **D5** | 实现字段是否重命名以对齐语义终点（`val_positive_casewise_dice_mean` → `positive_casewise_dice`） | `docs/protocols/G0_SAP.md` + 配置 |
| **D6** | M1/M2 容量混杂控制：同容量对照 or 降级 H2/G3 为探索性 | §8.8 + M1/M2 配置 |
| **D7** | N0 与自研预算差异的冻结方式；`augmentation.pending_parity` 6 项对齐或声明差异 | N0 配置 + §9.1 |
| **D8** | Yuan23 / HeviAI23 先验独立性审计结论与降级策略 | `g0_e_independent_test.yaml` |
| **D9** | 峰值显存超限时的统一缩减方案与新架构哈希 | `configs/architectures/m0_resenc_v23.yaml`（升版） |
| **D10** | 文献系统检索协议与检索截止日期 | `docs/literature/` |

集中清单见 `docs/research_plan.md` §20.2；实时状态见 `docs/STATUS.md`。

## 8. B1–B10：正式实验前的 blocker

| 编号 | 含义（状态与证据链接见 `docs/STATUS.md`） |
|---|---|
| **B1** | **已关闭（2026-09-18）**：主项目目录已启用 Git（用户决定；默认分支 `main`，基线 commit `23ca6cb`）；正式 run 可记录 `git rev-parse HEAD`；`third_party/nnUNet` 不 vendored（tag `v2.6.2` / commit `74ceb68`） |
| **B2** | G0-R 未冻结（真实对齐 QC 未运行 → 无法确认「仅重采样」是否足够） |
| **B3** | G0-E 未冻结（候选证据缺口、路径未定 → 没有独立最终 test） |
| **B4** | G0-SAP 的 SAP-A 未冻结（方法与候选集、MCID、功效备忘录待填；评测逻辑待实现） |
| **B5** | 峰值显存未实测（可能超出 GPU 上限）——**运行时遥测工具已就绪**（`--overfit` → `gpu_memory_profile.json`），实测待执行 |
| **B10** | plan-space PZ/TZ prior 物化 —— **已关闭（2026-09-17 完成全量 1500 例；编号保留供追溯）**：`data/processed/picai_zonal_yuan_v1/`（1500 NPZ + 1500 病例 JSON + canonical manifest）；剩余工程工作为读取端逐例数组级复校验与 loader-smoke（命令见 `p2a_zonal_prior_materialization.md`，状态见 `docs/STATUS.md`） |
| **B6** | 增强 `pending_parity` 6 项未对齐（M0–M4 与 N0 不完全配平） |
| **B7** | N0（1000 epoch/PlainConv）与 M0–M4（200 epoch/Residual-Encoder）预算与骨干不配平 |
| **B8** | 分区先验训练数据与目标病例的重叠未审计（影响 H3 归因） |
| **B9** | G0-E 配置哈希在清点运行后被修订 → 若以新配置冻结须重跑清点 |

## 9. 术语与缩写

| 术语 | 含义 |
|---|---|
| csPCa | clinically significant prostate cancer（临床显著前列腺癌；参考标准为组织学 ISUP ≥ 2） |
| PZ / TZ | peripheral zone（外周带）/ transition zone（移行带）；本项目的核心解剖先验 |
| WG | whole gland（全腺体）——**已退出所有主模型输入**，仅用于 QC、分层与单列消融 |
| T2W / ADC / HBV | T2 加权、表观扩散系数、高 b 值 DWI（官方定义 b ≥ 1000 s/mm²）；模态顺序固定为 T2W、ADC、HBV |
| DS | deep supervision（深度监督） |
| AP | average precision（病灶级平均精度，检测类主指标之一） |
| FROC | free-response ROC；主操作点冻结为 0.5 / 1.0 / 2.0 FP/exam |
| `min_overlap` | `picai_eval` 的命中判据，默认 IoU ≥ 0.1 |
| estimand | 目标估计量。最终 test 的语义终点名为 `positive_casewise_dice`（训练期实现字段为 `val_positive_casewise_dice_mean`） |
| MCID | minimal clinically/practically important difference（最小实际相关差异）——见 D4 |
| `conditional on the frozen trained seeds` | 患者级 bootstrap CI 的定位：只条件于已训练 seed，**不覆盖 seed 采样不确定性** |
| prior-friendly / oracle | 无法证明先验独立性时的降级标记：结果只能作探索性或上界分析，不得用于 H3 确认 |
| feasibility run | 未满足门条件而执行的 run；只能作工程可行性，不得并入正式比较 |
| legacy / feasibility | 旧 PlainConv M0 等历史实现的身份，不得与 v2.3 结果混表 |
| contextual reference | N0 的定位：成熟强参考，不作因果基线 |
| 冻结 / frozen | 值已被记录并绑定内容哈希；改动必须升版、说明原因并重记哈希 |
| amendment | 协议修订条目（含原因与「是否发生在查看结果之前」），记录于 `docs/protocol_changelog.md` |
| seg 哨兵（-1） | nnU-Net 预处理/填充的 ignore 标记，**不是第三个类别**（Dataset605 只定义 background=0 / lesion=1）；存储态 seg ∈ {-1,0,1}，读取适配层按官方 `RemoveLabelTansform(-1,0)` 语义映射为 0，返回严格 `uint8 ⊆ {0,1}`（常量 `SEG_SENTINEL_LABEL`/`SEG_STORAGE_LABELS`） |
| sidecar 成对提交 | `<case>.npz` 与 `<case>.json` 作为一对提交：staging → 备份 → 提交 → 落盘复核，任一步失败回滚（恢复旧对或清除新文件）；**不是**崩溃级事务原子，读取端 fail-closed |
| `zonal_prior_frozen_records` | manifest 深度校验成功后冻结的**逐例**哈希映射（`input_sha256`/`output_sha256`/`metadata_sha256`，深层不可变）；训练逐例读取时据此复核，防止预检后被替换 |
| `array_hash_checked` | 是否**逐个实际读取并校验了 NPZ 数组 SHA256**；为 `false` 时不得声称「全部 sidecar 已验证」 |
| `manifest_published` | canonical `manifest.json` 是否在本次运行中发布（仅全量、`n_failed==0`、发布前通过覆盖范围校验时才会发布） |
| `trainer_run` | `trainer.train()` 整体（内部先 train 再 validation，二者不可分割）——显存报告用该阶段名；**没有**独立的 train/validation 峰值 |
| `gpu_memory_profile.json` | `--overfit` 的运行时峰值显存报告：`process_memory.*` 为当前 PyTorch 进程（peak = 整个 `trainer_run` 的 high-water mark）、`gpu.*` 为整卡状态；成功 / `CUDA_OOM` / `ERROR` 都落盘，CPU 为 `NOT_APPLICABLE` |
| `peak_sample_source` | 峰值取自哪个采样点（`end` / `after_build` / `start`）——end 采样失败时的回退出处 |

## 10. 常见混淆点（FAQ）

1. **「代码写完、测试通过」≠ 门 PASS**：工程完成不使任何门变为 PASS。
2. **「G0-P 通过」≠ 数据可直接用于逐位置融合**：还须等 G0-R 冻结错位协议。
3. **validation 结果 ≠ 独立确认**：G3/G4 只在 validation 上；H3 的独立确认只能来自 G5。
4. **gate 权重 ≠ 因果贡献**：融合系数是结构量，不得解释为临床重要性或「贡献百分比」。
5. **「P2A 完成」≠ G2 通过**：G2 还要求 P2B 的真实数据三步验证与峰值显存实测。
6. **`protocol.status=FROZEN`（G0-SAP）≠ 三阶段完成**：它只表示 SAP-A 冻结。
7. **N0 不能当因果基线**：预算、骨干、增强都与 M 系列不配平。
8. **P0 与 G0 不是同一件事**：`P0C/P0D/P0E` 是**阶段**（去做），`G0-R/G0-E/G0-SAP` 是**门**（判定通过）。
9. **「M1–M4 代码已实现」≠ 可以训练**：实现按研究者工程指令早于 G2（偏离已登记），但 M0–M4 正式训练
   仍须 G2；M3/M4 还额外需要 prior 物化完成（B10，**2026-09-17 已完成**）与逐例冻结记录。
10. **显存报告里的 `peak_*` ≠ 整卡占用**：它是**当前 PyTorch 进程**的 high-water mark；`gpu.free_bytes_*`
    才反映整张卡（含外部进程占用）。报告里 `peak_scope`/`peak_is_process_peak_not_nvidia_smi` 与
    `peak_sample_source` 是判断语义的三个字段。
11. **「prior 文件存在」≠「prior 可信」**：读取端会逐例校验 source/configuration/plans_sha256/
    input+output+metadata 哈希/数组 SHA256/数值契约（finite、[0,1]、PZ+TZ≤1），任一不符即拒绝；
    哈希只能证明「与记录一致」，语义合法性由上述校验保证。

## 11. 一页流程（从数据到论文）

```text
[P0A 审计修正] → [P0B 数据物化] → ★G0-P（数据可用）
                                        │
        ┌───────────────────────────────┼───────────────────────────────┐
        ▼                               ▼                               ▼
[P0C 序列错位 QC]              [P0D 外部候选审计]              [P0E 评测/统计冻结]
 ★G0-R（错位协议冻结）          ★G0-E（路径 A/B/C 冻结）        ★G0-SAP（SAP-A 冻结）
        │                               │                               │
        ├──────────────► [P1 官方基线 N0] ★G1（可复现）                 │
        │                                                               │
        └──────────────► [P2A 骨干实现] → [P2B 真实数据验证] ★G2（自检基线可信）
                                                │
                            ┌───────────────────┴───────────────────┐
                            ▼                                       ▼
                    [P3 → M1] → [P4 → M2]                    （阶段门串行）
                            ★G3（动态融合成立）
                            │
                    [P5 → M3/M4] → [P6 机制实验] → [P7 H0 可选] → [P8 3 seed 复验]
                            ★G4（解剖条件开发证据成立）
                            │
                    SAP-B（validation 一次性填数）→ SAP-C
                            │
                    [P9 独立测试一次性评估] ★G5（独立确认）
                            │
                    [P10 论文整理]
```

> 说明：图中「★」= 阶段门；门未通过时**不得**进入其解锁的下一步。各门的**当前状态**见 `docs/STATUS.md`；
> 每步的可复制命令见 `docs/runbooks/`。
