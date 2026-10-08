# Research Plan

**基于预测解剖先验的病灶感知粗到细前列腺癌分割**
*Anatomy-Guided Lesion-Aware Coarse-to-Fine Prostate Cancer Segmentation*

> 一句话定义（README、Research Plan、代码注释保持一致）：
> **本项目研究预测前列腺解剖先验与病灶感知粗到细学习能否提高小型和困难前列腺癌病灶的检出与
> 完整分割，同时控制假阳性。**
> *This project investigates whether predicted prostate anatomy and lesion-aware
> coarse-to-fine learning can improve the detection and complete segmentation of small and
> difficult prostate cancer lesions while controlling false positives.*

**文档定位**：本文件只讨论**研究本身**——研究问题、科学假设、方法机制、研究边界，以及预先定义
的论文报告指标。它**不**记录具体实验流程、数据划分、训练参数、运行命令或已发生的结果：

- 具体实验矩阵、预算、stop rules 的运行判据 → `docs/Experiment_Plan.md`
- 指标定义与统计口径 → `docs/Evaluation_Protocol.md`
- 方法机制与实现边界的细节 → `docs/Method.md`
- 已发生的训练事实 → `docs/Training_Log.md`
- 已有证据支持的结论 → `docs/Findings.md`
- 工程变更 → `docs/Development_Log.md`
- 旧研究路线 → `docs/archive/`

---

## 1. Research Problem

前列腺癌病灶（csPCa）在 bpMRI 上是**稀疏、小体积、对比度低**的目标。以本项目使用的
Dataset605/606（PI-CAI 派生，1277 训练例 / 223 验证例，fold 0，阳性约 28%）为例，已完成训练
的五个模型出现三个**反复出现、且彼此不可替代**的失败模式：

1. **完全漏检 lesion**——阳性病例中约有 21–34 例的 GT 病灶**完全没有被分割出来**（Dice = 0）；
2. **已检出病灶仅分割出核心区域，病灶覆盖不足**——在"有重叠"的病例上，阳性体素召回中位数仅
   0.40–0.46；
3. **提高 sensitivity 后引入过多 false positives**——把每 batch 的一个 patch 槽位固定为阳性
   病灶中心裁剪后，完全漏检从 34 例降到 21 例，代价是阴性病例出现假阳的例数从 11 升到 30。

三者构成一个**结构性张力**：任何只沿一个方向施力的改动（更激进的采样、更强的损失重加权、
更宽的阈值）都会把另外两个推到更差。因此本项目的研究问题不是"如何再提高一点 Dice"，而是：

> **如何利用预测得到的解剖先验（WG / PZ / TZ）与 lesion-aware learning，减少小病灶与困难病灶
> 的完全漏检，提高已检出病灶的完整覆盖，同时避免假阳性显著增加？**

## 2. Clinical Motivation

- csPCa 的临床意义与体积、分级相关；**漏检**比"边界略不精确"的后果更严重——一个完全漏掉的
  病灶不会进入任何后续判断；
- 泌尿病理科医师做靶向活检时依据的是"候选病灶的位置 + 范围"，而不是整幅图像的 Dice；
  因此**病灶级检出**与**病例级完整覆盖**比体素平均指标更贴近使用场景；
- 前列腺的解剖结构（全腺体 WG、外周带 PZ、移行带 TZ）是**可预测的**、稳定的先验：病灶按定义
  位于腺体内，且不同分区的病灶在 MRI 上的表现不同。这使"用解剖结构约束病灶搜索与细化"成为
  一个临床上有意义、技术上可实现的切入点。
- 本项目**不**声称临床可用性。所有指标都是分割失败分析（segmentation failure analysis），
  不是 PI-CAI challenge 的 detection metric（不计算 AUROC / average precision / FROC /
  detection score）。

## 3. Evidence From Preliminary Experiments

以下是**已发生**的实验事实（细节见 `docs/Training_Log.md` 与 `docs/archive/`），它们共同构成
新主线的动机。它们全部是**单次运行**（nnU-Net v2.6.2 不设随机种子），因此只作为方向性证据。

### 3.1 输入级 modality 重加权：未观察到稳定的分割增益

旧的 `image_gate` / `anatomy_gate` 线把问题表述为"T2W / ADC / HBV 应以什么权重融合"。在
公平匹配（同一损失、同一阳性采样、同一增强、同一 optimizer 与调度）的两次配对比较中：

- `positive_sampling` → `image_gate_positive_sampling`：病例级配对 Dice 变化均值 **−0.0077**，
  95% CI **[−0.0540, +0.0403]**（跨 0）；
- `image_gate_positive_sampling` → `anatomy_gate_positive_sampling`：均值 **−0.0053**，
  95% CI **[−0.0525, +0.0381]**（跨 0）。

**表述纪律**：这**不能**被写成"gate 无效""证明 gate 没有作用""gate 降低性能"或"PZ/TZ 无用"。
CI 跨 0 既不支持提升也不支持下降，更不证明等效。可以、也应当写的是：

> Direct input-level modality reweighting did not provide a consistent segmentation
> improvement in the tested runs, motivating a shift toward lesion-aware anatomical modeling.

### 3.2 病灶暴露是最强的单变量杠杆，但有代价

在同一框架内只改变训练采样（每 batch 固定一个阳性病灶中心 patch）：

- 完全漏检病例：**34 → 21**；有重叠的病例：**29 → 42**；阳性体素召回中位数：**0 → 0.2372**；
- 病例级配对 Dice 变化均值 **+0.1093**，95% CI **[+0.0586, +0.1624]**（**不**跨 0）；
- 代价：阴性病例出现假阳的例数 **11 → 30**（+19）。

### 3.3 观察：早收敛 ≠ 更高上限

五个已完成 run 的最佳 EMA patch pseudo Dice 只有 0.58–0.62，而全量验证 Dice 为 0.17–0.21。
这说明 patch 级优化过程指标存在代理偏差，不能用来替代整例评估（口径见
`docs/Evaluation_Protocol.md`）。

### 3.4 由此得到的研究缺口

1. modality 加权不是瓶颈——瓶颈在**病灶是否被发现、是否被完整覆盖**；
2. 增加病灶暴露能改善检出，但**没有针对代价的控制手段**；
3. 已有的解剖信息（PZ/TZ）被用在**错误的位置**（调制 modality 权重），而不是用在
   **定位与细化**上；
4. 缺少以 lesion failure mode 为中心的评价体系（过去主要看 `foreground_mean Dice`）。

## 4. Main Hypothesis

> **显式的病灶定位（coarse lesionness）加上预测解剖先验提供的结构化上下文，可以在不显著增加
> 假阳负担的前提下，减少小病灶与困难病灶的完全漏检并改善已检出病灶的完整覆盖。**

可证伪的含义（逐条对应 stop rules，见 `docs/Experiment_Plan.md`）：

| 编号 | 假设 | 若被证伪的表现 |
|---|---|---|
| **H1** | 缩小无关背景的搜索空间（Anatomy-Guided ROI）本身即改善 lesion learning | macro Dice、lesion sensitivity、small-lesion sensitivity 三项均无改善 |
| **H2** | 显式 lesion localization（coarse lesionness + soft refinement）减少完全漏检 | completely missed lesions 未减少、lesion-level sensitivity 未提高 |
| **H3** | soft PZ/TZ 解剖上下文在已有 lesion-aware coarse-to-fine 之上进一步改善覆盖或控制 FP | 相对 C 的配对变化无稳定收益 |
| **H4** | 解剖约束的困难负样本挖掘可降低 FP 负担而不损害 sensitivity | FP burden 未下降或 sensitivity 下降 |

H1–H4 是**待验证假设**。本项目**不**提前声称任何一条已取得提升。

## 5. Stage 1: Predicted Anatomy Priors

**输入**：T2W（单通道）。
**输出**：`P(WG)`、`P(PZ)`、`P(TZ)` 三个 **soft probability map**（不是只有硬二值 mask）。

Stage 1 训练一个稳定、独立、**冻结**的 anatomy model。它的定位是
`anatomical prior generator`：

- 它**不是**论文的主要创新点，论文中不把"联合 WG/PZ/TZ 分割网络"包装成方法贡献；
- 它的误差是**下游真实输入分布的一部分**，不做任何人工修正。

### 必须成立的原则（fail-closed）

1. anatomy model **不读取 lesion GT**；
2. validation / test 病例的 anatomy prior **必须**来自该病例自身 MRI 的预测；
3. **禁止**把 GT WG/PZ/TZ 直接作为 lesion model 的推理输入（GT 只能用于显式标记的
   `ORACLE_GT` 上界分析，且不得作为正式最终性能）；
4. 训练 lesion model 时必须防止任何跨病例、跨 split 泄漏（由
   `scripts/data/check_split_integrity.py` 自动检查，遇泄漏 fail closed）；
5. 若改用 out-of-fold anatomy prediction，必须显式实现并记录；当前阶段使用**冻结 anatomy
   model 对全部病例预测**，训练/评估隔离逻辑同样清晰（先验由固定模型产生，不与 lesion
   训练过程耦合）；
6. 为什么必须 soft：硬 mask 的边界错误会**直接删除**跨越 WG 边界的病灶；soft probability
   让"解剖不确定"这一信息保留下来，可以被下游走 residual path 而不是被裁掉。

数据集的输入/输出/来源清单见 `docs/Method.md`；`predicted prior` 与 `GT` 的区分见
`docs/Evaluation_Protocol.md` 的「先验来源」小节。

## 6. Stage 2: Lesion-Aware Coarse-to-Fine Segmentation

**输入**：T2W + ADC + HBV + predicted `P(WG)` / `P(PZ)` / `P(TZ)`。
**核心目标**：small-lesion detection + complete lesion segmentation + false-positive control。

整体逻辑：

```text
T2W
 │
 ▼
Anatomy Model  ──▶  P(WG) / P(PZ) / P(TZ)          （Stage 1，冻结）
                        │
T2W / ADC / HBV         │
      +                 │
predicted anatomy       ▼
      │        Anatomy-Aware ROI                  （条件 B）
      ▼                 │
      └────────────────▶│
                        ▼
              Coarse Lesion Localization            （条件 C）
                        │
                        ▼
              Lesion-Aware Feature Refinement       （条件 C，soft guidance）
                        │
                        ▼
              Fine Lesion Segmentation
                        │
                        ▼
              Zone-Aware Refinement（可选）          （条件 D）
```

设计顺序（**任何时候都不允许颠倒**）：

```text
strong baseline → anatomy ROI → lesion localization → coarse-to-fine refinement
→ zone anatomy context → hard-negative control
```

而不是：`Gate → bigger Gate → attention Gate → reference Gate → another Gate`。

**单变量纪律**：论文主线的每个条件只允许相对其**上一条**改变一件事。任何新模块都只在 strong
baseline 上逐步增加，不同时修改 loss / sampling / network / ROI / anatomy 后再比较。

模块机制细节见 `docs/Method.md`。

## 7. Anatomy-Guided ROI

用 **predicted WG** 定位前列腺，但**禁止 hard mask**。

不做 `image = image * wg_mask`：WG 预测的边界错误会**直接删除**病灶，这是不可恢复的失败。采用

```text
predicted WG → bounding box → physical margin expansion → prostate-centred ROI
```

四条设计约束：

1. **margin 按物理距离 mm 定义**，不按固定 voxel 数。Dataset605/606 的 `3d_fullres` spacing 是
   `[3.0, 0.5, 0.5] mm`，同一 mm 边界在 z 轴与面内对应的 voxel 数相差 6 倍；每轴 voxel 数
   **向上取整**，保证实际物理边界不小于承诺值。
2. **保留 gland 周围安全边界与可能跨越 gland / zone 边缘的病灶**；
3. **不重采样**：crop 后通过 **padding** 满足网络输入尺寸，spacing / origin 保持不变，预测可以
   **无损**恢复到原空间；crop transform 完整保存为可审计的 provenance；
4. **空 WG 预测不得导致病灶被裁掉**：回退全视野并**显式记录**原因，绝不静默。

ROI 作为**独立消融**（Baseline vs Baseline + Anatomy ROI），先回答一个朴素问题：

> 单纯减少无关背景，是否已经能改善 lesion learning？

## 8. Lesionness Localization

这是新研究最重要的候选创新之一。它把模型的学习目标显式拆成两件事：

- **Where is the lesion?** —— coarse lesionness 头，追求高 sensitivity，目标不是漂亮边界，而是
  尽量不出现 `completely missed lesions`；
- **What is the exact lesion boundary?** —— 最终 segmentation 头。

在候选目标形式（Gaussian center heatmap / dilated lesion mask / coarse lesion mask /
distance-transform target）中选择**唯一一种**：**物理半径膨胀的 coarse lesion mask**。理由：

1. 与现有数据标签形式一致（仍是二值掩膜），可直接沿用 nnU-Net 原生损失与 deep supervision，
   不引入新的损失族；
2. 与 3D spacing 自洽（半径以 mm 定义，按各尺度真实 spacing 折算，使用椭球结构元素）；
3. 最容易解释（"该体素是否在某个真实病灶 r mm 邻域内"）；
4. 避免 Gaussian heatmap 的单体素峰与 sigma 调参带来的不可归因超参数。

硬约束：

- lesionness supervision **只来自 lesion GT**；validation / test **不使用**任何 GT 推导的目标；
- coarse 头与 final segmentation 头**完全联合训练**（同一次 backward），不是两阶段；
- 推理时使用 coarse 预测引导 refinement，且**只作为 soft guidance**。

## 9. Zone-Aware Refinement

### PZ/TZ 的作用被重新定义

以后**禁止**把 PZ/TZ 的主要作用写成"根据区域调整 T2W / ADC / HBV 的 modality weight"。

新的解释是：

> PZ/TZ 为 lesion localization 和 lesion refinement 提供 **anatomical context**。

也就是回答：

```text
Where is this candidate located?
What tissue context surrounds it?
Is this candidate in PZ, TZ, boundary or uncertain anatomy?
```

采用 **soft** `P(PZ)` / `P(TZ)`，不是硬 one-hot；`P(U) = 1 - max(P(PZ), P(TZ))` 承担分区边界与
解剖不确定区域的权重。

### 两个候选设计，默认先用最简单的

1. **直接 concatenation soft anatomy maps**（默认）——把 `P(PZ)/P(TZ)/P(U)` 拼到 refinement 的
   context 里；
2. **Zone-Aware Refinement（两个 expert + soft 融合）**：

   ```text
   F_out = P(PZ) * F_PZ + P(TZ) * F_TZ + P(U) * F_shared
   ```

   **只有在第 1 种被证明不足、且参数增量可控时才启用第 2 种。** 参数量差异必须显式报告
   （`zone_mode_parameter_delta`），不得笼统称"两者结构相同"。

如果 D 相对 C 无稳定收益，最终模型**停在 C**，并立即停止 anatomy 架构扩张。

## 10. Hard Negative Mining

针对第三个失败模式的训练策略增强：

> 如何让模型更多看到"最像癌但实际上不是癌"的困难背景？

**Anatomy-Constrained Hard Negative Mining** 的候选定义（预先冻结）：

```text
prediction confidence high  AND  GT background  AND  inside / near prostate WG
```

（等价形式：anatomical ROI 内的高损失负区域。）

分轮流程：Round 1 训练 strong baseline / coarse-to-fine → Round 2 对**训练 split**推理并挖掘
高置信假阳 → Round 3 训练时提高这些位置的采样概率。

**安全红线（全部 fail-closed）**：

- 挖掘**只在 training split 内**执行；
- validation / test **绝对不能**参与挖掘；
- **不得**从 validation error 反向挑训练样本；
- 困难负样本采样必须能**开关**（同一 Trainer 类的一个参数），方便作为独立 ablation；
- 最初只实现**静态离线集合**，不做在线复杂 memory bank。

定位：若效果稳定则进入最终方法；若效果有限，则作为 ablation / supplementary experiment。
**不强行把所有模块都塞进最终模型。**

## 11. Experimental Design

设计原则（具体矩阵、预算与命令见 `docs/Experiment_Plan.md`）：

1. 先把 **strong baseline** 确定下来：在同一次比较中只让**损失**不同
   （PI-CAI Focal+CE vs nnU-Net 原生 Dice+CE），其余全部继承同一套 nnU-Net 原生机制；胜者成为
   后续所有方法的唯一参照。
2. 之后形成最小主线 **A → B → C → D**，每个条件只相对上一条改变一件事：

   | 条件 | 内容 | 回答的问题 |
   |---|---|---|
   | **A** | strong baseline（T2W+ADC+HBV、nnU-Net、best loss、positive sampling） | 一个训练充分、病灶暴露合理的普通 nnU-Net 能达到什么水平？ |
   | **B** | A + Anatomy-Guided ROI | 缩小无关背景搜索空间是否改善 lesion learning？ |
   | **C** | B + lesion-aware coarse-to-fine | 显式 lesion localization 是否减少完全漏检，特别是小病灶？ |
   | **D** | C + zone-aware refinement | 已有 lesion-aware 学习后，PZ/TZ 是否进一步改善覆盖或控制 FP？ |

3. **E**（hard negative mining）是 training strategy enhancement，可进最终方法，也可只作
   ablation。
4. 硬负样本、难度采样等 strategy 层面的改动**必须**与架构层面的改动分开评估，否则无法归因。

## 12. Evaluation Protocol

**关键主张：以 lesion failure mode 为中心建立统一评价体系，不再只看 nnU-Net 的
`foreground_mean Dice`。**

分级（完整定义、空值语义、统计口径见 `docs/Evaluation_Protocol.md`）：

- **Primary endpoint**：**Positive-case macro Dice**（所有 GT positive cases 纳入；完全漏检病例
  Dice = 0，不得排除）。
- **Key secondary endpoints**（必须同时报告）：
  1. Lesion-level sensitivity（按 3D connected component 计算）；
  2. Small-lesion sensitivity（按 reference lesion 的**真实物理体积**分层；区间
     `< 500 / 500–1000 / > 1000 mm³` 只能称为 **exploratory size strata**，不是临床大小分类）；
  3. Matched-lesion Dice（只对成功匹配的 pair 计算，**必须**与 lesion sensitivity 同时报告，
     因为单独报告会隐藏完全漏检）；
  4. Completely missed lesion / case rate；
  5. Positive voxel recall（lesion 找到之后到底覆盖了多少 GT lesion）；
  6. Positive voxel precision（防止粗暴扩大分割）；
  7. False-positive burden（`negative cases with FP`、`FP components / case`、`FP voxel volume / case`，
     必要时按 predicted anatomy 分解为 inside WG / outside WG / PZ / TZ / uncertain）。
- **Exploratory**：病灶大小分层敏感度、按解剖区域的 FP 分解、表面距离指标、bootstrap CI。

**lesion-level analysis 不是附加实验，它就是论文的核心评价体系。**

## 13. Ablation Study

消融严格遵循"一次一个变量"的阶梯，而不是并联的所有组合：

- **A1 vs A2**：损失（Focal+CE vs 原生 Dice+CE），其余全同；
- **A vs B**：Anatomy-Guided ROI 的增量；
- **B vs C**：lesionness 头 + soft refinement 的增量（core method comparison）；
- **C vs D**：zone-aware refinement 的增量；并对照 `concat` 与 `zone_experts` 两种候选；
- **best(C, D) vs E**：困难负样本挖掘的增量（且必须报告关闭状态下的同规则复现）；
- **B 内部的可选消融**（仅在需要解释 ROI 的机制时进行，不作为主线）：ROI margin 的 mm 取值、
  ROI 内采样槽位比例的敏感性 —— 这些都必须作为**单独**报告，不得与其它改动混合；
- **C 内部的可选消融**：lesionness 半径（预先冻结值的敏感性）、是否启用 lesionness guidance；
- **ORACLE 上界**：用 `ORACLE_GT` 解剖标签替代 predicted prior 的 ROI/上下文，用于说明
  "anatomy 预测误差占了多少"，**必须显式标记，不得作为正式最终性能**。

## 14. Reproducibility

1. **显式 seed 支持**：控制项目自身能控制的部分（Python random、NumPy、PyTorch、采样、模型
   初始化）。
2. **诚实边界**：
   > explicit seed improves repeatability but does not guarantee bitwise determinism.
   nnU-Net v2.6.2 的增强由多进程 `NonDetMultiThreadedAugmenter` 驱动，其完成顺序非确定；
   因此本项目**不**声称 bitwise 可复现。
3. **最终候选必须做 3 个独立 seed 的完整训练**，报告 each run、`mean ± std` 与病例级 paired
   analysis；任何候选模块在单次 exploratory run 中出现很小的 Dice 上升，**都不能**立即作为最终
   创新。
4. **短预算探索机制**：先做 100 / 150 epoch 的 exploratory protocol，只用于 sanity check、
   direction screening 与 obvious failure elimination；短预算不同模型必须 epoch 相同、
   iterations per epoch 相同、scheduler 总周期匹配、validation protocol 相同。短预算结果
   **不能**与历史 1000-epoch 模型直接声明性能优劣。
5. **每个实验必须能打印并落盘冻结配置**（dataset / fold / seed / trainer / network / loss /
   sampling / epochs / batch size / patch size / spacing / anatomy prior source / ROI setting /
   lesionness setting / hard-negative setting）。
6. **split 完整性自动检查**：leakage 一律 fail closed。

## 15. Stop Rules

防止再次无限扩张。判据的具体阈值与判定脚本见 `docs/Experiment_Plan.md`。

- **Rule 1**：若 Anatomy ROI 相对 strong baseline —— macro Dice、lesion sensitivity、
  small-lesion sensitivity **三项都没改善**，则**不要**通过继续增加 ROI attention 来"救"。
- **Rule 2**：若 lesionness / coarse-to-fine 相对 B —— completely missed lesions 没减少、
  lesion-level sensitivity 没提高，则说明"coarse localization 假设未获支持"。**不要立即加入更
  复杂的 Transformer / Mamba**。
- **Rule 3**：若 zone-aware refinement 相对 C 无稳定收益，**立即停止 anatomy 架构扩张**：不继续
  `AnatomyGate v2`、`CrossAttention`、`ZoneTransformer`、`ZoneMamba`。
- **Rule 4**：最终候选必须 3 seed + `mean ± std` + 病例级 paired analysis；单次 run 的小幅上升
  不构成证据。
- **Rule 5**（贯穿）任何模块必须能回答"是否减少漏检 / 是否改善覆盖 / 是否减少假阳"三者之一；
  回答不了就不进主模型。

## 16. External Validation

若使用 Prostate158，定位为 **external distribution-shift stress test**，而不是"在另一个数据集
上也很好"的附加亮点：

- 训练第三序列与外部数据的序列**可能并非完全同质**，必须在报告中说明；
- 若最终模型依赖 predicted anatomy，**必须用冻结的 anatomy model** 在外部数据上生成 anatomy
  prior；
- **禁止**：人工修改外部 prior、使用 GT zone、在看到 test 结果后重新调参数。

## 17. Expected Contributions

> **以下三条全部是待验证假设。本项目不提前写成已经取得提升。**

### Contribution 1 — Lesion-Aware Coarse-to-Fine Learning

显式把 `lesion localization` 与 `precise lesion segmentation` 联系起来，以减少 small /
difficult lesion 的完全漏检。验证方式：条件 C 相对 B 的 completely missed lesions 与
lesion-level sensitivity 的配对变化。

### Contribution 2 — Predicted Anatomy-Guided Lesion Modeling

使用**真实推理条件下预测得到的** WG / PZ / TZ 作为 lesion localization 与 refinement 的结构化
先验。重点**不是** modality weighting。验证方式：条件 B/D 相对其前序条件的配对变化，以及
`ORACLE_GT` 上界与 predicted prior 的差距。

### Contribution 3 — Lesion-Centric Evaluation

系统评价 lesion sensitivity、small-lesion sensitivity、matched-lesion Dice、missed lesions、
false-positive burden，而不仅仅报告单一 Dice。验证方式：本项目的评价体系本身（
`docs/Evaluation_Protocol.md` + `src/zonal_reliability_fusion/evaluation/`）。

## 18. Limitations

1. **全部 preliminary evidence 都是单次运行**，无 run-to-run 方差估计；因此它们只能作为方向性
   证据，不能作为效应量。
2. **解剖先验是算法伪监督**：WG 来自已物化掩膜、PZ/TZ 来自自动分区结果，不是手工标注；因此
   Stage-1 的"真值"本身有系统偏差，且 `11050_1001070` 因缺少可用 WG 而被**显式排除**。
3. **anatomy 预测误差会传播**：Stage-2 的输入包含 predicted prior，因此最终指标包含解剖预测
   的误差；`ORACLE_GT` 上界只能说明误差"有多大空间"，不能把它当作可达性能。
4. **第三方序列同质性未知**：Dataset605/606 前三个 MRI 通道经逐数组一致性审计一致，但
   PZ/TZ 不参与判等；Prostate158 的第三序列可能与训练数据不完全同质。
5. **评估不做后处理**：不进行最小团块过滤、最大团块、形态学开闭、填洞或阈值优化；这使指标
   反映的是原始网络输出，而不是经过临床后处理的性能。
6. **评价是 segmentation failure analysis**，不是 detection metric：不计算 AUROC /
   average precision / FROC / detection score，也不做阈值敏感性分析来模拟检出率曲线。
7. **大小分层是探索性的**：`< 500 / 500–1000 / > 1000 mm³` 是按物理体积的工程分层，**不是**
   临床风险类别，也不对应任何指南分级。
8. **单中心、单 fold 主实验**：fold 0；外部数据只作为分布偏移压力测试。
9. **标签噪声**：病灶 GT 为自动生成/派生标签（PI-CAI 派生），其自身误差不会被本项目的指标
   所隔离。
