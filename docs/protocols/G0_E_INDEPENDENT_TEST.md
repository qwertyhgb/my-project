# G0-E：独立测试路径冻结（外部候选审计与决策树）

| 字段 | 值 |
|---|---|
| 协议 ID | `G0-E` |
| 版本 | `draft-0.1` |
| **协议状态** | **DRAFT（未冻结；不得据此宣称任何外部结论）**；门的执行状态见 `docs/STATUS.md` |
| 创建日期 | 2026-09-15 |
| 审核人 | _待填写_ |
| 冻结日期 | _待填写_ |
| 机器可读配置 | `configs/protocols/g0_e_independent_test.yaml` |
| 关联章节 | `docs/research_plan.md` §5.2、§5.3、§5.4、§12.4、§14 G0-E |

> **本协议只做数据/语义/先验审计，不查看任何模型在外部数据上的结果。**
> 审计完成前，外部候选不得被称为 test，也不得用其任何指标做方法选择。

## 1. 目的

在**任何正式 N0/M0–M4 run 之前**，冻结一条唯一、可执行、可追溯的确认性评价路径，答案必须是下列之一：

- **路径 A**：同任务的合格 external test（通过 §3–§7 全部审计）；
- **路径 B**：外部数据语义/样本量不足 → 事先降级为探索性外部验证（只报适用的限定指标）；
- **路径 C**：无合格外部集 → 从 PI-CAI 重划并冻结 internal test（当前 1277/223 split 作废但保留审计）。

路径在冻结后、且在查看任何正式模型结果后**不得更改**；否则该数据集失去测试资格。

## 2. 候选身份登记（每候选一份记录）

对每个候选（当前优先级：Prostate158 → 私有 PCA → 其它）填写统一登记表（字段见 YAML `candidates[]`）：

| 字段 | 说明 |
|---|---|
| `name` / `version` / `source` | 数据集名称、版本/日期、来源（机构/论文/DOI） |
| `license` / `usage_scope` | 许可与允许的使用范围（研究/再分发/商用）；确认证据（链接或文件副本） |
| `n_cases` / `n_lesions` | 病例数、病灶数（按 §5 定义统计） |
| `modalities` | 实际可用的模态清单（T2 / ADC / DWI(b值) / 其它） |
| `patient_overlap_with_picai` | 与 PI-CAI 的患者重叠排查结论（**必须为 0 才可作 test**；附排查方法） |
| `label_semantics` | 病灶/阴性/分级定义的逐条对照（§4） |
| `zonal_labels` | PZ/TZ 是否存在、编码、来源（§6） |
| `prior_independence` | 自动分区器训练数据独立性审计结论（§7） |
| `metrics_supported` | 该数据可支持的指标清单（§8），超出部分禁止报告 |
| `verdict` | `eligible_external_test` / `exploratory_only` / `rejected`（附理由） |

## 3. 患者重叠排查

- 使用 patient-level 标识（可用哈希化的姓名/生日/检查号等可溯源字段）与 PI-CAI 全量 1500 study 比对；
- 附排查脚本与输出（重叠数为 0 才通过）；
- 已知禁止项：**ProstateX 与 PI-CAI 已包含部分病例映射，禁止作为独立外部集**（research_plan §4.5）；
- 私有数据在机构层面无法确认患者独立性时，只能走路径 B。

## 4. 任务与标签语义审计

### 4.1 阳性 / 阴性 / 病灶定义

| 问题 | 记录内容 |
|---|---|
| 病灶标注是谁、依据什么标准（PI-RADS / 病理 / 混合）？ | 逐条引用原文或提供书面确认 |
| “阳性”定义是否为 `ISUP ≥ 2` csPCa？ | 是/否 + 实际定义 |
| 阴性病例是否存在、如何定义（PI-RADS ≤ 2 / 无随访 / 无活检）？ | 是/否 + 参考标准 |
| 病灶是否经过病理证实、是否有 index lesion 规则？ | 逐条记录 |
| 多阅片者/多标签情况 | 各来源标注清单（见 §4.3） |

**判定规则**：

- 若外部集的“肿瘤/病灶”定义与 PI-CAI `ISUP ≥ 2 csPCa` **不等价**，则该数据集只能作
  `external PCa lesion validation`（跨数据敏感性证据），不得替代同任务独立 test 上的 H3 确认；
- 若外部集**只有阳性病例**：只能评价适用的分割指标（如 Dice/lesion recall），
  不得报告完整 AUROC、FP/exam 或任何需要阴性对照的指标（research_plan §5.2 第 7 条）。

### 4.2 模态映射（T2 / ADC / DWI-HBV）

- 逐候选填写：`T2` → 哪一序列（是否 fs、轴位与否）；
- `ADC` → 是否为计算的 ADC（非 DWI 原始图）；
- `HBV` → 高 b 值 DWI 的 b 值区间与采集定义；**只有核实后才允许与 PI-CAI 的 HBV 对应**，
  未核实时标记 `mapping_uncertain` 并从主分析排除该序列；
- 记录几何审计结论（spacing / 层厚 / 方向码 / 与 T2 的网格关系），与 PI-CAI 的差异列出。

### 4.3 多阅片者 / 多标签

- 所有标注来源（reader1/reader2 / interrater / AI）分别登记；
- 主阅片者或聚合规则必须在**观察任何预测前**冻结，不得事后挑选更有利者；
- 来源之间的差异只作敏感性分析，不与主结果混合汇总。

## 5. 几何、强度与覆盖审计

- 几何：size / spacing / origin / direction 五元组逐例核对；与 PI-CAI 的典型几何对比；
- 强度：dtype、强度范围、异常值（如 12-bit 截断）审计；
- FOV/覆盖：病变与腺体是否完整覆盖；是否存在需要排除的病例（排除规则事先写下）；
- 输出 `geometry_intensity_audit.csv` 与汇总。

## 6. PZ/TZ 编码与先验公平性

- PZ/TZ 标签值→语义映射必须来自权威说明；无权威定义时标记
  `zonal_semantics_unresolved` 并不得用于 M4 的自动先验（只能人工真值 → §7 oracle 路径）；
- **先验公平性**：
  - 主要外部分析必须在 PI-CAI 与外部集上使用**同一套可执行的自动分区模型**或两域一致的冻结自动协议；
  - 不允许只给 M4 用人工真值、而让其它 anatomy baseline 使用更差先验；
  - 外部集只能用人工分区真值时，结果标记为 `oracle-prior external analysis`，不能作为主要部署证据；
- PZ/TZ 的几何编码（与影像同一网格/需要重采样）单独审计并记录。

## 7. 自动先验训练数据独立性（v2.2 硬性要求）

对每个候选分区器（Yuan23 / HeviAI23 / 其它）记录：

1. **训练数据来源**（论文/权重/代码/许可证的可溯源证据）；
2. **是否覆盖 PI-CAI 病例**（尤其 validation/test 病例）—— 覆盖 = 重叠；
3. **是否覆盖外部测试病例**—— 覆盖 = 重叠；
4. **判定**：
   - 无重叠 → 可作为**独立自动先验**（主分析可用）；
   - 有重叠 → 主分析必须改用 **out-of-fold 预测**或未接触目标病例的独立分区器；
   - 无法证明独立 → 相关结果只能标记 `oracle/upper-bound` 或 `prior-friendly analysis`，
     **不得作为确认性结果**（不能用于 H3 独立确认）；
5. **边界声明**：腐蚀/膨胀/patch-drop 等形态学扰动**只能评价先验鲁棒性**，
   **不得**被描述为消除或量化训练集重叠泄漏（与 research_plan §12.4 一致）。

## 8. 支持的指标与报告边界

- 仅报告该数据实际支持、且事先登记在 `metrics_supported` 的指标；
- 外部指标与 PI-CAI validation 指标**分表呈现，不混合汇总**；
- 缺少 ISUP、阴性病例或统一病灶定义时，仅报告对应可支持的子集指标；
- 预先声明：外部样本量若不足以支持确认性判定，须在**看结果前**降级为探索性
  （功效/精度备忘录见 `docs/protocols/G0_SAP.md`）。

## 9. 决策树（冻结前逐候选执行）

```
候选 C
 ├─ 许可/使用范围未确认 ────────────────────────────→ rejected
 ├─ 与 PI-CAI 患者重叠 ≠ 0 ────────────────────────→ rejected
 ├─ 模态映射 (T2/ADC/HBV) 有未核实项 ──────────────→ 主分析排除该序列；若 T2+至少一个扩散序列可用，继续
 ├─ 病灶语义与 ISUP≥2 csPCa 不等价 ────────────────→ exploratory_only（external PCa lesion validation）
 ├─ 只有阳性病例 ─────────────────────────────────→ exploratory_only（限定分割指标）
 ├─ PZ/TZ 语义未解 ───────────────────────────────→ 仅允许 oracle-prior 分析（不作主要部署证据）
 ├─ 分区器训练集与目标病例重叠/不可证明独立 ───────→ 该先验降级；若所有先验都不可独立 → exploratory_only
 └─ 以上全部通过（且样本量按功效备忘录足够） ──────→ eligible_external_test（路径 A）

若无 eligible_external_test：
 ├─ 至少一个 exploratory_only 候选 ─────────────→ 路径 B（事先降级，事前写入，不得事后改判）
 └─ 全部 rejected ───────────────────────────────→ 路径 C（任何正式模型结果产生前重划 PI-CAI internal test）
```

**路径 B/C 的额外要求**：

- 路径 B：降级决定必须写入本协议与 YAML 并记录哈希，且在任何正式结果产生前完成；
- 路径 C：当前 1277/223 split 作废但保留审计；重新划 patient-level、`center × csPCa` 分层的新
  train/validation/test（test 完全不参与开发）；已有基于旧 split 的 run 只作 feasibility。

## 10. 输出与可追溯字段（全部落盘）

| 字段 | 说明 |
|---|---|
| 输入 | 候选数据集路径（只读）与登记证据（许可、说明文件副本）的路径与 SHA256 |
| 输出 | `candidate_registry.csv`（§2 全字段）、`patient_overlap_report.json`、`label_semantics_audit.md`、`geometry_intensity_audit.csv`、`zonal_mapping_audit.md`、`prior_independence_audit.md`、`decision.md` |
| 版本 | 本协议版本 + 审计脚本版本 |
| 状态 | `DRAFT` → `FROZEN`（仅当 §9 决策写入且记录哈希后才可 FROZEN） |
| 审核人 / 日期 | 审计执行人、复核人、执行与冻结日期 |
| 内容哈希 | 本协议、YAML、全部输出文件的 SHA256 |
| 决策 | `path: A \| B \| C` + `chosen_dataset`（路径 A/B 时）或 `internal_test_redesign_version`（路径 C 时） |

## 11. 工具实现状态与待办

### 11.1 工具能力与状态口径

> **实现/执行状态**（是否已实现、是否已运行、缺多少证据、冻结进度）唯一维护在 `docs/STATUS.md` 与
> `configs/protocols/g0_e_independent_test.yaml` 的状态字段；本节只描述**能力要求与状态语义**。

| 工具 | 路径 | 必须提供的能力 |
|---|---|---|
| 候选清点与证据清单 | `scripts/audit/audit_g0e_candidates.py` | 文件清单级清点（目录名 / 文件名 / CSV 表头 / 行数）；证据状态汇总（`VERIFIED` / `UNKNOWN` / `CONFIGURED_EXCLUDED`）；输出 `candidate_registry.csv`、`evidence_status.json`、`inventory_summary.json`、`missing_evidence.md`、`audit_report.md`、`run_metadata.json`；`--dry-run` / `--no-progress` |
| 清点与状态逻辑 | `src/zonal_reliability_fusion/protocols/g0_e.py` | `AUTO_ONLY` 事实与人工证据分离；不推断独立性 / 可用性 |

**状态语义（必须区分，2026-09-17 更正）**：

- `inventory_status`：**文件清单级清点**（目录名/文件名/CSV 表头/行数）——只产生 `AUTO_ONLY` 事实；
- `evidence_audit_status`：**证据审计**（许可、标签语义、患者重叠、模态映射、先验独立性）——缺证据一律 `UNKNOWN`；
- `verdict` / `frozen_decision`：只能由研究者按 §9 决策树填写，工具不代填；
- **清点执行成功或工具运行成功，都不等于 G0-E 通过**；G0-E 仍为 Pending，直到路径 A/B/C 冻结并记录哈希。

**哈希绑定规则**：清点工具会把运行时的 `config_sha256` 写入 `run_metadata.json`。任何在运行之后对 YAML 的修改
（即使只是状态字段）都会使该绑定不再代表现行配置；**若要以修订后的配置冻结 G0-E，必须重新运行清点并绑定新哈希**
（既有产物保留，不追溯改写；当前哈希记录见 `docs/STATUS.md` B9 与 `docs/protocol_changelog.md`）。

**边界**：`AUTO_ONLY` 值（目录数、文件计数）仅供参考，**不作为判定依据**；`patient_overlap_with_picai`、
`prior_independence`、`license`、`label_semantics` 等缺证据时一律 `UNKNOWN`；`verdict` 与
`frozen_decision` 只能由研究者按 §9 决策树填写。工具运行成功**不代表 G0-E 通过**。

### 11.2 待办

- [ ] Prostate158：许可、模态映射、病灶语义、PZ/TZ 语义、样本量逐项审计
- [ ] 私有 PCA：ROI 语义与病理标准书面确认（未确认前不得作为外部 test）
- [ ] 患者重叠排查（全部候选）
- [ ] 分区器（Yuan23/HeviAI23）训练数据独立性审计（§7）
- [ ] 功效备忘录输入（阳性数/病灶数与 §5.2 指标范围）
- [ ] 依据 §9 冻结路径 A/B/C → 状态置 `FROZEN`

> 协议状态：**DRAFT（未冻结）**。缺少的任何审计项不得用假设填充；无法确认时必须如实降级
> （`exploratory_only` 或路径 C），并在 `decision.md` 写明理由。执行与冻结进度见 `docs/STATUS.md`。
