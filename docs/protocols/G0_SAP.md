# G0-SAP：评测与统计分析计划冻结

| 字段 | 值 |
|---|---|
| 协议 ID | `G0-SAP` |
| 版本 | `draft-0.3`（v2.3.1 amendment：三阶段生命周期 + 预测可见性字段拆分；v0.3 起含 `sap_b_results` 结果块、SAP-A 方法块哈希与差异审计，2026-09-17） |
| **协议状态** | **DRAFT（SAP-A 未冻结）**；实现与执行进度见 `docs/STATUS.md` |
| 创建日期 | 2026-09-15 |
| 审核人 | _待填写_ |
| 冻结日期 | _待填写_（SAP-A 冻结时填写） |
| 机器可读配置 | `configs/protocols/g0_sap.yaml` |
| 关联章节 | `docs/research_plan.md` §3.1、§11.1、§11.4、§13、§14 G0-SAP |

> 本协议冻结**评测口径与统计规则**。冻结后任何修改必须升版本并说明原因；
> 绝不使用观察到的模型效果反向制定阈值、成功标准或分析选择。

## 0.1 生命周期：SAP-A / SAP-B / SAP-C（v2.3.1 amendment）

本协议把「冻结」拆为三个时点，解决两类冲突：① 阈值必须在 PI-CAI validation 上选择（因此必须先有冻结
checkpoint），却又不得在冻结前"运行评测"；② 评测逻辑必须**先实现**才能定义候选网格与选择算法。
判定原则是**「方法与候选集」在看结果前冻结，「数值」在 checkpoint 冻结后一次性填充**。

| 阶段 | 时点 | 允许看到 | 必须完成 | 写入范围 | 禁止 |
|---|---|---|---|---|---|
| **SAP-A** | 任何模型预测之前 | 无模型预测 | 冻结**方法与候选集**：前景阈值候选网格与选择指标、tie-break、共同阈值 vs 每模型独立阈值规则、3D 连通性与最小体积候选、候选置信度聚合与 detection map 转换、`picai_eval` 版本与参数、统计方法（patient-cluster bootstrap、AP cohort 重算、seed 报告规则）、bootstrap 次数与种子、FROC 操作点、**功效/精度备忘录（含 MCID 与来源）**、§1.1 命名映射；记录 SAP-A 方法块哈希 | YAML 的**方法块**与 `power_memo`（含 `protocol.reviewer` / `frozen_at` / `hashes`） | 不得用任何 validation/test 数值做选择；不得查看任何模型预测 |
| **SAP-B** | 冻结 checkpoint 之后、最终 test 之前 | PI-CAI validation 预测 | 按 SAP-A 的算法与网格在 validation 上**一次性填数**，并记录来源哈希与差异审计 | **只写 `sap_b_results`** 与 `sap_phases.sap_b`、`sees_validation_predictions` | 不得改动 SAP-A 方法字段与 `power_memo`（方法块哈希不一致 = 校验失败）；不得查看最终 test |
| **SAP-C** | 评估阶段（属 G5） | 最终 test 预测 | 封存全部配置与哈希、冻结模型集后，对最终 test **一次性评估**并完整报告 | `sap_phases.sap_c`、`sees_final_test_predictions` | 评估后任何修改都使该结果失去确认性资格 |

**SAP-B 唯一结果载体：`sap_b_results`（初始全部为 `null`）**

| 字段 | 说明 |
|---|---|
| `status` | `NOT_STARTED` → `COMPLETED`（COMPLETED 时下列字段必须齐全） |
| `threshold_policy` | `shared` 或 `per_model`（与 SAP-A 冻结的阈值规则一致） |
| `selected_foreground_threshold` | `threshold_policy=shared` 时必填 |
| `selected_foreground_threshold_per_model` | `threshold_policy=per_model` 时必填：显式 `model_id → value` 映射 |
| `selected_connectivity_3d` / `selected_min_lesion_volume_mm3` | 从 SAP-A 候选集中选定的数值 |
| `source_sap_a_config_sha256` | **必须等于** `sap_phases.sap_a.method_sha256` |
| `frozen_checkpoint_manifest_sha256` / `validation_prediction_manifest_sha256` / `selection_report_sha256` | 来源与选择报告哈希 |
| `sap_b_diff_audit_sha256` | SAP-A 方法块 vs SAP-B 方法块的结构化差异审计文件哈希 |
| `completed_at` / `reviewer` / `notes` | 审计信息 |

**SAP-A 方法块哈希与差异审计（机器可强制）**：

- 方法块 = `SAP_A_METHOD_PATHS` 列举的全部路径（阈值/后处理候选、`picai_eval` 参数、统计与 bootstrap 规则、
  操作点、`endpoints`、`comparison_hierarchy`、`power_memo` 全部字段等）；
- 哈希算法：`sha256(canonical_json(SAP_A_METHOD_PATHS))`，由
  `scripts/evaluate/prepare_g0_sap_freeze.py` 打印并写入冻结包（`sap_a_method_hash.txt`）；
- SAP-A 冻结时写入 `sap_phases.sap_a.method_sha256`；**此后任何方法字段改动都会使哈希不一致 → 校验失败**；
- 差异审计：`--sap-a-baseline <SAP-A 冻结快照>` 会逐路径比对并拒绝任何方法字段改动
  （基线快照 = 冻结包中的 `sap_a_snapshot.json`）。

**预测可见性字段（必须分字段记录）**：

- `sees_validation_predictions`：SAP-A 必须为 `false`；SAP-B COMPLETED 后置 `true`；
- `sees_final_test_predictions`：SAP-C COMPLETED 前必须为 `false`；执行后置 `true`；
- 兼容字段 `sees_model_predictions`：语义 = `sees_final_test_predictions`，**必须恒等**（校验器强制，
  见 §9.2 不可变式清单）。

**实现与运行边界**：`scripts/evaluate/` 的实现与合成回归必须在 **SAP-A 冻结前**完成（否则无法定义候选网格与
选择算法）；但**在真实 validation/test 上运行**只能在 SAP-B / SAP-C 进行。冻结前**不得**把任何真实评测数值
（含 nnU-Net `summary.json`、临时脚本输出）当作论文主结果。

## 1. 主要终点与比较层级

- **Primary endpoint**：阳性患者 **case-wise Dice**（阳性 = GT 含前景体素；预测为空记 0）；
- 关键次要指标：lesion-level average precision（AP）；
- 关键次要 FROC 指标：预定 FP/exam 操作点的 lesion sensitivity；
- 阴性检查指标：false-positive case rate、FP/exam。

**主比较层级（按顺序，不得跳级、不得事后更换）**：

1. **M4 vs M2**（分区条件总增益）；
2. 仅在第 1 项成立后，确认 **M4 vs M3**（解剖用于 gate 是否优于普通输入）。

确认性判定基于**不参与 checkpoint、阈值或超参选择的冻结最终 test**（G0-E 路径 A 或 C）；
PI-CAI validation 上的同类结果只用于开发与方向性支持。

### 1.1 命名语义映射：训练期 checkpoint 指标 vs 最终 test 终点（2026-09-17 新增）

| 角色 | 名称 | 定义 | 使用范围 |
|---|---|---|---|
| **训练期 checkpoint 指标**（实现字段，硬冻结） | `val_positive_casewise_dice_mean` | 阳性病例 case-wise Dice 的算术平均，最大化 | 训练期 validation event 上的选模；带 `val_` 前缀只表示「由自研 validator 在验证事件上产出」 |
| **最终 test 的语义终点（estimand）** | `positive_casewise_dice` | **同一数学定义**，在冻结最终 test 上一次性评估 | 论文表格/摘要；不参与任何选择 |

**规则**：

1. 两者**不得混用命名**：引用训练曲线时必须写出 `val_` 前缀；论文主结果报告的是
   `positive_casewise_dice`（最终 test）；
2. **不得把 `val_` 字段名当作最终终点名称**（例如在方法/结果节写“主要终点为
   `val_positive_casewise_dice_mean`”即为口径错误）；
3. 若要把实现字段重命名为与语义终点一致（去掉 `val_` 前缀或改为 `test_` 前缀），属**待用户决定事项**
   （`docs/research_plan.md` §20.2 D5）：改名会使既有 checkpoint、CSV、配置与内容哈希失配，必须升版并重跑回归；
4. 机器可读映射：`configs/protocols/g0_sap.yaml::endpoints.primary_endpoint_semantics`。

## 2. 阈值与后处理冻结

以下各项的**候选网格、选择方法与 tie-break rule**必须在首个正式 best checkpoint 的论文指标被查看前冻结
（写入本协议与 `configs/protocols/g0_sap.yaml`）：

1. **前景阈值**：候选网格（如 0.1–0.9 步长 0.05，具体网格以冻结配置为准）、在 PI-CAI validation 上的
   选择方法（如最大化阳性 case-wise Dice）与 tie-break（取更接近 0.5 的候选，或冻结其它预先写下的规则）；
2. **3D 连通性**：连通域结构元（6/18/26 邻域）与是否按最大值保留；
3. **最小病灶体积**：候选网格（mm³ 或体素）与选择方法；
4. **候选置信度聚合**：每个连通域的聚合函数（如域内概率均值/最大值）与分割图→detection map 的转换；
5. **阈值与后处理只在 PI-CAI validation 上选择一次**，然后对所有冻结 test 与全部模型保持一致；
   若某模型使用独立阈值，必须事先声明并对每个模型完全同规则执行。

**AP 输入**：使用候选病灶连续置信度排序，不得把某个 Dice 二值阈值后的固定预测当作完整 AP 输入。

## 3. 检测评测（picai_eval）

- 库：`picai_eval`（版本在冻结时写入 `picai_eval.version`，并记录安装来源）；
- 默认参数：`min_overlap` 对应 **IoU ≥ 0.1**（`overlap_func` 与 split/merge 规则一并写入 YAML）；
- `case-confidence` 函数（用于 AUROC 类指标）必须预先写下；
- 任何参数改动必须在观察模型结果前写入版本化评测配置。

## 4. 统计分析（patient-cluster）

1. **分析单位**：患者为 cluster；同一患者的多个 study 随患者一起重采样，不当作独立观测；
2. **Dice**：多 study 患者先在患者内对阳性 study Dice 取算术平均（阴性 study 只进入 FP 指标）；
   对每个冻结 seed 计算患者级配对差；
3. **主效应**：三个预定 seed 的平均配对差；患者级 paired cluster bootstrap 95% CI；
   **CI 的准确定位是 `conditional on the frozen trained seeds`** —— 它不覆盖 seed 采样不确定度；
4. **seed 不确定性的报告规则**（不得违反）：
   - 必须完整报告全部 seed-specific 效应（3 个配对差与各自的患者级 CI）；
   - 必须报告跨 seed 均值、SD 与取值范围；
   - 必须核查效应方向是否由单一极端 seed 主导；
   - `patient × seed` two-way / multi-bootstrap 只能作为**敏感性分析**，且必须声明
     3 个 seed 不足以稳定估计 seed 分布（尤其尾部）；**不得**包装成强确认性的 seed-level CI；
5. **AP**：AP 不是病例级可加指标；每个 bootstrap replicate 必须重采样患者及其全部 study，
   然后在重构的整个 cohort 上分别重算 M4 与对照模型的 AP，最后取配对差；
   **不得**构造 per-case AP 或对病例级 AP 取平均；
6. **lesion/FROC**：同样按患者 cluster bootstrap，不把同一患者的多个病灶当作完全独立样本；
7. **FROC 操作点**：主操作点冻结为 `0.5、1.0、2.0 FP/exam`；其余曲线仅作补充展示；
8. **多重比较纪律**：主比较只看 §1 的层级顺序；PZ/TZ、输入组合、其它指标与阈值敏感性属于
   预先声明的次级/探索性分析，不事后抬升为主终点；
9. **可复现性**：保存每个 seed 的病例级预测、候选病灶、指标、bootstrap 索引/种子与完整统计配置。

## 5. 功效/精度备忘录（SAP-A 冻结条件，v2.2；时点纪律 v2.3.1）

**时点规则（保守口径，不得放宽）**：

- `power_memo`（含 `mcid`、`mcid_source`、`success_threshold`、效应/方差假设与敏感性范围）**全部属于 SAP-A**：
  必须在**不查看任何模型预测**（含 validation 预测）的前提下填写；
- **SAP-B 不得修改 `power_memo`**；`power_memo.*` 全部路径都在 SAP-A 方法块哈希的覆盖范围内，
  改动会使 `sap_phases.sap_a.method_sha256` 不一致并被校验器拒绝；
- **当前不存在任何允许在 SAP-B 填写的"盲态 nuisance quantity"白名单字段**。若今后确有必须在 SAP-B 填写的量，
  必须先提出书面理由与精确字段白名单，经研究者批准并在协议中升版登记后才能加入（不得自行混入）。

模板字段见 YAML `power_memo`：

| 字段 | 内容 |
|---|---|
| `primary_endpoint` | 阳性患者 case-wise Dice（固定） |
| `n_positive_patients` | 由最终 test 的实际患者数填写（路径 A/B/C 各自记录） |
| `assumed_paired_effect` | 假定配对效应（来源必须写明：文献/盲态 pilot/区间） |
| `assumed_paired_sd` | 假定 paired-difference SD 及**来源**（文献范围 / 盲态 nuisance estimate） |
| `sd_source` / `sd_sensitivity` | 来源与范围敏感性（乐观/悲观区间） |
| `mde_95ci_width` | 由上述假设推出的最小可检测效应与预期 95% CI 宽度 |
| `mcid`（v2.3.1） | **最小实际/临床相关差异**：判定「增益是否有实际意义」的阈值；必须写出来源；无法确立时写 `INSUFFICIENT DATA` |
| `mcid_source`（v2.3.1） | MCID 来源（文献范围 / 专家共识 / 盲态 pilot 区间）；**不得**用观察到的 M4 效果反推 |
| `success_threshold` | 写下时的判定阈值（必须在看到正式模型结果前冻结）；主判定同时报告 CI 相对 `mcid` 的位置 |
| `verdict` | `sufficient` / `INSUFFICIENT DATA` |

硬性规则：

- **不得**使用观察到的 M4 效果反向制定成功阈值；
- 缺少可靠方差来源时只能使用文献范围或**盲态** nuisance estimate，并做范围敏感性；
- 外部样本量/语义不足以支持确认性判定时，必须在**看结果前**将其降级为探索性验证；
- 数据不足时只保留模板字段并写 `INSUFFICIENT DATA`，**不得伪造具体 MDE 数值**。

## 6. 一次性评估规则（test 纪律）

1. 冻结模型集（M2/M3/M4，全部预定 seed）与全部评测配置后，对最终 test **一次性评估**；
2. 评估期间不得调参、不得改后处理、不得选择更有利的 seed 或先验版本；
3. 若 test 结果被用于任何模型/配置选择，该数据立即失去测试资格（结果作废）；
4. 失败病例、置信区间与全部预定指标必须完整报告，不得选择性展示。

## 7. 配置内容哈希与协议变更

- 冻结时记录（SAP-A）：本协议文件、`configs/protocols/g0_sap.yaml`、评测脚本（`scripts/evaluate/` 代码版本）、
  `picai_eval` 版本、计划使用的 split/manifest/plan 哈希；
- **绑定规则**：每个正式 run 保存 split/manifest/plan、训练配置、G0-R 预处理身份与 G0-SAP 评测配置的
  内容哈希；任一哈希不一致时，不得把结果并入同一确认性比较（research_plan §16）；
- **变更规则（按阶段区分）**：
  - **SAP-A 冻结后、SAP-B 期间**：只允许填写 `sap_b_results`（数值与审计哈希）与阶段/可见性字段；
    `power_memo` 及全部 SAP-A 方法字段**不得改动**（方法块哈希不一致 → 校验失败；`--sap-a-baseline`
    差异审计会逐路径报出改动）；
  - 若 SAP-B 中发现必须**改变方法**（而非只是填数），必须停止填数、升级协议版本并说明原因，
    且相关结果只能作为探索性证据（§6）；
  - SAP-C 封存后：任何修改都使该 test 结果失去确认性资格（§6）；
  - 版本序列：`draft-0.x` → `frozen-1.0`（SAP-A 冻结）→ 后续 `1.x`；每次修改必须重记哈希。

## 8. 输出与可追溯字段（全部落盘）

| 字段 | 说明 |
|---|---|
| 输入 | 冻结 checkpoint、split/manifest/plan 路径与哈希、G0-R 预处理身份记录 |
| 输出 | `configs/protocols/g0_sap.yaml`（SAP-A 冻结版 + `sap_b_results`）、`scripts/evaluate/` 与合成回归测试、`evaluation_config.sha256`、**`sap_a_method_hash.txt`**、**`sap_a_snapshot.json`**（差异审计基线）、`stats_config.json`（bootstrap 种子/索引）、`power_memo.md`、SAP-A→SAP-B 差异审计报告、主结果表与图 |
| 版本 | 本协议版本 + 评测脚本版本 + `picai_eval` 版本 |
| 状态 | SAP-A：`protocol.status` / `sap_phases.sap_a.status` = `DRAFT` → `FROZEN`（仅当 §1–§7 全部写入、MCID 与来源齐全、方法块哈希记录后才可冻结）；SAP-B / SAP-C：独立记录在 `sap_phases` 与 `sap_b_results.status` |
| 审核人 / 日期 | 统计与评测执行人、复核人、SAP-A 冻结与 SAP-B 填数日期（`sap_b_results.reviewer` / `completed_at`） |
| 决策 | 是否通过功效备忘录（`sufficient` / `INSUFFICIENT DATA` + 降级决定）；MCID 及其来源 |
| 预测可见性 | `sees_validation_predictions`（SAP-B COMPLETED 后 true）、`sees_final_test_predictions`（SAP-C COMPLETED 后 true）；兼容字段 `sees_model_predictions` 恒等于后者 |

## 9. 工具能力与实现要求

> 工具与评测逻辑的**实现/执行状态**（是否已实现、是否已运行、就绪检查结果、缺失字段数）唯一维护在
> `docs/STATUS.md`；本节只描述能力要求与校验不可变式，不记录运行事实。

### 9.1 工具能力（要求）

| 工具 | 路径 | 必须提供的能力 |
|---|---|---|
| 冻结载体与就绪检查 | `scripts/evaluate/prepare_g0_sap_freeze.py` | 配置硬约束静态校验（主终点 / 层级比较 / bootstrap 定位 / `picai_eval` 口径 / FROC 操作点 / 阶段一致性）；`--check-freeze` 报告 SAP-A 冻结阻塞项（`--fail-if-not-ready` 退出码 3）；打印 **SAP-A 方法块哈希**；可选 `--sap-a-baseline` 执行**结构化差异审计**；生成 `evaluation_config.sha256`、`sap_a_method_hash.txt`、`sap_a_snapshot.json`、`power_memo_template.md`、`freeze_checklist.md`、`stats_config.template.json`、`run_metadata.json`；输出路径隔离；`status=FROZEN` 且阻塞项非空时拒绝生成冻结载体 |
| 校验与模板逻辑 | `src/zonal_reliability_fusion/protocols/g0_sap.py` | 冻结必需字段清单（含 `power_memo.mcid` / `mcid_source` / `sap_phases.sap_a.method_sha256`）、功效备忘录字段完整性、阶段与可见性校验、方法块哈希与差异审计、模板渲染（数值一律 TODO/null） |

**SAP-A 冻结前必须就绪的字段**：以 `g0_sap.py::FREEZE_REQUIRED_FIELDS` 为唯一来源（阈值候选网格、选择指标、
tie-break、3D 连通性、最小体积、候选置信度聚合与 detection map 转换、`picai_eval` 版本与参数、bootstrap 次数与
种子、功效备忘录各字段含 `mcid`/`mcid_source`、`sap_phases.sap_a.method_sha256`、`reviewer`/`frozen_at` 与
`hashes`）。缺任何一项时**不得**置 FROZEN。

### 9.2 校验器必须强制的不可变式（invariants）

| # | 不可变式 |
|---|---|
| 1 | `protocol.sees_model_predictions` **必须恒等于** `protocol.sees_final_test_predictions` |
| 2 | `protocol.status=FROZEN` ⇔ `sap_phases.sap_a.status=FROZEN`（顶层状态**仅**表示 SAP-A） |
| 3 | SAP-A 未冻结时，`sees_validation_predictions` / `sees_final_test_predictions` 必须为 `false` |
| 4 | SAP-A 冻结时必须记录 `sap_phases.sap_a.method_sha256`，且其值必须等于当前方法块哈希（任何方法字段改动都会失败） |
| 5 | SAP-B `NOT_STARTED` 时 `sap_b_results` 的所有结果字段必须为空（不得预填） |
| 6 | SAP-B `COMPLETED` 时必须：SAP-A 已冻结、`sees_validation_predictions=true`、`sap_b_results.status=COMPLETED`、全部必填结果字段与 5 个 SHA256 完整；`threshold_policy=shared` 需共享阈值，`=per_model` 需显式 `model_id → value` 映射且不得同时填共享阈值 |
| 7 | `sap_b_results.source_sap_a_config_sha256` 必须等于 `sap_phases.sap_a.method_sha256` |
| 8 | `sees_final_test_predictions=true` 只能在 `sap_phases.sap_c.status=COMPLETED` 之后；SAP-C `COMPLETED` 要求 SAP-B 已 `COMPLETED` |
| 9 | 传入 `--sap-a-baseline` 时：SAP-A 方法块的任何逐路径改动都被差异审计拒绝 |

### 9.3 实现待办（不含运行状态）

- [ ] 实现 `scripts/evaluate/`（阈值选择、`picai_eval` 调用、patient-cluster bootstrap、AP cohort 重算、
      分层指标汇总）与**合成回归测试**——必须在 SAP-A 冻结前完成；
- [ ] SAP-A：填写方法与候选集、记录方法块哈希 → `protocol.status` 与 `sap_phases.sap_a.status` 置 `FROZEN`；
- [ ] SAP-B：在 PI-CAI validation 上按 SAP-A 算法一次性填数到 `sap_b_results` + 差异审计；
- [ ] 填写功效/精度备忘录（§5；无数据时写 `INSUFFICIENT DATA`）；
- [ ] 与 G0-E 决策（路径 A/B/C）对齐确认最终 test 的指标范围。

> 协议状态：**DRAFT（SAP-A 未冻结）**。实现/执行状态、就绪检查与阻塞项见 `docs/STATUS.md`；
> 通过条件与失败转向见 `docs/research_plan.md` §11.4 与 §14 G0-SAP。
