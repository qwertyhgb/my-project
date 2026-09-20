# 协议版本变更史（protocol changelog）

> **本文件是研究协议版本历史的唯一权威来源。**
> `docs/research_plan.md` 只保留当前有效条款与**简短**版本摘要；完整修改过程、原因与影响记录在本文件。
>
> 记录纪律（参照 SPIRIT 2025 关于「方案是活文件、每一版方案都应包含透明审计留痕，记录修改日期与内容描述」的原则，
> 见 `docs/literature/novelty_and_standards_check_20260917.md`）：
> 1. 每个版本必须写明**日期、修改内容、修改原因、是否发生在查看模型结果之前、对既有结果的影响**；
> 2. 已冻结的数据、split、plan 与事后结果不得被版本修订追溯改写；
> 3. 任何在观察到模型结果之后发生的修订，必须显式标注并降级相关结论；
> 4. 本文件只记录协议与协议载体，不记录运行事实（运行事实见 `docs/experiment_log.md`）。

| 版本 | 日期 | 主题 | 是否在查看正式模型结果前 | 对既有结果的影响 |
|---|---|---|---|---|
| v2.0 | 2026-09-14 | 研究计划建立（含跨中心/域泛化主线）与 G0-P 训练前条件 | 是（当时无任何模型结果） | — |
| v2.0（状态同步） | 2026-09-15 | 状态修正 + 代理命令执行边界写入 §16.2 | 是 | 无（不改变数据/split/plan） |
| v2.1 | 2026-09-15 | 收缩主线、融合位置与 M3/M4 设计修订、新增 G0-R/G0-SAP、N0 定位 | 是（M0 未真实训练） | 无；旧 M0 配置与 run 仍保留，但须重新迁移验证 |
| v2.2 | 2026-09-15 | 低频验证与固定预算落地、WG 退出主模型输入、序列消融口径重写、先验独立性、统计口径与功效备忘录 | 是（M0 未真实训练） | 无；Dice+CE 预实验被排除但不改写记录 |
| v2.3 | 2026-09-16 | M0–M4 共同骨干升级为 plan-driven Residual-Encoder，冻结浅层 skip 契约与主模型边界 | 是（M0–M4 均未真实训练） | 无；旧 PlainConv M0 降级为 legacy/feasibility，不得混表 |
| **v2.3.1** | 2026-09-17 | **协议 amendment**：G0-SAP 三阶段生命周期（SAP-A/B/C）与预测可见性字段拆分；MCID 与主判定规则；M1/M2 容量混杂控制；G0-R 证据充分性决策与文献系统检索要求 | 是（仍无任何正式模型结果；M0–M4 未训练） | 无历史结果可影响，但**改变实验设计与结论强度**：H2/G3 可能降级为探索性；成功判定需 MCID；G0-SAP 冻结时点重新定义 |
| （文档结构重构，不含协议修订） | 2026-09-17 | research_plan 文档边界重构；新增 STATUS/changelog/runbooks；修复状态矛盾与 G0-E 配置字段 | 是 | 无；除已登记为 v2.3.1 的四项实质性新增外，仅迁移与澄清 |
| （执行顺序豁免 + 标签语义澄清，不含条款修订） | 2026-09-17/18 | ① M1–M4 融合代码**提前实现**（研究者工程指令）；② seg 存储态 `-1` 哨兵映射，对齐官方 `RemoveLabelTansform(-1,0)`；③ 运行时峰值显存遥测（工程侧，详见 `docs/Development_Log.md`） | 是（仍无任何正式模型结果；M0–M4 未训练、未读真实 prior、未评测） | 无历史结果可影响；两点注意见下方专节 |
| **G0-R-AUTOMATED draft-0.3 → draft-0.4**（协议**载体版本升级**，门定义未变） | 2026-09-18 | ① 指标 `sigma*.usable` 守卫（缺失/false/非布尔一律 fail-closed；主指标 unusable → INSUFFICIENT_EVIDENCE，一致性指标 → SKIPPED）；② 校准 unit 条件网格完整性（`min_complete_units_per_pair: 3`、`require_complete_condition_grid: true`；不完整 unit 不得静默移出分母）；③ 输出 schema `g0-r-automated/0.4` | 是（仍无任何正式模型结果；M0–M4 未训练、未评测） | 无科学结果可影响；draft-0.2 失败产物与 draft-0.3 配置归档保留，不得追溯修改或跨 schema 混用 |
| **G0-R-AUTOMATED draft-0.2 → draft-0.3**（协议**载体版本升级**，门定义未变） | 2026-09-18 | ① 刚体诊断 transform 类型修复（`inPlace=True` + 显式类型校验 + 单元素 Euler Composite 解包，否则 fail-closed）；② 校准按 `(pair, case_id)` unit 分组、零位移参考 unit 内均值、零位移噪声 leave-one-out；③ 真实病例阈值按 pair 推导（unit midpoint 的 median）、与灵敏度门限分离命名、`decide_case` 强制 pair；④ 输出 schema 版本化（`g0-r-automated/0.3`） | 是（仍无任何正式模型结果；M0–M4 未训练、未评测） | 无科学结果可影响；draft-0.2 首次真实运行产物**原样保留**为历史证据，不得追溯修改或与 v0.3 结果混用 |

---

## G0-R-AUTOMATED draft-0.3 → draft-0.4（2026-09-18；**协议载体版本升级，门定义未变**）

- **修改内容（两项 fail-open 缺陷修复）**：
  ① **指标可用性守卫**：指标名解析尺度前缀（`sigma1.5.edge_f1_at_1.0mm` → `sigma1.5.usable`），
  只有该字段**显式为 `true`** 才可用；缺失 / `false` / 非布尔一律 fail-closed。
  主指标 unusable → `UNAVAILABLE` → `INSUFFICIENT_EVIDENCE`；一致性指标 unusable → `SKIPPED`
  （写明"边缘体素不足"，不得用于支持 ACCEPTABLE/FLAGGED）。输出记录
  `metric_usable` / `metric_unusable_reason`。修复前的缺陷：`_metric_value()` 与 `decide_case()`
  仅检查数值有限，**忽略 `sigma*.usable`**，使 `preprocessing.edge.min_edge_voxels` 事实失效；
  ② **校准 unit 条件网格完整性**：新增冻结配置项
  `calibration.min_complete_units_per_pair: 3` 与 `calibration.require_complete_condition_grid: true`；
  逐 `(case_id, pair, metric)` 审计（零位移重复数 == `stability_repeats`、每个非零位移 × 全部方向、
  `min_detectable_mm` 必须覆盖全部方向、全部记录数值有限且 `usable=true`）；只有 complete unit 参与
  pooled SD / 曲线 / 检出率 / midpoint；complete unit 不足或存在任何不完整 unit → 该 pair 校准失败。
  输出显式区分 `n_units_total` / `n_units_complete` / `n_units_at_min_detectable` /
  `n_units_participating` / `incomplete_unit_ids` / `missing_conditions` / `invalid_conditions`。
  修复前的缺陷：缺 `min_detectable_mm` 数据的 unit 会被**静默移出** detection-rate 分母与 midpoint 聚合，
  只剩 1 个 unit 也可能通过；
- **载体**：`configs/protocols/g0_r_alignment_qc_automated.yaml`（`protocol.version: draft-0.4`、
  `decision.rule_version: "0.4"`、`protocol.output_schema_version: g0-r-automated/0.4`）、
  `docs/protocols/G0_R_ALIGNMENT_QC_AUTOMATED.md`；
- **draft-0.3 归档（按字节）**：`configs/protocols/archive/g0_r_alignment_qc_automated_draft_0_3.yaml`
  ，SHA256 `beda4bbf1065c9c8493fe41a640f5e4a257076b256438b8dd493cbedfa678198`（旧 `protocol_hash` = `5aeed6e8d1c039a371fb8d9e553da4752dce89d3c51526d9afdcf1ec8318655c`）；draft-0.2 归档与失败运行产物均未修改；
- **修改原因**：独立审查发现上述两个 fail-open 缺陷；两者都会**放行本来应当判为证据不足的输入**，
  属安全问题而非精度问题；
- **是否在查看正式模型结果之前**：是（仍无任何正式模型结果）；
- **对既有结果的影响**：无科学结果可影响；**draft-0.3 从未在真实数据上运行**（故不存在需要重解释的
  draft-0.3 运行）；draft-0.2 失败运行仍为历史证据；
- **门定义与预注册数值未变**：抽样 16 例、序列对、位移集合（`[0,0.5,1,2,3,4] mm`）、方向集合、
  `min_detectable_mm=2.0`、检出率下限 `0.9`、`detection_multiplier=3.0`、单调性下限、零位移误报上限、
  `min_edge_voxels`、`input_policy`、`privacy`、不写回原则**全部未修改**（有逐项比对测试锁定）；
- **G0-R 仍为 PENDING / DRAFT**：draft-0.4 工具运行成功同样**不等于** PASS。

---

## G0-R-AUTOMATED draft-0.2 → draft-0.3（2026-09-18；**协议载体版本升级，门定义未变**）

- **修改内容（方法性，全部为已发生缺陷的修复）**：
  ① **刚体诊断 transform 类型**：改用 `SetInitialTransform(initial, inPlace=True)`（SimpleITK 2.5.3 下
  `inPlace=False` 会让 `Execute()` 返回 `CompositeTransform`，旧代码对其强转 Euler3D 会抛
  `Transform is not of type Euler3DTransform!`）；对 initial/final 做显式类型校验，兼容性解包只允许
  「确实是 CompositeTransform + 仅含一个 transform + 内部确实是 Euler3DTransform」，否则 fail-closed；
  ② **校准分组**：最小单元改为 `(case_id, pair)`，两个序列对完全独立；零位移参考为 unit 内均值，
  零位移噪声用 leave-one-out 残差（`max(k × pooled_within_unit_sd, max unit 内 LOO 退化, 0)`），
  禁止把病例间 / pair 间的绝对基线差异混入同一 SD；
  ③ **阈值与命名**：真实病例阈值按 pair 推导
  （`threshold(pair, metric) = median(unit_midpoint)`，`unit_midpoint = (unit 零位移基线 + unit 在
  min_detectable 处各方向均值)/2`），灵敏度门限与真实病例阈值分离命名；`decide_case()` 必须显式传
  `pair`，未知 pair 直接失败，**不得回退全局阈值**；
  ④ **输出 schema**：所有 JSON 输出带 `schema_version = g0-r-automated/0.3` 与协议身份；读取必须经
  `load_output_json()` 校验，禁止把 draft-0.2 旧输出按 v0.3 schema 解释；
- **载体**：`configs/protocols/g0_r_alignment_qc_automated.yaml`（`protocol.version: draft-0.3`、
  `decision.rule_version: "0.3"`）、`docs/protocols/G0_R_ALIGNMENT_QC_AUTOMATED.md`；
- **draft-0.2 归档（按字节）**：`configs/protocols/archive/g0_r_alignment_qc_automated_draft_0_2.yaml`
  ，SHA256 `67c479a055736693d290d017d2fc9bbe2f08558a593bc40cf76414920659c89a`
  （与失败运行 `outputs/diagnostics/g0_r_automated/20260918_074219/run_metadata.json` 的
  `config_sha256` 一致）；旧 `protocol_hash` =
  `9dfdd010fa60abc8dbd4ee91cee565dc85769694daa7a3deabe645de38e8ddd1`（保留可追溯）；
- **修改原因**：draft-0.2 首次真实运行（2026-09-18，`outputs/diagnostics/g0_r_automated/20260918_074219/`）
  记录 32/32 `REGISTRATION_DIAGNOSTIC_FAILED`（类型错误，全部相同异常），且合成位移校准在真实跨模态数据上
  检出率 0（`zero_sd≈0.0698` 主要来自病例间 / pair 间基线差异，2 mm 退化仅 ≈0.008）——两者都是**工具缺陷**，
  不是数据质量结论；
- **是否在查看正式模型结果之前**：是（仍无任何正式模型结果；M0–M4 未训练、未评测）；
- **对既有结果的影响**：无科学结果可影响；draft-0.2 失败运行产物**原样保留**（不修改 JSON/CSV/报告/哈希），
  仅作为历史证据与修复依据，**不得**与 draft-0.3 结果合并比较；
- **门定义未变**：`docs/research_plan.md` §4.2.1 / §14 的 G0-R 通过条件、失败转向与「冻结前不得启动
  P2B / 正式 M0 / N0」均未改变（`research_plan` 本轮未修改；其 §4.2.1/§14.1 中的「Automated v0.2」是
  **载体版本名**，版本对应关系以本条目为准）；**G0-R 仍为 PENDING / DRAFT**，
  draft-0.3 工具运行成功同样**不等于** PASS；
- **未改变的预注册项**：抽样 16 例清单、序列对、位移集合、方向集合、`min_detectable_mm = 2.0`、
  检出率下限 `0.9`、`detection_multiplier = 3.0`、单调性下限、零位移误报上限、`input_policy`、
  `privacy`、不写回原则。

---

## 执行顺序豁免与标签语义澄清（2026-09-17/18；**不含条款修订**）

**1. M1–M4 实现顺序偏离（研究者工程指令）**

- `docs/research_plan.md` §9.4.2 要求「M1–M4 实现」位于 **G2 通过之后**；本轮按研究者指令**提前实现**
  （只写代码 + 合成测试：不读真实数据、不启动训练、不产出任何科学结果）；
- **该文件的条款字面未改**（本轮 `docs/research_plan.md` 保持冻结）；本条目只登记偏离事实，
  不构成对 §9.4.2 的修订；
- **正式训练仍受 G2 约束**：M0–M4 正式训练的前置条件不变（G2 + G0-R + G0-E 路径冻结 + 增强/预算协议冻结）；
  M3/M4 另需 plan-space prior 物化完成（`docs/STATUS.md` B10）。

**2. seg 存储态 `-1` 哨兵映射（契约澄清，非口径变更）**

- 事实：nnU-Net 预处理的 `.b2nd` 中 seg 可含 `-1`；Dataset605 `dataset.json` 只定义
  `background=0 / lesion=1`，**无语义 ignore 类**；官方训练与验证 transform 均执行
  `RemoveLabelTansform(-1, 0)`（纯值映射，不裁剪、不截断）；
- 处置：读取适配层 `normalize_seg` 改为**先验证、后转换**——仅接受精确取值 `{-1,0,1}`，
  将 `-1` 映射为 0，返回严格 `uint8 ⊆ {0,1}`；`-2/2/0.5/1.5/NaN/±Inf/complex/字符串/object` 一律拒绝；
- **不改变任何已冻结内容**：split、plan、预处理产物、prior/manifest 均未变动；标签**语义口径仍是
  Focal+CE 的 0/1 二值**，`seg` 进入采样/损失/指标前不含哨兵；
- 记录：`docs/Development_Log.md`（代码变更）、`tests/unit/test_seg_sentinel_mapping.py`（32 项合成测试，
  含与官方 transform 的逐体素等价性）。

**3. 运行时峰值显存遥测（工程补充，非协议条款）**

- `--overfit` 现会写出 `outputs/diagnostics/<model_id>/<run_name>/diagnostics/gpu_memory_profile.json`
  （当前 **PyTorch 进程**的 `peak_allocated/reserved`，覆盖整个 `trainer_run`）；
- `models/profiling.py` 的静态 `peak_gpu_memory.status = NOT_MEASURED` **保持不变**，
  运行实测是独立产物，二者不得互相替代（`docs/STATUS.md` B5 关闭条件 = 实测报告存在）。

---

## v2.0（2026-09-14）研究计划建立

- **修改内容**：初始化 `docs/research_plan.md`，主线为「基于解剖分区条件自适应多序列融合的**跨中心**前列腺癌病灶分割」；
  建立 G0-P 的 8 项训练前条件（方向码修正、manifest/geometry audit 重建、ADC/HBV 重采样、canonical 二值标签、
  geometry_suspect 人工 QC、异常 WG 屏蔽、新 split、模态顺序与校验值冻结）。
- **修改原因**：在训练前冻结数据与几何口径。
- **是否在查看模型结果之前**：是（当时无模型结果）。
- **影响**：无历史结果可影响；该 8 项条件在 P0A/P0B 全部完成，G0-P 于 2026-09-14 判定 PASS。

## v2.0 状态同步（2026-09-15）

- **修改内容**：抬头、§6、§15 的滞后状态修正；把「代理可执行/不可执行的命令边界」写入 §16.2。
- **修改原因**：与 P0A/P0B 实际完成情况、以及 AGENTS.md 的命令归属规则保持一致。
- **是否在查看模型结果之前**：是。
- **影响**：无；不改变数据、split、plan。

## v2.1（2026-09-15）协议修订

来源：`docs/Development_Log.md`「研究协议 v2.1 修订」；当时的 §0.1.2 现迁入本文件。

- **修改内容**
  1. 将 NaMa、NesMFle、LeSMI 纳入最近邻自适应融合工作，**M2 明确定位为基线而非创新**；
  2. M4 由「解剖 logit 加法」改为**零初始化 conditional affine modulation**：`L = (1+γ_Z)⊙L_img + β_Z`；
  3. M3 改为使用相同 zone encoder，但解剖特征在序列融合**之后**注入，不能间接改写 image gate；
  4. 融合位置固定为冻结 `3d_fullres` plan 的 **encoder stage 2 输出**（累积 stride `(1,4,4)`），不再使用「1/4 分辨率」表述；
     分区 one-hot 降采样改称 **fractional zonal occupancy**，不得称为 softmax 概率；
  5. M0–M4 目标验证频率由「每 epoch」改为**每 5 epoch 全体积验证**，early stopping 改为按 **validation event** 计数；
  6. 新增 **G0-R**（序列错位决策）与 **G0-SAP**（评测/统计冻结）；G0-E 明确为「独立测试路径冻结」——
     同任务外部 test 不合格时，必须在任何可报告的正式 N0/M0–M4 run 前重划 PI-CAI internal test；
  7. 主终点固定为阳性患者 case-wise Dice，主比较按 M4−M2 → M4−M3 层级执行；明确 patient-cluster paired bootstrap
     与 AP 在每个 bootstrap cohort 内整体重算的区别；
  8. **N0 改为协议不完全配平的 contextual strong reference**，不对 N0−M0 差异作单一因果解释；
  9. 取消「跨中心分割 / 域泛化」主线；不再设置 held-out center，不做 LOCO。
- **修改原因**：修复正式 M1–M4 实验前的归因与可执行性问题；避免把「学习三个序列权重」本身当作创新。
- **是否在查看模型结果之前**：是（M0 尚未真实训练，仅合成测试）。
- **影响**：旧 M0 配置/代码**尚未迁移**，当时不得写成已完成；迁移与回归测试完成前不启动正式训练。
  已在 v2.2 完成迁移。

## v2.2（2026-09-15）协议修订 + 低频验证代码迁移

来源：`docs/Development_Log.md`「研究协议 v2.2 与低频验证代码迁移」；当时的 §0.1.1 现迁入本文件。

- **代码迁移（合成 CPU 回归，未读真实医学数据）**：`trainer.py` 的 validation-event 调度、最后一 epoch 必验证、
  非验证 epoch 不更新选模状态、best 只在 validation event 产生、resume 恢复 event 状态、协议变化拒绝/重置；
  `full_volume_validator.py`、`config/experiment.py`、`train_m0.py`、`configs/experiments/m0_picai_3d_fullres.yaml`
  同步；该轮的实际测试命令与结果见 `docs/experiment_log.md`（本文件不记录运行事实）。
- **协议修订**
  1. 把「每 5 epoch 验证」升级为**功能开发 checklist**（v2.3 起该 checklist 的实现记录迁至
     `docs/P2_M0_Implementation.md` 附录 A；`research_plan` 只保留规范性要求）；
  2. 正式 M0–M4 改为**固定 `max_epochs=200` + `validation.every_n_epochs=5` + 禁用「连续无改善」early stopping**，
     只保留 NaN/Inf/覆盖病例数错误等安全中止；PolyLR 终点与 `max_epochs` 一致；
  3. **WG 退出所有主模型（M0–M4）的 patch 采样与输入**；WG 只用于 QC、统一评测分层/假阳性分析与单列消融；
     不允许 M4 独占 WG；
  4. §12.2 重写：`all / −T2W / −ADC / −HBV` 是 **4 个全局输入配置，各只训练一次**，PZ/TZ/mixed/outside 是
     **评价分层而非训练维度**；推理期置零只作 `stress test`；
  5. 新增**先验独立性规则**（§5.4 / §12.4 / G0-E）：必须审计 Yuan23/HeviAI23 等分区器训练数据是否覆盖
     PI-CAI（尤其 validation/test）或外部测试病例；无法证明独立时只能标记 `oracle/upper-bound` 或
     `prior-friendly`；形态学扰动只评价鲁棒性，**不得**声称消除或量化泄漏；
  6. 统计协议修订：主 CI 定位为 **`conditional on the frozen trained seeds`**；必须同时报告全部 seed-specific
     效应与跨 seed 均值/SD/范围；3 seed 不得包装为强确认性 seed-level CI；AP 必须在每个 bootstrap cohort 内
     整体重算；新增**功效/精度备忘录**作为 G0-SAP 通过条件（数据不足时写 `INSUFFICIENT DATA`，不得伪造 MDE）。
- **修改原因**：把 v2.1 宣布但未落地的低频验证与预算协议变成可执行、可回归测试的工程事实；补齐归因纪律。
- **是否在查看模型结果之前**：是。
- **影响**：无历史结果可影响；「代码迁移或合成测试通过」**不得**使 G2/G0-R/G0-E/G0-SAP 任一变为 PASS。

## v2.3（2026-09-16）架构修订

来源：`docs/research_plan.md` 旧 §0.1、`docs/Development_Log.md`「P2A」「P2A 验收轮修复」；当时的 §0.1 现迁入本文件。

- **修改内容**
  1. 正式 M0–M4 的共同骨干由 PlainConv 3D U-Net 升级为 **plan-driven Residual-Encoder 3D U-Net**；
     继续沿用冻结 `3d_fullres` 的 spacing、patch、kernel、stride、通道上限、deep supervision 与推理几何；
     目标复杂度以 nnU-Net ResEnc-M 级别为起点，不采用 L/XL；
  2. Residual Encoder、普通 decoder、标准 skip、deep supervision 均为**共同骨干，不作创新**；
     M0–M4 必须使用同一骨干，禁止只为 M4 升级；
  3. M1–M4 的 modality-specific 分支只延伸到 encoder stage 2；stage 0/1 的三路浅层 skip 固定为
     `concat → 1×1×1 projection → InstanceNorm → LeakyReLU` 聚合；stage 2 skip 使用 `F_fuse`；
     stage 3 及以后使用共享 encoder skip；浅层 skip 聚合器不读取 PZ/TZ、不生成第二套序列 gate，
     并在 M1–M4 间完全一致（接口契约见 `docs/P2_M0_ResidualEncoder_v23.md` §7）；
  4. 主模型不加入 anatomy-conditioned skip、Attention U-Net gate、UNet++/UNet3+ 密集 skip、
     Transformer/Mamba bottleneck 或多尺度序列 gate；
  5. 旧 `PlanDrivenUNet3D` / PlainConv M0 及旧配置统一降级为 **legacy/feasibility**，不得与 v2.3 M1–M4 混入同一正式消融表；
  6. N0 定义不变，仍为成熟 contextual reference；N0 与 M 系列在训练预算和骨干上均不完全配平，不对 N0−M0 作单一因果解释。
- **修改原因**：在 M1–M4 尚未实现、M0 尚未真实训练之前，用更成熟的残差骨干保证后端能力，
  同时彻底冻结浅层 skip 处理，避免 gate 旁路或 M4 独占额外解剖路径。
- **是否在查看模型结果之前**：是（M0–M4 均未真实训练；仅 P2A 合成回归）。
- **影响**：无历史结果可影响。P2A 已完成实现与合成回归（G2 前置 1–7），G2 第 8 项仍 Pending（见 `docs/STATUS.md`）。

## v2.3.1（2026-09-17）协议 amendment（文档重构期间新增的实质性要求）

> **定性**：这**不是**文档迁移。以下四项要求在本轮新增，**发生在任何正式模型结果之前**，按 amendment 纪律
> 单独升版登记；它们会改变实验设计或结论强度，必须按 v2.3.1 执行。

- **修改内容**
  1. **G0-SAP 生命周期拆分（SAP-A / SAP-B / SAP-C）**：原条款把「阈值与后处理选择」笼统放在「冻结前」，
     与本项目实际流程（必须先有冻结 checkpoint，才能在 PI-CAI validation 上选阈值）冲突，并与
     「评测逻辑不得在冻结前实现」相互矛盾。现拆为：
     - **SAP-A（看模型结果前）**：冻结**方法与候选集**——前景阈值候选网格与选择指标、tie-break、
       共同阈值 vs 每模型独立阈值规则、3D 连通性与最小体积候选、候选置信度聚合与 detection map 转换、
       `picai_eval` 版本与参数、统计方法（patient-cluster bootstrap、AP cohort 重算、seed 规则）、
       bootstrap 次数与种子、FROC 操作点、功效/精度备忘录（可先为 `INSUFFICIENT DATA`）；
     - **SAP-B（checkpoint 冻结后）**：按 SAP-A 的算法与网格，在 PI-CAI validation 上**一次性填数**，
       只允许写入数值，不得更改 SAP-A 已冻结的方法；记录"仅数值变化"的差异审计；
     - **SAP-C（封存 + 最终 test）**：封存全部配置与哈希、冻结模型集后，对最终 test **一次性评估**；
       评估后任何修改均使结果失去确认性资格。
     预测可见性字段拆分：`sees_validation_predictions`（SAP-B 执行后置 true）与
     `sees_final_test_predictions`（SAP-C 执行后置 true）；兼容字段 `sees_model_predictions` 的语义
     = `sees_final_test_predictions`（保留以兼容现有代码校验，代码层强制拆分字段列为 P0E 实现待办）。
  2. **MCID 与主判定规则**：成功标准原仅有「效应为正且 95% CI 不跨 0」。现要求在看结果前写入
     最小实际/临床相关差异（MCID）及其来源，并在主判定中报告点估计、CI 与 CI 相对 MCID 的位置；
     无法确立 MCID 时只能声明「分割精度增益」，不得声称临床相关性。
  3. **M1/M2 容量混杂控制**：M2 相对 M1 额外引入 gate 子网络参数，原协议未要求容量对照。现要求
     M1/M2 运行前冻结二选一：**同容量对照**（参数规模相当但不读取跨序列比较信息）或
     **显式降级 H2/G3 为探索性证据**；M3/M4 沿用 §8.4 的 1% 参数差与 constant-occupancy 对照规则。
  4. **G0-R 证据充分性决策 + 文献系统检索要求**：明确 16 例自动相对指标（跨模态边缘一致性 + 合成位移校准）
     不足以单独支撑配准决策，须显式决定是否追加人工 landmark 复核或扩大抽样；新颖性表述必须以系统检索
     （数据库、检索式、截止日期、逐项机制对照）为前提，未完成前不得使用「首次」。
- **修改原因**：① 消除 G0-SAP 中「评测逻辑不得在冻结前实现」与「必须先实现才能冻结」的循环；
  ② 防止把统计显著但实际可忽略的差异写成成功；③ 防止把容量增益误读为动态融合机制收益；
  ④ 防止以非系统检索支撑新颖性声明。
- **是否在查看模型结果之前**：是（仍无任何正式模型结果；M0–M4 未训练）。
- **对既有结果的影响**：无历史结果可影响，但**结论强度规则变化**：H2/G3 在无容量对照时降级为探索性；
  成功判定需 MCID；G0-SAP 的通过时点由「一次冻结」改为「SAP-A 冻结 → SAP-B 填数 → SAP-C 封存评估」。
- **载体同步**：`docs/research_plan.md`（§2.2 / §3.1 / §4.2.1 / §8.8 / §11.4 / §14）；
  `docs/protocols/G0_SAP.md`（新增生命周期与字段语义，版本 `draft-0.1` → `draft-0.2`）；
  `configs/protocols/g0_sap.yaml`（新增预测可见性字段与 `sap_phases` 块，版本同步 `draft-0.2`）；
  `docs/runbooks/g0_sap_freeze.md`（改为 SAP-A/B/C 三段命令）。
- **哈希影响**：`configs/protocols/g0_sap.yaml` 修订后内容 SHA256 =
  `bdc14967b555a9a6bcf4f0fb44340d6441a55236090aaccc851dff1f38dd7ce4`。此前 `draft-0.1` 版本**未生成过冻结包**
  （冻结载体输出目录不存在），因此不存在需要作废的绑定；首次 SAP-A 冻结时以新哈希为准。
  `g0_e_independent_test.yaml` 在上一轮（同日的文档重构）已记录哈希变更（见下一条目）。

### v2.3.1 载体补记（2026-09-17）：G0-SAP 生命周期闭环

> **定性**：不新增科学要求，也不放宽任何 v2.3.1 条款；只把 SAP-A/B/C 在**机器可读 schema、校验器、
> runbook 与协议文档**之间对齐，并落实此前登记的两个 P0E 实现待办。

- **修改内容**
  1. **SAP-B 独立结果块**：`configs/protocols/g0_sap.yaml` 新增 `sap_b_results`（`status`、`threshold_policy`、
     `selected_foreground_threshold`（或显式 `model_id → value` 映射 `selected_foreground_threshold_per_model`）、
     `selected_connectivity_3d`、`selected_min_lesion_volume_mm3`、`source_sap_a_config_sha256`、
     `frozen_checkpoint_manifest_sha256`、`validation_prediction_manifest_sha256`、`selection_report_sha256`、
     `sap_b_diff_audit_sha256`、`completed_at`、`reviewer`）；初始全部为 `null`。
     **SAP-B 只写该块**，不得覆盖 SAP-A 的候选/方法字段。
  2. **SAP-A 方法块哈希与差异审计**：新增 `SAP_A_METHOD_PATHS` 与
     `sap_a_method_hash()`（`sha256(canonical_json(method_paths))`）、`diff_sap_a_methods()`；
     `sap_phases.sap_a.method_sha256` 为 SAP-A 冻结必需字段；SAP-B 的 `source_sap_a_config_sha256`
     必须与之相等；冻结载体新增 `sap_a_method_hash.txt` 与 `sap_a_snapshot.json`，
     工具新增 `--sap-a-baseline` 执行逐路径差异审计。
  3. **power_memo 时点纪律（保守）**：`power_memo`（含 MCID、MCID 来源、`success_threshold`、
     效应/方差假设与敏感性范围）**全部属于 SAP-A**；SAP-B 不得修改；**不存在**允许在 SAP-B 填写的
     盲态 nuisance quantity 白名单字段；如确需新增，必须先提出理由与精确字段白名单并升版登记。
  4. **两处 P0E 待办落实**（此前登记在 `docs/protocols/G0_SAP.md` §9.2）：
     - `power_memo.mcid` / `mcid_source` 纳入 `POWER_MEMO_FIELDS`、`FREEZE_REQUIRED_FIELDS` 与备忘录模板渲染；
     - 校验器强制：兼容字段 `sees_model_predictions` 恒等于 `sees_final_test_predictions`、
       `protocol.status=FROZEN` ⇔ `sap_phases.sap_a.status=FROZEN`、SAP-A 未冻结时不得有预测可见性、
       SAP-A 冻结后方法块哈希必须一致、SAP-B `COMPLETED` 时结果字段与 SHA256 齐全、
       `sees_final_test_predictions=true` 仅能在 SAP-C `COMPLETED` 之后。
  5. **顶层状态语义澄清**：`protocol.status=FROZEN` **仅表示 SAP-A 方法冻结**；SAP-B/SAP-C 状态只由
     `sap_phases.*.status` 表示（`protocol.status_semantics` 字段与 runbook 同步说明）。
- **修改原因**：SAP-B 原先没有独立的机器可读落盘位置，导致"只填数值"与"不得改动方法字段"在两个阶段之间
  缺少可校验边界；`power_memo` 的填写时点存在歧义；两项 P0E 校验缺口此前只停留在待办。
- **是否在查看模型结果之前**：是（无任何正式模型结果；M0–M4 未训练；SAP-A 未冻结）。
- **对既有结果的影响**：无历史结果可影响。**SAP-A 冻结门槛提高**：`mcid`、`mcid_source` 与
  `sap_phases.sap_a.method_sha256` 现在会阻止冻结；`protocol.status=FROZEN` 不再能单独表示阶段。
- **载体与实现**：`configs/protocols/g0_sap.yaml`（版本 `draft-0.2` → `draft-0.3`）、
  `src/zonal_reliability_fusion/protocols/g0_sap.py`、`src/.../protocols/common.py`（兼容字段一致性检查）、
  `scripts/evaluate/prepare_g0_sap_freeze.py`、`tests/unit/test_g0_sap_freeze.py`、
  `tests/unit/test_g0_scripts.py`、`tests/unit/test_g0_protocol_common.py`、
  `docs/protocols/G0_SAP.md`、`docs/runbooks/g0_sap_freeze.md`、`docs/research_plan.md` §11.4/§14.1。
- **哈希影响**：`configs/protocols/g0_sap.yaml` 内容 SHA256 记为
  `5c5847fd545899801a97bf3086ffc57022ed32c3bffbf95ba8c1ee57e84524fe`；仍未生成过冻结包，无绑定作废。
  实际测试与就绪检查结果见 `docs/experiment_log.md`（本文件不记录运行事实）。

## 2026-09-17 文档结构重构（除已登记为 v2.3.1 的条目外，不改变协议条款）

- **修改内容**
  1. `docs/research_plan.md` 文档边界重构：把运行事实、工程完成记录、命令、代理规则、重复状态迁出；
    计划内只保留研究定位、假设、数据/独立测试设计、模型与消融、冻结训练协议、评价与统计、阶段门定义、
    风险与失败转向、简短版本摘要与关联文档索引；
  2. 新建 `docs/STATUS.md`（唯一状态看板）、`docs/protocol_changelog.md`（本文件）、`docs/runbooks/`（用户执行命令）；
  3. 修正 `configs/protocols/g0_e_independent_test.yaml` 的 `audit_tool.status`：清单级清点**已执行**
     （`outputs/diagnostics/g0_e/20260915_093835/`），但**证据审计与 verdict 仍为 UNKNOWN/null**；
  4. 修正 `docs/protocols/G0_E_INDEPENDENT_TEST.md` §11.1 的同一过期状态；
  5. `configs/protocols/g0_sap.yaml` / `docs/protocols/G0_SAP.md` 增加**语义映射**：区分训练期 checkpoint 指标
     实现字段 `val_positive_casewise_dice_mean` 与最终 test 的语义终点名 `positive_casewise_dice`
     （未改动任何实现字段，命名变更需用户决定）；
  6. `docs/P2_M0_Implementation.md` 新增附录 A，接收自 research_plan §9.4.1 迁出的 v2.2 低频验证功能开发 checklist
    与文件清单（信息不丢失，不再在计划中维护实时状态）；
  7. 运行手册与协议文档中的动态事实（测试通过数、尚缺字段数、是否运行过）统一收敛到 `docs/STATUS.md`
     与 `docs/experiment_log.md`（见 `docs/STATUS.md` 的"状态唯一来源"声明）。
- **修改原因**：同一状态在抬头、§6、§9、§14、§15、§20 多处并行维护导致矛盾（例如 P2A 既写「已完成」又写「并行推进实现」）；
  计划文件混入运行日志与代理规则，违反「计划 vs 结果」的文档分工原则。
- **是否在查看模型结果之前**：是（仍无任何正式模型结果；M0–M4 未训练）。
- **对既有结果的影响**：
  - 除已登记为 v2.3.1 的四项新增外，不改变其他协议条款、数据、split、plan 或产物；
  - `configs/protocols/g0_e_independent_test.yaml` 内容哈希因此变化：清点运行（2026-09-15）绑定的是
    `90953cba3660f71858bbe19a7112574a7e380d1d9ace791635e3aa5f58cc16ce`，修订后现状为
    `9960a4522eec2354c90736ca5ef705390ac6bde589226800fee83980e53d1595`（SHA256 of file bytes）；
    **若需以修订后配置冻结 G0-E，必须重新运行清点并记录新哈希**（既有产物的已记录内容不被追溯改写，
    见 `docs/STATUS.md` B9）；
  - `configs/protocols/g0_sap.yaml` 新增非必填的映射字段与生命周期字段，不改变 `endpoints.primary` 与
    冻结就绪判据的字段清单（判据见 `docs/protocols/G0_SAP.md` §9；实测就绪状态见 `docs/STATUS.md`）。

## 自研消融族预算与验证调度变更（2026-09-20；**预注册**，D7 部分处置）

| 项 | 变更前 | 变更后 |
|---|---|---|
| `training.max_epochs`（`m0_resenc_picai_3d_fullres_v23` + `m1`–`m4`，共 5 个配置） | 200 | **1000** |
| `validation.every_n_epochs`（同上 5 个配置） | 5 | **50** |

**理由（基于正式 N0 的实测曲线，非结果后调整）**：正式 N0 在 1000-epoch 预算下，`Pseudo dice` 直到
**epoch 600 之后**才进入正常区间（首个非零 epoch 43；epoch 0–200 非零仅 81/200、峰值 0.090；
最后 200 epoch 才 200/200 非零、峰值 0.767 且仍未见收敛）。若 M0–M4 仍用 200 epoch，整段预算
将与 N0 的"双方均欠训练"状态重叠，`N0 vs M0` 的差距无法解释。同时把每 5 epoch 的全体积验证
（223 例 ≈ 11 min/轮）放宽为每 50 epoch，以控制单模型墙钟成本（验证轮数 40 → 20）。

**不变的规则**：**最后一个 epoch 必验证**（代码实现：`epoch == self.epochs - 1`）；
`checkpoint_metric = val_positive_casewise_dice_mean`、`maximize=true`、禁用"连续无改善"early stopping、
`expected_cases = 223 / 63 / 160`、`step_fraction=0.5`、`gaussian=true`、`mirror_tta=false` 全部不变。

**预注册声明**：本变更在**任何 M0–M4 训练结果产生之前**完成并登记（此前 M0–M4 从未训练，
`outputs/checkpoints/` 仅含 `.gitkeep`），因此**不属于**「结果产生后调整」。变更后首次运行即为
该预算下的正式（或 feasibility 身份的）run。

**仍不配平、因此仍待 D7 处置的部分**：骨干仍为 Residual-Encoder（N0 为 PlainConv）→
`N0 − M0` 差异**仍不能作单一因果解释**；D7 的「增加 matched-protocol run」（例如再跑一版 N0 的
官方 250-epoch 变体做对照）仍开放，须由研究者在看正式结果前决定。

**保留不动**：legacy `configs/experiments/m0_picai_3d_fullres.yaml`（v2.2 PlainConv，已排除的历史配置）
继续使用冻结的 200 epoch / 每 5 epoch 值；`tests/unit/test_train_m0_script.py::
test_repo_config_declares_formal_protocol` 仍针对该 legacy 文件断言 200/5。

**影响面**：5 个配置的内容哈希变化（这些配置从未训练过，故无既有 run 受影响）；`PolyLR` 终点由
`build_scheduler(..., epochs=total_epochs)` 自动跟随 `max_epochs`，无需改代码；新增测试
`test_formal_resenc_family_declares_1000_epoch_protocol` 锁定本次冻结值。
