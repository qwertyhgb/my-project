# Runbook：G0-SAP 评测与统计冻结（SAP-A / SAP-B / SAP-C）

- **协议**：`docs/protocols/G0_SAP.md`（`G0-SAP` draft-0.2，DRAFT）；生命周期定义见其 §0.1
- **配置**：`configs/protocols/g0_sap.yaml`
- **状态、阻塞项与就绪检查结果**：见 `docs/STATUS.md`（本手册**不维护状态**）
- **执行人**：研究者（SAP-B / SAP-C 读取真实影像与预测）

## 0. 阶段与允许看到的输入（先读）

| 阶段 | 时点 | 允许看到 | 允许做的事 | 允许写入 | 禁止 |
|---|---|---|---|---|---|
| **SAP-A** | 任何模型预测之前 | 无模型预测 | 冻结**方法与候选集**（阈值网格与选择指标、tie-break、共同/独立阈值规则、后处理候选、`picai_eval` 参数、统计与 bootstrap 规则、FROC 操作点、功效备忘录、MCID、命名映射） | 方法块 + `power_memo` + `protocol.reviewer`/`frozen_at`/`hashes` + `sap_phases.sap_a.method_sha256` | 不得用任何 validation/test 数值做选择 |
| **SAP-B** | 冻结 checkpoint 之后、最终 test 之前 | PI-CAI validation 预测 | 按 SAP-A 的算法与网格**一次性填数**并输出差异审计 | **仅 `sap_b_results`** + `sap_phases.sap_b` + `sees_validation_predictions` | 不得改动 SAP-A 方法字段与 `power_memo`；不得查看最终 test |
| **SAP-C** | 评估阶段（属 G5） | 最终 test 预测 | 封存全部配置与哈希、冻结模型集后**一次性评估**并完整报告 | `sap_phases.sap_c` + `sees_final_test_predictions` + `sees_model_predictions`（兼容字段，必须与之一致） | 评估后任何修改都使结果失去确认性资格 |

**术语澄清（避免与旧表述冲突）**：本手册中「不得查看论文指标」指**不得在 SAP-A 之前查看任何正式 checkpoint
的最终 test 指标**；SAP-B 允许使用 PI-CAI validation 的**开发指标**（阈值选择正是为此），但只能在 SAP-A 已冻结
的算法与候选网格下**一次性**使用。`sees_validation_predictions` 与 `sees_final_test_predictions` 必须分字段记录；
兼容字段 `sees_model_predictions` 必须**恒等于** `sees_final_test_predictions`（校验器强制）。

## 1. SAP-A：冻结方法与候选集（看模型结果前）

**前置条件**：G0-E 路径 A/B/C 已冻结（决定最终 test 与可用指标范围）；`scripts/evaluate/` 实现与合成回归已完成
（否则无法定义候选网格与选择算法）。

```bash
cd /opt/data/private/lm/my-projects
conda activate lm

# 1) 冻结就绪检查（只读；未就绪时退出码 3，属预期）
python scripts/evaluate/prepare_g0_sap_freeze.py --check-freeze

# 2) 只打印将写入的文件（不写文件）
python scripts/evaluate/prepare_g0_sap_freeze.py --dry-run

# 3) 生成冻结载体（evaluation_config.sha256、sap_a_method_hash.txt、sap_a_snapshot.json、
#    power_memo 模板、freeze_checklist、stats_config 模板、run_metadata）
python scripts/evaluate/prepare_g0_sap_freeze.py
```

**SAP-A 必须写齐的字段**（以 `g0_sap.py::FREEZE_REQUIRED_FIELDS` 为唯一来源，`--check-freeze` 会列出缺口）：

| 类别 | 字段 |
|---|---|
| 审阅与日期 | `protocol.reviewer`、`protocol.frozen_at` |
| 前景阈值 | `candidate_grid`、`selection_metric`、`tie_break_rule` |
| 后处理 | `connectivity_3d`、`min_lesion_volume.candidate_grid`、`candidate_confidence.aggregation`、`candidate_confidence.segmentation_to_detection_map` |
| picai_eval | `version`、`overlap_func`、`split_merge_rule`、`case_confidence_function` |
| 统计 | `statistics.bootstrap.n_resamples`、`statistics.bootstrap.seed` |
| 功效备忘录 | `status`、`n_positive_patients`、`assumed_paired_effect`、`assumed_paired_sd`、`sd_source`、`sd_sensitivity_range`、`mde_95ci_width`、`mcid`、`mcid_source`、`success_threshold` |
| SAP-A 方法块哈希 | `sap_phases.sap_a.method_sha256`（用上面的脚本打印值；**不得手算**） |
| 哈希 | `hashes` |

**填写纪律（协议 §5）**：

- **不得**用观察到的 M4 效果反向制定成功阈值或 MCID；
- 缺少可靠方差来源时，只能使用文献范围或**盲态** nuisance estimate，并做乐观/悲观区间敏感性；
- 外部样本量或语义不足以支持确认性判定时，必须在**看结果前**降级为探索性验证；
- 数据不足时只保留字段并写 `INSUFFICIENT DATA`，**不得伪造 MDE / MCID 数值**；
- `power_memo` 全部字段（含 MCID）属于 **SAP-A**，SAP-B 不得修改。

**SAP-A 收尾**：把 `sap_phases.sap_a.status` 与 `protocol.status` 同时置 `FROZEN`、填写 `reviewer`/`frozen_at`
并在 YAML 中写入脚本打印的方法块哈希 → 重跑本脚本生成冻结包（确认 `freeze_checklist.md` 阻塞项清零）→
保留 `sap_a_snapshot.json` 供 SAP-B 差异审计 → 更新 `docs/STATUS.md`，执行记录写入 `docs/experiment_log.md`。

## 2. SAP-B：在 validation 上一次性填数（checkpoint 冻结后）

**前置条件**：候选模型集与 checkpoint 已冻结；SAP-A 已 `FROZEN` 且方法块哈希未变。

1. 按 SAP-A 冻结的算法与候选网格，在 **PI-CAI validation** 上执行**一次**阈值与后处理选择；
2. **只填写 `sap_b_results`**（初始全部为 `null`）：
   `threshold_policy`、`selected_foreground_threshold` **或**
   `selected_foreground_threshold_per_model`、`selected_connectivity_3d`、
   `selected_min_lesion_volume_mm3`、`source_sap_a_config_sha256`（= SAP-A 方法块哈希）、
   `frozen_checkpoint_manifest_sha256`、`validation_prediction_manifest_sha256`、
   `selection_report_sha256`、`sap_b_diff_audit_sha256`、`completed_at`、`reviewer`；
   **不得**改动 SAP-A 方法字段与 `power_memo`；
3. 生成 **SAP-A → SAP-B 差异审计**并以 `--sap-a-baseline` 复核（应报告"未改动方法字段"）：

```bash
cd /opt/data/private/lm/my-projects
conda activate lm
python scripts/evaluate/prepare_g0_sap_freeze.py \
  --sap-a-baseline outputs/metrics/evaluation/<version>/sap_a_snapshot.json \
  --check-freeze
```

4. 置 `sap_phases.sap_b.status=COMPLETED` 与 `protocol.sees_validation_predictions=true`，
   并重新运行本脚本记录新哈希（`protocol.sees_model_predictions` 保持 `false`，因为最终 test 仍未查看）；
5. 更新 `docs/STATUS.md` 与 `docs/experiment_log.md`（含审计哈希与产物路径）。

**若发现必须改变方法**：停止填数，按协议 §7 升级版本并说明原因，相关结果只能作为探索性证据。

## 3. SAP-C：封存与最终 test 一次性评估（属 G5）

1. 封存全部配置与哈希（评测配置、split/manifest/plan、G0-R 预处理身份、训练配置）与冻结模型集
   （M2/M3/M4 + 全部预定 seed）；
2. 对最终 test 执行**一次性**评估，按 `docs/research_plan.md` §3.1/§13 报告全部预定指标、置信区间与失败病例；
3. 置 `sap_phases.sap_c.status=COMPLETED`、`protocol.sees_final_test_predictions=true`，并将兼容字段
   `protocol.sees_model_predictions` **同步设为 `true`**（校验器要求两者恒等）；结果与产物路径写入
   `docs/experiment_log.md`；
4. **评估后任何修改都使该 test 结果失去确认性资格**。

## 4. 停止条件

- SAP-A 未 `FROZEN`：不得在真实 validation / test 上运行评测，也不得把任何数值（含 nnU-Net `summary.json`、
  临时脚本输出）作为论文主结果；
- 出现方法性变更：升版 + 说明 + 相关结果降级为探索性；
- 任何用 test 结果做模型/配置选择的行为：该数据立即失去测试资格。
