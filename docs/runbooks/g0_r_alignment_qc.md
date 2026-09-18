# Runbook：G0-R 序列错位协议（全自动 QC，真实数据）

- **协议**：`docs/protocols/G0_R_ALIGNMENT_QC_AUTOMATED.md`（`G0-R-AUTOMATED` **draft-0.3**，DRAFT；draft-0.2 载体已归档于 `configs/protocols/archive/`）
- **配置**：`configs/protocols/g0_r_alignment_qc_automated.yaml`
- **状态与阻塞项**：见 `docs/STATUS.md`（本手册不维护状态）
- **执行人**：研究者（读取真实物化影像体素）

## 0. 前置条件

1. 抽样清单已冻结：`outputs/diagnostics/g0_r/20260915_093819/sampling_manifest.json`
   （`total_cases=16`，`decision=null`，`warnings=[]`；**不可替换、增删或重抽样**）；
2. 物化影像可读：`data/processed/picai/cases`（原始/物化数据只读，工具不写回、不配准写回、不联网）；
3. 目标输出目录不存在或为空（工具对非空目录直接拒绝覆盖）。

## 1. 命令（复制执行）

```bash
cd /opt/data/private/lm/my-projects
conda activate lm
source scripts/env_nnunet.sh

G0R_MANIFEST="outputs/diagnostics/g0_r/20260915_093819/sampling_manifest.json"
G0R_RUN="outputs/diagnostics/g0_r_automated/$(date -u +%Y%m%d_%H%M%S)"

# 1) 计划检查（不读影像体素、不写文件）
python scripts/audit/run_picai_alignment_qc_automated.py \
  --config configs/protocols/g0_r_alignment_qc_automated.yaml \
  --sampling-manifest "$G0R_MANIFEST" \
  --out-dir "$G0R_RUN" \
  --dry-run

# 2) 正式自动 QC（默认进度条；合成位移校准占主要耗时）
python scripts/audit/run_picai_alignment_qc_automated.py \
  --config configs/protocols/g0_r_alignment_qc_automated.yaml \
  --sampling-manifest "$G0R_MANIFEST" \
  --out-dir "$G0R_RUN"
```

可选：`--overlay`（仅审计用，无需人工查看）、`--max-cases N`（调试）、`--no-progress`（日志环境）。

## 2. 预期输出

```text
outputs/diagnostics/g0_r_automated/<UTC 时间戳>/
  automated_metrics.csv        每 case×pair 的自动指标 + 刚体诊断 + FOV 摘要
  per_case_decisions.csv       状态/原因/主指标与阈值/逐指标 checks
  calibration_summary.json     分 pair（(pair, case_id) unit）校准曲线、unit 内噪声阈值、单调性、
                               检出率、零位移误报率、分 pair passed、schema_version / grouping
  threshold_derivation.json    每 pair 阈值（by_pair）、unit baseline / min-detectable 均值 / midpoint、
                               聚合方法（median）、方向、sensitivity_passed
  decision_draft.json          候选决策（DRAFT）+ 分状态占比 + 成对结论
  qc_geometry.json / input_hashes.json / run_metadata.json / skipped.csv / qc_report.md
```

结束摘要会打印：成功/失败/跳过病例数、ADC 与 HBV 各自结论、校准是否通过、输出路径、候选决策。

## 3. 进度观察

- 进度条覆盖病例循环与校准循环（可用 `--no-progress` 关闭）；
- 结束摘要给出耗时与输出路径；`run_metadata.json` 记录协议哈希、命令与 `data_written_back=false`。

## 4. 成功判据（≠ 门 PASS）

1. `calibration_summary.json.passed = true`，且**两个 pair 的主指标各自通过**
   （`per_pair["T2W-ADC"].passed` 与 `per_pair["T2W-HBV"].passed` 均为 true；monotonicity ≥ 0.8、
   检出率 ≥ 0.9、零位移误报 ≤ 0.05）；
2. `decision_draft.json.candidate ∈ {resample-only, resample+rigid}`（**不是** `INSUFFICIENT_EVIDENCE`）；
3. ADC 与 HBV 的 `pair_conclusions` 分别给出结论，未跳过任一对；
4. 无 `INVALID_INPUT` / `FOV_INSUFFICIENT` / `REGISTRATION_DIAGNOSTIC_FAILED` 超限比例；
5. 所有 JSON 输出带 `schema_version = g0-r-automated/0.3`（读取一律用 `load_output_json()`）。

> **即使全部满足，也只是"DRAFT 候选可用"**：自动路径只产出候选，0 mm 状态不等于解剖学对齐真值；
> 冻结仍须研究者按 §5 执行。

## 5. 冻结程序（研究者）

1. 复核 `qc_report.md`、`calibration_summary.json`、`decision_draft.json`；
2. 在协议（或新建冻结记录）写入 `frozen_decision`、`protocol_hash`、`reviewer`、`frozen_at` 与本次
   `run_metadata.json` 哈希；
3. 若候选为 `resample+rigid`：**必须新建派生数据、重新预处理并复检**，不得在旧派生数据上直接训练；
4. 若为 `INSUFFICIENT_EVIDENCE`：**必须停止**，不得据此启动 P2B / 正式 M0 / N0；
5. 更新 `docs/STATUS.md`，并把执行记录写入 `docs/experiment_log.md`。

## 6. 已知局限（写进论文/回复审稿人时必须保留）

- 跨模态边缘一致性是**相对指标**，绝对数值依赖标准化与边缘阈值；判定完全依赖合成位移校准；
- 校准只在部分病例上进行（`calibration.max_cases`），中心/协议间对比度差异带来系统不确定性；
- 诊断性刚体估计在低对比、强各向异性或小 FOV 数据上可能失败（记为失败标记，不伪造结果）；
- 一致性指标在稠密边缘数据上可能饱和，会被排除出判定并记录；
- 抽样仅 16 例：**样本量与证据充分性属于待冻结决策**（`docs/research_plan.md` §20.2 D2），
  本 runbook 不对「是否足以支撑配准决策」下结论；是否追加人工 landmark 复核或扩大抽样由研究者决定；
- 合成位移只验证"指标对额外位移的响应"，**不能**证明真实病例解剖学对齐；观察到的 0 mm 状态不等价于真值；
- draft-0.2 的失败运行（`outputs/diagnostics/g0_r_automated/20260918_074219/`）**原样保留**为历史证据，
  不得修改、不得与 draft-0.3 结果合并比较。

## 7. 历史路径（不再是 blocker）

人工双阅片路径 `docs/protocols/G0_R_ALIGNMENT_QC.md` + `scripts/audit/render_picai_alignment_qc.py`
保留为历史草案与后备（`evaluation.landmark_rules.min_per_case_pair = null`）；如需启用人工路径，
必须先冻结 landmark 规则并由研究者执行渲染与填写。
