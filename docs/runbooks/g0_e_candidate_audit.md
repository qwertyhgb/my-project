# Runbook：G0-E 独立测试候选审计与路径冻结

- **协议**：`docs/protocols/G0_E_INDEPENDENT_TEST.md`（`G0-E` draft-0.1，DRAFT）
- **配置**：`configs/protocols/g0_e_independent_test.yaml`
- **状态与阻塞项**：见 `docs/STATUS.md`（本手册不维护状态）
- **执行人**：研究者（只读候选数据目录；**不查看任何模型在外部数据上的结果**）

## 0. 前置条件与重要说明

1. 候选目录只读：`/opt/data/private/lm/data/Prostate/{Prostate158,PCA,BPH,MSD_Prostate}`；
2. 工具只做**文件清单级**清点（目录名/文件名/CSV 表头/行数），**不推断独立性或可用性**；
3. **哈希绑定规则**：清点工具会把运行时的 `config_sha256` 写入 `run_metadata.json`。若配置在清点之后被修订，
   则该绑定不再代表现行配置 → **若要以修订后的配置冻结 G0-E，必须重新运行清点并绑定新哈希**
   （既有产物保留，不追溯改写；当前哈希变化记录见 `docs/STATUS.md` B9 与 `docs/protocol_changelog.md`）。

## 1. 命令（复制执行）

```bash
cd /opt/data/private/lm/my-projects
conda activate lm

# 1) 只看清点计划（不遍历目录内容、不写文件）
python scripts/audit/audit_g0e_candidates.py --dry-run

# 2) 正式清点（只读元数据；输出到 outputs/diagnostics/g0_e/<UTC 时间戳>/）
python scripts/audit/audit_g0e_candidates.py
```

可选：`--candidate Prostate158`（只审计单个候选）、`--out-dir <dir>`、`--no-progress`。

## 2. 预期输出

```text
outputs/diagnostics/g0_e/<时间戳>/
  candidate_registry.csv     候选登记表（未填字段留空，不得代填）
  evidence_status.json       逐字段证据状态（VERIFIED / AUTO_ONLY / UNKNOWN / CONFIGURED_EXCLUDED）
  inventory_summary.json     AUTO_ONLY 清点事实（目录数/文件计数/CSV 行数）
  missing_evidence.md        研究者待补证据清单（2026-09-15 运行：67 项）
  audit_report.md            审计报告模板（UNKNOWN/PENDING 明确标注）
  run_metadata.json          工具、时间、config_sha256、python/numpy 版本
```

## 3. 成功判据

- 命令正常结束并在 `run_metadata.json` 中记录新的 `config_sha256`；
- `candidate_registry.csv` 与 `evidence_status.json` 生成、**字段不得用假设填充**；
- `verdict` 与 `frozen_decision` 仍为空（工具不代填）。

## 4. 审计清单（研究者逐项补齐，任一未证实都只能降级）

| # | 项目 | 证据要求 |
|---|---|---|
| 1 | 许可与使用范围 | 官方许可/说明链接或文件副本（Prostate158 等公开集；私有 PCA/BPH 需机构书面确认） |
| 2 | 患者重叠 | 与 PI-CAI 全量 1500 study 的患者级比对（重叠必须为 0 才可作 test）；附脚本与输出 |
| 3 | 标签语义 | 病灶/阳性/阴性/分级定义与 PI-CAI `ISUP ≥ 2 csPCa` 逐条对照；不等价 → 只能 `exploratory_only` |
| 4 | 只有阳性病例 | 只能报告分割类指标；禁止 AUROC / FP-per-exam |
| 5 | 模态映射 | T2 轴位/fs、ADC 是否为计算 ADC、HBV 的 b 值区间；未核实序列从主分析排除 |
| 6 | PZ/TZ 编码 | 值→语义映射来源；无权威定义 → `zonal_semantics_unresolved`（仅 oracle 路径） |
| 7 | **先验独立性** | Yuan23 / HeviAI23 训练数据是否覆盖 PI-CAI 或外部病例；无法证明独立 → `oracle/prior-friendly`，不得作确认性结果 |
| 8 | 几何/强度审计 | size/spacing/origin/direction、dtype、强度范围、FOV 覆盖与事先写下的排除规则 |
| 9 | 指标范围 | 登记 `metrics_supported`，超出部分禁止报告；外部指标与 PI-CAI validation 分表呈现 |

参考：`docs/PICAI_Model_Readiness_Audit.md` §9，以及外部权威源（PI-CAI 官方数据页、`picai_labels` 仓库）——
见 `docs/literature/novelty_and_standards_check_20260917.md`。

## 5. 冻结程序（路径 A / B / C）

1. 按协议 §9 决策树逐候选判定；
2. 写入 `verdict`（`eligible_external_test` / `exploratory_only` / `rejected`）与理由；
3. 冻结 `frozen_decision`：路径 **A**（合格同任务 external test）/ **B**（事先降级为探索性外部验证）/
   **C**（无合格外部集 → 在任何正式模型结果产生前重划并冻结 PI-CAI internal test，当前 1277/223 split
   作废但保留审计）；
4. 记录协议、YAML、全部输出文件的内容哈希；
5. 更新 `docs/STATUS.md`，执行记录写入 `docs/experiment_log.md`。

## 6. 停止条件

- **路径 B/C 的决定必须在查看任何正式模型结果之前完成**，之后不得改判；
- 外部数据一旦被用于任何模型/配置选择，立即失去测试资格（结果作废）；
- 若所有候选均 `rejected` 且无法重划 internal test，则不得启动正式 N0/M0–M4 的可报告运行。
