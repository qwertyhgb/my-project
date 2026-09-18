# Runbooks：由研究者执行的运行手册

本目录保存**可直接复制执行的完整命令**、输出路径、进度观察方式与成功判据。
命令归属规则见根目录 `AGENTS.md` §2：数据处理、重采样、预处理、审计、推理、验证、评测与训练
**一律由研究者执行**；协作代理只整理命令、检查返回的日志与产物。

## 通用约定（每条命令都必须满足）

```bash
cd /opt/data/private/lm/my-projects
conda activate lm
source scripts/env_nnunet.sh        # 涉及 nnU-Net / plan / split 的命令
```

- Python 必须是 `/root/anaconda3/envs/lm/bin/python`；
- 长任务默认显示 tqdm 进度条（需要干净日志时用显式 `--no-progress`）；
- 结束后仍须核对结构化日志中的成功/失败/跳过数与输出路径；
- 运行完成后：**实际命令、环境、耗时、结果、失败与产物写入 `docs/experiment_log.md`**；
  代码/配置/协议变更写入 `docs/Development_Log.md`；状态变化更新 `docs/STATUS.md`。

## 手册索引

> 各门的**当前状态、阻塞项与就绪检查结果**见 `docs/STATUS.md`；本目录只保存**前置条件与命令**，
> 不维护任何动态事实（是否已运行、缺多少字段、测试通过数等）。
> 编号（`G0-*`、`G1`–`G5`、`P0`–`P10`、`SAP-A/B/C` 等）的含义与推进顺序见 `docs/GLOSSARY.md`。

| 手册 | 对应门 / 阶段 |
|---|---|
| `g0_r_alignment_qc.md` | G0-R 序列错位协议冻结（自动 QC） |
| `g0_e_candidate_audit.md` | G0-E 独立测试路径冻结（候选审计与路径 A/B/C） |
| `g0_sap_freeze.md` | G0-SAP 评测与统计冻结（SAP-A 冻结 / SAP-B 填数 / SAP-C 封存评估） |
| `p2a_zonal_prior_materialization.md` | P2A′：PZ/TZ prior 物化（plan-space sidecar，M3/M4 的前置数据） |
| `p2b_m0_validation.md` | G2 自研基线可信（P2B 真实数据三步验证 + 1-step 峰值显存实测） |
| `n0_training.md` | G1 官方基线可复现（正式 N0 训练） |

## 纪律

1. **工具运行成功 ≠ 阶段门 PASS**：门的状态只能由研究者按 `docs/research_plan.md` §14 的通过条件判定，
   并记录在本目录手册对应的协议载体与 `docs/STATUS.md`；
2. **不得覆盖既有产物**：所有工具输出到新建/空的时间戳目录；已排除的预实验、失败 run、legacy 产物
   **不得删除或覆盖**；
3. **不在真实数据上试跑**：任何 smoke / 单病例 / `--limit` 运行都属于真实数据操作，必须由研究者执行；
4. 命令涉及的配置文件若在运行后被修订，**旧运行的配置哈希绑定不再代表新配置**
   （示例见 `docs/protocol_changelog.md` 2026-09-17 条目）。
