# 从这里开始（日常入口）

> 本页只回答一件事：**我要做这一步，敲哪条命令。**
> 约定：所有命令在项目根目录执行；先 `conda activate lm`，再 `source scripts/env_nnunet.sh`。
> 日常只需要本页 + `docs/experiment_log.md`；其它文档都可以不看。

---

## 1. 现在的状态（一句话）

| 事项 | 状态 |
|---|---|
| 数据 | 1500 例已物化 + nnU-Net 预处理就绪（`data/`、`workdir/`）——**已就绪，随时可用** |
| 基线 N0 | 可随时启动（§3）；已有正式启动记录见 `docs/experiment_log.md` |
| 自研 M0–M4 | 配置与代码就绪，可随时启动（§4） |
| **前置/审批** | **不存在需要批准才能启动的东西**；未完成的门只影响结果标签（§7） |

---

## 2. 准备数据（已完成；仅在需要重做时用）

数据侧已经全部完成，日常不需要重跑。需要时看脚本自带的 `--help` 与对应 runbook：

| 目的 | 入口 |
|---|---|
| 病例级物化（1500 例） | `scripts/data/materialize_picai.py`（`--help`） |
| nnU-Net raw 准备 / split / 预处理 | `scripts/data/prepare_picai_nnunet_raw.py`、`scripts/data/create_nnunet_splits.py` |
| PZ/TZ prior 物化（M3/M4 用） | `python scripts/data/materialize_zonal_prior.py --source yuan --resume`，说明见 `docs/runbooks/p2a_zonal_prior_materialization.md` |

---

## 3. 训练官方基线 N0

```bash
cd /opt/data/private/lm/my-projects
conda activate lm
source scripts/env_nnunet.sh
CUDA_VISIBLE_DEVICES=0 python scripts/train/train_n0_picai_flce.py 605 3d_fullres 0
```

- 中断后续训：末尾加 `--continue-training`；
- 日志：`outputs/nnUNet_results/Dataset605_PICAI/nnUNetTrainerPICAI_FLCE_NoFFT__nnUNetPlans__3d_fullres/fold_0/training_log_*.txt`；
- 规模与耗时：1000 epoch、约 53 s/epoch、显存约 6 GB → **约 15 小时**；成功标志 = 日志出现 `Training done` + 产出 `checkpoint_final.pth`；
- **训练结束后评测**：`python scripts/evaluate/evaluate_n0_validation.py`（默认评测该 fold 的 223 例 validation，结果写入 `outputs/metrics/n0_validation_<时间戳>/`：`metrics.json` + `per_case.csv` + `summary.md`）；
- 细节（隔离规则、失败处理）：`docs/runbooks/n0_training.md`。

---

## 4. 训练自研模型（M0 / M1–M4）

```bash
cd /opt/data/private/lm/my-projects
conda activate lm
source scripts/env_nnunet.sh

# ① 只读自检（不用 GPU，先确认数据/配置/prior 都齐）
python scripts/train/validate_m0_setup.py --config configs/experiments/m0_resenc_picai_3d_fullres_v23.yaml

# ② 真实数据 loader 检查（不用 GPU；确认能正确读到影像与 prior）
python scripts/train/train_m0.py --config configs/experiments/m0_resenc_picai_3d_fullres_v23.yaml --loader-smoke

# ③ 小样本过拟合 + 显存实测（用 GPU，几分钟）
python scripts/train/train_m0.py --config configs/experiments/m0_resenc_picai_3d_fullres_v23.yaml \
  --overfit --cases 10005_1000005,10021_1000021

# ④ 正式训练
python scripts/train/train_m0.py --config configs/experiments/m0_resenc_picai_3d_fullres_v23.yaml
```

> **正式预算（2026-09-20 起）**：`max_epochs=1000`（与 N0 的 epoch 预算对齐）、全体积验证每 **50** epoch
> （**最后一个 epoch 必验证**）。单模型墙钟成本 ≈ 训练 17–25 h + 验证 20 轮 × ≈11 min ≈ **1 天量级**。
> 变更登记见 `docs/protocol_changelog.md`（2026-09-20）。

可选配置（`configs/experiments/`）：

| 模型 | 配置 |
|---|---|
| M0（plain 基线） | `m0_picai_3d_fullres.yaml` |
| M0（Residual-Encoder v2.3，正式） | `m0_resenc_picai_3d_fullres_v23.yaml` |
| M1 / M2 / M3 / M4（v2.4 融合） | `m1_equal_…`、`m2_image_gate_…`、`m3_zone_input_…`、`m4_conditioned_gate_…` |

其他有用开关：`--device cpu|cuda`、`--epochs`、`--resume-from <ckpt>`、`--run-name`、`--no-progress`。

---

## 5. 记录与分析

| 目的 | 做法 |
|---|---|
| **记录** | 每次运行后往 `docs/experiment_log.md` 追加一节：命令、时间、结果、未做事项（一行也算） |
| 看 N0 训练进度 | `tail -f <training_log_*.txt>`；对比 `progress.png` |
| M0 架构/参数量统计 | `python scripts/train/summarize_m0_resenc_architecture.py` |
| M0 显存实测 | `--overfit` 产出的 `outputs/diagnostics/<model>/<run>/diagnostics/gpu_memory_profile.json` |
| **评测 nnU-Net validation** | `python scripts/evaluate/evaluate_n0_validation.py` —— 默认评测 fold 0 的 223 例；输出 `outputs/metrics/<name>_<时间戳>/`（Dice、检出、患者级混淆矩阵、基于概率分数的代理 AP / AUC），并会明确标出哪些是代理指标 |
| **官方论文指标（lesion-level AP / PI-CAUC）** | 还差两步：① `picai_eval` **本环境未安装**（需先安装）；② 评测口径须按 `docs/protocols/G0_SAP.md` 先冻结 —— 在此之前只能报**代理指标**，不得写成官方口径 |

---

## 6. 常见问题

| 问题 | 答案 |
|---|---|
| 启动训练需要谁批准吗？ | **不需要。** 任何前置都不阻断启动 |
| 为什么日志里 `Pseudo dice` 一直很低？ | 它只是 online patch 指标，尤其在前 100 epoch 常接近 0；**不要**据此换 checkpoint |
| 用哪个 checkpoint 做推理评测？ | 固定 `checkpoint_final.pth`；**不要**用 `--val_best` |
| 能删旧的失败 run 吗？ | **不能**（`nnUNetTrainer__…`、`nnUNetTrainerPICAI_FLCE__…` 必须原样保留） |
| 新 run 会覆盖旧 run 吗？ | 不会：输出目录由 trainer 类名隔离 |

---

## 7. 那些 "G0/门/前置" 是什么（不看也行）

- `G0-*` 是**数据与协议层面的标签门**，不产出模型，**也不阻断任何训练**；
- 它们只影响一件事：结果能不能被写成"正式 / 可引用"。未冻结期间产出的结果身份 = **feasibility（可行性）**；
- 现状：G0-P 已通过；G0-R / G0-E / G0-SAP 未完成；
- 需要时再查：`docs/STATUS.md`（当前状态唯一权威）｜`docs/GLOSSARY.md`（编号释义）｜`docs/protocols/`（协议正文）。

---

## 8. 文档地图（日常只看前 3 个）

| 文件 | 作用 |
|---|---|
| `docs/START_HERE.md` | 本页：日常入口 |
| `docs/experiment_log.md` | 实际跑过什么、结果如何 |
| `docs/STATUS.md` | 当前状态与问题清单 |
| `docs/runbooks/*.md` | 各流程的详细命令与判据 |
| `docs/Development_Log.md` | 代码/配置/协议改了什么 |
| `docs/protocol_changelog.md` | 协议版本变更史 |
| `docs/research_plan.md` | 研究协议正文（**冻结版，不要改**） |
| `docs/GLOSSARY.md` · `docs/project_structure.md` | 编号释义 · 文件清单 |
| `docs/protocols/` · `docs/literature/` · `docs/figures/` | 协议草案 · 文献检索 · 图 |
| `docs/P2_M0_*.md` · `docs/N0_*.md` · `docs/Data_Analysis.md` · `docs/PICAI_Model_Readiness_Audit.md` | 历史报告与审计（存档性质，按需查） |
