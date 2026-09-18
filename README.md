# Anatomy-Conditioned Multi-sequence Prostate Lesion Segmentation

面向前列腺癌病灶分割的研究项目。当前主线为**解剖分区（PZ/TZ）条件自适应的 T2W/ADC/HBV 多序列融合**
（研究计划 **v2.3.1**，已取消「跨中心分割/域泛化」主线；v2.3.1 含 G0-SAP 三阶段生命周期等四项 amendment，
见 `docs/protocol_changelog.md`）。

## 文档地图（每类事实只有一个权威来源）

| 文档 | 职责（唯一权威来源） |
|---|---|
| `docs/research_plan.md` | **研究协议**：假设、数据与独立测试设计、模型与消融、冻结训练/评价/统计协议、阶段门定义、风险与失败转向、待冻结决策表 |
| `docs/STATUS.md` | **当前状态**：每个门的实时状态、证据链接、blocker、下一步 |
| `docs/GLOSSARY.md` | **标号与步骤总览**：G0-*/G1–G5/P0–P10/M0–M4/N0/H0–H4/D1–D10/B1–B10/SAP-A/B/C 各代表什么、推进顺序与门控依赖 |
| `docs/protocol_changelog.md` | **协议版本变更史**（含修改原因与对既有结果的影响） |
| `docs/experiment_log.md` | **实际运行记录**：命令、环境、耗时、结果、失败、产物 |
| `docs/Development_Log.md` | **代码/配置/协议实现变更** |
| `docs/runbooks/` | **由研究者执行的命令**、输出路径、进度观察与成功判据 |
| `docs/protocols/` + `configs/protocols/` | G0-R / G0-E / G0-SAP 的详细协议与机器可读载体 |
| `docs/literature/` | 参考资料总表与文献/规范核查记录 |
| `AGENTS.md` | 协作代理权限、conda 环境与进度规则 |

## 目录

- `configs/`：数据、模型、训练、实验与**协议（`configs/protocols/`，G0-R/G0-E/G0-SAP 机器可读草案）**配置
- `data/`：原始数据、处理中间结果、预处理数据、划分与元数据
- `docs/`：研究协议、状态看板、变更史、运行手册、实验/开发日志、文献与协议草案
- `src/zonal_reliability_fusion/`：项目核心 Python 包
- `scripts/`：数据准备、训练、评测和审计入口
- `tests/`：单元测试与集成测试
- `third_party/`：只读第三方参考代码，包括 nnU-Net
- `outputs/`：模型、预测、指标、图表和实验报告
- `logs/`：运行日志
- `workdir/`：可删除的预处理缓存和临时文件

## 当前状态

**唯一权威来源：`docs/STATUS.md`**（各门状态、证据链接、blocker、下一步）。本文件不重复维护任何动态状态
（是否已运行、缺多少字段、测试通过数等一律以 `docs/STATUS.md` 与 `docs/experiment_log.md` 为准）。

**状态纪律（不可协商）**：代码迁移、合成测试通过、工具实现或工具运行成功，**都不使任何门变为 PASS**；
未通过相应门之前，禁止启动 P2B、正式 M0 或 N0。

## G0 工具（能力清单）

> 各工具的**实现/执行状态**见 `docs/STATUS.md`；命令与成功判据见 `docs/runbooks/`。

- G0-R 抽样与盲审清单工具 `scripts/audit/audit_picai_alignment_qc.py`（确定性分层抽样 → 冻结的抽样清单）；
- G0-R 人工路径工具 `scripts/audit/render_picai_alignment_qc.py`（盲审图 + landmark **LPS 物理位移**；
  不含病灶标签或模型输出）——已降级为历史草案与后备；
- **G0-R 自动路径（当前主路径，协议 `draft-0.3`；draft-0.2 配置已归档于 `configs/protocols/archive/`）**：
  `scripts/audit/run_picai_alignment_qc_automated.py` +
  `src/.../protocols/g0_r_automated_qc.py`（协议 `docs/protocols/G0_R_ALIGNMENT_QC_AUTOMATED.md`，配置
  `configs/protocols/g0_r_alignment_qc_automated.yaml`）。四组自动证据：A 物理几何/FOV、B 多尺度跨模态边缘
  一致性（mm、双向对称、多容忍半径 F1）、C 诊断性刚体残余估计（仅内存、不写回）、D 合成已知位移校准
  （0–4 mm × 三方向 × 固定 seed，自动推导阈值）；只输出 `resample-only` / `resample+rigid` /
  `INSUFFICIENT_EVIDENCE` **候选**。**零人工阅片/landmark/双阅片/人工阈值**；不联网、不写回影像；
- G0-E 候选清点工具 `scripts/audit/audit_g0e_candidates.py`（文件清单级清点 + 证据缺口清单，
  缺证据一律 `UNKNOWN`，不代填 `verdict`）；
- G0-SAP 冻结载体工具 `scripts/evaluate/prepare_g0_sap_freeze.py`（硬约束与 SAP-A/B/C 阶段校验 +
  `--check-freeze` 就绪检查 + SAP-A 方法块哈希/差异审计 + 模板与哈希生成；不运行评测——SAP-B 结果只写入
  独立的 `sap_b_results` 块）；
- 共享校验层 `src/zonal_reliability_fusion/protocols/`。**工具不决定任何门的通过；图像 QC 不做配准**；
  真实数据运行由研究者执行（协议文档 + `docs/runbooks/`）。

## N0 = PI-CAI official-style nnU-Net v2 Focal+CE NoFFT

- 定义与术语边界见 `docs/N0_PICAI_Official_Baseline.md`。nnU-Net v2 默认的 Dice+CE 方案仅作为**已排除的预实验**
  保留（运行产物、日志与 checkpoint **不得删除或覆盖**，不得写成正式失败实验，见 `docs/experiment_log.md`）；
- 正式 N0 的启动条件与唯一训练命令见 `docs/runbooks/n0_training.md`：只有在 G0-R 冻结且确认沿用当前预处理、
  G0-E 已冻结独立测试路径、且预算与 checkpoint 规则冻结后，才可记为正式 N0；
  更早运行只能标记为 feasibility run（运行状态见 `docs/STATUS.md`）；
- 注意：本项目是 **v2 port**，不是逐位官方复现（官方为 nnU-Net v1、1295 标注扫描、5-fold CV）。

## v2.3 正式 M0（plan-driven Residual-Encoder 3D U-Net）

- v2.2 PlainConv M0 已完成 Focal+CE 损失迁移与低频验证基础设施，自 v2.3 起仅为 **legacy/feasibility**
  （旧配置与旧训练命令不得用于 v2.3 正式 G2）；
- **v2.3 正式 M0 = plan-driven Residual-Encoder 3D U-Net**：冻结身份为版本化架构配置
  `configs/architectures/m0_resenc_v23.yaml`（blocks_per_stage=(1,3,4,6,6,6,6)）、稳定架构哈希 `8bb5127e…`、
  参数量 148,193,644、解析 MACs 1.108 T/patch、§8.6 浅层 skip 接口契约与 checkpoint/resume 架构隔离；
  实验配置 `configs/experiments/m0_resenc_picai_3d_fullres_v23.yaml`（独立 run/输出身份）；
  P2A 实现记录见 `docs/P2_M0_ResidualEncoder_v23.md`；
- **G2 通过前不得启动 M0–M4 正式训练**；G2 前置/后置验收清单见 `docs/research_plan.md` §9.4.2，
  完成情况与峰值显存测量状态见 `docs/STATUS.md`；
- 实现细节见 `docs/P2_M0_ResidualEncoder_v23.md`；协议与 G2 要求见 `docs/research_plan.md` §7/§9.4.2/§14；
  P2B 命令（含 1-step 峰值显存实测）见 `docs/runbooks/p2b_m0_validation.md`。

## M1–M4 融合模型（v2.4）与 PZ/TZ prior

- **M1–M4 代码已实现**：`src/zonal_reliability_fusion/models/fusion_blocks.py`（`Q_l` 浅层 skip 聚合、
  `F_fuse` stage-2 融合、image gate、zone encoder、条件仿射头）、`fusion_multi_branch.py`；
  `build_model` 按 `experiment.model_id` 严格分派，**未知 model_id 报错、不回退到 M0**；
  4 个 v2.4 实验配置与互异输出目录：`configs/experiments/{m1_equal,m2_image_gate,m3_zone_input,
  m4_conditioned_gate}_picai_3d_fullres_v24.yaml`（M3/M4 绑定 `zonal_prior_source=yuan` 与
  `data/processed/picai_zonal_yuan_v1`）；
- **训练状态**：M1–M4 均**未训练**（也未读真实 prior）。实现顺序按研究者工程指令早于 G2，已登记于
  `docs/protocol_changelog.md`；**正式训练仍受 G2 约束**（`docs/research_plan.md` §9.4.2 / §14）。
  门/实现状态见 `docs/STATUS.md` §2；
- **PZ/TZ prior 物化**（M3/M4 的前置数据）：`docs/runbooks/p2a_zonal_prior_materialization.md`；
  产物为 `data/processed/picai_zonal_<source>_v1/<case>.npz + <case>.json + manifest.json`；
  读取端 fail-closed（source / configuration / plans_sha256 / 逐例 input+output+metadata 哈希 / 数值契约
  全部逐例校验；预检成功后冻结逐例记录，防止训练期间被替换）；
- **seg 标签存储态**：`.b2nd` 中 seg 可含 nnU-Net 填充哨兵 `-1`，读取适配层按官方
  `RemoveLabelTansform(-1, 0)` 的语义映射为 0，返回严格 `uint8 ⊆ {0,1}`；
- **运行时峰值显存**：`--overfit` 写
  `outputs/diagnostics/<model_id>/<run_name>/diagnostics/gpu_memory_profile.json`
  （`process_memory.*` 为当前进程、peak 覆盖整个 `trainer_run`；`gpu.*` 为整卡状态；CPU 路径
  `NOT_APPLICABLE`）。静态报告 `models/profiling.py` 的 `peak_gpu_memory=NOT_MEASURED` 语义不变，
  两者互不替代。
