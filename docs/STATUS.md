# 项目状态看板（唯一权威来源）

> **本文件是「当前状态」的唯一权威来源。**
> `docs/research_plan.md` 只定义研究协议（做什么、怎么判定、失败如何转向），**不再维护任何实时状态**；
> `docs/experiment_log.md` 记录实际执行过的命令与结果；`docs/Development_Log.md` 记录代码/配置/协议实现变更；
> `docs/protocol_changelog.md` 记录协议版本变更史。
>
> **状态纪律（不可协商）**：
> 1. 代码迁移、合成测试通过、工具实现完成或工具运行成功，**都不等于**任何科学阶段门 PASS；
> 2. validation 上的结果不得包装为独立确认结果；工程完成度不等于科学阶段门通过；
> 3. gate 权重不得解释为因果贡献；
> 4. 本文件只陈述**有产物或日志证据**的事实；缺证据一律写 `UNKNOWN / 需人工确认`，不得用假设填充。

| 项 | 值 |
|---|---|
| 最后更新 | 2026-09-18（**版本控制已启用：Git 基线 commit `23ca6cb`，B1 关闭**；PZ/TZ 链路 6 轮加固 + seg 哨兵兼容 + `--overfit` 运行时显存遥测；研究者已执行：3 例 prior dry-run + 全量 1500 例 prior 物化（B10 关闭）；**G0-R Automated：draft-0.2 真实运行失败（见 B2）→ draft-0.3 修复 → 审查发现两项 fail-open → draft-0.4 修复已实现并完成合成测试（真实重跑仍未执行）**；阶段门状态未变：仍无任何正式模型结果） |
| 当前研究计划版本 | **v2.3.1**（含四项实质性新增条款，见 `docs/protocol_changelog.md`「v2.3.1」；本文件不重复其内容） |
| 当前阶段 | G0-P = PASS；G0-R / G0-E / G0-SAP = PENDING（DRAFT，未冻结）；G1 未启动；G2 = NOT YET EVALUATED |
| 当前工程焦点 | M3/M4 的 plan-space PZ/TZ prior **工程产物已就绪**（1500 例全量物化完成，2026-09-17；B10 关闭）→ **真实 loader-smoke 复跑（含当前 reader 的逐例数组级复校验）** → M0/M4 显存实测 |

> **状态唯一来源声明**：本文件是「当前状态/动态事实」的唯一权威来源。`docs/research_plan.md`、`docs/protocols/`
> 只写协议状态（`DRAFT`/`FROZEN`）与要求；`docs/runbooks/` 只写前置条件与命令；
> 运行事实（命令、耗时、结果、产物）写 `docs/experiment_log.md`；代码/配置变更写 `docs/Development_Log.md`。
> 协议版本变更与 amendment 写 `docs/protocol_changelog.md`。**任何动态数字（测试通过数、缺失字段数、是否运行过）
> 只允许出现在本文件与实验日志中。**
>
> 编号含义（`G0-*`、`G1`–`G5`、`P0`–`P10`、`M0`–`M4`、`N0`、`H0`–`H4`、`D1`–`D10`、`B1`–`B9`、`SAP-A/B/C`）
> 见 `docs/GLOSSARY.md`；本文件只写状态，不重复解释编号。

---

## 1. 阶段门状态

| 门 | 状态 | 判定依据 / 证据（权威文件） | 阻塞项 |
|---|---|---|---|
| **G0-P**（PI-CAI 数据可用） | **PASS**（2026-09-14） | `docs/PICAI_Model_Readiness_Audit.md` §11 | — |
| **G0-R**（序列错位协议冻结） | **PENDING / DRAFT**（门状态**未变**） | 抽样已执行：`outputs/diagnostics/g0_r/20260915_093819/`（`decision=null`、`status=PENDING`）；**draft-0.2 真实运行已执行但失败**：`outputs/diagnostics/g0_r_automated/20260918_074219/`（16 例 × 2 对，1072.7 s，32/32 `REGISTRATION_DIAGNOSTIC_FAILED`、`calibration_passed=false`、候选 `INSUFFICIENT_EVIDENCE`；产物原样保留）；**draft-0.4 修复已实现 + 合成测试通过**（`tests/unit/test_g0_r_automated_qc.py` 52 passed；修复两项 fail-open：指标 `sigma*.usable` 守卫、校准 unit 条件网格完整性；协议 `protocol.version=draft-0.4`、`rule_version=0.4`、`output_schema_version=g0-r-automated/0.4`、`protocol_hash=13e7be50…`；draft-0.2/0.3 配置均按字节归档；`--dry-run` 已通过且不创建输出目录；见 `docs/experiment_log.md`、`docs/protocol_changelog.md`） | **draft-0.4 真实 16 例重跑未执行** → `frozen_decision` 未填写；命令见 `docs/runbooks/g0_r_alignment_qc.md`（先 `--dry-run`） |
| **G0-E**（独立测试路径冻结） | **PENDING / DRAFT** | 文件清单级清点已执行：`outputs/diagnostics/g0_e/20260915_093835/`（`run_metadata.json` 记录 `missing_evidence_items=67`；`candidate_registry.csv` 的 `verdict` 为空） | 许可、标签语义、患者重叠、模态映射、先验独立性全部 `UNKNOWN`；路径 A/B/C 未冻结 |
| **G0-SAP**（评测与统计协议冻结） | **PENDING / DRAFT**（协议 `draft-0.3`，v2.3.1 三阶段） | 冻结载体工具与校验器已实现（含 `sap_b_results` 结果块、SAP-A 方法块哈希、阶段/可见性不可变式、MCID 冻结必需字段）；实测 `--check-freeze` = `NOT READY`（**27 个字段待填**，`power_memo.status=INSUFFICIENT_DATA`）；**子阶段 SAP-A / SAP-B / SAP-C 均未开始** | 方法与候选集未冻结（SAP-A）；阈值/后处理数值待 SAP-B 写入 `sap_b_results`；`scripts/evaluate/` 评测逻辑**未实现** |
| **G1**（官方基线可复现） | **未启动**（正式训练未开始） | `outputs/nnUNet_results/Dataset605_PICAI/` 下只有 `nnUNetTrainer__nnUNetPlans__3d_fullres`（已排除的默认 Dice+CE 预实验，跑至 epoch 43 被中断、无 `checkpoint_final.pth`）与 `nnUNetTrainerPICAI_FLCE__nnUNetPlans__3d_fullres`（首次 FFT 崩溃 run：`debug.json` 记录 `current_epoch=0`）；**NoFFT 修复版训练器已实现并合成测试覆盖**（`src/.../integrations/nnunet_picai_flce.py` 的 `nnUNetTrainerPICAI_FLCE_NoFFT`），但**从未运行**（NoFFT 输出目录尚不存在） | 须 G0-R PASS + G0-E 路径冻结 + N0 预算/checkpoint 规则冻结；命令见 `docs/runbooks/n0_training.md` |
| **G2**（自研基线可信） | **NOT YET EVALUATED** | G2 前置 1–7 完成（`docs/P2_M0_ResidualEncoder_v23.md` §9）；第 8 项（真实数据 loader smoke / small-overfit / 诊断性首次全体积验证）未运行 | 须 G0-R 冻结预处理身份；命令草案见 `docs/runbooks/p2b_m0_validation.md` |
| **G3 / G4 / G5** | **未启动** | — | 依赖 G2、G0-SAP、G0-E；H3 的独立确认只能来自 G5 |

> 每个门的**定义、通过条件与失败转向**见 `docs/research_plan.md` §14；状态只在本文件维护。

## 2. 实现与工程状态

| 项 | 状态 | 证据 |
|---|---|---|
| v2.3 正式 M0（plan-driven Residual-Encoder 3D U-Net） | 已实现（P2A）+ 合成回归通过 | `docs/P2_M0_ResidualEncoder_v23.md`；架构哈希 `8bb5127e5c9a99ca1a0631cb365cdcf0b866441ac225911e17faa6bc995bbee5`；参数量 148,193,644；MACs 1,108,423,116,800/patch |
| M0 峰值显存 | **NOT_MEASURED**（仍无实测） | 仅有分析性激活下界代理（≈6.07 GB，fp32/batch=2）；**运行时遥测工具已实现**：`--overfit` 产出 `outputs/diagnostics/<model_id>/<run>/diagnostics/gpu_memory_profile.json`（含本进程 `peak_allocated/reserved`、相位、OOM/ERROR 落盘），命令见 `docs/runbooks/p2b_m0_validation.md` §1-E；实测待研究者执行（B5） |
| legacy PlainConv M0（v2.2） | legacy / feasibility，仅审计追溯 | `docs/P2_M0_Implementation.md`；旧配置与产物原位保留 |
| **M1–M4 融合模型**（`Q_l` 浅层 skip 聚合、`F_fuse` stage-2 融合、gate / 条件仿射） | **代码已实现 + 合成测试通过；未训练、未读真实 prior、未评测** | `src/.../models/fusion_blocks.py`、`fusion_multi_branch.py`（`build_model` 按 `model_id` 分派）；4 个 v2.4 实验配置 `configs/experiments/{m1_equal,m2_image_gate,m3_zone_input,m4_conditioned_gate}_picai_3d_fullres_v24.yaml`；架构哈希（v2.4）M1 `54b66e33…`、M2 `77d1de43…`、M3 `097c19ed…`、M4 `8a12527d…`；参数量 M1 155,367,756 / M2 155,417,679 / M3 155,551,343 / M4 155,531,797；测试 `tests/unit/test_fusion_models.py` |
| M1–M4 实现顺序与协议条款的关系 | **偏离已登记** | `docs/research_plan.md` §9.4.2 规定 M1–M4 实现位于 G2 之后；本轮按研究者工程指令**提前实现**（只写代码 + 合成测试，不读真实数据、不启动训练）。**训练仍受 G2 约束**（未通过 G2 不得启动 M0–M4 正式训练）；记录见 `docs/protocol_changelog.md` 2026-09-17/18 行 |
| **PZ/TZ prior 物化与深度校验链** | 代码完成 + 合成测试通过；真实 3 例 dry-run 已通过；**全量 1500 例物化已完成（2026-09-17 10:13，退出码 0；`n_ok=1500 failed=0`，canonical manifest 已发布，见 `docs/experiment_log.md`）**；产物尚未经当前 reader 逐例数组级复校验 | `src/.../data/zonal_prior.py`、`preprocessed_store.py`、`scripts/data/materialize_zonal_prior.py`；成对原子提交 + 失败回滚状态机、`validate_sidecar` 单一严格校验器、canonical manifest 发布门控、逐例冻结记录（`zonal_prior_frozen_records`）、无弱缓存、tqdm 惰性进度；运行事实见 `docs/experiment_log.md`；命令见 `docs/runbooks/p2a_zonal_prior_materialization.md` |
| **seg 存储态契约（nnU-Net -1 哨兵）** | 已修复（`normalize_seg` 先验证后转换，`-1→0`，返回严格 uint8 `{0,1}`） | `src/.../data/preprocessed_store.py`；`SEG_STORAGE_LABELS=(-1,0,1)`、`SEG_SENTINEL_LABEL=-1`；对齐官方 `RemoveLabelTansform(-1,0)` 语义；测试 `tests/unit/test_seg_sentinel_mapping.py`（32 项）。真实 loader-smoke 复跑待执行 |
| **`--overfit` 运行时 GPU 显存遥测** | 已实现（mock CUDA 合成测试覆盖成功 / OOM / 普通异常 / CPU） | `src/.../training/gpu_memory_profile.py`；报告语义：`process_memory.*` = **当前进程**（peak 为整个 `trainer_run` 的 high-water mark），`gpu.*` = 整卡（`mem_get_info`）；失败阶段按真实 `trainer.current_phase` 细分；测试 `tests/unit/test_gpu_memory_profile.py`（23 项） |
| `scripts/evaluate/` 评测逻辑 | **未实现**（仅冻结载体工具） | `docs/protocols/G0_SAP.md` §9 |
| G0-SAP 生命周期校验（SAP-A/B/C + `sap_b_results` + 方法块哈希/差异审计） | 已实现，合成测试覆盖合法与非法组合 | `src/.../protocols/g0_sap.py`、`tests/unit/test_g0_sap_freeze.py`、`tests/unit/test_g0_protocol_common.py` |
| **G0-R Automated（自动对齐 QC）** | **draft-0.4 已实现 + 合成测试通过（52 项）；真实运行：仅 draft-0.2（失败：32/32 类型错误 + 校准不通过，产物保留）；draft-0.3/0.4 真实重跑未执行** | `src/.../protocols/g0_r_automated_qc.py`、`scripts/audit/run_picai_alignment_qc_automated.py`；配置 `configs/protocols/g0_r_alignment_qc_automated.yaml`（draft-0.3，旧版归档 `configs/protocols/archive/`）；协议 `docs/protocols/G0_R_ALIGNMENT_QC_AUTOMATED.md`；运行事实见 `docs/experiment_log.md` |
| **代码版本控制（Git）** | **已启用**（默认分支 `main`；基线 commit `23ca6cb`，2026-09-18，用户决定） | `.gitignore`（排除数据/产物/二进制；`third_party/nnUNet` 以 pin 记录）；无远端、未推送；`git status` 干净 |
| 全量单元测试（合成 CPU，不读真实医学数据） | **741 passed / 0 failed**（2026-09-18；演进：499（G0-SAP 闭环）→ 647 → 655 → 658 → 659 → 691 → 706 → 714 → 733（G0-R draft-0.3）→ 741（G0-R draft-0.4）） | `docs/experiment_log.md` 2026-09-17/18 各条目 |
| 全仓库 Ruff | 新增/重写文件 All checks passed；**全仓库仍有约 148 个既有告警**（legacy 文件） | `docs/Development_Log.md`「P2A 验收轮修复」 |
| 增强对齐 `augmentation.pending_parity` | **6 项未对齐**（rotation/scaling/低分辨率模拟/noise/blur/gamma） | `docs/P2_M0_Implementation.md` §18 |

## 3. 正式实验前的 Blocker

| # | Blocker | 影响 | 处理归属 |
|---|---|---|---|
| B1 | ~~主项目目录不是 Git 仓库~~ → **已关闭（2026-09-18，用户决定启用 Git）**：`git init`（默认分支 `main`）已完成，基线 commit `23ca6cb`（196 文件 / 6.8 MB；`.git` 3.1 MB）；`.gitignore` 排除 `data/{raw,interim,processed}/`、`workdir/`、`outputs/` 产物与二进制（`*.pth/*.pt/*.npz/*.npy/*.pkl/*.h5/*.nii*/*.b2nd/*.mha`）；`third_party/nnUNet` 以 provenance 记录（tag `v2.6.2` / commit `74ceb68`），不 vendored | 正式 run 现可记录 commit；`research_plan` §16 的「用户先决定版本控制方案」要求已满足（该文件按要求保持未修改，状态以本文件为准） | 后续每个正式 run 记录 `git rev-parse HEAD` 与 `git status --porcelain`（干净工作区） |
| B2 | G0-R 未冻结：draft-0.2 真实运行失败（工具缺陷）后经 draft-0.3 → **draft-0.4** 修复，但**真实 16 例重跑未执行**（draft-0.3 从未在真实数据上运行；draft-0.4 仅合成测试 + `--dry-run`） | 无法确认「仅重采样」是否足够；P2B / 正式 N0 / M0 不得启动 | 研究者执行 `docs/runbooks/g0_r_alignment_qc.md`（draft-0.4；先 `--dry-run`，输出目录为新 UTC 时间戳，不覆盖旧目录） |
| B3 | G0-E 未冻结：候选证据 67 项缺失，`verdict=null` | 无独立最终 test；H3 无法进入确认性判定 | 研究者执行 `docs/runbooks/g0_e_candidate_audit.md` |
| B4 | G0-SAP 的 SAP-A 未冻结：`--check-freeze` 报 27 个字段待填 + 评测逻辑未实现 | 不得查看任何正式 checkpoint 的论文指标 | 研究者执行 `docs/runbooks/g0_sap_freeze.md`（SAP-A → SAP-B → SAP-C） |
| B5 | M0/M1–M4 峰值显存未实测（**遥测工具已就绪**） | 可能超出 24 GB；影响 M0–M4 能否按冻结 plan 训练 | P2B 实测（`docs/runbooks/p2b_m0_validation.md` §1-E：`--overfit` 1 iteration + 1 validation → `gpu_memory_profile.json`） |
| B10 | ~~plan-space PZ/TZ prior 未物化~~ → **已关闭（2026-09-17；工程产物就绪）**：`data/processed/picai_zonal_yuan_v1/` 含 1500 `.npz` + 1500 病例 `.json` + canonical `manifest.json`（`n_cases=1500`、`manifest_sha256=b5d2b71d…`、`source=yuan`、`plans_sha256=5ca3c61b…`）；研究者执行 `--source yuan --resume`，`n_ok=1500 failed=0`，退出码 0（证据与环境见 `docs/experiment_log.md`） | 剩余（工程）工作：用当前 reader 对 1500 例做数组级复校验 + 真实 loader-smoke；**不改变任何科学阶段门** | 研究者执行 `docs/runbooks/p2b_m0_validation.md` §1-B（loader-smoke） |
| B6 | 增强 `pending_parity` 6 项未对齐 | M0–M4 与正式 N0 的增强不完全配平 | 待冻结决策表 D7（`docs/research_plan.md` §20.2） |
| B7 | N0（1000 epoch / PlainConv）与 M0–M4（200 epoch / Residual-Encoder）预算与骨干不配平 | N0−M0 差异不能作单一因果解释 | 待冻结决策表 D7 |
| B8 | Yuan23 / HeviAI23 训练数据与 PI-CAI/外部的重叠**未审计**（`prior_independence=null`） | M4 收益可能无法归因于解剖条件融合 | G0-E §7 审计（待冻结决策表 D8） |
| B9 | **G0-E 配置在清点运行后被修订**：`configs/protocols/g0_e_independent_test.yaml` 的 `config_sha256` 由清点时的 `90953cba3660f71858bbe19a7112574a7e380d1d9ace791635e3aa5f58cc16ce` 变为现状 `9960a4522eec2354c90736ca5ef705390ac6bde589226800fee83980e53d1595` | 若以「新配置」冻结，必须重新运行清点并记录新哈希；既有清点产物对**已记录内容**仍然有效 | 见 `docs/protocol_changelog.md`「2026-09-17」 |

## 4. 下一步（均由研究者执行；命令见 runbooks）

### 4.1 工程优先路径（当前实际推进顺序；不改变任何门的状态判定）

1. ~~**PZ/TZ prior 物化**（关闭 B10）~~：**✅ 已完成**——3 例 dry-run 通过，全量 1500 例物化于 2026-09-17 成功
   （`ok=1500 failed=0`，canonical manifest 已发布；运行记录见 `docs/experiment_log.md`，命令见
   `docs/runbooks/p2a_zonal_prior_materialization.md`）。若后续写入契约变更须重跑（`--resume` 只跳过严格校验通过的病例）
2. **真实 loader-smoke 复跑**（seg 哨兵映射 + prior 逐例数组级复校验）→ `docs/runbooks/p2b_m0_validation.md` §1-B
3. **峰值显存实测**（关闭 B5）：M0 与 M4 各 1 iteration + 1 validation（`--overfit`），读
   `diagnostics/gpu_memory_profile.json` 的 `process_memory.peak_allocated_bytes/peak_reserved_bytes`
   → `docs/runbooks/p2b_m0_validation.md` §1-E
4. **G2 / M0 三步真实数据验证** 与 **G1 / N0（NoFFT 训练器）**：前置条件与命令见 4.2 与 §1

### 4.2 协议门路径（状态判定仍按 `docs/research_plan.md` §14）

1. **G0-R**：先 `--dry-run`，再运行自动 QC（**draft-0.4**；旧失败运行目录不得覆盖），按 `qc_report.md` 冻结 `frozen_decision`
   → `docs/runbooks/g0_r_alignment_qc.md`
2. **G0-E**：以当前配置重新运行候选清点（绑定新哈希），逐项补齐证据，冻结路径 A/B/C
   → `docs/runbooks/g0_e_candidate_audit.md`
3. **G0-SAP**（三阶段，v2.3.1）：实现评测与统计脚本 → **SAP-A** 在看模型结果前冻结方法与候选集（含 MCID、
   功效备忘录）→ checkpoint 冻结后 **SAP-B** 在 PI-CAI validation 上一次性填数并做差异审计 →
   **SAP-C**（属 G5）封存哈希后对最终 test 一次性评估
   → `docs/runbooks/g0_sap_freeze.md`
4. **P2B / G2**：G0-R 冻结后执行新 M0 三步真实数据验证（含峰值显存实测）
   → `docs/runbooks/p2b_m0_validation.md`
5. **G1 / N0**：G0-R PASS + G0-E 冻结 + 预算/checkpoint 规则冻结后，启动唯一正式训练命令
   → `docs/runbooks/n0_training.md`
6. ~~**B1（Git）**：需用户先决定复现性方案~~ → **✅ 已完成（2026-09-18）**：Git 已启用（基线 commit `23ca6cb`）；正式 run 记录 `git rev-parse HEAD` 即可
7. **文献新颖性**：系统检索方法与截止日期尚未冻结（`docs/research_plan.md` §2.2 / 待冻结决策表 D10）
