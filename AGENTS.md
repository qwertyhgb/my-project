# 项目协作规则

本文件适用于 `/opt/data/private/lm/my-projects` 及其全部子目录。除用户在当前对话中明确覆盖外，
所有协作代理必须遵守以下规则。

## 1. 运行环境

- 一律使用 conda 环境 **`lm`**（Python：`/root/anaconda3/envs/lm/bin/python`），禁止 `base` 或其他环境；
  执行前先 `conda activate lm`。
- 涉及 nnU-Net 的任何命令前必须先：
  ```bash
  cd /opt/data/private/lm/my-projects
  conda activate lm
  source scripts/env_nnunet.sh
  ```
  该脚本校验 `lm` 环境与固定版本 nnU-Net 源码，并把 `nnUNet_*` 指向项目内 `workdir/`、`outputs/`。
- 缺失的 Python 包由代理自行安装到 `lm`（`python -m pip install <pkg>`），安装后验证导入与版本；
  未经用户确认不得升降级可能破坏依赖的包（`nnunetv2`、`torch`、`numpy`、`SimpleITK` 等）。

## 2. 数据安全

- 原始医学数据保持**只读**。
- 不删除、不覆盖 `data/`、`workdir/`、`outputs/` 及任何既有 checkpoint（含 N0 baseline 与 legacy M0 产物）。
- 不静默删除、覆盖或排除病例；派生数据写入项目约定目录。

## 3. 长任务由用户运行

- 数据处理、planning/preprocessing、数据集物化、训练、验证（validation/verify）、推理、评测、
  批量分析等任何预计耗时较长的命令，一律整理为**可直接复制**的命令交由用户运行，代理不得自行启动。
- 代理可以阅读文件、检查代码、修改脚本/文档，并执行轻量、非破坏、非数据处理的秒级诊断命令
  （读取配置、解析 plan/split、`--help`、静态/语法检查、不读取真实医学数据的合成单元测试）。
- 不得因命令规模较小或使用 `--limit`、smoke test、单病例、`--epochs 1` 等参数而自行执行数据处理；
  除非用户在当前对话中明确授权该具体命令。命令归属不明确时，默认「交给用户运行」。
- 交付命令时注明工作目录、conda 环境（`lm`）、必要的 `source scripts/env_nnunet.sh`、预期输出、
  进度观察方式与成功判据。用户返回日志/产物后，代理负责检查结果、定位问题并给出下一条命令。

## 4. 长任务必须有进度

- 数据清点、转换、重采样、预处理、审计、推理、评估、训练必须默认显示可观察进度（优先 `tqdm`，
  至少含 当前/总量/比例/速率/预计剩余）；训练显示 epoch 进度，分布式只由主进程显示。
- 进度条不替代结构化日志：任务结束仍须输出 成功数/失败数/跳过数/耗时/输出路径。
- 新增或修改相关脚本时，检查进度是否覆盖主要耗时循环；默认开启，可提供 `--no-progress`。

## 5. nnU-Net-centered 与受控扩展

- 项目以固定版本 nnU-Net（`third_party/nnUNet`，v2.6.2，只读）为中心：默认优先复用其
  planning/preprocessing、网络 plans、训练主循环、optimizer、LR scheduler、deep
  supervision、checkpoint/resume、validation、sliding-window inference 与 prediction
  export。
- 当 nnU-Net 原生机制不能满足**明确的研究假设**时，允许在项目代码中实现受控扩展，
  包括：病例级采样概率、阳性/阴性病例平衡、lesion-centred patch sampling、
  boundary-aware 或 hard-example patch sampling，以及与研究问题直接相关的输入、
  损失或轻量网络模块。
- 允许项目侧第二套 dataloader/patch sampler，但必须同时满足：
  - 不修改 `third_party/nnUNet`（仍绝对禁止）；
  - 优先继承或包装原生 `nnUNetDataLoader`；
  - 复用 nnU-Net 预处理数据与 `class_locations`（不自行重采样原始医学图像、
    不读取原始 NIfTI）；
  - 不重新实现训练循环、optimizer、LR scheduler、checkpoint、验证器或滑窗推理；
  - 对缺失 metadata、无阳性病例、非法病例 ID 一律 fail-closed，不得静默当阴性；
  - 训练采样与验证采样明确分离，验证集默认保持 nnU-Net 原生采样行为（无验证泄漏）；
  - 有纯合成测试覆盖采样比例、前景保证与数据安全；
  - 使用不同 Trainer 类名与独立输出目录，不覆盖既有产物。
- 每个自定义采样策略必须对应一个明确研究假设；不得为提高单次结果同时混入多个
  不可归因的修改（如同时改采样、损失、patch size 与优化器）。
- 仍禁止复制 nnU-Net 的完整训练器、U-Net encoder/decoder、验证器和推理器。
- 所有训练从 `scripts/train/train_nnunet.py` 进入。
- **当前研究主线**（2026-10-08 本轮明确更新）：预测解剖与粗病灶定位共同条件化局部多模态融合。
  活跃扩展只包含 Stage-1 先验生成器、轻量独立 shallow stems + neutral projection、
  pre-fusion coarse lesionness、soft anatomy/lesion-conditioned residual fusion。
  禁 hard gating；neutral path 必须保留，residual末层零初始化。ROI与原logits refinement保留为
  SUPPORTING ablation；hard negatives待best(C,D)冻结后加入。旧gate/fusion不得恢复为主线。
  新融合实现不依赖legacy；不同loss家族使用明确Trainer类名，所有既有类名继续可解析。
  predicted-prior Dataset608与历史Dataset606隔离，记录IN_SAMPLE/OOF/HELD_OUT/EXTERNAL模式、
  checkpoint、split与geometry。不得把in-sample写成OOF或把hard export的WG失败当soft head失败。
  一次只改一个明确机制，边界见 docs/Method.md §7。
  `src/zonal_reliability_fusion/legacy/`仍只读，不扩张、不改变其实现。
- anatomy 相关操作的硬约束：anatomy model 不读 lesion GT；validation / test 的解剖先验必须来自
  该病例自身 MRI 的预测；**禁止**把 GT WG/PZ/TZ 作为 lesion model 的推理输入（GT 仅允许用于
  显式标记 `ORACLE_GT` 的上界分析）。split 泄漏由 `scripts/data/check_split_integrity.py`
  自动检查，遇泄漏 fail closed。

## 6. 文档纪律

- **治理文档八份**（2026-10-08 起）：`README.md`、`docs/Research_Plan.md`、`docs/Method.md`、
  `docs/Experiment_Plan.md`、`docs/Evaluation_Protocol.md`、`docs/Training_Log.md`、
  `docs/Findings.md`（横向汇总）、`docs/Development_Log.md`。各文件**单一职责**，禁止互相抄写：

  | 文档 | 只记录什么 | 不记录什么 |
  |---|---|---|
  | `README.md` | 当前项目状态与快速入口（唯一主线、pipeline、状态表、命令、目录结构） | 旧研究线介绍（最多一段 + 链接 archive） |
  | `docs/Research_Plan.md` | 研究问题、科学假设、方法机制、研究边界、预期贡献、限制、预先定义的报告指标分级 | 具体实验流程 / 数据划分 / 训练参数 / 运行命令 / 已发生的结果 |
  | `docs/Method.md` | 方法机制与实现边界（含每个模块的 fail-closed 清单） | 结果、指标定义、实验矩阵 |
  | `docs/Experiment_Plan.md` | 实验矩阵、预算、命令模板、stop rules 的运行判据 | 科学假设的论证、指标定义、已发生的结果 |
  | `docs/Evaluation_Protocol.md` | 指标定义、空值语义、匹配协议、统计口径、冻结常量（**唯一来源**） | 实验结果、方法论证 |
  | `docs/Training_Log.md` | 只记录**真实发生**的训练/验证（命令、时间、状态、结果、产物路径） | 协议叙述、跨 run 结论 |
  | `docs/Findings.md` | 只记录**已有证据支持**的结论 + 证据边界（Preliminary Findings / Mainline Findings 分开） | 单实验详情、尚未验证的推断 |
  | `docs/Development_Log.md` | 实质性代码/架构变更 | 训练事实 |

- **实验详情文档**位于 `docs/experiments/`，**每个实验一份**，文件名即 variant 名
  （如 `positive_sampling.md`、`lesion_roi.md`）。每份只记录该实验自身：标识、一次性配置、
  运行事实、训练动态、验证结果、产物、观察与限制、待办、后续记录模板。
  - 各实验文档**互相独立**：不写跨实验对比、不互相引用；要对比由读者自行并列阅读。
  - 分工：治理文档记录跨实验的事实与索引，实验文档记录单实验详情。
- **归档**：旧研究路线一律进入 `docs/archive/`（`old_research_plans/`、
  `legacy_gate_research/`、`historical_experiments/`），并带归档头（"已被 X 取代"）。归档材料
  不得被继续扩张；主 README / Research Plan / Experiment Plan 不得介绍旧模块。
  `src/zonal_reliability_fusion/legacy/` 下的代码**只读**：类名与实现逐字保留，仅供 checkpoint
  兼容与历史复现。
- 不再建立阶段门（G0/G1/G2/SAP/P2A/P2B 等）或步骤/runbook/readiness/gate 文档，也不为每个模型
  维护大 YAML；网络差异由 Trainer 类名与少量代码常量表达，结构参数继续来自 `nnUNetPlans.json`。
- **Trainer 状态分层**：`nnunet/trainers.py` 的 `ACTIVE_TRAINERS` / `SUPPORTING_TRAINERS` / `LEGACY_TRAINERS` 是
  Trainer 分层的唯一真源；训练入口默认只显示 ACTIVE，supporting条件需`--supporting`，归档条件需 `--legacy`。
  `PROJECT_TRAINERS` 必须包含**全部历史类名**（输出目录由类名决定，缺一个即让历史产物失联）。
- **单一状态真源**：variant 状态只在 `docs/Training_Log.md` 的「各 variant 状态」表维护；README 与
  Findings 引用它，不另立清单。
- **新模块进入主模型的门槛**：必须能回答"是否减少漏检 / 是否改善覆盖 / 是否减少假阳"三者之一
  （见 `docs/Research_Plan.md` §15 与 `docs/Experiment_Plan.md` §9 的 stop rules）。
- **表述纪律**：CI 跨 0 **不得**被写成"无差异/等效"，也**不得**写成"证明无效"；单次运行不得声明
  run-to-run 方差；`ORACLE_GT` 结果不得作为正式性能；predicted anatomy 与 GT anatomy 不得混淆。
- 代码/架构变更记入 `docs/Development_Log.md`；训练与验证的运行事实记入 `docs/Training_Log.md`。

## 7. 文件创建权限

- **只有用户的明确命令才能创建新文件。** 代理不得自行、顺手或"顺便"创建任何文件，包括但不限于：
  文档、说明、总结、报告、README、配置、YAML、脚本、测试、示例、模板、临时笔记。
- 允许创建的唯一情形：用户在当前对话中**明确指示或明确授权**创建该文件（含内容与位置）。
  用户要求实现某个功能时，只有该功能**必需**、且无法通过修改既有文件达成的文件才可以创建；
  能改既有文件的一律改既有文件。
- 新建文件的理由成立但用户未明确要求时，**先说明理由并征得同意**，不得先建后报。
- 新建文件前先确认它不落在 `data/`、`workdir/`、`outputs/`、`third_party/` 内（见 §2）。
- 本规则与 §6 一致：`docs/experiments/` 下的实验文档同样只能由用户命令创建。
