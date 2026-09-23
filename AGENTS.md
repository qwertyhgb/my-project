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

## 6. 文档纪律

- **治理文档五份**：`README.md`、`docs/Research_Plan.md`、`docs/Development_Log.md`、
  `docs/Training_Log.md`、`docs/Findings.md`。
  - `docs/Findings.md` 只记录**跨实验**的观察、问题优先级与下一步判据，是所有实验记录的横向汇总；
    单实验详情仍分别写入 `docs/experiments/` 下各自的文件。
- **实验详情文档**位于 `docs/experiments/`，**每个实验一份**，文件名即 variant 名
  （如 `baseline.md`、`image_gate.md`）。每份只记录该实验自身：标识、一次性配置、运行事实、
  训练动态、验证结果、产物、观察与限制、待办、后续记录模板。
  - 各实验文档**互相独立**：不写跨实验对比、不互相引用；要对比由读者自行并列阅读。
  - 分工：治理文档记录跨实验的事实与索引，实验文档记录单实验详情。
- 不再建立阶段门（G0/G1/G2/SAP/P2A/P2B 等）或步骤/runbook/readiness/gate 文档，也不为每个模型
  维护大 YAML；网络差异由 Trainer 类名与少量代码常量表达，结构参数继续来自 `nnUNetPlans.json`。
- `docs/Research_Plan.md` 可以记录研究问题、科学假设、方法机制、研究边界，以及**预先定义的论文
  报告指标**；不记录具体实验流程、数据划分、训练参数、运行命令或已发生的实验结果。已发生的结果
  继续归 `Training_Log` / `Findings` / `experiments`。
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
