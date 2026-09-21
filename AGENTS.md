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

## 5. nnU-Net-first

- 项目采用 **nnU-Net-first**：planning/preprocessing、dataloader、patch sampling、augmentation、
  deep supervision、optimizer/调度、training loop、checkpoint/resume、validation、sliding-window
  inference、prediction export 全部由固定版本 nnU-Net（`third_party/nnUNet`，v2.6.2，只读）负责。
- **不重复实现** nnU-Net 已有功能：不得再写第二套 trainer、checkpoint、滑窗推理、验证器、patch
  sampler、optimizer 或学习率调度器，也不得复制 nnU-Net 的 U-Net encoder/decoder。
- 项目扩展只允许：PI-CAI Focal+CE 损失、NoFFT blur 修复、基于原生网络的轻量 reliability gate、
  必要的 PZ/TZ 输入适配、统一训练入口、最少量的数据准备代码。
- 禁止直接修改 `third_party/nnUNet`。所有训练从 `scripts/train/train_nnunet.py` 进入。

## 6. 文档纪律

- 活跃文档只有四份：`README.md`、`docs/Research_Plan.md`、`docs/Development_Log.md`、
  `docs/Training_Log.md`。
- 不再建立阶段门（G0/G1/G2/SAP/P2A/P2B 等）或步骤/runbook/readiness/gate 文档，也不为每个模型
  维护大 YAML；网络差异由 Trainer 类名与少量代码常量表达，结构参数继续来自 `nnUNetPlans.json`。
- `docs/Research_Plan.md` 只记录研究问题、科学假设、研究内容、方法机制与研究边界，不写实验流程、
  数据划分、训练参数、评价指标、运行命令、阶段门或实施步骤。
- 代码/架构变更记入 `docs/Development_Log.md`；训练与验证的运行事实记入 `docs/Training_Log.md`。
