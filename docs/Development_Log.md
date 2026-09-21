# Development Log

简洁的代码/架构变更日志。只记录实质性结构变化，不保留旧阶段门（G0/G1/G2/SAP/P2A/P2B）历史。
训练与验证的运行事实见 `docs/Training_Log.md`。

---

## 2026-09-21 — nnU-Net-first 重构

把项目从「自写第二套训练框架 + 阶段门治理」彻底重构为 **nnU-Net-first**：通用机制全部交给
固定版本 nnU-Net（`third_party/nnUNet` v2.6.2，commit `74ceb68`，未修改），项目只保留研究必需
的最小扩展。

### 新增（唯一扩展点 `src/zonal_reliability_fusion/nnunet/`）

- `networks.py`：`SpatialModalityReliabilityGate`（两层 1x1x1 conv，末层零初始化）与
  `GatedNNUNet` 包装器。backbone 一律由 `get_network_from_plans` 构建（原生 `PlainConvUNet`），
  包装器代理 forward、暴露 `decoder` 属性使 `set_deep_supervision_enabled` 原样可用、透传 deep
  supervision 输出、strict state_dict、anatomy variant 严格校验 5 通道并对 PZ/TZ `clamp(0,1)`。
  权重定义 `scales = 3*softmax(logits)`，零初始化时 `scales≡1`，初始行为与原生 baseline 逐值一致。
- `transforms.py`：`MRIChannelRestrictedTransform` + `restrict_intensity_transforms_to_mri`，把
  强度增强限制在前 3 个 MRI 通道（PZ/TZ 豁免）；空间/镜像变换不包装，继续同步作用于全部通道。
- `trainers.py`：`PiCAIFocalCrossEntropyLoss`、NoFFT 增强修复 mixin、`nnUNetTrainerPICAI_FLCE_NoFFT`
  （baseline）、`nnUNetTrainerPICAI_ImageGate`、`nnUNetTrainerPICAI_AnatomyGate`。三者均继承
  `nnUNetTrainer`，复用同一 loss/optimizer/scheduler/epochs/采样/增强主体/checkpoint/validation/
  inference；gate variant 只覆盖 `build_network_architecture`，anatomy 另外覆盖 `get_training_transforms`。

### 新增入口

- `scripts/train/train_nnunet.py`：唯一训练入口，只做 variant → Trainer 类映射后调用 nnU-Net 官方
  `run_training`（在本进程内替换 `get_trainer_from_args`，不修改 nnU-Net 源码），无自写训练循环。
- `scripts/data/prepare_picai_nnunet.py`：统一数据准备（`baseline`/`zonal`/`splits` 三个 subcommand），
  带 tqdm 与成功/失败/跳过/耗时/输出目录汇总。`zonal` 由 `zonal_<source>.nii.gz` 派生与 T2W 网格
  一致的 PZ/TZ 两个 `[0,1]` 通道（`noNorm`），组织 Dataset606_PICAI_Zonal（5 通道）。

### 删除（旧的第二套框架与阶段门系统，共 168 个 tracked 文件）

- 自写训练框架：`src/.../training/`（trainer/checkpointing/optim/deep_supervision/losses/metrics/
  gpu_memory_profile）、`inference/sliding_window.py`、`evaluation/full_volume_validator.py`、
  `data/`（batch_provider/patch_sampler/preprocessed_store/transforms/picai/zonal_prior）。
- 自写整网实现：`models/`（m0_concat、m0_residual_concat、residual_unet3d、unet3d、residual_blocks、
  fusion_blocks、fusion_multi_branch、blocks、factory、profiling）。
- 旧 architecture config/hash/contract 系统：`config/`（architecture/experiment/plans）。
- 阶段门/协议系统：`protocols/`（g0_r/g0_e/g0_sap/g0_r_qc/g0_r_automated_qc/common）、
  `integrations/`（nnunet_picai_flce 已迁入 `nnunet/trainers.py`、nnunet_dataloader_probe）、`utils/`。
- 配置：整个 `configs/`（protocols/、architectures/ 的 M0–M4、experiments/ 的 M0–M4 YAML）。
- 脚本：`scripts/train/`（train_m0、train_n0_picai_flce、validate_m0_setup、summarize_m0_resenc_architecture、
  diagnose_n0_zero_dice）、`scripts/audit/`、`scripts/evaluate/`、`scripts/data/` 旧脚本、
  `scripts/analyze_*.py`、`check_registration.py`、`overlay_qc.py`。
- 文档：GLOSSARY/STATUS/START_HERE/research_plan(_plain_language)/protocol_changelog/
  PICAI_Model_Readiness_Audit/N0_PICAI_Official_Baseline/P2_M0_*/project_structure/Data_Analysis/
  experiment_log，以及 `docs/protocols/`、`docs/runbooks/`、`docs/literature/`、`docs/design/`、`docs/figures/`。
- 测试：仅验证已删除框架的测试（test_g0_*、test_checkpointing、test_trainer_synthetic、test_sliding_window、
  test_full_volume_validator、test_patch_sampler、test_batch_provider、test_m0_*、test_residual_*、
  test_architecture_config、test_skip_contract、test_train_m0_*、test_losses、test_picai、test_plans、
  test_profiling、test_progress、test_fusion_models、test_zonal_prior_pipeline、test_materialization_hardening、
  test_seg_sentinel_mapping、test_gpu_memory_profile、test_n0_zero_dice_diagnosis、test_evaluate_n0_validation、
  test_nnunet_picai_flce、test_loss_alignment_n0 等）。

### 保留（未改动）

`data/`、`workdir/`、`outputs/`（含 N0 baseline 与 legacy M0 产物）、`third_party/nnUNet`、
`scripts/env_nnunet.sh`。

### 测试

`tests/` 重写为纯合成 CPU 单元测试（不读取真实医学数据）：`test_nnunet_trainers.py`、
`test_nnunet_networks.py`、`test_prior_transforms.py`。覆盖 PI-CAI 损失与公开公式逐值一致、
三个 Trainer 为 `nnUNetTrainer` 子类且类名互异、NoFFT 保留 blur 且 benchmark=False、image/anatomy
gate 的通道契约、零初始化 `scales≡1` 与 gated MRI 恒等、deep supervision 输出格式与切换、strict
state_dict、PZ/TZ clamp、强度增强不动 PZ/TZ、空间变换同步、builder 可用于 inference、`train_nnunet.py --help`。
结果见 `docs/Training_Log.md` 与提交说明。

---

## 2026-09-21 — 重构修补：预测入口 / 依赖兼容边界 / 数据准备 fail-closed

在不重新设计、不恢复旧框架、不修改 `third_party/nnUNet` 的前提下修补上一轮重构的缺口。

### 预测加载链路（保持官方推理器）

- 新增最小项目预测入口 `scripts/inference/predict_nnunet.py`：在**进程内**把三个项目 Trainer 名
  直接映射到项目类（替换 `nnunetv2.inference.predict_from_raw_data.recursive_find_python_class`
  的解析：已知名直接返回、**不做递归扫描**，未知名字委托 nnU-Net 原函数），随后调用官方
  `predict_entry_point_modelfolder`。滑窗推理/预处理/导出仍全部由官方 `nnUNetPredictor` 完成，
  未自写推理器，未改动 nnU-Net 源码。
- `trainers.py` 增加 `PROJECT_TRAINERS` 注册表与 `resolve_trainer_class`，训练/预测入口共用。
- 新增合成测试 `tests/unit/test_predict_entry.py`：验证三个 checkpoint `trainer_name` 都能被预测
  入口解析到项目类（覆盖官方 `initialize_from_trained_model_folder` 实际调用的解析函数），
  而不仅是 `build_network_architecture`。

### nnU-Net 2.6.2 / dynamic-network-architectures 依赖兼容边界（如实记录，未升级）

- **现状**：`third_party/nnUNet` 为 v2.6.2，官方要求 `dynamic-network-architectures>=0.4.1,<0.5`；
  但 lm 环境中实际安装的 DNA 为 **0.3.1**，且 `nnunetv2` 元数据版本为 **2.5**（来自
  `/opt/data/private/lm/PENGWIN_Challenge/nnUNet` 的 editable install，要求 `DNA<0.4`）。
  因此 **2.6.2 的 DNA 下界未被满足**；未先 `source scripts/env_nnunet.sh` 时，`import nnunetv2` 会失败或
  解析到错误的环境分支，必须先 `source scripts/env_nnunet.sh` 把 `PYTHONPATH` 前置到
  `third_party/nnUNet` 才解析到固定的 2.6.2 源码。
- **不升级/不覆盖**：升级 DNA 到 0.4.x 会破坏依赖 `DNA<0.4` 的 PENGWIN nnUNetv2 2.5 editable 安装，
  故本项目保持 DNA 0.3.1 不变。
- **兼容方案与影响**：仅使用 `PlainConvUNet` 路径（Dataset605/606 的 `3d_fullres` plans 均为该类），
  其在 DNA 0.3.1 下可正常构建/前向/deep supervision/strict state_dict（合成测试通过）。
  **影响边界**：一旦改用需要 DNA>=0.4 的架构（如 `ResidualEncoderUNet` 新参数）或 2.6.2 中依赖
  新版 DNA 行为的代码路径，当前环境不保证可用，必须先解决 DNA 冲突。
- **入口保护**：`zonal_reliability_fusion.nnunet.check_fixed_nnunet_runtime()` 在训练/预测入口显式校验
  `nnunetv2` 解析到 `third_party/nnUNet`（否则报错，避免误用 2.5 fork）、`PlainConvUNet` 可由
  `pydoc` 直接定位（从而 `get_network_from_plans` 不退化为递归扫描），并**如实打印** DNA 版本是否满足
  `>=0.4.1,<0.5`；不满足时只 WARNING，不宣称依赖完全满足。

### 数据准备 fail-closed

- `scripts/data/prepare_picai_nnunet.py`：任一 `failed/conflict`（或完整运行时磁盘病例集合与选择集合
  不一致）先打印完整汇总，再 `SystemExit(2)`，且**不发布**新的 `dataset.json`；`numTraining` 改为
  `imagesTr` 中拥有全部通道的**实际完整病例数**；`--overwrite` + `--cases` 子集运行后若仍含选择外旧病例，
  明确 WARNING（绝不静默遗留）；`splits_final.json` 冲突时非零退出且不覆盖既有文件；删除未使用的
  `datetime` import。
- 新增合成测试 `tests/unit/test_prepare_picai_failclosed.py`（临时目录 + 纯合成小文件 / 合成 NIfTI，
  不读真实医学数据）。

### 其他

- 新增最小 `pytest.ini`：`testpaths=tests` + `norecursedirs`，使根目录 `pytest -q` 只收集 `tests/`，
  不扫描 `data/workdir/outputs/third_party/logs`。
- 静态质量：`ruff format` 全量格式化，`ruff check` 清零；带 shebang 的脚本设为可执行。
- `docs/Training_Log.md` 只保留真实训练事实与简短状态；待运行的操作命令统一集中到 `README.md`，
  避免重复的步骤文档。
- 新增测试合计后 `pytest -q` 共 **46 passed / 0 failed**（纯合成、CPU、不读真实数据）。

---

## 2026-09-21 — zonal-source 来源安全与 prior 复用校验

- `prepare_picai_nnunet.py zonal`：`--zonal-source` 改为**必填**（`required=True`，取消默认 `yuan`，
  仍限 `yuan/hevi`），消除“帮助文本说必须选、实际却有默认值”的不一致。
- 复用既有 PZ/TZ 前先做**来源/几何/内容校验**：两者都已存在且未 `--overwrite` 时，读取本次指定的
  `zonal_<source>.nii.gz`，校验其与 T2W 的 size/spacing/origin/direction，再由 PZ=1/TZ=2 派生预期
  float32 `[0,1]` 通道，逐项比对既有 PZ/TZ 的几何、形状、数值范围与内容；**完全一致才 `skipped`**，
  否则 `conflict`（来源/几何/内容不匹配）或 `failed`（不可读）。
- 只存在 PZ/TZ 之一且未 `--overwrite` → 判为 `conflict`，不覆盖已有文件、不补写缺失文件。
- **fail-closed**：任一 `conflict/failed` 先打印完整汇总再非零退出，且**不发布** `dataset.json`；
  仅显式 `--overwrite` 才重新原子生成 PZ/TZ（可用于来源切换或修复）。派生文件原子写入、原始数据只读、
  `noNorm`、与 T2W 网格一致、tqdm 与结构化汇总等约束不变。
- `dataset.json` 的 `description` 记录本次 `prior_source=zonal_<source>`，使来源可审计（不新增说明/元数据文件）。
- 新增纯合成测试（`tests/unit/test_prepare_picai_failclosed.py`）：缺 `--zonal-source` 时 argparse 非零退出；
  首次 `yuan` 生成五通道；来源一致时 `--resume` 可安全 `skipped`；yuan→hevi 内容不同且未 `--overwrite`
  非零退出且不改既有文件、不发布 dataset.json；只存在一个 prior 且未 `--overwrite` 非零退出且不生成
  缺失文件；`--overwrite` 可按新来源重新生成且内容正确；dataset.json 可审计 zonal 来源。
