# 项目结构约定

## 文档职责（每类事实只有一个权威来源）

| 文件 | 职责 |
|---|---|
| `docs/research_plan.md` | 研究协议（计划）：假设、数据与独立测试设计、模型与消融、冻结训练/评价/统计协议、阶段门定义、风险与失败转向、待冻结决策表 |
| `docs/STATUS.md` | **当前状态唯一权威**：门状态、实现状态、blocker、下一步 |
| `docs/GLOSSARY.md` | **标号与步骤总览**：门/阶段/模型/假设/决策/blocker 编号释义、依赖顺序与常见混淆点 |
| `docs/protocol_changelog.md` | 协议版本变更史（版本/日期/内容/原因/是否在查看结果前/对既有结果的影响） |
| `docs/experiment_log.md` | 实际运行命令、环境、耗时、结果、失败与产物 |
| `docs/Development_Log.md` | 代码、测试、配置与文档变更 |
| `docs/runbooks/` | 由研究者执行的命令、输出路径、进度观察与成功判据 |
| `docs/protocols/` | G0-R / G0-E / G0-SAP 详细协议（DRAFT；机器可读载体在 `configs/protocols/`） |
| `docs/literature/` | 参考资料总表与文献/规范核查记录 |
| `docs/P2_M0_Implementation.md` | legacy PlainConv M0 工程归档（含附录 A：v2.2 低频验证 checklist） |
| `docs/P2_M0_ResidualEncoder_v23.md` | v2.3 正式 M0（P2A）实现说明与 G2 前置验收 |
| `docs/PICAI_Model_Readiness_Audit.md` | G0-P 数据审计报告与判定 |
| `docs/N0_PICAI_Official_Baseline.md` | N0 定义、术语边界与训练命令 |
| `docs/Data_Analysis.md` | 数据集事实（历史方向建议已被 v2.1/v2.2 覆盖，见文首说明） |

**维护纪律**：状态只写 `docs/STATUS.md`；运行事实只写 `docs/experiment_log.md`；协议条款只写
`docs/research_plan.md` 与 `docs/protocols/`；代理规则只写 `AGENTS.md`。

## 核心代码

项目实现统一放在 `src/zonal_reliability_fusion/`。第三方代码不得直接修改；需要适配时，在 `integrations/` 中编写包装层或子类。

以下是 **v2.2 legacy PlainConv M0**（其历史实现曾达到 CODE READY）的现有模块布局，自 v2.3 起仅为 legacy/feasibility；
**v2.3 正式 M0（plan-driven Residual-Encoder 3D U-Net）已于 P2A 实现**（residual block、版本化架构配置与稳定哈希、
checkpoint 身份隔离与合成测试均已落地，见下方“v2.3 Residual-Encoder M0”小节与 `docs/P2_M0_ResidualEncoder_v23.md`）。
下列 legacy 模型文件和入口不得被解释为 v2.3 正式 M0：

- `config/`：`plans.py`（只读解析 nnUNetPlans）、`experiment.py`（M0 配置加载与冲突校验）
- `models/`：`blocks.py`（legacy ConvNormAct3d / 对齐裁剪）、`unet3d.py`（legacy PlainConv plan-driven 主干）、
  `m0_concat.py`（legacy M0 三通道 concat）
- `data/`：`preprocessed_store.py`（**唯一 import nnunetv2 的只读适配层**）、`patch_sampler.py`（M0–M4 共用采样）、
  `transforms.py`（同步镜像 + 强度接口 + 未实现增强的显式占位）、`batch_provider.py`（torch batch 迭代器）
- `training/`：`losses.py`（**默认 `PiCAIFocalCELoss`，与正式 N0 对齐**；`DiceCELoss` 为迁移前口径，不可与 N0 比较）、
  `deep_supervision.py`、`metrics.py`、`optim.py`、`checkpointing.py`、`trainer.py`
  （两种验证模式：`full_volume` 正式 / `diagnostic_patch` 诊断；v2.2 低频验证调度 `every_n_epochs=5`、
  validation-event 计数、非验证 epoch 不更新 best；early stopping 可选且正式主实验禁用）
- `inference/`：`sliding_window.py`（Gaussian 融合、覆盖断言、DS 主输出）
- `evaluation/`：`full_volume_validator.py`（按 `every_n_epochs`（正式 5，最后一个 epoch 必验证）遍历全部 223 个
  validation study 做完整 3D 滑窗推理，输出阳性 case-wise Dice、micro Dice 与阴性 FP 统计；病例级 CSV 落
  `outputs/metrics/m0/<run>/validation_cases/`）
- `utils/`：`progress.py`、`reproducibility.py`
- 入口脚本：`scripts/train/validate_m0_setup.py`、`scripts/train/train_m0.py`（现按实验配置的 `architecture` 块经
  `models/factory.build_model` 按 `experiment.model_id` 分派 M0–M4（未知 model_id 报错、不回退）；
  M0 仍按 `architecture` 块区分 legacy PlainConv / v2.3 Residual-Encoder；正式 G2 仍待真实数据三步验证）
- 配置：`configs/experiments/m0_picai_3d_fullres.yaml`（legacy PlainConv 配置，保留审计，不覆盖）

**v2.3 Residual-Encoder M0（P2A 已实现；正式 M0–M4 共同骨干，与 legacy 并存不覆盖）**：

- `models/residual_blocks.py`：`ResidualBlock3d`（post-activation，BasicBlockD 等价：conv1=Conv-IN-LeakyReLU、
  conv2=Conv-IN、加后 LeakyReLU；shortcut=identity 或 AvgPool(`ceil_mode=True`)+1×1×1 Conv(bias=False)-IN）、
  `StackedResidualBlocks3d`（首块承担 stride/通道变化）；
- `models/residual_unet3d.py`：`PlanDrivenResidualEncoderUNet3D`（default stem + residual encoder + 普通 U-Net decoder
  + DS 高→低；`zero_init_residual_last_norms`）；
- `models/m0_residual_concat.py`：`M0ResidualConcatModel`（三通道 concat；不读 PZ/TZ/WG/gate；带架构身份）；
- `models/profiling.py`：参数量、解析 MACs（无 forward）、hook MACs（缩小 plan 验证）、激活规模、结构摘要与显存状态；
- `models/factory.py`：`build_model`（按 `experiment.model_id` 分派 M0–M4；M0 走 `build_m0_model` 的行为，
  **未知 model_id / 变体缺失或与 model_id 不符 / M0 声明融合变体一律报错，不回退**）、`build_m0_model`
  （兼容入口：按 `architecture` 块分派 legacy/residual）、`model_architecture_identity`、`model_identity`；
- `config/architecture.py`：`ArchitectureSpec`（版本化架构配置）、canonical-JSON SHA-256 架构哈希、§8.6 浅层 skip
  接口契约（`SHALLOW_SKIP_CONTRACT`）、`verify_architecture_identity`（legacy/哈希不匹配拒绝）；
- 配置：`configs/architectures/m0_resenc_v23.yaml`（架构，blocks_per_stage=(1,3,4,6,6,6,6)）、
  `configs/experiments/m0_resenc_picai_3d_fullres_v23.yaml`（实验，独立 run/输出身份 `outputs/checkpoints/m0_resenc`）；
- 入口：`scripts/train/summarize_m0_resenc_architecture.py`（结构摘要/预算，只读、无 GPU）；`train_m0.py` /
  `validate_m0_setup.py` 改用工厂分派并在 resume 前做架构守卫；
- 边界不变：`models/`、`training/`、`inference/` 不 import `nnunetv2`；legacy PlainConv 文件/配置/产物原位保留。

**M1–M4 融合模型（v2.4，代码已实现；未训练）**：

- `models/fusion_blocks.py`：`ModalityProjection`（1×1×1 + InstanceNorm + LeakyReLU，逐模态）、
  `ShallowSkipAggregator`（stage 0/1 的三路 concat 聚合，构造时执行浅层 skip 契约守卫，禁止读取
  `pz_tz`/`wg`/gate/decoder 特征）、`ImageDrivenGate`（逐体素三模态 logits + 模态维 softmax，
  logits 层零初始化 → 初始严格等权）、`ZoneEncoder`（2 通道 PZ/TZ → `adaptive_avg_pool3d` 到融合 stage →
  conv stem + 残差块）、`ConditionalAffineHead`（输出 γ/β，末层零初始化 → 初始恒等调制）、`ZoneInjection`（M3 融合后加性注入）；
- `models/fusion_multi_branch.py`：`MultiBranchFusionUNet3D` 基类 + `M1EqualFusionModel` /
  `M2ImageGateFusionModel` / `M3ZoneInputFusionModel` / `M4ConditionedGateFusionModel`（三路模态独立浅层分支 →
  stage-2 融合 → 共享 encoder 后半段 + 普通 decoder；实现 `REQUIRED_INPUT_KEYS`、`model_identity()`、
  `parameter_breakdown()`、`summary()`、`set_diagnostics()/gate_diagnostics()`）；
- `config/architecture.py`：`fusion_family` 节解析与严格校验（M2/M3/M4 的 image gate 必须完全相同、
  M3/M4 的 zone encoder 必须完全相同）、`resolve_fusion_spec`（变体解析进入架构哈希 → 逐模型身份隔离）；
- 配置：`configs/architectures/fusion_resenc_v23.yaml` + 4 个实验
  `configs/experiments/{m1_equal,m2_image_gate,m3_zone_input,m4_conditioned_gate}_picai_3d_fullres_v24.yaml`
  （互异 `paths.output_root`，含 v24；M3/M4 绑定 `zonal_prior_source=yuan`）；
- 测试：`tests/unit/test_fusion_models.py`（shape/DS 顺序、梯度有限、AMP、等权/和为 1、零初始化恒等、
  拒绝 zonal/WG、身份隔离、哈希稳定、固定 seed 可复现等）。

**PZ/TZ prior 物化与读取链（M3/M4 数据前置；代码完成 + 合成测试通过；1500 例全量物化已于 2026-09-17 完成）**：

- `data/zonal_prior.py`：几何转换（transpose → crop → 最近邻重采样，复用 nnU-Net 逻辑）、`label_to_pz_tz`、
  `validate_prior_array`（finite / [0,1] / PZ+TZ≤1）、`require_oracle_pass`（lesion 与 `.b2nd` seg 逐体素一致）、
  `normalize_prior`（统一 contiguous float32）、`atomic_write_json`、**成对原子提交 + 失败回滚状态机**
  （`commit_sidecar_pair` / `SidecarCommitError`）、**唯一严格校验器** `validate_sidecar`、manifest 构造/校验
  （`build_manifest` / `verify_manifest`）、`ZonalPriorMaterializer`（逐例进度、canonical manifest 发布门控、
  resume 严格校验器判据、`collect_verified_records` 从磁盘重建）；
- `data/preprocessed_store.py`：`load_zonal_prior` / `load_zonal_prior_manifest`（逐例深度校验 + 惰性 tqdm +
  统计，**无跨调用弱缓存**）、`zonal_prior_frozen_records`（深层不可变的逐例冻结哈希）、
  `require_frozen_manifest`（正式训练未冻结即拒绝读取）、READY 状态一次性提交与失败复位；
- 入口：`scripts/data/materialize_zonal_prior.py`（`--dry-run`/`--resume`/`--overwrite`/`--limit-cases`/
  `--case-ids`/`--no-progress`）；只写 `data/processed/picai_zonal_<source>_v1/`，绝不触碰 raw/`.b2nd`/plan/split；
- seg 读取适配：`normalize_seg` 先验证后转换（`-1` 哨兵 → 0；返回严格 `uint8 ⊆ {0,1}`）。

**`--overfit` 运行时 GPU 显存遥测（代码完成 + mock CUDA 合成测试）**：

- `training/gpu_memory_profile.py`：`GpuMemoryProfiler`（`start` → `mark_phase` → `snapshot` → `finalize`）、
  `atomic_write_json`（tmp + fsync + `os.replace`）、`is_cuda_oom`、`bytes_to_mib/bytes_to_gib`、
  `MEASUREMENT_SCOPE`（进程 vs 整卡语义）、`PHASE_*`/`STATUS_*` 常量；
- 入口：`scripts/train/train_m0.py --overfit`（`reset_peak_memory_stats` 早于 `build_trainer`；
  成功 / `CUDA_OOM` / `ERROR` / CPU `NOT_APPLICABLE` 都落盘报告，路径见「实验输出」）；
- 与静态报告的关系：`models/profiling.py` 的 `peak_gpu_memory.status=NOT_MEASURED` **不变**，
  运行时实测是独立产物（不得互相替代）。

G0 协议工具（G0-R / G0-E / G0-SAP；**不决定任何门的通过**）。按模块区分数据访问边界：

- **只处理配置/元数据**（不读影像体素、不导入 torch/nnunetv2）：
  `common.py`（协议配置契约/内容哈希/运行元数据/输出路径隔离）、
  `g0_r.py`（确定性分层抽样、盲审模板、对齐指标 schema）、
  `g0_e.py`（文件清单级清点、证据状态 UNKNOWN 语义）、
  `g0_sap.py`（评测配置硬约束校验、SAP-A/B/C 阶段与预测可见性校验、SAP-A 方法块哈希与差异审计、
  冻结就绪检查、模板渲染）；
- **图像 QC 核心**（`g0_r_qc.py`）：**研究者运行时会只读物化影像体素**（T2W/ADC/HBV），
  用于物理坐标重采样、棋盘格/边缘叠加、landmark 物理位移与边界距离；**不修改数据、不训练、不推理、不配准**；
  模块被导入时不读影像、不占 GPU；
- 入口脚本：`scripts/audit/audit_picai_alignment_qc.py`（抽样/盲审清单，只读元数据）、
  `scripts/audit/render_picai_alignment_qc.py`（**人工路径（历史草案）**：图像 QC 渲染/指标/校验；
  研究者运行时只读物化影像体素；盲审材料不含病灶标签与模型输出）、
  `scripts/audit/run_picai_alignment_qc_automated.py`（**G0-R Automated v0.2 当前主路径**：
  零人工阅片/landmark/双阅片/人工阈值；A 几何+FOV → B 多尺度边缘一致性 → C 诊断刚体（仅内存）→
  D 合成位移校准 → 候选决策；只读物化影像、不写回、不联网）、
  `scripts/audit/audit_g0e_candidates.py`（文件清单/表头）、
  `scripts/evaluate/prepare_g0_sap_freeze.py`（冻结载体；不运行评测）
  —— 自动 QC 算法库为 `g0_r_automated_qc`（纯计算，不读/写文件）
  —— 均支持 `--dry-run` 与 `--no-progress`；图像 QC 另支持
  `--case-ids`/`--validate-landmarks`/`--allow-incomplete`/`--boundary-source`；
- 协议载体：`docs/protocols/*.md` + `configs/protocols/*.yaml`（状态一律 `DRAFT`，未冻结）；
  图像 QC 工具**已实现但真实病例未运行**，G0-R 仍为 Pending。

N0 官方基线（`N0 = PI-CAI official-style nnU-Net v2 Focal+CE NoFFT`）的适配与入口：

- `integrations/nnunet_picai_flce.py`：`PiCAIFocalCrossEntropyLoss`（复刻 PI-CAI 官方公开的
  `0.5×FocalLoss(gamma=2, alpha=None) + 0.5×CrossEntropyLoss`）+ `nnUNetTrainerPICAI_FLCE`
  （v2.6.2 trainer 子类，逐 deep-supervision 尺度替换损失；并关闭 Gaussian blur 的 FFT benchmark 作为本环境
  稳定性修复）+ `nnUNetTrainerPICAI_FLCE_NoFFT`（等价子类，仅提供独立输出身份，与首次失败 run 目录隔离）
- 入口脚本：`scripts/train/train_n0_picai_flce.py`（单 GPU；通过替换进程内 `get_trainer_from_args` 注入适配
  trainer，**不修改 third_party/**；显式拒绝多 GPU/静默 DDP）
- 测试：`tests/unit/test_nnunet_picai_flce.py`（合成 CPU）
- 定义、术语边界与运行命令见 `docs/N0_PICAI_Official_Baseline.md` 与 `docs/research_plan.md` §7.1

边界约定：`models/`、`training/`、`inference/` 不得 import `nnunetv2` / `nnUNetTrainer`；
`data/preprocessed_store.py` 是数据层的只读适配层；`integrations/` 是第三方适配层，允许在其中**惰性** import
`nnunetv2` / `batchgenerators(v2)`（如 N0 Focal+CE trainer、N0 排障探针）；
`M0` 不得引入 PZ/TZ、WG、gate、attention、预训练权重或病灶专用分支。

## 数据

- `data/raw/`：只读原始数据或指向原始数据的本地链接
- `data/interim/`：转换过程中的中间数据
- `data/processed/`：可直接训练的数据
  - `data/processed/picai/`：P0B 物化产物（每例 `cases/<case_id>/t2w.nii.gz, adc.nii.gz, hbv.nii.gz, lesion.nii.gz, zonal_yuan.nii.gz, zonal_hevi.nii.gz[, wg.nii.gz]`；另有 `materialization_report.csv`、`materialization_summary.json`、`materialization_validation.json|csv`）
- `data/splits/`：患者级数据划分
  - `picai_train_val_split.json`：当前正式 train/validation 划分（v2.0 协议，患者级/study 级列表）
  - `picai_nnunet_splits_validation.json`、`picai_nnunet_split_cases.csv`：nnU-Net 单 fold 划分的校验结果与 case→split 明细
  - `picai_cross_center_splits.json`：历史文件（仅审计追踪，不用于训练）
  - nnU-Net 实际读取的划分：`workdir/nnUNet_preprocessed/Dataset605_PICAI/splits_final.json`（fold 0 = train 1277 / val 223，
    在 `workdir/nnUNet_raw/Dataset605_PICAI/` 保留同内容副本以防重新规划时丢失）
- `data/metadata/`：去标识化的数据清单和中心信息（manifest、nnU-Net 病例映射、P0A 验证证据等）

原始影像、内部数据和可识别信息不得提交到版本库。

## nnU-Net 工作区（项目内）

用项目内环境脚本固定（覆盖 `/root/.bashrc` 的全局 `nnUNet_*` 变量；并把 `PYTHONPATH` 前置到
`third_party/nnUNet` 固定 v2.6.2，避免 lm 环境中可编辑安装的其他 nnunetv2 分支遮蔽）：

```bash
source scripts/env_nnunet.sh
```

- `nnUNet_raw`：`workdir/nnUNet_raw`（`Dataset605_PICAI`：imagesTr/labelsTr 为相对符号链接，指向 `data/processed/picai/cases/`）
- `nnUNet_preprocessed`：`workdir/nnUNet_preprocessed`
- `nnUNet_results`：`outputs/nnUNet_results`

## 实验输出

- `outputs/nnUNet_results/Dataset605_PICAI/`：官方 nnU-Net 训练产物。
  **正式 N0** = `nnUNetTrainerPICAI_FLCE_NoFFT__nnUNetPlans__3d_fullres/`（尚未创建，等待研究者启动）；
  `nnUNetTrainer__nnUNetPlans__3d_fullres/` 与 `nnUNetTrainerPICAI_FLCE__nnUNetPlans__3d_fullres/`
  分别是**已排除的默认 Dice+CE 预实验**与**首次 FFT 崩溃 run**，两者的日志、checkpoint 与目录
  **必须保留、不得删除或覆盖**。
- `outputs/checkpoints/`：模型权重（自研 M0：`outputs/checkpoints/m0/<run>/`）
- `outputs/metrics/m0/<run>/`：自研 M0 的 epoch 级 CSV/JSONL 指标
- `outputs/diagnostics/<model_id>/<run>/`：setup 检查、loader smoke、run manifest、loss 曲线与 overfit 产物
  （M0 仍为 `outputs/diagnostics/m0/...`）；
  **运行时峰值显存报告**固定在 `outputs/diagnostics/<model_id>/<run_name>/diagnostics/gpu_memory_profile.json`
  （`--overfit` 自动写出；成功 / `CUDA_OOM` / `ERROR` 都有，CPU 为 `NOT_APPLICABLE`）
- `data/processed/picai_zonal_<source>_v1/`：plan-space PZ/TZ prior sidecar
  （`<case>.npz` + `<case>.json` + canonical `manifest.json` + `materialization_summary.json`；
  与 `data/processed/picai/` 的病例级物化产物相互独立，均不得覆盖）
- `outputs/predictions/`：预测结果
- `outputs/metrics/`：结构化指标
- `outputs/figures/`：论文图表
- `outputs/diagnostics/`：调试与审计产物
- `outputs/ablations/`：消融实验汇总
- `outputs/external/`：外部数据集验证汇总（G0-E 通过后启用）
- `outputs/reports/`：最终报告

## 第三方代码

第三方仓库统一放在 `third_party/<repository>/`，并在 `third_party/README.md` 中记录来源、版本和用途。自研代码不得写入第三方仓库。

## 开发与运行约定

协作代理与开发流程的完整规则见项目根目录 `AGENTS.md`，要点：

1. **进度可见**：数据处理、训练、评测和审计等耗时脚本必须提供可观察的进度显示（如 tqdm 进度条或等价实现）。
2. **命令由研究者执行**：数据处理、训练，以及验证（validation/verify）、推理、评测、批量分析等预计耗时较长的命令均不代为运行；协作代理只给出完整、可直接复制的命令，由研究者本人手动执行；命令归属不明确时默认交由研究者运行。
3. **环境统一为 conda `lm`**：所有命令在 `/root/anaconda3/envs/lm` 下运行，命令中显式写出 `conda activate lm`（涉及 nnU-Net 时附 `source scripts/env_nnunet.sh`）；缺失的第三方包由协作代理安装到该环境并记录版本。
