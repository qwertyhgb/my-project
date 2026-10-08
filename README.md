# Anatomy-Guided Lesion-Aware Coarse-to-Fine Prostate Cancer Segmentation

**基于预测解剖先验的病灶感知粗到细前列腺癌分割**

> 本项目研究预测前列腺解剖先验与病灶感知粗到细学习能否提高小型和困难前列腺癌病灶的检出与
> 完整分割，同时控制假阳性。
>
> *This project investigates whether predicted prostate anatomy and lesion-aware
> coarse-to-fine learning can improve the detection and complete segmentation of small and
> difficult prostate cancer lesions while controlling false positives.*

---

## Goal

围绕前列腺癌（csPCa）病灶分割的三个**核心失败模式**，建立一条逻辑完整、实验可完成、结果可解释
的研究主线：

1. **完全漏检 lesion**（尤其是小病灶）；
2. **已检出病灶仅分割出核心区域，覆盖不足**；
3. **提高 sensitivity 后引入过多 false positives**。

新的核心科学问题不是"T2W / ADC / HBV 应该以什么权重融合"，而是：

> 如何利用预测得到的解剖先验 WG / PZ / TZ，以及 lesion-aware learning，减少小病灶和困难病灶的
> 完全漏检，提高已检出病灶的完整覆盖，同时避免假阳性显著增加？

## Current Research Question

> **显式的病灶定位（coarse lesionness）加上预测解剖先验提供的结构化上下文，能否在不显著增加
> 假阳负担的前提下，减少小病灶与困难病灶的完全漏检并改善已检出病灶的完整覆盖？**

- 研究问题、假设、方法机制、研究边界、预期贡献与限制 → `docs/Research_Plan.md`
- 方法机制与实现边界（含全部 fail-closed 清单） → `docs/Method.md`
- 实验矩阵、预算、命令模板、stop rules 判据 → `docs/Experiment_Plan.md`
- 指标定义与统计口径（**唯一**来源） → `docs/Evaluation_Protocol.md`
- 已发生的训练事实 → `docs/Training_Log.md`；有证据支持的结论 → `docs/Findings.md`
- 工程变更 → `docs/Development_Log.md`；旧研究路线 → `docs/archive/`

## Main Pipeline

```text
                    ┌──────────────────────────────┐
   T2W  ───────────▶│ Stage 1: anatomy model       │──▶ P(WG) / P(PZ) / P(TZ)  （soft, 冻结）
                    │ Dataset607 / 单通道 T2W      │      ↓ 不是主要创新点，只是先验生成器
                    └──────────────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │ 条件 B: Anatomy-Guided ROI    │  预测 WG → bbox → 物理 margin（mm）
                    │ （禁 hard mask；不重采样）    │  → 前列腺 ROI
                    └──────────────────────────────┘
                                   │
   T2W / ADC / HBV ────────────────┤
   predicted P(PZ) / P(TZ) ────────┤
                                   ▼
                    ┌──────────────────────────────┐
                    │ 条件 C: Lesion-Aware C2F      │  原生 backbone（共享 encoder/decoder）
                    │ ├── coarse lesionness 头      │  ← 物理半径膨胀的粗掩膜监督
                    │ └── soft 残差 refinement      │  ← F_out = F + delta（末层零初始化）
                    └──────────────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │ 条件 D: Zone-Aware Refinement │  soft P(PZ) / P(TZ) / P(U) 解剖上下文
                    └──────────────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │ 条件 E: Hard Negative Mining  │  解剖约束的高置信假阳位置采样（可开关）
                    └──────────────────────────────┘
                                   │
                                   ▼
                         final lesion segmentation
```

**设计顺序不允许颠倒**：`strong baseline → anatomy ROI → lesion localization →
coarse-to-fine refinement → zone anatomy context → hard-negative control`。

## Dataset

三层数据的职责**禁止混淆**（详细清单见 `docs/Method.md`）：

| 层 | 内容 | 角色 |
|---|---|---|
| raw inputs | **T2W / ADC / HBV** | 模型输入；原始医学数据**只读** |
| training labels | **lesion GT**（Dataset605/606） | 训练监督与验证参照 |
| predicted priors | **P(WG) / P(PZ) / P(TZ)** | Stage-1 模型对**该病例自身 MRI** 的预测 |
| oracle labels | GT WG / PZ / TZ | 仅上界分析，必须标记 `ORACLE_GT` |

| 数据集 | 内容 | 通道 | 病例 | 状态 |
|---|---|---|---|---|
| `Dataset605_PICAI` | PI-CAI 派生的三序列病灶分割 | 3（T2W / ADC / HBV） | 1500（train 1277 / val 223） | 已物化 |
| `Dataset606_PICAI_Zonal` | 在 605 基础上追加自动分区掩膜 | 5（+ PZ / TZ） | 同 605 的病例集合 | 已物化 |
| `Dataset607_PICAI_Anatomy` | Stage-1 解剖先验伪监督（位编码 `WG+2*PZ+4*TZ`） | 1（T2W） | 1499 study（train 1276 / val 223） | 已物化；模型**已训练 100 epoch + validation**（但 **WG 头不可用**，见下文） |
| `Prostate158` | 外部数据 | 外部序列 | — | 仅作为 distribution-shift stress test |

**禁止把 predicted anatomy 与 GT anatomy 混淆。** 所有 final lesion experiments 默认使用
**predicted anatomy priors**；GT anatomy 只能用于显式标记 `ORACLE_GT` 的上界分析，不得作为正式
最终性能。

`Dataset605 ↔ 606` 的归因边界：前三个 MRI 通道已经过**逐数组一致性审计**（PASS，
`outputs/reports/dataset605_606_mri_equivalence_audit_v2.json`），但 **PZ/TZ 不参与判等**，因此
基于 606 的条件（D）相对基于 605 的条件（C）的差异**不能**被声称"唯一差异是 PZ/TZ"。

## Strong Baseline

先确定 strong baseline，再在其上逐步叠加新模块。第一个比较**只变损失**：

| 臂 | variant | 损失 |
|---|---|---|
| **A1** | `positive_sampling` | PI-CAI `0.5*Focal(gamma=2)+0.5*CE`（项目实现） |
| **A2** | `dicece_positive_sampling` | nnU-Net v2.6.2 **原生** Dice+CE |

两者除损失外逐项相同：网络（原生 `PlainConvUNet`）、数据集与 split、patch/batch size、增强
（含 NoFFT 修复）、前景采样、deep supervision 权重、optimizer（SGD+Nesterov）、初始学习率、
PolyLR 总周期、epoch 数、checkpoint、validation、滑窗推理。**胜者成为后续所有条件的参照。**

> `positive_sampling` 是 **foreground-aware training stabilization strategy**，**不是**论文的
> 方法创新。它属于 strong baseline 的组成部分。

**当前状态**：

- A1 `positive_sampling`：**已完成**训练 + validation（见 `docs/Training_Log.md`）；
- A2 `dicece_positive_sampling`：**代码就绪、未训练**。

## Anatomy Prior Generation（Stage 1）

- 数据集契约：`zonal_reliability_fusion.anatomy.contracts`（单 T2W、位编码 region heads、
  fold 0 冻结 split、显式排除已知缺失病例）；
- 训练器：`nnUNetTrainerPICAI_AnatomyJoint_100ep_NoFFT`（100 epoch 工程性试跑预算）；
- 预测与导出校验：`zonal_reliability_fusion.anatomy.inference`（必须 `--save_probabilities`，
  逐病例校验三件套与物理元数据）；
- 先验质量与不确定度：`zonal_reliability_fusion.anatomy.validation`。

**这个模型不是主要创新点**，它是 `anatomical prior generator`。它的误差是下游的**真实输入分布**
的一部分，不做任何人工修正。

**当前状态**：数据已物化（1499 study），模型**已完成 100 epoch 训练与 validation**
（2026-10-06，223 例）。验证 split 的 soft prior 已随之落在 Stage-1 的 validation 目录里：

```text
outputs/nnUNet_results/Dataset607_PICAI_Anatomy/
  nnUNetTrainerPICAI_AnatomyJoint_100ep_NoFFT__nnUNetPlans__3d_fullres/fold_0/validation/
    <case>.npz   × 223   soft probability（键 probabilities，(3,Z,Y,X)，WG/PZ/TZ）
    <case>.pkl   × 223   物理元数据
    <case>.nii.gz × 223  原生 region 硬标签（下游**不**使用）
```

`scripts/data/check_split_integrity.py --anatomy-prior-dir <该目录>` 已验证这 223 例先验与
anatomy val split **逐例精确一致**（`[PASS] anatomy_prior_cases`）。

**训练 split 的先验尚未生成**（`workdir/anatomy_priors/**` 不存在）。

而验证集的逐区域结果暴露出一个**阻塞性问题**：

| region | Dice | 结论 |
|---|---:|---|
| WG `[1,3,5,7]` | **0.0056** | GT 参考体积并不小（`n_ref` 逐例均值 121637 voxel），但模型只预测约 618 voxel/例 ⇒ **WG 头实际不可用** |
| PZ `[2,3,6,7]` | 0.8985 | 可用 |
| TZ `[4,5,6,7]` | 0.9361 | 可用 |

`Mean Validation Dice = 0.6134` 是**三个区域的算术平均**，它掩盖了 WG 的失败。这直接阻塞条件 B
（ROI 的唯一来源是 predicted WG：空 WG 预测会回退全视野，ROI 退化为"没有 ROI"）。**因此下一步不是
启动 B，而是先解决 WG 预测**。原因尚未定位（优化/损失 vs 位编码与 region 定义下的数据问题），
**不预判**；诊断入口见 `docs/experiments/anatomy_joint_100ep.md`。

## Lesion Model（Stage 2）

| 条件 | variant | 相对上一条的**唯一**改动 | 状态 |
|---|---|---|---|
| **B** | `lesion_roi` | 训练 patch 采样空间（ROI 约束；网络/损失不变） | 代码就绪 / 未训练；**被 Stage-1 的 WG 结果阻塞** |
| **C** | `lesion_coarse_to_fine` | + coarse lesionness 头 + soft 残差 refinement + lesionness 辅助损失 | 代码就绪 / 未训练 |
| **D** | `lesion_zone_refine` | + 两个 soft zone 概率通道（`concat` 模式） | 代码就绪 / 未训练（需 Dataset606） |
| **E** | `lesion_hard_negative` | + 困难负样本采样（`--hard-negative-set` 为 `None` 时完全退化为 C/D） | 代码就绪 / 未训练 |

关键设计（细节见 `docs/Method.md`）：

- **禁止 `image * wg_mask`**：WG 边界错误会直接删除病灶；ROI 只做 bbox + **物理 margin（mm）**，
  每轴 voxel 数向上取整，**不重采样**（padding 而非 resize），crop transform 完整保存；
- **空 WG 预测回退全视野并记录原因**，绝不产生"把病灶裁掉"的 ROI；
- **lesionness 目标是物理半径膨胀的粗掩膜**（`LESIONNESS_DILATION_RADIUS_MM = 3.0`，预先冻结），
  用椭球结构元素保证各向异性下物理半径正确；
- **refinement 末层零初始化** ⇒ 初始 `F_out == F`，与 strong baseline **逐值一致**；
  `soft guidance` 永不硬裁区域，零 coarse score 区域仍有完整 residual path；
- **PZ/TZ 只作为解剖上下文**（`P(U) = 1 - max(P(PZ), P(TZ))` 覆盖分区边界与不确定区），
  **不是** modality 重加权；默认 `concat` 模式，`zone_experts` 只在被证明必要时才启用；
- **困难负样本只在训练 split 内挖掘**，validation 绝不参与。

## Evaluation

**以 lesion failure mode 为中心**，不再只看 `foreground_mean Dice`。完整定义与统计口径见
`docs/Evaluation_Protocol.md`。

| 级别 | 指标 |
|---|---|
| **Primary** | Positive-case macro Dice（完全漏检病例 Dice = 0，不得排除） |
| **Key secondary** | lesion-level sensitivity、small-lesion sensitivity、matched-lesion Dice、completely missed lesion/case rate、positive voxel recall、positive voxel precision、false-positive burden |
| **Exploratory** | 病灶大小分层、按预测解剖先验的 FP 分解、HD95 / ASSD / NSD@τ、bootstrap CI |

实现位置：`src/zonal_reliability_fusion/evaluation/`（**与模型完全解耦**，不 import torch / nnU-Net）。

命令：见下文「评估」小节（命令的**唯一**来源）。

## Experiment Status

**主实验（A → B → C → D）尚未开始运行。** 表格中的状态与 `docs/Training_Log.md` 的
「各 variant 状态」表为同一真源；本文档不维护第二份状态清单。

| 条件 | variant | 状态 |
|---|---|---|
| A1 | `positive_sampling` | 已完成（1000 ep + validation） |
| A2 | `dicece_positive_sampling` | 未训练（代码就绪） |
| Stage 1 | `anatomy_joint_100ep` | **已完成 100 ep + validation**；集合 Dice 0.6134，但逐区域 **WG 0.0056** / PZ 0.8985 / TZ 0.9361 ⇒ **WG 头不可用** |
| B | `lesion_roi` | 未训练；**当前被 Stage-1 的 WG 结果阻塞**（ROI 的唯一来源是 predicted WG） |
| C | `lesion_coarse_to_fine` | 未训练 |
| D | `lesion_zone_refine` | 未训练（需 Dataset606 的 zone 概率通道） |
| E | `lesion_hard_negative` | 未训练（需先做 Round-2 挖掘） |

已完成的**历史**运行（旧门控 / 特征融合线）见 `docs/Training_Log.md`；它们在新主线中的角色是
preliminary / negative evidence，说明见 `docs/archive/`。

## Training Commands

工作目录 `/opt/data/private/lm/my-projects`；conda 环境 **`lm`**；涉及 nnU-Net 的命令必须先
`source scripts/env_nnunet.sh`。**一次只运行一条命令；长任务由研究者本人执行。**

```bash
cd /opt/data/private/lm/my-projects
source /root/anaconda3/etc/profile.d/conda.sh && conda activate lm
source scripts/env_nnunet.sh
```

### 查看可选 variant（默认只显示 ACTIVE 主线）

```bash
python scripts/train/train_nnunet.py --help            # 只显示新主线条件
python scripts/train/train_nnunet.py --help --legacy   # 含归档的旧门控 / 融合条件
```

> **当前优先级提醒**：主实验的下一步**不是**直接把 A/B/C/D 全部排上。Stage-1 的 WG 头不可用会
> 阻塞条件 B，因此第一个待办是 **Stage-1 WG 的诊断与修正**（见
> `docs/experiments/anatomy_joint_100ep.md`）。`docs/Experiment_Plan.md` 描述的矩阵是完整计划，
> 不代表可以跳过这个前置。

### 阶段 0：strong baseline（A1 vs A2，唯一变量是损失）

```bash
# A2（A1 已完成，无需重跑）
nnUNet_compile=false CUDA_VISIBLE_DEVICES=0 python scripts/train/train_nnunet.py \
  dicece_positive_sampling 605 3d_fullres 0 \
  --seed 20261008 \
  --run-config outputs/run_configs/A2_dicece_positive_sampling_fold0_seed20261008.json
```

### 条件 B / C / D / E

```bash
# 条件 B 的前置：由预测 WG 构建 ROI 集合（不读 lesion GT、不读原始影像、不重采样）
python scripts/data/build_anatomy_roi_set.py \
  --prior-dir <Stage-1 的 validation 目录> \
  --lesion-dataset Dataset605_PICAI --fold 0 \
  --margin-mm 15 --wg-threshold 0.5 \
  --output workdir/anatomy_rois/Dataset605_PICAI_fold0.json

nnUNet_compile=false CUDA_VISIBLE_DEVICES=0 python scripts/train/train_nnunet.py \
  lesion_roi 605 3d_fullres 0 --seed 20261008 \
  --roi-set workdir/anatomy_rois/Dataset605_PICAI_fold0.json \
  --run-config outputs/run_configs/B_lesion_roi_fold0_seed20261008.json

nnUNet_compile=false CUDA_VISIBLE_DEVICES=0 python scripts/train/train_nnunet.py \
  lesion_coarse_to_fine 605 3d_fullres 0 --seed 20261008 \
  --roi-set workdir/anatomy_rois/Dataset605_PICAI_fold0.json \
  --run-config outputs/run_configs/C_lesion_c2f_fold0_seed20261008.json

nnUNet_compile=false CUDA_VISIBLE_DEVICES=0 python scripts/train/train_nnunet.py \
  lesion_zone_refine 606 3d_fullres 0 --seed 20261008 \
  --roi-set workdir/anatomy_rois/Dataset606_PICAI_Zonal_fold0.json \
  --run-config outputs/run_configs/D_lesion_zone_refine_fold0_seed20261008.json

# 条件 E 的前置：Round-2 困难负样本挖掘（只对训练 split）
python scripts/data/mine_hard_negatives.py \
  --prediction-dir outputs/predictions/round1_train_t2wadchbv \
  --reference-dir  workdir/nnUNet_raw/Dataset605_PICAI/labelsTr \
  --wg-dir         workdir/anatomy_priors/Dataset605_PICAI/train \
  --lesion-dataset Dataset605_PICAI --fold 0 \
  --confidence-threshold 0.5 --wg-threshold 0.5 --max-locations-per-case 8 \
  --output workdir/hard_negatives/Dataset605_PICAI_fold0_round1.json

nnUNet_compile=false CUDA_VISIBLE_DEVICES=0 python scripts/train/train_nnunet.py \
  lesion_hard_negative 605 3d_fullres 0 --seed 20261008 \
  --roi-set workdir/anatomy_rois/Dataset605_PICAI_fold0.json \
  --hard-negative-set workdir/hard_negatives/Dataset605_PICAI_fold0_round1.json \
  --run-config outputs/run_configs/E_lesion_hard_negative_fold0_seed20261008.json
```

### Stage 1（解剖先验生成器）

```bash
nnUNet_compile=false CUDA_VISIBLE_DEVICES=0 python scripts/train/train_nnunet.py \
  anatomy_joint_100ep 607 3d_fullres 0 --export-validation-probabilities
```

### 可选参数

```text
--seed <int>                显式随机种子（Python / NumPy / torch）；改善可重复性但**不**保证
                            bitwise 确定性（nnU-Net 的多进程增强顺序非确定）
--roi-set <path>            Anatomy-Guided ROI 集合（条件 B/C/D/E 必填）
--hard-negative-set <path>  困难负样本集合（条件 E 可选；不传则完全退化）
--run-config <path>         落盘本次运行的冻结配置（已存在则拒绝覆盖）
--continue-training         从同一输出目录的 checkpoint 续训
--validation-only           只运行 nnU-Net 原生 validation
--export-validation-probabilities  validation 时导出原生恢复概率（npz）
```

### 检查点与安全行为

- 输出目录由 **Trainer 类名**决定，入口**拒绝**从头覆盖已有目录，也**拒绝**在缺 checkpoint 时静默
  重训；
- 任一必需前置（ROI 集合、解剖 split、zone 通道数）缺失时**立即报错**，不静默退化。

## Evaluation Commands

```bash
# 1) split 泄漏检查（秒级，只读；任何 FAIL 必须先修复再训练）
python scripts/data/check_split_integrity.py \
  --anatomy-prior-dir outputs/nnUNet_results/Dataset607_PICAI_Anatomy/nnUNetTrainerPICAI_AnatomyJoint_100ep_NoFFT__nnUNetPlans__3d_fullres/fold_0/validation \
  --roi-set workdir/anatomy_rois/Dataset605_PICAI_fold0.json

# 2) summary 模式核对主终点（秒级，只读 summary.json）
python scripts/evaluate_segmentation.py \
  --model A1=outputs/nnUNet_results/Dataset605_PICAI/nnUNetTrainerPICAI_FLCE_PositiveSampling_NoFFT__nnUNetPlans__3d_fullres/fold_0 \
  --model A2=outputs/nnUNet_results/Dataset605_PICAI/nnUNetTrainerPICAI_DiceCE_PositiveSampling_NoFFT__nnUNetPlans__3d_fullres/fold_0 \
  --mode summary \
  --output outputs/reports/segmentation_metrics_A1_A2_summary.json

# 3) full 模式补充病灶实例、体积分层与解剖区域分解（长任务）
python scripts/evaluate_segmentation.py \
  --model A1=<fold_0 目录> --model A2=<fold_0 目录> --mode full \
  --nsd-tolerance-mm <研究者预先确定的值> \
  --volume-thresholds-mm3 500,1000 \
  --output outputs/reports/segmentation_metrics_A1_A2_full.json
```

`--nsd-tolerance-mm` 与 `--volume-thresholds-mm3` **必须**由研究者在运行前自行确定，工具
**不设**默认值，也不推荐取值。

## Prediction

```bash
python scripts/inference/predict_nnunet.py \
  -i <输入影像目录> -o <输出目录> \
  -m outputs/nnUNet_results/Dataset605_PICAI/<Trainer类名>__nnUNetPlans__3d_fullres \
  -f 0 -device cuda
```

Stage-1 解剖先验必须导出 soft probability：

```bash
python scripts/inference/predict_nnunet.py \
  -i <T2W 单通道目录> -o <先验输出目录> \
  -m outputs/nnUNet_results/Dataset607_PICAI_Anatomy/nnUNetTrainerPICAI_AnatomyJoint_100ep_NoFFT__nnUNetPlans__3d_fullres \
  -f 0 --save_probabilities -device cuda
```

其余参数与官方 `nnUNetv2_predict_from_modelfolder` 完全一致（`--help` 即官方帮助）。

## Repository Structure

```text
my-projects/
├── README.md                       # 本文件（新主线唯一入口）
├── AGENTS.md                       # 协作规则
├── pyproject.toml                  # 依赖声明 + ruff / pytest 配置
│
├── docs/
│   ├── Research_Plan.md            # 研究问题、假设、方法机制、边界
│   ├── Method.md                   # 方法实现细节与 fail-closed 清单
│   ├── Experiment_Plan.md          # 实验矩阵、预算、命令、stop rules
│   ├── Evaluation_Protocol.md      # 指标定义与统计口径（唯一来源）
│   ├── Training_Log.md             # 已发生的训练事实
│   ├── Findings.md                 # 有证据支持的结论
│   ├── Development_Log.md          # 工程变更
│   ├── REFACTOR_AUDIT.md           # 2026-10-08 重构前的只读审计快照
│   ├── experiments/                # 每个实验一份（variant 名即文件名）
│   └── archive/                    # 旧研究路线（只读）
│       ├── old_research_plans/
│       ├── legacy_gate_research/
│       └── historical_experiments/
│
├── src/zonal_reliability_fusion/   # 包名保留：为 checkpoint 与既有导入路径兼容
│   ├── anatomy/                    # Stage 1：契约、先验布局、预测校验、质量评价
│   ├── lesion/                     # Stage 2：ROI、lesionness、coarse-to-fine、zone 条件化
│   ├── sampling/                   # 阳性采样（baseline 组成）、困难负样本（条件 E）
│   ├── evaluation/                 # 评价体系（与模型解耦）
│   ├── nnunet/                     # nnU-Net 集成层（Trainer、损失、ROI 采样、运行时校验）
│   └── legacy/                     # 归档的 gate / fusion 实现（只读）
│
├── scripts/
│   ├── data/                       # 数据准备、解剖标签审计、ROI 集合、困难负样本、split 检查
│   ├── train/train_nnunet.py       # **唯一**训练入口
│   ├── inference/predict_nnunet.py # 官方预测入口 + Trainer 解析
│   ├── evaluate_segmentation.py    # 论文级评估（离线）
│   ├── evaluate_external_segmentation.py
│   └── env_nnunet.sh               # 环境脚本（必须 source）
│
├── tests/unit/                     # 纯合成 CPU 测试（不读真实医学数据）
│
├── data/  workdir/  outputs/  logs/   # 数据与产物（只读，不入库）
└── third_party/nnUNet/             # 固定 v2.6.2，**只读**
```

**刻意保留的不一致（研究逻辑优先于命名整齐）**：Python 包名仍为 `zonal_reliability_fusion`
（历史内部标识）。改名会让既有 checkpoint 的输出目录解析失效，且对研究本身无收益。当前包内容
已按研究阶段重新分层（`anatomy` / `lesion` / `sampling` / `evaluation` / `nnunet` / `legacy`）。
同理，`scripts/` 下既有脚本的路径保持不变（它们被 `Training_Log` / `Development_Log` 的历史记录
引用），新增能力放在同一目录下的新脚本里。

## Reproducibility

- **显式种子**：`--seed` 控制 Python random、NumPy、PyTorch、采样与模型初始化；
- **诚实边界**：

  > explicit seed improves repeatability but does not guarantee bitwise determinism.

  nnU-Net v2.6.2 的增强由多进程 `NonDetMultiThreadedAugmenter` 驱动，完成顺序非确定；本项目
  **不**声称 bitwise 可复现。
- **最终候选**：same fold / same budget / same evaluation / **3 explicit seeds**，报告 each run、
  `mean ± std` 与病例级 paired analysis；
- **短预算**（100 / 150 epoch）只用于 sanity check 与方向筛选，**不得**与 1000-epoch 结果直接声明
  性能优劣；
- **run 配置落盘**：`--run-config` 写出 dataset / fold / seed / trainer / network / loss / sampling /
  epochs / batch / patch / spacing / anatomy prior source / ROI / lesionness / hard-negative 设置；
- **split 泄漏自动检查**：`scripts/data/check_split_integrity.py`，遇泄漏 fail closed。

## Legacy Experiments

旧主线（输入级 / 特征级 modality gate、浅层序列特异特征融合、同区参照融合）已**归档**，保留作为
preliminary / negative evidence 与 checkpoint 兼容：

- 归档索引与使用规则 → `docs/archive/README.md`
- 旧研究线技术说明与表述纪律 → `docs/archive/legacy_gate_research/README.md`
- 旧研究计划 → `docs/archive/old_research_plans/`
- 旧实验文档 → `docs/archive/historical_experiments/`

一句话结论（**注意表述边界**）：

> Direct input-level modality reweighting did not provide a consistent segmentation
> improvement in the tested runs, motivating a shift toward lesion-aware anatomical modeling.

**不得**写成"gate 无效""证明 gate 没有作用""gate 降低性能"或"PZ/TZ 无用"：两次配对比较的病例级
Dice 变化 95% CI 均跨 0，且全部为单次运行；CI 跨 0 既不支持提升也不支持下降，更不证明等效。

旧条件**不再出现在主 README、默认训练命令或主实验流程中**；需要重现时加上 `--legacy`。

## Environment

- conda 环境 **`lm`**（`/root/anaconda3/envs/lm/bin/python`）；
- nnU-Net 固定版本 `third_party/nnUNet`（v2.6.2，commit `74ceb68`），**只读**；
- `third_party/nnUNet` 的 planning/preprocessing、dataloader 主体、augmentation 主体、
  deep supervision、optimizer/scheduler、training loop、checkpoint/resume、validation、
  sliding-window inference 与 prediction export 全部复用，**不重复实现**；
- 必须先 `source scripts/env_nnunet.sh`（把 `nnUNet_*` 指向项目内 `workdir/` 与 `outputs/`，
  并把 `PYTHONPATH` 钉到固定源码树）。

## Contributing Rules（摘要）

完整规则见 `AGENTS.md`。三条最容易违反的：

1. **只有用户的明确命令才能创建新文件**；
2. **不修改 `third_party/nnUNet`**；项目侧只做 `inherit / wrap / extend`；
3. **任何模块必须能回答**"是否减少漏检 / 是否改善覆盖 / 是否减少假阳"三者之一，否则不进主模型。
