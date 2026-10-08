# Experiment Plan

> 本文件记录**具体实验矩阵、预算、命令入口与 stop rules 的运行判据**。
> 科学问题与方法机制见 `docs/Research_Plan.md` 与 `docs/Method.md`；指标定义见
> `docs/Evaluation_Protocol.md`；已发生的运行事实见 `docs/Training_Log.md`。
>
> **本文件只描述计划。任何命令都必须由研究者本人执行；代理不启动长训练。**

---

## 1. 主线总览

```text
A  strong baseline
      ↓  只改训练 patch 采样空间
B  A + Anatomy-Guided ROI
      ↓  只加 coarse lesionness 头 + soft refinement + lesionness 辅助损失
C  B + Lesion-Aware Coarse-to-Fine
      ↓  只加两个 soft zone 概率通道（concat 模式）
D  C + Zone-Aware Refinement
      ↓  只加困难负样本采样（可开关）
E  best(C, D) + Anatomy-Constrained Hard Negative Mining
```

任何模块只有在能回答"是否减少漏检 / 是否改善覆盖 / 是否减少假阳"三者之一时才进入主模型。

---

## 2. 阶段 0：strong baseline 的确定（A1 vs A2）

**唯一变量：损失函数。**

| 臂 | variant | Trainer 类 | 损失 |
|---|---|---|---|
| A1 | `positive_sampling` | `nnUNetTrainerPICAI_FLCE_PositiveSampling_NoFFT` | PI-CAI `0.5*Focal(gamma=2)+0.5*CE`（项目实现） |
| A2 | `dicece_positive_sampling` | `nnUNetTrainerPICAI_DiceCE_PositiveSampling_NoFFT` | nnU-Net v2.6.2 原生 `DC_and_CE_loss` + `MemoryEfficientSoftDiceLoss` |

**必须逐项相同**（启动前逐条核对，任一不符即停止）：

- 网络：由 `nnUNetPlans.json` 构建的原生 `PlainConvUNet`（不含 gate / 浅层 feature path）；
- 数据集与 split：`Dataset605_PICAI` / `3d_fullres` / fold 0；
- patch size、batch size、前景采样比例、deep supervision 权重；
- 增强：同一默认管线 + 同一 NoFFT 修复；
- optimizer（SGD+Nesterov）、初始学习率、PolyLR 总周期、epoch 数；
- checkpoint 策略、validation、滑窗推理。

**启动日志必须显示**（任一不符即停止）：

```text
Dataset605 / fold 0 / 1277 train, 223 validation
batch_size=2 / positive_cases=362 / negative_cases=915
positive_cases_per_batch=1 / guaranteed_positive_patch_fraction=0.5
network=PlainConvUNet / input_channels=3
A2 额外：loss=DeepSupervisionWrapper (base=nnunetv2...DC_and_CE_loss)
```

**胜者成为后续所有条件的 Strong Baseline。** 比较依据以 primary endpoint
（`positive_case_macro_dice`）为主，并同时报告 key secondary（尤其 completely missed 与
FP burden）——若两者在 Dice 上接近而在失败模式上不同，需在实验文档中如实记录，不强行选出一个
"胜者"。

> **执行顺序的当前优先级**：Stage-1 的 anatomy 模型**已完成** 100 epoch + validation，但
> **WG 区域头不可用**（逐区域 Dice：WG 0.0056 / PZ 0.8985 / TZ 0.9361；集合平均 0.6134 掩盖了
> 这一点）。由于条件 B 的 ROI **唯一**来自 predicted WG，**必须先解决 WG 预测再启动 B**。
> 诊断与修正的入口见 `docs/experiments/anatomy_joint_100ep.md`；本文件其余内容描述完整矩阵，
> 不代表可以跳过该前置。

**命令模板**（工作目录 `/opt/data/private/lm/my-projects`，conda 环境 `lm`）：

```bash
cd /opt/data/private/lm/my-projects
source /root/anaconda3/etc/profile.d/conda.sh && conda activate lm
source scripts/env_nnunet.sh

nnUNet_compile=false CUDA_VISIBLE_DEVICES=0 python scripts/train/train_nnunet.py \
  positive_sampling 605 3d_fullres 0 \
  --seed 20261008 --run-config outputs/run_configs/A1_positive_sampling_fold0_seed20261008.json

nnUNet_compile=false CUDA_VISIBLE_DEVICES=0 python scripts/train/train_nnunet.py \
  dicece_positive_sampling 605 3d_fullres 0 \
  --seed 20261008 --run-config outputs/run_configs/A2_dicece_positive_sampling_fold0_seed20261008.json
```

> A1（`positive_sampling`）**已经完成**训练与 validation（见 `docs/Training_Log.md`）。
> 因此阶段 0 实际只需要按上面的 `--seed` 约定**补齐 A2**；若要严格对齐 seed，A1 也是
> 无种子运行，须在实验文档中如实说明该边界。

---

## 3. 条件 B：Anatomy-Guided ROI

**唯一变量：训练 patch 采样空间。**

| 项 | 值 |
|---|---|
| variant | `lesion_roi` |
| Trainer | `nnUNetTrainerPICAI_LesionROI_NoFFT` |
| 数据集 | `Dataset605_PICAI` / `3d_fullres` / fold 0 |
| 网络 | **原生 `PlainConvUNet`**（ROI 不改网络） |
| 损失 | 与 strong baseline 胜者**完全相同** |
| 采样 | 阳性采样（不变）+ 非前景槽位的 ROI 约束（`roi_sampling_probability = 0.75`，冻结） |

**前置（当前被阻塞）**：ROI 集合必须由 predicted WG 生成。Stage-1 的 WG 头目前不可用，因此
第一步是修正 Stage-1 并重训/重导出，**然后**才运行下面的 ROI 构建命令。

**前置：ROI 集合必须由预测 WG 生成**（`scripts/data/build_anatomy_roi_set.py`，长任务）：

```bash
# <Stage-1 的 validation 目录> = outputs/nnUNet_results/Dataset607_PICAI_Anatomy/
#   nnUNetTrainerPICAI_AnatomyJoint_100ep_NoFFT__nnUNetPlans__3d_fullres/fold_0/validation
# （该目录已含 223 例验证病例的 soft prior；训练 split 的先验需先由预测生成）
python scripts/data/build_anatomy_roi_set.py \
  --prior-dir <Stage-1 的 validation 目录> \
  --lesion-dataset Dataset605_PICAI --fold 0 \
  --margin-mm 15 --wg-threshold 0.5 \
  --output workdir/anatomy_rois/Dataset605_PICAI_fold0.json
```

训练：

```bash
nnUNet_compile=false CUDA_VISIBLE_DEVICES=0 python scripts/train/train_nnunet.py \
  lesion_roi 605 3d_fullres 0 --seed 20261008 \
  --roi-set workdir/anatomy_rois/Dataset605_PICAI_fold0.json \
  --run-config outputs/run_configs/B_lesion_roi_fold0_seed20261008.json
```

**回答的问题**：单纯减少无关背景搜索空间，是否已经能改善 lesion learning？

---

## 4. 条件 C：Lesion-Aware Coarse-to-Fine（核心方法比较）

**唯一变量：相对 B 加上 coarse lesionness 头 + soft 残差 refinement + lesionness 辅助损失。**

| 项 | 值 |
|---|---|
| variant | `lesion_coarse_to_fine` |
| Trainer | `nnUNetTrainerPICAI_LesionCoarseToFine_NoFFT` |
| 损失 | strong baseline 主损失 + `0.5 * lesionness`（`LESIONNESS_LOSS_WEIGHT = 0.5`） |
| lesionness 目标 | 物理半径膨胀（`LESIONNESS_DILATION_RADIUS_MM = 3.0`）的 coarse lesion mask |
| 采样 | 与 B 完全相同 |

```bash
nnUNet_compile=false CUDA_VISIBLE_DEVICES=0 python scripts/train/train_nnunet.py \
  lesion_coarse_to_fine 605 3d_fullres 0 --seed 20261008 \
  --roi-set workdir/anatomy_rois/Dataset605_PICAI_fold0.json \
  --run-config outputs/run_configs/C_lesion_c2f_fold0_seed20261008.json
```

**回答的问题**：显式 lesion localization 是否减少完全漏检，特别是小病灶漏检？
这是本论文**最重要的方法比较**。

---

## 5. 条件 D：Zone-Aware Refinement

**唯一变量：相对 C 加上两个 soft zone 概率通道（`concat` 模式）。**

| 项 | 值 |
|---|---|
| variant | `lesion_zone_refine` |
| Trainer | `nnUNetTrainerPICAI_LesionZoneRefine_NoFFT` |
| 数据集 | **`Dataset606_PICAI_Zonal`**（5 通道：T2W/ADC/HBV + P(PZ)/P(TZ)） |
| zone 模式 | `concat`（默认；`zone_experts` 只在 `concat` 被证明不足时才启用） |

```bash
nnUNet_compile=false CUDA_VISIBLE_DEVICES=0 python scripts/train/train_nnunet.py \
  lesion_zone_refine 606 3d_fullres 0 --seed 20261008 \
  --roi-set workdir/anatomy_rois/Dataset606_PICAI_Zonal_fold0.json \
  --run-config outputs/run_configs/D_lesion_zone_refine_fold0_seed20261008.json
```

**回答的问题**：在已经具有 lesion-aware coarse-to-fine learning 之后，PZ/TZ 是否能够进一步
改善病灶完整分割或控制 FP？

**只有 D 相对 C 有稳定收益，PZ/TZ 才作为最终方法贡献。若 D 无收益，最终模型停在 C，并立即
停止 anatomy 架构扩张。**

### 5.1 Dataset605 ↔ 606 的归因边界（必须写进实验文档）

条件 D 相对 C 的差异**不只是** PZ/TZ：两者数据集不同。前三个 MRI 通道已通过逐数组一致性审计
（`outputs/reports/dataset605_606_mri_equivalence_audit_v2.json`，PASS），但 **PZ/TZ 不参与判等**。
因此 D 相对 C 的结论必须带上这条边界，**不得**声称"唯一差异是 PZ/TZ"。

---

## 6. 条件 E：Anatomy-Constrained Hard Negative Mining

**唯一变量：相对 best(C, D) 加上困难负样本采样。**

| 项 | 值 |
|---|---|
| variant | `lesion_hard_negative` |
| Trainer | `nnUNetTrainerPICAI_LesionHardNegative_NoFFT` |
| 默认状态 | **关闭**（`hard_negative_set_path = None` ⇒ 完全退化为条件 C/D） |

**Round 2 前置：只对训练 split 挖掘**（长任务）：

```bash
python scripts/data/mine_hard_negatives.py \
  --prediction-dir outputs/predictions/round1_train_t2wadchbv \
  --reference-dir  workdir/nnUNet_raw/Dataset605_PICAI/labelsTr \
  --wg-dir         workdir/anatomy_priors/Dataset605_PICAI/train \
  --lesion-dataset Dataset605_PICAI --fold 0 \
  --confidence-threshold 0.5 --wg-threshold 0.5 --max-locations-per-case 8 \
  --output workdir/hard_negatives/Dataset605_PICAI_fold0_round1.json
```

训练：

```bash
nnUNet_compile=false CUDA_VISIBLE_DEVICES=0 python scripts/train/train_nnunet.py \
  lesion_hard_negative 605 3d_fullres 0 --seed 20261008 \
  --roi-set workdir/anatomy_rois/Dataset605_PICAI_fold0.json \
  --hard-negative-set workdir/hard_negatives/Dataset605_PICAI_fold0_round1.json \
  --run-config outputs/run_configs/E_lesion_hard_negative_fold0_seed20261008.json
```

**定位**：若效果稳定则进入最终方法；若效果有限，则作为 ablation / supplementary experiment，
**不强行塞进最终模型**。

---

## 7. 短预算探索机制（exploratory protocol）

**不要再所有想法一上来就跑 1000 epochs。** 任何新模块先做短预算筛选：

- 预算：**100 / 150 epoch**（同一批比较必须用同一预算）；
- 用途**只限**：sanity check、direction screening、obvious failure elimination；
- 必须满足：epoch 相同、iterations per epoch 相同、scheduler 总周期匹配、validation protocol 相同；
- **不得**与历史 1000-epoch 模型直接声明性能优劣；
- 候选通过短预算后才进入 full training。

先例：旧的四个 `*_100ep` 同区参照条件在**构造阶段**就失败（`KeyError: 'args'`），修复后从未重跑；
这说明"短预算先跑通"本身必须是一个显式步骤。已归档，仅供教训参考。

---

## 8. Full Training 规则

最终候选（**Strong baseline vs Final proposed model**）必须：

```text
same fold
same training budget
same evaluation protocol
3 explicit seeds
```

报告：each run、`mean ± std`、病例级 paired analysis（口径见 `docs/Evaluation_Protocol.md` §7）。

---

## 9. Stop Rules（运行判据）

| 规则 | 触发条件 | 行动 |
|---|---|---|
| **Rule 1** | B 相对 A：`positive_case_macro_dice`、`lesion_sensitivity_any_overlap`、`small_lesion_sensitivity_any_overlap` **三项都没改善** | **不要**通过继续增加 ROI attention 来"救"；记录为 negative evidence，短暂评估 ROI margin 取值后停止扩张 |
| **Rule 2** | C 相对 B：`completely_missed_lesion_rate` 未下降 **且** `lesion_sensitivity_any_overlap` 未提高 | coarse localization 假设未获支持；**不要**立即加入更复杂的 Transformer / Mamba；先检查 lesionness 目标与 guidance 是否真的生效（辅助损失是否在下降、coarse 头的指标是否合理） |
| **Rule 3** | D 相对 C 无稳定收益（配对 CI 跨 0，或收益小于 seed 间波动） | **立即停止 anatomy 架构扩张**：不继续 `AnatomyGate v2` / `CrossAttention` / `ZoneTransformer` / `ZoneMamba`；最终模型停在 C |
| **Rule 4** | 任何候选模块只有单次 exploratory run 的小幅 Dice 上升 | **不得**作为最终创新；必须 3 seed + `mean ± std` + 病例级 paired analysis |
| **Rule 5** | 某个模块无法回答"是否减少漏检 / 是否改善覆盖 / 是否减少假阳"三者之一 | 不进主模型 |
| **Rule 6** | 任何一步出现 split 泄漏（`scripts/data/check_split_integrity.py` 返回 FAIL） | **立即停止**，先修复泄漏再启动训练 |

### 9.1 判定用的具体指标

Rule 1–3 的"改善 / 无改善"判定一律使用：

- **主判据**：primary endpoint `positive_case_macro_dice` 的配对比较；
- **必要条件**：key secondary 中与该条件直接对应的失败模式指标（Rule 1/2 用
  `completely_missed_*` 与 `lesion_sensitivity_*`；Rule 3 用
  `matched_lesion_dice` 与 `false_positive_burden`）；
- **方向一致性**：主判据与其对应的失败模式指标必须**同向**，否则视为无稳定收益，不得择一报告。

---

## 10. 评价执行顺序（每个条件训练完成后）

一次只做**一步**，每步由研究者运行：

```bash
# 1) split 泄漏检查（秒级；任何 FAIL 都必须先修复）
python scripts/data/check_split_integrity.py \
  --anatomy-prior-dir outputs/nnUNet_results/Dataset607_PICAI_Anatomy/nnUNetTrainerPICAI_AnatomyJoint_100ep_NoFFT__nnUNetPlans__3d_fullres/fold_0/validation \
  --roi-set workdir/anatomy_rois/Dataset605_PICAI_fold0.json

# 2) summary 模式核对主终点（秒级）
python scripts/evaluate_segmentation.py \
  --model A1=<fold_0 目录> --model A2=<fold_0 目录> --mode summary \
  --output outputs/reports/segmentation_metrics_A1_A2_summary.json

# 3) full 模式补充病灶实例、体积分层与解剖区域分解（长任务）
python scripts/evaluate_segmentation.py \
  --model A1=<fold_0 目录> --model A2=<fold_0 目录> --mode full \
  --nsd-tolerance-mm <研究者预先确定的值> \
  --output outputs/reports/segmentation_metrics_A1_A2_full.json
```

`<fold_0 目录>` 形如
`outputs/nnUNet_results/Dataset605_PICAI/<Trainer类名>__nnUNetPlans__3d_fullres/fold_0`；
**命令与参数规范的唯一来源是 README 的「评估」小节**，本文件不维护第二份完整命令清单。

---

## 11. 每个实验必须留存的 run 配置

`--run-config <path>` 由训练入口写出（已存在则拒绝覆盖），字段包含：

```text
dataset / configuration / fold / variant / trainer / seed / seed_note
network / input_channels / num_input_channels / loss / sampling
epochs / batch_size / patch_size / spacing
anatomy_prior_source / roi_setting / lesionness_setting / hard_negative_setting
```

该文件与 `docs/Training_Log.md` 的条目、`docs/experiments/<variant>.md` 单实验文档三者共同构成
一次实验的完整记录。

---

## 12. 外部测试（Prostate158）

- 定位：**external distribution-shift stress test**；
- 必须用**冻结的 anatomy model**在外部数据上生成 anatomy prior；
- **禁止**人工修改外部 prior、使用 GT zone、或在看到 test 结果后重新调参数；
- 使用 `scripts/evaluate_external_segmentation.py`，指标口径与主实验一致。
