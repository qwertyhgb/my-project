# Training Log

训练与验证的运行事实日志。只记录真实运行（命令、时间、状态、关键结果、产物路径），不含协议/门控叙述。
代码变更见 `docs/Development_Log.md`。所有长任务由研究者本人运行；本文件在每次运行后更新。

---

## 2026-10-08 — 项目重构：新主线确立，主实验尚未运行

本节登记**重构本身造成的状态变化**，不包含任何新的训练结果。

- **旧线四个 `*_100ep` 条件原地归档**：构造阶段失败的根因已修复（`__init__` 反射问题），但
  **修复后从未重跑**，因此仍然零 epoch、零 checkpoint、零结果。它们现在归入 `--legacy` 视图，
  不再是主线。
- **新主线（Anatomy-Guided Lesion-Aware Coarse-to-Fine）的全部条件均未训练**：

| variant | 数据集 | 状态 | 输出目录 |
|---|---|---|---|
| `positive_sampling`（条件 A1） | Dataset605 / 3d_fullres / 0 | **已完成**（历史运行，无显式 seed；见下） | `outputs/nnUNet_results/Dataset605_PICAI/nnUNetTrainerPICAI_FLCE_PositiveSampling_NoFFT__nnUNetPlans__3d_fullres/fold_0/` |
| `dicece_positive_sampling`（条件 A2） | Dataset605 / 3d_fullres / 0 | 代码就绪，**未训练** | 目标：`.../nnUNetTrainerPICAI_DiceCE_PositiveSampling_NoFFT__nnUNetPlans__3d_fullres/fold_0/`（**未创建**） |
| `anatomy_joint_100ep`（Stage 1） | Dataset607 / 3d_fullres / 0 | **已完成**（训练 + validation；**WG 头不可用**，见下） | `outputs/nnUNet_results/Dataset607_PICAI_Anatomy/nnUNetTrainerPICAI_AnatomyJoint_100ep_NoFFT__nnUNetPlans__3d_fullres/fold_0/` |
| `lesion_roi`（条件 B） | Dataset605 / 3d_fullres / 0 | 代码就绪，**未训练** | 目标：`.../nnUNetTrainerPICAI_LesionROI_NoFFT__nnUNetPlans__3d_fullres/fold_0/`（**未创建**） |
| `lesion_coarse_to_fine`（条件 C） | Dataset605 / 3d_fullres / 0 | 代码就绪，**未训练** | 目标：`.../nnUNetTrainerPICAI_LesionCoarseToFine_NoFFT__nnUNetPlans__3d_fullres/fold_0/`（**未创建**） |
| `lesion_zone_refine`（条件 D） | Dataset606 / 3d_fullres / 0 | 代码就绪，**未训练**（需 Dataset606 的 zone 概率通道） | 目标：`.../nnUNetTrainerPICAI_LesionZoneRefine_NoFFT__nnUNetPlans__3d_fullres/fold_0/`（**未创建**） |
| `lesion_hard_negative`（条件 E） | Dataset605 / 3d_fullres / 0 | 代码就绪，**未训练**（需先做 Round-2 挖掘） | 目标：`.../nnUNetTrainerPICAI_LesionHardNegative_NoFFT__nnUNetPlans__3d_fullres/fold_0/`（**未创建**） |

- **前置产物的实际状态**：
  - 验证 split 的解剖 **soft prior 已经存在**（Stage-1 的 `validation/*.npz`，223 例），
    但**训练 split 的先验尚未生成**（`workdir/anatomy_priors/**` 不存在）；
  - `workdir/anatomy_rois/**`（ROI 集合）与 `workdir/hard_negatives/**`（困难负样本集合）
    在本次重构结束时**均不存在**。
- **本条的更正说明**：重构过程中最初把 Stage-1 记为"未训练"。经核查
  `outputs/nnUNet_results/Dataset607_PICAI_Anatomy/` 后确认它**已完成 100 epoch 训练与
  validation**（2026-10-06）。事实与逐区域结果见下一小节。

### Stage-1（`anatomy_joint_100ep`）的真实结果

- 数据集：`Dataset607_PICAI_Anatomy` / `3d_fullres` / fold 0（train 1276 / val 223 study）；
- 起止（UTC）：2026-10-06 11:02:00 → 12:28:20 训练完成 100 epoch；validation 至 12:38:15；
- 输出目录：`outputs/nnUNet_results/Dataset607_PICAI_Anatomy/nnUNetTrainerPICAI_AnatomyJoint_100ep_NoFFT__nnUNetPlans__3d_fullres/fold_0/`
  （`checkpoint_final.pth`、`checkpoint_best.pth`、`progress.png`、`debug.json`、
  `training_log_2026_10_6_11_01_59.txt`、`validation/`）。**勿删除 / 勿覆盖。**
- `validation/` 含 223 例的 `.npz`（soft probability，键 `probabilities`，形状 `(3,Z,Y,X)`）、
  223 例 `.pkl`（物理元数据）、223 例 `.nii.gz`（原生 region 导出）与 `summary.json`。
  即**验证 split 的解剖 soft prior 已经存在**。
- `Mean Validation Dice = 0.6134037658907373`——但这是**三个 region 的算术平均**，逐区域为：

| region | 标签集合 | Dice | TP | FP | FN | n_ref（逐例均值） | n_pred（逐例均值） |
|---|---|---:|---:|---:|---:|---:|---:|
| WG | `[1,3,5,7]` | **0.00558** | 334.79 | 282.83 | 121303.20 | 121637.99 | 617.63 |
| PZ | `[2,3,6,7]` | 0.89849 | 34955.74 | 3496.52 | 4224.70 | 39180.44 | 38452.26 |
| TZ | `[4,5,6,7]` | 0.93614 | 83959.15 | 4893.40 | 4946.05 | 88905.20 | 88852.55 |

- **关键观察（必须如实记录）**：**WG 区域头实际不可用**。GT 的 WG 参考体积并不小
  （`n_ref` 逐例均值 121637 voxel，与 PZ ∪ TZ 量级相当），但模型只预测出约 618 voxel/例，
  因此 Dice ≈ 0.006。集合平均的 0.6134 **掩盖**了这一点。
- **对主线的直接影响**：条件 B 的 Anatomy-Guided ROI **以 predicted WG 为唯一来源**，因此
  在当前 Stage-1 权重下**不可运行**（空 WG 预测会触发 `full_fov` 回退，退化成"没有 ROI"）。
  因此下一步不是启动 B，而是**先解决 WG 预测**（诊断 + 必要的数据/损失修正 + 重训 Stage-1）。
- 归因边界：本机无法从已有产物判断 WG 失败属于「优化/损失」还是「位编码与 region 定义下的
  数据问题」；两者都可能。诊断入口（由研究者运行）：
  `python -c "from zonal_reliability_fusion.anatomy.validation import anatomy_region_metrics"` 对应的
  解剖区域评估命令见 README「评估」小节；数据侧用
  `scripts/data/audit_prostate_anatomy_labels.py`（已存在，只读）。**本文件不预判原因。**
- 既有审计报告（只读）：`outputs/reports/prostate_anatomy_labels_audit_v2.json`
  （`inputs` stage：1500 例中 1499 例有 WG，1 例为已知缺失 `11050_1001070`；
  `source_audit_status = unresolved`，35 例"无可比较候选"）。


- **本次未启动**训练、validation、推理或任何数据处理；未创建任何模型目录或 checkpoint；
  未删除或覆盖任何既有产物。
- 条件 A1（`positive_sampling`）在时间上属于**历史运行**，其训练与命令记录在下方条目；
  它是新主线的 strong baseline 候选之一，但**没有显式 seed**，这一边界必须保留。

---

## 2026-10-08 — 串行队列运行事实、用户停止与 100ep 构造失败

队列命令（用户于 2026-10-06 14:10 起在后台串行执行，逐项 `tee` 到 `outputs/logs/<variant>.log`）：

| variant | 数据集 | 起止 | 结果 |
|---|---|---|---|
| feature_no_gate_positive_sampling_100ep | 606 | 14:11:00 → 14:11:05 | exit 1，构造阶段 `KeyError: 'args'`，零 epoch、无产物 |
| feature_anatomy_gate_positive_sampling_100ep | 606 | 14:11:05 → 14:11:12 | 同上 |
| zonal_reference_positive_sampling_100ep | 606 | 14:11:12 → 14:11:18 | 同上 |
| zonal_reference_adaptive_positive_sampling_100ep | 606 | 14:11:18 → 14:11:23 | 同上 |
| feature_anatomy_gate_positive_sampling | 606 | 2026-10-06 14:11:23 → 2026-10-07 07:57:23 | exit 0，1000 epoch 完成并 validation |
| anatomy_gate | 606 | 2026-10-07 07:57:23 → 2026-10-07 23:06:40 | exit 0，1000 epoch 完成并 validation |
| feature_image_gate_positive_sampling | 605 | 2026-10-07 23:06:40 → 2026-10-08 00:44 | **用户要求停止**，中断于 epoch 81 完成、epoch 82 起始 |

- **feature_anatomy_gate_positive_sampling（Dataset606）**：训练 1000 epoch 完成，
  `checkpoint_final.pth` 已生成；Mean Validation Dice = **0.20868**（223 例）。产物目录
  `outputs/nnUNet_results/Dataset606_PICAI_Zonal/nnUNetTrainerPICAI_FeatureAnatomyGate_PositiveSampling_NoFFT__nnUNetPlans__3d_fullres/fold_0/`。
- **anatomy_gate（Dataset606）**：训练 1000 epoch 完成（15 h 09 min），`checkpoint_final.pth` 已生成；
  Mean Validation Dice = **0.18267**（223 例）。产物目录
  `outputs/nnUNet_results/Dataset606_PICAI_Zonal/nnUNetTrainerPICAI_AnatomyGate__nnUNetPlans__3d_fullres/fold_0/`。
- **feature_image_gate_positive_sampling（Dataset605）**：用户要求停止全部训练后，以 SIGTERM 终止队列
  进程组（PID 314998）及其全部子进程；训练停在 epoch 81 完成处（best EMA pseudo Dice 0.3142，
  epoch 81 pseudo dice 0.5587），`checkpoint_best.pth`（00:44）与 `checkpoint_latest.pth`（00:01）
  均在，未生成 `checkpoint_final.pth`、无 validation。**可用原生 continue-training 续训，
  既有产物未删除或覆盖。**
- **四个 `*_100ep` 变体**：均在 Trainer 构造阶段失败，`outputs/nnUNet_results/Dataset606_PICAI_Zonal/`
  下无对应输出目录，无 checkpoint。根因与修复见 `docs/Development_Log.md` 同日条目；修复仅经静态与
  反射验证，**修复后尚未重跑，100 epoch 能否跑完未验证**。
- 本节仅为运行事实登记；本次未启动训练、validation、推理或数据处理，仅执行进程终止与产物核对。

---

## 2026-10-06 — 现有产物核对与新分支状态

- 原 `feature_no_gate_positive_sampling` / Dataset605 / fold 0 已有 2026-09-28 启动日志、
  `checkpoint_best.pth` 与 `checkpoint_latest.pth`。现有日志最后完整 epoch 为 335，
  后续记录 epoch 336 起始；无 `checkpoint_final.pth` 或 `validation/summary.json`。
  这是既有运行的产物核对，不是本次新启动训练；停止原因未据日志确认。
- 新增四个 `_100ep` 融合条件均为代码就绪、未运行：
  `feature_no_gate_positive_sampling_100ep`、`feature_anatomy_gate_positive_sampling_100ep`、
  `zonal_reference_positive_sampling_100ep`、`zonal_reference_adaptive_positive_sampling_100ep`。
  本次未启动训练、真实验证、推理或批量影像评估，无新 checkpoint 或结果可引用。

---

## N0 — baseline（已完成）

- variant / Trainer：`baseline` → `nnUNetTrainerPICAI_FLCE_NoFFT`
- 数据集 / 配置 / fold：Dataset605_PICAI / `3d_fullres` / fold 0
- 输入通道顺序：T2W、ADC、HBV（3 通道）
- 损失：PI-CAI 官方风格 `0.5*Focal(gamma=2) + 0.5*CE`；NoFFT 修复（blur benchmark 关闭）
- 训练：1000 epoch 已完成；`checkpoint_final.pth` 已生成
- 验证：nnU-Net `perform_actual_validation` 完成，223 例（**GT 阳性 63 例 / GT 阴性 160 例**）
- **nnU-Net Mean Validation Dice = 0.1682**（`validation/summary.json` 的 `foreground_mean.Dice`
  = 0.16816659）。**该数不是「阳性病例 Dice」，引用时必须带分母**，口径如下：

  nnU-Net v2.6.2 只在 `tp + fp + fn == 0` 时把逐例 Dice 记为 NaN
  （`third_party/nnUNet/nnunetv2/evaluation/evaluate_predictions.py:106`），聚合时对逐例值取
  `np.nanmean`。故 223 例中实际进入平均的只有 **74 例** = 63 例 GT 阳性 + **11 例 GT 阴性但因吐出
  假阳预测而被记 Dice = 0**；其余 149 例「GT 阴性且预测为空」记 NaN 被剔除。`mean` 与
  `foreground_mean` 数值完全相同（本数据集仅一个前景类），该字段**不含**任何解剖分区平均的含义。

  | 口径 | 分母 | Dice |
  |---|---|---|
  | **A** `foreground_mean.Dice`（README/日志所引用的就是这个） | 74 例 = 63 阳性 + 11 假阳性阴性 | **0.1682** |
  | B GT 阳性例 case-wise mean | 63 例 | 0.1975 |
  | C GT 阳性**且被检出**（Dice > 0）例 case-wise mean | 29 例 | 0.4291（micro 0.6908） |
  | D GT 阳性例 micro（体素加权） | 63 例 | 0.5212 |
  | E 全部 223 例、把 NaN 记 0 | 223 例 | 0.0558 |

- 63 例 GT 阳性的逐例 Dice 是**双峰分布**，而非「整体偏低」：34 例 Dice 恰为 0（其中 **32 例整例
  预测为空**），29 例 Dice > 0；四分位 `[0, 0, 0.411, 0.658]`，median = 0.0，max = 0.8689。
  单一均值把「整例漏检」和「分割质量」混为一谈，**不能据 A 判断模型好坏**。
- **不可与 legacy M0 直接比较**：M0 记录的 `positive case-wise mean Dice = 0.2104` 分母是阳性例，
  对应上表 **B**（N0 = 0.1975，N0 略低）；M0 的 `micro Dice = 0.5098` 对应上表 **D**
  （N0 = 0.5212，N0 略高）。两个方向不一致，因此「0.1682 < 0.2104 所以 N0 更差」是**误读** ——
  两者口径不同。M0 的数出自已删除的旧自写验证器，其逐例定义只能回 Git 历史核对，与 nnU-Net
  validation 的导出/几何细节未逐项对齐；任何跨 N0/M0 的结论都须先统一到同一口径。
- validation probabilities：**未导出**（如需，重跑 validation 时加 `--export-validation-probabilities`）
- 输出目录（勿删除/覆盖）：
  `outputs/nnUNet_results/Dataset605_PICAI/nnUNetTrainerPICAI_FLCE_NoFFT__nnUNetPlans__3d_fullres/fold_0/`
- 复现 / 续训 / 只验证 / 导出概率的命令见 README「训练与推理命令」。

---

## legacy M0 — 旧自写 Residual-Encoder 管线（已中断，仅历史保留）

- 性质：**旧自写训练框架**（非 nnU-Net-first），本次重构后不再续训、不再用于任何结论。
- 启动：2026-09-20 09:32 UTC；由用户于 2026-09-21 停止。
- 最后完整记录：**epoch 389**（即完成 390/1000）。
- 最好一次完整验证：**epoch 349**
  - positive case-wise mean Dice = 0.2104
  - micro Dice = 0.5098
  - negative FP case rate = 31.25%
- checkpoint（UTC 时间戳，均在磁盘上核实）：
  - `checkpoint_last.pth` — 2026-09-21 01:24:05
  - `checkpoint_interrupt.pth` — 2026-09-21 01:24:50
  - `checkpoint_best.pth` — 2026-09-20 23:51:03（对应 epoch 349 附近）
  - `checkpoint_epoch_0349.pth` — 2026-09-20 23:51:06
- 产物路径（**不得删除或覆盖**）：
  - `outputs/checkpoints/m0_resenc/m0_resenc_picai_3d_fullres_v23/checkpoints/`
  - `outputs/metrics/m0/`
  - `outputs/diagnostics/m0/`
- 说明：驱动这些产物的旧代码（自写 trainer/checkpoint/滑窗/验证器/整网）已在本次重构中删除；
  产物本身按用户要求完整保留，仅供历史查阅。Git 历史承担代码归档职责。

---

## 各 variant 状态

命令统一见 README「训练与推理命令」；运行后按下方模板在此登记真实事实。

| variant | 数据集/配置/fold | 状态 | 输出目录 |
|---|---|---|---|
| baseline | Dataset605 / 3d_fullres / 0 | 已完成（见上） | `outputs/nnUNet_results/Dataset605_PICAI/nnUNetTrainerPICAI_FLCE_NoFFT__nnUNetPlans__3d_fullres/fold_0/` |
| optimized_baseline | Dataset605 / 3d_fullres / 0 | **用户已停止 / 中止（见下）** | `outputs/nnUNet_results/Dataset605_PICAI/nnUNetTrainerPICAI_DiceCE_NoFFT__nnUNetPlans__3d_fullres/fold_0/` |
| image_gate | Dataset605 / 3d_fullres / 0 | 已完成 | `outputs/nnUNet_results/Dataset605_PICAI/nnUNetTrainerPICAI_ImageGate__nnUNetPlans__3d_fullres/fold_0/` |
| anatomy_gate | Dataset606 / 3d_fullres / 0 | **已完成**（训练 + validation，见 2026-10-08 节） | `outputs/nnUNet_results/Dataset606_PICAI_Zonal/nnUNetTrainerPICAI_AnatomyGate__nnUNetPlans__3d_fullres/fold_0/` |
| positive_sampling | Dataset605 / 3d_fullres / 0 | **已完成**（训练 + validation，见下） | `outputs/nnUNet_results/Dataset605_PICAI/nnUNetTrainerPICAI_FLCE_PositiveSampling_NoFFT__nnUNetPlans__3d_fullres/fold_0/` |
| image_gate_positive_sampling | Dataset605 / 3d_fullres / 0 | **已完成**（训练 + validation，见下） | `outputs/nnUNet_results/Dataset605_PICAI/nnUNetTrainerPICAI_ImageGate_PositiveSampling_NoFFT__nnUNetPlans__3d_fullres/fold_0/` |
| anatomy_gate_positive_sampling | Dataset606 / 3d_fullres / 0 | **已完成**（训练 + validation，见下） | `outputs/nnUNet_results/Dataset606_PICAI_Zonal/nnUNetTrainerPICAI_AnatomyGate_PositiveSampling_NoFFT__nnUNetPlans__3d_fullres/fold_0/` |
| feature_no_gate_positive_sampling | Dataset605 / 3d_fullres / 0 | **已启动，缺最终 validation**（2026-10-06 产物核对见首节） | `outputs/nnUNet_results/Dataset605_PICAI/nnUNetTrainerPICAI_FeatureNoGate_PositiveSampling_NoFFT__nnUNetPlans__3d_fullres/fold_0/` |
| feature_image_gate_positive_sampling | Dataset605 / 3d_fullres / 0 | **用户已停止 / 中断于 epoch 81**（见 2026-10-08 节） | `outputs/nnUNet_results/Dataset605_PICAI/nnUNetTrainerPICAI_FeatureImageGate_PositiveSampling_NoFFT__nnUNetPlans__3d_fullres/fold_0/` |
| feature_anatomy_gate_positive_sampling | Dataset606 / 3d_fullres / 0 | **已完成**（训练 + validation，见 2026-10-08 节） | `outputs/nnUNet_results/Dataset606_PICAI_Zonal/nnUNetTrainerPICAI_FeatureAnatomyGate_PositiveSampling_NoFFT__nnUNetPlans__3d_fullres/fold_0/` |

### image_gate（已完成）

- variant / Trainer：`image_gate` → `nnUNetTrainerPICAI_ImageGate`
- 数据集 / 配置 / fold：Dataset605_PICAI / `3d_fullres` / fold 0（1277 train / 223 val）
- 命令：`python scripts/train/train_nnunet.py image_gate 605 3d_fullres 0 --device cuda`
- 训练：2026-09-21 08:23 UTC → 2026-09-22 00:11 UTC（**15 h 48 min**）；1000 epoch 完成，
  `checkpoint_final.pth` 已生成
- 验证：2026-09-22 00:11 → 00:22 UTC；223 例；使用 `checkpoint_final.pth`（epoch 999）
- **nnU-Net Mean Validation Dice = 0.17025**（`validation/summary.json` 的 `foreground_mean.Dice`，
  223 例）
- 检出结构：阳性 63 例中有任意重叠 30 例（47.6%）、完全无重叠 33 例；阴性 160 例中 17 例出现预测
- validation probabilities：**未导出**
- 输出目录（勿删除/覆盖）：
  `outputs/nnUNet_results/Dataset605_PICAI/nnUNetTrainerPICAI_ImageGate__nnUNetPlans__3d_fullres/fold_0/`
- 详细记录：`docs/archive/historical_experiments/image_gate.md`

### optimized_baseline（**用户已停止 / 中止**）

- variant / Trainer：`optimized_baseline` → `nnUNetTrainerPICAI_DiceCE_NoFFT`
- 数据集 / 配置 / fold：Dataset605_PICAI / `3d_fullres` / fold 0
- 损失：nnU-Net v2.6.2 **原生 Dice+CE**（项目无自写损失；除损失外其余训练机制与 baseline 一致）
- 命令：`python scripts/train/train_nnunet.py optimized_baseline 605 3d_fullres 0 --device cuda`
- 启动：2026-09-22 00:48 UTC（`CUDA_VISIBLE_DEVICES=0`）；**由用户手动停止**
- 日志：`outputs/nnUNet_results/Dataset605_PICAI/nnUNetTrainerPICAI_DiceCE_NoFFT__nnUNetPlans__3d_fullres/fold_0/training_log_2026_9_22_00_48_55.txt`
- 停止时的运行事实（来自训练日志）：
  - 最后完整完成的 epoch：**376**；epoch 377 已经开始但没有完成（日志停在 epoch 377 起始行）；
  - 最大 pseudo Dice 约 **0.0008**，从未超过 0.05，大多数 epoch 为 0；
  - 未出现 NaN、学习率异常、MRO 错误或 Trainer 接线错误；
  - 未执行 actual validation（日志中无 validation 段落），**没有 `validation/summary.json`**。
- 状态定性：这是**严重硬前景/全背景塌缩**下的中止运行，**不是完成的训练，不是正式模型结果**；
  不得作为结论引用，也不得从它的 checkpoint 续训。
- checkpoint（**不得删除或覆盖**）：
  `checkpoint_best.pth`、`checkpoint_latest.pth`（该目录下另有 `debug.json`、`progress.png`）
- 输出目录（勿覆盖、勿读取其未完成产物）：
  `outputs/nnUNet_results/Dataset605_PICAI/nnUNetTrainerPICAI_DiceCE_NoFFT__nnUNetPlans__3d_fullres/fold_0/`

### positive_sampling（**已完成**：训练 + actual validation）

- variant / Trainer：`positive_sampling` → `nnUNetTrainerPICAI_FLCE_PositiveSampling_NoFFT`
- 数据集 / 配置 / fold：Dataset605_PICAI / `3d_fullres` / fold 0（1277 train / 223 val）
- 设计：与 `baseline` 唯一差异是训练集病例/patch 采样（每批固定一个阳性病灶 patch；
  验证 loader 保持原生）。
- 命令：`python scripts/train/train_nnunet.py positive_sampling 605 3d_fullres 0`
- 日志：
  `outputs/nnUNet_results/Dataset605_PICAI/nnUNetTrainerPICAI_FLCE_PositiveSampling_NoFFT__nnUNetPlans__3d_fullres/fold_0/training_log_2026_9_22_07_16_28.txt`
- 启动时采样摘要（来自训练日志）：`training_cases=1277`、`positive_cases=362`、`negative_cases=915`、
  `positive_cases_per_batch=1`、`batch_size=2`、`guaranteed_positive_patch_fraction=0.5`
- 训练：2026-09-22 07:16 UTC → 22:31 UTC（**15 h 15 min**）；**1000 epoch 全部完成**，
  `checkpoint_final.pth` 已生成
- 优化过程观察（**只描述优化过程，不是病例级指标**）：启动后约 8 分钟出现首个非零 per-epoch
  pseudo Dice（0.0004，07:24 UTC）；最佳 EMA pseudo Dice = **0.6157**（epoch 792，19:22 UTC）。
- 验证：2026-09-22 22:31 → 22:42 UTC（约 11 min）；223 例；使用 `checkpoint_final.pth`（epoch 999）
- **nnU-Net Mean Validation Dice = 0.207834**（`validation/summary.json` 的 `foreground_mean.Dice`，
  223 例；同文件 `foreground_mean.IoU = 0.150146`）
- 检出结构（同一 `summary.json` 逐例统计）：阳性 63 例中有预测 47 例（**74.6%**）、有任意重叠
  42 例（**66.7%**）；阴性 160 例中产生预测 30 例（18.8%）
- validation probabilities：**未导出**（`validation/` 下无 `.npz`）
- checkpoint（**勿删除/覆盖**）：`checkpoint_final.pth`、`checkpoint_best.pth`
- 输出目录（勿删除/覆盖）：
  `outputs/nnUNet_results/Dataset605_PICAI/nnUNetTrainerPICAI_FLCE_PositiveSampling_NoFFT__nnUNetPlans__3d_fullres/fold_0/`
- 边界：`0.207834` 是 nnU-Net 的 `nanmean` 逐例口径（含假阳阴性病例的 0 分），与 `baseline`
  的 `0.168166` 是**同一口径**下可比；但它仍是"固定 0.5 阈值 + 体素 Dice"这一与 PI-CAI 官方
  评估口径不同的打分方式，**不得**据此单独声称某个机制的收益。
- 详细记录：`docs/experiments/positive_sampling.md`

### image_gate_positive_sampling（**已完成**：训练 + actual validation）

- variant / Trainer：`image_gate_positive_sampling` → `nnUNetTrainerPICAI_ImageGate_PositiveSampling_NoFFT`
- 数据集 / 配置 / fold：Dataset605_PICAI / `3d_fullres` / fold 0（1277 train / 223 val，与
  `positive_sampling` 同一划分）
- 设计：与 `positive_sampling` **唯一**差异是输入级 3→8→3 image gate（末层零初始化）；
  旧输入级 MRI 门控问题的公平匹配臂；现归入初步实验
- 命令：`python scripts/train/train_nnunet.py image_gate_positive_sampling 605 3d_fullres 0 --device cuda`
- 日志：
  `outputs/nnUNet_results/Dataset605_PICAI/nnUNetTrainerPICAI_ImageGate_PositiveSampling_NoFFT__nnUNetPlans__3d_fullres/fold_0/training_log_2026_9_23_06_55_30.txt`
- 训练：2026-09-23 06:55 UTC → 22:35 UTC（**15 h 40 min**）；**1000 epoch 全部完成**，
  `checkpoint_final.pth` 已生成；无 warning / error / NaN
- 验证：2026-09-23 22:36 → 22:47 UTC（约 11 min）；223 例；使用 `checkpoint_final.pth`（epoch 999）
- **nnU-Net Mean Validation Dice = 0.20935**（`foreground_mean.Dice` = 0.20934826；同文件 IoU
  = 0.14883646）；该数字的分母为 **90 例** = 63 阳性 + **27 例假阳阴性**（133 例真阴记 NaN 被剔除）
- 阳性病例口径（与 `positive_sampling` 同口径）：macro Dice **0.299069**（median 0.288758）、
  micro Dice **0.539691**、体素召回 **0.398901**、`positive_voxel_precision` **0.834071**、
  `all_prediction_voxel_precision` **0.746372**
- 检出结构：阳性 63 例中有任意重叠 **40 例**（63.5%）、完全无重叠 **23 例**（其中 21 例整例无预测）；
  阴性 160 例中 27 例出现预测（假阳体素合计 32,575）
- 与 `positive_sampling` 的配对比较（旧输入级 RQ1）登记在 `docs/Findings.md` §3.10：配对均值 delta
  = −0.0077、CI95 含 0，**未观察到明确的分割增量效用**，但工作点更保守（precision 升高、
  假阳减少、漏检略增）
- validation probabilities：**未导出**
- checkpoint（**勿删除/覆盖**）：`checkpoint_final.pth`、`checkpoint_best.pth`
- 输出目录（勿删除/覆盖）：
  `outputs/nnUNet_results/Dataset605_PICAI/nnUNetTrainerPICAI_ImageGate_PositiveSampling_NoFFT__nnUNetPlans__3d_fullres/fold_0/`
- 详细记录：`docs/archive/historical_experiments/image_gate_positive_sampling.md`

### anatomy_gate_positive_sampling（**已完成**：训练 + actual validation）

- variant / Trainer：`anatomy_gate_positive_sampling` →
  `nnUNetTrainerPICAI_AnatomyGate_PositiveSampling_NoFFT`
- 数据集 / 配置 / fold：Dataset606_PICAI_Zonal / `3d_fullres` / fold 0（1277 train / 223 val；
  5 通道 = T2W/ADC/HBV + PZ/TZ，PZ/TZ 只进门控、不进分割 backbone）
- 设计：与 `image_gate_positive_sampling` 的**预期主要差异**是门控条件（PZ/TZ）与数据集
  （606 vs 605）；损失、采样、增强、optimizer、LR scheduler 与随机种子策略一致
- 旧输入级解剖条件比较的归因前置条件：Dataset605/606 前三 MRI 通道逐数组审计已通过
  （`outputs/reports/dataset605_606_mri_equivalence_audit_v2.json`，
  `status = MRI_ARRAY_AUDIT_PASS`，2026-09-24 03:33 UTC，1500 例，0 mismatch），
  早于本训练启动（同日 06:42 UTC）
- 命令：`CUDA_VISIBLE_DEVICES=0 python scripts/train/train_nnunet.py anatomy_gate_positive_sampling 606 3d_fullres 0`
- 日志：
  `outputs/nnUNet_results/Dataset606_PICAI_Zonal/nnUNetTrainerPICAI_AnatomyGate_PositiveSampling_NoFFT__nnUNetPlans__3d_fullres/fold_0/training_log_2026_9_24_06_42_38.txt`
- 启动时采样摘要（来自训练日志）：`training_cases=1277`、`positive_cases=362`、`negative_cases=915`、
  `positive_cases_per_batch=1`、`batch_size=2`、`guaranteed_positive_patch_fraction=0.5`
- 训练：2026-09-24 06:42 UTC → 22:41 UTC（**15 h 58 min 56 s**）；**1000 epoch 全部完成**，
  `checkpoint_final.pth` 已生成；无 warning / error / NaN；epoch 耗时 mean 55.83 s
- 优化过程观察（**只描述优化过程，不是病例级指标**）：首个非零 per-epoch pseudo Dice 在 epoch 3
  （0.0002）；首次 ≥0.10 在 epoch 17、≥0.20 在 epoch 19；最佳 EMA pseudo Dice = **0.6236**
  （epoch 952，21:56 UTC）
- 验证：2026-09-24 22:41 → 22:52 UTC（约 11 min 18 s）；223 例；使用 `checkpoint_final.pth`
  （epoch 999，内部 `current_epoch = 1000`）
- **nnU-Net Mean Validation Dice = 0.201140**（`validation/summary.json` 的 `foreground_mean.Dice`
  = 0.20113991；同文件 `foreground_mean.IoU = 0.14537463`）；该数字的分母为 **92 例** = 63 阳性
  + **29 例假阳阴性**（131 例真阴记 NaN 被剔除）
- 阳性病例口径：macro Dice **0.293728**（median 0.219963）、micro Dice **0.533166**、
  体素召回 **0.411569**、`positive_voxel_precision` **0.756744**、
  `all_prediction_voxel_precision` **0.681217**
- 检出结构：阳性 63 例中有任意重叠 **37 例**（58.7%）、完全无重叠 **26 例**（其中 22 例整例无预测）；
  阴性 160 例中 29 例出现预测（假阳体素合计 34,954）
- validation probabilities：**未导出**（`validation/` 下无 `.npz`）
- checkpoint（**勿删除/覆盖**）：`checkpoint_final.pth`（22:41 UTC）、`checkpoint_best.pth`
  （epoch 952，21:56 UTC）
- 输出目录（勿删除/覆盖）：
  `outputs/nnUNet_results/Dataset606_PICAI_Zonal/nnUNetTrainerPICAI_AnatomyGate_PositiveSampling_NoFFT__nnUNetPlans__3d_fullres/fold_0/`
- 边界：`0.201140` 是 nnU-Net 的 `nanmean` 逐例口径（含假阳阴性病例的 0 分），**不是**阳性病例
  Dice；跨 run 的解释与结论一律不在本文件登记。
- 旧输入级 RQ2 配对比较（与 `image_gate_positive_sampling`，delta = anatomy − image）报告已落盘：
  `outputs/reports/segmentation_metrics_rq2_anatomy_gate.json`；结论与完整指标登记在
  `docs/Findings.md` §3.11。
- 详细记录：`docs/archive/historical_experiments/anatomy_gate_positive_sampling.md`

### 三个浅层特征融合 variant（`feature_*_positive_sampling`）
- variant / Trainer：
  `feature_no_gate_positive_sampling` → `nnUNetTrainerPICAI_FeatureNoGate_PositiveSampling_NoFFT`；
  `feature_image_gate_positive_sampling` → `nnUNetTrainerPICAI_FeatureImageGate_PositiveSampling_NoFFT`；
  `feature_anatomy_gate_positive_sampling` → `nnUNetTrainerPICAI_FeatureAnatomyGate_PositiveSampling_NoFFT`
- 数据集 / 配置 / fold：前两者 Dataset605_PICAI / `3d_fullres` / fold 0；第三者为
  Dataset606_PICAI_Zonal / `3d_fullres` / fold 0
- 网络：三个参数不共享、不下采样的浅层 3×3×3 stem（`C_s = 8`）+ 1×1×1 投影回 3 通道 +
  由 plans 构建的原生 `PlainConvUNet`；后两者带零初始化的 feature gate（anatomy 的 gate 额外读
  clamp(0,1) 后的 PZ/TZ，PZ/TZ 不进入 stem/投影/backbone）
- 状态（2026-10-06 核对）：普通融合已启动，日志最后完整 epoch 335，缺最终 checkpoint / validation；
  其余两个门控条件未运行。中间训练动态不能代替完整滑窗验证结果。
- 结构、参数量与比较边界的登记位置见 `README.md`「浅层特征融合」小节

### Dataset606 / anatomy_gate 数据状态

Dataset606_PICAI_Zonal 已物化、完成 3d_fullres preprocessing，并写入固定 split：

- `workdir/nnUNet_raw/Dataset606_PICAI_Zonal/`：1500 例，5 通道（T2W/ADC/HBV + PZ/TZ，PZ/TZ 为 `noNorm`），
  labels `background=0` / `lesion=1`
- `workdir/nnUNet_preprocessed/Dataset606_PICAI_Zonal/`：含 `nnUNetPlans.json` 与 `nnUNetPlans_3d_fullres`
- `splits_final.json`（raw 与 preprocessed 各一份）：1 fold，**train=1277，val=223**
- `anatomy_gate`（原生采样旧 variant）：**已完成**训练 + validation（2026-10-07 07:57 → 23:06 UTC，
  Mean Validation Dice = 0.18267）；见 2026-10-08 节与「各 variant 状态」表。
  > 更正说明（2026-10-08）：本行此前写作"训练未运行"，与同一文件的状态表及 2026-10-08 节的运行事实
  > 冲突。运行事实以状态表与 2026-10-08 节为准；此处只是更正过期索引，不涉及任何结果改动。
- **Dataset605 ↔ Dataset606 前三 MRI 通道逐数组一致性审计：已执行并通过**，且在本训练启动前完成。
  工具 `scripts/data/audit_dataset605_606_mri_equivalence.py`（只读、fail-closed）；
  报告 `outputs/reports/dataset605_606_mri_equivalence_audit_v2.json`（2026-09-24 03:33 UTC，
  1500 例，`status = MRI_ARRAY_AUDIT_PASS`，三通道逐数组完全相同 `global_max|Δ| = 0`，
  `effective_labels_equal = true`，`n_cases_effective_input_mismatch = 0`，病例集合与 split
  含顺序一致；2 例纯 `-1↔0` 原始 seg 差异为信息性，不影响有效标签）。首轮 v1 报告（FAIL，
  起因是把合法的 -1 裁剪填充判为非法标签）按原样保留作追溯。审计只对前三个 MRI 通道判等，
  **PZ/TZ 不参与判等**。

---

## 运行记录模板

```
### <variant> — <dataset>/<config>/fold <k>
- 命令：
- 开始 / 结束（UTC）：
- 状态：完成 / 中断于 epoch N / 失败
- 关键结果：Mean Validation Dice = ，（可选）positive case-wise mean Dice / micro Dice / negative FP rate
- 输出目录：
- 备注：
```
