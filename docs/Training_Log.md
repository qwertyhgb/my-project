> 版本治理（2026-10-12）：V2 为唯一 ACTIVE；V1 为 FROZEN；更早 gate 为 LEGACY。
> 本文件是所有版本的唯一真实运行日志与 variant 状态表。原实验标题中的 A/B/C/D 均按其
> VERSION=V1 范围解释；新实验只用 v2_ 名字。历史事实、路径、时间不改写。

# Training Log

训练与验证的运行事实日志。只记录真实运行（命令、时间、状态、关键结果、产物路径），不含协议/门控叙述。
代码变更见 `docs/Development_Log.md`。所有长任务由研究者本人运行；本文件在每次运行后更新。

---

## 2026-10-08 — 项目重构初始快照（现行状态见「各 variant 状态」，WG解释已由末节诊断更正）

本节登记**重构本身造成的状态变化**，不包含任何新的训练结果。

- **旧线四个 `*_100ep` 条件原地归档**：构造阶段失败的根因已修复（`__init__` 反射问题），但
  **修复后从未重跑**，因此仍然零 epoch、零 checkpoint、零结果。它们现在归入 `--legacy` 视图，
  不再是主线。
- **新主线（Anatomy-Guided Lesion-Aware Coarse-to-Fine）的全部条件均未训练**：

| variant | 数据集 | 状态 | 输出目录 |
|---|---|---|---|
| `positive_sampling`（条件 A1） | Dataset605 / 3d_fullres / 0 | **已完成**（历史运行，无显式 seed；见下） | `outputs/nnUNet_results/Dataset605_PICAI/nnUNetTrainerPICAI_FLCE_PositiveSampling_NoFFT__nnUNetPlans__3d_fullres/fold_0/` |
| `dicece_positive_sampling`（条件 A2） | Dataset605 / 3d_fullres / 0 | 代码就绪，**未训练** | 目标：`.../nnUNetTrainerPICAI_DiceCE_PositiveSampling_NoFFT__nnUNetPlans__3d_fullres/fold_0/`（**未创建**） |
| `anatomy_joint_100ep`（Stage 1） | Dataset607 / 3d_fullres / 0 | **已完成**（训练 + validation；表内旧注“**WG 头不可用**”已 **[SUPERSEDED]**，见本节末尾标注与末节诊断） | `outputs/nnUNet_results/Dataset607_PICAI_Anatomy/nnUNetTrainerPICAI_AnatomyJoint_100ep_NoFFT__nnUNetPlans__3d_fullres/fold_0/` |
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

### Stage-1（`anatomy_joint_100ep`）的真实结果 [VERSION=V1]

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

> **SUPERSEDED BY 2026-10-08 SOFT-HEAD DIAGNOSIS** — 见
> [`docs/experiments/anatomy_joint_100ep.md`](experiments/anatomy_joint_100ep.md)。
> 以下“WG 头实际不可用 / WG 预测必然为空 / 条件 B 因此不可运行”的旧解释**已被后续的
> soft-head 只读诊断推翻**：原数值保留（ordered region hard export 的确如此），只是**解释失效**。
> 当前口径：soft WG head 未显示塌缩；Stage-1 不因此重训。（本节其余内容保持原样，不追改历史。）

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

## 2026-10-08 — 串行队列运行事实、用户停止与 100ep 构造失败 [VERSION=V1 / VERSION=LEGACY]

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

## 2026-10-06 — 现有产物核对与新分支状态 [VERSION=V1 / VERSION=LEGACY]

- 原 `feature_no_gate_positive_sampling` / Dataset605 / fold 0 已有 2026-09-28 启动日志、
  `checkpoint_best.pth` 与 `checkpoint_latest.pth`。现有日志最后完整 epoch 为 335，
  后续记录 epoch 336 起始；无 `checkpoint_final.pth` 或 `validation/summary.json`。
  这是既有运行的产物核对，不是本次新启动训练；停止原因未据日志确认。
- 新增四个 `_100ep` 融合条件均为代码就绪、未运行：
  `feature_no_gate_positive_sampling_100ep`、`feature_anatomy_gate_positive_sampling_100ep`、
  `zonal_reference_positive_sampling_100ep`、`zonal_reference_adaptive_positive_sampling_100ep`。
  本次未启动训练、真实验证、推理或批量影像评估，无新 checkpoint 或结果可引用。

---

## N0 — baseline（已完成） [VERSION=LEGACY]

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

## legacy M0 — 旧自写 Residual-Encoder 管线（已中断，仅历史保留） [VERSION=LEGACY]

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

本表是当前状态唯一来源；前面的重构表是历史快照。V2命令见 docs/v2/Experiment_Plan.md；V1旧命令见 docs/v1/Experiment_Plan_V1.md（须 --v1）；运行后按下方模板在此登记真实事实。

| VERSION | variant | 数据集/配置/fold | 状态 | 输出目录 |
|---|---|---|---|---|
| V2 | v2_early_fusion / v2_multi_encoder_simple / v2_adaptive_fusion / v2_anatomy_fusion / v2_coarse / v2_refine / v2_full_sched | 605/608，3d_fullres / 0 | CODE READY / SYNTHETIC TESTS ONLY / NOT TRAINED；旧均值与直接加权类保留兼容、不安排当前首轮；early-fusion对照复用下方V1 A2，E/F/SCHED暂不排期 | V2新产物归outputs/v2；共享A2保留V1原路径与来源，无新增V2训练事实 |
| V2 | v2_concat_fusion（B-concat，seed=20261008） | Dataset605 / 3d_fullres / 0 | **TRAINING IN PROGRESS**；2026-10-13北京时间08:59只读快照：连续完成epoch0–389（390/1000），正在epoch390；运行正常，final checkpoint与整例validation尚未完成 | outputs/v2/nnUNet_results/Dataset605_PICAI/nnUNetTrainerPICAI_V2_ConcatFusion_DiceCE_NoFFT_Seed20261008__nnUNetPlans__3d_fullres/fold_0/ |
| V2 | v2_concat_adaptive_fusion / v2_concat_anatomy_fusion | 605（C）/608（D），3d_fullres / 0 | CODE READY / SYNTHETIC TESTS ONLY / NOT TRAINED；资源未测，等待B结果后逐组推进 | outputs/v2/nnUNet_results下ConcatAdaptiveFusion/ConcatAnatomyFusion独立V2 Trainer目录 |
| V1 | native_dicece（seed=20261008） | Dataset605 / 3d_fullres / 0 | **1000ep + 223例validation及概率导出完成；Native/A2 full配对评价完成**；详情见`experiments/native_dicece_seed20261008.md` | `outputs/nnUNet_results/Dataset605_PICAI/nnUNetTrainerPICAI_DiceCE_NativeSampling_NoFFT_Seed20261008__nnUNetPlans__3d_fullres/fold_0/` |
| V1 | dicece_positive_sampling_100ep（A-short，seed=20261008） | Dataset605 / 3d_fullres / 0 | **100ep + 223例validation及概率导出完成**；仅短预算筛选参照 | `outputs/nnUNet_results/Dataset605_PICAI/nnUNetTrainerPICAI_DiceCE_PositiveSampling_100ep_NoFFT_Seed20261008__nnUNetPlans__3d_fullres/fold_0/` |
| V1 | neutral_fusion_dicece_100ep（B-short，seed=20261008） | Dataset605 / 3d_fullres / 0 | **100ep + 223例validation及概率导出完成；A-short/B-short full配对评价完成**；配置budget文字误标formal，但epochs/debug/日志均为100 | `outputs/nnUNet_results/Dataset605_PICAI/nnUNetTrainerPICAI_NeutralFusion_DiceCE_100ep_NoFFT_Seed20261008__nnUNetPlans__3d_fullres/fold_0/` |
| V1 | lesion_fusion_dicece_100ep（C-short，seed=20261008） | Dataset605 / 3d_fullres / 0 | **100ep + 223例validation及概率导出完成；A/B/C full配对评价完成** | `outputs/nnUNet_results/Dataset605_PICAI/nnUNetTrainerPICAI_LesionFusion_DiceCE_100ep_NoFFT_Seed20261008__nnUNetPlans__3d_fullres/fold_0/` |
| V1 | anatomy_lesion_fusion_dicece_100ep（D-short，seed=20261008） | Dataset608 / 3d_fullres / 0 | 训练日志epoch0–99完整，Training done与final checkpoint存在；原生整例validation进行中，尚无summary，608审计PASS | `outputs/nnUNet_results/Dataset608_PICAI_PredictedAnatomy/nnUNetTrainerPICAI_AnatomyLesionFusion_DiceCE_100ep_NoFFT_Seed20261008__nnUNetPlans__3d_fullres/fold_0/` |
| LEGACY | baseline | Dataset605 / 3d_fullres / 0 | 已完成（见上） | `outputs/nnUNet_results/Dataset605_PICAI/nnUNetTrainerPICAI_FLCE_NoFFT__nnUNetPlans__3d_fullres/fold_0/` |
| LEGACY | optimized_baseline | Dataset605 / 3d_fullres / 0 | **用户已停止 / 中止（见下）** | `outputs/nnUNet_results/Dataset605_PICAI/nnUNetTrainerPICAI_DiceCE_NoFFT__nnUNetPlans__3d_fullres/fold_0/` |
| LEGACY | image_gate | Dataset605 / 3d_fullres / 0 | 已完成 | `outputs/nnUNet_results/Dataset605_PICAI/nnUNetTrainerPICAI_ImageGate__nnUNetPlans__3d_fullres/fold_0/` |
| LEGACY | anatomy_gate | Dataset606 / 3d_fullres / 0 | **已完成**（训练 + validation，见 2026-10-08 节） | `outputs/nnUNet_results/Dataset606_PICAI_Zonal/nnUNetTrainerPICAI_AnatomyGate__nnUNetPlans__3d_fullres/fold_0/` |
| V1 | positive_sampling | Dataset605 / 3d_fullres / 0 | **已完成**（训练 + validation，见下） | `outputs/nnUNet_results/Dataset605_PICAI/nnUNetTrainerPICAI_FLCE_PositiveSampling_NoFFT__nnUNetPlans__3d_fullres/fold_0/` |
| LEGACY | image_gate_positive_sampling | Dataset605 / 3d_fullres / 0 | **已完成**（训练 + validation，见下） | `outputs/nnUNet_results/Dataset605_PICAI/nnUNetTrainerPICAI_ImageGate_PositiveSampling_NoFFT__nnUNetPlans__3d_fullres/fold_0/` |
| LEGACY | anatomy_gate_positive_sampling | Dataset606 / 3d_fullres / 0 | **已完成**（训练 + validation，见下） | `outputs/nnUNet_results/Dataset606_PICAI_Zonal/nnUNetTrainerPICAI_AnatomyGate_PositiveSampling_NoFFT__nnUNetPlans__3d_fullres/fold_0/` |
| LEGACY | feature_no_gate_positive_sampling | Dataset605 / 3d_fullres / 0 | **已启动，缺最终 validation**（2026-10-06 产物核对见首节） | `outputs/nnUNet_results/Dataset605_PICAI/nnUNetTrainerPICAI_FeatureNoGate_PositiveSampling_NoFFT__nnUNetPlans__3d_fullres/fold_0/` |
| LEGACY | feature_image_gate_positive_sampling | Dataset605 / 3d_fullres / 0 | **用户已停止 / 中断于 epoch 81**（见 2026-10-08 节） | `outputs/nnUNet_results/Dataset605_PICAI/nnUNetTrainerPICAI_FeatureImageGate_PositiveSampling_NoFFT__nnUNetPlans__3d_fullres/fold_0/` |
| LEGACY | feature_anatomy_gate_positive_sampling | Dataset606 / 3d_fullres / 0 | **已完成**（训练 + validation，见 2026-10-08 节） | `outputs/nnUNet_results/Dataset606_PICAI_Zonal/nnUNetTrainerPICAI_FeatureAnatomyGate_PositiveSampling_NoFFT__nnUNetPlans__3d_fullres/fold_0/` |
| V1 | positive_sampling（seed=20261008） | Dataset605 / 3d_fullres / 0 | **1000ep + 223例validation完成**；一次中断续训；详情见 `experiments/positive_sampling_seed20261008.md` | `outputs/nnUNet_results/Dataset605_PICAI/nnUNetTrainerPICAI_FLCE_PositiveSampling_NoFFT_Seed20261008__nnUNetPlans__3d_fullres/fold_0/` |
| V1 | dicece_positive_sampling（seed=20261008） | Dataset605 / 3d_fullres / 0 | **1000ep + 223例validation及概率导出完成**；详情见 `experiments/dicece_positive_sampling_seed20261008.md`；未加seed的原类名运行未启动 | `outputs/nnUNet_results/Dataset605_PICAI/nnUNetTrainerPICAI_DiceCE_PositiveSampling_NoFFT_Seed20261008__nnUNetPlans__3d_fullres/fold_0/` |
| V1 | anatomy_joint_100ep | Dataset607 / 3d_fullres / 0 | 100ep+validation完成；soft heads诊断完成，见末节 | 既有Stage-1目录保留 |
| V1 | neutral_fusion_flce / neutral_fusion_dicece（新B） | Dataset605 / 3d_fullres / 0 | CODE READY / NOT TRAINED；构造与合成检查见Development_Log | 各自独立Trainer类名目录，未创建 |
| V1 | lesion_fusion_flce / lesion_fusion_dicece（新C） | Dataset605 / 3d_fullres / 0 | CODE READY / NOT TRAINED；构造与合成检查见Development_Log | 各自独立Trainer类名目录，未创建 |
| V1 | anatomy_lesion_fusion_flce / anatomy_lesion_fusion_dicece（新D） | Dataset608 / 3d_fullres / 0 | CODE READY / NOT TRAINED（该行正式1000ep类）；608已物化/预处理，D-short状态见独立行 | 各自独立Trainer类名目录，未创建 |
| V1 | lesion_roi / lesion_coarse_to_fine / lesion_zone_refine | 原605/606 | SUPPORTING，未训练；不是新B/C/D；ROI坐标链仍待验证 | 原类名保留，未创建 |
| V1 | lesion_hard_negative | 原605 | SUPPORTING，未训练；固定原C，不能代表新E | 原类名保留，未创建 |
| V1 | 新E best(C,D)+hard negatives | 待父模型选定 | PLANNED / NOT IMPLEMENTED / NOT TRAINED | 尚未定义新Trainer输出 |

### image_gate（已完成） [VERSION=LEGACY]

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

### optimized_baseline（**用户已停止 / 中止**） [VERSION=LEGACY]

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

### positive_sampling（**已完成**：训练 + actual validation） [VERSION=V1]

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

### image_gate_positive_sampling（**已完成**：训练 + actual validation） [VERSION=LEGACY]

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

### anatomy_gate_positive_sampling（**已完成**：训练 + actual validation） [VERSION=LEGACY]

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

### 三个浅层特征融合 variant（`feature_*_positive_sampling`） [VERSION=LEGACY]
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

### Dataset606 / anatomy_gate 数据状态 [VERSION=LEGACY]

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

## 运行记录模板 [VERSION=V1]

```
### <variant> — <dataset>/<config>/fold <k> [VERSION=V1]
- 命令：
- 开始 / 结束（UTC）：
- 状态：完成 / 中断于 epoch N / 失败
- 关键结果：Mean Validation Dice = ，（可选）positive case-wise mean Dice / micro Dice / negative FP rate
- 输出目录：
- 备注：
```

---

## 2026-10-08 — 既有 Stage-1 soft heads 只读诊断（无新训练/推理） [VERSION=V1]

用户明确授权分析既有 validation 输出。conda lm + 固定 nnU-Net 环境，命令：

```bash
python -B -m zonal_reliability_fusion.anatomy.validation outputs/nnUNet_results/Dataset607_PICAI_Anatomy/nnUNetTrainerPICAI_AnatomyJoint_100ep_NoFFT__nnUNetPlans__3d_fullres/fold_0/validation
```

- 范围：既有223例；实际耗时568.04s；success=223 / failed=0 / skipped=0。
- 只读 `.npz/.pkl/.nii.gz` 与 reference，核对形状/物理几何，独立阈值固定为 **>0.5**。
- 输出只在终端，未创建报告文件；既有summary/checkpoint/数据保持原样。
- 本次不是新的 validation/inference；未运行模型预测或训练。

| soft head | macro Dice | macro recall | macro precision | macro pred/ref volume | TP（汇总） | FP（汇总） | FN（汇总） | 空预测 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| WG | 0.94830182 | 0.95322978 | 0.94529174 | 1.00988294 | 26082103 | 1510582 | 1043170 | 0/223 |
| PZ | 0.89928774 | 0.89655433 | 0.90461573 | 0.99326969 | 7843174 | 824719 | 894064 | 0/223 |
| TZ | 0.93613836 | 0.93800987 | 0.93614022 | 1.00472692 | 18722890 | 1091229 | 1102969 | 0/223 |

- WG micro Dice=0.95332881 / recall=0.96154251 / precision=0.94525426 / pred-ref volume ratio=1.01723161。
- 按原生region顺序从soft heads重建hard export，与既有223例NIfTI差异总数 **0 voxel**。
- 原hard-export Dice仍为WG 0.00558392 / PZ 0.89848902 / TZ 0.93613836，产物未修改。
- **更正本文件较早的解释**：“WG头实际不可用”“当前权重必然空WG回退”不成立；
  原数字是ordered hard-export的区域指标，不能解释为独立soft WG head失败。
- 先验与伪监督reference的一致性不能证明人工解剖准确性或下游lesion收益。

新融合B/C/D无新训练/validation结果。NO NEW LONG TRAINING STARTED。

---

## 2026-10-08 — 正式实验前收口（无新训练 / 无新 validation） [VERSION=V1]

本轮只做代码、测试、文档与只读诊断，**未启动**任何训练、validation、推理或数据处理；
未创建任何模型目录或 checkpoint，未删除或覆盖任何既有产物。状态变化（不含运行结果）：

- 短预算（100 epoch）筛选 Trainer 类与 8 个入口变体已实现，**未运行任何 100ep 训练**；
  正式 1000 epoch 类未被修改（类名与输出目录不变）。
- 融合系数按 `(logits / SOFTMAX_TEMPERATURE).softmax(dim=1)` 显式计算（T 仍固定 1.0，数值不变）。
- 新增只读工具 `scripts/data/audit_dataset605_608_equivalence.py`（Dataset605↔608 一致性审计）；
  **尚未在真实 Dataset608 上运行**（608 未物化/未预处理）。
- `A1 vs A2` 的 baseline selection rule 已冻结在 `docs/Experiment_Plan.md` §2.1（先于任何结果）。

NO NEW MODEL TRAINING OR VALIDATION WAS STARTED.

---

## 2026-10-10 — A1/A2 matched-seed full 评价失败（未发布结果） [VERSION=V1]

- 命令：`python scripts/evaluate_segmentation.py --model A1_final=<A1 fold_0> --model A2_final=<A2 fold_0>
  --mode full --nsd-tolerance-mm 2.0 --bootstrap-resamples 10000 --seed 20260922
  --output outputs/metrics/A1_A2_final_full_exploratory_NSD2mm_20261010T031106Z.json`
- 日志：`outputs/logs/A1_A2_full_20261010T031106Z.log`（64 行，stdout+stderr 合并）
- 日志结束汇总（原文）：`status=failed`、`mode=full`、`phase=full_cases`、`unit=model-case`、
  `ok=44`、`failed=402`、`skipped=0`、`elapsed=164.28 s`
- 退出码 **EXIT_CODE=2**；日志 `output` 行指向的结果 JSON **未发布**（无产物落盘）
- 失败形态：日志展示的前 50 条全部为 prediction 与 reference 的 `direction` 不严格相等，
  最大元素绝对差约 `3.47e-08`；日志明确写明「仅显示前 50 条，另有 352 条未显示」，
  因此其余 352 条的字段与量级**未经核实**
- 代码/协议侧处理：本轮为评价侧几何数值检查的**显式协议修订**
  （`docs/Development_Log.md` 同日条目；`docs/Evaluation_Protocol.md` §3.1），**不是**这次运行的替代
- **真实队列尚未用新规则重跑**：修复后的 full 评价**没有完成**，不得记为成功
- 边界：本次未启动任何新的训练/推理/预处理；未修改任何 prediction、reference 或 data/workdir 产物


## 2026-10-09 — A1 positive_sampling，seed=20261008 完成 [VERSION=V1]

- 运行：Dataset605_PICAI / 3d_fullres / fold0；Trainer=`nnUNetTrainerPICAI_FLCE_PositiveSampling_NoFFT_Seed20261008`。
- 用户首次启动，断网中断后以`--continue-training`恢复；日志从epoch41继续至999。
- 时间（北京时间）：2026-10-08 17:30:06首轮开始；2026-10-09 11:01:55训练完成；11:13:43 validation完成。
- checkpoint_final.current_epoch=1000；validation共223例，病例集合与固定val一致。
- nnU-Net foreground_mean.Dice=0.2137595923；positive-case macro Dice=0.2951918179。
- 完全无重叠阳性25/63；阴性FP病例24/160。本条只登记运行事实，不作跨run结论。
- 配置、两段命令、训练动态、summary-level分析与产物：`docs/experiments/positive_sampling_seed20261008.md`。
- 本次报告工作只读既有产物；未重跑训练、validation、推理或full评估。

## 2026-10-10 — A2 dicece_positive_sampling，seed=20261008 完成 [VERSION=V1]

- 运行：Dataset605_PICAI / 3d_fullres / fold0；Trainer=`nnUNetTrainerPICAI_DiceCE_PositiveSampling_NoFFT_Seed20261008`。
- 启动入口：`scripts/train/train_nnunet.py dicece_positive_sampling 605 3d_fullres 0 --seed 20261008 --seeded-output --export-validation-probabilities --run-config outputs/run_configs/A2_dicece_positive_sampling_fold0_seed20261008.json`；完整环境与命令见实验报告。
- 时间（北京时间）：2026-10-09 16:51:50 epoch0开始；2026-10-10 08:42:28训练完成；08:54:18 validation完成。
- epoch0–999各记录一次；`checkpoint_final.current_epoch=1000`。现有日志未显示中断续训。
- final 整例 validation 共223例，与固定val病例集合一致；`.nii.gz`、`.npz`、`.pkl`各223份。
- nnU-Net foreground_mean.Dice=0.1624880607（194例有限值）；阳性63例 macro Dice=0.5003600600。
- 阳性零重叠6/63，阳性空预测0/63；阴性有前景预测131/160。本条只登记运行事实，不作跨run结论。
- `checkpoint_best.current_epoch=968`，记录的最佳patch pseudo-Dice EMA=0.53213817；本次确认的是final整例验证，未确认best整例性能。
- 详细配置、训练动态、病例级结果、产物及证据边界：`docs/experiments/dicece_positive_sampling_seed20261008.md`。
- 本次报告工作只读既有日志、checkpoint logger、JSON及文件清单；未启动训练、推理、validation或医学体素评测。

### 2026-10-10 — A2 final 全验证集概率诊断 [VERSION=V1]

- 用户运行 `scripts/data/diagnose_picai_case_chain.py`，使用A2 `fold_0`与其`validation`概率目录，参数`--all-validation --no-plots`。
- 开始时间（北京时间）09:38:59；耗时309.475 s；successful=223 / failed=0 / skipped=0，完整队列阳性63、阴性160。
- 产物：`outputs/diagnostics/picai_probability_diagnostics_20261010T013859_953463Z/summary.json`。
- 概率恢复几何、概率argmax与硬分割、GT计数检查均通过；硬分割与原validation一致。
- p>0.5复现阳性macro Dice=0.500360060、零重叠6/63、阴性有FP131/160；p>0.01零重叠仍6/63、阴性有FP153/160。
- 本次代理仅读取既有JSON并补充上述实验报告，未自行启动数据处理或推理。

### 2026-10-10 — A2 best 概率导出及诊断 [VERSION=V1]

- 用户完成checkpoint_best.pth独立预测；产物`outputs/predictions/A2_best_fold0_seed20261008_20261010T015819Z/`，原final validation保留。
- 用户运行全验证集概率诊断，产物`outputs/diagnostics/picai_probability_diagnostics_20261010T021025_787594Z/summary.json`。
- 诊断开始（北京时间）10:10:25；耗时300.054 s；successful=223 / failed=0 / skipped=0；全部checks通过。
- p>0.5：阳性macro Dice=0.503831146，pooled recall=0.568709546，零重叠6/63，阴性有FP149/160、阴性FP体素322425。
- 详细记录及同run checkpoint比较见现有A2实验报告第10节。代理只读JSON并更新文档，未启动长任务。

### 2026-10-10 — A2 零重叠病例数据链诊断 [VERSION=V1]

- 用户运行既有case-chain脚本，对final/best零重叠并集7例加2例高Dice对照生成JSON与图像。
- 产物：`outputs/diagnostics/picai_case_chain_20261010T024730_612404Z/summary.json`。
- 开始（北京时间）10:47:30；耗时121.157 s；successful=9 / failed=0 / skipped=0；全部13项checks通过。
- 逐病例结果与检查范围见现有A2实验报告第11节；不能据9例检查声明全数据集处理正确。
- 代理仅读取既有JSON与两张既有PNG，未启动数据处理、训练或推理。

### 2026-10-10 — A1/A2 matched-seed final 统一summary评价 [VERSION=V1]

- 用户运行`python scripts/evaluate_segmentation.py --model A1_final=<A1_seed_fold0> --model A2_final=<A2_seed_fold0> --mode summary --bootstrap-resamples 10000 --seed 20260922 --output <新JSON>`。
- 产物：`outputs/metrics/A1_A2_final_fold0_seed20261008_20261010T025355Z.json`。
- succeeded；ok=2 models / failed=0；耗时0.10 s；每模型223例、阳性63、阴性160。
- A2-minus-A1配对Dice差=0.205168242，95% CI=[0.146075075,0.266933854]；improved/tied/worsened=51/6/6。
- newOverlap=19 / lostOverlap=0；missDelta=-19；negative_fp_case_delta=+107。
- 按Experiment_Plan §2.1冻结的primary-first规则，选A2 DiceCE家族作为后续B/C/D的loss参考；两家族产物保留。此选型不声明整体FP改善或多seed稳定性。
- 尚未执行统一full评价或启动B/C/D训练。summary不含病灶实例、物理FP体积或表面指标。


### 2026-10-10 — A1/A2 final full评价成功 [VERSION=V1]

- 用户在修订后的几何规则下完成full评价；NSD=2mm为探索性，seed=20260922、bootstrap=10000。
- 产物：`outputs/metrics/A1_A2_final_full_exploratory_NSD2mm_20261010T035655212238528Z.json`；日志`outputs/logs/A1_A2_full_20261010T035655212238528Z.log`。
- EVAL_EXIT=0、TEE_EXIT=0；ok=446 / failed=0 / skipped=0；耗时621.57s；两模型各223例计数核验通过。
- 实例any-overlap匹配A1=39/75、A2=60/75；未匹配预测团块A1=37、A2=226。
- A-short/B-short尚未运行；本次代理只读结果JSON并更新文档，未启动任何训练或真实数据评价。


### 2026-10-10 — A-short DiceCE+阳性采样完成 [VERSION=V1]

- 用户运行`dicece_positive_sampling_100ep 605 3d_fullres 0 --seed 20261008 --seeded-output --export-validation-probabilities --run-config outputs/run_configs/A_short_dicece_positive_sampling_fold0_seed20261008.json`。
- Trainer=`nnUNetTrainerPICAI_DiceCE_PositiveSampling_100ep_NoFFT_Seed20261008`；日志`training_log_2026_10_10_05_36_25.txt`；输出目录见唯一状态表。
- 时间（北京时间）：13:36:35 epoch0开始；15:12:12 Training done；15:24:05 Validation complete。
- 日志epoch0–99共100条；final checkpoint存在；validation病例与固定split一致，nii.gz/npz/pkl各223份。
- summary：阳性macro Dice=0.442427279；阳性零重叠7/63、空预测0/63；阴性有前景160/160；native foreground mean Dice=0.124990666。
- 本条仅记录运行事实，不与1000epoch结果比较；未完成本run的概率诊断/full实例评价。原生采样1000epoch参考尚未启动（本次只读路径检查）。
- 代理只读现有日志/JSON/文件清单并更新记录，未启动训练或数据处理。

### 2026-10-11 — 核查B-short完成及原生采样参考运行 [VERSION=V1]

- B-short用户已完成；日志`training_log_2026_10_10_08_02_37.txt`有完整epoch0–99，
  `checkpoint_final.pth`存在，debug中num_epochs=100、batch=2、LR=0.01、每epoch训练/验证迭代250/50。
- 北京时间2026-10-10 16:02:41 epoch0开始，18:00:58 Training done，18:15:58 Validation complete；
  日志epoch耗时均值69.1609秒（不含epoch之间及最终validation开销）。
- validation内nii.gz/npz/pkl各223份；summary中病例集合与Dataset605固定fold0验证split完全一致，
  63阳性、160阴性。这里只核对文件清单与JSON，不表示已解码核查概率或掩膜几何。
- B-short阳性macro Dice=0.429203366、中位数=0.478187663、micro Dice=0.555531055，
  阳性体素recall=0.560651541、precision=0.550503254、全队列prediction precision=0.255436577；
  阳性零重叠7/63、空预测0/63，阴性有前景159/160，阴性FP体素681950（voxel-based）。
- A-short/B-short病例身份与n_ref逐例相同；阳性Dice配对均值变化=-0.013223913，
  improved/tied/worsened=27/7/29，新增/丢失重叠均0。尚未运行配对bootstrap或full实例/表面评价。
- B-short run_config的epochs=100，但budget文字仍为formal；A-short也有相同文字误标。
  本轮保留原始配置，预算以Trainer/debug及完整epoch日志核实，不覆盖outputs。
- 原生采样参考进程存在；命令为`python -u scripts/train/train_nnunet.py native_dicece 605 3d_fullres 0 --seed 20261008 --seeded-output --export-validation-probabilities --run-config outputs/run_configs/native_dicece_fold0_seed20261008.json`。
  日志`training_log_2026_10_10_15_01_47.txt`，debug/run_config均为1000epoch、原生DiceCE、原生采样；
  最近读取到北京时间2026-10-11 10:09:02进入epoch701，epoch700完成，pseudo Dice=0.376。
  best/latest存在、final及最终验证summary尚未产生。Pseudo Dice不是完整病例分割终点。
- 核查仅只读日志、配置、JSON及进程清单；代理未启动训练、推理或真实医学数据评价。

### 2026-10-11 — A-short/B-short full配对评价完成 [VERSION=V1]

- 用户运行完成；产物`outputs/metrics/Ashort_Bshort_final_full_exploratory_NSD2mm_20261011T021156038922776Z.json`，
  日志`outputs/logs/Ashort_Bshort_full_20261011T021156038922776Z.log`。
  创建时间北京时间10:27:38；mode=full、schema=1.2、seed=20260922、bootstrap=10000、探索性NSD容差2 mm。
- 结束汇总status=succeeded、ok=446、failed=0、skipped=0、elapsed=942.86秒。
  每模型223例完整掩膜核验；spacing/origin差异0，direction最大差3.9913e-8，
  角点最大物理偏差1.4795e-5 mm，均在数值一致性规则内。没有重采样。
- A-short→B-short阳性macro Dice变化=-0.013223913，配对95% CI=[-0.035964361,+0.009019042]，
  improved/tied/worsened=27/7/29；完整零重叠7→7，新增/丢失病例重叠均0。
- 75个参考实例匹配62→60，any-overlap sensitivity=82.67%→80.00%；
  小/中/大实例分别16/25→15/25、10/14→9/14、36/36→36/36。
  全队列预测团块1007→905，未匹配预测团块945→845；阴性FP体素530992→681950。
- 阳性表面均值：HD95=29.890981→34.497012 mm，ASSD=10.528927→12.354478 mm，
  NSD@2mm=0.516179→0.506299。没有这些终点的配对CI，不声明显著变化。
- 代理只读产物并核对汇总计数/日志，未重跑真实评价；几何规则通过不是原始MRI模态配准质量证明。
  原生采样训练另一次日志快照已进入epoch724，未产生final；C-short目录/run_config尚不存在，未启动。

### 2026-10-11 — 原生DiceCE＋原生采样正式参考完成 [VERSION=V1]

- 用户报告完成，本轮核查日志`training_log_2026_10_10_15_01_47.txt`完整epoch0–999；
  final checkpoint current_epoch=1000，logger八个序列各1000项；best current_epoch=949。
  Trainer/debug确认原生nnUNetDataLoader、DC_and_CE_loss与PlainConvUNet，含NoFFT修补、无项目阳性病例平衡。
- 北京时间2026-10-10 23:01:51 epoch0开始，2026-10-11 14:53:15 Training done，15:05:05 Validation complete。
  训练wall time约15:51:24，最终验证约11:50；epoch日志计算时长均值55.541秒。
- 验证病例与固定fold0 split相同，223study/220patient，63阳性/160阴性；NIfTI/npz/pkl各223份且病例集合一致。
  split按patient ID前缀核对无train/val交集；summary逐例计数与Dice公式一致，未读取真实影像进行full核验。
- 阳性macro Dice=0.469091115，median=0.502262443，micro Dice=0.565079982；
  阳性voxel recall=0.531612705、precision=0.603044173、全队列prediction precision=0.384914008。
  阳性TP/FP/FN=308161/202848/271511；完全零重叠7/63、阳性空预测0；
  阴性有前景151/160，阴性FP体素289588（voxel-based）。原生foreground mean Dice=0.138096917。
- 本轮按用户授权创建独立报告[原生采样正式实验](experiments/native_dicece_seed20261008.md)，
  记录配置、运行、训练动态、验证分布、漏分病例、复现边界与待补终点，不包含跨实验比较。
- 尚未计算本run CI、实例匹配、物理体积/表面或概率诊断；best没有整例验证证据。
  本轮仅秒级只读日志/JSON/checkpoint元数据及logger，未启动任何训练或真实医学数据评测。

### 2026-10-11 — Native/A2 final完整配对评价完成 [VERSION=V1]

- 用户返回运行日志及EVAL_EXIT=0、TEE_EXIT=0；结果
  `outputs/metrics/native_A2_final_full_exploratory_NSD2mm_20261011T071549214283765Z.json`，
  日志`outputs/logs/native_A2_full_20261011T071549214283765Z.log`。
  status=succeeded、ok=446、failed=0、skipped=0、elapsed=761.73秒；
  schema=1.2、seed=20260922、bootstrap=10000、探索性NSD@2mm。
- 每模型223例掩膜计数与summary一致，几何规则均通过且未重采样；direction最大差3.9913e-8、
  角点最大物理偏差1.4795e-5 mm，spacing/origin差为0。
- Native→A2阳性macro Dice=0.469091115→0.500360060，配对变化=+0.031268945，
  95% CI=[+0.002529913,+0.063516397]；31改善/5持平/27下降。
  新增病例重叠`10519_1000529`、`10550_1000562`，丢失重叠`11284_1001307`；零重叠7→6。
- Any-overlap实例匹配57/75→60/75；小13/25→16/25、中8/14→9/14、大36/36→35/36。
  阴性FP病例151→131、阴性FP体素289588→230628，全队列未匹配预测团块287→226。
  阳性pooled recall=0.531613→0.518630，precision=0.603044→0.658966。
- Native自身bootstrap/实例/表面终点补入其独立实验报告；跨run解释写Findings。
  本轮代理仅只读已有JSON并更新文档，未启动训练或真实医学数据评价。C-short尚未发现产物目录。

### 2026-10-12 — C-short完成、Dataset608准备通过及D-short运行 [VERSION=V1]

- C日志`training_log_2026_10_11_07_33_34.txt`完整epoch0–99，无已发现错误/NaN。
  北京时间2026-10-11 15:33:37开始epoch0，19:50:37 Training done，20:06:42 Validation complete；
  epoch计算时长均值152.3876秒，final/best存在。
- C的223例summary及nii.gz/npz/pkl各223份，病例集合与605固定fold0验证split一致，
  63阳性/160阴性；逐例计数恒等式通过。本轮仅核对JSON/文件清单，未读真实医学数组。
  macro Dice=0.434474361、median=0.497877474、micro Dice=0.529358330；
  阳性recall=0.505159815、precision=0.555991836、全队列prediction precision=0.266435013；
  完全零重叠11/63、阳性空预测0、阴性FP病例159/160、FP体素572381。
- C−A阳性Dice配对均值=-0.007952918，29改善/7持平/27下降；
  C−B均值=+0.005270995，30改善/7持平/26下降；尚无配对bootstrap CI或full实例评价。
  两项对照新增重叠均0，丢失重叠均为`10433_1000441`、`10550_1000562`、`11284_1001307`、`11296_1001319`。
- Dataset608已物化并预处理；审计产物
  `outputs/reports/dataset605_608_equivalence_20261011T151439284226933Z.json`：
  `DATASET605_608_EQUIVALENCE_PASS`、prior契约VALID；raw/preprocessed各1500例通过，
  三路MRI逐值完全相同，MRI/标签mismatch均0，病例集合/顺序、split及metadata一致。
  C/D plans的architecture、patch、batch、spacing、batch_dice逐项相同。
- prior契约总计1276个IN_SAMPLE_PRED及224个HELD_OUT_PRED，其中全部223验证病例为HELD_OUT_PRED；
  另一个held-out病例来自训练split的Stage-1排除病例。不宣称OOF或人工解剖准确度。
- D日志`training_log_2026_10_11_22_26_11.txt`，debug为100epoch、6输入通道、batch2、LR0.01、
  ConditionedMultimodalNNUNet与LesionnessAuxiliaryLoss。北京时间2026-10-12 08:32:17进入epoch42；
  epoch41 pseudo Dice=0.3443，当前最高EMA约0.3910，无已发现错误/NaN，进程存在；未有final/整例summary。
  run_config为`outputs/run_configs/D_short_anatomy_lesion_fusion_dicece_fold0_seed20261008_20261011T151439284226933Z.json`，
  epochs=100但budget文字误标formal，保留原始产物，实际预算依据类名/debug/epochs。
- 本轮代理只读既有日志、配置、审计和summary并更新文档，未运行任何训练、推理或真实医学数据评价。

### 2026-10-12 — A/B/C-short完整配对评价完成 [VERSION=V1]

- 用户执行full评价及tee均exit0；产物
  `outputs/metrics/ABC_short_final_full_exploratory_NSD2mm_20261012T003738020176149Z.json`，
  日志`outputs/logs/ABC_short_full_20261012T003738020176149Z.log`；北京时间09:03:08生成。
  schema1.2、seed20260922、bootstrap10000、探索性NSD2mm；成功669、失败0、跳过0、耗时1530.75秒。
- 三模型各223例掩膜计数及数值几何核验通过；最大角点差1.4795e-5 mm，无重采样。
- C−A配对Dice=-0.007952918，CI95=[-0.033456685,+0.016312791]；
  C−B=+0.005270995，CI95=[-0.015854520,+0.026121346]，两项均跨0。
- A/B/C参考实例匹配62/60/55（分母75），小实例16/15/11（分母25），
  中实例10/9/9（分母14），大实例36/36/35（分母36）；未匹配预测团块945/845/735。
  Matched-lesion Dice均值0.541092/0.543292/0.588551，只覆盖各自匹配子集。
- A/B/C阳性HD95均值29.890981/34.497012/38.685617 mm，ASSD均值10.528927/12.354478/14.183726 mm；
  探索性NSD2mm均值0.516179/0.506299/0.508732，未计算这些终点的配对CI。
- C相对A/B丢失的四例重叠中，三例病例GT总体积240.408/164.250/127.332 mm³，
  另一例11284_1001307为2783.831 mm³；这是病例总体积，不直接作为单病灶大小。
  四例C预测均非空，不能把它们描述为全部预测背景。
- 最新D日志快照进入epoch51，final/整例summary尚无；代理只读评价JSON并更新记录，没有重跑评价。

### 2026-10-12 — V1 D-short只读状态核对 [VERSION=V1]

- 读取既有 `training_log_2026_10_11_22_26_11.txt` 的尾部，最后epoch起始行：`2026-10-12 03:36:30.493994: Epoch 98`（原日志UTC）。
- 未见checkpoint_final与validation/summary；不猜测完成或D性能。
- 当前进程未被本轮启动/停止；outputs与run-config均未改动。
- V1状态依据此快照更新唯一表；本次V2代码重构不作为真实训练事实。

### 2026-10-12 — V1 D-short训练完成、整例验证尚未完成 [VERSION=V1]

- 本轮后续只读日志：epoch0–99各一次，共100条；`Training done`时间为 `2026-10-12 03:43:14.458400`（原日志UTC）。
- final checkpoint文件存在；未加载checkpoint，未核实其内部epoch字段。
- 日志仍在逐例predicting，validation/summary.json尚不存在，不能报整例性能或评价完成。
- 用户既有进程继续运行；代理没有启动、停止或重跑training/validation，未写outputs。

## 2026-10-12 — B-concat正式训练启动 [VERSION=V2]

用户返回启动及前两轮日志；本条原始日志时间未标注时区，不推定或转换训练起止时间。
启动audit时间为日志2026-10-12 10:18:14，epoch0开始10:18:17；用户提供日志已完成
zero-based epoch0/1，10:22:57进入epoch2（显示3/1000）。这是早期训练事实，不是整例性能结果。

```bash
nnUNet_compile=false CUDA_VISIBLE_DEVICES=0 \
python -u -B scripts/train/train_nnunet.py \
  v2_concat_fusion 605 3d_fullres 0 \
  --seed 20261008 --seeded-output \
  --run-config outputs/v2/run_configs/V2_B_concat_fusion_fold0_seed20261008.json \
  --export-validation-probabilities
```

- Trainer：nnUNetTrainerPICAI_V2_ConcatFusion_DiceCE_NoFFT_Seed20261008；网络ConcatFusionUNet。
- 实际run-config确认：concat全尺度0–6、3×3×3投影、无policy、无anatomy/coarse/refiner/schedule；
  1000epoch、seed20261008、patch[16,320,320]/batch2、Dataset605/fold0。
- 原生DeepSupervisionWrapper(DC_and_CE_loss)，输入3、输出2；默认预处理与三路ZScore。
- 训练病例1277（positive362/negative915）；loader固定1个阳性槽位/batch2，验证采样原生。
  日志0.5保证为loader采样口径，不视为增强后所有patch实际病灶比例测量。

| zero-based epoch | train_loss | val_loss | patch pseudo Dice | epoch time(s) |
|---|---:|---:|---:|---:|
| 0 | 0.0088 | -0.0573 | 0.1168 | 141.16 |
| 1 | -0.1503 | -0.0841 | 0.1651 | 133.77 |

两轮平均137.465s，仅用于粗估剩余998轮约38.1h，不包含整例validation与后续速度变化。
只读进程/GPU检查发现主训练进程仍在运行，GPU利用率98%、总占用11562MiB（约11.3GiB）；
此为即时占用，不是训练峰值显存测量。未发现返回日志中的OOM/NaN/Traceback。
DNA依赖约束warning仍为已知PlainConvUNet兼容限制，不升级包或改变plans；
hiddenlayer缺失仅导致网络示意图未生成，训练已继续。固定V2-P2标签不是phase-scheduled训练。

产物：run-config已存在；日志training_log_2026_10_12_10_18_10.txt；
模型目录outputs/v2/nnUNet_results/Dataset605_PICAI/nnUNetTrainerPICAI_V2_ConcatFusion_DiceCE_NoFFT_Seed20261008__nnUNetPlans__3d_fullres/fold_0/。
本次只读核查时checkpoint_final.pth与validation/summary.json均未存在，不能登记为训练或验证完成。
原始用户日志附件：/root/.codex/attachments/252397f9-ba33-4f4a-a624-8173354c503b/已粘贴的文本.txt。
代理未启动、停止、续训或修改运行中的训练；后续用户返回日志后再更新。

## 2026-10-13 — B-concat训练中期只读快照 [VERSION=V2]

检查时间北京时间2026-10-13 08:59，读取既有training_log_2026_10_12_10_18_10.txt、
文件存在性/mtime与进程/GPU元数据；未加载checkpoint内容或医学影像，未运行验证/评测。
连续完成zero-based epoch0–389，共390/1000（39.0%），正在epoch390（第391轮）。
最后完成epoch389：train_loss=-0.3677、val_loss=-0.1600、patch pseudo Dice=0.4904，epoch time=134.30s。
当前LR=0.00641；最新日志在检查前数秒更新。主进程PID853989仍在运行，GPU利用率100%，
全卡占用11562MiB（约11.3GiB）；未见日志中的Traceback/OOM/NaN/Inf。

| zero-based epoch范围 | 数量 | 平均train_loss | 平均val_loss | 平均patch pseudo Dice | 平均epoch time(s) |
|---|---:|---:|---:|---:|---:|
| 0–99 | 100 | -0.297121 | -0.107440 | 0.344508 | 133.7740 |
| 100–199 | 100 | -0.337948 | -0.121723 | 0.408627 | 133.7007 |
| 200–299 | 100 | -0.352461 | -0.125691 | 0.416668 | 133.6811 |
| 300–389 | 90 | -0.369534 | -0.128756 | 0.443746 | 133.6581 |

最近20轮均值：train_loss=-0.374685、val_loss=-0.132405、pseudo Dice=0.442530、time=133.699s；
最近100轮均值：train_loss=-0.368519、val_loss=-0.129670、pseudo Dice=0.443471、time=133.6589s。
这只是单次训练的轮间patch动态，不是整例Dice_all或run-to-run方差；不能据此声明优于共享A2。
日志最新best EMA pseudo Dice为0.477200（原始时间2026-10-13 00:18:07，未根据EMA改选正式checkpoint）。

checkpoint_latest.pth存在，mtime北京时间2026-10-13 07:28:28；checkpoint_best.pth存在，
mtime北京时间08:18:09；均约923MB。final checkpoint与validation/summary.json仍不存在。
日志ETA=81559.7s，约22.7h，另需整例validation；估算不包含后续速度变化或日志外保存开销。
建议继续当前配置与预算，不停止、不重启、不同时启动C/D。代理未干预用户训练。
