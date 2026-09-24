# Training Log

训练与验证的运行事实日志。只记录真实运行（命令、时间、状态、关键结果、产物路径），不含协议/门控叙述。
代码变更见 `docs/Development_Log.md`。所有长任务由研究者本人运行；本文件在每次运行后更新。

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
| anatomy_gate | Dataset606 / 3d_fullres / 0 | 数据已就绪，训练未运行 | `outputs/nnUNet_results/Dataset606_PICAI_Zonal/nnUNetTrainerPICAI_AnatomyGate__nnUNetPlans__3d_fullres/fold_0/` |
| positive_sampling | Dataset605 / 3d_fullres / 0 | **已完成**（训练 + validation，见下） | `outputs/nnUNet_results/Dataset605_PICAI/nnUNetTrainerPICAI_FLCE_PositiveSampling_NoFFT__nnUNetPlans__3d_fullres/fold_0/` |
| image_gate_positive_sampling | Dataset605 / 3d_fullres / 0 | **已完成**（训练 + validation，见下） | `outputs/nnUNet_results/Dataset605_PICAI/nnUNetTrainerPICAI_ImageGate_PositiveSampling_NoFFT__nnUNetPlans__3d_fullres/fold_0/` |
| anatomy_gate_positive_sampling | Dataset606 / 3d_fullres / 0 | **未运行** | `outputs/nnUNet_results/Dataset606_PICAI_Zonal/nnUNetTrainerPICAI_AnatomyGate_PositiveSampling_NoFFT__nnUNetPlans__3d_fullres/fold_0/` |
| feature_no_gate_positive_sampling | Dataset605 / 3d_fullres / 0 | **未运行** | `outputs/nnUNet_results/Dataset605_PICAI/nnUNetTrainerPICAI_FeatureNoGate_PositiveSampling_NoFFT__nnUNetPlans__3d_fullres/fold_0/` |
| feature_image_gate_positive_sampling | Dataset605 / 3d_fullres / 0 | **未运行** | `outputs/nnUNet_results/Dataset605_PICAI/nnUNetTrainerPICAI_FeatureImageGate_PositiveSampling_NoFFT__nnUNetPlans__3d_fullres/fold_0/` |
| feature_anatomy_gate_positive_sampling | Dataset606 / 3d_fullres / 0 | **未运行** | `outputs/nnUNet_results/Dataset606_PICAI_Zonal/nnUNetTrainerPICAI_FeatureAnatomyGate_PositiveSampling_NoFFT__nnUNetPlans__3d_fullres/fold_0/` |

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
- 详细记录：`docs/experiments/image_gate.md`

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
  RQ1 的公平匹配臂
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
- 与 `positive_sampling` 的配对比较（RQ1）登记在 `docs/Findings.md` §3.10：配对均值 delta
  = −0.0077、CI95 含 0，**未观察到明确的分割增量效用**，但工作点更保守（precision 升高、
  假阳减少、漏检略增）
- validation probabilities：**未导出**
- checkpoint（**勿删除/覆盖**）：`checkpoint_final.pth`、`checkpoint_best.pth`
- 输出目录（勿删除/覆盖）：
  `outputs/nnUNet_results/Dataset605_PICAI/nnUNetTrainerPICAI_ImageGate_PositiveSampling_NoFFT__nnUNetPlans__3d_fullres/fold_0/`
- 详细记录：`docs/experiments/image_gate_positive_sampling.md`

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
- 状态：**三者均未运行**，无任何训练或 validation 产物，输出目录尚未创建
- 结构、参数量与比较边界的登记位置见 `README.md`「浅层特征融合」小节

### Dataset606 / anatomy_gate 数据状态

Dataset606_PICAI_Zonal 已物化、完成 3d_fullres preprocessing，并写入固定 split：

- `workdir/nnUNet_raw/Dataset606_PICAI_Zonal/`：1500 例，5 通道（T2W/ADC/HBV + PZ/TZ，PZ/TZ 为 `noNorm`），
  labels `background=0` / `lesion=1`
- `workdir/nnUNet_preprocessed/Dataset606_PICAI_Zonal/`：含 `nnUNetPlans.json` 与 `nnUNetPlans_3d_fullres`
- `splits_final.json`（raw 与 preprocessed 各一份）：1 fold，**train=1277，val=223**
- anatomy_gate：数据已就绪，**训练未运行**
- **Dataset605 ↔ Dataset606 前三 MRI 通道逐数组一致性审计**：工具已实现
  （`scripts/data/audit_dataset605_606_mri_equivalence.py`，含纯合成单元测试），
  **真实数据审计尚未执行** —— 启动 `anatomy_gate_positive_sampling` 前必须先由研究者本人运行并取得
  `status=MRI_ARRAY_AUDIT_PASS`。配置层面（split 集合与顺序、spacing、patch/batch、前三个 MRI
  通道的 normalization 与 per-channel intensity fingerprint）已核对一致，但这不等于数组一致。

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
