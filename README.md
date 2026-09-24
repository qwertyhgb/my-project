# Anatomy-Conditioned Multi-sequence Prostate Lesion Segmentation

前列腺癌（csPCa）病灶分割研究项目，采用 **nnU-Net-first** 架构。研究核心是在原生 nnU-Net
backbone 前加一个很小的 **spatial modality gate**，并考察解剖分区（PZ/TZ）先验是否能改变
T2W/ADC/HBV 的空间相对序列偏好。

> **术语说明**：类名/包名中的 `Reliability`（`zonal_reliability_fusion`、
> `SpatialModalityReliabilityGate`）是**历史内部标识**，保留它是为了 checkpoint、导入路径与
> 既有代码兼容。当前研究解释是 **learned sequence preference / scale**（序列相对偏好与尺度），
> **不是**校准可靠性、真实图像质量或因果贡献。

## nnU-Net-centered 架构与受控扩展

nnU-Net（固定版本 `third_party/nnUNet`，v2.6.2，commit `74ceb68`，只读）负责全部通用机制：
planning/preprocessing、data augmentation、deep supervision、optimizer 与学习率调度、
training loop、checkpoint/resume、validation、sliding-window inference、prediction export；
默认也负责 dataloader 与 patch sampling。

只有在 nnU-Net 原生机制不能满足**明确研究假设**时，项目才做受控扩展。项目自身保留：

1. PI-CAI Focal + CE 损失（`0.5*Focal(gamma=2) + 0.5*CE`）；
2. Gaussian blur 的 NoFFT 兼容修复（关闭不稳定的 FFT benchmark，blur 概率/sigma 不变）；
3. 一个基于原生 nnU-Net 网络的轻量 spatial modality reliability gate（输入级）；
4. 必要的 PZ/TZ 输入适配（Dataset606 的附加通道与增强边界）；
5. 阳性病例感知的训练 patch 采样（`positive_sampling`：每批固定一个阳性病灶 patch，
   见 `src/zonal_reliability_fusion/nnunet/sampling.py`）；
6. 浅层序列特异特征融合候选（Research Plan §8.10 的 `feature_*_positive_sampling`：三个参数不共享
   的浅层 3×3×3 stem + 1×1×1 投影回 3 通道，可选 feature gate；骨干仍由 plans 构建）；
7. 一个统一训练入口 `scripts/train/train_nnunet.py`；
8. 最少量的数据准备代码 `scripts/data/prepare_picai_nnunet.py`。

**不重复实现**第二套 trainer / checkpoint / 滑窗推理 / 验证器 / optimizer / 学习率调度器，
也不复制 nnU-Net 的 U-Net encoder/decoder。项目侧 dataloader 只允许**继承或包装**
`nnUNetDataLoader`、复用预处理 `class_locations`，并且训练采样与验证采样严格分离
（验证 loader 保持原生）。规则细则见 `AGENTS.md` §5。

### 十个 variant

| variant | Trainer 类 | 数据集 | 输入通道 | 网络 / 损失 / 采样 |
|---|---|---|---|---|
| `baseline` | `nnUNetTrainerPICAI_FLCE_NoFFT` | Dataset605 | 3（T2W/ADC/HBV） | 原生 `PlainConvUNet` + PI-CAI Focal+CE + 原生采样 |
| `optimized_baseline` | `nnUNetTrainerPICAI_DiceCE_NoFFT` | Dataset605 | 3（T2W/ADC/HBV） | 原生 `PlainConvUNet` + 原生 Dice+CE + 原生采样 |
| `image_gate` | `nnUNetTrainerPICAI_ImageGate` | Dataset605 | 3（T2W/ADC/HBV） | gate(3→3 权重) + 原生 backbone（Focal+CE）+ 原生采样 |
| `anatomy_gate` | `nnUNetTrainerPICAI_AnatomyGate` | Dataset606 | 5（+PZ/TZ） | gate(MRI+PZ/TZ→3 权重) + 原生 backbone（Focal+CE）+ 原生采样 |
| `positive_sampling` | `nnUNetTrainerPICAI_FLCE_PositiveSampling_NoFFT` | Dataset605 | 3（T2W/ADC/HBV） | 原生 `PlainConvUNet` + PI-CAI Focal+CE + **每批固定一个阳性病灶 patch** |
| `image_gate_positive_sampling` | `nnUNetTrainerPICAI_ImageGate_PositiveSampling_NoFFT` | Dataset605 | 3（T2W/ADC/HBV） | gate(3→3 权重) + 原生 backbone（Focal+CE）+ **阳性病例采样** |
| `anatomy_gate_positive_sampling` | `nnUNetTrainerPICAI_AnatomyGate_PositiveSampling_NoFFT` | Dataset606 | 5（+PZ/TZ） | gate(MRI+PZ/TZ→3 权重) + 原生 backbone（Focal+CE）+ **阳性病例采样** |
| `feature_no_gate_positive_sampling` | `nnUNetTrainerPICAI_FeatureNoGate_PositiveSampling_NoFFT` | Dataset605 | 3（T2W/ADC/HBV） | 3×独立浅层 stem + 1×1×1 投影 + 原生 backbone（**无 gate**）+ Focal+CE + **阳性病例采样** |
| `feature_image_gate_positive_sampling` | `nnUNetTrainerPICAI_FeatureImageGate_PositiveSampling_NoFFT` | Dataset605 | 3（T2W/ADC/HBV） | 同 stem/投影 + feature gate(3C_s→3 尺度) + 原生 backbone + Focal+CE + **阳性病例采样** |
| `feature_anatomy_gate_positive_sampling` | `nnUNetTrainerPICAI_FeatureAnatomyGate_PositiveSampling_NoFFT` | Dataset606 | 5（+PZ/TZ） | 同 stem/投影 + feature gate(3C_s+PZ/TZ→3 尺度，PZ/TZ 只进 gate) + 原生 backbone + Focal+CE + **阳性病例采样** |

gate 结构：`Conv3d(in,hidden,1) → InstanceNorm3d → LeakyReLU → Conv3d(hidden,3,1)`，末层
weight/bias 零初始化；`weights = softmax(logits,1)`（W，通道和为 1，`compute_weights`）、
`scales = 3*weights`（S，通道和为 3，`compute_gate_scales`）、`gated = mri*scales`。
零初始化时 `scales` 恒为 1，新模型初始行为与原生 baseline **逐值一致**。anatomy_gate 只把加权后的
前 3 个 MRI 通道送入 backbone，PZ/TZ 不进入分割 backbone（禁止 WG），且进入 gate 前 clamp(0,1)。

十个 Trainer 类名互不相同，nnU-Net 据此生成互不覆盖的 output folder。

#### `optimized_baseline`（**用户已停止 / 中止**）

- `baseline` 是**已完成**的 PI-CAI Focal+CE 基线（N0）；
- `optimized_baseline` 是**原生 nnU-Net Dice+CE** 的单变量 baseline，用于考察「N0 长期停留在
  背景预测」是否由损失选择导致；
- 它**不含 gate**：网络仍由 plans 构建为原生 `PlainConvUNet`（3 个 MRI 通道）；
- 除损失外，数据集、split、patch size、batch size、增强、前景采样、deep supervision、SGD、
  PolyLR、初始学习率、epoch 数、checkpoint、validation、inference 与 `baseline` 完全一致；
- Dice+CE 是 **nnU-Net v2.6.2 原生实现**（`DC_and_CE_loss` + `MemoryEfficientSoftDiceLoss`，
  deep supervision 由原生 `DeepSupervisionWrapper` 加权），项目**没有自写任何损失**；
- **该训练已被用户中止**：2026-09-22 00:48 UTC 启动，最后完整完成 epoch 376（epoch 377 已开始
  但未完成），最大 pseudo Dice 约 0.0008，从未超过 0.05；**没有完整 actual validation**，
  **不是完成的训练、不是正式模型结果**，其 checkpoint 保留但不得续训、不得引用为结果；
- Focal+CE 的 gate 分支与 Dice+CE baseline **不能混用**来归因 gate 的收益：若将来要在优化分支下
  比较 gate，image/anatomy gate 也必须改用相同 Dice+CE（本轮不实现这些类）。

#### `positive_sampling`（**已完成**：2026-09-22 07:16 → 22:31 UTC 训练，22:42 UTC validation）

- 与 `baseline` **唯一**的差异是训练集病例/patch 采样：训练 loader 为项目侧
  `PositiveCaseDataLoader`（继承固定 v2.6.2 原生 `nnUNetDataLoader`，只覆盖 `get_indices` 与
  槽位 force-foreground 决策）；
- batch size 2 时：slot 0 从全部训练病例均匀抽取（保留阴性监督），slot 1 固定来自当前 fold 的阳性
  训练病例并强制病灶中心裁剪 ⇒ 50% patch 槽位保证含病灶、100% batch 至少一个病灶中心 patch；
- 阳性病例只由**当前 fold 训练集**的预处理 `class_locations` 识别（含 `label_manager.
  foreground_labels`），任何 metadata 缺失/不合法都 fail-closed；
- 验证 loader 保持原生 `nnUNetDataLoader` 与原生采样，不做任何按验证标签的阳性重采样；
- loss（`0.5*Focal(gamma=2)+0.5*CE`）、网络（原生 `PlainConvUNet`，3 个 MRI 通道）、NoFFT 增强、
  optimizer（SGD+Nesterov）、PolyLR、1000 epochs、patch size、batch size、deep supervision、
  checkpoint、validation、inference 全部与 `baseline` 一致；不含 gate、不含 PZ/TZ；
- 输出目录为独立新目录，不覆盖 baseline / ImageGate / DiceCE 的任何产物；
- **当前状态**：**训练与 validation 均已完成**（2026-09-22 07:16 UTC 启动 → 22:31 UTC 训练结束，
  1000 epoch 全部完成 → 22:42 UTC actual validation 完成）。运行事实与真实数字见
  `docs/Training_Log.md`。

#### 公平匹配的 gate 组合（`image_gate_positive_sampling` **已完成训练 + validation**；`anatomy_gate_positive_sampling` **代码已实现，尚未训练**）

Research Plan 规定核心门控比较必须固定损失、病例/patch 采样、增强、optimizer、LR scheduler、
split 与随机种子策略。旧 `baseline` ↔ 旧 `image_gate`（同为 FLCE 与原生采样）本身仍是**原生采样
条件下**的受控 RQ1 对照，但它们**不能与 `positive_sampling` 分支混合比较**（采样不同），且单次
运行与 baseline 早期长期全背景的优化异常会限制结论强度。为在同一采样条件下比较门控，新增两个
组合 Trainer（不改动任何旧类行为，不复用任何旧 checkpoint，输出目录由类名自然隔离）：

| 比较 | 模型 A | 模型 B | 边界 |
|---|---|---|---|
| RQ1（阳性采样分支内） | `positive_sampling` | `image_gate_positive_sampling` | 仅归因 image gate |
| RQ2（阳性采样分支内） | `image_gate_positive_sampling` | `anatomy_gate_positive_sampling` | **预期主要差异**为 anatomy 条件（PZ/TZ）与数据集（605 vs 606）；逐数组审计完成前不作「唯一差异」声明 |
| RQ1（**原生采样**条件下的受控对照） | 旧 `baseline` | 旧 `image_gate` | 仅归因 image gate；不能与阳性采样分支混合比较，单次运行与早期优化异常限制结论强度 |

- `image_gate_positive_sampling`：Dataset605，3 通道；FLCE + image gate + 阳性病例采样；
- `anatomy_gate_positive_sampling`：Dataset606，5 通道；FLCE + anatomy gate（PZ/TZ 只进门控、
  不进分割 backbone，进入前 clamp(0,1)）+ 相同采样；
- 两者的 loss、增强（NoFFT）、optimizer、PolyLR、epoch、deep supervision、checkpoint、
  validation、inference 与验证 loader（原生 `nnUNetDataLoader`）全部一致；anatomy 的强度增强
  仍只作用于前 3 个 MRI 通道；
- **RQ2 的边界**：两者除门控条件外还使用不同数据集（Dataset605 vs Dataset606）。配置层面
  （split / spacing / patch size / batch size / 前三个 MRI 通道 normalization 与 fingerprint）
  已核对一致（见下），但在完成前三个 MRI 通道的**逐数组一致性审计**之前，不能宣称唯一差异
  是 PZ/TZ；
- **`image_gate_positive_sampling` 已完成训练 + actual validation**（2026-09-23 06:55 → 22:47 UTC；
  1000 epoch；223 例 validation 已落盘）：nnU-Net `foreground_mean.Dice` = 0.20935、阳性病例
  macro Dice 0.2991、micro Dice 0.5397、`positive_voxel_precision` 0.8341；**RQ1 配对比较的结论、
  全部指标与复现命令见 `docs/Findings.md` §3.10**（配对均值 delta −0.0077、CI95 含 0 ⇒
  本次单次运行**未观察到明确的分割增量效用**，但工作点更保守：precision ↑、假阳 ↓、漏检略增）；
  运行事实见 `docs/Training_Log.md`，逐实验细节见 `docs/experiments/image_gate_positive_sampling.md`；
- **`anatomy_gate_positive_sampling` 尚未训练**：其启动前置条件是 Dataset605/606 前三 MRI 通道的
  真实数据逐数组审计取得 `MRI_ARRAY_AUDIT_PASS`（工具与命令见下节）。

#### 浅层特征融合（`feature_*_positive_sampling`，**代码已实现，三个都尚未训练**）

Research Plan §8.10 的「浅层序列特异特征融合候选」。三个条件共享**完全相同**的浅层编码与投影结构、
FLCE 损失、阳性病例采样、增强、optimizer、LR scheduler、epoch、deep supervision、checkpoint、
validation 与滑窗推理，构成一组同层级比较：

| 比较 | 模型 A | 模型 B | 边界 |
|---|---|---|---|
| 浅层编码/投影本身的作用 | `feature_no_gate_positive_sampling` | `feature_image_gate_positive_sampling` | 仅归因 feature gate（同 stem/投影/backbone） |
| 浅层解剖条件的作用（RQ4） | `feature_image_gate_positive_sampling` | `feature_anatomy_gate_positive_sampling` | 仅归因 gate 条件（PZ/TZ）；数据集 605 vs 606 与 16 个 gate 参数的差异见下 |
| 表征层级的作用 | `positive_sampling` | `feature_no_gate_positive_sampling` | 浅层编码 + 投影引入的额外容量与输入分布变化 |

网络结构（`networks.FeatureFusionNNUNet`）：

- T2W / ADC / HBV 各有**一个参数互不共享**、**不下采样**的浅层 stem：两层 3×3×3 `Conv3d`
  （stride 1、padding 1，空间尺寸逐值保持），每层后接 `InstanceNorm3d` 与 LeakyReLU；
  统一通道常量 `C_s = FEATURE_STEM_CHANNELS = 8`。两层卷积的名义直接感受野为 5×5×5 体素，
  但 InstanceNorm 引入整块统计依赖，**不能**把 5×5×5 当作完整有效感受野或物理毫米邻域；
- 三路特征**拼接**（而非逐通道求和：三个独立 stem 的同序号通道没有语义对齐保证）后，用 1×1×1
  卷积**投影到恰好 3 个通道**，再交给由 plans 构建的原生 `PlainConvUNet`（骨干只接收 3 通道）；
- feature gate 读**拼接后的 stem 特征**（`3*C_s` 通道）并输出恰好 3 个 logit，`W = softmax(A,1)`、
  `S = 3W`；`S_m` 在序列 m 的**全部 `C_s` 个通道上共享**（每个序列只有一个尺度场）；
- `PZ/TZ` 只进入 feature gate（anatomy 条件的 gate 额外读 clamp(0,1) 后的 PZ/TZ），
  **不进入任何 stem、投影层或 backbone**（禁止 WG）；进入 gate 前 clamp(0,1)；
- feature anatomy 的强度增强仍只作用于前 3 个 MRI 通道，空间/镜像增强仍同步作用于全部通道。

参数量与显存（合成构建实测的参数量 + **解析估算**的显存，不是实测显存）：

| 模块 | 参数 |
|---|---|
| 每个 stem | 1 992（`conv1`+`conv2`+两组 InstanceNorm） |
| 三个 stem 合计 | 5 976 |
| 1×1×1 投影（24→3） | 75 |
| feature gate（image，24→8→3） | 243 |
| feature gate（anatomy，26→8→3） | 259 |
| **feature 模块合计** | no-gate **6 051** / image **6 294** / anatomy **6 310** |

对照真实 backbone（`PlainConvUNet`，约 4.46×10⁷ 参数），feature 模块约占 **0.014%**。
`C_s` 取值依据：patch `16×320×320`、batch size 2、fp32 时，`C_s = 8` 的 stem 顶层激活约 0.63 GB，
计入 InstanceNorm 与 autograd 中间量约 1.7–2.0 GB；`C_s = 16` 约为其两倍（3.2–4.0 GB），
而原生 stage 0 在满分辨率上的对照值约 0.84 GB。因此取保守的 `C_s = 8`；**未修改**
`nnUNetPlans.json`。

零初始化的边界（Research Plan §8.10 末段、H5）：

- feature gate 末层零初始化 ⇒ 初始 `S ≡ 1`，因此 feature gate 网络在**共享相同 stem / 投影 /
  backbone 权重**时与 `feature_no_gate` **逐值一致**（有单元测试守护）；
- 但这**不**等于原生输入级 nnU-Net：`E_m` 与 `P_ψ` 本身仍然改变进入 backbone 的输入分布。
  即使把 backbone 权重逐值对齐，两者的输出也不相同（有单元测试守护）。

参数量差异（**不得**声称 image 与 anatomy 参数量严格相同）：两者只有 `gate.conv1.weight` 的输入维
不同（`3*C_s+2` vs `3*C_s`），差值恒为 `gate_hidden_channels × num_prior_channels = 8 × 2 = 16`
个参数，由 `networks.feature_gate_parameter_delta()` 显式记录。这里**没有**采用零值占位通道去把
两者参数量凑成相等。

其他边界：三个条件与输入级 `image_gate` / `anatomy_gate` **属于不同表征层级**，不能混在同一张
归因表里；feature 版本相对原生 nnU-Net 的全部差异包含"更多参数 + 不同输入分布"，因此不能把全部
收益归因于解剖条件（H6）。**三个条件尚未训练**，本节不含任何训练结果。

#### Dataset605 与 Dataset606 的一致性边界

配置层面已核对一致：冻结 split（1277 train / 223 val，**具体病例 ID 与顺序逐 fold 相同**）、
`3d_fullres` 的 spacing `[3.0, 0.5, 0.5]`、patch size `[16, 320, 320]`、batch size 2、前三个 MRI
通道的 normalization scheme 与 per-channel 强度 fingerprint 均一致；两个 `nnUNetPlans.json` 的
全部差异只有 `dataset_name`、新增的第 4/5 通道（PZ/TZ，`noNorm`）及其两条 per-channel 属性。
但**这不等同于完成真实数据逐数组审计**：尚未对两数据集的 MRI 体素做逐数组比较，也没有重新
物化 Dataset606。任何把两个数据集的结果并列解释为「只差 PZ/TZ 通道」的结论都必须先完成该审计。

#### Dataset605 ↔ Dataset606 前三 MRI 通道逐数组审计（RQ2 前置，**真实数据尚未执行**）

工具：`scripts/data/audit_dataset605_606_mri_equivalence.py`（只读、fail-closed，含纯合成单元测试）。
它逐病例比较预处理后前三个 MRI 通道（T2W/ADC/HBV）的 `np.array_equal` 与浮点差值统计，并审计
病例集合、`splits_final.json`、lesion label；**PZ/TZ 不参与数值一致性判定**（仅记录取值范围）。
不修复数据、不重建 Dataset606、不重新 preprocessing。

```bash
cd /opt/data/private/lm/my-projects && conda activate lm && source scripts/env_nnunet.sh

python scripts/data/audit_dataset605_606_mri_equivalence.py \
    --output outputs/reports/dataset605_606_mri_equivalence_audit.json
```

成功判据：退出码 `0`、`status=MRI_ARRAY_AUDIT_PASS`、`n_cases_mismatch=0`、
`per_channel.*.exact_equal_cases = n_cases_checked`（PASS 之外退出码为非零，报告仍会写出）。
报告已存在时默认拒绝覆盖，需显式加 `--overwrite`。`--max-cases N` 只用于烟雾测试，此时 status
恒为 `MRI_ARRAY_AUDIT_PARTIAL`，**永不判 PASS**。

#### 为什么暂时不采用 DiceTopK / batch_dice=true / 类别权重 / patch size 或优化器修改

另一个 PI-CAI 项目的证据表明：**Dice+CE + 固定阳性采样**在早期仍可能停留在全背景（pseudo Dice
≈0）；而 **Focal+CE + 每批固定一个阳性病灶 patch + 正常长周期 PolyLR** 能够较早脱离全背景
（epoch 4 首次非零、epoch 19 约 0.11）。因此本轮只做**单变量**改动（只改采样），保留已经证明
可学的 Focal+CE，避免同时混入多个不可归因的修改。

## 环境

固定使用 conda 环境 **`lm`**（Python：`/root/anaconda3/envs/lm/bin/python`）。涉及 nnU-Net 的
任何命令前必须：

```bash
cd /opt/data/private/lm/my-projects
conda activate lm
source scripts/env_nnunet.sh   # 校验 lm 环境 + 固定 nnU-Net 源码，并把 nnUNet_* 指向项目内 workdir/outputs
```

`scripts/env_nnunet.sh` 把 `nnUNet_raw`/`nnUNet_preprocessed` 指向 `workdir/`、`nnUNet_results`
指向 `outputs/`，并把 `PYTHONPATH` 前置到 `third_party/nnUNet:src`。

## 训练与推理命令

长任务（数据准备、planning/preprocessing、训练、validation、inference、evaluation）一律由研究者本人
运行。每条命令前先：

```bash
cd /opt/data/private/lm/my-projects && conda activate lm && source scripts/env_nnunet.sh
```

### 十个 variant 的训练

```bash
python scripts/train/train_nnunet.py baseline 605 3d_fullres 0            # = 已完成的 N0
python scripts/train/train_nnunet.py optimized_baseline 605 3d_fullres 0  # 原生 Dice+CE（用户已中止，勿续训）
python scripts/train/train_nnunet.py image_gate 605 3d_fullres 0
python scripts/train/train_nnunet.py anatomy_gate 606 3d_fullres 0        # 需先准备并预处理 Dataset606
CUDA_VISIBLE_DEVICES=0 python scripts/train/train_nnunet.py positive_sampling 605 3d_fullres 0
# 公平匹配的 gate 组合（均与 positive_sampling 共用 FLCE + 阳性病例采样）
# image_gate_positive_sampling 已完成（2026-09-23，见 Training_Log / Findings §3.10）
CUDA_VISIBLE_DEVICES=0 python scripts/train/train_nnunet.py image_gate_positive_sampling 605 3d_fullres 0
# anatomy_gate_positive_sampling 尚未训练；启动前必须先做 Dataset605/606 MRI 数组审计（见「一致性边界」）
CUDA_VISIBLE_DEVICES=0 python scripts/train/train_nnunet.py anatomy_gate_positive_sampling 606 3d_fullres 0
# 浅层特征融合三条件（尚未训练；同一 stem/投影/损失/采样，见「浅层特征融合」小节）
CUDA_VISIBLE_DEVICES=0 python scripts/train/train_nnunet.py feature_no_gate_positive_sampling 605 3d_fullres 0
CUDA_VISIBLE_DEVICES=0 python scripts/train/train_nnunet.py feature_image_gate_positive_sampling 605 3d_fullres 0
CUDA_VISIBLE_DEVICES=0 python scripts/train/train_nnunet.py feature_anatomy_gate_positive_sampling 606 3d_fullres 0
```

> `positive_sampling`、两个 `*_positive_sampling` 组合与三个 `feature_*_positive_sampling`
> 都是**从零训练**：不加 `--continue-training`，不读取 DiceCE/baseline/gate 的任何 checkpoint，
> 也不会覆盖任何既有目录。训练开始后日志应出现实际训练病例数、阳性病例数与
> `positive_cases_per_batch=1`、`guaranteed_positive_patch_fraction=0.5`；若实际阳性病例数与
> 预处理不一致，采样器会直接报错而不是静默继续。
>
> 三个 `feature_*` 条件的显存高于原生骨干（`C_s = 8` 时估算额外约 1.7–2.0 GB，见「浅层特征融合」
> 小节）；首次运行前应确认显卡余量，尤其是 Dataset606（5 通道）。
>
> 注意：同一时间只运行一条训练；GPU 号用 `CUDA_VISIBLE_DEVICES` 控制。

可选参数：`--continue-training`、`--validation-only`、`--export-validation-probabilities`、
`--device {cuda,cpu,mps}`。入口只做 variant → Trainer 类映射后调用 nnU-Net 官方 `run_training`，
不含任何自写训练循环。

### baseline 只验证 / 导出概率

```bash
python scripts/train/train_nnunet.py baseline 605 3d_fullres 0 --validation-only
python scripts/train/train_nnunet.py baseline 605 3d_fullres 0 --validation-only --export-validation-probabilities
```

### 数据准备（anatomy_gate 前置；fail-closed）

```bash
python scripts/data/prepare_picai_nnunet.py baseline                   # 组织 Dataset605_PICAI raw
python scripts/data/prepare_picai_nnunet.py zonal --zonal-source yuan  # 组织 Dataset606_PICAI_Zonal raw（含 PZ/TZ）
nnUNetv2_plan_and_preprocess -d 606 --verify_dataset_integrity         # 官方 planning + preprocessing
python scripts/data/prepare_picai_nnunet.py splits --dataset-id 606 --dataset-name PICAI_Zonal  # 写入固定 fold
```

任一 failed/conflict 会先打印完整汇总再非零退出，且**不发布** `dataset.json`；`numTraining` 等于磁盘
实际完整病例数；`splits_final.json` 冲突非零退出。

### 推理（官方预测器 + 项目 Trainer 解析）

```bash
python scripts/inference/predict_nnunet.py \
    -i <输入影像目录> -o <输出目录> \
    -m outputs/nnUNet_results/Dataset605_PICAI/nnUNetTrainerPICAI_FLCE_NoFFT__nnUNetPlans__3d_fullres \
    -f 0 -device cuda
```

`-m` 换成对应 variant 的输出目录即可（image_gate / anatomy_gate / positive_sampling /
image_gate_positive_sampling）。该入口在
进程内把 checkpoint 的
`trainer_name` 映射到项目 Trainer 类，随后调用 nnU-Net 官方 `nnUNetPredictor`；滑窗推理与导出全部
沿用官方实现，未自写推理器，未修改 `third_party/nnUNet`。其余参数与官方
`nnUNetv2_predict_from_modelfolder` 一致（`--help` 即官方帮助）。

### 病灶分割评估（离线，不重跑 validation）

对**已经存在**的 validation 产物做统一口径的离线分析，使多个模型在完全相同的指标定义下可比。
不重写 nnU-Net 的 validation / inference / prediction。

```bash
# summary 模式：只读 summary.json，不读 NIfTI
python scripts/evaluate_segmentation.py \
    --model baseline=outputs/nnUNet_results/Dataset605_PICAI/nnUNetTrainerPICAI_FLCE_NoFFT__nnUNetPlans__3d_fullres/fold_0 \
    --model image_gate=outputs/nnUNet_results/Dataset605_PICAI/nnUNetTrainerPICAI_ImageGate__nnUNetPlans__3d_fullres/fold_0 \
    --model optimized_baseline=outputs/nnUNet_results/Dataset605_PICAI/nnUNetTrainerPICAI_DiceCE_NoFFT__nnUNetPlans__3d_fullres/fold_0 \
    --mode summary --bootstrap-resamples 10000 --seed 20260922 \
    --output outputs/reports/segmentation_metrics.json

# full 模式：长任务，只能在训练结束后由研究者本人运行。逐例读 NIfTI，追加 mm³ 物理体积、
# HD95 / ASSD / NSD@τ、体积分层与连通域分析。
# 容差与体积阈值必须由研究者事先确定，本工具不提供、也不推荐任何取值：用带引号的环境变量
# 传入，未设置时立即报错而不是静默取默认值。
: "${NSD_TOLERANCE_MM:?请先设置 NSD_TOLERANCE_MM}"
: "${VOLUME_THRESHOLDS_MM3:?请先设置 VOLUME_THRESHOLDS_MM3}"
python scripts/evaluate_segmentation.py \
    --model baseline=outputs/nnUNet_results/Dataset605_PICAI/nnUNetTrainerPICAI_FLCE_NoFFT__nnUNetPlans__3d_fullres/fold_0 \
    --model image_gate=outputs/nnUNet_results/Dataset605_PICAI/nnUNetTrainerPICAI_ImageGate__nnUNetPlans__3d_fullres/fold_0 \
    --model optimized_baseline=outputs/nnUNet_results/Dataset605_PICAI/nnUNetTrainerPICAI_DiceCE_NoFFT__nnUNetPlans__3d_fullres/fold_0 \
    --mode full --nsd-tolerance-mm "${NSD_TOLERANCE_MM}" \
    --volume-thresholds-mm3 "${VOLUME_THRESHOLDS_MM3}" \
    --output outputs/reports/segmentation_metrics_full.json
```

> `optimized_baseline` 已被用户中止（最后完整完成 epoch 376），**没有完整的
> `validation/summary.json`**，不是正式模型结果；上面两条命令都必须先删掉
> `--model optimized_baseline=...` 那一行。`positive_sampling` 的训练与 validation 均已完成
> （`validation/summary.json` 已存在），可以按同样格式加上 `--model positive_sampling=...`；
> 三个 `feature_*_positive_sampling` 条件**尚未训练**，没有产物可加。
> `full` 模式是长任务，且只能由研究者在训练**结束之后**运行。

- **长任务由研究者本人运行**；`full` 模式逐例读取 NIfTI，属耗时任务。
- 核心指标是**病灶分割**指标：`positive_macro_dice`（完全漏分记 0）、`positive_micro_dice`、
  `positive_voxel_recall`、`positive_voxel_precision`、`all_prediction_voxel_precision`、
  完全漏分结构、阴性病例假阳；`full` 模式追加 RVE/ARVE、HD95、ASSD、NSD@τ（mm）、
  病灶体积分层与连通域失败分析。
- **NSD@τ 是 surfel-area weighted 物理表面度量**：使用 DeepMind `surface-distance` 0.1 官方实现，
  按 surfel 物理面积（mm²）加权，不是表面体素计数；spacing 为数组轴序 (z, y, x) 的真实 mm。
  缺少该包时 fail-closed（**不会**静默退回旧算法）。
- **AUROC / average precision / FROC / PI-CAI challenge score 不属于本研究的主评价**，本工具
  不计算它们。本研究只使用 PI-CAI 数据集做**病灶分割**研究，不做 PI-CAI 病灶检测评价。
- 两种 precision 严格区分，不得笼统混用：`positive_voxel_precision`（仅阳性病例，论文核心口径）
  与 `all_prediction_voxel_precision`（分母含阴性病例假阳体素）。
- **summary 模式的严格校验**：逐例校验 TP/FP/FN/TN/n_pred/n_ref 为非负整数（拒绝小数、负数、
  NaN/Inf），强制 `TP+FP == n_pred` 与 `TP+FN == n_ref`，Dice **由 TP/FP/FN 重算**并与 summary
  中的值核对（不一致即失败）；真阴（空-空）病例的 Dice 必须为空。宏平均使用重算值，不信任
  summary 里的 Dice。
- **full 模式覆盖全部病例**：阳性、假阳阴性、真阴三类都读取 prediction 与 reference，逐例校验
  size / spacing / origin / direction 与 0/1 标签，由掩膜重算六项计数并与 summary **逐项**核对；
  `n_ref=0` 的病例其 reference 必须实际为空；**真阴病例（n_pred=0）不得跳过文件与几何检查**。
  每例只读一次，不重复读盘。
- **逐例异常按病例收集**：一例从读掩膜到计数核对、体积、`surface_metrics`、
  `component_analysis` / `_pred_component_stats`、entry 构造的**全部步骤**都在同一个异常边界内。
  某一例的 MedPy / SciPy / 体积 / 连通域失败只记录 `[model] case: 异常类型: 信息` 并继续处理其余
  病例，全部跑完后再统一报错（写明总数）。`KeyboardInterrupt` / `SystemExit` 不会被捕获。
  失败病例不会留下任何伪造的「已验证」结果。
- **结构化结束汇总**：成功与失败（含所有非 `EvaluationError` 的普通异常）都打印**恰好一份**
  `status / mode / phase / unit / ok / failed / skipped / elapsed / output`，异常随后**原样重抛**。
  计数单位**不能由 mode 推断**（`full` 可能在模型加载或可比性检查阶段就失败），而由 `phase` 决定：

  | phase | unit | ok / failed / skipped |
  |---|---|---|
  | `preflight` | not_applicable | 尚未开始计数（参数校验阶段） |
  | `model_loading` | `model` | 已加载 / 加载失败的**模型**数；skipped 不适用 |
  | `comparability` | `model` | 沿用模型计数（**不得**改标为 model-case） |
  | `full_cases` | `model-case` | 逐例通过 / 失败 / 未进入检查的**对**数 |
  | `publishing` / `completed` | 沿用上一阶段 | 沿用上一阶段 |

  只有真正进入逐例循环前才切到 `unit=model-case` 并重置计数；不可知的量渲染为
  `not_applicable`，**不伪造成 0**。`elapsed` 取自 `time.monotonic()` 差值。汇总只做统计，
  不参与指标计算，也不改变 fail-closed 判定。`--help` 不打印汇总（帮助不是一次运行）。
- fail-closed：任一模型缺产物、病例集合/身份/`n_ref` 不一致、任一例校验不通过时，先打印**完整**
  错误汇总（写明总条数，若截断会明确标注未显示条数）再非零退出，且**不发布**输出 JSON、不留临时
  文件；`--output` 指向已存在文件时拒绝覆盖；`--nsd-tolerance-mm` 必须是正的有限数（nan/inf 拒绝）。

### Prostate158 独立外测（跨域外部数据；原始影像只读）

用 Prostate158 在**不重训、不微调、不新建 Dataset ID** 的前提下外测已完成的 Dataset605 模型。
通道映射固定 `_0000=t2 / _0001=adc / _0002=dwi`；**Prostate158 DWI 只是第三通道（训练时为
HBV）的跨域输入，不宣称与 HBV 等价**。原始 Prostate158 影像与标注绝不修改：默认只在独立目录
建相对软链接；只有三通道网格不一致且物理坐标关系可信、覆盖完整，并**显式**加
`--allow-resample` 时，才在派生目录内对 adc/dwi 做线性重采样（T2 为参考网格；绝不自动配准）。
主参考固定 `adc_tumor_reader1`；`adc_tumor_reader2` 仅在可核对子集上做单独的读者敏感性分析。
**official test / 原作者 train / valid 三队列分开报告，不合并。** 以下均为长任务，由研究者运行。

```bash
# 0) 环境（每条命令前都应处于此状态）
cd /opt/data/private/lm/my-projects && conda activate lm && source scripts/env_nnunet.sh

P158=/opt/data/private/lm/data/Prostate/Prostate158
MODEL=outputs/nnUNet_results/Dataset605_PICAI/nnUNetTrainerPICAI_FLCE_PositiveSampling_NoFFT__nnUNetPlans__3d_fullres
# 注意：先固定模型与 checkpoint（默认 checkpoint_final.pth），看到 Prostate158 结果后不得更换。

# 1) official test 只读审计（不生成推理输入；逐例 header/标签/网格，进度条 + 结束汇总）
python scripts/data/prepare_prostate158_external.py audit \
    --csv  "$P158/test/prostate158_test/test.csv" --queue test \
    --output outputs/reports/prostate158_audit_test.json
# 成功判据：退出码 0，汇总 failed=0；查看 linkable / grid_mismatch(resample_candidate) 计数。
# 停止条件：failed>0（退出码 2）时先修数据问题；有 grid_mismatch 时**先停下来看审计明细再决定**，
#           绝不直接上 --allow-resample。

# 2) 默认 prepare：仅 linkable 病例，软链接到独立新目录（标签不进输入目录，只写清单）
python scripts/data/prepare_prostate158_external.py prepare \
    --csv  "$P158/test/prostate158_test/test.csv" --queue test \
    --output-dir workdir/external/prostate158_test
# 成功判据：退出码 0，images/ 下每例三个 _0000/_0001/_0002 软链接，external_input_manifest.json
#           记录参考标签路径与 skipped_cases；输出目录已存在非空会被拒绝（换新目录，勿加覆盖）。
# 仅当审计确认所有 grid_mismatch 都 resample_candidate=true 且你明确接受派生重采样时，
# 才改用：上条命令末尾加 --allow-resample（派生真实文件写入同一独立目录，原始影像不动）。

# 3) 用现有官方预测入口推理（不改滑窗；nnU-Net 按 plans 自动预处理/归一化）
python scripts/inference/predict_nnunet.py \
    -i workdir/external/prostate158_test/images \
    -o outputs/external_predictions/prostate158_test/positive_sampling \
    -m "$MODEL" -f 0 -device cuda
# 成功判据：每个病例生成一个 <P158_test_NNN>.nii.gz；数量与 manifest cases 一致。

# 4) 独立外测分割评估（清单驱动，不读/不写 validation/summary.json；只评分割，无 AUROC/FROC）
python scripts/evaluate_external_segmentation.py \
    --manifest workdir/external/prostate158_test/external_input_manifest.json \
    --model positive_sampling=outputs/external_predictions/prostate158_test/positive_sampling \
    --reader reader1 \
    --output outputs/reports/external_prostate158_test_positive_sampling_reader1.json \
    --note "Dataset605 FLCE_PositiveSampling, checkpoint_final, Prostate158 official test (cross-domain channel3=DWI)"
# 成功判据：退出码 0；输出含阳性 macro/median/micro Dice、召回、两种 precision、完全漏分、
#           阴性假阳病例数/体积；evaluation_grid_alignment 记录是否发生最近邻评估对齐。
# 停止条件：病例缺失、标签非 0/1、几何不可信或 FOV 不覆盖 → 非零退出且不发布 JSON。
# 多模型比较：重复 --model name=目录；病例集合/读者/清单版本不一致会被拒绝（不在交集上退化）。

# 5) 读者敏感性（仅对有 adc_tumor_reader2 的子集；不与 reader1 混组）
python scripts/evaluate_external_segmentation.py \
    --manifest workdir/external/prostate158_test/external_input_manifest.json \
    --model positive_sampling=outputs/external_predictions/prostate158_test/positive_sampling \
    --reader reader2 \
    --output outputs/reports/external_prostate158_test_positive_sampling_reader2.json

# 6) 原作者 train / valid 队列（阴性病例主要在这里；只做阴性假阳分析时同样分开跑）
python scripts/data/prepare_prostate158_external.py audit  --csv "$P158/train/prostate158_train/train.csv" --queue train --output outputs/reports/prostate158_audit_train.json
python scripts/data/prepare_prostate158_external.py prepare --csv "$P158/train/prostate158_train/train.csv" --queue train --output-dir workdir/external/prostate158_train
python scripts/inference/predict_nnunet.py -i workdir/external/prostate158_train/images -o outputs/external_predictions/prostate158_train/positive_sampling -m "$MODEL" -f 0 -device cuda
python scripts/evaluate_external_segmentation.py \
    --manifest workdir/external/prostate158_train/external_input_manifest.json \
    --model positive_sampling=outputs/external_predictions/prostate158_train/positive_sampling \
    --output outputs/reports/external_prostate158_train_positive_sampling.json
# valid 队列把上面的 train 全部替换为 valid（CSV: train/prostate158_train/valid.csv）。
# official test 与 train/valid 的结果**不得合并后当作官方测试集结果引用**。
```

## 输出目录

以下目录由 Trainer 类名自动隔离，互不覆盖（不要手工拼接输出路径）：

- baseline（N0，已完成，勿覆盖）：
  `outputs/nnUNet_results/Dataset605_PICAI/nnUNetTrainerPICAI_FLCE_NoFFT__nnUNetPlans__3d_fullres/fold_0/`
- optimized_baseline（**用户已中止**，2026-09-22 00:48 UTC 启动；无 actual validation，
  checkpoint 保留但勿续训、勿引用为结果）：
  `outputs/nnUNet_results/Dataset605_PICAI/nnUNetTrainerPICAI_DiceCE_NoFFT__nnUNetPlans__3d_fullres/fold_0/`
- image_gate：`outputs/nnUNet_results/Dataset605_PICAI/nnUNetTrainerPICAI_ImageGate__nnUNetPlans__3d_fullres/fold_0/`
- anatomy_gate：`outputs/nnUNet_results/Dataset606_PICAI_Zonal/nnUNetTrainerPICAI_AnatomyGate__nnUNetPlans__3d_fullres/fold_0/`
- positive_sampling（**已完成**：训练 2026-09-22 07:16 → 22:31 UTC，validation 22:42 UTC 完成）：
  `outputs/nnUNet_results/Dataset605_PICAI/nnUNetTrainerPICAI_FLCE_PositiveSampling_NoFFT__nnUNetPlans__3d_fullres/fold_0/`
- image_gate_positive_sampling（**已完成**：训练 2026-09-23 06:55 → 22:35 UTC，validation 22:47 UTC 完成）：
  `outputs/nnUNet_results/Dataset605_PICAI/nnUNetTrainerPICAI_ImageGate_PositiveSampling_NoFFT__nnUNetPlans__3d_fullres/fold_0/`
- anatomy_gate_positive_sampling（代码已实现，**尚未训练**，目录尚未创建）：
  `outputs/nnUNet_results/Dataset606_PICAI_Zonal/nnUNetTrainerPICAI_AnatomyGate_PositiveSampling_NoFFT__nnUNetPlans__3d_fullres/fold_0/`
- feature_no_gate_positive_sampling / feature_image_gate_positive_sampling /
  feature_anatomy_gate_positive_sampling（代码已实现，**尚未训练**，目录尚未创建）：
  `outputs/nnUNet_results/Dataset605_PICAI/nnUNetTrainerPICAI_FeatureNoGate_PositiveSampling_NoFFT__nnUNetPlans__3d_fullres/fold_0/`、
  `outputs/nnUNet_results/Dataset605_PICAI/nnUNetTrainerPICAI_FeatureImageGate_PositiveSampling_NoFFT__nnUNetPlans__3d_fullres/fold_0/`、
  `outputs/nnUNet_results/Dataset606_PICAI_Zonal/nnUNetTrainerPICAI_FeatureAnatomyGate_PositiveSampling_NoFFT__nnUNetPlans__3d_fullres/fold_0/`
- legacy M0（历史保留，不再续训，勿删除/覆盖）：`outputs/checkpoints/m0_resenc/`、`outputs/metrics/m0/`、`outputs/diagnostics/m0/`

## 目录结构（最小）

```
AGENTS.md                协作规则
README.md                本文件
docs/
  Research_Plan.md       研究问题、假设、方法机制与研究边界
  Development_Log.md     代码/架构变更日志
  Training_Log.md        训练与验证事实日志
  Findings.md            跨实验的观察、问题优先级与下一步判据
  experiments/           实验详情（每个实验一份独立记录，互不引用）
scripts/
  env_nnunet.sh                  nnU-Net 运行环境
  data/prepare_picai_nnunet.py   数据准备（baseline/zonal/splits，fail-closed）
  train/train_nnunet.py          统一训练入口
  inference/predict_nnunet.py    最小预测入口（官方预测器 + 项目 Trainer 解析）
  evaluate_segmentation.py       统一病灶分割评估（离线读 validation 产物；summary/full）
src/zonal_reliability_fusion/
  nnunet/trainers.py     PI-CAI 损失 + NoFFT 修复 + 十个 Trainer + Trainer 注册表
  nnunet/sampling.py     阳性病例感知训练采样（继承原生 nnUNetDataLoader；验证 loader 保持原生）
  nnunet/networks.py     spatial modality gate / 浅层序列 stem 特征融合 与原生 backbone 包装器
  nnunet/transforms.py   PZ/TZ 增强边界（强度只作用 MRI）
  nnunet/__init__.py     check_fixed_nnunet_runtime（nnU-Net/DNA/PlainConvUNet 运行时校验）
pytest.ini               只收集 tests/（不扫描 data/workdir/outputs/third_party）
tests/                   纯合成单元测试
third_party/nnUNet/      固定 v2.6.2（只读）
data/ workdir/ outputs/  数据、预处理缓存、训练产物
```
