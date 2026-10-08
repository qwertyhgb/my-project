# Sequence-Specific Shallow Feature Fusion for Prostate Lesion Segmentation

前列腺癌（csPCa）病灶分割研究项目，采用 **nnU-Net-first** 架构。新的研究主线依次检验
T2W/ADC/HBV 的序列特异浅层局部表征、特征级自适应融合，以及 PZ/TZ 作为融合条件的增量价值。
既有输入级门控实验是初步证据。原 `feature_no_gate_positive_sampling` 已启动但缺最终验证；
另外两个原特征门控条件未运行。2026-10-06 新增独立的同区参照融合短预算探索分支，代码就绪、未训练。
研究计划见
`docs/Research_Plan.md`，既有结果见 `docs/Findings.md`。

## 同区参照融合：独立短预算探索分支（2026-10-06）

首版复用 **Dataset606_PICAI_Zonal / 3d_fullres** 的现有五通道预处理输入
`T2W / ADC / HBV / PZ / TZ`，不重新物化或预处理原始影像。PZ/TZ 来自自动分区二值掩膜，
经既有插值处理形成**分区隶属权重，不是校准概率，也不是实测几何体素占比**。当前只实现融合假设；未实现新的腺体/分区
预测网络、腺体 ROI 裁剪或完整两阶段端到端系统，不使用 WG。

| variant | Trainer 类名 | 条件 |
|---|---|---|
| `feature_no_gate_positive_sampling_100ep` | `nnUNetTrainerPICAI_FeatureNoGate_PositiveSampling_100ep_NoFFT` | 相同五通道数据，网络忽略 PZ/TZ，普通三路特征拼接 |
| `feature_anatomy_gate_positive_sampling_100ep` | `nnUNetTrainerPICAI_FeatureAnatomyGate_PositiveSampling_100ep_NoFFT` | 普通分区条件 feature gate |
| `zonal_reference_positive_sampling_100ep` | `nnUNetTrainerPICAI_ZonalReference_PositiveSampling_100ep_NoFFT` | 同区参照残差修正，固定有效参照强度 |
| `zonal_reference_adaptive_positive_sampling_100ep` | `nnUNetTrainerPICAI_ZonalReferenceAdaptive_PositiveSampling_100ep_NoFFT` | 相同修正模块，额外学习修正强度 |

四个条件均保持旧的三个独立 `C_s=8` 浅层 stem、24→3 投影和原生 plans backbone，
统一使用 FLCE、阳性采样、NoFFT 与分区强度增强保护、原生优化器/验证/滑窗。
独立 Trainer 在 optimizer 初始化前将 `num_epochs=100`，所以原生 PolyLR 总周期也是 100；
每 epoch 250 次更新，总计 25,000 次。旧模型和旧预算不变；这不是截断旧 1000-epoch run，
也不从旧 checkpoint 初始化。暂未加入显式训练 seed 控制，结果只能作为探索性证据。

参照修正在面内降采样 4 倍的特征统计上计算：当前 spacing 下 coarse spacing 约为
`[3,2,2] mm`，局部核为 `[3,5,5]`，采样足迹约 `9×10×10 mm`。各区域先分别汇聚
`Z·H / Z·H² / Z`，再计算区域条件中心特征、局部均值和方差，避免区域边界在 pooling 时混合。
统计使用 float32，方差加 `1e-4`，标准化对比截断到 `[-5,5]`。参照包含中心，可能被病灶/增生
污染，不能称为正常组织真值；邻域受滑窗边界和 padding 影响，未声称窗口不变性。

局部池化后的权重均值与支持量按算法加权统计解释，不代表实测区域体积。
仅当局部参照的 coarse 网格权重和 ≥4 时，该区域贡献才有效；该支持量不是实测体素数。
coarse 修正强度以该位置的有效分区隶属权重均值为上界，自适应条件再乘学习式 sigmoid；
coarse 统计位置无局部支持时修正归零，仍通过普通 MRI 特征分支分割。
上采样后再乘原分辨率分区隶属权重，避免修正扩散到
分区外。参照模型缺通道/非有限分区输入报错；空局部分区不改变病例阳性/阴性身份。
修正末层零初始化，在共享 stem/投影/backbone 权重时与普通融合逐值相同。
模型仅额外加入局部池化和小卷积，真实显存、速度与效果尚未测量，不能据参数量推断耗时。

**先运行匹配基线**（工作目录与环境如下；一次只运行一条，选空闲 GPU）：

```bash
cd /opt/data/private/lm/my-projects
source /root/anaconda3/etc/profile.d/conda.sh
conda activate lm
source scripts/env_nnunet.sh
nnUNet_compile=false CUDA_VISIBLE_DEVICES=0 python scripts/train/train_nnunet.py \
  feature_no_gate_positive_sampling_100ep 606 3d_fullres 0
```

其余三个条件分别把 variant 替换成上表对应值，其余参数相同。四条件统一使用原生环境选项
`nnUNet_compile=false`，减少短预算运行的首次编译等待；每步速度仍需实测并记录该设置。首次运行不加
`--continue-training`；新分支拒绝覆盖已有同名输出目录，也拒绝在缺 checkpoint 时静默重训。
输出由类名隔离到 `outputs/nnUNet_results/Dataset606_PICAI_Zonal/<Trainer>__nnUNetPlans__3d_fullres/fold_0/`。
这些目录由用户实际训练时创建；本次未创建模型目录或 checkpoint。

启动日志应显示对应 mode、`epochs=100`、`iterations_per_epoch=250`、五通道输入；
采样日志应显示 1277 训练例、223 验证例、362 阳性训练例、915 阴性训练例、batch=2、
每 batch 保证一个病灶前景 patch。观察终端 epoch/loss/耗时与该目录 `progress.png`。
成功判据：完成 epoch 99、原生 actual validation 完成全部 223 例、生成
`checkpoint_final.pth` 和 `validation/summary.json`。patch pseudo Dice 不作为整例结果。
耗时仅能在真实运行后核定；100-epoch 结果不与旧长预算结果直接归因比较。

完成基线和候选后，先使用现有评估工具的 summary 模式核对主终点；full 模式按既有协议补充
病灶实例与物理体积分层，由用户另行运行。新增模块的有效性和对先验错误的稳健性均待验证。

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
6. 浅层序列特异特征融合主线（Research Plan §4–6 的 `feature_*_positive_sampling`：三个参数不共享
   的浅层 3×3×3 stem + 1×1×1 投影回 3 通道，可选 feature gate；骨干仍由 plans 构建）；
7. 一个统一训练入口 `scripts/train/train_nnunet.py`；
8. 最少量的数据准备代码 `scripts/data/prepare_picai_nnunet.py`。

**不重复实现**第二套 trainer / checkpoint / 滑窗推理 / 验证器 / optimizer / 学习率调度器，
也不复制 nnU-Net 的 U-Net encoder/decoder。项目侧 dataloader 只允许**继承或包装**
`nnUNetDataLoader`、复用预处理 `class_locations`，并且训练采样与验证采样严格分离
（验证 loader 保持原生）。规则细则见 `AGENTS.md` §5。

### 旧十一个 variant（新增四个短预算条件见首节）

| variant | Trainer 类 | 数据集 | 输入通道 | 网络 / 损失 / 采样 |
|---|---|---|---|---|
| `baseline` | `nnUNetTrainerPICAI_FLCE_NoFFT` | Dataset605 | 3（T2W/ADC/HBV） | 原生 `PlainConvUNet` + PI-CAI Focal+CE + 原生采样 |
| `optimized_baseline` | `nnUNetTrainerPICAI_DiceCE_NoFFT` | Dataset605 | 3（T2W/ADC/HBV） | 原生 `PlainConvUNet` + 原生 Dice+CE + 原生采样 |
| `dicece_positive_sampling` | `nnUNetTrainerPICAI_DiceCE_PositiveSampling_NoFFT` | Dataset605 | 3（T2W/ADC/HBV） | 原生 `PlainConvUNet` + 原生 Dice+CE + **阳性病例采样**（独立强参考基线，不属于主链） |
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

十一个 Trainer 类名互不相同，nnU-Net 据此生成互不覆盖的 output folder。

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

#### `dicece_positive_sampling`（**代码就绪 / 尚未训练**；独立强参考基线，**不属于 A→B→C→D 主研究链**）

- Trainer 类名 `nnUNetTrainerPICAI_DiceCE_PositiveSampling_NoFFT`，CLI 别名 `dicece_positive_sampling`；
- 组成：**nnU-Net v2.6.2 原生 Dice+CE**（`DC_and_CE_loss` + `MemoryEfficientSoftDiceLoss` +
  原生 `DeepSupervisionWrapper`，项目零自写损失）**＋** 项目现有 PositiveSampling
  **＋** 项目现有 NoFFT augmentation 修复 **＋** 原生 `PlainConvUNet`（3 个 MRI 通道）；
- 用途：隔离**损失函数**的影响，回答「在 PositiveSampling 已稳定病灶暴露之后，原生 Dice+CE 是否
  优于 PI-CAI Focal+CE（`positive_sampling`）」。已中止的 `optimized_baseline` 使用原生采样且未完成
  训练，不能回答该问题；
- 单变量边界：相对 `positive_sampling` **只有损失不同**；相对 `optimized_baseline`
  **只有训练采样不同**。不含 gate、不含 PZ/TZ、不含浅层 feature path，不改 patch/batch size、
  optimizer、初始学习率、PolyLR、epoch 数、deep supervision 与验证采样；
- 目标输出目录（由类名自然形成，独立、不覆盖任何既有产物）：
  `outputs/nnUNet_results/Dataset605_PICAI/nnUNetTrainerPICAI_DiceCE_PositiveSampling_NoFFT__nnUNetPlans__3d_fullres/fold_0`；
- **当前状态：代码已就绪并经纯合成 CPU 测试验证，尚未训练**（没有 checkpoint、没有 validation、
  没有任何结果数字）；不写入任何预期结果；
- 未来训练命令（**本轮未执行**，由研究者本人运行；启动前必须逐条满足下列条件）：

```bash
cd /opt/data/private/lm/my-projects && conda activate lm && source scripts/env_nnunet.sh
CUDA_VISIBLE_DEVICES=0 python scripts/train/train_nnunet.py \
  dicece_positive_sampling \
  605 \
  3d_fullres \
  0 \
  --device cuda
```

  1. 当前 B（`feature_no_gate_positive_sampling`）已完全训练与 validation 结束；
  2. GPU 已空闲；
  3. 没有同名训练进程；
  4. 上述目标输出目录不存在；
  5. 已执行 `source scripts/env_nnunet.sh`；
  6. 启动日志必须显示：Dataset605 / fold 0 / `1277 train, 223 validation` / `batch_size=2` /
     `positive_cases=362` / `negative_cases=915` / `positive_cases_per_batch=1` /
     `guaranteed_positive_patch_fraction=0.5` / DiceCE loss 路径
     （`loss=DeepSupervisionWrapper (base=nnunetv2...DC_and_CE_loss)`）/ `network=PlainConvUNet` /
     `input_channels=3`；
  7. 任一项不符合时立即停止，不继续训练。

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

#### 输入级门控的初步实验（`image_gate_positive_sampling` 与 `anatomy_gate_positive_sampling` **均已完成训练 + validation**）

这些已完成实验原先回答输入级门控问题，现在为特征级主线提供 preliminary evidence。
旧 `baseline` ↔ 旧 `image_gate` 同为 FLCE 与原生采样；它们不能与阳性采样分支混合归因。
同一采样条件下的两个组合 Trainer 继续保留独立结果和输出目录：

| 比较 | 模型 A | 模型 B | 边界 |
|---|---|---|---|
| 输入级 MRI 门控（阳性采样） | `positive_sampling` | `image_gate_positive_sampling` | 初步实验；只增加输入级 image gate |
| 输入级解剖条件（阳性采样） | `image_gate_positive_sampling` | `anatomy_gate_positive_sampling` | 初步实验；门控条件含 PZ/TZ，数据集为 605 vs 606，前三 MRI 通道审计已通过 |
| 输入级 MRI 门控（原生采样） | 旧 `baseline` | 旧 `image_gate` | 历史对照；不能与阳性采样分支混合归因 |

- `image_gate_positive_sampling`：Dataset605，3 通道；FLCE + image gate + 阳性病例采样；
- `anatomy_gate_positive_sampling`：Dataset606，5 通道；FLCE + anatomy gate（PZ/TZ 只进门控、
  不进分割 backbone，进入前 clamp(0,1)）+ 相同采样；
- 两者的 loss、增强（NoFFT）、optimizer、PolyLR、epoch、deep supervision、checkpoint、
  validation、inference 与验证 loader（原生 `nnUNetDataLoader`）全部一致；anatomy 的强度增强
  仍只作用于前 3 个 MRI 通道；
- **输入级解剖条件对照的边界**：两者除门控条件外还使用不同数据集（Dataset605 vs Dataset606）。配置层面
  （split / spacing / patch size / batch size / 前三个 MRI 通道 normalization 与 fingerprint）
  一致；前三 MRI 通道的**真实数据逐数组一致性审计已通过**（`MRI_ARRAY_AUDIT_PASS`，2026-09-24
  03:33 UTC，早于训练启动，见下节）。审计只对前三个 MRI 通道判等，**PZ/TZ 本身不参与判等**，
  且两臂仍来自不同数据集配置，因此**仍不得**把差异说成「只差 PZ/TZ 的严格单变量对照」；
- **`image_gate_positive_sampling` 已完成训练 + actual validation**（2026-09-23 06:55 → 22:47 UTC；
  1000 epoch；223 例 validation 已落盘）：nnU-Net `foreground_mean.Dice` = 0.20935、阳性病例
  macro Dice 0.2991、micro Dice 0.5397、`positive_voxel_precision` 0.8341；**输入级 MRI 门控配对比较的结论、
  全部指标与复现命令见 `docs/Findings.md` §3.10**（配对均值 delta −0.0077、CI95 含 0 ⇒
  本次单次运行**未观察到明确的分割增量效用**，但工作点更保守：precision ↑、假阳 ↓、漏检略增）；
  运行事实见 `docs/Training_Log.md`，逐实验细节见 `docs/experiments/image_gate_positive_sampling.md`；
- **`anatomy_gate_positive_sampling` 已完成训练 + actual validation**（2026-09-24 06:42 → 22:52 UTC；
  1000 epoch；223 例 validation 已落盘）：nnU-Net `foreground_mean.Dice` = 0.20114、阳性病例
  macro Dice 0.2937、micro Dice 0.5332、`positive_voxel_precision` 0.7567；其启动前置条件
  ——Dataset605/606 前三 MRI 通道真实数据逐数组审计——已取得 `MRI_ARRAY_AUDIT_PASS`
  （2026-09-24 03:33 UTC，早于训练启动）。运行事实见 `docs/Training_Log.md`，逐实验细节见
  `docs/experiments/anatomy_gate_positive_sampling.md`；**输入级解剖条件配对比较已完成**，报告
  `outputs/reports/segmentation_metrics_rq2_anatomy_gate.json`，结论登记在
  `docs/Findings.md` **§3.11**：本次单次运行中**未观察到 anatomy gate 相对 image gate 的明确配对
  改善**（配对均值 delta −0.0053、CI95 含 0；召回略升但 precision 明显下降、漏分与阴性假阳增加），
  **旧输入级解剖条件假设本次未获支持**——但这不证明 PZ/TZ 无效，也不证明两法等效。

#### 原特征级融合主线（`feature_*_positive_sampling`，普通融合已启动、缺最终验证；其余未运行）

Research Plan §4–6 的论文主线。三个特征级条件共享
**完全相同**的浅层编码与投影结构、
FLCE 损失、阳性病例采样、增强、optimizer、LR scheduler、epoch、deep supervision、checkpoint、
validation 与滑窗推理，构成一组同层级比较：

| 比较 | 模型 A | 模型 B | 边界 |
|---|---|---|---|
| RQ1：浅层表征路径整体作用 | `positive_sampling` | `feature_no_gate_positive_sampling` | stem、投影、额外参数及骨干输入表示共同改变 |
| RQ2：特征级自适应门控 | `feature_no_gate_positive_sampling` | `feature_image_gate_positive_sampling` | 共享 stem/投影/backbone，主要差异为 feature gate |
| RQ3：特征级解剖条件 | `feature_image_gate_positive_sampling` | `feature_anatomy_gate_positive_sampling` | PZ/TZ 只进 gate；数据集 605 vs 606 与 16 个 gate 参数的差异见下 |

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

零初始化的边界（见 Research Plan §4.2；以下两点是代码层事实，各有单元测试守护）：

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
收益归因于解剖条件（Research Plan §5–6）。普通融合已启动但缺最终验证，另外两个门控条件未运行；
本节不含最终验证结果。阳性采样是它们共同的训练稳定化条件，并非主要方法贡献。

#### Dataset605 与 Dataset606 的一致性边界

配置层面已核对一致：冻结 split（1277 train / 223 val，**具体病例 ID 与顺序逐 fold 相同**）、
`3d_fullres` 的 spacing `[3.0, 0.5, 0.5]`、patch size `[16, 320, 320]`、batch size 2、前三个 MRI
通道的 normalization scheme 与 per-channel 强度 fingerprint 均一致；两个 `nnUNetPlans.json` 的
全部差异只有 `dataset_name`、新增的第 4/5 通道（PZ/TZ，`noNorm`）及其两条 per-channel 属性。
但**这不等同于完成真实数据逐数组审计**。该审计随后已实际执行并通过（v2，2026-09-24 03:33 UTC，
`status = MRI_ARRAY_AUDIT_PASS`，1500 例、0 mismatch；见下节报告与成功判据），
`anatomy_gate_positive_sampling` 的训练晚于该审计完成。即便审计通过，两个数据集的结果并列
解释时仍须注意：审计只对**前三个 MRI 通道**做逐值相等判定，PZ/TZ 通道本身不参与判等。

#### Dataset605 ↔ Dataset606 前三 MRI 通道逐数组审计（解剖条件比较前置；v1 判据有缺陷已修正，v2 已执行并 PASS）

工具：`scripts/data/audit_dataset605_606_mri_equivalence.py`（只读、fail-closed，含纯合成单元测试）。
它逐病例比较预处理后前三个 MRI 通道（T2W/ADC/HBV）的 `np.array_equal` 与浮点差值统计，并审计
病例集合、`splits_final.json` 与 segmentation label；**PZ/TZ 不参与逐值相等判定，也不影响
status/PASS**（仅记录取值范围与 `contains_non_finite`，均为信息性诊断）。
不修复数据、不重建 Dataset606、不重新 preprocessing。

seg 的两层语义（必须区分，不得混用）：

- **原始 seg 逐值相同**（`raw_seg_equal`）：预处理 `_seg.b2nd` 的字节/数值完全一致，含两侧都含
  `-1` 的情况。预处理 seg 的合法原始取值是 `{-1, 0, 1}`：`-1` 来自固定版本 nnU-Net 的
  `crop_to_nonzero` 裁剪填充（`cropping.py:19,36`），不是任务标签；
- **有效标签逐值相同**（`effective_labels_equal`）：`-1` 按本项目实际训练/验证变换
  （`nnUNetTrainer.py:800,855` 的 `RemoveLabelTansform(-1, 0)`）映射为 0 后的标签逐值一致。

因此纯 `-1↔0` 的原始差异是**信息性**（`n_cases_raw_label_difference` / `raw_label_differences`，
不影响 PASS）；`0↔1`、`-1↔1`、`{-1,0,1}` 之外的取值（含 NaN/Inf/无符号回绕值）、MRI 任一体素
差异、形状/dtype 差异一律 FAIL（`n_cases_effective_input_mismatch` / `mismatched_cases`）。

**已执行结果（2026-09-24 03:33 UTC，v2 报告）**：1500/1500 例全部检查，
`status = MRI_ARRAY_AUDIT_PASS`、`n_cases_effective_input_mismatch = 0`、
`mri_exact_equal_all = true`、`effective_labels_equal = true`，三个 MRI 通道
`exact_equal_cases = 1500` 且 `global_max|Δ| = 0`；仅 2 例纯 `-1↔0` 原始 seg 差异
（`10879_1000895` 1 体素、`10980_1000999` 11 体素）记为信息性。v1 报告（FAIL，起因是把合法的
`-1` 裁剪填充判为非法标签）按原样保留作追溯。

```bash
cd /opt/data/private/lm/my-projects && conda activate lm && source scripts/env_nnunet.sh

# 复现（已执行过一次；重跑请换路径或显式 --overwrite）
python scripts/data/audit_dataset605_606_mri_equivalence.py \
    --output outputs/reports/dataset605_606_mri_equivalence_audit_v2.json
```

成功判据：退出码 `0`、`status=MRI_ARRAY_AUDIT_PASS`、`n_cases_effective_input_mismatch=0`、
`effective_labels_equal=true`、`mri_exact_equal_all=true`、
`per_channel.*.exact_equal_cases = n_cases_checked`；`n_cases_raw_label_difference` 允许 > 0
（信息性）。PASS 之外退出码非零，报告仍会写出。报告已存在时默认拒绝覆盖，需显式加
`--overwrite`。`--max-cases N` 只用于烟雾测试，此时 status 恒为 `MRI_ARRAY_AUDIT_PARTIAL`，
**永不判 PASS**。v1 报告（`..._audit.json`，FAIL）按原样保留作追溯，不修改。

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

### 旧十一个 variant 的训练（短预算命令见首节）

```bash
python scripts/train/train_nnunet.py baseline 605 3d_fullres 0            # = 已完成的 N0
python scripts/train/train_nnunet.py optimized_baseline 605 3d_fullres 0  # 原生 Dice+CE（用户已中止，勿续训）
# dicece_positive_sampling = 独立强参考基线（原生 Dice+CE + 阳性病例采样）；**尚未训练**：
#   仅当 B 已训练+validation 完成、GPU 空闲、无同名进程、目标输出目录不存在时才运行（启动判据见上节）
# CUDA_VISIBLE_DEVICES=0 python scripts/train/train_nnunet.py dicece_positive_sampling 605 3d_fullres 0
python scripts/train/train_nnunet.py image_gate 605 3d_fullres 0
python scripts/train/train_nnunet.py anatomy_gate 606 3d_fullres 0        # 需先准备并预处理 Dataset606
CUDA_VISIBLE_DEVICES=0 python scripts/train/train_nnunet.py positive_sampling 605 3d_fullres 0
# 输入级门控历史条件（均与 positive_sampling 共用 FLCE + 阳性病例采样）
# image_gate_positive_sampling 已完成（2026-09-23，见 Training_Log / Findings §3.10）
CUDA_VISIBLE_DEVICES=0 python scripts/train/train_nnunet.py image_gate_positive_sampling 605 3d_fullres 0
# anatomy_gate_positive_sampling 已完成（2026-09-24；MRI 数组审计已通过，见「一致性边界」）
# CUDA_VISIBLE_DEVICES=0 python scripts/train/train_nnunet.py anatomy_gate_positive_sampling 606 3d_fullres 0
# 原特征级融合三条件（普通融合已有运行，勿重复从头启动；其余未运行）
CUDA_VISIBLE_DEVICES=0 python scripts/train/train_nnunet.py feature_no_gate_positive_sampling 605 3d_fullres 0
CUDA_VISIBLE_DEVICES=0 python scripts/train/train_nnunet.py feature_image_gate_positive_sampling 605 3d_fullres 0
CUDA_VISIBLE_DEVICES=0 python scripts/train/train_nnunet.py feature_anatomy_gate_positive_sampling 606 3d_fullres 0
```

> `positive_sampling`、两个 `*_positive_sampling` 组合、三个 `feature_*_positive_sampling` 与
> `dicece_positive_sampling` 都是**从零训练**：不加 `--continue-training`，不读取
> DiceCE/baseline/gate 的任何 checkpoint，也不会覆盖任何既有目录。训练开始后日志应出现实际训练病例数、阳性病例数与
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
> 三个原 `feature_*_positive_sampling` 条件目前没有最终 validation 可加入评测；普通融合已有中间训练产物。
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
- **`full` 模式新增病灶实例级指标（`lesion_instance_metrics`，协议预先冻结）**：实例为 3D
  连通域（6-邻域 / face connectivity）；候选匹配为至少 1 个重叠体素；采用一对一词典序匹配
  （先最大化匹配数，再最大化总 intersection 体素数，完全平局时按组件 id 升序），**不以 Dice
  为匹配目标**。报告 `lesion_sensitivity_any_overlap`（命名即口径：any-overlap，不是严格病灶
  检测指标）、`small_lesion_sensitivity_any_overlap`、`matched_lesion_dice`（只作用于 matched
  pairs，天然不含 missed lesions）与探索性大小分层 `< 500 / 500–1000 / > 1000 mm³`；每例另存
  可审计的 reference / prediction lesion records。**不做任何预测后处理**（最小团块过滤、最大
  团块、形态学、阈值优化）。阴性病例不进 sensitivity 分母，其预测团块仍计入假阳。
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

用 Prostate158 在**不重训、不微调、不新建 Dataset ID** 的前提下，对内部验证选定的最终模型
进行跨域压力测试。既有 Dataset605 命令只是可用入口示例，不能据此把 `positive_sampling`
预先确定为最终模型。模型、checkpoint、阈值和后处理须在 PI-CAI 内部验证上先行确定并冻结；
看到 Prostate158 结果后不得再依结果改动这些选择，否则该集合成为开发集。
通道映射固定 `_0000=t2 / _0001=adc / _0002=dwi`；**Prostate158 DWI 只是第三通道（训练时为
HBV）的跨域输入，不宣称与 HBV 等价**。原始 Prostate158 影像与标注绝不修改：默认只在独立目录
建相对软链接；只有三通道网格不一致且物理坐标关系可信、覆盖完整，并**显式**加
`--allow-resample` 时，才在派生目录内对 adc/dwi 做线性重采样（T2 为参考网格；绝不自动配准）。
若最终模型需要 PZ/TZ，须先有独立、冻结、可复现且不使用外测病灶标签的分区生成流程；
没有这样的流程时，不能宣称 anatomy 模型与 MRI-only 模型具有完全等价的外测条件。
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
- anatomy_gate_positive_sampling（**已完成**：训练 2026-09-24 06:42 → 22:41 UTC，validation 22:52 UTC 完成）：
  `outputs/nnUNet_results/Dataset606_PICAI_Zonal/nnUNetTrainerPICAI_AnatomyGate_PositiveSampling_NoFFT__nnUNetPlans__3d_fullres/fold_0/`
- feature_no_gate_positive_sampling / feature_image_gate_positive_sampling /
  feature_anatomy_gate_positive_sampling（首个已有运行且缺最终 validation；后两者未运行）：
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
  nnunet/trainers.py     PI-CAI 损失 + NoFFT 修复 + 十五个 Trainer + Trainer 注册表
  nnunet/sampling.py     阳性病例感知训练采样（继承原生 nnUNetDataLoader；验证 loader 保持原生）
  nnunet/networks.py     modality gate / 浅层序列特征融合 / 同区参照残差 与原生 backbone 包装器
  nnunet/transforms.py   PZ/TZ 增强边界（强度只作用 MRI）
  nnunet/__init__.py     check_fixed_nnunet_runtime（nnU-Net/DNA/PlainConvUNet 运行时校验）
pytest.ini               只收集 tests/（不扫描 data/workdir/outputs/third_party）
tests/                   纯合成单元测试
third_party/nnUNet/      固定 v2.6.2（只读）
data/ workdir/ outputs/  数据、预处理缓存、训练产物
```


## 阶段一候选：T2W 联合 WG/PZ/TZ

代码入口已接入，真实准备、planning/preprocessing、split、训练和评价尚未执行。
唯一 variant=`anatomy_joint_100ep`，Trainer=`nnUNetTrainerPICAI_AnatomyJoint_100ep_NoFFT`，
绑定 `Dataset607_PICAI_Anatomy / 3d_fullres / fold 0`。100 epochs 是首次工程试跑预算，
不是论文最终预算或充分收敛声明。原生网络、region BCE+Dice、deep supervision、采样、
optimizer、PolyLR、patch validation 和滑窗全部复用；NoFFT 仅关闭已有 blur FFT benchmark。

监督用既有物化 WG 与 `zonal_yuan`：code=WG+2*PZ+4*TZ，保存标注成员组合，
不构造新的解剖类别，不使用 lesion，不将分区并集替代 WG。labels 按 background/WG/PZ/TZ
插入，区域为 0 / [1,3,5,7] / [2,3,6,7] / [4,5,6,7]，class order=[1,2,4]。
Yuan 是工程复用选择，不能据此称其较 Hevi 准确。35 例追查支持物化 WG 内容对应与
头信息差异，保留这些病例；不证明历史来源，不修改原始标签或 source_resolved。

显式排除唯一已知缺失 `11050_1001070`，完整监督候选为 1499 **study**，当前冻结划分
train=1276、validation=223（不是患者数）。阶段二仍保留全部 1500；本次不接入部分监督、
折外批量预测、阶段二融合或 ROI 裁剪。不存在 split、额外缺失、非法值、患者跨划分均失败。

下面每一步由用户单独运行并检查成功后再执行下一步，代理不自动启动。每次打开终端先执行：

```bash
cd /opt/data/private/lm/my-projects
source /root/anaconda3/etc/profile.d/conda.sh
conda activate lm
source scripts/env_nnunet.sh
```

1. 准备检查（会读取真实影像，但不写 Dataset607），再物化：

```bash
python scripts/data/prepare_picai_nnunet.py anatomy --exclude-known-missing-wg --dry-run
```

成功判据：1499 study，1276/223，failed=0、excluded_count=1；tqdm 与结束摘要可观察。
检查通过后单独执行：

```bash
python scripts/data/prepare_picai_nnunet.py anatomy --exclude-known-missing-wg
```

输出 `workdir/nnUNet_raw/Dataset607_PICAI_Anatomy/`，仅 `imagesTr/*_0000.nii.gz`
和派生 `labelsTr/*.nii.gz`；成功后发布 `dataset.json`。任一病例失败不发布新 dataset.json。
禁止覆盖；确需继续未完成准备时使用相同命令加 `--resume`，核对配置、源文件 SHA256、
物理网格和派生标签内容，冲突失败。默认验证全部 source，不能无条件跳过既有文件。

2. 原生 planning/preprocessing（只对新 Dataset607）：

```bash
nnUNetv2_plan_and_preprocess -d 607 -c 3d_fullres --verify_dataset_integrity
```

输出 `workdir/nnUNet_preprocessed/Dataset607_PICAI_Anatomy/`，观察原生校验、fingerprint 与
preprocessing 进度，不加 `--verbose`（会关闭原生进度条）。成功判据：退出 0、
`nnUNetPlans.json` 含 3d_fullres，并完成 1499 例预处理；不更改 spacing/patch/batch。
此命令只用于首次准备的新目录，不用它覆盖已有预处理产物。

3. 显式冻结 split（必须在训练前完成）：

```bash
python scripts/data/prepare_picai_nnunet.py splits \
  --dataset-id 607 --dataset-name PICAI_Anatomy --exclude-known-missing-wg
```

输出 raw/preprocessed 下 `splits_final.json`。终端打印过滤及结束摘要；成功判据：
fold 0 train=1276、val=223，病例无重复，患者无交集，与完整 raw 病例集合一致。
冲突拒绝覆盖，缺失 split 时 Trainer 不会随机回退。

4. 首次工程训练并保存原生恢复后的验证概率：

```bash
python scripts/train/train_nnunet.py anatomy_joint_100ep 607 3d_fullres 0 \
  --export-validation-probabilities
```

输出 `outputs/nnUNet_results/Dataset607_PICAI_Anatomy/nnUNetTrainerPICAI_AnatomyJoint_100ep_NoFFT__nnUNetPlans__3d_fullres/fold_0/`。
观察原生 epoch 日志及验证进度。成功判据：100-epoch 工程预算结束、checkpoint 可用、
validation 全部 223 例产生 `.npz/.pkl/.nii.gz`，最终 anatomy-probability-check failed=0。
拒绝从头使用已存在运行目录。`--continue-training` 仅用于未完成训练：必须有可读且匹配
契约的 checkpoint、没有 `checkpoint_final.pth`，且 validation 目录尚无任何产物。
已有 `checkpoint_final.pth` 时拒绝续训，避免重新进入训练结束保存流程。
训练已完成但尚无验证产物时，可单独运行首次验证：

```bash
python scripts/train/train_nnunet.py anatomy_joint_100ep 607 3d_fullres 0 \
  --validation-only --export-validation-probabilities
```

validation 目录已有任何内容（包括部分结果、summary 或嵌套目录）时，拒绝从头训练、
续训和 validation-only，防止原生验证重写同名产物；空目录可以接受。
本入口不支持验证重跑或另一个验证输出目录，不删除、清空或移动既有结果。
不改变每 epoch iterations、patch size 或 batch size。

5. 独立区域评价，写到全新输出目录：

```bash
python scripts/evaluate_segmentation.py anatomy \
  --prediction-dir outputs/nnUNet_results/Dataset607_PICAI_Anatomy/nnUNetTrainerPICAI_AnatomyJoint_100ep_NoFFT__nnUNetPlans__3d_fullres/fold_0/validation \
  --reference-dir workdir/nnUNet_raw/Dataset607_PICAI_Anatomy/labelsTr \
  --images-dir workdir/nnUNet_raw/Dataset607_PICAI_Anatomy/imagesTr \
  --dataset-json workdir/nnUNet_raw/Dataset607_PICAI_Anatomy/dataset.json \
  --plans workdir/nnUNet_preprocessed/Dataset607_PICAI_Anatomy/nnUNetPlans.json \
  --split-file workdir/nnUNet_preprocessed/Dataset607_PICAI_Anatomy/splits_final.json \
  --output-dir outputs/reports/picai_anatomy_joint_100ep_fold0_v1 \
  --native-ordered-export
```

观察逐 study tqdm、失败诊断和导出进度；成功判据：223 例完整评估、failed=0，
新目录中有 `anatomy_regions.json` 与 `independent_threshold_bitcode/`，原生预测和
`summary.json` 不改变。已有评价目录拒绝覆盖，重评需明确选择另一个新目录。

主口径是三个概率分别 >0.5 的独立掩膜，对原始位编码参考调用原生区域 mask/count 机制。
不修正包含，不强制 PZ/TZ 互斥。双空 Dice=null，均值排除且报告有效数/双空数；单空 Dice=0。
同时报告重叠/分区超出 WG 的体素比例及分子分母。评价只表示算法伪标签一致性。
`native_ordered_export` 是可选独立口径：原生 TZ 覆盖 PZ、PZ 覆盖 WG，不能冒充独立头指标。

预测仍经 `scripts/inference/predict_nnunet.py -m <该模型目录> -i <单T2W目录> -o <全新目录>
--save_probabilities -f 0` 调用原生 predictor/export。解剖模型强制保存概率并核验产物；
预测保护通过参数解析统一处理 `-m 路径` / `-m=路径`，`-i/-o` 与受限参数同样处理；
缺值、空值、重复或歧义在调用原生入口前失败。合法旧病灶参数仍原样委托原生入口。
本次不批量生成折外预测。`.npz` 的 probabilities 固定 WG/PZ/TZ，形状 `(3,Z,Y,X)`，
已经复用原生 correct-shape 恢复到原始 T2W 数组网格；`.pkl` 的 sitk_stuff 保存 spacing/
origin/direction，plans 指定轴变换；`.nii.gz` 是覆盖式整数图，不能反推软通道。
预处理 crop 外三个概率补零，不能称这些位置接受过网络预测。检查通道、shape、有限值、
[0,1] 范围以及 pkl、T2W、分割图的几何关系，失败不缩减评价分母。
