# 实验记录：`anatomy_gate_positive_sampling`

> 本文档只记录 `anatomy_gate_positive_sampling` 实验本身：配置、运行事实、训练动态、验证结果、
> 门控参数与产物。跨 run 的数值对照、配对 bootstrap 与综合判断统一在 `docs/Findings.md`；
> 本文件不展开其他实验的数字，也不预先写入任何尚未发生的外测结果。
> 所有数字均取自本实验自己的产物（`training_log_2026_9_24_06_42_38.txt`、
> `validation/summary.json`、`debug.json`、`checkpoint_final.pth`、`checkpoint_best.pth`），
> 并由原始 summary 与日志逐项复算。

---

## 1. 实验标识

| 项 | 值 |
|---|---|
| variant | `anatomy_gate_positive_sampling` |
| Trainer 类 | `nnUNetTrainerPICAI_AnatomyGate_PositiveSampling_NoFFT` |
| 数据集 / 配置 / fold | `Dataset606_PICAI_Zonal` / `3d_fullres` / fold 0 |
| 输出目录 | `outputs/nnUNet_results/Dataset606_PICAI_Zonal/nnUNetTrainerPICAI_AnatomyGate_PositiveSampling_NoFFT__nnUNetPlans__3d_fullres/fold_0/` |
| 训练入口 | `CUDA_VISIBLE_DEVICES=0 python scripts/train/train_nnunet.py anatomy_gate_positive_sampling 606 3d_fullres 0` |
| 预测入口 | `scripts/inference/predict_nnunet.py -m <上述输出目录> -f 0 -device cuda` |
| 研究问题 | **RQ2 在阳性采样分支内**：固定阳性病例采样下，把 PZ/TZ 作为**门控条件**（PZ/TZ 只进
  gate、不进分割 backbone）是否改变分割表现。本实验是 RQ2 的 anatomy 条件臂：与 image 条件臂的
  **预期主要差异**是门控条件（多 PZ/TZ 两个通道）与数据集（Dataset606 vs Dataset605）；两者共享
  损失、采样、增强、optimizer、LR scheduler 与随机种子策略。本文件只记录本臂自身的事实 |
| 状态 | **训练 + actual validation 均已完成**（2026-09-24，1000 epoch 全部完成；223 例 validation
  产物已落盘）。本记录不含任何外测结果 |
| RQ2 归因前置条件 | Dataset605/606 前三 MRI 通道的真实数据逐数组审计已在本实验启动**之前**完成并通过：
  `outputs/reports/dataset605_606_mri_equivalence_audit_v2.json`，`status = MRI_ARRAY_AUDIT_PASS`
  （2026-09-24 03:33:24，1500 例，`mri_exact_equal_all = true`、`effective_labels_equal = true`、
  `n_cases_effective_input_mismatch = 0`；2 例纯 `-1↔0` 原始 seg 差异为信息性）。本实验训练于
  同日 06:42 UTC 启动 |

---

## 2. 一次性配置

### 2.1 数据

- `Dataset606_PICAI_Zonal`：PI-CAI 训练集 1500 例；固定划分 fold 0：**1277 train / 223 validation**
  （`splits_final.json`，1 fold；病例 ID 与顺序和 Dataset605 fold 0 逐项相同，由上述审计的
  `case_set_equal = true` / `split_equal = true` 确认）。
- 通道顺序（固定不可变）：`0000` = T2W、`0001` = ADC、`0002` = HBV（3 通道，`ZScoreNormalization`）、
  `0003` = PZ、`0004` = TZ（2 通道，`NoNormalization`，为小数占用率 `[0,1]`；
  `dataset.json` 记录 `prior_source=zonal_yuan`）。`use_mask_for_norm = [False] × 5`。
- 前三个 MRI 通道与 Dataset605 的一致性：**逐数组审计 PASS**（见第 1 节；审计只对前三个通道做
  逐值相等判定，PZ/TZ 不参与判等）。PZ/TZ 的 plans fingerprint 均值：PZ `0.46299`、TZ `0.41906`
  （范围 `[0,1]`）。
- patch `16 × 320 × 320`，target spacing `[3.0, 0.5, 0.5]` mm（该 spacing 只是 plans/预处理网格；
  validation 导出的逐例参考掩膜位于各例**原始几何**，体素体积逐例不同，不能按统一 mm³ 换算），
  batch size 2，`do_dummy_2d_data_aug = True`，前景采样 `oversample_foreground_percent = 0.33`。
- 验证集参考病灶体积（体素，与同 fold 其他记录共享同一标注）：min **190** / median **2459** /
  max **123055**；63 例有参考病灶、160 例为阴性。

### 2.2 网络（anatomy gate + 原生 backbone）

- backbone：由 `nnUNetPlans.json` 经原生 `get_network_from_plans` 构建的 `PlainConvUNet`
  （7 个 stage，features `[32, 64, 128, 256, 320, 320, 320]`），**只吃 gate 加权后的前 3 个 MRI 通道**。
- 输入门控 `SpatialModalityReliabilityGate`（**5 通道条件**），结构：
  `Conv3d(5 → 8, kernel 1×1×1)` → `InstanceNorm3d` → LeakyReLU → `Conv3d(8 → 3, kernel 1×1×1)`；
  末层权重/偏置**零初始化**（训练起点 `scales ≡ 1`，初始行为与无 gate 输入逐值一致）；
  `weights = softmax(logits, dim=1)`（三通道和为 1），`scales = 3 × weights`，`gated = mri × scales`；
  PZ/TZ 在进入 gate 前 `clamp(0,1)`，**不进入分割 backbone**（禁止 WG）。
- 参数量（从 `checkpoint_final.pth` 按唯一存储去重实测，134 组唯一存储）：
  gate **91** 个参数（conv1 `8×5+8=48`、norm `8+8=16`、conv2 `3×8+3=27`，占约 **0.000204%**）；
  backbone **44,577,932**；去重合计 **44,578,023**。`state_dict` 键为 `gate.*`（6 个张量）与
  `backbone.*`（344 个张量），无其他模块。
- deep supervision 输出 **6 个**，最低分辨率 DS 权重置 0（同本项目其他 FLCE run）。
- 运行时 `debug.json` 确认 `num_input_channels = 5`；validation 日志中逐例输入形状为
  `torch.Size([5, 23|25|27, 384, 384])` 等，即模型实际按 5 通道读取。

### 2.3 训练

与本项目 FLCE run 一致的优化器与调度（日志 / `debug.json` 实测）：

- optimizer：SGD，`momentum = 0.99`（Nesterov），weight decay `3e-5`
- 学习率：`initial_lr = 0.01`，PolyLR `power = 0.9`
- 训练长度：**1000 epoch × 250 iter = 250,000 iter**
- `save_every = 50`；`torch.compile` 启用（epoch 0 耗时 151.64 s 为编译期）
- 环境：NVIDIA GeForce RTX 3090（`CUDA_VISIBLE_DEVICES=0`），torch 2.10.0+cu128，cudnn 91002；
  训练 loader `num_processes = 12`
- **随机种子：nnU-Net v2.6.2 不设任何随机种子**（权重初始化不可精确复现）

### 2.4 损失与增强

- `0.5 × Focal(gamma=2, alpha=None, smooth=1e-5) + 0.5 × CE`（PI-CAI 风格 Focal+CE）；
  `checkpoint` 内实际损失对象为 `DeepSupervisionWrapper(PiCAIFocalCrossEntropyLoss)`
- Focal 复刻 PI-CAI v1 实现：softmax → one-hot `clamp(smooth/(C−1), 1−smooth)` →
  `−(1−pt)^gamma · log(pt)`，逐体素取均值
- deep supervision 外层为原生 `DeepSupervisionWrapper`，权重 `[1, 1/2, 1/4, 1/8, 1/16, 1/32]`
  归一化，**最低分辨率权重置 0（不监督）**
- 增强：**NoFFT 修复**生效（`GaussianBlurTransform` 保留，仅关闭 FFT benchmark，blur 概率/sigma 不变）；
  **强度增强只作用于前 3 个 MRI 通道**（PZ/TZ 豁免），空间/镜像变换继续同步作用于全部 5 个通道

### 2.5 采样机制（与 image 条件臂共同的固定条件）

启动时日志打印（2026-09-24 06:42:41 UTC，已与预处理 `class_locations` 核对）：

```
Positive-case sampling enabled:
training_cases=1277
positive_cases=362
negative_cases=915
positive_cases_per_batch=1
batch_size=2
guaranteed_positive_patch_fraction=0.5
```

- 训练 loader 为项目侧 `PositiveCaseDataLoader`（继承固定 v2.6.2 原生 `nnUNetDataLoader`，
  只覆盖 `get_indices`）：先调用父类原生均匀选例（保留阴性监督），再把 batch **最后 1 个槽位**
  替换为从 362 个阳性病例中有放回抽取的病例并强制病灶中心裁剪（`force_fg`，父类 `get_bbox`，
  不自行实现）。`debug.json` 中 `dataloader_train.generator` 即为该类的实例。
- 因此 batch size 2 时：**slot 0 从全部 1277 例均匀抽取（保留阴性病例），slot 1 固定来自阳性集合，
  50% patch 保证含病灶**。
- 阳性病例只从当前 fold **训练集**的预处理 `class_locations` 识别（含
  `label_manager.foreground_labels`），任何 metadata 缺失 / 不合法 / 没有阳性病例都 fail-closed。
- **验证 loader 保持原生** `nnUNetDataLoader`（`debug.json` 中 `dataloader_val.generator` 为其实例），
  不做阳性重排：训练采样与验证采样严格分离。

---

## 3. 运行记录

| 项 | 值 |
|---|---|
| 训练起止（UTC） | 2026-09-24 06:42:44（epoch 0）→ 2026-09-24 22:41:40（`Training done.`） |
| 纯训练耗时 | **15 h 58 min 56 s** |
| 验证起止（UTC） | 22:41:40（首例 predicting）→ 22:52:59（`Validation complete`），约 **11 min 18 s** |
| 全流程 | 06:42:44 → 22:52:59，约 **16 h 10 min** |
| epoch 耗时 | mean **55.83 s** / median 55.39 s / min 54.65 s / max **151.64 s**（n=1000；max 为首 epoch 的 `torch.compile` 编译期） |
| 日志异常 | **无 warning / error / NaN / Traceback**（关键词扫描全为 0）；一次跑完 |
| 随机种子 | **nnU-Net v2.6.2 无种子**，权重初始化不可精确复现 |
| 验证用权重 | `checkpoint_final.pth`（内部 `current_epoch = 1000`，即 epoch 999 完成后；由 nnU-Net 训练尾部 actual validation 自动使用，未使用 `--val_best`） |
| validation 产物 | 223 个预测 `.nii.gz` + `summary.json`（2026-09-24 22:41–22:52 UTC 落盘） |
| 末尾日志 | `Validation complete` / `Mean Validation Dice:  0.20113991018109215` |

---

## 4. 训练动态

### 4.1 关键 epoch

| epoch | train_loss | val_loss | pseudo dice |
|---|---|---|---|
| 0 | 0.0333 | 0.0033 | 0.0000 |
| 3 | 0.0059 | 0.0030 | **0.0002（首个非零）** |
| 5 | 0.0047 | 0.0024 | 0.0461 |
| 7 | 0.0051 | 0.0035 | 0.0647 |
| 10 | 0.0049 | 0.0038 | 0.0003 |
| 17 | 0.0035 | 0.0032 | 0.1243 |
| 19 | 0.0036 | 0.0017 | 0.2605 |
| 22 | 0.0038 | 0.0027 | 0.3027 |
| 25 | 0.0036 | 0.0016 | 0.1195 |
| 48 | 0.0031 | 0.0015 | 0.5076 |
| 50 | 0.0042 | 0.0021 | 0.3795 |
| 100 | 0.0027 | 0.0016 | 0.3960 |
| 200 | 0.0024 | 0.0017 | 0.2545 |
| 300 | 0.0022 | 0.0014 | 0.6173 |
| 400 | 0.0019 | 0.0016 | 0.3033 |
| 450 | 0.0032 | 0.0015 | **0.7691（单 epoch 全程最高）** |
| 500 | 0.0024 | 0.0013 | 0.4657 |
| 600 | 0.0018 | 0.0021 | 0.5768 |
| 700 | 0.0017 | 0.0010 | 0.4888 |
| 800 | 0.0016 | 0.0024 | 0.6789 |
| 900 | 0.0013 | 0.0023 | 0.6003 |
| 950 | 0.0013 | 0.0011 | 0.6422 |
| 952 | 0.0011 | 0.0029 | 0.6905（该轮刷新最佳 EMA） |
| 990 | 0.0014 | 0.0013 | 0.5509 |
| 999 | 0.0012 | 0.0014 | 0.6134 |

（train/val loss 全程范围分别为 0.0011–0.0333 与 0.0006–0.0046。）

### 4.2 分箱均值（pseudo dice，每 100 epoch）

| epoch 区间 | mean pseudo dice | median | max |
|---|---:|---:|---:|
| 0–99 | **0.2939** | 0.3130 | 0.6129 |
| 100–199 | 0.5144 | 0.5449 | 0.7478 |
| 200–299 | 0.4950 | 0.5168 | 0.7340 |
| 300–399 | 0.5069 | 0.5610 | 0.7601 |
| 400–499 | 0.4837 | 0.4877 | 0.7691 |
| 500–599 | 0.4967 | 0.4941 | 0.7337 |
| 600–699 | 0.5233 | 0.5410 | 0.7474 |
| 700–799 | 0.5381 | 0.5539 | 0.7461 |
| 800–899 | 0.5342 | 0.5587 | 0.7244 |
| 900–999 | **0.5605** | 0.5862 | 0.7341 |

形态：训练极早期（前 100 epoch）快速抬升，之后长期在约 0.48–0.56 区间高位震荡，
**没有明显的末期阶跃**；最后 100 epoch 略高于中段，但不构成"收敛到平台"的证据。

### 4.3 学习率（PolyLR，power 0.9，日志取值）

| epoch | 0 | 100 | 300 | 500 | 700 | 900 | 999 |
|---|---:|---:|---:|---:|---:|---:|---:|
| lr | 0.01 | 0.0091 | 0.00725 | 0.00536 | 0.00338 | 0.00126 | 2e-05 |

### 4.4 训练动态事实

- **首个非零 pseudo dice 出现在 epoch 3**（0.0002）；首次单 epoch ≥ 0.05 在 epoch 7（0.0647）、
  ≥ 0.10 在 epoch 17（0.1243）、≥ 0.20 在 epoch 19（0.2605）、≥ 0.30 在 epoch 22（0.3027）、
  ≥ 0.50 在 epoch 48（0.5076）、≥ 0.70 在 epoch 111（0.7088）。
- 首次出现**连续 20 epoch** 全部 > 0.05 / > 0.10 在 **epoch 35**；> 0.20 在 **epoch 80**；
  > 0.30 在 **epoch 88**；> 0.40 在 **epoch 681**；**> 0.50 从未出现连续 20 epoch**。
- 最佳 EMA pseudo Dice：**0.6236 @ epoch 952**（`Yayy! New best EMA pseudo Dice` 共 77 次；
  `checkpoint_best.pth` 内部 `current_epoch = 953`，即完成 epoch 952 后写入，与日志一致）。
- 单 epoch pseudo dice 最高 **0.7691 @ epoch 450**。
- 最后 100 epoch：mean **0.5605** / median 0.5862 / min 0.1886 / max 0.7341。
- **逐 epoch 波动很大**（最小 0.0002、最大 0.7691），单点值不可作为进展判据；判据应看分箱均值或 EMA。
- **loss 与 patch pseudo dice 无单调关系**：train/val loss 全程在 1e-3 量级的小区间内，
  低 loss 不保证高 pseudo Dice。本实验实际使用的模型选择依据只是 EMA patch pseudo Dice 这一
  **监控口径**；`checkpoint_best` 从未做全 volume 验证（见第 8、9 节）。

---

## 5. 验证结果

### 5.1 nnU-Net 汇报指标

```
foreground_mean.Dice = 0.20113991018109215
foreground_mean.IoU  = 0.1453746308185973
```

### 5.2 该数字的构成（反推自 `summary.json`）

| 类别 | 数量 | 对 Dice 的贡献 |
|---|---|---|
| 真阴（无参考、无预测） | **131** | 记 NaN，**被剔除** |
| 假阳阴性（无参考、有预测） | **29** | 记 **0** |
| 阳性（有参考） | **63** | 实际 Dice，完全漏分记 0 |
| **分母** | **92**（63 + 29） | 分子（63 例 Dice 之和）= **18.5048717** |

校验：`18.5048717 / 92 = 0.20113991…`，与汇报值逐位一致。
（该口径把 29 个假阳阴性病例的 0 计入分母、剔除 131 个真阴病例；与下文阳性病例口径不同。）

### 5.3 分层口径（本实验自身 summary 复算）

| 指标 | 值 |
|---|---:|
| `positive_macro_dice_mean`（63 例，完全漏分记 0） | **0.293728** |
| positive macro Dice median | **0.219963** |
| positive macro Dice Q1 / Q3 | 0.000000 / 0.562028 |
| positive macro Dice min / max | 0.0000 / 0.8314 |
| positive micro Dice（pooled） | **0.533166** |
| 阳性病例 pred 体素数 ÷ ref 体素数 | 0.5439 |
| `positive_voxel_recall` = ΣTP/(ΣTP+ΣFN)，仅阳性 | 0.411569 |
| `positive_voxel_precision` = ΣTP/(ΣTP+ΣFP)，仅阳性（论文核心口径） | **0.756744** |
| `all_prediction_voxel_precision`（分母含 29 例假阳体素） | 0.681217 |
| 阳性病例 TP / FP（阳性区内）/ FN（体素） | 238,575 / 76,690 / 341,097 |

### 5.4 检出与假阳结构

| 项 | 数量 | 比例 |
|---|---:|---:|
| 阳性中**有任意重叠预测**（Dice > 0） | **37 / 63** | 58.7% |
| 阳性中**完全无重叠**（Dice = 0） | **26 / 63** | **41.3%** |
| ↳ 其中完全无预测（`n_pred = 0`） | 22 / 63 | 34.9% |
| ↳ 有预测但不重叠 | 4 / 63 | 6.3% |
| 阳性 Dice > 0.3 | 31 / 63 | 49.2% |
| 阳性 Dice > 0.5 | 22 / 63 | 34.9% |
| 阳性 Dice > 0.7 | 6 / 63 | 9.5% |
| 有重叠 37 例的 Dice mean / median | 0.5001 / 0.5468 | — |
| 有重叠 37 例（仅在这 37 例内部）中位 预测体积 ÷ 标注体积 | 0.4956 | — |
| 有重叠 37 例中位 病灶体素召回（TP/n_ref） | 0.4487 | — |
| 有重叠 37 例中位 预测精度（TP/n_pred） | 0.8772 | — |
| 阴性中出现预测（假阳病例） | **29 / 160** | **18.1%** |
| ↳ 这些病例的预测体积 min / Q1 / median / mean / Q3 / max（体素） | 12 / 208 / **534** / 1205.3 / 906 / 6097 | — |
| 阴性病例假阳体素合计 | **34,954** | — |

### 5.5 按参考病灶体积分箱

| n_ref 区间（体素） | n | 检出 | 召回 | 精确 | pred/ref | pooled Dice |
|---|---:|---:|---:|---:|---:|---:|
| 0–1000 | 16 | 6 | 0.316 | 0.224 | 1.410 | 0.262 |
| 1000–5000 | 27 | 14 | 0.182 | 0.753 | 0.241 | 0.293 |
| 5000–20000 | 15 | 13 | 0.305 | 0.848 | 0.360 | 0.449 |
| >20000 | 5 | 4 | 0.500 | 0.767 | 0.652 | 0.605 |

> 最小档（0–1000 体素）的 `pred/ref = 1.410`、精确仅 0.224：该档的"检出"里混有明显**过预测**，
> 与中、大档的欠分割方向相反，解读时必须分开。四档检出合计 6+14+13+4 = **37**，与总有重叠数一致。

> **口径限制**：分箱只基于 `summary.json` 的体素数。各例导出几何不同、体素对应的物理 mm³
> 逐例不同，因此本表**不能**换算为 mL 或物理体积；物理体积分层需逐例读取 spacing 后计算
> （full 模式，见第 9 节待办）。

按病灶负荷四分位分组（排序后按秩 16 / 16 / 16 / 15）：

| 病灶负荷（体素） | n | 检出 | 平均 Dice |
|---|---:|---:|---:|
| 190–965 | 16 | 6 | 0.2076 |
| 1128–2459 | 16 | 9 | 0.2383 |
| 2624–7001 | 16 | 10 | 0.3090 |
| 7724–123055 | 15 | 12 | 0.4285 |

按病灶负荷上限累积（检出定义 `Dice > 0`）：

| 病灶负荷上限 | 例数 | 检出 |
|---:|---:|---:|
| ≤ 500 体素 | 6 | **0** |
| ≤ 1000 体素 | 16 | 6 |
| ≤ 2000 体素 | 26 | 12 |
| ≤ 5000 体素 | 43 | 20 |

> **口径限制**：以上是**病例级病灶负荷**分析（每例整例二值统计），不是病灶**实例**分析；
> `summary.json` 只提供整例统计，实例级连通域分析需要 full 评估（见第 9 节待办）。

---

## 6. 门控参数状态（checkpoint CPU 只读；实际尺度分布未测量）

零初始化意味着训练前 `scales ≡ 1`；训练后从两个 checkpoint 读取（`conv2` 是唯一零初始化层，
其偏离零的程度可度量门控离开恒等映射的距离）：

| 参数 | 零初始化时 | final（epoch 999，**本验证所用权重**） | best（epoch 952 / 存档 `current_epoch = 953`） |
|---|---:|---:|---:|
| `conv2.weight` 平均绝对值 | 0 | 0.127963 | 0.128810 |
| `conv2.weight` Frobenius 范数 | 0 | **0.741572** | 0.746725 |
| `conv2.bias`（T2W, ADC, HBV） | [0, 0, 0] | [**+0.0675, +0.1127, −0.1802**] | [+0.0665, +0.1143, −0.1808] |
| `conv2.weight` 逐输出通道行范数（T2W, ADC, HBV） | [0, 0, 0] | [0.3642, 0.3082, **0.5677**] | [0.3661, 0.3110, **0.5717**] |
| `conv1.weight`（5 输入通道）平均绝对值 | — | 0.069730 | 0.070540 |

事实陈述（解释边界见 `Research_Plan.md` §5「概念与解释边界」）：

1. 门控参数**已离开零初始化**（conv2 Frobenius 0 → final 0.741572），不是训练后仍 `scales ≡ 1`。
   这是本节能从参数直接读出的**唯一**结论。
2. 偏置的符号模式（final +0.068 / +0.113 / −0.180；best +0.067 / +0.114 / −0.181）只是**参数层面**
   的观察：每个位置实际的 `scales = softmax(logits)` 还取决于该位置的输入激活（含 PZ/TZ 两个
   条件通道）与三个 logit 的相对大小，**不能**从偏置次序推断实际门控偏好是 T2W > ADC > HBV。
3. HBV 输出通道的行范数在两个存档中都最大（final 0.5677、best 0.5717），同样只是参数范数；
   **不能**据此断言"空间调节集中在 HBV 通道"。
4. **当前状态**：参数已更新；**实际尺度分布及其空间变化尚未测量**。需要对验证病例做一次
   前向、逐位置记录三个 softmax 权重（均值 / 离散度 / 空间方差），并同时观察 PZ/TZ 两类条件
   通道对 logits 的贡献，之后才能描述任何实际偏好（见第 9 节待办）。
5. **解释边界（`Research_Plan.md` §5）**：即便测得，门控权重也只定义为模型内部的
   relative sequence preference / spatial scale，**不是**经过校准的可靠度、真实图像质量、
   因果贡献，也不等同于放射科医师判读权重；权重高低不能单独解释最终分割的任何变化。
6. PZ/TZ 只作为门控条件参与：本实验**未**测得 PZ/TZ 对 logits 的实际影响方向与幅度，
   因此本文件不对"解剖条件是否被使用"作任何结论。

---

## 7. 产物清单

| 文件 | 说明 |
|---|---|
| `checkpoint_best.pth` | EMA pseudo Dice 最高存档（**epoch 952 的 EMA 0.6236**；存档内部 `current_epoch = 953`，2026-09-24 21:56 UTC 写入） |
| `checkpoint_final.pth` | 2026-09-24 22:41 UTC 写入；内部 `current_epoch = 1000`；**actual validation（22:41–22:52 UTC）所用权重**；第 6 节 final 数字取自此文件 |
| `validation/summary.json` | 逐例指标（第 5 节全部数字来源） |
| `validation/*.nii.gz` | **223** 个验证预测（与 summary 一一对应；日志 223 条 `predicting`） |
| `training_log_2026_9_24_06_42_38.txt` | 完整训练 + validation 日志（7558 行） |
| `debug.json` | 运行时配置快照（5 通道、采样摘要、`dataloader_*` 类、环境） |
| `progress.png` | loss / pseudo dice / LR / epoch 耗时曲线 |

**未产出**：验证概率（`validation/*.npz` 数量为 0，未加 `--export-validation-probabilities`）。

---

## 8. 本实验的观察与限制

1. **阳性采样条件下训练早期上升很快**：首个非零 pseudo dice epoch 3、首次 ≥0.10 在 epoch 17、
   ≥0.20 在 epoch 19、≥0.30 在 epoch 22；前 100 epoch 分箱均值 0.2939。之后长期在约 0.48–0.56
   高位震荡，> 0.5 从未连续保持 20 epoch。
2. **最终全量结果的工作点偏保守**：阳性病例预测体积 / 参考体积 0.5439、仅阳性 precision 0.7567，
   但体素召回只有 0.4116、26/63 阳性病例完全无重叠（其中 22 例完全无预测）；
   有重叠病例的中位召回也只有 0.4487。高精度、低召回的形态明显。
3. **代价在阴性病例**：29/160 阴性病例出现假阳（中位 534 体素，合计 34,954 体素），最大单个
   假阳体积 6,097 体素；含阴性假阳的 overall precision 降到 0.6812。
4. **最小病灶档最弱且方向相反**：≤1000 体素档 16 例仅 6 例检出（≤500 体素 6 例**全部未检出**），
   召回 0.316、精确 0.224、`pred/ref = 1.410`（过预测）。
5. **门控参数已明显离开零初始化**（第 6 节）；但实际 softmax 尺度分布、空间变化与 PZ/TZ 对
   logits 的实际贡献**尚未测量**，不能从参数范数 / 偏置次序推断任何实际偏好。即使测得，也只是
   模型内部的相对尺度，没有可靠度 / 因果含义。
6. **`checkpoint_best` 从未做全 volume 验证**：best（epoch 952）与 final（epoch 999）哪个在完整
   223 例上更好，本记录无法判断；第 5 节全部数字只来自 final。
7. **patch pseudo dice 全程大幅震荡**（单 epoch 0.0002–0.7691）：仅凭这一 patch 级口径**不能断定**
   训练是否已收敛或进入稳定平台；1000 epoch 是否足够、更长训练是否改变结果，本实验不能说明。
8. **初始化不可复现**（无随机种子）；本 run 与其他 run 的逐病例配对比较、bootstrap CI 与综合判断
   按文档分工统一记录在 `docs/Findings.md`，不在本文件展开。RQ2 的归因还要求对照臂与其数据集
   一致性前提同时成立（本实验的数据一致性前提见第 1 节），并且**不能**仅凭单次运行作因果断言。
9. 本实验只回答阳性采样条件下"anatomy 条件（PZ/TZ 只进门控）"的问题；不涉及浅层特征融合或其他
   表征层级的问题。

---

## 9. 待办

- [ ] 用 `checkpoint_best.pth`（epoch 952 EMA 0.6236）做一次全 volume validation / 推理，
  得到 best 权重在 223 例上的指标后再与 final 比较（当前只有 patch pseudo Dice）。
  **前置条件**：必须先设计**独立输出路径**（如把 best 权重复制/链接到独立目录再验证，或把
  预测/验证写到独立输出目录），**不得**覆盖 final 权重已产出的 `validation/summary.json`
  与 223 个验证预测。
- [ ] full 模式分割表面指标：`python scripts/evaluate_segmentation.py --mode full
      --nsd-tolerance-mm <τ> --volume-thresholds-mm3 <T1,T2> --output <路径>`
      （长任务；容差与体积阈值需事先指定）。
- [ ] 物理体积分层与连通域失败分析（full 模式追加）。
- [ ] 概率阈值敏感性：重跑 validation 加 `--export-validation-probabilities`
      （`python scripts/train/train_nnunet.py anatomy_gate_positive_sampling 606 3d_fullres 0
      --validation-only --export-validation-probabilities`）。
- [ ] 假阳分析：29 个假阳阴性病例（中位 534 体素）的空间位置需要预测坐标，`summary.json`
      不含坐标。
- [ ] gate 的空间权重图（按 T2W/ADC/HBV 的 mean/std 与空间方差）：需一次前向；同时记录
      PZ/TZ 两个条件通道对 logits 的贡献（当前完全没有测量）。
- [ ] 需要统计效力时：固定随机种子的重复运行（当前无法区分"anatomy 条件的效应"与初始化 /
      run-to-run 噪声；跨 run 配对比较见 `docs/Findings.md`）。

---

## 10. 后续记录模板

```
### <YYYY-MM-DD> — <标签；如 best-weight validation / reader / full metrics>
- 命令：
- 权重：
- 起止（UTC）：
- 关键指标（macroDice / medDice / microDice / recall / posPrec / allPrec / missed / negFP）：
- 备注：
```
