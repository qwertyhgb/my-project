# 实验记录：`image_gate_positive_sampling`

> 本文档只记录 `image_gate_positive_sampling` 实验本身：配置、运行事实、训练动态、验证结果、
> gate 权重与产物。跨 run 的数值对照、配对 bootstrap 与综合判断统一在 `docs/Findings.md`，
> 本文件不展开其他实验的数字，也不预先写入任何尚未发生的外测结果。
> 所有数字均取自本实验自己的产物（`training_log_2026_9_23_06_55_30.txt`、
> `validation/summary.json`、`debug.json` 与两个 checkpoint），并由原始 summary 逐项复算。

---

## 1. 实验标识

| 项 | 值 |
|---|---|
| variant | `image_gate_positive_sampling` |
| Trainer 类 | `nnUNetTrainerPICAI_ImageGate_PositiveSampling_NoFFT` |
| 数据集 / 配置 / fold | `Dataset605_PICAI` / `3d_fullres` / fold 0 |
| 输出目录 | `outputs/nnUNet_results/Dataset605_PICAI/nnUNetTrainerPICAI_ImageGate_PositiveSampling_NoFFT__nnUNetPlans__3d_fullres/fold_0/` |
| 训练入口 | `python scripts/train/train_nnunet.py image_gate_positive_sampling 605 3d_fullres 0 --device cuda` |
| 预测入口 | `scripts/inference/predict_nnunet.py -m <上述输出目录> -f 0 -device cuda` |
| 研究问题 | **RQ1 在阳性采样分支内的公平匹配条件**：固定阳性病例采样下，输入级 image gate 相对无 gate
  参照臂是否改变序列偏好与分割表现。预期与参照臂唯一的结构差异是 3→8→3 的输入级空间 gate |
| 状态 | **训练 + actual validation 均已完成**（2026-09-23，1000 epoch 全部完成；223 例 validation
  产物已落盘）。本记录不含 Prostate158 等任何外测结果 |

---

## 2. 一次性配置

### 2.1 数据

- PI-CAI 训练集 1500 例；固定划分 fold 0：**1277 train / 223 validation**（患者级无重叠；
  `splits_final.json`，1 fold）。
- 通道顺序（固定不可变）：`0000` = T2W、`0001` = ADC、`0002` = HBV（3 通道）。
- 归一化：三通道均为 `ZScoreNormalization`；`use_mask_for_norm = [False, False, False]`。
- patch `16 × 320 × 320`，target spacing `[3.0, 0.5, 0.5]` mm（该 spacing 只是 plans/预处理网格；
  validation 导出的逐例参考掩膜位于各例**原始几何**，体素体积逐例不同，不能按统一 mm³ 换算），
  batch size 2，`do_dummy_2d_data_aug = True`，前景采样 `oversample_foreground_percent = 0.33`。
- 验证集参考病灶体积（体素，与同 fold 其他 run 共享同一标注）：min **190** / Q1 **1046.5** /
  median **2459** / Q3 **6922** / max **123055**；63 例有参考病灶、160 例为阴性。

### 2.2 网络（输入级 image gate + 原生 backbone）

- backbone：由 `nnUNetPlans.json` 经原生 `get_network_from_plans` 构建的 `PlainConvUNet`
  （7 个 stage，features `[32, 64, 128, 256, 320, 320, 320]`）。
- 输入门控 `SpatialModalityReliabilityGate`（**无解剖先验，纯影像输入**），结构：
  `Conv3d(3 → 8, kernel 1×1×1)` → `InstanceNorm3d` → LeakyReLU → `Conv3d(8 → 3, kernel 1×1×1)`；
  末层权重/偏置**零初始化**（训练起点 `scales ≡ 1`，初始行为与无 gate 输入逐值一致）；
  `weights = softmax(logits, dim=1)`（三通道和为 1），`scales = 3 × weights`，
  `gated = mri × scales`。
- 参数量（从 `checkpoint_final.pth` 按唯一存储去重实测，134 组唯一存储）：
  gate 仅 **75** 个参数（conv1 32+16 + conv2 27 个，占骨干约 0.00017%）；
  backbone **44,577,932**；去重合计 **44,578,007**。`state_dict` 键为
  `gate.*`（6 个张量）与 `backbone.*`（344 个张量）。
- 不含 gate 键之外的任何额外模块；验证 loader 保持原生，gate 只包在输入级。
- deep supervision 输出 **6 个**，最低分辨率 DS 权重置 0（同本项目其他 FLCE run）。

### 2.3 训练

与同项目 FLCE run 完全一致的优化器与调度（日志 / `debug.json` 实测）：

- optimizer：SGD，`momentum = 0.99`（Nesterov），weight decay `3e-5`
- 学习率：`initial_lr = 0.01`，PolyLR `power = 0.9`
- 训练长度：**1000 epoch × 250 iter = 250,000 iter**
- `save_every = 50`；`torch.compile` 启用
- 环境：NVIDIA GeForce RTX 3090（`CUDA_VISIBLE_DEVICES=0`），torch 2.10.0+cu128，cudnn 91002
- **随机种子：nnU-Net v2.6.2 不设任何随机种子**（权重初始化不可精确复现）

### 2.4 损失

与同项目 FLCE run 完全一致：

- `0.5 × Focal(gamma=2, alpha=None, smooth=1e-5) + 0.5 × CE`（PI-CAI 风格 Focal+CE）
- Focal 复刻 PI-CAI v1 实现：softmax → one-hot `clamp(smooth/(C−1), 1−smooth)` →
  `−(1−pt)^gamma · log(pt)`，逐体素取均值
- deep supervision 外层为原生 `DeepSupervisionWrapper`，权重 `[1, 1/2, 1/4, 1/8, 1/16, 1/32]`，
  归一化，**最低分辨率权重置 0（不监督）**
- 增强：**NoFFT 修复**生效（`GaussianBlurTransform` 保留，仅关闭 FFT benchmark，blur 概率/
  sigma 不变）
- 本实验**不含**解剖通道与任何 PZ/TZ 先验。

### 2.5 采样机制（本实验与无 gate 参照臂共同的固定条件）

启动时日志打印（2026-09-23 06:55 UTC，已与预处理 `class_locations` 核对）：

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
  只覆盖 `get_indices`）：先调用父类原生均匀选例（保留阴性监督），再把 batch **最后
  1 个槽位**替换为从 362 个阳性病例中有放回抽取的病例并强制病灶中心裁剪
  （`force_fg`，父类 `get_bbox`，不自行实现）。
- 因此 batch size 2 时：**slot 0 从全部 1277 例均匀抽取（保留阴性病例），slot 1 固定来自
  阳性集合，50% patch 保证含病灶**。
- 阳性病例只从当前 fold **训练集**的预处理 `class_locations` 识别（含 `label_manager.
  foreground_labels`），任何 metadata 缺失 / 不合法 / 没有阳性标签集合都 fail-closed。
- **验证 loader 保持原生** `nnUNetDataLoader`，不做阳性重排：训练采样与验证采样严格分离。

---

## 3. 运行记录

| 项 | 值 |
|---|---|
| 训练起止（UTC） | 2026-09-23 06:55:34（epoch 0）→ 2026-09-23 22:35:45（epoch 999） |
| 纯训练耗时 | **15 h 40 min 11 s**（epoch 0→999 时间戳差） |
| 验证起止（UTC） | 22:36:44（首例 predicting）→ 22:47:35（Validation complete），约 **10 min 51 s** |
| 全流程 | 06:55:34 → 22:47:35，约 **15 h 52 min** |
| epoch 耗时 | mean **54.85 s** / median 54.81 s / min 54.30 s / max **77.66 s**（n=1000；
  max 为首 epoch 的 `torch.compile` 编译期） |
| 日志异常 | **无 warning / error / NaN / Traceback**（关键词扫描全为 0）；一次跑完 |
| 随机种子 | **nnU-Net v2.6.2 无种子**，权重初始化不可精确复现 |
| 验证用权重 | `checkpoint_final.pth`（内部 `current_epoch = 1000`，即 epoch 999 完成后；
  由 nnU-Net 训练尾部 actual validation 自动使用） |
| validation 产物 | 223 个预测 `.nii.gz` + `summary.json`（2026-09-23 22:36–22:47 UTC 落盘） |
| 末尾日志 | `Validation complete` / `Mean Validation Dice:  0.20934825678288785` |

---

## 4. 训练动态

### 4.1 关键 epoch

| epoch | train_loss | val_loss | pseudo dice |
|---|---|---|---|
| 0 | 0.0319 | 0.0034 | 0.0000 |
| 5 | 0.0053 | 0.0020 | 0.0000 |
| 7 | — | — | **0.0428（首个非零）** |
| 10 | — | — | 0.0530 |
| 25 | 0.0035 | 0.0020 | 0.2718 |
| 50 | — | — | 0.2833 |
| 100 | 0.0024 | 0.0015 | 0.4911 |
| 200 | — | — | 0.6142 |
| 300 | 0.0026 | 0.0014 | 0.5543 |
| 400 | — | — | 0.3288 |
| 500 | 0.0018 | 0.0009 | 0.2650 |
| 600 | — | — | 0.5702 |
| 696 | — | — | **0.7568（单 epoch 全程最高）** |
| 700 | — | — | 0.4254 |
| 800 | — | — | 0.4033 |
| 900 | — | — | 0.6352 |
| 950 | — | — | 0.4319 |
| 999 | 0.0011 | 0.0017 | **0.6294** |

（标 "—" 的格子未从日志摘录；train/val loss 全程范围分别为 0.0009–0.0319 与 0.0006–0.0044。）

### 4.2 分箱均值（pseudo dice，每 100 epoch）

| epoch 区间 | mean pseudo dice |
|---|---:|
| 0–99 | **0.3437** |
| 100–199 | 0.4413 |
| 200–299 | 0.5025 |
| 300–399 | 0.4637 |
| 400–499 | 0.4813 |
| 500–599 | 0.5077 |
| 600–699 | 0.5353 |
| 700–799 | 0.5389 |
| 800–899 | 0.5142 |
| 900–999 | **0.5519** |

形态：训练极早期（前 100 epoch）快速抬升，之后在约 0.45–0.55 区间高位震荡，没有明显的末期
阶跃或单调收敛；epoch 400–500 出现过一段下探。

### 4.3 学习率（PolyLR，power 0.9，日志取值）

| epoch | 0 | 100 | 300 | 500 | 700 | 900 | 999 |
|---|---:|---:|---:|---:|---:|---:|---:|
| lr | 0.01 | 0.0091 | 0.00725 | 0.00536 | 0.00338 | 0.00126 | 2e-05 |

### 4.4 训练动态事实

- **首个非零 pseudo dice 出现在 epoch 7**（0.0428）；首次单 epoch > 0.1 在 epoch 8（0.1598）。
- 首次连续（≥20 epoch）pseudo dice **> 0.05 在 epoch 50**；> 0.1 / > 0.2 / > 0.3 的连续 20 epoch
  窗口都在 **epoch 86**；> 0.5 **从未出现**连续 20 epoch（高位仍有频繁回撤）。
- 最佳 EMA pseudo Dice：**0.6110 @ epoch 698**（`Yayy! New best EMA pseudo Dice` 共 52 次；
  `checkpoint_best.pth` 内部 `current_epoch = 699`，即完成 epoch 698 后写入）。
- 单 epoch pseudo dice 最高 **0.7568 @ epoch 696**。
- 最后 100 epoch：mean **0.5519** / median 0.5758 / min 0.1613 / max 0.7385。
- **逐 epoch 波动很大**（如 epoch 33 已到 0.6277、epoch 500 低至 0.2650），单点值不可作为
  进展判据；判据应看分箱均值或 EMA。
- **loss 与 patch pseudo dice 无单调关系**：train/val loss 全程在 0.001–0.004 的小区间内，
  低 loss 不保证高 pseudo Dice。但 patch 级日志**不能**断定"按 `val_loss` 选模型在完整病例
  层面会选到什么结果"（那需要对不同 checkpoint 做全量验证）。本实验实际使用的模型选择依据
  只是 EMA patch pseudo Dice 这一**监控口径**。

---

## 5. 验证结果

### 5.1 nnU-Net 汇报指标

```
foreground_mean.Dice = 0.20934825678288785
foreground_mean.IoU  = 0.14883645858538752
```

### 5.2 该数字的构成（反推自 `summary.json`）

| 类别 | 数量 | 对 Dice 的贡献 |
|---|---|---|
| 真阴（无参考、无预测） | **133** | 记 NaN，**被剔除** |
| 假阳阴性（无参考、有预测） | **27** | 记 **0** |
| 阳性（有参考） | **63** | 实际 Dice，完全漏分记 0 |
| **分母** | **90**（63 + 27） | 分子（63 例 Dice 之和）= **18.8413431** |

校验：`18.8413431 / 90 = 0.2093482…`，与汇报值逐位一致。
（该口径把 27 个假阳阴性病例的 0 计入分母、剔除 133 个真阴病例；与下文阳性病例口径不同。）

### 5.3 分层口径（本实验自身 summary 复算）

| 指标 | 值 |
|---|---:|
| `positive_macro_dice_mean`（63 例，完全漏分记 0） | **0.299069** |
| positive macro Dice median | **0.288758** |
| positive macro Dice Q1 / Q3 | 0.000000 / 0.555789 |
| positive micro Dice（pooled） | **0.539691** |
| 阳性病例 pred 体素数 ÷ ref 体素数 | 0.4783 |
| `positive_voxel_recall` = ΣTP/(ΣTP+ΣFN)，仅阳性 | 0.398901 |
| `positive_voxel_precision` = ΣTP/(ΣTP+ΣFP)，仅阳性（论文核心口径） | **0.834071** |
| `all_prediction_voxel_precision`（分母含 27 例假阳体素） | 0.746372 |
| 阳性病例 TP / FP（阳性区内）/ FN（体素） | 231,232 / 46,001 / 348,440 |

### 5.4 检出与假阳结构

| 项 | 数量 | 比例 |
|---|---:|---:|
| 阳性中**有任意重叠预测**（Dice > 0） | **40 / 63** | 63.5% |
| 阳性中**完全无重叠**（Dice = 0） | **23 / 63** | **36.5%** |
| ↳ 其中完全无预测（`n_pred = 0`） | 21 / 63 | 33.3% |
| ↳ 有预测但不重叠 | 2 / 63 | 3.2% |
| 阳性 Dice > 0.5 | 20 / 63 | 31.7% |
| 阳性 Dice > 0.7 | 7 / 63 | 11.1% |
| 阳性 Dice 最大值 | 0.8263 | — |
| 有重叠 40 例的 Dice mean / median | 0.4710 / 0.5031 | — |
| 有重叠 40 例（仅在这 40 例内部）中位 预测体积 ÷ 标注体积 | 0.4546 | — |
| 有重叠 40 例中位 病灶体素召回（TP/n_ref） | 0.3983 | — |
| 有重叠 40 例中位 预测精度（TP/n_pred） | 0.8835 | — |
| 阴性中出现预测（假阳病例） | **27 / 160** | **16.9%** |
| ↳ 这些病例的预测体积 min / median / mean / Q1 / Q3 / max（体素） | 1 / **253** / 1206 / 53 / 1658 / 6182 | — |
| 阴性病例假阳体素合计 | **32,575** | — |

### 5.5 按参考病灶体积分箱

| n_ref 区间（体素） | n | 检出 | 召回 | 精确 | pred/ref | pooled Dice |
|---|---:|---:|---:|---:|---:|---:|
| 0–1000 | 16 | 7 | 0.272 | 0.202 | 1.345 | 0.232 |
| 1000–5000 | 27 | 16 | 0.226 | 0.779 | 0.290 | 0.350 |
| 5000–20000 | 15 | 13 | 0.249 | 0.834 | 0.299 | 0.384 |
| >20000 | 5 | 4 | 0.495 | 0.879 | 0.564 | 0.634 |

> 最小档（0–1000 体素）的 `pred/ref = 1.345`、精确仅 0.202：该档的"检出"里混有明显**过预测**，
> 与中、大档的欠分割方向相反，解读时必须分开。四档检出合计 7+16+13+4 = **40**，与总有重叠数一致。

> **口径限制**：分箱只基于 `summary.json` 的体素数。各例导出几何不同、体素对应的物理 mm³
> 逐例不同，因此本表**不能**换算为 mL 或物理体积；物理体积分层需逐例读取 spacing 后计算
> （full 模式，见第 9 节待办）。

按病灶负荷四分位分组（排序后按秩 16 / 16 / 16 / 15，与同 fold 其他实验记录同一稳定秩切分）：

| 病灶负荷（体素） | n | 检出 | 平均 Dice |
|---|---:|---:|---:|
| 190–965 | 16 | 7 | 0.2144 |
| 1128–2459 | 16 | 10 | 0.2318 |
| 2624–6922 | 16 | 11 | 0.3384 |
| 7724–123055 | 15 | 12 | 0.4191 |

按病灶负荷上限累积（检出定义 `Dice > 0`）：

| 病灶负荷上限 | 例数 | 检出 |
|---:|---:|---:|
| ≤ 500 体素 | 6 | 2 |
| ≤ 1000 体素 | 16 | 7 |
| ≤ 2000 体素 | 26 | 14 |

> **口径限制**：以上是**病例级病灶负荷**分析（每例整例二值统计），不是病灶**实例**分析；
> `summary.json` 只提供整例统计，实例级连通域分析需要 full 评估（见第 9 节待办）。

---

## 6. 门控参数状态（checkpoint CPU 只读；实际尺度分布未测量）

零初始化意味着训练前 `scales ≡ 1`；训练后从 `checkpoint_final.pth` 读取（`conv2` 是唯一
零初始化层，其偏离零的程度可度量门控离开恒等映射的距离）：

| 参数 | 零初始化时 | final（epoch 999，**本验证所用权重**） |
|---|---:|---:|
| `conv2.weight` 平均绝对值 | 0 | 0.0786 |
| `conv2.weight` Frobenius 范数 | 0 | **0.4940** |
| `conv2.bias`（T2W, ADC, HBV） | [0, 0, 0] | [**+0.1770, +0.0658, −0.2427**] |
| `conv2.weight` 逐输出通道行范数（T2W, ADC, HBV） | [0, 0, 0] | [0.2371, 0.1896, **0.3897**] |

`checkpoint_best.pth`（完成 epoch 698 后写入，EMA pseudo Dice 0.6110）中的同组量：

| 参数 | best（epoch 699 存档） |
|---|---:|
| `conv2.weight` 平均绝对值 | 0.0929 |
| `conv2.weight` Frobenius 范数 | 0.5780 |
| `conv2.bias`（T2W, ADC, HBV） | [+0.2016, +0.1013, −0.3029] |
| 逐输出通道行范数（T2W, ADC, HBV） | [0.2681, 0.2350, **0.4550**] |

事实陈述（解释边界见 `Research_Plan.md` §5「概念与解释边界」）：

1. 门控参数**已离开零初始化**（conv2 Frobenius 0 → final 0.4940），不是训练后仍 `scales ≡ 1`。
   这是本节能从参数直接读出的**唯一**结论。
2. 偏置的符号模式（final +0.177 / +0.066 / −0.243；best +0.202 / +0.101 / −0.303）只是
   **参数层面**的观察：每个位置实际的 `scales = softmax(logits)` 还取决于该位置的输入激活与
   三个 logit 的相对大小，**不能**从偏置次序推断实际门控偏好是 T2W > ADC > HBV。
3. HBV 输出通道的行范数在两个存档中都最大（final 0.3897、best 0.4550），同样只是参数范数；
   **不能**据此断言"空间调节集中在 HBV 通道"。
4. **当前状态**：参数已更新；**实际尺度分布及其空间变化尚未测量**。需要对验证病例做一次
   前向、逐位置记录三个 softmax 权重（均值 / 离散度 / 空间方差）后才能描述任何实际偏好
   （见第 9 节待办）。
5. **解释边界（`Research_Plan.md` §5）**：即便测得，门控权重也只定义为模型内部的
   relative sequence preference / spatial scale，**不是**经过校准的可靠度、真实图像质量、
   因果贡献，也不等同于放射科医师判读权重；权重高低不能单独解释最终分割的任何变化。

---

## 7. 产物清单

| 文件 | 说明 |
|---|---|
| `checkpoint_best.pth` | EMA pseudo Dice 最高存档（**epoch 698 EMA 0.6110**；存档内部
  `current_epoch = 699`，2026-09-23 17:53 UTC 写入） |
| `checkpoint_final.pth` | 2026-09-23 22:36 UTC 写入；内部 `current_epoch = 1000`；
  **actual validation（22:36–22:47 UTC）所用权重**；gate 数字取自此文件 |
| `validation/summary.json` | 逐例指标（第 5 节全部数字来源） |
| `validation/*.nii.gz` | **223** 个验证预测（与 summary 一一对应） |
| `training_log_2026_9_23_06_55_30.txt` | 完整训练 + validation 日志（7533 行） |
| `debug.json` | 运行时配置快照（环境、plans、采样摘要） |
| `progress.png` | loss / pseudo dice / LR / epoch 耗时曲线 |

**未产出**：验证概率（`.npz` 数量为 0，未加 `--export-validation-probabilities`）。

---

## 8. 本实验的观察与限制

1. **阳性采样条件下训练早期上升很快**：首个非零 pseudo dice epoch 7、首次 > 0.1 在 epoch 8；
   前 100 epoch 分箱均值 0.3437。之后约 0.45–0.55 高位震荡，> 0.5 从未连续保持 20 epoch。
2. **最终全量结果的工作点偏保守**：阳性病例预测体积 / 参考体积 0.4783、仅阳性 precision
   0.834（高），但体素召回只有 0.399、23/63 阳性病例完全无重叠（其中 21 例完全无预测）；
   有重叠病例的中位召回也仅 0.398。高精度、低召回的形态明显。
3. **代价在阴性病例**：27/160 阴性病例出现假阳（中位 253 体素，合计 32,575 体素），最大单个
   假阳体积 6,182 体素；含阴性假阳的 overall precision 降到 0.746。
4. **最小病灶档仍最弱且方向相反**：≤1000 体素档 16 例仅 7 例检出，召回 0.272、精确 0.202、
   `pred/ref = 1.345`（过预测）；≤500 体素 6 例仅 2 例检出。
5. **gate 参数已明显离开零初始化**（见第 6 节）；但实际 softmax 尺度分布及其空间变化
   **尚未测量**，不能从参数范数 / 偏置次序推断任何实际序列偏好。即使测得，也只是模型内部的
   相对尺度，没有可靠度 / 因果含义。
6. **`checkpoint_best` 从未做全 volume 验证**：best（epoch 698）与 final（epoch 999）哪个在
   完整 223 例上更好，本记录无法判断；第 5 节全部数字只来自 final。
7. **patch pseudo dice 全程大幅震荡**（单 epoch 0.16–0.76）：仅凭这一 patch 级口径**不能断定**
   训练是否已收敛或进入稳定平台；1000 epoch 是否足够、更长训练是否改变结果，本实验不能说明。
8. **初始化不可复现**（无随机种子）；与任何其他 run 的差异都不能仅凭单次运行做因果断言。
   本 run 与无 gate 阳性采样参照臂的逐病例配对比较、bootstrap CI 与综合判断按文档分工
   统一记录在 `docs/Findings.md`，不在本文件展开。
9. 本实验只回答阳性采样条件下"输入级 image gate"的问题；**不含解剖先验**，不能回答 RQ2。

---

## 9. 待办

- [ ] 用 `checkpoint_best.pth`（epoch 698 EMA 0.6110）做一次全 volume validation / 推理，
  得到 best 权重在 223 例上的指标后再与 final 比较（当前只有 patch pseudo Dice）。
  **前置条件**：必须先设计**独立输出路径**（如把 best 权重复制/链接到独立目录再验证，或把
  预测/验证写到独立输出目录），**不得**覆盖 final 权重已产出的 `validation/summary.json`
  与 223 个验证预测。
- [ ] full 模式分割表面指标：`python scripts/evaluate_segmentation.py --mode full
      --nsd-tolerance-mm <τ> --volume-thresholds-mm3 <T1,T2> --output <路径>`
      （长任务；容差与体积阈值需事先指定）。
- [ ] 物理体积分层与连通域失败分析（full 模式追加）。
- [ ] 概率阈值敏感性：重跑 validation 加 `--export-validation-probabilities`
      （`python scripts/train/train_nnunet.py image_gate_positive_sampling 605 3d_fullres 0
      --validation-only --export-validation-probabilities`）。
- [ ] 假阳分析：27 个假阳阴性病例（中位 253 体素）的空间位置需要预测坐标，`summary.json`
      不含坐标。
- [ ] gate 的空间权重图（按 T2W/ADC/HBV 的 mean/std 与空间方差）：需一次前向。
- [ ] 需要统计效力时：固定随机种子的重复运行（当前无法区分"gate+采样的联合效果"与初始化/
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
