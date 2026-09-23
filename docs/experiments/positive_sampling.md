# 实验记录：`positive_sampling`

> 本文档只记录 `positive_sampling` 实验本身：配置、运行事实、训练动态、验证结果与产物。
> 所有数字均取自本实验自己的产物（`training_log_2026_9_22_07_16_28.txt`、`validation/summary.json`、
> `debug.json` 与两个 checkpoint）。跨 run 的数值对照、配对 bootstrap、病例集合与综合判断统一
> 记录在 `docs/Findings.md`，本文件不展开横向比较。

---

## 1. 实验标识

| 项 | 值 |
|---|---|
| variant | `positive_sampling` |
| Trainer 类 | `nnUNetTrainerPICAI_FLCE_PositiveSampling_NoFFT` |
| 数据集 / 配置 / fold | `Dataset605_PICAI` / `3d_fullres` / fold 0 |
| 输出目录 | `outputs/nnUNet_results/Dataset605_PICAI/nnUNetTrainerPICAI_FLCE_PositiveSampling_NoFFT__nnUNetPlans__3d_fullres/fold_0/` |
| 训练入口 | `scripts/train/train_nnunet.py positive_sampling 605 3d_fullres 0 --device cuda` |
| 预测入口 | `scripts/inference/predict_nnunet.py -m <上述输出目录> -f 0` |

### 1.1 这项实验在研究里的位置

它**不是** `Research_Plan.md` 中的一个独立条件，而是项目侧的受控扩展（`AGENTS.md` §5、
`src/zonal_reliability_fusion/nnunet/sampling.py` 的模块 docstring）。它的作用有两层：

1. **优化修复**：已完成的 N0（`baseline`）在 1000 epoch 中有约 700 epoch 停留在"全部预测为背景"，
   到 epoch 756 才首次持续突破。本实验只改变**训练集病例/patch 采样**，考察这一现象是否由
   "稀疏小病灶 + 均匀病例采样"造成。
2. **固定采样的参照臂**：`Research_Plan.md` 规定核心门控比较必须固定损失、采样、增强、optimizer、
   LR scheduler 与 split。`positive_sampling` 因此是阳性采样分支内 **RQ1**（是否需要影像自适应融合）
   与 **RQ2**（显式分区条件是否提供额外归纳偏置）的参照臂：`image_gate_positive_sampling`、
   `anatomy_gate_positive_sampling` 与它配对使用。

**本实验不包含任何 gate，也不含 PZ/TZ**，因此它本身不回答 RQ1/RQ2，只提供这两个问题所需的
**同采样条件下的参照点**。

---

## 2. 一次性配置

### 2.1 数据

- PI-CAI 训练集 1500 例；固定划分 fold 0 = **1277 train / 223 validation**（患者级无重叠；
  `splits_final.json` 含 1 个 split）
- 通道顺序（固定不可变）：`0000` = T2W，`0001` = ADC，`0002` = HBV（共 3 通道，`num_input_channels = 3`）
- 归一化：三通道均为 `ZScoreNormalization`；`use_mask_for_norm = [False, False, False]`
- patch `16 × 320 × 320`，batch size `2`，target spacing `3.0 × 0.5 × 0.5` mm（1 体素 = 0.75 mm³）
- `do_dummy_2d_data_aug = True`
- 验证集参考病灶体积（体素）：
  min **190** / p25 **965** / median **2459**（≈ 1.84 mL）/ p75 **7001** / max **123055**；
  63 个阳性病例中 **43 例（68 %）小于 3.75 mL**
- 223 例中 **63 例有参考病灶**，160 例为阴性
- 数据集配置：`batch_dice = False`

### 2.2 网络

与 `baseline` **完全一致**（本实验不改网络）：

- backbone：由 `nnUNetPlans.json` 经原生 `get_network_from_plans` 构建的 `PlainConvUNet`
  - `n_stages = 7`，`features_per_stage = [32, 64, 128, 256, 320, 320, 320]`
  - `kernel_sizes = [[1,3,3], [1,3,3], [3,3,3], [3,3,3], [3,3,3], [3,3,3], [3,3,3]]`
  - `strides = [[1,1,1], [1,2,2], [1,2,2], [2,2,2], [2,2,2], [1,2,2], [1,2,2]]`
  - `n_conv_per_stage = [2]*7`，`n_conv_per_stage_decoder = [2]*6`，deep supervision 输出 **6 个**
  - 参数量 **44,577,932**（从 `checkpoint_final.pth` 去重后实测；`state_dict` 344 个条目、
    128 组唯一存储）
- **不含 gate**：`checkpoint_final.pth` 的 `state_dict` 中**没有任何 `gate.*` 键**（已核实）
- 无包装器，无额外模块；输入通道恒为 3

### 2.3 训练

与 `baseline` **完全一致**（本实验不改优化器、调度、增强、epoch）：

- optimizer：SGD，`momentum = 0.99`，`nesterov = True`，`weight_decay = 3e-5`
- 学习率：`initial_lr = 0.01`，`PolyLRScheduler(power = 0.9)`
- 训练长度：**1000 epoch × 250 iter = 250,000 iter**；`num_val_iterations_per_epoch = 50`
- 前景采样：`oversample_foreground_percent = 0.33`（原生机制，**未改**）
- `save_every = 50`；`torch.compile` 启用
- `enable_deep_supervision = True`
- 环境：NVIDIA GeForce RTX 3090（`CUDA_VISIBLE_DEVICES=0`），torch 2.10.0+cu128，cudnn 91002

### 2.4 损失

与 `baseline` **完全一致**：

- `0.5 × Focal(gamma=2, alpha=None, smooth=1e-5) + 0.5 × CE`（PI-CAI 风格）
- Focal 复刻 PI-CAI v1 实现：softmax → one-hot `clamp(smooth/(C−1), 1−smooth)` → `pt + smooth`
  → `−(1−pt)^gamma · log(pt)`，逐体素取均值
- deep supervision 外层为原生 `DeepSupervisionWrapper`，权重 `[1, 1/2, 1/4, 1/8, 1/16, 1/32]`
  归一化，**最低分辨率权重置 0（不监督）**
- 增强：**NoFFT 修复**生效 —— `GaussianBlurTransform` 保留，仅关闭其 FFT benchmark
  （blur 的概率与 sigma 不变）。训练日志中没有任何 benchmark 相关行
- 本实验不含解剖通道，因此未启用任何 PZ/TZ 相关的增强通道限制

### 2.5 采样机制（本实验**唯一**的改动）

启动时由 Trainer 自行打印（训练日志第 13–19 行）：

```
Positive-case sampling enabled:
training_cases=1277
positive_cases=362
negative_cases=915
positive_cases_per_batch=1
batch_size=2
guaranteed_positive_patch_fraction=0.5
```

- 训练 loader 换为项目侧 `PositiveCaseDataLoader`（**继承**固定 v2.6.2 原生 `nnUNetDataLoader`），
  只覆盖两件事：
  - `get_indices`：先调用父类原生均匀选例（保留阴性监督），再把 batch **最后
    `positive_cases_per_batch = 1` 个槽位**替换为从**阳性的 362 个训练病例**中有放回均匀抽取的病例；
  - `_oversample_last_XX_percent`：这些槽位**始终** `force_fg`，病灶中心 bbox 由父类 `get_bbox`
    用预处理 `class_locations` 计算（本实验不自行实现 `get_bbox`）。
- 因此 batch size 2 时：**slot 0** 从全部 1277 例均匀抽取（slot 0 仍可能是阴性），
  **slot 1** 固定来自 362 个阳性训练病例并强制病灶中心裁剪 ⇒ **100 % 的 batch 至少含一个病灶中心
  patch**、50 % 的 patch 槽位保证含病灶。
- 阳性病例只从**当前 fold 训练集**的预处理 `class_locations` 识别（结合
  `label_manager.foreground_labels`）；`.pkl` 缺失 / 不可读 / `class_locations` 缺失或格式不合法
  / 一个阳性都没有，全部 **fail-closed**，不会把 metadata 错误静默当成阴性，也绝不把验证病例
  加入训练阳性集合。
- `probabilistic_oversampling = False`（确定性槽位决策，才能保证阳性槽位始终病灶中心裁剪；
  概率式决策会被构造函数直接拒绝）。
- **验证 loader 保持原生 `nnUNetDataLoader` 与原生采样行为**，不按验证标签做任何阳性重采样。
- patch size、initial patch size、rotation、dummy 2D、mirror axes、deep-supervision scales 全部
  沿用 nnU-Net 配置；augmenter 的类型/进程数/缓存/pin memory 与原生 `get_dataloaders`（v2.6.2）一致。

---

## 3. 运行记录

| 项 | 值 |
|---|---|
| 训练起止（UTC） | 2026-09-22 07:16:32 → 2026-09-22 22:31:41（**15 h 15 min**） |
| 验证起止（UTC） | 2026-09-22 22:31:41 → 22:42:16（**10 min 35 s**，223 例） |
| epoch 耗时 | mean **53.29 s** / median 53.31 s / min 52.46 s / max 57.88 s（epoch 1–999） |
| 首 epoch 耗时 | **65.12 s**（`torch.compile` 编译期；第 2 个 epoch 起回落至 ~53 s） |
| 日志异常 | **无 warning / error / NaN / 中断**，一次跑完 |
| 随机种子 | **nnU-Net v2.6.2 不设置任何随机种子**（全库无 `manual_seed`）→ 权重初始化不可复现 |
| 验证用权重 | `checkpoint_final.pth`（`current_epoch = 1000`，即日志 epoch 999） |

> 说明：`debug.json` 是**初始化时刻**的快照，其 `current_epoch = 0`、`_best_ema = None` 属正常，
> 不代表运行结果；真实状态在 checkpoint 里。

---

## 4. 训练动态

### 4.1 关键 epoch

| epoch | train_loss | val_loss | pseudo dice |
|---|---|---|---|
| 0 | 0.0323 | 0.0038 | 0.0000 |
| 5 | 0.0055 | 0.0021 | 0.0000 |
| 8 | — | — | **0.0004**（首个非零值） |
| 10 | 0.0051 | 0.0025 | 0.0009 |
| 25 | 0.0047 | 0.0025 | 0.1681 |
| 50 | 0.0026 | 0.0017 | 0.3018 |
| 100 | 0.0026 | 0.0027 | 0.2936 |
| 150 | 0.0030 | 0.0015 | 0.4424 |
| 200 | 0.0029 | 0.0021 | 0.3543 |
| 300 | 0.0028 | 0.0044 | 0.0137 |
| 400 | 0.0021 | 0.0010 | 0.5240 |
| 500 | 0.0026 | 0.0022 | 0.5860 |
| 600 | 0.0022 | 0.0025 | 0.3827 |
| 700 | 0.0020 | 0.0011 | 0.6128 |
| 750 | 0.0022 | 0.0017 | 0.6460 |
| 800 | 0.0022 | 0.0013 | 0.6532 |
| 850 | 0.0017 | 0.0019 | 0.5805 |
| 900 | 0.0015 | 0.0012 | 0.5658 |
| 950 | 0.0014 | 0.0016 | 0.5780 |
| 975 | 0.0014 | 0.0012 | 0.5995 |
| 999 | 0.0013 | 0.0020 | **0.7810**（全程最高） |

### 4.2 分箱均值

| epoch 区间 | pseudo dice mean | median | max |
|---|---|---|---|
| 0–99 | **0.2250** | 0.1775 | 0.6888 |
| 100–199 | 0.4380 | 0.4611 | 0.7308 |
| 200–299 | 0.4662 | 0.4760 | 0.7562 |
| 300–399 | 0.4660 | 0.4878 | 0.7587 |
| 400–499 | 0.4439 | 0.4995 | 0.7112 |
| 500–599 | 0.5176 | 0.5525 | 0.7503 |
| 600–699 | 0.5420 | 0.5635 | 0.7509 |
| 700–799 | 0.5588 | 0.5715 | 0.7419 |
| 800–899 | 0.5638 | 0.5686 | 0.7460 |
| 900–999 | **0.5687** | 0.5907 | **0.7810** |

### 4.3 学习率（PolyLR，power 0.9）

| epoch | lr |
|---|---|
| 0 | 0.010000 |
| 100 | 0.009100 |
| 300 | 0.007250 |
| 500 | 0.005360 |
| 700 | 0.003380 |
| 800 | 0.002350 |
| 900 | 0.001260 |
| 999 | 0.000020 |

### 4.4 训练动态事实

- **背景平台期消失**：首次非零 pseudo dice 出现在 **epoch 8**（0.0004）；首个持续（≥20 epoch）
  > 0.05 在 **epoch 54**，> 0.20 在 **epoch 193**，> 0.40 在 **epoch 716**。
- 训练曲线的形状与同 fold 的 N0 `baseline`（同损失、同网络，唯一差异是训练采样）**完全不同**：
  N0 是"前约 700 epoch 恒定近零 + 末段阶跃"（其首次持续 > 0.05 在 **epoch 756**），本实验是
  "从 epoch 8 起持续上升、每 100 epoch 都抬升"。两个 run 的分箱曲线并排列表统一放在
  `docs/Findings.md` §3.6，本文件不重复 N0 的逐档数字。
- 最佳 EMA pseudo Dice：**0.6157 @ epoch 792**（19:22:44 UTC）；EMA 提升（`Yayy`）次数 **72**。
- 单 epoch 最高 pseudo dice：**0.7810 @ epoch 999**，即训练在被 epoch 上限截断时仍处于上升趋势。
- 最后 100 epoch：mean **0.5687** / median 0.5907 / max 0.7810 / min 0.2153。
- **逐 epoch 波动很大**（如 epoch 300 = 0.0137 而 epoch 400 = 0.5240），单点值不可作为进展判据；
  判据应看分箱均值或 EMA。
- **loss 与指标解耦**：全程 train_loss 在 0.0013–0.0323、val_loss 在 0.0010–0.0044，与 pseudo dice
  **无单调关系**；按 `val_loss` 选模型会选到更差的解。

---

## 5. 验证结果

### 5.1 nnU-Net 汇报指标

```
foreground_mean.Dice = 0.207834     (223 例，np.nanmean)
foreground_mean.IoU  = 0.150146
```

### 5.2 该数字的构成（反推自 `summary.json`）

| 类别 | 数量 | 对 Dice 的贡献 |
|---|---|---|
| 真阴（无参考、无预测） | 130 | 记 NaN，**被剔除** |
| 假阳阴性（无参考、有预测） | **30** | 记 **0** |
| 阳性（有参考） | 63 | 记实际 Dice |
| **分母** | **93** | 分子（63 例 Dice 之和）= 19.3285850 |

校验：`19.3285850 / 93 = 0.2078342469`，与汇报值一致。
若剔除 30 个假阳阴性病例，同一分子除以 63 得 **0.306803**。

### 5.3 分层口径（项目统一评估器 `--mode summary` 口径）

| 指标 | 值 |
|---|---|
| `positive_macro_dice`（阳性 63 例逐例平均，完全漏分记 0） | **0.3068** |
| `positive_macro_dice` median / max / min | **0.2372** / 0.8147 / 0.0000 |
| `positive_micro_dice`（阳性病例 pooled） | **0.5449** |
| `positive_voxel_recall` = ΣTP / (ΣTP + ΣFN)，仅阳性病例 | 0.4271 |
| `positive_voxel_precision` = ΣTP / (ΣTP + ΣFP)，仅阳性病例（**论文核心口径**） | **0.7522** |
| `all_prediction_voxel_precision`（分母含阴性病例假阳体素） | 0.6796 |
| 阳性病例 预测体积 ÷ 参考体积 | **0.5678** |
| 阳性病例 TP / FP / FN（体素） | 247,592 / 81,547 / 332,080 |
| 完全漏分病例数（`positive_missed_cases`） | **21** |
| 阴性假阳病例数（`negative_fp_cases`） | **30** |

仅在**有重叠的 42 例**（`Dice > 0`）中逐例取中位数：

| 指标中位数（42 例） | 值 |
|---|---|
| Dice | 0.5265 |
| 预测体积 ÷ 标注体积 | 0.510 |
| 病灶体素召回（`TP / n_ref`） | 0.458 |
| 预测精度（`TP / n_pred`） | 0.879 |

即：预测出来的区域绝大多数确实落在病灶内（精确 0.879），但只覆盖约 **46 %** 的标注区域 ——
仍是高精度、低召回的结构。

### 5.4 检出与假阳结构

| 项 | 数量 | 比例 |
|---|---|---|
| 阳性中**完全无重叠**（`Dice = 0`） | **21 / 63** | **33.3 %** |
| ↳ 其中完全无预测（`n_pred = 0`） | 16 / 63 | 25.4 % |
| ↳ 其中有预测但不重叠 | 5 / 63 | 7.9 % |
| 阳性中有重叠（`Dice > 0`） | **42 / 63** | **66.7 %** |
| 阳性中 `Dice > 0.5` | 23 / 63 | 36.5 % |
| 阳性中 `Dice > 0.7` | 9 / 63 | 14.3 % |
| 有重叠 42 例的 Dice mean / median | 0.4602 / 0.5265 | — |
| 阴性中出现预测（假阳病例） | **30 / 160** | **18.8 %** |
| ↳ 这些病例的预测体积 min / median / max | 5 / **534** / 7469 体素 | — |
| 阴性病例假阳体素合计 | 35,178 | — |

### 5.5 按参考病灶体积分箱

| n_ref 区间（体素） | n | 检出 | 召回 | 精确 | pred/ref | pooled Dice |
|---|---|---|---|---|---|---|
| 0–1000（< 0.75 mL） | 16 | 7 | 0.336 | 0.159 | 2.108 | 0.216 |
| 1000–5000（0.75–3.75 mL） | 27 | 19 | 0.195 | 0.747 | 0.261 | 0.309 |
| 5000–20000（3.75–15 mL） | 15 | 12 | 0.328 | 0.840 | 0.390 | 0.472 |
| > 20000（> 15 mL） | 5 | 4 | 0.513 | 0.781 | 0.656 | 0.619 |

> 最小档（< 0.75 mL）的 `pred/ref = 2.108`、精确仅 0.159，说明该档的"检出"主要是**过预测**而非
> 精确分割；这与其它三档"欠分割"的方向相反，必须分开解读。

按四分位分组（排序后按秩 16 / 16 / 16 / 15 切分，边界与本文件 §2.1 及 `baseline` / `image_gate`
记录一致；该切分对边界上的重复体素数按稳定秩归组）：

| 病灶负荷（体素） | n | 检出 | 平均 Dice |
|---|---:|---:|---:|
| 190–965 | 16 | 7 | 0.2207 |
| 1128–2459 | 16 | 12 | 0.2625 |
| 2624–7001 | 16 | 11 | 0.2959 |
| 7724–123055 | 15 | 12 | 0.4576 |

按病灶负荷上限累积（检出定义为 `Dice > 0`）：

| 病灶负荷上限 | 例数 | 检出 |
|---|---:|---:|
| ≤ 500 体素 | 6 | **1/6** |
| ≤ 1000 体素 | 16 | 7/16 |
| ≤ 2000 体素 | 26 | 15/26 |

> **口径限制**：以上是**病例级病灶负荷**分析，不是病灶**实例**级分析。一例可能含多个病灶，
> `summary.json` 只提供整例二值统计；真正的小病灶结论需要连通域级别的 lesion-size sensitivity。

---

## 6. 与其它 run 的对照

按 `AGENTS.md` 的文档分工，本文件不写跨实验对比；三 run 数值对照、逐病例配对比较、病例集合与
证据边界统一见 `docs/Findings.md`。

---

## 7. 产物清单

| 文件 | 说明 |
|---|---|
| `checkpoint_best.pth` | 写入于 19:22:45 UTC；`current_epoch = 793`（日志 epoch **792**，最后一次 EMA 提升，EMA pseudo dice **0.6157**） |
| `checkpoint_final.pth` | 写入于 22:31:41 UTC；`current_epoch = 1000`（日志 epoch 999）；**本次验证与第 5 节全部数字所用的权重** |
| `progress.png` | loss / pseudo dice / epoch 耗时 / LR 曲线 |
| `validation/summary.json` | 逐病例指标（第 5 节全部数字来源） |
| `validation/*.nii.gz` | **223** 个验证预测 |
| `training_log_2026_9_22_07_16_28.txt` | 完整训练日志（7553 行） |
| `debug.json` | **初始化时刻**的运行时配置快照 |

**未产出**：验证概率（`.npz` 数量为 0）未导出，因此本实验暂**无法**做分割概率阈值敏感性分析。

---

## 8. 本实验的观察与限制

1. **背景平台期被消除**：本实验分箱均值从 0.2250（epoch 0–99）升到 0.5687（epoch 900–999）。
   同 fold 的 N0 在同损失同网络下前约 700 epoch 恒定近零（两 run 的分箱曲线对比见
   `docs/Findings.md` §3.6）。这支持 `sampling.py` 中的假设：平台期主要来自
   **稀疏小病灶 + 均匀病例采样**下前景梯度的稀释，而不是损失函数本身。
   > 旁证（**方向性，不是结论**）：被用户中止的 `optimized_baseline`（原生 Dice+CE、原生采样）
   > 停在同类全背景平台（epoch 377、最大 pseudo Dice 约 0.0008、**无 actual validation**）；
   > 跨 run 的塌缩证据汇总见 `docs/Findings.md` §5。
2. **单变量可控**：除训练 loader 外，损失、网络、增强、optimizer、PolyLR、epoch、checkpoint、
   validation、推理与**验证 loader**全部与 `baseline` 一致，且"最后一槽位永远来自阳性集合、
   每批至少一个含病灶 patch"有单元测试与真实 MRO 测试守护。
3. **收益与代价并存**：漏分减少 13 例是主要收益，但阴性假阳病例从 11 增到 30。任何后续引用
   都应同时给出这两个方向，不能只报 Dice。
4. **仍有三分之一阳性病例完全无重叠**（21/63），其中 16 例完全无预测。检出上限仍未打开。
5. **训练在被 epoch 上限截断时仍在上升**（epoch 999 = 0.7810 为全程最高）。本实验不能说明
   1000 epoch 是否已足够，也不能说明更长训练的效果。
6. **`checkpoint_best` 从未做全 volume 验证**：本记录报告值与所有逐病例分析均来自
   `checkpoint_final`（epoch 999）。
7. **初始化不可复现**（无随机种子）；单次运行的差异不足以独立支持机制归因，证据边界统一见
   `docs/Findings.md` §6。
8. **本实验不含任何 gate 或解剖先验**，因此它**不能**回答 RQ1 / RQ2；它只提供阳性采样分支内
   同采样条件下的参照点。`image_gate` / `anatomy_gate` 使用**原生采样**，与它**不能**混在同一张
   归因表里。

---

## 9. 待办

- [ ] 导出验证概率：
      `python scripts/train/train_nnunet.py positive_sampling 605 3d_fullres 0 --validation-only --export-validation-probabilities`
      （会重写 `validation/` 目录，建议先备份 `summary.json`）
- [ ] 用 `checkpoint_best.pth` 推理一次，得到 epoch 792 权重的全 volume 指标
- [ ] 完成分割表面指标：`python scripts/evaluate_segmentation.py --mode full --nsd-tolerance-mm <τ>`
- [ ] 物理体积分层与连通域失败分析：同上再加 `--volume-thresholds-mm3 <T1,T2>`
- [ ] 概率就绪后：**分割概率阈值敏感性分析**（Dice / 召回 / 精确 vs 阈值）。本实验只做病灶
      **分割**评估，**不涉及** AUROC / AP / FROC / PI-CAI challenge score
- [ ] 假阳分析：30 个假阳阴性病例的预测位于何处（需要预测体素坐标，`summary.json` 不含）
- [ ] 需要统计效力时：固定随机种子的重复运行（当前无法区分"采样改动" 与 "run-to-run 方差"）

---

## 10. 后续记录模板

```
### <run 标签> — <日期>
- 命令：
- 起止（UTC）：
- 状态：完成 / 中断于 epoch N / 失败
- 权重：checkpoint_final / checkpoint_best（epoch ?）
- 采样摘要（training_cases / positive_cases / positive_cases_per_batch）：
- 关键指标（macroDice / medDice / microDice / recall / posPrec / missed / negFP）：
- 假阳代价（negFP、阴性假阳体素）：
- 输出目录：
- 备注：
```
