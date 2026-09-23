# 实验记录：`image_gate`

> 本文档只记录 `image_gate` 实验本身：配置、运行事实、训练动态、验证结果、门控权重与产物。
> 所有数字均取自本实验自己的产物。

---

## 1. 实验标识

| 项 | 值 |
|---|---|
| variant | `image_gate` |
| Trainer 类 | `nnUNetTrainerPICAI_ImageGate` |
| 数据集 / 配置 / fold | `Dataset605_PICAI` / `3d_fullres` / fold 0 |
| 输出目录 | `outputs/nnUNet_results/Dataset605_PICAI/nnUNetTrainerPICAI_ImageGate__nnUNetPlans__3d_fullres/fold_0/` |
| 训练入口 | `scripts/train/train_nnunet.py image_gate 605 3d_fullres 0 --device cuda` |
| 预测入口 | `scripts/inference/predict_nnunet.py -m <上述输出目录> -f 0` |

研究对象（对应 `Research_Plan.md` **RQ2 / H2**）：不提供任何解剖先验，仅凭 T2W / ADC / HBV
三序列自身，一个轻量空间门控能否形成**非均匀、非静态**的相对序列偏好。

---

## 2. 一次性配置

### 2.1 数据

- PI-CAI 训练集 1500 例；固定划分 fold 0 = **1277 train / 223 validation**（患者级无重叠）
- 通道顺序（固定不可变）：`0000` = T2W，`0001` = ADC，`0002` = HBV（共 3 通道）
- 归一化：三通道均为 `ZScoreNormalization`；`use_mask_for_norm = [False, False, False]`
- patch `16 × 320 × 320`，batch size `2`，target spacing `3.0 × 0.5 × 0.5` mm
- `do_dummy_2d_data_aug = True`
- 验证集参考病灶体积（体素，1 体素 = 0.75 mm³）：
  min **190** / p25 **965** / median **2459**（≈1.84 mL）/ p75 **7001** / max **123055**
- 223 例中 **63 例有参考病灶**，160 例为阴性

### 2.2 网络

- backbone：由 `nnUNetPlans.json` 经原生 `get_network_from_plans` 构建的 `PlainConvUNet`
  - `n_stages = 7`，`features_per_stage = [32, 64, 128, 256, 320, 320, 320]`
  - `n_conv_per_stage = [2]*7`，`n_conv_per_stage_decoder = [2]*6`，deep supervision 输出 **6 个**
  - 参数量 **44,577,932**（fp32 178.3 MB）
- 门控 `SpatialModalityReliabilityGate`（**输入通道 3**，无解剖先验）：

  ```
  Conv3d(3 → 8, kernel 1×1×1) → InstanceNorm3d(8, affine) → LeakyReLU → Conv3d(8 → 3, kernel 1×1×1)
  ```

  - 末层 `conv2.weight` / `conv2.bias` **零初始化**
  - `weights = softmax(logits, dim=1)`（沿通道维归一化，三序列竞争，Σw = 1）
  - `scales = 3 × weights`（Σscales = 3，均匀时 `[1,1,1]`）
  - `gated_mri = mri × scales` → 送入 backbone
  - 参数量 **75**（conv1 32 + norm 16 + conv2 27），占 backbone 的 **0.000168 %**（≈1/594,372）
  - 零初始化时 `scales ≡ 1`，与无门控的输入路径**逐值一致**；且第 0 步
    `∂L/∂conv1 = ∂L/∂conv2 ⊗ conv2.weight = 0`，第一层卷积从第 1 步起才开始学习
- backbone 恒接收 **3 个通道**；门控与 backbone 由 `GatedNNUNet` 包装，`decoder` 属性代理给
  backbone，使 nnU-Net 原生 `set_deep_supervision_enabled` 直接可用

### 2.3 训练

- optimizer：SGD，`momentum = 0.99`，`nesterov = True`，`weight_decay = 3e-5`
- 学习率：`initial_lr = 0.01`，`PolyLRScheduler(power = 0.9)`
- 训练长度：**1000 epoch × 250 iter = 250,000 iter**
- 前景采样：`oversample_foreground_percent = 0.33`；batch_size=2 时
  `round(2 × (1 − 0.33)) = 1`，即每个 batch 中 **2 个 patch 有 1 个被强制居中于前景**
- `save_every = 50`；`torch.compile` 启用
- 环境：NVIDIA RTX 3090（`CUDA_VISIBLE_DEVICES=0`），torch 2.10.0+cu128，cudnn 91002

### 2.4 损失

- `PI-CAI 风格`：`0.5 × Focal(gamma=2, alpha=None, smooth=1e-5) + 0.5 × CE`
- Focal 复刻 PI-CAI v1 实现：softmax → one-hot `clamp(smooth/(C−1), 1−smooth)` → `pt + smooth`
  → `−(1−pt)^gamma · log(pt)`，逐体素取均值
- deep supervision 外层为原生 `DeepSupervisionWrapper`，权重 `[1, 1/2, 1/4, 1/8, 1/16, 1/32]`
  归一化，**最低分辨率权重置 0（不监督）**
- 增强：**NoFFT 修复**生效 —— `GaussianBlurTransform` 保留，仅关闭其 FFT benchmark
  （blur 的概率与 sigma 不变）
- 本实验**不含**解剖通道，因此未启用任何 PZ/TZ 相关的增强通道限制

---

## 3. 运行记录

| 项 | 值 |
|---|---|
| 训练起止（UTC） | 2026-09-21 08:23:50 → 2026-09-22 00:11:50（**15 h 48 min**） |
| 验证起止（UTC） | 2026-09-22 00:11:50 → 00:22（约 10 min，223 例） |
| epoch 耗时 | mean **55.17 s** / median 55.00 s / min 54.43 s / max 149.49 s（首 epoch 含 `torch.compile`） |
| 首 epoch 耗时说明 | 149.49 s 是 TorchInductor 为「门控 + backbone」新计算图重新编译所致（编译期 CPU 满载、GPU 空闲），**非异常**；第 2 个 epoch 起回落至 ~55 s |
| 日志异常 | 无 warning / error / NaN / 中断，一次跑完 |
| 随机种子 | **nnU-Net v2.6.2 不设置任何随机种子**（全库无 `manual_seed`）→ 权重初始化不可复现 |
| 验证用权重 | `checkpoint_final.pth`（epoch 999） |

---

## 4. 训练动态

### 4.1 关键 epoch

| epoch | train_loss | val_loss | pseudo dice |
|---|---|---|---|
| 0 | 0.0274 | 0.0032 | 0.0000 |
| 5 | 0.0031 | 0.0029 | 0.0000 |
| 10 | 0.0033 | 0.0038 | 0.0000 |
| 50 | 0.0014 | 0.0014 | 0.0293 |
| 100 | 0.0021 | 0.0023 | 0.0063 |
| 200 | 0.0019 | 0.0013 | 0.2114 |
| 300 | 0.0023 | 0.0026 | 0.1980 |
| 400 | 0.0020 | 0.0018 | 0.1362 |
| 500 | 0.0016 | 0.0027 | 0.5542 |
| 600 | 0.0015 | 0.0024 | 0.0341 |
| 700 | 0.0013 | 0.0014 | 0.5861 |
| 800 | 0.0013 | 0.0015 | 0.5426 |
| 900 | 0.0013 | 0.0023 | 0.6981 |
| 950 | 0.0011 | 0.0018 | 0.6478 |
| 999 | 0.0010 | 0.0007 | 0.4042 |

### 4.2 分箱均值

| epoch 区间 | pseudo dice 均值 |
|---|---|
| 0–99 | 0.0602 |
| 100–199 | 0.1742 |
| 200–299 | 0.2008 |
| 300–399 | 0.2754 |
| 400–499 | 0.2893 |
| 500–599 | 0.3954 |
| 600–699 | 0.3935 |
| 700–799 | 0.4601 |
| 800–899 | 0.4673 |
| 900–999 | 0.5268 |

### 4.3 训练动态事实

- pseudo dice **从训练早期即非零并持续上升**：epoch 0–99 均值 0.0602，至 epoch 400–499 达 0.2893，
  epoch 900–999 达 0.5268
- 首次连续 ≥20 个 epoch 的 pseudo dice > 0.05：**epoch 457**；> 0.20：**epoch 772**
- 最佳 EMA pseudo dice：**0.5963 @ epoch 969**
- 单 epoch 最高 pseudo dice：**0.7627 @ epoch 666**
- 最后 100 epoch pseudo dice：mean 0.5268 / median 0.5910 / max 0.7484
- EMA 提升（`Yayy`）次数：**62**
- **逐 epoch 波动很大**：如 epoch 500 = 0.5542 而 epoch 600 = 0.0341，单点值不可作为进展判据
- **loss 与指标解耦**：整个训练 train/val loss 都在 0.001–0.0038，与 pseudo dice 无单调关系；
  按 `val_loss` 选模型会选到更差的解

---

## 5. 验证结果

### 5.1 nnU-Net 汇报指标

```
foreground_mean.Dice = 0.17025     (223 例，np.nanmean)
```

### 5.2 该数字的构成（反推自 `summary.json`）

| 类别 | 数量 | 对 Dice 的贡献 |
|---|---|---|
| 真阴（无参考、无预测） | 143 | 记 NaN，**被剔除** |
| 假阳阴性（无参考、有预测） | 17 | 记 **0** |
| 阳性（有参考） | 63 | 记实际 Dice |
| **分母** | **80** | 分子（63 例 Dice 之和）= 13.6200917 |

校验：`13.6200917 / 80 = 0.1702511461`，与汇报值一致。

### 5.3 分层口径

| 口径 | 值 |
|---|---|
| 阳性 63 例 per-case mean Dice | **0.2162** |
| 阳性 63 例 per-case median / max | 0.0000 / 0.8575 |
| 阳性病例 pooled（micro）Dice | **0.5244** |
| `positive_voxel_recall` = `ΣTP / (ΣTP + ΣFN)`（仅阳性病例） | 0.3927065651 |
| `positive_voxel_precision` = `ΣTP / (ΣTP + ΣFP)`（仅阳性病例，**论文核心口径**） | **0.7889982982** |
| `all_prediction_voxel_precision` = `ΣTP / Σn_pred`（分母**含阴性病例假阳体素**） | 0.7269530951 |
| 阳性病例 预测体积 ÷ 参考体积 | **0.4977** |
| 阳性病例 TP / FP / FN（体素） | 227,641 / 60,878 / 352,031 |

仅在**有重叠的 30 例**（`Dice > 0`）中逐例取中位数：

| 指标中位数（30 例） | 值 |
|---|---|
| 预测体积 ÷ 标注体积 | 0.455 |
| 病灶体素召回（`TP / n_ref`） | 0.376 |
| 预测精度（`TP / n_pred`） | 0.839 |

即：预测出来的区域多数确实落在病灶内，但只覆盖约 **38 %** 的标注区域 —— 高精度、低召回。

### 5.4 检出与假阳结构

| 项 | 数量 | 比例 |
|---|---|---|
| 阳性中**完全无重叠**（`Dice = 0`） | **33 / 63** | **52.4 %** |
| ↳ 其中完全无预测（`n_pred = 0`） | 30 / 63 | 47.6 % |
| ↳ 其中有预测但不重叠 | 3 / 63 | 4.8 % |
| 阳性中有重叠（`Dice > 0`） | 30 / 63 | 47.6 % |
| 阳性中 `Dice > 0.5` | 16 / 63 | 25.4 % |
| 阳性中 `Dice > 0.7` | 5 / 63 | 7.9 % |
| 有重叠 30 例的 Dice mean / median | 0.4540 / 0.5060 | — |
| 有预测 33 例的 Dice mean / median | 0.4127 / 0.4626 | — |
| 阴性中出现预测（假阳病例） | **17 / 160** | **10.6 %** |
| ↳ 这些病例的预测体积中位数 | **274 体素** | — |
| 阴性病例假阳体素合计 | 24,625 | — |

### 5.5 按参考病灶体积分箱

| n_ref 区间（体素） | n | 检出 | 召回 | 精确 | pred/ref | pooled Dice |
|---|---|---|---|---|---|---|
| 0–1000（<0.75 mL） | 16 | 4 | 0.167 | 0.147 | 1.130 | 0.156 |
| 1000–5000（0.75–3.75 mL） | 27 | 13 | 0.137 | 0.793 | 0.173 | 0.234 |
| 5000–20000（3.75–15 mL） | 15 | 9 | 0.306 | 0.704 | 0.434 | 0.426 |
| >20000（>15 mL） | 5 | 4 | 0.481 | 0.849 | 0.566 | 0.614 |

按病灶负荷上限累积（检出定义为 `Dice > 0`）：

| 病灶负荷上限 | 例数 | 检出 |
|---|---:|---:|
| ≤ 500 体素 | 6 | **0/6** |
| ≤ 1000 体素 | 16 | 4/16 |
| ≤ 2000 体素 | 26 | 8/26 |

按四分位分组（边界取自本实验 63 个阳性病例的标注体素数）：

| 病灶负荷（体素） | n | 检出 | 平均 Dice |
|---|---:|---:|---:|
| 190–965 | 16 | 4 | 0.1017 |
| 1128–2459 | 16 | 7 | 0.1445 |
| 2624–7001 | 16 | 9 | 0.2161 |
| 7724–123055 | 15 | 10 | 0.4149 |

> **口径限制**：以上是**病例级病灶负荷**分析，不是病灶**实例**级分析。一例可能含多个病灶，
> `summary.json` 只提供整例二值统计；真正的小病灶结论需要连通域级别的 lesion-size sensitivity。

---

## 6. 门控权重的实际状态

零初始化意味着训练前 `scales ≡ 1`。训练结束后从 `checkpoint_final.pth` 读取（CPU 只读）：

| 参数 | 零初始化时 | 训练后 |
|---|---|---|
| `conv2.weight` 平均绝对值 | 0 | 0.0979 |
| `conv2.weight` Frobenius 范数 | 0 | **0.5527** |
| `conv2.bias`（T2W, ADC, HBV） | [0, 0, 0] | **[+0.1148, +0.1501, −0.2649]** |
| `conv2.weight` 逐输出通道行范数（T2W, ADC, HBV） | [0, 0, 0] | [0.1948, 0.2714, **0.4403**] |

（`conv2` 是唯一零初始化的层，其偏离零的程度可直接度量门控离开恒等映射的距离。
`conv1` 使用常规初始化，未记录其初始值，故不列入对比。）

事实陈述：

1. 门控**已明显偏离恒等映射**（Frobenius 范数 0 → 0.5527），不是"未学习"状态。
2. 偏置项呈现出系统性的序列偏好次序：**T2W ≈ ADC > HBV**，即门控在学习中**主动压制了 HBV**。
3. HBV 对应输出通道的行范数最大（0.4403，约为 T2W 的 2.3 倍），说明**门控主要在对 HBV 做
   逐体素的空间调节**，对 T2W 的调节最弱。
4. **解释边界（`Research_Plan.md` §5.2 / §11.5）**：以上是模型内部学习到的相对缩放系数，
   不是经过校准的可靠度、不是临床重要性、不是因果贡献、不等同于放射科医师判读权重。
   权重高/低只能表述为「模型在该位置对该序列的相对缩放」。

---

## 7. 产物清单

| 文件 | 说明 |
|---|---|
| `checkpoint_best.pth` | **epoch 969**（最后一次 EMA 提升，EMA pseudo dice 0.5963） |
| `checkpoint_final.pth` | **epoch 999**；本次验证与第 6 节权重分析使用的权重 |
| `progress.png` | loss / pseudo dice / epoch 耗时 / LR 曲线 |
| `validation/summary.json` | 逐病例指标（本文档 5.x 节全部数字来源） |
| `validation/*.nii.gz` | 223 例验证预测 |
| `training_log_2026_9_21_08_23_50.txt` | 完整训练日志（7536 行） |
| `debug.json` | 运行时配置快照 |

**未产出**：验证概率（`.npz`）未导出，因此本实验暂无法做**分割概率阈值敏感性分析**。

---

## 8. 本实验的观察与限制

1. **门控是可训练的且确实学到了东西**：权重远离零初始化，并形成了 T2W ≈ ADC > HBV 的
   系统性偏好次序 —— RQ2 的初步正面证据。
2. **但门控只是输入级缩放，表达力有限**：`scales` 三个通道之和恒为 3（零和竞争），
   无法表达"三个序列同时都不可靠"或"整体需要增强"；也无法做跨序列语义对齐。
3. **汇报的 0.17025 由三层因素构成**：17 个假阳阴性病例各计一个 0（分母 63 → 80）；
   47.6% 的阳性病例零预测；模型系统性欠分割（预测体积为参考的 49.8 %，精确 0.789 但召回 0.393）。
4. **误差结构随病灶体积单调变化**：召回从 0.167 升到 0.481，精确稳定在 0.70–0.85。
5. **`checkpoint_best` 从未做全 volume 验证**：本实验报告值与所有逐病例分析均来自
   `checkpoint_final`（epoch 999）。
6. **初始化不可复现**：nnU-Net v2.6.2 无随机种子，本实验的单次结果不可精确复现；
   单次运行的差异不足以支持任何跨实验的机制归因。
7. **首 epoch 的 149 s 不是异常**：是新计算图的 `torch.compile` 编译，后续 epoch 正常。

---

## 9. 待办

- [ ] 导出验证概率：
      `python scripts/train/train_nnunet.py image_gate 605 3d_fullres 0 --validation-only --export-validation-probabilities`
      （会重写 `validation/` 目录，建议先备份 `summary.json`）
- [ ] 用 `checkpoint_best.pth` 推理一次，得到 epoch 969 权重的全 volume 指标
- [ ] 完成分割表面指标：`python scripts/evaluate_segmentation.py --mode full --nsd-tolerance-mm <τ>`
- [ ] 物理体积分层与连通域失败分析：同上再加 `--volume-thresholds-mm3 <T1,T2>`
- [ ] 概率就绪后：**分割概率阈值敏感性分析**（Dice / 召回 / 精确 vs 阈值）。
      本实验只做病灶**分割**评估，**不涉及** AUROC / AP / FROC / PI-CAI challenge score
- [ ] 采集门控的空间权重图（需要一次前向），按 `Research_Plan.md` §11 的三个可解释对象分析：
      权重均匀度 `H`、解剖响应 `R_Z`、空间变化 `V_m`
- [ ] 需要统计效力时：固定随机种子的重复运行

---

## 10. 后续记录模板

```
### <run 标签> — <日期>
- 命令：
- 起止（UTC）：
- 状态：完成 / 中断于 epoch N / 失败
- 权重：checkpoint_final / checkpoint_best（epoch ?）
- 关键指标：
- 门控权重状态（conv2 |W|_F / bias）：
- 输出目录：
- 备注：
```
