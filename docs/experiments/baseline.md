# 实验记录：`baseline`

> 本文档只记录 `baseline` 实验本身：配置、运行事实、训练动态、验证结果与产物。
> 所有数字均取自本实验自己的产物，未引用任何其他实验。

---

## 1. 实验标识

| 项 | 值 |
|---|---|
| variant | `baseline` |
| Trainer 类 | `nnUNetTrainerPICAI_FLCE_NoFFT` |
| 数据集 / 配置 / fold | `Dataset605_PICAI` / `3d_fullres` / fold 0 |
| 输出目录 | `outputs/nnUNet_results/Dataset605_PICAI/nnUNetTrainerPICAI_FLCE_NoFFT__nnUNetPlans__3d_fullres/fold_0/` |
| 训练入口 | `scripts/train/train_nnunet.py baseline 605 3d_fullres 0` |
| 预测入口 | `scripts/inference/predict_nnunet.py -m <上述输出目录> -f 0` |

研究对象：T2W / ADC / HBV 三序列通道拼接输入，原生 nnU-Net backbone，PI-CAI 风格 Focal+CE 损失。

---

## 2. 一次性配置

### 2.1 数据

- PI-CAI 训练集 1500 例；固定划分 fold 0 = **1277 train / 223 validation**（患者级无重叠）
- 通道顺序（固定不可变）：`0000` = T2W，`0001` = ADC，`0002` = HBV
- 归一化：三通道均为 `ZScoreNormalization`；`use_mask_for_norm = [False, False, False]`
- patch `16 × 320 × 320`，batch size `2`，target spacing `3.0 × 0.5 × 0.5` mm
- `do_dummy_2d_data_aug = True`
- 验证集参考病灶体积（体素，1 体素 = 0.75 mm³）：
  min **190** / p25 **965** / median **2459**（≈1.84 mL）/ p75 **7001** / max **123055**
- 223 例中 **63 例有参考病灶**，160 例为阴性

### 2.2 网络

- 由 `nnUNetPlans.json` 经原生 `get_network_from_plans` 构建：`PlainConvUNet`
- `n_stages = 7`，`features_per_stage = [32, 64, 128, 256, 320, 320, 320]`
- `n_conv_per_stage = [2]*7`，`n_conv_per_stage_decoder = [2]*6`
- deep supervision：**6 个输出**（含最低分辨率）
- 参数量：**44,577,932**（fp32 178.3 MB）

### 2.3 训练

- optimizer：SGD，`momentum = 0.99`，`nesterov = True`，`weight_decay = 3e-5`
- 学习率：`initial_lr = 0.01`，`PolyLRScheduler(power = 0.9)`，随 epoch 线性衰减到 ~0
- 训练长度：**1000 epoch × 250 iter = 250,000 iter**
- 前景采样：`oversample_foreground_percent = 0.33`；batch_size=2 时
  `round(2 × (1 − 0.33)) = 1`，即每个 batch 中 **2 个 patch 有 1 个被强制居中于前景**
- `save_every = 50`；`torch.compile` 启用
- 环境：NVIDIA RTX 3090，torch 2.10.0+cu128，cudnn 91002

### 2.4 损失

- `PI-CAI 风格`：`0.5 × Focal(gamma=2, alpha=None, smooth=1e-5) + 0.5 × CE`
- Focal 复刻 PI-CAI v1 实现：softmax → one-hot `clamp(smooth/(C−1), 1−smooth)` → `pt + smooth`
  → `−(1−pt)^gamma · log(pt)`，逐体素取均值
- deep supervision 外层为原生 `DeepSupervisionWrapper`，权重 `[1, 1/2, 1/4, 1/8, 1/16, 1/32]`
  归一化，**最低分辨率权重置 0（不监督）**
- Dataset605 无 ignore label、非 region 训练

---

## 3. 运行记录

| 项 | 值 |
|---|---|
| 训练起止（UTC） | 2026-09-19 11:15:10 → 2026-09-20 02:37:36（**15 h 22 min**） |
| 验证起止（UTC） | 2026-09-20 02:37:36 → 02:48:25（约 11 min，223 例） |
| epoch 耗时 | mean **53.74 s** / median 53.69 s / min 52.62 s / max 77.01 s（首 epoch 含 `torch.compile`） |
| 日志异常 | 无 warning / error / NaN / 中断，一次跑完 |
| 随机种子 | **nnU-Net v2.6.2 不设置任何随机种子**（全库无 `manual_seed`）→ 权重初始化不可复现 |
| 验证用权重 | `checkpoint_final.pth`（epoch 999） |

---

## 4. 训练动态

### 4.1 关键 epoch

| epoch | train_loss | val_loss | pseudo dice |
|---|---|---|---|
| 0 | 0.0302 | 0.0069 | 0.0000 |
| 5 | 0.0036 | 0.0042 | 0.0000 |
| 100 | 0.0023 | 0.0022 | 0.0000 |
| 300 | 0.0015 | 0.0023 | 0.0037 |
| 500 | 0.0012 | 0.0019 | 0.0023 |
| 700 | 0.0023 | 0.0020 | 0.0219 |
| 800 | 0.0014 | 0.0026 | 0.2616 |
| 900 | 0.0010 | 0.0015 | 0.3921 |
| 950 | 0.0011 | 0.0010 | 0.5842 |
| 999 | 0.0011 | 0.0009 | 0.3831 |

### 4.2 分箱均值

| epoch 区间 | pseudo dice 均值 |
|---|---|
| 0–99 | 0.0040 |
| 100–199 | 0.0037 |
| 200–299 | 0.0043 |
| 300–399 | 0.0037 |
| 400–499 | 0.0015 |
| 500–599 | 0.0084 |
| 600–699 | 0.0077 |
| 700–799 | 0.2186 |
| 800–899 | 0.4908 |
| 900–999 | 0.5237 |

### 4.3 训练动态事实

- **epoch 0–699 的 pseudo dice 均值恒在 0.0004–0.008 量级**，即模型在该区间内基本不输出病灶
- 首次连续 ≥20 个 epoch 的 pseudo dice > 0.05：**epoch 756**；> 0.20：**epoch 764**
- 最佳 EMA pseudo dice：**0.5820 @ epoch 986**
- 单 epoch 最高 pseudo dice：**0.7670 @ epoch 937**
- 最后 100 epoch pseudo dice：mean 0.5237 / median 0.5410 / max 0.7670 / min 0.0645
- EMA 提升（`Yayy`）次数：**66**
- **loss 与指标解耦**：pseudo dice 从 0.02（epoch 700）涨到 0.49（epoch 800）期间，
  train_loss 从 0.0023 降到 0.0014、val_loss 从 0.0020 升到 0.0026 —— 损失曲线不反映前景学习

---

## 5. 验证结果

### 5.1 nnU-Net 汇报指标

```
foreground_mean.Dice = 0.16816659097420447     (223 例，np.nanmean)
foreground_mean.IoU  = 0.12353439259349808
TP = 1053.4   FP = 511.8   FN = 1546.0   n_pred = 1565.2   n_ref = 2599.4
```

### 5.2 该数字的构成（反推自 `summary.json`）

| 类别 | 数量 | 对 Dice 的贡献 |
|---|---|---|
| 真阴（无参考、无预测） | 149 | 记 NaN，**被剔除** |
| 假阳阴性（无参考、有预测） | 11 | 记 **0** |
| 阳性（有参考） | 63 | 记实际 Dice |
| **分母** | **74** | 分子（63 例 Dice 之和）= 12.4443277 |

校验：`12.4443277 / 74 = 0.1681665910`，与汇报值 `0.16816659097420447` 一致。

### 5.3 分层口径

| 口径 | 值 |
|---|---|
| 阳性 63 例 per-case mean Dice | **0.1975** |
| 阳性 63 例 per-case median / max | 0.0000 / 0.8689 |
| 阳性病例 pooled（micro）Dice | **0.5212** |
| `positive_voxel_recall` = `ΣTP / (ΣTP + ΣFN)`（仅阳性病例） | 0.4052446901 |
| `positive_voxel_precision` = `ΣTP / (ΣTP + ΣFP)`（仅阳性病例，**论文核心口径**） | **0.7301160557** |
| `all_prediction_voxel_precision` = `ΣTP / Σn_pred`（分母**含阴性病例假阳体素**） | 0.6730222670 |
| 阳性病例 预测体积 ÷ 参考体积 | **0.5550** |

仅在**有重叠的 29 例**（`Dice > 0`）中逐例取中位数：

| 指标中位数（29 例） | 值 |
|---|---|
| 预测体积 ÷ 标注体积 | 0.377 |
| 病灶体素召回（`TP / n_ref`） | 0.353 |
| 预测精度（`TP / n_pred`） | 0.889 |

即：预测出来的区域多数确实落在病灶内，但只覆盖约 **35 %** 的标注区域 —— 高精度、低召回。

### 5.4 检出与假阳结构

| 项 | 数量 | 比例 |
|---|---|---|
| 阳性中**完全无重叠**（`Dice = 0`） | **34 / 63** | **54.0 %** |
| ↳ 其中完全无预测（`n_pred = 0`） | 32 / 63 | 50.8 % |
| ↳ 其中有预测但不重叠 | 2 / 63 | 3.2 % |
| 阳性中有重叠（`Dice > 0`） | 29 / 63 | 46.0 % |
| 阳性中 `Dice > 0.5` | 14 / 63 | 22.2 % |
| 阳性中 `Dice > 0.7` | 6 / 63 | 9.5 % |
| 有重叠 29 例的 Dice mean / median | 0.4291 / 0.4771 | — |
| 有预测 31 例的 Dice mean / median | 0.4014 / 0.4152 | — |
| 阴性中出现预测（假阳病例） | **11 / 160** | **6.9 %** |
| ↳ 这些病例的预测体积中位数 | 1,464 体素 | — |
| 阴性病例假阳体素合计 | 27,294 | — |
| 阳性病例 FN / FP 体素 | 344,763 / 86,833 | 比值 ≈ 4.0 |

### 5.5 按参考病灶体积分箱

| n_ref 区间（体素） | n | 检出 | 召回 | 精确 | pred/ref | pooled Dice |
|---|---|---|---|---|---|---|
| 0–1000（<0.75 mL） | 16 | 3 | 0.062 | 0.020 | 3.041 | 0.031 |
| 1000–5000（0.75–3.75 mL） | 27 | 13 | 0.125 | 0.819 | 0.153 | 0.217 |
| 5000–20000（3.75–15 mL） | 15 | 9 | 0.300 | 0.726 | 0.413 | 0.425 |
| >20000（>15 mL） | 5 | 4 | 0.509 | 0.819 | 0.621 | 0.627 |

按病灶负荷上限累积（检出定义为 `Dice > 0`）：

| 病灶负荷上限 | 例数 | 检出 |
|---|---:|---:|
| ≤ 500 体素 | 6 | **0/6** |
| ≤ 1000 体素 | 16 | 3/16 |
| ≤ 2000 体素 | 26 | 6/26 |

按四分位分组（边界取自本实验 63 个阳性病例的标注体素数）：

| 病灶负荷（体素） | n | 检出 | 平均 Dice |
|---|---:|---:|---:|
| 190–965 | 16 | 3 | 0.0518 |
| 1128–2459 | 16 | 7 | 0.1527 |
| 2624–7001 | 16 | 9 | 0.1797 |
| 7724–123055 | 15 | 10 | 0.4198 |

> **口径限制**：以上是**病例级病灶负荷**分析，不是病灶**实例**级分析。一例可能含多个病灶，
> `summary.json` 只提供整例二值统计；真正的小病灶结论需要连通域级别的 lesion-size sensitivity。

---

## 6. 产物清单

| 文件 | 说明 |
|---|---|
| `checkpoint_best.pth` | **epoch 986**（最后一次 EMA 提升，EMA pseudo dice 0.5820） |
| `checkpoint_final.pth` | **epoch 999**；本次验证使用的权重 |
| `progress.png` | loss / pseudo dice / epoch 耗时 / LR 曲线 |
| `validation/summary.json` | 逐病例指标（本文档 5.x 节全部数字来源） |
| `validation/*.nii.gz` | 223 例验证预测 |
| `training_log_2026_9_19_11_15_10.txt` | 完整训练日志（7540 行） |
| `debug.json` | 运行时配置快照 |

**未产出**：验证概率（`.npz`）未导出，因此本实验暂无法做**分割概率阈值敏感性分析**。

---

## 7. 本实验的观察与限制

1. **训练前 70% 的时间未产生前景预测**：epoch 0–699 的 pseudo dice 均值在 0.004 量级，
   学习集中在最后约 300 个 epoch。若从零重训，前 ~10 小时基本没有可观测进展。
2. **损失曲线不可用于监控**：整个训练过程 train/val loss 都在 0.001–0.004，且与 pseudo dice
   方向不一致；按 `val_loss` 选模型会选到全背景解。
3. **汇报的 0.1682 由三层因素共同压低**：
   - 11 个假阳阴性病例各计一个 0，把 63 例均值的分母从 63 拉到 74（0.1975 → 0.1682）；
   - 50.8% 的阳性病例零预测，分子直接损失一半；
   - 模型系统性欠分割：预测体积只有参考的 55.5%，`positive_voxel_precision` 为 0.7301，
     而 `positive_voxel_recall` 仅 0.4052。
4. **误差结构随病灶体积单调变化**：精确度在各档稳定在 0.72–0.82，召回从 0.06 升到 0.51。
   中间档（1000–5000 体素）最典型：精确 0.819 而召回仅 0.125 —— 只输出了病灶核心。
5. **`checkpoint_best` 从未做全 volume 验证**：本实验报告值与所有逐病例分析均来自
   `checkpoint_final`（epoch 999）。
6. **初始化不可复现**：nnU-Net v2.6.2 无随机种子，本实验的单次结果不可精确复现。

---

## 8. 待办

- [ ] 导出验证概率：
      `python scripts/train/train_nnunet.py baseline 605 3d_fullres 0 --validation-only --export-validation-probabilities`
      （会重写 `validation/` 目录，建议先备份 `summary.json`）
- [ ] 用 `checkpoint_best.pth` 推理一次，得到 epoch 986 权重的全 volume 指标
- [ ] 完成分割表面指标：`python scripts/evaluate_segmentation.py --mode full --nsd-tolerance-mm <τ>`
- [ ] 物理体积分层与连通域失败分析：同上再加 `--volume-thresholds-mm3 <T1,T2>`
- [ ] 概率就绪后：**分割概率阈值敏感性分析**（Dice / 召回 / 精确 vs 阈值）。
      本实验只做病灶**分割**评估，**不涉及** AUROC / AP / FROC / PI-CAI challenge score
- [ ] 需要统计效力时：固定随机种子的重复运行

---

## 9. 后续记录模板

```
### <run 标签> — <日期>
- 命令：
- 起止（UTC）：
- 状态：完成 / 中断于 epoch N / 失败
- 权重：checkpoint_final / checkpoint_best（epoch ?）
- 关键指标：
- 输出目录：
- 备注：
```
