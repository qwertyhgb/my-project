# Method

**Anatomy-Guided Lesion-Aware Coarse-to-Fine Prostate Cancer Segmentation**

> 本文件记录**方法机制与实现边界**：每个模块做什么、为什么这样做、在什么条件下会 fail-closed。
> 它不记录实验结果（→ `docs/Training_Log.md` / `docs/Findings.md`）、不记录指标定义
> （→ `docs/Evaluation_Protocol.md`）、不记录实验矩阵与预算（→ `docs/Experiment_Plan.md`）。
> 类名与函数名一律给出**可核查**的真实位置；代码是唯一真源，本文件是它的可读说明。

---

## 0. 总体数据流

```text
                    ┌─────────────────────────────┐
   T2W  ───────────▶│ Stage 1: anatomy model      │──▶ P(WG) / P(PZ) / P(TZ)   （soft, 冻结）
                    │ Dataset607 / 单通道 T2W     │
                    └─────────────────────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │ Anatomy-Guided ROI （条件B） │──▶ 前列腺 ROI（bbox + 物理 margin）
                    └─────────────────────────────┘
                                   │
   T2W / ADC / HBV ────────────────┤
   predicted P(PZ) / P(TZ) ────────┤
                                   ▼
                    ┌─────────────────────────────┐
                    │ Lesion-aware coarse-to-fine │
                    │ （条件 C / D）               │
                    │ shared backbone（原生）      │
                    │   ├── coarse lesionness 头   │
                    │   └── soft 残差 refinement   │
                    └─────────────────────────────┘
                                   │
                                   ▼
                         final lesion segmentation
```

三层数据的职责（禁止混淆，见 `docs/Evaluation_Protocol.md`）：

| 层 | 内容 | 角色 |
|---|---|---|
| raw inputs | T2W / ADC / HBV | 模型输入（原始医学数据只读） |
| training labels | lesion GT（Dataset605/606） | 训练监督与验证参照 |
| predicted priors | `P(WG)` / `P(PZ)` / `P(TZ)` | Stage-1 模型对**该病例自身 MRI** 的预测 |
| oracle labels | GT WG / PZ / TZ | 仅上界分析，必须标记 `ORACLE_GT` |

---

## 1. Stage 1 — Predicted Anatomy Priors

### 1.1 数据集契约

`zonal_reliability_fusion.anatomy.contracts`

| 项 | 值 |
|---|---|
| 数据集 | `Dataset607_PICAI_Anatomy` / `3d_fullres` |
| 输入通道 | **单个 T2W**（`channel_names == {"0000": "T2W"}`） |
| 标签 | 三个**非互斥**区域的位编码 `WG+2*PZ+4*TZ`（`regions_class_order = [1, 2, 4]`） |
| 来源 | `wg_source = materialized_wg`、`zonal_source = zonal_yuan`（算法伪监督，非手工标注） |
| 范围 | 1499 study（**study 数，不是患者数**）；train 1276 / val 223 |
| 显式排除 | `11050_1001070`（已知无可用 WG）：**不得**以空掩膜或 zonal union 顶替 |

`validate_anatomy_dataset` 逐项校验上述全部内容；`validate_anatomy_split` 额外要求 **fold 恒为 0**
且**无随机回退**（split 文件缺失即报错），并检查 train/val 的**患者级**不重叠。

### 1.2 位编码的语义

位模式**不是**新的解剖分类，而是"区域隶属组合"：一个体素可以同时属于 WG 与 PZ（编码 1+2=3）。
因此：

- 训练使用 nnU-Net 的 **region-based heads**（三个独立的 sigmoid 区域头），不是互斥 softmax；
- 评估用 `region_or_label_to_mask` 按区域展开，**空-空区域的 Dice 定义为 `None`**（不是 0，也
  不是 1）；
- 下游 lesion model **只读 soft probability**（`npz` 的 `probabilities`），**不读** `nii.gz` 里的
  argmax 硬标签。

### 1.3 训练器

`zonal_reliability_fusion.nnunet.bases.nnUNetTrainerPICAI_AnatomyJoint_100ep_NoFFT`

- 共享原生 `PlainConvUNet`、三个 region head；100 epoch 是**工程性试跑预算**，不是研究性预算；
- `__init__` 与 `initialize` **两处**都做数据集/配置/region/split 校验（fail-closed）；
- 拒绝 cascade 输入（`previous_stage_name` 必须为 `None`）；
- 输入通道数必须为 1。

### 1.4 导出与校验

`zonal_reliability_fusion.anatomy.inference`

- 预测必须 `--save_probabilities`（下游需要 soft）；不支持 partitioned / cascade 预测；
- 输入必须只含单通道 `*_0000.nii.gz`；输出目录非空即拒绝覆盖；
- 完成后逐病例校验三件套（`.npz` / `.pkl` / `.nii.gz`）齐全、概率形状 `(3, Z, Y, X)`、值域
  `[0, 1]`、物理元数据与**该病例自身 T2W** 同网格；任一病例失败即 fail-closed 并列出全部错误。

### 1.5 先验质量报告

`zonal_reliability_fusion.anatomy.validation`

- `anatomy_region_metrics`：WG / PZ / TZ 的 Dice 与 TP/FP/FN/TN（复用 nnU-Net 原生 region 定义）；
- `prior_uncertainty_report`：
  - `wg_low_confidence_fraction`——`threshold <= P(WG) <= 1 - threshold` 的体素比例，这正是 ROI
    边界最容易出错的位置；
  - `zone_undecided_fraction`——`P(PZ)` 与 `P(TZ)` 都低于阈值的比例；
  - `zone_margin = |P(PZ) - P(TZ)|` 的分布。

  这些是 **soft 输出的数值分布描述**，**不是**校准（calibration）声明，也不代表真实图像质量。

---

## 2. Anatomy-Guided ROI（条件 B）

`zonal_reliability_fusion.lesion.roi` + `zonal_reliability_fusion.nnunet.roi_sampling`

### 2.1 为什么不是 hard mask

```python
image = image * wg_mask     # ✗ 本项目明确禁止
```

原因：WG 预测的边界错误会**直接删除**跨越腺体边界的病灶，而且这一失败在进入网络之前发生、
不可恢复。ROI 只做 **bounding box + 物理 margin**，永远保留腺体周围的完整安全边界。

### 2.2 物理 margin 的定义

```text
predicted WG --(>= wg_threshold)--> 二值 mask --> bbox [lower, upper)
             --expand_box_by_physical_margin--> 每轴 voxel 扩张 = ceil(margin_mm / spacing_axis)
             --clip 到体积边界--> PROSTATE ROI
```

- `margin_mm` 必须由调用方显式给出，模块**不设**默认值；
- 每轴**向上取整**，保证实际物理边界 ≥ 承诺值；
- 为什么不能用固定 voxel 数：`3d_fullres` 的 spacing 是 `[3.0, 0.5, 0.5] mm`，用同一个 voxel
  常数会让 z 方向的物理边界比面内小 6 倍。

### 2.3 不重采样

crop 之后通过 **padding** 满足网络输入尺寸要求（`pad_plan`），因此：

- spacing / origin / direction 完全不变；
- 物理体积与评估协议的 mm³ 口径一致；
- 预测可以**无损**恢复到原空间（`ProstateROI.restore`，ROI 外填指定值）。

`ProstateROI` 完整保存 `lower` / `upper` / `original_shape` / `spacing_zyx` / `margin_mm` /
`wg_threshold` / `prior_source` / `fallback_reason`，并可通过 `to_json()` 写入 run 配置或评估报告。

### 2.4 空 / 小 WG 预测

| 情况 | 行为 |
|---|---|
| `WG >= threshold` 为空 | 默认 **回退全视野** 并记录 `fallback_reason="empty_wg_prediction"`；`empty_fallback="error"` 时抛错 |
| WG 覆盖整幅图像 | 回退全视野，记录 `fallback_reason="wg_covers_full_fov"` |
| WG 很小（ROI 小于 patch） | 仍保留完整 margin；patch 居中于 ROI 并夹到原生允许范围 |

**选择永远是"保留更多"**：空 WG 预测**不**允许产生"把病灶裁掉"的 ROI。

### 2.5 训练采样层

`ROIDataLoader`（继承 `PositiveCaseDataLoader`）只覆盖 `get_bbox`：

- **阳性槽位不受 ROI 约束**——阳性槽位仍由 nnU-Net 原生的病灶中心裁剪决定。这样条件 B 与
  strong baseline 的**阳性病灶暴露完全相同**，单变量边界更干净；否则"保证一个病灶 patch"
  这一性质就会依赖 WG 预测正确，把 anatomy 误差引入病灶暴露保证里。
- **非前景槽位**以 `roi_sampling_probability`（冻结常量，默认 0.75）的概率把 bbox 下界限制在 ROI
  内；其余槽位保持原生全视野采样。保留一部分全视野槽位是**有意**的：滑窗推理会在整幅图像
  （含腺体外背景）上运行，训练时完全排除这些位置会引入新的分布不匹配。
- **验证 loader 恒为原生** `nnUNetDataLoader`：ROI 永远不进入验证采样。

**为什么是"采样空间约束"而不是"裁剪输入"**：裁剪会在推理时引入 train/test 不一致并需要额外的
几何恢复步骤；约束采样空间直接回答条件 B 的科学问题，且不需要改变推理路径。
`ProstateROI` 的 crop/restore 能力保留给将来需要推理期裁剪的变体。

### 2.6 fail-closed 清单

- ROI 集合缺失 / schema 版本不符 / 数据集或 fold 归属不符 → 报错；
- ROI 集合包含**非训练 split** 病例 → 报错；
- `provenance.prior_source == ORACLE_GT` → 报错（GT 解剖标签不允许作为正式 ROI 来源）；
- `lower_zyx` / `upper_zyx` 非三个非负整数或区间非法 → 报错；
- `roi_sampling_probability` 不在 `[0, 1]` → 报错；
- `roi_boxes` 为空或含未知病例 ID → 报错。

---

## 3. Lesionness / Coarse-to-Fine（条件 C）

`zonal_reliability_fusion.lesion.lesionness` + `lesion.coarse_to_fine` + `nnunet.losses`

### 3.1 目标形式（唯一选择）

**物理半径膨胀的 coarse lesion mask**：

```text
lesion GT --(binary_dilation, 按 spacing 折算的椭球结构元素)--> coarse lesionness target
```

- 半径 `LESIONNESS_DILATION_RADIUS_MM = 3.0`（**预先冻结**，未看结果前固定，不得事后调参）；
- 膨胀使用**椭球**结构元素：各向异性 spacing 下用立方体会把 z 方向的物理半径放大 6 倍；
- 半径折算**向上取整**（物理覆盖不小于承诺）；
- 空 GT（阴性病例）→ 全零目标，这是**合法输入**，不是错误；
- 非二值标签（含 ignore `-1`）→ 报错，不得静默当成背景。

为什么不用 Gaussian center heatmap：峰值出现在单个体素上，3D 稀疏病灶下类不平衡极端，且 sigma
需要额外调参（引入不可归因超参数）。为什么不用 distance transform：语义更难解释，且与现有二值
标签形式不一致，需要新的 target 编码。

### 3.2 网络结构

`LesionAwareCoarseToFineNNUNet` 是对**原生 backbone 的薄包装**：

```text
输入 x = [T2W, ADC, HBV] (+ [P(PZ), P(TZ)] 若为条件 D)
  │
  ├─▶ backbone（由 nnUNetPlans.json 构建，不复制 encoder/decoder）
  │        │
  │        ├── 最粗尺度输出 ──▶ CoarseLesionnessHead（1×1×1，末层零初始化）──▶ lesionness logits
  │        │
  │        └── 全分辨率输出 F
  │                 │
  │                 │  context = [ sigmoid(lesionness), P(PZ), P(TZ), P(U) ]（上采样到全分辨率）
  │                 ▼
  │            ZoneAwareRefinement（末层零初始化）:  F_out = F + delta
  │                 │
  └─────────────────┴──▶ 返回**与原生 backbone 完全同形**的输出（DS 时等长列表）
```

四条硬约束（逐条对应 Research Plan §6）：

1. **coarse 预测不能硬裁掉任何区域**——本网络从不对输入或特征做空间裁剪；
2. **lesionness 只是 soft guidance**——不做阈值化、不当 attention mask；
3. **零 coarse score 区域仍保留 residual path**——`F_out = F + delta` 且 `delta` 末层零初始化，
   即使 coarse 头输出 0，`F` 仍原样通过；
4. **防止 coarse 头错误导致无法恢复**——由于 (3)，**初始状态与 strong baseline 逐值一致**；
   训练中即使 coarse 头误差较大，最终分割仍有完整的分割分支路径。

**不引入 Transformer / Mamba / cross attention**，除非后续实验明确证明 CNN 无法解决问题。

### 3.3 辅助损失

`LesionnessAuxiliaryLoss` 做 `main_loss(output, target) + weight * lesionness_loss(aux_logits, coarse_target)`：

- `LESIONNESS_LOSS_WEIGHT = 0.5`（**预先冻结**）；
- coarse 目标由**全分辨率 lesion GT** 先按真实 spacing 按物理半径膨胀，再 **max-pooling** 到 coarse
  网格。语义恰好是"该 coarse 体素附近 r mm 内是否有真实病灶"，不需要在 coarse 网格上重新解释
  spacing；不整除即报错（避免静默坐标错位）；
- 主损失与辅助损失在**同一次 backward** 中联合优化；
- 它**不替换**主损失，因此条件 C 相对 B 的差异可以清楚地归因到"多了一个 coarse 定位监督 + soft
  refinement"，而不是"换了损失"。

### 3.4 辅助输出通道的传递方式（如实说明的限制）

nnU-Net 的 `validation_step` 会对网络返回值直接 `argmax`，因此网络**必须只返回分割输出**。
lesionness logits 通过模块属性 `auxiliary_outputs` 传递，由 `LesionnessAuxiliaryLoss` 在同一个
`train_step` 内立即读取。

**限制**：这条通道依赖"同一进程内 forward 后立即取用"，因此只适用于本项目当前的单进程、单 GPU
配置（`num_gpus=1`）。若将来改用 DDP，必须改为显式返回元组并自定义
`train_step`/`validation_step`——本项目**不**采用那种做法，因为那会开始重写训练循环。缺失辅助
输出时 loss **直接报错**，不静默退化为纯主损失。

---

## 4. Zone-Aware Refinement（条件 D）

`zonal_reliability_fusion.lesion.zone_conditioning`

### 4.1 PZ/TZ 的作用定义

```text
P(PZ) / P(TZ) 回答：这个候选病灶长在哪里？周围是什么组织？
                     它落在 PZ / TZ / 分区边界 / 解剖不确定区？
```

**不是**"根据区域调整 T2W / ADC / HBV 的 modality weight"。接口是 context provider：
输入 lesion 的共享特征与 soft 区域概率，输出**同形状同通道数**的解剖条件化特征。描述符与实现
不得出现 "modality weight / sequence weight / reliability" 这类语义。

### 4.2 uncertain 的派生

```text
P(U) = 1 - max(P(PZ), P(TZ))
```

它不是 anatomy 模型直接输出的类别，而是由 soft 概率派生的**不确定度**：两个区域证据都不强时取
大值，天然覆盖分区边界与 anatomy 预测不确定的区域，**不需要**额外监督一个 uncertain 头。

### 4.3 两种模式

| 模式 | 公式 | 定位 |
|---|---|---|
| `concat`（**默认**） | `F_out = F + zero_init_conv([F, context])` | 参数最少、最易解释；直接检验"concat soft anatomy maps 是否已经足够" |
| `zone_experts` | `F_out = F + zero_init_conv([F_zone, context])`，`F_zone = P(PZ)·F_PZ + P(TZ)·F_TZ + P(U)·F_shared` | 检验"是否真的需要两个 expert" |

`zone_mode_parameter_delta` 报告两种模式的参数量差，使选择可以**基于可控的参数量差异**而不是
直觉。默认**不用** `zone_experts`。

两种模式的输出层都零初始化 ⇒ 初始 `F_out == F`（恒等）。**恒等只保证初始等价，不保证训练后
等价。**

### 4.4 输入契约

- 必须是 **soft probability**（浮点、值域 `[0, 1]`、有限）；整型、logits 或硬 one-hot 一律报错；
- 输入通道数为 **2**（PZ、TZ），派生出 3 通道 context；
- **强度增强只作用于 MRI 通道**（`lesion.prior_channels.restrict_intensity_transforms_to_mri`），
  先验通道只接受空间变换——否则 noise / blur / brightness 会把概率图扭曲成非法输入，让"先验
  错误是真实部署链条的一部分"这一前提被污染成"先验错误是增强伪影"。若在增强管线里找不到任何
  强度变换来包装，则**报错**而不是静默启动。

---

## 5. Hard Negative Mining（条件 E）

`zonal_reliability_fusion.sampling.hard_negative` + `scripts/data/mine_hard_negatives.py`

### 5.1 候选定义

```text
prediction_probability >= confidence_threshold
AND GT == background
AND WG_probability >= wg_threshold
```

把满足条件的假阳**连通域**（6-邻域）按体积降序，取前 `max_locations_per_case` 个连通域的**质心**
作为采样位置。质心位于**预处理坐标空间**，与 nnU-Net `class_locations` 一致。

### 5.2 分轮流程

| 轮次 | 做什么 |
|---|---|
| Round 1 | 训练 strong baseline / coarse-to-fine 模型 |
| Round 2 | **只对训练 split** 推理，按上面的定义挖掘，写出 `hard_negative_set.json`（含 provenance） |
| Round 3 | 训练时提高这些位置的采样概率 |

### 5.3 采样语义（与阳性采样严格区分）

- **阳性槽位**：来自阳性病例，强制病灶中心裁剪（`force_fg=True`）；
- **困难负样本槽位**：来自含困难负样本的病例，位置从该病例的 locations 中均匀有放回抽取，
  强制在**存储位置**裁剪，且 `force_fg=False`（该位置是背景）；
- 其余槽位：完全原生；
- **验证 loader 永远是原生** `nnUNetDataLoader`。

### 5.4 fail-closed 清单

- 挖掘集合包含**验证病例**或不在训练 split 的病例 → 报错；
- 挖掘集合的数据集 / fold 归属不符 → 报错；
- 坐标不是三个非负整数 → 报错；
- `positive_cases_per_batch + hard_negative_cases_per_batch > batch_size` → 报错；
- 困难负样本病例 ID 不属于该 loader 的训练 dataset → 报错；
- 每个 batch 的槽位决策队列**必须被完全消费**，否则下一轮 `get_indices` 报错（防止槽位与位置错位）；
- 困难负样本槽位被标记为 `force_fg` → 报错（原生前景裁剪会覆盖存储位置）；
- `hard_negative_set_path is None` ⇒ **完全退化**为条件 C/D 的行为（同一 Trainer 类的一个参数，
  可直接作为独立 ablation）。

---

## 6. Strong Baseline

`zonal_reliability_fusion.lesion.baseline` + `nnunet.bases`

| 条件 | 组成 | 回答的问题 |
|---|---|---|
| **A1** `positive_sampling` | nnU-Net + PI-CAI `0.5*Focal(gamma=2)+0.5*CE` + 阳性采样 | Focal+CE 能达到什么水平 |
| **A2** `dicece_positive_sampling` | nnU-Net + 原生 Dice+CE + 阳性采样 | 原生 Dice+CE 是否更好 |

**两者唯一允许不同的地方是损失函数。** 网络（原生 `PlainConvUNet`）、数据集与 split、fold、
patch size、batch size、增强（含 NoFFT 修复）、前景采样、deep supervision 权重、
optimizer（SGD+Nesterov）、初始学习率、PolyLR 总周期、epoch 数、checkpoint 策略、
validation 与滑窗推理全部继承同一套 nnU-Net 原生实现。

**阳性采样的定位**：它是 **foreground-aware training stabilization strategy**，**不是**论文的方法
创新；它属于 strong baseline 的一部分，因此出现在论文的"训练设置"，**不**出现在
"Method contribution"。

---

## 7. 单变量边界（逐条件的唯一改动）

| 条件 | 相对上一条的**唯一**改动 | 代码位置 |
|---|---|---|
| A2 vs A1 | 损失（Focal+CE → 原生 Dice+CE） | `nnunet/bases.py`（A2 不继承 FLCE mixin） |
| B vs A | 训练 patch 采样空间（ROI 约束） | `nnunet/roi_sampling.py` |
| C vs B | + coarse lesionness 头 + soft 残差 refinement + lesionness 辅助损失 | `lesion/coarse_to_fine.py`、`lesion/lesionness.py`、`nnunet/losses.py` |
| D vs C | + 两个 soft zone 概率通道（`concat` 模式） | `lesion/zone_conditioning.py` |
| E vs best(C, D) | + 困难负样本采样（`hard_negative_set_path`） | `sampling/hard_negative.py` |

**不要同时修改 loss / sampling / network / ROI / anatomy 然后再比较。**

---

## 8. 明确禁止的实现方向

除非后续实验**强烈支持**，否则**禁止**主动加入：

```text
Transformer / Swin / Mamba / large cross attention / foundation model / diffusion / LLM
```

也不禁止（但需谨慎）：

- 复制 nnU-Net 的完整训练器、U-Net encoder/decoder、验证器、推理器；
- 修改 `third_party/nnUNet`（**绝对禁止**，只读）；
- 自行重采样原始医学图像或从原始 NIfTI 计算病灶坐标（必须复用 nnU-Net 预处理数据与
  `class_locations`）；
- 把缺失 metadata 的病例静默当成阴性（一律 fail-closed）。

默认优先顺序永远是：

```text
simple → interpretable → controlled → reproducible → only then complex
```
