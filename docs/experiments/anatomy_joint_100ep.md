# `anatomy_joint_100ep`（Stage 1：解剖先验生成器）

> 单实验文档。**只记录本实验自身**，不写跨实验对比、不引用其它实验文档（规则见 `AGENTS.md` §6）。
> 指标定义见 `docs/Evaluation_Protocol.md`；方法机制见 `docs/Method.md`。

## 标识

| 项 | 值 |
|---|---|
| variant | `anatomy_joint_100ep` |
| Trainer 类 | `nnUNetTrainerPICAI_AnatomyJoint_100ep_NoFFT` |
| 数据集 / 配置 / fold | `Dataset607_PICAI_Anatomy` / `3d_fullres` / fold 0（冻结显式 split） |
| 在主线中的角色 | Stage 1 = `anatomical prior generator`。**不是论文的方法创新点** |

## 一次性配置（冻结）

| 项 | 值 |
|---|---|
| 输入通道 | 1（T2W） |
| 标签编码 | 位模式 `WG+2*PZ+4*TZ`；`regions_class_order = [1, 2, 4]`（WG / PZ / TZ） |
| 区域定义 | WG `[1,3,5,7]`、PZ `[2,3,6,7]`、TZ `[4,5,6,7]` |
| provenance | `wg_source = materialized_wg`、`zonal_source = zonal_yuan`（算法伪监督） |
| 显式排除 | `11050_1001070`（已知无可用 WG） |
| 病例数 | 1499 study（train 1276 / val 223） |
| 网络 | 由 `nnUNetPlans.json` 构建的原生 `PlainConvUNet`（`n_stages=7`） |
| patch / batch / spacing | `[16, 320, 320]` / 2 / `[3.0, 0.5, 0.5] mm` |
| 损失 | nnU-Net 原生 region-based Dice+BCE（**未**覆盖 `_build_loss`） |
| 预算 | 100 epoch（**工程性试跑预算**，不是研究性预算） |
| 采样 | nnU-Net 原生 dataloader（**不**使用阳性采样） |
| 增强 | 默认管线 + NoFFT 修复 |
| 种子 | **无显式 seed**（本次运行早于 `--seed` 支持） |

## 运行事实

| 项 | 值 |
|---|---|
| 起止（UTC） | 2026-10-06 **11:02:00 → 12:28:20**（100 epoch 训练） |
| validation | 至 2026-10-06 **12:38:15**（`Validation complete`） |
| epoch | **100 / 100 完成**（日志含 `Epoch 0` … `Epoch 99`） |
| 训练日志 | `training_log_2026_10_6_11_01_59.txt` |
| train_loss（末 3 epoch） | -0.9027 / -0.8976 / …（region Dice 损失的负值口径） |

## 验证结果

`validation/summary.json` 的 `Mean Validation Dice = 0.6134037658907373`。
**这是三个 region 的算术平均，不能单独引用**——逐区域结果如下：

| region | 标签集合 | Dice | IoU | TP | FP | FN | n_ref（逐例均值） | n_pred（逐例均值） |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **WG** | `[1,3,5,7]` | **0.00558** | 0.00280 | 334.79 | 282.83 | 121303.20 | 121637.99 | 617.63 |
| PZ | `[2,3,6,7]` | 0.89849 | 0.81724 | 34955.74 | 3496.52 | 4224.70 | 39180.44 | 38452.26 |
| TZ | `[4,5,6,7]` | 0.93614 | 0.88166 | 83959.15 | 4893.40 | 4946.05 | 88905.20 | 88852.55 |

（`foreground_mean`：Dice 0.61340 / IoU 0.56723 / TP 39749.89 / FP 2890.92 / FN 43491.32。）

## 产物

```text
outputs/nnUNet_results/Dataset607_PICAI_Anatomy/
└── nnUNetTrainerPICAI_AnatomyJoint_100ep_NoFFT__nnUNetPlans__3d_fullres/
    ├── plans.json / dataset.json / dataset_fingerprint.json
    └── fold_0/
        ├── checkpoint_final.pth        （约 341 MB）
        ├── checkpoint_best.pth         （约 341 MB）
        ├── progress.png / debug.json
        ├── training_log_2026_10_6_11_01_59.txt
        └── validation/
            ├── <case>.npz   × 223   soft probability（键 probabilities，(3,Z,Y,X)，WG/PZ/TZ）
            ├── <case>.pkl   × 223   物理元数据
            ├── <case>.nii.gz × 223  原生 region 导出的硬标签
            └── summary.json
```

**勿删除 / 勿覆盖**（含 checkpoint、validation 产物与日志）。

可直接用于下游的产物是 **`validation/*.npz`**：验证 split 的解剖 soft prior 已经存在。

## 2026-10-08 — 独立soft heads只读诊断

输入为本实验既有223例validation概率/reference；未新增训练或推理，原summary/NIfTI未改。
命令与耗时登记见Training_Log；本节只记录该实验自身诊断结果。
固定 **>0.5**（与原生region hard export一致），全部病例通过形状/物理几何校验。

| soft head | macro Dice | macro recall | macro precision | macro pred/ref volume ratio | TP sum | FP sum | FN sum | empty |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| WG | 0.94830182 | 0.95322978 | 0.94529174 | 1.00988294 | 26082103 | 1510582 | 1043170 | 0/223 |
| PZ | 0.89928774 | 0.89655433 | 0.90461573 | 0.99326969 | 7843174 | 824719 | 894064 | 0/223 |
| TZ | 0.93613836 | 0.93800987 | 0.93614022 | 1.00472692 | 18722890 | 1091229 | 1102969 | 0/223 |

WG micro Dice=0.95332881、recall=0.96154251、precision=0.94525426、pred/ref ratio=1.01723161。

每病例分位数先独立计算再等权平均，q=[0,.1,.5,.9,1]，不是pooled voxel quantiles：

| head | mean case quantiles within reference region |
|---|---|
| WG | [0.00255918, 0.87615816, 0.99927102, 0.99982368, 0.99994952] |
| PZ | [0.00007743, 0.53192432, 0.99942835, 0.99993453, 0.99998350] |
| TZ | [0.00038660, 0.84357364, 0.99961863, 0.99989143, 0.99997850] |

| head | mean case quantiles over full FOV |
|---|---|
| WG | [5.8194e-10, 8.1905e-7, 3.2213e-6, 1.8863e-5, 0.99994952] |
| PZ | [3.2195e-11, 1.3235e-7, 8.5891e-7, 5.6267e-6, 0.99998350] |
| TZ | [2.2376e-10, 3.6142e-7, 1.5562e-6, 9.2510e-6, 0.99997850] |

从soft heads按原生顺序写1/2/4重建硬导出，与全部223例既有NIfTI差异总数 **0 voxel**。
Success=223、failed=0、skipped=0；耗时568.04s，结果stdout，未创建新报告。

## 观察与限制（更正此前hard-export解释）

1. **soft WG没有显示塌缩**。独立threshold head的Dice/recall/precision与空预测数支持此结论。
2. 原hard-export WG Dice≈0.0056是ordered region export的指标；PZ/TZ后写覆盖WG，
   导出值2/4不在WG集合[1,3,5,7]。不能把该数解释为“WG头实际不可用”。
3. 因此“当前权重必然触发空WG全视野回退”的旧推断不成立；本次没有据此重训Stage-1。
4. reference仍为算法伪监督；不证明人工解剖准确性、概率校准、外部泛化或下游lesion收益。
5. 无显式seed、100ep工程预算；没有跨seed稳定性或训练上限结论。

## 待办

- [x] 独立soft-head质量与hard export重建诊断。
- [ ] 用户运行冻结模型对完整训练split的soft prior生成，记录in-sample/held-out来源。
- [ ] 核对训练/验证prior质量shift；预算允许时评估OOF成本。
- [ ] 下游实验根据真实prior pipeline与baseline选型推进；不预判下游收益。

## 后续记录模板

```text
### <variant> — <dataset>/<config>/fold <k>
- 命令：
- 开始 / 结束（UTC）：
- 状态：完成 / 中断于 epoch N / 失败
- 关键结果：Mean Validation Dice = ，逐区域 Dice = WG / PZ / TZ
- 输出目录：
- 备注：
```
