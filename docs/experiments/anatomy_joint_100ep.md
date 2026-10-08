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
| 损失 | nnU-Net 原生 region-based Dice+CE（**未**覆盖 `_build_loss`） |
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

## 观察与限制

1. **WG 区域头实际不可用**。GT 的 WG 参考体积并不小（`n_ref` 逐例均值 121637 voxel，与
   `PZ ∪ TZ` 同量级），但模型只预测出约 618 voxel/例（`n_pred` 逐例均值），因此 Dice ≈ 0.006。
   这不是"参考为空"导致的空-空约定问题，而是**预测侧几乎全为背景**。
2. **集合平均掩盖了失败**。`Mean Validation Dice = 0.6134` 由 0.0056 / 0.8985 / 0.9361 平均而来；
   只看这个数字会得出"Stage-1 可用"的错误结论。这正是 `docs/Evaluation_Protocol.md` 要求逐区域
   报告的现实案例。
3. **原因尚未定位**，且**本文件不预判**。至少有两种可能，且都需要额外证据才能排除：
   - 优化 / 损失侧：三个 region 头共享 backbone，其中一个头塌缩；
   - 数据 / 编码侧：位编码与 `regions_class_order` 下 WG region 的实际定义与预期不一致
     （注意 7 = 三个区域同时隶属，4 = TZ only，5 = WG+TZ 等组合的实际分布尚需核对）。
4. **无显式 seed**：本次运行早于 `--seed` 支持，无法与其它 run 做 run-to-run 比较。
5. **100 epoch 是工程性预算**：不构成"Stage-1 的上限"，也不能据此判断"解剖先验本来就不行"。
6. **下游不可用**：条件 B 的 Anatomy-Guided ROI 唯一来源是 predicted WG。当前权重下 ROI 会因
   `empty_wg_prediction` 回退到全视野（`ProstateROI.fallback_reason`），等价于"没有 ROI"。

## 待办

- [ ] **诊断 WG 失败**（优先级最高；两条线索都要查，不要只查一条）：
  - 数据侧（只读，已存在的工具）：`scripts/data/audit_prostate_anatomy_labels.py`，核对 WG /
    zonal / lesion 四类标签的取值分布与几何；既有报告
    `outputs/reports/prostate_anatomy_labels_audit_v2.json`（`inputs` stage：1500 例中 1499 例有 WG，
    1 例为已知缺失；`source_audit_status = unresolved`，35 例"无可比较候选"）。
  - 预测侧：对既有 223 例 `validation/*.npz` 逐区域复算 Dice / TP / FP / FN 与
    WG 概率的分布（`zonal_reliability_fusion.anatomy.validation.anatomy_region_metrics` 与
    `prior_uncertainty_report`），确认是"概率整体偏低"还是"阈值化后才为空"。
- [ ] 根据诊断结论决定修正方向（数据修正、region 定义核对、损失/采样调整），**一次只改一件事**。
- [ ] 修正后重训 Stage-1（新 Trainer 类名 + 独立输出目录，**不覆盖本次产物**），
  并为训练 split 生成先验（`workdir/anatomy_priors/...`）。
- [ ] 只有 WG 达到可用水平后，才启动条件 B。

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
