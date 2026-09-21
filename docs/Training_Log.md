# Training Log

训练与验证的运行事实日志。只记录真实运行（命令、时间、状态、关键结果、产物路径），不含协议/门控叙述。
代码变更见 `docs/Development_Log.md`。所有长任务由研究者本人运行；本文件在每次运行后更新。

---

## N0 — baseline（已完成）

- variant / Trainer：`baseline` → `nnUNetTrainerPICAI_FLCE_NoFFT`
- 数据集 / 配置 / fold：Dataset605_PICAI / `3d_fullres` / fold 0
- 输入通道顺序：T2W、ADC、HBV（3 通道）
- 损失：PI-CAI 官方风格 `0.5*Focal(gamma=2) + 0.5*CE`；NoFFT 修复（blur benchmark 关闭）
- 训练：1000 epoch 已完成；`checkpoint_final.pth` 已生成
- 验证：nnU-Net `perform_actual_validation` 完成，223 例
- **nnU-Net Mean Validation Dice = 0.1682**（`validation/summary.json` 的 `foreground_mean.Dice`
  = 0.16816659，223 例）
- validation probabilities：**未导出**（如需，重跑 validation 时加 `--export-validation-probabilities`）
- 输出目录（勿删除/覆盖）：
  `outputs/nnUNet_results/Dataset605_PICAI/nnUNetTrainerPICAI_FLCE_NoFFT__nnUNetPlans__3d_fullres/fold_0/`
- 复现 / 续训 / 只验证 / 导出概率的命令见 README「训练与推理命令」。

---

## legacy M0 — 旧自写 Residual-Encoder 管线（已中断，仅历史保留）

- 性质：**旧自写训练框架**（非 nnU-Net-first），本次重构后不再续训、不再用于任何结论。
- 启动：2026-09-20 09:32 UTC；由用户于 2026-09-21 停止。
- 最后完整记录：**epoch 389**（即完成 390/1000）。
- 最好一次完整验证：**epoch 349**
  - positive case-wise mean Dice = 0.2104
  - micro Dice = 0.5098
  - negative FP case rate = 31.25%
- checkpoint（UTC 时间戳，均在磁盘上核实）：
  - `checkpoint_last.pth` — 2026-09-21 01:24:05
  - `checkpoint_interrupt.pth` — 2026-09-21 01:24:50
  - `checkpoint_best.pth` — 2026-09-20 23:51:03（对应 epoch 349 附近）
  - `checkpoint_epoch_0349.pth` — 2026-09-20 23:51:06
- 产物路径（**不得删除或覆盖**）：
  - `outputs/checkpoints/m0_resenc/m0_resenc_picai_3d_fullres_v23/checkpoints/`
  - `outputs/metrics/m0/`
  - `outputs/diagnostics/m0/`
- 说明：驱动这些产物的旧代码（自写 trainer/checkpoint/滑窗/验证器/整网）已在本次重构中删除；
  产物本身按用户要求完整保留，仅供历史查阅。Git 历史承担代码归档职责。

---

## 待运行 variant（状态）

命令统一见 README「训练与推理命令」；运行后按下方模板在此登记真实事实。

| variant | 数据集/配置/fold | 状态 | 输出目录 |
|---|---|---|---|
| image_gate | Dataset605 / 3d_fullres / 0 | 未运行 | `outputs/nnUNet_results/Dataset605_PICAI/nnUNetTrainerPICAI_ImageGate__nnUNetPlans__3d_fullres/fold_0/` |
| anatomy_gate | Dataset606 / 3d_fullres / 0 | 未运行（Dataset606 尚未物化/预处理） | `outputs/nnUNet_results/Dataset606_PICAI_Zonal/nnUNetTrainerPICAI_AnatomyGate__nnUNetPlans__3d_fullres/fold_0/` |

---

## 运行记录模板

```
### <variant> — <dataset>/<config>/fold <k>
- 命令：
- 开始 / 结束（UTC）：
- 状态：完成 / 中断于 epoch N / 失败
- 关键结果：Mean Validation Dice = ，（可选）positive case-wise mean Dice / micro Dice / negative FP rate
- 输出目录：
- 备注：
```
