# Runbook：G1 正式 N0 训练（PI-CAI official-style nnU-Net v2 Focal+CE NoFFT）

- **定义与依据**：`docs/N0_PICAI_Official_Baseline.md`、`docs/research_plan.md` §7.1 / §10.1 / §14 G1
- **状态与阻塞项**：见 `docs/STATUS.md`（本手册不维护状态）
- **执行人**：研究者（单 GPU，长时间训练）

## 0. 隔离与保留（硬约束）

| 目录 | 身份 | 处置 |
|---|---|---|
| `nnUNetTrainer__nnUNetPlans__3d_fullres/` | 已排除的默认 Dice+CE **预实验**（Epoch 43 中断，未完成） | **保留、不得删除或覆盖；不得写成正式失败实验；不得补写任何全体积 Dice/AP/AUROC** |
| `nnUNetTrainerPICAI_FLCE__nnUNetPlans__3d_fullres/` | 首次 FFT 崩溃 run（epoch 0 崩溃，日志原样保留） | 同上 |
| `nnUNetTrainerPICAI_FLCE_NoFFT__nnUNetPlans__3d_fullres/` | **正式 N0** 输出身份 | 由本 runbook 启动后创建 |

## 1. 启动前必须冻结的三件事

1. **G0-R 已通过**，且确认「仅重采样 / 额外固定刚体配准」的决策**不改变当前预处理身份**
   （若改变 → 本 run 不得作为正式 N0）；
2. **G0-E 已冻结独立测试路径**（路径 A 或 C）；
3. **N0 训练预算与 checkpoint 规则**（`docs/research_plan.md` §9.4 / §20.2 D7）：
   - 预算：N0 = 官方 1000 epoch（或改用官方自带变体 `nnUNetTrainer_250epochs`，官方无 200 epoch 变体）；
   - 推理与评测固定使用 `checkpoint_final.pth`，**禁用 `--val_best`**；
   - `Pseudo dice` 只是 online patch 指标，**不作为训练质量判据，也不作为 checkpoint 选择依据**。

任一未满足时运行，产物只能标记为 **feasibility run**。

## 2. 命令（复制执行）

```bash
cd /opt/data/private/lm/my-projects
conda activate lm
source scripts/env_nnunet.sh
CUDA_VISIBLE_DEVICES=0 python scripts/train/train_n0_picai_flce.py 605 3d_fullres 0
```

预期输出目录：

```text
outputs/nnUNet_results/Dataset605_PICAI/nnUNetTrainerPICAI_FLCE_NoFFT__nnUNetPlans__3d_fullres/fold_0/
```

若**已实际完成至少一个 epoch 后中断**，才可在同一目录续训：

```bash
cd /opt/data/private/lm/my-projects
conda activate lm
source scripts/env_nnunet.sh
CUDA_VISIBLE_DEVICES=0 python scripts/train/train_n0_picai_flce.py 605 3d_fullres 0 --continue-training
```

## 3. 进度观察

- `training_log_*.txt`（每 epoch 的 loss、`Pseudo dice`、LR）与 `progress.png`；
- 训练日志中确认划分：`workdir/nnUNet_preprocessed/Dataset605_PICAI/splits_final.json`（1277 train / 223 validation）；
- 采样协议（`debug.json`，2026-09-15 预实验实测）：`num_epochs=1000`、250 iter/epoch、batch=2、
  `patch_size=[16,320,320]`、`oversample_foreground_percent=0.33`、deep supervision 开启；
- 参考耗时（预实验同配置）：稳定后约 53 s/epoch（RTX 3090，显存约 5.9 GB）。

## 4. 成功判据（G1）

1. 训练正常结束，日志出现 `Training done`；
2. 产出 `checkpoint_final.pth`，无未处理 Error/Traceback；
3. 在 G0-SAP 冻结的评测管线中完成全量验证，并按 `picai_eval` 口径计算论文指标；
4. **不得**用 nnU-Net 自带 `validation/summary.json`（223 例 `nanmean`）与自研 M0 的阳性病例 Dice 直接比较。

## 5. 停止与失败处理

| 现象 | 处理 |
|---|---|
| 与预实验相同的 FFT/blur 崩溃 | 确认 `benchmark=False` 生效；保留日志、不得覆盖；不改第三方代码 |
| `Pseudo dice` 长期接近 0 | 记录事实，不改用 `checkpoint_best.pth` 作正式评测；按 `docs/research_plan.md` §14 G1 失败处理排查重采样/标签映射/采样/增强兼容性与推理恢复 |
| 训练中断 | 检查 `checkpoint_last.pth` 与 `epoch_completed` 语义后决定是否 `--continue-training`；不要删除既有目录 |
| G0-R/G0-E 之后发生预处理或 split 重划 | 该 run 不得沿用为正式 N0，只能标记 feasibility |

## 6. 记录

- 实际命令、启动/结束时间、日志路径、耗时、显存、结果 → `docs/experiment_log.md`；
- 若为正式 N0，更新 `docs/STATUS.md` 的 G1 行，并在论文材料中显式声明
  「PI-CAI official-style Focal+CE v2 port，**非**逐位官方复现」（官方为 nnU-Net v1、1295 标注扫描、5-fold CV）。
