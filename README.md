# Anatomy-Conditioned Multi-sequence Prostate Lesion Segmentation

前列腺癌（csPCa）病灶分割研究项目，采用 **nnU-Net-first** 架构。研究核心是在原生 nnU-Net
backbone 前加一个很小的 **spatial modality reliability gate**，并考察解剖分区（PZ/TZ）先验
是否能提升 T2W/ADC/HBV 多序列融合的可靠性。

## nnU-Net-first 架构

nnU-Net（固定版本 `third_party/nnUNet`，v2.6.2，commit `74ceb68`，只读）负责全部通用机制：
planning/preprocessing、dataloader、patch sampling、data augmentation、deep supervision、
optimizer 与学习率调度、training loop、checkpoint/resume、validation、sliding-window inference、
prediction export。

项目自身**只**保留：

1. PI-CAI Focal + CE 损失（`0.5*Focal(gamma=2) + 0.5*CE`）；
2. Gaussian blur 的 NoFFT 兼容修复（关闭不稳定的 FFT benchmark，blur 概率/sigma 不变）；
3. 一个基于原生 nnU-Net 网络的轻量 spatial modality reliability gate；
4. 必要的 PZ/TZ 输入适配（Dataset606 的附加通道与增强边界）；
5. 一个统一训练入口 `scripts/train/train_nnunet.py`；
6. 最少量的数据准备代码 `scripts/data/prepare_picai_nnunet.py`。

**不重复实现**第二套 trainer / checkpoint / 滑窗推理 / 验证器 / patch sampler / optimizer /
学习率调度器，也不复制 nnU-Net 的 U-Net encoder/decoder。

### 三个 variant

| variant | Trainer 类 | 数据集 | 输入通道 | 网络 |
|---|---|---|---|---|
| `baseline` | `nnUNetTrainerPICAI_FLCE_NoFFT` | Dataset605 | 3（T2W/ADC/HBV） | 原生 `PlainConvUNet` |
| `image_gate` | `nnUNetTrainerPICAI_ImageGate` | Dataset605 | 3（T2W/ADC/HBV） | gate(3→3 权重) + 原生 backbone |
| `anatomy_gate` | `nnUNetTrainerPICAI_AnatomyGate` | Dataset606 | 5（+PZ/TZ） | gate(MRI+PZ/TZ→3 权重) + 原生 backbone |

gate 结构：`Conv3d(in,hidden,1) → InstanceNorm3d → LeakyReLU → Conv3d(hidden,3,1)`，末层
weight/bias 零初始化；`weights = softmax(logits,1)`、`scales = 3*weights`、`gated = mri*scales`。
零初始化时 `scales` 恒为 1，新模型初始行为与原生 baseline **逐值一致**。anatomy_gate 只把加权后的
前 3 个 MRI 通道送入 backbone，PZ/TZ 不进入分割 backbone（禁止 WG），且进入 gate 前 clamp(0,1)。

三个 Trainer 类名互不相同，nnU-Net 据此生成互不覆盖的 output folder。

## 环境

固定使用 conda 环境 **`lm`**（Python：`/root/anaconda3/envs/lm/bin/python`）。涉及 nnU-Net 的
任何命令前必须：

```bash
cd /opt/data/private/lm/my-projects
conda activate lm
source scripts/env_nnunet.sh   # 校验 lm 环境 + 固定 nnU-Net 源码，并把 nnUNet_* 指向项目内 workdir/outputs
```

`scripts/env_nnunet.sh` 把 `nnUNet_raw`/`nnUNet_preprocessed` 指向 `workdir/`、`nnUNet_results`
指向 `outputs/`，并把 `PYTHONPATH` 前置到 `third_party/nnUNet:src`。

## 训练与推理命令

长任务（数据准备、planning/preprocessing、训练、validation、inference、evaluation）一律由研究者本人
运行。每条命令前先：

```bash
cd /opt/data/private/lm/my-projects && conda activate lm && source scripts/env_nnunet.sh
```

### 三个 variant 的训练

```bash
python scripts/train/train_nnunet.py baseline 605 3d_fullres 0        # = 已完成的 N0
python scripts/train/train_nnunet.py image_gate 605 3d_fullres 0
python scripts/train/train_nnunet.py anatomy_gate 606 3d_fullres 0    # 需先准备并预处理 Dataset606
```

可选参数：`--continue-training`、`--validation-only`、`--export-validation-probabilities`、
`--device {cuda,cpu,mps}`。入口只做 variant → Trainer 类映射后调用 nnU-Net 官方 `run_training`，
不含任何自写训练循环。

### baseline 只验证 / 导出概率

```bash
python scripts/train/train_nnunet.py baseline 605 3d_fullres 0 --validation-only
python scripts/train/train_nnunet.py baseline 605 3d_fullres 0 --validation-only --export-validation-probabilities
```

### 数据准备（anatomy_gate 前置；fail-closed）

```bash
python scripts/data/prepare_picai_nnunet.py baseline                   # 组织 Dataset605_PICAI raw
python scripts/data/prepare_picai_nnunet.py zonal --zonal-source yuan  # 组织 Dataset606_PICAI_Zonal raw（含 PZ/TZ）
nnUNetv2_plan_and_preprocess -d 606 --verify_dataset_integrity         # 官方 planning + preprocessing
python scripts/data/prepare_picai_nnunet.py splits --dataset-id 606 --dataset-name PICAI_Zonal  # 写入固定 fold
```

任一 failed/conflict 会先打印完整汇总再非零退出，且**不发布** `dataset.json`；`numTraining` 等于磁盘
实际完整病例数；`splits_final.json` 冲突非零退出。

### 推理（官方预测器 + 项目 Trainer 解析）

```bash
python scripts/inference/predict_nnunet.py \
    -i <输入影像目录> -o <输出目录> \
    -m outputs/nnUNet_results/Dataset605_PICAI/nnUNetTrainerPICAI_FLCE_NoFFT__nnUNetPlans__3d_fullres \
    -f 0 -device cuda
```

`-m` 换成对应 variant 的输出目录即可（image_gate / anatomy_gate）。该入口在进程内把 checkpoint 的
`trainer_name` 映射到项目 Trainer 类，随后调用 nnU-Net 官方 `nnUNetPredictor`；滑窗推理与导出全部
沿用官方实现，未自写推理器，未修改 `third_party/nnUNet`。其余参数与官方
`nnUNetv2_predict_from_modelfolder` 一致（`--help` 即官方帮助）。

## 输出目录

- baseline（N0，已完成，勿覆盖）：
  `outputs/nnUNet_results/Dataset605_PICAI/nnUNetTrainerPICAI_FLCE_NoFFT__nnUNetPlans__3d_fullres/fold_0/`
- image_gate：`outputs/nnUNet_results/Dataset605_PICAI/nnUNetTrainerPICAI_ImageGate__nnUNetPlans__3d_fullres/fold_0/`
- anatomy_gate：`outputs/nnUNet_results/Dataset606_PICAI_Zonal/nnUNetTrainerPICAI_AnatomyGate__nnUNetPlans__3d_fullres/fold_0/`
- legacy M0（历史保留，不再续训，勿删除/覆盖）：`outputs/checkpoints/m0_resenc/`、`outputs/metrics/m0/`、`outputs/diagnostics/m0/`

## 目录结构（最小）

```
AGENTS.md                协作规则
README.md                本文件
docs/
  Research_Plan.md       研究问题、假设、方法机制与研究边界
  Development_Log.md     代码/架构变更日志
  Training_Log.md        训练与验证事实日志
scripts/
  env_nnunet.sh                  nnU-Net 运行环境
  data/prepare_picai_nnunet.py   数据准备（baseline/zonal/splits，fail-closed）
  train/train_nnunet.py          统一训练入口
  inference/predict_nnunet.py    最小预测入口（官方预测器 + 项目 Trainer 解析）
src/zonal_reliability_fusion/
  nnunet/trainers.py     PI-CAI 损失 + NoFFT 修复 + 三个 Trainer + Trainer 注册表
  nnunet/networks.py     reliability gate 与原生 backbone 包装器
  nnunet/transforms.py   PZ/TZ 增强边界（强度只作用 MRI）
  nnunet/__init__.py     check_fixed_nnunet_runtime（nnU-Net/DNA/PlainConvUNet 运行时校验）
pytest.ini               只收集 tests/（不扫描 data/workdir/outputs/third_party）
tests/                   纯合成单元测试
third_party/nnUNet/      固定 v2.6.2（只读）
data/ workdir/ outputs/  数据、预处理缓存、训练产物
```
