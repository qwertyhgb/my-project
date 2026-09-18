# N0：PI-CAI 官方 nnU-Net 基线的 v2 适配

## 结论

**术语（统一）**：`N0 = PI-CAI official-style nnU-Net v2 Focal+CE NoFFT`。本文下文的「N0」「NoFFT 版」
「Focal+CE 版」均指该定义；全项目文档使用同一名称（`docs/research_plan.md` §7.1/§10.1/§14 G1、
`docs/project_structure.md`、`docs/P2_M0_Implementation.md` §18、`README.md`）。

当前直接运行的 `nnUNetTrainer` **不是** PI-CAI 主办方公开的 nnU-Net
baseline：它使用 nnU-Net v2 默认的 Dice + Cross-Entropy。PI-CAI 官方公开
baseline 建立在 nnU-Net v1 上，明确将默认损失替换为：

$$
L = 0.5\,L_{\mathrm{Focal}}(\gamma=2,\ \alpha=\mathrm{None}) + 0.5\,L_{\mathrm{CE}}.
$$

其中 `alpha=None` 表示不额外设置类别权重。官方还采用患者无重叠的 5-fold
交叉验证，并集成五个模型。其公开训练模型使用 1295 个带 human-expert
ISUP >= 2 标注的 PI-CAI 扫描；本项目的 1500-study / 1277:223 固定 split
和补充 Pooch25 标签并不与它逐项相同。因此本实现应称为
**PI-CAI official-style Focal+CE v2 port**，不可宣称为逐位官方复现。

来源：

- [PI-CAI baseline 的 nnU-Net 文档](https://github.com/DIAGNijmegen/picai_baseline/blob/main/nnunet_baseline.md)
- [主办方指定 trainer](https://github.com/DIAGNijmegen/picai_baseline/blob/main/src/picai_baseline/nnunet/nnunet_baseline.py)
- [公开 Focal + CE 实现](https://github.com/DIAGNijmegen/picai_nnunet_gc_algorithm/blob/main/nnUNetTrainerV2_Loss_FL_and_CE.py)
- [公开 FocalLoss 默认参数](https://github.com/DIAGNijmegen/picai_nnunet_gc_algorithm/blob/main/nnUNetTrainerV2_focalLoss.py)

## 本项目适配的固定项

- 固定 nnU-Net：项目内 `third_party/nnUNet` v2.6.2，**不修改**第三方代码；
- 固定数据和计划：`Dataset605_PICAI`、已有 `3d_fullres` plan 和 frozen fold 0 split；
- 固定默认训练结构、前景过采样、增强、SGD + Nesterov、PolyLR、1,000 epoch；
- 唯一训练协议变化：把每个 deep-supervision 尺度的默认 Dice + CE 换为上述 Focal + CE；
- 首次 `nnUNetTrainerPICAI_FLCE` 运行在 FFT benchmark 中崩溃，其日志原样保留；修复版输出
  天然隔离在 `nnUNetTrainerPICAI_FLCE_NoFFT__nnUNetPlans__3d_fullres/fold_0/`，不会覆盖默认 trainer 或失败运行；
- 当前启动器仅支持单 GPU；多 GPU 必须另行实现和验证，不能静默使用。

### PyTorch 2.10 / FFT augmentation 兼容性

在本环境中，`torch=2.10.0+cu128`、`batchgenerators=0.25.1` 和
`fft-conv-pytorch=1.2.0` 的组合会在默认 `GaussianBlurTransform` 的
`benchmark=True` 路径中触发 `free(): corrupted unsorted chunks`，使后台增强
worker 退出。该错误在第一个训练 batch 前发生，与 Focal + CE 数值无关。

本适配 trainer 因此把该 transform 的 `benchmark` 设为 `False`：Gaussian blur
的发生概率（0.2）、sigma（0.5–1.0）、适用通道和其余增强均不变，只是不再试跑
不兼容的 `fft_conv_pytorch` 后端，固定为普通 PyTorch convolution。这是本环境的
稳定性修复，不是新的数据增强策略。默认 `nnUNetTrainer` 或第三方源码不会被修改。

## 状态与预实验排除（本节状态复核于 2026-09-18）

> 门状态、blocker 与执行顺序的唯一权威来源是 `docs/STATUS.md`（G1 行）；启动条件与唯一命令见
> `docs/runbooks/n0_training.md`。本节只陈述与本基线定义直接相关的运行事实。

- **NoFFT 版 N0 尚未在真实数据上启动**：`outputs/nnUNet_results/Dataset605_PICAI/` 下目前只有
  `nnUNetTrainer__nnUNetPlans__3d_fullres/`（预实验）与 `nnUNetTrainerPICAI_FLCE__nnUNetPlans__3d_fullres/`
  （首次崩溃 run）；`nnUNetTrainerPICAI_FLCE_NoFFT__nnUNetPlans__3d_fullres/` **尚不存在**。
- **NoFFT 训练器已实现并有合成测试覆盖**：`src/zonal_reliability_fusion/integrations/nnunet_picai_flce.py`
  的 `nnUNetTrainerPICAI_FLCE_NoFFT`（等价子类，仅独立输出身份），测试 `tests/unit/test_nnunet_picai_flce.py`
  （合成 CPU；断言 FFT benchmark 被关闭、DS 各尺度损失替换、多 GPU 拒绝）；启动器
  `scripts/train/train_n0_picai_flce.py`（单 GPU）。**该实现从未在真实数据上运行过**。
- 首次 `nnUNetTrainerPICAI_FLCE` 运行在 epoch 0 崩溃，日志原样保留于
  `outputs/nnUNet_results/Dataset605_PICAI/nnUNetTrainerPICAI_FLCE__nnUNetPlans__3d_fullres/fold_0/training_log_2026_9_15_02_17_05.txt`
  （日志在 `Epoch 0 / Current learning rate: 0.01` 之后中断，未产生 train_loss 或 checkpoint）。
- **已排除的预实验**：nnU-Net v2 默认 Dice+CE（`nnUNetTrainer`）在 2026-09-15 01:17 启动，日志在 Epoch 43
  开始后中断（未完成、无 `Training done`、无 `checkpoint_final.pth`），43 个 epoch 中 38 次 `Pseudo dice = 0`。
  其输出、日志与 checkpoint **必须保留、不得删除或覆盖**，且**不得**写成正式失败性能实验，**不得**补写任何
  全体积 Dice / AP / AUROC 指标。
- 代码现状：FFT benchmark 的关闭同时作用于 `nnUNetTrainerPICAI_FLCE`（父类实现）与
  `nnUNetTrainerPICAI_FLCE_NoFFT`（等价子类，仅为独立输出身份，用于与失败 run 的目录隔离）。

## 运行前检查与训练命令（由研究者执行）

> **v2.2 协议覆盖说明（2026-09-15）**：正式 N0 必须先通过 `docs/research_plan.md` 的 G0-R，并确认序列错位决策
> 不改变当前预处理身份；还必须先解决 G0-E，冻结使用同任务外部 test，或在无合格外部集时先重划 PI-CAI internal test。
> 在这两项完成前运行，下列命令的产物只能标记为 feasibility run；若 G0-R 改变预处理或 G0-E 导致 split 重划，
> 该产物不得转用为正式 N0。G0-SAP 还须在查看正式论文指标前通过。
> 三项门的协议草案（状态均为 DRAFT/PENDING，未冻结）：`docs/protocols/G0_R_ALIGNMENT_QC.md`、
> `docs/protocols/G0_E_INDEPENDENT_TEST.md`、`docs/protocols/G0_SAP.md`；机器可读配置见 `configs/protocols/`。
> 代码迁移（如 M0 低频验证）或合成测试通过**不得**作为任何门的 PASS 依据。

请先确保当前的默认 trainer 进程已按你的决定结束，且目标 GPU 空闲。下面命令会启动新的
真实训练，代理不得代为执行：

```bash
cd /opt/data/private/lm/my-projects
conda activate lm
source scripts/env_nnunet.sh
CUDA_VISIBLE_DEVICES=0 python scripts/train/train_n0_picai_flce.py 605 3d_fullres 0
```

预期输出：

```text
outputs/nnUNet_results/Dataset605_PICAI/
  nnUNetTrainerPICAI_FLCE_NoFFT__nnUNetPlans__3d_fullres/fold_0/
```

请监测该目录中的 `training_log_*.txt`、`progress.png` 与 checkpoint。该训练沿用
nnU-Net epoch 日志；目标是确认 `Pseudo dice` 不再像默认 Dice+CE 基线那样在 100 epoch
内持续为 0。它仍只是 online patch 指标，正式结论需要完成后以 frozen validation 和
PI-CAI detection-map 评测协议确认。

若 **NoFFT 修复版** 实际完成至少一个 epoch 后中断，才可在同一输出目录续训：

```bash
cd /opt/data/private/lm/my-projects
conda activate lm
source scripts/env_nnunet.sh
CUDA_VISIBLE_DEVICES=0 python scripts/train/train_n0_picai_flce.py 605 3d_fullres 0 --continue-training
```

## 测试

以下仅使用合成 tensor，未读取医学影像、未启动训练：

```bash
cd /opt/data/private/lm/my-projects
conda activate lm
PYTHONPATH="$PWD/third_party/nnUNet:$PWD/src" CUDA_VISIBLE_DEVICES="" \
  python -m pytest -p no:cacheprovider tests/unit/test_nnunet_picai_flce.py -q
```

成功判据：所有测试通过，且不初始化 CUDA。`--help` 也可用于确认启动器参数装配：

```bash
cd /opt/data/private/lm/my-projects
conda activate lm
source scripts/env_nnunet.sh
python scripts/train/train_n0_picai_flce.py --help
```
