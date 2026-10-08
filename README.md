# Anatomy-Guided Lesion-Aware Multimodal Coarse-to-Fine Fusion for Prostate Cancer Segmentation

**工作标题；方法与实验验证后再确定最终名称。**

本项目研究：预测前列腺解剖与粗病灶定位能否共同指导 T2W/ADC/HBV 的局部特征融合，
减少小型/困难病灶完全漏检、改善病灶覆盖，同时控制假阳性。

## Research and Pipeline

```text
T2W → frozen Stage-1 anatomy model → soft WG/PZ/TZ → anatomy context A
T2W/ADC/HBV → three shallow stems → neutral projection F0
F0 → coarse lesionness L
shallow features + L (+ A) → local softmax fusion → zero-init residual delta
F0 + delta → native PlainConvUNet → fine lesion segmentation
```

三个 failure modes 保持不变。融合服务于病灶定位与分割，不恢复独立的通道加权研究。

| 文档 | 职责 |
|---|---|
| [Research Plan](docs/Research_Plan.md) | 科学问题、假设、边界、待验证贡献 |
| [Method](docs/Method.md) | 数据流、实现、安全契约与限制 |
| [Experiment Plan](docs/Experiment_Plan.md) | 单变量实验树、预算、stop rules |
| [Evaluation Protocol](docs/Evaluation_Protocol.md) | 指标定义与统计唯一来源 |
| [Training Log](docs/Training_Log.md) | 真实运行与唯一 variant 状态表 |
| [Findings](docs/Findings.md) | 有证据支持的结论与边界 |
| [Development Log](docs/Development_Log.md) | 代码/架构变更与实际检查 |

主实验树：A strong baseline → B neutral fusion → C lesion-conditioned fusion →
D anatomy+lesion-conditioned fusion → E optional hard-negative strategy。
A1/A2 比较 FLCE 与原生 DiceCE，后续使用胜出 loss 家族。ROI 为 supporting ablation。
新融合骨架尚无训练效果证据；详细实现/构造/训练/validation 状态只在 Training_Log 维护。

## Stage-1 Diagnosis

2026-10-08 对既有 223 例验证输出做独立 soft-head 只读诊断。
**硬导出低 WG Dice 不能证明 soft WG 失败**；真实结果与完整口径见
[Stage-1 实验文档](docs/experiments/anatomy_joint_100ep.md)。当前没有据此重训 Stage 1。
region 硬导出按顺序覆盖重叠区域，不能把它当位编码 head reconstruction。

## Data and Prior Sources

| 数据集 | 角色 |
|---|---|
| Dataset605_PICAI | 3 MRI，1500 studies，train 1277 / val 223；baseline、B/C |
| Dataset606_PICAI_Zonal | 历史自动 zonal membership 通道；保留，不作为新 D 的 Stage-1 soft prior |
| Dataset607_PICAI_Anatomy | 单 T2W，1499 studies，train 1276 / val 223；算法解剖伪监督；显式排除已知缺失 WG 病例 |
| Dataset608_PICAI_PredictedAnatomy | 新 D 的独立 6 通道输入契约；**尚未真实物化或预处理** |
| Prostate158 | architecture 冻结后的 distribution-shift stress test |

正式 lesion 输入只允许 predicted anatomy；GT anatomy 仅显式 ORACLE_GT 上界分析。
先验记录 IN_SAMPLE_PRED/OOF_PRED/HELD_OUT_PRED/EXTERNAL_PRED 与 checkpoint、split、geometry。
当前未生成完整训练集先验，未实现 OOF 模型训练；不能把 metadata 支持写成 OOF 已完成。

## Environment and Safety

工作目录 `/opt/data/private/lm/my-projects`，只用 conda **lm**，nnU-Net 固定 v2.6.2，第三方源码只读。
所有下列命令先执行：

```bash
cd /opt/data/private/lm/my-projects
source /root/anaconda3/etc/profile.d/conda.sh
conda activate lm
source scripts/env_nnunet.sh
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

长任务由研究者本人运行，一次一条；不自动训练、推理、物化或预处理。
已有 data/workdir/outputs、checkpoint、validation、reports 不删除、不覆盖。
新建路径也必须经当前任务明确授权。完整规则见 [AGENTS.md](AGENTS.md)。

## Commands: Read-only Diagnosis and Tests

soft anatomy 全验证集只读诊断（约数分钟；逐例 tqdm；结果 stdout，不改产物）：

```bash
python -B -m zonal_reliability_fusion.anatomy.validation \
  outputs/nnUNet_results/Dataset607_PICAI_Anatomy/nnUNetTrainerPICAI_AnatomyJoint_100ep_NoFFT__nnUNetPlans__3d_fullres/fold_0/validation
```

成功判据：223 success、0 failed/skip；报告独立 heads 与硬导出重建差异。它是诊断，不做阈值选型。

纯合成测试（真实 Trainer 构造输出只在 pytest 临时目录，不读真实影像）：

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -B -m pytest -p no:cacheprovider
python -m ruff check --no-cache src scripts tests
```

真实 plans smoke 使用较小的合法 synthetic patch，不表示 full-size patch 显存或真实 dataloader 已验证。

## Commands: Training Templates (User-run Only)

```bash
python scripts/train/train_nnunet.py --help
python scripts/train/train_nnunet.py --help --supporting
python scripts/train/train_nnunet.py --help --legacy
```

A1 matched-seed 独立运行模板（保留历史 A1；默认 full 1000ep，**本轮未运行**）：

```bash
nnUNet_compile=false CUDA_VISIBLE_DEVICES=0 python scripts/train/train_nnunet.py \
  positive_sampling 605 3d_fullres 0 --seed 20261008 --seeded-output \
  --run-config outputs/run_configs/A1_positive_sampling_fold0_seed20261008.json
```

A2 必须另一次运行，使用相同 seed/settings：

```bash
nnUNet_compile=false CUDA_VISIBLE_DEVICES=0 python scripts/train/train_nnunet.py \
  dicece_positive_sampling 605 3d_fullres 0 --seed 20261008 --seeded-output \
  --run-config outputs/run_configs/A2_dicece_positive_sampling_fold0_seed20261008.json
```

输出由实际 Trainer 类名自然分离，例如 `...PositiveSampling_NoFFT_Seed20261008__nnUNetPlans__3d_fullres/fold_0/`。
run-config 已存在拒绝覆盖，已有模型目录拒绝从头重训；`--continue-training` 只用于未完成运行。
终端显示原生 epoch 进度；成功需 epoch 1000、final checkpoint 与 validation summary。

B/C/D variant 与 loss 家族映射见 Experiment_Plan。正式比较须先完成 baseline 选型，
短预算独立 Trainer 尚未实现，当前默认命令不能冒充 100ep 筛选。
`--seeded-output` 同样适用于 active fusion 类；不要只改 run-config 文件名后覆盖模型目录。
E 在 best(C,D) 冻结后单独建立；旧 supporting E 不能充当新 E。

## Commands: Independent Predicted-Prior Dataset (Not Run)

先验须覆盖完整 lesion split，包括已知 Stage-1 监督排除病例；验证只能来自 held-out 预测。
**以下模板不是当前下一步实验；完整先验尚未生成。**

1. 只在全新 Dataset608 路径物化，不改 Dataset605/606：

```bash
python -m zonal_reliability_fusion.anatomy.dataset \
  --source-raw workdir/nnUNet_raw/Dataset605_PICAI \
  --prior-dir <冻结Stage1对全部病例的soft导出目录> \
  --model-folder outputs/nnUNet_results/Dataset607_PICAI_Anatomy/nnUNetTrainerPICAI_AnatomyJoint_100ep_NoFFT__nnUNetPlans__3d_fullres \
  --output-raw workdir/nnUNet_raw/Dataset608_PICAI_PredictedAnatomy \
  --split-file workdir/nnUNet_preprocessed/Dataset605_PICAI/splits_final.json
```

2. 准备独立 frozen plans（保留 605 architecture/patch/batch/spacing）：

```bash
python -m zonal_reliability_fusion.nnunet.prior_preprocessor prepare \
  --source-preprocessed workdir/nnUNet_preprocessed/Dataset605_PICAI \
  --predicted-raw workdir/nnUNet_raw/Dataset608_PICAI_PredictedAnatomy \
  --target-preprocessed workdir/nnUNet_preprocessed/Dataset608_PICAI_PredictedAnatomy
```

3. 用户运行原生预处理包装（逐例进度；目标目录已有即拒绝）：

```bash
python -m zonal_reliability_fusion.nnunet.prior_preprocessor preprocess --processes 2
```

成功需完整病例数、零失败/跳过，独立 raw/preprocessed 路径、完整 provenance 与 split。
MRI 继续走原生 preprocessing；prior 按相同几何链线性插值，不改变 MRI crop。
失败保留 INCOMPLETE 目录，不能将部分病例结果作为完成；不提供自动删除/覆盖。

## Evaluation and Prediction

分割评价不改原协议，primary 为 positive-case macro Dice，漏检纳入分母；
lesion sensitivity 与 matched-lesion Dice 同时报告，必须报告 FP 代价。
具体定义见 Evaluation_Protocol。

```bash
python scripts/evaluate_segmentation.py \
  --model A1=<完成validation的fold_0目录> \
  --model A2=<完成validation的fold_0目录> \
  --mode summary --output <全新报告JSON路径>
```

full 模式读取真实 NIfTI，是长任务：额外使用 `--mode full`、显式 `--nsd-tolerance-mm` 与
`--volume-thresholds-mm3`；事先冻结参数，不事后挑阈值。不重采样或后处理预测。

预测继续使用 `scripts/inference/predict_nnunet.py` 原生滑窗入口。
Stage-1 必须 `--save_probabilities`。D 使用六通道同网格输入与项目 preprocessor；
不声称现有 patch-weight hook 能输出全体积权重。

## Repository Structure

```text
src/zonal_reliability_fusion/
  anatomy/       contracts, soft-head diagnosis, predicted dataset provenance/materializer
  multimodal/    new independent neutral/conditioned fusion, patch weights, parameter counts
  lesion/        frozen lesionness targets, supporting ROI/logits-refinement assets
  sampling/      positive sampling and supporting hard negatives
  evaluation/    frozen segmentation metrics
  nnunet/        native Trainer integration, prior preprocessing wrapper, registry
  legacy/        unchanged historical implementations
scripts/train/train_nnunet.py    only training entry
scripts/inference/predict_nnunet.py   native prediction entry + project resolvers
tests/unit/                     synthetic tests and true native constructor checks
third_party/nnUNet/              fixed v2.6.2, read-only
```

包名保留 `zonal_reliability_fusion` 以兼容历史产物。
旧路线只读归档：[archive](docs/archive/README.md)。旧输入级重加权在已测试运行中未显示一致增益，
不等于 gate 被证明无效；新条件化融合必须通过自身受控实验验证。
