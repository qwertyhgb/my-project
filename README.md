# V2 — Anatomy-Guided Lesion-Aware Coarse-to-Fine Multimodal Prostate Cancer Segmentation

**ACTIVE MAINLINE: VERSION=V2**。工作标题，不提前冻结 acronym。
研究预测 WG/PZ/TZ 是否能指导 T2W/ADC/HBV 的层次化融合，
减少漏检、改善覆盖，同时控制假阳性。当前先完成融合问题的四组核心对照；
粗病灶监督与局部细化保留为候选扩展，暂不排期。

V1 — Shallow Anatomy/Lesion-Conditioned Fusion 已冻结，见 [docs/v1](docs/v1/README.md)。
更早 gate 实验为 [LEGACY](docs/archive/README.md)，与 V1 分开保留。
包名 `zonal_reliability_fusion` 和全部历史 Trainer/import/checkpoint 身份不变。

## V2 pipeline and documentation

```text
T2W / ADC / HBV -> three full native modality encoders -> hierarchical MRI features
predicted soft WG/PZ/TZ -> anatomy pyramid -> selected deep adaptive fusion policy
all-scale concat / conv / norm / activation -> fused skips -> native shared decoder -> base lesion prediction
fused bottleneck -> coarse lesion evidence -> soft local candidate
local MRI features + anatomy + base + candidate -> zero-init residual refiner -> final logits
```

上图为完整候选框架；当前核心研究只启用融合与native shared decoder部分。
Anatomy tells HOW to fuse；lesion evidence tells WHERE to refine。
V2复用共享early-fusion强基线，逐组验证concat卷积融合、image-adaptive调节、anatomy-conditioned调节。
当前命令为`v2_concat_fusion / v2_concat_adaptive_fusion / v2_concat_anatomy_fusion`；
三者共享所有尺度的concat卷积投影，旧均值变体保留原义但不排期。机制只见V2 Method。
**尚无已完成的V2正式性能结果**；实际运行进展只见Training Log，合成检查不证明分割性能。
variant 当前状态唯一来源：[Training Log「各 variant 状态」](docs/Training_Log.md#各-variant-状态)。

| 文档 | 单一职责 |
|---|---|
| [V2 Research Plan](docs/v2/Research_Plan.md) | RQ、假设、候选贡献、科学边界 |
| [V2 Method](docs/v2/Method.md) | 实现、fail-closed契约、schedule |
| [V2 Experiment Plan](docs/v2/Experiment_Plan.md) | 控制矩阵、预算、用户运行命令、stop rules |
| [V2 Architecture](docs/v2/Architecture.md) | 图、tensor shapes、V1/V2与作者参考差异 |
| [V2 Reproducibility](docs/v2/Reproducibility.md) | 来源、checkpoint、环境与复现边界 |
| [Shared Evaluation Protocol](docs/Evaluation_Protocol.md) | 指标定义/空值/匹配/统计唯一来源 |
| [Shared Training Log](docs/Training_Log.md) | 真实运行事实与唯一状态表，版本显式 |
| [Versioned Findings](docs/Findings.md) | V1/V2证据入口，避免结果混淆 |
| [Development Log](docs/Development_Log.md) | 代码/架构变更与实际工程检查 |

## Environment and safety

```bash
cd /opt/data/private/lm/my-projects
source /root/anaconda3/etc/profile.d/conda.sh
conda activate lm
source scripts/env_nnunet.sh
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
```

只用 conda lm / official nnUNet v2.6.2源码；third_party只读。
现有DNA=0.3.1与上游声明范围冲突，本轮未改依赖，验证限PlainConvUNet路径（Method §9）。
training/validation/inference/preprocessing/materialization/evaluation 长任务由用户运行，代理不启动。
原始医学数据只读，data/workdir/outputs/checkpoint 不删除不覆盖，规则见 [AGENTS](AGENTS.md)。

## Data and prior inputs

Dataset605：三MRI lesion；Dataset606：历史zonal输入，保留；Dataset607：共享单T2W Stage-1；
Dataset608：既有六通道 predicted anatomy lesion 输入，已物化/预处理及equivalence audit通过，
本轮复用，不重生。具体证据与实际状态以 Training Log 为准；不维护第二份 variant 清单。
Prostate158 只在架构冻结后做外部 stress test。

正式 lesion anatomy 输入只允许患者自身 MRI 的 predicted soft WG/PZ/TZ。
保留 IN_SAMPLE_PRED / OOF_PRED / HELD_OUT_PRED / EXTERNAL_PRED、checkpoint、split与geometry；
GT anatomy只用于明确ORACLE_GT上界，不能作为正式推理输入。
Stage-1不读lesion GT；hard WG export与soft head质量必须分开解释。

## Read-only help and synthetic checks

```bash
python scripts/train/train_nnunet.py --help
python scripts/train/train_nnunet.py --help --v1
python scripts/train/train_nnunet.py --help --supporting
python scripts/train/train_nnunet.py --help --legacy
python -B -m pytest -p no:cacheprovider
python -m ruff check --no-cache src scripts tests
```

默认只显示 `v2_` 条件；旧V1命令增加 `--v1` 或明确 `v1_` 前缀，历史Trainer类名/输出不改。
V2正式入口要求 `--seed --seeded-output --run-config`，配置含 research_version/architecture/fusion/
anatomy/coarse/refiner/phase；已有路径拒绝覆盖。

当前研究收敛为预测解剖先验对多模态融合的作用，四组核心对照逐组推进；粗监督、细化和阶段训练暂不排期。
early-fusion对照复用已完成的V1 A2（同方法配置，保留旧版本来源与路径），不为V2类名重复训练。
下一步由用户运行V2-B资源检查，检查后再进行三encoder与后续matched-budget对照。
完整可复制命令、预期输出、进度与成功判据见
[Experiment Plan §4–5](docs/v2/Experiment_Plan.md)。
全尺寸显存未通过前不直接启动三encoder长训练。
预测仍经 `scripts/inference/predict_nnunet.py` 的原生滑窗与项目resolver；
评价仍经既有脚本和共享 Evaluation_Protocol，不移植参考fork的评价或后处理。
评价主表为 `Dice_all / Precision / Recall / HD95 / NSD`：Dice_all纳入全部固定病例、空空=1；
其余四列仍为阳性病例口径，诊断表保留Dice_pos、漏检与阴性假阳。
独立新字段/配对统计及历史兼容说明见 [Evaluation Protocol](docs/Evaluation_Protocol.md)。
新增字段不回写既有报告，也不宣称已确认参考论文Table I的统计口径。

## Repository structure

```text
src/zonal_reliability_fusion/
  v1/                     explicit frozen namespace; implementation identity retained
  multimodal/             frozen V1 shallow implementation at historical import path
  lesion/                 V1 supporting assets / shared prior channel contract
  v2/
    encoders/             compose three plans-native encoders
    fusion/               concat卷积投影 / soft policy；保留mean控制
    anatomy/              predicted context pyramid
    lesion/               bottleneck evidence / local zero-init refiner
    models/               native-decoder composition
    losses/               coarse Dice+BCE wrapper
    utils/                tensor contracts / optional synthetic resource helper
  nnunet/v2/              independent V2 Trainers / phase hooks
  nnunet/                 registry, native bases, preprocessing/runtime/compatibility
  anatomy/ evaluation/ sampling/   shared infrastructure
  legacy/                 unchanged historical gate implementations
docs/v1/                  frozen V1 plans/method/findings; experiment original paths retained
docs/v2/                  active V2 plans/method/architecture/reproducibility/findings
docs/experiments/         frozen per-run V1 / shared Stage-1 historical details
docs/archive/             earlier LEGACY material, read-only
scripts/train/train_nnunet.py     sole version-aware training entry
scripts/inference/ scripts/data/  common native prediction / prior pipeline / audits
tests/unit/v2/            pure synthetic V2 tests; existing compatibility tests retained
third_party/nnUNet/       official runtime v2.6.2, read-only
third_party/Prostate-Lesion-Segmentation/  Align then Refine, read-only reference
```

参考设计阅读范围和差异见 V2 Architecture §12；论文与源码仅提供机制参考，
未复现作者实验，也不把论文指标作为本项目baseline。
[third_party/README](third_party/README.md)保留原有只读provenance；主repo只跟踪该说明，第三方源码不vendoring。


## Output isolation

V2新产物统一写入 `outputs/v2/`，历史V1/LEGACY与共享Stage-1路径不迁移，不改变provenance。

```text
outputs/
  v2/
    nnUNet_results/     V2训练checkpoint、validation与原生日志
    run_configs/        V2_运行配置
    predictions/        V2独立预测，每次全新run目录
    metrics/ reports/   V2评价与分析报告
    logs/ diagnostics/ figures/ external/ ablations/ inference_inputs/
  nnUNet_results/       既有V1/LEGACY；Dataset607为共享Stage-1
  predictions/          既有V1预测；D_stage1_final_soft_*为共享先验
  reports/              既有历史报告与共享608/解剖审计
  run_configs/ metrics/ diagnostics/ logs/ checkpoints/   历史产物保留
```

训练入口在构造V2 Trainer之前切换原生results globals至 `outputs/v2/nnUNet_results`，
不改变raw/preprocessed路径；V1/LEGACY入口沿用原路径。V2 run-config必须位于
`outputs/v2/run_configs/`。V2预测入口拒绝写入历史区及既有预测目录；
包含V2模型的评价必须显式版本标签，落盘只允许 `outputs/v2/metrics/` 或 `reports/`。
路径检查拒绝 `..` 与symlink越界，历史文件名前缀不会被误认为V2输出授权。
`outputs/v2/`整体gitignore；新运行产物由用户执行命令生成，状态只见Training Log。
