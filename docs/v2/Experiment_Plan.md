# VERSION=V2 — Experiment Plan

科学假设见 [Research Plan](Research_Plan.md)，实现契约见 [Method](Method.md)。
指标与冻结统计常量只引用 [Evaluation Protocol](../Evaluation_Protocol.md)。
真实训练状态只维护 [Training Log](../Training_Log.md)，本文件不写结果。

## 1. Controlled experiment tree

当前按用户要求控制为硕士研究规模：只研究predicted anatomy是否改善多模态融合。
必要实验仅A/B/C/D四组，A复用配置一致的V1 A2共享基线，B/C/D是新的V2模型；
逐组运行和检查，不同时排队，不新增融合模块或训练策略。
V2-E/F/F-SCHED保留代码与兼容身份，当前不排期、不作为毕业研究的必要实验。

所有类名前缀为 `nnUNetTrainerPICAI_V2_`，后缀为 `_DiceCE_NoFFT`。
共同 family：原生 DiceCE + 项目 positive-case sampling + NoFFT；验证原生 loader，
optimizer/PolyLR/DS/validation/sliding window 原生；不是作者 fork 的 TopK/Tversky 或采样设置。
选用此 family 依据已有 V1 选型，只用于固定训练条件，不能把 V1 性能转成 V2 结果。

| 条件 | CLI variant | Trainer 中段 | 与父条件唯一机制差异 | Dataset |
|---|---|---|---|---|
| A（共享对照） | 复用历史 `dicece_positive_sampling`；`v2_early_fusion` 保留但不重跑同seed | DiceCE_PositiveSampling_NoFFT_Seed20261008 | 同方法配置的strong early-fusion reference，保留VERSION=V1来源 | 605 |
| V2-B-concat | v2_concat_fusion | ConcatFusion | 三完整独立encoder＋所有尺度可学习concat/conv/norm/activation | 605 |
| V2-C-concat | v2_concat_adaptive_fusion | ConcatAdaptiveFusion | 保留B全部投影，深层增加image-only soft policy | 605 |
| V2-D-concat | v2_concat_anatomy_fusion | ConcatAnatomyFusion | 保留C投影/policy位置，只增加predicted anatomy context | 608 |

V2-B对共享A2同时改变capacity；记录参数与资源，不能称为等参数模态隔离实验。
旧v2_multi_encoder_simple/v2_adaptive_fusion/v2_anatomy_fusion类与别名保留原义，但不安排首轮训练。
C-concat对B-concat包含controller参数，D-concat对C-concat包含anatomy pyramid；记录容量变化。
三者相同all-scale投影、kernel/norm/activation与深层policy位置；policy初始等权，公式只见Method §2.1。
不同时改 patch/loss/optimizer/sampling。每条边独立分析，不使用 V1 B/C/D 作为 V2 condition 名。

## 2. Dataset and paired split

首版使用已有 Dataset605/608，fold 0 固定 split；shape/architecture/patch/batch/spacing 来自相同 plans。
605→608 必须有已有 equivalence audit 的 PASS（raw-only 不够），输入额外三通道仅为 predicted priors。
检查当前 audit JSON 与 provenance，不重新物化/预处理，不改变 split。
formal priors 只允许 predicted；ORACLE_GT 必须单列上界，不计入正式树。
训练 in-sample 与验证 held-out 不混称 OOF；外部评价先冻结模型与先验策略。

## 3. Budget and seeds

同seed的early-fusion基线已存在，复用V1 A2作为SHARED_A2对照，不为V2类名重复训练。
下一步由用户做V2-B synthetic full-patch资源检查，再逐次进行V2-B/C/D matched-budget/fold/seed对照。
基线Trainer与V2-A继承同一网络/loss/sampling/optimizer；V2-A仅额外身份、契约检查和日志。
保留旧模型与run-config路径，不能复制、更名为V2已训练产物；不能改用采样不同的native_dicece替代。
首版可执行 Trainer 总预算为原生 1000 epoch，无偷偷复用 V1 100ep 输出或 CLI epochs 参数。
一项 GPU smoke 最多 5 个 synthetic iterations，不表示已训练或已验证真实 dataloader。
第一轮 matched seed=20261008，最后候选重复 20261008/20261009/20261010；
完成第一组后再核定其余compute budget，不自动排队训练。
首轮fold0/单seed四组中A复用、B/C/D新训练，最终只为选定的强基线与候选模型补其余两个seed；
不为每个消融重复3seed，不开展全套5-fold、外部队列或额外机制，避免研究范围持续扩大。
checkpoint_final 为主比较；不能事后择 best/final 提高单个条件。
不同 checkpoint 或预算的比较只作明确 exploratory analysis。

## 4. Synthetic resource commands — user-run before V2-B

以下共用工作目录与环境：

```bash
cd /opt/data/private/lm/my-projects
source /root/anaconda3/etc/profile.d/conda.sh
conda activate lm
source scripts/env_nnunet.sh
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
```

先确认 GPU 空闲，不与既有 V1 训练争显存；用户可选择 GPU 编号。
真实 plans、完整 plans patch/batch、纯合成 forward/backward，不读取真实 MRI：

```bash
CUDA_VISIBLE_DEVICES=0 python -B -m zonal_reliability_fusion.v2.utils.smoke \
  --plans workdir/nnUNet_preprocessed/Dataset605_PICAI/nnUNetPlans.json \
  --trainer nnUNetTrainerPICAI_V2_ConcatFusion_DiceCE_NoFFT \
  --device cuda --full-patch --iterations 1
```

输出仅 stdout：tqdm 与 parameters / shape / batch / elapsed / peak_cuda_allocated_bytes。
成功判据：finite loss、backward 完成、ok=1、failed=skipped=0，无 OOM；
peak 仍不包含真实 dataloader、optimizer step/momentum，需留资源余量。
本轮代理不运行 GPU/full-patch helper。

## 5. Formal training templates — user-run only

当前第一步为§4的B资源检查。A使用以下既有基线目录：

```text
outputs/nnUNet_results/Dataset605_PICAI/
  nnUNetTrainerPICAI_DiceCE_PositiveSampling_NoFFT_Seed20261008__nnUNetPlans__3d_fullres/fold_0/
```

来源审计：保存的plans.json/dataset.json与当前605对应文件完全一致，验证病例ID集合与当前fold0相同；
旧run-config记录seed=20261008、1000epoch、patch[16,320,320]/batch2与阳性采样。
训练与验证真实状态仅见Training Log的V1 A2行，不在本文件重新登记为V2训练事实。

下面V2-A命令仅保留为其他明确需要新运行的模板；当前seed20261008不要重复执行：

```bash
nnUNet_compile=false CUDA_VISIBLE_DEVICES=0 python scripts/train/train_nnunet.py \
  v2_early_fusion 605 3d_fullres 0 --seed 20261008 --seeded-output \
  --run-config outputs/v2/run_configs/V2_A_early_fusion_fold0_seed20261008.json \
  --export-validation-probabilities
```

V2-B须在§4资源检查通过后由用户运行；一次只运行这一组：

```bash
nnUNet_compile=false CUDA_VISIBLE_DEVICES=0 python scripts/train/train_nnunet.py \
  v2_concat_fusion 605 3d_fullres 0 --seed 20261008 --seeded-output \
  --run-config outputs/v2/run_configs/V2_B_concat_fusion_fold0_seed20261008.json \
  --export-validation-probabilities
```

当前B输出目录：`outputs/v2/nnUNet_results/Dataset605_PICAI/nnUNetTrainerPICAI_V2_ConcatFusion_DiceCE_NoFFT_Seed20261008__nnUNetPlans__3d_fullres/fold_0/`。
入口拒绝既有模型目录与 run-config 覆盖；resume 必须相同 Trainer/seed/plans/dataset/fold。
终端显示 VERSION=V2 epoch当前/总量/比例/速率/ETA，以及原生train_loss / val_loss / Epoch time；可另开终端 `tail -f` 该目录的 training_log。
成功需日志 epoch0–999 完整、checkpoint_final、223 例 validation/summary 与概率输出完整，
入口结束额外打印success_runs/failed_runs/skipped_runs/elapsed/output_path；失败保留既有产物并核查，不自动重训。日志及计数由用户返回后核对。

后续C/D使用上表concat家族对应CLI和独立`V2_...` run-config，D使用608；
前一组结果检查后再决定。E/F/SCHED当前不排期。

## 6. Phase-scheduled condition

当前不排期；本节仅保留已有实现的配置边界，不属于四组核心研究的必要实验。

V2-F 为所有模块从 epoch0 启用的对照；V2-F-SCHED 的唯一变化：
P1 epoch0–199 mean/main；P2 epoch200–599 anatomy/coarse；P3 epoch600–999 加 refiner。
同总预算1000、相同原生SGD/PolyLR、所有参数预先进入 optimizer；不冻结或分组 LR。
这是预设候选 schedule，未经真实训练验证。低 LR refinement 或 warm ramp 只能后续独立变体。
resume phase 由 checkpoint current_epoch 重算；formal prediction 使用完整结构和 final checkpoint。

## 7. Evaluation execution boundary

沿用共享 evaluator，配对 fixed cases；full analysis 属用户运行的长任务。
参考命令（须将占位符替换为已经完成的目录及全新报告路径）：

```bash
python scripts/evaluate_segmentation.py \
  --model SHARED_A2=<既有V1_A2的fold_0目录> --model V2_B=<V2_B完成的fold_0目录> \
  --mode summary --output <全新outputs/v2/metrics/SHARED_A2_V2_B报告.json>
```

summary 输出进度/成功失败计数与报告路径；成功需要全部病例、零失败/跳过，
包含历史A2的比较明确为MIXED版本来源；共用模型对照不等于重新训练了V2-A。
full 模式须按共享协议显式冻结 volume thresholds 与 NSD tolerance，不能事后选阈值。
primary、关键 secondary、空值语义与 paired CI 不在本文件重定义。

## 8. Mechanism and reproducibility checks

先通过每步 synthetic construct/shape/forward/backward、parameter count、native Trainer constructor，
再确认 registry、CLI version、predicted provenance、第三方 runtime 隔离。
记录代码 commit、Trainer、plans、seed/fold、audit/provenance 与 final checkpoint。
正式 prior 误差分析与 patch coefficients 分布只做离线研究，不把 lesion GT 接入网络。
最终统计报告显式 VERSION=V2；V1 结果保持原有身份。

## 9. Stop rules — operational interpretation

进入下一机制前，primary paired CI、漏检/覆盖/FP 必须一起看。
若V2-B−共享A2的primary CI全低于0或出现非有限loss/无法容纳资源，停止扩大架构，
诊断 capacity、初始化、optimization与资源；不立即加 anatomy/refiner。
CI 跨0且 failure-mode 无支持时，只能写未观察明确增量，不能写等效或证明无效；
不因短预算噪声宣称训练间方差，最终候选用多seed核验。

若 V2-D 无重复支持的漏检/覆盖/FP增量，停止增加 anatomy complexity；
保留negative evidence，不因为D没有收益立即叠加coarse/refiner/schedule。
只有对至少一个预定义 failure mode 有证据、并报告其代价，才考虑纳入最终模型。
不恢复 LEGACY gates，不同时混入 hard negatives、uncertainty、boundary、text 或 foundation model。


## 10. Artifact destinations

V2所有新运行产物统一归 `outputs/v2/`；原 `outputs/nnUNet_results` 与平铺metrics/run_configs等
保留历史V1/LEGACY及共享Stage-1，不迁移，不重命名旧目录，不改既有provenance。
训练入口自动切换native results globals；raw/preprocessed仍共享原workdir。
run-config必须位于 `outputs/v2/run_configs/`，仅文件名V2_不足以通过路径检查。
预测用既有 `scripts/inference/predict_nnunet.py -m <V2模型目录> -i <输入目录> -o outputs/v2/predictions/<全新V2_run目录>`，
不在本轮运行；原生tqdm观察进度，成功需全部预期病例导出、零失败，目录已存在即拒绝。
V2评价标签如V2_A/V2_B；跨版本对照的历史标签明确V1_/LEGACY_/SHARED_。
评价JSON写入v2/metrics或v2/reports，新报告已有即拒绝；混合对照记录MIXED，不将V1结果写成V2性能。
共享Stage-1 checkpoint、soft priors与Dataset608审计继续引用原位置。
日志重定向、机制图与外部产物使用对应v2子目录；模型原生训练日志留在V2模型fold目录。
