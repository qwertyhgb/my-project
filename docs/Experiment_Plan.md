# Experiment Plan

本文件记录实验矩阵、预算、命令入口与 stop rules；不记录结果。科学假设见 Research_Plan，
实现契约见 Method，指标与统计口径唯一来源为 Evaluation_Protocol，真实运行见 Training_Log。
所有训练、物化、预处理、推理、评测由研究者运行；本轮仅做已授权的既有输出诊断和合成测试。

## 1. Experimental Tree

```text
A  Native early-fusion strong baseline (best matched A1/A2 loss)
└─ B  Sequence-specific neutral fusion
   └─ C  Coarse lesionness-conditioned residual fusion
      └─ D  Predicted anatomy + lesion-conditioned fusion
         └─ E  Hard-negative sampling on best(C,D), only after selection
```

ROI 不混入默认主树，作为独立 supporting training ablation。
原 `lesion_roi` 等名字保留原含义；不能把原 ROI B 的结果记成新 neutral-fusion B。

## 2. Strong Baseline Selection

A1=`positive_sampling`（FLCE+阳性采样），A2=`dicece_positive_sampling`（原生 DiceCE+相同阳性采样）。
唯一变量是 loss：相同 native architecture、数据/split、patch/batch、augmentation、optimizer、
LR schedule、epoch、DS、checkpoint、validation/inference、显式 seed。
正式选择优先 matched-seed；历史无 seed A1 只作历史参照，不能称为 matched-seed。

历史 A1 目录已存在，禁止原命令从头覆盖。入口新增 `--seeded-output` 时从基类派生固定
`_Seed<seed>` 类名，独立目录并可由 checkpoint trainer_name 解析；只给不同 JSON 名不隔离模型。
A1/A2 同 seed 各自运行，按 primary + failure-mode endpoints 综合判断，不能只选单个 Dice。
若证据接近，保留不确定性，不强行选“胜者”。B/C/D 提供两个显式 loss 家族，按选型使用同一家族。

### 2.1 Baseline selection rule（2026-10-08 冻结，先于任何 A1/A2 结果）

判据顺序与 tie-break 原则在**看到任何 A1/A2 结果之前**固定；禁止事后按“哪个指标更好”择指标。
所有比较必须 matched seed、同 fold、同预算、同 inference、同 evaluation，
指标定义与空值语义引用 `docs/Evaluation_Protocol.md`（不在本文重新定义）。

1. **Primary**：`positive_case_macro_dice`（全部 GT 阳性病例纳入，完全漏检记 0）。
   单项配对均值差 **CI95 不跨 0** 才算“有明确差异”；CI 跨 0 一律记为“未观察到明确差异”，
   **不得**写成等效、也不得写成无效。
2. **Secondary**（仅当 primary 差异很小或 CI95 跨 0 时进入，顺序固定）：
   `completely_missed_lesion_rate` / `completely_missed_case_rate` →
   `lesion_sensitivity_any_overlap` → `small_lesion_sensitivity_any_overlap`。
   理由：当前研究的核心 failure mode 是漏检，而不是整体体素重叠。
3. **Third level**（仅当 sensitivity 仍接近时进入）：
   `negative_cases_with_fp`、`fp_components_per_case`、`fp_voxel_volume_per_case`、
   `positive_voxel_precision`（防止用扩大预测换 recall）。
4. **若仍无法区分**（冻结的处理原则，必须先于结果确定）：
   优先选择 **native DiceCE（A2）**。允许写下的唯一理由是
   “standard nnU-Net loss 更容易作为审稿人可接受的 strong reference”；
   **不得**写成“DiceCE better”，除非有实验支持。该选择是**可解释性/可接受性**决策，
   不是性能结论，登记时必须与实验结果分开叙述。
   替代方案（两个 loss 家族都进入 B/C/D 短预算 screening）需要 **2× 训练与评估预算**
   （每个条件两套短预算 + 后续正式预算），只有在短预算 screening 明确显示 loss 家族
   影响方向时才启用；不默认启用，也不在结果出来后再决定。

选择结论与依据的**运行事实**登记在 `docs/Training_Log.md`；**证据与边界**登记在
`docs/Findings.md`。选定后 B/C/D/E 只使用同一 loss 家族，另一家族保留为可解析的类名。

## 3. Variant Matrix

| 条件 | FLCE variant | DiceCE variant | 唯一干预 |
|---|---|---|---|
| B | neutral_fusion_flce | neutral_fusion_dicece | independent shallow stems + neutral projection |
| C | lesion_fusion_flce | lesion_fusion_dicece | pre-fusion coarse lesionness + conditioned residual branch + auxiliary supervision |
| D | anatomy_lesion_fusion_flce | anatomy_lesion_fusion_dicece | predicted WG/PZ/TZ 与派生 U context |
| E | 待 best(C,D) 冻结后建立 | 同左 | 困难负样本采样 |

B/C 使用 Dataset605；D 使用独立 Dataset608，冻结 605 网络结构/patch/batch/spacing，MRI
预处理逐值一致性由合成测试核对，真实 608 产物仍需用户核查。
每次正式比较逐项核对：split 与顺序、loss家族、epoch、iterations、DS、采样、增强、optimizer、
PolyLR 与推理设置。不得直接以默认 planning 重新决定 D 网络后声称单变量。

## 4. Frozen First Version

stem_channels=8；lesionness radius=3 mm；aux loss weight=0.5；softmax temperature=1；residual zero-init。
不做初始超参数网格搜索。无 Transformer/Mamba/cross-attention/多完整 encoder。
参数量使用相同真实 plans 对 A/B/C/D 报 total/trainable/delta/%；未测 FLOPs 或 GPU memory 明确未测。

## 5. Predicted Anatomy Prerequisites

先确认 Stage-1 soft heads 的质量（运行事实见 Training_Log）；不能凭硬导出 WG Dice 决定重训。
训练先验和验证先验必须来自可追踪冻结模型，验证患者不在其训练范围。
先支持显式 IN_SAMPLE_PRED，再评估 OOF 成本；不得把 in-sample 写成 OOF。
Dataset608 物化、冻结 plans 准备、预处理命令由 README 给出；只采用 dataset 路径，不维护
第二套 runtime injection。缺病例/几何/metadata 直接失败，禁止静默排除。
Stage-1 缺 WG supervision 的已知排除病例仍需其自身 MRI 的预测，不能用空 prior 顶替。

**C vs D 单变量解释的前置条件**：`scripts/data/audit_dataset605_608_equivalence.py`（只读、
fail-closed）必须对真实 Dataset605 与 Dataset608 的前三个 MRI 通道、lesion 标签、病例集合/顺序
与 split 取得 **PASS**（raw 与 preprocessed 两层）。仅有 raw 层通过（`RAW_ONLY_PASS`）**不足以**
支撑“D 只增加 predicted anatomy context”。该工具只回答 MRI/lesion/split 是否保持不变，
**不**评价 prior 质量；若存在数值差异必须记录 `max |delta|` / affected voxels / 来源，不得默认相同。
工具已就绪但**尚未在真实 Dataset608 上运行**（608 未物化/未预处理）。

## 6. Architecture and Construction Checks

先做合成网络 shape、初始恒等、梯度、权重归一化、非法概率与 provenance 测试；
再经真实 Trainer resolve/__init__/initialize/network/loss/backward。
真实 plans smoke 使用较小但满足下采样约束的合成 patch，说明不是原始 full-size patch 显存测试。
不能用测试通过替代真实训练/validation，更不能用它证明方法有效。

## 7. Budget

正式预算 1000 epoch。筛选预算固定 **100 epoch**，且**只有这一个档位**
（`zonal_reliability_fusion.nnunet.trainers.SHORT_BUDGET_EPOCHS = 100`；不同时维护 50/100/150/300）。

短预算运行**只**用于启动期 sanity 与方向筛选（`short-budget runs are only for sanity and
directional screening`）：

- **不得**与历史 1000 epoch 模型比较优劣，也不得据此声称某条件更好；
- 短预算内部必须同预算、同 seed、同评估互比：`A-short`（baseline 胜出 loss 家族的 100ep 臂）
  vs `B-short` vs `C-short` vs `D-short`。**没有同预算 baseline 就没有公平参照**，
  禁止只跑 B/C/D 短臂后与历史长预算结果对比；
- 短预算结论只用于决定“该条件是否值得进入正式预算”，不进入论文结论。

实现边界（不构成第二套训练框架）：

- 每个 A/B/C/D 各有 1000 epoch 正式类与 100 epoch 筛选类；筛选类名 = 正式类名插入 `_100ep`
  （例 `nnUNetTrainerPICAI_NeutralFusion_FLCE_100ep_NoFFT`），因此输出目录、checkpoint、
  validation 产物天然隔离。禁止用命令行临时改 `num_epochs` 后复用正式输出目录；
- 短预算类只在 `initialize()` 里同步 `num_epochs`，使训练循环上界、PolyLR 总周期
  （`PolyLRScheduler.max_steps`）与 checkpoint `current_epoch` 上界三者一致；
  训练循环、optimizer、scheduler、DS、checkpoint、validation、滑窗推理全部仍为原生实现；
- 正式 1000 epoch 类**不被修改**（无类级预算声明，回退即 nnU-Net 原生 1000）；
- 入口的 checkpoint 预算校验按 Trainer 类自身声明取值（短预算 100，正式 1000），
  并对每个短预算变体校验 dataset/plans/configuration/fold 一致。

2026-10-08 状态：短预算 Trainer 与其测试**已完成**；**尚未启动任何 100ep 或 1000ep 训练**。
B/C/D exploratory training 需在 baseline 选型（§2.1）之后另行安排。

## 8. Final Confirmation

最终 baseline/proposed：same fold / budget / evaluation / 3 explicit seeds。
报告各 run、mean±std、同 seed 病例配对；每 seed 使用独立输出目录，不重复覆盖同类名目录。
病例 bootstrap 不覆盖 run-to-run variance，种子不保证 bitwise determinism。

## 9. Stop Rules

| 比较 | 判据 | 行动 |
|---|---|---|
| B vs A | 表征改动未显示可重复收益 | 先判断是否保留 stem；不立即增加深度 |
| C vs B | complete misses 未下降且 lesion sensitivity 未提高 | lesion-conditioned fusion 假设未获支持；先检查 coarse target、aux loss 与 conditioning 数据流，不立即加 attention |
| D vs C | primary 与对应覆盖/sensitivity/FP 未显示稳定收益 | 停止 anatomy-conditioned fusion 扩张，模型可停在 C |
| E vs best(C,D) | FP 未下降或 sensitivity 明显下降 | 记录为 negative/ablation，不强行纳入最终模型 |
| 任一 | split/provenance/geometry 失败 | 停止该运行，先修复 |

CI跨0表示无明确配对改善，不证明等效或无效。单run小幅提升不构成稳定方法贡献。
需要同时报告 failure-mode指标和代价，不进行结果后择指标。

## 10. Mechanism Analysis

C的干预包含 coarse监督与条件残差整体，不从 C vs B 单独断言 softmax weighting 的因果作用；
需要时加入 auxiliary-only 或 capacity-matched对照。
系数按 lesion/background、预测zone、物理体积分层报告分布/entropy，标记为 model-behaviour analysis。
现有 hook 只导出 patch 系数；全体积聚合/恢复尚未实现，不把最近一个滑窗当整例系数。

## 11. Run Configuration

入口记录 dataset/config/fold/variant/trainer/seed、baseline loss、sampling、epoch/patch/batch/spacing、
fusion mode与冻结常量、prior source。计划不等于运行；状态唯一维护于 Training_Log。
默认新 B/C/D 不带 ROI 或hard-negative；E在父模型选定后另行安排。

## 12. External Stress Test

Prostate158 在 architecture 与选型冻结后才使用，不因外部结果重新调整模型。
prior 必须由冻结 pipeline 生成；不得使用 GT anatomy 或人工修正 prior。
第三序列不等同 HBV 时明确 distribution-shift stress test。
