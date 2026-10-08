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

## 6. Architecture and Construction Checks

先做合成网络 shape、初始恒等、梯度、权重归一化、非法概率与 provenance 测试；
再经真实 Trainer resolve/__init__/initialize/network/loss/backward。
真实 plans smoke 使用较小但满足下采样约束的合成 patch，说明不是原始 full-size patch 显存测试。
不能用测试通过替代真实训练/validation，更不能用它证明方法有效。

## 7. Budget

筛选预算 100/150 epoch，同一比较同 epoch、iterations、scheduler总周期、评估设置。
新增 active fusion Trainer 默认仍为原生 1000 epoch；**尚未提供独立短预算 Trainer**，
因此不能把当前默认命令称为 100ep 筛选命令。短预算实现后再安排 B/C/D exploratory training。
短预算不得与历史 1000ep 声称优劣；本轮不启动任何预算训练。

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
