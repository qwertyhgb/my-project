# Method

工作机制：**anatomy- and lesion-conditioned local multimodal fusion**。
方法与实现边界见本文；假设见 Research_Plan；指标唯一定义见 Evaluation_Protocol；结果见 Training_Log。

## 1. Stage-1 Soft Anatomy

复用 Dataset607 单 T2W 模型，三 sigmoid heads：WG、PZ、TZ。
位编码 reference 为 WG+2*PZ+4*TZ；soft probabilities 的通道顺序为 WG/PZ/TZ。
原生硬导出按 [1,2,4] 顺序阈值写入，后写区域覆盖前写区域；不是 argmax，也不是位编码重建。
因此独立 head 质量使用 `anatomy.validation.soft_head_metrics`，不能从硬整数图恢复 WG head。
诊断首先冻结 `>0.5`，复用原生严格大于阈值；ROI 的 `>=` 是独立既有规则。
输出独立 Dice/recall/precision/TP/FP/FN、预测/参考体积比、每病例概率分位数与硬导出重建差异。
病例分位数均值不等于 pooled voxel quantiles。诊断不写入原产物，不修改模型或数据。

Fail closed：概率浮点、有限、[0,1]、三通道、形状与物理元数据一致；reference 合法位编码；
病例重复或任一读取/几何失败即失败，不静默排除。

## 2. Predicted-Prior Input Pipeline

选用独立 `Dataset608_PICAI_PredictedAnatomy`，不做 runtime dataloader injection，不覆盖 606。
六通道顺序：T2W/ADC/HBV/P(WG)/P(PZ)/P(TZ)。先验通道 `noNorm`。
`anatomy.dataset.materialize_predicted_dataset` 只供研究者运行，链接既有 MRI/lesion label，
读取该病例 Stage-1 soft 导出，校验几何，按原网格写先验，不自行重采样。
输出目录必须全新；失败保留部分产物并明确 INCOMPLETE，不自动清理或覆盖。

`predicted_anatomy_contract` 逐病例记录 case_id、checkpoint、trainer、source split、prior_mode、
probability channel order 与 geometry，并记录 anatomy-training patient IDs。
IN_SAMPLE_PRED、OOF_PRED、HELD_OUT_PRED、EXTERNAL_PRED 不混淆；验证只接受 HELD_OUT_PRED，
且该患者不在 anatomy-training scope。当前物化器只支持同一冻结模型的 in-sample/held-out预测；
契约支持 OOF，不表示已经实现 cross-fitting 生成器。训练集已知解剖排除病例可为 HELD_OUT_PRED。

Fail closed：只接受 Dataset608，拒绝旧 606 或 ORACLE_GT；病例集合必须完整、患者无泄漏；
mode 必须与 anatomy-training scope 一致；MRI、label、prior 同网格，缺文件直接失败。
输入先验文件夹必须由受信任的冻结 Stage-1 pipeline 产生；声明的 provenance 不是密码学来源证明。

## 3. Native Preprocessing Wrapper

`nnunet.prior_preprocessor.PredictedAnatomyPreprocessor` 继承 DefaultPreprocessor。
先对前三 MRI 通道调用原生 run_case_npy，保留原生 crop/normalization/resampling/class_locations；
再把先验按相同 transpose/crop 对齐，用原生线性 probability interpolation 到相同输出网格。
这避免 soft prior 改变 MRI nonzero crop，或 cubic data interpolation 把先验变成越界概率。
强度增强只作用于 MRI；空间变换同步全部通道。无第二套训练/推理框架。

`prepare_frozen_plans` 从 605 复制架构、patch/batch/spacing/scheduler 对应配置，不重新选网络；
只加三个 noNorm 与项目 preprocessor，保留明确的源 plans 路径。
复用 MRI fingerprint 并写来源说明，不声称已对 608 做新 fingerprint 审计。
preprocessor 在进程内注册，训练/预测通过项目入口；不改第三方源码。
任何已有目标预处理目录直接拒绝，避免原生覆盖行为。真实物化/预处理仍由用户运行。

## 4. Minimal Fusion Network

实现：`multimodal/conditioned_fusion.py`，不 import legacy。

```text
T2W ─ independent shallow stem ─ H_T2 ─┐
ADC ─ independent shallow stem ─ H_ADC ├─ concat ─ 1x1 projection ─ F0 (3 channels)
HBV ─ independent shallow stem ─ H_HBV┘                         │
                                                                ├─ pool2 ─ coarse head ─ L
H_T2/H_ADC/H_HBV + L (+ soft anatomy A in D) ─ controller ─ softmax weights
weighted shallow features + L (+ A) ─ zero-init projection ─ delta (3 channels)
F0 + delta ─ unchanged native PlainConvUNet ─ fine lesion segmentation
```

三个 stem 各为两层 Conv3d(3³)+InstanceNorm+LeakyReLU，不共享参数。
stem_channels=8，neutral projection 24→3；coarse 分支在 F0 上先做 factor-two average pooling。
controller 两层 1³ 卷积，输出 [B,3,D,H,W]。
系数按 `weights = (controller_logits / SOFTMAX_TEMPERATURE).softmax(dim=1)` 计算：
温度在**前向里显式出现**，因此代码、run_config 与文档对该常量的语义一致；
`SOFTMAX_TEMPERATURE` 冻结为 1.0（不设 CLI 超参数、不做温度搜索），
T=1 时与直接 `logits.softmax(dim=1)` **逐值相同**（有单元测试）。
D 的 context 为 [WG,PZ,TZ,U]，U=1-max(PZ,TZ)，只是 uncertainty-like 数值通道。
residual 最后 1³ projection 零初始化，初始 delta=0。

零初始化时 controller 从分割主损失接收的梯度可为零，这是设计结果；aux head 由辅助监督学习。
测试区分初始恒等与 residual 权重非零后的梯度路径，不谎称初始所有分支均有非零梯度。
C/D 永远保留 neutral path，不以 L 或 anatomy 二值阈值硬乘输入、特征或裁剪。
D 只接收预测概率，输入必须有限且在 [0,1]；不能仅靠数值断言区分合法全零背景 patch 与硬掩膜来源，
因此来源由数据契约保证。张量 shape 不能代替物理几何检查，后者发生在物化与预处理。

热路径检查的代价已在真实 patch 形状（`[1,6,16,320,320]`，RTX 3090）上用合成张量测量：
有限性/范围的 reduce 与 bool 同步约 **0.350–0.351 ms/patch**，约为 fusion 前端（约 26.0 ms/patch）
的 **1.35%**。因此**保留**逐 patch 检查，不把 full-tensor 检查挪到仅在物化/预处理阶段：
没有证据支持用失去 fail-closed 换这点开销。边界：合成张量、单卡、非 profiler trace，
不含 backbone 前向，未测 FLOPs/显存，不构成训练吞吐声明。

## 5. Loss and Native Integration

复用 `LesionnessAuxiliaryLoss` 与物理膨胀 target，radius=3 mm、weight=0.5 保持冻结。
coarse target 从全分辨率训练 lesion GT 膨胀，再 max-pool；各向异性 spacing 来自 plans。
空 lesion 是合法阴性；非法标签/缺失 auxiliary logits/不可整除网格直接报错。
推理不需要 GT，网络只返回原生分割 tensor/DS list。

新 B/C/D 分别提供 FLCE 与 DiceCE 类，方法 mixin 与显式 baseline 继承组合，不复制训练器。
同一 A/B/C/D 另有 100 epoch 短预算类（类名插入 `_100ep`，独立输出目录）：短预算类只把
`num_epochs` 在 `initialize()` 中同步为 `SHORT_BUDGET_EPOCHS`，从而让训练循环上界与
PolyLR 总周期一致；其余机制完全不变，正式 1000 epoch 类不被修改。
train loop、optimizer、PolyLR、checkpoint、validation、sliding-window/export 全部原生。
辅助 logits 暂用同进程属性通道，prototype 限单进程，禁 DDP；项目 fusion Trainer 禁 compile，
避免编译包装隐藏该属性。A1/A2 正式对照需同样冻结 compile 设置。
原生 deep supervision 开关通过 decoder 代理保留；DS 关闭不会改变 pre-fusion coarse head。

## 6. Fusion Export and Behaviour Analysis

默认不保存系数。显式 `export_fusion_weights=True` 只保留最近一次 forward 的 detached CPU patch 系数。
`fusion_weight_artifact` 是 offline hook；必须提供病例/checkpoint/geometry provenance。
不是全体积滑窗聚合器；未实现、未声称具有全体积权重导出。
`fusion_weight_statistics` 对调用方给出的区域系数子集输出 mean/std/median/IQR/entropy。
区域掩膜与物理几何由调用方核对；GT lesion 仅用于离线分析。系数不作因果解释。
`parameter_counts` 报 total/trainable/delta/delta%；未测 FLOPs/GPU memory 不写成测量结果。

## 7. Single-Variable Boundaries

| 比较 | 唯一干预 |
|---|---|
| A2 vs A1 | baseline loss，其他设置与 seed 匹配 |
| B vs A | independent shallow stems + neutral projection |
| C vs B | pre-fusion lesionness + supervised conditioned residual branch（整体机制） |
| D vs C | predicted WG/PZ/TZ/U anatomy context，MRI 主路径相同 |
| E vs best(C,D) | 困难负样本采样，最终父模型先冻结 |

C vs B 不单独识别 auxiliary supervision 与 coefficients 的各自贡献；必要时做 auxiliary-only 消融。
ROI 不混入默认 B/C/D。原 `lesion_roi/coarse_to_fine/zone_refine/hard_negative` 是 SUPPORTING，
保留原类名与输出目录，不重新命名为新 B/C/D；旧 gate 为 LEGACY 只读。
旧 E 固定继承旧 C，不代表新 E 已落实 best(C,D)。新 E 仍为计划。

## 8. Safety and Supporting Assets

修复原 B/C classmethod/instance helper 绑定；辅助采样组合修复困难负样本 loader 与 ROI 的参数/队列。
原 logits refinement 可作 supporting ablation。旧 ROI builder 的 original→preprocessed bbox 坐标链
尚待修复/验证，不能将该 supporting 流程称为就绪或加入正式模型。
不删除或覆盖数据/checkpoint；所有新训练从 scripts/train/train_nnunet.py 进入，使用独立类名。
legacy 实现与 third_party 只读。不给多个模块同时变化的结果作单变量归因。
