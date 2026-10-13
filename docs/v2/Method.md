# VERSION=V2 — Method and Implementation Boundaries

V2 主线：Multi-Encoder Anatomy-Guided Coarse-to-Fine。研究论证见 [Research Plan](Research_Plan.md)，
数据流与 tensor 表见 [Architecture](Architecture.md)，预算/命令见 [Experiment Plan](Experiment_Plan.md)。
结果与状态不维护在本文件。

## 1. Native composition

`v2/encoders/multi_encoder.py` 通过固定 nnU-Net `get_network_from_plans` 独立构建三次单通道
PlainConvUNet，保留各自的 encoder 与第一份 decoder，其余临时 decoder 不进入模型。
没有复制 encoder/decoder 源码，没有 V1 shallow 24→3 compression。
普通 shared decoder、deep supervision 顺序、原生 segmentation heads 保持原样。
DNA 原生 decoder 持有第一个 encoder 引用；它与 `encoders[0]` 是同一对象，不重复计参数或 forward。
state dict 保留原生子模块命名，checkpoint 由同一个 V2 类严格重建。

Fail closed：只允许 plans-defined PlainConvUNet/Conv3d、binary two-head lesion、合法 3/6 通道、
有限 floating input、patch 整除累计 strides；不支持 cascade、2D、region labels、ignore label。
首版 Trainer 限 single process、fold 0、3d_fullres，不静默跳过 metadata 检查。

## 2. Hierarchical fusion

### 2.1 当前concat对照家族

`v2_concat_fusion / v2_concat_adaptive_fusion / v2_concat_anatomy_fusion`使用独立Trainer，
不改变旧均值/直接加权类的checkpoint结构或方法语义。三者所有尺度（当前0–6）均为：
concat三份C_s通道特征 → Conv3d(3C_s,C_s,3,padding=1) → plans-defined norm → plans-defined activation。
卷积bias、norm/activation及其kwargs从原生encoder读取；融合卷积按原生He初始化。
核固定3×3×3，decoder输入形状、原生网络plans、MRI encoder、采样、主损失、优化器与预算均不变。
该kernel参考作者FusionBlock，但本项目仍使用原生PlainConvEncoder/UNetDecoder，不声称论文复现。

B-concat没有controller，所有尺度直接可学习concat投影。
C-concat仅在原selected deep scales（当前4/5/6）增加图像policy；D-concat只再加入predicted anatomy context。
相同尺度的融合投影结构完全相同，没有把C/D的concat投影换成直接加权求和。

```text
alpha = softmax(controller(concat(MRI features [, anatomy context])), modality)
adjusted_m = 3 * alpha_m * F_m
fused = projection(concat(adjusted_T2, adjusted_ADC, adjusted_HBV))
```

controller最后一层weight/bias零初始化，alpha初始1/3，乘3后恢复B-concat的输入数值；
仅末层为零，首次反传可更新该层，不叠加零scalar gate。encoder/decoder/projection参数相同时
B/C/D初始预测相同。构造顺序为先全部共同encoder/decoder/projection，再可选policy/anatomy，
合成测试确认相同torch seed下共同参数逐值相同；不据此宣称真实训练bitwise确定性。
浅层三者都直接concat投影。D的anatomy仍只进入policy，不进入MRI encoder，不硬遮挡输出。
这三类从头全程启用，无coarse、refiner或phase schedule，网络拒绝切换到mean-only P1。

Fail closed：三份相同shape、floating 5D、plans-defined通道和finite features；
无policy的projection拒绝context，anatomy policy缺context即失败；D继承§3的全部先验契约。
输出为原生DS list/tensor；权重hook只输出真实存在的深层policy weights，不虚构B的注意力图。

### 2.2 保留的均值/直接加权家族（当前不排期）

完整 hierarchy 来自三独立原生 encoders。所有尺度的 skip 都进入 shared decoder。
V2-B 使用 `(F_T2+F_ADC+F_HBV)/3`，无 trainable fusion 参数。
V2-C 及之后在 `range(max(1,n_stages-3),n_stages)` 使用 1×1 controller：
concat features (+ anatomy context) → 16 channels → LeakyReLU → 3 logits → modality softmax。
浅层 skip 继续 mean。当前 7-stage plans 对应 zero-based scales 4/5/6（第 5/6/7 层，末层为 bottleneck）。

`alpha=softmax_m(G_s(...))`，`F_fused=sum_m(alpha_m F_m)`，每位置系数和为 1。
不硬编码区域模态偏好，不 hard gate，不删 neutral mean path；P1 全部使用 mean。
controller 最后层使用常规初始化，P1→P2 是显式机制切换而非保证数值连续的 warm ramp；
切换稳定性属于后续验证，不能预先宣称 schedule 更优。

Fail closed：必须三份相同 feature shape、finite features、正确 context shape；
image-adaptive 不接收 anatomy，anatomy-conditioned 缺 context 即失败。

## 3. Predicted anatomy contract

输入顺序 `[T2W,ADC,HBV,P(WG),P(PZ),P(TZ)]`。概率图可重叠，不要求 sum=1，
不从 ordered hard export 重建 soft WG。每个概率有限且在 [0,1]。
Anatomy pyramid 对每个 selected grid trilinear resize，再独立 1×1 projection 3→8；
context 只进入 fusion controller，refiner 使用原 soft maps，不作为 MRI encoder 的输入。

共用 `anatomy/dataset.py` 检查 Dataset608 channel order、predicted source、完整病例集合、
prior_mode、checkpoint、source_split、geometry 与 anatomy-training patients。
共用 `PredictedAnatomyPreprocessor` 保留同步几何与 noNorm；强度增强只作用前三 MRI 通道。
Trainer 再核对固定 split JSON 与 prior contract；case ID 非法、空/重复集合、patient overlap 即失败。
验证 loader 保持原生采样，不用 validation GT 生成 priors。

不能仅靠 tensor 判定概率来自 prediction：来源安全依赖上述 metadata 与已有物化/审计链。
正式 Trainer 拒绝 ORACLE_GT；上界分析需另立显式 variant，不在本次实现。
复用已有 Dataset608，不重生、不覆盖原审计与 Stage-1 产物。

## 4. Coarse evidence and loss

本节及§5–6仅适用于保留的原候选类，不叠加到当前concat三组。

fused bottleneck → Conv3d(C_L,1,1) → coarse logits。`sigmoid` 后作为 coarse probability。
training target 从 full-resolution binary lesion GT 通过 adaptive max pooling 到 bottleneck grid，
保留小阳性，不沿用 V1 的物理膨胀目标。推理不访问 target。

原生 main DiceCE/deep supervision 不改；auxiliary = BCEWithLogits + (1−soft Dice)，
`total=main+0.5*auxiliary`。P1 不计算 aux，P2/P3 启用。没有另加 loss family 或新监督分支。
logits 通过单进程 network side channel 交给 loss；每次 forward 清空并重建，不输出给 predictor。
compile 与 DDP 首版禁用，避免 side channel 与阶段分支语义不明。

Fail closed：target binary singleton channel、fresh coarse logits、finite auxiliary；
模型 forward 不读 GT，coarse head 不能用于 GT-guided inference。

## 5. Local late refinement

采用允许的等价 logit residual，原生 decoder 完整执行一次：
`candidate=detach(max(softmax(base)[:,lesion], up(sigmoid(coarse))))`。
三个最高分辨率 MRI features 各 1×1 投影为 8 channels，concat base logits、soft anatomy、candidate，
再局部 3×3 Conv → 8 → LeakyReLU → 1×1 Conv → 2-logit correction。
`final=base+candidate*delta_logits`；delta 的末层 weight/bias 为零，初始与 base 逐值相等。
零初始化采用单一设计，不再叠加零 scalar gate 以避免双零导致分支不学习。

candidate 为 soft spatial weight，没有 threshold、hard ROI、entropy、ensemble 或 boundary branch。
局部性来自卷积 receptive field 与 probability weighting；本版仍计算整张 patch refiner，
不是稀疏 ROI 加速，不宣称减少 compute。base/coarse 均低时可能无法补回完全漏检。
只有最高分辨率 DS logit 被修正，其余原生 DS 输出保持 base。
输出始终 tensor 或原生 DS list，兼容 predictor/sliding-window export。

## 6. Phase scheduling

`nnUNetTrainerPICAI_V2_FullScheduled_DiceCE_NoFFT`：总原生预算 1000 epoch，边界为代码常量 (200,600)。

| Epoch (zero-based) | Phase | 启用机制 |
|---|---|---|
| 0–199 | V2-P1 | three encoders + mean fusion + native decoder/main loss |
| 200–599 | V2-P2 | anatomy-conditioned fusion + coarse supervision；refiner bypass |
| 600–999 | V2-P3 | previous + local residual refiner |

所有模块在初始化时创建并进入原生 SGD；未启用模块无梯度，首次启用前 weight decay 不更新它们。
不 freeze/unfreeze、不重建 optimizer、不新增 LR groups；PolyLR 是同一原生 1000-epoch schedule。
因此 P3 的 refiner 也使用当时衰减 LR，不声称已获得最佳细化优化。
`on_epoch_start` 与 `load_checkpoint` 之后按恢复的 current_epoch 重新设置 phase；
后者同时覆盖 native validation-only（没有 epoch hook）。原生 checkpoint 保存/加载逻辑不复制。
类构建默认 final architecture，prediction 不依赖未保存的临时 Python phase 属性；
中期 checkpoint 做整例推理会按完整 architecture 执行，不能冒充当时的 P1/P2 行为，
首版正式比较统一 checkpoint_final。online validation 使用当前 phase 与原生 loader。

## 7. Capacity and resource limits

encoder/decoder的width、depth、kernels、strides及patch/batch完全来自plans；
concat新增投影核为§2.1明确的3×3×3常量，不修改原生网络plans。
`compute_conv_feature_map_size` 包含三 encoder、decoder 与新增 conv 输出；
仅为 conv feature-count，不含 concat/softmax 临时 tensor、梯度、optimizer，不等于 peak memory。
parameter count 去除共享引用重复。`v2.utils.smoke` 只生成 synthetic tensors，
输出参数、iteration time、可选 GPU peak allocated bytes；CPU 小 patch 不验证 full patch 显存。
helper 未执行 SGD step，peak 不涵盖持久 momentum 与真实增强 pipeline，不替代真实训练预算。

若 full-width 超预算，先讨论 width reduction / selected independent stages / partial sharing，
任何折中必须独立版本与配套对照，记录在本节；目前未采用任何折中，更不能回退成 V1 shallow compression。

## 8. Version and runtime safety

V2 网络只导入 shared infrastructure 与 `v2/`，不依赖 `legacy/` 或 V1 实现。
registry 为 checkpoint compatibility 保留历史类导入，这不等于 V2 调用其网络。
唯一真源 `nnunet/trainers.py` 分 ACTIVE_V2 / FROZEN_V1 / SUPPORTING / LEGACY，
所有历史类名与 seeded reconstruction 保留。训练入口默认 V2、旧 CLI 需 `--v1`/`v1_`。
V2 要求 seed、seeded-output、run-config；输出类名含 V2，不覆盖 V1。
第三方 nnUNet v2.6.2 与 Align then Refine 均只读，后者不进入 import path。


## 9. Existing dependency compatibility limit

实际 `lm` 中 dynamic-network-architectures=0.3.1，不满足 official nnUNet v2.6.2
声明的 >=0.4.1,<0.5。现有环境还保留另一editable nnUNet来源的依赖约束；
项目通过env_nnunet.sh锁定official源码，不能靠package版本号判断实际import。
本轮不升级/覆盖任何依赖，三encoder/shared-decoder均限当前PlainConvUNet路径，
其本地合成构造/forward/backward已检查；这不证明完整依赖符合上游要求。
ResidualEncoderUNet或其他依赖新DNA的架构未支持，不应据此扩张兼容性声明。


## 10. V2 output lifecycle boundary

V2训练入口使用 `outputs/v2/nnUNet_results`；run-config位于v2/run_configs，
独立预测位于v2/predictions，评价位于v2/metrics或v2/reports。
仅切换native process-local路径，checkpoint/validation/inference实现仍为原生。
原outputs中的V1/LEGACY与共享Stage-1不移动；命名、import、provenance、历史checkpoint兼容不变。
路径守卫检查canonical containment并阻止symlink转向历史目录；V2模型对照标签显式版本。
