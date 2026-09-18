# P2-A：legacy PlainConv M0 自研基线（v2.2 工程实现归档）

> **v2.3 覆盖说明（2026-09-16，最高优先级）**：本文件主体记录的是 v2.2 `PlanDrivenUNet3D`
> **PlainConv M0 的历史实现**，用于审计可复用的数据、损失、trainer、低频验证与滑窗基础设施；它不再是
> `docs/research_plan.md` v2.3 定义的正式 M0。v2.3 正式 M0 已统一升级为 plan-driven Residual-Encoder 3D U-Net，
> **其 P2A 实现与合成回归已完成**（residual block、版本化架构配置/稳定哈希、checkpoint 身份隔离、§8.6 接口契约、
> 参数/MACs 摘要与合成测试均已落地；§9.4.2 G2 前置 1–7 = ✅，当时的全量 `tests/unit` 为 435 passed），
> **但真实数据三步验证（第 8 项）未运行、G2 = NOT YET EVALUATED**；v2.3 实现与运行命令见
> `docs/P2_M0_ResidualEncoder_v23.md`。**动态数字（当前测试基线、实现/运行状态）不在本文件维护**：
> 见 `docs/STATUS.md` 与 `docs/experiment_log.md`（2026-09-18 基线：全量 `tests/unit` 714 passed）。
> 在研究者按该新文件执行并通过 G2 前，**不得执行本文旧 PlainConv 命令来宣称完成正式 G2，也不得把旧
> PlainConv checkpoint 与新架构混用**。与本文旧状态冲突时，以研究计划 v2.3.1 §6、§9.4.2、§14/G2、§20 与
> `docs/P2_M0_ResidualEncoder_v23.md` 为准。

> **v2.2 状态更新（2026-09-15，取代原 v2.1 覆盖说明）**：v2.1 宣布的低频验证迁移**已完成并落地**——
> `validation.every_n_epochs=5`（第 5、10…个 epoch 验证，**最后一个 epoch 必验证**）、非验证 epoch 不调用
> validator/不更新 best/patience、**正式主实验禁用“连续无改善”early stopping 的固定 200-epoch 预算**、
> resume 恢复 validation-event/best 状态、协议变化默认拒绝 resume（显式放行时重置选模状态）。
> 以上由合成 CPU 回归测试与仓库配置自检锁定，见 `docs/research_plan.md` §9.4.1 与本文 §18。
> 下文 §8/§16/§18 已同步为 v2.2 目标值；其他章节中的历史记录（如 §15 复审记录、§16.5 旧标题注释）仅用于
> 追溯——与 v2.2 目标值矛盾时，以 `configs/experiments/m0_picai_3d_fullres.yaml`、代码实现与 §18 差距表为准。

> 历史状态：**legacy PlainConv 代码 = CODE READY（含 v2.2 低频验证迁移）**（限定标准续训路径：fresh run，或相同配置下从
> `checkpoint_last.pth` 续训）；**v2.2 正式训练 = NOT READY（待真实数据三步验证；G2 = NOT YET EVALUATED）**。
> 复审结论（第三轮）：core 流程 PASS；三项非核心问题（best/周期 checkpoint 缺 epoch 元数据、续训只核对协议
> 未核对完整配置、resume 与已有日志未对齐）已修复并加回归测试（见 §17.1/§17.2/§17.4）；
> `--allow-mid-epoch-resume` / `--allow-protocol-change` 语义已修正（绝对 epoch 计数、放行时重置选模状态），
> 但**正式实验不建议使用**——正式训练中断后请从 `checkpoint_last.pth` 恢复；从旧周期 checkpoint 做分支实验请换新 `--run-name`。
> 首轮 Code Ready 提交经评审发现 3 个阻塞问题（CUDA 训练会失败、`class_locations` 官方 (n,4) 格式处理错误、
> 滑窗与 smoke 缺少真实进度条）与 6 项对齐问题，已全部修复并以合成测试回归（见 §15）。
> P2-A3 追加（v2.2 已迁移）：**每 5 epoch 全体积验证（223 例）+ 按阳性 case-wise Dice 选择 checkpoint**（见 §16）；
> 其复审提出的 4 个续训安全问题（中断 epoch 语义、协议核对、RNG 被覆盖、指标未硬冻结）已修复（见 §17）。
> 本轮同样**只完成代码与合成 CPU 测试**：未读取真实病例、未使用 GPU、未运行训练或真实验证。
> 本文件描述的是代码实现状态，不代表 M0 有效，也不代表已通过 G2。
> 本文件历史上约定真实数据命令由研究者执行；v2.3 覆盖后，旧 PlainConv 命令当前**不再列入执行队列**。
>
> **损失口径状态（2026-09-15 更新）**：M0 已完成 **Focal+CE 迁移**——`training/losses.py` 新增
> `PiCAIFocalCELoss`（`0.5×Focal(γ=2, α=None) + 0.5×CE`，逐 deep-supervision 尺度），与正式 N0
> （`PI-CAI official-style nnU-Net v2 Focal+CE NoFFT`）**逐值对齐**，由 `tests/unit/test_loss_alignment_n0.py`
> 锁定；`configs/experiments/m0_picai_3d_fullres.yaml` 已切换为 `loss.name: focal_ce` 并做显式性校验。
> `DiceCELoss` 保留可用，但**与正式 N0 不可做正式性能比较**。
>
> **v2.2 低频验证/固定预算迁移（2026-09-15）**：已完成——`validation.every_n_epochs=5`、validation-event
> 调度、非验证 epoch 不更新选模状态、`early_stopping.enabled=false`（正式主实验禁用“连续无改善”停止）、
> `patience=8`（validation-event 单位，仅通用代码保留）；由仓库配置自检与合成回归测试锁定，见 §18 与
> `docs/research_plan.md` §9.4.1。
> **v2.3 处理**：不再对旧 PlainConv M0 执行真实数据 G2。下一步先实现 Residual-Encoder M0；完成研究计划
> §9.4.2 的 G2 前置 1–7 项并更新本文件的命令后，才由研究者执行新 M0 的真实数据验收。增强仍只实现同步镜像
>（`augmentation.pending_parity` 6 项未对齐）。差距清单见 §18。

固定参照：`third_party/nnUNet` v2.6.2（commit `74ceb6803d10dcee29b2cc481678d3a3d069f281`）；
计划文件：`workdir/nnUNet_preprocessed/Dataset605_PICAI/nnUNetPlans.json`（`nnUNetPlans` / `3d_fullres`）；
划分文件：`workdir/nnUNet_preprocessed/Dataset605_PICAI/splits_final.json`（单 fold：train 1277 / val 223）。

---

## 1. legacy PlainConv M0 边界

本节的 legacy M0 = **T2W + ADC + HBV 三通道普通 concat + PlainConv 基线**，位于
`src/zonal_reliability_fusion/`。其数据与训练基础设施可复用，但网络身份不代表 v2.3 正式 M0：

| 允许 | 禁止（M0 一律不使用） |
|:--|:--|
| Conv3d + InstanceNorm3d + LeakyReLU 的 plan-driven 3D U-Net | PZ/TZ、WG 等解剖先验输入 |
| 三序列通道拼接 | 序列特异性 stem / 多分支融合 |
| skip connection + deep supervision | attention / gate / Transformer / Mamba |
| 滑窗推理 + Gaussian 融合 + 可选镜像 TTA | 预训练 encoder、病灶专用分支、modality dropout/corruption |

判别依据（可自查）：`rg -n "PZ|TZ|zonal|gate|attention" src/zonal_reliability_fusion/models/` 只应命中注释与文档字符串。

## 2. 模型结构与 plan 参数映射

`Plan3DConfig`（`config/plans.py`）读取并校验 plan；`PlanDrivenUNet3D`（`models/unet3d.py`）
按 plan 构建 encoder/decoder；`M0ConcatModel`（`models/m0_concat.py`）固定模态语义与通道校验。

| plan 字段（3d_fullres） | 值 | 在自研网络中的用法 |
|:--|:--|:--|
| `spacing` | `[3.0, 0.5, 0.5]` | 记录进 run 快照（预处理已完成重采样，训练不再插值） |
| `patch_size` | `[16, 320, 320]` | 采样器 patch；配置必须与之一致，否则启动即报错 |
| `batch_size` | `2` | 训练 batch（配置校验项） |
| `n_stages` | `7` | encoder 级数；decoder 级数 = 6 |
| `features_per_stage` | `[32,64,128,256,320,320,320]` | 各级通道数 |
| `kernel_sizes` | `[[1,3,3],[1,3,3],[3,3,3]×5]` | 各级卷积核（odd → same padding `k//2`） |
| `strides` | `[[1,1,1],[1,2,2],[1,2,2],[2,2,2],[2,2,2],[1,2,2],[1,2,2]]` | 各级第一次卷积的下采样 stride；decoder 用 `ConvTranspose3d(kernel=stride, stride=stride)` |
| `n_conv_per_stage` | `[2]×7` | encoder 每级卷积次数（第一次带 stride） |
| `n_conv_per_stage_decoder` | `[2]×6` | decoder 每级卷积次数（拼接后） |
| `conv_bias` / `norm_op_kwargs` / `nonlin_kwargs` | `True` / `eps=1e-5, affine=True` / `LeakyReLU(0.01)` | 自研块固定实现；plan 不一致即报错 |
| `batch_dice` | `False` | 损失按逐样本 Dice 计算 |

各 stage 空间尺寸（由 plan 推导，`Plan3DConfig.stage_shapes()`）：

```
stage0 (16,320,320) → stage1 (16,160,160) → stage2 (16,80,80) → stage3 (8,40,40)
      → stage4 (4,20,20) → stage5 (4,10,10) → stage6 (4,5,5)   [bottleneck]
```

## 3. 输入输出 shape

- 输入：`[B, 3, D, H, W]`，**channel 0/1/2 固定为 T2W/ADC/HBV**（顺序错误在模型入口报错，配置层也校验）；
- 输出：二分类 logits，主输出 `[B, 2, D, H, W]`；**网络内部不做 softmax**；
- 训练/验证：`deep_supervision=True` 时返回 6 个输出（高→低）：
  `[(16,320,320), (16,160,160), (16,80,80), (8,40,40), (4,20,20), (4,10,10)]`；
- 推理：使用最高分辨率输出（`M0ConcatModel.main_logits` / 滑窗内部自动取 `output[0]`）。

## 4. Deep supervision

- 权重 `w_i = 1/2^i`，**最低分辨率置 0**，再归一化（sum=1）→ 6 个输出时为
  `[0.516, 0.258, 0.129, 0.065, 0.032, 0.0]`；只有 1 个输出时返回 `(1.0,)`（不会把唯一权重置 0）；
- target 用 **nearest** 下采样到各输出尺寸（不产生新标签值）；
- 最低分辨率权重为 0 的输出不参与反向（跳过计算），与 nnU-Net v2.6.2 行为一致。

## 5. 损失

**当前口径（与正式 N0 对齐）：`PiCAIFocalCELoss`**

`L = focal_weight·Focal(gamma, alpha=None) + ce_weight·CE`，默认 `focal_weight = ce_weight = 0.5`、
`gamma = 2.0`、`smooth = 1e-5`，逐 deep-supervision 尺度计算（DS 权重同 §4）：

- 公式复刻 PI-CAI 官方公开实现：Softmax → one-hot `clamp(smooth, 1−smooth)` → `pt + smooth` →
  `−(1−pt)^γ · log(pt)` → mean；`alpha=None` 即**无类别权重**（不是 `focalLossAlpha75` 变体）；
- CE 直接接收 logits；负标签/越界标签/非整数标签**显式报错**（Dataset605 无 ignore label），不静默当背景；
- 全背景 target 不产生 NaN（有专门单测）；
- 与正式 N0（`integrations/nnunet_picai_flce.PiCAIFocalCrossEntropyLoss`）在同一 logits/target 上
  **逐值一致（atol ≤ 1e-7）**，DS 加权结果与官方 `DeepSupervisionWrapper` 一致
  （`tests/unit/test_loss_alignment_n0.py`）；
- 实现**不依赖 nnunetv2**（`training/` 的边界要求）。

**配置与可追溯性**：`configs/experiments/m0_picai_3d_fullres.yaml` 的 `loss.name` 显式声明口径
（`focal_ce` / `dice_ce`）；`validate_loss_config` 对「缺参数、两口径混用、未知名称」显式报错，
保证每个 run 的 resolved 快照都能回答「这次训练用的是哪种损失」。

**历史口径（保留可用，但与正式 N0 不可比）：`DiceCELoss = ce_weight·CrossEntropy + dice_weight·(−Dice)`**

- CrossEntropy 直接接收 logits；Dice 在 softmax 概率上计算；
- **Dice 项返回 `−Dice`（与 nnU-Net v2.6.2 的 `SoftDiceLoss.forward` 完全一致，无 +1.0 偏移）**，
  因此 Dice 项 ∈ [−1, 0]，总 loss 可能为负；`dice_score()` 提供正 Dice（[0,1]）用于人类可读的日志；
- Dice 默认排除 background；`smooth=1e-5`；分母 clip 到 `1e-8`（官方同样处理）；`batch_dice=False`（与 plan 一致）；
- 全阴性 target：分母仍由 softmax 概率提供（>0），**不产生 NaN**（有专门单测）；
- 训练期指标：前景 Dice、TP/FP/FN、train/val loss；正式 PI-CAI lesion AP（picai_eval）**不在本阶段实现**。

## 6. Patch 采样

`data/patch_sampler.py`（M0–M4 共用，纯 numpy）：

- patch 来自 plan；训练按官方语义强制前景：**每个 batch 固定 `bs − round(bs·(1−oversample))` 个样本**
  （`bs=2`、`oversample=0.33` → 每批 1/2；验证集固定 0），不是逐样本独立概率；
- 前景位置优先取 `properties['class_locations']`（preprocessor 写入 `.pkl`），缺失时从 seg 现算；
  官方保存的是 **(n, 4)**（首列为标签索引，后 3 列为空间坐标，官方用 `selected_voxel[i+1]`）；
  `PatchSampler.normalize_class_locations` 同时接受 (n,3)/(n,4)；
- 前景 patch **以体素为中心**取（`voxel − patch//2`，官方同法），越界时平移 bbox（官方用 padding）；
- **阴性病例安全回退**为随机 crop（记录 `foreground_fallback=True`，不报错、不伪造前景）；
- **验证集 force_fg=False**，不启用任何前景过采样；
- 影像小于 patch 时显式 padding（data=0.0，seg=0，仅补高端），影像与标签严格同步（同一 bbox，有单测）；
- 数据读取经 `data/preprocessed_store.py`（唯一 import `nnunetv2` 的位置，惰性 `NNUNetDatasetBlosc2`）。

## 7. Augmentation 现状与暂存差异（诚实说明）

| 类别 | 状态 |
|:--|:--|
| 随机镜像（三轴、影像/标签同步） | **已实现并测试**，训练默认启用 |
| 分序列强度增强（逐通道 scale + offset） | 已实现接口，**默认关闭** |
| rotation / scaling / noise / blur / gamma | **仅有显式接口**（`UnavailableAugmentor`，调用即报错），本轮不实现未测试的 3D 空间变换 |
| 验证集 | 显式空管道（无任何随机增强） |
| modality corruption / dropout / 域随机化 | M0 禁止，未实现 |

> **差异显式冻结（不再只是文字说明）**：配置 `augmentation.pending_parity` 列出尚未对齐的增强项
> （rotation / scaling / low_resolution_simulation / gaussian_noise / blur / gamma），
> 配置校验会把它们作为 warning 打印；任何 run 的 resolved 快照都带有这份清单。
> 在补齐并通过验证前，任何"N0 vs M0"的性能差异都可能部分来自增强差异，**不得声称已完全对齐**；
> 正式 N0/M0 比较前必须补齐，或把该清单作为明确记录的实验差异冻结（见 §12 与 §15）。

## 8. 优化器与训练循环

| 项 | 值 | 来源 |
|:--|:--|:--|
| optimizer | SGD，lr=1e-2，momentum=0.99，nesterov=True，weight_decay=3e-5 | 配置 |
| scheduler | PolyLR，power=0.9，按 epoch step（`lr = lr0·(1-epoch/max_steps)^0.9`） | 配置 |
| grad clip | norm 12 | 配置 |
| AMP | 仅 CUDA 启用；CPU 请求时给出 WARN 并按 FP32 运行 | 配置 + 运行时判定 |
| max_epochs / iters | **200（v2.2 固定预算，非最优）** / 250（epoch）；PolyLR 终点 = 200 | 配置 |
| 正式验证 | **每 5 epoch（`every_n_epochs=5`，最后一个 epoch 必验证）**全体积验证 **223 例**（阳性 63 / 阴性 160），随机 patch 仅用于诊断 | 配置 + §16 |
| checkpoint 指标 | `val_positive_casewise_dice_mean`（最大化，min_delta=1e-4；**best 只在 validation event 产生**） | 配置 |
| early stopping | **正式 M0–M4 禁用（`enabled=false`）**；通用代码保留，启用时 `patience=8` 按 **validation event** 计数、`min_epochs=50` 按 training epoch，min_delta=1e-4 与 checkpoint 一致 | 配置（v2.2） |
| 诊断验证 iters | 50（仅 loader smoke / small-overfit 使用） | 配置 |
| fold / seed | 0 / 0 | 配置 |

训练器（`training/trainer.py`）：显式 `device`（**构造时执行 `model.to(device)`**，并在 manifest 记录
`model_parameter_device`）、import 不初始化 CUDA；epoch + batch 两级 tqdm、验证同样有进度；
`--no-progress` 可关闭；日志 `metrics/epoch_metrics.csv|jsonl`；checkpoint（原子写入）分
**last / best（按 `val_positive_casewise_dice_mean` 最大化，见 §16）/ 周期 / interrupt**；
resume 后 epoch 与 LR 连续（PolyLR 的 `max_steps` 对齐到本次 run 的 epochs），并核对完整训练协议（见 §17）。

数据管线（`data/batch_provider.py`）：`num_workers` / `pin_memory` 来自配置并经 `torch.utils.data.DataLoader` 真实生效；
每个样本的随机种子由 `(seed, epoch, index)` 稳定混合 → 采样与 worker 数无关、无隐藏 RNG 状态；
训练器每轮调用 `set_epoch(epoch)`，因此 **resume 后采样序列可复现**（不依赖保存 RNG 状态）。
`pin_memory` 仅在 CUDA 设备下启用。

## 9. 滑窗推理

`inference/sliding_window.py`：patch 来自 plan；`step_fraction=0.5`（可用 nnU-Net 语义的步长推导，覆盖最后一个位置）；
小于 patch 的输入先 padding、输出裁回；Gaussian importance map（中心 > 边缘，处处 > 0）；
累计权重处处 > 0（显式断言，拒绝不完整结果）；float32 累计 logits，融合后再 softmax；
支持 `batch_size>1`；DS 模型只取 `output[0]`；`mirror_tta` 默认关闭（开启时对 8 种翻转的 logits 求平均——
与 nnU-Net 的概率平均存在差异，**需在正式评估前对齐确认**）；带病例级 tqdm。

## 10. 与官方 N0 的异同（当前）

| 维度 | N0（官方 nnUNetv2） | M0（自研） |
|:--|:--|:--|
| 网络实现 | `dynamic_network_architectures.PlainConvUNet` | 自研 `PlanDrivenUNet3D`（结构参数同 plan，代码独立） |
| 通道/归一化 | Conv3d+IN+LeakyReLU | 同 |
| patch/batch/DS 权重 | plan 与 1/2^i 归一化 | 同（有单测） |
| 损失 | `0.5×Focal(γ=2, α=None) + 0.5×CE`（PI-CAI official-style） | 同（`PiCAIFocalCELoss`，与 N0 逐值对齐，见 §5/§18；`tests/unit/test_loss_alignment_n0.py` 锁定） |
| 优化器/调度/裁剪 | SGD+Nesterov+PolyLR(0.9)，clip 12 | 同（值全部来自配置） |
| 数据读取 | nnUNetDataLoader（blosc2 局部切片） | 自研 Provider（整例读取后 crop；语义相同、吞吐较低） |
| 前景过采样 | 0.33（含 `class_locations`） | 同 |
| 增强 | 完整（rotation/scaling/noise/blur/gamma/低分辨率模拟） | **当前仅镜像**（差异见 §7） |
| 滑窗 | nnU-Net 滑窗 + Gaussian | 自研实现（语义对齐，TTA 概率平均方式待确认） |
| 训练框架 | nnUNetTrainer（epoch/日志/DDP 等） | 自研 M0Trainer |

## 11. legacy PlainConv 未验证项目（历史记录，不再作为 v2.3 G2 验收）

1. `PreprocessedStore` 对真实 `.b2nd` 的读取与 seg 形状契约（`.pkl` → `[1,D,H,W]`、取值 {0,1}）；
2. 真实病例上 patch sampler 的前景命中率与 bbox 行为；
3. 真实数据前向的显存占用、patch 尺寸可行性（`16×320×320` + 7 级 + DS 输出）；
4. small-overfit 收敛性（loss 下降、Dice 上升）；
5. 滑窗推理在全尺寸病例上的形状恢复与耗时；
6. 与 N0 的训练曲线/指标对齐度（增强差异待评估）。

## 12. legacy PlainConv small-overfit 标准（归档，v2.3 暂勿执行）

- 使用 2–4 例**阳性**病例，固定 `--iterations`（建议 ≥60）、`--epochs 1`；
- 判据（同时满足）：train loss 单调/总体下降且明显低于初始值；前景 Dice 上升（>0.3 视为有拟合迹象）；
  产出 `loss_curve.png` 与独立 `checkpoints/`（**不得**写入正式训练目录）；
- 失败处理：先查数据/标签/采样与 LR，而不是提升模型复杂度；过拟合不通过不得启动正式训练。

## 13. M0 与 M1–M4 的接口边界

- 主干与数据层已按可替换设计：`Plan3DConfig`（结构来源）、`PlanDrivenUNet3D`（共享主干）、
  `PatchSampler` / `PreprocessingStore` / `BatchProvider`（M0–M4 共用同一 sampler 与增强协议）、
  `DiceCELoss` + `DeepSupervisionLoss`、`M0Trainer`、滑窗与指标；
- 后续 M1–M4 只需新增各自的输入投影/融合模块与配置，**不得**修改 M0 的采样比例、损失口径与推理协议；
- 本轮**未实现**任何 gate / 分区条件化 / 多分支 stem 代码，避免提前引入未验证结构。

> **后续更新（2026-09-17/18）**：M1–M4 的输入投影 / 多分支 stem / gate / 分区条件化**已在后续轮次实现**
> （`models/fusion_blocks.py`、`models/fusion_multi_branch.py`，代码 + 合成测试，**未训练**），
> 且未修改 M0 的采样比例、损失口径与推理协议；实现顺序与协议条款的关系见
> `docs/protocol_changelog.md` 2026-09-17/18 条目，当前状态见 `docs/STATUS.md` §2。

## 14. 代码位置与产物

| 类型 | 路径 |
|:--|:--|
| 配置层 | `src/zonal_reliability_fusion/config/{plans,experiment}.py` |
| 模型层 | `src/zonal_reliability_fusion/models/{blocks,unet3d,m0_concat}.py` |
| 数据层 | `src/zonal_reliability_fusion/data/{preprocessed_store,patch_sampler,transforms,batch_provider}.py` |
| 训练层 | `src/zonal_reliability_fusion/training/{losses,deep_supervision,metrics,optim,checkpointing,trainer}.py` |
| 推理层 | `src/zonal_reliability_fusion/inference/sliding_window.py` |
| 工具 | `src/zonal_reliability_fusion/utils/{progress,reproducibility}.py` |
| 脚本 | `scripts/train/validate_m0_setup.py`、`scripts/train/train_m0.py` |
| 配置 | `configs/experiments/m0_picai_3d_fullres.yaml` |
| 测试 | `tests/unit/test_{plans,m0_model,losses,patch_sampler,sliding_window,checkpointing,progress,batch_provider,trainer_synthetic}.py` |
| 训练输出 | `outputs/checkpoints/m0/<run>/`、`outputs/metrics/m0/<run>/`、`outputs/diagnostics/m0/<run>/` |

---

## 15. 复审问题与修复（2026-09-14 第二轮）

首轮 Code Ready 提交经评审返回，问题与修复如下（全部以**合成数据**回归，未读真实病例、未使用 GPU）：

### 15.1 阻塞问题（已修复）

| # | 问题 | 修复 | 回归证据 |
|:--|:--|:--|:--|
| 1 | **CUDA 训练会直接失败**：trainer 只记录 `device`，未执行 `model.to(device)` → "CUDA 输入 + CPU 模型" | `M0Trainer.__init__` 中显式 `self.model = self.model.to(self.device)`；manifest 记录 `model_parameter_device` | `test_trainer_moves_model_to_device_and_syncs_data_epoch`、`test_trainer_manifest_records_model_device` |
| 2 | **`class_locations` 官方格式是 (n,4)**（首列标签索引），此前强制 (n,3) → 真实数据直接 `ValueError` | 新增 `PatchSampler.normalize_class_locations`（接受 (3)/(4)，取后 3 列）；前景 patch 改为官方"以体素为中心" | `test_class_locations_official_nx4_format`、`test_foreground_patch_is_centered_on_voxel` |
| 3 | **滑窗与 loader smoke 无真实进度条**（`make_progress` 未接入循环），违反 AGENTS.md §1 | 滑窗 `_single_pass`（含 TTA 每个翻转）与 smoke 的病例/batch 循环全部接入 `make_progress` | `test_progress_flag_is_wired`（progress=True 有 tqdm 输出、False 无输出） |

### 15.2 正式训练前应修正项（已修复）

| # | 问题 | 修复 |
|:--|:--|:--|
| 4 | 前景请求语义与官方不同（逐样本 33% 概率 vs 每批固定数） | 改为官方 `bs − round(bs·(1−oversample))` 每批固定数；测试断言 bs=2/4 与 oversample=1.0 的行为 |
| 5 | `num_workers` / `pin_memory` 配置项未被使用（同步整例读取会拖慢训练） | 数据加载改走 `torch.utils.data.DataLoader`，配置项真实生效（pin_memory 仅在 CUDA 启用）；`num_workers>0` 有回归测试 |
| 6 | checkpoint 未保存采样器隐藏 RNG 状态 | 采样改为 `(seed, epoch, index)` 确定性生成，**不存在隐藏状态**；训练器每轮 `set_epoch(epoch)`，resume 后序列可复现 |
| 7 | 自研 Dice+CE 比官方恒定高 1.0（曲线不可直接比较）——**历史口径 `DiceCELoss`；M0 现行默认已改为 `focal_ce`（见 §5/§18）** | `SoftDiceLoss` 返回 `−Dice`（官方口径），新增"CE − Dice"等值性测试；`components()` 同时给 `dice_loss`（≤0）与 `dice_score`（正）。**逐值比对官方 `SoftDiceLoss`（合成张量，batch_dice=False/True × 含前景/全阴性）：最大绝对差 7.45e-09，`total == CE + Dice` 差 ≤3e-08** |
| 8 | `experiment_log.md` 记的 113 passed 与 P2 文件实际数不符 | 更正为：P2 九文件 94 passed（修复后 102），全量含 `test_picai` 19 项共 113（修复后 121） |
| 9 | `research_plan.md` 仍写 `G0-P = Conditional Pass` | §6/§14 修正为 **PASS**（与审计报告一致，最小同步） |
| 10 | 增强差异只在文档中描述 | 新增配置 `augmentation.pending_parity` + 校验 warning，成为可追踪的显式冻结项 |

### 15.3 复审后仍存在的状态

- 所有修复均为**合成 CPU 测试**验证；真实 `.b2nd` 读取、CUDA 前向、显存占用、收敛性仍**未验证**（见 §11）；
- 增强未对齐项（`pending_parity`）仍未实现，属已知差异；
- 因此本文档状态为 **IMPLEMENTED / FIX APPLIED（待复审）**，需评审再次确认后才可标记 CODE READY。

---

## 16. P2-A3（v2.2 已迁移）：低频全体积验证 + Dice 选择 best + early stopping（可选）

### 16.1 训练/验证数据流（v2.2：每 5 epoch 低频全体积验证）

```
set_epoch(epoch)                      # 采样序列由 (seed, epoch, index) 决定，可复现且无隐藏状态
→ 训练 250 iteration（随机 patch、0.33 前景过采样、同步镜像增强）
→ 仅当 (epoch+1) % 5 == 0 或 epoch == max_epochs-1（最后一个 epoch 必验证）时执行全体积验证：
     全部 223 个 validation study 逐一做完整 3D 滑窗推理
     sliding_window_predict：patch=[16,320,320]、step_fraction=0.5、Gaussian 融合、mirror TTA 关闭
     关闭 deep supervision（只取最高分辨率输出）→ argmax → 在预处理空间算 TP/FP/FN/Dice
     写病例级 CSV（outputs/metrics/m0/<run>/validation_cases/epoch_XXXX.csv）+ epoch 汇总（CSV/JSONL）
     更新 best / bad_validation_events（best 只在 validation event 产生）
→ 非验证 epoch：不调用 validator、不解析 checkpoint 指标、不更新 best/patience；
     CSV 的全体积指标字段写空串占位、validation_skipped=1、validation_event=0
→ 每个 epoch 都保存 checkpoint_last 与周期 checkpoint（保证可恢复）
→ （可选，正式主实验禁用）early stopping：bad_validation_events >= patience（validation event 计数）且
     epoch+1 >= min_epochs（training epoch 计数）→ early stop（写 early_stop_summary.json）
```

- 不用随机 validation patch 代替完整体积；正式模式**不创建**随机 validation patch provider（有测试）；
- 验证期间不改动模型参数；结束后恢复 deep-supervision 状态与 train/eval 状态（有测试）；
- 不落盘每个 epoch 的预测体积；只保存病例级指标；
- best checkpoint 冻结后，才单独运行一次完整预测并保存结果（后续阶段实现）。

### 16.2 checkpoint Dice 的数学定义

对每个 validation study `i`（预处理空间，argmax 后的二值预测）：

```
TP_i = |{v: pred_i(v)=1 且 gt_i(v)=1}|
FP_i = |{v: pred_i(v)=1 且 gt_i(v)=0}|
FN_i = |{v: pred_i(v)=0 且 gt_i(v)=1}|
Dice_i = 2·TP_i / (2·TP_i + FP_i + FN_i)        # 阳性病例预测为空 → 0.0
```

- 阳性病例集合 `P = {i: GT 前景体素 > 0}`（期望 63 个），
  `val_positive_casewise_dice_mean = (1/|P|)·Σ_{i∈P} Dice_i`（checkpoint 指标，最大化）；
- `val_positive_casewise_dice_median = median{Dice_i : i ∈ P}`；
- `val_micro_dice = 2·Σ_i TP_i / (2·Σ_i TP_i + Σ_i FP_i + Σ_i FN_i)`（对全部 223 例累加）；
- **阴性病例（期望 160 个）不进入主要 Dice 均值**；`GT 与预测均为空` 记 `Dice_i = 0`（且不被计入均值）；
- 阴性表现：`val_negative_fp_case_rate`（有 FP 的阴性例占比）、`val_fp_voxels_per_negative_exam`（阴性例平均 FP 体素数）；
- 每轮硬校验 `223 / 63 / 160`（覆盖、重复、遗漏），不一致立即报错且不保存 best；预测出现 NaN/Inf 同样立即报错。

### 16.3 validation event、early stopping（可选）与状态恢复（v2.2）

- 每个 full-volume validation event 递增 `validation_events_completed` 并记录 `last_validation_epoch`；
  `bad_validation_events` **只在 validation event 更新**（非验证 epoch 不递增）；
- `EarlyStoppingConfig(enabled=false（正式 M0–M4）, min_epochs=50, patience=8, min_delta=1e-4)`：
  `patience` 以 **validation event** 计数、`min_epochs` 以 **training epoch** 计数，只能在 validation event 后触发；
  与 checkpoint 共用同一指标与同一 min_delta（构造时校验两者一致，避免"两套标准"）；
- checkpoint 额外保存：`checkpoint_metric`、`best_metric`、`best_epoch`、`bad_validation_events`（兼容旧键
  `early_stopping_bad_epochs`）、`validation_events_completed`、`last_validation_epoch`、`validation_every_n_epochs`、
  `min_epochs`、`patience`、`min_delta`、`validation_protocol` 快照；
- `resume()` 恢复：model/optimizer/scheduler/scaler/RNG、`start_epoch = epoch+1`、`best_metric/best_epoch`、
  `validation_events_completed / bad_validation_events / last_validation_epoch`；`validation_every_n_epochs` 纳入
  协议核对，不一致默认**拒绝续训**（显式放行时重置 best/event 计数并打 WARN）；
- PolyLR 的 `max_steps` 对齐到本次 run 的 `max_epochs`（v2.2 = 200），恢复后学习率由 epoch 索引重新计算 → 连续；
- 采样序列由 `(seed, epoch, index)` 决定，恢复后同一 epoch 位级一致。

### 16.4 配置与运行目录安全

- 配置：`training.max_epochs=200`、`training.diagnostic_validation_iterations`（仅诊断）、`validation.*`、`early_stopping.*`；
  加载时校验 mode 取值、指标存在且非 `val_loss`、expected 计数为正且相加等于总数、**`every_n_epochs >= 1`（v2.2）**、
  `step_fraction ∈ (0,1]`、`min_epochs <= max_epochs`、**正式模式允许 `early_stopping.enabled=false`（v2.2 正式默认，
  启用时打印告警）**、`expected_cases` 必须等于冻结 split 的 val 数；
- 正式 run 启动前检查 checkpoints / metrics / diagnostics 目录：**非空且未显式 `--resume-from` 时拒绝启动**
  （不追加旧 CSV、不覆盖旧 checkpoint）。

### 16.5 v2.2 尚未验证项（均属 G2；合成测试不覆盖）

1. 真实 `.b2nd` 上逐例全体积推理的速度、显存与数值（需 GPU；由研究者执行）；
2. `expected_positive_cases=63 / expected_negative_cases=160` 是否与真实 validation 标签分布一致
   （代码会在首个 validation event 立即校验并报错，不会静默）；
3. 一个完整训练 epoch + 一次全体积验证的真实耗时（用于核对固定 200-epoch 预算的可行性）；
4. `every_n_epochs=5` 调度在真实训练中首个 validation event（第 5 epoch、最后一个 epoch）的行为与 CSV schema；
5. （若今后显式启用）early stopping 的真实行为；正式 M0–M4 不启用。

---

## 17. 续训安全性（P2-A3 复审修复）

正式训练可达 200 epoch，中断与续训的正确性必须显式保证（本轮按复审意见修复）：

### 17.1 epoch 语义显式化

- checkpoint 记录 `epoch_completed`（该 epoch 是否已完整结束）、`completed_epochs`（**绝对**已完整完成的
  epoch 数，跨 resume 累计，不受"本进程只跑了几个 epoch"影响）、`iterations_completed`
  （被打断 epoch 内**已完成**的 iteration 数，被打断的那个不计入）；
- `last / best / 周期` checkpoint 一律 `epoch_completed=True`（`epoch` = 该完整 epoch 的索引），
  且**三者携带完全相同的 epoch 元数据**（`last` / `best` / `periodic` 都写 `completed_epochs` 与
  `iterations_completed`；未显式给出时按 `epoch + 1` 自动补全），因此从任一 checkpoint 恢复都能做一致性校验；
- `checkpoint_interrupt.pth` 由 Ctrl-C 触发，**只写自己、不覆盖 last/best**，且一定是
  `epoch_completed=False`（含 `interrupted=True` 标记）；
- `resume()` 规则：
  - `epoch_completed=True` → `start_epoch = epoch + 1`（正常续训）；
  - `epoch_completed=False` → **默认拒绝**并提示改用 `checkpoint_last.pth`；显式 `allow_mid_epoch=True`
    （CLI：`--allow-mid-epoch-resume`）才继续，此时 `start_epoch = completed_epochs`，即**重跑被打断的 epoch**
    （模型权重为中断时刻状态，该 epoch 已完成的迭代会被重复执行——这一点在 checkpoint 与日志中都明确记录）。

### 17.2 续训一致性：协议 + 完整配置冻结

- `resume()` 会核对 `checkpoint_metric`、`maximize_metric`、`early_stopping_enabled`、`min_epochs`、`patience`、
  `min_delta`、`validation_protocol`；任一不一致或缺失即**默认拒绝**（CLI 需显式 `--allow-protocol-change`）；
- **协议放行时重置选模状态**：旧 `best_metric` / `best_epoch` / `bad_epochs` 在新协议下不一定可比，
  因此显式允许协议变化时会将其清零并打 WARN，选模从续训点重新开始；正式实验不建议改变协议续训；
- **元数据一致性**：`epoch_completed=True` 的 checkpoint（last/best/周期）会校验
  `completed_epochs == epoch + 1`，不一致直接拒绝，避免 epoch 计数漂移；
- **完整配置冻结**：除协议字段外，还会对 resolved 配置快照做扁平化逐项比对（续训时冻结，
  默认拒绝差异），冻结前缀：
  `experiment. / data. / model. / loss. / optimizer. / scheduler. / training. / inference. / validation. /
  early_stopping. / plan. / train_pipeline. / val_pipeline. / provenance.plans_sha256 / provenance.splits_sha256`
  —— 覆盖 seed、fold、模态顺序 `[T2W, ADC, HBV]`、patch/batch、增强、loss、optimizer/SGD 超参、
  iterations/epoch、plan 解析结果与 **plan/split 文件哈希**；
  唯一允许显式延长的训练项是 `training.max_epochs`（如 200 → 300）；
  `num_workers / pin_memory / checkpoint_every / progress` 等仅影响吞吐/遥测的字段允许变化（打印记录，不阻塞）；
  冻结前缀之外的其他差异也会打印出来，不会被静默忽略；

### 17.3 RNG 与学习率

- `resume()` 恢复 RNG 后置内部标记，`train()` **不再调用 `set_seed`**（旧实现会在续训开始时覆盖已恢复的 RNG 状态）；
- PolyLR 的 `max_steps` 对齐本次 run 的 `max_epochs`，学习率由 epoch 索引重新计算 → 续训后 LR 连续；
- 采样序列由 `(seed, epoch, index)` 决定，续训后同一 epoch 位级一致。

### 17.4 日志对齐（禁止"从旧 checkpoint 回到同一 run"）

`resume()` 还会核对日志：**checkpoint 的续训起点必须等于 `metrics/epoch_metrics.csv` 最后一个 epoch + 1**，
否则直接拒绝，并提示改用 `checkpoint_last.pth` 或换新的 `--run-name`。原因：从较旧的周期 checkpoint
恢复到同一 run 会产生重复/错序的 epoch 行，并覆盖同名的病例级 CSV。该检查对正常续训（含 mid-epoch 重跑）
都是自然满足的：mid-epoch 时被打断的 epoch 未写入日志，`start_epoch = completed_epochs = CSV 最后 epoch + 1`。

### 17.5 冻结指标与方向

- `ValidationProtocol.validate()` 在 `full_volume` 模式下**硬冻结** `checkpoint_metric =
  val_positive_casewise_dice_mean` 且 `maximize = True`：不允许换成 `val_micro_dice` / `val_loss` 等其他字段，
  也不允许最小化方向（配置层与 validator 共用同一实现，均有测试）；
- `trainer` 侧对正式模式同样强制"指标非 val_loss 且 maximize=True"，形成双重防线。

---

## 18. 与正式 N0 的协议差距（2026-09-15；损失与 v2.2 验证协议均已迁移，其余待办）

正式 N0 = `PI-CAI official-style nnU-Net v2 Focal+CE NoFFT`（唯一方法学变化 = 损失）。

**已完成（1）损失迁移**：M0 已迁移到 Focal+CE——`training/losses.py::PiCAIFocalCELoss` 与 N0 适配层在同一
logits/target 上逐值一致（atol ≤ 1e-7），DS 加权结果与官方 `DeepSupervisionWrapper` 一致，配置显式声明
`loss.name: focal_ce` 并对非法组合显式报错（`tests/unit/test_loss_alignment_n0.py` 共 23 项锁定）。
M1–M4 必须复用同一实现与校验，**不得**引入第三种损失口径。

**已完成（2）v2.2 低频验证与固定预算迁移**：`validation.every_n_epochs=5`（最后一个 epoch 必验证）、
validation-event 调度（非验证 epoch 不调用 validator、不解析指标、不更新 best/patience；CSV 标记
`validation_skipped=1`）、`early_stopping.enabled=false`（正式主实验禁用“连续无改善”停止）、`patience=8`
（validation event 单位，仅通用代码保留）、resume 恢复 event/best 状态、协议变化默认拒绝 resume；
由合成回归测试与仓库配置自检锁定（`tests/unit/test_trainer_synthetic.py` 的 lowfreq 系列、
`tests/unit/test_plans.py`、`tests/unit/test_train_m0_script.py`；详见 `research_plan.md` §9.4.1）。

**legacy PlainConv 未完成项（历史记录，不再安排为 v2.3 G2）**：真实数据 loader smoke、small-overfit、正式训练、
全体积验证；`augmentation.pending_parity` 的 6 项增强对齐；训练预算差异的显式冻结。v2.3 的 G2 必须改在新的
Residual-Encoder M0 上执行，不能通过补跑这些旧项获得。
`DiceCELoss` 保留可用，但**与正式 N0 不可做正式性能比较**，也不得用其差异做创新归因。

| 项 | 正式 N0 | M0 现状 | 处理 |
|:--|:--|:--|:--|
| 损失 | `0.5×Focal(γ=2, α=None) + 0.5×CE`，逐 DS 尺度 | **同为** `0.5×Focal(γ=2, α=None) + 0.5×CE`（`PiCAIFocalCELoss`） | ✅ 已完成，逐值对齐有测试锁定 |
| DS 权重 | v2.6.2 默认（`1/2^i`，最低分辨率不监督） | 同 | ✅ 已一致 |
| 采样 | v2.6.2 默认（0.33 前景过采样；同 plan patch/batch） | 语义一致 | ✅ 已一致（`pending_parity` 不涉及采样） |
| 验证调度 | nnU-Net 在线 `Pseudo dice`（每 epoch，patch 级） | **每 5 epoch 全体积验证**（223/63/160 硬校验；best 只在 validation event） | ✅ v2.2 已迁移（与 N0 口径不同，声明为差异） |
| early stopping | v2.6.2 默认（可关闭） | **正式主实验禁用**（固定 200-epoch 预算） | ✅ v2.2 已冻结 |
| 增强 | v2.6.2 默认增强（rotation/scaling/noise/blur/gamma/低分辨率模拟） | **只实现同步镜像**（`augmentation.pending_parity` 6 项） | ❗ 正式比较前补齐，或显式冻结为实验差异 |
| 优化器 / 调度 | SGD + Nesterov + PolyLR（官方默认口径） | 同（值全部来自配置）；PolyLR 终点 = 200 epoch | ✅ 已一致（终点随预算，声明差异） |
| 训练预算 | 1000 epoch（官方默认） | `max_epochs=200`（v2.2 固定预算） | ❗ 正式比较前显式声明差异，或对齐 |
| 评测 | nnU-Net `--val`（223 例 `nanmean`，有 FP 的阴性例计 0 并计入） | 阳性病例 case-wise Dice（223/63/160 硬校验） | ❗ 论文指标统一走 `picai_eval` 口径（`scripts/evaluate/` 待实现；G0-SAP 草案见 `docs/protocols/G0_SAP.md`） |

**v2.3 下一步**：上述旧 PlainConv 入口暂不得用于正式 G2。先完成 `docs/research_plan.md` §9.4.2 的 G2 前置
1–7 项；随后在本文件新增 Residual-Encoder M0 的结构身份、配置哈希、输出目录与成功判据，并由研究者执行更新后的
setup → loader-smoke → small-overfit → 首次全体积验证命令。旧配置、旧命令和旧 checkpoint 只作
legacy/feasibility 审计，不能补标为 v2.3 正式结果。

---

## 附录 A：v2.2 低频全体积验证——功能开发 checklist 与文件清单（迁移自 research_plan §9.4.1，2026-09-17）

> **迁移说明**：本附录接收自 `docs/research_plan.md` 旧 §9.4.1 迁出的**实现完成记录**（含 ✅/⏳ 状态与文件清单）。
> 研究协议侧的**规范性要求**保留在 `docs/research_plan.md` §9.4.1；本节只记录工程实现历史，不构成协议条款来源。
> 状态标记口径：✅ = 当时代码已实现且被合成测试覆盖；⏳ = 需真实数据/GPU 验证（属 G2）。

1. ✅ **validation-event 调度**：`full_volume` 模式按 `every_n_epochs` 触发；1-based epoch 整除 N 时为验证事件，
   否则跳过；`every_n_epochs>=1` 合法，`0/负数` 在配置层、协议层与 trainer 层均报错。
2. ✅ **最终 epoch 必验证**：`epoch == max_epochs-1` 即使不是 N 的倍数也强制触发一次全体积验证。
3. ✅ **非验证 epoch 不更新选模状态**：不调用 `FullVolumeValidator`、不解析 checkpoint 指标、不更新
   `best_metric/best_epoch`、不递增 `bad_validation_events`；日志与 CSV 标记 `validation_skipped=1`、
   `validation_event=0`。
4. ✅ **best 只在 validation event 产生**：`is_best` 在非验证 epoch 恒为 0，`checkpoint_best.pth` 只可能落在验证 epoch。
5. ✅ **checkpoint_last / 周期 checkpoint 在非验证 epoch 仍正常保存**：跳过验证不影响可恢复性（含绝对 epoch 计数）。
6. ✅ **resume 恢复 event 状态**：连续恢复 `validation_events_completed`、`last_validation_epoch`、`best_metric`、
   `best_epoch`、`bad_validation_events`（启用 early stopping 时）与 epoch/LR。
7. ✅ **协议一致性**：`validation_every_n_epochs` 纳入 resume 协议校验；协议变化默认**拒绝 resume**，显式
   `--allow-protocol-change` 放行时**重置**不可比的 best/event 状态并打 WARN。
8. ✅ **CSV schema 稳定**：跳过 epoch 对全体积指标字段写空串占位，保证每行字段集合与顺序一致（不产生错列）。
9. ⏳ **真实 223 例全体积推理**的速度、显存与数值（属 G2，需 GPU，由研究者执行）。

**v2.2 已涉及的文件（基础设施保留；不代表 v2.3 网络已更新）**：

- 配置：`configs/experiments/m0_picai_3d_fullres.yaml`（`every_n_epochs=5`、`early_stopping.enabled=false`、
  `patience=8`（validation-event 单位）、`max_epochs=200`）；
- 配置层：`src/zonal_reliability_fusion/config/experiment.py`（取消“full_volume 强制启用 early stopping”，
  改为启用时告警；`every_n_epochs>=1` 校验）；
- validator：`src/zonal_reliability_fusion/evaluation/full_volume_validator.py`（`ValidationProtocol.validate`
  放开为 `every_n_epochs>=1`）；
- trainer：`src/zonal_reliability_fusion/training/trainer.py`（`_is_validation_epoch` 调度、event 计数、
  best/patience 只在验证事件更新、checkpoint extra 与 resume 恢复新状态、日志/摘要改为 validation-event 口径）；
- 脚本：`scripts/train/train_m0.py`（把 `every_n_epochs` 接入 trainer，更新文档与打印）；
- 测试：`tests/unit/test_trainer_synthetic.py`（低频调度/最终 epoch/best 只在事件/event 计数 patience/
  min_epochs 按 epoch/resume 恢复/协议变化拒绝与重置/非法 N）、`test_full_volume_validator.py`、
  `test_plans.py`、`test_train_m0_script.py`（仓库配置目标值）。

**v2.3 侧对应关系**：新骨干须把架构身份纳入配置快照与 resume 校验，并在 Residual-Encoder M0 上重跑合成回归；
真实数据三步验证见 `docs/runbooks/p2b_m0_validation.md`，状态见 `docs/STATUS.md`。
