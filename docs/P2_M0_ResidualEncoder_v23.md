# P2A：v2.3 plan-driven Residual-Encoder M0 实现说明

> 状态（2026-09-16）：**P2A 实现 + 合成回归完成**；`docs/research_plan.md` §9.4.2 的 **G2 前置 1–7 项已完成**
> （有测试证据），**第 8 项（真实数据三步验证）仍 Pending**；**G2 = NOT YET EVALUATED**。
> 本轮**未读取任何真实医学影像、未使用 GPU、未运行 loader smoke / small-overfit / 训练 / 全体积验证 / 推理 / 评测**。
> 本文件描述 v2.3 正式 M0；legacy PlainConv M0 的历史实现见 `docs/P2_M0_Implementation.md`（保留为审计记录）。
>
> **验收轮修订（2026-09-16）**：根据研究者验收意见完成两项无真实数据修复：（1）修正架构配置的 shortcut 语义为
> “下采样仅由 stride 触发、projection 仅由通道变化触发”（与 `BasicBlockD` 实现一致；旧描述“stride 或通道变化均投影”不准确）；
> （2）为所有冻结规则字段加**严格白名单校验**（声明模型未实现的值——如 `projection_kernel=[3,3,3]`、
> `initialization.method=xavier`、`decoder.skip_merge=sum`、`deep_supervision.output_order=low_to_high`——均在**加载时拒绝**），
> 杜绝“改了配置但不生效”。两项修复后架构哈希由 `747b190a…` 变为 **`8bb5127e…`**（参数量/MACs 不变，因为只修正了声明、
> 未改变已正确的网络行为）。

## 1. 正式 M0 定义（v2.3）

```text
T2W + ADC + HBV 普通通道拼接（[B,3,D,H,W]，channel 0/1/2 固定）
  → default stem（1×Conv3d-IN-LeakyReLU，不下采样）
  → plan-driven Residual-Encoder（每级 StackedResidualBlocks3d；blocks_per_stage=(1,3,4,6,6,6,6)）
  → 普通 U-Net decoder（ConvTranspose3d → center_crop_or_pad → concat skip → stacked conv → 每级 seg head）
  → deep supervision（高→低分辨率，权重 1/2^i、最低层置 0、归一化）
  → 二分类病灶 logits（网络内不做 softmax）
```

M0 **不读取** PZ/TZ、WG、gate；无 attention/Transformer/Mamba/多分支/预训练权重/病灶专用分支（§7.2/§7.3）。

## 2. 冻结架构表（完整）

| 维度 | 冻结取值 | 来源 |
|---|---|---|
| encoder_type | `residual_encoder` | 架构配置 |
| block_type | `basic_post_activation`（BasicBlockD 等价） | 架构配置 |
| activation_order | `post_activation`：`LeakyReLU(conv2(conv1(x)) + shortcut(x))` | 架构配置 |
| conv1 | `Conv3d(k,stride,same-pad,bias) → InstanceNorm3d → LeakyReLU` | 冻结 plan 算子 |
| conv2 | `Conv3d(k,stride=1,same-pad,bias) → InstanceNorm3d`（**无激活**） | 冻结 plan 算子 |
| shortcut（identity） | 空间与通道均相同 → `nn.Identity` | 架构配置 |
| shortcut（downsample） | 有 stride → `AvgPool3d(kernel=stride,stride=stride,ceil_mode=True)` | 架构配置（§7.3 奇数尺寸） |
| shortcut（projection） | 通道变化 → `Conv3d(1×1×1,bias=False) → InstanceNorm3d`（**先 pool 后 1×1×1**，ResNet-D） | 架构配置 |
| 隐式裁剪通道 | **禁止**（`implicit_channel_crop_forbidden=true`） | 架构配置 |
| normalization | `InstanceNorm3d(eps=1e-5, affine=True)` | 冻结 plan |
| nonlinearity | `LeakyReLU(negative_slope=0.01, inplace=True)` | 冻结 plan |
| conv_bias | `True`（主路）；projection 1×1×1 恒 `False` | 冻结 plan / 架构配置 |
| initialization | He/Kaiming `fan_in`+`leaky_relu(a=0.01)`；conv bias=0；IN weight=1/bias=0；**加法前最后一个 IN（conv2 的 norm）weight/bias 置 0** | 架构配置 |
| stem | 1×`Conv3d-IN-LeakyReLU`，stride=(1,1,1)，不下采样 | 架构配置（对齐官方 default stem） |
| n_stages | 7 | 冻结 plan |
| blocks_per_stage | `(1,3,4,6,6,6,6)`（残差块总数 32） | 架构配置（官方 ResEnc-M encoder 拓扑） |
| features_per_stage | `(32,64,128,256,320,320,320)` | 冻结 plan |
| kernel_sizes | `((1,3,3),(1,3,3),(3,3,3),(3,3,3),(3,3,3),(3,3,3),(3,3,3))` | 冻结 plan |
| strides | `((1,1,1),(1,2,2),(1,2,2),(2,2,2),(2,2,2),(1,2,2),(1,2,2))` | 冻结 plan |
| decoder | 普通 U-Net（`residual=false`），`ConvTranspose3d(kernel=stride,stride=stride)`，`center_crop_or_pad` 对齐，concat skip | 冻结 plan / 架构配置 |
| n_conv_per_stage_decoder | `(2,2,2,2,2,2)` | 冻结 plan |
| deep supervision | 输出高→低；权重 `1/2^i`、最低层 0、归一化到 1；单输出权重 1.0；每级独立 seg head | 现有协议 |
| fusion_stage | 2（§8.5，累积 stride `(1,4,4)`） | 架构配置 |
| shallow_skip_contract_id | `shallow_skip.v23.concat_conv1x1_in_leakyrelu`（§8.6，仅接口） | 架构配置 |
| in_channels / num_classes | 3 / 2 | 冻结 plan |
| patch / batch / spacing | `[16,320,320]` / 2 / `[3.0,0.5,0.5]` | 冻结 plan（**不进入架构哈希**，另行冻结） |

`stage_shapes`（= encoder 各级输出，与 plan 一致）：
`[(16,320,320),(16,160,160),(16,80,80),(8,40,40),(4,20,20),(4,10,10),(4,5,5)]`；
DS 输出（高→低，6 个）：`[(16,320,320),(16,160,160),(16,80,80),(8,40,40),(4,20,20),(4,10,10)]`。

## 3. 对照表：官方 ResEnc-M / 冻结 plan / 本项目最终选择 / legacy PlainConv

本地官方源码依据：`third_party/nnUNet/documentation/resenc_presets.md`、
`third_party/nnUNet/nnunetv2/experiment_planning/experiment_planners/resencUNet_planner.py`
（`ResEncUNetPlanner.UNet_blocks_per_stage_encoder=(1,3,4,6,6,6,6,...)`、`UNet_blocks_per_stage_decoder=(1,1,...)`）、
`dynamic_network_architectures`（`ResidualEncoderUNet` / `ResidualEncoder` / `BasicBlockD` / `InitWeights_He` /
`init_last_bn_before_add_to_0`）。

| 项 | 官方 ResEnc-M preset | 冻结 3d_fullres plan | **本项目 v2.3 M0 最终选择** | legacy PlainConv M0 |
|---|---|---|---|---|
| encoder | ResidualEncoder(BasicBlockD) | PlainConvUNet | **ResidualEncoder（自研复刻 BasicBlockD 行为）** | PlainConv |
| encoder blocks/stage | `(1,3,4,6,6,6,6)` | `n_conv_per_stage=(2×7)` | **`(1,3,4,6,6,6,6)`（取自官方 M）** | `(2×7)` plain conv |
| decoder conv/stage | `(1,1,1,1,1,1)` | `(2,2,2,2,2,2)` | **`(2,2,2,2,2,2)`（沿用冻结 plan）** | `(2,2,2,2,2,2)` |
| patch/features/strides | 由 M preset **重新规划**（VRAM 目标 9–11GB，去 5% batch cap） | `[16,320,320]` / `(32..320)` / 见上 | **完全沿用冻结 plan，不重新规划** | 同 plan |
| stem | default（1 conv，无 stride） | — | **default（1 conv，无 stride）** | 无独立 stem（stage0 首个 plain conv） |
| shortcut | ResNet-D：avgpool(ceil_mode=False)+1×1 | — | **avgpool(`ceil_mode=True`)+1×1(bias=False)+IN** | 无 |
| init | He(1e-2)+zero last-norm | — | **He(fan_in,leaky_relu,0.01)+zero conv2 IN** | He（无 zero last-norm） |
| 参数量 | 依 M 自身规划而定 | — | **148,193,644** | 44,577,932 |

**与官方 ResEnc-M 的差异及原因**：

1. **不重新规划 plan**：官方 M preset 会按 VRAM 目标重规划 patch/features/batch；本项目 plan 已冻结
   （G0-P PASS），只升级 encoder block 身份，故 param/MAC 与官方 M 不可直接比较。**不宣称等价**。
2. **decoder 用 `(2×6)` 而非官方 M 的 `(1×6)`**：§7.2/§7.3 规定 v2.3 只改 encoder，decoder 是 M0–M4 共同骨干、
   沿用冻结 plan，保证与 legacy 的唯一差异是 encoder（干净归因）。
3. **shortcut avgpool `ceil_mode=True`**：官方依赖 patch 可被 stride 整除；本项目按 §7.3 要求覆盖奇数尺寸。
   对可整除尺寸（冻结 plan 全部 stage）`ceil_mode` 与默认**逐尺寸等价**，不改变真实 plan 任何 shape。
4. **自研复刻而非 import**：`models/residual_blocks.py` 复刻 `BasicBlockD` 的可验证行为，不 import 第三方，
   不复制其代码；`models/`/`training/`/`inference/` 边界不引入 `nnunetv2`。

**与 legacy PlainConv M0 的差异**：唯一差异是 **encoder（PlainConv stacked → stem + residual stacked）** 与由此带来的
架构身份/参数量；decoder、DS、skip 对齐、损失（Focal+CE）、优化器、训练预算、低频验证、滑窗推理**完全一致**。
旧 `PlanDrivenUNet3D` / `M0ConcatModel` / `configs/experiments/m0_picai_3d_fullres.yaml` 原位保留，未改写。

## 4. 参数量 / FLOPs / 显存（方法与限制如实标注）

产物：`outputs/diagnostics/m0_resenc/m0_resenc_picai_3d_fullres_v23/architecture_summary.json`
（由 `scripts/train/summarize_m0_resenc_architecture.py` 生成；**只读 plan/arch 配置、CPU 构建模型、无 forward 真实 patch、无 GPU**）。

| 指标 | 值 | 方法 / 口径 |
|---|---|---|
| 总参数量 | **148,193,644** | 直接统计 `sum(p.numel())`（精确） |
| 可训练参数量 | **148,193,644** | 全部可训练（精确） |
| 残差块总数 | 32 | `sum(blocks_per_stage)` |
| MACs（encoder） | 785,883,136,000 | 解析枚举（无 forward） |
| MACs（decoder，含 DS heads） | 322,539,980,800 | 解析枚举（无 forward） |
| **MACs（合计，单 patch batch=1）** | **1,108,423,116,800（≈1.108 T MACs ≈ 2.217 TFLOPs）** | `1 MAC=1 乘加`，`FLOPs≈2×MACs`；batch=2 时 ×2 |
| 激活元素（逐样本） | 758,148,000 | 所有 Conv/ConvTranspose/seg-head 输出元素和（nnU-Net 同类代理量） |
| 峰值显存 | **NOT_MEASURED（未实测）** | 本轮无 GPU 授权；须在 P2B 由研究者在真实 GPU 测量。**运行时遥测工具已实现**：`--overfit`（1 iteration + 1 validation）→ `outputs/diagnostics/<model_id>/<run>/diagnostics/gpu_memory_profile.json`；命令与判据见 `docs/runbooks/p2b_m0_validation.md` §1-E，状态见 `docs/STATUS.md` B5 |
| 分析性激活字节（fp32,batch=2） | 6,065,184,000 B（≈6.07 GB，**下界代理**） | 激活元素 × 4B × batch；**非峰值显存** |

- **MACs 计算方法**：`models/profiling.py::analytical_macs` 按冻结 plan + 架构 spec **纯算术**枚举每个
  `Conv3d`（按输出元素计 `Cout·∏So·(Cin/g)·∏k`）与 `ConvTranspose3d`（按输入元素计 `Cin·∏Si·(Cout/g)·∏k`）。
  **包含** decoder 与全部 deep-supervision seg head；**不包含**无权重算子（AvgPool / InstanceNorm / LeakyReLU /
  残差加法 / concat）。输入 shape = `[3,16,320,320]`（单 patch）。
- **公式验证**：`count_macs_with_hooks`（forward-hook 计数）在**缩小的合成 plan** 上与解析值逐值相等
  （`[4,8,8]`：258,560==258,560；`[8,16,16]` 4 级：10,741,760==10,741,760），由 `tests/unit/test_profiling.py` 锁定；
  真实 `[16,320,320]` patch **只走解析路径，不做昂贵 forward**。
- **显存**：明确区分「实测 / 分析估计 / 未测量」——本轮**未测量**（无 GPU）。解析激活字节只是**下界代理**，
  未计权重（≈0.59 GB）、梯度、SGD momentum、反向存储的中间激活与框架开销；**不得当作实测值**。
  ⚠️ **可行性风险（须在 P2B 核实）**：ResEnc-M 级 block 数在冻结 `[16,320,320]`/batch=2 下激活规模很大，
  峰值显存**可能超过 24GB**；若超限，按 `research_plan.md` §19 **统一缩减所有 M0–M4 的 blocks_per_stage**
  （不单独缩减 M4、不上 L/XL），并**升版 + 记录新架构哈希**后重跑合成回归。本轮不预设结论、不编造显存数字。

## 5. 架构哈希与身份

- `architecture_sha256 = 8bb5127e5c9a99ca1a0631cb365cdcf0b866441ac225911e17faa6bc995bbee5`（验收轮修正 shortcut 语义后的冻结值）
- 算法：`sha256(canonical_json(structural_dict))`，`canonical_json` = `json.dumps(sort_keys=True, separators=(",",":"))`；
  **不受 YAML 键顺序/无关空白影响**（`test_hash_independent_of_key_order`）。
- **哈希输入（结构字段）**：`architecture_name, architecture_version, encoder_type, block_type, activation_order,
  n_stages, blocks_per_stage, features_per_stage, kernel_sizes, strides, stem, shortcut, normalization, nonlinearity,
  conv_bias, initialization, decoder, deep_supervision, shallow_skip_contract, fusion_stage, in_channels, num_classes`。
- **不进入哈希（非结构）**：`provenance`（plan 路径 / plans_sha256 / 官方参考）、`source_path`、
  `patch_size / spacing / batch_size`（数据几何，另行冻结在 resolved config 的 `plan.*` 与 `provenance.plans_sha256`）、
  运行时间、绝对输出路径。
- **任一（可自由取值的）结构字段改变 → 哈希改变**（`test_any_structural_field_change_changes_hash`：name/version/
  blocks_per_stage/features/kernel/stride/conv_bias/eps/slope/zero_init/n_conv_decoder/in_channels/num_classes）；
  **规则描述字段受严格白名单冻结**（stem/shortcut/initialization/decoder/deep_supervision/norm.op/nonlin.op 等只能声明
  模型实际实现的唯一取值，声明未实现值→加载时 `ArchitectureConfigError`；`test_illegal_or_unsupported_value_rejected` 覆盖 ~35 例）；
  frozen spec 的 `structural_dict()/to_dict()` 返回**深拷贝**，外部修改返回值不改变哈希
  （`test_mutating_returned_dict_does_not_change_hash`）。
- resolved config 与 checkpoint 均保存 `architecture_name/version/sha256`（+ `encoder_type`）。

## 6. checkpoint / resume 隔离策略

1. **写入**：`train_m0.build_trainer` 把 `model.architecture_identity()` 写入 `config_snapshot["architecture"]` 与
   `config_snapshot["provenance"]["architecture_sha256"]`；legacy（identity=None）保持原快照不变。
2. **resume 冻结**：`M0Trainer.FROZEN_CONFIG_PREFIXES` 新增 `"architecture."` 与 `"provenance.architecture_sha256"`；
   架构哈希不匹配 → `_verify_resume_consistency` **默认拒绝续训**（`test_trainer_resume_rejects_architecture_hash_mismatch`）。
3. **加载前显式守卫**：`run_training` 在 `trainer.resume()` **之前**调用
   `training.checkpointing.verify_checkpoint_architecture(path, expected_identity)`，给出**清楚的拒绝错误**：
   - checkpoint 无架构身份（疑似 legacy PlainConv）→ `ArchitectureIdentityError`（含 "legacy" 提示）；
   - name/version/sha256/encoder_type 不一致 → `ArchitectureIdentityError`（列出差异）。
4. **权重结构隔离**：residual 的 `state_dict` 键（`net.stem.*`、`net.encoder_stages.*.blocks.*.conv1/conv2_conv/
   conv2_norm/skip.*`）与 legacy（`net.encoder_stages.*.blocks.*.conv/norm/act`）不同 → `load_state_dict(strict=True)`
   加载 legacy 权重到 residual **必然 RuntimeError**（`test_legacy_state_dict_strict_load_into_residual_rejected`）。
5. **禁止 strict=False**：`load_checkpoint` 默认 `strict=True`（`test_load_checkpoint_defaults_to_strict_true`）；
   项目代码路径从不传 `strict=False`。
6. **输出隔离**：v2.3 `experiment.name=m0_resenc_picai_3d_fullres_v23`、`output_root=outputs/checkpoints/m0_resenc`
   （与 legacy `outputs/checkpoints/m0` 不共用；`test_output_identity_isolated_from_legacy`）。

## 7. §8.6 浅层 skip 接口契约（P2A 当时仅接口；聚合器 `ShallowSkipAggregator` 后续已实现，见本节末「后续更新」）

`config/architecture.py::SHALLOW_SKIP_CONTRACT`（机器可读，`contract_id=shallow_skip.v23.concat_conv1x1_in_leakyrelu`）：

- 适用 stage `[0,1]`；输入 = 三路 modality-specific encoder 特征（T2W/ADC/HBV，count=3）；
- 目标通道来自 `plan.features_per_stage[stage]`；未来拓扑固定 `concat → conv1x1x1 → instance_norm3d → leaky_relu`；
- **禁止**读取 `pz_tz / wg / sequence_gate_weight / sequence_logits / decoder_feature`；
- **不产生**第二套 modality logits；`stage2_skip=F_fuse`；`stage3_and_deeper_skip=shared_encoder_skip`；
- M1–M4 使用**完全相同**的 `Q_l` 拓扑/初始化/通道（`identical_across_m1_m4=true`）。

运行时守卫 `assert_shallow_skip_inputs_allowed(names)` 供未来 `Q_l` 调用；P2A **只提供接口 + 契约 + 测试**
（`tests/unit/test_skip_contract.py`），**不实现** `Q_l` / `F_fuse` / gate / M1–M4。

> **后续更新（2026-09-17/18）**：上述「不实现」描述的是 **P2A 当时的范围**——`Q_l` / `F_fuse` / gate /
> M1–M4 后续已在 `models/fusion_blocks.py`、`models/fusion_multi_branch.py` 中实现（代码 + 合成测试；
> **未训练**）；实现顺序与协议条款的关系见 `docs/protocol_changelog.md` 2026-09-17/18 条目，
> 当前状态见 `docs/STATUS.md` §2。本文件的参数量/哈希等 M0 事实不受影响。
契约漂移（改拓扑、削弱禁止项、改 contract_id）在 `validate_shallow_skip_contract` 处**报错**。

## 8. 文件与身份隔离

**新增（v2.3 独立身份）**：

- `src/zonal_reliability_fusion/models/residual_blocks.py`（`ResidualBlock3d` / `StackedResidualBlocks3d`）
- `src/zonal_reliability_fusion/models/residual_unet3d.py`（`PlanDrivenResidualEncoderUNet3D` / `zero_init_residual_last_norms`）
- `src/zonal_reliability_fusion/models/m0_residual_concat.py`（`M0ResidualConcatModel` / `FORBIDDEN_INPUTS`）
- `src/zonal_reliability_fusion/models/profiling.py`（参数量 / 解析 MACs / hook MACs / 激活 / 结构摘要）
- `src/zonal_reliability_fusion/models/factory.py`（`build_m0_model` 按配置分派 / `model_architecture_identity` / `resolve_architecture_spec`）
- `src/zonal_reliability_fusion/config/architecture.py`（`ArchitectureSpec` / 哈希 / §8.6 契约 / 身份校验）
- `configs/architectures/m0_resenc_v23.yaml`（版本化架构配置）
- `configs/experiments/m0_resenc_picai_3d_fullres_v23.yaml`（v2.3 实验配置，独立 run/output 身份）
- `scripts/train/summarize_m0_resenc_architecture.py`（结构摘要 / 预算分析；`--no-progress` / `--reduced-forward-check`）
- 测试：`tests/unit/test_{residual_blocks,residual_unet3d,architecture_config,skip_contract,profiling,m0_residual_concat,residual_checkpoint_resume,train_m0_resenc_script}.py`

**最小附加修改（不破坏 legacy）**：

- `config/experiment.py`：新增**可选** `architecture: ArchitectureRef|None`（缺省 None → legacy 行为不变）
- `config/__init__.py` / `models/__init__.py` / `training/__init__.py`：导出新符号（保留全部旧导出）
- `training/trainer.py`：`FROZEN_CONFIG_PREFIXES` 新增 `"architecture."`、`"provenance.architecture_sha256"`
- `training/checkpointing.py`：新增 `verify_checkpoint_architecture`
- `scripts/train/train_m0.py` / `validate_m0_setup.py`：改用 `build_m0_model` 工厂；resume 前架构守卫；打印架构身份
- `tests/conftest.py`：新增 mini/odd residual plan、spec 工厂与 `mini_resenc_model` 夹具；`write_plan` 建父目录

**保留不动（legacy，未改写/未删除）**：`PlanDrivenUNet3D`、`M0ConcatModel`、`blocks.py`、`unet3d.py`、`m0_concat.py`、
`configs/experiments/m0_picai_3d_fullres.yaml`、旧测试、旧输出目录。

## 9. §9.4.2 G2 前置完成情况（1–7 ✅，8 ⏳）

| # | 项 | 状态 | 证据 |
|---|---|---|---|
| 1 | residual block（identity/通道投影/stride/奇数/He 初始化） | ✅ | `test_residual_blocks.py`（12 项） |
| 2 | 版本化架构配置（冻结 blocks/激活序/shortcut/初始化/预算/哈希） | ✅ | `configs/architectures/m0_resenc_v23.yaml` + `config/architecture.py` + `test_architecture_config.py` |
| 3 | encoder/decoder stage shape、DS 顺序与冻结 plan 一致 | ✅ | `test_residual_unet3d.py`（stage shape/DS）+ 真实 plan 结构摘要 artifact |
| 4 | §8.6 浅层 skip **接口契约**（边界/通道/禁读 PZ-TZ-WG-gate/schema） | ✅（仅接口） | `SHALLOW_SKIP_CONTRACT` + `test_skip_contract.py`；**未实现** Q_l/F_fuse |
| 5 | 参数量/可训练/FLOPs/峰值显存估计/结构摘要 | ✅（显存=分析估计+NOT_MEASURED） | `models/profiling.py` + summarize 脚本 + artifact + `test_profiling.py` |
| 6 | 架构名/block 配置/哈希纳入 checkpoint/resume；legacy 明确拒绝；禁 strict=False | ✅ | `verify_checkpoint_architecture` + trainer 冻结前缀 + `test_residual_checkpoint_resume.py` |
| 7 | 合成输入上 forward/backward/AMP/DS/滑窗 shape/resume 回归（不读真实数据） | ✅ | `test_residual_unet3d.py` + `test_residual_checkpoint_resume.py` |
| 8 | G0-R 冻结预处理身份后，研究者执行真实 loader smoke / small-overfit / 诊断性首次全体积验证 | ⏳ **Pending** | 本轮未运行（属 P2B/G2） |

**G2 = NOT YET EVALUATED**（第 8 项未做）；G0-R/G0-E/G0-SAP 仍 Pending/DRAFT，本轮**未**改任何门为 PASS。

## 10. 测试与静态检查（真实结果）

```bash
cd /opt/data/private/lm/my-projects
# 全量单元测试（合成 CPU；不读真实医学数据、不使用 GPU）
PYTHONPATH="$PWD/third_party/nnUNet:$PWD/src" CUDA_VISIBLE_DEVICES="" \
  /root/anaconda3/envs/lm/bin/python -m pytest -p no:cacheprovider tests/unit -q
# → 456 passed, 7 warnings（既有 321 项无回归 + P2A 两轮共新增 135 项，含验收轮白名单/语义测试）

# 新增/重写文件 ruff（All checks passed）
/root/anaconda3/envs/lm/bin/python -m ruff check <新增文件…>   # → All checks passed!
```

新增测试文件与覆盖项见 §8；全量 `tests/unit` 由 321 → **456 passed / 0 failed**（含验收轮新增的白名单/shortcut 语义测试）。

## 11. P2B（研究者运行；本轮**未执行**）命令草案

> 前提：G0-R 已冻结预处理身份；在真实 GPU 上执行；每条命令均须 `conda activate lm` 且涉及 nnU-Net 时
> `source scripts/env_nnunet.sh`。**先在真实 GPU 上核实峰值显存**（§4 风险）。

```bash
cd /opt/data/private/lm/my-projects
conda activate lm
source scripts/env_nnunet.sh
CFG=configs/experiments/m0_resenc_picai_3d_fullres_v23.yaml

# A) setup 检查（只读 plan/split/arch，构建 M0，打印参数量/架构身份；不读病例、不 forward、不用 GPU）
CUDA_VISIBLE_DEVICES="" python scripts/train/validate_m0_setup.py --config "$CFG"

# B) loader smoke（真实数据契约检查；不训练、不 forward、不用 GPU）
CUDA_VISIBLE_DEVICES="" python scripts/train/train_m0.py --config "$CFG" --loader-smoke --limit-cases 2

# C) small-overfit（少量阳性病例，诊断随机 patch；独立 diagnostic 目录）
CUDA_VISIBLE_DEVICES=0 python scripts/train/train_m0.py --config "$CFG" --overfit \
  --cases <阳性case_id,...> --iterations 60 --epochs 1 --device cuda

# D) 诊断性首次全体积验证 / 正式训练（显存核实后；固定 200-epoch、每 5 epoch 全体积验证）
CUDA_VISIBLE_DEVICES=0 python scripts/train/train_m0.py --config "$CFG" --device cuda
```

成功判据：A 打印 `encoder_type=residual_encoder` 与 `architecture_sha256=8bb5127e…`、参数量 148,193,644；
B 输出 seg_labels⊆{0,1}、patch==plan、通道=3、无 NaN/Inf；C loss 下降（诊断，不代表有效）；
D 显存不溢出、正常产生 checkpoint 与全体积 Dice。**任一真实结果只写入 `experiment_log.md`（实际运行后）**。

## 12. 明确声明

本轮 P2A：**未读取任何真实医学影像体素**（`.b2nd`/`.pkl`/NIfTI）；**未使用 GPU**（全程 `CUDA_VISIBLE_DEVICES=""`）；
**未训练 / 未推理 / 未评测 / 未 loader smoke / 未 small-overfit / 未全体积验证**；未使用外部数据；
未修改 `third_party/`、原始/物化数据、split、nnU-Net plan；未覆盖/删除 legacy PlainConv M0 代码/配置/产物；
未把 G2 / G0-R / G0-E / G0-SAP 标记为 PASS。所有数值来自合成张量、只读 JSON/YAML 与解析计算。
