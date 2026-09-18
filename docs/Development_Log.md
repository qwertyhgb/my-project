# 开发日志

按日期记录代码变更、验证过程、已知问题和修复结果。

## 2026-09-14

- 初始化 `docs/research_plan.md`，将研究主线固定为“基于解剖分区条件自适应多序列融合的跨中心前列腺癌病灶分割”。
- 采用官方 nnUNetv2 强基线与自研 nnU-Net-inspired 3D U-Net 双轨方案。
- 固定 N0 与 M0–M5 的容量控制、解剖条件和可靠性训练消融矩阵。
- 明确 PI-CAI 为第一阶段唯一主数据，WG/PZ/TZ 使用冻结的现有 AI 先验，不先训练分区网络。
- 明确“可靠性自适应”需由受控序列退化和未见中心结果验证后才能作为正式贡献。
- 旧的残余错位/局部对应融合方向不再作为本项目研究主线。

### G0：PI-CAI 模型就绪数据审计（本日新增）

**新增代码（核心）**
- `src/zonal_reliability_fusion/data/picai.py`：统一数据访问/几何/标签模块。几何五元组读取、SimpleITK index-to-physical-point 物理角点与包围盒（非 size×spacing 简化）、orientation 编码、五类网格关系分类（exact_same_grid / same_physical_space_different_grid / partial_physical_overlap / geometry_suspect / missing_or_unreadable）、canonical 病灶来源选择（Pooch25 优先否则 resampled）、标签二值映射（resampled 2–5→1、Pooch25 1→1）、物理坐标重采样（影像线性/标签最近邻）、患者泄漏检查。
- `scripts/data/build_picai_manifest.py`：1500 study 统一 manifest（含路径/存在性/canonical 来源/几何/标签唯一值/网格状态/异常原因）+ 预期数字验证 JSON。
- `scripts/audit/audit_picai_geometry.py`：全量物理几何审计（10500 行=1500×7）。
- `scripts/audit/audit_picai_zones.py`：腺体/分区先验全量审计（WG/Yuan23/HeviAI23，重采样到 T2W 网格后算 Dice/体积/连通域/越界，支持 --resume）。
- `scripts/audit/generate_picai_qc.py`：物理空间 QC 图（先重采样再叠加；含 index vs physical 对照）。
- `scripts/data/create_picai_splits.py`：患者级 train/val 划分（center×csPCa 分层，无泄漏；**测试由外部数据集承担**）。
- `tests/conftest.py`、`tests/unit/test_picai.py`：18 个单元测试（标签映射/物理角点/患者泄漏/几何分类/分区标签值/canonical 来源）。

**修复**
- `scripts/check_registration.py`：`extent` 由 `size*spacing` 改为 SimpleITK index-to-physical-point 的 8 角点包围盒。
- `scripts/overlay_qc.py`：由“仅按数组索引直接叠加”改为“物理空间重采样叠加 + index 空间对照（明确 index 叠加仅限同网格/数组空间）”。
- `scripts/data/build_picai_manifest.py`：修复 bug——影像序列(adc/hbv)的 `_grid` 列原被硬编码为 `exact_same_grid`，改为相对 T2W 的真实 `classify_grid`。
- `docs/Data_Analysis.md`：私有 BPH/PCA 表述去推断化（“良性增生区/癌灶/金标准”→“语义待确认的二值 ROI”；删除“目录 ID 无交集=无重复患者”“必然同一机构/域偏移较小”；129.8 mL 改为“异常大且语义存疑，需 WG/人工证据确认”）。
- `docs/research_plan.md`：公式定界符由 `\[ \]` / `\( \)` 改为 `$$ $$` / `$ $`，修复在 GitHub/VS Code 等 Markdown 渲染器中的公式显示异常（仅格式修复，公式内容不变）。

**验证（详见 docs/PICAI_Model_Readiness_Audit.md 与 experiment_log.md）**
- manifest 预期数字全部通过：1500/1476/1075/425/220/205/RUMC 800(236)/PCNN 350(109)/ZGT 350(80)，all_pass=True。
- geometry 审计：无缺失/不可读文件；ADC/HBV 与 T2W 为 same_physical_space_different_grid（需重采样）；HeviAI23 分区 346 例 partial；5 例 ADC/HBV geometry_suspect。
- splits：train/val 按患者划分、无泄漏（测试由外部数据集承担）。
- 单元测试 18 passed。

### P0A 审计修正 + P0B 数据物化（v2.0，本日新增）

**新增/修改代码**

- `src/zonal_reliability_fusion/data/picai.py`：`orientation_code` 改用 SimpleITK 官方接口
  `DICOMOrientImageFilter_GetOrientationFromDirectionCosines`（单位 direction → `LPS`；
  旧实现按 RAS 语义手工解释单位阵，已修正）。仅修正方向字符串报告，不重定向影像。
- `tests/unit/test_picai.py`：修正错误断言（`RAS`/`LAS` → `LPS`/`RPS`），新增斜扫 direction
  合法方向码测试；**19 passed**。
- `scripts/data/materialize_picai.py`（新增）：P0B 物化脚本。T2W 参考网格物理重采样
  （ADC/HBV 线性、标签最近邻）、canonical 二值标签、Yuan/Hevi 分区先验、WG（异常例显式禁用）、
  临时文件读回校验后 `os.replace` 原子重命名、`--dry-run/--limit/--cases/--resume/--overwrite/--verify`、
  运行前磁盘估算、逐例失败记录与结构化 summary。
- `scripts/audit/generate_picai_materialized_qc.py`（新增）：物化后 QC 图（T2W 网格叠加 +
  T2W|ADC/HBV 棋盘格 + 覆盖度统计；summary 合并写入支持追加）。
- `scripts/data/create_picai_splits.py`（修改）：输出改为 `data/splits/picai_train_val_split.json`；
  scope 改为 G0-E 口径；新增 `--compare-legacy` 患者/study 一致性校验与多 study 患者单集合复核；
  旧文件保留不改。
- `scripts/data/prepare_picai_nnunet_raw.py`（新增）：nnU-Net raw 数据集（相对 symlink，不复制体素）
  + `dataset.json`（0000=T2W/0001=ADC/0002=HBV；label: lesion）+ 病例映射 + Dataset ID 冲突检查。
- `scripts/data/audit_materialization_report.py`（新增）：物化报告 × manifest 交叉审计
  （覆盖/终态/阳性保留/阴性为空/标签值/精确网格 fg 一致/WG 禁用）。
- `scripts/data/build_picai_manifest.py`、`scripts/audit/audit_picai_geometry.py`：加入 tqdm
  进度条与 `--no-progress` 选项并输出结构化结束摘要（AGENTS.md §1）。
- `docs/project_structure.md`：新增物化数据布局与 nnUNet 工作区说明；空目录
  `outputs/domain_generalization/`、`configs/domain_generalization/` 分别更名
  `outputs/external/`、`configs/external/` 以对齐 v2.0（均为空目录，无产物丢失）。

**归档**

- P0A 前旧版 manifest/validation/geometry 产物已归档至
  `outputs/diagnostics/picai_model_readiness/archive_pre_p0a/`（不删除历史）。

**nnUNet 工作区与环境**

- 新增 `scripts/env_nnunet.sh`：项目内 nnUNet 运行环境（覆盖 `.bashrc` 全局 `nnUNet_*` 变量；
  raw/preprocessed → `workdir/`，results → `outputs/nnUNet_results`；`PYTHONPATH` 前置
  `third_party/nnUNet` 固定 v2.6.2，避免 lm 环境中可编辑安装的其他 nnunetv2 分支遮蔽）。
- nnUNet 数据集迁移至项目内 `workdir/nnUNet_raw/Dataset605_PICAI`；项目外误放副本
  （6000 个 symlink + dataset.json，24 MB）已清理删除，`/opt/data/private/lm/projects/` 整体移除。

**N0 前置完成：`splits_final.json` 与 G0-P 判定**

- `scripts/data/create_nnunet_splits.py`（新增）：把冻结划分（`data/splits/picai_train_val_split.json`）
  转换为 nnU-Net 单 fold `splits_final.json`（fold 0；preprocessed 主文件 + raw 同内容副本），
  含 14 项校验（数量与声明一致、无重复/交集、与预处理 case 集合一一对应、多 study 患者单侧、回读校验）、
  临时文件原子重命名、默认不覆盖（内容一致则幂等跳过）、`--dry-run / --overwrite / --no-progress`。
  实际结果：train 1277 / val 223，14/14 PASS。
- `docs/PICAI_Model_Readiness_Audit.md`：G0-P 由 Conditional Pass 更新为 **PASS**；§1 条件 3/4/6/8 更新为
  完成；§5.4 补记全量物化（1500/1500，0 失败，27.3 GB，审计 all_pass=True）；§8 补记 3d_fullres 预处理
  （1500/1500）与 `splits_final.json`；§10 移除“splits_final 尚未生成”改为 nnU-Net 环境变量注意事项；
  §11 给出最终判定；附录修正 raw 路径为项目内 `workdir/` 并补充预处理/划分产物与新增脚本。
- `docs/experiment_log.md`：补记全量物化完工、全量审计、全量 `--verify`、3d_fullres plan/preprocess 与
  `splits_final.json` 生成（仅记录实际运行命令与结果）。

### P2-A：M0 自研基线工程实现（CODE READY，本日新增）

**新增源码（`src/zonal_reliability_fusion/`，自研，不包装官方模型）**

- `config/plans.py`：`Plan3DConfig` 只读解析 `nnUNetPlans.json`（3d_fullres），校验 conv/norm/nonlin、
  kernel 奇偶、stride、列表长度、patch 各 stage shape ≥1，暴露 spacing/patch/batch/features/kernels/strides
  与 resolved 快照；`config/experiment.py`：M0 配置强类型加载（YAML/JSON）与 plan/split/fold/输出目录冲突校验。
- `models/blocks.py`：`ConvNormAct3d`、`StackedConvBlocks3d`、`center_crop_or_pad_3d`（差异 >2 体素即报错）、
  Kaiming/He 初始化；`models/unet3d.py`：plan-driven encoder/decoder（ConvTranspose(kernel=stride) + skip 对齐 +
  每级 head + DS 输出高→低）；`models/m0_concat.py`：M0 三通道 concat 基线（固定模态语义、无 gate/分区/分支）。
- `data/preprocessed_store.py`：唯一 import `nnunetv2` 的只读适配层（惰性 `NNUNetDatasetBlosc2`），
  data [C,D,H,W]/seg [1,D,H,W] 契约校验；`data/patch_sampler.py`：前景过采样（`class_locations` 优先、
  阴性安全回退）、小图显式 padding、影像/标签同步；`data/transforms.py`：同步镜像（已测试）+
  分序列强度接口（默认关闭）+ rotation/scale/noise/blur/gamma 显式占位（调用即报错）；`data/batch_provider.py`：
  训练/验证 batch 迭代器（验证不启用前景过采样）。
- `training/losses.py`（Dice+CE，全阴性无 NaN）、`training/deep_supervision.py`（1/2^i、最低为 0、归一化、单输出保护）、
  `training/metrics.py`（前景 Dice、TP/FP/FN）、`training/optim.py`（SGD+Nesterov、PolyLR 状态可续训）、
  `training/checkpointing.py`（原子写入、last/best/周期/interrupt）、`training/trainer.py`（显式 device、
  AMP 仅 CUDA、两级进度条、CSV/JSONL 日志、resume 连续）。
- `inference/sliding_window.py`（Gaussian 融合、覆盖断言、DS 主输出、batch、可选镜像 TTA 默认关闭）；
  `utils/progress.py`（默认开启、可关闭、仅主进程）、`utils/reproducibility.py`（seed/RNG 状态/环境快照/
  split 与 plan 的 SHA256/代码版本，无 git 时记录 unavailable）。

**新增脚本与配置**

- `scripts/train/validate_m0_setup.py`（命令 A：只读 plan/split，构建 M0 并打印参数量与 stage shape，不读病例）；
- `scripts/train/train_m0.py`（命令 B `--loader-smoke` 真实数据契约检查；命令 C `--overfit` 独立 diagnostic
  目录；默认模式为正式训练）；
- `configs/experiments/m0_picai_3d_fullres.yaml`（全部超参与 plan 对齐，加载时校验冲突）。

**新增测试（合成 CPU，不读真实数据）**

- `tests/unit/test_{plans,m0_model,losses,patch_sampler,sliding_window,checkpointing,progress,batch_provider,trainer_synthetic}.py`；
  `tests/conftest.py` 增加 mini plan / 模型夹具并强制测试环境不可见 GPU。

**未做的事**：未实现 M1–M4 与任何 PZ/TZ gate；未读取真实病例；未启动 CUDA/训练；未修改 `third_party/`。

**复审修复（同日第二轮）**

- `training/trainer.py`：构造时显式 `model.to(device)`（修复 CUDA 训练阻塞问题），manifest 增加
  `model_parameter_device`；新增 `data_sources`（每轮 `set_epoch(epoch)`，保证采样序列可复现且无隐藏 RNG 状态）。
- `data/patch_sampler.py`：新增 `normalize_class_locations`，兼容官方 `class_locations` 的 **(n,4)** 格式
  （首列标签索引，后 3 列为坐标）；前景 patch 改为官方"以体素为中心"取法；越界改平移 bbox。
- `data/batch_provider.py`（重写）：数据加载改走 `torch.utils.data.DataLoader`，`num_workers`/`pin_memory`
  配置项真实生效；新增 `PatchDataset`（`(seed, epoch, index)` 确定性采样）；前景请求改为官方
  "每批固定 `bs − round(bs·(1−oversample))`"语义。
- `training/losses.py`：`SoftDiceLoss` 返回 `−Dice`（与官方 `SoftDiceLoss.forward` 一致，去掉 +1.0 偏移），
  分母 `clip(…, 1e-8)`；`dice_score()` 提供正 Dice；`components()` 同步调整。
- `inference/sliding_window.py`：滑窗耗时循环（含 TTA 每个翻转）接入 `make_progress`。
- `config/experiment.py` + `configs/experiments/m0_picai_3d_fullres.yaml`：新增 `AugmentationConfig`
  与 `augmentation.pending_parity`（显式冻结尚未对齐的增强差异，校验时 warning 提示）；
  `transforms.build_train_transforms` 支持从配置开启分通道强度增强。
- `scripts/train/train_m0.py`：loader smoke 的病例与 batch 循环加进度条；训练/验证 provider 接入
  `num_workers`/`pin_memory` 与配置化增强；trainer 注册 `data_sources`。
- 测试：新增 8 项回归测试（模型设备、`(n,4)` class_locations、居中取 patch、每批前景数、epoch 采样可复现、
  多进程 DataLoader、滑窗进度开关、CE−Dice 等值性）；`tests/unit` 全量 121 passed。
- `docs/research_plan.md`：§6/§14 的 `G0-P = Conditional Pass` 修正为 **PASS**（与审计报告一致的最小同步）。
- `docs/P2_M0_Implementation.md`：状态改为 IMPLEMENTED / FIX APPLIED（待复审），新增 §15 复审问题与修复。

### P2-A3：每 epoch 全体积验证 + Dice 选择 best + early stopping（本日新增）

**新增源码**

- `src/zonal_reliability_fusion/evaluation/__init__.py`、`full_volume_validator.py`：`FullVolumeValidator`
  遍历 fold 0 全部 validation study（223 例）做完整 3D 滑窗推理；输出阳性 case-wise Dice 的 mean/median、
  micro Dice、阴性 FP 统计与耗时；病例级 CSV 写入 `<metrics_dir>/validation_cases/epoch_XXXX.csv`；
  硬校验覆盖/重复/遗漏与 223/63/160 计数、预测 NaN/Inf；推理期间关闭 deep supervision 并恢复 DS 与 train/eval 状态。
- `training/trainer.py`（扩展）：两种验证模式（`full_volume` 正式 / `diagnostic_patch` 诊断，兼容别名
  `formal_full_volume`）；**按显式字段名**取 checkpoint 指标（缺失/非有限立即报错）；early stopping
  （min_epochs/patience/min_delta，与 checkpoint 同指标同 min_delta）；checkpoint 额外保存并恢复
  `checkpoint_metric/best_metric/best_epoch/early_stopping_bad_epochs/min_epochs/patience/min_delta/validation_protocol`；
  `early_stop_summary.json`；正式模式拒绝 `val_batches`。
- `config/experiment.py`：新增 `ValidationConfig`、`EarlyStoppingConfigFile`，`training.max_epochs` 与
  `diagnostic_validation_iterations`（旧字段名兼容）；加载时执行验证协议合法性校验（与 validator 共用同一实现）；
  `validate_against_plan` 增加"expected_cases 必须等于 split val 数""正式模式必须启用 early stopping"等检查。
- `configs/experiments/m0_picai_3d_fullres.yaml`：`max_epochs: 200`、`validation`（full_volume、223/63/160、
  step_fraction 0.5、Gaussian、mirror TTA 关闭、sliding_window_batch_size 1、inner_patch_progress false）、
  `early_stopping`（50/30/1e-4）。
- `scripts/train/train_m0.py`：`build_validation_components`（正式模式只建 validator、诊断模式只建随机 patch provider）；
  `ensure_fresh_run_dirs`（正式 run 目录非空且未 resume → 拒绝启动）；正式训练按 Dice 选择 best 并支持 resume。
- `scripts/train/validate_m0_setup.py`：setup 检查打印并记录验证协议与 early stopping 快照。

**新增测试（合成 CPU；不读真实病例、不初始化 CUDA）**

- `tests/unit/test_full_volume_validator.py`：覆盖/重复/遗漏、阳性完美 Dice=1、阳性空预测 Dice=0、
  阴性空—空不进入主指标、阴性 FP 统计、micro Dice、病例级 CSV、DS/train 状态恢复、非有限预测报错、进度开关、
  协议规则校验；
- `tests/unit/test_train_m0_script.py`：run 目录安全、正式/诊断验证组件装配（正式模式不创建随机 patch provider）、
  仓库配置协议自检；
- `tests/unit/test_trainer_synthetic.py` 追加：Dice 选 best、min_delta 抑制更新、early stopping 不早于 min_epochs、
  patience 触发停止、resume 恢复 early-stop 计数与 best、指标缺失/非有限报错、模式参数校验；
- `tests/unit/test_plans.py` 追加：验证协议配置的加载期校验（指标不得为 val_loss、计数关系、every_n_epochs=1、
  step_fraction 范围、min_epochs<=max_epochs、expected_cases 与 split 一致、诊断模式告警）。

**修复的缺陷**：`load_experiment_config` 中配置对象构造写成了 `return`，导致其后新增的协议校验成为不可达代码
（合成测试暴露，已改为 `cfg = ... ; validate_validation_protocol(cfg) ; return cfg`）；验证模式命名统一为
`full_volume`/`diagnostic_patch`（保留 `formal_full_volume` 别名）。

**未做的事**：未读取真实病例、未使用 GPU、未运行训练或真实验证；未修改 `third_party/`。

**P2-A3 复审修复（续训安全性，同日第三轮）**

- `training/checkpointing.py`：新增统一 payload 构造 `build_checkpoint_payload`，payload 增加
  `epoch_completed` / `completed_epochs` / `iterations_completed`；`load_checkpoint` 一并返回；
  `interrupt` 仍只写 `checkpoint_interrupt.pth`（不覆盖 last/best）。
- `training/trainer.py`：
  - 训练循环记录 `current_epoch` / `current_iteration` / `current_phase`，Ctrl-C 时写出准确的
    "中断时正在执行 epoch E（阶段=…），该 epoch 已完成 K 个 iteration，此前完整完成 N 个 epoch"；
  - `resume()` 新增 `allow_mid_epoch` 与 `allow_protocol_change`：协议字段（`checkpoint_metric` /
    `maximize_metric` / `early_stopping_enabled` / `min_epochs` / `patience` / `min_delta` /
    `validation_protocol`）不一致或缺失默认**拒绝续训**；中断 checkpoint 默认拒绝，显式允许时
    `start_epoch = completed_epochs`（重跑被打断的 epoch）；
  - `train()` 在 RNG 来自 checkpoint 时不再 `set_seed`；
  - 正式模式额外强制 `maximize=True`。
- `evaluation/full_volume_validator.py`：`full_volume` 模式的 checkpoint 指标与方向**硬冻结**
  （仅 `val_positive_casewise_dice_mean` + `maximize=True`），其余字段只作记录。
- `scripts/train/train_m0.py`：新增 `--allow-mid-epoch-resume`、`--allow-protocol-change`，并在脚本
  头部安全约定中说明中断/续训语义。
- 测试：`test_trainer_synthetic.py` 新增/改写 4 项（中断语义与 last/中断两种续训路径、协议变化拒绝与显式放行、
  RNG 不被覆盖、续训状态连续）；`test_full_volume_validator.py` 与 `test_plans.py` 增加指标/方向冻结的拒绝用例。
  全量 163 passed（P2 测试文件 144 项）。
- 文档：`P2_M0_Implementation.md` 修正 `best(val_loss)` 过期表述并新增 §17「续训安全性」；本文件与
  `experiment_log.md` 记录本轮修复（仍只使用合成 CPU 测试）。

**P2-A3 复审第二轮修复（续训边界，同日晚）**

- `training/trainer.py`：
  - 新增 `_epoch_offset`（本次进程开始前已完整完成的 epoch 数，`resume()` 时置为 `start_epoch`），
    中断时 `completed_epochs = _epoch_offset + len(self._history)` —— **绝对计数**，修复 resume 后再中断
    时的 epoch 漂移；正常保存路径写入 `completed_epochs = epoch + 1`；
  - `_verify_resume_protocol` 返回"协议是否变化"；显式 `allow_protocol_change=True` 放行时**重置**
    `best_metric / best_epoch / bad_epochs`（旧指标在新协议下不一定可比）并打 WARN；
  - `epoch_completed=True` 的 checkpoint 增加 `completed_epochs == epoch + 1` 一致性校验。
- `tests/unit/test_trainer_synthetic.py`：新增 `test_interrupt_after_resume_records_absolute_completed_epochs`
  （进程 A 在 epoch 4 中断 → 4；续训后在 epoch 5 再中断 → 5；再续训 → `start_epoch=5`），并扩展
  `test_resume_rejects_protocol_change`（重置后从续训点重新选模）。全量 **164 passed**（P2 145 项）。
- `scripts/train/train_m0.py`：`--allow-protocol-change` 帮助文本说明会重置选模状态。
- 文档：`P2_M0_Implementation.md` §17.1（绝对计数）/§17.2（重置 + 一致性校验）。

**P2-A3 复审第三轮修复（checkpoint 元数据 / 配置冻结 / 日志对齐）**

- `training/checkpointing.py`：`build_checkpoint_payload` 统一补全 `completed_epochs`（完整 epoch 缺省填
  `epoch + 1`）；`CheckpointManager.save` 把同一份 epoch 元数据传给 **last / best / 周期** 三类文件
  （此前只有 last 带，best/周期为 None）。
- `training/trainer.py`：
  - `_verify_resume_protocol` → `_verify_resume_consistency`：除协议字段外，对 resolved 配置快照做扁平化
    逐项比对（新增 `FROZEN_CONFIG_PREFIXES` / `EXTENDABLE_CONFIG_PATHS` / `EXTENDABLE_CONFIG_SUFFIXES`
    与 `_flatten_mapping`）；冻结 seed、模态顺序、patch/batch、增强、loss、optimizer、iterations、
    plan 解析与 plan/split 哈希；`training.max_epochs` 允许显式延长，吞吐/遥测字段仅打印；
  - 新增 `last_logged_epoch()` 与 resume 的日志对齐校验（续训起点必须 = CSV 最后 epoch + 1），
    防止从旧 checkpoint 恢复到同一 run 造成重复/错序 epoch 行与病例 CSV 覆盖；
  - `resume()` 文档与拒绝提示同步更新。
- `scripts/train/train_m0.py`：`--resume-from` 帮助文本说明协议/配置冻结与日志对齐检查。
- 测试：`test_checkpointing.py` 新增元数据一致性测试；`test_trainer_synthetic.py` 新增配置冻结
  （4 个分支）与日志对齐（2 个分支）测试，并让夹具真正透传 `config_snapshot`。全量 **167 passed**。
- 文档：`P2_M0_Implementation.md` §17.1（三类 checkpoint 元数据一致）/§17.2（完整配置冻结）/
  新增 §17.4（日志对齐，原冻结指标节顺延为 §17.5）。

## 2026-09-15

**协作规则更新（用户指定）与文档同步**

- `AGENTS.md`：
  - §2 由「数据处理与训练命令由用户运行」扩展为「数据处理、训练与长时命令由用户运行」：明确验证
    （validation/verify）、推理、评测、批量分析等预计耗时较长的命令同样不得由代理自行启动，必须整理为命令
    交由用户运行；代理可自行执行的仅限秒级只读诊断（读取配置、解析 plan/split、`--help`、静态与语法检查）；
    `--limit` / smoke test / 单病例 / `--epochs 1` 同等受限；命令归属不明确时默认交由用户运行；
  - 新增 §4「运行环境与依赖」：4.1 所有命令统一在 conda 环境 `lm`（`/root/anaconda3/envs/lm`）下运行，代理执行前
    须确认解释器路径，nnU-Net 命令先 `source scripts/env_nnunet.sh`；4.2 缺失依赖由代理自行安装到 `lm`
    （`python -m pip install` 或 `conda install -n lm`），禁止装到 base/用户目录，安装后验证版本并记录，
    未经确认不得升降级 nnunetv2/torch/numpy/SimpleITK 等既有依赖；
  - §1（进度）与 §3（数据安全）编号保持不变，避免破坏既有 `AGENTS.md §1` 引用。
- 同步文档：
  - `docs/research_plan.md` §16.2「命令执行边界」纳入长时验证/推理/评测命令、`lm` 环境与命令归属兜底条款；
    新增 §16.3「运行环境与依赖」（环境统一、缺包安装与记录、禁止破坏既有依赖）。
  - `docs/project_structure.md`「开发与运行约定」第 2 条扩展为涵盖长时验证/推理命令，新增第 3 条环境统一为
    conda `lm` 与缺包安装约定。
- 本轮（规则更新）仅文档变更：未修改代码、未运行任何数据处理/训练/推理/评测命令。

**研究计划状态同步（同日；用户指出「近期唯一下一步」滞后）**

- 滞后事实：`docs/research_plan.md` §20 仍写「先完成 P0A审计修正 + P0B训练数据物化」，但 P0A/P0B 已于
  2026-09-14 完成并记录，N0 官方基线训练亦已于 2026-09-15 01:17 由研究者启动。
- `docs/research_plan.md` 同步（版本号保持 `v2.0`，仅作状态同步并标注日期；未改动任何协议条款）：
  - 抬头「当前阶段」更新为：P0A/P0B 已完成、G0-P = PASS、N0 训练进行中（fold 0）、M0 待真实数据验证；
  - §6 补记 N0 已启动及其实际运行协议（官方默认 `num_epochs=1000`、`save_every=50`），并列出正式比较前
    必须显式冻结的两项：① N0（1000 epoch）与自研 M0–M4（200 epoch）的训练预算差异；
    ② N0 checkpoint 使用规则（固定 `checkpoint_final.pth`，禁用 `--val_best`）；
  - §14 G1 补「当前状态：进行中」；
  - §15 实施顺序标注 P0A/P0B 已完成、P0C 待办、P1 进行中、P2 代码 CODE READY；
  - §20 由「P0A+P0B」改写为当前下一步：等 N0 完成 → `--val` 全量验证 → 实现 `scripts/evaluate/`
    （picai_eval 口径）→ 冻结两项实验差异 → 进入 P2 的 M0 真实数据三步验证。

**P1：N0 训练启动记录**

- `docs/experiment_log.md` 新增 2026-09-15 条目：N0 启动时间与日志路径、`debug.json` 实际协议、
  划分来源（1277/223）、GPU 与 epoch 耗时观察、`Pseudo dice=0.0` 的成因与「不作为判据」结论、
  `checkpoint_best.pth` 为 epoch 0 权重及「最终评测固定用 `checkpoint_final.pth`」的处置、待决的训练预算差异。
- 代码未改动；训练由研究者执行（**状态更正**见本日「P1：PI-CAI official-style Focal+CE 的 nnU-Net v2 适配」条目：
  该默认 Dice+CE 预实验已于 01:58 在 Epoch 43 中断并被排除，不再作为正式 N0 baseline）。

**其他文档滞后同步（同日）**

- `README.md`：主线表述由「跨中心…可靠性自适应」改为 v2.0 口径（解剖分区条件自适应的 T2W/ADC/HBV 多序列融合）；
  末段过期的「项目刚完成目录初始化；研究方案、模型实现和实验配置尚未冻结」改为 2026-09-15 现状
  （G0-P PASS、N0 训练进行中、M0 CODE READY 待真实数据验证、G0-E Pending）。**H1 标题
  `Cross-Center Prostate Lesion Segmentation` 保留未改**（涉及命名决策，待用户确认）。
- `docs/PICAI_Model_Readiness_Audit.md` §11：在「N0 启动条件已满足」条目下追加 2026-09-15 状态更新
  （命令已执行、训练进行中、`Pseudo dice` 长期为 0 的成因与 `checkpoint_final.pth` 冻结规则），
  原结论不改写、不删除。
- 过程记录：本轮对 `docs/research_plan.md` 的多处**并行**编辑曾出现 §6/§15 未落盘（写入丢失），
  已逐条重做并复核（§6/§14/§15/§20 与抬头均已确认在位）；后续对同一文件的多处修改应串行执行。
- **状态更正（同日稍后）**：上述三条状态描述（`research_plan` 抬头、`README.md` 末段、审计报告 §11）
  已在本日后续两轮同步中被更新——默认 Dice+CE run 已终止并**排除**，正式 N0 重新定义为
  `PI-CAI official-style nnU-Net v2 Focal+CE NoFFT`（尚未启动真实训练）。本文件与 `experiment_log.md` 中
  较早的条目保留原样，仅作过程记录，不代表当前状态。

**P1 排障：N0「Pseudo dice 长期为 0」只读诊断工具（本日新增）**

- 新增 `src/zonal_reliability_fusion/integrations/nnunet_dataloader_probe.py`：`integrations/` 内的 nnU-Net
  **只读适配层**（全部第三方依赖惰性 import，模块导入不触发 nnunetv2）。含纯函数（patch 前景统计、官方 FG
  过采样标志复刻、class_locations 校验、逐例异常判定、聚合、决策树判定）与官方适配（`nnUNetDatasetBlosc2`
  只读打开、官方 `nnUNetDataLoader` 构造、官方 `get_training_transforms` 增强管道 + 参数复刻交叉校验）。
- 新增 `scripts/train/diagnose_n0_zero_dice.py`：病例 / 采样 / 增强三级只读诊断入口。支持
  `--split train|val|both`、`--max-cases`、`--max-batches`、`--with-augmentation`、`--seed`、`--output`、
  `--no-progress`、`--fail-on-anomaly`；输出 `static_audit.json`、`training_log_audit.json`、
  `case_audit.csv(+summary)`、`batch_probe.csv(+summary)`、`verdict.json`、`run_summary.json`，默认写入
  `outputs/diagnostics/n0_zero_dice/<timestamp>/`。只读 `.b2nd`/`.pkl`，不写回、不触碰 `nnUNet_results/`。
- 新增 `tests/unit/test_n0_zero_dice_diagnosis.py`（36 项；合成数组 + tmp_path 元数据；不读真实数据、
  不 import nnunetv2、不使用 GPU）。
- 环境变更（AGENTS §4.2）：`lm` 环境缺少 pytest，已安装 **pytest 9.1.1**（附 iniconfig 2.3.0 / pluggy 1.6.0 /
  tomli 2.4.1）并验证 `import pytest` 与版本；全量 `tests/unit` 现为 **203 passed / 0 failed**。

**静态审计结论（源码/元数据证据；供后续引用）**

1. 运行中训练进程（PID 11498）环境：`CONDA_DEFAULT_ENV=lm`、
   `PYTHONPATH=/opt/data/private/lm/my-projects/third_party/nnUNet:/opt/data/private/lm/my-projects/src`、
   `nnUNet_raw/preprocessed/results` 全部指向项目内 → **实际运行的就是固定 v2.6.2 源码树**
   （`lm` 中另有其它项目的可编辑安装 nnunetv2 2.5，被 PYTHONPATH 前置规避）。
2. **v2.6.2 的验证 dataloader 同样启用前景过采样**：`get_dataloaders()` 中 `dl_val` 也传入
   `oversample_foreground_percent=self.oversample_foreground_percent`（0.33）。因此「验证只用随机 patch、
   极少含病灶，故 pseudo dice 恒为 0」这一常见解释**在本项目不成立**。
3. FG 强制语义：`_oversample_last_XX_percent` = `not idx < round(bs*(1-p))`；bs=2/p=0.33 → 每 batch 恰好
   1/2 个 patch 被强制取在前景体素上（bs=1 时为 0 个，属官方行为）。
4. 病例采样为**有放回**均匀抽样（batchgenerators `DataLoader.get_indices` + `infinite=True`）；
   阴性病例被要求前景时**静默回退随机 crop**（`get_bbox` 的 fallback 分支）。
5. Pseudo dice 定义：该 epoch 内 validation patch 的 tp/fp/fn **求和**后 `2tp/(2tp+fp+fn)`，硬标签（argmax），
   `tp_hard[1:]` 去掉背景 → 它是 patch 级指标，**不是全体积 Dice**。
6. 训练 loss = CE − Dice（`DC_and_CE_loss` + `MemoryEfficientSoftDiceLoss`，`do_bg=False`，`batch_dice=False`）；
   实测 loss < 0 ⟹ 训练 patch 目标确含前景、且模型 softmax 已与前景重叠（**数据→标签→损失闭环在训练侧是活的**）。
7. 增强与深监督：spatial p=0.2 旋转 / p=0.2 缩放(0.7–1.4)、`do_dummy_2d_data_aug=True`（±180°）、
   深监督下采样 `order=0`（最近邻）、DS 权重最低层置 0 → 静态未发现「标签被抹掉」的机制（仍需探针实测）。
8. 划分与映射链路：`static_audit` **17/17 通过**（dataset.json 通道/标签、plans、splits 1277/223、
   CSV `case_id=patient_id_study_id`、CSV↔manifest 映射与 csPCa、CSV 集合 == splits_final 集合）。

**冒烟测试（只读元数据与日志）**：`--skip-case-audit --no-progress` → 17/17 静态检查通过；日志审计显示
（**该时刻的运行中途快照**）34 epoch、pseudo dice 非零 4 次、**首次非零在 epoch 22**、train_loss 首值 +0.001 /
末值 −0.0687。该 run 的最终统计为 **43 条 epoch 指标（38 次为 0、5 次非零）**，随后在 **Epoch 43 刚开始时中断**。
产物 `outputs/diagnostics/n0_zero_dice/smoke_metadata_only/`。

- 未做：未修改 `third_party/`；未改动任何训练配置与数据；未执行任何读取真实体素或使用 GPU 的命令
  （真实数据诊断命令已交付用户执行）。

**P1：PI-CAI official-style Focal+CE 的 nnU-Net v2 适配（本日新增）**

- 新增 `src/zonal_reliability_fusion/integrations/nnunet_picai_flce.py`：
  - `PiCAIFocalCrossEntropyLoss`：复刻 PI-CAI 官方公开的
    `0.5 × FocalLoss(gamma=2, alpha=None) + 0.5 × CrossEntropyLoss`（Softmax → one-hot
    clamp(smooth, 1−smooth) → `pt + smooth` → `−(1−pt)^γ log pt`；`alpha=None` 即无类别权重），
    并显式拒绝 ignore 标签、负标签、越界标签与非整数标签；
  - `nnUNetTrainerPICAI_FLCE`：v2.6.2 `nnUNetTrainer` 子类，逐 deep-supervision 尺度替换为上述损失；
    DS 权重沿用 v2.6.2 默认（`1/2^i`，最低分辨率不监督，DDP 分支 1e-6），其余训练行为全部继承默认实现；
  - **运行时兼容性修复**：重写 `get_training_transforms`，把唯一的 `GaussianBlurTransform.benchmark` 置 `False`
    并清空 `benchmark_use_fft`；**blur 概率(0.2)、sigma(0.5–1.0)、适用通道与其余增强均不变**；若未找到唯一
    blur transform 则直接抛错，避免静默回到不稳定的 FFT 路径。该修复位于父类，`NoFFT` 子类继承之；
  - `nnUNetTrainerPICAI_FLCE_NoFFT`：等价子类，**仅提供独立输出身份**，用于与首次失败 run 的目录隔离。
- 新增 `scripts/train/train_n0_picai_flce.py`：单 GPU 启动器。通过替换进程内 `get_trainer_from_args` 注入适配
  trainer，**不修改 `third_party/`**；支持 `--continue-training`、`--validation-only`、
  `--export-validation-probabilities`、`--device`；显式拒绝多 GPU（DDP 需另行实现与验证）。
- 新增 `tests/unit/test_nnunet_picai_flce.py`（合成 CPU）：与公开公式逐值一致（atol 1e-7）并可反传、
  target 无通道维、拒绝 ignore/越界/非整数标签、trainer 继承关系、blur benchmark 被安全关闭。
- 同步的既有事实（不得改写、不得删除产物）：
  - 默认 `nnUNetTrainer`（Dice+CE）预实验于 01:17 启动，日志在 **Epoch 43** 中断且未完成
    （43 epoch 中 38 次 `Pseudo dice = 0`，5 次非零 1e-4~4e-4，首次非零 epoch 22）；仅存 `checkpoint_best.pth`。
    该 run **已排除**，不再作为正式 N0 baseline，只作为「预实验 / 被排除的默认 Dice+CE 方案」保留；
  - 首次 `nnUNetTrainerPICAI_FLCE` 运行在 epoch 0 因 FFT benchmark 崩溃（日志原样保留）；
  - **NoFFT 版尚未在真实数据上启动**（`nnUNetTrainerPICAI_FLCE_NoFFT__nnUNetPlans__3d_fullres/` 尚不存在）。
- 术语：N0 统一表述为 `PI-CAI official-style nnU-Net v2 Focal+CE NoFFT`；**不得**称为逐位官方复现
  （官方为 nnU-Net v1、1295 标注扫描、患者无重叠 5-fold CV + 集成；本项目为 v2、1500 study、单 fold）。
**P2：M0 迁移到 Focal+CE（与正式 N0 对齐；本日新增）**

- `training/losses.py`：新增 `PiCAIFocalCELoss`——复刻 PI-CAI 官方公开的
  `0.5 × Focal(γ=2, α=None) + 0.5 × CE`（Softmax → one-hot `clamp(smooth,1−smooth)` → `pt+smooth` →
  `−(1−pt)^γ·log pt` → mean），并新增严格的 target 校验 `_validate_target_strict`（负/越界/非整数标签显式报错，
  与 N0 适配层同语义）；实现**不依赖 nnunetv2**。`DiceCELoss` 保留，但标注为「与正式 N0 不可比」。
  模块 docstring 与 `__all__` 同步更新。
- `config/experiment.py`：`LossConfig` 重构为显式口径——新增 `name`（`focal_ce` / `dice_ce`）、
  `focal_weight`、`gamma`；`dice_weight`/`include_background` 改为可缺省；新增 `_optional_number` /
  `_optional_bool` 辅助与 `validate_loss_config`（缺参数、两口径混用、未知名称、`ce_weight ≤ 0`、
  `smooth ∉ [0,1)` 均显式报错），并在 `load_experiment_config` 中调用。
- `configs/experiments/m0_picai_3d_fullres.yaml`：`loss` 切换为 `name: focal_ce`
  （`focal_weight 0.5 / ce_weight 0.5 / gamma 2.0 / smooth 1e-5`），并注明 `batch_dice` 仅用于 plan 一致性校验。
- `scripts/train/train_m0.py`：`build_model_and_loss` 按 `cfg.loss.name` 分派（`focal_ce` → `PiCAIFocalCELoss`；
  `dice_ce` → `DiceCELoss`），未知名称报错；DS 包装与权重逻辑不变。
- `training/__init__.py`：导出 `PiCAIFocalCELoss`。
- 测试：
  - 新增 `tests/unit/test_loss_alignment_n0.py`（**23 项**）：M0 损失与 N0 适配层在同一 logits/target 上逐值一致
    （含全背景 target、无通道维 target、梯度有限性）、两者对非法 target 抛同类错误、DS 权重符合 N0 默认公式、
    DS 加权结果与官方 `DeepSupervisionWrapper` 一致、仓库配置声明 `focal_ce`、配置层拒绝 10 组非法组合；
  - `tests/unit/test_losses.py`：导入新增 `PiCAIFocalCELoss`；
  - `tests/unit/test_plans.py` 夹具改为 `focal_ce`；`tests/unit/test_train_m0_script.py` 夹具显式标注 `dice_ce`
    （保留历史口径的加载覆盖）。
- 验证（合成 CPU，不读真实数据、不使用 GPU）：`tests/unit` 全量 **233 passed / 0 failed**；并已用仓库配置实际加载验证
  `loss.name=focal_ce` 与 resolved 快照可追溯（`dice_weight=None`、`include_background=None`）。
- 未做：未修改 `third_party/`；未改动数据/split/plan；未运行 loader smoke、small-overfit、真实训练或全体积验证
  （G2 仍为 NOT YET EVALUATED）。M1–M4 复用同一损失实现，不得引入第三种口径。
- 文档遗漏修正（用户指出，同日）：
  - `docs/P2_M0_Implementation.md` §10「与官方 N0 的异同（当前）」对照表的「损失」行仍写
    `Dice+CE（batch_dice=False）`，与迁移后实现不符 → 更正为
    `0.5×Focal(γ=2, α=None) + 0.5×CE`（`PiCAIFocalCELoss`，与 N0 逐值对齐，测试锁定）；其余各行的
    差异项（增强仅镜像、训练框架不同等）经复核仍然成立。
  - `docs/experiment_log.md` P1 预实验条目里的 `checkpoint_best.pth` 描述（原写「为 epoch 0 权重」）已过期：
    实测 `Pseudo dice` 自 epoch 22 起非零，best 于 01:44（≈epoch 25）被 EMA 更新 → 更正为「首次 epoch 0
    无条件保存，此后 EMA 改善才覆盖」，并保留「最终评测固定用 `checkpoint_final.pth`、禁用 `--val_best`」结论。
  - 验证集前景过采样的说明：`docs/experiment_log.md` L210–215、`docs/PICAI_Model_Readiness_Audit.md` §11
    已在早前同步中更正为「validation dataloader 同样启用 0.33 前景过采样」；本轮全库扫描确认**无残留**
    相反表述（`P2_M0_Implementation.md`/`research_plan.md` 中出现的「随机 patch」均指自研 M0 的诊断模式或
    N0 训练时的 patch 采样，语境正确）。

- 测试稳健性修复：`tests/unit/test_n0_zero_dice_diagnosis.py::test_adapter_import_does_not_import_nnunetv2`
  原先直接断言进程内 `sys.modules` 不含 nnunetv2，在按文档化命令（`PYTHONPATH=third_party/nnUNet:src`）运行全量
  `tests/unit` 时会被其它测试文件的 import 污染而**假失败**；已改为**在独立子进程**中导入适配层并断言 nnunetv2
  未被导入。修复后：`tests/unit/test_nnunet_picai_flce.py` **7 passed**；全量 `tests/unit` **210 passed / 0 failed**。
- 未做：未修改 `third_party/nnUNet`；未改动数据、split 与 plan；未运行任何训练或推理；未为任何 run 补写指标。

### 研究协议 v2.1 修订（本日新增）

- `docs/research_plan.md` 由 v2.0 升为 v2.1；本次只修订方法/评测协议，未改动已冻结数据、split 或 nnU-Net plan。
- related work 增加 NaMa、NesMFle 与已有 hard-zone sequence 工作，将 M2 明确定位为近邻基线，核心差异收缩为
  “冻结 PZ/TZ 先验直接条件化空间序列 gate”。
- M4 目标设计从 `L_img + L_zone` 改为零初始化的 conditional affine modulation：
  `L = (1 + gamma_z) * L_img + beta_z`；M3 改为使用相同 zone encoder，但在序列融合后注入解剖特征。
- 融合位置固定为 `3d_fullres` plan 的 encoder stage 2 输出（zero-based，累积 stride `(1,4,4)`）；分区 one-hot
  降采样改为 fractional zonal occupancy，不称 softmax probability。
- M0–M4 目标全体积验证频率由每 epoch 改为每 5 epoch，early stopping 改为 `patience=8 validation events`；
  **现有 M0 YAML/代码尚未迁移，不得写成已完成，迁移与回归测试前不启动 M0 正式训练**。
- 新增 G0-R（序列错位决策）与 G0-SAP（评测/统计冻结），两者当前均为 Pending；G0-E 继续 Pending。
- 将 G0-E 明确为“独立测试路径冻结”：同任务外部 test 不合格时，必须在任何可报告的正式 N0/M0–M4 run 前
  重划 PI-CAI internal test，避免先训练后划 test 造成泄漏。
- 主终点固定为阳性患者 case-wise Dice，主比较按 M4−M2 → M4−M3 层级执行；明确 Dice 的 patient-cluster paired
  bootstrap 与 AP 在每个 bootstrap cohort 内整体重算的区别。
- N0 改为协议不完全配平的 contextual strong reference，不对 N0−M0 差异作单一因果解释。
- 同步 `README.md` 的计划版本、G0-R/G0-E/G0-SAP 状态与 M0 待迁移事项。
- 在 N0/M0 工程说明顶部增加 v2.1 覆盖提示：保留旧实现值供追溯，但不得据此绕过 G0-R 或低频验证迁移。
- 在 `docs/research_plan.md` §8.0 增加 M4 文本版总体预模型图，并标注 M1/M2/M3 的退化路径与共享 skip 约束。
- 未做：未修改 M0 配置/训练代码，未实现 M1–M4 或 `scripts/evaluate/`，未运行真实数据验证、训练、推理或评测。

### 研究协议 v2.2 与低频验证代码迁移（本日新增）

**一、代码迁移：低频全体积验证 + 固定预算（合成 CPU 测试；未读真实医学数据、未使用 GPU）**

- `src/zonal_reliability_fusion/training/trainer.py`：
  - `validation_every_n_epochs` 放开为 `>= 1`（原实现只支持 1，其他值直接报错）；
  - 新增 `_is_validation_epoch`：`full_volume` 模式每 N 个 epoch（1-based 整除）触发一次验证，
    **最后一个 epoch 即使不是 N 的倍数也必验证**；`diagnostic_patch` 模式仍每 epoch 验证；
  - 非验证 epoch：不调用 `FullVolumeValidator`、不解析 checkpoint 指标、不更新 `best_metric/best_epoch`、
    不递增 `bad_validation_events`；CSV 全体积指标字段写空串占位（稳定 schema），标记
    `validation_event=0` / `validation_skipped=1`；`checkpoint_last` 与周期 checkpoint 照常保存；
  - best checkpoint 只在 validation event 产生（`is_best` 在非验证 epoch 恒为 0）；
  - 新增 `validation_events_completed` / `last_validation_epoch` / `bad_validation_events`（兼容旧键
    `early_stopping_bad_epochs`），写入 checkpoint extra 并在 resume 时恢复；
  - early stopping（可选，正式 M0–M4 禁用）：`patience` 以 **validation event** 计数、`min_epochs` 以
    **training epoch** 计数，只能在 validation event 后触发；停止日志与 `early_stop_summary.json`
    改为 validation-event 口径（不再写“连续 N 个 epoch 未改善”）；
  - `PROTOCOL_KEYS` 增加 `validation_every_n_epochs`：协议变化默认拒绝 resume，显式放行时重置
    best/event 状态；`TrainerSummary` 增加 event 计数字段。
- `src/zonal_reliability_fusion/evaluation/full_volume_validator.py`：`ValidationProtocol.validate()` 的
  `every_n_epochs` 放开为 `>= 1`；新增 `FULL_VOLUME_METRIC_FIELDS`（trainer 侧稳定 CSV schema 的字段来源）。
- `src/zonal_reliability_fusion/config/experiment.py`：`validation.every_n_epochs` 校验改为 `minimum=1`；
  取消“full_volume 必须启用 early stopping”硬校验，改为启用时告警（v2.2 正式协议禁用）。
- `configs/experiments/m0_picai_3d_fullres.yaml`：`validation.every_n_epochs: 5`、
  `early_stopping.enabled: false`、`patience: 8`（validation-event 单位，仅通用代码保留）、
  `max_epochs: 200`（固定预算）。
- `scripts/train/train_m0.py`：把 `every_n_epochs` 接入 trainer；正式训练打印验证频率与结束摘要
  （含 `validation_events_completed`）；文档字符串同步 v2.2 口径。
- 测试（合成 CPU）：`tests/unit/test_trainer_synthetic.py` 新增低频验证系列 8 项（调度 / 最终 epoch 必验证 /
  非验证不更新 best / event 计数 patience / min_epochs 按 training epoch / resume 恢复 event 状态 /
  调度变化拒绝与显式放行后重置 / 非法 N）；`tests/unit/test_full_volume_validator.py`、
  `tests/unit/test_plans.py`（`every_n_epochs=5` 合法、正式模式允许禁用 early stopping）、
  `tests/unit/test_train_m0_script.py`（仓库配置目标值自检）同步更新。
- 验证（实际运行；合成 CPU、不读真实医学数据、不使用 GPU）：
  `PYTHONPATH="$PWD/third_party/nnUNet:$PWD/src" CUDA_VISIBLE_DEVICES="" python -m pytest -p no:cacheprovider tests/unit -q`
  → **242 passed / 0 failed**（2026-09-15 复核运行；含既有 `test_picai.py` 等全部测试文件）。

**二、研究计划 v2.1 → v2.2（`docs/research_plan.md`）**

- §9.4 / 新增 §9.4.1：把“每 5 epoch 验证”写成**功能开发 checklist** 并记录实现状态（✅ 已实现项 /
  ⏳ 待真实数据项）；正式 M0–M4 改为固定 `max_epochs=200` + 禁用“连续无改善”early stopping（只保留
  NaN/Inf/覆盖病例数错误等安全中止）；每个 run 报告实际完成 epoch 与 validation event 数；PolyLR 终点 = 200。
- §8.6：WG 退出所有主模型（M0–M4）的 patch 采样与输入；只允许用于 QC、统一评测分层/假阳性分析与
  单列消融；不允许 M4 独占 WG。
- §12.2 重写：`all / −T2W / −ADC / −HBV` 是 **4 个全局输入配置（各训练一次）**，PZ/TZ/mixed/outside 是
  **评价分层而非训练维度**；推理期置零只作 `stress test`。
- §12.4 / §5.4：新增先验独立性规则——必须审计分区器训练数据是否覆盖 PI-CAI/外部测试病例；无重叠才可作
  独立自动先验；无法证明独立时降级 `oracle/upper-bound` / `prior-friendly`；形态学扰动只评价鲁棒性，
  不得声称消除或量化泄漏。
- §13：主 CI 明确定位为 `conditional on the frozen trained seeds`；完整报告全部 seed-specific 效应与跨 seed
  均值/SD/范围；`patient × seed` multi-bootstrap 仅作敏感性且须声明 3 seed 不足以估计 seed 分布；
  AP 必须在每个 bootstrap cohort 内整体重算；新增**功效/精度备忘录**规则（不得用观察到的 M4 效果反向
  制定阈值；数据不足时写 `INSUFFICIENT DATA`，不得伪造 MDE）。
- §3.1 / §11.4 / §14 / §15 / §19 / §20：同步统计口径说明、评测配置载体（G0-SAP 草案）、G0-R/G0-E/G0-SAP
  的 DRAFT 状态与下一步（M0 真实数据三步验证不依赖 G0-R/G0-E，可先在 GPU 空闲时执行）。

**三、协议载体（新建草案；状态一律 DRAFT / PENDING，未冻结）**

- `docs/protocols/G0_R_ALIGNMENT_QC.md` + `configs/protocols/g0_r_alignment_qc.yaml`：确定性抽样规则
  （中心/阳阴/geometry_suspect 全纳入/极端 FOV/间距跨度）、T2W↔ADC 与 T2W↔HBV 分别评价、mm 位移与
  标准化视觉等级、双阅片者与分歧处理、MI/NMI 仅作辅助（禁止单独用跨模态原始 NCC 决策是否需要配准）、
  **盲态 pilot 确定阈值**的预注册决策规则、配准失败/FOV 不足/局部恶化与排除规则、配准前后 QC、
  输入输出/版本/状态/审核人/日期/内容哈希字段；
- `docs/protocols/G0_E_INDEPENDENT_TEST.md` + `configs/protocols/g0_e_independent_test.yaml`：候选登记
  （许可、患者重叠、任务/标签语义、T2/ADC/DWI-HBV 映射、阳性/阴性/病灶定义、PZ/TZ 编码）、
  **自动先验训练数据独立性审计**、支持指标边界与 A/B/C 决策树（合格 external test / 事先降级探索性 /
  重划 PI-CAI internal test）；
- `docs/protocols/G0_SAP.md` + `configs/protocols/g0_sap.yaml`：主终点与 M4−M2 → M4−M3 层级比较、
  阈值/3D 连通性/最小体积/候选置信度冻结、`picai_eval` 版本与参数（默认 IoU ≥ 0.1）、patient-cluster
  bootstrap、AP cohort 级重算、seed 不确定性报告规则、**功效/精度备忘录模板字段**、test 一次性评估、
  配置内容哈希与协议变更规则；
- 三个 YAML 均通过 `yaml.safe_load` 解析校验；所有未确定数值字段为 `null` 并注明由盲态 pilot / 冻结时填写，
  **未伪造任何阈值、样本量或 MDE 数值**。

**四、文档同步（清理旧口径冲突）**

- `docs/Data_Analysis.md`：顶部新增历史说明（数据事实仍有效、可引用；研究方向建议已被 v2.1/v2.2 覆盖）；
  §5 第 9 条“域适应”与 §6“跨数据集泛化/域适应、LOCO”标记为**历史候选方向**，不属于当前主线；
  数据事实、标签语义、几何审计内容保持不变。
- `README.md`：更新为 v2.2 状态（G0-R/G0-E/G0-SAP = Pending + 协议草案 DRAFT；低频验证迁移已完成；
  G2 仍待真实数据三步验证；明确“代码迁移或合成测试通过不得使任何门变为 PASS”）。
- `docs/P2_M0_Implementation.md`：顶部状态改为“代码 CODE READY（含 v2.2 迁移）/ 正式训练 NOT READY”；
  §8 / §16.1 / §16.3 / §16.4 / §16.5 / §18 更新为 v2.2 值（旧值标注为历史记录）。
- 本文件本条 + `docs/experiment_log.md` 同步记录本轮实际运行的测试命令与结果。

**未做**：未修改 `third_party/`；未改动数据、split 与 nnU-Net plan；未实现 M1–M4 或 `scripts/evaluate/`；
**未运行任何读取真实医学数据的命令**（loader smoke、small-overfit、正式训练、推理、全体积验证、评测均未运行）；
未把 G0-R / G0-E / G0-SAP / G2 标记为 PASS。

**验收修正（2026-09-15，用户验收意见：有条件通过，唯一阻塞项）**

- 阻塞项：`docs/research_plan.md` §4.4 仍写 WG 可用于“裁剪/采样”，与 §8.6“WG 退出所有主模型（M0–M4）的
  patch 采样与输入”自相矛盾。
- 修正：§4.4 的 WG 行改为——"WG：Bosma22b AI whole-gland mask，仅用于数据质控、统一评测分层、假阳性分析
  与单列消融；不得参与 M0–M4 的裁剪、patch 采样或网络输入"；与 §0.1 修订摘要、§8.6、§9.3 口径统一。
- 复核：全库 WG 相关表述扫描；`research_plan.md` 内所有“WG + 采样/裁剪/输入”行（§0.1、§4.4、§8.6）
  修正后一致；`P2_M0_Implementation.md`、`project_structure.md` 的“M0 不得引入 WG”表述一致；
  `PICAI_Model_Readiness_Audit.md` §5.4 中“WG（Bosma22b，仅用于后续采样/QC）”为 P0B 物化阶段的历史记录
  （按审计报告“不删改历史”约定保留；以 v2.2 研究计划为当前协议最高权威，该行不构成对 WG 用途的有效声明）。
- 未做：未修改代码、配置、数据、split 与 plan；未运行任何数据/训练/推理/评测命令；未改变任何门的
  Pending/DRAFT/NOT-EVALUATED 状态。

### G0 协议工具实现（G0-R / G0-E / G0-SAP；只读元数据，未执行真实数据）

**新增代码（`src/zonal_reliability_fusion/protocols/`，不导入 torch / nnunetv2）**

- `common.py`：协议配置加载与契约校验（id/version/status/`sees_model_predictions`）、字段助手、
  内容哈希（SHA256）、运行元数据（附 `config_sha256`）、输出目录隔离（禁止落入 `outputs/checkpoints`、
  `outputs/nnUNet_results`、`workdir/nnUNet_*`）、JSON/CSV 写入器、未填字段检测
  （`None`/`TODO`/`INSUFFICIENT_DATA` 均视为未填）。
- `g0_r.py`：G0-R 确定性分层抽样（中心 × 阳阴性 × `geometry_suspect` 全纳入 × partial overlap ×
  极端 FOV × 间距跨度；`n_cases=null` → 只取最小必选集合；给定 n_cases → 按 case_id 确定性补齐）；
  双阅片盲审模板；对齐指标 schema（`PAIRS` / `METRIC_UNITS` / `AUXILIARY_METRICS`）与行级校验；
  抽样 manifest 保留 `decision=None`（工具不得决定配准方案）。
- `g0_e.py`：候选数据集**文件清单级**清点（病例目录、模态/标注文件名计数、CSV 表头与行数、
  empty 占位；不打开体素）；证据状态汇总（`VERIFIED` / `UNKNOWN` / `CONFIGURED_EXCLUDED` / `AUTO_ONLY`）；
  待补证据清单与报告模板；**不推断**独立性、任务等价性或可用性。
- `g0_sap.py`：G0-SAP 硬约束校验（主终点、M4−M2 → M4−M3 顺序与 precondition、
  `ci_interpretation=conditional_on_the_frozen_trained_seeds`、bootstrap 类型/CI、seed-level 仅 sensitivity、
  `picai_eval` + IoU ≥ 0.1 + FROC 0.5/1.0/2.0、功效备忘录字段完整性）；冻结就绪检查
  （24 项必需字段 + `power_memo.status=sufficient`）；功效备忘录 / 冻结清单 / 统计模板渲染（数值一律 TODO/null）。

**新增入口脚本（均支持 `--dry-run` / `--no-progress`；不读影像、不判 PASS）**

- `scripts/audit/audit_picai_alignment_qc.py`（G0-R：抽样 + 盲审清单 + `--validate-metrics` 指标 schema 校验）；
- `scripts/audit/audit_g0e_candidates.py`（G0-E：清点 + 证据清单 + `missing_evidence.md`）；
- `scripts/evaluate/prepare_g0_sap_freeze.py`（G0-SAP：静态校验 + `--check-freeze` 就绪检查 +
  冻结包模板 + 输出隔离；`status=FROZEN` 且阻塞项非空时**拒绝**生成冻结载体）。

**配置载体更新（仍为 DRAFT，未冻结）**

- `configs/protocols/g0_sap.yaml`：新增 `checkpoint_selection`、`case_level_metrics`、`strata`、
  `seed_aggregation` 结构化块（引用 research_plan 已冻结值 0.5/0.75/0.67/0.80，未编造新数值）；
- `configs/protocols/g0_r_alignment_qc.yaml`：登记已实现工具与新增输出文件（manifest/盲审表/schema/元数据）；
- `configs/protocols/g0_e_independent_test.yaml`：候选补 `local_root`、登记清点工具与新增输出。

**测试与验证（合成 CPU；不读真实医学数据）**

- 新增 `tests/unit/test_{g0_protocol_common,g0_r_sampling,g0_e_inventory,g0_sap_freeze,g0_scripts}.py` 共
  **55 项**：抽样确定性与分层命中、n_cases 补齐、盲审行数、schema 错误分支；清点计数与 UNKNOWN 不推断；
  SAP 硬约束拒绝矩阵与冻结就绪；三个脚本的 `--dry-run` 不写文件、输出生成、`--fail-if-not-ready`（exit 3）、
  `status=FROZEN` 拒绝生成；
- 全量 `tests/unit`：**297 passed / 0 failed**（既有 242 项无回归）；
- 静态检查：三个脚本 `--help` 通过；`--check-freeze` 对仓库配置实测 **NOT READY**（24 字段待填 + 备忘录未 sufficient）。

**文档同步**：`docs/research_plan.md`（G0-R/G0-E/G0-SAP 门状态与 §11.4 增加工具状态）、`README.md`、
`docs/protocols/G0_{R_ALIGNMENT_QC,E_INDEPENDENT_TEST,SAP}.md`（新增"工具实现状态"节）、
`docs/project_structure.md`、`docs/experiment_log.md`（本日测试记录）。

**未做**：未运行任何真实数据命令（G0-R 抽样、G0-E 清点、M0 三步验证、训练/推理/评测均未运行）；
未实现 `scripts/evaluate/` 的评测逻辑；未填写任何阈值/MDE/verdict；未把任何门标记为 PASS；
未修改 `third_party/`、原始数据、冻结 split/plan 或既有训练产物。

### G0-R 图像 QC 执行工具（2026-09-15；已实现，未在真实病例运行）

> 背景：研究者已执行 G0-R 抽样（2026-09-15，16 例 → `outputs/diagnostics/g0_r/20260915_093819/`，
> `decision=null`）。本次实现"真实图像对齐 QC 执行工具"，仍**不代表 G0-R 通过**。

**新增代码 `src/zonal_reliability_fusion/protocols/g0_r_qc.py`**

- 抽样清单契约（`load_sampling_manifest`：protocol.id / cases 唯一性 / pairs 合法性；
  `select_cases`：**只允许抽样清单子集**，不在清单中直接报错，不替换/增删/重抽样）；
- 物化路径解析（`resolve_series_paths`：沿用 `scripts/data/materialize_picai.py::SEQ_FILES` 既有命名
  `cases/<case_id>/{t2w,adc,hbv}.nii.gz`，不另造命名规则）+ 缺文件检测；
- **物理坐标重采样**（`prepare_pair_arrays`：同网格直用、不同网格用 `picai.resample_to_geometry` 线性重采样到
  T2W 网格；**不做任何配准**）；`PairGeometryRecord` 记录两序列几何、`same_grid_as_t2w`、`resampled_now`、
  物理包围盒重叠率与 `physical-space resample ONLY (no registration)` 声明；
- 显示合成（纯 numpy）：`normalize_for_display` / `checkerboard_image` / `edge_map` /
  `edge_overlay_image`（T2W 红、moving 绿、重合黄 → 错位呈红绿分离）/ `select_slice_indices`
  （按 z 深度 25/50/75%，**不使用病灶或腺体信息**，避免判读线索）；
- landmark：模板与校验（`validate_landmark_rows`：列、病例子集、pair、坐标非负与 FOV 范围；坐标序
  `i,j,k = x,y,z`）、位移计算（`compute_landmark_metrics`：Δ索引 × spacing 欧氏范数 → `landmark_displacement_mm`）
  与按 case×pair 汇总；
- 边界距离（可选）：`compute_boundary_metrics`（Hausdorff + 双向平均表面距离，SimpleITK，物理间距）与
  `read_boundary_source`（CSV/JSON）；**无来源 / 缺条目 / 空 mask 一律 skipped（UNKNOWN），绝不伪造**。

**新增入口 `scripts/audit/render_picai_alignment_qc.py`（三模式）**

- **渲染**：对抽样清单病例逐例渲染 `overlay/<case>_{adc,hbv}_{checkerboard,edge}.png`（图上标注 case_id /
  序列对 / 方向 / k 层位 / "physical-space resample ONLY"；**不含病灶标签与模型输出**），并输出
  `landmark_template.csv`、`boundary_template.csv`、`qc_index.csv`、`qc_geometry.json`、
  `run_metadata.json`（含输入影像 SHA256 与跳过原因）、`skipped.csv`；
- **指标**（`--landmarks-csv`）：人工坐标 → `alignment_metrics.csv`（写前用既有
  `g0_r.validate_alignment_metrics_rows` 自检；可选 `--boundary-source` 追加边界指标）+ `landmark_summary.json`
  + `metrics_skipped.json`；
- **校验**（`--validate-landmarks`）：列 / 病例子集 / 坐标范围校验 → `landmarks_validation.json`，非法退出码 1；
- 安全：输出路径隔离（禁止落入训练产物）、**拒绝覆盖非空目录**（保护 `20260915_093819` 等既有产物）、
  `--case-ids` 仅允许清单子集、`--dry-run` / `--no-progress`。

**测试**（合成 3D NIfTI；`tests/unit/test_g0_r_qc.py`，16 项）与全量回归见 `docs/experiment_log.md`
同条目：**313 passed / 0 failed**（既有 297 项无回归）。

**研究者运行命令（真实 16 例；由研究者执行，代理不得代为运行）**

```bash
cd /opt/data/private/lm/my-projects
conda activate lm
# 1) 渲染（先 --dry-run 看计划；成功后生成 PNG/模板/几何/哈希）
python scripts/audit/render_picai_alignment_qc.py --dry-run
python scripts/audit/render_picai_alignment_qc.py
# 2) 双阅片者填写 <run>/landmark_template.csv（另存为 landmarks_filled.csv；坐标序 i,j,k = x,y,z）
python scripts/audit/render_picai_alignment_qc.py --validate-landmarks --landmarks-csv <run>/landmarks_filled.csv
python scripts/audit/render_picai_alignment_qc.py --landmarks-csv <run>/landmarks_filled.csv [--boundary-source <csv>]
# 3) 用既有 schema 复核指标
python scripts/audit/audit_picai_alignment_qc.py --validate-metrics <run>/alignment_metrics.csv
```

**未做**：未读取任何真实病例影像（测试仅用临时目录的合成 NIfTI）、未渲染真实数据、未使用 GPU、
未运行推理/训练/评测；未修改原始/物化数据、split、plan、`third_party/` 或既有训练产物；
未填写任何阈值 / `decision`；**G0-R 仍为 DRAFT/Pending**（工具产物不等于 PASS）。

### G0-R 图像 QC 工具修复（2026-09-15；代码/测试/文档，未运行真实病例）

**1. 指标模式强制校验（不可绕过）** — `metrics_mode()` 现在先调用 `validate_landmark_rows`
（严格整数、非负、T2W/moving 分别 FOV、病例子集、`coordinate_space`、重复记录）与
`check_landmark_completeness`（病例与 case×pair 覆盖、`measured_by`、最小数量规则）；任一未通过即
**非零退出且不写 `alignment_metrics.csv`**，并写出 `landmarks_validation.json`
（含 landmark SHA256、错误数、覆盖病例与 case×pair、blockers）。新增 `--allow-incomplete` 仅用于诊断放行：
产物标记 **INCOMPLETE（不得用于冻结）**，且越界/非法行一律不产出位移值。

**2. 严格整数坐标** — 新增 `parse_strict_index()`：只接受有限整数值（`10` / `10.0` / `1e2`），
拒绝 `10.9`（**不再静默截断**）、NaN/Inf、空字符串；校验收集**全部行的全部错误**（不再遇错即返回）。

**3. 坐标空间与毫米距离（冻结）** — 冻结为 `coordinate_space = native_index_per_series`：
`t2w_*` 在 T2W 物化网格、`moving_*` 在对应 ADC/HBV 物化网格（两者均可由研究者在 ITK-SNAP / 3D Slicer
中直接打开物化 `.nii.gz` 标注）；位移改为 `SimpleITK TransformIndexToPhysicalPoint` 的 **LPS 物理坐标欧氏距离**
（正确纳入各序列的 origin / direction / spacing，`geometry_from_record_fields` + `index_to_physical_lps`）；
FOV 分别按 T2W 网格与 moving 网格检查；`qc_geometry.json` 补齐 **T2W origin/direction**（此前缺失，
会导致指标无法计算）。

**4. 完整性检查** — 新增 `check_landmark_completeness()`：报告 16 例覆盖、每 case×pair 计数、
`measured_by` 缺失行、未知病例、重复记录；**不编造最小数量阈值**——协议
`evaluation.landmark_rules.min_per_case_pair` 为 `null` 时输出 blocker
`landmark_min_count_rule_not_frozen`，要求研究者在真实运行前冻结（冻结后按该值检查）。

**5. 文档矛盾修正** — `protocols/__init__.py` 与 `docs/project_structure.md` 改为**按模块区分**：
`common/g0_r/g0_e/g0_sap` 只处理配置/元数据；`g0_r_qc` 与渲染脚本在研究者运行时会**只读物化影像体素**
用于 QC（不修改数据、不训练、不推理、不配准）；`protocols/__init__.py` 入口清单加入渲染脚本；
协议 §4.2 写入坐标空间与待冻结规则；各处继续标注"已实现但真实病例未运行 / G0-R 仍为 DRAFT-Pending"。

**6. lint** — `ruff check` 逐项修复 6 项 + shebang 权限：`EXE001`（`chmod +x`）、`RUF100`×3（移除无效
`noqa: E402`）、`RUF034`（imshow 无用条件）、`UP035`（`Mapping/Sequence` 改从 `collections.abc` 导入）、
`RUF046`（`round()` 结果冗余 `int()`）；未使用任何全局 ignore。

**测试与验证**（合成 3D NIfTI；`tests/unit/test_g0_r_qc.py`）：**24 passed**；全量 `tests/unit` =
**321 passed / 0 failed**；`ruff` = **All checks passed**；`--help` OK。详见 `docs/experiment_log.md`
同条目。

**未做**：未运行真实渲染 / 真实指标计算 / 真实 `--dry-run`；未读取任何真实病例影像；未使用 GPU；
未修改原始/物化数据、split、plan、`third_party/` 或既有训练产物；未填写任何阈值 / `decision`；
**G0-R 仍为 DRAFT/Pending**。

### G0-R Automated v0.2：全自动 QC 路径（2026-09-16；代码 / 协议 / 合成测试，未运行真实病例）

**新增文件**

| 文件 | 作用 |
|---|---|
| `configs/protocols/g0_r_alignment_qc_automated.yaml` | 自动协议唯一配置载体：隐私/输入策略、FOV 策略、预处理与边缘参数、指标与容忍半径、刚体诊断参数、校准条件与种子、阈值推导公式、判定逻辑与 `INSUFFICIENT_EVIDENCE` 触发条件；`protocol_hash` 绑定（排除 `inputs/outputs/progress`） |
| `src/zonal_reliability_fusion/protocols/g0_r_automated_qc.py` | 纯计算算法库（不读文件、不写文件）：几何/FOV 证据、强度标准化、多尺度物理边缘、双向 mm 距离与多容忍半径 F1、slicewise 审计、诊断性刚体估计（可注入 runner）、已知位移注入、校准汇总、阈值推导、单例与总体判定、`protocol_hash`、输入策略强制 |
| `scripts/audit/run_picai_alignment_qc_automated.py` | 一键 CLI：`--dry-run`（不读影像体素、不写文件）、非空目录拒绝覆盖、逐例读取→A/B/C→校准→释放数组（避免长期持有全部影像）、全量输出与结束摘要 |
| `docs/protocols/G0_R_ALIGNMENT_QC_AUTOMATED.md` | 自动协议正文（DRAFT）：证据 A–D、阈值公式、判定顺序、输出清单、冻结程序、局限性与非目标 |
| `tests/unit/test_g0_r_automated_qc.py` | 25 项合成测试（含轴序锁定、单调响应、校准失败→INSUFFICIENT_EVIDENCE、CLI 行为、AST 边界扫描） |

**边界（代码与测试双重强制）**：不读 lesion/WG/PZ-TZ/模型预测（`assert_automated_input_policy` 拒绝开启）；
无网络/外部 API import；无影像写回 API（`sitk.WriteImage`/`ImageFileWriter`/`np.save` 全部禁止，AST 扫描 +
端到端断言输入哈希不变）；刚体估计仅内存；输出一律写入项目内新建/空目录。

**实现中由合成测试捕获并修复的真实缺陷**（详见 `docs/experiment_log.md` 同名条目与协议 §7.4）：
① `array_to_image` spacing 轴序错误；② 距离图语义应为"到目标掩膜前景的距离"；
③ 位移注入的常量零填充会造出人为边界、破坏单调性 → 改为 `nearest` 并在配置冻结。

**验证**：`tests/unit/test_g0_r_automated_qc.py` **25 passed**；全量 `tests/unit` **481 passed**；
`ruff` **All checks passed**；`--help` OK。均为合成数据与静态检查。

**未做**：未读取任何真实病例影像、未运行真实 QC / 配准 / 训练 / 评测、未运行真实 `--dry-run`；
未修改原始/物化数据、split、plan、`third_party/` 或既有产物；**G0-R 仍为 DRAFT/Pending**，
自动运行成功不等于 PASS；冻结前不得启动 P2B / 正式 M0 / N0。

### v2.3 研究计划依赖与文档状态同步（2026-09-16；仅文档/配置注释）

- 修复 G2 循环依赖：将 `research_plan.md` §9.4.2 拆为 G2 前置的 Residual-Encoder M0/公共接口契约（1–8）与
  G2 后置的 M1–M4 融合实现（9–11）；G2 不再要求尚未实现的 M1–M4 作为自身前置。
- 统一正式 N0 门：启动正式训练必须同时满足 G0-R PASS 与 G0-E 独立测试路径冻结；此前执行只能标记为
  feasibility。训练预算/骨干差异和 `checkpoint_final.pth` 规则改为启动前冻结。
- 修正架构冻结措辞：当前只冻结 v2.3 设计边界与接口原则；每 stage block 数、激活顺序、参数预算和最终架构哈希
  仍由 P2A 的版本化配置与测试冻结，不把计划意图写成实现事实。
- 同步 G0-E 真实状态：2026-09-15 文件清单级清点已经执行并产生
  `outputs/diagnostics/g0_e/20260915_093835/`，但证据与 verdict 仍为 `UNKNOWN/null`，G0-E 继续 Pending。
- 更新 `README.md`、`docs/P2_M0_Implementation.md`、`docs/project_structure.md`、
  `docs/PICAI_Model_Readiness_Audit.md` 与旧 M0 YAML 注释：现有 `PlanDrivenUNet3D`/配置/命令统一标记为
  v2.2 legacy PlainConv；v2.3 Residual-Encoder M0 为 CODE PENDING，旧命令不得用于正式 G2。

本次未修改 Python 代码、模型实现、数据、split、plan、`third_party/` 或任何训练产物；未运行数据处理、训练、
推理、验证或评测。文档修订不改变 G0-R/G0-E/G0-SAP/G2 的 Pending 状态。

### P2A：v2.3 plan-driven Residual-Encoder M0 实现（2026-09-16；合成 CPU 测试，未读真实数据/未用 GPU）

**新增源码（`src/zonal_reliability_fusion/`，自研，不 import nnunetv2）**

- `models/residual_blocks.py`：`ResidualBlock3d`（post-activation，复刻 `BasicBlockD` 可验证行为：conv1=Conv3d-IN-LeakyReLU、
  conv2=Conv3d-IN（无激活）、`out=LeakyReLU(conv2(conv1(x))+shortcut(x))`；shortcut=identity 或 ResNet-D
  `AvgPool3d(kernel=stride,stride=stride,ceil_mode=True)`→（通道变化时）`1×1×1 Conv(bias=False)`-IN）、
  `StackedResidualBlocks3d`（首块承担 stride/通道变化，其余 identity）；初始化 He(fan_in,leaky_relu,0.01)+conv bias=0+
  IN weight=1/bias=0，并把加法前最后一个 IN（conv2 的 norm）置 0（对齐官方 `init_last_bn_before_add_to_0`）。
  **奇数尺寸适配**：shortcut 的 AvgPool 用 `ceil_mode=True`，使 shortcut 与 odd-kernel+same-pad 的 strided conv 输出
  尺寸一致（官方靠 patch 可整除回避；本项目按 §7.3 覆盖奇数尺寸）；对可整除尺寸（冻结 plan 全部 stage）逐尺寸等价。
- `models/residual_unet3d.py`：`PlanDrivenResidualEncoderUNet3D`（default stem=1 conv 不下采样 + residual encoder +
  普通 U-Net decoder：ConvTranspose3d(kernel=stride)→center_crop_or_pad→concat→stacked conv→每级 seg head；DS 高→低）、
  `zero_init_residual_last_norms`；构建时 `validate_architecture_against_plan` 强制结构字段与冻结 plan 一致。
- `models/m0_residual_concat.py`：`M0ResidualConcatModel`（三通道 concat；`forward` 不接受 zonal/wg/gate 关键字→TypeError；
  `architecture_identity()`；summary/format_summary）、`FORBIDDEN_INPUTS`。
- `models/profiling.py`：`count_parameters`、`analytical_macs`（纯算术枚举 Conv3d/ConvTranspose3d，**无 forward**）、
  `count_macs_with_hooks`（forward-hook 计数，仅用于缩小 plan 验证公式）、`analytical_feature_map_elements`、
  `structure_summary`（峰值显存 `status=NOT_MEASURED` + 分析性激活字节下界代理，明确非实测）。
- `models/factory.py`：`build_m0_model`（按 `cfg.architecture` 分派：None→legacy `M0ConcatModel`，有→`M0ResidualConcatModel`）、
  `resolve_architecture_spec`、`verify_ref_matches_spec`、`model_architecture_identity`（legacy 返回 None）。
- `config/architecture.py`：`ArchitectureSpec`（版本化架构配置，`structural_dict()` 返回**深拷贝**）、canonical-JSON
  （sort_keys）→SHA-256 `architecture_sha256`、`load_architecture_config`/`architecture_spec_from_mapping`、
  `validate_architecture_against_plan`、`SHALLOW_SKIP_CONTRACT`+`validate_shallow_skip_contract`+`assert_shallow_skip_inputs_allowed`
  （§8.6 仅接口）、`ArchitectureRef`、`verify_architecture_identity`（缺失/legacy/哈希不匹配→`ArchitectureIdentityError`）。

**新增配置**：`configs/architectures/m0_resenc_v23.yaml`（blocks_per_stage=(1,3,4,6,6,6,6)=官方 ResEnc-M encoder 拓扑；
features/kernel/stride/n_conv_decoder 逐值等于冻结 plan；含 provenance）、
`configs/experiments/m0_resenc_picai_3d_fullres_v23.yaml`（独立 `experiment.name` 与 `output_root=outputs/checkpoints/m0_resenc`）。

**新增入口**：`scripts/train/summarize_m0_resenc_architecture.py`（结构摘要/参数/解析 MACs/显存状态；`--no-progress`、
`--reduced-forward-check`、结束输出成功/失败/跳过/耗时/路径；只读、无 GPU、不 forward 真实 patch）。

**最小附加修改（不破坏 legacy）**：`config/experiment.py`（新增**可选** `architecture: ArchitectureRef|None`，缺省 None→旧行为不变）、
`config/__init__.py`/`models/__init__.py`/`training/__init__.py`（导出新符号，保留旧导出）、`training/trainer.py`
（`FROZEN_CONFIG_PREFIXES` 增 `architecture.`/`provenance.architecture_sha256`）、`training/checkpointing.py`
（新增 `verify_checkpoint_architecture`）、`scripts/train/train_m0.py`（改用 `build_m0_model` 工厂；resume 前架构守卫；
把架构身份写入 config_snapshot）、`scripts/train/validate_m0_setup.py`（工厂分派+打印架构身份）、`tests/conftest.py`
（新增 mini/odd residual plan、`build_resenc_arch_doc`、`resenc_spec`/`mini_resenc_model` 夹具；`write_plan` 建父目录）。

**保留不动（legacy）**：`PlanDrivenUNet3D`、`M0ConcatModel`、`blocks.py`、`unet3d.py`、`m0_concat.py`、
`configs/experiments/m0_picai_3d_fullres.yaml`、旧测试、旧输出目录，均未改写/删除。

**发现并修复的缺陷（合成测试/产物核验暴露）**：`ArchitectureSpec.structural_dict()/to_dict()` 早期返回内部嵌套 dict 的
**引用**，导致 `summarize` 脚本的 `--reduced-forward-check` 修改派生 doc 时**污染 frozen spec 的 decoder 字段并改变架构哈希**
（产物中 `architecture_identity` 与 `structure_summary` 哈希一度不一致）。修复：`structural_dict()` 对全部嵌套 dict 返回
`copy.deepcopy`；新增回归 `test_mutating_returned_dict_does_not_change_hash`。修复后两处哈希一致
（`747b190a48877c077acc927735e11ca151333f73571e9e4de35d2a0113dcb653`）。

**验证（实际运行；合成 CPU、不读真实医学数据、不使用 GPU、未训练/推理/评测）**：

| # | 命令 | 结果 |
|:--|:--|:--|
| 1 | `PYTHONPATH="$PWD/third_party/nnUNet:$PWD/src" CUDA_VISIBLE_DEVICES="" python -m pytest -p no:cacheprovider tests/unit -q` | **435 passed / 0 failed**（既有 321 项无回归 + 新增 114 项） |
| 2 | `python -m ruff check <全部新增/重写文件>` | **All checks passed!**（新增文件；legacy 文件的既有告警未纳入本轮范围） |
| 3 | `python scripts/train/summarize_m0_resenc_architecture.py --reduced-forward-check --no-progress` | 参数 148,193,644；MACs（单 patch）encoder 785,883,136,000 + decoder 322,539,980,800 = 1,108,423,116,800；reduced-forward hook==analytical（258,560）；显存 NOT_MEASURED |

新增测试文件：`test_residual_blocks.py`、`test_residual_unet3d.py`、`test_architecture_config.py`、`test_skip_contract.py`、
`test_profiling.py`、`test_m0_residual_concat.py`、`test_residual_checkpoint_resume.py`、`test_train_m0_resenc_script.py`
（覆盖任务书九·1–23：identity/通道/stride/stride+通道 shortcut、奇数尺寸、He+zero-last-norm 初始化、stage shape、
DS 数量/顺序/shape、forward/backward、CPU autocast、缩小滑窗、参数统计、哈希稳定/敏感、同架构 resume、哈希不匹配拒绝、
legacy checkpoint 拒绝、strict=False 禁用、新配置只建 residual/旧配置只建 legacy、M0 拒 PZ/TZ/WG/gate、§8.6 契约守卫、解析 MACs==hook MACs）。

**未做**：未读取任何真实病例影像（`.b2nd`/`.pkl`/NIfTI）；未使用 GPU；未运行 loader smoke、small-overfit、训练、
全体积验证、推理或评测；未实现 M1–M4、`Q_l`、`F_fuse`、gate；未修改 `third_party/`、原始/物化数据、split、plan；
未把 G2/G0-R/G0-E/G0-SAP 标记为 PASS（§9.4.2 仅将 G2 前置 1–7 标 ✅，第 8 项仍 ⏳，G2 仍 NOT YET EVALUATED）。
**峰值显存未实测**（无 GPU 授权），解析激活下界代理 ≈6.07 GB（fp32,batch=2）已在文档标注为「非实测、须 P2B 核实」。

**文档同步**：新建 `docs/P2_M0_ResidualEncoder_v23.md`（实现说明/对照表/预算/哈希/隔离/P2B 命令草案）；更新
`docs/research_plan.md`（抬头、§0.1、§9.4.2、§14/G2、§20）、`README.md`、`docs/project_structure.md`、
`docs/P2_M0_Implementation.md`（保留为 legacy 历史，补 v2.3 P2A 已完成指针）。

### P2A 验收轮修复（2026-09-16；研究者验收意见；仍合成 CPU、未读真实数据/未用 GPU）

研究者验收：P2A 核心实现与合成测试可信，但要求先完成两项无真实数据修复再签收。已完成：

**修复 1：架构配置与实际 shortcut 行为不一致。**`configs/architectures/m0_resenc_v23.yaml` 原写 `projection_when:
stride_or_channel_change`（“stride 或通道变化均投影”），但实现（与 `BasicBlockD` 一致）在**仅 stride 变化**时只做
`AvgPool3d`（无投影 conv），只有**通道变化**才加 `1×1×1 Conv+IN`。修正为 `downsample_when: stride_change` +
`projection_when: channel_change`（新增 `downsample_when` 字段）；`config/architecture.py::_parse_shortcut` 与
`tests/conftest.py::build_resenc_arch_doc` 同步。修复只改变**声明**（网络行为本就正确），故参数量/MACs 不变；
但 shortcut 是结构字段 → 架构哈希由 `747b190a…` 变为 **`8bb5127e5c9a99ca1a0631cb365cdcf0b866441ac225911e17faa6bc995bbee5`**
（上一条目记录的 `747b190a…` 为 deepcopy 修复当刻的历史值，现已被本修复取代）。已重生成
`outputs/diagnostics/m0_resenc/…/architecture_summary.json`（identity 与 summary 哈希一致）。

**修复 2：版本化架构配置“改了但不生效”漏洞。**原解析器会接受并哈希模型未实现的值（如
`shortcut.projection_kernel=[3,3,3]`、`initialization.method=xavier`、`decoder.skip_merge=sum`、
`deep_supervision.output_order=low_to_high`）而不报错。新增**严格白名单校验**（`config/architecture.py`：
`STEM_SUPPORTED`/`SHORTCUT_SUPPORTED`/`NORMALIZATION_SUPPORTED`/`NONLINEARITY_SUPPORTED`/`INIT_SUPPORTED`/
`DECODER_SUPPORTED`/`DEEP_SUPERVISION_SUPPORTED` + `_require_frozen`）：所有冻结规则字段只能声明模型实际实现的唯一
取值，声明未实现值→**加载时** `ArchitectureConfigError`；数值型字段（features/kernel/stride/n_conv_decoder/eps/slope/conv_bias）
仍由 `validate_architecture_against_plan` 与冻结 plan 逐值核对。新增一致性校验：`initialization.negative_slope` 必须
等于 `nonlinearity.negative_slope`。测试：`test_illegal_or_unsupported_value_rejected` 扩充至 ~35 例（含研究者指出的 4 例）；
`test_any_structural_field_change_changes_hash` 收敛为 12 个可自由取值字段（白名单字段改入拒绝组）；新增
`test_shortcut_semantics_frozen_downsample_vs_projection`。

**文档状态同步（清除过期“未实现/CODE PENDING”）**：`docs/research_plan.md` §6（v2.3 自研架构状态）、§9.2（损失迁移状态）、
§10.1 M0 表项、§15 实施阶段第 7 项、§20 “在此期间”项目；`docs/PICAI_Model_Readiness_Audit.md`（状态说明与§11 两处）；
`README.md`、`docs/P2_M0_ResidualEncoder_v23.md` 的架构哈希引用——统一改为“P2A 已完成、G2 第 8 项 Pending、G2 = NOT YET
EVALUATED”。M1–M4 / `scripts/evaluate/` / G0-R 图像 QC 真实运行等**确实未完成**的“尚未实现”表述保留不改。

**lint 范围声明（诚实）**：P2A 新增/重写文件 `ruff check` 全部 **All checks passed**；但**全仓库仍有约 148 个既有 Ruff
告警**（legacy 文件，如 `blocks.py`/`unet3d.py`/`train_m0.py` 的 UP035/DTZ005/PLR0124/RUF100 等），**不声称“全仓库 lint 全绿”**；
本轮不 retro-fit legacy 文件（避免超出范围、改变既有行为）。

**验证（实际重跑；合成 CPU、不读真实医学数据、不用 GPU、未训练/推理/评测）**：全量 `tests/unit` → **456 passed / 0 failed**
（较上轮 435 新增 21 项白名单/语义测试）；P2A 新增/重写文件 `ruff check` → All checks passed；
`summarize_m0_resenc_architecture.py --reduced-forward-check` → 参数 148,193,644、MACs 1,108,423,116,800、reduced-forward
hook==analytical（258,560）、显存 NOT_MEASURED、新哈希 `8bb5127e…`。

**未做**：未读真实病例影像、未用 GPU、未 loader smoke/small-overfit/训练/全体积验证/推理/评测；未实现 M1–M4/`Q_l`/`F_fuse`/gate；
未改 `third_party/`/数据/split/plan；未覆盖/删除 legacy 产物；未把 G2/G0-R/G0-E/G0-SAP 标为 PASS（G2 第 8 项仍 Pending）。

## 2026-09-17

### 文档结构重构与一致性修复（仅文档/配置状态字段；未运行数据/训练/推理/评测）

**背景**：`docs/research_plan.md`（约 1515 行）同时承载研究协议、完整版本修改史、实时状态看板、工程验收
checklist、实际实验日志、可复制训练命令与代理执行规则，导致同一状态在多处并行维护并出现矛盾
（如 §20 既写 P2A 已完成、又写"并行推进 P2A 实现"）。本次按「协议 = 计划 / 日志 = 事实 / 状态 = 单一来源」
重建文档边界。

**新增文件**

| 文件 | 作用 |
|---|---|
| `docs/STATUS.md` | **当前状态唯一权威**：门状态表、实现状态、9 项 blocker（含 Git 缺失、G0-R/G0-E/G0-SAP 未冻结、显存未实测、G0-E 配置哈希变更）、下一步 |
| `docs/protocol_changelog.md` | 协议版本变更史（v2.0 → v2.0 状态同步 → v2.1 → v2.2 → v2.3 → 本次重构），逐条记录日期/内容/原因/是否在查看结果前/对既有结果的影响 |
| `docs/runbooks/README.md`、`g0_r_alignment_qc.md`、`g0_e_candidate_audit.md`、`g0_sap_freeze.md`、`p2b_m0_validation.md`、`n0_training.md` | 研究者执行的命令、输出路径、进度观察、成功判据与停止条件（自 research_plan §14/§20 与各协议迁出） |
| `docs/literature/novelty_and_standards_check_20260917.md` | 文献与规范核查记录（SPIRIT 2025、CLAIM 2024、PI-CAI 官方数据页、picai_labels 含 Pooch25、MIGF 邻接工作）：含检索方式、限制、未做事项与系统检索协议建议 |
| `docs/literature/references.md` | 参考资料总表（自 research_plan §21 迁出并补注释） |

**research_plan.md 变更（协议条款未变，只做边界重构与补充）**

- §0：删除 v2.1/v2.2/v2.3 长篇修改史（→ `protocol_changelog.md`），改为版本摘要 + **文档边界表**（每类事实的
  唯一权威来源）+ 阅读纪律（工程完成≠门 PASS；validation≠独立确认；gate 权重≠因果贡献；不伪造文献结论）；
- §2.2（新增）：新颖性表述纪律 + 系统检索未完成的现状（不得用"首次"，上限表述为"据我们所知"）；
- §3.1（新增）：MCID 缺口与判定规则（待冻结决策 D4）；
- §4.2/§4.2.1/§4.3：补 PI-CAI 官方依据（公开训练集不共配准；HBV b ≥ 1000 s/mm²；ADC 绝对值不可跨中心比较；
  Pooch25 为二分类 0/1、须引用 Pooch et al. 2025）与 G0-R 证据边界（16 例、相对指标、合成校准）；
- §6：由"当前 G0 状态"改为"数据可用性与正式 run 启动条件"（状态迁出）；
- §8.8（新增）：容量与混杂控制（M2 vs M1 的 gate 参数容量混杂；未设同容量对照则 H2/G3 降级为探索性）；
- §9.1/§9.2/§9.4/§9.4.1/§9.4.2：删除工程完成清单与实现状态，保留规范性要求（低频验证 checklist 迁入
  `docs/P2_M0_Implementation.md` 附录 A；G2 前置/后置清单改为要求表）；
- §11.1/§11.4：区分训练期 checkpoint 实现字段 `val_positive_casewise_dice_mean` 与最终 test 语义终点
  `positive_casewise_dice`；明确共同阈值 vs 每模型独立阈值属待冻结选择；
- §12.4：补先验独立性"未审计"的现状与降级规则；
- §14：改为门定义表 + 逐门补充约束（状态迁出到 `STATUS.md`）；
- §15/§16：阶段与依赖保留，去掉进度标记；代理权限/conda/进度条规则不再复述（指向 `AGENTS.md`），
  新增 Git 缺失的复现性 blocker 要求；
- §19：新增 gate 塌缩（含邻接模态门控工作的提示）、M2/M1 容量混杂、MCID 缺失、阈值口径、G0-R 证据边界、
  无版本控制、新颖性检索缺失等风险行；
- §20：改为运行入口索引 + **待冻结决策表 D1–D10** + 纪律提醒（命令与"近期下一步"迁出）；
- §21：精简为核心来源表（完整清单在 `docs/literature/references.md`）。

**一致性修复**

1. `configs/protocols/g0_e_independent_test.yaml`：`audit_tool.status` 由 `IMPLEMENTED_NOT_EXECUTED` 更正为
   `INVENTORY_EXECUTED_EVIDENCE_AUDIT_NOT_EXECUTED`，并记录执行产物路径与 `config_sha256`
   （依据：`docs/experiment_log.md` 2026-09-15 条目、`docs/Development_Log.md` v2.3 同步条目、
   `outputs/diagnostics/g0_e/20260915_093835/run_metadata.json`）；
2. `docs/protocols/G0_E_INDEPENDENT_TEST.md` §11.1 同步更正，并写明"配置在运行后被修订 → 若以新配置冻结须重跑清点"；
3. `docs/research_plan.md` §20 的过期矛盾（P2A 已完成 vs 并行推进实现）随 §20 重写消除；
4. `configs/protocols/g0_sap.yaml` + `docs/protocols/G0_SAP.md`：新增主终点的**语义映射**（未改实现字段名）；
5. `README.md`：状态收敛为摘要 + 指向 `docs/STATUS.md`；`docs/project_structure.md`：新增文档职责表。

**静态校验（实际运行；只读、秒级、不读真实医学数据、不用 GPU）**

| # | 命令 | 结果 |
|:--|:--|:--|
| 1 | `yaml.safe_load` 解析 4 个 `configs/protocols/*.yaml` | 全部通过；`g0_e` 的 `audit_tool.status` 已更新，其余 `protocol.status=DRAFT` |
| 2 | `python scripts/evaluate/prepare_g0_sap_freeze.py --check-freeze` | 仍为 `NOT READY`（24 个字段待填 + `power_memo.status=insufficient_data`），**阻塞项数量未因新增映射字段改变** |
| 3 | `git rev-parse --is-inside-work-tree`；`git -C third_party/nnUNet rev-parse HEAD` | 主项目**不是 Git 仓库**（已记为 blocker B1）；`third_party/nnUNet` = `74ceb6803d10dcee29b2cc481678d3a3d069f281`（tag `v2.6.2`，与文档一致） |
| 4 | `ls outputs/diagnostics/{g0_r,g0_r_automated,g0_e}`、`outputs/nnUNet_results/Dataset605_PICAI`、`outputs/checkpoints` | 核实：G0-R 抽样与 G0-E 清点产物存在、`g0_r_automated` 不存在（真实 QC 未运行）、NoFFT 输出目录不存在、`outputs/checkpoints` 为空 |

**未做**：未运行任何数据处理、重采样、审计、验证、推理、评测或训练；未修改 `src/`、`tests/`、数据、split、
plan、`third_party/`；未删除或覆盖任何既有日志与产物；未初始化 Git；未把任何门标为 PASS；
`configs/protocols/g0_e_independent_test.yaml` 的内容哈希因状态字段修订而变化（旧运行的绑定哈希见
`docs/protocol_changelog.md` 2026-09-17 条目）。

### 验收轮修正：v2.3.1 amendment + G0-SAP 生命周期 + 状态单源（2026-09-17；仅文档与协议载体）

**背景（验收意见）**：重构主体通过，但有 3 项未完成：① 本轮实际新增了实质性协议要求，不能写成「协议条款未变」；
② G0-SAP 存在「评测逻辑不得在冻结前实现」与「必须先实现才能冻结」的循环，且「不得查看模型指标」与
「在 validation 上选阈值」冲突；③ 多个 runbook 与协议文档仍直接写动态事实（是否运行过、缺失字段数、测试数）。

**修正 1：把实质性新增登记为正式 amendment `v2.3.1`（在正式模型结果前新增）**

- 四项实质性条款（均发生在任何正式模型结果之前）：
  1. **G0-SAP 三阶段生命周期（SAP-A 方法冻结 / SAP-B validation 填数 / SAP-C 封存后最终 test）** 与
     预测可见性字段拆分（`sees_validation_predictions` / `sees_final_test_predictions`）；
  2. **MCID 与主判定规则**（成功标准不得只有「CI 不跨 0」）；
  3. **M1/M2 容量混杂控制**（同容量对照或显式降级 H2/G3 为探索性）；
  4. **G0-R 证据充分性决策 + 文献系统检索要求**。
- 登记位置：`docs/protocol_changelog.md` 新增 `v2.3.1` 条目（含修改原因与对既有结果的影响），表格新增版本行，
  并明确「本轮不是仅文档迁移」；`docs/research_plan.md` 抬升版本号为 v2.3.1 并在 §0.1 列出四项新增。

**修正 2：G0-SAP 生命周期（消除循环与冲突）**

- `docs/research_plan.md` §11.4：新增 SAP-A/B/C 时点表（允许看到什么、必须完成什么、禁止什么），并明确
  「`scripts/evaluate/` 的**实现**必须在 SAP-A 冻结前完成；**在真实 validation/test 上运行**只能在 SAP-B/C」；
  §14 的 G0-SAP 通过条件与失败转向同步更新。
- `docs/protocols/G0_SAP.md`：新增 §0.1 生命周期与字段语义；§5 功效备忘录加入 `mcid`/`mcid_source` 行与来源纪律；
  §7 变更规则改为按阶段区分（SAP-A 只允许填数、方法性修改必须升版）；§8 输出字段加入差异审计与预测可见性；
  §9 改为「工具能力与实现待办」（移除实测结果与循环表述），并列出三项实现待办（评测脚本、拆分字段校验、
  MCID 纳入冻结校验）；协议版本 `draft-0.1` → `draft-0.2`。
- `configs/protocols/g0_sap.yaml`：新增 `sees_validation_predictions` / `sees_final_test_predictions`（保留
  `sees_model_predictions` 为兼容字段，语义 = final-test 可见性）、新增 `sap_phases` 块、`power_memo` 增
  `mcid`/`mcid_source`；版本同步 `draft-0.2`。
- `docs/runbooks/g0_sap_freeze.md`：重写为 SAP-A/B/C 三段命令与停止条件，显式澄清「不得查看论文指标」指
  最终 test 指标（SAP-B 允许使用 validation 开发指标，但仅限 SAP-A 冻结算法下一次性使用）。

**修正 3：动态事实收敛到 `docs/STATUS.md`**

- `docs/runbooks/`：5 个手册的「当前状态」行删除或改为指向 `docs/STATUS.md`；`README.md` 的索引表去掉状态列；
  `g0_e_candidate_audit.md` 把「配置已被修订」改写为通用的**哈希绑定规则**；`g0_sap_freeze.md` 不再写
  「24 个字段」等实测数字。
- `docs/protocols/`：`G0_R_ALIGNMENT_QC_AUTOMATED.md`（删除测试数与真实运行状态）、`G0_R_ALIGNMENT_QC.md`
  （§10 改为能力与待办，删除 `[x]` 执行记录）、`G0_E_INDEPENDENT_TEST.md`（§11.1 改为能力与状态语义 + 哈希绑定规则）、
  `G0_SAP.md`（§9 同上）。协议文档只保留 `DRAFT/FROZEN` 与字段语义。
- `docs/STATUS.md`：新增「状态唯一来源声明」段（动态数字只允许出现在 `STATUS.md` 与 `experiment_log.md`）；
  更新计划版本为 v2.3.1；G0-SAP 行补记 SAP-A/B/C 均未开始与 MCID 校验待办；补充版本引用
  （`README.md`、`PICAI_Model_Readiness_Audit.md`、`P2_M0_Implementation.md` 的「以研究计划 v2.3 为准」→ v2.3.1）。
- `docs/protocol_changelog.md`：删除「242 passed」运行事实（改为指向 `docs/experiment_log.md`），
  并在文档重构条目中把「不改变任何协议条款」改为「除已登记为 v2.3.1 的四项新增外」。

**验证（只读静态检查 + 合成 CPU 测试；记录见 `docs/experiment_log.md`）**：4 个协议 YAML 解析通过；
`prepare_g0_sap_freeze.py --check-freeze` 仍为 `NOT READY`（24 字段，未伪造）；`g0_sap.yaml` 新哈希
`bdc14967…`；全量 `tests/unit` **481 passed / 0 failed**。

**未做**：未修改 `src/`、`tests/`、数据、split、plan、`third_party/`；未运行数据处理/审计/验证/推理/评测/训练；
未把任何门标为 PASS；未初始化 Git。

### G0-SAP 生命周期闭环（2026-09-17；代码 / 配置 / 测试 / 文档；未运行真实数据）

**目标**：让 SAP-A（冻结方法）/ SAP-B（填写校准结果）/ SAP-C（封存并最终测试）在机器可读 schema、校验器、
runbook 与协议文档之间完全一致。

**1. 机器可读 schema（`configs/protocols/g0_sap.yaml`，版本 `draft-0.2` → `draft-0.3`）**

- 新增**独立结果块 `sap_b_results`**（初始全 `null`）：`status`、`threshold_policy`、
  `selected_foreground_threshold`、`selected_foreground_threshold_per_model`（`model_id → value` 映射）、
  `selected_connectivity_3d`、`selected_min_lesion_volume_mm3`、`source_sap_a_config_sha256`、
  `frozen_checkpoint_manifest_sha256`、`validation_prediction_manifest_sha256`、`selection_report_sha256`、
  `sap_b_diff_audit_sha256`、`completed_at`、`reviewer`、`notes`；**SAP-B 只写该块**；
- `protocol.status_semantics` 明确 `FROZEN` 仅表示 SAP-A；`sap_phases.sap_a` 新增 `method_sha256` 与算法标识；
- YAML 顶部注释改为三阶段填写时点说明（SAP-A 必填/冻结、SAP-B 结果字段有意保持 null、SAP-C 前不得查看最终 test）。

**2. 校验器（`src/zonal_reliability_fusion/protocols/g0_sap.py`）**

- 新增常量：`SAP_A_METHOD_PATHS`（方法块路径清单，含 `power_memo` 全部字段）、
  `SAP_A_STATUSES` / `SAP_B_STATUSES` / `SAP_C_STATUSES`、`THRESHOLD_POLICIES`、
  `SAP_B_REQUIRED_RESULT_FIELDS` / `SAP_B_POLICY_FIELDS` / `SAP_B_ALL_FIELDS` / `SAP_B_HASH_FIELDS`、
  `SAP_A_METHOD_HASH_ALGORITHM`；
- 新增函数：`canonical_json`、`sap_a_method_snapshot`、`sap_a_method_hash`、`diff_sap_a_methods`、
  `validate_sap_phases(doc, sap_a_baseline=None)`；`validate_g0_sap_config` / `assert_g0_sap_config`
  新增 `sap_a_baseline` 关键字参数并纳入阶段校验；
- 强制不可变式（9 条，见 `docs/protocols/G0_SAP.md` §9.2）：兼容字段恒等、`FROZEN` ⇔ SAP-A 冻结、
  SAP-A 未冻结不可有预测可见性、SAP-A 冻结必须记录方法块哈希且与实际一致、SAP-B 未开始不得预填结果、
  SAP-B `COMPLETED` 必须字段/SHA256/可见性齐全且 `source_sap_a_config_sha256` 等于方法块哈希、
  `shared`/`per_model` 二选一且不得混填、SAP-C 前最终 test 不可见、基线差异审计拒绝任何方法字段改动；
- `POWER_MEMO_FIELDS` 与 `FREEZE_REQUIRED_FIELDS` 纳入 `power_memo.mcid` / `mcid_source`，并新增
  `sap_phases.sap_a.method_sha256`；备忘录模板新增 MCID 来源列与"SAP-A only / SAP-B 不得修改"规则；
  冻结清单改为 SAP-A/B/C 三步并打印方法块哈希；`stats_config_template` 增加方法块哈希与阶段状态。

**3. 共享校验（`src/.../protocols/common.py`）**

- `validate_protocol_block`：当协议声明 `sees_final_test_predictions` 时，强制
  `sees_model_predictions` 与之恒等（G0-R/G0-E 等未声明该字段的协议保持原行为：仍拒绝 `true`）。

**4. 冻结载体工具（`scripts/evaluate/prepare_g0_sap_freeze.py`）**

- 新增 `--sap-a-baseline`（结构化差异审计）；打印 SAP-A 方法块哈希与三阶段状态；
- 冻结包新增 `sap_a_method_hash.txt`（算法、哈希、填入位置、用途）与 `sap_a_snapshot.json`
  （差异审计基线）；`run_metadata.json` 记录 `sap_phases` 与方法块哈希；收尾提示改为 SAP-A/B/C 三步。

**5. 测试（合成 CPU；`tests/unit/`）**

- `test_g0_sap_freeze.py`：fixture 更新（MCID 字段、`sap_phases`、`sap_b_results`），新增
  `fully_filled_sap_a` / `completed_sap_b` 构造器与 18 项测试（MCID 阻止冻结、SAP-A 冻结需协议状态与哈希、
  可见性组合、方法块改动拒绝、SAP-B 预填拒绝、shared/per_model 通过、结果/哈希缺失失败、
  source hash 不一致失败、SAP-C 可见性门控、基线差异审计、差异路径报告、哈希稳定性/敏感性）；
- `test_g0_scripts.py`：冻结拒绝用例改为"自洽但未就绪的 SAP-A 冻结声明"；冻结包文件清单加入两个新文件；
- `test_g0_protocol_common.py`：保持原 G0-R 行为（无拆分字段时仍拒绝 `sees_model_predictions=true`）。

**6. 文档同步**

- `docs/protocols/G0_SAP.md`：§0.1 增写入范围列与 `sap_b_results` 字段表、方法块哈希与差异审计小节；
  §5 明确 power memo 全属 SAP-A、SAP-B 不得修改、当前无白名单字段；§7 变更规则按阶段重写；
  §8 输出字段更新；§9 改为「能力要求 + 9 条校验不可变式 + 实现待办」；
- `docs/runbooks/g0_sap_freeze.md`：SAP-A 增加方法块哈希步骤；SAP-B 只写 `sap_b_results` 并给出
  `--sap-a-baseline` 命令；SAP-C 补充兼容字段同步；
- `docs/research_plan.md` §11.4/§14.1：生命周期表增加写入范围列、方法块哈希、power memo 时点纪律与
  顶层状态语义；
- `docs/STATUS.md`：G0-SAP 行与 B4 更新（`draft-0.3`、27 个待填字段、SAP-A/B/C 均未开始、校验器已实现）；
  `docs/protocol_changelog.md` 新增「v2.3.1 载体补记」。

**验证（记录见 `docs/experiment_log.md`）**：4 个协议 YAML 解析通过；`--check-freeze` = `NOT READY`
（27 个字段，未伪造数值）；目标测试 60 passed；全量 `tests/unit` **499 passed / 0 failed**；
`g0_sap.yaml` 新哈希 `5c5847fd…`。

**lint 范围声明（诚实）**：本次新增代码无新增 Ruff 告警；`ruff check` 在
`src/.../protocols/g0_sap.py`（UP035 导入风格、SIM102 嵌套 if）、`src/.../protocols/common.py`
（UP035 / RUF100 / RUF022）、`scripts/evaluate/prepare_g0_sap_freeze.py`
（EXE001 / I001 / RUF100×3）与两个测试文件上报告的剩余条目**均为本轮之前已存在的既有告警**，
本轮不 retro-fit（避免超出范围、改变既有行为）。

**未做**：未运行数据处理、真实验证、推理、评测或训练；未读取真实医学影像；未生成任何冻结包；
未把任何门标为 PASS；未改 `third_party/`、数据、split、plan。

### M1–M4 融合模型实现（代码交付；2026-09-17）

> 范围：只改代码/配置；**未新增任何计划、协议、状态或 runbook 文档**。
> 本条目只记录「改了什么」；实际运行事实见 `docs/experiment_log.md`。

**1. 架构配置层（`src/zonal_reliability_fusion/config/architecture.py`）**

- 新增融合家族身份常量与白名单：`FUSION_FAMILY_KEY`、`FUSION_VARIANT_M1_EQUAL/M2_IMAGE_GATE/M3_ZONE_INPUT/
  M4_CONDITIONED_GATE`、`FUSION_VARIANTS`、`MODEL_ID_TO_FUSION_VARIANT` / `FUSION_VARIANT_TO_MODEL_ID`、
  `FUSION_BRANCH_SUPPORTED`、`FUSION_MODALITY_PROJECTION_SUPPORTED`、`FUSION_SHALLOW_SKIP_SUPPORTED`、
  `FUSION_GATE_SUPPORTED`、`FUSION_CONDITIONING_SUPPORTED`、`FUSION_ZONE_ENCODER_SUPPORTED`、
  `FUSION_ZONE_INJECTION_SUPPORTED`、`FUSION_VARIANT_PRESENCE`；
- 新增 `_require_list` / `_require_str_tuple` 助手与 `_parse_fusion_family`：严格校验 `fusion_family` 节
  （family_id、modality_order、三路分支、projection、浅层 skip 聚合、四变体），
  **任何未实现的声明（gate 拓扑、zone encoder 宽度等）在加载时报错**；
  机器强制跨变体一致性：M2/M3/M4 的 gate 必须完全一致、M3/M4 的 zone encoder 必须完全一致；
- `ArchitectureSpec` 新增字段 `fusion_family`（不进入哈希）与 `fusion`（**进入哈希**）及只读属性
  `fusion_variant` / `fusion_model_id`；`structural_dict()` 仅在 `fusion` 非空时追加 `fusion` 键 →
  **M0 架构哈希逐位不变**；`identity()` 在融合模型上额外输出 `fusion_variant` / `model_id`；
- 新增 `resolve_fusion_spec(spec, variant)`：把变体解析成进入哈希的单模型融合块；
- `ArchitectureRef` 新增可选字段 `fusion_variant`。

**2. 实验配置层**

- `src/zonal_reliability_fusion/config/experiment.py`：解析 `architecture.fusion_variant`（缺省 None，legacy/M0 行为不变）。

**3. 模型层**

- 新增 `src/zonal_reliability_fusion/models/fusion_blocks.py`：`ModalityProjection`、`ShallowSkipAggregator`
  （构造时执行 §8.6 契约守卫，禁止读取 pz_tz/wg/gate/decoder 特征）、`ImageDrivenGate`（logits 层零初始化 →
  初始严格等权 1/3）、`ZoneEncoder`（2 通道 PZ/TZ → adaptive_avg_pool3d 到融合 stage → conv stem + 2 个残差块）、
  `ConditionalAffineHead`（输出 gamma/beta，末层零初始化 → 初始恒等调制）、`ZoneInjection`（M3 融合后加性注入）；
- 新增 `src/zonal_reliability_fusion/models/fusion_multi_branch.py`：`MultiBranchFusionUNet3D` 基类 +
  `M1EqualFusionModel` / `M2ImageGateFusionModel` / `M3ZoneInputFusionModel` / `M4ConditionedGateFusionModel`。
  结构：三路模态独立浅层分支（stem + stage0/1/2）→ stage0/1 用 Q_l 聚合 skip → stage2 融合（M1 严格 1/3 等权；
  M2/M4 逐体素 softmax；M4 用 `L=(1+tanh(gamma))·L_img+beta`）→ 共享 encoder stage 3..6 → 普通 U-Net decoder；
  实现 `REQUIRED_INPUT_KEYS`、`model_identity()`、`parameter_breakdown()`、`summary()/format_summary()`、
  `set_diagnostics()/gate_diagnostics()`（诊断量 detached，默认关闭）；
- `src/zonal_reliability_fusion/models/factory.py`：新增 `build_model(cfg, plan)` 按 `experiment.model_id`
  分派 M0–M4（M0 保持 `build_m0_model` 行为；M1–M4 必须声明且匹配 `fusion_variant`；**未知 model_id、
  变体缺失或与 model_id 不符、M0 声明 fusion_variant 一律报错**）；新增 `model_identity()`；
- `src/zonal_reliability_fusion/models/__init__.py`：导出新类与工厂接口。

**4. 配置产物**

- 新增 `configs/architectures/fusion_resenc_v23.yaml`（M1–M4 融合家族；骨干字段与
  `m0_resenc_v23.yaml` 逐值一致）；
- 新增 4 个实验配置：`m1_equal` / `m2_image_gate` / `m3_zone_input` / `m4_conditioned_gate`
  `_picai_3d_fullres_v23.yaml`，各自独立 `paths.output_root`（`outputs/checkpoints/<tag>`）。

**5. 训练入口泛化（M0 命令兼容）**

- `scripts/train/train_m0.py`：`build_model_and_loss` 改用 `build_model`；快照增加
  `model_identity`（model_id/backbone/融合结构/架构哈希）；metrics/diagnostics/loader-smoke/overfit
  目录按 `experiment.model_id` 分目录（M0 仍为 `outputs/metrics/m0/...`、`outputs/diagnostics/m0/...`，**不变**）；
- `scripts/train/validate_m0_setup.py`：改用 `build_model`。

**6. 测试**

- 新增 `tests/unit/test_fusion_models.py`（43 项）：M1–M4 shape/DS 顺序、forward+backward 梯度有限、
  AMP 前向、M1 严格 1/3、M2/M3/M4 逐体素权重和为 1、M4 零初始化恒等、M4 条件头激活后 prior 改变 gate、
  M3 prior 不改变 gate logits、M1/M2 拒绝 zonal、M3/M4 缺 zonal 报错、WG 3 通道拒绝、prior 空间/取值校验、
  非有限输入与错误通道数失败、浅层 skip 契约守卫、stage-2 融合同时进入共享 encoder 与 decoder skip、
  跨模型身份隔离、参数量分解与哈希稳定、固定 seed 可复现、工厂分派与拒绝路径、真实融合配置四变体解析、
  声明缺失/改坏即报错。

**参数量分解（真实 plan 3d_fullres，合成构建，无数据读取）**

| 模型 | architecture_sha256 | 总参数 | gate | zone encoder | conditioning | zone injection | 浅层 skip | 三分支 | 共享后端 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M0 | `8bb5127e…`（不变） | 148,193,644 | — | — | — | — | — | — | — |
| M1 | `a039e73e…` | 155,367,756 | 0 | 0 | 0 | 0 | 15,648 | 10,663,104 | 144,638,700 |
| M2 | `8e263fd6…` | 155,417,679 | 49,923 | 0 | 0 | 0 | 15,648 | 10,663,104 | 144,638,700 |
| M3 | `80412d4a…` | 155,551,343 | 49,923 | 112,800 | 0 | 20,864 | 15,648 | 10,663,104 | 144,638,700 |
| M4 | `d9531261…` | 155,531,797 | 49,923 | 112,800 | 1,318 | 0 | 15,648 | 10,663,104 | 144,638,700 |

- **容量混杂事实（必须与 H2/G3 结论一起报告）**：M2 相对 M1 多出 gate 49,923 个参数（+0.032%），
  M2−M1 的差异同时包含「动态融合」与「这部分容量」，未设同容量对照前不得把 M2−M1 全部归因于 gate；
- M3 与 M4 的 zone encoder 参数量相同（112,800），差异只在注入方式（20,864 加性投影 vs 1,318 条件头）。

**未做（诚实边界）**：未运行任何真实数据 loader/overfit/训练/推理/评测；未实现 zonal prior 的物化工具与
prior 感知的 patch crop/滑窗（M3/M4 目前**只有模型侧接口**，训练入口尚未接入 prior）；
未补齐 6 项增强；未改动 `third_party/`、数据、split、plan；未读取真实医学影像。

### P0 架构修正 + PZ/TZ prior 端到端链路（代码交付；2026-09-17）

> 只改代码/配置/测试；**未新增任何协议、STATUS、runbook、研究计划或文献文档**。
> 本轮**未运行**任何真实数据处理、loader smoke、验证、推理、评测、GPU overfit 或训练。

**1. P0-A：gate 输入错误（实质 bug）**

- 修正前 `ImageDrivenGate` 读取**投影前**的 stage2 特征 `F_m`；修正为读取 modality projection 输出
  `U_m = P_m(F_m)`，即 `L_img = G_img([U_T2W,U_ADC,U_HBV])`、`F_fuse = Σ softmax(L)_m · U_m`；
- 新增 hook spy 测试：断言 gate 收到的三个张量与三个 projection 的输出**逐值相等**（`atol=0, rtol=0`）。

**2. P0-B：M3 公式偏差**

- `ZoneInjection`（`F_fuse + Q([F_fuse,E_z])`）→ 重命名并重写为 `ZoneFusionProjection`，
  **严格实现** `F_out = Q([F_fuse,E_z])`（concat → 1×1×1 Conv → IN → LeakyReLU），**无隐式残差相加**；
- 配置声明同步：`zone_injection.type: post_fusion_projection`、`output: Q([F_fuse,E_z])`、`implicit_residual: false`；
- 新增测试：输出严格等于 `Q([·])`，且**不等于** `F_fuse + Q([·])`。

**3. P0-C：架构声明与实现不一致**

- gate `logits_topology`、conditioning `head_topology` 补上遗漏的 `instance_norm3d`
  （实际实现：`conv1x1x1 → InstanceNorm3d → LeakyReLU → conv1x1x1`）；
- zone encoder 补全 `stem_topology` / `block_type` / `block_topology` / `block_zero_init_last_norm: false` / `shortcut`；
- `FUSION_GATE_SUPPORTED.logits_source` 由 `three_way_stage_features` 改为 `modality_projection_outputs`；
- 融合架构版本 **v2.3 → v2.4**：新增 `configs/architectures/fusion_resenc_v24.yaml`（旧 v23 文件为本人上轮创建、
  从未绑定任何 run/checkpoint，已随升版移除），4 个实验配置同步到 v2.4；
- **M0 架构哈希逐位不变**：`8bb5127e5c9a99ca1a0631cb365cdcf0b866441ac225911e17faa6bc995bbee5`；
  M1–M4 新哈希（v2.4）：M1 `54b66e33…`、M2 `77d1de43…`、M3 `097c19ed…`、M4 `8a12527d…`；
- 测试不再复制 FUSION_FAMILY：直接读取真实 v2.4 配置的 `fusion_family` + `conftest.build_resenc_arch_doc(plan)`
  构造 mini plan 版本；并新增「改坏 topology/存在性/implicit_residual 必须报错」的负例测试。

**4. P0-D：GPU 热路径**

- `forward` 不再调用 `torch.isfinite(...).all()` / `float(tensor.min())` / `float(tensor.max())`
  （每 iteration 的 GPU 同步已移除）；只保留 O(1) 的 shape/通道/网格检查；
- 完整校验（finite / 逐通道 [0,1] / **PZ+TZ <= 1+tol** / WG 3 通道拒绝）移入
  `validate_zonal_prior`（`models/fusion_blocks.py`）与 CPU 数据入口（`PatchDataset`）；
  模型侧提供 `validate_inputs()` 与 `strict_input_validation=True` 显式开关（默认关闭）；
- 新增测试：PZ=TZ=1 的 overlap、越界、NaN、WG 通道、错误通道数全部失败。

**5. P1：PZ/TZ prior 物化工具（新文件）**

- `src/zonal_reliability_fusion/data/zonal_prior.py`：plan-space 转换（**transpose → crop(properties 中的图像
  nonzero bbox) → resample**，与 nnU-Net v2.6.2 `DefaultPreprocessor.run_case_npy` 顺序一致，最近邻由
  `configuration_manager.resampling_fn_seg` 注入）、`label_to_pz_tz`（0/1/2 → [2,D,H,W]）、
  `oracle_check_lesion`（同一转换应用于原始 lesion，必须与 `.b2nd` seg 逐体素一致）、sidecar 读写与
  `ZonalPriorMaterializer`（逐例成功/跳过/失败 + 失败清单 + 统计）；
- `scripts/data/materialize_zonal_prior.py`：`--source yuan|hevi`（必填）、读取 0/1/2 标签、复用官方几何、
  lesion oracle 不一致即拒绝该例、写 `<out_root>/<case_id>.npz(+.json)`（含来源/plan 哈希/几何/输入输出哈希/
  标签映射/oracle 结果/失败原因）、默认不覆盖、`--resume` 安全续跑、`--overwrite` 显式重算、tqdm 进度与
  `--no-progress`、结束打印成功/失败/跳过/耗时/输出路径；**不修改任何 `.b2nd` / raw / split / plan**。

**6. P1：数据链路 / 训练 / 推理接入**

- `PreprocessedStore`：新增 `zonal_prior_root`/`zonal_prior_source`（必须成对、来源白名单）+
  `get_case(..., include_zonal_prior=)` + `load_zonal_prior()`（sidecar 缺失、来源不符、形状≠seg、元数据 shape
  不一致、越界、overlap 全部报错，不静默兜底）；
- `PatchSampler`：`sample(..., zonal_prior=)` → `PatchSample.zonal_prior`，image/seg/prior 使用**同一 bbox**
  （`prior_bbox == bbox` 由测试锁定）与**同一 padding**（prior padding 值 0）；
- `AugmentationPipeline` / `MirrorAugmentor`：空间变换同步作用于 prior，强度增强只作用于 image；
  未提供 prior 时返回 2-tuple（旧接口不变）；
- `PatchDataset` / `TorchBatchProvider`：`include_zonal_prior` 开关；M0–M2 仍是 `(data, seg)`，
  M3/M4 为 `(data, seg, prior)`；CPU 入口校验 prior（有限/[0,1]/overlap），缺 prior 立即 `BatchProviderError`；
- `M0Trainer`：`_to_device` 支持 2/3-tuple；`_forward_model` 按 `REQUIRED_INPUT_KEYS` 调用
  `model(image)` 或 `model(image, zonal_prior=...)`，缺 prior / 多余 prior 都立即报错；
- `sliding_window_predict(..., zonal_prior=)`：与 image 相同 padding、相同窗口坐标、相同 batch、相同镜像翻转
  （测试用坐标场验证每个 patch 的 prior 与 image 逐体素对齐）；`FullVolumeValidator` 从 store 读取同病例 prior
  并在 run() 中与模型声明核对（不一致即失败）；
- 诊断：`set_diagnostics_mode("off"|"stats"|"cpu")`（默认 off；stats 只留标量，cpu 才 offload 完整张量）；
- 复杂度：`fusion_group_macs_analytical` / `fusion_group_macs_with_hooks`（分组：三分支、projection、浅层 skip、
  gate、zone encoder、conditioning、zone injection、共享后端），测试在 mini plan 上逐组逐值对齐。

**7. 参数量 / MACs（真实 3d_fullres plan，解析；合成构建）**

| 模型 | sha256（v2.4） | 参数 | gate | zone enc | cond | inj | MACs 总 | MACs gate | MACs zone |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M1 | `54b66e33…` | 155,367,756 | 0 | 0 | 0 | 0 | 2,032,847,718,400 | 0 | 0 |
| M2 | `77d1de43…` | 155,417,679 | 49,923 | 0 | 0 | 0 | 2,037,920,204,800 | 5,072,486,400 | 0 |
| M3 | `097c19ed…` | 155,551,343 | 49,923 | 112,800 | 0 | 20,864 | 2,051,518,924,800 | 5,072,486,400 | 11,501,568,000 |
| M4 | `8a12527d…` | 155,531,797 | 49,923 | 112,800 | 1,318 | 0 | 2,049,546,291,200 | 5,072,486,400 | 11,501,568,000 |

**未做（诚实边界）**：未运行物化工具、真实 loader smoke、GPU overfit、显存实测或任何训练；
6 项增强（rotation/scaling/low-resolution/noise/blur/gamma）**留待下一轮**，本轮未实现。

### 物化前最后加固（PZ/TZ prior 链路；代码/配置/测试；2026-09-17）

> 只改代码、配置、测试；**未新增/扩写研究计划、协议、STATUS、runbook、README 或文献文档**；
> **未运行**真实物化、loader smoke、验证、推理、评测、GPU overfit 或训练。

**1. v2.4 实验身份**
- `m1_equal` / `m2_image_gate` 的 `experiment.name` 由 v23 改为 v24；四个配置文件、`architecture_version`、
  `experiment.name`、`paths.output_root` 版本一致；M3/M4 的来源标记（yuan）同时出现在实验名与输出目录；
  四个输出目录互不相同；**M0 身份与哈希未改动**。

**2. loader-smoke 的 prior 假通过**
- `train_m0.py::run_loader_smoke` 按 `requires_zonal_prior(cfg)` 分派：M0–M2 只读 image（2-tuple，
  不触碰 prior）；M3/M4 使用 `get_case(..., include_zonal_prior=True)`、`sampler.sample(..., zonal_prior=)`、
  `TorchBatchProvider(..., include_zonal_prior=True)`，解包 3-tuple 并检查 prior；
- 终端与 JSON 记录：prior source、shape、dtype、finite、min/max、`max(PZ+TZ)`、sampler 的
  `prior_bbox_identical`、batch 级 prior 统计；JSON 额外记录 `requires_zonal_prior` 与 manifest 信息；
- 缺 sidecar / 来源不符 / manifest 校验失败 / bbox 不同步 / batch 元组数目不符 → 立即 `SystemExit`，
  **不存在“prior 未读却成功”的路径**。

**3. 严格离散标签**
- `label_to_pz_tz`：先 `np.isfinite`，再用 `np.isin` 做**精确成员判断**（不再 `int(v)` 截断）；
  1.5 / 2.1 / -0.2 / NaN / Inf 一律拒绝，错误信息保留原始非法值（如 `2.0999999046325684`）。

**4. oracle fail-closed**
- `ZonalPriorMaterializer.run` 在写盘前调用 `require_oracle_pass`：metadata 必须含 oracle，
  `exact_match is True`、`shape_match is True`、`mismatch_voxels == 0`；任一不满足 → 该例 failed，
  **不写 `.npz`/`.json`**（即使 converter 忘记抛异常）。

**5. sidecar 完整性与实验绑定**
- `PreprocessedStore` 新增 `expected_plans_sha256` / `expected_configuration`（启用 prior 时必填），
  `load_zonal_prior` 逐项校验：source、configuration、plans_sha256、output_dtype == float32、
  `output_channel_semantics == [PZ,TZ]`、oracle 通过、**数组 SHA256 == metadata.output_sha256**、
  output_shape、数值契约（finite / [0,1] / PZ+TZ<=1+tol）与 seg 空间尺寸；
- `build_store` 传入当前 plan 文件哈希与 configuration；
- 新增数据集级 manifest（排序 case 记录 + input/output/metadata 哈希 + canonical `manifest_sha256`）：
  `materializer` 写完 sidecar 后写 manifest；`store.load_zonal_prior_manifest(expected= train∪val)`
  校验重复/缺失/多余/哈希自洽/来源/plan/config，并逐条与单文件 sidecar 交叉核对；
- `train_m0.py` 的 `config_snapshot["provenance"]["zonal_prior_manifest_sha256"]` 记录该哈希；
  `M0Trainer.FROZEN_CONFIG_PREFIXES` 增加 `provenance.zonal_prior_manifest_sha256`
  → **resume 时 manifest 变更会被拒绝**（不只是打印）。

**6. 原子写入与安全 resume**
- `zonal_prior` 全部产物（npz / json / manifest / summary）改为同目录临时文件 → `flush` + `fsync` →
  `os.replace`；replace 失败不会留下最终文件（测试 monkeypatch 验证）；
- `sidecar_identity_ok` 提供轻量身份检查（source/config/plan/input hashes/oracle/数组哈希/契约），
  `--resume` 先做该检查（只做文件哈希 + npz 校验，不做 resample、不读 `.b2nd` 大数组），命中即 skipped；
  身份不符 → 必须真正调用 converter；缺一半/损坏 → 失败或显式重算，绝不静默跳过。

**7. 物化性能**
- `CaseConverter` 复用单个 `nnUNetDatasetBlosc2`（惰性构造一次，计数可断言）；oracle 只读取 `seg`
  （`data` 保持官方 lazy 句柄，不物化三通道影像）；不再每例重建 + 重复扫描 identifier。

**8. dry-run 名副其实**
- `--dry-run` 对选中病例执行完整只读转换 + oracle + prior 契约 + 元数据构造，**零文件写入**
  （npz/json/manifest/summary 全部跳过），打印 ok/failed/oracle mismatch/耗时，任一失败退出码非 0；
  未限定子集时要求显式 `--case-ids` 或正数 `--limit-cases`；
- 完整运行（无子集）额外校验处理集合 == fold 0 的 train∪val，不等则退出码非 0。

**9. setup readiness**
- `validate_m0_setup.py`：M3/M4 在 `require_data=True` 时检查 prior root、manifest、source、plan hash、
  configuration 与 train∪val 覆盖；`--no-require-data` 明确输出 `PRIOR_NOT_CHECKED` /
  `NOT_TRAINING_READY` 并写入 `setup_check.json`；M1/M2 完全不检查 prior（`N/A`）。

**10. 新增/改写测试**
- 新增 `tests/unit/test_materialization_hardening.py`（30 项）：v24 配置身份与输出目录唯一性、
  loader-smoke 三向分支（M4 读 prior / M4 缺 sidecar 失败 / M2 不读 prior）、JSON prior 字段、
  非整数与 NaN/Inf 标签拒绝、oracle fail-closed（含缺 oracle）、sidecar 篡改与 5 类字段不符、
  manifest 缺/多/重复/哈希变化、resume 冻结键拒绝、原子写失败无残留、有效 resume 不调用 converter、
  半成品 sidecar 重算、dry-run 零写入、dataset 只构造一次、store manifest 往返与交叉校验；
- `tests/unit/test_zonal_prior_pipeline.py` 适配新契约（oracle 不通过 → `n_failed=1` 且无文件）

### PZ/TZ 物化与就绪检查的 P0/P1 修复（代码/配置/测试；2026-09-17）

> 只改代码、配置、测试；未新增/重构任何计划、协议、状态、runbook 或 README 文档；
> 未运行真实物化、loader smoke、验证、推理、评测、GPU overfit 或训练。

**A. sidecar 与 manifest 的正确生成（`data/zonal_prior.py`）**
- 新增 `normalize_prior()`：哈希/写盘/校验统一使用 **contiguous float32**（消除 float64 输入
  「哈希按 float64、NPZ 按 float32」的不一致）；
- 新增 `finalize_sidecar_metadata()`：集中补齐 `case_id/output_shape/output_dtype/
  output_channel_semantics/output_sha256`，并强制 `output_sha256` 为**非空 64 位小写 hex**；
- `write_prior_sidecar()` 现在返回 `(npz, json, status, 最终落盘 metadata)`：写盘后**重新读取 JSON**，
  与内存对象做 canonical 比对，不一致直接报错；`CaseOutcome.metadata` 与 manifest 只使用该最终对象；
- `metadata_sha256` 定义唯一化：`metadata_sha256_of(meta) = canonical_json(meta)` 的 SHA256，
  且只对**实际落盘 JSON** 计算。

**B. canonical manifest 发布规则（`zonal_prior.py` + `scripts/data/materialize_zonal_prior.py`）**
- `ZonalPriorMaterializer.run(..., expected_case_ids=...)`：仅当本轮病例集合**完整等于** expected、
  `n_failed == 0`、`n_ok + n_skipped == len(expected)` 时才尝试发布；
- 发布前用 `verify_manifest()` 校验 cases/case_ids/n_cases/排序/唯一/覆盖范围/来源/plan/config/自校验；
- 子集运行、dry-run、任何失败：**不创建、不覆盖**既有 `manifest.json`，summary 写明
  `manifest_published=false` 与原因；
- 全量 resume：用 `collect_verified_records()` 从磁盘**逐个校验并重建**全部 sidecar 记录
  （identity_probe 存在时同时核对来源/plan/config/输入哈希），不再只收集本轮 outcome；
- CLI 在**运行前**确定 train∪val 并做覆盖预检（不等），未通过直接退出码 2；完整运行若最终未发布
  manifest 也判为失败（退出码非 0）。

**C. sidecar 成对提交（`commit_sidecar_pair`）**
- NPZ 与 JSON 先各写临时文件（flush+fsync），再依次 `os.replace` 提交；
- 覆盖场景先把既有有效文件备份为 `.<name>.bak`，第二步提交失败时**回滚**：删除新提交文件并恢复旧对；
- 新建场景第二步失败时删除已提交的 NPZ，**不留可被误认为完整的残缺对**；
- 删除 `_atomic_write_bytes` 中重复的不可达 `raise`；
- 通过故障注入（第 2/第 4 次 `os.replace` 失败）测试证明上述两条路径。

**D. manifest / readiness 深度校验**
- `verify_manifest()`：逐条校验字段存在性、`output_sha256`/`metadata_sha256`/`input_sha256` 的
  64 位小写 hex 格式、`case_ids` 与 `cases` 顺序完全一致、按 case_id 升序、无重复、`n_cases` 一致；
- `PreprocessedStore.load_zonal_prior_manifest()`：逐例检查 `.json` 与 `.npz` **均存在**、实际
  canonical metadata SHA256 == 记录值、input/output 哈希一致、output 哈希非空且 == **实际数组**
  SHA256，并设置 `zonal_prior_array_hash_checked`（供 setup 如实报告）；
- `validate_m0_setup.py`：prior readiness 记录 `array_hash_checked`；默认（require-data）模式下
  prior 不就绪时**写出诊断 JSON 后返回退出码 1**；`--no-require-data` 仍为
  `PRIOR_NOT_CHECKED` + `NOT_TRAINING_READY`。

**E. v2.4 实验身份**
- M1–M4 的 `paths.output_root` 改为含 v24 的互异路径
  （`outputs/checkpoints/<tag>_v24`；M3/M4 同时保留 `yuan` 来源：`<tag>_yuan_v24`）；
- 测试改为**正向断言**（路径组件必须含 v24 / 不得含 v23 / M3/M4 必须含来源），并加入
  「去掉 v24 的同一路径必须判为不合法」的负向对照。

**修改文件**：`src/zonal_reliability_fusion/data/zonal_prior.py`、
`src/zonal_reliability_fusion/data/preprocessed_store.py`、`scripts/data/materialize_zonal_prior.py`、
`scripts/train/validate_m0_setup.py`、4 个 `configs/experiments/*_v24.yaml`（仅 output_root）、
`tests/unit/test_materialization_hardening.py`（含 10 项新回归测试）。

**未做（诚实边界）**：未物化任何真实病例；未运行 loader smoke / 训练 / 推理 / 评测；
M3/M4 仍**不是**训练就绪；6 项增强未开始。

### PZ/TZ 物化链最后一轮代码加固（2026-09-17）

> 只改代码/测试；未修改 research_plan / STATUS / protocol / runbook / README；
> 未运行真实物化、loader smoke、GPU、训练、推理或评测。

**A. `commit_sidecar_pair` 统一异常状态机（`data/zonal_prior.py`）**
- staging（2 个 tmp，flush+fsync）→ 备份 → 提交 NPZ/JSON → **落盘复核**（JSON canonical + 数组 SHA256 + shape）
  全部纳入**同一个** try/except；精确记录 `staged` / `backups` / `committed` 三张清单；
- 任一步异常 → `_rollback_pair`：删除本轮新文件与临时文件、`os.replace(backup, final)` 恢复全部旧文件；
- 备份文件名为本轮唯一（`.<name>.<pid>.<uuid10>.bak`），**不再删除固定名历史 .bak**；
- 无法恢复时抛 `SidecarCommitError(case_id, stage, original, recovery_errors)`，消息同时含原始错误与
  恢复错误，并保留 `.bak` 供人工恢复（绝不静默 suppress）；
- 表述修正为「异常安全 + 失败回滚 + 读取端 fail-closed」，不再声称跨路径崩溃级事务原子。

**B. 写盘 fail-closed（公开路径不再依赖上游检查）**
- `commit_sidecar_pair` 内：`normalize_prior` → `validate_prior_array` → `_require_writable_metadata`
  （source ∈ {yuan,hevi}、configuration 非空、plans_sha256 与 input_sha256 为合法 hex）→
  `require_oracle_pass` → 写盘 → 落盘复核；任一不满足即拒绝且不留文件；
- `write_prior_sidecar(resume=True)` 的**弱比较已删除**，改用统一严格校验器：损坏数组、错误
  configuration、错误 channel semantics、oracle 失败一律**不返回 skipped**（重写修好）。

**C. 统一深度校验器 `validate_sidecar()`**
- 单函数覆盖：case_id、source、configuration、plans_sha256、input_sha256（与期望一致且格式合法）、
  metadata_sha256（== 实际 canonical JSON）、output_sha256（格式 + 与期望一致）、oracle 严格通过、
  output_shape（== 实际数组，可再比对 seg 空间尺寸）、output_dtype、channel semantics、
  数组 SHA256、`validate_prior_array`（finite / [0,1] / PZ+TZ<=1）；返回 `ValidatedSidecar`
  （metadata + 已读取 prior + 两种哈希 + array_hash_checked），避免同一 NPZ 重复解压；
- `sidecar_identity_ok` 退化为该函数的布尔包装；`PreprocessedStore.load_zonal_prior` 与
  `load_zonal_prior_manifest` 均改为调用它（三处规则不再各自维护）。

**D. 全量校验的进度与统计**
- `load_zonal_prior_manifest(..., verify_array_hash=True, progress=False, desc=...)`：库默认**不打印**、
  不建进度条；`progress=True` 时 tqdm 显示当前/总数/比例/速率/ETA；
- 逐例失败**全部收集**后再抛（消息含 n_failed 与前 5 条例），统计写入
  `store.zonal_prior_validation_stats`（n_cases/n_ok/n_failed/array_hash_checked/elapsed_sec/manifest_sha256/cached）；
- 缓存键 = (病例集合, verify 标志, manifest+各 sidecar 的 mtime/size 指纹)：同进程重复调用不重扫，
  文件被改动（mtime/size 变化）立即失效重扫；
- 三个入口接线：`validate_m0_setup.py`（新增 `--no-progress`）、`train_m0.py` 正式训练启动、
  `train_m0.py --mode loader-smoke`，均在脚本层打印统计。

**E. 描述与快照顺序**
- `validate_m0_setup.py`：删除「不读取任何病例体素」表述，改为「不读取 MRI/lesion .b2nd 内容、
  不做 forward/GPU；require-data 模式会顺序读取 derived PZ/TZ NPZ 并校验哈希」；
- `train_m0.py`：新增 `assert_prior_snapshot_consistency()`，并在 `build_trainer` 中**先**完成
  manifest 深度校验（`load_prior_manifest_sha256`）再生成 `train/val_pipeline` 快照，强制
  `pipeline.zonal_prior.manifest_sha256 == provenance.zonal_prior_manifest_sha256` 且
  `array_hash_checked is True`；同一次启动借助 store 缓存不重复扫描；
- 顺手清理 `train_m0.py` 既有静态问题（EXE001/DTZ005/PLR0124/RUF059），`np.isfinite`/`torch.isfinite`
  统一到 `_all_finite()`（避免 numpy↔torch 桥接弃用警告）。

**修改文件**：`src/zonal_reliability_fusion/data/zonal_prior.py`、
`src/zonal_reliability_fusion/data/preprocessed_store.py`、`scripts/train/train_m0.py`、
`scripts/train/validate_m0_setup.py`、`tests/unit/test_materialization_hardening.py`、
`tests/unit/test_zonal_prior_pipeline.py`。

**仍未做**：未物化真实病例、未跑真实 loader smoke / 训练 / 推理 / 评测；M3/M4 仍**不是**训练就绪。

### PZ/TZ 链路最后小范围修补（4 项；2026-09-17）

> 只改必要代码与测试；未修改 research_plan / STATUS / protocol / runbook / README；
> 未运行真实数据、GPU、训练、推理或评测。**未重写 sidecar 异常回滚状态机**。

1. **真实进度条**（`data/preprocessed_store.py`）：`list(tqdm(iterator, ...))` → 惰性 `tqdm(iterator, ...)`，
   进度与逐例校验交错推进（`progress/a → validate/a → progress/b → validate/b`），统计仍由脚本层输出。
2. **NPZ 真实存储 dtype**（`data/zonal_prior.py`）：`read_sidecar_prior` 不再无条件转 float32，
   保留 NPZ 内 `pz_tz` 的原始 dtype；`validate_sidecar` 先要求**真实 dtype == float32**，
   通过后才 `normalize_prior` 参与 shape/哈希/契约校验；`commit_sidecar_pair` 落盘复核同样校验 dtype。
   因此「float16 存储 + metadata 谎报 float32 + 全部哈希同步重算」仍被拒绝。
3. **训练逐例读取绑定冻结 manifest**（`data/preprocessed_store.py`）：
   `load_zonal_prior_manifest` 成功后将 `case_id → {input_sha256, output_sha256, metadata_sha256}`
   存为**不可变**（`MappingProxyType`）冻结记录；`load_zonal_prior`/`get_case(include_zonal_prior=True)`
   在记录存在时把三项期望传给 `validate_sidecar` 逐例复核；新增 `require_frozen_manifest` 开关，
   `scripts/train/train_m0.py::build_store` 对 M3/M4 置 True → 未通过深度预检不得读取 prior。
4. **失败清空 READY + 删除弱缓存**：每次深度校验开始前复位
   `zonal_prior_manifest_sha256=None`、`array_hash_checked=False`、冻结记录清空；
   全部成功后才一次性提交，并返回 manifest 深拷贝。**删除** `_manifest_fs_stamp` / 跨调用缓存
   （setup / loader-smoke / build_trainer 各只需扫描一次，mtime+size 弱指纹不承担完整性职责）。
5. **warning 清理**（`scripts/train/train_m0.py`）：batch prior 的 finite 检查改用 `_all_finite(prior)`
   （torch 用 `torch.isfinite`），不再对 Torch tensor 调用 `np.isfinite`，两条 NumPy `__array_wrap__`
   警告消失（全量 warnings 9 → 7）。

**修改文件**：`src/zonal_reliability_fusion/data/zonal_prior.py`、
`src/zonal_reliability_fusion/data/preprocessed_store.py`、`scripts/train/train_m0.py`、
`scripts/train/validate_m0_setup.py`（统计打印）、`tests/unit/test_materialization_hardening.py`（+8 项）、
`tests/unit/test_zonal_prior_pipeline.py`（store 属性补齐）。

### 数据链与 --overfit 显存遥测修补（代码/脚本/测试；2026-09-17/18）

> 只改代码、脚本与合成测试；未新增/重构计划、协议、状态或 runbook 文档；
> 未运行真实数据、GPU、训练、推理或评测。

**1. 微型修补：dry-run 进度 + 冻结记录深层不可变**
- `data/zonal_prior.py`：`ZonalPriorMaterializer.run` 的 `if progress and not dry_run:` → `if progress:`
  （dry-run 同样逐例显示进度；转换与 oracle 仍逐例执行，语义不变）；
- `data/preprocessed_store.py`：冻结记录的**嵌套** `input_sha256` 也包装为 `MappingProxyType`
  （外层与内层均不可变，修改触发 `TypeError`）；
- 测试：`tests/unit/test_materialization_hardening.py` 88 → **91**（进度与 converter 逐例交错、
  dry-run 零写入、深层不可变）。

**2. 真实物化脚本 reader 实例化修复**
- `scripts/data/materialize_zonal_prior.py::build_context`：
  `plans_manager.image_reader_writer_class()()` → `image_reader_writer_class()`；
- 只读核对依据（fixed third_party/nnUNet + `lm` 环境）：`PlansManager.image_reader_writer_class`
  是 `@property`（getter 返回 `Type[BaseReaderWriter]`），`SimpleITKIO()()` 抛
  `TypeError: 'SimpleITKIO' object is not callable`；`CaseConverter._read_seg_only` 正是消费该实例的
  `read_seg`。该缺陷使真实 dry-run 在 `build_context` 阶段即崩溃（与数据无关）；
- 测试：新增 `test_r19_build_context_instantiates_reader_exactly_once`（向 `sys.modules` 注入
  合成 `PlansManager` 复现 property 语义：property 只读一次、返回实例且具备 `read_seg`、
  对实例再调用抛 `TypeError` —— 旧代码必然失败）。

**3. nnU-Net seg 填充哨兵（-1）兼容**
- 根因：`.b2nd` 存储态 seg ∈ {-1,0,1}，**-1 是 nnU-Net 填充/ignore 哨兵**（Dataset605 只定义 0/1；
  官方训练与验证 transform 均执行 `RemoveLabelTansform(-1, 0)`，实现为纯值映射），
  旧 `normalize_seg` 只接受 {0,1} → 真实 loader-smoke 报 `seg 取值必须 ⊆ {0,1}`；
- `data/preprocessed_store.py`：`normalize_seg` 改为**先验证、后转换**
  （dtype 白名单 → shape → 有限性 → 整数性 → 取值 ⊆ {-1,0,1} → 拷贝后 `out = (probe == 1)`）；
  禁止 `astype(uint8)`（-1 会变 255）/`np.clip`/`int()` 截断/原地修改；返回 contiguous uint8 ⊆ {0,1}；
  新增并导出 `SEG_STORAGE_LABELS`、`SEG_SENTINEL_LABEL`；模块契约 docstring 与
  `PreprocessedCase.seg` 注释同步（存储态可含 -1，返回态严格 {0,1}）；
- 测试：新增 `tests/unit/test_seg_sentinel_mapping.py`（**32** 项）——含与 `RemoveLabelTansform(-1,0)`
  的逐体素等价性、`-2/2/0.5/1.5/NaN/±Inf/complex/字符串/object/datetime64` 全部拒绝且不泄漏 `TypeError`、
  合成 `get_case` 路径契约字段、provider batch 无 -1/255、阴性回退语义不变。

**4. `--overfit` 运行时 GPU 峰值显存遥测**
- 新增 `training/gpu_memory_profile.py`：`GpuMemoryProfiler`（start → mark_phase → snapshot → finalize）、
  `atomic_write_json`（tmp + fsync + `os.replace`）、`is_cuda_oom`、`bytes_to_mib/bytes_to_gib`
  （按 `*_bytes[+后缀]` 通用派生 `*_mib/*_gib`）、状态/阶段常量、`MEASUREMENT_SCOPE`
  （明确 peak = **当前 PyTorch 进程** high-water mark，非整卡占用、非 nvidia-smi；`gpu.*` 为整卡状态）；
- `scripts/train/train_m0.py::run_overfit` 接入：**build 之前** `start()`
  （含 `reset_peak_memory_stats`，否则漏掉模型驻留）→ `mark_phase("build_trainer")` → 构建后回填
  模型身份/参数量/AMP/manifest 并 `snapshot("after_build")` → `mark_phase("trainer_run")` →
  `trainer.train()` → `snapshot("after_trainer_run")` → `finalize`；成功 / CUDA OOM / 普通异常都写
  `outputs/diagnostics/<model_id>/<run_name>/diagnostics/gpu_memory_profile.json`；
- CPU 路径：`status=NOT_APPLICABLE`（`reason="device is not CUDA"`），**不触碰任何 CUDA API**，
  CUDA 专属区块为 `null`（不伪造 0）；
- 不改变 `--overfit` 的训练/loss/optimizer/checkpoint/validation/输出目录语义；
  `models/profiling.py` 的静态 `NOT_MEASURED` 未改动（运行时实测是独立产物）。

**5. 遥测复审修补（真实分界 / 初始化保护 / 病例示例）**
- `trainer.train()` 是不可分割的一段（内部先 `_run_epoch(train=True)` 再 `_run_epoch(train=False)`）：
  只保留 `trainer_run` 阶段与 `after_trainer_run` 采样，**删除伪 `after_train`/`after_validation`**；
  报告新增 `peak_scope` 明确「峰值覆盖整个 trainer_run，不存在独立 train/validation 峰值」；
- 异常时按**真实** `trainer.current_phase`（`trainer.py:250/490/689` 已存在，未改 Trainer）细分失败阶段：
  `"train"` → `phase=train`、`"validation"` → `phase=validation`、缺失/未知 → `trainer_run`；
- `profiler.start()` 纳入保护：reset / `mem_get_info` / `get_device_properties` 任一步异常 →
  `ERROR` + `phase=initialize_cuda` + 非 0；`is_available=False` 时**不进入** `build_trainer`；
- `finalize` 的 end 采样单独容错：失败记 `final_sample_error` 并用已有样本落盘，peak 回退
  `end → after_build → start`（出处写 `peak_sample_source`）；CPU 的 `NOT_APPLICABLE` 收尾 `phase=finalize`；
  `empty_cache` 改为**仅当报告成功落盘**后才调用；
- 病例示例修正：`--cases` 提示由阴性病例 `10006_1000006`（fold-0 train，csPCa=NO）改为
  `10005_1000005,10021_1000021`（均 fold-0 train、csPCa=YES；**未改 split**）；
- 测试：`tests/unit/test_gpu_memory_profile.py` 15 → **23**（trainer_run 分界、validation OOM → phase=validation、
  未知 phase 回退、峰值覆盖 build+train+validation、reset / mem_get_info 失败仍落盘、CUDA 不可用跳过 build、
  end 采样失败、CPU `phase=finalize`、病例示例、`empty_cache` 时序）。

**修改文件**：`src/zonal_reliability_fusion/data/zonal_prior.py`、
`src/zonal_reliability_fusion/data/preprocessed_store.py`、
`src/zonal_reliability_fusion/training/gpu_memory_profile.py`（新增）、
`scripts/data/materialize_zonal_prior.py`、`scripts/train/train_m0.py`、
`tests/unit/test_materialization_hardening.py`、`tests/unit/test_seg_sentinel_mapping.py`（新增）、
`tests/unit/test_gpu_memory_profile.py`（新增）。


### G0-R Automated v0.3：刚体 transform 类型修复 + 校准分组重设计（2026-09-18）

> 只改代码/配置/测试/文档；未读取真实医学影像、未使用 GPU、未训练、未推理；draft-0.2 失败运行产物未修改。

**1. 刚体诊断（`estimate_rigid_diagnostic`）**
- 根因（纯合成复现）：SimpleITK 2.5.3 下 `SetInitialTransform(initial, inPlace=False)` 使 `Execute()`
  返回 `CompositeTransform`（内含 1 个 Euler3D），旧代码 `sitk.Euler3DTransform(final)` 抛
  `Transform is not of type Euler3DTransform!` → 真实运行 32/32 全部 `REGISTRATION_DIAGNOSTIC_FAILED`；
- 修复：`inPlace=True` + 新增 `as_euler3d_transform()`（只允许直接 Euler3D 或「单元素 Euler3D 的
  CompositeTransform」解包，否则 fail-closed）；输出新增 `final_transform_type` / `transform_unwrap`；
- `_finalize_registration()` 收紧：`translation_mm`/`rotation_deg` 必须为 3 个有限值，
  `metric_before`/`metric_after` 必须有限，`n_iterations` 非负；任一不满足即失败标记（不伪造零位移）。

**2. 校准与阈值（v0.3 方法性变更）**
- `summarize_calibration(..., threshold_aggregation=...)`：按 `(pair, case_id)` unit 分组，
  零位移参考 = unit 内均值，零位移噪声 = `max(k × pooled_within_unit_sd, max unit 内 LOO 退化, 0)`；
  逐 pair 输出曲线 / 检出率 / 单调性 / 零位移误报 / unit baseline / midpoint / passed；
  记录顺序归一化（输入顺序不影响任何统计量）；`require_each_pair_primary_pass` 生效；
- `derive_thresholds()`：`derivation: median_of_paired_unit_midpoints_by_pair`，
  阈值按 pair 存于 `metrics.<metric>.by_pair.<pair>`，记录 unit baseline / min-detectable 均值 /
  midpoint / 聚合方法 / 方向 / `sensitivity_passed`；与校准记录的 threshold 交叉校验（不一致即报错）；
- `decide_case(..., pair=...)`：必须显式传 pair，未知 pair 抛错（禁止跨 pair 混用与全局回退）；
  一致性指标按**该 pair** 是否通过校准决定 SKIPPED/UNAVAILABLE；
- `decide_overall()`：新增 `pair_calibration_passed` / `pairs_without_calibration` 与对应失败原因。

**3. 输出 schema 与 CLI**
- 新增 `OUTPUT_SCHEMA_VERSION = "g0-r-automated/0.3"`；CLI 新增 `assert_schema_version()`（配置与代码不一致
  启动即失败）、`schema_envelope()`、`load_output_json()`（读取旧 schema 直接拒绝）；
- 所有 JSON 输出带 schema/协议身份；`input_hashes.json` 结构改为 `{schema…, input_sha256: {...}}`；
- `qc_report.md` 增补分 pair 校准/阈值/unit midpoint 与「运行成功 ≠ PASS」声明；`run_metadata.extra`
  增加 schema、分对校准与分对阈值。

**4. 测试（`tests/unit/test_g0_r_automated_qc.py` 25 → 44）**
- 删除「只要求不崩溃」的旧 smoke 断言，改为：结构化合成对**必须 OK**（Euler3D 类型、有限性、
  固定 seed 可复现）；新增 Composite 单元素解包 / 多元素拒绝 / 错误内部类型拒绝 / 最终类型不符 fail-closed；
- 新增 v0.3 校准回归 9 类：pair 分离（旧混合 SD 检出率 0 的对照）、unit 内噪声不含病例间基线差异、
  单 pair 不敏感 → 总体失败、零位移 FPR 上限、不敏感指标不可被改阈值"洗白"、pair 专属阈值与
  pair-aware 判定（未知 pair 显式失败）、输入顺序不变性、fail-closed（缺距离 / 缺零位移 / 非有限 / unit 不足）、
  一致性指标按 pair SKIPPED；
- 新增归档配置字节一致 + 哈希一致、v0.3「不得静默改变」项逐项比对、输出 schema 守卫测试。

**修改文件**：`src/zonal_reliability_fusion/protocols/g0_r_automated_qc.py`、
`scripts/audit/run_picai_alignment_qc_automated.py`、`tests/unit/test_g0_r_automated_qc.py`、
`configs/protocols/g0_r_alignment_qc_automated.yaml`、
`configs/protocols/archive/g0_r_alignment_qc_automated_draft_0_2.yaml`（新增，按字节归档）、
`configs/protocols/archive/README.md`（新增）、`docs/protocols/G0_R_ALIGNMENT_QC_AUTOMATED.md`、
`docs/runbooks/g0_r_alignment_qc.md`、`docs/protocol_changelog.md`、`docs/STATUS.md`、
`docs/experiment_log.md`、`README.md`、`docs/GLOSSARY.md`。


### 版本控制启用（Git 基线；2026-09-18）

- 依据用户指令执行 `git init`（默认分支 `main`；宿主 git 2.25.1，不支持 `git init -b`，改用
  `git symbolic-ref HEAD refs/heads/main`）；**未修改任何 git config**（沿用既有全局身份），**未添加远端、未推送**；
- 基线 commit `23ca6cb`：`chore(vcs): 启用 Git 版本控制并建立基线（用户决定；STATUS B1 关闭）`，
  196 文件 / 57,260 行插入 / 6.8 MB（`.git` 3.1 MB）；
- 跟踪范围：`src/`、`scripts/`、`configs/`、`tests/`、`docs/`、`AGENTS.md`、`README.md`、`.gitignore`、
  `data/metadata/`（4 个冻结元数据文件，≤2.7 MB）、`data/splits/`（5 个 split 文件）、各空目录 `.gitkeep`；
- 排除（`.gitignore` 增补，未删除既有规则）：`data/{raw,interim,processed}/`、`workdir/`、`outputs/` 运行产物
  （含 342 MB 的 `outputs/nnUNet_results/`）、`logs/**`，以及二进制兜底
  `*.pth/*.pt/*.npz/*.npy/*.pkl/*.h5/*.nii/*.nii.gz/*.b2nd/*.mha`；
- 未入库的既有产物仍**原位保留**（含 `outputs/diagnostics/g0_r_automated/20260918_074219/` 失败证据、
  代码快照目录），仅不进入版本库，由 `docs/experiment_log.md` 与哈希追溯。


### G0-R Automated draft-0.4：两项 fail-open 修复（2026-09-18）

> 只改代码/配置/测试/文档；未读取真实医学影像、未使用 GPU、未训练、未推理；未修改任何既有科学阈值与
> draft-0.2 失败产物。

**1. 指标 usable 守卫（`sigma*.usable`）**
- 新增 `metric_scale_prefix()`（正则解析 `sigma<数字>[.<数字>].`，尺度含小数点如 `sigma1.5`）、
  `metric_usable_key()`、`metric_usable()`、`metric_value_with_reason()`；
- 判定顺序改为：**先 usable 守卫（fail-closed）→ 再取有限数值**；缺失 / `false` / 非布尔一律不可用；
- `_unit_metric_detail()` 只接受 `usable===true` 的记录，其它写入 `invalid_conditions`（含原因）；
- `decide_case()`：主指标 unusable → `UNAVAILABLE`（→ `INSUFFICIENT_EVIDENCE`）；
  一致性指标 unusable → `SKIPPED`（写明不得用于支持 ACCEPTABLE/FLAGGED）；每个 check 记录
  `metric_usable` / `metric_unusable_reason`。

**2. 校准 unit 条件网格完整性**
- 新增常量 `CALIBRATION_MIN_COMPLETE_UNITS=3`、`REQUIRE_COMPLETE_CONDITION_GRID=True`，并由配置
  `calibration.min_complete_units_per_pair` / `require_complete_condition_grid` 显式冻结；
- `_unit_metric_detail()` 期望网格改为**配置驱动**（`stability_repeats` 条零位移 + 每个非零
  `displacements_mm` × 全部 `directions`），逐项输出 `missing_conditions` / `invalid_conditions` /
  `n_zero_expected` / `n_zero_observed` / 每个 distance 的 expected/observed/missing directions /
  `complete` / `min_detectable_complete`；
- `_metric_calibration_for_pair()`：只有 complete unit 参与 pooled SD/曲线/检出率/midpoint；
  新增失败原因 `incomplete_unit_condition_grid` / `insufficient_complete_units` /
  `min_detectable_mm_uncovered_units`；输出 `n_units_total` / `n_units_complete` /
  `n_units_at_min_detectable` / `n_units_participating` / `incomplete_unit_ids` / `excluded_unit_ids`；
- `summarize_calibration()` / `derive_thresholds()` 的计数与完整性字段同步（顶层与每 pair/每 metric）。

**3. 版本与 CLI**
- `OUTPUT_SCHEMA_VERSION = "g0-r-automated/0.4"`；配置 `protocol.version: draft-0.4`、
  `decision.rule_version: "0.4"`；draft-0.3 配置按字节归档（SHA256 `beda4bbf1065c9c8493fe41a640f5e4a257076b256438b8dd493cbedfa678198`）；
- CLI 报告/摘要/`run_metadata` 输出完整性与 usable 审计（`calibration_units`、`metric_usable_keys`）；
- `--dry-run` 实测通过（协议 draft-0.4、schema 0.4、hash `13e7be50752db794…`），**未创建输出目录**。

**4. 测试（`tests/unit/test_g0_r_automated_qc.py` 44 → 52）**
- 合成助手升级：指标名必须带 `sigma<scale>.` 前缀并携带 `sigma*.usable`；新增注入能力
  （`drop_conditions` / `nan_conditions` / `unusable_units`）与 `_three_unit_spec()`（3 unit × 2 pair）；
- 新增 11 类回归：NaN@min-detectable、缺方向@min-detectable、`usable=false` 主指标、真实病例
  unusable → `INSUFFICIENT_EVIDENCE`、usable 字段缺失/非布尔 fail-closed、complete unit 不足（1/2 个）、
  3 个 complete unit 正常通过、计数与 missing/invalid 审计字段、pair 不得互相补足 unit、
  输入顺序不变性、v0.2/v0.3 输出被 v0.4 loader 拒绝。

**修改文件**：`src/zonal_reliability_fusion/protocols/g0_r_automated_qc.py`、
`scripts/audit/run_picai_alignment_qc_automated.py`、`tests/unit/test_g0_r_automated_qc.py`、
`configs/protocols/g0_r_alignment_qc_automated.yaml`、
`configs/protocols/archive/g0_r_alignment_qc_automated_draft_0_3.yaml`（新增，按字节归档）、
`configs/protocols/archive/README.md`、`docs/protocols/G0_R_ALIGNMENT_QC_AUTOMATED.md`、
`docs/runbooks/g0_r_alignment_qc.md`、`docs/protocol_changelog.md`、`docs/STATUS.md`、
`docs/experiment_log.md`、`README.md`、`docs/GLOSSARY.md`、
`outputs/diagnostics/code_snapshots/g0_r_v02_before_fix_20260918_081501/RESTORE.md`（校验命令修正）。
