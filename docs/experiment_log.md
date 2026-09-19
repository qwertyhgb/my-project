# 实验日志

> 只记录**实际运行过**的命令与结果；计划中的内容不写成本已完成。

## 2026-09-14

### G0 数据审计（首轮）

- manifest 构建、几何审计、分区审计、划分生成：见 `docs/PICAI_Model_Readiness_Audit.md` §3–§7 与
  `logs/manifest_build.log`、`logs/geometry_audit_p0a.log`、`logs/zones_audit.log`。

### P0A：方向码修正与审计重建

| # | 命令 | 耗时 | 结果 |
|:--|:--|--:|:--|
| 1 | `pytest tests/unit/test_picai.py`（LPS 断言修正后） | 秒级 | 19 passed |
| 2 | `python scripts/data/build_picai_manifest.py` | 201.5s | 0 错误，`all_pass=True`；仅 7 个 `*_orient` 列 RAS→LPS |
| 3 | `python scripts/audit/audit_picai_geometry.py` | 500.5s | 10500 行，0 错误；仅 `orientation` 列变化 |
| 4 | 差分验证脚本（archive vs 新版） | 秒级 | `p0a_orientation_fix_verification.json`、`p0a_geometry_orientation_fix_verification.json` |
| 5 | geometry 比对（archive vs 新） | 秒级 | **仅 `orientation` 列变化（10500 行 RAS→LPS）**；`grid_status` / `severity` 分布与异常行数与旧完全一致 |

输出：`data/metadata/picai_manifest.csv`、`picai_manifest_validation.json`、
`outputs/diagnostics/picai_model_readiness/geometry_audit.csv`、`geometry_anomalies.csv`、
`p0a_orientation_fix_verification.json`、`p0a_geometry_orientation_fix_verification.json`。
旧版产物归档：`outputs/diagnostics/picai_model_readiness/archive_pre_p0a/`。
日志：`logs/manifest_build_p0a.log`、`logs/geometry_audit_p0a.log`。

### P0B：数据物化

| # | 命令 | 耗时 | 结果 |
|:--|:--|--:|:--|
| 1 | `python scripts/data/materialize_picai.py --dry-run` | 秒级 | 1500 例估算：未压缩≈128.9 GB、压缩后≈70.9 GB；剩余 26.8 TB；headroom_ok |
| 2 | smoke（10 例；前两次因脚本缺陷失败：`NameError resample_array`、`NameError sitk`，修复后通过） | 27.1s | 10/10 ok |
| 3 | 补测 WG 禁用例 `--cases 11050_1001070 --resume` | 5.9s | ok；`wg_status=excluded_known_faulty_bosma22b`，无 wg 文件 |
| 4 | smoke 报告重建（11 例 `--overwrite --resume`） | 31.9s | 11/11 ok，validation_ok 全通过 |
| 5 | 5 例 geometry_suspect 物化后 QC（`scripts/audit/generate_picai_materialized_qc.py`） | ~10s | 5 张 QC 图；**腺体内 ADC/HBV 覆盖度 100%（0 缺失）**；全部保留并标注 |
| 6 | 追加 QC：`10459_1000467`（阳性+非精确网格）、`10000_1000000`（阴性） | ~10s | 病灶轮廓可见、分区与对齐正常 |
| 7 | 全量物化 `python scripts/data/materialize_picai.py --resume`（后台，tqdm 进度） | **4799.4s（≈80 min）** | **1500/1500 ok，0 失败，0 跳过；27.3 GB**；`materialization_summary.json`（日志 `logs/materialize_full_p0b.log`） |
| 8 | 中途审计 `python scripts/data/audit_materialization_report.py --allow-incomplete`（306 例时） | 秒级 | all_pass=True：0 失败；73 阳性 0 丢失；233 阴性 0 异常；299 例 exact-grid 前景体素与 manifest 一致；WG 禁用例正确 |

全量完工补记（2026-09-14）：

| # | 命令 | 耗时 | 结果 |
|:--|:--|--:|:--|
| 9 | `python scripts/data/audit_materialization_report.py`（全量 1500 例） | 秒级 | **all_pass=True**（11 项检查全通过）：覆盖 1500/1500；0 失败行；`validation_ok` 全 True；425 阳性 0 丢失；1075 阴性 0 异常；1491 例 exact-grid 前景体素与 manifest 完全一致；WG 禁用例正确 |
| 10 | `python scripts/data/materialize_picai.py --verify` | 247s（tqdm 1500/1500） | `{'complete': 1500}`、`all_complete=True`、`problem_cases=[]`；`materialization_validation.json/csv`（日志 `logs/materialize_verify_p0b.log`） |
| 11 | `source scripts/env_nnunet.sh` 后执行 `nnUNetv2_plan_and_preprocess -d 605 -c 3d_fullres`（研究者执行） | 见日志（09:19 结束） | **1500/1500，0 Error / 0 Traceback / 0 Warning**（研究者核验）；每例 1 影像 `.b2nd` + 1 `_seg.b2nd` + 1 `.pkl`（各 1500，无空文件，三类文件病例 ID 一一对应）；`dataset_fingerprint.json`、`nnUNetPlans.json` 齐备；≈45 GB（`logs/nnunet_plan_preprocess_p0b.log`） |
| 12 | `python scripts/data/create_nnunet_splits.py`（研究者执行；先 `--dry-run` 后正式） | 0.2s | `splits_final.json`（fold 0，**train 1277 / val 223**）；**14/14 校验 PASS**；preprocessed 与 raw 副本 sha256 一致；`picai_nnunet_splits_validation.json`、`picai_nnunet_split_cases.csv`（1500 行） |

### Split 更新

- 命令：`python scripts/data/create_picai_splits.py --compare-legacy data/splits/picai_cross_center_splits.json`
- 结果：新文件 `data/splits/picai_train_val_split.json`（train 1256 患者/1277 study/362 阳性；val 220/223/63）；
  legacy 一致性校验 **pass=True**（train/validation 的 patient 与 study 列表完全一致）；
  无患者泄漏；23 名多 study 患者均单集合；旧文件 md5 未变。
- 校验 JSON：`data/splits/picai_train_val_split_vs_legacy_check.json`。

### nnU-Net raw 准备（11 例子集预演）

- 命令：`python scripts/data/prepare_picai_nnunet_raw.py --cases <11 例> --no-progress`
- 结果：创建 `Dataset605_PICAI` 子集（11 例；dataset.json：0000=T2W/0001=ADC/0002=HBV，
  labels background/lesion；病例映射 CSV）。
- 自动 Reader = **SimpleITKIO**；`verify_dataset_integrity(..., num_processes=4)` 无错误（子集）。
- **全量链接**（1500 例，imagesTr 4500 + labelsTr 1500）与**全量完整性检查 PASS（0 错误）**已完成；
  数据集位于项目内 `workdir/nnUNet_raw/Dataset605_PICAI`（初版曾误放项目外
  `/opt/data/private/lm/projects/nnunet/`，已按项目惯例迁移，项目外副本已清理完毕，`/opt/data/private/lm/projects/` 已整体移除）。
- 运行环境：`source scripts/env_nnunet.sh`（固定 `third_party/nnUNet` v2.6.2 源码树 +
  项目内 raw/preprocessed/results 路径；实测 `nnunetv2` 解析到项目固定树）。
- plan/preprocess（3d_fullres）由研究者执行，**已完成**（1500/1500，0 Error / 0 Warning，≈45 GB；见上表 #11）。

### P2-A：M0 自研基线（合成 CPU 单元测试；CODE READY）

> 说明：以下命令**不读取任何真实病例数据、不启用 CUDA、不训练模型**；仅使用合成张量。

| # | 命令 | 耗时 | 结果 |
|:--|:--|--:|:--|
| 1 | `CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider tests/unit/test_plans.py tests/unit/test_m0_model.py tests/unit/test_losses.py tests/unit/test_patch_sampler.py tests/unit/test_sliding_window.py tests/unit/test_checkpointing.py tests/unit/test_progress.py tests/unit/test_batch_provider.py tests/unit/test_trainer_synthetic.py -q` | 4.9s | **94 passed**（九个 P2 测试文件自身）；同批并跑既有 `test_picai.py`（19 项）时为 113 passed；0 failed |
| 2 | 静态检查：`ast.parse` 两个新脚本 + `--help`（argparse 装配，不读数据/不建模型） | 秒级 | 通过 |
| 3 | 质量扫描：`grep` 硬编码 GPU 编号 / `torch.cuda` 使用点 / `models/` 内 softmax / `git -C third_party/nnUNet status` | 秒级 | 无硬编码 GPU 编号；`torch.cuda` 仅在 device=cuda 或 AMP 路径；模型内部无 softmax；third_party 未修改 |

**P2-A 复审修复（同日，第二轮）**：评审发现 2 个会导致真实训练失败的功能问题 + 若干对齐问题，已修复：

| # | 命令 | 耗时 | 结果 |
|:--|:--|--:|:--|
| 4 | P2 九个测试文件（修复后，新增 8 项回归测试） | 5.0s | **102 passed** |
| 5 | `pytest tests/unit`（全量，含 `test_picai.py` 19 项） | 6.2s | **121 passed**，0 failed |
| 6 | 静态复核：`model.to(device)`、`class_locations` 两格式、滑窗/smoke 进度条、loss 数值口径 | 秒级 | 见 `docs/P2_M0_Implementation.md`「复审问题与修复」 |
| 7 | 损失口径与官方逐值比对：只读 import `nnunetv2.training.loss.dice.SoftDiceLoss`，在合成张量上比较（batch_dice=False/True × 含前景/全阴性） | 秒级 | **自研 Dice 项与官方一致**：最大绝对差 **7.45e-09**（float32 精度内）；`total == CE + Dice`（差 ≤3e-08） |

修复内容（均以**合成数据**验证，未读真实病例、未用 GPU）：

1. **CUDA 训练阻塞**：trainer 现显式 `model.to(device)`，并在 `run_manifest.json` 记录
   `model_parameter_device`；新增回归测试。
2. **`class_locations` 官方 (n,4) 格式**：新增 `PatchSampler.normalize_class_locations`（兼容 (n,3)/(n,4)，
   取后 3 列即空间坐标），并新增 (n,4) 复现测试；前景 patch 改为官方"以体素为中心"取法。
3. **进度条**：滑窗（含 TTA 每个翻转）、loader smoke 的病例循环与 batch 循环均接入 `make_progress`；
   新增"progress=True 有输出 / False 无输出"的回归测试。
4. **前景过采样语义**：改为官方的"每批固定 `bs - round(bs*(1-oversample))` 个样本"（bs=2、0.33 → 每批 1/2），
   不再是逐样本 33% 概率；验证集仍固定 0。
5. **num_workers / pin_memory 接入**：数据加载走 `torch.utils.data.DataLoader`（配置项真实生效）。
6. **采样可复现性**：每个样本种子由 `(seed, epoch, index)` 稳定混合，采样与 worker 数无关；
   trainer 每轮调用 `set_epoch(epoch)`，resume 后序列可复现（不再依赖隐藏 RNG 状态）。
7. **损失数值口径**：`SoftDiceLoss` 改为返回 `-Dice`（与官方 `SoftDiceLoss.forward` 一致，去掉此前的 +1.0 偏移），
   并新增"CE − Dice"等值性测试；`components()` 同时给出 `dice_loss`（≤0）与 `dice_score`（正 Dice）。
8. **增强差异冻结**：配置新增 `augmentation.pending_parity`（rotation/scaling/低分辨率模拟/noise/blur/gamma），
   校验时以 warning 显式提示，防止被误称"已完全对齐"。
9. **文档一致性**：`research_plan.md` §6/§14 的 `G0-P = Conditional Pass` 已修正为 **PASS**（与审计报告一致）。

- 期间修复的问题（均为测试暴露，已复测）：滑窗切片未去掉 batch 维导致 6 维输入（已限定单例推理 `B=1`）；
  trainer 的重复 keyword-only 标记（语法错误）；`center_crop_or_pad_3d` 对齐保护改为"差异 >2 体素即报错"；
  PolyLR 恢复状态不再改写 optimizer lr（LR 由 epoch 索引决定，保证续训连续）。

### P2-A3：全体积验证 / Dice 选 best / early stopping（合成 CPU 测试；未运行真实训练或真实验证）

> 说明：本节所有命令**只使用假内存 store、合成 3D 张量与 CPU**；不读取 `.b2nd`/NIfTI、不初始化 CUDA、
> 不启动训练或真实推理。

| # | 命令 | 耗时 | 结果 |
|:--|:--|--:|:--|
| 1 | `CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider tests/unit -q` | 6.7s | **159 passed**（0 failed；含既有 `test_picai.py` 19 项） |
| 2 | 同上，仅 P2 测试文件（含新增 `test_full_volume_validator.py`、`test_train_m0_script.py`） | 6.5s | **140 passed**（评审基线 102 项全部继续通过，新增 38 项） |
| 3 | 静态检查：`ast.parse` 两个训练脚本 + `train_m0.py --help`（不读数据、不建模型） | 秒级 | 通过 |
| 4 | 配置自检：解析 `configs/experiments/m0_picai_3d_fullres.yaml` | 秒级 | `max_epochs=200`、mode=full_volume、metric=`val_positive_casewise_dice_mean`、expected=223/63/160、early stopping 50/30/1e-4 |
| 5 | 依赖边界核验：`grep` 官方训练循环 / `nnunetv2` import 点 / `git -C third_party/nnUNet status` | 秒级 | 未使用 `nnUNetTrainer`/`nnUNetDataLoader`；唯一 `nnunetv2` import 仍在 `PreprocessedStore`（惰性）；third_party 未修改 |

- 测试覆盖（对应验收清单）：全部 validation ID 恰好访问一次、无随机抽样/遗漏/重复、阳性完美 Dice=1、
  阳性空预测 Dice=0、阴性空—空不进入主指标、阴性 FP 统计、micro Dice、最大化 Dice 保存 best、
  改善小于 `min_delta` 不覆盖 best、early stopping 不早于 `min_epochs`、patience 触发正常停止、
  resume 恢复 early-stop counter 与 best Dice、DS 与 train/eval 状态恢复、progress 开关、
  非有限预测立即报错、expected 计数不一致失败、正式模式不创建随机 validation patch provider、
  原有 P2 测试全部通过。
- 修复：`load_experiment_config` 的 `return` 使其后协议校验不可达（已修）；验证模式命名统一
  （`full_volume` / `diagnostic_patch`，保留 `formal_full_volume` 别名）。
- **未运行**：真实数据 loader smoke、small-overfit、正式训练、全体积真实验证、任何 GPU 计算。真实
  `.b2nd` 上的速度/显存、223/63/160 的真实标签分布与 200 epoch 预算是否合适，均待研究者运行后确认。

**P2-A3 复审修复（续训安全性；仍为合成 CPU 测试）**

| # | 命令 | 耗时 | 结果 |
|:--|:--|--:|:--|
| 6 | `pytest tests/unit -q`（修复后全量） | 8.6s | **163 passed / 0 failed**（含既有 `test_picai.py` 19 项） |
| 7 | 同上，仅 P2 测试文件 | 8.7s | **144 passed**（复审基线 140 项全部继续通过，新增 4 项） |

修复内容（对应评审四点，均有回归测试）：

1. **中断 checkpoint 的 epoch 语义**：checkpoint 新增 `epoch_completed` / `completed_epochs` /
   `iterations_completed`；`interrupt` 只写自己且强制 `epoch_completed=False`；`resume()` 对中断 checkpoint
   **默认拒绝**（提示改用 `checkpoint_last.pth`），显式 `allow_mid_epoch=True` 时才重跑被打断的 epoch。
2. **续训协议核对**：`resume()` 核对 `checkpoint_metric / maximize_metric / early_stopping_enabled /
   min_epochs / patience / min_delta / validation_protocol`，不一致默认拒绝（`allow_protocol_change=True` 可显式放行）。
3. **RNG 不再被覆盖**：`resume()` 恢复 RNG 后置标记，`train()` 不再调用 `set_seed`。
4. **冻结指标与方向**：`full_volume` 模式硬冻结 `val_positive_casewise_dice_mean` + `maximize=True`
   （配置层与 validator 共用同一实现），trainer 侧再校验一次。
5. 文档修正：`P2_M0_Implementation.md` 的 `best(val_loss)` 改为按 Dice 最大化（§8），新增 §17 续训安全性说明。

**P2-A3 复审第二轮修复（续训边界；仍为合成 CPU 测试）**

| # | 命令 | 耗时 | 结果 |
|:--|:--|--:|:--|
| 8 | `pytest tests/unit -q`（修复后全量） | 10.2s | **164 passed / 0 failed** |
| 9 | 同上，仅 P2 测试文件 | — | **145 passed**（第二轮基线 144 项继续通过，新增 1 项） |

1. **`completed_epochs` 改为绝对计数**：中断时使用 `_epoch_offset + len(self._history)`（`_epoch_offset` 在
   `resume()` 后 = `start_epoch`），修复"resume 后再中断时被记为 0/1"的问题；正常保存路径同样写入
   `completed_epochs = epoch + 1`。回归测试 `test_interrupt_after_resume_records_absolute_completed_epochs`
   （进程 A 在 epoch 4 中断 → `completed_epochs=4`；从 last 续训后在 epoch 5 再次中断 → `completed_epochs=5`，
   修复前会错记为 1；再以 `allow_mid_epoch=True` 续训 → `start_epoch=5`）。
2. **协议放行后不再继承旧选模状态**：`_verify_resume_protocol` 返回"协议是否变化"，变化且被显式允许时
   重置 `best_metric / best_epoch / bad_epochs` 并打 WARN；`test_resume_rejects_protocol_change` 增加
   "重置后从续训点重新选模（`best_epoch=2`）"断言。
3. **元数据一致性**：`epoch_completed=True` 时校验 `completed_epochs == epoch + 1`，不一致直接拒绝续训。
4. 文档/CLI 同步：`P2_M0_Implementation.md` §17.1/§17.2、`train_m0.py --help`。

**P2-A3 复审第三轮修复（checkpoint 元数据 / 配置冻结 / 日志对齐；仍为合成 CPU 测试）**

| # | 命令 | 耗时 | 结果 |
|:--|:--|--:|:--|
| 10 | `pytest tests/unit -q`（修复后全量） | 13.1s | **167 passed / 0 failed** |
| 11 | 端到端元数据抽查（合成 trainer + 临时目录，仅 CPU） | 0.3s | `last/best/periodic` 全部 `epoch_completed=True, completed_epochs=epoch+1, iterations_completed=2`（修复前 best/periodic 为 None）；`CSV 最后 epoch=2` |

1. **best/周期 checkpoint 缺元数据**：`build_checkpoint_payload` 统一补全（完整 epoch 未给定时自动
   `epoch + 1`），`CheckpointManager.save` 把同一份 `epoch_completed/completed_epochs/iterations_completed`
   传给 last / best / 周期三类文件；新增 `test_manager_writes_epoch_metadata_to_all_checkpoints`。
2. **续训只核对协议、未核对完整配置**：`resume()` 现按冻结前缀逐项比对 resolved 配置快照
   （`experiment./data./model./loss./optimizer./scheduler./training./inference./validation./early_stopping./
   plan./train_pipeline./val_pipeline./provenance.plans_sha256|splits_sha256`），不一致默认拒绝；
   `training.max_epochs` 为唯一允许显式延长的训练项，`num_workers/pin_memory/checkpoint_every/progress`
   仅记录不阻塞；回归 `test_resume_rejects_run_config_change_and_allows_max_epochs_extension`
   （optimizer.lr 变化拒绝、plan/split 哈希变化拒绝、非冻结项放行、max_epochs 延长放行）。
3. **resume 与日志未对齐**：新增 `last_logged_epoch()` 与对齐校验（续训起点必须 = CSV 最后 epoch + 1），
   从旧周期 checkpoint 恢复到同一 run 会被拒绝；回归 `test_resume_rejects_stale_checkpoint_log_mismatch`
   （`checkpoint_epoch_0000.pth` 拒绝、`checkpoint_last.pth` 通过）。
- 未运行：任何真实数据 loader smoke、small-overfit、正式训练、推理（均由研究者执行；命令见
  `docs/P2_M0_Implementation.md` 与最终汇报）。

## 2026-09-15

### P1：N0 官方基线训练——默认 Dice+CE 预实验（研究者执行；后已排除，见下方「P1：N0 损失变更」）

- 命令：`nnUNetv2_train 605 3d_fullres 0`（conda `lm`；先 `source scripts/env_nnunet.sh`；tmux 会话 `lm`）。
- 启动时间：2026-09-15 01:17:28；日志：
  `outputs/nnUNet_results/Dataset605_PICAI/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/training_log_2026_9_15_01_17_28.txt`。
- 划分来源（日志确认）：`workdir/nnUNet_preprocessed/Dataset605_PICAI/splits_final.json`，1 个 split，**1277 train / 223 validation**。
- 实际协议（`debug.json`）：`num_epochs=1000`（官方默认）、`num_iterations_per_epoch=250`、
  `num_val_iterations_per_epoch=50`、`save_every=50`、`batch_size=2`、`patch_size=[16,320,320]`、
  `oversample_foreground_percent=0.33`、`enable_deep_supervision=True`、`device=cuda:0`、`torch.compile` 启用。
- 运行观察（截至 epoch 8，01:27）：GPU RTX 3090 利用率 96%、显存 ≈5.9 GB；epoch 耗时
  150.9 s（首轮 compile 预热）→ 84.2 s → 稳定 ≈53 s；`train_loss` +0.001（epoch 0）→ −0.067 ~ −0.072；
  `val_loss` −0.065 ~ −0.076；lr 按 PolyLR 从 0.01 正常衰减。
- **`Pseudo dice` 的准确含义（2026-09-15 更正）**：`Pseudo dice` 是每 epoch 50 个 validation batch 的在线
  patch 级指标，而不是全体积 Dice；validation dataloader **同样启用 0.33 前景过采样**（batch=2 时每 batch
  1 个强制前景 patch，见 `third_party/nnUNet/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py` 的
  `get_dataloaders`）。因此其长期接近 0 表示**模型的硬预测前景重叠极低**，不能简单归因于 validation patch
  未命中病灶，也不能作为正式性能结论；它同样**不作为训练质量判据，也不作为 checkpoint 选择依据**。
  （更正前该条目曾写为「验证只用随机 patch、极少含病灶」，与实际实现不符。）
- 已知事项与处置（2026-09-15 更正）：`checkpoint_best.pth` 首次在 epoch 0 **无条件**保存，此后仅当 EMA
  pseudo Dice 改善时才被覆盖——本项目实测 `Pseudo dice` 自 epoch 22 起出现非零，best 随后于 01:44（≈epoch 25）
  被更新，因此「best 停留在 epoch 0」不成立。由于该指标只是 online patch 级、且不是最终评测口径，
  **最终推理与评测固定使用 `checkpoint_final.pth`，不使用 `--val_best`**（见 `research_plan.md` §6）。
- 待决事项：N0 训练预算（1000）与自研 M0–M4（`max_epochs=200`）不一致，需在正式 N0 vs M0 比较前
  显式冻结或对齐（官方变体可选 250 epoch，无 200）。
- 结果：该默认 Dice+CE run 于 01:58 在 **Epoch 43** 开始后中断且**未完成**（无 `Training done`、无
  `checkpoint_final.pth`、无 `validation/`）；**已排除**，不再作为正式 N0 baseline。详见下方
  「### P1：N0 损失变更」。本条目仅记录该 run 实际发生过的内容，不代表正式 N0 结果。

### P1 排障：N0「Pseudo dice 长期为 0」只读诊断工具（合成测试 + 元数据冒烟）

| # | 命令 | 耗时 | 结果 |
|:--|:--|--:|:--|
| 1 | `CUDA_VISIBLE_DEVICES="" pytest tests/unit -q`（新增诊断工具测试后全量） | 14.0 s | **203 passed / 0 failed**（新增 `test_n0_zero_dice_diagnosis.py` 36 项） |
| 2 | `python scripts/train/diagnose_n0_zero_dice.py --skip-case-audit --no-progress --output outputs/diagnostics/n0_zero_dice/smoke_metadata_only` | 0.14 s | 静态审计 **17/17 通过**；日志审计（**该时刻的运行中途快照**）：已写入 34 epoch、pseudo dice 非零 4 次、**首次非零 epoch 22**、train_loss 首 +0.001 / 末 −0.0687；判定 `NEEDS_BATCH_PROBE` |

- 说明：命令 2 **只读 JSON/CSV 元数据与 `training_log_*.txt`**，未读取任何 `.b2nd`/`.pkl`/影像体素，未使用 GPU，
  未触碰正在运行的训练进程（PID 11498）。
- **快照说明（避免与最终统计冲突）**：上表命令 2 的「34 epoch / 非零 4 次」是该命令运行当刻（01:51）的诊断快照，
  **不是**该 run 的最终统计。该 run 的最终日志记录为 **43 条 epoch 指标（38 次为 0、5 次非零）**，随后在
  **Epoch 43 刚开始时中断**（见下方「### P1：N0 损失变更」表第 1 行）。
- 环境变更：`lm` 缺少 pytest，已按 AGENTS.md §4.2 安装 pytest 9.1.1（见 `docs/Development_Log.md`）。
- 真实数据诊断（病例审计、采样/增强探针、冻结 checkpoint 全体积评测、受控 overfit）命令已交付用户执行，
  结果待研究者返回后补记。

### P2：M0 损失迁移到 Focal+CE（合成测试；未运行真实数据）

| # | 命令 | 耗时 | 结果 |
|:--|:--|--:|:--|
| 1 | `CUDA_VISIBLE_DEVICES="" PYTHONPATH="$PWD/third_party/nnUNet:$PWD/src" pytest tests/unit/test_loss_alignment_n0.py -q` | 5.5 s | **23 passed**（M0 损失与 N0 适配层逐值一致；DS 权重与官方 `DeepSupervisionWrapper` 一致；配置显式性含 10 组拒绝用例） |
| 2 | 同上，`tests/unit -q`（全量） | 16.6 s | **233 passed / 0 failed** |

- 说明：两条命令均使用**合成张量**、强制 CPU，未读取任何真实医学影像 / `.b2nd` / `.pkl`，未使用 GPU，未启动训练；
  实现细节见 `docs/Development_Log.md` 的「P2：M0 迁移到 Focal+CE」条目。
- **未运行**：M0 的真实数据 setup 检查、loader smoke、small-overfit、正式训练与全体积验证
  （**G2 仍为 NOT YET EVALUATED**）；`outputs/checkpoints/m0/`、`outputs/metrics/m0/` 目前仍为空。

### P1：N0 损失变更（研究者执行；预实验 + 首次崩溃）

| # | 命令 | 耗时 | 结果 |
|:--|:--|--:|:--|
| 1 | `nnUNetv2_train 605 3d_fullres 0`（默认 `nnUNetTrainer`，Dice+CE；01:17:28 启动） | 至 01:58 日志中断 | **预实验（已排除）**。日志在 `Epoch 43` 开始后中断：无 `Training done`、无 `checkpoint_final.pth`、无 `validation/`。43 个 epoch 中 **38 次 `Pseudo dice = 0`**，5 次非零（1e-4~4e-4，首次非零在 epoch 22）；`train_loss` 由 epoch 0 的 +0.001 降到 −0.07 区间后基本持平。仅存 `checkpoint_best.pth`（epoch 0 保存，epoch 25 附近被 EMA 更新）。**产物、日志与 checkpoint 必须保留，不得删除或覆盖；不得写成正式失败实验** |
| 2 | `CUDA_VISIBLE_DEVICES=0 python scripts/train/train_n0_picai_flce.py 605 3d_fullres 0`（首次 `nnUNetTrainerPICAI_FLCE`，Focal+CE；02:17:05 启动） | 崩溃于 epoch 0 | 日志：`outputs/nnUNet_results/Dataset605_PICAI/nnUNetTrainerPICAI_FLCE__nnUNetPlans__3d_fullres/fold_0/training_log_2026_9_15_02_17_05.txt`。日志在 `Epoch 0 / Current learning rate: 0.01` 之后中断，**未产生任何 train_loss 或 checkpoint**。后台增强 worker 触发原生 `free(): corrupted unsorted chunks`（`GaussianBlurTransform` 的 `benchmark=True` → `fft_conv_pytorch`），与 Focal+CE 数值无关。**日志原样保留** |

- **未运行**：NoFFT 修复版（`nnUNetTrainerPICAI_FLCE_NoFFT`）的真实训练——其输出目录
  `nnUNetTrainerPICAI_FLCE_NoFFT__nnUNetPlans__3d_fullres/` 尚未创建，等待研究者执行唯一训练命令
  （见 `docs/N0_PICAI_Official_Baseline.md`）。
- **未产生**：以上两个 run 均**没有**全体积 Dice / AP / AUROC / validation 结果，不得补写；`Pseudo dice` 是
  online patch 指标，不作为正式结论。
- 合成测试（本日，不读真实数据、不使用 GPU）：`tests/unit/test_nnunet_picai_flce.py` 见 `docs/Development_Log.md`。

### P2：v2.2 低频验证迁移复核（代理执行；合成 CPU，未读真实医学数据）

| # | 命令 | 耗时 | 结果 |
|:--|:--|--:|:--|
| 1 | `PYTHONPATH="$PWD/third_party/nnUNet:$PWD/src" CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider tests/unit -q` | 32.0s | **242 passed / 0 failed**（含低频验证系列：调度 / 最终 epoch 必验证 / 非验证 epoch 不更新 best 与 patience / patience 按 validation event 计数 / min_epochs 按 training epoch / resume 恢复 event 状态 / 调度变化默认拒绝与显式放行重置 / 非法 N） |
| 2 | 三个协议 YAML 的 `yaml.safe_load` 解析校验（`configs/protocols/*.yaml`） | 秒级 | 三个文件全部解析通过，`status: DRAFT`；未确定字段均为 `null`（未伪造阈值/功效数值） |
| 3 | 只读数据事实核对（`data/metadata/picai_manifest.csv` 列统计） | 秒级 | 1500 study；`geometry_status`：same_physical_space_different_grid 840 / partial_physical_overlap 655 / geometry_suspect 5；中心 RUMC 800 / PCNN 350 / ZGT 350；csPCa YES 425 / NO 1075（用于 G0-R 抽样分层与协议文档引用） |

- 说明：以上均为**合成 CPU 测试或秒级只读检查**；未读取 `.b2nd` / `.pkl` / 真实影像体素，未使用 GPU，
  未启动训练、推理、评测或全体积验证；`third_party/` 未修改。
- **未运行（仍待研究者执行）**：M0 的 `validate_m0_setup.py` → `--loader-smoke` → `--overfit`、
  任何真实训练与全体积验证、G0-R 对齐 QC 的真实数据部分、`scripts/evaluate/`（尚未实现）。
- v2.2 迁移的工程变更记录见 `docs/Development_Log.md`「研究协议 v2.2 与低频验证代码迁移」。

### G0 协议工具实现（代理执行；合成 CPU 测试 + 秒级只读检查，未读真实医学数据）

| # | 命令 | 耗时 | 结果 |
|:--|:--|--:|:--|
| 1 | `PYTHONPATH="$PWD/third_party/nnUNet:$PWD/src" CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider tests/unit/test_g0_protocol_common.py tests/unit/test_g0_r_sampling.py tests/unit/test_g0_e_inventory.py tests/unit/test_g0_sap_freeze.py tests/unit/test_g0_scripts.py -q` | 2.2s | **55 passed**（G0-R 抽样/盲审/schema；G0-E 清点/UNKNOWN；G0-SAP 校验/冻结就绪/模板；三个脚本入口装配） |
| 2 | 同上，全量 `tests/unit`（含全部既有测试） | 15.9s | **297 passed / 0 failed**（既有 242 项无回归） |
| 3 | `python scripts/audit/audit_picai_alignment_qc.py --help`、`python scripts/audit/audit_g0e_candidates.py --help`、`python scripts/evaluate/prepare_g0_sap_freeze.py --help` | 秒级 | 全部 OK（argparse 装配，不读数据、不建模型） |
| 4 | `python scripts/evaluate/prepare_g0_sap_freeze.py --check-freeze`（只读 `configs/protocols/g0_sap.yaml`） | 秒级 | `status=DRAFT`、`freeze_readiness: NOT READY（阻塞项 2）`：24 个字段待填 + `power_memo.status=insufficient_data`（未伪造任何数值） |

- 说明：以上均为**合成 CPU 测试或秒级只读配置检查**；未读取 `.b2nd`/影像体素、未使用 GPU、未启动训练/推理/评测；
  三个 G0 工具的**真实数据运行未执行**（命令见下方"研究者待执行"），G0-R/G0-E/G0-SAP 均保持 PENDING/DRAFT。
- **未运行（仍待研究者执行，属 G0-R/G0-E）**：
  - `python scripts/audit/audit_picai_alignment_qc.py`（抽样 + 盲审清单；只读 manifest/几何审计元数据）；
  - `python scripts/audit/audit_g0e_candidates.py`（候选文件清单级清点 + 待补证据清单）。

### G0-R 抽样工具执行（研究者执行；只读元数据）

- 产物：`outputs/diagnostics/g0_r/20260915_093819/`（`sampled_cases.csv`、`sampling_manifest.json`、
  `blind_review_sheet.csv`、`alignment_metrics_schema.json`、`run_metadata.json`）。
- 结果（**读自 `sampling_manifest.json`，非补写**）：`total_cases=16`、`n_cases_source=minimum_required`；
  分层命中 `geometry_suspect=5 / center=3 / lesion_status=2 / partial_overlap=2 / extreme_fov=2 / spacing_span=4`；
  `warnings=[]`、`decision=null`、`status=PENDING`；config sha256 `ab168586…`、manifest sha256 `0226cd22…`。
- 备注：该抽样清单**不可替换、增删或重抽样**；G0-R 仍为 DRAFT/Pending（抽样成功不等于通过）。

### G0-R 图像 QC 执行工具实现（代理执行；合成测试，未运行真实病例）

| # | 命令 | 耗时 | 结果 |
|:--|:--|--:|:--|
| 1 | `PYTHONPATH="$PWD/third_party/nnUNet:$PWD/src" CUDA_VISIBLE_DEVICES="" python -m pytest -p no:cacheprovider tests/unit/test_g0_r_qc.py -q` | 8.2s | **16 passed**（物理坐标重采样与同网格短路、checkerboard/edge-overlay、切片选择、manifest 子集限制与非法病例拒绝、landmark mm 位移与 FOV 校验、无边界来源安全跳过、Hausdorff/平均表面距离、渲染端到端（PNG×8/索引/模板/几何/输入 SHA256）、输出隔离与 dry-run、`--no-progress`、与既有 metrics schema 兼容） |
| 2 | 同上，全量 `tests/unit -q`（含既有全部测试） | 20.9s | **313 passed / 0 failed**（既有 297 项无回归；日志 `logs/unit_tests_2026-09-15_g0_r_qc.log`） |
| 3 | `python scripts/audit/render_picai_alignment_qc.py --help` | 秒级 | OK（argparse 装配，不读数据） |

- 说明：测试使用**合成 3D NIfTI（临时目录）**；未读取任何真实病例影像、未使用 GPU、未渲染真实数据、
  未生成任何论文指标；`third_party/`、原始数据、物化数据、split、plan 与训练产物均未改动。
- **未运行（仍待研究者执行）**：真实 16 例的渲染 → landmark 填写与校验 → 指标计算（命令见 Development_Log
  「G0-R 图像 QC 执行工具」条目与协议 §10.2）。

### G0-R 图像 QC 工具修复与验收（代理执行；合成测试，未运行真实病例）

修复内容：指标模式强制 landmark 校验（不可绕过）、严格整数坐标（拒绝 `10.9` 静默截断与 NaN/Inf/空）、
landmark 位移改为 **LPS 物理坐标**（各序列 size/spacing/origin/direction）、FOV 按 T2W/moving 各自网格分别检查、
完整性检查（病例与 case×pair 覆盖、`measured_by`、重复）、`--allow-incomplete` 诊断放行与 INCOMPLETE 标记、
`qc_geometry.json` 补齐 T2W origin/direction、ruff 6 项 + shebang 权限。

| # | 命令 | 耗时 | 结果 |
|:--|:--|--:|:--|
| 1 | `pytest -p no:cacheprovider tests/unit/test_g0_r_qc.py -q` | 12.4s | **24 passed**（新增 8 项回归：严格整数解析、多错误全收集、T2W/moving 分开 FOV、重复与缺列、物理位移含 origin/spacing/direction、严格拒绝不写指标、`--allow-incomplete` 标 INCOMPLETE、越界行不产出指标） |
| 2 | 全量 `pytest -p no:cacheprovider tests/unit -q` | 26.7s | **321 passed / 0 failed**（既有 313 项无回归） |
| 3 | `python -m ruff check src/.../g0_r_qc.py scripts/audit/render_picai_alignment_qc.py tests/unit/test_g0_r_qc.py` | 秒级 | **All checks passed!**（原 7 项：EXE001 / RUF100×3 / RUF034 / UP035 / RUF046 全部逐项修复） |
| 4 | `python scripts/audit/render_picai_alignment_qc.py --help` | 秒级 | OK（含 `--allow-incomplete` 等全部参数） |

- 说明：以上均为**合成 3D NIfTI（临时目录）**测试与静态检查；未读取真实病例影像、未渲染真实数据、未使用 GPU。
- **坐标定义已修改**（`native_index_per_series`：T2W 索引在 T2W 物化网格、moving 索引在对应 ADC/HBV 物化网格；
  位移经 SimpleITK `TransformIndexToPhysicalPoint` 计算），并已写入
  `docs/protocols/G0_R_ALIGNMENT_QC.md` §4.2 与 `configs/protocols/g0_r_alignment_qc.yaml`。
- **未运行（仍待研究者执行）**：真实 16 例渲染、landmark 填写与校验、指标计算；且
  `evaluation.landmark_rules.min_per_case_pair` 仍为 `null`（必须由研究者先冻结）。

### G0-R Automated v0.2（2026-09-16；代码/协议/合成测试；未运行真实病例）

**目标**：把 G0-R 改造为**零人工阅片、零人工 landmark、零双阅片、零人工阈值**的全自动本地 QC 流程。
新增 `scripts/audit/run_picai_alignment_qc_automated.py`、`src/.../protocols/g0_r_automated_qc.py`、
`configs/protocols/g0_r_alignment_qc_automated.yaml`、`docs/protocols/G0_R_ALIGNMENT_QC_AUTOMATED.md`、
`tests/unit/test_g0_r_automated_qc.py`。人工协议 `G0_R_ALIGNMENT_QC.md` 保留为历史草案。

**自动证据链**：A 物理几何与 FOV → B 多尺度跨模态边缘一致性（物理 mm、双向对称 edge-to-edge、
多容忍半径 F1、slicewise 仅审计）→ C 诊断性刚体残余估计（SimpleITK Euler3D + Mattes MI，仅内存、
失败记 `REGISTRATION_DIAGNOSTIC_FAILED`）→ D 合成已知位移校准（0–4 mm × 三方向 × 固定 seed，
含强度扰动稳定性）→ 阈值推导（`midpoint_zero_vs_min_detectable`）→ 单例判定 → 总体候选决策。

**实现过程中由合成测试发现并修复的三个真实缺陷**（均已写入协议 §7.4 与测试断言）：

1. `array_to_image` 的 spacing 轴序错误（numpy `(z,y,x)` vs SimpleITK `(x,y,z)`）→ 已修正并锁定；
2. 距离图语义错误（应为"到目标掩膜前景的距离"而非"到前景边界"）→ 改用
   `scipy.ndimage.distance_transform_edt(~mask, sampling=(sz,sy,sz))`；
3. 注入已知位移用常量零填充会造出人为强边界、破坏"位移越大指标越差"的单调性 →
   `border_mode=nearest` 并在配置中冻结，`constant` 被代码显式拒绝。

| # | 命令 | 耗时 | 结果 |
|:--|:--|--:|:--|
| 1 | `pytest -p no:cacheprovider tests/unit/test_g0_r_automated_qc.py -q` | 3.7s | **25 passed**（含真实 SimpleITK registration smoke、CLI dry-run、非空目录拒绝、输入哈希不变、AST 边界扫描） |
| 2 | 全量 `pytest -p no:cacheprovider tests/unit -q` | 33.7s | **481 passed / 0 failed**（既有 456 项无回归） |
| 3 | `python -m ruff check src/.../g0_r_automated_qc.py scripts/audit/run_picai_alignment_qc_automated.py tests/unit/test_g0_r_automated_qc.py` | 秒级 | **All checks passed!** |
| 4 | `python scripts/audit/run_picai_alignment_qc_automated.py --help` | 秒级 | OK（含 `--dry-run` / `--no-progress` / `--overlay` / `--max-cases` 等） |

- 说明：以上均为**合成 3D NIfTI（临时目录）**测试与静态检查；未读取真实病例影像、未运行真实 QC/配准/训练；
  未运行真实 `--dry-run`（由研究者执行）。
- **未运行（仍待研究者执行）**：`--dry-run` → 正式自动 QC → 按 `qc_report.md` 冻结 `frozen_decision`；
  **G0-R 仍为 DRAFT/Pending**，自动运行成功不等于 PASS。

## 2026-09-17

### 文档重构期间的只读核对（代理执行；秒级、只读、不读真实医学数据）

> 变更内容（文档/配置状态字段）见 `docs/Development_Log.md` 同名条目；本节只记录实际运行过的命令。

| # | 命令 | 耗时 | 结果 |
|:--|:--|--:|:--|
| 1 | `yaml.safe_load` 解析 `configs/protocols/{g0_e_independent_test,g0_sap,g0_r_alignment_qc,g0_r_alignment_qc_automated}.yaml` | 秒级 | 4 个文件全部解析通过；`protocol.status` 均为 `DRAFT`；`g0_e.audit_tool.status = INVENTORY_EXECUTED_EVIDENCE_AUDIT_NOT_EXECUTED`（本次更正后） |
| 2 | `python scripts/evaluate/prepare_g0_sap_freeze.py --check-freeze` | 秒级 | `status=DRAFT`、`freeze_readiness: NOT READY`：**24 个字段待填** + `power_memo.status=insufficient_data`；与 2026-09-15 口径一致（新增语义映射字段未改变阻塞项数量）；工具不修改配置、不判 PASS |
| 3 | `git rev-parse --is-inside-work-tree`；`git -C third_party/nnUNet rev-parse HEAD` / `describe --tags` | 秒级 | 主项目**不是 Git 仓库**（正式实验前 blocker，见 `docs/STATUS.md` B1）；`third_party/nnUNet` = `74ceb6803d10dcee29b2cc481678d3a3d069f281`，tag `v2.6.2`（与文档一致） |
| 4 | `ls outputs/diagnostics/{g0_r,g0_r_automated,g0_e}`、`ls outputs/nnUNet_results/Dataset605_PICAI`、`ls outputs/checkpoints`、`ls outputs/metrics` | 秒级 | G0-R 抽样产物（16 例）与 G0-E 清点产物存在；**`outputs/diagnostics/g0_r_automated/` 不存在**（G0-R 自动 QC 真实运行未执行）；NoFFT 输出目录不存在（正式 N0 未启动）；`outputs/checkpoints`、`outputs/metrics` 为空（M0 未训练） |
| 5 | `cat outputs/diagnostics/g0_e/20260915_093835/run_metadata.json` 与 `candidate_registry.csv` | 秒级 | 清点运行记录：`missing_evidence_items=67`、`config_sha256=90953cba…`；候选 `verdict` 列为空 → 证据审计未完成 |

- 说明：以上均为**秒级只读检查**（配置解析、冻结就绪检查、目录清点、Git 只读查询）；未读取任何真实影像体素 /
  `.b2nd` / `.pkl`，未使用 GPU，未启动数据处理、训练、推理、验证或评测；`third_party/` 未修改。
- **未运行（仍待研究者执行）**：G0-R 自动 QC 真实 16 例、G0-E 证据审计与路径冻结、G0-SAP 评测实现与冻结、
  P2B 新 M0 三步验证、正式 N0 训练。
- **注意**：`configs/protocols/g0_e_independent_test.yaml` 在清点运行后被修订（仅状态字段），其内容哈希已变化：
  清点时 `90953cba…` → 现状 `9960a4522eec2354c90736ca5ef705390ac6bde589226800fee83980e53d1595`（`sha256sum`）；
  若以修订后配置冻结 G0-E，必须重新运行清点并绑定新哈希（见 `docs/protocol_changelog.md`「2026-09-17」）。

### v2.3.1 amendment 与 SAP 生命周期落地后的只读校验（代理执行；秒级 + 合成 CPU 测试）

> 背景：按验收意见把本轮新增的四项实质性要求登记为 **v2.3.1 amendment**，并把 G0-SAP 拆为
> SAP-A（冻结方法与候选集）/ SAP-B（validation 上一次性填数）/ SAP-C（封存后最终 test 一次性评估）；
> 同时把动态事实从 runbooks 与协议文档收敛到 `docs/STATUS.md`。变更记录见 `docs/Development_Log.md`。

| # | 命令 | 耗时 | 结果 |
|:--|:--|--:|:--|
| 1 | `yaml.safe_load` 解析 4 个 `configs/protocols/*.yaml` | 秒级 | 全部解析通过；`g0_sap` = `draft-0.2`（新增 `sap_phases` 与 `power_memo.mcid/mcid_source`）；`g0_e`/`g0_r`/`g0_r_automated` 版本与状态不变（`DRAFT`） |
| 2 | `python scripts/evaluate/prepare_g0_sap_freeze.py --check-freeze` | 秒级 | `NOT READY`：**仍为 24 个字段待填** + `power_memo.status=insufficient_data`；MCID 字段尚未纳入 `FREEZE_REQUIRED_FIELDS`（已记为 P0E 实现待办，未伪造数值） |
| 3 | `sha256sum configs/protocols/g0_sap.yaml` | 秒级 | `bdc14967b555a9a6bcf4f0fb44340d6441a55236090aaccc851dff1f38dd7ce4`（修订前 `draft-0.1` 未生成过冻结包，无绑定作废） |
| 4 | `PYTHONPATH="$PWD/third_party/nnUNet:$PWD/src" CUDA_VISIBLE_DEVICES="" python -m pytest -p no:cacheprovider tests/unit -q` | 44.3s | **481 passed / 0 failed**（合成 CPU；不读真实医学数据、不使用 GPU） |

- 说明：以上均为**静态检查或合成 CPU 测试**；未运行数据处理、重采样、审计、验证、推理、评测或训练；
  未修改 `src/`、`tests/`、数据、split、plan、`third_party/`；未删除或覆盖任何既有产物；
  未把任何门标为 PASS（G0-R/G0-E/G0-SAP 仍 DRAFT，G2 仍 NOT YET EVALUATED）。
- **未运行（仍待研究者执行）**：G0-R 自动 QC 真实 16 例；G0-E 证据审计与路径冻结；G0-SAP 的 SAP-A/B/C；
  P2B 新 M0 三步验证；正式 N0 训练。

### G0-SAP 生命周期闭环后的校验（代理执行；静态检查 + 合成 CPU 测试；2026-09-17）

> 背景：为 SAP-B 增加独立结果块 `sap_b_results`、SAP-A 方法块哈希与差异审计、阶段/可见性不可变式，
> 并把 MCID 与 `sap_phases.sap_a.method_sha256` 纳入冻结必需字段。变更记录见 `docs/Development_Log.md`。

| # | 命令 | 耗时 | 结果 |
|:--|:--|--:|:--|
| 1 | `yaml.safe_load` 解析 4 个 `configs/protocols/*.yaml` | 秒级 | 全部解析通过；`g0_sap` = `draft-0.3`（新增 `sap_phases.sap_a.method_sha256`、`protocol.status_semantics` 与 `sap_b_results` 结果块）；`g0_e`/`g0_r`/`g0_r_automated` 版本与状态不变（`DRAFT`） |
| 2 | `python scripts/evaluate/prepare_g0_sap_freeze.py --check-freeze` | 秒级 | `status=DRAFT`、`sap_phases: sap_a=NOT_STARTED sap_b=NOT_STARTED sap_c=NOT_STARTED`；`freeze_readiness: NOT READY（阻塞项 2）`，未填字段清单**共 27 项**（含新增 `power_memo.mcid`、`power_memo.mcid_source`、`sap_phases.sap_a.method_sha256`）；打印当次 SAP-A 方法块哈希 `93240b2e…`（**随方法字段填写而变化**，冻结时写入 YAML） |
| 3 | `python scripts/evaluate/prepare_g0_sap_freeze.py --dry-run` | 秒级 | 计划写入 7 个文件（新增 `sap_a_method_hash.txt`、`sap_a_snapshot.json`）；不写任何文件 |
| 4 | `pytest tests/unit/test_g0_sap_freeze.py tests/unit/test_g0_protocol_common.py tests/unit/test_g0_scripts.py -q` | 2.9s | **60 passed**（覆盖 SAP-A/B/C 合法与非法组合、MCID 阻止冻结、SAP-B 结果缺失/哈希缺失、可见性不一致、方法块改动与差异审计拒绝） |
| 5 | `pytest tests/unit -q`（全量，合成 CPU） | 38.9s | **499 passed / 0 failed**（较上一轮 481 新增 18 项） |
| 6 | `sha256sum configs/protocols/g0_sap.yaml` | 秒级 | `5c5847fd545899801a97bf3086ffc57022ed32c3bfbf95ba8c1ee57e84524fe` |

| 7 | `python -m ruff check`（本次修改的 5 个文件） | 秒级 | 新增代码无新增告警；剩余条目均为既有告警（不 retro-fit） |
| 8 | 本地 Markdown 链接检查（脚本；9 个文档） | 秒级 | 断链 **0** |

- 说明：以上全部为**静态检查、YAML 解析与合成 CPU 单元测试**；未读取真实医学影像、未使用 GPU、未启动训练 /
  推理 / 验证 / 评测；未运行真实 `--dry-run` 于任何真实数据（本工具不读数据）；
  未把任何门标为 PASS（G0-R/G0-E/G0-SAP 仍 DRAFT，G2 仍 NOT YET EVALUATED）。
- **注意（哈希链）**：SAP-A 方法块哈希取决于**当前**方法字段取值；研究者填写方法块后必须重新运行工具取新哈希，
  再写入 `sap_phases.sap_a.method_sha256`。任何 SAP-A 冻结后的方法改动都会因哈希不一致而被校验器拒绝。

### M1–M4 融合模型实现轮（代理执行；静态检查 + 合成 CPU 测试；2026-09-17）

> 本条目只记录**实际运行过**的命令与结果；代码变更见 `docs/Development_Log.md`。
> **未读取真实医学影像、未使用 GPU、未启动训练/推理/验证/评测。**

| # | 命令 | 耗时 | 结果 |
|:--|:--|--:|:--|
| 1 | `pytest tests/unit/test_fusion_models.py -q`（新增测试，首轮） | 4.2s | 26 passed / 1 failed（M3 诊断量记录位置错误）→ 修正后 **43 passed / 0 failed** |
| 2 | `pytest tests/unit -q`（全量，合成 CPU） | 36–43s | **542 passed / 0 failed**（较上一轮 499 新增 43 项；无回归） |
| 3 | `python -c "load_architecture_config('configs/architectures/m0_resenc_v23.yaml').architecture_sha256()"` | 秒级 | `8bb5127e5c9a99ca1a0631cb365cdcf0b866441ac225911e17faa6bc995bbee5`（与 P2A 记录**逐位一致**，M0 身份未被破坏） |
| 4 | `python -c "load_plan(...); build_model(cfg, plan)"`（M0–M4，真实 plan，仅构建模型不读数据） | 约 60s | M0 148,193,644；M1 155,367,756；M2 155,417,679；M3 155,551,343；M4 155,531,797。SHA256 逐模型不同：M1 `a039e73e…`、M2 `8e263fd6…`、M3 `80412d4a…`、M4 `d9531261…`。gate=49,923（M2/M3/M4），zone encoder=112,800（M3=M4），conditioning=1,318（M4），zone injection=20,864（M3） |
| 5 | `ruff check`（本轮新增/改动文件：architecture.py / experiment.py / fusion_blocks.py / fusion_multi_branch.py / factory.py / models/__init__.py / train_m0.py / validate_m0_setup.py / test_fusion_models.py） | 秒级 | 新增代码 **All checks passed**；`train_m0.py` 与 `validate_m0_setup.py` 的剩余条目为既有告警（EXE001/I001/RUF100/DTZ005/PLR0124/RUF059），未 retro-fit |
| 6 | 4 个新实验配置解析检查（`load_experiment_config`） | 秒级 | 4 个配置全部解析通过；`model_id` 与 `fusion_variant` 一一对应（M1↔m1_equal … M4↔m4_conditioned_gate） |

- 说明：以上均为**静态检查、配置解析与合成 CPU 单元测试**；合成张量随机生成，不使用任何真实病例数据。
- **未运行（仍待用户执行）**：zonal prior 物化工具（尚未实现）；M0–M4 真实数据 setup/loader-smoke/overfit/训练；
  峰值显存实测；N0 训练；任何评测。

### P0 修正 + PZ/TZ prior 链路（代理执行；静态检查 + 合成 CPU 测试；2026-09-17）

> 只记录**实际运行过**的命令；**未读取真实医学数据、未使用 GPU、未运行物化/loader smoke/训练/推理/评测**。

| # | 命令 | 耗时 | 结果 |
|:--|:--|--:|:--|
| 1 | `pytest tests/unit/test_fusion_models.py -q`（重写后） | 43s | **50 passed**（含 gate 输入 spy、M3 严格 Q、overlap 拒绝、MACs 解析=hook、诊断三模式、两步训练信号） |
| 2 | `pytest tests/unit/test_zonal_prior_pipeline.py -q`（新增） | 3s | **18 passed**（物化 ok/skip/fail/oracle mismatch、sampler/pad bbox 同步、镜像同步、provider 3-tuple 与 prior 校验、trainer 调用分派、滑窗 patch 对齐、配置契约、sidecar 读取失败路径） |
| 3 | `pytest tests/unit -q`（全量，合成 CPU） | 284s | **567 passed / 0 failed**（较上一轮 542 新增 25；无回归） |
| 4 | `load_architecture_config(configs/architectures/m0_resenc_v23.yaml).architecture_sha256()` | 秒级 | `8bb5127e5c9a99ca1a0631cb365cdcf0b866441ac225911e17faa6bc995bbee5`（与历史记录**逐位一致**） |
| 5 | `build_model(cfg, plan)`（M1–M4 v2.4，真实 plan，仅构建） | 约 60s | M1 `54b66e33…` 2,032,847,718,400 MACs；M2 `77d1de43…` 2,037,920,204,800；M3 `097c19ed…` 2,051,518,924,800；M4 `8a12527d…` 2,049,546,291,200（分组值见 Development_Log） |
| 6 | `validate_m0_setup.py --config m4_conditioned_gate_picai_3d_fullres_v24.yaml --no-require-data` | 3.2s | 通过；architecture v2.4 / sha `8a12527d…`；输出写入 `outputs/diagnostics/m4/m4_conditioned_gate_yuan_picai_3d_fullres_v24/setup_check.json`；**未读病例、未用 GPU** |
| 7 | `load_experiment_config`（M1–M4 + 3 个负例：M2 带 prior、M4 缺 source、M4 输出目录缺来源标记） | 秒级 | 4 个配置解析通过；3 个负例均被 `ExperimentConfigError` 拒绝 |
| 8 | `ruff check`（本轮新增文件：`data/zonal_prior.py`、`scripts/data/materialize_zonal_prior.py`、两个测试文件、`fusion_blocks.py`、`fusion_multi_branch.py`、`profiling.py`） | 秒级 | 新增代码无告警；剩余条目为既有文件的历史 UP035/I001 风格告警（未 retro-fit） |

- 说明：以上全部为**合成 CPU 单元测试、配置解析与静态检查**；随机张量为合成生成，不涉及真实病例。
- **未运行（仍待用户执行）**：`materialize_zonal_prior.py` 真实物化；M0–M4 真实 loader smoke；GPU overfit 与
  峰值显存实测；正式训练；任何验证/推理/评测。

### 物化前加固轮的运行记录（代理执行；静态检查 + 合成 CPU 测试；2026-09-17）

> 只记录实际运行的命令；**未读取真实病例、未运行物化/loader smoke/训练/推理/评测**。

| # | 命令 | 耗时 | 结果 |
|:--|:--|--:|:--|
| 1 | `pytest tests/unit/test_materialization_hardening.py -q` | 5s | **30 passed**（配置身份、loader-smoke 分支、严格标签、oracle fail-closed、sidecar/manifest、原子写、resume、dataset 复用、dry-run） |
| 2 | `pytest tests/unit/test_zonal_prior_pipeline.py -q` | 3s | **18 passed**（已适配 `configuration`/`plans_sha256` 必填与 oracle fail-closed 语义） |
| 3 | `pytest tests/unit -q`（分批执行：81 + 516） | 14s + 92s | **597 passed / 0 failed**（较上一轮 567 新增 30；无回归） |
| 4 | `ruff check`（`data/zonal_prior.py`、`scripts/data/materialize_zonal_prior.py`、新测试、`preprocessed_store.py`） | 秒级 | 本轮新增代码 All checks passed |
| 5 | 只读证明脚本（合成数据）：M1–M4 配置身份表 | 秒级 | 四配置 `arch=v2.4`、name/root 含 v24、M3/M4 含 `yuan`、输出目录互不相同 |
| 6 | 只读证明脚本（合成数据）：sidecar+manifest 哈希链 | 秒级 | `manifest_sha256=58202bdb…`（2 例）；覆盖不足被拒；数组篡改 → `output_sha256_mismatch` |
| 7 | `pytest -k "loader_smoke or dry_run" -v` | 5s | **4 passed**（M4 3-tuple、M4 缺 sidecar 失败、M2 只读 image、dry-run 零写入） |
| 8 | `validate_m0_setup.py --config m4…v24.yaml --no-require-data` | 3s | `prior readiness: PRIOR_NOT_CHECKED（training_ready=False）` + `NOT_TRAINING_READY` 提示；JSON 含 `prior_readiness` |
| 9 | `validate_m0_setup.py --config m2…v24.yaml --no-require-data` | 3s | `prior readiness: N/A（required=False）`——M1/M2 不检查 prior |

- 全部为合成 CPU 测试、配置解析、静态检查与合成哈希链演示；合成数组随机/构造生成，不含真实病例数据。
- **未运行（仍待用户执行）**：真实 1500 例物化、M0–M4 真实 loader smoke、GPU overfit 与峰值显存实测、
  正式训练、任何验证/推理/评测。

### PZ/TZ 物化 P0/P1 修复轮的运行记录（合成测试；2026-09-17）

| # | 命令 | 耗时 | 结果 |
|:--|:--|--:|:--|
| 1 | `python -m pytest -q tests/unit/test_materialization_hardening.py` | 10s | **39 passed**（含 10 项新回归：首轮 manifest 可加载、metadata_sha256==落盘 JSON、第 2/4 次 replace 失败回滚、失败不发布 manifest、子集/dry-run 不触碰 manifest、全量 resume 覆盖完整集合、伪造/空哈希/缺 NPZ/case_ids 不一致拒绝、float64→float32 哈希一致、v24 判据、setup 退出码） |
| 2 | `python -m pytest -q tests/unit/test_zonal_prior_pipeline.py` | 3s | **18 passed** |
| 3 | `python -m pytest -q tests/unit`（source env_nnunet.sh + conda lm） | 89s | **606 passed / 0 failed**（本轮 606 = 上轮 597 + 9 净增；无回归） |
| 4 | `python -m ruff check <本轮 6 个 Python 文件>` | 秒级 | **All checks passed** |
| 5 | 合成端到端证明脚本（/tmp/zrf_proof_e2e.py，不读真实数据） | 秒级 | 首轮 2 例：`manifest_published=True`、两 hash 均为合法 hex、`metadata_sha256 == 实际落盘 JSON canonical hash`、float64→float32 落盘哈希一致、`store.load_zonal_prior_manifest` 直接通过（`array_hash_checked=True`） |
| 6 | 同上（resume 分支） | 秒级 | 全量 resume 3 例：`skipped=2 ok=1 converter_calls=1`、manifest `case_ids=[case_a,case_b,case_c]`、published=True |
| 7 | 同上（篡改分支） | 秒级 | 元数据篡改 → `metadata_sha256 与 manifest 不一致` 被拒；数组篡改 → `sidecar_identity_ok=False reason=output_sha256_mismatch` |
| 8 | 同上（成对提交分支） | 秒级 | 第 2 次 `os.replace` 失败 → 输出目录**无任何残留最终文件** |

- 全部为合成数组、静态检查与配置解析；**未读取真实病例、未运行物化/loader smoke/训练/推理/评测**。
- 真实 1500 例物化命令**本轮未提供**（按要求：修复再审之前不给正式物化命令）。

### PZ/TZ 物化链加固轮的运行记录（合成测试；2026-09-17）

| # | 命令 | 耗时 | 结果 |
|:--|:--|--:|:--|
| 1 | `python -m pytest -q tests/unit/test_materialization_hardening.py` | 12s | **80 passed**（上轮 39 → +41；含 5 条提交失败路径注入、不可恢复回滚、唯一备份、8 项写盘 fail-closed、写盘拒绝 NaN/越界/重叠/错误通道、resume 不跳过损坏 sidecar、11 项「哈希自洽的恶意 sidecar」拒绝、快照一致性 4 项、进度/统计/缓存 4 项） |
| 2 | `python -m pytest -q tests/unit/test_zonal_prior_pipeline.py` | 2s | **18 passed** |
| 3 | `python -m pytest -q tests/unit`（`source scripts/env_nnunet.sh` + conda lm） | 97s | **647 passed / 0 failed**（上轮 606 → +41，无回归） |
| 4 | `python -m ruff check <本轮 6 个 Python 文件>` | 秒级 | **All checks passed** |
| 5 | 合成证明脚本（/tmp/zrf_r3_proof.py，不读真实数据） | 秒级 | 覆盖旧对时 replace#1–#4 失败：`SidecarCommitError(stage=backup/commit)`、原始 OSError 保留、回滚错误 0、旧 JSON/NPZ 逐字节原样、残留 `[]` |
| 6 | 同上（恶意 sidecar） | 秒级 | 攻击者同步重算 metadata/manifest 哈希后 manifest 自校验通过，但深度校验仍拒绝：`sidecar configuration='2d' 与当前配置不一致` |
| 7 | 同上（进度/统计） | 秒级 | `progress=False` → 进度条 0 个；`progress=True` → `desc/total=3`；统计 `cases=3 ok=3 failed=0 array_hash_checked=True`；重复调用 `cached=True` |
| 8 | 同上（快照一致性） | 秒级 | 一致 → 通过；sha 不一致 / `array_hash_checked=False` → SystemExit 拦截 |

- 全部为合成数组、静态检查与配置解析；**未读取真实病例、未运行物化/loader smoke/训练/推理/评测**。
- 按要求：1500 例正式物化命令仍未提供；六项增强未开始；M3/M4 未标记为训练就绪。

### PZ/TZ 最后小范围修补轮的运行记录（合成测试；2026-09-17）

| # | 命令 | 耗时 | 结果 |
|:--|:--|--:|:--|
| 1 | `python -m pytest -q tests/unit/test_materialization_hardening.py` | 11s | **88 passed**（上轮 80 → +8：进度交错、进度 total、float16 dtype、冻结记录阻止替换、未冻结即读取拦截、失败清 READY、同 mtime/size 修改不复用、`_all_finite` 无桥接警告） |
| 2 | `python -m pytest -q tests/unit` | 96s | **655 passed / 0 failed**，warnings **9 → 7**（两条 NumPy `__array_wrap__` 消失） |
| 3 | `ruff check`（本轮 6 个 Python 文件） | 秒级 | **All checks passed** |
| 4 | 合成证明脚本（/tmp/zrf_r4_proof.py） | 秒级 | ① 事件序列 `progress/a→validate/a→progress/b→validate/b`（旧实现为 `progress/a,progress/b,validate/a,validate/b`）；② float16 NPZ + 哈希自洽 → manifest 自校验通过但仍被拒（`真实存储 dtype=float16`）；③ 冻结记录：自洽替换被 `metadata_sha256 与 manifest 不一致` 拒绝、`require_frozen_manifest` 未冻结即拒绝、同批其他病例不受影响；④ 成功后 `sha/array_hash_checked/records=2`，失败后 `None/False/0`，同 mtime+size 改写仍被拒 |

- 全部为合成数组、静态检查与配置解析；未读取真实病例、未运行物化/loader smoke/训练/推理/评测。

### 数据链与显存遥测修补轮的运行记录（用户执行的真实运行 + 代理合成测试；2026-09-17/18）

**用户执行的真实运行（代理未运行）**

| # | 运行 | 结果 |
|:--|:--|:--|
| 1 | `materialize_zonal_prior.py --source yuan --dry-run --limit-cases 3` | `ok=3 skipped=0 failed=0`，耗时 **8.53s**（2.84 s/it），**零写入**（`data/processed/picai_zonal_yuan_v1/` 未创建），`exit_code=0`；`plans sha256=5ca3c61b5dcc…`、`transpose_forward=(0,1,2)`、`target_spacing=(3.0,0.5,0.5)`；3 例（`11050_1001070` / `10005_1000005` / `10008_1000008`）均有病灶（3472 / 478 / 444 体素），oracle 非空转 |
| 2 | 真实 loader-smoke（哨兵问题暴露） | `PreprocessedStore.get_case("10000_1000000")` 读到 seg `[-1 0]` → `PreprocessedStoreError: seg 取值必须 ⊆ {0,1}，实际含 [-1 0]`（修复见 Development_Log §3；真实复跑待执行） |
| 3 | M4 阳性/阴性 loader-smoke（用户报告） | 已通过；背景参数：M4 参数量 **155,531,797**、batch=2、patch=[16,320,320]、AMP=True、RTX 3090 24 GB（外部进程占用约 6.2 GB） |
| 4 | 数据/预处理状态（只读核对） | 病例级物化 1500 例（`data/processed/picai/materialization_summary.json`：`n_ok=1500`、4799.4s、2026-09-14）；nnU-Net 预处理 3000 个 `.b2nd`；**plan-space prior 当时尚未物化（已于 2026-09-17 10:13 完成全量物化，见下节）** |

**代理执行的合成测试与静态检查（不读真实病例、不使用 GPU）**

| 轮次 | 命令 | 结果 |
|:--|:--|:--|
| 微型修补（dry-run 进度 + 深层不可变） | `pytest -q tests/unit/test_materialization_hardening.py` + 全量 | **91 passed**；全量 **658 passed** |
| reader 实例化修复 | 同上（`-k "r17 or r19"`）+ 全量 | 3 passed；全量 **659 passed** |
| seg 哨兵兼容 | `pytest -q tests/unit/test_seg_sentinel_mapping.py` + 全量 | **32 passed**；全量 **691 passed** |
| --overfit 显存遥测 | `pytest -q tests/unit/test_gpu_memory_profile.py` + 全量 | **15 passed**；全量 **706 passed** |
| 遥测复审修补 | 同上 + 全量 | **23 passed**；全量 **714 passed**（warnings 7） |
| 静态检查 | `ruff check <各轮修改文件>` | 每轮 **All checks passed** |

- 全部为合成数组 / mock CUDA（`FakeCudaApi`、`ExplodingCudaApi`）/ mock trainer / 静态检查与配置解析；
- 代理**未**运行 1500 例正式物化（该项已由研究者执行完成，见下节）、未读取真实病例做 prior/seg 校验、
  未做 GPU forward、未训练/推理/评测。
- 待用户执行（按依赖顺序）：① 3 例 dry-run（**已通过，见上表 #1**）→ ② ~~1500 例正式物化~~（**已完成，见下节**）
  → ③ 哨兵修复后的真实 loader-smoke 复跑 → ④ M4 1-step 显存 smoke
  （`--cases 10005_1000005,10021_1000021`，
  报告落 `outputs/diagnostics/m4/<run_name>/diagnostics/gpu_memory_profile.json`）。

### PZ/TZ prior 全量物化：首次真实全量运行（研究者执行；2026-09-17）

**完整命令**（证据：`/root/.bash_history` 第 354–356 条，tmux 会话 `lm`；同一会话此前有两次 3 例 dry-run，见上节）：

```bash
cd /opt/data/private/lm/my-projects
conda activate lm
source scripts/env_nnunet.sh
python scripts/data/materialize_zonal_prior.py   --source yuan   --resume
rc=$?
echo "exit_code=$rc"
```

| 项 | 记录 |
|:--|:--|
| 退出码 | **0**。当次 stdout 未留存为文件（终端回滚缓冲已被后续无关作业日志覆盖）。判定依据：`materialization_summary.json` 由该脚本 `return` 前写出，其 `n_failed=0` 且 `manifest_published=true`，在该版本代码中只可能走 `failure=False → return 0` 分支；且此后无任何 `materialize_zonal_prior` 重跑记录 |
| 时间窗 | 案例循环 ≈09:31:16–10:09:09（`elapsed_sec=2272.89` ≈ 37m53s；首例 sidecar 09:31:21、末例 10:09:09）；10:09:09–10:13:45（≈4m36s）为全量 resume 重建全部记录 → 校验 → 发布 canonical manifest → 写 summary |
| 结果 | `ok=1500  skipped=0  failed=0`；`oracle_mismatch=[]`；`manifest_published=true`、`manifest_n_cases=1500`、`manifest_sha256=b5d2b71dbcb437afa3a44033e4fbda56e49abbdbf6715b55e36d9a4b68a339e2`；`source=yuan`、`configuration=3d_fullres`、`plans_sha256=5ca3c61b5dcc22b294d09ae999b9282425cc29fcc1cf10cfb138c1bf64fd45c1` |
| 产物 | `data/processed/picai_zonal_yuan_v1/`：1500 `.npz` + 1500 病例 `.json` + `manifest.json` + `materialization_summary.json`（68 MB）；目录内无 tmp/bak 残留 |
| 代理只读复核（2026-09-18；仅文件清单与元数据，未读 NPZ 数组内容） | ① 文件名集合：NPZ == JSON == manifest `case_ids` == `splits_final.json` fold 0 的 train∪val（1277+223=1500）；② `manifest_sha256` 用 canonical JSON 复算一致；③ 1500 条记录的 `output_sha256`/`metadata_sha256` 全为合法 64 位小写 hex、无空值；④ 抽样 `10000_1000000`：sidecar 字段齐全（`case_id/source/configuration/plans_sha256/input_sha256/output_sha256/output_shape/output_dtype/output_channel_semantics/oracle`） |
| 代码版本对应 | 写入端 `zonal_prior.py`（mtime 09-17 09:15）与 `materialize_zonal_prior.py`（09-17 09:22）均早于本次运行（09:31 起）且此后未再修改 → 产物与**当前写入端**代码一致；读取端 `preprocessed_store.py` 于 09-18 02:01 加固（晚于本次运行）→ **尚未用当前读取端逐例复校验** |
| 本轮**未**做 | 未用当前 reader（`PreprocessedStore.load_zonal_prior_manifest`）对 1500 例做数组级复校验（数组 SHA256/shape/oracle 重验）；未跑真实 loader-smoke；未训练/推理/评测 |

> 该运行属**工程产物就绪**（B10 关闭，见 `docs/STATUS.md`），**不改变任何科学阶段门**（STATUS §1 阶段门状态未变）；6 项增强仍未开始。


### G0-R Automated draft-0.2 首次真实运行（研究者执行；2026-09-18）——失败证据原样保留

| 项 | 记录 |
|:--|:--|
| 输出目录 | `outputs/diagnostics/g0_r_automated/20260918_074219/`（**只读保留，未修改任何字段/CSV/报告/哈希**） |
| 协议身份 | `protocol_hash = 9dfdd010fa60abc8dbd4ee91cee565dc85769694daa7a3deabe645de38e8ddd1`；`config_sha256 = 67c479a055736693d290d017d2fc9bbe2f08558a593bc40cf76414920659c89a` |
| 规模与耗时 | 16 例 × 2 对 = 32 case×pair；跳过 0；用时 **1072.7 s**（`run_metadata.created_at = 2026-09-18T08:00:19`） |
| 结果 | **32/32 `REGISTRATION_DIAGNOSTIC_FAILED`**（全部同一异常 `SimpleITK 异常: Transform is not of type Euler3DTransform!`）；`calibration_passed = false`；候选 `INSUFFICIENT_EVIDENCE`；T2W-ADC / T2W-HBV 结论均为 `INSUFFICIENT_EVIDENCE` |
| 校准数值（draft-0.2 混合口径） | 主指标 `sigma1.5.edge_f1_at_1.0mm`：`zero_mean≈0.234249`、`zero_sd≈0.069802`（18 条零位移记录来自 3 例 × 2 对 × 3 次重复，组内差异 ~1e-4、组间基线差 ~0.22）；2 mm 退化均值 ≈0.008；`detection_rate_at_min = 0.000`；`monotonicity = 1.0`；`zero_false_positive_rate = 0.0` |
| 归因（代码复核 + 纯合成复现） | ① 类型错误：`SetInitialTransform(..., inPlace=False)` 返回 `CompositeTransform`，旧代码强转 Euler3D；② 校准口径：混合绝对 SD 把病例间/pair 间基线差异算进 `zero_sd`，掩盖 unit 内 2 mm 退化 |

> 该运行**不是**数据质量结论，也不改变任何门；产物按项目规则原样保留，仅作为 draft-0.3 的修复依据。

### G0-R Automated v0.3 修复轮的合成测试记录（代理执行；2026-09-18）

| # | 命令 | 结果 |
|:--|:--|:--|
| 1 | `python -m pytest -q tests/unit/test_g0_r_automated_qc.py` | **44 passed**（该文件 25 → 44） |
| 2 | `python -m pytest -q tests/unit` | **733 passed**（上一基线 714 → +19） |
| 3 | `ruff check src/.../g0_r_automated_qc.py scripts/audit/run_picai_alignment_qc_automated.py tests/unit/test_g0_r_automated_qc.py` | **All checks passed** |
| 4 | `python -m compileall -q <同两个文件>` | OK |
| 5 | 纯合成端到端演示（`/tmp/g0r_v03_synth_demo.py`，真实 SimpleITK 诊断路径 + 3 例 × 2 对） | 刚体诊断 **4/4 OK**（`final_transform_type=Euler3DTransform`、`transform_unwrap=direct`）；分 pair 校准 `T2W-ADC passed=True`、`T2W-HBV passed=True`（`det@2mm=1.00`、`mono=1.00`、`zero_FPR=0.00`）；阈值按 pair 生成；旧 v0.2 输出被 `load_output_json()` 拒绝 |

- 全部为合成数组 / 合成 NIfTI / 静态检查；**未读取真实医学影像、未使用 GPU、未训练、未推理、未评测**；
- **未运行**真实 G0-R（draft-0.3）——命令见 `docs/runbooks/g0_r_alignment_qc.md`，由研究者执行。


### 版本控制启用（研究者指令；2026-09-18）

| 项 | 记录 |
|:--|:--|
| 命令 | `git init`（默认分支 `main`）→ 增补 `.gitignore` → `git add -A` → 基线提交 |
| 结果 | 基线 commit **`23ca6cb`**；196 文件、57,260 行插入、6.8 MB；`.git` 3.1 MB；提交后 `git status` 干净 |
| 排除 | `data/{raw,interim,processed}/`（26 GB）、`workdir/`（45 GB）、`outputs/nnUNet_results/`（342 MB，含 341 MB checkpoint）、`outputs/` 其余运行产物；二进制兜底模式 |
| 未做 | 未配置/修改 git config、未添加远端、未推送；未使用 GPU、未读真实医学影像、未训练/推理/评测 |
| 影响 | `docs/STATUS.md` B1 关闭；正式 run 可记录 `git rev-parse HEAD` 作为「代码版本」证据 |


### G0-R Automated draft-0.4 修复轮的合成验证（代理执行；2026-09-18）

| # | 命令 | 结果 |
|:--|:--|:--|
| 1 | `python -m pytest -q tests/unit/test_g0_r_automated_qc.py` | **52 passed**（上一基线 44 → +8；含 11 类新回归） |
| 2 | `python -m pytest -q tests/unit` | **741 passed / 0 failed**（上一基线 733 → +8） |
| 3 | `ruff check <本轮 3 个 Python 文件>` | **All checks passed** |
| 4 | `python -m compileall -q <同两个文件>` | OK |
| 5 | `--dry-run`（研究者允许；读取 manifest 元数据与文件是否存在，**不读影像体素、不写文件**） | 协议 `G0-R-AUTOMATED / draft-0.4`、`schema=g0-r-automated/0.4`、`hash=13e7be50752db794…`；16 例 × 2 对 = 32 行，三序列齐全；**输出目录未创建**（已核对 `test -e` 为假） |

- 全部为合成数组 / 合成 NIfTI / 静态检查；**未读取真实医学影像、未使用 GPU、未训练、未推理、未评测**；
- 代理**未运行**真实 G0-R（draft-0.4）；**研究者已于同日 08:58 执行真实 16 例运行**（记录见下节 `20260918_085811`）；draft-0.3 从未在真实数据上运行；draft-0.2 失败运行产物保持原样；
- 预注册数值（位移/方向/min_detectable/检出率/multiplier/单调性/FPR 上限/min_edge_voxels/
  输入与隐私策略）**均未修改**，并有逐项比对测试锁定。

### G0-R Automated draft-0.4：首次真实全量运行（研究者执行；2026-09-18）

| 项 | 记录 |
|:--|:--|
| 完整命令 | 证据：`/root/.bash_history` 末段（tmux 会话内执行；**dry-run 与正式运行共用同一 `G0R_RUN` 变量**，故写入同一时间戳目录）——<br>`cd /opt/data/private/lm/my-projects` → `conda activate lm` → `source scripts/env_nnunet.sh`<br>`G0R_MANIFEST="outputs/diagnostics/g0_r/20260915_093819/sampling_manifest.json"`<br>`G0R_RUN="outputs/diagnostics/g0_r_automated/$(date -u +%Y%m%d_%H%M%S)"`<br>`python scripts/audit/run_picai_alignment_qc_automated.py --config configs/protocols/g0_r_alignment_qc_automated.yaml --sampling-manifest "$G0R_MANIFEST" --out-dir "$G0R_RUN" --dry-run`<br>`python scripts/audit/run_picai_alignment_qc_automated.py --config configs/protocols/g0_r_alignment_qc_automated.yaml --sampling-manifest "$G0R_MANIFEST" --out-dir "$G0R_RUN"` |
| 输出目录 | `outputs/diagnostics/g0_r_automated/20260918_085811/`（10 个文件；**只读保留，未修改任何字段/CSV/报告/哈希**） |
| 协议身份 | `protocol.version=draft-0.4`、`rule_version=0.4`、`output_schema_version=g0-r-automated/0.4`、`protocol_hash=13e7be50752db794e94c2d69dda3e2cf7dff8a42f6c8a23abb862e996a977b75`；`config_sha256=571cc6a3e204d9c430392056a4fdd59fd949cc109220d7d4fd7fb06a90573721`（**与当前 canonical 配置实测 SHA256 逐字节一致**） |
| 规模与耗时 | 16 例 × 2 对 = 32 case×pair；跳过 0；**用时 1102.8 s**（≈18m23s）；`run_metadata.created_at=2026-09-18T09:16:42` |
| 结果 | **`REGISTRATION_DIAGNOSTIC_FAILED` = 0/32**（draft-0.2 为 32/32；SimpleITK 类型强转缺陷已消除）；`INVALID_INPUT` = 0；FOV 不足 = 0；32/32 行状态均为 `INSUFFICIENT_EVIDENCE`；`calibration_passed=false`；候选 `INSUFFICIENT_EVIDENCE`；`data_written_back=false` |
| 校准（draft-0.4 口径） | unit = `(case_id, pair)`；零位移 `within_unit_mean`、噪声参考 `leave_one_out`；6 个 unit（每 pair 3 个）全部 complete、min-detectable 覆盖 6/6；条件 18 条；记录 108 条；主指标 `sigma1.5.edge_f1_at_1.0mm`：T2W-ADC `detection_rate@2.0mm = 0.667`、T2W-HBV `= 0.667`，均 < 0.9 → **两个 pair 主指标均不通过**；`zero_false_positive_rate = 0.0`、`monotonicity = 1.0` |
| 分 pair 阈值 | `edge_f1_at_1.0mm`：ADC 0.2168153899264315 / HBV 0.22974296477213557；`edge_f1_at_2.0mm`：ADC 0.34065889485812384 / HBV 0.31421439591208933；`chamfer_mm`：ADC 4.338266237742856 / HBV 5.741749641307042（聚合方法 `median of paired unit midpoints by pair`） |
| 归因 | **不是工具缺陷**，而是主指标在真实跨模态边缘数据上**对 2 mm 合成位移不够敏感**（检出率 0.667 < 0.9）→ 按协议 fail-closed 得到 `INSUFFICIENT_EVIDENCE` |
| 退出码 | **未留存**（命令未含 `rc=$?`，stdout 未落盘）；判定依据：10 个产物齐全且 `qc_report.md` / `run_metadata.json` 均由流程末尾写出、`created_at` 已写入 |
| 已知小缺陷 | `qc_report.md` 的标题与 §5 标题仍硬编码「Automated v0.3」字样（正文与元数据均为 draft-0.4）；仅文本层，不影响数值、阈值与哈希 |
| 代理只读复核（2026-09-19） | 仅读 JSON/CSV/报告元数据（未读影像体素）：核对上述哈希、计数、阈值、原因字段与产物清单 |
| 本轮**未**做 | 未修改或覆盖任何失败产物；**`frozen_decision` 未填写**；未训练/推理/评测；该工具为 CPU 路径（未使用 GPU） |

> 门状态：G0-R 仍为 **PENDING / DRAFT**；自动候选为 DRAFT，**不等于** G0-R PASS（见 `docs/STATUS.md` §1）。
> 是否转人工 landmark 复核 / 更换主指标 / 扩大抽样属 `docs/research_plan.md` §20.2 **D2**，由研究者决定。

### G0-R draft-0.4 结果复核与 D2 选项收敛（代理只读核查；2026-09-19）

| 项 | 事实 |
|:--|:--|
| 复核对象 | `outputs/diagnostics/g0_r_automated/20260918_085811/`（只读 JSON/CSV/报告元数据；**未读影像体素**） |
| 三个指标的实测灵敏度（`detection_rate@2.0mm`，要求 ≥ 0.9） | 主指标 `sigma1.5.edge_f1_at_1.0mm`：T2W-ADC **0.667** / T2W-HBV **0.667**；`sigma1.5.edge_f1_at_2.0mm`：0.667 / 0.667；`sigma1.5.chamfer_mm`：**0.778** / **0.444** → **无任何指标达标**（`monotonicity = 1.0`、`zero_false_positive_rate = 0.0` 均满足） |
| 2 mm 退化量级（主指标，T2W-ADC） | 每 unit 均值 +0.0194 / +0.0088 / +0.0034（对应零位移基线 0.348 / 0.221 / 0.212）；**个别方向样本退化为负值**（min −0.0084）→ 检出率被这些方向拉低 |
| **校准集构成（新发现）** | 抽样清单 `outputs/diagnostics/g0_r/20260915_093819/sampling_manifest.json`：`per_stratum_counts` 含 `geometry_suspect: 5`，且该层 `selection_order` = **0–4**；配置 `calibration.max_cases: 3` 规定「从 manifest 顺序取前 N 例（确定性）」→ 校准 3 例 `10057_1000057 / 10161_1000164 / 10489_1000497` **全部**是 P0A `geometry_audit.csv` 中 `severity = 2` 的病例（2 例 `geometry_suspect`、1 例 `partial_physical_overlap`） |
| 后果 | draft-0.4 的结论「主指标对 2 mm 位移不敏感」**被几何异常混杂**；不能据此断言该指标在几何干净（`severity = 0`）病例上也不敏感。这是**未决前置**，不是新结论 |
| 已排除的 D2 选项 | ① 人工 landmark 复核：**不可行**（研究者 2026-09-19 声明）；② 更换主指标：**实测关闭**（三个指标在两个 pair 上无一达标） |
| 未运行 / 未改动 | 未运行任何真实 QC、训练、推理、评测；未使用 GPU；未修改任何既有产物；**未改动任何阈值、位移集合、检出率要求或抽样清单**；`docs/research_plan.md` 未触碰 |

> 该复核只收敛 D2 的可选项，**不改变任何门状态**：G0-R 仍为 `PENDING / DRAFT`，`frozen_decision` 仍未填写。

### N0 基线训练启动（研究者执行；2026-09-19 11:15）

| 项 | 记录 |
|:--|:--|
| 命令 | `cd /opt/data/private/lm/my-projects` → `conda activate lm` → `source scripts/env_nnunet.sh` → `CUDA_VISIBLE_DEVICES=0 python scripts/train/train_n0_picai_flce.py 605 3d_fullres 0` |
| 输出目录 | `outputs/nnUNet_results/Dataset605_PICAI/nnUNetTrainerPICAI_FLCE_NoFFT__nnUNetPlans__3d_fullres/fold_0/`（**新建**；未覆盖既有 `nnUNetTrainer__…` 与 `nnUNetTrainerPICAI_FLCE__…`） |
| 日志 | `training_log_2026_9_19_11_15_10.txt`（`Epoch 0` 于 11:15:17 开始） |
| 进程与资源 | 1 个父进程 + 18 个子进程（12 train + 6 val dataloader，与 `debug.json` 一致）；GPU 约 6.3 GB（单 run） |
| FFT 崩溃检查 | 日志中 `corrupted` / `free()` / `Traceback` / `Error` 命中 **0 次** → 关闭 FFT benchmark 的修复生效 |
| 训练配置 | `Dataset605_PICAI` / `3d_fullres` / `nnUNetPlans`：patch `[16,320,320]`、batch 2、spacing `[3.0,0.5,0.5]`、PlainConvUNet 7 阶段、deep supervision 开；损失 = PI-CAI Focal+CE |
| 预期与判据 | 1000 epoch × ≈53 s/epoch → **约 15 小时**；成功判据 = 日志出现 `Training done` + 产出 `checkpoint_final.pth` |
| 结果身份 | 三项前置（G0-R / G0-E / D7）未冻结 → 本次 run 记为 **feasibility run**（不影响结果可用性；冻结完成后可升级为正式 N0） |
| 本轮未做 | 未推理、未评测；未删除、未覆盖任何既有 run 与产物；未使用 NPZ/影像做任何额外处理 |

### 文档与工作流简化（2026-09-19）

| 项 | 变化 |
|:--|:--|
| 新增日常入口 | `docs/START_HERE.md`（一页：准备数据 / 训基线 / 训自研 / 记录分析 / 常见问题） |
| 启动前置降级 | `docs/runbooks/n0_training.md` §1 与 `docs/STATUS.md` §3/§4.1：从「启动前必须冻结」改为「**不阻断启动**，只影响结果身份」；`docs/research_plan.md` 文本未改动 |
| 缓存清理 | 删除 `.pytest_cache/`、`.ruff_cache/`、17 个 `__pycache__/`（约 200 K）与临时脚本；未删除任何数据、产物、日志或文档 |
