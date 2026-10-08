# Development Log

简洁的代码/架构变更日志。只记录实质性结构变化，不保留旧阶段门（G0/G1/G2/SAP/P2A/P2B）历史。
训练与验证的运行事实见 `docs/Training_Log.md`。

## 2026-10-08 — 正式实验前最后一次收口（短预算筛选臂 / 608 审计 / 语义清理）

本轮**不新增任何研究模块**，只修复语义、补齐公平比较所需的工具，并冻结 baseline 判据；
未启动任何训练、validation、推理或数据处理。

### 实现与修复

- `multimodal/conditioned_fusion.py`：融合系数由 `logits.softmax(dim=1)` 改为显式
  `(controller_logits / SOFTMAX_TEMPERATURE).softmax(dim=1)`。常量仍冻结 **1.0**（无 CLI 旋钮、
  不做温度搜索）；T=1 时与直接 softmax **逐值相同**。同时清理 controller / lesionness 的
  `logits` 变量名冲突，避免两个不同 logits 在阅读时混淆。
- `nnunet/trainers.py`：新增 `SHORT_BUDGET_EPOCHS = 100`、`_ShortBudgetMixin` 与 8 个短预算类
  （A1/A2 + B/C/D × FLCE/DiceCE，类名 = 正式类名插入 `_100ep`），并登记 `SHORT_BUDGET_TRAINERS`。
  混搭在 `initialize()` 中、`super().initialize()` **之前**写 `self.num_epochs`，使训练循环上界、
  PolyLR 总周期（`PolyLRScheduler.max_steps`）与 checkpoint `current_epoch` 上界三者一致；
  正式 1000 epoch 类不声明类级预算、实现未动。短预算类并入 `ACTIVE_TRAINERS`（独立输出目录）。
- `scripts/train/train_nnunet.py`：ACTIVE 变体加入 8 个 `*_100ep` 变体；新增
  `expected_epoch_limit()` / `is_short_budget()`；checkpoint 预算校验与 run_config 的
  `epochs` / `fusion_constants` 改为从 Trainer 类与代码常量派生，不再硬编码 1000 与 temperature=1.0。
- 新增只读工具 `scripts/data/audit_dataset605_608_equivalence.py`（fail-closed）：raw 层比较
  病例集合/顺序、`splits_final.json`、`dataset.json`、6 通道结构、T2W/ADC/HBV 的
  size/spacing/origin/direction 与逐数组相等性、lesion 的原始/有效标签语义；preprocessed 层
  比较前三个 MRI 通道与 seg，并核对两侧 `nnUNetPlans.json`。PASS 要求两层；仅 raw 通过为
  `RAW_ONLY_PASS`（退出码 2）。608 prior 通道只做信息性统计（不评价 prior 质量）。
  **尚未在真实 Dataset608 上运行**（608 未物化/未预处理）。
- 文档：Experiment Plan 新增 §2.1「baseline selection rule」（先于结果冻结）并重写 §7 Budget
  （单一 100 epoch 档位、同预算互比、实现边界、禁止与 1000 epoch 比较）；Method 记录温度的显式
  计算与热路径检查的实测代价；README 给出短预算与 608 审计命令；Training Log 为旧 WG 解释加
  `SUPERSEDED` 标注并登记本轮无训练。

### 实际执行的检查

均在 conda `lm`、固定 nnU-Net 环境执行；测试只使用合成数据与 pytest 临时目录。

- `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B -m pytest -p no:cacheprovider`：
  **717 passed，174 dependency deprecation warnings，42.84s**（本轮新增 40 个用例：短预算 12、
  temperature 1、608 审计 28 中 27 为新增文件 + 既有套件不受影响）。
- `python -m ruff check --no-cache src scripts tests`：通过。
- `compileall`、`git diff --check`、训练入口 `--help`、审计工具 `--help`：通过。
- 短预算真实性检查（`tests/unit/test_multimodal_fusion.py`）：8 个短预算类经真实
  `nnUNetTrainer.__init__` + `initialize()` 后 `num_epochs == lr_scheduler.max_steps == 100`，
  正式同类仍为 1000；短预算类不覆盖 `run_training` / `configure_optimizers` /
  `perform_actual_validation`。入口 resume 守卫对 100 epoch 类只接受 `current_epoch <= 100`
  且 trainer 名匹配的 checkpoint。
- 热路径 fail-closed 检查的合成 CUDA 计时（真实 patch 形状 `[1,6,16,320,320]`，RTX 3090，3 次
  独立 trial，每次 30 iteration）：`fuse` 前端 25.98 / 26.01 / 26.00 ms，
  三项检查（`isfinite().all()` + 两个范围 `.any()`）0.351 / 0.351 / 0.350 ms，
  占比 **1.34–1.35%**。**结论：保留逐 patch 检查**，不做热点路径弱化。
  边界：合成张量、单卡、非 profiler trace，不含 backbone 前向，不构成吞吐声明。

### 参数与未验证边界

网络容量**未变**（本轮只改系数计算方式、预算类与工具）。用真实 Dataset605 plans 重新打印：

| model | total / trainable | delta | delta % |
|---|---:|---:|---:|
| A | 44,577,932 | 0 | 0 |
| B | 44,583,983 | 6,051 | 0.013574 |
| C | 44,584,913 | 6,981 | 0.015660 |
| D | 44,584,957 | 7,025 | 0.015759 |

未验证：短预算训练是否真的在 100 epoch 内产生可用的方向信号（未运行任何 100ep 训练）；
Dataset608 审计未在真实数据上运行；未测 FLOPs 与 GPU 显存。

## 2026-10-08 — 预测解剖与粗定位共同条件化局部多模态融合

### 实现与修复

- 新增 `multimodal/__init__.py` / `conditioned_fusion.py`：三个独立 width=8 shallow stems，
  neutral 24→3 projection，pre-fusion factor-two coarse lesionness，softmax local controller，
  zero-init residual；网络包装原生3通道backbone，不依赖legacy。
- `nnunet/trainers.py`：修复原supporting Trainer的classmethod/instance helper绑定；
  新B/C/D各提供FLCE与原生DiceCE继承家族，复用positive sampling、loss、原生训练与推理。
  ACTIVE=9、SUPPORTING=4、LEGACY=13，共26个静态类名，保留全部20个原类名。
  新E尚未绑定；不把固定原C父类的历史HN类称为best(C,D)。
- `anatomy/validation.py`：增加独立soft heads与ordered hard-export重建只读诊断、
  quantiles、macro/micro汇总与进度，空分母为null。真实诊断事实只记Training_Log与对应实验文档。
- `anatomy/dataset.py`：新增独立Dataset608 predicted-prior契约与用户运行物化入口。
  完整病例/患者split/provenance/geometry检查，拒绝Dataset606与覆盖；不自行重采样。
  单冻结模型生成器标记IN_SAMPLE/HELD_OUT，不声称生成OOF。
- 新增 `nnunet/prior_preprocessor.py`：继承原生preprocessor，MRI原生处理后对prior采用
  相同transpose/crop及原生线性概率重采样，避免prior改变MRI crop或cubic概率越界。
  从605冻结网络/spacing/patch/batch；拒绝已有目标。真实数据物化/预处理未执行。
- `scripts/train/train_nnunet.py`：ACTIVE/SUPPORTING/LEGACY分层入口，新增seeded独立类名与
  输出目录；从头运行拒绝已有输出，恢复核对自身Trainer/预算/plans/config/fold/dataset。
  `scripts/inference/predict_nnunet.py`注册项目preprocessor，推理仍走原生入口。
- `sampling/hard_negative.py`：组合ROI与HN loader，保留非挖掘槽位ROI并消费两套队列。
  不修改验证采样，不将此修复视为新E实现。
- 更新README/Research_Plan/Method/Experiment_Plan/AGENTS的唯一主线与边界；
  Evaluation_Protocol仅补独立soft-head诊断和系数行为分析口径，lesion评价常量/匹配/schema未改。
  Training_Log/Findings/anatomy实验详情更正hard WG与soft WG混淆，保留历史数值及其边界。

### 实际执行的检查

在conda lm、固定nnU-Net环境执行；不读取真实影像的测试只写pytest临时目录。

- `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B -m pytest -p no:cacheprovider`：
  **677 passed，174 dependency deprecation warnings，42.25s**。
- `ruff check --no-cache src scripts tests`：通过。
- `compileall.compile_dir` 对src/scripts/tests：通过，pycache仅写临时目录。
- 训练入口`--help`、`git diff --check`：通过。
- 新增 `tests/unit/test_multimodal_fusion.py`；更新既有Trainer注册与HN测试。
  包含独立参数/梯度、初始恒等、softmax归一化、非法概率、几何与provenance、seeded解析、
  MRI预处理逐值一致（含非恒等transpose）、合成物化拒绝覆盖、组合loader连续batch队列检查。
- 真Trainer smoke：A1/A2与B/C/D两loss家族共8类，各minimal/real605架构；
  resolve→__init__→initialize→native optimizer/scheduler/loss→synthetic forward/backward，
  以及原生DS关闭后的无GT forward。真实605架构输入缩为[16,64,64]；
  **不是**原始[16,320,320]全patch显存验证，未构造真实数据训练dataloader或启动训练loop。
- lm缺失测试依赖，安装并验证pytest9.1.1、ruff0.16.10、MedPy0.5.2、surface-distance0.1
  （后者带absl-py2.5.0）；没有升降级torch/numpy/SimpleITK/nnunetv2。

### 参数与未验证边界

使用真实Dataset605 nnUNetPlans架构构造；以下total=trainable，delta相对A：

| model | total / trainable | delta | delta % |
|---|---:|---:|---:|
| A | 44,577,932 | 0 | 0 |
| B | 44,583,983 | 6,051 | 0.013574 |
| C | 44,584,913 | 6,981 | 0.015660 |
| D | 44,584,957 | 7,025 | 0.015759 |

工具为`parameter_counts`；未测FLOPs、GPU full-patch显存/速度。
系数offline hook仅最近一次forward的patch，含case/checkpoint/geometry；
全体积滑窗聚合恢复与按病例自动分区分析尚未实现。
原ROI original→preprocessed bbox坐标链尚待修复/验证，supporting不标为正式就绪。
新B/C/D仅骨架/合成构造已验证，无训练或validation效果；短预算独立Trainer与新E待后续。
第三方及legacy实现未改，未commit/push，未删除/覆盖既有数据或模型。

## 2026-10-08 — 系统级重构：确立单一主线（Anatomy-Guided Lesion-Aware Coarse-to-Fine）

### 背景

仓库此前同时存在三条竞争性主线（旧输入级 Gate 线、特征级融合线、短预算同区参照 + 两阶段解剖候选），
README 首页不是主线、Trainer 数量口径三处不一致（11 / 15 / 16），文档与事实有 6 处冲突。本轮做
**系统级重构、清理与重新定线**，把项目收敛到单一主线：
**基于预测解剖先验的病灶感知粗到细前列腺癌分割**。

### 新增

- `pyproject.toml`——依赖声明 + ruff / pytest 配置（此前 ruff 使用默认激进规则，pytest 配置只在
  `pytest.ini`）。
- `src/zonal_reliability_fusion/evaluation/`（`protocol.py` / `case_metrics.py` /
  `lesion_metrics.py` / `anatomy_metrics.py`）——把评价体系从 2352 行的脚本中抽出来，成为与模型
  **完全解耦**的独立包（不 import torch / nnU-Net）。
- `src/zonal_reliability_fusion/anatomy/`（`contracts.py` / `dataset.py` / `inference.py` /
  `validation.py`）——Stage-1 的契约、先验布局、预测校验与质量评价。
- `src/zonal_reliability_fusion/lesion/`（`baseline.py` / `roi.py` / `lesionness.py` /
  `coarse_to_fine.py` / `zone_conditioning.py` / `prior_channels.py`）——新主线的方法本体：
  Anatomy-Guided ROI（物理 margin，禁 hard mask，不重采样）、物理半径膨胀的 lesionness 目标、
  原生 backbone + coarse 头 + soft 残差 refinement、soft PZ/TZ 解剖上下文。
- `src/zonal_reliability_fusion/sampling/`（`positive_sampling.py` / `hard_negative.py`）——阳性
  采样移入并把训练 loader 的构造抽成扩展点（`_resolve_train_loader_class` /
  `_train_loader_kwargs`），使困难负样本采样可以叠加而**不复制** `get_dataloaders`。
- `src/zonal_reliability_fusion/nnunet/`（`runtime.py` / `bases.py` / `losses.py` /
  `augmentation.py` / `roi_sampling.py` / `seeds.py`）——运行时校验、共用 Trainer 基类、
  只有两样东西的损失模块、NoFFT 修补、ROI 训练采样层、显式种子控制。
- `scripts/data/check_split_integrity.py`——§24 要求的 split 泄漏自动检查，输出 PASS / FAIL，
  遇泄漏 fail closed。
- `scripts/data/build_anatomy_roi_set.py`——由**预测 WG** 构建 ROI 集合（不读 lesion GT、不读
  原始影像、不重采样）。
- `scripts/data/mine_hard_negatives.py`——Round-2 困难负样本挖掘（`--split` 只允许 `train`）。
- `docs/`：`Research_Plan.md` 重写、`Method.md`、`Experiment_Plan.md`、
  `Evaluation_Protocol.md`、`REFACTOR_AUDIT.md`、`archive/`（README + legacy_gate_research +
  old_research_plans + historical_experiments）。
- 测试：`test_lesion_roi.py`、`test_lesionness_coarse_to_fine.py`、`test_hard_negative_sampling.py`、
  `test_anatomy_priors.py`、`test_split_integrity_and_roi_set.py`。

### 修改

- `src/zonal_reliability_fusion/nnunet/trainers.py`——重写为 ACTIVE（B/C/D/E + Stage 1）+
  legacy 兼容再导出 + `ACTIVE_TRAINERS` / `LEGACY_TRAINERS` / `PROJECT_TRAINERS` 分层注册表。
- `scripts/train/train_nnunet.py`——variant 分 `ACTIVE_VARIANTS` / `LEGACY_VARIANTS`，默认
  `--help` 只显示主线，加 `--legacy` 才显示归档条件；新增 `--seed` / `--roi-set` /
  `--hard-negative-set` / `--run-config`；新条件在构造 Trainer 前做前置校验（缺 ROI 集合即报错，
  不静默退化）。
- `scripts/inference/predict_nnunet.py`——`PROJECT_TRAINER_NAMES` 改为**由注册表派生**（消除
  第二份手写清单），解剖先验逻辑委托给 `anatomy.inference`。
- `scripts/evaluate_segmentation.py`——指标实现改为从 `evaluation/` 导入，脚本只保留 CLI、产物
  读取、模型间可比性检查、编排与落盘；顶部加 `sys.path` 引导使其可独立运行。
- `scripts/data/prepare_picai_nnunet.py`——解剖契约导入收敛到 `anatomy.contracts`（canonical）。
- `docs/Findings.md`——旧结论重新归类为 **Preliminary Findings**，新增 **Mainline Findings**
  章节（当前只记录「尚无证据」与判据指针），并修正「五次完成训练」为实际的 7 次。
- `docs/Training_Log.md`——新增 2026-10-08 重构条目与新主线未训练状态表；更正一处过期索引
  （`anatomy_gate` 曾写作"训练未运行"，实际已完成）。
- `README.md`——彻底重写：只介绍新主线、单一 pipeline、strong baseline、评价体系、状态表、
  命令与目录结构。
- `AGENTS.md` —— §6 文档纪律与新的五份治理文档对齐。
- `.gitignore`——补充生成型缓存、新主线中间产物、更多二进制兜底。

### 测试

- 修正 26 个**既有失败**：其中 20 个是环境缺少 `medpy`（已安装），5 个 `test_short_zonal_*` 断言
  已迁移到 `initialize()` 的旧实现，1 个硬编码 variant 集合缺少新条件。
- 把结构性测试从"硬编码名单"改为"与注册表一致"的断言（`test_nnunet_trainers.py` /
  `test_positive_case_sampling.py` / `test_predict_entry.py`），使新增条件不再需要同步修改三处。
- `pytest`：**552 passed, 0 failed**。
- `ruff check src scripts tests`：**All checks passed**。
- `python -m compileall src scripts tests`：通过。

### 未运行（明确声明）

- **未启动任何训练**：`dicece_positive_sampling`、`anatomy_joint_100ep`、`lesion_roi`、
  `lesion_coarse_to_fine`、`lesion_zone_refine`、`lesion_hard_negative` 全部未训练；
- **未运行** validation、推理、模型实例化、dataloader 真实构造、GPU 命令；
- **未运行**数据准备、preprocessing、planning、ROI 集合构建、困难负样本挖掘；
- **未读取**任何真实医学影像；测试全部为纯合成 CPU 测试；
- **未创建**任何模型输出目录 / checkpoint / 先验产物；
- **未删除或覆盖**任何既有 checkpoint、validation 产物、summary、训练日志或数据；
- **未修改** `third_party/nnUNet`（仍为只读 v2.6.2）；
- **未推送**到远端。

### 兼容性声明

- 全部 16 个历史 Trainer 类名与实现**逐字保留**（现位于 `legacy/fusion_trainers.py`），
  `PROJECT_TRAINERS` 仍包含全部历史键，因此既有 checkpoint 的输出目录解析不受影响；
- `nnunet/networks.py`、`nnunet/transforms.py`、`nnunet/sampling.py` 保留为**兼容转发层**，
  历史导入路径与测试引用继续有效；
- Python 包名仍为 `zonal_reliability_fusion`（历史内部标识），改名会让既有 checkpoint 输出目录
  解析失效且无研究收益；
- `scripts/` 下既有脚本路径保持不变（被本文件与 `Training_Log` 的历史记录引用）。
- **历史文档引用映射**：本文件与 `Training_Log` 的旧条目中出现的
  `docs/experiments/image_gate.md` / `image_gate_positive_sampling.md` /
  `anatomy_gate_positive_sampling.md` 现已移至 `docs/archive/historical_experiments/` 下的同名文件。
  治理文档（README / Findings / Training_Log）内的链接已更新为新路径；**历史条目正文一字未改**，
  以便保留当时的写作语境。

---

---

## 2026-10-06 — 解剖标签审计工具修订（同日第二轮，代码就绪，真实审计仍未运行）

针对首版审计工具做了一轮严格性修订。**首版条目中以下表述已被本轮取代**，以本轮为准：

- 首版把 WG 结论写成 7 种取值，且未区分「候选未完成比较」与「已比较且不匹配」；
- 首版把退出码 0 定义为「三个状态均成立」，未区分 `null`（未评估 / 子集）；
- 首版有 `--overwrite` 开关，与「只生成全新报告」冲突；
- 首版把候选四类区域编码描述为在若干场景下自洽，容易被读成**唯一**合法编码。

**标签值检查（禁止整数截断）**：新增标签契约（`LABEL_CONTRACTS`）分别定义 WG / zonal /
lesion / Prostate158 四类标签的维度、有限性、整数性与允许取值集合。原 `_sorted_unique`
用 `int(v)` 会把 `1.5`、`2.7` 静默变成合法值，已删除。观测值以精确文本原样保留
（`0.5` / `1.5` / `2.7` / 负值 / `nan` / `±inf` 各有独立文本），非整数、NaN/±Inf、
非法取值、维度错误一律逐例记录并进入问题汇总。`label_foreground_mask` 只取**允许的非零取值**，
非法或非有限体素绝不进入前景或 ROI——不再有 `arr > 0` 捷径。NaN/Inf 只影响涉事病例的结论，
报告仍是合法 JSON（非有限值写为显式文本）。实测确认 SimpleITK 的 NIfTI 往返会把 NaN/±Inf
写成 0，因此非有限值的端到端测试在**读取层**注入，而不是指望磁盘保留。

**总体状态覆盖全部被请求任务**：`inputs`（数据完整性）改为**强制**审计项，不能用
`--stages` 关闭，因此只跑 `--stages roi` 也会完成存在性、几何基准、标签契约、患者关系与
重复 ID 检查。ROI 的 `lesion_missing` / `lesion_read_error` / 非法标签 / 几何错误 /
T2W 基准不可用，以及 wg / zonal / prostate158 各 stage 的失败，全部进入 `problems`。
新增 `source_audit_status`（`resolved` / `unresolved` / `not_evaluated`）：**未请求来源审计时
`source_resolved` 为 `null`**，「没检查」不构成已解决的证据。已知缺失与未认可状态严格分开：
只有显式允许清单（`KNOWN_WG_MISSING_STATUSES`）中的状态算已知缺失，
其他非 `exact` 的 `wg_status` 记为未认可并进入问题汇总。`--max-cases` 子集模式下
`scope=subset`，`data_valid` / `source_resolved` 均为 `null`、`execution_complete` 为 `false`。

**覆盖率分母完整性**：区分「合法空病灶标签」（真实阴性，实例数 0，可计入分母）与
「参考标签不可用」（不可读 / 非法 / 几何非法 / 与 T2W 不同网格 → 实例数 `null`，
**不当作阴性**）。汇总新增 `n_cases_reference_evaluable`、`n_cases_reference_unavailable`、
`denominator_complete`、`n_reference_instances_observed`；分母不完整时全部「全部参考病灶」
口径字段置 `null`，另在 `evaluable_subset_only` 中显式标注范围报告可评估子集。
按 overall / train / validation 分别判定完整性（`per_split_denominator_complete`）。
腺体来源不可用但参考标签合法时，回退完整 T2W FOV 并**继续计算真实覆盖**，计入分母。
实例数一律由实际参考掩膜的 3D 6-邻域连通域计算，无任何预先给定的数量。

**统一 T2W 几何基准**：新增 `Geometry.validity_problems()`（维度、spacing 正且有限、
origin 有限、direction 正交归一且 |det|=1、size 正）与 `compare_geometry()`。
所有几何比较显式 `rtol=0`、`atol=1e-4`，避免相对容差掩盖尺度差异。lesion 必须与 T2W
同网格才能按数组 bbox 计算覆盖；不一致时记 `lesion_grid_mismatch_vs_t2w` 并**明确声明
不得把 lesion 的 FOV 称为完整 T2W FOV**。ROI 回退目标改为完整 **T2W** FOV。

**保守 WG 来源判定**：候选缺失 / 不可读 / 几何非法 / 不同网格一律记为 `not_comparable`
并逐项记录读取状态与原因，**绝不**当作「已比较且不匹配」。只有竞争候选也完成比较且不匹配时
才写 `only_bosma_match` / `only_guerbet_match`；候选未完成比较时新增
`match_with_unresolved_uniqueness`，全部不可比时新增 `no_comparable_candidate`。
`unique_source_unresolved` 与 `all_candidates_completed_comparison` 显式入报告。

**区域编码说明修订**：补充「同一体素 PZ、TZ 两头同时阳性」的测试，核验导出时
后写的区域覆盖先写的区域；明确区分三种掩膜层级（原始 sigmoid 头阈值掩膜 / 导出整数图
重建的区域掩膜 / 训练参考区域掩膜）。同时固定：候选四类区域编码只是**一个**自洽候选，
不是唯一合法编码（纯多类同样自洽）；任意预测头组合与导出区域**不保证**严格一致。

**输出收紧**：移除 `--overwrite`。报告只能是**全新** JSON 路径；已存在、与任一输入文件重合、
后缀属于影像或 checkpoint 类（`.nii.gz` / `.npy` / `.npz` / `.pth` / `.b2nd` / `.pkl` /
`.csv` 等）、或落在受保护目录下一律直接拒绝。保留临时文件 + `os.replace` 原子写入与逐例诊断。

**报告口径更正（两处过度声明）**：其一，外层划分的多 study 患者**不是**「一律恰好两个
study」。按 `data/splits/picai_nnunet_split_cases.csv` 实测：train 侧 1256 名患者 =
1236 名 1 个 study + 19 名 2 个 + **1 名 3 个**（共 1277）；validation 侧 220 名 =
217 名 1 个 + 3 名 2 个（共 223）；全体 1476 名 = 1453 + 22 + 1（共 1500）。
因此阶段一的内部折必须按患者分组，且分组逻辑要能处理「一个患者 3 个 study」的情形。
其二，**K 增大不保证折外预测质量单调提高**：折外质量还取决于每个内部折的训练集规模是否
与阶段二实际使用的模型可比、以及单次运行的随机性；K 的取舍应在实测单折耗时与训练集规模后
再定，不能预设单调关系。

**回归**：`pytest -q` 全量 **473 passed**（新增/重写 97 项纯合成测试，CPU、`tmp_path` 内自造
NIfTI，不读真实医学数据）。本节只新增/修改标签值检查、状态判定、分母汇总、几何基准、
WG 保守判定、输出策略与对应测试；未创建 Dataset607、Trainer 或预测缓存，未修改第三方源码、
原始医学数据、现有预处理产物或 checkpoint。**真实数据审计尚未运行**，
WG 来源、PZ∪TZ 与 WG 的关系、Prostate158 取值与 ROI 覆盖率一律待用户运行后核定。

---

## 2026-10-06 — 前列腺解剖标签只读审计工具（代码就绪，真实审计未运行）

新增 `scripts/data/audit_prostate_anatomy_labels.py`（只读、fail-closed）。它回答阶段一之前的四个
证据缺口，且**本次未在真实医学数据上运行**：物化 WG 的来源内容对应、`PZ ∪ TZ` 与物化 WG 的
并集/包含关系、`Prostate158` 解剖标签的整数值与几何、以及由现有标签生成「腺体代理 ROI」时
对全部参考病灶实例的覆盖。

**输入与分母**：病例集合一律取自 `picai_manifest.csv`，**不使用 glob 成功读取的病例当分母**；
split 与既有物化报告作为并列的第三类证据。报告显式区分 `original` / `materialized` /
`existing_report` 三类证据，可互相矛盾但不取其一当真值；自动解剖标签不被写成人工真值。
已知缺失 WG 的病例记录 `wg_status` 后保留在分母里，不静默排除。

**WG 内容对应**：先比较 size/spacing/origin/direction，同网格才比较体素；不同网格判
`grid_mismatch` 并记录原因，**不自动重采样**。结论分为 `only_bosma_match` /
`only_guerbet_match` / `both_match` / `neither_match` / `missing` / `grid_mismatch` /
`read_error`。两套来源同时一致记为**来源歧义**而非程序失败。**不使用压缩文件大小推断来源**；
体素一致只证明内容对应，不证明历史生成来源。

**分区关系**：对 Yuan / Hevi 分别记录值域、体素数、mm³ 体积、并集与 WG 的 Dice、并集在 WG
外的体积、WG 不在并集内的体积、空标签与异常几何，并单独计算两者的 PZ/TZ 一致性。
跨网格比较默认不执行。两套自动标签之间的 Dice 明确标注为**算法间一致性**，不是准确度或噪声下限。

**Prostate158**：按队列与读者分别测量 `t2_anatomy_reader{1,2}` 的整数值与几何。文献依据
（Adams/Bressem et al., Comput Biol Med 2022, DOI 10.1016/j.compbiomed.2022.105817 把解剖类别
写作 central gland（central zone + transitional zone）与 peripheral zone，而官方 README 表头用
「Transitional Zone」一词）与**本地观测到的整数值**分列存放，语义保持 `semantic_unresolved`：
不得把 CG 改名为 TZ，也不得仅凭整数值反推解剖语义。本次不读取、不推断 `BPH` / `PCA` 的
`ROI.nii` 语义，也不实现全数据集两两去重。

**腺体代理 ROI 覆盖**：候选为物化 WG、Yuan 的 PZ∪TZ、Hevi 的 PZ∪TZ。ROI = 三维最大连通域
（6-邻域，与 `scripts/evaluate_segmentation.py` 的 `CONNECTIVITY` 同值并测试核对）的包围盒，
按固定物理余量向外扩展；余量转体素按 numpy 轴序 (z,y,x) 用 `ceil` **向外取整**后裁剪到体积范围。
来源标签缺失、不可读、为空或与 lesion 网格/形状不一致时回退完整 T2W/lesion FOV 并逐例记录原因。
对所有参考病灶实例统计覆盖率、总体体素覆盖率、实例覆盖率宏平均、完整覆盖比例、完全漏失数与
回退病例的覆盖结果；分母是**全部**参考实例，不限于检出/有交集/预测成功的病例。按外层
train/validation 分开汇总，并在报告里写明后续 ROI 参数选择只应使用训练侧。
ROI 只在内存中构造，不落盘裁剪数据、不修改任何输入。该 stage 明确标注为**现有标签**的覆盖，
不是阶段一预测模型的效果上界。

**状态与安全**：`execution_complete` / `data_valid` / `source_resolved` 三个独立状态，**不提供
合并的 `all_pass`**；来源不明或算法间不一致是观测结果，不等于执行失败。异常病例保留在报告中，
分母与排除原因明确。默认拒绝覆盖既有报告，输出前校验不得落在 `data/`、`workdir/`、
`third_party/` 与 `/opt/data/private/lm/data/` 之下；先写同目录临时文件再 `os.replace` 原子提交。
退出码 0 = 三个状态均成立；2 = 报告已写出但 `data_valid` 或 `source_resolved` 为假；
3 = 前置条件失败（不写报告）。tqdm 覆盖各主要读取循环，结束打印成功/失败/跳过/耗时/输出路径。

**候选阶段一区域编码的修正（纯合成验证，未训练）**：上一轮提出的
`labels={background:0, WG:[1,2], PZ:1, TZ:2}` + `regions_class_order=[3,1,2]` 被证明自相矛盾 ——
仅 WG 头阳性时导出整数值 3，而 WG 区域由 `(1,2)` 重建，**看不到**标签 3，原生 WG 区域指标恒为
完全漏分。改用固定版本 `LabelManager` 与 `nnunetv2.evaluation.evaluate_predictions` 的原生函数
在纯合成张量上验证：只有互斥区域集合才能经「整数图导出 + 原生区域评测」精确还原。
候选 `labels={background:0, WG:[1,2,3], PZ:2, TZ:3}` + `regions_class_order=[1,2,3]` 在上述
五种单头/组合场景下区域掩膜与「哪些头阳性」严格一致。同时固定两条边界：独立 sigmoid **不保证**
`WG ⊇ PZ ∪ TZ`（两个方向都可能违反），包含关系必须测量而非声称；若只训练 PZ/TZ 多类别，
派生腺体概率是 `p_PZ + p_TZ` 而**不是** `max`，且代理概率无法表达 WG 独有的腺体组织。
本轮只验证候选编码的导出一致性，**不确定最终阶段一编码**。

**测试**：新增 `tests/unit/test_audit_prostate_anatomy_labels.py`（60 项纯合成测试，CPU，
`tmp_path` 内自造 NIfTI，不读真实医学数据）。覆盖两套来源同时一致与单来源一致、`neither_match`、
几何不一致、已知缺失、意外缺失、非法标签、空标签、并集相等与双向不包含、区域导出/原生评测
一致性、非等距 spacing 与轴顺序、余量向外取整与边界裁剪、ROI 全覆盖/部分覆盖/完全漏失/多病灶、
缺失或空腺体标签回退完整 FOV、读取失败不缩小分母、拒绝覆盖既有报告、输入文件逐字节未修改。
测试还发现并修正了一处真实缺陷：实际施加余量按「连通域相对 ROI 向外扩了多少」计算，原先的
减序写反，被体积边界裁剪时会报成负值。

**回归**：`pytest -q` 全量 **436 passed**（含新增 60 项）。为消除 8 项既有失败，按 AGENTS.md §1
向 `lm` 安装既有评估器 fail-closed 所要求的 `surface-distance==0.1` 与 `medpy`（均以
`--no-deps` 安装，未升降级 `nnunetv2`/`torch`/`numpy`/`SimpleITK`；安装后核对
numpy 2.2.6、scipy 1.15.3、SimpleITK 2.5.3、torch 2.10.0+cu128 不变）。

本次只新增上述两个文件并更新本日志；未创建阶段一数据集、Trainer、预测缓存或实验文档，
未修改第三方源码、原始医学数据、现有预处理产物或 checkpoint。**真实数据审计尚未运行，
其结论（WG 来源、PZ∪TZ 与 WG 的关系、Prostate158 取值、ROI 覆盖率）一律待用户运行后核定，
不得登记为已完成。**

---

## 2026-10-06 — 同区参照融合与独立短预算对照（未训练）

最终回归：`test_nnunet_networks.py`、`test_nnunet_trainers.py`、
`test_positive_case_sampling.py`、`test_predict_entry.py`、`test_prior_transforms.py`
合计 **211 passed**（CPU 合成测试，8.18 秒）。仅有现有依赖的弃用警告；
不涉及真实医学数据或训练。

在既有 `networks.py` 实现 `ZonalReferenceResidualFusion` / `ZonalReferenceFusionNNUNet`：
区域条件局部一阶/二阶统计 → 标准化对比 → 零初始化特征修正；固定支持强度与学习式强度
两种条件。统计在 float32 下进行，区域重叠归一化、无局部支持回退、缺通道/非有限先验报错。
上采样后再以原分辨率分区隶属权重限制修正，防止扩散到分区外。
模型读取 Dataset606 已有自动分区二值掩膜经既有插值形成的隶属权重，不使用病灶标签构造网络输入，不宣称校准不确定性。

新增四个 `*_100ep_NoFFT` Trainer，分别对应普通融合、普通分区 gate、固定参照修正和
自适应参照修正；共同设置原生预算与 PolyLR horizon 为 100，统一 FLCE、阳性采样及增强。
不更改旧 Trainer、现有 plans 或固定 nnU-Net 源码，不自行实现训练器/推理器。
训练入口新增四个 `_100ep` 别名，并限制 Dataset606 / 3d_fullres；新分支拒绝同名目录覆盖
和缺 checkpoint 的静默重训。预测入口完整注册新 Trainer 名，保持 checkpoint 可解析。

修改已有网络、Trainer、训练/预测入口、相关合成测试及五份治理文档；未新增项目文件，
未创建实验详情文档，未修改原始医学数据、预处理产物或任何 checkpoint。
纯合成测试覆盖同区统计（包括跨 coarse voxel 的分区边界）、分区隶属权重重叠、空/稀少参照回退、
零初始化、分区外上采样回退、半精度二阶矩防溢出、梯度、深监督、strict state_dict、推理解析、
预算/调度和入口安全约束。探索命令统一用原生 `nnUNet_compile=false`，减少首次编译等待，
实际 epoch 耗时仍需记录；未改写编译钩子。
为运行测试，按 AGENTS.md 将缺失的 pytest 9.1.1 安装到 lm；未升降级 torch/nnunetv2/numpy/SimpleITK。
首版尚无真实显存、耗时或分割结果。完整腺体/分区预测与 ROI 两阶段系统尚未接入。

---

## 2026-09-28 — 新增独立强参考基线 `nnUNetTrainerPICAI_DiceCE_PositiveSampling_NoFFT`（代码接线，未训练）

为回答「在 PositiveSampling 已稳定病灶暴露之后，nnU-Net 原生 Dice+CE 是否优于 PI-CAI Focal+CE
（A = `positive_sampling`）」新增一个**单变量损失消融 Trainer**。已中止的 `optimized_baseline`
（DiceCE_NoFFT）使用原生采样且只训练到约 epoch 376 就停止，不能回答该问题。

定位：**独立于 A→B→C→D 主研究链的增强参考基线 / 损失消融**，不是新的主研究阶段。本阶段只做
代码接线、纯合成 CPU 轻量验证与文档登记；**未训练、未 validation、未 inference、未真实评估**。

### 新 Trainer 的组成与唯一变量

- 类名 `nnUNetTrainerPICAI_DiceCE_PositiveSampling_NoFFT`，CLI 别名 `dicece_positive_sampling`；
- 定义方式：`class nnUNetTrainerPICAI_DiceCE_PositiveSampling_NoFFT(PositiveCaseSamplingMixin, nnUNetTrainerPICAI_DiceCE_NoFFT)`，
  MRO = 自身 → `PositiveCaseSamplingMixin` → `nnUNetTrainerPICAI_DiceCE_NoFFT` →
  `NoFFTAugmentationMixin` → `nnUNetTrainer` → `object`；
- 相对 A（`nnUNetTrainerPICAI_FLCE_PositiveSampling_NoFFT`）**唯一变量**：损失
  （`0.5*Focal(gamma=2)+0.5*CE` → 原生 Dice+CE）；采样/增强/网络/优化器/调度/验证全部同源；
- 相对 `nnUNetTrainerPICAI_DiceCE_NoFFT` **唯一变量**：训练集病例/patch 采样（原生 → 阳性病例采样）；
- 复用组件（未复制任何 nnU-Net 实现）：原生 `nnUNetTrainer._build_loss`
  （`DC_and_CE_loss` + `MemoryEfficientSoftDiceLoss` + 原生 `DeepSupervisionWrapper`）、
  `PositiveCaseSamplingMixin.get_dataloaders`（训练 loader `PositiveCaseDataLoader`，
  验证 loader 保持原生 `nnUNetDataLoader`）、`NoFFTAugmentationMixin.get_training_transforms`、
  原生 `nnUNetTrainer.build_network_architecture`（plans 驱动的原生 `PlainConvUNet`，3 通道）；
- 唯一新增的运行期行为：覆盖 `initialize()`，在原生初始化之后追加**一行可审计日志**
  （实际 loss / network / 输入输出通道数），不改变训练循环与任何训练机制。

### 修改的文件（未新建任何文件）

- `src/zonal_reliability_fusion/nnunet/trainers.py`：新增上述类 + 注册进 `PROJECT_TRAINERS`
  （10 → 11 项）+ 模块 docstring 同步；
- `scripts/train/train_nnunet.py`：新增 variant `dicece_positive_sampling`（精确映射到该类，
  仍为进程内直接映射，不依赖递归扫描）+ docstring / `--help` 文本同步；
- `scripts/inference/predict_nnunet.py`：`PROJECT_TRAINER_NAMES` 补入新类名（现有测试强制该
  常量与 `PROJECT_TRAINERS` 集合相等，避免未来该 Trainer 的 checkpoint 无法被官方预测器解析）；
- 测试：`tests/unit/test_nnunet_trainers.py`、`tests/unit/test_positive_case_sampling.py`、
  `tests/unit/test_predict_entry.py`（同步既有全量清单断言 + 新增专项测试）；
- `README.md`：variant 表 / 命令清单 / 目录结构同步，并新增该基线的说明、未来训练命令与启动判据。

### 新增/修改的测试（纯合成 CPU，不读真实数据、不用 CUDA）

新增专项测试覆盖：精确 MRO；hook 来源逐个核对（`_build_loss` 为原生函数、`get_dataloaders`
来自采样 mixin、`get_training_transforms` 来自 NoFFT mixin、`build_network_architecture` 为
原生函数、`configure_optimizers` / `train_step` / `validation_step` / 验证 transform 仍为原生）；
与 A 的单变量对比（仅损失函数不同）；与 `DiceCE_NoFFT` 的单变量对比（仅 dataloader 接线不同）；
损失身份与语义（`DC_and_CE_loss` + `MemoryEfficientSoftDiceLoss` + `RobustCrossEntropyLoss`，
`weight_ce=weight_dice=1`、`do_bg=False`、无 focal gamma / alpha / 类别权重，deep-supervision
权重与 `DiceCE_NoFFT` 逐值一致）；同一合成 logits/target 下损失数值与 `DiceCE_NoFFT` 逐值相同
且 backward 梯度有限；网络与 A、`DiceCE_NoFFT` 逐参数相同（`state_dict` 键集合与取值全等，
无 gate / backbone / stems / projection）；CPU 合成前向 → 原生 DiceCE → backward 全部梯度有限；
`initialize` 审计日志调用原生实现恰好一次并打印实际对象类型。采样侧复用现有合成 metadata 测试
（真实 MRO 下训练 loader 为 `PositiveCaseDataLoader`、每批一个阳性槽位且 target 含前景、验证
loader 为原生 `nnUNetDataLoader` 且只用验证 identifiers、metadata 缺失/无阳性病例 fail-closed）。

### 实际通过的轻量检查

`python -m compileall -q src scripts tests`、`python -m pytest -q`（**346 passed**）、
`python -m ruff check src scripts tests`、`python -m ruff format --check src scripts tests`、
`git diff --check`、`python scripts/train/train_nnunet.py --help`（exit 0，choices 含新 variant）
全部通过。未运行训练入口的真实运行。

### 明确声明

训练、validation、推理、真实数据评估均**未运行**；未读取任何真实 NIfTI/预处理数组；未修改
`data/`、`workdir/`、`outputs/`、`third_party/`、任何 checkpoint，也未触碰正在训练的 B 进程；
目标输出目录在本轮前后均不存在。DNA 0.3.1 与固定 nnU-Net 2.6.2 的依赖冲突如实保留（未解决）。

---

## 2026-09-28 — 病灶实例级评价协议冻结与实现（在 B 模型结果可见之前）

目的：在 `feature_no_gate_positive_sampling`（B）训练完成、其结果对本代理可见之前，把 A
（`positive_sampling`）与 B 的病灶实例级比较协议**先冻结再实现**，避免评价定义被结果影响。
本阶段未运行任何训练、validation、inference 或真实数据评测。

### 冻结的协议（同步写入模块 docstring、README 与 Research Plan §7.2）

- 实例 = 3D 连通域，`CONNECTIVITY = 1`（6-邻域 / face connectivity）；
- 候选匹配 = intersection ≥ 1 voxel（主指标因此命名为 `lesion_sensitivity_any_overlap`）；
- 一对一词典序 matching：maximum cardinality → maximum total intersection voxels →
  (reference id, prediction id) 升序确定性 tie-break；**不**以 Dice 为 matching 目标
  （matched-lesion Dice 是报告终点之一，先最大化 Dice 会产生 metric-optimizing-the-metric，
  Dice 只用于匹配后的质量评价）；
- 未匹配 reference lesion 不从 sensitivity 分母删除；阴性病例不进分母、其预测团块仍算假阳；
- 探索性大小分层 `< 500 / 500–1000 / > 1000 mm³`（不是临床风险类别，边界严格）；
- 不使用任何预测后处理（最小团块 / 最大团块 / 形态学 / 阈值优化）。

### 实现（`scripts/evaluate_segmentation.py`，SCHEMA_VERSION 1.0 → 1.1）

- 新增 `_lesion_size_stratum`；`_min_cost_flow_unit_matching`（unit-capacity 二部图 SSP
  最小费用流，同时给出最大基数与最大总权重）、`_max_weight_matching_exact_size`；
  `_match_lesion_instances`（id 升序贪心 + 精确可行性判据，结果只由交点权重与组件 id 决定，
  与 dict/set/随机/文件/模型/病例顺序无关）；`_validate_lesion_matching`（一对一、
  intersection ≥ 1、匹配数上界，违反即 `EvaluationError`）；`lesion_instance_metrics`
  （单病例，含可审计的 reference / prediction lesion records）；`_aggregate_lesion_instance_metrics`
  （Σ 逐病例 = 全局的计数与分布聚合，不重新匹配）。
- 新字段只进入 full 模式：per-case entry 增加 `lesion_instance_metrics`（逐实例记录，
  含 component id、voxels、volume_mm3、size_stratum、matched、intersection、matched_dice）；
  全局块报告 `lesion_sensitivity_any_overlap`、`matched_reference_fraction`、
  `small_lesion_sensitivity_any_overlap`、`matched_lesion_dice`（count/mean/median/q1/q3）、
  三层 `size_strata`、predicted / matched / unmatched predicted lesion 计数与 matching 协议
  声明。summary 模式只增加 `lesion_instance_metrics: null` 占位；既有 `component_analysis`、
  病例级 `size_strata`（研究者给定阈值）与全部旧字段语义不变。
- fail-closed 延续：实例级计算位于 full 模式逐例异常边界内，任一例失败即整轮失败、不发布 JSON。

### 测试（`tests/unit/test_evaluate_segmentation.py`，纯合成 CPU）

新增 19 项：单病灶 perfect/partial、完全漏分、空阴性、阴性假阳不污染 sensitivity 分母、
多病灶 perfect、一对多、多对一、maximum cardinality 优先（2 对小重叠优于 1 对大重叠）、
次级总 intersection 目标、精确平局 id tie-break 可重复且与字典插入顺序无关、贪心可改配以达
最大基数、6-邻域（角/边接触属于不同 lesion）、大小边界（499.999/500/1000/1000.001 与整数组
路径）、各向异性 spacing 体积（0.75 mm³/体素）、三层分层指标、JSON 聚合一致性
（Σ 逐病例 = 全局）、匹配一致性校验拒绝非法配对。文件全量 95 项通过；`ruff check`、
`ruff format --check`、`compileall`、`git diff --check` 通过。

### 本阶段未运行

未运行训练、validation、inference 或真实数据 full evaluation；未读取 B 的中间训练结果；
未修改 trainer / network / sampling / third_party / 数据准备 / checkpoint 格式。真实数据上的
A/B full evaluation 仍由研究者用冻结后的同一命令在两侧产物齐备后运行。

---

## 2026-09-24 — 审计工具 v2：区分「原始 seg 逐值相同」与「有效标签相同」

v1 真实审计 FAIL 的直接原因：1487 例预处理 seg 含 `-1`，v1 误以 dataset.json 的 `{0, 1}` 为唯一
合法取值而判非法。从固定版本 nnU-Net v2.6.2 源码核对的事实（只读）：

- `-1` 的来源：`preprocessing/cropping/cropping.py:19` 定义 `crop_to_nonzero(..., nonzero_label=-1)`，
  `:36` 执行 `seg[(seg == 0) & (~nonzero_mask)] = nonzero_label`（nonzero 裁剪框外写 -1）；
  `preprocessing/preprocessors/default_preprocessor.py:66` 以默认参数调用。
- `-1 → 0` 的映射：`training/nnUNetTrainer/nnUNetTrainer.py:800`（get_training_transforms）与
  `:855`（get_validation_transforms）都追加 `RemoveLabelTansform(-1, 0)`
  （batchgeneratorsv2 `RemoveLabelTansform._apply_to_segmentation` 将 `label_value` 改写为
  `set_to`），即进入损失计算前 `-1` 已变为 0。

### 修改（`scripts/data/audit_dataset605_606_mri_equivalence.py`）

- 合法原始 seg 取值固定为 `{-1, 0, 1}`（模块常量 `ALLOWED_RAW_SEG_LABELS`，附源码出处）；
  `dataset.json` 声明的任务标签必须落在该集合内（`labels_within_allowed_raw`），否则 fail-closed。
- 检查顺序固定：shape → dtype → 有限性 → 取值合法性 →（全部通过后）才做整数转换与逐值比较；
  NaN/Inf/非法值产生结构化失败条目（`seg_non_finite_values` / `unexpected_label_values`），
  绝不异常崩溃、绝不截断成合法值。取值合法性在 float64 上判定（规避无符号/负数表示陷阱），
  报告的取值从原始数组提取。
- 逐例同时报告四件事并明确区分语义：`raw_seg_equal`（原始逐值相同）、`raw_unequal_voxels` 与
  `raw_diff_pairs`（如 `"-1→0": 11`，只存汇总不存大数组）、`effective_labels_equal`（-1 映射为 0
  后逐值相同）、`foreground_equal`（`(seg == 1)` 逐值相同）。逐例摘要落
  `per_case_seg_summary`。
- 分类与计数拆分，消除「status=PASS 但 n_cases_mismatch>0」的矛盾：
  - `n_cases_effective_input_mismatch`（FAIL 驱动：MRI 任一体素差异、有效标签差异（含 0↔1、
    -1↔1）、非法取值、形状/dtype/非有限值）→ `mismatched_cases`；
  - `n_cases_raw_label_difference`（信息性：原始 seg 有差异但有效标签相同，即纯 `-1↔0`）→
    `raw_label_differences`；
  - `n_cases_raw_identical` / `n_cases_effective_equal` / `n_cases_foreground_equal`、
    `effective_labels_equal`、`mri_exact_equal_all`；控制台同时打印有效不一致数与原始差异数。
  - 删除歧义字段 `n_cases_mismatch`、`n_cases_exact_equal`、`label_equal`。
- PASS 判据：完整检查全部共同病例（无 `--max-cases` 子集、无读取失败）＋ 病例集合与 split 一致
  ＋ dataset.json 合法且两数据集一致 ＋ 前三个 MRI 通道逐数组完全相同 ＋ 所有 seg 合法且有效标签
  逐值相同。原始 `-1↔0` 差异不计入失败。默认输出路径改为
  `outputs/reports/dataset605_606_mri_equivalence_audit_v2.json`；v1 报告按原样保留作追溯。
- PZ/TZ：不参与前三 MRI 通道的逐值相等判定、不影响 status/PASS；`min/max` 改为 nan 感知并新增
  `contains_non_finite`，在报告 note 与控制台中明确标注「仅信息性」。
- v1 FAIL 报告中 1487 例 `unexpected_label_values`（labels `[-1]`）在 v2 语义下全部合法且原始逐值
  相同；仅 10879_1000895（1 体素）与 10980_1000999（11 体素）为 `-1→0` 原始差异 → 信息性，
  有效标签与 `(seg == 1)` 均逐值相同（用户已在真实数据上只读诊断核实）。

### 测试（`tests/unit/test_dataset605_606_mri_equivalence_audit.py`，纯合成 CPU）

38 项，新增/改写覆盖：两侧 seg 完全相同且都含 -1 → PASS；同一位置 -1↔0（含反向 0→-1）→
PASS 且信息性记录（`raw_diff_pairs` 计数准确）；一例 1 个、一例 11 个 -1→0 → PASS 且计数准确；
0↔1、-1↔1 → FAIL；标签 2、NaN、Inf、seg dtype/shape 差异、uint8 回绕值 255 → 结构化 FAIL；
非法值存在时不再做逐值比较；`dataset.json` labels 越界 → FAIL；`--max-cases` 子集与缺文件/
split 不同 → 永不 PASS；CLI 端到端（nnU-Net `save_case` 写真实 `.b2nd`/`.pkl`、真实 JSON 顶层
结构：splits 数组 + dataset 对象）覆盖 PASS / FAIL / 信息性差异 / 非 dataset.json 崩溃路径。
`pytest -q` 全仓 313 passed；`ruff check`、`ruff format --check`、`compileall`、`git diff --check`、
`--help` 全部通过。未运行真实数据审计与任何训练；未修改 third_party、data、workdir、outputs 中
任何现有文件；Research_Plan、Findings、Training_Log 与实验记录文档未改动。

---

## 2026-09-24 — Dataset605/606 MRI 数组审计工具 + Anatomy Gate 启动前校验与文档同步

为 `anatomy_gate_positive_sampling`（RQ2 公平匹配臂）的正式启动做前置准备：**只新增一个只读审计
工具与其测试、补齐启动前 dry-run 测试、同步文档**；未修改模型数学形式、损失、采样算法、
optimizer / LR / epoch / 增强 / patch / batch / deep supervision，未新增 research question，
未启动任何 1000 epoch 训练。

### 新增

- `scripts/data/audit_dataset605_606_mri_equivalence.py`：Dataset605 与 Dataset606 **预处理后**
  前三个 MRI 通道（`0000`/`0001`/`0002` = T2W/ADC/HBV）的逐数组一致性审计。只读、fail-closed，
  通过 nnU-Net 自己的 `infer_dataset_class` + `load_case` 读取（与训练看到的数组完全一致，不自行
  解析 `.b2nd`）。逐例检查：array shape / 通道数 / dtype / NaN·Inf / 三个 MRI 通道的
  `np.array_equal` 与浮点差值统计（max、mean、unequal voxels、fraction）/ lesion label
  （shape、dtype、唯一值、逐值相等）；并审计病例集合、`splits_final.json`（fold 数、逐 fold
  train/val 的集合与顺序、交叉污染）与 `dataset.json`（前三个通道名、labels 定义）。
  **PZ/TZ 不参与数值一致性判定**（只记录取值范围作为信息性诊断）。status 取值
  `MRI_ARRAY_AUDIT_PASS` / `MRI_ARRAY_AUDIT_FAIL` / `MRI_ARRAY_AUDIT_PARTIAL`（用了
  `--max-cases` 的子集运行**永不判 PASS**），报告落
  `outputs/reports/dataset605_606_mri_equivalence_audit.json`，已存在时默认拒绝覆盖。
  不修复数据、不重建 Dataset606、不重新 preprocessing（`inputs_modified: false` 写入报告）。
  自带 tqdm 进度与结束汇总（成功/失败/耗时/路径）。
- `tests/unit/test_dataset605_606_mri_equivalence_audit.py`：21 项纯合成 CPU 测试（`tmp_path` +
  内存数组 + 占位文件，不读真实医学数据），覆盖完全一致→PASS、单 voxel 不同→FAIL、shape /
  通道数 / dtype / 标签越界 / NaN·Inf →FAIL、缺病例→FAIL、缺预处理文件→FAIL、split 集合不同→FAIL
  而**仅顺序不同不判 FAIL**（记录顺序差异）、fold 数不同→FAIL、lesion label 不同→FAIL、
  PZ/TZ 完全不参与前三 MRI 判定、报告可 JSON 序列化且含 mismatch 明细、**输入文件与输入数组
  逐字节/逐值未被修改**、`--max-cases` 子集恒为 PARTIAL、CLI 在缺 `nnUNet_preprocessed` 时
  fail-closed、报告已存在时拒绝覆盖。
- `tests/unit/test_nnunet_trainers.py` 新增 `test_combined_anatomy_gate_synthetic_dry_run`：
  `anatomy_gate_positive_sampling` 的启动前 dry-run —— 5 通道网络在 `deep_supervision=True` 下
  forward 出 DS 列表（合成 3-stage → 2 个分辨率）→ 项目 FLCE 算损失 → `backward` 成功，且 gate 与
  backbone 都收到有限梯度。

### 修复（审计工具自身，由测试与真实接线验证发现）

- 审计脚本原先只用「case 集合 / split / label / 各 MRI 通道聚合」推导 status，**会漏掉逐例结构性
  失配**（通道数、空间形状、dtype、NaN/Inf、越界标签等只进入 `mismatched_cases` 而未进入
  `problems`）→ 会产生 fail-open 的 `PASS`。现改为：只要 `n_cases_checked − n_cases_exact_equal > 0`
  即记 FAIL 并写出结构性不一致条目；该缺陷由 `test_channel_count_difference_fails` 等测试暴露。
- 元数据检查原先假设 `dataset.json` 的通道键是 `"0"/"1"/"2"`，但**真实预处理 dataset.json 用的是
  `"0000"/"0001"/...`** ⇒ 会把键格式差异误判成「前三个通道名不匹配」而使审计 FAIL。现按通道序号
  排序后取前三个，兼容两种键格式；已用真实 `workdir/nnUNet_preprocessed/*/dataset.json` 复验通过
  （`mri_channel_names_equal=True`，`T2W/ADC/HBV`）。新增
  `test_zero_padded_channel_keys_pass` 守护该格式。该缺陷由**不读数组的真实路径接线验证**发现。

### 文档同步

- `docs/Training_Log.md`：`image_gate_positive_sampling` 从「未运行」改为「已完成」，补齐训练 /
  validation 时间、关键指标与产物路径；登记审计工具「tooling ready, real-data audit not yet
  executed」。
- `docs/Findings.md`：新增 §3.10「RQ1 公平匹配比较」（配对 delta、CI95、改善/平局/变差计数、
  新增/丢失重叠、阴性假阳与假阳体素、复现命令），§1 / §2 / §3.4 / §3.5 / §3.6 / §4 / §5.2 / §5.3 /
  §6 / §7 同步为四个 run 的口径；结论按证据克制表述（无明确增量、工作点更保守），明确禁止
  「gate 无效 / 证明 gate 没有作用」等表述。
- `README.md`：variant 状态与输出目录、命令注释、一致性边界与新增审计命令段落同步。
- `docs/Findings.md` §3.2：删除小病灶分箱表里 `(< 0.75 mL)` 等**不可换算**的物理体积标注，改为
  纯体素口径 + 单位限制说明（与研究者在 `docs/experiments/image_gate_positive_sampling.md` 中
  对同一问题的更正一致：逐例导出几何不同，体素≠统一 mm³）。
- `docs/Research_Plan.md` **未修改**（本轮实现与其表述不冲突）。

### 未执行 / 未改动

- **真实数据审计未执行**（需研究者本人运行，见 README「Dataset605 ↔ Dataset606 前三 MRI 通道
  逐数组审计」）；`anatomy_gate_positive_sampling` 正式训练未启动。
- 未删除、未覆盖任何既有产物；Prostate158 管线保持不动、未扩大。

---

为已完成的 Dataset605 模型增加 Prostate158 跨域外测管线。**本轮只实现代码与纯合成测试，
未运行任何真实数据 audit / prepare / 推理 / 评估，未触碰正在训练的进程。**

### 新增脚本

- `scripts/data/prepare_prostate158_external.py`（CSV 驱动，两个子命令）：
  - `audit` 只读：逐例核对 t2/adc/dwi 与主参考 `adc_tumor_reader1` 的存在/可读性、size / spacing /
    origin / direction / 物理 FOV、标签 0/1 取值与阴阳性；分类为 `linkable` / `grid_mismatch`
    （附 `resample_candidate` 与阻塞原因）/ `failed`。阴性必须由字段非空、可读、合法且**经验证
    为空**的掩膜确定（如 `empty.nii.gz`）；CSV 空字段 / 缺文件 / 读取失败一律 fail-closed。
  - `prepare` 在**独立新目录**（暂存目录构建后原子改名）生成 `images/<P158_{queue}_{ID}>_000x.nii.gz`：
    默认仅相对软链接（创建前后校验 realpath 仍在允许根内；标签不进输入目录，只写
    `external_input_manifest.json`）；`--allow-resample` 默认关闭，开启后仅对同一物理坐标系、
    方向正交归一且 FOV 覆盖 T2 的 adc/dwi 用 SimpleITK 线性插值派生到 T2 网格（记录源/目标几何，
    不做配准、不动原始文件）。派生 ID `P158_{test|train|valid}_{ID:03d}` 稳定唯一可反查 CSV 行；
    绝对路径 / `..` 越界 / 重复 ID 拒绝；输出目录已存在非空拒绝覆盖；带 tqdm 与结构化结束汇总。
  - 通道映射 `_0000=t2 / _0001=adc / _0002=dwi`，审计报告与清单都写入跨域声明
    （Prostate158 DWI 不等同于训练用 HBV）。
- `scripts/evaluate_external_segmentation.py`：独立外测评估入口，**不依赖** `validation/summary.json`。
  读取 prepare 清单（含 `manifest_id`）、原始参考标签路径与独立预测目录；复用
  `scripts/evaluate_segmentation.py` 的**纯函数**（importlib 加载，未改其 CLI / 指标 / 测试行为）。
  - 指标口径与内部评估一致：阳性 macro Dice（完全漏分记 0）、median/micro Dice、体素召回、
    仅阳性 precision、含阴性假阳的 overall precision、完全漏分（含空预测/错位拆分）、
    阴性假阳病例数与体积分布；队列无阴性时这些指标记 `not_applicable`（null）而非 0。
  - 预测与参考异网格时：仅在同一物理坐标系、方向正交归一且预测 FOV 覆盖参考网格时，把**派生
    预测掩膜**最近邻映射到参考网格（原始参考从不重采样，不做数组索引强比），对齐事实逐例记录；
    不可信即非零退出。多模型比较强制同清单版本 / 同读者 / 同病例集合 / 同参考路径，否则拒绝；
    `--reader reader2` 只在有可核对 reader2 的子集上出敏感性结果，缺预测不回退 reader1。
- `tests/unit/test_prostate158_external.py`：20 个纯合成 CPU 用例（tmp_path + 合成 NIfTI），
  覆盖通道顺序与软链接目标、稳定 ID、越界/重复/缺通道/缺主标签、空字段阴性 fail-closed、
  size 相同但 origin/direction 不同不算对齐、可验证重采样与不可对齐 fail-closed、完美/部分/漏分/
  错位/阴性假阳指标、异网格最近邻评估对齐、读者不一致与集合不一致拒绝比较、输出拒覆盖、
  失败不留半成品、原始输入与参考标签不被修改。
- README 新增「Prostate158 独立外测」小节：audit → prepare（默认软链接）→ 现有 predict 入口
  （固定模型/checkpoint）→ 独立外测评估的可复制命令，三队列分开，附成功判据与停止条件。

---

## 2026-09-23 — 实现 Research Plan §8.10 的浅层序列特异特征融合候选（三个未训练条件）

把计划 §8.10 的 `Feature no gate` / `Feature image gate` / `Feature anatomy gate` 落成三个新的、
互相匹配的模型条件。**只做实现与合成测试，未运行任何训练**。

### 新增网络（`src/zonal_reliability_fusion/nnunet/networks.py`）

- `ShallowSequenceStem`：单个 MRI 序列的浅层 stem，两层 3×3×3 `Conv3d`（stride 1、padding 1，
  **空间尺寸逐值保持**），每层后接 `InstanceNorm3d(affine=True)` 与 LeakyReLU；只接受单序列
  `[B,1,...]` 输入，输出 `[B,C_s,...]`。
- `FEATURE_STEM_CHANNELS = 8`（`C_s`）。取值是**先评估后取保守值**：patch `16×320×320`、
  batch size 2、fp32 时，`C_s = 8` 的 stem 顶层激活约 0.63 GB、计入 InstanceNorm/autograd 中间量
  约 1.7–2.0 GB；`C_s = 16` 约为其两倍；原生 `PlainConvUNet` stage 0 的对照值约 0.84 GB。
  **未修改** `nnUNetPlans.json`（骨干仍完全由 plans 构建）。
- `FeatureFusionNNUNet`：三个**参数不共享**的 stem（`nn.ModuleList`）→ 拼接（`3*C_s`，按计划
  采用拼接而非逐通道求和）→ 1×1×1 投影到**恰好 3 通道** → 原生 backbone。可选 feature gate 读
  **拼接后的 stem 特征**并输出恰好 3 个 logit；`S_m` 通过 `scales[:, m:m+1]` 广播在 `H_m` 的
  全部 `C_s` 个通道上共享。`Z` 只进入 gate（anatomy 条件下先 `clamp(0,1)`），**不进入**任何 stem、
  投影层或 backbone。`use_gate=False` 时 `self.gate is None`，`state_dict` 不含任何 `gate.*` 键。
  暴露 `decoder`（代理）与 `compute_conv_feature_map_size`（代理）。
- `feature_gate_parameter_delta()`：把 image 与 anatomy 的 gate 参数差（`hidden × prior = 16`）
  显式记录下来。**没有**采用零值占位通道去把两者参数量凑成相等 —— 那样会让"唯一差异是 PZ/TZ"
  更难核查。结论中不得声称两者参数量严格相同。

### 新增 Trainer（`src/zonal_reliability_fusion/nnunet/trainers.py`）

`_FeatureFusionTrainerBase(nnUNetTrainerPICAI_FLCE_PositiveSampling_NoFFT)` 只提供一个共享的静态
网络构建辅助；三个子类只覆盖 `build_network_architecture`：

| variant | Trainer | 数据集 | 通道 |
|---|---|---|---|
| `feature_no_gate_positive_sampling` | `nnUNetTrainerPICAI_FeatureNoGate_PositiveSampling_NoFFT` | 605 | 3 |
| `feature_image_gate_positive_sampling` | `nnUNetTrainerPICAI_FeatureImageGate_PositiveSampling_NoFFT` | 605 | 3 |
| `feature_anatomy_gate_positive_sampling` | `nnUNetTrainerPICAI_FeatureAnatomyGate_PositiveSampling_NoFFT` | 606 | 5 |

- MRO 精确为 `子类 -> _FeatureFusionTrainerBase -> FLCE_PositiveSampling_NoFFT ->
  PositiveCaseSamplingMixin -> FLCE_NoFFT -> NoFFTAugmentationMixin ->
  PICAIFocalCrossEntropyLossMixin -> nnUNetTrainer -> object`，因此自动复用 PI-CAI FLCE
  损失、NoFFT 修复、阳性病例采样，以及 nnU-Net 原生的 optimizer / PolyLR / 1000 epoch /
  deep supervision / checkpoint / validation / 滑窗推理，**不重实现训练循环**。
- 三个新类**不继承** 输入级 `ImageGate` / `AnatomyGate` 及其 `_positive_sampling` 组合、
  也不继承 `DiceCE` 分支，避免混淆两个表征层级。
- feature anatomy 的 `get_training_transforms` 直接**委托**给既有
  `nnUNetTrainerPICAI_AnatomyGate.get_training_transforms`（同一段代码，未修改、未复制其定义），
  因此强度增强仍只作用于前 3 个 MRI 通道、空间/镜像增强仍同步作用于全部通道。未改动
  `nnUNetTrainerPICAI_AnatomyGate` 本身。
- `PROJECT_TRAINERS`、`scripts/train/train_nnunet.py` 的 `VARIANT_TO_TRAINER`（七个 → 十个）与
  `scripts/inference/predict_nnunet.py` 的 `PROJECT_TRAINER_NAMES` 已同步。输出目录由类名自然隔离，
  三个新目录与既有目录不重合，不加载、不覆盖任何旧 checkpoint。

### 零初始化的边界（写入 README 与测试）

feature gate 末层零初始化 ⇒ 初始 `S ≡ 1` ⇒ feature gate 网络在**共享相同 stem/投影/backbone 权重**
时与 `feature_no_gate` **逐值一致**。但这**不**等价于原生输入级 nnU-Net：`E_m` 与 `P_ψ` 本身仍然
改变进入 backbone 的输入分布；测试里即使把 backbone 权重逐值对齐，两者输出也不相同。

### 测试（纯合成 CPU，不读真实数据、不写 outputs）

- `tests/unit/test_nnunet_networks.py` 新增浅层特征融合一节（29 项）：三个 stem 参数不共享（不同
  tensor 对象、不同存储、权重不等）、stem 保持空间尺寸（含奇数尺寸）、两层 3×3×3/stride 1、
  骨干只接收 3 通道（forward pre-hook 捕获）、零初始化等价（image 与 anatomy 两种）、
  与原生 nnU-Net 不等价（backbone 权重逐值对齐后仍不同）、`W` 非负且和为 1、`S` 非负且和为 3 且
  `S = 3W`、每个序列只有一个共享尺度场（逐通道比值恒等）、forward 与手工重建逐值一致、
  image gate 响应 MRI 变化、固定 MRI 下改变空间结构 PZ/TZ 只影响 gate 与输出而 stem 逐值不变、
  PZ/TZ clamp（越界与显式 clamp 逐值等价）、no-gate 无 gate 且调用即报错、通道严格性、
  deep supervision 格式、`decoder` 代理切换、strict state_dict 往返、三个变体互不兼容、
  gate 参数差 = 16 且其余模块形状一致、不复制 U-Net。
- `tests/unit/test_nnunet_trainers.py`：`_ALL_PROJECT_TRAINERS` 扩到 10，类名互异断言 7 → 10；
  三个新 Trainer 的损失仍解析到 PI-CAI FLCE（不是 Dice+CE）、`positive_cases_per_batch == 1`、
  不继承输入级 gate / DiceCE 分支、通道契约、inference builder 输出 2 类张量、anatomy 的 PZ/TZ
  只进 gate、三个网络互不 strict 互认、`get_dataloaders` 必须来自采样 mixin；NoFFT 与 MRI-only
  强度增强参数化用例覆盖新 Trainer；train 入口 variant 集合 7 → 10。
- `tests/unit/test_positive_case_sampling.py`：把三个 feature Trainer 加入
  `test_combined_trainers_dataloaders_via_real_mro` 的参数化列表，用**真实 MRO** 验证训练 loader 是
  `PositiveCaseDataLoader`（继承原生 `nnUNetDataLoader`）、验证 loader 仍是原生
  `nnUNetDataLoader` 本类，且验证 batch 只用验证 identifiers（无泄漏）；同步扩充 variant 集合期望值。
- `tests/unit/test_predict_entry.py`：同步扩充 `PROJECT_TRAINER_NAMES` / 解析期望值。
- 全量：**254 passed / 0 failed**。

### 文档（只改既有文件）

- `README.md`：新增「浅层特征融合」小节（结构、参数量表、解析显存估算、零初始化的边界、
  gate 参数差、比较关系与边界），variant 表与输出目录补三个新条件，训练命令补三条（注明尚未训练）。
- `docs/Training_Log.md`：登记三个新 variant 为**未运行**；并按 `validation/summary.json` 与训练
  日志**修正事实** —— `positive_sampling` 由「训练中」改为「已完成」（2026-09-22 07:16 → 22:31 UTC，
  1000 epoch 完成；22:42 UTC validation 完成；Mean Validation Dice = 0.207834）。
- `docs/Research_Plan.md` **未改**（实施过程不进计划）。

### 未运行

**未运行任何训练、validation、推理或真实数据评测**；未读取真实医学 NIfTI；未修改 `data/`、
`workdir/`、`outputs/`、`third_party/` 与任何既有 checkpoint。

---

## 2026-09-22 — Research Plan 一致性修补：NSD 物理表面、PZ/TZ 术语、公平匹配 gate 组合与门控读取接口

按 `docs/Research_Plan.md` 对代码做的四项一致性修补；不改变任何既有 Trainer 的类名与行为、
不改变已物化的数据、不运行任何训练/推理/评测。

### NSD 改为 surfel-area weighted 物理表面度量（`scripts/evaluate_segmentation.py`）

- 删除旧的 `_surface`（6-邻域表面体素）与 `nsd_surface_voxel`（按表面体素计数），改为
  `nsd_surface_area`：用 DeepMind `surface-distance` 0.1 的官方函数
  `compute_surface_distances(mask_gt, mask_pred, spacing_mm)` +
  `compute_surface_dice_at_tolerance(surface_distances, tolerance_mm)`，按物理表面测度
  μ（surfel 面积，mm²）加权；`spacing_mm` 仍是数组轴序 (z, y, x)（由现有 `_spacing_zyx` 从
  SimpleITK 的 (x, y, z) 反转而来）。
- 空掩膜口径不变：GT 非空、预测为空 → NSD = 0、HD95/ASSD = null；双空 → NSD = null（不纳入
  阳性表面统计）；GT 非空、预测非空但无重叠 → 正常计算。`--nsd-tolerance-mm` 仍必须显式提供且
  为正的有限数。
- 新增 `_surface_distance_metrics()` 延迟导入；缺包时抛 `EvaluationError`（fail-closed），
  `full` 模式在读取任何 NIfTI **之前**先做导入检查，不静默退回旧算法。
- full 报告的 `nsd_method` 由「surface-voxel based…」改为
  `surfel-area weighted physical-surface NSD (surface-distance 0.1; spacing in mm)`。
- HD95 / ASSD / Dice / Recall / Precision / 漏分与体积指标定义未动。
- 测试（`tests/unit/test_evaluate_segmentation.py`）：重写容差测试为「单体素相邻对沿三轴各自的
  真实 spacing 给出 0.5/1.0 跳变」的解析构造；新增 surfel-area 与 surface-voxel counting 的
  可区分性测试（5×5×1 平板 + 远离孤立体素：面积加权 ≈ 0.9416 vs 计数 0.9，断言实现等于前者、
  不等于后者）；新增缺包 fail-closed 测试与 `nsd_method` 文本断言。

### PZ/TZ description 术语修正（`scripts/data/prepare_picai_nnunet.py`）

- Dataset606 的 `dataset.json` description 由过强的 `fractional occupancy [0,1]` 改为
  `algorithm-derived PZ/TZ zonal membership channels ([0,1], not calibrated probabilities or
  geometric occupancy; noNorm; prior_source=zonal_<source>)`；保留 `[0,1]`、`noNorm` 与
  `prior_source` 审计字段。仅影响**以后重新生成**时的代码行为；已物化的 Dataset606 未被改写。

### 新增两个与 positive_sampling 公平匹配的 gate Trainer（`trainers.py` 等）

- 新增 `nnUNetTrainerPICAI_ImageGate_PositiveSampling_NoFFT` 与
  `nnUNetTrainerPICAI_AnatomyGate_PositiveSampling_NoFFT`，MRO 分别为
  `(cls, PositiveCaseSamplingMixin, ImageGate|AnatomyGate, _GatedTrainerBase,
  nnUNetTrainerPICAI_FLCE_NoFFT, NoFFTAugmentationMixin, PICAIFocalCrossEntropyLossMixin,
  nnUNetTrainer, object)`。因此 loss 仍为 PI-CAI FLCE、NoFFT 修复与 gate 结构不变
  （image=3 通道；anatomy=5 通道，PZ/TZ 只进 gate 且进入前 clamp、强度增强仍只作用于 MRI），
  训练 loader 为 `PositiveCaseDataLoader`（`positive_cases_per_batch=1`），验证 loader 仍为原生
  `nnUNetDataLoader`；optimizer / PolyLR / epoch / deep supervision / checkpoint / validation /
  inference 全部继续由 nnU-Net 原生实现。旧五个 Trainer 类名与行为完全未变。
- 入口同步：`scripts/train/train_nnunet.py` 增加 variant `image_gate_positive_sampling`（605）与
  `anatomy_gate_positive_sampling`（606）及其帮助文本；`predict_nnunet.py` 的
  `PROJECT_TRAINER_NAMES` 与 `PROJECT_TRAINERS` 注册表加入两个新类名。
- 两个新类的 docstring（以及模块头）写明公平比较边界：RQ2 的**预期主要差异**是门控条件与
  数据集（605 vs 606），在前三个 MRI 通道逐数组一致性审计完成前不作「唯一差异是 PZ/TZ」
  声明；旧 `baseline` ↔ 旧 `image_gate` 是原生采样条件下的受控 RQ1 对照，不能与阳性采样分支
  混合比较。

### 门控读取接口与术语（`networks.py`）

- `SpatialModalityReliabilityGate.compute_weights(x)` 返回 W = softmax(logits, dim=1)（非负、
  通道和为 1）；`forward` 返回 S = 3W（通道和为 3），数学行为与修补前逐值一致。
- `GatedNNUNet.compute_gate_scales(x)` 复用 `_gate_input`（通道检查 + anatomy prior clamp 不可
  绕过）；`forward` 调用它后 `mri * scales` 送入 backbone。不缓存 batch 大张量、不改变任何
  state_dict key。
- Python 类名/包名中的 `Reliability` 保持不动（checkpoint/导入兼容），在模块与类 docstring、
  `README.md` 中明确其为**历史内部标识**，当前解释为 learned sequence preference / scale。

### 测试

- `tests/unit/test_nnunet_networks.py`：新增 W 非负且和为 1、S 非负且和为 3、零初始化 W=1/3、
  `compute_gate_scales` 的 prior clamp（空间变化越界图案；注意 gate 内 InstanceNorm 会消除
  空间常数 prior）、forward 等于 `backbone(mri * S)` 且非恒等。
- `tests/unit/test_nnunet_trainers.py`：新增两个组合 Trainer 的精确 MRO、方法复用（builder/
  transforms/dataloaders/optimizer/step）、FLCE 损失、3/5 通道契约、anatomy backbone 仅接受
  3 个 MRI 通道、零初始化与 clamp、NoFFT + MRI-only 强度增强；更新七类名、七 variant 与注册表
  断言。
- `tests/unit/test_positive_case_sampling.py`：新增基于**真实类层次**的宿主测试（训练
  loader=PositiveCaseDataLoader、验证 loader=原生 nnUNetDataLoader、阳性最后一槽 + 病灶中心
  裁剪、验证 batch 仅来自验证 identifiers），并同步 7-name/7-variant 断言。
- `tests/unit/test_predict_entry.py`、`tests/unit/test_prior_transforms.py`、
  `tests/unit/test_prepare_picai_failclosed.py` 同步更新（7 个 `trainer_name` 解析、anatomy
  增强参数化到两个类、zonal description 断言）。
- `pytest -q` → **192 passed / 0 failed**；`ruff check` / `ruff format --check` 全部通过。

### 文档

- `README.md`：variant 表扩为 7 行并列明公平比较关系（`positive_sampling` ↔
  `image_gate_positive_sampling` = RQ1；`image_gate_positive_sampling` ↔
  `anatomy_gate_positive_sampling` = RQ2；旧 `baseline` ↔ 旧 `image_gate` 为原生采样条件下的
  受控 RQ1 对照，但不能与阳性采样分支混合比较、单次运行与早期优化异常限制结论强度）；新增
  Dataset605/606 一致性边界（配置层面已核对一致，但**未**做逐数组审计）；NSD 说明改为
  surfel-area weighted；术语澄清。
- `docs/Training_Log.md`：`positive_sampling` 状态由「尚未运行」更新为**训练中**（2026-09-22
  07:16 UTC 启动），登记启动时采样摘要与进行中的 patch pseudo Dice 观察，并明确其非最终结果。

### 未改动 / 未运行

- 未修改 `third_party/nnUNet`、`data/`、`workdir/`、`outputs/`、任何 checkpoint；
  已物化的 Dataset606 旧 `description` 未被改写，数值数据未变；
- 未运行任何训练、validation、inference、planning/preprocessing 或真实数据评测；
  `positive_sampling` 训练仍在运行，本轮未停止/重启/干扰；
- 未新增/升级/降级依赖；未 commit、未 `git add`；
- Zone-only gate、WG gate、Direct anatomy、boundary-aware/hard-example sampler、新损失、
  新 optimizer/scheduler、gate-map 全体积导出与 PZ/TZ 亚组评估**均未实现**。

---

## 2026-09-22 — 受控项目侧 patch sampler：positive_sampling（AGENTS.md 放宽 + 新 Trainer）

### AGENTS.md §5 放宽为「nnU-Net-centered 与受控扩展」

- **删除**的绝对限制：dataloader / patch sampling 必须全部由 nnU-Net 原生实现；不得再写第二套
  patch sampler；项目扩展只限于原有损失、gate 与 PZ/TZ 适配。
- **新增**受控扩展条件：当原生机制不能满足明确研究假设时，允许实现病例级采样概率、阳性/阴性
  病例平衡、lesion-centred / boundary-aware / hard-example patch sampling，以及与研究问题直接
  相关的输入、损失或轻量网络模块。
- 项目侧 dataloader/patch sampler 的硬性边界（仍**绝对禁止**修改 `third_party/nnUNet`）：
  优先继承或包装 `nnUNetDataLoader`；复用预处理数据与 `class_locations`（不重采样原始影像、
  不读原始 NIfTI）；不重新实现训练循环/optimizer/LR scheduler/checkpoint/验证器/滑窗推理；
  对缺失 metadata、无阳性病例、非法病例 ID fail-closed；训练与验证采样明确分离、验证默认保持
  原生；有纯合成测试；用独立 Trainer 类名与独立输出目录。
- 环境、数据安全、长任务、进度、文档纪律与文件创建权限等章节**未改动**；未重建阶段门或 runbook。

### 为什么需要项目侧采样（研究假设）

nnU-Net v2.6.2 默认先**均匀抽取病例**，再按 `oversample_foreground_percent` 决定某个 batch 槽位
是否强制前景裁剪。若被选中的是阴性病例，`force_fg` 也无法产生病灶中心 patch。Dataset605 fold 0
的阳性病例比例为 362/1277 ≈ 28.35%，batch size 2 时真正保证含病灶的 patch 槽位约为
`0.5 × 28.35% ≈ 14%`，且 `baseline`（N0）与已被用户中止的 `optimized_baseline`（Dice+CE）都出现
长期停留在全背景预测的现象。因此本轮只改变**训练集病例/patch 采样**这一个变量。

### 新增 `src/zonal_reliability_fusion/nnunet/sampling.py`

- `PositiveCaseDataLoader(nnUNetDataLoader)`：只覆盖两处。
  - `get_indices`：先调用父类原生选例（无限模式下的均匀有放回抽样），保留前面普通槽位，再把
    batch 最后 `positive_cases_per_batch` 个位置替换为从阳性病例集合中均匀有放回抽取的病例；
    父类返回值数量不等于 `batch_size` 时直接报错。
  - `_oversample_last_XX_percent`：最后 `positive_cases_per_batch` 个槽位始终 return True
    （force foreground）。bbox、padding、数据读取与 patch 裁剪全部继续由父类 `get_bbox` /
    `generate_train_batch` 用预处理 `class_locations` 完成，**不自行实现 get_bbox、不计算病灶
    坐标、不读原始 NIfTI**。
  - 构造期 fail-closed：阳性 ID 去重排序后必须非空且全部属于该 loader 的 dataset identifiers；
    `1 <= positive_cases_per_batch <= batch_size`；拒绝 `probabilistic_oversampling=True`
    （概率式槽位决策会绕过 force_fg 保证）。
- `PositiveCaseSamplingMixin`：只覆盖 `get_dataloaders`。
  - 训练 loader 换成 `PositiveCaseDataLoader`，`sampling_probabilities=None`、
    `probabilistic_oversampling=False`、`positive_cases_per_batch=1`；transforms 仍经
    `self.get_training_transforms(...)` 解析（NoFFT 修复等增强 mixin 依然生效）；patch size、
    initial patch size、deep-supervision scales、rotation、dummy 2D、mirror axes 继续来自
    nnU-Net 配置；augmenter 类型/进程数/缓存/pin memory 与固定 v2.6.2 逐行一致。
  - 验证 loader 保持原生 `nnUNetDataLoader`，不按验证标签做任何阳性重采样，actual
    full-volume validation 完全交给 nnU-Net。
  - 识别阳性病例的 `identify_positive_cases`：只遍历**当前 fold 训练 dataset** 的
    identifiers，逐例读取预处理 `.pkl` 的 `class_locations` 并结合
    `label_manager.foreground_labels` 判断；`.pkl` 缺失/不可读、properties 非 dict、
    `class_locations` 缺失或非 dict、缺少前景标签条目、非可计数条目、无任何阳性病例、
    region 类标签都会显式报错（fail-closed），绝不静默当阴性，绝不注入 validation 病例。
  - 训练日志打印一次结构化摘要：`training_cases / positive_cases / negative_cases /
    positive_cases_per_batch / batch_size / guaranteed_positive_patch_fraction`，数值全部来自
    运行时实际 fold，不硬编码。

### 新增 Trainer `nnUNetTrainerPICAI_FLCE_PositiveSampling_NoFFT`

- MRO：`nnUNetTrainerPICAI_FLCE_PositiveSampling_NoFFT → PositiveCaseSamplingMixin →
  nnUNetTrainerPICAI_FLCE_NoFFT → NoFFTAugmentationMixin →
  PICAIFocalCrossEntropyLossMixin → nnUNetTrainer → object`。
- 因此：loss 仍是 `0.5*Focal(gamma=2)+0.5*CE`（deep supervision 权重不变）；NoFFT 修复仍生效；
  网络仍是 plans 构建的原生 `PlainConvUNet`（3 个 MRI 通道，无 gate、无 PZ/TZ）；optimizer
  （SGD+Nesterov）、PolyLR、1000 epochs、batch_dice、patch/batch size、checkpoint/resume、
  validation、inference 全部继承原生实现。
- **不**继承 `nnUNetTrainerPICAI_DiceCE_NoFFT` / `nnUNetTrainerPICAI_ImageGate` /
  `nnUNetTrainerPICAI_AnatomyGate` / `_GatedTrainerBase`。

### 入口与注册表

- `scripts/train/train_nnunet.py`：新增第 5 个 variant `positive_sampling` → 新 Trainer 类；
  docstring、argparse help 与示例同步更新；四个既有 variant 保留不变。
- `PROJECT_TRAINERS` 注册表与 `scripts/inference/predict_nnunet.py` 的
  `PROJECT_TRAINER_NAMES` 加入新类名，使 checkpoint 的 `trainer_name` 能在预测入口被解析；
  预测仍完全交给官方 `nnUNetPredictor`，未新增推理器。

### 测试

- 新增 `tests/unit/test_positive_case_sampling.py`（纯合成 CPU，tmp_path + 合成数组/合成 pkl，
  不读真实医学数据、不写 data/workdir/outputs、不启动训练）：覆盖阳性识别（含 np.int64 键、
  缺失/不可读 pkl、class_locations 缺失/非 dict、缺标签条目、无前景标签、无阳性病例）、
  loader 构造 fail-closed（空/未知 ID、per_batch=0/>batch_size、probabilistic_oversampling）、
  槽位行为（末位恒阳性、首位覆盖含阴性的完整训练集、原生对照、force_fg 标志、连续 batch 的
  阳性 target 含前景）、mixin 接线（训练 loader 是 PositiveCaseDataLoader、验证 loader 是原生
  nnUNetDataLoader、摘要数值来自运行时）、新 Trainer 的 MRO/损失/网络/NoFFT/五类名互异。
  核心采样行为均被实际执行验证，而非仅比对类名。
- 同步更新 `tests/unit/test_nnunet_trainers.py`（子类、NoFFT、inference builder、五类名互异、
  五 variant 映射、`PROJECT_TRAINERS` 注册表）与 `tests/unit/test_predict_entry.py`（预测入口
  五个 `trainer_name` 的解析）。
- `pytest -q` → **168 passed / 0 failed**；`ruff check` / `ruff format --check` 全部通过
  （`python -m ruff format src scripts tests` 只重排了本轮新增的两个文件）。

### 未改动 / 未运行

- 未修改 `third_party/nnUNet`、`data/`、`workdir/`、`outputs/`、`logs/`、任何 checkpoint、
  `docs/Research_Plan.md`；
- 未执行任何真实训练、validation、inference、预处理、planning 或评测；未新增/升级/降级依赖；
- `positive_sampling` 的真实训练**尚未运行**（命令见 `README.md`「训练与推理命令」）。

---

## 2026-09-22 — 统一分割评估器：RunStats 阶段化 + 普通异常也打印汇总

对评估入口的第二轮最小修补，仍不改变任何指标定义。

### `RunStats` 增加 phase / unit（修正被 mode 单一推断的单位）

- 问题：单位此前由 `mode` 推断（`full` ⇒ model-case）。但 `full` 可能在**模型加载**或**可比性检查**
  阶段就失败，此时 `ok` / `failed` 是**模型**计数，标成 model-case 是错的。
- 新增 `phase`（`preflight` / `model_loading` / `comparability` / `full_cases` / `publishing` /
  `completed`）与 `unit`（`model` / `model-case`），并新增 `PHASE_*` / `UNIT_*` / `NOT_APPLICABLE`
  常量。单位由 phase 强制决定，同一次汇总中的 ok / failed / skipped **共享同一个 unit**。
- 阶段切换统一走 `begin_model_loading()` / `begin_comparability()` / `begin_full_cases(total)` /
  `begin_publishing()` / `mark_completed()`，计数只通过 `model_loaded()` / `model_failed()` /
  `full_case_ok()` / `full_case_failed()` / `end_full_cases()` 累加。
- **只有真正进入逐例循环之前**才切到 `model-case` 并重置计数；可比性检查失败时单位仍是 `model`。
- 不可知的量保持 `None` 并渲染为 `not_applicable`（如 `preflight` 的 unit 与三个计数、`model_loading`
  的 `skipped`），**不伪造成 0**。
- 统计字段仍**不参与**指标计算，也不影响 fail-closed 判定。

### 所有普通失败都打印结构化汇总

- `main()` 在保留 `EvaluationError` 分支的同时新增 `except Exception` 分支：同样打印
  `status=failed` 的汇总后**原样重抛**（退出码仍由 `__main__` 决定）；两个分支互斥，配合唯一的
  `_report_failure()`，同一异常**不会打印两份汇总**。
- `KeyboardInterrupt` / `SystemExit` 派生自 `BaseException` 而非 `Exception`，不会被捕获。
- 成功路径在 `_run` 返回后调用 `mark_completed()`，因此 `phase=completed`。

### 脚本模块 docstring 清理

- 删除 `--model baseline=<baseline>/fold_0` 这类尖括号占位符命令，以及 `--nsd-tolerance-mm 1.0`、
  `--volume-thresholds-mm3 500,2000` 这两个**会被误读为推荐值**的示例。
- 用法段改为直接指向 `README.md` 的「病灶分割评估」小节作为**唯一**命令来源，避免维护第二份命令；
  明确容差与阈值必须由研究者自行确定，本脚本不设默认值。

### 测试

`pytest -q tests/unit/test_evaluate_segmentation.py` → **73 passed / 0 failed**
（上一轮 67 项全部保留通过，新增 6 项）：full + 两个缺失 fold（`phase=model_loading`、`unit=model`、
`failed=2`）、full + 一成功一失败加载（不得标成 model-case）、多模型可比性失败
（`phase=comparability`、`unit=model`、`ok=2`）、full 正常跑完（`unit=model-case` 一直延续到
`completed`）、进入逐例循环时立即切到 `full_cases`/`model-case` 且计数已重置（部分失败 `(1,1,0)`）、
`_publish_json` 抛 `OSError` 时 stderr **有且只有一份**失败汇总、异常原样抛出、无 JSON 与临时文件。
另有 3 项既有测试随本次 API 变更同步更新（`ok_unit` → `phase`/`unit`、summary 模式 `skipped` 由
`0` 改为 `not_applicable`）。

### 未运行

**未在任何真实模型上运行评估**；未读取任何真实医学 NIfTI；未新增任何文件；未改动 Trainer /
network / data / workdir / outputs / third_party / `docs/Research_Plan.md` / `docs/Training_Log.md`。

---

## 2026-09-22 — 统一分割评估器：逐例异常收集与结构化结束汇总

对上一版评估入口的最小修补，不改变任何指标定义。

### 逐例异常收集范围

- 新增 `_build_full_case_entry(cid, case, tolerance_mm)`：把一例从**读取掩膜**到**几何/标签校验、
  阴性 reference 必须为空、六项计数核对、物理体积、`surface_metrics`、`component_analysis` /
  `_pred_component_stats`、entry 构造**的完整流程收进同一函数。函数要么返回完整 entry，要么抛出
  异常；`mask_verified=True` 只在全部步骤成功后写入，因此失败病例**不会留下伪造的"已验证"结果**。
- `run_full_mode` 的逐例 `try/except` 现在包住整段流程（此前只覆盖 `load_mask_pair`）。任一例的
  MedPy / SciPy / 体积 / 连通域失败只记录 `[model] case: 异常类型: 信息` 并 `continue`，其余病例
  继续处理；全部病例跑完后统一 `raise`，写明错误总数且**不发布** JSON、不留临时文件。
- `except Exception` 天然**不捕获** `KeyboardInterrupt` / `SystemExit`（二者派生自 `BaseException`
  而非 `Exception`）。
- `_verify_case_counts` 改为返回**不含** model/case 前缀的差异描述，前缀由调用方统一添加。

### 结构化结束汇总

- 新增 `RunStats`：字段 `ok` / `failed` / `skipped` / `elapsed` / `output`；`elapsed` 为
  `time.monotonic()` 差值（不受系统时间调整影响）。该统计**只用于汇总**，不参与指标计算，也不影响
  fail-closed 判定。
- `ok` 的统计单位随模式变化，并随汇总一起打印（`OK_UNIT_SUMMARY` / `OK_UNIT_FULL`）：
  `summary` 模式 = 成功加载并通过逐例计数校验的 **model** 数；`full` 模式 = 成功读取并通过全部
  逐例检查的 **model-case 对** 数（`full` 下 `ok` / `failed` / `skipped` 由 `run_full_mode`
  覆盖为逐例计数，`skipped` 为未进入逐例检查的对数，正常为 0）。
- `main()` 拆为「建立统计 + 打印汇总」与「实际执行 `_run(args, run)`」：成功路径打印
  `status: succeeded`，失败路径在 stderr 打印 `status: failed` 后原样重抛（退出码仍为 2）。
  `--help` 在建立统计之前由 argparse 退出，不打印汇总（帮助不是一次运行）。tqdm 进度条不替代该汇总。

### 测试

`pytest -q tests/unit/test_evaluate_segmentation.py` → **67 passed / 0 failed**
（原 59 项全部保留通过，新增 8 项）：`surface_metrics` 与 `component_analysis` 逐例抛异常时
**所有病例都被执行**且汇总同时含全部 case id 与正确总数、失败后不产生 JSON 与临时文件、失败不向
report 写入任何伪造结果与聚合块、`KeyboardInterrupt` 原样抛出、`full` 模式的 ok/failed 以
model-case 对为单位（全通过 3/0/0，部分失败 1/2/0）、`RunStats` 字段与单位、成功与失败两条路径
都打印结构化汇总。

### 文档

- `README.md`：可复制命令去掉 Bash 尖括号占位符；`baseline` / `image_gate` / `optimized_baseline`
  一律使用项目内完整相对路径；NSD 容差与体积分层阈值改用带引号的环境变量
  （`"${NSD_TOLERANCE_MM:?...}"`），未设置即报错，不替研究者推荐取值；注明 `full` 是长任务且只能在
  训练结束后运行；补充逐例异常收集与结构化汇总的说明。

### 未运行

**未在任何真实模型上运行评估**；未读取任何真实医学 NIfTI；未新增任何文件；未改动 Trainer /
network / data / workdir / outputs / third_party / `docs/Research_Plan.md`。

---

## 2026-09-22 — 新增统一病灶分割评估入口

### 新增

- `scripts/evaluate_segmentation.py`：统一的论文级**病灶分割**评估入口。**离线只读**已存在的
  nnU-Net validation 产物，不重写 validation / inference / prediction，不训练模型。
  - `--mode summary`（默认）：只读 `FOLD_DIR/validation/summary.json`，输出体素计数派生的全部指标，
    不读 NIfTI；加载时对每例计数做严格校验并**由 TP/FP/FN 重算 Dice**（见下「fail-closed」）。
  - `--mode full`：额外读取 `metric_per_case` 指向的 prediction / reference NIfTI，**遍历并校验
    全部病例**（阳性 / 假阳阴性 / 真阴），追加 mm³ 物理体积与 RVE/ARVE、HD95 / ASSD / NSD@τ、
    病灶体积分层、连通域失败分析；必须显式提供 `--nsd-tolerance-mm`（不设置默认值）。
    tqdm 覆盖逐病例主循环，总量 = 所有模型病例数之和；`--no-progress` 可关闭。
  - `--model NAME=FOLD_DIR` 可重复；`--bootstrap-resamples`（默认 10000）与 `--seed`
    （默认 20260922）保证 CI 可复现；`--output` 可选，不提供时只打印终端表格。
- `tests/unit/test_evaluate_segmentation.py`：**59 项**纯合成 CPU 测试（tmp_path + 合成 summary.json +
  合成小型 NIfTI），不读真实医学数据、不写 `data/` `workdir/` `outputs/`、不实例化 Trainer、
  不启动 inference、不使用 GPU。

### 指标口径（避免歧义）

- **两种 precision 严格区分**：`positive_voxel_precision = ΣTP/(ΣTP+ΣFP)`（仅阳性病例，论文核心
  口径）与 `all_prediction_voxel_precision = ΣTP/(ΣTP+ΣFP_阳性+ΣFP_阴性)`（分母含阴性病例假阳
  体素）。二者不得笼统称为 "precision"。
- 阳性病例**完全漏分时 Dice 记为 0**，不得排除；`positive_macro_dice` 为逐例宏平均，
  `positive_micro_dice = 2ΣTP/(2ΣTP+ΣFP+ΣFN)` 只汇总阳性病例。
- 完全漏分结构拆为 `positive_overlap_cases` / `positive_missed_cases` /
  `positive_empty_prediction_cases` / `positive_wrong_location_cases`。
- 体素数量一律标注 **voxel-based**；mm³ 只在 full 模式由 spacing 计算。SPACING 轴序经核实：
  SimpleITK 的 `GetSpacing()` 为 (sx, sy, sz)，而数组轴序为 (z, y, x)，内部按
  `spacing[::-1]` 对齐后再用于距离/体积，避免把体素距离当作 mm。
- 表面指标复用环境现有实现 `medpy.metric.binary.hd95 / assd`（尊重各轴 spacing，单位 mm）；
  NSD@τ 为本实现，**surface-voxel based**（非 surface-area weighted），并在 JSON 中记录方法与容差。
- GT 非空、预测为空 → NSD = 0 且 HD95/ASSD = null，同时计入 missed；GT 与预测均为空 → 不纳入
  阳性表面统计。
- **明确不计算 AUROC / average precision / FROC / PI-CAI challenge score**，JSON 中
  `metrics_scope` 字段固定声明这一点。

### fail-closed 行为

- **summary 模式逐例严格校验**：TP/FP/FN/TN/n_pred/n_ref 必须是非负整数——拒绝小数（不做
  `int()` 静默截断）、负数、NaN/Inf；强制 `TP+FP == n_pred` 与 `TP+FN == n_ref`；**Dice 一律由
  TP/FP/FN 重算**，summary 中的值只用于一致性核对（偏差超 `1e-6` 即失败），宏平均使用重算值；
  真阴（空-空）病例的 Dice 必须为空/NaN，给出有限值即失败。
- 任一模型缺少 `validation/summary.json`、case id 重复、prediction/reference case id 不一致时失败。
- **模型加载错误一次性汇总**：先遍历全部 `--model` 再统一报告，不因第一个模型失败就提前退出。
- 多模型时校验 case id 集合、GT 阳性/阴性身份、逐例 `n_ref` 完全一致；不一致即非零退出，
  **禁止退化为只比较病例交集**。
- 配对 bootstrap 按**病例配对**重采样（同一批索引同时重采样两个模型）；CI 跨 0 时只输出
  `no_clear_paired_improvement`，不宣称显著提升，也不提供 p 值。
- **full 模式遍历并校验全部病例**（阳性 / 假阳阴性 / 真阴）：每例读取 prediction 与 reference，
  校验 size / spacing / origin / direction 与 0/1 标签，由掩膜重算 TP/FP/FN/TN/n_pred/n_ref 并与
  summary **逐项**核对；`n_ref=0` 的病例其 reference 必须实际为空；**真阴病例（n_pred=0）不得跳过
  文件与几何检查**。每例只读一次；阴性假阳团块统计复用已加载掩膜，不存在二次读盘路径。
- 错误汇总由 `_format_error_report` 统一输出：写明总条数；超过显示上限（50 条）时明确标注
  「仅显示前 50 条，另有 M 条未显示」，不在截断的同时宣称完整。
- 任一失败即非零退出，**不发布**输出 JSON，也不留临时文件。
- 输出使用临时文件 + `os.replace` 原子发布；目标已存在时拒绝覆盖；JSON 以
  `allow_nan=False` 写出，无定义值统一为 `null`（不写 NaN/Infinity）。
- `--nsd-tolerance-mm` 必须是**正的有限数**（nan / inf / -inf / 非正一律拒绝），full 模式必须显式
  提供；`--volume-thresholds-mm3` 保持正数、有限、严格递增。

### 测试

`pytest -q tests/unit/test_evaluate_segmentation.py` → **59 passed / 0 failed**。

覆盖：完美分割、部分重叠、空预测、完全错位、阴性空/假阳、宏 Dice 含 0、micro 只汇总阳性、
两种 precision 区分、阴性 FP 病例率与体积分布、多模型不可比（集合/身份/n_ref）、
配对 bootstrap 可复现、identical masks（Dice=1 / HD95=0 / ASSD=0 / NSD=1）、
单体素平移验证 mm 距离尊重各向异性 spacing、NSD 容差生效、空预测表面处理、
各向异性 spacing 的 mm³ 体积、体积阈值参数校验、输出拒绝覆盖、失败不发布半成品、
CLI `--help` 与失败退出码、模块导入不拉起 torch / nnunetv2。

以及严格校验用例：summary 计数非自洽（`TP+FP != n_pred`、`TP+FN != n_ref`）、小数/负数/NaN 计数、
Dice 与重算值不一致、真阴给有限 Dice、阳性缺 Dice、阴性 spacing 不一致、
真阴文件缺失（prediction 与 reference 各一）、阴性预测含非 0/1 标签、阴性 reference 实际非空、
阴性 n_pred/FP 与掩膜不一致、混合病例集（阳性 + 假阳阴性 + 真阴）完整通过、
full 模式每例只读一次、nan/inf/非正容差拒绝、多模型加载错误一次性汇总。

### 未运行

**未在任何真实模型上运行评估**（`optimized_baseline` 于 2026-09-22 00:48 UTC 起仍在训练，
尚无 validation 产物）。真实运行由研究者在训练结束后执行。

---

## 2026-09-22 — 文档结构：新增 Findings 与文件创建权限

### 文档结构

- 新增 `docs/Findings.md`：只记录**跨实验**的观察、问题优先级与下一步判据，是实验记录的横向汇总。
  治理文档由四份变为**五份**（`README.md`、`Research_Plan.md`、`Development_Log.md`、
  `Training_Log.md`、`Findings.md`）。
- 将 `docs/experiments/` 正式纳入 `AGENTS.md` §6：**每个实验一份**，文件名即 variant 名，
  记录该实验自身的配置/运行事实/训练动态/验证结果/产物/观察与限制。各实验文档
  **互相独立：不写跨实验对比、不互相引用**。
- 分工明确：治理文档记录跨实验的事实与索引，实验文档记录单实验详情。
- `README.md` 目录结构同步更新。

### 新增规则

- `AGENTS.md` 新增 **§7 文件创建权限**：**只有用户的明确命令才能创建新文件**；禁止代理自行、
  顺手或"顺便"创建文档/说明/总结/报告/配置/脚本/测试/模板/临时笔记；能改既有文件的一律改既有
  文件；理由成立但用户未明确要求时先征得同意，不得先建后报；新建文件不得落在
  `data/`、`workdir/`、`outputs/`、`third_party/`。

### 文档更新

- `docs/Training_Log.md`：`image_gate` 状态由「运行中」更新为「已完成」，登记训练起止、验证、
  Mean Validation Dice（0.17025）与检出结构。
- 未改动任何代码、测试、`third_party/` 或 `docs/Research_Plan.md`；未运行任何训练/验证/推理。

---

## 2026-09-21 — 新增 optimized_baseline（原生 Dice+CE 单变量分支）

针对已完成的 N0（`nnUNetTrainerPICAI_FLCE_NoFFT`）长期停留在背景预测的现象，新增一个
**单变量**优化基线：只把损失从项目的 PI-CAI Focal+CE 换成 nnU-Net v2.6.2 原生 Dice+CE。

### 新增

- `nnUNetTrainerPICAI_DiceCE_NoFFT(NoFFTAugmentationMixin, nnUNetTrainer)`：**不覆盖**
  `_build_loss`，MRO 直接解析到 `nnUNetTrainer._build_loss`，因此用的是原生 `DC_and_CE_loss`
  （`batch_dice` 取自 plans、`do_bg=False`、`weight_ce=weight_dice=1`、
  `MemoryEfficientSoftDiceLoss`）与原生 `DeepSupervisionWrapper` 权重；**无自写 Dice/CE**。
- 网络仍由原生 `get_network_from_plans` 按 plans 构建（`PlainConvUNet`、3 通道），**无 gate**；
  该类硬性不继承 `PICAIFocalCrossEntropyLossMixin` / `nnUNetTrainerPICAI_FLCE_NoFFT` /
  `_GatedTrainerBase`，现有 FLCE baseline 与两个 gate variant 行为完全不变。
- 保留 NoFFT 修复（`GaussianBlurTransform` 仍在，仅关闭 FFT benchmark，blur 概率/sigma 不变）。
- 训练入口新增 variant `optimized_baseline`（仍只做映射后调用官方 `run_training`，无自写训练循环、
  optimizer、scheduler、checkpoint 或 epoch 参数）；预测入口 `PROJECT_TRAINER_NAMES` 与
  `PROJECT_TRAINERS` 注册表同步加入新类名。
- 测试扩展 `tests/unit/test_nnunet_trainers.py`、`tests/unit/test_predict_entry.py`：继承关系
  （不含被禁基类、MRO 中也不含）、`_build_loss` 与原生实现为同一函数、实际调用 `_build_loss()`
  得到原生 Dice+CE（DS 时外层为原生 wrapper）、builder 返回无 gate 的原生网络、NoFFT 仍生效、
  四类名互异、train 入口有且仅有四个 variant、预测入口能解析新 `trainer_name`。
  `pytest -q` **58 passed / 0 failed**（纯合成 CPU，不读真实医学数据、不写 outputs）。

### 未改动 / 未运行

网络、数据集、split、patch/batch size、增强、前景采样、deep supervision、SGD、PolyLR、初始学习率、
epoch 数、checkpoint、validation、inference 全部不变；`docs/Research_Plan.md` 未改。
`optimized_baseline` **尚未训练**（`image_gate` 训练进行中，GPU 被占用）。

---

## 2026-09-21 — nnU-Net-first 重构

把项目从「自写第二套训练框架 + 阶段门治理」彻底重构为 **nnU-Net-first**：通用机制全部交给
固定版本 nnU-Net（`third_party/nnUNet` v2.6.2，commit `74ceb68`，未修改），项目只保留研究必需
的最小扩展。

### 新增（唯一扩展点 `src/zonal_reliability_fusion/nnunet/`）

- `networks.py`：`SpatialModalityReliabilityGate`（两层 1x1x1 conv，末层零初始化）与
  `GatedNNUNet` 包装器。backbone 一律由 `get_network_from_plans` 构建（原生 `PlainConvUNet`），
  包装器代理 forward、暴露 `decoder` 属性使 `set_deep_supervision_enabled` 原样可用、透传 deep
  supervision 输出、strict state_dict、anatomy variant 严格校验 5 通道并对 PZ/TZ `clamp(0,1)`。
  权重定义 `scales = 3*softmax(logits)`，零初始化时 `scales≡1`，初始行为与原生 baseline 逐值一致。
- `transforms.py`：`MRIChannelRestrictedTransform` + `restrict_intensity_transforms_to_mri`，把
  强度增强限制在前 3 个 MRI 通道（PZ/TZ 豁免）；空间/镜像变换不包装，继续同步作用于全部通道。
- `trainers.py`：`PiCAIFocalCrossEntropyLoss`、NoFFT 增强修复 mixin、`nnUNetTrainerPICAI_FLCE_NoFFT`
  （baseline）、`nnUNetTrainerPICAI_ImageGate`、`nnUNetTrainerPICAI_AnatomyGate`。三者均继承
  `nnUNetTrainer`，复用同一 loss/optimizer/scheduler/epochs/采样/增强主体/checkpoint/validation/
  inference；gate variant 只覆盖 `build_network_architecture`，anatomy 另外覆盖 `get_training_transforms`。

### 新增入口

- `scripts/train/train_nnunet.py`：唯一训练入口，只做 variant → Trainer 类映射后调用 nnU-Net 官方
  `run_training`（在本进程内替换 `get_trainer_from_args`，不修改 nnU-Net 源码），无自写训练循环。
- `scripts/data/prepare_picai_nnunet.py`：统一数据准备（`baseline`/`zonal`/`splits` 三个 subcommand），
  带 tqdm 与成功/失败/跳过/耗时/输出目录汇总。`zonal` 由 `zonal_<source>.nii.gz` 派生与 T2W 网格
  一致的 PZ/TZ 两个 `[0,1]` 通道（`noNorm`），组织 Dataset606_PICAI_Zonal（5 通道）。

### 删除（旧的第二套框架与阶段门系统，共 168 个 tracked 文件）

- 自写训练框架：`src/.../training/`（trainer/checkpointing/optim/deep_supervision/losses/metrics/
  gpu_memory_profile）、`inference/sliding_window.py`、`evaluation/full_volume_validator.py`、
  `data/`（batch_provider/patch_sampler/preprocessed_store/transforms/picai/zonal_prior）。
- 自写整网实现：`models/`（m0_concat、m0_residual_concat、residual_unet3d、unet3d、residual_blocks、
  fusion_blocks、fusion_multi_branch、blocks、factory、profiling）。
- 旧 architecture config/hash/contract 系统：`config/`（architecture/experiment/plans）。
- 阶段门/协议系统：`protocols/`（g0_r/g0_e/g0_sap/g0_r_qc/g0_r_automated_qc/common）、
  `integrations/`（nnunet_picai_flce 已迁入 `nnunet/trainers.py`、nnunet_dataloader_probe）、`utils/`。
- 配置：整个 `configs/`（protocols/、architectures/ 的 M0–M4、experiments/ 的 M0–M4 YAML）。
- 脚本：`scripts/train/`（train_m0、train_n0_picai_flce、validate_m0_setup、summarize_m0_resenc_architecture、
  diagnose_n0_zero_dice）、`scripts/audit/`、`scripts/evaluate/`、`scripts/data/` 旧脚本、
  `scripts/analyze_*.py`、`check_registration.py`、`overlay_qc.py`。
- 文档：GLOSSARY/STATUS/START_HERE/research_plan(_plain_language)/protocol_changelog/
  PICAI_Model_Readiness_Audit/N0_PICAI_Official_Baseline/P2_M0_*/project_structure/Data_Analysis/
  experiment_log，以及 `docs/protocols/`、`docs/runbooks/`、`docs/literature/`、`docs/design/`、`docs/figures/`。
- 测试：仅验证已删除框架的测试（test_g0_*、test_checkpointing、test_trainer_synthetic、test_sliding_window、
  test_full_volume_validator、test_patch_sampler、test_batch_provider、test_m0_*、test_residual_*、
  test_architecture_config、test_skip_contract、test_train_m0_*、test_losses、test_picai、test_plans、
  test_profiling、test_progress、test_fusion_models、test_zonal_prior_pipeline、test_materialization_hardening、
  test_seg_sentinel_mapping、test_gpu_memory_profile、test_n0_zero_dice_diagnosis、test_evaluate_n0_validation、
  test_nnunet_picai_flce、test_loss_alignment_n0 等）。

### 保留（未改动）

`data/`、`workdir/`、`outputs/`（含 N0 baseline 与 legacy M0 产物）、`third_party/nnUNet`、
`scripts/env_nnunet.sh`。

### 测试

`tests/` 重写为纯合成 CPU 单元测试（不读取真实医学数据）：`test_nnunet_trainers.py`、
`test_nnunet_networks.py`、`test_prior_transforms.py`。覆盖 PI-CAI 损失与公开公式逐值一致、
三个 Trainer 为 `nnUNetTrainer` 子类且类名互异、NoFFT 保留 blur 且 benchmark=False、image/anatomy
gate 的通道契约、零初始化 `scales≡1` 与 gated MRI 恒等、deep supervision 输出格式与切换、strict
state_dict、PZ/TZ clamp、强度增强不动 PZ/TZ、空间变换同步、builder 可用于 inference、`train_nnunet.py --help`。
结果见 `docs/Training_Log.md` 与提交说明。

---

## 2026-09-21 — 重构修补：预测入口 / 依赖兼容边界 / 数据准备 fail-closed

在不重新设计、不恢复旧框架、不修改 `third_party/nnUNet` 的前提下修补上一轮重构的缺口。

### 预测加载链路（保持官方推理器）

- 新增最小项目预测入口 `scripts/inference/predict_nnunet.py`：在**进程内**把三个项目 Trainer 名
  直接映射到项目类（替换 `nnunetv2.inference.predict_from_raw_data.recursive_find_python_class`
  的解析：已知名直接返回、**不做递归扫描**，未知名字委托 nnU-Net 原函数），随后调用官方
  `predict_entry_point_modelfolder`。滑窗推理/预处理/导出仍全部由官方 `nnUNetPredictor` 完成，
  未自写推理器，未改动 nnU-Net 源码。
- `trainers.py` 增加 `PROJECT_TRAINERS` 注册表与 `resolve_trainer_class`，训练/预测入口共用。
- 新增合成测试 `tests/unit/test_predict_entry.py`：验证三个 checkpoint `trainer_name` 都能被预测
  入口解析到项目类（覆盖官方 `initialize_from_trained_model_folder` 实际调用的解析函数），
  而不仅是 `build_network_architecture`。

### nnU-Net 2.6.2 / dynamic-network-architectures 依赖兼容边界（如实记录，未升级）

- **现状**：`third_party/nnUNet` 为 v2.6.2，官方要求 `dynamic-network-architectures>=0.4.1,<0.5`；
  但 lm 环境中实际安装的 DNA 为 **0.3.1**，且 `nnunetv2` 元数据版本为 **2.5**（来自
  `/opt/data/private/lm/PENGWIN_Challenge/nnUNet` 的 editable install，要求 `DNA<0.4`）。
  因此 **2.6.2 的 DNA 下界未被满足**；未先 `source scripts/env_nnunet.sh` 时，`import nnunetv2` 会失败或
  解析到错误的环境分支，必须先 `source scripts/env_nnunet.sh` 把 `PYTHONPATH` 前置到
  `third_party/nnUNet` 才解析到固定的 2.6.2 源码。
- **不升级/不覆盖**：升级 DNA 到 0.4.x 会破坏依赖 `DNA<0.4` 的 PENGWIN nnUNetv2 2.5 editable 安装，
  故本项目保持 DNA 0.3.1 不变。
- **兼容方案与影响**：仅使用 `PlainConvUNet` 路径（Dataset605/606 的 `3d_fullres` plans 均为该类），
  其在 DNA 0.3.1 下可正常构建/前向/deep supervision/strict state_dict（合成测试通过）。
  **影响边界**：一旦改用需要 DNA>=0.4 的架构（如 `ResidualEncoderUNet` 新参数）或 2.6.2 中依赖
  新版 DNA 行为的代码路径，当前环境不保证可用，必须先解决 DNA 冲突。
- **入口保护**：`zonal_reliability_fusion.nnunet.check_fixed_nnunet_runtime()` 在训练/预测入口显式校验
  `nnunetv2` 解析到 `third_party/nnUNet`（否则报错，避免误用 2.5 fork）、`PlainConvUNet` 可由
  `pydoc` 直接定位（从而 `get_network_from_plans` 不退化为递归扫描），并**如实打印** DNA 版本是否满足
  `>=0.4.1,<0.5`；不满足时只 WARNING，不宣称依赖完全满足。

### 数据准备 fail-closed

- `scripts/data/prepare_picai_nnunet.py`：任一 `failed/conflict`（或完整运行时磁盘病例集合与选择集合
  不一致）先打印完整汇总，再 `SystemExit(2)`，且**不发布**新的 `dataset.json`；`numTraining` 改为
  `imagesTr` 中拥有全部通道的**实际完整病例数**；`--overwrite` + `--cases` 子集运行后若仍含选择外旧病例，
  明确 WARNING（绝不静默遗留）；`splits_final.json` 冲突时非零退出且不覆盖既有文件；删除未使用的
  `datetime` import。
- 新增合成测试 `tests/unit/test_prepare_picai_failclosed.py`（临时目录 + 纯合成小文件 / 合成 NIfTI，
  不读真实医学数据）。

### 其他

- 新增最小 `pytest.ini`：`testpaths=tests` + `norecursedirs`，使根目录 `pytest -q` 只收集 `tests/`，
  不扫描 `data/workdir/outputs/third_party/logs`。
- 静态质量：`ruff format` 全量格式化，`ruff check` 清零；带 shebang 的脚本设为可执行。
- `docs/Training_Log.md` 只保留真实训练事实与简短状态；待运行的操作命令统一集中到 `README.md`，
  避免重复的步骤文档。
- 新增测试合计后 `pytest -q` 共 **46 passed / 0 failed**（纯合成、CPU、不读真实数据）。

---

## 2026-09-21 — zonal-source 来源安全与 prior 复用校验

- `prepare_picai_nnunet.py zonal`：`--zonal-source` 改为**必填**（`required=True`，取消默认 `yuan`，
  仍限 `yuan/hevi`），消除“帮助文本说必须选、实际却有默认值”的不一致。
- 复用既有 PZ/TZ 前先做**来源/几何/内容校验**：两者都已存在且未 `--overwrite` 时，读取本次指定的
  `zonal_<source>.nii.gz`，校验其与 T2W 的 size/spacing/origin/direction，再由 PZ=1/TZ=2 派生预期
  float32 `[0,1]` 通道，逐项比对既有 PZ/TZ 的几何、形状、数值范围与内容；**完全一致才 `skipped`**，
  否则 `conflict`（来源/几何/内容不匹配）或 `failed`（不可读）。
- 只存在 PZ/TZ 之一且未 `--overwrite` → 判为 `conflict`，不覆盖已有文件、不补写缺失文件。
- **fail-closed**：任一 `conflict/failed` 先打印完整汇总再非零退出，且**不发布** `dataset.json`；
  仅显式 `--overwrite` 才重新原子生成 PZ/TZ（可用于来源切换或修复）。派生文件原子写入、原始数据只读、
  `noNorm`、与 T2W 网格一致、tqdm 与结构化汇总等约束不变。
- `dataset.json` 的 `description` 记录本次 `prior_source=zonal_<source>`，使来源可审计（不新增说明/元数据文件）。
- 新增纯合成测试（`tests/unit/test_prepare_picai_failclosed.py`）：缺 `--zonal-source` 时 argparse 非零退出；
  首次 `yuan` 生成五通道；来源一致时 `--resume` 可安全 `skipped`；yuan→hevi 内容不同且未 `--overwrite`
  非零退出且不改既有文件、不发布 dataset.json；只存在一个 prior 且未 `--overwrite` 非零退出且不生成
  缺失文件；`--overwrite` 可按新来源重新生成且内容正确；dataset.json 可审计 zonal 来源。


## 2026-10-06 — WG 定向追查与阶段一候选数据契约

只修改既有审计脚本、对应合成测试与两份获准文档；不创建 Dataset607、split，
不物化数据、不训练、不读真实影像、不实施 ROI 硬裁剪。

### 基准报告与 35 例原因

基准 `outputs/reports/prostate_anatomy_labels_audit_v2.json` SHA256：
`c88478a776f1c1dc61f6ab9890db7cf822248a92adc27c288d752953a876d60a`。
1464 例仅 Bosma 内容对应；35 例 no_comparable_candidate，另 1 例已知 WG 缺失。
35 例物化 WG 与 Bosma 几何均合法，size/spacing/origin 一致，唯一差异为 direction；
Bosma 因不同物理网格未比较，数组 shape/voxel_equal 尚未知。Guerbet 35 例均完成同网格
比较并 mismatch，差异 702–18260 体素。训练侧 33 例、验证侧 2 例。
这不能证实头信息转换或任何历史生成过程，不放宽原有绝对容差。

完整病例清单（基准未解决且非缺失）：
10001_1000001, 10055_1000055, 10062_1000062, 10190_1000193, 10244_1000248,
10268_1000272, 10395_1000401, 10432_1000440, 10458_1000466, 10459_1000467,
10533_1000543, 10555_1000567, 10577_1000590, 10601_1000615, 10634_1000649,
10644_1000660, 10656_1000672, 10729_1000745, 10790_1000806, 10822_1000838,
10842_1000858, 10845_1000861, 10863_1000879, 10907_1000924, 10909_1000926,
10939_1000958, 10996_1001015, 11011_1001031, 11030_1001050, 11154_1001177,
11263_1001286, 11373_1001396, 11379_1001402, 11454_1001478, 11474_1001498。

`--targeted-followup BASELINE_JSON` 独立入口绕过 Inputs.resolve/full audit，
只从基准未解决条目取病例和四类路径，不扫描全病例。已认可缺失条目单列且不读影像。
输出 scope=targeted_followup，基准字节 SHA256、逐例原始几何、SimpleITK/nibabel 版本、
RAS qform/sform 与 codes；物理位移用相同参考 voxel-centre 索引八角点计算 LPS 毫米差，
明确参考索引在不同尺寸的候选中可能越界。索引数组一致与物理网格一致分别报告。
显式 `--derived-resampling` 才用物理坐标、最近邻、外部值 0 在内存比较，标注派生结果，
不写标签。诊断完成不自动升级来源 resolved。全新 JSON 排他发布，拒绝覆盖；
tqdm、逐例诊断与成功/失败/已知缺失跳过/耗时/输出汇总。

### 首选候选契约（尚未接入训练）

保留原始 `wg.nii.gz` 和 `zonal_yuan.nii.gz` 不改动；PZ=(Yuan==1)、TZ=(Yuan==2)。
未来派生训练整数图 code=WG+2*PZ+4*TZ，仅严格二值且同形状可编码；物化时还必须
校验三个掩膜与 T2W 严格同网格、合法值与来源摘要，不能仅凭形状编码真实病例。

|整数|标注成员组合|
|---|---|
|0|均无|
|1|WG|
|2|PZ|
|3|WG+PZ|
|4|TZ|
|5|WG+TZ|
|6|PZ+TZ|
|7|WG+PZ+TZ|

6/7 支持一般组合与预测重叠；原始单张 Yuan 整数图的 PZ/TZ 互斥，不会产生 6/7。
这些整数是算法标注成员组合，尤其 WG 外分区不是新正常解剖类别。
原生 dataset labels 按插入顺序：background=0，WG=[1,3,5,7]，PZ=[2,3,6,7]，
TZ=[4,5,6,7]；regions_class_order=[1,2,4]。LabelManager 得到 3 输出头，
原生 ConvertSegmentationToRegions 得到独立三通道目标，原生 BCE+Dice 与 deep supervision；
patch validation 使用各头 sigmoid>0.5 与对应区域目标，不改变采样、optimizer 或主循环。

原生 export_prediction 的 correct-shape 函数先按 plans 重采样 logits，sigmoid 后恢复
preprocessing crop 和轴序；save_probabilities=True 的 npz 三通道为原始 T2W 数组网格，
结合 pkl properties 的 SimpleITK 几何可写三个浮点通道给阶段二，通道顺序固定 WG/PZ/TZ。
这需要读写适配和网格/有限值校验，不重新实现 sliding-window inference。
原生非零 crop 外区域概率补零须披露，不能称该处做过网络预测。

原生整数导出按 WG→PZ→TZ 顺序赋值，TZ 覆盖 PZ、PZ 覆盖 WG；只写 0/1/2/4，
丢失共同成员信息。用同一 label regions 评估时 WG 仅 code=1、PZ 仅 code=2、TZ 仅 code=4。
所以原生 whole-case summary 不是独立头指标。最小项目侧扩展：原生恢复概率后按 >0.5
重新位编码得到 0..7 派生整数图，并对原始位编码参考调用原生 region evaluator；
单独保留并命名 native_ordered_export 指标，避免混称。只增加概率后处理/评价适配，
不用复制训练器、网络、验证器或推理器。该扩展本次只用合成验证，未训练接入。

备选：8 类 softmax 原子分类，三个软预测由对应原子概率求和；无损存储参考，
但改变 CE+Dice 目标与输出头数，argmax 原子图与三个边缘概率 >0.5 仍可能不同，
仍需独立边缘概率导出与指标；因此不作为首选。直接三个二值文件不能直接作为
原生单 segmentation 训练输入；需要多目标加载与区域有效掩膜扩展，非最小首版。

### 缺失、患者隔离与范围

11050_1001070 无可用物化 WG；不能记作空 WG、不能用 PZ∪TZ 补成真实 WG。
首选完整监督候选集合为基准输入清单全部病例减此明确 ID：1499 例（train 1276，
validation 223）；35 例仍具有可用物化 WG，不因来源追查静默排除，来源历史仍待确认。
实际训练只能使用相应外层训练患者中的完整监督病例，不能将上述 validation 纳入训练。
阶段二继续保留全部 1500 例。
原生 ignore 是单张空间有效掩膜，对所有区域共享；不能仅忽略 WG 而保留同体素 PZ/TZ。
如部分监督，需要项目侧 region-specific validity 进入 loss 与 patch/whole-case 指标，
并审查 Dice 分母、deep supervision 和 region sampling；本次不实施。
未来折外分组基于全部 1500 例患者先分配，缺失病例患者也分配 held-out fold；该患者
所有检查都从对应模型训练中排除，仅完整监督训练患者贡献监督，然后该模型为缺失
病例推理。外层验证患者由仅外层训练患者训练的模型推理；不生成新 split。

### 合成验证与下一次接入位置

合成穷举 0..7：编码→原生区域转换→BCE+Dice→梯度→阈值→无损导出→区域 Dice；
WG 独有、内外分区、矛盾头、PZ/TZ 重叠全部覆盖。完美头无损导出 Dice=(1,1,1)，
原生顺序整数导出 Dice=(0.4,2/3,1)；人为矛盾头后独立 Dice=(6/7,8/9,8/9)。
目标正确时 loss 小于反向目标，梯度有限。定向合成测试验证基准限定、SHA、拒绝覆盖、
qform/sform、内存重采样与输入文件摘要不变、已知缺失不读影像、重复 ID fail-closed。

下一次经授权接入须检查/修改已有：`scripts/data/prepare_picai_nnunet.py`（单 T2W、
位编码与原生 region dataset 定义、显式缺失范围），`scripts/train/train_nnunet.py`
（解剖任务入口与概率保存），`src/zonal_reliability_fusion/nnunet/trainers.py`
（必要的薄适配，仍继承原生），`scripts/evaluate_segmentation.py`（独立头区域评价与
原生整数口径分离）。患者分组与概率通道消费接入相应既有 prepare/train 流程；
复用原生预测器与 export correct-shape，不新增模型变体。本次不改这些文件。

ROI 边界：现有标签 ROI 未完全漏失实例，但训练侧最差实例覆盖约 61.4%–69.0%；
不是阶段一模型结果。首版阶段二完整 FOV+预测软分区，ROI 留作训练侧选参的独立消融。
待用户运行确认：35 例索引内容、角点物理差、qform/sform 关系与可选派生比较；
这些仍不能单独证实历史标签生成过程。真实阶段一效果与训练接入尚未验证。


本次获准测试文件最终 **104 passed**（conda lm，固定 nnU-Net 2.6.2，CPU 合成数据）；
共 8 条上游依赖弃用警告（SWIG/SciPy/NumPy 接口）。补充原生 correct-shape 合成测试：2³ logits
重采样到 4³、恢复 crop、逆转轴序到原网格 (7,8,6)，三通道阈值及位编码均正确，
crop 外补零且数值有限。没有读取真实影像，真实追查尚未运行。


## 2026-10-06 — 阶段一 T2W 联合 WG/PZ/TZ 最小接入

本次只修改用户授权的 12 个既有文件，保留全部既有工作区修改，不新增源代码、测试、
配置或文档。只读取两份现有报告及 JSON/CSV 元数据、检查源码与执行 CPU 合成测试。
未读取真实医学影像，未创建 Dataset607 或 split，未执行真实准备、planning/preprocessing、
训练、validation、推理或评价；官方 planning 仅执行 --help。

### 已完成追查的事实与采用决定

读取 `outputs/reports/prostate_anatomy_labels_audit_v2.json` 与
`outputs/reports/prostate_anatomy_wg_followup_v1.json`。后一报告 success=35、failed=0、
skipped_known_missing=1；逐条确认 35 例物化 WG 与 T2W 同网格、与 Bosma 索引数组相等、
与 Bosma qform 相同，但 Bosma 读取 direction 不同。最大角点位移范围
0.0257517119–0.1813855134 mm；Bosma sform code=0，物化 WG/T2W sform code>0。
这支持内容对应和头信息差异，不证实历史生成来源。本轮采用既有物化 WG，保留这 35 例，
不修改标签、不重采样 Bosma、不放宽 1e-4 绝对几何容差（rtol=0），不改 source_resolved。

### 数据准备与冻结划分

`scripts/data/prepare_picai_nnunet.py` 新增 `anatomy`，固定 Dataset607_PICAI_Anatomy，
仅 T2W_0000。监督为 materialized wg==1 和 zonal_yuan==1/2，code=WG+2*PZ+4*TZ。
保留 background/WG/PZ/TZ 插入顺序与 region lists，regions_class_order=[1,2,4]。
组合整数描述算法标注成员，不是新正常解剖类别；不依赖真实参考出现 6/7，不读取 lesion。

必须提供 `--exclude-known-missing-wg`：只排除 11050_1001070，记录原因及 manifest、冻结
split SHA256。完整监督 1499 study、train=1276、validation=223（非患者数）。
检查 manifest 身份、重复、冻结 study 集合覆盖及患者隔离；额外缺失/非法病例逐例报错，
任何失败不发布新的 dataset.json。三维/有限/精确整数/合法值检查在编码前，三图均校验
合法物理网格并与 T2W 一致，不以 shape 代替几何。SimpleITK 可将存储 NaN 读成零，
故新路径通过 nibabel 的原始缩放数组检查非有限值，几何仍按既有 SimpleITK 定义比较。

禁止 overwrite/子集准备/另一个 Dataset 名。派生标签和 dataset.json 用排他发布，
原始文件只读，T2W 为链接。resume 核对配置、每个源文件 SHA256、链接目标、派生标签
数值/几何和剩余病例集合，不无条件 skipped；冲突保持既有内容。dry-run 读取/检查但不物化。
默认 tqdm；结束打印成功/失败/跳过/明确排除/耗时/输出，报告全部失败原因。
`splits` 对 607 增加独立过滤及 raw contract/磁盘集合检查，必须存在 preprocessed 目录，
写 train=1276/val=223；所有目标先检查冲突，排他发布，拒绝 overwrite。605/606 行为不改。

### Trainer、入口和概率

`trainers.py` 集中定义解剖契约与轻量校验，新增唯一
`nnUNetTrainerPICAI_AnatomyJoint_100ep_NoFFT`，MRO 是该类→NoFFT→原生 nnUNetTrainer。
不继承 FLCE/PositiveCaseSampling，不复制网络、训练器、验证器或推理器。
共享 plans 网络/encoder/decoder，原生三个 region sigmoid 输出、BCE+Dice、deep supervision、
采样、增强、optimizer/PolyLR、patch validation 与滑窗。只在原生构造后、initialize 前设置
num_epochs=100，scheduler max_steps=100；iterations/patch/batch 未改变。100 epochs 仅工程试跑。

构造/initialize 核验单 T2W、严格 regions/顺序、三头、Dataset607/3d_fullres、非 cascade、
明确 fold0 split；do_split 是薄校验，核对冻结病例范围/患者隔离/预处理 identifiers，不随机回退。
`train_nnunet.py` 注册 anatomy_joint_100ep，改用显式 SHORT_FUSION_VARIANTS 分组，
保留旧四个 Dataset606 限制。解剖必须保存验证概率；从头拒绝已存在运行目录；续训拒绝
不存在/不可读/缺字段/错误 Trainer、dataset provenance、fold、epoch 或 optimizer 的 checkpoint。
正式加载权重和续训仍由原生机制完成，不静默 fresh start。

`predict_nnunet.py` 注册新 Trainer，继续调用 native predictor/export。解剖模型要求
--save_probabilities、单 T2W 文件和全新输出；不支持本轮未接入的分片/cascade 扩展。
概率与整数的恢复全部调用原生 correct-shape：npz 为 `(3,Z,Y,X)`，顺序 WG/PZ/TZ，
pkl sitk_stuff 为物理几何，plans 提供 transpose，nii.gz 为覆盖式整数。crop 外补零须披露。
公共产物校验核对病例集、三个通道、shape、有限性、[0,1]、原始 shape/轴序及 pkl/参考/
原生导出几何；训练后的 validation 也调用该校验，不从整数反推软通道。

### 独立区域评价

`evaluate_segmentation.py anatomy` 是独立入口；旧病灶 summary/full 行为保留。
主指标从恢复概率分别 >0.5，无包含修正、无分区互斥，位编码后调用原生
region_or_label_to_mask/compute_tp_fp_fn_tn 机制计算各区域 Dice。
可选 native_ordered_export 单独报告；原生 TZ 覆盖 PZ、PZ 覆盖 WG，不能作为主评价输入。
每例/汇总报告 WG/PZ/TZ Dice 与 n_valid/n_empty_both/n_expected；双空为 null（均值排除并计数），
单空为 0。宏均值按有效 study 等权。PZ/TZ 重叠以分区并集为分母；PZ/TZ 超出 WG 各以
自身分区为分母，分区并集超出 WG 以并集为分母；每例和汇总体素分子/分母明确，零分母 null。

必须存在全部 223 例的概率、properties、整数预测；参考与 T2W 全 raw 集合应匹配 1499 例
契约，网格不符、非法值或病例不符显式失败，不缩分母后宣称完整成功。全部病例检查通过
才创建全新独立输出目录，保存 independent_threshold_bitcode/ 和 anatomy_regions.json；
不覆盖 native 预测/summary.json。此结果是算法伪标签一致性，不是人工解剖准确率。

### 验证、范围与后续

获准四份测试文件共 **235 passed**（12.43s，CPU 合成；173 条上游 SWIG/SciPy/NumPy 弃用
警告），AST 9 文件通过，git diff --check 通过，anatomy prepare/evaluation --help 通过。
覆盖位编码穷举、native region targets、原生三头网络、BCE+Dice/deep supervision 有限梯度、
100-epoch scheduler、MRO 无病灶损失/阳性采样、非整数/NaN cast 前失败、已知缺失显式排除、
额外缺失、同 shape 异网格、resume/拒绝覆盖、冻结 split 与患者隔离、缺失 split 不调用训练、
新 variant/预测解析及旧四分支限制、checkpoint 合法性、原生 crop/逆转轴序后的概率评价、
错误通道/非有限/越界概率、缺失病例/产物/pkl、异常 metadata/几何/参考的 fail-closed。
完美独立头 Dice=(1,1,1)，同一概率的 native ordered export Dice=(0.4,2/3,1)。

README 提供每步用户命令、lm/env 初始化、输出、进度与成功判据；Research_Plan 修订历史
排除表述，区分特征融合路线与当前两阶段候选，并强调解剖模块不能仅凭联合区域宣称创新。
未实现阶段二融合、ROI 裁剪、部分监督、折外批量预测或新采样策略。阶段二保留全部 1500，
缺失病例未来仍须由未见其患者的模型提供预测；本次只有冻结单 split 工程试跑接入。
真实源数据准备、plans/preprocessing、实际 GPU 训练/恢复、真实概率几何与实际区域效果
尚未验证；下一步须用户逐条执行 README 命令并返回结束日志，不自动连续启动。

最终准备回归另通过 **21 passed**；增加 resume 对区域插入顺序的显式核对，防止普通 dict
相等判断忽略顺序。原始监督与既有产物均未由代理改写。


## 2026-10-06 — 阶段一预测参数与验证输出保护修复

仅修改本轮获准的六个既有文件：predict_nnunet.py、train_nnunet.py、对应两份 unit 测试、
README.md、Development_Log.md，保留全部原有工作区修改，不创建新文件或改第三方源码。
未读取真实影像，未执行真实准备、训练、validation、推理或评价。

预测保护不再用 arguments.index 查找参数。新增 argparse 语法解析，覆盖固定 nnU-Net
model-folder 入口的选项、普通/等号/原生缩写写法；关键路径与受限参数一致处理，
缺值、空值、重复选项（包括别名指向同一选项）与歧义显式失败。
解剖模型的 save_probabilities/新输出/单通道检查和受限参数拒绝基于解析后的值，
通过后仍将原始 argv 原样委托 native entry point，不复制 predictor 或 export。
合法旧病灶入口（包括 continue_prediction 和既有输出目录）保持原生委托行为。
参数表对应固定 v2.6.2；未知选项 fail-closed，不允许借其绕过检查。

guard_anatomy_checkpoint 在调用原生 run_training 与构造 Trainer 前检查 validation 状态：
已有任何目录内容（包括部分概率、summary、嵌套目录）或不可用路径时拒绝所有会写入
该 validation 目录的操作；不删除、清空或移动既有结果，不增加另一个验证输出目录。
存在 checkpoint_final.pth 时拒绝 continue-training，避免再次进入训练结束保存流程。
未完成、无 validation 产物且有有效匹配 checkpoint 的正常续训继续允许；
已完成、无 validation 产物且最终 checkpoint 可用时允许 validation-only。
空 validation 目录可接受。原有 checkpoint 内容与契约校验继续执行，旧融合分支不改。

两份获准测试在 conda lm + scripts/env_nnunet.sh 下共 **139 passed / 7 warnings**，
耗时 6.68s；警告为既有上游 SWIG/SciPy 弃用警告。覆盖普通/等号参数的同一检查、
无 save_probabilities、已有输出、重复/缺值/空值/歧义、普通/等号受限参数、旧病灶 argv
原样委托；被拒绝时 mock native prediction 调用数为零。
训练测试模拟 fresh/continue/validation-only 与已有概率/summary/嵌套结果，完成后续训
也被拒绝；mock native run_training 调用数为零，并禁止构造 Trainer，前后文件集合及
字节内容完全相同。有效未完成续训/完成后首次验证分别在无目录和空目录状态通过 guard。
四个 Python 文件 AST 检查通过，git diff --check 通过。

README 已同步续训与首次验证限制。以上是合成保护逻辑验证；真实 checkpoint 恢复、
GPU 训练和首次 validation 未运行，实际运行行为仍需用户后续命令确认。


## 2026-10-06 — 现成分区融合输入的语义措辞修正

仅修改获准的 networks.py、trainers.py、README.md 与本开发记录，保留既有工作区修改。
Dataset606 PZ/TZ 统一描述为自动分区二值掩膜经既有插值处理形成的分区隶属权重，
不是校准概率，也不是实测几何体素占比。局部池化权重均值、局部平均支持权重及 coarse
网格权重和仅按算法加权统计解释。局部参照包含中心、可能包含病灶，不能称为正常组织真值。
修正同区参照模块的 docstring/相关注释、短预算 Trainer 初始化日志和 README 对应说明；
保留代码中的 occupancy/mass 变量名与全部计算。历史 dataset description 修订记录中的
旧术语引文保留，未修改任何 dataset.json 或产物。

AST 语法检查通过；剔除 docstring 并仅归一化获准日志字面量后，两份源码的计算 AST 与
本轮修改前完全相同。forward、网络结构、常量值、损失、采样、预算、类名与 state_dict
结构均未改变，阶段一执行行为未改变。git diff --check 通过。
未新增文件、未读取真实影像、未启动训练；本次仅措辞修改，不运行训练或数据处理测试。


## 2026-10-08 — 修复四个 `*_100ep` 变体构造阶段的 `KeyError: 'args'`

四个 100ep 变体（`feature_no_gate_positive_sampling_100ep`、`feature_anatomy_gate_positive_sampling_100ep`、
`zonal_reference_positive_sampling_100ep`、`zonal_reference_adaptive_positive_sampling_100ep`）在串行队列中
各自 5–7 秒内退出（exit 1），零 epoch，无任何 checkpoint 或输出目录。崩溃发生在 Trainer 构造阶段，
早于建网络与训练循环，与数据集、epoch 预算、`nnUNet_compile` 无关。

根因在固定版本 nnU-Net 的一处反射：`_ShortZonalFusionTrainer.__init__` 写作
`def __init__(self, *args, **kwargs)` 并转发给父类，而 `nnUNetTrainer.__init__` 用
`inspect.signature(self.__init__).parameters.keys()` 从**子类**签名取参数名，再从**父类**作用域的
`locals()` 取值；`*args`/`**kwargs` 带来的 `'args'`、`'kwargs'` 在父类 locals 中不存在 → `KeyError: 'args'`。
全仓仅此一处该写法，故恰好这四个类（均为 `_ShortZonalFusionTrainer` 子类）受影响，其余 Trainer 不受影响。

修复方式：删除 `_ShortZonalFusionTrainer.__init__`，把 spacing 校验与 `self.num_epochs = 100` 移入既有的
`initialize()` 覆写。不照抄父类完整签名，`my_init_kwargs` 保持原生，checkpoint 重建与 resume 行为不变。

时序依据（只读固定 v2.6.2 源码确认）：`self.initialize()` 由 `on_train_start()` 调用
（nnUNetTrainer.py 第 894–896 行），并非在 `__init__` 内；它早于 `configure_optimizers()`
（第 507 行以 `self.num_epochs` 构造 `PolyLRScheduler`）与训练循环上界 `range(self.current_epoch, self.num_epochs)`
（第 1365 行），因此 100 epoch 的训练循环与 PolyLR 总周期同步生效；`self.configuration_manager`
在 `__init__` 已就绪，spacing 校验语义不变；resume 时经 `load_checkpoint -> initialize()`
（第 1176 行）同样恢复该预算。`self.num_epochs = 100` 置于 `super().initialize()` 之前，
使整条基类 initialize 链均可见该预算。父类 `__init__` 第 149 行的 `num_iterations_per_epoch` 占位值不受影响。

未修改 `third_party/nnUNet`，未改动网络结构、损失、采样、融合方式、通道数与任何数值逻辑，
未新增文件，未读取真实影像。

验证仅限静态与反射：模块 import 成功；`_ShortZonalFusionTrainer` 不再定义自有 `__init__`；
四个 Trainer 的 `__init__` 均解析到 `nnUNetTrainer.__init__`，签名 `(self, plans, configuration, fold,
dataset_json, device)` 全为具名参数、无 var-args，故反射要取的每个名字都是父类自身局部变量，
`KeyError` 不再可能发生；文件 lint 无报错。真实训练能否跑完 100 epoch 仍需用户后续命令确认。
