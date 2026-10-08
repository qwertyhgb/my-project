# Repository Refactor Audit（2026-10-08）

> Phase 0 的只读审计产出。**本文件是重构前状态的快照**，之后不再更新；项目当前状态见
> `README.md` 与 `docs/Research_Plan.md`。
> 审计范围：`README.md`、`docs/`、`src/`、`scripts/`、`tests/`、`AGENTS.md`、`git status`、
> `git log --oneline -n 20`。审计期间未修改任何文件、未读取任何真实医学影像、未运行任何训练。

## 0. 审计时刻的仓库状态

- `git status --short`：**clean**（无未提交改动）；HEAD = `4128a78`。
- 跟踪文件 65 个。源码规模很小（`src/` 6 个 Python 文件），但**文档规模远大于代码**：
  `README.md` 799 行 / 60 KB，`docs/` 10 个 Markdown 共约 6600 行，其中
  `docs/Research_Plan_old.md` 1462 行、`docs/Development_Log.md` 1460 行。
- 未跟踪但存在的大体积目录：`workdir/`（22513 文件）、`data/`（13515 文件）、
  `outputs/`（2469 文件）、`logs/`（15 个历史日志）、`checkpoint_backup_B_yYi4MEKK/`
  （2 个 checkpoint，各约 340 MB）。
- `third_party/nnUNet` 只读，v2.6.2；`.gitignore` 已排除 `third_party/*`（仅保留 README）。

## 1. 发现的问题

### 1.1 结构性问题（最高优先级）

| # | 问题 | 证据 |
|---|---|---|
| S1 | **三条竞争性主线并存**：① 旧输入级 Gate 线、② 特征级融合线（自称"主线"）、③ 短预算同区参照 + 两阶段解剖候选 | `docs/Research_Plan.md` §4.4 与尾章给出两套互不引用的方案；`README.md` L10 首节介绍短预算分支 |
| S2 | **README 首页不是主线**：最新（2026-10-06）的短预算探索分支被放在文件最前，11 个旧 variant 混杂其后，读者无法判断哪个是主线 | `README.md` L10–L75 |
| S3 | **Trainer 数量口径三处不一致**：README 说"十一个"（L102/L124/L384）、"十五个"（L664），训练入口说"十六个" | `README.md:102,124,384,664`；`scripts/train/train_nnunet.py:34` |
| S4 | **`docs/Research_Plan_old.md` 孤立且无归档标识**：全文没有「已被取代 / 不要使用」标记，仓库内零处引用 | grep 无命中 |
| S5 | **`docs/Development_Log.md` 日期倒序被打乱**：`2026-10-06` 出现 5 次，`2026-10-08` 反在文件末尾 | `Development_Log.md:8,83,160,1281,1379,1414,1430` |
| S6 | 源码与文档的组织方式不一致：`scripts/evaluate_segmentation.py`（2352 行）同时承担指标定义与 CLI 编排；`src/zonal_reliability_fusion` 包名与当前研究内容（解剖引导病灶建模）已不匹配 | 全仓库 |

### 1.2 事实性错误 / 过期文档（必须修）

| # | 错误 | 位置 |
|---|---|---|
| C1 | `feature_anatomy_gate_positive_sampling` 被 README 写作"未运行"，实际已完成 1000 epoch + validation（Dice 0.20868） | README L240/L298/L640 vs Training_Log L18/L22-24 |
| C2 | `feature_image_gate_positive_sampling` 被 README 写作"未运行"，实际已训练到 epoch 81 后被用户停止 | README L240/L400/L640 vs Training_Log L20/L28-32 |
| C3 | `anatomy_gate`（legacy 606）无任何状态标注，实际已完成（Dice 0.18267） | Training_Log L19/L25-27/L125 |
| C4 | `Findings.md` 只承认"五次完成训练"，实际已有 7 次完成 validation | Findings L25 vs Training_Log L120-131 |
| C5 | 同一文件内自相矛盾：`anatomy_gate` 在 L125 记为"已完成"，在 L293 记为"训练未运行" | Training_Log |
| C6 | 6 处失效章节引用：`RQ4`、`Research Plan §8.10`、`§5.2/§11.5`、`§5「概念与解释边界」` —— 现行计划里这些章节都不存在 | `train_nnunet.py:25,29`、`Development_Log.md:476,478`、`experiments/image_gate.md:258` 等 |
| C7 | `README.md` 的「输出目录」小节与 `Training_Log.md` 的状态表逐行重复；README 首节命令与「训练与推理命令」小节重复 | README L622-644 等 |

### 1.3 测试层

审计时刻基线（`conda activate lm; python -m pytest -q`）：

- **26 个失败**，其中
  - 20 个因环境缺少 `medpy`（NSD/HD95 表面指标依赖）而失败 —— **环境问题，不是代码缺陷**；
  - 5 个 `test_short_zonal_*`：断言 `__init__` 里设置 `num_epochs=100`，但代码已把该逻辑移到
    `initialize()` —— **测试过期**；
  - 1 个 `test_train_entry_resolves_positive_sampling_variant`：硬编码 variant 集合，
    缺少 `anatomy_joint_100ep` —— **测试过期**。
- 测试通过数与 `Development_Log` 里记录的 473 一致量级，说明测试本身健康。

### 1.4 生成型垃圾（GENERATED）

- `src/`、`scripts/`、`tests/` 下 8 个 `__pycache__` 目录共 38 个 `.pyc`；
- `.pytest_cache/`（92 KB）、`.ruff_cache/`（64 KB）；
- `logs/` 下 15 个历史日志与 1 个 `.pid`（**历史记录，保留**，已在 `.gitignore` 中排除）；
- `outputs/diagnostics/code_snapshots/g0_r_v02_before_fix_20260918_081501/`（旧阶段代码快照，
  **历史记录，保留**）。

### 1.5 需要保护、不得改动的资产

- **已有 checkpoint / validation 产物**（输出目录由 Trainer 类名决定，改名即失联）：
  Dataset605：`nnUNetTrainerPICAI_FLCE_NoFFT`、`..._DiceCE_NoFFT`、`..._FLCE_PositiveSampling_NoFFT`、
  `..._ImageGate`、`..._ImageGate_PositiveSampling_NoFFT`、`..._FeatureNoGate_PositiveSampling_NoFFT`、
  `..._FeatureImageGate_PositiveSampling_NoFFT`；
  Dataset606：`..._AnatomyGate`、`..._AnatomyGate_PositiveSampling_NoFFT`、
  `..._FeatureAnatomyGate_PositiveSampling_NoFFT`；
  legacy M0：`outputs/checkpoints/m0_resenc/**`（4 个 checkpoint，驱动代码已删除）。
- `workdir/nnUNet_preprocessed/`：Dataset605、Dataset606、Dataset607_PICAI_Anatomy 均已物化
  （607：1499 例 study、1499 个 `.pkl`、`splits_final.json`、`nnUNetPlans.json`）。
- ⚠️ **审计更正（同日发现）**：本报告初版依据文档层摘要写"Dataset607 没有对应的训练输出目录、
  Stage-1 从未训练"，**这是错误的**。实际核查 `outputs/nnUNet_results/Dataset607_PICAI_Anatomy/`
  显示 Stage-1 **已完成 100 epoch 训练 + validation**（2026-10-06 11:02 → 12:38 UTC），有
  `checkpoint_final.pth` / `checkpoint_best.pth` / `validation/summary.json`（223 例）与
  `validation/*.npz`（soft prior）。
  逐区域 Dice：**WG 0.0056** / PZ 0.8985 / TZ 0.9361（集合平均 0.6134）。
  这一错误本身是本次审计最有价值的发现：**文档层摘要与产物层事实不一致**，说明"以文档为准"
  的审计方法不可靠，必须核对产物。事实已更正到 `docs/Training_Log.md` 与 `README.md`。
- `data/`、`outputs/`、`logs/` 的既有内容一律只读。

## 2. 分类结果（KEEP / ARCHIVE / REWRITE / DELETE）

### KEEP（新主线继续使用）

| 路径 | 角色 |
|---|---|
| `src/zonal_reliability_fusion/nnunet/runtime.py` | 固定 nnU-Net 运行时校验 |
| `src/.../nnunet/losses.py`（由 `trainers.py` 拆出） | PI-CAI Focal+CE（strong baseline A1） |
| `src/.../nnunet/augmentation.py`（同上） | NoFFT 兼容修复 |
| `src/.../sampling/positive_sampling.py` | **strong baseline 的组成部分**（不是创新点） |
| `src/.../anatomy/contracts.py` | Stage-1 数据集契约（从 `trainers.py` 迁出） |
| `scripts/train/train_nnunet.py` | 唯一训练入口（AGENTS.md §5 强制） |
| `scripts/inference/predict_nnunet.py` | 官方预测入口 + Trainer 解析 |
| `scripts/evaluate_segmentation.py` | 论文级评估入口（指标实现要迁到 `src/`） |
| `scripts/evaluate_external_segmentation.py` | Prostate158 外测评估 |
| `scripts/data/prepare_picai_nnunet.py` | Dataset605/606/607 数据准备 |
| `scripts/data/prepare_prostate158_external.py` | 外测数据准备 |
| `scripts/data/audit_prostate_anatomy_labels.py` | 解剖标签审计（Stage-1 的依据） |
| `tests/unit/test_evaluate_segmentation.py` | 病灶实例级指标的 2000+ 行测试（核心资产） |
| `tests/unit/test_positive_case_sampling.py` | 阳性采样测试 |
| `tests/unit/test_nnunet_trainers.py` / `test_nnunet_networks.py` | Trainer/网络测试 |
| `tests/unit/test_prepare_picai_failclosed.py` | 数据准备 fail-closed 测试 |
| `tests/unit/test_predict_entry.py` | 预测入口测试 |
| `tests/unit/test_prior_transforms.py` | 先验通道增强边界测试 |
| `tests/unit/test_prostate158_external.py` | 外测测试 |
| `pytest.ini` / `tests/conftest.py` | 测试配置与夹具 |
| `scripts/env_nnunet.sh` | 环境脚本（校验环境与固定源码） |
| `data/metadata/*`、`data/splits/*` | 小体积清单，历史事实 |

### ARCHIVE（有历史实验或 checkpoint 依赖，移动到归档位置）

| 原路径 | 归档位置 | 原因 |
|---|---|---|
| `src/.../nnunet/networks.py` 的 gate/fusion 类 | `src/.../legacy/fusion_networks.py` | checkpoint 兼容 + negative evidence |
| `src/.../nnunet/transforms.py` | `src/.../legacy/prior_transforms.py` | 行为契约被新主线继承，实现归档 |
| `src/.../nnunet/trainers.py` 中的 13 个 legacy Trainer | `src/.../legacy/fusion_trainers.py` | 类名决定输出目录，逐字保留 |
| `docs/Research_Plan_old.md` | `docs/archive/old_research_plans/` | 旧主线，加归档头 |
| `docs/experiments/image_gate.md`、`image_gate_positive_sampling.md`、`anatomy_gate_positive_sampling.md` | `docs/archive/historical_experiments/` | 旧 Gate 线实验 |
| `scripts/data/audit_dataset605_606_mri_equivalence.py` | 原位保留（脚本），文档标注 LEGACY | 旧 RQ2 归因审计；数据集事实仍有效 |

### REWRITE（必须重写）

| 路径 | 理由 |
|---|---|
| `README.md` | 三条主线混杂、事实错误 C1–C3、命中 C7 重复 |
| `docs/Research_Plan.md` | 必须只保留单一主线（新问题、Stage 1/2、四个条件、评价体系、stop rules） |
| `docs/Findings.md` | 需要重新分类为 Preliminary Findings，并修 C4（run 集合） |
| `AGENTS.md` | §6 文档纪律需与新的五份治理文档一致（新增 `Evaluation_Protocol.md` 等） |

### DELETE（dead / generated）

| 路径 | 理由 |
|---|---|
| `__pycache__/`、`*.pyc` | 生成物 |
| `.pytest_cache/`、`.ruff_cache/` | 本地缓存 |
| （无源码可删） | 审计未发现"无 import、无测试、无 checkpoint 兼容需要、无历史复现价值"的源码 |

## 3. 审计结论（对重构的直接影响）

1. **研究逻辑重构优先**：`src/` 只有 6 个文件，重命名的收益极低而风险高（checkpoint 输出目录
   由类名决定）。因此保留 `zonal_reliability_fusion` 包名与全部历史 Trainer 类名，只在**内部
   分层**：active 进 `anatomy/` `lesion/` `sampling/` `evaluation/`，legacy 进 `legacy/`。
2. **最大的实际问题不是代码而是文档**：C1–C6 都是"文档说了与事实不符的话"。本轮必须让
   README / Research Plan / Training Log / Findings 四者口径一致，并建立单一 variant 状态真源。
3. **Stage-1 已经训练完毕，但 WG 区域头不可用**（逐区域 Dice 0.0056 / 0.8985 / 0.9361）：这决定
   Phase 3 的重点不是"把解剖管线接起来"，而是**先修 WG 预测**——因为条件 B 的 ROI 唯一来自
   predicted WG，WG 不可用会让 ROI 退化为"没有 ROI"。
4. **`evaluate_segmentation.py` 是必须搬进 `src/` 的核心资产**：病灶实例级匹配、大小分层、
   bootstrap 配对比较是论文评价体系本身，不应该藏在 2352 行的脚本里。
5. **环境缺 `medpy` 是 20 个测试失败的唯一原因**：属于环境补齐，不是代码缺陷 —— 但它说明项目
   从未在干净环境上跑过一次完整测试；`pyproject.toml` 的依赖声明是必要的。
