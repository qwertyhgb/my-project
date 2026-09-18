# PI-CAI 模型就绪数据审计报告（G0-P）

> **状态说明（2026-09-18）**：本文件是 G0-P 数据就绪的**历史审计证据**，不负责定义 v2.3 模型架构或当前训练门。
> G0-P PASS 继续有效；正式 M0 已从旧 PlainConv 升级为 Residual-Encoder（v2.3 P2A 已实现并通过合成回归；G2 第 8 项
> 真实数据验证仍 Pending），正式 N0 启动也须同时满足 G0-R PASS 与 G0-E 路径冻结。
> **当前状态、blocker 与执行顺序的唯一权威来源是 `docs/STATUS.md`**（协议条款仍以 `docs/research_plan.md` v2.3.1 为准）；
> 本文件涉及的数据事实（1500 例病例级物化、canonical 标签、split、nnU-Net 预处理）不受后续代码进展影响。
>
> **后续进展索引（2026-09-17/18，均不改变本报告的 G0-P 判定）**：
> ① M1–M4 融合模型代码已实现（合成测试，**未训练**）；② plan-space PZ/TZ prior 物化链与深度校验已实现，
> 真实 3 例 dry-run 与**全量 1500 例物化**（2026-09-17，`ok=1500 failed=0`，canonical manifest 已发布；B10 关闭）
> 均已由研究者执行完成（`docs/STATUS.md` B10、`docs/experiment_log.md`）；③ seg 读取适配已对齐 nnU-Net 的 `-1` 填充哨兵语义
> （返回严格 `uint8 ⊆ {0,1}`）；④ `--overfit` 运行时峰值显存遥测已实现（报告落
> `outputs/diagnostics/<model_id>/<run_name>/diagnostics/gpu_memory_profile.json`，静态 `NOT_MEASURED` 语义不变）。

> 项目：`/opt/data/private/lm/my-projects`　数据：`/opt/data/private/lm/data/Prostate/PI-CAI`
> 依据：`docs/research_plan.md` v2.0 §6「当前 G0 状态与训练前条件」、§14「G0-P：PI-CAI 数据可用」
> 环境：conda `lm`（Python 3.10.6 · SimpleITK 2.5.3 · numpy 2.2.6 · pandas 2.3.3 · scipy 1.15.3 · nnunetv2 2.6.2）
> 更新：2026-09-14（P0A 审计修正 + P0B 数据物化 + 全量验收与 N0 前置 `splits_final.json`）。旧版报告归档于
> `outputs/diagnostics/picai_model_readiness/archive_pre_p0a/PICAI_Model_Readiness_Audit_pre_p0a.md`。
> 执行说明：本报告所列命令均已实际运行（见 `docs/experiment_log.md`）；原始数据只读；未训练任何模型；未修改 `third_party/`。

---

## 1. 摘要与 G0-P 结论

依据 research_plan v2.0 §6 的 8 项训练前条件，逐项状态：

| # | 条件 | 状态 | 证据 |
|:--|:--|:--|:--|
| 1 | 修正方向码（单位 direction = LPS，非 RAS） | ✅ 完成 | §3.1；单元测试 19 passed |
| 2 | 重新生成含 orientation 的 manifest 与 geometry audit | ✅ 完成 | §3.2/§3.3；仅 orientation 列变化 |
| 3 | ADC/HBV 按物理坐标重采样至冻结参考网格 | ✅ 完成（1500/1500） | §5.4；0 失败，27.3 GB |
| 4 | 生成覆盖 1500 study 的 canonical 二值病灶标签 | ✅ 完成（425 阳性 / 1075 阴性） | §5.4；阳性 0 丢失、阴性 0 异常、取值全合法 |
| 5 | 5 例 geometry_suspect 重采样后人工 QC 并记录决定 | ✅ 完成（全部保留） | §6 |
| 6 | 所有 WG 依赖环节屏蔽 `11050_1001070` 异常 WG | ✅ 完成 | §5.3/§5.4；全量 `wg_status=excluded_known_faulty_bosma22b`，不替换 |
| 7 | 生成 `picai_train_val_split.json` 并复核患者泄漏 | ✅ 完成 | §7；与旧划分列表一致，无泄漏 |
| 8 | nnU-Net raw 模态顺序/标签/校验值冻结 | ✅ 完成 | §8；全量完整性 PASS、3d_fullres 预处理 1500/1500、`splits_final.json` 14/14 PASS |

- **当前判定：G0-P = PASS**（2026-09-14，条件 1–8 全部完成）；证据：§5.4（全量物化/审计/完整性校验）、
> §8（nnU-Net 全量完整性、3d_fullres 预处理完成、`splits_final.json` 14/14 校验 PASS）、§11（最终判定）。
- **外部测试：G0-E = Pending**（§9）。当前没有任何外部数据集被冻结为 test；私有 PCA 的 ROI 语义、
> 病理标准与 csPCa 定义尚未确认，**不能**作为 csPCa 外部金标准或已就绪的测试集。
- 本报告所依据的首轮审计结论（manifest 数字、几何分布、分区质控）在 §4 保留；其中与 v2.0 不符的
> 表述（“跨中心划分”“域泛化前提”等）已删除或改写。

---

## 2. 执行范围与方法

- 原始数据全程只读；派生数据写入 `data/processed/`、`data/metadata/`、`outputs/`；未修改 `third_party/nnUNet`
> （v2.6.2，commit `74ceb6803d10dcee29b2cc481678d3a3d069f281`）。
- 所有空间计算基于 **SimpleITK index-to-physical-point**（8 角点物理包围盒），不使用 size×spacing 简化；
> 跨网格比较/重采样一律在物理坐标下进行（影像线性、标签最近邻）。
- 方向编码统一使用 SimpleITK/ITK 的 DICOM **LPS** 约定（§3.1）。
- 本轮（P0A+P0B）只做审计修正与数据物化：不重跑分区质量分析（v2.0 §6 明确不需要）、不训练模型、
> 不查看任何外部数据上的模型结果。
- 所有长任务带 tqdm 进度条并输出结构化摘要（AGENTS.md §1）。

---

## 3. P0A：方向码修正与审计重建

### 3.1 orientation 修正（RAS→LPS）

**问题**：`src/zonal_reliability_fusion/data/picai.py` 的 `orientation_code` 手工把单位 direction 解释为
RAS 约定。SimpleITK/ITK 物理坐标为 DICOM **LPS** 约定，单位 direction 应返回 `LPS`。

**修正**：改用 SimpleITK 官方接口
`sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(direction)`（传入扁平 9 元组）。
仅修正方向字符串报告，**不对影像做任何重定向或翻转**。实测行为（SimpleITK 2.5.3）：

| 输入 direction | 输出 |
|:--|:--|
| 单位阵 | `LPS` |
| x 轴翻转（+x 指向 R） | `RPS` |
| 绕 z 旋转 20°（斜扫） | `LPS`（合法三字符码：L/R、P/A、I/S 各出现一次） |

**测试**：`tests/unit/test_picai.py` 修正旧断言（`RAS`/`LAS` → `LPS`/`RPS`），新增斜扫合法码用例；
`PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider tests/unit/test_picai.py -q` → **19 passed**。

### 3.2 manifest 重建与差分验证

命令：`python scripts/data/build_picai_manifest.py`（耗时 201.5s，0 错误，`all_pass=True`）。
与 `archive_pre_p0a/` 旧版逐列比对：

- 1500 行、列结构、case_id 集合完全一致；validation 检查项完全一致（1500/1476/1075/425/220/205；
  RUMC 800(236)/PCNN 350(109)/ZGT 350(80) 全部通过）。
- **仅 7 个 `*_orient` 列发生变化**（t2w/adc/hbv/wg_bosma22b/zonal_yuan23/zonal_heviai23/wg_guerbet23，
  各 1500 例全部 `RAS→LPS`）；其余列（含标签值、前景体素、网格分类）零变化。

验证证据：`outputs/diagnostics/picai_model_readiness/p0a_orientation_fix_verification.json`。

### 3.3 geometry 审计重建与差分验证

命令：`python scripts/audit/audit_picai_geometry.py`（耗时 500.5s，10500 行，0 错误；异常表 1054 行）。
与旧版逐列比对：

- 10500 行、case×role 键集合完全一致；**仅 `orientation` 列变化（10500 行 RAS→LPS）**；
  `grid_status` 分布与 `severity` 完全一致（即几何分类统计未被本次修正影响）。

验证证据：`outputs/diagnostics/picai_model_readiness/p0a_geometry_orientation_fix_verification.json`。

### 3.4 历史产物归档

P0A 之前的 manifest/validation/geometry 产物与旧版审计报告已归档至
`outputs/diagnostics/picai_model_readiness/archive_pre_p0a/`，未删除任何历史审计产物。

---

## 4. 首轮审计结论（摘要保留；本轮未重跑）

以下为 2026-09-14 首轮审计的结论（分区质量与 QC 部分未在本轮重跑）：

- **canonical 病灶标签**：`Pooch25(205) ∪ human_expert/resampled(1295)`；resampled 的 `2–5` 与 Pooch25 的 `1` 二值化为前景 1，
> 阴性保持全 0。
- **几何事实**：ADC/HBV 与 T2W 属于“物理空间对应但采样网格不同/部分重叠”（T2W 平面内 0.30–0.63 mm vs ADC/HBV 0.86–2.51 mm），
> 融合前必须物理坐标重采样；HeviAI23 有 346 例与 T2W 部分重叠（需 NN 重采样）；canonical lesion 9 例非 exact。
- **分区/腺体**（全量）：0 空 mask、0 非法标签、0 PZ-TZ 重叠；Yuan PZ∪TZ vs WG Dice 中位 0.949、Hevi 0.939、
> PZ Dice(Yuan|Hevi) 中位 0.891；`11050_1001070` 的 WG 与 PZ∪TZ Dice=0（官方 README 亦标注 faulty，禁用）。
- **canonical zonal prior**：Yuan23（主）；HeviAI23（敏感性）；本地无真实 softmax，第一阶段为 hard prior。
- 需关注（保留+标注）：5 例 ADC/HBV `geometry_suspect`（本报告 §6 给出重采样后决定）、WG-Dice≤0.7 的 6 例、
> 低 PZ 一致性 `11274_1001297`。

---

## 5. P0B：数据物化

### 5.1 规则与产物布局

脚本：`scripts/data/materialize_picai.py`。对每个 study（原始数据只读）：

- 以该例**轴位 T2W 完整物理网格**（size/spacing/origin/direction）作为病例内唯一参考；
- ADC/HBV 按物理坐标线性重采样到 T2W 网格，round+clip 后存 **uint16**；
- canonical 病灶标签先二值化（`2–5/1→1`），最近邻重采样到 T2W 网格（uint8）；
- Yuan23（主）/HeviAI23（敏感性）最近邻重采样，保持 `{0,1,2}`（uint8）；
- WG（Bosma22b，仅用于后续采样/QC）最近邻重采样；`11050_1001070` **禁用且不替换**（显式记录）；
- 每个文件先写 `cases/<id>/_tmp/<name>.nii.gz`，读回校验（几何一致/唯一值/阳性不丢失/阴性不凭空出现）后
> `os.replace` 原子重命名；
- 支持 `--dry-run / --limit / --cases / --resume / --overwrite / --verify`、运行前磁盘估算、逐例失败记录、
> 结构化 `materialization_report.csv` 与 `materialization_summary.json`。

产物：`data/processed/picai/cases/<case_id>/{t2w,adc,hbv,lesion,zonal_yuan,zonal_hevi[,wg]}.nii.gz`。

### 5.2 磁盘估算

`--dry-run`（1500 例）：参考网格总预算体素 ≈1.29×10¹⁰；未压缩 ≈128.9 GB；按保守压缩率预计 ≈70.9 GB；
剩余空间 ≈26.8 TB（`/opt/data/private` 挂载），headroom_ok=True。实测物化体积远低于保守估算（见 §5.4）。

### 5.3 smoke test（11 例）与单例验证

覆盖：阳性、阴性、Pooch25、非 exact lesion grid、全部 5 例 `geometry_suspect`、三中心、WG 禁用例。

| 项目 | 结果 |
|:--|:--|
| 11 例 smoke（`--cases ...`） | 11/11 ok；validation_ok 全 True；耗时 31.9s（重建） |
| `11050_1001070` 单例（`--cases`） | ok；`wg_status=excluded_known_faulty_bosma22b`；目录仅 6 个文件（无 wg.nii.gz） |
| 非 exact lesion grid 阳性例体积守恒 | `10459_1000467`：1.377→1.381 mL（1.003×，0.5→0.3 mm 网格）；`10408_1000415`：0.148→0.148 mL（1.000×） |
| 逐体素一致性抽查（exact-grid） | 3 例 t2w 与 canonical lesion 二值图**逐体素一致**（如 478→478、444→444、0→0） |
| 中途审计（306 例时，`audit_materialization_report.py --allow-incomplete`） | all_pass=True：0 失败；73 阳性 0 丢失；233 阴性 0 异常；299 例 exact-grid fg 与 manifest 完全一致；标签值全部合法；WG 禁用例正确 |

### 5.4 全量物化结果（已完成）

命令：`python scripts/data/materialize_picai.py --resume`（研究者执行；后台运行，tqdm 进度；日志 `logs/materialize_full_p0b.log`）

| 项目 | 结果 |
|:--|:--|
| 完成情况 | **1500/1500 ok，0 失败，0 跳过**（`n_selected=1500, n_ok=1500, n_failed=0, n_skipped_complete=0`） |
| 总耗时 | **4799.4 s（≈80 分钟）** |
| 磁盘占用 | **27,305,223,089 B ≈ 27.3 GB**（`du` 计 26 GiB） |
| WG 禁用例 | `11050_1001070` → `wg_status=excluded_known_faulty_bosma22b`，不替换、不静默排除 |
| 全量审计（`audit_materialization_report.py`） | **all_pass=True**（11 项检查全通过）：覆盖 1500/1500、0 失败行、`validation_ok` 1500/1500、425 阳性 0 丢失、1075 阴性 0 异常、lesion/Yuan/Hevi/WG 取值全合法、**1491 例 exact-grid 前景体素与 manifest 完全一致**、WG 禁用例正确 |
| 全量完整性校验（`materialize_picai.py --verify`） | `{'complete': 1500}`、`all_complete=True`、`problem_cases=[]`（日志 `logs/materialize_verify_p0b.log`） |
| 产物 | `materialization_report.csv`、`materialization_summary.json`、`materialization_audit.json`、`materialization_validation.json/csv` |

---

## 6. 物化后 QC 与 geometry_suspect 决定

QC 脚本：`scripts/audit/generate_picai_materialized_qc.py`（读取物化产物；T2W 网格物理空间叠加 +
T2W|ADC、T2W|HBV 棋盘格 + 覆盖度统计；阳性例优先显示病灶层）。图：`qc_figures_post_p0b/`。

**5 例 `geometry_suspect`（均为阴性病例）逐张目检结论：全部保留并标注。**

| case_id | ADC/HBV 视野外零值占比（全图） | 腺体内 ADC/HBV 零值占比 | 目检结论 |
|:--|--:|--:|:--|
| `10057_1000057` | 2.7% / 3.2% | 0.0000 / 0.0000 | 对齐正常，保留 |
| `10161_1000164` | 0.1% / 0.0% | 0.0000 / 0.0000 | 对齐正常，保留 |
| `10489_1000497` | 26.5% / 33.4% | 0.0000 / 0.0000 | 视野缺口在腺体外，保留并标注 partial FOV |
| `10805_1000821` | 8.1% / 13.6% | 0.0000 / 0.0000 | 对齐正常，保留 |
| `11414_1001438` | 8.6% / 9.2% | 0.0000 / 0.0000 | 对齐正常，保留 |

- 结论与依据：重采样后**腺体（WG）内 ADC/HBV 覆盖度 100%**；视野外缺失均在腺体外；分区与叠加图一致；
> 5 例均为阴性（空病灶），不存在病灶丢失问题。故不需要修复或排除，保留并在后续模型分析中作为
> “partial FOV” 标注病例。
- 追加 QC：`10459_1000467`（阳性 + 非 exact lesion grid，病灶红轮廓可见、位于 PZ 区、落于 WG 内）、
> `10000_1000000`（阴性对照）均正常。

---

## 7. train/validation 划分（v2.0 协议）

- 命令：`python scripts/data/create_picai_splits.py --compare-legacy data/splits/picai_cross_center_splits.json`
- 新文件：`data/splits/picai_train_val_split.json`；旧文件
> `picai_cross_center_splits.json` 保留（历史追踪，不用于训练）。

| 集合 | Patients | Studies | 阳性患者 | 按中心（n / pos） |
|:--|--:|--:|--:|:--|
| train | 1256 | 1277 | 362 | RUMC 674/201 · PCNN 288/93 · ZGT 294/68 |
| validation | 220 | 223 | 63 | RUMC 118/35 · PCNN 50/16 · ZGT 52/12 |

- 口径：PI-CAI 三个中心共同进入 train/validation（中心仅作分层因素）；patient-level；
  `center × patient_csPCa` 分层；seed=0；val_ratio=0.15；PI-CAI 内部不设 test。
- 验证：**与旧文件 patient/study 列表逐项一致（legacy check pass=True，四组列表全等）**；
> 无患者泄漏（`all_splits_pass_leak_check=True`）；23 名多 study 患者全部单集合（violations=[]）；
> 旧文件 md5 未变化。校验 JSON：`data/splits/picai_train_val_split_vs_legacy_check.json`。

---

## 8. nnU-Net raw 准备与格式检查

- 工作区：**项目内**（`workdir/nnUNet_raw`、`workdir/nnUNet_preprocessed`、`outputs/nnUNet_results`），
  由 `scripts/env_nnunet.sh` 固定（覆盖 `.bashrc` 全局变量；`PYTHONPATH` 前置 `third_party/nnUNet` v2.6.2）。
- 数据集：**`Dataset605_PICAI`**（ID 605，已检查无冲突；001–010 为 MSD 保留，501/604 为其他项目）。
- 结构：`imagesTr/<case>_0000.nii.gz→t2w`、`_0001→adc`、`_0002→hbv`（**0000=T2W、0001=ADC、0002=HBV**，
  固定顺序）；`labelsTr/<case>.nii.gz→lesion`（二值）；均为**相对符号链接**，不复制体素。
- `dataset.json`：`channel_names` 0/1/2 = T2W/ADC/HBV；`labels` background/lesion；`file_ending=.nii.gz`；
  `numTraining` 与实际病例数一致（完整性检查要求严格相等）。
- 病例映射：`data/metadata/picai_nnunet_case_mapping.csv`；准备摘要：
  `data/metadata/picai_nnunet_raw_prep_summary.json`。
- **格式完整性检查（子集 11 例）**：自动 Reader = **SimpleITKIO**（与项目 SimpleITK 全链路一致）；
  `nnunetv2.experiment_planning.verify_dataset_integrity.verify_dataset_integrity(...)` 无错误通过。
- **全量链接**：1500/1500（imagesTr 4500 + labelsTr 1500，相对 symlink），`dataset.json` `numTraining=1500`。
- **全量完整性检查**：`verify_dataset_integrity` **PASS（0 错误）**。注：该次运行发生在
  `scripts/env_nnunet.sh` 建立之前，lm 默认导入为 AlignThenRefine fork；已逐文件比对确认该 fork 的
  `experiment_planning/`、`preprocessing/`、`imageio/SimpleITKIO` 与固定 v2.6.2 在本项目用到的代码路径上
  源码一致（仅多一个未被使用的 `SimpleITKIOWithReorient` 类），结论不受影响；后续运行统一先
  `source scripts/env_nnunet.sh`。
- **plan/preprocess（3d_fullres）**：研究者执行（`source scripts/env_nnunet.sh`），**已完成**：1500/1500，
  0 Error / 0 Traceback / 0 Warning（研究者核验）；每例产物为 1 个影像 `.b2nd` + 1 个 `_seg.b2nd` + 1 个 `.pkl`
> （各 1500、无空文件、三类文件病例 ID 一一对应）；`dataset_fingerprint.json`、`nnUNetPlans.json` 齐备；
> 预处理目录 ≈45 GB。规划结果：`patch_size=(16,320,320)`、`spacing=(3.0,0.5,0.5)`、`batch_size=2`、
> `DefaultPreprocessor`（日志 `logs/nnunet_plan_preprocess_p0b.log`）。
- **`splits_final.json`（单 fold 冻结划分）**：`scripts/data/create_nnunet_splits.py`（新增）生成
> `workdir/nnUNet_preprocessed/Dataset605_PICAI/splits_final.json`（`[{"train": …, "val": …}]`，
> **train 1277 / val 223**），并在 `workdir/nnUNet_raw/Dataset605_PICAI/` 写同一内容副本（nnU-Net 的 plan
> 阶段会把 raw 中的该文件复制到 preprocessed，可防止将来重新规划时丢失）；两份 sha256 一致（`db1d838d…`）。
> **校验 14/14 PASS**：两侧列表无重复、无交集、与声明数一致（1277/223、1256/220）、train ∪ validation ==
> 预处理 case 集合（`gt_segmentations` 与 `nnUNetPlans_3d_fullres` 双重比对，各 1500）、23 名多 study
> 患者全部单侧、回读 fold 数=1 且内容一致。证据：`data/splits/picai_nnunet_splits_validation.json`、
> `data/splits/picai_nnunet_split_cases.csv`（1500 行）。
> 注：若该文件缺失，`nnUNetTrainer.do_split()` 会自行生成默认 5-fold（seed=12345），既不使用冻结划分，
> 也可能把同一患者的不同 study 分到 train / val 两侧。

---

## 9. 外部数据状态（G0-E = Pending）

- 当前**没有任何外部数据集被冻结为最终 test**；外部评估尚未执行，也不得查看任何外部数据上的模型结果。
- 候选与状态（research_plan v2.0 §4.5/§5.4）：
  - **Prostate158**：外部测试首选候选；仍需核对肿瘤定义、DWI/HBV 对应、PZ/TZ 标签映射与先验公平性；
  - **私有 PCA**：ROI 语义、病理标准、病例级诊断与任务映射**未确认**，当前不可作为 csPCa 外部金标准；
  - 私有 BPH：ROI 语义待确认，不作为病灶分割测试集；MSD_Prostate：解剖分区数据，不作为病灶测试集；
    ProstateX 与 PI-CAI 存在病例映射，不得作为独立外部集。
- G0-E 通过条件（语义与几何审计、患者重叠排查、先验协议与指标范围冻结）见 research_plan v2.0 §5.4。

---

## 10. 未解决问题与遗留风险

1. `marksheet.csv` 无 scanner/vendor 字段；域分析只能依赖 center（v2.0 已不以域为研究变量）。
2. 两套 zonal 均为 AI 结果；分区分层结论不得表述为人工解剖金标准。
3. HeviAI23 真实 softmax（Zenodo）未获取；软先验仅为可选扩展。
4. 低 WG-一致性病例（172 例 ≤0.9）未逐例人工复核；M4 阶段如性能异常应优先排查该子集。
5. `10489_1000497` 等 partial FOV 病例已保留并标注；后续训练采样/FOV 处理需留意。
6. G0-E（外部测试就绪）未完成，正式 M0–M4 比较与外部测试前必须完成。
7. 运行 nnU-Net 必须显式设置 `nnUNet_raw` / `nnUNet_preprocessed` / `nnUNet_results`（推荐 `source scripts/env_nnunet.sh`）：
   shell 全局变量若仍指向已移除的旧路径（`/opt/data/private/lm/projects/nnunet/…`），训练会读不到数据或读不到 `splits_final.json`。

---

## 11. G0-P 最终判定

**G0-P = PASS**（2026-09-14）。§1 表 8 项条件全部完成，逐项证据：

| # | 条件 | 证据 |
|:--|:--|:--|
| 1 | 方向码修正（单位 direction = LPS） | §3.1；`test_picai.py` 19 passed |
| 2 | manifest / geometry 审计重建（仅 orientation 变化） | §3.2/§3.3；两处差分验证 JSON |
| 3 | 1500 例 ADC/HBV 物理坐标重采样物化 | §5.4；1500/1500 ok，0 失败 |
| 4 | 1500 例 canonical 二值标签 | §5.4；425 阳性 0 丢失、1075 阴性 0 异常、取值全合法 |
| 5 | 5 例 geometry_suspect 重采样后 QC | §6；腺体内覆盖 100%，全部保留并标注 partial FOV |
| 6 | WG 异常例屏蔽 | §5.4；`11050_1001070` 显式禁用且不替换 |
| 7 | 冻结 train/val 划分（患者级，无泄漏） | §7；legacy 一致性 pass=True |
| 8 | nnU-Net raw/预处理/划分冻结 | §8；全量完整性 PASS、3d_fullres 1500/1500、`splits_final.json` 14/14 PASS |

- 无失败病例、无静默排除；除已标注的 `11050_1001070`（WG 禁用）与 5 例 partial FOV 外无保留项。
- **N0 的数据/split/plan 工程条件已满足**；这不等于 v2.3 定义的正式启动门已经满足。正式 N0 仍须等待
  G0-R PASS 与 G0-E 独立测试路径冻结，且唯一命令以 `docs/N0_PICAI_Official_Baseline.md` 当前版本为准。
  - **2026-09-15 状态更新（含更正）**：该命令已于 2026-09-15 01:17 由研究者执行，即**默认 Dice+CE 预实验**；
    其日志在 **Epoch 43** 开始后中断、**未完成**（无 `Training done`、无 `checkpoint_final.pth`、无 `validation/`）。
    该 run **已排除**，不再作为正式 N0 baseline，产物、日志与 checkpoint 仅作审计留痕保留。
    **更正**：后续静态审计证实 v2.6.2 的**验证 dataloader 同样启用前景过采样**（每 batch 1/2 patch 被强制取在
    前景体素上），因此「验证只用随机 patch、极少含病灶」**不能**解释该 run 中 43 个 epoch 里 38 次
    `Pseudo dice = 0`；`Pseudo dice` 是 patch 级 online 指标，不作为正式结论。
    正式 N0 的定义与唯一训练命令见 `docs/N0_PICAI_Official_Baseline.md`
    （`N0 = PI-CAI official-style nnU-Net v2 Focal+CE NoFFT`，**尚未启动真实训练**）。
- **允许进入 P1/N0 与 P2A 的工程开发**（G0-P PASS 为开发前置）。本报告曾记录的 CODE READY 仅对应 v2.2
  legacy PlainConv M0；v2.3 Residual-Encoder M0 已于 P2A 实现并通过合成回归（§9.4.2 G2 前置 1–7 = ✅），但**不能继承
  旧 G2 状态**，真实数据三步验证（第 8 项）仍 Pending、G2 = NOT YET EVALUATED。G2 判定条件见
  `docs/research_plan.md` §9.4.2 / §14 与 `docs/P2_M0_ResidualEncoder_v23.md`。
- G0-E（外部测试就绪）仍为 **Pending**；它不阻塞代码开发或明确标记的 feasibility 调试，但与 G0-R 一样，
  是正式 N0 训练身份和正式 M0–M4 结果的前置门。

---

## 附录：产物清单（本轮新增/更新）

| 类型 | 路径 |
|:--|:--|
| manifest / validation | `data/metadata/picai_manifest.csv`、`picai_manifest_validation.json`（已按 LPS 重建） |
| P0A 验证证据 | `outputs/diagnostics/picai_model_readiness/p0a_orientation_fix_verification.json`、`p0a_geometry_orientation_fix_verification.json` |
| 历史归档 | `outputs/diagnostics/picai_model_readiness/archive_pre_p0a/`（旧 manifest/geometry/报告） |
| geometry 审计 | `outputs/diagnostics/picai_model_readiness/geometry_audit.csv`（10500 行）、`geometry_anomalies.csv`（1054 行） |
| 物化产物 | `data/processed/picai/cases/<case_id>/*.nii.gz`、`materialization_report.csv`、`materialization_summary.json`、`materialization_audit.json`、`materialization_validation.json/csv` |
| 物化后 QC | `outputs/diagnostics/picai_model_readiness/qc_figures_post_p0b/`（7 张 PNG + summary JSON） |
| 划分 | `data/splits/picai_train_val_split.json`（+ `_vs_legacy_check.json`）；nnU-Net 划分 `workdir/nnUNet_preprocessed/Dataset605_PICAI/splits_final.json`（raw 同内容副本）、`data/splits/picai_nnunet_splits_validation.json`、`data/splits/picai_nnunet_split_cases.csv` |
| nnU-Net raw | `workdir/nnUNet_raw/Dataset605_PICAI/`（symlink + dataset.json）；`data/metadata/picai_nnunet_case_mapping.csv` |
| nnU-Net 预处理 | `workdir/nnUNet_preprocessed/Dataset605_PICAI/`（3d_fullres 1500/1500，≈45 GB） |
| 脚本 | `scripts/data/materialize_picai.py`、`scripts/data/audit_materialization_report.py`、`scripts/data/prepare_picai_nnunet_raw.py`、`scripts/audit/generate_picai_materialized_qc.py`、`scripts/data/create_picai_splits.py`（更新）、`scripts/data/create_nnunet_splits.py`（新增，nnU-Net 划分）、`scripts/data/build_picai_manifest.py` / `scripts/audit/audit_picai_geometry.py`（进度条） |
| 核心模块/测试 | `src/zonal_reliability_fusion/data/picai.py`（orientation 修正）、`tests/unit/test_picai.py`（19 passed） |

> 复现命令与耗时见 `docs/experiment_log.md`；代码变更见 `docs/Development_Log.md`；结构与运行约定见 `docs/project_structure.md` 与 `AGENTS.md`。
