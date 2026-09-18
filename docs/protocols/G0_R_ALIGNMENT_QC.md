# G0-R：序列错位协议（对齐 QC 与预处理身份冻结）

| 字段 | 值 |
|---|---|
| 协议 ID | `G0-R` |
| 版本 | `draft-0.1` |
| **协议状态** | **DRAFT（未冻结；不得据此启动正式 run）**；门的执行状态见 `docs/STATUS.md` |
| 创建日期 | 2026-09-15 |
| 审核人 | _待填写_（冻结时由研究者签名/记录） |
| 冻结日期 | _待填写_ |
| 机器可读配置 | `configs/protocols/g0_r_alignment_qc.yaml` |
| 关联章节 | `docs/research_plan.md` §4.2.1、§14 G0-R、§19 |

> **历史说明（2026-09-16）**：本人工协议（人工 landmark、双阅片、盲态视觉等级、人工阈值选择）**已降级为历史草案**。
> G0-R 的**当前执行路径为全自动协议** `docs/protocols/G0_R_ALIGNMENT_QC_AUTOMATED.md`
> （`G0-R-AUTOMATED` draft-0.3）：零人工阅片、零人工 landmark、零双阅片、零人工阈值，
> 由 `scripts/audit/run_picai_alignment_qc_automated.py` 一键运行并给出候选决策。
> 本文件保留其数据事实、抽样规则与 QC 概念（仍可被自动路径引用），但 `evaluation.landmark_rules.min_per_case_pair`
> 等人工字段**不再是 G0-R 的 blocker**；G0-R 仍为 DRAFT/Pending。

> **本协议不评价任何病灶模型。** 执行期间不得查看任何模型（N0/M0–M4/H0）的输出、指标或预测图；
> 只使用原始影像、物化数据与解剖先验完成对齐 QC。违反该条将导致 G0-R 结论作废。

## 1. 目的与边界

### 1.1 要回答的唯一问题

现有 PI-CAI 物化数据（T2W 参考网格 + ADC/HBV 物理坐标重采样）**是否足以支撑逐位置（patch-of-voxel）
序列融合**，还是必须额外引入固定刚体配准并重建派生数据？

### 1.2 输出（唯一允许的产出）

1. 一份**预注册的预处理身份**：`resample-only` 或 `resample+rigid`（二者之一，必须二选一并冻结）；
2. 与之配套的失败/排除/降级规则（§7）；
3. 全部 QC 证据文件与内容哈希（§9）。

### 1.3 明确不做

- 不调参、不比较模型、不评价分割性能；
- 不因“某个模型在这种预处理下更好”而改变选择（模型输出不可见）；
- 不引入非线性配准、可变形配准或配准网络（超出研究边界，见 research_plan §17）；
- 不重跑全部分区质量分析（仅当 §6 判定需要重建派生数据时，按 §7.3 重建并复检）。

## 2. 输入与已核实的数据事实（只读）

| 项 | 路径/说明 |
|---|---|
| 原始 PI-CAI | `/opt/data/private/lm/data/Prostate/PI-CAI`（**只读**） |
| manifest | `data/metadata/picai_manifest.csv`（1500 study；含 `t2w_grid`/`adc_grid`/`hbv_grid`/`geometry_status`/`anomaly_reason`） |
| 几何审计 | `outputs/diagnostics/picai_model_readiness/geometry_audit.csv`（10500 行 = 1500×7）、`geometry_anomalies.csv` |
| 物化数据 | `data/processed/picai/cases/<case_id>/{t2w,adc,hbv,lesion,wg,zonal_yuan,zonal_hevi}.nii.gz`（1500 例） |
| 物化审计 | `data/processed/picai/materialization_{report.csv,audit.json,summary.json,validation.csv,validation.json}` |
| 现有 QC 工具 | `scripts/audit/generate_picai_qc.py`、`scripts/audit/generate_picai_materialized_qc.py`（含物理空间叠加与棋盘格对照） |

**已核实（2026-09-14/15，来自 manifest 与几何审计，不得在 QC 中重新解释为“已配准”）**：

- `geometry_status`：`partial_physical_overlap` 655、`same_physical_space_different_grid` 840、
  `geometry_suspect` 5；
- ADC 网格状态：`same_physical_space_different_grid` 1153、`partial_physical_overlap` 343、`geometry_suspect` 4；
  HBV 网格状态：1145 / 351 / 4；
- 中心分布：RUMC 800、PCNN 350、ZGT 350；csPCa：YES 425、NO 1075；
- T2W↔ADC/HBV 原始为不同采样网格（T2 高分辨、扩散像低分辨），“网格不同”本身不等于“未配准”；
  头信息无法反映序列间残余运动/形变错位（`docs/Data_Analysis.md` §2.1/§4.2）。

## 3. 抽样规则（确定性、可复现）

抽样为**确定性**（脚本按固定规则排序取例），不引入随机性；总样本量由盲态 pilot（§5）确定后写入
`configs/protocols/g0_r_alignment_qc.yaml` 的 `sampling.n_cases`，冻结前不得改动分层配额。

### 3.1 必选分层（每层至少覆盖）

1. **中心**：RUMC / PCNN / ZGT 各自 ≥ 1 例；
2. **病灶状态**：csPCa 阳性与阴性各自 ≥ 1 例；
3. **几何异常**：
   - `geometry_suspect` 5 例**全部纳入**（4 例 ADC/HBV suspect + 1 例其它）；
   - `partial_physical_overlap` ≥ 2 例（ADC/HBV 与 T2W 部分重叠，覆盖 FOV 差异情形）；
4. **极端 FOV**：按 `t2w_extent` / `adc_extent` / `hbv_extent` 的物理包围盒体积或 z 向跨度排序，
   取最大与最小各 ≥ 1 例（FOV 明显不等或层数极端的病例）；
5. **离散度覆盖**：按 `adc_spacing`（平面内间距 0.86–2.51 mm）与 `t2w_spacing`（0.3–0.63 mm）排序，
   覆盖两端（各 ≥ 1 例）。

### 3.2 抽样实现与环境

- 建议在 `scripts/audit/` 下新增只读脚本（示例名 `audit_picai_alignment_qc.py`），
  读取 manifest/几何审计/物化数据并生成 §9 规定的输出；
- 抽样结果落盘为 `sampled_cases.csv`（含分层标签列，便于复核"每层覆盖"是否满足）；
- 执行环境：`conda activate lm`；涉及 nnU-Net 路径时先 `source scripts/env_nnunet.sh`；
- 长任务必须显示进度条（AGENTS.md §1），结束时输出成功/失败/跳过数与输出路径。

## 4. 评价对象与指标

### 4.1 评价对（必须分别评价）

- **T2W↔ADC**；
- **T2W↔HBV**。

两组**分别**给出结论；不允许只评价其中一组后外推（例如仅看 ADC 就宣布“序列对齐可接受”）。

### 4.2 定量指标

1. **毫米单位的几何指标（主）**：
   - landmark/边界位移：在腺体（WG，Bosma22b；`11050_1001070` 的 WG 除外）边界或解剖 landmark
     上测量对应点位移，单位 **mm**，报告逐例值（median / max）；
   - 边界一致性：T2W 与 ADC/HBV 上同一解剖边界（如腺体轮廓、膀胱/直肠界面）的 Hausdorff 距离或
     平均表面距离（mm）；
   - 位移测量必须记录测量所依据的坐标约定（物化数据为 LPS，见 manifest 的 `*_orient`）。
   - **v2.2 landmark 坐标空间（冻结）**：`coordinate_space = native_index_per_series` ——
     `t2w_*` 为 **T2W 物化网格**上的体素索引，`moving_*` 为对应 **ADC/HBV 物化网格**上的体素索引
     （`(i, j, k) = (x, y, z)`）；两个网格都可以由研究者在 ITK-SNAP / 3D Slicer 中直接打开磁盘上的
     物化 `.nii.gz` 标注，**不要求研究者填写仅在内存中重采样得到的网格坐标**。毫米位移由 SimpleITK
     `TransformIndexToPhysicalPoint` 按**各自** `size/spacing/origin/direction` 转 LPS 物理坐标后取欧氏距离；
     FOV 检查分别按 T2W 网格与对应 moving 网格执行；`qc_geometry.json` 必须保存两侧完整几何。
     少数值坐标（如 `10.9`）一律拒绝，**不得静默截断**。
   - **landmark 最小数量规则（待冻结）**：`configs/protocols/g0_r_alignment_qc.yaml` 的
     `evaluation.landmark_rules.min_per_case_pair` 当前为 `null`；研究者必须在真实运行前冻结该值，
     工具不得代为编造阈值（未冻结时工具输出明确 blocker）。
2. **棋盘格与边缘叠加（overlay，辅助判读）**：
   - checkerboard overlay：两序列交叉棋盘格融合，检查棋盘边界是否存在系统性错位台阶；
   - edge overlay：对两序列做边缘提取后叠加，检查边缘是否成对（错位会形成双轮廓）；
   - 叠加必须**先物理空间重采样再叠加**，严禁仅按数组索引直接叠加不同网格影像
     （既有约定见 `scripts/overlay_qc.py` 与 Development_Log）。
3. **标准化视觉等级（半定量）**：每例每个评价对给出等级 0–3，定义必须写入机器可读配置并在
   pilot 前冻结：
   - `0` 无可辨错位；`1` 轻微、仅边缘可见；`2` 明显错位但主要解剖结构仍对应；`3` 严重错位（结构不可对应）。
   - 等级定义与示例图（每级 1 张）作为 `visual_scale_reference` 一并冻结。
4. **辅助量（只能辅助，不得单独决策）**：
   - 可计算互信息/归一化互信息（MI/NMI）作为辅助佐证；
   - **不得**使用跨模态原始强度 NCC 单独决定“是否需要配准”——跨模态强度关系不成线性，
     原始 NCC 低不代表错位（反之亦然），只能与 §4.2.1/§4.2.2 的结果联合解读；
   - 任何辅助量在报告中的角色必须标注为 `auxiliary_only`。

### 4.3 双阅片者与分歧处理

- 至少两名阅片者独立完成 §4.2.3 的视觉等级与 §4.2.2 的叠加判读；
- 逐例记录双方等级；分歧定义：等级差 ≥ 1 视为分歧；
- 分歧处理：**先不加权、不平均**，由第三方（或双方复核）在不知道对方等级的前提下复核原图，
  按预注册规则（`tie_break_rule`，写入 YAML 后冻结）给出最终等级；
- 报告必须给出：每例双方原始等级、最终等级、分歧计数与处理记录（双人签字/时间戳）。

## 5. 盲态 pilot 与阈值确定程序（不得捏造数值）

**本协议不预设任何具体位移阈值。** 阈值必须通过以下程序产生，并在**查看任何模型结果之前**冻结：

1. 在 §3 抽样病例上计算 §4.2 的定量指标，完成视觉分级；
2. 由阅片者对"可接受/不可接受"给出**盲态**判断（只依据影像与叠加图）；
3. 用 pilot 数据确定将视觉等级 `≤1` 与 `≥2` 分开的位移阈值候选（例如以中位数/分位数为界），
   写入 `threshold_decision` 字段，并记录：样本量、分位数、选择理由、备选阈值与敏感性结论；
4. 阈值与决策规则一旦写入并冻结（记录内容哈希），**在本轮 G0-R 中不得修改**；
   若 pilot 显示数据不足以确定阈值，必须在协议中显式写出 `INSUFFICIENT DATA` 与待补的病例类别，
   不得用默认值蒙混。

## 6. 预注册决策规则（二选一，冻结后不得事后更改）

按以下顺序判定（全部条件写入 YAML 的 `decision_rule` 字段；阈值来自 §5）：

1. 统计 §3 抽样中**不可接受**（最终视觉等级 ≥ 2 或位移超阈值）的病例比例，按 T2W↔ADC 与 T2W↔HBV 分别计算；
2. **若两组的不可接受比例均低于** `acceptable_fraction_max`（由 §5 pilot 确定并冻结）
   → 冻结为 **`resample-only`**；
3. **若任一组不可接受比例达到或超过** `acceptable_fraction_max`
   → 冻结为 **`resample+rigid`**（在物化流程前增加一次性固定刚体配准，参数与实现写入配置；
   配准只允许刚体、只以 T2W 为参考、所有模型统一使用）；
4. 无论取何结果，选择必须在**任何正式 N0/M0–M4 run 之前**写入本协议与
   `configs/protocols/g0_r_alignment_qc.yaml`，并记录哈希；
5. 冻结后若因任何原因修改（例如发现 QC 脚本缺陷），必须升版本（`draft-0.2` / `frozen-1.0`）并说明原因，
   旧结论作废重跑（不得混用）。

**位移敏感性实验**（可选补充，不得用于事后挑主结果）：在冻结主协议后，可对少量病例做人为位移
（如 ±1/2/3 mm）以评估 gate 融合对残余错位的敏感性；该实验与主协议结论分开报告。

## 7. 失败、FOV 不足、局部恶化与排除规则

### 7.1 分类与默认处置

| 情形 | 判定依据 | 处置（冻结前确认，写入 YAML） |
|---|---|---|
| 配准不可行（刚性配准无法收敛/结果病态） | 配准后指标不改善或恶化 | 该例标记 `alignment_failed`，**保留在数据集中**（不静默删除），在主协议下标注并在敏感性分析中报告 |
| FOV 不足（某序列无法覆盖腺体） | 与 WG 的交集覆盖度 | 标记 `fov_insufficient`；若覆盖度低于冻结阈值，仅从**该序列参与的子分析**中排除并显式报告，不整体删例 |
| 局部恶化（配准后局部错位增大） | 配准前后逐例对比 | 标记 `local_degradation`；按预注册规则决定该例采用配准前/后版本，并记录 |
| 几何 suspect（5 例） | manifest `geometry_status=geometry_suspect` | 全部纳入 QC；处理决定逐例记录（沿用 P0B 已记录的人工 QC 结论，见 `docs/experiment_log.md`） |

### 7.2 禁止事项

- 禁止静默删除、覆盖或排除病例（原始数据只读；派生数据写入 `data/processed/` 新版本目录）；
- 禁止仅因“该例让对齐 QC 不好看”而调整抽样；
- 排除规则必须在看到模型结果前冻结；冻结后新增排除 = 协议变更（升版本）。

### 7.3 若判定为 `resample+rigid`

1. 先写死配准实现与参数（工具、版本、插值、参考序列、失败回退），记录到 YAML；
2. 派生数据写入**新目录**（如 `data/processed/picai_rigid_v1/`），旧物化数据保留；
3. 重新生成 nnU-Net raw 链接与 3d_fullres plan/preprocess（由研究者执行，命令见
   `docs/experiment_log.md` 同类记录），并重新冻结 `splits_final.json`（split 本身以 patient 为单位，预期不变）；
4. **重建前的任何 run（含 N0/M0）只作 feasibility**，不得与重建后的正式 run 混用（research_plan §14 G0-R 失败处理）；
5. 重建后必须复跑本协议的抽样 QC（配准后判据见 §8）。

## 8. 配准前/后 QC

- **配准前**：完成 §3 抽样、§4 指标计算与双阅片者判读，产出 §9 全部文件；
- **配准后（仅 `resample+rigid` 路径）**：
  - 对同一批抽样病例复算 §4.2 的全部定量指标，并逐例对比配准前后；
  - 验证：缸体边界位移下降、无新引入的局部恶化、FOV/覆盖未变差；
  - 未达成的病例按 §7.1 记录并进入敏感性分析；
- 无论哪条路径，QC 报告必须包含 T2W↔ADC 与 T2W↔HBV 两组的**逐例明细 + 汇总**。

## 9. 输出与可追溯字段（全部落盘）

| 字段 | 说明 |
|---|---|
| 输入文件 | manifest / geometry_audit / 物化 cases 目录（只读）的路径与 SHA256 |
| 输出文件 | `sampled_cases.csv`、`alignment_metrics.csv`、`visual_scores.csv`、`overlay/<case_id>_{adc,hbv}_{checkerboard,edge}.png`、`qc_report.md` |
| 版本 | 本协议版本 + QC 脚本版本（代码版本信息由脚本记录） |
| 状态 | `DRAFT` → `FROZEN`（只有全部 §5 阈值与 §6 决策写入后才可置 FROZEN） |
| 审核人 | 双阅片者姓名/标识 + 第三方复核记录 |
| 日期 | 执行日期、审阅日期、冻结日期 |
| 内容哈希 | 本协议文件、`configs/protocols/g0_r_alignment_qc.yaml`、上述全部输出文件的 SHA256（写入 `configs/protocols/g0_r_alignment_qc.yaml` 的 `hashes` 字段或独立记录文件） |
| 决策 | `preprocessing_identity: resample-only \| resample+rigid`（冻结后不可变） |

## 10. 工具能力与待办

> 工具与流程的**实现/执行状态**（是否已实现、是否已运行、冻结进度）唯一维护在 `docs/STATUS.md`；
> 本节只描述能力要求与待办，不记录运行事实。

### 10.1 工具能力（要求）

| 工具 | 路径 | 必须提供的能力 |
|---|---|---|
| 抽样与盲审清单 | `scripts/audit/audit_picai_alignment_qc.py` | 确定性分层抽样（中心 / 阳阴性 / `geometry_suspect` 全纳入 / partial overlap / 极端 FOV / 间距跨度）；输出 `sampled_cases.csv`、`sampling_manifest.json`（含 `decision=null`）、`blind_review_sheet.csv`、`alignment_metrics_schema.json`、`run_metadata.json`；`--validate-metrics` 校验指标 schema；`--dry-run` / `--no-progress` |
| 图像 QC 执行工具 | `scripts/audit/render_picai_alignment_qc.py` | 在**抽样清单**病例上：ADC/HBV **仅按物理坐标重采样**到 T2W 网格（**不做配准**）→ 每 case×序列对生成 checkerboard 与 edge-overlay PNG（不含病灶标签或模型输出）；输出 landmark 模板（含冻结的 `coordinate_space`）、`qc_geometry.json`、输入 SHA256 与跳过原因；**指标模式先强制完整校验**（严格整数坐标、T2W/moving 分别 FOV、病例子集与 case×pair 覆盖、重复记录、`measured_by`），未通过时**非零退出且不写指标**；`--allow-incomplete` 仅诊断放行并标 INCOMPLETE；位移按 **LPS 物理坐标**计算；可选边界距离；支持 `--dry-run`/`--no-progress`；拒绝覆盖非空目录 |
| 共享校验层 | `src/zonal_reliability_fusion/protocols/{common,g0_r,g0_r_qc}.py` | 配置契约校验、内容哈希、运行元数据、输出路径隔离；物理重采样记录、切片选择、显示合成（纯 numpy）、严格整数解析、landmark 物理位移/完整性检查、边界指标构造 |

**边界**：抽样工具只读 manifest / 几何审计元数据；图像 QC 工具读取物化影像（研究者运行时）并**只做物理坐标
重采样**（不执行、不推荐任何配准）。两工具均不决定配准方案、不填写 `threshold_decision` / `decision_rule` /
`decision`，也不显示病灶标签或模型输出。**渲染切片按深度比例（25%/50%/75%）选择，不使用病灶或腺体信息**
（避免把判读者引向标签区域）。真实数据运行与盲态 pilot 由研究者执行；即使工具运行成功，
**G0-R 在 §5 阈值与 §6 决策写入并冻结之前不得判为通过**（执行状态见 `docs/STATUS.md`）。

### 10.2 待办

- [ ] **冻结 `evaluation.landmark_rules.min_per_case_pair`**（当前 `null`；工具会输出 blocker，不代填）
- [ ] 在抽样清单上运行图像 QC 渲染（叠加图 + landmark 模板 + 几何记录）
- [ ] 双阅片者：填写 landmark（`--validate-landmarks` 校验）与 `blind_review_sheet.csv` 视觉等级（§4.3）
- [ ] 运行指标模式生成 `alignment_metrics.csv`，并用 `audit_picai_alignment_qc.py --validate-metrics` 复核
- [ ] 执行盲态 pilot 并确定阈值（§5）→ 写入 YAML
- [ ] 依据 §6 冻结 `preprocessing_identity`
- [ ] 记录全部哈希与审核信息 → 状态置 `FROZEN`
- [ ] 若 `resample+rigid`：按 §7.3 重建派生数据并复检

> 协议状态：**DRAFT（未冻结）**。在上述全部完成并通过研究者审核前，G0-R 不得标记为 PASS；
> N0/M0–M4 的正式 run 不得启动（research_plan §14 / §15；执行入口见 `docs/runbooks/`；
> 当前状态见 `docs/STATUS.md`）。
