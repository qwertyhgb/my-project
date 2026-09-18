# G0-R 全自动序列对齐 QC 协议（Automated v0.3）

- **状态：DRAFT（未冻结）**；`protocol.id = G0-R-AUTOMATED`，`version = draft-0.3`
- 机器可读载体：`configs/protocols/g0_r_alignment_qc_automated.yaml`（本文件的唯一权威参数来源）
- 执行工具：`scripts/audit/run_picai_alignment_qc_automated.py`
- 核心算法库：`src/zonal_reliability_fusion/protocols/g0_r_automated_qc.py`
- 人工协议（历史草案，保留不改）：`docs/protocols/G0_R_ALIGNMENT_QC.md`（`G0-R` draft-0.1）
- **历史版本（按字节归档，不再执行）**：`configs/protocols/archive/g0_r_alignment_qc_automated_draft_0_2.yaml`
  （SHA256 `67c479a0…c89a`，见同目录 `README.md`）；draft-0.2 的首次真实运行产物位于
  `outputs/diagnostics/g0_r_automated/20260918_074219/`，**原样保留、不得追溯修改**。

## 0. v0.3 相对 draft-0.2 的方法性变更（必须整体升版本，禁止混用）

| # | 变更 | 原因（已发生的真实缺陷） |
|---|---|---|
| 1 | 刚体诊断改 `SetInitialTransform(..., inPlace=True)`；initial/final 显式类型校验，只允许直接 `Euler3DTransform` 或「单元素 Euler3D 的 `CompositeTransform`」解包，否则 fail-closed | draft-0.2 首次真实运行中 32/32 病例报 `SimpleITK 异常: Transform is not of type Euler3DTransform!`（`inPlace=False` 返回 `CompositeTransform`，被旧代码强转 Euler3D） |
| 2 | 校准按 **`(pair, case_id)` unit** 分组；零位移参考为 unit 内均值；零位移噪声用 leave-one-out 残差 | draft-0.2 把不同病例、T2W-ADC/T2W-HBV、多次零位移重复混进同一个分布，`zero_sd` 主要反映**病例间/pair 间绝对基线差异**，掩盖 unit 内由位移引起的退化（真实运行 `zero_sd≈0.0698`，2 mm 退化仅 ≈0.008，检出率 0/100%） |
| 3 | 真实病例阈值按 pair 推导（`median_of_paired_unit_midpoints_by_pair`），与灵敏度门限**分开命名**；`decide_case` 必须显式传 `pair`，未知 pair 直接失败 | 单一全局阈值会让 ADC/HBV 互相借用阈值；灵敏度门限与病例绝对阈值混用无法审计 |
| 4 | 所有 JSON 输出带 `schema_version = g0-r-automated/0.3`；读取必须经 `load_output_json()` 校验 | 防止把 v0.2 旧输出按 v0.3 schema 静默解释 |

**未改变（不得静默修改）**：抽样的 16 例清单、序列对（T2W-ADC / T2W-HBV）、位移集合
（`[0, 0.5, 1, 2, 3, 4] mm`）、方向集合（x/y/z）、`min_detectable_mm = 2.0`、检出率要求
（`≥ 0.9`）、`detection_multiplier = 3.0`、单调性下限、零位移误报上限、`input_policy`、`privacy` 与
「不写回影像」原则。

> 本协议是 `G0-R`（§4.2.1 序列错位协议）的**全自动执行路径**：不需要人工看图、人工 landmark、双阅片、
> 人工阈值选择或人工逐例判断。自动路径只输出**候选**（`resample-only` / `resample+rigid` /
> `INSUFFICIENT_EVIDENCE`），**不等于 G0-R PASS**，冻结仍由研究者执行（§9）。

---

## 1. 目标与不主张

**目标**：在冻结的 16 例抽样清单上，用可复现的自动指标 + 合成已知位移校准，给出"是否存在系统性残余错位"
的**定量证据**与预注册的候选决策。

**明确不主张**：

- 不主张自动指标等于解剖学真值（跨模态边缘一致性是**相对指标**，只通过校准获得可比性）；
- 不主张自动报告是人工阅片 / 双阅片结论（本流程完全不产生人工判读）；
- 不主张自动候选即为协议冻结（冻结需研究者写入 `frozen_decision` 等字段）。

## 2. 边界与隐私（硬约束，由代码强制）

| 约束 | 强制方式 |
|---|---|
| 全自动，无人工判读 | `protocol.automated_only: true`、`human_visual_review_required: false`、`human_landmarks_required: false` |
| 不使用 lesion / WG / PZ-TZ / 模型预测 / 训练结果 | `input_policy.*` 全为 `false`；`assert_automated_input_policy()` 在启动时校验，任一开启即拒绝运行 |
| 不外发任何影像 / PNG / 特征 | `privacy.external_upload_allowed: false`、`external_api_allowed: false`；代码静态扫描禁止网络 import（测试锁定） |
| 原始/物化影像只读 | `privacy.source_images_read_only: true`；代码中不存在 `sitk.WriteImage` / `ImageFileWriter` / `np.save`（测试锁定）；端到端测试断言输入文件哈希不变 |
| 不做任何配准写回 | A/B 指标在**仅物理重采样**的数据上计算；C 的刚体估计**只在内存中**，结果只作为诊断证据；配置 `data_written_back: false` 写入 `run_metadata.json` |

## 3. 输入

- `--sampling-manifest`：冻结的抽样清单（`outputs/diagnostics/g0_r/20260915_093819/sampling_manifest.json`）；
  病例集合、序列对与分层标记不可替换/增删/重抽样（`g0_r_qc.load_sampling_manifest` 契约校验）；
- `--materialized-root`：物化影像根目录（`data/processed/picai/cases`），只读；
- 每个 case×pair：T2W 为 reference，ADC/HBV 为 moving；不同网格时**只按物理坐标重采样**到 T2W 网格
  （`g0_r_qc.prepare_pair_arrays`，线性插值，**不做任何配准**）。

## 4. 证据 A：物理几何与 FOV（`fov_policy`）

分别对 `T2W-ADC`、`T2W-HBV` 输出：

- 两侧 `size / spacing / origin / direction`（写入 `qc_geometry.json`）；
- 有效物理重叠比例（逐轴，`picai.axes_overlap`）与 `partial_overlap` 标记；
- 重采样后 z 向层数与物理范围；
- 影像体检：空影像、非有限值、常量（相对容差判定，避免重采样浮点抖动误判）、非背景体素占比过低；
- 几何体检：size/spacing/origin/direction 完整性、spacing 是否为有限正值、direction 是否近似正交；
- 输入 SHA256（`input_hashes.json`）。

判定：`overlap_ratio` 可用轴数 < `min_overlapping_axes`、任选轴重叠 < `min_axis_overlap_ratio`、
z 层数 < `min_covered_slices_z` → `FOV_INSUFFICIENT`；影像/几何体检失败 → `INVALID_INPUT`。
**FOV 不足或输入异常一律单独标记，不静默删除病例**（写入 `per_case_decisions.csv` 与 `skipped.csv`）。

## 5. 证据 B：多尺度跨模态边缘一致性（不做配准）

1. 强度标准化（`robust_percentile_scaling`：1–99 百分位裁剪 + 背景掩膜 + 线性映射到 [0, 1]；
   其它方法名一律拒绝）；
2. 多尺度物理平滑（`scales_sigma_mm`，σ 以 **mm** 给出，按各轴 spacing 折算为体素）；
3. 物理梯度幅值（`np.gradient` 按 spacing 求导，单位 1/mm；**轴序约定见 §7.4**）；
4. 百分位阈值化 → 二值边缘（每个尺度独立）；
5. 双向 mm 距离：`d(A→B)`、`d(B→A)` 由"到目标掩膜前景的欧氏距离变换"在**物理单位**下计算；
6. 指标（键格式 `sigma<尺度mm>.<指标>`）：
   - `edge_f1_at_<r>mm`（多容忍半径 r ∈ `tolerance_radii_mm`，对称 F1）、
     `edge_precision/recall_at_<r>mm`、`chamfer_mm`、`mean/median/p95/max_a_to_b_mm`；
   - 分 slice 统计（`slicewise: true`，仅供分布审计，**不参与阈值判定**）。

**禁止**只用原始强度 NCC 或单一互信息量作为结论依据（本协议不使用这两个量判定；边缘一致性为主要证据）。

主指标：`sigma1.5.edge_f1_at_1.0mm`（`lower_is_worse`）；一致性指标：`sigma1.5.edge_f1_at_2.0mm`
（`lower_is_worse`）与 `sigma1.5.chamfer_mm`（`higher_is_worse`）。

## 6. 证据 C：诊断性刚体残余估计（仅诊断）

- T2W = reference，moving = ADC/HBV；变换 `Euler3D`；初始化 `centered_geometry`；
  metric = Mattes MI（32 bins）；随机采样 10%（固定 seed）；RegularStepGradientDescent，
  lr=1.0 / minStep=0.001 / 150 迭代；3 级 shrink `[4,2,1]` + 物理 sigma `[2,1,0] mm`；线性插值；
- **transform 类型契约（v0.3 修复，必须保持）**：
  1. 使用 `SetInitialTransform(initial, inPlace=True)`——SimpleITK 2.5.3 下 `inPlace=False` 会让
     `Execute()` 返回 `CompositeTransform`，对其做 Euler3D 强转会抛
     `Transform is not of type Euler3DTransform!`；
  2. `initial` 与 `final` 都必须通过 `as_euler3d_transform()` 的**显式类型校验**；
  3. 兼容性解包**只允许**：对象确实是 `CompositeTransform`、且**仅含一个** transform、且该内部 transform
     确实是 `Euler3DTransform`（解包方式必须写入输出 `transform_unwrap`）；
  4. 其余情况（多元素 Composite、内部类型不符、其他类型、缺字段）一律
     `REGISTRATION_DIAGNOSTIC_FAILED`，**绝不伪造零位移、绝不静默采用未知 transform**；
- 输出 `translation_mm`、`rotation_deg` 与幅值、`metric_before/after`、迭代数与停止条件、
  `final_transform_type`、`transform_unwrap`；
- **数值与结构校验（fail-closed）**：`translation_mm` / `rotation_deg` 必须是 3 个有限值，
  `metric_before` / `metric_after` 必须为有限数，`n_iterations` 必须为非负整数；任一不满足即失败标记；
- **只在内存中**估计：不写回影像、不改变物化数据、不得直接作为训练预处理；
- 固定 seed 下结果可复现（ITK 多线程归约允许 ~1e-9 量级浮点差异，远小于 mm/度量级）；
- 该证据**不能单独决定结论**，只与 A/B/D 联合使用。

> 与 draft-0.2 的关系：draft-0.2 的该证据在真实数据上 100% 失败（32/32），失败原因不是数据质量，
> 而是上述类型错误；修复不改变任何算法参数（metric/optimizer/迭代/采样/seed 未变）。

## 7. 证据 D：合成已知位移校准（自动阈值来源）

### 7.1 注入与稳定性（与 draft-0.2 相同，未修改）

1. **注入位移**：对读到的 moving 图像在内存中注入 `displacements_mm = [0, 0.5, 1, 2, 3, 4]` ×
   `directions = [x, y, z]`，插值 `linear`，边界模式 `nearest`。
   `border_mode=constant` 被代码显式拒绝：零填充会在图像一侧制造人为强边界，污染边缘指标并破坏
   "位移越大指标越差"的单调性（该缺陷已在合成测试中发现并修复）。
2. **零位移稳定性**：每个 unit 做 `stability_repeats` 次固定 seed 的强度扰动（σ = 2% 的 p99−p1 相对尺度）；
   同一病例的两个序列对使用**不同** seed 偏移（`pair_noise_offset`），仍然完全确定可复现。

### 7.2 校准分组（v0.3）

- **最小单元 unit = `(case_id, pair)`**（`calibration.grouping: [pair, case_id]`）；
- T2W-ADC 与 T2W-HBV **完全独立汇总**，禁止把两者的绝对指标混进同一分布；
- 「指标对注入位移是否敏感」与「真实病例的绝对判定阈值」**拆开**：前者用 unit 内退化与噪声阈值，
  后者按 pair 用 unit midpoint 聚合（§8），两者不得互相替代。

### 7.3 零位移参考与噪声阈值（v0.3，精确公式）

对每个 `(unit, metric)`：

```
zero_reference(u)      = mean(该 unit 的零位移重复值)                     # calibration.zero_reference
degradation(u, d, dir) = zero_reference(u) − observed            (lower_is_worse)
                       = observed − zero_reference(u)            (higher_is_worse)

# 零位移噪声（leave-one-out，避免把自身算进参考均值）
loo_degradation(u, i)  = degradation(mean(其他零位移重复), value_i)
pooled_within_unit_sd  = sqrt( Σ_u (n_u − 1)·sd_u² / Σ_u (n_u − 1) )      # 只含 unit 内残差
noise_threshold        = max( detection_multiplier × pooled_within_unit_sd,
                              max_u, i loo_degradation(u, i),
                              0 )
```

阈值定义保证「零位移样本不会被判为变差」：`zero_false_positive_rate = 0`。
**病例间 / pair 间的绝对基线差异不得进入该阈值**（v0.2 的缺陷正是如此）。

### 7.4 灵敏度曲线与通过条件

对每个 pair × 每个 metric 分别计算：

- 每个距离的 `degradation_mean/sd/min/max`（样本 = 该 pair 的全部 `(unit, direction)`）；
- 每个 unit、每个方向的 degradation（写入 `unit_details`）；
- `detection_rate(d) = 比例{ degradation(u, d, dir) > noise_threshold }`（样本 = `(unit, direction)`）；
- `zero_false_positive_rate`（对零位移 leave-one-out 退化统计）；
- `monotonicity`：先按 unit 求各距离的退化均值，再对 unit 取均值（0 mm 点退化恒为 0），
  统计所有 `(d_i < d_j)` 配对中 `agg(d_j) > agg(d_i)` 的比例。

通过条件（数值与 draft-0.2 相同，**未因任何结果调整**）：

```
monotonicity ≥ 0.8
detection_rate_at_min_detectable ≥ 0.9      (min_detectable_mm = 2.0)
zero_false_positive_rate ≤ 0.05
min_detectable 处的 unit 内退化 > 0
```

`calibration.require_each_pair_primary_pass: true`：**两个 pair 的主指标都必须各自通过**；
任一 pair 失败或缺失 → 总体 `calibration.passed = false` → 候选必为 `INSUFFICIENT_EVIDENCE`。
某 pair 的校准 unit 不足、主指标不敏感或阈值不可用时，该 pair 直接返回
`INSUFFICIENT_EVIDENCE`，**不得回退到全局阈值**。

### 7.4 轴序与单位约定（实现锁定的关键细节）

numpy 数组轴序为 `(z, y, x)`，而几何 `spacing` 为 `(x, y, z)`：

- 数组 → SimpleITK 图像：spacing 直接按 `(sx, sy, sz)` 设置（**不反转**）；
- 按数组轴给物理尺度时（`np.gradient`、高斯 σ、距离变换 `sampling`）必须用 `(sz, sy, sx)`。

`tests/unit/test_g0_r_automated_qc.py::test_distance_map_respects_axis_order_and_physical_units`
锁定该约定（z 轴 3 mm 与 x 轴 0.5 mm 分别验证）。两个真实缺陷已在该测试下被修复：
`array_to_image` 的 spacing 轴序错误、距离图语义（应为"到目标掩膜前景的距离"而非"到前景边界"）。

## 8. 阈值推导与自动决策（预注册、版本化、确定性）

**命名分离（v0.3，不可混用）**：

| 名称 | 含义 | 存放位置 |
|---|---|---|
| **灵敏度门限**（`noise_threshold` / 检出率判定） | 判断"该指标在该 pair 上是否对注入位移敏感" | `calibration_summary.json` |
| **真实病例阈值**（case threshold） | 判断真实病例的指标是否超出 | `threshold_derivation.json` 的 `by_pair` |

**真实病例阈值推导（v0.3 冻结公式）**：`thresholds.derivation: median_of_paired_unit_midpoints_by_pair`

```
unit_zero_baseline(u)  = 该 unit 零位移参考均值
unit_min_mean(u)       = 该 unit 在 min_detectable_mm 处各方向指标均值
unit_midpoint(u)       = ( unit_zero_baseline(u) + unit_min_mean(u) ) / 2
threshold(pair, metric) = median{ unit_midpoint(u) : u ∈ 该 pair }        # thresholds.aggregation_by_pair
```

- 阈值**按 pair 保存**：`thresholds.metrics.<metric>.by_pair.T2W-ADC` / `...T2W-HBV`；
- `unit_zero_baselines` / `unit_min_detectable_means` / `unit_midpoints` / 聚合方法 / 指标方向 /
  `sensitivity_passed` 全部写入 `threshold_derivation.json`；
- `decide_case()` **必须**接收 `pair`：只使用该 pair 的阈值与该校准单元，**禁止跨 pair 混用**；
  未知 pair 直接抛错（fail-closed），不存在全局回退；
- 不得依据模型结果、人工结果或真实 QC 最终结论调整阈值；公式与来源记录在 `threshold_derivation.json`。

**判定顺序（冻结）**：

1. 输入/几何体检（A）→ `INVALID_INPUT`；
2. FOV（A）→ `FOV_INSUFFICIENT`；
3. 诊断刚体（C）失败 → `REGISTRATION_DIAGNOSTIC_FAILED`；
4. 指标判定（B 阈值来自 D）：
   - 主指标必须可校准且可用（否则 `INSUFFICIENT_EVIDENCE`）；
   - 一致性指标若**未通过合成校准**（对注入位移不敏感 / 饱和）→ 记为 `SKIPPED` 并排除出判定（如实记录，不当作证据不足）；
   - 通过校准的指标全部 `ACCEPTABLE` → `ACCEPTABLE`；全部 `FLAGGED` → `FLAGGED`；混合 → `INSUFFICIENT_EVIDENCE`（指标冲突）。

**总体候选（冻结逻辑，改变必须改代码版本）**：

- `calibration_not_passed`、`INVALID_INPUT` / `FOV_INSUFFICIENT` / `REGISTRATION_DIAGNOSTIC_FAILED`
  比例超过 `max_*_fraction`、存在指标冲突、无病例 → `INSUFFICIENT_EVIDENCE`；
- 否则任一 case×pair `FLAGGED` → `resample+rigid`；
- 否则（两对均未发现系统性残余错位）→ `resample-only`。

**ADC 与 HBV 必须独立统计**（`pair_conclusions` 分别给出 `verdict`），不允许只评价一组后外推。

## 9. 输出与冻结程序

真实运行输出到**新建/空目录**（非空即拒绝覆盖，不删除既有产物）：
`outputs/diagnostics/g0_r_automated/<UTC 时间戳>/`

```
automated_metrics.csv        每个 case×pair 的全部自动指标 + 刚体诊断 + FOV 摘要
per_case_decisions.csv       状态、原因、主指标值/阈值、逐指标 checks(JSON)、pair 标记
calibration_summary.json     按 (pair, case_id) 分组的校准曲线、unit 内噪声阈值、单调性、检出率、
                             零位移误报率、分 pair 通过状态（含 schema_version / grouping / 公式定义）
threshold_derivation.json    阈值、方向、来源（零位移与 min_detectable 值）
decision_draft.json          候选决策（DRAFT）、触发原因、分状态占比、成对结论
qc_geometry.json             两侧完整几何（size/spacing/origin/direction）
input_hashes.json            输入影像 SHA256
run_metadata.json            协议哈希、命令、耗时、data_written_back=false
skipped.csv                  跳过项与原因（不静默删例）
qc_report.md                 自动报告（见下）
overlay/                     可选（`--overlay`）；仅审计用，无需人工查看
```

`qc_report.md` 分别写清：输入完整性、FOV/几何异常、T2W-ADC 自动结论、T2W-HBV 自动结论、
**分 pair 合成校准结果（含每个 unit 的零位移基线 / min-detectable 均值 / midpoint 与 pair 阈值）**、
自动候选决策、不能冻结的原因、局限性，并显式声明**不是**人工阅片/双阅片证据。

**输出 schema 与兼容性（v0.3）**：所有 JSON 输出都带
`schema_version = g0-r-automated/0.3`、`protocol_version`、`protocol_hash`、`draft: true`、
`calibration_grouping`。读取本工具输出**必须**使用
`scripts/audit/run_picai_alignment_qc_automated.py::load_output_json()`（schema 不匹配即拒绝）；
**禁止**把 `20260918_074219/`（draft-0.2 产物）按 v0.3 schema 解释或与之直接合并比较。

**冻结程序（研究者）**：

1. 运行 §10 的一键命令（先 `--dry-run`，再正式运行）；
2. 检查 `calibration_summary.json.passed`、`decision_draft.json.candidate`、`qc_report.md`；
3. 在本协议（或新建冻结记录）中写入 `frozen_decision`、`protocol_hash`、`reviewer`、`frozen_at`
   与该次运行的 `run_metadata.json` 哈希；
4. 若候选为 `resample+rigid`：**必须新建派生数据、重新预处理并复检**，不得在旧派生数据上直接训练；
5. 若为 `INSUFFICIENT_EVIDENCE`：**必须停止**，不得据此启动 P2B / 正式 M0 / N0。

**协议状态**：**DRAFT（未冻结）**。执行状态（工具/测试/真实运行/冻结进度）唯一维护在 `docs/STATUS.md`；
自动运行成功同样不等于 PASS。

## 10. 一键真实运行命令（由研究者执行）

```bash
cd /opt/data/private/lm/my-projects
conda activate lm
source scripts/env_nnunet.sh

G0R_MANIFEST="outputs/diagnostics/g0_r/20260915_093819/sampling_manifest.json"
G0R_RUN="outputs/diagnostics/g0_r_automated/$(date -u +%Y%m%d_%H%M%S)"

# 1) 计划检查（不读影像体素、不写文件）
python scripts/audit/run_picai_alignment_qc_automated.py \
  --config configs/protocols/g0_r_alignment_qc_automated.yaml \
  --sampling-manifest "$G0R_MANIFEST" \
  --out-dir "$G0R_RUN" \
  --dry-run

# 2) 正式自动 QC（默认显示进度条；耗时随数据规模增长，校准占主要部分）
python scripts/audit/run_picai_alignment_qc_automated.py \
  --config configs/protocols/g0_r_alignment_qc_automated.yaml \
  --sampling-manifest "$G0R_MANIFEST" \
  --out-dir "$G0R_RUN"
```

结束打印：成功/失败/跳过病例数、ADC 与 HBV 各自自动结论、合成校准是否通过、输出路径、候选决策、
是否为 `INSUFFICIENT_EVIDENCE`、是否可以提交自动冻结候选。

## 11. 局限性

- 跨模态边缘一致性是相对指标，绝对数值依赖标准化与边缘阈值；判定完全依赖合成位移校准；
- **合成位移只能验证"指标对额外位移的响应"，不能证明真实病例已解剖学对齐**；
- **观察到的 0 mm 状态不等于解剖学对齐真值**（0 mm 只是"未额外注入位移"的状态）；
- 自动结论始终只是**相对 QC 候选**，不是人工阅片、不是解剖学真值；
- 校准只在部分病例（`calibration.max_cases`）上进行，中心/协议间的对比度差异带来系统不确定性；
- 诊断性刚体估计在低对比、强各向异性或小 FOV 数据上可能失败（记为失败标记，不伪造结果）；
- 一致性指标在稠密边缘数据上可能饱和（对位移不敏感），此时会被排除出判定并在报告中记录；
- 若某中心的数据质量整体偏低，`INVALID_INPUT` / `FOV_INSUFFICIENT` 比例超限会直接给出
  `INSUFFICIENT_EVIDENCE`——这是**预期行为**（不强行给出二选一结论）；
- **证据充分性（16 例、相对指标、合成校准）与是否需要人工 landmark 复核或扩大抽样，属研究者决策**
  （`docs/research_plan.md` §20.2 D2）；代理不得替研究者决定，也不得用自动结果替代该决策；
- **即使 draft-0.3 工具运行成功（校准通过、候选非 `INSUFFICIENT_EVIDENCE`），也不自动使 G0-R PASS**：
  冻结仍需研究者按 §9 写入 `frozen_decision` / `reviewer` / `frozen_at`。
