# Evaluation Protocol

> **口径冻结文档。** 本文件是论文评价体系的**唯一**指标定义来源；代码实现在
> `src/zonal_reliability_fusion/evaluation/`，离线 CLI 在 `scripts/evaluate_segmentation.py`。
> 改动本文件中的任何定义、常量或空值语义都视为**协议变更**，必须在
> `docs/Development_Log.md` 中显式记录，并说明是否需要重算历史结果。

一句话原则：**以 lesion failure mode 为中心评价，而不是只看 nnU-Net 的 `foreground_mean Dice`。**

---

## 1. 终点分级

| 级别 | 指标 | 说明 |
|---|---|---|
| **Primary** | `positive_case_macro_dice` | 所有 GT positive cases 纳入；**完全漏检病例 Dice = 0，不得排除** |
| **Key secondary** | `lesion_sensitivity_any_overlap` | 病灶实例级 sensitivity（匹配协议见 §4） |
| | `small_lesion_sensitivity_any_overlap` | small 层的病灶实例级 sensitivity |
| | `matched_lesion_dice` | **只对成功匹配的 pair** 计算；**必须**与 sensitivity 同时报告 |
| | `completely_missed_lesion_rate` / `completely_missed_case_rate` | 本项目最重要的 failure mode |
| | `positive_voxel_recall` | lesion 找到之后到底覆盖了多少 GT lesion |
| | `positive_voxel_precision` | 防止粗暴扩大分割 |
| | `false_positive_burden` | 见 §6 |
| **Exploratory** | lesion size strata sensitivity | 按 reference lesion 物理体积分层 |
| | anatomy-region FP burden | 见 §6.3 |
| | `surface_metrics`（HD95 / ASSD / NSD@τ） | 表面距离 |
| | bootstrap CI | 不确定性区间 |

分级常量在 `evaluation/protocol.py::ENDPOINT_TIERS` 中冻结。

**报告纪律**：`matched_lesion_dice` **不得**单独报告——它天然排除了全部 missed lesion，单独报告
会隐藏完全漏检。

---

## 2. 关键口径（避免歧义）

### 2.1 两个 precision 不是一回事

| 名称 | 公式 | 分母 |
|---|---|---|
| `positive_voxel_precision` | `ΣTP / (ΣTP + ΣFP)` | **仅阳性病例**（论文核心口径） |
| `all_prediction_voxel_precision` | `ΣTP / (ΣTP + ΣFP + ΣFP_阴性)` | 含阴性病例的假阳体素 |

两者**不得**都笼统称为 "precision"。

### 2.2 Dice 的空值与漏检语义

- 阳性病例**完全漏分**（预测为空）→ Dice 记 **0**，**不得**排除该病例；
- `n_ref == 0` 且 `n_pred == 0`（真阴）→ Dice 为 `None`（JSON `null`），**不是** 0，也**不是** 1；
- `n_ref == 0` 且有预测（假阳阴性）→ Dice 为 0；
- summary 中的 Dice 若给出，必须与由 TP/FP/FN 重算值一致（容差 `DICE_MATCH_TOLERANCE = 1e-6`）；
  真阴病例的 Dice 必须为空/NaN，写成 1 视为数据错误。

### 2.3 分母为 0 一律为 `null`

`ratio` 在分母为 0 时返回 `None`，**不得**伪造成 0。适用于所有比值型指标（sensitivity、ratio、
rate、precision 类）。

### 2.4 体素 vs 物理体积

- 体素数量一律标注 **voxel-based**，**不得**冒充 mm³；
- 物理体积 = 体素数 × 该病例**真实 spacing** 的乘积（`voxel_volume_mm3`），**不使用**数组 shape
  猜 spacing；
- `case_lesion_burden_mm3` 是**病例总阳性体积**，不是单病灶大小。

### 2.5 不做任何预测后处理

最小团块过滤 / 最大团块 / 形态学开闭 / 填洞 / 阈值优化**一律禁止**。指标反映原始网络输出。

### 2.6 不计算 challenge detection metric

本工具**不**计算 AUROC / average precision / FROC / PI-CAI challenge score。评价对象是
**segmentation failure analysis**。

### 2.7 patch pseudo Dice 不是整例结果

训练日志里的 patch-level pseudo Dice（含 EMA）只描述优化过程，存在代理偏差，**不得**用来替代
整例验证结果，也**不得**用于跨模型性能声明。

---

## 3. 病例集合与几何

- **不做重采样**：prediction 与 reference 必须指向同一物理网格；几何不一致一律 fail-closed，
  评估**不**重采样、**不**修改掩膜、**不**静默修复几何、**不**静默排除病例；
- reference 掩膜必须是二值 0/1；含其它取值即报错；
- summary 声明 `n_ref = 0` 时，reference 掩膜**必须实际为空**，否则报错；
- 由掩膜重算 TP/FP/FN/TN/`n_pred`/`n_ref` 六项计数，并与 `summary.json` **逐项核对**，不一致
  即 fail-closed（`verify_case_counts`）。

### 3.1 几何数值一致性检查（**2026-10-10 修订**）

**修订原因**：一次 A1/A2 full 评价在**修订前**的「四字段逐值严格相等」规则下整体
fail-closed（具体运行事实、退出码、计数与日志路径见 `docs/Training_Log.md` 2026-10-10 条目；
代码与测试侧记录见 `docs/Development_Log.md` 同日条目——运行结果只登记在那里，本节只写规则）。

已展示的失败条目全部是 `direction` 不严格相等，元素绝对差在 `1e-8` 量级，**与 NIfTI 头部
浮点存储/往返误差（qform/sform 以 float32 存储、方向余弦经四元数往返）的量级相符**；
**但这只是量级相符，不等于已经证明"没有网格错位"**：其余未展示失败条目的字段与量级尚未核实，
**是否满足本节新规则必须在对完整队列重跑之后才能判定**。修订前的规则是
「size / spacing / origin / direction **逐值严格相等**」，**当时没有冻结任何数值容差**——
本次是**显式的规则修订**，不是"一直如此"；新规则能否让真实队列通过属于**待验证**事项，
不得预先宣布。

**修订后的判定（`evaluation/case_metrics.py::geometry_check`）**：

1. **整型尺寸**：`size` 必须**严格相等**，**没有**尺寸容差；
2. **浮点字段**：`spacing`、`origin`、`direction` 有限性通过后，用**绝对**容差比较，
   `rtol = 0`（不使用 `numpy.allclose` 的默认相对容差）：

   | 字段 | 绝对容差 | 单位 |
   |---|---|---|
   | `spacing` | `GEOMETRY_ATOL_MM = 1e-6` | mm |
   | `origin` | `GEOMETRY_ATOL_MM = 1e-6` | mm |
   | `direction` | `GEOMETRY_ATOL_DIRECTION = 1e-6` | 无量纲（方向余弦） |

   `spacing` 还必须**全为正**；NaN / Inf 一律拒绝；
3. **物理角点检查**：direction 的微小差异会随图像范围累积，因此额外比较两侧网格的
   **体素中心角点**物理坐标（每轴取 `index ∈ {0, size-1}`；`size == 1` 的退化轴只取 0；
   映射与 ITK `TransformIndexToPhysicalPoint` 一致），要求

   ```text
   max ‖p_pred − p_ref‖₂  <=  GEOMETRY_CORNER_ATOL_MM = 1e-3  mm
   ```

   逐元素差异在容差内、但累计角点偏差超限时**仍然拒绝**。
4. 任一环节不通过即 `ok = False`，`full` 模式**逐例收集失败、整体不发布结果 JSON**；
   失败文本必须给出字段名、最大绝对差、对应容差与单位（角点条目额外给出 mm）。

**语义边界（必须照此表述）**：以上是**浮点存储/往返精度的数值一致性检查**，用于区分
"同一网格的舍入差异"与"真实网格不一致"；它**不是**临床容许误差标准、**不是**解剖配准或
对齐质量的证明，也**不**声称"alignment 已验证"。这些容差是工程数值检查参数，
**不得**依据"真实病例能否通过"来反复放宽；如需调整，必须由独立证据（例如按字段测得的
头部存储粒度）驱动并再次记录协议修订。

**为什么能吸舍入、又拦得住真实错位**（设计论证，不是本次队列的结论）：真实错位（轴的交换/
翻转、错误 origin、不同 spacing/分辨率、配准到别的网格）在元素层面就是 `1e-3`–`1e0` 量级，
或直接改变整数 `size`，远大于上述容差；而同一 NIfTI 网格两次写盘的头部舍入在 `direction` 上
只有 `~1e-8` 量级，在 `spacing`/`origin` 上受同量级浮点粒度限制。该论证只说明容差选得开，
**不**证明本次队列中的差异一定全部是舍入。

**未证实部分（不得当作结论）**：失败日志只展示了前若干条（全部为 `direction`），其余失败
条目的字段与量级**尚未核实**；因此本次修订不针对单一字段放宽，而是对四个字段统一采用
"整型严格 + 浮点绝对容差 + 角点累计阈值"。真实队列**尚未**用新规则重跑，
**重跑前不得声称几何检查问题已解决**。

**角点检查在本队列尺度上的作用边界**（诚实说明）：对 extent ≲ 200 mm 的前列腺 MRI，
`direction` 元素容差 `1e-6` 通常是**更紧**的约束；角点阈值 `1e-3 mm` 主要约束
更大范围网格上的角度累积（extent ≳ 1000 mm 时成为绑定约束），并防止整体同向偏移绕过
逐元素判断。两者同时满足才接受。

### 3.2 结果中的几何 provenance

`full` 模式的输出 JSON 必须记录本次实际采用的几何规则与执行结果（不能只在终端声明）：

- 顶层 `geometry_check`：`rule`（含 `revision`、各字段绝对容差与单位、角点阈值与
  `<=` 边界、`fail_closed=True`、`resampling="none"`）与 `applied_in_modes`；
- `models[*].geometry_check`：该模型 `n_cases_checked` / `n_cases_exactly_equal` /
  `n_cases_with_tolerated_difference`、逐字段 `max_abs_diff` 与 `max_corner_delta_mm`；
- `models[*].per_case[*].geometry`：该病例的 `exact_equal` / `nonzero_fields` /
  最大差与角点结果。**容差内的非零差异必须留痕**，不依赖终端日志。

`SCHEMA_VERSION` 因此由 `1.1` 递增为 `1.2`（新增上述字段，且几何判定会改变哪些病例能进入
结果）。`summary` 模式不读取掩膜，因此**不执行**几何检查（规则仍然写在顶层 provenance 中）。

---

## 4. 病灶实例级匹配协议（冻结）

### 4.1 实例定义

**3D connected component，`CONNECTIVITY = 1`（6-邻域 / face connectivity）**。
仅通过角或边接触的体素**不**属于同一 lesion。不改成 18/26 邻域。

### 4.2 候选匹配

`intersection >= 1 voxel`。

### 4.3 一对一匹配目标（字典序）

1. **最大化匹配数**（maximum cardinality）——尽可能多的 reference lesion 被匹配；
2. 相同匹配数下**最大化总 intersection 体素数**；
3. 仍相同则按 `(reference id, prediction id)` **升序**给出确定性唯一解。

**不**采用"匈牙利最大化 Dice / IoU"或"贪心 Dice / 重叠"作为 matching 目标：`matched-lesion
Dice` 本身是报告终点之一，先最大化 Dice 会产生 **metric-optimizing-the-metric** 风险。Dice
只用于**匹配之后**的质量评价。

实现（`evaluation/lesion_metrics.py`）：`_min_cost_flow_unit_matching`（unit-capacity 最小费用流）
+ `_max_weight_matching_exact_size`（精确可行性判据）+ `_match_lesion_instances`（id 升序贪心）。
结果只由 `(交点权重, 组件 id)` 决定，**与容器迭代顺序、随机性、文件/模型/病例顺序无关**。

### 4.4 主指标命名与分母

```text
lesion_sensitivity_any_overlap = matched reference lesions / all reference lesions
```

- 未匹配（missed）lesion **不从分母删除**；
- 阴性病例（`n_ref == 0`）贡献 0 个分母，不参与 sensitivity，但其 prediction components 全部
  计入 `unmatched_predicted_lesions`（假阳负担）。

### 4.5 一致性断言

`_validate_lesion_matching` 保证：一对一（同一组件不被匹配两次）、每对 `intersection >= 1`、
匹配数不超过任一侧实例数。任一条不满足即抛 `EvaluationError`（整体 fail-closed）。

### 4.6 大小分层（探索性）

| 键 | 定义 |
|---|---|
| `small_lt_500_mm3` | `V < 500 mm³`（`499.999 -> small`，`500.000 -> medium`） |
| `medium_500_to_1000_mm3` | `500 <= V <= 1000 mm³` |
| `large_gt_1000_mm3` | `V > 1000 mm³`（`1000.001 -> large`） |

分层依据是**单个 reference lesion** 的物理体积。这些只能称为 **exploratory size strata**，
**不是**临床大小分类或风险类别。该层 0 个 reference lesion 时 sensitivity 为 `None`，不是 0。

---

## 5. 表面指标

| 指标 | 实现 | 口径 |
|---|---|---|
| `nsd`（NSD@τ） | DeepMind `surface-distance==0.1` 官方实现 | 按**物理表面测度 μ（surfel 面积，mm²）加权**，不是表面体素计数 |
| `hd95_mm` / `assd_mm` | `medpy` | mm，`spacing` 为数组轴序 `(z, y, x)` 的真实 mm |

空值语义：

- GT 非空、预测为空 → `nsd = 0`，`hd95` / `assd = null`（该病例同时计入 missed）；
- GT 与预测均为空 → 全 `null`（不纳入阳性病灶表面统计）；
- GT 非空、预测非空但无重叠 → 正常计算物理表面距离（Dice 为 0，计入 wrong-location）。

缺少 `surface-distance` 时 **fail-closed**（`EvaluationError`），**不**静默退回旧的表面体素计数
实现（那会改变已声明的论文指标口径）。NSD 容差 τ **必须**由研究者在运行前显式给出，工具**不设**
默认值。

---

## 6. False-positive burden

### 6.1 全局口径（必须报告）

```text
negative_cases_with_fp          阴性病例中出现假阳的例数
fp_components_per_case          FP 连通域数 / 病例
fp_voxel_volume_per_case        FP 体素体积 / 病例
unmatched_predicted_lesions     unmatched prediction components / 全部病例
```

### 6.2 case-level 与 component-level 互补

体素口径回答"多少 FP 体积"，团块口径回答"有多少个假阳团块"。两者都必须报告：一个巨大的腺体内
FP 团块在体素口径里很显眼，在团块口径里只有 1 个；反之亦然。

### 6.3 按预测解剖先验的区域分解（exploratory）

`evaluation/anatomy_metrics.false_positive_burden`：

```text
inside_wg       P(WG) >= wg_threshold
outside_wg      P(WG) <  wg_threshold
PZ / TZ / uncertain   仅在 inside_wg 内按 argmax(P(PZ), P(TZ)) 唯一归属；
                       两者都 < zone_threshold 时记为 uncertain
```

每个体素**恰好**归一 zone，PZ 与 TZ 不会双计。阈值由调用方显式传入，工具不设默认值。

**先验来源纪律**：

- 正式结果**必须**使用 `prior_source="predicted_prior"`（冻结 anatomy 模型对该病例自身 MRI 的预测）；
- 使用 GT 解剖标签时**必须**显式传 `prior_source="ORACLE_GT"`，该标记会**原样写入输出**，使
  oracle 分析**不可能**被误当成正式最终性能；
- 传入未识别的来源字符串 → 报错。

### 6.4 病灶定位分解

`lesion_anatomy_localization` 给出每个 reference lesion 的解剖定位与 lesion 级 sensitivity，用于
回答"小病灶漏检是否集中在某个解剖位置"。它**复用** §4 的同一匹配协议，因此"missed"的定义与主
终点**完全一致**，不会出现两套漏检定义。

---

## 7. 统计口径

### 7.1 bootstrap

- 默认重采样次数 `DEFAULT_BOOTSTRAP_RESAMPLES = 10000`，随机种子 `DEFAULT_SEED = 20260922`；
- 单模型指标的 CI → `bootstrap_ci_mean`（病例级重采样）；
- 模型间**配对**比较 → `bootstrap_ci_paired_mean` / `bootstrap_ci_paired_ratio`：
  **同一批病例索引同时重采样两个模型**，禁止分别重采样；
- 比值型指标的某次重采样分母为 0 时**丢弃该次样本**。

### 7.2 配对比较必须报告的内容

```text
mean_paired_dice_delta          配对病例级 Dice 变化均值
CI95                            配对 bootstrap 95% CI
improved / tied / worsened      按 TIE_TOLERANCE = 1e-12 分类的病例数
new_overlap_cases               由"无重叠"变为"有重叠"的病例数
lost_overlap_cases              反向变化
complete_miss_delta             完全漏检病例的变化（带符号）
negative_fp_case_delta          阴性假阳病例的变化（带符号）
interpretation                  解释标签（见 §7.3）
```

### 7.3 解释纪律

- CI **跨 0** 时：`no_clear_paired_improvement`。这**既不支持提升，也不支持下降，更不证明等效**；
- 只有 CI **不跨 0** 且改动可归因于单一变量时，才可以声明方向性差异；
- **单次运行不能声明 run-to-run 方差**：nnU-Net v2.6.2 不设随机种子，配对 CI 只反映**病例间**
  变异，不包含训练随机性。新主线的最终候选必须补 3 seed 的 `mean ± std`（见 §9）。

### 7.4 输出与失败语义

- 输出 JSON 使用**原子发布**（临时文件 + `os.replace`），**已存在则拒绝覆盖**；
- `allow_nan=False`：NaN/Infinity 泄漏时直接失败，不写出非法 JSON；
- 任一模型失败、任一病例几何不一致、病例集合/身份/`n_ref` 不一致时：先打印完整错误汇总，再
  **非零退出，且不发布**输出 JSON（fail-closed）；
- 结束必须打印结构化汇总（status / mode / phase / unit / ok / failed / skipped / elapsed /
  output）；`skipped` 不可知时渲染为 `not_applicable`，**不伪造成 0**；
- `SCHEMA_VERSION` 当前为 `1.2`；改动指标集合或输出契约时必须递增
  （`1.2` = 2026-10-10 几何检查修订：新增顶层 `geometry_check` 与
  `models[*].geometry_check` / `per_case[*].geometry`，见 §3.1/§3.2）。

---

## 8. 冻结常量与变更规则

以下常量通过 `evaluation/protocol.py` 冻结。改动任一项都意味着历史结果**可能需要重算**：

| 常量 | 值 | 改动的后果 |
|---|---|---|
| `CONNECTIVITY` | `1`（6-邻域） | 病灶实例数改变 → 全部 lesion 级指标不可比 |
| `LESION_SIZE_SMALL_MAX_MM3` | `500.0` | 分层人数改变 |
| `LESION_SIZE_MEDIUM_MAX_MM3` | `1000.0` | 分层人数改变 |
| `DICE_MATCH_TOLERANCE` | `1e-6` | 只影响 fail-closed 判定的严格度 |
| `TIE_TOLERANCE` | `1e-12` | `improved/tied/worsened` 计数改变 |
| `SCHEMA_VERSION` | `1.2` | 输出格式契约 |
| `GEOMETRY_ATOL_MM` | `1e-6` | spacing/origin 绝对容差（mm，rtol=0）；改动会改变哪些病例能进入 full 结果 |
| `GEOMETRY_ATOL_DIRECTION` | `1e-6` | direction 绝对容差（无量纲，rtol=0）；同上 |
| `GEOMETRY_CORNER_ATOL_MM` | `1e-3` | 角点物理偏差阈值（mm，`<=` 接受）；同上 |

**注**：最后三个常量是 **2026-10-10 修订新增**（原因见 §3.1），不是此前冻结过的常量；
它们改动时同样必须记录协议修订，并说明是否需要重跑历史 full 结果。

---

## 9. 两种运行模式

| 模式 | 读取 | 产出 | 代价 |
|---|---|---|---|
| `summary`（默认） | 只读 `FOLD_DIR/validation/summary.json` | 体素计数派生的全部指标 | 秒级 |
| `full` | 追加读取 `summary.json` 中 `metric_per_case` 指向的 prediction / reference NIfTI | 几何数值一致性检查（§3.1）、物理体积（mm³）、RVE/ARVE、表面指标、病灶体积分层、连通域失败分析、**病灶实例级指标**、解剖区域分解 | 长任务（必须由研究者运行） |

**评估与模型完全解耦**：两种模式都只读 nnU-Net 的 validation 产物，**绝不**重写 validation /
inference / prediction，**绝不**训练模型，**绝不**在评估中重采样或修改掩膜。

`summary` 模式可以覆盖的模型集合由「该模型是否已完成 validation（存在
`validation/summary.json`）」决定；`full` 模式额外要求 prediction / reference NIfTI 仍然存在且
几何一致。

---

## 10. 解剖先验模型（Stage 1）自身的评价

Stage-1 使用算法解剖伪监督，质量不等于人工解剖准确度。三个 sigmoid heads 必须独立评价：
`anatomy.validation.soft_head_metrics` 对 soft probabilities 直接按固定 `>0.5` 阈值与 reference
位编码各 bit 比较，报告 Dice/recall/precision/TP/FP/FN、prediction/reference volume ratio、
空预测数与概率分位数。macro 逐病例等权；micro 先汇总计数再算比值；分母0仍为null。

原生 `regions_class_order=[1,2,4]` 硬导出后写覆盖前写，不能从硬整数标签恢复独立WG head。
`anatomy_region_metrics` 接受位编码数组；原生ordered export的指标只表示该导出表示，
不能据其低WG Dice断言 soft WG 塌缩。
诊断同时从soft heads按原生顺序重建硬导出并与NIfTI逐体素比较，报告mismatch voxels。

概率分位数 [0,.1,.5,.9,1] 分别在整FOV与reference-positive体素中计算。
跨病例输出明确为“病例分位数均值”，不是 pooled voxel quantiles，不构成calibration声明。
几何、reference、metadata或病例读取失败必须整体失败，不静默排除。
阈值曲线若增加，只是诊断；不能选择最佳阈值伪装为预先冻结。

既有hard summary保持原样；独立soft-head诊断不改 lesion evaluation schema/常量。
运行事实与结果放 Training_Log/单实验文档，不在协议中写性能。

### 10.1 Fusion Behaviour Analysis（mechanism only）

融合系数解释为 model fusion coefficients，不作医学因果贡献。
可按 lesion/background、预测WG/PZ/TZ/uncertain与reference物理体积分层分析。
输出各modality的mean、population std、median、IQR，以及三系数的自然对数entropy均值。
空区域返回null，不虚构0。系数必须有限、[0,1]且逐位置和为1。
GT lesion仅用于离线区域划分，不进入推理。
每份系数artifact必须带case/checkpoint/channel order/geometry与空间范围provenance。
现有hook只输出network-input patch，不能标成full-volume；不从最后一个滑窗系数推断整例分布。
本分析不进入primary endpoint，不修改既有分割匹配、阈值、大小分层或空值语义。

---

## 11. 外部数据（Prostate158）的评价

- 定位为 **external distribution-shift stress test**；同质性与训练第三序列的差异必须在报告中
  说明；
- 若模型依赖 predicted anatomy，**必须用冻结的 anatomy model** 在外部数据上生成 prior；
- **禁止**人工修改外部 prior、使用 GT zone、或在看到 test 结果后重新调参数；
- 指标口径与主实验**完全一致**（同一 `Evaluation_Protocol`），不做任何放宽。

---

## 12. 3 seed 最终报告模板

最终候选（strong baseline vs final proposed model）必须：

```text
same fold / same training budget / same evaluation / 3 explicit seeds
```

报告：

```text
each run                     每个 seed 的 primary + key secondary 指标
mean ± std                   跨 seed 的均值与标准差
paired case-level changes    病例级配对变化（同一 seed 内配对；跨 seed 报告分布）
```

并附上：`explicit seed improves repeatability but does not guarantee bitwise determinism.`

**短预算（100 / 150 epoch）结果不得与 1000-epoch 结果直接声明性能优劣。**
