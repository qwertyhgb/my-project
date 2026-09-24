# Development Log

简洁的代码/架构变更日志。只记录实质性结构变化，不保留旧阶段门（G0/G1/G2/SAP/P2A/P2B）历史。
训练与验证的运行事实见 `docs/Training_Log.md`。

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
