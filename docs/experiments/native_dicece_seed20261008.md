> VERSION=V1 — 冻结历史实验文档；原始事实保留。共享 Stage-1 按其 V1 运行来源记录。
> 当前状态见 [Training Log](../Training_Log.md)，V2 不复用该实验身份。

# native_dicece_seed20261008 — 原生DiceCE＋原生采样正式实验报告

记录日期：2026-10-11。本文只记录该实验自身，不包含跨实验性能比较。
指标定义见[Evaluation_Protocol](../Evaluation_Protocol.md)，运行状态唯一来源为
[Training_Log](../Training_Log.md)。时间统一为北京时间，原始日志使用UTC。

## 1. 完成状态与结论范围

**1000个epoch及223例整例验证完成，223份概率文件已导出。**
目前结果属于final模型：阳性macro Dice为0.469091，完全零重叠7/63，
阴性出现任意前景151/160。模型能够分割部分病灶，但阴性假阳及部分病例完全漏分仍需分析。
这些结果不证明网络性能上限，也不能单独归因于数据难度、数据处理或某项网络机制。

本次核查读取既有日志、JSON、文件清单与checkpoint元数据/logger，未执行训练、推理、
真实影像处理或full评价。概率文件数量齐全不等于概率数组质量已审计。
初次核查没有自行计算bootstrap CI；用户后续完成的统一full评价见§9，补齐CI、物理体积与实例/表面指标。

## 2. 实验标识与产物

| 字段 | 实际值 |
|---|---|
| CLI variant | `native_dicece` |
| Trainer | `nnUNetTrainerPICAI_DiceCE_NativeSampling_NoFFT_Seed20261008` |
| Dataset / configuration / fold | Dataset605_PICAI / 3d_fullres / 0 |
| Seed / 预算 | 20261008 / 1000 epochs |
| 训练 / 验证病例 | 1277 / 223 |
| 验证阳性 / 阴性 | 63 / 160 |
| 最终checkpoint | `checkpoint_final.pth`，current_epoch=1000 |
| best checkpoint | `checkpoint_best.pth`，current_epoch=949，即完成epoch948后保存 |
| 日志 | `training_log_2026_10_10_15_01_47.txt` |
| 配置记录 | `outputs/run_configs/native_dicece_fold0_seed20261008.json` |

输出目录：

```text
outputs/nnUNet_results/Dataset605_PICAI/
  nnUNetTrainerPICAI_DiceCE_NativeSampling_NoFFT_Seed20261008__nnUNetPlans__3d_fullres/
    plans.json / dataset.json / dataset_fingerprint.json
    fold_0/
      checkpoint_final.pth
      checkpoint_best.pth
      debug.json
      progress.png
      training_log_2026_10_10_15_01_47.txt
      validation/summary.json
      validation/{case_id}.nii.gz / .npz / .pkl
```

原训练曲线保存在上述`progress.png`，无需重新生成文件即可查看。

## 3. 配置、原生性与采样边界

| 配置 | 值 / 核查证据 |
|---|---|
| nnU-Net | 项目固定源码v2.6.2，`third_party/nnUNet` |
| 输入 / 标签 | T2W、ADC、HBV；background=0、lesion=1 |
| 网络 | 原生PlainConvUNet，7 stages，由plans构建 |
| 各级features | 32 / 64 / 128 / 256 / 320 / 320 / 320 |
| Encoder / decoder卷积数 | 各stage两层；decoder六级各两层 |
| Kernel | 前两级[1,3,3]，后五级[3,3,3] |
| Stride | [1,1,1] / [1,2,2] / [1,2,2] / [2,2,2] / [2,2,2] / [1,2,2] / [1,2,2] |
| Patch / batch | [16,320,320] / 2 |
| 预处理spacing | [3.0,0.5,0.5] mm，数组轴顺序 |
| 归一化 | 三路ZScoreNormalization，use_mask_for_norm=False |
| 损失 | 原生DC_and_CE_loss＋MemoryEfficientSoftDiceLoss，由DeepSupervisionWrapper包装 |
| 优化器 | SGD，LR=0.01、momentum=0.99、Nesterov=True、weight_decay=3e-5 |
| Scheduler | 原生PolyLR，1000 epochs |
| 每epoch迭代 | train=250、val=50 |
| Train / val loader | 均为原生nnUNetDataLoader，debug记录实际类型 |
| 前景过采样 | oversample_foreground_percent=0.33，probabilistic_oversampling=False |
| 增强 / 并行 | dummy 2D=True；训练/验证增强worker为12/6 |
| 设备 / PyTorch | cuda:0、RTX 3090；2.10.0+cu128 |
| Checkpoint周期 | save_every=50；best依据patch pseudo-Dice EMA |

### 3.1 原生采样不是“完全没有前景过采样”

本Trainer继承`NoFFTAugmentationMixin, nnUNetTrainer`，不含项目阳性病例平衡mixin，
没有覆盖原生网络、loss、dataloader、optimizer或训练循环。
原生病例选择与前景patch过采样仍然存在：先选择病例，再在满足条件的槽位中尝试前景中心裁剪。
病例若为阴性，不能凭前景槽位凭空产生病灶，也不会因此自动改选阳性病例。

在本次batch=2、非概率过采样下，源码使用
`sample_idx >= round(batch_size*(1-oversample_foreground_percent))`判定槽位，
因此最后一个槽位被标记为force_fg。**这不等于每批保证一个阳性病例，也不等于33%的病灶体素。**
实际增强后的病灶暴露比例没有在本轮测量。

### 3.2 工程修补与未启用模块

NoFFT只关闭Gaussian blur的FFT benchmark，保留模糊增强及其概率/sigma语义。
因此准确名称为“项目固定nnU-Net的原生DiceCE/原生采样参考，含NoFFT工程修补”，
不声称逐字未修改的上游运行。
三通道原生骨干不包含shallow stems、条件融合、解剖先验、coarse lesionness、ROI或hard negatives。
run_config中的通用`fusion_constants`不表示实例化了融合模块，`fusion_condition=null`。

## 4. 启动与运行时间

此前进程清单实际观察到的命令为：

```bash
cd /opt/data/private/lm/my-projects
source /root/anaconda3/etc/profile.d/conda.sh
conda activate lm
source scripts/env_nnunet.sh

python -u scripts/train/train_nnunet.py \
  native_dicece 605 3d_fullres 0 \
  --seed 20261008 --seeded-output \
  --export-validation-probabilities \
  --run-config outputs/run_configs/native_dicece_fold0_seed20261008.json
```

这是运行事实记录，不是建议覆盖已有产物重跑的命令。
进程清单不能还原所有环境变量；本轮不把compile环境变量写成已独立核实的启动事实。
debug记录网络类型为PlainConvUNet。

| 事件 | 北京时间 / 时长 |
|---|---|
| epoch0开始 | 2026-10-10 23:01:51 |
| Training done | 2026-10-11 14:53:15 |
| Validation complete | 2026-10-11 15:05:05 |
| epoch0至训练结束 | 15:51:24 |
| 训练结束至验证结束 | 0:11:50 |
| epoch0至验证结束 | 16:03:14 |
| 日志epoch计算时长之和 | 55540.78秒，约15:25:41 |
| 日志epoch时长mean / median | 55.541 / 55.500秒 |
| 日志epoch时长min / max | 55.290 / 59.730秒 |

一个训练日志完整覆盖epoch0–999，各epoch恰好一次；final logger八个序列各1000项。
没有发现日志中的续训记录，不把它作为外部进程历史的完整证明。
日志提示未安装hiddenlayer，未绘制网络结构图；后续训练与验证完成，属于可选绘图功能缺失。
debug中的current_epoch=0是启动快照，完成状态依据final checkpoint与完整日志判定。

## 5. 训练动态

下表取final checkpoint logger的每100epoch均值，避免仅凭日志四位小数判断：

| epoch区间 | Train loss | Val loss | Patch pseudo-Dice | EMA pseudo-Dice |
|---|---:|---:|---:|---:|
| 0–99 | -0.112843 | -0.105295 | 0.213173 | 0.190447 |
| 100–199 | -0.136276 | -0.127456 | 0.314319 | 0.309385 |
| 200–299 | -0.145306 | -0.131207 | 0.346555 | 0.343919 |
| 300–399 | -0.145957 | -0.134532 | 0.367071 | 0.364122 |
| 400–499 | -0.151442 | -0.136610 | 0.362149 | 0.365562 |
| 500–599 | -0.154217 | -0.139337 | 0.380413 | 0.381577 |
| 600–699 | -0.157237 | -0.139282 | 0.390353 | 0.385732 |
| 700–799 | -0.163563 | -0.146461 | 0.453136 | 0.445068 |
| 800–899 | -0.170533 | -0.147130 | 0.442254 | 0.445658 |
| 900–999 | -0.180917 | -0.147235 | 0.465672 | 0.463571 |

末轮train loss=-0.171289、val loss=-0.179494、patch pseudo-Dice=0.644546，
EMA=0.461860、LR=1.9952623e-5。日志中最高单轮pseudo-Dice约0.6808（epoch771）。
最高EMA=0.501815617（epoch948），best checkpoint的current_epoch=949与之对应。

早期已经出现非零patch重叠，随后区间均值总体提高，但并非每阶段单调增长。
负loss来自原生负soft Dice项与CE相加，不是负概率或训练错误。
末轮patch Dice较高不能替代整例结果；best由patch EMA选择，
目前没有best整例验证证据，不能推断它必然优于final。曲线本身也不能证明没有过拟合。

## 6. 整例验证结果

### 6.1 完整性与统计口径

summary的223个study对应220个patient，病例集合与固定fold0 split一致。
split有1277训练study；按ID前缀核对，训练与验证patient集合没有交集。
这只是split标识检查，不替代所有来源与患者身份的完整审计。
NIfTI、npz、pkl各223份，三类文件名集合均与验证split一致。
逐例核对`n_ref=TP+FN`、`n_pred=TP+FP`及Dice计数公式通过。

阳性统计纳入全部63例，完全零重叠记零。原生foreground mean Dice=0.138096917，
其nanmean包含有假阳的阴性病例零分、排除干净阴性，不能作为阳性macro Dice。
以上是summary与文件清单核对，没有读取真实影像数组重新计算计数或核验几何。

### 6.2 阳性与体素指标

| 指标 | 结果 |
|---|---:|
| 阳性macro Dice | 0.469091115 |
| Dice Q1 / median / Q3 | 0.255248 / 0.502262 / 0.713185 |
| 阳性macro IoU | 0.349414526 |
| 阳性micro Dice | 0.565079982 |
| 阳性pooled voxel recall | 0.531612705 |
| 阳性pooled voxel precision | 0.603044173 |
| 全队列prediction voxel precision | 0.384914008 |
| 阳性TP / FP / FN | 308161 / 202848 / 271511 |
| 阳性reference / prediction体素 | 579672 / 511009 |
| 阳性预测/参考体素总量比 | 0.881548531 |
| 任意重叠病例 | 56/63（88.89%） |
| 完全零重叠病例 | 7/63（11.11%） |
| 空预测阳性病例 | 0/63 |

总量接近参考不等于空间覆盖充分：仍有271511个FN体素及202848个阳性FP体素。
该表体素数均为voxel-based，不能按预处理spacing直接冒充原网格物理体积。
任意病例重叠不是一对一病灶实例召回，且不包含PI-CAI检测指标。

### 6.3 阳性Dice分布与零重叠病例

| Dice区间（描述性） | 病例数 |
|---|---:|
| [0,0.2) | 13（其中7例为0） |
| [0.2,0.4) | 10 |
| [0.4,0.6) | 13 |
| [0.6,0.8) | 21 |
| [0.8,1] | 6 |

| 完全零重叠病例 | GT体素 | 预测体素 |
|---|---:|---:|
| 10294_1000300 | 1142 | 64 |
| 10365_1000371 | 856 | 8384 |
| 10519_1000529 | 336 | 223 |
| 10550_1000562 | 219 | 159 |
| 10921_1000938 | 1128 | 3656 |
| 11169_1001192 | 2664 | 578 |
| 11185_1001208 | 3563 | 1004 |

七例均有预测前景但与GT不重叠；不能仅凭这些计数诊断错位原因、图像难度或标注问题。
这些是study级漏分，不等于七个病灶实例漏检。

### 6.4 阴性假阳

| 指标 | 结果 |
|---|---:|
| 阴性有前景病例 | 151/160（94.375%） |
| 干净阴性病例 | 9/160 |
| 阴性FP体素总数 | 289588 |
| 全部160阴性FP体素均值 / 中位数 | 1809.925 / 635 |
| 单病例最大阴性FP体素 | 25781 |

这里均值/中位数的分母包含9个干净阴性，与评价工具仅对有假阳阴性病例的分布口径区分。
任意前景率不等于临床假阳病灶数；还未检查团块数量、大小与位置。
阴性假阳持续存在，使全队列precision低于仅阳性precision，限制了整体实用性判断。

## 7. 初次核查的复现与证据边界（后续full补充见§9）

- Seed已应用，但多进程增强顺序不保证逐位确定；仅单fold、单seed，无法估计训练间方差。
- 未计算本run的bootstrap CI。验证集中有重复patient的study，后续病例bootstrap不能宣称已校正患者内相关。
- final checkpoint与日志完整；best仅有patch EMA证据，不能选择性用其峰值当整例性能。
- full几何、表面、实例评价以及概率诊断尚无本run的产物证据，不填入推测数值。
- NoFFT保留增强语义，但没有记录本次实际增强样本流与前景暴露比例。
- run_config没有运行时Git commit/工作区diff记录；当前HEAD不能证明训练期间源码版本。
- 当前读取的元数据文件SHA256如下，只用于本次核查快照，不声称它们独立证明训练时未变化：

```text
workdir/nnUNet_preprocessed/Dataset605_PICAI/nnUNetPlans.json
5ca3c61b5dcc22b294d09ae999b9282425cc29fcc1cf10cfb138c1bf64fd45c1
workdir/nnUNet_preprocessed/Dataset605_PICAI/splits_final.json
db1d838dd73ee5a2cab3bcba771987f4ef783186bd3ea3a1b96e25de5b607148
outputs/run_configs/native_dicece_fold0_seed20261008.json
605c21f4561742b6cd868c9786bb15eb873ef950b3e8488c8c460ed4e7c70f0e
```

## 8. 后续补充与记录方式

统一full评价已由用户完成，见§9；概率诊断与best整例验证仍待完成。
再分析概率分布或best时，保持独立产物目录与明确checkpoint。
正式结果仍不通过验证集阈值搜索或新增过滤来替换当前原生输出。

后续更新本报告时记录日期、checkpoint、产物路径、统计口径和真实检查结果；
只写该run自身，跨run比较写Findings，运行事实和状态同步Training_Log。
网络实现改变记录Development_Log，并保留本次Trainer类名与checkpoint兼容性。

## 9. 2026-10-11补充：final完整分割评价

本节只记录本run自身。用户完成full评价，产物为
`outputs/metrics/native_A2_final_full_exploratory_NSD2mm_20261011T071549214283765Z.json`
的`models.Native_final`，创建时间北京时间15:28:30。
本run核验223例；联合任务日志成功446、失败0、跳过0，耗时761.73秒。
schema=1.2、bootstrap=10000、seed=20260922、探索性NSD容差2 mm；代理只读结果，没有重跑。

### 9.1 几何与掩膜核验

223/223病例的真实掩膜计数与summary一致，二值标签检查通过。
22例几何逐值相等，201例存在容差内浮点差异；spacing/origin最大差均0，
direction最大差3.9912884e-8，角点最大物理偏差1.4795162e-5 mm，均通过冻结数值规则。
没有重采样；这不证明原始模态配准质量或标签真实性。

### 9.2 CI与物理体积

- 阳性macro Dice=0.469091115，病例bootstrap 95% CI=[0.399614114,0.537891467]。
- 阳性micro Dice=0.565079982，病例bootstrap 95% CI=[0.384609065,0.694415441]。
- 阳性参考总体积逐病例均值3747.136541 mm³、中位数1410.696075 mm³。
- RVE中位数=-0.039505，ARVE中位数=0.332604；物理体积由各例原网格spacing计算。

病例bootstrap仍不覆盖run-to-run方差，也不是patient-cluster bootstrap。

### 9.3 病灶实例与假阳团块

| 指标 | 结果 |
|---|---:|
| 参考实例 / 一对一匹配实例 | 75 / 57 |
| Any-overlap实例敏感性 | 76.00% |
| 小实例：<500 mm³ | 13/25（52.00%） |
| 中实例：500–1000 mm³ | 8/14（57.14%） |
| 大实例：>1000 mm³ | 36/36（100.00%） |
| 全队列预测团块 / 未匹配预测团块 | 344 / 287 |
| Matched-lesion Dice均值 / 中位数 | 0.564140 / 0.618381 |

实例为6邻域连通域，按冻结的any-overlap一对一匹配；不是PI-CAI检测指标。
Matched-lesion Dice只覆盖57个成功匹配实例，不能作为全部75实例的分割质量。
小/中实例漏匹配仍突出；大实例任意重叠100%不代表覆盖完整。
大小分层为探索性，未匹配预测团块包含全队列，不能与阴性FP病例数互换。

### 9.4 表面指标

| 指标（全部63阳性） | 均值 | 中位数 |
|---|---:|---:|
| HD95，mm | 17.643829 | 6.140040 |
| ASSD，mm | 8.377304 | 2.153787 |
| 探索性NSD@2mm | 0.572852 | 0.652307 |

七个完全零重叠病例均有非空预测，因此也有定义的HD95/ASSD。
表面统计不删除这些失败病例；均值高于中位数提示存在较大的距离尾部，
但未经影像/团块位置分析，不能确定具体解剖原因。未提供这些终点的配对CI。
