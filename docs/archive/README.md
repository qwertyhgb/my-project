# docs/archive —— 旧研究路线归档

本目录保存**不再是项目主线**的研究材料。它们**不删除**，原因是三类不可替代的价值：

1. **checkpoint 兼容**：`legacy/` 里的 Trainer 类名决定已有训练产物的输出目录，类名与实现必须
   逐字保留，对应的实验文档也必须可核查；
2. **preliminary / negative evidence**：已完成的输入级门控实验与特征级融合实验是新主线转向的
   直接动机（"单纯 input-level modality weighting 未观察到稳定最终分割增益"），论文的
   motivation 章节需要引用它们；
3. **历史事实不被篡改**：`docs/Training_Log.md` 与 `docs/Development_Log.md` 里对这些实验的
   记录必须能追溯到对应文档。

## 为什么归档而不是删除

新主线（Anatomy-Guided Lesion-Aware Coarse-to-Fine）**取代**旧主线（modality gating / feature
fusion），但取代不等于否定。归档让两件事同时成立：

- 新读者打开仓库只看到一条主线（README / Research Plan / Experiment Plan 里不再出现旧模块）；
- 论文里可以诚实地说"我们试过输入级与特征级重加权，没有稳定收益，因此转向病灶感知的解剖建模"。

## 目录内容

| 路径 | 内容 | 对应代码（只读） |
|---|---|---|
| `old_research_plans/Research_Plan_2026-09_modality_gating.md` | 2026-09 的研究计划全文（分区条件自适应多序列融合） | — |
| `legacy_gate_research/README.md` | 旧门控 / 特征融合研究线的技术说明与结论边界 | `src/zonal_reliability_fusion/legacy/fusion_networks.py`、`legacy/fusion_trainers.py` |
| `historical_experiments/image_gate.md` | 输入级 image gate 单实验记录 | `nnUNetTrainerPICAI_ImageGate` |
| `historical_experiments/image_gate_positive_sampling.md` | image gate + 阳性采样单实验记录 | `nnUNetTrainerPICAI_ImageGate_PositiveSampling_NoFFT` |
| `historical_experiments/anatomy_gate_positive_sampling.md` | anatomy gate + 阳性采样单实验记录 | `nnUNetTrainerPICAI_AnatomyGate_PositiveSampling_NoFFT` |

实验**事实**（命令、耗时、Dice、输出目录、checkpoint 清单）仍完整保留在
`docs/Training_Log.md`；本目录只保存"当时怎么理解这件事"的文本。

## 使用规则

- **禁止**在归档材料上继续做方法扩张（包括 `AnatomyGate v2`、`CrossAttention`、`ZoneTransformer`、
  `ZoneMamba` 这类"再来一个 gate"的方向）；
- **禁止**让新主线的默认训练流程依赖 `legacy/` 下的任何模块（`nnunet/networks.py` 与
  `nnunet/transforms.py` 的兼容转发层除外，它们只用于解析历史 checkpoint）；
- 需要引用旧实验时，链接本目录，不要在 README 首页介绍它们。

## 旧主线的证据强度（不要过度解读）

- 输入级 image gate（`positive_sampling` → `image_gate_positive_sampling`）：配对 Dice 变化
  95% CI **跨 0**，判定为 `no_clear_paired_improvement`；
- 输入级解剖条件（`image_gate_positive_sampling` → `anatomy_gate_positive_sampling`）：
  配对 Dice 变化 95% CI **跨 0**，同样未获支持；
- CI 跨 0 **既不支持提升，也不支持下降，更不证明等效**；
- 全部对比均为**单次运行**（nnU-Net v2.6.2 不设随机种子），因此不能声称"gate 无效"或
  "PZ/TZ 无用"，只能说"在已测试的这一次运行里没有观察到明确的配对改善"。

详细结论与边界见 `docs/Findings.md` 的「Preliminary Findings」章节。
