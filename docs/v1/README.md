# V1 — FROZEN

正式名称：**Shallow Anatomy/Lesion-Conditioned Fusion**。2026-10-12 已由 V2 取代。
V1 不增加网络模块，不扩张研究问题。更早 gate 研究属于 LEGACY，见 [archive](../archive/README.md)。

| 冻结文件 | 职责 |
|---|---|
| [Research Plan](Research_Plan_V1.md) | V1 科学问题与假设快照 |
| [Method](Method_V1.md) | V1 方法与实现快照 |
| [Experiment Plan](Experiment_Plan_V1.md) | V1 设计与旧命令快照 |
| [Findings](Findings_V1.md) | V1 证据及更早 Preliminary evidence 的历史边界 |
| [原实验详情目录](../experiments/) | V1 run 详情与共享 Stage-1 历史记录，原路径保留 |
| [Training Log](../Training_Log.md) | 唯一当前状态表及真实训练事实 |

实现保持原路径 `src/zonal_reliability_fusion/multimodal/conditioned_fusion.py` 与历史 Trainer 定义，
显式 V1 namespace 只再导出，避免移动实现改变 checkpoint/import semantics。
根 Method/Research/Experiment/Findings 文档转为版本入口；共享 Evaluation_Protocol 不复制。

历史命令必须增加 `--v1`，例如 `train_nnunet.py --v1 neutral_fusion_dicece ...`；
也可用显式 `v1_neutral_fusion_dicece`。这两者解析到同一个旧 Trainer，不重命名历史输出。
Stage-1 使用 `--v1 anatomy_joint_100ep`，其实现仍为 V1/V2 共享基础设施。
默认 help 只暴露 V2。当前 V1 D 运行不中断，也不读写其 checkpoint。
