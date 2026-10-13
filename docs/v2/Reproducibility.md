# VERSION=V2 — Reproducibility

实现边界在 [Method](Method.md)，运行模板在 [Experiment Plan](Experiment_Plan.md)。
本文件只维护复现来源与安全约束，不复制 run 结果或另建状态表。

- Python package 仍为 `zonal_reliability_fusion`；conda `lm`，固定 official nnUNet v2.6.2。
- 每次 nnU-Net 命令先 source `scripts/env_nnunet.sh`，校验实际源码来源而非仅版本号。
- Align then Refine checkout commit/source/license 见 [third-party provenance](../../third_party/README.md)；
  不安装、不加入 PYTHONPATH、不 runtime import、不注册其 Trainer。
- 三完整 encoders 与共享 decoder 参数全部由本地 plans 构建；V2 run-config 记录完整 architecture。
- 当前concat三组为独立类名，所有尺度投影相同；run-config另外记录projection kernel/scales、
  policy尺度、倍率与零初始化。旧均值/直接加权类与checkpoint语义保留。
- 同seed先构造全部共同参数再构造policy/anatomy；等权policy初始还原concat基线。
  此构造测试不证明真实训练bitwise确定性，也不证明动态权重能改善性能。
- registry 的所有 V1/LEGACY 名字与 seeded 类名可解析；V2 名字含 V2，自然分离输出目录。
- 首版 family 为 native DiceCE + positive-case sampling + NoFFT，与 V1 已选 family 匹配；
  不复用 V1 checkpoints 做 V2 初始化，也不更名既有 run-config。
- seed/fold/预算/配对方案见 Experiment Plan；显式 seed 不保证多进程 bitwise determinism。
- 输入 safety 依赖 probability tensor checks + predicted metadata + geometry/materialization/audit chain；
  checkpoint、split、IN_SAMPLE/OOF/HELD_OUT/EXTERNAL 标记保留，不改 Dataset608。
- resume 保留原生 optimizer/scheduler/state，阶段由 current_epoch 重算；formal prediction 统一final。
- new model/run-config 已存在即拒绝覆盖；data/workdir/outputs/checkpoint 全部保留。
- pure synthetic tests 可读 plans/split JSON，所有影像/target 为随机或人工tensor，输出仅pytest临时目录。
- CPU真实plans小patch不证明fullpatch GPU budget；optional helper 为stdout-only，GPU任务由用户运行。
- full evaluation、training、validation、inference、materialization、prior regeneration 均不由代理启动。
- governance files 分工维持：指标唯一 Evaluation_Protocol、run事实唯一 Training_Log、代码变更 Development_Log，
  V1/V2 Findings 独立。旧顶层文档为版本入口，V1快照只读，实验旧路径保留可追溯。

实际资源helper输出既有dependency warning：DNA=0.3.1，与official nnUNet v2.6.2
>=0.4.1,<0.5范围冲突。本轮沿用现有依赖，仅验证PlainConvUNet路径，
不宣称上游依赖完全满足，也不自动pip升降级。源码来源校验与此依赖限制分别记录。

V2 artifact root固定为 `outputs/v2/`。原生results路径由训练入口进程内切换，
同时更新nnunetv2.paths与native Trainer模块的import-time快照；不改third_party源码或共享预处理路径。
run-config/预测/评价的路径守卫在任何真实计算前检查，拒绝历史区、越界symlink与覆盖。
实际运行状态只见Training Log；V1/LEGACY/Stage-1产物保持原身份、原路径。不能用更名把V1归为V2。
