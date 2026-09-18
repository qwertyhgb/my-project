# Runbook：P2B / G2 新 M0 真实数据三步验证

- **协议**：`docs/research_plan.md` §9.4.2（G2 前置第 8 项）、§14 G2
- **实现说明**：`docs/P2_M0_ResidualEncoder_v23.md`（§9 状态表、§11 命令草案）
- **配置**：`configs/experiments/m0_resenc_picai_3d_fullres_v23.yaml`（架构：`configs/architectures/m0_resenc_v23.yaml`）
- **状态与阻塞项**：见 `docs/STATUS.md`（本手册不维护状态）
- **执行人**：研究者（真实数据 + GPU）

## 0. 前置条件

1. **G0-R 已冻结预处理身份**（`frozen_decision` 已写入；若为 `resample+rigid`，必须使用新派生数据与重新预处理的 plan）；
2. 固定环境：`conda activate lm` + `source scripts/env_nnunet.sh`（校验 `lm` 与项目内 nnU-Net v2.6.2 源码树）；
3. 明确本阶段的定位：P2B 产物**只用于管线验收与显存核实，不作为论文结果**；
4. 本 runbook 的所有命令**均未在本轮执行**，须由研究者在 GPU 空闲时运行。

## 1. 命令（复制执行）

```bash
cd /opt/data/private/lm/my-projects
conda activate lm
source scripts/env_nnunet.sh
CFG=configs/experiments/m0_resenc_picai_3d_fullres_v23.yaml

# A) setup 检查（只读 plan/split/arch，构建 M0，打印参数量/架构身份；不读病例、不 forward、不用 GPU）
CUDA_VISIBLE_DEVICES="" python scripts/train/validate_m0_setup.py --config "$CFG"

# B) loader smoke（真实数据契约检查；不训练、不 forward、不用 GPU）
#    未指定 --cases 时默认取 train 前 N 例；如需指定阳性病例，从
#    data/splits/picai_nnunet_split_cases.csv 中选择（不得凭记忆编造 case_id）
CUDA_VISIBLE_DEVICES="" python scripts/train/train_m0.py --config "$CFG" --loader-smoke --limit-cases 2

# C) small-overfit（诊断性；独立 diagnostic 输出目录）
CUDA_VISIBLE_DEVICES=0 python scripts/train/train_m0.py --config "$CFG" --overfit \
  --cases <阳性case_id,...> --iterations 60 --epochs 1 --device cuda

# D) 诊断性首次全体积验证 / 正式训练（显存核实后；固定 200-epoch、每 5 epoch 全体积验证）
CUDA_VISIBLE_DEVICES=0 python scripts/train/train_m0.py --config "$CFG" --device cuda

# E) 1-step 显存实测（关闭 B5）：1 个 train iteration + 1 个 validation iteration
#    报告由 --overfit 自动落盘（成功 / CUDA OOM / 普通异常都会写）：
#      outputs/diagnostics/<model_id>/<run_name>/diagnostics/gpu_memory_profile.json
#    M0（自研基线）：
CUDA_VISIBLE_DEVICES=0 python scripts/train/train_m0.py --config "$CFG" --overfit \
  --cases <阳性case_id,...> --iterations 1 --val-iterations 1 --epochs 1 \
  --run-name mem_smoke_m0_b2_amp --device cuda
#    M4（融合模型；需要 prior 已物化，见 p2a_zonal_prior_materialization.md）：
CUDA_VISIBLE_DEVICES=0 python scripts/train/train_m0.py \
  --config configs/experiments/m4_conditioned_gate_picai_3d_fullres_v24.yaml --overfit \
  --cases 10005_1000005,10021_1000021 --iterations 1 --val-iterations 1 --epochs 1 \
  --run-name mem_smoke_m4_b2_amp --device cuda
```

> 说明：A/B 不占 GPU；C/D/E 需要 GPU。`--overfit` 与 `--loader-smoke` 的完整参数见
> `python scripts/train/train_m0.py --help`。**A–E 属于真实数据操作，代理不得代为执行。**
>
> `--cases` 只填**阳性**病例（fold 0 的 train 划分内）。示例 `10005_1000005,10021_1000021`
> （两例均为 fold-0 train、`case_csPCa=YES`）；**不要使用阴性病例**（如 `10006_1000006`，`csPCa=NO`）
> 作为「阳性病例」示例。病例号从 `data/metadata/picai_manifest.csv` 与
> `data/splits/picai_nnunet_split_cases.csv` 查得，不得凭记忆编造。
>
> 显存报告的语义：`process_memory.*` 是**当前 PyTorch 进程**的分配统计
> （`peak_allocated_bytes/peak_reserved_bytes` 为整个 `trainer_run` = build+train+validation 的
> high-water mark，**没有**独立的 train/validation 峰值）；`gpu.free_bytes_*`/`total_bytes` 来自
> `torch.cuda.mem_get_info`，是**整张卡**状态（含外部进程占用）。若报告 `status=CUDA_OOM`，
> 报告仍会落盘并保留失败阶段与峰值，且**不会**自动缩小 batch/patch 或切 CPU。

## 2. 预期输出

| 步骤 | 产物 |
|---|---|
| A | `outputs/diagnostics/m0_resenc/<run>/setup_check.json`（plan/split/架构身份摘要） |
| B | loader smoke 摘要 + run manifest（`outputs/diagnostics/m0_resenc/<run>/`） |
| C | overfit 日志与 loss 曲线（独立 diagnostic 目录） |
| D | `outputs/checkpoints/m0_resenc/<run>/`、`outputs/metrics/m0_resenc/<run>/`（epoch CSV、validation_cases CSV、run_manifest.json） |
| E | `outputs/diagnostics/<model_id>/<run_name>/diagnostics/gpu_memory_profile.json`（运行时峰值显存；成功 / CUDA OOM / ERROR 都写） |

## 3. 进度观察

- 训练显示 epoch 进度（每 epoch 250 iteration）；全体积验证（每 5 epoch 一次，223 例）显示病例级进度；
- 日志环境可用 `--no-progress`；
- 结束摘要必须包含：`epochs_done`、`validation_events_completed`、耗时与输出路径。

## 4. 成功判据

1. **A**：打印 `encoder_type=residual_encoder`、`architecture_sha256=8bb5127e…`、参数量 `148,193,644`；
2. **B**：`seg_dtype=uint8`、`seg_labels ⊆ {0,1}`、`labels_binary=True`（存储态 seg 可含 nnU-Net 哨兵
   `-1`，读取适配层按官方 `RemoveLabelTansform(-1,0)` 语义映射为 0）、patch 与冻结 plan 一致、
   输入通道 = 3、无 NaN/Inf；
3. **C**：loss 下降（仅诊断，不代表有效）；
4. **D**：
   - **峰值显存被实测并记录**（这是当前 `NOT_MEASURED` 的关闭条件，见 `docs/STATUS.md` B5；
     实测载体是步骤 E 的 `gpu_memory_profile.json`，把它作为 D 的准入证据）；
   - 无 OOM；正常产生 checkpoint；全体积验证覆盖 223 / 阳性 63 / 阴性 160，数量不一致时必须报错且不保存 best；
   - 训练正常结束，产出最终配置/plan/split 哈希与完整 run manifest。

## 5. 失败与转向

| 失败模式 | 转向 |
|---|---|
| 显存超限（冻结 `[16,320,320]` / batch=2） | 按 `docs/research_plan.md` §19 **统一缩减所有 M0–M4 的 blocks_per_stage**（不单独缩减 M4、不上 L/XL），升版 + 记录新架构哈希 + 重跑合成回归；属待冻结决策 D9 |
| residual shortcut / stage shape / 浅层 skip 旁路错误 | 排查实现与契约（`docs/P2_M0_ResidualEncoder_v23.md` §6/§7），不通过增加 attention/Transformer 绕过 |
| 训练不稳定或收敛异常 | 先核对增强 `pending_parity`、损失口径、采样与 LR 协议，不改变已冻结的数据/split/plan |
| G0-R 冻结变更预处理 | 必须使用新派生数据与重新预处理的 plan；旧 run 只能标记 feasibility |

## 6. 记录

- 真实命令、环境、耗时、结果、失败与产物 → `docs/experiment_log.md`；
- 代码/配置变更 → `docs/Development_Log.md`；
- G2 判定与状态 → `docs/STATUS.md`（G2 通过条件见 `docs/research_plan.md` §14）。
