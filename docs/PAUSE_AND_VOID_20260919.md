# 操作暂停与前置规定作废声明（2026-09-19，研究者指令）

> 本文件是**一次性事件记录**：研究者于 2026-09-19 指示「全部文件进行整理，所有操作暂停，规定全部作废」。
> 状态类事实（当前是否暂停、哪些前置被作废）以 `docs/STATUS.md` 为准；本文件保留**声明内容、范围、
> 后果与撤销方法**，以便日后可追溯、可回滚。

---

## 1. 声明与范围

- **声明人**：项目研究者（用户）
- **时间**：2026-09-19（本地时间）
- **指令内容**：
  1. **所有操作暂停**——代理不再启动任何新操作（数据转换、物化、QC、验证、推理、评测、训练），直到研究者解除；
  2. **前置规定作废**——下列"启动前必须满足"的限制**不再作为任何操作的启动前置条件**；
  3. **文件整理**——对仓库与文档做一次现状整理与索引（见 §5）。

**不改变的部分**：所有既有数据、产物、日志、提交历史与归档哈希**原样保留**（见 §4）。本声明**没有**删除、
覆盖或移动任何既有文件。

## 2. 被作废的前置规定（逐条对应原文位置）

| # | 原文位置 | 被作废的内容 |
|---|---|---|
| 1 | `docs/research_plan.md` §14.1（G0-R 行） | 「为 `INSUFFICIENT_EVIDENCE` 时必须停止。**冻结前不得启动 P2B、正式 M0 或 N0。**」 |
| 2 | `docs/research_plan.md` §14（G1 行） | 以「G0-R/G0-E 已冻结 + N0 预算/checkpoint 规则冻结」作为 G1 的**启动前置** |
| 3 | `docs/runbooks/n0_training.md` §1 | 「启动前必须冻结的三件事」（G0-R 通过 / G0-E 冻结路径 / 预算与 checkpoint 规则） |
| 4 | `docs/runbooks/p2b_m0_validation.md`、`docs/STATUS.md` §4.1 | 「G0-R → P2B → G1」的**工程优先顺序约束** |
| 5 | `docs/STATUS.md` §3（B2/B3/B4/B6/B7/B8/B9） | 上述 blocker 作为**启动前置**的效力（不再阻塞任何 run 的启动） |
| 6 | `docs/research_plan.md` §16 | 「正式 run 前须由用户决定版本控制方案」——**已完成**（B1 于 2026-09-18 关闭，Git 已启用），无剩余约束 |

**两点必须写清（避免误解）**：

- 被作废的是**顺序/前置约束**，**不是**把任何门改判为 PASS。G0-R / G0-E / G0-SAP 的技术状态仍为
  「未冻结 / 未评测」（见 `docs/STATUS.md` §1）；
- `docs/research_plan.md` 的**文本未被修改**（遵循研究者此前的长期指令「该文件不要动」）；
  其条款效力以本声明与 `docs/STATUS.md` 为准。若研究者要求连原文一起改写，见 §8 未决项。

## 3. 作废后的直接影响（诚实说明）

**可以直接做的**（不再需要任何前置）：N0 / M0–M4 训练、推理、评测、P2B 验证、G0-R/G0-E/G0-SAP 工具运行。
命令分别见 `docs/runbooks/`（N0：`n0_training.md` §2；P2B：`p2b_m0_validation.md`；G0-R：`g0_r_alignment_qc.md`）。

**同时必须随结果一起记录的证据等级**（这不是新的限制，而是"未冻结"的客观后果）：

1. 在 G0-R / G0-E / G0-SAP 未冻结、D7 预算差异未冻结的状态下产出的任何结果，其证据身份为
   **exploratory / feasibility**；
2. 因此**不得**在论文、报告或对外材料中表述为「官方基线复现」「确认性比较」「H3 独立确认」；
3. 若要升级为确认性证据，必须在**看到结果之前**重新冻结相应协议条款并登记到 `docs/protocol_changelog.md`；
   事后升级必须在 changelog 中显式标注，并相应降级结论强度。

> 说明：作废前置规定的实际效果是**把上述风险的决定权交回研究者**，而不是让未冻结状态的结果自动变得有效。

## 4. 原样保留、未被触碰的内容（不可"作废"的既有事实）

- **数据**：原始医学数据（`/opt/data/private/lm/data/...`）与全部派生数据（`data/`、`workdir/`）只读未动；
- **既有产物与证据**：
  - `outputs/diagnostics/g0_r_automated/20260918_074219/`（G0-R draft-0.2 失败运行，10 个文件，mtime 仍 09-18 08:00）；
  - `outputs/nnUNet_results/Dataset605_PICAI/nnUNetTrainer__nnUNetPlans__3d_fullres/`（默认 Dice+CE 预实验，Epoch 43 中断）
    与 `.../nnUNetTrainerPICAI_FLCE__nnUNetPlans__3d_fullres/`（FFT 崩溃 run）；
  - `outputs/diagnostics/code_snapshots/g0_r_v02_before_fix_20260918_081501/`（12/12 SHA256 校验通过）；
  - `data/processed/picai_zonal_yuan_v1/`（1500 例 PZ/TZ prior 工程产物）；
- **协议归档**：`configs/protocols/archive/g0_r_alignment_qc_automated_draft_0_2.yaml`（`67c479a0…`）、
  `..._draft_0_3.yaml`（`beda4bbf…`）；
- **Git 历史**：`23ca6cb`（基线）→ `2043552`（B1 记录）→ `b8ac1e0`（G0-R draft-0.4）→ 本次记录提交。

## 5. 文件整理结果（本次）

| 项 | 结果 |
|---|---|
| 仓库工作区 | `git status` 干净；顶层仅有 `AGENTS.md`、`README.md`（无散落临时文件） |
| 版本控制范围 | 跟踪文本文件（`src/ scripts/ configs/ tests/ docs/ AGENTS.md README.md`、`data/metadata`、`data/splits`）；`.gitignore` 排除 `data/{raw,interim,processed}`、`workdir/`、`outputs/` 产物与二进制（`*.pth/*.npz/*.nii*/*.b2nd` 等） |
| 代理临时脚本 | 已清理 `/tmp` 下本轮产生的辅助脚本（不在版本库内） |
| 是否移动/删除文件 | **未移动、未删除、未重命名任何既有文件**（避免破坏 100+ 处文档交叉引用） |
| 如需目录级重组 | 请给出目标结构；我会同步更新全部引用后一次性提交，保证可回滚 |

## 6. 当前运行状态（2026-09-19 记录时快照）

- **本项目：无任何运行中的作业**——无训练 / QC / 推理进程；GPU 空闲（RTX 3090，已用 2 MiB，空闲 24257 MiB，利用率 0%）；
- `tmux` 会话 `lm` 中运行的是**另一个项目**（`/opt/data/private/sh/nnunet_tinyIA_tuning`，Dataset508_WMC
  focus-as-refine），已于 2026-09-19 09:55 完成 Validation 并返回 shell 提示符；
  **本项目未干预该作业**（它不属于本仓库，也不是本项目资源）。

## 7. 如何撤销本声明

```bash
cd /opt/data/private/lm/my-projects
git log --oneline | head -5                     # 找到本声明所在提交（§4 末）
git revert --no-edit <本声明提交>                # 撤销 STATUS/changelog/日志改动
rm docs/PAUSE_AND_VOID_20260919.md              # 如不再需要该记录
git commit -am "revert: 恢复前置规定（研究者决定）"
```

## 8. 未决项（等待研究者确认）

1. `docs/research_plan.md` §14 / §14.1 / §16 的**原文是否也要改写或标注作废**？
   （当前状态：文本未修改，效力以本声明 + `docs/STATUS.md` 为准）
2. 「所有操作暂停」是否解除？解除后**第一项**要跑什么（N0 训练 / G0-R draft-0.4 重跑 / P2B 验证）？
