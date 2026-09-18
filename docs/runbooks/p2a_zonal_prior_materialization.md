# Runbook：PZ/TZ prior 物化（plan-space sidecar）

- **用途**：把 `zonal_{yuan,hevi}.nii.gz` 按 **nnU-Net plan 几何**物化为 loader 可直接读取的
  `<case>.npz + <case>.json + manifest.json`（M3/M4 的 prior 输入）
- **实现**：`src/zonal_reliability_fusion/data/zonal_prior.py`、
  `scripts/data/materialize_zonal_prior.py`
- **状态与阻塞项**：见 `docs/STATUS.md`（B10；本手册只写前置条件与命令，不维护状态）
- **执行人**：研究者（读真实标签 + 写派生数据；代理不得代为执行）

## 0. 前置条件

1. **病例级物化已完成**：`data/processed/picai/cases/<case_id>/` 存在
   `zonal_yuan.nii.gz` / `zonal_hevi.nii.gz` / `lesion.nii.gz`（1500 例，含 WG 缺陷病例的排除记录）；
2. **nnU-Net 预处理已完成**：`workdir/nnUNet_preprocessed/Dataset605_PICAI/` 含 `nnUNetPlans.json`、
   `splits_final.json` 与 `nnUNetPlans_3d_fullres/<case>.b2nd`（oracle 的参照 seg 来源）；
3. 固定环境：`conda activate lm` + `source scripts/env_nnunet.sh`；
4. **不覆盖既有数据**：输出目录默认 `data/processed/picai_zonal_<source>_v1/`，
   绝不触碰 raw NIfTI / `.b2nd` / plan / split；默认拒绝覆盖，重算需显式 `--overwrite`；
5. 本手册的命令**由研究者执行**；代理只整理命令与检查产物。

## 1. 命令（复制执行）

```bash
cd /opt/data/private/lm/my-projects
conda activate lm
source scripts/env_nnunet.sh

# A) 3 例只读 dry-run：完整转换 + oracle + prior 契约检查，零写入（先验证几何是否正确）
python scripts/data/materialize_zonal_prior.py --source yuan --dry-run --limit-cases 3

# B) 全量 1500 例物化（首次；写入 data/processed/picai_zonal_yuan_v1/）
python scripts/data/materialize_zonal_prior.py --source yuan
rc=$?; echo "exit_code=$rc"

# C) 中断后续跑（已通过严格校验的病例跳过，不重算；与 --overwrite 互斥）
python scripts/data/materialize_zonal_prior.py --source yuan --resume

# D) 另一来源（可选；与 yuan 不得混用，输出到独立目录 picai_zonal_hevi_v1）
python scripts/data/materialize_zonal_prior.py --source hevi
```

> `--dry-run` **必须**显式限定子集（`--case-ids` 或正的 `--limit-cases`），否则拒绝启动（防误触全量只读转换）。
> 需要干净日志时加 `--no-progress`；选择子集（`--case-ids`/`--limit-cases`）时**不会**创建或覆盖
> canonical `manifest.json`（子集运行不得发布数据集级清单）。

## 2. 预期输出

| 路径 | 内容 |
|---|---|
| `data/processed/picai_zonal_yuan_v1/<case>.npz` | plan-space PZ/TZ 两通道（float32，`[2,D,H,W]`） |
| `data/processed/picai_zonal_yuan_v1/<case>.json` | sidecar 元数据：`case_id`/`source`/`configuration`/`plans_sha256`/`input_sha256`/`output_shape`/`output_dtype`/`output_channel_semantics`/`output_sha256`/`oracle` |
| `data/processed/picai_zonal_yuan_v1/manifest.json` | 数据集级 canonical 清单（仅全量且 `n_failed==0` 时原子发布；含逐例 `metadata_sha256`/`output_sha256`） |
| `data/processed/picai_zonal_yuan_v1/materialization_summary.json` | 本次运行的统计（成功/跳过/失败、耗时、失败清单） |

## 3. 进度观察

- 逐例 tqdm 进度条（当前/总数/比例/速率/ETA）；`--no-progress` 关闭；
- dry-run 同样显示逐例进度（转换与 oracle 仍逐例执行）；
- 结束打印：`ok=… skipped=… failed=…`、输出目录、统计文件路径。

## 4. 成功判据

1. **A（dry-run）**：`ok=3 skipped=0 failed=0`，退出码 `0`，且输出目录**未被创建**（零写入）；
   打印的 `plans sha256`、`transpose_forward`、`target_spacing` 与冻结 plan 一致；
2. **B（全量）**：`ok=1500 skipped=0 failed=0`；出现
   `canonical manifest 已发布: …/manifest.json  n_cases=1500`；退出码 `0`；
3. **下游可读**：`validate_m0_setup.py --config configs/experiments/m4_conditioned_gate_picai_3d_fullres_v24.yaml`
   由 `PRIOR_NOT_READY` 变为 `PRIOR_READY`（`array_hash_checked=true`，逐例数组 SHA256 已校验）；
4. **失败时**：退出码 `1`，且**不会**创建也不会覆盖 `manifest.json`（避免「半个数据集」被当成完整清单）。

## 5. 失败与转向

| 失败模式 | 含义 | 转向 |
|---|---|---|
| `failed>0`（逐例） | 该例缺输入、oracle 不一致、prior 契约违规（finite/[0,1]/PZ+TZ≤1）或写盘失败 | 看 `materialization_summary.json` 的失败清单；修复后 `--resume` 重跑（不会重算已完成例） |
| 退出码 `2` | 运行前的覆盖预检失败（病例集合 ≠ fold 0 的 train∪val） | 检查 `splits_final.json` 与 `--case-ids`/`--limit-cases`，不要用子集冒充全量 |
| `plans_sha256` 不一致 | plan 已变更，旧 sidecar 与新 plan 不再匹配 | **必须重跑物化**（`--overwrite`），不得沿用手工修改的文件 |
| 中断（断电/Ctrl-C） | 可能留下临时文件；成对提交保证不会留下「只有 npz 没有 json」的残缺对 | 直接 `--resume` 重跑 |
| `source` 混用（yuan/hevi） | 两个来源的空间/标签口径不同 | 保持独立输出目录，禁止在同一配置里混用 |

## 6. 记录

- 实际命令、环境、耗时、结果、失败与产物 → `docs/experiment_log.md`；
- 代码/配置变更 → `docs/Development_Log.md`；
- 状态变化（B10 是否关闭、M3/M4 是否数据就绪）→ `docs/STATUS.md`。
