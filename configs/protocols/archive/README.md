# 协议配置归档（configs/protocols/archive）

本目录保存**历史版本的机器可读协议配置**，按字节原样保留（不添加注释、不重排、不修改），以保证
内容哈希可追溯。档案文件**不是**当前执行载体；当前载体见 `configs/protocols/` 下的同名文件。

| 归档文件 | 协议版本 | 归档日期 | 原 SHA256 | 说明 |
|---|---|---|---|---|
| `g0_r_alignment_qc_automated_draft_0_2.yaml` | `G0-R-AUTOMATED` draft-0.2 | 2026-09-18 | `67c479a055736693d290d017d2fc9bbe2f08558a593bc40cf76414920659c89a` | 首次真实运行（`outputs/diagnostics/g0_r_automated/20260918_074219/`）所使用的配置；该运行 32/32 `REGISTRATION_DIAGNOSTIC_FAILED`、`calibration_passed=false`、候选 `INSUFFICIENT_EVIDENCE` |

## 校验方式

```bash
cd /opt/data/private/lm/my-projects
sha256sum configs/protocols/archive/g0_r_alignment_qc_automated_draft_0_2.yaml
# 期望：67c479a055736693d290d017d2fc9bbe2f08558a593bc40cf76414920659c89a
```

该哈希与失败运行 `run_metadata.json` 中记录的 `config_sha256` 一致（可交叉核对）。

## 与原运行的关系

- 失败运行的产物（JSON/CSV/报告/哈希）**原样保留、不得追溯修改**；其中记录的是 draft-0.2 的
  `config_sha256` 与 `protocol_hash = 9dfdd010fa60abc8dbd4ee91cee565dc85769694daa7a3deabe645de38e8ddd1`；
- 升级到 draft-0.3 后，`protocol_hash` 必然变化（校准分组与阈值推导公式属于协议身份字段）；
- 新旧版本的输出**不得混用**：v0.3 代码读取输出时会校验 `schema_version = g0-r-automated/0.3`，
  v0.2 目录不满足该 schema，属历史证据而非可直接比较的结果。
