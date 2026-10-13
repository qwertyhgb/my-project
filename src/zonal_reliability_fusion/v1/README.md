# V1 — Shallow Anatomy/Lesion-Conditioned Fusion (FROZEN)

VERSION=V1。三个浅层独立 stems → 24→3 neutral projection → pre-backbone lesionness
与 anatomy 条件残差 → 未改变的单共享 encoder PlainConvUNet。

为保留 Python class module、import identity 和 state-dict keys，原始 implementation 冻结在
`../multimodal/conditioned_fusion.py`，`multimodal/shallow_conditioned_fusion.py` 是显式 V1 再导出。
历史 Trainer 定义仍冻结在 `../nnunet/trainers.py`，由 `FROZEN_V1_TRAINERS` 分层；
支持消融仍在 `../lesion/` 与 `SUPPORTING_TRAINERS`。V2 runtime 不调用这些网络。
共用 Stage-1、评价、采样和 baseline families 不复制。

仅允许不改变既有实验语义的 bug fix / reproducibility / compatibility；研究机制变更必须用新 variant。
`legacy/` 是更早 gate 实验，不属于 V1。包名不改；新 V1 CLI 别名带 `v1_`，旧类名仍解析。
