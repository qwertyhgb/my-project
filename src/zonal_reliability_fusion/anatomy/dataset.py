"""Stage 1 解剖先验的**输入/输出布局**与来源标记。

这个模块回答一个具体问题：**「predicted anatomy prior」到底存放在哪里、以什么形式、
由谁生成？** 因为新主线要求 lesion model 只吃**预测**先验，任何把预测先验与 GT 标签混淆的
路径都会让整个研究结论失效，所以这一层必须显式、可审计。

三层数据的区分（禁止混淆）
--------------------------
======================  ==========================  ================================
层                      内容                        在本项目中的角色
======================  ==========================  ================================
raw inputs              T2W / ADC / HBV             模型输入（只读原始医学数据）
training labels         lesion GT（Dataset605/606） 训练监督与验证参照
predicted priors        P(WG) / P(PZ) / P(TZ)       Stage-1 模型对该病例自身 MRI 的预测
oracle labels           GT WG/PZ/TZ                 仅上界分析，必须标记 ``ORACLE_GT``
======================  ==========================  ================================

为什么必须是 soft probability 而不是硬 mask
------------------------------------------
1. 硬 mask 的边界错误会**直接删除**跨越 WG 边界的病灶（见 :mod:`..lesion.roi` 的设计约束）；
2. 下游的 ROI 与 zone 条件化都需要可微的软证据，而不是 0/1 台阶；
3. 概率图保留了「anatomy 不确定」这一信息，uncertain 区域可以走 residual path 而不是被裁掉。

导出契约
--------
Stage-1 的训练与验证使用 nnU-Net 原生 ``--export-validation-probabilities``，它写出：

- ``<case>.npz``——唯一键 ``probabilities``，形状 ``(3, Z, Y, X)``，顺序与
  ``regions_class_order`` 一致（WG, PZ, TZ）；
- ``<case>.pkl``——nnU-Net 原生物理元数据（``sitk_stuff`` / ``spacing`` /
  ``shape_before_cropping``）；
- ``<case>.nii.gz``——原生 region 导出（取 argmax 后的硬标签）。

**argmax 只在评估与展示时使用**：任何下游病灶模型读的是 ``.npz`` 里的 soft probability，
不是 ``.nii.gz`` 里的硬标签。
"""

from __future__ import annotations

from pathlib import Path

from zonal_reliability_fusion.anatomy.contracts import (
    ANATOMY_CONFIGURATION,
    ANATOMY_DATASET,
    ANATOMY_PREDICTION_SUFFIXES,
)

#: 三个 region 头在 ``probabilities`` 数组第 0 维上的顺序
ANATOMY_PROBABILITY_CHANNELS = ("WG", "PZ", "TZ")

#: anatomy 先验的默认存放根目录（相对项目根）。目录由用户创建，本模块不自动创建。
DEFAULT_ANATOMY_PRIOR_DIR = "workdir/anatomy_priors"

#: 先验来源标记——与 ``evaluation.anatomy_metrics`` 共用同一套取值，避免两处定义漂移
PRIOR_LAYOUT = {
    "produced_by": (
        f"{ANATOMY_DATASET} / {ANATOMY_CONFIGURATION} 的冻结 anatomy 模型，"
        "对该病例**自身** MRI 的原生滑窗推理输出"
    ),
    "required_suffixes": ANATOMY_PREDICTION_SUFFIXES,
    "probability_array": (
        "npz 唯一键 'probabilities'，形状 (3, Z, Y, X)，通道顺序 WG/PZ/TZ，float，值域 [0,1]"
    ),
    "soft_not_binary": (
        "下游只读 soft probability；npz 的 argmax 硬标签仅供评估与展示"
    ),
    "forbidden": (
        "禁止使用 GT WG/PZ/TZ 作为 lesion 模型的推理输入；"
        "禁止手工编辑某个病例的先验；禁止在看到 test 结果后重新生成先验"
    ),
}


def prior_directory(root: str | Path, *, dataset_name: str = "Dataset605_PICAI") -> Path:
    """返回某数据集解剖先验的目录（**只计算路径，不创建目录**）。

    目录布局：``<root>/<dataset_name>/prototype_prior/``。是否存在由用户决定；调用方负责
    在缺失时给出清晰的错误信息，而不是静默生成空先验。
    """
    return Path(root) / str(dataset_name) / "prototype_prior"


def case_prior_paths(directory: str | Path, case_id: str) -> dict[str, Path]:
    """返回一个病例的全部先验文件路径（不检查存在性）。"""
    folder = Path(directory)
    return {
        "probabilities_npz": folder / f"{case_id}.npz",
        "properties_pkl": folder / f"{case_id}.pkl",
        "native_export": folder / f"{case_id}.nii.gz",
    }
