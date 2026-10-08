#!/usr/bin/env python3
"""前列腺解剖标签只读审计：来源对应、分区关系、Prostate158 取值、腺体代理 ROI 覆盖。

为什么需要它
------------
阶段一「联合解剖分割」需要一个可核实标签来源与可核实 ROI 生成方式。当前证据缺口是：

1. ``data/processed/picai/cases/<case_id>/wg.nii.gz`` 的**来源未被任何代码或配置记录**
   （仓库内无任何脚本生成 ``data/processed/picai/cases/*``），只有在库的
   ``whole_gland/AI/Bosma22b`` 与 ``whole_gland/AI/Guerbet23`` 两套候选；
2. 物化 WG 与 ``zonal_{yuan,hevi}.nii.gz`` 的 ``PZ ∪ TZ`` 关系（包含/互斥/差异体积）从未测量；
3. ``Prostate158`` 解剖标签的**整数值**从未核实，其**解剖语义**由论文给出，两者是不同证据；
4. 没有任何 ROI 覆盖率统计。

本工具只做**只读**审计，回答上面四个问题。

它**不做**什么
--------------
- 不重采样、不配准、不裁剪、不落盘任何影像；不修改任何输入文件。
- 不在跨网格时自动比较体素（默认判 ``grid_mismatch`` 并记录原因）。
- 不使用压缩文件大小推断标签来源（尺寸不是证据）。
- 不读取、不推断 ``BPH`` / ``PCA`` 的 ``ROI.nii`` 语义。
- 不实现跨数据集两两去重（那属于另一个审计任务）。
- 不产生阶段一数据集、Trainer、预测缓存或实验文档。

三个独立状态（**没有**含糊的 ``all_pass``）
-------------------------------------------
- ``execution_complete``：前置条件成立、``scope == "full_dataset"``、且每个被请求的 stage
  都跑完（含逐例错误记录）。``--max-cases`` 子集模式下恒为 ``false``。
- ``data_valid``：无意外缺失、无读取失败、无非法标签值（含非整数与 NaN/Inf）、
  无非法几何、无必需比较的网格不一致、无重复 ``case_id``、患者—病例关系自洽，
  且各 stage（``wg`` / ``zonal`` / ``prostate158`` / ``roi``）的失败均已汇总。
  子集模式下为 ``null``（不得对全数据做有效性声明）。
- ``source_resolved``：每个物化 WG 都能**唯一**对应到一套候选来源
  （``only_bosma_match`` 或 ``only_guerbet_match``）。**只有当竞争候选也完成了比较且不匹配**
  才允许这样判定；候选缺失/不可读/几何非法/不同网格一律记为 ``not_comparable``，
  结果是 ``match_with_unresolved_uniqueness`` 或 ``no_comparable_candidate``。
  ``both_match``（来源歧义）、``neither_match``、``grid_mismatch``、``missing``、
  ``read_error`` 都算**未解决**，但**不等于执行失败**。
- ``source_audit_status`` 取 ``resolved`` / ``unresolved`` / ``not_evaluated``。
  **未请求来源审计时是 ``not_evaluated``，且 ``source_resolved`` 为 ``null``** ——
  「没检查」绝不构成来源已解决的证据。

「来源不明」、「算法间不一致」与「未评估」是**观测结果**，不是程序错误。

数据完整性不可跳过
------------------
``inputs`` stage（存在性、几何基准、标签契约、患者关系、重复 ID）**恒被执行**，
不能用 ``--stages`` 关闭；因此即使只请求 ``--stages roi`` 也会完成必要检查。

标签值必须先按契约检查，禁止整数截断
------------------------------------
``1.5`` 就是 ``1.5``，不会被读成 ``1``。检查维度、有限性、整数性与允许值集合；
非法取值**绝不**参与 ``arr > 0`` 之类的前景或 ROI 构造。NaN/Inf 只影响涉事病例的结论，
不会让整份报告丢失。

覆盖率分母
----------
参考病灶实例数一律由**实际参考掩膜的 3D 6-邻域连通域**计算，不存在预先给定的数量。
参考标签不可读 / 非法 / 与 T2W 不同网格时，该病例实例数记为 ``null``（未知），
**不当作 0 个实例的阴性病例**；此时 ``denominator_complete`` 为假，
所有「全部参考病灶」口径的比例字段置 ``null``，只在 ``evaluable_subset_only``
中报告显式标注范围的可评估子集统计。

输出
----
只允许写**全新**的 JSON 报告：没有 ``--overwrite``，已存在、与输入文件重合、
后缀属于影像/checkpoint 类、或落在受保护目录下都会被直接拒绝。

证据分级（报告中原样保留，禁止混用）
------------------------------------
- ``original``：``/opt/data/private/lm/data/Prostate/PI-CAI/labels/...`` 下的官方原始标签；
- ``materialized``：``data/processed/picai/cases/<case_id>/...`` 下的物化标签；
- ``existing_report``：``materialization_report.csv`` / ``picai_manifest.csv`` 里已记录的值。

三者可以互相矛盾；本工具**并列报告**，不取其一当真值。

ROI 覆盖审计的边界
------------------
该 stage 评估的是**现有标签**生成 ROI 的覆盖，**不是阶段一预测模型的效果上界**。
ROI 只生成在内存中，不落盘裁剪数据，不修改任何输入。

用法（长任务，由研究者本人运行）
--------------------------------
    cd /opt/data/private/lm/my-projects
    source /root/anaconda3/etc/profile.d/conda.sh
    conda activate lm
    source scripts/env_nnunet.sh
    python scripts/data/audit_prostate_anatomy_labels.py \
        --output outputs/reports/prostate_anatomy_labels_audit_v1.json

成功判据：退出码 0，``status.execution_complete`` / ``data_valid`` / ``source_resolved``
**均为 ``true``**（``null`` 不算为真）。若退出码 2，报告**已经写出**，
先读 ``status.problems`` 与 ``status.source_problems`` 再决定下一步。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from tqdm import tqdm

# --------------------------------------------------------------------------- 常量

PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: 项目内默认路径（全部只读）
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "metadata" / "picai_manifest.csv"
DEFAULT_SPLIT_CASES = PROJECT_ROOT / "data" / "splits" / "picai_nnunet_split_cases.csv"
DEFAULT_MATERIALIZATION_REPORT = (
    PROJECT_ROOT / "data" / "processed" / "picai" / "materialization_report.csv"
)
DEFAULT_CASES_ROOT = PROJECT_ROOT / "data" / "processed" / "picai" / "cases"
DEFAULT_PICAI_LABELS_ROOT = Path("/opt/data/private/lm/data/Prostate/PI-CAI/labels")
DEFAULT_PROSTATE158_ROOT = Path("/opt/data/private/lm/data/Prostate/Prostate158")
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "reports" / "prostate_anatomy_labels_audit_v1.json"

#: 禁止写入的目录前缀（原始数据 / 项目内有既有产物的目录）
FORBIDDEN_OUTPUT_ROOTS: tuple[Path, ...] = (
    Path("/opt/data/private/lm/data"),
    PROJECT_ROOT / "data",
    PROJECT_ROOT / "workdir",
    PROJECT_ROOT / "third_party",
)

#: 物化病例目录内的标签文件名（固定约定）
CASE_T2W = "t2w.nii.gz"
CASE_LESION = "lesion.nii.gz"
CASE_WG = "wg.nii.gz"
CASE_ZONAL_TEMPLATE = "zonal_{source}.nii.gz"

#: 物化 WG 的候选来源（PI-CAI 官方 anatomical_delineations/whole_gland/AI/ 下）
WG_SOURCES = {
    "bosma22b": Path("anatomical_delineations") / "whole_gland" / "AI" / "Bosma22b",
    "guerbet23": Path("anatomical_delineations") / "whole_gland" / "AI" / "Guerbet23",
}
#: 分区标签的候选来源
ZONAL_SOURCES = ("yuan", "hevi")
ZONAL_SOURCE_DIRS = {
    "yuan": Path("anatomical_delineations") / "zonal_pz_tz" / "AI" / "Yuan23",
    "hevi": Path("anatomical_delineations") / "zonal_pz_tz" / "AI" / "HeviAI23",
}
#: 与 ``prepare_picai_nnunet.py`` 一致的标签约定（代码约定，非影像核验结果）
ZONAL_LABEL_PZ = 1
ZONAL_LABEL_TZ = 2
ZONAL_ALLOWED_VALUES = (0, ZONAL_LABEL_PZ, ZONAL_LABEL_TZ)

#: 物化 WG 与 lesion 的合法取值（二值）
WG_ALLOWED_VALUES = (0, 1)
LESION_ALLOWED_VALUES = (0, 1)

#: 标签契约：WG / zonal / lesion / Prostate158 各自独立定义（维度、允许值集合、语义来源）
LABEL_CONTRACT_WG = "wg"
LABEL_CONTRACT_ZONAL = "zonal"
LABEL_CONTRACT_LESION = "lesion"
LABEL_CONTRACT_PROSTATE158_ANATOMY = "prostate158_anatomy"

#: 标签检查结论
LABEL_STATUS_OK = "ok"
LABEL_STATUS_EMPTY = "empty"  # 合法空标签（有限且全部为 0）
LABEL_STATUS_NON_FINITE = "non_finite"  # 含 NaN / +Inf / -Inf
LABEL_STATUS_WRONG_NDIM = "wrong_ndim"
LABEL_STATUS_ILLEGAL_VALUES = "illegal_values"  # 含非整数值或超出允许集合
LABEL_STATUS_READ_ERROR = "read_error"
LABEL_NON_EVALUABLE_STATUSES = (
    LABEL_STATUS_NON_FINITE,
    LABEL_STATUS_WRONG_NDIM,
    LABEL_STATUS_ILLEGAL_VALUES,
    LABEL_STATUS_READ_ERROR,
)

LABEL_CONTRACTS: dict[str, dict[str, Any]] = {
    LABEL_CONTRACT_WG: {
        "name": LABEL_CONTRACT_WG,
        "ndim": 3,
        "allowed_values": tuple(float(v) for v in WG_ALLOWED_VALUES),
        "description": "物化 WG：二值 {0,1}",
    },
    LABEL_CONTRACT_ZONAL: {
        "name": LABEL_CONTRACT_ZONAL,
        "ndim": 3,
        "allowed_values": tuple(float(v) for v in ZONAL_ALLOWED_VALUES),
        "description": (
            "分区：{0=background, 1=PZ, 2=TZ}。1/2 的语义来自 "
            "prepare_picai_nnunet.py 的代码约定与 HeviAI23 提交信息，"
            "本审计不据此反推解剖语义"
        ),
    },
    LABEL_CONTRACT_LESION: {
        "name": LABEL_CONTRACT_LESION,
        "ndim": 3,
        "allowed_values": tuple(float(v) for v in LESION_ALLOWED_VALUES),
        "description": "物化 lesion：二值 {0,1}（已由物化阶段二值化）",
    },
    LABEL_CONTRACT_PROSTATE158_ANATOMY: {
        "name": LABEL_CONTRACT_PROSTATE158_ANATOMY,
        "ndim": 3,
        #: 允许集合**未知**（整数映射与解剖语义都未核实）→ 只检查有限性/整数性/维度，
        #: 不声明合法取值集合，也不据此反推解剖语义
        "allowed_values": None,
        "description": (
            "Prostate158 解剖标注：整数映射未核实，允许集合未知；"
            "只做有限性/整数性/维度检查"
        ),
    },
}

#: 已知（有据可查）的 WG 缺失记录：**只有**明确列出且附来源的状态才被当作「已知缺失」。
#: 依据 ``data/processed/picai/materialization_audit.json`` 的 ``wg_excluded_case``
#: （case 11050_1001070，status=excluded_known_faulty_bosma22b）与
#: ``materialization_summary.json`` 的 ``wg_excluded`` 列表。
#: 其他任何非 ``exact`` 的 ``wg_status`` 一律按「**未被认可的**状态」记录，
#: 不得自动当成允许的已知缺失。
KNOWN_WG_MISSING_STATUSES = ("excluded_known_faulty_bosma22b",)

#: 3D 6-邻域（scipy.ndimage 结构元素阶数），与 ``scripts/evaluate_segmentation.py``
#: 的 ``CONNECTIVITY`` 保持一致；测试会核对两者相等，避免出现第二套实例定义。
CONNECTIVITY = 1

#: WG 内容对应结论
WG_ONLY_BOSMA = "only_bosma_match"
WG_ONLY_GUERBET = "only_guerbet_match"
WG_BOTH = "both_match"
WG_NEITHER = "neither_match"
WG_MISSING = "missing"
WG_GRID_MISMATCH = "grid_mismatch"
#: 与某候选内容一致，但竞争候选**未完成比较** → 唯一性未解决
WG_MATCH_UNRESOLVED = "match_with_unresolved_uniqueness"
#: 没有任何候选完成比较（全部缺失/不可读/几何非法/不同网格）
WG_NO_COMPARABLE_CANDIDATE = "no_comparable_candidate"
WG_READ_ERROR = "read_error"
WG_OUTCOMES = (
    WG_ONLY_BOSMA,
    WG_ONLY_GUERBET,
    WG_BOTH,
    WG_NEITHER,
    WG_MATCH_UNRESOLVED,
    WG_NO_COMPARABLE_CANDIDATE,
    WG_MISSING,
    WG_GRID_MISMATCH,
    WG_READ_ERROR,
)
#: 唯一确定来源的结论（``both_match`` 是来源歧义，``match_with_unresolved_uniqueness``
#: 是唯一性未解决，都不算已解决）
WG_RESOLVED_OUTCOMES = (WG_ONLY_BOSMA, WG_ONLY_GUERBET)

#: 单个候选来源的比较状态
CANDIDATE_MATCH = "match"
CANDIDATE_MISMATCH = "mismatch"
CANDIDATE_NOT_COMPARABLE = "not_comparable"

#: 分区并集与 WG 的关系标签
REL_EQUAL = "union_equals_wg"
REL_UNION_SUBSET = "union_subset_of_wg"
REL_WG_SUBSET = "wg_subset_of_union"
REL_PARTIAL = "partial_overlap"
REL_DISJOINT = "disjoint"

#: ROI 候选来源
ROI_CANDIDATES = ("wg_materialized", "yuan_pz_tz_union", "hevi_pz_tz_union")
ROI_FALLBACK_FULL_FOV = "fallback_full_t2w_fov"
ROI_FROM_LABEL = "from_label_largest_component_bbox"

#: ROI 来源可用性（与「前景区间是否为空」区分）
ROI_SOURCE_OK = "ok"
ROI_SOURCE_MISSING = "missing"
ROI_SOURCE_READ_ERROR = "read_error"
ROI_SOURCE_GEOMETRY_INVALID = "geometry_invalid"
ROI_SOURCE_LABEL_NOT_USABLE = "label_not_usable"
ROI_SOURCE_EMPTY = "empty_label"
#: 来源不可用 → 必须回退完整 T2W FOV（不重采样、不丢弃病例）
ROI_SOURCE_UNAVAILABLE_STATUSES = (
    ROI_SOURCE_MISSING,
    ROI_SOURCE_READ_ERROR,
    ROI_SOURCE_GEOMETRY_INVALID,
    ROI_SOURCE_LABEL_NOT_USABLE,
    ROI_SOURCE_EMPTY,
)

#: 参考病灶（覆盖率分母）可用性。**不可用 ≠ 空病灶**：
#: 空病灶是真实阴性（实例数 0，可计入分母），不可用是未知（分母不完整）。
REFERENCE_EVALUABLE_NONEMPTY = "evaluable_nonempty"
REFERENCE_EVALUABLE_EMPTY = "evaluable_empty"
REFERENCE_UNAVAILABLE = "unavailable"

#: 几何比较容差：``rtol`` 显式为 0，只允许绝对容差，避免相对容差掩盖尺度差异。
GEOMETRY_ATOL = 1e-4
GEOMETRY_RTOL = 0.0
#: direction 余弦矩阵的正交归一容差
DIRECTION_ATOL = 1e-4

#: Prostate158 解剖语义依据（**证据与推断分离**；整数值不能确定语义）
PROSTATE158_SEMANTIC_BASIS = {
    "primary_paper": {
        "citation": (
            "Adams LC, Makowski MR, Engel G, Rattunde M, Busch F, Asbach P, Niehues SM, "
            "Vinayahalingam S, van Ginneken B, Litjens G, Bressem KK. "
            "Prostate158 - An expert-annotated 3T MRI dataset and algorithm for prostate "
            "cancer detection. Comput Biol Med 2022;148:105817."
        ),
        "doi": "10.1016/j.compbiomed.2022.105817",
        "statement": (
            "论文摘要明确写明其解剖像素级标注为 central gland (central zone and "
            "transitional zone) 与 peripheral zone。即论文的解剖类别是 CG 与 PZ，"
            "其中 TZ 是 CG 的组成部分，而不是与 CG 并列的第三类。"
        ),
        "evidence_strength": "primary_literature_abstract",
    },
    "official_repository": {
        "url": "https://github.com/kbressem/prostate158",
        "statement": (
            "官方 README 的 baseline 指标表把三列写作 Transitional Zone / Peripheral Zone / "
            "Cancer，并在 inter-rater 表中给出 tz / pz / tu 三组列名。"
            "README 未给出标签整数值定义，也未说明 'Transitional Zone' 与论文 'central gland' "
            "的关系。"
        ),
        "evidence_strength": "official_repository_readme",
    },
    "local_file_evidence": {
        "statement": (
            "本地 train/valid/test CSV 的解剖列名为 t2_anatomy_reader1 / t2_anatomy_reader2；"
            "test/interrater.csv 含 dice_interrater_tz / _pz / _tu 三组列。"
            "列名沿用 README 的 tz/pz 措辞，不构成解剖语义证据。"
        ),
        "evidence_strength": "local_metadata_column_names",
    },
    "semantic_status": "semantic_unresolved",
    "semantic_note": (
        "论文（CG + PZ）与官方仓库 README（用 'Transitional Zone' 一词）表述不一致，"
        "且两处都未给出标签整数映射。因此："
        "(1) 不得把 CG 改名为 TZ；"
        "(2) 不得仅凭整数值集合反推解剖语义；"
        "(3) 在读到论文正文或官方标签说明前，解剖语义保持 semantic_unresolved，"
        "整数值集合只作为独立事实单独报告。"
    ),
}

#: 可选 stage（``inputs`` 恒被执行，见 ``Inputs.stages``）
STAGE_CHOICES = ("inputs", "wg", "zonal", "prostate158", "roi")

#: 退出码
EXIT_OK = 0
EXIT_UNRESOLVED = 2
EXIT_PREFLIGHT = 3


class AuditError(RuntimeError):
    """前置条件不满足（环境、路径、必需列缺失等），fail-closed。"""


# --------------------------------------------------------------------------- 小工具


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    """把 numpy / Path / 非有限数转成可 JSON 序列化的形式（NaN/Inf 转字符串，不静默变 0）。"""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return value
    return value


def _ratio(numerator: float, denominator: float) -> float | None:
    """安全除法：分母 0 → ``None``（不是 0，也不是 1）。"""
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def _dice(mask_a: np.ndarray, mask_b: np.ndarray) -> float | None:
    """二值掩膜 Dice；两侧都空 → ``None``；一侧空 → 0.0。"""
    a = np.count_nonzero(mask_a)
    b = np.count_nonzero(mask_b)
    if a == 0 and b == 0:
        return None
    inter = np.count_nonzero(mask_a & mask_b)
    return 2.0 * inter / float(a + b)


def read_csv_rows(path: Path, *, key: str) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """读取 CSV，返回 ``(rows, diagnostics)``；``diagnostics`` 记录重复键与缺失列。"""
    if not path.is_file():
        raise AuditError(f"缺少必需 CSV：{path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    diagnostics: dict[str, Any] = {
        "path": str(path),
        "n_rows": len(rows),
        "fieldnames": fieldnames,
        "missing_required_columns": [c for c in (key,) if c not in fieldnames],
    }
    counts: dict[str, int] = {}
    for row in rows:
        value = row.get(key, "")
        counts[value] = counts.get(value, 0) + 1
    duplicates = sorted(k for k, v in counts.items() if v > 1)
    diagnostics["duplicate_keys"] = duplicates
    diagnostics["n_duplicate_key_rows"] = int(sum(counts[d] for d in duplicates))
    if diagnostics["missing_required_columns"]:
        raise AuditError(
            f"{path} 缺少必需列 {diagnostics['missing_required_columns']}；"
            f"实际列：{fieldnames}"
        )
    return rows, diagnostics


def _value_repr(value: float) -> str:
    """观测标签值的精确规范化文本。

    整数值去掉多余小数位；**分数值原样保留**（``repr``），绝不截断成整数。
    NaN / ±Inf 有独立文本，不参与取值集合判断。
    """
    as_float = float(value)
    if math.isnan(as_float):
        return "nan"
    if math.isinf(as_float):
        return "inf" if as_float > 0 else "-inf"
    if as_float.is_integer() and abs(as_float) < 2**53:
        return str(int(as_float))
    return repr(as_float)


def _non_finite_masks(array: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 ``(nan_mask, pos_inf_mask, neg_inf_mask)``；非浮点 dtype 全为 False。"""
    if np.issubdtype(array.dtype, np.floating):
        return (
            np.isnan(array),
            np.isposinf(array),
            np.isneginf(array),
        )
    empty = np.zeros(array.shape, dtype=bool)
    return empty, empty, empty


def label_foreground_mask(array: np.ndarray, contract_name: str) -> np.ndarray:
    """只把**允许的非零取值**当作前景。

    未通过契约检查（非法值、非有限值、维度错误）的数组绝不允许经 ``arr > 0``
    之类的捷径构造「可信」前景或 ROI：非法体素在这里被显式排除。
    """
    contract = LABEL_CONTRACTS[contract_name]
    allowed = contract["allowed_values"]
    as_float = array.astype(np.float64, copy=False)
    nan_mask, pos_inf, neg_inf = _non_finite_masks(array)
    finite = ~(nan_mask | pos_inf | neg_inf)
    if allowed is None:
        # 允许集合未知（例如 Prostate158 未核实的整数映射）：
        # 只把「有限的非零值」当前景，且必须与有限性一起判定
        return (as_float != 0.0) & finite
    allowed_nonzero = [v for v in allowed if v != 0.0]
    if not allowed_nonzero:
        return np.zeros(array.shape, dtype=bool)
    return np.isin(as_float, allowed_nonzero) & finite


def inspect_label_array(array: np.ndarray, contract_name: str) -> dict:
    """按标签契约检查数组：维度、有限性、整数性、允许取值集合、空标签。

    **不先转换成整数**：观测值以精确文本原样保留（``1.5`` 就是 ``'1.5'``，不会变成 ``'1'``）。
    返回结构纯 JSON 安全（不含数组），可整份写入报告。
    """
    contract = LABEL_CONTRACTS[contract_name]
    allowed_raw = contract["allowed_values"]
    values_unconstrained = allowed_raw is None
    allowed = [] if values_unconstrained else list(allowed_raw)
    allowed_reprs = {_value_repr(v) for v in allowed}

    info: dict[str, Any] = {
        "contract": contract_name,
        "contract_description": contract["description"],
        "allowed_values": (
            None
            if values_unconstrained
            else sorted(allowed_reprs, key=lambda s: float(s))
        ),
        "allowed_values_unconstrained": values_unconstrained,
        "expected_ndim": int(contract["ndim"]),
        "ndim": int(np.ndim(array)),
        "shape": [int(s) for s in np.shape(array)],
        "dtype": str(array.dtype),
        "integer_dtype": bool(np.issubdtype(array.dtype, np.integer)),
        "status": None,
        "problems": [],
        "observed_values": [],
        "observed_values_are_integer": None,
        "n_nan": 0,
        "n_pos_inf": 0,
        "n_neg_inf": 0,
        "n_non_finite": 0,
        "n_non_integer_voxels": 0,
        "n_illegal_voxels": 0,
        "n_foreground_voxels": 0,
        "n_total_voxels": int(np.size(array)),
        "is_empty": None,
    }

    if info["ndim"] != info["expected_ndim"]:
        info["problems"].append(
            f"维度为 {info['ndim']}D（shape={info['shape']}），"
            f"契约 {contract_name} 要求 {contract['ndim']}D"
        )

    nan_mask, pos_inf, neg_inf = _non_finite_masks(array)
    info["n_nan"] = int(np.count_nonzero(nan_mask))
    info["n_pos_inf"] = int(np.count_nonzero(pos_inf))
    info["n_neg_inf"] = int(np.count_nonzero(neg_inf))
    info["n_non_finite"] = info["n_nan"] + info["n_pos_inf"] + info["n_neg_inf"]
    if info["n_non_finite"]:
        info["problems"].append(
            f"含非有限值：NaN {info['n_nan']}、+Inf {info['n_pos_inf']}、"
            f"-Inf {info['n_neg_inf']}（共 {info['n_non_finite']} 体素）"
        )

    finite_mask = ~(nan_mask | pos_inf | neg_inf)
    finite_values = array[finite_mask]
    as_float = finite_values.astype(np.float64, copy=False)
    if as_float.size:
        unique_finite = np.unique(as_float)
        info["observed_values"] = [_value_repr(v) for v in unique_finite.tolist()]
        integral = np.equal(as_float, np.rint(as_float))
        info["observed_values_are_integer"] = bool(integral.all())
        info["n_non_integer_voxels"] = int(np.count_nonzero(~integral))
        if info["n_non_integer_voxels"]:
            info["problems"].append(
                f"含非整数值 {info['n_non_integer_voxels']} 体素；"
                f"观测取值（精确）{info['observed_values']}"
            )
        if values_unconstrained:
            # 允许集合未知：只承认「非整数」为非法，不对整数取值集合作任何声明
            info["n_illegal_voxels"] = 0
        else:
            illegal_mask = ~np.isin(as_float, allowed)
            info["n_illegal_voxels"] = int(np.count_nonzero(illegal_mask))
            if info["n_illegal_voxels"]:
                illegal_values = np.unique(as_float[illegal_mask])
                info["problems"].append(
                    f"含超出允许集合 {sorted(allowed_reprs, key=float)} 的取值："
                    f"{[_value_repr(v) for v in illegal_values.tolist()]}"
                    f"（{info['n_illegal_voxels']} 体素）"
                )
    else:
        info["observed_values"] = []
        info["observed_values_are_integer"] = None

    info["n_foreground_voxels"] = int(
        np.count_nonzero(label_foreground_mask(array, contract_name))
    )
    info["is_empty"] = info["n_foreground_voxels"] == 0

    if info["ndim"] != info["expected_ndim"]:
        info["status"] = LABEL_STATUS_WRONG_NDIM
    elif info["n_non_finite"]:
        info["status"] = LABEL_STATUS_NON_FINITE
    elif info["n_illegal_voxels"] or info["n_non_integer_voxels"]:
        info["status"] = LABEL_STATUS_ILLEGAL_VALUES
    elif info["is_empty"]:
        info["status"] = LABEL_STATUS_EMPTY
    else:
        info["status"] = LABEL_STATUS_OK
    return info


def label_is_usable(info: dict) -> bool:
    """标签是否可用于后续统计（合法空标签**可用**：它对应真实的「无前景」）。"""
    return info.get("status") not in LABEL_NON_EVALUABLE_STATUSES


# --------------------------------------------------------------------------- 几何


class Geometry:
    """NIfTI 几何（SimpleITK 语义）：``size_xyz`` / ``spacing_xyz`` / ``origin_xyz`` / direction。"""

    __slots__ = ("path", "size_xyz", "spacing_xyz", "origin_xyz", "direction")

    def __init__(self, path: Path, size_xyz, spacing_xyz, origin_xyz, direction):
        self.path = Path(path)
        self.size_xyz = tuple(int(v) for v in size_xyz)
        self.spacing_xyz = tuple(float(v) for v in spacing_xyz)
        self.origin_xyz = tuple(float(v) for v in origin_xyz)
        self.direction = tuple(float(v) for v in direction)

    @property
    def shape_zyx(self) -> tuple[int, int, int]:
        """numpy 数组轴序 (z, y, x)。"""
        return (self.size_xyz[2], self.size_xyz[1], self.size_xyz[0])

    @property
    def spacing_zyx(self) -> tuple[float, float, float]:
        """numpy 数组轴序下的 spacing (z, y, x)。"""
        return (self.spacing_xyz[2], self.spacing_xyz[1], self.spacing_xyz[0])

    @property
    def voxel_volume_mm3(self) -> float:
        return float(self.spacing_xyz[0] * self.spacing_xyz[1] * self.spacing_xyz[2])

    def as_dict(self) -> dict:
        return {
            "path": str(self.path),
            "size_xyz": list(self.size_xyz),
            "spacing_xyz": list(self.spacing_xyz),
            "origin_xyz": list(self.origin_xyz),
            "direction": list(self.direction),
        }

    # ------------------------------------------------------------------ 合法性
    def validity_problems(self) -> list[str]:
        """几何自身是否合法：维度、spacing 正且有限、origin 有限、direction 正交归一。"""
        problems: list[str] = []
        if len(self.size_xyz) != 3:
            problems.append(f"维度为 {len(self.size_xyz)}D，需要 3D（size={self.size_xyz}）")
        if len(self.spacing_xyz) != 3:
            problems.append(f"spacing 长度为 {len(self.spacing_xyz)}，需要 3")
        finite_spacing = np.isfinite(np.asarray(self.spacing_xyz, dtype=np.float64))
        if not bool(finite_spacing.all()):
            problems.append(f"spacing 含非有限值：{self.spacing_xyz}")
        elif not all(value > 0 for value in self.spacing_xyz):
            problems.append(f"spacing 非全正：{self.spacing_xyz}")
        if len(self.origin_xyz) != 3 or not bool(
            np.isfinite(np.asarray(self.origin_xyz, dtype=np.float64)).all()
        ):
            problems.append(f"origin 非法：{self.origin_xyz}")
        direction = np.asarray(self.direction, dtype=np.float64)
        if direction.size != 9:
            problems.append(f"direction 长度不是 9：{self.direction}")
        elif not bool(np.isfinite(direction).all()):
            problems.append("direction 含非有限值")
        else:
            matrix = direction.reshape(3, 3)
            gram = matrix @ matrix.T
            if not np.allclose(gram, np.eye(3), rtol=GEOMETRY_RTOL, atol=DIRECTION_ATOL):
                problems.append("direction 不是正交归一矩阵")
            elif abs(abs(float(np.linalg.det(matrix))) - 1.0) > DIRECTION_ATOL:
                problems.append(f"direction 行列式绝对值偏离 1：{float(np.linalg.det(matrix))}")
        if self.size_xyz and any(int(s) <= 0 for s in self.size_xyz):
            problems.append(f"size 含非正数：{self.size_xyz}")
        return problems

    @property
    def is_valid(self) -> bool:
        return not self.validity_problems()

    # ------------------------------------------------------------------ 比较
    def same_grid_as(
        self,
        other: "Geometry",
        *,
        atol: float = GEOMETRY_ATOL,
        rtol: float = GEOMETRY_RTOL,
    ) -> bool:
        """严格同网格判定：``rtol`` 显式为 0，只允许绝对容差。"""
        if self.size_xyz != other.size_xyz:
            return False
        return (
            np.allclose(self.spacing_xyz, other.spacing_xyz, rtol=rtol, atol=atol)
            and np.allclose(self.origin_xyz, other.origin_xyz, rtol=rtol, atol=atol)
            and np.allclose(self.direction, other.direction, rtol=rtol, atol=atol)
        )

    def grid_differences(
        self,
        other: "Geometry",
        *,
        atol: float = GEOMETRY_ATOL,
        rtol: float = GEOMETRY_RTOL,
    ) -> list[str]:
        diffs: list[str] = []
        if self.size_xyz != other.size_xyz:
            diffs.append(f"size {self.size_xyz} != {other.size_xyz}")
        if not np.allclose(self.spacing_xyz, other.spacing_xyz, rtol=rtol, atol=atol):
            diffs.append(f"spacing {self.spacing_xyz} != {other.spacing_xyz}")
        if not np.allclose(self.origin_xyz, other.origin_xyz, rtol=rtol, atol=atol):
            diffs.append(f"origin {self.origin_xyz} != {other.origin_xyz}")
        if not np.allclose(self.direction, other.direction, rtol=rtol, atol=atol):
            diffs.append("direction differs")
        return diffs


def compare_geometry(reference: "Geometry", other: "Geometry") -> dict:
    """统一几何比较：各自合法性 + 可比较性 + 差异明细。"""
    reference_problems = reference.validity_problems()
    other_problems = other.validity_problems()
    comparable = not reference_problems and not other_problems
    same = reference.same_grid_as(other) if comparable else False
    return {
        "reference": reference.as_dict(),
        "other": other.as_dict(),
        "reference_valid": not reference_problems,
        "other_valid": not other_problems,
        "reference_problems": reference_problems,
        "other_problems": other_problems,
        "comparable": comparable,
        "same_grid": same,
        "differences": reference.grid_differences(other) if comparable else [],
    }


def read_geometry(path: Path) -> Geometry:
    """只读文件头（不读体素）；SimpleITK 延迟导入，便于 ``--help`` 与合成测试。"""
    import SimpleITK as sitk  # noqa: PLC0415 - 延迟导入

    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.ReadImageInformation()
    return Geometry(
        path,
        reader.GetSize(),
        reader.GetSpacing(),
        reader.GetOrigin(),
        reader.GetDirection(),
    )


def read_array(path: Path) -> np.ndarray:
    """读体素为 numpy 数组，轴序 (z, y, x)。"""
    import SimpleITK as sitk  # noqa: PLC0415

    return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))


# --------------------------------------------------------------------------- 输入解析


class Inputs:
    """一次审计的全部输入路径与元数据（只读）。"""

    def __init__(self, args: argparse.Namespace):
        self.manifest_path = Path(args.manifest)
        self.split_cases_path = Path(args.split_cases)
        self.materialization_report_path = Path(args.materialization_report)
        self.cases_root = Path(args.cases_root)
        self.picai_labels_root = Path(args.picai_labels_root)
        self.prostate158_root = Path(args.prostate158_root)
        self.margin_mm = float(args.roi_margin_mm)
        self.max_cases = args.max_cases
        requested = tuple(args.stages)
        missing = [s for s in requested if s not in STAGE_CHOICES]
        if missing:
            raise AuditError(f"未知 stage：{missing}；可选 {list(STAGE_CHOICES)}")
        self.requested_stages = requested
        # ``inputs``（数据完整性）是**强制**审计项：不能因为只请求了其他 stage 而被跳过。
        ordered = ["inputs"] + [s for s in STAGE_CHOICES if s != "inputs" and s in requested]
        self.stages = tuple(ordered)
        # 明确记录「被请求但本工具不评估」的范围，供状态判定使用
        self.not_evaluated_stages = tuple(
            s for s in STAGE_CHOICES if s != "inputs" and s not in requested
        )

        self.manifest_rows: list[dict[str, str]] = []
        self.manifest_diagnostics: dict[str, Any] = {}
        self.split_by_case: dict[str, str] = {}
        self.split_diagnostics: dict[str, Any] = {}
        self.patient_by_case: dict[str, str] = {}
        self.report_by_case: dict[str, dict[str, str]] = {}
        self.materialization_diagnostics: dict[str, Any] = {}
        self.case_ids: tuple[str, ...] = ()
        self.input_hashes: dict[str, str] = {}

    # ---------------------------------------------------------------- 解析
    def resolve(self) -> None:
        if not self.cases_root.is_dir():
            raise AuditError(f"物化病例目录不存在：{self.cases_root}")
        if not self.picai_labels_root.is_dir():
            raise AuditError(f"PI-CAI labels 目录不存在：{self.picai_labels_root}")

        self.manifest_rows, self.manifest_diagnostics = read_csv_rows(
            self.manifest_path, key="case_id"
        )
        split_rows, self.split_diagnostics = read_csv_rows(
            self.split_cases_path, key="case_id"
        )
        report_rows, self.materialization_diagnostics = read_csv_rows(
            self.materialization_report_path, key="case_id"
        )
        for row in split_rows:
            self.split_by_case[row["case_id"]] = row.get("split", "")
            self.patient_by_case[row["case_id"]] = row.get("patient_id", "")
        for row in report_rows:
            self.report_by_case[row["case_id"]] = row

        # 病例集合以 manifest 为准（**不用 glob 成功读取的病例当分母**）
        self.case_ids = tuple(row["case_id"] for row in self.manifest_rows)
        if self.max_cases is not None:
            self.case_ids = self.case_ids[: int(self.max_cases)]

        for path in (
            self.manifest_path,
            self.split_cases_path,
            self.materialization_report_path,
        ):
            self.input_hashes[str(path)] = sha256_file(path)

    # ---------------------------------------------------------------- 路径
    def case_dir(self, case_id: str) -> Path:
        return self.cases_root / case_id

    def case_label(self, case_id: str, filename: str) -> Path:
        return self.case_dir(case_id) / filename

    def wg_source_path(self, case_id: str, source: str) -> Path:
        return self.picai_labels_root / WG_SOURCES[source] / f"{case_id}.nii.gz"

    def zonal_source_path(self, case_id: str, source: str) -> Path:
        return self.picai_labels_root / ZONAL_SOURCE_DIRS[source] / f"{case_id}.nii.gz"

    def existing_report_record(self, case_id: str) -> dict[str, str]:
        return self.report_by_case.get(case_id, {})

    @property
    def n_cases_from_manifest(self) -> int:
        return len(self.manifest_rows)

    @property
    def is_subset(self) -> bool:
        """是否处于 ``--max-cases`` 子集模式（此时不得对全数据做有效性声明）。"""
        return len(self.case_ids) < self.n_cases_from_manifest

    def known_missing_wg(self, case_id: str) -> dict:
        """WG 缺失/异常的分类。

        **只有** ``KNOWN_WG_MISSING_STATUSES`` 中显式列出的状态才算「已被认可的已知缺失」；
        其他任何非 ``exact`` 的 ``wg_status`` 都记为 ``unrecognized_status``，
        既不算已知缺失，也不算程序错误 —— 但必须逐例出现在报告里并进入问题汇总。
        """
        record = self.report_by_case.get(case_id)
        status = (record or {}).get("wg_status", "")
        if not status or status == "exact":
            return {"category": "none", "wg_status": status, "source": ""}
        if status in KNOWN_WG_MISSING_STATUSES:
            return {
                "category": "recognized_known_missing",
                "wg_status": status,
                "source": (
                    "data/processed/picai/materialization_audit.json 的 wg_excluded_case "
                    "与 materialization_summary.json 的 wg_excluded"
                ),
            }
        return {
            "category": "unrecognized_status",
            "wg_status": status,
            "source": (
                "materialization_report.csv 记录了非 exact 的 wg_status，"
                "但它不在本工具认可的已知缺失清单里，需人工确认后才能当作已知缺失"
            ),
        }

    def as_dict(self) -> dict:
        return {
            "manifest": str(self.manifest_path),
            "split_cases": str(self.split_cases_path),
            "materialization_report": str(self.materialization_report_path),
            "cases_root": str(self.cases_root),
            "picai_labels_root": str(self.picai_labels_root),
            "prostate158_root": str(self.prostate158_root),
            "roi_margin_mm": self.margin_mm,
            "requested_stages": list(self.requested_stages),
            "stages_run": list(self.stages),
            "stages_not_evaluated": list(self.not_evaluated_stages),
            "inputs_stage_forced": True,
            "inputs_stage_forced_note": (
                "数据完整性检查恒被执行，不能通过 --stages 关闭"
            ),
            "max_cases": self.max_cases,
            "scope": "subset" if self.is_subset else "full_dataset",
            "input_sha256": dict(self.input_hashes),
            "n_cases_from_manifest": self.n_cases_from_manifest,
            "n_cases_audited": len(self.case_ids),
        }


# --------------------------------------------------------------------------- stage 1


def audit_case_inputs(inputs: Inputs, case_id: str) -> dict:
    """单病例输入完整性：存在性、几何、来源文件、报告记录、患者关系。"""
    entry: dict[str, Any] = {
        "case_id": case_id,
        "patient_id": inputs.patient_by_case.get(case_id, ""),
        "split": inputs.split_by_case.get(case_id, "unknown"),
        "files": {},
        "geometry": {},
        "geometry_validity": {},
        "geometry_problems": [],
        "geometry_notes": [],
        "unexpected_missing": [],
        "optional_missing": [],
        "read_errors": [],
        "geometry_read_errors": [],
    }

    # 必需：t2w / lesion（缺失即 data_valid=False）
    required = {
        CASE_T2W: inputs.case_label(case_id, CASE_T2W),
        CASE_LESION: inputs.case_label(case_id, CASE_LESION),
    }
    # 可选：wg / zonal（缺失逐例记录，但不直接判 data_valid=False；
    # wg 的已知缺失由 materialization_report 的 wg_status 单独记录）
    optional = {
        CASE_WG: inputs.case_label(case_id, CASE_WG),
        CASE_ZONAL_TEMPLATE.format(source="yuan"): inputs.case_label(
            case_id, CASE_ZONAL_TEMPLATE.format(source="yuan")
        ),
        CASE_ZONAL_TEMPLATE.format(source="hevi"): inputs.case_label(
            case_id, CASE_ZONAL_TEMPLATE.format(source="hevi")
        ),
    }
    for source in WG_SOURCES:
        optional[f"source_wg_{source}"] = inputs.wg_source_path(case_id, source)
    for source in ZONAL_SOURCES:
        optional[f"source_zonal_{source}"] = inputs.zonal_source_path(case_id, source)

    for name, path in required.items():
        exists = path.is_file()
        entry["files"][name] = {"path": str(path), "exists": exists, "required": True}
        if not exists:
            entry["unexpected_missing"].append(name)
    for name, path in optional.items():
        exists = path.is_file()
        entry["files"][name] = {"path": str(path), "exists": exists, "required": False}
        if not exists:
            entry["optional_missing"].append(name)

    # 逐文件读头 + 合法性检查（t2w / lesion / wg / zonal 各自独立）
    geometries: dict[str, Geometry] = {}
    for name, path in {**required, **optional}.items():
        if not entry["files"][name]["exists"]:
            continue
        try:
            geometry = read_geometry(path)
        except Exception as exc:  # noqa: BLE001 - 读取失败必须逐例记录
            target = (
                entry["read_errors"] if name in required else entry["geometry_read_errors"]
            )
            target.append(
                {"item": name, "path": str(path), "error": f"{type(exc).__name__}: {exc}"}
            )
            continue
        geometries[name] = geometry
        entry["geometry"][name] = geometry.as_dict()
        problems = geometry.validity_problems()
        if problems:
            entry["geometry_validity"][name] = problems
            entry["geometry_problems"].append({"item": name, "problems": problems})

    # 统一 T2W 几何基准：lesion / wg / zonal 都必须与 T2W 处在同一网格
    t2w = geometries.get(CASE_T2W)
    entry["t2w_geometry_usable"] = bool(t2w is not None and t2w.is_valid)
    comparisons: dict[str, dict] = {}
    if t2w is None:
        entry["geometry_notes"].append("T2W 几何不可用：无法建立几何基准")
    else:
        for name in (CASE_LESION, CASE_WG, CASE_ZONAL_TEMPLATE.format(source="yuan"),
                     CASE_ZONAL_TEMPLATE.format(source="hevi")):
            other = geometries.get(name)
            if other is None:
                continue
            comparison = compare_geometry(t2w, other)
            comparisons[name] = comparison
            if not comparison["comparable"]:
                entry["geometry_notes"].append(
                    f"{name}: 几何非法或 T2W 几何非法，无法比较"
                )
            elif not comparison["same_grid"]:
                entry["geometry_notes"].append(f"{name}: 网格与 T2W 不同")
    entry["geometry_comparisons"] = comparisons
    lesion_comparison = comparisons.get(CASE_LESION)
    entry["lesion_on_t2w_grid"] = (
        None
        if lesion_comparison is None
        else bool(lesion_comparison["same_grid"])
    )

    # WG 缺失分类：已知缺失（有据可查）与未被认可的状态分开记录
    known = inputs.known_missing_wg(case_id)
    entry["known_missing_wg"] = known
    record = inputs.existing_report_record(case_id)
    entry["existing_report_record"] = {
        "wg_status": record.get("wg_status", ""),
        "wg_unique": record.get("wg_unique", ""),
        "yuan_unique": record.get("yuan_unique", ""),
        "hevi_unique": record.get("hevi_unique", ""),
        "yuan_resampled": record.get("yuan_resampled", ""),
        "hevi_resampled": record.get("hevi_resampled", ""),
        "lesion_grid": record.get("lesion_grid", ""),
        "canonical_source": record.get("canonical_source", ""),
    }
    # 患者—病例关系
    patient_id = entry["patient_id"]
    entry["patient_relation_ok"] = bool(
        patient_id and case_id.startswith(patient_id + "_")
    )
    return entry


def stage_inputs(inputs: Inputs, *, progress: bool) -> dict:
    entries: list[dict] = []
    iterator = tqdm(
        inputs.case_ids, desc="inputs", unit="case", disable=not progress
    )
    for case_id in iterator:
        entries.append(audit_case_inputs(inputs, case_id))

    n_unexpected_missing = sum(1 for e in entries if e["unexpected_missing"])
    n_read_errors = sum(1 for e in entries if e["read_errors"])
    n_geometry_read_errors = sum(1 for e in entries if e["geometry_read_errors"])
    n_geometry_problems = sum(1 for e in entries if e["geometry_problems"])
    n_missing_wg = sum(1 for e in entries if not e["files"][CASE_WG]["exists"])
    n_recognized_known_missing_wg = sum(
        1
        for e in entries
        if e["known_missing_wg"]["category"] == "recognized_known_missing"
    )
    n_unrecognized_wg_status = sum(
        1 for e in entries if e["known_missing_wg"]["category"] == "unrecognized_status"
    )
    n_patient_relation_bad = sum(1 for e in entries if not e["patient_relation_ok"])
    n_t2w_geometry_unusable = sum(1 for e in entries if not e["t2w_geometry_usable"])
    n_lesion_not_on_t2w_grid = sum(
        1 for e in entries if e["lesion_on_t2w_grid"] is False
    )
    n_lesion_grid_unknown = sum(
        1 for e in entries if e["lesion_on_t2w_grid"] is None
    )

    # 患者是否跨外层划分
    patient_splits: dict[str, set[str]] = {}
    for entry in entries:
        patient_splits.setdefault(entry["patient_id"], set()).add(entry["split"])
    patients_spanning = sorted(
        pid for pid, splits in patient_splits.items() if len(splits) > 1
    )

    return {
        "n_cases": len(entries),
        "n_cases_t2w_present": sum(1 for e in entries if e["files"][CASE_T2W]["exists"]),
        "n_cases_lesion_present": sum(
            1 for e in entries if e["files"][CASE_LESION]["exists"]
        ),
        "n_cases_wg_present": sum(1 for e in entries if e["files"][CASE_WG]["exists"]),
        "n_cases_wg_missing": n_missing_wg,
        # 已知缺失（有据可查）与未被认可的 wg_status 严格分开
        "n_cases_wg_missing_recognized_known": n_recognized_known_missing_wg,
        "n_cases_wg_status_unrecognized": n_unrecognized_wg_status,
        "known_missing_wg_status_allowlist": list(KNOWN_WG_MISSING_STATUSES),
        "recognized_known_missing_wg_cases": sorted(
            e["case_id"]
            for e in entries
            if e["known_missing_wg"]["category"] == "recognized_known_missing"
        ),
        "unrecognized_wg_status_cases": sorted(
            e["case_id"]
            for e in entries
            if e["known_missing_wg"]["category"] == "unrecognized_status"
        ),
        "n_cases_unexpected_missing": n_unexpected_missing,
        "n_cases_read_error": n_read_errors,
        "n_cases_geometry_read_error": n_geometry_read_errors,
        "n_cases_geometry_problem": n_geometry_problems,
        "n_cases_t2w_geometry_unusable": n_t2w_geometry_unusable,
        "n_cases_lesion_not_on_t2w_grid": n_lesion_not_on_t2w_grid,
        "n_cases_lesion_grid_unknown": n_lesion_grid_unknown,
        "n_cases_patient_relation_bad": n_patient_relation_bad,
        "n_patients_spanning_splits": len(patients_spanning),
        "patients_spanning_splits": patients_spanning,
        "manifest_duplicate_case_ids": inputs.manifest_diagnostics.get(
            "duplicate_keys", []
        ),
        "split_duplicate_case_ids": inputs.split_diagnostics.get("duplicate_keys", []),
        "materialization_report_duplicate_case_ids": inputs.materialization_diagnostics.get(
            "duplicate_keys", []
        ),
        "case_ids_in_manifest_absent_from_split": sorted(
            set(inputs.case_ids) - set(inputs.split_by_case)
        ),
        "case_ids_in_materialization_report_absent_from_manifest": sorted(
            set(inputs.report_by_case) - {r["case_id"] for r in inputs.manifest_rows}
        ),
        "forced_note": (
            "inputs stage 恒被执行（不能通过 --stages 关闭），"
            "因此即使只请求 roi 也会完成必要的数据完整性检查"
        ),
        "per_case": entries,
    }


# --------------------------------------------------------------------------- stage 2


def _wg_candidate_report(inputs: Inputs, case_id: str, source: str) -> dict:
    """单个候选来源的读取与几何状态（不涉及体素）。"""
    path = inputs.wg_source_path(case_id, source)
    report: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
        "read_status": None,
        "geometry": None,
        "geometry_problems": [],
        "grid_comparable": False,
        "grid_differences": [],
        "match_result": CANDIDATE_NOT_COMPARABLE,
        "not_comparable_reasons": [],
        "shape": None,
        "shape_matches_materialized": None,
        "voxel_equal": None,
        "unequal_voxels": None,
        "label_info": None,
    }
    if not report["exists"]:
        report["read_status"] = "missing"
        report["not_comparable_reasons"].append("候选文件不存在")
        return report
    try:
        geometry = read_geometry(path)
    except Exception as exc:  # noqa: BLE001
        report["read_status"] = "read_error"
        report["not_comparable_reasons"].append(
            f"候选头读取失败：{type(exc).__name__}: {exc}"
        )
        return report
    report["read_status"] = "ok"
    report["geometry"] = geometry.as_dict()
    problems = geometry.validity_problems()
    report["geometry_problems"] = problems
    if problems:
        report["not_comparable_reasons"].append(f"候选几何非法：{problems}")
    return report


def compare_wg_case(inputs: Inputs, case_id: str) -> dict:
    """物化 WG 与两套候选来源的内容对应（同网格才比较体素；否则不自动重采样）。

    保守判定：**只有竞争候选完成比较且不匹配**，才能断言来源唯一。
    候选缺失 / 不可读 / 几何非法 / 不同网格一律记为 ``not_comparable``，
    **绝不**当成「已比较且不匹配」。
    """
    result: dict[str, Any] = {
        "case_id": case_id,
        "split": inputs.split_by_case.get(case_id, "unknown"),
        "outcome": None,
        "reason": "",
        "materialized": {"path": str(inputs.case_label(case_id, CASE_WG))},
        "materialized_usable": False,
        "candidates": {},
        "match_summary": {},
        "unique_source_unresolved": None,
        "known_missing_wg": inputs.known_missing_wg(case_id),
    }
    for source in WG_SOURCES:
        result["candidates"][source] = _wg_candidate_report(inputs, case_id, source)

    wg_path = inputs.case_label(case_id, CASE_WG)
    if not wg_path.is_file():
        result["outcome"] = WG_MISSING
        result["reason"] = "materialized wg.nii.gz 不存在"
        result["unique_source_unresolved"] = True
        return result
    try:
        wg_geom = read_geometry(wg_path)
    except Exception as exc:  # noqa: BLE001
        result["outcome"] = WG_READ_ERROR
        result["reason"] = f"物化 WG 头读取失败：{type(exc).__name__}: {exc}"
        result["unique_source_unresolved"] = True
        return result

    wg_problems = wg_geom.validity_problems()
    result["materialized"]["geometry"] = wg_geom.as_dict()
    result["materialized"]["geometry_problems"] = wg_problems
    if wg_problems:
        result["outcome"] = WG_READ_ERROR
        result["reason"] = f"物化 WG 几何非法，无法作为比较基准：{wg_problems}"
        result["unique_source_unresolved"] = True
        return result

    # 候选的网格可比较性
    for source, report in result["candidates"].items():
        if report["read_status"] != "ok" or report["geometry_problems"]:
            continue
        candidate_geom = read_geometry(Path(report["path"]))
        comparison = compare_geometry(wg_geom, candidate_geom)
        report["grid_comparable"] = bool(comparison["comparable"])
        report["grid_differences"] = comparison["differences"]
        if not comparison["same_grid"]:
            report["grid_comparable"] = False
            report["not_comparable_reasons"].append(
                "与物化 WG 网格不一致；按约定不自动重采样，因此不做体素比较"
            )

    comparable = [
        s for s, r in result["candidates"].items() if r["grid_comparable"]
    ]
    if not comparable:
        reasons_detail = {
            s: r["not_comparable_reasons"] for s, r in result["candidates"].items()
        }
        any_grid = any(
            r["not_comparable_reasons"] and "网格不一致" in r["not_comparable_reasons"][-1]
            for r in result["candidates"].values()
        )
        result["outcome"] = WG_GRID_MISMATCH if any_grid else WG_NO_COMPARABLE_CANDIDATE
        result["reason"] = (
            "没有任何候选来源完成比较（缺失/不可读/几何非法/网格不一致）；"
            "按约定不自动重采样，因此不做体素比较"
        )
        result["match_summary"] = reasons_detail
        result["unique_source_unresolved"] = True
        return result

    try:
        wg_array = read_array(wg_path)
    except Exception as exc:  # noqa: BLE001
        result["outcome"] = WG_READ_ERROR
        result["reason"] = f"物化 WG 体素读取失败：{type(exc).__name__}: {exc}"
        result["unique_source_unresolved"] = True
        return result

    wg_label = inspect_label_array(wg_array, LABEL_CONTRACT_WG)
    result["materialized"].update(
        {
            "geometry": wg_geom.as_dict(),
            "label_info": wg_label,
            "voxels_fg": wg_label["n_foreground_voxels"],
            "volume_mm3": float(
                wg_label["n_foreground_voxels"] * wg_geom.voxel_volume_mm3
            ),
        }
    )
    if not label_is_usable(wg_label):
        result["outcome"] = WG_READ_ERROR
        result["reason"] = (
            "物化 WG 未通过标签契约检查，不能作为比较基准："
            f"{wg_label['problems']}"
        )
        result["unique_source_unresolved"] = True
        return result
    result["materialized_usable"] = True

    for source in comparable:
        report = result["candidates"][source]
        try:
            array = read_array(Path(report["path"]))
        except Exception as exc:  # noqa: BLE001
            report["read_status"] = "read_error"
            report["match_result"] = CANDIDATE_NOT_COMPARABLE
            report["not_comparable_reasons"].append(
                f"候选体素读取失败：{type(exc).__name__}: {exc}"
            )
            continue
        report["shape"] = [int(s) for s in array.shape]
        report["shape_matches_materialized"] = array.shape == wg_array.shape
        if array.shape != wg_array.shape:
            report["match_result"] = CANDIDATE_NOT_COMPARABLE
            report["not_comparable_reasons"].append(
                f"形状不一致：{list(array.shape)} != {list(wg_array.shape)}"
            )
            continue
        report["label_info"] = inspect_label_array(array, LABEL_CONTRACT_WG)
        equal = bool(np.array_equal(array, wg_array))
        report["voxel_equal"] = equal
        report["unequal_voxels"] = int(np.count_nonzero(array != wg_array))
        report["match_result"] = CANDIDATE_MATCH if equal else CANDIDATE_MISMATCH

    matches = [
        s for s, r in result["candidates"].items() if r["match_result"] == CANDIDATE_MATCH
    ]
    mismatches = [
        s
        for s, r in result["candidates"].items()
        if r["match_result"] == CANDIDATE_MISMATCH
    ]
    not_comparable = [
        s
        for s, r in result["candidates"].items()
        if r["match_result"] == CANDIDATE_NOT_COMPARABLE
    ]
    result["match_summary"] = {
        "matched_candidates": sorted(matches),
        "mismatched_candidates": sorted(mismatches),
        "not_comparable_candidates": sorted(not_comparable),
        "all_candidates_completed_comparison": not not_comparable,
    }

    if len(matches) == 2:
        outcome = WG_BOTH
        reason = (
            "物化 WG 与 Bosma22b、Guerbet23 都逐体素一致 → **来源歧义**；"
            "体素一致只证明内容对应，不能证明历史生成来源"
        )
        unresolved = True
    elif len(matches) == 1 and not_comparable:
        outcome = WG_MATCH_UNRESOLVED
        reason = (
            f"与候选 {matches[0]} 逐体素一致，但另一候选未完成比较"
            f"（{not_comparable}）；**唯一性未解决**，不得断言来源唯一"
        )
        unresolved = True
    elif len(matches) == 1:
        matched = matches[0]
        outcome = WG_ONLY_BOSMA if matched == "bosma22b" else WG_ONLY_GUERBET
        reason = (
            f"仅与 {matched} 逐体素一致，且另一候选已完成比较且不匹配"
            "（内容对应，仍不证明历史生成来源）"
        )
        unresolved = False
    elif not matches and not_comparable:
        outcome = WG_NO_COMPARABLE_CANDIDATE
        reason = (
            f"没有候选匹配，但仍有候选未完成比较（{not_comparable}）"
            "→ 结论不完整"
        )
        unresolved = True
    else:
        outcome = WG_NEITHER
        reason = "与两套候选来源都不逐体素一致（两候选均已完成比较）"
        unresolved = True

    result["outcome"] = outcome
    result["reason"] = reason
    result["unique_source_unresolved"] = unresolved
    result["voxel_identity_note"] = (
        "体素一致只能证明**内容对应**，不能证明历史生成来源"
    )
    return result


def stage_wg_content_match(inputs: Inputs, *, progress: bool) -> dict:
    entries: list[dict] = []
    iterator = tqdm(
        inputs.case_ids, desc="wg-correspondence", unit="case", disable=not progress
    )
    for case_id in iterator:
        entries.append(compare_wg_case(inputs, case_id))

    counts: dict[str, int] = {outcome: 0 for outcome in WG_OUTCOMES}
    per_split: dict[str, dict[str, int]] = {}
    for entry in entries:
        counts[entry["outcome"]] = counts.get(entry["outcome"], 0) + 1
        bucket = per_split.setdefault(
            entry["split"], {outcome: 0 for outcome in WG_OUTCOMES}
        )
        bucket[entry["outcome"]] = bucket.get(entry["outcome"], 0) + 1
    candidate_status_counts: dict[str, dict[str, int]] = {}
    for source in WG_SOURCES:
        status: dict[str, int] = {}
        for entry in entries:
            report = entry["candidates"].get(source, {})
            key = f"{report.get('read_status')}/{report.get('match_result')}"
            status[key] = status.get(key, 0) + 1
        candidate_status_counts[source] = dict(sorted(status.items()))
    n_unresolved = sum(
        1 for entry in entries if entry.get("unique_source_unresolved")
    )
    return {
        "n_cases": len(entries),
        "counts": counts,
        "counts_per_split": per_split,
        "candidate_status_counts": candidate_status_counts,
        "n_source_resolved": sum(counts[o] for o in WG_RESOLVED_OUTCOMES),
        "n_unique_source_unresolved": n_unresolved,
        "n_ambiguous": counts.get(WG_BOTH, 0),
        "n_match_with_unresolved_uniqueness": counts.get(WG_MATCH_UNRESOLVED, 0),
        "n_no_comparable_candidate": counts.get(WG_NO_COMPARABLE_CANDIDATE, 0),
        "source_ambiguity_note": (
            "两套候选来源同时逐体素一致时记为来源歧义（both_match）；"
            "这不是程序失败，也不是来源已解决"
        ),
        "conservative_rule_note": (
            "候选缺失/不可读/几何非法/不同网格一律记为 not_comparable，"
            "绝不当作「已比较且不匹配」；只有竞争候选完成比较且不匹配时，"
            "才写 only_bosma_match / only_guerbet_match"
        ),
        "size_is_not_evidence_note": (
            "本 stage 不使用文件大小或压缩体积推断来源；"
            "体素一致只能证明内容对应，不能证明历史生成来源"
        ),
        "per_case": entries,
    }


# --------------------------------------------------------------------------- stage 3


def zonal_case_relations(
    inputs: Inputs, case_id: str, source: str
) -> dict:
    """单个分区来源与物化 WG 的关系，以及与该来源自身标签值的统计。"""
    entry: dict[str, Any] = {
        "case_id": case_id,
        "source": source,
        "split": inputs.split_by_case.get(case_id, "unknown"),
        "empty_label": False,
        "illegal_values": [],
        "abnormal_geometry": "",
        "geometry_valid": None,
        "geometry_problems": [],
        "label_usable": None,
        "comparable_to_wg": False,
        "skip_code": None,
        "comparison_skipped_reason": "",
    }
    zonal_path = inputs.case_label(case_id, CASE_ZONAL_TEMPLATE.format(source=source))
    wg_path = inputs.case_label(case_id, CASE_WG)
    if not zonal_path.is_file():
        entry["skip_code"] = "zonal_file_missing"
        entry["comparison_skipped_reason"] = "分区标签文件不存在"
        return entry
    try:
        zonal_geom = read_geometry(zonal_path)
    except Exception as exc:  # noqa: BLE001
        entry["skip_code"] = "zonal_header_read_error"
        entry["comparison_skipped_reason"] = f"分区头读取失败：{type(exc).__name__}: {exc}"
        return entry
    zonal_geometry_problems = zonal_geom.validity_problems()
    entry["geometry_valid"] = not zonal_geometry_problems
    entry["geometry_problems"] = zonal_geometry_problems
    if zonal_geometry_problems:
        entry["abnormal_geometry"] = f"分区几何非法：{zonal_geometry_problems}"
    try:
        zonal_arr = read_array(zonal_path)
    except Exception as exc:  # noqa: BLE001
        entry["skip_code"] = "zonal_voxel_read_error"
        entry["comparison_skipped_reason"] = (
            f"分区体素读取失败：{type(exc).__name__}: {exc}"
        )
        return entry

    # 标签契约检查（不做整数截断；非法值绝不参与前景/ROI 构造）
    zonal_label = inspect_label_array(zonal_arr, LABEL_CONTRACT_ZONAL)
    entry["label_info"] = zonal_label
    entry["unique_values"] = zonal_label["observed_values"]
    entry["illegal_values"] = [
        v for v in zonal_label["observed_values"] if v not in {
            _value_repr(a) for a in ZONAL_ALLOWED_VALUES
        }
    ]
    entry["label_usable"] = label_is_usable(zonal_label)
    voxel_volume = zonal_geom.voxel_volume_mm3
    entry["voxel_volume_mm3"] = voxel_volume
    pz = zonal_arr == ZONAL_LABEL_PZ
    tz = zonal_arr == ZONAL_LABEL_TZ
    union = label_foreground_mask(zonal_arr, LABEL_CONTRACT_ZONAL)
    entry["n_voxels"] = {
        _value_repr(v): int(np.count_nonzero(np.asarray(zonal_arr, dtype=np.float64) == v))
        for v in [float(x) for x in ZONAL_ALLOWED_VALUES]
    }
    entry["volume_mm3"] = {
        key: float(count * voxel_volume) for key, count in entry["n_voxels"].items()
    }
    entry["union_voxels"] = int(np.count_nonzero(union))
    entry["union_volume_mm3"] = float(np.count_nonzero(union) * voxel_volume)
    entry["pz_voxels"] = int(np.count_nonzero(pz))
    entry["tz_voxels"] = int(np.count_nonzero(tz))
    entry["pz_tz_overlap_voxels"] = int(np.count_nonzero(pz & tz))
    entry["empty_label"] = not bool(np.any(union))
    # 单标签整数图里 PZ 与 TZ 天然互斥；非零重叠意味着多标签叠加，属异常
    if entry["pz_tz_overlap_voxels"]:
        entry["abnormal_geometry"] = (
            f"PZ 与 TZ 掩膜重叠 {entry['pz_tz_overlap_voxels']} 体素（单标签整数图不应重叠）"
        )
    if not entry["label_usable"]:
        entry["skip_code"] = "zonal_label_illegal"
        entry["comparison_skipped_reason"] = (
            "分区标签未通过契约检查，不用它构造前景或与 WG 比较："
            f"{zonal_label['problems']}"
        )
        return entry

    if not wg_path.is_file():
        entry["skip_code"] = "wg_missing"
        entry["comparison_skipped_reason"] = "物化 WG 缺失，无法比较"
        return entry
    try:
        wg_geom = read_geometry(wg_path)
    except Exception as exc:  # noqa: BLE001
        entry["skip_code"] = "wg_header_read_error"
        entry["comparison_skipped_reason"] = f"WG 头读取失败：{type(exc).__name__}: {exc}"
        return entry
    wg_geometry_problems = wg_geom.validity_problems()
    if wg_geometry_problems:
        entry["skip_code"] = "wg_geometry_invalid"
        entry["comparison_skipped_reason"] = f"WG 几何非法：{wg_geometry_problems}"
        return entry
    if zonal_geometry_problems:
        entry["skip_code"] = "zonal_geometry_invalid"
        entry["comparison_skipped_reason"] = "分区几何非法，不做跨网格比较"
        return entry
    if not zonal_geom.same_grid_as(wg_geom):
        entry["skip_code"] = "grid_mismatch"
        entry["comparison_skipped_reason"] = (
            "分区与物化 WG 网格不一致；按约定不自动重采样，跨网格比较不执行"
        )
        entry["grid_differences"] = zonal_geom.grid_differences(wg_geom)
        return entry
    try:
        wg_arr = read_array(wg_path)
    except Exception as exc:  # noqa: BLE001
        entry["skip_code"] = "wg_voxel_read_error"
        entry["comparison_skipped_reason"] = f"WG 体素读取失败：{type(exc).__name__}: {exc}"
        return entry
    if wg_arr.shape != zonal_arr.shape:
        entry["skip_code"] = "shape_mismatch"
        entry["comparison_skipped_reason"] = "分区与 WG 数组形状不一致"
        return entry

    wg_label = inspect_label_array(wg_arr, LABEL_CONTRACT_WG)
    entry["wg_label_info"] = wg_label
    if not label_is_usable(wg_label):
        entry["skip_code"] = "wg_label_illegal"
        entry["comparison_skipped_reason"] = (
            f"物化 WG 未通过契约检查，不以它作基准：{wg_label['problems']}"
        )
        return entry
    wg_mask = label_foreground_mask(wg_arr, LABEL_CONTRACT_WG)
    entry["comparable_to_wg"] = True
    entry["wg_voxels"] = int(np.count_nonzero(wg_mask))
    entry["union_vs_wg_dice"] = _dice(union, wg_mask)
    union_outside_wg = int(np.count_nonzero(union & ~wg_mask))
    wg_outside_union = int(np.count_nonzero(wg_mask & ~union))
    entry["union_outside_wg_voxels"] = union_outside_wg
    entry["union_outside_wg_volume_mm3"] = float(union_outside_wg * voxel_volume)
    entry["wg_outside_union_voxels"] = wg_outside_union
    entry["wg_outside_union_volume_mm3"] = float(wg_outside_union * voxel_volume)
    entry["wg_outside_union_fraction_of_wg"] = _ratio(
        wg_outside_union, int(np.count_nonzero(wg_mask))
    )
    entry["union_outside_wg_fraction_of_union"] = _ratio(
        union_outside_wg, int(np.count_nonzero(union))
    )
    entry["intersection_voxels"] = int(np.count_nonzero(union & wg_mask))
    if entry["union_voxels"] == 0 and entry["wg_voxels"] == 0:
        entry["relation"] = "both_empty"
    elif union_outside_wg == 0 and wg_outside_union == 0:
        entry["relation"] = REL_EQUAL
    elif union_outside_wg == 0:
        entry["relation"] = REL_UNION_SUBSET
    elif wg_outside_union == 0:
        entry["relation"] = REL_WG_SUBSET
    elif entry["intersection_voxels"] == 0:
        entry["relation"] = REL_DISJOINT
    else:
        entry["relation"] = REL_PARTIAL
    return entry


def zonal_cross_source_agreement(
    inputs: Inputs, case_id: str
) -> dict:
    """Yuan 与 Hevi 之间的 PZ / TZ 一致性（算法间一致性，不是准确度）。"""
    entry: dict[str, Any] = {
        "case_id": case_id,
        "comparable": False,
        "skip_code": None,
        "reason": "",
    }
    arrays: dict[str, np.ndarray] = {}
    geoms: dict[str, Geometry] = {}
    for source in ZONAL_SOURCES:
        path = inputs.case_label(case_id, CASE_ZONAL_TEMPLATE.format(source=source))
        if not path.is_file():
            entry["skip_code"] = f"{source}_file_missing"
            entry["reason"] = f"{source} 标签文件不存在"
            return entry
        try:
            geoms[source] = read_geometry(path)
            arrays[source] = read_array(path)
        except Exception as exc:  # noqa: BLE001
            entry["skip_code"] = f"{source}_read_error"
            entry["reason"] = f"{source} 读取失败：{type(exc).__name__}: {exc}"
            return entry
    for source in ZONAL_SOURCES:
        problems = geoms[source].validity_problems()
        if problems:
            entry["skip_code"] = f"{source}_geometry_invalid"
            entry["reason"] = f"{source} 几何非法：{problems}"
            return entry
        info = inspect_label_array(arrays[source], LABEL_CONTRACT_ZONAL)
        entry[f"{source}_label_info"] = info
        if not label_is_usable(info):
            entry["skip_code"] = f"{source}_label_illegal"
            entry["reason"] = f"{source} 标签未通过契约检查：{info['problems']}"
            return entry
    if not geoms["yuan"].same_grid_as(geoms["hevi"]):
        entry["skip_code"] = "cross_source_grid_mismatch"
        entry["reason"] = "Yuan 与 Hevi 网格不一致；跨网格比较不执行"
        entry["grid_differences"] = geoms["yuan"].grid_differences(geoms["hevi"])
        return entry
    if arrays["yuan"].shape != arrays["hevi"].shape:
        entry["skip_code"] = "cross_source_shape_mismatch"
        entry["reason"] = "Yuan 与 Hevi 数组形状不一致"
        return entry
    entry["comparable"] = True
    for label, name in ((ZONAL_LABEL_PZ, "pz"), (ZONAL_LABEL_TZ, "tz")):
        a = arrays["yuan"] == label
        b = arrays["hevi"] == label
        entry[f"{name}_dice"] = _dice(a, b)
        entry[f"{name}_yuan_voxels"] = int(np.count_nonzero(a))
        entry[f"{name}_hevi_voxels"] = int(np.count_nonzero(b))
        entry[f"{name}_intersection_voxels"] = int(np.count_nonzero(a & b))
    entry["union_dice"] = _dice(
        (arrays["yuan"] > 0), (arrays["hevi"] > 0)
    )
    return entry


def stage_zonal_relations(inputs: Inputs, *, progress: bool) -> dict:
    relations: list[dict] = []
    agreements: list[dict] = []
    iterator = tqdm(
        inputs.case_ids, desc="zonal-relations", unit="case", disable=not progress
    )
    for case_id in iterator:
        for source in ZONAL_SOURCES:
            relations.append(zonal_case_relations(inputs, case_id, source))
        agreements.append(zonal_cross_source_agreement(inputs, case_id))

    summary: dict[str, Any] = {}
    for source in ZONAL_SOURCES:
        rows = [r for r in relations if r["source"] == source]
        dice = [
            r["union_vs_wg_dice"]
            for r in rows
            if r.get("comparable_to_wg") and r.get("union_vs_wg_dice") is not None
        ]
        summary[source] = {
            "n_cases": len(rows),
            "n_comparable_to_wg": sum(1 for r in rows if r.get("comparable_to_wg")),
            "n_empty_label": sum(1 for r in rows if r.get("empty_label")),
            "n_label_usable": sum(1 for r in rows if r.get("label_usable")),
            "n_label_not_usable": sum(
                1 for r in rows if r.get("label_usable") is False
            ),
            "n_illegal_values": sum(1 for r in rows if r.get("illegal_values")),
            "n_geometry_invalid": sum(
                1 for r in rows if r.get("geometry_valid") is False
            ),
            "n_abnormal_geometry": sum(1 for r in rows if r.get("abnormal_geometry")),
            "n_comparison_skipped": sum(
                1 for r in rows if r.get("comparison_skipped_reason")
            ),
            # 结构化跳过原因，供总体状态判定逐项消费
            "skip_code_counts": _counter(
                r.get("skip_code") for r in rows if r.get("skip_code")
            ),
            "relation_counts": _counter(
                r.get("relation") for r in rows if r.get("relation")
            ),
            "union_vs_wg_dice": _distribution(dice),
            "wg_outside_union_fraction_of_wg": _distribution(
                [
                    r["wg_outside_union_fraction_of_wg"]
                    for r in rows
                    if r.get("wg_outside_union_fraction_of_wg") is not None
                ]
            ),
            "value_sets_observed": _counter(
                ";".join(str(v) for v in r.get("unique_values", []))
                for r in rows
            ),
        }
    comparable_agreements = [a for a in agreements if a.get("comparable")]
    return {
        "n_cases": len(inputs.case_ids),
        "per_source_summary": summary,
        "cross_source_agreement": {
            "n_cases": len(agreements),
            "n_comparable": len(comparable_agreements),
            "n_skipped": len(agreements) - len(comparable_agreements),
            "skip_code_counts": _counter(
                a.get("skip_code") for a in agreements if a.get("skip_code")
            ),
            "pz_dice": _distribution(
                [a["pz_dice"] for a in comparable_agreements if a.get("pz_dice") is not None]
            ),
            "tz_dice": _distribution(
                [a["tz_dice"] for a in comparable_agreements if a.get("tz_dice") is not None]
            ),
            "union_dice": _distribution(
                [
                    a["union_dice"]
                    for a in comparable_agreements
                    if a.get("union_dice") is not None
                ]
            ),
        },
        "interpretation_note": (
            "两个自动标签之间的 Dice 是**算法间一致性**，"
            "既不是准确度，也不是标签噪声下限；"
            "这些指标描述差异，不能证明哪一个算法正确。"
        ),
        "per_case_relations": relations,
        "per_case_cross_source": agreements,
    }


# --------------------------------------------------------------------------- stage 4


def _prostate158_queues(root: Path) -> list[dict]:
    """Prostate158 三个队列的 CSV 位置（读取 CSV，不依赖 glob）。"""
    return [
        {
            "queue": "train",
            "csv": root / "train" / "prostate158_train" / "train.csv",
            "base": root / "train" / "prostate158_train",
        },
        {
            "queue": "valid",
            "csv": root / "train" / "prostate158_train" / "valid.csv",
            "base": root / "train" / "prostate158_train",
        },
        {
            "queue": "test",
            "csv": root / "test" / "prostate158_test" / "test.csv",
            "base": root / "test" / "prostate158_test",
        },
    ]


#: Prostate158 解剖列 → 读者标识
PROSTATE158_ANATOMY_COLUMNS = {
    "t2_anatomy_reader1": "reader1",
    "t2_anatomy_reader2": "reader2",
}


def stage_prostate158(inputs: Inputs, *, progress: bool) -> dict:
    """读取 Prostate158 解剖标签的实际值域与几何；语义保持 unresolved。"""
    entries: list[dict] = []
    file_diagnostics: list[dict] = []
    plans: list[tuple[str, str, Path, Path]] = []
    for spec in _prostate158_queues(inputs.prostate158_root):
        if not spec["csv"].is_file():
            file_diagnostics.append(
                {"queue": spec["queue"], "csv": str(spec["csv"]), "exists": False}
            )
            continue
        rows, diagnostics = read_csv_rows(spec["csv"], key="ID")
        file_diagnostics.append({"queue": spec["queue"], **diagnostics})
        for row in rows:
            for column, reader in PROSTATE158_ANATOMY_COLUMNS.items():
                rel = (row.get(column) or "").strip()
                if not rel:
                    continue
                plans.append(
                    (spec["queue"], reader, Path(rel), spec["base"])
                )

    iterator = tqdm(
        plans, desc="prostate158", unit="file", disable=not progress
    )
    for queue, reader, rel, base in iterator:
        path = rel if rel.is_absolute() else base / rel
        entry: dict[str, Any] = {
            "queue": queue,
            "reader": reader,
            "case_id": rel.stem.split("/")[0] if rel.parts else "",
            "path": str(path),
            "exists": path.is_file(),
        }
        if not entry["exists"]:
            entry["status"] = "missing"
            entries.append(entry)
            continue
        try:
            geom = read_geometry(path)
            arr = read_array(path)
        except Exception as exc:  # noqa: BLE001
            entry["status"] = "read_error"
            entry["error"] = f"{type(exc).__name__}: {exc}"
            entries.append(entry)
            continue
        # 允许集合未知 → 只检查维度/有限性/整数性（不做整数截断，不做语义声明）
        label_info = inspect_label_array(arr, LABEL_CONTRACT_PROSTATE158_ANATOMY)
        geometry_problems = geom.validity_problems()
        entry.update(
            {
                "status": "ok",
                "geometry": geom.as_dict(),
                "geometry_problems": geometry_problems,
                "label_info": label_info,
                "unique_values": label_info["observed_values"],
                "label_integer_valued": label_info["observed_values_are_integer"],
                "n_non_finite": label_info["n_non_finite"],
                "n_non_integer_voxels": label_info["n_non_integer_voxels"],
                "foreground_voxels": label_info["n_foreground_voxels"],
                "integer_dtype": label_info["integer_dtype"],
            }
        )
        entries.append(entry)

    by_queue_reader: dict[str, dict] = {}
    for entry in entries:
        key = f"{entry['queue']}/{entry['reader']}"
        bucket = by_queue_reader.setdefault(
            key,
            {
                "n_files": 0,
                "n_ok": 0,
                "n_missing": 0,
                "n_read_error": 0,
                "n_geometry_invalid": 0,
                "n_non_integer": 0,
                "n_non_finite": 0,
                "value_sets": {},
            },
        )
        bucket["n_files"] += 1
        if entry["status"] == "ok":
            bucket["n_ok"] += 1
            if entry["geometry_problems"]:
                bucket["n_geometry_invalid"] += 1
            if entry["n_non_integer_voxels"]:
                bucket["n_non_integer"] += 1
            if entry["n_non_finite"]:
                bucket["n_non_finite"] += 1
            key2 = ";".join(entry["unique_values"])
            bucket["value_sets"][key2] = bucket["value_sets"].get(key2, 0) + 1
        elif entry["status"] == "missing":
            bucket["n_missing"] += 1
        else:
            bucket["n_read_error"] += 1

    observed_value_sets = sorted(
        {
            ";".join(entry["unique_values"])
            for entry in entries
            if entry.get("status") == "ok"
        }
    )
    return {
        "cohort": "Prostate158",
        "semantic_status": PROSTATE158_SEMANTIC_BASIS["semantic_status"],
        "semantic_basis": PROSTATE158_SEMANTIC_BASIS,
        "semantic_rule": (
            "整数值集合与解剖语义是两种独立证据：本 stage 只测量整数值与几何，"
            "不据整数值推断 CG/TZ/PZ 语义；也不得把论文的 central gland 改名为 TZ。"
        ),
        "allowed_values": None,
        "allowed_values_note": (
            "Prostate158 的整数→解剖映射未核实，因此**不声明**合法取值集合；"
            "只检查维度、有限性与整数性"
        ),
        "csv_diagnostics": file_diagnostics,
        "n_files": len(entries),
        "n_ok": sum(1 for e in entries if e.get("status") == "ok"),
        "n_missing": sum(1 for e in entries if e.get("status") == "missing"),
        "n_read_error": sum(1 for e in entries if e.get("status") == "read_error"),
        "n_geometry_invalid": sum(
            1 for e in entries if e.get("geometry_problems")
        ),
        "n_non_integer": sum(1 for e in entries if e.get("n_non_integer_voxels")),
        "n_non_finite": sum(1 for e in entries if e.get("n_non_finite")),
        "observed_value_sets": observed_value_sets,
        "by_queue_reader": by_queue_reader,
        "per_file": entries,
    }


# --------------------------------------------------------------------------- stage 5


def largest_component_bbox(mask: np.ndarray) -> tuple[tuple[int, int], ...] | None:
    """3D 6-邻域最大连通域的包围盒（numpy 轴序 (z,y,x)，stop 为开区间）。"""
    from scipy import ndimage  # noqa: PLC0415

    if not np.any(mask):
        return None
    structure = ndimage.generate_binary_structure(mask.ndim, CONNECTIVITY)
    labelled, n_components = ndimage.label(mask, structure=structure)
    if n_components == 0:
        return None
    counts = np.bincount(labelled.ravel())
    counts[0] = 0
    largest = int(counts.argmax())
    slices = ndimage.find_objects(labelled == largest, max_label=1)
    if not slices:
        return None
    bbox = []
    for axis_slice in slices[0]:
        start = 0 if axis_slice.start is None else int(axis_slice.start)
        stop = mask.shape[len(bbox)] if axis_slice.stop is None else int(axis_slice.stop)
        bbox.append((start, stop))
    return tuple(bbox)


def expand_bbox_mm(
    bbox: Sequence[tuple[int, int]],
    margin_mm: float,
    spacing_zyx: Sequence[float],
    shape_zyx: Sequence[int],
) -> tuple[tuple[int, int], ...]:
    """按物理余量向外扩展包围盒。

    - 余量转体素用 ``ceil``（**向外取整**），保证至少 ``margin_mm`` 的物理余量；
    - 轴顺序：``spacing_zyx`` 必须与 ``bbox`` / ``shape_zyx`` 同为 numpy 轴序 (z,y,x)；
    - 结果裁剪到 ``[0, shape]``。
    """
    if margin_mm < 0:
        raise AuditError(f"ROI 余量必须非负，收到 {margin_mm}")
    expanded: list[tuple[int, int]] = []
    for axis, (start, stop) in enumerate(bbox):
        spacing = float(spacing_zyx[axis])
        if spacing <= 0:
            raise AuditError(f"spacing 必须为正，轴 {axis} 收到 {spacing}")
        # 减去极小量避免浮点噪声把恰好整数倍的余量多推一个体素
        n_voxels = int(math.ceil(margin_mm / spacing - 1e-9)) if margin_mm > 0 else 0
        new_start = max(0, start - n_voxels)
        new_stop = min(int(shape_zyx[axis]), stop + n_voxels)
        expanded.append((new_start, new_stop))
    return tuple(expanded)


def instance_coverages(
    lesion_mask: np.ndarray, roi_bbox: Sequence[tuple[int, int]]
) -> dict:
    """每个参考病灶实例被 ROI 覆盖的比例（分母 = **全部**参考实例）。"""
    from scipy import ndimage  # noqa: PLC0415

    result: dict[str, Any] = {
        "n_instances": 0,
        "instance_coverages": [],
        "n_instances_fully_covered": 0,
        "n_instances_fully_missed": 0,
        "total_lesion_voxels": 0,
        "covered_lesion_voxels": 0,
    }
    if not np.any(lesion_mask):
        return result
    structure = ndimage.generate_binary_structure(lesion_mask.ndim, CONNECTIVITY)
    labelled, n_instances = ndimage.label(lesion_mask, structure=structure)
    result["n_instances"] = int(n_instances)
    if n_instances == 0:
        return result
    total_counts = np.bincount(labelled.ravel())
    slices = list(
        slice(
            max(0, bbox[0]),
            min(lesion_mask.shape[axis], bbox[1]),
        )
        for axis, bbox in enumerate(roi_bbox)
    )
    covered_counts = np.zeros_like(total_counts)
    if all(s.stop > s.start for s in slices):
        sub = labelled[tuple(slices)]
        ids, counts = np.unique(sub[sub > 0], return_counts=True)
        for instance_id, count in zip(ids.tolist(), counts.tolist()):
            covered_counts[instance_id] = count
    coverages = [
        float(covered_counts[i]) / float(total_counts[i])
        for i in range(1, n_instances + 1)
    ]
    result["instance_coverages"] = coverages
    result["n_instances_fully_covered"] = int(sum(1 for c in coverages if c >= 1.0))
    result["n_instances_fully_missed"] = int(sum(1 for c in coverages if c <= 0.0))
    result["total_lesion_voxels"] = int(total_counts[1:].sum())
    result["covered_lesion_voxels"] = int(covered_counts[1:].sum())
    return result


def _roi_source_mask(
    inputs: Inputs, case_id: str, candidate: str
) -> tuple[np.ndarray | None, Geometry | None, dict]:
    """取某候选来源的**允许前景**掩膜；返回 ``(mask, geometry, diagnostics)``。

    非法或非有限的标签取值绝不进入 ROI：不使用 ``arr > 0`` 之类的捷径，
    而是经标签契约检查后只取允许的非零取值。
    """
    diagnostics: dict[str, Any] = {
        "candidate": candidate,
        "path": "",
        "label_contract": None,
        "exists": False,
        "read_status": None,
        "geometry": None,
        "geometry_problems": [],
        "label_info": None,
        "status": None,
        "reason": "",
        "n_foreground_voxels": None,
    }
    if candidate == "wg_materialized":
        path = inputs.case_label(case_id, CASE_WG)
        contract = LABEL_CONTRACT_WG
    elif candidate == "yuan_pz_tz_union":
        path = inputs.case_label(case_id, CASE_ZONAL_TEMPLATE.format(source="yuan"))
        contract = LABEL_CONTRACT_ZONAL
    elif candidate == "hevi_pz_tz_union":
        path = inputs.case_label(case_id, CASE_ZONAL_TEMPLATE.format(source="hevi"))
        contract = LABEL_CONTRACT_ZONAL
    else:  # pragma: no cover - 参数由 choices 限制
        raise AuditError(f"未知 ROI 候选来源：{candidate}")
    diagnostics["path"] = str(path)
    diagnostics["label_contract"] = contract
    diagnostics["exists"] = path.is_file()
    if not diagnostics["exists"]:
        diagnostics["status"] = ROI_SOURCE_MISSING
        diagnostics["reason"] = "来源标签文件不存在"
        return None, None, diagnostics
    try:
        geometry = read_geometry(path)
    except Exception as exc:  # noqa: BLE001
        diagnostics["status"] = ROI_SOURCE_READ_ERROR
        diagnostics["read_status"] = "header_read_error"
        diagnostics["reason"] = f"来源头读取失败：{type(exc).__name__}: {exc}"
        return None, None, diagnostics
    diagnostics["geometry"] = geometry.as_dict()
    problems = geometry.validity_problems()
    diagnostics["geometry_problems"] = problems
    if problems:
        diagnostics["status"] = ROI_SOURCE_GEOMETRY_INVALID
        diagnostics["read_status"] = "ok"
        diagnostics["reason"] = f"来源几何非法：{problems}"
        return None, geometry, diagnostics
    try:
        array = read_array(path)
    except Exception as exc:  # noqa: BLE001
        diagnostics["status"] = ROI_SOURCE_READ_ERROR
        diagnostics["read_status"] = "voxel_read_error"
        diagnostics["reason"] = f"来源体素读取失败：{type(exc).__name__}: {exc}"
        return None, geometry, diagnostics
    diagnostics["read_status"] = "ok"
    label_info = inspect_label_array(array, contract)
    diagnostics["label_info"] = label_info
    if not label_is_usable(label_info):
        diagnostics["status"] = ROI_SOURCE_LABEL_NOT_USABLE
        diagnostics["reason"] = (
            f"来源标签未通过契约检查，不用于构造 ROI：{label_info['problems']}"
        )
        return None, geometry, diagnostics
    mask = label_foreground_mask(array, contract)
    diagnostics["n_foreground_voxels"] = int(np.count_nonzero(mask))
    if not diagnostics["n_foreground_voxels"]:
        diagnostics["status"] = ROI_SOURCE_EMPTY
        diagnostics["reason"] = "来源标签合法但为空（无允许前景）"
        return None, geometry, diagnostics
    diagnostics["status"] = ROI_SOURCE_OK
    return mask, geometry, diagnostics


def roi_case_audit(
    inputs: Inputs, case_id: str, candidate: str
) -> dict:
    """单病例单候选：以 **T2W 为几何基准**生成 ROI + 全部参考病灶实例的覆盖统计。

    覆盖率分母 = **全部**参考病灶实例。参考标签不可读 / 非法 / 与 T2W 不同网格时，
    分母是**未知**（``n_reference_instances = null``），
    **绝不**写成 0 —— 0 只对应「合法空病灶标签」这一真实阴性情形。

    腺体来源不可用但参考标签合法时，回退完整 **T2W** FOV 仍可计算真实覆盖；
    这与「参考标签不可用」是两种不同情形。
    """
    entry: dict[str, Any] = {
        "case_id": case_id,
        "candidate": candidate,
        "split": inputs.split_by_case.get(case_id, "unknown"),
        # 默认 False：只有拿到合法 T2W 几何后才置 True（不重采样、不臆测基准）
        "t2w_baseline_usable": False,
        "t2w_geometry_problems": [],
        "lesion_geometry_problems": [],
        "lesion_on_t2w_grid": None,
        "reference_state": None,
        "reference_evaluable": False,
        "reference_unavailable_reason": None,
        "roi_status": None,
        "fallback": False,
        "fallback_fov": None,
        "fallback_reason": None,
        "n_reference_instances": None,
        "instance_coverages": None,
        "instance_coverage_macro_mean": None,
        "n_instances_fully_covered": None,
        "n_instances_fully_missed": None,
        "total_lesion_voxels": None,
        "covered_lesion_voxels": None,
        "voxel_coverage": None,
    }

    def _unavailable(reason: str) -> dict:
        entry["reference_state"] = REFERENCE_UNAVAILABLE
        entry["reference_evaluable"] = False
        entry["reference_unavailable_reason"] = reason
        entry["reference_unavailable_note"] = (
            "参考病灶实例数**未知**：不得当作 0 个实例的阴性病例，"
            "也不得据此声明覆盖全部参考病灶"
        )
        return entry

    # ---- 1) T2W 几何基准 ----
    t2w_path = inputs.case_label(case_id, CASE_T2W)
    if not t2w_path.is_file():
        entry["roi_status"] = "t2w_missing"
        return _unavailable("T2W 影像不存在，无法建立几何基准")
    try:
        t2w_geometry = read_geometry(t2w_path)
    except Exception as exc:  # noqa: BLE001
        entry["roi_status"] = "t2w_read_error"
        return _unavailable(f"T2W 头读取失败：{type(exc).__name__}: {exc}")
    t2w_problems = t2w_geometry.validity_problems()
    entry["t2w_geometry_problems"] = t2w_problems
    if t2w_problems:
        entry["roi_status"] = "t2w_geometry_invalid"
        return _unavailable(f"T2W 几何非法：{t2w_problems}")
    entry["t2w_baseline_usable"] = True
    entry["t2w_geometry"] = t2w_geometry.as_dict()
    entry["t2w_shape_zyx"] = [int(s) for s in t2w_geometry.shape_zyx]
    entry["t2w_spacing_zyx"] = list(t2w_geometry.spacing_zyx)

    # ---- 2) 参考病灶（覆盖率分母） ----
    lesion_path = inputs.case_label(case_id, CASE_LESION)
    if not lesion_path.is_file():
        entry["roi_status"] = "lesion_missing"
        return _unavailable("物化 lesion 标签不存在，无法统计参考病灶覆盖")
    try:
        lesion_geometry = read_geometry(lesion_path)
    except Exception as exc:  # noqa: BLE001
        entry["roi_status"] = "lesion_read_error"
        return _unavailable(f"lesion 头读取失败：{type(exc).__name__}: {exc}")
    lesion_problems = lesion_geometry.validity_problems()
    entry["lesion_geometry"] = lesion_geometry.as_dict()
    entry["lesion_geometry_problems"] = lesion_problems
    if lesion_problems:
        entry["roi_status"] = "lesion_geometry_invalid"
        return _unavailable(f"lesion 几何非法：{lesion_problems}")
    if not lesion_geometry.same_grid_as(t2w_geometry):
        entry["roi_status"] = "lesion_grid_mismatch_vs_t2w"
        entry["lesion_on_t2w_grid"] = False
        entry["lesion_vs_t2w_differences"] = lesion_geometry.grid_differences(
            t2w_geometry
        )
        return _unavailable(
            "lesion 与 T2W 不同网格：无法按数组 bbox 计算覆盖；"
            "按约定不重采样，且**不得**把 lesion 的 FOV 称为完整 T2W FOV"
        )
    entry["lesion_on_t2w_grid"] = True
    try:
        lesion_array = read_array(lesion_path)
    except Exception as exc:  # noqa: BLE001
        entry["roi_status"] = "lesion_read_error"
        return _unavailable(f"lesion 体素读取失败：{type(exc).__name__}: {exc}")
    lesion_label = inspect_label_array(lesion_array, LABEL_CONTRACT_LESION)
    entry["lesion_label_info"] = lesion_label
    entry["lesion_unique_values"] = lesion_label["observed_values"]
    if not label_is_usable(lesion_label):
        entry["roi_status"] = "illegal_lesion_values"
        return _unavailable(
            f"lesion 未通过标签契约检查，不能作为覆盖率分母："
            f"{lesion_label['problems']}"
        )
    if tuple(lesion_array.shape) != tuple(t2w_geometry.shape_zyx):
        entry["roi_status"] = "lesion_shape_mismatch_vs_t2w"
        return _unavailable(
            f"lesion 数组形状 {list(lesion_array.shape)} 与 T2W "
            f"{list(t2w_geometry.shape_zyx)} 不一致"
        )
    lesion_mask = label_foreground_mask(lesion_array, LABEL_CONTRACT_LESION)
    entry["reference_state"] = (
        REFERENCE_EVALUABLE_NONEMPTY if bool(lesion_mask.any())
        else REFERENCE_EVALUABLE_EMPTY
    )
    entry["reference_evaluable"] = True
    entry["reference_instances_known"] = True

    # ---- 3) ROI 来源（以 T2W 为基准，不重采样） ----
    mask, source_geometry, diagnostics = _roi_source_mask(inputs, case_id, candidate)
    entry["source_diagnostics"] = diagnostics
    full_t2w_bbox = tuple((0, int(s)) for s in t2w_geometry.shape_zyx)

    def _fallback(reason: str) -> None:
        entry["roi_status"] = ROI_FALLBACK_FULL_FOV
        entry["fallback"] = True
        entry["fallback_fov"] = "full_t2w_fov"
        entry["fallback_reason"] = reason

    source_status = diagnostics.get("status")
    if source_status in ROI_SOURCE_UNAVAILABLE_STATUSES or mask is None:
        roi_bbox = full_t2w_bbox
        _fallback(
            f"ROI 来源不可用（{source_status}）：{diagnostics.get('reason', '')}；"
            "回退完整 T2W FOV 并继续计算真实病灶覆盖"
        )
    elif source_geometry is None:
        roi_bbox = full_t2w_bbox
        _fallback("ROI 来源几何缺失；回退完整 T2W FOV")
    elif not source_geometry.same_grid_as(t2w_geometry):
        roi_bbox = full_t2w_bbox
        _fallback(
            "ROI 来源与 T2W 网格不一致；按约定不重采样，回退完整 T2W FOV"
        )
        entry["source_vs_t2w_differences"] = source_geometry.grid_differences(
            t2w_geometry
        )
    elif tuple(mask.shape) != tuple(t2w_geometry.shape_zyx):
        roi_bbox = full_t2w_bbox
        _fallback(
            f"ROI 来源数组形状 {list(mask.shape)} 与 T2W "
            f"{list(t2w_geometry.shape_zyx)} 不一致；回退完整 T2W FOV"
        )
    else:
        component_bbox = largest_component_bbox(mask)
        if component_bbox is None:  # pragma: no cover - 空标签已在来源层拦下
            roi_bbox = full_t2w_bbox
            _fallback("ROI 来源无可用连通域；回退完整 T2W FOV")
        else:
            roi_bbox = expand_bbox_mm(
                component_bbox,
                inputs.margin_mm,
                t2w_geometry.spacing_zyx,
                t2w_geometry.shape_zyx,
            )
            entry["roi_status"] = ROI_FROM_LABEL
            entry["largest_component_bbox_zyx"] = [list(b) for b in component_bbox]

    entry["roi_bbox_zyx"] = [list(b) for b in roi_bbox]
    component_bbox = entry.get("largest_component_bbox_zyx")
    if component_bbox is not None:
        # 实际施加的余量 = 连通域相对 ROI 向外各扩了多少体素。
        # 若请求的余量被体积边界裁剪，这里的值会小于请求值，必须如实报告。
        entry["applied_margin_voxels_zyx"] = [
            (
                int(component_bbox[axis][0]) - int(roi_bbox[axis][0]),
                int(roi_bbox[axis][1]) - int(component_bbox[axis][1]),
            )
            for axis in range(len(roi_bbox))
        ]
    else:
        entry["applied_margin_voxels_zyx"] = None
    entry["roi_margin_mm"] = inputs.margin_mm
    entry["roi_spacing_zyx"] = list(t2w_geometry.spacing_zyx)

    # ---- 4) 覆盖率（分母 = 全部参考病灶实例，由实际掩膜计算） ----
    coverage = instance_coverages(lesion_mask, roi_bbox)
    entry["n_reference_instances"] = int(coverage["n_instances"])
    entry["n_instances_fully_covered"] = coverage["n_instances_fully_covered"]
    entry["n_instances_fully_missed"] = coverage["n_instances_fully_missed"]
    entry["total_lesion_voxels"] = coverage["total_lesion_voxels"]
    entry["covered_lesion_voxels"] = coverage["covered_lesion_voxels"]
    entry["voxel_coverage"] = _ratio(
        coverage["covered_lesion_voxels"], coverage["total_lesion_voxels"]
    )
    entry["instance_coverages"] = coverage["instance_coverages"]
    if coverage["instance_coverages"]:
        entry["instance_coverage_macro_mean"] = float(
            np.mean(coverage["instance_coverages"])
        )
    return entry


#: 「全部参考病灶」口径下，分母不完整时必须置 null 的字段
ROI_ALL_REFERENCE_FIELDS = (
    "n_reference_instances",
    "instance_coverage_macro_mean",
    "instance_coverage_median",
    "n_instances_fully_covered",
    "instance_fully_covered_ratio",
    "n_instances_fully_missed",
    "instance_fully_missed_ratio",
    "overall_voxel_coverage",
    "total_lesion_voxels",
    "covered_lesion_voxels",
)


def _roi_instance_stats(entries: Sequence[dict]) -> dict:
    """在给定病例子集上汇总**实际测量**的病灶实例统计（不做任何外推）。"""
    coverages: list[float] = []
    total_voxels = 0
    covered_voxels = 0
    n_instances = 0
    n_full = 0
    n_missed = 0
    for entry in entries:
        coverages.extend(entry.get("instance_coverages") or [])
        total_voxels += int(entry.get("total_lesion_voxels") or 0)
        covered_voxels += int(entry.get("covered_lesion_voxels") or 0)
        n_instances += int(entry.get("n_reference_instances") or 0)
        n_full += int(entry.get("n_instances_fully_covered") or 0)
        n_missed += int(entry.get("n_instances_fully_missed") or 0)
    return {
        "n_cases": len(entries),
        "n_cases_with_lesion_instances": sum(
            1 for e in entries if e.get("n_reference_instances")
        ),
        "n_reference_instances": n_instances,
        "instance_coverage_macro_mean": (
            float(np.mean(coverages)) if coverages else None
        ),
        "instance_coverage_median": (
            float(np.median(coverages)) if coverages else None
        ),
        "n_instances_fully_covered": n_full,
        "instance_fully_covered_ratio": _ratio(n_full, n_instances),
        "n_instances_fully_missed": n_missed,
        "instance_fully_missed_ratio": _ratio(n_missed, n_instances),
        "overall_voxel_coverage": _ratio(covered_voxels, total_voxels),
        "total_lesion_voxels": total_voxels,
        "covered_lesion_voxels": covered_voxels,
    }


def _roi_summary(entries: list[dict]) -> dict:
    """按候选来源汇总。

    **分母完整性**：参考病灶标签不可读/非法/与 T2W 不同网格时，该病例的实例数未知；
    只要存在这样的病例，``denominator_complete`` 为假，
    所有「全部参考病灶」口径的比例字段置 ``null``（不是 0、不是 1），
    并另行给出只覆盖可评估子集的统计（字段名显式标注其范围）。
    """
    evaluable = [e for e in entries if e.get("reference_evaluable")]
    unavailable = [e for e in entries if not e.get("reference_evaluable")]
    denominator_complete = not unavailable
    subset = _roi_instance_stats(evaluable)
    n_observed = int(subset["n_reference_instances"])

    summary: dict[str, Any] = {
        "n_cases": len(entries),
        "n_cases_reference_evaluable": len(evaluable),
        "n_cases_reference_unavailable": len(unavailable),
        "denominator_complete": bool(denominator_complete),
        "denominator_note": (
            "全部病例的参考病灶实例数均已实测，可报告「全部参考病灶」口径"
            if denominator_complete
            else (
                f"{len(unavailable)} 例的参考病灶实例数**未知**（不可读/非法/与 T2W 不同网格）；"
                "「全部参考病灶」口径的比例已置 null，"
                "只能报告 evaluable_subset_only 中显式标注范围的可评估子集统计"
            )
        ),
        "n_reference_instances_observed": n_observed,
        "reference_unavailable_cases": sorted(
            e["case_id"] for e in unavailable
        ),
        "reference_unavailable_reason_counts": _counter(
            e.get("roi_status") for e in unavailable
        ),
        "t2w_baseline_unusable_cases": sorted(
            e["case_id"] for e in entries if e.get("t2w_baseline_usable") is not True
        ),
        "reference_state_counts": _counter(e.get("reference_state") for e in entries),
        "instance_counts_measured_from_reference_masks": True,
        "instance_count_note": (
            "实例数一律由实际参考掩膜的 3D 6-邻域连通域计算，"
            "不存在任何预先固定或外部给定的数量"
        ),
    }

    if denominator_complete:
        summary.update({key: subset[key] for key in ROI_ALL_REFERENCE_FIELDS})
        summary["reference_scope"] = "all_cases"
    else:
        # 「全部参考病灶」口径：分母不完整 → 全部置 null
        summary.update({key: None for key in ROI_ALL_REFERENCE_FIELDS})
        summary["reference_scope"] = "all_cases_denominator_incomplete"

    summary["evaluable_subset_only"] = {
        "scope": (
            "仅参考病灶可评估的子集；**不代表全部参考病灶**，"
            "不得用于声明覆盖全部病例"
        ),
        "denominator": "sum_over_evaluable_cases_only",
        **subset,
    }

    summary.update(
        {
            "n_cases_fallback": sum(1 for e in entries if e.get("fallback")),
            "fallback_cases": sorted(e["case_id"] for e in entries if e.get("fallback")),
            "fallback_note": (
                "回退完整 T2W FOV 的病例仍会计算真实病灶覆盖（前提是参考标签合法），"
                "因此它们计入分母；这与参考标签不可用是两种情形"
            ),
            "n_instances_in_fallback_cases": sum(
                int(e.get("n_reference_instances") or 0)
                for e in evaluable
                if e.get("fallback")
            ),
            "n_instances_fully_missed_in_fallback_cases": sum(
                int(e.get("n_instances_fully_missed") or 0)
                for e in evaluable
                if e.get("fallback")
            ),
            "roi_status_counts": _counter(e.get("roi_status") for e in entries),
            "roi_fallback_trigger_counts": _counter(
                (e.get("source_diagnostics") or {}).get("status")
                for e in entries
                if e.get("fallback")
            ),
        }
    )
    return summary


def stage_roi_coverage(inputs: Inputs, *, progress: bool) -> dict:
    per_candidate: dict[str, list[dict]] = {c: [] for c in ROI_CANDIDATES}
    iterator = tqdm(
        inputs.case_ids, desc="roi-coverage", unit="case", disable=not progress
    )
    for case_id in iterator:
        for candidate in ROI_CANDIDATES:
            per_candidate[candidate].append(roi_case_audit(inputs, case_id, candidate))

    summary: dict[str, Any] = {}
    for candidate, entries in per_candidate.items():
        by_split: dict[str, dict] = {}
        for split in sorted({e.get("split", "unknown") for e in entries}):
            by_split[split] = _roi_summary(
                [e for e in entries if e.get("split", "unknown") == split]
            )
        summary[candidate] = {
            "overall": _roi_summary(entries),
            "per_split": by_split,
            # 每个划分**各自**判断分母完整性，避免用 overall 掩盖某个划分的不完整
            "per_split_denominator_complete": {
                split: bool(stats["denominator_complete"])
                for split, stats in by_split.items()
            },
            "n_cases_lesion_missing": sum(
                1 for e in entries if e.get("roi_status") == "lesion_missing"
            ),
            "n_cases_reference_unavailable": sum(
                1 for e in entries if not e.get("reference_evaluable")
            ),
        }
    return {
        "roi_definition": {
            "component": "3D 最大连通域（6-邻域，scipy.ndimage structure order 1）",
            "bbox": "该连通域的体素包围盒",
            "margin_mm": inputs.margin_mm,
            "margin_to_voxels": (
                "ceil(margin_mm / spacing)，逐轴按 numpy 轴序 (z,y,x) 换算，"
                "向外取整后裁剪到体积范围"
            ),
            "geometry_baseline": (
                "统一以 **T2W 几何**为基准：lesion 必须与 T2W 同网格才按数组 bbox 计算覆盖；"
                "lesion 与 T2W 不一致时不做比较，也**不把 lesion 的 FOV 称为完整 T2W FOV**"
            ),
            "fallback": (
                "来源标签缺失/不可读/几何非法/取值非法/为空/与 T2W 网格或形状不一致时，"
                "回退完整 **T2W** FOV（不重采样），并逐例记录 fallback 触发原因"
            ),
            "no_writeback": "ROI 只在内存中构造，不写回任何影像或裁剪数据",
        },
        "reference_instances": {
            "definition": (
                "物化 lesion.nii.gz 的 3D 6-邻域连通域；"
                "与 scripts/evaluate_segmentation.py 的 CONNECTIVITY 一致"
            ),
            "denominator": (
                "**全部**参考病灶实例；不限于预测成功、有交集或已检出的病例。"
                "实例数一律由实际参考掩膜计算，不存在预先给定的固定数量"
            ),
            "unavailable_rule": (
                "参考标签不可读/非法/与 T2W 不同网格时，该病例实例数**未知**"
                "（null），不得当作 0 个实例的阴性病例；"
                "此时「全部参考病灶」口径的比例字段一律为 null，"
                "只在 evaluable_subset_only 中报告显式标注范围的可评估子集"
            ),
        },
        "scope_limit": (
            "本 stage 只评估**现有标签**生成 ROI 的覆盖，"
            "不能称为阶段一预测模型的效果上界；"
            "后续 ROI 参数选择只应使用外层 train 侧，不得依据 validation 结果反复调参"
        ),
        "per_candidate_summary": summary,
        "per_case": per_candidate,
    }


# --------------------------------------------------------------------------- 汇总工具


def _distribution(values: Iterable[float]) -> dict:
    data = [float(v) for v in values if v is not None]
    if not data:
        return {"n": 0, "min": None, "median": None, "mean": None, "max": None}
    arr = np.asarray(data, dtype=np.float64)
    return {
        "n": int(arr.size),
        "min": float(arr.min()),
        "median": float(np.median(arr)),
        "mean": float(arr.mean()),
        "max": float(arr.max()),
    }


def _counter(values: Iterable[Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        key = "null" if value is None else str(value)
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


# --------------------------------------------------------------------------- 状态判定


def _inputs_problems(inputs_stage: dict) -> list[str]:
    """数据完整性（``inputs`` stage，恒被执行）的问题清单。"""
    problems: list[str] = []
    if inputs_stage.get("n_cases_unexpected_missing"):
        problems.append(
            f"{inputs_stage['n_cases_unexpected_missing']} 例存在**意外**缺失的必需文件"
            "（t2w/lesion）"
        )
    if inputs_stage.get("n_cases_read_error"):
        problems.append(
            f"{inputs_stage['n_cases_read_error']} 例存在必需文件头读取失败"
        )
    if inputs_stage.get("n_cases_geometry_read_error"):
        problems.append(
            f"{inputs_stage['n_cases_geometry_read_error']} 例存在标签/来源文件头读取失败"
        )
    if inputs_stage.get("n_cases_geometry_problem"):
        problems.append(
            f"{inputs_stage['n_cases_geometry_problem']} 例存在非法几何"
            "（维度/spacing/origin/direction）"
        )
    if inputs_stage.get("n_cases_t2w_geometry_unusable"):
        problems.append(
            f"{inputs_stage['n_cases_t2w_geometry_unusable']} 例的 T2W 几何不可用"
            "（无法建立几何基准）"
        )
    if inputs_stage.get("n_cases_lesion_not_on_t2w_grid"):
        problems.append(
            f"{inputs_stage['n_cases_lesion_not_on_t2w_grid']} 例的 lesion 与 T2W 不同网格"
        )
    if inputs_stage.get("n_cases_wg_status_unrecognized"):
        problems.append(
            f"{inputs_stage['n_cases_wg_status_unrecognized']} 例的 wg_status 未被"
            f"认可为已知缺失（允许清单 {inputs_stage.get('known_missing_wg_status_allowlist')}），"
            "需人工确认"
        )
    if inputs_stage.get("n_cases_patient_relation_bad"):
        problems.append(
            f"{inputs_stage['n_cases_patient_relation_bad']} 例的 case_id 与 patient_id "
            "关系不自洽"
        )
    if inputs_stage.get("manifest_duplicate_case_ids"):
        problems.append(
            f"manifest 存在重复 case_id：{inputs_stage['manifest_duplicate_case_ids'][:5]}"
        )
    if inputs_stage.get("split_duplicate_case_ids"):
        problems.append(
            f"split CSV 存在重复 case_id：{inputs_stage['split_duplicate_case_ids'][:5]}"
        )
    if inputs_stage.get("n_patients_spanning_splits"):
        problems.append(
            f"{inputs_stage['n_patients_spanning_splits']} 名患者的 study 跨外层 "
            "train/validation"
        )
    if inputs_stage.get("case_ids_in_manifest_absent_from_split"):
        problems.append(
            f"{len(inputs_stage['case_ids_in_manifest_absent_from_split'])} 例在 split "
            "CSV 中缺失"
        )
    return problems


#: zonal 来源不可用/不可比时进入问题汇总的 skip code
ZONAL_SKIP_CODES_AS_PROBLEMS = (
    "zonal_file_missing",
    "zonal_header_read_error",
    "zonal_voxel_read_error",
    "zonal_label_illegal",
    "zonal_geometry_invalid",
    "shape_mismatch",
)


def _zonal_problems(zonal_stage: dict) -> list[str]:
    problems: list[str] = []
    for source, summary in zonal_stage["per_source_summary"].items():
        if summary.get("n_label_not_usable"):
            problems.append(
                f"zonal/{source} 有 {summary['n_label_not_usable']} 例标签未通过契约检查"
                "（非法取值/非整数/非有限/维度错误）"
            )
        if summary.get("n_geometry_invalid"):
            problems.append(
                f"zonal/{source} 有 {summary['n_geometry_invalid']} 例几何非法"
            )
        if summary.get("n_abnormal_geometry"):
            problems.append(
                f"zonal/{source} 有 {summary['n_abnormal_geometry']} 例几何/标签异常"
            )
        skip_counts = summary.get("skip_code_counts") or {}
        n_blocking = sum(skip_counts.get(code, 0) for code in ZONAL_SKIP_CODES_AS_PROBLEMS)
        if n_blocking:
            detail = {
                code: skip_counts[code]
                for code in ZONAL_SKIP_CODES_AS_PROBLEMS
                if skip_counts.get(code)
            }
            problems.append(
                f"zonal/{source} 有 {n_blocking} 例分区读取/标签/形状问题导致无法比较："
                f"{detail}"
            )
        if skip_counts.get("grid_mismatch"):
            problems.append(
                f"zonal/{source} 有 {skip_counts['grid_mismatch']} 例与物化 WG 不同网格"
                "（按约定不重采样，未做比较）"
            )
    return problems


def _prostate158_problems(p158: dict) -> list[str]:
    problems: list[str] = []
    if p158.get("n_read_error"):
        problems.append(f"Prostate158 有 {p158['n_read_error']} 个文件读取失败")
    if p158.get("n_missing"):
        problems.append(
            f"Prostate158 有 {p158['n_missing']} 个已声明解剖标签文件缺失"
        )
    if p158.get("n_geometry_invalid"):
        problems.append(
            f"Prostate158 有 {p158['n_geometry_invalid']} 个文件几何非法"
        )
    if p158.get("n_non_finite"):
        problems.append(
            f"Prostate158 有 {p158['n_non_finite']} 个文件含非有限取值（NaN/Inf）"
        )
    if p158.get("n_non_integer"):
        problems.append(
            f"Prostate158 有 {p158['n_non_integer']} 个文件含非整数值"
        )
    return problems


#: ROI 参考标签不可用 → 进入问题汇总（分母不完整是数据问题，不是执行问题）
ROI_REFERENCE_UNAVAILABLE_STATUSES = (
    "lesion_missing",
    "lesion_read_error",
    "lesion_geometry_invalid",
    "lesion_grid_mismatch_vs_t2w",
    "lesion_shape_mismatch_vs_t2w",
    "illegal_lesion_values",
    "t2w_missing",
    "t2w_read_error",
    "t2w_geometry_invalid",
)


def _roi_problems(candidate: str, stats: dict) -> list[str]:
    problems: list[str] = []
    counts = stats.get("roi_status_counts") or {}
    for status in ROI_REFERENCE_UNAVAILABLE_STATUSES:
        if counts.get(status):
            problems.append(
                f"roi[{candidate}] 有 {counts[status]} 例 {status}"
                "（参考病灶实例数未知，覆盖率分母不完整）"
            )
    if not stats.get("denominator_complete"):
        problems.append(
            f"roi[{candidate}] 覆盖率分母**不完整**："
            f"{stats.get('n_cases_reference_unavailable')} / {stats.get('n_cases')} 例的"
            "参考病灶实例数未知，「全部参考病灶」口径已置 null"
        )
    return problems


def _wg_source_status(wg_stage: dict | None) -> tuple[str, list[str], Sequence[str]]:
    """来源审计状态：``not_evaluated`` / ``resolved`` / ``unresolved``。

    **未请求来源审计时返回 not_evaluated，绝不能把「没检查」当成 resolved 的证据。**
    """
    if wg_stage is None:
        return "not_evaluated", [], ()
    source_problems: list[str] = []
    counts = wg_stage["counts"]
    detail = {
        WG_BOTH: "两套候选来源同时逐体素一致 → 来源歧义（both_match）",
        WG_NEITHER: "与两套候选来源都不一致（neither_match）",
        WG_MATCH_UNRESOLVED: (
            "与某候选内容一致，但竞争候选未完成比较 → 唯一性未解决"
            "（match_with_unresolved_uniqueness）"
        ),
        WG_NO_COMPARABLE_CANDIDATE: (
            "没有任何候选完成比较（no_comparable_candidate）"
        ),
        WG_GRID_MISMATCH: "无法在同一网格上比较（grid_mismatch），按约定不自动重采样",
        WG_MISSING: "物化 WG 缺失（missing）",
        WG_READ_ERROR: "物化 WG 或来源读取/几何失败（read_error）",
    }
    for outcome in WG_OUTCOMES:
        if outcome in WG_RESOLVED_OUTCOMES:
            continue
        if counts.get(outcome):
            source_problems.append(f"{counts[outcome]} 例：{detail[outcome]}")
    status = "resolved" if not source_problems else "unresolved"
    return status, source_problems, WG_RESOLVED_OUTCOMES


def evaluate_status(report: dict, stages: Sequence[str]) -> dict:
    """三个独立状态 + 显式的「未评估」清单；不使用含糊的 ``all_pass``。"""
    stages = tuple(stages)
    run_stages = report.get("stages", {})
    inputs_stage = run_stages.get("inputs", {})
    scope = report.get("inputs", {}).get("scope", "full_dataset")

    # ---- 数据完整性：inputs 恒被执行，因此这些检查永远可见 ----
    problems = _inputs_problems(inputs_stage)
    if "wg" in stages:
        wg_stage = run_stages.get("wg")
        counts = (wg_stage or {}).get("counts", {})
        if counts.get(WG_MISSING) or counts.get(WG_READ_ERROR):
            problems.append(
                f"wg stage：物化 WG 缺失 {counts.get(WG_MISSING, 0)} 例、"
                f"读取/几何失败 {counts.get(WG_READ_ERROR, 0)} 例"
            )
    if "zonal" in stages:
        problems.extend(_zonal_problems(run_stages["zonal"]))
    if "prostate158" in stages:
        problems.extend(_prostate158_problems(run_stages["prostate158"]))
    if "roi" in stages:
        for candidate, summary in run_stages["roi"]["per_candidate_summary"].items():
            problems.extend(_roi_problems(candidate, summary["overall"]))

    # ---- 来源审计状态 ----
    source_problems: list[str] = []
    if "wg" in stages:
        source_audit_status, source_problems, _ = _wg_source_status(run_stages.get("wg"))
    else:
        source_audit_status = "not_evaluated"

    not_evaluated: list[str] = []
    for stage in STAGE_CHOICES:
        if stage != "inputs" and stage not in stages:
            not_evaluated.append(stage)
    if source_audit_status == "not_evaluated":
        not_evaluated.append("source_resolution")

    execution_complete = bool(report.get("execution", {}).get("completed"))
    data_valid: bool | None = not problems
    source_resolved: bool | None = {
        "resolved": True,
        "unresolved": False,
        "not_evaluated": None,
    }[source_audit_status]

    # ---- 子集模式：不得对全数据做任何有效性声明 ----
    subset_note = None
    if scope == "subset":
        subset_note = (
            f"本次使用 --max-cases，只审计 {report.get('inputs', {}).get('n_cases_audited')}"
            f" / {report.get('inputs', {}).get('n_cases_from_manifest')} 例，"
            "不得据此对全数据做 data_valid / source_resolved 声明"
        )
        data_valid = None
        source_resolved = None
        execution_complete = False

    return {
        "scope": scope,
        "execution_complete": execution_complete,
        "data_valid": data_valid,
        "source_resolved": source_resolved,
        "source_audit_status": source_audit_status,
        "stages_run": list(stages),
        "stages_not_evaluated": not_evaluated,
        "not_evaluated_note": (
            "未请求的 stage 明确标为未评估；「没检查」**不构成** "
            "source_resolved / data_valid 为真的证据"
        ),
        "problems": problems,
        "source_problems": source_problems,
        "subset_note": subset_note,
        "note": (
            "三个状态互相独立：来源不明或算法间不一致属于**观测结果**，"
            "不等于执行失败（execution_complete 仍可为真）；"
            "本报告不提供合并的 all_pass 字段"
        ),
    }


# --------------------------------------------------------------------------- 输出


#: 明确禁止作为报告输出的后缀（影像 / checkpoint / 模型权重等）
FORBIDDEN_OUTPUT_SUFFIXES = (
    ".nii.gz",
    ".nii",
    ".npz",
    ".npy",
    ".pth",
    ".pt",
    ".b2nd",
    ".pkl",
    ".bin",
    ".csv",
)


def assert_output_allowed(path: Path, *, input_paths: Sequence[Path] = ()) -> None:
    """报告只能写到**全新**的 JSON 路径。

    拒绝：受保护目录（原始数据 / data / workdir / third_party）、
    影像与 checkpoint 类后缀、以及任何与输入文件重合的路径。
    不做覆盖 —— 没有 ``--overwrite``。
    """
    resolved = path.resolve()
    for root in FORBIDDEN_OUTPUT_ROOTS:
        try:
            resolved.relative_to(root.resolve())
        except ValueError:
            continue
        raise AuditError(
            f"拒绝写入受保护目录：{resolved} 位于 {root} 之下。"
            "报告只能写到项目 outputs/ 等非原始数据位置。"
        )
    lowered = resolved.name.lower()
    if lowered.endswith(".tmp") or ".tmp" in lowered:
        raise AuditError(f"报告路径不得含 .tmp 片段：{resolved}")
    for suffix in FORBIDDEN_OUTPUT_SUFFIXES:
        if lowered.endswith(suffix):
            raise AuditError(
                f"报告必须是新 JSON：{resolved} 以 {suffix} 结尾，"
                "不得写到影像 / checkpoint / 逗号分隔表的位置。"
            )
    if not lowered.endswith(".json"):
        raise AuditError(f"报告必须是 .json 文件：{resolved}")
    for input_path in input_paths:
        if resolved == Path(input_path).resolve():
            raise AuditError(
                f"报告路径与输入文件重合：{resolved}。不得覆盖或改写任何输入文件。"
            )
    if path.exists():
        raise AuditError(
            f"报告已存在，拒绝覆盖：{resolved}\n"
            "本工具不提供 --overwrite：请换一个全新的 --output 路径。"
        )


def publish_report(report: dict, output_path: Path) -> None:
    """原子写：先写同目录临时文件再 ``os.replace``，避免半份报告。

    只允许写**全新**的 JSON 路径（无覆盖开关，见 ``assert_output_allowed``）。
    """
    assert_output_allowed(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        _json_safe(report), ensure_ascii=False, indent=2, sort_keys=False
    )
    tmp_path = output_path.with_name(output_path.name + f".tmp{os.getpid()}")
    try:
        tmp_path.write_text(payload + "\n", encoding="utf-8")
        os.replace(tmp_path, output_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


# --------------------------------------------------------------------------- 编排


def run_audit(inputs: Inputs, *, progress: bool = True) -> dict:
    stages = inputs.stages
    report: dict[str, Any] = {
        "schema": "prostate_anatomy_labels_audit",
        "schema_version": "1.0",
        "generated_at_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "inputs": inputs.as_dict(),
        "evidence_classes": {
            "original": "PI-CAI labels 目录下的官方原始标签（只读）",
            "materialized": "data/processed/picai/cases/<case_id>/ 下的物化标签",
            "existing_report": (
                "materialization_report.csv / picai_manifest.csv 已记录的字段"
            ),
            "note": (
                "三类证据并列报告，可互相矛盾；本工具不取其一当真值，"
                "也不把自动解剖标签写成人工真值"
            ),
        },
        "scope": "subset" if inputs.is_subset else "full_dataset",
        "stages": {},
        "inputs_modified": False,
    }
    started = time.time()
    try:
        if "inputs" in stages:
            report["stages"]["inputs"] = stage_inputs(inputs, progress=progress)
        if "wg" in stages:
            report["stages"]["wg"] = stage_wg_content_match(inputs, progress=progress)
        if "zonal" in stages:
            report["stages"]["zonal"] = stage_zonal_relations(inputs, progress=progress)
        if "prostate158" in stages:
            report["stages"]["prostate158"] = stage_prostate158(
                inputs, progress=progress
            )
        if "roi" in stages:
            report["stages"]["roi"] = stage_roi_coverage(inputs, progress=progress)
        report["execution"] = {"completed": True, "stages_run": list(stages)}
    finally:
        report["elapsed_seconds"] = round(time.time() - started, 3)
    report["status"] = evaluate_status(report, stages)
    return report


# --------------------------------------------------------------------------- CLI


# Candidate supervision contract: bits describe annotator membership, not anatomy classes.
ANATOMY_REGIONS = {"background": 0, "WG": [1, 3, 5, 7],
                   "PZ": [2, 3, 6, 7], "TZ": [4, 5, 6, 7]}


def encode_anatomy_membership(wg, pz, tz):
    arrays = [np.asarray(a) for a in (wg, pz, tz)]
    if len({a.shape for a in arrays}) != 1:
        raise AuditError("membership masks must have identical shapes")
    if any(not np.isin(a, [0, 1]).all() for a in arrays):
        raise AuditError("membership masks must be binary; missing is not empty")
    return sum(a.astype(np.uint8) * bit for a, bit in zip(arrays, (1, 2, 4)))


def decode_anatomy_membership(encoded):
    encoded = np.asarray(encoded)
    if not np.isin(encoded, range(8)).all():
        raise AuditError("invalid membership code")
    return np.stack([(encoded.astype(np.uint8) & bit) != 0 for bit in (1, 2, 4)])


def corner_displacements(reference, other):
    """Same reference voxel-centre indices in two LPS physical grids, millimetres."""
    from itertools import product
    indices = np.asarray(list(product(*[(0, n - 1) for n in reference.size_xyz])))
    def coordinates(g):
        return (indices * np.asarray(g.spacing_xyz)) @ np.asarray(g.direction).reshape(3, 3).T + g.origin_xyz
    a, b = coordinates(reference), coordinates(other)
    norms = np.linalg.norm(b - a, axis=1)
    return {"indices_xyz": indices.tolist(), "reference_lps_mm": a.tolist(),
            "other_lps_mm": b.tolist(), "delta_lps_mm": (b-a).tolist(),
            "distance_mm": norms.tolist(), "max_mm": float(norms.max()),
            "rms_mm": float(np.sqrt(np.mean(norms**2))),
            "index_domain": "reference voxel centres; other indices may be outside its extent"}


def select_followup_entries(baseline):
    try:
        entries = baseline["stages"]["wg"]["per_case"]
        selected = [e for e in entries if e["unique_source_unresolved"] is True]
        ids = [e["case_id"] for e in selected]
        if len(ids) != len(set(ids)) or any(
            not isinstance(i, str) or not i or Path(i).name != i for i in ids
        ):
            raise ValueError("invalid or duplicate case IDs")
        return selected
    except (KeyError, TypeError, ValueError) as exc:
        raise AuditError(f"invalid baseline WG entries: {exc}") from exc


def followup_image(path):
    import SimpleITK as sitk
    import nibabel as nib
    geometry = read_geometry(path)
    image = nib.load(str(path))  # header only; voxel reads use SimpleITK below
    qform, qcode = image.get_qform(coded=True)
    sform, scode = image.get_sform(coded=True)
    return geometry, {"geometry": geometry.as_dict(),
        "geometry_problems": geometry.validity_problems(),
        "nifti_ras": {"qform": None if qform is None else qform.tolist(),
                      "qform_code": int(qcode),
                      "sform": None if sform is None else sform.tolist(),
                      "sform_code": int(scode)},
        "readers": {"SimpleITK": sitk.Version_VersionString(), "nibabel": nib.__version__}}


def run_targeted_followup(args):
    import time
    started = time.monotonic()
    if args.max_cases is not None:
        raise AuditError("targeted followup forbids --max-cases; scope is pinned to baseline")
    baseline_path, output = Path(args.targeted_followup), Path(args.output)
    assert_output_allowed(output, input_paths=[baseline_path])
    raw = baseline_path.read_bytes()
    import hashlib
    baseline = json.loads(raw)
    selected = select_followup_entries(baseline)
    known = [e for e in selected if e["outcome"] == WG_MISSING and
             e.get("known_missing_wg", {}).get("category") == "recognized_known_missing"]
    targets = [e for e in selected if e not in known]
    inputs_by_id = {e["case_id"]: e for e in baseline["stages"]["inputs"]["per_case"]}
    # Paths are pinned to the baseline, never inferred by scanning directories.
    paths_by_id = {}
    for entry in targets:
        paths_by_id[entry["case_id"]] = {
            "t2w": inputs_by_id[entry["case_id"]]["files"][CASE_T2W]["path"],
            "materialized": entry["materialized"]["path"],
            **{k: v["path"] for k, v in entry["candidates"].items()}}
    assert_output_allowed(output, input_paths=[baseline_path] +
                          [Path(p) for paths in paths_by_id.values() for p in paths.values()])
    results = []
    for entry in tqdm(targets, desc="targeted-wg-followup", unit="case", disable=args.no_progress):
        result = {"case_id": entry["case_id"], "baseline_entry": entry,
                  "images": {}, "comparisons": {}, "errors": [], "source_resolved": False}
        geometries, arrays = {}, {}
        paths = paths_by_id[entry["case_id"]]
        for name, path in paths.items():
            try:
                geometries[name], result["images"][name] = followup_image(Path(path))
                if name != "t2w":
                    arrays[name] = read_array(Path(path))
                    result["images"][name]["label_info"] = inspect_label_array(arrays[name], "wg")
            except Exception as exc:
                result["errors"].append(f"{name}: {type(exc).__name__}: {exc}")
        for ref_name, other_name in [("t2w", n) for n in paths if n != "t2w"] + [
                ("materialized", n) for n in entry["candidates"]]:
            if ref_name not in geometries or other_name not in geometries:
                continue
            ref, other = geometries[ref_name], geometries[other_name]
            comparison = compare_geometry(ref, other)
            if ref.is_valid and other.is_valid:
                comparison["corner_displacements"] = corner_displacements(ref, other)
            if ref_name in arrays and other_name in arrays:
                a, b = arrays[ref_name], arrays[other_name]
                comparison["index_space"] = {"same_shape": a.shape == b.shape,
                    "equal": bool(np.array_equal(a, b)) if a.shape == b.shape else None,
                    "unequal_voxels": int(np.count_nonzero(a != b)) if a.shape == b.shape else None}
                usable = all(label_is_usable(result["images"][n]["label_info"])
                             for n in (ref_name, other_name))
                comparison["original_physical_voxel_equal"] = bool(np.array_equal(a, b)) if comparison["same_grid"] and usable else None
                if args.derived_resampling and ref.is_valid and other.is_valid and usable:
                    import SimpleITK as sitk
                    target = sitk.ReadImage(paths[ref_name])
                    source = sitk.ReadImage(paths[other_name])
                    derived = sitk.GetArrayFromImage(sitk.Resample(source, target, sitk.Transform(), sitk.sitkNearestNeighbor, 0, source.GetPixelID()))
                    comparison["derived_comparison"] = {"kind": "physical_nearest_neighbor_in_memory",
                        "default_outside_value": 0, "equal": bool(np.array_equal(a, derived)),
                        "unequal_voxels": int(np.count_nonzero(a != derived)),
                        "dice": _dice(a == 1, derived == 1), "original_voxel_identity": False}
            result["comparisons"][f"{ref_name}_vs_{other_name}"] = comparison
        results.append(result)
        tqdm.write(f"{entry['case_id']}: comparisons={len(result['comparisons'])}, errors={result['errors']}; source remains unconfirmed")
    report = {"schema": "prostate_anatomy_targeted_followup", "scope": "targeted_followup",
        "baseline": {"path": str(baseline_path), "sha256": hashlib.sha256(raw).hexdigest()},
        "selection": "only baseline unresolved WG entries; known missing recorded without image reads",
        "known_missing": known, "selected_case_ids": [e["case_id"] for e in selected],
        "derived_resampling_enabled": args.derived_resampling,
        "geometry_atol": GEOMETRY_ATOL, "geometry_rtol": GEOMETRY_RTOL,
        "note": "Index equality and derived equality do not establish physical grid identity or historical generation process.",
        "per_case": results, "summary": {"success": sum(not r["errors"] for r in results),
            "failed": sum(bool(r["errors"]) for r in results), "skipped_known_missing": len(known)},
        "elapsed_seconds": time.monotonic()-started, "output_path": str(output)}
    publish_report(report, output)
    print(json.dumps({**report["summary"], "elapsed_seconds": report["elapsed_seconds"], "output_path": str(output)}))
    return EXIT_UNRESOLVED if any(r["errors"] for r in results) else EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audit_prostate_anatomy_labels.py",
        description=(
            "前列腺解剖标签只读审计：物化 WG 的来源内容对应、PZ/TZ 与 WG 的并集关系、"
            "Prostate158 解剖整数值与几何、现有标签生成腺体代理 ROI 的覆盖。"
            "只读、不重采样、不写回影像、fail-closed。"
        ),
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--split-cases", default=str(DEFAULT_SPLIT_CASES))
    parser.add_argument(
        "--materialization-report", default=str(DEFAULT_MATERIALIZATION_REPORT)
    )
    parser.add_argument("--cases-root", default=str(DEFAULT_CASES_ROOT))
    parser.add_argument("--picai-labels-root", default=str(DEFAULT_PICAI_LABELS_ROOT))
    parser.add_argument("--prostate158-root", default=str(DEFAULT_PROSTATE158_ROOT))
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=STAGE_CHOICES,
        default=list(STAGE_CHOICES),
        help=(
            "要运行的 stage（默认全部）。注意：inputs 是**强制**审计项，"
            "即使只请求 roi 也会执行，数据完整性检查不会被跳过"
        ),
    )
    parser.add_argument(
        "--roi-margin-mm",
        type=float,
        default=5.0,
        help="腺体代理 ROI 的固定物理余量（mm），按 ceil 向外取整到体素",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help=(
            "只审计前 N 例（烟雾测试用）；此时 scope=subset，"
            "报告不含 data_valid / source_resolved 的全数据声明，且 execution_complete 为 false"
        ),
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=(
            "**全新** JSON 报告路径；工具不提供 --overwrite，"
            "已存在或与输入文件重合时直接拒绝"
        ),
    )
    parser.add_argument("--targeted-followup", metavar="BASELINE_JSON", help="Only unresolved WG entries from this baseline; bypass full audit")
    parser.add_argument("--derived-resampling", action="store_true", help="Explicit opt-in physical nearest-neighbour in-memory derived comparison")
    parser.add_argument("--no-progress", action="store_true")
    return parser


def _print_summary(report: dict, output_path: Path, *, written: bool) -> None:
    status = report.get("status", {})
    stages = report.get("stages", {})
    print("\n=== 前列腺解剖标签审计 ===")
    print(f"  scope              : {status.get('scope')}")
    print(f"  execution_complete : {status.get('execution_complete')}")
    print(f"  data_valid         : {status.get('data_valid')}")
    print(f"  source_resolved    : {status.get('source_resolved')}")
    print(f"  source_audit_status: {status.get('source_audit_status')}")
    if status.get("stages_not_evaluated"):
        print(f"  未评估             : {status['stages_not_evaluated']}")
    if status.get("subset_note"):
        print(f"  子集说明           : {status['subset_note']}")
    if "inputs" in stages:
        s = stages["inputs"]
        print(
            f"  病例              : {s['n_cases']}"
            f"（t2w {s['n_cases_t2w_present']} / lesion {s['n_cases_lesion_present']}"
            f" / wg {s['n_cases_wg_present']}）"
        )
        print(
            f"  缺失与失败         : 意外缺失 {s['n_cases_unexpected_missing']}"
            f" / 头读取失败 {s['n_cases_read_error']}"
            f" / WG 缺失 {s['n_cases_wg_missing']}"
            f"（已认可已知缺失 {s['n_cases_wg_missing_recognized_known']}"
            f" / 未认可状态 {s['n_cases_wg_status_unrecognized']}）"
        )
        print(
            f"  几何基准           : 非法几何 {s['n_cases_geometry_problem']}"
            f" / T2W 不可用 {s['n_cases_t2w_geometry_unusable']}"
            f" / lesion 不对齐 T2W {s['n_cases_lesion_not_on_t2w_grid']}"
        )
    if "wg" in stages:
        counts = stages["wg"]["counts"]
        print(
            "  WG 来源对应        : "
            + " ".join(f"{k}={counts.get(k, 0)}" for k in WG_OUTCOMES)
        )
        print(
            f"  WG 唯一性          : 已解决 {stages['wg']['n_source_resolved']}"
            f" / 未解决 {stages['wg']['n_unique_source_unresolved']}"
        )
    if "zonal" in stages:
        for source, summary in stages["zonal"]["per_source_summary"].items():
            dice = summary["union_vs_wg_dice"]
            print(
                f"  zonal/{source:<4s}         : 可比 {summary['n_comparable_to_wg']}"
                f" / 空标签 {summary['n_empty_label']}"
                f" / 跳过 {summary['n_comparison_skipped']}"
                f" | union-vs-WG Dice 中位数 {dice['median']}"
            )
        x = stages["zonal"]["cross_source_agreement"]
        print(
            f"  Yuan vs Hevi 一致性 : 可比 {x['n_comparable']}"
            f" | PZ Dice 中位数 {x['pz_dice']['median']}"
            f" | TZ Dice 中位数 {x['tz_dice']['median']}"
        )
    if "prostate158" in stages:
        p = stages["prostate158"]
        print(
            f"  Prostate158        : ok {p['n_ok']} / 缺失 {p['n_missing']}"
            f" / 读取失败 {p['n_read_error']} | 语义 {p['semantic_status']}"
        )
        print(f"  观测值集合          : {p['observed_value_sets']}")
    if "roi" in stages:
        margin = stages["roi"]["roi_definition"]["margin_mm"]
        for candidate, summary in stages["roi"]["per_candidate_summary"].items():
            overall = summary["overall"]
            print(
                f"  ROI[{candidate:<20s}] margin={margin}mm"
                f" | 分母完整 {overall['denominator_complete']}"
                f" | 可评估 {overall['n_cases_reference_evaluable']}"
                f" / 不可用 {overall['n_cases_reference_unavailable']}"
                f" | 实测实例 {overall['n_reference_instances_observed']}"
                f" | 回退 {overall['n_cases_fallback']}"
            )
            if overall["denominator_complete"]:
                print(
                    f"      全部参考病灶口径 : 完全覆盖 "
                    f"{overall['n_instances_fully_covered']} / 完全漏失 "
                    f"{overall['n_instances_fully_missed']} / 体素覆盖 "
                    f"{overall['overall_voxel_coverage']}"
                )
            else:
                subset = overall["evaluable_subset_only"]
                print(
                    "      全部参考病灶口径 : null（分母不完整）"
                    f"；可评估子集仅 {subset['n_cases']} 例、"
                    f"实测实例 {subset['n_reference_instances']}"
                )
    for key in ("problems", "source_problems"):
        for item in status.get(key, []):
            print(f"  [{key}] {item}")
    print(f"  报告已写出          : {written}")
    if written:
        print(f"  报告路径            : {output_path}")
    print(f"  耗时                : {report.get('elapsed_seconds')} 秒")
    print("=== 审计结束 ===")
    print(
        "  说明：审计执行成功（execution_complete）与来源已解决"
        "（source_audit_status=resolved）是两件事；"
        "来源歧义或算法间不一致不等于程序故障。"
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_path = Path(args.output)
    if args.targeted_followup:
        try:
            return run_targeted_followup(args)
        except (AuditError, OSError, ValueError, KeyError) as exc:
            print(f"[targeted-followup] {exc}", file=sys.stderr)
            return EXIT_PREFLIGHT
    if args.derived_resampling:
        print("--derived-resampling requires --targeted-followup", file=sys.stderr)
        return EXIT_PREFLIGHT

    # 输出路径检查放在最前面，避免先跑几小时再失败。本工具无 --overwrite。
    try:
        assert_output_allowed(
            output_path,
            input_paths=(
                args.manifest,
                args.split_cases,
                args.materialization_report,
            ),
        )
    except AuditError as exc:
        print(f"[audit] 输出路径不被接受：{exc}", file=sys.stderr)
        return EXIT_PREFLIGHT

    try:
        inputs = Inputs(args)
    except AuditError as exc:
        print(f"[audit] 参数无效：{exc}", file=sys.stderr)
        return EXIT_PREFLIGHT

    # 再次校验（输入路径已知），仍不允许覆盖输入文件
    try:
        assert_output_allowed(
            output_path,
            input_paths=(
                inputs.manifest_path,
                inputs.split_cases_path,
                inputs.materialization_report_path,
            ),
        )
    except AuditError as exc:
        print(f"[audit] 输出路径不被接受：{exc}", file=sys.stderr)
        return EXIT_PREFLIGHT

    try:
        inputs.resolve()
    except AuditError as exc:
        print(f"[audit] 前置条件不满足：{exc}", file=sys.stderr)
        return EXIT_PREFLIGHT

    report = run_audit(inputs, progress=not args.no_progress)

    written = False
    try:
        publish_report(report, output_path)
        written = True
    except AuditError as exc:
        print(f"[audit] 报告写入失败：{exc}", file=sys.stderr)

    _print_summary(report, output_path, written=written)
    status = report["status"]
    if not written:
        return EXIT_PREFLIGHT
    # 只有三个状态**都显式为真**才返回 0；null（未评估 / 子集）不算为真。
    if (
        status["execution_complete"] is True
        and status["data_valid"] is True
        and status["source_resolved"] is True
    ):
        return EXIT_OK
    return EXIT_UNRESOLVED


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AuditError as exc:  # pragma: no cover - 前置失败路径
        print(f"[audit] FAILED: {exc}", file=sys.stderr)
        raise SystemExit(EXIT_PREFLIGHT) from exc
