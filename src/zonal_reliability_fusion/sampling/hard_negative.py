"""解剖约束的困难负样本挖掘（新主线条件 E：training-strategy enhancement）。

针对的失败模式
--------------
阳性采样（:mod:`.positive_sampling`）已被既有实验证明能明显减少完全漏检，代价是假阳上升。
本模块回答的后续问题：

> 如何让模型更多看到「最像癌但实际上不是癌」的困难背景？

候选困难负样本的定义（预先冻结）
--------------------------------
一个体素级候选必须同时满足三个条件：

1. **prediction confidence high**：第一轮模型的阳性概率 ``>= confidence_threshold``；
2. **GT background**：该体素在第一轮使用的训练 split 标签上**不是**病灶（``pred == 1 and
   ref == 0``，即假阳）；
3. **inside / near prostate WG**：``P(WG) >= wg_threshold``（预测解剖先验，来自冻结的
   anatomy prior generator），即候选落在腺体内或腺体边缘。

再把满足条件的假阳**连通域**按体积从大到小排序，取前 ``max_locations_per_case`` 个连通域的
质心作为采样位置。质心落在预处理的坐标空间，与 nnU-Net ``class_locations`` 的坐标空间一致。

分轮（staged）流程
------------------
- **Round 1**：用 strong baseline / coarse-to-fine 模型训练，得到 checkpoint；
- **Round 2**：只对**训练 split**病例做推理，按上面的定义挖掘困难负样本，写成一个
  ``hard_negative_set.json``（含 provenance：dataset / fold / trainer / checkpoint 路径）；
- **Round 3**：训练时提高这些位置的采样概率（本模块的 ``HardNegativeDataLoader``）。

安全红线（全部 fail-closed）
---------------------------
1. **挖掘只能在训练 split 内执行**：:func:`validate_mining_scope` 逐病例核对，任何
   validation / test 病例出现在挖掘结果中立即报错。
2. **候选必须同时有 pred/ref/WG 三方证据**：形状或几何不一致即报错，不静默跳过病例。
3. **不得从 validation error 反向挑训练样本**：挖掘输入只允许训练 split 的预测与标签。
4. **开关化**：困难负样本采样由 ``hard_negative_set`` 参数显式启用；不传即完全不启用，
   因此它可以作为独立 ablation，且默认训练流程不受影响。
5. **不做在线复杂 memory bank**：只有静态、离线的 JSON 集合；不做在线更新、不跨轮累积。

采样语义（与阳性采样严格区分）
------------------------------
- 阳性采样槽位：来自**阳性病例**并强制病灶中心裁剪（``force_fg=True``）。
- 困难负样本槽位：来自**含困难负样本的病例**并强制在**存储的困难负位置**裁剪
  （``force_fg=False``——该位置是背景，走 nnU-Net 原生随机裁剪分支会被覆盖，因此本类
  在 ``get_bbox`` 里显式给出 bbox 下界，而不是依赖 ``force_fg``）。
- 其余槽位：完全保持 nnU-Net 原生行为。
- 验证 loader：**永远是**原生 ``nnUNetDataLoader``，不注入任何困难负样本。
"""

from __future__ import annotations

import json
import os
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from tqdm import tqdm

from zonal_reliability_fusion.sampling.positive_sampling import (
    PositiveCaseDataLoader,
    PositiveCaseSamplingMixin,
)

#: 困难负样本集合 JSON 的 schema 版本；字段语义变化时递增
HARD_NEGATIVE_SCHEMA_VERSION = "1.0"
#: 挖掘结果的默认存放目录（相对项目根）；由用户显式指定路径，本模块不自动创建
DEFAULT_HARD_NEGATIVE_DIR = "workdir/hard_negatives"


class HardNegativeMiningError(RuntimeError):
    """困难负样本挖掘或加载过程中的可预期失败（fail-closed）。"""


def select_hard_negative_locations(
    pred_probability: np.ndarray,
    ref_cancer: np.ndarray,
    wg_probability: np.ndarray,
    *,
    confidence_threshold: float,
    wg_threshold: float,
    max_locations_per_case: int,
    min_component_voxels: int = 1,
) -> list[list[int]]:
    """从单病例的预测概率、GT 与预测 WG 先验中选出困难负样本位置（质心）。

    纯函数：不读文件、不写文件、不依赖 nnU-Net。返回 ``[[z, y, x], ...]``，按连通域体积
    从大到小排序（同体积按字典序坐标升序，保证确定性）。

    三个条件（与模块 docstring 一致）：预测概率高 AND GT 是背景 AND 在预测 WG 内。
    连通域体积小于 ``min_component_voxels`` 的候选被丢弃（默认 1，即不丢弃）。

    几何/形状不一致、阈值越界、数组含非有限值一律报错（fail-closed）。
    """
    if not 0.0 <= confidence_threshold <= 1.0:
        raise HardNegativeMiningError(
            f"confidence_threshold 必须落在 [0,1]，收到 {confidence_threshold}"
        )
    if not 0.0 <= wg_threshold <= 1.0:
        raise HardNegativeMiningError(
            f"wg_threshold 必须落在 [0,1]，收到 {wg_threshold}"
        )
    if int(max_locations_per_case) < 0:
        raise HardNegativeMiningError("max_locations_per_case 不能为负")
    if int(min_component_voxels) < 1:
        raise HardNegativeMiningError("min_component_voxels 至少为 1")

    probability = np.asarray(pred_probability, dtype=np.float64)
    reference = np.asarray(ref_cancer, dtype=bool)
    wg = np.asarray(wg_probability, dtype=np.float64)
    shapes = {probability.shape, reference.shape, wg.shape}
    if len(shapes) != 1:
        raise HardNegativeMiningError(
            "prediction / reference / WG 概率形状必须一致；"
            f"收到 {probability.shape} / {reference.shape} / {wg.shape}；挖掘不做重采样"
        )
    for name, array in (("prediction", probability), ("WG", wg)):
        if not bool(np.isfinite(array).all()):
            raise HardNegativeMiningError(f"{name} 含非有限值")

    candidate = (
        (probability >= confidence_threshold) & (~reference) & (wg >= wg_threshold)
    )
    if not bool(candidate.any()):
        return []

    from scipy import ndimage

    structure = ndimage.generate_binary_structure(3, 1)
    labels, n_components = ndimage.label(candidate, structure=structure)
    entries: list[tuple[int, tuple[int, int, int]]] = []
    for comp_id in range(1, int(n_components) + 1):
        component = labels == comp_id
        size = int(component.sum())
        if size < int(min_component_voxels):
            continue
        coords = np.nonzero(component)
        centroid = tuple(int(round(float(c.mean()))) for c in coords)
        entries.append((size, centroid))
    # 体积降序、坐标升序：结果与容器迭代顺序无关，可重复
    entries.sort(key=lambda item: (-item[0], item[1]))
    if max_locations_per_case:
        entries = entries[: int(max_locations_per_case)]
    return [[z, y, x] for _size, (z, y, x) in entries]


def build_hard_negative_set(
    case_ids: Sequence[str],
    load_case_arrays: Callable[[str], tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    dataset_name: str,
    fold: int,
    train_cases: Sequence[str],
    val_cases: Sequence[str],
    provenance: Mapping[str, Any],
    confidence_threshold: float,
    wg_threshold: float,
    max_locations_per_case: int,
    min_component_voxels: int = 1,
    progress: bool = True,
) -> dict:
    """对训练 split 病例构建困难负样本集合（Round 2 的编排层，本身不做 IO 决策）。

    ``load_case_arrays(case_id) -> (pred_probability, ref_cancer, wg_probability)`` 由调用方
    提供：挖掘本身**不**知道数据来自哪里，这样单元测试可以用合成数组驱动，而真实运行时由
    ``scripts/data/mine_hard_negatives.py`` 负责读取第一轮预测、训练标签与 predicted WG。

    ``train_cases`` / ``val_cases`` 用于 :func:`validate_mining_scope` 的 split 完整性检查；
    ``case_ids`` 必须是 ``train_cases`` 的子集。逐例进度默认开启（``progress=False`` 关闭）。
    """
    validate_mining_scope(
        {"cases": {cid: [] for cid in case_ids}},
        train_cases=train_cases,
        val_cases=val_cases,
    )
    cases: dict[str, dict] = {}
    for case_id in tqdm(
        case_ids, desc="hard-negative mining", unit="case", disable=not progress
    ):
        arrays = load_case_arrays(case_id)
        if not isinstance(arrays, tuple) or len(arrays) != 3:
            raise HardNegativeMiningError(
                f"{case_id}: load_case_arrays 必须返回 (prediction, reference, WG) 三元组"
            )
        locations = select_hard_negative_locations(
            arrays[0],
            arrays[1],
            arrays[2],
            confidence_threshold=confidence_threshold,
            wg_threshold=wg_threshold,
            max_locations_per_case=max_locations_per_case,
            min_component_voxels=min_component_voxels,
        )
        cases[str(case_id)] = {"locations": locations}
    return {
        "schema_version": HARD_NEGATIVE_SCHEMA_VERSION,
        "dataset_name": str(dataset_name),
        "fold": int(fold),
        "definition": {
            "candidate_rule": (
                "prediction_probability >= confidence_threshold AND reference == background "
                "AND WG_probability >= wg_threshold"
            ),
            "location": "候选假阳连通域（6-邻域）的质心，预处理坐标空间，与 class_locations 一致",
            "selection_order": (
                "连通域体积降序；同体积按质心坐标字典序升序（确定性，与容器迭代顺序无关）"
            ),
            "coordinate_space": (
                "preprocessed (cropped) 空间，与 nnU-Net class_locations 相同；"
                "挖掘不做重采样、不做坐标变换"
            ),
            "not_online": (
                "静态离线集合；不做在线 memory bank、不跨轮累积、不从 validation 反向挑选"
            ),
        },
        "thresholds": {
            "confidence_threshold": float(confidence_threshold),
            "wg_threshold": float(wg_threshold),
            "min_component_voxels": int(min_component_voxels),
            "max_locations_per_case": int(max_locations_per_case),
        },
        "split": {
            "train_cases": sorted(str(c) for c in train_cases),
            "val_cases": sorted(str(c) for c in val_cases),
        },
        "provenance": dict(provenance),
        "cases": cases,
        "summary": {
            "mined_cases": len(cases),
            "cases_with_hard_negatives": sum(
                1 for entry in cases.values() if entry["locations"]
            ),
            "total_locations": sum(len(entry["locations"]) for entry in cases.values()),
        },
    }


def validate_mining_scope(
    payload: Mapping[str, Any],
    *,
    train_cases: Iterable[str],
    val_cases: Iterable[str],
) -> None:
    """split 完整性守卫：挖掘病例必须全部属于训练 split，且与验证 split 完全不相交。

    任何越界立即抛 :class:`HardNegativeMiningError`（fail-closed），不静默丢弃病例。
    """
    train = {str(c) for c in train_cases}
    val = {str(c) for c in val_cases}
    if not train:
        raise HardNegativeMiningError("train_cases 为空：无法判定挖掘范围")
    overlap = train & val
    if overlap:
        raise HardNegativeMiningError(
            f"train_cases 与 val_cases 相交（{len(overlap)} 例，例如 {sorted(overlap)[:3]}）："
            "split 本身不合法，拒绝继续"
        )
    mined = {str(c) for c in payload.get("cases", {})}
    leaked = mined & val
    if leaked:
        raise HardNegativeMiningError(
            f"困难负样本挖掘包含 {len(leaked)} 个验证病例（例如 {sorted(leaked)[:3]}）："
            "验证集绝不参与挖掘，拒绝继续"
        )
    outside = mined - train
    if outside:
        raise HardNegativeMiningError(
            f"困难负样本挖掘包含 {len(outside)} 个不在训练 split 的病例"
            f"（例如 {sorted(outside)[:3]}）：拒绝继续"
        )


def write_hard_negative_set(path: str | os.PathLike, payload: Mapping[str, Any]) -> Path:
    """原子写出挖掘结果；已存在则拒绝覆盖（fail-closed，与评估器落盘策略一致）。"""
    target = Path(path)
    if target.exists():
        raise HardNegativeMiningError(f"输出文件已存在，拒绝静默覆盖: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False, allow_nan=False)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return target


def load_hard_negative_set(
    path: str | os.PathLike, *, expected_dataset_name: str, expected_fold: int
) -> dict:
    """读取并校验挖掘结果：schema 版本、数据集/fold 归属、坐标合法性与 split 范围。"""
    source = Path(path)
    if not source.is_file():
        raise HardNegativeMiningError(f"困难负样本集合不存在: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HardNegativeMiningError(f"困难负样本集合不是合法 JSON: {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise HardNegativeMiningError(f"困难负样本集合顶层必须是对象: {source}")
    if payload.get("schema_version") != HARD_NEGATIVE_SCHEMA_VERSION:
        raise HardNegativeMiningError(
            f"schema_version 不匹配：期望 {HARD_NEGATIVE_SCHEMA_VERSION}，"
            f"实际 {payload.get('schema_version')!r}"
        )
    if payload.get("dataset_name") != expected_dataset_name:
        raise HardNegativeMiningError(
            f"数据集归属不匹配：集合为 {payload.get('dataset_name')!r}，"
            f"当前训练为 {expected_dataset_name!r}"
        )
    if int(payload.get("fold", -1)) != int(expected_fold):
        raise HardNegativeMiningError(
            f"fold 归属不匹配：集合为 {payload.get('fold')!r}，当前训练 fold 为 {expected_fold}"
        )
    split = payload.get("split")
    if not isinstance(split, dict) or not split.get("train_cases") or not split.get("val_cases"):
        raise HardNegativeMiningError("缺少 split.train_cases / split.val_cases，无法校验范围")
    validate_mining_scope(
        payload, train_cases=split["train_cases"], val_cases=split["val_cases"]
    )
    cases = payload.get("cases")
    if not isinstance(cases, dict):
        raise HardNegativeMiningError("缺少 cases 字段")
    for case_id, entry in cases.items():
        if not isinstance(entry, dict) or not isinstance(entry.get("locations"), list):
            raise HardNegativeMiningError(f"{case_id}: locations 缺失或不是列表")
        for location in entry["locations"]:
            if (
                not isinstance(location, list)
                or len(location) != 3
                or any(
                    isinstance(value, bool) or not isinstance(value, int) or value < 0
                    for value in location
                )
            ):
                raise HardNegativeMiningError(
                    f"{case_id}: 非法困难负样本坐标 {location!r}"
                    "（要求三个非负整数，预处理坐标空间）"
                )
    return payload


class HardNegativeDataLoader(PositiveCaseDataLoader):
    """在阳性采样之上，把若干 batch 槽位固定到存储的困难负样本位置。

    与父类的差异**只有一处**：``get_bbox`` 对「困难负样本槽位」直接给出以存储位置为中心的
    下界坐标，其余槽位原样委托给 nnU-Net 原生 ``get_bbox``。

    槽位分配（确定性，不依赖随机数决策）
    ------------------------------------
    - 最后 ``positive_cases_per_batch`` 个槽位：阳性病例 + 病灶中心裁剪（父类行为，不变）；
    - 其前的 ``hard_negative_cases_per_batch`` 个槽位：来自**含困难负样本的病例**，位置从该
      病例的 locations 中均匀有放回抽取，且**不**走 ``force_fg``（该位置是背景）；
    - 其余槽位：完全原生。

    fail-closed 约束：``positive_cases_per_batch + hard_negative_cases_per_batch <=
    batch_size``；困难负样本病例 ID 必须属于本 loader 的训练 dataset；每次
    ``get_indices`` 都会重置并按 batch 顺序填满决策队列，若上一次 batch 的决策未被
    ``get_bbox`` 全部消费则报错（防止槽位与位置错位）。
    """

    def __init__(
        self,
        *args,
        hard_negative_locations: Mapping[str, Sequence[Sequence[int]]],
        hard_negative_cases_per_batch: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        locations: dict[str, tuple[tuple[int, int, int], ...]] = {}
        for case_id, entries in hard_negative_locations.items():
            if not entries:
                continue
            locations[str(case_id)] = tuple(
                (int(z), int(y), int(x)) for z, y, x in entries
            )
        if not locations:
            raise ValueError(
                "hard_negative_locations 不能为空：至少要有一个含困难负样本的训练病例。"
                "若本轮不想启用困难负样本采样，请不要使用 HardNegativeDataLoader。"
            )
        unknown = set(locations) - set(self.indices)
        if unknown:
            raise ValueError(
                "困难负样本位置必须属于该 loader 的训练 dataset identifiers；"
                f"发现 {len(unknown)} 个未知 ID（例如 {sorted(unknown)[:3]}）"
            )
        positive_count = int(self.positive_cases_per_batch)
        hard_negative_count = int(hard_negative_cases_per_batch)
        if hard_negative_count < 1:
            raise ValueError("hard_negative_cases_per_batch 必须至少为 1")
        if positive_count + hard_negative_count > self.batch_size:
            raise ValueError(
                "positive_cases_per_batch + hard_negative_cases_per_batch 不能超过 "
                f"batch_size；收到 {positive_count} + {hard_negative_count} > {self.batch_size}"
            )
        if self.get_do_oversample == self._probabilistic_oversampling:
            raise ValueError(
                "HardNegativeDataLoader 要求 probabilistic_oversampling=False："
                "概率式 oversampling 无法保证阳性槽位始终执行病灶中心裁剪。"
            )
        self.hard_negative_locations = locations
        self.hard_negative_cases_per_batch = hard_negative_count
        self._hard_negative_case_ids = tuple(sorted(locations))
        #: 每个 batch 槽位 -> 存储的困难负样本位置（``None`` 表示该槽位不是困难负样本槽位）
        self._slot_decisions: deque[tuple[int, int, int] | None] = deque()

    def _first_hard_negative_position(self) -> int:
        return self.batch_size - self.positive_cases_per_batch - self.hard_negative_cases_per_batch

    def get_indices(self):
        """父类（阳性采样）选例后，再把中间区段替换为含困难负样本的病例。

        同时按 batch 槽位顺序填满决策队列；若上一 batch 的队列未被完全消费，说明
        ``get_bbox`` 的调用契约被破坏，直接报错而不是静默错位。
        """
        if self._slot_decisions:
            raise RuntimeError(
                f"上一个 batch 的困难负样本槽位决策未全部消费（剩余 "
                f"{len(self._slot_decisions)} 个）：nnU-Net 的 get_bbox 调用契约已改变，"
                "拒绝继续以免槽位与位置错位。"
            )
        selected_keys = list(super().get_indices())
        first = self._first_hard_negative_position()
        stop = self.batch_size - self.positive_cases_per_batch
        chosen = np.random.choice(
            self._hard_negative_case_ids,
            size=self.hard_negative_cases_per_batch,
            replace=True,
        ).tolist()
        selected_keys[first:stop] = chosen
        decisions: list[tuple[int, int, int] | None] = [None] * self.batch_size
        for offset, case_id in enumerate(chosen):
            pool = self.hard_negative_locations[case_id]
            decisions[first + offset] = pool[int(np.random.randint(len(pool)))]
        self._slot_decisions = deque(decisions)
        return selected_keys

    def get_bbox(self, data_shape, force_fg, class_locations, overwrite_class=None, verbose=False):
        """困难负样本槽位：以存储位置为中心给出 bbox 下界；其余槽位走原生实现。"""
        if not self._slot_decisions:
            raise RuntimeError(
                "get_bbox 在未经过 get_indices 的情况下被调用：nnU-Net 的 dataloader 契约已"
                "改变，拒绝继续（否则困难负样本槽位无法与位置对应）。"
            )
        location = self._slot_decisions.popleft()
        if location is None:
            return super().get_bbox(
                data_shape, force_fg, class_locations, overwrite_class, verbose
            )
        if force_fg:
            raise RuntimeError(
                "困难负样本槽位不应被标记为 force_fg：该位置是背景，nnU-Net 原生的前景裁剪"
                "会覆盖它。请检查 _oversample_last_XX_percent 的槽位划分。"
            )
        need_to_pad = self.need_to_pad.copy()
        dim = len(data_shape)
        for d in range(dim):
            if need_to_pad[d] + data_shape[d] < self.patch_size[d]:
                need_to_pad[d] = self.patch_size[d] - data_shape[d]
        lbs = [-need_to_pad[i] // 2 for i in range(dim)]
        bbox_lbs = [
            max(lbs[i], int(location[i]) - self.patch_size[i] // 2) for i in range(dim)
        ]
        bbox_ubs = [bbox_lbs[i] + self.patch_size[i] for i in range(dim)]
        return bbox_lbs, bbox_ubs

    def _oversample_last_XX_percent(self, sample_idx: int) -> bool:
        """只有最后 ``positive_cases_per_batch`` 个槽位强制前景；困难负样本槽位不强制。"""
        return sample_idx >= self.batch_size - self.positive_cases_per_batch


class HardNegativeMiningMixin(PositiveCaseSamplingMixin):
    """在阳性采样之上启用困难负样本采样；不传集合即完全不启用（可作独立 ablation）。

    子类必须设置 :attr:`hard_negative_set_path`（含困难负样本的 JSON 路径）。``None`` 时
    本 mixin 完全退化为 :class:`PositiveCaseSamplingMixin` 的行为，因此「开 / 关」是同一个
    Trainer 类的一个参数，不需要两套代码。
    """

    #: 困难负样本集合 JSON 路径；``None`` = 不启用（默认）
    hard_negative_set_path: str | None = None
    #: 每个 batch 固定给困难负样本的槽位数
    hard_negative_cases_per_batch: int = 1

    def _load_hard_negative_locations(self, dataset_train) -> dict:
        payload = load_hard_negative_set(
            self.hard_negative_set_path,
            expected_dataset_name=self.plans_manager.dataset_name,
            expected_fold=self.fold,
        )
        identifiers = set(dataset_train.identifiers)
        return {
            case_id: entry["locations"]
            for case_id, entry in payload["cases"].items()
            if entry["locations"] and case_id in identifiers
        }

    def _resolve_train_loader_class(self) -> type[nnUNetDataLoader]:
        """启用困难负样本时返回其 loader；未配置路径时完全退化为父类行为。"""
        if self.hard_negative_set_path is None:
            return super()._resolve_train_loader_class()
        return (ROIHardNegativeDataLoader if getattr(self, "roi_set_path", None)
                else HardNegativeDataLoader)

    def _train_loader_kwargs(self, dataset_train) -> dict:
        if self.hard_negative_set_path is None:
            return super()._train_loader_kwargs(dataset_train)
        locations = self._load_hard_negative_locations(dataset_train)
        if not locations:
            raise HardNegativeMiningError(
                f"困难负样本集合 {self.hard_negative_set_path} 在当前 fold 的训练 split 中"
                "没有任何可用位置：拒绝静默退化为普通阳性采样"
            )
        kwargs = super()._train_loader_kwargs(dataset_train)
        kwargs.update(
            {
                "hard_negative_locations": locations,
                "hard_negative_cases_per_batch": self.hard_negative_cases_per_batch,
            }
        )
        self.print_to_log_file(
            "Hard-negative mining enabled:\n"
            f"source={self.hard_negative_set_path}\n"
            f"cases_with_hard_negatives={len(locations)}\n"
            f"total_locations={sum(len(v) for v in locations.values())}\n"
            f"hard_negative_cases_per_batch={self.hard_negative_cases_per_batch}"
        )
        return kwargs


# Compose existing native-loader extensions; keep ROI for non-mined slots.
from zonal_reliability_fusion.nnunet.roi_sampling import ROIDataLoader


class ROIHardNegativeDataLoader(HardNegativeDataLoader, ROIDataLoader):
    """Consume both slot queues when a mined coordinate bypasses ROI get_bbox."""

    def get_bbox(self, data_shape, force_fg, class_locations, overwrite_class=None, verbose=False):
        if self._slot_decisions and self._slot_decisions[0] is not None:
            self._current_key(force_fg)
        return super().get_bbox(data_shape, force_fg, class_locations, overwrite_class, verbose)
