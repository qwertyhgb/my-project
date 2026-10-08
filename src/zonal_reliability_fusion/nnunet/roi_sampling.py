"""Anatomy-Guided ROI 的训练采样层（新主线条件 B 的活跃实现）。

语义
----
条件 B 的问题是「**单纯减少无关背景**是否已经能改善 lesion learning？」。本模块用最不具
侵入性的方式回答它：**约束训练 patch 的采样空间**到前列腺 ROI（由预测 WG 加物理 margin
得到），而不是裁剪或掩掉输入影像。

为什么不做硬裁剪
----------------
1. ``image * wg_mask`` 会把 WG 边界错误直接变成「病灶在进入网络之前被删除」；
2. 裁剪输入会在推理时引入 train/test 不一致（nnU-Net 的滑窗推理在整幅图像上运行），并且需要
   额外的几何恢复步骤；
3. 约束采样空间既能直接回答条件 B 的科学问题，也**不需要**改变推理路径。

:mod:`zonal_reliability_fusion.lesion.roi` 里的 :class:`~..lesion.roi.ProstateROI` 提供了
完整的 crop transform 与无损几何恢复，供将来需要推理期裁剪的变体使用；本模块不复用它做
输入裁剪，只用它的 ``lower`` / ``upper`` 约束采样范围。

非全 ROI 采样比例
-----------------
:data:`ROI_SAMPLING_PROBABILITY_DEFAULT` 固定为 0.75：75% 的非前景槽位在 ROI 内采样，
其余 25% 保持原生全视野采样。保留一部分全视野槽位是**有意**的——滑窗推理会在整幅图像（含
腺体外背景）上运行，训练时完全排除这些位置会引入新的分布不匹配。该常量在看结果之前固定，
若需要敏感性检查必须作为**单独**的消融报告，不得事后调参。

split 与泄漏
------------
ROI 集合由 :func:`load_prostate_roi_set` 校验：病例集合必须与 provenance 中的
``train_cases`` / ``val_cases`` 一致，且 ROI 只用于**训练** loader。验证 loader 保持原生
``nnUNetDataLoader`` 与原生采样行为。
"""

from __future__ import annotations

import json
import os
from collections import deque
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader

from zonal_reliability_fusion.evaluation.anatomy_metrics import (
    ALLOWED_PRIOR_SOURCES,
    PRIOR_SOURCE_ORACLE_GT,
)
from zonal_reliability_fusion.sampling.positive_sampling import (
    PositiveCaseDataLoader,
    PositiveCaseSamplingMixin,
)

#: ROI 集合 JSON 的 schema 版本
ROI_SET_SCHEMA_VERSION = "1.0"
#: ROI 内采样槽位的默认比例（预先冻结的工程常量，见模块 docstring）
ROI_SAMPLING_PROBABILITY_DEFAULT = 0.75
#: ROI 集合的默认存放目录（相对项目根）；由用户创建，本模块不自动创建
DEFAULT_ROI_SET_DIR = "workdir/anatomy_rois"


class ROISetError(RuntimeError):
    """ROI 集合加载或校验中的可预期失败（fail-closed）。"""


def load_prostate_roi_set(
    path: str | os.PathLike,
    *,
    expected_dataset_name: str,
    expected_fold: int,
) -> dict:
    """读取并校验解剖约束 ROI 集合。

    校验项（全部 fail-closed）：

    - schema 版本、数据集归属、fold 归属；
    - ``provenance.prior_source`` 必须是 :data:`..evaluation.anatomy_metrics.ALLOWED_PRIOR_SOURCES`
      之一；使用 GT 解剖标签时必须显式标记 ``ORACLE_GT``，使 oracle 分析无法被误当作正式结果；
    - 每个病例必须有合法的 ``lower_zyx`` / ``upper_zyx``（三个非负整数、严格递增）；
    - 病例集合必须与 ``split.train_cases`` 一致（ROI 只能作用于训练 split）。
    """
    source = Path(path)
    if not source.is_file():
        raise ROISetError(f"ROI 集合不存在: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ROISetError(f"ROI 集合不是合法 JSON: {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ROISetError(f"ROI 集合顶层必须是对象: {source}")
    if payload.get("schema_version") != ROI_SET_SCHEMA_VERSION:
        raise ROISetError(
            f"schema_version 不匹配：期望 {ROI_SET_SCHEMA_VERSION}，"
            f"实际 {payload.get('schema_version')!r}"
        )
    if payload.get("dataset_name") != expected_dataset_name:
        raise ROISetError(
            f"数据集归属不匹配：集合为 {payload.get('dataset_name')!r}，"
            f"当前训练为 {expected_dataset_name!r}"
        )
    if int(payload.get("fold", -1)) != int(expected_fold):
        raise ROISetError(
            f"fold 归属不匹配：集合为 {payload.get('fold')!r}，当前 fold 为 {expected_fold}"
        )
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ROISetError("缺少 provenance")
    prior_source = provenance.get("prior_source")
    if prior_source not in ALLOWED_PRIOR_SOURCES:
        raise ROISetError(
            f"provenance.prior_source 必须是 {ALLOWED_PRIOR_SOURCES} 之一，收到 {prior_source!r}"
        )
    if prior_source == PRIOR_SOURCE_ORACLE_GT:
        raise ROISetError(
            "provenance.prior_source == ORACLE_GT：GT 解剖标签只允许用于上限（oracle）分析，"
            "禁止作为正式 lesion 实验的 ROI 来源"
        )
    split = payload.get("split")
    if not isinstance(split, Mapping) or not split.get("train_cases") or not split.get("val_cases"):
        raise ROISetError("缺少 split.train_cases / split.val_cases，无法校验范围")
    train_cases = {str(c) for c in split["train_cases"]}
    val_cases = {str(c) for c in split["val_cases"]}
    if train_cases & val_cases:
        raise ROISetError("split.train_cases 与 split.val_cases 相交，split 本身不合法")
    cases = payload.get("cases")
    if not isinstance(cases, Mapping) or not cases:
        raise ROISetError("缺少 cases 字段或 cases 为空")
    leaked = {str(c) for c in cases} - train_cases
    if leaked:
        raise ROISetError(
            f"ROI 集合包含 {len(leaked)} 个非训练 split 病例"
            f"（例如 {sorted(leaked)[:3]}）：ROI 只能作用于训练 split"
        )
    for case_id, entry in cases.items():
        if not isinstance(entry, Mapping):
            raise ROISetError(f"{case_id}: entry 必须是对象")
        for key in ("lower_zyx", "upper_zyx"):
            value = entry.get(key)
            if (
                not isinstance(value, list)
                or len(value) != 3
                or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in value)
            ):
                raise ROISetError(
                    f"{case_id}: {key} 必须是三个非负整数，收到 {value!r}"
                )
        lower, upper = entry["lower_zyx"], entry["upper_zyx"]
        if any(upper[i] <= lower[i] for i in range(3)):
            raise ROISetError(f"{case_id}: ROI 区间非法：[{lower}, {upper})")
    return payload


class ROIDataLoader(PositiveCaseDataLoader):
    """在阳性采样之上，把一部分**非前景**槽位的 bbox 约束到前列腺 ROI 内。

    与父类的差异**只有一处**：非前景槽位（``force_fg=False``）以
    ``roi_sampling_probability`` 的概率把 bbox 下界限制在 ROI 内；其余槽位（含全部阳性槽位）
    完全委托给 nnU-Net 原生 ``get_bbox``。

    阳性槽位**不**受 ROI 约束：阳性槽位由 nnU-Net 原生的病灶中心裁剪决定，若被 ROI 限制，
    analytics 上「保证一个病灶 patch」的性质会依赖 WG 预测正确，从而把 anatomy 误差引入到
    病灶暴露保证里。保持阳性槽位原样，使条件 B 与 strong baseline 的阳性暴露**完全相同**，
    单变量边界更干净。
    """

    def __init__(
        self,
        *args,
        roi_boxes: Mapping[str, Sequence[Sequence[int]]],
        roi_sampling_probability: float = ROI_SAMPLING_PROBABILITY_DEFAULT,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not 0.0 <= float(roi_sampling_probability) <= 1.0:
            raise ValueError(
                f"roi_sampling_probability 必须落在 [0,1]，收到 {roi_sampling_probability}"
            )
        boxes: dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]] = {}
        for case_id, box in roi_boxes.items():
            if not isinstance(box, Sequence) or len(box) != 2:
                raise ValueError(
                    f"{case_id}: roi_boxes 的每个条目必须是 (lower_zyx, upper_zyx) 二元组"
                )
            lower, upper = (tuple(int(v) for v in part) for part in box)
            if len(lower) != 3 or len(upper) != 3:
                raise ValueError(f"{case_id}: ROI 边界必须各含三个整数")
            if any(upper[i] <= lower[i] for i in range(3)):
                raise ValueError(f"{case_id}: ROI 区间非法：[{lower}, {upper})")
            boxes[str(case_id)] = (lower, upper)
        if not boxes:
            raise ValueError(
                "roi_boxes 不能为空：ROIDataLoader 至少需要一个训练病例的 ROI。"
                "若本轮不想启用 ROI 约束，请不要使用 ROIDataLoader。"
            )
        unknown = set(boxes) - set(self.indices)
        if unknown:
            raise ValueError(
                "ROI 必须属于该 loader 的训练 dataset identifiers；"
                f"发现 {len(unknown)} 个未知 ID（例如 {sorted(unknown)[:3]}）"
            )
        self.roi_boxes = boxes
        self.roi_sampling_probability = float(roi_sampling_probability)
        #: 当前 batch 的槽位 -> 病例 ID 队列。nnU-Net v2.6.2 的 ``generate_train_batch``
        #: 不给 ``get_bbox`` 传槽位索引，因此只能由 ``get_indices`` 填队列、``get_bbox``
        #: 按槽位顺序消费。队列长度与消费关系有显式断言，契约变化时直接报错而不是静默错位。
        self._pending_keys: deque[str] | None = None

    def get_indices(self):
        """父类（阳性采样）选例后记录槽位顺序，供 ``get_bbox`` 定位当前批次的病例。"""
        if self._pending_keys:
            raise RuntimeError(
                f"上一个 batch 的槽位队列未被完全消费（剩余 {len(self._pending_keys)} 个）："
                "nnU-Net 的 get_bbox 调用契约已改变，拒绝继续以免槽位与病例错位。"
            )
        selected_keys = list(super().get_indices())
        self._pending_keys = deque(str(key) for key in selected_keys)
        return selected_keys

    def _current_key(self, force_fg: bool) -> str | None:
        """按槽位顺序取当前病例 ID；阳性槽位不消费 ROI 决策（与原生行为保持一致）。"""
        if self._pending_keys is None:
            raise RuntimeError(
                "get_bbox 在未经过 get_indices 的情况下被调用：nnU-Net 的 dataloader 契约"
                "已改变，拒绝继续（否则 ROI 约束无法与病例对应）。"
            )
        if not self._pending_keys:
            raise RuntimeError(
                "槽位队列已空但 get_bbox 仍被调用：契约被破坏，拒绝继续。"
            )
        return self._pending_keys.popleft()

    def _native_bounds(self, data_shape, case_identifier: str):
        """计算 nnU-Net v2.6.2 ``get_bbox`` 使用的原生 ``lbs`` / ``ubs``。

        这里镜像固定版本中「need_to_pad 不足时补足、再由 patch size 推出可采样范围」的
        计算（见 ``nnunetv2/training/dataloading/data_loader.py`` 的 ``get_bbox``）。测试
        ``test_roi_sampling_matches_native_bounds`` 会对同一形状与原生实现逐值比对，从而在
        nnU-Net 升级改变该计算时**显式失败**，而不是静默错位。
        """
        need_to_pad = self.need_to_pad.copy()
        dim = len(data_shape)
        for d in range(dim):
            if need_to_pad[d] + data_shape[d] < self.patch_size[d]:
                need_to_pad[d] = self.patch_size[d] - data_shape[d]
        lbs = [-need_to_pad[i] // 2 for i in range(dim)]
        ubs = [
            data_shape[i] + need_to_pad[i] // 2 + need_to_pad[i] % 2 - self.patch_size[i]
            for i in range(dim)
        ]
        return lbs, ubs

    def get_bbox(self, data_shape, force_fg, class_locations, overwrite_class=None, verbose=False):
        """非前景槽位按概率把 bbox 约束到 ROI；其余情况完全走原生实现。"""
        case_identifier = self._current_key(force_fg)
        if force_fg:
            # 阳性槽位保持 nnU-Net 原生的病灶中心裁剪，ROI 不参与——这样条件 B 与 strong
            # baseline 的阳性病灶暴露完全相同，单变量边界更干净
            return super().get_bbox(
                data_shape, force_fg, class_locations, overwrite_class, verbose
            )
        # 逐槽位独立抽样：期望的非前景槽位 ROI 比例即为 roi_sampling_probability
        if np.random.uniform() >= self.roi_sampling_probability:
            return super().get_bbox(
                data_shape, force_fg, class_locations, overwrite_class, verbose
            )
        box = self.roi_boxes.get(case_identifier)
        if box is None:
            return super().get_bbox(
                data_shape, force_fg, class_locations, overwrite_class, verbose
            )
        lower, upper = box
        lbs, ubs = self._native_bounds(data_shape, case_identifier)
        bbox_lbs = []
        for axis in range(len(data_shape)):
            # 点 (z,y,x) 为中心时 bbox 下界的允许范围：patch 完整落在 ROI 内且不越原生边界
            roi_low = max(lbs[axis], int(lower[axis]))
            roi_high = min(ubs[axis], int(upper[axis]) - self.patch_size[axis])
            if roi_high < roi_low:
                # ROI 在该轴小于 patch size：把 patch 居中于 ROI，再夹到原生允许范围。
                # 这等价于「把 ROI 补齐到网络输入尺寸」，**不**改变 spacing、不重采样。
                centre = (int(lower[axis]) + int(upper[axis])) // 2
                bbox_lbs.append(
                    int(min(max(centre - self.patch_size[axis] // 2, lbs[axis]), ubs[axis]))
                )
            else:
                bbox_lbs.append(int(np.random.randint(roi_low, roi_high + 1)))
        bbox_ubs = [bbox_lbs[i] + self.patch_size[i] for i in range(len(data_shape))]
        return bbox_lbs, bbox_ubs


class ROISamplingMixin(PositiveCaseSamplingMixin):
    """训练 loader 换成 :class:`ROIDataLoader`；ROI 集合路径由 Trainer 提供。

    子类必须设置 ``roi_set_path``（含 predicted WG 派生的 ROI）。验证 loader 仍由父类保持
    原生 ``nnUNetDataLoader``，因此 ROI 永远不进入验证采样。
    """

    #: ROI 集合 JSON 路径；由 Trainer 的 ``initialize`` 校验存在与归属
    roi_set_path: str | None = None
    #: ROI 内采样槽位比例
    roi_sampling_probability: float = ROI_SAMPLING_PROBABILITY_DEFAULT

    def _resolve_train_loader_class(self) -> type[nnUNetDataLoader]:
        if self.roi_set_path is None:
            raise ROISetError(
                f"{type(self).__name__} 必须显式提供 roi_set_path；"
                "拒绝在路径缺失时静默退化为普通阳性采样"
            )
        return ROIDataLoader

    def _train_loader_kwargs(self, dataset_train) -> dict:
        if self.roi_set_path is None:
            raise ROISetError(
                f"{type(self).__name__} 必须显式提供 roi_set_path；拒绝静默退化"
            )
        payload = load_prostate_roi_set(
            self.roi_set_path,
            expected_dataset_name=self.plans_manager.dataset_name,
            expected_fold=self.fold,
        )
        identifiers = {str(i) for i in dataset_train.identifiers}
        boxes = {
            case_id: [entry["lower_zyx"], entry["upper_zyx"]]
            for case_id, entry in payload["cases"].items()
            if case_id in identifiers
        }
        if not boxes:
            raise ROISetError(
                f"ROI 集合 {self.roi_set_path} 在当前 fold 的训练 split 中没有任何病例："
                "拒绝静默退化为普通阳性采样"
            )
        kwargs = super()._train_loader_kwargs(dataset_train)
        kwargs.update(
            {
                "roi_boxes": boxes,
                "roi_sampling_probability": self.roi_sampling_probability,
            }
        )
        return kwargs


__all__ = (
    "DEFAULT_ROI_SET_DIR",
    "ROI_SAMPLING_PROBABILITY_DEFAULT",
    "ROI_SET_SCHEMA_VERSION",
    "ROIDataLoader",
    "ROISamplingMixin",
    "ROISetError",
    "load_prostate_roi_set",
)
