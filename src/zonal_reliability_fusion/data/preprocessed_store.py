"""nnU-Net 预处理数据（.b2nd/.pkl）只读适配层。

**第三方依赖边界**：全项目唯一允许 import `nnunetv2` 的位置是本模块
`PreprocessedStore._open_dataset()`（惰性 import，固定 v2.6.2 的 `nnUNetDatasetBlosc2`）。
模型、损失、trainer、推理与滑窗模块都不得依赖 nnunetv2 / nnUNetTrainer。
不修改 `third_party/`，不自行发明 .b2nd 解析格式。

数据契约（P2-A 约定）：
- data:  [C, D, H, W]，C 必须为 3（T2W/ADC/HBV），float32；
- seg:   存储态为 [D,H,W] 或 [1,D,H,W]，取值必须 ⊆ {-1, 0, 1}；其中 **-1 是 nnU-Net 预处理/
         填充哨兵**（Dataset605 只定义 background=0 与 lesion=1，无语义 ignore 类），由
         `normalize_seg` 按官方 `RemoveLabelTansform(-1, 0)` 的语义映射为 0；
         读取返回的 `PreprocessedCase.seg` 已是 [1, D, H, W] **uint8 且严格 ⊆ {0, 1}**；
- properties: 保留 `spacing`、`class_locations`、`shape_before_cropping` 等字段；
- 空间参考系为 nnU-Net 预处理后的内部空间（与官方训练一致）。

本模块只做读取与契约校验，不做采样、不做增强。
"""
from __future__ import annotations

import copy
import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

try:  # tqdm 在项目脚本中为标准依赖
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore[assignment]

EXPECTED_CHANNELS = 3
FOREGROUND_LABEL = 1
#: 存储态 seg（nnU-Net 预处理产物）允许的精确取值；-1 是填充/ignore 哨兵，不是第三个类别
SEG_STORAGE_LABELS: tuple[int, ...] = (-1, 0, 1)
#: nnU-Net 填充/ignore 哨兵（进入 sampler/loss/metric 前必须映射为 0）
SEG_SENTINEL_LABEL = -1
#: PZ/TZ prior 的通道数（channel 0 = PZ，channel 1 = TZ）；WG 不得进入该 sidecar
ZONAL_PRIOR_CHANNELS = 2
#: sidecar 的 PZ/TZ occupancy 允许的数值容差（重采样后的 fractional occupancy）
ZONAL_PRIOR_TOL = 1e-4


class PreprocessedStoreError(RuntimeError):
    """预处理数据缺失、结构非法或契约不匹配。"""


@dataclass(frozen=True)
class CaseProperties:
    """从 .pkl 中提取的关键属性（保留原始 dict 以便追溯）。"""

    spacing: tuple[float, ...] | None
    shape_before_cropping: tuple[int, ...] | None
    class_locations: Mapping[int, np.ndarray] | None = None
    raw: Mapping[str, object] = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class PreprocessedCase:
    """单个病例的预处理数组与属性。"""

    case_id: str
    data: np.ndarray  # [C, D, H, W] float32
    seg: np.ndarray  # [1, D, H, W] uint8；已由 normalize_seg 完成 -1→0，严格取值 {0,1}
    properties: CaseProperties
    #: 可选：plan-space PZ/TZ sidecar，[2, D, H, W] float32（channel 0 = PZ，1 = TZ）；未启用时为 None
    zonal_prior: np.ndarray | None = None


def normalize_seg(seg: np.ndarray, case_id: str = "") -> np.ndarray:
    """把 nnU-Net 存储态 seg 规范为 [1, D, H, W] **uint8 且严格 ⊆ {0,1}**。

    存储态契约（nnU-Net 预处理产物）：只允许精确取值 `{-1, 0, 1}`；其中 **-1 是 nnU-Net 的
    填充/ignore 哨兵**（本项目 Dataset605 只定义 background=0、lesion=1，无第三个语义类别），
    官方训练与验证 transform 均执行 `RemoveLabelTansform(-1, 0)`（纯值映射，不裁剪、不截断）。

    转换顺序（**先验证、后转换**，任何一步失败都不产生输出）：
    1) dtype 白名单：bool / 有符号整型 / 无符号整型 / 浮点；complex、字符串、object 等直接拒绝；
    2) shape：[D,H,W] → 补通道；[1,D,H,W] 保持；其余拒绝；
    3) 有限性：NaN / +Inf / -Inf 拒绝；
    4) 整数性：0.5 / 1.5 等非整数标签拒绝；
    5) 取值集合：出现 {-1,0,1} 之外的任何值即拒绝（并列出实际取值）；
    6) 拷贝后映射：`out = (probe == 1)`，即 -1（哨兵）与 0 都落到 0、1 保持 1；返回 contiguous uint8。

    明确**禁止**：先 `astype(uint8)`（-1 会静默变成 255）、`np.clip`、`int()` 截断、
    原地修改调用者输入；也**不会**用宽松转换掩盖非法标签（如 -2 / 2 / 浮点小数）。
    """
    arr = np.asarray(seg)
    if arr.dtype.kind not in ("b", "i", "u", "f"):
        raise PreprocessedStoreError(
            f"{case_id}: seg dtype={arr.dtype}（kind={arr.dtype.kind!r}）不受支持；"
            "只接受 bool/整型/浮点标签数组（complex、字符串、object 等一律拒绝）"
        )
    if arr.ndim == 3:
        arr = arr[None]
    elif not (arr.ndim == 4 and arr.shape[0] == 1):
        raise PreprocessedStoreError(f"{case_id}: seg 形状必须为 [D,H,W] 或 [1,D,H,W]，收到 {arr.shape}")

    out = np.zeros(arr.shape, dtype=np.uint8)  # 新数组：绝不原地修改输入
    if arr.size == 0:
        return np.ascontiguousarray(out)

    # 取值探针统一用 float64：{-1,0,1} 在 float64 中精确可表示，且避免 uint64→int64 溢出
    probe = arr.astype(np.float64)
    if not np.isfinite(probe).all():
        bad = probe[~np.isfinite(probe)].reshape(-1)[:5].tolist()
        raise PreprocessedStoreError(f"{case_id}: seg 含非有限值（NaN/Inf），示例 {bad}")
    rounded = np.rint(probe)
    if not np.array_equal(probe, rounded):
        bad = probe[probe != rounded].reshape(-1)[:5].tolist()
        raise PreprocessedStoreError(
            f"{case_id}: seg 含非整数标签 {bad}（拒绝截断/四舍五入，必须为精确整数）"
        )
    uniq = np.unique(probe)
    illegal = [float(v) for v in uniq if v not in (-1.0, 0.0, 1.0)]
    if illegal:
        shown = [int(v) if float(v).is_integer() else v for v in illegal[:10]]
        raise PreprocessedStoreError(
            f"{case_id}: seg 取值必须 ⊆ {{-1,0,1}}（-1 为 nnU-Net 填充哨兵，不是第三个类别），"
            f"实际含 {shown}"
        )
    # 等价于官方的 RemoveLabelTansform(-1, 0) 之后再取二值标签：-1 与 0 → 0，1 → 1
    out[probe == 1.0] = 1
    return np.ascontiguousarray(out)


def normalize_data(data: np.ndarray, case_id: str = "", expected_channels: int = EXPECTED_CHANNELS) -> np.ndarray:
    """把 data 规范为 [C, D, H, W] float32，并校验通道数。"""
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 4:
        raise PreprocessedStoreError(f"{case_id}: data 形状必须为 [C,D,H,W]，收到 {arr.shape}")
    if arr.shape[0] != expected_channels:
        raise PreprocessedStoreError(f"{case_id}: data 通道数必须为 {expected_channels}，收到 {arr.shape[0]}")
    return np.ascontiguousarray(arr)


class PreprocessedStore:
    """只读访问 nnUNet_preprocessed/Dataset605_PICAI/nnUNetPlans_3d_fullres。

    仅使用 fold 0（冻结单 fold 划分）。构造时只读取 splits_final.json 与目录清单，
    不加载任何病例体素；`get_case()` 才会读取 .b2nd/.pkl。
    """

    def __init__(
        self,
        dataset_folder: str | Path,
        splits_file: str | Path,
        *,
        fold: int = 0,
        dataset_identifier: str = "nnUNetPlans_3d_fullres",
        zonal_prior_root: str | Path | None = None,
        zonal_prior_source: str | None = None,
        expected_plans_sha256: str | None = None,
        expected_configuration: str | None = None,
        require_frozen_manifest: bool = False,
    ) -> None:
        self.dataset_folder = Path(dataset_folder)
        self.splits_file = Path(splits_file)
        self.fold = int(fold)
        self.dataset_identifier = dataset_identifier
        # PZ/TZ prior sidecar（可选；M3/M4 必须显式提供，M0–M2 必须不提供）
        if (zonal_prior_root is None) != (zonal_prior_source is None):
            raise PreprocessedStoreError(
                "zonal_prior_root 与 zonal_prior_source 必须同时提供或同时省略"
                "（来源不得静默推断）"
            )
        if zonal_prior_source is not None and zonal_prior_source not in ("yuan", "hevi"):
            raise PreprocessedStoreError(
                f"zonal_prior_source={zonal_prior_source!r} 非法；必须是 'yuan' 或 'hevi'"
            )
        if zonal_prior_source is not None and (expected_plans_sha256 is None or expected_configuration is None):
            raise PreprocessedStoreError(
                "启用 PZ/TZ prior 时必须同时提供 expected_plans_sha256 与 expected_configuration："
                "sidecar 必须与当前 plan/config 绑定（只比 shape 不够）"
            )
        self.zonal_prior_source = zonal_prior_source
        self.zonal_prior_root = Path(zonal_prior_root) if zonal_prior_root is not None else None
        self.expected_plans_sha256 = expected_plans_sha256
        self.expected_configuration = expected_configuration
        #: 数据集级 manifest 哈希（load_zonal_prior_manifest 成功后填充；进入 run 快照/checkpoint）
        self.zonal_prior_manifest_sha256 = None
        #: manifest 校验是否逐个实际校验了 NPZ 数组内容（False 时不得声称「全部 sidecar 已验证」）
        self.zonal_prior_array_hash_checked = False
        #: 最近一次 manifest 深度校验的统计（n_cases/n_ok/n_failed/elapsed_sec/array_hash_checked）
        self.zonal_prior_validation_stats: dict = {}
        #: 冻结的 manifest 记录：case_id → {input_sha256, output_sha256, metadata_sha256}
        #: （外层与嵌套 input_sha256 均为 MappingProxyType，深层不可变）
        #: 逐例读取必须与它比对，确保训练实际读取的 prior 与快照中冻结的 manifest 一致
        self.zonal_prior_frozen_records: Mapping[str, Mapping[str, Any]] = MappingProxyType({})
        #: 正式 M3/M4 训练置 True：未成功冻结 manifest 记录前，任何逐例 prior 读取都必须失败
        self.require_frozen_manifest = bool(require_frozen_manifest)
        self.data_folder = self.dataset_folder / dataset_identifier
        if not self.data_folder.is_dir():
            raise PreprocessedStoreError(
                f"预处理数据目录不存在: {self.data_folder}；请确认 Dataset605 已完成 plan/preprocess"
            )
        if not self.splits_file.is_file():
            raise PreprocessedStoreError(f"split 文件不存在: {self.splits_file}")

        splits = json.loads(self.splits_file.read_text())
        if not isinstance(splits, list) or not splits:
            raise PreprocessedStoreError(f"{self.splits_file} 必须是非空 fold 列表")
        if self.fold < 0 or self.fold >= len(splits):
            raise PreprocessedStoreError(f"fold={self.fold} 超出范围（文件含 {len(splits)} 个 fold）")
        fold_doc = splits[self.fold]
        self.train_ids: tuple[str, ...] = tuple(sorted(str(i) for i in fold_doc["train"]))
        self.val_ids: tuple[str, ...] = tuple(sorted(str(i) for i in fold_doc["val"]))
        if set(self.train_ids) & set(self.val_ids):
            raise PreprocessedStoreError("train / val 存在交集，split 文件不合法")

        self._dataset = None

    # ------------------------------------------------------------------ 打开
    def _open_dataset(self):
        """惰性打开第三方数据集对象（唯一的第三方依赖点）。"""
        if self._dataset is None:
            try:
                from nnunetv2.training.dataloading.nnunet_dataset import (
                    nnUNetDatasetBlosc2,
                )
            except ImportError as exc:  # pragma: no cover - 环境已确认可用
                raise PreprocessedStoreError(
                    "无法 import nnunetv2；请先 `source scripts/env_nnunet.sh`（固定 v2.6.2 源码树）"
                ) from exc
            self._dataset = nnUNetDatasetBlosc2(folder=str(self.data_folder))
            available = set(self._dataset.identifiers)
            missing_train = [i for i in self.train_ids if i not in available]
            missing_val = [i for i in self.val_ids if i not in available]
            if missing_train or missing_val:
                raise PreprocessedStoreError(
                    f"split 中的病例在预处理目录缺失：train {missing_train[:5]}（共 {len(missing_train)}），"
                    f"val {missing_val[:5]}（共 {len(missing_val)}）"
                )
        return self._dataset

    @property
    def available_ids(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.train_ids) | set(self.val_ids)))

    # ------------------------------------------------------------------ 读取
    def get_case(
        self,
        case_id: str,
        *,
        expected_channels: int = EXPECTED_CHANNELS,
        include_zonal_prior: bool = False,
    ) -> PreprocessedCase:
        """读取单个病例（data/seg/properties，可选 PZ/TZ prior）并做契约校验。

        `include_zonal_prior=True` 时必须已配置 `zonal_prior_root`/`zonal_prior_source`；
        sidecar 缺失、形状与 seg 不一致、取值越界或 PZ+TZ>1 一律报错（不静默兜底）。
        """
        dataset = self._open_dataset()
        data_raw, seg_raw, _seg_prev, properties_raw = dataset[case_id]
        data = normalize_data(data_raw, case_id=case_id, expected_channels=expected_channels)
        seg = normalize_seg(seg_raw, case_id=case_id)
        if seg.shape[-3:] != data.shape[-3:]:
            raise PreprocessedStoreError(
                f"{case_id}: data 空间尺寸 {data.shape[-3:]} 与 seg {seg.shape[-3:]} 不一致"
            )
        props = CaseProperties(
            spacing=tuple(float(v) for v in properties_raw["spacing"]) if "spacing" in properties_raw else None,
            shape_before_cropping=tuple(int(v) for v in properties_raw["shape_before_cropping"])
            if "shape_before_cropping" in properties_raw
            else None,
            class_locations=properties_raw.get("class_locations"),
            raw=properties_raw,
        )
        prior = self.load_zonal_prior(case_id, seg_shape=seg.shape[-3:]) if include_zonal_prior else None
        return PreprocessedCase(case_id=case_id, data=data, seg=seg, properties=props, zonal_prior=prior)

    # ------------------------------------------------------------------ PZ/TZ sidecar
    def load_zonal_prior(self, case_id, *, seg_shape, verify_array_hash: bool = True):
        """读取并**完整校验**单个病例的 plan-space PZ/TZ sidecar（统一校验器 `validate_sidecar`）。

        校验项与 manifest 校验完全一致：case_id、source、configuration、plans_sha256、input_sha256 格式、
        oracle 严格通过、output_shape/dtype/channel semantics、数组 SHA256、数值契约
        （finite / [0,1] / PZ+TZ<=1），并要求 prior 空间尺寸与 seg 逐值一致（禁止训练时临时重采样）。
        """
        from .zonal_prior import ZonalPriorError, validate_sidecar

        if self.zonal_prior_root is None or self.zonal_prior_source is None:
            raise PreprocessedStoreError(
                "未配置 zonal_prior_root/zonal_prior_source，禁止读取 PZ/TZ prior"
                "（M3/M4 必须在实验配置中显式声明来源）"
            )
        frozen = self.zonal_prior_frozen_records.get(str(case_id))
        if frozen is None and self.require_frozen_manifest:
            raise PreprocessedStoreError(
                f"{case_id}: 尚未成功冻结 manifest 记录，禁止读取 prior"
                "（正式 M3/M4 训练必须先通过 load_zonal_prior_manifest 深度预检；"
                "不得在未冻结状态下逐例读取）"
            )
        try:
            validated = validate_sidecar(
                self.zonal_prior_root,
                case_id,
                expected_source=self.zonal_prior_source,
                expected_configuration=self.expected_configuration,
                expected_plans_sha256=self.expected_plans_sha256,
                # 冻结记录存在时必须逐值复核（防止预检后被替换）
                expected_input_sha256=None if frozen is None else frozen.get("input_sha256"),
                expected_metadata_sha256=None if frozen is None else frozen.get("metadata_sha256"),
                expected_output_sha256=None if frozen is None else frozen.get("output_sha256"),
                expected_array_shape=seg_shape,
                verify_array_hash=verify_array_hash,
            )
        except ZonalPriorError as exc:
            raise PreprocessedStoreError(f"{case_id}: PZ/TZ sidecar 校验失败: {exc}") from exc
        if validated.prior is None:  # pragma: no cover - read_array 默认 True
            raise PreprocessedStoreError(f"{case_id}: 未读取 prior 数组（不得声称已校验）")
        return validated.prior

    def load_zonal_prior_manifest(
        self,
        expected_case_ids,
        *,
        verify_array_hash: bool = True,
        progress: bool = False,
        desc: str = "prior sidecar 校验",
    ):
        """读取并**逐个深度校验**数据集级 manifest（覆盖范围必须等于 train∪val）。

        逐例调用统一校验器 `zonal_prior.validate_sidecar`：case_id、source、configuration、
        plans_sha256、input_sha256（与 manifest 完全一致且格式合法）、metadata_sha256（== 实际
        canonical JSON）、output_sha256（与 manifest 一致）、output_shape（== 实际数组）、
        output_dtype/channel semantics、oracle 严格通过、数组 SHA256、数值契约
        （finite / [0,1] / PZ+TZ<=1）。任一例不通过都在返回前失败（fail-closed）。

        `progress=True` 时显示 tqdm 进度（当前/总数/比例/速率/ETA，**惰性推进**，与校验交错）；
        库函数不做任何默认打印，统计放在 `self.zonal_prior_validation_stats`，由脚本层输出。
        每次调用都完整重扫（不保留跨调用缓存：mtime/size 之类的弱指纹不足以承担完整性职责），
        成功后一次性提交 READY 状态与不可变的逐例冻结记录（训练逐例读取据此复核）。
        """
        import time

        from .zonal_prior import (
            MANIFEST_FILENAME,
            ZonalPriorError,
            validate_sidecar,
            verify_manifest,
        )

        ids = [str(c) for c in expected_case_ids]
        # 任何一次非缓存深度校验开始前，先把 READY 状态与冻结记录复位为未就绪；
        # 只有全部病例成功后才一次性提交。异常路径绝不残留上一次的成功状态。
        self._reset_zonal_prior_ready()
        if self.zonal_prior_root is None:
            raise PreprocessedStoreError("未配置 zonal_prior_root，无法读取 prior manifest")
        path = self.zonal_prior_root / MANIFEST_FILENAME
        if not path.is_file():
            raise PreprocessedStoreError(
                f"prior manifest 不存在: {path}；请先运行 scripts/data/materialize_zonal_prior.py"
            )
        try:
            doc = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise PreprocessedStoreError(f"prior manifest 损坏: {path}: {exc}") from exc
        try:
            manifest = verify_manifest(
                doc,
                ids,
                source=self.zonal_prior_source,
                plans_sha256=self.expected_plans_sha256,
                configuration=self.expected_configuration,
            )
        except Exception as exc:
            raise PreprocessedStoreError(f"prior manifest 校验失败: {exc}") from exc

        t0 = time.time()
        # 惰性迭代：进度条每消费一例后才推进，必须与校验交错（不能用 list(tqdm(...)) 预先跑完）
        records_iter: Iterable = list(manifest["cases"])
        if progress and tqdm is not None:
            records_iter = tqdm(
                list(manifest["cases"]), desc=desc, total=len(manifest["cases"]), mininterval=1.0
            )
        n_ok = 0
        failures: list[str] = []
        for record in records_iter:
            case_id = str(record["case_id"])
            try:
                validate_sidecar(
                    self.zonal_prior_root,
                    case_id,
                    expected_source=self.zonal_prior_source,
                    expected_configuration=self.expected_configuration,
                    expected_plans_sha256=self.expected_plans_sha256,
                    expected_input_sha256=record.get("input_sha256") or {},
                    expected_metadata_sha256=record.get("metadata_sha256"),
                    expected_output_sha256=record.get("output_sha256"),
                    verify_array_hash=verify_array_hash,
                )
            except ZonalPriorError as exc:
                failures.append(f"{case_id}: {exc}")
                continue
            n_ok += 1

        stats = {
            "n_cases": len(manifest["cases"]),
            "n_ok": n_ok,
            "n_failed": len(failures),
            "array_hash_checked": bool(verify_array_hash),
            "elapsed_sec": round(time.time() - t0, 2),
            "manifest_sha256": str(manifest["manifest_sha256"]),
        }
        self.zonal_prior_validation_stats = stats
        if failures:
            # 失败路径：READY 状态与冻结记录保持未就绪（已在校验前复位）
            raise PreprocessedStoreError(
                f"prior sidecar 深度校验失败：{len(failures)}/{len(manifest['cases'])} 例不通过"
                f"（array_hash_checked={bool(verify_array_hash)}）；前 5 条例：{failures[:5]}"
            )
        # 全部成功 → 一次性提交 READY 状态 + 不可变冻结记录
        frozen = {
            str(record["case_id"]): MappingProxyType(
                {
                    # 嵌套哈希同样冻结：冻结记录必须做到深层不可变
                    "input_sha256": MappingProxyType(dict(record.get("input_sha256") or {})),
                    "output_sha256": str(record["output_sha256"]),
                    "metadata_sha256": str(record["metadata_sha256"]),
                }
            )
            for record in manifest["cases"]
        }
        self.zonal_prior_frozen_records = MappingProxyType(frozen)
        self.zonal_prior_array_hash_checked = bool(verify_array_hash)
        self.zonal_prior_manifest_sha256 = str(manifest["manifest_sha256"])
        return copy.deepcopy(manifest)

    def _reset_zonal_prior_ready(self) -> None:
        """把 prior READY 状态复位为未就绪（冻结记录清空）。"""
        self.zonal_prior_manifest_sha256 = None
        self.zonal_prior_array_hash_checked = False
        self.zonal_prior_frozen_records = MappingProxyType({})

    def describe_zonal_prior(self) -> dict:
        """写入 run manifest / 快照的 prior 数据描述（含 plan/config 绑定与 manifest 哈希）。"""
        return {
            "enabled": self.zonal_prior_root is not None,
            "source": self.zonal_prior_source,
            "root": None if self.zonal_prior_root is None else str(self.zonal_prior_root),
            "configuration": self.expected_configuration,
            "plans_sha256": self.expected_plans_sha256,
            "manifest_sha256": self.zonal_prior_manifest_sha256,
            "array_hash_checked": self.zonal_prior_array_hash_checked,
            "frozen_records": len(self.zonal_prior_frozen_records),
            "require_frozen_manifest": bool(self.require_frozen_manifest),
            "channels": list(range(ZONAL_PRIOR_CHANNELS)),
            "channel_semantics": ["PZ", "TZ"],
            "wg_allowed": False,
        }

    def iter_cases(
        self,
        case_ids: Sequence[str] | None = None,
        *,
        progress: bool = True,
        desc: str = "preprocessed",
    ) -> Iterator[PreprocessedCase]:
        """按顺序遍历病例（默认带 tqdm 进度条；只读）。"""
        ids = list(case_ids) if case_ids is not None else list(self.available_ids)
        iterator: Iterable[str] = ids
        if progress and tqdm is not None:
            iterator = tqdm(ids, desc=desc, mininterval=1.0)
        for case_id in iterator:
            yield self.get_case(case_id)


def verify_case_contract(case: PreprocessedCase, *, expected_channels: int = EXPECTED_CHANNELS) -> dict:
    """对单个病例做契约检查，返回结构化结果（供 loader smoke 命令打印）。"""
    data, seg = case.data, case.seg
    labels = np.unique(seg) if seg.size else np.array([], dtype=np.uint8)
    return {
        "case_id": case.case_id,
        "data_shape": list(data.shape),
        "data_dtype": str(data.dtype),
        "seg_shape": list(seg.shape),
        "seg_dtype": str(seg.dtype),
        "seg_labels": [int(v) for v in labels],
        "channels_ok": data.shape[0] == expected_channels,
        "labels_binary": bool(np.all(np.isin(labels, [0, 1]))),
        "finite": bool(np.isfinite(data).all()),
        "spacing": list(case.properties.spacing) if case.properties.spacing else None,
        "has_class_locations": bool(case.properties.class_locations),
    }


__all__ = [
    "EXPECTED_CHANNELS",
    "FOREGROUND_LABEL",
    "SEG_SENTINEL_LABEL",
    "SEG_STORAGE_LABELS",
    "ZONAL_PRIOR_CHANNELS",
    "ZONAL_PRIOR_TOL",
    "CaseProperties",
    "PreprocessedCase",
    "PreprocessedStore",
    "PreprocessedStoreError",
    "normalize_data",
    "normalize_seg",
    "verify_case_contract",
]
