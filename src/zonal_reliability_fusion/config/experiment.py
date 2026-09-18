"""M0 实验配置加载与合法性检查。

配置文件（`configs/experiments/m0_picai_3d_fullres.yaml` 或 `.json`）中的每一项都必须
显式声明；加载时检查与冻结 plan 的冲突、模态顺序、fold 与输出目录位置。

本模块只读配置与 JSON（plan / split），不读取影像数据。
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .architecture import ArchitectureRef
from .plans import Plan3DConfig

MODALITY_ORDER: tuple[str, str, str] = ("T2W", "ADC", "HBV")
PROJECT_ROOT = Path(__file__).resolve().parents[3]

#: M0 尚未对齐 nnU-Net 的增强项（显式冻结，避免被误称为"已完全对齐"）
DEFAULT_PENDING_PARITY: tuple[str, ...] = (
    "rotation",
    "scaling",
    "low_resolution_simulation",
    "gaussian_noise",
    "blur",
    "gamma",
)


class ExperimentConfigError(ValueError):
    """实验配置缺失、类型非法或与 plan / 项目约定冲突。"""


# --------------------------------------------------------------------------- 子配置
@dataclass(frozen=True)
class ExperimentMeta:
    name: str
    model_id: str
    seed: int
    fold: int


@dataclass(frozen=True)
class PathsConfig:
    plans: str
    preprocessed_dataset: str
    splits: str
    output_root: str


@dataclass(frozen=True)
class ModelConfig:
    in_channels: int
    num_classes: int
    deep_supervision: bool
    configuration: str


@dataclass(frozen=True)
class DataConfig:
    modality_order: tuple[str, ...]
    batch_size: int
    patch_size: tuple[int, int, int]
    oversample_foreground: float
    num_workers: int
    pin_memory: bool
    #: M3/M4 必需：PZ/TZ prior 来源（yuan|hevi）与 plan-space sidecar 根目录；M0/M1/M2 必须为 None
    zonal_prior_source: str | None = None
    zonal_prior_root: str | None = None


@dataclass(frozen=True)
class AugmentationConfig:
    """增强配置（当前能力的**明确冻结**；未实现项列在 pending_parity）。

    背景：M0 目前只实现了"同步镜像"；nnU-Net 的 rotation/scaling/低分辨率模拟/noise/blur/gamma
    尚未实现。在正式 N0/M0 比较前必须补齐，或把差异作为明确记录的实验差异（见
    `docs/P2_M0_Implementation.md`）。`pending_parity` 就是这份差异清单，防止被遗忘或误称"已对齐"。
    """

    mirror: bool
    mirror_p_per_axis: float
    per_channel_intensity: bool
    pending_parity: tuple[str, ...]


@dataclass(frozen=True)
class LossConfig:
    """损失配置（research_plan §9.2）。

    ``name``：

    - ``"focal_ce"``：与正式 N0（`PI-CAI official-style nnU-Net v2 Focal+CE NoFFT`）对齐，
      ``0.5 × Focal(γ=2, α=None) + 0.5 × CE``，逐 deep-supervision 尺度计算；
    - ``"dice_ce"``：迁移前的自研实现（Dice + CE）；保留可用，但**与正式 N0 不可做正式比较**。

    ``batch_dice`` 恒取冻结 plan 的值（用于 plan 一致性校验），在 ``focal_ce`` 下不代表损失项。
    未使用的字段必须省略（``focal_ce`` 不得声明 ``dice_weight``/``include_background``，反之亦然），
    加载时会显式报错，避免两种口径被混用。
    """

    name: str
    ce_weight: float
    smooth: float
    batch_dice: bool
    focal_weight: float | None = None
    gamma: float | None = None
    dice_weight: float | None = None
    include_background: bool | None = None


@dataclass(frozen=True)
class OptimizerConfig:
    name: str
    lr: float
    momentum: float
    nesterov: bool
    weight_decay: float


@dataclass(frozen=True)
class SchedulerConfig:
    name: str
    power: float


@dataclass(frozen=True)
class TrainingConfig:
    """正式训练预算（M0–M4 必须完全一致）。"""

    max_epochs: int
    iterations_per_epoch: int
    diagnostic_validation_iterations: int
    amp: bool
    grad_clip: float
    checkpoint_every: int
    progress: bool

    @property
    def epochs(self) -> int:
        """向后兼容别名：`epochs` 即 max_epochs。"""
        return self.max_epochs


@dataclass(frozen=True)
class ValidationConfig:
    """验证协议配置（正式训练 = 每 `every_n_epochs` 个 epoch 全体积；诊断 = 每 epoch 随机 patch）。"""

    mode: str
    every_n_epochs: int
    checkpoint_metric: str
    maximize: bool
    expected_cases: int
    expected_positive_cases: int
    expected_negative_cases: int
    step_fraction: float
    gaussian: bool
    mirror_tta: bool
    sliding_window_batch_size: int
    inner_patch_progress: bool
    save_case_metrics: bool


@dataclass(frozen=True)
class EarlyStoppingConfigFile:
    """early stopping 配置（字段与 `training.EarlyStoppingConfig` 对应）。"""

    enabled: bool
    min_epochs: int
    patience: int
    min_delta: float


@dataclass(frozen=True)
class InferenceConfig:
    step_fraction: float
    gaussian: bool
    mirror_tta: bool


@dataclass(frozen=True)
class M0ExperimentConfig:
    experiment: ExperimentMeta
    paths: PathsConfig
    model: ModelConfig
    data: DataConfig
    augmentation: AugmentationConfig
    loss: LossConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    training: TrainingConfig
    validation: ValidationConfig
    early_stopping: EarlyStoppingConfigFile
    inference: InferenceConfig
    #: v2.3：版本化架构引用（缺失=None → legacy PlainConv M0；存在 → Residual-Encoder M0）。
    #: 严格附加字段：旧配置无此块，行为完全不变。
    architecture: ArchitectureRef | None = None
    source_path: str = ""

    # --------------------------------------------------------------- 导出
    def resolved_snapshot(self) -> dict:
        """可写入实验输出目录的完整 resolved 配置。"""
        snap = asdict(self)
        snap["source_path"] = self.source_path
        return snap

    def resolve_path(self, value: str) -> Path:
        """把配置中的路径解析为绝对路径（支持 $ZRF_PROJECT_ROOT 与相对项目根）。"""
        expanded = os.path.expandvars(os.path.expanduser(value))
        path = Path(expanded)
        return path if path.is_absolute() else (PROJECT_ROOT / path)


# --------------------------------------------------------------------------- 加载
def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExperimentConfigError(f"配置节 {name!r} 必须是映射，收到 {type(value).__name__}")
    return value


def _require_int(value: Any, name: str, minimum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ExperimentConfigError(f"{name} 必须为整数，收到 {value!r}")
    if minimum is not None and value < minimum:
        raise ExperimentConfigError(f"{name} 必须 >= {minimum}，收到 {value}")
    return value


def _require_number(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ExperimentConfigError(f"{name} 必须为数值，收到 {value!r}")
    return float(value)


def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ExperimentConfigError(f"{name} 必须为布尔值，收到 {value!r}")
    return value


def _optional_number(value: Any, name: str) -> float | None:
    """可省略的数值字段：缺省时返回 None（是否允许省略由 `validate_loss_config` 判定）。"""
    if value is None:
        return None
    return _require_number(value, name)


def _optional_bool(value: Any, name: str) -> bool | None:
    """可省略的布尔字段：缺省时返回 None（是否允许省略由 `validate_loss_config` 判定）。"""
    if value is None:
        return None
    return _require_bool(value, name)


def _require_str(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ExperimentConfigError(f"{name} 必须为非空字符串，收到 {value!r}")
    return value


def _parse_architecture_ref(value: Any) -> ArchitectureRef | None:
    """解析可选的 `architecture` 块（v2.3）；缺省返回 None（legacy PlainConv）。"""
    if value is None:
        return None
    amap = _require_mapping(value, "architecture")
    fusion_variant = amap.get("fusion_variant")
    return ArchitectureRef(
        config_path=_require_str(amap.get("config_path"), "architecture.config_path"),
        architecture_name=_require_str(amap.get("architecture_name"), "architecture.architecture_name"),
        architecture_version=_require_str(amap.get("architecture_version"), "architecture.architecture_version"),
        encoder_type=_require_str(amap.get("encoder_type"), "architecture.encoder_type"),
        fusion_variant=(
            None
            if fusion_variant is None
            else _require_str(fusion_variant, "architecture.fusion_variant")
        ),
    )


def _require_int_tuple3(value: Any, name: str) -> tuple[int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ExperimentConfigError(f"{name} 必须为长度 3 的列表，收到 {value!r}")
    out = tuple(_require_int(v, f"{name}[{i}]", minimum=1) for i, v in enumerate(value))
    return out  # type: ignore[return-value]


def load_experiment_config(path: str | Path) -> M0ExperimentConfig:
    """读取 YAML（若可用）或 JSON 配置并构建强类型 dataclass。"""
    path = Path(path)
    if not path.is_file():
        raise ExperimentConfigError(f"配置文件不存在: {path}")
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        try:
            import yaml  # PyYAML 为可选依赖
        except ImportError as exc:  # pragma: no cover - 环境已确认可用
            raise ExperimentConfigError(
                f"缺少 PyYAML，无法读取 {path}；请改用 JSON 配置或安装 pyyaml"
            ) from exc
        doc = yaml.safe_load(path.read_text())
    elif suffix == ".json":
        doc = json.loads(path.read_text())
    else:
        raise ExperimentConfigError(f"不支持的配置后缀 {suffix!r}（支持 .yaml/.yml/.json）")
    if not isinstance(doc, Mapping):
        raise ExperimentConfigError(f"{path} 顶层必须是映射")

    exp = _require_mapping(doc.get("experiment"), "experiment")
    paths = _require_mapping(doc.get("paths"), "paths")
    model = _require_mapping(doc.get("model"), "model")
    data = _require_mapping(doc.get("data"), "data")
    augmentation = _require_mapping(doc.get("augmentation") or {}, "augmentation")
    loss = _require_mapping(doc.get("loss"), "loss")
    optimizer = _require_mapping(doc.get("optimizer"), "optimizer")
    scheduler = _require_mapping(doc.get("scheduler"), "scheduler")
    training = _require_mapping(doc.get("training"), "training")
    validation = _require_mapping(doc.get("validation"), "validation")
    early_stopping = _require_mapping(doc.get("early_stopping") or {}, "early_stopping")
    inference = _require_mapping(doc.get("inference"), "inference")

    modality_order = data.get("modality_order")
    if not isinstance(modality_order, (list, tuple)):
        raise ExperimentConfigError(f"data.modality_order 必须是列表，收到 {modality_order!r}")

    cfg = M0ExperimentConfig(
        experiment=ExperimentMeta(
            name=_require_str(exp.get("name"), "experiment.name"),
            model_id=_require_str(exp.get("model_id"), "experiment.model_id"),
            seed=_require_int(exp.get("seed"), "experiment.seed", minimum=0),
            fold=_require_int(exp.get("fold"), "experiment.fold", minimum=0),
        ),
        paths=PathsConfig(
            plans=_require_str(paths.get("plans"), "paths.plans"),
            preprocessed_dataset=_require_str(paths.get("preprocessed_dataset"), "paths.preprocessed_dataset"),
            splits=_require_str(paths.get("splits"), "paths.splits"),
            output_root=_require_str(paths.get("output_root"), "paths.output_root"),
        ),
        model=ModelConfig(
            in_channels=_require_int(model.get("in_channels"), "model.in_channels", minimum=1),
            num_classes=_require_int(model.get("num_classes"), "model.num_classes", minimum=2),
            deep_supervision=_require_bool(model.get("deep_supervision"), "model.deep_supervision"),
            configuration=_require_str(model.get("configuration"), "model.configuration"),
        ),
        data=DataConfig(
            modality_order=tuple(str(m) for m in modality_order),
            batch_size=_require_int(data.get("batch_size"), "data.batch_size", minimum=1),
            patch_size=_require_int_tuple3(data.get("patch_size"), "data.patch_size"),
            oversample_foreground=_require_number(data.get("oversample_foreground"), "data.oversample_foreground"),
            num_workers=_require_int(data.get("num_workers"), "data.num_workers", minimum=0),
            pin_memory=_require_bool(data.get("pin_memory"), "data.pin_memory"),
            zonal_prior_source=_optional_str(data.get("zonal_prior_source"), "data.zonal_prior_source"),
            zonal_prior_root=_optional_str(data.get("zonal_prior_root"), "data.zonal_prior_root"),
        ),
        augmentation=AugmentationConfig(
            mirror=_require_bool(augmentation.get("mirror", True), "augmentation.mirror"),
            mirror_p_per_axis=_require_number(
                augmentation.get("mirror_p_per_axis", 0.5), "augmentation.mirror_p_per_axis"
            ),
            per_channel_intensity=_require_bool(
                augmentation.get("per_channel_intensity", False), "augmentation.per_channel_intensity"
            ),
            pending_parity=tuple(str(v) for v in (augmentation.get("pending_parity") or DEFAULT_PENDING_PARITY)),
        ),
        loss=LossConfig(
            name=_require_str(loss.get("name"), "loss.name"),
            ce_weight=_require_number(loss.get("ce_weight"), "loss.ce_weight"),
            smooth=_require_number(loss.get("smooth"), "loss.smooth"),
            batch_dice=_require_bool(loss.get("batch_dice"), "loss.batch_dice"),
            focal_weight=_optional_number(loss.get("focal_weight"), "loss.focal_weight"),
            gamma=_optional_number(loss.get("gamma"), "loss.gamma"),
            dice_weight=_optional_number(loss.get("dice_weight"), "loss.dice_weight"),
            include_background=_optional_bool(loss.get("include_background"), "loss.include_background"),
        ),
        optimizer=OptimizerConfig(
            name=_require_str(optimizer.get("name"), "optimizer.name"),
            lr=_require_number(optimizer.get("lr"), "optimizer.lr"),
            momentum=_require_number(optimizer.get("momentum"), "optimizer.momentum"),
            nesterov=_require_bool(optimizer.get("nesterov"), "optimizer.nesterov"),
            weight_decay=_require_number(optimizer.get("weight_decay"), "optimizer.weight_decay"),
        ),
        scheduler=SchedulerConfig(
            name=_require_str(scheduler.get("name"), "scheduler.name"),
            power=_require_number(scheduler.get("power"), "scheduler.power"),
        ),
        training=TrainingConfig(
            # `max_epochs` 为正式名称；`epochs` 作为旧配置的兼容别名
            max_epochs=_require_int(
                training.get("max_epochs", training.get("epochs")), "training.max_epochs", minimum=1
            ),
            iterations_per_epoch=_require_int(training.get("iterations_per_epoch"), "training.iterations_per_epoch", minimum=1),
            diagnostic_validation_iterations=_require_int(
                training.get("diagnostic_validation_iterations", training.get("validation_iterations")),
                "training.diagnostic_validation_iterations",
                minimum=1,
            ),
            amp=_require_bool(training.get("amp"), "training.amp"),
            grad_clip=_require_number(training.get("grad_clip"), "training.grad_clip"),
            checkpoint_every=_require_int(training.get("checkpoint_every"), "training.checkpoint_every", minimum=1),
            progress=_require_bool(training.get("progress"), "training.progress"),
        ),
        validation=ValidationConfig(
            mode=_require_str(validation.get("mode"), "validation.mode"),
            every_n_epochs=_require_int(validation.get("every_n_epochs"), "validation.every_n_epochs", minimum=1),
            checkpoint_metric=_require_str(validation.get("checkpoint_metric"), "validation.checkpoint_metric"),
            maximize=_require_bool(validation.get("maximize"), "validation.maximize"),
            expected_cases=_require_int(validation.get("expected_cases"), "validation.expected_cases", minimum=1),
            expected_positive_cases=_require_int(
                validation.get("expected_positive_cases"), "validation.expected_positive_cases", minimum=1
            ),
            expected_negative_cases=_require_int(
                validation.get("expected_negative_cases"), "validation.expected_negative_cases", minimum=1
            ),
            step_fraction=_require_number(validation.get("step_fraction"), "validation.step_fraction"),
            gaussian=_require_bool(validation.get("gaussian"), "validation.gaussian"),
            mirror_tta=_require_bool(validation.get("mirror_tta"), "validation.mirror_tta"),
            sliding_window_batch_size=_require_int(
                validation.get("sliding_window_batch_size"), "validation.sliding_window_batch_size", minimum=1
            ),
            inner_patch_progress=_require_bool(
                validation.get("inner_patch_progress"), "validation.inner_patch_progress"
            ),
            save_case_metrics=_require_bool(validation.get("save_case_metrics"), "validation.save_case_metrics"),
        ),
        early_stopping=EarlyStoppingConfigFile(
            enabled=_require_bool(early_stopping.get("enabled", False), "early_stopping.enabled"),
            min_epochs=_require_int(early_stopping.get("min_epochs", 1), "early_stopping.min_epochs", minimum=1),
            patience=_require_int(early_stopping.get("patience", 1), "early_stopping.patience", minimum=1),
            min_delta=_require_number(early_stopping.get("min_delta", 0.0), "early_stopping.min_delta"),
        ),
        inference=InferenceConfig(
            step_fraction=_require_number(inference.get("step_fraction"), "inference.step_fraction"),
            gaussian=_require_bool(inference.get("gaussian"), "inference.gaussian"),
            mirror_tta=_require_bool(inference.get("mirror_tta"), "inference.mirror_tta"),
        ),
        architecture=_parse_architecture_ref(doc.get("architecture")),
        source_path=str(path),
    )
    validate_validation_protocol(cfg)  # 协议规则与 validator 共用同一实现
    validate_loss_config(cfg)  # 损失口径的显式性与自洽性（focal_ce 与正式 N0 对齐）
    validate_model_data_contract(cfg)  # 模型编号 ↔ PZ/TZ prior 数据契约（M3/M4 必填，M0–M2 禁止）
    return cfg


def _optional_str(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _require_str(value, name)


#: PZ/TZ prior 的合法来源（必须显式声明，不得静默选择）
ZONAL_PRIOR_SOURCES: tuple[str, ...] = ("yuan", "hevi")
#: 需要 ZP/TZ prior 的模型编号（其余模型必须不声明任何 prior 配置）
ZONAL_PRIOR_MODEL_IDS: tuple[str, ...] = ("M3", "M4")


def validate_model_data_contract(cfg: M0ExperimentConfig) -> None:
    """模型编号 ↔ prior 数据契约（M3/M4 必须配 prior；M0/M1/M2 不得声明 prior，禁止静默混用）。

    同时强制 **Yuan 与 Hevi 不共用实验身份与输出目录**：声明的 source 必须出现在
    `experiment.name` 与 `paths.output_root` 中（否则报错）。
    """
    model_id = str(cfg.experiment.model_id)
    source = cfg.data.zonal_prior_source
    root = cfg.data.zonal_prior_root
    needs_prior = model_id in ZONAL_PRIOR_MODEL_IDS
    if needs_prior:
        if source is None or root is None:
            raise ExperimentConfigError(
                f"model_id={model_id} 必须显式声明 data.zonal_prior_source 与 data.zonal_prior_root"
                f"（合法来源 {list(ZONAL_PRIOR_SOURCES)}；不得静默选择或运行时推断）"
            )
        if source not in ZONAL_PRIOR_SOURCES:
            raise ExperimentConfigError(
                f"data.zonal_prior_source={source!r} 非法；必须是 {list(ZONAL_PRIOR_SOURCES)} 之一"
            )
        for field_name, value in (("experiment.name", cfg.experiment.name), ("paths.output_root", cfg.paths.output_root)):
            if source not in str(value):
                raise ExperimentConfigError(
                    f"{field_name}={value!r} 必须包含 prior 来源标记 {source!r}："
                    "Yuan 与 Hevi 不得共用同一实验身份或输出目录（请用 *_yuan / *_hevi 命名）"
                )
    else:
        if source is not None or root is not None:
            raise ExperimentConfigError(
                f"model_id={model_id} 不读取 PZ/TZ prior，但配置声明了 "
                f"data.zonal_prior_source={source!r} / data.zonal_prior_root={root!r}；"
                "M0/M1/M2 必须保持为 None（不得静默忽略或混用）"
            )


def validate_loss_config(cfg: M0ExperimentConfig) -> None:
    """校验 loss 配置的显式性与自洽性（research_plan §9.2）。

    目的：避免 ``focal_ce``（与正式 N0 对齐）与 ``dice_ce``（迁移前口径）被混用或误配，
    保证每个 run 的 resolved 快照都能明确回答「这次训练用的是哪种损失」。
    """
    loss = cfg.loss
    if loss.name == "focal_ce":
        if loss.focal_weight is None:
            raise ExperimentConfigError("loss.name=focal_ce 时必须提供 loss.focal_weight（推荐 0.5）")
        if loss.gamma is None:
            raise ExperimentConfigError("loss.name=focal_ce 时必须提供 loss.gamma（推荐 2.0）")
        if loss.dice_weight is not None:
            raise ExperimentConfigError("loss.name=focal_ce 不得声明 loss.dice_weight（两种口径不可混用）")
        if loss.include_background is not None:
            raise ExperimentConfigError(
                "loss.name=focal_ce 不得声明 loss.include_background（该字段仅用于 dice_ce）"
            )
    elif loss.name == "dice_ce":
        if loss.dice_weight is None:
            raise ExperimentConfigError("loss.name=dice_ce 时必须提供 loss.dice_weight")
        if loss.include_background is None:
            raise ExperimentConfigError("loss.name=dice_ce 时必须提供 loss.include_background")
        if loss.focal_weight is not None or loss.gamma is not None:
            raise ExperimentConfigError("loss.name=dice_ce 不得声明 loss.focal_weight/loss.gamma")
    else:
        raise ExperimentConfigError(
            f"未知 loss.name={loss.name!r}；仅支持 'focal_ce'（与正式 N0 对齐）与 'dice_ce'（迁移前口径）"
        )
    if loss.ce_weight <= 0:
        raise ExperimentConfigError(f"loss.ce_weight 必须为正，收到 {loss.ce_weight}")
    if not 0.0 <= loss.smooth < 1.0:
        raise ExperimentConfigError(f"loss.smooth 必须在 [0, 1) 内，收到 {loss.smooth}")


def validation_protocol_from_config(cfg: M0ExperimentConfig):
    """由实验配置构造 `ValidationProtocol`（延迟导入，保持 config 层不依赖 torch）。"""
    from ..evaluation.full_volume_validator import ValidationProtocol

    v = cfg.validation
    return ValidationProtocol(
        mode=v.mode,
        every_n_epochs=v.every_n_epochs,
        checkpoint_metric=v.checkpoint_metric,
        maximize=v.maximize,
        expected_cases=v.expected_cases,
        expected_positive_cases=v.expected_positive_cases,
        expected_negative_cases=v.expected_negative_cases,
        step_fraction=v.step_fraction,
        gaussian=v.gaussian,
        mirror_tta=v.mirror_tta,
        sliding_window_batch_size=v.sliding_window_batch_size,
        inner_patch_progress=v.inner_patch_progress,
        save_case_metrics=v.save_case_metrics,
    )


def validate_validation_protocol(cfg: M0ExperimentConfig) -> None:
    """验证协议规则（与 validator 共用同一份规则实现）。"""
    from ..evaluation.full_volume_validator import VALIDATION_MODES  # noqa: F401  (规则来源单一化)

    protocol = validation_protocol_from_config(cfg)
    try:
        protocol.validate()
    except ValueError as exc:
        raise ExperimentConfigError(f"validation 配置非法: {exc}") from exc
    es = cfg.early_stopping
    if es.enabled and int(es.min_epochs) > int(cfg.training.max_epochs):
        raise ExperimentConfigError(
            f"early_stopping.min_epochs={es.min_epochs} 不能大于 training.max_epochs={cfg.training.max_epochs}"
        )
    if es.enabled and float(es.min_delta) < 0:
        raise ExperimentConfigError(f"early_stopping.min_delta 必须 >= 0，收到 {es.min_delta}")


# --------------------------------------------------------------------------- 校验
def read_split_summary(path: str | Path) -> dict:
    """只读取 splits_final.json 的结构摘要（不读取任何病例数据）。"""
    path = Path(path)
    if not path.is_file():
        raise ExperimentConfigError(f"split 文件不存在: {path}")
    doc = json.loads(path.read_text())
    if not isinstance(doc, list) or not doc:
        raise ExperimentConfigError(f"{path} 必须是至少含 1 个 fold 的列表")
    for i, fold in enumerate(doc):
        if not isinstance(fold, Mapping) or "train" not in fold or "val" not in fold:
            raise ExperimentConfigError(f"{path} 的 fold {i} 缺少 train/val 键")
    return {
        "path": str(path),
        "n_folds": len(doc),
        "n_train": len(doc[0]["train"]),
        "n_val": len(doc[0]["val"]),
    }


def validate_against_plan(
    cfg: M0ExperimentConfig,
    plan: Plan3DConfig,
    *,
    project_root: Path | None = None,
    require_data: bool = True,
) -> list[str]:
    """检查配置与冻结 plan / 项目约定的冲突。

    返回 warning 列表；发现硬冲突时抛出 ExperimentConfigError。
    """
    project_root = Path(project_root) if project_root is not None else PROJECT_ROOT
    warnings: list[str] = []

    if tuple(cfg.data.modality_order) != MODALITY_ORDER:
        raise ExperimentConfigError(
            f"data.modality_order 必须严格为 {list(MODALITY_ORDER)}（channel 0/1/2），"
            f"收到 {list(cfg.data.modality_order)}"
        )
    if cfg.model.in_channels != len(MODALITY_ORDER):
        raise ExperimentConfigError(
            f"model.in_channels 必须为 {len(MODALITY_ORDER)}（T2W/ADC/HBV），收到 {cfg.model.in_channels}"
        )
    if cfg.model.num_classes != 2:
        raise ExperimentConfigError(f"model.num_classes 必须为 2（背景/病灶），收到 {cfg.model.num_classes}")
    if cfg.model.configuration != plan.configuration:
        raise ExperimentConfigError(
            f"model.configuration={cfg.model.configuration!r} 与 plan 的 {plan.configuration!r} 不一致"
        )
    if tuple(cfg.data.patch_size) != tuple(plan.patch_size):
        raise ExperimentConfigError(
            f"data.patch_size={tuple(cfg.data.patch_size)} 与 plan 的 {tuple(plan.patch_size)} 冲突"
        )
    if cfg.data.batch_size != plan.batch_size:
        raise ExperimentConfigError(
            f"data.batch_size={cfg.data.batch_size} 与 plan 的 {plan.batch_size} 冲突"
        )
    if cfg.loss.batch_dice != plan.batch_dice:
        raise ExperimentConfigError(
            f"loss.batch_dice={cfg.loss.batch_dice} 与 plan 的 batch_dice={plan.batch_dice} 不一致"
        )
    if cfg.experiment.fold != 0:
        raise ExperimentConfigError(f"本次训练只允许 fold=0（冻结单 fold 划分），收到 {cfg.experiment.fold}")
    if not (0.0 <= cfg.data.oversample_foreground <= 1.0):
        raise ExperimentConfigError(f"data.oversample_foreground 必须在 [0,1]，收到 {cfg.data.oversample_foreground}")
    if not (0.0 < cfg.inference.step_fraction <= 1.0):
        raise ExperimentConfigError(f"inference.step_fraction 必须在 (0,1]，收到 {cfg.inference.step_fraction}")
    if cfg.optimizer.name.upper() != "SGD":
        warnings.append(f"optimizer.name={cfg.optimizer.name!r} 不是 SGD（与 N0 对齐项不符）")
    if cfg.scheduler.name.upper() not in ("POLYLR", "POLY"):
        warnings.append(f"scheduler.name={cfg.scheduler.name!r} 不是 PolyLR（与 N0 对齐项不符）")
    if cfg.augmentation.pending_parity:
        warnings.append(
            "尚未对齐 nnU-Net 的增强：" + ", ".join(cfg.augmentation.pending_parity)
            + "（正式 N0/M0 比较前必须补齐或作为明确实验差异冻结）"
        )

    plans_path = cfg.resolve_path(cfg.paths.plans)
    if plans_path != Path(plan.source_path):
        warnings.append(f"配置内 plans 路径 {plans_path} 与已解析 plan 的 {plan.source_path} 不同")
    if not plans_path.is_file():
        raise ExperimentConfigError(f"paths.plans 不存在: {plans_path}")

    output_root = cfg.resolve_path(cfg.paths.output_root)
    if not str(output_root.resolve()).startswith(str(project_root.resolve())):
        raise ExperimentConfigError(
            f"paths.output_root={output_root} 必须位于项目内 {project_root}"
        )

    splits = read_split_summary(cfg.resolve_path(cfg.paths.splits))
    if splits["n_folds"] != 1:
        raise ExperimentConfigError(
            f"split 文件应只含单个 fold（冻结划分），实际 {splits['n_folds']} 个: {splits['path']}"
        )
    if splits["n_train"] < 1 or splits["n_val"] < 1:
        raise ExperimentConfigError(f"split 为空: {splits}")
    if cfg.validation.mode == "full_volume":
        if int(cfg.validation.expected_cases) != int(splits["n_val"]):
            raise ExperimentConfigError(
                f"validation.expected_cases={cfg.validation.expected_cases} 与冻结 split 的 val 数 "
                f"{splits['n_val']} 不一致；全体积验证必须覆盖 split 的全部 validation study"
            )
        # v2.2：正式 M0–M4 使用固定 200-epoch 预算，**禁用**“连续无改善”early stopping，
        # 以保证所有模型与 seed 获得相同训练预算；只保留 NaN/Inf/覆盖病例数错误等安全中止。
        # 若显式启用（例如工程调试），patience 必须按 validation event 计数（见 trainer）。
        if cfg.early_stopping.enabled:
            warnings.append(
                "full_volume 启用了 early_stopping：patience 按 validation event 计数；正式 M0–M4 协议"
                "要求禁用（enabled=false）以保证固定训练预算，请确认这是有意为之"
            )
    else:
        warnings.append(
            f"validation.mode={cfg.validation.mode!r} 为诊断模式（随机 patch），不得用于正式训练与模型比较"
        )
    warnings.append(
        f"checkpoint 指标 = {cfg.validation.checkpoint_metric}（maximize={cfg.validation.maximize}，"
        f"min_delta={cfg.early_stopping.min_delta}）；外部 test 不参与任何选择"
    )

    pre_dir = cfg.resolve_path(cfg.paths.preprocessed_dataset)
    if require_data and not pre_dir.is_dir():
        raise ExperimentConfigError(f"paths.preprocessed_dataset 不存在: {pre_dir}")
    if not require_data and not pre_dir.is_dir():
        warnings.append(f"preprocessed_dataset 尚不存在（require_data=False）: {pre_dir}")

    return warnings


__all__ = [
    "MODALITY_ORDER",
    "PROJECT_ROOT",
    "ExperimentConfigError",
    "M0ExperimentConfig",
    "load_experiment_config",
    "read_split_summary",
    "validate_against_plan",
]
