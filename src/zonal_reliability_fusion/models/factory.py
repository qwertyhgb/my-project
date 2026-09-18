"""模型工厂：按 `experiment.model_id` 明确构建 M0–M4，杜绝身份混淆与静默回退。

分派规则（任务书六）：

- `model_id: M0` + **无** `architecture` 块 → legacy `M0ConcatModel`（PlainConv，v2.2，保持旧配置行为不变）；
- `model_id: M0` + **有** `architecture` 块 → `M0ResidualConcatModel`（v2.3 Residual-Encoder）；
- `model_id: M1…M4` → 必须同时声明 `architecture.fusion_variant` 且与模型编号一一对应，
  经 `resolve_fusion_spec` 解析出该模型的融合结构后构建对应模型；
- **未知 model_id / 变体缺失 / 变体与 model_id 不符 → 一律报错**，禁止回退到 M0。

`model_architecture_identity` / `model_identity` 供 checkpoint/resume 身份隔离与日志使用。
"""
from __future__ import annotations

from typing import Any

from ..config.architecture import (
    ENCODER_TYPE_RESIDUAL,
    FUSION_VARIANT_TO_MODEL_ID,
    MODEL_ID_TO_FUSION_VARIANT,
    ArchitectureConfigError,
    ArchitectureRef,
    ArchitectureSpec,
    load_architecture_config,
    resolve_fusion_spec,
    validate_architecture_against_plan,
)
from ..config.plans import Plan3DConfig
from .fusion_multi_branch import (
    FUSION_MODEL_IDS,
    M1EqualFusionModel,
    M2ImageGateFusionModel,
    M3ZoneInputFusionModel,
    M4ConditionedGateFusionModel,
)
from .m0_concat import M0ConcatModel
from .m0_residual_concat import M0ResidualConcatModel

#: 支持的模型编号（M0 legacy/residual + M1–M4 融合）
SUPPORTED_MODEL_IDS: tuple[str, ...] = ("M0", *FUSION_MODEL_IDS)

_FUSION_MODEL_CLASSES: dict[str, type] = {
    "M1": M1EqualFusionModel,
    "M2": M2ImageGateFusionModel,
    "M3": M3ZoneInputFusionModel,
    "M4": M4ConditionedGateFusionModel,
}


def verify_ref_matches_spec(ref: ArchitectureRef, spec: ArchitectureSpec) -> None:
    """实验配置声明的架构身份必须与加载到的版本化架构配置一致（防止张冠李戴）。"""
    diffs = {
        field: {"experiment": getattr(ref, field), "architecture_config": getattr(spec, field)}
        for field in ("architecture_name", "architecture_version", "encoder_type")
        if getattr(ref, field) != getattr(spec, field)
    }
    if diffs:
        raise ArchitectureConfigError(
            f"实验配置声明的架构身份与架构配置文件不一致：{diffs}；"
            "请核对 experiment.architecture 与 configs/architectures/*.yaml"
        )
    if ref.fusion_variant is not None and spec.fusion_family is None:
        raise ArchitectureConfigError(
            f"实验配置声明 architecture.fusion_variant={ref.fusion_variant!r}，"
            "但架构配置没有 fusion_family 节（M0 架构配置不能用于 M1–M4）"
        )


def resolve_architecture_spec(
    cfg: Any, plan: Plan3DConfig, *, fusion_variant: str | None = None
) -> ArchitectureSpec:
    """加载并校验实验配置引用的 `ArchitectureSpec`（含与冻结 plan 的一致性）。

    `fusion_variant` 非空时，进一步解析该变体的融合结构（进入架构哈希）。
    """
    ref = getattr(cfg, "architecture", None)
    if ref is None:
        raise ArchitectureConfigError("实验配置缺少 architecture 块，无法解析 residual 架构 spec")
    spec = load_architecture_config(cfg.resolve_path(ref.config_path))
    verify_ref_matches_spec(ref, spec)
    if spec.encoder_type != ENCODER_TYPE_RESIDUAL:
        raise ArchitectureConfigError(
            f"architecture.encoder_type={spec.encoder_type!r} != {ENCODER_TYPE_RESIDUAL!r}；"
            "新配置不得回退到 PlainConv"
        )
    validate_architecture_against_plan(spec, plan)
    variant = fusion_variant if fusion_variant is not None else ref.fusion_variant
    if variant is not None:
        spec = resolve_fusion_spec(spec, variant)
    return spec


def build_m0_model(cfg: Any, plan: Plan3DConfig):
    """按配置构建 M0：legacy PlainConv（无 architecture 块）或 v2.3 Residual-Encoder（有 architecture 块）。"""
    ref = getattr(cfg, "architecture", None)
    if ref is None:
        return M0ConcatModel(
            plan,
            in_channels=cfg.model.in_channels,
            num_classes=cfg.model.num_classes,
            deep_supervision=cfg.model.deep_supervision,
        )
    if getattr(ref, "fusion_variant", None) is not None:
        raise ArchitectureConfigError(
            f"model_id=M0 不得声明 architecture.fusion_variant={ref.fusion_variant!r}"
            "（融合变体只属于 M1–M4）"
        )
    spec = resolve_architecture_spec(cfg, plan)
    return M0ResidualConcatModel(
        plan,
        spec,
        in_channels=cfg.model.in_channels,
        num_classes=cfg.model.num_classes,
        deep_supervision=cfg.model.deep_supervision,
    )


def build_model(cfg: Any, plan: Plan3DConfig):
    """按 `experiment.model_id` 构建 M0–M4；未知 model_id 或身份不符一律报错（禁止回退 M0）。"""
    model_id = str(getattr(cfg.experiment, "model_id", ""))
    if model_id == "M0":
        return build_m0_model(cfg, plan)
    variant = MODEL_ID_TO_FUSION_VARIANT.get(model_id)
    if variant is None:
        raise ArchitectureConfigError(
            f"未知 experiment.model_id={model_id!r}；支持 {list(SUPPORTED_MODEL_IDS)}"
            "（未知 model_id 必须报错，禁止回退到 M0）"
        )
    ref = getattr(cfg, "architecture", None)
    if ref is None:
        raise ArchitectureConfigError(
            f"model_id={model_id} 需要 architecture 块，并显式声明 fusion_variant: {variant!r}"
        )
    if ref.fusion_variant is None:
        raise ArchitectureConfigError(
            f"model_id={model_id} 的 architecture 块必须显式声明 fusion_variant: {variant!r}"
            "（缺失时禁止推断）"
        )
    if ref.fusion_variant != variant:
        raise ArchitectureConfigError(
            f"model_id={model_id} 与 architecture.fusion_variant={ref.fusion_variant!r} 不一致，"
            f"必须为 {variant!r}（模型编号与融合变体严格一一对应）"
        )
    spec = resolve_architecture_spec(cfg, plan, fusion_variant=variant)
    if spec.fusion_variant != variant:
        raise ArchitectureConfigError(
            f"解析后的融合变体 {spec.fusion_variant!r} != {variant!r}（架构配置与实验配置不一致）"
        )
    if spec.fusion_model_id != model_id:
        raise ArchitectureConfigError(
            f"架构配置声明的 fusion.model_id={spec.fusion_model_id!r} != experiment.model_id={model_id!r}"
        )
    model_cls = _FUSION_MODEL_CLASSES[model_id]
    return model_cls(
        plan,
        spec,
        in_channels=cfg.model.in_channels,
        num_classes=cfg.model.num_classes,
        deep_supervision=cfg.model.deep_supervision,
    )


def model_architecture_identity(model: Any) -> dict | None:
    """返回模型的架构身份；legacy PlainConv（无 `architecture_identity`）返回 None。"""
    fn = getattr(model, "architecture_identity", None)
    if fn is None:
        return None
    return dict(fn())


def model_identity(model: Any) -> dict | None:
    """返回完整模型身份（model_id/backbone/融合结构/架构哈希）；legacy 返回 None。"""
    fn = getattr(model, "model_identity", None)
    if fn is None:
        return None
    return dict(fn())


__all__ = [
    "FUSION_VARIANT_TO_MODEL_ID",
    "SUPPORTED_MODEL_IDS",
    "build_m0_model",
    "build_model",
    "model_architecture_identity",
    "model_identity",
    "resolve_architecture_spec",
    "verify_ref_matches_spec",
]
