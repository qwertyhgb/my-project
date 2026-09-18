"""v2.3 版本化架构配置、稳定架构哈希与 §8.6 浅层 skip 接口契约。

本模块是 v2.3 plan-driven Residual-Encoder M0 的**结构身份来源**，只做配置解析、规范化
序列化、SHA-256 哈希与身份校验；**不依赖 torch / nnunetv2，不读取任何影像数据**。

设计要点（对齐 research_plan §7.2/§7.3/§8.6/§9.4.2）：

- 网络结构字段（每 stage block 数、features、kernel、stride、shortcut/初始化/decoder/deep
  supervision 身份、浅层 skip 契约、融合 stage）**全部显式冻结在版本化架构配置中**，不在代码里
  隐式推断；features/kernel/stride 必须与冻结 `3d_fullres` plan 一致（`validate_architecture_against_plan`）；
- **架构哈希**只对结构字段（`ArchitectureSpec.structural_dict`）做 sorted-key canonical JSON 的 SHA-256，
  因此不受 YAML 键顺序/无关空白影响；运行时间、绝对输出路径、plan 文件路径/文件哈希等**非结构字段**
  一律不进入架构哈希（它们另行冻结在 resolved config 的 `plan.*` 与 `provenance.*`）；
- legacy PlainConv 与 Residual-Encoder 的架构身份互斥，checkpoint/resume 必须核对
  `architecture_name / architecture_version / architecture_sha256`，不匹配默认拒绝（见
  `verify_architecture_identity`），禁止 `strict=False` 静默迁移；
- §8.6 浅层 skip 只冻结**接口契约**（输入边界、通道来源、拓扑、禁止读取 PZ/TZ/WG/gate/decoder、
  不产生第二套 modality logits、stage2=F_fuse、stage3+=共享 encoder skip），P2A **不实现** Q_l / F_fuse /
  M1–M4；契约以机器可读常量给出并由 `validate_shallow_skip_contract` 校验，防止被静默削弱。
"""
from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- 常量
#: encoder 身份
ENCODER_TYPE_RESIDUAL = "residual_encoder"
ENCODER_TYPE_PLAIN = "plain_conv"

#: residual block 身份（对齐 dynamic_network_architectures.BasicBlockD 的可验证行为）
BLOCK_TYPE_BASIC = "basic_post_activation"
#: 激活顺序：conv1(Conv→IN→LeakyReLU) → conv2(Conv→IN) → (+shortcut) → LeakyReLU
ACTIVATION_ORDER_POST = "post_activation"

#: §8.6 浅层 skip 接口契约 id（被架构配置引用；改变契约必须改变 id 与哈希）
SHALLOW_SKIP_CONTRACT_ID = "shallow_skip.v23.concat_conv1x1_in_leakyrelu"

#: §8.6 机器可读契约（P2A 只冻结接口，不实现聚合器）
SHALLOW_SKIP_CONTRACT: dict[str, Any] = {
    "contract_id": SHALLOW_SKIP_CONTRACT_ID,
    "applies_to_stages": [0, 1],
    "inputs": {
        "kind": "modality_specific_encoder_features",
        "count": 3,
        "modality_order": ["T2W", "ADC", "HBV"],
    },
    "target_channels_source": "plan.features_per_stage[stage]",
    "topology": ["concat", "conv1x1x1", "instance_norm3d", "leaky_relu"],
    "forbidden_inputs": ["pz_tz", "wg", "sequence_gate_weight", "sequence_logits", "decoder_feature"],
    "produces_second_modality_logits": False,
    "stage2_skip": "F_fuse",
    "stage3_and_deeper_skip": "shared_encoder_skip",
    "identical_across_m1_m4": True,
}

#: 主实验融合 stage（§8.5：encoder stage 2，累积 stride (1,4,4)）
FUSION_STAGE = 2

# --------------------------------------------------------------------------- 融合家族（M1–M4）
#: 版本化架构配置中承载 M1–M4 融合结构的节名；M0 配置**不含**该节 → M0 的架构哈希不变
FUSION_FAMILY_KEY = "fusion_family"

FUSION_VARIANT_M1_EQUAL = "m1_equal"
FUSION_VARIANT_M2_IMAGE_GATE = "m2_image_gate"
FUSION_VARIANT_M3_ZONE_INPUT = "m3_zone_input"
FUSION_VARIANT_M4_CONDITIONED_GATE = "m4_conditioned_gate"
FUSION_VARIANTS: tuple[str, ...] = (
    FUSION_VARIANT_M1_EQUAL,
    FUSION_VARIANT_M2_IMAGE_GATE,
    FUSION_VARIANT_M3_ZONE_INPUT,
    FUSION_VARIANT_M4_CONDITIONED_GATE,
)
#: 模型编号 ↔ 融合变体（双向严格映射；未知 model_id 必须报错，禁止回退到 M0）
MODEL_ID_TO_FUSION_VARIANT: dict[str, str] = {
    "M1": FUSION_VARIANT_M1_EQUAL,
    "M2": FUSION_VARIANT_M2_IMAGE_GATE,
    "M3": FUSION_VARIANT_M3_ZONE_INPUT,
    "M4": FUSION_VARIANT_M4_CONDITIONED_GATE,
}
FUSION_VARIANT_TO_MODEL_ID: dict[str, str] = {v: k for k, v in MODEL_ID_TO_FUSION_VARIANT.items()}

#: 模态顺序（channel 0/1/2 = T2W/ADC/HBV；M1–M4 与 M0 必须一致）
FUSION_MODALITY_ORDER: tuple[str, ...] = ("T2W", "ADC", "HBV")

# --- 冻结组件白名单：配置只能声明模型**唯一实现**的行为，声明未实现值在加载时报错 ---
FUSION_BRANCH_SUPPORTED: dict[str, Any] = {
    "stages": (0, 1, 2),
    "stem_type": "stacked_conv_blocks",
    "stem_n_convs": 1,
    "independent_parameters": True,
    "blocks_source": "backbone_blocks_per_stage",
}
FUSION_MODALITY_PROJECTION_SUPPORTED: dict[str, Any] = {
    "topology": ("conv1x1x1", "instance_norm3d", "leaky_relu"),
    "per_modality": True,
}
FUSION_SHALLOW_SKIP_SUPPORTED: dict[str, Any] = {
    "topology": ("concat", "conv1x1x1", "instance_norm3d", "leaky_relu"),
    "stages": (0, 1),
    "target_channels_source": "plan.features_per_stage[stage]",
    "reads_only": "modality_specific_stage_features",
}
FUSION_GATE_SUPPORTED: dict[str, Any] = {
    "type": "image_driven_softmax_over_modalities",
    "logits_source": "modality_projection_outputs",  # 输入必须是 U_m = P_m(F_m)，不是投影前特征
    "logits_topology": ("conv1x1x1", "instance_norm3d", "leaky_relu", "conv1x1x1"),
    "softmax_dim": "modality",
    "weight_sum_to_one": True,
    "last_layer_zero_init": True,
}
FUSION_CONDITIONING_SUPPORTED: dict[str, Any] = {
    "type": "bounded_conditional_affine",
    "formula": "(1 + tanh(gamma)) * L_img + beta",
    "gamma_activation": "tanh",
    "head_topology": ("conv1x1x1", "instance_norm3d", "leaky_relu", "conv1x1x1"),
    "head_last_layer_zero_init": True,
    "spatial": True,
    "enters_gate": True,
    "enters_shallow_skip": False,
    "enters_decoder_directly": False,
}
FUSION_ZONE_ENCODER_SUPPORTED: dict[str, Any] = {
    "input_channels": 2,
    "channel_semantics": ("pz", "tz"),
    "wg_forbidden": True,
    "n_blocks": 2,
    "features": 32,
    "kernel_size": (3, 3, 3),
    "adapt": "adaptive_avg_pool3d_to_fusion_stage",
    "stem_topology": ("conv3d_same_padding", "instance_norm3d", "leaky_relu"),
    "block_type": "residual_post_activation_basic",
    "block_topology": (
        "conv3d_same_padding",
        "instance_norm3d",
        "leaky_relu",
        "conv3d_same_padding",
        "instance_norm3d",
        "add_shortcut",
        "leaky_relu",
    ),
    "block_zero_init_last_norm": False,
    "shortcut": "identity_or_avgpool_then_1x1x1_projection",
    "shared_between_m3_m4": True,
}
FUSION_ZONE_INJECTION_SUPPORTED: dict[str, Any] = {
    # M3 严格实现 Q([F_fuse, E_z])：输出**替换**共享输入，不做 F_fuse + Q(...) 残差相加
    "type": "post_fusion_projection",
    "output": "Q([F_fuse,E_z])",
    "topology": ("concat", "conv1x1x1", "instance_norm3d", "leaky_relu"),
    "enters": "shared_feature_after_fusion",
    "implicit_residual": False,
    "enters_gate": False,
    "enters_shallow_skip": False,
}

#: 各变体必须出现的组件（presence 规则；缺失/多余一律拒绝）
FUSION_VARIANT_PRESENCE: dict[str, tuple[bool, bool, bool, bool]] = {
    # (gate, conditioning, zone_encoder, zone_injection)
    FUSION_VARIANT_M1_EQUAL: (False, False, False, False),
    FUSION_VARIANT_M2_IMAGE_GATE: (True, False, False, False),
    FUSION_VARIANT_M3_ZONE_INPUT: (True, False, True, True),
    FUSION_VARIANT_M4_CONDITIONED_GATE: (True, True, True, False),
}


class ArchitectureConfigError(ValueError):
    """架构配置缺失、类型非法或与冻结 plan / §8.6 契约冲突。"""


class ArchitectureIdentityError(ValueError):
    """checkpoint/resume 的架构身份缺失或不一致（含 legacy PlainConv → Residual 的拒绝）。"""


# --------------------------------------------------------------------------- 序列化 / 哈希
def _json_default(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"无法规范化序列化的类型: {type(value).__name__}")


def canonical_json(obj: Any) -> str:
    """sorted-key、紧凑分隔符的规范化 JSON（不受键顺序/无关空白影响）。"""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=_json_default)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def architecture_sha256(structural: Mapping[str, Any]) -> str:
    """对结构字段做 canonical JSON → SHA-256。"""
    return sha256_text(canonical_json(structural))


# --------------------------------------------------------------------------- 校验助手
def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ArchitectureConfigError(f"架构配置节 {name!r} 必须是映射，收到 {type(value).__name__}")
    return value


def _require_str(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ArchitectureConfigError(f"{name} 必须为非空字符串，收到 {value!r}")
    return value


def _require_int(value: Any, name: str, *, minimum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ArchitectureConfigError(f"{name} 必须为整数，收到 {value!r}")
    if minimum is not None and value < minimum:
        raise ArchitectureConfigError(f"{name} 必须 >= {minimum}，收到 {value}")
    return value


def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ArchitectureConfigError(f"{name} 必须为布尔值，收到 {value!r}")
    return value


def _require_number(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ArchitectureConfigError(f"{name} 必须为数值，收到 {value!r}")
    return float(value)


def _require_int_tuple(value: Any, name: str, length: int) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ArchitectureConfigError(f"{name} 必须为长度 {length} 的列表，收到 {value!r}")
    return tuple(_require_int(v, f"{name}[{i}]", minimum=1) for i, v in enumerate(value))


def _require_list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, (list, tuple)):
        raise ArchitectureConfigError(f"{name} 必须为列表，收到 {value!r}")
    return list(value)


def _require_str_tuple(value: Any, name: str, length: int) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ArchitectureConfigError(f"{name} 必须为长度 {length} 的列表，收到 {value!r}")
    return tuple(_require_str(v, f"{name}[{i}]") for i, v in enumerate(value))


def _require_vec3_tuple(value: Any, name: str, length: int) -> tuple[tuple[int, int, int], ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ArchitectureConfigError(f"{name} 必须为长度 {length} 的列表，收到 {value!r}")
    out = []
    for i, v in enumerate(value):
        t = _require_int_tuple(v, f"{name}[{i}]", 3)
        out.append((t[0], t[1], t[2]))
    return tuple(out)


def _freeze_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    """把配置映射转成可哈希的纯 dict（列表转 tuple 由 canonical_json 处理）。"""
    return json.loads(json.dumps(value, default=_json_default))


# --------------------------------------------------------------------------- 浅层 skip 契约
def validate_shallow_skip_contract(doc: Mapping[str, Any]) -> dict[str, Any]:
    """校验声明的 §8.6 契约与冻结契约一致（防止被静默削弱）。返回规范化契约 dict。"""
    frozen = _freeze_mapping(SHALLOW_SKIP_CONTRACT)
    given = _freeze_mapping(doc)
    if given.get("contract_id") != SHALLOW_SKIP_CONTRACT_ID:
        raise ArchitectureConfigError(
            f"shallow_skip_contract.contract_id={given.get('contract_id')!r} 与冻结契约 "
            f"{SHALLOW_SKIP_CONTRACT_ID!r} 不一致；改变 §8.6 契约必须升版并显式记录"
        )
    missing = [k for k in frozen if k not in given]
    if missing:
        raise ArchitectureConfigError(f"shallow_skip_contract 缺少字段: {missing}")
    diff = {k: {"frozen": frozen[k], "given": given[k]} for k in frozen if frozen[k] != given.get(k)}
    if diff:
        raise ArchitectureConfigError(f"shallow_skip_contract 与冻结 §8.6 契约不一致: {diff}")
    return given


def assert_shallow_skip_inputs_allowed(provided: Iterable[str]) -> None:
    """§8.6 运行时输入边界守卫（供未来 M1–M4 的 Q_l 聚合器调用；P2A 仅提供并测试）。

    `provided` 是聚合器实际读取的输入名集合；命中任何 forbidden_inputs 立即报错。
    """
    forbidden = set(SHALLOW_SKIP_CONTRACT["forbidden_inputs"])
    hit = sorted({str(p) for p in provided} & forbidden)
    if hit:
        raise ArchitectureConfigError(
            f"§8.6 浅层 skip 聚合器禁止读取以下输入: {hit}（Q_l 只能读取三路 modality-specific 特征，"
            "不得读取 PZ/TZ、WG、序列 gate 权重/ logits 或 decoder 特征，也不得产生第二套 modality logits）"
        )


# --------------------------------------------------------------------------- 架构 spec
@dataclass(frozen=True)
class ArchitectureSpec:
    """v2.3 版本化架构配置（结构身份 + 冻结规则）。

    `structural_dict()` 是**架构哈希的唯一输入**：只含定义网络模块/参数的结构字段；
    `to_dict()` 额外携带 provenance（plan 路径/文件哈希、配置源路径）用于可追溯，但**不进入哈希**。
    """

    architecture_name: str
    architecture_version: str
    encoder_type: str
    block_type: str
    activation_order: str
    n_stages: int
    blocks_per_stage: tuple[int, ...]
    features_per_stage: tuple[int, ...]
    kernel_sizes: tuple[tuple[int, int, int], ...]
    strides: tuple[tuple[int, int, int], ...]
    stem: dict[str, Any]
    shortcut: dict[str, Any]
    normalization: dict[str, Any]
    nonlinearity: dict[str, Any]
    conv_bias: bool
    initialization: dict[str, Any]
    decoder: dict[str, Any]
    deep_supervision: dict[str, Any]
    shallow_skip_contract: dict[str, Any]
    fusion_stage: int
    in_channels: int
    num_classes: int
    provenance: dict[str, Any] = field(default_factory=dict)
    source_path: str = ""
    #: M1–M4 融合家族原始声明（不进入哈希；仅用于可追溯与变体解析）
    fusion_family: dict[str, Any] | None = None
    #: 解析后的**单模型融合结构**（进入哈希；M0 为 None）
    fusion: dict[str, Any] | None = None

    # --------------------------------------------------------------- 结构 / 哈希
    @property
    def shallow_skip_contract_id(self) -> str:
        return str(self.shallow_skip_contract.get("contract_id", ""))

    @property
    def fusion_variant(self) -> str | None:
        """融合变体名（M0 为 None）。"""
        return None if self.fusion is None else str(self.fusion.get("variant"))

    @property
    def fusion_model_id(self) -> str | None:
        """融合模型编号（M1–M4；M0 为 None）。"""
        return None if self.fusion is None else str(self.fusion.get("model_id"))

    def structural_dict(self) -> dict[str, Any]:
        """架构哈希输入：只含结构字段（无路径、无时间、无 plan 文件哈希、无数据几何）。

        所有嵌套 dict 均返回**深拷贝**：`ArchitectureSpec` 是 frozen dataclass，不得因调用方修改
        返回的 dict 而连带改变内部状态与架构哈希（否则哈希不再稳定/不可信）。
        """
        return {
            "architecture_name": self.architecture_name,
            "architecture_version": self.architecture_version,
            "encoder_type": self.encoder_type,
            "block_type": self.block_type,
            "activation_order": self.activation_order,
            "n_stages": self.n_stages,
            "blocks_per_stage": list(self.blocks_per_stage),
            "features_per_stage": list(self.features_per_stage),
            "kernel_sizes": [list(k) for k in self.kernel_sizes],
            "strides": [list(s) for s in self.strides],
            "stem": copy.deepcopy(self.stem),
            "shortcut": copy.deepcopy(self.shortcut),
            "normalization": copy.deepcopy(self.normalization),
            "nonlinearity": copy.deepcopy(self.nonlinearity),
            "conv_bias": self.conv_bias,
            "initialization": copy.deepcopy(self.initialization),
            "decoder": copy.deepcopy(self.decoder),
            "deep_supervision": copy.deepcopy(self.deep_supervision),
            "shallow_skip_contract": copy.deepcopy(self.shallow_skip_contract),
            "fusion_stage": self.fusion_stage,
            "in_channels": self.in_channels,
            "num_classes": self.num_classes,
            **({"fusion": copy.deepcopy(self.fusion)} if self.fusion is not None else {}),
        }

    def architecture_sha256(self) -> str:
        return architecture_sha256(self.structural_dict())

    def identity(self) -> dict[str, Any]:
        """写入 resolved config / checkpoint 的架构身份三元组（+ encoder_type + 融合变体）。"""
        out = {
            "architecture_name": self.architecture_name,
            "architecture_version": self.architecture_version,
            "architecture_sha256": self.architecture_sha256(),
            "encoder_type": self.encoder_type,
        }
        if self.fusion is not None:
            out["fusion_variant"] = self.fusion_variant
            out["model_id"] = self.fusion_model_id
        return out

    def to_dict(self) -> dict[str, Any]:
        """完整可追溯快照（结构字段 + provenance + source_path + 计算出的哈希）。"""
        out = self.structural_dict()
        out["provenance"] = _freeze_mapping(self.provenance)
        out["source_path"] = self.source_path
        out["fusion_family"] = copy.deepcopy(self.fusion_family)
        out["architecture_sha256"] = self.architecture_sha256()
        return out


@dataclass(frozen=True)
class ArchitectureRef:
    """实验配置中的架构引用（指向版本化架构配置并断言期望身份，防止张冠李戴）。"""

    config_path: str
    architecture_name: str
    architecture_version: str
    encoder_type: str
    #: M1–M4 必须显式声明融合变体（M0/legacy 为 None）；变体名进入架构哈希
    fusion_variant: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_path": self.config_path,
            "architecture_name": self.architecture_name,
            "architecture_version": self.architecture_version,
            "encoder_type": self.encoder_type,
            "fusion_variant": self.fusion_variant,
        }


# --------------------------------------------------------------------------- 冻结取值白名单
# 版本化架构配置的规则字段只能声明**模型实际实现的唯一取值**；声明任何未实现的值（例如
# shortcut.projection_kernel=[3,3,3]、initialization.method=xavier、decoder.skip_merge=sum、
# deep_supervision.output_order=low_to_high）都会在**加载时**被拒绝，杜绝“改了配置但不生效”。
# 数值型字段（features/kernel/stride/n_conv_decoder/eps/negative_slope/conv_bias）不在此白名单，
# 它们由 `validate_architecture_against_plan` 与冻结 plan 逐值核对。
STEM_SUPPORTED: dict[str, Any] = {
    "type": "stacked_conv_blocks",
    "n_convs": 1,
    "stride": (1, 1, 1),
    "does_downsample": False,
}
# shortcut 真实语义（与 BasicBlockD 一致）：identity=同尺寸同通道；**下采样仅由 stride 触发**（AvgPool）；
# **projection（1×1×1 Conv+IN）仅由通道变化触发**；stride+通道同时变化时先 AvgPool 后 projection。
SHORTCUT_SUPPORTED: dict[str, Any] = {
    "identity_when": "same_spatial_and_channels",
    "downsample_when": "stride_change",
    "projection_when": "channel_change",
    "downsample_op": "avgpool3d_kernel_eq_stride",
    "downsample_before_projection": True,
    "projection_kernel": (1, 1, 1),
    "projection_conv_bias": False,
    "projection_norm": True,
    "implicit_channel_crop_forbidden": True,
}
NORMALIZATION_SUPPORTED: dict[str, Any] = {"op": "InstanceNorm3d"}
NONLINEARITY_SUPPORTED: dict[str, Any] = {"op": "LeakyReLU", "inplace": True}
INIT_SUPPORTED: dict[str, Any] = {
    "method": "he_kaiming_normal",
    "mode": "fan_in",
    "nonlinearity": "leaky_relu",
    "conv_bias_init": "zeros",
    "norm_weight_init": "ones",
    "norm_bias_init": "zeros",
}
DECODER_SUPPORTED: dict[str, Any] = {
    "type": "plain_unet_decoder",
    "upsample": "convtranspose3d",
    "transpconv_kernel_equals_stride": True,
    "skip_align": "center_crop_or_pad",
    "skip_merge": "concat",
    "stage_block": "stacked_conv_blocks",
    "per_level_seg_heads": True,
    "residual": False,
}
DEEP_SUPERVISION_SUPPORTED: dict[str, Any] = {
    "output_order": "high_to_low_resolution",
    "weight_scheme": "1_over_2_pow_i",
    "lowest_resolution_weight_zero": True,
    "weights_normalized_to_one": True,
    "single_output_weight": 1.0,
}


def _require_frozen(parsed: Mapping[str, Any], supported: Mapping[str, Any], name: str) -> None:
    """校验 parsed 中受支持的字段逐值等于模型实现的唯一取值；不符即拒绝（不允许声明未实现的行为）。"""
    diffs = {
        key: {"supported": supported[key], "given": parsed.get(key)}
        for key in supported
        if parsed.get(key) != supported[key]
    }
    if diffs:
        raise ArchitectureConfigError(
            f"{name} 含模型未实现/不支持的取值（架构行为由版本化配置冻结，只能声明唯一受支持值）: {diffs}"
        )


# --------------------------------------------------------------------------- 加载
def _parse_stem(doc: Mapping[str, Any]) -> dict[str, Any]:
    stem = _require_mapping(doc.get("stem"), "stem")
    parsed = {
        "type": _require_str(stem.get("type"), "stem.type"),
        "n_convs": _require_int(stem.get("n_convs"), "stem.n_convs", minimum=1),
        "stride": _require_int_tuple(stem.get("stride"), "stem.stride", 3),
        "does_downsample": _require_bool(stem.get("does_downsample"), "stem.does_downsample"),
    }
    _require_frozen(parsed, STEM_SUPPORTED, "stem")
    return parsed


def _parse_shortcut(doc: Mapping[str, Any]) -> dict[str, Any]:
    sc = _require_mapping(doc.get("shortcut"), "shortcut")
    parsed = {
        "identity_when": _require_str(sc.get("identity_when"), "shortcut.identity_when"),
        "downsample_when": _require_str(sc.get("downsample_when"), "shortcut.downsample_when"),
        "projection_when": _require_str(sc.get("projection_when"), "shortcut.projection_when"),
        "downsample_op": _require_str(sc.get("downsample_op"), "shortcut.downsample_op"),
        "downsample_before_projection": _require_bool(
            sc.get("downsample_before_projection"), "shortcut.downsample_before_projection"
        ),
        "projection_kernel": _require_int_tuple(sc.get("projection_kernel"), "shortcut.projection_kernel", 3),
        "projection_conv_bias": _require_bool(sc.get("projection_conv_bias"), "shortcut.projection_conv_bias"),
        "projection_norm": _require_bool(sc.get("projection_norm"), "shortcut.projection_norm"),
        "implicit_channel_crop_forbidden": _require_bool(
            sc.get("implicit_channel_crop_forbidden"), "shortcut.implicit_channel_crop_forbidden"
        ),
    }
    _require_frozen(parsed, SHORTCUT_SUPPORTED, "shortcut")
    return parsed


def _parse_normalization(doc: Mapping[str, Any]) -> dict[str, Any]:
    norm = _require_mapping(doc.get("normalization"), "normalization")
    parsed = {
        "op": _require_str(norm.get("op"), "normalization.op"),
        "eps": _require_number(norm.get("eps"), "normalization.eps"),
        "affine": _require_bool(norm.get("affine"), "normalization.affine"),
    }
    _require_frozen(parsed, NORMALIZATION_SUPPORTED, "normalization")
    return parsed


def _parse_nonlinearity(doc: Mapping[str, Any]) -> dict[str, Any]:
    nl = _require_mapping(doc.get("nonlinearity"), "nonlinearity")
    parsed = {
        "op": _require_str(nl.get("op"), "nonlinearity.op"),
        "negative_slope": _require_number(nl.get("negative_slope"), "nonlinearity.negative_slope"),
        "inplace": _require_bool(nl.get("inplace"), "nonlinearity.inplace"),
    }
    _require_frozen(parsed, NONLINEARITY_SUPPORTED, "nonlinearity")
    return parsed


def _parse_initialization(doc: Mapping[str, Any]) -> dict[str, Any]:
    init = _require_mapping(doc.get("initialization"), "initialization")
    parsed = {
        "method": _require_str(init.get("method"), "initialization.method"),
        "mode": _require_str(init.get("mode"), "initialization.mode"),
        "nonlinearity": _require_str(init.get("nonlinearity"), "initialization.nonlinearity"),
        "negative_slope": _require_number(init.get("negative_slope"), "initialization.negative_slope"),
        "conv_bias_init": _require_str(init.get("conv_bias_init"), "initialization.conv_bias_init"),
        "norm_weight_init": _require_str(init.get("norm_weight_init"), "initialization.norm_weight_init"),
        "norm_bias_init": _require_str(init.get("norm_bias_init"), "initialization.norm_bias_init"),
        "zero_init_last_norm_before_add": _require_bool(
            init.get("zero_init_last_norm_before_add"), "initialization.zero_init_last_norm_before_add"
        ),
    }
    _require_frozen(parsed, INIT_SUPPORTED, "initialization")
    return parsed


def _parse_decoder(doc: Mapping[str, Any], n_stages: int) -> dict[str, Any]:
    dec = _require_mapping(doc.get("decoder"), "decoder")
    n_conv_dec = dec.get("n_conv_per_stage_decoder")
    if not isinstance(n_conv_dec, (list, tuple)) or len(n_conv_dec) != n_stages - 1:
        raise ArchitectureConfigError(
            f"decoder.n_conv_per_stage_decoder 必须为长度 {n_stages - 1} 的列表，收到 {n_conv_dec!r}"
        )
    parsed = {
        "type": _require_str(dec.get("type"), "decoder.type"),
        "upsample": _require_str(dec.get("upsample"), "decoder.upsample"),
        "transpconv_kernel_equals_stride": _require_bool(
            dec.get("transpconv_kernel_equals_stride"), "decoder.transpconv_kernel_equals_stride"
        ),
        "skip_align": _require_str(dec.get("skip_align"), "decoder.skip_align"),
        "skip_merge": _require_str(dec.get("skip_merge"), "decoder.skip_merge"),
        "stage_block": _require_str(dec.get("stage_block"), "decoder.stage_block"),
        "n_conv_per_stage_decoder": [
            _require_int(v, f"decoder.n_conv_per_stage_decoder[{i}]", minimum=1) for i, v in enumerate(n_conv_dec)
        ],
        "per_level_seg_heads": _require_bool(dec.get("per_level_seg_heads"), "decoder.per_level_seg_heads"),
        "residual": _require_bool(dec.get("residual"), "decoder.residual"),
    }
    _require_frozen(parsed, DECODER_SUPPORTED, "decoder")
    return parsed


def _parse_deep_supervision(doc: Mapping[str, Any]) -> dict[str, Any]:
    ds = _require_mapping(doc.get("deep_supervision"), "deep_supervision")
    parsed = {
        "output_order": _require_str(ds.get("output_order"), "deep_supervision.output_order"),
        "weight_scheme": _require_str(ds.get("weight_scheme"), "deep_supervision.weight_scheme"),
        "lowest_resolution_weight_zero": _require_bool(
            ds.get("lowest_resolution_weight_zero"), "deep_supervision.lowest_resolution_weight_zero"
        ),
        "weights_normalized_to_one": _require_bool(
            ds.get("weights_normalized_to_one"), "deep_supervision.weights_normalized_to_one"
        ),
        "single_output_weight": _require_number(ds.get("single_output_weight"), "deep_supervision.single_output_weight"),
    }
    _require_frozen(parsed, DEEP_SUPERVISION_SUPPORTED, "deep_supervision")
    return parsed


def _parse_fusion_gate(doc: Mapping[str, Any]) -> dict[str, Any]:
    gate = _require_mapping(doc, "fusion_family.variants.*.gate")
    parsed = {
        "type": _require_str(gate.get("type"), "gate.type"),
        "logits_source": _require_str(gate.get("logits_source"), "gate.logits_source"),
        "logits_topology": tuple(
            _require_str(v, "gate.logits_topology[i]")
            for v in _require_list(gate.get("logits_topology"), "gate.logits_topology")
        ),
        "softmax_dim": _require_str(gate.get("softmax_dim"), "gate.softmax_dim"),
        "weight_sum_to_one": _require_bool(gate.get("weight_sum_to_one"), "gate.weight_sum_to_one"),
        "last_layer_zero_init": _require_bool(gate.get("last_layer_zero_init"), "gate.last_layer_zero_init"),
    }
    _require_frozen(parsed, FUSION_GATE_SUPPORTED, "gate")
    return parsed


def _parse_fusion_conditioning(doc: Mapping[str, Any]) -> dict[str, Any]:
    cond = _require_mapping(doc, "fusion_family.variants.*.conditioning")
    parsed = {
        "type": _require_str(cond.get("type"), "conditioning.type"),
        "formula": _require_str(cond.get("formula"), "conditioning.formula"),
        "gamma_activation": _require_str(cond.get("gamma_activation"), "conditioning.gamma_activation"),
        "head_topology": tuple(
            _require_str(v, "conditioning.head_topology[i]")
            for v in _require_list(cond.get("head_topology"), "conditioning.head_topology")
        ),
        "head_last_layer_zero_init": _require_bool(
            cond.get("head_last_layer_zero_init"), "conditioning.head_last_layer_zero_init"
        ),
        "spatial": _require_bool(cond.get("spatial"), "conditioning.spatial"),
        "enters_gate": _require_bool(cond.get("enters_gate"), "conditioning.enters_gate"),
        "enters_shallow_skip": _require_bool(
            cond.get("enters_shallow_skip"), "conditioning.enters_shallow_skip"
        ),
        "enters_decoder_directly": _require_bool(
            cond.get("enters_decoder_directly"), "conditioning.enters_decoder_directly"
        ),
    }
    _require_frozen(parsed, FUSION_CONDITIONING_SUPPORTED, "conditioning")
    return parsed


def _parse_fusion_zone_encoder(doc: Mapping[str, Any]) -> dict[str, Any]:
    zone = _require_mapping(doc, "fusion_family.variants.*.zone_encoder")
    parsed = {
        "input_channels": _require_int(zone.get("input_channels"), "zone_encoder.input_channels", minimum=1),
        "channel_semantics": _require_str_tuple(
            zone.get("channel_semantics"), "zone_encoder.channel_semantics", 2
        ),
        "wg_forbidden": _require_bool(zone.get("wg_forbidden"), "zone_encoder.wg_forbidden"),
        "n_blocks": _require_int(zone.get("n_blocks"), "zone_encoder.n_blocks", minimum=1),
        "features": _require_int(zone.get("features"), "zone_encoder.features", minimum=1),
        "kernel_size": _require_int_tuple(zone.get("kernel_size"), "zone_encoder.kernel_size", 3),
        "adapt": _require_str(zone.get("adapt"), "zone_encoder.adapt"),
        "stem_topology": tuple(
            _require_str(v, "zone_encoder.stem_topology[i]")
            for v in _require_list(zone.get("stem_topology"), "zone_encoder.stem_topology")
        ),
        "block_type": _require_str(zone.get("block_type"), "zone_encoder.block_type"),
        "block_topology": tuple(
            _require_str(v, "zone_encoder.block_topology[i]")
            for v in _require_list(zone.get("block_topology"), "zone_encoder.block_topology")
        ),
        "block_zero_init_last_norm": _require_bool(
            zone.get("block_zero_init_last_norm"), "zone_encoder.block_zero_init_last_norm"
        ),
        "shortcut": _require_str(zone.get("shortcut"), "zone_encoder.shortcut"),
        "shared_between_m3_m4": _require_bool(
            zone.get("shared_between_m3_m4"), "zone_encoder.shared_between_m3_m4"
        ),
    }
    _require_frozen(parsed, FUSION_ZONE_ENCODER_SUPPORTED, "zone_encoder")
    return parsed


def _parse_fusion_zone_injection(doc: Mapping[str, Any]) -> dict[str, Any]:
    inj = _require_mapping(doc, "fusion_family.variants.*.zone_injection")
    parsed = {
        "type": _require_str(inj.get("type"), "zone_injection.type"),
        "output": _require_str(inj.get("output"), "zone_injection.output"),
        "topology": tuple(
            _require_str(v, "zone_injection.topology[i]")
            for v in _require_list(inj.get("topology"), "zone_injection.topology")
        ),
        "enters": _require_str(inj.get("enters"), "zone_injection.enters"),
        "implicit_residual": _require_bool(inj.get("implicit_residual"), "zone_injection.implicit_residual"),
        "enters_gate": _require_bool(inj.get("enters_gate"), "zone_injection.enters_gate"),
        "enters_shallow_skip": _require_bool(inj.get("enters_shallow_skip"), "zone_injection.enters_shallow_skip"),
    }
    _require_frozen(parsed, FUSION_ZONE_INJECTION_SUPPORTED, "zone_injection")
    return parsed


def _parse_fusion_family(doc: Mapping[str, Any]) -> dict[str, Any] | None:
    """解析并严格校验 `fusion_family` 节（M1–M4 的融合结构身份来源）。

    M0 架构配置**没有**该节 → 返回 None（M0 的架构哈希保持不变）。
    任何未实现的声明（例如 gate 拓扑、zone encoder 宽度）都会在此报错，杜绝「改了配置但不生效」。
    """
    raw = doc.get(FUSION_FAMILY_KEY)
    if raw is None:
        return None
    fam = _require_mapping(raw, FUSION_FAMILY_KEY)
    family_id = _require_str(fam.get("family_id"), "fusion_family.family_id")
    modality_order = _require_str_tuple(fam.get("modality_order"), "fusion_family.modality_order", 3)
    if modality_order != FUSION_MODALITY_ORDER:
        raise ArchitectureConfigError(
            f"fusion_family.modality_order={list(modality_order)} 必须为 {list(FUSION_MODALITY_ORDER)}"
            "（channel 0/1/2 = T2W/ADC/HBV，与 M0 及冻结 plan 一致）"
        )

    branch = {
        "stages": tuple(
            _require_int(v, "fusion_family.shallow_branch.stages[i]", minimum=0)
            for v in _require_list(fam.get("shallow_branch", {}).get("stages"), "shallow_branch.stages")
        ),
        "stem_type": _require_str(
            _require_mapping(fam.get("shallow_branch"), "shallow_branch").get("stem_type"),
            "shallow_branch.stem_type",
        ),
        "stem_n_convs": _require_int(
            _require_mapping(fam.get("shallow_branch"), "shallow_branch").get("stem_n_convs"),
            "shallow_branch.stem_n_convs",
            minimum=1,
        ),
        "independent_parameters": _require_bool(
            _require_mapping(fam.get("shallow_branch"), "shallow_branch").get("independent_parameters"),
            "shallow_branch.independent_parameters",
        ),
        "blocks_source": _require_str(
            _require_mapping(fam.get("shallow_branch"), "shallow_branch").get("blocks_source"),
            "shallow_branch.blocks_source",
        ),
    }
    _require_frozen(branch, FUSION_BRANCH_SUPPORTED, "fusion_family.shallow_branch")

    proj_doc = _require_mapping(fam.get("modality_projection"), "fusion_family.modality_projection")
    projection = {
        "topology": tuple(
            _require_str(v, "modality_projection.topology[i]")
            for v in _require_list(proj_doc.get("topology"), "modality_projection.topology")
        ),
        "per_modality": _require_bool(proj_doc.get("per_modality"), "modality_projection.per_modality"),
    }
    _require_frozen(projection, FUSION_MODALITY_PROJECTION_SUPPORTED, "fusion_family.modality_projection")

    skip_doc = _require_mapping(fam.get("shallow_skip_aggregation"), "fusion_family.shallow_skip_aggregation")
    shallow_skip = {
        "topology": tuple(
            _require_str(v, "shallow_skip_aggregation.topology[i]")
            for v in _require_list(skip_doc.get("topology"), "shallow_skip_aggregation.topology")
        ),
        "stages": tuple(
            _require_int(v, "shallow_skip_aggregation.stages[i]", minimum=0)
            for v in _require_list(skip_doc.get("stages"), "shallow_skip_aggregation.stages")
        ),
        "target_channels_source": _require_str(
            skip_doc.get("target_channels_source"), "shallow_skip_aggregation.target_channels_source"
        ),
        "reads_only": _require_str(skip_doc.get("reads_only"), "shallow_skip_aggregation.reads_only"),
    }
    _require_frozen(shallow_skip, FUSION_SHALLOW_SKIP_SUPPORTED, "fusion_family.shallow_skip_aggregation")
    if tuple(branch["stages"]) != tuple(range(FUSION_STAGE + 1)):
        raise ArchitectureConfigError(
            f"fusion_family.shallow_branch.stages={list(branch['stages'])} 必须为 "
            f"{list(range(FUSION_STAGE + 1))}（三路分支编码到融合 stage {FUSION_STAGE}）"
        )

    variants_doc = _require_mapping(fam.get("variants"), "fusion_family.variants")
    declared = set(variants_doc.keys())
    expected = set(FUSION_VARIANTS)
    if declared != expected:
        raise ArchitectureConfigError(
            f"fusion_family.variants 必须精确声明 {sorted(expected)}；"
            f"缺失={sorted(expected - declared)}，多余={sorted(declared - expected)}"
        )

    variants: dict[str, dict[str, Any]] = {}
    for index, variant in enumerate(FUSION_VARIANTS):
        vdoc = _require_mapping(variants_doc.get(variant), f"fusion_family.variants.{variant}")
        model_id = _require_str(vdoc.get("model_id"), f"variants.{variant}.model_id")
        if model_id != FUSION_VARIANT_TO_MODEL_ID[variant]:
            raise ArchitectureConfigError(
                f"variants.{variant}.model_id={model_id!r} 必须为 {FUSION_VARIANT_TO_MODEL_ID[variant]!r}"
                "（模型编号与融合变体严格一一对应，防止张冠李戴）"
            )
        gate_p, cond_p, zone_p, inj_p = FUSION_VARIANT_PRESENCE[variant]
        gate_raw, cond_raw, zone_raw, inj_raw = (
            vdoc.get("gate"),
            vdoc.get("conditioning"),
            vdoc.get("zone_encoder"),
            vdoc.get("zone_injection"),
        )
        presence = {"gate": gate_raw is not None, "conditioning": cond_raw is not None,
                    "zone_encoder": zone_raw is not None, "zone_injection": inj_raw is not None}
        required = {"gate": gate_p, "conditioning": cond_p, "zone_encoder": zone_p, "zone_injection": inj_p}
        wrong = {k: {"declared": presence[k], "required": required[k]} for k in presence if presence[k] != required[k]}
        if wrong:
            raise ArchitectureConfigError(
                f"variants.{variant} 的组件存在性与冻结定义不符: {wrong}"
                f"（gate/conditioning/zone_encoder/zone_injection 的必选组合见 FUSION_VARIANT_PRESENCE）"
            )
        variants[variant] = {
            "model_id": model_id,
            "gate": _parse_fusion_gate(gate_raw) if gate_p else None,
            "conditioning": _parse_fusion_conditioning(cond_raw) if cond_p else None,
            "zone_encoder": _parse_fusion_zone_encoder(zone_raw) if zone_p else None,
            "zone_injection": _parse_fusion_zone_injection(inj_raw) if inj_p else None,
        }
        _ = index

    # 跨变体一致性（机器强制，而不是靠人工纪律）：
    # 1) M2/M3/M4 的 image gate 必须**完全相同**（M3/M4 的解剖对照才有意义）；
    gate_ref = variants[FUSION_VARIANT_M2_IMAGE_GATE]["gate"]
    for variant in (FUSION_VARIANT_M3_ZONE_INPUT, FUSION_VARIANT_M4_CONDITIONED_GATE):
        if variants[variant]["gate"] != gate_ref:
            raise ArchitectureConfigError(
                f"variants.{variant}.gate 必须与 {FUSION_VARIANT_M2_IMAGE_GATE}.gate 完全相同"
                "（M3/M4 只允许改变解剖信息的注入方式，不允许同时改变图像 gate）"
            )
    # 2) M3/M4 的 zone encoder 必须完全相同（同宽度/深度/融合尺度）
    if variants[FUSION_VARIANT_M3_ZONE_INPUT]["zone_encoder"] != variants[FUSION_VARIANT_M4_CONDITIONED_GATE]["zone_encoder"]:
        raise ArchitectureConfigError(
            "variants.m3_zone_input.zone_encoder 与 variants.m4_conditioned_gate.zone_encoder 必须完全相同"
            "（M3/M4 必须使用相同 zone encoder 宽度、深度与融合尺度）"
        )

    return {
        "family_id": family_id,
        "modality_order": list(modality_order),
        "shallow_branch": branch,
        "modality_projection": projection,
        "shallow_skip_aggregation": shallow_skip,
        "variants": variants,
    }


def resolve_fusion_spec(spec: ArchitectureSpec, variant: str) -> ArchitectureSpec:
    """把 `spec.fusion_family` 中声明的某个变体解析成**进入结构哈希**的 `spec.fusion` 块。

    M0 架构配置没有 fusion_family → 调用本函数报错；M1–M4 必须经此解析，其 architecture_sha256
    因此逐模型不同（checkpoint/resume 无法跨模型宽松加载）。
    """
    if spec.fusion_family is None:
        raise ArchitectureConfigError(
            "架构配置没有 fusion_family 节，无法解析 M1–M4 融合身份（M0 配置不含融合结构）"
        )
    if variant not in FUSION_VARIANTS:
        raise ArchitectureConfigError(f"未知 fusion_variant={variant!r}；允许值 {list(FUSION_VARIANTS)}")
    declared = spec.fusion_family["variants"].get(variant)
    if declared is None:
        raise ArchitectureConfigError(
            f"架构配置未声明变体 {variant!r}；已声明 {sorted(spec.fusion_family['variants'])}"
        )
    resolved = {
        "family_id": spec.fusion_family["family_id"],
        "variant": variant,
        "model_id": declared["model_id"],
        "stage": spec.fusion_stage,
        "modality_order": list(spec.fusion_family["modality_order"]),
        "shallow_branch": copy.deepcopy(spec.fusion_family["shallow_branch"]),
        "modality_projection": copy.deepcopy(spec.fusion_family["modality_projection"]),
        "shallow_skip_aggregation": copy.deepcopy(spec.fusion_family["shallow_skip_aggregation"]),
        "gate": copy.deepcopy(declared["gate"]),
        "conditioning": copy.deepcopy(declared["conditioning"]),
        "zone_encoder": copy.deepcopy(declared["zone_encoder"]),
        "zone_injection": copy.deepcopy(declared["zone_injection"]),
    }
    if resolved["stage"] != FUSION_STAGE:
        raise ArchitectureConfigError(f"fusion.stage={resolved['stage']} 必须为 {FUSION_STAGE}")
    return replace(spec, fusion=resolved)


def load_architecture_config(path: str | Path) -> ArchitectureSpec:
    """读取版本化架构配置（YAML/JSON），校验并构建 `ArchitectureSpec`（含稳定哈希）。"""
    path = Path(path)
    if not path.is_file():
        raise ArchitectureConfigError(f"架构配置文件不存在: {path}")
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - 环境已确认可用
            raise ArchitectureConfigError(f"缺少 PyYAML，无法读取 {path}") from exc
        doc = yaml.safe_load(path.read_text())
    elif suffix == ".json":
        doc = json.loads(path.read_text())
    else:
        raise ArchitectureConfigError(f"不支持的架构配置后缀 {suffix!r}（支持 .yaml/.yml/.json）")
    if not isinstance(doc, Mapping):
        raise ArchitectureConfigError(f"{path} 顶层必须是映射")
    return architecture_spec_from_mapping(doc, source_path=str(path))


def architecture_spec_from_mapping(doc: Mapping[str, Any], *, source_path: str = "") -> ArchitectureSpec:
    """从映射构建并校验 `ArchitectureSpec`（load 与测试共用同一实现）。"""
    blocks_per_stage_raw = doc.get("blocks_per_stage")
    features_raw = doc.get("features_per_stage")
    if not isinstance(features_raw, (list, tuple)):
        raise ArchitectureConfigError(f"features_per_stage 必须为列表，收到 {features_raw!r}")
    n_stages = _require_int(doc.get("n_stages", len(features_raw)), "n_stages", minimum=1)
    if len(features_raw) != n_stages:
        raise ArchitectureConfigError(
            f"features_per_stage 长度 {len(features_raw)} 与 n_stages={n_stages} 不一致"
        )
    blocks_per_stage = _require_int_tuple(blocks_per_stage_raw, "blocks_per_stage", n_stages)
    features_per_stage = _require_int_tuple(features_raw, "features_per_stage", n_stages)
    kernel_sizes = _require_vec3_tuple(doc.get("kernel_sizes"), "kernel_sizes", n_stages)
    strides = _require_vec3_tuple(doc.get("strides"), "strides", n_stages)

    encoder_type = _require_str(doc.get("encoder_type"), "encoder_type")
    if encoder_type != ENCODER_TYPE_RESIDUAL:
        raise ArchitectureConfigError(
            f"本模块只加载 residual 架构；encoder_type={encoder_type!r} != {ENCODER_TYPE_RESIDUAL!r}"
            "（legacy PlainConv 不由版本化架构配置描述）"
        )
    activation_order = _require_str(doc.get("activation_order"), "activation_order")
    if activation_order != ACTIVATION_ORDER_POST:
        raise ArchitectureConfigError(
            f"activation_order 必须显式冻结为 {ACTIVATION_ORDER_POST!r}，收到 {activation_order!r}"
        )
    block_type = _require_str(doc.get("block_type"), "block_type")
    if block_type != BLOCK_TYPE_BASIC:
        raise ArchitectureConfigError(f"block_type 必须为 {BLOCK_TYPE_BASIC!r}，收到 {block_type!r}")

    if strides[0] != (1, 1, 1):
        raise ArchitectureConfigError(f"strides[0]={strides[0]} 必须为 (1,1,1)（stem/stage0 不下采样）")
    for i, k in enumerate(kernel_sizes):
        if any(v % 2 == 0 for v in k):
            raise ArchitectureConfigError(f"kernel_sizes[{i}]={k} 含偶数核；仅支持 odd kernel + same padding")

    contract = validate_shallow_skip_contract(_require_mapping(doc.get("shallow_skip_contract"), "shallow_skip_contract"))
    fusion_stage = _require_int(doc.get("fusion_stage"), "fusion_stage", minimum=0)
    if fusion_stage != FUSION_STAGE:
        raise ArchitectureConfigError(f"fusion_stage 必须为 {FUSION_STAGE}（§8.5 encoder stage 2），收到 {fusion_stage}")
    if fusion_stage >= n_stages:
        raise ArchitectureConfigError(f"fusion_stage={fusion_stage} 超出 n_stages={n_stages}")

    spec = ArchitectureSpec(
        architecture_name=_require_str(doc.get("architecture_name"), "architecture_name"),
        architecture_version=_require_str(doc.get("architecture_version"), "architecture_version"),
        encoder_type=encoder_type,
        block_type=block_type,
        activation_order=activation_order,
        n_stages=n_stages,
        blocks_per_stage=blocks_per_stage,
        features_per_stage=features_per_stage,
        kernel_sizes=kernel_sizes,
        strides=strides,
        stem=_parse_stem(doc),
        shortcut=_parse_shortcut(doc),
        normalization=_parse_normalization(doc),
        nonlinearity=_parse_nonlinearity(doc),
        conv_bias=_require_bool(doc.get("conv_bias"), "conv_bias"),
        initialization=_parse_initialization(doc),
        decoder=_parse_decoder(doc, n_stages),
        deep_supervision=_parse_deep_supervision(doc),
        shallow_skip_contract=contract,
        fusion_stage=fusion_stage,
        in_channels=_require_int(doc.get("in_channels"), "in_channels", minimum=1),
        num_classes=_require_int(doc.get("num_classes"), "num_classes", minimum=2),
        provenance=_freeze_mapping(_require_mapping(doc.get("provenance") or {}, "provenance")),
        source_path=source_path,
        fusion_family=_parse_fusion_family(doc),
    )
    # 交叉一致性：initialization 与 nonlinearity 的 LeakyReLU 斜率必须自洽（He 初始化使用同一斜率）
    if abs(spec.initialization["negative_slope"] - spec.nonlinearity["negative_slope"]) > 1e-12:
        raise ArchitectureConfigError(
            f"initialization.negative_slope={spec.initialization['negative_slope']} 必须等于 "
            f"nonlinearity.negative_slope={spec.nonlinearity['negative_slope']}（He 初始化使用同一 LeakyReLU 斜率）"
        )
    # 以下为白名单已强制的冗余安全网（保留以给出更具体的报错）
    if spec.decoder["residual"]:
        raise ArchitectureConfigError("decoder.residual 必须为 false（§7.3：decoder 是普通 U-Net，非残差）")
    if spec.stem["does_downsample"] or tuple(spec.stem["stride"]) != (1, 1, 1):
        raise ArchitectureConfigError("stem 不得下采样（does_downsample=false 且 stride=(1,1,1)）")
    if not spec.shortcut["implicit_channel_crop_forbidden"]:
        raise ArchitectureConfigError("shortcut.implicit_channel_crop_forbidden 必须为 true（禁止隐式裁剪通道）")
    return spec


def validate_architecture_against_plan(spec: ArchitectureSpec, plan: Any) -> list[str]:
    """校验架构结构字段与冻结 plan 完全一致（features/kernel/stride/stage 数/算子/融合 stage）。

    返回 warning 列表；发现硬冲突时抛 `ArchitectureConfigError`。`plan` 为 `Plan3DConfig`。
    """
    warnings: list[str] = []
    if spec.n_stages != plan.n_stages:
        raise ArchitectureConfigError(f"architecture.n_stages={spec.n_stages} 与 plan 的 {plan.n_stages} 不一致")
    if tuple(spec.features_per_stage) != tuple(plan.features_per_stage):
        raise ArchitectureConfigError(
            f"architecture.features_per_stage={list(spec.features_per_stage)} 与冻结 plan "
            f"{list(plan.features_per_stage)} 不一致（features 必须来自 plan）"
        )
    if tuple(spec.kernel_sizes) != tuple(plan.kernel_sizes):
        raise ArchitectureConfigError(
            f"architecture.kernel_sizes={spec.kernel_sizes} 与冻结 plan {plan.kernel_sizes} 不一致"
        )
    if tuple(spec.strides) != tuple(plan.strides):
        raise ArchitectureConfigError(
            f"architecture.strides={spec.strides} 与冻结 plan {plan.strides} 不一致"
        )
    if list(spec.decoder["n_conv_per_stage_decoder"]) != list(plan.n_conv_per_stage_decoder):
        raise ArchitectureConfigError(
            f"decoder.n_conv_per_stage_decoder={list(spec.decoder['n_conv_per_stage_decoder'])} 与冻结 plan "
            f"{list(plan.n_conv_per_stage_decoder)} 不一致（decoder 是共同骨干，沿用 plan）"
        )
    if spec.normalization["op"] != "InstanceNorm3d":
        raise ArchitectureConfigError(f"normalization.op 必须为 InstanceNorm3d，收到 {spec.normalization['op']!r}")
    if abs(spec.normalization["eps"] - float(plan.norm_eps)) > 1e-12:
        raise ArchitectureConfigError(
            f"normalization.eps={spec.normalization['eps']} 与 plan 的 {plan.norm_eps} 不一致"
        )
    if spec.normalization["affine"] != bool(plan.norm_affine):
        raise ArchitectureConfigError(
            f"normalization.affine={spec.normalization['affine']} 与 plan 的 {plan.norm_affine} 不一致"
        )
    if spec.nonlinearity["op"] != "LeakyReLU":
        raise ArchitectureConfigError(f"nonlinearity.op 必须为 LeakyReLU，收到 {spec.nonlinearity['op']!r}")
    if abs(spec.nonlinearity["negative_slope"] - float(plan.negative_slope)) > 1e-12:
        raise ArchitectureConfigError(
            f"nonlinearity.negative_slope={spec.nonlinearity['negative_slope']} 与 plan 的 "
            f"{plan.negative_slope} 不一致"
        )
    if spec.conv_bias != bool(plan.conv_bias):
        raise ArchitectureConfigError(f"conv_bias={spec.conv_bias} 与 plan 的 {plan.conv_bias} 不一致")
    if spec.fusion_stage >= spec.n_stages:
        raise ArchitectureConfigError(f"fusion_stage={spec.fusion_stage} 超出 n_stages={spec.n_stages}")
    warnings.append(
        "架构结构字段与冻结 plan 一致（features/kernel/stride/n_conv_decoder/norm/nonlin/conv_bias）；"
        f"blocks_per_stage={list(spec.blocks_per_stage)}（来自版本化架构配置，不来自 plan）"
    )
    return warnings


def verify_architecture_identity(expected: Mapping[str, Any], found: Mapping[str, Any] | None) -> None:
    """核对 checkpoint/resume 的架构身份；缺失或不一致抛 `ArchitectureIdentityError`（含 legacy 检测）。

    `expected` / `found` 至少含 `architecture_name / architecture_version / architecture_sha256`。
    """
    if not found:
        raise ArchitectureIdentityError(
            "拒绝加载：checkpoint 未记录架构身份（architecture_name/version/sha256 缺失）。"
            "这通常是 legacy PlainConv M0 checkpoint，不得加载到 v2.3 Residual-Encoder M0；"
            "禁止用 strict=False 静默迁移。"
        )
    name = found.get("architecture_name")
    if name is None:
        raise ArchitectureIdentityError(
            "拒绝加载：checkpoint 缺少 architecture_name（疑似 legacy PlainConv checkpoint）；"
            f"期望 Residual-Encoder 身份 {dict(expected)}"
        )
    diffs = {
        key: {"expected": expected.get(key), "checkpoint": found.get(key)}
        for key in ("architecture_name", "architecture_version", "architecture_sha256", "encoder_type")
        if key in expected and expected.get(key) != found.get(key)
    }
    if diffs:
        raise ArchitectureIdentityError(
            f"拒绝续训：架构身份不一致 {diffs}；架构哈希不匹配默认拒绝 resume，"
            "确需改变架构请升版并新建独立 run/输出目录（不得 strict=False 绕过）。"
        )


__all__ = [
    "ACTIVATION_ORDER_POST",
    "BLOCK_TYPE_BASIC",
    "ENCODER_TYPE_PLAIN",
    "ENCODER_TYPE_RESIDUAL",
    "FUSION_BRANCH_SUPPORTED",
    "FUSION_CONDITIONING_SUPPORTED",
    "FUSION_FAMILY_KEY",
    "FUSION_GATE_SUPPORTED",
    "FUSION_MODALITY_ORDER",
    "FUSION_MODALITY_PROJECTION_SUPPORTED",
    "FUSION_SHALLOW_SKIP_SUPPORTED",
    "FUSION_STAGE",
    "FUSION_VARIANTS",
    "FUSION_VARIANT_M1_EQUAL",
    "FUSION_VARIANT_M2_IMAGE_GATE",
    "FUSION_VARIANT_M3_ZONE_INPUT",
    "FUSION_VARIANT_M4_CONDITIONED_GATE",
    "FUSION_VARIANT_PRESENCE",
    "FUSION_VARIANT_TO_MODEL_ID",
    "FUSION_ZONE_ENCODER_SUPPORTED",
    "FUSION_ZONE_INJECTION_SUPPORTED",
    "MODEL_ID_TO_FUSION_VARIANT",
    "SHALLOW_SKIP_CONTRACT",
    "SHALLOW_SKIP_CONTRACT_ID",
    "ArchitectureConfigError",
    "ArchitectureIdentityError",
    "ArchitectureRef",
    "ArchitectureSpec",
    "architecture_sha256",
    "architecture_spec_from_mapping",
    "assert_shallow_skip_inputs_allowed",
    "canonical_json",
    "load_architecture_config",
    "resolve_fusion_spec",
    "sha256_text",
    "validate_architecture_against_plan",
    "validate_shallow_skip_contract",
    "verify_architecture_identity",
]
