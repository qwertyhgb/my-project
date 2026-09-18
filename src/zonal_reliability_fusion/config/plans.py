"""nnU-Net plan 解析（只读，plan-driven 配置来源）。

自研 M0–M4 的网络结构参数必须全部来自冻结的 plan 文件：

    workdir/nnUNet_preprocessed/Dataset605_PICAI/nnUNetPlans.json
    plans_name    = nnUNetPlans
    configuration = 3d_fullres

本模块只做 JSON 字段读取、校验与形状推导，**不读取任何影像数据**（.b2nd / .nii.gz），
也**不修改** plan 文件。解析后的 resolved 快照会随实验输出一起保存，保证可追溯。

术语：
- stage shape：某一级 encoder 输出在 (D, H, W) 上的空间尺寸；
- decoder 输出：从最高分辨率到最低分辨率排列（与 deep supervision 顺序一致）。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, Tuple

# 自研网络固定实现：Conv3d + InstanceNorm3d + LeakyReLU（与冻结 plan 的 arch_kwargs 对齐）
CONV_OP_EXPECTED = "torch.nn.modules.conv.Conv3d"
NORM_OP_EXPECTED = "torch.nn.modules.instancenorm.InstanceNorm3d"
NONLIN_EXPECTED = "torch.nn.LeakyReLU"

DEFAULT_NORM_EPS = 1e-5
DEFAULT_LEAKY_SLOPE = 0.01


class PlanConfigError(ValueError):
    """plan 文件缺失、结构非法或与自研网络假设不一致。"""


def _conv_output_size(in_size: int, kernel: int, stride: int) -> int:
    """odd kernel + same padding 的输出尺寸：floor((in - 1) / stride) + 1。"""
    if in_size < 1:
        raise PlanConfigError(f"输入尺寸必须 >= 1，收到 {in_size}")
    return (in_size - 1) // stride + 1


def _require_positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise PlanConfigError(f"{name} 必须为正整数，收到 {value!r}")
    return value


def _as_int_tuple(value: Any, name: str, length: int | None = None) -> Tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise PlanConfigError(f"{name} 必须是列表/元组，收到 {type(value).__name__}")
    out = tuple(_require_positive_int(v, name) for v in value)
    if length is not None and len(out) != length:
        raise PlanConfigError(f"{name} 长度必须为 {length}，收到 {len(out)}")
    return out


def _as_vector_tuple(value: Any, name: str, length: int) -> Tuple[Tuple[int, ...], ...]:
    if not isinstance(value, (list, tuple)):
        raise PlanConfigError(f"{name} 必须是列表/元组，收到 {type(value).__name__}")
    out = tuple(_as_int_tuple(v, f"{name}[{i}]", length) for i, v in enumerate(value))
    return out


@dataclass(frozen=True)
class Plan3DConfig:
    """从 nnUNetPlans.json 解析出的、经过校验的 3D 配置快照。"""

    dataset_name: str
    plans_name: str
    configuration: str
    data_identifier: str
    spacing: Tuple[float, float, float]
    patch_size: Tuple[int, int, int]
    batch_size: int
    features_per_stage: Tuple[int, ...]
    kernel_sizes: Tuple[Tuple[int, int, int], ...]
    strides: Tuple[Tuple[int, int, int], ...]
    n_conv_per_stage: Tuple[int, ...]
    n_conv_per_stage_decoder: Tuple[int, ...]
    conv_bias: bool
    norm_eps: float
    norm_affine: bool
    negative_slope: float
    batch_dice: bool
    source_path: str
    raw: Mapping[str, Any] = field(repr=False, compare=False, default_factory=dict)

    # ------------------------------------------------------------------ 属性
    @property
    def n_stages(self) -> int:
        return len(self.features_per_stage)

    @property
    def n_decoder_stages(self) -> int:
        return self.n_stages - 1

    @property
    def spatial_dims(self) -> int:
        return len(self.patch_size)

    # ------------------------------------------------------------- 形状推导
    def stage_shapes(self) -> Tuple[Tuple[int, int, int], ...]:
        """各级 encoder 输出的空间尺寸（含 stage 0，即 patch 本身）。"""
        shapes = []
        current = self.patch_size
        for i in range(self.n_stages):
            current = tuple(
                _conv_output_size(current[d], self.kernel_sizes[i][d], self.strides[i][d])
                for d in range(self.spatial_dims)
            )
            shapes.append(current)
        return tuple(shapes)

    def decoder_output_shapes(self) -> Tuple[Tuple[int, int, int], ...]:
        """deep-supervision 各输出的空间尺寸，**从高分辨率到低分辨率**。

        长度 = n_stages - 1，与 nnU-Net 的 deep_supervision_scales 一致。
        """
        return self.stage_shapes()[: self.n_decoder_stages]

    def bottleneck_shape(self) -> Tuple[int, int, int]:
        return self.stage_shapes()[-1]

    # ------------------------------------------------------------------ 导出
    def resolved_snapshot(self) -> dict:
        """写入实验输出目录的 resolved 配置快照（可追溯）。"""
        return {
            "source_path": self.source_path,
            "dataset_name": self.dataset_name,
            "plans_name": self.plans_name,
            "configuration": self.configuration,
            "data_identifier": self.data_identifier,
            "spacing": list(self.spacing),
            "patch_size": list(self.patch_size),
            "batch_size": self.batch_size,
            "features_per_stage": list(self.features_per_stage),
            "kernel_sizes": [list(k) for k in self.kernel_sizes],
            "strides": [list(s) for s in self.strides],
            "n_conv_per_stage": list(self.n_conv_per_stage),
            "n_conv_per_stage_decoder": list(self.n_conv_per_stage_decoder),
            "conv_bias": self.conv_bias,
            "norm_eps": self.norm_eps,
            "norm_affine": self.norm_affine,
            "negative_slope": self.negative_slope,
            "batch_dice": self.batch_dice,
            "stage_shapes": [list(s) for s in self.stage_shapes()],
            "decoder_output_shapes": [list(s) for s in self.decoder_output_shapes()],
        }

    # ------------------------------------------------------------------ 加载
    @staticmethod
    def from_json(
        path: str | Path,
        configuration: str = "3d_fullres",
        dataset_name: str | None = None,
    ) -> "Plan3DConfig":
        """从 nnUNetPlans.json 读取并校验指定 configuration。"""
        path = Path(path)
        if not path.is_file():
            raise PlanConfigError(f"plan 文件不存在: {path}")
        try:
            doc = json.loads(path.read_text())
        except json.JSONDecodeError as exc:  # pragma: no cover - 防御性
            raise PlanConfigError(f"plan 文件不是合法 JSON: {path} ({exc})") from exc

        configurations = doc.get("configurations")
        if not isinstance(configurations, dict) or configuration not in configurations:
            raise PlanConfigError(
                f"{path} 中不存在 configuration={configuration!r}；"
                f"现有: {sorted(configurations) if isinstance(configurations, dict) else 'N/A'}"
            )
        cfg = configurations[configuration]

        arch = cfg.get("architecture") or {}
        class_name = arch.get("network_class_name", "")
        arch_kwargs = arch.get("arch_kwargs") or {}
        if not arch_kwargs:
            raise PlanConfigError(f"{path} 的 configuration={configuration} 缺少 architecture.arch_kwargs")

        n_stages = _require_positive_int(arch_kwargs.get("n_stages"), "n_stages")
        features = _as_int_tuple(arch_kwargs.get("features_per_stage"), "features_per_stage", n_stages)
        kernels = _as_vector_tuple(arch_kwargs.get("kernel_sizes"), "kernel_sizes", 3)
        strides = _as_vector_tuple(arch_kwargs.get("strides"), "strides", 3)
        n_conv = _as_int_tuple(arch_kwargs.get("n_conv_per_stage"), "n_conv_per_stage", n_stages)
        n_conv_dec = _as_int_tuple(
            arch_kwargs.get("n_conv_per_stage_decoder"), "n_conv_per_stage_decoder", n_stages - 1
        )
        if len(kernels) != n_stages or len(strides) != n_stages:
            raise PlanConfigError(
                f"kernel_sizes/strides 长度必须等于 n_stages={n_stages}，"
                f"收到 kernels={len(kernels)} strides={len(strides)}"
            )

        # 自研网络假设：Conv3d + InstanceNorm3d + LeakyReLU，odd kernel + same padding
        for i, k in enumerate(kernels):
            if any(v % 2 == 0 for v in k):
                raise PlanConfigError(f"kernel_sizes[{i}]={k} 含偶数核；自研网络仅支持 odd kernel + same padding")
        if strides[0] != (1, 1, 1):
            raise PlanConfigError(f"strides[0]={strides[0]} 必须为 (1,1,1)（stem 不做下采样）")

        conv_op = str(arch_kwargs.get("conv_op", ""))
        norm_op = str(arch_kwargs.get("norm_op", ""))
        nonlin = str(arch_kwargs.get("nonlin", ""))
        if conv_op != CONV_OP_EXPECTED:
            raise PlanConfigError(f"conv_op={conv_op!r} 与自研实现 {CONV_OP_EXPECTED!r} 不一致")
        if norm_op != NORM_OP_EXPECTED:
            raise PlanConfigError(f"norm_op={norm_op!r} 与自研实现 {NORM_OP_EXPECTED!r} 不一致")
        if nonlin != NONLIN_EXPECTED:
            raise PlanConfigError(f"nonlin={nonlin!r} 与自研实现 {NONLIN_EXPECTED!r} 不一致")

        norm_kwargs = arch_kwargs.get("norm_op_kwargs") or {}
        norm_eps = float(norm_kwargs.get("eps", DEFAULT_NORM_EPS))
        norm_affine = bool(norm_kwargs.get("affine", True))
        nonlin_kwargs = arch_kwargs.get("nonlin_kwargs") or {}
        negative_slope = float(nonlin_kwargs.get("negative_slope", DEFAULT_LEAKY_SLOPE))
        dropout_op = arch_kwargs.get("dropout_op")
        if dropout_op not in (None, "None"):
            raise PlanConfigError(f"plan 中 dropout_op={dropout_op!r}；自研网络不使用 dropout")

        spacing = cfg.get("spacing")
        if not isinstance(spacing, (list, tuple)) or len(spacing) != 3:
            raise PlanConfigError(f"spacing 必须为长度 3 的列表，收到 {spacing!r}")
        spacing = tuple(float(s) for s in spacing)
        if any(s <= 0 for s in spacing):
            raise PlanConfigError(f"spacing 必须为正数，收到 {spacing}")

        patch_size = _as_int_tuple(cfg.get("patch_size"), "patch_size", 3)
        batch_size = _require_positive_int(cfg.get("batch_size"), "batch_size")

        plan = Plan3DConfig(
            dataset_name=str(doc.get("dataset_name", "")),
            plans_name=str(doc.get("plans_name", "")),
            configuration=configuration,
            data_identifier=str(cfg.get("data_identifier", "")),
            spacing=spacing,
            patch_size=patch_size,
            batch_size=batch_size,
            features_per_stage=features,
            kernel_sizes=kernels,
            strides=strides,
            n_conv_per_stage=n_conv,
            n_conv_per_stage_decoder=n_conv_dec,
            conv_bias=bool(arch_kwargs.get("conv_bias", True)),
            norm_eps=norm_eps,
            norm_affine=norm_affine,
            negative_slope=negative_slope,
            batch_dice=bool(cfg.get("batch_dice", False)),
            source_path=str(path),
            raw={"network_class_name": class_name, "spacing": spacing, "patch_size": patch_size},
        )
        if dataset_name is not None and plan.dataset_name and plan.dataset_name != dataset_name:
            raise PlanConfigError(
                f"plan 的 dataset_name={plan.dataset_name!r} 与期望 {dataset_name!r} 不一致"
            )
        plan._validate_shapes()
        return plan

    def _validate_shapes(self) -> None:
        """逐级 shape 必须为正；最深一级也要 >= 1（否则 patch 无法通过全部 stride）。"""
        shapes = self.stage_shapes()
        for i, shape in enumerate(shapes):
            if any(v < 1 for v in shape):
                raise PlanConfigError(
                    f"stage {i} 的空间尺寸 {shape} 出现 0 维：patch={self.patch_size} "
                    f"无法通过 strides={self.strides}"
                )
        if any(v < 1 for v in shapes[-1]):
            raise PlanConfigError(f"bottleneck 尺寸 {shapes[-1]} 非法")


def load_plan(path: str | Path, configuration: str = "3d_fullres", dataset_name: str | None = None) -> Plan3DConfig:
    """便捷入口：`load_plan(path, '3d_fullres')`。"""
    return Plan3DConfig.from_json(path, configuration=configuration, dataset_name=dataset_name)


__all__ = ["Plan3DConfig", "PlanConfigError", "load_plan"]
