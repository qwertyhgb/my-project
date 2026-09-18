"""结构摘要与预算分析：参数量、FLOPs/MACs、激活规模与显存状态（**不读取真实数据、不使用 GPU**）。

方法与边界（任务书八）：

- **参数量**：直接从构建好的模型统计（total / trainable），精确、无 forward；
- **MACs/FLOPs**：``analytical_macs`` 用**纯算术**按冻结 plan + 架构 spec 枚举每个 Conv3d/ConvTranspose3d
  的乘加数（**不做 forward**，因此对真实 ``[16,320,320]`` patch 也安全、快速）。约定：
  ``1 MAC = 1 乘加``，``FLOPs ≈ 2 × MACs``；统计**包含** decoder 与所有 deep-supervision seg head，
  **不包含**无权重算子（AvgPool/InstanceNorm/LeakyReLU/残差加法）。``count_macs_with_hooks`` 是等价的
  forward-hook 实现，仅用于在**缩小的合成 plan** 上验证解析公式（单元测试对齐二者）；
- **激活规模**：``analytical_feature_map_elements`` 给出所有卷积/转置卷积输出激活元素数（nnU-Net
  ``compute_conv_feature_map_size`` 同类代理量），用于**分析性**显存估计；
- **峰值显存**：本轮**无 GPU 授权，绝不实测**；``peak_gpu_memory.status = NOT_MEASURED``，只给分析性
  激活字节下界代理，明确区分「实测 / 分析估计 / 未测量」，不得把估计写成实测值。
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from ..config.architecture import ArchitectureSpec
from ..config.plans import Plan3DConfig

BYTES_PER_ELEMENT_FP32 = 4
BYTES_PER_ELEMENT_AMP = 2  # fp16/bf16 激活的粗略上界参考


def _prod(values: Sequence[int]) -> int:
    return int(math.prod(int(v) for v in values))


# --------------------------------------------------------------------------- 参数量
def count_parameters(model: nn.Module) -> dict[str, int]:
    total = int(sum(p.numel() for p in model.parameters()))
    trainable = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    return {"total": total, "trainable": trainable, "non_trainable": total - trainable}


# --------------------------------------------------------------------------- 解析 MACs
def analytical_macs(
    plan: Plan3DConfig,
    spec: ArchitectureSpec,
    in_channels: int,
    num_classes: int,
) -> dict[str, int]:
    """按冻结 plan + 架构 spec 纯算术枚举 MACs（不含 forward）。

    与 `PlanDrivenResidualEncoderUNet3D` 的构建逐层对应；返回 encoder/decoder/total（单位：MACs）。
    """
    feats = plan.features_per_stage
    kernels = plan.kernel_sizes
    strides = plan.strides
    shapes = plan.stage_shapes()
    n_stages = plan.n_stages
    blocks = tuple(spec.blocks_per_stage)
    if len(blocks) != n_stages:
        raise ValueError(f"blocks_per_stage 长度 {len(blocks)} 与 n_stages={n_stages} 不一致")

    encoder = 0
    # stem：1 个 Conv3d（in_channels → feats[0]，stride 1，输出 = shapes[0]）
    encoder += feats[0] * _prod(shapes[0]) * int(in_channels) * _prod(kernels[0])
    # residual encoder stages
    for s in range(n_stages):
        in_ch = feats[0] if s == 0 else feats[s - 1]
        out_spatial = _prod(shapes[s])
        kvol = _prod(kernels[s])
        n_blk = blocks[s]
        # 第一个 block：conv1(stride) + conv2 + （通道变化时）1×1×1 projection
        encoder += feats[s] * out_spatial * in_ch * kvol  # conv1
        encoder += feats[s] * out_spatial * feats[s] * kvol  # conv2
        if in_ch != feats[s]:
            encoder += feats[s] * out_spatial * in_ch * 1  # 1×1×1 projection shortcut
        # 其余 block：in==out==feats[s]，stride 1，identity shortcut
        for _ in range(n_blk - 1):
            encoder += feats[s] * out_spatial * feats[s] * kvol  # conv1
            encoder += feats[s] * out_spatial * feats[s] * kvol  # conv2

    decoder = 0
    n_dec = plan.n_decoder_stages
    n_conv_dec = plan.n_conv_per_stage_decoder
    for idx in range(n_dec):
        skip_index = n_stages - 2 - idx
        c_below = feats[skip_index + 1]
        c_skip = feats[skip_index]
        k_t = strides[skip_index + 1]  # transpconv kernel = stride
        in_spatial = _prod(shapes[skip_index + 1])
        out_spatial = _prod(shapes[skip_index])
        kdec = _prod(kernels[skip_index])
        # transpconv（按输入元素计：Cin·prod(Si)·Cout·prod(k)）
        decoder += c_below * in_spatial * c_skip * _prod(k_t)
        # stacked convs（在 skip 分辨率 shapes[skip_index] 上）
        n_conv = int(n_conv_dec[idx])
        decoder += c_skip * out_spatial * (2 * c_skip) * kdec  # 第一个：concat 后 2*c_skip → c_skip
        for _ in range(n_conv - 1):
            decoder += c_skip * out_spatial * c_skip * kdec
        # deep-supervision seg head（1×1×1）
        decoder += int(num_classes) * out_spatial * c_skip * 1

    return {"encoder": int(encoder), "decoder": int(decoder), "total": int(encoder + decoder)}


def count_macs_with_hooks(model: nn.Module, sample_input: torch.Tensor) -> dict[str, int]:
    """forward-hook 版 MAC 计数（**仅在缩小的合成 plan 上用于验证解析公式**；对真实 patch 昂贵，勿用）。

    约定与 `analytical_macs` 完全一致：Conv3d 按输出元素计，ConvTranspose3d 按输入元素计，逐样本（除以 batch）。
    """
    total = {"macs": 0}

    def _hook(module: nn.Module, inputs: Any, output: Any) -> None:
        x = inputs[0]
        batch = int(x.shape[0])
        if isinstance(module, nn.ConvTranspose3d):
            cin = int(module.in_channels)
            cout = int(module.out_channels)
            groups = int(module.groups)
            kvol = _prod(module.kernel_size)
            si = _prod(x.shape[-3:])
            total["macs"] += cin * si * (cout // groups) * kvol // batch
        elif isinstance(module, nn.Conv3d):
            cin = int(module.in_channels)
            cout = int(module.out_channels)
            groups = int(module.groups)
            kvol = _prod(module.kernel_size)
            so = _prod(output.shape[-3:])
            total["macs"] += cout * so * (cin // groups) * kvol // batch

    handles = []
    for m in model.modules():
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
            handles.append(m.register_forward_hook(_hook))
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(sample_input)
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)
    return {"total": int(total["macs"])}


# --------------------------------------------------------------------------- 激活规模 / 显存
def analytical_feature_map_elements(
    plan: Plan3DConfig,
    spec: ArchitectureSpec,
    in_channels: int,
    num_classes: int,
) -> dict[str, int]:
    """所有 Conv3d/ConvTranspose3d/seg-head 输出激活元素数之和（逐样本；nnU-Net 同类代理量）。"""
    feats = plan.features_per_stage
    shapes = plan.stage_shapes()
    n_stages = plan.n_stages
    blocks = tuple(spec.blocks_per_stage)

    enc = 0
    enc += feats[0] * _prod(shapes[0])  # stem 输出
    for s in range(n_stages):
        out_spatial = _prod(shapes[s])
        n_blk = blocks[s]
        # 每个 residual block：conv1 输出 + conv2 输出（均在 shapes[s]）
        enc += n_blk * (feats[s] * out_spatial) * 2
    dec = 0
    n_conv_dec = plan.n_conv_per_stage_decoder
    for idx in range(plan.n_decoder_stages):
        skip_index = n_stages - 2 - idx
        c_skip = feats[skip_index]
        out_spatial = _prod(shapes[skip_index])
        dec += c_skip * out_spatial  # transpconv 输出
        dec += int(n_conv_dec[idx]) * (c_skip * out_spatial)  # stacked conv 输出
        dec += int(num_classes) * out_spatial  # seg head 输出
    return {"encoder": int(enc), "decoder": int(dec), "total": int(enc + dec)}


def structure_summary(
    model: nn.Module,
    plan: Plan3DConfig,
    spec: ArchitectureSpec,
    *,
    in_channels: int,
    num_classes: int,
    batch_size: int | None = None,
) -> dict[str, Any]:
    """可复现的结构摘要（参数量 + 每 stage block/通道/shape + DS 输出 + MACs + 激活规模 + 架构哈希 + 显存状态）。

    **不做 forward、不读数据、不使用 GPU**；MACs/激活为解析值，显存标记为未实测。
    """
    params = count_parameters(model)
    macs = analytical_macs(plan, spec, in_channels, num_classes)
    acts = analytical_feature_map_elements(plan, spec, in_channels, num_classes)
    bs = int(batch_size if batch_size is not None else plan.batch_size)
    ident = spec.identity()
    activation_bytes_fp32 = acts["total"] * BYTES_PER_ELEMENT_FP32 * bs
    return {
        "architecture_name": ident["architecture_name"],
        "architecture_version": ident["architecture_version"],
        "architecture_sha256": ident["architecture_sha256"],
        "encoder_type": spec.encoder_type,
        "block_type": spec.block_type,
        "activation_order": spec.activation_order,
        "n_stages": plan.n_stages,
        "blocks_per_stage": list(spec.blocks_per_stage),
        "features_per_stage": list(plan.features_per_stage),
        "strides": [list(s) for s in plan.strides],
        "kernel_sizes": [list(k) for k in plan.kernel_sizes],
        "stage_shapes": [list(s) for s in plan.stage_shapes()],
        "decoder_output_shapes": [list(s) for s in plan.decoder_output_shapes()],
        "patch_size": list(plan.patch_size),
        "batch_size": bs,
        "parameters": params,
        "macs": {
            **macs,
            "unit": "MACs (multiply-accumulate); FLOPs ~= 2 * MACs",
            "method": "analytical enumeration over Conv3d/ConvTranspose3d (no forward pass)",
            "input_shape": [int(in_channels), *plan.patch_size],
            "includes_decoder": True,
            "includes_deep_supervision_heads": True,
            "excludes": ["avgpool", "instance_norm", "leaky_relu", "residual_add", "concat"],
        },
        "activation_feature_map_elements": {
            **acts,
            "method": "analytical sum of conv/transpconv/seg-head output elements per sample",
        },
        "peak_gpu_memory": {
            "status": "NOT_MEASURED",
            "reason": "本轮无 GPU 授权，未做实测；峰值显存须在 P2B 由研究者在真实 GPU 上测量",
            "analytical_activation_bytes_fp32": int(activation_bytes_fp32),
            "analytical_activation_bytes_amp": int(acts["total"] * BYTES_PER_ELEMENT_AMP * bs),
            "note": "解析激活字节仅为**下界代理**（单次前向激活元素和 × 字节 × batch），"
            "不等于峰值显存（未计权重/梯度/优化器状态/框架开销/反向中间量）；不得当作实测值",
        },
    }


# --------------------------------------------------------------------------- M1–M4 融合模型 MACs 分解
#: 与架构配置/模型模块一一对应的 MACs 分组（用于容量混杂排查；M0 的 MACs 不得冒充这些数字）
FUSION_MAC_GROUPS: tuple[str, ...] = (
    "branch_stems",
    "branch_stages_shallow",
    "shallow_skip_aggregators",
    "modality_projections",
    "gate",
    "zone_encoder",
    "conditioning",
    "zone_injection",
    "shared_encoder_and_decoder",
)


def _fusion_group_members(model: nn.Module) -> dict[str, list[nn.Module]]:
    """按模型属性把子模块归入 MACs 分组（含所有嵌套子模块）。"""

    def expand(attr: str) -> list[nn.Module]:
        module = getattr(model, attr, None)
        if not isinstance(module, nn.Module):
            return []
        return list(module.modules())

    shared: list[nn.Module] = []
    for attr in ("shared_encoder_stages", "decoder_transpconvs", "decoder_stages", "seg_layers"):
        shared.extend(expand(attr))
    return {
        "branch_stems": expand("branch_stems"),
        "branch_stages_shallow": expand("branch_stages"),
        "shallow_skip_aggregators": expand("shallow_skip_aggregators"),
        "modality_projections": expand("modality_projections"),
        "gate": expand("gate"),
        "zone_encoder": expand("zone_encoder"),
        "conditioning": expand("conditioning"),
        "zone_injection": expand("zone_injection"),
        "shared_encoder_and_decoder": shared,
    }


def _conv_macs(module: nn.Module, inputs: Any, output: Any, batch: int) -> int:
    """与 `analytical_macs` 相同的计数约定：Conv3d 按输出元素、ConvTranspose3d 按输入元素。"""
    x = inputs[0]
    if isinstance(module, nn.ConvTranspose3d):
        return int(module.in_channels) * _prod(x.shape[-3:]) * (int(module.out_channels) // int(module.groups)) * _prod(
            module.kernel_size
        ) // batch
    cin = int(module.in_channels)
    cout = int(module.out_channels)
    groups = int(module.groups)
    return cout * _prod(output.shape[-3:]) * (cin // groups) * _prod(module.kernel_size) // batch


def fusion_group_macs_with_hooks(model: nn.Module, *, prior_shape: Sequence[int] | None = None) -> dict[str, int]:
    """按分组统计 M1–M4 的 MACs（forward-hook；**只应在缩小的合成 plan 上使用**）。

    返回 ``{group: macs, ..., "total": macs}``（逐样本）。约定与 `analytical_macs` 一致：
    1 MAC = 1 乘加，包含 decoder 与所有 deep-supervision head，不含 pool/norm/激活/加法/concat。
    """
    groups = _fusion_group_members(model)
    owner = {id(module): name for name, modules in groups.items() for module in modules}
    acc: dict[str, int] = {name: 0 for name in FUSION_MAC_GROUPS}
    acc["unassigned"] = 0

    def _hook(module: nn.Module, inputs: Any, output: Any) -> None:
        batch = int(inputs[0].shape[0])
        macs = _conv_macs(module, inputs, output, batch)
        acc[owner.get(id(module), "unassigned")] += macs

    handles = [
        m.register_forward_hook(_hook)
        for m in model.modules()
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d))
    ]
    patch = tuple(int(v) for v in model.plan.patch_size)
    image = torch.zeros(1, int(model.in_channels), *patch)
    kwargs: dict[str, Any] = {}
    if getattr(model, "zone_encoder", None) is not None:
        kwargs["zonal_prior"] = torch.zeros(1, 2, *(tuple(prior_shape) if prior_shape else patch))
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(image, **kwargs)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    acc["total"] = int(sum(acc[name] for name in FUSION_MAC_GROUPS) + acc["unassigned"])
    return acc


def fusion_group_macs_analytical(model: nn.Module, *, prior_shape: Sequence[int] | None = None) -> dict[str, int]:
    """M1–M4 的**解析** MACs 分解（不 forward）：按 plan 空间尺寸与模块属性逐层枚举。

    与 `fusion_group_macs_with_hooks` 在缩小合成 plan 上必须逐值一致（单元测试锁定）。
    """
    plan = model.plan
    fs = int(model.fusion_stage)
    feats = tuple(int(v) for v in plan.features_per_stage)
    kernels = tuple(tuple(int(v) for v in k) for k in plan.kernel_sizes)
    shapes = tuple(tuple(int(v) for v in s) for s in plan.stage_shapes())
    blocks = tuple(int(v) for v in model.blocks_per_stage)

    def residual_macs(in_ch: int, out_ch: int, spatial: int, kvol: int, n_blocks: int) -> int:
        macs = out_ch * spatial * in_ch * kvol + out_ch * spatial * out_ch * kvol
        if in_ch != out_ch:
            macs += out_ch * spatial * in_ch  # 1×1×1 projection shortcut
        for _ in range(n_blocks - 1):
            macs += 2 * out_ch * spatial * out_ch * kvol
        return int(macs)

    branch_stages = 0
    for s in range(fs + 1):
        in_ch = feats[0] if s == 0 else feats[s - 1]
        branch_stages += residual_macs(in_ch, feats[s], _prod(shapes[s]), _prod(kernels[s]), blocks[s])

    acc: dict[str, int] = {}
    acc["branch_stems"] = 3 * feats[0] * _prod(shapes[0]) * 1 * _prod(kernels[0])
    acc["branch_stages_shallow"] = 3 * branch_stages
    acc["shallow_skip_aggregators"] = sum(
        feats[s] * _prod(shapes[s]) * (3 * feats[s]) for s in (0, 1)
    )
    c2 = feats[fs]
    s2 = _prod(shapes[fs])
    acc["modality_projections"] = 3 * c2 * s2 * c2

    gate = getattr(model, "gate", None)
    if gate is not None:
        acc["gate"] = (
            int(gate.proj.out_channels) * s2 * int(gate.proj.in_channels)
            + int(gate.logits.out_channels) * s2 * int(gate.logits.in_channels)
        )
    else:
        acc["gate"] = 0

    zone = getattr(model, "zone_encoder", None)
    if zone is not None:
        f_ch = int(zone.features)
        kvol = _prod(zone.kernel_size)
        zone_macs = f_ch * s2 * int(zone.input_channels) * kvol
        zone_macs += int(zone.n_blocks) * (2 * f_ch * s2 * f_ch * kvol)
        acc["zone_encoder"] = int(zone_macs)
    else:
        acc["zone_encoder"] = 0

    cond = getattr(model, "conditioning", None)
    if cond is not None:
        acc["conditioning"] = int(cond.proj.out_channels) * s2 * int(cond.proj.in_channels) + int(
            cond.head.out_channels
        ) * s2 * int(cond.head.in_channels)
    else:
        acc["conditioning"] = 0

    inj = getattr(model, "zone_injection", None)
    if inj is not None:
        acc["zone_injection"] = int(inj.conv.out_channels) * s2 * int(inj.conv.in_channels)
    else:
        acc["zone_injection"] = 0

    # 共享 encoder（stage fs+1..n-1）+ 普通 decoder（与 analytical_macs 的 decoder 段一致）
    shared = 0
    for s in range(fs + 1, plan.n_stages):
        shared += residual_macs(feats[s - 1], feats[s], _prod(shapes[s]), _prod(kernels[s]), blocks[s])
    for idx in range(plan.n_decoder_stages):
        skip_index = plan.n_stages - 2 - idx
        c_below = feats[skip_index + 1]
        c_skip = feats[skip_index]
        k_t = tuple(int(v) for v in plan.strides[skip_index + 1])
        in_spatial = _prod(shapes[skip_index + 1])
        out_spatial = _prod(shapes[skip_index])
        kdec = _prod(kernels[skip_index])
        shared += c_below * in_spatial * c_skip * _prod(k_t)
        n_conv = int(plan.n_conv_per_stage_decoder[idx])
        shared += c_skip * out_spatial * (2 * c_skip) * kdec
        for _ in range(n_conv - 1):
            shared += c_skip * out_spatial * c_skip * kdec
        shared += int(model.num_classes) * out_spatial * c_skip
    acc["shared_encoder_and_decoder"] = int(shared)
    acc["total"] = int(sum(acc[name] for name in FUSION_MAC_GROUPS))
    return acc


__all__ = [
    "BYTES_PER_ELEMENT_AMP",
    "BYTES_PER_ELEMENT_FP32",
    "FUSION_MAC_GROUPS",
    "analytical_feature_map_elements",
    "analytical_macs",
    "count_macs_with_hooks",
    "count_parameters",
    "fusion_group_macs_analytical",
    "fusion_group_macs_with_hooks",
    "structure_summary",
]
