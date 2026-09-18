"""M1–M4：三路 modality-specific 浅层分支 + encoder stage 2 融合 + 共享 Residual-Encoder 后半段。

结构（与冻结架构配置 `configs/architectures/fusion_resenc_v23.yaml` 逐值对应）：

```
输入 [B,3,D,H,W]（channel 0/1/2 = T2W/ADC/HBV）
  ├─ branch_T2W: stem(1ch→C0) → stage0 → stage1 → stage2  ─┐
  ├─ branch_ADC: 同结构、参数独立                            ├─ stage0/1 skip: Q_l = 1×1×1+IN+LeakyReLU(concat 三路)
  └─ branch_HBV: 同结构、参数独立                            ┘   stage2: F_fuse
                                                                    │
        M1: F_fuse = (P_T2W + P_ADC + P_HBV)/3（严格 1/3 等权）      │
        M2: gate(三路 stage2 特征) → 逐体素 3 模态 logits → softmax   │
        M3: 同 M2 的 gate + E_z 在融合后作为普通特征注入              │
        M4: 同 M2 的 gate + (1+tanh(gamma))⊙L_img + beta（条件仿射）  │
                                                                    ▼
                       共享 encoder stage 3..n-1 → 普通 U-Net decoder → DS 输出
```

边界（任务书四/五/七）：

- M1/M2 **不接受** zonal 输入（传入即报错）；M3/M4 缺 zonal 输入立即报错，**不得**用全零张量静默代替；
- 浅层 skip 只聚合三路 modality-specific 特征，不读取 PZ/TZ/WG/gate 权重/decoder 特征（Q_l 内部守卫）；
- PZ/TZ 只影响 M4 的 gate logits（与 M3 的融合后注入），**不进入**浅层 skip 或 decoder；
- WG 在任何模型的任何入口都被拒绝（prior 必须为 2 通道）；
- gate 权重与调制量只是结构诊断量，不得解释为临床/因果贡献。
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

import torch
from torch import nn

from ..config.architecture import (
    ACTIVATION_ORDER_POST,
    BLOCK_TYPE_BASIC,
    ENCODER_TYPE_RESIDUAL,
    FUSION_VARIANT_M1_EQUAL,
    FUSION_VARIANT_M2_IMAGE_GATE,
    FUSION_VARIANT_M3_ZONE_INPUT,
    FUSION_VARIANT_M4_CONDITIONED_GATE,
    ArchitectureSpec,
    validate_architecture_against_plan,
)
from ..config.plans import Plan3DConfig
from .blocks import StackedConvBlocks3d, apply_he_init, center_crop_or_pad_3d
from .fusion_blocks import (
    N_MODALITIES,
    ZONE_PRIOR_CHANNELS,
    ConditionalAffineHead,
    ImageDrivenGate,
    ModalityProjection,
    ShallowSkipAggregator,
    ZoneEncoder,
    ZoneFusionProjection,
    prior_occupancy_stats,
    validate_image_tensor,
    validate_zonal_prior,
)
from .m0_concat import MODALITY_ORDER
from .residual_blocks import StackedResidualBlocks3d
from .residual_unet3d import zero_init_residual_last_norms

#: 模型编号（M1–M4）；M0 由 M0ResidualConcatModel 承担
FUSION_MODEL_IDS: tuple[str, ...] = ("M1", "M2", "M3", "M4")

#: 每个模型必须提供的输入键（trainer/推理入口据此取数据；M3/M4 必须提供 zonal_prior）
INPUT_KEY_IMAGE = "image"
INPUT_KEY_ZONAL = "zonal_prior"

#: 诊断量键（仅在 diagnostics_enabled 时填充，全部 detached，避免持有计算图）
DIAGNOSTIC_KEYS: tuple[str, ...] = (
    # 「image logits」= 条件化之前的图像驱动 gate logits（M4 的调制对象）
    "image_logits",
    "gate_logits",
    "gate_weights",
    "final_weights",
    "conditioned_logits",
    "gamma",
    "beta",
    "fused_stage2",
    "zone_features",
    "projected_features",
)


class MultiBranchFusionUNet3D(nn.Module):
    """M1–M4 共同实现（差异只在 `FUSION_VARIANT` 决定的融合头）。"""

    MODEL_ID: ClassVar[str] = ""
    FUSION_VARIANT: ClassVar[str] = ""
    REQUIRED_INPUT_KEYS: ClassVar[tuple[str, ...]] = (INPUT_KEY_IMAGE,)
    ACCEPTS_ANATOMY_INPUTS: ClassVar[bool] = False

    def __init__(
        self,
        plan: Plan3DConfig,
        spec: ArchitectureSpec,
        *,
        in_channels: int = 3,
        num_classes: int = 2,
        deep_supervision: bool = True,
        modality_order: Sequence[str] = MODALITY_ORDER,
        strict_input_validation: bool = False,
    ) -> None:
        super().__init__()
        if not self.MODEL_ID or not self.FUSION_VARIANT:
            raise TypeError("MultiBranchFusionUNet3D 是抽象基类，请使用 M1/M2/M3/M4 具体模型")
        if tuple(modality_order) != MODALITY_ORDER:
            raise ValueError(
                f"模态顺序固定为 {list(MODALITY_ORDER)}（channel 0/1/2），收到 {list(modality_order)}"
            )
        if in_channels != len(MODALITY_ORDER):
            raise ValueError(f"in_channels 必须为 {len(MODALITY_ORDER)}（T2W/ADC/HBV），收到 {in_channels}")

        validate_architecture_against_plan(spec, plan)
        if spec.encoder_type != ENCODER_TYPE_RESIDUAL:
            raise ValueError(f"本类只构建 residual encoder，收到 encoder_type={spec.encoder_type!r}")
        if spec.block_type != BLOCK_TYPE_BASIC or spec.activation_order != ACTIVATION_ORDER_POST:
            raise ValueError(
                f"本类只实现 {BLOCK_TYPE_BASIC}/{ACTIVATION_ORDER_POST}，收到 "
                f"{spec.block_type!r}/{spec.activation_order!r}"
            )
        if spec.in_channels != in_channels or spec.num_classes != num_classes:
            raise ValueError(
                f"in_channels/num_classes 与架构配置不一致：模型=({in_channels},{num_classes}) "
                f"配置=({spec.in_channels},{spec.num_classes})"
            )
        if spec.fusion is None:
            raise ValueError(
                "架构配置缺少 fusion 块：M1–M4 必须通过 architecture.fusion_variant 解析融合身份"
                "（M0 配置不能用于构建融合模型）"
            )
        if spec.fusion.get("variant") != self.FUSION_VARIANT:
            raise ValueError(
                f"架构配置的 fusion.variant={spec.fusion.get('variant')!r} 与模型 {self.MODEL_ID} "
                f"要求的 {self.FUSION_VARIANT!r} 不一致（禁止张冠李戴）"
            )
        if spec.fusion.get("model_id") != self.MODEL_ID:
            raise ValueError(
                f"架构配置的 fusion.model_id={spec.fusion.get('model_id')!r} 与模型类 {self.MODEL_ID} 不一致"
            )

        self.plan = plan
        self.spec = spec
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.deep_supervision = bool(deep_supervision)
        self.modality_order = tuple(modality_order)
        self.blocks_per_stage = tuple(spec.blocks_per_stage)
        self.fusion_stage = int(spec.fusion_stage)
        self.fusion_variant = str(self.FUSION_VARIANT)
        self.backbone_id = "residual_encoder_v23"
        # 诊断量：默认关闭；"cpu" 模式把 detached 张量搬到 CPU，"stats" 只保留标量统计
        self.diagnostics_mode = "off"
        self._diagnostics: dict[str, torch.Tensor] = {}
        self._diagnostic_stats: dict[str, float] = {}
        # 训练热路径默认不做 finite/range 检查（避免每 iteration 的 GPU 同步）；
        # 需要严格校验时（调试/CPU 入口）显式打开或调用 validate_inputs()
        self.strict_input_validation = bool(strict_input_validation)

        norm_eps = plan.norm_eps
        norm_affine = plan.norm_affine
        negative_slope = plan.negative_slope
        conv_bias = plan.conv_bias
        zero_init_last_norm = bool(spec.initialization.get("zero_init_last_norm_before_add", True))
        features = tuple(plan.features_per_stage)
        common = {
            "conv_bias": conv_bias,
            "norm_eps": norm_eps,
            "norm_affine": norm_affine,
            "negative_slope": negative_slope,
        }

        # ------------------------------------------------ 三路 modality-specific 浅层分支（结构相同、参数独立）
        self.branch_stems = nn.ModuleList(
            [
                StackedConvBlocks3d(
                    in_channels=1,
                    out_channels=features[0],
                    kernel_size=plan.kernel_sizes[0],
                    n_convs=1,
                    stride=(1, 1, 1),
                    **common,
                )
                for _ in range(N_MODALITIES)
            ]
        )
        branch_stages = []
        for _ in range(N_MODALITIES):
            stages = []
            for i in range(self.fusion_stage + 1):
                in_ch = features[0] if i == 0 else features[i - 1]
                stages.append(
                    StackedResidualBlocks3d(
                        n_blocks=self.blocks_per_stage[i],
                        in_channels=in_ch,
                        out_channels=features[i],
                        kernel_size=plan.kernel_sizes[i],
                        stride=plan.strides[i],
                        zero_init_last_norm=zero_init_last_norm,
                        **common,
                    )
                )
            branch_stages.append(nn.ModuleList(stages))
        self.branch_stages = nn.ModuleList(branch_stages)

        # ------------------------------------------------ stage 0/1 浅层 skip 聚合（§8.6 契约）
        self.shallow_skip_aggregators = nn.ModuleList(
            [
                ShallowSkipAggregator(features[stage], features[stage], stage=stage, **common)
                for stage in (0, 1)
            ]
        )

        # ------------------------------------------------ 融合前逐模态 projection
        self.modality_projections = nn.ModuleList(
            [ModalityProjection(features[self.fusion_stage], **common) for _ in range(N_MODALITIES)]
        )

        # ------------------------------------------------ 融合头（按变体构建）
        fusion = spec.fusion
        self.gate: ImageDrivenGate | None = None
        self.zone_encoder: ZoneEncoder | None = None
        self.conditioning: ConditionalAffineHead | None = None
        self.zone_injection: ZoneFusionProjection | None = None

        if fusion["gate"] is not None:
            self.gate = ImageDrivenGate(
                features[self.fusion_stage],
                last_layer_zero_init=bool(fusion["gate"]["last_layer_zero_init"]),
                **common,
            )
        if fusion["zone_encoder"] is not None:
            zone_cfg = fusion["zone_encoder"]
            self.zone_encoder = ZoneEncoder(
                int(zone_cfg["features"]),
                n_blocks=int(zone_cfg["n_blocks"]),
                kernel_size=tuple(zone_cfg["kernel_size"]),
                **common,
            )
        if fusion["conditioning"] is not None:
            if self.zone_encoder is None:
                raise ValueError("conditioning 需要 zone_encoder（M4 必须声明 zone_encoder）")
            self.conditioning = ConditionalAffineHead(
                self.zone_encoder.features,
                head_last_layer_zero_init=bool(fusion["conditioning"]["head_last_layer_zero_init"]),
                **common,
            )
        if fusion["zone_injection"] is not None:
            if self.zone_encoder is None:
                raise ValueError("zone_injection 需要 zone_encoder（M3 必须声明 zone_encoder）")
            self.zone_injection = ZoneFusionProjection(
                features[self.fusion_stage], self.zone_encoder.features, **common
            )
        self.ACCEPTS_ANATOMY_INPUTS = self.zone_encoder is not None

        # ------------------------------------------------ 共享 encoder 后半段（与 M0 后端同结构）
        shared_stages = []
        for i in range(self.fusion_stage + 1, plan.n_stages):
            in_ch = features[self.fusion_stage] if i == self.fusion_stage + 1 else features[i - 1]
            shared_stages.append(
                StackedResidualBlocks3d(
                    n_blocks=self.blocks_per_stage[i],
                    in_channels=in_ch,
                    out_channels=features[i],
                    kernel_size=plan.kernel_sizes[i],
                    stride=plan.strides[i],
                    zero_init_last_norm=zero_init_last_norm,
                    **common,
                )
            )
        self.shared_encoder_stages = nn.ModuleList(shared_stages)

        # ------------------------------------------------ 普通 U-Net decoder（与 M0 逐值一致）
        transpconvs, decoder_stages, seg_layers = [], [], []
        for s in range(1, plan.n_stages):
            skip_index = plan.n_stages - 1 - s
            features_below = features[skip_index + 1]
            features_skip = features[skip_index]
            stride = plan.strides[skip_index + 1]
            transpconvs.append(
                nn.ConvTranspose3d(features_below, features_skip, kernel_size=stride, stride=stride, bias=conv_bias)
            )
            decoder_stages.append(
                StackedConvBlocks3d(
                    in_channels=2 * features_skip,
                    out_channels=features_skip,
                    kernel_size=plan.kernel_sizes[skip_index],
                    n_convs=plan.n_conv_per_stage_decoder[s - 1],
                    stride=(1, 1, 1),
                    **common,
                )
            )
            seg_layers.append(nn.Conv3d(features_skip, self.num_classes, 1, 1, 0, bias=True))
        self.decoder_transpconvs = nn.ModuleList(transpconvs)
        self.decoder_stages = nn.ModuleList(decoder_stages)
        self.seg_layers = nn.ModuleList(seg_layers)

        # 全局 He 初始化 → 残差块加法前 norm 置 0 → **再**恢复 gate/条件头的零初始化
        apply_he_init(self, negative_slope=negative_slope)
        self._zero_init_blocks = zero_init_residual_last_norms(self)
        if self.gate is not None and self.gate.last_layer_zero_init:
            self.gate.zero_init_last_layer()
        if self.conditioning is not None and self.conditioning.head_last_layer_zero_init:
            self.conditioning.zero_init_head()

    # ------------------------------------------------------------------ 诊断
    def set_diagnostics_mode(self, mode: str) -> None:
        """设置诊断模式：``off``（默认）/ ``stats``（只保留标量统计）/ ``cpu``（detached 张量搬到 CPU）。

        - ``stats``：不长期持有完整张量，适合长时间训练时的轻量监控；
        - ``cpu``：保留完整张量但强制 offload 到 CPU，不占用 GPU 显存（显式开启时才有开销）。
        """
        if mode not in ("off", "stats", "cpu"):
            raise ValueError(f"diagnostics_mode 必须为 off/stats/cpu，收到 {mode!r}")
        self.diagnostics_mode = mode
        self._diagnostics = {}
        self._diagnostic_stats = {}

    def set_diagnostics(self, enabled: bool) -> None:
        """兼容入口：True → ``cpu`` 模式，False → ``off``。"""
        self.set_diagnostics_mode("cpu" if enabled else "off")

    def gate_diagnostics(self) -> dict[str, torch.Tensor]:
        """返回最近一次 forward 的诊断张量（仅 ``cpu`` 模式下非空；已 detached 并在 CPU）。"""
        return dict(self._diagnostics)

    def diagnostics_stats(self) -> dict[str, float]:
        """返回最近一次 forward 的标量诊断统计（``stats`` 模式）。"""
        return dict(self._diagnostic_stats)

    def _record(self, name: str, value: torch.Tensor) -> None:
        mode = self.diagnostics_mode
        if mode == "off" or name not in DIAGNOSTIC_KEYS:
            return
        if mode == "cpu":
            self._diagnostics[name] = value.detach().to("cpu")
            return
        with torch.no_grad():
            flat = value.detach().float().reshape(-1)
            self._diagnostic_stats[f"{name}_mean"] = float(flat.mean())
            self._diagnostic_stats[f"{name}_std"] = float(flat.std()) if flat.numel() > 1 else 0.0
            self._diagnostic_stats[f"{name}_absmax"] = float(flat.abs().max())

    # ------------------------------------------------------------------ 输入校验
    def validate_inputs(self, image: torch.Tensor, zonal_prior: torch.Tensor | None = None) -> None:
        """显式的完整输入校验（结构 + finite + prior occupancy/overlap）。

        **不在训练热路径中调用**：CPU 数据入口（`PatchDataset`）已做等价检查，
        debug/评测入口可显式调用本方法或把 `strict_input_validation=True` 传给构造器。
        """
        validate_image_tensor(image, self.in_channels, context=self.MODEL_ID)
        if zonal_prior is None:
            if self.zone_encoder is not None:
                raise ValueError(
                    f"{self.MODEL_ID} 需要 zonal_prior（[B,2,D,H,W] = PZ/TZ fractional occupancy）"
                )
            return
        if self.zone_encoder is None:
            raise ValueError(f"{self.MODEL_ID} 不读取 PZ/TZ 先验（gate 只由图像特征驱动）")
        validate_zonal_prior(zonal_prior, image_shape=tuple(image.shape), context=self.MODEL_ID)

    def _check_structure(self, image: torch.Tensor, prior: torch.Tensor | None) -> None:
        """热路径检查：只做 O(1) 的 shape/通道/网格检查，不做 finite/range（避免 GPU 同步）。"""
        if image.ndim != 5:
            raise ValueError(f"输入必须是 [B, 3, D, H, W]，收到 shape={tuple(image.shape)}")
        if image.shape[1] != self.in_channels:
            raise ValueError(
                f"输入通道必须为 {self.in_channels}（T2W/ADC/HBV），收到 {image.shape[1]}；"
                "模态顺序固定为 " + "/".join(self.modality_order)
            )
        if prior is None:
            return
        if prior.ndim != 5 or prior.shape[1] != len(ZONE_PRIOR_CHANNELS):
            raise ValueError(
                f"zonal prior 必须是 [B, {len(ZONE_PRIOR_CHANNELS)}, D, H, W]"
                f"（channel = {list(ZONE_PRIOR_CHANNELS)}）；收到 shape={tuple(prior.shape)}。"
                "WG（3 通道）或其他第三通道一律拒绝。"
            )
        if prior.shape[0] != image.shape[0] or tuple(prior.shape[-3:]) != tuple(image.shape[-3:]):
            raise ValueError(
                f"zonal prior 必须与 image 同一 batch/网格：prior={tuple(prior.shape)} image={tuple(image.shape)}"
            )

    # ------------------------------------------------------------------ 前向
    def forward(
        self,
        image: torch.Tensor,
        *,
        zonal_prior: torch.Tensor | None = None,
        deep_supervision: bool | None = None,
    ):
        """前向；DS 打开返回**高→低**分辨率输出列表，关闭返回单个主输出。

        融合定义（M1–M4 一致的部分）::

            U_m      = P_m(F_m)                                # modality projection
            L_img    = G_img([U_T2W, U_ADC, U_HBV])            # 三路 projection 输出作为 gate 输入
            M1:      w_m = 1/3 ;  M2/M3: w = softmax(L_img) ;  M4: w = softmax((1+tanh(gamma))⊙L_img+β)
            F_fuse   = Σ_m w_m · U_m
            skip[2]  = F_fuse（M3 为 Q([F_fuse, E_z])）
        """
        if self.zone_encoder is None:
            if zonal_prior is not None:
                raise ValueError(
                    f"{self.MODEL_ID} 不读取 PZ/TZ 先验（gate 只由图像特征驱动）；"
                    "zonal_prior 必须为 None。"
                )
        else:
            if zonal_prior is None:
                raise ValueError(
                    f"{self.MODEL_ID} 需要 zonal_prior（[B,2,D,H,W] = PZ/TZ fractional occupancy）；"
                    "缺 prior 时必须报错，禁止用全零张量静默代替。"
                )
        # 热路径只做 shape/网格检查（O(1)）；finite/occupancy/overlap 检查在 CPU 数据入口完成，
        # 或显式打开 strict_input_validation / 调用 validate_inputs()
        self._check_structure(image, zonal_prior)
        if self.strict_input_validation:
            if zonal_prior is not None:
                validate_zonal_prior(zonal_prior, image_shape=tuple(image.shape), context=self.MODEL_ID)
            if not bool(torch.isfinite(image).all()):
                raise ValueError(f"{self.MODEL_ID}: 输入影像含非有限值（NaN/Inf）")
        ds = self.deep_supervision if deep_supervision is None else bool(deep_supervision)
        self._diagnostics = {}
        if self.diagnostics_mode == "stats":
            self._diagnostic_stats = {}

        # 三路浅层分支
        branch_features: list[list[torch.Tensor]] = []
        for m in range(N_MODALITIES):
            f = self.branch_stems[m](image[:, m : m + 1])
            feats = []
            for stage in self.branch_stages[m]:
                f = stage(f)
                feats.append(f)
            branch_features.append(feats)

        skips: list[torch.Tensor] = [
            self.shallow_skip_aggregators[0](*(bf[0] for bf in branch_features)),
            self.shallow_skip_aggregators[1](*(bf[1] for bf in branch_features)),
        ]

        # 融合（encoder stage 2）
        stage2 = [bf[self.fusion_stage] for bf in branch_features]
        projected = [self.modality_projections[m](stage2[m]) for m in range(N_MODALITIES)]
        modality_features = torch.stack(projected, dim=1)  # [B, 3, C, *stage2_spatial]
        self._record("projected_features", modality_features)

        zone_features = None
        if self.zone_encoder is not None:
            zone_features = self.zone_encoder(zonal_prior, stage2[0].shape[-3:])
            self._record("zone_features", zone_features)

        gate_weights = None
        gate_logits = None
        conditioned_logits = None
        gamma = beta = None

        if self.gate is not None:
            # gate 的输入必须是三路 modality projection 的输出 U_m（不是投影前的 stage2 特征）
            gate_logits, gate_weights = self.gate(projected[0], projected[1], projected[2])
            self._record("gate_logits", gate_logits)
            self._record("gate_weights", gate_weights)
            # 「image logits」= 条件化**之前**的图像驱动 gate logits（M4 的调制对象）
            self._record("image_logits", gate_logits)
            final_logits = gate_logits
            if self.conditioning is not None:
                assert zone_features is not None  # M4 必然同时声明 zone_encoder 与 conditioning
                gamma, beta = self.conditioning(zone_features)
                final_logits = (1.0 + torch.tanh(gamma)) * gate_logits + beta
                conditioned_logits = final_logits
                self._record("conditioned_logits", conditioned_logits)
                self._record("gamma", gamma)
                self._record("beta", beta)
            weights = torch.softmax(final_logits, dim=1)
        else:
            # M1：严格 1/3 等权（不使用任何可学习 gate）
            spatial = tuple(modality_features.shape[-3:])
            weights = modality_features.new_full((modality_features.shape[0], N_MODALITIES, *spatial), 1.0 / N_MODALITIES)
        fused = (modality_features * weights.unsqueeze(2)).sum(dim=1)
        self._record("final_weights", weights)

        if self.zone_injection is not None:
            assert zone_features is not None
            # M3：严格 Q([F_fuse, E_z])（无 F_fuse 残差相加）；Q 会替换进入共享 encoder / skip 的特征
            fused = self.zone_injection(fused, zone_features)
        # fused_stage2 = 真正进入共享 encoder 与 stage-2 decoder skip 的融合特征（M3 为 Q([F_fuse, E_z])）
        self._record("fused_stage2", fused)

        skips.append(fused)

        # 共享 encoder 后半段
        features = fused
        for stage in self.shared_encoder_stages:
            features = stage(features)
            skips.append(features)

        # 普通 U-Net decoder
        outputs: list[torch.Tensor] = []
        for idx in range(self.plan.n_decoder_stages):
            skip_index = self.plan.n_stages - 2 - idx
            upsampled = self.decoder_transpconvs[idx](features)
            skip = skips[skip_index]
            upsampled = center_crop_or_pad_3d(upsampled, skip.shape[-3:])
            features = self.decoder_stages[idx](torch.cat((upsampled, skip), dim=1))
            if ds:
                outputs.append(self.seg_layers[idx](features))

        if ds:
            return outputs[::-1]
        return self.seg_layers[-1](features)

    # ------------------------------------------------------------------ 契约 / 工具
    def set_deep_supervision(self, enabled: bool) -> None:
        self.deep_supervision = bool(enabled)

    @staticmethod
    def main_logits(output: torch.Tensor | Sequence[torch.Tensor]) -> torch.Tensor:
        if isinstance(output, (list, tuple)):
            if not output:
                raise ValueError("空的 deep-supervision 输出列表")
            return output[0]
        return output

    def architecture_identity(self) -> dict:
        return self.spec.identity()

    def model_identity(self) -> dict[str, Any]:
        """模型身份（checkpoint/日志）：model_id + backbone + 融合结构 + 架构哈希。"""
        fusion = self.spec.fusion or {}
        identity = {
            "model_id": self.MODEL_ID,
            "backbone_id": self.backbone_id,
            "fusion_variant": fusion.get("variant"),
            "fusion_family_id": fusion.get("family_id"),
            "fusion_stage": self.fusion_stage,
            "shallow_branch_stages": list(fusion.get("shallow_branch", {}).get("stages", [])),
            "shallow_branch_independent_parameters": fusion.get("shallow_branch", {}).get(
                "independent_parameters"
            ),
            "gate_type": (fusion.get("gate") or {}).get("type"),
            "zone_encoder": fusion.get("zone_encoder"),
            "zone_injection_type": (fusion.get("zone_injection") or {}).get("type"),
            "conditioning_type": (fusion.get("conditioning") or {}).get("type"),
            "architecture_sha256": self.spec.architecture_sha256(),
        }
        return identity

    def num_parameters(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))

    def num_trainable_parameters(self) -> int:
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))

    @staticmethod
    def _count(module: nn.Module | None) -> int:
        return 0 if module is None else int(sum(p.numel() for p in module.parameters()))

    def parameter_breakdown(self) -> dict[str, int]:
        """参数量分解（容量混杂排查：gate / zone encoder / 浅层 skip / 分支 / 共享后端分开报告）。"""
        shared_backend = (
            self._count(self.shared_encoder_stages)
            + self._count(self.decoder_transpconvs)
            + self._count(self.decoder_stages)
            + self._count(self.seg_layers)
        )
        breakdown = {
            "branch_stems": self._count(self.branch_stems),
            "branch_stages_shallow": self._count(self.branch_stages),
            "shallow_skip_aggregators": self._count(self.shallow_skip_aggregators),
            "modality_projections": self._count(self.modality_projections),
            "gate": self._count(self.gate),
            "zone_encoder": self._count(self.zone_encoder),
            "conditioning": self._count(self.conditioning),
            "zone_injection": self._count(self.zone_injection),
            "shared_encoder_and_decoder": shared_backend,
        }
        breakdown["total"] = self.num_parameters()
        return breakdown

    def prior_occupancy_stats(self, zonal_prior: torch.Tensor) -> dict[str, float]:
        """PZ/TZ occupancy 标量统计（训练日志用；不保留张量、不做梯度）。"""
        return prior_occupancy_stats(zonal_prior)

    def macs_breakdown(self) -> dict[str, int]:
        """解析 MACs 分组（不 forward；可在真实 plan 上安全调用）。

        分组：浅层三分支（stem / stage0-2）、浅层 skip 聚合器、modality projection、gate、
        zone encoder、conditioning、zone injection、共享 encoder+decoder；单位 = MACs（1 MAC = 1 乘加）。
        """
        from .profiling import fusion_group_macs_analytical

        return fusion_group_macs_analytical(self)

    def macs_breakdown_with_hooks(self) -> dict[str, int]:
        """forward-hook 版分组 MACs（**仅用于缩小的合成 plan**，验证解析公式；真实 patch 上昂贵）。"""
        from .profiling import fusion_group_macs_with_hooks

        return fusion_group_macs_with_hooks(self)

    def encoder_stage_shapes(self) -> tuple[tuple[int, int, int], ...]:
        return self.plan.stage_shapes()

    def expected_output_shapes(self, patch_size: Sequence[int] | None = None) -> tuple[tuple[int, int, int], ...]:
        if patch_size is not None and tuple(int(v) for v in patch_size) != tuple(self.plan.patch_size):
            raise ValueError(f"patch_size={tuple(patch_size)} 与 plan 的 {tuple(self.plan.patch_size)} 不一致")
        return self.plan.decoder_output_shapes()

    def summary(self) -> dict:
        info = {
            "model_id": self.MODEL_ID,
            "backbone_id": self.backbone_id,
            "fusion_variant": self.fusion_variant,
            "fusion_stage": self.fusion_stage,
            "accepts_anatomy_inputs": self.ACCEPTS_ANATOMY_INPUTS,
            "required_input_keys": list(self.REQUIRED_INPUT_KEYS),
            "modality_order": list(self.modality_order),
            "in_channels": self.in_channels,
            "num_classes": self.num_classes,
            "deep_supervision": self.deep_supervision,
            "dataset_name": self.plan.dataset_name,
            "configuration": self.plan.configuration,
            "architecture_name": self.spec.architecture_name,
            "architecture_version": self.spec.architecture_version,
            "architecture_sha256": self.spec.architecture_sha256(),
            "n_stages": self.plan.n_stages,
            "patch_size": list(self.plan.patch_size),
            "spacing": list(self.plan.spacing),
            "features_per_stage": list(self.plan.features_per_stage),
            "blocks_per_stage": list(self.blocks_per_stage),
            "strides": [list(s) for s in self.plan.strides],
            "kernel_sizes": [list(k) for k in self.plan.kernel_sizes],
            "n_conv_per_stage_decoder": list(self.plan.n_conv_per_stage_decoder),
            "stage_shapes": [list(s) for s in self.plan.stage_shapes()],
            "decoder_output_shapes": [list(s) for s in self.plan.decoder_output_shapes()],
            "parameter_breakdown": self.parameter_breakdown(),
            "macs_breakdown": self.macs_breakdown(),
            "num_parameters": self.num_parameters(),
        }
        return info

    def format_summary(self) -> str:
        info = self.summary()
        breakdown = info["parameter_breakdown"]
        lines = [
            (
                f"model_id            : {info['model_id']}（variant={info['fusion_variant']}，"
                f"backbone={info['backbone_id']}）"
            ),
            f"architecture        : {info['architecture_name']} {info['architecture_version']}",
            f"architecture_sha256 : {info['architecture_sha256']}",
            f"fusion_stage        : {info['fusion_stage']}（encoder stage 2）",
            (
                f"输入                : [B, {info['in_channels']}, *patch] "
                f"modality={info['modality_order']}；required_keys={info['required_input_keys']}"
            ),
            f"接受解剖输入        : {info['accepts_anatomy_inputs']}",
            f"patch_size          : {info['patch_size']}  spacing={info['spacing']}",
            f"stage_shapes        : {info['stage_shapes']}",
            f"DS output shapes    : {info['decoder_output_shapes']}（高→低）",
            f"参数量（总）        : {info['num_parameters']:,}",
            "参数量分解          : "
            + ", ".join(f"{k}={v:,}" for k, v in breakdown.items() if k != "total"),
        ]
        return "\n".join(lines)


class M1EqualFusionModel(MultiBranchFusionUNet3D):
    """M1：三路独立浅层分支 + 严格 1/3 等权融合（无 gate、无解剖输入）。"""

    MODEL_ID = "M1"
    FUSION_VARIANT = FUSION_VARIANT_M1_EQUAL
    REQUIRED_INPUT_KEYS = (INPUT_KEY_IMAGE,)
    ACCEPTS_ANATOMY_INPUTS = False


class M2ImageGateFusionModel(MultiBranchFusionUNet3D):
    """M2：三路独立浅层分支 + 图像驱动空间 gate（无解剖输入）。"""

    MODEL_ID = "M2"
    FUSION_VARIANT = FUSION_VARIANT_M2_IMAGE_GATE
    REQUIRED_INPUT_KEYS = (INPUT_KEY_IMAGE,)
    ACCEPTS_ANATOMY_INPUTS = False


class M3ZoneInputFusionModel(MultiBranchFusionUNet3D):
    """M3：与 M2 完全相同的 image gate；PZ/TZ 在融合后作为**普通特征**注入（不进 gate）。"""

    MODEL_ID = "M3"
    FUSION_VARIANT = FUSION_VARIANT_M3_ZONE_INPUT
    REQUIRED_INPUT_KEYS = (INPUT_KEY_IMAGE, INPUT_KEY_ZONAL)
    ACCEPTS_ANATOMY_INPUTS = True


class M4ConditionedGateFusionModel(MultiBranchFusionUNet3D):
    """M4：PZ/TZ 通过有界 conditional-affine 直接调制序列 gate logits（核心创新）。"""

    MODEL_ID = "M4"
    FUSION_VARIANT = FUSION_VARIANT_M4_CONDITIONED_GATE
    REQUIRED_INPUT_KEYS = (INPUT_KEY_IMAGE, INPUT_KEY_ZONAL)
    ACCEPTS_ANATOMY_INPUTS = True


__all__ = [
    "DIAGNOSTIC_KEYS",
    "FUSION_MODEL_IDS",
    "INPUT_KEY_IMAGE",
    "INPUT_KEY_ZONAL",
    "M1EqualFusionModel",
    "M2ImageGateFusionModel",
    "M3ZoneInputFusionModel",
    "M4ConditionedGateFusionModel",
    "MultiBranchFusionUNet3D",
]
