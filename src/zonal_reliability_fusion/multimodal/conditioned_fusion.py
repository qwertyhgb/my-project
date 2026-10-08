"""Neutral path plus soft conditioned residual fusion before native nnU-Net.

B: independent shallow stems + neutral projection.
C: B + pre-backbone lesionness conditioning + auxiliary GT-only supervision.
D: C + predicted soft WG/PZ/TZ and derived uncertainty-like context.

Auxiliary logits use the existing single-process loss side channel. Weight export
is an explicit patch-level hook, not a replacement sliding-window inferencer.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

STEM_CHANNELS = 8
#: 局部融合 softmax 的温度。**预先冻结为 1.0**：不做搜索、不设 CLI 超参数。
#: 它显式出现在前向计算里（``weights = (logits / T).softmax(dim=1)``），因此代码、run_config
#: 与文档对这一常量的语义完全一致；T=1 时与直接 ``logits.softmax(dim=1)`` 数值完全相同。
SOFTMAX_TEMPERATURE = 1.0


class ConditionedMultimodalNNUNet(nn.Module):
    """Small, sequence-specific front end around an unchanged 3-channel backbone."""

    def __init__(self, backbone, *, condition="neutral", stem_channels=STEM_CHANNELS):
        super().__init__()
        if condition not in ("neutral", "lesion", "anatomy"):
            raise ValueError("condition must be neutral, lesion, or anatomy")
        self.condition = condition
        self.backbone = backbone
        self.stems = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(1, stem_channels, 3, padding=1),
                nn.InstanceNorm3d(stem_channels, affine=True), nn.LeakyReLU(inplace=True),
                nn.Conv3d(stem_channels, stem_channels, 3, padding=1),
                nn.InstanceNorm3d(stem_channels, affine=True), nn.LeakyReLU(inplace=True),
            ) for _ in range(3)
        ])
        self.neutral_projection = nn.Conv3d(3 * stem_channels, 3, 1)
        self.auxiliary_outputs = None
        self.export_fusion_weights = False
        self.last_fusion_weights = None
        if condition != "neutral":
            # Fixed factor-two pooling gives an actual coarse, pre-fusion branch.
            self.lesionness_head = nn.Sequential(
                nn.Conv3d(3, 8, 3, padding=1), nn.LeakyReLU(inplace=True), nn.Conv3d(8, 1, 1)
            )
            anatomy_channels = 4 if condition == "anatomy" else 0
            self.controller = nn.Sequential(
                nn.Conv3d(3 * stem_channels + 1 + anatomy_channels, 8, 1),
                nn.LeakyReLU(inplace=True), nn.Conv3d(8, 3, 1),
            )
            self.residual_projection = nn.Conv3d(stem_channels + 1 + anatomy_channels, 3, 1)
            nn.init.zeros_(self.residual_projection.weight)
            nn.init.zeros_(self.residual_projection.bias)

    @property
    def decoder(self):
        return self.backbone.decoder

    def fuse(self, x):
        expected = 6 if self.condition == "anatomy" else 3
        if x.ndim != 5 or x.shape[1] != expected:
            raise ValueError(f"expected [B,{expected},D,H,W], got {tuple(x.shape)}")
        if not torch.isfinite(x).all():
            raise ValueError("non-finite multimodal input")
        if min(x.shape[2:]) < 2 or any(n % 2 for n in x.shape[2:]):
            raise ValueError("coarse grid requires even 3D dimensions >=2")
        anatomy = None
        if self.condition == "anatomy":
            prior = x[:, 3:]
            if not prior.is_floating_point() or (prior < 0).any() or (prior > 1).any():
                raise ValueError("anatomy must be finite floating probabilities in [0,1]")
            anatomy = torch.cat((prior, 1 - prior[:, 1:].amax(dim=1, keepdim=True)), dim=1)
        features = [stem(x[:, i:i+1]) for i, stem in enumerate(self.stems)]
        concatenated = torch.cat(features, dim=1)
        neutral = self.neutral_projection(concatenated)
        self.auxiliary_outputs = None
        self.last_fusion_weights = None
        if self.condition == "neutral":
            return neutral, None, neutral
        lesionness_logits = self.lesionness_head(F.avg_pool3d(neutral, 2))
        lesionness = F.interpolate(lesionness_logits.sigmoid(), size=neutral.shape[2:],
                                  mode="trilinear", align_corners=False)
        context = [lesionness] + ([] if anatomy is None else [anatomy])
        controller_logits = self.controller(torch.cat([concatenated, *context], dim=1))
        # Temperature is applied explicitly instead of being asserted only in the run config;
        # at the frozen value 1.0 this is numerically identical to a plain softmax over dim=1.
        weights = (controller_logits / SOFTMAX_TEMPERATURE).softmax(dim=1)
        adaptive = sum(weights[:, i:i+1] * feature for i, feature in enumerate(features))
        delta = self.residual_projection(torch.cat([adaptive, *context], dim=1))
        self.auxiliary_outputs = {"lesionness_logits": lesionness_logits}
        return neutral + delta, weights, neutral

    def forward(self, x):
        fused, weights, _ = self.fuse(x)
        if self.export_fusion_weights and weights is not None:
            self.last_fusion_weights = weights.detach().cpu()
        return self.backbone(fused)

    def fusion_weight_artifact(self, *, case_id, checkpoint, geometry):
        """Explicit patch hook. Caller must supply geometry/case/checkpoint provenance."""
        if not case_id or not checkpoint or not geometry:
            raise ValueError("case/checkpoint/geometry provenance is required")
        if self.last_fusion_weights is None:
            raise RuntimeError("enable export_fusion_weights and run forward first")
        return {"weights": self.last_fusion_weights, "case_id": case_id,
                "checkpoint": str(checkpoint), "geometry": geometry,
                "channel_order": ["T2W", "ADC", "HBV"],
                "condition": self.condition, "space": "network input patch",
                "interpretation": "model fusion coefficients; not causal contributions"}

    def compute_conv_feature_map_size(self, input_size):
        # Planning stays native; this proxy is not a measured memory/compute claim.
        return self.backbone.compute_conv_feature_map_size(input_size)


def parameter_counts(models):
    """Compare supplied real architectures to the first (baseline) model."""
    result, baseline = {}, None
    for name, model in models.items():
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if baseline is None:
            baseline = total
        result[name] = {"total": total, "trainable": trainable, "delta": total - baseline,
                        "delta_percent": 100 * (total - baseline) / baseline}
    return result


def fusion_weight_statistics(weights):
    """Offline regional subsets can be passed as [3,N]; no causal interpretation."""
    w = torch.as_tensor(weights, dtype=torch.float64)
    if w.ndim != 2 or w.shape[0] != 3:
        raise ValueError("expected [3,N] coefficient subset")
    if not torch.isfinite(w).all() or (w < 0).any() or (w > 1).any():
        raise ValueError("invalid coefficients")
    if not torch.allclose(w.sum(0), torch.ones(w.shape[1], dtype=w.dtype), atol=1e-6):
        raise ValueError("coefficients must sum to one")
    if w.shape[1] == 0:
        return {"mean": None, "std": None, "median": None, "IQR": None, "mean_entropy": None}
    q = torch.quantile(w, torch.tensor([.25, .5, .75], dtype=w.dtype), dim=1)
    return {"mean": w.mean(1).tolist(), "std": w.std(1, correction=0).tolist(),
            "median": q[1].tolist(), "IQR": (q[2] - q[0]).tolist(),
            "mean_entropy": float(-(w * w.clamp_min(1e-12).log()).sum(0).mean())}
