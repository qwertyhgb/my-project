"""Preserved mean control and common learned concat projection for the new V2 family."""
import torch
from torch import nn

from zonal_reliability_fusion.v2.utils.tensor_contracts import check_modalities


class SimpleFusion(nn.Module):
    def forward(self, features):
        check_modalities(features)
        return sum(features) / 3


CONCAT_KERNEL_SIZE = (3, 3, 3)
POLICY_FEATURE_SCALE = 3.0


class ConcatFusion(nn.Module):
    """Common learned projection; optional soft policy rescales modalities before concat."""
    def __init__(self, channels, *, norm_op, norm_op_kwargs, nonlin, nonlin_kwargs,
                 conv_bias, policy=None):
        super().__init__()
        from nnunetv2.utilities.network_initialization import InitWeights_He

        self.channels = channels
        self.block = nn.Sequential(
            nn.Conv3d(3 * channels, channels, CONCAT_KERNEL_SIZE, padding=1, bias=conv_bias),
            norm_op(channels, **(norm_op_kwargs or {})) if norm_op else nn.Identity(),
            nonlin(**(nonlin_kwargs or {})) if nonlin else nn.Identity(),
        )
        self.block.apply(InitWeights_He(1e-2))
        self.policy = None
        if policy is not None:
            self.set_policy(policy)

    def set_policy(self, policy):
        if self.policy is not None:
            raise ValueError("concat policy is already configured")
        self.policy = policy
        # Uniform weights reproduce the unweighted baseline exactly: 3*(1/3)=1.
        nn.init.zeros_(policy.controller[-1].weight)
        nn.init.zeros_(policy.controller[-1].bias)

    def forward(self, features, context=None):
        check_modalities(features)
        if any(f.ndim != 5 or f.shape[1] != self.channels or not f.is_floating_point()
               for f in features):
            raise ValueError("concat fusion requires floating 3D features with plans-defined channels")
        weights = None
        if self.policy is not None:
            # Reuse the existing controller/contract; its mean output is not the fused skip.
            _, weights = self.policy(features, context)
            features = [POLICY_FEATURE_SCALE * weights[:, i:i+1] * f
                        for i, f in enumerate(features)]
        elif context is not None:
            raise ValueError("unconditioned concat projection must not receive anatomy")
        return self.block(torch.cat(features, dim=1)), weights
