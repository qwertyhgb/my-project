"""Location-wise modality softmax; anatomy enters only the policy controller."""
import torch
from torch import nn

from zonal_reliability_fusion.v2.utils.tensor_contracts import check_modalities

CONTROLLER_CHANNELS = 16


class AdaptiveFusion(nn.Module):
    def __init__(self, channels, context_channels=0):
        super().__init__()
        self.context_channels = context_channels
        self.controller = nn.Sequential(
            nn.Conv3d(3 * channels + context_channels, CONTROLLER_CHANNELS, 1),
            nn.LeakyReLU(inplace=True), nn.Conv3d(CONTROLLER_CHANNELS, 3, 1),
        )

    def forward(self, features, context=None):
        check_modalities(features)
        values = list(features)
        if self.context_channels:
            if (context is None or context.shape[:2] != (features[0].shape[0], self.context_channels)
                    or context.shape[2:] != features[0].shape[2:] or not torch.isfinite(context).all()):
                raise ValueError("missing or invalid anatomy pyramid context")
            values.append(context)
        elif context is not None:
            raise ValueError("image-adaptive fusion must not receive anatomy")
        weights = self.controller(torch.cat(values, dim=1)).softmax(dim=1)
        fused = sum(weights[:, i:i+1] * f for i, f in enumerate(features))
        return fused, weights
