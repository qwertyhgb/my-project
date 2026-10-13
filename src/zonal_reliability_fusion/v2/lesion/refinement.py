"""Late local logit correction: spatial conv, soft candidate, zero-init output."""
import torch
from torch import nn
from torch.nn import functional as F

REFINER_CHANNELS = 8


class LocalMultimodalRefiner(nn.Module):
    def __init__(self, feature_channels, heads):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Conv3d(feature_channels, REFINER_CHANNELS, 1) for _ in range(3)
        ])
        self.hidden = nn.Sequential(
            nn.Conv3d(3 * REFINER_CHANNELS + heads + 3 + 1, REFINER_CHANNELS, 3, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        self.output = nn.Conv3d(REFINER_CHANNELS, heads, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, base, features, anatomy, coarse_logits):
        coarse = F.interpolate(coarse_logits.sigmoid(), size=base.shape[2:],
                               mode="trilinear", align_corners=False)
        candidate = torch.maximum(base.softmax(1)[:, 1:2], coarse).detach()
        local = [projection(feature) for projection, feature
                 in zip(self.projections, features, strict=True)]
        context = torch.cat([*local, base, anatomy, candidate], dim=1)
        return base + candidate * self.output(self.hidden(context))
