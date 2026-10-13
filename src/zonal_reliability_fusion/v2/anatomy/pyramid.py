"""Resize soft overlapping heads to selected encoder grids, then encode context."""
from torch import nn
from torch.nn import functional as F

from zonal_reliability_fusion.v2.utils.tensor_contracts import check_probabilities

ANATOMY_FEATURE_CHANNELS = 8


class AnatomyPyramid(nn.Module):
    def __init__(self, scales):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Sequential(nn.Conv3d(3, ANATOMY_FEATURE_CHANNELS, 1), nn.LeakyReLU())
            for _ in range(scales)
        ])

    def forward(self, probabilities, shapes):
        check_probabilities(probabilities)
        if len(shapes) != len(self.projections):
            raise ValueError("anatomy pyramid scale count mismatch")
        return [projection(F.interpolate(probabilities, size=shape, mode="trilinear",
                                         align_corners=False))
                for projection, shape in zip(self.projections, shapes, strict=True)]
