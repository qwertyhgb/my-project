"""Binary representation-level lesion evidence from fused bottleneck."""
from torch import nn


class CoarseLesionHead(nn.Conv3d):
    def __init__(self, channels):
        super().__init__(channels, 1, kernel_size=1)
