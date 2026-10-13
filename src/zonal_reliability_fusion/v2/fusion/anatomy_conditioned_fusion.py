"""Separate named anatomy-conditioned mechanism; no fourth MRI encoder."""
from zonal_reliability_fusion.v2.anatomy.pyramid import ANATOMY_FEATURE_CHANNELS
from zonal_reliability_fusion.v2.fusion.adaptive_fusion import AdaptiveFusion


class AnatomyConditionedFusion(AdaptiveFusion):
    def __init__(self, channels):
        super().__init__(channels, context_channels=ANATOMY_FEATURE_CHANNELS)
