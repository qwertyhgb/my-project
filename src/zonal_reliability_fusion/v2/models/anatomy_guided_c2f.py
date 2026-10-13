"""V2 composition around native encoders/decoder; outputs stay nnU-Net compatible."""
import math

from torch import nn

from zonal_reliability_fusion.v2.anatomy.pyramid import ANATOMY_FEATURE_CHANNELS, AnatomyPyramid
from zonal_reliability_fusion.v2.encoders.multi_encoder import build_native_branches
from zonal_reliability_fusion.v2.fusion.adaptive_fusion import CONTROLLER_CHANNELS, AdaptiveFusion
from zonal_reliability_fusion.v2.fusion.anatomy_conditioned_fusion import AnatomyConditionedFusion
from zonal_reliability_fusion.v2.fusion.simple_fusion import ConcatFusion, SimpleFusion
from zonal_reliability_fusion.v2.lesion.coarse_head import CoarseLesionHead
from zonal_reliability_fusion.v2.lesion.refinement import REFINER_CHANNELS, LocalMultimodalRefiner
from zonal_reliability_fusion.v2.utils.tensor_contracts import check_input


class AnatomyGuidedC2F(nn.Module):
    def __init__(self, class_name, kwargs, required_imports, heads, deep_supervision=True,
                 *, fusion_type="simple", coarse_head=False, refiner=False):
        super().__init__()
        if fusion_type not in ("simple", "adaptive", "anatomy"):
            raise ValueError("unknown V2 fusion type")
        if heads != 2 or (coarse_head and fusion_type != "anatomy") or (refiner and not coarse_head):
            raise ValueError("V2 requires binary lesion heads and ordered mechanism ablations")
        self.encoders, self.decoder = build_native_branches(
            class_name, kwargs, required_imports, heads, deep_supervision)
        self.fusion_type = fusion_type
        self.anatomy_conditioned = fusion_type == "anatomy"
        widths = self.encoders[0].output_channels
        self.selected_scales = tuple(range(max(1, len(widths) - 3), len(widths)))
        self.fusions = nn.ModuleDict()
        if fusion_type != "simple":
            fusion = AnatomyConditionedFusion if self.anatomy_conditioned else AdaptiveFusion
            self.fusions.update({str(s): fusion(widths[s]) for s in self.selected_scales})
        self.simple = SimpleFusion()
        self.anatomy_pyramid = AnatomyPyramid(len(self.selected_scales)) if self.anatomy_conditioned else None
        self.coarse_head = CoarseLesionHead(widths[-1]) if coarse_head else None
        self.refiner = LocalMultimodalRefiner(widths[0], heads) if refiner else None
        self.phase = "V2-P3" if refiner else "V2-P1" if fusion_type == "simple" else "V2-P2"
        self.coarse_logits = None
        self.export_fusion_weights = False
        self.last_fusion_weights = {}

    def set_phase(self, phase):
        if phase not in ("V2-P1", "V2-P2", "V2-P3"):
            raise ValueError("unknown training phase")
        self.phase = phase

    def forward(self, x):
        check_input(x, self.anatomy_conditioned)
        # Reject non-divisible shapes before native transpose-conv/skip concatenation.
        divisor = [math.prod(stride[a] for stride in self.encoders[0].strides) for a in range(3)]
        if any(n % d for n, d in zip(x.shape[2:], divisor, strict=True)):
            raise ValueError(f"patch must be divisible by encoder strides {divisor}")
        skips = [encoder(x[:, i:i+1]) for i, encoder in enumerate(self.encoders)]
        self.last_fusion_weights = {}
        self.coarse_logits = None
        contexts = {}
        if self.anatomy_pyramid is not None and self.phase != "V2-P1":
            shapes = [skips[0][s].shape[2:] for s in self.selected_scales]
            contexts = dict(zip(self.selected_scales, self.anatomy_pyramid(x[:, 3:], shapes), strict=True))
        fused = []
        for s in range(len(skips[0])):
            features = [branch[s] for branch in skips]
            if str(s) in self.fusions and self.phase != "V2-P1":
                feature, weights = self.fusions[str(s)](features, contexts.get(s))
                if self.export_fusion_weights and weights is not None:
                    self.last_fusion_weights[s] = weights.detach().cpu()
            else:
                feature = self.simple(features)
            fused.append(feature)
        output = self.decoder(fused)
        if self.coarse_head is not None and self.phase != "V2-P1":
            self.coarse_logits = self.coarse_head(fused[-1])
        if self.refiner is not None and self.phase == "V2-P3":
            base = output[0] if isinstance(output, list) else output
            refined = self.refiner(base, [branch[0] for branch in skips], x[:, 3:], self.coarse_logits)
            output = [refined, *output[1:]] if isinstance(output, list) else refined
        return output

    def compute_conv_feature_map_size(self, input_size):
        """Count conv outputs incl. all branches/extensions; not a peak-memory estimate."""
        total = sum(int(e.compute_conv_feature_map_size(input_size)) for e in self.encoders)
        total += int(self.decoder.compute_conv_feature_map_size(input_size))
        shape = list(input_size)
        for s, stride in enumerate(self.encoders[0].strides):
            shape = [n // d for n, d in zip(shape, stride, strict=True)]
            volume = math.prod(shape)
            if str(s) in self.fusions:
                total += volume * (CONTROLLER_CHANNELS + 3)
                if self.anatomy_pyramid is not None:
                    total += volume * ANATOMY_FEATURE_CHANNELS
        if self.coarse_head is not None:
            total += math.prod(shape)
        if self.refiner is not None:
            stride = self.encoders[0].strides[0]
            volume = math.prod(n // d for n, d in zip(input_size, stride, strict=True))
            total += volume * (4 * REFINER_CHANNELS + 2)
        return total


class ConcatFusionUNet(AnatomyGuidedC2F):
    """Independent B/C/D family; identical all-scale learned projection and native decoder."""
    def __init__(self, class_name, kwargs, required_imports, heads, deep_supervision=True,
                 *, fusion_type="concat"):
        if fusion_type not in ("concat", "concat_adaptive", "concat_anatomy"):
            raise ValueError("unknown concat fusion variant")
        super().__init__(class_name, kwargs, required_imports, heads, deep_supervision,
                         fusion_type="simple")
        self.fusion_type = fusion_type
        self.anatomy_conditioned = fusion_type == "concat_anatomy"
        encoder = self.encoders[0]
        self.fusions = nn.ModuleDict()
        # Build all common projections before optional branches consume RNG.
        for s, width in enumerate(encoder.output_channels):
            self.fusions[str(s)] = ConcatFusion(
                width, norm_op=encoder.norm_op, norm_op_kwargs=encoder.norm_op_kwargs,
                nonlin=encoder.nonlin, nonlin_kwargs=encoder.nonlin_kwargs,
                conv_bias=encoder.conv_bias,
            )
        if fusion_type != "concat":
            for s in self.selected_scales:
                width = encoder.output_channels[s]
                policy = AnatomyConditionedFusion(width) if self.anatomy_conditioned else AdaptiveFusion(width)
                self.fusions[str(s)].set_policy(policy)
        self.anatomy_pyramid = AnatomyPyramid(len(self.selected_scales)) if self.anatomy_conditioned else None
        self.phase = "V2-P2"

    def set_phase(self, phase):
        if phase != "V2-P2":
            raise ValueError("concat family uses fixed all-on fusion without phase scheduling")
        super().set_phase(phase)

    def compute_conv_feature_map_size(self, input_size):
        total = sum(int(e.compute_conv_feature_map_size(input_size)) for e in self.encoders)
        total += int(self.decoder.compute_conv_feature_map_size(input_size))
        shape = list(input_size)
        for s, stride in enumerate(self.encoders[0].strides):
            shape = [n // d for n, d in zip(shape, stride, strict=True)]
            volume = math.prod(shape)
            total += volume * self.encoders[0].output_channels[s]
            if self.fusions[str(s)].policy is not None:
                total += volume * (CONTROLLER_CHANNELS + 3)
                if self.anatomy_pyramid is not None:
                    total += volume * ANATOMY_FEATURE_CHANNELS
        return total
