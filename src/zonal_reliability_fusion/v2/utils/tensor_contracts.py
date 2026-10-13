"""Fail-closed tensor contracts, independent of anatomy provenance validation."""
import torch


def check_input(x, anatomy_conditioned):
    channels = 6 if anatomy_conditioned else 3
    if x.ndim != 5 or x.shape[1] != channels or not x.is_floating_point():
        raise ValueError(f"expected floating [B,{channels},Z,Y,X]")
    if min(x.shape) < 1 or not torch.isfinite(x).all():
        raise ValueError("empty or non-finite MRI/prior input")
    if anatomy_conditioned:
        check_probabilities(x[:, 3:])


def check_probabilities(p):
    if p.ndim != 5 or p.shape[1] != 3 or not p.is_floating_point():
        raise ValueError("expected floating WG/PZ/TZ probabilities [B,3,Z,Y,X]")
    if not torch.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("predicted anatomy probabilities must be finite in [0,1]")


def check_modalities(features):
    if len(features) != 3 or any(f.shape != features[0].shape for f in features):
        raise ValueError("three modality features must have identical shapes")
    if any(not torch.isfinite(f).all() for f in features):
        raise ValueError("non-finite modality features")
