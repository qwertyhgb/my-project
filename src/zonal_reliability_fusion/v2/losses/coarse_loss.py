"""Auxiliary Dice+BCE, with max-pooled positives and native main loss untouched."""
import torch
from torch import nn
from torch.nn import functional as F

COARSE_LOSS_WEIGHT = 0.5


class CoarseAuxiliaryLoss(nn.Module):
    def __init__(self, main_loss, network):
        super().__init__()
        self.main_loss = main_loss
        # Avoid registering the model as a loss child and duplicate checkpoint parameters.
        object.__setattr__(self, "network", network)

    def forward(self, output, target):
        loss = self.main_loss(output, target)
        if self.network.phase == "V2-P1":
            return loss
        logits = self.network.coarse_logits
        if logits is None:
            raise ValueError("coarse logits unavailable; forward must precede loss")
        full = target[0] if isinstance(target, (list, tuple)) else target
        if full.ndim != 5 or full.shape[1] != 1 or not ((full == 0) | (full == 1)).all():
            raise ValueError("coarse target must be binary lesion GT [B,1,Z,Y,X]")
        coarse = F.adaptive_max_pool3d(full.float(), logits.shape[2:])
        logits = logits.float()
        probability = logits.sigmoid()
        axes = (2, 3, 4)
        dice = (2 * (probability * coarse).sum(axes) + 1e-5) / (
            probability.sum(axes) + coarse.sum(axes) + 1e-5)
        auxiliary = F.binary_cross_entropy_with_logits(logits, coarse) + 1 - dice.mean()
        if not torch.isfinite(auxiliary):
            raise ValueError("non-finite coarse loss")
        return loss + COARSE_LOSS_WEIGHT * auxiliary
