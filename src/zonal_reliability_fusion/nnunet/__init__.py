"""Shared integration with fixed official nnU-Net v2.6.2.

trainers.py is the version registry: ACTIVE V2 / FROZEN V1 / supporting / LEGACY.
V2 hooks live in nnunet/v2; native bases, augmentation, runtime and prior
preprocessing remain shared. Training, optimizer, scheduler, checkpoints,
validation and sliding-window prediction remain native implementations.
"""

from __future__ import annotations

from zonal_reliability_fusion.nnunet.runtime import (
    DNA_REQUIRED_RANGE,
    FIXED_NNUNET_ROOT,
    PLAIN_CONV_UNET_DOTTED_PATH,
    PROJECT_ROOT,
    check_fixed_nnunet_runtime,
)

__all__ = (
    "DNA_REQUIRED_RANGE",
    "FIXED_NNUNET_ROOT",
    "PLAIN_CONV_UNET_DOTTED_PATH",
    "PROJECT_ROOT",
    "check_fixed_nnunet_runtime",
)
