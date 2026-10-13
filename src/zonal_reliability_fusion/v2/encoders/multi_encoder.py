"""Compose native plans-built encoders; never copy third-party implementations."""
from torch import nn


def build_native_branches(class_name, kwargs, required_imports, heads, deep_supervision):
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

    if class_name != "dynamic_network_architectures.architectures.unet.PlainConvUNet":
        raise ValueError("V2 first version requires plans-defined PlainConvUNet")
    if kwargs.get("conv_op") != "torch.nn.modules.conv.Conv3d":
        raise ValueError("V2 requires 3D plans")
    branches = nn.ModuleList()
    decoder = None
    for _ in range(3):
        native = get_network_from_plans(
            class_name, kwargs, required_imports, 1, heads,
            allow_init=True, deep_supervision=deep_supervision,
        )
        branches.append(native.encoder)
        if decoder is None:
            decoder = native.decoder
    return branches, decoder
