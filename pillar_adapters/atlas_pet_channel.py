"""Utilities for adapting an 11-channel CT Atlas checkpoint to 12-channel PET/CT."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


def find_first_conv3d_with_in_channels(module: nn.Module, in_channels: int = 11) -> tuple[str, nn.Conv3d]:
    """Find the first Conv3d that likely acts as a 3D patch embedding."""
    for name, child in module.named_modules():
        if isinstance(child, nn.Conv3d) and child.in_channels == in_channels:
            return name, child
    raise ValueError(f"No nn.Conv3d with in_channels={in_channels} found")


def replace_module(root: nn.Module, dotted_name: str, new_module: nn.Module) -> None:
    """Replace a nested module by dotted module path."""
    parts = dotted_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def expand_first_ct_patch_embed_to_petct(
    model: nn.Module,
    old_channels: int = 11,
    new_channels: int = 12,
    pet_init: str = "zero",
    module_name: Optional[str] = None,
) -> str:
    """Expand the first 11-channel Conv3d to accept an extra PET channel.

    The original CT weights are copied unchanged. The new PET channel is
    zero-initialized by default so the model starts with CT-only behavior.

    Returns:
        The dotted module name that was replaced.
    """
    if new_channels <= old_channels:
        raise ValueError("new_channels must be greater than old_channels")

    if module_name is None:
        module_name, old = find_first_conv3d_with_in_channels(model, old_channels)
    else:
        old = model
        for part in module_name.split("."):
            old = getattr(old, part)
        if not isinstance(old, nn.Conv3d):
            raise TypeError(f"{module_name} is not an nn.Conv3d")

    new = nn.Conv3d(
        in_channels=new_channels,
        out_channels=old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        dilation=old.dilation,
        groups=old.groups,
        bias=old.bias is not None,
        padding_mode=old.padding_mode,
        device=old.weight.device,
        dtype=old.weight.dtype,
    )

    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, :old_channels].copy_(old.weight)
        if pet_init == "mean":
            new.weight[:, old_channels:new_channels].copy_(old.weight.mean(dim=1, keepdim=True))
        elif pet_init == "zero":
            pass
        else:
            raise ValueError("pet_init must be 'zero' or 'mean'")
        if old.bias is not None:
            new.bias.copy_(old.bias)

    replace_module(model, module_name, new)
    return module_name

