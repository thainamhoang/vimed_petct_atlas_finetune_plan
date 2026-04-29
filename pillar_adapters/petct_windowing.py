"""PET/CT windowing helpers for Pillar fine-tuning.

The preprocessing cache stores raw-ish CT plus normalized PET as two channels:
    x_raw[0] = CT in HU-like units after clipping
    x_raw[1] = PET normalized to [0, 1]

Atlas/Pillar CT checkpoints use 11 CT channels: ten anatomical windows plus
minmax. This module creates those 11 CT channels and appends PET as channel 12.
"""

from __future__ import annotations

from typing import Iterable

import torch


CT_WINDOWS = {
    "lung": {"center": -600.0, "width": 1500.0},
    "mediastinum": {"center": 50.0, "width": 400.0},
    "abdomen": {"center": 40.0, "width": 400.0},
    "liver": {"center": 80.0, "width": 150.0},
    "bone": {"center": 400.0, "width": 1800.0},
    "brain": {"center": 40.0, "width": 80.0},
    "subdural": {"center": 75.0, "width": 215.0},
    "stroke": {"center": 40.0, "width": 40.0},
    "temporal_bone": {"center": 600.0, "width": 2800.0},
    "soft_tissue": {"center": 50.0, "width": 350.0},
}

DEFAULT_CT_WINDOW_ORDER = tuple(CT_WINDOWS.keys()) + ("minmax",)


def _window_ct(ct: torch.Tensor, center: float, width: float) -> torch.Tensor:
    low = center - width / 2.0
    return torch.clamp((ct - low) / width, 0.0, 1.0)


def apply_ct_windows(
    ct: torch.Tensor,
    window_order: Iterable[str] = DEFAULT_CT_WINDOW_ORDER,
    minmax_min: float = -1024.0,
    minmax_max: float = 3071.0,
) -> torch.Tensor:
    """Apply Pillar/RAVE CT windows.

    Args:
        ct: Tensor shaped (B, 1, D, H, W) or (1, D, H, W).

    Returns:
        Tensor shaped (B, 11, D, H, W) or (11, D, H, W).
    """
    squeeze_batch = False
    if ct.ndim == 4:
        ct = ct.unsqueeze(0)
        squeeze_batch = True
    if ct.ndim != 5 or ct.shape[1] != 1:
        raise ValueError(f"Expected CT shape (B,1,D,H,W) or (1,D,H,W), got {tuple(ct.shape)}")

    windows = []
    for name in window_order:
        if name == "minmax":
            windows.append(torch.clamp((ct - minmax_min) / (minmax_max - minmax_min), 0.0, 1.0))
        else:
            spec = CT_WINDOWS[name]
            windows.append(_window_ct(ct, spec["center"], spec["width"]))
    out = torch.cat(windows, dim=1)
    return out.squeeze(0) if squeeze_batch else out


def make_petct_atlas_input(x_raw: torch.Tensor) -> torch.Tensor:
    """Convert raw two-channel PET/CT to Atlas-ready 12-channel input.

    Args:
        x_raw: Tensor shaped (B, 2, D, H, W) or (2, D, H, W).

    Returns:
        Tensor shaped (B, 12, D, H, W) or (12, D, H, W).
    """
    squeeze_batch = False
    if x_raw.ndim == 4:
        x_raw = x_raw.unsqueeze(0)
        squeeze_batch = True
    if x_raw.ndim != 5 or x_raw.shape[1] != 2:
        raise ValueError(f"Expected x_raw shape (B,2,D,H,W) or (2,D,H,W), got {tuple(x_raw.shape)}")

    ct = x_raw[:, 0:1].float()
    pet = torch.clamp(x_raw[:, 1:2].float(), 0.0, 1.0)
    out = torch.cat([apply_ct_windows(ct), pet], dim=1)
    return out.squeeze(0) if squeeze_batch else out

