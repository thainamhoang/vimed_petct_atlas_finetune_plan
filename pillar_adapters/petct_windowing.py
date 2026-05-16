"""PET/CT windowing helpers for Pillar fine-tuning and dual-stream report generation.

The preprocessing cache stores raw-ish CT plus normalized PET as two channels:
    x_raw[0] = CT in HU-like units after clipping
    x_raw[1] = PET normalized to [0, 1]

This module supports two workflows:
1. Pillar classifier compatibility: 11 CT windows + 1 PET channel = 12 channels.
2. Dual-stream report generation: separate CT and PET window tensors.
"""

from __future__ import annotations

from collections import OrderedDict
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

# Clinically sensible CT windows for the first dual-stream chest baseline.
DUAL_STREAM_CT_WINDOWS = OrderedDict(
    [
        ("wide", {"lo": -1024.0, "hi": 3071.0}),
        ("lung", {"center": -600.0, "width": 1500.0}),
        ("mediastinum", {"center": 40.0, "width": 400.0}),
        ("soft_tissue", {"center": 50.0, "width": 350.0}),
        ("bone", {"center": 400.0, "width": 1800.0}),
        ("liver", {"center": 60.0, "width": 160.0}),
    ]
)

# PET is already normalized to [0,1]. These windows emphasize progressively
# broader uptake ranges while keeping each channel in [0,1].
DUAL_STREAM_PET_WINDOWS = OrderedDict(
    [
        ("pet_low", {"lo": 0.00, "hi": 0.25}),
        ("pet_mid", {"lo": 0.00, "hi": 0.50}),
        ("pet_high", {"lo": 0.00, "hi": 0.75}),
        ("pet_full", {"lo": 0.00, "hi": 1.00}),
    ]
)


def _window_ct(ct: torch.Tensor, center: float, width: float) -> torch.Tensor:
    low = center - width / 2.0
    return torch.clamp((ct - low) / width, 0.0, 1.0)


def _window_range(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return torch.clamp((x - lo) / (hi - lo + 1e-6), 0.0, 1.0)


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


def _ensure_volume(x: torch.Tensor, expected_channels: int = 1) -> tuple[torch.Tensor, bool]:
    squeeze_batch = False
    if x.ndim == 3:
        x = x.unsqueeze(0)
    if x.ndim == 4:
        x = x.unsqueeze(0)
        squeeze_batch = True
    if x.ndim != 5 or x.shape[1] != expected_channels:
        raise ValueError(f"Expected shape (B,{expected_channels},D,H,W) or ({expected_channels},D,H,W) or (D,H,W), got {tuple(x.shape)}")
    return x.float(), squeeze_batch


def make_ct_window_tensor(
    ct: torch.Tensor,
    window_specs: OrderedDict[str, dict] = DUAL_STREAM_CT_WINDOWS,
) -> torch.Tensor:
    """Create multi-window chest CT channels for the dual-stream model.

    Input may be (D,H,W), (1,D,H,W), or (B,1,D,H,W).
    Returns (C,D,H,W) or (B,C,D,H,W).
    """
    ct, squeeze_batch = _ensure_volume(ct, expected_channels=1)
    outputs = []
    for spec in window_specs.values():
        if "center" in spec:
            outputs.append(_window_ct(ct, spec["center"], spec["width"]))
        else:
            outputs.append(_window_range(ct, spec["lo"], spec["hi"]))
    out = torch.cat(outputs, dim=1)
    return out.squeeze(0) if squeeze_batch else out


def make_pet_window_tensor(
    pet: torch.Tensor,
    window_specs: OrderedDict[str, dict] = DUAL_STREAM_PET_WINDOWS,
) -> torch.Tensor:
    """Create PET uptake windows for the dual-stream model.

    PET is expected to already be normalized to [0,1].
    Input may be (D,H,W), (1,D,H,W), or (B,1,D,H,W).
    Returns (C,D,H,W) or (B,C,D,H,W).
    """
    pet, squeeze_batch = _ensure_volume(torch.clamp(pet, 0.0, 1.0), expected_channels=1)
    outputs = [_window_range(pet, spec["lo"], spec["hi"]) for spec in window_specs.values()]
    out = torch.cat(outputs, dim=1)
    return out.squeeze(0) if squeeze_batch else out


def make_dual_stream_window_inputs(x_raw: torch.Tensor) -> dict[str, torch.Tensor]:
    """Convert a raw two-channel PET/CT tensor into separate CT and PET windows.

    Args:
        x_raw: Tensor shaped (2,D,H,W) or (B,2,D,H,W).

    Returns:
        {
            "ct_windows": (C_ct,D,H,W) or (B,C_ct,D,H,W),
            "pet_windows": (C_pet,D,H,W) or (B,C_pet,D,H,W),
        }
    """
    squeeze_batch = False
    if x_raw.ndim == 4:
        x_raw = x_raw.unsqueeze(0)
        squeeze_batch = True
    if x_raw.ndim != 5 or x_raw.shape[1] != 2:
        raise ValueError(f"Expected x_raw shape (B,2,D,H,W) or (2,D,H,W), got {tuple(x_raw.shape)}")

    ct = x_raw[:, 0:1].float()
    pet = torch.clamp(x_raw[:, 1:2].float(), 0.0, 1.0)
    ct_windows = make_ct_window_tensor(ct)
    pet_windows = make_pet_window_tensor(pet)
    if squeeze_batch:
        ct_windows = ct_windows.squeeze(0)
        pet_windows = pet_windows.squeeze(0)
    return {"ct_windows": ct_windows, "pet_windows": pet_windows}
