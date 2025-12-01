"""
Auto-correction service wrapping gsmod 0.1.4 features.

Provides both GPU (torch) and CPU (gsmod) implementations:
- auto_enhance: Combined enhancement (exposure + contrast + white balance)
- auto_contrast: Percentile-based histogram stretching (Photoshop-style)
- auto_exposure: 18% gray midtone targeting
- auto_white_balance: Gray World and White Patch methods
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from gsmod import ColorValues

if TYPE_CHECKING:
    from src.domain.entities import GSTensor

logger = logging.getLogger(__name__)


# =============================================================================
# GPU (Torch) Implementation - Fast path
# =============================================================================


def _auto_exposure_torch(sh0: torch.Tensor, target_midtone: float = 0.45) -> ColorValues:
    """GPU auto-exposure: adjust brightness to target midtone."""
    current_mean = sh0.mean()
    brightness = (target_midtone / (current_mean + 1e-6)).clamp(0.3, 3.0)
    return ColorValues(brightness=brightness.item())


def _auto_contrast_torch(
    sh0: torch.Tensor, clip_percent: float = 0.1, per_channel: bool = False
) -> ColorValues:
    """GPU auto-contrast: percentile-based histogram stretching."""
    low_p = clip_percent / 100.0
    high_p = 1.0 - low_p

    if per_channel:
        # Per-channel stretching (more aggressive)
        p_low = torch.quantile(sh0, low_p, dim=0)
        p_high = torch.quantile(sh0, high_p, dim=0)
        current_range = (p_high - p_low).mean()
    else:
        # Global stretching
        p_low = torch.quantile(sh0, low_p)
        p_high = torch.quantile(sh0, high_p)
        current_range = p_high - p_low

    # Target ~90% of dynamic range
    target_range = 0.9
    contrast = (target_range / (current_range + 1e-6)).clamp(0.5, 3.0)
    return ColorValues(contrast=contrast.item())


def _auto_white_balance_torch(
    sh0: torch.Tensor, method: str = "gray_world"
) -> ColorValues:
    """GPU auto white-balance: Gray World or White Patch."""
    if method == "white_patch":
        # Use brightest pixels (top 1%)
        brightness = sh0.sum(dim=-1)
        k = max(1, int(sh0.shape[0] * 0.01))
        _, indices = brightness.topk(k)
        reference = sh0[indices].mean(dim=0)
    else:  # gray_world
        reference = sh0.mean(dim=0)

    # Compute channel gains to neutralize
    target_gray = reference.mean()
    r_gain = target_gray / (reference[0] + 1e-6)
    g_gain = target_gray / (reference[1] + 1e-6)
    b_gain = target_gray / (reference[2] + 1e-6)

    # Convert RGB gains to temperature/tint (approximate)
    # Temperature: red vs blue balance
    # Tint: green vs magenta balance
    temperature = ((r_gain - b_gain) / (r_gain + b_gain + 1e-6)).clamp(-1, 1) * 0.5
    tint = ((g_gain - (r_gain + b_gain) / 2) / (g_gain + 1e-6)).clamp(-1, 1) * 0.3

    return ColorValues(temperature=temperature.item(), tint=tint.item())


def _auto_enhance_torch(
    sh0: torch.Tensor, strength: float = 1.0, preserve_warmth: bool = True
) -> ColorValues:
    """GPU auto-enhance: combined exposure + contrast + white balance."""
    # Compute individual corrections
    exposure = _auto_exposure_torch(sh0)
    contrast = _auto_contrast_torch(sh0, clip_percent=0.1)
    wb = _auto_white_balance_torch(sh0, method="gray_world")

    # Blend based on strength
    brightness = 1.0 + (exposure.brightness - 1.0) * strength
    contrast_val = 1.0 + (contrast.contrast - 1.0) * strength

    # Optionally reduce white balance correction to preserve warmth
    temp_scale = 0.5 if preserve_warmth else 1.0
    temperature = wb.temperature * strength * temp_scale
    tint = wb.tint * strength * temp_scale

    return ColorValues(
        brightness=brightness,
        contrast=contrast_val,
        temperature=temperature,
        tint=tint,
    )


# Dispatch table for torch implementations
_TORCH_CORRECTION_FUNCS = {
    "auto_enhance": lambda sh0: _auto_enhance_torch(sh0, strength=1.0, preserve_warmth=True),
    "auto_contrast": lambda sh0: _auto_contrast_torch(sh0, clip_percent=0.1, per_channel=False),
    "auto_exposure": lambda sh0: _auto_exposure_torch(sh0, target_midtone=0.45),
    "auto_wb_gray": lambda sh0: _auto_white_balance_torch(sh0, method="gray_world"),
    "auto_wb_white": lambda sh0: _auto_white_balance_torch(sh0, method="white_patch"),
}


# =============================================================================
# CPU (gsmod) Implementation - Fallback for accuracy
# =============================================================================


def _apply_auto_correction_cpu(sh0_numpy, correction_type: str) -> ColorValues:
    """CPU fallback using gsmod's GSDataPro-based auto-correction."""
    import numpy as np
    from gsmod import GSDataPro
    from gsmod.color.auto import (
        auto_contrast,
        auto_enhance,
        auto_exposure,
        auto_white_balance,
    )

    n = sh0_numpy.shape[0]
    data_pro = GSDataPro.from_arrays(
        means=np.zeros((n, 3), dtype=np.float32),
        scales=np.ones((n, 3), dtype=np.float32),
        quats=np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)).astype(np.float32),
        opacities=np.ones((n, 1), dtype=np.float32),
        sh0=sh0_numpy.astype(np.float32),
        shN=None,
    )

    funcs = {
        "auto_enhance": lambda d: auto_enhance(d, strength=1.0, preserve_warmth=True),
        "auto_contrast": lambda d: auto_contrast(d, clip_percent=0.1, per_channel=False),
        "auto_exposure": lambda d: auto_exposure(d, target_midtone=0.45, clip_percent=1.0),
        "auto_wb_gray": lambda d: auto_white_balance(d, clip_percent=1.0, method="gray_world"),
        "auto_wb_white": lambda d: auto_white_balance(d, clip_percent=1.0, method="white_patch"),
    }

    result = funcs[correction_type](data_pro)
    return result.to_color_values()


# =============================================================================
# Public API
# =============================================================================


def apply_auto_correction(
    gaussians: "GSTensor",
    correction_type: str,
    use_gpu: bool = True,
) -> ColorValues:
    """
    Apply auto-correction and return ColorValues.

    Parameters
    ----------
    gaussians : GSTensor
        Gaussian tensor data containing sh0 colors
    correction_type : str
        One of: "auto_enhance", "auto_contrast", "auto_exposure",
                "auto_wb_gray", "auto_wb_white"
    use_gpu : bool
        If True, use fast torch-based implementation (default).
        If False, use gsmod's CPU implementation (more accurate).

    Returns
    -------
    ColorValues
        Computed optimal color parameters
    """
    if correction_type not in _TORCH_CORRECTION_FUNCS:
        raise ValueError(
            f"Unknown correction type: {correction_type}. "
            f"Valid types: {list(_TORCH_CORRECTION_FUNCS.keys())}"
        )

    # Get sh0 data
    sh0 = gaussians.sh0
    if not isinstance(sh0, torch.Tensor):
        sh0 = torch.tensor(sh0, dtype=torch.float32)

    if use_gpu and sh0.is_cuda:
        # Fast GPU path
        return _TORCH_CORRECTION_FUNCS[correction_type](sh0.float())
    else:
        # CPU fallback (more accurate but slower)
        sh0_numpy = sh0.detach().cpu().numpy()
        return _apply_auto_correction_cpu(sh0_numpy, correction_type)
