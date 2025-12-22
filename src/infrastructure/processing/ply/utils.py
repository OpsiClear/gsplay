"""
Utility functions for PLY file operations.

Spherical harmonics conversion utilities for Gaussian color handling.
"""

import numpy as np
import torch


# Spherical Harmonics conversion constant
SH_C0 = 0.28209479177387814  # sqrt(1/(4*pi))


def sh2rgb(sh: torch.Tensor) -> torch.Tensor:
    """Convert Spherical Harmonics DC coefficient to RGB.

    Args:
        sh: SH tensor (N, 3) - DC coefficients

    Returns:
        RGB tensor (N, 3) in [0, 1] range
    """
    return sh * SH_C0 + 0.5


def rgb2sh(rgb: torch.Tensor) -> torch.Tensor:
    """Convert RGB to Spherical Harmonics DC coefficient.

    Args:
        rgb: RGB tensor (N, 3) in [0, 1] range

    Returns:
        SH tensor (N, 3) - DC coefficients
    """
    return (rgb - 0.5) / SH_C0


# NumPy versions for compatibility
def sh2rgb_np(sh: np.ndarray) -> np.ndarray:
    """NumPy version of sh2rgb."""
    return sh * SH_C0 + 0.5


def rgb2sh_np(rgb: np.ndarray) -> np.ndarray:
    """NumPy version of rgb2sh."""
    return (rgb - 0.5) / SH_C0
