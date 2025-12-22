"""Gaussian data interpolation for continuous time support.

This module provides interpolation functions for blending between
discrete keyframes of Gaussian data. It handles the specific requirements
of 3D Gaussian Splatting data:

- Positions (means): Linear interpolation
- Scales: Log-space interpolation for stability
- Quaternions: Spherical linear interpolation (SLERP)
- Opacities and colors: Linear interpolation

Example
-------
>>> from src.domain.interpolation import interpolate_gaussian_data
>>> # Blend between two keyframes at t=0.5
>>> blended = interpolate_gaussian_data(frame0, frame1, t=0.5)
"""

from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from src.domain.data import GaussianData


class InterpolationMethod(Enum):
    """Available interpolation methods."""

    NEAREST = auto()  # Snap to nearest keyframe
    LINEAR = auto()  # Linear interpolation (lerp) with SLERP for quats
    CUBIC = auto()  # Cubic interpolation (Catmull-Rom) - future


def slerp_quaternions(
    q0: np.ndarray,
    q1: np.ndarray,
    t: float,
) -> np.ndarray:
    """Spherical linear interpolation for quaternions.

    Interpolates between two quaternion arrays using SLERP, which produces
    smooth rotational motion without distortion.

    Parameters
    ----------
    q0 : np.ndarray
        Start quaternions [N, 4] in wxyz format
    q1 : np.ndarray
        End quaternions [N, 4] in wxyz format
    t : float
        Interpolation factor in [0, 1]

    Returns
    -------
    np.ndarray
        Interpolated quaternions [N, 4] in wxyz format

    Notes
    -----
    - Handles quaternion hemisphere consistency (shortest path)
    - Falls back to linear interpolation for nearly parallel quaternions
    - Output is normalized
    """
    # Ensure quaternions are in the same hemisphere (shortest path)
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)
    dot = np.abs(dot)

    # Threshold for linear interpolation (nearly parallel quaternions)
    linear_threshold = 0.9995
    linear_mask = dot > linear_threshold

    # Compute angle theta
    # Clamp dot to avoid numerical issues with arccos
    dot_clamped = np.clip(dot, -1.0, 1.0)
    theta = np.arccos(dot_clamped)
    sin_theta = np.sin(theta)

    # Avoid division by zero for very small angles
    safe_sin_theta = np.where(np.abs(sin_theta) < 1e-8, 1.0, sin_theta)

    # SLERP weights
    w0 = np.sin((1 - t) * theta) / safe_sin_theta
    w1 = np.sin(t * theta) / safe_sin_theta

    # Compute interpolated quaternion
    slerp_result = w0 * q0 + w1 * q1

    # Linear interpolation for nearly parallel quaternions
    linear_result = (1 - t) * q0 + t * q1

    # Use linear where appropriate
    result = np.where(linear_mask, linear_result, slerp_result)

    # Normalize to unit quaternion
    norm = np.linalg.norm(result, axis=-1, keepdims=True)
    result = result / np.maximum(norm, 1e-8)

    return result


def lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Linear interpolation between two arrays.

    Parameters
    ----------
    a : np.ndarray
        Start values
    b : np.ndarray
        End values
    t : float
        Interpolation factor in [0, 1]

    Returns
    -------
    np.ndarray
        Interpolated values: (1-t)*a + t*b
    """
    return (1 - t) * a + t * b


def lerp_log_space(
    a: np.ndarray,
    b: np.ndarray,
    t: float,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Linear interpolation in log space.

    More stable for scale values that can span multiple orders of magnitude.
    Performs: exp((1-t)*log(a) + t*log(b))

    Parameters
    ----------
    a : np.ndarray
        Start values (must be positive)
    b : np.ndarray
        End values (must be positive)
    t : float
        Interpolation factor in [0, 1]
    epsilon : float
        Minimum value to avoid log(0)

    Returns
    -------
    np.ndarray
        Interpolated values
    """
    log_a = np.log(np.maximum(a, epsilon))
    log_b = np.log(np.maximum(b, epsilon))
    return np.exp(lerp(log_a, log_b, t))


def interpolate_gaussian_data(
    data0: GaussianData,
    data1: GaussianData,
    t: float,
    method: InterpolationMethod = InterpolationMethod.LINEAR,
) -> GaussianData:
    """Interpolate between two GaussianData instances.

    Blends all Gaussian properties between two keyframes using
    appropriate interpolation methods for each property type.

    Parameters
    ----------
    data0 : GaussianData
        Start keyframe data
    data1 : GaussianData
        End keyframe data
    t : float
        Interpolation factor in [0, 1]
        0.0 = data0, 1.0 = data1
    method : InterpolationMethod
        Interpolation method (default LINEAR)

    Returns
    -------
    GaussianData
        Interpolated Gaussian data

    Raises
    ------
    ValueError
        If data0 and data1 have different numbers of Gaussians

    Notes
    -----
    Property-specific interpolation:
    - means: Linear interpolation
    - scales: Log-space interpolation (more stable for varying sizes)
    - quats: Spherical linear interpolation (SLERP)
    - opacities: Linear interpolation
    - sh0: Linear interpolation (colors)
    - shN: Linear interpolation (higher-order SH, if present)

    Both inputs must have the same number of Gaussians. Topology changes
    (different Gaussian counts between frames) require higher-level
    correspondence handling.
    """
    from src.domain.data import GaussianData

    # Handle edge cases
    if t <= 0.0:
        return data0.clone() if hasattr(data0, "clone") else _clone_gaussian_data(data0)
    if t >= 1.0:
        return data1.clone() if hasattr(data1, "clone") else _clone_gaussian_data(data1)

    # NEAREST method: snap to closest keyframe
    if method == InterpolationMethod.NEAREST:
        if t < 0.5:
            return data0.clone() if hasattr(data0, "clone") else _clone_gaussian_data(data0)
        else:
            return data1.clone() if hasattr(data1, "clone") else _clone_gaussian_data(data1)

    # Ensure same number of Gaussians
    if data0.n_gaussians != data1.n_gaussians:
        raise ValueError(
            f"Cannot interpolate: different Gaussian counts "
            f"({data0.n_gaussians} vs {data1.n_gaussians}). "
            f"Topology changes require correspondence handling."
        )

    # Ensure CPU data is available
    data0._ensure_cpu()
    data1._ensure_cpu()

    # Interpolate each property
    # Positions: linear interpolation
    means = lerp(data0.means, data1.means, t)

    # Scales: log-space interpolation for stability
    scales = lerp_log_space(data0.scales, data1.scales, t)

    # Quaternions: SLERP
    quats = slerp_quaternions(data0.quats, data1.quats, t)

    # Opacities: linear interpolation
    opacities = lerp(data0.opacities, data1.opacities, t)

    # Colors (SH0): linear interpolation
    sh0 = lerp(data0.sh0, data1.sh0, t)

    # Higher-order SH: linear interpolation if present in both
    shN = None
    if data0.shN is not None and data1.shN is not None:
        if data0.shN.shape == data1.shN.shape:
            shN = lerp(data0.shN, data1.shN, t)
        # If shapes differ, drop higher-order SH

    return GaussianData(
        means=means,
        scales=scales,
        quats=quats,
        opacities=opacities,
        sh0=sh0,
        shN=shN,
        format_info=data0.format_info,  # Inherit format from first frame
        source_path=data0.source_path,
    )


def _clone_gaussian_data(data: GaussianData) -> GaussianData:
    """Create a copy of GaussianData.

    Helper function for data that doesn't have a clone() method.
    """
    from src.domain.data import GaussianData

    data._ensure_cpu()

    return GaussianData(
        means=data.means.copy() if data.means is not None else None,
        scales=data.scales.copy() if data.scales is not None else None,
        quats=data.quats.copy() if data.quats is not None else None,
        opacities=data.opacities.copy() if data.opacities is not None else None,
        sh0=data.sh0.copy() if data.sh0 is not None else None,
        shN=data.shN.copy() if data.shN is not None else None,
        format_info=data.format_info,
        source_path=data.source_path,
    )


__all__ = [
    "InterpolationMethod",
    "interpolate_gaussian_data",
    "lerp",
    "lerp_log_space",
    "slerp_quaternions",
]
