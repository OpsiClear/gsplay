"""
Utility functions for PLY file operations.

Includes spherical harmonics conversion, spatial sorting (Morton codes),
and other common utilities used across PLY loaders/writers.
"""

import torch
import numpy as np


# Spherical Harmonics conversion
# -------------------------------

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


# Morton code spatial sorting
# ---------------------------

def part1by2_vec(x: torch.Tensor) -> torch.Tensor:
    """Interleave bits of x with 0s for Morton code computation.

    Args:
        x: Input tensor (N,) with integer values

    Returns:
        Bit-interleaved tensor (N,)
    """
    x = x & 0x000003FF
    x = (x ^ (x << 16)) & 0xFF0000FF
    x = (x ^ (x << 8)) & 0x0300F00F
    x = (x ^ (x << 4)) & 0x030C30C3
    x = (x ^ (x << 2)) & 0x09249249
    return x


def encode_morton3_vec(
    x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
) -> torch.Tensor:
    """Compute Morton codes for 3D coordinates.

    Morton codes (Z-order curve) provide spatial locality for 3D points.
    Used to sort Gaussians for better cache coherence during rendering.

    Args:
        x: X coordinates (N,)
        y: Y coordinates (N,)
        z: Z coordinates (N,)

    Returns:
        Morton codes (N,)
    """
    return (part1by2_vec(z) << 2) + (part1by2_vec(y) << 1) + part1by2_vec(x)


def sort_gaussians_by_morton(
    centers: torch.Tensor,
    indices: torch.Tensor | None = None
) -> torch.Tensor:
    """Sort Gaussian centers using Morton codes for spatial locality.

    Args:
        centers: Gaussian centers (N, 3)
        indices: Optional existing indices (N,), defaults to torch.arange(N)

    Returns:
        Sorted indices (N,)
    """
    if indices is None:
        indices = torch.arange(centers.shape[0], device=centers.device)

    # Compute min and max values
    min_vals, _ = torch.min(centers, dim=0)
    max_vals, _ = torch.max(centers, dim=0)

    # Compute scaling factors
    lengths = max_vals - min_vals
    lengths[lengths == 0] = 1  # Prevent division by zero

    # Normalize and scale to 10-bit integer range (0-1023)
    scaled_centers = ((centers - min_vals) / lengths * 1024).floor().to(torch.int32)

    # Extract x, y, z coordinates
    x, y, z = scaled_centers[:, 0], scaled_centers[:, 1], scaled_centers[:, 2]

    # Compute Morton codes
    morton_codes = encode_morton3_vec(x, y, z)

    # Sort by Morton code
    _, sorted_idx = torch.sort(morton_codes)

    return indices[sorted_idx]


# Pack/Unpack for compressed PLY
# ------------------------------

def pack_unorm(value: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack normalized float [0,1] to unsigned integer.

    Args:
        value: Normalized tensor (N,) in [0, 1]
        bits: Number of bits (e.g., 8, 10, 11)

    Returns:
        Packed integer tensor (N,)
    """
    max_val = (1 << bits) - 1
    return (value * max_val).round().to(torch.int32)


def unpack_unorm(packed: torch.Tensor, bits: int) -> torch.Tensor:
    """Unpack unsigned integer to normalized float [0,1].

    Args:
        packed: Packed integer tensor (N,)
        bits: Number of bits (e.g., 8, 10, 11)

    Returns:
        Normalized float tensor (N,) in [0, 1]
    """
    max_val = (1 << bits) - 1
    mask = max_val
    return (packed & mask).float() / max_val


def pack_111011(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Pack 3D vector into 32-bit value (11-10-11 bit allocation).

    Args:
        x: X component (N,) in [0, 1]
        y: Y component (N,) in [0, 1]
        z: Z component (N,) in [0, 1]

    Returns:
        Packed 32-bit tensor (N,)
    """
    x_packed = pack_unorm(x, 11) << 21
    y_packed = pack_unorm(y, 10) << 11
    z_packed = pack_unorm(z, 11)
    return x_packed | y_packed | z_packed


def unpack_111011(
    packed: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unpack 32-bit value to 3D vector (11-10-11 bit allocation).

    Args:
        packed: Packed 32-bit tensor (N,)

    Returns:
        Tuple of (x, y, z) tensors, each (N,) in [0, 1]
    """
    x = unpack_unorm(packed >> 21, 11)
    y = unpack_unorm(packed >> 11, 10)
    z = unpack_unorm(packed, 11)
    return x, y, z


def pack_8888(
    x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, w: torch.Tensor
) -> torch.Tensor:
    """Pack 4 channels into 32-bit value (8 bits each).

    Args:
        x: First component (N,) in [0, 1]
        y: Second component (N,) in [0, 1]
        z: Third component (N,) in [0, 1]
        w: Fourth component (N,) in [0, 1]

    Returns:
        Packed 32-bit tensor (N,)
    """
    x_packed = pack_unorm(x, 8) << 24
    y_packed = pack_unorm(y, 8) << 16
    z_packed = pack_unorm(z, 8) << 8
    w_packed = pack_unorm(w, 8)
    return x_packed | y_packed | z_packed | w_packed


def unpack_8888(
    packed: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unpack 32-bit value to 4 channels (8 bits each).

    Args:
        packed: Packed 32-bit tensor (N,)

    Returns:
        Tuple of (x, y, z, w) tensors, each (N,) in [0, 1]
    """
    x = unpack_unorm(packed >> 24, 8)
    y = unpack_unorm(packed >> 16, 8)
    z = unpack_unorm(packed >> 8, 8)
    w = unpack_unorm(packed, 8)
    return x, y, z, w


def pack_rotation(q: torch.Tensor) -> torch.Tensor:
    """Pack quaternion using smallest-three encoding.

    Stores the 3 smallest components in 10 bits each, plus 2 bits
    to identify which component was largest. The largest component
    is reconstructed from the constraint |q| = 1.

    Args:
        q: Quaternion tensor (N, 4) - should be normalized

    Returns:
        Packed 32-bit tensor (N,)
    """
    # Find largest component
    abs_q = torch.abs(q)
    largest_idx = torch.argmax(abs_q, dim=1)

    # Prepare output
    packed = torch.zeros(q.shape[0], dtype=torch.int32, device=q.device)

    # Normalize to [0, 1] range: (x + 1) / 2
    # Then scale to fit in 10 bits
    norm = 1.0 / (np.sqrt(2) * 0.5)

    for idx in range(4):
        mask = largest_idx == idx
        if not mask.any():
            continue

        # Select the other 3 components
        components = []
        for i in range(4):
            if i != idx:
                components.append(q[mask, i])

        # Pack the 3 smaller components
        a, b, c = components[0], components[1], components[2]
        a_norm = (a / norm + 0.5).clamp(0, 1)
        b_norm = (b / norm + 0.5).clamp(0, 1)
        c_norm = (c / norm + 0.5).clamp(0, 1)

        a_packed = pack_unorm(a_norm, 10) << 20
        b_packed = pack_unorm(b_norm, 10) << 10
        c_packed = pack_unorm(c_norm, 10)
        which = idx << 30

        packed[mask] = a_packed[mask] | b_packed[mask] | c_packed[mask] | which

    return packed


def unpack_rotation(packed: torch.Tensor) -> torch.Tensor:
    """Unpack quaternion from smallest-three encoding.

    Args:
        packed: Packed 32-bit tensor (N,)

    Returns:
        Quaternion tensor (N, 4)
    """
    norm = 1.0 / (np.sqrt(2) * 0.5)

    # Unpack the 3 stored components
    a = (unpack_unorm(packed >> 20, 10) - 0.5) * norm
    b = (unpack_unorm(packed >> 10, 10) - 0.5) * norm
    c = (unpack_unorm(packed, 10) - 0.5) * norm

    # Reconstruct fourth component
    m = torch.sqrt(torch.clamp(1.0 - (a * a + b * b + c * c), min=0.0))

    # Determine which component was largest
    which = (packed >> 30) & 0x3

    # Reconstruct full quaternion
    q = torch.zeros(packed.shape[0], 4, device=packed.device)

    mask0 = which == 0
    mask1 = which == 1
    mask2 = which == 2
    mask3 = which == 3

    q[mask0] = torch.stack([m[mask0], a[mask0], b[mask0], c[mask0]], dim=1)
    q[mask1] = torch.stack([a[mask1], m[mask1], b[mask1], c[mask1]], dim=1)
    q[mask2] = torch.stack([a[mask2], b[mask2], m[mask2], c[mask2]], dim=1)
    q[mask3] = torch.stack([a[mask3], b[mask3], c[mask3], m[mask3]], dim=1)

    return q


# Numpy versions for compatibility
# --------------------------------

def sh2rgb_np(sh: np.ndarray) -> np.ndarray:
    """NumPy version of sh2rgb."""
    return sh * SH_C0 + 0.5


def rgb2sh_np(rgb: np.ndarray) -> np.ndarray:
    """NumPy version of rgb2sh."""
    return (rgb - 0.5) / SH_C0
