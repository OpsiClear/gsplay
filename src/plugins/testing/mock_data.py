"""Mock data utilities for plugin testing.

Provides functions to create mock Gaussian data and test files.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from src.domain.data import GaussianData


def create_mock_gaussian_data(
    n_gaussians: int = 100,
    sh_degree: int = 0,
    seed: int | None = None,
) -> GaussianData:
    """Create mock GaussianData for testing.

    Parameters
    ----------
    n_gaussians : int
        Number of Gaussians to generate
    sh_degree : int
        Spherical harmonic degree (0-3)
    seed : int | None
        Random seed for reproducibility

    Returns
    -------
    GaussianData
        Mock Gaussian data
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate random means in a unit cube
    means = np.random.rand(n_gaussians, 3).astype(np.float32)

    # Generate scales (small positive values)
    scales = np.random.rand(n_gaussians, 3).astype(np.float32) * 0.1 + 0.01

    # Generate quaternions (normalized)
    quats = np.random.randn(n_gaussians, 4).astype(np.float32)
    quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)

    # Generate opacities
    opacities = np.random.rand(n_gaussians).astype(np.float32)

    # Generate colors (RGB in [0, 1])
    sh0 = np.random.rand(n_gaussians, 3).astype(np.float32)

    # Generate higher-order SH if requested
    shN = None
    if sh_degree > 0:
        sh_coeffs = (sh_degree + 1) ** 2 - 1  # Total SH coeffs minus DC
        shN = np.random.randn(n_gaussians, sh_coeffs, 3).astype(np.float32) * 0.1

    return GaussianData(
        means=means,
        scales=scales,
        quats=quats,
        opacities=opacities,
        sh0=sh0,
        shN=shN,
    )


def create_mock_ply_files(
    output_dir: str | Path | None = None,
    n_frames: int = 5,
    n_gaussians: int = 100,
    seed: int = 42,
) -> list[Path]:
    """Create mock PLY files for testing.

    Parameters
    ----------
    output_dir : str | Path | None
        Output directory (uses temp dir if None)
    n_frames : int
        Number of frames to generate
    n_gaussians : int
        Number of Gaussians per frame
    seed : int
        Random seed for reproducibility

    Returns
    -------
    list[Path]
        Paths to created PLY files
    """
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="gsplay_test_"))
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(seed)
    files = []

    for i in range(n_frames):
        # Create mock data with slight variations per frame
        frame_seed = seed + i
        data = create_mock_gaussian_data(n_gaussians, seed=frame_seed)

        # Write PLY file
        file_path = output_dir / f"frame_{i:04d}.ply"
        _write_mock_ply(file_path, data)
        files.append(file_path)

    return files


def _write_mock_ply(path: Path, data: GaussianData) -> None:
    """Write a minimal PLY file for testing.

    This creates a simplified PLY that can be read by gsply.

    Parameters
    ----------
    path : Path
        Output file path
    data : GaussianData
        Gaussian data to write
    """
    n = data.n_gaussians

    # Build header
    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
        "property float scale_0",
        "property float scale_1",
        "property float scale_2",
        "property float rot_0",
        "property float rot_1",
        "property float rot_2",
        "property float rot_3",
        "property float opacity",
        "property float f_dc_0",
        "property float f_dc_1",
        "property float f_dc_2",
        "end_header",
    ]
    header = "\n".join(header_lines) + "\n"

    # Build binary data
    # PLY format stores log scales and logit opacity
    log_scales = np.log(data.scales + 1e-7)
    logit_opacity = np.log(data.opacities / (1 - data.opacities + 1e-7))

    # Convert SH0 RGB to SH DC coefficients
    sh_dc = (data.sh0 - 0.5) / 0.28209479177387814  # C0 = 1/(2*sqrt(pi))

    # Pack data
    dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("scale_0", "<f4"),
            ("scale_1", "<f4"),
            ("scale_2", "<f4"),
            ("rot_0", "<f4"),
            ("rot_1", "<f4"),
            ("rot_2", "<f4"),
            ("rot_3", "<f4"),
            ("opacity", "<f4"),
            ("f_dc_0", "<f4"),
            ("f_dc_1", "<f4"),
            ("f_dc_2", "<f4"),
        ]
    )

    packed = np.zeros(n, dtype=dtype)
    packed["x"] = data.means[:, 0]
    packed["y"] = data.means[:, 1]
    packed["z"] = data.means[:, 2]
    packed["scale_0"] = log_scales[:, 0]
    packed["scale_1"] = log_scales[:, 1]
    packed["scale_2"] = log_scales[:, 2]
    packed["rot_0"] = data.quats[:, 0]
    packed["rot_1"] = data.quats[:, 1]
    packed["rot_2"] = data.quats[:, 2]
    packed["rot_3"] = data.quats[:, 3]
    packed["opacity"] = logit_opacity
    packed["f_dc_0"] = sh_dc[:, 0]
    packed["f_dc_1"] = sh_dc[:, 1]
    packed["f_dc_2"] = sh_dc[:, 2]

    # Write file
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        packed.tofile(f)


__all__ = [
    "create_mock_gaussian_data",
    "create_mock_ply_files",
]
