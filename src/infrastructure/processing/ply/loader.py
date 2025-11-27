"""
Unified PLY loader using gsply v0.2.5 with GSTensor optimization.

Provides a single entry point for loading Gaussian splatting PLY files.
Uses gsply v0.2.5 GPU loading interface which auto-detects and handles both standard and compressed formats.

Performance:
- Uses gsply v0.2.5 plyread_gpu() for direct GPU loading (no CPU intermediate)
- For CPU loading: plyread() + GSTensor.from_gsdata()
- Eliminates unnecessary CPU->GPU transfers for GPU path
"""

import logging
from pathlib import Path
import tempfile
import os
from contextlib import contextmanager

import gsply
import torch
from gsply import GSTensor

from src.infrastructure.io.path_io import UniversalPath

logger = logging.getLogger(__name__)


@contextmanager
def _temp_ply_file(file_path: UniversalPath):
    """Context manager for handling remote PLY files via temporary local file."""
    if file_path.is_remote:
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(file_path.read_bytes())
        try:
            yield tmp_path
        finally:
            os.unlink(tmp_path)
    else:
        yield str(file_path)


def load_ply(
    file_path: str | Path | UniversalPath,
    device: str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load Gaussian splatting PLY file using gsply.

    Auto-detects and loads both standard and compressed PLY formats.
    Uses gsply v0.2.5 plyread_gpu() for GPU loading (direct to GPU, no CPU intermediate).
    For CPU loading, uses plyread() + GSTensor.from_gsdata().

    Args:
        file_path: Path to PLY file (local or cloud)
        device: Device to load tensors to ('cpu' or 'cuda')

    Returns:
        Tuple of (means, scales, quats, opacities, sh0, shN):
        - means: (N, 3) positions
        - scales: (N, 3) log scales
        - quats: (N, 4) quaternion rotations
        - opacities: (N,) logit opacities
        - sh0: (N, 3) DC spherical harmonics coefficients
        - shN: (N, K, 3) higher-order SH coefficients

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file cannot be loaded
    """
    file_path = UniversalPath(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"PLY file not found: {file_path}")

    logger.debug(f"[PLY Loader] Loading {file_path.name}")

    try:
        with _temp_ply_file(file_path) as ply_path:
            if device != "cpu":
                gstensor = gsply.plyread_gpu(ply_path, device=device)
            else:
                data = gsply.plyread(ply_path)
                gstensor = gsply.GSTensor.from_gsdata(data, device="cpu")

        logger.debug(f"[PLY Loader] Loaded {gstensor.means.shape[0]} Gaussians from {file_path.name}")

        shN = gstensor.shN if gstensor.shN is not None else torch.zeros(
            (gstensor.means.shape[0], 0, 3), dtype=torch.float32, device=device
        )
        return (
            gstensor.means,
            gstensor.scales,
            gstensor.quats,
            gstensor.opacities,
            gstensor.sh0,
            shN,
        )
    except FileNotFoundError:
        raise
    except Exception as e:
        # Re-raise InterruptRenderException without wrapping so renderer can handle it
        if e.__class__.__name__ == "InterruptRenderException":
            raise
        raise ValueError(f"Failed to load PLY file {file_path.name}: {e}") from e


def load_ply_as_gsdata(
    file_path: str | Path | UniversalPath
) -> gsply.GSData:
    """Load PLY file directly as GSData (NumPy arrays, CPU-optimized).

    This is the optimal loading path for CPU-based processing.
    Loads directly to NumPy arrays without any PyTorch tensor creation.

    Args:
        file_path: Path to PLY file (local or cloud)

    Returns:
        GSData containing all Gaussian data as NumPy arrays

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file cannot be loaded
    """
    file_path = UniversalPath(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"PLY file not found: {file_path}")

    logger.debug(f"[PLY Loader] Loading {file_path.name} as GSData (NumPy)")

    try:
        with _temp_ply_file(file_path) as ply_path:
            data = gsply.plyread(ply_path)
        logger.debug(f"[PLY Loader] Loaded {data.means.shape[0]} Gaussians as GSData (NumPy)")
        return data
    except FileNotFoundError:
        raise
    except Exception as e:
        # Re-raise InterruptRenderException without wrapping so renderer can handle it
        if e.__class__.__name__ == "InterruptRenderException":
            raise
        raise ValueError(f"Failed to load PLY file {file_path.name}: {e}") from e


def load_ply_as_gstensor(
    file_path: str | Path | UniversalPath,
    device: str = "cpu"
) -> GSTensor:
    """Load PLY file directly as GSTensor (PyTorch tensors, GPU-optimized).

    This is the optimal loading path for GPU-based processing.
    Uses gsply v0.2.5 plyread_gpu() for direct GPU loading (no CPU intermediate).
    For CPU loading, uses plyread() + GSTensor.from_gsdata().

    Args:
        file_path: Path to PLY file (local or cloud)
        device: Device to load tensors to ('cpu' or 'cuda')

    Returns:
        GSTensor containing all Gaussian data as PyTorch tensors

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file cannot be loaded

    Example:
        >>> # Old pattern (2 lines):
        >>> means, scales, quats, opacities, sh0, shN = load_ply(path, device)
        >>> gaussians = GSTensor(means, scales, quats, opacities, sh0, shN)
        >>>
        >>> # New pattern (1 line):
        >>> gaussians = load_ply_as_gstensor(path, device)
    """
    file_path = UniversalPath(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"PLY file not found: {file_path}")

    logger.debug(f"[PLY Loader] Loading {file_path.name} as GSTensor (device={device})")

    try:
        with _temp_ply_file(file_path) as ply_path:
            if device != "cpu":
                gstensor = gsply.plyread_gpu(ply_path, device=device)
            else:
                data = gsply.plyread(ply_path)
                gstensor = gsply.GSTensor.from_gsdata(data, device="cpu")
        logger.debug(f"[PLY Loader] Loaded {gstensor.means.shape[0]} Gaussians as GSTensor")
        return gstensor
    except FileNotFoundError:
        raise
    except Exception as e:
        # Re-raise InterruptRenderException without wrapping so renderer can handle it
        if e.__class__.__name__ == "InterruptRenderException":
            raise
        raise ValueError(f"Failed to load PLY file {file_path.name}: {e}") from e
