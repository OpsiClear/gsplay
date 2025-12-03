"""
Unified PLY writer using gsply v0.2.5 native save() methods.

Provides a single entry point for writing Gaussian splatting PLY files.
Uses gsply v0.2.5 GSTensor.save() and GSData.save() for object-oriented I/O.
Supports both standard and compressed formats with automatic GPU acceleration.
"""

import logging
from pathlib import Path
from typing import Literal
import tempfile
import os
from contextlib import contextmanager

import numpy as np
import torch
from gsply import GSTensor, GSData

from src.infrastructure.io.path_io import UniversalPath

logger = logging.getLogger(__name__)

# Constants
_MIN_SCALE = 1e-8
_MIN_OPACITY = 1e-8
_MAX_OPACITY = 1 - 1e-8
_COMPRESSED_SCALE_MIN = -20.0
_COMPRESSED_SCALE_MAX = 20.0


def _ensure_shn_shape(
    shN: torch.Tensor | np.ndarray | None, n_gaussians: int
) -> torch.Tensor | np.ndarray | None:
    """Ensure shN has shape (N, K, 3) format."""
    if shN is not None and shN.ndim == 2:
        n_coeffs = shN.shape[1] // 3
        return shN.reshape(n_gaussians, n_coeffs, 3)
    return shN


@contextmanager
def _temp_ply_file(file_path: UniversalPath):
    """Context manager for handling remote PLY files via temporary local file."""
    if file_path.is_remote:
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            yield tmp_path
        finally:
            os.unlink(tmp_path)
    else:
        yield str(file_path)


def write_ply(
    file_path: str | Path | UniversalPath,
    data: GSTensor | GSData,
    format: Literal["ply", "compressed"] = "ply",
) -> None:
    """Write Gaussian splatting PLY file using gsply v0.2.5 native save() methods.

    Uses gsply v0.2.5 GSTensor.save() or GSData.save() for object-oriented I/O.
    Explicitly normalizes data to PLY format (log scales, logit opacities) before saving
    if not already in PLY format. Automatically handles GPU acceleration for compressed format.

    Args:
        file_path: Output path (local or cloud)
        data: GSTensor or GSData object containing Gaussian data (any format)
        format: Output format - "ply" (standard) or "compressed"

    Raises:
        ValueError: If format is unsupported or data is invalid
        TypeError: If data is not GSTensor or GSData

    Note:
        PLY format stores scales in log space and opacities in logit space.
        Data is normalized to PLY format before saving if not already in PLY format.
    """
    file_path = UniversalPath(file_path)

    logger.debug(f"[PLY Writer] Writing {format} format to {file_path.name} using gsply native save()")

    if not isinstance(data, (GSTensor, GSData)):
        raise TypeError(f"data must be GSTensor or GSData, got {type(data)}")

    compressed = format == "compressed"

    # Ensure SHN shape is correct before saving (modify in-place)
    if isinstance(data, GSTensor):
        data.shN = _ensure_shn_shape(data.shN, data.means.shape[0])
    else:
        data.shN = _ensure_shn_shape(data.shN, data.means.shape[0])

    # Normalize to PLY format before saving (log scales, logit opacities, SH colors)
    # This ensures data is in correct format regardless of input format
    # For compressed format, GSTensor.save() checks format flags, but we normalize explicitly for safety
    # For uncompressed format, plywrite() expects PLY format, so normalization is required
    from gsply.formats import DataFormat

    if isinstance(data, GSTensor):
        # Check if data is already in PLY format by checking format flags
        needs_normalize = True
        if data._format is not None:
            scales_format = data._format.get("scales")
            opacities_format = data._format.get("opacities")
            # Data is in PLY format if scales are log and opacities are logit
            if scales_format == DataFormat.SCALES_PLY and opacities_format == DataFormat.OPACITIES_PLY:
                needs_normalize = False

        if needs_normalize:
            data = data.normalize(inplace=False)  # Create copy to avoid modifying original

        # Convert sh0 from RGB back to SH format if needed
        # PLY format expects SH coefficients, not RGB colors
        if hasattr(data, 'is_sh0_rgb') and data.is_sh0_rgb:
            data = data.to_sh(inplace=False)
    else:
        # GSData: check format flags
        needs_normalize = True
        if data._format is not None:
            scales_format = data._format.get("scales")
            opacities_format = data._format.get("opacities")
            if scales_format == DataFormat.SCALES_PLY and opacities_format == DataFormat.OPACITIES_PLY:
                needs_normalize = False

        if needs_normalize:
            data = data.normalize(inplace=False)  # Create copy to avoid modifying original

        # Convert sh0 from RGB back to SH format if needed
        # PLY format expects SH coefficients, not RGB colors
        if hasattr(data, 'is_sh0_rgb') and data.is_sh0_rgb:
            data = data.to_sh(inplace=False)

    # Use gsply v0.2.5 native save() methods
    # Data is now guaranteed to be in PLY format (log scales, logit opacities)
    with _temp_ply_file(file_path) as ply_path:
        if isinstance(data, GSTensor):
            # GSTensor.save() uses GPU compression for compressed format
            data.save(ply_path, compressed=compressed)
        else:
            # GSData.save() wraps plywrite()
            data.save(ply_path, compressed=compressed)
        
        # Upload to cloud storage if needed
        if file_path.is_remote:
            with open(ply_path, "rb") as f:
                file_path.write_bytes(f.read())

    logger.debug(
        f"[PLY Writer] Successfully wrote {format} to {file_path.name} using gsply native save()"
    )


def write_ply_bytes(
    data: GSTensor | GSData,
    format: Literal["ply", "compressed"] = "ply",
) -> bytes:
    """Write Gaussian splatting PLY to bytes (in-memory) using gsply v0.2.5 native save() methods.

    Uses gsply v0.2.5 GSTensor.save() or GSData.save() for object-oriented I/O.
    Explicitly normalizes data to PLY format (log scales, logit opacities) before saving
    if not already in PLY format. Useful for streaming or API responses without writing to disk.

    Args:
        data: GSTensor or GSData object containing Gaussian data (any format)
        format: Output format - "ply" or "compressed"

    Returns:
        PLY file data as bytes

    Raises:
        ValueError: If format is unsupported
        TypeError: If data is not GSTensor or GSData

    Note:
        PLY format stores scales in log space and opacities in logit space.
        Data is normalized to PLY format before saving if not already in PLY format.
    """
    if not isinstance(data, (GSTensor, GSData)):
        raise TypeError(f"data must be GSTensor or GSData, got {type(data)}")

    compressed = format == "compressed"

    # Ensure SHN shape is correct before saving (modify in-place)
    if isinstance(data, GSTensor):
        data.shN = _ensure_shn_shape(data.shN, data.means.shape[0])
    else:
        data.shN = _ensure_shn_shape(data.shN, data.means.shape[0])

    # Normalize to PLY format before saving (log scales, logit opacities, SH colors)
    # This ensures data is in correct format regardless of input format
    from gsply.formats import DataFormat

    if isinstance(data, GSTensor):
        needs_normalize = True
        if data._format is not None:
            scales_format = data._format.get("scales")
            opacities_format = data._format.get("opacities")
            if scales_format == DataFormat.SCALES_PLY and opacities_format == DataFormat.OPACITIES_PLY:
                needs_normalize = False

        if needs_normalize:
            data = data.normalize(inplace=False)

        # Convert sh0 from RGB back to SH format if needed
        if hasattr(data, 'is_sh0_rgb') and data.is_sh0_rgb:
            data = data.to_sh(inplace=False)
    else:
        needs_normalize = True
        if data._format is not None:
            scales_format = data._format.get("scales")
            opacities_format = data._format.get("opacities")
            if scales_format == DataFormat.SCALES_PLY and opacities_format == DataFormat.OPACITIES_PLY:
                needs_normalize = False

        if needs_normalize:
            data = data.normalize(inplace=False)

        # Convert sh0 from RGB back to SH format if needed
        if hasattr(data, 'is_sh0_rgb') and data.is_sh0_rgb:
            data = data.to_sh(inplace=False)

    # Use gsply v0.2.5 native save() methods via temporary file
    # Data is now guaranteed to be in PLY format (log scales, logit opacities, SH colors)
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        if isinstance(data, GSTensor):
            data.save(tmp_path, compressed=compressed)
        else:
            data.save(tmp_path, compressed=compressed)
        
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        os.unlink(tmp_path)
