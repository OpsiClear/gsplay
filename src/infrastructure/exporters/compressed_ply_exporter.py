"""
Compressed PLY exporter implementation using gsply.

Exports Gaussian Splatting data to compressed PLY format using chunk-based
quantization, achieving 16 bytes/splat vs 232 bytes/splat (14.5x compression).
Uses native gsply GSTensor.save(compressed=True) for reliable export.

Format: PlayCanvas Super-compressed PLY
Reference: https://github.com/playcanvas/splat-transform

Storage Format (via gsply):
- Scales: LOG space, clamped to [-20, 20]
- Opacities: LOGIT space (gsply converts to linear internally)
- Positions: Chunk-relative quantized positions
- Rotations: Smallest-three quaternion encoding

Supports local filesystem and cloud storage via UniversalPath.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.domain.entities import GSTensor
from src.domain.interfaces import ModelInterface
from src.infrastructure.io.path_io import UniversalPath


logger = logging.getLogger(__name__)


class CompressedPlyExporter:
    """
    Export Gaussian data to compressed PLY format using gsply.

    Uses native gsply GSTensor.save(compressed=True) method for reliable export.
    Compressed PLY format stores:
    - Chunk metadata (min/max bounds for position, scale, color)
    - Packed vertex data (32-bit quantized position, rotation, scale, color)
    - Optional SH coefficients (uint8 quantized)

    Achieves 16 bytes/splat vs 232 bytes/splat for standard PLY.
    """

    def __init__(self, **config: Any):
        """
        Initialize compressed PLY exporter.

        Parameters
        ----------
        **config : Any
            Configuration options (currently unused, reserved for future)
        """
        self.config = config

    def get_file_extension(self) -> str:
        """Get file extension for compressed PLY format."""
        return ".compressed.ply"

    def export_frame(
        self, gaussian_data: GSTensor, output_path: str | Path | UniversalPath, **options: Any
    ) -> None:
        """
        Export single frame of Gaussian data to compressed PLY file.

        Uses native gsply GSTensor.save(compressed=True) method for reliable export.
        Supports local filesystem and cloud storage paths.

        Parameters
        ----------
        gaussian_data : GSTensor
            Gaussian data to export (gsply.GSTensor)
        output_path : str | Path | UniversalPath
            Output PLY file path (local or cloud)
        **options : Any
            Export options (currently unused)

        Raises
        ------
        ValueError
            If gaussian_data is empty
        """
        # Validate input
        if len(gaussian_data) == 0:
            raise ValueError("Cannot export empty Gaussian data")

        # Convert to UniversalPath for cloud storage support
        output_path = UniversalPath(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Normalize data to PLY format (log scales, logit opacities) if needed
        # Use inplace=False to avoid modifying original data
        export_data = gaussian_data.normalize(inplace=False)

        # Convert sh0 from RGB back to SH format if needed
        # PLY format expects SH coefficients, not RGB colors
        if hasattr(export_data, "is_sh0_rgb") and export_data.is_sh0_rgb:
            export_data = export_data.to_sh(inplace=False)

        # Use native gsply save() method with compressed=True
        # GSTensor.save() respects the device: GPU compression if on GPU, CPU if on CPU
        # For remote paths, save to temp file first then upload
        if output_path.is_remote:
            import os
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                export_data.save(tmp_path, compressed=True)
                with open(tmp_path, "rb") as f:
                    output_path.write_bytes(f.read())
            finally:
                os.unlink(tmp_path)
        else:
            export_data.save(str(output_path), compressed=True)

        logger.debug(f"Exported compressed PLY: {output_path} ({len(gaussian_data)} gaussians)")

    def export_sequence(
        self,
        model: ModelInterface,
        output_dir: str | Path | UniversalPath,
        apply_edits_fn: Any = None,
        progress_callback: Any = None,
        **options: Any,
    ) -> int:
        """
        Export all frames from model to compressed PLY files.

        Supports local filesystem and cloud storage paths.

        Parameters
        ----------
        model : ModelInterface
            Model to export from
        output_dir : str | Path | UniversalPath
            Output directory for PLY files (local or cloud)
        apply_edits_fn : callable | None
            Optional function to apply edits: fn(gaussian_data) -> gaussian_data
        progress_callback : callable | None
            Optional progress callback: fn(frame_idx, total_frames) -> None
        **options : Any
            Export options

        Returns
        -------
        int
            Number of frames successfully exported

        Raises
        ------
        ValueError
            If model has no frames to export
        """
        # Convert to UniversalPath for cloud storage support
        output_dir = UniversalPath(output_dir)

        total_frames = model.get_total_frames()

        if total_frames == 0:
            raise ValueError("Model has no frames to export")

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Exporting {total_frames} frames to compressed PLY: {output_dir}")

        exported_count = 0

        for frame_idx in tqdm(range(total_frames), desc="Exporting frames", unit="frame"):
            try:
                # Get frame time
                normalized_time = model.get_frame_time(frame_idx)

                # Get Gaussian data
                gaussian_data = model.get_gaussians_at_normalized_time(
                    normalized_time=normalized_time
                )

                if gaussian_data is None or len(gaussian_data) == 0:
                    tqdm.write(f"Frame {frame_idx}: No gaussian data, skipping")
                    continue

                # Apply edits if provided
                if apply_edits_fn is not None:
                    gaussian_data = apply_edits_fn(gaussian_data)

                    # Check if edits removed all gaussians
                    if gaussian_data is None or len(gaussian_data) == 0:
                        tqdm.write(f"Frame {frame_idx}: All gaussians filtered out, skipping")
                        continue

                # Export frame
                ext = self.get_file_extension()
                output_path = output_dir / f"frame_{frame_idx:05d}{ext}"
                self.export_frame(gaussian_data, output_path)

                exported_count += 1

                # Progress callback
                if progress_callback is not None:
                    progress_callback(frame_idx, total_frames)

            except Exception as e:
                tqdm.write(f"Frame {frame_idx}: Export failed - {e}")
                logger.error(f"Frame {frame_idx}: Export failed - {e}", exc_info=True)
                continue

        logger.info(f"Compressed PLY export complete: {exported_count}/{total_frames} frames")

        return exported_count
