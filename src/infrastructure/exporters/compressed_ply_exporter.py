"""
Compressed PLY exporter implementation using gsply.

Exports Gaussian Splatting data to compressed PLY format using chunk-based
quantization, achieving 16 bytes/splat vs 232 bytes/splat (14.5x compression).

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

import torch
from tqdm import tqdm

from src.domain.entities import GSTensor
from src.domain.interfaces import ModelInterface
from src.infrastructure.io.path_io import UniversalPath

logger = logging.getLogger(__name__)


class CompressedPlyExporter:
    """
    Export Gaussian data to compressed PLY format using gsply.

    Uses gsply library for consistent PLY I/O architecture.
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
        self,
        gaussian_data: GSTensor,
        output_path: str | Path | UniversalPath,
        **options: Any
    ) -> None:
        """
        Export single frame of Gaussian data to compressed PLY file.

        Supports local filesystem and cloud storage paths.

        Parameters
        ----------
        gaussian_data : GSTensor
            Gaussian data to export
        output_path : str | Path | UniversalPath
            Output PLY file path (local or cloud)
        **options : Any
            Export options (currently unused)

        Raises
        ------
        ValueError
            If gaussian_data is empty
        """
        from src.infrastructure.processing.ply import write_ply

        # Validate input
        if len(gaussian_data) == 0:
            raise ValueError("Cannot export empty Gaussian data")

        # Prepare SH coefficients (pad/truncate to 15 bands for PLY compatibility)
        sh0, shN = self._prepare_sh_for_export(gaussian_data)

        # Create modified GSTensor with prepared SH coefficients
        # Note: Device is preserved from input gaussian_data (GPU/CPU as requested)
        # write_ply uses gsply v0.2.5 native save() methods internally
        # GSTensor.save() respects the device: GPU compression if on GPU, CPU if on CPU
        export_data = GSTensor(
            means=gaussian_data.means,
            scales=gaussian_data.scales,
            quats=gaussian_data.quats,
            opacities=gaussian_data.opacities,
            sh0=sh0.squeeze(1),  # [N, 1, 3] -> [N, 3]
            shN=shN,
        )

        # Convert to UniversalPath for cloud storage support
        output_path = UniversalPath(output_path)

        # Export using write_ply (uses GSTensor.save() internally)
        # Device handling: If export_data is on GPU, uses GPU compression; if on CPU, uses CPU
        write_ply(
            file_path=output_path,
            data=export_data,
            format="compressed",
        )

        logger.debug(f"Exported compressed PLY: {output_path} ({gaussian_data.means.shape[0]} gaussians)")

    def export_sequence(
        self,
        model: ModelInterface,
        output_dir: str | Path | UniversalPath,
        apply_edits_fn: Any = None,
        progress_callback: Any = None,
        **options: Any
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
                gaussian_data = model.get_gaussians_at_normalized_time(normalized_time=normalized_time)

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

    def _prepare_sh_for_export(
        self, gaussian_data: GSTensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare SH coefficients for PLY export.

        Handles both full SH coefficients and RGB-only data.
        Ensures all tensors are on the same device as input data.

        Parameters
        ----------
        gaussian_data : GSTensor
            Input Gaussian data

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            sh0 [N, 1, 3] and shN [N, 15, 3] for export (on same device as input)
        """
        # Get device and dtype from input data
        device = gaussian_data.means.device
        dtype = gaussian_data.means.dtype

        # Check if we have full SH coefficients
        if hasattr(gaussian_data, "sh_coeffs") and gaussian_data.shN is not None:
            sh_coeffs = gaussian_data.shN.detach()  # Keep on same device

            # Split into DC (sh0) and higher order (shN)
            sh0 = sh_coeffs[:, 0:1, :]  # First band [N, 1, 3]

            if sh_coeffs.shape[1] > 1:
                # We have higher order coefficients
                shN = sh_coeffs[:, 1:, :]  # Shape [N, K-1, 3]

                # Pad or truncate to 15 bands (gsply expects 16 total: 1 DC + 15 higher)
                num_higher_bands = shN.shape[1]
                if num_higher_bands < 15:
                    # Pad with zeros (on same device)
                    padding = torch.zeros(
                        shN.shape[0], 15 - num_higher_bands, 3, dtype=dtype, device=device
                    )
                    shN = torch.cat([shN, padding], dim=1)
                elif num_higher_bands > 15:
                    # Truncate
                    shN = shN[:, :15, :]
            else:
                # Only DC component, create zeros for higher order (on same device)
                shN = torch.zeros(sh0.shape[0], 15, 3, dtype=dtype, device=device)

            logger.debug(
                f"Exporting with full SH: sh_degree={gaussian_data}, "
                f"sh0={sh0.shape}, shN={shN.shape}, device={device}"
            )

        else:
            # Only RGB colors available - convert to SH format using gsply v0.2.5 to_sh() method
            # Use inplace=True for better performance (we create new GSTensor for export anyway)
            gstensor_sh = gaussian_data.to_sh(inplace=True)
            sh0_tensor = gstensor_sh.sh0.detach()  # Keep on same device
            
            # Reshape from [N, 3] to [N, 1, 3] for export format
            sh0 = sh0_tensor.unsqueeze(1)  # Shape: [N, 1, 3]

            # No higher order coefficients (on same device)
            shN = torch.zeros(sh0.shape[0], 15, 3, dtype=dtype, device=device)

            logger.debug(
                f"Exporting RGB-to-SH conversion using gsply.to_sh(): colors={gaussian_data.sh0.shape}, "
                f"sh0={sh0.shape}, shN={shN.shape}, device={device}"
            )

        return sh0, shN

