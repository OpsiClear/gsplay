"""
PLY data sink (exporter) implementation.

This module provides the DataSinkProtocol implementation for exporting
Gaussian data to PLY format.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from src.domain.data import GaussianData
from src.domain.interfaces import DataSinkMetadata, DataSinkProtocol


logger = logging.getLogger(__name__)


class PlySink(DataSinkProtocol):
    """PLY file exporter.

    Exports GaussianData to standard PLY format using gsply.

    Example
    -------
    >>> sink = PlySink()
    >>> sink.export(gaussian_data, "/path/to/output.ply")
    """

    @classmethod
    def metadata(cls) -> DataSinkMetadata:
        """Return metadata about this sink type."""
        return DataSinkMetadata(
            name="PLY",
            description="Standard PLY format",
            file_extension=".ply",
            supports_animation=True,
            config_schema=None,
        )

    def export(self, data: GaussianData, path: str, **options: Any) -> None:
        """Export single frame to PLY file.

        Parameters
        ----------
        data : GaussianData
            Gaussian data to export
        path : str
            Output file path
        **options : Any
            Export options (currently unused)
        """
        # Ensure directory exists
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to GSTensor for saving (uses gsply native save)
        gstensor = data.to_gstensor(device="cpu")

        # Normalize to PLY format (log scales, logit opacities)
        gstensor = gstensor.normalize(inplace=False)

        # Convert sh0 from RGB back to SH format if needed
        # PLY format expects SH coefficients, not RGB colors
        if hasattr(gstensor, "is_sh0_rgb") and gstensor.is_sh0_rgb:
            gstensor = gstensor.to_sh(inplace=False)

        # Use gsply's native save method
        gstensor.save(str(output_path), compressed=False)

        logger.debug("Exported %d gaussians to %s", data.n_gaussians, path)

    def export_sequence(
        self,
        frames: Iterator[GaussianData],
        output_dir: str,
        **options: Any,
    ) -> int:
        """Export sequence of frames to PLY files.

        Parameters
        ----------
        frames : Iterator[GaussianData]
            Iterator of frames to export
        output_dir : str
            Output directory
        **options : Any
            Export options:
            - filename_pattern: str = "frame_{:06d}.ply"
            - binary: bool = True
            - normalize: bool = True
            - progress_callback: callable(frame_idx, total) = None

        Returns
        -------
        int
            Number of frames successfully exported
        """
        filename_pattern = options.get("filename_pattern", "frame_{:06d}.ply")
        progress_callback = options.get("progress_callback")

        # Ensure output directory exists
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        exported_count = 0

        for idx, frame in enumerate(frames):
            try:
                filename = filename_pattern.format(idx)
                frame_path = output_path / filename

                self.export(frame, str(frame_path), **options)
                exported_count += 1

                if progress_callback:
                    progress_callback(idx, None)  # Total unknown for iterator

            except Exception as e:
                logger.error("Failed to export frame %d: %s", idx, e)
                continue

        logger.info("Exported %d frames to %s", exported_count, output_dir)
        return exported_count


class CompressedPlySink(DataSinkProtocol):
    """Compressed PLY file exporter.

    Exports GaussianData to compressed PLY format (16 bytes/splat).

    Example
    -------
    >>> sink = CompressedPlySink()
    >>> sink.export(gaussian_data, "/path/to/output.ply")
    """

    @classmethod
    def metadata(cls) -> DataSinkMetadata:
        """Return metadata about this sink type."""
        return DataSinkMetadata(
            name="Compressed PLY",
            description="Compressed PLY format (16 bytes/splat)",
            file_extension=".ply",
            supports_animation=True,
            config_schema=None,
        )

    def export(self, data: GaussianData, path: str, **options: Any) -> None:
        """Export single frame to compressed PLY file.

        Parameters
        ----------
        data : GaussianData
            Gaussian data to export
        path : str
            Output file path
        **options : Any
            Export options
        """
        # Import the existing compressed PLY exporter
        from src.infrastructure.exporters.compressed_ply_exporter import (
            CompressedPlyExporter,
        )

        # Ensure directory exists
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to GSData/GSTensor for the existing exporter
        gsdata = data.to_gsdata()

        # Use existing exporter
        exporter = CompressedPlyExporter()
        exporter.export_frame(gsdata, output_path)

        logger.debug("Exported %d gaussians (compressed) to %s", data.n_gaussians, path)

    def export_sequence(
        self,
        frames: Iterator[GaussianData],
        output_dir: str,
        **options: Any,
    ) -> int:
        """Export sequence of frames to compressed PLY files.

        Parameters
        ----------
        frames : Iterator[GaussianData]
            Iterator of frames to export
        output_dir : str
            Output directory
        **options : Any
            Export options

        Returns
        -------
        int
            Number of frames successfully exported
        """
        filename_pattern = options.get("filename_pattern", "frame_{:06d}.ply")
        progress_callback = options.get("progress_callback")

        # Ensure output directory exists
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        exported_count = 0

        for idx, frame in enumerate(frames):
            try:
                filename = filename_pattern.format(idx)
                frame_path = output_path / filename

                self.export(frame, str(frame_path), **options)
                exported_count += 1

                if progress_callback:
                    progress_callback(idx, None)

            except Exception as e:
                logger.error("Failed to export compressed frame %d: %s", idx, e)
                continue

        logger.info("Exported %d compressed frames to %s", exported_count, output_dir)
        return exported_count
