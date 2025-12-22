"""
Demo Data Sink (Exporter) - Reference Implementation.

This module demonstrates how to implement a DataSink plugin for exporting
Gaussian data from the GSPlay. This example exports to a
simple JSON format for demonstration purposes.

IMPLEMENTING A DATA SINK
========================

A DataSink (exporter) plugin must implement the DataSinkProtocol:

1. REQUIRED CLASS METHODS:
   - metadata() -> DataSinkMetadata

2. REQUIRED INSTANCE METHODS:
   - export(data: GaussianData, path: str, **options) -> None
   - export_sequence(frames: Iterator[GaussianData], output_dir: str, **options) -> int


DATASINKMETADATA FIELDS
=======================

    name: str
        Display name shown in UI (e.g., "PLY", "Compressed PLY")

    description: str
        Brief description (e.g., "Standard PLY format")

    file_extension: str
        Output file extension INCLUDING DOT (e.g., ".ply", ".json", ".bin")

    supports_animation: bool = True
        Whether this sink can export multiple frames.
        Set False for formats that only support single frames.

    config_schema: type | None = None
        Optional dataclass defining export options.
        Used for validation and UI generation.


EXPORT OPTIONS
==============

The export methods receive **options kwargs. Common options include:

    filename_pattern: str
        Pattern for sequence filenames (e.g., "frame_{:06d}.ply")
        Use Python format string syntax with frame index.

    progress_callback: callable(frame_idx: int, total: int | None)
        Called after each frame export for progress reporting.
        Total may be None for iterators with unknown length.

    binary: bool
        Write binary format (if supported by file type).

    normalize: bool
        Apply normalization before export (convert to PLY format).

Format-specific options should be documented in the sink's docstring.


REGISTRATION
============

After implementing your sink, register it:

    >>> from src.infrastructure.registry import DataSinkRegistry
    >>> DataSinkRegistry.register("json", DemoJsonSink)

Or add to the default registration in:
    src/infrastructure/registry/__init__.py


USAGE
=====

Once registered, the sink can be used via the export API:

    >>> from src.infrastructure.registry import DataSinkRegistry
    >>> sink_class = DataSinkRegistry.get("json")
    >>> sink = sink_class()
    >>> sink.export(gaussian_data, "/path/to/output.json")

Or for sequences:

    >>> exported = sink.export_sequence(
    ...     frames=frame_iterator,
    ...     output_dir="/path/to/output",
    ...     filename_pattern="frame_{:04d}.json",
    ...     progress_callback=lambda idx, total: print(f"Frame {idx}"),
    ... )
    >>> print(f"Exported {exported} frames")
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.domain.data import GaussianData
from src.domain.interfaces import DataSinkMetadata, DataSinkProtocol


logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Schema (Optional)
# =============================================================================


@dataclass
class DemoJsonSinkConfig:
    """Configuration for DemoJsonSink export options.

    This dataclass documents available export options. It's optional
    but recommended for discoverability and validation.

    Attributes
    ----------
    indent : int | None
        JSON indentation level. None = compact, 2 = readable.
    include_metadata : bool
        Include format_info and source_path in output.
    precision : int
        Decimal places for floating point values.
    max_gaussians : int | None
        Limit exported Gaussians per frame (for testing). None = all.
    """

    indent: int | None = 2
    include_metadata: bool = True
    precision: int = 6
    max_gaussians: int | None = None


# =============================================================================
# Data Sink Implementation
# =============================================================================


class DemoJsonSink(DataSinkProtocol):
    """Demo data sink that exports Gaussian data to JSON format.

    This is a reference implementation showing the minimum requirements
    for a DataSink plugin. JSON is used for human-readability - real
    exporters would use more efficient binary formats.

    Output Format
    -------------
    The JSON output structure:

    {
        "n_gaussians": 10000,
        "format_info": {
            "is_scales_ply": false,
            "is_opacities_ply": false,
            "is_sh0_rgb": true,
            "sh_degree": null
        },
        "source_path": "path/to/source",
        "data": {
            "means": [[x, y, z], ...],
            "scales": [[sx, sy, sz], ...],
            "quats": [[w, x, y, z], ...],
            "opacities": [o1, o2, ...],
            "sh0": [[r, g, b], ...]
        }
    }

    Example
    -------
    >>> sink = DemoJsonSink()
    >>> sink.export(gaussian_data, "/tmp/output.json", precision=4)
    >>>
    >>> # Export a sequence
    >>> exported = sink.export_sequence(
    ...     frames=[frame0, frame1, frame2],
    ...     output_dir="/tmp/sequence",
    ... )
    >>> print(f"Exported {exported} frames")
    Exported 3 frames

    Notes
    -----
    In a real exporter implementation, you would:
    - Use efficient binary formats (PLY, NPZ, custom binary)
    - Handle large datasets with streaming writes
    - Support compression
    - Validate output integrity
    """

    # =========================================================================
    # CLASS METHODS (Required by Protocol)
    # =========================================================================

    @classmethod
    def metadata(cls) -> DataSinkMetadata:
        """Return metadata about this sink type.

        This method is called by the registry to discover available exporters
        and their capabilities. It should return consistent, static metadata.

        Returns
        -------
        DataSinkMetadata
            Metadata describing this sink's capabilities.
        """
        return DataSinkMetadata(
            name="Demo JSON",
            description="Human-readable JSON format (for testing/debugging)",
            file_extension=".json",
            supports_animation=True,
            config_schema=DemoJsonSinkConfig,
        )

    # =========================================================================
    # REQUIRED METHODS
    # =========================================================================

    def export(self, data: GaussianData, path: str, **options: Any) -> None:
        """Export single frame to JSON file.

        This is the primary method for exporting a single frame. It should:
        1. Ensure the output directory exists
        2. Convert/process the data as needed
        3. Write to the output file
        4. Log success/failure

        Parameters
        ----------
        data : GaussianData
            Gaussian data to export. May be on CPU or GPU.
        path : str
            Output file path. Directory will be created if needed.
        **options : Any
            Export options (see DemoJsonSinkConfig for available options):
            - indent: int | None = 2
            - include_metadata: bool = True
            - precision: int = 6
            - max_gaussians: int | None = None

        Raises
        ------
        IOError
            If file cannot be written.
        ValueError
            If data is invalid or empty.
        """
        # Parse options
        indent = options.get("indent", 2)
        include_metadata = options.get("include_metadata", True)
        precision = options.get("precision", 6)
        max_gaussians = options.get("max_gaussians")

        # Validate data
        if data.n_gaussians == 0:
            raise ValueError("Cannot export empty GaussianData")

        # Ensure output directory exists
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Ensure data is on CPU (GaussianData handles conversion)
        data._ensure_cpu()

        # Apply max_gaussians limit if specified
        n = data.n_gaussians
        if max_gaussians is not None and max_gaussians < n:
            n = max_gaussians
            logger.debug("Limiting export to %d of %d gaussians", n, data.n_gaussians)

        # Build output dictionary
        output = {
            "n_gaussians": n,
        }

        # Add metadata if requested
        if include_metadata:
            output["format_info"] = {
                "is_scales_ply": data.format_info.is_scales_ply,
                "is_opacities_ply": data.format_info.is_opacities_ply,
                "is_sh0_rgb": data.format_info.is_sh0_rgb,
                "sh_degree": data.format_info.sh_degree,
            }
            if data.source_path:
                output["source_path"] = data.source_path

        # Convert numpy arrays to lists with precision control
        output["data"] = {
            "means": self._array_to_list(data.means[:n], precision),
            "scales": self._array_to_list(data.scales[:n], precision),
            "quats": self._array_to_list(data.quats[:n], precision),
            "opacities": self._array_to_list(data.opacities[:n], precision),
            "sh0": self._array_to_list(data.sh0[:n], precision),
        }

        # Include shN if present
        if data.shN is not None and data.shN.size > 0:
            output["data"]["shN"] = self._array_to_list(data.shN[:n], precision)

        # Write JSON file
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=indent)

        logger.debug(
            "Exported %d gaussians to %s (%.1f KB)",
            n,
            path,
            output_path.stat().st_size / 1024,
        )

    def export_sequence(
        self,
        frames: Iterator[GaussianData],
        output_dir: str,
        **options: Any,
    ) -> int:
        """Export sequence of frames to JSON files.

        This method handles batch export of multiple frames. It should:
        1. Create output directory
        2. Iterate through frames
        3. Export each frame with proper naming
        4. Report progress
        5. Handle errors gracefully (continue on single-frame failures)
        6. Return count of successfully exported frames

        Parameters
        ----------
        frames : Iterator[GaussianData]
            Iterator of frames to export. Can be a list, generator, etc.
        output_dir : str
            Output directory. Will be created if needed.
        **options : Any
            Export options (passed to export() for each frame):
            - filename_pattern: str = "frame_{:06d}.json"
            - progress_callback: callable(frame_idx, total) = None
            - All options supported by export()

        Returns
        -------
        int
            Number of frames successfully exported.

        Notes
        -----
        This implementation:
        - Continues on individual frame failures (logs error, moves on)
        - Supports unknown-length iterators (total=None in callback)
        - Uses customizable filename patterns
        """
        # Parse options
        filename_pattern = options.get("filename_pattern", "frame_{:06d}.json")
        progress_callback = options.get("progress_callback")

        # Ensure output directory exists
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Export frames
        exported_count = 0

        for idx, frame in enumerate(frames):
            try:
                # Generate filename from pattern
                filename = filename_pattern.format(idx)
                frame_path = output_path / filename

                # Export single frame
                self.export(frame, str(frame_path), **options)
                exported_count += 1

                # Report progress
                if progress_callback:
                    progress_callback(idx, None)  # Total unknown for iterators

            except Exception as e:
                # Log error but continue with other frames
                logger.error("Failed to export frame %d: %s", idx, e)
                continue

        logger.info(
            "Exported %d frames to %s",
            exported_count,
            output_dir,
        )

        return exported_count

    # =========================================================================
    # INTERNAL METHODS
    # =========================================================================

    def _array_to_list(self, arr: np.ndarray, precision: int) -> list[list[float]] | list[float]:
        """Convert numpy array to nested list with precision control.

        Parameters
        ----------
        arr : np.ndarray
            Input array [N, ...] or [N]
        precision : int
            Decimal places to round to

        Returns
        -------
        list
            Nested list structure matching input shape
        """
        if arr.ndim == 1:
            return [round(float(x), precision) for x in arr]
        else:
            return [[round(float(x), precision) for x in row] for row in arr]


# =============================================================================
# REGISTRATION HELPER
# =============================================================================


def register_demo_sink() -> None:
    """Register the demo sink with the registry.

    Call this function to make the demo sink available:

    >>> from src.plugins.demo.demo_sink import register_demo_sink
    >>> register_demo_sink()
    >>> # Now "demo-json" is available as an export format
    """
    from src.infrastructure.registry import DataSinkRegistry

    DataSinkRegistry.register("demo-json", DemoJsonSink)
    logger.info("Registered demo-json data sink")
