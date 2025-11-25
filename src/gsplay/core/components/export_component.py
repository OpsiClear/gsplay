"""
Export component for handling all export operations.

This component is responsible for:
- Exporting frame sequences to PLY files
- Managing export settings and paths
- Progress tracking
- Integration with exporter factory and data sink registry

Supports two modes:
1. Legacy mode: Uses GSTensor and ExporterFactory
2. New mode: Uses GaussianData and DataSinkRegistry
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from tqdm import tqdm

from src.domain.entities import GSTensor
from src.domain.interfaces import ModelInterface
from src.gsplay.config.settings import ExportSettings
from src.infrastructure.exporters import ExporterFactory
from src.infrastructure.io.path_io import UniversalPath
from src.gsplay.interaction.events import EventBus, EventType

if TYPE_CHECKING:
    from src.domain.data import GaussianData

logger = logging.getLogger(__name__)


@contextmanager
def suppress_excessive_logging():
    """
    Context manager to temporarily suppress excessive logging during export.

    Reduces log level for noisy loggers (gsply, PLY writer) to WARNING during export
    to keep console output clean with tqdm progress bar.
    """
    # Loggers to suppress during export
    noisy_loggers = [
        "gsply.torch.compression",
        "gsply.torch",
        "src.infrastructure.processing.ply.writer",
        "src.infrastructure.exporters.compressed_ply_exporter",
        "src.infrastructure.exporters.ply_exporter",
    ]

    # Store original log levels (use effective level to handle hierarchy)
    original_levels = {}
    for logger_name in noisy_loggers:
        log = logging.getLogger(logger_name)
        # Store effective level (considers parent loggers)
        original_levels[logger_name] = log.getEffectiveLevel()
        # Set level explicitly to WARNING to suppress INFO/DEBUG
        log.setLevel(logging.WARNING)

    try:
        yield
    finally:
        # Restore original log levels
        for logger_name, original_level in original_levels.items():
            log = logging.getLogger(logger_name)
            # If logger had no explicit level set, remove it to restore parent's level
            if original_level == logging.NOTSET:
                log.setLevel(logging.NOTSET)
            else:
                log.setLevel(original_level)


class ExportComponent:
    """
    Component responsible for exporting Gaussian data.

    Handles:
    - Frame sequence export
    - Format selection
    - Path management
    - Progress tracking
    - Event emission for export status
    """

    def __init__(
        self,
        event_bus: EventBus | None = None,
        default_output_dir: (Path | str) | None = None,
    ):
        """
        Initialize export component.

        Parameters
        ----------
        event_bus : EventBus | None
            Event bus for emitting export events
        default_output_dir : (Path | str) | None
            Default directory for exports
        """
        self.event_bus = event_bus
        self.default_output_dir = (
            UniversalPath(default_output_dir) if default_output_dir else None
        )
        logger.debug("ExportComponent initialized")

    def export_frame_sequence(
        self,
        model: ModelInterface,
        export_settings: ExportSettings,
        edit_applier: Callable[[GSTensor], GSTensor] | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> bool:
        """
        Export frame sequence using configured exporter.

        Parameters
        ----------
        model : ModelInterface
            Model to export frames from
        export_settings : ExportSettings
            Export configuration
        edit_applier : Callable[[GSTensor], GSTensor] | None
            Function to apply edits to GSTensor before export
        progress_callback : Callable[[int, int], None] | None
            Callback for progress updates: callback(current_frame, total_frames)

        Returns
        -------
        bool
            True if export succeeded, False otherwise
        """
        try:
            # Determine output directory
            output_dir = self._resolve_output_dir(export_settings.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Exporting to: {output_dir}")

            # Create exporter
            exporter = ExporterFactory.get_exporter(export_settings.format)

            # Get frame range
            total_frames = model.get_total_frames()
            start_frame = export_settings.start_frame or 0
            end_frame = export_settings.end_frame or total_frames - 1

            # Validate range
            if start_frame < 0 or end_frame >= total_frames or start_frame > end_frame:
                raise ValueError(
                    f"Invalid frame range [{start_frame}, {end_frame}] "
                    f"for model with {total_frames} frames"
                )

            frames_to_export = range(start_frame, end_frame + 1)
            num_frames = len(frames_to_export)

            logger.info(
                f"Exporting {num_frames} frames "
                f"(frame {start_frame} to {end_frame}) "
                f"in format '{export_settings.format}'"
            )

            # Export each frame with tqdm progress bar
            # Suppress excessive logging during export to keep console clean
            use_tqdm = sys.stderr.isatty()
            with suppress_excessive_logging():
                with tqdm(
                    total=num_frames,
                    desc="Exporting frames",
                    unit="frame",
                    file=sys.stderr,
                    disable=not use_tqdm,
                    dynamic_ncols=True,
                    miniters=1,
                    mininterval=0.1,
                ) as pbar:
                    for idx, frame_idx in enumerate(frames_to_export):
                        # Get gaussians for this frame
                        frame_time = model.get_frame_time(frame_idx)
                        gaussian_data = model.get_gaussians_at_normalized_time(
                            frame_time
                        )

                        if gaussian_data is None:
                            # Use tqdm.write() for logging during progress bar to avoid breaking the bar
                            tqdm.write(
                                f"Warning: No data for frame {frame_idx}, skipping",
                                file=sys.stderr,
                            )
                            pbar.update(1)
                            continue

                        # Apply edits if provided
                        # edit_applier uses export device, so data will be on correct device after this
                        if edit_applier:
                            gaussian_data = edit_applier(gaussian_data)
                        else:
                            # No edits - still need to ensure data is on export device
                            export_device = export_settings.export_device
                            if export_device:
                                from src.gsplay.processing.gs_bridge import (
                                    DefaultGSBridge,
                                )

                                bridge = DefaultGSBridge()
                                gaussian_data, _ = bridge.ensure_tensor_on_device(
                                    gaussian_data, export_device
                                )

                        # Determine output path
                        output_path = output_dir / f"frame_{frame_idx:05d}.ply"

                        # Export frame
                        exporter.export_frame(
                            gaussian_data,
                            output_path,
                            **export_settings.exporter_options,
                        )

                        # Update progress bar
                        pbar.update(1)

                        # Also call progress callback if provided (for backward compatibility)
                        if progress_callback:
                            progress_callback(idx + 1, num_frames)

            # Emit export completed event
            if self.event_bus:
                self.event_bus.emit(
                    EventType.EXPORT_COMPLETED,
                    source="export_component",
                    num_frames=num_frames,
                    output_dir=str(output_dir),
                )

            logger.info(
                f"Export completed: {num_frames} frames written to {output_dir}"
            )
            return True

        except Exception as e:
            logger.error(f"Export failed: {e}", exc_info=True)

            # Emit export failed event
            if self.event_bus:
                self.event_bus.emit(
                    EventType.EXPORT_FAILED, source="export_component", error=str(e)
                )

            return False

    def export_single_frame(
        self,
        gaussian_data: GSTensor,
        output_path: Path | str,
        format: str = "ply",
        **exporter_options,
    ) -> bool:
        """
        Export a single frame.

        Parameters
        ----------
        gaussian_data : GSTensor
            Gaussian data to export
        output_path : Path | str
            Output file path
        format : str
            Export format
        **exporter_options
            Format-specific options

        Returns
        -------
        bool
            True if export succeeded
        """
        try:
            output_path = UniversalPath(output_path)
            exporter = ExporterFactory.get_exporter(format)

            logger.info(f"Exporting single frame to: {output_path}")
            exporter.export_frame(gaussian_data, output_path, **exporter_options)

            logger.info("Single frame exported successfully")
            return True

        except Exception as e:
            logger.error(f"Single frame export failed: {e}", exc_info=True)
            return False

    def _resolve_output_dir(self, requested_dir: (Path | str) | None) -> UniversalPath:
        """
        Resolve the output directory.

        Parameters
        ----------
        requested_dir : Optional[Path | str]
            Requested output directory

        Returns
        -------
        UniversalPath
            Resolved output directory
        """
        if requested_dir:
            return UniversalPath(requested_dir)
        elif self.default_output_dir:
            return self.default_output_dir
        else:
            # Fallback to current directory with timestamp
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return UniversalPath(f"./export_{timestamp}")

    def set_default_output_dir(self, output_dir: Path | str) -> None:
        """Set the default output directory for exports."""
        self.default_output_dir = UniversalPath(output_dir)
        logger.debug(f"Default export directory set to: {output_dir}")

    # =========================================================================
    # GaussianData-based export (New Registry API)
    # =========================================================================

    def export_gaussian_data(
        self,
        data: GaussianData,
        output_path: Path | str,
        sink_format: str = "ply",
        **options,
    ) -> bool:
        """Export GaussianData using DataSinkRegistry.

        This is the new API that works with GaussianData abstraction.

        Parameters
        ----------
        data : GaussianData
            Gaussian data to export
        output_path : Path | str
            Output file path
        sink_format : str
            Sink format name (e.g., "ply", "compressed-ply")
        **options
            Format-specific export options

        Returns
        -------
        bool
            True if export succeeded

        Example
        -------
        >>> from src.domain.data import GaussianData
        >>> data = GaussianData.from_gsdata(gsdata)
        >>> component.export_gaussian_data(data, "output.ply", "ply")
        """
        try:
            from src.infrastructure.registry import DataSinkRegistry, register_defaults

            # Ensure registry is initialized
            register_defaults()

            output_path = UniversalPath(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Get sink from registry
            sink_class = DataSinkRegistry.get(sink_format)
            if sink_class is None:
                available = DataSinkRegistry.names()
                raise ValueError(
                    f"Unknown sink format: '{sink_format}'. "
                    f"Available: {', '.join(available)}"
                )

            sink = sink_class()

            logger.info(f"Exporting GaussianData to: {output_path}")
            sink.export(data, str(output_path), **options)

            logger.info("Export completed successfully")
            return True

        except Exception as e:
            logger.error(f"Export failed: {e}", exc_info=True)
            return False

    def export_gaussian_data_sequence(
        self,
        data_source,  # DataSourceProtocol
        output_dir: Path | str,
        sink_format: str = "ply",
        edit_applier: Callable[[GaussianData], GaussianData] | None = None,
        start_frame: int = 0,
        end_frame: int | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
        **options,
    ) -> bool:
        """Export frame sequence using GaussianData abstraction.

        This is the new API that works with DataSourceProtocol and DataSinkProtocol.

        Parameters
        ----------
        data_source : DataSourceProtocol
            Data source to export frames from
        output_dir : Path | str
            Output directory
        sink_format : str
            Sink format name (e.g., "ply", "compressed-ply")
        edit_applier : Callable[[GaussianData], GaussianData] | None
            Function to apply edits to GaussianData before export
        start_frame : int
            First frame to export
        end_frame : int | None
            Last frame to export (inclusive), None for all frames
        progress_callback : Callable[[int, int], None] | None
            Callback for progress updates
        **options
            Format-specific export options

        Returns
        -------
        bool
            True if export succeeded
        """
        try:
            from src.infrastructure.registry import DataSinkRegistry, register_defaults

            # Ensure registry is initialized
            register_defaults()

            output_dir = UniversalPath(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Get sink from registry
            sink_class = DataSinkRegistry.get(sink_format)
            if sink_class is None:
                available = DataSinkRegistry.names()
                raise ValueError(
                    f"Unknown sink format: '{sink_format}'. "
                    f"Available: {', '.join(available)}"
                )

            sink = sink_class()
            sink_meta = sink_class.metadata()

            # Determine frame range
            total_frames = data_source.total_frames
            if end_frame is None:
                end_frame = total_frames - 1

            if start_frame < 0 or end_frame >= total_frames or start_frame > end_frame:
                raise ValueError(
                    f"Invalid frame range [{start_frame}, {end_frame}] "
                    f"for source with {total_frames} frames"
                )

            frames_to_export = range(start_frame, end_frame + 1)
            num_frames = len(frames_to_export)

            logger.info(
                f"Exporting {num_frames} frames using GaussianData "
                f"(frame {start_frame} to {end_frame}) "
                f"in format '{sink_format}'"
            )

            # Export each frame
            use_tqdm = sys.stderr.isatty()
            with suppress_excessive_logging():
                with tqdm(
                    total=num_frames,
                    desc="Exporting frames",
                    unit="frame",
                    file=sys.stderr,
                    disable=not use_tqdm,
                    dynamic_ncols=True,
                    miniters=1,
                    mininterval=0.1,
                ) as pbar:
                    for idx, frame_idx in enumerate(frames_to_export):
                        # Get frame as GaussianData
                        frame_data = data_source.get_frame(frame_idx)

                        if frame_data is None:
                            tqdm.write(
                                f"Warning: No data for frame {frame_idx}, skipping",
                                file=sys.stderr,
                            )
                            pbar.update(1)
                            continue

                        # Apply edits if provided
                        if edit_applier:
                            frame_data = edit_applier(frame_data)

                        # Export frame
                        output_path = output_dir / f"frame_{frame_idx:05d}{sink_meta.file_extension}"
                        sink.export(frame_data, str(output_path), **options)

                        pbar.update(1)

                        if progress_callback:
                            progress_callback(idx + 1, num_frames)

            # Emit export completed event
            if self.event_bus:
                self.event_bus.emit(
                    EventType.EXPORT_COMPLETED,
                    source="export_component",
                    num_frames=num_frames,
                    output_dir=str(output_dir),
                )

            logger.info(
                f"Export completed: {num_frames} frames written to {output_dir}"
            )
            return True

        except Exception as e:
            logger.error(f"Export failed: {e}", exc_info=True)

            if self.event_bus:
                self.event_bus.emit(
                    EventType.EXPORT_FAILED, source="export_component", error=str(e)
                )

            return False


# Export public API
__all__ = ["ExportComponent"]
