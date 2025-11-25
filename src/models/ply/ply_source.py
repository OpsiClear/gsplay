"""
PLY sequence data source implementation.

This module provides the DataSourceProtocol implementation for loading
PLY file sequences. It wraps the existing OptimizedPlyModel functionality
while conforming to the new unified data source interface.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.domain.data import GaussianData
from src.domain.interfaces import DataSourceProtocol, DataSourceMetadata
from src.infrastructure.io.path_io import UniversalPath
from src.infrastructure.io.discovery import discover_and_sort_ply_files

logger = logging.getLogger(__name__)


@dataclass
class PlySourceConfig:
    """Configuration for PLY data source.

    Attributes:
        ply_folder: Path to folder containing PLY files
        enable_concurrent_prefetch: Enable background prefetching (default: True)
        processing_mode: Processing mode (default: "all_gpu")
        device: Target device (default: "cuda")
    """

    ply_folder: str = "."
    enable_concurrent_prefetch: bool = True
    processing_mode: str = "all_gpu"
    device: str = "cuda"


class PlyDataSource(DataSourceProtocol):
    """PLY sequence data source.

    Implements DataSourceProtocol for loading sequences of PLY files.
    Internally uses OptimizedPlyModel for efficient frame loading with optional prefetch.

    Example
    -------
    >>> source = PlyDataSource({"ply_folder": "/path/to/plys"})
    >>> frame = source.get_frame(0)
    >>> print(f"Loaded {frame.n_gaussians} gaussians")
    """

    @classmethod
    def metadata(cls) -> DataSourceMetadata:
        """Return metadata about this source type."""
        return DataSourceMetadata(
            name="PLY Sequence",
            description="Load PLY files from a folder",
            file_extensions=[".ply"],
            config_schema=PlySourceConfig,
            supports_streaming=True,
            supports_seeking=True,
        )

    @classmethod
    def can_load(cls, path: str) -> bool:
        """Check if this loader can handle the given path.

        Returns True if:
        - Path is a directory containing .ply files
        - Path is a .ply file
        """
        try:
            p = UniversalPath(path)
            if p.is_dir():
                # Check if directory contains PLY files
                ply_files = list(p.glob("*.ply"))
                return len(ply_files) > 0
            # Single PLY file
            return str(p).lower().endswith(".ply")
        except Exception:
            return False

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize PLY data source.

        Parameters
        ----------
        config : dict[str, Any]
            Configuration dictionary with keys from PlySourceConfig
        """
        # Parse config
        self._config = PlySourceConfig(
            ply_folder=config.get("ply_folder", "."),
            enable_concurrent_prefetch=config.get("enable_concurrent_prefetch", True),
            processing_mode=config.get("processing_mode", "all_gpu"),
            device=config.get("device", "cuda"),
        )

        # Discover PLY files
        self._ply_folder = UniversalPath(self._config.ply_folder)
        self._ply_files = discover_and_sort_ply_files(self._config.ply_folder)

        if not self._ply_files:
            raise ValueError(f"No PLY files found in: {self._config.ply_folder}")

        logger.info(
            "PlyDataSource initialized with %d PLY files from %s",
            len(self._ply_files),
            self._ply_folder,
        )

        # Lazy-initialize the underlying model
        self._model = None

    def _ensure_model(self) -> None:
        """Ensure the underlying OptimizedPlyModel is initialized."""
        if self._model is not None:
            return

        # Import here to avoid circular imports
        from src.models.ply.optimized_model import OptimizedPlyModel

        self._model = OptimizedPlyModel(
            ply_files=[str(p) for p in self._ply_files],
            device=self._config.device,
            enable_concurrent_prefetch=self._config.enable_concurrent_prefetch,
            processing_mode=self._config.processing_mode,
        )

    @property
    def total_frames(self) -> int:
        """Total number of frames available."""
        return len(self._ply_files)

    @property
    def ply_folder(self) -> UniversalPath:
        """Get the PLY folder path."""
        return self._ply_folder

    @property
    def device(self) -> str:
        """Get the device."""
        return self._config.device

    def get_frame(self, index: int) -> GaussianData:
        """Get frame at specific index.

        Parameters
        ----------
        index : int
            Frame index (0-based)

        Returns
        -------
        GaussianData
            Frame data
        """
        if index < 0 or index >= self.total_frames:
            raise IndexError(f"Frame index {index} out of range [0, {self.total_frames})")

        # Convert to normalized time
        if self.total_frames == 1:
            normalized_time = 0.0
        else:
            normalized_time = index / (self.total_frames - 1)

        return self.get_frame_at_time(normalized_time)

    def get_frame_at_time(self, normalized_time: float) -> GaussianData:
        """Get frame at normalized time [0, 1].

        Parameters
        ----------
        normalized_time : float
            Normalized time in range [0.0, 1.0]

        Returns
        -------
        GaussianData
            Frame data at the specified time
        """
        self._ensure_model()

        # Get data from underlying model
        result = self._model.get_gaussians_at_normalized_time(normalized_time)

        if result is None:
            raise RuntimeError(f"Failed to load frame at time {normalized_time}")

        # Convert to GaussianData
        # Result can be GSData (CPU) or GSTensor (GPU)
        from gsply import GSData
        from gsply.torch import GSTensor

        if isinstance(result, GSTensor):
            return GaussianData.from_gstensor(result, source_path=str(self._ply_folder))
        elif isinstance(result, GSData):
            return GaussianData.from_gsdata(result, source_path=str(self._ply_folder))
        else:
            raise TypeError(f"Unexpected result type: {type(result)}")

    # =========================================================================
    # Backward compatibility - expose underlying model features
    # =========================================================================

    def get_recommended_max_scale(self) -> float:
        """Get recommended max_scale from first frame (for UI initialization)."""
        self._ensure_model()
        return self._model.get_recommended_max_scale()

    def get_last_profile(self) -> dict[str, Any] | None:
        """Get latest frame load profile."""
        if self._model is None:
            return None
        return self._model.get_last_profile()

    # =========================================================================
    # TimeSampledModel compatibility (for existing viewer code)
    # =========================================================================

    def get_gaussians_at_normalized_time(self, normalized_time: float):
        """Compatibility method for existing viewer code.

        Returns raw GSData/GSTensor (not GaussianData) for backward compatibility.
        """
        self._ensure_model()
        return self._model.get_gaussians_at_normalized_time(normalized_time)

    def get_total_frames(self) -> int:
        """Compatibility method for existing viewer code."""
        return self.total_frames

    def get_frame_time(self, frame_idx: int) -> float:
        """Compatibility method for existing viewer code."""
        if self.total_frames <= 1:
            return 0.0
        return frame_idx / (self.total_frames - 1)
