"""Base class for data source plugins.

Provides sensible defaults and utility methods for implementing
the BaseGaussianSource protocol.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from src.domain.interfaces import (
    BaseGaussianSource,
    SourceMetadata,
    PluginState,
    HealthStatus,
    HealthCheckResult,
)
from src.domain.lifecycle import LifecycleMixin
from src.domain.data import GaussianData

logger = logging.getLogger(__name__)


class BaseDataSource(LifecycleMixin, ABC):
    """Base class for data source plugins.

    Provides:
    - Lifecycle management via LifecycleMixin
    - Health monitoring defaults
    - Utility methods for time/index conversion
    - Config handling boilerplate

    Subclasses only need to implement:
    - total_frames (property)
    - get_frame_at_time(normalized_time) -> GaussianData
    - can_load(path) -> bool (classmethod)

    The metadata() classmethod is provided by the @source_metadata decorator.

    Example
    -------
    >>> @source_metadata(name="My Format", ...)
    >>> class MySource(BaseDataSource):
    ...     @property
    ...     def total_frames(self) -> int:
    ...         return len(self._files)
    ...
    ...     def get_frame_at_time(self, normalized_time: float) -> GaussianData:
    ...         index = self._time_to_index(normalized_time)
    ...         return self._load_frame(self._files[index])
    ...
    ...     @classmethod
    ...     def can_load(cls, path: str) -> bool:
    ...         return path.endswith(".myformat")
    """

    # Override in subclass or use @source_metadata decorator
    _source_metadata: SourceMetadata | None = None

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the data source.

        Parameters
        ----------
        config : dict
            Configuration dictionary. Common keys:
            - device: Target device (default "cuda")
            - Any source-specific config
        """
        super().__init__()
        self.config = config
        self.device = config.get("device", "cuda")

        # Transition to READY (subclasses can override on_init for heavy setup)
        self._state = PluginState.READY

    @classmethod
    def metadata(cls) -> SourceMetadata:
        """Return metadata about this source type.

        Override this method or use the @source_metadata decorator.
        """
        if cls._source_metadata is not None:
            return cls._source_metadata

        # Default metadata if not decorated
        return SourceMetadata(
            name=cls.__name__,
            description=f"Data source: {cls.__name__}",
            file_extensions=[],
            config_schema=None,
        )

    @classmethod
    @abstractmethod
    def can_load(cls, path: str) -> bool:
        """Check if this source can handle the given path.

        Must be implemented by subclasses.
        """
        ...

    @property
    @abstractmethod
    def total_frames(self) -> int:
        """Total number of frames available.

        Must be implemented by subclasses.
        """
        ...

    @abstractmethod
    def get_frame_at_time(self, normalized_time: float) -> GaussianData:
        """Get frame at normalized time [0, 1].

        Must be implemented by subclasses.

        Parameters
        ----------
        normalized_time : float
            Normalized time in range [0.0, 1.0]

        Returns
        -------
        GaussianData
            Frame data at the specified time
        """
        ...

    # --- Time Domain Support ---

    @property
    def time_domain(self):
        """Get the time domain for this source.

        Default implementation returns discrete frames based on total_frames.
        Override in subclasses for continuous time support.

        Returns
        -------
        TimeDomain
            Description of how this source represents time
        """
        from src.domain.time import TimeDomain
        return TimeDomain.discrete(self.total_frames)

    def get_frame_at_source_time(self, source_time: float) -> GaussianData:
        """Get frame at source-native time.

        Default implementation converts source time to normalized time
        and calls get_frame_at_time(). Override for custom behavior.

        Parameters
        ----------
        source_time : float
            Time in source-native units

        Returns
        -------
        GaussianData
            Frame data at the specified time
        """
        td = self.time_domain
        normalized_time = td.to_normalized(source_time)
        return self.get_frame_at_time(normalized_time)

    # --- Utility Methods ---

    def _time_to_index(self, normalized_time: float) -> int:
        """Convert normalized time to frame index.

        Parameters
        ----------
        normalized_time : float
            Normalized time in [0.0, 1.0]

        Returns
        -------
        int
            Frame index in [0, total_frames-1]
        """
        if self.total_frames <= 1:
            return 0

        index = int(round(normalized_time * (self.total_frames - 1)))
        return max(0, min(index, self.total_frames - 1))

    def _index_to_time(self, index: int) -> float:
        """Convert frame index to normalized time.

        Parameters
        ----------
        index : int
            Frame index

        Returns
        -------
        float
            Normalized time in [0.0, 1.0]
        """
        if self.total_frames <= 1:
            return 0.0
        return index / (self.total_frames - 1)

    def _create_empty_gaussian_data(self) -> GaussianData:
        """Create empty GaussianData (useful for error cases).

        Returns
        -------
        GaussianData
            Empty GaussianData with zero Gaussians
        """
        return GaussianData(
            means=np.zeros((0, 3), dtype=np.float32),
            scales=np.zeros((0, 3), dtype=np.float32),
            quats=np.zeros((0, 4), dtype=np.float32),
            opacities=np.zeros((0,), dtype=np.float32),
            sh0=np.zeros((0, 3), dtype=np.float32),
        )

    # --- Backward Compatibility ---

    def get_gaussians_at_normalized_time(self, normalized_time: float) -> Any:
        """Backward compatible method - delegates to get_frame_at_time.

        This method exists for compatibility with code that uses the old
        TimeSampledModel interface. New code should use get_frame_at_time.
        """
        return self.get_frame_at_time(normalized_time)

    def get_total_frames(self) -> int:
        """Backward compatible method - returns total_frames property."""
        return self.total_frames

    def get_frame_time(self, frame_idx: int) -> float:
        """Backward compatible method - returns normalized time for frame."""
        return self._index_to_time(frame_idx)

    # --- Health Check ---

    def health_check(self) -> HealthCheckResult:
        """Perform health check and return result."""
        if self._state != PluginState.READY:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Source in {self._state.name} state",
            )

        return HealthCheckResult(
            status=HealthStatus.HEALTHY,
            message=f"Ready with {self.total_frames} frames",
            details={
                "total_frames": self.total_frames,
                "device": self.device,
            },
        )

    def get_diagnostics(self) -> dict[str, Any]:
        """Get diagnostic information about this source."""
        return {
            "class": self.__class__.__name__,
            "state": self._state.name,
            "total_frames": self.total_frames,
            "device": self.device,
            "config_keys": list(self.config.keys()),
        }


__all__ = ["BaseDataSource"]
