"""Interpolated PLY Model - PLY sequence with smooth interpolation between frames.

This module wraps OptimizedPlyModel to add interpolation capability,
allowing queries at arbitrary times between discrete PLY frames.

Example
-------
>>> model = InterpolatedPlyModel({"ply_folder": "./frames/"})
>>> # Query at t=0.5 - blends frame 0 and frame 1
>>> data = model.get_frame_at_source_time(0.5)
>>> # Query at t=2.7 - blends frame 2 and frame 3
>>> data = model.get_frame_at_source_time(2.7)
"""

from __future__ import annotations

import logging
from typing import Any

from src.domain.data import GaussianData
from src.domain.interfaces import (
    HealthCheckResult,
    HealthStatus,
    PluginState,
    SourceMetadata,
)
from src.domain.interpolation import (
    InterpolationMethod,
    interpolate_gaussian_data,
)
from src.domain.lifecycle import LifecycleMixin
from src.domain.time import TimeDomain
from src.models.ply.optimized_model import OptimizedPlyModel


logger = logging.getLogger(__name__)


class InterpolatedPlyModel(LifecycleMixin):
    """PLY sequence with interpolation between keyframes.

    Wraps an OptimizedPlyModel and adds interpolation capability,
    allowing queries at arbitrary times between discrete frames.

    Implements both BaseGaussianSource and InterpolatableSource protocols.

    Attributes
    ----------
    interpolation_method : InterpolationMethod
        Method used for interpolation (default: LINEAR)

    Example
    -------
    >>> model = InterpolatedPlyModel({"ply_folder": "./frames/"})
    >>> model.time_domain.is_continuous  # True
    >>> data = model.get_frame_at_source_time(0.5)  # Blends frames 0 and 1
    """

    # --- BaseGaussianSource Protocol Class Methods ---

    @classmethod
    def metadata(cls) -> SourceMetadata:
        """Return metadata about this source type."""
        return SourceMetadata(
            name="Interpolated PLY",
            description="PLY sequence with smooth interpolation between frames",
            file_extensions=[".ply"],
            supports_streaming=True,
            supports_seeking=True,
            version="1.0.0",
        )

    @classmethod
    def can_load(cls, path: str) -> bool:
        """Check if this source can handle the given path.

        Delegates to OptimizedPlyModel.
        """
        return OptimizedPlyModel.can_load(path)

    def __init__(
        self,
        config: dict[str, Any],
        device: str = "cuda",
        interpolation_method: InterpolationMethod = InterpolationMethod.LINEAR,
    ):
        """Initialize interpolated PLY model.

        Parameters
        ----------
        config : dict
            Configuration dictionary with 'ply_folder' key.
            Supports time configuration from OptimizedPlyConfig:
            - source_fps: Original capture FPS
            - frame_count, frame_start, frame_end: Frame range
            - playback_fps: Suggested playback speed
            - lock_playback_fps: Lock playback FPS in UI
            - autoplay: Auto-start playback
        device : str
            Target device (default "cuda")
        interpolation_method : InterpolationMethod
            Interpolation method (default LINEAR)
        """
        super().__init__()

        self._base_model = OptimizedPlyModel(config, device=device)
        self._interpolation_method = interpolation_method
        self._device = device

        # Cache for keyframes (LRU-style with fixed size)
        self._keyframe_cache: dict[int, GaussianData] = {}
        self._cache_order: list[int] = []
        self._cache_size = 4  # Keep 4 keyframes in memory

        # Build time domain with source_fps from base model
        self._time_domain = TimeDomain.interpolated(
            keyframe_times=[float(i) for i in range(self._base_model.total_frames)],
            source_fps=self._base_model.source_fps,
        )

        self._state = PluginState.READY
        logger.info(
            f"InterpolatedPlyModel initialized with {self._base_model.total_frames} keyframes"
            + (
                f" (source_fps={self._base_model.source_fps})"
                if self._base_model.source_fps
                else ""
            )
        )

    # --- BaseGaussianSource Protocol Properties ---

    @property
    def total_frames(self) -> int:
        """Total number of keyframes available."""
        return self._base_model.total_frames

    @property
    def time_domain(self) -> TimeDomain:
        """Get the time domain for this source.

        Returns interpolated keyframe time domain (continuous).
        """
        return self._time_domain

    # --- Time Configuration Properties (delegated to base model) ---

    @property
    def source_fps(self) -> float | None:
        """Original capture FPS, or None if not specified."""
        return self._base_model.source_fps

    @source_fps.setter
    def source_fps(self, value: float | None) -> None:
        """Set source FPS (for runtime configuration from UI).

        Updates both the base model and rebuilds the time domain.
        """
        self._base_model.source_fps = value
        # Rebuild time domain with new source_fps
        self._time_domain = TimeDomain.interpolated(
            keyframe_times=[float(i) for i in range(self._base_model.total_frames)],
            source_fps=value,
        )

    @property
    def playback_fps(self) -> float:
        """Suggested playback FPS."""
        return self._base_model.playback_fps

    @property
    def lock_playback_fps(self) -> bool:
        """Whether UI should lock playback FPS to configured value."""
        return self._base_model.lock_playback_fps

    @property
    def autoplay(self) -> bool:
        """Whether to auto-start playback on load."""
        return self._base_model.autoplay

    # --- InterpolatableSource Protocol ---

    @property
    def keyframe_count(self) -> int:
        """Number of keyframes available."""
        return self._base_model.total_frames

    def get_keyframe(self, index: int) -> GaussianData:
        """Get data at a specific keyframe index.

        Uses caching to avoid reloading recently accessed frames.

        Parameters
        ----------
        index : int
            Keyframe index (0 to keyframe_count - 1)

        Returns
        -------
        GaussianData
            Data at the specified keyframe
        """
        # Clamp index
        index = max(0, min(index, self.keyframe_count - 1))

        # Check cache
        if index in self._keyframe_cache:
            # Move to end of LRU order
            if index in self._cache_order:
                self._cache_order.remove(index)
            self._cache_order.append(index)
            return self._keyframe_cache[index]

        # Load from base model
        normalized_time = index / max(1, self.keyframe_count - 1)
        data = self._base_model.get_frame_at_time(normalized_time)

        # Cache management (LRU eviction)
        if len(self._keyframe_cache) >= self._cache_size:
            # Remove oldest entry
            oldest_idx = self._cache_order.pop(0)
            del self._keyframe_cache[oldest_idx]

        self._keyframe_cache[index] = data
        self._cache_order.append(index)

        return data

    def get_keyframe_time(self, index: int) -> float:
        """Get the source-native time for a keyframe.

        Parameters
        ----------
        index : int
            Keyframe index

        Returns
        -------
        float
            Time in source units (frames)
        """
        return float(index)

    @property
    def interpolation_method(self) -> str:
        """Get the interpolation method name."""
        return self._interpolation_method.name.lower()

    # --- Time-Based Access ---

    def get_frame_at_source_time(self, source_time: float) -> GaussianData:
        """Get interpolated frame at source time (frames).

        Parameters
        ----------
        source_time : float
            Time in frames (e.g., 0.5 = halfway between frame 0 and 1)

        Returns
        -------
        GaussianData
            Interpolated Gaussian data
        """
        # Clamp to valid range
        source_time = max(0.0, min(source_time, self.keyframe_count - 1))

        # Find surrounding keyframes
        frame_idx = int(source_time)
        t = source_time - frame_idx

        # Handle exact keyframe hit (or very close)
        if t < 1e-6:
            return self.get_keyframe(frame_idx)

        # Handle last frame
        if frame_idx >= self.keyframe_count - 1:
            return self.get_keyframe(self.keyframe_count - 1)

        # Load surrounding keyframes
        data0 = self.get_keyframe(frame_idx)
        data1 = self.get_keyframe(frame_idx + 1)

        # Interpolate
        return interpolate_gaussian_data(
            data0,
            data1,
            t,
            method=self._interpolation_method,
        )

    def get_frame_at_time(self, normalized_time: float) -> GaussianData:
        """Get frame at normalized time [0, 1].

        BaseGaussianSource protocol method.

        Parameters
        ----------
        normalized_time : float
            Normalized time in [0.0, 1.0]

        Returns
        -------
        GaussianData
            Interpolated frame data
        """
        source_time = self._time_domain.from_normalized(normalized_time)
        return self.get_frame_at_source_time(source_time)

    # --- Backward Compatibility ---

    def get_gaussians_at_normalized_time(self, normalized_time: float) -> GaussianData:
        """Backward compatible method."""
        return self.get_frame_at_time(normalized_time)

    def get_total_frames(self) -> int:
        """Backward compatible method."""
        return self.total_frames

    def get_frame_time(self, frame_idx: int) -> float:
        """Backward compatible method."""
        if self.keyframe_count <= 1:
            return 0.0
        return frame_idx / (self.keyframe_count - 1)

    # --- Optional Protocol Support ---

    def get_recommended_max_scale(self) -> float | None:
        """Delegate to base model for scale recommendation."""
        if hasattr(self._base_model, "get_recommended_max_scale"):
            return self._base_model.get_recommended_max_scale()
        return None

    # --- Lifecycle Management ---

    def on_init(self) -> None:
        """Initialize resources."""
        if hasattr(self._base_model, "on_init"):
            self._base_model.on_init()

    def on_load(self) -> None:
        """Prepare for active use."""
        if hasattr(self._base_model, "on_load"):
            self._base_model.on_load()

    def on_unload(self) -> None:
        """Release resources when inactive."""
        # Clear keyframe cache
        self._keyframe_cache.clear()
        self._cache_order.clear()

        if hasattr(self._base_model, "on_unload"):
            self._base_model.on_unload()

    def on_shutdown(self, timeout: float = 5.0) -> None:
        """Final cleanup."""
        self._keyframe_cache.clear()
        self._cache_order.clear()

        if hasattr(self._base_model, "on_shutdown"):
            self._base_model.on_shutdown(timeout)

        self._state = PluginState.TERMINATED

    # --- Health Monitoring ---

    def health_check(self) -> HealthCheckResult:
        """Perform health check."""
        if self._state != PluginState.READY:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Model in {self._state.name} state",
            )

        return HealthCheckResult(
            status=HealthStatus.HEALTHY,
            message=f"Ready with {self.keyframe_count} keyframes (interpolated)",
            details={
                "keyframe_count": self.keyframe_count,
                "cache_size": len(self._keyframe_cache),
                "interpolation_method": self.interpolation_method,
            },
        )

    def get_diagnostics(self) -> dict[str, Any]:
        """Get diagnostic information."""
        return {
            "class": self.__class__.__name__,
            "state": self._state.name,
            "keyframe_count": self.keyframe_count,
            "interpolation_method": self.interpolation_method,
            "cache_size": len(self._keyframe_cache),
            "cache_keys": list(self._keyframe_cache.keys()),
            "device": self._device,
        }


__all__ = ["InterpolatedPlyModel"]
