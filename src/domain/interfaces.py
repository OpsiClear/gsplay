"""Domain interfaces (protocols) for dependency inversion.

This module defines the core protocols for the plugin system:
- BaseGaussianSource: Primary protocol for Gaussian data sources
- LifecycleProtocol: Lifecycle management for plugins
- HealthCheckable: Health monitoring for plugins
- DataSinkProtocol: Export/sink protocol (separate from sources)
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np


if TYPE_CHECKING:
    from src.domain.data import GaussianData
    from src.domain.entities import GSData, GSTensor
    from src.domain.time import TimeDomain


# ============================================================================
# Plugin State and Lifecycle
# ============================================================================


class PluginState(Enum):
    """Plugin lifecycle states."""

    CREATED = auto()  # Constructor called, not yet initialized
    INITIALIZING = auto()  # on_init() in progress
    READY = auto()  # Fully initialized, ready for use
    SUSPENDED = auto()  # Temporarily suspended (e.g., GPU memory freed)
    SHUTTING_DOWN = auto()  # on_shutdown() in progress
    TERMINATED = auto()  # Fully cleaned up


class HealthStatus(Enum):
    """Health status levels for plugins."""

    HEALTHY = auto()
    DEGRADED = auto()
    UNHEALTHY = auto()
    UNKNOWN = auto()


@dataclass
class HealthCheckResult:
    """Result of a health check."""

    status: HealthStatus
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    latency_ms: float | None = None


@runtime_checkable
class LifecycleProtocol(Protocol):
    """Protocol for plugins with managed lifecycle.

    Plugins implementing this protocol gain deterministic resource management
    and context manager support.
    """

    @property
    def state(self) -> PluginState:
        """Current lifecycle state."""
        ...

    def on_init(self) -> None:
        """Called after construction for heavy initialization.

        Use for:
        - GPU resource allocation
        - Thread pool creation
        - Network connections
        - File discovery

        Raises
        ------
        PluginInitError
            If initialization fails
        """
        ...

    def on_load(self) -> None:
        """Called when plugin becomes active/visible.

        Use for:
        - Loading first frame into cache
        - Establishing streaming connections
        - Warming caches
        """
        ...

    def on_unload(self) -> None:
        """Called when plugin becomes inactive.

        Use for:
        - Releasing GPU memory (tensors)
        - Pausing background threads
        - Closing network connections
        """
        ...

    def on_shutdown(self, timeout: float = 5.0) -> None:
        """Called for final cleanup with timeout.

        Parameters
        ----------
        timeout : float
            Maximum seconds to wait for cleanup. After timeout,
            resources are forcibly released.

        Use for:
        - ThreadPoolExecutor shutdown
        - File handle cleanup
        - Network disconnection
        """
        ...


@runtime_checkable
class HealthCheckable(Protocol):
    """Protocol for components that support health checks."""

    def health_check(self) -> HealthCheckResult:
        """Perform health check and return result."""
        ...

    def get_diagnostics(self) -> dict[str, Any]:
        """Get detailed diagnostic information."""
        ...


# ============================================================================
# Edit Pipeline Protocols (for dependency injection)
# ============================================================================


class EditManagerProtocol(Protocol):
    """Protocol for edit manager implementations.

    This allows models layer to work with edit managers without importing
    from viewer layer, using dependency injection.
    """

    def apply_edits(
        self,
        data: GSData | GSTensor,
        frame_idx: int = 0,
        scene_bounds: dict[str, Any] | None = None,
    ) -> GSTensor:
        """Apply configured edits to Gaussian data.

        Parameters
        ----------
        data : GSData | GSTensor
            Input Gaussian data
        frame_idx : int
            Current frame index (for caching)
        scene_bounds : dict | None
            Scene bounds for volume filtering

        Returns
        -------
        GSTensor
            Edited Gaussian data on GPU
        """
        ...

    def get_edit_settings_hash(self) -> int:
        """Get hash of current edit settings for cache invalidation."""
        ...


# Type alias for edit manager factory function
EditManagerFactory = Callable[[dict[str, Any], str], EditManagerProtocol]


# ============================================================================
# Data Loader Interface (legacy, kept for compatibility with data initialization)
# ============================================================================


class DataLoaderInterface(Protocol):
    """Standard interface for data loaders (used for camera/point initialization)."""

    def get_camera_data(self) -> dict[str, Any] | None: ...

    def get_points_for_initialization(self) -> np.ndarray: ...


# ============================================================================
# Unified Gaussian Source Protocol (NEW - Primary Plugin Interface)
# ============================================================================


@dataclass
class SourceMetadata:
    """Metadata for a Gaussian data source.

    Used by the registry to provide information about available sources,
    enable auto-detection, and support UI generation.

    Attributes
    ----------
    name : str
        Display name (e.g., "PLY Sequence")
    description : str
        Brief description of the source
    file_extensions : list[str]
        Supported file extensions [".ply", ".splat"]
    config_schema : type | None
        Config dataclass for validation and UI generation
    supports_streaming : bool
        Whether frames can be loaded on-demand (vs all upfront)
    supports_seeking : bool
        Whether random frame access is supported
    version : str
        Plugin version string
    """

    name: str
    description: str
    file_extensions: list[str] = field(default_factory=list)
    config_schema: type | None = None
    supports_streaming: bool = True
    supports_seeking: bool = True
    version: str = "1.0.0"


@runtime_checkable
class BaseGaussianSource(Protocol):
    """Primary protocol for Gaussian data sources.

    This is the unified interface that all data sources must implement.
    It replaces the previous DataSourceProtocol and TimeSampledModel.

    Required Methods (4 total):
    - metadata() -> SourceMetadata (classmethod)
    - can_load(path) -> bool (classmethod)
    - total_frames -> int (property)
    - get_frame_at_time(normalized_time) -> GaussianData

    Example
    -------
    >>> class MySource(BaseGaussianSource):
    ...     @classmethod
    ...     def metadata(cls) -> SourceMetadata:
    ...         return SourceMetadata(name="My Format", description="Load .xyz files")
    ...
    ...     @classmethod
    ...     def can_load(cls, path: str) -> bool:
    ...         return path.endswith(".xyz")
    ...
    ...     @property
    ...     def total_frames(self) -> int:
    ...         return len(self._files)
    ...
    ...     def get_frame_at_time(self, normalized_time: float) -> GaussianData:
    ...         index = int(normalized_time * (self.total_frames - 1))
    ...         return self._load_frame(index)
    """

    @classmethod
    def metadata(cls) -> SourceMetadata:
        """Return metadata about this source type.

        This method is called by the registry to discover available sources
        and their capabilities.

        Returns
        -------
        SourceMetadata
            Metadata describing this source's capabilities
        """
        ...

    @classmethod
    def can_load(cls, path: str) -> bool:
        """Check if this source can handle the given path.

        This method enables auto-detection when a user provides a path
        without specifying the source type.

        Parameters
        ----------
        path : str
            File or directory path to check

        Returns
        -------
        bool
            True if this source can load the path
        """
        ...

    @property
    def total_frames(self) -> int:
        """Total number of frames available.

        Returns
        -------
        int
            Number of frames (1 for static scenes)
        """
        ...

    def get_frame_at_time(self, normalized_time: float) -> GaussianData:
        """Get frame at normalized time [0, 1].

        This is the primary method for retrieving Gaussian data.

        Parameters
        ----------
        normalized_time : float
            Normalized time in range [0.0, 1.0]
            0.0 = first frame, 1.0 = last frame

        Returns
        -------
        GaussianData
            Frame data at the specified time

        Raises
        ------
        PluginLoadError
            If frame loading fails
        """
        ...

    # -------------------------------------------------------------------------
    # Time Domain Support (optional - has sensible defaults)
    # -------------------------------------------------------------------------

    @property
    def time_domain(self) -> TimeDomain:
        """Get the time domain for this source.

        The time domain describes how this source represents time:
        - Discrete frames (default for PLY sequences)
        - Continuous seconds (neural network models)
        - Interpolated keyframes (PLY with interpolation)

        Default implementation returns discrete frames based on total_frames.

        Returns
        -------
        TimeDomain
            Description of how this source represents time
        """
        ...

    def get_frame_at_source_time(self, source_time: float) -> GaussianData:
        """Get frame at source-native time.

        This method uses the source's native time representation
        (frames, seconds, etc.) rather than normalized [0, 1].

        For discrete sources, this typically rounds to nearest frame.
        For continuous sources, this returns interpolated/computed data.

        Parameters
        ----------
        source_time : float
            Time in source-native units (frames, seconds, etc.)
            Use time_domain to understand the units.

        Returns
        -------
        GaussianData
            Frame data at the specified time

        Note
        ----
        Default implementation converts to normalized time and calls
        get_frame_at_time(). Sources may override for better precision.
        """
        ...


# ============================================================================
# Optional Extension Protocols
# ============================================================================


@runtime_checkable
class ProfilableSource(Protocol):
    """Optional protocol for sources that expose performance profiling."""

    def get_last_profile(self) -> dict[str, Any] | None:
        """Get the latest frame load profile for diagnostics."""
        ...


@runtime_checkable
class RecommendedScaleSource(Protocol):
    """Optional protocol for sources that can recommend scale filtering."""

    def get_recommended_max_scale(self) -> float | None:
        """Get recommended max_scale percentile from first frame.

        Used for UI slider initialization.
        """
        ...


# ============================================================================
# Continuous Time Protocols
# ============================================================================


@runtime_checkable
class InterpolatableSource(Protocol):
    """Protocol for sources that support interpolation between keyframes.

    Sources implementing this protocol have discrete keyframes but can
    blend between them to produce data at arbitrary times.

    Example
    -------
    >>> class InterpolatedPlyModel(InterpolatableSource):
    ...     @property
    ...     def keyframe_count(self) -> int:
    ...         return len(self._ply_files)
    ...
    ...     def get_keyframe(self, index: int) -> GaussianData:
    ...         return self._load_frame(index)
    ...
    ...     def get_keyframe_time(self, index: int) -> float:
    ...         return float(index)  # Frames are at integer times
    ...
    ...     @property
    ...     def interpolation_method(self) -> str:
    ...         return "linear"
    """

    @property
    def keyframe_count(self) -> int:
        """Number of keyframes available.

        Returns
        -------
        int
            Total number of keyframes
        """
        ...

    def get_keyframe(self, index: int) -> GaussianData:
        """Get data at a specific keyframe index.

        Parameters
        ----------
        index : int
            Keyframe index (0 to keyframe_count - 1)

        Returns
        -------
        GaussianData
            Data at the specified keyframe
        """
        ...

    def get_keyframe_time(self, index: int) -> float:
        """Get the source-native time for a keyframe.

        Parameters
        ----------
        index : int
            Keyframe index

        Returns
        -------
        float
            Time in source units
        """
        ...

    @property
    def interpolation_method(self) -> str:
        """Get the interpolation method used.

        Returns
        -------
        str
            Method name: "nearest", "linear", "slerp", "cubic"
        """
        ...


@runtime_checkable
class ContinuousTimeSource(Protocol):
    """Protocol for sources that natively produce data at any time.

    Neural network models (4DGS, D-NeRF style) implement this protocol.
    They take time as an input and produce Gaussians directly without
    interpolation between discrete keyframes.

    Example
    -------
    >>> class My4DGSModel(ContinuousTimeSource):
    ...     def evaluate(self, t: float) -> GaussianData:
    ...         with torch.no_grad():
    ...             outputs = self.model(t)
    ...         return GaussianData.from_tensors(**outputs)
    ...
    ...     @property
    ...     def supports_batched_time(self) -> bool:
    ...         return True  # Can evaluate multiple times efficiently
    """

    def evaluate(self, t: float) -> GaussianData:
        """Evaluate the model at time t.

        Parameters
        ----------
        t : float
            Time in source-native units

        Returns
        -------
        GaussianData
            Gaussian data produced by the model
        """
        ...

    @property
    def supports_batched_time(self) -> bool:
        """Whether this source can evaluate multiple times efficiently.

        If True, evaluate_batch() can be used for better performance
        when multiple frames are needed (e.g., video export).

        Returns
        -------
        bool
            True if batched evaluation is supported
        """
        ...

    def evaluate_batch(self, times: list[float]) -> list[GaussianData]:
        """Evaluate at multiple times efficiently.

        Only called if supports_batched_time is True.

        Parameters
        ----------
        times : list[float]
            Times in source-native units

        Returns
        -------
        list[GaussianData]
            Gaussian data for each time
        """
        ...


# ============================================================================
# Composite Model Interface
# ============================================================================


@runtime_checkable
class CompositeSourceProtocol(Protocol):
    """Protocol for sources that support multiple layers/assets."""

    layer_configs: dict[str, dict[str, Any]]

    def get_layer_info(self) -> dict[str, dict[str, Any]]:
        """Get information about all layers."""
        ...

    def set_layer_visibility(self, layer_id: str, visible: bool) -> None:
        """Set the visibility of a specific layer."""
        ...


# ============================================================================
# Data Sink Protocol (Exporters)
# ============================================================================


@dataclass
class DataSinkMetadata:
    """Metadata for a data sink (exporter) type.

    Used by the registry to provide information about available exporters.
    """

    name: str  # Display name (e.g., "PLY Export")
    description: str  # Brief description
    file_extension: str  # Output extension ".ply"
    supports_animation: bool = True  # Can export sequences?
    config_schema: type | None = None  # Config dataclass


class DataSinkProtocol(Protocol):
    """Protocol for data sinks (exporters).

    Implementations provide a unified interface for exporting Gaussian data
    to various formats.
    """

    @classmethod
    def metadata(cls) -> DataSinkMetadata:
        """Return metadata about this sink type."""
        ...

    def export(self, data: GaussianData, path: str, **options: Any) -> None:
        """Export single frame.

        Parameters
        ----------
        data : GaussianData
            Gaussian data to export
        path : str
            Output file path
        **options : Any
            Format-specific export options
        """
        ...

    def export_sequence(
        self,
        frames: Iterator[GaussianData],
        output_dir: str,
        **options: Any,
    ) -> int:
        """Export sequence of frames.

        Parameters
        ----------
        frames : Iterator[GaussianData]
            Iterator of frames to export
        output_dir : str
            Output directory
        **options : Any
            Format-specific export options

        Returns
        -------
        int
            Number of frames successfully exported
        """
        ...


# ============================================================================
# Additional Protocols (Exporter, Composite)
# ============================================================================


@runtime_checkable
class CompositeModelInterface(Protocol):
    """Interface for models that support multiple layers/assets."""

    layer_configs: dict[str, dict[str, Any]]

    def get_layer_info(self) -> dict[str, dict[str, Any]]:
        """Get info about all layers."""
        ...

    def set_layer_visibility(self, layer_id: str, visible: bool) -> None:
        """Set visibility of a specific layer."""
        ...


@runtime_checkable
class ExporterInterface(Protocol):
    """Standard interface for Gaussian data exporters.

    Enables modular export to different file formats (PLY, OBJ, GLTF, etc.)
    without modifying viewer code.
    """

    def export_frame(
        self, gaussian_data: GSData | GSTensor, output_path: Path, **options: Any
    ) -> None:
        """Export single frame of Gaussian data to file.

        Parameters
        ----------
        gaussian_data : GSData | GSTensor
            Gaussian data to export (either NumPy or PyTorch format)
        output_path : Path
            Output file path
        **options : Any
            Format-specific export options
        """
        ...

    def export_sequence(
        self,
        model: ModelInterface,
        output_dir: Path,
        apply_edits_fn: Any = None,
        progress_callback: Any = None,
        **options: Any,
    ) -> int:
        """Export entire sequence from model.

        Parameters
        ----------
        model : ModelInterface
            Model to export from
        output_dir : Path
            Output directory
        apply_edits_fn : callable, optional
            Function to apply edits to gaussian data
        progress_callback : callable, optional
            Progress callback (frame_idx, total_frames)
        **options : Any
            Format-specific export options

        Returns
        -------
        int
            Number of frames exported
        """
        ...


# ============================================================================
# Backward Compatibility Aliases
# ============================================================================

# ModelInterface is used by the viewer code - alias to BaseGaussianSource
ModelInterface = BaseGaussianSource

# DataSourceProtocol was the old name - alias to BaseGaussianSource
DataSourceProtocol = BaseGaussianSource

# DataSourceMetadata was the old name - alias to SourceMetadata
DataSourceMetadata = SourceMetadata


# ============================================================================
# Legacy Aliases (for migration period - will be removed)
# ============================================================================

# These aliases help with migration but should not be used in new code
DataSourceMetadata = SourceMetadata  # Alias for backward compatibility
DataSourceProtocol = BaseGaussianSource  # Alias for backward compatibility


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Lifecycle
    "PluginState",
    "HealthStatus",
    "HealthCheckResult",
    "LifecycleProtocol",
    "HealthCheckable",
    # Edit Pipeline
    "EditManagerProtocol",
    "EditManagerFactory",
    # Data Loader
    "DataLoaderInterface",
    # Gaussian Source (primary)
    "SourceMetadata",
    "BaseGaussianSource",
    # Optional extensions
    "ProfilableSource",
    "RecommendedScaleSource",
    "CompositeSourceProtocol",
    # Continuous time protocols
    "InterpolatableSource",
    "ContinuousTimeSource",
    # Data Sink
    "DataSinkMetadata",
    "DataSinkProtocol",
    # Legacy aliases
    "DataSourceMetadata",
    "DataSourceProtocol",
]
