"""Domain interfaces (protocols) for dependency inversion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Any, Iterator, TYPE_CHECKING, Callable
from pathlib import Path
import numpy as np
from src.domain.entities import GSData, GSTensor

if TYPE_CHECKING:
    from src.domain.data import GaussianData


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


class TimeSampledModel(Protocol):
    """
    Interface for models that expose time-normalized playback semantics.

    Implementations must support normalized-time sampling, frame counts, and
    mapping between frame indices and normalized times so downstream systems
    can drive playback without assuming implementation details.
    """

    def get_gaussians_at_normalized_time(
        self,
        normalized_time: float,
    ) -> GSData | GSTensor | None: ...

    def get_total_frames(self) -> int: ...

    def get_frame_time(self, frame_idx: int) -> float: ...


class ProfilableModel(Protocol):
    """
    Optional interface for exposing the latest performance profile.
    """

    def get_last_profile(self) -> dict[str, Any] | None: ...


class ModelInterface(TimeSampledModel, Protocol):
    """
    Backwards-compatible alias for the common viewer-facing capabilities.
    """

    ...


class ConfigurableModelInterface(ModelInterface, Protocol):
    """
    Extended interface for models that can be created from configuration.

    This standardizes model creation across all model types.
    """
    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        device: str = "cuda"
    ) -> "ConfigurableModelInterface":
        """
        Create model instance from configuration dictionary.

        Parameters
        ----------
        config : dict[str, Any]
            Model-specific configuration parameters
        device : str
            Device to use for computation

        Returns
        -------
        ConfigurableModelInterface
            Configured model instance
        """
        ...


class DataLoaderInterface(Protocol):
    """
    Standard interface for data loaders.
    """
    def get_camera_data(self) -> dict[str, Any] | None: ...
    def get_points_for_initialization(self) -> np.ndarray: ...


class CompositeModelInterface(Protocol):
    """
    Interface for models that support multiple layers/assets.
    """
    layer_configs: dict[str, dict[str, Any]]

    def get_layer_info(self) -> dict[str, dict[str, Any]]: ...
    def set_layer_visibility(self, layer_id: str, visible: bool) -> None: ...


class ExporterInterface(Protocol):
    """
    Standard interface for Gaussian data exporters.

    Enables modular export to different file formats (PLY, OBJ, GLTF, etc.)
    without modifying viewer code.
    """

    def export_frame(
        self,
        gaussian_data: GSData | GSTensor,
        output_path: Path,
        **options: Any
    ) -> None:
        """
        Export single frame of Gaussian data to file.

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
        **options: Any
    ) -> int:
        """
        Export entire sequence from model.

        Parameters
        ----------
        model : ModelInterface
            Model to export from
        output_dir : Path
            Output directory
        apply_edits_fn : callable | None
            Optional function to apply edits: fn(gaussian_data) -> gaussian_data
        progress_callback : callable | None
            Optional progress callback: fn(frame_idx, total_frames) -> None
        **options : Any
            Format-specific export options

        Returns
        -------
        int
            Number of frames successfully exported
        """
        ...

    def get_file_extension(self) -> str:
        """
        Get file extension for this format.

        Returns
        -------
        str
            File extension including dot (e.g., '.ply', '.obj', '.bin')
        """
        ...


# ============================================================================
# Data Source / Sink Protocols (New Registry-Based Architecture)
# ============================================================================


@dataclass
class DataSourceMetadata:
    """Metadata for a data source type.

    Used by the registry to provide information about available loaders.
    """

    name: str  # Display name (e.g., "PLY Sequence")
    description: str  # Brief description
    file_extensions: list[str]  # Supported extensions [".ply", ".splat"]
    config_schema: type | None = None  # Config dataclass for validation
    supports_streaming: bool = True  # Can stream frames?
    supports_seeking: bool = True  # Can seek to arbitrary frame?


class DataSourceProtocol(Protocol):
    """Protocol for data sources (loaders).

    Implementations provide a unified interface for loading Gaussian data
    from various formats (PLY sequences, SPLAT files, etc.).

    The key output is GaussianData - a unified abstraction that can be
    converted to GSData/GSTensor for processing.
    """

    @classmethod
    def metadata(cls) -> DataSourceMetadata:
        """Return metadata about this source type."""
        ...

    @classmethod
    def can_load(cls, path: str) -> bool:
        """Check if this loader can handle the given path."""
        ...

    @property
    def total_frames(self) -> int:
        """Total number of frames available."""
        ...

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
        ...

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
        ...


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

    The key input is GaussianData - a unified abstraction that loaders produce.
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
