"""
Demo Data Source (Loader) - Reference Implementation.

This module demonstrates how to implement a DataSource plugin for loading
Gaussian data into the GSPlay. This example generates random
Gaussian point clouds for demonstration purposes.

IMPLEMENTING A DATA SOURCE
==========================

A DataSource (loader) plugin must implement the DataSourceProtocol:

1. REQUIRED CLASS METHODS:
   - metadata() -> DataSourceMetadata
   - can_load(path: str) -> bool

2. REQUIRED INSTANCE METHODS/PROPERTIES:
   - total_frames -> int (property)
   - get_frame(index: int) -> GaussianData
   - get_frame_at_time(normalized_time: float) -> GaussianData

3. BACKWARD COMPATIBILITY METHODS (for viewer integration):
   - get_gaussians_at_normalized_time(normalized_time: float) -> GSData | GSTensor
   - get_total_frames() -> int
   - get_frame_time(frame_idx: int) -> float


DATASOURCEMETADATA FIELDS
=========================

    name: str
        Display name shown in UI (e.g., "PLY Sequence")

    description: str
        Brief description (e.g., "Load PLY files from a folder")

    file_extensions: list[str]
        Supported file extensions [".ply", ".splat", etc.]

    config_schema: type | None
        Optional dataclass defining configuration parameters.
        Used for validation and documentation.

    supports_streaming: bool = True
        Whether frames can be loaded on-demand (vs all upfront).
        Most sources should support streaming.

    supports_seeking: bool = True
        Whether random frame access is supported.
        Set False for network streams that only support sequential access.


GAUSSIANDATA STRUCTURE
======================

GaussianData is the unified data container for Gaussian splatting data.
All sources must produce GaussianData objects.

    Attributes:
        means: np.ndarray [N, 3]      - Gaussian centers (x, y, z)
        scales: np.ndarray [N, 3]     - Gaussian scales (may be log-space)
        quats: np.ndarray [N, 4]      - Quaternion rotations (w, x, y, z)
        opacities: np.ndarray [N]     - Opacity values (may be logit-space)
        sh0: np.ndarray [N, 3]        - DC color coefficients (RGB or SH)
        shN: np.ndarray [N, K, 3]     - Higher-order SH (optional)
        format_info: FormatInfo       - Tracks encoding state

    Format Info:
        is_scales_ply: bool           - True = log-space, False = linear
        is_opacities_ply: bool        - True = logit-space, False = linear
        is_sh0_rgb: bool              - True = RGB [0,1], False = SH coefficients
        sh_degree: int | None         - SH degree (0, 1, 2, 3) or None

    Conversions:
        to_gsdata() -> GSData         - Convert to gsply CPU container
        to_gstensor(device) -> GSTensor - Convert to gsply GPU container
        from_gsdata(gsdata) -> GaussianData (classmethod)
        from_gstensor(gstensor) -> GaussianData (classmethod)


REGISTRATION
============

After implementing your source, register it:

    >>> from src.infrastructure.registry import DataSourceRegistry
    >>> DataSourceRegistry.register("demo-random", DemoRandomSource)

Or add to the default registration in:
    src/infrastructure/registry/__init__.py


USAGE
=====

Once registered, the source can be used via JSON config:

    {
        "module": "demo-random",
        "config": {
            "n_gaussians": 10000,
            "n_frames": 100,
            "seed": 42
        }
    }

Or via ModelFactory:

    >>> from src.infrastructure.model_factory import ModelFactory
    >>> model, loader, metadata = ModelFactory.create(
    ...     "demo-random",
    ...     {"n_gaussians": 10000, "n_frames": 100},
    ...     device="cuda"
    ... )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.domain.data import GaussianData, FormatInfo
from src.domain.interfaces import DataSourceProtocol, DataSourceMetadata

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Schema
# =============================================================================


@dataclass
class DemoRandomSourceConfig:
    """Configuration for DemoRandomSource.

    This dataclass defines all configurable parameters. Using a dataclass
    provides type hints, defaults, and documentation for users.

    Attributes
    ----------
    n_gaussians : int
        Number of Gaussians per frame. Default: 10000
    n_frames : int
        Total number of frames to generate. Default: 100
    seed : int | None
        Random seed for reproducibility. None = non-deterministic.
    bounds : tuple[float, float]
        Spatial bounds for Gaussian positions. Default: (-1.0, 1.0)
    scale_range : tuple[float, float]
        Range for Gaussian scales (linear). Default: (0.001, 0.05)
    device : str
        Target device (passed from viewer). Default: "cuda"
    """

    n_gaussians: int = 10000
    n_frames: int = 100
    seed: int | None = 42
    bounds: tuple[float, float] = (-1.0, 1.0)
    scale_range: tuple[float, float] = (0.001, 0.05)
    device: str = "cuda"


# =============================================================================
# Data Source Implementation
# =============================================================================


class DemoRandomSource(DataSourceProtocol):
    """Demo data source that generates random Gaussian point clouds.

    This is a reference implementation showing the minimum requirements
    for a DataSource plugin. It generates random but consistent data
    for each frame (using frame index as part of the random seed).

    Example
    -------
    >>> # Direct instantiation
    >>> source = DemoRandomSource({
    ...     "n_gaussians": 5000,
    ...     "n_frames": 50,
    ...     "seed": 123
    ... })
    >>> print(f"Total frames: {source.total_frames}")
    Total frames: 50
    >>>
    >>> # Get a frame
    >>> frame = source.get_frame(0)
    >>> print(f"Gaussians: {frame.n_gaussians}")
    Gaussians: 5000

    Notes
    -----
    In a real loader implementation, you would:
    - Parse actual file formats (PLY, SPLAT, NPZ, etc.)
    - Implement caching for efficient playback
    - Handle I/O errors gracefully
    - Support cloud storage if needed (via UniversalPath)
    """

    # =========================================================================
    # CLASS METHODS (Required by Protocol)
    # =========================================================================

    @classmethod
    def metadata(cls) -> DataSourceMetadata:
        """Return metadata about this source type.

        This method is called by the registry to discover available loaders
        and their capabilities. It should return consistent, static metadata.

        Returns
        -------
        DataSourceMetadata
            Metadata describing this source's capabilities.
        """
        return DataSourceMetadata(
            name="Demo Random",
            description="Generate random Gaussian point clouds for testing",
            file_extensions=[],  # No file extensions - pure generator
            config_schema=DemoRandomSourceConfig,
            supports_streaming=True,
            supports_seeking=True,
        )

    @classmethod
    def can_load(cls, path: str) -> bool:
        """Check if this source can handle the given path.

        This method enables auto-detection when a user provides a path
        without specifying the module type. The registry calls this on
        all registered sources to find a compatible loader.

        Parameters
        ----------
        path : str
            File or directory path to check.

        Returns
        -------
        bool
            True if this source can load the path, False otherwise.

        Notes
        -----
        For this demo source, we always return False since it's a generator
        that doesn't load from files. Real loaders should check:
        - File extension matches supported extensions
        - Directory contains expected files
        - File header/magic bytes are valid
        """
        # This is a generator, not a file loader
        return False

    # =========================================================================
    # INSTANCE INITIALIZATION
    # =========================================================================

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the data source.

        Parameters
        ----------
        config : dict[str, Any]
            Configuration dictionary with keys matching DemoRandomSourceConfig.
            Unknown keys are ignored, missing keys use defaults.

        Notes
        -----
        The ModelFactory passes configuration from JSON config files.
        Always use .get() with defaults for optional parameters.
        """
        # Parse configuration into typed dataclass
        self._config = DemoRandomSourceConfig(
            n_gaussians=config.get("n_gaussians", 10000),
            n_frames=config.get("n_frames", 100),
            seed=config.get("seed", 42),
            bounds=config.get("bounds", (-1.0, 1.0)),
            scale_range=config.get("scale_range", (0.001, 0.05)),
            device=config.get("device", "cuda"),
        )

        # Initialize random generator (for reproducibility)
        self._rng = np.random.default_rng(self._config.seed)

        # Cache for generated frames (optional but recommended)
        self._frame_cache: dict[int, GaussianData] = {}

        logger.info(
            "DemoRandomSource initialized: %d gaussians x %d frames (seed=%s)",
            self._config.n_gaussians,
            self._config.n_frames,
            self._config.seed,
        )

    # =========================================================================
    # REQUIRED PROPERTIES
    # =========================================================================

    @property
    def total_frames(self) -> int:
        """Total number of frames available.

        Returns
        -------
        int
            Number of frames that can be loaded (1-indexed).
        """
        return self._config.n_frames

    # =========================================================================
    # REQUIRED METHODS
    # =========================================================================

    def get_frame(self, index: int) -> GaussianData:
        """Get frame at specific index.

        This is the primary method for loading frame data. It should:
        1. Validate the index is in range
        2. Load/generate the frame data
        3. Return as GaussianData

        Parameters
        ----------
        index : int
            Frame index (0-based). Must be in [0, total_frames).

        Returns
        -------
        GaussianData
            Frame data with all Gaussian attributes populated.

        Raises
        ------
        IndexError
            If index is out of range.
        """
        # Validate index
        if index < 0 or index >= self.total_frames:
            raise IndexError(
                f"Frame index {index} out of range [0, {self.total_frames})"
            )

        # Check cache first (optional optimization)
        if index in self._frame_cache:
            logger.debug("Cache hit for frame %d", index)
            return self._frame_cache[index]

        # Generate frame data
        frame_data = self._generate_frame(index)

        # Cache the result (optional)
        self._frame_cache[index] = frame_data

        return frame_data

    def get_frame_at_time(self, normalized_time: float) -> GaussianData:
        """Get frame at normalized time [0, 1].

        This method converts normalized time to frame index. It enables
        smooth playback by mapping continuous time to discrete frames.

        Parameters
        ----------
        normalized_time : float
            Time value in range [0.0, 1.0].
            0.0 = first frame, 1.0 = last frame.

        Returns
        -------
        GaussianData
            Frame data at the specified time.

        Notes
        -----
        The conversion formula handles edge cases:
        - Single frame: always returns frame 0
        - Multiple frames: linear interpolation
        """
        # Handle single-frame case
        if self.total_frames <= 1:
            return self.get_frame(0)

        # Convert to frame index (round to nearest)
        frame_idx = int(round(normalized_time * (self.total_frames - 1)))
        frame_idx = max(0, min(frame_idx, self.total_frames - 1))

        return self.get_frame(frame_idx)

    # =========================================================================
    # BACKWARD COMPATIBILITY (Required for GSPlay Integration)
    # =========================================================================
    # These methods provide compatibility with the existing viewer code
    # that expects the ModelInterface/TimeSampledModel interface.

    def get_gaussians_at_normalized_time(self, normalized_time: float):
        """Compatibility method for viewer code.

        The viewer expects this method to return raw GSData or GSTensor,
        not GaussianData. This wrapper handles the conversion.

        Parameters
        ----------
        normalized_time : float
            Normalized time in [0.0, 1.0].

        Returns
        -------
        GSData | GSTensor
            Raw Gaussian data for the viewer.
        """
        frame_data = self.get_frame_at_time(normalized_time)

        # Convert to appropriate format based on device
        if self._config.device.startswith("cuda"):
            return frame_data.to_gstensor(self._config.device)
        else:
            return frame_data.to_gsdata()

    def get_total_frames(self) -> int:
        """Compatibility method for viewer code."""
        return self.total_frames

    def get_frame_time(self, frame_idx: int) -> float:
        """Compatibility method for viewer code.

        Converts frame index to normalized time.
        """
        if self.total_frames <= 1:
            return 0.0
        return frame_idx / (self.total_frames - 1)

    # =========================================================================
    # INTERNAL METHODS
    # =========================================================================

    def _generate_frame(self, frame_idx: int) -> GaussianData:
        """Generate random Gaussian data for a frame.

        This is where the actual data generation/loading happens.
        In a real loader, this would read from files.

        Parameters
        ----------
        frame_idx : int
            Frame index to generate.

        Returns
        -------
        GaussianData
            Generated frame data.
        """
        # Use frame-specific seed for reproducibility
        if self._config.seed is not None:
            frame_seed = self._config.seed + frame_idx
            rng = np.random.default_rng(frame_seed)
        else:
            rng = self._rng

        n = self._config.n_gaussians
        lo, hi = self._config.bounds
        scale_lo, scale_hi = self._config.scale_range

        # Generate Gaussian attributes
        # -----------------------------------------------------------------
        # means: [N, 3] - Gaussian centers in world coordinates
        means = rng.uniform(lo, hi, size=(n, 3)).astype(np.float32)

        # Add some animation (gentle oscillation based on frame index)
        time_factor = frame_idx / max(1, self.total_frames - 1)
        means[:, 1] += 0.1 * np.sin(2 * np.pi * time_factor + means[:, 0])

        # scales: [N, 3] - Gaussian scales (LINEAR space, not log)
        # Using linear space since format_info.is_scales_ply=False
        scales = rng.uniform(scale_lo, scale_hi, size=(n, 3)).astype(np.float32)

        # quats: [N, 4] - Quaternion rotations (w, x, y, z)
        # Generate random unit quaternions
        quats = rng.standard_normal((n, 4)).astype(np.float32)
        quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)

        # opacities: [N] - Opacity values (LINEAR space, 0-1)
        # Using linear space since format_info.is_opacities_ply=False
        opacities = rng.uniform(0.5, 1.0, size=(n,)).astype(np.float32)

        # sh0: [N, 3] - RGB colors in [0, 1]
        # Using RGB format since format_info.is_sh0_rgb=True
        sh0 = rng.uniform(0.0, 1.0, size=(n, 3)).astype(np.float32)

        # Create GaussianData with format information
        # -----------------------------------------------------------------
        # FormatInfo tells downstream processors what format the data is in
        format_info = FormatInfo(
            is_scales_ply=False,      # Scales are LINEAR (not log-space)
            is_opacities_ply=False,   # Opacities are LINEAR (not logit-space)
            is_sh0_rgb=True,          # Colors are RGB (not SH coefficients)
            sh_degree=None,           # No higher-order SH
        )

        return GaussianData(
            means=means,
            scales=scales,
            quats=quats,
            opacities=opacities,
            sh0=sh0,
            shN=None,  # No higher-order spherical harmonics
            format_info=format_info,
            source_path=f"demo://frame_{frame_idx}",
        )


# =============================================================================
# REGISTRATION HELPER
# =============================================================================


def register_demo_source() -> None:
    """Register the demo source with the registry.

    Call this function to make the demo source available:

    >>> from src.plugins.demo.demo_source import register_demo_source
    >>> register_demo_source()
    >>> # Now "demo-random" is available as a module type
    """
    from src.infrastructure.registry import DataSourceRegistry

    DataSourceRegistry.register("demo-random", DemoRandomSource)
    logger.info("Registered demo-random data source")
