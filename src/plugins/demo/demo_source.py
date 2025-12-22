"""Demo Data Source - Reference Implementation.

Demonstrates how to implement a data source plugin using the new plugin system.
This example generates random Gaussian point clouds for testing.

Usage
-----
Register via pyproject.toml entry_points or manually:

    >>> from src.infrastructure.registry import SourceRegistry
    >>> SourceRegistry.register("demo-random", DemoRandomSource)

Then use in JSON config:

    {
        "module": "demo-random",
        "config": {
            "n_gaussians": 10000,
            "n_frames": 100,
            "seed": 42
        }
    }
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from src.domain.data import FormatInfo, GaussianData
from src.plugins.base import BaseDataSource, source_metadata


logger = logging.getLogger(__name__)


@dataclass
class DemoRandomSourceConfig:
    """Configuration for DemoRandomSource.

    Attributes
    ----------
    n_gaussians : int
        Number of Gaussians per frame.
    n_frames : int
        Total number of frames to generate.
    seed : int | None
        Random seed for reproducibility.
    bounds : tuple[float, float]
        Spatial bounds for Gaussian positions.
    scale_range : tuple[float, float]
        Range for Gaussian scales (linear).
    device : str
        Target device.
    """

    n_gaussians: int = 10000
    n_frames: int = 100
    seed: int | None = 42
    bounds: tuple[float, float] = (-1.0, 1.0)
    scale_range: tuple[float, float] = (0.001, 0.05)
    device: str = "cuda"


@source_metadata(
    name="Demo Random",
    description="Generate random Gaussian point clouds for testing",
    file_extensions=[],
    version="2.0.0",
    config_schema=DemoRandomSourceConfig,
)
class DemoRandomSource(BaseDataSource):
    """Demo data source that generates random Gaussian point clouds.

    This is a reference implementation showing how to create a plugin
    using the new BaseDataSource base class.
    """

    def __init__(self, config: dict) -> None:
        super().__init__(config)

        # Parse configuration
        self._n_gaussians = config.get("n_gaussians", 10000)
        self._n_frames = config.get("n_frames", 100)
        self._seed = config.get("seed", 42)
        self._bounds = config.get("bounds", (-1.0, 1.0))
        self._scale_range = config.get("scale_range", (0.001, 0.05))
        self._device = config.get("device", "cuda")

        # Initialize RNG
        self._rng = np.random.default_rng(self._seed)

        logger.info(
            "DemoRandomSource initialized: %d gaussians x %d frames",
            self._n_gaussians,
            self._n_frames,
        )

    @classmethod
    def can_load(cls, path: str) -> bool:
        """This is a generator, not a file loader."""
        return False

    @property
    def total_frames(self) -> int:
        return self._n_frames

    def get_frame_at_time(self, normalized_time: float) -> GaussianData:
        """Generate random Gaussian data for the given time."""
        frame_idx = self._time_to_index(normalized_time)

        # Use frame-specific seed for reproducibility
        if self._seed is not None:
            rng = np.random.default_rng(self._seed + frame_idx)
        else:
            rng = self._rng

        n = self._n_gaussians
        lo, hi = self._bounds
        scale_lo, scale_hi = self._scale_range

        # Generate Gaussian attributes
        means = rng.uniform(lo, hi, size=(n, 3)).astype(np.float32)

        # Add animation (gentle oscillation)
        time_factor = frame_idx / max(1, self._n_frames - 1)
        means[:, 1] += 0.1 * np.sin(2 * np.pi * time_factor + means[:, 0])

        scales = rng.uniform(scale_lo, scale_hi, size=(n, 3)).astype(np.float32)

        quats = rng.standard_normal((n, 4)).astype(np.float32)
        quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)

        opacities = rng.uniform(0.5, 1.0, size=(n,)).astype(np.float32)

        sh0 = rng.uniform(0.0, 1.0, size=(n, 3)).astype(np.float32)

        return GaussianData(
            means=means,
            scales=scales,
            quats=quats,
            opacities=opacities,
            sh0=sh0,
            shN=None,
            format_info=FormatInfo(
                is_scales_ply=False,
                is_opacities_ply=False,
                is_sh0_rgb=True,
                sh_degree=None,
            ),
            source_path=f"demo://frame_{frame_idx}",
        )


def register_demo_source() -> None:
    """Register the demo source with the registry."""
    from src.infrastructure.registry import SourceRegistry

    SourceRegistry.register("demo-random", DemoRandomSource)
    logger.info("Registered demo-random data source")
