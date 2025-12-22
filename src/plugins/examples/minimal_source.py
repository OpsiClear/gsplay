"""Minimal example data source plugin.

This demonstrates the simplest possible implementation of a data source,
showing only the required methods.
"""

from __future__ import annotations

import numpy as np

from src.domain.data import GaussianData
from src.plugins.base import BaseDataSource, source_metadata


@source_metadata(
    name="Minimal Example",
    description="Example source generating random Gaussians",
    file_extensions=[".minimal"],
    version="1.0.0",
)
class MinimalSource(BaseDataSource):
    """Minimal example data source.

    This source generates random Gaussian data - useful as a template
    for implementing your own data source.
    """

    def __init__(self, config: dict) -> None:
        super().__init__(config)

        # Example: parse config options
        self._n_frames = config.get("n_frames", 10)
        self._n_gaussians = config.get("n_gaussians", 100)
        self._seed = config.get("seed", 42)

    @classmethod
    def can_load(cls, path: str) -> bool:
        """Check if we can load from this path."""
        return str(path).lower().endswith(".minimal")

    @property
    def total_frames(self) -> int:
        """Return total number of frames."""
        return self._n_frames

    def get_frame_at_time(self, normalized_time: float) -> GaussianData:
        """Generate random Gaussian data for the given time."""
        np.random.seed(self._seed + self._time_to_index(normalized_time))

        return GaussianData(
            means=np.random.rand(self._n_gaussians, 3).astype(np.float32),
            scales=np.random.rand(self._n_gaussians, 3).astype(np.float32) * 0.1,
            quats=self._random_quaternions(self._n_gaussians),
            opacities=np.random.rand(self._n_gaussians).astype(np.float32),
            sh0=np.random.rand(self._n_gaussians, 3).astype(np.float32),
        )

    def _random_quaternions(self, n: int) -> np.ndarray:
        """Generate random unit quaternions."""
        quats = np.random.randn(n, 4).astype(np.float32)
        return quats / np.linalg.norm(quats, axis=1, keepdims=True)


# Example of how to register this plugin:
# In pyproject.toml:
#   [project.entry-points."gsplay.plugins"]
#   minimal = "src.plugins.examples.minimal_source:MinimalSource"
