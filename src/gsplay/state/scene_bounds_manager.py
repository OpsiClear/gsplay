"""
Scene bounds calculation and management.

This module extracts scene bounds logic from UniversalGSPlay,
following Single Responsibility Principle.
"""

from __future__ import annotations

import logging

import numpy as np

from src.domain.interfaces import ModelInterface
from src.domain.services import TransformService

logger = logging.getLogger(__name__)


class SceneBoundsManager:
    """
    Manages scene bounds calculation and storage.

    This class is responsible for:
    - Calculating scene bounds from Gaussian data
    - Storing and providing access to bounds information
    - Recalculating bounds on demand
    """

    def __init__(self):
        """Initialize the scene bounds manager."""
        self._initial_scene_bounds: dict[str, object] | None = None

    def calculate_bounds(self, model: ModelInterface) -> None:
        """
        Calculate scene bounds from first frame of model.

        Parameters
        ----------
        model : ModelInterface
            Model to calculate bounds from
        """
        if not model:
            logger.warning("No model provided for bounds calculation")
            return

        try:
            logger.debug("Calculating scene bounds...")

            # Get first frame
            gaussians = model.get_gaussians_at_normalized_time(normalized_time=0.0)

            if gaussians is None or gaussians.means.shape[0] == 0:
                logger.warning("No gaussians available for bounds calculation")
                return

            # Calculate bounds
            points = gaussians.means.detach().cpu().numpy()

            bounds = TransformService.calculate_scene_bounds(
                points, percentile_min=5.0, percentile_max=95.0, padding=0.1
            )

            center, radius = TransformService.calculate_bounding_sphere(
                points, center=np.array(bounds.center)
            )

            # Store bounds
            self._initial_scene_bounds = {
                "center": bounds.center,
                "min_coords": bounds.min_coords,
                "max_coords": bounds.max_coords,
                "size": bounds.size,
                "sizes": bounds.size,  # Legacy compatibility
                "max_size": float(np.max(bounds.size)),  # Legacy compatibility
                "sphere_radius": radius,
            }

            logger.info(
                f"Scene bounds: center={bounds.center}, "
                f"size={bounds.size}, radius={radius:.3f}"
            )

        except Exception as e:
            logger.error(f"Failed to calculate scene bounds: {e}", exc_info=True)

    def get_bounds(self) -> dict[str, object] | None:
        """
        Get current scene bounds.

        Returns
        -------
        dict[str, object] | None
            Scene bounds dictionary or None if not calculated
        """
        return self._initial_scene_bounds

    def has_bounds(self) -> bool:
        """
        Check if bounds have been calculated.

        Returns
        -------
        bool
            True if bounds exist
        """
        return self._initial_scene_bounds is not None

    def get_center(self) -> tuple[float, float, float] | None:
        """Get scene center coordinates."""
        if not self._initial_scene_bounds:
            return None
        return self._initial_scene_bounds["center"]

    def get_radius(self) -> float | None:
        """Get scene bounding sphere radius."""
        if not self._initial_scene_bounds:
            return None
        return self._initial_scene_bounds["sphere_radius"]

    def get_size(self) -> tuple[float, float, float] | None:
        """Get scene bounding box size."""
        if not self._initial_scene_bounds:
            return None
        return self._initial_scene_bounds["size"]

