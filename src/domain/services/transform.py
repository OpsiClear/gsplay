"""
Scene bounds calculation services.

Provides pure geometric operations for scene bounding box calculation.
Actual Gaussian transforms are delegated to gsmod's GSTensorPro/GSDataPro.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.domain.entities import SceneBounds

logger = logging.getLogger(__name__)


class TransformService:
    """
    Service for scene geometry calculations.

    NOTE: Gaussian transforms (rotation, translation, scale) are handled by
    gsmod's GSTensorPro.transform() and GSDataPro.transform() methods.
    This class only provides bounds calculation utilities.
    """

    @staticmethod
    def calculate_scene_bounds(
        points: np.ndarray,
        percentile_min: float = 5.0,
        percentile_max: float = 95.0,
        padding: float = 0.1,
    ) -> "SceneBounds":
        """
        Calculate scene bounding box from point cloud.

        Uses percentiles to ignore outliers and focus on main scene content.

        :param points: Point cloud [N, 3]
        :param percentile_min: Lower percentile for bounds (default 5.0)
        :param percentile_max: Upper percentile for bounds (default 95.0)
        :param padding: Padding factor to add to bounds (default 0.1 = 10%)
        :return: Calculated scene bounds
        """
        from src.domain.entities import SceneBounds

        try:
            # Use percentile-based bounds to avoid outliers
            min_coords = np.percentile(points, percentile_min, axis=0)
            max_coords = np.percentile(points, percentile_max, axis=0)

            # Calculate center and sizes
            center = (min_coords + max_coords) / 2
            sizes = max_coords - min_coords

            # Add padding
            min_coords = min_coords - sizes * padding
            max_coords = max_coords + sizes * padding
            sizes = max_coords - min_coords
            center = (min_coords + max_coords) / 2

            logger.debug(
                "Scene bounds: X [%.3f, %.3f], Y [%.3f, %.3f], Z [%.3f, %.3f]",
                min_coords[0], max_coords[0],
                min_coords[1], max_coords[1],
                min_coords[2], max_coords[2],
            )

            return SceneBounds(
                min_coords=tuple(min_coords),
                max_coords=tuple(max_coords),
                center=tuple(center),
                size=tuple(sizes),
            )

        except Exception as exc:
            logger.error("Scene bounds calculation failed: %s", exc, exc_info=True)
            return SceneBounds()

    @staticmethod
    def calculate_bounding_sphere(
        points: np.ndarray, center: np.ndarray | None = None, percentile: float = 95.0
    ) -> tuple[np.ndarray, float]:
        """
        Calculate bounding sphere from point cloud.

        :param points: Point cloud [N, 3]
        :param center: Sphere center (default: centroid of points)
        :param percentile: Percentile of distances to use as radius (default 95.0)
        :return: Sphere center [3] and radius
        """
        if center is None:
            center = np.mean(points, axis=0)

        # Calculate distances from center
        distances = np.linalg.norm(points - center, axis=1)

        # Use percentile to ignore outliers
        radius = np.percentile(distances, percentile) * 1.1  # Add 10% padding

        logger.debug(
            "Bounding sphere: center=%s, radius=%.3f (%sth percentile)",
            center, radius, percentile,
        )

        return center, radius
