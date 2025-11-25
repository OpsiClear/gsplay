"""
Domain-level filter configuration dataclasses.

These are pure data structures for filter parameters that can be used
across all layers without UI dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.domain.entities import SceneBounds


@dataclass
class VolumeFilter:
    """Volume filtering parameters for spatial Gaussian selection.

    Note: Basic filtering (opacity, scale) is handled by FilterValues.
    This class handles complex spatial filtering (sphere, cuboid).
    """

    # Filter type: 'sphere', 'cuboid', or 'none'
    filter_type: str = "none"
    processing_mode: str = "all_gpu"

    # Common thresholds
    opacity_threshold: float = 0.0  # min_opacity
    max_opacity: float = 1.0
    min_scale: float = 0.0
    max_scale: float = 100.0
    use_cpu_filtering: bool = False

    # Sphere filter parameters
    sphere_center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    sphere_radius: float = 1.0
    sphere_radius_factor: float = 1.0  # UI multiplier

    # Cuboid filter parameters
    cuboid_center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    cuboid_size: tuple[float, float, float] = (1.0, 1.0, 1.0)
    cuboid_size_factor_x: float = 1.0  # UI multipliers
    cuboid_size_factor_y: float = 1.0
    cuboid_size_factor_z: float = 1.0

    # Scene bounds (for reference)
    scene_bounds: SceneBounds | None = None

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for easy serialization."""
        return asdict(self)

    def is_active(self) -> bool:
        """Check if any spatial filtering is active."""
        return (
            self.filter_type != "none"
            or self.opacity_threshold > 0.0
            or self.max_opacity < 1.0
            or self.min_scale > 0.0
            or self.max_scale < 100.0
        )

    def get_effective_sphere_radius(self) -> float:
        """Get sphere radius accounting for factor."""
        return self.sphere_radius * self.sphere_radius_factor

    def get_effective_cuboid_size(self) -> tuple[float, float, float]:
        """Get cuboid size accounting for factors."""
        return (
            self.cuboid_size[0] * self.cuboid_size_factor_x,
            self.cuboid_size[1] * self.cuboid_size_factor_y,
            self.cuboid_size[2] * self.cuboid_size_factor_z,
        )
