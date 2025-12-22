"""
Centralized constants for Gaussian Splatting operations.

This module provides a single source of truth for all magic numbers and constants
used throughout the GSPlay codebase, grouped by their usage context.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SphericalHarmonics:
    """Constants related to Spherical Harmonics color representation."""

    C0: float = 0.28209479177387814
    """SH normalization constant: sqrt(1/(4*pi))"""


@dataclass(frozen=True)
class FormatDetection:
    """Thresholds for auto-detecting data formats (log vs linear space)."""

    LOG_SCALE_THRESHOLD: float = -5.0
    """If minimum scale value is below this, assume log-space format"""


@dataclass(frozen=True)
class NumericalStability:
    """Small values for numerical stability and clamping."""

    EPS: float = 1e-8
    """Epsilon for general numerical stability (division, log, logit)"""

    MIN_SCALE: float = 1e-6
    """Minimum scale value after exp activation"""

    MAX_SCALE: float = 1e3
    """Maximum scale value after exp activation"""

    MIN_OPACITY: float = 1e-8
    """Minimum opacity value for logit transform"""

    MAX_OPACITY: float = 1 - 1e-8
    """Maximum opacity value for logit transform (must be < 1)"""

    MIN_GAMMA_INPUT: float = 1e-6
    """Minimum value for gamma correction input"""


@dataclass(frozen=True)
class Filtering:
    """Constants for scale filtering and outlier detection."""

    DEFAULT_PERCENTILE: float = 0.995
    """Default percentile (99.5%) for scale threshold recommendation"""

    @classmethod
    def get_percentile_label(cls, percentile: float | None = None) -> str:
        """Get human-readable label for percentile value."""
        p = percentile or cls.DEFAULT_PERCENTILE
        return f"{p * 100:.1f}th percentile"


@dataclass(frozen=True)
class RenderingDefaults:
    """Default values for rendering parameters."""

    DEFAULT_DEVICE: str = "cuda:0"
    """Default GPU device for tensor operations"""

    DEFAULT_FPS: int = 30
    """Default playback framerate"""

    BUFFER_SIZE_MULTIPLIER: int = 3
    """Buffer size as multiple of FPS (e.g., 3 * 30fps = 90 frames)"""


@dataclass(frozen=True)
class LogSpace:
    """Constants for log-space clamping before storage."""

    MIN_LOG_SCALE: float = -20.0
    """Minimum log scale value for stable storage"""

    MAX_LOG_SCALE: float = 20.0
    """Maximum log scale value for stable storage"""


class GaussianConstants:
    """
    Central registry of all Gaussian Splatting constants.

    Usage:
        from src.infrastructure.processing.gaussian_constants import GaussianConstants as GC

        # Use SH constant
        rgb = sh * GC.SH.C0 + 0.5

        # Use format detection
        if min_scale < GC.Format.LOG_SCALE_THRESHOLD:
            # Data is in log space

        # Use numerical stability
        scales = torch.exp(log_scales).clamp(GC.Numerical.MIN_SCALE, GC.Numerical.MAX_SCALE)

        # Use filtering
        percentile = torch.quantile(scales, GC.Filtering.DEFAULT_PERCENTILE)
    """

    SH = SphericalHarmonics()
    Format = FormatDetection()
    Numerical = NumericalStability()
    Filtering = Filtering()
    Rendering = RenderingDefaults()
    LogSpace = LogSpace()

    # Aliases for backwards compatibility
    SH_C0 = SH.C0
    LOG_SCALE_THRESHOLD = Format.LOG_SCALE_THRESHOLD
    EPS = Numerical.EPS


# Export all submodules for convenience
__all__ = [
    "Filtering",
    "FormatDetection",
    "GaussianConstants",
    "LogSpace",
    "NumericalStability",
    "RenderingDefaults",
    "SphericalHarmonics",
]
