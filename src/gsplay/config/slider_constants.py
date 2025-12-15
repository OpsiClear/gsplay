"""Centralized slider bound constants for UI controls.

This module provides named constants for slider min/max values used throughout
the UI. Centralizing these values ensures consistency and makes updates easier.
"""

from __future__ import annotations


class SliderBounds:
    """Named constants for UI slider min/max values."""

    # === Position Controls ===
    # Translation (scene movement)
    TRANSLATION_MIN = -10.0
    TRANSLATION_MAX = 10.0

    # Filter center position (sphere, box, ellipsoid)
    FILTER_CENTER_MIN = -20.0
    FILTER_CENTER_MAX = 20.0

    # Frustum position (wider range for camera frustum)
    FRUSTUM_POSITION_MIN = -50.0
    FRUSTUM_POSITION_MAX = 50.0

    # Pivot point position
    PIVOT_MIN = -20.0
    PIVOT_MAX = 20.0

    # === Rotation Controls (degrees) ===
    ROTATION_MIN = -180.0
    ROTATION_MAX = 180.0

    # === Scale Controls ===
    # Main uniform scale
    MAIN_SCALE_MIN = 0.1
    MAIN_SCALE_MAX = 5.0

    # Relative per-axis scale multipliers
    RELATIVE_SCALE_MIN = 0.5
    RELATIVE_SCALE_MAX = 2.0

    # === Filter Size/Radius ===
    SPHERE_RADIUS_MIN = 0.1
    SPHERE_RADIUS_MAX = 50.0

    BOX_SIZE_MIN = 0.1
    BOX_SIZE_MAX = 50.0

    ELLIPSOID_RADIUS_MIN = 0.1
    ELLIPSOID_RADIUS_MAX = 50.0

    # === Frustum Parameters ===
    FRUSTUM_FOV_MIN = 1.0
    FRUSTUM_FOV_MAX = 179.0

    FRUSTUM_ASPECT_MIN = 0.1
    FRUSTUM_ASPECT_MAX = 10.0

    FRUSTUM_NEAR_MIN = 0.001
    FRUSTUM_NEAR_MAX = 100.0

    FRUSTUM_FAR_MIN = 0.1
    FRUSTUM_FAR_MAX = 1000.0

    # === Opacity/Alpha ===
    OPACITY_MIN = 0.0
    OPACITY_MAX = 1.0

    # === Scale Filtering ===
    SCALE_FILTER_MIN = 0.0
    SCALE_FILTER_MAX = 100.0

    # === Color Adjustments ===
    # Temperature/Tint (normalized 0-1)
    COLOR_NORMALIZED_MIN = 0.0
    COLOR_NORMALIZED_MAX = 1.0

    # Brightness/Contrast/etc multipliers
    COLOR_MULTIPLIER_MIN = 0.0
    COLOR_MULTIPLIER_MAX = 2.0

    # Hue shift (degrees, same as rotation)
    HUE_SHIFT_MIN = -180.0
    HUE_SHIFT_MAX = 180.0

    # Saturation adjustments
    SATURATION_MIN = -1.0
    SATURATION_MAX = 1.0
