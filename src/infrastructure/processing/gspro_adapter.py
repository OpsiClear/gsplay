"""
Adapter for gsmod CPU processing library.

Provides clean interfaces matching our GSPlayConfig structures for
color adjustments and scene transforms using gsmod's optimized CPU implementations.
"""

import logging

import numpy as np
from gsply import GSData, GSTensor
from gsmod import Color, Transform
from gsmod.config.values import ColorValues, TransformValues
from gsmod.transform.api import quaternion_to_euler

logger = logging.getLogger(__name__)

# Suppress gsmod's verbose logging (set to WARNING by default)
logging.getLogger('gsmod.filter').setLevel(logging.WARNING)
logging.getLogger('gsmod.filter.pipeline').setLevel(logging.WARNING)
logging.getLogger('gsmod.filter.api').setLevel(logging.WARNING)
logging.getLogger('gsmod.transform').setLevel(logging.WARNING)
logging.getLogger('gsmod.transform.pipeline').setLevel(logging.WARNING)
logging.getLogger('gsmod.color').setLevel(logging.WARNING)
logging.getLogger('gsmod.color.pipeline').setLevel(logging.WARNING)

# Pipeline caches to avoid recreating gsmod objects every frame
_COLOR_PIPELINE_CACHE: dict[tuple[float, ...], Color] = {}
_TRANSFORM_PIPELINE_CACHE: dict[tuple[float, ...], Transform] = {}


def _color_pipeline_key(adjustments: ColorValues) -> tuple[float, ...]:
    return (
        adjustments.temperature,
        adjustments.brightness,
        adjustments.contrast,
        adjustments.gamma,
        adjustments.saturation,
        adjustments.shadows,
        adjustments.highlights,
        adjustments.vibrance,
        adjustments.hue_shift,
    )


def _get_color_pipeline(adjustments: ColorValues) -> Color | None:
    """Return a cached gsmod Color pipeline for the provided adjustments."""
    key = _color_pipeline_key(adjustments)
    pipeline = _COLOR_PIPELINE_CACHE.get(key)
    if pipeline is not None:
        return pipeline

    pipeline = Color()
    if adjustments.temperature != 0.0:
        pipeline.temperature(adjustments.temperature)
    if adjustments.brightness != 1.0:
        pipeline.brightness(adjustments.brightness)
    if adjustments.contrast != 1.0:
        pipeline.contrast(adjustments.contrast)
    if adjustments.gamma != 1.0:
        pipeline.gamma(adjustments.gamma)
    if adjustments.saturation != 1.0:
        pipeline.saturation(adjustments.saturation)
    if adjustments.shadows != 0.0:
        pipeline.shadows(adjustments.shadows)
    if adjustments.highlights != 0.0:
        pipeline.highlights(adjustments.highlights)
    if adjustments.vibrance != 1.0:
        pipeline.vibrance(adjustments.vibrance)
    if abs(adjustments.hue_shift) >= 0.5:
        pipeline.hue_shift(adjustments.hue_shift)

    if pipeline.is_identity():
        _COLOR_PIPELINE_CACHE[key] = pipeline
        return pipeline

    pipeline.compile()
    _COLOR_PIPELINE_CACHE[key] = pipeline
    return pipeline


def _transform_pipeline_key(transform: TransformValues) -> tuple[float, ...]:
    translate = getattr(transform, "translate", getattr(transform, "translation", (0.0, 0.0, 0.0)))
    rotate = getattr(transform, "rotate", getattr(transform, "rotation", (0.0, 0.0, 0.0, 1.0)))
    scale_value = (
        float(transform.scale)
        if isinstance(transform.scale, (int, float))
        else float(transform.scale[0])
    )
    return (*translate, scale_value, *rotate)


def _get_transform_pipeline(transform: TransformValues) -> Transform | None:
    """Return a cached gsmod Transform pipeline for the provided transform."""
    key = _transform_pipeline_key(transform)
    pipeline = _TRANSFORM_PIPELINE_CACHE.get(key)
    if pipeline is not None:
        return pipeline

    pipeline = Transform()
    scale_value = (
        float(transform.scale)
        if isinstance(transform.scale, (int, float))
        else float(transform.scale[0])
    )
    if scale_value != 1.0:
        pipeline.scale(scale_value)
    rotate = getattr(transform, "rotate", getattr(transform, "rotation", None))
    if rotate is not None and any(float(x) for x in rotate[1:]):
        euler = quaternion_to_euler(np.asarray(rotate, dtype=np.float32))
        pipeline.rotate_euler(euler)
    translate = getattr(transform, "translate", getattr(transform, "translation", None))
    if translate is not None and any(float(x) for x in translate):
        translation = np.array(translate, dtype=np.float32)
        pipeline.translate(translation)

    if pipeline.is_identity():
        _TRANSFORM_PIPELINE_CACHE[key] = pipeline
        return pipeline

    pipeline.compile()
    _TRANSFORM_PIPELINE_CACHE[key] = pipeline
    return pipeline

def apply_color_adjustments_cpu(
    colors: np.ndarray,
    adjustments: ColorValues
) -> np.ndarray:
    """
    Apply color adjustments on CPU using gsmod Color pipeline.

    Uses gsmod's optimized Numba kernels for all operations (temperature, brightness,
    contrast, gamma, saturation, vibrance, hue_shift, shadows, highlights).

    Args:
        colors: RGB colors [N, 3] (NumPy array)
        adjustments: ColorValues configuration

    Returns:
        Adjusted colors [N, 3] (NumPy array)
    """
    pipeline = _get_color_pipeline(adjustments)
    if pipeline is None or pipeline.is_identity():
        return colors.astype(np.float32, copy=False)

    adjusted = pipeline._apply_to_colors(colors, inplace=False)
    return np.clip(adjusted, 0.0, 1.0)


def apply_scene_transform_cpu(
    means: np.ndarray,
    quats: np.ndarray,
    scales: np.ndarray,
    transform: TransformValues
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply scene transform on CPU using gsmod's transform API.

    Args:
        means: Positions [N, 3] (NumPy array)
        quats: Quaternions [N, 4] (NumPy array, wxyz format)
        scales: Scales [N, 3] (NumPy array)
        transform: TransformValues configuration

    Returns:
        Tuple of (transformed_means, transformed_quats, transformed_scales)
    """
    pipeline = _get_transform_pipeline(transform)
    if pipeline is None or pipeline.is_identity():
        return means, quats, scales

    transformed_means, transformed_quats, transformed_scales = pipeline._apply_to_arrays(
        means,
        quats,
        scales,
        inplace=False,
    )
    return transformed_means, transformed_quats, transformed_scales


# ============================================================================
# Container-Based API (Recommended)
# ============================================================================
# These functions accept and return GSData containers for cleaner API usage

def apply_color_adjustments_gsdata(
    data: GSData | GSTensor,
    adjustments: ColorValues
) -> GSData:
    """
    Apply color adjustments using container-based API (recommended).

    Accepts GSData or GSTensor, applies color adjustments, returns GSData.
    This eliminates the need for manual field extraction in calling code.

    Args:
        data: Gaussian data (GSData or GSTensor)
        adjustments: ColorValues configuration

    Returns:
        GSData with adjusted colors (all fields preserved)

    Example:
        >>> # Clean container-based API
        >>> adjusted_data = apply_color_adjustments_gsdata(gsdata, adjustments)
        >>> # vs old API requiring manual extraction:
        >>> adjusted_colors = apply_color_adjustments_cpu(gsdata.sh0, adjustments)
        >>> gsdata.sh0 = adjusted_colors
    """
    # Convert GSTensor to GSData if needed
    if isinstance(data, GSTensor):
        gsdata = data.to_gsdata()
    else:
        gsdata = data

    # Apply color adjustments to sh0 field
    adjusted_colors = apply_color_adjustments_cpu(gsdata.sh0, adjustments)

    # Update colors in-place (GSData fields are mutable)
    gsdata.sh0 = adjusted_colors

    return gsdata


def apply_scene_transform_gsdata(
    data: GSData | GSTensor,
    transform: TransformValues
) -> GSData:
    """
    Apply scene transform using container-based API (recommended).

    Accepts GSData or GSTensor, applies transform, returns GSData.
    This eliminates placeholder zeros and provides cleaner API.

    Args:
        data: Gaussian data (GSData or GSTensor)
        transform: TransformValues configuration

    Returns:
        GSData with transformed geometry (all fields preserved)

    Example:
        >>> # Clean container-based API
        >>> transformed_data = apply_scene_transform_gsdata(gsdata, transform)
        >>> # vs old API requiring tuple unpacking:
        >>> means, quats, scales = apply_scene_transform_cpu(
        ...     gsdata.means, gsdata.quats, gsdata.scales, transform
        ... )
        >>> gsdata.means, gsdata.quats, gsdata.scales = means, quats, scales
    """
    # Convert GSTensor to GSData if needed
    if isinstance(data, GSTensor):
        gsdata = data.to_gsdata()
    else:
        gsdata = data

    pipeline = _get_transform_pipeline(transform)
    if pipeline is None or pipeline.is_identity():
        return gsdata

    pipeline.apply(gsdata, inplace=True)
    return gsdata


def apply_opacity_scaling_gsdata(
    data: GSData | GSTensor,
    alpha_scaler: float
) -> GSData:
    """
    Apply opacity scaling using container-based API (recommended).

    Uses gsmod's format-aware opacity processing that handles both
    linear [0, 1] and PLY (logit) opacity formats correctly.

    Args:
        data: Gaussian data (GSData or GSTensor)
        alpha_scaler: Opacity scaling factor (1.0 = no change)

    Returns:
        GSData with scaled opacities (all fields preserved)

    Example:
        >>> scaled_data = apply_opacity_scaling_gsdata(gsdata, 1.5)
    """
    if alpha_scaler == 1.0:
        return data if isinstance(data, GSData) else data.to_gsdata()

    if isinstance(data, GSTensor):
        gsdata = data.to_gsdata()
    else:
        gsdata = data

    # Use gsmod's format-aware opacity adjustment
    from gsmod.config.values import OpacityValues
    from gsmod.opacity.apply import apply_opacity_values

    values = OpacityValues(scale=alpha_scaler)
    is_ply = getattr(gsdata, "is_opacities_ply", False)
    gsdata.opacities = apply_opacity_values(gsdata.opacities, values, is_ply_format=is_ply)

    return gsdata
