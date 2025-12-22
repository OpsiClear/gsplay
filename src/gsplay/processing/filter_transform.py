"""
Filter parameter inverse-transformation utilities.

Transforms filter parameters from WORLD space (what user sees in visualization)
to LOCAL space (where filtering operates on original Gaussian positions).

The visualization shows filters in WORLD space (after scene transforms).
Filtering operates on LOCAL/original positions (before transforms).
This module inverse-transforms filter parameters so filtering produces
the selection shown by the visualization.

gsmod transform formula (with center/pivot):
    P_world = R @ ((P_local - center) * scale) + center + translation

Inverse transform:
    P_local = R^T @ (P_world - center - translation) / scale + center
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from gsmod.config.values import FilterValues, TransformValues

logger = logging.getLogger(__name__)


def inverse_transform_filter_values(
    filter_values: FilterValues,
    transform_values: TransformValues | None,
) -> FilterValues:
    """Transform filter parameters from WORLD to LOCAL space.

    The visualization shows filters in WORLD space (after scene transforms).
    Filtering operates on LOCAL positions (before transforms).
    This function inverse-transforms filter parameters so filtering produces
    the selection shown by the visualization.

    Parameters
    ----------
    filter_values : FilterValues
        Filter parameters in WORLD space (from UI)
    transform_values : TransformValues | None
        Scene transformation being applied

    Returns
    -------
    FilterValues
        Filter parameters in LOCAL space for filtering
    """
    from gsmod.config.values import FilterValues as FV

    # Fast path: if no transform, return as-is
    if transform_values is None:
        return filter_values

    # Check if transform is neutral (identity)
    if hasattr(transform_values, "is_neutral") and transform_values.is_neutral():
        return filter_values

    # Use gsmod's to_matrix() for exact consistency with visualization
    M_inv = None
    if hasattr(transform_values, "to_matrix"):
        M = transform_values.to_matrix()  # 4x4 forward transform matrix
        M_inv = np.linalg.inv(M)  # Exact inverse matrix

    # Extract transform components (needed for scale/rotation handling)
    translation = np.array(
        getattr(transform_values, "translation", (0.0, 0.0, 0.0)), dtype=np.float64
    )

    # Handle scale - can be scalar or 3-vector
    scale_raw = getattr(transform_values, "scale", 1.0)
    if hasattr(scale_raw, "__len__"):
        scale = np.array(scale_raw, dtype=np.float64)
    else:
        scale = np.array([scale_raw, scale_raw, scale_raw], dtype=np.float64)

    # Get rotation quaternion (wxyz format)
    rotation = getattr(transform_values, "rotation", (1.0, 0.0, 0.0, 0.0))

    # Get center/pivot point (None if not set)
    center_raw = getattr(transform_values, "center", None)
    if center_raw is not None:
        center = np.array(center_raw, dtype=np.float64)
    else:
        center = np.zeros(3, dtype=np.float64)

    # Safeguard against zero or near-zero scale values
    MIN_SCALE = 1e-6
    scale = np.clip(scale, MIN_SCALE, None)

    # Build rotation matrices
    R = _quaternion_to_matrix(rotation)
    R_inv = R.T  # Inverse of rotation matrix is transpose

    # Build kwargs for new FilterValues, starting with pass-through params
    kwargs = {
        "min_opacity": filter_values.min_opacity,
        "max_opacity": filter_values.max_opacity,
        "min_scale": filter_values.min_scale,
        "max_scale": filter_values.max_scale,
        "invert": filter_values.invert,
    }

    # Track if we need scene rotation for filter shapes without their own rotation
    has_scene_rotation = not _is_identity_quaternion(rotation)

    # === Sphere filter ===
    if filter_values.sphere_radius < float("inf"):
        sphere_center_world = np.array(filter_values.sphere_center, dtype=np.float64)
        sphere_center_local = _inverse_transform_point(
            sphere_center_world, R_inv, translation, scale, center, M_inv
        )

        # Check if scale is uniform
        if _is_uniform_scale(scale):
            # Uniform scale: sphere stays sphere
            sphere_radius_local = filter_values.sphere_radius / scale[0]
            kwargs["sphere_center"] = tuple(sphere_center_local)
            kwargs["sphere_radius"] = sphere_radius_local
        else:
            # Non-uniform scale: sphere becomes ellipsoid
            logger.debug("Non-uniform scale detected: converting sphere filter to ellipsoid")
            kwargs["sphere_center"] = filter_values.sphere_center  # Keep original
            kwargs["sphere_radius"] = float("inf")  # Disable sphere

            # Set up ellipsoid instead
            kwargs["ellipsoid_center"] = tuple(sphere_center_local)
            kwargs["ellipsoid_radii"] = tuple(filter_values.sphere_radius / scale)
            if has_scene_rotation:
                kwargs["ellipsoid_rot"] = _matrix_to_axis_angle(R_inv)
    else:
        kwargs["sphere_center"] = filter_values.sphere_center
        kwargs["sphere_radius"] = filter_values.sphere_radius

    # === Box filter ===
    if filter_values.box_min is not None and filter_values.box_max is not None:
        box_min_world = np.array(filter_values.box_min, dtype=np.float64)
        box_max_world = np.array(filter_values.box_max, dtype=np.float64)
        box_center_world = (box_min_world + box_max_world) / 2.0
        box_half_extents = (box_max_world - box_min_world) / 2.0

        # Transform center
        box_center_local = _inverse_transform_point(
            box_center_world, R_inv, translation, scale, center, M_inv
        )

        # Scale half-extents (per-axis)
        box_half_extents_local = box_half_extents / scale

        kwargs["box_min"] = tuple(box_center_local - box_half_extents_local)
        kwargs["box_max"] = tuple(box_center_local + box_half_extents_local)

        # Compose box rotation with inverse scene rotation
        if filter_values.box_rot is not None:
            kwargs["box_rot"] = _compose_inverse_rotation(filter_values.box_rot, R)
        elif has_scene_rotation:
            # No box rotation, but scene has rotation - apply inverse scene rotation
            kwargs["box_rot"] = _matrix_to_axis_angle(R_inv)
        else:
            kwargs["box_rot"] = None
    else:
        kwargs["box_min"] = filter_values.box_min
        kwargs["box_max"] = filter_values.box_max
        kwargs["box_rot"] = filter_values.box_rot

    # === Ellipsoid filter ===
    # (Only set if not already set by sphere->ellipsoid conversion above)
    if "ellipsoid_radii" not in kwargs:
        if filter_values.ellipsoid_radii is not None:
            ellipsoid_center_world = np.array(
                filter_values.ellipsoid_center or (0.0, 0.0, 0.0), dtype=np.float64
            )
            ellipsoid_center_local = _inverse_transform_point(
                ellipsoid_center_world, R_inv, translation, scale, center, M_inv
            )

            # Scale radii per-axis
            ellipsoid_radii_local = (
                np.array(filter_values.ellipsoid_radii, dtype=np.float64) / scale
            )

            kwargs["ellipsoid_center"] = tuple(ellipsoid_center_local)
            kwargs["ellipsoid_radii"] = tuple(ellipsoid_radii_local)

            # Compose ellipsoid rotation with inverse scene rotation
            if filter_values.ellipsoid_rot is not None:
                kwargs["ellipsoid_rot"] = _compose_inverse_rotation(filter_values.ellipsoid_rot, R)
            elif has_scene_rotation:
                kwargs["ellipsoid_rot"] = _matrix_to_axis_angle(R_inv)
            else:
                kwargs["ellipsoid_rot"] = None
        else:
            kwargs["ellipsoid_center"] = filter_values.ellipsoid_center
            kwargs["ellipsoid_radii"] = filter_values.ellipsoid_radii
            kwargs["ellipsoid_rot"] = filter_values.ellipsoid_rot

    # === Frustum filter ===
    if filter_values.frustum_pos is not None:
        frustum_pos_world = np.array(filter_values.frustum_pos, dtype=np.float64)
        frustum_pos_local = _inverse_transform_point(
            frustum_pos_world, R_inv, translation, scale, center, M_inv
        )

        kwargs["frustum_pos"] = tuple(frustum_pos_local)

        # Scale near/far planes - use geometric mean for non-uniform scale
        scale_factor = float(np.cbrt(np.prod(scale)))  # geometric mean
        kwargs["frustum_near"] = filter_values.frustum_near / scale_factor
        kwargs["frustum_far"] = filter_values.frustum_far / scale_factor
        kwargs["frustum_fov"] = filter_values.frustum_fov  # FOV unchanged
        kwargs["frustum_aspect"] = filter_values.frustum_aspect  # Aspect unchanged

        # Compose frustum rotation with inverse scene rotation
        if filter_values.frustum_rot is not None:
            kwargs["frustum_rot"] = _compose_inverse_rotation(filter_values.frustum_rot, R)
        elif has_scene_rotation:
            kwargs["frustum_rot"] = _matrix_to_axis_angle(R_inv)
        else:
            kwargs["frustum_rot"] = None
    else:
        kwargs["frustum_pos"] = filter_values.frustum_pos
        kwargs["frustum_rot"] = filter_values.frustum_rot
        kwargs["frustum_fov"] = filter_values.frustum_fov
        kwargs["frustum_aspect"] = filter_values.frustum_aspect
        kwargs["frustum_near"] = filter_values.frustum_near
        kwargs["frustum_far"] = filter_values.frustum_far

    return FV(**kwargs)


def _inverse_transform_point(
    point_world: np.ndarray,
    R_inv: np.ndarray,
    translation: np.ndarray,
    scale: np.ndarray,
    center: np.ndarray,
    M_inv: np.ndarray | None = None,
) -> np.ndarray:
    """Transform a point from WORLD to LOCAL space.

    If M_inv (the inverse of gsmod's to_matrix()) is provided, uses it for
    exact consistency with the forward transform. Otherwise falls back to
    manual computation.

    Forward (gsmod formula with center):
        P_world = R @ ((P_local - center) * scale) + center + translation

    Inverse:
        P_local = R_inv @ (P_world - center - translation) / scale + center

    Parameters
    ----------
    point_world : np.ndarray
        Point in world space
    R_inv : np.ndarray
        Inverse rotation matrix (R.T)
    translation : np.ndarray
        Translation vector
    scale : np.ndarray
        Per-axis scale vector
    center : np.ndarray
        Center/pivot point (zeros if not set)
    M_inv : np.ndarray | None
        4x4 inverse transformation matrix from gsmod. If provided, used for
        exact consistency.

    Returns
    -------
    np.ndarray
        Point in local space
    """
    # Use exact matrix inverse if available
    if M_inv is not None:
        point_h = np.array([point_world[0], point_world[1], point_world[2], 1.0])
        result_h = M_inv @ point_h
        return result_h[:3]

    # Fallback: manual inverse transform
    return R_inv @ (point_world - center - translation) / scale + center


def _quaternion_to_matrix(q_wxyz: tuple) -> np.ndarray:
    """Convert quaternion (w, x, y, z) to 3x3 rotation matrix."""
    w, x, y, z = q_wxyz
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _matrix_to_axis_angle(R: np.ndarray) -> tuple[float, float, float]:
    """Convert rotation matrix to axis-angle representation (axis * angle)."""
    trace = float(R[0, 0] + R[1, 1] + R[2, 2])
    angle = math.acos(max(-1.0, min(1.0, (trace - 1.0) / 2.0)))

    if angle < 1e-6:
        return (0.0, 0.0, 0.0)

    denom = 2.0 * math.sin(angle)
    if abs(denom) < 1e-8:
        # Near 180 degrees - use alternative extraction
        # Find largest diagonal element
        if R[0, 0] >= R[1, 1] and R[0, 0] >= R[2, 2]:
            ax = math.sqrt(max(0, (R[0, 0] + 1) / 2))
            ay = R[0, 1] / (2 * ax) if ax > 1e-6 else 0
            az = R[0, 2] / (2 * ax) if ax > 1e-6 else 0
        elif R[1, 1] >= R[2, 2]:
            ay = math.sqrt(max(0, (R[1, 1] + 1) / 2))
            ax = R[0, 1] / (2 * ay) if ay > 1e-6 else 0
            az = R[1, 2] / (2 * ay) if ay > 1e-6 else 0
        else:
            az = math.sqrt(max(0, (R[2, 2] + 1) / 2))
            ax = R[0, 2] / (2 * az) if az > 1e-6 else 0
            ay = R[1, 2] / (2 * az) if az > 1e-6 else 0
    else:
        ax = (R[2, 1] - R[1, 2]) / denom
        ay = (R[0, 2] - R[2, 0]) / denom
        az = (R[1, 0] - R[0, 1]) / denom

    return (ax * angle, ay * angle, az * angle)


def _axis_angle_to_matrix(axis_angle: tuple) -> np.ndarray:
    """Convert axis-angle (axis * angle) to rotation matrix."""
    ax, ay, az = axis_angle
    angle = math.sqrt(ax * ax + ay * ay + az * az)

    if angle < 1e-8:
        return np.eye(3, dtype=np.float64)

    # Normalize axis
    ax, ay, az = ax / angle, ay / angle, az / angle

    c = math.cos(angle)
    s = math.sin(angle)
    t = 1 - c

    return np.array(
        [
            [t * ax * ax + c, t * ax * ay - s * az, t * ax * az + s * ay],
            [t * ax * ay + s * az, t * ay * ay + c, t * ay * az - s * ax],
            [t * ax * az - s * ay, t * ay * az + s * ax, t * az * az + c],
        ],
        dtype=np.float64,
    )


def _compose_inverse_rotation(
    filter_rot_axis_angle: tuple,
    R_scene: np.ndarray,
) -> tuple[float, float, float]:
    """Compose filter rotation with inverse scene rotation.

    The filter rotation R_filter is defined in WORLD space.
    To get the equivalent rotation in LOCAL space, we need to conjugate
    by the scene rotation:

        R_local = R_scene^T @ R_filter @ R_scene

    This transforms the rotation axis from world coordinates to local
    coordinates, which is the correct way to express a world-space
    rotation in the local frame.

    Parameters
    ----------
    filter_rot_axis_angle : tuple
        Filter rotation in axis-angle format (ax, ay, az) in WORLD space
    R_scene : np.ndarray
        Scene rotation matrix (NOT the inverse)

    Returns
    -------
    tuple
        Filter rotation in LOCAL space as axis-angle
    """
    R_filter = _axis_angle_to_matrix(filter_rot_axis_angle)
    # Conjugate: R_local = R_scene^T @ R_filter @ R_scene
    R_scene_inv = R_scene.T
    R_composed = R_scene_inv @ R_filter @ R_scene
    return _matrix_to_axis_angle(R_composed)


def _is_identity_quaternion(q: tuple, tolerance: float = 1e-6) -> bool:
    """Check if quaternion is identity (no rotation)."""
    w, x, y, z = q
    return (
        abs(w - 1.0) < tolerance
        and abs(x) < tolerance
        and abs(y) < tolerance
        and abs(z) < tolerance
    )


def _is_uniform_scale(scale: np.ndarray, tolerance: float = 1e-6) -> bool:
    """Check if scale is uniform (same on all axes)."""
    return np.allclose(scale, scale[0], atol=tolerance)
