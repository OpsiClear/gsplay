"""
Filter visualization helpers for the viewer.

Renders wireframe gizmos for spatial filters (sphere, box, ellipsoid, frustum).
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import viser
    from gsmod.config.values import FilterValues

logger = logging.getLogger(__name__)


def _axis_angle_to_quaternion(axis_angle: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """Convert axis-angle rotation to quaternion (w, x, y, z)."""
    ax, ay, az = axis_angle
    angle = math.sqrt(ax * ax + ay * ay + az * az)

    if angle < 1e-8:
        return (1.0, 0.0, 0.0, 0.0)

    half_angle = angle / 2.0
    s = math.sin(half_angle) / angle

    return (
        math.cos(half_angle),
        ax * s,
        ay * s,
        az * s,
    )


def _euler_to_quaternion(rx: float, ry: float, rz: float) -> tuple[float, float, float, float]:
    """Convert Euler angles (radians) to quaternion (w, x, y, z)."""
    # ZYX convention
    cx, sx = math.cos(rx / 2), math.sin(rx / 2)
    cy, sy = math.cos(ry / 2), math.sin(ry / 2)
    cz, sz = math.cos(rz / 2), math.sin(rz / 2)

    return (
        cx * cy * cz + sx * sy * sz,  # w
        sx * cy * cz - cx * sy * sz,  # x
        cx * sy * cz + sx * cy * sz,  # y
        cx * cy * sz - sx * sy * cz,  # z
    )


def _create_wireframe_sphere(
    n_longitude: int = 16,
    n_latitude: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Create wireframe sphere vertices and line indices."""
    vertices = []
    lines = []

    # Generate vertices
    for i in range(n_latitude + 1):
        lat = math.pi * i / n_latitude - math.pi / 2
        for j in range(n_longitude):
            lon = 2 * math.pi * j / n_longitude
            x = math.cos(lat) * math.cos(lon)
            y = math.cos(lat) * math.sin(lon)
            z = math.sin(lat)
            vertices.append([x, y, z])

    # Generate longitude lines
    for j in range(n_longitude):
        for i in range(n_latitude):
            idx1 = i * n_longitude + j
            idx2 = (i + 1) * n_longitude + j
            lines.append([idx1, idx2])

    # Generate latitude lines
    for i in range(n_latitude + 1):
        for j in range(n_longitude):
            idx1 = i * n_longitude + j
            idx2 = i * n_longitude + (j + 1) % n_longitude
            lines.append([idx1, idx2])

    return np.array(vertices, dtype=np.float32), np.array(lines, dtype=np.int32)


def _create_wireframe_box() -> tuple[np.ndarray, np.ndarray]:
    """Create wireframe box vertices and line indices (unit cube centered at origin)."""
    vertices = np.array([
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5],
    ], dtype=np.float32)

    lines = np.array([
        # Bottom face
        [0, 1], [1, 2], [2, 3], [3, 0],
        # Top face
        [4, 5], [5, 6], [6, 7], [7, 4],
        # Vertical edges
        [0, 4], [1, 5], [2, 6], [3, 7],
    ], dtype=np.int32)

    return vertices, lines


def _create_wireframe_frustum(
    fov: float = 1.047,
    aspect: float = 1.0,
    near: float = 0.1,
    far: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Create wireframe frustum vertices and line indices."""
    tan_fov = math.tan(fov / 2)

    # Near plane corners
    near_h = near * tan_fov
    near_w = near_h * aspect

    # Far plane corners
    far_h = far * tan_fov
    far_w = far_h * aspect

    # Vertices: camera at origin, looking along -Z
    vertices = np.array([
        # Origin (camera position)
        [0, 0, 0],
        # Near plane corners
        [-near_w, -near_h, -near],
        [near_w, -near_h, -near],
        [near_w, near_h, -near],
        [-near_w, near_h, -near],
        # Far plane corners
        [-far_w, -far_h, -far],
        [far_w, -far_h, -far],
        [far_w, far_h, -far],
        [-far_w, far_h, -far],
    ], dtype=np.float32)

    lines = np.array([
        # Rays from origin to far corners
        [0, 5], [0, 6], [0, 7], [0, 8],
        # Near plane
        [1, 2], [2, 3], [3, 4], [4, 1],
        # Far plane
        [5, 6], [6, 7], [7, 8], [8, 5],
        # Connect near to far
        [1, 5], [2, 6], [3, 7], [4, 8],
    ], dtype=np.int32)

    return vertices, lines


class FilterVisualizer:
    """Manages 3D visualizations for spatial filters."""

    # Colors for different filter types (RGB, 0-255) - as numpy arrays
    SPHERE_COLOR = np.array([66, 135, 245], dtype=np.uint8)    # Blue
    BOX_COLOR = np.array([245, 166, 66], dtype=np.uint8)       # Orange
    ELLIPSOID_COLOR = np.array([168, 66, 245], dtype=np.uint8) # Purple
    FRUSTUM_COLOR = np.array([66, 245, 135], dtype=np.uint8)   # Green

    def __init__(self, server: viser.ViserServer):
        self._server = server
        self._visible = False
        self._current_type: str | None = None

        # Scene handles for each filter type
        self._sphere_handle = None
        self._box_handle = None
        self._ellipsoid_handle = None
        self._frustum_handle = None

        # Precompute wireframe geometries
        self._sphere_verts, self._sphere_lines = _create_wireframe_sphere()
        self._box_verts, self._box_lines = _create_wireframe_box()

    @property
    def visible(self) -> bool:
        return self._visible

    @visible.setter
    def visible(self, value: bool) -> None:
        self._visible = value
        self._update_visibility()

    def _remove_handle(self, attr: str) -> None:
        """Safely remove a viser handle and clear the reference."""
        handle = getattr(self, attr)
        if handle is None:
            return

        try:
            handle.remove()
        except KeyError:
            logger.debug("Handle %s already removed; skipping", attr)

        setattr(self, attr, None)

    def _update_visibility(self) -> None:
        """Update visibility of all handles."""
        if self._sphere_handle is not None:
            self._sphere_handle.visible = self._visible and self._current_type == "Sphere"
        if self._box_handle is not None:
            self._box_handle.visible = self._visible and self._current_type == "Box"
        if self._ellipsoid_handle is not None:
            self._ellipsoid_handle.visible = self._visible and self._current_type == "Ellipsoid"
        if self._frustum_handle is not None:
            self._frustum_handle.visible = self._visible and self._current_type == "Frustum"

    def update(self, filter_type: str, filter_values: FilterValues) -> None:
        """Update visualization based on current filter settings."""
        self._current_type = filter_type

        try:
            if filter_type == "Sphere":
                self._update_sphere(filter_values)
            elif filter_type == "Box":
                self._update_box(filter_values)
            elif filter_type == "Ellipsoid":
                self._update_ellipsoid(filter_values)
            elif filter_type == "Frustum":
                self._update_frustum(filter_values)

            self._update_visibility()
        except Exception as e:
            logger.error(f"Failed to update filter visualization: {e}", exc_info=True)

    def _update_sphere(self, fv: FilterValues) -> None:
        """Update sphere visualization."""
        center = fv.sphere_center
        radius = fv.sphere_radius if fv.sphere_radius < float("inf") else 10.0

        # Scale vertices by radius and translate to center
        verts = self._sphere_verts * radius + np.array(center, dtype=np.float32)

        # Remove old handle if exists
        self._remove_handle("_sphere_handle")

        # Create line segments - use single color (shape (3,))
        self._sphere_handle = self._server.scene.add_line_segments(
            name="/filter_viz/sphere",
            points=verts[self._sphere_lines.flatten()].reshape(-1, 2, 3),
            colors=self.SPHERE_COLOR,
            line_width=2.0,
        )

    def _update_box(self, fv: FilterValues) -> None:
        """Update box visualization."""
        if fv.box_min is None or fv.box_max is None:
            return

        box_min = np.array(fv.box_min, dtype=np.float32)
        box_max = np.array(fv.box_max, dtype=np.float32)
        center = (box_min + box_max) / 2
        size = box_max - box_min

        # Scale and translate vertices
        verts = self._box_verts * size + center

        # Apply rotation if present
        if fv.box_rot is not None:
            quat = _axis_angle_to_quaternion(fv.box_rot)
            # For line segments, we need to rotate around center
            verts_centered = verts - center
            verts = self._rotate_points(verts_centered, quat) + center

        # Remove old handle if exists
        self._remove_handle("_box_handle")

        # Create line segments - use single color (shape (3,))
        self._box_handle = self._server.scene.add_line_segments(
            name="/filter_viz/box",
            points=verts[self._box_lines.flatten()].reshape(-1, 2, 3),
            colors=self.BOX_COLOR,
            line_width=2.0,
        )

    def _update_ellipsoid(self, fv: FilterValues) -> None:
        """Update ellipsoid visualization."""
        if fv.ellipsoid_radii is None:
            return

        center = np.array(fv.ellipsoid_center or (0, 0, 0), dtype=np.float32)
        radii = np.array(fv.ellipsoid_radii, dtype=np.float32)

        # Scale sphere vertices by radii
        verts = self._sphere_verts * radii

        # Apply rotation if present
        if fv.ellipsoid_rot is not None:
            quat = _axis_angle_to_quaternion(fv.ellipsoid_rot)
            verts = self._rotate_points(verts, quat)

        # Translate to center
        verts = verts + center

        # Remove old handle if exists
        self._remove_handle("_ellipsoid_handle")

        # Create line segments - use single color (shape (3,))
        self._ellipsoid_handle = self._server.scene.add_line_segments(
            name="/filter_viz/ellipsoid",
            points=verts[self._sphere_lines.flatten()].reshape(-1, 2, 3),
            colors=self.ELLIPSOID_COLOR,
            line_width=2.0,
        )

    def _update_frustum(self, fv: FilterValues) -> None:
        """Update frustum visualization."""
        if fv.frustum_pos is None:
            return

        # Create frustum geometry
        # Scale far distance for visualization (cap at reasonable size)
        viz_far = min(fv.frustum_far, 20.0)
        verts, lines = _create_wireframe_frustum(
            fov=fv.frustum_fov,
            aspect=fv.frustum_aspect,
            near=fv.frustum_near,
            far=viz_far,
        )

        # Apply rotation if present
        if fv.frustum_rot is not None:
            quat = _axis_angle_to_quaternion(fv.frustum_rot)
            verts = self._rotate_points(verts, quat)

        # Translate to position
        verts = verts + np.array(fv.frustum_pos, dtype=np.float32)

        # Remove old handle if exists
        self._remove_handle("_frustum_handle")

        # Create line segments - use single color (shape (3,))
        self._frustum_handle = self._server.scene.add_line_segments(
            name="/filter_viz/frustum",
            points=verts[lines.flatten()].reshape(-1, 2, 3),
            colors=self.FRUSTUM_COLOR,
            line_width=2.0,
        )

    def _rotate_points(
        self,
        points: np.ndarray,
        quat: tuple[float, float, float, float],
    ) -> np.ndarray:
        """Rotate points by quaternion (w, x, y, z)."""
        w, x, y, z = quat

        # Rotation matrix from quaternion
        r00 = 1 - 2 * (y * y + z * z)
        r01 = 2 * (x * y - w * z)
        r02 = 2 * (x * z + w * y)
        r10 = 2 * (x * y + w * z)
        r11 = 1 - 2 * (x * x + z * z)
        r12 = 2 * (y * z - w * x)
        r20 = 2 * (x * z - w * y)
        r21 = 2 * (y * z + w * x)
        r22 = 1 - 2 * (x * x + y * y)

        R = np.array([
            [r00, r01, r02],
            [r10, r11, r12],
            [r20, r21, r22],
        ], dtype=np.float32)

        return points @ R.T

    def clear(self) -> None:
        """Remove all visualizations."""
        self._remove_handle("_sphere_handle")
        self._remove_handle("_box_handle")
        self._remove_handle("_ellipsoid_handle")
        self._remove_handle("_frustum_handle")
