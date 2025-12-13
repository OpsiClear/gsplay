"""
Filter visualization helpers for the viewer.

Renders wireframe gizmos for spatial filters (sphere, box, ellipsoid, frustum).
Applies scene transformation so visualizations match transformed Gaussian positions.
Supports interactive manipulation via viser transform controls.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    import viser
    from gsmod.config.values import FilterValues, TransformValues

logger = logging.getLogger(__name__)

# Type alias for gizmo update callback
# Callback receives: (filter_type, center_xyz, rotation_axis_angle_or_none)
GizmoUpdateCallback = Callable[[str, tuple[float, float, float], tuple[float, float, float] | None], None]


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


def _quaternion_to_axis_angle(quat: tuple[float, float, float, float]) -> tuple[float, float, float]:
    """Convert quaternion (w, x, y, z) to axis-angle rotation."""
    w, x, y, z = quat

    # Normalize
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-8:
        return (0.0, 0.0, 0.0)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm

    # Ensure w is positive for consistent angle
    if w < 0:
        w, x, y, z = -w, -x, -y, -z

    # Compute angle
    angle = 2.0 * math.acos(max(-1.0, min(1.0, w)))

    if angle < 1e-8:
        return (0.0, 0.0, 0.0)

    # Compute axis
    s = math.sin(angle / 2.0)
    if abs(s) < 1e-8:
        return (0.0, 0.0, 0.0)

    return (x / s * angle, y / s * angle, z / s * angle)


def _create_wireframe_sphere(
    n_longitude: int = 24,
    n_latitude: int = 12,
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


def _create_axis_lines(size: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Create axis indicator lines (X=red, Y=green, Z=blue)."""
    vertices = np.array([
        [0, 0, 0], [size, 0, 0],  # X axis
        [0, 0, 0], [0, size, 0],  # Y axis
        [0, 0, 0], [0, 0, size],  # Z axis
    ], dtype=np.float32)
    lines = np.array([[0, 1], [2, 3], [4, 5]], dtype=np.int32)
    return vertices, lines


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
    """Manages 3D visualizations for spatial filters.

    Visualizations are transformed by the scene transformation so they
    appear in the correct position relative to the transformed Gaussians.
    """

    # Colors for different filter types (RGB, 0-255) - vibrant, modern palette
    SPHERE_COLOR = np.array([59, 130, 246], dtype=np.uint8)    # Bright blue
    BOX_COLOR = np.array([249, 115, 22], dtype=np.uint8)       # Vibrant orange
    ELLIPSOID_COLOR = np.array([168, 85, 247], dtype=np.uint8) # Vivid purple
    FRUSTUM_COLOR = np.array([34, 197, 94], dtype=np.uint8)    # Emerald green

    # Axis colors (RGB)
    AXIS_X_COLOR = np.array([239, 68, 68], dtype=np.uint8)     # Red
    AXIS_Y_COLOR = np.array([34, 197, 94], dtype=np.uint8)     # Green
    AXIS_Z_COLOR = np.array([59, 130, 246], dtype=np.uint8)    # Blue

    # Line widths
    MAIN_LINE_WIDTH = 3.0
    AXIS_LINE_WIDTH = 4.0

    def __init__(self, server: viser.ViserServer):
        self._server = server
        self._visible = False
        self._current_type: str | None = None
        self._transform_values: TransformValues | None = None

        # Scene handles for each filter type
        self._sphere_handle = None
        self._box_handle = None
        self._ellipsoid_handle = None
        self._frustum_handle = None

        # Axis handles for orientation
        self._axis_x_handle = None
        self._axis_y_handle = None
        self._axis_z_handle = None

        # Transform control (gizmo) for interactive manipulation
        self._gizmo_handle = None
        self._gizmo_enabled = False
        self._gizmo_callback: GizmoUpdateCallback | None = None
        self._last_filter_values: FilterValues | None = None
        self._updating_from_gizmo = False  # Prevent feedback loops

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

        # Update axis visibility (hide when gizmo is enabled - gizmo has its own axes)
        show_axes = self._visible and self._current_type in ("Box", "Ellipsoid", "Frustum") and not self._gizmo_enabled
        for handle in [self._axis_x_handle, self._axis_y_handle, self._axis_z_handle]:
            if handle is not None:
                handle.visible = show_axes

        # Update gizmo visibility
        if self._gizmo_handle is not None:
            self._gizmo_handle.visible = self._visible and self._gizmo_enabled and self._current_type in ("Sphere", "Box", "Ellipsoid", "Frustum")

    def _remove_axes(self) -> None:
        """Remove axis handles."""
        self._remove_handle("_axis_x_handle")
        self._remove_handle("_axis_y_handle")
        self._remove_handle("_axis_z_handle")

    def _add_axes(self, center: np.ndarray, size: float, rotation: tuple | None = None) -> None:
        """Add axis indicator lines at the given center.

        Parameters
        ----------
        center : np.ndarray
            Center position for the axes
        size : float
            Length of axis lines
        rotation : tuple | None
            Optional rotation as quaternion (w, x, y, z)
        """
        self._remove_axes()

        axis_verts, axis_lines = _create_axis_lines(size * 0.5)

        # Apply rotation if present
        if rotation is not None:
            axis_verts = self._rotate_points(axis_verts, rotation)

        # Translate to center
        axis_verts = axis_verts + center

        # Apply scene transform
        axis_verts = self._apply_scene_transform(axis_verts)

        # Create X axis (red)
        self._axis_x_handle = self._server.scene.add_line_segments(
            name="/filter_viz/axis_x",
            points=axis_verts[axis_lines[0].flatten()].reshape(-1, 2, 3),
            colors=self.AXIS_X_COLOR,
            line_width=self.AXIS_LINE_WIDTH,
        )

        # Create Y axis (green)
        self._axis_y_handle = self._server.scene.add_line_segments(
            name="/filter_viz/axis_y",
            points=axis_verts[axis_lines[1].flatten()].reshape(-1, 2, 3),
            colors=self.AXIS_Y_COLOR,
            line_width=self.AXIS_LINE_WIDTH,
        )

        # Create Z axis (blue)
        self._axis_z_handle = self._server.scene.add_line_segments(
            name="/filter_viz/axis_z",
            points=axis_verts[axis_lines[2].flatten()].reshape(-1, 2, 3),
            colors=self.AXIS_Z_COLOR,
            line_width=self.AXIS_LINE_WIDTH,
        )

    # -------------------------------------------------------------------------
    # Gizmo (Transform Controls) Management
    # -------------------------------------------------------------------------

    def set_gizmo_enabled(self, enabled: bool) -> None:
        """Enable or disable the interactive gizmo.

        Parameters
        ----------
        enabled : bool
            True to show transform controls for interactive manipulation
        """
        self._gizmo_enabled = enabled
        self._update_visibility()

        if not enabled:
            self._remove_gizmo()

    @property
    def gizmo_enabled(self) -> bool:
        """Whether the interactive gizmo is enabled."""
        return self._gizmo_enabled

    def set_gizmo_callback(self, callback: GizmoUpdateCallback | None) -> None:
        """Register a callback for when the gizmo is manipulated.

        Parameters
        ----------
        callback : GizmoUpdateCallback | None
            Function called with (filter_type, center_xyz, rotation_axis_angle_or_none)
            when user drags the gizmo. Set to None to remove callback.
        """
        self._gizmo_callback = callback

    def _remove_gizmo(self) -> None:
        """Remove the transform control gizmo."""
        if self._gizmo_handle is not None:
            try:
                self._gizmo_handle.remove()
            except Exception:
                logger.debug("Gizmo handle already removed")
            self._gizmo_handle = None

    def _update_gizmo(
        self,
        center: np.ndarray,
        rotation_quat: tuple[float, float, float, float] | None,
        gizmo_scale: float,
        disable_rotations: bool = False,
    ) -> None:
        """Create or update the transform control gizmo.

        Parameters
        ----------
        center : np.ndarray
            Filter center position in filter space (will be transformed to display space)
        rotation_quat : tuple | None
            Filter rotation as quaternion (w, x, y, z), or None for no rotation
        gizmo_scale : float
            Visual scale of the gizmo
        disable_rotations : bool
            If True, only show position handles (for sphere filter)
        """
        if not self._gizmo_enabled:
            return

        # Transform center to display space for gizmo position
        display_center = self._apply_scene_transform(center.reshape(1, 3))[0]

        # Compose rotations: filter rotation + scene transform rotation
        final_quat = rotation_quat or (1.0, 0.0, 0.0, 0.0)
        if self._transform_values is not None and hasattr(self._transform_values, 'rotation'):
            scene_rot = self._transform_values.rotation
            if scene_rot is not None:
                # Compose: scene_rot * filter_rot
                final_quat = self._quat_multiply(scene_rot, final_quat)

        # Check if we need to recreate gizmo (e.g., disable_rotations changed)
        needs_recreate = False
        if self._gizmo_handle is not None:
            # Check if disable_rotations setting changed
            if hasattr(self._gizmo_handle, 'disable_rotations'):
                if self._gizmo_handle.disable_rotations != disable_rotations:
                    needs_recreate = True
                    logger.debug(f"Gizmo needs recreate: disable_rotations changed from {self._gizmo_handle.disable_rotations} to {disable_rotations}")

        # Remove gizmo if it needs to be recreated
        if needs_recreate:
            self._remove_gizmo()

        # Create or update gizmo
        if self._gizmo_handle is None:
            self._gizmo_handle = self._server.scene.add_transform_controls(
                name="/filter_viz/gizmo",
                scale=gizmo_scale,
                line_width=3.0,
                position=tuple(display_center),
                wxyz=final_quat,
                disable_rotations=disable_rotations,
                disable_sliders=True,  # Use drag handles only
                depth_test=False,  # Always visible
                opacity=0.9,
            )

            # Register callbacks
            @self._gizmo_handle.on_update
            def on_gizmo_update(event) -> None:
                self._handle_gizmo_update()

        elif not self._updating_from_gizmo:
            # Update existing gizmo position/rotation
            # Skip if update is triggered by gizmo drag (gizmo is already at correct position)
            self._gizmo_handle.position = tuple(display_center)
            self._gizmo_handle.wxyz = final_quat

    def _handle_gizmo_update(self) -> None:
        """Handle gizmo manipulation - convert back to filter space and call callback."""
        if self._gizmo_handle is None or self._gizmo_callback is None:
            return

        if self._updating_from_gizmo:
            logger.warning("Gizmo update called while _updating_from_gizmo is True (possible re-entrant call)")
            return

        self._updating_from_gizmo = True
        try:
            # Get gizmo position in display space
            display_pos = np.array(self._gizmo_handle.position, dtype=np.float32)

            # Convert from display space back to filter space
            filter_pos = self._inverse_scene_transform(display_pos.reshape(1, 3))[0]

            # Get gizmo rotation
            gizmo_quat = tuple(self._gizmo_handle.wxyz)

            # Remove scene transform rotation to get filter-space rotation
            filter_quat = gizmo_quat
            if self._transform_values is not None and hasattr(self._transform_values, 'rotation'):
                scene_rot = self._transform_values.rotation
                if scene_rot is not None:
                    # filter_rot = scene_rot^-1 * gizmo_rot
                    scene_rot_inv = self._quat_inverse(scene_rot)
                    filter_quat = self._quat_multiply(scene_rot_inv, gizmo_quat)

            # Convert quaternion to axis-angle
            rotation_aa = _quaternion_to_axis_angle(filter_quat) if self._current_type != "Sphere" else None

            # Call the callback
            try:
                self._gizmo_callback(
                    self._current_type or "None",
                    (float(filter_pos[0]), float(filter_pos[1]), float(filter_pos[2])),
                    rotation_aa,
                )
            except Exception as e:
                logger.error(f"Gizmo callback raised exception: {e}", exc_info=True)
        finally:
            self._updating_from_gizmo = False

    def _inverse_scene_transform(self, verts: np.ndarray) -> np.ndarray:
        """Apply inverse scene transformation to convert from display space to filter space.

        Parameters
        ----------
        verts : np.ndarray
            Vertices in display space (N, 3)

        Returns
        -------
        np.ndarray
            Vertices in filter space (N, 3)
        """
        if self._transform_values is None:
            return verts

        tv = self._transform_values

        # Check if transform is neutral
        if hasattr(tv, 'is_neutral') and tv.is_neutral():
            return verts

        # Use gsmod's inverse_matrix() if available
        if hasattr(tv, 'inverse_matrix'):
            M_inv = tv.inverse_matrix()
            N = len(verts)
            verts_h = np.ones((N, 4), dtype=np.float32)
            verts_h[:, :3] = verts
            result_h = verts_h @ M_inv.T
            return result_h[:, :3].astype(np.float32)

        # Fallback: manual inverse transform
        # Original transform: P_display = S*R*(P - center) + center + translate
        # Inverse: P_filter = R^-1 * S^-1 * (P_display - center - translate) + center

        translate = getattr(tv, 'translate', None) or getattr(tv, 'translation', (0.0, 0.0, 0.0)) or (0.0, 0.0, 0.0)
        translate = np.array(translate, dtype=np.float32)

        scale_raw = getattr(tv, 'scale', 1.0) or 1.0
        if hasattr(scale_raw, '__len__'):
            scale = np.array(scale_raw, dtype=np.float32)
        else:
            scale = np.array([scale_raw, scale_raw, scale_raw], dtype=np.float32)

        rotate = getattr(tv, 'rotate', None) or getattr(tv, 'rotation', (1.0, 0.0, 0.0, 0.0)) or (1.0, 0.0, 0.0, 0.0)

        center = getattr(tv, 'center', None)
        if center is not None:
            center = np.array(center, dtype=np.float32)

        result = verts.copy()

        # Inverse of: translate -> center back -> rotate -> scale -> center subtract
        # 1. Subtract translation
        if np.any(np.abs(translate) > 1e-6):
            result = result - translate

        # 2. Subtract center
        if center is not None:
            result = result - center

        # 3. Inverse rotate
        w, x, y, z = rotate
        if abs(w - 1.0) > 1e-6 or abs(x) > 1e-6 or abs(y) > 1e-6 or abs(z) > 1e-6:
            rotate_inv = (w, -x, -y, -z)  # Quaternion inverse
            result = self._rotate_points(result, rotate_inv)

        # 4. Inverse scale
        if not np.allclose(scale, 1.0):
            result = result / scale

        # 5. Add center back
        if center is not None:
            result = result + center

        return result

    def _quat_multiply(
        self,
        q1: tuple[float, float, float, float],
        q2: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        """Multiply two quaternions (w, x, y, z format)."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        )

    def _quat_inverse(
        self,
        q: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        """Compute quaternion inverse (conjugate for unit quaternions)."""
        w, x, y, z = q
        return (w, -x, -y, -z)

    def update(
        self,
        filter_type: str,
        filter_values: FilterValues | None,
        transform_values: TransformValues | None = None,
    ) -> None:
        """Update visualization based on current filter settings.

        Parameters
        ----------
        filter_type : str
            Type of filter ("Sphere", "Box", "Ellipsoid", "Frustum", or "None")
        filter_values : FilterValues | None
            Filter parameters (None will clear visualization)
        transform_values : TransformValues | None
            Scene transformation to apply to visualization so it matches
            the transformed Gaussian positions
        """
        self._current_type = filter_type
        self._transform_values = transform_values
        self._last_filter_values = filter_values

        # Handle None filter_values by clearing visualization
        if filter_values is None:
            self._remove_gizmo()
            self._update_visibility()
            return

        try:
            if filter_type == "Sphere":
                self._update_sphere(filter_values)
            elif filter_type == "Box":
                self._update_box(filter_values)
            elif filter_type == "Ellipsoid":
                self._update_ellipsoid(filter_values)
            elif filter_type == "Frustum":
                self._update_frustum(filter_values)
            else:
                # No spatial filter - remove gizmo
                self._remove_gizmo()

            self._update_visibility()
        except Exception as e:
            logger.error(f"Failed to update filter visualization: {e}", exc_info=True)

    def _update_sphere(self, fv: FilterValues) -> None:
        """Update sphere visualization."""
        if fv.sphere_center is None:
            return

        center = np.array(fv.sphere_center, dtype=np.float32)
        radius = fv.sphere_radius if fv.sphere_radius < float("inf") else 10.0

        # Scale vertices by radius and translate to center
        verts = self._sphere_verts * radius + center

        # Apply scene transformation so visualization matches transformed Gaussians
        # Filter operates on original data, but we display in transformed space
        verts = self._apply_scene_transform(verts)

        # Remove old handle if exists
        self._remove_handle("_sphere_handle")

        # Create line segments - use single color (shape (3,))
        self._sphere_handle = self._server.scene.add_line_segments(
            name="/filter_viz/sphere",
            points=verts[self._sphere_lines.flatten()].reshape(-1, 2, 3),
            colors=self.SPHERE_COLOR,
            line_width=self.MAIN_LINE_WIDTH,
        )

        # Sphere has no rotation - remove any existing axes
        self._remove_axes()

        # Update gizmo (position only, no rotation for sphere)
        self._update_gizmo(
            center=center,
            rotation_quat=None,
            gizmo_scale=radius * 0.3,
            disable_rotations=True,
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

        # Apply box rotation if present
        if fv.box_rot is not None:
            quat = _axis_angle_to_quaternion(fv.box_rot)
            # For line segments, we need to rotate around center
            verts_centered = verts - center
            verts = self._rotate_points(verts_centered, quat) + center

        # Get rotation quaternion for axes (before scene transform)
        box_quat = _axis_angle_to_quaternion(fv.box_rot) if fv.box_rot is not None else None

        # Apply scene transformation so visualization matches transformed Gaussians
        # Filter operates on original data, but we display in transformed space
        verts = self._apply_scene_transform(verts)

        # Remove old handle if exists
        self._remove_handle("_box_handle")

        # Create line segments - use single color (shape (3,))
        self._box_handle = self._server.scene.add_line_segments(
            name="/filter_viz/box",
            points=verts[self._box_lines.flatten()].reshape(-1, 2, 3),
            colors=self.BOX_COLOR,
            line_width=self.MAIN_LINE_WIDTH,
        )

        # Add axis indicators at box center
        axis_size = min(size) * 0.6  # Scale axes relative to smallest box dimension
        self._add_axes(center, axis_size, box_quat)

        # Update gizmo (position + rotation)
        self._update_gizmo(
            center=center,
            rotation_quat=box_quat,
            gizmo_scale=min(size) * 0.4,
            disable_rotations=False,
        )

    def _update_ellipsoid(self, fv: FilterValues) -> None:
        """Update ellipsoid visualization."""
        if fv.ellipsoid_radii is None:
            return

        center = np.array(fv.ellipsoid_center or (0, 0, 0), dtype=np.float32)
        radii = np.array(fv.ellipsoid_radii, dtype=np.float32)

        # Scale sphere vertices by radii
        verts = self._sphere_verts * radii

        # Apply ellipsoid rotation if present
        ellipsoid_quat = None
        if fv.ellipsoid_rot is not None:
            ellipsoid_quat = _axis_angle_to_quaternion(fv.ellipsoid_rot)
            verts = self._rotate_points(verts, ellipsoid_quat)

        # Translate to center
        verts = verts + center

        # Apply scene transformation so visualization matches transformed Gaussians
        # Filter operates on original data, but we display in transformed space
        verts = self._apply_scene_transform(verts)

        # Remove old handle if exists
        self._remove_handle("_ellipsoid_handle")

        # Create line segments - use single color (shape (3,))
        self._ellipsoid_handle = self._server.scene.add_line_segments(
            name="/filter_viz/ellipsoid",
            points=verts[self._sphere_lines.flatten()].reshape(-1, 2, 3),
            colors=self.ELLIPSOID_COLOR,
            line_width=self.MAIN_LINE_WIDTH,
        )

        # Add axis indicators at ellipsoid center
        axis_size = min(radii) * 0.6  # Scale axes relative to smallest radius
        self._add_axes(center, axis_size, ellipsoid_quat)

        # Update gizmo (position + rotation)
        self._update_gizmo(
            center=center,
            rotation_quat=ellipsoid_quat,
            gizmo_scale=min(radii) * 0.4,
            disable_rotations=False,
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

        # Apply frustum rotation if present
        frustum_quat = None
        if fv.frustum_rot is not None:
            frustum_quat = _axis_angle_to_quaternion(fv.frustum_rot)
            verts = self._rotate_points(verts, frustum_quat)

        # Translate to position
        frustum_pos = np.array(fv.frustum_pos, dtype=np.float32)
        verts = verts + frustum_pos

        # Apply scene transformation so visualization matches transformed Gaussians
        # Filter operates on original data, but we display in transformed space
        verts = self._apply_scene_transform(verts)

        # Remove old handle if exists
        self._remove_handle("_frustum_handle")

        # Create line segments - use single color (shape (3,))
        self._frustum_handle = self._server.scene.add_line_segments(
            name="/filter_viz/frustum",
            points=verts[lines.flatten()].reshape(-1, 2, 3),
            colors=self.FRUSTUM_COLOR,
            line_width=self.MAIN_LINE_WIDTH,
        )

        # Add axis indicators at frustum position (camera location)
        axis_size = viz_far * 0.15  # Scale axes relative to frustum depth
        self._add_axes(frustum_pos, axis_size, frustum_quat)

        # Update gizmo (position + rotation)
        self._update_gizmo(
            center=frustum_pos,
            rotation_quat=frustum_quat,
            gizmo_scale=viz_far * 0.2,
            disable_rotations=False,
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

    def _apply_scene_transform(self, verts: np.ndarray) -> np.ndarray:
        """Apply scene transformation to visualization vertices.

        Uses gsmod's to_matrix() to get the exact same 4x4 transformation
        matrix used to transform Gaussians. This guarantees perfect consistency
        between filter visualization and actual Gaussian positions.

        Parameters
        ----------
        verts : np.ndarray
            Vertices in filter space (N, 3)

        Returns
        -------
        np.ndarray
            Vertices transformed to display space (N, 3)
        """
        if self._transform_values is None:
            return verts

        tv = self._transform_values

        # Check if transform is neutral (no-op)
        if hasattr(tv, 'is_neutral') and tv.is_neutral():
            return verts

        # Use gsmod's to_matrix() for exact consistency with Gaussian transform
        if hasattr(tv, 'to_matrix'):
            # Get the 4x4 transformation matrix from gsmod
            M = tv.to_matrix()  # 4x4 matrix

            # Convert to homogeneous coordinates, transform, convert back
            N = len(verts)
            verts_h = np.ones((N, 4), dtype=np.float32)
            verts_h[:, :3] = verts

            # Apply transformation: result = verts_h @ M.T
            # (M is row-major, so we transpose for column vectors)
            result_h = verts_h @ M.T
            return result_h[:, :3].astype(np.float32)

        # Fallback: manual transform (for older gsmod versions without to_matrix)
        # Get translation - handle both attribute names
        translate = getattr(tv, 'translate', None)
        if translate is None:
            translate = getattr(tv, 'translation', (0.0, 0.0, 0.0))
        if translate is None:
            translate = (0.0, 0.0, 0.0)
        translate = np.array(translate, dtype=np.float32)

        # Get scale - handle both scalar and per-axis
        scale_raw = getattr(tv, 'scale', 1.0)
        if scale_raw is None:
            scale_raw = 1.0
        if hasattr(scale_raw, '__len__'):
            scale = np.array(scale_raw, dtype=np.float32)
        else:
            scale = np.array([scale_raw, scale_raw, scale_raw], dtype=np.float32)

        # Get rotation quaternion - handle both attribute names
        rotate = getattr(tv, 'rotate', None)
        if rotate is None:
            rotate = getattr(tv, 'rotation', (1.0, 0.0, 0.0, 0.0))
        if rotate is None:
            rotate = (1.0, 0.0, 0.0, 0.0)

        # Get center/pivot point
        center = getattr(tv, 'center', None)
        if center is not None:
            center = np.array(center, dtype=np.float32)

        # Apply transformation: P_world = SR @ (P - center) + center + translation
        result = verts.copy()

        # 1. Subtract center (if set)
        if center is not None:
            result = result - center

        # 2. Apply per-axis scale
        if not np.allclose(scale, 1.0):
            result = result * scale

        # 3. Rotate (if not identity quaternion)
        w, x, y, z = rotate
        if abs(w - 1.0) > 1e-6 or abs(x) > 1e-6 or abs(y) > 1e-6 or abs(z) > 1e-6:
            result = self._rotate_points(result, rotate)

        # 4. Add center back (if set)
        if center is not None:
            result = result + center

        # 5. Translate
        if np.any(np.abs(translate) > 1e-6):
            result = result + translate

        return result

    def clear(self) -> None:
        """Remove all visualizations and reset state."""
        self._remove_handle("_sphere_handle")
        self._remove_handle("_box_handle")
        self._remove_handle("_ellipsoid_handle")
        self._remove_handle("_frustum_handle")
        self._remove_axes()
        self._remove_gizmo()

        # Reset state to prevent inconsistencies
        self._current_type = None
        self._last_filter_values = None
        self._visible = False
        # Note: _gizmo_enabled is intentionally NOT reset - user preference persists
        # Note: _updating_from_gizmo should already be False (clear shouldn't be called during update)
