"""
UI handle collection for the Universal 4D Gaussian Splatting GSPlay.

This module contains the UIHandles dataclass that holds all viser UI control references.
Separated from settings.py to improve maintainability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import numpy as np
import viser
from gsmod import ColorValues, TransformValues, FilterValues

from src.gsplay.config.rotation_conversions import (
    euler_deg_to_axis_angle as _euler_deg_to_axis_angle,
    axis_angle_to_euler_deg as _axis_angle_to_euler_deg,
    quaternion_to_matrix as _quaternion_to_matrix,
    matrix_to_euler_deg as _matrix_to_euler_deg,
    camera_to_frustum_axis_angle as _camera_to_frustum_axis_angle,
    camera_to_frustum_euler_deg,
)

if TYPE_CHECKING:
    from src.domain.filters import VolumeFilter


@dataclass
class UIHandles:
    """Collection of UI control handles from viser.

    This replaces storing ~30 individual attributes on the main class.
    """

    # Data loader controls
    data_path_input: viser.GuiTextHandle | None = None
    load_data_button: viser.GuiButtonHandle | None = None

    # Info display (compact InfoPanel)
    info_panel: Any | None = None  # InfoPanel from layout.py

    # Animation controls
    time_slider: viser.GuiSliderHandle | None = None
    auto_play: viser.GuiButtonGroupHandle | None = None
    play_speed: viser.GuiSliderHandle | None = None
    render_quality: viser.GuiSliderHandle | None = None
    jpeg_quality_slider: viser.GuiSliderHandle | None = None
    auto_quality_checkbox: viser.GuiCheckboxHandle | None = None

    # View/Camera controls
    zoom_slider: viser.GuiSliderHandle | None = None
    azimuth_slider: viser.GuiSliderHandle | None = None
    elevation_slider: viser.GuiSliderHandle | None = None
    roll_slider: viser.GuiSliderHandle | None = None
    look_at_x_slider: viser.GuiSliderHandle | None = None
    look_at_y_slider: viser.GuiSliderHandle | None = None
    look_at_z_slider: viser.GuiSliderHandle | None = None

    # Color adjustment controls
    temperature_slider: viser.GuiSliderHandle | None = None
    tint_slider: viser.GuiSliderHandle | None = None
    brightness_slider: viser.GuiSliderHandle | None = None
    contrast_slider: viser.GuiSliderHandle | None = None
    saturation_slider: viser.GuiSliderHandle | None = None
    vibrance_slider: viser.GuiSliderHandle | None = None
    hue_shift_slider: viser.GuiSliderHandle | None = None
    gamma_slider: viser.GuiSliderHandle | None = None
    shadows_slider: viser.GuiSliderHandle | None = None
    highlights_slider: viser.GuiSliderHandle | None = None
    fade_slider: viser.GuiSliderHandle | None = None
    shadow_tint_hue_slider: viser.GuiSliderHandle | None = None
    shadow_tint_sat_slider: viser.GuiSliderHandle | None = None
    highlight_tint_hue_slider: viser.GuiSliderHandle | None = None
    highlight_tint_sat_slider: viser.GuiSliderHandle | None = None
    alpha_scaler_slider: viser.GuiSliderHandle | None = None
    reset_colors_button: viser.GuiButtonHandle | None = None
    reset_colors_advanced_button: viser.GuiButtonHandle | None = None

    # Unified color adjustment controls (gsmod 0.1.4 auto-correction + presets + advanced)
    color_adjustment_dropdown: viser.GuiDropdownHandle | None = None
    apply_adjustment_button: viser.GuiButtonHandle | None = None

    # Scene transformation controls
    translation_x_slider: viser.GuiSliderHandle | None = None
    translation_y_slider: viser.GuiSliderHandle | None = None
    translation_z_slider: viser.GuiSliderHandle | None = None
    global_scale_slider: viser.GuiSliderHandle | None = None
    # World-axis rotation sliders (truly gimbal-lock free via quaternion multiplication)
    rotate_x_slider: viser.GuiSliderHandle | None = None
    rotate_y_slider: viser.GuiSliderHandle | None = None
    rotate_z_slider: viser.GuiSliderHandle | None = None
    reset_pose_button: viser.GuiButtonHandle | None = None
    center_button: viser.GuiButtonHandle | None = None

    # Volume filtering controls - basic
    min_opacity_slider: viser.GuiSliderHandle | None = None
    max_opacity_slider: viser.GuiSliderHandle | None = None
    min_scale_slider: viser.GuiSliderHandle | None = None
    max_scale_slider: viser.GuiSliderHandle | None = None

    # Spatial filter type
    spatial_filter_type: viser.GuiDropdownHandle | None = None

    # Sphere filter
    sphere_center_x: viser.GuiSliderHandle | None = None
    sphere_center_y: viser.GuiSliderHandle | None = None
    sphere_center_z: viser.GuiSliderHandle | None = None
    sphere_radius: viser.GuiSliderHandle | None = None

    # Box filter
    box_min_x: viser.GuiSliderHandle | None = None
    box_min_y: viser.GuiSliderHandle | None = None
    box_min_z: viser.GuiSliderHandle | None = None
    box_max_x: viser.GuiSliderHandle | None = None
    box_max_y: viser.GuiSliderHandle | None = None
    box_max_z: viser.GuiSliderHandle | None = None
    box_rot_x: viser.GuiSliderHandle | None = None
    box_rot_y: viser.GuiSliderHandle | None = None
    box_rot_z: viser.GuiSliderHandle | None = None

    # Ellipsoid filter
    ellipsoid_center_x: viser.GuiSliderHandle | None = None
    ellipsoid_center_y: viser.GuiSliderHandle | None = None
    ellipsoid_center_z: viser.GuiSliderHandle | None = None
    ellipsoid_radius_x: viser.GuiSliderHandle | None = None
    ellipsoid_radius_y: viser.GuiSliderHandle | None = None
    ellipsoid_radius_z: viser.GuiSliderHandle | None = None
    ellipsoid_rot_x: viser.GuiSliderHandle | None = None
    ellipsoid_rot_y: viser.GuiSliderHandle | None = None
    ellipsoid_rot_z: viser.GuiSliderHandle | None = None

    # Frustum filter
    frustum_fov: viser.GuiSliderHandle | None = None
    frustum_aspect: viser.GuiSliderHandle | None = None
    frustum_near: viser.GuiSliderHandle | None = None
    frustum_far: viser.GuiSliderHandle | None = None
    frustum_pos_x: viser.GuiSliderHandle | None = None
    frustum_pos_y: viser.GuiSliderHandle | None = None
    frustum_pos_z: viser.GuiSliderHandle | None = None
    frustum_rot_x: viser.GuiSliderHandle | None = None
    frustum_rot_y: viser.GuiSliderHandle | None = None
    frustum_rot_z: viser.GuiSliderHandle | None = None
    frustum_use_camera: viser.GuiButtonHandle | None = None

    # Other filter controls
    processing_mode_dropdown: viser.GuiDropdownHandle | None = None
    use_cpu_filtering_checkbox: viser.GuiCheckboxHandle | None = None
    reset_filter_button: viser.GuiButtonHandle | None = None
    show_filter_viz: viser.GuiCheckboxHandle | None = None

    # Export controls
    export_path: viser.GuiTextHandle | None = None
    export_format: viser.GuiDropdownHandle | None = None
    export_device: viser.GuiDropdownHandle | None = None
    export_ply_button: viser.GuiButtonHandle | None = None

    # Config menu controls
    config_path_input: viser.GuiTextHandle | None = None
    config_buttons: viser.GuiButtonHandle | None = None  # Export Config button
    load_config_button: viser.GuiButtonHandle | None = None  # Load Config button (under play)

    # Instance control
    terminate_button: viser.GuiButtonHandle | None = None

    def get_color_values(self) -> ColorValues:
        """Extract current color values from UI with proper mapping."""
        # Get raw UI values
        temp_ui = self.temperature_slider.value if self.temperature_slider else 0.5
        tint_ui = self.tint_slider.value if self.tint_slider else 0.5
        shadows_ui = self.shadows_slider.value if self.shadows_slider else 1.0
        highlights_ui = self.highlights_slider.value if self.highlights_slider else 1.0

        # Map UI ranges to gsmod ranges
        # Temperature: UI [0, 1] -> gsmod [-1, 1]
        temperature = float(np.clip((temp_ui - 0.5) * 2.0, -1.0, 1.0))

        # Tint: UI [0, 1] -> gsmod [-1, 1]
        tint = float(np.clip((tint_ui - 0.5) * 2.0, -1.0, 1.0))

        # Shadows/Highlights: UI [0, 2] -> gsmod [-1, 1]
        shadows = float(np.clip(shadows_ui - 1.0, -1.0, 1.0))
        highlights = float(np.clip(highlights_ui - 1.0, -1.0, 1.0))

        return ColorValues(
            brightness=self.brightness_slider.value if self.brightness_slider else 1.0,
            contrast=self.contrast_slider.value if self.contrast_slider else 1.0,
            saturation=self.saturation_slider.value if self.saturation_slider else 1.0,
            vibrance=self.vibrance_slider.value if self.vibrance_slider else 1.0,
            hue_shift=self.hue_shift_slider.value if self.hue_shift_slider else 0.0,
            gamma=self.gamma_slider.value if self.gamma_slider else 1.0,
            temperature=temperature,
            tint=tint,
            shadows=shadows,
            highlights=highlights,
            fade=self.fade_slider.value if self.fade_slider else 0.0,
            shadow_tint_hue=self.shadow_tint_hue_slider.value if self.shadow_tint_hue_slider else 0.0,
            shadow_tint_sat=self.shadow_tint_sat_slider.value if self.shadow_tint_sat_slider else 0.0,
            highlight_tint_hue=self.highlight_tint_hue_slider.value if self.highlight_tint_hue_slider else 0.0,
            highlight_tint_sat=self.highlight_tint_sat_slider.value if self.highlight_tint_sat_slider else 0.0,
        )

    def get_transform_values(self) -> TransformValues:
        """Extract current transform values from UI with proper mapping.

        Uses world-axis rotation (truly gimbal-lock free via quaternion multiplication).
        Each rotation slider value is applied as rotation around that world axis.
        The rotations are composed via quaternion multiplication: Z * Y * X order.
        """
        from src.gsplay.rendering.quaternion_utils import (
            quat_from_axis_angle,
            quat_multiply,
            quat_normalize,
        )

        # Get world-axis rotation values (degrees)
        rot_x_deg = self.rotate_x_slider.value if self.rotate_x_slider else 0.0
        rot_y_deg = self.rotate_y_slider.value if self.rotate_y_slider else 0.0
        rot_z_deg = self.rotate_z_slider.value if self.rotate_z_slider else 0.0

        # Convert to radians
        rot_x_rad = np.radians(rot_x_deg)
        rot_y_rad = np.radians(rot_y_deg)
        rot_z_rad = np.radians(rot_z_deg)

        # Create quaternions for rotation around each world axis
        # Each slider independently rotates around its world axis
        # quaternion_utils uses wxyz format
        quat_x = quat_from_axis_angle(np.array([1.0, 0.0, 0.0]), rot_x_rad)
        quat_y = quat_from_axis_angle(np.array([0.0, 1.0, 0.0]), rot_y_rad)
        quat_z = quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), rot_z_rad)

        # Compose rotations: apply X, then Y, then Z (in world coordinates)
        # For world-axis rotation, order is: quat_z * quat_y * quat_x
        quat_wxyz = quat_multiply(quat_z, quat_multiply(quat_y, quat_x))
        quat_wxyz = quat_normalize(quat_wxyz)

        # gsmod uses wxyz format (w, x, y, z) - same as quaternion_utils
        w, x, y, z = quat_wxyz
        rotation_wxyz = (float(w), float(x), float(y), float(z))

        translation = (
            float(self.translation_x_slider.value) if self.translation_x_slider else 0.0,
            float(self.translation_y_slider.value) if self.translation_y_slider else 0.0,
            float(self.translation_z_slider.value) if self.translation_z_slider else 0.0,
        )
        scale_value = float(self.global_scale_slider.value) if self.global_scale_slider else 1.0

        return TransformValues(
            translation=translation, scale=scale_value, rotation=rotation_wxyz
        )

    def get_filter_values(
        self,
        camera_position: tuple[float, float, float] | None = None,
        camera_rotation: tuple[float, float, float, float] | None = None,
    ) -> FilterValues:
        """Extract current filter values from UI with full gsmod support.

        Parameters
        ----------
        camera_position : tuple[float, float, float] | None
            Camera position (x, y, z) for frustum filter. If None and Frustum
            is selected, uses (0, 0, 0).
        camera_rotation : tuple[float, float, float, float] | None
            Camera rotation as quaternion (w, x, y, z) for frustum filter.
            Will be converted to axis-angle for FilterValues.
        """
        # Basic opacity/scale filtering
        min_opacity = self.min_opacity_slider.value if self.min_opacity_slider else 0.0
        max_opacity = self.max_opacity_slider.value if self.max_opacity_slider else 1.0
        min_scale = self.min_scale_slider.value if self.min_scale_slider else 0.0
        max_scale = self.max_scale_slider.value if self.max_scale_slider else 100.0

        # Get spatial filter type
        spatial_type = self.spatial_filter_type.value if self.spatial_filter_type else "None"

        # Sphere filter
        sphere_radius = float("inf")
        sphere_center = (0.0, 0.0, 0.0)
        if spatial_type == "Sphere":
            sphere_radius = self.sphere_radius.value if self.sphere_radius else 10.0
            sphere_center = (
                self.sphere_center_x.value if self.sphere_center_x else 0.0,
                self.sphere_center_y.value if self.sphere_center_y else 0.0,
                self.sphere_center_z.value if self.sphere_center_z else 0.0,
            )

        # Box filter
        box_min = None
        box_max = None
        box_rotation = None
        if spatial_type == "Box":
            box_min = (
                self.box_min_x.value if self.box_min_x else -5.0,
                self.box_min_y.value if self.box_min_y else -5.0,
                self.box_min_z.value if self.box_min_z else -5.0,
            )
            box_max = (
                self.box_max_x.value if self.box_max_x else 5.0,
                self.box_max_y.value if self.box_max_y else 5.0,
                self.box_max_z.value if self.box_max_z else 5.0,
            )
            # Box rotation from UI - convert Euler angles to axis-angle
            if hasattr(self, "box_rot_x") and self.box_rot_x:
                rx = self.box_rot_x.value
                ry = self.box_rot_y.value if self.box_rot_y else 0.0
                rz = self.box_rot_z.value if self.box_rot_z else 0.0
                box_rotation = _euler_deg_to_axis_angle(rx, ry, rz)

        # Ellipsoid filter
        ellipsoid_center = None
        ellipsoid_radii = None
        ellipsoid_rotation = None
        if spatial_type == "Ellipsoid":
            ellipsoid_center = (
                self.ellipsoid_center_x.value if self.ellipsoid_center_x else 0.0,
                self.ellipsoid_center_y.value if self.ellipsoid_center_y else 0.0,
                self.ellipsoid_center_z.value if self.ellipsoid_center_z else 0.0,
            )
            ellipsoid_radii = (
                self.ellipsoid_radius_x.value if self.ellipsoid_radius_x else 5.0,
                self.ellipsoid_radius_y.value if self.ellipsoid_radius_y else 5.0,
                self.ellipsoid_radius_z.value if self.ellipsoid_radius_z else 5.0,
            )
            # Ellipsoid rotation from UI - convert Euler angles to axis-angle
            if hasattr(self, "ellipsoid_rot_x") and self.ellipsoid_rot_x:
                rx = self.ellipsoid_rot_x.value
                ry = self.ellipsoid_rot_y.value if self.ellipsoid_rot_y else 0.0
                rz = self.ellipsoid_rot_z.value if self.ellipsoid_rot_z else 0.0
                ellipsoid_rotation = _euler_deg_to_axis_angle(rx, ry, rz)

        # Frustum filter - read from UI controls (camera extrinsics as fallback)
        frustum_pos = None
        frustum_rot = None
        frustum_fov = 1.047  # 60 degrees in radians
        frustum_aspect = 1.0
        frustum_near = 0.1
        frustum_far = 100.0
        if spatial_type == "Frustum":
            # Read position from UI sliders (fallback to camera_position if not set)
            if hasattr(self, "frustum_pos_x") and self.frustum_pos_x:
                frustum_pos = (
                    self.frustum_pos_x.value,
                    self.frustum_pos_y.value if self.frustum_pos_y else 0.0,
                    self.frustum_pos_z.value if self.frustum_pos_z else 0.0,
                )
            elif camera_position is not None:
                frustum_pos = camera_position
            else:
                frustum_pos = (0.0, 0.0, 0.0)

            # Read rotation from UI sliders (Euler degrees -> axis-angle)
            if hasattr(self, "frustum_rot_x") and self.frustum_rot_x:
                # UI gives Euler angles in degrees, convert to axis-angle
                rx = self.frustum_rot_x.value if self.frustum_rot_x else 0.0
                ry = self.frustum_rot_y.value if self.frustum_rot_y else 0.0
                rz = self.frustum_rot_z.value if self.frustum_rot_z else 0.0
                frustum_rot = _euler_deg_to_axis_angle(rx, ry, rz)
            elif camera_rotation is not None:
                frustum_rot = _camera_to_frustum_axis_angle(camera_rotation)
            else:
                frustum_rot = (0.0, 0.0, 0.0)

            # Get FOV/aspect from UI
            fov_deg = self.frustum_fov.value if self.frustum_fov else 60.0
            frustum_fov = fov_deg * math.pi / 180.0
            frustum_aspect = self.frustum_aspect.value if self.frustum_aspect else 1.0
            frustum_near = self.frustum_near.value if self.frustum_near else 0.1
            frustum_far = self.frustum_far.value if self.frustum_far else 100.0

        return FilterValues(
            min_opacity=min_opacity,
            max_opacity=max_opacity,
            min_scale=min_scale,
            max_scale=max_scale,
            sphere_radius=sphere_radius,
            sphere_center=sphere_center,
            box_min=box_min,
            box_max=box_max,
            box_rot=box_rotation,
            ellipsoid_center=ellipsoid_center,
            ellipsoid_radii=ellipsoid_radii,
            ellipsoid_rot=ellipsoid_rotation,
            frustum_pos=frustum_pos,
            frustum_rot=frustum_rot,
            frustum_fov=frustum_fov,
            frustum_aspect=frustum_aspect,
            frustum_near=frustum_near,
            frustum_far=frustum_far,
        )

    def set_color_values(
        self, values: ColorValues, alpha_scaler: float | None = None
    ) -> None:
        """Update UI sliders with color values (inverse mapping)."""
        if self.temperature_slider:
            # Map [-1, 1] -> [0, 1]
            self.temperature_slider.value = (values.temperature / 2.0) + 0.5

        if self.tint_slider:
            # Map [-1, 1] -> [0, 1]
            self.tint_slider.value = (values.tint / 2.0) + 0.5

        if self.shadows_slider:
            # Map [-1, 1] -> [0, 2]
            self.shadows_slider.value = values.shadows + 1.0

        if self.highlights_slider:
            # Map [-1, 1] -> [0, 2]
            self.highlights_slider.value = values.highlights + 1.0

        if self.brightness_slider:
            self.brightness_slider.value = values.brightness
        if self.contrast_slider:
            self.contrast_slider.value = values.contrast
        if self.saturation_slider:
            self.saturation_slider.value = values.saturation
        if self.vibrance_slider:
            self.vibrance_slider.value = values.vibrance
        if self.hue_shift_slider:
            self.hue_shift_slider.value = values.hue_shift
        if self.gamma_slider:
            self.gamma_slider.value = values.gamma

        # New color controls
        if self.fade_slider:
            self.fade_slider.value = values.fade
        if self.shadow_tint_hue_slider:
            self.shadow_tint_hue_slider.value = values.shadow_tint_hue
        if self.shadow_tint_sat_slider:
            self.shadow_tint_sat_slider.value = values.shadow_tint_sat
        if self.highlight_tint_hue_slider:
            self.highlight_tint_hue_slider.value = values.highlight_tint_hue
        if self.highlight_tint_sat_slider:
            self.highlight_tint_sat_slider.value = values.highlight_tint_sat

        if alpha_scaler is not None and self.alpha_scaler_slider:
            self.alpha_scaler_slider.value = alpha_scaler

    def set_transform_values(self, values: TransformValues) -> None:
        """Update UI sliders with transform values.

        For rotation, decomposes the quaternion into XYZ Euler angles for display.
        This is only for UI display - the actual rotation computation in
        get_transform_values() uses gimbal-lock-free world-axis composition.
        """
        # gsmod uses 'translation' attribute
        translation = getattr(values, "translation", (0.0, 0.0, 0.0))
        if self.translation_x_slider:
            self.translation_x_slider.value = float(translation[0])
        if self.translation_y_slider:
            self.translation_y_slider.value = float(translation[1])
        if self.translation_z_slider:
            self.translation_z_slider.value = float(translation[2])

        if self.global_scale_slider:
            scale_value = getattr(values, "scale", 1.0)
            # Handle both scalar and vector scale
            if isinstance(scale_value, (float, int)):
                self.global_scale_slider.value = float(scale_value)
            else:
                self.global_scale_slider.value = float(scale_value[0])

        # Convert quaternion to Euler XYZ for rotation slider display
        # gsmod uses 'rotation' attribute in wxyz format (w, x, y, z)
        rotation = getattr(values, "rotation", (1.0, 0.0, 0.0, 0.0))
        if rotation is not None and any(
            s is not None
            for s in [self.rotate_x_slider, self.rotate_y_slider, self.rotate_z_slider]
        ):
            # Convert to tuple if needed
            if hasattr(rotation, "tolist"):
                rotation = tuple(rotation.tolist())
            else:
                rotation = tuple(rotation)

            # gsmod uses wxyz format (w, x, y, z)
            w, x, y, z = rotation
            # Normalize quaternion
            norm = np.sqrt(w*w + x*x + y*y + z*z)
            if norm > 1e-8:
                w, x, y, z = w/norm, x/norm, y/norm, z/norm

            # Convert quaternion to rotation matrix (use wxyz version)
            R = _quaternion_to_matrix((w, x, y, z))
            # Convert to Euler XYZ in degrees
            rx, ry, rz = _matrix_to_euler_deg(R)

            if self.rotate_x_slider:
                self.rotate_x_slider.value = float(rx)
            if self.rotate_y_slider:
                self.rotate_y_slider.value = float(ry)
            if self.rotate_z_slider:
                self.rotate_z_slider.value = float(rz)

    def get_alpha_scaler(self) -> float:
        """Read the opacity multiplier from the UI."""
        if self.alpha_scaler_slider:
            return float(self.alpha_scaler_slider.value)
        return 1.0

    def set_alpha_scaler(self, alpha_scaler: float) -> None:
        """Set the opacity multiplier slider if available."""
        if self.alpha_scaler_slider:
            self.alpha_scaler_slider.value = alpha_scaler

    def set_camera_values(
        self,
        azimuth: float,
        elevation: float,
        roll: float,
        distance: float,
        scene_bounds: dict | None = None,
        look_at: tuple[float, float, float] | None = None,
    ) -> None:
        """Update view control sliders with camera values.

        Parameters
        ----------
        azimuth : float
            Azimuth angle in degrees (0-360)
        elevation : float
            Elevation angle in degrees (-180 to 180)
        roll : float
            Roll angle in degrees (-180 to 180)
        distance : float
            Distance from look-at point
        scene_bounds : dict | None
            Scene bounds for calculating zoom slider (log2 scale)
        look_at : tuple[float, float, float] | None
            Camera target point (x, y, z)
        """
        if self.azimuth_slider:
            self.azimuth_slider.value = azimuth
        if self.elevation_slider:
            self.elevation_slider.value = elevation
        if self.roll_slider:
            self.roll_slider.value = roll

        # Look-at (camera target) sliders
        if look_at is not None:
            if self.look_at_x_slider:
                self.look_at_x_slider.value = float(np.clip(look_at[0], -50.0, 50.0))
            if self.look_at_y_slider:
                self.look_at_y_slider.value = float(np.clip(look_at[1], -50.0, 50.0))
            if self.look_at_z_slider:
                self.look_at_z_slider.value = float(np.clip(look_at[2], -50.0, 50.0))

        # Zoom uses log2 scale: zoom_log = log2(distance / default_distance)
        # where default_distance = scene_extent * 2.5
        if self.zoom_slider and scene_bounds:
            extent = scene_bounds.get("max_size", 10.0)
            default_distance = extent * 2.5
            if default_distance > 0 and distance > 0:
                actual_zoom = distance / default_distance
                zoom_log = np.log2(actual_zoom)
                # Clamp to slider range
                zoom_log = float(np.clip(zoom_log, -8.0, 3.0))
                self.zoom_slider.value = zoom_log

    def set_volume_filter(self, vf: VolumeFilter) -> None:
        """Update filter controls from a VolumeFilter config."""
        if self.spatial_filter_type:
            if vf.filter_type == "sphere":
                self.spatial_filter_type.value = "Sphere"
            elif vf.filter_type == "cuboid":
                self.spatial_filter_type.value = "Box"  # UI uses "Box" not "Cuboid"
            else:
                self.spatial_filter_type.value = "None"

        # Sphere center (use correct attribute names)
        if self.sphere_center_x and hasattr(vf, "sphere_center"):
            self.sphere_center_x.value = float(vf.sphere_center[0])
        if self.sphere_center_y and hasattr(vf, "sphere_center"):
            self.sphere_center_y.value = float(vf.sphere_center[1])
        if self.sphere_center_z and hasattr(vf, "sphere_center"):
            self.sphere_center_z.value = float(vf.sphere_center[2])

        # Sphere radius
        if self.sphere_radius and hasattr(vf, "sphere_radius_factor"):
            self.sphere_radius.value = float(vf.sphere_radius_factor)

        # Opacity/scale filters
        if self.min_opacity_slider and hasattr(vf, "opacity_threshold"):
            self.min_opacity_slider.value = float(vf.opacity_threshold)
        if self.max_opacity_slider and hasattr(vf, "max_opacity"):
            self.max_opacity_slider.value = float(vf.max_opacity)
        if self.min_scale_slider and hasattr(vf, "min_scale"):
            self.min_scale_slider.value = float(vf.min_scale)
        if self.max_scale_slider and hasattr(vf, "max_scale"):
            self.max_scale_slider.value = float(vf.max_scale)
        if self.use_cpu_filtering_checkbox and hasattr(vf, "use_cpu_filtering"):
            self.use_cpu_filtering_checkbox.value = bool(vf.use_cpu_filtering)

    def set_filter_values(self, fv: FilterValues) -> None:
        """Update spatial filter controls from FilterValues."""
        if self.min_opacity_slider and hasattr(fv, "min_opacity"):
            self.min_opacity_slider.value = float(fv.min_opacity)
        if self.max_opacity_slider and hasattr(fv, "max_opacity"):
            self.max_opacity_slider.value = float(fv.max_opacity)
        if self.min_scale_slider and hasattr(fv, "min_scale"):
            self.min_scale_slider.value = float(fv.min_scale)
        if self.max_scale_slider and hasattr(fv, "max_scale"):
            self.max_scale_slider.value = float(fv.max_scale)

        spatial_type = "None"
        sphere_radius = getattr(fv, "sphere_radius", float("inf"))
        if getattr(fv, "frustum_pos", None) is not None:
            spatial_type = "Frustum"
        elif getattr(fv, "ellipsoid_radii", None) is not None:
            spatial_type = "Ellipsoid"
        elif getattr(fv, "box_min", None) is not None and getattr(fv, "box_max", None) is not None:
            spatial_type = "Box"
        elif math.isfinite(sphere_radius):
            spatial_type = "Sphere"

        if self.spatial_filter_type:
            self.spatial_filter_type.value = spatial_type

        if hasattr(fv, "sphere_center") and fv.sphere_center is not None:
            if self.sphere_center_x:
                self.sphere_center_x.value = float(fv.sphere_center[0])
            if self.sphere_center_y:
                self.sphere_center_y.value = float(fv.sphere_center[1])
            if self.sphere_center_z:
                self.sphere_center_z.value = float(fv.sphere_center[2])
        if self.sphere_radius and math.isfinite(sphere_radius):
            self.sphere_radius.value = float(sphere_radius)

        if hasattr(fv, "box_min") and fv.box_min is not None:
            if self.box_min_x:
                self.box_min_x.value = float(fv.box_min[0])
            if self.box_min_y:
                self.box_min_y.value = float(fv.box_min[1])
            if self.box_min_z:
                self.box_min_z.value = float(fv.box_min[2])
        if hasattr(fv, "box_max") and fv.box_max is not None:
            if self.box_max_x:
                self.box_max_x.value = float(fv.box_max[0])
            if self.box_max_y:
                self.box_max_y.value = float(fv.box_max[1])
            if self.box_max_z:
                self.box_max_z.value = float(fv.box_max[2])
        if hasattr(fv, "box_rot") and fv.box_rot is not None:
            rx, ry, rz = _axis_angle_to_euler_deg(tuple(fv.box_rot))
            if self.box_rot_x:
                self.box_rot_x.value = float(rx)
            if self.box_rot_y:
                self.box_rot_y.value = float(ry)
            if self.box_rot_z:
                self.box_rot_z.value = float(rz)

        if hasattr(fv, "ellipsoid_center") and fv.ellipsoid_center is not None:
            if self.ellipsoid_center_x:
                self.ellipsoid_center_x.value = float(fv.ellipsoid_center[0])
            if self.ellipsoid_center_y:
                self.ellipsoid_center_y.value = float(fv.ellipsoid_center[1])
            if self.ellipsoid_center_z:
                self.ellipsoid_center_z.value = float(fv.ellipsoid_center[2])
        if hasattr(fv, "ellipsoid_radii") and fv.ellipsoid_radii is not None:
            if self.ellipsoid_radius_x:
                self.ellipsoid_radius_x.value = float(fv.ellipsoid_radii[0])
            if self.ellipsoid_radius_y:
                self.ellipsoid_radius_y.value = float(fv.ellipsoid_radii[1])
            if self.ellipsoid_radius_z:
                self.ellipsoid_radius_z.value = float(fv.ellipsoid_radii[2])
        if hasattr(fv, "ellipsoid_rot") and fv.ellipsoid_rot is not None:
            rx, ry, rz = _axis_angle_to_euler_deg(tuple(fv.ellipsoid_rot))
            if self.ellipsoid_rot_x:
                self.ellipsoid_rot_x.value = float(rx)
            if self.ellipsoid_rot_y:
                self.ellipsoid_rot_y.value = float(ry)
            if self.ellipsoid_rot_z:
                self.ellipsoid_rot_z.value = float(rz)

        if hasattr(fv, "frustum_fov") and self.frustum_fov:
            self.frustum_fov.value = float(fv.frustum_fov)
        if hasattr(fv, "frustum_aspect") and self.frustum_aspect:
            self.frustum_aspect.value = float(fv.frustum_aspect)
        if hasattr(fv, "frustum_near") and self.frustum_near:
            self.frustum_near.value = float(fv.frustum_near)
        if hasattr(fv, "frustum_far") and self.frustum_far:
            self.frustum_far.value = float(fv.frustum_far)
        if hasattr(fv, "frustum_pos") and fv.frustum_pos is not None:
            if self.frustum_pos_x:
                self.frustum_pos_x.value = float(fv.frustum_pos[0])
            if self.frustum_pos_y:
                self.frustum_pos_y.value = float(fv.frustum_pos[1])
            if self.frustum_pos_z:
                self.frustum_pos_z.value = float(fv.frustum_pos[2])
        if hasattr(fv, "frustum_rot") and fv.frustum_rot is not None:
            rx, ry, rz = _axis_angle_to_euler_deg(tuple(fv.frustum_rot))
            if self.frustum_rot_x:
                self.frustum_rot_x.value = float(rx)
            if self.frustum_rot_y:
                self.frustum_rot_y.value = float(ry)
            if self.frustum_rot_z:
                self.frustum_rot_z.value = float(rz)

