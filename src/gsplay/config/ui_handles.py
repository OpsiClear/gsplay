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
    # Main uniform scale
    scale_slider: viser.GuiSliderHandle | None = None
    # Per-axis relative scale (multiplied by main scale)
    scale_x_slider: viser.GuiSliderHandle | None = None
    scale_y_slider: viser.GuiSliderHandle | None = None
    scale_z_slider: viser.GuiSliderHandle | None = None
    # World-axis rotation sliders (truly gimbal-lock free via quaternion multiplication)
    rotate_x_slider: viser.GuiSliderHandle | None = None
    rotate_y_slider: viser.GuiSliderHandle | None = None
    rotate_z_slider: viser.GuiSliderHandle | None = None
    # Pivot point for rotation/scale (gsmod 0.1.7 center parameter)
    use_pivot_checkbox: viser.GuiCheckboxHandle | None = None
    pivot_x_slider: viser.GuiSliderHandle | None = None
    pivot_y_slider: viser.GuiSliderHandle | None = None
    pivot_z_slider: viser.GuiSliderHandle | None = None
    copy_center_button: viser.GuiButtonHandle | None = None
    # Action buttons
    reset_pose_button: viser.GuiButtonHandle | None = None
    center_button: viser.GuiButtonHandle | None = None
    align_up_button: viser.GuiButtonHandle | None = None

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

    # Box filter (center + size for intuitive behavior with rotation)
    box_center_x: viser.GuiSliderHandle | None = None
    box_center_y: viser.GuiSliderHandle | None = None
    box_center_z: viser.GuiSliderHandle | None = None
    box_size_x: viser.GuiSliderHandle | None = None
    box_size_y: viser.GuiSliderHandle | None = None
    box_size_z: viser.GuiSliderHandle | None = None
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

    # Button to compute scene center from Gaussian mean
    use_scene_center: viser.GuiButtonHandle | None = None

    # Button to align filter rotation to camera up direction
    align_to_camera_up: viser.GuiButtonHandle | None = None

    # Other filter controls
    processing_mode_dropdown: viser.GuiDropdownHandle | None = None
    use_cpu_filtering_checkbox: viser.GuiCheckboxHandle | None = None
    reset_filter_button: viser.GuiButtonHandle | None = None
    show_filter_viz: viser.GuiCheckboxHandle | None = None  # Also enables interactive gizmo

    # Export controls
    export_path: viser.GuiTextHandle | None = None
    export_format: viser.GuiDropdownHandle | None = None
    export_device: viser.GuiDropdownHandle | None = None
    export_ply_button: viser.GuiButtonHandle | None = None

    # Config menu controls
    config_path_input: viser.GuiTextHandle | None = None
    config_buttons: viser.GuiButtonHandle | None = None  # Export Config button
    load_config_button: viser.GuiButtonHandle | None = None  # Load Config button (under play)
    reference_sphere_slider: viser.GuiSliderHandle | None = None  # Reference sphere radius

    # Instance control
    terminate_button: viser.GuiButtonHandle | None = None

    def _get_value(self, control, default: float) -> float:
        """Safely get control value with fallback.

        Parameters
        ----------
        control : GuiSliderHandle | GuiCheckboxHandle | None
            The UI control to read from
        default : float
            Default value if control is None

        Returns
        -------
        float
            The control's value or the default
        """
        return control.value if control else default

    def get_color_values(self) -> ColorValues:
        """Extract current color values from UI with proper mapping."""
        # Get raw UI values
        temp_ui = self._get_value(self.temperature_slider, 0.5)
        tint_ui = self._get_value(self.tint_slider, 0.5)
        shadows_ui = self._get_value(self.shadows_slider, 1.0)
        highlights_ui = self._get_value(self.highlights_slider, 1.0)

        # Map UI ranges to gsmod ranges
        # Temperature: UI [0, 1] -> gsmod [-1, 1]
        temperature = float(np.clip((temp_ui - 0.5) * 2.0, -1.0, 1.0))

        # Tint: UI [0, 1] -> gsmod [-1, 1]
        tint = float(np.clip((tint_ui - 0.5) * 2.0, -1.0, 1.0))

        # Shadows/Highlights: UI [0, 2] -> gsmod [-1, 1]
        shadows = float(np.clip(shadows_ui - 1.0, -1.0, 1.0))
        highlights = float(np.clip(highlights_ui - 1.0, -1.0, 1.0))

        return ColorValues(
            brightness=self._get_value(self.brightness_slider, 1.0),
            contrast=self._get_value(self.contrast_slider, 1.0),
            saturation=self._get_value(self.saturation_slider, 1.0),
            vibrance=self._get_value(self.vibrance_slider, 1.0),
            hue_shift=self._get_value(self.hue_shift_slider, 0.0),
            gamma=self._get_value(self.gamma_slider, 1.0),
            temperature=temperature,
            tint=tint,
            shadows=shadows,
            highlights=highlights,
            fade=self._get_value(self.fade_slider, 0.0),
            shadow_tint_hue=self._get_value(self.shadow_tint_hue_slider, 0.0),
            shadow_tint_sat=self._get_value(self.shadow_tint_sat_slider, 0.0),
            highlight_tint_hue=self._get_value(self.highlight_tint_hue_slider, 0.0),
            highlight_tint_sat=self._get_value(self.highlight_tint_sat_slider, 0.0),
        )

    def get_transform_values(self) -> TransformValues:
        """Extract current transform values from UI with proper mapping.

        Uses world-axis rotation (truly gimbal-lock free via quaternion multiplication).
        Each rotation slider value is applied as rotation around that world axis.
        The rotations are composed via quaternion multiplication: Z * Y * X order.

        Supports gsmod 0.1.7 features:
        - Per-axis scale (scale_x, scale_y, scale_z)
        - Rotation/scale center (pivot point)
        """
        from src.gsplay.rendering.quaternion_utils import (
            quat_from_axis_angle,
            quat_multiply,
            quat_normalize,
        )

        # Get world-axis rotation values (degrees)
        rot_x_deg = self._get_value(self.rotate_x_slider, 0.0)
        rot_y_deg = self._get_value(self.rotate_y_slider, 0.0)
        rot_z_deg = self._get_value(self.rotate_z_slider, 0.0)

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

        # Translation
        translation = (
            self._get_value(self.translation_x_slider, 0.0),
            self._get_value(self.translation_y_slider, 0.0),
            self._get_value(self.translation_z_slider, 0.0),
        )

        # Scale: main_scale * (rel_x, rel_y, rel_z)
        main_scale = self._get_value(self.scale_slider, 1.0)
        rel_x = self._get_value(self.scale_x_slider, 1.0)
        rel_y = self._get_value(self.scale_y_slider, 1.0)
        rel_z = self._get_value(self.scale_z_slider, 1.0)
        scale = (main_scale * rel_x, main_scale * rel_y, main_scale * rel_z)

        # Center/pivot point (gsmod 0.1.7) - None if checkbox unchecked
        center = None
        if self.use_pivot_checkbox and self.use_pivot_checkbox.value:
            center = (
                self._get_value(self.pivot_x_slider, 0.0),
                self._get_value(self.pivot_y_slider, 0.0),
                self._get_value(self.pivot_z_slider, 0.0),
            )

        return TransformValues(
            translation=translation,
            scale=scale,
            rotation=rotation_wxyz,
            center=center,
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
        min_opacity = self._get_value(self.min_opacity_slider, 0.0)
        max_opacity = self._get_value(self.max_opacity_slider, 1.0)
        min_scale = self._get_value(self.min_scale_slider, 0.0)
        max_scale = self._get_value(self.max_scale_slider, 100.0)

        # Get spatial filter type
        spatial_type = self.spatial_filter_type.value if self.spatial_filter_type else "None"

        # Sphere filter
        sphere_radius = float("inf")
        sphere_center = (0.0, 0.0, 0.0)
        if spatial_type == "Sphere":
            sphere_radius = self._get_value(self.sphere_radius, 10.0)
            sphere_center = (
                self._get_value(self.sphere_center_x, 0.0),
                self._get_value(self.sphere_center_y, 0.0),
                self._get_value(self.sphere_center_z, 0.0),
            )

        # Box filter (compute min/max from center + size)
        box_min = None
        box_max = None
        box_rotation = None
        if spatial_type == "Box":
            # Get center and size from UI
            cx = self._get_value(self.box_center_x, 0.0)
            cy = self._get_value(self.box_center_y, 0.0)
            cz = self._get_value(self.box_center_z, 0.0)
            sx = self._get_value(self.box_size_x, 10.0)
            sy = self._get_value(self.box_size_y, 10.0)
            sz = self._get_value(self.box_size_z, 10.0)
            # Compute min/max from center and half-extents
            half_x, half_y, half_z = sx / 2, sy / 2, sz / 2
            box_min = (cx - half_x, cy - half_y, cz - half_z)
            box_max = (cx + half_x, cy + half_y, cz + half_z)
            # Box rotation from UI - convert Euler angles to axis-angle
            if hasattr(self, "box_rot_x") and self.box_rot_x:
                rx = self._get_value(self.box_rot_x, 0.0)
                ry = self._get_value(self.box_rot_y, 0.0)
                rz = self._get_value(self.box_rot_z, 0.0)
                box_rotation = _euler_deg_to_axis_angle(rx, ry, rz)

        # Ellipsoid filter
        ellipsoid_center = None
        ellipsoid_radii = None
        ellipsoid_rotation = None
        if spatial_type == "Ellipsoid":
            ellipsoid_center = (
                self._get_value(self.ellipsoid_center_x, 0.0),
                self._get_value(self.ellipsoid_center_y, 0.0),
                self._get_value(self.ellipsoid_center_z, 0.0),
            )
            ellipsoid_radii = (
                self._get_value(self.ellipsoid_radius_x, 5.0),
                self._get_value(self.ellipsoid_radius_y, 5.0),
                self._get_value(self.ellipsoid_radius_z, 5.0),
            )
            # Ellipsoid rotation from UI - convert Euler angles to axis-angle
            if hasattr(self, "ellipsoid_rot_x") and self.ellipsoid_rot_x:
                rx = self._get_value(self.ellipsoid_rot_x, 0.0)
                ry = self._get_value(self.ellipsoid_rot_y, 0.0)
                rz = self._get_value(self.ellipsoid_rot_z, 0.0)
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
                    self._get_value(self.frustum_pos_x, 0.0),
                    self._get_value(self.frustum_pos_y, 0.0),
                    self._get_value(self.frustum_pos_z, 0.0),
                )
            elif camera_position is not None:
                frustum_pos = camera_position
            else:
                frustum_pos = (0.0, 0.0, 0.0)

            # Read rotation from UI sliders (Euler degrees -> axis-angle)
            if hasattr(self, "frustum_rot_x") and self.frustum_rot_x:
                # UI gives Euler angles in degrees, convert to axis-angle
                rx = self._get_value(self.frustum_rot_x, 0.0)
                ry = self._get_value(self.frustum_rot_y, 0.0)
                rz = self._get_value(self.frustum_rot_z, 0.0)
                frustum_rot = _euler_deg_to_axis_angle(rx, ry, rz)
            elif camera_rotation is not None:
                frustum_rot = _camera_to_frustum_axis_angle(camera_rotation)
            else:
                frustum_rot = (0.0, 0.0, 0.0)

            # Get FOV/aspect from UI
            fov_deg = self._get_value(self.frustum_fov, 60.0)
            frustum_fov = fov_deg * math.pi / 180.0
            frustum_aspect = self._get_value(self.frustum_aspect, 1.0)
            frustum_near = self._get_value(self.frustum_near, 0.1)
            frustum_far = self._get_value(self.frustum_far, 100.0)

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

        Supports gsmod 0.1.7 features:
        - Per-axis scale (scale_x, scale_y, scale_z)
        - Rotation/scale center (pivot point)
        """
        # gsmod uses 'translation' attribute
        translation = getattr(values, "translation", (0.0, 0.0, 0.0))
        if self.translation_x_slider:
            self.translation_x_slider.value = float(translation[0])
        if self.translation_y_slider:
            self.translation_y_slider.value = float(translation[1])
        if self.translation_z_slider:
            self.translation_z_slider.value = float(translation[2])

        # Scale: main_scale * (rel_x, rel_y, rel_z)
        # Slider bounds: main [0.1, 5.0], rel [0.5, 2.0]
        REL_MIN, REL_MAX = 0.5, 2.0
        MAIN_MIN, MAIN_MAX = 0.1, 5.0

        scale_value = getattr(values, "scale", (1.0, 1.0, 1.0))
        if isinstance(scale_value, (float, int)):
            # Uniform scale
            scale_f = float(scale_value)
            if MAIN_MIN <= scale_f <= MAIN_MAX:
                main_scale = scale_f
                rel_x, rel_y, rel_z = 1.0, 1.0, 1.0
            else:
                # Outside main range - use relative compensation
                if scale_f < MAIN_MIN:
                    main_scale = MAIN_MIN
                    rel_val = scale_f / MAIN_MIN
                else:
                    main_scale = MAIN_MAX
                    rel_val = scale_f / MAIN_MAX
                rel_val = max(REL_MIN, min(REL_MAX, rel_val))
                rel_x, rel_y, rel_z = rel_val, rel_val, rel_val
        else:
            sx, sy, sz = float(scale_value[0]), float(scale_value[1]), float(scale_value[2])
            scales = np.array([sx, sy, sz])

            # Use np.allclose for uniformity check (matches TransformValues.is_neutral)
            if np.allclose(scales, scales[0], rtol=1e-5, atol=1e-8):
                # Uniform - use as main scale
                main_scale = max(MAIN_MIN, min(MAIN_MAX, sx))
                rel_x, rel_y, rel_z = 1.0, 1.0, 1.0
            else:
                # Non-uniform - find optimal main_scale
                # For each axis: REL_MIN <= s/main <= REL_MAX
                main_lower = max(s / REL_MAX for s in scales)
                main_upper = min(s / REL_MIN for s in scales)

                if main_lower <= main_upper:
                    main_scale = float(np.sqrt(main_lower * main_upper))
                else:
                    main_scale = float(np.exp(np.mean(np.log(scales))))

                main_scale = max(MAIN_MIN, min(MAIN_MAX, main_scale))
                rel_x = max(REL_MIN, min(REL_MAX, sx / main_scale))
                rel_y = max(REL_MIN, min(REL_MAX, sy / main_scale))
                rel_z = max(REL_MIN, min(REL_MAX, sz / main_scale))

        if self.scale_slider:
            self.scale_slider.value = main_scale
        if self.scale_x_slider:
            self.scale_x_slider.value = rel_x
        if self.scale_y_slider:
            self.scale_y_slider.value = rel_y
        if self.scale_z_slider:
            self.scale_z_slider.value = rel_z

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

        # Center/pivot point (gsmod 0.1.7)
        center = getattr(values, "center", None)
        if center is not None:
            if self.use_pivot_checkbox:
                self.use_pivot_checkbox.value = True
            if self.pivot_x_slider:
                self.pivot_x_slider.value = float(center[0])
                self.pivot_x_slider.visible = True
            if self.pivot_y_slider:
                self.pivot_y_slider.value = float(center[1])
                self.pivot_y_slider.visible = True
            if self.pivot_z_slider:
                self.pivot_z_slider.value = float(center[2])
                self.pivot_z_slider.visible = True
        else:
            if self.use_pivot_checkbox:
                self.use_pivot_checkbox.value = False
            # Hide pivot sliders
            if self.pivot_x_slider:
                self.pivot_x_slider.visible = False
            if self.pivot_y_slider:
                self.pivot_y_slider.visible = False
            if self.pivot_z_slider:
                self.pivot_z_slider.visible = False
        # Note: copy_center_button (Bake View) visibility is not tied to pivot

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

    def is_transform_active(self) -> bool:
        """Check if any scene transformation is applied (non-neutral state).

        Uses gsmod's TransformValues.is_neutral() for threshold consistency.
        Filter controls should be locked when transform is active to avoid
        confusion (filter operates on original data, not transformed).
        """
        # Use gsmod's is_neutral() for consistent thresholds
        # This ensures UI and gsmod always agree on what's "active"
        transform_values = self.get_transform_values()
        return not transform_values.is_neutral()

    def set_filter_controls_disabled(self, disabled: bool) -> None:
        """Enable or disable all filter controls.

        Used to lock filter adjustment when scene transformation is active,
        since filtering operates on original (untransformed) data.

        Parameters
        ----------
        disabled : bool
            True to disable (lock) controls, False to enable
        """
        # Spatial filter type dropdown
        if self.spatial_filter_type:
            self.spatial_filter_type.disabled = disabled

        # Opacity/scale sliders
        for control in [
            self.min_opacity_slider,
            self.max_opacity_slider,
            self.min_scale_slider,
            self.max_scale_slider,
        ]:
            if control:
                control.disabled = disabled

        # Sphere filter controls
        for control in [
            self.sphere_center_x,
            self.sphere_center_y,
            self.sphere_center_z,
            self.sphere_radius,
        ]:
            if control:
                control.disabled = disabled

        # Box filter controls
        for control in [
            self.box_center_x,
            self.box_center_y,
            self.box_center_z,
            self.box_size_x,
            self.box_size_y,
            self.box_size_z,
            self.box_rot_x,
            self.box_rot_y,
            self.box_rot_z,
        ]:
            if control:
                control.disabled = disabled

        # Ellipsoid filter controls
        for control in [
            self.ellipsoid_center_x,
            self.ellipsoid_center_y,
            self.ellipsoid_center_z,
            self.ellipsoid_radius_x,
            self.ellipsoid_radius_y,
            self.ellipsoid_radius_z,
            self.ellipsoid_rot_x,
            self.ellipsoid_rot_y,
            self.ellipsoid_rot_z,
        ]:
            if control:
                control.disabled = disabled

        # Frustum filter controls
        for control in [
            self.frustum_fov,
            self.frustum_aspect,
            self.frustum_near,
            self.frustum_far,
            self.frustum_pos_x,
            self.frustum_pos_y,
            self.frustum_pos_z,
            self.frustum_rot_x,
            self.frustum_rot_y,
            self.frustum_rot_z,
            self.frustum_use_camera,
        ]:
            if control:
                control.disabled = disabled

        # Helper buttons
        for control in [
            self.use_scene_center,
            self.align_to_camera_up,
            self.reset_filter_button,
        ]:
            if control:
                control.disabled = disabled

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

        # Convert box_min/box_max to center/size for UI
        if hasattr(fv, "box_min") and fv.box_min is not None and hasattr(fv, "box_max") and fv.box_max is not None:
            bmin = fv.box_min
            bmax = fv.box_max
            # Compute center and size
            cx = (bmin[0] + bmax[0]) / 2
            cy = (bmin[1] + bmax[1]) / 2
            cz = (bmin[2] + bmax[2]) / 2
            sx = bmax[0] - bmin[0]
            sy = bmax[1] - bmin[1]
            sz = bmax[2] - bmin[2]
            if self.box_center_x:
                self.box_center_x.value = float(cx)
            if self.box_center_y:
                self.box_center_y.value = float(cy)
            if self.box_center_z:
                self.box_center_z.value = float(cz)
            if self.box_size_x:
                self.box_size_x.value = float(sx)
            if self.box_size_y:
                self.box_size_y.value = float(sy)
            if self.box_size_z:
                self.box_size_z.value = float(sz)
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

