"""UI setup module for viewer initialization.

Extracts UI setup logic from the main app class for better separation of concerns.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.gsplay.ui.filter_visualizer import FilterVisualizer
from src.gsplay.ui.layers import create_layer_controls


if TYPE_CHECKING:
    import viser

    from src.gsplay.config.settings import UIHandles
    from src.gsplay.core.app import UniversalGSPlay

logger = logging.getLogger(__name__)


class UISetup:
    """Handles UI component setup and callback registration.

    Extracts UI initialization logic from UniversalGSPlay to reduce
    complexity and improve testability.
    """

    def __init__(self, viewer: UniversalGSPlay):
        """Initialize UI setup helper.

        Parameters
        ----------
        viewer : UniversalGSPlay
            The viewer application instance
        """
        self._viewer = viewer

    @property
    def server(self) -> viser.ViserServer:
        """Get viser server."""
        return self._viewer.server

    @property
    def ui(self) -> UIHandles | None:
        """Get UI handles."""
        return self._viewer.ui

    @property
    def config(self):
        """Get viewer config."""
        return self._viewer.config

    @property
    def model(self):
        """Get current model."""
        return self._viewer.model

    def _register_callbacks(self, control_names: list[str], callback) -> None:
        """Register callback on multiple UI controls by attribute name.

        Parameters
        ----------
        control_names : list[str]
            List of attribute names on self.ui to register
        callback : callable
            The callback function to register on each control
        """
        if not self.ui:
            return
        for name in control_names:
            control = getattr(self.ui, name, None)
            if control:
                control.on_update(callback)

    def setup_all(self) -> FilterVisualizer | None:
        """Run all UI setup tasks.

        Returns
        -------
        FilterVisualizer | None
            The created filter visualizer, or None if UI not ready
        """
        if not self.ui:
            logger.warning("UI not initialized, skipping UI setup")
            return None

        # Setup layer controls for composite models
        self.setup_layer_controls()

        # Setup filter visualizer
        filter_visualizer = self.setup_filter_visualizer()

        # Setup auto-learn color controls
        self.setup_auto_learn_color()

        # Setup export option handlers
        self.setup_export_handlers()

        return filter_visualizer

    def setup_layer_controls(self) -> None:
        """Setup layer management UI if model supports layers."""
        if hasattr(self.model, "get_layer_info") and hasattr(self.model, "set_layer_visibility"):
            logger.info("Setting up layer controls for multi-layer model")
            layer_controls = create_layer_controls(self.server, self.model)
            logger.info(f"Layer controls created: {list(layer_controls.keys())}")
        else:
            logger.debug("Model does not support layers, skipping layer controls")

    def setup_filter_visualizer(self) -> FilterVisualizer:
        """Setup filter visualization gizmos and callbacks.

        Returns
        -------
        FilterVisualizer
            The created filter visualizer
        """
        filter_visualizer = FilterVisualizer(self.server)
        logger.debug("Filter visualizer created")

        if not self.ui:
            return filter_visualizer

        # Store reference for callbacks
        viewer = self._viewer

        # Callback to update visualization when show checkbox changes
        # Also enables/disables the interactive gizmo (merged control)
        def on_show_filter_viz_change(_) -> None:
            if filter_visualizer and self.ui and self.ui.show_filter_viz:
                show = self.ui.show_filter_viz.value
                filter_visualizer.visible = show
                filter_visualizer.set_gizmo_enabled(show)  # Gizmo follows visibility
                viewer._update_filter_visualization()

        # Callback to update visualization and config when filter parameters change
        def on_filter_change(_) -> None:
            viewer._update_filter_visualization()
            camera_pos, camera_rot = viewer._get_camera_state()
            self.config.filter_values = self.ui.get_filter_values(
                camera_position=camera_pos,
                camera_rotation=camera_rot,
            )

        # Register show/hide callback
        if self.ui.show_filter_viz:
            self.ui.show_filter_viz.on_update(on_show_filter_viz_change)

        # Callback from gizmo manipulation to update UI sliders
        def on_gizmo_update(
            filter_type: str,
            center: tuple[float, float, float],
            rotation_aa: tuple[float, float, float] | None,
        ) -> None:
            """Update UI sliders when gizmo is manipulated.

            Parameters
            ----------
            filter_type : str
                The active filter type ("Sphere", "Box", "Ellipsoid", "Frustum", or "None")
            center : tuple[float, float, float]
                The gizmo center position (x, y, z)
            rotation_aa : tuple[float, float, float] | None
                The gizmo rotation as axis-angle, or None for sphere (no rotation)
            """
            if not self.ui:
                return

            # Early exit for "None" or invalid filter types
            if filter_type not in ("Sphere", "Box", "Ellipsoid", "Frustum"):
                logger.debug(f"Ignoring gizmo update for filter type: {filter_type}")
                return

            from src.gsplay.config.rotation_conversions import axis_angle_to_euler_deg
            from src.gsplay.config.slider_constants import SliderBounds as SB

            # Helper to clamp values to slider bounds
            def clamp(value: float, min_val: float, max_val: float) -> float:
                return max(min_val, min(max_val, value))

            if filter_type == "Sphere":
                if self.ui.sphere_center_x:
                    self.ui.sphere_center_x.value = clamp(
                        center[0], SB.FILTER_CENTER_MIN, SB.FILTER_CENTER_MAX
                    )
                if self.ui.sphere_center_y:
                    self.ui.sphere_center_y.value = clamp(
                        center[1], SB.FILTER_CENTER_MIN, SB.FILTER_CENTER_MAX
                    )
                if self.ui.sphere_center_z:
                    self.ui.sphere_center_z.value = clamp(
                        center[2], SB.FILTER_CENTER_MIN, SB.FILTER_CENTER_MAX
                    )

            elif filter_type == "Box":
                if self.ui.box_center_x:
                    self.ui.box_center_x.value = clamp(
                        center[0], SB.FILTER_CENTER_MIN, SB.FILTER_CENTER_MAX
                    )
                if self.ui.box_center_y:
                    self.ui.box_center_y.value = clamp(
                        center[1], SB.FILTER_CENTER_MIN, SB.FILTER_CENTER_MAX
                    )
                if self.ui.box_center_z:
                    self.ui.box_center_z.value = clamp(
                        center[2], SB.FILTER_CENTER_MIN, SB.FILTER_CENTER_MAX
                    )
                if rotation_aa is not None:
                    rx, ry, rz = axis_angle_to_euler_deg(rotation_aa)
                    if self.ui.box_rot_x:
                        self.ui.box_rot_x.value = clamp(rx, SB.ROTATION_MIN, SB.ROTATION_MAX)
                    if self.ui.box_rot_y:
                        self.ui.box_rot_y.value = clamp(ry, SB.ROTATION_MIN, SB.ROTATION_MAX)
                    if self.ui.box_rot_z:
                        self.ui.box_rot_z.value = clamp(rz, SB.ROTATION_MIN, SB.ROTATION_MAX)

            elif filter_type == "Ellipsoid":
                if self.ui.ellipsoid_center_x:
                    self.ui.ellipsoid_center_x.value = clamp(
                        center[0], SB.FILTER_CENTER_MIN, SB.FILTER_CENTER_MAX
                    )
                if self.ui.ellipsoid_center_y:
                    self.ui.ellipsoid_center_y.value = clamp(
                        center[1], SB.FILTER_CENTER_MIN, SB.FILTER_CENTER_MAX
                    )
                if self.ui.ellipsoid_center_z:
                    self.ui.ellipsoid_center_z.value = clamp(
                        center[2], SB.FILTER_CENTER_MIN, SB.FILTER_CENTER_MAX
                    )
                if rotation_aa is not None:
                    rx, ry, rz = axis_angle_to_euler_deg(rotation_aa)
                    if self.ui.ellipsoid_rot_x:
                        self.ui.ellipsoid_rot_x.value = clamp(rx, SB.ROTATION_MIN, SB.ROTATION_MAX)
                    if self.ui.ellipsoid_rot_y:
                        self.ui.ellipsoid_rot_y.value = clamp(ry, SB.ROTATION_MIN, SB.ROTATION_MAX)
                    if self.ui.ellipsoid_rot_z:
                        self.ui.ellipsoid_rot_z.value = clamp(rz, SB.ROTATION_MIN, SB.ROTATION_MAX)

            elif filter_type == "Frustum":
                if self.ui.frustum_pos_x:
                    self.ui.frustum_pos_x.value = clamp(
                        center[0], SB.FRUSTUM_POSITION_MIN, SB.FRUSTUM_POSITION_MAX
                    )
                if self.ui.frustum_pos_y:
                    self.ui.frustum_pos_y.value = clamp(
                        center[1], SB.FRUSTUM_POSITION_MIN, SB.FRUSTUM_POSITION_MAX
                    )
                if self.ui.frustum_pos_z:
                    self.ui.frustum_pos_z.value = clamp(
                        center[2], SB.FRUSTUM_POSITION_MIN, SB.FRUSTUM_POSITION_MAX
                    )
                if rotation_aa is not None:
                    rx, ry, rz = axis_angle_to_euler_deg(rotation_aa)
                    if self.ui.frustum_rot_x:
                        self.ui.frustum_rot_x.value = clamp(rx, SB.ROTATION_MIN, SB.ROTATION_MAX)
                    if self.ui.frustum_rot_y:
                        self.ui.frustum_rot_y.value = clamp(ry, SB.ROTATION_MIN, SB.ROTATION_MAX)
                    if self.ui.frustum_rot_z:
                        self.ui.frustum_rot_z.value = clamp(rz, SB.ROTATION_MIN, SB.ROTATION_MAX)

            # Update config and trigger visualization update
            camera_pos, camera_rot = viewer._get_camera_state()
            self.config.filter_values = self.ui.get_filter_values(
                camera_position=camera_pos,
                camera_rotation=camera_rot,
            )

        # Register gizmo update callback
        filter_visualizer.set_gizmo_callback(on_gizmo_update)

        # Register filter type change callback
        if self.ui.spatial_filter_type:
            self.ui.spatial_filter_type.on_update(on_filter_change)

        # Register callbacks for all spatial filter controls
        self._register_callbacks(
            [
                # Sphere filter
                "sphere_center_x",
                "sphere_center_y",
                "sphere_center_z",
                "sphere_radius",
                # Box filter
                "box_center_x",
                "box_center_y",
                "box_center_z",
                "box_size_x",
                "box_size_y",
                "box_size_z",
                "box_rot_x",
                "box_rot_y",
                "box_rot_z",
                # Ellipsoid filter
                "ellipsoid_center_x",
                "ellipsoid_center_y",
                "ellipsoid_center_z",
                "ellipsoid_radius_x",
                "ellipsoid_radius_y",
                "ellipsoid_radius_z",
                "ellipsoid_rot_x",
                "ellipsoid_rot_y",
                "ellipsoid_rot_z",
                # Frustum filter
                "frustum_fov",
                "frustum_aspect",
                "frustum_near",
                "frustum_far",
                "frustum_pos_x",
                "frustum_pos_y",
                "frustum_pos_z",
                "frustum_rot_x",
                "frustum_rot_y",
                "frustum_rot_z",
            ],
            on_filter_change,
        )

        # Register "Use Current Camera" button callback
        if self.ui.frustum_use_camera:

            @self.ui.frustum_use_camera.on_click
            def on_use_camera_click(_) -> None:
                viewer._copy_camera_to_frustum()

        # Register "Use Scene Center" button callback
        if self.ui.use_scene_center:

            @self.ui.use_scene_center.on_click
            def on_use_scene_center_click(_) -> None:
                viewer._use_scene_center_for_filter()

        # Register "Align to Camera Up" button callback
        if self.ui.align_to_camera_up:

            @self.ui.align_to_camera_up.on_click
            def on_align_to_camera_up_click(_) -> None:
                viewer._align_filter_to_camera_up()

        # Callback to update visualization and lock filter controls when scene transform changes
        def on_transform_change(_) -> None:
            viewer._update_filter_visualization()
            # Lock filter controls and gizmo when transformation is active
            # (filter operates on original data, adjusting while transformed is confusing)
            if self.ui:
                transform_active = self.ui.is_transform_active()
                self.ui.set_filter_controls_disabled(transform_active)
                # Disable gizmo when transform is active, restore based on checkbox when not
                if filter_visualizer:
                    if transform_active:
                        filter_visualizer.set_gizmo_enabled(False)
                    elif self.ui.show_filter_viz and self.ui.show_filter_viz.value:
                        filter_visualizer.set_gizmo_enabled(True)

        # Register callbacks for scene transformation controls
        self._register_callbacks(
            [
                "scale_slider",
                "scale_x_slider",
                "scale_y_slider",
                "scale_z_slider",
                "translation_x_slider",
                "translation_y_slider",
                "translation_z_slider",
                "rotate_x_slider",
                "rotate_y_slider",
                "rotate_z_slider",
                "pivot_x_slider",
                "pivot_y_slider",
                "pivot_z_slider",
                "use_pivot_checkbox",
            ],
            on_transform_change,
        )

        logger.debug("Filter visualizer callbacks registered")

        # Check initial transform state and lock filter controls/gizmo if needed
        # (e.g., when loading a config with active transforms)
        if self.ui.is_transform_active():
            self.ui.set_filter_controls_disabled(True)
            filter_visualizer.set_gizmo_enabled(False)
            logger.debug("Filter controls and gizmo locked: transform is active")

        return filter_visualizer

    def setup_auto_learn_color(self) -> None:
        """Setup unified color adjustment callback."""
        if not self.ui:
            return

        viewer = self._viewer

        # Register "Apply" button callback for unified color adjustment
        if self.ui.apply_adjustment_button:

            @self.ui.apply_adjustment_button.on_click
            def on_apply_adjustment(_) -> None:
                viewer._apply_color_adjustment()

        logger.debug("Color adjustment callback registered")

    def setup_export_handlers(self) -> None:
        """Setup handlers for export format/device changes.

        Updates the export path automatically when options change,
        unless user has manually edited the path.
        """
        if not self.ui:
            return

        viewer = self._viewer

        def on_export_option_change(_) -> None:
            viewer._update_export_path_on_option_change()

        # Register format change handler
        if self.ui.export_format:
            self.ui.export_format.on_update(on_export_option_change)

        # Register device change handler
        if self.ui.export_device:
            self.ui.export_device.on_update(on_export_option_change)

        logger.debug("Export option handlers registered")
