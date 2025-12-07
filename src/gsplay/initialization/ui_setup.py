"""UI setup module for viewer initialization.

Extracts UI setup logic from the main app class for better separation of concerns.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.gsplay.ui.layers import create_layer_controls
from src.gsplay.ui.filter_visualizer import FilterVisualizer

if TYPE_CHECKING:
    import viser
    from src.gsplay.core.app import UniversalGSPlay
    from src.gsplay.config.settings import UIHandles

logger = logging.getLogger(__name__)


class UISetup:
    """Handles UI component setup and callback registration.

    Extracts UI initialization logic from UniversalGSPlay to reduce
    complexity and improve testability.
    """

    def __init__(self, viewer: "UniversalGSPlay"):
        """Initialize UI setup helper.

        Parameters
        ----------
        viewer : UniversalGSPlay
            The viewer application instance
        """
        self._viewer = viewer

    @property
    def server(self) -> "viser.ViserServer":
        """Get viser server."""
        return self._viewer.server

    @property
    def ui(self) -> "UIHandles | None":
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
        if hasattr(self.model, "get_layer_info") and hasattr(
            self.model, "set_layer_visibility"
        ):
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
        def on_show_filter_viz_change(_) -> None:
            if filter_visualizer and self.ui and self.ui.show_filter_viz:
                filter_visualizer.visible = self.ui.show_filter_viz.value
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

        # Register filter type change callback
        if self.ui.spatial_filter_type:
            self.ui.spatial_filter_type.on_update(on_filter_change)

        # Register callbacks for sphere filter
        for control in [
            self.ui.sphere_center_x,
            self.ui.sphere_center_y,
            self.ui.sphere_center_z,
            self.ui.sphere_radius,
        ]:
            if control:
                control.on_update(on_filter_change)

        # Register callbacks for box filter
        for control in [
            self.ui.box_min_x,
            self.ui.box_min_y,
            self.ui.box_min_z,
            self.ui.box_max_x,
            self.ui.box_max_y,
            self.ui.box_max_z,
            self.ui.box_rot_x,
            self.ui.box_rot_y,
            self.ui.box_rot_z,
        ]:
            if control:
                control.on_update(on_filter_change)

        # Register callbacks for ellipsoid filter
        for control in [
            self.ui.ellipsoid_center_x,
            self.ui.ellipsoid_center_y,
            self.ui.ellipsoid_center_z,
            self.ui.ellipsoid_radius_x,
            self.ui.ellipsoid_radius_y,
            self.ui.ellipsoid_radius_z,
            self.ui.ellipsoid_rot_x,
            self.ui.ellipsoid_rot_y,
            self.ui.ellipsoid_rot_z,
        ]:
            if control:
                control.on_update(on_filter_change)

        # Register callbacks for frustum filter
        for control in [
            self.ui.frustum_fov,
            self.ui.frustum_aspect,
            self.ui.frustum_near,
            self.ui.frustum_far,
            self.ui.frustum_pos_x,
            self.ui.frustum_pos_y,
            self.ui.frustum_pos_z,
            self.ui.frustum_rot_x,
            self.ui.frustum_rot_y,
            self.ui.frustum_rot_z,
        ]:
            if control:
                control.on_update(on_filter_change)

        # Register "Use Current Camera" button callback
        if self.ui.frustum_use_camera:

            @self.ui.frustum_use_camera.on_click
            def on_use_camera_click(_) -> None:
                viewer._copy_camera_to_frustum()

        # Callback to update visualization when scene transform changes
        def on_transform_change(_) -> None:
            viewer._update_filter_visualization()

        # Register callbacks for scene transformation controls
        for control in [
            getattr(self.ui, 'global_scale', None),
            getattr(self.ui, 'translate_x', None),
            getattr(self.ui, 'translate_y', None),
            getattr(self.ui, 'translate_z', None),
            getattr(self.ui, 'rotate_x', None),
            getattr(self.ui, 'rotate_y', None),
            getattr(self.ui, 'rotate_z', None),
        ]:
            if control:
                control.on_update(on_transform_change)

        logger.debug("Filter visualizer callbacks registered")
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
