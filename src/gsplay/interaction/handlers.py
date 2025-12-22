"""
Event handlers and callback management for the Universal GSPlay.

This module handles all UI event callbacks, debouncing logic, and
handler registration.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

from src.gsplay.interaction.events import EventBus, EventType


if TYPE_CHECKING:
    from src.gsplay.config.settings import UIHandles
    from src.gsplay.interaction.playback import PlaybackController
    from src.gsplay.nerfview import GSPlay

logger = logging.getLogger(__name__)


class HandlerManager:
    """
    Manages event handlers and debounced callbacks for the viewer.

    This class centralizes all callback logic and provides debouncing
    for smooth user interaction. It translates UI events into Domain Events.
    """

    def __init__(self, event_bus: EventBus):
        """
        Initialize handler manager.

        Parameters
        ----------
        event_bus : EventBus
            Event bus for emitting events
        """
        self.event_bus = event_bus

        # Debouncing timers
        self._rerender_timer: threading.Timer | None = None
        self._mouse_rerender_timer: threading.Timer | None = None
        self._last_mouse_time: float = 0.0

        # Playback controller reference (for direct control)
        self.playback_controller: PlaybackController | None = None

        # GSPlay reference (for direct rerender if needed, though prefer events)
        self.viewer: GSPlay | None = None

        logger.debug("HandlerManager initialized")

    def set_viewer(self, viewer: GSPlay) -> None:
        """Set the viewer instance."""
        self.viewer = viewer
        logger.debug("GSPlay set in HandlerManager")

    def set_playback_controller(self, controller: PlaybackController) -> None:
        """Set the playback controller."""
        self.playback_controller = controller
        logger.debug("PlaybackController set in HandlerManager")

    def trigger_rerender(self, delay_ms: float = 200.0) -> None:
        """
        Trigger debounced rerender via event bus.

        Parameters
        ----------
        delay_ms : float
            Delay in milliseconds before rerendering (default 200ms)
        """
        # Update edit history (this is a bit implicit, ideally should be an event too)
        # But for now we assume the App listens to RERENDER_REQUESTED and updates history

        # Cancel any pending rerender
        if self._rerender_timer is not None:
            self._rerender_timer.cancel()

        # Schedule new rerender after delay
        def do_rerender():
            self.event_bus.emit(EventType.RERENDER_REQUESTED, source="handler_manager")

        self._rerender_timer = threading.Timer(delay_ms / 1000.0, do_rerender)
        self._rerender_timer.start()

        logger.debug(f"Scheduled rerender request with {delay_ms}ms delay")

    def trigger_immediate_rerender(self) -> None:
        """
        Trigger immediate rerender (no debouncing).

        Used for volume filtering for real-time response.
        """
        # Cancel any pending rerender
        if self._rerender_timer is not None:
            self._rerender_timer.cancel()

        # Immediate rerender
        self.event_bus.emit(EventType.RERENDER_REQUESTED, source="handler_manager")

        logger.debug("Triggered immediate rerender request")

    def _setup_slider_group(self, sliders: list, group_name: str, immediate: bool = False) -> None:
        """
        Setup callbacks for a group of sliders with identical behavior.

        Parameters
        ----------
        sliders : list
            List of slider handles (may contain None)
        group_name : str
            Name of the slider group for logging
        immediate : bool
            If True, use immediate rerender; otherwise use debounced rerender
        """

        def callback(_):
            if immediate:
                self.trigger_immediate_rerender()
            else:
                self.trigger_rerender()

        active_count = 0
        for slider in sliders:
            if slider is not None:
                slider.on_update(callback)
                active_count += 1

        logger.debug(f"Registered {active_count} {group_name} slider callbacks")

    def setup_time_slider_callback(self, ui: UIHandles) -> None:
        """
        Setup time slider callback for immediate frame updates.

        Supports both discrete frame indices and continuous source time.

        Parameters
        ----------
        ui : UIHandles
            UI handles
        """
        if ui.time_slider is not None:

            @ui.time_slider.on_update
            def _(_) -> None:
                if self.playback_controller:
                    # Check if we have a continuous time domain
                    time_domain = self.playback_controller.time_domain
                    if time_domain is not None and time_domain.is_continuous:
                        # Use source time for continuous sources
                        self.playback_controller.set_source_time(ui.time_slider.value)
                    else:
                        # Use frame index for discrete sources
                        self.playback_controller.set_frame(int(ui.time_slider.value))
                else:
                    # Fallback
                    self.event_bus.emit(EventType.RERENDER_REQUESTED, source="time_slider")

            logger.debug("Time slider callback registered")

    def setup_color_callbacks(self, ui: UIHandles) -> None:
        """
        Setup callbacks for all color adjustment sliders.

        Parameters
        ----------
        ui : UIHandles
            UI handles
        """
        color_sliders = [
            ui.temperature_slider,
            ui.tint_slider,
            ui.brightness_slider,
            ui.contrast_slider,
            ui.saturation_slider,
            ui.vibrance_slider,
            ui.hue_shift_slider,
            ui.gamma_slider,
            ui.shadows_slider,
            ui.highlights_slider,
            ui.fade_slider,
            ui.shadow_tint_hue_slider,
            ui.shadow_tint_sat_slider,
            ui.highlight_tint_hue_slider,
            ui.highlight_tint_sat_slider,
            ui.alpha_scaler_slider,
        ]

        self._setup_slider_group(color_sliders, "color")

    def setup_transform_callbacks(self, ui: UIHandles) -> None:
        """
        Setup callbacks for transform sliders.

        Parameters
        ----------
        ui : UIHandles
            UI handles
        """
        transform_sliders = [
            ui.translation_x_slider,
            ui.translation_y_slider,
            ui.translation_z_slider,
            # Scale (main + per-axis relative)
            ui.scale_slider,
            ui.scale_x_slider,
            ui.scale_y_slider,
            ui.scale_z_slider,
            # Rotation
            ui.rotate_x_slider,
            ui.rotate_y_slider,
            ui.rotate_z_slider,
            # Pivot point (gsmod 0.1.7 center)
            ui.pivot_x_slider,
            ui.pivot_y_slider,
            ui.pivot_z_slider,
        ]

        self._setup_slider_group(transform_sliders, "transform")

        # Use Pivot checkbox triggers rerender
        if ui.use_pivot_checkbox is not None:

            @ui.use_pivot_checkbox.on_update
            def _on_use_pivot_change(_):
                self.event_bus.emit(EventType.RERENDER_REQUESTED)

        # Bake View button - bakes camera view into model transform
        if ui.copy_center_button is not None:

            @ui.copy_center_button.on_click
            def _on_bake_view(_):
                self.event_bus.emit(EventType.BAKE_VIEW_REQUESTED)

    def setup_animation_callbacks(self, ui: UIHandles) -> None:
        """
        Setup callbacks for animation controls.

        Parameters
        ----------
        ui : UIHandles
            UI handles
        """
        if ui.play_speed is not None:

            def update_fps(_):
                if self.playback_controller:
                    # set_fps returns False if FPS is locked
                    if not self.playback_controller.set_fps(ui.play_speed.value):
                        # Reset slider to current FPS (locked value)
                        ui.play_speed.value = (
                            self.playback_controller.config.animation.play_speed_fps
                        )
                else:
                    self.trigger_rerender()

            ui.play_speed.on_update(update_fps)

        if ui.auto_play is not None:
            # Use PlaybackButtons wrapper interface
            def toggle_playback(_):
                if self.playback_controller:
                    self.playback_controller.toggle_play()
                else:
                    self.trigger_rerender()

            ui.auto_play.on_click(toggle_playback)

        if ui.render_quality is not None:

            def update_quality(_):
                # Sync quality slider with nerfview's viewer_res
                if self.viewer and hasattr(self.viewer, "render_tab_state"):
                    self.viewer.render_tab_state.viewer_res = int(ui.render_quality.value)
                self.trigger_rerender()

            ui.render_quality.on_update(update_quality)

        if ui.jpeg_quality_slider is not None:

            def update_jpeg_quality(_):
                # Update JPEG quality for streamed images
                if self.viewer:
                    quality = int(ui.jpeg_quality_slider.value)
                    self.viewer.jpeg_quality_static = quality
                    # Movement quality is 67% of static when auto-quality is on
                    if getattr(self.viewer, "auto_quality_enabled", False):
                        self.viewer.jpeg_quality_move = max(30, int(quality * 0.67))
                    else:
                        self.viewer.jpeg_quality_move = quality
                    logger.info(f"JPEG quality updated to {quality}")
                self.trigger_rerender()

            ui.jpeg_quality_slider.on_update(update_jpeg_quality)

        if ui.auto_quality_checkbox is not None:

            def update_auto_quality(_):
                # Toggle adaptive quality during camera movement
                if self.viewer:
                    self.viewer.auto_quality_enabled = ui.auto_quality_checkbox.value
                    # Also update the JPEG move quality based on new setting
                    if ui.jpeg_quality_slider is not None:
                        quality = int(ui.jpeg_quality_slider.value)
                        if self.viewer.auto_quality_enabled:
                            self.viewer.jpeg_quality_move = max(30, int(quality * 0.67))
                        else:
                            self.viewer.jpeg_quality_move = quality
                    logger.info(
                        f"Auto quality {'enabled' if self.viewer.auto_quality_enabled else 'disabled'}"
                    )

            ui.auto_quality_checkbox.on_update(update_auto_quality)

        # Source FPS input (for runtime configuration)
        if ui.source_fps_input is not None:

            def update_source_fps(_):
                new_fps = ui.source_fps_input.value
                # 0 means "not specified" - treat as None
                fps_value = new_fps if new_fps > 0 else None

                # Update model's source_fps if available (via playback controller)
                if self.playback_controller is not None:
                    model = self.playback_controller._model
                    if model is not None and hasattr(model, "source_fps"):
                        model.source_fps = fps_value
                        logger.info(f"Source FPS updated to {fps_value}")

                        # Refresh playback controller's cached time domain
                        self.playback_controller.refresh_time_domain()

                        # Re-emit MODEL_CHANGED to update time display
                        if hasattr(model, "time_domain"):
                            self.event_bus.emit(
                                EventType.MODEL_CHANGED,
                                time_domain=model.time_domain,
                            )

            ui.source_fps_input.on_update(update_source_fps)

        logger.debug("Animation callbacks registered")

    def setup_volume_filter_callbacks(self, ui: UIHandles) -> None:
        """
        Setup callbacks for volume filtering controls.

        Uses immediate rerender for real-time response.

        Parameters
        ----------
        ui : UIHandles
            UI handles
        """
        filter_controls = [
            # Basic opacity/scale
            ui.min_opacity_slider,
            ui.max_opacity_slider,
            ui.min_scale_slider,
            ui.max_scale_slider,
            # Sphere filter
            ui.sphere_center_x,
            ui.sphere_center_y,
            ui.sphere_center_z,
            ui.sphere_radius,
            # Box filter (center + size)
            ui.box_center_x,
            ui.box_center_y,
            ui.box_center_z,
            ui.box_size_x,
            ui.box_size_y,
            ui.box_size_z,
            # Ellipsoid filter
            ui.ellipsoid_center_x,
            ui.ellipsoid_center_y,
            ui.ellipsoid_center_z,
            ui.ellipsoid_radius_x,
            ui.ellipsoid_radius_y,
            ui.ellipsoid_radius_z,
            # Frustum filter
            ui.frustum_fov,
            ui.frustum_aspect,
            ui.frustum_near,
            ui.frustum_far,
            # Other
            ui.use_cpu_filtering_checkbox,
        ]

        self._setup_slider_group(filter_controls, "volume filter", immediate=True)

    def setup_filter_type_callback(
        self,
        ui: UIHandles,
        initial_scene_bounds: dict | None = None,
    ) -> None:
        """
        Setup callback for spatial filter type dropdown.

        Handles filter initialization and rerender triggering.
        Visibility is handled by the dropdown's own callback in layout.py.

        Parameters
        ----------
        ui : UIHandles
            UI handles
        initial_scene_bounds : dict | None
            Initial scene bounds for centering filters
        """
        if ui.spatial_filter_type is None:
            return

        @ui.spatial_filter_type.on_update
        def _(event):
            spatial_type = ui.spatial_filter_type.value
            logger.info(f"Spatial filter type changed to: {spatial_type}")

            # Initialize filter defaults when activated
            if spatial_type == "Sphere" and ui.sphere_radius is not None:
                # Set reasonable default radius if not set
                if ui.sphere_radius.value < 0.1:
                    ui.sphere_radius.value = 10.0
            elif spatial_type == "Box":
                # Set reasonable default box size if not set
                if ui.box_size_x is not None and ui.box_size_x.value < 0.2:
                    ui.box_size_x.value = 10.0
                    ui.box_size_y.value = 10.0
                    ui.box_size_z.value = 10.0
            elif spatial_type == "Ellipsoid":
                # Set reasonable default radii if not set
                if ui.ellipsoid_radius_x is not None and ui.ellipsoid_radius_x.value < 0.1:
                    ui.ellipsoid_radius_x.value = 5.0
                    ui.ellipsoid_radius_y.value = 5.0
                    ui.ellipsoid_radius_z.value = 5.0

            # Trigger rerender
            self.event_bus.emit(EventType.RERENDER_REQUESTED, source="spatial_filter_type")

        logger.debug("Spatial filter type callback registered")

    def setup_processing_mode_callback(self, ui: UIHandles) -> None:
        """
        Setup callback for processing mode dropdown.

        Parameters
        ----------
        ui : UIHandles
            UI handles
        """
        if ui.processing_mode_dropdown is None:
            return

        @ui.processing_mode_dropdown.on_update
        def _(event):
            from src.infrastructure.processing_mode import ProcessingMode

            mode_str = ui.processing_mode_dropdown.value
            logger.info(f"Processing mode changed to: {mode_str}")

            # Convert UI string to mode value for config
            try:
                mode = ProcessingMode.from_string(mode_str)

                # Log detailed processing path info
                filter_device = "CPU" if mode.filter_on_cpu else "GPU"
                color_device = "CPU" if mode.color_on_cpu else "GPU"
                transform_device = "CPU" if mode.transform_on_cpu else "GPU"

                logger.info(
                    f"Processing path enabled: Filter={filter_device}, "
                    f"Color={color_device}, Transform={transform_device}, "
                    f"CPU->GPU transfers={mode.transfer_count}"
                )

                logger.debug(f"Processing mode enum: {mode.value}")
            except ValueError as e:
                logger.error(f"Invalid processing mode: {mode_str}, error: {e}")

            # Trigger immediate rerender
            self.trigger_immediate_rerender()

        logger.debug("Processing mode callback registered")

    def setup_button_callbacks(self, ui: UIHandles) -> None:
        """
        Setup callbacks for all buttons.

        Parameters
        ----------
        ui : UIHandles
            UI handles
        """
        button_mappings = [
            (ui.export_ply_button, EventType.EXPORT_REQUESTED, {}),
            (ui.reset_colors_button, EventType.RESET_COLORS_REQUESTED, {}),
            (ui.reset_colors_advanced_button, EventType.RESET_COLORS_REQUESTED, {}),
            (ui.reset_pose_button, EventType.RESET_TRANSFORM_REQUESTED, {}),
            (ui.reset_filter_button, EventType.RESET_FILTER_REQUESTED, {}),
            (ui.center_button, EventType.CENTER_REQUESTED, {}),
            (ui.align_up_button, EventType.ALIGN_UP_REQUESTED, {}),
        ]

        for button, event_type, event_data in button_mappings:
            if button is not None:

                def make_callback(et, ed):
                    def callback(_):
                        self.event_bus.emit(et, **ed)

                    return callback

                button.on_click(make_callback(event_type, event_data))

        # Load data button (special case - needs path from input)
        if ui.load_data_button is not None and ui.data_path_input is not None:

            @ui.load_data_button.on_click
            def _(event):
                path = ui.data_path_input.value
                logger.info(f"Requesting load data from: {path}")
                self.event_bus.emit(EventType.LOAD_DATA_REQUESTED, path=path)

        # Terminate button
        if ui.terminate_button is not None:

            @ui.terminate_button.on_click
            def _(event):
                logger.info("Terminate instance requested")
                self.event_bus.emit(EventType.TERMINATE_REQUESTED)

        logger.debug("Button callbacks registered")

    def setup_export_scope_callbacks(self, ui: UIHandles) -> None:
        """
        Setup callbacks for export scope dropdown and time range controls.

        Handles visibility toggling and dynamic button label updates.

        Parameters
        ----------
        ui : UIHandles
            UI handles
        """
        if ui.export_scope_dropdown is None:
            return

        @ui.export_scope_dropdown.on_update
        def _on_scope_change(_):
            scope = ui.export_scope_dropdown.value
            show_time_range = scope == "Custom Time Range"

            # Toggle visibility of time range controls
            for ctrl in [
                ui.export_start_time_slider,
                ui.export_end_time_slider,
                ui.export_time_step_slider,
                ui.export_frame_preview,
                ui.export_snap_to_keyframe,
            ]:
                if ctrl:
                    ctrl.visible = show_time_range

            # Update button label based on scope
            if ui.export_ply_button:
                if scope == "Snapshot at Current Time":
                    ui.export_ply_button.content = "Export 1 Frame"
                elif scope == "Custom Time Range":
                    self._update_export_frame_preview(ui)
                else:
                    ui.export_ply_button.content = "Export All Frames"

            logger.debug(f"Export scope changed to: {scope}")

        # Update preview when time range controls change
        for slider in [
            ui.export_start_time_slider,
            ui.export_end_time_slider,
            ui.export_time_step_slider,
        ]:
            if slider:

                @slider.on_update
                def _on_time_range_change(_):
                    self._update_export_frame_preview(ui)

        # Update preview when snap checkbox changes
        if ui.export_snap_to_keyframe:

            @ui.export_snap_to_keyframe.on_update
            def _on_snap_change(_):
                self._update_export_frame_preview(ui)

        logger.debug("Export scope callbacks registered")

    def _update_export_frame_preview(self, ui: UIHandles) -> None:
        """
        Update the frame count preview for time range export.

        Handles snap-to-keyframe mode by calculating deduped keyframe count.

        Parameters
        ----------
        ui : UIHandles
            UI handles
        """
        if not all(
            [ui.export_start_time_slider, ui.export_end_time_slider, ui.export_time_step_slider]
        ):
            return

        start = ui.export_start_time_slider.value
        end = ui.export_end_time_slider.value
        step = ui.export_time_step_slider.value

        if step <= 0 or end < start:
            if ui.export_frame_preview:
                ui.export_frame_preview.value = "Invalid range"
            if ui.export_ply_button:
                ui.export_ply_button.content = "Export"
            return

        # Calculate sample count
        import numpy as np

        sample_times = np.arange(start, end + step * 0.5, step)
        raw_count = len(sample_times)

        # Check if snap is enabled and we have playback controller with time_domain
        snap_enabled = ui.export_snap_to_keyframe and ui.export_snap_to_keyframe.value

        if snap_enabled and self.playback_controller:
            time_domain = self.playback_controller.time_domain
            if (
                time_domain
                and time_domain.keyframe_times is not None
                and len(time_domain.keyframe_times) > 0
            ):
                # Calculate actual count after snap + dedup
                # Use same algorithm as export_source_time_range for consistency
                keyframe_indices = [
                    time_domain.source_time_to_nearest_keyframe(float(t))[0] for t in sample_times
                ]
                # Deduplicate consecutive (same algorithm as export)
                unique_count = 0
                prev_idx = None
                for idx in keyframe_indices:
                    if idx != prev_idx:
                        unique_count += 1
                        prev_idx = idx

                if ui.export_frame_preview:
                    ui.export_frame_preview.value = (
                        f"~{unique_count} keyframes (from {raw_count} samples)"
                    )
                if ui.export_ply_button:
                    ui.export_ply_button.content = f"Export {unique_count} Keyframes"
                return

        # Default: show raw count
        if ui.export_frame_preview:
            ui.export_frame_preview.value = f"~{raw_count} frames"
        if ui.export_ply_button:
            ui.export_ply_button.content = f"Export {raw_count} Frames"

    def setup_all_callbacks(self, ui: UIHandles, initial_scene_bounds: dict | None = None) -> None:
        """
        Setup all UI callbacks.

        Parameters
        ----------
        ui : UIHandles
            UI handles
        initial_scene_bounds : dict | None
            Initial scene bounds for filter initialization
        """
        logger.debug("Setting up all UI callbacks")

        self.setup_time_slider_callback(ui)
        self.setup_color_callbacks(ui)
        self.setup_transform_callbacks(ui)
        self.setup_animation_callbacks(ui)
        self.setup_volume_filter_callbacks(ui)
        self.setup_filter_type_callback(ui, initial_scene_bounds)
        self.setup_processing_mode_callback(ui)
        self.setup_button_callbacks(ui)
        self.setup_export_scope_callbacks(ui)

        logger.debug("All UI callbacks registered successfully")

    def cleanup(self) -> None:
        """Cancel any pending timers and cleanup resources."""
        if self._rerender_timer is not None:
            self._rerender_timer.cancel()
            self._rerender_timer = None

        if self._mouse_rerender_timer is not None:
            self._mouse_rerender_timer.cancel()
            self._mouse_rerender_timer = None

        logger.debug("HandlerManager cleanup complete")
