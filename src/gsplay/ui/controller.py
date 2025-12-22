"""
UI Controller for managing UI updates based on application events.

This module decouples the UI update logic from the main application class,
subscribing to events and updating UI components accordingly.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.gsplay.interaction.events import Event, EventBus, EventType


if TYPE_CHECKING:
    from src.gsplay.config.settings import UIHandles

logger = logging.getLogger(__name__)


class UIController:
    """
    Controller for updating UI components in response to events.
    """

    def __init__(self, ui: UIHandles, event_bus: EventBus):
        """
        Initialize UI controller.

        Parameters
        ----------
        ui : UIHandles
            UI handles to update
        event_bus : EventBus
            Event bus to subscribe to
        """
        self.ui = ui
        self.event_bus = event_bus
        self._updating_from_event = False  # Guard against recursion
        self._setup_subscriptions()
        logger.debug("UIController initialized")

    def _setup_subscriptions(self) -> None:
        """Subscribe to relevant events."""
        self._subscriptions = [
            (EventType.MODEL_LOADED, self._on_model_loaded),
            (EventType.MODEL_LOAD_STARTED, self._on_model_load_started),
            (EventType.MODEL_LOAD_FAILED, self._on_model_load_failed),
            (EventType.RENDER_RESOLUTION_CHANGED, self._on_resolution_changed),
            (EventType.RENDER_STATS_UPDATED, self._on_render_stats_updated),
            (EventType.FRAME_CHANGED, self._on_frame_changed),
            (EventType.PLAY_PAUSE_TOGGLED, self._on_play_pause_toggled),
            (EventType.FPS_CHANGED, self._on_fps_changed),
        ]

        for event_type, callback in self._subscriptions:
            self.event_bus.subscribe(event_type, callback)

    def _on_model_load_started(self, event: Event) -> None:
        """Handle model load started event."""
        self._update_ui_control(self.ui.load_data_button, disabled=True)
        logger.debug("UI updated for model load start")

    def _on_model_loaded(self, event: Event) -> None:
        """Handle model loaded event."""
        data = event.data
        total_frames = data.get("total_frames", 0)
        source_path = data.get("source_path", "")

        # Update time slider
        self._update_ui_control(
            self.ui.time_slider, max=max(0, total_frames - 1), value=0, disabled=False
        )

        # Update data path input
        if source_path:
            self._update_ui_control(self.ui.data_path_input, value=source_path)

        # Re-enable load button
        self._update_ui_control(self.ui.load_data_button, disabled=False)

        # Update info panel
        if self.ui.info_panel:
            self.ui.info_panel.set_frame_index(0, total_frames)

        logger.info(f"UI updated for loaded model: {total_frames} frames")

    def _on_model_load_failed(self, event: Event) -> None:
        """Handle model load failed event."""
        self._update_ui_control(self.ui.load_data_button, disabled=False)
        logger.warning(f"Model load failed: {event.data.get('error')}")

    def _on_resolution_changed(self, event: Event) -> None:
        """Handle render resolution changed event."""
        resolution = event.data.get("resolution")
        if resolution:
            self._update_ui_control(self.ui.render_quality, value=resolution)

    def _on_render_stats_updated(self, event: Event) -> None:
        """Handle render stats updated event."""
        data = event.data

        if not self.ui.info_panel:
            return

        # Update info panel with new stats
        if "frame_index" in data:
            total = data.get("total_frames")
            self.ui.info_panel.set_frame_index(data["frame_index"], total)

        if "frame_filename" in data:
            self.ui.info_panel.set_file_name(data["frame_filename"])

        if "gaussian_count" in data:
            self.ui.info_panel.set_gaussian_count(data["gaussian_count"])

        if "throughput_fps" in data:
            self.ui.info_panel.set_loader_fps(data["throughput_fps"])

        if "render_fps" in data:
            self.ui.info_panel.set_render_fps(data["render_fps"])

    def _on_frame_changed(self, event: Event) -> None:
        """Handle frame changed event."""
        frame_index = event.data.get("frame_index")
        if frame_index is not None:
            self._update_ui_control(self.ui.time_slider, value=frame_index)

    def _on_play_pause_toggled(self, event: Event) -> None:
        """Handle play/pause toggled event."""
        playing = event.data.get("playing", False)
        self._update_ui_control(self.ui.auto_play, value=" Play" if playing else "Pause")

    def _on_fps_changed(self, event: Event) -> None:
        """Handle FPS changed event."""
        # Skip if we're already updating (prevents recursion)
        if self._updating_from_event:
            return
        fps = event.data.get("fps")
        if fps is not None:
            self._update_ui_control(self.ui.play_speed, value=fps)

    def _update_ui_control(self, control, **kwargs) -> None:
        """
        Update UI control attributes with null-check.

        Parameters
        ----------
        control : Any
            UI control to update (can be None)
        **kwargs
            Attributes to set on the control
        """
        if control is not None:
            self._updating_from_event = True
            try:
                for attr, value in kwargs.items():
                    # Only update if value is different to prevent callback loops
                    current_value = getattr(control, attr, None)
                    if current_value != value:
                        setattr(control, attr, value)
            finally:
                self._updating_from_event = False

    def cleanup(self) -> None:
        """Unsubscribe from all events."""
        for event_type, callback in self._subscriptions:
            self.event_bus.unsubscribe(event_type, callback)
