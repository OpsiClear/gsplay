"""
Playback controller for managing animation state and loop.

This module handles the animation loop, frame advancement, and playback state,
decoupling it from the UI and rendering logic.
"""

from __future__ import annotations

import logging
import time
import threading
from typing import TYPE_CHECKING

from src.gsplay.interaction.events import EventBus, EventType
from src.gsplay.config.settings import GSPlayConfig

if TYPE_CHECKING:
    from src.domain.interfaces import ModelInterface

logger = logging.getLogger(__name__)


class PlaybackController:
    """
    Controller for managing animation playback.

    Handles:
    - Play/Pause state
    - Frame advancement
    - Loop logic
    - FPS control
    """

    def __init__(self, config: GSPlayConfig, event_bus: EventBus):
        """
        Initialize playback controller.

        Parameters
        ----------
        config : GSPlayConfig
            GSPlay configuration
        event_bus : EventBus
            Event bus for emitting playback events
        """
        self.config = config
        self.event_bus = event_bus
        self._model: ModelInterface | None = None
        self._stop_event = threading.Event()

        # State tracking
        self._last_frame_time = 0.0
        self._last_frame_index = -1

        logger.debug("PlaybackController initialized")

    def set_model(self, model: ModelInterface | None) -> None:
        """Set the model to animate."""
        self._model = model
        if model:
            total_frames = model.get_total_frames()
            # Ensure current frame is valid
            if self.config.animation.current_frame >= total_frames:
                self.config.animation.current_frame = 0
        logger.debug(f"PlaybackController model set: {model}")

    def play(self) -> None:
        """Start playback."""
        if not self.config.animation.auto_play:
            self.config.animation.auto_play = True
            self.event_bus.emit(EventType.PLAY_PAUSE_TOGGLED, playing=True)
            logger.info("Playback started")

    def pause(self) -> None:
        """Pause playback."""
        if self.config.animation.auto_play:
            self.config.animation.auto_play = False
            self.event_bus.emit(EventType.PLAY_PAUSE_TOGGLED, playing=False)
            logger.info("Playback paused")

    def toggle_play(self) -> None:
        """Toggle playback state."""
        if self.config.animation.auto_play:
            self.pause()
        else:
            self.play()

    def set_frame(self, frame_index: int) -> None:
        """
        Set current frame index.

        Parameters
        ----------
        frame_index : int
            New frame index
        """
        if not self._model:
            return

        total_frames = self._model.get_total_frames()
        if total_frames <= 0:
            return

        # Clamp to valid range
        frame_index = max(0, min(frame_index, total_frames - 1))

        if self.config.animation.current_frame != frame_index:
            self.config.animation.current_frame = frame_index
            self.event_bus.emit(EventType.FRAME_CHANGED, frame_index=frame_index)

    def set_fps(self, fps: float) -> None:
        """Set playback speed in FPS."""
        self.config.animation.play_speed_fps = max(0.1, fps)
        self.event_bus.emit(
            EventType.FPS_CHANGED, fps=self.config.animation.play_speed_fps
        )

    def run_loop(self) -> None:
        """
        Run the main playback loop.

        This method blocks until stopped (intended to be run in main thread or separate thread).
        """
        logger.info("Starting playback loop")

        while not self._stop_event.is_set():
            try:
                if self.config.animation.auto_play and self._model:
                    self._advance_frame()

                # Sleep to maintain FPS
                fps = self.config.animation.play_speed_fps
                time.sleep(1.0 / fps)

            except KeyboardInterrupt:
                logger.info("Playback loop interrupted")
                break
            except Exception as e:
                logger.error(f"Error in playback loop: {e}", exc_info=True)
                time.sleep(1.0)  # Prevent tight loop on error

    def _advance_frame(self) -> None:
        """Advance to next frame based on loop logic."""
        if not self._model:
            return

        total_frames = self._model.get_total_frames()
        if total_frames <= 1:
            return

        current_frame = self.config.animation.current_frame
        next_frame = current_frame + 1

        # Loop logic
        if next_frame >= total_frames:
            next_frame = 0
            logger.debug("Playback looped to start")

        self.set_frame(next_frame)

    def stop(self) -> None:
        """Stop the playback loop."""
        self._stop_event.set()
        logger.info("Playback loop stopped")
