"""
Playback controller for managing animation state and loop.

This module handles the animation loop, frame advancement, and playback state,
decoupling it from the UI and rendering logic.

Supports both discrete frame-based and continuous time-based playback
through the TimeDomain abstraction.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING

from src.domain.time import TimeDomain
from src.gsplay.config.settings import GSPlayConfig
from src.gsplay.interaction.events import EventBus, EventType


if TYPE_CHECKING:
    from src.domain.interfaces import ModelInterface

logger = logging.getLogger(__name__)


class PlaybackController:
    """
    Controller for managing animation playback.

    Handles:
    - Play/Pause state
    - Frame advancement (discrete or continuous)
    - Loop logic
    - FPS control
    - Time domain awareness
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

        # Time domain tracking
        self._time_domain: TimeDomain | None = None
        self._current_source_time: float = 0.0

        # FPS locking (prevents UI from changing playback speed)
        self._lock_playback_fps: bool = False

        logger.debug("PlaybackController initialized")

    def set_model(self, model: ModelInterface | None) -> None:
        """Set the model to animate and adapt to its time domain.

        Also reads time configuration from the model:
        - playback_fps: Sets animation speed
        - lock_playback_fps: Locks FPS (UI cannot change)
        - autoplay: Auto-starts playback
        """
        self._model = model

        if model is not None:
            # Get time domain from model (with fallback for legacy models)
            if hasattr(model, "time_domain"):
                self._time_domain = model.time_domain
            else:
                # Legacy model: create discrete time domain
                total_frames = model.get_total_frames()
                self._time_domain = TimeDomain.discrete(total_frames)

            # Initialize source time to start
            self._current_source_time = self._time_domain.min_time

            # Ensure current frame is valid
            total_frames = model.get_total_frames()
            if self.config.animation.current_frame >= total_frames:
                self.config.animation.current_frame = 0

            # --- Read time configuration from model ---
            # Playback FPS
            if hasattr(model, "playback_fps"):
                self.config.animation.play_speed_fps = model.playback_fps
                logger.debug(f"Applied model playback_fps: {model.playback_fps}")

            # FPS locking
            if hasattr(model, "lock_playback_fps"):
                self._lock_playback_fps = model.lock_playback_fps
                logger.debug(f"FPS lock: {self._lock_playback_fps}")
            else:
                self._lock_playback_fps = False

            # Autoplay
            if hasattr(model, "autoplay") and model.autoplay:
                self.config.animation.auto_play = True
                logger.info("Autoplay enabled from model config")

            # Emit model changed event with time domain and playback config
            self.event_bus.emit(
                EventType.MODEL_CHANGED,
                time_domain=self._time_domain,
                playback_fps=self.config.animation.play_speed_fps,
                lock_playback_fps=self._lock_playback_fps,
                autoplay=self.config.animation.auto_play,
            )

            logger.info(
                f"Model set with time domain: "
                f"range [{self._time_domain.min_time}, {self._time_domain.max_time}], "
                f"continuous={self._time_domain.is_continuous}"
            )
        else:
            self._time_domain = None
            self._current_source_time = 0.0
            self._lock_playback_fps = False

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

        For discrete sources, this sets the exact frame.
        For continuous sources, this sets to keyframe time if available.

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

            # Update source time
            if self._time_domain is not None:
                if self._time_domain.keyframe_times:
                    # Use keyframe time if available
                    if 0 <= frame_index < len(self._time_domain.keyframe_times):
                        self._current_source_time = self._time_domain.keyframe_times[frame_index]
                else:
                    # Treat frame index as source time
                    self._current_source_time = float(frame_index)

            self.event_bus.emit(
                EventType.FRAME_CHANGED,
                frame_index=frame_index,
                source_time=self._current_source_time,
            )

    # --- Source Time Methods ---

    @property
    def time_domain(self) -> TimeDomain | None:
        """Get the current time domain."""
        return self._time_domain

    def refresh_time_domain(self) -> None:
        """Refresh the cached time domain from the model.

        Call this after changing model properties that affect time domain
        (e.g., source_fps).
        """
        if self._model is None:
            return

        if hasattr(self._model, "time_domain"):
            self._time_domain = self._model.time_domain
            logger.debug(
                f"Time domain refreshed: "
                f"range [{self._time_domain.min_time}, {self._time_domain.max_time}], "
                f"source_fps={self._time_domain.source_fps}"
            )

    @property
    def current_source_time(self) -> float:
        """Get current time in source units."""
        return self._current_source_time

    @property
    def current_normalized_time(self) -> float:
        """Get current time as normalized [0, 1]."""
        if self._time_domain is None:
            return 0.0
        return self._time_domain.to_normalized(self._current_source_time)

    def set_source_time(self, source_time: float) -> None:
        """Set current time in source units.

        This is the primary method for continuous time sources.

        Parameters
        ----------
        source_time : float
            Time in source-native units (frames, seconds, etc.)
        """
        if self._time_domain is None:
            return

        # Clamp to valid range
        source_time = self._time_domain.clamp(source_time)

        if self._current_source_time != source_time:
            self._current_source_time = source_time

            # Update config frame index for backward compatibility
            if self._time_domain.is_discrete:
                # Discrete: snap to nearest frame
                self.config.animation.current_frame = round(source_time)
            else:
                # Continuous: estimate frame for UI
                normalized = self._time_domain.to_normalized(source_time)
                total_frames = self._model.get_total_frames() if self._model else 1
                self.config.animation.current_frame = int(normalized * (total_frames - 1))

            self.event_bus.emit(
                EventType.FRAME_CHANGED,
                frame_index=self.config.animation.current_frame,
                source_time=source_time,
                normalized_time=self.current_normalized_time,
            )

    @property
    def lock_playback_fps(self) -> bool:
        """Whether playback FPS is locked (cannot be changed via UI)."""
        return self._lock_playback_fps

    def set_fps(self, fps: float, force: bool = False) -> bool:
        """Set playback speed in FPS.

        Parameters
        ----------
        fps : float
            Desired playback FPS
        force : bool
            If True, ignores lock and sets FPS anyway (for internal use)

        Returns
        -------
        bool
            True if FPS was changed, False if locked and not forced
        """
        if self._lock_playback_fps and not force:
            logger.debug(f"FPS change to {fps} blocked: FPS is locked")
            return False

        self.config.animation.play_speed_fps = max(0.1, fps)
        self.event_bus.emit(EventType.FPS_CHANGED, fps=self.config.animation.play_speed_fps)
        return True

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
        """Advance time based on playback speed and time domain."""
        if not self._model or self._time_domain is None:
            return

        fps = self.config.animation.play_speed_fps

        if self._time_domain.is_continuous:
            # Continuous time advancement
            if self._time_domain.keyframe_times is None:
                # Truly continuous (neural network): advance by 1/fps seconds
                delta = 1.0 / fps
            else:
                # Interpolated keyframes: advance by 1 frame per tick
                delta = 1.0

            new_time = self._current_source_time + delta

            # Loop logic
            if new_time > self._time_domain.max_time:
                new_time = self._time_domain.min_time
                logger.debug("Playback looped to start")

            self.set_source_time(new_time)
        else:
            # Discrete frame advancement (original behavior)
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
