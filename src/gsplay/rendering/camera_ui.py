"""
Camera UI controls for the universal viewer.

This module contains factory functions for creating camera-related UI controls
and the PlaybackButton helper class.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING

import numpy as np
import viser

if TYPE_CHECKING:
    from src.gsplay.rendering.camera import SuperSplatCamera

logger = logging.getLogger(__name__)


class PlaybackButton:
    """Wrapper that mimics button group interface for play/pause toggle."""

    def __init__(self, btn: viser.GuiButtonHandle, initial_playing: bool):
        self._btn = btn
        self._is_playing = initial_playing
        self._value = "Pause" if initial_playing else " Play"
        self._callbacks: list = []

    @property
    def value(self) -> str:
        return self._value

    @value.setter
    def value(self, v: str) -> None:
        self._value = v
        # Update button state based on value
        if v.strip() == "Play":
            self._is_playing = True
            self._btn.label = "Pause"
            self._btn.icon = viser.Icon.PLAYER_PAUSE
        else:
            self._is_playing = False
            self._btn.label = "Play"
            self._btn.icon = viser.Icon.PLAYER_PLAY

    def on_click(self, callback) -> None:
        self._callbacks.append(callback)

        @self._btn.on_click
        def _(event):
            # Toggle state
            self._is_playing = not self._is_playing
            if self._is_playing:
                self._value = " Play"
                self._btn.label = "Pause"
                self._btn.icon = viser.Icon.PLAYER_PAUSE
            else:
                self._value = "Pause"
                self._btn.label = "Play"
                self._btn.icon = viser.Icon.PLAYER_PLAY

            for cb in self._callbacks:
                cb(event)


def create_view_controls(
    server: viser.ViserServer, camera: SuperSplatCamera
) -> tuple:
    """
    Create View controls for camera (Zoom, Azimuth, Elevation, Roll, FOV, Presets).

    Parameters
    ----------
    server : viser.ViserServer
        Viser server instance
    camera : SuperSplatCamera
        Camera controller instance

    Returns
    -------
    tuple
        (zoom_slider, azimuth_slider, elevation_slider, roll_slider, setup_sync_func)
        setup_sync_func should be called after all controls are created to set up camera sync
    """
    # Flag to prevent circular updates (will be shared via closure)
    # Using timestamp approach because slider callbacks fire asynchronously
    # after synchronous flag would be reset
    _sync_until = [0.0]  # Block slider callbacks until this time

    def is_user_interaction() -> bool:
        """Check if this is a real user interaction (not programmatic sync).

        Returns False (skip processing) if:
        - Recently synced from camera (within cooldown window)
        - App owns camera (rotation, presets, cooldown period)
        """
        # Block during sync cooldown (slider callbacks are async)
        if time.perf_counter() < _sync_until[0]:
            return False
        if camera.is_app_controlled():
            return False
        return True

    # Zoom (logarithmic scale)
    zoom_min, zoom_max = -8.0, 3.0
    initial_zoom_log = 0.0
    if camera.state is not None and camera.scene_bounds:
        extent = camera.scene_bounds.get("max_size", 10.0)
        default_distance = extent * 2.5
        if camera.state.distance > 0:
            actual_zoom = camera.state.distance / default_distance
            initial_zoom_log = float(np.clip(np.log2(actual_zoom), zoom_min, zoom_max))

    zoom_slider = server.gui.add_slider(
        "Zoom",
        min=zoom_min,
        max=zoom_max,
        step=0.01,
        initial_value=initial_zoom_log,
        hint="Camera distance (log scale: 0=default, -7=0.01x, +1=2x farther)",
    )

    @zoom_slider.on_update
    def _(_) -> None:
        if not is_user_interaction():
            return
        if camera.state is not None and camera.scene_bounds:
            extent = camera.scene_bounds.get("max_size", 10.0)
            default_distance = extent * 2.5
            zoom_multiplier = 2.0**zoom_slider.value
            new_distance = default_distance * zoom_multiplier
            with camera.state_lock:
                # Use set_from_orbit to update distance (distance is a computed property)
                camera.state.set_from_orbit(
                    camera.state.azimuth,
                    camera.state.elevation,
                    camera.state.roll,
                    new_distance,
                    camera.state.look_at,
                )
            camera.apply_state_to_camera()

    # Manual orbit - initialize with current state values
    initial_azimuth = 45.0
    initial_elevation = 30.0
    initial_roll = 0.0
    if camera.state is not None:
        with camera.state_lock:
            initial_azimuth = camera.state.azimuth
            initial_elevation = camera.state.elevation
            initial_roll = camera.state.roll

    azimuth_slider = server.gui.add_slider(
        "Azimuth",
        min=0.0,
        max=360.0,
        step=1.0,
        initial_value=initial_azimuth,
    )

    elevation_slider = server.gui.add_slider(
        "Elevation",
        min=-89.0,
        max=89.0,
        step=1.0,
        initial_value=initial_elevation,
    )

    roll_slider = server.gui.add_slider(
        "Roll",
        min=-180.0,
        max=180.0,
        step=1.0,
        initial_value=initial_roll,
    )

    @azimuth_slider.on_update
    def _(_) -> None:
        if not is_user_interaction():
            return
        if camera.state is not None:
            with camera.state_lock:
                camera.state.set_from_euler(
                    azimuth_slider.value,
                    camera.state.elevation,
                    camera.state.roll,
                )
            camera.apply_state_to_camera()

    @elevation_slider.on_update
    def _(_) -> None:
        if not is_user_interaction():
            return
        if camera.state is not None:
            with camera.state_lock:
                camera.state.set_from_euler(
                    camera.state.azimuth,
                    elevation_slider.value,
                    camera.state.roll,
                )
            camera.apply_state_to_camera()

    @roll_slider.on_update
    def _(_) -> None:
        if not is_user_interaction():
            return
        if camera.state is not None:
            with camera.state_lock:
                camera.state.set_from_euler(
                    camera.state.azimuth,
                    camera.state.elevation,
                    roll_slider.value,
                )
            camera.apply_state_to_camera()

    # Look-at (camera target) sliders
    initial_look_at = np.zeros(3)
    if camera.state is not None:
        with camera.state_lock:
            initial_look_at = camera.state.look_at.copy()

    look_at_x_slider = server.gui.add_slider(
        "Target X",
        min=-50.0,
        max=50.0,
        step=0.1,
        initial_value=float(initial_look_at[0]),
        hint="Camera target X coordinate",
    )

    look_at_y_slider = server.gui.add_slider(
        "Target Y",
        min=-50.0,
        max=50.0,
        step=0.1,
        initial_value=float(initial_look_at[1]),
        hint="Camera target Y coordinate",
    )

    look_at_z_slider = server.gui.add_slider(
        "Target Z",
        min=-50.0,
        max=50.0,
        step=0.1,
        initial_value=float(initial_look_at[2]),
        hint="Camera target Z coordinate",
    )

    @look_at_x_slider.on_update
    def _(_) -> None:
        if not is_user_interaction():
            return
        if camera.state is not None:
            with camera.state_lock:
                camera.state.look_at[0] = look_at_x_slider.value
            camera.apply_state_to_camera()

    @look_at_y_slider.on_update
    def _(_) -> None:
        if not is_user_interaction():
            return
        if camera.state is not None:
            with camera.state_lock:
                camera.state.look_at[1] = look_at_y_slider.value
            camera.apply_state_to_camera()

    @look_at_z_slider.on_update
    def _(_) -> None:
        if not is_user_interaction():
            return
        if camera.state is not None:
            with camera.state_lock:
                camera.state.look_at[2] = look_at_z_slider.value
            camera.apply_state_to_camera()

    # FOV
    fov_slider = server.gui.add_slider(
        "FOV",
        min=10.0,
        max=120.0,
        step=1.0,
        initial_value=60.0,
    )

    @fov_slider.on_update
    def _(_) -> None:
        fov_radians = np.radians(fov_slider.value)
        for client in server.get_clients().values():
            client.camera.fov = fov_radians

    # Preset views
    ortho_buttons = server.gui.add_button_group(
        "Preset",
        (" Top ", "Front", " Side"),
    )

    @ortho_buttons.on_click
    def _(_) -> None:
        camera.stop_auto_rotation()
        view = ortho_buttons.value.strip()
        if camera.state is not None:
            with camera.state_lock:
                if view == "Top":
                    camera.state.set_from_euler(0.0, 90.0, 0.0)
                elif view == "Front":
                    camera.state.set_from_euler(0.0, 0.0, 0.0)
                elif view == "Side":
                    camera.state.set_from_euler(90.0, 0.0, 0.0)
            camera.apply_state_to_camera()

    # Transform buttons
    transform_buttons = server.gui.add_button_group(
        "Transform",
        ("Reset", " Flip"),
    )

    @transform_buttons.on_click
    def _(_) -> None:
        camera.stop_auto_rotation()
        action = transform_buttons.value.strip()
        if camera.state is not None:
            if action == "Reset":
                # Full reset: orientation, distance, AND look_at (camera target)
                # Use set_preset_view which properly resets all camera state
                camera.set_preset_view("iso")
            elif action == "Flip":
                # Flip: invert elevation and rotate azimuth 180 degrees
                with camera.state_lock:
                    current_azimuth = camera.state.azimuth
                    current_elevation = camera.state.elevation
                    current_roll = camera.state.roll
                    new_azimuth = (current_azimuth + 180.0) % 360.0
                    new_elevation = -current_elevation
                    camera.state.set_from_euler(new_azimuth, new_elevation, current_roll)
                camera.apply_state_to_camera()

    # Auto-rotation controls
    rotation_speed_slider = server.gui.add_slider(
        "Rotate Speed",
        min=5.0,
        max=180.0,
        step=5.0,
        initial_value=abs(camera._rotation_speed),
        hint="Rotation speed in degrees per second",
    )

    # Flag to prevent circular updates
    updating_rotation_ui = [False]

    @rotation_speed_slider.on_update
    def _(_) -> None:
        if updating_rotation_ui[0]:
            return
        # Update speed while preserving direction
        if camera._rotation_active:
            direction = 1.0 if camera._rotation_speed > 0 else -1.0
            camera.set_rotation_speed(rotation_speed_slider.value * direction)
        else:
            # Just update the stored speed for next start
            camera._rotation_speed = rotation_speed_slider.value

    rotate_buttons = server.gui.add_button_group(
        "Rotate",
        (" ↻ ", "Stop", " ↺ "),
    )

    @rotate_buttons.on_click
    def _(_) -> None:
        if updating_rotation_ui[0]:
            return
        action = rotate_buttons.value.strip()
        if action == "↻":
            camera.start_auto_rotation(axis="y", speed=rotation_speed_slider.value)
        elif action == "↺":
            camera.start_auto_rotation(axis="y", speed=-rotation_speed_slider.value)
        elif action == "Stop":
            camera.stop_auto_rotation()

    # Note: Observer pattern was removed in refactoring. Rotation state sync
    # is now handled directly via trigger_slider_sync() callback in CameraController.

    # Create slider sync function (will be registered with camera controller)
    def sync_sliders_with_camera(force: bool = False):
        """Sync UI sliders with camera state.

        Extracts azimuth/elevation/distance/look_at directly from viser's camera
        (source of truth for what user sees). Roll comes from our state since
        viser's orbit control doesn't preserve roll.

        Parameters
        ----------
        force : bool
            If True, bypass is_app_controlled() check (used by preset views)
        """
        # Skip during rotation to prevent feedback loop - rotation_step()
        # writes directly to viser, slider sync would read stale values
        if not force and camera.is_app_controlled():
            return

        # Get viser client for extracting current camera state
        clients = list(server.get_clients().values())
        if not clients:
            return

        try:
            client = clients[0]
            pos = np.asarray(client.camera.position, dtype=np.float64)
            target = np.asarray(client.camera.look_at, dtype=np.float64)

            # Extract spherical coords from viser's camera (matches what user sees)
            offset = pos - target
            distance = float(np.linalg.norm(offset))
            if distance < 1e-6:
                return

            offset_norm = offset / distance

            # Extract elevation from Y component
            elevation = float(np.degrees(np.arcsin(np.clip(offset_norm[1], -1.0, 1.0))))

            # Extract azimuth from XZ plane
            horiz_dist = np.sqrt(offset[0] ** 2 + offset[2] ** 2)
            if horiz_dist > 1e-6:
                azimuth = float(np.degrees(np.arctan2(offset[0], offset[2]))) % 360.0
            else:
                azimuth = azimuth_slider.value  # Keep current at poles

            # Roll comes from our state (viser doesn't preserve it)
            roll = camera.state.roll if camera.state else 0.0

            # Block slider callbacks for 100ms (async callbacks may fire later)
            _sync_until[0] = time.perf_counter() + 0.1

            # Update sliders (batched for proper GUI refresh)
            with client.atomic():
                azimuth_slider.value = azimuth
                elevation_slider.value = np.clip(elevation, -89.0, 89.0)
                roll_slider.value = np.clip(roll, -180.0, 180.0)

                # Sync look_at (camera target) sliders
                look_at_x_slider.value = np.clip(float(target[0]), -50.0, 50.0)
                look_at_y_slider.value = np.clip(float(target[1]), -50.0, 50.0)
                look_at_z_slider.value = np.clip(float(target[2]), -50.0, 50.0)

                # Sync zoom slider
                if camera.scene_bounds:
                    extent = camera.scene_bounds.get("max_size", 10.0)
                    default_distance = extent * 2.5
                    if default_distance > 0 and distance > 0:
                        actual_zoom = distance / default_distance
                        zoom_slider.value = np.clip(np.log2(actual_zoom), -8.0, 3.0)

        except Exception as e:
            logger.debug(f"Error syncing camera sliders: {e}")

    # Register slider sync callback with camera controller
    camera.set_slider_sync_callback(sync_sliders_with_camera)

    # Create sync function that sets up client callbacks
    def setup_camera_sync():
        """Set up camera sync callbacks for new clients."""

        @server.on_client_connect
        def _(client: viser.ClientHandle) -> None:
            if camera.state is not None:
                # Apply stored camera state after viser initializes the client
                def apply_state_after_init():
                    time.sleep(0.2)  # Wait for viser to be ready
                    try:
                        camera.apply_state_to_camera([client])
                        # Mark client as initialized - now on_update will sync state
                        camera.mark_client_initialized(client)
                        # Sync fov/aspect from viser immediately after initialization.
                        # This is needed because the on_update that fires during
                        # apply_state_to_camera is skipped (client not initialized yet).
                        # Without this, fov/aspect stay at defaults and rotation renders wrong.
                        if camera._state is not None:
                            with camera._lock:
                                camera._state.fov = client.camera.fov
                                camera._state.aspect = client.camera.aspect
                    except Exception:
                        pass  # Client may have disconnected

                threading.Thread(target=apply_state_after_init, daemon=True).start()

            @client.camera.on_update
            def _(_) -> None:
                # Sync state from user interactions (ignored until client initialized)
                camera.sync_state_from_client(client)
                sync_sliders_with_camera()

        @server.on_client_disconnect
        def _(client: viser.ClientHandle) -> None:
            # Clean up client tracking to prevent memory leak
            camera.remove_client(client)

    return (zoom_slider, azimuth_slider, elevation_slider, roll_slider, setup_camera_sync)


def create_fps_control(server: viser.ViserServer, config=None):
    """
    Create FPS slider for playback speed.

    Parameters
    ----------
    server : viser.ViserServer
        Viser server instance
    config : GSPlayConfig | None
        GSPlay configuration for animation settings

    Returns
    -------
    viser.GuiSliderHandle
        FPS slider handle
    """
    initial_fps = 30.0
    if config is not None and hasattr(config, "animation"):
        initial_fps = config.animation.play_speed_fps

    return server.gui.add_slider(
        "FPS",
        min=1.0,
        max=120.0,
        step=1.0,
        initial_value=initial_fps,
        hint="Playback frames per second",
    )


def create_quality_controls(server: viser.ViserServer, config=None) -> tuple:
    """
    Create Quality controls (Quality, JPEG, Auto Quality) for Config tab.

    Parameters
    ----------
    server : viser.ViserServer
        Viser server instance
    config : GSPlayConfig | None
        GSPlay configuration for render settings

    Returns
    -------
    tuple
        (render_quality, jpeg_quality_slider, auto_quality_checkbox)
    """
    # Check if compact mode is enabled - if so, default to Mid preset
    compact_mode = getattr(config, "compact_ui", False) if config else False

    if compact_mode:
        # Mid preset values for compact mode
        initial_quality = 1080
        initial_jpeg = 60
    else:
        # Default values (closer to High)
        initial_quality = 1280
        initial_jpeg = 90

    if config is not None and hasattr(config, "render_settings") and not compact_mode:
        initial_jpeg = config.render_settings.jpeg_quality_static

    # Quality presets button group
    quality_presets = server.gui.add_button_group(
        "Preset",
        options=("Low", "Mid", "High"),
        hint="Quick quality presets for different use cases",
    )

    render_quality = server.gui.add_slider(
        "Quality",
        min=540,
        max=2048,
        step=1,
        initial_value=initial_quality,
        hint="Maximum rendering resolution (higher = sharper but slower)",
    )

    jpeg_quality_slider = server.gui.add_slider(
        "JPEG",
        min=10,
        max=100,
        step=5,
        initial_value=initial_jpeg,
        hint="JPEG compression quality for streamed images",
    )

    auto_quality_checkbox = server.gui.add_checkbox(
        "Auto Quality",
        initial_value=True,
        hint="Reduce quality during camera movement for smoother navigation",
    )

    # Wire up preset button group
    @quality_presets.on_click
    def _on_preset_click(event: viser.GuiEvent) -> None:
        preset = event.target.value
        if preset == "Low":
            render_quality.value = 540
            jpeg_quality_slider.value = 30
        elif preset == "Mid":
            render_quality.value = 1080
            jpeg_quality_slider.value = 60
        elif preset == "High":
            render_quality.value = 1440
            jpeg_quality_slider.value = 90

    return (render_quality, jpeg_quality_slider, auto_quality_checkbox)


def create_playback_controls(server: viser.ViserServer, config=None, time_domain=None):
    """
    Create Frame/Time slider and Play/Pause toggle button for animation playback.

    Parameters
    ----------
    server : viser.ViserServer
        Viser server instance
    config : GSPlayConfig | None
        GSPlay configuration for animation settings
    time_domain : TimeDomain | None
        Time domain from the current source. If None, uses default discrete frames.

    Returns
    -------
    tuple
        (time_slider, playback_button)
    """
    initial_frame = 0
    initial_auto_play = False
    if config is not None and hasattr(config, "animation"):
        initial_frame = config.animation.current_frame
        initial_auto_play = config.animation.auto_play

    # Determine slider configuration from time domain
    if time_domain is not None and time_domain.is_continuous:
        # Continuous time slider
        slider_label = "Time"
        slider_min = time_domain.min_time
        slider_max = time_domain.max_time
        # Fine granularity for continuous (1000 steps across range)
        slider_step = max(0.001, (time_domain.max_time - time_domain.min_time) / 1000)
        slider_initial = time_domain.min_time
        slider_hint = "Continuous time (seconds or interpolated)"
    elif time_domain is not None:
        # Discrete time domain (non-default)
        slider_label = "Frame"
        slider_min = int(time_domain.min_time)
        slider_max = int(time_domain.max_time)
        slider_step = 1
        slider_initial = int(time_domain.min_time)
        slider_hint = "Frame index"
    else:
        # Default: discrete frames (placeholder until model loads)
        slider_label = "Frame"
        slider_min = 0
        slider_max = 1
        slider_step = 1
        slider_initial = initial_frame
        slider_hint = "Time frame for 4D content"

    # Create slider
    time_slider = server.gui.add_slider(
        slider_label,
        min=slider_min,
        max=slider_max,
        step=slider_step,
        initial_value=slider_initial,
        hint=slider_hint,
    )

    # Single toggle button - starts as Play (not playing) or Pause (playing)
    initial_label = "Pause" if initial_auto_play else "Play"
    initial_icon = viser.Icon.PLAYER_PAUSE if initial_auto_play else viser.Icon.PLAYER_PLAY

    toggle_button = server.gui.add_button(
        initial_label,
        icon=initial_icon,
        hint="Toggle animation playback",
    )

    return (time_slider, PlaybackButton(toggle_button, initial_auto_play))


def update_time_slider_for_source(time_slider, time_domain) -> None:
    """Update time slider configuration when source changes.

    Called when a new model is loaded to adapt the slider to its time domain.

    Parameters
    ----------
    time_slider : viser.GuiSliderHandle
        The time slider handle
    time_domain : TimeDomain
        Time domain from the new source
    """
    if time_domain is None:
        return

    if time_domain.is_continuous:
        # Continuous time
        time_slider.min = time_domain.min_time
        time_slider.max = time_domain.max_time
        time_slider.step = max(0.001, (time_domain.max_time - time_domain.min_time) / 1000)
        time_slider.value = time_domain.min_time
    else:
        # Discrete frames
        time_slider.min = int(time_domain.min_time)
        time_slider.max = int(time_domain.max_time)
        time_slider.step = int(time_domain.step) if time_domain.step else 1
        time_slider.value = int(time_domain.min_time)


def create_supersplat_camera_controls(
    server: viser.ViserServer,
    scene_bounds: dict | None = None,
) -> "SuperSplatCamera":
    """
    Create SuperSplat-style camera controller (UI created separately in ui.py).

    Parameters
    ----------
    server : viser.ViserServer
        Viser server instance
    scene_bounds : dict | None
        Scene bounds dictionary from SceneBoundsManager

    Returns
    -------
    SuperSplatCamera
        Camera controller instance
    """
    from src.gsplay.rendering.camera import SuperSplatCamera

    camera = SuperSplatCamera(server, scene_bounds)
    logger.info("SuperSplat camera controller initialized")
    return camera
