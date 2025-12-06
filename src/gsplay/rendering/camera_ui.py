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
    updating_from_camera = [False]  # Use list for mutable closure

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
        if updating_from_camera[0]:
            return
        camera.stop_auto_rotation()
        if camera.state is not None and camera.scene_bounds:
            extent = camera.scene_bounds.get("max_size", 10.0)
            default_distance = extent * 2.5
            zoom_multiplier = 2.0**zoom_slider.value
            new_distance = default_distance * zoom_multiplier
            with camera.state_lock:
                camera.state.distance = new_distance
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
        min=-180.0,
        max=180.0,
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
        if updating_from_camera[0]:
            return
        camera.stop_auto_rotation()
        if camera.state is not None:
            with camera.state_lock:
                # Use set_from_euler to update quaternion orientation
                camera.state.set_from_euler(
                    azimuth_slider.value,
                    camera.state.elevation,
                    camera.state.roll,
                )
            camera.apply_state_to_camera()

    @elevation_slider.on_update
    def _(_) -> None:
        if updating_from_camera[0]:
            return
        camera.stop_auto_rotation()
        if camera.state is not None:
            with camera.state_lock:
                # Use set_from_euler to update quaternion orientation
                camera.state.set_from_euler(
                    camera.state.azimuth,
                    elevation_slider.value,
                    camera.state.roll,
                )
            camera.apply_state_to_camera()

    @roll_slider.on_update
    def _(_) -> None:
        if updating_from_camera[0]:
            return
        camera.stop_auto_rotation()
        if camera.state is not None:
            with camera.state_lock:
                # Use set_from_euler to update quaternion orientation
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
        if updating_from_camera[0]:
            return
        if camera.state is not None:
            with camera.state_lock:
                camera.state.look_at[0] = look_at_x_slider.value
            camera.apply_state_to_camera()

    @look_at_y_slider.on_update
    def _(_) -> None:
        if updating_from_camera[0]:
            return
        if camera.state is not None:
            with camera.state_lock:
                camera.state.look_at[1] = look_at_y_slider.value
            camera.apply_state_to_camera()

    @look_at_z_slider.on_update
    def _(_) -> None:
        if updating_from_camera[0]:
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
        initial_value=30.0,
        hint="Rotation speed in degrees per second",
    )

    @rotation_speed_slider.on_update
    def _(_) -> None:
        camera._rotation_speed = rotation_speed_slider.value

    rotate_buttons = server.gui.add_button_group(
        "Rotate",
        (" CW ", "Stop", "CCW "),
    )

    @rotate_buttons.on_click
    def _(_) -> None:
        action = rotate_buttons.value.strip()
        if action == "CW":
            camera.start_auto_rotation(axis="y", speed=rotation_speed_slider.value)
        elif action == "CCW":
            camera.start_auto_rotation(axis="y", speed=-rotation_speed_slider.value)
        elif action == "Stop":
            camera.stop_auto_rotation()

    # Create sync function that can be called after all controls are set up
    def setup_camera_sync():
        """Set up camera sync callbacks."""

        def sync_sliders_with_camera():
            """Sync UI sliders with camera state (quaternion-based)."""
            if camera.state is None:
                return

            try:
                # Read Euler angles from quaternion state properties
                with camera.state_lock:
                    azimuth = camera.state.azimuth
                    elevation = camera.state.elevation
                    roll = camera.state.roll
                    distance = camera.state.distance
                    look_at = camera.state.look_at.copy()

                # Update UI sliders (prevent circular updates)
                updating_from_camera[0] = True

                # Clamp elevation to slider range (-180 to 180)
                # With quaternions, elevation is already in -90 to 90 range
                azimuth_slider.value = azimuth % 360.0
                elevation_slider.value = np.clip(elevation, -180.0, 180.0)
                roll_slider.value = np.clip(roll, -180.0, 180.0)

                # Sync look_at (camera target) sliders
                look_at_x_slider.value = np.clip(float(look_at[0]), -50.0, 50.0)
                look_at_y_slider.value = np.clip(float(look_at[1]), -50.0, 50.0)
                look_at_z_slider.value = np.clip(float(look_at[2]), -50.0, 50.0)

                if camera.scene_bounds:
                    extent = camera.scene_bounds.get("max_size", 10.0)
                    default_distance = extent * 2.5
                    if default_distance > 0 and distance > 0:
                        actual_zoom = distance / default_distance
                        zoom_slider.value = np.clip(np.log2(actual_zoom), -8.0, 3.0)

                updating_from_camera[0] = False

            except Exception as e:
                logger.debug(f"Error syncing camera sliders: {e}")
                updating_from_camera[0] = False

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
                    except Exception:
                        pass  # Client may have disconnected

                threading.Thread(target=apply_state_after_init, daemon=True).start()

            @client.camera.on_update
            def _(_) -> None:
                # Sync state from user interactions (ignored until client initialized)
                camera.sync_state_from_client(client)
                sync_sliders_with_camera()

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


def create_playback_controls(server: viser.ViserServer, config=None):
    """
    Create Frame slider and Play/Pause toggle button for animation playback.

    Parameters
    ----------
    server : viser.ViserServer
        Viser server instance
    config : GSPlayConfig | None
        GSPlay configuration for animation settings

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

    # Frame slider
    time_slider = server.gui.add_slider(
        "Frame",
        min=0,
        max=1,
        step=1,
        initial_value=initial_frame,
        hint="Time frame for 4D content",
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
