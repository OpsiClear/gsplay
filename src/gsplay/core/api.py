"""
Clean programmatic API for viewer control.

Provides a type-safe, validated wrapper around viser UI controls.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from gsmod import ColorValues, TransformValues

from src.gsplay.config.settings import VolumeFilter


if TYPE_CHECKING:
    from src.gsplay.core.app import UniversalGSPlay


logger = logging.getLogger(__name__)


@dataclass
class GSPlayState:
    """Snapshot of current viewer state."""

    # Playback state
    current_frame: int
    total_frames: int
    is_playing: bool
    playback_fps: float

    color_values: ColorValues
    alpha_scaler: float
    transform_values: TransformValues
    volume_filter: VolumeFilter
    render_quality: int

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "playback": {
                "current_frame": self.current_frame,
                "total_frames": self.total_frames,
                "is_playing": self.is_playing,
                "playback_fps": self.playback_fps,
            },
            "color_values": asdict(self.color_values),
            "alpha_scaler": self.alpha_scaler,
            "transform_values": asdict(self.transform_values),
            "volume_filter": self.volume_filter.to_dict(),
            "render_quality": self.render_quality,
        }


class GSPlayAPI:
    """
    Clean programmatic interface to viewer controls.

    This wrapper provides a type-safe, validated API for controlling
    the viewer programmatically, decoupled from viser implementation details.
    """

    def __init__(self, viewer: UniversalGSPlay):
        """
        Initialize API wrapper.

        Parameters
        ----------
        viewer : UniversalGSPlay
            The viewer instance to control
        """
        self._viewer = viewer

    # --- Frame Control ---

    def seek_to_frame(self, frame: int) -> None:
        """
        Seek to a specific frame.

        Parameters
        ----------
        frame : int
            Frame index (0-based)

        Raises
        ------
        ValueError
            If frame is out of range
        RuntimeError
            If no model is loaded
        """
        if self._viewer.model is None:
            raise RuntimeError("No model loaded")

        total_frames = self._viewer.model.get_total_frames()
        if not 0 <= frame < total_frames:
            raise ValueError(f"Frame {frame} out of range [0, {total_frames})")

        if self._viewer.ui and self._viewer.ui.time_slider:
            self._viewer.ui.time_slider.value = frame
            logger.debug(f"Seeked to frame {frame}")

    def get_current_frame(self) -> int:
        """
        Get current frame index.

        Returns
        -------
        int
            Current frame index (0-based)
        """
        if self._viewer.ui and self._viewer.ui.time_slider:
            return int(self._viewer.ui.time_slider.value)
        return 0

    def play(self) -> None:
        """Start playback."""
        if self._viewer.playback_controller:
            self._viewer.playback_controller.play()
            logger.debug("Playback started via API")

    def pause(self) -> None:
        """Pause playback."""
        if self._viewer.playback_controller:
            self._viewer.playback_controller.pause()
            logger.debug("Playback paused via API")

    def is_playing(self) -> bool:
        """
        Check if playback is active.

        Returns
        -------
        bool
            True if playing, False if paused
        """
        return self._viewer.config.animation.auto_play

    def set_playback_speed(self, fps: float) -> None:
        """
        Set playback speed.

        Parameters
        ----------
        fps : float
            Frames per second (1-120)

        Raises
        ------
        ValueError
            If fps is out of range
        """
        if not 1 <= fps <= 120:
            raise ValueError("FPS must be in range [1, 120]")

        if self._viewer.ui and self._viewer.ui.play_speed:
            self._viewer.ui.play_speed.value = fps
            logger.debug(f"Playback speed set to {fps} FPS")

    # --- Color Adjustments ---

    def set_temperature(self, value: float) -> None:
        """
        Set color temperature.

        Parameters
        ----------
        value : float
            Temperature adjustment in gsmod range [-1.0, 1.0]

        Raises
        ------
        ValueError
            If value is out of range
        """
        if not -1.0 <= value <= 1.0:
            raise ValueError("Temperature must be in range [-1.0, 1.0]")

        if self._viewer.ui and self._viewer.ui.temperature_slider:
            self._viewer.ui.temperature_slider.value = (value + 1.0) / 2.0
            logger.debug(f"Temperature set to {value} (ui={(value + 1.0) / 2.0})")
        self._viewer.config.color_values.temperature = value

    def set_brightness(self, value: float) -> None:
        """
        Set brightness.

        Parameters
        ----------
        value : float
            Brightness multiplier (0.0 to 5.0)

        Raises
        ------
        ValueError
            If value is out of range
        """
        if not 0.0 <= value <= 5.0:
            raise ValueError("Brightness must be in range [0.0, 5.0]")

        if self._viewer.ui and self._viewer.ui.brightness_slider:
            self._viewer.ui.brightness_slider.value = value
            logger.debug(f"Brightness set to {value}")
        self._viewer.config.color_values.brightness = value

    def set_contrast(self, value: float) -> None:
        """
        Set contrast.

        Parameters
        ----------
        value : float
            Contrast multiplier (0.0 to 5.0)

        Raises
        ------
        ValueError
            If value is out of range
        """
        if not 0.0 <= value <= 5.0:
            raise ValueError("Contrast must be in range [0.0, 5.0]")

        if self._viewer.ui and self._viewer.ui.contrast_slider:
            self._viewer.ui.contrast_slider.value = value
            logger.debug(f"Contrast set to {value}")
        self._viewer.config.color_values.contrast = value

    def set_saturation(self, value: float) -> None:
        """
        Set saturation.

        Parameters
        ----------
        value : float
            Saturation multiplier (0.0 to 5.0)

        Raises
        ------
        ValueError
            If value is out of range
        """
        if not 0.0 <= value <= 5.0:
            raise ValueError("Saturation must be in range [0.0, 5.0]")

        if self._viewer.ui and self._viewer.ui.saturation_slider:
            self._viewer.ui.saturation_slider.value = value
            logger.debug(f"Saturation set to {value}")
        self._viewer.config.color_values.saturation = value

    def set_color_values(
        self,
        values: ColorValues,
        alpha_scaler: float | None = None,
    ) -> None:
        """
        Set multiple color adjustments at once using gsmod ColorValues.

        Parameters
        ----------
        values : ColorValues
            Color adjustments in gsmod ranges
        alpha_scaler : float | None
            Optional opacity multiplier (handled separately from ColorValues)
        """
        self._viewer.config.color_values = values
        if alpha_scaler is not None:
            self._viewer.config.alpha_scaler = alpha_scaler

        if self._viewer.ui:
            self._viewer.ui.set_color_values(
                values,
                alpha_scaler=self._viewer.config.alpha_scaler,
            )
        logger.debug("Color values applied via API")

    def set_color_adjustments(self, values: ColorValues, alpha_scaler: float | None = None) -> None:
        """Alias for set_color_values for backward compatibility."""
        self.set_color_values(values, alpha_scaler)

    def reset_color_adjustments(self) -> None:
        """Reset all color adjustments to defaults."""
        self._viewer._handle_color_reset()
        logger.debug("Color adjustments reset")

    # --- Scene Transform ---

    def set_translation(self, x: float, y: float, z: float) -> None:
        """
        Set scene translation.

        Parameters
        ----------
        x : float
            X translation
        y : float
            Y translation
        z : float
            Z translation
        """
        if self._viewer.ui:
            if self._viewer.ui.translation_x_slider:
                self._viewer.ui.translation_x_slider.value = x
            if self._viewer.ui.translation_y_slider:
                self._viewer.ui.translation_y_slider.value = y
            if self._viewer.ui.translation_z_slider:
                self._viewer.ui.translation_z_slider.value = z
            logger.debug(f"Translation set to ({x}, {y}, {z})")

    def set_rotation(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
        """
        Set scene rotation using world-axis angles (truly gimbal-lock free).

        Each parameter represents rotation around that world axis in degrees.
        Rotations are composed via quaternion multiplication internally.

        Parameters
        ----------
        x : float
            Rotation around world X axis (degrees, -180 to 180)
        y : float
            Rotation around world Y axis (degrees, -180 to 180)
        z : float
            Rotation around world Z axis (degrees, -180 to 180)
        """
        if self._viewer.ui:
            if self._viewer.ui.rotate_x_slider:
                self._viewer.ui.rotate_x_slider.value = x
            if self._viewer.ui.rotate_y_slider:
                self._viewer.ui.rotate_y_slider.value = y
            if self._viewer.ui.rotate_z_slider:
                self._viewer.ui.rotate_z_slider.value = z
            logger.debug(f"Rotation set to x={x}, y={y}, z={z}")

    def set_scale(
        self,
        scale: float | tuple[float, float, float],
    ) -> None:
        """
        Set scene scale (uniform or per-axis).

        Parameters
        ----------
        scale : float or tuple[float, float, float]
            Uniform scale factor (float) or per-axis scale (sx, sy, sz).
            Effective scale range: [0.05, 10.0] (main [0.1, 5.0] × rel [0.5, 2.0]).

        Raises
        ------
        ValueError
            If scale values are out of valid range
        """
        import numpy as np

        # Slider bounds
        REL_MIN, REL_MAX = 0.5, 2.0
        MAIN_MIN, MAIN_MAX = 0.1, 5.0
        # Effective range: [MAIN_MIN * REL_MIN, MAIN_MAX * REL_MAX] = [0.05, 10.0]
        EFF_MIN, EFF_MAX = MAIN_MIN * REL_MIN, MAIN_MAX * REL_MAX

        if isinstance(scale, (int, float)):
            # Uniform scale
            if not EFF_MIN <= scale <= EFF_MAX:
                raise ValueError(f"Scale must be in range [{EFF_MIN}, {EFF_MAX}]")
            scale_f = float(scale)
            if MAIN_MIN <= scale_f <= MAIN_MAX:
                # Fits in main slider range - use directly
                main_scale = scale_f
                rel_x, rel_y, rel_z = 1.0, 1.0, 1.0
            else:
                # Outside main range - use main at limit with relative compensation
                if scale_f < MAIN_MIN:
                    main_scale = MAIN_MIN
                    rel_val = scale_f / MAIN_MIN  # Will be < 1.0
                else:
                    main_scale = MAIN_MAX
                    rel_val = scale_f / MAIN_MAX  # Will be > 1.0
                rel_val = max(REL_MIN, min(REL_MAX, rel_val))
                rel_x, rel_y, rel_z = rel_val, rel_val, rel_val
        else:
            # Per-axis scale - decompose into main + relative
            sx, sy, sz = float(scale[0]), float(scale[1]), float(scale[2])
            for v in (sx, sy, sz):
                if not EFF_MIN <= v <= EFF_MAX:
                    raise ValueError(f"Scale values must be in range [{EFF_MIN}, {EFF_MAX}]")

            scales = np.array([sx, sy, sz])
            if np.allclose(scales, scales[0], rtol=1e-5, atol=1e-8):
                # Uniform
                main_scale = max(MAIN_MIN, min(MAIN_MAX, sx))
                rel_x, rel_y, rel_z = 1.0, 1.0, 1.0
            else:
                # Non-uniform - find optimal main_scale
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

        if self._viewer.ui:
            if self._viewer.ui.scale_slider:
                self._viewer.ui.scale_slider.value = main_scale
            if self._viewer.ui.scale_x_slider:
                self._viewer.ui.scale_x_slider.value = rel_x
            if self._viewer.ui.scale_y_slider:
                self._viewer.ui.scale_y_slider.value = rel_y
            if self._viewer.ui.scale_z_slider:
                self._viewer.ui.scale_z_slider.value = rel_z
            logger.debug(f"Scale set to main={main_scale}, rel=({rel_x}, {rel_y}, {rel_z})")

    def reset_transform(self) -> None:
        """Reset scene transform to defaults."""
        self._viewer._handle_pose_reset()
        logger.debug("Transform reset")

    # --- Export ---

    def export_frames(
        self,
        format: str = "compressed-ply",
        output_path: Path | str | None = None,
    ) -> None:
        """
        Export frame sequence with current edits applied.

        Parameters
        ----------
        format : str
            Export format ("compressed-ply" or "ply")
        output_path : Path | str | None
            Output directory (uses config default if None)

        Raises
        ------
        ValueError
            If format is invalid
        """
        if format not in ["compressed-ply", "ply"]:
            raise ValueError(f"Invalid format: {format}")

        # Update export settings temporarily
        if self._viewer.ui:
            if self._viewer.ui.export_format:
                self._viewer.ui.export_format.value = format.title().replace("-", " ")
            if output_path and self._viewer.ui.export_path:
                self._viewer.ui.export_path.value = str(output_path)

        # Trigger export (always with edits)
        self._viewer._handle_export_ply()
        logger.info(f"Export triggered: format={format}")

    # --- State Queries ---

    def get_state(self) -> GSPlayState:
        """
        Get current viewer state snapshot.

        Returns
        -------
        GSPlayState
            Current viewer state
        """
        ui = self._viewer.ui
        if not ui:
            raise RuntimeError("UI not initialized")

        config = self._viewer.config

        return GSPlayState(
            current_frame=int(ui.time_slider.value) if ui.time_slider else 0,
            total_frames=self._viewer.model.get_total_frames() if self._viewer.model else 0,
            is_playing=ui.auto_play.value.strip() == "Play" if ui.auto_play else False,
            playback_fps=float(ui.play_speed.value)
            if ui.play_speed
            else config.animation.play_speed_fps,
            color_values=config.color_values,
            alpha_scaler=config.alpha_scaler,
            transform_values=config.transform_values,
            volume_filter=config.volume_filter,
            render_quality=int(ui.render_quality.value)
            if ui.render_quality
            else config.render_settings.jpeg_quality_static,
        )

    def set_render_quality(self, quality: int) -> None:
        """
        Set render quality (resolution).

        Parameters
        ----------
        quality : int
            Render resolution in pixels (400-2000)

        Raises
        ------
        ValueError
            If quality is out of range
        """
        if not 400 <= quality <= 2000:
            raise ValueError("Quality must be in range [400, 2000]")

        if self._viewer.ui and self._viewer.ui.render_quality:
            self._viewer.ui.render_quality.value = quality
            logger.debug(f"Render quality set to {quality}")

    # --- Auto-Rotation (Camera View) ---

    def rotate_cw(self, speed_dps: float = 30.0) -> None:
        """
        Start clockwise camera rotation.

        Parameters
        ----------
        speed_dps : float
            Rotation speed in degrees per second (default 30)
        """
        if not self._viewer.camera_controller:
            logger.error("No camera controller available")
            return

        self._viewer.camera_controller.start_auto_rotation(axis="y", speed=speed_dps)
        logger.debug(f"Started CW camera rotation at {speed_dps} deg/sec")

    def rotate_ccw(self, speed_dps: float = 30.0) -> None:
        """
        Start counter-clockwise camera rotation.

        Parameters
        ----------
        speed_dps : float
            Rotation speed in degrees per second (default 30)
        """
        if not self._viewer.camera_controller:
            logger.error("No camera controller available")
            return

        self._viewer.camera_controller.start_auto_rotation(axis="y", speed=-speed_dps)
        logger.debug(f"Started CCW camera rotation at {speed_dps} deg/sec")

    def stop_rotation(self) -> None:
        """Stop camera auto-rotation."""
        if not self._viewer.camera_controller:
            logger.error("No camera controller available")
            return

        self._viewer.camera_controller.stop_auto_rotation()
        logger.debug("Stopped camera rotation")

    def is_rotating(self) -> bool:
        """
        Check if camera auto-rotation is active.

        Returns
        -------
        bool
            True if rotating, False otherwise
        """
        if self._viewer.camera_controller:
            return self._viewer.camera_controller._rotation_active
        return False

    def get_rotation_speed(self) -> float:
        """
        Get current rotation speed.

        Returns
        -------
        float
            Rotation speed in degrees per second
        """
        if self._viewer.camera_controller:
            return abs(self._viewer.camera_controller._rotation_speed)
        return 0.0
