"""
Configuration dataclasses for the Universal 4D Gaussian Splatting GSPlay.

This module provides type-safe configuration using Python 3.10+ dataclasses.
Supports local filesystem and cloud storage paths.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field

import numpy as np
from gsmod import ColorValues, FilterValues, TransformValues
from gsmod.transform.api import euler_to_quaternion

from src.domain.filters import VolumeFilter  # Re-exported from domain

# Re-export UIHandles from its new home for backward compatibility
from src.gsplay.config.ui_handles import UIHandles
from src.infrastructure.io.path_io import UniversalPath


logger = logging.getLogger(__name__)


# Re-export for backward compatibility
__all__ = ["AnimationSettings", "ExportSettings", "GSPlayConfig", "UIHandles", "VolumeFilter"]


@dataclass
class PlyLoadingConfig:
    """Configuration for PLY file loading and preprocessing."""

    enable_concurrent_prefetch: bool = True

    # Quality filtering thresholds (user-configurable)
    opacity_threshold: float = 0.01  # Filter Gaussians with opacity < threshold
    scale_threshold: float = 1e-7  # Filter Gaussians with scale < threshold
    enable_quality_filtering: bool = True  # Enable/disable quality filtering

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for easy serialization."""
        return asdict(self)


@dataclass
class ExportSettings:
    """Export configuration for sequence and video export.

    Supports local filesystem and cloud storage paths.

    Attributes
    ----------
    export_scope : ExportScope
        Scope of frames to export (current frame, all keyframes, or time range)
    source_time_start : float | None
        Start time in source units (for TIME_RANGE scope)
    source_time_end : float | None
        End time in source units (for TIME_RANGE scope)
    source_time_step : float | None
        Step size in source units (for TIME_RANGE scope)
    """

    # Sequence export
    export_path: UniversalPath = field(default_factory=lambda: UniversalPath("./export_with_edits"))
    export_format: str = "compressed-ply"  # Export format: "compressed-ply", "ply"
    export_device: str = (
        "cpu"  # Export processing device: "cpu" or "cuda" (or "cuda:0", "cuda:1", etc.)
    )
    start_frame: int | None = None  # Starting frame (None = first frame)
    end_frame: int | None = None  # Ending frame (None = last frame)
    exporter_options: dict = field(default_factory=dict)  # Format-specific options

    # Continuous time export support
    export_scope: str = "all_keyframes"  # "current_frame", "all_keyframes", "time_range"
    source_time_start: float | None = None  # Start time in source units
    source_time_end: float | None = None  # End time in source units
    source_time_step: float | None = None  # Step size in source units

    # Aliases for backward compatibility
    @property
    def output_dir(self) -> UniversalPath:
        """Alias for export_path."""
        return self.export_path

    @property
    def format(self) -> str:
        """Alias for export_format."""
        return self.export_format

    # Video export
    video_fps: float = 30.0
    video_duration_sec: float = 10.0
    video_width: int = 800
    video_height: int = 600

    def __post_init__(self):
        """Validate settings after initialization."""
        if self.export_scope == "time_range":
            if self.source_time_step is not None and self.source_time_step <= 0:
                raise ValueError(f"source_time_step must be positive, got {self.source_time_step}")

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary (convert UniversalPath to str)."""
        data = asdict(self)
        data["export_path"] = str(self.export_path)
        return data


@dataclass
class RenderSettings:
    """Rendering quality and performance settings."""

    # JPEG quality for frontend image streaming (1-100)
    jpeg_quality_static: int = 90  # Quality when camera is static/UI updates
    jpeg_quality_move: int = 60  # Quality during camera movement (lower for performance)

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for easy serialization."""
        return asdict(self)


@dataclass
class AnimationSettings:
    """Animation and playback settings."""

    auto_play: bool = False
    play_speed_fps: float = 30.0  # Playback FPS
    current_frame: int = 0  # Current time frame

    # Auto-rotation settings
    auto_rotate: str = "off"  # "off", "cw", "ccw"
    rotation_speed_dps: float = 30.0  # Degrees per second

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for easy serialization."""
        return asdict(self)


@dataclass
class GSPlayConfig:
    """Main viewer configuration.

    Supports local filesystem and cloud storage paths.
    """

    # Network
    port: int = 6019
    host: str = "0.0.0.0"  # Bind to all interfaces for external access
    stream_port: int = -1  # WebSocket stream port (-1 = auto on viser_port+1, 0 = disabled)

    # Device
    device: str = "cuda"  # Will be auto-detected if cuda unavailable

    # Model configuration
    model_config_path: UniversalPath | None = None

    # Output paths
    output_dir: UniversalPath = field(default_factory=lambda: UniversalPath("./nerfview_output"))

    # Sub-configurations
    color_values: ColorValues = field(default_factory=ColorValues)
    transform_values: TransformValues = field(default_factory=TransformValues)
    filter_values: FilterValues = field(default_factory=FilterValues)

    # Opacity multiplier (not in gsmod ColorValues)
    alpha_scaler: float = 1.0

    # UI-specific state
    volume_filter: VolumeFilter = field(default_factory=VolumeFilter)

    export_settings: ExportSettings = field(default_factory=ExportSettings)
    animation: AnimationSettings = field(default_factory=AnimationSettings)
    ply_loading: PlyLoadingConfig = field(default_factory=PlyLoadingConfig)
    render_settings: RenderSettings = field(default_factory=RenderSettings)

    # Edit history (tracks per-Gaussian edits)
    edit_history: dict[int, dict[str, object]] = field(default_factory=dict)
    edits_active: bool = False

    # Processing mode
    processing_mode: str = "all_gpu"

    # View-only mode (hides input path, config save, and export options)
    view_only: bool = False

    # Compact UI mode (mobile-friendly with smaller control panel)
    compact_ui: bool = False

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for serialization."""
        data = {
            "port": self.port,
            "host": self.host,
            "stream_port": self.stream_port,
            "device": self.device,
            "model_config_path": (str(self.model_config_path) if self.model_config_path else None),
            "output_dir": str(self.output_dir),
            "color_values": asdict(self.color_values),
            "transform_values": asdict(self.transform_values),
            "filter_values": asdict(self.filter_values),
            "alpha_scaler": self.alpha_scaler,
            "volume_filter": self.volume_filter.to_dict(),
            "export_settings": self.export_settings.to_dict(),
            "animation": self.animation.to_dict(),
            "ply_loading": self.ply_loading.to_dict(),
            "render_settings": self.render_settings.to_dict(),
            "edits_active": self.edits_active,
            "processing_mode": self.processing_mode,
            "view_only": self.view_only,
            "compact_ui": self.compact_ui,
        }
        return data

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> GSPlayConfig:
        """Create GSPlayConfig from dictionary (supports cloud storage paths)."""
        # Convert string paths back to UniversalPath objects
        if data.get("model_config_path"):
            data["model_config_path"] = UniversalPath(data["model_config_path"])
        if "output_dir" in data:
            data["output_dir"] = UniversalPath(data["output_dir"])

        # Recursively create nested dataclasses
        if "color_values" in data:
            data["color_values"] = ColorValues(**data["color_values"])
        if "transform_values" in data:
            tv_data = data["transform_values"] or {}
            try:
                data["transform_values"] = TransformValues(**tv_data)
            except TypeError:
                tv_kwargs = tv_data.copy()
                if "translate" in tv_kwargs and "translation" not in tv_kwargs:
                    tv_kwargs["translation"] = tv_kwargs.pop("translate")
                if "rotate" in tv_kwargs and "rotation" not in tv_kwargs:
                    tv_kwargs["rotation"] = tv_kwargs.pop("rotate")
                data["transform_values"] = TransformValues(**tv_kwargs)
        if "filter_values" in data:
            data["filter_values"] = FilterValues(**data["filter_values"])
        if "volume_filter" in data:
            data["volume_filter"] = VolumeFilter(**data["volume_filter"])
        if "export_settings" in data:
            if "export_path" in data["export_settings"]:
                data["export_settings"]["export_path"] = UniversalPath(
                    data["export_settings"]["export_path"]
                )
            data["export_settings"] = ExportSettings(**data["export_settings"])
        if "animation" in data:
            data["animation"] = AnimationSettings(**data["animation"])
        if "ply_loading" in data:
            ply_data = dict(data["ply_loading"] or {})
            ply_data.pop("enable_cache", None)
            data["ply_loading"] = PlyLoadingConfig(**ply_data)
        if "render_settings" in data:
            data["render_settings"] = RenderSettings(**data["render_settings"])

        # Default alpha scaler if missing
        if "alpha_scaler" not in data:
            data["alpha_scaler"] = 1.0

        # Handle legacy config migration
        if "color_adjustments" in data and "color_values" not in data:
            ca_data = data["color_adjustments"] or {}
            temp_ui = ca_data.get("temperature", 0.5)
            shadows_ui = ca_data.get("shadows", 1.0)
            highlights_ui = ca_data.get("highlights", 1.0)

            data["color_values"] = ColorValues(
                brightness=ca_data.get("brightness", 1.0),
                contrast=ca_data.get("contrast", 1.0),
                saturation=ca_data.get("saturation", 1.0),
                vibrance=ca_data.get("vibrance", 1.0),
                hue_shift=ca_data.get("hue_shift", 0.0),
                gamma=ca_data.get("gamma", 1.0),
                temperature=(temp_ui - 0.5) * 2.0,
                shadows=shadows_ui - 1.0,
                highlights=highlights_ui - 1.0,
            )
            if "alpha_scaler" not in data:
                data["alpha_scaler"] = ca_data.get("alpha_scaler", 1.0)

        if "scene_transform" in data and "transform_values" not in data:
            st_data = data["scene_transform"] or {}
            euler_deg = np.array(
                [
                    st_data.get("rotation_x", 0.0),
                    st_data.get("rotation_y", 0.0),
                    st_data.get("rotation_z", 0.0),
                ],
                dtype=np.float32,
            )
            euler_rad = np.radians(euler_deg)
            quat = euler_to_quaternion(euler_rad)
            translate = np.array(
                [
                    st_data.get("translation_x", 0.0),
                    st_data.get("translation_y", 0.0),
                    st_data.get("translation_z", 0.0),
                ],
                dtype=np.float32,
            )
            scale_value = st_data.get("global_scale", 1.0)
            try:
                data["transform_values"] = TransformValues(
                    translate=translate,
                    scale=scale_value,
                    rotate=quat,
                )
            except TypeError:
                data["transform_values"] = TransformValues(
                    translation=translate,
                    scale=scale_value,
                    rotation=quat,
                )

        return cls(**data)
