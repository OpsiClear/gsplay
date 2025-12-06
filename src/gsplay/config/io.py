"""
Configuration import/export functionality for viewer settings.

Exports and imports viewer configurations (pose, filter, rotation, color editing)
to/from YAML files per-dataset.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from dataclasses import asdict

if TYPE_CHECKING:
    from src.gsplay.config.settings import GSPlayConfig
    from src.gsplay.rendering.camera import SuperSplatCamera

logger = logging.getLogger(__name__)


def _flatten_tuples(obj: Any) -> Any:
    """
    Recursively flatten tuples into individual fields for YAML serialization.

    Converts:
    - sphere_center: (x, y, z) -> sphere_center_x, sphere_center_y, sphere_center_z
    - cuboid_center: (x, y, z) -> cuboid_center_x, cuboid_center_y, cuboid_center_z
    - cuboid_size: (x, y, z) -> cuboid_size_x, cuboid_size_y, cuboid_size_z
    - look_at: [x, y, z] -> look_at_x, look_at_y, look_at_z

    Parameters
    ----------
    obj : Any
        Object to flatten (dict, list, tuple, or primitive)

    Returns
    -------
    Any
        Object with tuples flattened into individual fields
    """
    if isinstance(obj, dict):
        result = {}
        for key, value in obj.items():
            if key in ("sphere_center", "cuboid_center") and isinstance(
                value, (tuple, list)
            ):
                # Flatten 3D tuple to x, y, z fields
                if len(value) == 3:
                    result[f"{key}_x"] = float(value[0])
                    result[f"{key}_y"] = float(value[1])
                    result[f"{key}_z"] = float(value[2])
                else:
                    result[key] = value
            elif key == "cuboid_size" and isinstance(value, (tuple, list)):
                # Flatten 3D tuple to x, y, z fields
                if len(value) == 3:
                    result["cuboid_size_x"] = float(value[0])
                    result["cuboid_size_y"] = float(value[1])
                    result["cuboid_size_z"] = float(value[2])
                else:
                    result[key] = value
            elif key == "look_at" and isinstance(value, (tuple, list)):
                # Flatten 3D array to x, y, z fields
                if len(value) == 3:
                    result["look_at_x"] = float(value[0])
                    result["look_at_y"] = float(value[1])
                    result["look_at_z"] = float(value[2])
                else:
                    result[key] = value
            elif isinstance(value, (dict, list, tuple)):
                result[key] = _flatten_tuples(value)
            else:
                result[key] = value
        return result
    elif isinstance(obj, (list, tuple)):
        return [_flatten_tuples(item) for item in obj]
    else:
        return obj


def _unflatten_tuples(obj: Any) -> Any:
    """
    Recursively unflatten individual fields back into tuples.

    Converts:
    - sphere_center_x, sphere_center_y, sphere_center_z -> sphere_center: (x, y, z)
    - cuboid_center_x, cuboid_center_y, cuboid_center_z -> cuboid_center: (x, y, z)
    - cuboid_size_x, cuboid_size_y, cuboid_size_z -> cuboid_size: (x, y, z)
    - look_at_x, look_at_y, look_at_z -> look_at: [x, y, z]

    Parameters
    ----------
    obj : Any
        Object to unflatten (dict, list, or primitive)

    Returns
    -------
    Any
        Object with flattened fields converted back to tuples
    """
    if isinstance(obj, dict):
        result = {}
        # Track which keys we've processed
        processed_keys = set()

        for key, value in obj.items():
            if key in processed_keys:
                continue

            # Check for flattened tuple fields (x, y, z pattern)
            if key.endswith("_x") and not key.startswith("cuboid_size_factor"):
                base_key = key[:-2]
                if base_key in ("sphere_center", "cuboid_center", "look_at"):
                    # Check if y and z exist
                    y_key = f"{base_key}_y"
                    z_key = f"{base_key}_z"
                    if y_key in obj and z_key in obj:
                        result[base_key] = (
                            float(obj[key]),
                            float(obj[y_key]),
                            float(obj[z_key]),
                        )
                        processed_keys.add(key)
                        processed_keys.add(y_key)
                        processed_keys.add(z_key)
                        continue
            elif key == "cuboid_size_x":
                # Handle cuboid_size separately (not cuboid_size_factor_x)
                if "cuboid_size_y" in obj and "cuboid_size_z" in obj:
                    result["cuboid_size"] = (
                        float(obj["cuboid_size_x"]),
                        float(obj["cuboid_size_y"]),
                        float(obj["cuboid_size_z"]),
                    )
                    processed_keys.add("cuboid_size_x")
                    processed_keys.add("cuboid_size_y")
                    processed_keys.add("cuboid_size_z")
                    continue

            # Process nested structures recursively
            if isinstance(value, dict):
                result[key] = _unflatten_tuples(value)
                processed_keys.add(key)
            elif isinstance(value, list):
                result[key] = [_unflatten_tuples(item) for item in value]
                processed_keys.add(key)
            else:
                result[key] = value
                processed_keys.add(key)

        return result
    elif isinstance(obj, list):
        return [_unflatten_tuples(item) for item in obj]
    else:
        return obj


def export_viewer_config(
    config: GSPlayConfig,
    camera_controller: SuperSplatCamera | None,
    output_path: Path | str,
    ui_handles=None,
) -> None:
    """
    Export viewer configuration to YAML file.

    Exports:
    - Camera pose (azimuth, elevation, roll, distance, look_at)
    - Scene transform (translation, rotation, scale)
    - Volume filter settings
    - Spatial filter settings
    - Color adjustments
    - Animation settings (auto_play, play_speed, current_frame)
    - Render settings (render_quality, jpeg_quality)
    - Processing mode

    Parameters
    ----------
    config : GSPlayConfig
        GSPlay configuration
    camera_controller : SuperSplatCamera | None
        Camera controller instance (for camera pose)
    output_path : Path | str
        Path to output YAML file
    ui_handles : UIHandles | None
        UI handles for reading current UI state
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    export_data = {
        "camera": {},
        "transform_values": {},
        "volume_filter": config.volume_filter.to_dict(),
        "color_values": {},
        "alpha_scaler": config.alpha_scaler,
        "animation": config.animation.to_dict(),
        "render_settings": config.render_settings.to_dict(),
        "processing_mode": config.processing_mode,
        "export_settings": config.export_settings.to_dict(),
    }

    # Export transform values (convert from gsmod format)
    translate = getattr(
        config.transform_values,
        "translate",
        getattr(config.transform_values, "translation", (0.0, 0.0, 0.0)),
    )
    rotate = getattr(
        config.transform_values,
        "rotate",
        getattr(config.transform_values, "rotation", (0.0, 0.0, 0.0, 1.0)),
    )
    scale_value = getattr(config.transform_values, "scale", 1.0)

    # Normalize quaternion to have w >= 0 for consistent Euler conversion on import
    # (q and -q represent the same rotation)
    rotate_list = rotate.tolist() if hasattr(rotate, "tolist") else list(rotate)
    # gsmod format: (x, y, z, w)
    if len(rotate_list) == 4 and rotate_list[3] < 0:
        rotate_list = [-v for v in rotate_list]

    export_data["transform_values"] = {
        "translate": translate.tolist() if hasattr(translate, "tolist") else list(translate),
        "rotate": rotate_list,
        "scale": float(scale_value)
        if isinstance(scale_value, (int, float))
        else scale_value.tolist(),
    }

    # Export color values
    export_data["color_values"] = asdict(config.color_values)
    # Export spatial filter values
    export_data["filter_values"] = asdict(config.filter_values)

    # Export camera pose if available
    if camera_controller is not None and camera_controller.state is not None:
        with camera_controller.state_lock:
            state = camera_controller.state
            export_data["camera"] = {
                "version": 2,  # Quaternion format version
                "orientation": state.orientation.tolist()
                if hasattr(state.orientation, "tolist")
                else list(state.orientation),
                "distance": float(state.distance),
                "look_at": state.look_at.tolist()
                if hasattr(state.look_at, "tolist")
                else list(state.look_at),
                # Also export Euler angles for human readability and backwards compat
                "azimuth": float(state.azimuth),
                "elevation": float(state.elevation),
                "roll": float(state.roll),
            }

    # Export UI-specific state if handles available
    if ui_handles is not None:
        ui_state = {}

        # Animation UI state
        if ui_handles.time_slider is not None:
            ui_state["current_frame"] = int(ui_handles.time_slider.value)
        if ui_handles.auto_play is not None:
            ui_state["auto_play"] = ui_handles.auto_play.value
        if ui_handles.play_speed is not None:
            ui_state["play_speed"] = float(ui_handles.play_speed.value)

        # Render quality UI state
        if ui_handles.render_quality is not None:
            ui_state["render_quality"] = float(ui_handles.render_quality.value)
        if ui_handles.jpeg_quality_slider is not None:
            ui_state["jpeg_quality"] = int(ui_handles.jpeg_quality_slider.value)
        if ui_handles.auto_quality_checkbox is not None:
            ui_state["auto_quality"] = bool(ui_handles.auto_quality_checkbox.value)

        # Filter UI state
        if ui_handles.spatial_filter_type is not None:
            ui_state["spatial_filter_type"] = ui_handles.spatial_filter_type.value
        if ui_handles.show_filter_viz is not None:
            ui_state["show_filter_viz"] = bool(ui_handles.show_filter_viz.value)
        if ui_handles.use_cpu_filtering_checkbox is not None:
            ui_state["use_cpu_filtering"] = bool(ui_handles.use_cpu_filtering_checkbox.value)

        if ui_state:
            export_data["ui_state"] = ui_state

    # Flatten tuples into individual fields for YAML compatibility
    export_data = _flatten_tuples(export_data)

    # Write YAML file
    with open(output_path, "w") as f:
        yaml.dump(export_data, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Exported viewer config to {output_path}")


def import_viewer_config(
    config: GSPlayConfig,
    camera_controller: SuperSplatCamera | None,
    input_path: Path | str,
    ui_handles=None,
) -> None:
    """
    Import viewer configuration from YAML file.

    Imports:
    - Camera pose (azimuth, elevation, roll, distance, look_at)
    - Scene transform (translation, rotation, scale)
    - Volume filter settings
    - Color adjustments
    - Animation settings (auto_play, play_speed, current_frame)
    - Render settings (render_quality, jpeg_quality)
    - Processing mode
    - UI state (spatial_filter_type, show_filter_viz, etc.)

    Parameters
    ----------
    config : GSPlayConfig
        GSPlay configuration to update
    camera_controller : SuperSplatCamera | None
        Camera controller instance (for camera pose)
    input_path : Path | str
        Path to input YAML file
    ui_handles : UIHandles | None
        UI handles for updating UI controls
    """
    input_path = Path(input_path)

    if not input_path.exists():
        logger.error(f"Config file not found: {input_path}")
        return

    # Try safe_load first (handles standard YAML)
    try:
        with open(input_path, "r") as f:
            import_data = yaml.safe_load(f)
    except yaml.constructor.ConstructorError:
        # Fallback: file may contain Python-specific tags (e.g., !!python/tuple)
        # Use unsafe loader
        logger.warning("Config file contains Python-specific tags, using unsafe loader")
        with open(input_path, "r") as f:
            import_data = yaml.load(f, Loader=yaml.Loader)

    if not import_data:
        logger.error(f"Empty or invalid config file: {input_path}")
        return

    # Unflatten flattened tuple fields back to tuples
    import_data = _unflatten_tuples(import_data)

    # Import camera pose
    if "camera" in import_data and camera_controller is not None:
        camera_data = import_data["camera"]
        logger.debug(f"Importing camera: {camera_data}")
        logger.debug(
            f"camera_controller.state={camera_controller.state is not None}, "
            f"scene_bounds={camera_controller.scene_bounds is not None}"
        )

        if camera_controller.state is None:
            logger.warning("camera_controller.state is None, cannot import camera pose")

        if camera_controller.scene_bounds is None:
            logger.warning("camera_controller.scene_bounds is None, camera may not be applied correctly")

        if camera_controller.state is not None:
            import numpy as np

            with camera_controller.state_lock:
                # Check version to determine import method
                version = camera_data.get("version", 1)

                if version >= 2 and "orientation" in camera_data:
                    # New quaternion format (version 2+)
                    orientation = camera_data["orientation"]
                    if isinstance(orientation, (list, tuple)):
                        camera_controller.state.orientation = np.array(
                            orientation, dtype=np.float64
                        )
                    logger.debug("Imported quaternion orientation from config")
                else:
                    # Legacy Euler angle format (version 1 or missing)
                    azimuth = float(camera_data.get("azimuth", 45.0))
                    elevation = float(camera_data.get("elevation", 30.0))
                    roll = float(camera_data.get("roll", 0.0))
                    # Convert Euler angles to quaternion
                    camera_controller.state.set_from_euler(azimuth, elevation, roll)
                    logger.debug("Converted legacy Euler angles to quaternion")

                # Import distance and look_at (common to both versions)
                if "distance" in camera_data:
                    camera_controller.state.distance = float(camera_data["distance"])
                if "look_at" in camera_data:
                    look_at = camera_data["look_at"]
                    if isinstance(look_at, (list, tuple)):
                        camera_controller.state.look_at = np.array(
                            look_at, dtype=np.float64
                        )

            # Log the state we're about to apply
            logger.debug(
                f"Applying camera state: azimuth={camera_controller.state.azimuth:.1f}, "
                f"elevation={camera_controller.state.elevation:.1f}, "
                f"roll={camera_controller.state.roll:.1f}, "
                f"distance={camera_controller.state.distance:.2f}"
            )

            # Apply camera state to viser camera
            camera_controller.apply_state_to_camera()

            # Update UI sliders to match imported camera state
            if ui_handles is not None:
                look_at = camera_controller.state.look_at
                ui_handles.set_camera_values(
                    azimuth=camera_controller.state.azimuth,
                    elevation=camera_controller.state.elevation,
                    roll=camera_controller.state.roll,
                    distance=camera_controller.state.distance,
                    scene_bounds=camera_controller.scene_bounds,
                    look_at=tuple(look_at) if look_at is not None else None,
                )

            logger.info("Imported camera pose from config")

    # Import transform values (with migration from old scene_transform)
    if "transform_values" in import_data:
        from gsmod import TransformValues
        import numpy as np

        tv_data = import_data["transform_values"]
        translate = np.array(tv_data.get("translate", [0.0, 0.0, 0.0]), dtype=np.float32)
        rotate = np.array(tv_data.get("rotate", [0.0, 0.0, 0.0, 1.0]), dtype=np.float32)
        scale_value = tv_data.get("scale", 1.0)
        try:
            config.transform_values = TransformValues(
                translate=translate,
                rotate=rotate,
                scale=scale_value,
            )
        except TypeError:
            config.transform_values = TransformValues(
                translation=translate,
                rotation=rotate,
                scale=scale_value,
            )

        # Update UI sliders to match imported transform values
        if ui_handles is not None:
            ui_handles.set_transform_values(config.transform_values)

        logger.info("Imported transform values from config")
    elif "scene_transform" in import_data:
        # Migration: Convert old SceneTransform to TransformValues
        from gsmod import TransformValues
        from gsmod.transform.api import euler_to_quaternion
        import numpy as np

        st_data = import_data["scene_transform"]
        # Convert Euler (degrees) to quaternion
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
            config.transform_values = TransformValues(
                translate=translate,
                rotate=quat,
                scale=scale_value,
            )
        except TypeError:
            config.transform_values = TransformValues(
                translation=translate,
                rotation=quat,
                scale=scale_value,
            )

        # Update UI sliders to match migrated transform values
        if ui_handles is not None:
            ui_handles.set_transform_values(config.transform_values)

        logger.info("Migrated old scene_transform to transform_values")

    # Import volume filter
    if "volume_filter" in import_data:
        from src.gsplay.config.settings import VolumeFilter

        filter_data = import_data["volume_filter"].copy()

        # Handle flattened fields (in case unflatten didn't catch them)
        for field in ["sphere_center", "cuboid_center"]:
            if f"{field}_x" in filter_data:
                filter_data[field] = (
                    float(filter_data.pop(f"{field}_x")),
                    float(filter_data.pop(f"{field}_y")),
                    float(filter_data.pop(f"{field}_z")),
                )

        if "cuboid_size_x" in filter_data:
            filter_data["cuboid_size"] = (
                float(filter_data.pop("cuboid_size_x")),
                float(filter_data.pop("cuboid_size_y")),
                float(filter_data.pop("cuboid_size_z")),
            )

        # Ensure tuple fields are tuples (handle lists from old YAML files)
        for field in ["sphere_center", "cuboid_center", "cuboid_size"]:
            if field in filter_data and isinstance(filter_data[field], list):
                filter_data[field] = tuple(filter_data[field])

        config.volume_filter = VolumeFilter(**filter_data)
        logger.info("Imported volume filter from config")

    # Import color values (with migration from old color_adjustments)
    if "color_values" in import_data:
        from gsmod import ColorValues

        cv_data = import_data["color_values"]
        config.color_values = ColorValues(**cv_data)
        logger.info("Imported color values from config")
    elif "color_adjustments" in import_data:
        # Migration: Convert old ColorAdjustments to ColorValues
        from gsmod import ColorValues

        ca_data = import_data["color_adjustments"]
        # Map UI ranges to gsmod ranges
        temp_ui = ca_data.get("temperature", 0.5)
        shadows_ui = ca_data.get("shadows", 1.0)
        highlights_ui = ca_data.get("highlights", 1.0)

        config.color_values = ColorValues(
            brightness=ca_data.get("brightness", 1.0),
            contrast=ca_data.get("contrast", 1.0),
            saturation=ca_data.get("saturation", 1.0),
            vibrance=ca_data.get("vibrance", 1.0),
            hue_shift=ca_data.get("hue_shift", 0.0),
            gamma=ca_data.get("gamma", 1.0),
            temperature=(temp_ui - 0.5) * 2.0,  # [0,1] → [-1,1]
            shadows=shadows_ui - 1.0,  # [0,2] → [-1,1]
            highlights=highlights_ui - 1.0,  # [0,2] → [-1,1]
        )
        config.alpha_scaler = ca_data.get("alpha_scaler", config.alpha_scaler)
        logger.info("Migrated old color_adjustments to color_values")

    if "alpha_scaler" in import_data:
        config.alpha_scaler = float(import_data["alpha_scaler"])

    # Import spatial filter values
    if "filter_values" in import_data:
        from gsmod import FilterValues
        fv_data = import_data["filter_values"] or {}

        # Normalize list fields to tuples where appropriate
        tuple_fields = [
            "sphere_center",
            "box_min",
            "box_max",
            "box_rot",
            "ellipsoid_center",
            "ellipsoid_radii",
            "ellipsoid_rot",
            "frustum_pos",
            "frustum_rot",
        ]
        for field in tuple_fields:
            if field in fv_data and isinstance(fv_data[field], list):
                fv_data[field] = tuple(fv_data[field])

        config.filter_values = FilterValues(**fv_data)
        logger.info("Imported spatial filter values from config")

    # Import animation settings
    if "animation" in import_data:
        from src.gsplay.config.settings import AnimationSettings

        anim_data = import_data["animation"]
        config.animation = AnimationSettings(
            auto_play=anim_data.get("auto_play", False),
            play_speed_fps=anim_data.get("play_speed_fps", 30.0),
            current_frame=anim_data.get("current_frame", 0),
            auto_rotate=anim_data.get("auto_rotate", "off"),
            rotation_speed_dps=anim_data.get("rotation_speed_dps", 30.0),
        )
        logger.info("Imported animation settings from config")

    # Import render settings
    if "render_settings" in import_data:
        from src.gsplay.config.settings import RenderSettings

        render_data = import_data["render_settings"]
        config.render_settings = RenderSettings(
            jpeg_quality_static=render_data.get("jpeg_quality_static", 90),
            jpeg_quality_move=render_data.get("jpeg_quality_move", 60),
        )
        logger.info("Imported render settings from config")

    # Import processing mode
    if "processing_mode" in import_data:
        config.processing_mode = import_data["processing_mode"]
        logger.info("Imported processing mode from config")

    # Import export settings
    if "export_settings" in import_data:
        from src.gsplay.config.settings import ExportSettings
        from src.infrastructure.io.path_io import UniversalPath

        es_data = import_data["export_settings"]
        # Convert export_path back to UniversalPath
        if "export_path" in es_data:
            es_data["export_path"] = UniversalPath(es_data["export_path"])
        config.export_settings = ExportSettings(
            export_path=es_data.get("export_path", config.export_settings.export_path),
            export_format=es_data.get("export_format", "compressed-ply"),
            export_device=es_data.get("export_device", "cpu"),
            start_frame=es_data.get("start_frame"),
            end_frame=es_data.get("end_frame"),
            video_fps=es_data.get("video_fps", 30.0),
            video_duration_sec=es_data.get("video_duration_sec", 10.0),
            video_width=es_data.get("video_width", 800),
            video_height=es_data.get("video_height", 600),
        )
        # Update UI handles for export settings
        if ui_handles is not None:
            if ui_handles.export_path is not None:
                ui_handles.export_path.value = str(config.export_settings.export_path)
            if ui_handles.export_format is not None:
                ui_handles.export_format.value = config.export_settings.export_format
            if ui_handles.export_device is not None:
                ui_handles.export_device.value = config.export_settings.export_device
        logger.info("Imported export settings from config")

    # Import UI-specific state and update UI handles
    if "ui_state" in import_data and ui_handles is not None:
        ui_state = import_data["ui_state"]

        # Animation UI state
        if "current_frame" in ui_state and ui_handles.time_slider is not None:
            ui_handles.time_slider.value = int(ui_state["current_frame"])
        if "auto_play" in ui_state and ui_handles.auto_play is not None:
            ui_handles.auto_play.value = ui_state["auto_play"]
        if "play_speed" in ui_state and ui_handles.play_speed is not None:
            ui_handles.play_speed.value = float(ui_state["play_speed"])

        # Render quality UI state
        if "render_quality" in ui_state and ui_handles.render_quality is not None:
            ui_handles.render_quality.value = float(ui_state["render_quality"])
        if "jpeg_quality" in ui_state and ui_handles.jpeg_quality_slider is not None:
            ui_handles.jpeg_quality_slider.value = int(ui_state["jpeg_quality"])
        if "auto_quality" in ui_state and ui_handles.auto_quality_checkbox is not None:
            ui_handles.auto_quality_checkbox.value = bool(ui_state["auto_quality"])

        # Filter UI state
        if "spatial_filter_type" in ui_state and ui_handles.spatial_filter_type is not None:
            ui_handles.spatial_filter_type.value = ui_state["spatial_filter_type"]
        if "show_filter_viz" in ui_state and ui_handles.show_filter_viz is not None:
            ui_handles.show_filter_viz.value = bool(ui_state["show_filter_viz"])
        if "use_cpu_filtering" in ui_state and ui_handles.use_cpu_filtering_checkbox is not None:
            ui_handles.use_cpu_filtering_checkbox.value = bool(ui_state["use_cpu_filtering"])

        logger.info("Imported UI state from config")

    logger.info(f"Imported viewer config from {input_path}")
