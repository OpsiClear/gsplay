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
) -> None:
    """
    Export viewer configuration to YAML file.

    Exports:
    - Camera pose (azimuth, elevation, roll, distance, look_at)
    - Scene transform (translation, rotation, scale)
    - Volume filter settings
    - Spatial filter settings
    - Color adjustments

    Parameters
    ----------
    config : GSPlayConfig
        GSPlay configuration
    camera_controller : SuperSplatCamera | None
        Camera controller instance (for camera pose)
    output_path : Path | str
        Path to output YAML file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    export_data = {
        "camera": {},
        "transform_values": {},
        "volume_filter": config.volume_filter.to_dict(),
        "color_values": {},
        "alpha_scaler": config.alpha_scaler,
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

    export_data["transform_values"] = {
        "translate": translate.tolist() if hasattr(translate, "tolist") else list(translate),
        "rotate": rotate.tolist() if hasattr(rotate, "tolist") else list(rotate),
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
                "azimuth": float(state.azimuth),
                "elevation": float(state.elevation),
                "roll": float(state.roll),
                "distance": float(state.distance),
                "look_at": state.look_at.tolist()
                if hasattr(state.look_at, "tolist")
                else list(state.look_at),
            }

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
) -> None:
    """
    Import viewer configuration from YAML file.

    Imports:
    - Camera pose (azimuth, elevation, roll, distance, look_at)
    - Scene transform (translation, rotation, scale)
    - Volume filter settings
    - Color adjustments

    Parameters
    ----------
    config : GSPlayConfig
        GSPlay configuration to update
    camera_controller : SuperSplatCamera | None
        Camera controller instance (for camera pose)
    input_path : Path | str
        Path to input YAML file
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
        if camera_controller.state is not None:
            import numpy as np

            with camera_controller.state_lock:
                if "azimuth" in camera_data:
                    camera_controller.state.azimuth = float(camera_data["azimuth"])
                if "elevation" in camera_data:
                    camera_controller.state.elevation = float(camera_data["elevation"])
                if "roll" in camera_data:
                    camera_controller.state.roll = float(camera_data["roll"])
                if "distance" in camera_data:
                    camera_controller.state.distance = float(camera_data["distance"])
                if "look_at" in camera_data:
                    look_at = camera_data["look_at"]
                    # Handle both list and tuple
                    if isinstance(look_at, (list, tuple)):
                        camera_controller.state.look_at = np.array(
                            look_at, dtype=np.float32
                        )

            # Apply camera state to viser camera
            camera_controller.apply_state_to_camera()
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

    logger.info(f"Imported viewer config from {input_path}")
