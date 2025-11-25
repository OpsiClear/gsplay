"""
Layer management UI controls for CompositeModel.

This module provides UI controls for managing multiple Gaussian layers
in the CompositeModel (visibility, opacity, per-layer edits).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import viser

if TYPE_CHECKING:
    from src.domain.interfaces import CompositeModelInterface
    from src.models.composite.composite_model import CompositeModel
    from src.gsplay.config.settings import GSPlayConfig

logger = logging.getLogger(__name__)


def _create_layer_color_controls(
    server: viser.ViserServer,
    layer_config: GSPlayConfig,
    model: CompositeModel,
    layer_id: str,
) -> dict[str, viser.GuiInputHandle]:
    """Create color adjustment controls for a layer."""
    from gsmod import ColorValues

    adj = layer_config.color_values
    controls = {}

    # Map gsmod temperature [-1, 1] to UI [0, 1]
    temp_ui = (adj.temperature / 2.0) + 0.5
    controls["temperature"] = server.gui.add_slider(
        "Temperature", min=0.0, max=1.0, step=0.01, initial_value=temp_ui
    )

    controls["brightness"] = server.gui.add_slider(
        "Brightness", min=0.0, max=2.0, step=0.01, initial_value=adj.brightness
    )

    controls["contrast"] = server.gui.add_slider(
        "Contrast", min=0.0, max=2.0, step=0.01, initial_value=adj.contrast
    )

    controls["saturation"] = server.gui.add_slider(
        "Saturation", min=0.0, max=2.0, step=0.01, initial_value=adj.saturation
    )

    controls["vibrance"] = server.gui.add_slider(
        "Vibrance", min=0.0, max=2.0, step=0.01, initial_value=adj.vibrance
    )

    controls["hue_shift"] = server.gui.add_slider(
        "Hue Shift", min=-180.0, max=180.0, step=1.0, initial_value=adj.hue_shift
    )

    controls["gamma"] = server.gui.add_slider(
        "Gamma", min=0.5, max=2.0, step=0.01, initial_value=adj.gamma
    )

    # Map gsmod shadows [-1, 1] to UI [0, 2] (approximate mapping for slider)
    shadows_ui = adj.shadows + 1.0
    controls["shadows"] = server.gui.add_slider(
        "Shadows", min=0.5, max=1.5, step=0.01, initial_value=shadows_ui
    )

    # Map gsmod highlights [-1, 1] to UI [0, 2]
    highlights_ui = adj.highlights + 1.0
    controls["highlights"] = server.gui.add_slider(
        "Highlights", min=0.5, max=1.5, step=0.01, initial_value=highlights_ui
    )

    # Opacity multiplier (handled alongside ColorValues)
    controls["alpha_scaler"] = server.gui.add_slider(
        "Opacity",
        min=0.0,
        max=3.0,
        step=0.01,
        initial_value=getattr(layer_config, "alpha_scaler", 1.0),
        hint="Opacity adjustment: <1=reduce, 1=normal, >1=boost toward opaque",
    )

    controls["reset"] = server.gui.add_button("Reset")

    # Setup callbacks
    def _make_color_callback(param_name: str):
        def _callback(event: viser.GuiEvent):
            # Map UI values to gsmod values
            temp = (controls["temperature"].value - 0.5) * 2.0
            shadows = controls["shadows"].value - 1.0
            highlights = controls["highlights"].value - 1.0

            new_adj = ColorValues(
                temperature=temp,
                brightness=controls["brightness"].value,
                contrast=controls["contrast"].value,
                saturation=controls["saturation"].value,
                vibrance=controls["vibrance"].value,
                hue_shift=controls["hue_shift"].value,
                gamma=controls["gamma"].value,
                shadows=shadows,
                highlights=highlights,
            )
            model.update_layer_edits(
                layer_id,
                color_values=new_adj,
                alpha_scaler=controls["alpha_scaler"].value,
            )
            logger.debug(f"Layer '{layer_id}' color.{param_name}: {event.target.value}")

        return _callback

    for param in [
        "temperature",
        "brightness",
        "contrast",
        "saturation",
        "vibrance",
        "hue_shift",
        "gamma",
        "shadows",
        "highlights",
        "alpha_scaler",
    ]:
        controls[param].on_update(_make_color_callback(param))

    @controls["reset"].on_click
    def _on_reset(event: viser.GuiEvent):
        default_adj = ColorValues()
        # Map defaults back to UI
        controls["temperature"].value = 0.5  # (0.0 / 2.0) + 0.5
        controls["brightness"].value = default_adj.brightness
        controls["contrast"].value = default_adj.contrast
        controls["saturation"].value = default_adj.saturation
        controls["vibrance"].value = default_adj.vibrance
        controls["hue_shift"].value = default_adj.hue_shift
        controls["gamma"].value = default_adj.gamma
        controls["shadows"].value = 1.0  # 0.0 + 1.0
        controls["highlights"].value = 1.0  # 0.0 + 1.0
        controls["alpha_scaler"].value = 1.0
        model.update_layer_edits(
            layer_id,
            color_values=default_adj,
            alpha_scaler=1.0,
        )
        logger.info(f"Layer '{layer_id}' color adjustments reset to defaults")

    return controls


def _create_layer_transform_controls(
    server: viser.ViserServer,
    layer_config: GSPlayConfig,
    model: CompositeModel,
    layer_id: str,
) -> dict[str, viser.GuiInputHandle]:
    """Create transform controls for a layer."""
    import numpy as np
    from gsmod import TransformValues
    from gsmod.transform.api import euler_to_quaternion

    trans = layer_config.transform_values
    translate = getattr(
        trans, "translate", getattr(trans, "translation", (0.0, 0.0, 0.0))
    )
    controls = {}

    # Initialize Euler angles to identity (0,0,0)
    # Note: We don't convert existing quaternions back to Euler since that's ambiguous
    init_rot_x = 0.0
    init_rot_y = 0.0
    init_rot_z = 0.0

    controls["translation_x"] = server.gui.add_slider(
        "Translation X",
        min=-10.0,
        max=10.0,
        step=0.01,
        initial_value=float(translate[0]),
        hint="Move layer along X axis",
    )

    controls["translation_y"] = server.gui.add_slider(
        "Translation Y",
        min=-10.0,
        max=10.0,
        step=0.01,
        initial_value=float(translate[1]),
        hint="Move layer along Y axis",
    )

    controls["translation_z"] = server.gui.add_slider(
        "Translation Z",
        min=-10.0,
        max=10.0,
        step=0.01,
        initial_value=float(translate[2]),
        hint="Move layer along Z axis",
    )

    controls["rotation_x"] = server.gui.add_slider(
        "Rotation X",
        min=-180.0,
        max=180.0,
        step=1.0,
        initial_value=init_rot_x,
    )

    controls["rotation_y"] = server.gui.add_slider(
        "Rotation Y",
        min=-180.0,
        max=180.0,
        step=1.0,
        initial_value=init_rot_y,
    )

    controls["rotation_z"] = server.gui.add_slider(
        "Rotation Z",
        min=-180.0,
        max=180.0,
        step=1.0,
        initial_value=init_rot_z,
    )

    # Handle scalar vs vector scale
    scale_value = getattr(trans, "scale", 1.0)
    init_scale = (
        float(scale_value)
        if isinstance(scale_value, (float, int))
        else float(scale_value[0])
    )
    controls["global_scale"] = server.gui.add_slider(
        "Scale",
        min=0.1,
        max=5.0,
        step=0.01,
        initial_value=init_scale,
        hint="Uniformly scale layer (positions and Gaussian sizes)",
    )

    controls["reset"] = server.gui.add_button("Reset")

    # Setup callbacks
    def _make_transform_callback(param_name: str):
        def _callback(event: viser.GuiEvent):
            # Convert Euler to Quaternion
            euler_rad = np.radians(
                np.array(
                    [
                        controls["rotation_x"].value,
                        controls["rotation_y"].value,
                        controls["rotation_z"].value,
                    ],
                    dtype=np.float32,
                )
            )
            quat = euler_to_quaternion(euler_rad)

            translate_vec = np.array(
                [
                    controls["translation_x"].value,
                    controls["translation_y"].value,
                    controls["translation_z"].value,
                ],
                dtype=np.float32,
            )
            scale_val = controls["global_scale"].value
            try:
                new_trans = TransformValues(
                    translate=translate_vec,
                    scale=scale_val,
                    rotate=quat,
                )
            except TypeError:
                new_trans = TransformValues(
                    translation=translate_vec,
                    scale=scale_val,
                    rotation=quat,
                )
            model.update_layer_edits(layer_id, transform_values=new_trans)
            logger.debug(
                f"Layer '{layer_id}' transform.{param_name}: {event.target.value}"
            )

        return _callback

    for param in [
        "translation_x",
        "translation_y",
        "translation_z",
        "global_scale",
        "rotation_x",
        "rotation_y",
        "rotation_z",
    ]:
        controls[param].on_update(_make_transform_callback(param))

    @controls["reset"].on_click
    def _on_reset(event: viser.GuiEvent):
        default_trans = TransformValues()
        controls["translation_x"].value = 0.0
        controls["translation_y"].value = 0.0
        controls["translation_z"].value = 0.0
        controls["global_scale"].value = 1.0
        controls["rotation_x"].value = 0.0
        controls["rotation_y"].value = 0.0
        controls["rotation_z"].value = 0.0
        model.update_layer_edits(layer_id, transform_values=default_trans)
        logger.info(f"Layer '{layer_id}' transform reset to defaults")

    return controls


def _create_layer_filter_controls(
    server: viser.ViserServer,
    layer_config: GSPlayConfig,
    model: CompositeModel,
    layer_id: str,
) -> dict[str, viser.GuiInputHandle]:
    """Create filter controls for a layer."""
    from src.gsplay.config.settings import VolumeFilter

    filt = layer_config.volume_filter
    controls = {}

    # Filter type selection
    controls["filter_type"] = server.gui.add_dropdown(
        "Filter Type",
        ["None", "Sphere", "Cuboid"],
        initial_value=filt.filter_type.capitalize()
        if filt.filter_type != "none"
        else "None",
    )

    controls["opacity_threshold"] = server.gui.add_slider(
        "Min Opacity",
        min=0.0,
        max=1.0,
        step=0.01,
        initial_value=filt.opacity_threshold,
        hint="Filters out Gaussians with opacity < this value",
    )

    slider_max = max(10.0, filt.max_scale * 2.0)
    controls["max_scale"] = server.gui.add_slider(
        "Max Scale",
        min=0.001,
        max=slider_max,
        step=0.01,
        initial_value=filt.max_scale,
        hint="Filters out Gaussians with scale > this value (removes outliers)",
    )

    # Common controls for both filters
    controls["center_x"] = server.gui.add_slider(
        "Center X",
        min=-10.0,
        max=10.0,
        step=0.01,
        initial_value=filt.sphere_center[0],
    )

    controls["center_y"] = server.gui.add_slider(
        "Center Y",
        min=-10.0,
        max=10.0,
        step=0.01,
        initial_value=filt.sphere_center[1],
    )

    controls["center_z"] = server.gui.add_slider(
        "Center Z",
        min=-10.0,
        max=10.0,
        step=0.01,
        initial_value=filt.sphere_center[2],
    )

    # Sphere specific controls (normalized 0-1)
    controls["sphere_radius"] = server.gui.add_slider(
        "Sphere Size",
        min=0.0,
        max=1.0,
        step=0.01,
        initial_value=filt.sphere_radius_factor,
        visible=(filt.filter_type == "sphere"),
    )

    # Cuboid specific controls (normalized 0-1)
    controls["cuboid_x"] = server.gui.add_slider(
        "Width",
        min=0.0,
        max=1.0,
        step=0.01,
        initial_value=filt.cuboid_size_factor_x,
        visible=(filt.filter_type == "cuboid"),
    )

    controls["cuboid_y"] = server.gui.add_slider(
        "Height",
        min=0.0,
        max=1.0,
        step=0.01,
        initial_value=filt.cuboid_size_factor_y,
        visible=(filt.filter_type == "cuboid"),
    )

    controls["cuboid_z"] = server.gui.add_slider(
        "Depth",
        min=0.0,
        max=1.0,
        step=0.01,
        initial_value=filt.cuboid_size_factor_z,
        visible=(filt.filter_type == "cuboid"),
    )

    controls["reset"] = server.gui.add_button("Reset")

    # Setup callbacks
    def _make_filter_callback():
        def _callback(event: viser.GuiEvent):
            filter_type_str = controls["filter_type"].value.lower()
            new_filt = VolumeFilter(
                filter_type=filter_type_str,
                sphere_center=(
                    controls["center_x"].value,
                    controls["center_y"].value,
                    controls["center_z"].value,
                ),
                sphere_radius_factor=controls["sphere_radius"].value,
                cuboid_size_factor_x=controls["cuboid_x"].value,
                cuboid_size_factor_y=controls["cuboid_y"].value,
                cuboid_size_factor_z=controls["cuboid_z"].value,
                opacity_threshold=controls["opacity_threshold"].value,
                max_scale=controls["max_scale"].value,
            )
            model.update_layer_edits(layer_id, volume_filter=new_filt)

            # Update visibility of filter-specific controls
            controls["sphere_radius"].visible = filter_type_str == "sphere"
            controls["cuboid_x"].visible = filter_type_str == "cuboid"
            controls["cuboid_y"].visible = filter_type_str == "cuboid"
            controls["cuboid_z"].visible = filter_type_str == "cuboid"

            logger.debug(f"Layer '{layer_id}' filter updated")

        return _callback

    callback = _make_filter_callback()
    for param in [
        "filter_type",
        "opacity_threshold",
        "max_scale",
        "center_x",
        "center_y",
        "center_z",
        "sphere_radius",
        "cuboid_x",
        "cuboid_y",
        "cuboid_z",
    ]:
        controls[param].on_update(callback)

    @controls["reset"].on_click
    def _on_reset(event: viser.GuiEvent):
        default_filt = VolumeFilter()
        controls["filter_type"].value = "None"
        controls["opacity_threshold"].value = default_filt.opacity_threshold
        controls["max_scale"].value = default_filt.max_scale
        controls["center_x"].value = default_filt.sphere_center[0]
        controls["center_y"].value = default_filt.sphere_center[1]
        controls["center_z"].value = default_filt.sphere_center[2]
        controls["sphere_radius"].value = default_filt.sphere_radius_factor
        controls["cuboid_x"].value = default_filt.cuboid_size_factor_x
        controls["cuboid_y"].value = default_filt.cuboid_size_factor_y
        controls["cuboid_z"].value = default_filt.cuboid_size_factor_z
        controls["sphere_radius"].visible = False
        controls["cuboid_x"].visible = False
        controls["cuboid_y"].visible = False
        controls["cuboid_z"].visible = False
        model.update_layer_edits(layer_id, volume_filter=default_filt)
        logger.info(f"Layer '{layer_id}' filter reset to defaults")

    return controls


def create_layer_controls(
    server: viser.ViserServer, model: CompositeModelInterface
) -> dict[str, dict[str, viser.GuiInputHandle]]:
    """
    Create layer management UI controls for a model with layer support.

    Parameters
    ----------
    server : viser.ViserServer
        Viser server instance
    model : CompositeModelInterface
        Model implementing the layer management interface

    Returns
    -------
    dict[str, dict[str, viser.GuiInputHandle]]
        Dictionary mapping layer_id to dict of control handles:
        {
            "layer_id": {
                "visibility": viser.GuiCheckboxHandle,
                "opacity": viser.GuiSliderHandle,
                "info": viser.GuiMarkdownHandle,
                "edit_controls": {
                    "color": {...},
                    "transform": {...},
                    "filter": {...}
                }
            }
        }
    """
    layer_controls = {}

    with server.gui.add_folder("Layer Management"):
        layer_info = model.get_layer_info()

        # Sort layers by z_order for display
        sorted_layers = sorted(layer_info.items(), key=lambda x: x[1]["z_order"])

        for layer_id, info in sorted_layers:
            with server.gui.add_folder(f"Layer: {layer_id}"):
                # Layer info display
                info_text = (
                    f"**Type**: {info['type']}  \n"
                    f"**Frames**: {info['frames']}  \n"
                    f"**Z-Order**: {info['z_order']}  \n"
                    f"**Static**: {info['static']}"
                )
                info_display = server.gui.add_markdown(info_text)

                # Visibility checkbox
                visibility_checkbox = server.gui.add_checkbox(
                    "Visible", initial_value=info["visible"]
                )

                # Opacity slider
                opacity_slider = server.gui.add_slider(
                    "Opacity",
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    initial_value=info["opacity_multiplier"],
                )

                # Get layer's GSPlayConfig
                layer_config = model.get_layer_viewer_config(layer_id)

                # Per-layer edit controls using tabs
                edit_controls = {}
                # Create tab group for layer edit controls
                edit_tabs = server.gui.add_tab_group()

                # Color tab
                with edit_tabs.add_tab("Color", icon=None):
                    edit_controls["color"] = _create_layer_color_controls(
                        server, layer_config, model, layer_id
                    )

                # Transform tab
                with edit_tabs.add_tab("Transform", icon=None):
                    edit_controls["transform"] = _create_layer_transform_controls(
                        server, layer_config, model, layer_id
                    )

                # Filter tab
                with edit_tabs.add_tab("Filter", icon=None):
                    edit_controls["filter"] = _create_layer_filter_controls(
                        server, layer_config, model, layer_id
                    )

                # Store handles
                layer_controls[layer_id] = {
                    "visibility": visibility_checkbox,
                    "opacity": opacity_slider,
                    "info": info_display,
                    "edit_controls": edit_controls,
                }

                # Setup callbacks
                @visibility_checkbox.on_update
                def _on_visibility_change(event: viser.GuiEvent, lid=layer_id):
                    model.set_layer_visibility(lid, event.target.value)
                    logger.info(f"Layer '{lid}' visibility: {event.target.value}")

                @opacity_slider.on_update
                def _on_opacity_change(event: viser.GuiEvent, lid=layer_id):
                    model.layer_configs[lid]["opacity_multiplier"] = event.target.value
                    logger.info(f"Layer '{lid}' opacity: {event.target.value:.2f}")

    logger.info(f"Created layer controls for {len(layer_controls)} layers")

    return layer_controls
