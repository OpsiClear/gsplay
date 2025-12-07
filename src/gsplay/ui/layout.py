"""
UI layout and control creation for the Universal GSPlay.

This module handles creating all viser UI controls and returning them
as a structured UIHandles dataclass.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING

import viser

# Import InfoPanel from its new home
from src.gsplay.ui.panels.info_panel import InfoPanel, create_info_panel

if TYPE_CHECKING:
    from src.gsplay.config.settings import UIHandles, GSPlayConfig

logger = logging.getLogger(__name__)


def create_transform_controls(
    server: viser.ViserServer, config: GSPlayConfig
) -> tuple[
    viser.GuiSliderHandle,
    viser.GuiSliderHandle,
    viser.GuiSliderHandle,
    viser.GuiSliderHandle,
    viser.GuiSliderHandle,
    viser.GuiSliderHandle,
    viser.GuiSliderHandle,
    viser.GuiButtonHandle,
    viser.GuiButtonHandle,
    viser.GuiButtonHandle,
]:
    """
    Create scene transform controls (scale and rotation).

    Uses world-axis rotation sliders (truly gimbal-lock free).
    Each rotation slider represents cumulative rotation around that world axis.
    Rotation deltas are applied via quaternion multiplication - no Euler
    angle decomposition means no singularities at any orientation.

    Parameters
    ----------
    server : viser.ViserServer
        Viser server instance
    config : GSPlayConfig
        GSPlay configuration

    Returns
    -------
    tuple
        (translation_x, translation_y, translation_z, global_scale,
         rotate_x, rotate_y, rotate_z, reset_button, center_button)
    """
    # Scene transformation (translation + scale + rotation combined)
    translate = tuple(
        float(x)
        for x in getattr(
            config.transform_values,
            "translate",
            getattr(config.transform_values, "translation", (0.0, 0.0, 0.0)),
        )
    )
    scale_value = getattr(config.transform_values, "scale", 1.0)

    # Scene rotation sliders use world-axis rotation (truly gimbal-lock free)
    # Each slider represents cumulative rotation around that world axis
    # Rotation is applied incrementally via quaternion multiplication
    # Initial values start at 0.0 (no rotation applied yet from sliders)

    translation_x = server.gui.add_slider(
        "Translation X",
        min=-10.0,
        max=10.0,
        step=0.01,
        initial_value=translate[0],
        hint="Move scene along X axis",
    )

    translation_y = server.gui.add_slider(
        "Translation Y",
        min=-10.0,
        max=10.0,
        step=0.01,
        initial_value=translate[1],
        hint="Move scene along Y axis",
    )

    translation_z = server.gui.add_slider(
        "Translation Z",
        min=-10.0,
        max=10.0,
        step=0.01,
        initial_value=translate[2],
        hint="Move scene along Z axis",
    )

    # World-axis rotation sliders (truly gimbal-lock free)
    # Each slider accumulates rotation around that world axis via quaternion multiplication
    # Changes are applied incrementally - no Euler angle decomposition needed
    rotate_x = server.gui.add_slider(
        "Rotate X",
        min=-180.0,
        max=180.0,
        step=1.0,
        initial_value=0.0,
        hint="Cumulative rotation around world X axis (pitch)",
    )

    rotate_y = server.gui.add_slider(
        "Rotate Y",
        min=-180.0,
        max=180.0,
        step=1.0,
        initial_value=0.0,
        hint="Cumulative rotation around world Y axis (yaw)",
    )

    rotate_z = server.gui.add_slider(
        "Rotate Z",
        min=-180.0,
        max=180.0,
        step=1.0,
        initial_value=0.0,
        hint="Cumulative rotation around world Z axis (roll)",
    )

    global_scale = server.gui.add_slider(
        "Scale",
        min=0.1,
        max=5.0,
        step=0.01,
        initial_value=float(scale_value)
        if isinstance(scale_value, (int, float))
        else scale_value[0],
        hint="Uniformly scale entire scene (positions and Gaussian sizes)",
    )

    center_button = server.gui.add_button("Center", hint="Center scene at origin")
    reset_button = server.gui.add_button("Reset")

    logger.debug("Created transform controls")
    return (
        translation_x,
        translation_y,
        translation_z,
        global_scale,
        rotate_x,
        rotate_y,
        rotate_z,
        reset_button,
        center_button,
    )


def create_config_menu(
    server: viser.ViserServer,
    config: GSPlayConfig,
    camera_controller=None,
    viewer_app=None,
) -> tuple[
    viser.GuiDropdownHandle,
    viser.GuiTextHandle,
    viser.GuiButtonGroupHandle,
    viser.GuiButtonGroupHandle,
    viser.GuiButtonGroupHandle,
]:
    """
    Create Config menu with processing mode, grid, world axis, and config export/import.

    Parameters
    ----------
    server : viser.ViserServer
        Viser server instance
    config : GSPlayConfig
        GSPlay configuration
    camera_controller : SuperSplatCamera | None
        Camera controller instance (for export/import)
    viewer_app : UniversalGSPlay | None
        GSPlay app instance (for accessing config and UI)

    Returns
    -------
    tuple
        (processing_mode_dropdown, data_path_input, load_data_button, config_path_input, config_buttons)
    """
    from src.infrastructure.processing_mode import ProcessingMode

    # Convert current config to display string
    try:
        current_mode = ProcessingMode.from_string(config.processing_mode)
        initial_mode = current_mode.to_display_string()
    except (ValueError, AttributeError):
        initial_mode = "All GPU"

    # Determine default config path (gsplay.yaml in data folder)
    default_config_path = "gsplay.yaml"
    if config.model_config_path:
        model_path = Path(str(config.model_config_path))
        if model_path.is_dir():
            default_config_path = str(model_path / "gsplay.yaml")
        else:
            default_config_path = str(model_path.parent / "gsplay.yaml")

    # Mode dropdown
    processing_mode = server.gui.add_dropdown(
        "Mode",
        ["All GPU", "Color+Transform GPU", "Transform GPU", "Color GPU", "All CPU"],
        initial_value=initial_mode,
        hint=(
            "Where to run processing stages:\n"
            "- All GPU: Fastest (default)\n"
            "- Color+Transform GPU: Filter on CPU, rest on GPU\n"
            "- Transform GPU: Filter+Color on CPU, Transform on GPU\n"
            "- Color GPU: Filter+Transform on CPU, Color on GPU\n"
            "- All CPU: Max GPU memory savings"
        ),
    )

    # Grid control
    grid_buttons = None
    if camera_controller is not None:
        grid_buttons = server.gui.add_button_group(
            "Grid",
            (" On ", "Off "),
        )
        # Set initial value based on camera state
        if camera_controller.grid_visible:
            grid_buttons.value = " On "
        else:
            grid_buttons.value = "Off "

        @grid_buttons.on_click
        def _(_) -> None:
            is_visible = grid_buttons.value.strip() == "On"
            camera_controller.grid_handle.visible = is_visible
            camera_controller.grid_visible = is_visible

    # World axis control
    world_axis_buttons = None
    if camera_controller is not None:
        world_axis_buttons = server.gui.add_button_group(
            "World Axis",
            (" On ", "Off "),
        )
        # Set initial value based on camera state
        if camera_controller.world_axis_visible:
            world_axis_buttons.value = " On "
        else:
            world_axis_buttons.value = "Off "

        @world_axis_buttons.on_click
        def _(_) -> None:
            is_visible = world_axis_buttons.value.strip() == "On"
            camera_controller.world_axis_handle.visible = is_visible
            camera_controller.world_axis_visible = is_visible

    # Config file path input
    config_path_input = server.gui.add_text(
        "Config Path",
        initial_value=default_config_path,
        hint="Path to YAML config file for export",
    )

    # Single Export Config button
    config_buttons = server.gui.add_button(
        "Export Config",
        icon=viser.Icon.DOWNLOAD,
        hint="Export current settings to config file",
    )

    # Setup callbacks
    if viewer_app is not None:
        from src.gsplay.config.io import export_viewer_config

        @config_buttons.on_click
        def _(event) -> None:
            try:
                output_path = Path(config_path_input.value)
                export_viewer_config(
                    viewer_app.config,
                    camera_controller,
                    output_path,
                    ui_handles=viewer_app.ui,
                )
                logger.info(f"Config exported to {output_path}")
            except Exception as e:
                logger.error(f"Failed to export config: {e}", exc_info=True)

    logger.debug("Created config menu")
    return (
        processing_mode,
        config_path_input,
        config_buttons,
        grid_buttons,
        world_axis_buttons,
    )


def create_data_loader_controls(
    server: viser.ViserServer,
    config: GSPlayConfig,
    viewer_app=None,
) -> tuple[viser.GuiTextHandle, viser.GuiButtonHandle]:
    """
    Create data loading controls (Data Path and Load button).

    Parameters
    ----------
    server : viser.ViserServer
        Viser server instance
    config : GSPlayConfig
        GSPlay configuration
    viewer_app : UniversalGSPlay | None
        GSPlay app instance (for accessing config and UI)

    Returns
    -------
    tuple
        (data_path_input, load_data_button)
    """
    data_path_input = server.gui.add_text(
        "Data Path",
        initial_value=str(config.model_config_path)
        if config.model_config_path
        else "./",
        hint="Path to PLY sequence folder or JSON config",
    )

    load_data_button = server.gui.add_button(
        "Load Data",
        icon=viser.Icon.FOLDER_OPEN,
        hint="Load data from specified path",
    )

    # Setup callbacks
    if viewer_app is not None:
        # Auto-update config path when data path changes (on first load)
        _config_path_auto_updated = False

        def update_config_path_from_data_path() -> None:
            """Update config path based on current data path."""
            nonlocal _config_path_auto_updated
            if not _config_path_auto_updated and data_path_input.value:
                try:
                    data_path = Path(data_path_input.value)
                    if data_path.exists():
                        if data_path.is_dir():
                            new_config_path = str(data_path / "config.yaml")
                        else:
                            new_config_path = str(data_path.parent / "config.yaml")
                        # Update config path in Config menu if it exists
                        if viewer_app.ui and viewer_app.ui.config_path_input:
                            viewer_app.ui.config_path_input.value = new_config_path
                        _config_path_auto_updated = True
                except Exception:
                    pass

        @data_path_input.on_update
        def _(_) -> None:
            update_config_path_from_data_path()

        @load_data_button.on_click
        def _(_) -> None:
            """Update config path when Load Data is clicked."""
            update_config_path_from_data_path()

    logger.debug("Created data loader controls")
    return (data_path_input, load_data_button)


def create_export_menu(
    server: viser.ViserServer,
    config: GSPlayConfig,
) -> tuple[
    viser.GuiTextHandle,
    viser.GuiDropdownHandle,
    viser.GuiDropdownHandle,
    viser.GuiButtonHandle,
]:
    """
    Create export controls.

    Parameters
    ----------
    server : viser.ViserServer
        Viser server instance
    config : GSPlayConfig
        GSPlay configuration

    Returns
    -------
    tuple
        (export_path, export_format, export_device, export_ply_button)
    """
    export_path = server.gui.add_text(
        "Save To Path", str(config.export_settings.export_path)
    )

    # Get available export formats from registry
    from src.infrastructure.registry import register_defaults, DataSinkRegistry
    register_defaults()

    sink_metadata = DataSinkRegistry.list_all()
    format_options = [meta.name for meta in sink_metadata] if sink_metadata else ["PLY"]
    # Default to first format or "Compressed PLY" if available
    initial_format = "Compressed PLY" if "Compressed PLY" in format_options else format_options[0]

    export_format = server.gui.add_dropdown(
        "Export Format",
        options=format_options,
        initial_value=initial_format,
        hint="Export format (populated from DataSinkRegistry)",
    )

    # Export device selection
    import torch

    export_device_options = ["CPU"]
    if torch.cuda.is_available():
        export_device_options.append("GPU")

    initial_device = (
        "GPU" if config.export_settings.export_device.startswith("cuda") else "CPU"
    )
    export_device = server.gui.add_dropdown(
        "Device",
        options=export_device_options,
        initial_value=initial_device,
        hint="Device for export processing: CPU (safer, slower) or GPU (faster, requires GPU memory)",
    )

    export_ply_button = server.gui.add_button(
        "Export All Frames with Current Edits"
    )

    logger.debug("Created export menu")
    return (
        export_path,
        export_format,
        export_device,
        export_ply_button,
    )


def create_volume_filter_controls(
    server: viser.ViserServer, config: GSPlayConfig
) -> dict:
    """
    Create volume filtering controls with full gsmod FilterValues support.

    Parameters
    ----------
    server : viser.ViserServer
        Viser server instance
    config : GSPlayConfig
        GSPlay configuration

    Returns
    -------
    dict
        Dictionary of all filter control handles
    """
    controls = {}

    # Get filter_values from config if available
    fv = getattr(config, "filter_values", None)

    # === Opacity/Scale Filtering ===
    controls["min_opacity"] = server.gui.add_slider(
        "Min Opacity",
        min=0.0,
        max=1.0,
        step=0.01,
        initial_value=fv.min_opacity if fv else 0.0,
        hint="Filter Gaussians with opacity < this",
    )

    controls["max_opacity"] = server.gui.add_slider(
        "Max Opacity",
        min=0.0,
        max=1.0,
        step=0.01,
        initial_value=fv.max_opacity if fv else 1.0,
        hint="Filter Gaussians with opacity > this",
    )

    controls["min_scale"] = server.gui.add_slider(
        "Min Scale",
        min=0.0,
        max=1.0,
        step=0.001,
        initial_value=fv.min_scale if fv else 0.0,
        hint="Filter Gaussians with scale < this",
    )

    initial_max_scale = fv.max_scale if fv else 100.0
    initial_max_scale = max(0.001, min(100.0, initial_max_scale))
    controls["max_scale"] = server.gui.add_slider(
        "Max Scale",
        min=0.001,
        max=100.0,
        step=0.01,
        initial_value=initial_max_scale,
        hint="Filter Gaussians with scale > this",
    )

    # === Spatial Filter Type ===
    controls["spatial_type"] = server.gui.add_dropdown(
        "Spatial Filter",
        ["None", "Sphere", "Box", "Ellipsoid", "Frustum"],
        initial_value="None",
        hint="Select spatial filter type",
    )

    # Show filter visualization toggle
    controls["show_filter_viz"] = server.gui.add_checkbox(
        "Show Filter",
        initial_value=False,
        hint="Show wireframe visualization of the spatial filter",
    )

    # === Sphere Filter ===
    sphere_center = fv.sphere_center if fv else (0.0, 0.0, 0.0)
    sphere_radius = fv.sphere_radius if fv and fv.sphere_radius != float("inf") else 10.0

    controls["sphere_center_x"] = server.gui.add_slider(
        "Sphere Center X", min=-20.0, max=20.0, step=0.1,
        initial_value=sphere_center[0], visible=False,
    )
    controls["sphere_center_y"] = server.gui.add_slider(
        "Sphere Center Y", min=-20.0, max=20.0, step=0.1,
        initial_value=sphere_center[1], visible=False,
    )
    controls["sphere_center_z"] = server.gui.add_slider(
        "Sphere Center Z", min=-20.0, max=20.0, step=0.1,
        initial_value=sphere_center[2], visible=False,
    )
    controls["sphere_radius"] = server.gui.add_slider(
        "Sphere Radius", min=0.01, max=50.0, step=0.1,
        initial_value=sphere_radius, visible=False,
        hint="Radius of sphere filter",
    )

    # === Box Filter ===
    box_min = fv.box_min if fv and fv.box_min else (-5.0, -5.0, -5.0)
    box_max = fv.box_max if fv and fv.box_max else (5.0, 5.0, 5.0)

    controls["box_min_x"] = server.gui.add_slider(
        "Box Min X", min=-20.0, max=20.0, step=0.1,
        initial_value=box_min[0], visible=False,
    )
    controls["box_min_y"] = server.gui.add_slider(
        "Box Min Y", min=-20.0, max=20.0, step=0.1,
        initial_value=box_min[1], visible=False,
    )
    controls["box_min_z"] = server.gui.add_slider(
        "Box Min Z", min=-20.0, max=20.0, step=0.1,
        initial_value=box_min[2], visible=False,
    )
    controls["box_max_x"] = server.gui.add_slider(
        "Box Max X", min=-20.0, max=20.0, step=0.1,
        initial_value=box_max[0], visible=False,
    )
    controls["box_max_y"] = server.gui.add_slider(
        "Box Max Y", min=-20.0, max=20.0, step=0.1,
        initial_value=box_max[1], visible=False,
    )
    controls["box_max_z"] = server.gui.add_slider(
        "Box Max Z", min=-20.0, max=20.0, step=0.1,
        initial_value=box_max[2], visible=False,
    )
    # Box rotation (degrees in UI, converted to radians internally)
    box_rot = fv.box_rot if fv and fv.box_rot else (0.0, 0.0, 0.0)
    controls["box_rot_x"] = server.gui.add_slider(
        "Box Rot X", min=-180.0, max=180.0, step=1.0,
        initial_value=math.degrees(box_rot[0]) if box_rot else 0.0, visible=False,
        hint="Box rotation around X axis (degrees)",
    )
    controls["box_rot_y"] = server.gui.add_slider(
        "Box Rot Y", min=-180.0, max=180.0, step=1.0,
        initial_value=math.degrees(box_rot[1]) if box_rot else 0.0, visible=False,
        hint="Box rotation around Y axis (degrees)",
    )
    controls["box_rot_z"] = server.gui.add_slider(
        "Box Rot Z", min=-180.0, max=180.0, step=1.0,
        initial_value=math.degrees(box_rot[2]) if box_rot else 0.0, visible=False,
        hint="Box rotation around Z axis (degrees)",
    )

    # === Ellipsoid Filter ===
    ellipsoid_center = fv.ellipsoid_center if fv and fv.ellipsoid_center else (0.0, 0.0, 0.0)
    ellipsoid_radii = fv.ellipsoid_radii if fv and fv.ellipsoid_radii else (5.0, 5.0, 5.0)

    controls["ellipsoid_center_x"] = server.gui.add_slider(
        "Ellipsoid Center X", min=-20.0, max=20.0, step=0.1,
        initial_value=ellipsoid_center[0], visible=False,
    )
    controls["ellipsoid_center_y"] = server.gui.add_slider(
        "Ellipsoid Center Y", min=-20.0, max=20.0, step=0.1,
        initial_value=ellipsoid_center[1], visible=False,
    )
    controls["ellipsoid_center_z"] = server.gui.add_slider(
        "Ellipsoid Center Z", min=-20.0, max=20.0, step=0.1,
        initial_value=ellipsoid_center[2], visible=False,
    )
    controls["ellipsoid_radius_x"] = server.gui.add_slider(
        "Ellipsoid Radius X", min=0.01, max=50.0, step=0.1,
        initial_value=ellipsoid_radii[0], visible=False,
    )
    controls["ellipsoid_radius_y"] = server.gui.add_slider(
        "Ellipsoid Radius Y", min=0.01, max=50.0, step=0.1,
        initial_value=ellipsoid_radii[1], visible=False,
    )
    controls["ellipsoid_radius_z"] = server.gui.add_slider(
        "Ellipsoid Radius Z", min=0.01, max=50.0, step=0.1,
        initial_value=ellipsoid_radii[2], visible=False,
    )
    # Ellipsoid rotation (degrees in UI, converted to radians internally)
    ellipsoid_rot = fv.ellipsoid_rot if fv and fv.ellipsoid_rot else (0.0, 0.0, 0.0)
    controls["ellipsoid_rot_x"] = server.gui.add_slider(
        "Ellipsoid Rot X", min=-180.0, max=180.0, step=1.0,
        initial_value=math.degrees(ellipsoid_rot[0]) if ellipsoid_rot else 0.0, visible=False,
        hint="Ellipsoid rotation around X axis (degrees)",
    )
    controls["ellipsoid_rot_y"] = server.gui.add_slider(
        "Ellipsoid Rot Y", min=-180.0, max=180.0, step=1.0,
        initial_value=math.degrees(ellipsoid_rot[1]) if ellipsoid_rot else 0.0, visible=False,
        hint="Ellipsoid rotation around Y axis (degrees)",
    )
    controls["ellipsoid_rot_z"] = server.gui.add_slider(
        "Ellipsoid Rot Z", min=-180.0, max=180.0, step=1.0,
        initial_value=math.degrees(ellipsoid_rot[2]) if ellipsoid_rot else 0.0, visible=False,
        hint="Ellipsoid rotation around Z axis (degrees)",
    )

    # === Frustum Filter ===
    frustum_fov_deg = (fv.frustum_fov if fv else 1.047) * 180.0 / 3.14159

    controls["frustum_fov"] = server.gui.add_slider(
        "Frustum FOV", min=10.0, max=120.0, step=1.0,
        initial_value=frustum_fov_deg, visible=False,
        hint="Field of view in degrees",
    )
    controls["frustum_aspect"] = server.gui.add_slider(
        "Frustum Aspect", min=0.5, max=3.0, step=0.1,
        initial_value=fv.frustum_aspect if fv else 1.0, visible=False,
        hint="Width/height ratio",
    )
    controls["frustum_near"] = server.gui.add_slider(
        "Frustum Near", min=0.01, max=10.0, step=0.01,
        initial_value=fv.frustum_near if fv else 0.1, visible=False,
    )
    controls["frustum_far"] = server.gui.add_slider(
        "Frustum Far", min=1.0, max=500.0, step=1.0,
        initial_value=fv.frustum_far if fv else 100.0, visible=False,
    )
    # Frustum position (camera position)
    frustum_pos = fv.frustum_pos if fv and fv.frustum_pos else (0.0, 0.0, 0.0)
    controls["frustum_pos_x"] = server.gui.add_slider(
        "Frustum Pos X", min=-50.0, max=50.0, step=0.1,
        initial_value=frustum_pos[0] if frustum_pos else 0.0, visible=False,
        hint="Camera X position",
    )
    controls["frustum_pos_y"] = server.gui.add_slider(
        "Frustum Pos Y", min=-50.0, max=50.0, step=0.1,
        initial_value=frustum_pos[1] if frustum_pos else 0.0, visible=False,
        hint="Camera Y position",
    )
    controls["frustum_pos_z"] = server.gui.add_slider(
        "Frustum Pos Z", min=-50.0, max=50.0, step=0.1,
        initial_value=frustum_pos[2] if frustum_pos else 0.0, visible=False,
        hint="Camera Z position",
    )
    # Frustum rotation (camera rotation as Euler angles in degrees)
    frustum_rot = fv.frustum_rot if fv and fv.frustum_rot else (0.0, 0.0, 0.0)
    controls["frustum_rot_x"] = server.gui.add_slider(
        "Frustum Rot X", min=-180.0, max=180.0, step=1.0,
        initial_value=math.degrees(frustum_rot[0]) if frustum_rot else 0.0, visible=False,
        hint="Camera pitch (degrees)",
    )
    controls["frustum_rot_y"] = server.gui.add_slider(
        "Frustum Rot Y", min=-180.0, max=180.0, step=1.0,
        initial_value=math.degrees(frustum_rot[1]) if frustum_rot else 0.0, visible=False,
        hint="Camera yaw (degrees)",
    )
    controls["frustum_rot_z"] = server.gui.add_slider(
        "Frustum Rot Z", min=-180.0, max=180.0, step=1.0,
        initial_value=math.degrees(frustum_rot[2]) if frustum_rot else 0.0, visible=False,
        hint="Camera roll (degrees)",
    )
    # Button to copy current camera state
    controls["frustum_use_camera"] = server.gui.add_button(
        "Use Current Camera", visible=False,
        hint="Copy current camera position and rotation to frustum filter",
    )

    # CPU filtering option (hidden, used for fallback on non-CUDA devices)
    controls["use_cpu_filtering"] = server.gui.add_checkbox(
        "CPU Filtering",
        initial_value=config.volume_filter.use_cpu_filtering,
        visible=False,
    )

    controls["reset_button"] = server.gui.add_button("Reset")

    # Setup visibility callbacks for spatial filter type
    @controls["spatial_type"].on_update
    def _update_spatial_visibility(_):
        spatial_type = controls["spatial_type"].value

        # Sphere controls
        sphere_visible = spatial_type == "Sphere"
        controls["sphere_center_x"].visible = sphere_visible
        controls["sphere_center_y"].visible = sphere_visible
        controls["sphere_center_z"].visible = sphere_visible
        controls["sphere_radius"].visible = sphere_visible

        # Box controls
        box_visible = spatial_type == "Box"
        controls["box_min_x"].visible = box_visible
        controls["box_min_y"].visible = box_visible
        controls["box_min_z"].visible = box_visible
        controls["box_max_x"].visible = box_visible
        controls["box_max_y"].visible = box_visible
        controls["box_max_z"].visible = box_visible
        controls["box_rot_x"].visible = box_visible
        controls["box_rot_y"].visible = box_visible
        controls["box_rot_z"].visible = box_visible

        # Ellipsoid controls
        ellipsoid_visible = spatial_type == "Ellipsoid"
        controls["ellipsoid_center_x"].visible = ellipsoid_visible
        controls["ellipsoid_center_y"].visible = ellipsoid_visible
        controls["ellipsoid_center_z"].visible = ellipsoid_visible
        controls["ellipsoid_radius_x"].visible = ellipsoid_visible
        controls["ellipsoid_radius_y"].visible = ellipsoid_visible
        controls["ellipsoid_radius_z"].visible = ellipsoid_visible
        controls["ellipsoid_rot_x"].visible = ellipsoid_visible
        controls["ellipsoid_rot_y"].visible = ellipsoid_visible
        controls["ellipsoid_rot_z"].visible = ellipsoid_visible

        # Frustum controls
        frustum_visible = spatial_type == "Frustum"
        controls["frustum_fov"].visible = frustum_visible
        controls["frustum_aspect"].visible = frustum_visible
        controls["frustum_near"].visible = frustum_visible
        controls["frustum_far"].visible = frustum_visible
        controls["frustum_pos_x"].visible = frustum_visible
        controls["frustum_pos_y"].visible = frustum_visible
        controls["frustum_pos_z"].visible = frustum_visible
        controls["frustum_rot_x"].visible = frustum_visible
        controls["frustum_rot_y"].visible = frustum_visible
        controls["frustum_rot_z"].visible = frustum_visible
        controls["frustum_use_camera"].visible = frustum_visible

    logger.debug("Created volume filter controls with full gsmod support")
    return controls


def create_color_controls(
    server: viser.ViserServer, config: GSPlayConfig
) -> dict:
    """
    Create basic color enhancement controls.

    Parameters
    ----------
    server : viser.ViserServer
        Viser server instance
    config : GSPlayConfig
        GSPlay configuration

    Returns
    -------
    dict
        Dictionary of basic color control handles
    """
    cv = config.color_values
    temp_ui = (cv.temperature + 1.0) / 2.0  # [-1,1] -> [0,1]
    tint_ui = (cv.tint + 1.0) / 2.0  # [-1,1] -> [0,1]

    controls = {}

    # Temperature: UI [0, 1] maps to gsmod [-1, 1]
    controls["temperature"] = server.gui.add_slider(
        "Temperature",
        min=0.0,
        max=1.0,
        step=0.01,
        initial_value=temp_ui,
        hint="0=cool, 0.5=neutral, 1=warm",
    )

    # Tint: UI [0, 1] maps to gsmod [-1, 1]
    controls["tint"] = server.gui.add_slider(
        "Tint",
        min=0.0,
        max=1.0,
        step=0.01,
        initial_value=tint_ui,
        hint="0=green, 0.5=neutral, 1=magenta",
    )

    controls["brightness"] = server.gui.add_slider(
        "Brightness", min=0.0, max=5.0, step=0.01, initial_value=cv.brightness
    )

    controls["contrast"] = server.gui.add_slider(
        "Contrast", min=0.0, max=5.0, step=0.01, initial_value=cv.contrast
    )

    controls["saturation"] = server.gui.add_slider(
        "Saturation", min=0.0, max=5.0, step=0.01, initial_value=cv.saturation
    )

    controls["gamma"] = server.gui.add_slider(
        "Gamma", min=0.1, max=5.0, step=0.01, initial_value=cv.gamma
    )

    # Opacity (from OpacityValues)
    controls["alpha_scaler"] = server.gui.add_slider(
        "Opacity",
        min=0.0,
        max=3.0,
        step=0.01,
        initial_value=config.alpha_scaler,
        hint="<1=fade, 1=normal, >1=boost",
    )

    # Unified color adjustment dropdown (gsmod 0.1.4 auto-correction + presets + advanced)
    from src.gsplay.core.handlers.color_presets import get_dropdown_options

    controls["color_adjustment"] = server.gui.add_dropdown(
        "Adjustment",
        get_dropdown_options(),
        initial_value="Auto Enhance",
        hint="Auto-correction (gsmod 0.1.4), style presets, or histogram learning",
    )
    controls["apply_button"] = server.gui.add_button(
        "Apply",
        hint="Apply selected color adjustment",
    )

    controls["reset_button"] = server.gui.add_button("Reset")

    logger.debug("Created basic color controls")
    return controls


def create_color_advanced_controls(
    server: viser.ViserServer, config: GSPlayConfig
) -> dict:
    """
    Create advanced color enhancement controls.

    Parameters
    ----------
    server : viser.ViserServer
        Viser server instance
    config : GSPlayConfig
        GSPlay configuration

    Returns
    -------
    dict
        Dictionary of advanced color control handles
    """
    cv = config.color_values
    shadows_ui = cv.shadows + 1.0  # [-1,1] -> [0,2]
    highlights_ui = cv.highlights + 1.0  # [-1,1] -> [0,2]

    controls = {}

    controls["vibrance"] = server.gui.add_slider(
        "Vibrance", min=0.0, max=5.0, step=0.01, initial_value=cv.vibrance
    )

    controls["hue_shift"] = server.gui.add_slider(
        "Hue Shift", min=-180.0, max=180.0, step=1.0, initial_value=cv.hue_shift
    )

    # Shadows/Highlights: UI [0, 2] maps to gsmod [-1, 1]
    controls["shadows"] = server.gui.add_slider(
        "Shadows",
        min=0.0,
        max=2.0,
        step=0.01,
        initial_value=shadows_ui,
        hint="1.0=neutral",
    )

    controls["highlights"] = server.gui.add_slider(
        "Highlights",
        min=0.0,
        max=2.0,
        step=0.01,
        initial_value=highlights_ui,
        hint="1.0=neutral",
    )

    # Fade (black point lift)
    controls["fade"] = server.gui.add_slider(
        "Fade",
        min=0.0,
        max=1.0,
        step=0.01,
        initial_value=cv.fade,
        hint="Lifts black point (matte/film look)",
    )

    # Split toning - shadows
    controls["shadow_tint_hue"] = server.gui.add_slider(
        "Shadow Hue",
        min=-180.0,
        max=180.0,
        step=1.0,
        initial_value=cv.shadow_tint_hue,
        hint="Hue for shadow tint",
    )

    controls["shadow_tint_sat"] = server.gui.add_slider(
        "Shadow Tint",
        min=0.0,
        max=1.0,
        step=0.01,
        initial_value=cv.shadow_tint_sat,
        hint="Intensity of shadow color tint",
    )

    # Split toning - highlights
    controls["highlight_tint_hue"] = server.gui.add_slider(
        "Highlight Hue",
        min=-180.0,
        max=180.0,
        step=1.0,
        initial_value=cv.highlight_tint_hue,
        hint="Hue for highlight tint",
    )

    controls["highlight_tint_sat"] = server.gui.add_slider(
        "Highlight Tint",
        min=0.0,
        max=1.0,
        step=0.01,
        initial_value=cv.highlight_tint_sat,
        hint="Intensity of highlight color tint",
    )

    controls["reset_button"] = server.gui.add_button("Reset")

    logger.debug("Created advanced color controls")
    return controls


def setup_ui_layout(
    server: viser.ViserServer,
    config: GSPlayConfig,
    camera_controller=None,
    viewer_app=None,
) -> UIHandles:
    """
    Create complete UI layout for the viewer.

    Parameters
    ----------
    server : viser.ViserServer
        Viser server instance
    config : GSPlayConfig
        GSPlay configuration
    camera_controller : SuperSplatCamera, optional
        Camera controller instance (if provided, UI will be created here)

    Returns
    -------
    UIHandles
        Dataclass containing all UI control handles
    """
    from src.gsplay.config.settings import UIHandles

    logger.debug("Setting up UI layout")
    view_only = getattr(config, "view_only", False)
    compact_ui = getattr(config, "compact_ui", False)

    # Info panel at very top (compact markdown display)
    info_panel = create_info_panel(server)

    # Spacer after info
    server.gui.add_markdown(content=" ")

    # Data loader (hidden in view-only mode)
    data_path_input = None
    load_data_button = None
    if not view_only:
        data_path_input, load_data_button = create_data_loader_controls(
            server, config, viewer_app=viewer_app
        )

    # Initialize playback/render controls
    time_slider = None
    auto_play = None
    play_speed = None
    render_quality = None
    jpeg_quality_slider = None
    auto_quality_checkbox = None
    setup_camera_sync = None
    # View/Camera controls
    zoom_slider = None
    azimuth_slider = None
    elevation_slider = None
    roll_slider = None

    # Playback controls (after data loader) - FPS and frame controls at root level
    if camera_controller is not None:
        from src.gsplay.rendering.camera import (
            create_fps_control,
            create_quality_controls,
            create_playback_controls,
            create_view_controls,
        )

        # Spacer before playback section
        server.gui.add_markdown(content=" ")

        # FPS control at root level
        play_speed = create_fps_control(server, config)

        # Playback controls (Frame slider + Play/Pause button)
        time_slider, auto_play = create_playback_controls(server, config)

    # Load Config button (under play controls)
    load_config_button = server.gui.add_button(
        "Load Config",
        icon=viser.Icon.UPLOAD,
        hint="Load settings from gsplay.yaml in data folder",
    )

    # Setup Load Config callback
    if viewer_app is not None:
        from src.gsplay.config.io import import_viewer_config
        from src.gsplay.interaction.events import EventType
        from src.infrastructure.processing_mode import ProcessingMode

        @load_config_button.on_click
        def _load_config_click(event) -> None:
            try:
                # Determine config path from model path
                if not config.model_config_path:
                    logger.warning("No model path set, cannot load config")
                    return

                model_path = Path(str(config.model_config_path))
                if model_path.is_dir():
                    config_path = model_path / "gsplay.yaml"
                else:
                    config_path = model_path.parent / "gsplay.yaml"

                if not config_path.exists():
                    logger.warning(f"Config file not found: {config_path}")
                    return

                logger.info(f"Loading config from {config_path}")

                import_viewer_config(
                    viewer_app.config,
                    camera_controller,
                    config_path,
                    ui_handles=viewer_app.ui,
                )

                # Update UI from imported config
                if viewer_app.ui:
                    viewer_app.ui.set_color_values(
                        viewer_app.config.color_values,
                        alpha_scaler=viewer_app.config.alpha_scaler,
                    )
                    viewer_app.ui.set_transform_values(
                        viewer_app.config.transform_values
                    )
                    viewer_app.ui.set_volume_filter(
                        viewer_app.config.volume_filter
                    )
                    viewer_app.ui.set_filter_values(
                        viewer_app.config.filter_values
                    )
                    # Update processing mode dropdown
                    if viewer_app.ui.processing_mode_dropdown is not None:
                        try:
                            mode = ProcessingMode.from_string(viewer_app.config.processing_mode)
                            viewer_app.ui.processing_mode_dropdown.value = mode.to_display_string()
                        except (ValueError, AttributeError):
                            pass

                    if viewer_app.event_bus:
                        viewer_app.event_bus.emit(EventType.RERENDER_REQUESTED)

                logger.info(f"Config loaded from {config_path}")
            except Exception as e:
                logger.error(f"Failed to load config: {e}", exc_info=True)

    # Spacer before tabs/folders
    server.gui.add_markdown(content=" ")

    # Initialize export controls (may be hidden in view-only mode)
    export_path = None
    export_format = None
    export_device = None
    export_ply_button = None
    config_path_input = None
    config_buttons = None

    if compact_ui:
        # Compact mode: wrap tab groups in collapsible folders
        # Main folder containing View/Config/Convert tabs
        with server.gui.add_folder("Main"):
            main_tabs = server.gui.add_tab_group()

            # View tab (camera controls)
            if camera_controller is not None:
                with main_tabs.add_tab("View", icon=None):
                    view_controls = create_view_controls(server, camera_controller)
                    zoom_slider, azimuth_slider, elevation_slider, roll_slider, setup_camera_sync = view_controls

                # Set up camera sync after all controls are created
                if setup_camera_sync:
                    setup_camera_sync()

            # Config tab (quality settings + config menu)
            with main_tabs.add_tab("Config", icon=None):
                # Quality controls (Quality, JPEG, Auto Quality)
                if camera_controller is not None:
                    render_quality, jpeg_quality_slider, auto_quality_checkbox = create_quality_controls(
                        server, config
                    )

                # Config menu (processing mode, grid, axis, save/load)
                (
                    processing_mode_dropdown,
                    cfg_path_input,
                    cfg_buttons,
                    grid_buttons,
                    world_axis_buttons,
                ) = create_config_menu(server, config, camera_controller, viewer_app=viewer_app)
                # Only expose config save controls if not view-only
                if not view_only:
                    config_path_input = cfg_path_input
                    config_buttons = cfg_buttons
                else:
                    # Hide config save controls in view-only mode
                    if cfg_path_input:
                        cfg_path_input.visible = False
                    if cfg_buttons:
                        cfg_buttons.visible = False

                # Terminate instance button
                terminate_button = server.gui.add_button(
                    "Terminate Instance",
                    icon=viser.Icon.POWER,
                    color="red",
                    hint="Shut down this viewer instance",
                )

            # Convert tab (hidden in view-only mode)
            if not view_only:
                with main_tabs.add_tab("Convert", icon=None):
                    (
                        export_path,
                        export_format,
                        export_device,
                        export_ply_button,
                    ) = create_export_menu(server, config)

        # Edit folder containing Transform/Filter/Color/Color+ tabs
        with server.gui.add_folder("Edit"):
            edit_tabs = server.gui.add_tab_group()

            # Transform tab
            with edit_tabs.add_tab("Transform", icon=None):
                (
                    translation_x,
                    translation_y,
                    translation_z,
                    global_scale,
                    rotate_x,
                    rotate_y,
                    rotate_z,
                    reset_pose,
                    center_button,
                ) = create_transform_controls(server, config)

            # Filter tab
            with edit_tabs.add_tab("Filter", icon=None):
                filter_controls = create_volume_filter_controls(server, config)

            # Color tab (basic)
            with edit_tabs.add_tab("Color", icon=None):
                color_controls = create_color_controls(server, config)

            # Color+ tab (advanced)
            with edit_tabs.add_tab("Color+", icon=None):
                color_advanced_controls = create_color_advanced_controls(server, config)

    else:
        # Standard mode: use tab groups
        # Create tab group for View/Config/Convert
        main_tabs = server.gui.add_tab_group()

        # View tab (camera controls) - first tab
        if camera_controller is not None:
            with main_tabs.add_tab("View", icon=None):
                view_controls = create_view_controls(server, camera_controller)
                zoom_slider, azimuth_slider, elevation_slider, roll_slider, setup_camera_sync = view_controls

            # Set up camera sync after all controls are created
            if setup_camera_sync:
                setup_camera_sync()

        # Config tab (quality settings + config menu)
        with main_tabs.add_tab("Config", icon=None):
            # Quality controls (Quality, JPEG, Auto Quality)
            if camera_controller is not None:
                render_quality, jpeg_quality_slider, auto_quality_checkbox = create_quality_controls(
                    server, config
                )

            # Config menu (processing mode, grid, axis, save/load)
            (
                processing_mode_dropdown,
                cfg_path_input,
                cfg_buttons,
                grid_buttons,
                world_axis_buttons,
            ) = create_config_menu(server, config, camera_controller, viewer_app=viewer_app)
            # Only expose config save controls if not view-only
            if not view_only:
                config_path_input = cfg_path_input
                config_buttons = cfg_buttons
            else:
                # Hide config save controls in view-only mode
                if cfg_path_input:
                    cfg_path_input.visible = False
                if cfg_buttons:
                    cfg_buttons.visible = False

            # Terminate instance button
            terminate_button = server.gui.add_button(
                "Terminate Instance",
                icon=viser.Icon.POWER,
                color="red",
                hint="Shut down this viewer instance",
            )

        # Convert tab (hidden in view-only mode)
        if not view_only:
            with main_tabs.add_tab("Convert", icon=None):
                (
                    export_path,
                    export_format,
                    export_device,
                    export_ply_button,
                ) = create_export_menu(server, config)

        # Spacer between tab groups
        server.gui.add_markdown(content=" ")

        # Create tab group for Transform, Filter, and Color controls
        edit_tabs = server.gui.add_tab_group()

        # Transform tab
        with edit_tabs.add_tab("Transform", icon=None):
            (
                translation_x,
                translation_y,
                translation_z,
                global_scale,
                rotate_x,
                rotate_y,
                rotate_z,
                reset_pose,
                center_button,
            ) = create_transform_controls(server, config)

        # Filter tab
        with edit_tabs.add_tab("Filter", icon=None):
            filter_controls = create_volume_filter_controls(server, config)

        # Color tab (basic)
        with edit_tabs.add_tab("Color", icon=None):
            color_controls = create_color_controls(server, config)

        # Color+ tab (advanced)
        with edit_tabs.add_tab("Color+", icon=None):
            color_advanced_controls = create_color_advanced_controls(server, config)

    # Assemble into UIHandles dataclass
    ui = UIHandles(
        # Data loader
        data_path_input=data_path_input,
        load_data_button=load_data_button,
        # Info display (compact InfoPanel)
        info_panel=info_panel,
        # Animation
        time_slider=time_slider,
        auto_play=auto_play,
        play_speed=play_speed,
        render_quality=render_quality,
        jpeg_quality_slider=jpeg_quality_slider,
        auto_quality_checkbox=auto_quality_checkbox,
        # View/Camera controls
        zoom_slider=zoom_slider,
        azimuth_slider=azimuth_slider,
        elevation_slider=elevation_slider,
        roll_slider=roll_slider,
        # Color adjustments - basic (from color_controls dict)
        temperature_slider=color_controls["temperature"],
        tint_slider=color_controls["tint"],
        brightness_slider=color_controls["brightness"],
        contrast_slider=color_controls["contrast"],
        saturation_slider=color_controls["saturation"],
        gamma_slider=color_controls["gamma"],
        alpha_scaler_slider=color_controls["alpha_scaler"],
        reset_colors_button=color_controls["reset_button"],
        # Unified color adjustment controls (gsmod 0.1.4)
        color_adjustment_dropdown=color_controls["color_adjustment"],
        apply_adjustment_button=color_controls["apply_button"],
        # Color adjustments - advanced (from color_advanced_controls dict)
        vibrance_slider=color_advanced_controls["vibrance"],
        hue_shift_slider=color_advanced_controls["hue_shift"],
        shadows_slider=color_advanced_controls["shadows"],
        highlights_slider=color_advanced_controls["highlights"],
        fade_slider=color_advanced_controls["fade"],
        shadow_tint_hue_slider=color_advanced_controls["shadow_tint_hue"],
        shadow_tint_sat_slider=color_advanced_controls["shadow_tint_sat"],
        highlight_tint_hue_slider=color_advanced_controls["highlight_tint_hue"],
        highlight_tint_sat_slider=color_advanced_controls["highlight_tint_sat"],
        reset_colors_advanced_button=color_advanced_controls["reset_button"],
        # Scene transforms
        translation_x_slider=translation_x,
        translation_y_slider=translation_y,
        translation_z_slider=translation_z,
        global_scale_slider=global_scale,
        rotate_x_slider=rotate_x,
        rotate_y_slider=rotate_y,
        rotate_z_slider=rotate_z,
        reset_pose_button=reset_pose,
        center_button=center_button,
        # Volume filtering (from dict) - basic
        min_opacity_slider=filter_controls["min_opacity"],
        max_opacity_slider=filter_controls["max_opacity"],
        min_scale_slider=filter_controls["min_scale"],
        max_scale_slider=filter_controls["max_scale"],
        # Spatial filter type
        spatial_filter_type=filter_controls["spatial_type"],
        # Sphere filter
        sphere_center_x=filter_controls["sphere_center_x"],
        sphere_center_y=filter_controls["sphere_center_y"],
        sphere_center_z=filter_controls["sphere_center_z"],
        sphere_radius=filter_controls["sphere_radius"],
        # Box filter
        box_min_x=filter_controls["box_min_x"],
        box_min_y=filter_controls["box_min_y"],
        box_min_z=filter_controls["box_min_z"],
        box_max_x=filter_controls["box_max_x"],
        box_max_y=filter_controls["box_max_y"],
        box_max_z=filter_controls["box_max_z"],
        box_rot_x=filter_controls["box_rot_x"],
        box_rot_y=filter_controls["box_rot_y"],
        box_rot_z=filter_controls["box_rot_z"],
        # Ellipsoid filter
        ellipsoid_center_x=filter_controls["ellipsoid_center_x"],
        ellipsoid_center_y=filter_controls["ellipsoid_center_y"],
        ellipsoid_center_z=filter_controls["ellipsoid_center_z"],
        ellipsoid_radius_x=filter_controls["ellipsoid_radius_x"],
        ellipsoid_radius_y=filter_controls["ellipsoid_radius_y"],
        ellipsoid_radius_z=filter_controls["ellipsoid_radius_z"],
        ellipsoid_rot_x=filter_controls["ellipsoid_rot_x"],
        ellipsoid_rot_y=filter_controls["ellipsoid_rot_y"],
        ellipsoid_rot_z=filter_controls["ellipsoid_rot_z"],
        # Frustum filter
        frustum_fov=filter_controls["frustum_fov"],
        frustum_aspect=filter_controls["frustum_aspect"],
        frustum_near=filter_controls["frustum_near"],
        frustum_far=filter_controls["frustum_far"],
        frustum_pos_x=filter_controls["frustum_pos_x"],
        frustum_pos_y=filter_controls["frustum_pos_y"],
        frustum_pos_z=filter_controls["frustum_pos_z"],
        frustum_rot_x=filter_controls["frustum_rot_x"],
        frustum_rot_y=filter_controls["frustum_rot_y"],
        frustum_rot_z=filter_controls["frustum_rot_z"],
        frustum_use_camera=filter_controls["frustum_use_camera"],
        # Other filter controls
        processing_mode_dropdown=processing_mode_dropdown,
        use_cpu_filtering_checkbox=filter_controls["use_cpu_filtering"],
        reset_filter_button=filter_controls["reset_button"],
        show_filter_viz=filter_controls["show_filter_viz"],
        # Export
        export_path=export_path,
        export_format=export_format,
        export_device=export_device,
        export_ply_button=export_ply_button,
        # Config menu
        config_path_input=config_path_input,
        config_buttons=config_buttons,
        load_config_button=load_config_button,
        # Instance control
        terminate_button=terminate_button,
    )

    logger.debug("UI layout created successfully")
    return ui

