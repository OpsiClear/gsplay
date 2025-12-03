"""
Main viewer application class.

This module provides the UniversalGSPlay class that orchestrates
all viewer components. Supports local filesystem and cloud storage.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import torch
import viser
from gsmod import ColorValues, FilterValues, TransformValues

from src.domain.entities import GSTensor
from src.domain.interfaces import (
    ModelInterface,
    DataLoaderInterface,
    DataSourceProtocol,
)
from src.infrastructure.processing_mode import ProcessingMode
from src.infrastructure.io.path_io import UniversalPath
from src.gsplay.config.settings import GSPlayConfig, UIHandles, VolumeFilter
from src.gsplay.core.container import create_edit_manager
from src.gsplay.interaction.handlers import HandlerManager
from src.gsplay.ui.layers import create_layer_controls
from src.gsplay.state.scene_bounds_manager import SceneBoundsManager
from src.gsplay.rendering.camera import create_supersplat_camera_controls
from src.gsplay.ui.layout import setup_ui_layout
from src.gsplay.ui.filter_visualizer import FilterVisualizer
from src.gsplay.core.api import GSPlayAPI
from src.gsplay.core.components import ModelComponent, RenderComponent, ExportComponent
from src.gsplay.interaction.events import EventBus, Event, EventType
from src.gsplay.ui.controller import UIController
from src.gsplay.interaction.playback import PlaybackController
from src.gsplay.streaming import (
    WebSocketStreamServer as StreamServer,
    set_websocket_server as set_stream_server,
)

logger = logging.getLogger(__name__)


class UniversalGSPlay:
    """
    Main viewer application.

    Orchestrates UI, model loading, rendering, and event handling.
    Delegates specialized tasks to focused modules:
    - ModelComponent: Model loading and lifecycle management
    - RenderComponent: Rendering pipeline setup
    - ExportComponent: Frame export operations
    - SceneBoundsManager: Scene bounds calculation and management
    - EditManager: Edit history and application of edits
    - HandlerManager: UI event handlers
    - Nerfview: 3D rendering and camera controls
    - UIController: UI updates based on events
    - PlaybackController: Animation loop and state
    """

    def __init__(self, config: GSPlayConfig):
        """
        Initialize the viewer.

        Parameters
        ----------
        config : GSPlayConfig
            GSPlay configuration
        """
        self.config = config

        # Setup device
        self.device = self._setup_device()
        logger.debug(f"Using device: {self.device}")

        # Create viser server
        self.server = viser.ViserServer(
            port=config.port,
            host=config.host,
            verbose=True,
        )
        logger.debug(f"Viser server started on {config.host}:{config.port}")

        # Event bus for component communication
        self.event_bus = EventBus(name="viewer")

        # Initialize components
        self.model_component = ModelComponent(
            device=self.device, event_bus=self.event_bus
        )
        self.render_component = RenderComponent(
            server=self.server,
            device=self.device,
            output_dir=config.output_dir,
            event_bus=self.event_bus,
        )
        self.export_component = ExportComponent(
            event_bus=self.event_bus,
            default_output_dir=None,  # Will be set from source path
        )

        # UI components (initialized in setup_viewer)
        self.ui: UIHandles | None = None
        self.ui_controller: UIController | None = None
        self.handlers: HandlerManager | None = None

        # Managers
        self.scene_bounds_manager = SceneBoundsManager()
        self.edit_manager = create_edit_manager(config, self.device)
        self.camera_controller = None  # SuperSplatCamera, created in setup_viewer
        self.filter_visualizer: FilterVisualizer | None = None
        self.playback_controller = PlaybackController(config, self.event_bus)
        self._apply_edits_fn: Callable[[GSTensor], GSTensor] | None = None

        # Programmatic API (initialized after setup_viewer)
        self.api: GSPlayAPI | None = None

        # Stream server for view-only output (WebSocket-based, low latency)
        self.stream_server: StreamServer | None = None

        # Control server for external commands (HTTP-based)
        self.control_server = None  # ControlServer, started in setup_viewer

        # Track if user manually edited export path (don't auto-update if so)
        self._user_edited_export_path: bool = False
        self._last_auto_export_path: str | None = None

    @property
    def model(self) -> ModelInterface | None:
        """Get the current model from model component."""
        return self.model_component.get_model()

    @property
    def data_loader(self) -> DataLoaderInterface | None:
        """Get the current data loader from model component."""
        return self.model_component.get_data_loader()

    @property
    def source_path(self) -> Path | None:
        """Get the source path from model component."""
        return self.model_component.get_source_path()

    def _is_export_path_placeholder(self) -> bool:
        """Return True if export path is still the placeholder value."""
        export_settings = getattr(self.config, "export_settings", None)
        if not export_settings or not getattr(export_settings, "export_path", None):
            return True
        current = str(export_settings.export_path).replace("\\", "/").rstrip("/")
        return current in ("./export_with_edits", "export_with_edits")

    def _build_default_export_dir(
        self, export_format: str, export_device: str | None = None
    ) -> UniversalPath:
        """Create export path: {source_name}_export_{device}_{format}.

        Parameters
        ----------
        export_format : str
            Export format (e.g., "ply", "compressed-ply")
        export_device : str | None
            Export device (e.g., "cpu", "cuda:0"). If None, reads from config/UI.
        """
        # Get source folder as base (export folder created inside source path)
        if self.source_path:
            base_dir = UniversalPath(self.source_path)
            source_name = UniversalPath(self.source_path).name
        else:
            base_dir = UniversalPath(".")
            source_name = "export"

        # Get device string (normalize to "gpu" or "cpu")
        if export_device is None:
            if self.ui and self.ui.export_device:
                device_value = self.ui.export_device.value.upper()
                export_device = "gpu" if device_value == "GPU" else "cpu"
            else:
                export_device = "cpu"
        else:
            # Normalize device string - handle "cuda:0", "cuda", "gpu", "CPU", etc.
            device_lower = export_device.lower()
            export_device = "gpu" if device_lower.startswith("cuda") or device_lower == "gpu" else "cpu"

        # Normalize format string (remove dashes, use underscore)
        format_str = export_format.replace("-", "_")

        # Build path: {source_name}_export_{device}_{format}
        folder_name = f"{source_name}_export_{export_device}_{format_str}"
        return base_dir / folder_name

    def _set_export_path(
        self, export_path: UniversalPath | str, *, is_auto: bool = False
    ) -> None:
        """Store export path on config and sync UI control if it exists.

        Parameters
        ----------
        export_path : UniversalPath | str
            The export path to set
        is_auto : bool
            If True, this is an auto-generated path (track for comparison)
        """
        resolved_path = UniversalPath(export_path)
        self.config.export_settings.export_path = resolved_path
        if self.ui and self.ui.export_path:
            self.ui.export_path.value = str(resolved_path)

        if is_auto:
            self._last_auto_export_path = str(resolved_path)
            self._user_edited_export_path = False

    def _initialize_export_path(
        self,
        export_format: str | None = None,
        export_device: str | None = None,
        *,
        force: bool = False,
    ) -> UniversalPath:
        """Ensure the export path has a user-visible default."""
        if export_format is None:
            export_format = self.config.export_settings.export_format.lower()

        if not force and not self._is_export_path_placeholder():
            return self.config.export_settings.export_path

        export_path = self._build_default_export_dir(export_format, export_device)
        self._set_export_path(export_path, is_auto=True)
        return export_path

    def _update_export_path_on_option_change(self) -> None:
        """Update export path when format/device options change.

        Only updates if user hasn't manually edited the path.
        """
        # Check if user manually edited the path
        if self._user_edited_export_path:
            return

        # Check if current UI path matches the last auto-generated path
        # Use UI value (what user sees) not config value
        # Normalize paths for comparison (handle Windows backslash vs forward slash)
        current_path = ""
        if self.ui and self.ui.export_path:
            current_path = self.ui.export_path.value.strip().replace("\\", "/")

        last_auto = (self._last_auto_export_path or "").replace("\\", "/")
        if last_auto and current_path != last_auto:
            # User edited the path, don't auto-update
            self._user_edited_export_path = True
            return

        # Get current format and device from UI
        export_format = "compressed-ply"
        if self.ui and self.ui.export_format:
            format_map = {
                "compressed ply": "compressed-ply",
                "ply": "ply",
            }
            ui_value = self.ui.export_format.value.lower()
            export_format = format_map.get(ui_value, ui_value)

        export_device = None
        if self.ui and self.ui.export_device:
            export_device = self.ui.export_device.value.lower()

        # Generate new path
        export_path = self._build_default_export_dir(export_format, export_device)
        self._set_export_path(export_path, is_auto=True)

    def _refresh_render_pipeline(self, model: ModelInterface | None) -> None:
        """Rebuild the render function so the viewer reflects the active model."""
        if not self.ui or model is None:
            return

        if self._apply_edits_fn is None:

            def apply_edits_wrapper(gaussians: GSTensor) -> GSTensor:
                return self.edit_manager.apply_edits(
                    gaussians, scene_bounds=self.scene_bounds_manager.get_bounds()
                )

            self._apply_edits_fn = apply_edits_wrapper

        render_fn = self.render_component.create_render_function(
            model=model,
            ui=self.ui,
            apply_edits_fn=self._apply_edits_fn,
            config=self.config,
        )

        if self.viewer:
            self.viewer.render_fn = render_fn
            if self.ui.time_slider:
                self.viewer.total_frames = model.get_total_frames()
                self.viewer.time_slider = self.ui.time_slider

    @property
    def viewer(self) -> object | None:
        """Get the nerfview viewer from render component."""
        return self.render_component.get_viewer()

    def _setup_device(self) -> str:
        """
        Auto-detect CUDA or fallback to CPU.

        Returns
        -------
        str
            Device string (e.g., 'cuda:0', 'cuda:1', or 'cpu')
        """
        # Check if device is CUDA (handles "cuda", "cuda:0", "cuda:1", etc.)
        is_cuda = self.config.device.startswith("cuda")

        if is_cuda and torch.cuda.is_available():
            # Extract device number if specified (e.g., "cuda:1" -> 1)
            if ":" in self.config.device:
                device_num = int(self.config.device.split(":")[1])
                if device_num >= torch.cuda.device_count():
                    logger.warning(
                        f"GPU {device_num} requested but only {torch.cuda.device_count()} available, "
                        f"using GPU 0"
                    )
                    device_num = 0
            else:
                device_num = 0

            device_name = torch.cuda.get_device_name(device_num)
            device_str = f"cuda:{device_num}"
            logger.debug(f"CUDA available: {device_name} (using {device_str})")
            return device_str
        else:
            if is_cuda:
                logger.warning("CUDA requested but not available, using CPU")
            return "cpu"

    def load_model_from_config(
        self, config_dict: dict[str, object], config_file: str | None = None
    ) -> None:
        """
        Load model based on configuration dictionary using ModelComponent.

        Parameters
        ----------
        config_dict : dict[str, object]
            Configuration dictionary with 'module' and 'config' keys
        config_file : str | None
            Optional path to config file (for reference)
        """
        module_type = config_dict.get("module", "load-ply")

        # Add PLY loading config from viewer config if available
        extra_config = {}
        if module_type == "load-ply":
            if hasattr(self.config, "ply_loading"):
                extra_config["enable_concurrent_prefetch"] = (
                    self.config.ply_loading.enable_concurrent_prefetch
                )
                extra_config["opacity_threshold"] = (
                    self.config.ply_loading.opacity_threshold
                )
                extra_config["scale_threshold"] = (
                    self.config.ply_loading.scale_threshold
                )
                extra_config["enable_quality_filtering"] = (
                    self.config.ply_loading.enable_quality_filtering
                )
            # Pass global processing_mode from GSPlayConfig
            if hasattr(self.config, "processing_mode"):
                try:
                    edit_mode = ProcessingMode.from_string(self.config.processing_mode)
                    extra_config["processing_mode"] = edit_mode.loader_mode
                except ValueError:
                    logger.warning(
                        "Invalid processing mode '%s' in config; using ALL_GPU for loader",
                        getattr(self.config, "processing_mode", "unknown"),
                    )
                    extra_config["processing_mode"] = ProcessingMode.ALL_GPU.value

        # Use ModelComponent to load model
        model, data_loader, metadata = self.model_component.load_from_config(
            config_dict, config_file=config_file, **extra_config
        )

        # Update export component with source path
        if self.model_component.get_source_path():
            self.export_component.set_default_output_dir(
                self.model_component.get_source_path()
            )
        self._initialize_export_path()

        # Process recommended max_scale
        recommended_max_scale = self.model_component.get_recommended_max_scale()
        if recommended_max_scale is not None:
            self.config.volume_filter.max_scale = recommended_max_scale
            logger.info(
                f"Set initial max_scale to {recommended_max_scale:.6f} "
                f"(99.5th percentile from first frame)"
            )

            # Update UI slider if already created
            if (
                self.ui
                and hasattr(self.ui, "max_scale_slider")
                and self.ui.max_scale_slider
            ):
                self.ui.max_scale_slider.value = recommended_max_scale
                # Optionally adjust slider max to be 2x the calculated value
                slider_max = max(10.0, recommended_max_scale * 2.0)
                self.ui.max_scale_slider.max = slider_max
                logger.debug(
                    f"Updated max_scale slider: value={recommended_max_scale:.6f}, "
                    f"max={slider_max:.2f}"
                )

        # Calculate scene bounds
        self.scene_bounds_manager.calculate_bounds(model)

        # Update UI if already created
        if self.ui and hasattr(self.ui, "time_slider") and self.ui.time_slider:
            total_frames = model.get_total_frames()
            self.ui.time_slider.max = total_frames - 1
            self.ui.time_slider.value = 0
            # Update info panel with total frames
            if self.ui.info_panel:
                self.ui.info_panel.set_frame_index(0, total_frames)
            logger.info(
                f"Time slider configured: range 0-{total_frames - 1} "
                f"({total_frames} frames, slider=N shows frame_N.ply)"
            )

    def setup_viewer(self) -> None:
        """Setup UI, handlers, and nerfview viewer."""
        logger.debug("Setting up viewer...")

        # Configure viser theme to match launcher style (cyan accent #06b6d4)
        # Use floating panel for compact/mobile UI, fixed panel (right sidebar) for desktop
        layout = "floating" if self.config.compact_ui else "fixed"

        self.server.gui.configure_theme(
            control_layout=layout,
            control_width="medium",
            dark_mode=True,
            brand_color=(6, 182, 212),
            show_logo=False,
            show_share_button=False,
        )

        # Create camera controller first (UI will be created in setup_ui_layout)
        self.camera_controller = create_supersplat_camera_controls(
            self.server, self.scene_bounds_manager.get_bounds()
        )

        # Create UI (includes camera UI right after Info panel)
        self.ui = setup_ui_layout(
            self.server, self.config, self.camera_controller, viewer_app=self
        )
        logger.debug("UI layout created")

        # Initialize UI Controller
        self.ui_controller = UIController(self.ui, self.event_bus)

        # Subscribe to events
        self.event_bus.subscribe(EventType.MODEL_LOADED, self._on_model_loaded)
        self.event_bus.subscribe(EventType.FRAME_CHANGED, self._on_frame_changed)

        # Subscribe to command events
        self.event_bus.subscribe(
            EventType.LOAD_DATA_REQUESTED, self._on_load_data_requested
        )
        self.event_bus.subscribe(EventType.EXPORT_REQUESTED, self._on_export_requested)
        self.event_bus.subscribe(
            EventType.RESET_COLORS_REQUESTED, self._on_reset_colors_requested
        )
        self.event_bus.subscribe(
            EventType.RESET_TRANSFORM_REQUESTED, self._on_reset_transform_requested
        )
        self.event_bus.subscribe(
            EventType.RESET_FILTER_REQUESTED, self._on_reset_filter_requested
        )
        self.event_bus.subscribe(
            EventType.CENTER_REQUESTED, self._on_center_requested
        )
        self.event_bus.subscribe(
            EventType.RERENDER_REQUESTED, self._on_rerender_requested
        )
        self.event_bus.subscribe(
            EventType.TERMINATE_REQUESTED, self._on_terminate_requested
        )

        # Update time slider if model already loaded
        if self.model:
            # Ensure playback controller has the model
            self.playback_controller.set_model(self.model)

            if self.ui.time_slider:
                total_frames = self.model.get_total_frames()
                self.ui.time_slider.max = total_frames - 1
                self.ui.time_slider.value = 0
                # Update info panel with total frames
                if self.ui.info_panel:
                    self.ui.info_panel.set_frame_index(0, total_frames)
                logger.debug(
                    f"Updated time slider: 0-{total_frames - 1} ({total_frames} frames)"
                )

        # Create handlers
        self.handlers = HandlerManager(self.event_bus)
        self.handlers.set_playback_controller(self.playback_controller)

        # Setup all UI callbacks
        self.handlers.setup_all_callbacks(
            self.ui, self.scene_bounds_manager.get_bounds()
        )
        logger.debug("Event handlers registered")

        # Create render function with edit wrapper
        def apply_edits_wrapper(gaussians: GSTensor) -> GSTensor:
            return self.edit_manager.apply_edits(
                gaussians, scene_bounds=self.scene_bounds_manager.get_bounds()
            )

        self._apply_edits_fn = apply_edits_wrapper

        # Setup rendering using RenderComponent
        self.render_component.setup_viewer(
            model=self.model,
            ui=self.ui,
            apply_edits_fn=self._apply_edits_fn,
            mode="rendering",
            time_enabled=True,
            jpeg_quality_static=self.config.render_settings.jpeg_quality_static,
            jpeg_quality_move=self.config.render_settings.jpeg_quality_move,
            config=self.config,
        )

        # Configure render quality
        self.render_component.configure_quality(self.ui)

        # Set viewer in handlers
        self.handlers.set_viewer(self.viewer)

        # Add layer controls for CompositeModel
        self._setup_layer_controls()

        # Setup filter visualizer
        self._setup_filter_visualizer()

        # Setup auto-learn color controls
        self._setup_auto_learn_color()

        # Setup export option handlers
        self._setup_export_handlers()

        # Initialize export path now that UI exists (model may have been loaded before UI)
        self._initialize_export_path(force=True)

        # Initialize programmatic API
        self.api = GSPlayAPI(self)
        logger.debug("Programmatic API initialized")

        # Start stream server if configured
        self._start_stream_server()

        # Start control server for external commands
        self._start_control_server()

    def _start_stream_server(self) -> None:
        """Start WebSocket stream server if configured.

        Stream port convention:
        - stream_port == 0: Streaming disabled
        - stream_port != 0 (including -1): Enable streaming on viser_port + 1

        WebSocket streaming provides ~100-150ms latency over the internet.
        """
        if self.config.stream_port == 0:
            return  # Streaming disabled

        try:
            stream_port = self.config.port + 1
            target_fps = int(self.config.animation.play_speed_fps)

            self.stream_server = StreamServer(
                port=stream_port,
                target_fps=target_fps,
            )
            actual_port = self.stream_server.start()

            # Register globally so renderer can find it
            set_stream_server(self.stream_server)

            logger.info(
                f"Stream server: http://{self.config.host}:{actual_port}/"
            )
        except Exception as e:
            logger.warning(f"Failed to start stream server: {e}")
            self.stream_server = None

    def _start_control_server(self) -> None:
        """Start HTTP control server for remote commands.

        Control port convention: viser_port + 2
        Provides endpoints: /center-scene, /get-state, /set-translation
        """
        try:
            from src.gsplay.control.server import ControlServer

            control_port = self.config.port + 2

            self.control_server = ControlServer(control_port, self)
            actual_port = self.control_server.start()

            logger.info(
                f"Control server: http://{self.config.host}:{actual_port}/"
            )
        except Exception as e:
            logger.warning(f"Failed to start control server: {e}")
            self.control_server = None

    def _on_model_loaded(self, event: Event) -> None:
        """
        Handle model loaded event.

        Updates scene bounds, camera controller, and render pipeline.
        """
        model = self.model_component.get_model()
        if not model:
            return

        # Update playback controller
        self.playback_controller.set_model(model)

        # Calculate scene bounds
        self.scene_bounds_manager.calculate_bounds(model)

        # Update camera controller
        if self.camera_controller:
            self.camera_controller.update_scene_bounds(
                self.scene_bounds_manager.get_bounds()
            )

        # Refresh render pipeline
        self._refresh_render_pipeline(model)

        # Trigger rerender
        if self.viewer:
            self.render_component.rerender()

    def _on_frame_changed(self, event: Event) -> None:
        """Handle frame changed event from playback controller."""
        # Trigger rerender when frame changes
        if self.viewer:
            self.render_component.rerender()

    def _on_load_data_requested(self, event: Event) -> None:
        """Handle load data request."""
        path = event.data.get("path")
        if path:
            self._handle_load_data(path)

    def _on_export_requested(self, event: Event) -> None:
        """Handle export request."""
        self._handle_export_ply()

    def _on_reset_colors_requested(self, event: Event) -> None:
        """Handle reset colors request."""
        self._handle_color_reset()

    def _on_reset_transform_requested(self, event: Event) -> None:
        """Handle reset transform request."""
        self._handle_pose_reset()

    def _on_reset_filter_requested(self, event: Event) -> None:
        """Handle reset filter request."""
        self._handle_filter_reset()

    def _on_center_requested(self, event: Event) -> None:
        """Handle center request - center scene at origin."""
        self._handle_center_scene()

    def _on_rerender_requested(self, event: Event) -> None:
        """Handle rerender request."""
        # Update edit history first (captures current UI state)
        self._update_edit_history()

        if self.viewer:
            self.render_component.rerender()

    def _on_terminate_requested(self, event: Event) -> None:
        """Handle terminate request - gracefully stop the viewer."""
        logger.info("Terminate requested via UI")
        if self.playback_controller:
            self.playback_controller.stop()

    def _setup_layer_controls(self) -> None:
        """Setup layer management UI if model supports layers."""
        # Use protocol-based check instead of isinstance
        if hasattr(self.model, "get_layer_info") and hasattr(
            self.model, "set_layer_visibility"
        ):
            logger.info("Setting up layer controls for multi-layer model")
            layer_controls = create_layer_controls(self.server, self.model)
            logger.info(f"Layer controls created: {list(layer_controls.keys())}")
        else:
            logger.debug("Model does not support layers, skipping layer controls")

    def _setup_filter_visualizer(self) -> None:
        """Setup filter visualization gizmos and callbacks."""
        # Create filter visualizer
        self.filter_visualizer = FilterVisualizer(self.server)
        logger.debug("Filter visualizer created")

        if not self.ui:
            return

        # Callback to update visualization when show checkbox changes
        def on_show_filter_viz_change(_) -> None:
            if self.filter_visualizer and self.ui and self.ui.show_filter_viz:
                self.filter_visualizer.visible = self.ui.show_filter_viz.value
                # Update with current filter values
                self._update_filter_visualization()

        # Callback to update visualization and config when filter parameters change
        def on_filter_change(_) -> None:
            self._update_filter_visualization()
            # Also sync filter_values to config so filter uses updated rotation values
            camera_pos, camera_rot = self._get_camera_state()
            self.config.filter_values = self.ui.get_filter_values(
                camera_position=camera_pos,
                camera_rotation=camera_rot,
            )

        # Register show/hide callback
        if self.ui.show_filter_viz:
            self.ui.show_filter_viz.on_update(on_show_filter_viz_change)

        # Register filter type change callback
        if self.ui.spatial_filter_type:
            self.ui.spatial_filter_type.on_update(on_filter_change)

        # Register callbacks for sphere filter
        for control in [
            self.ui.sphere_center_x,
            self.ui.sphere_center_y,
            self.ui.sphere_center_z,
            self.ui.sphere_radius,
        ]:
            if control:
                control.on_update(on_filter_change)

        # Register callbacks for box filter
        for control in [
            self.ui.box_min_x,
            self.ui.box_min_y,
            self.ui.box_min_z,
            self.ui.box_max_x,
            self.ui.box_max_y,
            self.ui.box_max_z,
            self.ui.box_rot_x,
            self.ui.box_rot_y,
            self.ui.box_rot_z,
        ]:
            if control:
                control.on_update(on_filter_change)

        # Register callbacks for ellipsoid filter
        for control in [
            self.ui.ellipsoid_center_x,
            self.ui.ellipsoid_center_y,
            self.ui.ellipsoid_center_z,
            self.ui.ellipsoid_radius_x,
            self.ui.ellipsoid_radius_y,
            self.ui.ellipsoid_radius_z,
            self.ui.ellipsoid_rot_x,
            self.ui.ellipsoid_rot_y,
            self.ui.ellipsoid_rot_z,
        ]:
            if control:
                control.on_update(on_filter_change)

        # Register callbacks for frustum filter
        for control in [
            self.ui.frustum_fov,
            self.ui.frustum_aspect,
            self.ui.frustum_near,
            self.ui.frustum_far,
            self.ui.frustum_pos_x,
            self.ui.frustum_pos_y,
            self.ui.frustum_pos_z,
            self.ui.frustum_rot_x,
            self.ui.frustum_rot_y,
            self.ui.frustum_rot_z,
        ]:
            if control:
                control.on_update(on_filter_change)

        # Register "Use Current Camera" button callback
        if self.ui.frustum_use_camera:

            @self.ui.frustum_use_camera.on_click
            def on_use_camera_click(_) -> None:
                self._copy_camera_to_frustum()

        # Callback to update visualization when scene transform changes
        # This ensures the filter visualization moves with the transformed Gaussians
        def on_transform_change(_) -> None:
            self._update_filter_visualization()

        # Register callbacks for scene transformation controls
        for control in [
            getattr(self.ui, 'global_scale', None),
            getattr(self.ui, 'translate_x', None),
            getattr(self.ui, 'translate_y', None),
            getattr(self.ui, 'translate_z', None),
            getattr(self.ui, 'rotate_x', None),
            getattr(self.ui, 'rotate_y', None),
            getattr(self.ui, 'rotate_z', None),
        ]:
            if control:
                control.on_update(on_transform_change)

        logger.debug("Filter visualizer callbacks registered")

    def _setup_auto_learn_color(self) -> None:
        """Setup unified color adjustment callback."""
        if not self.ui:
            return

        # Register "Apply" button callback for unified color adjustment
        if self.ui.apply_adjustment_button:

            @self.ui.apply_adjustment_button.on_click
            def on_apply_adjustment(_) -> None:
                self._apply_color_adjustment()

        logger.debug("Color adjustment callback registered")

    def _setup_export_handlers(self) -> None:
        """Setup handlers for export format/device changes.

        Updates the export path automatically when options change,
        unless user has manually edited the path.
        """
        if not self.ui:
            return

        def on_export_option_change(_) -> None:
            self._update_export_path_on_option_change()

        # Register format change handler
        if self.ui.export_format:
            self.ui.export_format.on_update(on_export_option_change)

        # Register device change handler
        if self.ui.export_device:
            self.ui.export_device.on_update(on_export_option_change)

        logger.debug("Export option handlers registered")

    def _apply_color_adjustment(self) -> None:
        """Apply selected color adjustment from unified dropdown.

        Handles three categories:
        - correction: gsmod 0.1.4 auto-correction (auto_enhance, auto_contrast, etc.)
        - stylize: preset color profiles (vibrant, dramatic, etc.)
        - advanced: legacy histogram learning (auto_fit basic/standard/full)
        """
        if not self.model:
            logger.warning("No model loaded")
            return

        try:
            from gsmod import ColorValues
            from src.gsplay.core.handlers.color_presets import (
                get_adjustment_type,
                get_preset_values,
            )

            # Get selected option from dropdown
            option = "Auto Enhance"
            if self.ui and self.ui.color_adjustment_dropdown:
                option = self.ui.color_adjustment_dropdown.value

            category, key = get_adjustment_type(option)

            # Get current frame gaussians
            gaussians = self._get_current_frame_gaussians()
            if gaussians is None:
                logger.warning("Could not get frame data")
                return

            if category == "correction":
                # gsmod 0.1.4 auto-correction functions
                from src.gsplay.core.handlers.auto_correction import (
                    apply_auto_correction,
                )

                color_values = apply_auto_correction(gaussians, key)

            elif category == "stylize":
                # Style preset (direct preset values)
                color_values = get_preset_values(key)

            elif category == "advanced":
                # Legacy histogram learning - get colors tensor
                colors = self._get_colors_from_gaussians(gaussians)
                if colors is None:
                    logger.warning("Could not extract colors")
                    return
                color_values = self._histogram_learn(colors, key)

            else:
                color_values = ColorValues()

            # Update UI sliders with computed values
            if self.ui:
                self.ui.set_color_values(color_values)

            # Update config
            self.config.color_values = color_values

            # Trigger rerender
            if self.viewer:
                self.render_component.rerender()

            logger.info(
                f"Applied '{option}': brightness={color_values.brightness:.3f}, "
                f"contrast={color_values.contrast:.3f}, gamma={color_values.gamma:.3f}"
            )

        except ImportError as e:
            logger.error(f"gsmod 0.1.4 auto-correction not available: {e}")
        except Exception as e:
            logger.error(f"Failed to apply color adjustment: {e}", exc_info=True)

    def _get_current_frame_gaussians(self) -> "GSTensor | None":
        """Get GSTensor for current frame."""
        frame_idx = 0
        if self.ui and self.ui.time_slider:
            frame_idx = int(self.ui.time_slider.value)

        total_frames = self.model.get_total_frames()
        normalized_time = (
            frame_idx / max(1, total_frames - 1) if total_frames > 1 else 0.0
        )

        return self.model.get_gaussians_at_normalized_time(normalized_time)

    def _get_colors_from_gaussians(
        self, gaussians: "GSTensor | None"
    ) -> "torch.Tensor | None":
        """Extract SH0 colors from gaussians as GPU tensor."""
        import torch

        if gaussians is None or not hasattr(gaussians, "sh0"):
            return None

        sh0 = gaussians.sh0
        if not isinstance(sh0, torch.Tensor):
            sh0 = torch.tensor(sh0, dtype=torch.float32)

        return sh0.to(self.device)

    # Histogram learning configuration by level
    # NOTE: saturation/vibrance excluded - causes grayscale (degenerate solution)
    _HISTOGRAM_LEARN_CONFIG: dict[str, tuple[list[str], int, float]] = {
        "basic": (["brightness", "contrast", "gamma"], 100, 0.02),
        "standard": (
            ["brightness", "contrast", "gamma", "temperature", "tint"],
            150,
            0.02,
        ),
        "full": (
            [
                "brightness",
                "contrast",
                "gamma",
                "temperature",
                "tint",
                "shadows",
                "highlights",
                "fade",
            ],
            200,
            0.015,
        ),
    }

    def _histogram_learn(self, colors: "torch.Tensor", level: str) -> "ColorValues":
        """Legacy histogram learning.

        Parameters
        ----------
        colors : torch.Tensor
            Source SH0 colors (N, 3)
        level : str
            One of "basic", "standard", "full"

        Returns
        -------
        ColorValues
            Learned normalization parameters
        """
        from gsmod.histogram.result import HistogramResult
        import numpy as np

        # Get config for level (default to standard)
        norm_params, n_epochs, lr = self._HISTOGRAM_LEARN_CONFIG.get(
            level, self._HISTOGRAM_LEARN_CONFIG["standard"]
        )

        # Create neutral target histogram
        # Target: mean=0.5, std=0.289 (uniform distribution stats)
        neutral_target = HistogramResult(
            counts=np.ones((3, 64), dtype=np.int64),
            bin_edges=np.linspace(0, 1, 65),
            mean=np.array([0.5, 0.5, 0.5]),
            std=np.array([0.289, 0.289, 0.289]),
            min_val=np.array([0.0, 0.0, 0.0]),
            max_val=np.array([1.0, 1.0, 1.0]),
            n_samples=1000,
        )

        return neutral_target.learn_from(
            colors,
            params=norm_params,
            n_epochs=n_epochs,
            lr=lr,
            verbose=False,
        )

    def _update_filter_visualization(self) -> None:
        """Update filter visualization based on current UI state.

        The visualization is transformed by the scene transformation so it
        appears in the correct position relative to the transformed Gaussians.
        """
        if not self.filter_visualizer or not self.ui:
            return

        # Get current filter type
        filter_type = (
            self.ui.spatial_filter_type.value if self.ui.spatial_filter_type else "None"
        )

        # Get current filter values from UI
        filter_values = self.ui.get_filter_values()

        # Get current transform values to apply to visualization
        transform_values = self.ui.get_transform_values()

        # Update visualizer with both filter and transform values
        self.filter_visualizer.update(filter_type, filter_values, transform_values)

    def _copy_camera_to_frustum(self) -> None:
        """Copy current camera position/rotation to frustum UI controls.

        The camera operates in transformed (viewer) space, but the frustum filter
        operates on original Gaussian positions. This method applies the inverse
        scene transformation to convert camera coordinates to world space.
        """
        if not self.ui:
            return

        camera_pos, camera_rot = self._get_camera_state()
        if not camera_pos or not camera_rot:
            return

        # Get current scene transformation
        transform_values = self.ui.get_transform_values()

        # Apply inverse scene transformation to camera position/rotation
        frustum_pos, frustum_euler_deg = self._inverse_transform_camera(
            camera_pos, camera_rot, transform_values
        )

        # Set frustum position
        if self.ui.frustum_pos_x:
            self.ui.frustum_pos_x.value = frustum_pos[0]
            if self.ui.frustum_pos_y:
                self.ui.frustum_pos_y.value = frustum_pos[1]
            if self.ui.frustum_pos_z:
                self.ui.frustum_pos_z.value = frustum_pos[2]

        # Set frustum rotation (already in degrees)
        if self.ui.frustum_rot_x:
            self.ui.frustum_rot_x.value = float(frustum_euler_deg[0])
            if self.ui.frustum_rot_y:
                self.ui.frustum_rot_y.value = float(frustum_euler_deg[1])
            if self.ui.frustum_rot_z:
                self.ui.frustum_rot_z.value = float(frustum_euler_deg[2])

        # Trigger visualization update
        self._update_filter_visualization()
        logger.debug("Copied camera state to frustum filter (with inverse transform)")

    def _inverse_transform_camera(
        self,
        camera_pos: tuple[float, float, float],
        camera_rot: tuple[float, float, float, float],
        transform_values,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Apply inverse scene transformation to camera and convert to frustum frame.

        The scene transformation is applied to Gaussians as: scale -> rotate -> translate
        So the inverse is: inverse_translate -> inverse_rotate -> inverse_scale
        """
        import numpy as np
        from src.gsplay.config.ui_handles import _camera_to_frustum_euler_deg

        pos = np.array(camera_pos, dtype=np.float64)

        # Check if transform is neutral
        if transform_values is None or (
            hasattr(transform_values, "is_neutral") and transform_values.is_neutral()
        ):
            euler_deg = _camera_to_frustum_euler_deg(camera_rot)
            return tuple(pos), euler_deg

        # Get transform components
        translation = np.array(
            getattr(transform_values, "translation", (0.0, 0.0, 0.0)), dtype=np.float64
        )
        scale = float(getattr(transform_values, "scale", 1.0))
        scene_rot_quat = getattr(transform_values, "rotation", (1.0, 0.0, 0.0, 0.0))

        # Build scene rotation matrix from quaternion
        w, x, y, z = scene_rot_quat
        R_scene = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
        ], dtype=np.float64)

        # Inverse transform position: pos_world = ((pos_viewer - trans) @ R_scene) / scale
        pos_world = ((pos - translation) @ R_scene) / scale

        # Build camera rotation matrix from quaternion
        cw, cx, cy, cz = camera_rot
        R_cam = np.array([
            [1 - 2*(cy*cy + cz*cz), 2*(cx*cy - cw*cz), 2*(cx*cz + cw*cy)],
            [2*(cx*cy + cw*cz), 1 - 2*(cx*cx + cz*cz), 2*(cy*cz - cw*cx)],
            [2*(cx*cz - cw*cy), 2*(cy*cz + cw*cx), 1 - 2*(cx*cx + cy*cy)],
        ], dtype=np.float64)

        # Compose: R_world = R_scene @ R_cam
        R_world = R_scene @ R_cam

        # Convert R_world to quaternion using Shepperd's method
        trace = R_world[0, 0] + R_world[1, 1] + R_world[2, 2]
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            qw, qx = 0.25 / s, (R_world[2, 1] - R_world[1, 2]) * s
            qy, qz = (R_world[0, 2] - R_world[2, 0]) * s, (R_world[1, 0] - R_world[0, 1]) * s
        elif R_world[0, 0] > R_world[1, 1] and R_world[0, 0] > R_world[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R_world[0, 0] - R_world[1, 1] - R_world[2, 2])
            qw, qx = (R_world[2, 1] - R_world[1, 2]) / s, 0.25 * s
            qy, qz = (R_world[0, 1] + R_world[1, 0]) / s, (R_world[0, 2] + R_world[2, 0]) / s
        elif R_world[1, 1] > R_world[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R_world[1, 1] - R_world[0, 0] - R_world[2, 2])
            qw, qx = (R_world[0, 2] - R_world[2, 0]) / s, (R_world[0, 1] + R_world[1, 0]) / s
            qy, qz = 0.25 * s, (R_world[1, 2] + R_world[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R_world[2, 2] - R_world[0, 0] - R_world[1, 1])
            qw, qx = (R_world[1, 0] - R_world[0, 1]) / s, (R_world[0, 2] + R_world[2, 0]) / s
            qy, qz = (R_world[1, 2] + R_world[2, 1]) / s, 0.25 * s

        # Normalize and convert to frustum frame
        norm = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
        world_rot_quat = (qw/norm, qx/norm, qy/norm, qz/norm)
        euler_deg = _camera_to_frustum_euler_deg(world_rot_quat)

        return tuple(float(x) for x in pos_world), euler_deg

    def _get_camera_state(
        self,
    ) -> tuple[
        tuple[float, float, float] | None,
        tuple[float, float, float, float] | None,
    ]:
        """Get current camera position and rotation from viser.

        Returns
        -------
        tuple
            (camera_position, camera_rotation) where:
            - camera_position is (x, y, z) or None
            - camera_rotation is quaternion (w, x, y, z) or None
        """
        try:
            clients = list(self.server.get_clients().values())
            if not clients:
                return None, None
            client = clients[0]
            camera = client.camera
            position = tuple(float(x) for x in camera.position)
            rotation = tuple(float(x) for x in camera.wxyz)  # w, x, y, z
            return position, rotation
        except Exception:
            return None, None

    def _update_edit_history(self) -> None:
        """Update edit settings from current UI state."""
        if not self.ui:
            return

        # Update config from UI
        self.config.color_values = self.ui.get_color_values()
        self.config.alpha_scaler = self.ui.get_alpha_scaler()
        self.config.transform_values = self.ui.get_transform_values()

        # Update filter values from UI with camera state for frustum filter
        camera_pos, camera_rot = self._get_camera_state()
        self.config.filter_values = self.ui.get_filter_values(
            camera_position=camera_pos,
            camera_rotation=camera_rot,
        )

        # Sync processing mode from dropdown
        if self.ui.processing_mode_dropdown:
            from src.infrastructure.processing_mode import ProcessingMode

            try:
                mode = ProcessingMode.from_string(
                    self.ui.processing_mode_dropdown.value
                )
                self.config.processing_mode = mode.value
                self.config.volume_filter.processing_mode = mode.value
            except (ValueError, AttributeError) as e:
                logger.warning(f"Invalid processing mode in UI: {e}")

        if self.ui.use_cpu_filtering_checkbox:
            self.config.volume_filter.use_cpu_filtering = (
                self.ui.use_cpu_filtering_checkbox.value
            )

        # Check if any edits are active (use filter_values, not volume_filter)
        from src.gsplay.processing.volume_filter import is_filter_active

        self.config.edits_active = (
            not self.config.color_values.is_neutral()
            or not self.config.transform_values.is_neutral()
            or self.config.alpha_scaler != 1.0
            or is_filter_active(self.config.filter_values)
        )

    def _handle_export_ply(self) -> None:
        """Handle frame sequence export using ExportComponent."""
        if not self.model:
            logger.error("No model loaded to export")
            return

        # Turn off auto-play when export starts
        if self.playback_controller:
            self.playback_controller.pause()
        if self.config.animation.auto_play:
            self.config.animation.auto_play = False
        if self.ui and self.ui.auto_play:
            self.ui.auto_play.value = (
                "Pause"  # "Pause" when paused, " Play" when playing
            )

        # Build format map from registry (display name -> registry key)
        from src.infrastructure.registry import register_defaults, DataSinkRegistry

        register_defaults()

        format_map = {}
        for key in DataSinkRegistry.names():
            sink_class = DataSinkRegistry.get(key)
            if sink_class:
                try:
                    meta = sink_class.metadata()
                    format_map[meta.name.lower()] = key
                except Exception:
                    format_map[key.lower()] = key

        # Get export format from UI or config
        export_format = "compressed-ply"  # Default
        if self.ui and self.ui.export_format:
            ui_value = self.ui.export_format.value.lower()
            export_format = format_map.get(ui_value, ui_value)
        elif hasattr(self.config.export_settings, "export_format"):
            export_format = self.config.export_settings.export_format.lower()

        # Get export device from UI or config
        export_device = "cpu"  # Default to CPU for safety
        if self.ui and self.ui.export_device:
            device_value = self.ui.export_device.value.upper()
            if device_value == "GPU":
                import torch

                if torch.cuda.is_available():
                    export_device = "cuda:0"  # Use first GPU
                else:
                    logger.warning(
                        "GPU requested but not available, falling back to CPU"
                    )
                    export_device = "cpu"
            else:
                export_device = "cpu"
        elif hasattr(self.config.export_settings, "export_device"):
            export_device = self.config.export_settings.export_device

        logger.info(f"Starting {export_format.upper()} export on {export_device}...")

        try:
            # Get export path from UI if set, otherwise use default
            if self.ui and self.ui.export_path and self.ui.export_path.value.strip():
                output_dir = UniversalPath(self.ui.export_path.value.strip())
            else:
                # Fallback to default if UI path is empty
                output_dir = self._build_default_export_dir(export_format)
                # Only update UI if path was empty (don't overwrite user's choice)
                if self.ui and self.ui.export_path:
                    self.ui.export_path.value = str(output_dir)

            logger.info(f"Exporting to: {output_dir}")

            # Create wrapper for export with scene bounds
            # Use export device for edits to avoid redundant transfers
            from src.gsplay.core.container import create_edit_manager

            export_edit_manager = create_edit_manager(self.config, export_device)

            def apply_edits_for_export(gaussians: GSTensor) -> GSTensor:
                return export_edit_manager.apply_edits(
                    gaussians, scene_bounds=self.scene_bounds_manager.get_bounds()
                )

            # Create export settings
            from src.gsplay.config.settings import ExportSettings

            export_settings = ExportSettings(
                export_format=export_format,
                export_path=output_dir,
                export_device=export_device,
            )

            # Export using ExportComponent (tqdm progress bar will be shown automatically)
            success = self.export_component.export_frame_sequence(
                model=self.model,
                export_settings=export_settings,
                edit_applier=apply_edits_for_export,
            )

            if success:
                logger.info(f"Export completed successfully to {output_dir}")
            else:
                logger.error("Export failed")

        except ValueError as e:
            logger.error(f"Export configuration error: {e}")
        except Exception as e:
            logger.error(f"Export failed: {e}", exc_info=True)

    def _handle_color_reset(self) -> None:
        """Reset all color adjustments to defaults."""
        logger.info("Resetting color adjustments")

        self.config.color_values = ColorValues()
        self.config.alpha_scaler = 1.0

        if self.ui:
            self.ui.set_color_values(
                self.config.color_values, alpha_scaler=self.config.alpha_scaler
            )

        if self.viewer:
            self.viewer.rerender(None)

    def _handle_pose_reset(self) -> None:
        """Reset all transforms to defaults."""
        logger.info("Resetting transforms")

        self.config.transform_values = TransformValues()

        if self.ui:
            self.ui.set_transform_values(self.config.transform_values)

        if self.viewer:
            self.viewer.rerender(None)

    def _handle_center_scene(self) -> None:
        """Center scene at origin by applying inverse centroid translation.

        Respects active filters - only considers visible/filtered gaussians
        when calculating centroid.
        """
        import numpy as np

        if self.model is None:
            logger.warning("Cannot center scene: no model loaded")
            return

        # Get current frame gaussians
        gaussians = self._get_current_frame_gaussians()
        if gaussians is None:
            logger.warning("Cannot center scene: no gaussians available")
            return

        # Get means as numpy array
        means = gaussians.means
        if hasattr(means, "detach"):
            means = means.detach().cpu().numpy()
        elif not isinstance(means, np.ndarray):
            means = np.array(means)

        total_count = len(means)

        # Apply filter mask if filtering is active
        fv = self.config.filter_values
        if not fv.is_neutral():
            try:
                from gsmod.filter.apply import compute_filter_mask

                # Create GSData-like object for compute_filter_mask
                # It needs means, scales, opacities attributes
                class _GaussianData:
                    pass

                data = _GaussianData()
                data.means = means

                # Get scales and opacities for filtering
                scales = gaussians.scales
                if hasattr(scales, "detach"):
                    scales = scales.detach().cpu().numpy()
                elif not isinstance(scales, np.ndarray):
                    scales = np.array(scales)
                data.scales = scales

                opacities = gaussians.opacities
                if hasattr(opacities, "detach"):
                    opacities = opacities.detach().cpu().numpy()
                elif not isinstance(opacities, np.ndarray):
                    opacities = np.array(opacities)
                data.opacities = opacities

                # Compute filter mask
                mask = compute_filter_mask(data, fv)
                means = means[mask]

                logger.info(
                    f"Centering on {len(means)}/{total_count} filtered gaussians "
                    f"({len(means)/total_count*100:.1f}%)"
                )
            except ImportError:
                logger.warning("gsmod filter module unavailable, centering on all gaussians")
            except Exception as e:
                logger.warning(f"Filter mask computation failed: {e}, centering on all gaussians")

        if len(means) == 0:
            logger.warning("Cannot center scene: no gaussians after filtering")
            return

        # Calculate centroid
        centroid = np.mean(means, axis=0)
        logger.info(f"Scene centroid: [{centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f}]")

        # Apply inverse translation to center at origin
        # gsmod formula: new_pos = old_pos + translate
        # To move centroid to origin: 0 = centroid + translate -> translate = -centroid
        tx, ty, tz = -float(centroid[0]), -float(centroid[1]), -float(centroid[2])
        self.api.set_translation(tx, ty, tz)

        # Trigger rerender
        if self.viewer:
            self.render_component.rerender()

        logger.info(f"Scene centered: translation=[{tx:.3f}, {ty:.3f}, {tz:.3f}]")

    def _handle_filter_reset(self) -> None:
        """Reset all filtering to defaults."""
        logger.info("Resetting volume filters")

        # Reset to default FilterValues and VolumeFilter
        self.config.filter_values = FilterValues()
        self.config.volume_filter = VolumeFilter()

        if self.ui:
            # Reset spatial filter type
            if self.ui.spatial_filter_type:
                self.ui.spatial_filter_type.value = "None"
            # Reset opacity/scale sliders
            if self.ui.min_opacity_slider:
                self.ui.min_opacity_slider.value = 0.0
            if self.ui.max_opacity_slider:
                self.ui.max_opacity_slider.value = 1.0
            if self.ui.min_scale_slider:
                self.ui.min_scale_slider.value = 0.0
            if self.ui.max_scale_slider:
                self.ui.max_scale_slider.value = 100.0

        if self.viewer:
            self.viewer.rerender(None)

    # =========================================================================
    # GaussianData API (New Unified Data IO)
    # =========================================================================

    def get_current_frame_as_gaussian_data(self):
        """Get current frame as GaussianData.

        Returns the current frame using the new unified GaussianData abstraction.
        This is the preferred way to get frame data for export or processing.

        Returns
        -------
        GaussianData | None
            Current frame as GaussianData, or None if no model loaded
        """
        from src.domain.data import GaussianData

        if not self.model:
            return None

        # Get current frame index
        frame_idx = 0
        if self.ui and self.ui.time_slider:
            frame_idx = int(self.ui.time_slider.value)

        # Check if model is a DataSourceProtocol (new API)
        if hasattr(self.model, "get_frame"):
            return self.model.get_frame(frame_idx)

        # Fall back to legacy API
        total_frames = self.model.get_total_frames()
        normalized_time = (
            frame_idx / max(1, total_frames - 1) if total_frames > 1 else 0.0
        )
        gaussians = self.model.get_gaussians_at_normalized_time(normalized_time)

        if gaussians is None:
            return None

        # Convert to GaussianData
        from gsply import GSData
        from gsply.torch import GSTensor as GSPlyTensor

        if isinstance(gaussians, GSPlyTensor):
            return GaussianData.from_gstensor(gaussians)
        elif isinstance(gaussians, GSData):
            return GaussianData.from_gsdata(gaussians)
        else:
            logger.warning(f"Unknown gaussians type: {type(gaussians)}")
            return None

    def export_current_frame_as_gaussian_data(
        self,
        output_path: str,
        sink_format: str = "ply",
        apply_edits: bool = True,
        **options,
    ) -> bool:
        """Export current frame using GaussianData abstraction.

        Parameters
        ----------
        output_path : str
            Output file path
        sink_format : str
            Sink format name (e.g., "ply", "compressed-ply")
        apply_edits : bool
            Whether to apply current edits before export
        **options
            Format-specific export options

        Returns
        -------
        bool
            True if export succeeded
        """
        data = self.get_current_frame_as_gaussian_data()
        if data is None:
            logger.error("No frame data available")
            return False

        # Apply edits if requested
        if apply_edits:
            from src.gsplay.processing.gs_bridge import DefaultGSBridge

            bridge = DefaultGSBridge()

            # Convert to GSTensorPro for processing
            tensor_pro, _ = bridge.gaussian_data_to_gstensor_pro(data, self.device)

            # Apply edits
            tensor_pro = self.edit_manager.apply_edits(
                tensor_pro, scene_bounds=self.scene_bounds_manager.get_bounds()
            )

            # Convert back to GaussianData
            data = bridge.gstensor_pro_to_gaussian_data(tensor_pro)

        # Export using ExportComponent
        return self.export_component.export_gaussian_data(
            data, output_path, sink_format, **options
        )

    def get_data_source(self) -> DataSourceProtocol | None:
        """Get the model as a DataSourceProtocol if it supports the interface.

        Returns
        -------
        DataSourceProtocol | None
            The model as DataSourceProtocol, or None if not supported
        """
        if self.model is None:
            return None

        # Check if model implements DataSourceProtocol methods
        if hasattr(self.model, "get_frame") and hasattr(self.model, "total_frames"):
            return self.model

        return None

    def _handle_load_data(self, path: str) -> None:
        """
        Load new data from specified path using ModelComponent.

        Parameters
        ----------
        path : str
            Path to PLY folder or JSON config file
        """
        logger.info(f"Loading data from: {path}")

        try:
            # Use ModelComponent to load from path
            # This will emit MODEL_LOADED event, which triggers _on_model_loaded
            # and UIController._on_model_loaded
            model, data_loader, metadata = self.model_component.load_from_path(path)

            # Track current config/model path for future sessions
            self.config.model_config_path = UniversalPath(path)

            # Update export component with source path
            if self.model_component.get_source_path():
                self.export_component.set_default_output_dir(
                    self.model_component.get_source_path()
                )
            self._initialize_export_path(force=True)

            logger.info(f"Successfully loaded data from: {path}")

        except Exception as e:
            logger.error(f"Failed to load data: {e}", exc_info=True)

    def run(self) -> None:
        """Run the main viewer loop."""
        import time
        import warnings

        logger.info(f"GSPlay running on http://{self.config.host}:{self.config.port}")

        try:
            # Use PlaybackController's loop instead of the old run_autoplay_loop
            self.playback_controller.run_loop()
        except KeyboardInterrupt:
            logger.info("GSPlay stopped by user")
        finally:
            logger.info("Starting graceful shutdown...")

            # Stop playback loop
            self.playback_controller.stop()

            # Suppress websocket shutdown warnings (harmless during cleanup)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            websocket_logger = logging.getLogger("websockets.server")
            original_level = websocket_logger.level
            websocket_logger.setLevel(logging.CRITICAL)

            # Cleanup handlers
            if self.handlers:
                try:
                    self.handlers.cleanup()
                    logger.debug("Handlers cleaned up")
                except Exception as e:
                    logger.debug(f"Handler cleanup error (ignored): {e}")

            # Cleanup UI controller
            if self.ui_controller:
                self.ui_controller.cleanup()

            # Stop stream server
            if self.stream_server:
                try:
                    self.stream_server.stop()
                    logger.debug("Stream server stopped")
                except Exception as e:
                    logger.debug(f"Stream server cleanup error (ignored): {e}")

            # Stop control server
            if self.control_server:
                try:
                    self.control_server.stop()
                    logger.debug("Control server stopped")
                except Exception as e:
                    logger.debug(f"Control server cleanup error (ignored): {e}")

            # Give websocket connections time to close gracefully
            # This prevents "cannot schedule new futures after shutdown" errors
            time.sleep(0.5)

            # Restore websocket logging
            websocket_logger.setLevel(original_level)

            logger.info("GSPlay shutdown complete")
