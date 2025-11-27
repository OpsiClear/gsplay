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

    def _build_default_export_dir(self, export_format: str) -> UniversalPath:
        """Create timestamped export path rooted at the source path (or CWD)."""
        base_dir = (
            UniversalPath(self.source_path) if self.source_path else UniversalPath(".")
        )
        timestamp = datetime.now().strftime("%Y%m%d")
        return base_dir / f"{timestamp}_{export_format}"

    def _set_export_path(self, export_path: UniversalPath | str) -> None:
        """Store export path on config and sync UI control if it exists."""
        resolved_path = UniversalPath(export_path)
        self.config.export_settings.export_path = resolved_path
        if self.ui and self.ui.export_path:
            self.ui.export_path.value = str(resolved_path)

    def _initialize_export_path(
        self,
        export_format: str | None = None,
        *,
        force: bool = False,
    ) -> UniversalPath:
        """Ensure the export path has a user-visible default."""
        if export_format is None:
            export_format = self.config.export_settings.export_format.lower()

        if not force and not self._is_export_path_placeholder():
            return self.config.export_settings.export_path

        export_path = self._build_default_export_dir(export_format)
        self._set_export_path(export_path)
        return export_path

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

        # Configure viser theme based on compact_ui setting
        compact_ui = getattr(self.config, "compact_ui", False)
        self.server.gui.configure_theme(
            control_layout="floating" if compact_ui else "collapsible",
            control_width="small" if compact_ui else "medium",
            dark_mode=True,
            brand_color=(100, 180, 255),
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

        # Initialize programmatic API
        self.api = GSPlayAPI(self)
        logger.debug("Programmatic API initialized")

        # Start stream server if configured
        self._start_stream_server()

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
        """Setup auto-fit color callback."""
        if not self.ui:
            return

        # Register "Auto Fit" button callback
        if self.ui.auto_fit_button:

            @self.ui.auto_fit_button.on_click
            def on_auto_fit(_) -> None:
                self._auto_fit_colors()

        logger.debug("Auto-fit color callback registered")

    def _auto_fit_colors(self) -> None:
        """Fit color parameters to selected target profile.

        Two-stage approach:
        1. Learn normalization params to bring colors to neutral baseline
        2. Compose with preset style adjustments
        """
        if not self.model:
            logger.warning("No model loaded")
            return

        try:
            import torch
            from gsmod import ColorValues
            from gsmod.histogram.result import HistogramResult
            import numpy as np

            # Get selected profile
            profile = "neutral"
            if self.ui and self.ui.color_profile_dropdown:
                profile = self.ui.color_profile_dropdown.value.lower()

            # Get current frame data
            frame_idx = 0
            if self.ui and self.ui.time_slider:
                frame_idx = int(self.ui.time_slider.value)

            # Convert frame index to normalized time
            total_frames = self.model.get_total_frames()
            normalized_time = (
                frame_idx / max(1, total_frames - 1) if total_frames > 1 else 0.0
            )

            gaussians = self.model.get_gaussians_at_normalized_time(normalized_time)
            if gaussians is None:
                logger.warning("Could not get frame data")
                return

            # Get source colors as tensor
            if hasattr(gaussians, "sh0"):
                source_colors = gaussians.sh0
                if not isinstance(source_colors, torch.Tensor):
                    source_colors = torch.tensor(source_colors, dtype=torch.float32)
            else:
                source_colors = torch.tensor(gaussians.sh0, dtype=torch.float32)

            source_colors = source_colors.to(self.device)

            # Stage 1: Create neutral target histogram and learn normalization
            # Target: mean=0.5, std=0.289 (uniform distribution stats)
            neutral_target = HistogramResult(
                counts=np.ones((3, 64), dtype=np.int64),  # Uniform
                bin_edges=np.linspace(0, 1, 65),
                mean=np.array([0.5, 0.5, 0.5]),
                std=np.array([0.289, 0.289, 0.289]),  # std of uniform [0,1]
                min_val=np.array([0.0, 0.0, 0.0]),
                max_val=np.array([1.0, 1.0, 1.0]),
                n_samples=1000,
            )

            # Get learn level
            learn_level = "standard"
            if self.ui and self.ui.learn_level_dropdown:
                learn_level = self.ui.learn_level_dropdown.value.lower()

            # Select parameters based on learn level
            # NOTE: saturation/vibrance excluded - causes grayscale (degenerate solution)
            # These are stylistic params, not normalization params
            if learn_level == "basic":
                # 3 params - fast, core tonal only
                norm_params = ["brightness", "contrast", "gamma"]
                n_epochs, lr = 100, 0.02
            elif learn_level == "full":
                # 8 params - tonal + white balance + range
                norm_params = [
                    "brightness",
                    "contrast",
                    "gamma",  # Core tonal
                    "temperature",
                    "tint",  # White balance
                    "shadows",
                    "highlights",  # Tonal range
                    "fade",  # Lifted blacks
                ]
                n_epochs, lr = 200, 0.015
            else:  # standard
                # 5 params - tonal + white balance
                norm_params = [
                    "brightness",
                    "contrast",
                    "gamma",  # Core tonal
                    "temperature",
                    "tint",  # White balance
                ]
                n_epochs, lr = 150, 0.02

            norm_values = neutral_target.learn_from(
                source_colors,
                params=norm_params,
                n_epochs=n_epochs,
                lr=lr,
                verbose=False,
            )

            # Stage 2: Get preset style adjustments
            # For "neutral" profile, just use normalization
            if profile == "neutral":
                color_values = norm_values
            else:
                # Get preset adjustments (these are relative to neutral)
                preset_values = self._get_preset_values(profile)

                # Compose: normalization + preset
                color_values = self._compose_color_values(norm_values, preset_values)

            # Update UI sliders with fitted values
            if self.ui:
                self.ui.set_color_values(color_values)

            # Update config
            self.config.color_values = color_values

            # Trigger rerender
            if self.viewer:
                self.render_component.rerender()

            logger.info(
                f"Auto-fit to '{profile}': brightness={color_values.brightness:.3f}, "
                f"contrast={color_values.contrast:.3f}, gamma={color_values.gamma:.3f}, "
                f"saturation={color_values.saturation:.3f}"
            )

        except Exception as e:
            logger.error(f"Failed to auto-fit colors: {e}", exc_info=True)

    def _get_preset_values(self, profile: str) -> ColorValues:
        """Get preset color adjustments for a profile (relative to neutral)."""
        from src.gsplay.core.handlers.color_presets import get_preset_values

        return get_preset_values(profile)

    def _compose_color_values(
        self, base: ColorValues, style: ColorValues
    ) -> ColorValues:
        """Compose two ColorValues: apply base normalization, then style adjustments."""
        from src.gsplay.core.handlers.color_presets import compose_color_values

        return compose_color_values(base, style)

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

            # Give websocket connections time to close gracefully
            # This prevents "cannot schedule new futures after shutdown" errors
            time.sleep(0.5)

            # Restore websocket logging
            websocket_logger.setLevel(original_level)

            logger.info("GSPlay shutdown complete")
