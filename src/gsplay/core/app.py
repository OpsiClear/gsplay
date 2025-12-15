"""
Main viewer application class.

This module provides the UniversalGSPlay class that orchestrates
all viewer components. Supports local filesystem and cloud storage.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import viser
from gsmod import ColorValues, TransformValues

if TYPE_CHECKING:
    import numpy as np
    from src.gsplay.ui.filter_visualizer import FilterVisualizer

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
from src.gsplay.state.scene_bounds_manager import SceneBoundsManager
from src.gsplay.initialization.ui_setup import UISetup
from src.gsplay.rendering.camera import create_supersplat_camera_controls
from src.gsplay.ui.layout import setup_ui_layout
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

        # Set rerender callback so camera presets trigger immediate re-render
        self.camera_controller.set_rerender_callback(
            lambda: self.render_component.rerender() if self.viewer else None
        )

        # NOTE: Rotation rendering is now decoupled from CameraController.
        # SharedRenderer polls CameraController._state directly during rotation,
        # so no render callback is needed. This keeps rotation logic clean -
        # it only updates camera state, not rendering.

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
            EventType.BAKE_VIEW_REQUESTED, self._on_bake_view_requested
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

        # Set universal_viewer on GSPlay for rotation coordination.
        # SharedRenderer polls CameraController._state directly during rotation,
        # and viewer.py blocks on_update to prevent conflicts.
        # MUST be done AFTER setup_viewer() which creates the GSPlay instance.
        viewer = self.render_component.get_viewer()
        if viewer is not None:
            viewer.universal_viewer = self

        # Configure render quality
        self.render_component.configure_quality(self.ui)

        # Set viewer in handlers
        self.handlers.set_viewer(self.viewer)

        # Setup UI components using UISetup helper
        ui_setup = UISetup(self)
        self.filter_visualizer = ui_setup.setup_all()

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

    def _on_bake_view_requested(self, event: Event) -> None:
        """Handle bake view request - bake camera view into model transform."""
        self._bake_camera_view()

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

    def _use_scene_center_for_filter(self) -> None:
        """Set filter center to mean of current frame's Gaussian positions.

        Computes the centroid of all Gaussian positions in the current frame
        and uses that as the center for the active spatial filter.
        """
        if not self.ui:
            return

        # Get current filter type
        filter_type = (
            self.ui.spatial_filter_type.value if self.ui.spatial_filter_type else "None"
        )
        if filter_type not in ("Sphere", "Box", "Ellipsoid"):
            logger.debug(f"Filter type {filter_type} doesn't support scene center")
            return

        # Get current frame's Gaussians
        gaussians = self._get_current_frame_gaussians()
        if gaussians is None or gaussians.means is None:
            logger.warning("No Gaussian data available to compute scene center")
            return

        # Compute mean position (centroid)
        means = gaussians.means
        if hasattr(means, 'cpu'):
            # PyTorch tensor
            center = means.mean(dim=0).cpu().numpy()
        else:
            # NumPy array
            center = means.mean(axis=0)

        center_x, center_y, center_z = float(center[0]), float(center[1]), float(center[2])

        # Set center for the appropriate filter type
        if filter_type == "Sphere":
            if self.ui.sphere_center_x:
                self.ui.sphere_center_x.value = center_x
            if self.ui.sphere_center_y:
                self.ui.sphere_center_y.value = center_y
            if self.ui.sphere_center_z:
                self.ui.sphere_center_z.value = center_z
            logger.debug(f"Set sphere center to scene centroid: ({center_x:.3f}, {center_y:.3f}, {center_z:.3f})")

        elif filter_type == "Box":
            # Box now uses center/size controls - just set center directly
            if self.ui.box_center_x:
                self.ui.box_center_x.value = center_x
            if self.ui.box_center_y:
                self.ui.box_center_y.value = center_y
            if self.ui.box_center_z:
                self.ui.box_center_z.value = center_z
            logger.debug(f"Set box center to scene centroid: ({center_x:.3f}, {center_y:.3f}, {center_z:.3f})")

        elif filter_type == "Ellipsoid":
            if self.ui.ellipsoid_center_x:
                self.ui.ellipsoid_center_x.value = center_x
            if self.ui.ellipsoid_center_y:
                self.ui.ellipsoid_center_y.value = center_y
            if self.ui.ellipsoid_center_z:
                self.ui.ellipsoid_center_z.value = center_z
            logger.debug(f"Set ellipsoid center to scene centroid: ({center_x:.3f}, {center_y:.3f}, {center_z:.3f})")

        # Trigger visualization update
        self._update_filter_visualization()

    def _align_filter_to_camera_up(self) -> None:
        """Align filter rotation so Z-axis points in camera up direction.

        Computes the rotation needed to align the filter's Z-axis with the
        camera's current up direction, similar to bake view logic.
        """
        import numpy as np
        import viser.transforms as vt
        from src.gsplay.config.rotation_conversions import matrix_to_euler_deg

        if not self.ui:
            return

        # Get current filter type
        filter_type = (
            self.ui.spatial_filter_type.value if self.ui.spatial_filter_type else "None"
        )
        if filter_type not in ("Box", "Ellipsoid"):
            logger.debug(f"Filter type {filter_type} doesn't support rotation")
            return

        try:
            # Get camera up direction from viser
            clients = list(self.server.get_clients().values())
            if not clients:
                logger.warning("No clients connected")
                return
            client = clients[0]

            # Get camera rotation matrix from wxyz quaternion
            viser_wxyz = np.array(client.camera.wxyz, dtype=np.float64)
            R_camera = vt.SO3(viser_wxyz).as_matrix()

            # In viser convention, rotation matrix columns are [right, up, forward]
            # R = [right, up, forward]
            camera_right = R_camera[:, 0]
            camera_up = R_camera[:, 1]

            # Normalize (should already be unit vectors, but be safe)
            camera_right = camera_right / np.linalg.norm(camera_right)
            camera_up = camera_up / np.linalg.norm(camera_up)

            logger.debug(f"Camera right: {camera_right}, up: {camera_up}")

            # Build rotation matrix where:
            # - Z = camera_up (filter Z aligns with camera up)
            # - X = camera_right (filter X aligns with camera right)
            # - XZ plane = view plane (the plane you're looking at)
            # - Y = cross(Z, X) = points backward from camera
            x_axis = camera_right
            z_axis = camera_up
            y_axis = np.cross(z_axis, x_axis)
            y_axis = y_axis / np.linalg.norm(y_axis)

            # Rotation matrix: columns are new basis vectors
            R = np.column_stack([x_axis, y_axis, z_axis])

            # Convert to Euler angles (degrees)
            rx, ry, rz = matrix_to_euler_deg(R)

            # Set rotation for the appropriate filter type
            if filter_type == "Box":
                if self.ui.box_rot_x:
                    self.ui.box_rot_x.value = rx
                if self.ui.box_rot_y:
                    self.ui.box_rot_y.value = ry
                if self.ui.box_rot_z:
                    self.ui.box_rot_z.value = rz
                logger.debug(
                    f"Aligned box rotation to camera up: ({rx:.1f}, {ry:.1f}, {rz:.1f})"
                )

            elif filter_type == "Ellipsoid":
                if self.ui.ellipsoid_rot_x:
                    self.ui.ellipsoid_rot_x.value = rx
                if self.ui.ellipsoid_rot_y:
                    self.ui.ellipsoid_rot_y.value = ry
                if self.ui.ellipsoid_rot_z:
                    self.ui.ellipsoid_rot_z.value = rz
                logger.debug(
                    f"Aligned ellipsoid rotation to camera up: ({rx:.1f}, {ry:.1f}, {rz:.1f})"
                )

            # Trigger visualization update
            self._update_filter_visualization()

        except Exception as e:
            logger.error(f"Failed to align filter to camera up: {e}", exc_info=True)

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
        from src.gsplay.config.rotation_conversions import camera_to_frustum_euler_deg

        pos = np.array(camera_pos, dtype=np.float64)

        # Check if transform is neutral
        if transform_values is None or (
            hasattr(transform_values, "is_neutral") and transform_values.is_neutral()
        ):
            euler_deg = camera_to_frustum_euler_deg(camera_rot)
            return tuple(pos), euler_deg

        # Get transform components
        translation = np.array(
            getattr(transform_values, "translation", (0.0, 0.0, 0.0)), dtype=np.float64
        )
        # Handle both scalar and per-axis scale (gsmod 0.1.7)
        scale_raw = getattr(transform_values, "scale", 1.0)
        if isinstance(scale_raw, (tuple, list)):
            scale = np.array(scale_raw, dtype=np.float64)
        else:
            scale = np.array([float(scale_raw)] * 3, dtype=np.float64)
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
        euler_deg = camera_to_frustum_euler_deg(world_rot_quat)

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

        # Get client for batched GUI update
        clients = list(self.server.get_clients().values()) if self.server else []
        client = clients[0] if clients else None

        if self.ui:
            if client:
                # Batch all slider updates for proper GUI refresh
                with client.atomic():
                    self.ui.set_transform_values(self.config.transform_values)
            else:
                self.ui.set_transform_values(self.config.transform_values)
            # Unlock filter controls since transform is now neutral
            self.ui.set_filter_controls_disabled(False)
            # Re-enable gizmo if filter viz is shown
            if self.filter_visualizer and self.ui.show_filter_viz and self.ui.show_filter_viz.value:
                self.filter_visualizer.set_gizmo_enabled(True)

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
        # Filter operates on original data - not affected by scene transform
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

                # Compute filter mask using original space filter values
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

        # Lock filter controls and gizmo since center scene applies transformation
        if self.ui and self.ui.is_transform_active():
            self.ui.set_filter_controls_disabled(True)
            if self.filter_visualizer:
                self.filter_visualizer.set_gizmo_enabled(False)

        # Trigger rerender
        if self.viewer:
            self.render_component.rerender()

        logger.info(f"Scene centered: translation=[{tx:.3f}, {ty:.3f}, {tz:.3f}]")

    def _bake_camera_view(self) -> None:
        """Bake current camera view into model transform.

        Computes new model transform such that when camera resets to default
        isometric view (azimuth=45°, elevation=30°), the scene appears identical
        to the current view.

        IMPORTANT: When the camera is far from the default view (>90° away), the
        model will receive a large rotation (possibly near 180°) to preserve the
        view. This is mathematically correct - the model must rotate to compensate
        for the camera movement. For example:
        - Camera at azimuth=225° (behind the scene) requires ~180° model rotation
        - Camera at azimuth=45° (near default) requires minimal model rotation

        Math (view preservation with pivot/center correction):
        - Camera delta: R_delta = R_default @ R_current.T
        - Translation delta: t_delta = pos_default - R_delta @ pos_current
        - New rotation: R_new = R_delta @ R_model
        - New translation: t_new = R_delta @ t_model + t_delta
                                 + (R_delta - I) @ c_old           # old center correction
                                 + (I - R_new) @ (c_old - c_new)   # pivot change correction

        The center corrections are needed because gsmod applies:
            P_world = R @ (P - center) + center + translation

        When "Use Pivot" is checked, c_new is set to -t_model (negative of current
        translation), which makes future rotations pivot around the world origin.
        """
        import numpy as np
        import viser.transforms as vt
        from src.gsplay.rendering.quaternion_utils import (
            quat_multiply,
            rotation_matrix_to_quat,
        )

        if self.camera_controller is None or self.camera_controller.state is None:
            logger.warning("No camera available for bake view")
            return

        if self.ui is None:
            logger.warning("No UI available for bake view")
            return

        try:
            # 1. Get current camera state from viser (position/look_at are reliable)
            clients = list(self.server.get_clients().values())
            if not clients:
                logger.warning("No viser clients connected for bake view")
                return

            client = clients[0]

            # Read position/look_at from viser (these are reliable, unlike wxyz)
            pos_original = np.array(client.camera.position, dtype=np.float64)
            look_at_original = np.array(client.camera.look_at, dtype=np.float64)
            viser_wxyz = np.array(client.camera.wxyz, dtype=np.float64)

            # Also read internal state for comparison
            internal_az = self.camera_controller.state.azimuth if self.camera_controller.state else 0.0
            internal_el = self.camera_controller.state.elevation if self.camera_controller.state else 0.0
            internal_roll = self.camera_controller.state.roll if self.camera_controller.state else 0.0

            logger.debug(
                f"[BakeView] Viser state: pos={pos_original}, look_at={look_at_original}, "
                f"wxyz={viser_wxyz}"
            )
            logger.debug(
                f"[BakeView] Internal state: az={internal_az:.2f}, el={internal_el:.2f}, "
                f"roll={internal_roll:.2f}"
            )

            # 2. Extract azimuth/elevation from position/look_at geometry
            # This is UNAMBIGUOUS - no quaternion convention issues!
            offset = pos_original - look_at_original
            distance = float(np.linalg.norm(offset))
            if distance < 1e-6:
                logger.warning("Camera too close to look_at point")
                return

            offset_norm = offset / distance

            # Extract elevation from Y component
            az_current = float(np.degrees(np.arctan2(offset[0], offset[2])))
            el_current = float(np.degrees(np.arcsin(np.clip(offset_norm[1], -1.0, 1.0))))

            # Roll: use internal state (viser orbit controls don't change roll)
            roll_current = 0.0
            if self.camera_controller and self.camera_controller.state:
                roll_current = self.camera_controller.state.roll

            # Log extracted vs internal angles for debugging
            az_diff = abs(az_current - internal_az)
            el_diff = abs(el_current - internal_el)
            # Handle azimuth wraparound (e.g., -135 vs 225)
            if az_diff > 180:
                az_diff = 360 - az_diff
            logger.debug(
                f"[BakeView] Extracted: az={az_current:.2f}, el={el_current:.2f}, roll={roll_current:.2f}"
            )
            if az_diff > 5.0 or el_diff > 5.0:
                logger.warning(
                    f"[BakeView] State mismatch! Extracted az/el differ from internal: "
                    f"az_diff={az_diff:.2f}, el_diff={el_diff:.2f}"
                )

            # Warn about gimbal lock risk near poles
            if abs(el_current) > 85.0:
                logger.warning(
                    f"[BakeView] Near pole (el={el_current:.2f}°) - azimuth may be unstable!"
                )

            # Check if look_at is not at origin (could affect results)
            look_at_dist = float(np.linalg.norm(look_at_original))
            if look_at_dist > 0.01:
                logger.debug(
                    f"[BakeView] Note: look_at not at origin: {look_at_original} (dist={look_at_dist:.3f})"
                )

            # 3. Compute R_current and R_default using VISER's look-at convention
            # CRITICAL: Must match what viser ACTUALLY produces for camera.wxyz!
            #
            # Viser's look-at formula (different from OpenGL):
            # - forward = normalize(look_at - position) [toward target]
            # - right = normalize(cross(forward, up_hint))
            # - up = cross(forward, right)  [NOTE: forward x right, not right x forward!]
            # - R = [right, up, forward]  [camera looks down +Z toward target]
            #
            # This differs from OpenGL which has:
            # - up = cross(right, forward)
            # - R = [right, up, -forward]  [camera looks down -Z]

            def viser_look_at_matrix(position: np.ndarray, target: np.ndarray) -> np.ndarray:
                """Compute rotation matrix using VISER's look-at convention."""
                forward = target - position
                dist = np.linalg.norm(forward)
                if dist < 1e-6:
                    return np.eye(3)
                forward = forward / dist

                up_hint = np.array([0.0, 1.0, 0.0])
                right = np.cross(forward, up_hint)
                right_norm = np.linalg.norm(right)
                if right_norm < 1e-6:
                    # Looking straight up/down
                    up_hint = np.array([0.0, 0.0, 1.0])
                    right = np.cross(forward, up_hint)
                    right_norm = np.linalg.norm(right)
                right = right / right_norm

                # VISER convention: up = forward x right (not right x forward!)
                up = np.cross(forward, right)

                # VISER convention: camera looks down +Z, so column 2 = forward (not -forward!)
                return np.column_stack([right, up, forward])

            # R_current from viser's actual wxyz (this is what rendering uses!)
            R_current = vt.SO3(viser_wxyz).as_matrix()

            # For comparison, compute what our viser_look_at_matrix would give
            R_current_formula = viser_look_at_matrix(pos_original, look_at_original)
            formula_vs_viser = float(np.linalg.norm(R_current - R_current_formula, 'fro'))
            logger.debug(f"[BakeView] R_current (viser wxyz) vs R_current (viser_look_at formula): diff={formula_vs_viser:.6f}")

            # Default iso view: az=45°, el=30°
            az_default, el_default = 45.0, 30.0
            az_rad = np.radians(az_default)
            el_rad = np.radians(el_default)

            # Default camera position at iso view around ORIGIN
            pos_default = distance * np.array([
                np.sin(az_rad) * np.cos(el_rad),  # X
                np.sin(el_rad),                    # Y (up)
                np.cos(az_rad) * np.cos(el_rad),  # Z
            ])

            # R_default using viser's look-at convention (matches what viser will produce)
            R_default = viser_look_at_matrix(pos_default, np.zeros(3))

            # 4. Compute camera delta (rotation that transforms current view to default view)
            # R_delta such that: R_default = R_delta @ R_current
            # => R_delta = R_default @ R_current.T
            R_delta = R_default @ R_current.T
            q_delta = rotation_matrix_to_quat(R_delta)

            # Diagnostic: check if camera is at/near default position
            az_diff_from_default = abs(az_current - 45.0)
            if az_diff_from_default > 180:
                az_diff_from_default = 360 - az_diff_from_default
            el_diff_from_default = abs(el_current - 30.0)
            roll_diff_from_default = abs(roll_current)  # default roll is 0
            look_at_at_origin = float(np.linalg.norm(look_at_original)) < 0.01
            near_default = (az_diff_from_default < 5.0 and el_diff_from_default < 5.0 and
                           roll_diff_from_default < 5.0 and look_at_at_origin)

            if near_default:
                R_diff = float(np.linalg.norm(R_current - R_default, 'fro'))
                logger.debug(
                    f"[BakeView] Camera near default (az={az_current:.1f}, el={el_current:.1f}, roll={roll_current:.1f})! "
                    f"R_current vs R_default diff: {R_diff:.6f}"
                )
                if R_diff > 0.01:
                    logger.warning(
                        f"[BakeView] R_current != R_default even at default position!\n"
                        f"R_current (from viser wxyz):\n{R_current}\n"
                        f"R_default (from viser_look_at_matrix):\n{R_default}"
                    )

            # Log R_delta info - if close to identity, camera is already near default
            R_delta_trace = float(np.trace(R_delta))  # Identity has trace=3
            logger.debug(f"[BakeView] R_delta trace: {R_delta_trace:.4f} (3.0 = identity)")

            # Translation delta (use original position, not current which is now at iso)
            t_delta = pos_default - R_delta @ pos_original
            logger.debug(f"[BakeView] t_delta: {t_delta}")

            # 5. Get current model transform
            tv_current = self.ui.get_transform_values()
            q_model = np.array(tv_current.rotation)  # wxyz
            t_model = np.array(tv_current.translation)

            # IMPORTANT: c_old is the center ACTUALLY used in the current transform
            # This is needed for correct view preservation math
            c_old = np.array(tv_current.center) if tv_current.center is not None else np.zeros(3)

            # Determine the NEW center we want after baking
            # If "Use Pivot" is checked, use negative of current translation as the new pivot
            use_pivot = self.ui.use_pivot_checkbox.value if self.ui.use_pivot_checkbox else False
            if use_pivot:
                # Use negative of current translation as pivot point
                c_new = -t_model.copy()
                logger.debug(f"[BakeView] Use Pivot enabled: setting pivot to -{t_model} = {c_new}")
            else:
                # Keep the same center as before
                c_new = c_old.copy()

            logger.debug(f"[BakeView] c_old={c_old}, c_new={c_new}")

            # Read raw slider values for debugging
            slider_rx = self.ui.rotate_x_slider.value if self.ui.rotate_x_slider else 0.0
            slider_ry = self.ui.rotate_y_slider.value if self.ui.rotate_y_slider else 0.0
            slider_rz = self.ui.rotate_z_slider.value if self.ui.rotate_z_slider else 0.0

            logger.debug(
                f"[BakeView] Slider values: rx={slider_rx:.2f}, ry={slider_ry:.2f}, rz={slider_rz:.2f}"
            )
            logger.debug(
                f"[BakeView] Current model: q={q_model}, t={t_model}, c_old={c_old}, c_new={c_new}"
            )

            # 6. Apply delta to model rotation: q_new = q_delta * q_model
            q_new = quat_multiply(q_delta, q_model)

            # 7. Apply delta to model translation WITH center correction
            #
            # The gsmod transform applies: P_world = R @ (P - c) + c + t
            # For view preservation, the correct formula is:
            #   t_new = R_delta @ t_model + t_delta + (R_delta - I) @ c_old + (I - R_new) @ (c_old - c_new)
            #
            # This accounts for:
            # - (R_delta - I) @ c_old: correction for rotation around the OLD center
            # - (I - R_new) @ (c_old - c_new): correction for changing the pivot point
            #
            R_new = R_delta @ vt.SO3(q_model).as_matrix()
            old_center_correction = R_delta @ c_old - c_old  # = (R_delta - I) @ c_old
            pivot_change_correction = (c_old - c_new) - R_new @ (c_old - c_new)  # = (I - R_new) @ (c_old - c_new)
            t_new = R_delta @ t_model + t_delta + old_center_correction + pivot_change_correction

            logger.debug(
                f"[BakeView] old_center_correction={old_center_correction}, "
                f"pivot_change_correction={pivot_change_correction}"
            )

            # 8. Decompose new rotation to Euler angles (extrinsic XYZ: R = Rz @ Ry @ Rx)
            # This matches the composition order in get_transform_values()
            R_new = vt.SO3(q_new).as_matrix()
            rx, ry, rz = self._decompose_rotation_xyz(R_new)

            # Verify round-trip: compose back and check matrix difference
            from src.gsplay.rendering.quaternion_utils import quat_from_axis_angle
            qx_verify = quat_from_axis_angle(np.array([1, 0, 0]), rx)
            qy_verify = quat_from_axis_angle(np.array([0, 1, 0]), ry)
            qz_verify = quat_from_axis_angle(np.array([0, 0, 1]), rz)
            q_verify = quat_multiply(qz_verify, quat_multiply(qy_verify, qx_verify))
            q_verify = q_verify / np.linalg.norm(q_verify)
            R_verify = vt.SO3(q_verify).as_matrix()
            matrix_diff = float(np.linalg.norm(R_new - R_verify))

            logger.debug(
                f"[BakeView] New model: q_new={q_new}, t_new={t_new}, "
                f"euler=({np.degrees(rx):.2f}, {np.degrees(ry):.2f}, {np.degrees(rz):.2f})"
            )
            logger.debug(
                f"[BakeView] Round-trip verify: q_verify={q_verify}, matrix_diff={matrix_diff:.6f}"
            )
            if matrix_diff > 1e-4:
                logger.warning(
                    f"[BakeView] ROUND-TRIP ERROR! Matrix difference {matrix_diff:.6f} > threshold"
                )

            # 9. Enter APP mode and sync internal state to match what we set
            self.camera_controller._enter_app_mode()

            try:
                # Sync internal state to match the camera position we already set
                self.camera_controller.state.set_from_orbit(
                    azimuth=45.0,
                    elevation=30.0,
                    roll=0.0,
                    distance=distance,
                    look_at=np.zeros(3),
                )

                # 10. Set model sliders and camera together (batched for GUI refresh)
                with client.atomic():
                    if self.ui.rotate_x_slider:
                        self.ui.rotate_x_slider.value = float(np.degrees(rx))
                    if self.ui.rotate_y_slider:
                        self.ui.rotate_y_slider.value = float(np.degrees(ry))
                    if self.ui.rotate_z_slider:
                        self.ui.rotate_z_slider.value = float(np.degrees(rz))

                    if self.ui.translation_x_slider:
                        self.ui.translation_x_slider.value = float(t_new[0])
                    if self.ui.translation_y_slider:
                        self.ui.translation_y_slider.value = float(t_new[1])
                    if self.ui.translation_z_slider:
                        self.ui.translation_z_slider.value = float(t_new[2])

                    # Scale unchanged

                    # Update pivot sliders to the new center value
                    # (when use_pivot=True, c_new = -t_model)
                    if use_pivot:
                        if self.ui.pivot_x_slider:
                            self.ui.pivot_x_slider.value = float(c_new[0])
                        if self.ui.pivot_y_slider:
                            self.ui.pivot_y_slider.value = float(c_new[1])
                        if self.ui.pivot_z_slider:
                            self.ui.pivot_z_slider.value = float(c_new[2])

                    # 11. Set camera to iso position
                    # Let viser compute wxyz from position/look_at/up (its internal formula)
                    # Our R_default was computed using viser_look_at_matrix which matches viser's formula
                    client.camera.position = tuple(pos_default)
                    client.camera.look_at = (0.0, 0.0, 0.0)
                    client.camera.up_direction = (0.0, 1.0, 0.0)

                # Diagnostic: verify R_default matches viser's wxyz after setting
                wxyz_after = np.array(client.camera.wxyz, dtype=np.float64)
                R_after = vt.SO3(wxyz_after).as_matrix()
                formula_vs_viser = float(np.linalg.norm(R_default - R_after, 'fro'))
                logger.debug(f"[BakeView] Viser wxyz after set: {wxyz_after}")
                logger.debug(f"[BakeView] R_default (formula) vs R_after (viser): diff={formula_vs_viser:.6f}")
                if formula_vs_viser > 0.01:
                    logger.warning(
                        f"[BakeView] FORMULA MISMATCH! R_default vs viser wxyz: {formula_vs_viser:.6f}\n"
                        f"R_default:\n{R_default}\n"
                        f"R_after (viser):\n{R_after}"
                    )

                # Lock filter controls and gizmo since bake view applies transformation
                # (atomic() may suppress callbacks, so lock explicitly)
                if self.ui.is_transform_active():
                    self.ui.set_filter_controls_disabled(True)
                    if self.filter_visualizer:
                        self.filter_visualizer.set_gizmo_enabled(False)

                # 12. Trigger final rerender
                if self.viewer:
                    self.render_component.rerender()

            finally:
                # 13. Exit APP mode (with longer cooldown to prevent race conditions)
                # Use 0.3s to ensure viser has fully updated before accepting new input
                self.camera_controller._enter_user_mode(cooldown=0.3)

            # Compute angle difference from default for user info
            az_from_default = abs(az_current - 45.0)
            if az_from_default > 180:
                az_from_default = 360 - az_from_default
            el_from_default = abs(el_current - 30.0)
            cam_angle_diff = np.sqrt(az_from_default**2 + el_from_default**2)

            # Compute rotation magnitude from R_delta trace
            # trace = 1 + 2*cos(angle), so angle = arccos((trace-1)/2)
            rotation_angle = np.degrees(np.arccos(np.clip((R_delta_trace - 1) / 2, -1, 1)))

            if cam_angle_diff < 5.0:
                logger.info(
                    f"Baked view (near default - minimal change): rotation=({np.degrees(rx):.1f}, {np.degrees(ry):.1f}, {np.degrees(rz):.1f})"
                )
            else:
                logger.info(
                    f"Baked view: rotation=({np.degrees(rx):.1f}, {np.degrees(ry):.1f}, {np.degrees(rz):.1f}), "
                    f"translation=({t_new[0]:.3f}, {t_new[1]:.3f}, {t_new[2]:.3f}), "
                    f"cam_delta={cam_angle_diff:.0f}°, rot_delta={rotation_angle:.0f}°"
                )
                if cam_angle_diff > 90:
                    logger.info(
                        f"Note: Camera was {cam_angle_diff:.0f}° from default iso view - "
                        f"model rotated {rotation_angle:.0f}° to preserve scene appearance"
                    )

        except Exception as e:
            logger.error(f"Failed to bake camera view: {e}", exc_info=True)

    def _decompose_rotation_xyz(self, R: "np.ndarray") -> tuple[float, float, float]:
        """Decompose rotation matrix to extrinsic XYZ Euler angles.

        This matches the composition order in get_transform_values():
        R = Rz @ Ry @ Rx (extrinsic XYZ)

        Returns (rx, ry, rz) in radians.
        """
        import numpy as np

        # For R = Rz @ Ry @ Rx:
        # R[2,0] = -sin(ry)
        # R[2,1] = sin(rx)*cos(ry)
        # R[2,2] = cos(rx)*cos(ry)
        # R[0,0] = cos(ry)*cos(rz)
        # R[1,0] = cos(ry)*sin(rz)

        sy = -R[2, 0]
        cy = np.sqrt(1 - sy * sy)

        if cy > 1e-6:
            # Normal case
            rx = np.arctan2(R[2, 1], R[2, 2])
            ry = np.arctan2(sy, cy)
            rz = np.arctan2(R[1, 0], R[0, 0])
        else:
            # Gimbal lock (ry = +/-90 degrees)
            rx = np.arctan2(-R[1, 2], R[1, 1])
            ry = np.pi / 2 if sy > 0 else -np.pi / 2
            rz = 0.0

        return (rx, ry, rz)

    def _handle_filter_reset(self) -> None:
        """Reset filter parameters to defaults (keeps current filter type)."""
        logger.info("Resetting filter parameters")

        if self.ui:
            # Get current filter type (keep it)
            current_type = self.ui.spatial_filter_type.value if self.ui.spatial_filter_type else "None"

            # Reset opacity/scale sliders
            if self.ui.min_opacity_slider:
                self.ui.min_opacity_slider.value = 0.0
            if self.ui.max_opacity_slider:
                self.ui.max_opacity_slider.value = 1.0
            if self.ui.min_scale_slider:
                self.ui.min_scale_slider.value = 0.0
            if self.ui.max_scale_slider:
                self.ui.max_scale_slider.value = 100.0

            # Reset parameters for current filter type
            if current_type == "Sphere":
                if self.ui.sphere_center_x:
                    self.ui.sphere_center_x.value = 0.0
                if self.ui.sphere_center_y:
                    self.ui.sphere_center_y.value = 0.0
                if self.ui.sphere_center_z:
                    self.ui.sphere_center_z.value = 0.0
                if self.ui.sphere_radius:
                    self.ui.sphere_radius.value = 10.0

            elif current_type == "Box":
                if self.ui.box_center_x:
                    self.ui.box_center_x.value = 0.0
                if self.ui.box_center_y:
                    self.ui.box_center_y.value = 0.0
                if self.ui.box_center_z:
                    self.ui.box_center_z.value = 0.0
                if self.ui.box_size_x:
                    self.ui.box_size_x.value = 10.0
                if self.ui.box_size_y:
                    self.ui.box_size_y.value = 10.0
                if self.ui.box_size_z:
                    self.ui.box_size_z.value = 10.0
                if self.ui.box_rot_x:
                    self.ui.box_rot_x.value = 0.0
                if self.ui.box_rot_y:
                    self.ui.box_rot_y.value = 0.0
                if self.ui.box_rot_z:
                    self.ui.box_rot_z.value = 0.0

            elif current_type == "Ellipsoid":
                if self.ui.ellipsoid_center_x:
                    self.ui.ellipsoid_center_x.value = 0.0
                if self.ui.ellipsoid_center_y:
                    self.ui.ellipsoid_center_y.value = 0.0
                if self.ui.ellipsoid_center_z:
                    self.ui.ellipsoid_center_z.value = 0.0
                if self.ui.ellipsoid_radius_x:
                    self.ui.ellipsoid_radius_x.value = 5.0
                if self.ui.ellipsoid_radius_y:
                    self.ui.ellipsoid_radius_y.value = 5.0
                if self.ui.ellipsoid_radius_z:
                    self.ui.ellipsoid_radius_z.value = 5.0
                if self.ui.ellipsoid_rot_x:
                    self.ui.ellipsoid_rot_x.value = 0.0
                if self.ui.ellipsoid_rot_y:
                    self.ui.ellipsoid_rot_y.value = 0.0
                if self.ui.ellipsoid_rot_z:
                    self.ui.ellipsoid_rot_z.value = 0.0

            elif current_type == "Frustum":
                if self.ui.frustum_pos_x:
                    self.ui.frustum_pos_x.value = 0.0
                if self.ui.frustum_pos_y:
                    self.ui.frustum_pos_y.value = 0.0
                if self.ui.frustum_pos_z:
                    self.ui.frustum_pos_z.value = 0.0
                if self.ui.frustum_rot_x:
                    self.ui.frustum_rot_x.value = 0.0
                if self.ui.frustum_rot_y:
                    self.ui.frustum_rot_y.value = 0.0
                if self.ui.frustum_rot_z:
                    self.ui.frustum_rot_z.value = 0.0
                if self.ui.frustum_fov:
                    self.ui.frustum_fov.value = 60.0
                if self.ui.frustum_aspect:
                    self.ui.frustum_aspect.value = 1.0
                if self.ui.frustum_near:
                    self.ui.frustum_near.value = 0.1
                if self.ui.frustum_far:
                    self.ui.frustum_far.value = 100.0

            # Update config from reset UI values
            camera_pos, camera_rot = None, None
            if self.viewer:
                camera_pos, camera_rot = self.viewer._get_camera_state()
            self.config.filter_values = self.ui.get_filter_values(
                camera_position=camera_pos,
                camera_rotation=camera_rot,
            )

        # Reset VolumeFilter config
        self.config.volume_filter = VolumeFilter()

        if self.viewer:
            self.viewer._update_filter_visualization()
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
