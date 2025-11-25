"""
Render component for managing the rendering pipeline.

This component is responsible for:
- Creating render functions
- Managing nerfview viewer instance
- Rendering configuration
- Render quality and resolution management
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

import viser

from src.domain.entities import GSTensor
from src.domain.interfaces import ModelInterface
from src.gsplay.config.settings import UIHandles
from src.gsplay.interaction.events import EventBus, EventType
from src.gsplay.nerfview import GSPlay

if TYPE_CHECKING:
    from src.gsplay.config.settings import GSPlayConfig

logger = logging.getLogger(__name__)


class RenderComponent:
    """
    Component responsible for rendering pipeline management.

    Handles:
    - Render function creation
    - Nerfview viewer lifecycle
    - Render quality/resolution configuration
    - Integration with edit pipeline
    """

    def __init__(
        self,
        server: viser.ViserServer,
        device: str = "cuda",
        output_dir: str | None = None,
        event_bus: EventBus | None = None,
    ):
        """
        Initialize render component.

        Parameters
        ----------
        server : viser.ViserServer
            Viser server instance
        device : str
            Device for rendering ('cuda' or 'cpu')
        output_dir : Optional[str]
            Output directory for rendered images
        event_bus : Optional[EventBus]
            Event bus for emitting render events
        """
        self.server = server
        self.device = device
        self.output_dir = output_dir
        self.event_bus = event_bus

        # GSPlay state
        self.viewer: GSPlay | None = None
        self.render_fn: Callable | None = None

        logger.debug(f"RenderComponent initialized (device={device})")

    def create_render_function(
        self,
        model: ModelInterface,
        ui: UIHandles,
        apply_edits_fn: Callable[[GSTensor], GSTensor] | None = None,
        config: GSPlayConfig | None = None,
    ) -> Callable:
        """
        Create render function for nerfview.

        Parameters
        ----------
        model : ModelInterface
            Model to render
        ui : UIHandles
            UI handles for accessing current state
        apply_edits_fn : Optional[Callable]
            Function to apply edits to GSTensor

        Returns
        -------
        Callable
            Render function compatible with nerfview
        """
        from src.gsplay.rendering.renderer import create_render_function

        logger.debug("Creating render function")

        self.render_fn = create_render_function(
            model=model,
            ui=ui,
            device=self.device,
            apply_edits_fn=apply_edits_fn,
            config=config,
            event_bus=self.event_bus,
        )

        return self.render_fn

    def create_viewer(
        self,
        render_fn: Callable | None = None,
        mode: str = "rendering",
        time_enabled: bool = True,
        jpeg_quality_static: int = 90,
        jpeg_quality_move: int = 60,
    ) -> GSPlay:
        """
        Create nerfview viewer instance.

        Parameters
        ----------
        render_fn : Optional[Callable]
            Render function (uses stored function if None)
        mode : str
            GSPlay mode ('rendering' or 'training')
        time_enabled : bool
            Enable time controls
        jpeg_quality_static : int
            JPEG quality for static rendering (1-100, default 90)
        jpeg_quality_move : int
            JPEG quality during camera movement (1-100, default 60)

        Returns
        -------
        Any
            Nerfview viewer instance
        """
        if render_fn is None:
            if self.render_fn is None:
                raise ValueError(
                    "No render function available. Call create_render_function first."
                )
            render_fn = self.render_fn

        logger.info("Creating nerfview viewer")
        logger.debug(
            f"JPEG quality: static={jpeg_quality_static}, move={jpeg_quality_move}"
        )

        # Emit viewer creation event
        if self.event_bus:
            self.event_bus.emit(
                EventType.VIEWER_CREATED, source="render_component", mode=mode
            )

        self.viewer = GSPlay(
            server=self.server,
            render_fn=render_fn,
            output_dir=self.output_dir,
            mode=mode,
            time_enabled=time_enabled,
            jpeg_quality_static=jpeg_quality_static,
            jpeg_quality_move=jpeg_quality_move,
        )

        logger.debug("Nerfview viewer created")

        return self.viewer

    def setup_viewer(
        self,
        model: ModelInterface,
        ui: UIHandles,
        apply_edits_fn: Callable[[GSTensor], GSTensor] | None = None,
        mode: str = "rendering",
        time_enabled: bool = True,
        jpeg_quality_static: int = 90,
        jpeg_quality_move: int = 60,
        config: GSPlayConfig | None = None,
    ) -> GSPlay:
        """
        Setup complete rendering pipeline (render function + viewer).

        Parameters
        ----------
        model : ModelInterface
            Model to render
        ui : UIHandles
            UI handles
        apply_edits_fn : Optional[Callable]
            Function to apply edits
        mode : str
            GSPlay mode
        time_enabled : bool
            Enable time controls
        jpeg_quality_static : int
            JPEG quality for static rendering (1-100, default 90)
        jpeg_quality_move : int
            JPEG quality during camera movement (1-100, default 60)

        Returns
        -------
        Any
            Nerfview viewer instance
        """
        # Create render function
        self.create_render_function(model, ui, apply_edits_fn, config)

        # Create viewer
        self.create_viewer(
            mode=mode,
            time_enabled=time_enabled,
            jpeg_quality_static=jpeg_quality_static,
            jpeg_quality_move=jpeg_quality_move,
        )

        return self.viewer

    def configure_quality(self, ui: UIHandles) -> None:
        """
        Configure render quality from UI settings.

        Parameters
        ----------
        ui : UIHandles
            UI handles with quality settings
        """
        if not self.viewer:
            logger.warning("Cannot configure quality: no viewer instance")
            return

        # Hide the Render Res control (we use Quality slider instead)
        if hasattr(self.viewer, "_rendering_tab_handles"):
            render_res_control = self.viewer._rendering_tab_handles.get(
                "render_res_vec2"
            )
            if render_res_control is not None:
                render_res_control.visible = False
                logger.debug("Hidden Render Res control from nerfview")

        # Sync initial quality slider value with viewer_res
        if ui and ui.render_quality:
            self.viewer.render_tab_state.viewer_res = int(ui.render_quality.value)
            logger.debug(
                f"Set initial viewer_res to {self.viewer.render_tab_state.viewer_res}"
            )

    def get_viewer(self) -> GSPlay | None:
        """Get the nerfview viewer instance."""
        return self.viewer

    def rerender(self) -> None:
        """Trigger a re-render of the scene."""
        if self.viewer:
            self.viewer.rerender(None)
        else:
            logger.warning("Cannot rerender: no viewer instance")

    def set_resolution(self, resolution: int) -> None:
        """
        Set render resolution.

        Parameters
        ----------
        resolution : int
            Render resolution (pixels)
        """
        if not self.viewer:
            logger.warning("Cannot set resolution: no viewer instance")
            return

        if hasattr(self.viewer, "render_tab_state"):
            self.viewer.render_tab_state.viewer_res = resolution
            logger.debug(f"Set render resolution to {resolution}")

            # Emit resolution changed event
            if self.event_bus:
                self.event_bus.emit(
                    EventType.RENDER_RESOLUTION_CHANGED,
                    source="render_component",
                    resolution=resolution,
                )

    def cleanup(self) -> None:
        """Clean up rendering resources."""
        logger.debug("Cleaning up render component")

        self.viewer = None
        self.render_fn = None

        if self.event_bus:
            self.event_bus.emit(EventType.VIEWER_DESTROYED, source="render_component")


# Export public API
__all__ = ["RenderComponent"]
