"""
Embedded nerfview viewer implementation.

Adapted from the upstream nerfview project so that we can customize behavior
alongside the rest of the viewer without a third-party dependency.

Synchronized Viewing
--------------------
All connected clients share the same view:
- Single renderer broadcasts to ALL clients via server.scene
- When any client moves the camera, all others follow
- UI operations affect all clients simultaneously
"""

from __future__ import annotations

import dataclasses
import logging
import time
from pathlib import Path
from threading import Lock
from typing import Callable, Literal, Tuple

import numpy as np
from numpy.typing import NDArray
import viser
import viser.transforms as vt

from ._renderer import SharedRenderer, RenderTask
from .render_panel import RenderTabState

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class CameraState(object):
    fov: float
    aspect: float
    c2w: NDArray[np.float32]

    def get_K(self, img_wh: Tuple[int, int]) -> NDArray[np.float32]:
        W, H = img_wh
        focal_length = H / 2.0 / np.tan(self.fov / 2.0)
        K = np.array(
            [
                [focal_length, 0.0, W / 2.0],
                [0.0, focal_length, H / 2.0],
                [0.0, 0.0, 1.0],
            ]
        )
        return K


VIEWER_LOCK = Lock()


def with_viewer_lock(fn: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        with VIEWER_LOCK:
            return fn(*args, **kwargs)

    return wrapper


class GSPlay(object):
    """Main viewer class with synchronized multi-client support.

    All connected clients share the same view - when any client moves the
    camera, all others follow. A single renderer broadcasts frames to all
    clients via server.scene.set_background_image().

    Args:
        server: The viser server object to bind to.
        render_fn: Function that takes (camera_state, render_tab_state) and
            returns an image as a float32 numpy array [H,W,3] in range [0,1].
            Optionally returns (image, depth) tuple.
        mode: "rendering" or "training" mode.
    """

    def __init__(
        self,
        server: viser.ViserServer,
        render_fn: Callable,
        output_dir: Path | None = None,
        mode: Literal["rendering", "training"] = "rendering",
        time_enabled: bool = False,
        time_slider: viser.GuiSliderHandle | None = None,
        total_frames: int = 0,
        universal_viewer: object | None = None,
        jpeg_quality_static: int = 90,
        jpeg_quality_move: int = 60,
    ):
        # Public states
        self.time_enabled = time_enabled
        self.server = server
        self.render_fn = render_fn
        self.mode = mode
        self.lock = VIEWER_LOCK
        self.state = "preparing"
        self.output_dir = output_dir if output_dir is not None else Path("./results")
        self.time_slider = time_slider
        self.total_frames = total_frames
        self.universal_viewer = universal_viewer
        self.jpeg_quality_static = jpeg_quality_static
        self.jpeg_quality_move = jpeg_quality_move
        self.auto_quality_enabled: bool = True

        # Private states
        self._renderer: SharedRenderer | None = None
        self._step: int = 0
        self._last_update_step: int = 0
        self._last_move_time: float = 0.0

        # Initialize GUIs
        server.scene.set_global_visibility(True)
        server.on_client_disconnect(self._disconnect_client)
        server.on_client_connect(self._connect_client)
        server.gui.set_panel_label("gsplay")
        if self.mode == "training":
            self._init_training_tab()
            self._populate_training_tab()
        self.render_tab_state = RenderTabState()
        self.state = mode

        # Start the shared renderer
        self._renderer = SharedRenderer(viewer=self, lock=self.lock)
        self._renderer.start()
        logger.info("Viewer started (synchronized mode)")

    def _init_training_tab(self):
        self._training_tab_handles = {}
        self._training_folder = self.server.gui.add_folder("Training")

    def _populate_training_tab(self):
        server = self.server
        with self._training_folder:
            step_number = server.gui.add_number(
                "Step",
                min=0,
                max=1000000,
                step=1,
                disabled=True,
                initial_value=0,
            )
            pause_train_button = server.gui.add_button(
                "Pause",
                icon=viser.Icon.PLAYER_PAUSE,
                hint="Pause the training.",
            )
            resume_train_button = server.gui.add_button(
                "Resume",
                icon=viser.Icon.PLAYER_PLAY,
                visible=False,
                hint="Resume the training.",
            )

            @pause_train_button.on_click
            @resume_train_button.on_click
            def _(_) -> None:
                pause_train_button.visible = not pause_train_button.visible
                resume_train_button.visible = not resume_train_button.visible
                if self.state != "completed":
                    self.state = "paused" if self.state == "training" else "training"

            train_util_slider = self.server.gui.add_slider(
                "Train Util", min=0.0, max=1.0, step=0.05, initial_value=0.9
            )
            train_util_slider.on_update(self.rerender)

        self._training_tab_handles = {
            "step_number": step_number,
            "pause_train_button": pause_train_button,
            "resume_train_button": resume_train_button,
            "train_util_slider": train_util_slider,
        }

    def rerender(self, _):
        """Trigger a re-render (called when UI settings change)."""
        if self._renderer is not None:
            camera_state = self._renderer.get_camera_state()
            if camera_state is not None:
                self._renderer.submit(RenderTask("rerender", camera_state))

    def _disconnect_client(self, client: viser.ClientHandle):
        """Handle client disconnection."""
        client_id = client.client_id
        logger.debug(f"Client {client_id} disconnected")

    def _connect_client(self, client: viser.ClientHandle):
        """Handle new client connection."""
        client_id = client.client_id
        logger.debug(f"Client {client_id} connected")

        # NOTE: Do NOT initialize renderer camera here from viser's default camera.
        # The camera_ui module will apply the stored camera state after a short delay,
        # which will trigger on_update and properly initialize the renderer camera.
        # This ensures the renderer uses our stored camera state, not viser's default.

        @client.camera.on_update
        def _(_: viser.CameraHandle):
            self._last_move_time = time.perf_counter()

            # Only update renderer if client has been initialized with our camera state.
            # This prevents viser's default camera from being used for initial renders.
            if self.universal_viewer is not None:
                camera_ctrl = getattr(self.universal_viewer, "camera_controller", None)
                if camera_ctrl is not None:
                    if client.client_id not in camera_ctrl._initialized_clients:
                        return  # Skip - client not yet initialized with our camera state
                    # Skip during rotation - rotation callback handles rendering
                    if getattr(camera_ctrl, "_rotation_active", False):
                        return
                    # Skip briefly after rotation stops (backend state is authoritative)
                    stop_time = getattr(camera_ctrl, "_rotation_stop_time", 0.0)
                    if time.time() - stop_time < 0.5:
                        return

            with self.server.atomic():
                camera_state = self.get_camera_state(client)
                if self._renderer is not None:
                    self._renderer.update_camera(camera_state)
                    self._renderer.submit(RenderTask("move", camera_state))

    def get_camera_state(self, client: viser.ClientHandle) -> CameraState:
        camera = client.camera
        c2w = np.concatenate(
            [
                np.concatenate(
                    [vt.SO3(camera.wxyz).as_matrix(), camera.position[:, None]], 1
                ),
                [[0, 0, 0, 1]],
            ],
            0,
        )
        return CameraState(
            fov=camera.fov,
            aspect=camera.aspect,
            c2w=c2w,
        )

    def update(self, step: int, num_train_rays_per_step: int):
        """Update viewer during training (training mode only)."""
        if self.mode == "rendering":
            raise ValueError("`update` method is only available in training mode.")
        if step < 5:
            return
        self._step = step
        self._training_tab_handles["step_number"].value = step

        if self._renderer is None:
            return

        # Stop training while user moves camera to make viewing smoother
        while time.perf_counter() - self._last_move_time < 0.1:
            time.sleep(0.05)

        if (
            self.state == "training"
            and self._training_tab_handles["train_util_slider"].value != 1
        ):
            assert (
                self.render_tab_state.num_train_rays_per_sec is not None
            ), "User must keep track of `num_train_rays_per_sec` to use `update`."
            train_s = self.render_tab_state.num_train_rays_per_sec
            view_s = self.render_tab_state.num_view_rays_per_sec
            train_util = self._training_tab_handles["train_util_slider"].value
            view_n = self.render_tab_state.viewer_res**2
            train_n = num_train_rays_per_step
            train_time = train_n / train_s
            view_time = view_n / view_s
            update_every = (
                train_util * view_time / (train_time - train_util * train_time)
            )
            if step > self._last_update_step + update_every:
                self._last_update_step = step
                camera_state = self._renderer.get_camera_state()
                if camera_state is not None:
                    self._renderer.submit(RenderTask("update", camera_state))

    def _after_render(self):
        # This function will be called each time render_fn is called.
        # It can be used to update the viewer panel.
        pass

    def complete(self):
        print("Training complete, disable training tab.")
        self.state = "completed"
        self._training_tab_handles["pause_train_button"].disabled = True
        self._training_tab_handles["resume_train_button"].disabled = True
        self._training_tab_handles["train_util_slider"].disabled = True
