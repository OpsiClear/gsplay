"""
Background renderer threads that drive the embedded nerfview viewer.

This module provides two renderer implementations:

1. **Renderer** (per-client): Each client gets its own renderer thread.
   Good for independent viewing but doesn't sync between clients.

2. **SharedRenderer** (broadcast): Single renderer that broadcasts to ALL clients.
   All clients see the same view. When any client moves the camera, all clients
   follow. This is the recommended mode for collaborative viewing.

Architecture (SharedRenderer)
-----------------------------
    Client A ─┐
              ├──► SharedRenderer ──► GPU JPEG ──► BackgroundImageMessage
    Client B ─┘          │                │                │
                         │                │                ▼
                    shared_camera         └──► stream  ALL clients

When Client A moves camera:
1. SharedRenderer updates shared_camera state
2. Client B's camera is synced to match Client A
3. Single render is broadcast to all via direct BackgroundImageMessage

GPU JPEG Optimization
---------------------
The SharedRenderer uses our GPU JPEG encoder and sends the pre-encoded bytes
directly to viser via BackgroundImageMessage, bypassing viser's internal
PIL/OpenCV encoding. This eliminates double-encoding overhead.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import sys
import threading
import time
import traceback
from typing import TYPE_CHECKING, Literal, Tuple, get_args

import numpy as np
import viser
import viser.transforms as vt
from viser._messages import BackgroundImageMessage

from src.gsplay.rendering.jpeg_encoder import get_cached_jpeg

if TYPE_CHECKING:
    from .viewer import RenderCamera, GSPlay

logger = logging.getLogger(__name__)


def _encode_depth_png(depth: np.ndarray) -> bytes | None:
    """Encode depth array to PNG bytes (same format as viser).

    Viser encodes depth as fixed-point 24-bit values stored in a 4-channel
    image (uint32 viewed as 4 x uint8). OpenCV then encodes as BGR PNG.
    Range: [0, 167.77215] with precision 1e-5.

    Parameters
    ----------
    depth : np.ndarray
        Depth array with shape (H, W) or (H, W, 1).

    Returns
    -------
    bytes | None
        PNG-encoded depth bytes, or None on failure.
    """
    if depth is None:
        return None

    try:
        # Convert to fixed-point (same as viser)
        # Range: [0, 167.77215] with precision 1e-5
        if len(depth.shape) == 3 and depth.shape[2] == 1:
            depth = depth.squeeze(-1)
        depth_fixed = np.clip(depth * 100_000, 0, 2**24 - 1).astype(np.uint32)
        # View uint32 as 4 x uint8 (viser's format)
        intdepth = depth_fixed.reshape((*depth_fixed.shape[:2], 1)).view(np.uint8)
        # intdepth now has shape (H, W, 4)

        # Try OpenCV first (faster, expects BGR)
        try:
            import cv2
            success, encoded = cv2.imencode(".png", intdepth)
            if success:
                return encoded.tobytes()
        except ImportError:
            pass

        # Fallback to imageio (handles multi-channel)
        try:
            import imageio.v3 as iio
            return iio.imwrite("<bytes>", intdepth, extension=".png")
        except ImportError:
            pass

        return None

    except Exception as e:
        logger.debug(f"Depth encoding failed: {e}")
        return None


RenderState = Literal["low_move", "low_static", "high"]
RenderAction = Literal["rerender", "move", "static", "update"]


@dataclasses.dataclass
class RenderTask:
    """Task submitted to the renderer."""

    action: RenderAction
    camera_state: "RenderCamera" | None = None


@dataclasses.dataclass
class CachedFrame:
    """Cached rendered frame with metadata for reuse during interrupts.

    When a render is interrupted (e.g., during rapid camera movement),
    the renderer can reuse this cached frame if it's still valid for
    the current animation frame. This prevents visual artifacts while
    maintaining smooth camera controls.
    """

    img: np.ndarray
    depth: np.ndarray | None
    frame_index: int

    def is_valid_for(self, current_frame_index: int) -> bool:
        """Check if this cached frame can be used for the given frame index.

        Parameters
        ----------
        current_frame_index : int
            The current animation frame index.

        Returns
        -------
        bool
            True if the cached frame is for the same animation frame.
        """
        return self.frame_index == current_frame_index


class InterruptRenderException(Exception):
    """Raised when a render should be interrupted (e.g., camera moved)."""

    pass


class set_trace_context(object):
    def __init__(self, func):
        self.func = func

    def __enter__(self):
        sys.settrace(self.func)
        return self

    def __exit__(self, *_, **__):
        sys.settrace(None)


class Renderer(threading.Thread):
    """This class is responsible for rendering images in the background."""

    def __init__(
        self,
        viewer: "GSPlay",
        client: viser.ClientHandle,
        lock: threading.Lock,
    ):
        super().__init__(daemon=True)

        self.viewer = viewer
        self.client = client
        self.lock = lock

        self.running = True
        self.is_prepared_fn = lambda: self.viewer.state != "preparing"

        self._render_event = threading.Event()
        self._state: RenderState = "low_static"
        self._task: RenderTask | None = None

        self._target_fps = 30
        self._may_interrupt_render = False
        self._old_version = False
        
        # Cache for last complete frame (see CachedFrame docstring)
        self._cached_frame: CachedFrame | None = None

        self._define_transitions()

    def _define_transitions(self):
        transitions: dict[RenderState, dict[RenderAction, RenderState]] = {
            s: {a: s for a in get_args(RenderAction)} for s in get_args(RenderState)
        }
        transitions["low_move"]["static"] = "low_static"
        transitions["low_static"]["static"] = "high"
        transitions["low_static"]["update"] = "high"
        transitions["low_static"]["move"] = "low_move"
        transitions["high"]["move"] = "low_move"
        transitions["high"]["rerender"] = "low_static"
        self.transitions = transitions

    def _may_interrupt_trace(self, frame, event, arg):
        if event == "line":
            if self._may_interrupt_render:
                self._may_interrupt_render = False
                raise InterruptRenderException
        return self._may_interrupt_trace

    def _broadcast_frame(self) -> None:
        """Broadcast cached JPEG frame to stream server.

        The JPEG was already encoded by render_fn via encode_and_cache().
        """
        try:
            from src.gsplay.streaming import get_stream_server
            from src.gsplay.rendering.jpeg_encoder import get_cached_jpeg

            stream_server = get_stream_server()
            if stream_server is None:
                return

            # Get pre-encoded JPEG bytes (encoded in renderer.py)
            jpeg_bytes = get_cached_jpeg()
            if jpeg_bytes is not None:
                stream_server.broadcast_jpeg(jpeg_bytes)

        except ImportError:
            pass  # Streaming or jpeg_encoder not available

    def _get_img_wh(self, aspect: float) -> Tuple[int, int]:
        max_img_res = self.viewer.render_tab_state.viewer_res

        # If auto-quality enabled AND camera is actively moving, use adaptive resolution
        if (
            getattr(self.viewer, "auto_quality_enabled", False)
            and self._state == "low_move"
        ):
            num_view_rays_per_sec = self.viewer.render_tab_state.num_view_rays_per_sec
            target_fps = self._target_fps
            num_viewer_rays = num_view_rays_per_sec / target_fps
            H = (num_viewer_rays / aspect) ** 0.5
            H = int(round(H, -1))
            # Floor at 50% of max resolution (not too low to be unusable)
            min_res = max(120, int(max_img_res * 0.5))
            H = max(min(max_img_res, H), min_res)
            W = int(H * aspect)
            if W > max_img_res:
                W = max_img_res
                H = int(W / aspect)
        else:
            # Fixed resolution (current behavior)
            H = max_img_res
            W = int(H * aspect)
            if W > max_img_res:
                W = max_img_res
                H = int(W / aspect)
        return W, H

    def submit(self, task: RenderTask):
        if self._task is None:
            self._task = task
        elif task.action == "update" and (
            self._state == "low_move" or self._task.action in ["move", "rerender"]
        ):
            return
        else:
            self._task = task

        if self._state == "high" and self._task.action in ["move", "rerender"]:
            self._may_interrupt_render = True
        self._render_event.set()

    def _get_current_frame_index(self) -> int:
        """Get current frame index from viewer's time slider."""
        if self.viewer.time_slider is not None:
            return int(self.viewer.time_slider.value)
        return 0

    def run(self):
        while self.running:
            while not self.is_prepared_fn():
                time.sleep(0.1)
            if not self._render_event.wait(0.2):
                self.submit(
                    RenderTask("static", self.viewer.get_camera_state(self.client))
                )
            self._render_event.clear()
            task = self._task
            assert task is not None
            #  print(self._state, task.action, self.transitions[self._state][task.action])
            if self._state == "high" and task.action == "static":
                continue
            self._state = self.transitions[self._state][task.action]
            assert task.camera_state is not None
            # Determine JPEG quality BEFORE render (so render_fn can use it for encoding)
            if (
                getattr(self.viewer, "auto_quality_enabled", False)
                and task.action == "move"
            ):
                jpeg_quality = self.viewer.jpeg_quality_move
            else:
                jpeg_quality = self.viewer.jpeg_quality_static

            # Get current frame index BEFORE render to detect if it changes
            current_frame_index = self._get_current_frame_index()

            try:
                with self.lock, set_trace_context(self._may_interrupt_trace):
                    tic = time.perf_counter()
                    W, H = img_wh = self._get_img_wh(task.camera_state.aspect)
                    self.viewer.render_tab_state.viewer_width = W
                    self.viewer.render_tab_state.viewer_height = H
                    self.viewer.render_tab_state.jpeg_quality = jpeg_quality

                    if not self._old_version:
                        try:
                            rendered = self.viewer.render_fn(
                                task.camera_state,
                                self.viewer.render_tab_state,
                            )
                        except TypeError:
                            self._old_version = True
                            print(
                                "[WARNING] Your API will be deprecated in the future, please update your render_fn."
                            )
                            rendered = self.viewer.render_fn(task.camera_state, img_wh)
                    else:
                        rendered = self.viewer.render_fn(task.camera_state, img_wh)

                    self.viewer._after_render()
                    if isinstance(rendered, tuple):
                        img, depth = rendered
                    else:
                        img, depth = rendered, None
                    self.viewer.render_tab_state.num_view_rays_per_sec = (W * H) / (
                        time.perf_counter() - tic
                    )

                    # Cache this complete frame
                    self._cached_frame = CachedFrame(img, depth, current_frame_index)

            except InterruptRenderException:
                # Render interrupted - try to reuse cached frame if valid
                if self._cached_frame is not None and self._cached_frame.is_valid_for(
                    current_frame_index
                ):
                    img, depth = self._cached_frame.img, self._cached_frame.depth
                    _skip_gpu_broadcast = True
                else:
                    # No valid cached frame - skip to avoid showing stale content
                    continue
            except Exception as e:
                # Graceful error handling instead of fatal exit
                self._consecutive_failures = getattr(self, "_consecutive_failures", 0) + 1
                logger.error(f"Render error ({self._consecutive_failures}): {e}")
                traceback.print_exc()

                if self._consecutive_failures > 5:
                    logger.warning("Multiple render failures, pausing briefly")
                    time.sleep(1.0)
                    self._consecutive_failures = 0

                # Use cached frame if available
                if self._cached_frame is not None:
                    img, depth = self._cached_frame.img, self._cached_frame.depth
                    _skip_gpu_broadcast = True
                else:
                    continue
            else:
                # Normal render completed - reset failure count
                self._consecutive_failures = 0
                _skip_gpu_broadcast = False

            self.client.scene.set_background_image(
                img,
                format="jpeg",
                jpeg_quality=jpeg_quality,
                depth=depth,
            )

            # Broadcast frame to stream server (pre-encoded JPEG)
            # Skip if render was interrupted - cache may be incomplete/corrupt
            if not _skip_gpu_broadcast:
                self._broadcast_frame()


class SharedRenderer(threading.Thread):
    """Single renderer that broadcasts to ALL connected clients.

    Unlike the per-client Renderer, SharedRenderer:
    - Uses a single render thread for the entire application
    - Broadcasts frames via server.scene.set_background_image() to all clients
    - Maintains a shared camera state that all clients follow
    - Synchronizes camera position across clients when any one moves

    This provides a collaborative viewing experience where all users
    see the exact same view.
    """

    def __init__(
        self,
        viewer: "GSPlay",
        lock: threading.Lock,
    ):
        super().__init__(daemon=True)

        self.viewer = viewer
        self.server = viewer.server
        self.lock = lock

        self.running = True
        self.is_prepared_fn = lambda: self.viewer.state != "preparing"

        self._render_event = threading.Event()
        self._state: RenderState = "low_static"
        self._task: RenderTask | None = None

        self._target_fps = 30
        self._may_interrupt_render = False
        self._old_version = False

        # Cache for last complete frame (see CachedFrame docstring)
        self._cached_frame: CachedFrame | None = None

        # Rotation slider sync throttle
        self._last_slider_sync: float = 0.0

        self._define_transitions()

    def _define_transitions(self):
        transitions: dict[RenderState, dict[RenderAction, RenderState]] = {
            s: {a: s for a in get_args(RenderAction)} for s in get_args(RenderState)
        }
        transitions["low_move"]["static"] = "low_static"
        transitions["low_static"]["static"] = "high"
        transitions["low_static"]["update"] = "high"
        transitions["low_static"]["move"] = "low_move"
        transitions["high"]["move"] = "low_move"
        transitions["high"]["rerender"] = "low_static"
        self.transitions = transitions

    def _may_interrupt_trace(self, frame, event, arg):
        if event == "line":
            if self._may_interrupt_render:
                self._may_interrupt_render = False
                raise InterruptRenderException
        return self._may_interrupt_trace

    def _get_render_camera(self) -> "RenderCamera | None":
        """Get camera for rendering - from viser or headless state.

        Priority:
        1. If viser clients exist: read from viser (source of truth)
        2. If no clients: use headless camera state (for streaming)
        """
        clients = list(self.server.get_clients().values())
        if clients:
            return self.viewer.get_camera_state(clients[0])
        # Fall back to headless camera for streaming without browser
        return self._get_headless_camera()

    def _get_headless_camera(self) -> "RenderCamera | None":
        """Build RenderCamera from headless state when no clients connected.

        This enables continued rendering during rotation when viser window
        is closed, allowing streaming to continue.
        """
        camera_ctrl = self._get_camera_controller()
        if camera_ctrl is None:
            return None
        if camera_ctrl._headless_wxyz is None or camera_ctrl._headless_position is None:
            return None

        import viser.transforms as vt
        from src.gsplay.nerfview.viewer import RenderCamera

        wxyz = camera_ctrl._headless_wxyz
        position = camera_ctrl._headless_position

        # Build c2w using same construction as viewer.py
        R = vt.SO3(wxyz).as_matrix()
        c2w = np.concatenate(
            [
                np.concatenate([R, position[:, None]], 1),
                [[0, 0, 0, 1]],
            ],
            0,
        )

        return RenderCamera(
            fov=camera_ctrl._headless_fov,
            aspect=camera_ctrl._headless_aspect,
            c2w=c2w,
        )

    def _broadcast_frame(self, depth: np.ndarray | None) -> None:
        """Broadcast pre-encoded JPEG frame to all clients.

        Uses JPEG bytes already encoded by render_fn via encode_and_cache().
        Sends directly to both viser clients and stream server.

        Parameters
        ----------
        depth : np.ndarray | None
            Depth array (optional).
        """
        # Get pre-encoded JPEG bytes (encoded in renderer.py)
        jpeg_bytes = get_cached_jpeg()
        if jpeg_bytes is None:
            logger.warning("[BROADCAST] No cached JPEG - frame not sent!")
            return


        # Encode depth if present
        depth_bytes = _encode_depth_png(depth)

        # Send directly to viser (bypasses set_background_image encoding)
        msg = BackgroundImageMessage(
            format="jpeg",
            rgb_data=jpeg_bytes,
            depth_data=depth_bytes,
        )
        self.server._websock_server.queue_message(msg)

        # Also send to stream server
        try:
            from src.gsplay.streaming import get_stream_server
            stream_server = get_stream_server()
            if stream_server is not None:
                stream_server.broadcast_jpeg(jpeg_bytes)
        except ImportError:
            pass

    def _get_img_wh(self, aspect: float) -> Tuple[int, int]:
        max_img_res = self.viewer.render_tab_state.viewer_res

        if (
            getattr(self.viewer, "auto_quality_enabled", False)
            and self._state == "low_move"
        ):
            num_view_rays_per_sec = self.viewer.render_tab_state.num_view_rays_per_sec
            target_fps = self._target_fps
            num_viewer_rays = num_view_rays_per_sec / target_fps
            H = (num_viewer_rays / aspect) ** 0.5
            H = int(round(H, -1))
            min_res = max(120, int(max_img_res * 0.5))
            H = max(min(max_img_res, H), min_res)
            W = int(H * aspect)
            if W > max_img_res:
                W = max_img_res
                H = int(W / aspect)
        else:
            H = max_img_res
            W = int(H * aspect)
            if W > max_img_res:
                W = max_img_res
                H = int(W / aspect)
        return W, H

    def submit(self, task: RenderTask):
        """Submit a render task."""
        if self._task is None:
            self._task = task
        elif task.action == "update" and (
            self._state == "low_move" or self._task.action in ["move", "rerender"]
        ):
            return
        else:
            self._task = task

        # Only interrupt renders for user-driven camera movement, not programmatic rotation.
        # During auto-rotation, interrupting causes blinks because:
        # 1. Interrupted render doesn't complete JPEG encoding
        # 2. _broadcast_frame is skipped when using cached frame
        # 3. No frame is sent to clients → visible blink
        # Auto-rotation is smooth and predictable, so interrupts aren't needed.
        should_interrupt = (
            self._state == "high" and self._task.action in ["move", "rerender"]
        )
        if should_interrupt:
            # Check if auto-rotation is active - if so, don't interrupt
            rotation_active = False
            if self.viewer.universal_viewer is not None:
                camera_ctrl = getattr(
                    self.viewer.universal_viewer, "camera_controller", None
                )
                if camera_ctrl is not None:
                    rotation_active = getattr(camera_ctrl, "_rotation_active", False)

            if not rotation_active:
                self._may_interrupt_render = True

        self._render_event.set()

    def _get_current_frame_index(self) -> int:
        """Get current frame index from viewer's time slider."""
        if self.viewer.time_slider is not None:
            return int(self.viewer.time_slider.value)
        return 0

    def _get_camera_controller(self):
        """Get camera controller from viewer's universal_viewer."""
        if self.viewer.universal_viewer is None:
            return None
        ctrl = getattr(self.viewer.universal_viewer, "camera_controller", None)
        # Debug: Log when rotation is active but we might miss it
        if ctrl is not None and getattr(ctrl, "_rotation_active", False):
            if not hasattr(self, "_last_rotation_log") or time.time() - self._last_rotation_log > 3:
                logger.info(f"[ROTATION] Detected rotation_active=True on camera_controller")
                self._last_rotation_log = time.time()
        return ctrl

    def _build_camera_state_for_rotation(self, camera_ctrl):
        """Build camera state for rotation using the same construction as viewer.py.

        This builds c2w using np.concatenate in the exact same way as
        viewer.py's get_camera_state() to ensure identical memory layout.

        Parameters
        ----------
        camera_ctrl : CameraController
            Camera controller with current rotation state

        Returns
        -------
        RenderCamera or None
            Camera state for rendering, or None if unavailable
        """
        if camera_ctrl is None or camera_ctrl._state is None:
            return self._get_render_camera()

        from src.gsplay.nerfview.viewer import RenderCamera
        from src.gsplay.rendering.quaternion_utils import quat_from_euler_deg

        # Get spherical state thread-safely
        with camera_ctrl._lock:
            state = camera_ctrl._state
            azimuth = state._azimuth
            elevation = state._elevation
            roll = state._roll
            distance = state._distance
            look_at = state.look_at.copy()
            fov = state.fov
            aspect = state.aspect

        # Build quaternion from euler angles
        wxyz = quat_from_euler_deg(azimuth, elevation, roll)

        # Build rotation matrix using viser's SO3 (same as viewer.py)
        R = vt.SO3(wxyz).as_matrix()

        # Compute position from spherical
        forward = -R[:, 2]
        position = look_at.astype(np.float64) - forward * distance

        # Build c2w using EXACT same construction as viewer.py's get_camera_state()
        # This ensures identical memory layout and dtype
        c2w = np.concatenate(
            [
                np.concatenate(
                    [R, position[:, None]], 1
                ),
                [[0, 0, 0, 1]],
            ],
            0,
        )

        # Get fov/aspect from viser if defaults
        viser_camera = self._get_render_camera()
        if viser_camera is not None:
            if abs(fov - 1.0) < 0.01 or abs(aspect - 1.0) < 0.01:
                fov = viser_camera.fov
                aspect = viser_camera.aspect
        else:
            if abs(fov - 1.0) < 0.01:
                fov = 0.82
            if abs(aspect - 1.0) < 0.01:
                aspect = 16.0 / 9.0

        return RenderCamera(
            fov=fov,
            aspect=aspect,
            c2w=c2w,
        )

    def _get_camera_state_from_controller(self):
        """Get camera state directly from CameraController.

        DEPRECATED: Use _build_camera_state_for_rotation instead.
        This method is kept for fallback compatibility.
        """
        camera_ctrl = self._get_camera_controller()
        if camera_ctrl is None or camera_ctrl._state is None:
            return self._get_render_camera()

        from src.gsplay.nerfview.viewer import RenderCamera

        with camera_ctrl._lock:
            state = camera_ctrl._state
            fov = state.fov
            aspect = state.aspect
            c2w = state.c2w.copy()

        if np.isnan(c2w).any() or np.isinf(c2w).any():
            viser_camera = self._get_render_camera()
            if viser_camera is not None:
                return viser_camera
            logger.warning("Camera c2w invalid and no viser camera - using identity")
            c2w = np.eye(4, dtype=np.float32)
            c2w[2, 3] = 5.0

        viser_camera = self._get_render_camera()
        if viser_camera is not None:
            if abs(fov - 1.0) < 0.01 or abs(aspect - 1.0) < 0.01:
                fov = viser_camera.fov
                aspect = viser_camera.aspect
        else:
            if abs(fov - 1.0) < 0.01:
                fov = 0.82
            if abs(aspect - 1.0) < 0.01:
                aspect = 16.0 / 9.0

        return RenderCamera(
            fov=fov,
            aspect=aspect,
            c2w=c2w,
        )

    def run(self):
        """Main render loop - renders and broadcasts to all clients.

        During rotation, polls camera state from CameraController at 20 FPS.
        Otherwise waits for explicit render events from user camera movement.
        """
        while self.running:
            while not self.is_prepared_fn():
                time.sleep(0.1)

            # Check if rotation is active
            camera_ctrl = self._get_camera_controller()
            rotation_active = (
                camera_ctrl is not None
                and getattr(camera_ctrl, "_rotation_active", False)
            )

            if rotation_active:
                # =====================================================================
                # ROTATION RENDERING - VISER ROUND-TRIP REQUIRED
                # =====================================================================
                #
                # We push wxyz directly to viser, then read back camera state.
                # Building c2w locally produces black renders for unknown reasons,
                # even though the matrices appear mathematically identical.
                # =====================================================================

                # Clear pending events, don't interrupt during rotation
                self._render_event.clear()
                self._may_interrupt_render = False

                # Advance rotation using quaternion math (pushes directly to viser)
                camera_ctrl.rotation_step()

                # Throttle slider sync to every 250ms
                current_time = time.perf_counter()
                if current_time - self._last_slider_sync > 0.25:
                    camera_ctrl.trigger_slider_sync()
                    self._last_slider_sync = current_time

                # rotation_step() already pushed to viser (or stored headless state)

                # Get camera for rendering (from viser or headless state)
                camera_state = self._get_render_camera()
                if camera_state is None:
                    time.sleep(0.05)
                    continue

                task = RenderTask("move", camera_state)
                render_start = time.perf_counter()
            else:
                # Normal mode: wait for explicit render events or periodic static renders
                if not self._render_event.wait(0.2):
                    # Timeout - do a static render with fresh camera from viser
                    camera_state = self._get_render_camera()
                    if camera_state is not None:
                        self.submit(RenderTask("static", camera_state))
                    continue

                self._render_event.clear()
                task = self._task
                if task is None:
                    continue

            if self._state == "high" and task.action == "static":
                continue

            self._state = self.transitions[self._state][task.action]

            # Use task's camera state, or get fresh from viser
            camera_state = task.camera_state
            if camera_state is None:
                camera_state = self._get_render_camera()
            if camera_state is None:
                continue

            # Determine JPEG quality BEFORE render (so render_fn can use it for encoding)
            if (
                getattr(self.viewer, "auto_quality_enabled", False)
                and task.action == "move"
            ):
                jpeg_quality = self.viewer.jpeg_quality_move
            else:
                jpeg_quality = self.viewer.jpeg_quality_static

            # Get current frame index BEFORE render to detect if it changes
            current_frame_index = self._get_current_frame_index()

            try:
                with self.lock, set_trace_context(self._may_interrupt_trace):
                    tic = time.perf_counter()
                    W, H = img_wh = self._get_img_wh(camera_state.aspect)
                    self.viewer.render_tab_state.viewer_width = W
                    self.viewer.render_tab_state.viewer_height = H
                    self.viewer.render_tab_state.jpeg_quality = jpeg_quality

                    if not self._old_version:
                        try:
                            rendered = self.viewer.render_fn(
                                camera_state,
                                self.viewer.render_tab_state,
                            )
                        except TypeError:
                            self._old_version = True
                            logger.warning(
                                "API deprecated - please update render_fn signature"
                            )
                            rendered = self.viewer.render_fn(camera_state, img_wh)
                    else:
                        rendered = self.viewer.render_fn(camera_state, img_wh)

                    self.viewer._after_render()
                    if isinstance(rendered, tuple):
                        img, depth = rendered
                    else:
                        img, depth = rendered, None
                    self.viewer.render_tab_state.num_view_rays_per_sec = (W * H) / (
                        time.perf_counter() - tic
                    )

                    # Cache this complete frame
                    self._cached_frame = CachedFrame(img, depth, current_frame_index)

            except InterruptRenderException:
                # Render interrupted - try to reuse cached frame if valid
                if self._cached_frame is not None and self._cached_frame.is_valid_for(
                    current_frame_index
                ):
                    img, depth = self._cached_frame.img, self._cached_frame.depth
                    _skip_gpu_broadcast = True
                else:
                    # No valid cached frame - skip to avoid showing stale content
                    continue
            except Exception as e:
                # Graceful error handling instead of fatal exit
                self._consecutive_failures = getattr(self, "_consecutive_failures", 0) + 1
                logger.error(f"SharedRenderer error ({self._consecutive_failures}): {e}")
                traceback.print_exc()

                if self._consecutive_failures > 5:
                    logger.warning("Multiple render failures, pausing briefly")
                    time.sleep(1.0)
                    self._consecutive_failures = 0

                # Use cached frame if available
                if self._cached_frame is not None:
                    img, depth = self._cached_frame.img, self._cached_frame.depth
                    _skip_gpu_broadcast = True
                else:
                    continue
            else:
                # Normal render completed - reset failure count
                self._consecutive_failures = 0
                _skip_gpu_broadcast = False

            # BROADCAST to ALL clients (pre-encoded JPEG)
            if not _skip_gpu_broadcast:
                self._broadcast_frame(depth)

            # Frame rate limiting for rotation mode (~20 FPS)
            # This prevents overwhelming the GPU and ensures smooth rotation
            if rotation_active:
                elapsed = time.perf_counter() - render_start
                target_frame_time = 0.05  # 20 FPS
                if elapsed < target_frame_time:
                    time.sleep(target_frame_time - elapsed)

