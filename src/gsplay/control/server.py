"""HTTP control server for remote GSPlay commands.

Provides a simple HTTP API for external processes to control a running
GSPlay instance. Runs on viser_port + 2.

Endpoints
---------
POST /center-scene
    Calculate scene centroid and apply inverse translation to center at origin.

POST /get-state
    Get current viewer state (playback, colors, transforms).

POST /set-translation
    Set scene translation directly.

POST /rotate-cw
    Start clockwise camera rotation. Optional body: {"speed": 30.0}

POST /rotate-ccw
    Start counter-clockwise camera rotation. Optional body: {"speed": 30.0}

POST /rotate-stop
    Stop camera rotation.

GET /rotation-state
    Get current rotation state (active, speed, direction).

GET /playback-state
    Get current animation playback state (playing: true/false).

POST /play
    Start animation playback.

POST /pause
    Pause animation playback.

POST /toggle-playback
    Toggle animation playback state.
"""

from __future__ import annotations

import json
import logging
import threading
from functools import partial
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.gsplay.core.app import UniversalGSPlay

logger = logging.getLogger(__name__)


class ControlRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for control commands."""

    viewer: "UniversalGSPlay"

    def log_message(self, format: str, *args) -> None:
        """Override to use logging instead of stderr."""
        logger.debug(f"Control API: {format % args}")

    def _send_json(self, data: dict, status: int = 200) -> None:
        """Send JSON response."""
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, error: str, status: int = 400) -> None:
        """Send JSON error response."""
        self._send_json({"ok": False, "error": error}, status)

    def do_OPTIONS(self) -> None:
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        """Handle GET requests."""
        path = self.path.strip("/")

        if path == "rotation-state":
            self._handle_get_rotation_state()
        elif path == "playback-state":
            self._handle_get_playback_state()
        else:
            self._send_error_json(f"Unknown endpoint: {path}", 404)

    def do_POST(self) -> None:
        """Handle POST requests."""
        path = self.path.strip("/")

        if path == "center-scene":
            self._handle_center_scene()
        elif path == "get-state":
            self._handle_get_state()
        elif path == "set-translation":
            self._handle_set_translation()
        elif path == "rotate-cw":
            self._handle_rotate_cw()
        elif path == "rotate-ccw":
            self._handle_rotate_ccw()
        elif path == "rotate-stop":
            self._handle_rotate_stop()
        elif path == "play":
            self._handle_play()
        elif path == "pause":
            self._handle_pause()
        elif path == "toggle-playback":
            self._handle_toggle_playback()
        else:
            self._send_error_json(f"Unknown command: {path}", 404)

    def _read_json_body(self) -> dict | None:
        """Read and parse JSON request body."""
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            return {}
        try:
            body = self.rfile.read(content_length)
            return json.loads(body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.warning(f"Failed to parse request body: {e}")
            return None

    def _handle_center_scene(self) -> None:
        """Calculate centroid and apply inverse translation.

        Delegates to viewer's _handle_center_scene() which respects filters.
        """
        viewer = self.viewer

        if viewer.model is None:
            self._send_error_json("No model loaded", 503)
            return

        try:
            # Get current frame gaussians for response data
            gaussians = viewer._get_current_frame_gaussians()
            if gaussians is None:
                self._send_error_json("No gaussians available", 503)
                return

            # Get means for centroid calculation (respecting filters)
            means = gaussians.means
            if hasattr(means, "detach"):
                means = means.detach().cpu().numpy()
            elif not isinstance(means, np.ndarray):
                means = np.array(means)

            total_count = len(means)

            # Apply filter mask if filtering is active
            fv = viewer.config.filter_values
            if not fv.is_neutral():
                try:
                    from gsmod.filter.apply import compute_filter_mask

                    class _GaussianData:
                        pass

                    data = _GaussianData()
                    data.means = means

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

                    mask = compute_filter_mask(data, fv)
                    means = means[mask]
                except Exception as e:
                    logger.warning(f"Filter mask failed: {e}, using all gaussians")

            if len(means) == 0:
                self._send_error_json("No gaussians after filtering", 503)
                return

            centroid = np.mean(means, axis=0).tolist()

            # Apply inverse translation to center at origin
            # gsmod formula: new_pos = old_pos + translate
            # To move centroid to origin: 0 = centroid + translate -> translate = -centroid
            tx, ty, tz = -centroid[0], -centroid[1], -centroid[2]
            viewer.api.set_translation(tx, ty, tz)

            # Trigger rerender
            viewer.render_component.rerender()

            self._send_json({
                "ok": True,
                "centroid": centroid,
                "translation_applied": [tx, ty, tz],
                "filtered_count": len(means),
                "total_count": total_count,
            })
            logger.info(f"Scene centered: centroid={centroid} ({len(means)}/{total_count} gaussians)")

        except Exception as e:
            logger.error(f"center-scene failed: {e}", exc_info=True)
            self._send_error_json(str(e), 500)

    def _handle_get_state(self) -> None:
        """Get current viewer state."""
        viewer = self.viewer

        try:
            state = viewer.api.get_state()
            self._send_json({
                "ok": True,
                "state": state.to_dict(),
            })
        except Exception as e:
            logger.error(f"get-state failed: {e}", exc_info=True)
            self._send_error_json(str(e), 500)

    def _handle_set_translation(self) -> None:
        """Set scene translation directly."""
        viewer = self.viewer
        body = self._read_json_body()

        if body is None:
            self._send_error_json("Invalid JSON body", 400)
            return

        x = body.get("x", 0.0)
        y = body.get("y", 0.0)
        z = body.get("z", 0.0)

        try:
            viewer.api.set_translation(float(x), float(y), float(z))
            viewer.render_component.rerender()

            self._send_json({
                "ok": True,
                "translation": [x, y, z],
            })
            logger.info(f"Translation set: ({x}, {y}, {z})")

        except Exception as e:
            logger.error(f"set-translation failed: {e}", exc_info=True)
            self._send_error_json(str(e), 500)

    def _handle_rotate_cw(self) -> None:
        """Start clockwise camera rotation."""
        viewer = self.viewer
        body = self._read_json_body() or {}
        speed = body.get("speed", 30.0)

        try:
            viewer.api.rotate_cw(float(speed))
            self._send_json({"ok": True, "direction": "cw", "speed": speed})
            logger.info(f"Camera rotation started: CW at {speed} deg/sec")
        except Exception as e:
            logger.error(f"rotate-cw failed: {e}", exc_info=True)
            self._send_error_json(str(e), 500)

    def _handle_rotate_ccw(self) -> None:
        """Start counter-clockwise camera rotation."""
        viewer = self.viewer
        body = self._read_json_body() or {}
        speed = body.get("speed", 30.0)

        try:
            viewer.api.rotate_ccw(float(speed))
            self._send_json({"ok": True, "direction": "ccw", "speed": speed})
            logger.info(f"Camera rotation started: CCW at {speed} deg/sec")
        except Exception as e:
            logger.error(f"rotate-ccw failed: {e}", exc_info=True)
            self._send_error_json(str(e), 500)

    def _handle_rotate_stop(self) -> None:
        """Stop camera rotation."""
        viewer = self.viewer

        try:
            viewer.api.stop_rotation()
            self._send_json({"ok": True})
            logger.info("Camera rotation stopped")
        except Exception as e:
            logger.error(f"rotate-stop failed: {e}", exc_info=True)
            self._send_error_json(str(e), 500)

    def _handle_get_rotation_state(self) -> None:
        """Get current rotation state."""
        viewer = self.viewer

        try:
            if viewer.camera_controller:
                state = viewer.camera_controller.get_rotation_state()
                self._send_json({"ok": True, **state})
            else:
                self._send_json({
                    "ok": True,
                    "active": False,
                    "speed": 0.0,
                    "axis": "y",
                    "direction": "stopped",
                })
        except Exception as e:
            logger.error(f"rotation-state failed: {e}", exc_info=True)
            self._send_error_json(str(e), 500)

    def _handle_get_playback_state(self) -> None:
        """Get current playback state."""
        viewer = self.viewer

        try:
            is_playing = viewer.api.is_playing()
            self._send_json({"ok": True, "playing": is_playing})
        except Exception as e:
            logger.error(f"playback-state failed: {e}", exc_info=True)
            self._send_error_json(str(e), 500)

    def _handle_play(self) -> None:
        """Start animation playback."""
        viewer = self.viewer

        try:
            viewer.api.play()
            self._send_json({"ok": True, "playing": True})
            logger.info("Playback started")
        except Exception as e:
            logger.error(f"play failed: {e}", exc_info=True)
            self._send_error_json(str(e), 500)

    def _handle_pause(self) -> None:
        """Pause animation playback."""
        viewer = self.viewer

        try:
            viewer.api.pause()
            self._send_json({"ok": True, "playing": False})
            logger.info("Playback paused")
        except Exception as e:
            logger.error(f"pause failed: {e}", exc_info=True)
            self._send_error_json(str(e), 500)

    def _handle_toggle_playback(self) -> None:
        """Toggle animation playback state."""
        viewer = self.viewer

        try:
            is_playing = viewer.api.is_playing()
            if is_playing:
                viewer.api.pause()
            else:
                viewer.api.play()
            new_state = not is_playing
            self._send_json({"ok": True, "playing": new_state})
            logger.info(f"Playback toggled: {'playing' if new_state else 'paused'}")
        except Exception as e:
            logger.error(f"toggle-playback failed: {e}", exc_info=True)
            self._send_error_json(str(e), 500)


class ControlServer:
    """HTTP server for remote control of GSPlay instance.

    Runs in a daemon thread, listening on viser_port + 2.
    """

    def __init__(self, port: int, viewer: "UniversalGSPlay"):
        """Initialize control server.

        Parameters
        ----------
        port : int
            Port to bind to.
        viewer : UniversalGSPlay
            The viewer instance to control.
        """
        self.port = port
        self.viewer = viewer
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._started_event = threading.Event()

    def start(self) -> int:
        """Start the control server.

        Returns
        -------
        int
            The port the server is running on.
        """
        if self._server is not None:
            return self.port

        self._started_event.clear()
        self._thread = threading.Thread(target=self._run_server, daemon=True)
        self._thread.start()

        # Wait for server to start
        if not self._started_event.wait(timeout=5.0):
            raise RuntimeError("Control server failed to start within timeout")

        return self.port

    def _run_server(self) -> None:
        """Run HTTP server in dedicated thread."""
        try:
            # Create handler class with viewer reference
            handler = type(
                "BoundControlHandler",
                (ControlRequestHandler,),
                {"viewer": self.viewer},
            )

            self._server = HTTPServer(("0.0.0.0", self.port), handler)
            self._started_event.set()

            logger.debug(f"Control server listening on port {self.port}")
            self._server.serve_forever()

        except Exception as e:
            logger.error(f"Control server error: {e}")
            self._started_event.set()

    def stop(self) -> None:
        """Stop the control server."""
        if self._server:
            self._server.shutdown()
            self._server = None
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        logger.debug("Control server stopped")
