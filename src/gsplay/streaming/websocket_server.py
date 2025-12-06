"""WebSocket streaming server for low-latency JPEG delivery.

This module provides low-latency WebSocket streaming through:
- Persistent connection (no HTTP overhead per frame)
- Push-based delivery (no polling)
- Binary transport (direct JPEG bytes)

Architecture
------------
    ┌─────────────────────────────────────────────────────────────────┐
    │                    WebSocketStreamServer                        │
    │  - Async WebSocket server (websockets library)                  │
    │  - Broadcasts JPEG frames to all connected clients              │
    │  - Thread-safe FrameBuffer for cross-thread frame sharing       │
    └─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                       Browser Client                            │
    │  - WebSocket binary messages                                    │
    │  - Direct blob URL to <img> tag                                 │
    │  - ~100-150ms latency                                           │
    └─────────────────────────────────────────────────────────────────┘

Usage
-----
    from src.gsplay.streaming.websocket_server import WebSocketStreamServer

    server = WebSocketStreamServer(port=8080, frame_buffer=frame_buffer)
    server.start()

    # Viewer accesses: http://localhost:8080/ (HTML page)
    # WebSocket connects to: ws://localhost:8080/ws
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from contextlib import suppress
from http import HTTPStatus
from typing import Set

logger = logging.getLogger(__name__)


class FrameBuffer:
    """Thread-safe frame buffer for storing the latest rendered frame."""

    def __init__(self):
        self._frame: bytes | None = None
        self._frame_id: int = 0
        self._lock = threading.Lock()
        self._new_frame_event = threading.Event()

    def set_frame(self, jpeg_data: bytes) -> None:
        """Store a new JPEG frame.

        Parameters
        ----------
        jpeg_data : bytes
            JPEG-encoded frame data.
        """
        with self._lock:
            self._frame = jpeg_data
            self._frame_id += 1
        self._new_frame_event.set()

    def get_frame(self) -> tuple[bytes | None, int]:
        """Get the current frame.

        Returns
        -------
        tuple[bytes | None, int]
            (jpeg_data, frame_id) tuple.
        """
        with self._lock:
            return self._frame, self._frame_id

    def wait_for_new_frame(self, timeout: float = 1.0) -> bool:
        """Wait for a new frame to be available.

        Parameters
        ----------
        timeout : float
            Maximum time to wait in seconds.

        Returns
        -------
        bool
            True if a new frame is available, False if timeout.
        """
        result = self._new_frame_event.wait(timeout)
        self._new_frame_event.clear()
        return result

# Check for websockets availability
_WEBSOCKETS_AVAILABLE = False
try:
    import websockets
    from websockets.asyncio.server import serve, ServerConnection
    from websockets.http11 import Request, Response
    _WEBSOCKETS_AVAILABLE = True
except ImportError:
    logger.debug("websockets not available - WebSocket streaming disabled")
    websockets = None


def is_websocket_available() -> bool:
    """Check if websockets library is installed."""
    return _WEBSOCKETS_AVAILABLE


class WebSocketStreamServer:
    """WebSocket server for low-latency JPEG streaming.

    Uses the lightweight `websockets` library (single dependency) to provide
    push-based binary frame delivery with ~100-150ms latency.
    """

    def __init__(
        self,
        port: int = 0,
        target_fps: int = 30,
        frame_buffer: "FrameBuffer | None" = None,
    ):
        """Initialize the WebSocket stream server.

        Parameters
        ----------
        port : int
            Port to bind to. Use 0 for auto-assignment.
        target_fps : int
            Target frame rate for streaming.
        frame_buffer : FrameBuffer | None
            Shared frame buffer. If None, creates a new one.
        """
        if not _WEBSOCKETS_AVAILABLE:
            raise RuntimeError(
                "websockets not installed. Run: pip install websockets"
            )

        self.port = port
        self.target_fps = target_fps
        self._frame_interval = 1.0 / target_fps

        # Create or use provided frame buffer
        if frame_buffer is None:
            self.frame_buffer = FrameBuffer()
        else:
            self.frame_buffer = frame_buffer

        self._clients: Set[ServerConnection] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._started_event = threading.Event()
        self._server = None
        self._broadcast_task: asyncio.Task | None = None

    def start(self) -> int:
        """Start the WebSocket streaming server.

        Returns
        -------
        int
            The port the server is running on.
        """
        if self._running:
            return self.port

        self._running = True
        self._started_event.clear()
        self._thread = threading.Thread(target=self._run_async_server, daemon=True)
        self._thread.start()

        # Wait for server to start (with timeout)
        if not self._started_event.wait(timeout=10.0):
            self._running = False
            raise RuntimeError("WebSocket server failed to start within timeout")

        return self.port

    def _run_async_server(self) -> None:
        """Run the async server in a dedicated thread."""
        try:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._start_server())
            self._loop.run_forever()
        except Exception as e:
            logger.error(f"WebSocket server error: {e}")
            self._started_event.set()
        finally:
            if self._loop:
                self._loop.close()

    async def _start_server(self) -> None:
        """Start the WebSocket server."""
        self._server = await serve(
            self._handle_client,
            "0.0.0.0",
            self.port,
            process_request=self._handle_http,
        )

        # Get actual port if auto-assigned
        for sock in self._server.sockets:
            addr = sock.getsockname()
            self.port = addr[1]
            break

        self._started_event.set()
        logger.info(f"WebSocket stream server started on port {self.port}")

        # Start broadcast task
        self._broadcast_task = asyncio.create_task(self._broadcast_loop())

    async def _handle_http(
        self, connection: ServerConnection, request: Request
    ) -> Response | None:
        """Handle HTTP requests (serve HTML page).

        Returns None for WebSocket upgrade requests to allow the upgrade to proceed.
        """
        # Check for WebSocket upgrade request - let these pass through
        upgrade_header = request.headers.get("Upgrade", "").lower()
        if upgrade_header == "websocket":
            return None  # Allow WebSocket upgrade to proceed

        path = request.path.split("?")[0].rstrip("/")

        # Serve HTML viewer page for root path (HTTP GET only, not WebSocket)
        if path in ("", "/", "/view"):
            html = self._get_viewer_html()
            return Response(
                HTTPStatus.OK,
                "OK",
                websockets.Headers([
                    ("Content-Type", "text/html; charset=utf-8"),
                    ("Content-Length", str(len(html))),
                ]),
                html.encode(),
            )

        # Status endpoint
        if path == "/status":
            import json
            frame_data, frame_id = self.frame_buffer.get_frame()
            status = json.dumps({
                "ok": True,
                "clients": len(self._clients),
                "has_frame": frame_data is not None,
                "frame_id": frame_id,
            })
            return Response(
                HTTPStatus.OK,
                "OK",
                websockets.Headers([
                    ("Content-Type", "application/json"),
                    ("Content-Length", str(len(status))),
                ]),
                status.encode(),
            )

        # Let WebSocket upgrade requests pass through
        return None

    async def _handle_client(self, websocket: ServerConnection) -> None:
        """Handle a WebSocket client connection."""
        self._clients.add(websocket)
        client_id = id(websocket)
        logger.debug(f"WebSocket client connected: {client_id}")

        try:
            # Keep connection alive, handle any incoming messages
            async for message in websocket:
                # Client can send "ping" for latency measurement
                if message == "ping":
                    await websocket.send("pong")
        except websockets.ConnectionClosed:
            pass
        except Exception as e:
            logger.debug(f"WebSocket client error: {e}")
        finally:
            self._clients.discard(websocket)
            logger.debug(f"WebSocket client disconnected: {client_id}")

    async def _broadcast_loop(self) -> None:
        """Continuously broadcast frames to all connected clients."""
        last_frame_id = -1
        last_send_time = 0.0

        while self._running:
            try:
                # Wait for frame interval
                await asyncio.sleep(self._frame_interval * 0.5)

                # Get current frame
                frame_data, frame_id = self.frame_buffer.get_frame()

                # Skip if no frame or same frame (unless keepalive needed)
                now = time.time()
                if frame_data is None:
                    continue

                is_new_frame = frame_id != last_frame_id
                needs_keepalive = (now - last_send_time) >= 1.0

                if not is_new_frame and not needs_keepalive:
                    continue

                last_frame_id = frame_id
                last_send_time = now

                # Broadcast to all clients
                if self._clients:
                    # Use websockets.broadcast for efficient multi-send
                    websockets.broadcast(self._clients, frame_data)

            except Exception as e:
                if self._running:
                    logger.debug(f"Broadcast error: {e}")
                await asyncio.sleep(0.1)

    def stop(self) -> None:
        """Stop the WebSocket streaming server."""
        self._running = False

        if self._loop and self._server:
            # Close all client connections and drain background tasks
            async def close_all():
                if self._broadcast_task:
                    self._broadcast_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await self._broadcast_task

                for client in list(self._clients):
                    try:
                        await client.close()
                    except Exception:
                        pass

                self._server.close()
                await self._server.wait_closed()

            future = asyncio.run_coroutine_threadsafe(close_all(), self._loop)
            with suppress(Exception):
                future.result(timeout=5.0)

            self._loop.call_soon_threadsafe(self._loop.stop)

        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None
            self._broadcast_task = None
            self._server = None

        logger.info("WebSocket stream server stopped")

    def broadcast_jpeg(self, jpeg_data: bytes) -> None:
        """Broadcast a JPEG frame (compatibility with StreamServer API).

        Parameters
        ----------
        jpeg_data : bytes
            JPEG-encoded frame data.
        """
        self.frame_buffer.set_frame(jpeg_data)

    def _get_viewer_html(self) -> str:
        """Generate the WebSocket viewer HTML page."""
        return """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, user-scalable=no">
    <title>GStream</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        html, body {
            width: 100%;
            height: 100%;
            overflow: hidden;
            background: #000;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }
        .viewport {
            width: 100%;
            height: 100%;
            display: flex;
            flex-direction: column;
        }
        .stream-area {
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
            position: relative;
        }
        #stream {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }
        .loading {
            position: absolute;
            color: #666;
            font-size: 14px;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 12px;
        }
        .loading.hidden { display: none; }
        .spinner {
            width: 24px;
            height: 24px;
            border: 2px solid #333;
            border-top-color: #666;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        .toolbar {
            height: 36px;
            background: rgba(20,20,20,0.95);
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 0 12px;
            gap: 8px;
            border-top: 1px solid #222;
        }
        .toolbar button {
            background: transparent;
            border: 1px solid #333;
            color: #888;
            padding: 4px 12px;
            border-radius: 3px;
            cursor: pointer;
            font-size: 11px;
        }
        .toolbar button:hover {
            background: #222;
            color: #ccc;
        }
        .toolbar button.active {
            background: #0891b2;
            border-color: #0891b2;
            color: white;
        }
    </style>
</head>
<body>
    <div class="viewport">
        <div class="stream-area">
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <span>Connecting...</span>
            </div>
            <img id="stream" alt="Stream">
        </div>
        <div class="toolbar">
            <button onclick="rotateCCW()" title="Rotate CCW [Left]">&lt;</button>
            <button onclick="rotateStop()" title="Stop rotation [Esc]">||</button>
            <button onclick="rotateCW()" title="Rotate CW [Right]">&gt;</button>
            <button id="playBtn" onclick="togglePlayback()" title="Play/Pause [Space]">Play</button>
            <button onclick="toggleFullscreen()" title="Fullscreen [F]">[ ]</button>
        </div>
    </div>

    <script>
        const img = document.getElementById('stream');
        const loading = document.getElementById('loading');
        const playBtn = document.getElementById('playBtn');

        let ws = null;
        let currentBlobUrl = null;
        let reconnectTimer = null;
        let isPlaying = false;

        function connect() {
            if (ws) ws.close();

            const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(protocol + '//' + location.host + '/');
            ws.binaryType = 'arraybuffer';

            ws.onopen = () => loading.classList.add('hidden');

            ws.onmessage = (event) => {
                if (typeof event.data === 'string') return;
                const blob = new Blob([event.data], { type: 'image/jpeg' });
                const url = URL.createObjectURL(blob);
                if (currentBlobUrl) URL.revokeObjectURL(currentBlobUrl);
                currentBlobUrl = url;
                img.src = url;
            };

            ws.onclose = () => {
                loading.classList.remove('hidden');
                if (reconnectTimer) clearTimeout(reconnectTimer);
                reconnectTimer = setTimeout(connect, 2000);
            };
        }

        function getControlUrl() {
            const streamPort = parseInt(location.port);
            return location.protocol + '//' + location.hostname + ':' + (streamPort + 1);
        }

        function togglePlayback() {
            fetch(getControlUrl() + '/toggle-playback', {method: 'POST'})
                .then(r => r.json())
                .then(data => {
                    if (data.ok) {
                        isPlaying = data.playing;
                        playBtn.textContent = isPlaying ? 'Pause' : 'Play';
                        playBtn.classList.toggle('active', isPlaying);
                    }
                })
                .catch(() => {});
        }

        function rotateCW() { fetch(getControlUrl() + '/rotate-cw', {method: 'POST'}).catch(() => {}); }
        function rotateCCW() { fetch(getControlUrl() + '/rotate-ccw', {method: 'POST'}).catch(() => {}); }
        function rotateStop() { fetch(getControlUrl() + '/rotate-stop', {method: 'POST'}).catch(() => {}); }

        function toggleFullscreen() {
            if (!document.fullscreenElement) {
                document.documentElement.requestFullscreen().catch(() => {});
            } else {
                document.exitFullscreen().catch(() => {});
            }
        }

        document.addEventListener('keydown', (e) => {
            if (e.key === 'f' || e.key === 'F') toggleFullscreen();
            if (e.key === ' ') { e.preventDefault(); togglePlayback(); }
            if (e.key === 'ArrowLeft') rotateCCW();
            if (e.key === 'ArrowRight') rotateCW();
            if (e.key === 'Escape') rotateStop();
        });

        connect();
    </script>
</body>
</html>"""


# =============================================================================
# Module-Level API
# =============================================================================


_websocket_server: WebSocketStreamServer | None = None


def get_websocket_server() -> WebSocketStreamServer | None:
    """Get the global WebSocket stream server instance."""
    return _websocket_server


def set_websocket_server(server: WebSocketStreamServer) -> None:
    """Set the global WebSocket stream server instance."""
    global _websocket_server
    _websocket_server = server
