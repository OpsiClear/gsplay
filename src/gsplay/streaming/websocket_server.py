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
    <title>GSPlay WebSocket Stream</title>
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
        .info-bar {
            height: 36px;
            background: rgba(20,20,20,0.95);
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 12px;
            font-size: 11px;
            color: #888;
            border-top: 1px solid #222;
        }
        .info-bar .title {
            color: #aaa;
            font-weight: 500;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .info-bar .badge {
            background: #059669;
            color: white;
            padding: 2px 5px;
            border-radius: 3px;
            font-size: 9px;
            font-weight: 600;
        }
        .info-bar .status {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .info-bar .stat {
            display: flex;
            align-items: center;
            gap: 4px;
        }
        .info-bar .dot {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: #444;
        }
        .info-bar .dot.connected { background: #22c55e; }
        .info-bar .dot.connecting { background: #eab308; }
        .info-bar .dot.error { background: #ef4444; }
        .info-bar button {
            background: transparent;
            border: 1px solid #333;
            color: #888;
            padding: 4px 10px;
            border-radius: 3px;
            cursor: pointer;
            font-size: 10px;
        }
        .info-bar button:hover {
            background: #222;
            color: #ccc;
        }
        .info-bar button.recording {
            background: #dc2626;
            border-color: #dc2626;
            color: white;
            animation: rec-pulse 1s infinite;
        }
        .info-bar button.streaming {
            background: #059669;
            border-color: #059669;
            color: white;
        }
        @keyframes rec-pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.6; }
        }
        .rec-indicator {
            position: absolute;
            top: 12px;
            right: 12px;
            background: #dc2626;
            color: white;
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
            animation: rec-pulse 1s infinite;
            display: none;
        }
        .rec-indicator.visible { display: block; }
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
            <div class="rec-indicator" id="recIndicator">● REC</div>
        </div>
        <div class="info-bar">
            <span class="title">
                GSPlay Stream
                <span class="badge">WS</span>
            </span>
            <div class="status">
                <div class="stat">
                    <span class="dot" id="dot"></span>
                    <span id="statusText">Connecting</span>
                </div>
                <div class="stat" id="fpsContainer">
                    <span id="fps">--</span> fps
                </div>
                <div class="stat" id="latencyContainer">
                    <span id="latency">--</span> ms
                </div>
                <button id="streamBtn" onclick="toggleStream()">⏹ Stop</button>
                <button id="recordBtn" onclick="toggleRecording()">⏺ Record</button>
                <button onclick="toggleFullscreen()">Fullscreen</button>
            </div>
        </div>
    </div>

    <script>
        const img = document.getElementById('stream');
        const loading = document.getElementById('loading');
        const dot = document.getElementById('dot');
        const statusText = document.getElementById('statusText');
        const fpsEl = document.getElementById('fps');
        const latencyEl = document.getElementById('latency');
        const streamBtn = document.getElementById('streamBtn');
        const recordBtn = document.getElementById('recordBtn');
        const recIndicator = document.getElementById('recIndicator');

        let ws = null;
        let frameCount = 0;
        let lastFpsTime = performance.now();
        let currentBlobUrl = null;
        let reconnectTimer = null;
        let streamEnabled = true;

        // Recording state
        let isRecording = false;
        let mediaRecorder = null;
        let recordedChunks = [];
        let recordCanvas = null;
        let recordCtx = null;

        function setStatus(state, text) {
            dot.className = 'dot ' + state;
            statusText.textContent = text;
            if (state === 'connected') {
                loading.classList.add('hidden');
            } else {
                loading.classList.remove('hidden');
            }
        }

        function connect() {
            if (!streamEnabled) return;
            if (ws) {
                ws.close();
            }

            const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = protocol + '//' + location.host + '/';

            setStatus('connecting', 'Connecting...');
            ws = new WebSocket(wsUrl);
            ws.binaryType = 'arraybuffer';

            ws.onopen = () => {
                setStatus('connected', 'Live');
                updateStreamBtn();
                measureLatency();
            };

            ws.onmessage = (event) => {
                if (typeof event.data === 'string') {
                    if (event.data === 'pong') {
                        const latency = performance.now() - window._pingTime;
                        latencyEl.textContent = Math.round(latency / 2);
                    }
                    return;
                }

                const blob = new Blob([event.data], { type: 'image/jpeg' });
                const url = URL.createObjectURL(blob);

                if (currentBlobUrl) {
                    URL.revokeObjectURL(currentBlobUrl);
                }
                currentBlobUrl = url;
                img.src = url;

                // Draw to recording canvas if recording
                if (isRecording && recordCtx) {
                    const tempImg = new Image();
                    tempImg.onload = () => {
                        const scale = Math.min(recordCanvas.width / tempImg.width, recordCanvas.height / tempImg.height);
                        const x = (recordCanvas.width - tempImg.width * scale) / 2;
                        const y = (recordCanvas.height - tempImg.height * scale) / 2;
                        recordCtx.fillStyle = '#000';
                        recordCtx.fillRect(0, 0, recordCanvas.width, recordCanvas.height);
                        recordCtx.drawImage(tempImg, x, y, tempImg.width * scale, tempImg.height * scale);
                    };
                    tempImg.src = url;
                }

                frameCount++;
                const now = performance.now();
                const elapsed = now - lastFpsTime;
                if (elapsed >= 1000) {
                    fpsEl.textContent = Math.round(frameCount * 1000 / elapsed);
                    frameCount = 0;
                    lastFpsTime = now;
                }
            };

            ws.onerror = () => {
                setStatus('error', 'Error');
            };

            ws.onclose = () => {
                if (streamEnabled) {
                    setStatus('error', 'Disconnected');
                    if (reconnectTimer) clearTimeout(reconnectTimer);
                    reconnectTimer = setTimeout(connect, 2000);
                } else {
                    setStatus('error', 'Stopped');
                }
                updateStreamBtn();
            };
        }

        function disconnect() {
            if (reconnectTimer) {
                clearTimeout(reconnectTimer);
                reconnectTimer = null;
            }
            if (ws) {
                ws.close();
                ws = null;
            }
            setStatus('error', 'Stopped');
            updateStreamBtn();
        }

        function toggleStream() {
            streamEnabled = !streamEnabled;
            if (streamEnabled) {
                connect();
            } else {
                // Stop recording if active
                if (isRecording) stopRecording();
                disconnect();
            }
        }

        function updateStreamBtn() {
            if (streamEnabled && ws && ws.readyState === WebSocket.OPEN) {
                streamBtn.textContent = '⏹ Stop';
                streamBtn.classList.add('streaming');
            } else {
                streamBtn.textContent = '▶ Start';
                streamBtn.classList.remove('streaming');
            }
        }

        function measureLatency() {
            if (ws && ws.readyState === WebSocket.OPEN) {
                window._pingTime = performance.now();
                ws.send('ping');
                setTimeout(measureLatency, 2000);
            }
        }

        function toggleRecording() {
            if (isRecording) {
                stopRecording();
            } else {
                startRecording();
            }
        }

        function startRecording() {
            recordCanvas = document.createElement('canvas');
            recordCanvas.width = 1920;
            recordCanvas.height = 1080;
            recordCtx = recordCanvas.getContext('2d');

            const stream = recordCanvas.captureStream(30);
            mediaRecorder = new MediaRecorder(stream, {
                mimeType: 'video/webm;codecs=vp9',
                videoBitsPerSecond: 8000000
            });

            recordedChunks = [];
            mediaRecorder.ondataavailable = (e) => {
                if (e.data.size > 0) recordedChunks.push(e.data);
            };

            mediaRecorder.onstop = () => {
                const blob = new Blob(recordedChunks, { type: 'video/webm' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                const timestamp = new Date().toISOString().slice(0, 19).replace(/[:.]/g, '-');
                a.download = 'gsplay_stream_' + timestamp + '.webm';
                a.click();
                URL.revokeObjectURL(url);
            };

            mediaRecorder.start(100);
            isRecording = true;
            recordBtn.textContent = '⏹ Stop';
            recordBtn.classList.add('recording');
            recIndicator.classList.add('visible');
        }

        function stopRecording() {
            if (mediaRecorder && mediaRecorder.state !== 'inactive') {
                mediaRecorder.stop();
            }
            isRecording = false;
            recordBtn.textContent = '⏺ Record';
            recordBtn.classList.remove('recording');
            recIndicator.classList.remove('visible');
        }

        function toggleFullscreen() {
            if (!document.fullscreenElement) {
                document.documentElement.requestFullscreen().catch(() => {});
            } else {
                document.exitFullscreen().catch(() => {});
            }
        }

        document.addEventListener('keydown', (e) => {
            if (e.key === 'f' || e.key === 'F') toggleFullscreen();
            if (e.key === 's' || e.key === 'S') toggleStream();
            if ((e.key === 'r' || e.key === 'R') && !e.ctrlKey) {
                if (e.shiftKey) {
                    toggleRecording();
                } else {
                    connect();
                }
            }
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
