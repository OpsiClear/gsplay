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
    from websockets.asyncio.server import ServerConnection, serve
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
        frame_buffer: FrameBuffer | None = None,
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
            raise RuntimeError("websockets not installed. Run: pip install websockets")

        self.port = port
        self.target_fps = target_fps
        self._frame_interval = 1.0 / target_fps

        # Create or use provided frame buffer
        if frame_buffer is None:
            self.frame_buffer = FrameBuffer()
        else:
            self.frame_buffer = frame_buffer

        self._clients: set[ServerConnection] = set()
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

    async def _handle_http(self, connection: ServerConnection, request: Request) -> Response | None:
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
                websockets.Headers(
                    [
                        ("Content-Type", "text/html; charset=utf-8"),
                        ("Content-Length", str(len(html))),
                    ]
                ),
                html.encode(),
            )

        # Status endpoint
        if path == "/status":
            import json

            frame_data, frame_id = self.frame_buffer.get_frame()
            status = json.dumps(
                {
                    "ok": True,
                    "clients": len(self._clients),
                    "has_frame": frame_data is not None,
                    "frame_id": frame_id,
                }
            )
            return Response(
                HTTPStatus.OK,
                "OK",
                websockets.Headers(
                    [
                        ("Content-Type", "application/json"),
                        ("Content-Length", str(len(status))),
                    ]
                ),
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
        return r"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, user-scalable=no">
    <title>GSPlay</title>
    <style>
        /* Industrial Minimalist Dark Theme */
        :root {
            --bg-void: #0a0a0a;
            --bg-primary: #0f0f0f;
            --bg-secondary: #161616;
            --bg-tertiary: #1e1e1e;
            --bg-hover: #262626;
            --text-primary: #e8e8e8;
            --text-secondary: #8a8a8a;
            --text-muted: #5a5a5a;
            --accent: #b8b8b8;
            --accent-dim: rgba(184, 184, 184, 0.12);
            --border-color: #2a2a2a;
            --transition-base: 0.15s ease-out;
            --transition-smooth: 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            --radius-md: 4px;
            --radius-lg: 6px;
            --radius-xl: 8px;
            --shadow-md: 0 4px 12px rgba(0, 0, 0, 0.5), 0 2px 4px rgba(0, 0, 0, 0.3);
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        html, body {
            width: 100%; height: 100%;
            overflow: hidden;
            background: var(--bg-void);
            font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }
        #stream {
            position: fixed;
            inset: 0;
            width: 100%; height: 100%;
            object-fit: contain;
        }
        .loading {
            position: fixed;
            inset: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            background: var(--bg-void);
            z-index: 10;
            transition: opacity var(--transition-smooth);
        }
        .loading.hidden { opacity: 0; pointer-events: none; }
        .spinner {
            width: 20px; height: 20px;
            border: 1.5px solid var(--border-color);
            border-top-color: var(--accent);
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        .controls {
            position: fixed;
            bottom: 0; left: 0; right: 0;
            display: flex;
            justify-content: center;
            padding: 20px;
            background: linear-gradient(transparent, rgba(15,15,15,0.8));
            opacity: 0;
            transition: opacity var(--transition-smooth);
            z-index: 20;
        }
        .controls:hover, .controls.visible { opacity: 1; }
        .bar {
            display: flex;
            align-items: center;
            gap: 2px;
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border-radius: var(--radius-xl);
            padding: 4px;
            box-shadow: var(--shadow-md);
        }
        .bar button {
            display: flex;
            align-items: center;
            justify-content: center;
            width: 44px; height: 44px;
            background: transparent;
            border: none;
            border-radius: var(--radius-lg);
            color: var(--text-secondary);
            cursor: pointer;
            transition: all var(--transition-base);
            -webkit-tap-highlight-color: transparent;
        }
        .bar button:hover {
            background: var(--bg-hover);
            color: var(--text-primary);
        }
        .bar button:active { transform: scale(0.92); background: var(--accent-dim); }
        .bar button.on {
            background: var(--accent-dim);
            color: var(--accent);
        }
        .bar button svg { width: 22px; height: 22px; }
        .sep { width: 1px; height: 24px; background: var(--border-color); margin: 0 6px; }
        @media (hover: none) {
            .controls { opacity: 1; background: none; padding: 16px; }
            .bar { background: var(--bg-primary); border-color: var(--border-color); padding: 6px; }
            .bar button { width: 48px; height: 48px; }
            .bar button svg { width: 24px; height: 24px; }
        }
    </style>
</head>
<body>
    <div class="loading" id="loading"><div class="spinner"></div></div>
    <img id="stream" alt="">
    <div class="controls" id="controls">
        <div class="bar">
            <button id="ccwBtn" title="Rotate left"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M1 4v6h6"/><path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"/></svg></button>
            <button id="pauseRotBtn" title="Stop rotation"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><path d="M10 8v8M14 8v8"/></svg></button>
            <button id="cwBtn" title="Rotate right"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M23 4v6h-6"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/></svg></button>
            <div class="sep"></div>
            <button id="playBtn" title="Play/Pause"><svg id="playIcon" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg><svg id="pauseIcon" viewBox="0 0 24 24" fill="currentColor" style="display:none"><path d="M6 4h4v16H6zM14 4h4v16h-4z"/></svg></button>
            <div class="sep"></div>
            <button id="fsBtn" title="Fullscreen"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M8 3H5a2 2 0 00-2 2v3m18 0V5a2 2 0 00-2-2h-3m0 18h3a2 2 0 002-2v-3M3 16v3a2 2 0 002 2h3"/></svg></button>
        </div>
    </div>
<script>
const img = document.getElementById('stream');
const loading = document.getElementById('loading');
const controls = document.getElementById('controls');
const playBtn = document.getElementById('playBtn');
const playIcon = document.getElementById('playIcon');
const pauseIcon = document.getElementById('pauseIcon');
const ccwBtn = document.getElementById('ccwBtn');
const pauseRotBtn = document.getElementById('pauseRotBtn');
const cwBtn = document.getElementById('cwBtn');
const fsBtn = document.getElementById('fsBtn');

let ws, blobUrl, reconnect, hideTimer;
let isPlaying = false, rotDir = 'stopped';

function showControls() {
    controls.classList.add('visible');
    clearTimeout(hideTimer);
    hideTimer = setTimeout(() => controls.classList.remove('visible'), 3000);
}
document.addEventListener('mousemove', showControls);
document.addEventListener('touchstart', showControls);

function connect() {
    if (ws) ws.close();
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const m = location.pathname.match(/^(\/s\/[^/]+)/);
    ws = new WebSocket(proto + '//' + location.host + (m ? m[1] + '/ws' : '/'));
    ws.binaryType = 'arraybuffer';
    ws.onopen = () => loading.classList.add('hidden');
    ws.onmessage = e => {
        if (typeof e.data === 'string') return;
        const url = URL.createObjectURL(new Blob([e.data], {type: 'image/jpeg'}));
        if (blobUrl) URL.revokeObjectURL(blobUrl);
        blobUrl = url;
        img.src = url;
    };
    ws.onclose = () => {
        loading.classList.remove('hidden');
        clearTimeout(reconnect);
        reconnect = setTimeout(connect, 2000);
    };
}

function api(endpoint) {
    const m = location.pathname.match(/^\/s\/([^/]+)/);
    if (m) return location.origin + '/c/' + m[1] + '/' + endpoint;
    const p = parseInt(location.port);
    return isNaN(p) ? null : location.protocol + '//' + location.hostname + ':' + (p+1) + '/' + endpoint;
}

function updatePlayUI() {
    playIcon.style.display = isPlaying ? 'none' : 'block';
    pauseIcon.style.display = isPlaying ? 'block' : 'none';
    playBtn.classList.toggle('on', isPlaying);
}
function updateRotUI() {
    ccwBtn.classList.toggle('on', rotDir === 'ccw');
    pauseRotBtn.classList.toggle('on', rotDir === 'stopped');
    cwBtn.classList.toggle('on', rotDir === 'cw');
}

// Optimistic updates with poll suppression
let lastAction = 0;
function act(endpoint, update) {
    lastAction = Date.now();
    update();  // Optimistic UI update
    const u = api(endpoint);
    if (u) fetch(u, {method:'POST'}).catch(()=>{});
}

playBtn.onclick = () => {
    isPlaying = !isPlaying; updatePlayUI();
    lastAction = Date.now();
    const u = api('toggle-playback');
    if (u) fetch(u, {method:'POST'}).then(r=>r.json()).then(d => {
        if (d.ok) { isPlaying = d.playing; updatePlayUI(); }
    }).catch(()=>{ isPlaying = !isPlaying; updatePlayUI(); });
};
ccwBtn.onclick = () => act('rotate-ccw', () => { rotDir = 'ccw'; updateRotUI(); });
cwBtn.onclick = () => act('rotate-cw', () => { rotDir = 'cw'; updateRotUI(); });
pauseRotBtn.onclick = () => act('rotate-stop', () => { rotDir = 'stopped'; updateRotUI(); });
fsBtn.onclick = () => {
    if (!document.fullscreenElement) document.documentElement.requestFullscreen().catch(()=>{});
    else document.exitFullscreen().catch(()=>{});
};

function poll() {
    if (Date.now() - lastAction < 1500) return;  // Skip poll after recent action
    const u1 = api('playback-state'), u2 = api('rotation-state');
    if (u1) fetch(u1).then(r=>r.json()).then(d => {
        if (d.ok && d.playing !== isPlaying) { isPlaying = d.playing; updatePlayUI(); }
    }).catch(()=>{});
    if (u2) fetch(u2).then(r=>r.json()).then(d => {
        if (d.ok && d.direction !== rotDir) { rotDir = d.direction; updateRotUI(); }
    }).catch(()=>{});
}
setTimeout(poll, 500);
setInterval(poll, 1000);

document.addEventListener('keydown', e => {
    if (e.key === 'f' || e.key === 'F') fsBtn.click();
    if (e.key === ' ') { e.preventDefault(); playBtn.click(); }
    if (e.key === 'ArrowLeft') ccwBtn.click();
    if (e.key === 'ArrowRight') cwBtn.click();
    if (e.key === 'Escape') pauseRotBtn.click();
});

// Touch-based rotation (swipe to spin)
let touchStartX = 0, touchActive = false, touchDir = null;
function startRotation(dir) {
    if (touchDir === dir) return;
    touchDir = dir;
    rotDir = dir; updateRotUI();  // Immediate UI feedback
    const u = api('rotate-' + dir);
    if (u) fetch(u, {method:'POST'}).catch(()=>{
        // Revert on failure
        if (touchDir === dir) { touchDir = null; rotDir = 'stopped'; updateRotUI(); }
    });
}
function stopRotation() {
    if (!touchDir) return;
    touchDir = null;
    rotDir = 'stopped'; updateRotUI();  // Immediate UI feedback
    const u = api('rotate-stop');
    if (u) fetch(u, {method:'POST'}).catch(()=>{});
}
img.addEventListener('touchstart', e => {
    if (e.touches.length === 1) {
        touchStartX = e.touches[0].clientX;
        touchActive = true;
        img.style.opacity = '0.9';
    }
}, {passive: true});
img.addEventListener('touchmove', e => {
    if (!touchActive || e.touches.length !== 1) return;
    const dx = e.touches[0].clientX - touchStartX;
    if (Math.abs(dx) > 40) {
        startRotation(dx > 0 ? 'cw' : 'ccw');
    } else if (Math.abs(dx) < 20 && touchDir) {
        stopRotation();
    }
}, {passive: true});
img.addEventListener('touchend', () => {
    touchActive = false;
    img.style.opacity = '1';
    stopRotation();
}, {passive: true});
img.addEventListener('touchcancel', () => {
    touchActive = false;
    img.style.opacity = '1';
    stopRotation();
}, {passive: true});

connect();
showControls();
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
