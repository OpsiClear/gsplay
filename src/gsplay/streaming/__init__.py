"""Streaming module for view-only rendering output.

Uses WebSocket for low-latency (~100-150ms) push-based JPEG streaming.
"""

from .websocket_server import (
    FrameBuffer,
    WebSocketStreamServer,
    get_websocket_server,
    set_websocket_server,
)


# Aliases for backward compatibility
StreamServer = WebSocketStreamServer
get_stream_server = get_websocket_server
set_stream_server = set_websocket_server

__all__ = [
    "FrameBuffer",
    "WebSocketStreamServer",
    "get_websocket_server",
    "set_websocket_server",
    # Aliases
    "StreamServer",
    "get_stream_server",
    "set_stream_server",
]
