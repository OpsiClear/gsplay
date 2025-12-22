"""WebSocket proxy service for routing to GSPlay instances.

This module provides a WebSocket reverse proxy that allows external access
to GSPlay instances through a single entry point (the launcher).

Usage:
    External URL: https://gsplay.4dgst.win/v/{instance_id}/
    Routes to: ws://localhost:{instance_port}/
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import websockets
import websockets.exceptions
from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState


if TYPE_CHECKING:
    from websockets.asyncio.client import ClientConnection

    from gsplay_launcher.models import GSPlayInstance

logger = logging.getLogger(__name__)


class WebSocketProxy:
    """Bidirectional WebSocket proxy for GSPlay instances.

    Proxies WebSocket connections from external clients to internal
    GSPlay viser instances.
    """

    def __init__(self, timeout: float = 30.0):
        """Initialize proxy.

        Parameters
        ----------
        timeout : float
            Connection timeout in seconds.
        """
        self.timeout = timeout
        self._active_connections: dict[str, int] = {}  # instance_id -> count

    async def proxy(
        self,
        client_ws: WebSocket,
        instance: GSPlayInstance,
        port_override: int | None = None,
    ) -> None:
        """Proxy WebSocket connection to a GSPlay instance.

        Parameters
        ----------
        client_ws : WebSocket
            The incoming client WebSocket connection.
        instance : GSPlayInstance
            The target GSPlay instance.
        port_override : int | None
            Optional port override. If None, uses instance.port.
        """
        # Determine backend URL
        backend_host = "127.0.0.1" if instance.host == "0.0.0.0" else instance.host
        port = port_override if port_override is not None else instance.port
        backend_url = f"ws://{backend_host}:{port}/"

        # Get subprotocol from client connection
        subprotocols = client_ws.scope.get("subprotocols", [])

        logger.info(
            "Proxying WebSocket for instance %s to %s (subprotocols: %s)",
            instance.id,
            backend_url,
            subprotocols,
        )

        # Track connection
        self._active_connections[instance.id] = self._active_connections.get(instance.id, 0) + 1

        try:
            # Set Origin and Host headers to match what viser expects (localhost)
            # This bypasses viser's origin checking for proxied connections
            extra_headers = {
                "Origin": f"http://{backend_host}:{port}",
                "Host": f"{backend_host}:{port}",
            }

            async with websockets.connect(
                backend_url,
                subprotocols=subprotocols if subprotocols else None,
                additional_headers=extra_headers,
                close_timeout=5,
                ping_interval=20,
                ping_timeout=20,
            ) as backend_ws:
                # Create bidirectional proxy tasks
                await asyncio.gather(
                    self._forward_client_to_backend(client_ws, backend_ws),
                    self._forward_backend_to_client(backend_ws, client_ws),
                )

        except websockets.exceptions.ConnectionClosed as e:
            logger.debug("Backend WebSocket closed: %s", e)
        except Exception as e:
            logger.error("WebSocket proxy error: %s", e)
        finally:
            # Decrement connection count
            self._active_connections[instance.id] = max(
                0, self._active_connections.get(instance.id, 1) - 1
            )

            # Ensure client connection is closed
            if client_ws.client_state != WebSocketState.DISCONNECTED:
                try:
                    await client_ws.close()
                except Exception:
                    pass

    async def _forward_client_to_backend(
        self,
        client_ws: WebSocket,
        backend_ws: ClientConnection,
    ) -> None:
        """Forward messages from client to backend."""
        try:
            while True:
                # Receive from client (handles both text and binary)
                message = await client_ws.receive()

                if message["type"] == "websocket.receive":
                    if "text" in message:
                        await backend_ws.send(message["text"])
                    elif "bytes" in message:
                        await backend_ws.send(message["bytes"])
                elif message["type"] == "websocket.disconnect":
                    break

        except WebSocketDisconnect:
            logger.debug("Client disconnected")
        except Exception as e:
            logger.debug("Client->Backend forward error: %s", e)

    async def _forward_backend_to_client(
        self,
        backend_ws: ClientConnection,
        client_ws: WebSocket,
    ) -> None:
        """Forward messages from backend to client."""
        try:
            async for message in backend_ws:
                if client_ws.client_state == WebSocketState.DISCONNECTED:
                    break

                if isinstance(message, str):
                    await client_ws.send_text(message)
                elif isinstance(message, bytes):
                    await client_ws.send_bytes(message)

        except Exception as e:
            logger.debug("Backend->Client forward error: %s", e)

    def get_connection_count(self, instance_id: str) -> int:
        """Get active connection count for an instance.

        Parameters
        ----------
        instance_id : str
            Instance ID.

        Returns
        -------
        int
            Number of active proxy connections.
        """
        return self._active_connections.get(instance_id, 0)


# Global proxy instance
_proxy: WebSocketProxy | None = None


def get_proxy() -> WebSocketProxy:
    """Get or create the global WebSocket proxy."""
    global _proxy
    if _proxy is None:
        _proxy = WebSocketProxy()
    return _proxy
