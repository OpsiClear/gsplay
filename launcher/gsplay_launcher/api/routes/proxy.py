"""WebSocket and HTTP proxy routes for GSPlay instances."""

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket, status
from fastapi.responses import Response

from gsplay_launcher.api.dependencies import get_instance_manager
from gsplay_launcher.models import InstanceStatus
from gsplay_launcher.services.id_encoder import decode_instance_id
from gsplay_launcher.services.instance_manager import InstanceManager, InstanceNotFoundError

router = APIRouter(tags=["proxy"])

_PROXY_ALLOWED_STATUSES = frozenset({InstanceStatus.STARTING, InstanceStatus.RUNNING, InstanceStatus.ORPHANED})

_URL_CLEANUP_SCRIPT = b"""<script>
(function(){
  const orig = history.replaceState.bind(history);
  history.replaceState = function(state, title, url) {
    if (url && typeof url === 'string' && url.includes('websocket=')) {
      url = url.split('?')[0];
    }
    return orig(state, title, url);
  };
})();
</script>"""


# =============================================================================
# Viewer Proxy (WebSocket + HTTP) - /v/{instance_id}/
# =============================================================================

async def _proxy_websocket_impl(websocket: WebSocket, instance_id: str, manager: InstanceManager) -> None:
    """Internal implementation for WebSocket proxy."""
    from gsplay_launcher.services.websocket_proxy import get_proxy

    try:
        instance = manager.get_by_viewer_id(instance_id)
    except InstanceNotFoundError:
        await websocket.close(code=4004, reason="Instance not found")
        return

    if instance.status not in _PROXY_ALLOWED_STATUSES:
        await websocket.close(code=4003, reason="Instance not running")
        return

    requested_subprotocol = websocket.scope.get("subprotocols", [None])[0] if websocket.scope.get("subprotocols") else None
    await websocket.accept(subprotocol=requested_subprotocol)
    await get_proxy().proxy(websocket, instance)


@router.websocket("/v/{instance_id}/")
async def proxy_websocket_with_slash(websocket: WebSocket, instance_id: str, manager: InstanceManager = Depends(get_instance_manager)) -> None:
    """Proxy WebSocket connection to a GSPlay instance (with trailing slash)."""
    await _proxy_websocket_impl(websocket, instance_id, manager)


@router.websocket("/v/{instance_id}")
async def proxy_websocket_no_slash(websocket: WebSocket, instance_id: str, manager: InstanceManager = Depends(get_instance_manager)) -> None:
    """Proxy WebSocket connection to a GSPlay instance (without trailing slash)."""
    await _proxy_websocket_impl(websocket, instance_id, manager)


async def _proxy_http_impl(request: Request, instance_id: str, path: str, manager: InstanceManager) -> Response:
    """Internal implementation for HTTP proxy."""
    try:
        instance = manager.get_by_viewer_id(instance_id)
    except InstanceNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Instance not found")

    if instance.status not in _PROXY_ALLOWED_STATUSES:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Instance not running")

    backend_host = "127.0.0.1" if instance.host == "0.0.0.0" else instance.host
    backend_url = f"http://{backend_host}:{instance.port}/{path}"
    if request.url.query:
        backend_url += f"?{request.url.query}"

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.request(
                method=request.method, url=backend_url,
                headers={k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")},
                content=await request.body() if request.method in ("POST", "PUT", "PATCH") else None,
            )
            content = response.content
            if path == "" and "text/html" in response.headers.get("content-type", ""):
                content = content.replace(b"<head>", b"<head>" + _URL_CLEANUP_SCRIPT, 1)
            return Response(
                content=content, status_code=response.status_code,
                headers={k: v for k, v in response.headers.items() if k.lower() not in ("content-encoding", "transfer-encoding", "content-length")},
            )
        except httpx.ConnectError:
            raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Cannot connect to instance {instance_id}")
        except httpx.TimeoutException:
            raise HTTPException(status.HTTP_504_GATEWAY_TIMEOUT, f"Instance {instance_id} timed out")


@router.get("/v/{instance_id}/")
async def proxy_http_root(request: Request, instance_id: str, manager: InstanceManager = Depends(get_instance_manager)) -> Response:
    """Proxy HTTP requests to a GSPlay instance root."""
    return await _proxy_http_impl(request, instance_id, "", manager)


@router.get("/v/{instance_id}/{path:path}")
async def proxy_http_path(request: Request, instance_id: str, path: str, manager: InstanceManager = Depends(get_instance_manager)) -> Response:
    """Proxy HTTP requests to a GSPlay instance with path."""
    return await _proxy_http_impl(request, instance_id, path, manager)


# =============================================================================
# Stream Proxy (WebSocket + HTTP) - /s/{token}/
# =============================================================================

def _resolve_stream_instance(token: str, manager: InstanceManager):
    """Resolve instance from stream token or encoded ID."""
    instance = manager.get_by_stream_token(token)
    if instance is not None:
        return instance

    instance_id = decode_instance_id(token)
    if instance_id is None:
        return None

    try:
        return manager.get(instance_id)
    except InstanceNotFoundError:
        return None


async def _proxy_stream_impl(request: Request, token: str, path: str, manager: InstanceManager) -> Response:
    """Internal implementation for stream HTTP proxy."""
    instance = _resolve_stream_instance(token, manager)
    if instance is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Invalid stream token")

    if instance.stream_port == 0:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Streaming not enabled for this instance")

    if instance.status not in _PROXY_ALLOWED_STATUSES:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Instance not running")

    actual_stream_port = instance.port + 1
    backend_host = "127.0.0.1" if instance.host == "0.0.0.0" else instance.host
    backend_url = f"http://{backend_host}:{actual_stream_port}/{path}"
    if request.url.query:
        backend_url += f"?{request.url.query}"

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.request(
                method=request.method, url=backend_url,
                headers={k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")},
            )
            content = response.content
            if path in ("", "view") and b"text/html" in response.headers.get("content-type", "").encode():
                base_path = f"/s/{token}"
                content = content.replace(
                    b"const wsUrl = protocol + '//' + location.host + '/';",
                    f"const wsUrl = protocol + '//' + location.host + '{base_path}/ws';".encode()
                )
            return Response(
                content=content, status_code=response.status_code,
                headers={k: v for k, v in response.headers.items() if k.lower() not in ("content-encoding", "transfer-encoding", "content-length")},
            )
        except httpx.ConnectError:
            raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Cannot connect to stream server at port {actual_stream_port}")
        except httpx.TimeoutException:
            raise HTTPException(status.HTTP_504_GATEWAY_TIMEOUT, "Stream timed out")


async def _proxy_stream_websocket_impl(websocket: WebSocket, token: str, manager: InstanceManager) -> None:
    """Internal implementation for stream WebSocket proxy."""
    from gsplay_launcher.services.websocket_proxy import get_proxy

    instance = _resolve_stream_instance(token, manager)
    if instance is None:
        await websocket.close(code=4004, reason="Invalid stream token")
        return

    if instance.stream_port == 0:
        await websocket.close(code=4003, reason="Streaming not enabled")
        return

    if instance.status not in _PROXY_ALLOWED_STATUSES:
        await websocket.close(code=4003, reason="Instance not running")
        return

    await websocket.accept()
    await get_proxy().proxy(websocket, instance, port_override=instance.port + 1)


@router.get("/s/{token}/")
async def proxy_stream_root(request: Request, token: str, manager: InstanceManager = Depends(get_instance_manager)) -> Response:
    """Proxy HTTP requests to stream server (view page)."""
    return await _proxy_stream_impl(request, token, "", manager)


@router.get("/s/{token}/view")
async def proxy_stream_view(request: Request, token: str, manager: InstanceManager = Depends(get_instance_manager)) -> Response:
    """Proxy view-only HTML page."""
    return await _proxy_stream_impl(request, token, "", manager)


@router.websocket("/s/{token}/ws")
async def proxy_stream_websocket(websocket: WebSocket, token: str, manager: InstanceManager = Depends(get_instance_manager)) -> None:
    """Proxy WebSocket stream connection."""
    await _proxy_stream_websocket_impl(websocket, token, manager)


@router.get("/s/{token}/status")
async def proxy_stream_status(request: Request, token: str, manager: InstanceManager = Depends(get_instance_manager)) -> Response:
    """Proxy stream status endpoint for debugging."""
    return await _proxy_stream_impl(request, token, "status", manager)
