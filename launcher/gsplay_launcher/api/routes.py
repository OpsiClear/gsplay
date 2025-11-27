"""FastAPI routes for instance management.

This module defines all API routes for the GSPlay Launcher:
- Instance management (CRUD operations)
- GPU information
- File browser
- WebSocket/HTTP proxy for accessing GSPlay instances
"""

from __future__ import annotations

import logging

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket, status
from fastapi.responses import Response

from gsplay_launcher.api.schemas import (
    BrowseConfigResponse,
    BrowseLaunchRequest,
    BrowseResponse,
    CreateInstanceRequest,
    DirectoryEntryResponse,
    ErrorResponse,
    GpuInfo,
    GpuInfoResponse,
    HealthResponse,
    InstanceListResponse,
    InstanceResponse,
    PortInfoResponse,
)
from gsplay_launcher.models import InstanceStatus
from gsplay_launcher.services.file_browser import (
    FileBrowserService,
    PathNotFoundError,
    PathSecurityError,
)
from gsplay_launcher.services.gpu_info import get_gpu_service
from gsplay_launcher.services.instance_manager import (
    ConfigPathError,
    InstanceManager,
    InstanceNotFoundError,
    PortInUseError,
)
from gsplay_launcher.services.process_manager import ProcessStartError

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

# Status values that indicate an instance can accept proxied connections
_PROXY_ALLOWED_STATUSES = frozenset({
    InstanceStatus.STARTING,
    InstanceStatus.RUNNING,
    InstanceStatus.ORPHANED,
})

# Main API router (prefixed with /api)
router = APIRouter(prefix="/api", tags=["instances"])

# =============================================================================
# Dependency Injection
# =============================================================================

_instance_manager: InstanceManager | None = None


def get_instance_manager() -> InstanceManager:
    """Get the instance manager dependency."""
    if _instance_manager is None:
        raise RuntimeError("Instance manager not initialized")
    return _instance_manager


def set_instance_manager(manager: InstanceManager) -> None:
    """Set the global instance manager."""
    global _instance_manager
    _instance_manager = manager


# Global file browser (set by app.py if browse_path is configured)
_file_browser: FileBrowserService | None = None


def get_file_browser() -> FileBrowserService | None:
    """Get the file browser service (may be None if disabled)."""
    return _file_browser


def set_file_browser(browser: FileBrowserService) -> None:
    """Set the global file browser service."""
    global _file_browser
    _file_browser = browser


# =============================================================================
# Instance Management Routes
# =============================================================================


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
)
def health_check(
    manager: InstanceManager = Depends(get_instance_manager),
) -> HealthResponse:
    """Check launcher health status."""
    instances = manager.list_all()
    active = [i for i in instances if i.is_active]
    return HealthResponse(
        status="healthy",
        instance_count=len(instances),
        active_count=len(active),
    )


@router.get(
    "/instances",
    response_model=InstanceListResponse,
    summary="List all instances",
)
def list_instances(
    manager: InstanceManager = Depends(get_instance_manager),
) -> InstanceListResponse:
    """List all GSPlay instances."""
    instances = manager.list_all()
    return InstanceListResponse(
        instances=[InstanceResponse.from_instance(i) for i in instances],
        total=len(instances),
    )


@router.get(
    "/instances/{instance_id}",
    response_model=InstanceResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Get instance by ID",
)
def get_instance(
    instance_id: str,
    manager: InstanceManager = Depends(get_instance_manager),
) -> InstanceResponse:
    """Get a specific instance by ID."""
    try:
        instance = manager.get(instance_id)
        return InstanceResponse.from_instance(instance)
    except InstanceNotFoundError as e:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(e))


@router.post(
    "/instances",
    response_model=InstanceResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="Create and start instance",
)
def create_instance(
    body: CreateInstanceRequest,
    manager: InstanceManager = Depends(get_instance_manager),
) -> InstanceResponse:
    """Create and start a new gsplay instance."""
    try:
        instance = manager.create_and_start(
            config_path=body.config_path,
            name=body.name,
            port=body.port,
            host=body.host,
            gpu=body.gpu,
            cache_size=body.cache_size,
            view_only=body.view_only,
            compact=body.compact,
            log_level=body.log_level,
            custom_ip=body.custom_ip,
        )
        return InstanceResponse.from_instance(instance)
    except ConfigPathError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))
    except PortInUseError as e:
        raise HTTPException(status.HTTP_409_CONFLICT, str(e))
    except ProcessStartError as e:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, str(e))


@router.post(
    "/instances/{instance_id}/stop",
    response_model=InstanceResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Stop running instance",
)
def stop_instance(
    instance_id: str,
    manager: InstanceManager = Depends(get_instance_manager),
) -> InstanceResponse:
    """Stop a running instance."""
    try:
        instance = manager.stop(instance_id)
        return InstanceResponse.from_instance(instance)
    except InstanceNotFoundError as e:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(e))


@router.delete(
    "/instances/{instance_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={404: {"model": ErrorResponse}},
    summary="Delete instance",
)
def delete_instance(
    instance_id: str,
    manager: InstanceManager = Depends(get_instance_manager),
) -> None:
    """Delete an instance (stops it first if running)."""
    try:
        manager.delete(instance_id)
    except InstanceNotFoundError as e:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(e))


@router.get(
    "/ports/next",
    response_model=PortInfoResponse,
    summary="Get next available port",
)
def get_next_port(
    manager: InstanceManager = Depends(get_instance_manager),
) -> PortInfoResponse:
    """Get information about the next available port."""
    next_port = manager.get_next_available_port()
    return PortInfoResponse(
        next_available=next_port,
        range_start=manager.config.gsplay_port_start,
        range_end=manager.config.gsplay_port_end,
    )


# =============================================================================
# GPU Information Routes
# =============================================================================


@router.get(
    "/gpu",
    response_model=GpuInfoResponse,
    responses={503: {"model": ErrorResponse}},
    summary="Get GPU information",
)
def get_gpu_info() -> GpuInfoResponse:
    """Get GPU information from nvidia-smi."""
    gpu_service = get_gpu_service()
    info = gpu_service.get_info()
    if info is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "nvidia-smi not available or failed",
        )
    return GpuInfoResponse(
        gpus=[
            GpuInfo(
                index=gpu.index,
                name=gpu.name,
                memory_used=gpu.memory_used,
                memory_total=gpu.memory_total,
                utilization=gpu.utilization,
                temperature=gpu.temperature,
            )
            for gpu in info.gpus
        ],
        driver_version=info.driver_version,
        cuda_version=info.cuda_version,
    )


# =============================================================================
# File Browser Routes
# =============================================================================


@router.get(
    "/browse/config",
    response_model=BrowseConfigResponse,
    summary="Get browse configuration",
)
def get_browse_config(
    manager: InstanceManager = Depends(get_instance_manager),
) -> BrowseConfigResponse:
    """Check if file browser is enabled and get root path."""
    browser = get_file_browser()
    if browser is None:
        return BrowseConfigResponse(
            enabled=False,
            external_url=manager.config.external_url,
        )
    return BrowseConfigResponse(
        enabled=True,
        root_path=str(browser.root),
        default_custom_ip=manager.config.custom_ip,
        external_url=manager.config.external_url,
    )


@router.get(
    "/browse",
    response_model=BrowseResponse,
    responses={
        400: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    summary="Browse directory",
)
def browse_directory(path: str = "") -> BrowseResponse:
    """Browse a directory under the configured root path."""
    browser = get_file_browser()
    if browser is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "File browser is not enabled",
        )

    try:
        result = browser.browse(path)
        return BrowseResponse(
            current_path=result.current_path,
            breadcrumbs=result.breadcrumbs,
            entries=[
                DirectoryEntryResponse(
                    name=e.name,
                    path=e.path,
                    is_directory=e.is_directory,
                    is_ply_folder=e.is_ply_folder,
                    ply_count=e.ply_count,
                    total_size_mb=e.total_size_mb,
                    modified_at=e.modified_at,
                )
                for e in result.entries
            ],
            browse_enabled=True,
        )
    except PathSecurityError as e:
        raise HTTPException(status.HTTP_403_FORBIDDEN, str(e))
    except PathNotFoundError as e:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(e))


@router.post(
    "/browse/launch",
    response_model=InstanceResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    summary="Launch instance from browser path",
)
def launch_from_browser(
    body: BrowseLaunchRequest,
    manager: InstanceManager = Depends(get_instance_manager),
) -> InstanceResponse:
    """Create and start a GSPlay instance from a browser path."""
    browser = get_file_browser()
    if browser is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "File browser is not enabled",
        )

    try:
        # Get absolute path from browser
        absolute_path = browser.get_absolute_path(body.path)

        # Create instance using the instance manager
        instance = manager.create_and_start(
            config_path=str(absolute_path),
            name=body.name,
            port=body.port,
            gpu=body.gpu,
            cache_size=body.cache_size,
            view_only=body.view_only,
            compact=body.compact,
            custom_ip=body.custom_ip,
        )
        return InstanceResponse.from_instance(instance)
    except PathSecurityError as e:
        raise HTTPException(status.HTTP_403_FORBIDDEN, str(e))
    except PathNotFoundError as e:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(e))
    except ConfigPathError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))
    except PortInUseError as e:
        raise HTTPException(status.HTTP_409_CONFLICT, str(e))
    except ProcessStartError as e:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, str(e))


# =============================================================================
# WebSocket Proxy Routes - Access GSPlay instances via /v/{instance_id}/
# =============================================================================

# Separate router for proxy (no /api prefix)
proxy_router = APIRouter(tags=["proxy"])


async def _proxy_websocket_impl(
    websocket: WebSocket,
    instance_id: str,
    manager: InstanceManager,
) -> None:
    """Internal implementation for WebSocket proxy."""
    from gsplay_launcher.services.websocket_proxy import get_proxy

    try:
        instance = manager.get(instance_id)
    except InstanceNotFoundError:
        await websocket.close(code=4004, reason="Instance not found")
        return

    # Check if instance can accept connections
    if instance.status not in _PROXY_ALLOWED_STATUSES:
        await websocket.close(code=4003, reason="Instance not running")
        return

    # Accept the WebSocket connection with the requested subprotocol (viser requires this)
    requested_subprotocol = None
    if websocket.scope.get("subprotocols"):
        requested_subprotocol = websocket.scope["subprotocols"][0]

    await websocket.accept(subprotocol=requested_subprotocol)

    # Proxy to backend
    proxy = get_proxy()
    await proxy.proxy(websocket, instance)


@proxy_router.websocket("/v/{instance_id}/")
async def proxy_websocket_with_slash(
    websocket: WebSocket,
    instance_id: str,
    manager: InstanceManager = Depends(get_instance_manager),
) -> None:
    """Proxy WebSocket connection to a GSPlay instance (with trailing slash)."""
    await _proxy_websocket_impl(websocket, instance_id, manager)


@proxy_router.websocket("/v/{instance_id}")
async def proxy_websocket_no_slash(
    websocket: WebSocket,
    instance_id: str,
    manager: InstanceManager = Depends(get_instance_manager),
) -> None:
    """Proxy WebSocket connection to a GSPlay instance (without trailing slash)."""
    await _proxy_websocket_impl(websocket, instance_id, manager)


# Script to inject into HTML to prevent viser from adding ?websocket= to URL
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


async def _proxy_http_impl(
    request: Request,
    instance_id: str,
    path: str,
    manager: InstanceManager,
) -> Response:
    """Internal implementation for HTTP proxy."""
    try:
        instance = manager.get(instance_id)
    except InstanceNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Instance not found")

    # Check if instance can accept connections
    if instance.status not in _PROXY_ALLOWED_STATUSES:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Instance not running")

    # Build backend URL
    backend_host = "127.0.0.1" if instance.host == "0.0.0.0" else instance.host
    backend_url = f"http://{backend_host}:{instance.port}/{path}"

    # Forward query string
    if request.url.query:
        backend_url += f"?{request.url.query}"

    # Proxy the request
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            # Forward request
            response = await client.request(
                method=request.method,
                url=backend_url,
                headers={
                    k: v for k, v in request.headers.items()
                    if k.lower() not in ("host", "content-length")
                },
                content=await request.body() if request.method in ("POST", "PUT", "PATCH") else None,
            )

            content = response.content
            content_type = response.headers.get("content-type", "")

            # Inject URL cleanup script into HTML responses (root page only)
            if path == "" and "text/html" in content_type:
                # Insert script right after <head>
                content = content.replace(b"<head>", b"<head>" + _URL_CLEANUP_SCRIPT, 1)

            # Return proxied response
            return Response(
                content=content,
                status_code=response.status_code,
                headers={
                    k: v for k, v in response.headers.items()
                    if k.lower() not in ("content-encoding", "transfer-encoding", "content-length")
                },
            )

        except httpx.ConnectError:
            raise HTTPException(
                status.HTTP_502_BAD_GATEWAY,
                f"Cannot connect to instance {instance_id}",
            )
        except httpx.TimeoutException:
            raise HTTPException(
                status.HTTP_504_GATEWAY_TIMEOUT,
                f"Instance {instance_id} timed out",
            )


@proxy_router.get("/v/{instance_id}/")
async def proxy_http_root(
    request: Request,
    instance_id: str,
    manager: InstanceManager = Depends(get_instance_manager),
) -> Response:
    """Proxy HTTP requests to a GSPlay instance root."""
    return await _proxy_http_impl(request, instance_id, "", manager)


@proxy_router.get("/v/{instance_id}/{path:path}")
async def proxy_http_path(
    request: Request,
    instance_id: str,
    path: str,
    manager: InstanceManager = Depends(get_instance_manager),
) -> Response:
    """Proxy HTTP requests to a GSPlay instance with path."""
    return await _proxy_http_impl(request, instance_id, path, manager)
