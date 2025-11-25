"""FastAPI routes for instance management."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status

import subprocess

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
from gsplay_launcher.services.file_browser import (
    FileBrowserService,
    PathNotFoundError,
    PathSecurityError,
)
from gsplay_launcher.services.instance_manager import (
    ConfigPathError,
    InstanceManager,
    InstanceNotFoundError,
    PortInUseError,
)
from gsplay_launcher.services.process_manager import ProcessStartError

logger = logging.getLogger(__name__)

# Router for instance endpoints
router = APIRouter(prefix="/api", tags=["instances"])

# Global instance manager (set by app.py)
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


def _parse_nvidia_smi() -> GpuInfoResponse | None:
    """Parse nvidia-smi output for GPU information."""
    try:
        # Query GPU info with specific format
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None

        gpus = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 6:
                gpus.append(
                    GpuInfo(
                        index=int(parts[0]),
                        name=parts[1],
                        memory_used=int(parts[2]),
                        memory_total=int(parts[3]),
                        utilization=int(parts[4]),
                        temperature=int(parts[5]),
                    )
                )

        # Get driver and CUDA version
        driver_result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        driver_version = driver_result.stdout.strip().split("\n")[0] if driver_result.returncode == 0 else "unknown"

        # Parse CUDA version from nvidia-smi output header
        header_result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        cuda_version = "unknown"
        if header_result.returncode == 0:
            for line in header_result.stdout.split("\n"):
                if "CUDA Version:" in line:
                    cuda_version = line.split("CUDA Version:")[1].strip().split()[0]
                    break

        return GpuInfoResponse(
            gpus=gpus,
            driver_version=driver_version,
            cuda_version=cuda_version,
        )
    except Exception as e:
        logger.warning("Failed to get GPU info: %s", e)
        return None


@router.get(
    "/gpu",
    response_model=GpuInfoResponse,
    responses={503: {"model": ErrorResponse}},
    summary="Get GPU information",
)
def get_gpu_info() -> GpuInfoResponse:
    """Get GPU information from nvidia-smi."""
    info = _parse_nvidia_smi()
    if info is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "nvidia-smi not available or failed",
        )
    return info


# Browse endpoints


@router.get(
    "/browse/config",
    response_model=BrowseConfigResponse,
    summary="Get browse configuration",
)
def get_browse_config() -> BrowseConfigResponse:
    """Check if file browser is enabled and get root path."""
    browser = get_file_browser()
    if browser is None:
        return BrowseConfigResponse(enabled=False)
    return BrowseConfigResponse(
        enabled=True,
        root_path=str(browser.root),
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
