"""Instance management routes (CRUD operations)."""

from fastapi import APIRouter, Depends, HTTPException, status

from gsplay_launcher.api.dependencies import (
    get_external_url,
    get_instance_manager,
    get_network_url,
)
from gsplay_launcher.api.schemas import (
    CreateInstanceRequest,
    ErrorResponse,
    InstanceListResponse,
    InstanceResponse,
    PortInfoResponse,
)
from gsplay_launcher.services.instance_manager import (
    ConfigPathError,
    InstanceManager,
    InstanceNotFoundError,
    PortInUseError,
)
from gsplay_launcher.services.process_manager import ProcessStartError


router = APIRouter(tags=["instances"])


@router.get("/instances", response_model=InstanceListResponse, summary="List all instances")
def list_instances(
    manager: InstanceManager = Depends(get_instance_manager),
    external_url: str | None = Depends(get_external_url),
    network_url: str | None = Depends(get_network_url),
) -> InstanceListResponse:
    """List all GSPlay instances."""
    instances = manager.list_all()
    return InstanceListResponse(
        instances=[InstanceResponse.from_instance(i, external_url, network_url) for i in instances],
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
    external_url: str | None = Depends(get_external_url),
    network_url: str | None = Depends(get_network_url),
) -> InstanceResponse:
    """Get a specific instance by ID."""
    try:
        instance = manager.get(instance_id)
        return InstanceResponse.from_instance(instance, external_url, network_url)
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
    external_url: str | None = Depends(get_external_url),
    network_url: str | None = Depends(get_network_url),
) -> InstanceResponse:
    """Create and start a new gsplay instance."""
    try:
        instance = manager.create_and_start(
            config_path=body.config_path,
            name=body.name,
            port=body.port,
            host=body.host,
            stream_port=body.stream_port,
            gpu=body.gpu,
            view_only=body.view_only,
            compact=body.compact,
            log_level=body.log_level,
            custom_ip=body.custom_ip,
            viewer_id=body.viewer_id,
            stream_token=body.stream_token,
        )
        return InstanceResponse.from_instance(instance, external_url, network_url)
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
    external_url: str | None = Depends(get_external_url),
    network_url: str | None = Depends(get_network_url),
) -> InstanceResponse:
    """Stop a running instance."""
    try:
        instance = manager.stop(instance_id)
        return InstanceResponse.from_instance(instance, external_url, network_url)
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


@router.get("/ports/next", response_model=PortInfoResponse, summary="Get next available port")
def get_next_port(manager: InstanceManager = Depends(get_instance_manager)) -> PortInfoResponse:
    """Get information about the next available port."""
    next_port = manager.get_next_available_port()
    return PortInfoResponse(
        next_available=next_port,
        range_start=manager.config.gsplay_port_start,
        range_end=manager.config.gsplay_port_end,
    )
