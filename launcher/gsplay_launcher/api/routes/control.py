"""Instance control routes."""

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import Response

from gsplay_launcher.api.dependencies import get_instance_manager
from gsplay_launcher.api.schemas import ErrorResponse
from gsplay_launcher.models import InstanceStatus
from gsplay_launcher.services.instance_manager import InstanceManager, InstanceNotFoundError


router = APIRouter(tags=["control"])

_PROXY_ALLOWED_STATUSES = frozenset(
    {InstanceStatus.STARTING, InstanceStatus.RUNNING, InstanceStatus.ORPHANED}
)


@router.post(
    "/instances/{instance_id}/control/{command}",
    responses={
        404: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    summary="Send control command to instance",
)
async def control_instance(
    request: Request,
    instance_id: str,
    command: str,
    manager: InstanceManager = Depends(get_instance_manager),
) -> Response:
    """Proxy control command to a running GSPlay instance. Control API runs on viser_port + 2."""
    try:
        instance = manager.get(instance_id)
    except InstanceNotFoundError as e:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(e))

    if instance.status not in _PROXY_ALLOWED_STATUSES:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Instance not running")

    control_port = instance.port + 2
    backend_host = "127.0.0.1" if instance.host == "0.0.0.0" else instance.host
    backend_url = f"http://{backend_host}:{control_port}/{command}"
    body = await request.body()

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.post(
                backend_url, content=body, headers={"Content-Type": "application/json"}
            )
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers={"Content-Type": "application/json"},
            )
        except httpx.ConnectError:
            raise HTTPException(
                status.HTTP_502_BAD_GATEWAY,
                f"Cannot connect to control server on port {control_port}",
            )
        except httpx.TimeoutException:
            raise HTTPException(status.HTTP_504_GATEWAY_TIMEOUT, "Control command timed out")
