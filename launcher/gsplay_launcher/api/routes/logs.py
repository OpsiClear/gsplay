"""Log streaming routes."""

from fastapi import APIRouter, Depends, HTTPException, status
from sse_starlette.sse import EventSourceResponse

from gsplay_launcher.api.dependencies import get_instance_manager
from gsplay_launcher.api.schemas import ErrorResponse, LogResponse
from gsplay_launcher.services.instance_manager import InstanceManager, InstanceNotFoundError
from gsplay_launcher.services.log_service import get_log_service


router = APIRouter(tags=["logs"])


@router.get(
    "/instances/{instance_id}/logs",
    response_model=LogResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Get instance logs",
)
def get_instance_logs(
    instance_id: str,
    lines: int = 200,
    offset: int = 0,
    manager: InstanceManager = Depends(get_instance_manager),
) -> LogResponse:
    """Get log lines from an instance."""
    try:
        instance = manager.get(instance_id)
    except InstanceNotFoundError as e:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(e))

    log_service = get_log_service()
    chunk = log_service.read_logs(instance.port, lines=lines, offset=offset)
    return LogResponse(
        lines=chunk.lines,
        total_lines=chunk.total_lines,
        offset=chunk.offset,
        has_more=chunk.has_more,
    )


@router.get("/instances/{instance_id}/logs/stream", summary="Stream instance logs (SSE)")
async def stream_instance_logs(
    instance_id: str,
    manager: InstanceManager = Depends(get_instance_manager),
):
    """Stream log lines from an instance via Server-Sent Events."""
    try:
        instance = manager.get(instance_id)
    except InstanceNotFoundError as e:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(e))

    log_service = get_log_service()

    async def event_generator():
        async for line in log_service.stream_logs(instance.port):
            yield {"event": "log", "data": line}

    return EventSourceResponse(event_generator())
