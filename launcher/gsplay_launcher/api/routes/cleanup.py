"""Process cleanup routes."""

from fastapi import APIRouter

from gsplay_launcher.api.schemas import CleanupResponse, CleanupStopRequest, ProcessInfo


router = APIRouter(tags=["cleanup"])


@router.get("/cleanup", response_model=CleanupResponse, summary="Discover GSPlay processes")
def discover_processes() -> CleanupResponse:
    """Discover all running GSPlay processes (including orphaned ones)."""
    from gsplay_launcher.cli.cleanup import find_gsplay_processes

    processes = find_gsplay_processes()
    return CleanupResponse(
        processes=[
            ProcessInfo(
                pid=p.pid,
                port=p.port,
                config_path=p.config_path,
                memory_mb=round(p.memory_mb, 1),
                status=p.status,
            )
            for p in processes
        ],
        total=len(processes),
    )


@router.post("/cleanup/stop", response_model=CleanupResponse, summary="Stop GSPlay processes")
def stop_processes(body: CleanupStopRequest) -> CleanupResponse:
    """Stop discovered GSPlay processes."""
    from gsplay_launcher.cli.cleanup import find_gsplay_processes
    from gsplay_launcher.services.process_manager import stop_process

    processes = find_gsplay_processes()
    if body.pid is not None:
        processes = [p for p in processes if p.pid == body.pid]

    for proc in processes:
        stop_process(proc.pid, force=body.force)

    remaining = find_gsplay_processes()
    return CleanupResponse(
        processes=[
            ProcessInfo(
                pid=p.pid,
                port=p.port,
                config_path=p.config_path,
                memory_mb=round(p.memory_mb, 1),
                status=p.status,
            )
            for p in remaining
        ],
        total=len(remaining),
    )
