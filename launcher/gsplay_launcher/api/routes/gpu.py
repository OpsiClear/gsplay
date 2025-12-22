"""GPU and system information routes."""

import sys

from fastapi import APIRouter, HTTPException, status

from gsplay_launcher.api.schemas import (
    ErrorResponse,
    GpuInfo,
    GpuInfoResponse,
    MsvcStatusResponse,
    SystemStatsResponse,
)
from gsplay_launcher.services.gpu_info import get_gpu_service, get_system_stats


router = APIRouter(tags=["system"])


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
            status.HTTP_503_SERVICE_UNAVAILABLE, "nvidia-smi not available or failed"
        )
    return GpuInfoResponse(
        gpus=[
            GpuInfo(
                index=g.index,
                name=g.name,
                memory_used=g.memory_used,
                memory_total=g.memory_total,
                utilization=g.utilization,
                temperature=g.temperature,
            )
            for g in info.gpus
        ],
        driver_version=info.driver_version,
        cuda_version=info.cuda_version,
    )


@router.get("/system", response_model=SystemStatsResponse, summary="Get system CPU/memory stats")
def get_system_info() -> SystemStatsResponse:
    """Get system CPU and memory statistics."""
    stats = get_system_stats()
    return SystemStatsResponse(
        cpu_percent=stats.cpu_percent,
        memory_used_gb=stats.memory_used_gb,
        memory_total_gb=stats.memory_total_gb,
        memory_percent=stats.memory_percent,
    )


@router.get(
    "/system/msvc", response_model=MsvcStatusResponse, summary="Get MSVC compiler status (Windows)"
)
def get_msvc_status_endpoint() -> MsvcStatusResponse:
    """Check MSVC compiler availability for CUDA JIT compilation."""
    if sys.platform != "win32":
        return MsvcStatusResponse(
            available=True,
            in_path=True,
            vcvars_path=None,
            message="MSVC not required on this platform",
            platform=sys.platform,
        )
    from gsplay_launcher.services.process_manager import get_msvc_status

    s = get_msvc_status()
    return MsvcStatusResponse(
        available=s["available"],
        in_path=s["in_path"],
        vcvars_path=s["vcvars_path"],
        message=s["message"],
        platform=sys.platform,
    )
