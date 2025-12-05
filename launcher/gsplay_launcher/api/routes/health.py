"""Health check route."""

from fastapi import APIRouter, Depends

from gsplay_launcher.api.dependencies import get_instance_manager
from gsplay_launcher.api.schemas import HealthResponse
from gsplay_launcher.services.instance_manager import InstanceManager

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="Health check")
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
