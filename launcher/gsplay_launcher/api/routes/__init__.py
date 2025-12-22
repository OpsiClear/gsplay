"""API route modules for GSPlay Launcher.

Each module handles a single responsibility following SRP.
"""

from fastapi import APIRouter

from gsplay_launcher.api.routes.browse import router as browse_router
from gsplay_launcher.api.routes.cleanup import router as cleanup_router
from gsplay_launcher.api.routes.control import router as control_router
from gsplay_launcher.api.routes.gpu import router as gpu_router
from gsplay_launcher.api.routes.health import router as health_router
from gsplay_launcher.api.routes.instances import router as instances_router
from gsplay_launcher.api.routes.logs import router as logs_router
from gsplay_launcher.api.routes.proxy import router as proxy_router


# Main API router (prefixed with /api)
api_router = APIRouter(prefix="/api", tags=["api"])
api_router.include_router(health_router)
api_router.include_router(instances_router)
api_router.include_router(gpu_router)
api_router.include_router(browse_router)
api_router.include_router(logs_router)
api_router.include_router(cleanup_router)
api_router.include_router(control_router)

__all__ = ["api_router", "proxy_router"]
