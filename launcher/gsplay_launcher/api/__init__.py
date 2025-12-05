"""API module for FastAPI routes and schemas."""

from gsplay_launcher.api.app import create_app, set_config
from gsplay_launcher.api.routes import api_router, proxy_router

__all__ = [
    "create_app",
    "api_router",
    "proxy_router",
    "set_config",
]
