"""API module for FastAPI routes and schemas."""

from gsplay_launcher.api.app import create_app, set_config
from gsplay_launcher.api.routes import proxy_router, router

__all__ = [
    "create_app",
    "proxy_router",
    "router",
    "set_config",
]
