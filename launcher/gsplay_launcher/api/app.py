"""FastAPI application factory for GSPlay Launcher."""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from gsplay_launcher.api.dependencies import STATE_KEY, AppState
from gsplay_launcher.api.routes import api_router, proxy_router
from gsplay_launcher.config import LauncherConfig
from gsplay_launcher.services.file_browser import FileBrowserService
from gsplay_launcher.services.id_encoder import set_config as set_encoder_config
from gsplay_launcher.services.instance_manager import InstanceManager


logger = logging.getLogger(__name__)

# Global config (set before creating app)
_config: LauncherConfig | None = None


def set_config(config: LauncherConfig) -> None:
    """Set the global configuration."""
    global _config
    _config = config


def _find_frontend_dir() -> Path:
    """Find the frontend dist directory."""
    if env_path := os.environ.get("GSPLAY_FRONTEND_DIR"):
        path = Path(env_path).resolve()
        if (path / "index.html").exists():
            logger.info("Using frontend from GSPLAY_FRONTEND_DIR: %s", path)
            return path
        logger.warning("GSPLAY_FRONTEND_DIR set but no index.html found: %s", path)

    cwd = Path.cwd().resolve()
    for candidate in [cwd / "launcher" / "frontend" / "dist", cwd / "frontend" / "dist"]:
        if (candidate / "index.html").exists():
            logger.info("Using frontend from: %s", candidate)
            return candidate

    source_file = Path(__file__).resolve()
    dev_frontend = source_file.parent.parent.parent / "frontend" / "dist"
    if (dev_frontend / "index.html").exists():
        logger.info("Using frontend from dev location: %s", dev_frontend)
        return dev_frontend

    package_static = source_file.parent.parent / "static"
    if (package_static / "index.html").exists():
        logger.info("Using frontend from package static: %s", package_static)
        return package_static

    raise FileNotFoundError(
        "Frontend build not found. Please build the frontend first:\n"
        "  cd launcher/frontend && deno task build\n"
        "Or set GSPLAY_FRONTEND_DIR environment variable."
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan events."""
    if _config is None:
        raise RuntimeError("Config not set before creating app")

    # Initialize ID encoder for secure URLs
    set_encoder_config(_config)
    logger.debug("ID encoder initialized")

    # Create application state container (replaces global mutable state)
    state = AppState()

    # Initialize instance manager
    logger.info("Initializing instance manager...")
    state.instance_manager = InstanceManager(_config)
    state.instance_manager.initialize()

    # Initialize file browser if configured
    if _config.browse_path and _config.browse_path.is_dir():
        logger.info("Initializing file browser with root: %s", _config.browse_path)
        state.file_browser = FileBrowserService(_config.browse_path)
    else:
        logger.info("File browser disabled (no browse_path configured)")

    # Attach state to app for dependency injection
    setattr(app.state, STATE_KEY, state)

    logger.info("Launcher started on http://%s:%d", _config.host, _config.port)
    yield
    logger.info("Launcher shutting down")


def create_app(config: LauncherConfig | None = None) -> FastAPI:
    """Create and configure FastAPI application."""
    if config is not None:
        set_config(config)

    app = FastAPI(
        title="GSPlay",
        description="Launch and manage Gaussian Splatting GSPlay instances",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow all origins for WebSocket proxy support
        allow_credentials=False,  # Cannot use credentials with wildcard origin
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(api_router)
    app.include_router(proxy_router)

    # Serve frontend
    static_dir = _find_frontend_dir()
    frontend_html = (static_dir / "index.html").read_text(encoding="utf-8")

    @app.get("/", response_class=HTMLResponse)
    async def dashboard() -> str:
        return frontend_html

    app.mount("/assets", StaticFiles(directory=str(static_dir / "assets")), name="assets")
    logger.info("Serving SolidJS frontend from %s", static_dir)

    return app
