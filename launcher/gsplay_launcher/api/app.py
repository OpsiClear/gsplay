"""FastAPI application factory for GSPlay Launcher."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from gsplay_launcher.api.routes import (
    proxy_router,
    router,
    set_file_browser,
    set_instance_manager,
)
from gsplay_launcher.config import LauncherConfig
from gsplay_launcher.services.file_browser import FileBrowserService
from gsplay_launcher.services.id_encoder import set_config as set_encoder_config
from gsplay_launcher.services.instance_manager import InstanceManager

logger = logging.getLogger(__name__)

# Template directory
_TEMPLATES_DIR = Path(__file__).parent.parent / "templates"

# Global config (set before creating app)
_config: LauncherConfig | None = None


def set_config(config: LauncherConfig) -> None:
    """Set the global configuration."""
    global _config
    _config = config


def _find_frontend_dir() -> Path | None:
    """Find the frontend dist directory.

    Search order:
    1. GSPLAY_FRONTEND_DIR environment variable (explicit override)
    2. Package static directory (gsplay_launcher/static/) - works for pip install
    3. Relative frontend/dist (for development with editable install)

    Returns
    -------
    Path | None
        Path to frontend dist directory, or None if not found.
    """
    # 1. Environment variable takes precedence (for production deployments)
    if env_path := os.environ.get("GSPLAY_FRONTEND_DIR"):
        path = Path(env_path).resolve()
        if (path / "index.html").exists():
            logger.info("Using frontend from GSPLAY_FRONTEND_DIR: %s", path)
            return path
        logger.warning("GSPLAY_FRONTEND_DIR set but no index.html found: %s", path)

    source_file = Path(__file__).resolve()

    # 2. Package static directory (gsplay_launcher/static/)
    # This works for both pip install and editable install
    package_static = source_file.parent.parent / "static"
    if (package_static / "index.html").exists():
        logger.info("Using frontend from package static: %s", package_static)
        return package_static

    # 3. Relative frontend/dist (development fallback)
    dev_frontend = source_file.parent.parent.parent / "frontend" / "dist"
    if (dev_frontend / "index.html").exists():
        logger.info("Using frontend from dev location: %s", dev_frontend)
        return dev_frontend

    logger.warning("No frontend build found")
    return None


def _load_dashboard_template() -> str:
    """Load the dashboard HTML template.

    Returns
    -------
    str
        Dashboard HTML content.

    Raises
    ------
    FileNotFoundError
        If template file is missing.
    """
    template_path = _TEMPLATES_DIR / "dashboard.html"
    if not template_path.exists():
        raise FileNotFoundError(f"Dashboard template not found: {template_path}")
    
    # Read the existing template content
    html_content = template_path.read_text(encoding="utf-8")

    return html_content


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan events."""
    if _config is None:
        raise RuntimeError("Config not set before creating app")

    # Initialize ID encoder for secure URLs
    set_encoder_config(_config)
    logger.debug("ID encoder initialized")

    # Startup: initialize instance manager
    logger.info("Initializing instance manager...")
    manager = InstanceManager(_config)
    manager.initialize()
    set_instance_manager(manager)

    # Initialize file browser if configured
    if _config.browse_path and _config.browse_path.is_dir():
        logger.info("Initializing file browser with root: %s", _config.browse_path)
        browser = FileBrowserService(_config.browse_path)
        set_file_browser(browser)
    else:
        logger.info("File browser disabled (no browse_path configured)")

    logger.info("Launcher started on http://%s:%d", _config.host, _config.port)

    yield

    # Shutdown: nothing to clean up (processes survive)
    logger.info("Launcher shutting down")


def create_app(config: LauncherConfig | None = None) -> FastAPI:
    """Create and configure FastAPI application.

    Parameters
    ----------
    config : LauncherConfig | None
        Launcher configuration. If None, uses global config.

    Returns
    -------
    FastAPI
        Configured application.
    """
    if config is not None:
        set_config(config)

    app = FastAPI(
        title="GSPlay",
        description="Launch and manage Gaussian Splatting GSPlay instances",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS for frontend development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",  # Vite dev server
            "http://127.0.0.1:5173",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API routes
    app.include_router(router)

    # Proxy routes (WebSocket and HTTP proxy for GSPlay instances)
    app.include_router(proxy_router)

    # Check for built frontend
    static_dir = _find_frontend_dir()

    if static_dir:
        # Serve SolidJS frontend
        frontend_html = (static_dir / "index.html").read_text(encoding="utf-8")

        @app.get("/", response_class=HTMLResponse)
        async def dashboard() -> str:
            return frontend_html

        # Mount static assets
        app.mount("/assets", StaticFiles(directory=str(static_dir / "assets")), name="assets")
        logger.info("Serving SolidJS frontend from %s", static_dir)
    else:
        # Fall back to embedded dashboard template
        dashboard_html = _load_dashboard_template()

        @app.get("/", response_class=HTMLResponse)
        async def dashboard() -> str:
            return dashboard_html

        logger.info("Serving embedded dashboard (no frontend build)")

    return app
