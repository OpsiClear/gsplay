"""FastAPI application factory for GSPlay Launcher."""

from __future__ import annotations

import logging
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

    # Load dashboard template
    dashboard_html = _load_dashboard_template()

    # Serve dashboard at root
    @app.get("/", response_class=HTMLResponse)
    async def dashboard() -> str:
        return dashboard_html

    # Check for built frontend and serve static assets
    static_dir = Path(__file__).parent.parent.parent / "frontend" / "dist"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
        logger.info("Serving static assets from %s", static_dir)
    else:
        logger.info("Serving embedded dashboard (no build)")

    return app
