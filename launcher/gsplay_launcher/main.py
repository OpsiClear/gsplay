"""CLI entry point for the gsplay launcher."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Annotated

import tyro
import uvicorn

from gsplay_launcher.api.app import create_app, set_config
from gsplay_launcher.config import LauncherConfig


def build_frontend(logger: logging.Logger) -> bool:
    """Build the frontend if source is newer than dist.

    Returns True if build succeeded or was skipped, False on error.
    """
    # Get the launcher package directory
    launcher_dir = Path(__file__).parent.parent
    frontend_dir = launcher_dir / "frontend"
    src_dir = frontend_dir / "src"
    dist_dir = frontend_dir / "dist"
    index_html = dist_dir / "index.html"

    # Check if frontend source exists
    if not src_dir.exists():
        logger.debug("Frontend source not found, skipping build")
        return True

    # Check if build is needed
    needs_build = False
    if not index_html.exists():
        needs_build = True
        logger.info("Frontend dist not found, building...")
    else:
        # Check if any source file is newer than dist
        dist_mtime = index_html.stat().st_mtime
        for src_file in src_dir.rglob("*"):
            if src_file.is_file() and src_file.stat().st_mtime > dist_mtime:
                needs_build = True
                logger.info("Frontend source changed, rebuilding...")
                break

    if not needs_build:
        logger.debug("Frontend is up to date")
        return True

    # Find deno executable
    deno_path = shutil.which("deno")
    if not deno_path:
        # Check common locations
        home = Path.home()
        for candidate in [
            home / ".deno" / "bin" / "deno",
            Path("/usr/local/bin/deno"),
            Path("/usr/bin/deno"),
        ]:
            if candidate.exists():
                deno_path = str(candidate)
                break

    if not deno_path:
        logger.warning("Deno not found, skipping frontend build. Install deno or build manually.")
        return True  # Not a fatal error

    # Run build
    try:
        logger.info("Building frontend with deno...")
        result = subprocess.run(
            [deno_path, "task", "build"],
            cwd=frontend_dir,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            logger.error("Frontend build failed:\n%s", result.stderr)
            return False
        logger.info("Frontend build completed")
        return True
    except subprocess.TimeoutExpired:
        logger.error("Frontend build timed out")
        return False
    except Exception as e:
        logger.error("Frontend build error: %s", e)
        return False


def setup_logging(level: str) -> None:
    """Configure logging.

    Parameters
    ----------
    level : str
        Logging level name.
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Reduce noise from third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def main(
    host: Annotated[str, tyro.conf.arg(help="Host to bind the launcher API (0.0.0.0 for external access)")] = "0.0.0.0",
    port: Annotated[int, tyro.conf.arg(help="Port for the launcher API")] = 8000,
    gsplay_host: Annotated[str, tyro.conf.arg(help="Host for GSPlay instances (0.0.0.0 for external access)")] = "0.0.0.0",
    log_level: Annotated[str, tyro.conf.arg(help="Logging level")] = "INFO",
    gsplay_script: Annotated[
        str,
        tyro.conf.arg(help="Path to gsplay main.py script"),
    ] = "src/gsplay/core/main.py",
    data_dir: Annotated[
        str,
        tyro.conf.arg(help="Directory for launcher data"),
    ] = "data",
    gsplay_port_start: Annotated[
        int,
        tyro.conf.arg(help="Start of gsplay port range"),
    ] = 6020,
    gsplay_port_end: Annotated[
        int,
        tyro.conf.arg(help="End of gsplay port range"),
    ] = 6100,
    browse_path: Annotated[
        str | None,
        tyro.conf.arg(help="Root directory for file browser (omit to disable)"),
    ] = None,
    custom_ip: Annotated[
        str | None,
        tyro.conf.arg(help="Default IP address for instance URLs (omit for auto-detect)"),
    ] = None,
    external_url: Annotated[
        str | None,
        tyro.conf.arg(help="External base URL for proxy access (e.g., https://gsplay.4dgst.win)"),
    ] = None,
    network_url: Annotated[
        str | None,
        tyro.conf.arg(help="Persistent base URL for viser viewer and streaming (e.g., https://gsplay.example.com). Port appended automatically."),
    ] = None,
    view_only: Annotated[
        bool,
        tyro.conf.arg(help="Force all instances to launch in view-only mode (hides data loader UI)"),
    ] = False,
    history_limit: Annotated[
        int,
        tyro.conf.arg(help="Maximum number of launch history entries to show in UI"),
    ] = 5,
) -> None:
    """GSPlay Launcher - Manage Gaussian Splatting GSPlay instances.

    Launch a FastAPI server that provides REST APIs to create, manage,
    and stop GSPlay instances. Each gsplay runs as a separate process
    that can survive launcher restarts.

    Examples
    --------
    Start launcher with defaults:
        uv run src/main.py

    Start on custom port:
        uv run src/main.py --port 8080

    With custom gsplay script path:
        uv run src/main.py --gsplay-script /path/to/gsplay/main.py
    """
    setup_logging(log_level)
    logger = logging.getLogger(__name__)

    # Resolve paths from current working directory
    cwd = Path.cwd()
    gsplay_script_path = (cwd / gsplay_script).resolve()
    data_dir_path = (cwd / data_dir).resolve()

    # Resolve browse path if provided
    browse_path_resolved: Path | None = None
    if browse_path:
        # Path() handles both absolute and relative paths correctly
        browse_path_resolved = Path(browse_path).resolve()
        if not browse_path_resolved.is_dir():
            logger.error("Browse path does not exist or is not a directory: %s", browse_path_resolved)
            raise SystemExit(1)

    # Create configuration
    config = LauncherConfig(
        host=host,
        port=port,
        gsplay_host=gsplay_host,
        gsplay_port_start=gsplay_port_start,
        gsplay_port_end=gsplay_port_end,
        data_dir=data_dir_path,
        gsplay_script=gsplay_script_path,
        browse_path=browse_path_resolved,
        custom_ip=custom_ip,
        external_url=external_url.rstrip('/') if external_url else None,
        network_url=network_url.rstrip('/') if network_url else None,
        view_only=view_only,
        history_limit=history_limit,
    )

    # Validate configuration
    try:
        config.validate()
    except ValueError as e:
        logger.error("Configuration error: %s", e)
        raise SystemExit(1)

    logger.info("Starting GSPlay Launcher")
    logger.info("  Host: %s", host)
    logger.info("  Port: %d", port)
    logger.info("  GSPlay host: %s", gsplay_host)
    logger.info("  GSPlay script: %s", gsplay_script_path)
    logger.info("  Data directory: %s", data_dir_path)
    logger.info("  GSPlay port range: [%d, %d)", gsplay_port_start, gsplay_port_end)
    if browse_path_resolved:
        logger.info("  Browse path: %s", browse_path_resolved)
    else:
        logger.info("  Browse path: disabled")
    if custom_ip:
        logger.info("  Custom IP: %s", custom_ip)
    else:
        logger.info("  Custom IP: auto-detect")
    if external_url:
        logger.info("  External URL: %s", external_url)
    else:
        logger.info("  External URL: disabled (use --external-url to enable)")
    if network_url:
        logger.info("  Network URL: %s", network_url)
    else:
        logger.info("  Network URL: disabled (use --network-url for persistent URLs)")
    if view_only:
        logger.info("  View-only mode: enabled (all instances will hide data loader)")

    # Build frontend if needed
    build_frontend(logger)

    # Set config and create app
    set_config(config)
    app = create_app()

    # Run server
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=log_level.lower(),
    )


def cli() -> None:
    """Entry point for installed script."""
    tyro.cli(main)


if __name__ == "__main__":
    cli()
