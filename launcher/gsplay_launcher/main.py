"""CLI entry point for the gsplay launcher."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import tyro
import uvicorn

from gsplay_launcher.api.app import create_app, set_config
from gsplay_launcher.config import LauncherConfig


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
