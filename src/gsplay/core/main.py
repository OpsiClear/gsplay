"""
Universal 4D Gaussian Splatting GSPlay - Main Entry Point.

This is the CLI entry point that uses tyro for argument parsing.
"""

from __future__ import annotations

# --- Monkeypatch for viser on Windows with uv/pip entry points ---
try:
    import viser._tunnel

    _original_is_multiprocess_ok = viser._tunnel._is_multiprocess_ok

    def _safe_is_multiprocess_ok() -> bool:
        try:
            return _original_is_multiprocess_ok()
        except (FileNotFoundError, OSError):
            return False

    viser._tunnel._is_multiprocess_ok = _safe_is_multiprocess_ok
except ImportError:
    pass
# -----------------------------------------------------------------

import json
import logging
from pathlib import Path
from typing import Annotated

import torch
import tyro

from src.gsplay.core.app import UniversalGSPlay
from src.gsplay.config.settings import GSPlayConfig

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    """
    Setup logging configuration.

    Parameters
    ----------
    level : str
        Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def main(
    config: Annotated[Path, tyro.conf.Positional],
    port: int = 6019,
    host: str = "0.0.0.0",
    log_level: str = "INFO",
    gpu: int | None = None,
    cache_size: int = 100,
    view_only: bool = False,
    compact: bool = False,
) -> None:
    """
    Universal 4D Gaussian Splatting GSPlay.

    Parameters
    ----------
    config : Path
        Path to model configuration (JSON file or PLY folder)
    port : int
        Viser server port (default: 6019)
    host : str
        Host to bind to (default: 0.0.0.0 for external access, use 127.0.0.1 for localhost only)
    log_level : str
        Logging level: DEBUG, INFO, WARNING, ERROR (default: INFO)
    gpu : int | None
        GPU device number (e.g., 0, 1, 2). If None, defaults to GPU 0 if CUDA is available, otherwise CPU.
        Examples: --gpu 0, --gpu 3
    cache_size : int
        Number of frames to cache in memory (default: 100)
    view_only : bool
        Hide input path, config save, and export options from UI (default: False)
    compact : bool
        Mobile-friendly UI with smaller control panel and collapsible sections (default: False)

    Examples
    --------
    View PLY sequence:
        uv run python src/viewer/main.py --config ./export_with_edits

    Use specific GPU:
        uv run python src/viewer/main.py --config ./export_with_edits --gpu 0
        uv run python src/viewer/main.py --config ./export_with_edits --gpu 3

    Default behavior (uses GPU 0 if available, otherwise CPU):
        uv run python src/viewer/main.py --config ./export_with_edits

    Custom port:
        uv run python src/viewer/main.py --config ./export_with_edits --port 8080

    Jellyfin streaming:
        uv run python src/viewer/main.py --config ./module_config/gif_elly.json

    Debug logging:
        uv run python src/viewer/main.py --config ./export_with_edits --log-level DEBUG
    """
    # Setup logging
    setup_logging(log_level)

    # Convert GPU number to device string
    if gpu is not None:
        device = f"cuda:{gpu}"
    else:
        # Default to GPU 0 if CUDA is available, otherwise CPU
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"

    logger.info("=== GSPlay ===")
    logger.info(f"Config: {config}")
    logger.info(f"Host: {host}")
    logger.info(f"Port: {port}")
    logger.info(f"Device: {device}")

    # Create viewer config
    viewer_config = GSPlayConfig(
        port=port,
        host=host,
        device=device,
        model_config_path=config,
        view_only=view_only,
        compact_ui=compact,
    )

    # Create viewer
    viewer = UniversalGSPlay(viewer_config)

    # Load model
    if config.is_dir():
        # PLY folder
        logger.info(f"Loading PLY files from directory: {config}")
        config_dict = {
            "module": "load-ply",
            "config": {
                "ply_folder": str(config),
                "device": device,
                "cache_size": cache_size,
            },
        }
        viewer.load_model_from_config(config_dict, config_file=str(config))

    elif config.is_file() and config.suffix == ".json":
        # JSON config
        logger.info(f"Loading configuration from: {config}")
        with open(config, "r") as f:
            config_dict = json.load(f)
        viewer.load_model_from_config(config_dict, config_file=str(config))

    else:
        logger.error(f"Invalid config: {config}")
        logger.error("Config must be either a PLY folder or JSON file")
        return

    # Setup viewer
    viewer.setup_viewer()

    # Run
    viewer.run()


def cli() -> None:
    """Entry point for the installed script."""
    tyro.cli(main)


if __name__ == "__main__":
    cli()
