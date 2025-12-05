"""Configuration dataclasses for the launcher."""

from __future__ import annotations

import hashlib
import logging
import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


def _generate_url_secret() -> str:
    """Generate a random URL secret for ID encoding."""
    return secrets.token_hex(16)


@dataclass
class LauncherConfig:
    """Main launcher configuration.

    Parameters
    ----------
    host : str
        Host to bind the launcher API server.
    port : int
        Port for the launcher API server.
    gsplay_host : str
        Host for GSPlay instances to bind to (0.0.0.0 for external access).
    gsplay_port_start : int
        Start of port range for GSPlay instances.
    gsplay_port_end : int
        End of port range for GSPlay instances.
    data_dir : Path
        Directory for storing launcher state.
    gsplay_script : Path
        Path to the gsplay main.py script.
    process_stop_timeout : float
        Timeout in seconds for graceful process shutdown.
    browse_path : Path | None
        Root directory for file browser (None = disabled).
    custom_ip : str | None
        Default custom IP for instance URLs (None = auto-detect).
    external_url : str | None
        External base URL for reverse proxy access (e.g., https://gsplay.4dgst.win).
        When set, enables proxy routes at /v/{instance_id}/ for external access.
    network_url : str | None
        Persistent base URL for both viser viewer and streaming channel
        (e.g., https://gsplay.example.com). Overrides auto-detected IP for direct URLs.
        Port is appended automatically (viewer: port, stream: port+1).
    history_limit : int
        Maximum number of launch history entries to show in UI.
    """

    host: str = "0.0.0.0"  # Bind to all interfaces for external access
    port: int = 8000
    gsplay_host: str = "0.0.0.0"  # GSPlay instances bind to all interfaces
    gsplay_port_start: int = 6020
    gsplay_port_end: int = 6100
    data_dir: Path = field(default_factory=lambda: Path("./data"))
    gsplay_script: Path = field(
        default_factory=lambda: Path("./src/gsplay/core/main.py")
    )
    process_stop_timeout: float = 10.0
    browse_path: Path | None = None  # Root for file browser (None = disabled)
    custom_ip: str | None = None  # Default custom IP for URLs (None = auto)
    external_url: str | None = None  # External base URL for proxy access (e.g., https://gsplay.4dgst.win)
    network_url: str | None = None  # Persistent base URL for both viser viewer and streaming channel (e.g., https://gsplay.example.com). Overrides auto-detected IP for direct URLs.
    view_only: bool = False  # Force all instances to launch in view-only mode
    history_limit: int = 5  # Maximum number of launch history entries to show in UI
    url_secret: str = field(default_factory=_generate_url_secret)  # Secret for encoding instance IDs in URLs

    @property
    def state_file(self) -> Path:
        """Path to the instances state file."""
        return self.data_dir / "instances.json"

    def ensure_data_dir(self) -> None:
        """Ensure the data directory exists."""
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def validate(self) -> None:
        """Validate configuration values.

        Raises
        ------
        ValueError
            If configuration is invalid.
        """
        if self.gsplay_port_start >= self.gsplay_port_end:
            msg = "gsplay_port_start must be less than gsplay_port_end"
            raise ValueError(msg)

        if self.port < 1 or self.port > 65535:
            msg = f"Invalid launcher port: {self.port}"
            raise ValueError(msg)

        # Ensure port start is even (convention: even ports for viser, odd for stream)
        if self.gsplay_port_start % 2 != 0:
            logger.warning(
                "gsplay_port_start=%d is odd; adjusting to %d for even/odd port convention",
                self.gsplay_port_start,
                self.gsplay_port_start + 1,
            )
            self.gsplay_port_start += 1

        if not self.gsplay_script.exists():
            logger.warning(
                "GSPlay script not found at %s - "
                "processes may fail to start",
                self.gsplay_script,
            )
