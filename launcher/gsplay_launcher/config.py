"""Configuration dataclasses for the launcher."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


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

        if not self.gsplay_script.exists():
            logger.warning(
                "GSPlay script not found at %s - "
                "processes may fail to start",
                self.gsplay_script,
            )
