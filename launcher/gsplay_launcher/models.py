"""Domain models for gsplay instance management."""

from __future__ import annotations

import json
import socket
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4


def _get_machine_ip() -> str:
    """Get machine's Ethernet IP address for external URL generation."""
    try:
        # Create a socket to determine the primary network interface IP
        # This doesn't actually connect, just determines which interface would be used
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        # Fallback to localhost if we can't determine IP
        return "127.0.0.1"


class InstanceStatus(str, Enum):
    """Status of a gsplay instance.

    State transitions:
        PENDING -> STARTING -> RUNNING
        RUNNING -> STOPPING -> STOPPED
        RUNNING -> FAILED (process died unexpectedly)
        STARTING -> FAILED (failed to start)
        * -> ORPHANED (found after launcher restart)
    """

    PENDING = "pending"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"
    ORPHANED = "orphaned"


def _generate_short_id() -> str:
    """Generate a short unique ID."""
    return uuid4().hex[:8]


def _now_iso() -> str:
    """Get current time as ISO string."""
    return datetime.now().isoformat()


@dataclass
class GSPlayInstance:
    """Represents a gsplay instance with its state and configuration.

    Parameters
    ----------
    id : str
        Unique identifier (short UUID).
    name : str
        Human-readable name for the instance.
    config_path : str
        Path to PLY folder or JSON config file.
    port : int
        Port the gsplay is/will be running on.
    host : str
        Host to bind the gsplay server (0.0.0.0 for external access).
    gpu : int | None
        GPU device number (None for default).

    view_only : bool
        Whether to hide editing UI.
    compact : bool
        Whether to use compact/mobile UI.
    log_level : str
        Logging level for the gsplay.
    custom_ip : str | None
        Custom IP address for URL generation (None = auto-detect).
    viewer_id : str | None
        Custom ID for viewer proxy path /v/{id}/ (None = use instance id).
    stream_token : str | None
        Custom token for stream proxy path /s/{token}/ (None = use encoded instance id).
    status : InstanceStatus
        Current instance status.
    pid : int | None
        Process ID when running.
    created_at : str
        ISO timestamp of creation.
    started_at : str | None
        ISO timestamp of last start.
    stopped_at : str | None
        ISO timestamp of last stop.
    error_message : str | None
        Error message if status is FAILED.
    """

    id: str = field(default_factory=_generate_short_id)
    name: str = ""
    config_path: str = ""
    port: int = 6020
    host: str = "0.0.0.0"  # Bind to all interfaces for external access
    stream_port: int = -1  # WebSocket stream port (-1 = auto-assign to viser_port+1, 0 = disabled)
    gpu: int | None = None
    view_only: bool = False
    compact: bool = False
    log_level: str = "INFO"
    custom_ip: str | None = None  # Custom IP for URL (None = auto-detect)
    viewer_id: str | None = None  # Custom ID for /v/{id}/ path (None = use instance id)
    stream_token: str | None = None  # Custom token for /s/{token}/ path (None = use encoded id)
    status: InstanceStatus = InstanceStatus.PENDING
    pid: int | None = None
    created_at: str = field(default_factory=_now_iso)
    started_at: str | None = None
    stopped_at: str | None = None
    error_message: str | None = None

    @property
    def url(self) -> str:
        """Get the gsplay URL for external access."""
        # Use custom IP if provided, otherwise auto-detect
        if self.custom_ip:
            return f"http://{self.custom_ip}:{self.port}"
        # Use machine IP for external access when bound to 0.0.0.0
        if self.host == "0.0.0.0":
            return f"http://{_get_machine_ip()}:{self.port}"
        return f"http://{self.host}:{self.port}"

    @property
    def stream_url(self) -> str | None:
        """Get the WebSocket stream URL if streaming is enabled.

        Stream port convention:
        - stream_port == 0: Streaming disabled
        - stream_port != 0: Streaming enabled, actual port = viser_port + 1

        This ensures viser uses even ports and stream uses odd ports.
        """
        if self.stream_port == 0:
            return None
        # Actual stream port is always viser_port + 1
        actual_stream_port = self.port + 1
        # Use custom IP if provided, otherwise auto-detect
        if self.custom_ip:
            return f"http://{self.custom_ip}:{actual_stream_port}/view"
        # Use machine IP for external access when bound to 0.0.0.0
        if self.host == "0.0.0.0":
            return f"http://{_get_machine_ip()}:{actual_stream_port}/view"
        return f"http://{self.host}:{actual_stream_port}/view"

    @property
    def is_active(self) -> bool:
        """Check if instance is in an active state (process may be running)."""
        return self.status in (
            InstanceStatus.STARTING,
            InstanceStatus.RUNNING,
            InstanceStatus.ORPHANED,
        )

    def to_cli_args(self) -> list[str]:
        """Convert to CLI argument list for subprocess.

        Returns
        -------
        list[str]
            Arguments to pass to the gsplay script.
        """
        args = [
            self.config_path,
            "--port",
            str(self.port),
            "--host",
            self.host,
            "--log-level",
            self.log_level,
        ]

        # Any non-zero value enables streaming (gsplay will use viser_port + 1)
        if self.stream_port != 0:
            args.extend(["--stream-port", str(self.stream_port)])

        if self.gpu is not None:
            args.extend(["--gpu", str(self.gpu)])

        if self.view_only:
            args.append("--view-only")

        if self.compact:
            args.append("--compact")

        return args

    def mark_starting(self) -> None:
        """Mark instance as starting."""
        self.status = InstanceStatus.STARTING
        self.error_message = None

    def mark_running(self, pid: int) -> None:
        """Mark instance as running with given PID."""
        self.status = InstanceStatus.RUNNING
        self.pid = pid
        self.started_at = _now_iso()
        self.error_message = None

    def mark_stopping(self) -> None:
        """Mark instance as stopping."""
        self.status = InstanceStatus.STOPPING

    def mark_stopped(self) -> None:
        """Mark instance as stopped."""
        self.status = InstanceStatus.STOPPED
        self.stopped_at = _now_iso()
        self.pid = None

    def mark_failed(self, error: str) -> None:
        """Mark instance as failed with error message."""
        self.status = InstanceStatus.FAILED
        self.error_message = error
        self.pid = None

    def mark_orphaned(self) -> None:
        """Mark instance as orphaned (found after restart)."""
        self.status = InstanceStatus.ORPHANED

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GSPlayInstance:
        """Create instance from dictionary.

        Parameters
        ----------
        data : dict
            Dictionary with instance data.

        Returns
        -------
        GSPlayInstance
            Reconstructed instance.
        """
        data = data.copy()
        data["status"] = InstanceStatus(data["status"])
        return cls(**data)


@dataclass
class LauncherState:
    """Complete launcher state for persistence.

    Parameters
    ----------
    instances : dict[str, GSPlayInstance]
        Map of instance ID to instance.
    next_port_hint : int
        Hint for next port to try allocating.
    """

    instances: dict[str, GSPlayInstance] = field(default_factory=dict)
    next_port_hint: int = 6020

    def to_json(self) -> str:
        """Serialize to JSON string."""
        data = {
            "instances": {k: v.to_dict() for k, v in self.instances.items()},
            "next_port_hint": self.next_port_hint,
        }
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> LauncherState:
        """Deserialize from JSON string.

        Parameters
        ----------
        json_str : str
            JSON string to parse.

        Returns
        -------
        LauncherState
            Reconstructed state.
        """
        data = json.loads(json_str)
        instances = {k: GSPlayInstance.from_dict(v) for k, v in data.get("instances", {}).items()}
        return cls(
            instances=instances,
            next_port_hint=data.get("next_port_hint", 6020),
        )
