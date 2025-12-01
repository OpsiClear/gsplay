"""Pydantic schemas for API request/response models."""

from __future__ import annotations

from pydantic import BaseModel, Field

from gsplay_launcher.models import GSPlayInstance
from gsplay_launcher.services.id_encoder import encode_instance_id


class CreateInstanceRequest(BaseModel):
    """Request body for creating a new instance."""

    config_path: str = Field(..., description="Path to PLY folder or JSON config")
    name: str = Field("", description="Human-readable name")
    port: int | None = Field(None, description="Port number (auto-assigned if null)")
    host: str | None = Field(None, description="Host to bind to (uses launcher default if null, 0.0.0.0 for external access)")
    stream_port: int = Field(-1, description="WebSocket stream port (-1 = auto-assign to viser_port+1, 0 = disabled)")
    gpu: int | None = Field(None, description="GPU device number")
    cache_size: int = Field(100, description="Frame cache size")
    view_only: bool = Field(False, description="Hide editing UI")
    compact: bool = Field(False, description="Use compact/mobile UI")
    log_level: str = Field("INFO", description="Logging level")
    custom_ip: str | None = Field(None, description="Custom IP for Open button URL (auto-detect if null)")


class InstanceResponse(BaseModel):
    """Response containing instance details."""

    id: str
    name: str
    status: str
    port: int
    url: str
    stream_port: int = 0
    stream_url: str | None = None  # Direct stream URL (for local access)
    encoded_stream_path: str | None = None  # Encoded path for proxy access (e.g., /s/TOKEN/)
    config_path: str
    gpu: int | None
    cache_size: int = 100
    view_only: bool = False
    compact: bool = False
    pid: int | None
    created_at: str
    started_at: str | None
    stopped_at: str | None
    error_message: str | None

    @classmethod
    def from_instance(
        cls,
        instance: GSPlayInstance,
        external_url: str | None = None,
    ) -> InstanceResponse:
        """Create response from GSPlayInstance.

        Parameters
        ----------
        instance : GSPlayInstance
            Instance to convert.
        external_url : str | None
            External base URL for encoded stream paths. If provided,
            encoded_stream_path will be a full URL.

        Returns
        -------
        InstanceResponse
            Response model.
        """
        # Generate encoded stream path if streaming is enabled
        # Any non-zero stream_port means streaming is enabled
        encoded_stream_path = None
        if instance.stream_port != 0:
            try:
                token = encode_instance_id(instance.id)
                path = f"/s/{token}/"
                if external_url:
                    # Full URL with external base
                    encoded_stream_path = f"{external_url.rstrip('/')}{path}"
                else:
                    # Just the path (frontend will use relative URL)
                    encoded_stream_path = path
            except Exception:
                pass  # Encoder not initialized yet

        return cls(
            id=instance.id,
            name=instance.name,
            status=instance.status.value,
            port=instance.port,
            url=instance.url,
            stream_port=instance.stream_port,
            stream_url=instance.stream_url,
            encoded_stream_path=encoded_stream_path,
            config_path=instance.config_path,
            gpu=instance.gpu,
            cache_size=instance.cache_size,
            view_only=instance.view_only,
            compact=instance.compact,
            pid=instance.pid,
            created_at=instance.created_at,
            started_at=instance.started_at,
            stopped_at=instance.stopped_at,
            error_message=instance.error_message,
        )


class InstanceListResponse(BaseModel):
    """Response containing list of instances."""

    instances: list[InstanceResponse]
    total: int


class PortInfoResponse(BaseModel):
    """Response for port availability check."""

    next_available: int | None
    range_start: int
    range_end: int


class HealthResponse(BaseModel):
    """Response for health check."""

    status: str
    instance_count: int
    active_count: int


class ErrorResponse(BaseModel):
    """Error response model."""

    detail: str


class GpuInfo(BaseModel):
    """Information about a single GPU."""

    index: int
    name: str
    memory_used: int  # MB
    memory_total: int  # MB
    utilization: int  # %
    temperature: int  # C


class GpuInfoResponse(BaseModel):
    """Response containing GPU information."""

    gpus: list[GpuInfo]
    driver_version: str
    cuda_version: str


class SystemStatsResponse(BaseModel):
    """Response containing system CPU/memory stats."""

    cpu_percent: float
    memory_used_gb: float
    memory_total_gb: float
    memory_percent: float


# Browse API schemas


class DirectoryEntryResponse(BaseModel):
    """A directory entry with PLY metadata."""

    name: str
    path: str  # Relative path from browse root
    is_directory: bool
    is_ply_folder: bool = False
    ply_count: int = 0
    total_size_mb: float = 0.0
    subfolder_count: int = 0  # Number of subdirectories (for non-PLY folders)
    modified_at: str | None = None


class BrowseResponse(BaseModel):
    """Response for directory browsing."""

    current_path: str
    breadcrumbs: list[dict[str, str]]  # [{name, path}, ...]
    entries: list[DirectoryEntryResponse]
    browse_enabled: bool = True


class BrowseConfigResponse(BaseModel):
    """Response for browse configuration check."""

    enabled: bool
    root_path: str | None = None
    default_custom_ip: str | None = None
    external_url: str | None = None  # External base URL for proxy access (e.g., https://gsplay.4dgst.win)
    view_only: bool = False  # If true, all instances are forced to view-only mode
    history_limit: int = 5  # Maximum number of launch history entries to show in UI


class BrowseLaunchRequest(BaseModel):
    """Request body for launching from browser path."""

    path: str = Field(..., description="Relative path to launch")
    name: str = Field("", description="Human-readable name")
    port: int | None = Field(None, description="Port number (auto-assigned if null)")
    stream_port: int = Field(-1, description="WebSocket stream port (-1 = auto-assign to viser_port+1, 0 = disabled)")
    gpu: int | None = Field(None, description="GPU device number")
    cache_size: int = Field(100, description="Frame cache size")
    view_only: bool = Field(False, description="Hide editing UI")
    compact: bool = Field(False, description="Use compact/mobile UI")
    custom_ip: str | None = Field(None, description="Custom IP for Open button URL (auto-detect if null)")


# Log API schemas


class LogResponse(BaseModel):
    """Response containing log lines."""

    lines: list[str]
    total_lines: int
    offset: int
    has_more: bool


# Cleanup API schemas


class ProcessInfo(BaseModel):
    """Information about a GSPlay process."""

    pid: int
    port: int | None
    config_path: str | None
    memory_mb: float
    status: str


class CleanupResponse(BaseModel):
    """Response for cleanup operations."""

    processes: list[ProcessInfo]
    total: int


class CleanupStopRequest(BaseModel):
    """Request body for stopping processes."""

    force: bool = Field(False, description="Force kill without graceful shutdown")
    pid: int | None = Field(None, description="Stop specific PID (None = stop all)")
