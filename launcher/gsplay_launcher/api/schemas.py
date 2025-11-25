"""Pydantic schemas for API request/response models."""

from __future__ import annotations

from pydantic import BaseModel, Field

from gsplay_launcher.models import GSPlayInstance


class CreateInstanceRequest(BaseModel):
    """Request body for creating a new instance."""

    config_path: str = Field(..., description="Path to PLY folder or JSON config")
    name: str = Field("", description="Human-readable name")
    port: int | None = Field(None, description="Port number (auto-assigned if null)")
    host: str | None = Field(None, description="Host to bind to (uses launcher default if null, 0.0.0.0 for external access)")
    gpu: int | None = Field(None, description="GPU device number")
    cache_size: int = Field(100, description="Frame cache size")
    view_only: bool = Field(False, description="Hide editing UI")
    compact: bool = Field(False, description="Use compact/mobile UI")
    log_level: str = Field("INFO", description="Logging level")


class InstanceResponse(BaseModel):
    """Response containing instance details."""

    id: str
    name: str
    status: str
    port: int
    url: str
    config_path: str
    gpu: int | None
    pid: int | None
    created_at: str
    started_at: str | None
    stopped_at: str | None
    error_message: str | None

    @classmethod
    def from_instance(cls, instance: GSPlayInstance) -> InstanceResponse:
        """Create response from GSPlayInstance.

        Parameters
        ----------
        instance : GSPlayInstance
            Instance to convert.

        Returns
        -------
        InstanceResponse
            Response model.
        """
        return cls(
            id=instance.id,
            name=instance.name,
            status=instance.status.value,
            port=instance.port,
            url=instance.url,
            config_path=instance.config_path,
            gpu=instance.gpu,
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


# Browse API schemas


class DirectoryEntryResponse(BaseModel):
    """A directory entry with PLY metadata."""

    name: str
    path: str  # Relative path from browse root
    is_directory: bool
    is_ply_folder: bool = False
    ply_count: int = 0
    total_size_mb: float = 0.0
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


class BrowseLaunchRequest(BaseModel):
    """Request body for launching from browser path."""

    path: str = Field(..., description="Relative path to launch")
    name: str = Field("", description="Human-readable name")
    port: int | None = Field(None, description="Port number (auto-assigned if null)")
    gpu: int | None = Field(None, description="GPU device number")
    cache_size: int = Field(100, description="Frame cache size")
    view_only: bool = Field(False, description="Hide editing UI")
    compact: bool = Field(False, description="Use compact/mobile UI")
