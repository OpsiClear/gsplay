"""GPU and system information services."""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass

import psutil

logger = logging.getLogger(__name__)


def _safe_int(value: str, default: int = 0) -> int:
    """Safely parse an integer from nvidia-smi output.

    nvidia-smi can return '[N/A]' for unavailable values.
    """
    value = value.strip()
    if not value or value == "[N/A]" or value.startswith("["):
        return default
    try:
        return int(value)
    except ValueError:
        return default


@dataclass(frozen=True)
class SystemStats:
    """System CPU and memory statistics."""

    cpu_percent: float
    memory_used_gb: float
    memory_total_gb: float
    memory_percent: float


def get_system_stats() -> SystemStats:
    """Get current system CPU and memory stats."""
    cpu_percent = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory()
    return SystemStats(
        cpu_percent=cpu_percent,
        memory_used_gb=round(mem.used / (1024**3), 1),
        memory_total_gb=round(mem.total / (1024**3), 1),
        memory_percent=mem.percent,
    )


@dataclass(frozen=True)
class GpuInfo:
    """Information about a single GPU."""

    index: int
    name: str
    memory_used: int  # MB
    memory_total: int  # MB
    utilization: int  # percentage
    temperature: int  # Celsius

    @property
    def memory_percent(self) -> float:
        """Memory usage as percentage."""
        if self.memory_total == 0:
            return 0.0
        return (self.memory_used / self.memory_total) * 100


@dataclass(frozen=True)
class SystemGpuInfo:
    """System-wide GPU information."""

    gpus: tuple[GpuInfo, ...]
    driver_version: str
    cuda_version: str

    @property
    def gpu_count(self) -> int:
        """Number of GPUs detected."""
        return len(self.gpus)


class GpuInfoService:
    """Service for querying GPU information via nvidia-smi.

    This service caches the last successful result to handle
    transient nvidia-smi failures gracefully.
    """

    def __init__(self, timeout: float = 5.0) -> None:
        """Initialize GPU info service.

        Parameters
        ----------
        timeout : float
            Timeout in seconds for nvidia-smi commands.
        """
        self._timeout = timeout
        self._last_result: SystemGpuInfo | None = None

    def get_info(self) -> SystemGpuInfo | None:
        """Get current GPU information.

        Returns
        -------
        SystemGpuInfo | None
            GPU information, or None if nvidia-smi is unavailable.
        """
        try:
            gpus = self._query_gpu_stats()
            driver_version = self._query_driver_version()
            cuda_version = self._query_cuda_version()

            result = SystemGpuInfo(
                gpus=tuple(gpus),
                driver_version=driver_version,
                cuda_version=cuda_version,
            )
            self._last_result = result
            return result

        except (subprocess.TimeoutExpired, FileNotFoundError, OSError, ValueError) as e:
            logger.warning("Failed to query GPU info: %s", e)
            return self._last_result

    def _query_gpu_stats(self) -> list[GpuInfo]:
        """Query per-GPU statistics."""
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=self._timeout,
            check=True,
        )

        gpus = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 6:
                gpus.append(
                    GpuInfo(
                        index=_safe_int(parts[0]),
                        name=parts[1],
                        memory_used=_safe_int(parts[2]),
                        memory_total=_safe_int(parts[3]),
                        utilization=_safe_int(parts[4]),
                        temperature=_safe_int(parts[5]),
                    )
                )
        return gpus

    def _query_driver_version(self) -> str:
        """Query NVIDIA driver version."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=self._timeout,
            )
            if result.returncode == 0:
                return result.stdout.strip().split("\n")[0]
        except Exception:
            pass
        return "unknown"

    def _query_cuda_version(self) -> str:
        """Query CUDA version from nvidia-smi header."""
        try:
            result = subprocess.run(
                ["nvidia-smi"],
                capture_output=True,
                text=True,
                timeout=self._timeout,
            )
            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if "CUDA Version:" in line:
                        return line.split("CUDA Version:")[1].strip().split()[0]
        except Exception:
            pass
        return "unknown"


# Singleton instance
_gpu_service: GpuInfoService | None = None


def get_gpu_service() -> GpuInfoService:
    """Get the singleton GPU info service."""
    global _gpu_service
    if _gpu_service is None:
        _gpu_service = GpuInfoService()
    return _gpu_service
