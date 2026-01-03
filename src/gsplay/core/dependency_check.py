"""
Fast dependency checking without installation logic.

This module provides lightweight checks for required dependencies.
No heavy imports (torch, etc.) at module level - all checks are lazy.
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass, field


@dataclass
class DependencyStatus:
    """Status of a single dependency."""

    name: str
    available: bool
    version: str | None = None
    message: str = ""
    fix_command: str | None = None


@dataclass
class EnvironmentCheck:
    """Complete environment check result."""

    python_ok: bool = True
    torch_available: bool = False
    torch_version: str | None = None
    cuda_available: bool = False
    cuda_version: str | None = None
    gpu_name: str | None = None
    gpu_memory: str | None = None
    gsplat_available: bool = False
    gsplat_version: str | None = None
    triton_available: bool = False
    triton_version: str | None = None
    system_cuda_version: str | None = None
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_ready(self) -> bool:
        """Check if environment is ready for gsplay."""
        return (
            self.python_ok
            and self.torch_available
            and self.cuda_available
            and self.gsplat_available
        )

    def get_install_instructions(self) -> str:
        """Get installation instructions for missing dependencies."""
        lines = []

        if not self.torch_available or not self.cuda_available:
            cuda = self.system_cuda_version or "128"
            cuda_short = cuda.replace(".", "")[:3]
            lines.append("# Install PyTorch with CUDA support:")
            lines.append(
                f"uv pip install torch torchvision "
                f"--index-url https://download.pytorch.org/whl/cu{cuda_short}"
            )
            lines.append("")

        if not self.gsplat_available:
            lines.append("# Install gsplat (requires compilation):")
            lines.append("uv pip install gsplat --no-build-isolation")
            lines.append("")

        if not self.triton_available:
            pkg = "triton-windows" if sys.platform == "win32" else "triton"
            lines.append(f"# Install {pkg}:")
            lines.append(f"uv pip install {pkg}")

        return "\n".join(lines)


# =============================================================================
# CUDA Detection (from nvidia-smi / nvcc)
# =============================================================================

CUDA_TO_INDEX = {
    "12.8": "cu128",
    "12.6": "cu126",
    "12.4": "cu124",
    "12.1": "cu121",
    "11.8": "cu118",
}


def detect_system_cuda() -> str | None:
    """
    Detect CUDA version from nvidia-smi or nvcc.

    Returns:
        CUDA version string (e.g., "12.8") or None if not detected
    """
    # Try nvidia-smi first
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            match = re.search(r"CUDA Version:\s*(\d+\.\d+)", result.stdout)
            if match:
                return match.group(1)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass

    # Try nvcc as fallback
    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            match = re.search(r"release (\d+\.\d+)", result.stdout)
            if match:
                return match.group(1)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass

    return None


def get_torch_index_url(cuda_version: str) -> str | None:
    """Get PyTorch index URL for the given CUDA version."""
    # Try exact match
    if cuda_version in CUDA_TO_INDEX:
        return f"https://download.pytorch.org/whl/{CUDA_TO_INDEX[cuda_version]}"

    # Try major.minor match
    cuda_mm = ".".join(cuda_version.split(".")[:2])
    if cuda_mm in CUDA_TO_INDEX:
        return f"https://download.pytorch.org/whl/{CUDA_TO_INDEX[cuda_mm]}"

    # Find closest match
    try:
        cuda_major = int(cuda_version.split(".")[0])
        cuda_minor = int(cuda_version.split(".")[1]) if "." in cuda_version else 0

        best_match = None
        best_distance = float("inf")

        for cv in CUDA_TO_INDEX:
            cv_major = int(cv.split(".")[0])
            cv_minor = int(cv.split(".")[1])
            if cv_major == cuda_major:
                distance = abs(cv_minor - cuda_minor)
                if distance < best_distance:
                    best_distance = distance
                    best_match = cv

        if best_match:
            return f"https://download.pytorch.org/whl/{CUDA_TO_INDEX[best_match]}"
    except (ValueError, IndexError):
        pass

    return None


# =============================================================================
# Dependency Checks (Lazy - no imports at module level)
# =============================================================================


def check_python() -> DependencyStatus:
    """Check Python version."""
    major, minor, patch = sys.version_info[:3]
    version = f"{major}.{minor}.{patch}"

    if major < 3 or (major == 3 and minor < 12):
        return DependencyStatus(
            name="Python",
            available=False,
            version=version,
            message=f"Python {version} is not supported (requires 3.12+)",
            fix_command="Install Python 3.12+",
        )

    return DependencyStatus(
        name="Python",
        available=True,
        version=version,
        message=f"Python {version}",
    )


def check_torch() -> DependencyStatus:
    """Check PyTorch installation and CUDA availability."""
    try:
        import torch

        version = torch.__version__
        cuda_available = torch.cuda.is_available()

        if cuda_available:
            cuda_version = torch.version.cuda or "unknown"
            device_name = torch.cuda.get_device_name(0)
            return DependencyStatus(
                name="PyTorch",
                available=True,
                version=f"{version} (CUDA {cuda_version})",
                message=f"PyTorch {version} with CUDA {cuda_version} on {device_name}",
            )
        else:
            return DependencyStatus(
                name="PyTorch",
                available=True,
                version=version,
                message=f"PyTorch {version} (CPU only - CUDA not available)",
                fix_command="Reinstall PyTorch with CUDA support",
            )

    except ImportError:
        cuda = detect_system_cuda()
        if cuda:
            index_url = get_torch_index_url(cuda)
            fix = f"uv pip install torch torchvision --index-url {index_url}"
        else:
            fix = "Install NVIDIA drivers, then run install.sh"

        return DependencyStatus(
            name="PyTorch",
            available=False,
            message="PyTorch is not installed",
            fix_command=fix,
        )


def check_gsplat() -> DependencyStatus:
    """Check gsplat installation."""
    try:
        import gsplat

        version = getattr(gsplat, "__version__", "unknown")
        return DependencyStatus(
            name="gsplat",
            available=True,
            version=version,
            message=f"gsplat {version}",
        )
    except ImportError:
        return DependencyStatus(
            name="gsplat",
            available=False,
            message="gsplat is not installed",
            fix_command="uv pip install gsplat --no-build-isolation",
        )


def check_triton() -> DependencyStatus:
    """Check triton installation."""
    try:
        import triton

        version = getattr(triton, "__version__", "unknown")
        return DependencyStatus(
            name="triton",
            available=True,
            version=version,
            message=f"triton {version}",
        )
    except ImportError:
        pkg = "triton-windows" if sys.platform == "win32" else "triton"
        return DependencyStatus(
            name="triton",
            available=False,
            message=f"{pkg} is not installed",
            fix_command=f"uv pip install {pkg}",
        )


def check_viser() -> DependencyStatus:
    """Check viser installation."""
    try:
        import viser

        version = getattr(viser, "__version__", "unknown")
        return DependencyStatus(
            name="viser",
            available=True,
            version=version,
            message=f"viser {version}",
        )
    except ImportError:
        return DependencyStatus(
            name="viser",
            available=False,
            message="viser is not installed",
            fix_command="uv pip install viser",
        )


# =============================================================================
# Full Environment Check
# =============================================================================


def check_environment() -> EnvironmentCheck:
    """
    Perform a complete environment check.

    Returns an EnvironmentCheck with all dependency statuses.
    """
    result = EnvironmentCheck()

    # Python version
    py = check_python()
    result.python_ok = py.available
    if not py.available:
        result.issues.append(py.message)

    # System CUDA
    result.system_cuda_version = detect_system_cuda()

    # PyTorch and CUDA
    try:
        import torch

        result.torch_available = True
        result.torch_version = torch.__version__
        result.cuda_available = torch.cuda.is_available()

        if result.cuda_available:
            result.cuda_version = torch.version.cuda
            result.gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            result.gpu_memory = f"{props.total_memory / 1e9:.1f} GB"
        else:
            result.issues.append("PyTorch installed but CUDA not available")

    except ImportError:
        result.issues.append("PyTorch is not installed")

    # gsplat
    gsplat_status = check_gsplat()
    result.gsplat_available = gsplat_status.available
    result.gsplat_version = gsplat_status.version
    if not gsplat_status.available:
        result.issues.append("gsplat is not installed")

    # triton (warning only, not required)
    triton_status = check_triton()
    result.triton_available = triton_status.available
    result.triton_version = triton_status.version
    if not triton_status.available:
        result.warnings.append("triton not installed (some GPU optimizations unavailable)")

    return result


def check_environment_fast() -> tuple[bool, list[str]]:
    """
    Quick environment check - returns (is_ready, issues).

    This is the fast path for run_view() to check before importing heavy modules.
    """
    issues = []

    # Check torch + CUDA
    try:
        import torch

        if not torch.cuda.is_available():
            issues.append("CUDA not available")
    except ImportError:
        issues.append("PyTorch not installed")

    # Check gsplat
    try:
        import gsplat  # noqa: F401
    except ImportError:
        issues.append("gsplat not installed")

    return len(issues) == 0, issues
