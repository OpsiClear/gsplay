"""
Bootstrap module for managing PyTorch installation.

This module checks for PyTorch availability and CUDA support on first run,
and offers to install the correct version if needed.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path


# CUDA version to PyTorch index URL mapping
CUDA_TO_INDEX = {
    "12.8": "cu128",
    "12.6": "cu126",
    "12.4": "cu124",
    "12.1": "cu121",
    "11.8": "cu118",
}

# Minimum required versions
TORCH_VERSION = "2.9.0"
TORCHVISION_VERSION = "0.24.0"
GSPLAT_VERSION = "1.0.0"
TRITON_VERSION = "3.0.0"


def get_cache_dir() -> Path:
    """Get the cache directory for gsplay."""
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    cache_dir = base / "gsplay"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def is_torch_setup_complete() -> bool:
    """Check if torch setup has been completed."""
    marker = get_cache_dir() / ".torch_setup_complete"
    return marker.exists()


def mark_torch_setup_complete() -> None:
    """Mark torch setup as complete."""
    marker = get_cache_dir() / ".torch_setup_complete"
    marker.touch()


def check_torch_available() -> tuple[bool, bool, str]:
    """
    Check if PyTorch is available and has CUDA support.

    Returns:
        Tuple of (torch_available, cuda_available, message)
    """
    try:
        import torch

        torch_available = True
        cuda_available = torch.cuda.is_available()

        if cuda_available:
            cuda_version = torch.version.cuda or "unknown"
            device_name = torch.cuda.get_device_name(0)
            msg = f"PyTorch {torch.__version__} with CUDA {cuda_version} ({device_name})"
        else:
            msg = f"PyTorch {torch.__version__} (CPU only)"

        return torch_available, cuda_available, msg

    except ImportError:
        return False, False, "PyTorch is not installed"


def check_gsplat_available() -> tuple[bool, str]:
    """
    Check if gsplat is available.

    Returns:
        Tuple of (gsplat_available, message)
    """
    try:
        import gsplat

        version = getattr(gsplat, "__version__", "unknown")
        return True, f"gsplat {version}"
    except ImportError:
        return False, "gsplat is not installed"


def check_triton_available() -> tuple[bool, str]:
    """
    Check if triton is available.

    Returns:
        Tuple of (triton_available, message)
    """
    try:
        import triton

        version = getattr(triton, "__version__", "unknown")
        return True, f"triton {version}"
    except ImportError:
        return False, "triton is not installed"


def detect_cuda_version() -> str | None:
    """
    Detect the installed CUDA version from nvidia-smi.

    Returns:
        CUDA version string (e.g., "12.8") or None if not detected
    """
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            # Parse "CUDA Version: 12.8" from output
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
    """
    Get the PyTorch index URL for the given CUDA version.

    Args:
        cuda_version: CUDA version string (e.g., "12.8")

    Returns:
        Index URL or None if no matching version found
    """
    # Try exact match first
    if cuda_version in CUDA_TO_INDEX:
        return f"https://download.pytorch.org/whl/{CUDA_TO_INDEX[cuda_version]}"

    # Try major.minor match
    cuda_major_minor = ".".join(cuda_version.split(".")[:2])
    if cuda_major_minor in CUDA_TO_INDEX:
        return f"https://download.pytorch.org/whl/{CUDA_TO_INDEX[cuda_major_minor]}"

    # Find closest matching version (prefer higher compatible version)
    cuda_major = int(cuda_version.split(".")[0])
    cuda_minor = int(cuda_version.split(".")[1]) if "." in cuda_version else 0

    best_match = None
    best_distance = float("inf")

    for cv in CUDA_TO_INDEX:
        cv_major = int(cv.split(".")[0])
        cv_minor = int(cv.split(".")[1])

        # Only consider same major version
        if cv_major == cuda_major:
            distance = abs(cv_minor - cuda_minor)
            if distance < best_distance:
                best_distance = distance
                best_match = cv

    if best_match:
        return f"https://download.pytorch.org/whl/{CUDA_TO_INDEX[best_match]}"

    return None


def install_torch(index_url: str) -> bool:
    """
    Install PyTorch from the given index URL.

    Args:
        index_url: PyTorch wheel index URL

    Returns:
        True if installation succeeded, False otherwise
    """
    packages = [
        f"torch>={TORCH_VERSION}",
        f"torchvision>={TORCHVISION_VERSION}",
    ]

    print(f"\nInstalling PyTorch from {index_url}...")
    print(f"Packages: {', '.join(packages)}\n")

    # Try uv first (faster), fall back to pip
    try:
        cmd = ["uv", "pip", "install"] + packages + ["--index-url", index_url]
        result = subprocess.run(cmd, check=False)
        if result.returncode == 0:
            return True
    except FileNotFoundError:
        pass

    # Fall back to pip
    try:
        cmd = [sys.executable, "-m", "pip", "install"] + packages + ["--index-url", index_url]
        result = subprocess.run(cmd, check=False)
        return result.returncode == 0
    except Exception as e:
        print(f"Installation failed: {e}")
        return False


def install_gsplat() -> bool:
    """
    Install gsplat with --no-build-isolation flag.

    gsplat requires compilation and needs torch to be installed first.

    Returns:
        True if installation succeeded, False otherwise
    """
    package = f"gsplat>={GSPLAT_VERSION}"

    print("\nInstalling gsplat (this may take a few minutes)...")
    print(f"Package: {package}\n")

    # Try uv first (faster), fall back to pip
    # gsplat needs --no-build-isolation for proper compilation
    try:
        cmd = ["uv", "pip", "install", package, "--no-build-isolation"]
        result = subprocess.run(cmd, check=False)
        if result.returncode == 0:
            return True
    except FileNotFoundError:
        pass

    # Fall back to pip
    try:
        cmd = [sys.executable, "-m", "pip", "install", package, "--no-build-isolation"]
        result = subprocess.run(cmd, check=False)
        return result.returncode == 0
    except Exception as e:
        print(f"Installation failed: {e}")
        return False


def install_triton() -> bool:
    """
    Install triton (or triton-windows on Windows).

    Returns:
        True if installation succeeded, False otherwise
    """
    # Use platform-specific package name
    if sys.platform == "win32":
        package = f"triton-windows>={TRITON_VERSION}"
    else:
        package = f"triton>={TRITON_VERSION}"

    print(f"\nInstalling {package.split('>=')[0]}...")
    print(f"Package: {package}\n")

    # Try uv first (faster), fall back to pip
    try:
        cmd = ["uv", "pip", "install", package]
        result = subprocess.run(cmd, check=False)
        if result.returncode == 0:
            return True
    except FileNotFoundError:
        pass

    # Fall back to pip
    try:
        cmd = [sys.executable, "-m", "pip", "install", package]
        result = subprocess.run(cmd, check=False)
        return result.returncode == 0
    except Exception as e:
        print(f"Installation failed: {e}")
        return False


def prompt_yes_no(prompt: str, default: bool = True) -> bool:
    """Prompt the user for yes/no input."""
    suffix = " [Y/n] " if default else " [y/N] "
    try:
        response = input(prompt + suffix).strip().lower()
        if not response:
            return default
        return response in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        print()
        return False


def is_ephemeral_environment() -> bool:
    """Check if we're running in an ephemeral environment (like uvx)."""
    # Check for uvx-style temp directory patterns
    venv_path = sys.prefix
    temp_patterns = [
        "/.cache/uv/",  # Linux/macOS uvx
        "\\AppData\\Local\\uv\\",  # Windows uvx
        "/tmp/",
        "\\Temp\\",
    ]
    return any(pattern in venv_path for pattern in temp_patterns)


def ensure_dependencies() -> bool:
    """
    Ensure PyTorch with CUDA, triton, and gsplat are available.

    This function checks for PyTorch, triton, and gsplat availability.
    If they are missing or PyTorch doesn't have CUDA support,
    it offers to install the correct versions.

    Returns:
        True if all dependencies are available, False otherwise
    """
    # Check current state
    torch_available, cuda_available, torch_msg = check_torch_available()
    triton_available, triton_msg = check_triton_available()
    gsplat_available, gsplat_msg = check_gsplat_available()

    # If everything is available, we're good
    if torch_available and cuda_available and triton_available and gsplat_available:
        if not is_torch_setup_complete():
            print(f"✓ {torch_msg}")
            print(f"✓ {triton_msg}")
            print(f"✓ {gsplat_msg}")
            mark_torch_setup_complete()
        return True

    # If setup was already completed but something isn't available now
    if is_torch_setup_complete():
        if torch_available and gsplat_available:
            if not cuda_available:
                print(f"⚠ {torch_msg}")
                print("  CUDA was available during setup but isn't now.")
                print("  Continuing with CPU mode...\n")
            return True
        # Something was uninstalled? Need to reinstall
        print("⚠ Dependencies were installed but are now missing. Re-running setup...\n")

    # Detect system CUDA
    print("GSPlay - First-time setup")
    print("=" * 40)

    cuda_version = detect_cuda_version()

    if cuda_version is None:
        print("\n⚠ No NVIDIA GPU or CUDA detected.")
        print("  GSPlay requires CUDA for GPU-accelerated rendering.")
        print("\n  To use GSPlay:")
        print("  1. Install NVIDIA drivers: https://www.nvidia.com/drivers")
        print("  2. Ensure nvidia-smi works")
        print("  3. Run gsplay again\n")

        if torch_available:
            print(f"  Current: {torch_msg}")
            print("  You can continue with CPU mode (very slow).\n")
            if prompt_yes_no("Continue with CPU mode?", default=False):
                mark_torch_setup_complete()
                return True
        return False

    print(f"\n✓ Detected CUDA {cuda_version}")

    # Get appropriate PyTorch index
    index_url = get_torch_index_url(cuda_version)
    if index_url is None:
        print(f"\n⚠ No PyTorch build available for CUDA {cuda_version}")
        print("  Supported CUDA versions:", ", ".join(sorted(CUDA_TO_INDEX.keys())))
        return False

    # Check if we're in an ephemeral environment
    if is_ephemeral_environment():
        print("\n⚠ Running in ephemeral environment (uvx/temp venv)")
        print("  Installation won't persist between runs.")
        print("\n  For persistent installation, use:")
        print("    uv pip install gsplay")
        print(f"    uv pip install torch torchvision --index-url {index_url}")
        print("    uv pip install triton  # or triton-windows on Windows")
        print("    uv pip install gsplat --no-build-isolation")
        print()
        if not prompt_yes_no("Install anyway for this session?", default=True):
            return False

    # Determine what needs to be installed
    need_torch = not torch_available or not cuda_available
    need_triton = not triton_available
    need_gsplat = not gsplat_available

    print("\nInstallation plan:")
    if need_torch:
        if torch_available:
            print(f"  • PyTorch: reinstall with CUDA {cuda_version} (current: {torch_msg})")
        else:
            print(f"  • PyTorch: install with CUDA {cuda_version}")
    else:
        print(f"  • PyTorch: already installed ({torch_msg})")

    if need_triton:
        triton_pkg = "triton-windows" if sys.platform == "win32" else "triton"
        print(f"  • {triton_pkg}: install (GPU kernel acceleration)")
    else:
        print(f"  • triton: already installed ({triton_msg})")

    if need_gsplat:
        print("  • gsplat: install (requires compilation)")
    else:
        print(f"  • gsplat: already installed ({gsplat_msg})")

    if not need_torch and not need_triton and not need_gsplat:
        mark_torch_setup_complete()
        return True

    print(f"\n  PyTorch index: {index_url}")

    # Prompt for installation
    if not prompt_yes_no("\nProceed with installation?", default=True):
        if torch_available and gsplat_available:
            print("\nContinuing with current installation...")
            mark_torch_setup_complete()
            return True
        return False

    # Install PyTorch if needed
    if need_torch:
        if not install_torch(index_url):
            print("\n✗ PyTorch installation failed.")
            print("  Try installing manually:")
            print(f"    uv pip install torch torchvision --index-url {index_url}")
            return False

        print("\n✓ PyTorch installed successfully!")

        # Verify installation - need to reload torch module
        for mod in list(sys.modules.keys()):
            if mod.startswith("torch"):
                del sys.modules[mod]

        torch_available, cuda_available, torch_msg = check_torch_available()
        if cuda_available:
            print(f"✓ {torch_msg}")
        else:
            print(f"⚠ {torch_msg}")
            print("  CUDA still not available after installation.")
            print("  This may indicate a driver issue.")

    # Install triton if needed
    if need_triton:
        if not install_triton():
            triton_pkg = "triton-windows" if sys.platform == "win32" else "triton"
            print(f"\n✗ {triton_pkg} installation failed.")
            print("  Try installing manually:")
            print(f"    uv pip install {triton_pkg}")
            return False

        print("\n✓ triton installed successfully!")

        # Verify installation
        if "triton" in sys.modules:
            del sys.modules["triton"]

        triton_available, triton_msg = check_triton_available()
        if triton_available:
            print(f"✓ {triton_msg}")

    # Install gsplat if needed
    if need_gsplat:
        if not install_gsplat():
            print("\n✗ gsplat installation failed.")
            print("  Try installing manually:")
            print("    uv pip install gsplat --no-build-isolation")
            return False

        print("\n✓ gsplat installed successfully!")

        # Verify installation
        if "gsplat" in sys.modules:
            del sys.modules["gsplat"]

        gsplat_available, gsplat_msg = check_gsplat_available()
        if gsplat_available:
            print(f"✓ {gsplat_msg}")
        else:
            print("⚠ gsplat import failed after installation")

    print("\n" + "=" * 40)
    print("Setup complete!")
    print("=" * 40 + "\n")

    mark_torch_setup_complete()
    return True


# Keep old name for backwards compatibility
ensure_torch = ensure_dependencies


def reset_torch_setup() -> None:
    """Reset the torch setup state (for testing or re-running setup)."""
    marker = get_cache_dir() / ".torch_setup_complete"
    if marker.exists():
        marker.unlink()
        print("Torch setup state reset. Run gsplay to re-run setup.")
