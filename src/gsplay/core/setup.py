"""
Interactive setup wizard for gsplay dependencies.

This module handles explicit dependency installation via `gsplay setup`.
It does NOT auto-install at runtime - that's handled by install.sh/install.ps1.
"""

from __future__ import annotations

import subprocess
import sys

from .dependency_check import (
    check_gsplat,
    check_torch,
    check_triton,
    detect_system_cuda,
    get_torch_index_url,
)


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


def install_package(package: str, extra_args: list[str] | None = None) -> bool:
    """
    Install a package using uv or pip.

    Returns True if installation succeeded.
    """
    args = extra_args or []

    # Try uv first
    try:
        cmd = ["uv", "pip", "install", package] + args
        print(f"  Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=False)
        if result.returncode == 0:
            return True
    except FileNotFoundError:
        pass

    # Fall back to pip
    try:
        cmd = [sys.executable, "-m", "pip", "install", package] + args
        print(f"  Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=False)
        return result.returncode == 0
    except Exception as e:
        print(f"  Error: {e}")
        return False


def run_setup(force: bool = False, yes: bool = False) -> bool:
    """
    Run the interactive setup wizard.

    Args:
        force: Force reinstallation even if dependencies are present
        yes: Skip confirmation prompts (for CI/automated use)

    Returns:
        True if setup completed successfully
    """
    print("GSPlay Setup")
    print("=" * 50)
    print()

    # Detect system CUDA
    cuda_version = detect_system_cuda()
    if cuda_version:
        print(f"[OK] Detected CUDA {cuda_version}")
    else:
        print("⚠ No CUDA detected (nvidia-smi not found)")
        print("  GSPlay requires an NVIDIA GPU with CUDA support.")
        print()
        print("  To fix:")
        print("  1. Install NVIDIA drivers: https://www.nvidia.com/drivers")
        print("  2. Ensure 'nvidia-smi' works in terminal")
        print("  3. Run 'gsplay setup' again")
        return False

    # Check what needs to be installed
    torch_status = check_torch()
    triton_status = check_triton()
    gsplat_status = check_gsplat()

    need_torch = force or not torch_status.available or "CPU only" in torch_status.message
    need_triton = force or not triton_status.available
    need_gsplat = force or not gsplat_status.available

    print()
    print("Current status:")
    if torch_status.available:
        print(f"  - PyTorch: {torch_status.message}")
    else:
        print(f"  - PyTorch: [X] {torch_status.message}")

    if triton_status.available:
        print(f"  - triton: {triton_status.message}")
    else:
        print(f"  - triton: [X] {triton_status.message}")

    if gsplat_status.available:
        print(f"  - gsplat: {gsplat_status.message}")
    else:
        print(f"  - gsplat: [X] {gsplat_status.message}")

    # Nothing to do?
    if not need_torch and not need_triton and not need_gsplat:
        print()
        print("[OK] All dependencies are installed!")
        print("  Run: gsplay <path-to-ply-folder>")
        return True

    # Show installation plan
    print()
    print("Installation plan:")

    index_url = get_torch_index_url(cuda_version)
    if need_torch:
        print(f"  - PyTorch + torchvision (CUDA {cuda_version})")
        print(f"    Index: {index_url}")

    if need_triton:
        pkg = "triton-windows" if sys.platform == "win32" else "triton"
        print(f"  - {pkg}")

    if need_gsplat:
        print("  - gsplat (requires compilation, may take a few minutes)")

    # Confirm
    print()
    if not yes and not prompt_yes_no("Proceed with installation?"):
        print("Setup cancelled.")
        return False

    success = True

    # Install PyTorch
    if need_torch:
        print()
        print("Installing PyTorch...")
        if not install_package("torch", ["--index-url", index_url]):
            print("[X] Failed to install torch")
            success = False
        elif not install_package("torchvision", ["--index-url", index_url]):
            print("[X] Failed to install torchvision")
            success = False
        else:
            print("[OK] PyTorch installed")

    # Install triton
    if need_triton and success:
        print()
        pkg = "triton-windows" if sys.platform == "win32" else "triton"
        print(f"Installing {pkg}...")
        if not install_package(pkg):
            print(f"[X] Failed to install {pkg}")
            # triton is optional, don't fail setup
            print("  (continuing without triton - some optimizations unavailable)")
        else:
            print(f"[OK] {pkg} installed")

    # Install gsplat
    if need_gsplat and success:
        print()
        print("Installing gsplat (this may take a few minutes)...")
        if not install_package("gsplat", ["--no-build-isolation"]):
            print("[X] Failed to install gsplat")
            print()
            print("  Common fixes:")
            print("  - Ensure C++ compiler is installed")
            print("  - On Windows: Install Visual Studio Build Tools")
            print("  - Try: pip install ninja && pip install gsplat --no-build-isolation")
            success = False
        else:
            print("[OK] gsplat installed")

    # Summary
    print()
    print("=" * 50)
    if success:
        print("[OK] Setup complete!")
        print()
        print("  Run: gsplay <path-to-ply-folder>")
    else:
        print("[X] Setup incomplete - some dependencies failed to install")
        print()
        print("  Run 'gsplay doctor' for diagnostics")
        print("  Or try 'gsplay setup' again after fixing the issues above")

    return success
