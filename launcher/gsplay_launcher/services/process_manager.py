"""Process management for GSPlay instances using psutil."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import psutil

from gsplay_launcher.models import GSPlayInstance


logger = logging.getLogger(__name__)


# =============================================================================
# Windows MSVC Environment Detection
# =============================================================================

_VCVARS_CACHE: dict | None = None


def _is_msvc_available() -> bool:
    """Check if MSVC compiler (cl.exe) is available in the current environment."""
    return shutil.which("cl") is not None


def _find_vcvars64() -> Path | None:
    """Find vcvars64.bat for Visual Studio.

    Searches common installation paths for VS 2022, 2019, and 2017.

    Returns
    -------
    Path | None
        Path to vcvars64.bat or None if not found.
    """
    # Common Visual Studio installation paths
    # Note: BuildTools can be in either Program Files or Program Files (x86)
    vs_paths = [
        # VS 2022 (Program Files)
        Path(
            r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
        ),
        Path(
            r"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
        ),
        Path(
            r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
        ),
        Path(
            r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
        ),
        # VS 2022 (Program Files x86) - BuildTools sometimes installs here
        Path(
            r"C:\Program Files (x86)\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
        ),
        Path(
            r"C:\Program Files (x86)\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
        ),
        Path(
            r"C:\Program Files (x86)\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
        ),
        Path(
            r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
        ),
        # VS 2019
        Path(
            r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
        ),
        Path(
            r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Auxiliary\Build\vcvars64.bat"
        ),
        Path(
            r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
        ),
        Path(
            r"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
        ),
        # VS 2017
        Path(
            r"C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvars64.bat"
        ),
        Path(
            r"C:\Program Files (x86)\Microsoft Visual Studio\2017\Professional\VC\Auxiliary\Build\vcvars64.bat"
        ),
        Path(
            r"C:\Program Files (x86)\Microsoft Visual Studio\2017\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
        ),
        Path(
            r"C:\Program Files (x86)\Microsoft Visual Studio\2017\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
        ),
    ]

    # Also check VSINSTALLDIR environment variable
    vs_install_dir = os.environ.get("VSINSTALLDIR")
    if vs_install_dir:
        custom_path = Path(vs_install_dir) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
        vs_paths.insert(0, custom_path)

    for path in vs_paths:
        if path.exists():
            logger.info("Found vcvars64.bat at: %s", path)
            return path

    return None


def _get_vcvars_env() -> dict | None:
    """Get environment variables after running vcvars64.bat.

    This runs vcvars64.bat in a subprocess and captures the resulting
    environment, which can then be used for launching processes that
    need MSVC. The MSVC environment is MERGED with the current environment
    to preserve paths needed for uv, Python, etc.

    Returns
    -------
    dict | None
        Environment dict with MSVC paths merged in, or None if unavailable.
    """
    global _VCVARS_CACHE

    if _VCVARS_CACHE is not None:
        return _VCVARS_CACHE

    vcvars_path = _find_vcvars64()
    if vcvars_path is None:
        logger.warning("Could not find vcvars64.bat - MSVC environment unavailable")
        return None

    try:
        # Run vcvars64.bat and then echo the environment
        # We use 'set' to dump all environment variables after vcvars runs
        cmd = f'cmd /c ""{vcvars_path}" x64 && set"'
        logger.info("Loading MSVC environment from: %s", vcvars_path)

        result = subprocess.run(
            cmd,
            check=False,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0:
            logger.error("vcvars64.bat failed: %s", result.stderr)
            return None

        # Parse the environment variables from vcvars output
        vcvars_env = {}
        for line in result.stdout.splitlines():
            if "=" in line:
                key, _, value = line.partition("=")
                vcvars_env[key] = value

        # Start with the CURRENT environment (preserves uv, Python, etc.)
        merged_env = os.environ.copy()

        # Merge in the MSVC environment, with special handling for PATH
        # We PREPEND the MSVC paths to ensure cl.exe is found
        for key, value in vcvars_env.items():
            if key.upper() == "PATH":
                # Prepend MSVC paths to existing PATH
                current_path = merged_env.get("PATH", merged_env.get("Path", ""))
                # Get the new paths from vcvars (compare with original)
                original_path = os.environ.get("PATH", os.environ.get("Path", ""))
                new_paths = [p for p in value.split(";") if p and p not in original_path]
                if new_paths:
                    merged_env["PATH"] = ";".join(new_paths) + ";" + current_path
                    logger.debug("Added %d MSVC paths to PATH", len(new_paths))
            elif key.upper() not in [k.upper() for k in merged_env]:
                # Add new variables that don't exist in current env
                merged_env[key] = value
            elif key in ("INCLUDE", "LIB", "LIBPATH"):
                # For compiler include/lib paths, prepend the new values
                current = merged_env.get(key, "")
                if current:
                    merged_env[key] = value + ";" + current
                else:
                    merged_env[key] = value

        _VCVARS_CACHE = merged_env
        logger.info("MSVC environment loaded successfully")
        return merged_env

    except subprocess.TimeoutExpired:
        logger.error("Timeout running vcvars64.bat")
        return None
    except Exception as e:
        logger.error("Error loading MSVC environment: %s", e)
        return None


def get_msvc_status() -> dict:
    """Get the current MSVC availability status.

    Returns
    -------
    dict
        Status dict with keys:
        - available: bool - whether MSVC is usable
        - in_path: bool - whether cl.exe is already in PATH
        - vcvars_path: str | None - path to vcvars64.bat if found
        - message: str - human-readable status message
    """
    in_path = _is_msvc_available()
    vcvars = _find_vcvars64()

    if in_path:
        return {
            "available": True,
            "in_path": True,
            "vcvars_path": None,
            "message": "MSVC compiler (cl.exe) is available in PATH",
        }
    elif vcvars:
        return {
            "available": True,
            "in_path": False,
            "vcvars_path": str(vcvars),
            "message": f"MSVC available via: {vcvars}",
        }
    else:
        return {
            "available": False,
            "in_path": False,
            "vcvars_path": None,
            "message": "MSVC not found. Install Visual Studio Build Tools for CUDA JIT compilation.",
        }


def stop_process(pid: int, force: bool = False, timeout: float = 10.0) -> bool:
    """Stop a process by PID.

    This is the unified process termination function used by both
    ProcessManager and the cleanup utilities.

    Parameters
    ----------
    pid : int
        Process ID to stop.
    force : bool
        If True, force kill immediately without graceful shutdown.
    timeout : float
        Timeout for graceful shutdown before force kill.

    Returns
    -------
    bool
        True if process was stopped successfully.
    """
    try:
        proc = psutil.Process(pid)

        if force:
            # Force kill immediately
            logger.info("Force killing process %d", pid)
            _force_kill_process(proc)
            return True

        # Graceful termination
        logger.info("Terminating process %d", pid)
        proc.terminate()

        try:
            proc.wait(timeout=timeout)
            logger.info("Process %d terminated gracefully", pid)
            return True
        except psutil.TimeoutExpired:
            logger.warning("Process %d did not terminate, force killing", pid)
            _force_kill_process(proc)
            return True

    except psutil.NoSuchProcess:
        logger.debug("Process %d already dead", pid)
        return True
    except psutil.AccessDenied:
        logger.error("Access denied for process %d", pid)
        return False
    except Exception as e:
        logger.error("Failed to stop process %d: %s", pid, e)
        return False


def _force_kill_process(proc: psutil.Process) -> None:
    """Force kill process and all children.

    Parameters
    ----------
    proc : psutil.Process
        Process to kill.
    """
    try:
        # Kill children first
        children = proc.children(recursive=True)
        for child in children:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass

        # Kill main process
        proc.kill()
        proc.wait(timeout=5.0)
        logger.info("Process %d force killed", proc.pid)

    except psutil.NoSuchProcess:
        pass
    except psutil.TimeoutExpired:
        logger.error("Process %d could not be killed", proc.pid)


class ProcessManager:
    """Manages gsplay process lifecycle.

    Parameters
    ----------
    gsplay_script : Path
        Path to the gsplay main.py script.
    python_cmd : str
        Python command to use (e.g., "uv").
    stop_timeout : float
        Timeout in seconds for graceful shutdown.
    """

    def __init__(
        self,
        gsplay_script: Path,
        python_cmd: str = "uv",
        stop_timeout: float = 10.0,
    ) -> None:
        self.gsplay_script = gsplay_script
        self.python_cmd = python_cmd
        self.stop_timeout = stop_timeout
        self._msvc_env: dict | None = None
        self._msvc_checked = False

    def _ensure_msvc_env(self) -> dict | None:
        """Ensure MSVC environment is loaded on Windows if needed.

        Returns
        -------
        dict | None
            Environment dict with MSVC paths, or None to use default env.
        """
        if sys.platform != "win32":
            return None

        if self._msvc_checked:
            return self._msvc_env

        self._msvc_checked = True

        # Check if MSVC is already available in current environment
        if _is_msvc_available():
            logger.info("MSVC compiler (cl.exe) already available in PATH")
            return None

        # Try to load MSVC environment from vcvars64.bat
        logger.info("MSVC not in PATH, attempting to load Developer Command Prompt environment...")
        self._msvc_env = _get_vcvars_env()

        if self._msvc_env:
            logger.info("Will launch gsplay with MSVC environment (x64 Native Tools)")
        else:
            logger.warning(
                "MSVC environment not available. gsplat JIT compilation may fail. "
                "Install Visual Studio Build Tools or run from Developer Command Prompt."
            )

        return self._msvc_env

    def start(self, instance: GSPlayInstance) -> int:
        """Start a gsplay process.

        On Windows, this will automatically detect if MSVC is available
        and load the Developer Command Prompt environment if needed for
        gsplat CUDA JIT compilation.

        Parameters
        ----------
        instance : GSPlayInstance
            Instance configuration.

        Returns
        -------
        int
            Process ID of started gsplay.

        Raises
        ------
        ProcessStartError
            If process fails to start.
        """
        # Use the installed 'gsplay' command instead of script path
        cmd = [
            self.python_cmd,
            "run",
            "gsplay",
            *instance.to_cli_args(),
        ]

        logger.info("Starting gsplay: %s", " ".join(cmd))

        # Create log file for gsplay output
        log_dir = Path.cwd() / "data" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"gsplay_{instance.port}.log"

        # Platform-specific process isolation flags
        stdout_file = open(log_file, "w")
        kwargs: dict = {
            "stdout": stdout_file,
            "stderr": subprocess.STDOUT,
            "stdin": subprocess.DEVNULL,
        }

        if sys.platform == "win32":
            # Windows: create new process group and detach
            kwargs["creationflags"] = (
                subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
            )

            # Load MSVC environment if needed for CUDA JIT compilation
            msvc_env = self._ensure_msvc_env()
            if msvc_env:
                kwargs["env"] = msvc_env
                logger.info("Launching with MSVC Developer Command Prompt environment")
        else:
            # Unix: start new session
            kwargs["start_new_session"] = True

        try:
            process = subprocess.Popen(cmd, **kwargs)
            pid = process.pid
            logger.info("Started gsplay process with PID %d", pid)
            return pid
        except Exception as e:
            logger.error("Failed to start gsplay: %s", e)
            raise ProcessStartError(str(e)) from e

    def stop(self, pid: int, force: bool = False) -> bool:
        """Stop a process gracefully, with force kill fallback.

        Parameters
        ----------
        pid : int
            Process ID to stop.
        force : bool
            If True, force kill immediately without graceful shutdown.

        Returns
        -------
        bool
            True if process was stopped.
        """
        return stop_process(pid, force=force, timeout=self.stop_timeout)

    def is_running(self, pid: int) -> bool:
        """Check if a process is running.

        Parameters
        ----------
        pid : int
            Process ID to check.

        Returns
        -------
        bool
            True if process is running.
        """
        try:
            proc = psutil.Process(pid)
            return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            return False

    def get_process_info(self, pid: int) -> dict | None:
        """Get information about a process.

        Parameters
        ----------
        pid : int
            Process ID.

        Returns
        -------
        dict | None
            Process info or None if not found.
        """
        try:
            proc = psutil.Process(pid)
            return {
                "pid": proc.pid,
                "status": proc.status(),
                "cpu_percent": proc.cpu_percent(),
                "memory_mb": proc.memory_info().rss / (1024 * 1024),
                "cmdline": proc.cmdline(),
            }
        except psutil.NoSuchProcess:
            return None


class ProcessStartError(Exception):
    """Raised when a process fails to start."""

    pass
