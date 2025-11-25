"""Process management for GSPlay instances using psutil."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

import psutil

from gsplay_launcher.models import GSPlayInstance

logger = logging.getLogger(__name__)


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

    def start(self, instance: GSPlayInstance) -> int:
        """Start a gsplay process.

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

    def stop(self, pid: int) -> bool:
        """Stop a process gracefully, with force kill fallback.

        Parameters
        ----------
        pid : int
            Process ID to stop.

        Returns
        -------
        bool
            True if process was stopped.
        """
        try:
            proc = psutil.Process(pid)

            # Try graceful termination first
            logger.info("Terminating process %d", pid)
            proc.terminate()

            try:
                proc.wait(timeout=self.stop_timeout)
                logger.info("Process %d terminated gracefully", pid)
                return True
            except psutil.TimeoutExpired:
                # Force kill
                logger.warning("Process %d did not terminate, force killing", pid)
                self._force_kill(proc)
                return True

        except psutil.NoSuchProcess:
            logger.debug("Process %d already dead", pid)
            return True
        except Exception as e:
            logger.error("Failed to stop process %d: %s", pid, e)
            return False

    def _force_kill(self, proc: psutil.Process) -> None:
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
