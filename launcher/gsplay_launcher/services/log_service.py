"""Log service for reading and streaming instance logs."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path


logger = logging.getLogger(__name__)


@dataclass
class LogChunk:
    """A chunk of log lines with position info."""

    lines: list[str]
    total_lines: int
    offset: int
    has_more: bool


class LogService:
    """Service for reading and streaming log files.

    Parameters
    ----------
    log_dir : Path
        Directory where log files are stored.
    """

    def __init__(self, log_dir: Path | None = None) -> None:
        self.log_dir = log_dir or (Path.cwd() / "data" / "logs")

    def get_log_path(self, port: int) -> Path:
        """Get log file path for an instance by port.

        Parameters
        ----------
        port : int
            Instance port number.

        Returns
        -------
        Path
            Path to log file.
        """
        return self.log_dir / f"gsplay_{port}.log"

    def read_logs(
        self,
        port: int,
        lines: int = 100,
        offset: int = 0,
    ) -> LogChunk:
        """Read log lines from an instance log file.

        Parameters
        ----------
        port : int
            Instance port number.
        lines : int
            Number of lines to return (from the end if offset=0).
        offset : int
            Line offset from the end (0 = most recent).

        Returns
        -------
        LogChunk
            Log lines with metadata.
        """
        log_path = self.get_log_path(port)

        if not log_path.exists():
            return LogChunk(lines=[], total_lines=0, offset=0, has_more=False)

        try:
            all_lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            total = len(all_lines)

            if total == 0:
                return LogChunk(lines=[], total_lines=0, offset=0, has_more=False)

            # Calculate slice indices (from the end)
            # offset=0 means get the last `lines` lines
            end_idx = total - offset
            start_idx = max(0, end_idx - lines)

            if end_idx <= 0:
                return LogChunk(lines=[], total_lines=total, offset=offset, has_more=False)

            result_lines = all_lines[start_idx:end_idx]
            has_more = start_idx > 0

            return LogChunk(
                lines=result_lines,
                total_lines=total,
                offset=offset,
                has_more=has_more,
            )

        except Exception as e:
            logger.error("Failed to read log file %s: %s", log_path, e)
            return LogChunk(
                lines=[f"Error reading log: {e}"], total_lines=0, offset=0, has_more=False
            )

    async def stream_logs(
        self,
        port: int,
        poll_interval: float = 0.5,
    ):
        """Stream log lines as they are written.

        Parameters
        ----------
        port : int
            Instance port number.
        poll_interval : float
            How often to check for new lines (seconds).

        Yields
        ------
        str
            New log lines as they appear.
        """
        log_path = self.get_log_path(port)
        last_position = 0

        # If file exists, start from current end
        if log_path.exists():
            try:
                content = log_path.read_text(encoding="utf-8", errors="replace")
                last_position = len(content)
                content.count("\n")
            except Exception:
                pass

        while True:
            try:
                if not log_path.exists():
                    await asyncio.sleep(poll_interval)
                    continue

                # Read file content
                content = log_path.read_text(encoding="utf-8", errors="replace")
                current_length = len(content)

                # Check if file was truncated (reset)
                if current_length < last_position:
                    last_position = 0

                # Get new content
                if current_length > last_position:
                    new_content = content[last_position:]
                    new_lines = new_content.splitlines()

                    for line in new_lines:
                        if line:  # Skip empty lines
                            yield line

                    last_position = current_length

                await asyncio.sleep(poll_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error streaming log: %s", e)
                await asyncio.sleep(poll_interval)


# Global service instance
_log_service: LogService | None = None


def get_log_service() -> LogService:
    """Get or create the global log service."""
    global _log_service
    if _log_service is None:
        _log_service = LogService()
    return _log_service
