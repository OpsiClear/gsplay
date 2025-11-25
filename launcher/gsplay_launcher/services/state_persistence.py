"""JSON-based state persistence for launcher."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from gsplay_launcher.models import LauncherState

logger = logging.getLogger(__name__)


class StatePersistence:
    """Handles state persistence to JSON file with atomic writes.

    Parameters
    ----------
    state_file : Path
        Path to the JSON state file.
    """

    def __init__(self, state_file: Path) -> None:
        self.state_file = state_file
        self._ensure_directory()

    def _ensure_directory(self) -> None:
        """Ensure the parent directory exists."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> LauncherState:
        """Load state from file.

        Returns
        -------
        LauncherState
            Loaded state or new empty state if file doesn't exist.
        """
        if not self.state_file.exists():
            logger.info("No state file found at %s, creating new state", self.state_file)
            return LauncherState()

        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                content = f.read()

            state = LauncherState.from_json(content)
            logger.info(
                "Loaded state with %d instances from %s",
                len(state.instances),
                self.state_file,
            )
            return state

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.error("Failed to load state file: %s", e)
            # Backup corrupted file
            backup = self.state_file.with_suffix(".json.bak")
            if self.state_file.exists():
                self.state_file.rename(backup)
                logger.info("Backed up corrupted state to %s", backup)
            return LauncherState()

    def save(self, state: LauncherState) -> None:
        """Save state to file atomically.

        Parameters
        ----------
        state : LauncherState
            State to save.

        Raises
        ------
        OSError
            If save fails.
        """
        temp_file = self.state_file.with_suffix(".json.tmp")

        try:
            content = state.to_json()

            with open(temp_file, "w", encoding="utf-8") as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())

            # Atomic rename (works on POSIX, best-effort on Windows)
            temp_file.replace(self.state_file)
            logger.debug("Saved state with %d instances", len(state.instances))

        except Exception as e:
            logger.error("Failed to save state: %s", e)
            if temp_file.exists():
                temp_file.unlink()
            raise
