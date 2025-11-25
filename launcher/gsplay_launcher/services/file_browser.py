"""File browser service for discovering PLY folders."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


class PathSecurityError(Exception):
    """Raised when path traversal is detected."""

    def __init__(self, path: str) -> None:
        self.path = path
        super().__init__(f"Path traversal detected: {path}")


class PathNotFoundError(Exception):
    """Raised when path does not exist."""

    def __init__(self, path: str) -> None:
        self.path = path
        super().__init__(f"Path not found: {path}")


@dataclass
class DirectoryEntry:
    """Represents a directory entry with PLY metadata."""

    name: str
    path: str  # Relative path from browse root
    is_directory: bool
    is_ply_folder: bool = False
    ply_count: int = 0
    total_size_mb: float = 0.0
    modified_at: str | None = None


@dataclass
class BrowseResult:
    """Result of browsing a directory."""

    current_path: str
    breadcrumbs: list[dict[str, str]]  # [{name, path}, ...]
    entries: list[DirectoryEntry]


class FileBrowserService:
    """Service for browsing directories and detecting PLY folders.

    Parameters
    ----------
    root : Path
        Root directory for file browsing.
    """

    def __init__(self, root: Path) -> None:
        self._root = root.resolve()
        logger.info("FileBrowserService initialized with root: %s", self._root)

    @property
    def root(self) -> Path:
        """Get the root directory."""
        return self._root

    def _validate_path(self, relative_path: str) -> Path:
        """Validate and resolve a relative path.

        Parameters
        ----------
        relative_path : str
            Path relative to the root.

        Returns
        -------
        Path
            Resolved absolute path.

        Raises
        ------
        PathSecurityError
            If path traversal is detected.
        PathNotFoundError
            If path does not exist.
        """
        # Handle empty path as root
        if not relative_path or relative_path in (".", "/", "\\"):
            return self._root

        # Normalize and resolve
        target = (self._root / relative_path).resolve()

        # Security check: must be within root
        try:
            target.relative_to(self._root)
        except ValueError:
            raise PathSecurityError(relative_path)

        if not target.exists():
            raise PathNotFoundError(relative_path)

        return target

    def _detect_ply_folder(self, directory: Path) -> tuple[bool, int, float]:
        """Detect if a directory contains PLY files.

        Parameters
        ----------
        directory : Path
            Directory to scan.

        Returns
        -------
        tuple[bool, int, float]
            (is_ply_folder, ply_count, total_size_mb)
        """
        try:
            ply_files = list(directory.glob("*.ply"))
            if not ply_files:
                return False, 0, 0.0

            total_size = sum(f.stat().st_size for f in ply_files)
            total_size_mb = total_size / (1024 * 1024)
            return True, len(ply_files), round(total_size_mb, 2)
        except PermissionError:
            return False, 0, 0.0
        except Exception as e:
            logger.warning("Error scanning directory %s: %s", directory, e)
            return False, 0, 0.0

    def _build_breadcrumbs(self, relative_path: str) -> list[dict[str, str]]:
        """Build breadcrumb navigation trail.

        Parameters
        ----------
        relative_path : str
            Current relative path.

        Returns
        -------
        list[dict[str, str]]
            List of {name, path} dictionaries.
        """
        breadcrumbs = [{"name": "Root", "path": ""}]

        if not relative_path or relative_path in (".", "/", "\\"):
            return breadcrumbs

        # Normalize path separators
        normalized = relative_path.replace("\\", "/").strip("/")
        parts = normalized.split("/")

        cumulative_path = ""
        for part in parts:
            if part:
                cumulative_path = f"{cumulative_path}/{part}" if cumulative_path else part
                breadcrumbs.append({"name": part, "path": cumulative_path})

        return breadcrumbs

    def _get_modified_time(self, path: Path) -> str | None:
        """Get ISO formatted modified time."""
        try:
            mtime = path.stat().st_mtime
            dt = datetime.fromtimestamp(mtime, tz=timezone.utc)
            return dt.isoformat()
        except Exception:
            return None

    def browse(self, relative_path: str = "") -> BrowseResult:
        """Browse a directory and return its contents.

        Parameters
        ----------
        relative_path : str
            Path relative to the root.

        Returns
        -------
        BrowseResult
            Directory contents with metadata.

        Raises
        ------
        PathSecurityError
            If path traversal is detected.
        PathNotFoundError
            If path does not exist.
        """
        target = self._validate_path(relative_path)

        if not target.is_dir():
            raise PathNotFoundError(relative_path)

        entries: list[DirectoryEntry] = []

        try:
            for item in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
                # Skip hidden files
                if item.name.startswith("."):
                    continue

                # Get relative path from root
                try:
                    item_relative = str(item.relative_to(self._root)).replace("\\", "/")
                except ValueError:
                    continue

                if item.is_dir():
                    is_ply, ply_count, total_size = self._detect_ply_folder(item)
                    entries.append(
                        DirectoryEntry(
                            name=item.name,
                            path=item_relative,
                            is_directory=True,
                            is_ply_folder=is_ply,
                            ply_count=ply_count,
                            total_size_mb=total_size,
                            modified_at=self._get_modified_time(item),
                        )
                    )
                else:
                    # Include files but mark them as non-directory
                    entries.append(
                        DirectoryEntry(
                            name=item.name,
                            path=item_relative,
                            is_directory=False,
                            modified_at=self._get_modified_time(item),
                        )
                    )

        except PermissionError:
            logger.warning("Permission denied accessing %s", target)

        # Normalize relative_path for response
        normalized_path = relative_path.replace("\\", "/").strip("/") if relative_path else ""

        return BrowseResult(
            current_path=normalized_path,
            breadcrumbs=self._build_breadcrumbs(normalized_path),
            entries=entries,
        )

    def get_absolute_path(self, relative_path: str) -> Path:
        """Get absolute path for launching.

        Parameters
        ----------
        relative_path : str
            Path relative to the root.

        Returns
        -------
        Path
            Absolute path.

        Raises
        ------
        PathSecurityError
            If path traversal is detected.
        PathNotFoundError
            If path does not exist.
        """
        return self._validate_path(relative_path)
