"""
PLY discovery helpers scoped to the infrastructure I/O layer.

Provides a single entry point for enumerating and sorting frames across local
and remote storage by leveraging the UniversalPath abstraction.
"""

from __future__ import annotations

from pathlib import Path

from src.infrastructure.io.path_io import UniversalPath
from src.shared.math import natural_sort_key


def discover_and_sort_ply_files(ply_folder: str | Path | UniversalPath) -> list[UniversalPath]:
    """
    Discover and numerically sort PLY files from a folder or remote bucket.

    Supports local filesystem, S3, GCS, Azure, and HTTP/HTTPS paths.
    """
    ply_folder_path = UniversalPath(ply_folder)

    if not ply_folder_path.exists():
        raise FileNotFoundError(f"PLY folder not found: {ply_folder_path}")

    if not ply_folder_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {ply_folder_path}")

    ply_files = ply_folder_path.glob("*.ply")
    if not ply_files:
        raise FileNotFoundError(f"No .ply files found in: {ply_folder_path}")

    return sorted(ply_files, key=lambda p: natural_sort_key(p.name))

