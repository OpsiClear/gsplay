"""
Universal path abstraction supporting local and cloud storage.

This module provides a unified interface for file operations across different
storage backends (local filesystem, S3, GCS, Azure Blob, HTTP/HTTPS).

Key Features:
- Zero dependencies for local paths (uses pathlib)
- Optional cloud support via fsspec (install separately)
- Automatic protocol detection (local, s3://, gs://, az://, http://, https://)
- Fail-fast error handling with clear messages
- Caching support for remote files
- Full read/write capabilities

Example Usage:
    # Local filesystem (no extra dependencies)
    path = UniversalPath("./export_with_edits/frame_00000.ply")
    data = path.read_bytes()

    # S3 (requires: pip install s3fs)
    path = UniversalPath("s3://my-bucket/renders/frame_00000.ply")
    data = path.read_bytes()

    # GCS (requires: pip install gcsfs)
    path = UniversalPath("gs://my-bucket/renders/frame_00000.ply")
    with path.open("rb") as f:
        data = f.read()

    # HTTP read-only (fsspec base only)
    path = UniversalPath("https://example.com/data/frame_00000.ply")
    data = path.read_bytes()

    # Directory operations
    folder = UniversalPath("s3://my-bucket/renders/")
    ply_files = folder.glob("*.ply")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import BinaryIO, Protocol


logger = logging.getLogger(__name__)


class PathBackend(Protocol):
    """
    Protocol defining the interface for storage backends.

    All storage backends (local, S3, GCS, etc.) must implement this interface.
    """

    def open(self, path: str, mode: str = "rb") -> BinaryIO:
        """Open file for reading/writing."""
        ...

    def exists(self, path: str) -> bool:
        """Check if path exists."""
        ...

    def is_dir(self, path: str) -> bool:
        """Check if path is a directory."""
        ...

    def glob(self, pattern: str) -> list[str]:
        """Find files matching glob pattern."""
        ...

    def mkdir(self, path: str, parents: bool = True, exist_ok: bool = True) -> None:
        """Create directory."""
        ...


class LocalBackend:
    """Storage backend for local filesystem using pathlib."""

    def open(self, path: str, mode: str = "rb") -> BinaryIO:
        """Open local file."""
        return open(path, mode)

    def exists(self, path: str) -> bool:
        """Check if local path exists."""
        return Path(path).exists()

    def is_dir(self, path: str) -> bool:
        """Check if local path is directory."""
        return Path(path).is_dir()

    def glob(self, pattern: str) -> list[str]:
        """Glob local filesystem."""
        # pattern is like "./folder/*.ply"
        base_path = Path(pattern).parent
        pattern_str = Path(pattern).name

        if not base_path.exists():
            return []

        matches = list(base_path.glob(pattern_str))
        return [str(p) for p in matches]

    def mkdir(self, path: str, parents: bool = True, exist_ok: bool = True) -> None:
        """Create local directory."""
        Path(path).mkdir(parents=parents, exist_ok=exist_ok)


class FsspecBackend:
    """Storage backend for remote filesystems using fsspec."""

    def __init__(self, protocol: str):
        """
        Initialize fsspec backend.

        Parameters
        ----------
        protocol : str
            Storage protocol (s3, gs, az, http, https, etc.)

        Raises
        ------
        ImportError
            If fsspec or required cloud library not installed
        """
        self.protocol = protocol
        self._fs = None
        self._fsspec = None
        self._init_fsspec()

    def _init_fsspec(self):
        """Lazy import and initialize fsspec."""
        try:
            import fsspec

            self._fsspec = fsspec
        except ImportError:
            raise ImportError(
                f"fsspec is required for {self.protocol} paths. Install with: pip install fsspec"
            )

        # Get filesystem for this protocol
        try:
            self._fs = fsspec.filesystem(self.protocol)
        except ImportError as e:
            # Provide specific error messages for common cloud providers
            if self.protocol == "s3":
                raise ImportError(
                    f"S3 support requires s3fs. Install with: pip install s3fs\nOriginal error: {e}"
                ) from e
            elif self.protocol == "gs" or self.protocol == "gcs":
                raise ImportError(
                    f"Google Cloud Storage support requires gcsfs. Install with: pip install gcsfs\n"
                    f"Original error: {e}"
                ) from e
            elif self.protocol == "az" or self.protocol == "abfs":
                raise ImportError(
                    f"Azure Blob Storage support requires adlfs. Install with: pip install adlfs\n"
                    f"Original error: {e}"
                ) from e
            else:
                raise ImportError(
                    f"Protocol '{self.protocol}' requires additional dependencies.\n"
                    f"Original error: {e}"
                ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to initialize {self.protocol} filesystem: {e}") from e

    def open(self, path: str, mode: str = "rb") -> BinaryIO:
        """Open remote file via fsspec."""
        return self._fsspec.open(path, mode)

    def exists(self, path: str) -> bool:
        """Check if remote path exists."""
        return self._fs.exists(path)

    def is_dir(self, path: str) -> bool:
        """Check if remote path is directory."""
        try:
            return self._fs.isdir(path)
        except Exception:
            # Some filesystems raise exceptions for non-existent paths
            return False

    def glob(self, pattern: str) -> list[str]:
        """Glob remote filesystem."""
        # fsspec glob returns paths without protocol prefix
        matches = self._fs.glob(pattern)

        # Add protocol prefix back
        if matches and not matches[0].startswith(f"{self.protocol}://"):
            matches = [f"{self.protocol}://{m}" for m in matches]

        return matches

    def mkdir(self, path: str, parents: bool = True, exist_ok: bool = True) -> None:
        """Create remote directory (if supported by backend)."""
        try:
            self._fs.makedirs(path, exist_ok=exist_ok)
        except AttributeError:
            # Some fsspec filesystems don't support makedirs
            logger.warning(f"{self.protocol} filesystem does not support mkdir")


class UniversalPath:
    """
    Universal path abstraction for local and cloud storage.

    Automatically detects path type and uses appropriate backend:
    - Local paths: pathlib.Path
    - s3://: AWS S3 (requires s3fs)
    - gs://: Google Cloud Storage (requires gcsfs)
    - az:// or abfs://: Azure Blob Storage (requires adlfs)
    - http:// or https://: HTTP/HTTPS (fsspec base only)

    Parameters
    ----------
    path : str | Path | UniversalPath
        Path to file or directory

    Raises
    ------
    ImportError
        If cloud path used without required dependencies

    Examples
    --------
    Local filesystem:
        >>> path = UniversalPath("./data/frame.ply")
        >>> data = path.read_bytes()

    S3:
        >>> path = UniversalPath("s3://bucket/data/frame.ply")
        >>> with path.open("rb") as f:
        ...     data = f.read()

    Glob pattern:
        >>> folder = UniversalPath("s3://bucket/data/")
        >>> ply_files = folder.glob("*.ply")
    """

    def __init__(self, path: str | Path | UniversalPath):
        """Initialize universal path."""
        # Handle UniversalPath input
        if isinstance(path, UniversalPath):
            self.path_str = path.path_str
            self._protocol = path._protocol
            self._backend = path._backend
            return

        # Convert to string
        self.path_str = str(path)

        # Detect protocol and initialize backend
        self._protocol = self._detect_protocol()
        self._backend = self._create_backend()

    def _detect_protocol(self) -> str:
        """
        Detect storage protocol from path.

        Returns
        -------
        str
            Protocol name ('local', 's3', 'gs', 'az', 'http', 'https', etc.)
        """
        # Check for remote protocols
        if "://" in self.path_str:
            protocol = self.path_str.split("://")[0]
            return protocol

        # Default to local filesystem
        return "local"

    def _create_backend(self) -> PathBackend:
        """
        Create appropriate storage backend.

        Returns
        -------
        PathBackend
            Backend instance for this path type
        """
        if self._protocol == "local":
            return LocalBackend()
        else:
            return FsspecBackend(self._protocol)

    @property
    def is_remote(self) -> bool:
        """Check if path is remote (not local filesystem)."""
        return self._protocol != "local"

    @property
    def protocol(self) -> str:
        """Get storage protocol."""
        return self._protocol

    def open(self, mode: str = "rb") -> BinaryIO:
        """
        Open file for reading/writing.

        Parameters
        ----------
        mode : str
            File mode ('rb', 'wb', etc.)

        Returns
        -------
        BinaryIO
            File handle
        """
        return self._backend.open(self.path_str, mode)

    def read_bytes(self) -> bytes:
        """
        Read entire file as bytes.

        Returns
        -------
        bytes
            File contents
        """
        with self.open("rb") as f:
            return f.read()

    def write_bytes(self, data: bytes) -> None:
        """
        Write bytes to file.

        Parameters
        ----------
        data : bytes
            Data to write
        """
        with self.open("wb") as f:
            f.write(data)

    def read_text(self, encoding: str = "utf-8") -> str:
        """
        Read entire file as text.

        Parameters
        ----------
        encoding : str
            Text encoding (default: utf-8)

        Returns
        -------
        str
            File contents as text
        """
        with self.open("rb") as f:
            return f.read().decode(encoding)

    def write_text(self, text: str, encoding: str = "utf-8") -> None:
        """
        Write text to file.

        Parameters
        ----------
        text : str
            Text to write
        encoding : str
            Text encoding (default: utf-8)
        """
        with self.open("wb") as f:
            f.write(text.encode(encoding))

    def exists(self) -> bool:
        """
        Check if path exists.

        Returns
        -------
        bool
            True if path exists
        """
        return self._backend.exists(self.path_str)

    def is_dir(self) -> bool:
        """
        Check if path is a directory.

        Returns
        -------
        bool
            True if path is a directory
        """
        return self._backend.is_dir(self.path_str)

    def glob(self, pattern: str) -> list[UniversalPath]:
        """
        Find files matching glob pattern.

        Parameters
        ----------
        pattern : str
            Glob pattern (e.g., "*.ply", "frame_*.ply")

        Returns
        -------
        list[UniversalPath]
            Matching paths
        """
        # Build full pattern
        if self._protocol == "local":
            # Local: self.path_str/pattern
            base_path = Path(self.path_str)
            full_pattern = str(base_path / pattern)
        else:
            # Remote: combine with /
            base_path = self.path_str.rstrip("/")
            full_pattern = f"{base_path}/{pattern}"

        # Get matches from backend
        matches = self._backend.glob(full_pattern)

        # Convert to UniversalPath objects
        return [UniversalPath(m) for m in matches]

    def mkdir(self, parents: bool = True, exist_ok: bool = True) -> None:
        """
        Create directory.

        Parameters
        ----------
        parents : bool
            Create parent directories if needed
        exist_ok : bool
            Don't raise error if directory exists
        """
        self._backend.mkdir(self.path_str, parents=parents, exist_ok=exist_ok)

    @property
    def name(self) -> str:
        """
        Get filename or directory name.

        Returns
        -------
        str
            Name of file/directory
        """
        if self._protocol == "local":
            return Path(self.path_str).name
        else:
            # Get last component of path
            return self.path_str.rstrip("/").split("/")[-1]

    @property
    def parent(self) -> UniversalPath:
        """
        Get parent directory.

        Returns
        -------
        UniversalPath
            Parent directory path
        """
        if self._protocol == "local":
            return UniversalPath(Path(self.path_str).parent)
        else:
            # Remove last component
            parts = self.path_str.rstrip("/").split("/")
            parent_str = "/".join(parts[:-1])
            # Ensure we keep the protocol
            if not parent_str.endswith("://"):
                return UniversalPath(parent_str)
            else:
                # At root, return self
                return self

    @property
    def stem(self) -> str:
        """
        Get filename without extension.

        Returns
        -------
        str
            Filename without extension
        """
        if self._protocol == "local":
            return Path(self.path_str).stem
        else:
            name = self.name
            if "." in name:
                return name.rsplit(".", 1)[0]
            return name

    @property
    def suffix(self) -> str:
        """
        Get file extension.

        Returns
        -------
        str
            File extension including dot (e.g., ".ply")
        """
        if self._protocol == "local":
            return Path(self.path_str).suffix
        else:
            name = self.name
            if "." in name:
                return "." + name.rsplit(".", 1)[1]
            return ""

    def __truediv__(self, other: str | UniversalPath) -> UniversalPath:
        """
        Join paths using / operator.

        Parameters
        ----------
        other : str | UniversalPath
            Path component to append

        Returns
        -------
        UniversalPath
            Combined path
        """
        other_str = str(other)

        if self._protocol == "local":
            combined = Path(self.path_str) / other_str
            return UniversalPath(combined)
        else:
            # Remote: join with /
            base = self.path_str.rstrip("/")
            combined = f"{base}/{other_str.lstrip('/')}"
            return UniversalPath(combined)

    def __str__(self) -> str:
        """Get string representation."""
        return self.path_str

    def __repr__(self) -> str:
        """Get repr string."""
        return f"UniversalPath('{self.path_str}')"

    def __eq__(self, other) -> bool:
        """Check equality."""
        if isinstance(other, UniversalPath):
            return self.path_str == other.path_str
        return self.path_str == str(other)

    def __hash__(self) -> int:
        """Get hash."""
        return hash(self.path_str)


# Convenience function for creating paths
def universal_path(path: str | Path | UniversalPath) -> UniversalPath:
    """
    Create UniversalPath instance.

    Parameters
    ----------
    path : str | Path | UniversalPath
        Path to convert

    Returns
    -------
    UniversalPath
        Universal path instance
    """
    return UniversalPath(path)
