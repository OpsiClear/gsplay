"""
Format-aware PLY loader registry.

Encapsulates Gaussian PLY format detection and dispatches to the correct
loader strategy (compressed GPU path vs. uncompressed CPU path). New loader
strategies (e.g., streaming readers) can be registered without modifying
the caller, satisfying SOLID and Clean Architecture boundaries.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import gsply
from gsply.formats import detect_format

from src.domain.entities import GSData, GSTensor
from src.infrastructure.io.path_io import UniversalPath
from src.infrastructure.processing.ply.loader import load_ply_as_gsdata


logger = logging.getLogger(__name__)


class PlyFrameEncoding(str, Enum):
    """Supported Gaussian PLY encodings."""

    COMPRESSED = "compressed"
    UNCOMPRESSED = "uncompressed"
    UNKNOWN = "unknown"


@dataclass
class PlyLoadResult:
    """Container returned by loader strategies."""

    encoding: PlyFrameEncoding
    sh_degree: int | None
    gstensor: GSTensor | None
    gsdata: GSData | None = None


class PlyLoaderError(RuntimeError):
    """Raised when a loader strategy cannot satisfy the request."""

    def __init__(self, message: str, *, recoverable: bool = False) -> None:
        super().__init__(message)
        self.recoverable = recoverable


class PlyLoaderStrategy(Protocol):
    """Strategy protocol for Gaussian PLY loaders."""

    def load(self, file_path: UniversalPath, *, device: str) -> PlyLoadResult: ...


class CompressedPlyLoader:
    """GPU-accelerated loader for PlayCanvas compressed PLY files."""

    def load(self, file_path: UniversalPath, *, device: str) -> PlyLoadResult:
        try:
            gstensor = gsply.plyread_gpu(str(file_path), device=device)
        except ValueError as exc:
            raise PlyLoaderError(
                f"Compressed PLY GPU load failed for {file_path}",
                recoverable=True,
            ) from exc
        return PlyLoadResult(
            encoding=PlyFrameEncoding.COMPRESSED,
            sh_degree=None,
            gstensor=gstensor,
            gsdata=None,
        )


class UncompressedPlyLoader:
    """CPU loader for standard Gaussian PLY files."""

    def load(self, file_path: UniversalPath, *, device: str) -> PlyLoadResult:
        gsdata = load_ply_as_gsdata(file_path)
        gstensor = gsply.GSTensor.from_gsdata(gsdata, device=device)
        return PlyLoadResult(
            encoding=PlyFrameEncoding.UNCOMPRESSED,
            sh_degree=None,
            gstensor=gstensor,
            gsdata=gsdata,
        )


class FormatAwarePlyLoader:
    """
    Detects Gaussian PLY encoding and dispatches to the appropriate loader strategy.

    Allows registering new strategies per format to support future data sources
    (e.g., streaming, remote decompression) without changing model code.
    """

    def __init__(self, *, device: str = "cuda") -> None:
        self.device = device
        self._strategies: dict[PlyFrameEncoding, PlyLoaderStrategy] = {
            PlyFrameEncoding.COMPRESSED: CompressedPlyLoader(),
            PlyFrameEncoding.UNCOMPRESSED: UncompressedPlyLoader(),
            PlyFrameEncoding.UNKNOWN: UncompressedPlyLoader(),
        }
        self._format_cache: dict[str, tuple[PlyFrameEncoding, int | None]] = {}
        # Folder-level default format (bypasses per-file detection when set)
        self._default_format: tuple[PlyFrameEncoding, int | None] | None = None

    def register_strategy(
        self,
        encoding: PlyFrameEncoding,
        strategy: PlyLoaderStrategy,
    ) -> None:
        """Register/override a loader strategy for a specific encoding."""
        self._strategies[encoding] = strategy

    def set_device(self, device: str) -> None:
        """Update the target device for downstream loaders."""
        self.device = device

    def set_default_format(self, encoding: PlyFrameEncoding, sh_degree: int | None) -> None:
        """Set folder-level default format to bypass per-file detection.

        Call this after summarize_folder() confirms a homogeneous sequence.
        Saves ~0.1ms per frame by skipping file header inspection.
        """
        self._default_format = (encoding, sh_degree)
        logger.debug(
            "[FormatAwarePlyLoader] Default format set: %s (sh=%s)",
            encoding.value,
            sh_degree,
        )

    def clear_default_format(self) -> None:
        """Clear the folder-level default format."""
        self._default_format = None

    def detect_format(self, file_path: UniversalPath) -> tuple[PlyFrameEncoding, int | None]:
        """Detect encoding + SH degree for a single PLY file.

        Uses folder-level default if set (fastest), then per-file cache,
        then actual file inspection (slowest).
        """
        # Fast path: use folder-level default if set
        if self._default_format is not None:
            return self._default_format

        # Check per-file cache
        cache_key = str(file_path)
        if cache_key in self._format_cache:
            return self._format_cache[cache_key]

        # Inspect file header (slowest path)
        encoding = PlyFrameEncoding.UNKNOWN
        sh_degree: int | None = None
        try:
            is_compressed, sh_degree = detect_format(str(file_path))
            if is_compressed:
                encoding = PlyFrameEncoding.COMPRESSED
            elif sh_degree is not None:
                encoding = PlyFrameEncoding.UNCOMPRESSED
        except Exception as exc:  # pragma: no cover - detection best effort
            logger.debug(
                "[FormatAwarePlyLoader] Failed to detect format for %s: %s",
                file_path,
                exc,
            )

        self._format_cache[cache_key] = (encoding, sh_degree)
        return encoding, sh_degree

    def load_for_gpu(self, file_path: UniversalPath) -> PlyLoadResult:
        """
        Load a PLY file ready for GPU processing.

        Compressed files are routed to the GPU loader; uncompressed fall back to
        the CPU loader + tensor conversion. Recoverable loader errors automatically
        downgrade the cached encoding to avoid repeated failures.
        """
        encoding, sh_degree = self.detect_format(file_path)
        strategy = self._strategies.get(encoding, self._strategies[PlyFrameEncoding.UNKNOWN])

        try:
            result = strategy.load(file_path, device=self.device)
        except PlyLoaderError as exc:
            if exc.recoverable and encoding == PlyFrameEncoding.COMPRESSED:
                logger.warning(
                    "[FormatAwarePlyLoader] %s. Downgrading %s to uncompressed path.",
                    exc,
                    file_path,
                )
                encoding = PlyFrameEncoding.UNCOMPRESSED
                self._format_cache[str(file_path)] = (encoding, sh_degree)
                strategy = self._strategies[PlyFrameEncoding.UNCOMPRESSED]
                result = strategy.load(file_path, device=self.device)
            else:
                raise

        if result.sh_degree is None:
            result.sh_degree = sh_degree
        if result.encoding == PlyFrameEncoding.UNKNOWN:
            result.encoding = encoding
        return result

    def summarize_folder(
        self,
        ply_files: Sequence[UniversalPath] | Sequence[str],
    ) -> tuple[PlyFrameEncoding, int | None] | None:
        """
        Inspect a subset of files to produce a folder-level hint.

        Returns the first decisive encoding encountered or None if everything
        remains unknown (e.g., unreadable paths).
        """
        for raw_path in ply_files:
            path = raw_path if isinstance(raw_path, UniversalPath) else UniversalPath(raw_path)
            encoding, sh_degree = self.detect_format(path)
            if encoding != PlyFrameEncoding.UNKNOWN:
                return encoding, sh_degree
        return None


__all__ = [
    "FormatAwarePlyLoader",
    "PlyFrameEncoding",
    "PlyLoadResult",
    "PlyLoaderStrategy",
]
