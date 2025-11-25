"""
Registry for data sink (exporter) implementations.

Provides centralized registration and discovery of exporters,
enabling easy extension with new format support.
"""

from __future__ import annotations

import logging
from typing import Type

from src.domain.interfaces import DataSinkProtocol, DataSinkMetadata

logger = logging.getLogger(__name__)


class DataSinkRegistry:
    """Registry for data sink (exporter) implementations.

    Provides:
    - Registration of exporters by name
    - Lookup by name
    - Metadata discovery for UI population

    Example
    -------
    >>> # Register a new sink
    >>> DataSinkRegistry.register("npz", NpzSink)
    >>>
    >>> # Create sink by name
    >>> sink_class = DataSinkRegistry.get("npz")
    >>> sink = sink_class()
    >>> sink.export(gaussian_data, "/path/to/output.npz")
    >>>
    >>> # List available formats for UI
    >>> formats = DataSinkRegistry.list_all()
    >>> for meta in formats:
    ...     print(f"{meta.name}: {meta.description}")
    """

    _sinks: dict[str, Type[DataSinkProtocol]] = {}

    @classmethod
    def register(cls, name: str, sink_class: Type[DataSinkProtocol]) -> None:
        """Register a data sink implementation.

        Parameters
        ----------
        name : str
            Sink identifier (e.g., "ply", "npz", "splat")
        sink_class : Type[DataSinkProtocol]
            Class implementing DataSinkProtocol
        """
        cls._sinks[name] = sink_class
        try:
            meta = sink_class.metadata()
            logger.info(
                "Registered data sink: %s (%s) - extension: %s",
                name,
                meta.description,
                meta.file_extension,
            )
        except Exception as e:
            logger.warning(
                "Registered data sink %s but metadata() failed: %s",
                name,
                e,
            )

    @classmethod
    def get(cls, name: str) -> Type[DataSinkProtocol] | None:
        """Get a registered sink by name.

        Parameters
        ----------
        name : str
            Sink identifier

        Returns
        -------
        Type[DataSinkProtocol] | None
            Sink class or None if not found
        """
        return cls._sinks.get(name)

    @classmethod
    def list_all(cls) -> list[DataSinkMetadata]:
        """List all registered sinks with metadata.

        Returns
        -------
        list[DataSinkMetadata]
            Metadata for all registered sinks
        """
        result = []
        for sink in cls._sinks.values():
            try:
                result.append(sink.metadata())
            except Exception:
                pass
        return result

    @classmethod
    def names(cls) -> list[str]:
        """Get list of registered sink names.

        Returns
        -------
        list[str]
            Sorted list of sink identifiers
        """
        return sorted(cls._sinks.keys())

    @classmethod
    def find_by_extension(cls, extension: str) -> Type[DataSinkProtocol] | None:
        """Find a sink that outputs files with the given extension.

        Parameters
        ----------
        extension : str
            File extension (e.g., ".ply", ".npz")

        Returns
        -------
        Type[DataSinkProtocol] | None
            First matching sink class, or None
        """
        # Normalize extension
        if not extension.startswith("."):
            extension = f".{extension}"

        for sink in cls._sinks.values():
            try:
                meta = sink.metadata()
                if meta.file_extension == extension:
                    return sink
            except Exception:
                continue
        return None

    @classmethod
    def clear(cls) -> None:
        """Clear all registered sinks (for testing)."""
        cls._sinks.clear()
