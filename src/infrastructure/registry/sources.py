"""
Registry for data source implementations.

Provides centralized registration and discovery of data loaders,
enabling easy extension with new format support.
"""

from __future__ import annotations

import logging
from typing import Type

from src.domain.interfaces import DataSourceProtocol, DataSourceMetadata

logger = logging.getLogger(__name__)


class DataSourceRegistry:
    """Registry for data source implementations.

    Provides:
    - Registration of data sources by name
    - Lookup by name or path auto-detection
    - Metadata discovery for UI population

    Example
    -------
    >>> # Register a new source
    >>> DataSourceRegistry.register("splat", SplatDataSource)
    >>>
    >>> # Create source by name
    >>> source_class = DataSourceRegistry.get("splat")
    >>> source = source_class(config)
    >>>
    >>> # Auto-detect source from path
    >>> source_class = DataSourceRegistry.find_for_path("/path/to/file.splat")
    """

    _sources: dict[str, Type[DataSourceProtocol]] = {}

    @classmethod
    def register(cls, name: str, source_class: Type[DataSourceProtocol]) -> None:
        """Register a data source implementation.

        Parameters
        ----------
        name : str
            Source identifier (e.g., "load-ply", "splat")
        source_class : Type[DataSourceProtocol]
            Class implementing DataSourceProtocol
        """
        cls._sources[name] = source_class
        try:
            meta = source_class.metadata()
            logger.info(
                "Registered data source: %s (%s) - extensions: %s",
                name,
                meta.description,
                meta.file_extensions,
            )
        except Exception as e:
            logger.warning(
                "Registered data source %s but metadata() failed: %s",
                name,
                e,
            )

    @classmethod
    def get(cls, name: str) -> Type[DataSourceProtocol] | None:
        """Get a registered source by name.

        Parameters
        ----------
        name : str
            Source identifier

        Returns
        -------
        Type[DataSourceProtocol] | None
            Source class or None if not found
        """
        return cls._sources.get(name)

    @classmethod
    def list_all(cls) -> list[DataSourceMetadata]:
        """List all registered sources with metadata.

        Returns
        -------
        list[DataSourceMetadata]
            Metadata for all registered sources
        """
        result = []
        for name, source in cls._sources.items():
            try:
                result.append(source.metadata())
            except Exception as e:
                logger.debug(f"Failed to get metadata for source '{name}': {e}")
        return result

    @classmethod
    def find_for_path(cls, path: str) -> Type[DataSourceProtocol] | None:
        """Find a source that can load the given path.

        Checks each registered source's can_load() method.

        Parameters
        ----------
        path : str
            File or directory path

        Returns
        -------
        Type[DataSourceProtocol] | None
            First matching source class, or None
        """
        for name, source in cls._sources.items():
            try:
                if source.can_load(path):
                    return source
            except Exception as e:
                logger.debug(f"Source '{name}' can_load check failed for '{path}': {e}")
                continue
        return None

    @classmethod
    def names(cls) -> list[str]:
        """Get list of registered source names.

        Returns
        -------
        list[str]
            Sorted list of source identifiers
        """
        return sorted(cls._sources.keys())

    @classmethod
    def clear(cls) -> None:
        """Clear all registered sources (for testing)."""
        cls._sources.clear()
