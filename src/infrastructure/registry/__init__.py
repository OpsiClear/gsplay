"""
Data source and sink registries.

This module provides centralized registration and discovery of data loaders
and exporters, enabling easy extension with new format support.

Usage
-----
>>> from src.infrastructure.registry import (
...     DataSourceRegistry,
...     DataSinkRegistry,
...     register_defaults,
... )
>>>
>>> # Initialize default sources/sinks
>>> register_defaults()
>>>
>>> # Get a source by name
>>> source_class = DataSourceRegistry.get("load-ply")
>>>
>>> # Get a sink by name
>>> sink_class = DataSinkRegistry.get("ply")
"""

import logging

from .sources import DataSourceRegistry
from .sinks import DataSinkRegistry

logger = logging.getLogger(__name__)

_defaults_registered = False


def register_default_sources() -> None:
    """Register all built-in data sources."""
    # Import here to avoid circular imports
    from src.models.ply.ply_source import PlyDataSource

    DataSourceRegistry.register("load-ply", PlyDataSource)


def register_default_sinks() -> None:
    """Register all built-in data sinks."""
    # Import here to avoid circular imports
    from src.infrastructure.exporters.ply_sink import PlySink

    DataSinkRegistry.register("ply", PlySink)

    # Register compressed PLY if available
    try:
        from src.infrastructure.exporters.compressed_ply_sink import CompressedPlySink

        DataSinkRegistry.register("compressed-ply", CompressedPlySink)
    except ImportError:
        logger.debug("CompressedPlySink not available")


def register_defaults() -> None:
    """Register all default sources and sinks.

    Safe to call multiple times - will only register once.
    """
    global _defaults_registered
    if _defaults_registered:
        return

    logger.info("Registering default data sources and sinks")
    register_default_sources()
    register_default_sinks()
    _defaults_registered = True


__all__ = [
    "DataSourceRegistry",
    "DataSinkRegistry",
    "register_defaults",
    "register_default_sources",
    "register_default_sinks",
]
