"""
Data source and sink registries.

This module provides centralized registration and discovery of data loaders
and exporters, enabling easy extension with new format support.

Usage
-----
>>> from src.infrastructure.registry import (
...     SourceRegistry,
...     DataSinkRegistry,
...     register_defaults,
... )
>>>
>>> # Initialize default sources/sinks
>>> register_defaults()
>>>
>>> # Create a source with validation
>>> source = SourceRegistry.create_validated(
...     "load-ply",
...     {"ply_folder": "/path/to/data"},
...     device="cuda",
... )
>>>
>>> # Get a sink by name
>>> sink_class = DataSinkRegistry.get("ply")
"""

import logging

from .sinks import DataSinkRegistry
from .source_registry import SourceRegistry
from .sources import DataSourceRegistry  # Legacy alias


logger = logging.getLogger(__name__)

_defaults_registered = False


def register_default_sources() -> None:
    """Register all built-in data sources.

    Now registers directly with SourceRegistry (OptimizedPlyModel implements
    BaseGaussianSource protocol directly).
    """
    # Import here to avoid circular imports
    from src.models.ply.interpolated_model import InterpolatedPlyModel
    from src.models.ply.optimized_model import OptimizedPlyModel

    SourceRegistry.register("load-ply", OptimizedPlyModel)
    SourceRegistry.register("interpolated-ply", InterpolatedPlyModel)

    # Also register with legacy DataSourceRegistry for backward compatibility
    # This will be removed in a future version
    try:
        DataSourceRegistry.register("load-ply", OptimizedPlyModel)
        DataSourceRegistry.register("interpolated-ply", InterpolatedPlyModel)
    except Exception:
        pass  # Ignore if already registered


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
    "DataSinkRegistry",
    "DataSourceRegistry",  # Legacy alias
    "SourceRegistry",
    "register_default_sinks",
    "register_default_sources",
    "register_defaults",
]
