"""Unified registry for Gaussian data sources.

This module provides a centralized registry that combines:
- Source registration and lookup
- Configuration validation
- Entry point discovery
- Lifecycle management integration

Replaces the previous DataSourceRegistry + ModelFactory._builders approach.
"""

from __future__ import annotations

import logging
from importlib.metadata import entry_points
from typing import Any

from src.domain.interfaces import BaseGaussianSource, SourceMetadata
from src.infrastructure.validation.config_validator import ConfigValidator
from src.shared.exceptions import ConfigValidationError, PluginInitError


logger = logging.getLogger(__name__)


class SourceRegistry:
    """Unified registry for Gaussian data sources.

    Provides:
    - Registration of sources by name
    - Validated source creation with config checking
    - Auto-discovery from entry_points
    - Path-based source auto-detection

    Example
    -------
    >>> # Register a source
    >>> SourceRegistry.register("my-format", MyFormatSource)
    >>>
    >>> # Create with validation
    >>> source = SourceRegistry.create_validated(
    ...     "my-format",
    ...     {"path": "/data/scene"},
    ...     device="cuda",
    ... )
    >>>
    >>> # Auto-detect from path
    >>> source_class = SourceRegistry.find_for_path("/data/scene.ply")
    """

    _sources: dict[str, type[BaseGaussianSource]] = {}
    _entry_points_loaded: bool = False

    @classmethod
    def register(
        cls,
        name: str,
        source_class: type[BaseGaussianSource],
        *,
        validate: bool = True,
    ) -> None:
        """Register a data source implementation.

        Parameters
        ----------
        name : str
            Source identifier (e.g., "load-ply", "splat")
        source_class : Type[BaseGaussianSource]
            Class implementing BaseGaussianSource protocol
        validate : bool
            Whether to validate the class implements required methods
        """
        if validate:
            cls._validate_source_class(source_class, name)

        cls._sources[name] = source_class

        try:
            meta = source_class.metadata()
            logger.info(
                "Registered source: %s (%s) - extensions: %s",
                name,
                meta.description,
                meta.file_extensions,
            )
        except Exception as e:
            logger.warning(
                "Registered source %s but metadata() failed: %s",
                name,
                e,
            )

    @classmethod
    def _validate_source_class(cls, source_class: type, name: str) -> None:
        """Validate that a class implements BaseGaussianSource."""
        required_methods = ["metadata", "can_load", "total_frames", "get_frame_at_time"]
        missing = []

        for method in required_methods:
            if not hasattr(source_class, method):
                missing.append(method)

        if missing:
            raise TypeError(
                f"Source '{name}' missing required methods: {missing}. "
                f"Must implement BaseGaussianSource protocol."
            )

    @classmethod
    def get(cls, name: str) -> type[BaseGaussianSource] | None:
        """Get a registered source by name.

        Parameters
        ----------
        name : str
            Source identifier

        Returns
        -------
        Type[BaseGaussianSource] | None
            Source class or None if not found
        """
        # Ensure entry points are loaded
        cls._load_entry_points()
        return cls._sources.get(name)

    @classmethod
    def create_validated(
        cls,
        name: str,
        config: dict[str, Any],
        device: str = "cuda",
        **kwargs: Any,
    ) -> BaseGaussianSource:
        """Create a source instance with validated configuration.

        Parameters
        ----------
        name : str
            Source identifier
        config : dict[str, Any]
            Configuration dictionary
        device : str
            Target device for GPU operations
        **kwargs : Any
            Additional arguments passed to source constructor

        Returns
        -------
        BaseGaussianSource
            Initialized source instance

        Raises
        ------
        ValueError
            If source name is not registered
        ConfigValidationError
            If configuration is invalid
        PluginInitError
            If source initialization fails
        """
        source_class = cls.get(name)
        if source_class is None:
            available = cls.names()
            raise ValueError(f"Unknown source type: '{name}'. Available: {', '.join(available)}")

        # Validate config against schema if available
        try:
            meta = source_class.metadata()
            if meta.config_schema is not None:
                validated_config = ConfigValidator.validate_or_raise(
                    config,
                    meta.config_schema,
                    plugin_name=name,
                )
                config = validated_config
        except ConfigValidationError:
            raise
        except Exception as e:
            logger.debug("Config validation skipped for %s: %s", name, e)

        # Add device to config
        full_config = {**config, "device": device, **kwargs}

        # Create instance
        try:
            source = source_class(full_config)
        except Exception as e:
            raise PluginInitError(
                f"Failed to create source '{name}': {e}",
                plugin_name=name,
                cause=e,
            )

        # Initialize if source has lifecycle
        if hasattr(source, "on_init") and hasattr(source, "state"):
            try:
                from src.domain.interfaces import PluginState

                if source.state == PluginState.CREATED:
                    source.on_init()
            except Exception as e:
                raise PluginInitError(
                    f"Source '{name}' initialization failed: {e}",
                    plugin_name=name,
                    cause=e,
                )

        return source

    @classmethod
    def find_for_path(cls, path: str) -> type[BaseGaussianSource] | None:
        """Find a source that can load the given path.

        Checks each registered source's can_load() method.

        Parameters
        ----------
        path : str
            File or directory path

        Returns
        -------
        Type[BaseGaussianSource] | None
            First matching source class, or None
        """
        cls._load_entry_points()

        for name, source in cls._sources.items():
            try:
                if source.can_load(path):
                    return source
            except Exception as e:
                logger.debug("Source '%s' can_load check failed for '%s': %s", name, path, e)
                continue
        return None

    @classmethod
    def list_all(cls) -> list[SourceMetadata]:
        """List all registered sources with metadata.

        Returns
        -------
        list[SourceMetadata]
            Metadata for all registered sources
        """
        cls._load_entry_points()

        result = []
        for name, source in cls._sources.items():
            try:
                result.append(source.metadata())
            except Exception as e:
                logger.debug("Failed to get metadata for source '%s': %s", name, e)
        return result

    @classmethod
    def names(cls) -> list[str]:
        """Get list of registered source names.

        Returns
        -------
        list[str]
            Sorted list of source identifiers
        """
        cls._load_entry_points()
        return sorted(cls._sources.keys())

    @classmethod
    def clear(cls) -> None:
        """Clear all registered sources (for testing)."""
        cls._sources.clear()
        cls._entry_points_loaded = False

    # Entry point discovery

    @classmethod
    def _load_entry_points(cls) -> None:
        """Load sources from entry_points (lazy, called once)."""
        if cls._entry_points_loaded:
            return

        cls._entry_points_loaded = True

        try:
            eps = entry_points(group="gsplay.plugins")
        except TypeError:
            # Python < 3.10 compatibility
            all_eps = entry_points()
            eps = all_eps.get("gsplay.plugins", [])

        for ep in eps:
            try:
                source_class = ep.load()
                # Only register if not already registered
                if ep.name not in cls._sources:
                    cls.register(ep.name, source_class, validate=True)
                    logger.debug("Loaded source from entry_point: %s", ep.name)
            except Exception as e:
                logger.warning(
                    "Failed to load source entry_point '%s': %s",
                    ep.name,
                    e,
                )

    @classmethod
    def discover_from_entry_points(cls) -> list[str]:
        """Explicitly discover and load all entry points.

        Returns
        -------
        list[str]
            Names of newly discovered sources
        """
        before = set(cls._sources.keys())
        cls._entry_points_loaded = False  # Force reload
        cls._load_entry_points()
        after = set(cls._sources.keys())
        return list(after - before)


__all__ = ["SourceRegistry"]
