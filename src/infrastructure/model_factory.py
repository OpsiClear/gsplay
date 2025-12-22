"""
Model factory for creating model instances based on configuration.

This module provides a thin facade over SourceRegistry for backward compatibility.
New code should use SourceRegistry.create_validated() directly.
"""

import logging
from typing import Any, Protocol

from src.domain.interfaces import BaseGaussianSource, DataLoaderInterface


logger = logging.getLogger(__name__)


def _ensure_registry_initialized() -> None:
    """Ensure the data source registry is initialized with defaults."""
    from src.infrastructure.registry import register_defaults

    register_defaults()


class ModelFactoryInterface(Protocol):
    """Protocol for model factory implementations."""

    @staticmethod
    def create(
        module_type: str,
        module_config: dict[str, Any],
        device: str,
        config_file: str | None = None,
        **kwargs: Any,
    ) -> tuple[BaseGaussianSource, DataLoaderInterface | None, dict[str, Any]]:
        """Create a model instance based on configuration."""
        ...


class ModelFactory:
    """
    Factory for creating model instances based on configuration.

    This factory is now a thin facade over SourceRegistry. New code should
    use SourceRegistry.create_validated() directly.

    Maintains backward compatibility with existing code that uses:
    - ModelFactory.create()
    - ModelFactory.create_from_path()
    - ModelFactory.register_builder()
    """

    # Legacy builder support (deprecated)
    _builders: dict[str, Any] = {}

    @classmethod
    def register_builder(cls, module_type: str, builder: Any) -> None:
        """Register a legacy builder (deprecated).

        Use SourceRegistry.register() instead.
        """
        cls._builders[module_type] = builder
        logger.debug("Registered legacy builder: %s", module_type)

    @staticmethod
    def create(
        module_type: str,
        module_config: dict[str, Any],
        device: str,
        config_file: str | None = None,
        **kwargs: Any,
    ) -> tuple[BaseGaussianSource, DataLoaderInterface | None, dict[str, Any]]:
        """
        Create a model instance based on configuration.

        Parameters
        ----------
        module_type : str
            Type of module to create ('load-ply', 'composite')
        module_config : dict
            Module-specific configuration parameters
        device : str
            Device to use for computation ('cuda', 'cpu')
        config_file : str | None
            Optional path to original config file (for reference)
        **kwargs
            Additional parameters passed to model constructors

        Returns
        -------
        tuple[BaseGaussianSource, DataLoaderInterface | None, dict[str, Any]]
            - The created model instance
            - Optional data loader (None for most models)
            - Metadata dict with optional fields like 'source_path' and 'recommended_max_scale'

        Raises
        ------
        ValueError
            If module_type is not recognized or configuration is invalid
        """
        logger.debug("Creating model: %s", module_type)

        # Ensure registry is initialized
        _ensure_registry_initialized()

        # Handle composite model specially (it has unique requirements)
        if module_type == "composite":
            return ModelFactory._create_composite_model(
                module_config, device, config_file, **kwargs
            )

        # Check legacy builders first
        builder = ModelFactory._builders.get(module_type)
        if builder:
            return builder(module_config, device, config_file, **kwargs)

        # Use unified SourceRegistry
        from src.infrastructure.registry import SourceRegistry

        source = SourceRegistry.create_validated(
            module_type,
            module_config,
            device=device,
            **kwargs,
        )

        # Build metadata
        metadata: dict[str, Any] = {
            "config_file": config_file,
            "total_frames": source.total_frames,
        }

        # Extract additional metadata if available
        if hasattr(source, "ply_folder"):
            metadata["source_path"] = source.ply_folder
        if hasattr(source, "get_recommended_max_scale"):
            recommended = source.get_recommended_max_scale()
            if recommended is not None:
                metadata["recommended_max_scale"] = recommended

        return source, None, metadata

    @staticmethod
    def create_from_path(path: str, device: str = "cuda") -> BaseGaussianSource:
        """Auto-detect source type from path and create model.

        Parameters
        ----------
        path : str
            File or directory path
        device : str
            Device to use

        Returns
        -------
        BaseGaussianSource
            Created model instance

        Raises
        ------
        ValueError
            If no loader can handle the path
        """
        _ensure_registry_initialized()

        from src.infrastructure.registry import SourceRegistry

        source_class = SourceRegistry.find_for_path(path)

        if source_class is None:
            raise ValueError(f"No loader found for: {path}")

        config = {"ply_folder": path, "device": device}
        source = source_class(config)
        return source

    @staticmethod
    def _create_composite_model(
        config: dict[str, Any],
        device: str,
        config_file: str | None = None,
        **kwargs: Any,
    ) -> tuple[BaseGaussianSource, None, dict[str, Any]]:
        """Create composite multi-asset model.

        Accepts optional `edit_manager_factory` in kwargs for per-layer edit
        pipeline support. If not provided, edits are disabled.
        """
        from src.models.composite.composite_model import CompositeModel

        logger.info("Loading composite multi-asset model")

        # Extract layers config
        layers_config = config.get("layers", [])
        if not layers_config:
            raise ValueError("Composite module requires 'layers' in config")

        # Convert list to dict[layer_id, config]
        layer_dict = {layer["id"]: layer for layer in layers_config}

        # Extract optional edit_manager_factory from kwargs
        edit_manager_factory = kwargs.pop("edit_manager_factory", None)

        model = CompositeModel(
            layer_configs=layer_dict,
            device=device,
            edit_manager_factory=edit_manager_factory,
        )

        metadata: dict[str, Any] = {
            "layers": list(layer_dict.keys()),
            "total_frames": model.total_frames,
        }

        logger.info(
            "Loaded composite model with %d layers: %s",
            len(layer_dict),
            ", ".join(layer_dict.keys()),
        )

        return model, None, metadata


# Export for convenience
__all__ = ["ModelFactory", "ModelFactoryInterface"]
