"""
Model factory for creating model instances based on configuration.

This module provides a centralized factory for instantiating different model types,
removing model-specific logic from the viewer layer and enabling easier extensibility.

Now integrates with DataSourceRegistry for unified source discovery.
"""
import logging
from collections.abc import Callable
from typing import Any, Protocol

from src.domain.interfaces import ModelInterface, DataLoaderInterface, ConfigurableModelInterface
from src.infrastructure.io.path_io import UniversalPath
from src.infrastructure.io.discovery import discover_and_sort_ply_files
from src.infrastructure.processing.gaussian_constants import GaussianConstants as GC

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
        **kwargs
    ) -> tuple[ModelInterface, DataLoaderInterface | None, dict[str, Any]]:
        """
        Create a model instance based on configuration.

        Parameters
        ----------
        module_type : str
            Type of module to create ('load-ply', 'sogs', 'composite', etc.)
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
        tuple[ModelInterface, DataLoaderInterface | None, dict[str, Any]]
            - The created model instance
            - Optional data loader (None for streaming models)
            - Metadata dict with optional fields like 'source_path' and 'recommended_max_scale'
        """
        ...


type BuilderResult = tuple[ModelInterface, DataLoaderInterface | None, dict[str, Any]]
type ModelBuilder = Callable[[dict[str, Any], str, str | None], BuilderResult]


class ModelFactory:
    """
    Factory for creating model instances based on configuration.

    This factory centralizes all model instantiation logic, making it easy
    to add new model types without modifying the viewer code.
    """

    # Registry of model types that support from_config
    _configurable_models: dict[str, type[ConfigurableModelInterface]] = {}
    _builders: dict[str, ModelBuilder] = {}

    @classmethod
    def register_model(
        cls,
        module_type: str,
        model_class: type[ConfigurableModelInterface]
    ) -> None:
        """
        Register a model class that implements from_config.

        Parameters
        ----------
        module_type : str
            Module type identifier (e.g., "load-ply", "sogs")
        model_class : type[ConfigurableModelInterface]
            Model class implementing from_config method
        """
        cls._configurable_models[module_type] = model_class
        logger.debug(f"Registered configurable model: {module_type} -> {model_class.__name__}")

    @classmethod
    def register_builder(
        cls,
        module_type: str,
        builder: ModelBuilder,
    ) -> None:
        """
        Register a builder callable for a module type.

        Parameters
        ----------
        module_type : str
            Module type identifier
        builder : ModelBuilder
            Callable that returns (model, data_loader, metadata)
        """
        cls._builders[module_type] = builder
        logger.debug("Registered model builder: %s -> %s", module_type, builder.__name__)

    @staticmethod
    def create(
        module_type: str,
        module_config: dict[str, Any],
        device: str,
        config_file: str | None = None,
        **kwargs
    ) -> tuple[ModelInterface, DataLoaderInterface | None, dict[str, Any]]:
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
        tuple[ModelInterface, DataLoaderInterface | None, dict[str, Any]]
            - The created model instance
            - Optional data loader (None for streaming models)
            - Metadata dict with optional fields like 'source_path' and 'recommended_max_scale'

        Raises
        ------
        ValueError
            If module_type is not recognized or configuration is invalid
        ImportError
            If required model module cannot be imported
        """
        logger.debug(f"Creating model: {module_type}")

        # Ensure registry is initialized
        _ensure_registry_initialized()

        # First check legacy builders (for backward compatibility)
        builder = ModelFactory._builders.get(module_type)
        if builder:
            return builder(module_config, device, config_file, **kwargs)

        configurable = ModelFactory._configurable_models.get(module_type)
        if configurable:
            return ModelFactory._create_from_configurable(
                configurable,
                module_config,
                device,
                config_file=config_file,
                **kwargs,
            )

        # Check registry for new-style data sources
        from src.infrastructure.registry import DataSourceRegistry
        source_class = DataSourceRegistry.get(module_type)
        if source_class:
            return ModelFactory._create_from_data_source(
                source_class,
                module_config,
                device,
                config_file=config_file,
                **kwargs,
            )

        # Collect all available types for error message
        all_types = set(ModelFactory._builders.keys())
        all_types.update(DataSourceRegistry.names())

        raise ValueError(
            f"Unknown module type: {module_type}. "
            f"Registered types: {', '.join(sorted(all_types))}"
        )

    @staticmethod
    def _create_from_data_source(
        source_class: type,
        config: dict[str, Any],
        device: str,
        config_file: str | None = None,
        **kwargs,
    ) -> tuple[ModelInterface, DataLoaderInterface | None, dict[str, Any]]:
        """Create model from a DataSourceProtocol implementation."""
        # Add device to config
        config_with_device = {**config, "device": device}

        # Create source instance
        source = source_class(config_with_device)

        # Build metadata
        metadata = {
            "config_file": config_file,
            "total_frames": source.total_frames,
        }

        # Extract additional metadata if available
        if hasattr(source, "ply_folder"):
            metadata["source_path"] = source.ply_folder
        if hasattr(source, "get_recommended_max_scale"):
            metadata["recommended_max_scale"] = source.get_recommended_max_scale()

        # DataSource implements TimeSampledModel methods for compatibility
        return source, None, metadata

    @staticmethod
    def create_from_path(path: str, device: str = "cuda") -> ModelInterface:
        """Auto-detect source type from path and create model.

        Parameters
        ----------
        path : str
            File or directory path
        device : str
            Device to use

        Returns
        -------
        ModelInterface
            Created model instance

        Raises
        ------
        ValueError
            If no loader can handle the path
        """
        _ensure_registry_initialized()

        from src.infrastructure.registry import DataSourceRegistry
        source_class = DataSourceRegistry.find_for_path(path)

        if source_class is None:
            raise ValueError(f"No loader found for: {path}")

        config = {"ply_folder": path, "device": device}
        source = source_class(config)
        return source

    @staticmethod
    def _create_ply_model(
        config: dict[str, Any],
        device: str,
        config_file: str | None = None,
        **kwargs
    ) -> tuple[ModelInterface, DataLoaderInterface, dict[str, Any]]:
        """Create PLY model and data loader."""
        from src.models.ply.optimized_model import (
            OptimizedPlyModel,
            OptimizedPlyDataLoader,
        )

        ply_folder = config.get("ply_folder", ".")
        enable_concurrent_prefetch = config.get("enable_concurrent_prefetch", True)

        # Processing mode and quality filtering (unified with VolumeFilter)
        processing_mode = config.get("processing_mode", "all_gpu")
        opacity_threshold = config.get("opacity_threshold", 0.01)
        scale_threshold = config.get("scale_threshold", 1e-7)
        enable_quality_filtering = config.get("enable_quality_filtering", True)

        logger.debug(f"Loading PLY files from: {ply_folder}")
        logger.debug(f"Processing mode: {processing_mode}")

        # Store source path for default export location
        source_path = UniversalPath(ply_folder)

        # Discover and sort PLY files using SINGLE authoritative function
        ply_files = discover_and_sort_ply_files(ply_folder)

        logger.info(
            f"Found {len(ply_files)} PLY files (numerically sorted by "
            f"discover_and_sort_ply_files)"
        )

        # Log file order for verification
        if len(ply_files) > 20:
            logger.debug(f"First 10 frames: {[f.name for f in ply_files[:10]]}")
            logger.debug(f"Last 10 frames: {[f.name for f in ply_files[-10:]]}")
        else:
            logger.debug(f"All frames: {[f.name for f in ply_files]}")

        # Create model and data loader
        model = OptimizedPlyModel(
            ply_files=[str(p) for p in ply_files],
            device=device,
            enable_concurrent_prefetch=enable_concurrent_prefetch,
            processing_mode=processing_mode,
            opacity_threshold=opacity_threshold,
            scale_threshold=scale_threshold,
            enable_quality_filtering=enable_quality_filtering,
        )

        data_loader = OptimizedPlyDataLoader(
            ply_files=[str(p) for p in ply_files],
        )

        # Get recommended max_scale from model
        recommended_max_scale = model.get_recommended_max_scale()

        metadata = {
            "source_path": source_path,
            "recommended_max_scale": recommended_max_scale,
            "total_frames": len(ply_files),
        }

        if recommended_max_scale is not None:
            logger.info(
                f"Calculated initial max_scale: {recommended_max_scale:.6f} "
                f"({GC.Filtering.get_percentile_label()} from first frame)"
            )

        return model, data_loader, metadata

    @staticmethod
    def _create_composite_model(
        config: dict[str, Any],
        device: str,
        config_file: str | None = None,
        **kwargs
    ) -> tuple[ModelInterface, None, dict[str, Any]]:
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

        metadata = {
            "layers": list(layer_dict.keys()),
            "total_frames": model.get_total_frames() if hasattr(model, "get_total_frames") else 0,
        }

        logger.info(
            f"Loaded composite model with {len(layer_dict)} layers: "
            f"{', '.join(layer_dict.keys())}"
        )

        # Composite doesn't need a separate data loader
        return model, None, metadata

    @staticmethod
    def _create_from_configurable(
        model_class: type[ConfigurableModelInterface],
        config: dict[str, Any],
        device: str,
        config_file: str | None = None,
        **kwargs,
    ) -> BuilderResult:
        """
        Build a model that implements ConfigurableModelInterface.

        Currently returns the model without an explicit data loader; more
        sophisticated models can register dedicated builders for richer metadata.
        """
        model = model_class.from_config(config, device=device, **kwargs)
        metadata = {
            "config_file": config_file,
        }
        return model, None, metadata


# Export for convenience
__all__ = ["ModelFactory", "ModelFactoryInterface"]

# Register built-in builders
ModelFactory.register_builder("load-ply", ModelFactory._create_ply_model)
ModelFactory.register_builder("composite", ModelFactory._create_composite_model)
