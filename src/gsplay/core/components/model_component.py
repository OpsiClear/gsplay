"""
Model component for managing model loading and lifecycle.

This component is responsible for:
- Loading models from configuration
- Model lifecycle management
- Metadata processing
- Model state tracking
"""

import logging
from pathlib import Path
from typing import Any

from src.domain.interfaces import ModelInterface, DataLoaderInterface
from src.infrastructure.io.path_io import UniversalPath
from src.gsplay.interaction.events import EventBus, EventType

logger = logging.getLogger(__name__)


class ModelComponent:
    """
    Component responsible for model loading and management.

    Handles:
    - Model creation from configuration
    - Model lifecycle (load, reload, unload)
    - Metadata extraction and processing
    - Event emission for model state changes
    """

    def __init__(self, device: str = "cuda", event_bus: EventBus | None = None):
        """
        Initialize model component.

        Parameters
        ----------
        device : str
            Device for model ('cuda' or 'cpu')
        event_bus : EventBus | None
            Event bus for emitting model events
        """
        self.device = device
        self.event_bus = event_bus

        # Model state
        self.model: ModelInterface | None = None
        self.data_loader: DataLoaderInterface | None = None
        self.source_path: Path | None = None
        self.metadata: dict[str, Any] = {}

        logger.debug(f"ModelComponent initialized (device={device})")

    def load_from_config(
        self,
        config_dict: dict[str, Any],
        config_file: str | None = None,
        **extra_config,
    ) -> tuple[ModelInterface, DataLoaderInterface | None, dict[str, Any]]:
        """
        Load model from configuration dictionary.

        Parameters
        ----------
        config_dict : dict[str, Any]
            Configuration with 'module' and 'config' keys
        config_file : str | None
            Path to config file (for reference)
        **extra_config
            Additional config options to merge

        Returns
        -------
        tuple[ModelInterface, DataLoaderInterface | None, dict[str, Any]]
            Loaded model, data loader, and metadata

        Raises
        ------
        ValueError
            If configuration is invalid
        """
        # Emit loading started event
        if self.event_bus:
            self.event_bus.emit(
                EventType.MODEL_LOAD_STARTED,
                source="model_component",
                config=config_dict,
            )

        try:
            from src.infrastructure.model_factory import ModelFactory

            module_type = config_dict.get("module", "load-ply")
            module_config = dict(config_dict.get("config", {}))

            # Merge extra config
            module_config.update(extra_config)

            logger.info(f"Loading model (module={module_type}, device={self.device})")

            # Create model using factory
            self.model, self.data_loader, self.metadata = ModelFactory.create(
                module_type=module_type,
                module_config=module_config,
                device=self.device,
                config_file=config_file,
            )

            # Extract source path from metadata
            if "source_path" in self.metadata:
                self.source_path = self.metadata["source_path"]
                logger.debug(f"Source path: {self.source_path}")

            # Emit loading completed event
            if self.event_bus:
                self.event_bus.emit(
                    EventType.MODEL_LOADED,
                    source="model_component",
                    module_type=module_type,
                    total_frames=self.model.get_total_frames(),
                    source_path=str(self.source_path) if self.source_path else None,
                )

            logger.info(
                f"Model loaded successfully: {self.model.get_total_frames()} frames"
            )

            return self.model, self.data_loader, self.metadata

        except Exception as e:
            logger.error(f"Model loading failed: {e}", exc_info=True)

            # Emit loading failed event
            if self.event_bus:
                self.event_bus.emit(
                    EventType.MODEL_LOAD_FAILED, source="model_component", error=str(e)
                )

            raise

    def load_from_path(
        self, path: str | Path, **extra_config
    ) -> tuple[ModelInterface, DataLoaderInterface | None, dict[str, Any]]:
        """
        Load model from path (PLY folder or JSON config).

        Parameters
        ----------
        path : str | Path
            Path to PLY folder or JSON config file
        **extra_config
            Additional config options

        Returns
        -------
        tuple[ModelInterface, DataLoaderInterface | None, dict[str, Any]]
            Loaded model, data loader, and metadata

        Raises
        ------
        ValueError
            If path is invalid or doesn't exist
        """
        import json

        path_obj = UniversalPath(path)

        if not path_obj.exists():
            raise ValueError(f"Path does not exist: {path}")

        # Determine config type
        if path_obj.is_dir():
            # PLY folder
            config_dict = {
                "module": "load-ply",
                "config": {
                    "ply_folder": str(path_obj),
                },
            }
            logger.info(f"Loading PLY sequence from directory: {path_obj}")

        elif path_obj.is_file() and path_obj.suffix == ".json":
            # JSON config
            with open(path_obj, "r") as f:
                config_dict = json.load(f)
            logger.info(f"Loading from JSON config: {path_obj}")

        else:
            raise ValueError(
                f"Invalid path type: {path}. Must be PLY folder or JSON config."
            )

        return self.load_from_config(
            config_dict, config_file=str(path_obj), **extra_config
        )

    def reload(self) -> bool:
        """
        Reload the current model.

        Returns
        -------
        bool
            True if reload succeeded
        """
        if not self.source_path:
            logger.warning("Cannot reload: no source path available")
            return False

        try:
            logger.info(f"Reloading model from: {self.source_path}")
            self.load_from_path(self.source_path)
            return True

        except Exception as e:
            logger.error(f"Model reload failed: {e}", exc_info=True)
            return False

    def unload(self) -> None:
        """Unload the current model and free resources."""
        if self.model:
            logger.info("Unloading model")

            # Emit unloading event
            if self.event_bus:
                self.event_bus.emit(EventType.MODEL_UNLOADED, source="model_component")

            self.model = None
            self.data_loader = None
            self.metadata = {}

            logger.debug("Model unloaded")

    def get_model(self) -> ModelInterface | None:
        """Get the current model."""
        return self.model

    def get_data_loader(self) -> DataLoaderInterface | None:
        """Get the current data loader."""
        return self.data_loader

    def get_metadata(self) -> dict[str, Any]:
        """Get model metadata."""
        return self.metadata.copy()

    def get_source_path(self) -> Path | None:
        """Get the source path of the loaded model."""
        return self.source_path

    def is_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self.model is not None

    def get_recommended_max_scale(self) -> float | None:
        """
        Get recommended max_scale from metadata.

        Returns
        -------
        float | None
            Recommended max_scale value, or None if not available
        """
        return self.metadata.get("recommended_max_scale")


# Export public API
__all__ = ["ModelComponent"]
