"""
Composite model for multi-asset Gaussian Splatting.

This module provides a model that manages multiple sub-models as layers,
enabling composition of static and dynamic Gaussian datasets.

Note: Uses dependency injection for edit management to avoid coupling
to the viewer layer. The edit_manager_factory should be provided by
the viewer when creating CompositeModel instances.
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from src.domain.entities import GSTensor, GaussianLayer, CompositeGSTensor
from src.domain.filters import VolumeFilter
from src.domain.interfaces import ModelInterface, EditManagerProtocol, EditManagerFactory
from src.infrastructure.processing_mode import ProcessingMode
from gsmod import ColorValues, TransformValues, FilterValues

if TYPE_CHECKING:
    from src.gsplay.config.settings import GSPlayConfig

logger = logging.getLogger(__name__)


class CompositeModel(ModelInterface):
    """
    A composite model that manages multiple sub-models as layers.

    This model allows combining multiple Gaussian datasets (e.g., static
    background + dynamic foreground, multiple PLY sequences) into a single
    unified rendering. Each layer is managed independently and can be
    controlled via visibility, z-ordering, and opacity.

    The model implements the standard ModelInterface, so it can be used
    as a drop-in replacement for single-asset models.

    Attributes:
        models: Dictionary mapping layer_id to ModelInterface instances
        layer_configs: Dictionary mapping layer_id to layer configuration
        layer_edit_managers: Dictionary mapping layer_id to EditManager instances
        layer_viewer_configs: Dictionary mapping layer_id to GSPlayConfig instances
        layer_scene_bounds: Dictionary mapping layer_id to SceneBounds
        total_frames: Total frames (synchronized across all layers)
    """

    def __init__(
        self,
        layer_configs: dict[str, dict[str, Any]],
        device: str = "cuda",
        edit_manager_factory: EditManagerFactory | None = None,
    ):
        """
        Initialize the composite model with multiple layers.

        Args:
            layer_configs: Dictionary mapping layer_id to layer config:
                {
                    "layer_id": {
                        "type": "ply",
                        "config": {...},  # Type-specific config
                        "visible": bool,  # Default visibility
                        "z_order": int,   # Rendering order
                        "opacity_multiplier": float,  # Opacity adjustment
                        "static": bool,   # If True, only load frame 0
                        "time_range": [start_frame, end_frame] | None,  # Visibility range
                        "edits": {  # Optional per-layer edits
                            "color_adjustments": {...},
                            "scene_transform": {...},
                            "volume_filter": {...}
                        }
                    }
                }
            device: Target device for computation
            edit_manager_factory: Optional factory function to create edit managers.
                If provided, enables per-layer edit pipelines. Signature:
                (config: dict, device: str) -> EditManagerProtocol
        """
        self.device = device
        self.models: dict[str, ModelInterface] = {}
        self.layer_configs = layer_configs
        self._edit_manager_factory = edit_manager_factory

        # Per-layer edit pipeline (only used if factory provided)
        self.layer_edit_managers: dict[str, EditManagerProtocol] = {}
        self.layer_viewer_configs: dict[str, Any] = {}  # GSPlayConfig when factory provided
        self.layer_scene_bounds: dict[str, dict[str, Any]] = {}

        # Create sub-models for each layer
        for layer_id, config in layer_configs.items():
            logger.info(f"Loading layer '{layer_id}' (type: {config['type']})")
            model = self._create_model_from_config(config, device)
            self.models[layer_id] = model

            # Create per-layer edit manager if factory provided
            if edit_manager_factory is not None:
                layer_viewer_config = self._create_layer_viewer_config(config)
                self.layer_viewer_configs[layer_id] = layer_viewer_config
                self.layer_edit_managers[layer_id] = edit_manager_factory(
                    layer_viewer_config, device
                )

            # Calculate per-layer scene bounds
            self._calculate_layer_bounds(layer_id, model)

        # Determine total frames (use maximum across all layers)
        self.total_frames = max(
            (model.get_total_frames() for model in self.models.values()), default=0
        )

        logger.info(
            f"CompositeModel initialized with {len(self.models)} layers, "
            f"{self.total_frames} total frames, per-layer edits enabled"
        )

    def _create_model_from_config(
        self, config: dict[str, Any], device: str
    ) -> ModelInterface:
        """
        Factory method to create a model instance from config.

        Args:
            config: Layer configuration
            device: Target device

        Returns:
            Model instance implementing ModelInterface

        Raises:
            ValueError: If model type is unknown
        """
        model_type = config["type"]
        model_config = config["config"]

        if model_type == "ply":
            from src.models.ply.optimized_model import (
                create_optimized_ply_model_from_folder,
            )

            ply_folder = model_config.get("ply_folder")

            # Extract PLY loading config (with defaults matching PlyLoadingConfig)
            opacity_threshold = model_config.get("opacity_threshold", 0.01)
            scale_threshold = model_config.get("scale_threshold", 1e-7)
            enable_quality_filtering = model_config.get(
                "enable_quality_filtering", True
            )
            enable_concurrent_prefetch = model_config.get(
                "enable_concurrent_prefetch", True
            )

            # Get processing_mode from edits config (VolumeFilter) or model config
            # This unifies PLY loading mode with edit processing mode
            edits = config.get("edits", {})
            volume_filter = edits.get("volume_filter", {})
            raw_mode = volume_filter.get(
                "processing_mode", model_config.get("processing_mode", "all_gpu")
            )
            try:
                processing_mode = ProcessingMode.from_string(raw_mode).loader_mode
            except ValueError:
                logger.warning(
                    "Invalid processing mode '%s' for layer '%s'; defaulting loader to ALL_GPU",
                    raw_mode,
                    config.get("id", "unknown"),
                )
                processing_mode = ProcessingMode.ALL_GPU.value

            return create_optimized_ply_model_from_folder(
                ply_folder=ply_folder,
                device=device,
                enable_concurrent_prefetch=enable_concurrent_prefetch,
                processing_mode=processing_mode,
                opacity_threshold=opacity_threshold,
                scale_threshold=scale_threshold,
                enable_quality_filtering=enable_quality_filtering,
            )

        else:
            raise ValueError(f"Unknown model type: {model_type}. Supported: 'ply'")

    def _create_layer_viewer_config(self, config: dict[str, Any]) -> Any:
        """
        Create a GSPlayConfig from layer configuration.

        Args:
            config: Layer configuration dict

        Returns:
            GSPlayConfig instance with per-layer edit settings

        Note:
            GSPlayConfig is lazily imported to avoid coupling models layer
            to viewer layer at module load time.
        """
        # Lazy import to avoid coupling at module level
        from src.gsplay.config.settings import GSPlayConfig

        viewer_config = GSPlayConfig()

        # Extract edit settings from config
        edits = config.get("edits", {})

        # Color values
        if "color_values" in edits:
            cv = edits["color_values"]
            viewer_config.color_values = ColorValues(
                temperature=cv.get("temperature", 0.0),
                brightness=cv.get("brightness", 1.0),
                contrast=cv.get("contrast", 1.0),
                saturation=cv.get("saturation", 1.0),
                vibrance=cv.get("vibrance", 1.0),
                hue_shift=cv.get("hue_shift", 0.0),
                gamma=cv.get("gamma", 1.0),
                shadows=cv.get("shadows", 0.0),
                highlights=cv.get("highlights", 0.0),
            )
        if "alpha_scaler" in edits:
            viewer_config.alpha_scaler = float(edits.get("alpha_scaler", 1.0))

        # Transform values
        if "transform_values" in edits:
            tv = edits["transform_values"]
            import numpy as np

            viewer_config.transform_values = TransformValues(
                translate=np.array(
                    tv.get("translate", [0.0, 0.0, 0.0]), dtype=np.float32
                ),
                rotate=np.array(
                    tv.get("rotate", [0.0, 0.0, 0.0, 1.0]), dtype=np.float32
                ),
                scale=tv.get("scale", 1.0),
            )

        # Filter values
        if "filter_values" in edits:
            fv = edits["filter_values"]
            viewer_config.filter_values = FilterValues(
                min_opacity=fv.get("min_opacity", 0.0),
                max_opacity=fv.get("max_opacity", 1.0),
                frustum_near=fv.get("frustum_near", 0.0),
                frustum_far=fv.get("frustum_far", 1.0),
            )

        # Volume filter (UI-specific spatial filtering)
        if "volume_filter" in edits:
            vf = edits["volume_filter"]
            viewer_config.volume_filter = VolumeFilter(
                filter_type=vf.get("filter_type", "none"),
                sphere_center=tuple(vf.get("sphere_center", (0.0, 0.0, 0.0))),
                sphere_radius_factor=vf.get("sphere_radius_factor", 1.0),
                cuboid_center=tuple(vf.get("cuboid_center", (0.0, 0.0, 0.0))),
                cuboid_size_factor_x=vf.get("cuboid_size_factor_x", 1.0),
                cuboid_size_factor_y=vf.get("cuboid_size_factor_y", 1.0),
                cuboid_size_factor_z=vf.get("cuboid_size_factor_z", 1.0),
            )

        # Mark edits as active if any edits are configured
        viewer_config.edits_active = bool(edits)

        return viewer_config

    def _calculate_layer_bounds(self, layer_id: str, model: ModelInterface) -> None:
        """
        Calculate scene bounds for a layer.

        Args:
            layer_id: ID of the layer
            model: Model instance for this layer
        """
        try:
            # Get first frame to calculate bounds
            gaussian_data = model.get_gaussians_at_normalized_time(0.0)
            if gaussian_data is None:
                logger.warning(
                    f"Could not calculate bounds for layer '{layer_id}': no data at frame 0"
                )
                return

            # Calculate bounds from means
            means = gaussian_data.means
            if means.shape[0] == 0:
                logger.warning(f"Layer '{layer_id}' has no gaussians")
                return

            # Move to CPU for bounds calculation
            if means.device.type == "cuda":
                means = means.cpu()

            import torch

            mins = torch.min(means, dim=0)[0]
            maxs = torch.max(means, dim=0)[0]

            center = ((mins + maxs) / 2).numpy()
            size = (maxs - mins).numpy()
            radius = float(torch.norm(maxs - mins).item() / 2)

            self.layer_scene_bounds[layer_id] = {
                "center": tuple(center.tolist()),
                "size": tuple(size.tolist()),
                "radius": radius,
            }

            logger.debug(
                f"Layer '{layer_id}' bounds: center={center}, size={size}, radius={radius:.3f}"
            )

        except Exception as e:
            logger.error(f"Error calculating bounds for layer '{layer_id}': {e}")

    def get_gaussians_at_normalized_time(
        self,
        normalized_time: float,
    ) -> GSTensor | None:
        """
        Get merged Gaussian data from all visible layers at given time.

        This method:
        1. Fetches GSTensor from each visible layer
        2. Applies per-layer edits (transform, filter, color)
        3. Wraps them in GaussianLayer with metadata
        4. Creates CompositeGSTensor
        5. Merges layers into single GSTensor for rendering

        Args:
            normalized_time: Normalized time in [0.0, 1.0]

        Returns:
            Merged GSTensor from all visible layers, or None if no layers
        """
        layers = []

        # Convert normalized time to frame index for time_range filtering
        current_frame = (
            int(normalized_time * (self.total_frames - 1))
            if self.total_frames > 1
            else 0
        )

        for layer_id, model in self.models.items():
            config = self.layer_configs[layer_id]

            # Check visibility
            visible = config.get("visible", True)
            if not visible:
                continue

            # Check time_range visibility
            time_range = config.get("time_range")
            if time_range is not None:
                start_frame, end_frame = time_range
                if current_frame < start_frame or current_frame > end_frame:
                    logger.debug(
                        f"Layer '{layer_id}' not visible at frame {current_frame} "
                        f"(range: {start_frame}-{end_frame})"
                    )
                    continue

            # For static layers, always use frame 0
            if config.get("static", False):
                layer_time = 0.0
            else:
                layer_time = normalized_time

            # Fetch Gaussian data from sub-model
            try:
                gaussian_data = model.get_gaussians_at_normalized_time(layer_time)
                if gaussian_data is None:
                    logger.warning(
                        f"Layer '{layer_id}' returned None at time {layer_time}"
                    )
                    continue
            except Exception as e:
                logger.error(f"Error fetching layer '{layer_id}': {e}")
                continue

            # Apply per-layer edits
            try:
                if layer_id in self.layer_edit_managers:
                    scene_bounds = self.layer_scene_bounds.get(layer_id)
                    gaussian_data = self.layer_edit_managers[layer_id].apply_edits(
                        gaussian_data, scene_bounds=scene_bounds
                    )
            except Exception as e:
                logger.error(f"Error applying edits to layer '{layer_id}': {e}")
                # Continue with unedited data

            # Wrap in GaussianLayer with metadata
            layer = GaussianLayer(
                data=gaussian_data,
                layer_id=layer_id,
                visible=True,  # Already filtered above
                z_order=config.get("z_order", 0),
                opacity_multiplier=config.get("opacity_multiplier", 1.0),
            )
            layers.append(layer)

        # If no visible layers, return None
        if not layers:
            logger.debug("No visible layers available at current time")
            return None

        # Create composite and merge
        composite = CompositeGSTensor(layers=layers)
        try:
            merged = composite.merge()
            return merged
        except ValueError as e:
            logger.error(f"Failed to merge layers: {e}")
            return None

    def get_total_frames(self) -> int:
        """
        Get the total number of frames across all layers.

        Returns:
            Maximum frame count across all layers
        """
        return self.total_frames

    def get_frame_time(self, frame_idx: int) -> float:
        """
        Get normalized time for a given frame index.

        Args:
            frame_idx: Frame index (0-based)

        Returns:
            Normalized time in [0.0, 1.0]
        """
        if self.total_frames <= 1:
            return 0.0
        return frame_idx / (self.total_frames - 1)

    def set_layer_visibility(self, layer_id: str, visible: bool) -> None:
        """
        Set the visibility of a specific layer.

        Args:
            layer_id: ID of the layer to modify
            visible: New visibility state

        Raises:
            KeyError: If layer_id doesn't exist
        """
        if layer_id not in self.layer_configs:
            raise KeyError(f"Layer '{layer_id}' not found")

        self.layer_configs[layer_id]["visible"] = visible
        logger.debug(f"Layer '{layer_id}' visibility set to {visible}")

    def get_layer_ids(self) -> list[str]:
        """
        Get all layer IDs in the order they were added.

        Returns:
            List of layer IDs
        """
        return list(self.models.keys())

    def get_layer_info(self) -> dict[str, dict[str, Any]]:
        """
        Get information about all layers.

        Returns:
            Dictionary mapping layer_id to layer info:
            {
                "layer_id": {
                    "type": str,
                    "visible": bool,
                    "z_order": int,
                    "frames": int,
                    "static": bool,
                    "opacity_multiplier": float,
                    "time_range": [int, int] | None,
                    "has_edits": bool,
                }
            }
        """
        info = {}
        for layer_id, model in self.models.items():
            config = self.layer_configs[layer_id]
            info[layer_id] = {
                "type": config["type"],
                "visible": config.get("visible", True),
                "z_order": config.get("z_order", 0),
                "frames": model.get_total_frames(),
                "static": config.get("static", False),
                "opacity_multiplier": config.get("opacity_multiplier", 1.0),
                "time_range": config.get("time_range"),
                "has_edits": layer_id in self.layer_viewer_configs
                and self.layer_viewer_configs[layer_id].edits_active,
            }
        return info

    def get_layer_viewer_config(self, layer_id: str) -> GSPlayConfig | None:
        """
        Get the GSPlayConfig for a specific layer.

        Args:
            layer_id: ID of the layer

        Returns:
            GSPlayConfig instance for the layer, or None if not found
        """
        return self.layer_viewer_configs.get(layer_id)

    def update_layer_edits(
        self,
        layer_id: str,
        color_values: ColorValues | None = None,
        transform_values: TransformValues | None = None,
        filter_values: FilterValues | None = None,
        volume_filter: VolumeFilter | None = None,
        alpha_scaler: float | None = None,
    ) -> None:
        """
        Update edit settings for a specific layer.

        Args:
            layer_id: ID of the layer to update
            color_values: New color values (if provided)
            transform_values: New transform values (if provided)
            filter_values: New filter values (if provided)
            volume_filter: New volume filter settings (if provided)

        Raises:
            KeyError: If layer_id doesn't exist
        """
        if layer_id not in self.layer_viewer_configs:
            raise KeyError(f"Layer '{layer_id}' not found")

        config = self.layer_viewer_configs[layer_id]

        if color_values is not None:
            config.color_values = color_values

        if transform_values is not None:
            config.transform_values = transform_values

        if filter_values is not None:
            config.filter_values = filter_values

        if volume_filter is not None:
            config.volume_filter = volume_filter

        if alpha_scaler is not None:
            config.alpha_scaler = alpha_scaler

        # Mark edits as active if any non-default edits exist
        config.edits_active = (
            not config.color_values.is_neutral()
            or not config.transform_values.is_neutral()
            or config.alpha_scaler != 1.0
            or config.volume_filter.is_active()
        )

        logger.debug(
            f"Updated edits for layer '{layer_id}', edits_active={config.edits_active}"
        )
