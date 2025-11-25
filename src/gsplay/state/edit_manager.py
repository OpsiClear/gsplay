"""
Edit history management and application.

This module extracts edit management logic from UniversalGSPlay,
following Single Responsibility Principle.

Supports 5 processing modes for CPU/GPU optimization:
- All GPU: Everything on GPU (fastest, default)
- Color+Transform GPU: Filter on CPU, rest on GPU
- Transform GPU: Filter+Color on CPU, Transform on GPU
- Color GPU: Filter+Transform on CPU, Color on GPU
- All CPU: Everything on CPU (max GPU memory savings)
"""

from __future__ import annotations

import logging

from gsply import GSData, GSTensor

from src.infrastructure.processing_mode import ProcessingMode
from src.gsplay.config.settings import GSPlayConfig
from src.gsplay.processing import (
    AllCpuStrategy,
    AllGpuStrategy,
    ColorGpuStrategy,
    ColorProcessor,
    ColorTransformGpuStrategy,
    DefaultColorProcessor,
    DefaultGSBridge,
    DefaultOpacityAdjuster,
    DefaultSceneTransformer,
    EditContext,
    GSBridge,
    OpacityAdjuster,
    ProcessingStrategy,
    SceneTransformer,
    TransformGpuStrategy,
    VolumeFilterService,
)

logger = logging.getLogger(__name__)


class EditManager:
    """
    Manages edit history and application of edits to Gaussian data.

    This class is responsible for:
    - Tracking edit history
    - Computing edit settings hash for cache invalidation
    - Applying transforms, filters, and color adjustments
    - Managing edit state (active/inactive)
    """

    def __init__(
        self,
        config: GSPlayConfig,
        device: str,
        *,
        color_processor: ColorProcessor | None = None,
        scene_transformer: SceneTransformer | None = None,
        opacity_adjuster: OpacityAdjuster | None = None,
        volume_filter: VolumeFilterService | None = None,
        gs_bridge: GSBridge | None = None,
        strategies: dict[ProcessingMode, ProcessingStrategy] | None = None,
    ):
        """
        Initialize the edit manager.

        Parameters
        ----------
        config : GSPlayConfig
            GSPlay configuration containing edit settings
        device : str
            Device for computation ('cuda' or 'cpu')
        """
        self.config = config
        self.device = device
        self.edit_history: dict[int, dict[str, object]] = {}

        # Edit result caching for performance
        self._edit_cache: dict[int, GSTensor] = {}
        self._edit_settings_hash: int | None = None
        self._current_frame_idx: int = 0
        self._last_edit_profile: dict[str, object] | None = None
        self._color_processor = color_processor or DefaultColorProcessor()
        self._scene_transformer = scene_transformer or DefaultSceneTransformer()
        self._opacity_adjuster = opacity_adjuster or DefaultOpacityAdjuster()
        self._volume_filter = volume_filter or VolumeFilterService()
        self._gs_bridge = gs_bridge or DefaultGSBridge()
        self._strategies: dict[ProcessingMode, ProcessingStrategy] = strategies or {
            ProcessingMode.ALL_GPU: AllGpuStrategy(),
            ProcessingMode.ALL_CPU: AllCpuStrategy(),
            ProcessingMode.COLOR_GPU: ColorGpuStrategy(),
            ProcessingMode.COLOR_TRANSFORM_GPU: ColorTransformGpuStrategy(),
            ProcessingMode.TRANSFORM_GPU: TransformGpuStrategy(),
        }

    def compute_settings_hash(self) -> int:
        """
        Compute a fast hash of current edit settings for cache invalidation.

        Returns
        -------
        int
            Hash value representing current edit settings
        """
        # Use fast tuple hashing instead of JSON serialization
        # This is ~100x faster than json.dumps()
        transform = self.config.transform_values
        translate = tuple(
            float(x)
            for x in getattr(transform, "translate", getattr(transform, "translation", (0.0, 0.0, 0.0)))
        )
        rotate = tuple(
            float(x)
            for x in getattr(transform, "rotate", getattr(transform, "rotation", (0.0, 0.0, 0.0, 1.0)))
        )
        scale = (
            float(transform.scale)
            if isinstance(transform.scale, (int, float))
            else float(transform.scale[0])
        )
        transform_tuple = (*translate, scale, *rotate)

        color = self.config.color_values
        color_tuple = (
            color.temperature,
            color.brightness,
            color.contrast,
            color.saturation,
            color.vibrance,
            color.hue_shift,
            color.gamma,
            color.shadows,
            color.highlights,
            self.config.alpha_scaler,
        )

        # Use filter_values (updated from UI) for hash computation
        fv = self.config.filter_values
        volume_tuple = (
            fv.min_opacity,
            fv.max_opacity,
            fv.min_scale,
            fv.max_scale,
            fv.sphere_radius,
            fv.sphere_center,
            fv.box_min,
            fv.box_max,
            fv.ellipsoid_center,
            fv.ellipsoid_radii,
            self.config.volume_filter.processing_mode,  # Include mode for cache invalidation
        )

        return hash(
            (transform_tuple, color_tuple, volume_tuple, self.config.edits_active)
        )

    def apply_edits(
        self,
        gaussians: GSData | GSTensor,
        scene_bounds: dict[str, object] | None = None,
    ) -> GSTensor:
        """
        Apply all edits to Gaussian data using configured processing mode.

        Routes to mode-specific implementation based on config.volume_filter.processing_mode.

        Parameters
        ----------
        gaussians : GSData | GSTensor
            Input Gaussian data (GSData from CPU modes, GSTensor from GPU mode)
        scene_bounds : dict[str, object] | None
            Scene bounds for volume filtering

        Returns
        -------
        GSTensor
            Edited Gaussian data on GPU, ready for rendering
        """
        if not self.config.edits_active:
            # No edits - convert to GSTensor if needed for rendering
            if isinstance(gaussians, GSData):
                # Use gsply v0.2.5 GPU loading interface
                return GSTensor.from_gsdata(gaussians, device=self.device)
            return gaussians

        # Determine processing mode
        try:
            mode = ProcessingMode.from_string(self.config.volume_filter.processing_mode)
        except (ValueError, AttributeError):
            # Fall back to default if invalid/missing
            logger.warning(
                f"Invalid processing mode '{self.config.volume_filter.processing_mode}', "
                f"using ALL_GPU"
            )
            mode = ProcessingMode.ALL_GPU

        strategy = self._strategies.get(mode)
        if strategy is None:
            logger.error("Unknown processing mode %s, falling back to ALL_GPU", mode)
            strategy = self._strategies[ProcessingMode.ALL_GPU]

        context = EditContext(
            config=self.config,
            device=self.device,
            color_processor=self._color_processor,
            scene_transformer=self._scene_transformer,
            opacity_adjuster=self._opacity_adjuster,
            volume_filter=self._volume_filter,
            gaussian_bridge=self._gs_bridge,
        )
        result = strategy.apply(context, gaussians, scene_bounds)
        self._record_edit_profile(strategy.mode.value, result.timings)
        return result.gaussians

    def clear_edit_history(self) -> None:
        """Clear all edit history and caches."""
        self.edit_history.clear()
        self._edit_cache.clear()
        self._edit_settings_hash = None
        logger.info("Edit history cleared")

    def invalidate_cache(self) -> None:
        """Invalidate the edit cache."""
        self._edit_cache.clear()
        self._edit_settings_hash = None
        logger.info("Edit cache invalidated")

    def _record_edit_profile(self, mode: str, timings: dict[str, float]) -> None:
        """Store most recent edit timing breakdown for perf logging."""
        self._last_edit_profile = {"mode": mode, **timings}
