"""
Opacity adjustment adapter used by the edit pipeline.

Uses gsmod's format-aware opacity processing that handles both
linear [0, 1] and PLY (logit) opacity formats correctly.
"""

from __future__ import annotations

from gsmod.config.values import OpacityValues
from gsmod.opacity.apply import apply_opacity_values

from src.domain.entities import GSData, GSTensor

from .protocols import OpacityAdjuster


class DefaultOpacityAdjuster(OpacityAdjuster):
    """Opacity adjustments that work on CPU or GPU.

    Uses gsmod's format-aware opacity processing that handles both
    linear [0, 1] and PLY (logit) opacity formats correctly.
    """

    def apply_gpu(self, gaussians: GSTensor, alpha: float) -> GSTensor:
        """Apply opacity scaling on GPU.

        Uses GSTensorPro.opacity() if available, otherwise uses gsmod directly.

        :param gaussians: GSTensor or GSTensorPro
        :param alpha: Opacity scale factor (1.0 = no change)
        :return: Modified gaussians
        """
        if alpha == 1.0:
            return gaussians

        values = OpacityValues(scale=alpha)

        # Use GSTensorPro.opacity() if available
        if hasattr(gaussians, "opacity"):
            return gaussians.opacity(values, inplace=True)

        # Fallback: use gsmod's GaussianProcessor
        from gsmod.processing import GaussianProcessor

        processor = GaussianProcessor()
        return processor.opacity(gaussians, values, inplace=True)

    def apply_cpu(self, data: GSData, alpha: float) -> GSData:
        """Apply opacity scaling on CPU.

        Uses gsmod's format-aware opacity adjustment.

        :param data: GSData or GSDataPro
        :param alpha: Opacity scale factor (1.0 = no change)
        :return: Modified data
        """
        if alpha == 1.0:
            return data

        values = OpacityValues(scale=alpha)

        # Use GSDataPro.opacity() if available
        if hasattr(data, "opacity"):
            return data.opacity(values, inplace=True)

        # Fallback: use gsmod directly
        is_ply = getattr(data, "is_opacities_ply", False)
        data.opacities = apply_opacity_values(data.opacities, values, is_ply_format=is_ply)
        return data
