"""Opacity processing using gsmod's native GPU-accelerated operations.

Uses GSTensorPro.opacity() for GPU and GSDataPro.opacity() for CPU.
"""

from __future__ import annotations

from gsmod.config.values import OpacityValues
from src.domain.entities import GSData, GSDataPro, GSTensor, GSTensorPro

from .protocols import OpacityAdjuster


class DefaultOpacityAdjuster(OpacityAdjuster):
    """Opacity adjuster using gsmod's native GPU-accelerated operations.

    GPU: GSTensorPro.opacity() with PyTorch CUDA operations
    CPU: GSDataPro.opacity() with format-aware processing
    """

    def apply_gpu(self, gaussians: GSTensor, alpha: float) -> GSTensor:
        """Apply opacity scaling using native GSTensorPro.opacity().

        Uses inplace=False to let gsmod handle copying internally.
        """
        if alpha == 1.0:
            return gaussians

        # Wrap GSTensor as GSTensorPro if needed (preserves format state)
        if not isinstance(gaussians, GSTensorPro):
            gaussians = GSTensorPro.from_gstensor(gaussians)

        # Native API with inplace=False handles copying internally
        return gaussians.opacity(OpacityValues(scale=alpha), inplace=False)

    def apply_cpu(self, data: GSData, alpha: float) -> GSData:
        """Apply opacity scaling using native GSDataPro.opacity()."""
        if alpha == 1.0:
            return data

        # Ensure we have GSDataPro for native API
        if not isinstance(data, GSDataPro):
            data = GSDataPro.from_gsdata(data)

        # Native API with inplace=False handles copying internally
        return data.opacity(OpacityValues(scale=alpha), inplace=False)
