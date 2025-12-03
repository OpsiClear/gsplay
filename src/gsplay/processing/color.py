"""Color processing using gsmod's native Triton-accelerated operations.

Uses GSTensorPro.color() which internally uses Triton kernels for
brightness/saturation adjustments when available.
"""

from __future__ import annotations

from gsmod import ColorValues
from src.domain.entities import GSData, GSDataPro, GSTensor, GSTensorPro

from .protocols import ColorProcessor


class DefaultColorProcessor(ColorProcessor):
    """Color processor using gsmod's native Triton-accelerated operations.

    GPU: GSTensorPro.color() with Triton kernels for brightness/saturation
    CPU: GSDataPro.color() with Numba LUT kernels
    """

    def apply_gpu(
        self,
        gaussians: GSTensor,
        color_values: ColorValues,
        device: str,
    ) -> GSTensor:
        """Apply color adjustments using native GSTensorPro.color().

        Uses inplace=False to let gsmod handle copying internally.
        """
        if color_values.is_neutral():
            return gaussians

        # Wrap GSTensor as GSTensorPro if needed (preserves format state)
        if not isinstance(gaussians, GSTensorPro):
            gaussians = GSTensorPro.from_gstensor(gaussians)

        # Native API with inplace=False handles copying internally
        return gaussians.color(color_values, inplace=False)

    def apply_cpu(
        self,
        data: GSData,
        color_values: ColorValues,
    ) -> GSData:
        """Apply color adjustments using native GSDataPro.color()."""
        if color_values.is_neutral():
            return data

        # Ensure we have GSDataPro for native API
        if not isinstance(data, GSDataPro):
            data = GSDataPro.from_gsdata(data)

        # Native API with inplace=False handles copying internally
        return data.color(color_values, inplace=False)
