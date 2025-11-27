"""
Color processing adapter using gsmod's optimized color pipelines.

Uses:
- GSDataPro.color() for CPU processing (Numba LUT kernels)
- GSTensorPro.color() for GPU processing (PyTorch operations)
"""

from __future__ import annotations

from gsmod import ColorValues
from src.domain.entities import GSData, GSDataPro, GSTensor, GSTensorPro

from .protocols import ColorProcessor


class DefaultColorProcessor(ColorProcessor):
    """Color processor using gsmod for both CPU and GPU.

    For GPU: Uses GSTensorPro.color() with GPU-accelerated operations
    For CPU: Uses GSDataPro.color() with Numba LUT kernels (1015M Gaussians/sec)
    """

    def apply_gpu(
        self,
        gaussians: GSTensor,
        color_values: ColorValues,
        device: str,
    ) -> GSTensor:
        """Apply color adjustments on GPU using gsmod GSTensorPro.color().

        Performance: 50-100x faster than CPU version
        Uses gsmod's GPU-optimized color operations with automatic RGB conversion.
        """
        # Skip if neutral (no adjustments needed)
        if color_values.is_neutral():
            return gaussians

        # Convert to GSTensorPro for gsmod processing
        # Convert to GSTensorPro if needed - avoid CPU round-trips
        source_format = getattr(gaussians, '_format', None)

        if isinstance(gaussians, GSTensorPro):
            tensor_pro = gaussians
        elif isinstance(gaussians, GSTensor):
            # Direct GPU wrapping - no CPU transfer needed
            current_device = str(gaussians.means.device)
            if current_device != device:
                gaussians = gaussians.to(device)
            tensor_pro = GSTensorPro(
                means=gaussians.means,
                scales=gaussians.scales,
                quats=gaussians.quats,
                opacities=gaussians.opacities,
                sh0=gaussians.sh0,
                shN=gaussians.shN,
            )
            # Preserve format tracking from source
            if source_format is not None:
                tensor_pro._format = source_format.copy()
        else:
            # GSData path - single CPU->GPU transfer
            tensor_pro = GSTensorPro.from_gsdata(gaussians, device=device)
            if source_format is not None:
                tensor_pro._format = source_format.copy()

        # Apply color adjustments using gsmod's GPU-optimized operations
        # Note: Even with inplace=True, method returns GSTensorPro
        tensor_pro = tensor_pro.color(color_values, inplace=True)

        return tensor_pro

    def apply_cpu(
        self,
        data: GSData,
        color_values: ColorValues,
    ) -> GSData:
        """Apply color adjustments on CPU using gsmod's Numba LUT kernels.

        Performance: 1015M Gaussians/sec (0.10ms for 100K Gaussians)
        """
        # Skip if neutral (no adjustments needed)
        if color_values.is_neutral():
            return data

        # Convert to GSDataPro for gsmod processing
        if isinstance(data, GSDataPro):
            data_pro = data
        else:
            data_pro = GSDataPro.from_gsdata(data)

        # Apply color adjustments using gsmod's optimized Numba kernels
        data_pro.color(color_values, inplace=True)

        return data_pro
