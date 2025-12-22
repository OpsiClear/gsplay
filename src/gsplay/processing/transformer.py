"""Scene transform processing using gsmod's native GPU-accelerated operations.

Uses GSTensorPro.transform() for GPU and GSDataPro.transform() for CPU.
"""

from __future__ import annotations

from gsmod import TransformValues

from src.domain.entities import GSData, GSDataPro, GSTensor, GSTensorPro

from .protocols import SceneTransformer


class DefaultSceneTransformer(SceneTransformer):
    """Scene transformer using gsmod's native GPU-accelerated operations.

    GPU: GSTensorPro.transform() with PyTorch CUDA operations
    CPU: GSDataPro.transform() with Numba kernels
    """

    def apply_gpu(
        self,
        gaussians: GSTensor,
        transform_values: TransformValues,
        device: str,
    ) -> GSTensor:
        """Apply transform using native GSTensorPro.transform().

        Uses inplace=False to let gsmod handle copying internally.
        """
        if transform_values.is_neutral():
            return gaussians

        # Wrap GSTensor as GSTensorPro if needed (preserves format state)
        if not isinstance(gaussians, GSTensorPro):
            gaussians = GSTensorPro.from_gstensor(gaussians)

        # Native API with inplace=False handles copying internally
        return gaussians.transform(transform_values, inplace=False)

    def apply_cpu(
        self,
        data: GSData,
        transform_values: TransformValues,
    ) -> GSData:
        """Apply transform using native GSDataPro.transform()."""
        if transform_values.is_neutral():
            return data

        # Ensure we have GSDataPro for native API
        if not isinstance(data, GSDataPro):
            data = GSDataPro.from_gsdata(data)

        # Native API with inplace=False handles copying internally
        return data.transform(transform_values, inplace=False)
