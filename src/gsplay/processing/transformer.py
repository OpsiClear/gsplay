"""
Scene transform adapter using gsmod's optimized transform pipelines.

Uses:
- GSDataPro.transform() for CPU processing (Numba kernels)
- GSTensorPro.transform() for GPU processing (PyTorch operations)
"""

from __future__ import annotations

from gsmod import TransformValues
from src.domain.entities import GSData, GSDataPro, GSTensor, GSTensorPro

from .protocols import SceneTransformer
import numpy as np


def _canonical_transform_values(values: TransformValues) -> TransformValues:
    """Return a new TransformValues with scalar scale and tuple fields to satisfy gsmod comparisons."""
    translate = tuple(
        float(x)
        for x in getattr(values, "translate", getattr(values, "translation", (0.0, 0.0, 0.0)))
    )
    rotate = tuple(
        float(x)
        for x in getattr(values, "rotate", getattr(values, "rotation", (1.0, 0.0, 0.0, 0.0)))
    )
    scale_arr = np.asarray(getattr(values, "scale", 1.0), dtype=np.float32).reshape(-1)
    scale_val = float(scale_arr[0])
    try:
        return TransformValues(translate=translate, rotate=rotate, scale=scale_val)
    except TypeError:
        return TransformValues(translation=translate, rotation=rotate, scale=scale_val)


class DefaultSceneTransformer(SceneTransformer):
    """Scene transformer using gsmod's optimized transform pipelines.

    For GPU: Uses GSTensorPro.transform() with GPU-accelerated operations
    For CPU: Uses GSDataPro.transform() with Numba kernels (698M Gaussians/sec)
    """

    def apply_gpu(
        self,
        gaussians: GSTensor,
        transform_values: TransformValues,
        device: str,
    ) -> GSTensor:
        """Apply scene transform on GPU using gsmod TransformValues.

        Performance: 20-50x faster than CPU version
        """
        transform_values = _canonical_transform_values(transform_values)

        # Handle array-like fields that break gsmod equality checks
        try:
            neutral = transform_values.is_neutral()
        except ValueError:
            rot = np.asarray(getattr(transform_values, "rotation", transform_values.rotate), dtype=np.float32).reshape(-1)
            trans = np.asarray(getattr(transform_values, "translation", transform_values.translate), dtype=np.float32).reshape(-1)
            scale_arr = np.asarray(transform_values.scale, dtype=np.float32).reshape(-1)
            neutral = (
                np.allclose(scale_arr[0], 1.0)
                and np.allclose(rot, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
                and np.allclose(trans, np.array([0.0, 0.0, 0.0], dtype=np.float32))
            )
        if neutral:
            return gaussians

        # Convert to GSTensorPro if needed
        # Use from_gsdata to ensure proper gsply normalization (same as CPU path)
        source_format = getattr(gaussians, '_format', None)

        if isinstance(gaussians, GSTensorPro):
            tensor_pro = gaussians
        else:
            # First convert GSTensor to GSData if needed, then to GSTensorPro
            if isinstance(gaussians, GSTensor):
                gsdata = gaussians.to_gsdata()
            else:
                gsdata = gaussians
            tensor_pro = GSTensorPro.from_gsdata(gsdata, device=device)
            # Preserve format tracking from source
            if source_format is not None:
                tensor_pro._format = source_format.copy()

        # Apply transform using gsmod's GPU-optimized operations
        # Note: Even with inplace=True, method returns GSTensorPro
        tensor_pro = tensor_pro.transform(transform_values, inplace=True)

        return tensor_pro

    def apply_cpu(
        self,
        data: GSData,
        transform_values: TransformValues,
    ) -> GSData:
        """Apply scene transform on CPU using gsmod's Numba kernels.

        Performance: 698M Gaussians/sec (1.43ms for 1M Gaussians)
        """
        transform_values = _canonical_transform_values(transform_values)

        try:
            neutral = transform_values.is_neutral()
        except ValueError:
            rot = np.asarray(getattr(transform_values, "rotation", transform_values.rotate), dtype=np.float32).reshape(-1)
            trans = np.asarray(getattr(transform_values, "translation", transform_values.translate), dtype=np.float32).reshape(-1)
            scale_arr = np.asarray(transform_values.scale, dtype=np.float32).reshape(-1)
            neutral = (
                np.allclose(scale_arr[0], 1.0)
                and np.allclose(rot, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
                and np.allclose(trans, np.array([0.0, 0.0, 0.0], dtype=np.float32))
            )
        if neutral:
            return data

        # Convert to GSDataPro for gsmod processing
        if isinstance(data, GSDataPro):
            data_pro = data
        else:
            data_pro = GSDataPro.from_gsdata(data)

        # Apply transform using gsmod's optimized Numba kernels
        data_pro.transform(transform_values, inplace=True)

        return data_pro
