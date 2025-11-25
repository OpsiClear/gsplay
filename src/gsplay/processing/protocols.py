"""
Protocols that describe pluggable processors used by the edit pipeline.
"""

from __future__ import annotations

from typing import Protocol

from gsmod import ColorValues, TransformValues
from src.domain.entities import GSData, GSTensor


class GSBridge(Protocol):
    """Abstraction that owns GSData/GSTensor conversions across devices."""

    def ensure_gsdata(self, gaussians: GSData | GSTensor) -> GSData: ...

    def ensure_tensor_on_device(
        self,
        gaussians: GSData | GSTensor,
        device: str,
    ) -> tuple[GSTensor, float]: ...


class ColorProcessor(Protocol):
    """Protocol for color processing across devices."""

    def apply_gpu(
        self,
        gaussians: GSTensor,
        color_values: ColorValues,
        device: str,
    ) -> GSTensor:
        """Apply color adjustments on GPU."""
        ...

    def apply_cpu(
        self,
        data: GSData,
        color_values: ColorValues,
    ) -> GSData:
        """Apply color adjustments on CPU."""
        ...


class SceneTransformer(Protocol):
    """Protocol for scene transformation across devices."""

    def apply_gpu(
        self,
        gaussians: GSTensor,
        transform_values: TransformValues,
        device: str,
    ) -> GSTensor:
        """Apply scene transform on GPU."""
        ...

    def apply_cpu(
        self,
        data: GSData,
        transform_values: TransformValues,
    ) -> GSData:
        """Apply scene transform on CPU."""
        ...


class OpacityAdjuster(Protocol):
    """Abstraction for opacity scaling on CPU or GPU."""

    def apply_gpu(self, gaussians: GSTensor, alpha: float) -> GSTensor: ...

    def apply_cpu(self, data: GSData, alpha: float) -> GSData: ...
