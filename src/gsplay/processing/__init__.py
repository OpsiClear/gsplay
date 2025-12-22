"""
Processing subsystem for the viewer edit pipeline.
"""

from .color import DefaultColorProcessor
from .context import EditContext, ProcessingResult
from .gs_bridge import DefaultGSBridge
from .opacity import DefaultOpacityAdjuster
from .protocols import ColorProcessor, GSBridge, OpacityAdjuster, SceneTransformer
from .strategies import (
    AllCpuStrategy,
    AllGpuStrategy,
    ColorGpuStrategy,
    ColorTransformGpuStrategy,
    ProcessingStrategy,
    TransformGpuStrategy,
)
from .transformer import DefaultSceneTransformer
from .volume_filter import VolumeFilterService


__all__ = [
    "AllCpuStrategy",
    "AllGpuStrategy",
    "ColorGpuStrategy",
    "ColorProcessor",
    "ColorTransformGpuStrategy",
    "DefaultColorProcessor",
    "DefaultGSBridge",
    "DefaultOpacityAdjuster",
    "DefaultSceneTransformer",
    "EditContext",
    "GSBridge",
    "OpacityAdjuster",
    "ProcessingResult",
    "ProcessingStrategy",
    "SceneTransformer",
    "TransformGpuStrategy",
    "VolumeFilterService",
]
