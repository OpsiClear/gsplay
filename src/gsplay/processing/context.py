"""
Dataclasses shared across processing strategies.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.domain.entities import GSTensor
from src.gsplay.config.settings import GSPlayConfig

from .protocols import ColorProcessor, GSBridge, OpacityAdjuster, SceneTransformer
from .volume_filter import VolumeFilterService


@dataclass
class EditContext:
    config: GSPlayConfig
    device: str
    color_processor: ColorProcessor
    scene_transformer: SceneTransformer
    opacity_adjuster: OpacityAdjuster
    volume_filter: VolumeFilterService
    gaussian_bridge: GSBridge


@dataclass
class ProcessingResult:
    gaussians: GSTensor
    timings: dict[str, float]
