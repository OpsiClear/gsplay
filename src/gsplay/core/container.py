"""
GSPlay-level container for wiring processing dependencies.

Centralises construction of the CPU/GPU processors so the rest of the codebase
can request fully-wired services (e.g. EditManager instances) without importing
infrastructure modules directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.infrastructure.processing_mode import ProcessingMode
from src.gsplay.config.settings import GSPlayConfig
from src.gsplay.state.edit_manager import EditManager
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
    GSBridge,
    OpacityAdjuster,
    ProcessingStrategy,
    SceneTransformer,
    TransformGpuStrategy,
    VolumeFilterService,
)


@dataclass
class ProcessingProviders:
    """Bundle of processing dependencies for EditManager/strategies."""

    color_processor: ColorProcessor
    scene_transformer: SceneTransformer
    opacity_adjuster: OpacityAdjuster
    volume_filter: VolumeFilterService
    gs_bridge: GSBridge
    strategies: dict[ProcessingMode, ProcessingStrategy]


def build_default_processing_providers() -> ProcessingProviders:
    """Factory for the default viewer processing stack."""
    strategies: dict[ProcessingMode, ProcessingStrategy] = {
        ProcessingMode.ALL_GPU: AllGpuStrategy(),
        ProcessingMode.ALL_CPU: AllCpuStrategy(),
        ProcessingMode.COLOR_GPU: ColorGpuStrategy(),
        ProcessingMode.COLOR_TRANSFORM_GPU: ColorTransformGpuStrategy(),
        ProcessingMode.TRANSFORM_GPU: TransformGpuStrategy(),
    }
    return ProcessingProviders(
        color_processor=DefaultColorProcessor(),
        scene_transformer=DefaultSceneTransformer(),
        opacity_adjuster=DefaultOpacityAdjuster(),
        volume_filter=VolumeFilterService(),
        gs_bridge=DefaultGSBridge(),
        strategies=strategies,
    )


def create_edit_manager(
    config: GSPlayConfig,
    device: str,
    providers: Optional[ProcessingProviders] = None,
) -> EditManager:
    """
    Build an EditManager with injected processing dependencies.

    Tests can supply custom providers (e.g. fake GSBridge implementations) to
    avoid touching GPU-accelerated components.
    """
    providers = providers or build_default_processing_providers()
    return EditManager(
        config,
        device,
        color_processor=providers.color_processor,
        scene_transformer=providers.scene_transformer,
        opacity_adjuster=providers.opacity_adjuster,
        volume_filter=providers.volume_filter,
        gs_bridge=providers.gs_bridge,
        strategies=providers.strategies,
    )


__all__ = [
    "ProcessingProviders",
    "build_default_processing_providers",
    "create_edit_manager",
]
