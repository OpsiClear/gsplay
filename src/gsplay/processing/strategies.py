"""
Processing strategies for the edit pipeline.

Operation Order:
1. FILTER - spatial selection on original positions (reduces data early)
2. TRANSFORM - spatial positioning of filtered data
3. COLOR - appearance adjustments
4. OPACITY - transparency adjustments

This ensures filtering operates on original asset positions before any
transforms are applied, reducing data volume early in the pipeline.

CPU/GPU Consistency:
- ALL_GPU: Filter(GPU) -> Transform(GPU) -> Color(GPU) -> Opacity(GPU)
- ALL_CPU: Filter(CPU) -> Transform(CPU) -> Color(CPU) -> Opacity(CPU) -> Transfer
- COLOR_TRANSFORM_GPU: Filter(CPU) -> Transfer -> Transform(GPU) -> Color(GPU) -> Opacity(GPU)
- TRANSFORM_GPU: Filter(CPU) -> Transfer -> Transform(GPU) -> Color(GPU) -> Opacity(GPU)
- COLOR_GPU: Filter(CPU) -> Transform(CPU) -> Transfer -> Color(GPU) -> Opacity(GPU)
"""

from __future__ import annotations

from typing import Protocol

from gsmod import GSDataPro
from gsmod.torch import GSTensorPro

from src.domain.entities import GSData, GSTensor
from src.infrastructure.processing_mode import ProcessingMode
from src.shared.perf import PerfMonitor

from .context import EditContext, ProcessingResult


class ProcessingStrategy(Protocol):
    """Base contract implemented by each processing mode strategy."""

    mode: ProcessingMode

    def apply(
        self,
        context: EditContext,
        gaussians: GSData | GSTensor,
        scene_bounds: dict[str, object] | None,
    ) -> ProcessingResult: ...


class _BaseStrategy:
    """Utility mixin shared by the concrete strategies."""

    def _record_total(self, monitor: PerfMonitor, timings: dict[str, float]) -> dict[str, float]:
        stage_timings, total_ms = monitor.stop()
        stage_timings.update(timings)
        stage_timings["total_ms"] = total_ms
        return stage_timings


class AllGpuStrategy(_BaseStrategy, ProcessingStrategy):
    mode = ProcessingMode.ALL_GPU

    def apply(
        self,
        context: EditContext,
        gaussians: GSData | GSTensor,
        scene_bounds: dict[str, object] | None,
    ) -> ProcessingResult:
        monitor = PerfMonitor(self.mode.value)
        tensor, transfer_ms = context.gaussian_bridge.ensure_tensor_on_device(
            gaussians, context.device
        )
        monitor.record("transfer_ms", transfer_ms)

        # FILTER FIRST - reduces data before transform
        with monitor.track("filter_ms"):
            filtered = context.volume_filter.filter_gpu(tensor, context.config, scene_bounds)
            if filtered is not None:
                tensor = filtered

        # Transform on filtered data
        with monitor.track("transform_ms"):
            tensor = context.scene_transformer.apply_gpu(
                tensor,
                context.config.transform_values,
                context.device,
            )

        with monitor.track("color_ms"):
            tensor = context.color_processor.apply_gpu(
                tensor,
                context.config.color_values,
                context.device,
            )

        with monitor.track("opacity_ms"):
            tensor = context.opacity_adjuster.apply_gpu(
                tensor,
                context.config.alpha_scaler,
            )

        timings = self._record_total(monitor, {})
        return ProcessingResult(tensor, timings)


class AllCpuStrategy(_BaseStrategy, ProcessingStrategy):
    mode = ProcessingMode.ALL_CPU

    def apply(
        self,
        context: EditContext,
        gaussians: GSData | GSTensor,
        scene_bounds: dict[str, object] | None,
    ) -> ProcessingResult:
        monitor = PerfMonitor(self.mode.value)
        data = context.gaussian_bridge.ensure_gsdata(gaussians)

        # FILTER FIRST - reduces data before transform
        with monitor.track("filter_ms"):
            data = context.volume_filter.filter_cpu(data, context.config, scene_bounds)

        # Transform on filtered data
        with monitor.track("transform_ms"):
            data = context.scene_transformer.apply_cpu(data, context.config.transform_values)

        with monitor.track("color_ms"):
            data = context.color_processor.apply_cpu(data, context.config.color_values)

        with monitor.track("opacity_ms"):
            data = context.opacity_adjuster.apply_cpu(
                data,
                context.config.alpha_scaler,
            )

        with monitor.track("transfer_ms"):
            # Use appropriate conversion based on data type:
            # GSDataPro (from color/opacity ops) -> GSTensorPro
            # GSData -> GSTensor
            if isinstance(data, GSDataPro):
                tensor = GSTensorPro.from_gsdata(data, device=context.device)
            else:
                tensor = GSTensor.from_gsdata(data, device=context.device)

        timings = self._record_total(monitor, {})
        return ProcessingResult(tensor, timings)


class ColorTransformGpuStrategy(_BaseStrategy, ProcessingStrategy):
    mode = ProcessingMode.COLOR_TRANSFORM_GPU

    def apply(
        self,
        context: EditContext,
        gaussians: GSData | GSTensor,
        scene_bounds: dict[str, object] | None,
    ) -> ProcessingResult:
        monitor = PerfMonitor(self.mode.value)
        data = context.gaussian_bridge.ensure_gsdata(gaussians)

        # FILTER FIRST on CPU - reduces data before transfer
        with monitor.track("filter_ms"):
            data = context.volume_filter.filter_cpu(data, context.config, scene_bounds)

        with monitor.track("transfer_ms"):
            # Use appropriate conversion based on data type
            if isinstance(data, GSDataPro):
                tensor = GSTensorPro.from_gsdata(data, device=context.device)
            else:
                tensor = GSTensor.from_gsdata(data, device=context.device)

        # Transform on GPU (filtered data)
        with monitor.track("transform_ms"):
            tensor = context.scene_transformer.apply_gpu(
                tensor,
                context.config.transform_values,
                context.device,
            )

        with monitor.track("color_ms"):
            tensor = context.color_processor.apply_gpu(
                tensor,
                context.config.color_values,
                context.device,
            )

        with monitor.track("opacity_ms"):
            tensor = context.opacity_adjuster.apply_gpu(
                tensor,
                context.config.alpha_scaler,
            )

        timings = self._record_total(monitor, {})
        return ProcessingResult(tensor, timings)


class TransformGpuStrategy(_BaseStrategy, ProcessingStrategy):
    mode = ProcessingMode.TRANSFORM_GPU

    def apply(
        self,
        context: EditContext,
        gaussians: GSData | GSTensor,
        scene_bounds: dict[str, object] | None,
    ) -> ProcessingResult:
        monitor = PerfMonitor(self.mode.value)
        data = context.gaussian_bridge.ensure_gsdata(gaussians)

        # FILTER FIRST on CPU - reduces data before transfer
        with monitor.track("filter_ms"):
            data = context.volume_filter.filter_cpu(data, context.config, scene_bounds)

        with monitor.track("transfer_ms"):
            # Transfer filtered data to target device
            # Use appropriate conversion based on data type
            if isinstance(data, GSDataPro):
                tensor = GSTensorPro.from_gsdata(data, device=context.device)
            else:
                tensor = GSTensor.from_gsdata(data, device=context.device)

        # Transform on GPU (step 2)
        with monitor.track("transform_ms"):
            tensor = context.scene_transformer.apply_gpu(
                tensor,
                context.config.transform_values,
                context.device,
            )

        # Color on GPU (step 3)
        with monitor.track("color_ms"):
            tensor = context.color_processor.apply_gpu(
                tensor,
                context.config.color_values,
                context.device,
            )

        # Opacity on GPU (step 4)
        with monitor.track("opacity_ms"):
            tensor = context.opacity_adjuster.apply_gpu(
                tensor,
                context.config.alpha_scaler,
            )

        timings = self._record_total(monitor, {})
        return ProcessingResult(tensor, timings)


class ColorGpuStrategy(_BaseStrategy, ProcessingStrategy):
    mode = ProcessingMode.COLOR_GPU

    def apply(
        self,
        context: EditContext,
        gaussians: GSData | GSTensor,
        scene_bounds: dict[str, object] | None,
    ) -> ProcessingResult:
        monitor = PerfMonitor(self.mode.value)
        data = context.gaussian_bridge.ensure_gsdata(gaussians)

        # FILTER FIRST on CPU - reduces data before transform
        with monitor.track("filter_ms"):
            data = context.volume_filter.filter_cpu(data, context.config, scene_bounds)

        # Transform on CPU (filtered data)
        with monitor.track("transform_ms"):
            data = context.scene_transformer.apply_cpu(data, context.config.transform_values)

        with monitor.track("transfer_ms"):
            # Transfer directly to target device
            # Use appropriate conversion based on data type
            if isinstance(data, GSDataPro):
                tensor = GSTensorPro.from_gsdata(data, device=context.device)
            else:
                tensor = GSTensor.from_gsdata(data, device=context.device)

        with monitor.track("color_ms"):
            tensor = context.color_processor.apply_gpu(
                tensor,
                context.config.color_values,
                context.device,
            )

        with monitor.track("opacity_ms"):
            tensor = context.opacity_adjuster.apply_gpu(
                tensor,
                context.config.alpha_scaler,
            )

        timings = self._record_total(monitor, {})
        return ProcessingResult(tensor, timings)
