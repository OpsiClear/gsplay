"""
Processing strategies for the edit pipeline.
"""

from __future__ import annotations

from typing import Protocol

from src.domain.entities import GSData, GSTensor
from src.infrastructure.processing_mode import ProcessingMode
from src.shared.perf import PerfMonitor

from .context import EditContext, ProcessingResult
from .volume_filter import is_filter_active


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

    def _record_total(
        self, monitor: PerfMonitor, timings: dict[str, float]
    ) -> dict[str, float]:
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

        # GPU filtering - filter_gpu checks internally if filtering is active
        with monitor.track("filter_ms"):
            mask = context.volume_filter.filter_gpu(
                tensor, context.config, scene_bounds
            )
            if mask is not None:
                # Preserve format tracking through slicing
                # (GSTensor.__getitem__ may return plain GSTensor)
                source = tensor  # Keep reference for format copy
                tensor = tensor[mask]
                # Restore format if lost during slicing using public API
                if hasattr(tensor, "copy_format_from"):
                    result_format = getattr(tensor, "_format", None)
                    if not result_format:
                        tensor.copy_format_from(source)
                elif hasattr(source, "_format"):
                    result_format = getattr(tensor, "_format", None)
                    if not result_format:
                        tensor._format = source._format

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

        # CPU filtering - filter_cpu checks internally if filtering is active
        with monitor.track("filter_ms"):
            context.volume_filter.filter_cpu(data, context.config, scene_bounds)

        with monitor.track("transform_ms"):
            data = context.scene_transformer.apply_cpu(
                data, context.config.transform_values
            )

        with monitor.track("color_ms"):
            data = context.color_processor.apply_cpu(
                data, context.config.color_values
            )

        with monitor.track("opacity_ms"):
            data = context.opacity_adjuster.apply_cpu(
                data,
                context.config.alpha_scaler,
            )

        with monitor.track("transfer_ms"):
            # Use gsply v0.2.5 GPU loading interface - loads directly to target device
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

        # CPU filtering - filter_cpu checks internally if filtering is active
        with monitor.track("filter_ms"):
            context.volume_filter.filter_cpu(data, context.config, scene_bounds)

        with monitor.track("transfer_ms"):
            # Always create CPU tensors first, then transfer to GPU if needed
            tensor = GSTensor.from_gsdata(data, device="cpu")
            if context.device != "cpu":
                tensor = tensor.to(context.device)

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

        # CPU filtering - filter_cpu checks internally if filtering is active
        with monitor.track("filter_ms"):
            context.volume_filter.filter_cpu(data, context.config, scene_bounds)

        with monitor.track("color_ms"):
            data = context.color_processor.apply_cpu(
                data, context.config.color_values
            )

        with monitor.track("opacity_ms"):
            data = context.opacity_adjuster.apply_cpu(
                data,
                context.config.alpha_scaler,
            )

        with monitor.track("transfer_ms"):
            # Always create CPU tensors first, then transfer to GPU if needed
            tensor = GSTensor.from_gsdata(data, device="cpu")
            if context.device != "cpu":
                tensor = tensor.to(context.device)

        with monitor.track("transform_ms"):
            tensor = context.scene_transformer.apply_gpu(
                tensor,
                context.config.transform_values,
                context.device,
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

        # CPU filtering - filter_cpu checks internally if filtering is active
        with monitor.track("filter_ms"):
            context.volume_filter.filter_cpu(data, context.config, scene_bounds)

        with monitor.track("transform_ms"):
            data = context.scene_transformer.apply_cpu(
                data, context.config.transform_values
            )

        with monitor.track("transfer_ms"):
            # Always create CPU tensors first, then transfer to GPU if needed
            tensor = GSTensor.from_gsdata(data, device="cpu")
            if context.device != "cpu":
                tensor = tensor.to(context.device)

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
