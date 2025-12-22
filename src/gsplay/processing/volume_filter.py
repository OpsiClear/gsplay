"""
CPU/GPU volume filtering helpers for the edit pipeline.

Uses gsmod's FilterValues for unified filtering configuration:
- GSDataPro.filter() for CPU processing (Numba kernels)
- GSTensorPro.filter() for GPU processing (PyTorch operations)

Performance (gsmod 0.1.3):
- GPU: ~1ms for 100K Gaussians, ~11ms for 1M Gaussians
- Supports: opacity, scale, sphere, box (axis-aligned + rotated), ellipsoid, frustum
- Invert mode: exclude instead of include filtering

NOTE: Filter operates on ORIGINAL (untransformed) Gaussian data. Filter values
are specified in original space and are NOT affected by scene transformations
or bake view. See strategies.py for pipeline order: FILTER → TRANSFORM.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from gsmod.config.values import FilterValues

from src.domain.entities import GSData, GSDataPro, GSTensor, GSTensorPro
from src.gsplay.config.settings import GSPlayConfig


logger = logging.getLogger(__name__)


def is_filter_active(fv: FilterValues) -> bool:
    """Check if any filtering is active in FilterValues.

    Uses gsmod's is_neutral() method for accurate detection, with
    additional check for invert mode which can affect results.

    Parameters
    ----------
    fv : FilterValues
        Filter values to check

    Returns
    -------
    bool
        True if any filtering is enabled
    """
    # Use gsmod's is_neutral() - checks all filter parameters
    # invert alone doesn't change anything, so only check when filters are active
    return not fv.is_neutral()


class VolumeFilterService:
    """Encapsulates CPU/GPU volume filtering behaviour."""

    def filter_cpu(
        self,
        data: GSData,
        config: GSPlayConfig,
        scene_bounds: dict[str, Any] | None,
    ) -> GSData:
        """Apply CPU filtering using gsmod's GSDataPro.filter().

        Uses GSDataPro's unified filter method that supports:
        - Opacity (min/max)
        - Scale (min/max)
        - Sphere
        - Box (axis-aligned and rotated via box_rot)
        - Ellipsoid (with rotation)
        - Frustum (with rotation)
        - Invert mode (exclude instead of include)

        NOTE: Filter operates on ORIGINAL (untransformed) Gaussian data.
        Filter values are in original space - not affected by scene transformations.

        Returns
        -------
        GSData
            The filtered data (GSDataPro if filtering was applied).
        """
        fv = config.filter_values
        if fv.is_neutral():
            return data

        # Filter operates directly on original (untransformed) Gaussian data
        # Filter values are in original space - not affected by scene transformations

        try:
            start_time = time.perf_counter()
            input_count = len(data.means)

            # Wrap in GSDataPro for optimized CPU filtering (matches filter_gpu pattern)
            if isinstance(data, GSDataPro):
                data_pro = data
            else:
                data_pro = GSDataPro.from_gsdata(data)

            # Use gsmod's optimized CPU filter with original space filter values
            filtered = data_pro.filter(fv, inplace=False)

            kept = len(filtered.means)
            filter_time = (time.perf_counter() - start_time) * 1000
            percentage = (kept / input_count * 100.0) if input_count else 0.0
            logger.debug(
                "[CPU Filter] %d/%d Gaussians kept (%.1f%%) in %.2fms",
                kept,
                input_count,
                percentage,
                filter_time,
            )
            return filtered

        except Exception as exc:  # pragma: no cover - defensive
            logger.error("CPU volume filter failed: %s", exc, exc_info=True)
            return data

    def filter_gpu(
        self,
        gaussians: GSTensor,
        config: GSPlayConfig,
        scene_bounds: dict[str, Any] | None,
    ) -> GSTensor | None:
        """Apply GPU filtering using gsmod's GSTensorPro.filter().

        Uses GSTensorPro's unified filter method that supports:
        - Opacity (min/max)
        - Scale (min/max)
        - Sphere
        - Box (axis-aligned and rotated via box_rot)
        - Ellipsoid (with rotation)
        - Frustum (with rotation)
        - Invert mode (exclude instead of include)

        Performance: ~1ms for 100K Gaussians, ~11ms for 1M Gaussians on GPU.

        NOTE: Filter parameters are inverse-transformed from WORLD to LOCAL
        space so that filtering on original positions matches the visualization.

        Returns
        -------
        GSTensor | None
            Filtered tensor if filtering was applied, None if no filtering needed.
        """
        fv = config.filter_values

        # Fast path: skip if no filtering configured
        if fv.is_neutral():
            return None

        # Filter operates directly on original (untransformed) Gaussian data
        # Filter values are in original space - not affected by scene transformations

        try:
            start_time = time.perf_counter()
            n_gaussians = len(gaussians.means)

            # Wrap in GSTensorPro for optimized GPU filtering
            # GSTensorPro wraps existing tensors directly (no copy, no CPU transfer)
            if isinstance(gaussians, GSTensorPro):
                tensor_pro = gaussians
            else:
                # Wrap GSTensor as GSTensorPro (preserves format state)
                tensor_pro = GSTensorPro.from_gstensor(gaussians)

            # Use gsmod's optimized GPU filter with original space filter values
            filtered = tensor_pro.filter(fv, inplace=False)

            # Return None if filtering had no effect (all Gaussians passed)
            kept = len(filtered.means)
            if kept == n_gaussians:
                return None

            filter_time = (time.perf_counter() - start_time) * 1000
            logger.debug(
                "[GPU Filter] %d/%d kept (%.1f%%) in %.2fms",
                kept,
                n_gaussians,
                (kept / n_gaussians * 100.0) if n_gaussians else 0.0,
                filter_time,
            )
            return filtered

        except Exception as exc:
            logger.error("GPU volume filter failed: %s", exc, exc_info=True)
            return None
