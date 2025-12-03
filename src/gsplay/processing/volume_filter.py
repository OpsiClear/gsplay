"""
CPU/GPU volume filtering helpers for the edit pipeline.

Uses gsmod's FilterValues for unified filtering configuration:
- GSDataPro.filter() for CPU processing (Numba kernels)
- GSTensorPro.filter() for GPU processing (PyTorch operations)

Performance (gsmod 0.1.3):
- GPU: ~1ms for 100K Gaussians, ~11ms for 1M Gaussians
- Supports: opacity, scale, sphere, box (axis-aligned + rotated), ellipsoid, frustum
- Invert mode: exclude instead of include filtering
"""

from __future__ import annotations

import logging
import time
from typing import Any

from gsmod.config.values import FilterValues
from src.domain.entities import GSData, GSTensor, GSTensorPro
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
        """Apply CPU filtering using gsmod's compute_filter_mask with FilterValues.

        This supports all filter types including rotated box, ellipsoid, and frustum.

        Returns
        -------
        GSData
            The filtered data (may be modified in-place, but returned for API consistency
            with filter_gpu which returns a new GSTensor).
        """
        fv = config.filter_values
        if fv.is_neutral():
            return data

        try:
            from gsmod.filter.apply import compute_filter_mask
        except ImportError as exc:
            logger.error("gsmod filter module unavailable: %s", exc)
            return data

        try:
            start_time = time.perf_counter()
            input_count = len(data.means)

            # Use gsmod's compute_filter_mask which supports all FilterValues params
            mask = compute_filter_mask(data, fv)

            # Apply mask to data in-place
            data.means = data.means[mask]
            data.scales = data.scales[mask]
            data.quats = data.quats[mask]
            data.opacities = data.opacities[mask]
            data.sh0 = data.sh0[mask]
            if data.shN is not None:
                data.shN = data.shN[mask]

            kept = len(data.means)
            filter_time = (time.perf_counter() - start_time) * 1000
            percentage = (kept / input_count * 100.0) if input_count else 0.0
            logger.debug(
                "[CPU Filter] %d/%d Gaussians kept (%.1f%%) in %.2fms",
                kept,
                input_count,
                percentage,
                filter_time,
            )
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

        Returns
        -------
        GSTensor | None
            Filtered tensor if filtering was applied, None if no filtering needed.
        """
        fv = config.filter_values

        # Fast path: skip if no filtering configured
        if fv.is_neutral():
            return None

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

            # Use gsmod's optimized GPU filter (inplace=False for non-destructive)
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
