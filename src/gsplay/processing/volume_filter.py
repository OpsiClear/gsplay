"""
CPU/GPU volume filtering helpers for the edit pipeline.

Uses gsmod's FilterValues for unified filtering configuration:
- GSDataPro.filter() for CPU processing (Numba kernels)
- GSTensorPro.filter() for GPU processing (PyTorch operations)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

from gsmod.config.values import FilterValues
from src.domain.entities import GSData, GSTensor, GSTensorPro
from src.gsplay.config.settings import GSPlayConfig

logger = logging.getLogger(__name__)


def is_filter_active(fv: FilterValues) -> bool:
    """Check if any filtering is active in FilterValues.

    Parameters
    ----------
    fv : FilterValues
        Filter values to check

    Returns
    -------
    bool
        True if any filtering is enabled
    """
    return (
        fv.min_opacity > 0.0
        or fv.max_opacity < 1.0
        or fv.min_scale > 0.0
        or fv.max_scale < 100.0
        or fv.sphere_radius < float("inf")
        or fv.box_min is not None
        or fv.ellipsoid_radii is not None
        or fv.frustum_pos is not None
    )


class VolumeFilterService:
    """Encapsulates CPU/GPU volume filtering behaviour."""

    def __init__(self) -> None:
        self._pipeline = None
        self._pipeline_key: tuple | None = None

    def filter_cpu(
        self,
        data: GSData,
        config: GSPlayConfig,
        scene_bounds: dict[str, Any] | None,
    ) -> None:
        """Apply CPU filtering using gsmod's compute_filter_mask with FilterValues.

        This supports all filter types including rotated box, ellipsoid, and frustum.
        """
        fv = config.filter_values
        if not is_filter_active(fv):
            return

        try:
            from gsmod.filter.apply import compute_filter_mask
        except ImportError as exc:
            logger.error("gsmod filter module unavailable: %s", exc)
            return

        try:
            import time

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

    def filter_gpu(
        self,
        gaussians: GSTensor,
        config: GSPlayConfig,
        scene_bounds: dict[str, Any] | None,
    ) -> torch.Tensor | None:
        """Apply GPU filtering using GSTensorPro.filter() with FilterValues.

        Performance: 50-100x faster than CPU version

        Uses config.filter_values (updated from UI) for filtering parameters.
        """
        # Use filter_values from config (updated from UI)
        fv = config.filter_values

        # Check if any filtering is active (use helper function)
        if not is_filter_active(fv):
            return None

        device = gaussians.means.device
        dtype = gaussians.means.dtype

        try:
            import time

            start_time = time.perf_counter()
            n_gaussians = len(gaussians.means)

            # Convert to GSTensorPro if needed for filter() method
            # Use GSTensorPro which has format-aware filter methods
            if isinstance(gaussians, GSTensorPro):
                tensor_pro = gaussians
            else:
                # Wrap in GSTensorPro - note: format tracking may not be preserved
                # The is_opacities_ply/is_scales_ply properties rely on _format dict
                tensor_pro = GSTensorPro(
                    means=gaussians.means,
                    scales=gaussians.scales,
                    quats=gaussians.quats,
                    opacities=gaussians.opacities,
                    sh0=gaussians.sh0,
                    shN=gaussians.shN,
                )
                # Copy format using public API
                if hasattr(gaussians, "copy_format_from"):
                    tensor_pro.copy_format_from(gaussians)
                elif hasattr(gaussians, "_format"):
                    # Fallback for older gsply versions
                    tensor_pro._format = gaussians._format.copy()

            # Build mask directly using GSTensorPro's format-aware filter methods
            # This is more efficient than cloning and filtering
            mask = torch.ones(n_gaussians, dtype=torch.bool, device=device)

            # Min opacity filtering - use GSTensorPro's format-aware method
            if fv.min_opacity > 0:
                # filter_min_opacity handles is_opacities_ply internally
                opacity_mask = tensor_pro.filter_min_opacity(fv.min_opacity)
                # Ensure mask is flattened to match
                if opacity_mask.dim() > 1:
                    opacity_mask = opacity_mask.flatten()
                mask &= opacity_mask

            # Max opacity filtering
            if fv.max_opacity < 1.0:
                # Get actual opacity values (handle PLY format)
                if hasattr(tensor_pro, "is_opacities_ply") and tensor_pro.is_opacities_ply:
                    # Convert from logit to sigmoid
                    opacities = torch.sigmoid(tensor_pro.opacities)
                else:
                    opacities = tensor_pro.opacities
                if opacities.dim() > 1:
                    opacities = opacities.flatten()
                mask &= opacities <= fv.max_opacity

            # Min scale filtering
            if fv.min_scale > 0:
                # Get actual scale values (handle PLY format)
                if hasattr(tensor_pro, "is_scales_ply") and tensor_pro.is_scales_ply:
                    # Convert from log to exp
                    scales = torch.exp(tensor_pro.scales)
                else:
                    scales = tensor_pro.scales
                max_scales = scales.max(dim=1).values
                mask &= max_scales >= fv.min_scale

            # Max scale filtering - use GSTensorPro's format-aware method
            if fv.max_scale < 100.0:
                # filter_max_scale handles is_scales_ply internally
                scale_mask = tensor_pro.filter_max_scale(fv.max_scale)
                mask &= scale_mask

            # Sphere filtering
            if fv.sphere_radius < float("inf"):
                center = torch.tensor(
                    fv.sphere_center, dtype=dtype, device=device
                )
                distances = torch.norm(tensor_pro.means - center, dim=1)
                mask &= distances <= fv.sphere_radius

            # Box filtering (with rotation support)
            if fv.box_min is not None and fv.box_max is not None:
                import math as _math
                box_min = torch.tensor(fv.box_min, dtype=dtype, device=device)
                box_max = torch.tensor(fv.box_max, dtype=dtype, device=device)
                box_center = (box_min + box_max) / 2
                box_half_size = (box_max - box_min) / 2

                # Transform points to local box space
                local_points = tensor_pro.means - box_center

                # Apply inverse rotation if present
                if fv.box_rot is not None:
                    ax, ay, az = fv.box_rot
                    angle = _math.sqrt(ax * ax + ay * ay + az * az)
                    if angle > 1e-6:
                        # Build rotation matrix using Rodrigues' formula
                        axis = torch.tensor(
                            [ax / angle, ay / angle, az / angle],
                            dtype=dtype, device=device,
                        )
                        K = torch.tensor([
                            [0, -axis[2], axis[1]],
                            [axis[2], 0, -axis[0]],
                            [-axis[1], axis[0], 0],
                        ], dtype=dtype, device=device)
                        R = (
                            torch.eye(3, dtype=dtype, device=device)
                            + _math.sin(angle) * K
                            + (1 - _math.cos(angle)) * (K @ K)
                        )
                        # Apply inverse rotation (R^T) to transform to local space
                        local_points = local_points @ R

                # Check if points are inside axis-aligned box in local space
                mask &= torch.all(torch.abs(local_points) <= box_half_size, dim=1)

            # Ellipsoid filtering (with rotation support)
            if fv.ellipsoid_radii is not None and fv.ellipsoid_center is not None:
                import math as _math
                center = torch.tensor(fv.ellipsoid_center, dtype=dtype, device=device)
                radii = torch.tensor(fv.ellipsoid_radii, dtype=dtype, device=device)

                # Transform points to local ellipsoid space
                local_points = tensor_pro.means - center

                # Apply inverse rotation if present
                if fv.ellipsoid_rot is not None:
                    ax, ay, az = fv.ellipsoid_rot
                    angle = _math.sqrt(ax * ax + ay * ay + az * az)
                    if angle > 1e-6:
                        # Build rotation matrix using Rodrigues' formula
                        axis = torch.tensor(
                            [ax / angle, ay / angle, az / angle],
                            dtype=dtype, device=device,
                        )
                        K = torch.tensor([
                            [0, -axis[2], axis[1]],
                            [axis[2], 0, -axis[0]],
                            [-axis[1], axis[0], 0],
                        ], dtype=dtype, device=device)
                        R = (
                            torch.eye(3, dtype=dtype, device=device)
                            + _math.sin(angle) * K
                            + (1 - _math.cos(angle)) * (K @ K)
                        )
                        # Apply inverse rotation (R^T) to transform to local space
                        local_points = local_points @ R

                # Normalized distance in local space: sum((p / radii)^2) <= 1
                normalized_dist_sq = (local_points ** 2 / (radii ** 2 + 1e-8)).sum(dim=1)
                mask &= normalized_dist_sq <= 1.0

            # Frustum filtering (camera view culling)
            if fv.frustum_pos is not None:
                import math

                # Camera position
                cam_pos = torch.tensor(
                    fv.frustum_pos, dtype=dtype, device=device
                )

                # Build rotation matrix from axis-angle
                if fv.frustum_rot is not None:
                    ax, ay, az = fv.frustum_rot
                    angle = math.sqrt(ax * ax + ay * ay + az * az)
                    if angle > 1e-6:
                        # Rodrigues' rotation formula
                        axis = torch.tensor(
                            [ax / angle, ay / angle, az / angle],
                            dtype=dtype,
                            device=device,
                        )
                        K = torch.tensor(
                            [
                                [0, -axis[2], axis[1]],
                                [axis[2], 0, -axis[0]],
                                [-axis[1], axis[0], 0],
                            ],
                            dtype=dtype,
                            device=device,
                        )
                        R = (
                            torch.eye(3, dtype=dtype, device=device)
                            + math.sin(angle) * K
                            + (1 - math.cos(angle)) * (K @ K)
                        )
                    else:
                        R = torch.eye(3, dtype=dtype, device=device)
                else:
                    R = torch.eye(3, dtype=dtype, device=device)

                # Transform points to camera space (world-to-local)
                # Convention: camera looks along -Z, X is right, Y is up
                # Use @ R (not @ R.T) to apply inverse rotation, consistent with box/ellipsoid
                points_cam = (tensor_pro.means - cam_pos) @ R

                # Extract camera space coordinates
                x_cam = points_cam[:, 0]
                y_cam = points_cam[:, 1]
                z_cam = points_cam[:, 2]

                # Depth check: must be in front of camera (negative z in camera space)
                # Near/far clipping
                depth_mask = (-z_cam >= fv.frustum_near) & (-z_cam <= fv.frustum_far)

                # Horizontal FOV check
                # fov is vertical FOV, compute horizontal from aspect
                tan_half_fov_y = math.tan(fv.frustum_fov / 2.0)
                tan_half_fov_x = tan_half_fov_y * fv.frustum_aspect

                # Points within FOV cone (avoid division by zero)
                z_safe = torch.clamp(-z_cam, min=1e-6)
                x_ratio = torch.abs(x_cam) / z_safe
                y_ratio = torch.abs(y_cam) / z_safe

                fov_mask = (x_ratio <= tan_half_fov_x) & (y_ratio <= tan_half_fov_y)

                mask &= depth_mask & fov_mask

            filter_time = (time.perf_counter() - start_time) * 1000
            inside_count = mask.sum().item()
            logger.debug(
                "[GPU Filter] %d/%d Gaussians kept (%.1f%%) in %.2fms",
                inside_count,
                len(mask),
                (inside_count / len(mask) * 100.0) if len(mask) else 0.0,
                filter_time,
            )
            return mask
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("GPU volume filter failed: %s", exc, exc_info=True)
            return None

    def _get_or_create_pipeline(
        self,
        config: GSPlayConfig,
        scene_bounds: dict[str, Any],
    ):
        from gsmod import SceneBounds
        from gsmod.filter.api import _apply_filter

        vf = config.volume_filter
        key = (
            vf.filter_type,
            tuple(float(x) for x in vf.sphere_center),
            float(vf.sphere_radius_factor),
            tuple(float(x) for x in vf.cuboid_center),
            float(vf.cuboid_size_factor_x),
            float(vf.cuboid_size_factor_y),
            float(vf.cuboid_size_factor_z),
            float(vf.opacity_threshold),
            float(vf.max_scale),
            tuple(float(x) for x in scene_bounds["min_coords"]),
            tuple(float(x) for x in scene_bounds["max_coords"]),
        )

        if self._pipeline is not None and self._pipeline_key == key:
            return self._pipeline

        bounds = SceneBounds(
            min=np.asarray(scene_bounds["min_coords"], dtype=np.float32),
            max=np.asarray(scene_bounds["max_coords"], dtype=np.float32),
        )

        # Calculate absolute values from normalized factors
        bounds_size = bounds.max - bounds.min
        sphere_radius_abs = np.linalg.norm(bounds_size) / 2.0 * vf.sphere_radius_factor
        cuboid_size_abs = (
            bounds_size[0] * vf.cuboid_size_factor_x,
            bounds_size[1] * vf.cuboid_size_factor_y,
            bounds_size[2] * vf.cuboid_size_factor_z,
        )

        # Create a simple wrapper object that stores parameters and applies filter
        class FilterPipeline:
            def __init__(
                self,
                filter_type: str,
                sphere_center: tuple,
                sphere_radius: float,
                cuboid_center: tuple,
                cuboid_size: tuple,
                opacity_threshold: float,
                max_scale: float,
            ):
                self.filter_type = filter_type
                self.sphere_center = sphere_center
                self.sphere_radius = sphere_radius
                self.cuboid_center = cuboid_center
                self.cuboid_size = cuboid_size
                self.opacity_threshold = opacity_threshold
                self.max_scale = max_scale

            def apply(self, data, inplace: bool = True):
                positions = np.asarray(data.means, dtype=np.float32)
                opacities = np.asarray(data.opacities, dtype=np.float32).flatten()
                scales = np.asarray(data.scales, dtype=np.float32)

                # Compute filter mask using gsmod
                mask = _apply_filter(
                    positions=positions,
                    opacities=opacities,
                    scales=scales,
                    filter_type=self.filter_type if self.filter_type != "none" else "none",
                    sphere_center=self.sphere_center,
                    sphere_radius=self.sphere_radius,
                    cuboid_center=self.cuboid_center,
                    cuboid_size=self.cuboid_size,
                    opacity_threshold=self.opacity_threshold,
                    max_scale=self.max_scale,
                )

                # Apply mask to data
                if inplace:
                    data.means = data.means[mask]
                    data.scales = data.scales[mask]
                    data.quats = data.quats[mask]
                    data.opacities = data.opacities[mask]
                    data.sh0 = data.sh0[mask]
                    if data.shN is not None:
                        data.shN = data.shN[mask]
                    return data
                else:
                    filtered = data.copy()
                    filtered.means = data.means[mask]
                    filtered.scales = data.scales[mask]
                    filtered.quats = data.quats[mask]
                    filtered.opacities = data.opacities[mask]
                    filtered.sh0 = data.sh0[mask]
                    if data.shN is not None:
                        filtered.shN = data.shN[mask]
                    return filtered

        pipeline = FilterPipeline(
            filter_type=vf.filter_type,
            sphere_center=vf.sphere_center,
            sphere_radius=sphere_radius_abs,
            cuboid_center=vf.cuboid_center,
            cuboid_size=cuboid_size_abs,
            opacity_threshold=vf.opacity_threshold,
            max_scale=vf.max_scale,
        )

        self._pipeline = pipeline
        self._pipeline_key = key
        return pipeline
