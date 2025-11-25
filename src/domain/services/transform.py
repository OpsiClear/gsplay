"""
Transform services for Gaussian Splatting data.

Provides pure geometric operations that belong to the domain layer and are
agnostic of infrastructure or presentation concerns.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
from gsply import GSTensor

if TYPE_CHECKING:
    from src.domain.entities import SceneBounds
    from gsmod.config.values import TransformValues

logger = logging.getLogger(__name__)


class TransformService:
    """
    Service for applying geometric transformations to Gaussian data.

    All methods are pure functions that return new GSTensor without modifying inputs.
    """

    @staticmethod
    def get_rotation_matrix(
        rotation_angles: tuple[float, float, float], device: str = "cpu"
    ) -> torch.Tensor:
        """
        Compute 3D rotation matrix from Euler angles (ZYX order).

        :param rotation_angles: Rotation angles (rx, ry, rz) in degrees
        :param device: Torch device for the output matrix
        :return: 3x3 rotation matrix [3, 3]
        """
        rx, ry, rz = rotation_angles

        # Convert to radians
        rx_rad, ry_rad, rz_rad = np.radians(rx), np.radians(ry), np.radians(rz)

        # Build rotation matrices for each axis
        Rx = torch.tensor(
            [
                [1, 0, 0],
                [0, np.cos(rx_rad), -np.sin(rx_rad)],
                [0, np.sin(rx_rad), np.cos(rx_rad)],
            ],
            dtype=torch.float32,
            device=device,
        )

        Ry = torch.tensor(
            [
                [np.cos(ry_rad), 0, np.sin(ry_rad)],
                [0, 1, 0],
                [-np.sin(ry_rad), 0, np.cos(ry_rad)],
            ],
            dtype=torch.float32,
            device=device,
        )

        Rz = torch.tensor(
            [
                [np.cos(rz_rad), -np.sin(rz_rad), 0],
                [np.sin(rz_rad), np.cos(rz_rad), 0],
                [0, 0, 1],
            ],
            dtype=torch.float32,
            device=device,
        )

        # Combine in ZYX order
        R = Rz @ Ry @ Rx
        return R

    @staticmethod
    def rotation_matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
        """
        Convert rotation matrix to quaternion (w, x, y, z).

        :param R: 3x3 rotation matrix
        :return: Normalized quaternion [4] (w, x, y, z)
        """
        device = R.device

        # Shepperd's method for numerical stability
        qw = torch.sqrt(1.0 + R[0, 0] + R[1, 1] + R[2, 2]) / 2.0
        qx = (R[2, 1] - R[1, 2]) / (4.0 * qw)
        qy = (R[0, 2] - R[2, 0]) / (4.0 * qw)
        qz = (R[1, 0] - R[0, 1]) / (4.0 * qw)

        quat = torch.tensor([qw, qx, qy, qz], device=device)
        return torch.nn.functional.normalize(quat, p=2, dim=0)

    @staticmethod
    def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """
        Multiply two quaternions (Hamilton product).

        :param q1: First quaternion [4] or [N, 4] (w, x, y, z)
        :param q2: Second quaternion [4] or [N, 4] (w, x, y, z)
        :return: Product quaternion, same shape as input
        """
        if q1.ndim == 1:
            # Single quaternion
            w1, x1, y1, z1 = q1.unbind()
            w2, x2, y2, z2 = q2.unbind()

            return torch.stack(
                [
                    w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                    w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                    w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                    w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                ],
                dim=0,
            )
        else:
            # Batch of quaternions
            w1, x1, y1, z1 = q1.T
            w2, x2, y2, z2 = q2.T

            return torch.stack(
                [
                    w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                    w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                    w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                    w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                ],
                dim=-1,
            )

    @staticmethod
    def rotate_scene(
        gaussian_data: GSTensor,
        rotation_angles: tuple[float, float, float],
        device: str = "cpu",
    ) -> GSTensor:
        """
        Rotate the entire scene by applying rotation to means and quaternions.

        :param gaussian_data: Input Gaussian data
        :param rotation_angles: Rotation angles (rx, ry, rz) in degrees
        :param device: Torch device
        :return: Rotated Gaussian data
        """
        rx, ry, rz = rotation_angles

        # Early exit if no rotation
        if rx == 0 and ry == 0 and rz == 0:
            return gaussian_data

        try:
            # Get rotation matrix
            R = TransformService.get_rotation_matrix(rotation_angles, device)

            # Rotate the means (positions)
            rotated_means = torch.matmul(gaussian_data.means, R.T)

            # Convert rotation matrix to quaternion
            rot_quat = TransformService.rotation_matrix_to_quaternion(R)

            # Expand to batch for multiplication
            rot_quat_batch = rot_quat.unsqueeze(0).expand(gaussian_data.quats.shape[0], -1)

            # Compose rotations: new_quat = rot_quat * existing_quat
            rotated_quats = TransformService.quaternion_multiply(rot_quat_batch, gaussian_data.quats)

            # Normalize to ensure valid rotations
            rotated_quats = torch.nn.functional.normalize(rotated_quats, p=2, dim=1)

            # Modify in-place (efficient, zero allocation)
            gaussian_data.means = rotated_means
            gaussian_data.quats = rotated_quats
            return gaussian_data

        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Scene rotation failed: %s", exc, exc_info=True)
            return gaussian_data

    @staticmethod
    def rotate_point(
        point: np.ndarray | list[float],
        rotation_angles: tuple[float, float, float],
    ) -> np.ndarray:
        """
        Rotate a single point around the origin.

        :param point: Point to rotate [3]
        :param rotation_angles: Rotation angles (rx, ry, rz) in degrees
        :return: Rotated point [3]
        """
        rx, ry, rz = rotation_angles

        # Early exit if no rotation
        if rx == 0 and ry == 0 and rz == 0:
            return np.array(point) if isinstance(point, list) else point

        # Convert to radians
        rx_rad, ry_rad, rz_rad = np.radians(rx), np.radians(ry), np.radians(rz)

        # Build rotation matrices (same order as rotate_scene)
        Rx = np.array(
            [
                [1, 0, 0],
                [0, np.cos(rx_rad), -np.sin(rx_rad)],
                [0, np.sin(rx_rad), np.cos(rx_rad)],
            ]
        )

        Ry = np.array(
            [
                [np.cos(ry_rad), 0, np.sin(ry_rad)],
                [0, 1, 0],
                [-np.sin(ry_rad), 0, np.cos(ry_rad)],
            ]
        )

        Rz = np.array(
            [
                [np.cos(rz_rad), -np.sin(rz_rad), 0],
                [np.sin(rz_rad), np.cos(rz_rad), 0],
                [0, 0, 1],
            ]
        )

        # Combine in ZYX order
        R = Rz @ Ry @ Rx

        # Apply rotation
        point_array = np.array(point) if isinstance(point, list) else point
        rotated = R @ point_array

        return rotated

    @staticmethod
    def apply_scale_transform(
        gaussian_data: GSTensor, scale: float
    ) -> GSTensor:
        """
        Apply uniform scale to Gaussian positions and sizes.

        When scaling the scene, both the Gaussian positions (means) and the Gaussian ellipsoid
        sizes (scales) must be scaled proportionally to maintain correct visual appearance.

        :param gaussian_data: Input Gaussian data
        :param scale: Scale factor
        :return: Scaled Gaussian data
        """
        if scale == 1.0:
            return gaussian_data

        try:
            # Modify in-place (efficient, zero allocation)
            gaussian_data.means = gaussian_data.means * scale

            # Scale Gaussian ellipsoid sizes proportionally
            # This is CRITICAL: without this, Gaussians appear too small
            # (if scale > 1) or too large (if scale < 1) relative to spacing
            gaussian_data.scales = gaussian_data.scales * scale

            return gaussian_data

        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Scale transform failed: %s", exc, exc_info=True)
            return gaussian_data

    @staticmethod
    def apply_translation_transform(
        gaussian_data: GSTensor, translation: tuple[float, float, float]
    ) -> GSTensor:
        """
        Apply translation to Gaussian positions.

        :param gaussian_data: Input Gaussian data
        :param translation: Translation offsets (tx, ty, tz)
        :return: Translated Gaussian data
        """
        tx, ty, tz = translation

        # Early exit if no translation
        if tx == 0.0 and ty == 0.0 and tz == 0.0:
            return gaussian_data

        try:
            # Create translation vector
            translation_vec = torch.tensor(
                [tx, ty, tz],
                dtype=gaussian_data.means.dtype,
                device=gaussian_data.means.device
            )

            # Modify in-place (efficient, zero allocation)
            gaussian_data.means = gaussian_data.means + translation_vec
            return gaussian_data

        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Translation transform failed: %s", exc, exc_info=True)
            return gaussian_data

    @staticmethod
    def build_srt_matrix(
        scale: float,
        rotation_angles: tuple[float, float, float],
        translation: tuple[float, float, float],
        device: str = "cpu"
    ) -> torch.Tensor:
        """
        Build a single 4x4 transformation matrix for Scale-Rotate-Translate.

        This is more efficient than applying transformations sequentially.

        :param scale: Uniform scale factor
        :param rotation_angles: Rotation angles (rx, ry, rz) in degrees
        :param translation: Translation offsets (tx, ty, tz)
        :param device: Torch device
        :return: 4x4 transformation matrix [4, 4]
        """
        # Get 3x3 rotation matrix
        R = TransformService.get_rotation_matrix(rotation_angles, device)

        # Build 4x4 SRT matrix
        # [sR | t]
        # [0  | 1]
        M = torch.eye(4, dtype=torch.float32, device=device)
        M[:3, :3] = R * scale  # Scale and rotate
        M[:3, 3] = torch.tensor(translation, dtype=torch.float32, device=device)  # Translate

        return M

    @staticmethod
    def apply_full_transform(
        gaussian_data: GSTensor, transform: "TransformValues", device: str = "cpu"
    ) -> GSTensor:
        """
        Apply complete scene transform (translation + rotation + scale).

        Optimized: Direct 3x3 affine transformation without homogeneous coordinates.
        Computes rotation matrix once and reuses it. Uses GPU device from tensor.

        Transformation order: Scale -> Rotate -> Translate (SRT)
        This is the standard order in computer graphics.

        :param gaussian_data: Input Gaussian data
        :param transform: Transform parameters
        :param device: Torch device (deprecated, uses tensor's device)
        :return: Transformed Gaussian data (modified in-place)
        """
        # Early exit if identity transform
        if hasattr(transform, "is_neutral") and transform.is_neutral():
            return gaussian_data

        try:
            # Use device from tensor (more efficient, avoids device parameter)
            tensor_device = gaussian_data.means.device
            tensor_dtype = gaussian_data.means.dtype
            
            # Compute rotation matrix once (or identity if no rotation)
            translation_tuple = tuple(
                float(x)
                for x in getattr(
                    transform, "translate", getattr(transform, "translation", (0.0, 0.0, 0.0))
                )
            )
            scale = (
                float(transform.scale)
                if isinstance(transform.scale, (int, float))
                else float(transform.scale[0])
            )
            rotation_quat = getattr(transform, "rotate", getattr(transform, "rotation", (1.0, 0.0, 0.0, 0.0)))
            from gsmod.transform.api import quaternion_to_euler

            rotation_rad = quaternion_to_euler(np.asarray(rotation_quat, dtype=np.float32))
            rotation_tuple = tuple(np.degrees(rotation_rad))
            has_rotation = not (rotation_tuple[0] == 0 and rotation_tuple[1] == 0 and rotation_tuple[2] == 0)

            if has_rotation:
                R = TransformService.get_rotation_matrix(rotation_tuple, tensor_device.type)
            else:
                R = torch.eye(3, dtype=tensor_dtype, device=tensor_device)

            # Build scaled rotation matrix: sR = scale * R
            sR = R * scale

            # Apply affine transformation: means' = means @ (sR).T + t
            # This is more efficient than homogeneous coordinates
            # Use in-place matmul if possible (torch >= 1.9)
            transformed_means = torch.matmul(gaussian_data.means, sR.T)

            # Add translation if needed (vectorized)
            if not (translation_tuple[0] == 0 and translation_tuple[1] == 0 and translation_tuple[2] == 0):
                t = torch.tensor(
                    translation_tuple,
                    dtype=tensor_dtype,
                    device=tensor_device
                )
                transformed_means = transformed_means + t

            # Handle quaternion rotation (reuse R matrix)
            transformed_quats = gaussian_data.quats
            if has_rotation:
                # Reuse R instead of recomputing
                rot_quat = TransformService.rotation_matrix_to_quaternion(R)
                rot_quat_batch = rot_quat.unsqueeze(0).expand(gaussian_data.quats.shape[0], -1)

                # Compose rotations: new_quat = rot_quat * existing_quat
                transformed_quats = TransformService.quaternion_multiply(
                    rot_quat_batch,
                    gaussian_data.quats
                )
                transformed_quats = torch.nn.functional.normalize(transformed_quats, p=2, dim=1)

            # Modify in-place (efficient, zero allocation)
            gaussian_data.means = transformed_means
            gaussian_data.quats = transformed_quats

            # Scale scales in-place if needed
            if scale != 1.0:
                gaussian_data.scales = gaussian_data.scales * scale

            return gaussian_data

        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Full transform failed: %s", exc, exc_info=True)
            return gaussian_data

    @staticmethod
    def calculate_scene_bounds(
        points: np.ndarray,
        percentile_min: float = 5.0,
        percentile_max: float = 95.0,
        padding: float = 0.1,
    ) -> SceneBounds:
        """
        Calculate scene bounding box from point cloud.

        Uses percentiles to ignore outliers and focus on main scene content.

        :param points: Point cloud [N, 3]
        :param percentile_min: Lower percentile for bounds (default 5.0)
        :param percentile_max: Upper percentile for bounds (default 95.0)
        :param padding: Padding factor to add to bounds (default 0.1 = 10%)
        :return: Calculated scene bounds
        """
        from src.domain.entities import SceneBounds

        try:
            # Use percentile-based bounds to avoid outliers
            min_coords = np.percentile(points, percentile_min, axis=0)
            max_coords = np.percentile(points, percentile_max, axis=0)

            # Calculate center and sizes
            center = (min_coords + max_coords) / 2
            sizes = max_coords - min_coords

            # Add padding
            min_coords = min_coords - sizes * padding
            max_coords = max_coords + sizes * padding
            sizes = max_coords - min_coords
            center = (min_coords + max_coords) / 2

            # Log bounds info
            logger.debug(
                "Scene bounds calculated: X [%s, %s], Y [%s, %s], Z [%s, %s]",
                f"{min_coords[0]:.3f}",
                f"{max_coords[0]:.3f}",
                f"{min_coords[1]:.3f}",
                f"{max_coords[1]:.3f}",
                f"{min_coords[2]:.3f}",
                f"{max_coords[2]:.3f}",
            )
            logger.debug("Center: %s, Size: %s", center, sizes)

            return SceneBounds(
                min_coords=tuple(min_coords),
                max_coords=tuple(max_coords),
                center=tuple(center),
                size=tuple(sizes),
            )

        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Scene bounds calculation failed: %s", exc, exc_info=True)
            # Return default bounds
            return SceneBounds()

    @staticmethod
    def calculate_bounding_sphere(
        points: np.ndarray, center: np.ndarray | None = None, percentile: float = 95.0
    ) -> tuple[np.ndarray, float]:
        """
        Calculate bounding sphere from point cloud.

        :param points: Point cloud [N, 3]
        :param center: Sphere center (default: centroid of points)
        :param percentile: Percentile of distances to use as radius (default 95.0)
        :return: Sphere center [3] and radius
        """
        if center is None:
            center = np.mean(points, axis=0)

        # Calculate distances from center
        distances = np.linalg.norm(points - center, axis=1)

        # Use percentile to ignore outliers
        radius = np.percentile(distances, percentile) * 1.1  # Add 10% padding

        logger.debug(
            "Bounding sphere: center=%s, radius=%.3f (%sth percentile)",
            center,
            radius,
            percentile,
        )

        return center, radius

