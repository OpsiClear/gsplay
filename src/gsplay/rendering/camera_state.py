"""
Camera state using spherical coordinates as primary representation.

Industry standard for orbit cameras (three.js OrbitControls, camera-controls).
c2w matrix is computed lazily when needed for rendering.

Design principle: Store what's natural for orbit camera.
- Spherical coordinates (azimuth, elevation, roll, distance) are primary
- c2w matrix is derived on-demand
- Elevation clamped to ±89° to avoid pole singularity

RENDERING:
    During rotation, we build camera state locally (no viser round-trip):
    - c2w from _compute_c2w() using vt.SO3(quaternion).as_matrix()
    - fov/aspect cached from client updates

    apply_to_viser() sets wxyz directly (not position/look_at/up) to sync
    the browser view. This ensures viser uses OUR quaternion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

import viser.transforms as vt

from .quaternion_utils import (
    quat_from_euler_deg,
    quat_to_rotation_matrix,
)


def _default_look_at() -> np.ndarray:
    """Default look-at point (origin)."""
    return np.zeros(3, dtype=np.float32)


@dataclass
class CameraState:
    """
    Orbit camera state for UI display and app-controlled animations.

    This is NOT the source of truth for rendering. The render pipeline uses
    RenderCamera (from viewer.py) which is built directly from viser's camera.

    Data Flow:
    - USER MODE: Viser owns camera. We extract spherical coords for UI display.
    - APP MODE: We own this state (rotation, presets). Push to viser, render from viser.

    IMPORTANT: Never render directly from this class's c2w - it causes black renders
    due to numerical differences with viser's internal camera construction.

    Attributes
    ----------
    _azimuth : float
        Horizontal angle in degrees (0-360)
    _elevation : float
        Vertical angle in degrees (-89 to 89, clamped)
    _roll : float
        Camera tilt in degrees (-180 to 180)
    _distance : float
        Distance from look_at point
    look_at : np.ndarray
        (3,) point the camera orbits around
    fov : float
        Field of view in radians
    aspect : float
        Aspect ratio (width / height)
    """

    # PRIMARY: Spherical coordinates (natural for orbit camera)
    _azimuth: float = 45.0
    _elevation: float = 30.0
    _roll: float = 0.0
    _distance: float = 10.0

    # Previous elevation for pole-crossing detection
    _prev_elevation: float = 30.0

    # Orbit center
    look_at: np.ndarray = field(default_factory=_default_look_at)

    # Viewport parameters
    fov: float = 1.0  # radians (~57 degrees)
    aspect: float = 1.0

    # Lazy c2w cache (computed on demand)
    _c2w_cache: np.ndarray | None = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        """Validate and normalize inputs."""
        self.look_at = np.asarray(self.look_at, dtype=np.float32)
        if self.look_at.shape != (3,):
            raise ValueError(f"look_at must be (3,), got {self.look_at.shape}")
        # Clamp elevation to avoid poles
        self._elevation = float(np.clip(self._elevation, -89.0, 89.0))
        self._azimuth = float(self._azimuth) % 360.0

    # =========================================================================
    # Cache Management
    # =========================================================================

    def _invalidate_c2w(self) -> None:
        """Invalidate c2w cache. Call after any spherical parameter change."""
        self._c2w_cache = None

    def _compute_c2w(self) -> np.ndarray:
        """Compute c2w matrix from spherical coordinates.

        Uses viser's SO3 for rotation matrix to ensure exact compatibility
        with how viser constructs c2w from camera.wxyz.

        NOTE: This c2w is currently only used for:
        - position/up/forward property access
        - NOT for rendering (viser round-trip is used for rendering)
        """
        # Build rotation from Euler angles
        q = quat_from_euler_deg(self._azimuth, self._elevation, self._roll)

        # Use viser's SO3 to get rotation matrix
        R = vt.SO3(q).as_matrix()

        # Camera looks down -Z, position = look_at - forward * distance
        forward = -R[:, 2]
        position = self.look_at.astype(np.float64) - forward * self._distance

        # Build 4x4 c2w matrix (use float64 to match viser's convention)
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = R
        c2w[:3, 3] = position
        return c2w

    # =========================================================================
    # Properties (Direct Reads for Spherical, Lazy for c2w-derived)
    # =========================================================================

    @property
    def c2w(self) -> np.ndarray:
        """4x4 camera-to-world matrix (lazy computed)."""
        if self._c2w_cache is None:
            self._c2w_cache = self._compute_c2w()
        return self._c2w_cache

    @property
    def azimuth(self) -> float:
        """Azimuth angle in degrees (0-360)."""
        return self._azimuth

    @property
    def elevation(self) -> float:
        """Elevation angle in degrees (-89 to 89)."""
        return self._elevation

    @property
    def roll(self) -> float:
        """Roll angle in degrees (-180 to 180)."""
        return self._roll

    @property
    def distance(self) -> float:
        """Distance from look_at point."""
        return self._distance

    @property
    def position(self) -> np.ndarray:
        """Camera position in world coordinates."""
        return self.c2w[:3, 3].copy()

    @property
    def rotation_matrix(self) -> np.ndarray:
        """3x3 rotation matrix from c2w."""
        return self.c2w[:3, :3].copy()

    @property
    def forward(self) -> np.ndarray:
        """Camera forward direction (looks down -Z in local space)."""
        return -self.c2w[:3, 2]

    @property
    def up(self) -> np.ndarray:
        """Camera up direction (+Y in local space)."""
        return self.c2w[:3, 1].copy()

    @property
    def right(self) -> np.ndarray:
        """Camera right direction (+X in local space)."""
        return self.c2w[:3, 0].copy()

    # =========================================================================
    # Mutation Methods
    # =========================================================================

    def set_from_orbit(
        self,
        azimuth: float,
        elevation: float,
        roll: float = 0.0,
        distance: float | None = None,
        look_at: np.ndarray | None = None,
    ) -> None:
        """
        Set camera from orbit parameters.

        Parameters
        ----------
        azimuth : float
            Horizontal angle in degrees (0-360)
        elevation : float
            Vertical angle in degrees (-89 to 89)
        roll : float
            Camera tilt in degrees (-180 to 180)
        distance : float, optional
            Distance from look_at. If None, keeps current distance.
        look_at : np.ndarray, optional
            New look_at point. If None, keeps current.
        """
        if look_at is not None:
            self.look_at = np.asarray(look_at, dtype=np.float32)
        if distance is not None:
            self._distance = float(distance)

        self._azimuth = float(azimuth) % 360.0
        self._elevation = float(np.clip(elevation, -89.0, 89.0))
        self._roll = float(roll)
        self._invalidate_c2w()

    def set_from_viser(
        self,
        position: tuple | np.ndarray,
        look_at: tuple | np.ndarray,
        up_direction: tuple | np.ndarray,
    ) -> None:
        """
        Set camera from viser camera parameters.

        Extracts azimuth, elevation, and distance from position/look_at.
        Roll is NOT extracted - it's kept unchanged because:
        1. Roll extraction from up_direction is ambiguous (0° vs 180°)
        2. Viser's orbit controls don't change roll anyway
        3. Roll changes should come from UI sliders or preset views

        Parameters
        ----------
        position : array-like
            Camera position (3,)
        look_at : array-like
            Point camera is looking at (3,)
        up_direction : array-like
            Camera up direction hint (3,) - currently unused
        """
        # Store previous elevation for pole-crossing detection
        self._prev_elevation = self._elevation

        pos = np.asarray(position, dtype=np.float64)
        target = np.asarray(look_at, dtype=np.float64)

        # Update look_at
        self.look_at = target.astype(np.float32)

        # Extract distance
        offset = pos - target
        self._distance = float(np.linalg.norm(offset))
        if self._distance < 1e-6:
            return  # Invalid, keep current spherical state

        # Normalize offset for angle extraction
        offset_norm = offset / self._distance

        # Extract elevation from Y component (unambiguous)
        self._elevation = float(np.degrees(np.arcsin(np.clip(offset_norm[1], -1.0, 1.0))))
        self._elevation = float(np.clip(self._elevation, -89.0, 89.0))

        # Extract azimuth from XZ plane (unambiguous)
        horiz_dist = np.sqrt(offset[0] ** 2 + offset[2] ** 2)
        if horiz_dist > 1e-6:
            self._azimuth = float(np.degrees(np.arctan2(offset[0], offset[2]))) % 360.0
        # else: at pole, keep current azimuth (undefined)

        # Roll is intentionally NOT extracted from up_direction.
        # Keep existing roll value unchanged.

        self._invalidate_c2w()

    def set_from_viser_full(
        self,
        position: tuple | np.ndarray,
        look_at: tuple | np.ndarray,
        up_direction: tuple | np.ndarray,
    ) -> None:
        """
        Set camera from viser parameters INCLUDING roll handling.

        This method syncs all parameters. It extracts azimuth/elevation from
        position/look_at geometry (reliable), and handles roll by checking if
        viser's up_direction indicates Y-up (roll=0).

        Parameters
        ----------
        position : tuple | np.ndarray
            Camera position from viser
        look_at : tuple | np.ndarray
            Look-at target from viser
        up_direction : tuple | np.ndarray
            Up direction from viser - used to detect if roll should be reset
        """
        pos = np.asarray(position, dtype=np.float64)
        target = np.asarray(look_at, dtype=np.float64)
        up = np.asarray(up_direction, dtype=np.float64)

        self.look_at = target.astype(np.float32)

        # Extract distance from geometry
        offset = pos - target
        self._distance = float(np.linalg.norm(offset))
        if self._distance < 1e-6:
            return

        offset_norm = offset / self._distance

        # Extract azimuth from XZ plane
        horiz_dist = np.sqrt(offset[0] ** 2 + offset[2] ** 2)
        if horiz_dist > 1e-6:
            self._azimuth = float(np.degrees(np.arctan2(offset[0], offset[2]))) % 360.0

        # Extract elevation from Y component
        self._elevation = float(np.degrees(np.arcsin(np.clip(offset_norm[1], -1.0, 1.0))))
        self._elevation = float(np.clip(self._elevation, -89.0, 89.0))

        # Handle roll: viser's orbit control enforces Y-up (roll=0)
        # Check if up_direction is close to world Y-up
        up_norm = np.linalg.norm(up)
        if up_norm > 1e-6:
            up = up / up_norm
            # If up is mostly vertical (within ~10 degrees of Y axis), assume roll=0
            if abs(up[1]) > 0.98:  # cos(10°) ≈ 0.98
                self._roll = 0.0
            # Otherwise keep existing roll (user set it via slider)

        self._invalidate_c2w()

    def set_from_euler(
        self, azimuth: float, elevation: float, roll: float = 0.0
    ) -> None:
        """Set orientation from Euler angles (keeps distance/look_at)."""
        self.set_from_orbit(azimuth, elevation, roll, self._distance, self.look_at)

    # =========================================================================
    # Conversion Methods
    # =========================================================================

    def to_viser_params(self) -> dict:
        """
        Convert to viser camera parameters.

        Returns
        -------
        dict
            {"position": tuple, "look_at": tuple, "up_direction": tuple}
        """
        return {
            "position": tuple(float(x) for x in self.position),
            "look_at": tuple(float(x) for x in self.look_at),
            "up_direction": tuple(float(x) for x in self.up),
        }

    def get_K(self, img_wh: Tuple[int, int]) -> NDArray[np.float32]:
        """
        Get camera intrinsic matrix for given image size.

        Parameters
        ----------
        img_wh : tuple[int, int]
            Image width and height

        Returns
        -------
        np.ndarray
            3x3 intrinsic matrix
        """
        W, H = img_wh
        focal_length = H / 2.0 / np.tan(self.fov / 2.0)
        K = np.array(
            [
                [focal_length, 0.0, W / 2.0],
                [0.0, focal_length, H / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        return K

    def copy(self) -> "CameraState":
        """Create a deep copy of this state."""
        return CameraState(
            _azimuth=self._azimuth,
            _elevation=self._elevation,
            _roll=self._roll,
            _distance=self._distance,
            look_at=self.look_at.copy(),
            fov=self.fov,
            aspect=self.aspect,
        )

    # =========================================================================
    # Legacy Compatibility
    # =========================================================================

    @property
    def orientation_quat(self) -> np.ndarray:
        """Orientation as quaternion (wxyz format)."""
        return quat_from_euler_deg(self._azimuth, self._elevation, self._roll)

    @property
    def euler_angles(self) -> tuple[float, float, float]:
        """(azimuth, elevation, roll) in degrees."""
        return (self._azimuth, self._elevation, self._roll)

    @property
    def orientation(self) -> np.ndarray:
        """Legacy: get orientation as quaternion (wxyz)."""
        return self.orientation_quat

    @orientation.setter
    def orientation(self, q: np.ndarray) -> None:
        """Legacy: set orientation from quaternion - not fully supported."""
        # This would require quaternion to euler conversion
        # For now, just invalidate cache
        self._invalidate_c2w()

    @classmethod
    def from_euler(
        cls,
        azimuth: float,
        elevation: float,
        roll: float,
        distance: float,
        look_at: np.ndarray,
    ) -> "CameraState":
        """Create CameraState from Euler angles."""
        return cls(
            _azimuth=azimuth,
            _elevation=elevation,
            _roll=roll,
            _distance=distance,
            look_at=look_at,
        )
