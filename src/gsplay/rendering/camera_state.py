"""
Camera state dataclass for the universal viewer.

This module contains the CameraState dataclass which represents the authoritative
camera state using quaternion orientation, decoupling logical camera position
from viser's internal representation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from .quaternion_utils import (
    quat_from_axis_angle,
    quat_from_euler_deg,
    quat_multiply,
    quat_normalize,
    quat_to_euler_deg,
)

if TYPE_CHECKING:
    pass


def _default_orientation() -> np.ndarray:
    """Default orientation (identity quaternion)."""
    return np.array([1.0, 0.0, 0.0, 0.0])


def _default_look_at() -> np.ndarray:
    """Default look-at point (origin)."""
    return np.zeros(3)


@dataclass
class CameraState:
    """
    Explicit camera state - single source of truth.

    This dataclass represents the authoritative camera state using quaternion
    orientation, decoupling our logical camera position from viser's internal
    camera representation. All camera operations should update this state first,
    then apply to viser.

    Attributes
    ----------
    orientation : np.ndarray
        Camera orientation as quaternion (wxyz format)
    distance : float
        Distance from camera to look_at point
    look_at : np.ndarray
        (3,) array - point camera is looking at
    """

    orientation: np.ndarray = field(default_factory=_default_orientation)
    distance: float = 5.0
    look_at: np.ndarray = field(default_factory=_default_look_at)

    def __post_init__(self):
        """Ensure arrays are numpy arrays and normalize quaternion."""
        self.orientation = np.asarray(self.orientation, dtype=np.float64)
        self.look_at = np.asarray(self.look_at, dtype=np.float64)

        # Normalize quaternion
        self.orientation = quat_normalize(self.orientation)

    @property
    def azimuth(self) -> float:
        """
        Get azimuth angle in degrees for UI display.

        Returns
        -------
        float
            Azimuth angle (0-360 degrees)
        """
        az, _el, _roll = quat_to_euler_deg(self.orientation)
        return az

    @property
    def elevation(self) -> float:
        """
        Get elevation angle in degrees for UI display.

        Returns
        -------
        float
            Elevation angle (-90 to 90 degrees)
        """
        _az, el, _roll = quat_to_euler_deg(self.orientation)
        return el

    @property
    def roll(self) -> float:
        """
        Get roll angle in degrees for UI display.

        Returns
        -------
        float
            Roll angle (-180 to 180 degrees)
        """
        _az, _el, roll = quat_to_euler_deg(self.orientation)
        return roll

    @property
    def is_flipped(self) -> bool:
        """
        Check if camera is in flipped orientation (past the poles).

        With quaternions, this concept is less meaningful, but we keep it
        for compatibility. Returns True if looking down (elevation < -45).

        Returns
        -------
        bool
            True if elevation is significantly below horizon
        """
        return self.elevation < -45

    def set_from_euler(
        self, azimuth: float, elevation: float, roll: float = 0.0
    ) -> None:
        """
        Set orientation from Euler angles.

        Parameters
        ----------
        azimuth : float
            Azimuth angle in degrees (0-360)
        elevation : float
            Elevation angle in degrees (-90 to 90)
        roll : float, optional
            Roll angle in degrees (-180 to 180), default 0
        """
        self.orientation = quat_from_euler_deg(azimuth, elevation, roll)

    def rotate(self, axis: np.ndarray, angle_deg: float) -> None:
        """
        Apply incremental rotation around an axis.

        Parameters
        ----------
        axis : np.ndarray
            Rotation axis (3,) - in world coordinates
        angle_deg : float
            Rotation angle in degrees
        """
        delta = quat_from_axis_angle(axis, np.radians(angle_deg))
        self.orientation = quat_normalize(quat_multiply(delta, self.orientation))

    def copy(self) -> "CameraState":
        """
        Create a deep copy of this state.

        Returns
        -------
        CameraState
            New CameraState with copied values
        """
        return CameraState(
            orientation=self.orientation.copy(),
            distance=self.distance,
            look_at=self.look_at.copy(),
        )

    @classmethod
    def from_euler(
        cls,
        azimuth: float,
        elevation: float,
        roll: float,
        distance: float,
        look_at: np.ndarray,
    ) -> "CameraState":
        """
        Create CameraState from Euler angles.

        For backwards compatibility with existing code.

        Parameters
        ----------
        azimuth : float
            Azimuth angle in degrees
        elevation : float
            Elevation angle in degrees
        roll : float
            Roll angle in degrees
        distance : float
            Distance from camera to look_at
        look_at : np.ndarray
            Point camera is looking at

        Returns
        -------
        CameraState
            New state with given parameters
        """
        state = cls(distance=distance, look_at=look_at)
        state.set_from_euler(azimuth, elevation, roll)
        return state
