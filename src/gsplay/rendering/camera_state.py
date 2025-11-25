"""
Camera state dataclass for the universal viewer.

This module contains the CameraState dataclass which represents the authoritative
camera state, decoupling logical camera position from viser's internal representation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CameraState:
    """
    Explicit camera state - single source of truth.

    This dataclass represents the authoritative camera state, decoupling
    our logical camera position from viser's internal camera representation.
    All camera operations should update this state first, then apply to viser.
    """

    azimuth: float  # 0-360 degrees (rotation around Y axis)
    elevation: float  # -180 to 180 degrees (angle above/below horizon, unlimited)
    roll: float  # -180 to 180 degrees (camera tilt around view axis)
    distance: float  # Distance from camera to look_at point
    look_at: np.ndarray  # (3,) array - point camera is looking at

    @property
    def is_flipped(self) -> bool:
        """
        Check if camera is in flipped orientation (past the poles).

        Returns
        -------
        bool
            True if elevation is beyond +/-90 deg, indicating camera has passed
            through a pole and is viewing from the opposite hemisphere.
        """
        return abs(self.elevation) > 90
