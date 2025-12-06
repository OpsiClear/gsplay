"""Rotation conversion utilities for UI handles.

Provides conversions between quaternions, Euler angles, axis-angle,
and rotation matrices for UI display and FilterValues computation.
"""

from __future__ import annotations

import math

import numpy as np


def camera_to_frustum_quaternion(
    camera_rotation: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Convert camera quaternion to frustum rotation quaternion.

    Viser's camera.wxyz is camera-to-world rotation. Apply 180 deg rotation
    around X (quaternion 0,1,0,0) to flip viewing direction from +Z to -Z.
    """
    w, x, y, z = camera_rotation
    # q_camera * q_flip where q_flip = (0, 1, 0, 0) for 180 deg around X
    # Result: (-x, w, z, -y)
    return (-x, w, z, -y)


def camera_to_frustum_axis_angle(
    camera_rotation: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    """Convert camera quaternion to axis-angle for frustum rotation.

    Viser's camera.wxyz is camera-to-world rotation. The frustum is created
    looking along -Z, but viser's camera convention has +Z as the look direction
    in local space. Apply 180 deg rotation around X to flip the viewing direction.
    """
    R_c2w = quaternion_to_matrix(camera_rotation)
    # Flip Y and Z (180 deg around X) to correct viewing direction
    FLIP_X = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    R_frustum = R_c2w @ FLIP_X
    return matrix_to_axis_angle(R_frustum)


def camera_to_frustum_euler_deg(
    camera_rotation: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    """Convert camera quaternion to Euler XYZ in degrees for frustum rotation.

    Viser's camera.wxyz is camera-to-world rotation. The frustum is created
    looking along -Z, but viser's camera convention has +Z as the look direction
    in local space. Apply 180 deg rotation around X to flip the viewing direction.
    """
    R_c2w = quaternion_to_matrix(camera_rotation)
    # Flip Y and Z (180 deg around X) to correct viewing direction
    FLIP_X = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    R_frustum = R_c2w @ FLIP_X
    return matrix_to_euler_deg(R_frustum)


def euler_deg_to_axis_angle(
    rx_deg: float, ry_deg: float, rz_deg: float
) -> tuple[float, float, float] | None:
    """Convert Euler angles (degrees, XYZ order) to axis-angle representation."""
    rx = math.radians(rx_deg)
    ry = math.radians(ry_deg)
    rz = math.radians(rz_deg)

    # Skip if no rotation
    if abs(rx) < 1e-6 and abs(ry) < 1e-6 and abs(rz) < 1e-6:
        return None

    # Convert Euler XYZ to quaternion
    # Using extrinsic XYZ (equivalent to intrinsic ZYX)
    cx, sx = math.cos(rx / 2), math.sin(rx / 2)
    cy, sy = math.cos(ry / 2), math.sin(ry / 2)
    cz, sz = math.cos(rz / 2), math.sin(rz / 2)

    # Quaternion from Euler XYZ (extrinsic)
    qw = cx * cy * cz + sx * sy * sz
    qx = sx * cy * cz - cx * sy * sz
    qy = cx * sy * cz + sx * cy * sz
    qz = cx * cy * sz - sx * sy * cz

    # Convert quaternion to axis-angle
    # angle = 2 * arccos(w), axis = (x, y, z) / sin(angle/2)
    angle = 2.0 * math.acos(max(-1.0, min(1.0, qw)))
    if angle < 1e-6:
        return None

    sin_half = math.sqrt(max(0.0, 1.0 - qw * qw))
    if sin_half < 1e-6:
        return None

    # Return axis-angle as axis * angle
    return (
        (qx / sin_half) * angle,
        (qy / sin_half) * angle,
        (qz / sin_half) * angle,
    )


def axis_angle_to_euler_deg(
    axis_angle: tuple[float, float, float]
) -> tuple[float, float, float]:
    """Convert axis-angle (axis * angle) to Euler XYZ in degrees."""
    ax, ay, az = axis_angle
    angle = math.sqrt(ax * ax + ay * ay + az * az)
    if angle < 1e-8:
        return (0.0, 0.0, 0.0)

    ux, uy, uz = ax / angle, ay / angle, az / angle
    half = angle / 2.0
    s = math.sin(half)
    w = math.cos(half)
    x = ux * s
    y = uy * s
    z = uz * s

    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - w * z)
    r02 = 2 * (x * z + w * y)
    r10 = 2 * (x * y + w * z)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z - w * x)
    r20 = 2 * (x * z - w * y)
    r21 = 2 * (y * z + w * x)
    r22 = 1 - 2 * (x * x + y * y)

    sy = -r20
    sy = max(-1.0, min(1.0, sy))
    ry = math.asin(sy)

    if abs(sy) < 0.9999:
        rx = math.atan2(r21, r22)
        rz = math.atan2(r10, r00)
    else:
        rx = math.atan2(-r12, r11)
        rz = 0.0

    return (math.degrees(rx), math.degrees(ry), math.degrees(rz))


def matrix_to_axis_angle(R: np.ndarray) -> tuple[float, float, float]:
    """Convert rotation matrix to axis-angle (axis * angle)."""
    trace = float(R[0, 0] + R[1, 1] + R[2, 2])
    angle = math.acos(max(-1.0, min(1.0, (trace - 1.0) / 2.0)))
    if angle < 1e-6:
        return (0.0, 0.0, 0.0)
    denom = 2.0 * math.sin(angle)
    ax = (R[2, 1] - R[1, 2]) / denom
    ay = (R[0, 2] - R[2, 0]) / denom
    az = (R[1, 0] - R[0, 1]) / denom
    return (ax * angle, ay * angle, az * angle)


def matrix_to_euler_deg(R: np.ndarray) -> tuple[float, float, float]:
    """Convert rotation matrix to Euler XYZ in degrees."""
    sy = -R[2, 0]
    sy = max(-1.0, min(1.0, sy))
    ry = math.asin(sy)

    if abs(sy) < 0.9999:
        rx = math.atan2(R[2, 1], R[2, 2])
        rz = math.atan2(R[1, 0], R[0, 0])
    else:
        rx = math.atan2(-R[1, 2], R[1, 1])
        rz = 0.0

    return (math.degrees(rx), math.degrees(ry), math.degrees(rz))


def quaternion_to_matrix(q: tuple[float, float, float, float]) -> np.ndarray:
    """Convert quaternion (w, x, y, z) to rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=float)


def quaternion_to_matrix_xyzw(q: tuple[float, float, float, float]) -> np.ndarray:
    """Convert quaternion (x, y, z, w) to rotation matrix.

    gsmod uses (x, y, z, w) format for quaternions.
    """
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=float)


def matrix_to_quaternion(R: np.ndarray) -> tuple[float, float, float, float]:
    """Convert rotation matrix to quaternion (w, x, y, z)."""
    trace = float(R[0, 0] + R[1, 1] + R[2, 2])
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    norm = math.sqrt(w * w + x * x + y * y + z * z)
    return (w / norm, x / norm, y / norm, z / norm)


def euler_deg_to_quaternion_xyzw(
    rx_deg: float, ry_deg: float, rz_deg: float
) -> tuple[float, float, float, float]:
    """Convert Euler XYZ angles (degrees) to quaternion (x, y, z, w).

    This is the exact inverse of matrix_to_euler_deg to ensure round-trip consistency.
    Uses extrinsic XYZ convention (rotate around fixed axes).
    """
    rx = math.radians(rx_deg)
    ry = math.radians(ry_deg)
    rz = math.radians(rz_deg)

    # Half angles
    cx, sx = math.cos(rx / 2), math.sin(rx / 2)
    cy, sy = math.cos(ry / 2), math.sin(ry / 2)
    cz, sz = math.cos(rz / 2), math.sin(rz / 2)

    # Quaternion from extrinsic XYZ Euler angles
    # Order: first rotate around X, then Y, then Z (fixed frame)
    w = cx * cy * cz + sx * sy * sz
    x = sx * cy * cz - cx * sy * sz
    y = cx * sy * cz + sx * cy * sz
    z = cx * cy * sz - sx * sy * cz

    # Normalize to ensure w >= 0 for consistency
    if w < 0:
        w, x, y, z = -w, -x, -y, -z

    return (x, y, z, w)
