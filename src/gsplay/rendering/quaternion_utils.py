"""
Quaternion utilities for camera orientation.

All quaternions use wxyz format (w, x, y, z) to match viser's convention.
w is the scalar component, (x, y, z) is the vector component.
"""

from __future__ import annotations

import numpy as np


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Multiply two quaternions (Hamilton product).

    Parameters
    ----------
    q1 : np.ndarray
        First quaternion (wxyz format)
    q2 : np.ndarray
        Second quaternion (wxyz format)

    Returns
    -------
    np.ndarray
        Product quaternion q1 * q2 (wxyz format)
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """
    Compute quaternion conjugate.

    Parameters
    ----------
    q : np.ndarray
        Quaternion (wxyz format)

    Returns
    -------
    np.ndarray
        Conjugate quaternion (wxyz format)
    """
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """
    Normalize quaternion to unit length.

    Parameters
    ----------
    q : np.ndarray
        Quaternion (wxyz format)

    Returns
    -------
    np.ndarray
        Normalized quaternion (wxyz format)
    """
    norm = np.linalg.norm(q)
    if norm < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / norm


def quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """
    Create quaternion from axis-angle representation.

    Parameters
    ----------
    axis : np.ndarray
        Rotation axis (3,) - will be normalized
    angle : float
        Rotation angle in radians

    Returns
    -------
    np.ndarray
        Rotation quaternion (wxyz format)
    """
    axis = np.asarray(axis, dtype=np.float64)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])

    axis = axis / axis_norm
    half_angle = angle / 2.0
    s = np.sin(half_angle)
    c = np.cos(half_angle)

    return np.array([c, axis[0] * s, axis[1] * s, axis[2] * s])


def quat_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to 3x3 rotation matrix.

    Parameters
    ----------
    q : np.ndarray
        Quaternion (wxyz format)

    Returns
    -------
    np.ndarray
        3x3 rotation matrix
    """
    q = quat_normalize(q)
    w, x, y, z = q

    # Precompute products
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    return np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ])


def rotation_matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """
    Convert 3x3 rotation matrix to quaternion.

    Uses Shepperd's method for numerical stability.

    Parameters
    ----------
    R : np.ndarray
        3x3 rotation matrix

    Returns
    -------
    np.ndarray
        Quaternion (wxyz format)
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = np.array([w, x, y, z])
    return quat_normalize(q)


def quat_from_euler_deg(azimuth: float, elevation: float, roll: float) -> np.ndarray:
    """
    Create quaternion from Euler angles (azimuth, elevation, roll).

    Convention: Y-up coordinate system
    - Azimuth: rotation around Y axis (0-360 deg, positive = CCW from above)
    - Elevation: rotation around local X axis (-90 to 90 deg, positive = up)
    - Roll: rotation around local Z axis (-180 to 180 deg)

    Rotation order: azimuth (Y) -> elevation (X) -> roll (Z)

    Parameters
    ----------
    azimuth : float
        Azimuth angle in degrees (rotation around Y)
    elevation : float
        Elevation angle in degrees (rotation around X)
    roll : float
        Roll angle in degrees (rotation around Z)

    Returns
    -------
    np.ndarray
        Quaternion (wxyz format)
    """
    # Convert to radians
    az = np.radians(azimuth)
    el = np.radians(elevation)
    ro = np.radians(roll)

    # Create individual rotation quaternions
    # For orbit camera looking at origin from position (r*sin(az)*cos(el), r*sin(el), r*cos(az)*cos(el))
    # The camera faces toward -forward direction (toward origin)
    #
    # Start with camera at (0, 0, r) looking at origin (forward = [0, 0, -1])
    # Azimuth rotates camera position around Y axis
    # Elevation tilts camera position up/down (positive = camera above target)
    #
    # Azimuth: rotate around Y axis (horizontal orbit)
    q_az = quat_from_axis_angle(np.array([0.0, 1.0, 0.0]), az)
    # Elevation: rotate around local X axis (vertical orbit)
    # Negative because positive elevation means camera ABOVE target (Y > 0),
    # which requires pitching the forward vector DOWN
    q_el = quat_from_axis_angle(np.array([1.0, 0.0, 0.0]), -el)
    # Roll: rotate around local Z axis (camera tilt)
    q_ro = quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), ro)

    # Combine: azimuth * elevation * roll
    q = quat_multiply(q_az, quat_multiply(q_el, q_ro))
    return quat_normalize(q)


def quat_to_euler_deg(q: np.ndarray) -> tuple[float, float, float]:
    """
    Convert quaternion to Euler angles (azimuth, elevation, roll).

    For orbit camera convention:
    - Azimuth: horizontal angle around Y axis (0 = looking from +Z toward origin)
    - Elevation: vertical angle (positive = camera above target)
    - Roll: camera tilt around view axis

    Parameters
    ----------
    q : np.ndarray
        Quaternion (wxyz format)

    Returns
    -------
    tuple[float, float, float]
        (azimuth, elevation, roll) in degrees
        - azimuth: 0-360
        - elevation: -90 to 90
        - roll: -180 to 180
    """
    q = quat_normalize(q)
    R = quat_to_rotation_matrix(q)

    # Forward direction (camera looks down -Z in local space)
    # forward = R @ [0, 0, -1]
    forward = -R[:, 2]

    # For orbit camera, the camera position relative to target is -forward * distance
    # So the "toward camera" direction is -forward
    toward_camera = -forward

    # Calculate elevation from toward_camera.y
    # Positive elevation = camera above target = positive Y component
    elevation = np.degrees(np.arcsin(np.clip(toward_camera[1], -1.0, 1.0)))

    # Calculate azimuth from toward_camera.x and toward_camera.z
    # atan2(x, z) gives angle from +Z axis toward +X axis
    azimuth = np.degrees(np.arctan2(toward_camera[0], toward_camera[2]))
    if azimuth < 0:
        azimuth += 360.0

    # Calculate roll from up vector
    camera_up = R[:, 1]

    # Expected up direction without roll (perpendicular to forward, in vertical plane)
    world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(world_up, forward)
    right_norm = np.linalg.norm(right)

    if right_norm < 1e-6:
        # Looking straight up or down - azimuth is undefined, roll captures all rotation
        # This is expected gimbal lock behavior
        roll = 0.0  # At poles, just report 0 roll for simplicity
    else:
        right = right / right_norm
        expected_up = np.cross(forward, right)

        # Roll is angle from expected_up to camera_up around forward axis
        # Positive roll = camera tilts clockwise (from camera's view)
        cos_roll = np.clip(np.dot(expected_up, camera_up), -1.0, 1.0)
        # Use camera_up cross expected_up (reversed order) for correct sign
        sin_roll = np.dot(np.cross(camera_up, expected_up), forward)
        roll = np.degrees(np.arctan2(sin_roll, cos_roll))

    return (azimuth, elevation, roll)


def quat_slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    """
    Spherical linear interpolation between two quaternions.

    Parameters
    ----------
    q1 : np.ndarray
        Start quaternion (wxyz format)
    q2 : np.ndarray
        End quaternion (wxyz format)
    t : float
        Interpolation parameter [0, 1]

    Returns
    -------
    np.ndarray
        Interpolated quaternion (wxyz format)
    """
    q1 = quat_normalize(q1)
    q2 = quat_normalize(q2)

    # Compute cosine of angle between quaternions
    dot = np.dot(q1, q2)

    # If dot < 0, negate one quaternion to take shorter path
    if dot < 0:
        q2 = -q2
        dot = -dot

    # If very close, use linear interpolation
    if dot > 0.9995:
        result = q1 + t * (q2 - q1)
        return quat_normalize(result)

    # Compute SLERP
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)

    s1 = np.sin((1.0 - t) * theta) / sin_theta
    s2 = np.sin(t * theta) / sin_theta

    return s1 * q1 + s2 * q2


def quat_from_look_at(
    position: np.ndarray,
    target: np.ndarray,
    up: np.ndarray = None,
) -> np.ndarray:
    """
    Create orientation quaternion from look-at parameters.

    Parameters
    ----------
    position : np.ndarray
        Camera position (3,)
    target : np.ndarray
        Point to look at (3,)
    up : np.ndarray, optional
        Up direction hint (3,), defaults to [0, 1, 0]

    Returns
    -------
    np.ndarray
        Quaternion (wxyz format)
    """
    if up is None:
        up = np.array([0.0, 1.0, 0.0])

    # Forward direction (from position to target) = camera's -Z axis direction
    forward = target - position
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])
    forward = forward / forward_norm

    # Right direction = forward x up (right-handed coordinate system)
    # This gives camera's +X axis direction
    right = np.cross(forward, up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-10:
        # forward is parallel to up - use fallback
        if abs(forward[1]) < 0.9:
            right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
        else:
            right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-10:
            return np.array([1.0, 0.0, 0.0, 0.0])
    right = right / right_norm

    # Recompute up to ensure orthogonality = camera's +Y axis direction
    up_ortho = np.cross(right, forward)

    # Build rotation matrix
    # Camera convention: local -Z is view direction, so column 2 is -forward
    R = np.column_stack([right, up_ortho, -forward])

    return rotation_matrix_to_quat(R)
