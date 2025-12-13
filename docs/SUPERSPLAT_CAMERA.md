# Camera System Documentation

This document describes the camera system in gsplay, including controls, architecture, and implementation details.

## Overview

The viewer uses a mode-based camera ownership system with spherical coordinates as the primary representation. The camera supports orbit controls, preset views, continuous rotation, and a "Bake View" feature for model alignment.

### Attribution

This project builds upon:
- **[nerfview](https://github.com/hangg7/nerfview)** - The original NeRF viewer that provided the foundation for our rendering architecture
- **[viser](https://github.com/nerfstudio-project/viser)** - The web-based 3D visualization framework powering the UI and camera controls
- **[PlayCanvas SuperSplat](https://github.com/playcanvas/supersplat)** - Inspiration for camera control patterns

## Features

### 1. **Grid Display**
- Ground plane grid for spatial reference
- Automatically sized based on scene bounds
- Toggle visibility with the "[G]rid Toggle" button

### 2. **Focus/Frame Scene**
- Automatically frame the camera to view the entire scene
- Click the "[F]rame Scene" button
- Camera positions itself optimally based on scene bounds

### 3. **Preset Camera Views**
- Quickly jump to standard viewpoints:
  - **Top**: View from directly above (elevation 89°)
  - **Bottom**: View from directly below (elevation -89°)
  - **Front**: View from front (azimuth 0°)
  - **Back**: View from back (azimuth 180°)
  - **Left**: View from left (azimuth 270°)
  - **Right**: View from right (azimuth 90°)
  - **Iso**: Isometric view (azimuth 45°, elevation 30°)

### 4. **Continuous Rotation**
- Clockwise/counter-clockwise rotation around Y axis
- Uses quaternion math to avoid gimbal lock
- Configurable speed (degrees per second)
- Works in headless mode (no browser connected)

### 5. **Bake View**
- Rotates/translates the model to preserve the current view
- When camera resets to default isometric position, the model appears unchanged
- Uses view preservation formula: `R_delta = R_default @ R_current.T`

### 6. **Built-in Orbit Controls**
- **Left Mouse Button**: Orbit around focal point
- **Right Mouse Button**: Pan camera
- **Mouse Wheel**: Zoom in/out
- These are provided by viser's built-in three.js controls

## Architecture

### Mode-Based Camera Ownership

The camera operates in two modes to eliminate race conditions:

```
1. USER_MODE: Viser owns the camera
   - User can orbit/pan/zoom with mouse
   - We sync FROM viser to update our spherical state
   - Slider changes are processed normally

2. APP_MODE: We own the camera
   - During rotation, presets, programmatic changes
   - We push TO viser, ignore callbacks FROM viser
   - Slider callbacks are blocked
```

Mode transitions:
- `start_auto_rotation()` → APP_MODE
- `stop_auto_rotation()` → USER_MODE (with brief cooldown)
- `set_preset_view()` → APP_MODE briefly, then USER_MODE

### Coordinate System

**Spherical Coordinates (Primary Representation):**
- **Azimuth**: Horizontal angle around Y axis (0-360°, 0° = looking from +Z)
- **Elevation**: Vertical angle (-89° to 89°, positive = camera above target)
- **Roll**: Camera tilt around view axis (-180° to 180°)
- **Distance**: Distance from look_at target

**Quaternions:**
- All quaternions use **wxyz format** (w, x, y, z) to match viser's convention
- w is the scalar component, (x, y, z) is the vector component

### Viser Look-At Convention (CRITICAL)

Viser uses a **different look-at convention** than standard OpenGL:

**Viser convention:**
```python
forward = normalize(look_at - position)  # toward target
right = normalize(cross(forward, up_hint))
up = cross(forward, right)  # NOTE: forward × right, NOT right × forward!
R = [right, up, forward]  # camera looks down +Z toward target
```

**OpenGL convention (DO NOT USE with viser):**
```python
up = cross(right, forward)
R = [right, up, -forward]  # camera looks down -Z
```

**Key Rule:** Never set `camera.wxyz` explicitly. Let viser compute it from `position`, `look_at`, and `up_direction`.

## Code Structure

```
src/gsplay/rendering/
├── camera.py           # CameraController class with mode-based ownership
├── camera_state.py     # CameraState dataclass (spherical coords primary)
├── camera_ui.py        # UI factory functions for camera controls
└── quaternion_utils.py # Quaternion math utilities (wxyz format)
```

### Key Classes

**`CameraController`** (`rendering/camera.py`):
- Manages camera state with thread-safe access
- Mode-based ownership (USER vs APP mode)
- Preset views, continuous rotation, pole crossing
- Grid and world axis visualization

**`CameraState`** (`rendering/camera_state.py`):
- Spherical coordinates as primary representation
- Lazy c2w matrix computation
- Methods: `set_from_orbit()`, `set_from_viser()`

**`quaternion_utils.py`**:
- `quat_multiply()` - Hamilton product
- `quat_normalize()` - Unit quaternion
- `quat_from_axis_angle()` - Axis-angle to quaternion
- `quat_to_rotation_matrix()` - Quaternion to 3x3 matrix
- `rotation_matrix_to_quat()` - 3x3 matrix to quaternion
- `quat_from_euler_deg()` - Euler angles to quaternion (OpenGL convention)
- `quat_to_euler_deg()` - Quaternion to Euler angles

## Bake View Implementation

The "Bake View" feature in `core/app.py` preserves the current view when camera resets:

```python
def viser_look_at_matrix(position, target):
    """Compute rotation matrix using VISER's look-at convention."""
    forward = normalize(target - position)
    right = normalize(cross(forward, up_hint))
    up = cross(forward, right)  # NOT cross(right, forward)!
    return column_stack([right, up, forward])  # NOT -forward!

# R_current from viser's actual wxyz (what rendering uses)
R_current = SO3(camera.wxyz).as_matrix()

# R_default using viser's convention
R_default = viser_look_at_matrix(pos_default, target)

# View preservation
R_delta = R_default @ R_current.T
R_new_model = R_delta @ R_old_model
```

## Usage Notes

### Starting Rotation
```python
camera.start_auto_rotation(axis="y", speed=20.0)  # 20°/sec clockwise
camera.start_auto_rotation(axis="y", speed=-20.0)  # counter-clockwise
```

### Setting Preset View
```python
camera.set_preset_view("iso")  # azimuth=45°, elevation=30°
camera.set_preset_view("top")  # elevation=89°
```

### Accessing State
```python
state = camera.get_state()  # Thread-safe copy
az, el, roll = state.azimuth, state.elevation, state.roll
```

## References

- [SuperSplat Camera Controls](https://developer.playcanvas.com/user-manual/gaussian-splatting/editing/supersplat/camera-controls/)
- [Viser Documentation](https://viser.studio/)
- [PlayCanvas SuperSplat](https://github.com/playcanvas/supersplat)
