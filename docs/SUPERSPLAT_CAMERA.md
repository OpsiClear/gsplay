# SuperSplat-Style Camera Controls

This document describes the SuperSplat-inspired camera controls added to the Universal 4D Viewer.

## Overview

The viewer now includes enhanced camera controls inspired by [PlayCanvas SuperSplat](https://github.com/playcanvas/supersplat), making navigation more intuitive for examining 3D Gaussian Splat scenes.

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
  - **Top**: View from directly above
  - **Front**: View from the front
  - **Right**: View from the right side
  - **Iso**: Isometric view (45-degree angle)
- Use the button group in the "Camera (SuperSplat Style)" panel

### 4. **FOV Adjustment**
- Adjust field of view (10° - 120°)
- Slider in the camera controls panel
- Real-time update for all connected clients

### 5. **Built-in Orbit Controls**
- **Left Mouse Button**: Orbit around focal point
- **Right Mouse Button**: Pan camera
- **Mouse Wheel**: Zoom in/out
- These are provided by viser's built-in three.js controls

## Usage

The SuperSplat camera controls are automatically initialized when you start the viewer:

```bash
uv run src/viewer/main.py --config ./export_with_edits
```

The controls appear in a new "Camera (SuperSplat Style)" panel in the UI.

## Implementation Details

### What's Implemented

The implementation uses viser's existing API (`client.camera` properties):
- `camera.position` - Camera world position
- `camera.look_at` - Target point
- `camera.up_direction` - Up vector
- `camera.fov` - Field of view in radians

### Current Limitations (viser v1.0.15)

**Not Available:**
- ❌ Keyboard controls (arrow keys for fly mode)
- ❌ Double-click to set focal point
- ❌ Real-time keyboard event handling

These features require client-side (JavaScript) event handling that viser v1.0.15 doesn't support. Future viser versions may add:
- `ScenePointerEventType = Literal["click", "rect-select", "double-click"]`
- Keyboard event messages

### Workarounds

Instead of keyboard shortcuts, we provide:
- ✅ **GUI Buttons** - Click "[F]rame Scene" instead of pressing F
- ✅ **GUI Buttons** - Click "[G]rid Toggle" instead of pressing G
- ✅ **Button Group** - Select preset views instead of hotkeys

## Code Structure

```
src/viewer/
├── supersplat_camera.py    # Camera controller implementation
└── app.py                  # Integration into UniversalViewer
```

### Key Classes

**`SuperSplatCamera`**
- Manages grid visibility
- Provides preset view positioning
- Handles scene bounds updates
- Focus/frame functionality

**`create_supersplat_camera_controls()`**
- Factory function that creates camera controller
- Sets up UI controls
- Returns initialized controller instance

## Extending the Camera Controls

To add new features:

1. **Add new preset view:**
```python
views = {
    "top": (np.array([0, distance, 0]), np.array([0, 0, -1])),
    "your_view": (offset_vector, up_vector),  # Add here
}
```

2. **Add new camera action:**
```python
def your_action(self) -> None:
    for client in self.server.get_clients().values():
        # Modify camera
        client.camera.position = ...
```

3. **Add UI button:**
```python
your_button = server.gui.add_button("Your Action")

@your_button.on_click
def _(_) -> None:
    camera.your_action()
```

## Future Enhancements

When viser adds keyboard/double-click support, we can add:

1. **Keyboard Fly Mode**
   - Arrow keys for WASD-style movement
   - Shift/Ctrl for speed modifiers

2. **Double-Click Focal Point**
   - Double-click anywhere in scene
   - Camera orbits around clicked point

3. **Hotkey Bindings**
   - F: Focus on scene
   - G: Toggle grid
   - 1-7: Preset views

## References

- [SuperSplat Camera Controls](https://developer.playcanvas.com/user-manual/gaussian-splatting/editing/supersplat/camera-controls/)
- [Viser Documentation](https://viser.studio/)
- [PlayCanvas SuperSplat](https://github.com/playcanvas/supersplat)
