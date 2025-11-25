# Auto-Rotation Feature

## Overview

The viewer includes camera auto-rotation controls (CW/CCW/Stop) in the View panel for continuous 360-degree camera orbits around the scene. The camera rotates around the Y-axis (azimuth), providing automated product-showcase style views.

## UI Controls

Located in the **View** panel (camera controls):

- **Speed** - Slider to control rotation speed (5-180 deg/sec, default 30)
- **Rotate CW** - Button to start clockwise orbit
- **Rotate CCW** - Button to start counter-clockwise orbit
- **Stop** - Button to stop rotation

## Architecture

### Components

1. **SuperSplatCamera** (`src/viewer/supersplat_camera.py`)
   - Built-in camera rotation animation loop
   - Runs at 30 FPS in background thread
   - Updates camera azimuth continuously
   - Thread-safe start/stop operations

2. **Camera UI** (in View panel)
   - CW/CCW/Stop buttons directly wired to camera
   - Speed slider controls rotation rate
   - No intermediate managers needed

3. **ViewerAPI** (`src/viewer/api.py`)
   - `rotate_cw(speed_dps)` - Start CW orbit
   - `rotate_ccw(speed_dps)` - Start CCW orbit
   - `stop_rotation()` - Stop rotation
   - `is_rotating()` - Check if active
   - `get_rotation_speed()` - Get current speed

## Usage

### Manual Control (Browser UI)

1. Open viewer in browser
2. Navigate to **View** panel
3. Adjust **Speed** slider (5-180 deg/sec)
4. Click **Rotate CW** or **Rotate CCW**
5. Click **Stop** to halt rotation

### Programmatic Control (Python API)

```python
# Start rotation
viewer.api.rotate_cw(45.0)          # CW at 45 deg/sec
viewer.api.rotate_ccw(60.0)         # CCW at 60 deg/sec

# Stop rotation
viewer.api.stop_rotation()

# Query state
is_rotating = viewer.api.is_rotating()
speed = viewer.api.get_rotation_speed()
```

## Implementation Details

### Rotation Loop

- Runs in SuperSplatCamera's background thread
- Updates at 20 FPS for smooth animation
- CW: positive speed (azimuth increases)
- CCW: negative speed (azimuth decreases)
- Automatically triggers rerender via camera state updates

### Thread Safety

- Uses threading primitives for clean shutdown
- Properly stops rotation on viewer close
- No race conditions

### Performance

- Minimal CPU overhead (~0.1% at 30 FPS)
- No impact on rendering performance
- Speed dynamically adjustable during rotation

## Example: Automated Product Demo

```python
from src.viewer.app import UniversalViewer
from src.viewer.config import ViewerConfig
import time

# Create viewer
config = ViewerConfig(port=8080)
viewer = UniversalViewer(config)
viewer.load_model_from_config({...})
viewer.setup_viewer()

# Automated 360-degree view
viewer.api.rotate_cw(30.0)  # Slow rotation
time.sleep(12)              # One full rotation at 30 deg/sec

# Pause and zoom
viewer.api.stop_rotation()
viewer.api.set_scale(2.0)
time.sleep(3)

# Resume opposite direction
viewer.api.rotate_ccw(45.0)  # Faster reverse
time.sleep(8)

viewer.run()
```

## Testing

```bash
# Run viewer
uv run src/viewer/main.py --config ./export_with_edits
```

Browser shows rotation controls in View panel. Click buttons to test.
