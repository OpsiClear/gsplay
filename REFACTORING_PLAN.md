# GSPlay Code Improvement Plan

This document outlines verified code improvements that enhance elegance and robustness without breaking changes.

## Priority 1: Slider Constants Module (High Impact)

### Problem
155 numeric slider bound values scattered across 41 lines in 2 files:
- `src/gsplay/ui/layout.py` (35 instances)
- `src/gsplay/initialization/ui_setup.py` (6 instances)

### Solution
Create `src/gsplay/config/slider_constants.py`:

```python
"""Centralized slider bound constants for UI controls."""

class SliderBounds:
    """Named constants for UI slider min/max values."""

    # Position controls (translation, filter centers)
    TRANSLATION_MIN, TRANSLATION_MAX = -10.0, 10.0
    FILTER_CENTER_MIN, FILTER_CENTER_MAX = -20.0, 20.0
    FRUSTUM_POSITION_MIN, FRUSTUM_POSITION_MAX = -50.0, 50.0

    # Rotation controls (degrees)
    ROTATION_MIN, ROTATION_MAX = -180.0, 180.0

    # Scale controls
    MAIN_SCALE_MIN, MAIN_SCALE_MAX = 0.1, 5.0
    RELATIVE_SCALE_MIN, RELATIVE_SCALE_MAX = 0.5, 2.0

    # Filter size/radius
    SPHERE_RADIUS_MIN, SPHERE_RADIUS_MAX = 0.1, 50.0
    BOX_SIZE_MIN, BOX_SIZE_MAX = 0.1, 50.0
    ELLIPSOID_RADIUS_MIN, ELLIPSOID_RADIUS_MAX = 0.1, 50.0

    # Opacity
    OPACITY_MIN, OPACITY_MAX = 0.0, 1.0

    # Scale filtering
    SCALE_FILTER_MIN, SCALE_FILTER_MAX = 0.0, 100.0
```

### Files to Update
1. `src/gsplay/ui/layout.py` - Replace magic numbers with constants
2. `src/gsplay/initialization/ui_setup.py` - Replace magic numbers with constants

---

## Priority 2: UIHandles Value Helper (52 occurrences)

### Problem
Pattern `self.X.value if self.X else default` repeated 52 times in `ui_handles.py`.

### Solution
Add helper method to UIHandles class:

```python
def _get_value(self, control, default):
    """Safely get control value with fallback."""
    return control.value if control else default
```

### Before
```python
temp_ui = self.temperature_slider.value if self.temperature_slider else 0.5
tint_ui = self.tint_slider.value if self.tint_slider else 0.5
```

### After
```python
temp_ui = self._get_value(self.temperature_slider, 0.5)
tint_ui = self._get_value(self.tint_slider, 0.5)
```

### Files to Update
1. `src/gsplay/config/ui_handles.py` - Add helper, refactor get_color_values(), get_transform_values(), get_filter_values()

---

## Priority 3: Filter Callback Registration (55 lines -> ~15 lines)

### Problem
4 identical loops registering callbacks for filter controls in `ui_setup.py`.

### Solution
Add helper method:

```python
def _register_callbacks(self, control_names: list[str], callback) -> None:
    """Register callback on multiple UI controls by attribute name."""
    for name in control_names:
        control = getattr(self.ui, name, None)
        if control:
            control.on_update(callback)
```

### Before (55 lines)
```python
for control in [self.ui.sphere_center_x, self.ui.sphere_center_y, ...]:
    if control:
        control.on_update(on_filter_change)
# Repeated 4 times for sphere, box, ellipsoid, frustum
```

### After (~15 lines)
```python
sphere_controls = ['sphere_center_x', 'sphere_center_y', 'sphere_center_z', 'sphere_radius']
box_controls = ['box_center_x', 'box_center_y', 'box_center_z', 'box_size_x', ...]
ellipsoid_controls = ['ellipsoid_center_x', ...]
frustum_controls = ['frustum_fov', 'frustum_aspect', ...]

for controls in [sphere_controls, box_controls, ellipsoid_controls, frustum_controls]:
    self._register_callbacks(controls, on_filter_change)
```

### Files to Update
1. `src/gsplay/initialization/ui_setup.py`

---

## Priority 4: ProcessingMode Enum Enhancement

### Problem
`layout.py` uses hardcoded strings `["All GPU", "Color+Transform GPU", ...]` instead of the existing `ProcessingMode` enum.

### Solution
Add classmethod to ProcessingMode:

```python
@classmethod
def get_display_options(cls) -> list[str]:
    """Get list of display strings for UI dropdowns."""
    return [mode.to_display_string() for mode in cls]

@classmethod
def get_default_display(cls) -> str:
    """Get default mode display string."""
    return cls.ALL_GPU.to_display_string()
```

### Before
```python
initial_mode = "All GPU"
processing_mode = server.gui.add_dropdown(
    "Mode",
    ["All GPU", "Color+Transform GPU", "Transform GPU", "Color GPU", "All CPU"],
    ...
)
```

### After
```python
from src.infrastructure.processing_mode import ProcessingMode

initial_mode = ProcessingMode.get_default_display()
processing_mode = server.gui.add_dropdown(
    "Mode",
    ProcessingMode.get_display_options(),
    ...
)
```

### Files to Update
1. `src/infrastructure/processing_mode.py` - Add classmethods
2. `src/gsplay/ui/layout.py` - Use enum methods

---

## Priority 5: Silent Exception Handling (13 cases)

### Problem
~13 cases of `except Exception: pass` without logging or recovery.

### Problematic Locations
1. `src/infrastructure/registry/sources.py:97-98` - Silent skip on metadata error
2. `src/infrastructure/registry/sinks.py:100-101` - Silent skip on metadata error
3. `src/gsplay/ui/layout.py:422-423` - Silent UI cleanup failure
4. `src/gsplay/ui/layout.py:535-536` - Silent config path update failure

### Solution
Add logging to silent exceptions:

```python
# Before
except Exception:
    pass

# After
except Exception as e:
    logger.debug(f"Non-critical error during cleanup: {e}")
```

For registry metadata collection, add warning:
```python
except Exception as e:
    logger.warning(f"Failed to get metadata for source '{name}': {e}")
    # Continue to next source
```

---

## Implementation Order

1. **Slider Constants** - Highest impact, affects most files
2. **UIHandles Helper** - High repetition reduction
3. **Filter Callbacks** - Clear duplication elimination
4. **ProcessingMode** - Small but improves consistency
5. **Exception Handling** - Important for debugging

## Non-Breaking Guarantee

All changes:
- Add new helper methods/constants (additive)
- Refactor internal implementation (no API change)
- Preserve exact same behavior
- No changes to public interfaces or return values
