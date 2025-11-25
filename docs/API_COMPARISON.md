# API Design Comparison: Direct vs Wrapper

## Current Architecture

You already have `UIHandles` (a dataclass container), but it doesn't provide a clean programmatic interface.

## Option 1: Direct Access (No Wrapper)

### Pros
- Simple, no extra code
- Direct access to viser features
- Less maintenance

### Cons
- Verbose and error-prone
- No validation
- Coupled to viser implementation
- Unclear API for external users

### Example Usage
```python
# Accessing the viewer
viewer.ui.time_slider.value = 42
viewer.ui.temperature_slider.value = 50
viewer.ui.auto_play.value = "On"

# State queries (verbose)
state = {
    "frame": viewer.ui.time_slider.value,
    "temperature": viewer.ui.temperature_slider.value,
    "is_playing": viewer.ui.auto_play.value == "On"
}

# Triggering export (needs to call internal method)
viewer._handle_export_ply()  # Accessing private method!
```

## Option 2: Wrapper API (Recommended)

### Pros
- Clean, discoverable API
- Type-safe with validation
- Decoupled from viser (easier to change later)
- Self-documenting
- Testable

### Cons
- ~100 lines of wrapper code
- One more layer (minimal overhead)

### Example Usage
```python
# Clean API
viewer.api.seek_to_frame(42)
viewer.api.set_temperature(50)
viewer.api.play()
viewer.api.pause()

# State queries (clean)
state = viewer.api.get_state()
# Returns: ViewerState(frame=42, is_playing=True, temperature=50, ...)

# Triggering export
viewer.api.export_frames(format="compressed-ply")

# Batch operations
viewer.api.set_color_adjustments(
    temperature=50,
    brightness=10,
    contrast=5
)
```

## Recommendation

**Use a wrapper if you plan to:**
1. Script automation (batch processing, testing)
2. External integrations (other tools calling your viewer)
3. Build CLI commands
4. Provide programmatic control to users

**Skip the wrapper if:**
1. Only manual browser-based control
2. No external API consumers
3. Simple, one-off scripts

## Minimal Wrapper Implementation

A thin wrapper (~100 lines) provides huge ergonomic benefits:

```python
class ViewerAPI:
    """Clean programmatic interface to viewer controls."""

    def __init__(self, viewer: UniversalViewer):
        self._viewer = viewer

    # Frame control
    def seek_to_frame(self, frame: int) -> None:
        """Seek to specific frame."""
        if not 0 <= frame < self._viewer.model.get_total_frames():
            raise ValueError(f"Frame {frame} out of range")
        self._viewer.ui.time_slider.value = frame

    def play(self) -> None:
        """Start playback."""
        self._viewer.ui.auto_play.value = "On"

    def pause(self) -> None:
        """Pause playback."""
        self._viewer.ui.auto_play.value = "Off"

    # State queries
    def get_state(self) -> ViewerState:
        """Get current viewer state."""
        return ViewerState(
            frame=self._viewer.ui.time_slider.value,
            is_playing=self._viewer.ui.auto_play.value == "On",
            temperature=self._viewer.ui.temperature_slider.value,
            # ... more fields
        )

    # Color adjustments
    def set_temperature(self, value: float) -> None:
        """Set color temperature (-100 to 100)."""
        if not -100 <= value <= 100:
            raise ValueError("Temperature must be in range [-100, 100]")
        self._viewer.ui.temperature_slider.value = value

    # Export
    def export_frames(self, format: str = "compressed-ply") -> None:
        """Export frame sequence."""
        self._viewer._handle_export_ply()
```

Usage:
```python
viewer = UniversalViewer(config)
viewer.setup_viewer()

# Now you have both:
viewer.ui.time_slider.value = 42  # Direct access (if needed)
viewer.api.seek_to_frame(42)      # Clean API (recommended)
```

## My Recommendation

**Create a minimal wrapper** (~100 lines). It provides:
- Much better developer experience
- Type safety and validation
- Future flexibility
- Self-documenting API

The cost is minimal, the benefits are significant.
