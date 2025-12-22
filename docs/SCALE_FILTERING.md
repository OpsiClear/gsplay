# Scale Filtering

## Overview

Automatically removes Gaussians with extremely large scales (outliers) that cause rendering artifacts like "ink spills" or visual noise.

## How It Works

1. **Auto-calculation**: On first frame load, calculates 99.5th percentile of scale values
2. **UI initialization**: Sets "Max Scale" slider to calculated value
3. **User control**: Adjust slider to filter more/less aggressively
4. **Filtering**: Applied during rendering and export

## Usage

### Default Behavior

The "Max Scale" slider in the Filter panel is automatically set to the 99.5th percentile of scales from your first frame. This works well for most datasets.

```
Filter Panel:
├── Filter Type: [None/Sphere/Cuboid]
├── Min Opacity: [0.05]
└── Max Scale: [0.045]  <-- Auto-set to 99.5th percentile
```

### Adjusting the Filter

- **Left (lower value)**: More aggressive filtering, removes more Gaussians
- **Right (higher value)**: Less aggressive filtering, keeps more Gaussians
- **Maximum (10.0+)**: Disables filtering

### When to Adjust

- **Lower**: If you still see artifacts or "ink spills"
- **Higher**: If legitimate details are being removed
- **Disable**: For debugging or if already pre-filtered

## Technical Details

**Model** (`src/models/ply/optimized_model.py`):

- Calculates 99.5th percentile on first load
- Exposes via `get_recommended_max_scale()`

**Filtering** (`src/viewer/edit_manager.py`):

- Applies mask: `scales.max(dim=1).values <= config.volume_filter.max_scale`
- Single source of truth: `config.volume_filter.max_scale`

## Performance

- Percentile calculation: ~0.1ms (one-time, first load)
- Per-frame filtering: ~0.05ms
- Net effect: Positive (fewer Gaussians to render)

## Migration Note

The old `max_scale_percentile` parameter is deprecated (does nothing). The viewer now uses UI-driven filtering with auto-calculated initial value.
