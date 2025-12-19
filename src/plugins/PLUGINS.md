# Plugin Development Guide

This guide explains how to create and install plugins for GSPlay.

## Table of Contents

1. [Data Type Contract](#data-type-contract)
2. [Plugin Types Overview](#plugin-types-overview)
3. [Skill 1: Discrete Frame Source](#skill-1-discrete-frame-source-basegaussiansource)
4. [Skill 2: Interpolatable Source](#skill-2-interpolatable-source-interpolatablesource)
5. [Skill 3: Continuous Time Source](#skill-3-continuous-time-source-continuoustimesource)
6. [Skill 4: Data Sink (Exporter)](#skill-4-data-sink-exporter-datasinkprotocol)
7. [Installing Plugins](#installing-plugins)
8. [Testing Plugins](#testing-plugins)
9. [Best Practices](#best-practices)
10. [Reference Implementations](#reference-implementations)

---

## Data Type Contract

**CRITICAL: All plugins MUST comply with the GSPlay data type contract.**

### Supported Data Types

GSPlay ONLY accepts these standard gsply data types:

| Type | Module | Description | Usage |
|------|--------|-------------|-------|
| `GaussianData` | `src.domain.data` | Unified CPU/GPU wrapper | Plugin I/O (preferred) |
| `GSData` | `gsply` | CPU numpy container | Internal processing |
| `GSTensor` | `gsply.torch` | GPU PyTorch container | Rendering pipeline |
| `GSDataPro` | `gsmod` | Extended CPU container | Advanced features |
| `GSTensorPro` | `gsmod.torch` | Extended GPU container | Advanced features |

### Required Fields (ALL Types)

Every Gaussian data object MUST have these fields:

```python
# Field specifications
means: np.ndarray | torch.Tensor     # [N, 3] Gaussian centers
scales: np.ndarray | torch.Tensor    # [N, 3] Scale factors
quats: np.ndarray | torch.Tensor     # [N, 4] Quaternion rotations (wxyz or xyzw)
opacities: np.ndarray | torch.Tensor # [N] or [N, 1] Opacity values
sh0: np.ndarray | torch.Tensor       # [N, 3] DC color coefficients
```

Optional fields:
- `shN`: Higher-order SH coefficients `[N, K, 3]` where K depends on SH degree
- `format_info`: Format tracking (scales encoding, opacity encoding, etc.)

### Data Format Encoding

Use `FormatInfo` to track data encoding state:

```python
from src.domain.data import FormatInfo

FormatInfo(
    is_scales_ply=False,    # False = linear, True = log-space (raw PLY)
    is_opacities_ply=False, # False = linear [0,1], True = logit-space
    is_sh0_rgb=True,        # True = RGB [0,1], False = SH coefficients
    sh_degree=None,         # 0, 1, 2, 3, or None
)
```

### Enforcement

The export system validates all data before export:

```python
# Validation rejects non-standard types with clear error messages:
# TypeError: [export] Unsupported data type: MyCustomData.
#            Expected one of: GSData, GSTensor, GaussianData, ...
#            All input sources must decode to standard gsply types.
```

---

## Plugin Types Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PLUGIN TYPE MATRIX                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  INPUT PLUGINS (Data Sources)                                           │
│  =============================                                          │
│                                                                          │
│  ┌─────────────────────┐    ┌─────────────────────┐                     │
│  │ BaseGaussianSource  │    │ InterpolatableSource│                     │
│  ├─────────────────────┤    ├─────────────────────┤                     │
│  │ Discrete frames     │    │ Keyframe + blend    │                     │
│  │ PLY sequences       │    │ Smooth playback     │                     │
│  │ get_frame_at_time() │    │ get_keyframe()      │                     │
│  │ total_frames        │    │ keyframe_count      │                     │
│  └─────────────────────┘    └─────────────────────┘                     │
│                                                                          │
│  ┌─────────────────────┐                                                │
│  │ ContinuousTimeSource│                                                │
│  ├─────────────────────┤                                                │
│  │ Neural networks     │                                                │
│  │ Arbitrary time t    │                                                │
│  │ evaluate(t)         │                                                │
│  │ evaluate_batch()    │                                                │
│  └─────────────────────┘                                                │
│                                                                          │
│  OUTPUT PLUGINS (Data Sinks)                                            │
│  ============================                                           │
│                                                                          │
│  ┌─────────────────────┐                                                │
│  │ DataSinkProtocol    │                                                │
│  ├─────────────────────┤                                                │
│  │ Export single frame │                                                │
│  │ Export sequences    │                                                │
│  │ Progress callbacks  │                                                │
│  └─────────────────────┘                                                │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Input → Output Combinations

| Input Type | Output Mode | Description |
|------------|-------------|-------------|
| Discrete → Discrete | Direct copy | Frame-by-frame export |
| Discrete → Continuous | Interpolate on export | Smooth export at any FPS |
| Continuous → Discrete | Sample at intervals | Bake to frame sequence |
| Continuous → Continuous | Continuous → Snap | Export nearest keyframe only |

---

## Time Configuration

**NEW in v2.0**: Sources can specify time-related parameters in their config.

### Time Config Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source_fps` | `float \| None` | `None` | Original capture FPS. Enables dual time display. |
| `frame_count` | `int \| None` | `None` | Override auto-detected frame count. |
| `frame_start` | `int` | `0` | Start frame index (0-based). |
| `frame_end` | `int \| None` | `None` | End frame index (exclusive). `None` = all frames. |
| `playback_fps` | `float` | `30.0` | Suggested playback speed in FPS. |
| `lock_playback_fps` | `bool` | `False` | If `True`, UI cannot change playback FPS. |
| `autoplay` | `bool` | `False` | If `True`, start playback automatically on load. |

### Config Examples

**Basic PLY (auto frame count, no timing):**
```yaml
module: load-ply
config:
  ply_folder: ./frames/
```

**Video-derived PLY with source FPS:**
```yaml
module: load-ply
config:
  ply_folder: ./frames/
  source_fps: 30.0           # Original 30fps video
  playback_fps: 30.0         # Play at 1x speed
  lock_playback_fps: true    # Don't allow UI to change
  autoplay: true
```

**Subset of frames:**
```yaml
module: load-ply
config:
  ply_folder: ./frames/
  frame_start: 10            # Start from frame 10
  frame_end: 50              # Use frames 10-49 only
  playback_fps: 15.0         # Slower playback
```

**Limited frame count:**
```yaml
module: load-ply
config:
  ply_folder: ./frames/
  frame_count: 100           # Max 100 frames
  source_fps: 24.0           # 24fps source
```

### Time Display Formats

When `source_fps` is set, the UI shows dual time format:

| `source_fps` | Display Format | Example |
|--------------|----------------|---------|
| `None` (default) | `"Frame {n}"` | "Frame 50" |
| `30.0` | `"{s:.2f}s (Frame {n})"` | "1.67s (Frame 50)" |

### Frame Range Validation

If `frame_count`, `frame_start`, or `frame_end` exceed available files:
1. A warning is logged
2. Values are clamped to valid range
3. Playback continues with available frames

Example warning:
```
WARNING: frame_end=200 exceeds available files (150), clamping to 150
```

### Accessing Time Config in Plugins

Plugins that implement time config should expose these properties:

```python
class MySource(BaseGaussianSource):
    @property
    def source_fps(self) -> float | None:
        """Original capture FPS, or None if not specified."""
        return self._source_fps

    @source_fps.setter
    def source_fps(self, value: float | None) -> None:
        """Set source FPS (for runtime configuration)."""
        self._source_fps = value

    @property
    def playback_fps(self) -> float:
        """Suggested playback FPS."""
        return self._playback_fps

    @property
    def lock_playback_fps(self) -> bool:
        """Whether UI should lock playback FPS."""
        return self._lock_playback_fps

    @property
    def autoplay(self) -> bool:
        """Whether to auto-start playback on load."""
        return self._autoplay
```

### Time Domain with source_fps

When building a `TimeDomain`, pass `source_fps` to enable dual display:

```python
from src.domain.time import TimeDomain

# Discrete frames with source FPS
time_domain = TimeDomain.discrete(
    total_frames=100,
    source_fps=30.0,  # Enables frame-to-seconds conversion
)

# Interpolated keyframes with source FPS
time_domain = TimeDomain.interpolated(
    keyframe_times=[float(i) for i in range(100)],
    source_fps=30.0,
)
```

---

## Skill 1: Discrete Frame Source (BaseGaussianSource)

**Use case**: Loading sequences of PLY files, pre-computed Gaussian data, or static scenes.

### Protocol Requirements

```python
from src.domain.interfaces import BaseGaussianSource, SourceMetadata
from src.domain.data import GaussianData

class MyDiscreteSource(BaseGaussianSource):
    """4 required methods only."""

    @classmethod
    def metadata(cls) -> SourceMetadata:
        """REQUIRED: Return source metadata."""
        ...

    @classmethod
    def can_load(cls, path: str) -> bool:
        """REQUIRED: Check if source can handle path."""
        ...

    @property
    def total_frames(self) -> int:
        """REQUIRED: Total number of discrete frames."""
        ...

    def get_frame_at_time(self, normalized_time: float) -> GaussianData:
        """REQUIRED: Return frame at normalized time [0, 1]."""
        ...
```

### Data I/O Specification

**Input**: Configuration dict with source-specific parameters
```python
config = {
    "path": "/data/ply_sequence/",
    "device": "cuda",
    # ... source-specific options
}
```

**Output**: `GaussianData` with ALL required fields
```python
def get_frame_at_time(self, normalized_time: float) -> GaussianData:
    # Convert normalized time [0, 1] to frame index
    index = int(round(normalized_time * (self.total_frames - 1)))

    # Load your data...

    return GaussianData(
        means=np.array(..., dtype=np.float32),      # [N, 3]
        scales=np.array(..., dtype=np.float32),     # [N, 3]
        quats=np.array(..., dtype=np.float32),      # [N, 4]
        opacities=np.array(..., dtype=np.float32),  # [N]
        sh0=np.array(..., dtype=np.float32),        # [N, 3]
        format_info=FormatInfo(
            is_scales_ply=False,  # Your data's encoding
            is_opacities_ply=False,
            is_sh0_rgb=True,
        ),
        source_path=str(file_path),
    )
```

### Reference Implementation

```python
"""Minimal Discrete Frame Source - Complete Example"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from src.domain.interfaces import BaseGaussianSource, SourceMetadata
from src.domain.data import GaussianData, FormatInfo
from src.domain.time import TimeDomain


@dataclass
class MyFormatConfig:
    """Configuration schema for validation."""
    path: str = "."
    device: str = "cuda"


class MyFormatSource(BaseGaussianSource):
    """Discrete frame source for .myformat files."""

    @classmethod
    def metadata(cls) -> SourceMetadata:
        return SourceMetadata(
            name="My Format",
            description="Load .myformat Gaussian sequences",
            file_extensions=[".myformat"],
            config_schema=MyFormatConfig,
            version="1.0.0",
        )

    @classmethod
    def can_load(cls, path: str) -> bool:
        """Auto-detection logic."""
        p = Path(path)
        if p.is_dir():
            return any(p.glob("*.myformat"))
        return p.suffix.lower() == ".myformat"

    def __init__(self, config: dict) -> None:
        self._path = Path(config.get("path", "."))
        self._device = config.get("device", "cuda")
        self._files = sorted(self._path.glob("*.myformat"))

        if not self._files:
            raise ValueError(f"No .myformat files in {self._path}")

    @property
    def total_frames(self) -> int:
        return len(self._files)

    @property
    def time_domain(self) -> TimeDomain:
        """Optional: Define time representation."""
        return TimeDomain.discrete(self.total_frames)

    def get_frame_at_time(self, normalized_time: float) -> GaussianData:
        # Convert to frame index
        if self.total_frames <= 1:
            index = 0
        else:
            index = int(round(normalized_time * (self.total_frames - 1)))
            index = max(0, min(index, self.total_frames - 1))

        return self._load_frame(self._files[index])

    def _load_frame(self, file_path: Path) -> GaussianData:
        # YOUR LOADING LOGIC HERE
        # Example: load data from file
        # data = np.load(file_path)

        # Placeholder - replace with actual loading
        n = 1000  # Number of Gaussians
        return GaussianData(
            means=np.random.randn(n, 3).astype(np.float32),
            scales=np.abs(np.random.randn(n, 3)).astype(np.float32) * 0.1,
            quats=self._random_quaternions(n),
            opacities=np.random.rand(n).astype(np.float32),
            sh0=np.random.rand(n, 3).astype(np.float32),
            format_info=FormatInfo(
                is_scales_ply=False,
                is_opacities_ply=False,
                is_sh0_rgb=True,
            ),
            source_path=str(file_path),
        )

    def _random_quaternions(self, n: int) -> np.ndarray:
        """Generate valid unit quaternions."""
        quats = np.random.randn(n, 4).astype(np.float32)
        return quats / np.linalg.norm(quats, axis=1, keepdims=True)
```

### Time Domain (Optional)

Override `time_domain` property to customize time representation:

```python
@property
def time_domain(self) -> TimeDomain:
    # Default: discrete frames
    return TimeDomain.discrete(self.total_frames)

    # With source FPS for frame-to-seconds conversion:
    # return TimeDomain.discrete(self.total_frames, source_fps=30.0)
```

---

## Skill 2: Interpolatable Source (InterpolatableSource)

**Use case**: Smooth playback between discrete keyframes (PLY sequences with interpolation).

### Protocol Requirements

```python
from src.domain.interfaces import InterpolatableSource
from src.domain.data import GaussianData

class MyInterpolatableSource(InterpolatableSource):
    """Keyframes + interpolation support."""

    @property
    def keyframe_count(self) -> int:
        """REQUIRED: Number of keyframes."""
        ...

    def get_keyframe(self, index: int) -> GaussianData:
        """REQUIRED: Get data at keyframe index (0 to keyframe_count-1)."""
        ...

    def get_keyframe_time(self, index: int) -> float:
        """REQUIRED: Get source time for keyframe."""
        ...

    @property
    def interpolation_method(self) -> str:
        """REQUIRED: 'nearest', 'linear', 'slerp', or 'cubic'."""
        ...
```

### Data I/O Specification

**Input**: Same as BaseGaussianSource config

**Output**: `GaussianData` for keyframes + interpolated frames

Key difference: Must support **both** discrete keyframe access AND continuous time queries.

```python
# Keyframe access (discrete, exact data)
def get_keyframe(self, index: int) -> GaussianData:
    """Direct keyframe access - NO interpolation."""
    return self._load_keyframe(index)

# Time-based access (may interpolate)
def get_frame_at_source_time(self, source_time: float) -> GaussianData:
    """Interpolate between surrounding keyframes."""
    ...
```

### Reference Implementation

```python
"""Interpolatable Source - Complete Example"""

from __future__ import annotations
from typing import Any
import numpy as np

from src.domain.interfaces import (
    BaseGaussianSource,
    InterpolatableSource,
    SourceMetadata,
)
from src.domain.data import GaussianData, FormatInfo
from src.domain.time import TimeDomain
from src.domain.interpolation import interpolate_gaussian_data, InterpolationMethod


class InterpolatedFormatSource(BaseGaussianSource, InterpolatableSource):
    """PLY-like source with smooth interpolation between keyframes."""

    @classmethod
    def metadata(cls) -> SourceMetadata:
        return SourceMetadata(
            name="Interpolated Format",
            description="Smooth interpolation between keyframes",
            file_extensions=[".myformat"],
            version="1.0.0",
        )

    @classmethod
    def can_load(cls, path: str) -> bool:
        return path.lower().endswith(".myformat")

    def __init__(self, config: dict[str, Any]) -> None:
        self._files = self._discover_files(config.get("path", "."))
        self._device = config.get("device", "cuda")
        self._interpolation = InterpolationMethod.LINEAR

        # LRU cache for keyframes
        self._cache: dict[int, GaussianData] = {}
        self._cache_order: list[int] = []
        self._cache_size = 4

        # Build time domain with keyframe times
        self._time_domain = TimeDomain.interpolated(
            keyframe_times=[float(i) for i in range(len(self._files))],
        )

    # --- BaseGaussianSource Protocol ---

    @property
    def total_frames(self) -> int:
        return len(self._files)

    @property
    def time_domain(self) -> TimeDomain:
        return self._time_domain

    def get_frame_at_time(self, normalized_time: float) -> GaussianData:
        """Get interpolated frame at normalized time [0, 1]."""
        source_time = self._time_domain.from_normalized(normalized_time)
        return self.get_frame_at_source_time(source_time)

    def get_frame_at_source_time(self, source_time: float) -> GaussianData:
        """Get interpolated frame at source time (frames)."""
        # Clamp to valid range
        source_time = max(0.0, min(source_time, self.keyframe_count - 1))

        # Find surrounding keyframes
        lower_idx = int(source_time)
        t = source_time - lower_idx

        # Exact keyframe hit
        if t < 1e-6:
            return self.get_keyframe(lower_idx)

        # Last frame
        if lower_idx >= self.keyframe_count - 1:
            return self.get_keyframe(self.keyframe_count - 1)

        # Interpolate
        data0 = self.get_keyframe(lower_idx)
        data1 = self.get_keyframe(lower_idx + 1)

        return interpolate_gaussian_data(
            data0, data1, t, method=self._interpolation
        )

    # --- InterpolatableSource Protocol ---

    @property
    def keyframe_count(self) -> int:
        return len(self._files)

    def get_keyframe(self, index: int) -> GaussianData:
        """Get keyframe data (with LRU caching)."""
        index = max(0, min(index, self.keyframe_count - 1))

        # Check cache
        if index in self._cache:
            # Update LRU order
            if index in self._cache_order:
                self._cache_order.remove(index)
            self._cache_order.append(index)
            return self._cache[index]

        # Load from disk
        data = self._load_keyframe(index)

        # Cache with LRU eviction
        if len(self._cache) >= self._cache_size:
            oldest = self._cache_order.pop(0)
            del self._cache[oldest]

        self._cache[index] = data
        self._cache_order.append(index)

        return data

    def get_keyframe_time(self, index: int) -> float:
        """Keyframes are at integer frame times."""
        return float(index)

    @property
    def interpolation_method(self) -> str:
        return self._interpolation.name.lower()

    # --- Internal Methods ---

    def _discover_files(self, path: str) -> list:
        from pathlib import Path
        return sorted(Path(path).glob("*.myformat"))

    def _load_keyframe(self, index: int) -> GaussianData:
        # YOUR LOADING LOGIC HERE
        n = 1000
        return GaussianData(
            means=np.random.randn(n, 3).astype(np.float32),
            scales=np.abs(np.random.randn(n, 3)).astype(np.float32) * 0.1,
            quats=self._random_quats(n),
            opacities=np.random.rand(n).astype(np.float32),
            sh0=np.random.rand(n, 3).astype(np.float32),
            format_info=FormatInfo(is_scales_ply=False, is_opacities_ply=False, is_sh0_rgb=True),
            source_path=str(self._files[index]),
        )

    def _random_quats(self, n: int) -> np.ndarray:
        q = np.random.randn(n, 4).astype(np.float32)
        return q / np.linalg.norm(q, axis=1, keepdims=True)
```

### Interpolation Methods

Available methods in `src.domain.interpolation`:

| Method | Description | Use Case |
|--------|-------------|----------|
| `NEAREST` | No blending, nearest keyframe | Fast preview |
| `LINEAR` | Linear blend all attributes | General use |
| `SLERP` | Spherical interpolation for quats | Smooth rotation |
| `CUBIC` | Cubic spline (future) | High quality |

### Export with Snap-to-Keyframe

When exporting interpolatable sources, users can choose:

1. **Interpolated export**: Export at sample times with interpolation
2. **Snap-to-keyframe**: Export nearest keyframe only (no interpolation)

The export system handles deduplication automatically when multiple sample times snap to the same keyframe.

---

## Skill 3: Continuous Time Source (ContinuousTimeSource)

**Use case**: Neural network models (4DGS, D-NeRF) that produce Gaussians at arbitrary time.

### Protocol Requirements

```python
from src.domain.interfaces import ContinuousTimeSource
from src.domain.data import GaussianData

class MyNeuralSource(ContinuousTimeSource):
    """Native continuous time support."""

    def evaluate(self, t: float) -> GaussianData:
        """REQUIRED: Evaluate model at time t."""
        ...

    @property
    def supports_batched_time(self) -> bool:
        """REQUIRED: Can evaluate_batch() be used?"""
        ...

    def evaluate_batch(self, times: list[float]) -> list[GaussianData]:
        """REQUIRED if supports_batched_time=True."""
        ...
```

### Data I/O Specification

**Input**: Model checkpoint + config
```python
config = {
    "checkpoint": "/path/to/model.pth",
    "device": "cuda",
    "time_scale": 1.0,  # Source-specific
}
```

**Output**: `GaussianData` at arbitrary time t

```python
def evaluate(self, t: float) -> GaussianData:
    """Evaluate neural network at time t."""
    with torch.no_grad():
        outputs = self.model(t)  # Your neural network

    # Convert to GaussianData
    return GaussianData(
        means=outputs["means"].cpu().numpy(),
        scales=outputs["scales"].cpu().numpy(),
        quats=outputs["quats"].cpu().numpy(),
        opacities=outputs["opacities"].cpu().numpy(),
        sh0=outputs["sh0"].cpu().numpy(),
        format_info=FormatInfo(
            is_scales_ply=False,  # Neural nets usually output linear
            is_opacities_ply=False,
            is_sh0_rgb=True,
        ),
    )
```

### Reference Implementation

```python
"""Continuous Time Source - Neural Network Example"""

from __future__ import annotations
from typing import Any
import numpy as np

from src.domain.interfaces import (
    BaseGaussianSource,
    ContinuousTimeSource,
    SourceMetadata,
)
from src.domain.data import GaussianData, FormatInfo
from src.domain.time import TimeDomain


class NeuralGaussianSource(BaseGaussianSource, ContinuousTimeSource):
    """Neural network source that produces Gaussians at any time."""

    @classmethod
    def metadata(cls) -> SourceMetadata:
        return SourceMetadata(
            name="Neural 4DGS",
            description="Neural network 4D Gaussian Splatting model",
            file_extensions=[".pth", ".ckpt"],
            version="1.0.0",
        )

    @classmethod
    def can_load(cls, path: str) -> bool:
        return path.lower().endswith((".pth", ".ckpt"))

    def __init__(self, config: dict[str, Any]) -> None:
        self._device = config.get("device", "cuda")
        self._checkpoint = config.get("checkpoint")
        self._duration = config.get("duration", 1.0)  # seconds

        # Load your model
        # self._model = load_model(self._checkpoint).to(self._device)
        self._model = None  # Placeholder

        # Time domain: continuous seconds
        self._time_domain = TimeDomain.continuous(self._duration)

    # --- BaseGaussianSource Protocol ---

    @property
    def total_frames(self) -> int:
        # For continuous sources, estimate based on 30fps
        return max(1, int(self._duration * 30))

    @property
    def time_domain(self) -> TimeDomain:
        return self._time_domain

    def get_frame_at_time(self, normalized_time: float) -> GaussianData:
        source_time = self._time_domain.from_normalized(normalized_time)
        return self.evaluate(source_time)

    def get_frame_at_source_time(self, source_time: float) -> GaussianData:
        return self.evaluate(source_time)

    # --- ContinuousTimeSource Protocol ---

    def evaluate(self, t: float) -> GaussianData:
        """Evaluate neural network at time t (seconds)."""
        import torch

        t = max(0.0, min(t, self._duration))

        # YOUR NEURAL NETWORK INFERENCE HERE
        # Example:
        # with torch.no_grad():
        #     outputs = self._model(torch.tensor([t], device=self._device))

        # Placeholder output
        n = 5000
        return GaussianData(
            means=np.random.randn(n, 3).astype(np.float32),
            scales=np.abs(np.random.randn(n, 3)).astype(np.float32) * 0.05,
            quats=self._random_quats(n),
            opacities=np.random.rand(n).astype(np.float32),
            sh0=np.random.rand(n, 3).astype(np.float32),
            format_info=FormatInfo(
                is_scales_ply=False,
                is_opacities_ply=False,
                is_sh0_rgb=True,
            ),
        )

    @property
    def supports_batched_time(self) -> bool:
        """Enable batched evaluation for efficient export."""
        return True

    def evaluate_batch(self, times: list[float]) -> list[GaussianData]:
        """Evaluate at multiple times efficiently."""
        # If your model supports batched input, use it here
        # Otherwise, fall back to sequential
        return [self.evaluate(t) for t in times]

    def _random_quats(self, n: int) -> np.ndarray:
        q = np.random.randn(n, 4).astype(np.float32)
        return q / np.linalg.norm(q, axis=1, keepdims=True)
```

### Time Domain for Continuous Sources

```python
# Continuous seconds (most common for neural networks)
TimeDomain.continuous(duration=5.0, start_time=0.0)

# Custom continuous domain
TimeDomain(
    min_time=0.0,
    max_time=10.0,
    is_continuous=True,
)
```

---

## Skill 4: Data Sink (Exporter) (DataSinkProtocol)

**Use case**: Export Gaussian data to file formats (PLY, custom formats).

### Protocol Requirements

```python
from src.domain.interfaces import DataSinkProtocol, DataSinkMetadata
from src.domain.data import GaussianData

class MySink(DataSinkProtocol):
    """Export Gaussian data to files."""

    @classmethod
    def metadata(cls) -> DataSinkMetadata:
        """REQUIRED: Return sink metadata."""
        ...

    def export(self, data: GaussianData, path: str, **options) -> None:
        """REQUIRED: Export single frame."""
        ...

    def export_sequence(
        self,
        frames: Iterator[GaussianData],
        output_dir: str,
        **options,
    ) -> int:
        """REQUIRED: Export frame sequence. Return count."""
        ...
```

### Data I/O Specification

**Input**: `GaussianData` (validated by export system)

The export system validates ALL data before passing to sinks:
- Type must be `GaussianData`, `GSData`, `GSTensor`, or Pro variants
- All required fields must be present and non-None

**Output**: Files in your format

```python
def export(self, data: GaussianData, path: str, **options) -> None:
    # Ensure data is on CPU for file writing
    data._ensure_cpu()

    # Access validated fields
    means = data.means       # [N, 3] np.ndarray
    scales = data.scales     # [N, 3] np.ndarray
    quats = data.quats       # [N, 4] np.ndarray
    opacities = data.opacities  # [N] np.ndarray
    sh0 = data.sh0           # [N, 3] np.ndarray

    # Write to file
    self._write_file(path, means, scales, quats, opacities, sh0)
```

### Reference Implementation

```python
"""Data Sink - Complete Example"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Iterator
import numpy as np

from src.domain.interfaces import DataSinkProtocol, DataSinkMetadata
from src.domain.data import GaussianData


class MyFormatSink(DataSinkProtocol):
    """Export Gaussian data to .myformat files."""

    @classmethod
    def metadata(cls) -> DataSinkMetadata:
        return DataSinkMetadata(
            name="My Format",
            description="Export to .myformat files",
            file_extension=".myformat",
            supports_animation=True,
        )

    def export(self, data: GaussianData, path: str, **options: Any) -> None:
        """Export single frame to file."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Ensure CPU data is available
        data._ensure_cpu()

        # Get export options
        precision = options.get("precision", np.float32)

        # YOUR FILE WRITING LOGIC HERE
        # Example: save as numpy archive
        np.savez_compressed(
            output_path,
            means=data.means.astype(precision),
            scales=data.scales.astype(precision),
            quats=data.quats.astype(precision),
            opacities=data.opacities.astype(precision),
            sh0=data.sh0.astype(precision),
            # Metadata
            is_scales_ply=data.format_info.is_scales_ply,
            is_opacities_ply=data.format_info.is_opacities_ply,
            is_sh0_rgb=data.format_info.is_sh0_rgb,
        )

    def export_sequence(
        self,
        frames: Iterator[GaussianData],
        output_dir: str,
        **options: Any,
    ) -> int:
        """Export sequence of frames."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Get options
        filename_pattern = options.get("filename_pattern", "frame_{:06d}.myformat")
        progress_callback = options.get("progress_callback")

        count = 0
        for idx, frame in enumerate(frames):
            filename = filename_pattern.format(idx)
            self.export(frame, str(output_path / filename), **options)
            count += 1

            # Report progress
            if progress_callback:
                progress_callback(idx, None)

        return count
```

### Export Options

Common options passed to `export()` and `export_sequence()`:

| Option | Type | Description |
|--------|------|-------------|
| `filename_pattern` | str | Pattern for sequence files |
| `progress_callback` | callable | `(current, total) -> None` |
| `precision` | dtype | Output precision |
| `convert_format` | bool | Convert to standard PLY encoding |

---

## Installing Plugins

### Method 1: Entry Points (Recommended)

Add to your package's `pyproject.toml`:

```toml
[project.entry-points."gsplay.plugins"]
my-source = "my_package.source:MySource"

[project.entry-points."gsplay.sinks"]
my-sink = "my_package.sink:MySink"
```

Plugins are auto-discovered when GSPlay starts.

### Method 2: Manual Registration

```python
from src.infrastructure.registry import SourceRegistry

# Register your source
from my_package import MySource
SourceRegistry.register("my-source", MySource)
```

### Method 3: Package Register Function

```python
# my_plugin/__init__.py
def register():
    from src.infrastructure.registry import SourceRegistry
    from .source import MySource
    SourceRegistry.register("my-source", MySource)
```

---

## Testing Plugins

### Using Test Harness

```python
from src.plugins.testing import PluginTestHarness

harness = PluginTestHarness(MySource)
results = harness.run_all_tests(
    config={"path": "/test/data"},
    device="cuda",
)
harness.print_results()
```

### Using Mock Data

```python
from src.plugins.testing import create_mock_gaussian_data

# Create test data
mock_data = create_mock_gaussian_data(
    n_gaussians=1000,
    sh_degree=0,
    seed=42,
)

# Test your sink
sink = MySink()
sink.export(mock_data, "/tmp/test.myformat")
```

### Pytest Example

```python
import pytest
from src.plugins.testing import create_mock_gaussian_data

@pytest.fixture
def mock_data():
    return create_mock_gaussian_data(n_gaussians=100, seed=42)

def test_export_single_frame(mock_data, tmp_path):
    sink = MySink()
    output = tmp_path / "test.myformat"
    sink.export(mock_data, str(output))
    assert output.exists()
```

---

## Best Practices

### 1. Always Use FormatInfo

```python
FormatInfo(
    is_scales_ply=False,    # False = linear (recommended)
    is_opacities_ply=False, # False = linear [0,1] (recommended)
    is_sh0_rgb=True,        # True = RGB [0,1] (recommended)
    sh_degree=None,         # Set if using higher-order SH
)
```

### 2. Handle Errors with PluginLoadError

```python
from src.shared.exceptions import PluginLoadError

def get_frame_at_time(self, normalized_time: float) -> GaussianData:
    try:
        return self._load_frame(index)
    except FileNotFoundError as e:
        raise PluginLoadError(
            f"Frame not found: {index}",
            plugin_name="my-source",
            recoverable=True,
            cause=e,
        )
```

### 3. Use Logging

```python
import logging
logger = logging.getLogger(__name__)

logger.info("Loaded %d frames", count)
logger.warning("Missing file: %s", path)
logger.error("Failed: %s", error)
```

### 4. Implement Caching for Keyframes

```python
def get_keyframe(self, index: int) -> GaussianData:
    if index not in self._cache:
        self._cache[index] = self._load_from_disk(index)
    return self._cache[index]
```

### 5. Support Progress Callbacks

```python
def export_sequence(self, frames, output_dir, **options):
    progress = options.get("progress_callback")
    for idx, frame in enumerate(frames):
        self.export(frame, ...)
        if progress:
            progress(idx, total)
```

---

## Reference Implementations

### Minimal Examples

- `src/plugins/examples/minimal_source.py` - Basic discrete source (~50 lines)

### Demo (Full Featured)

- `src/plugins/demo/demo_source.py` - Complete source with all features
- `src/plugins/demo/demo_sink.py` - Complete sink implementation

### Production

- `src/models/ply/optimized_model.py` - PLY sequence loader (discrete)
- `src/models/ply/interpolated_model.py` - PLY with interpolation
- `src/infrastructure/exporters/ply_sink.py` - PLY exporter

---

## Summary: Input/Output Matrix

| Plugin Type | Input | Output | Key Method |
|-------------|-------|--------|------------|
| BaseGaussianSource | Config dict | `GaussianData` | `get_frame_at_time()` |
| InterpolatableSource | Config dict | `GaussianData` | `get_keyframe()` + interpolate |
| ContinuousTimeSource | Config dict | `GaussianData` | `evaluate(t)` |
| DataSinkProtocol | `GaussianData` | Files | `export()` |

All plugins MUST:
1. Return/accept standard gsply types only
2. Include all required Gaussian fields (means, scales, quats, opacities, sh0)
3. Specify format encoding via FormatInfo
