# Plugin Development Guide

This guide explains how to create and install plugins for the Universal 4D Viewer.

## Table of Contents

1. [Overview](#overview)
2. [Plugin Types](#plugin-types)
3. [Creating a Data Source (Loader)](#creating-a-data-source-loader)
4. [Creating a Data Sink (Exporter)](#creating-a-data-sink-exporter)
5. [Installing Plugins](#installing-plugins)
6. [Using Plugins](#using-plugins)
7. [Best Practices](#best-practices)

---

## Overview

The Universal 4D Viewer uses a registry-based plugin architecture with two extension points:

```
[Files/Sources] --> DataSource --> GaussianData --> [Viewer] --> DataSink --> [Output]
```

- **DataSource (Loader)**: Loads Gaussian data from various formats
- **DataSink (Exporter)**: Exports Gaussian data to various formats

The key abstraction is `GaussianData` - a unified container that bridges all plugins.

---

## Plugin Types

### DataSource (Loader)

| Attribute | Description |
|-----------|-------------|
| Protocol | `DataSourceProtocol` |
| Registry | `DataSourceRegistry` |
| Output | `GaussianData` |
| Purpose | Load Gaussian data from files/streams |

### DataSink (Exporter)

| Attribute | Description |
|-----------|-------------|
| Protocol | `DataSinkProtocol` |
| Registry | `DataSinkRegistry` |
| Input | `GaussianData` |
| Purpose | Export Gaussian data to files |

---

## Creating a Data Source (Loader)

### Step 1: Define Your Configuration (Optional but Recommended)

```python
from dataclasses import dataclass

@dataclass
class MySourceConfig:
    """Configuration for MySource.

    Attributes
    ----------
    input_path : str
        Path to input file or directory.
    device : str
        Target device ("cuda" or "cpu").
    """
    input_path: str = "."
    device: str = "cuda"
```

### Step 2: Implement DataSourceProtocol

```python
from typing import Any
from src.domain.data import GaussianData, FormatInfo
from src.domain.interfaces import DataSourceProtocol, DataSourceMetadata

class MySource(DataSourceProtocol):
    """My custom data source."""

    # =========================================================================
    # REQUIRED CLASS METHODS
    # =========================================================================

    @classmethod
    def metadata(cls) -> DataSourceMetadata:
        """Return metadata about this source type."""
        return DataSourceMetadata(
            name="My Format",                    # Display name in UI
            description="Load .myformat files",  # Brief description
            file_extensions=[".myformat"],       # Supported extensions
            config_schema=MySourceConfig,        # Config dataclass (optional)
            supports_streaming=True,             # Can load frames on-demand?
            supports_seeking=True,               # Can jump to any frame?
        )

    @classmethod
    def can_load(cls, path: str) -> bool:
        """Check if this source can handle the given path.

        Called by auto-detection when user doesn't specify module type.
        """
        return path.lower().endswith(".myformat")

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize from configuration dictionary."""
        self._config = MySourceConfig(
            input_path=config.get("input_path", "."),
            device=config.get("device", "cuda"),
        )
        # Initialize your loader here
        self._frames = self._discover_frames()

    # =========================================================================
    # REQUIRED PROPERTIES
    # =========================================================================

    @property
    def total_frames(self) -> int:
        """Total number of frames available."""
        return len(self._frames)

    # =========================================================================
    # REQUIRED METHODS
    # =========================================================================

    def get_frame(self, index: int) -> GaussianData:
        """Get frame at specific index (0-based)."""
        if index < 0 or index >= self.total_frames:
            raise IndexError(f"Frame {index} out of range")

        # Load your data here
        means, scales, quats, opacities, sh0 = self._load_frame(index)

        return GaussianData(
            means=means,           # [N, 3] float32
            scales=scales,         # [N, 3] float32
            quats=quats,           # [N, 4] float32
            opacities=opacities,   # [N] float32
            sh0=sh0,               # [N, 3] float32
            shN=None,              # [N, K, 3] or None
            format_info=FormatInfo(
                is_scales_ply=False,     # True if log-space
                is_opacities_ply=False,  # True if logit-space
                is_sh0_rgb=True,         # True if RGB, False if SH
                sh_degree=None,          # 0, 1, 2, 3, or None
            ),
            source_path=str(self._frames[index]),
        )

    def get_frame_at_time(self, normalized_time: float) -> GaussianData:
        """Get frame at normalized time [0.0, 1.0]."""
        if self.total_frames <= 1:
            return self.get_frame(0)
        idx = int(round(normalized_time * (self.total_frames - 1)))
        idx = max(0, min(idx, self.total_frames - 1))
        return self.get_frame(idx)

    # =========================================================================
    # BACKWARD COMPATIBILITY (Required for Viewer)
    # =========================================================================

    def get_gaussians_at_normalized_time(self, normalized_time: float):
        """Return raw GSData/GSTensor for viewer compatibility."""
        frame = self.get_frame_at_time(normalized_time)
        if self._config.device.startswith("cuda"):
            return frame.to_gstensor(self._config.device)
        return frame.to_gsdata()

    def get_total_frames(self) -> int:
        """Compatibility wrapper."""
        return self.total_frames

    def get_frame_time(self, frame_idx: int) -> float:
        """Convert frame index to normalized time."""
        if self.total_frames <= 1:
            return 0.0
        return frame_idx / (self.total_frames - 1)

    # =========================================================================
    # YOUR IMPLEMENTATION
    # =========================================================================

    def _discover_frames(self) -> list:
        """Discover available frames."""
        # Your implementation here
        pass

    def _load_frame(self, index: int):
        """Load frame data from file."""
        # Your implementation here
        pass
```

### DataSourceMetadata Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Display name (e.g., "PLY Sequence") |
| `description` | `str` | Brief description |
| `file_extensions` | `list[str]` | Supported extensions [".ply", ".splat"] |
| `config_schema` | `type \| None` | Config dataclass for validation |
| `supports_streaming` | `bool` | Can load frames on-demand (default: True) |
| `supports_seeking` | `bool` | Can jump to any frame (default: True) |

---

## Creating a Data Sink (Exporter)

### Step 1: Define Your Configuration (Optional)

```python
from dataclasses import dataclass

@dataclass
class MySinkConfig:
    """Configuration for MySink export options."""
    compression: bool = True
    precision: int = 6
```

### Step 2: Implement DataSinkProtocol

```python
from pathlib import Path
from typing import Any, Iterator
from src.domain.data import GaussianData
from src.domain.interfaces import DataSinkProtocol, DataSinkMetadata

class MySink(DataSinkProtocol):
    """My custom data sink (exporter)."""

    # =========================================================================
    # REQUIRED CLASS METHODS
    # =========================================================================

    @classmethod
    def metadata(cls) -> DataSinkMetadata:
        """Return metadata about this sink type."""
        return DataSinkMetadata(
            name="My Format",                    # Display name in UI
            description="Export to .myformat",   # Brief description
            file_extension=".myformat",          # Output extension (with dot)
            supports_animation=True,             # Can export sequences?
            config_schema=MySinkConfig,          # Config dataclass (optional)
        )

    # =========================================================================
    # REQUIRED METHODS
    # =========================================================================

    def export(self, data: GaussianData, path: str, **options: Any) -> None:
        """Export single frame to file.

        Parameters
        ----------
        data : GaussianData
            Gaussian data to export (may be CPU or GPU).
        path : str
            Output file path.
        **options
            Format-specific options (compression, precision, etc.)
        """
        # Parse options
        compression = options.get("compression", True)
        precision = options.get("precision", 6)

        # Ensure output directory exists
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Ensure data is on CPU
        data._ensure_cpu()

        # Write your format
        self._write_file(
            output_path,
            data.means,
            data.scales,
            data.quats,
            data.opacities,
            data.sh0,
            data.shN,
            compression=compression,
        )

    def export_sequence(
        self,
        frames: Iterator[GaussianData],
        output_dir: str,
        **options: Any,
    ) -> int:
        """Export sequence of frames.

        Parameters
        ----------
        frames : Iterator[GaussianData]
            Iterator of frames to export.
        output_dir : str
            Output directory.
        **options
            filename_pattern : str = "frame_{:06d}.myformat"
            progress_callback : callable(frame_idx, total) = None
            + all options from export()

        Returns
        -------
        int
            Number of frames successfully exported.
        """
        filename_pattern = options.get("filename_pattern", "frame_{:06d}.myformat")
        progress_callback = options.get("progress_callback")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        exported = 0
        for idx, frame in enumerate(frames):
            try:
                filename = filename_pattern.format(idx)
                self.export(frame, str(output_path / filename), **options)
                exported += 1
                if progress_callback:
                    progress_callback(idx, None)
            except Exception as e:
                logger.error(f"Failed to export frame {idx}: {e}")
                continue

        return exported

    # =========================================================================
    # YOUR IMPLEMENTATION
    # =========================================================================

    def _write_file(self, path, means, scales, quats, opacities, sh0, shN, **kwargs):
        """Write data to file."""
        # Your implementation here
        pass
```

### DataSinkMetadata Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Display name (e.g., "PLY") |
| `description` | `str` | Brief description |
| `file_extension` | `str` | Output extension with dot (".ply") |
| `supports_animation` | `bool` | Can export sequences (default: True) |
| `config_schema` | `type \| None` | Config dataclass for validation |

---

## Installing Plugins

### Method 1: Manual Registration (Recommended for Development)

```python
# In your application startup or __init__.py
from src.infrastructure.registry import DataSourceRegistry, DataSinkRegistry

# Register your source
from my_plugin import MySource
DataSourceRegistry.register("my-format", MySource)

# Register your sink
from my_plugin import MySink
DataSinkRegistry.register("my-format", MySink)
```

### Method 2: Add to Default Registration

Edit `src/infrastructure/registry/__init__.py`:

```python
def register_default_sources() -> None:
    """Register all built-in data sources."""
    from src.models.ply.ply_source import PlyDataSource
    DataSourceRegistry.register("load-ply", PlyDataSource)

    # ADD YOUR SOURCE HERE:
    from src.plugins.my_plugin import MySource
    DataSourceRegistry.register("my-format", MySource)


def register_default_sinks() -> None:
    """Register all built-in data sinks."""
    from src.infrastructure.exporters.ply_sink import PlySink
    DataSinkRegistry.register("ply", PlySink)

    # ADD YOUR SINK HERE:
    from src.plugins.my_plugin import MySink
    DataSinkRegistry.register("my-format", MySink)
```

### Method 3: Plugin Package with Auto-Registration

Create a plugin package with registration helper:

```python
# src/plugins/my_plugin/__init__.py
from .source import MySource
from .sink import MySink

def register():
    """Register all plugins from this package."""
    from src.infrastructure.registry import DataSourceRegistry, DataSinkRegistry
    DataSourceRegistry.register("my-format", MySource)
    DataSinkRegistry.register("my-format", MySink)

__all__ = ["MySource", "MySink", "register"]
```

Then call `register()` at startup:

```python
from src.plugins.my_plugin import register
register()
```

---

## Using Plugins

### Using a Registered Source

**Via JSON Config:**

```json
{
    "module": "my-format",
    "config": {
        "input_path": "/path/to/data"
    }
}
```

**Via ModelFactory:**

```python
from src.infrastructure.model_factory import ModelFactory

model, loader, metadata = ModelFactory.create(
    module_type="my-format",
    module_config={"input_path": "/path/to/data"},
    device="cuda",
)
```

**Via Direct Instantiation:**

```python
from src.plugins.my_plugin import MySource

source = MySource({"input_path": "/path/to/data"})
frame = source.get_frame(0)
print(f"Loaded {frame.n_gaussians} gaussians")
```

### Using a Registered Sink

**Via Registry:**

```python
from src.infrastructure.registry import DataSinkRegistry

sink_class = DataSinkRegistry.get("my-format")
sink = sink_class()
sink.export(gaussian_data, "/path/to/output.myformat")
```

**Via Direct Instantiation:**

```python
from src.plugins.my_plugin import MySink

sink = MySink()
sink.export(gaussian_data, "/path/to/output.myformat", compression=True)

# Export sequence
exported = sink.export_sequence(
    frames=frame_iterator,
    output_dir="/path/to/output",
    filename_pattern="frame_{:04d}.myformat",
    progress_callback=lambda idx, total: print(f"Frame {idx}"),
)
```

---

## Best Practices

### 1. Format Information

Always set `FormatInfo` correctly to prevent processing errors:

```python
FormatInfo(
    is_scales_ply=True,      # True = log-space (raw PLY), False = linear
    is_opacities_ply=True,   # True = logit-space (raw PLY), False = linear [0,1]
    is_sh0_rgb=False,        # True = RGB [0,1], False = SH coefficients
    sh_degree=3,             # SH degree: 0, 1, 2, 3, or None
)
```

### 2. Error Handling

```python
def get_frame(self, index: int) -> GaussianData:
    try:
        # Load data
        ...
    except FileNotFoundError:
        raise RuntimeError(f"Frame {index} not found")
    except Exception as e:
        logger.error(f"Error loading frame {index}: {e}")
        raise
```

### 3. Caching

Implement caching for efficient playback:

```python
def __init__(self, config):
    self._cache: dict[int, GaussianData] = {}

def get_frame(self, index: int) -> GaussianData:
    if index in self._cache:
        return self._cache[index]

    frame = self._load_frame(index)
    self._cache[index] = frame
    return frame
```

### 4. Progress Reporting

Support progress callbacks in sequence operations:

```python
def export_sequence(self, frames, output_dir, **options):
    progress_callback = options.get("progress_callback")

    for idx, frame in enumerate(frames):
        self.export(frame, ...)
        if progress_callback:
            progress_callback(idx, total_frames)
```

### 5. Logging

Use the logging module, not print:

```python
import logging
logger = logging.getLogger(__name__)

logger.debug("Loading frame %d", index)
logger.info("Initialized with %d frames", total)
logger.error("Failed to load: %s", error)
```

### 6. Type Hints

Use Python 3.12+ type hints:

```python
def get_frame(self, index: int) -> GaussianData:
    ...

def export(self, data: GaussianData, path: str, **options: Any) -> None:
    ...
```

---

## Reference Implementations

See the demo plugins for complete working examples:

- `src/plugins/demo/demo_source.py` - DataSource reference implementation
- `src/plugins/demo/demo_sink.py` - DataSink reference implementation

Production implementations:

- `src/models/ply/ply_source.py` - PLY sequence loader
- `src/infrastructure/exporters/ply_sink.py` - PLY exporter
