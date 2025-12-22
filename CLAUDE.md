# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a real-time gsplay for rendering dynamic 4D Gaussian Splatting scenes. The system supports:

1. **Local PLY Mode**: Loads sequences of .ply files directly from disk
2. **Composite Mode**: Combines multiple PLY sources with layer management
3. **Plugin System**: Extensible architecture for custom data sources and sinks

## Installation & Setup

This project uses `uv` for package management:

```bash
git clone https://github.com/OpsiClear/gsplay.git
cd gsplay
uv venv
# Activate the environment (.venv\Scripts\Activate.ps1 on Windows)
source .venv/bin/activate
# Install PyTorch for your CUDA version first
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
# Install the package
uv pip install -e .
```

If you encounter issues with `gsplat` or other dependencies, use:

```bash
uv sync --no-build-isolation
```

## Running the Application

### Quick Start (Local PLY Files)

```bash
# Direct folder path (simplest method)
uv run gsplay --config ./path/to/ply/folder

# Or using the installed script
gsplay --config ./path/to/ply/folder
```

### Configuration Files

Create a JSON config for more options:

```json
{
    "module": "load-ply",
    "config": {
        "ply_folder": "./path/to/ply/folder/"
    }
}
```

For composite/multi-layer scenes:

```json
{
    "module": "composite",
    "config": {
        "layers": [
            {"name": "layer1", "path": "./layer1/"},
            {"name": "layer2", "path": "./layer2/"}
        ]
    }
}
```

## Architecture Overview

### Clean Architecture Structure

The codebase follows Clean Architecture principles with clear separation of concerns:

```
src/
├── domain/                  # Core business entities & interfaces (no dependencies)
│   ├── entities.py          # GSTensor, GSData type definitions
│   ├── data.py              # GaussianData wrapper (unified CPU/GPU)
│   ├── interfaces.py        # ModelInterface, DataLoaderInterface, plugin protocols
│   ├── time.py              # TimeDomain (discrete/continuous/interpolated)
│   ├── interpolation.py     # Gaussian interpolation methods
│   ├── filters.py           # Filter definitions
│   ├── lifecycle.py         # Lifecycle management
│   └── services/
│       └── transform.py     # Pure transform logic
│
├── infrastructure/          # External dependencies & I/O
│   ├── io/
│   │   ├── discovery.py     # File discovery utilities
│   │   └── path_io.py       # Path I/O operations
│   ├── processing/
│   │   ├── gaussian_constants.py
│   │   ├── processing_mode.py
│   │   └── ply/             # PLY loading/writing
│   │       ├── loader.py
│   │       ├── writer.py
│   │       ├── format_loader.py
│   │       └── utils.py
│   ├── exporters/           # Export format implementations
│   │   ├── ply_sink.py
│   │   ├── ply_exporter.py
│   │   ├── compressed_ply_sink.py
│   │   └── factory.py
│   ├── registry/            # Plugin registry system
│   │   ├── source_registry.py
│   │   ├── sources.py
│   │   └── sinks.py
│   ├── validation/          # Config validation
│   │   └── config_validator.py
│   ├── resilience/          # Retry and circuit breaker patterns
│   │   ├── retry.py
│   │   └── circuit_breaker.py
│   ├── resources/           # Resource management
│   │   ├── gpu_manager.py
│   │   └── executor_manager.py
│   ├── health/              # Health monitoring
│   │   └── monitor.py
│   └── model_factory.py     # Model instantiation factory
│
├── models/                  # Application layer - model implementations
│   ├── ply/
│   │   ├── optimized_model.py    # OptimizedPlyModel (main implementation)
│   │   └── interpolated_model.py # Interpolated PLY model
│   └── composite/
│       └── composite_model.py    # CompositeModel (multi-layer scenes)
│
├── plugins/                 # Plugin system (see src/plugins/PLUGINS.md)
│   ├── base/                # Base classes for plugins
│   │   ├── data_source_base.py
│   │   └── decorators.py
│   ├── demo/                # Demo plugins
│   ├── examples/            # Example implementations
│   ├── testing/             # Plugin test harness
│   ├── discovery.py         # Plugin auto-discovery
│   └── PLUGINS.md           # Comprehensive plugin development guide
│
├── gsplay/                  # Presentation layer
│   ├── core/                # Main entry point
│   │   ├── main.py          # CLI entry point
│   │   ├── app.py           # UniversalGSPlay orchestration
│   │   ├── api.py           # API definitions
│   │   ├── container.py     # Dependency injection
│   │   ├── components/      # Component-based architecture
│   │   │   ├── model_component.py
│   │   │   ├── render_component.py
│   │   │   └── export_component.py
│   │   └── handlers/        # Event handlers
│   │       ├── auto_correction.py
│   │       └── color_presets.py
│   ├── config/              # Configuration
│   │   ├── settings.py      # GSPlayConfig
│   │   ├── ui_handles.py    # UIHandles
│   │   ├── rotation_conversions.py
│   │   ├── slider_constants.py
│   │   └── io.py
│   ├── rendering/           # Render pipeline
│   │   ├── renderer.py      # Main render function
│   │   ├── camera.py        # CameraController with mode-based ownership
│   │   ├── camera_state.py  # CameraState (spherical coords primary)
│   │   ├── camera_ui.py     # Camera UI controls
│   │   ├── quaternion_utils.py # Quaternion math utilities
│   │   └── jpeg_encoder.py  # GPU JPEG encoding (torchvision/nvJPEG)
│   ├── processing/          # Data transformation pipeline
│   │   ├── color.py         # Color transformations
│   │   ├── opacity.py       # Opacity adjustments
│   │   ├── volume_filter.py # Volume filtering
│   │   ├── filter_transform.py
│   │   ├── gs_bridge.py     # GaussianData <-> gsply conversion
│   │   ├── transformer.py   # Edit transformer
│   │   ├── strategies.py    # Processing strategies
│   │   ├── protocols.py     # Processing protocols
│   │   └── context.py       # Processing context
│   ├── state/               # State management
│   │   ├── edit_manager.py  # Frame-level edits
│   │   └── scene_bounds_manager.py # Volume filtering geometry
│   ├── streaming/           # WebSocket streaming (GStream)
│   │   └── websocket_server.py
│   ├── control/             # HTTP control server
│   │   └── server.py
│   ├── dispatch/            # Event dispatching
│   │   └── dispatcher.py
│   ├── interaction/         # Event handlers
│   │   ├── events.py        # Event bus system
│   │   ├── handlers.py      # UI event handlers
│   │   └── playback.py      # Playback control
│   ├── initialization/      # UI setup
│   │   └── ui_setup.py
│   ├── ui/                  # UI layout and components
│   │   ├── layout.py
│   │   ├── controller.py
│   │   ├── layers.py
│   │   ├── filter_visualizer.py
│   │   └── panels/
│   │       └── info_panel.py
│   └── nerfview/            # Embedded nerfview viewer
│       ├── viewer.py
│       ├── render_panel.py
│       └── _renderer.py
│
└── shared/                  # Cross-cutting concerns
    ├── math.py              # Math utils (knn, quaternions)
    ├── exceptions.py        # Custom exceptions
    └── perf.py              # Performance monitoring
```

### Dependency Flow (Clean Architecture)

```
gsplay/ --> models/ --> domain/
   |          |           ^
   v          v           |
infrastructure/ ---------+
   (depends on domain interfaces)

plugins/ --> domain/interfaces
   |
   v
infrastructure/registry/
```

### Key Components

**Domain Layer** (`src/domain/`):

- `entities.py`: GSTensor, GSData type re-exports from gsply
- `data.py`: GaussianData wrapper (unified CPU/GPU container)
- `interfaces.py`: ModelInterface, BaseGaussianSource, InterpolatableSource, ContinuousTimeSource, DataSinkProtocol
- `time.py`: TimeDomain with discrete/continuous/interpolated support
- `interpolation.py`: Gaussian data interpolation methods
- `services/transform.py`: Pure transform logic

**Infrastructure Layer** (`src/infrastructure/`):

- `processing/ply/`: PLY file loading, format detection, writing
- `exporters/`: Export format implementations (PlySink, CompressedPlySink)
- `registry/`: Plugin registry for sources and sinks
- `validation/`: Configuration validation
- `resilience/`: Retry and circuit breaker patterns
- `model_factory.py`: Model instantiation factory

**Models Layer** (`src/models/`):

- `ply/optimized_model.py`: OptimizedPlyModel (main PLY implementation)
- `ply/interpolated_model.py`: Interpolated PLY model
- `composite/composite_model.py`: CompositeModel (multi-layer scenes)

**Plugin System** (`src/plugins/`):

- See `src/plugins/PLUGINS.md` for comprehensive plugin development guide
- Supports: BaseGaussianSource, InterpolatableSource, ContinuousTimeSource, DataSinkProtocol
- Auto-discovery via entry points in pyproject.toml

**Presentation Layer** (`src/gsplay/`):

- `core/main.py`: Main CLI entry point
- `core/app.py`: UniversalGSPlay orchestration, bake view logic
- `core/components/`: Component-based architecture (model, render, export)
- `processing/`: Data transformation pipeline (color, opacity, filters)
- `state/`: State management (edit manager, scene bounds)
- `rendering/`: Render pipeline, camera controller, JPEG encoding
- `streaming/`: WebSocket streaming server (GStream)
- `control/`: HTTP control server for remote commands
- `interaction/events.py`: Event bus system

**Shared Utilities** (`src/shared/`):

- `math.py`: knn, set_random_seed
- `exceptions.py`: Custom exceptions (PluginLoadError, etc.)
- `perf.py`: Performance monitoring utilities

### Data Types

The system uses these Gaussian data types:

| Type | Module | Description |
|------|--------|-------------|
| `GaussianData` | `src.domain.data` | Unified CPU/GPU wrapper (preferred for plugin I/O) |
| `GSData` | `gsply` | CPU numpy container |
| `GSTensor` | `gsply.torch` | GPU PyTorch container |
| `GSDataPro` | `gsmod` | Extended CPU container |
| `GSTensorPro` | `gsmod.torch` | Extended GPU container |

### Critical Implementation Details

**Format Detection**: OptimizedPlyModel automatically detects whether PLY data is in log-space or linear-space by inspecting min/max values, preventing common rendering artifacts.

**Seek Operations**: Debounced and interruptible. The UI remains responsive while dragging timeline.

**JPEG Encoding**: Uses `torchvision.io.encode_jpeg` with nvJPEG for GPU-accelerated JPEG encoding.

## Module Configuration

The gsplay supports these module types in JSON configs:

- **"load-ply"**: Local PLY file sequences
- **"composite"**: Multi-layer composite scenes

Custom plugins can be registered via entry points:

```toml
[project.entry-points."gsplay.plugins"]
my-source = "my_package.source:MySource"

[project.entry-points."gsplay.sinks"]
my-sink = "my_package.sink:MySink"
```

## Common Development Commands

```bash
# Run gsplay with PLY files
uv run gsplay --config ./path/to/ply/folder

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html

# Pre-commit hooks
uv pip install pre-commit          # Install pre-commit
pre-commit install                  # Install git hooks
pre-commit run --all-files          # Run all hooks manually
pre-commit autoupdate               # Update hook versions

# Linting & Formatting (manual)
ruff check src/ --fix               # Lint with auto-fix
ruff format src/                    # Format code
```

## Pre-commit Hooks

This project uses pre-commit for automated code quality checks. Hooks include:

- **ruff**: Linting + formatting (replaces black, isort, flake8)
- **ty**: Type checking (Astral's fast type checker, 60x faster than mypy)
- **codespell**: Typo detection
- **bandit**: Security scanning
- **markdownlint**: Markdown linting
- **shellcheck**: Shell script linting
- **commitizen**: Conventional commit messages

Install hooks after cloning:

```bash
uv pip install -e ".[dev]"
pre-commit install
```

**Note**: ty is configured with lenient settings for gradual type adoption. See `[tool.ty]` in pyproject.toml to adjust strictness as the codebase improves.

## Important Conventions

- Always use `uv run` to execute Python scripts
- Use Python 3.12+ type hints with `tyro` for CLI argument parsing
- Use logging instead of print statements for info messages
- Device management: Default is "cuda:0", configured in gsplay/config.py
- Frame indexing: Zero-based throughout the codebase
- Time normalization: frame_time returns values in [0.0, 1.0] range
- **Clean Architecture**: Follow dependency rule (domain <- models <- infrastructure, gsplay)
- **Import Convention**: Use absolute imports from src.domain, src.infrastructure, etc.
- **Plugin Development**: See `src/plugins/PLUGINS.md` for comprehensive guide

## Dependencies

Core dependencies (defined in pyproject.toml):

- **torch/torchvision**: Deep learning framework, GPU JPEG encoding (nvJPEG)
- **viser**: Web-based 3D gsplay UI
- **gsplat**: Gaussian splatting rasterization
- **gsmod**: Gaussian splatting modifications (includes gsply)
- **tyro**: Type-safe CLI argument parsing
- **numpy**: Numerical computation
- **triton**: GPU kernel acceleration

## Camera System & Viser Convention

### Viser Look-At Convention (CRITICAL)

Viser uses a **different look-at convention** than standard OpenGL:

**Viser convention:**

- `forward = normalize(look_at - position)` [toward target]
- `right = normalize(cross(forward, up_hint))`
- `up = cross(forward, right)` [NOTE: `forward x right`, not `right x forward`!]
- `R = [right, up, forward]` [camera looks down **+Z** toward target]

**OpenGL convention (DO NOT USE with viser):**

- `up = cross(right, forward)`
- `R = [right, up, -forward]` [camera looks down **-Z**]

### Bake View Implementation

The "Bake View" feature rotates/translates the model to preserve the current view when camera resets to default isometric position (az=45 deg, el=30 deg).

**Key files:**

- `src/gsplay/core/app.py`: `_bake_camera_view()` - main bake logic
- `src/gsplay/rendering/camera.py`: `apply_to_viser()` - camera state application

**Critical implementation rules:**

1. **Use viser's wxyz directly for R_current**: Read `camera.wxyz` from viser (what rendering uses)

2. **Use `viser_look_at_matrix()` for R_default**: Matches viser's internal computation

   ```python
   def viser_look_at_matrix(position, target):
       forward = normalize(target - position)
       right = normalize(cross(forward, up_hint))
       up = cross(forward, right)  # NOT cross(right, forward)!
       return column_stack([right, up, forward])  # NOT -forward!
   ```

3. **Never set `camera.wxyz` explicitly**: Let viser compute it from position/look_at/up
   - `apply_to_viser()` sets only `position`, `look_at`, `up_direction`
   - `_bake_camera_view()` sets only `position`, `look_at`, `up_direction`

4. **View preservation formula:**

   ```
   R_delta = R_default @ R_current.T
   R_new_model = R_delta @ R_old_model
   t_new = R_delta @ t_old + t_delta + center_correction
   ```

## Key Gotchas

1. **PyTorch First**: Install PyTorch before running `uv pip install -e .`
2. **Hard Refresh**: If you see "viser Version mismatch" error, do Ctrl+Shift+R in browser
3. **Scale/Opacity Activation**: OptimizedPlyModel auto-detects format - don't manually apply exp/sigmoid without checking
4. **Scale Filtering**: UI "Max Scale" slider controls filtering (auto-initialized to 99.5th percentile from first frame)
5. **No Circular Imports**: Domain layer cannot import from infrastructure or models. Use dependency inversion.
6. **Data Types**: Use GaussianData from src.domain.data for plugin I/O (unified wrapper)
7. **Viser Camera Convention**: Never use `quat_from_euler_deg` to set `camera.wxyz` - it uses OpenGL convention which differs from viser's internal look-at computation. Always let viser compute wxyz from position/look_at/up.
8. **Plugin Registration**: Use entry points in pyproject.toml for auto-discovery
