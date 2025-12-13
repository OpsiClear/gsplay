# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a real-time gsplay for rendering dynamic 4D Gaussian Splatting scenes. The system supports:
1. **Local PLY Mode**: Loads sequences of .ply files directly from disk
2. **Composite Mode**: Combines multiple PLY sources with layer management

## Installation & Setup

This project uses `uv` for package management:

```bash
git clone https://github.com/OpsiClear/universal_4d_gsplay.git
cd universal_4d_gsplay
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
uv run src/gsplay/main.py --config ./path/to/ply/folder

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
│   ├── entities.py          # GSTensor (single source of truth)
│   ├── interfaces.py        # ModelInterface, DataLoaderInterface
│   └── services/            # Color adjustments, transforms (pure logic)
│
├── infrastructure/          # External dependencies & I/O
│   ├── cache/
│   │   └── frame_cache.py   # BinaryFrameCache for frame caching
│   ├── io/
│   │   ├── discovery.py     # File discovery utilities
│   │   └── path_io.py       # Path I/O operations
│   ├── processing/
│   │   ├── data_validation.py
│   │   ├── gaussian_constants.py
│   │   ├── gspro_adapter.py
│   │   └── ply/             # PLY loading/writing
│   │       ├── loader.py
│   │       ├── writer.py
│   │       └── utils.py
│   ├── exporters/           # Export format implementations
│   └── model_factory.py     # Model instantiation factory
│
├── models/                  # Application layer - model implementations
│   ├── ply/
│   │   └── optimized_model.py  # OptimizedPlyModel (main implementation)
│   └── composite/
│       └── composite_model.py  # CompositeModel (multi-layer scenes)
│
├── gsplay/                  # Presentation layer
│   ├── core/                # Main entry point (main.py, app.py, api.py)
│   ├── config/              # GSPlayConfig, UIHandles, rotation utils
│   ├── rendering/           # Render pipeline, camera controller, quaternion utils
│   │   ├── camera.py        # CameraController with mode-based ownership
│   │   ├── camera_state.py  # CameraState (spherical coords primary)
│   │   ├── quaternion_utils.py # Quaternion math utilities
│   │   └── renderer.py      # Render function creation
│   ├── ui/                  # UI layout and components
│   ├── interaction/         # Event handlers
│   └── nerfview/            # Embedded nerfview viewer
│
└── shared/                  # Cross-cutting concerns
    ├── math.py              # Math utils (knn, quaternions)
    └── exceptions.py        # Custom exceptions
```

### Dependency Flow (Clean Architecture)

```
gsplay/ --> models/ --> domain/
   |          |           ^
   v          v           |
infrastructure/ ---------+
   (depends on domain interfaces)
```

### Key Components

**Domain Layer** (`src/domain/`):
- `entities.py`: GSTensor dataclass (single source of truth)
- `interfaces.py`: ModelInterface, DataLoaderInterface protocols
- `services/`: ColorAdjustmentService (pure business logic)

**Infrastructure Layer** (`src/infrastructure/`):
- `cache/frame_cache.py`: BinaryFrameCache for frame caching
- `processing/ply/`: PLY file loading, format detection, writing
- `processing/gspro_adapter.py`: GSPRO format adapter
- `model_factory.py`: Model instantiation factory
- `exporters/`: Export format implementations

**Models Layer** (`src/models/`):
- `ply/optimized_model.py`: OptimizedPlyModel (main PLY implementation)
- `composite/composite_model.py`: CompositeModel (multi-layer scenes)

**Presentation Layer** (`src/gsplay/`):
- `core/main.py`: Main entry point
- `core/app.py`: UniversalGSPlay orchestration, bake view logic
- `config/`: GSPlayConfig, UIHandles
- `rendering/camera.py`: CameraController with mode-based ownership
- `rendering/camera_state.py`: CameraState using spherical coordinates
- `rendering/quaternion_utils.py`: Quaternion math (wxyz format)

**Shared Utilities** (`src/shared/`):
- `math.py`: knn, set_random_seed
- `exceptions.py`: Custom exceptions

### Critical Implementation Details

**Format Detection**: OptimizedPlyModel automatically detects whether PLY data is in log-space or linear-space by inspecting min/max values, preventing common rendering artifacts.

**Seek Operations**: Debounced and interruptible. The UI remains responsive while dragging timeline.

## Module Configuration

The gsplay supports these module types in JSON configs:

- **"load-ply"**: Local PLY file sequences
- **"composite"**: Multi-layer composite scenes

Each module type has specific configuration parameters defined in its respective loader class.

## Common Development Commands

```bash
# Run gsplay with PLY files
uv run src/gsplay/main.py --config ./path/to/ply/folder
# OR using installed script
gsplay --config ./path/to/ply/folder
```

## Important Conventions

- Always use `uv run` to execute Python scripts (as per global CLAUDE.md)
- Use Python 3.12+ type hints with `tyro` for CLI argument parsing
- Use logging instead of print statements for info messages
- Device management: Default is "cuda:0", configured in gsplay/config.py
- Frame indexing: Zero-based throughout the codebase
- Time normalization: frame_time returns values in [0.0, 1.0] range
- **Clean Architecture**: Follow dependency rule (domain <- models <- infrastructure, gsplay)
- **Import Convention**: Use absolute imports from src.domain, src.infrastructure, etc.

## Dependencies

Core dependencies (defined in pyproject.toml):
- **viser**: Web-based 3D gsplay UI
- **gsplat**: Gaussian splatting rasterization
- **gsply/gspro**: Gaussian data handling
- **tyro**: Type-safe CLI argument parsing
- **numpy/torch**: Numerical computation

## Camera System & Viser Convention

### Viser Look-At Convention (CRITICAL)

Viser uses a **different look-at convention** than standard OpenGL:

**Viser convention:**
- `forward = normalize(look_at - position)` [toward target]
- `right = normalize(cross(forward, up_hint))`
- `up = cross(forward, right)` [NOTE: `forward × right`, not `right × forward`!]
- `R = [right, up, forward]` [camera looks down **+Z** toward target]

**OpenGL convention (DO NOT USE with viser):**
- `up = cross(right, forward)`
- `R = [right, up, -forward]` [camera looks down **-Z**]

### Bake View Implementation

The "Bake View" feature rotates/translates the model to preserve the current view when camera resets to default isometric position (az=45°, el=30°).

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
6. **Single GSTensor**: Use src.domain.entities.GSTensor (consolidated, no duplicates)
7. **Viser Camera Convention**: Never use `quat_from_euler_deg` to set `camera.wxyz` - it uses OpenGL convention which differs from viser's internal look-at computation. Always let viser compute wxyz from position/look_at/up.
