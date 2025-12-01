<div align="center">

# gsplay

### Real-Time Viewing & Rendering for Dynamic 4D Gaussian Splatting Scenes

[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9%2B-EE4C2C.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[Features](#key-features) | [Installation](#installation) | [Quick Start](#quick-start) | [Documentation](#documentation) | [Architecture](#system-architecture)

</div>

---

## Overview

**gsplay** is a high-performance, real-time viewer for rendering dynamic 4D Gaussian Splatting scenes. Load and play sequences of PLY files with an intuitive web-based interface, manage multiple instances through a modern launcher dashboard, and stream live previews to monitor your scenes.

**Key Capabilities:**
- **Local PLY Playback**: Load and render sequences of `.ply` files directly from disk
- **Web-Based Launcher**: Manage multiple viewer instances with a modern dashboard UI
- **Real-Time Streaming**: WebSocket-based live preview streaming from running instances
- **Real-Time Performance**: GPU-accelerated rendering at 60+ FPS
- **Interactive Navigation**: Full camera controls with responsive seek and variable playback speed
- **Jellyfin Integration** *(Coming Soon)*: Stream pre-compressed 4D scenes from media servers

---

## Key Features

- **High-Performance PLY Loading**: Optimized PLY I/O with automatic format detection and frame caching
- **Multi-Instance Launcher**: Web-based dashboard to launch, monitor, and manage multiple viewer instances
- **Live Stream Preview**: Real-time WebSocket streaming of viewer output with recording support
- **Responsive Seek**: Debounced and interruptible seek system for smooth timeline scrubbing
- **Variable Playback Speed**: Rendering speed decoupled from source framerate
- **Clean Architecture**: Modular design following Clean Architecture principles for maintainability
- **Launch History**: Quick access to recently launched scenes with one-click relaunch

---

## Installation

### Prerequisites

- **Python 3.12** (3.13 not yet supported)
- **CUDA-capable GPU** (NVIDIA GPU with CUDA 12.8)
- **Visual Studio 2022** with C++ build tools (Windows only, required for compiling gsplat)
- **uv** package manager ([Install uv](https://github.com/astral-sh/uv))

### 1. Clone the Repository

```bash
git clone https://github.com/opsiclear/gsplay.git
cd gsplay
```

### 2. Run Installation Script

**Windows:**
```powershell
.\install.ps1
```

**Linux:**
```bash
chmod +x install.sh
./install.sh
```

The installation script will:
1. Set up MSVC compiler environment (Windows only)
2. Run `uv sync` to install all dependencies from `pyproject.toml`
3. Install `gsplat` from GitHub with JIT compilation for CUDA 12.8
4. Verify the installation

This creates a virtual environment in `.venv/` with:
- PyTorch 2.9.1 with CUDA 12.8 support
- JIT-compiled `gsplat` for optimal performance
- All other dependencies including `viser`, `gsmod`, etc.

---

## Quick Start

### Option 1: Direct CLI (Single Instance)

```bash
# Launch viewer with a PLY folder
uv run gsplay --config ./path/to/ply/folder

# Or with explicit module config
uv run python -m gsplay.core.main --config ./path/to/ply/folder
```

### Option 2: Web Launcher (Recommended for Multiple Instances)

```bash
# Start the launcher with a browse directory
uv run -m gsplay_launcher --browse-path /path/to/your/data

# Open http://localhost:8000 in your browser
```

The launcher provides:
- File browser to navigate and launch PLY folders
- GPU selection and instance configuration
- Live stream preview with recording
- Launch history for quick access to recent scenes

### Configuration Examples

**Local PLY (JSON config):**
```json
{
    "module": "load-ply",
    "config": {
        "ply_folder": "./export_with_edits/"
    }
}
```

**Launcher CLI Options:**
```bash
uv run -m gsplay_launcher \
  --browse-path /data/scenes \
  --port 8000 \
  --history-limit 10
```

---

## Usage

Once running, the viewer will print a URL (e.g., `http://localhost:6019`). Open this in your browser.

### Controls

- **Camera**:
  - Rotate: Left-click + drag
  - Pan: Right-click + drag
  - Zoom: Scroll wheel
- **Playback**:
  - Auto Play: Toggle checkbox to start/pause animation
  - Play Speed: Adjust FPS slider to control playback speed
  - Seek: Drag time slider to jump to specific frames

### Troubleshooting

**"viser Version mismatch" error:**
- Perform a hard refresh in your browser:
  - Windows/Linux: `Ctrl + Shift + R`
  - Mac: `Cmd + Shift + R`

---

## System Architecture


Following **Clean Architecture** principles for clear separation of concerns:

```
src/
├── gsplay/                  # Main viewer application
│   ├── core/                # App entry point (main.py, app.py)
│   ├── config/              # Configuration and UI handles
│   ├── rendering/           # Render pipeline and camera
│   ├── streaming/           # WebSocket stream server
│   ├── ui/                  # UI layout and components
│   └── nerfview/            # Embedded viser-based viewer
│
├── domain/                  # Core business logic (no infrastructure deps)
│   ├── entities.py          # GSTensor dataclass
│   ├── interfaces.py        # Model/DataLoader protocols
│   └── services/            # Pure math helpers (color, transform)
│
├── infrastructure/          # External dependencies & I/O adapters
│   ├── io/                  # Filesystem helpers
│   │   ├── path_io.py       # Path utilities
│   │   └── discovery.py     # PLY file discovery
│   ├── cache/               # Frame caching
│   ├── processing/          # PLY loader/writer/activation
│   └── exporters/           # Export format implementations
│
├── models/
│   ├── ply/                 # OptimizedPlyModel
│   └── composite/           # CompositeModel (multi-layer)
│
└── shared/
    ├── math.py              # Math utilities
    └── exceptions.py        # Custom exceptions
```

Layering guidelines:
- **Domain** contains only dataclasses, protocols, and stateless services that work with plain tensors/arrays—no viewer or filesystem imports.
- **Infrastructure** adapts frameworks and external services (PLY I/O, caching, exporters) to the domain interfaces; it can depend on domain but never on viewer code.
- **Models + Viewer** make up the application/presentation layer: they orchestrate user interactions, rendering, and model lifecycles while calling into domain services for pure math and infrastructure for I/O.

### Data Flow

```
PLY Files on Disk
    |
    v
OptimizedPlyModel (models/ply/optimized_model.py)
  - Discovers and sorts PLY files
  - Auto-detects format (log-space vs linear)
  - Caches frames for fast playback
    |
    v
GSPlay Viewer (gsplay/core/app.py)
  - Provides UI using viser
  - Drives animation loop
  - Renders using gsplat
    |
    v
WebSocket Stream Server (gsplay/streaming/)
  - JPEG-encoded frame streaming
  - Connected to launcher for preview
```

### Launcher Architecture

```
Web Browser (SolidJS Frontend)
    |
    v
FastAPI Backend (gsplay_launcher/)
  - Instance lifecycle management
  - File browser with PLY detection
  - WebSocket proxy for streams
    |
    v
GSPlay Instances (subprocess per instance)
  - Independent viewer processes
  - Configurable GPU assignment
  - State persisted across restarts
```

---

## Project Structure

```
gsplay/
├── README.md                   # You are here
├── install.ps1                 # Windows installation script
├── install.sh                  # Linux installation script
├── src/                        # Main source code
│   ├── gsplay/                 # Viewer application
│   │   ├── core/               # Core app logic (main.py, app.py)
│   │   ├── rendering/          # Render pipeline
│   │   ├── streaming/          # WebSocket stream server
│   │   ├── ui/                 # UI layout and components
│   │   └── nerfview/           # Embedded viser viewer
│   ├── domain/                 # Domain models and interfaces
│   ├── infrastructure/         # I/O adapters (PLY, cache, etc.)
│   ├── models/                 # Model implementations (PLY, composite)
│   └── shared/                 # Shared utilities
├── launcher/                   # Web-based launcher
│   ├── gsplay_launcher/        # FastAPI backend
│   │   ├── api/                # REST API routes
│   │   ├── services/           # Instance management
│   │   └── main.py             # CLI entry point
│   └── frontend/               # SolidJS frontend
│       └── src/                # Components and stores
├── tests/                      # Unit and integration tests
├── pyproject.toml              # Package configuration
└── uv.lock                     # Locked dependencies
```

---

## Development

### Running Tests

```bash
# Install test dependencies
uv pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=src --cov-report=html
```

### Code Style

This project uses:
- **Type hints**: Python 3.12+ type syntax
- **CLI tools**: `tyro` for type-safe argument parsing
- **Logging**: Use logging instead of print statements
- **Package manager**: `uv run` for all script execution

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Follow Clean Architecture principles
4. Add tests for new functionality
5. Commit changes (`git commit -m 'Add amazing feature'`)
6. Push to branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

---

## Documentation

The documentation is organized into three focused guides:

- **[docs/GUIDE.md](docs/GUIDE.md)** - Complete user guide covering installation, configuration, and usage
- **[docs/CONTRIBUTING.md](docs/CONTRIBUTING.md)** - Developer guide with architecture, conventions, and contribution guidelines
- **[docs/CHANGELOG.md](docs/CHANGELOG.md)** - Version history and development phases
- **[docs/archive/](docs/archive/)** - Historical development documentation

Quick answers:
- **First time here?** Start with [docs/GUIDE.md](docs/GUIDE.md)
- **Want to contribute?** See [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md)
- **Looking for version history?** Check [docs/CHANGELOG.md](docs/CHANGELOG.md)
- **Need legacy/dev docs?** Browse [docs/archive/](docs/archive/)

---

## Performance

### Optimized PLY Loading
- **Format Auto-Detection**: Automatically detects log-space vs linear-space PLY data
- **Caching**: Smart caching system for fast frame switching
- **Parallel Loading**: Multi-threaded PLY file loading

### GPU Acceleration
- **CUDA-optimized**: All rendering and decompression on GPU
- **Low-end GPU support**: Configurable quality settings for varied hardware
- **Efficient Memory**: Smart buffer management to prevent overruns

---

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

This project builds upon excellent open-source work:

### Core Dependencies

- **[viser](https://github.com/nerfstudio-project/viser)** - The web-based 3D visualization framework that powers the viewer UI. Viser provides the interactive scene graph, camera controls, and GUI components. Licensed under MIT.

- **[nerfview](https://github.com/hangg7/nerfview)** - The original NeRF viewer implementation that inspired and provided the foundation for our rendering architecture. Core viewer logic has been adapted and extended in `src/gsplay/nerfview/`. Licensed under MIT.

- **[gsplat](https://github.com/nerfstudio-project/gsplat)** - GPU-accelerated Gaussian splatting rasterization library. Provides the high-performance CUDA kernels for real-time rendering. Licensed under Apache 2.0.

### Other Dependencies

- **[gsply](https://github.com/opsiclear/gsply)** - PLY file I/O for Gaussian Splatting data
- **[Jellyfin](https://jellyfin.org/)** *(Planned)* - Open-source media server for future streaming support

---

## Citation

If you use this work in your research, please cite:

```bibtex
@software{gsplay,
  title={gsplay: Real-Time Viewing for Dynamic 4D Gaussian Splatting},
  author={OpsiClear},
  year={2024-2025},
  url={https://github.com/opsiclear/gsplay}
}
```

---

<div align="center">

**Made with Python, PyTorch, and CUDA**

[Report Bug](https://github.com/opsiclear/gsplay/issues) | [Request Feature](https://github.com/opsiclear/gsplay/issues)

</div>
