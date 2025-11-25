<div align="center">

# Universal 4D Viewer

### Real-Time Viewing & Rendering for Dynamic 4D Gaussian Splatting Scenes

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[Features](#key-features) | [Installation](#installation) | [Quick Start](#quick-start) | [Documentation](#documentation) | [Architecture](#system-architecture)

</div>

---

## Overview

**Universal 4D Viewer** is a high-performance, real-time viewer for rendering dynamic 4D Gaussian Splatting scenes. Supports PLY files and GIFStream checkpoint-based models for viewing complex, dynamic 3D content.

**Key Capabilities:**
- **Jellyfin Integration**: Stream pre-compressed 4D scenes from standard Jellyfin media servers
- **Local PLY Playback**: Load and render sequences of `.ply` files directly from disk
- **Real-Time Performance**: GPU-accelerated decompression and rendering at 60+ FPS
- **Multi-Stream Synchronization**: Perfect frame alignment across 11+ video streams
- **Interactive Navigation**: Full camera controls with responsive seek and variable playback speed

---

## Key Features

- **Real-time Decompression & Rendering**: GPU-accelerated pipeline using custom compression library to decode multiple video streams into Gaussian Splatting parameters
- **Robust Multi-Stream Synchronization**: Sophisticated synchronization manager ensures perfect frame alignment across all streams, handling network jitter and preventing visual artifacts
- **Responsive Seek Functionality**: Debounced and interruptible seek system with clean stream re-synchronization
- **Variable Playback Speed**: Rendering speed decoupled from source framerate for smooth playback at any speed
- **Clean Architecture**: Modular design following Clean Architecture principles for maintainability and extensibility
- **Optimized PLY Loading**: High-performance PLY I/O with automatic format detection and caching

---

## Installation

### Prerequisites

- **Python 3.10+**
- **CUDA-capable GPU** (NVIDIA GPU with CUDA 11.8+ or 12.x)
- **Visual Studio** with C++ build tools (required for compiling native extensions)
- **uv** package manager ([Install uv](https://github.com/astral-sh/uv))

### 1. Clone the Repository

This project includes the `third_party/compression` submodule (nerfview is now embedded under `src/viewer/nerfview`). Clone recursively:

```bash
git clone https://github.com/OpsiClear/universal_4d_viewer.git --recursive
cd universal_4d_viewer
```

### 2. Build in Visual Studio Developer Console

Open **Developer Command Prompt for Visual Studio** (or **Developer PowerShell for Visual Studio**) and navigate to the project directory:

```bash
# Sync all dependencies and create virtual environment
uv sync

# Install gsplat separately
uv pip install gsplat
```

This will:
- Create a virtual environment in `.venv/`
- Install all dependencies from `pyproject.toml`
- Install `compression` library from `third_party/compression`
- Install the embedded viewer utilities (vendored from nerfview) inside `src/viewer/nerfview`
- Install PyTorch with appropriate CUDA support
- Set up the `src` package for development
- Install `gsplat` for Gaussian splatting rendering (requires Visual Studio C++ compiler)

---

## Quick Start

### Local PLY Files (Simplest)

```bash
# Direct folder path
viewer --config ./path/to/ply/folder

# Or with Python
uv run src/viewer/main.py --config ./path/to/ply/folder
```

### Jellyfin Streaming

1. **Place `meta_bundle.json`** (from compression tool) in `assets/` directory
2. **Configure Jellyfin settings** in a JSON file in `module_config/`
3. **Run the viewer:**

```bash
viewer --config ./module_config/gif_elly.json
```

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

**Jellyfin Streaming:**
```json
{
    "module": "sogs",
    "config": {
        "jellyfin_url": "https://your-server.com",
        "api_key": "your-api-key",
        "video_ids": ["id1", "id2", ...]
    }
}
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
├── domain/                  # Core business logic (no infrastructure deps)
│   ├── entities.py, interfaces.py
│   └── services/            # Pure math helpers (color, transform, ...)
│
├── infrastructure/          # External dependencies & I/O adapters
│   ├── config.py, processing_mode.py, model_factory.py
│   ├── io/                  # Filesystem + streaming helpers
│   │   ├── path_io.py       # UniversalPath (local/S3/GCS/Azure)
│   │   ├── discovery.py     # discover_and_sort_ply_files()
│   │   └── streaming.py     # Jellyfin VideoManager client
│   ├── cache/frame_cache.py # Hybrid RAM+disk BinaryFrameCache
│   ├── processing/          # gspro + PLY activation/loader/writer
│   │   ├── gaussian_constants.py, data_validation.py, gspro_adapter.py
│   │   └── ply/{activation_service,loader,writer,utils}.py
│   └── exporters/
│       ├── factory.py
│       ├── ply_exporter.py
│       └── compressed_ply_exporter.py
│
├── models/
│   ├── ply/optimized_model.py
│   ├── gifstream/{loader.py, model.py}
│   └── composite/composite_model.py
│
├── viewer/
│   ├── main.py, app.py, config.py, rendering.py
│   ├── container.py, edit_manager.py, events.py, handlers.py, ui.py
│   ├── processing/          # GSBridge + processors/strategies/volume filter
│   ├── components/          # Model/render/export components
│   └── nerfview/            # Embedded viser-based viewer
│
└── shared/
    ├── math.py, perf.py
    └── exceptions.py
```

Layering guidelines:
- **Domain** contains only dataclasses, protocols, and stateless services that work with plain tensors/arrays—no viewer, gspro, or filesystem imports.
- **Infrastructure** adapts frameworks and external services (Jellyfin, gspro, filesystem, exporters) to the domain interfaces; it can depend on domain but never on viewer code.
- **Models + Viewer** make up the application/presentation layer: they orchestrate user interactions, rendering, and model lifecycles while calling into domain services for pure math and infrastructure for I/O.

### Multi-Threaded Pipeline

```
Jellyfin Server (Compressed MP4 streams)
    |
    v
VideoManager (infrastructure/streaming.py)
  - Opens cv2.VideoCapture for each stream
  - Dedicated thread per stream for buffering
  - Backpressure to prevent buffer overruns
  - Synchronized seeking across all streams
    |
    v
StreamingModel (models/streaming/model.py)
  - Requests synchronized frame bundles
  - Passes data to CompressionAdapter
  - Holds render-ready Gaussian data
    |
    v
Viewer (viewer/main.py)
  - Provides UI using viser
  - Drives animation loop
  - Renders using gsplat
```

---

## Project Structure

```
universal_4d_viewer/
├── README.md                   # You are here
├── src/                        # Main source code
│   ├── domain/                 # Core business logic
│   ├── infrastructure/         # External dependencies & I/O
│   ├── models/                 # Application layer
│   ├── viewer/                 # Presentation layer
│   └── shared/                 # Cross-cutting utilities
├── tests/                      # Unit and integration tests
├── docs/                       # Documentation (see section above)
│   ├── GUIDE.md                # User guide and configuration
│   ├── CONTRIBUTING.md         # Developer guide
│   ├── CHANGELOG.md            # Version history
│   └── archive/                # Historical development docs
├── third_party/                # Git submodules
│   ├── compression/            # Video compression/decompression (submodule)
│   └── MLEntropy/              # ML entropy encoding utilities
├── datasets/                   # Dataset loaders
├── gsplat_utils/               # Custom gsplat utilities
├── module_config/              # Runtime configuration examples
├── pyproject.toml              # Package configuration
└── CLAUDE.md                   # Project development guidelines
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

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **gsplat**: GPU-accelerated Gaussian splatting
- **viser**: Web-based 3D visualization
- **Jellyfin**: Open-source media server
- **nerfview**: NeRF rendering and camera controls (core viewer logic now vendored into `src/viewer/nerfview`)

---

## Citation

If you use this work in your research, please cite:

```bibtex
@software{universal_4d_viewer,
  title={Universal 4D Viewer: Real-Time Streaming for Dynamic Gaussian Splatting},
  author={OpsiClear},
  year={2024},
  url={https://github.com/OpsiClear/universal_4d_viewer}
}
```

---

<div align="center">

**Made with Python, PyTorch, and CUDA**

[Report Bug](https://github.com/OpsiClear/universal_4d_viewer/issues) | [Request Feature](https://github.com/OpsiClear/universal_4d_viewer/issues)

</div>
