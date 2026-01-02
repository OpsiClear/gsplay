# Quick Start

Get up and running with GSPlay in minutes.

## Installation

### Prerequisites

- Python 3.12
- NVIDIA GPU with CUDA 12.8+
- [uv](https://docs.astral.sh/uv/) package manager

### Install from Source

```bash
# Clone the repository
git clone https://github.com/opsiclear/gsplay.git
cd gsplay

# Create virtual environment
uv venv

# Activate (Linux/macOS)
source .venv/bin/activate
# Activate (Windows)
.venv\Scripts\Activate.ps1

# Install PyTorch for your CUDA version
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install GSPlay
uv pip install -e .
```

## Basic Usage

### View PLY Files

```bash
# View a folder of PLY files
gsplay /path/to/ply/folder

# Specify a port
gsplay /path/to/ply/folder --port 6020

# Enable streaming
gsplay /path/to/ply/folder --stream-port 6021
```

### Use the Launcher

The launcher provides a web-based dashboard for managing multiple viewer instances:

```bash
# Start the launcher
gsplay_launcher --browse-path /data/scenes

# Access at http://localhost:8000
```

## Configuration Files

For advanced configuration, create a JSON config file:

```json
{
    "module": "load-ply",
    "config": {
        "ply_folder": "./path/to/ply/folder/"
    }
}
```

Then run:

```bash
gsplay --config config.json
```

## Next Steps

- [Cloud Storage](cloud_storage.md) - Load PLY files from S3, GCS, or Azure
- [Plugins](plugins.md) - Create custom data sources
- [CLI Reference](../cli/gsplay.rst) - Full command-line options
