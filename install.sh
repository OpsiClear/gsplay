#!/bin/bash
# GSPlay Installation Script for Linux/macOS
set -e

INSTALL_MODE="${GSPLAY_INSTALL_MODE:-local}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --global) INSTALL_MODE="global"; shift ;;
        --local) INSTALL_MODE="local"; shift ;;
        --help|-h)
            echo "GSPlay Installer"
            echo ""
            echo "Usage: ./install.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --local   Install to project .venv/ (default)"
            echo "  --global  Install to ~/.gsplay/ with PATH integration"
            echo "  --help    Show this help message"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== GSPlay Installation (Linux/macOS) ==="
echo ""

# =============================================================================
# [1/5] Prerequisites
# =============================================================================

echo "[1/5] Checking prerequisites..."

# Check uv
if ! command -v uv &> /dev/null; then
    echo "Error: 'uv' not installed."
    echo "Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
echo "  [OK] uv: $(uv --version)"

# Detect CUDA
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: CUDA not detected."
    exit 1
fi
CUDA_VERSION=$(nvidia-smi 2>/dev/null | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" | head -1)
if [ -z "$CUDA_VERSION" ]; then
    echo "Error: Could not detect CUDA version."
    exit 1
fi
echo "  [OK] CUDA: $CUDA_VERSION"

# Map CUDA to PyTorch index
major=$(echo $CUDA_VERSION | cut -d. -f1)
minor=$(echo $CUDA_VERSION | cut -d. -f2)
if [ "$major" -eq 12 ]; then
    if [ "$minor" -ge 8 ]; then CUDA_INDEX="cu128"
    elif [ "$minor" -ge 6 ]; then CUDA_INDEX="cu126"
    elif [ "$minor" -ge 4 ]; then CUDA_INDEX="cu124"
    else CUDA_INDEX="cu121"; fi
elif [ "$major" -eq 11 ]; then CUDA_INDEX="cu118"
else CUDA_INDEX="cu121"; fi
echo "  [OK] PyTorch index: $CUDA_INDEX"

# =============================================================================
# [2/5] Setup Environment
# =============================================================================

echo ""
echo "[2/5] Setting up environment..."

if [ "$INSTALL_MODE" = "global" ]; then
    INSTALL_DIR="$HOME/.gsplay"
    mkdir -p "$INSTALL_DIR"
    if [ ! -d "$INSTALL_DIR/venv" ]; then uv venv "$INSTALL_DIR/venv" --python 3.12; fi
    source "$INSTALL_DIR/venv/bin/activate"
    export UV_PROJECT_ENVIRONMENT="$INSTALL_DIR/venv"
    echo "  [OK] Global: $INSTALL_DIR"
else
    echo "  [OK] Local: .venv/"
fi

# =============================================================================
# [3/5] Install Dependencies
# =============================================================================

echo ""
echo "[3/5] Installing dependencies..."

# PyTorch
uv pip install torch torchvision --index-url "https://download.pytorch.org/whl/$CUDA_INDEX"

# gsplay
if [ "$INSTALL_MODE" = "global" ]; then
    uv pip install -e "$SCRIPT_DIR"
else
    uv sync
fi

# =============================================================================
# [4/5] gsplat (CUDA compilation)
# =============================================================================

echo ""
echo "[4/5] Installing gsplat..."

uv pip install gsplat --no-build-isolation --no-cache-dir

# =============================================================================
# [5/5] Verify & Frontend
# =============================================================================

echo ""
echo "[5/5] Verifying..."

run_cmd="uv run python"
if [ "$INSTALL_MODE" = "global" ]; then run_cmd="python"; fi

$run_cmd -c "import torch; from gsplat.cuda._backend import _C; print('  [OK] PyTorch:', torch.__version__); print('  [OK] CUDA:', torch.cuda.is_available()); print('  [OK] gsplat: compiled')"

# Optional: Build frontend
if command -v deno &> /dev/null && [ -d "$SCRIPT_DIR/launcher/frontend" ]; then
    cd "$SCRIPT_DIR/launcher/frontend"
    deno task build &>/dev/null || true
    if [ -d "dist" ]; then
        rm -rf "$SCRIPT_DIR/launcher/gsplay_launcher/static"
        cp -r dist "$SCRIPT_DIR/launcher/gsplay_launcher/static"
        echo "  [OK] Frontend built"
    fi
    cd "$SCRIPT_DIR"
fi

# =============================================================================
# Done
# =============================================================================

echo ""
echo "=== Installation Complete ==="
echo ""
if [ "$INSTALL_MODE" = "global" ]; then
    echo "Run: gsplay <path-to-ply-folder>"
else
    echo "Run: uv run gsplay <path-to-ply-folder>"
fi
