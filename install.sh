#!/bin/bash
set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}=== GSPlay Installation (Linux/macOS) ===${NC}"
echo ""

# =============================================================================
# Configuration
# =============================================================================

# Installation mode: "local" (project .venv) or "global" (~/.gsplay)
INSTALL_MODE="${GSPLAY_INSTALL_MODE:-local}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --global)
            INSTALL_MODE="global"
            shift
            ;;
        --local)
            INSTALL_MODE="local"
            shift
            ;;
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
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# =============================================================================
# Prerequisites Check
# =============================================================================

echo -e "${YELLOW}[1/6] Checking prerequisites...${NC}"

# Check for uv
if ! command -v uv &> /dev/null; then
    echo -e "${RED}Error: 'uv' is not installed.${NC}"
    echo "Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
echo "  ✓ uv found: $(uv --version)"

# Check for CUDA
detect_cuda_version() {
    # Try nvidia-smi first
    if command -v nvidia-smi &> /dev/null; then
        cuda_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
        if [ -n "$cuda_version" ]; then
            # Extract CUDA version from nvidia-smi output
            cuda_version=$(nvidia-smi 2>/dev/null | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" | head -1)
            if [ -n "$cuda_version" ]; then
                echo "$cuda_version"
                return 0
            fi
        fi
    fi

    # Try nvcc
    if command -v nvcc &> /dev/null; then
        cuda_version=$(nvcc --version 2>/dev/null | grep -oP "release \K[0-9]+\.[0-9]+" | head -1)
        if [ -n "$cuda_version" ]; then
            echo "$cuda_version"
            return 0
        fi
    fi

    return 1
}

CUDA_VERSION=$(detect_cuda_version) || true
if [ -z "$CUDA_VERSION" ]; then
    echo -e "${RED}Error: CUDA not detected (nvidia-smi not found).${NC}"
    echo "GSPlay requires an NVIDIA GPU with CUDA support."
    echo ""
    echo "To fix:"
    echo "  1. Install NVIDIA drivers: https://www.nvidia.com/drivers"
    echo "  2. Verify with: nvidia-smi"
    echo "  3. Run this installer again"
    exit 1
fi
echo "  ✓ CUDA detected: $CUDA_VERSION"

# Map CUDA version to PyTorch index
get_torch_index() {
    local cuda=$1
    local major=$(echo $cuda | cut -d. -f1)
    local minor=$(echo $cuda | cut -d. -f2)

    # Map to supported PyTorch CUDA versions
    if [ "$major" -eq 12 ]; then
        if [ "$minor" -ge 8 ]; then
            echo "cu128"
        elif [ "$minor" -ge 6 ]; then
            echo "cu126"
        elif [ "$minor" -ge 4 ]; then
            echo "cu124"
        else
            echo "cu121"
        fi
    elif [ "$major" -eq 11 ]; then
        echo "cu118"
    else
        echo "cu121"  # Default fallback
    fi
}

CUDA_INDEX=$(get_torch_index "$CUDA_VERSION")
TORCH_INDEX_URL="https://download.pytorch.org/whl/${CUDA_INDEX}"
echo "  ✓ PyTorch index: $TORCH_INDEX_URL"

# =============================================================================
# Setup Installation Directory
# =============================================================================

echo ""
if [ "$INSTALL_MODE" = "global" ]; then
    echo -e "${YELLOW}[2/6] Setting up global installation (~/.gsplay/)...${NC}"

    INSTALL_DIR="$HOME/.gsplay"
    mkdir -p "$INSTALL_DIR"

    # Create virtual environment
    if [ ! -d "$INSTALL_DIR/venv" ]; then
        uv venv "$INSTALL_DIR/venv" --python 3.12
    fi

    # Activate for this script
    source "$INSTALL_DIR/venv/bin/activate"

    # Set UV_PROJECT_ENVIRONMENT so uv uses our venv
    export UV_PROJECT_ENVIRONMENT="$INSTALL_DIR/venv"
else
    echo -e "${YELLOW}[2/6] Setting up local installation (.venv/)...${NC}"
    INSTALL_DIR="$(pwd)"
fi

# =============================================================================
# Install Dependencies
# =============================================================================

echo ""
echo -e "${YELLOW}[3/6] Installing PyTorch with CUDA ${CUDA_VERSION} support...${NC}"
uv pip install torch torchvision --index-url "$TORCH_INDEX_URL"

echo ""
echo -e "${YELLOW}[4/6] Installing gsplay and dependencies...${NC}"
if [ "$INSTALL_MODE" = "global" ]; then
    # For global install, install from current directory or PyPI
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    if [ -f "$SCRIPT_DIR/pyproject.toml" ]; then
        uv pip install -e "$SCRIPT_DIR"
    else
        uv pip install gsplay
    fi
else
    # For local install, use uv sync
    uv sync
fi

echo ""
echo -e "${YELLOW}[5/6] Installing gsplat (compiling CUDA extensions)...${NC}"
echo "  This may take a few minutes..."
uv pip install jaxtyping ninja rich packaging numpy
uv pip install gsplat --no-build-isolation --no-cache-dir

# Trigger JIT compilation
echo "  Triggering JIT compilation..."
python -c "from gsplat.cuda._backend import _C; print('  ✓ gsplat CUDA extensions compiled')" 2>/dev/null || \
    uv run python -c "from gsplat.cuda._backend import _C; print('  ✓ gsplat CUDA extensions compiled')"

# =============================================================================
# Build Launcher Frontend (Optional)
# =============================================================================

echo ""
echo -e "${YELLOW}[6/6] Building launcher frontend...${NC}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DENO_BIN=""

# Find deno
if command -v deno &> /dev/null; then
    DENO_BIN="deno"
elif [ -f "$HOME/.deno/bin/deno" ]; then
    DENO_BIN="$HOME/.deno/bin/deno"
fi

if [ -n "$DENO_BIN" ] && [ -d "$SCRIPT_DIR/launcher/frontend" ]; then
    cd "$SCRIPT_DIR/launcher/frontend"
    "$DENO_BIN" task build 2>/dev/null || true
    if [ -d "dist" ]; then
        rm -rf "$SCRIPT_DIR/launcher/gsplay_launcher/static"
        mkdir -p "$SCRIPT_DIR/launcher/gsplay_launcher/static"
        cp -r dist/* "$SCRIPT_DIR/launcher/gsplay_launcher/static/"
        echo "  ✓ Frontend built successfully"
    fi
    cd "$SCRIPT_DIR"
else
    echo "  ⚠ deno not found, skipping frontend build"
    echo "    Launcher will use fallback HTML dashboard"
fi

# =============================================================================
# PATH Integration (Global Install Only)
# =============================================================================

if [ "$INSTALL_MODE" = "global" ]; then
    echo ""
    echo -e "${YELLOW}Setting up PATH integration...${NC}"

    # Create wrapper scripts
    mkdir -p "$HOME/.local/bin"

    cat > "$HOME/.local/bin/gsplay" << 'WRAPPER'
#!/bin/bash
source "$HOME/.gsplay/venv/bin/activate"
python -m src.gsplay.core.main "$@"
WRAPPER
    chmod +x "$HOME/.local/bin/gsplay"

    cat > "$HOME/.local/bin/gsplay-launcher" << 'WRAPPER'
#!/bin/bash
source "$HOME/.gsplay/venv/bin/activate"
python -m gsplay_launcher "$@"
WRAPPER
    chmod +x "$HOME/.local/bin/gsplay-launcher"

    # Check if ~/.local/bin is in PATH
    if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
        echo ""
        echo -e "${YELLOW}Add this to your shell profile (~/.bashrc or ~/.zshrc):${NC}"
        echo ""
        echo "  export PATH=\"\$HOME/.local/bin:\$PATH\""
        echo ""
        echo "Then restart your shell or run: source ~/.bashrc"
    else
        echo "  ✓ ~/.local/bin is already in PATH"
    fi
fi

# =============================================================================
# Verification
# =============================================================================

echo ""
echo -e "${YELLOW}Verifying installation...${NC}"
if [ "$INSTALL_MODE" = "global" ]; then
    python -c "
import torch
print(f'  ✓ PyTorch: {torch.__version__}')
print(f'  ✓ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  ✓ GPU: {torch.cuda.get_device_name(0)}')
from gsplat.cuda._backend import _C
print('  ✓ gsplat: OK')
"
else
    uv run python -c "
import torch
print(f'  ✓ PyTorch: {torch.__version__}')
print(f'  ✓ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  ✓ GPU: {torch.cuda.get_device_name(0)}')
from gsplat.cuda._backend import _C
print('  ✓ gsplat: OK')
"
fi

# =============================================================================
# Done
# =============================================================================

echo ""
echo -e "${GREEN}=== Installation Complete ===${NC}"
echo ""
if [ "$INSTALL_MODE" = "global" ]; then
    echo -e "${CYAN}Run the viewer:${NC}"
    echo "  gsplay <path-to-ply-folder>"
    echo ""
    echo -e "${CYAN}Run the launcher:${NC}"
    echo "  gsplay-launcher --browse-path <path>"
else
    echo -e "${CYAN}Run the viewer:${NC}"
    echo "  uv run gsplay <path-to-ply-folder>"
    echo ""
    echo -e "${CYAN}Run the launcher:${NC}"
    echo "  uv run -m gsplay_launcher --browse-path <path>"
fi
