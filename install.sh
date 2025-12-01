#!/bin/bash
set -e

echo -e "\033[0;36m=== gsplay Installation (Linux) ===\033[0m"
echo ""

# Step 1: Check for uv
if ! command -v uv &> /dev/null; then
    echo -e "\033[0;31mError: 'uv' is not installed. Please install it first.\033[0m"
    exit 1
fi

# Step 2: Sync dependencies
echo -e "\033[0;33m[1/5] Running uv sync...\033[0m"
uv sync

# Step 3: Install gsplat from GitHub
echo -e "\033[0;33m[2/5] Installing gsplat from GitHub and compiling CUDA extensions...\033[0m"
# Ensure build dependencies are present
uv pip install jaxtyping ninja rich packaging numpy
uv pip install git+https://github.com/nerfstudio-project/gsplat.git --no-build-isolation --no-cache-dir

# Step 4: JIT compilation
echo -e "\033[0;33m[3/5] Triggering JIT compilation...\033[0m"
uv run python -c "from gsplat.cuda._backend import _C; print('gsplat CUDA extensions compiled successfully')"

# Step 5: Build launcher frontend
echo -e "\033[0;33m[4/5] Building launcher frontend...\033[0m"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DENO_BIN=""

# Find deno
if command -v deno &> /dev/null; then
    DENO_BIN="deno"
elif [ -f "$HOME/.deno/bin/deno" ]; then
    DENO_BIN="$HOME/.deno/bin/deno"
fi

if [ -n "$DENO_BIN" ]; then
    cd "$SCRIPT_DIR/launcher/frontend"
    "$DENO_BIN" task build
    # Copy to package static directory
    rm -rf "$SCRIPT_DIR/launcher/gsplay_launcher/static"
    mkdir -p "$SCRIPT_DIR/launcher/gsplay_launcher/static"
    cp -r "$SCRIPT_DIR/launcher/frontend/dist/"* "$SCRIPT_DIR/launcher/gsplay_launcher/static/"
    cd "$SCRIPT_DIR"
    echo -e "\033[0;32mFrontend built successfully\033[0m"
else
    echo -e "\033[0;33mWarning: deno not found, skipping frontend build.\033[0m"
    echo -e "\033[0;33mLauncher will use fallback HTML dashboard.\033[0m"
    echo -e "\033[0;33mTo build frontend later: install deno and run ./launcher/build_frontend.sh\033[0m"
fi

# Step 6: Verify
echo -e "\033[0;33m[5/5] Verifying installation...\033[0m"
uv run python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); from gsplat.cuda._backend import _C; print('gsplat CUDA: OK')"

echo ""
echo -e "\033[0;32m=== Installation Complete ===\033[0m"
echo -e "\033[0;36mRun the viewer with: uv run gsplay --config <path>\033[0m"
echo -e "\033[0;36mRun the launcher with: uv run -m gsplay_launcher --browse-path <path>\033[0m"
