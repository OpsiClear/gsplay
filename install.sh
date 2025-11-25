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
echo -e "\033[0;33m[1/4] Running uv sync...\033[0m"
uv sync

# Step 3: Install gsplat from GitHub
echo -e "\033[0;33m[2/4] Installing gsplat from GitHub and compiling CUDA extensions...\033[0m"
# Ensure build dependencies are present
uv pip install jaxtyping ninja rich packaging numpy
uv pip install git+https://github.com/nerfstudio-project/gsplat.git --no-build-isolation --no-cache-dir

# Step 4: JIT compilation
echo -e "\033[0;33m[3/4] Triggering JIT compilation...\033[0m"
uv run python -c "from gsplat.cuda._backend import _C; print('gsplat CUDA extensions compiled successfully')"

# Step 5: Verify
echo -e "\033[0;33m[4/4] Verifying installation...\033[0m"
uv run python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); from gsplat.cuda._backend import _C; print('gsplat CUDA: OK')"

echo ""
echo -e "\033[0;32m=== Installation Complete ===\033[0m"
echo -e "\033[0;36mRun the viewer with: uv run gsplay --config <path>\033[0m"
