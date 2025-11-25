# Universal 4D Viewer Installation Script
# JIT compiles gsplat with Python 3.12 + PyTorch 2.9.1 + CUDA 12.8

$ErrorActionPreference = "Stop"

Write-Host "=== Universal 4D Viewer Installation (JIT Compile) ===" -ForegroundColor Cyan
Write-Host ""

# Step 1: Setup MSVC compiler environment
Write-Host "[1/4] Setting up MSVC compiler environment..." -ForegroundColor Yellow
$vcvarsall = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
if (-not (Test-Path $vcvarsall)) {
    throw "vcvarsall.bat not found at $vcvarsall"
}

# Import MSVC environment variables into PowerShell
$envBefore = @{}
Get-ChildItem Env: | ForEach-Object { $envBefore[$_.Name] = $_.Value }

cmd /c "`"$vcvarsall`" x64 && set" | ForEach-Object {
    if ($_ -match "^([^=]+)=(.*)$") {
        $name = $matches[1]
        $value = $matches[2]
        if ($envBefore[$name] -ne $value) {
            [Environment]::SetEnvironmentVariable($name, $value, "Process")
        }
    }
}
Write-Host "MSVC environment loaded" -ForegroundColor Green

# Set DISTUTILS_USE_SDK to avoid duplicate VC env activation
[Environment]::SetEnvironmentVariable("DISTUTILS_USE_SDK", "1", "Process")

# Step 2: Sync dependencies
Write-Host "[2/4] Running uv sync..." -ForegroundColor Yellow
uv sync
if ($LASTEXITCODE -ne 0) { throw "uv sync failed" }



# Step 4: Install gsplat dependencies first (needed for build)
Write-Host "[3/4] Installing gsplat from GitHub and compiling CUDA extensions..." -ForegroundColor Yellow
uv pip install jaxtyping ninja rich packaging numpy
if ($LASTEXITCODE -ne 0) { throw "gsplat dependencies installation failed" }

# Install gsplat from GitHub with no build isolation (so it can use torch)
uv pip install git+https://github.com/nerfstudio-project/gsplat.git --no-build-isolation --no-cache-dir
if ($LASTEXITCODE -ne 0) { throw "gsplat installation failed" }

# Trigger JIT compilation
Write-Host "Compiling gsplat CUDA extensions (this may take several minutes)..." -ForegroundColor Yellow
uv run python -c "from gsplat.cuda._backend import _C; print('gsplat CUDA extensions compiled successfully')"
if ($LASTEXITCODE -ne 0) { throw "gsplat CUDA compilation failed" }

# Step 5: Verify installation
Write-Host "[4/4] Verifying installation..." -ForegroundColor Yellow
uv run python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); from gsplat.cuda._backend import _C; print('gsplat CUDA: OK')"
if ($LASTEXITCODE -ne 0) { throw "Verification failed" }

Write-Host ""
Write-Host "=== Installation Complete ===" -ForegroundColor Green
Write-Host "Run the viewer with: uv run src/viewer/main.py --config <path>" -ForegroundColor Cyan
