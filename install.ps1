# GSPlay Installation Script for Windows
param(
    [switch]$Global,
    [switch]$Local,
    [switch]$Help
)

$ErrorActionPreference = "Stop"

if ($Help) {
    Write-Host "GSPlay Installer"
    Write-Host ""
    Write-Host "Usage: .\install.ps1 [OPTIONS]"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -Local    Install to project .venv/ (default)"
    Write-Host "  -Global   Install to ~\.gsplay\ with PATH integration"
    Write-Host "  -Help     Show this help message"
    exit 0
}

$InstallMode = if ($Global) { "global" } else { "local" }
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "=== GSPlay Installation (Windows) ===" -ForegroundColor Cyan
Write-Host ""

# =============================================================================
# [1/5] Prerequisites
# =============================================================================

Write-Host "[1/5] Checking prerequisites..." -ForegroundColor Yellow

# Check uv
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Error: 'uv' not installed." -ForegroundColor Red
    Write-Host "Install: powershell -c `"irm https://astral.sh/uv/install.ps1 | iex`""
    exit 1
}
Write-Host "  [OK] uv: $(uv --version)"

# Detect CUDA
$nvidiaSmi = nvidia-smi 2>$null
if (-not $nvidiaSmi) {
    Write-Host "Error: CUDA not detected." -ForegroundColor Red
    exit 1
}
$CudaVersion = [regex]::Match($nvidiaSmi, "CUDA Version:\s*(\d+\.\d+)").Groups[1].Value
Write-Host "  [OK] CUDA: $CudaVersion"

# Map CUDA to PyTorch index
$major, $minor = $CudaVersion.Split(".") | ForEach-Object { [int]$_ }
$CudaIndex = if ($major -eq 12) {
    if ($minor -ge 8) { "cu128" } elseif ($minor -ge 6) { "cu126" } elseif ($minor -ge 4) { "cu124" } else { "cu121" }
} elseif ($major -eq 11) { "cu118" } else { "cu121" }
Write-Host "  [OK] PyTorch index: $CudaIndex"

# =============================================================================
# [2/5] MSVC Setup
# =============================================================================

Write-Host ""
Write-Host "[2/5] Setting up MSVC..." -ForegroundColor Yellow

# Find vcvarsall.bat
$VcVarsAll = @(
    "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat",
    "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat",
    "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat"
) | Where-Object { Test-Path $_ } | Select-Object -First 1

if (-not $VcVarsAll) {
    Write-Host "Error: Visual Studio Build Tools not found." -ForegroundColor Red
    Write-Host "Install: https://visualstudio.microsoft.com/visual-cpp-build-tools/"
    exit 1
}

# Import MSVC environment
cmd /c "`"$VcVarsAll`" x64 && set" | ForEach-Object {
    if ($_ -match "^([^=]+)=(.*)$") {
        [Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
    }
}

# Fix PATH: Remove Git's link, ensure cl.exe is findable
$env:PATH = ($env:PATH -split ';' | Where-Object { $_ -notlike "*Git*usr*bin*" }) -join ';'
$env:DISTUTILS_USE_SDK = "1"

# Verify cl.exe is in PATH and add its directory explicitly
$clExe = Get-Command cl.exe -ErrorAction SilentlyContinue
if (-not $clExe) {
    Write-Host "Error: cl.exe not found after MSVC setup" -ForegroundColor Red
    exit 1
}
# Ensure MSVC bin is at front of PATH for child processes
$msvcBin = Split-Path $clExe.Source
$env:PATH = "$msvcBin;$env:PATH"
Write-Host "  [OK] MSVC: $($clExe.Source)"

# =============================================================================
# [3/5] Install Dependencies
# =============================================================================

Write-Host ""
Write-Host "[3/5] Installing dependencies..." -ForegroundColor Yellow

if ($InstallMode -eq "global") {
    $InstallDir = "$env:USERPROFILE\.gsplay"
    New-Item -ItemType Directory -Path $InstallDir -Force -ErrorAction SilentlyContinue | Out-Null
    if (-not (Test-Path "$InstallDir\venv")) { uv venv "$InstallDir\venv" --python 3.12 }
    & "$InstallDir\venv\Scripts\Activate.ps1"
    $env:UV_PROJECT_ENVIRONMENT = "$InstallDir\venv"
}

# PyTorch
uv pip install torch torchvision --index-url "https://download.pytorch.org/whl/$CudaIndex"
if ($LASTEXITCODE -ne 0) { throw "PyTorch installation failed" }

# gsplay
if ($InstallMode -eq "global") {
    uv pip install -e $ScriptDir
} else {
    uv sync
}

# =============================================================================
# [4/5] gsplat (CUDA compilation)
# =============================================================================

Write-Host ""
Write-Host "[4/5] Installing gsplat..." -ForegroundColor Yellow

# Clear stale cache
Remove-Item -Recurse -Force "$env:LOCALAPPDATA\torch_extensions\*\gsplat*" -ErrorAction SilentlyContinue

# Set compilation flags
$env:TORCH_CUDA_ARCH_LIST = "8.0;8.6;8.9;9.0+PTX"
$env:MAX_JOBS = "4"

# Install gsplat
uv pip install gsplat --no-build-isolation --no-cache-dir
if ($LASTEXITCODE -ne 0) { throw "gsplat installation failed" }

# Patch for MSVC (remove GCC-only -Wno-attributes flag)
$backendPy = ".venv\Lib\site-packages\gsplat\cuda\_backend.py"
if (Test-Path $backendPy) {
    (Get-Content $backendPy -Raw) -replace `
        'extra_cflags = \[opt_level, "-Wno-attributes"\]', `
        'extra_cflags = [opt_level] if os.name == "nt" else [opt_level, "-Wno-attributes"]' |
    Set-Content $backendPy -NoNewline
    Write-Host "  [OK] Patched for MSVC"
}

# Clear cache after patch and trigger JIT
Remove-Item -Recurse -Force "$env:LOCALAPPDATA\torch_extensions\*\gsplat*" -ErrorAction SilentlyContinue

# =============================================================================
# [5/5] Verify & Frontend
# =============================================================================

Write-Host ""
Write-Host "[5/5] Verifying..." -ForegroundColor Yellow

$runCmd = if ($InstallMode -eq "global") { "python" } else { "uv run python" }
Invoke-Expression "$runCmd -c `"import torch; from gsplat.cuda._backend import _C; print('  [OK] PyTorch:', torch.__version__); print('  [OK] CUDA:', torch.cuda.is_available()); print('  [OK] gsplat: compiled')`""
if ($LASTEXITCODE -ne 0) { throw "Verification failed" }

# Optional: Build frontend
$DenoBin = Get-Command deno -ErrorAction SilentlyContinue
if ($DenoBin -and (Test-Path "$ScriptDir\launcher\frontend")) {
    Push-Location "$ScriptDir\launcher\frontend"
    $ErrorActionPreference = "Continue"
    & deno task build *>&1 | Out-Null
    $ErrorActionPreference = "Stop"
    Pop-Location
    if (Test-Path "$ScriptDir\launcher\frontend\dist") {
        $StaticDir = "$ScriptDir\launcher\gsplay_launcher\static"
        Remove-Item -Recurse -Force $StaticDir -ErrorAction SilentlyContinue
        Copy-Item -Recurse "$ScriptDir\launcher\frontend\dist" $StaticDir
        Write-Host "  [OK] Frontend built"
    }
}

# =============================================================================
# Done
# =============================================================================

Write-Host ""
Write-Host "=== Installation Complete ===" -ForegroundColor Green
Write-Host ""
if ($InstallMode -eq "global") {
    Write-Host "Run: gsplay <path-to-ply-folder>" -ForegroundColor Cyan
} else {
    Write-Host "Run: uv run gsplay <path-to-ply-folder>" -ForegroundColor Cyan
}
