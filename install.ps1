# GSPlay Installation Script for Windows
# Auto-detects CUDA version and installs matching PyTorch

param(
    [switch]$Global,
    [switch]$Local,
    [switch]$Help
)

$ErrorActionPreference = "Stop"

# =============================================================================
# Help
# =============================================================================

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

# =============================================================================
# Configuration
# =============================================================================

$InstallMode = if ($Global) { "global" } else { "local" }

Write-Host "=== GSPlay Installation (Windows) ===" -ForegroundColor Cyan
Write-Host ""

# =============================================================================
# Prerequisites Check
# =============================================================================

Write-Host "[1/6] Checking prerequisites..." -ForegroundColor Yellow

# Check for uv
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Error: 'uv' is not installed." -ForegroundColor Red
    Write-Host "Install with: powershell -ExecutionPolicy ByPass -c `"irm https://astral.sh/uv/install.ps1 | iex`""
    exit 1
}
Write-Host "  √ uv found: $(uv --version)"

# Detect CUDA version from nvidia-smi
function Get-CudaVersion {
    try {
        $nvidiaSmi = nvidia-smi 2>$null
        if ($LASTEXITCODE -eq 0 -and $nvidiaSmi) {
            $match = [regex]::Match($nvidiaSmi, "CUDA Version:\s*(\d+\.\d+)")
            if ($match.Success) {
                return $match.Groups[1].Value
            }
        }
    } catch {}

    # Try nvcc
    try {
        $nvcc = nvcc --version 2>$null
        if ($LASTEXITCODE -eq 0 -and $nvcc) {
            $match = [regex]::Match($nvcc, "release (\d+\.\d+)")
            if ($match.Success) {
                return $match.Groups[1].Value
            }
        }
    } catch {}

    return $null
}

$CudaVersion = Get-CudaVersion
if (-not $CudaVersion) {
    Write-Host "Error: CUDA not detected (nvidia-smi not found)." -ForegroundColor Red
    Write-Host "GSPlay requires an NVIDIA GPU with CUDA support."
    Write-Host ""
    Write-Host "To fix:"
    Write-Host "  1. Install NVIDIA drivers: https://www.nvidia.com/drivers"
    Write-Host "  2. Verify with: nvidia-smi"
    Write-Host "  3. Run this installer again"
    exit 1
}
Write-Host "  √ CUDA detected: $CudaVersion"

# Map CUDA version to PyTorch index
function Get-TorchIndex {
    param([string]$Cuda)

    $parts = $Cuda.Split(".")
    $major = [int]$parts[0]
    $minor = [int]$parts[1]

    if ($major -eq 12) {
        if ($minor -ge 8) { return "cu128" }
        elseif ($minor -ge 6) { return "cu126" }
        elseif ($minor -ge 4) { return "cu124" }
        else { return "cu121" }
    } elseif ($major -eq 11) {
        return "cu118"
    } else {
        return "cu121"  # Default fallback
    }
}

$CudaIndex = Get-TorchIndex -Cuda $CudaVersion
$TorchIndexUrl = "https://download.pytorch.org/whl/$CudaIndex"
Write-Host "  √ PyTorch index: $TorchIndexUrl"

# =============================================================================
# Setup MSVC Compiler
# =============================================================================

Write-Host ""
Write-Host "[2/6] Setting up MSVC compiler environment..." -ForegroundColor Yellow

# Find vcvarsall.bat
$VcVarsAll = $null
$VsPaths = @(
    "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat",
    "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat",
    "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat",
    "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall.bat",
    "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat",
    "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat"
)

foreach ($path in $VsPaths) {
    if (Test-Path $path) {
        $VcVarsAll = $path
        break
    }
}

if (-not $VcVarsAll) {
    Write-Host "Error: Visual Studio Build Tools not found." -ForegroundColor Red
    Write-Host "Install from: https://visualstudio.microsoft.com/visual-cpp-build-tools/"
    Write-Host "Select 'Desktop development with C++'"
    exit 1
}

# Import MSVC environment
$envBefore = @{}
Get-ChildItem Env: | ForEach-Object { $envBefore[$_.Name] = $_.Value }

cmd /c "`"$VcVarsAll`" x64 && set" | ForEach-Object {
    if ($_ -match "^([^=]+)=(.*)$") {
        $name = $matches[1]
        $value = $matches[2]
        if ($envBefore[$name] -ne $value) {
            [Environment]::SetEnvironmentVariable($name, $value, "Process")
        }
    }
}
Write-Host "  √ MSVC environment loaded"

# Set DISTUTILS_USE_SDK
[Environment]::SetEnvironmentVariable("DISTUTILS_USE_SDK", "1", "Process")

# =============================================================================
# Setup Installation Directory
# =============================================================================

Write-Host ""
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

if ($InstallMode -eq "global") {
    Write-Host "[3/6] Setting up global installation (~\.gsplay\)..." -ForegroundColor Yellow

    $InstallDir = "$env:USERPROFILE\.gsplay"
    if (-not (Test-Path $InstallDir)) {
        New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
    }

    # Create virtual environment
    if (-not (Test-Path "$InstallDir\venv")) {
        uv venv "$InstallDir\venv" --python 3.12
    }

    # Activate for this script
    & "$InstallDir\venv\Scripts\Activate.ps1"

    # Set UV_PROJECT_ENVIRONMENT
    $env:UV_PROJECT_ENVIRONMENT = "$InstallDir\venv"
} else {
    Write-Host "[3/6] Setting up local installation (.venv\)..." -ForegroundColor Yellow
    $InstallDir = $ScriptDir
}

# =============================================================================
# Install Dependencies
# =============================================================================

Write-Host ""
Write-Host "[4/6] Installing PyTorch with CUDA $CudaVersion support..." -ForegroundColor Yellow
uv pip install torch torchvision --index-url $TorchIndexUrl
if ($LASTEXITCODE -ne 0) { throw "PyTorch installation failed" }

Write-Host ""
Write-Host "[5/6] Installing gsplay and dependencies..." -ForegroundColor Yellow
if ($InstallMode -eq "global") {
    if (Test-Path "$ScriptDir\pyproject.toml") {
        uv pip install -e $ScriptDir
    } else {
        uv pip install gsplay
    }
} else {
    uv sync
}
if ($LASTEXITCODE -ne 0) { throw "gsplay installation failed" }

Write-Host ""
Write-Host "[6/6] Installing gsplat (compiling CUDA extensions)..." -ForegroundColor Yellow
Write-Host "  This may take a few minutes..."
uv pip install jaxtyping ninja rich packaging numpy
uv pip install gsplat --no-build-isolation --no-cache-dir
if ($LASTEXITCODE -ne 0) { throw "gsplat installation failed" }

# Trigger JIT compilation
Write-Host "  Triggering JIT compilation..."
if ($InstallMode -eq "global") {
    python -c "from gsplat.cuda._backend import _C; print('  √ gsplat CUDA extensions compiled')"
} else {
    uv run python -c "from gsplat.cuda._backend import _C; print('  √ gsplat CUDA extensions compiled')"
}
if ($LASTEXITCODE -ne 0) { throw "gsplat CUDA compilation failed" }

# =============================================================================
# Build Launcher Frontend (Optional)
# =============================================================================

Write-Host ""
Write-Host "Building launcher frontend..." -ForegroundColor Yellow
$DenoBin = $null

if (Get-Command deno -ErrorAction SilentlyContinue) {
    $DenoBin = "deno"
} elseif (Test-Path "$env:USERPROFILE\.deno\bin\deno.exe") {
    $DenoBin = "$env:USERPROFILE\.deno\bin\deno.exe"
}

if ($DenoBin -and (Test-Path "$ScriptDir\launcher\frontend")) {
    Push-Location "$ScriptDir\launcher\frontend"
    try {
        & $DenoBin task build 2>$null
        if ($LASTEXITCODE -eq 0 -and (Test-Path "dist")) {
            Pop-Location
            $StaticDir = "$ScriptDir\launcher\gsplay_launcher\static"
            if (Test-Path $StaticDir) { Remove-Item -Recurse -Force $StaticDir }
            New-Item -ItemType Directory -Path $StaticDir -Force | Out-Null
            Copy-Item -Recurse "$ScriptDir\launcher\frontend\dist\*" $StaticDir
            Write-Host "  √ Frontend built successfully"
        } else {
            Pop-Location
            Write-Host "  ! Frontend build failed, using fallback dashboard" -ForegroundColor Yellow
        }
    } catch {
        Pop-Location
        Write-Host "  ! Frontend build failed, using fallback dashboard" -ForegroundColor Yellow
    }
} else {
    Write-Host "  ! deno not found, skipping frontend build" -ForegroundColor Yellow
    Write-Host "    Launcher will use fallback HTML dashboard"
}

# =============================================================================
# PATH Integration (Global Install Only)
# =============================================================================

if ($InstallMode -eq "global") {
    Write-Host ""
    Write-Host "Setting up PATH integration..." -ForegroundColor Yellow

    # Create wrapper scripts
    $BinDir = "$env:USERPROFILE\.local\bin"
    if (-not (Test-Path $BinDir)) {
        New-Item -ItemType Directory -Path $BinDir -Force | Out-Null
    }

    # gsplay.cmd
    @"
@echo off
call "$env:USERPROFILE\.gsplay\venv\Scripts\activate.bat"
python -m src.gsplay.core.main %*
"@ | Out-File -FilePath "$BinDir\gsplay.cmd" -Encoding ASCII

    # gsplay-launcher.cmd
    @"
@echo off
call "$env:USERPROFILE\.gsplay\venv\Scripts\activate.bat"
python -m gsplay_launcher %*
"@ | Out-File -FilePath "$BinDir\gsplay-launcher.cmd" -Encoding ASCII

    # Check if bin dir is in PATH
    $UserPath = [Environment]::GetEnvironmentVariable("PATH", "User")
    if ($UserPath -notlike "*$BinDir*") {
        Write-Host ""
        Write-Host "To add gsplay to PATH, run:" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "  `$env:PATH = `"$BinDir;`$env:PATH`""
        Write-Host ""
        Write-Host "Or add permanently via System Properties > Environment Variables"
    } else {
        Write-Host "  √ $BinDir is already in PATH"
    }
}

# =============================================================================
# Verification
# =============================================================================

Write-Host ""
Write-Host "Verifying installation..." -ForegroundColor Yellow
$verifyScript = @"
import torch
print(f'  √ PyTorch: {torch.__version__}')
print(f'  √ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  √ GPU: {torch.cuda.get_device_name(0)}')
from gsplat.cuda._backend import _C
print('  √ gsplat: OK')
"@

if ($InstallMode -eq "global") {
    python -c $verifyScript
} else {
    uv run python -c $verifyScript
}
if ($LASTEXITCODE -ne 0) { throw "Verification failed" }

# =============================================================================
# Done
# =============================================================================

Write-Host ""
Write-Host "=== Installation Complete ===" -ForegroundColor Green
Write-Host ""
if ($InstallMode -eq "global") {
    Write-Host "Run the viewer:" -ForegroundColor Cyan
    Write-Host "  gsplay <path-to-ply-folder>"
    Write-Host ""
    Write-Host "Run the launcher:" -ForegroundColor Cyan
    Write-Host "  gsplay-launcher --browse-path <path>"
} else {
    Write-Host "Run the viewer:" -ForegroundColor Cyan
    Write-Host "  uv run gsplay <path-to-ply-folder>"
    Write-Host ""
    Write-Host "Run the launcher:" -ForegroundColor Cyan
    Write-Host "  uv run -m gsplay_launcher --browse-path <path>"
}
