# Documentation Update Summary

## Changes Made - 2025-11-25

### Project Renaming

Updated all references from "Universal 4D Viewer" to "gsplay" across the codebase:

### Files Updated

#### Core Documentation

1. **README.md** (root)
   - Updated project title to "gsplay"
   - Updated Python requirements: 3.12+ (was 3.10+)
   - Updated PyTorch requirements: 2.9+ (was 2.0+)
   - Updated repository URL: `https://github.com/opsiclear/gsplay`
   - Simplified installation instructions with new install scripts
   - Updated CLI command examples to use `uv run gsplay`
   - Updated project structure to reflect current organization
   - Updated citation information

2. **docs/README.md**
   - Updated references from "Universal 4D Viewer" to "gsplay"
   - Updated GitHub issues URL to new repository

### Installation Scripts

3. **install.ps1** (Windows)
   - Removed redundant PyTorch installation step (now handled by `uv sync`)
   - Updated step numbers from `[1/5]` to `[1/4]`
   - Updated title to "gsplay Installation (Windows)"
   - Script now relies on `pyproject.toml` for PyTorch with CUDA 12.8

4. **install.sh** (Linux)
   - Updated title to "gsplay Installation (Linux)"
   - Consistent with Windows script approach
   - Updated final run command to `uv run gsplay`

### Configuration

5. **pyproject.toml**
   - Project name: `gsplay`
   - Python requirement: `>=3.12,<3.13`
   - PyTorch requirement: `>=2.9.0`
   - Removed `gsplat` from dependencies (installed via scripts)
   - Updated `uv.sources` configuration
   - Maintained pytorch-cu128 index for CUDA support

6. **.gitignore**
   - Added `data/` and `.claude/` to ignore list

### Git Configuration

7. **Git Repository**
   - Reset upstream to `https://github.com/opsiclear/gsplay.git`
   - Cleared git history with fresh "Initial commit"
   - Untracked `.claude/` directory

## Key Technical Changes

### Installation Process

- **Before**: Manual steps for PyTorch, gsplat, and dependencies
- **After**: Streamlined with `install.ps1` / `install.sh` scripts
  1. `uv sync` installs all dependencies (including PyTorch 2.9.1+cu128)
  2. Script installs `gsplat` from GitHub with `--no-build-isolation`
  3. JIT compilation and verification

### Dependency Management

- **PyTorch**: Now sourced from `pytorch-cu128` index via `pyproject.toml`
- **gsplat**: Removed from `pyproject.toml`, installed via scripts for JIT compilation
- **gsmod**: Changed from local path to PyPI package

### CLI Usage

- **Before**: `viewer --config <path>` or `uv run src/viewer/main.py`
- **After**: `uv run gsplay --config <path>` or `uv run python -m gsplay.core.main`

## Files Still Requiring Review

The following documentation files may contain outdated references and should be reviewed:

1. `docs/CONTRIBUTING.md`
2. `docs/CLOUD_STORAGE.md`
3. `docs/API_COMPARISON.md`
4. `docs/PLY_ARCHITECTURE.md`
5. `docs/ROTATION_FEATURE.md`
6. `docs/SCALE_FILTERING.md`
7. `docs/SUPERSPLAT_CAMERA.md`
8. `docs/code_simplifications.md`
9. `docs/architecture/*.md` files
10. `launcher/README.md`
11. `tests/README.md`
12. `AGENTS.md`
13. `CLAUDE.md`

## Verification Checklist

- [x] Main README.md updated
- [x] Installation scripts updated and synchronized
- [x] pyproject.toml reflects current configuration
- [x] Git upstream reset to opsiclear/gsplay
- [x] docs/README.md updated
- [ ] All other documentation files reviewed
- [ ] CONTRIBUTING.md updated
- [ ] CLAUDE.md updated

## Next Steps

1. Review remaining documentation files for outdated references
2. Update CONTRIBUTING.md with current development workflow
3. Update CLAUDE.md with current project structure
4. Test installation scripts on clean environments
5. Verify all CLI commands work as documented
