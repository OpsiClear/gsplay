# PLY I/O Architecture

## Overview

PLY I/O uses the `gsply` library for all file operations. The codebase adds utilities that gsply doesn't provide (SH conversion, spatial sorting).

## Architecture

```
src/infrastructure/processing/ply/
├── __init__.py            # Public API exports (load_ply, write_ply, utilities)
├── loader.py              # load_ply() using gsply
├── writer.py              # write_ply() using gsply
├── activation_service.py  # Smart activation (CPU/GPU detection)
└── utils.py               # SH conversion, Morton codes
```

## API

### Loading

```python
from src.infrastructure.ply import load_ply

# Auto-detects standard or compressed format
means, scales, quats, opacities, sh0, shN = load_ply(
    "path/to/file.ply",  # Local or cloud (s3://, gs://, az://)
    device="cuda"
)
```

### Writing

```python
from src.infrastructure.ply import write_ply

# Standard PLY
write_ply("output.ply", means, scales, quats, opacities, sh0, shN, format="ply")

# Compressed PLY (PlayCanvas format, 14.5x smaller)
write_ply("output.ply", means, scales, quats, opacities, sh0, shN, format="compressed")

# Cloud storage
write_ply("s3://bucket/output.ply", means, scales, quats, opacities, sh0, shN)
```

### Utilities

```python
from src.infrastructure.ply import sh2rgb, rgb2sh, sort_gaussians_by_morton

# Spherical harmonics conversion
rgb_colors = sh2rgb(sh0)  # Torch tensors
sh0 = rgb2sh(rgb_colors)

# Spatial sorting for cache coherence
means, scales, quats, opacities, sh0, shN = sort_gaussians_by_morton(
    means, scales, quats, opacities, sh0, shN
)
```

## What gsply Provides

- `plyread()`, `plywrite()` - File I/O
- `encode()`, `decode()` - Bytes I/O
- `detect_format()` - Format detection
- Auto-detection of standard vs compressed PLY
- JIT compilation for performance
- Supports SH degrees 0-3

## What We Add (utils.py)

gsply only does I/O. We provide:

1. **SH Conversion**: `sh2rgb()`, `rgb2sh()` (torch + numpy versions)
2. **Spatial Sorting**: `sort_gaussians_by_morton()` for cache coherence
3. **Pack/Unpack**: Quantization utilities (if needed)

## Cloud Storage

All functions accept `str | Path | UniversalPath`:

```python
# Works with any path type
load_ply("./local/file.ply")
load_ply("s3://bucket/file.ply")
load_ply("gs://bucket/file.ply")
write_ply("az://container/output.ply", ...)
```

See [CLOUD_STORAGE.md](./CLOUD_STORAGE.md) for setup and authentication.

## Benefits

- **Minimal code**: ~600 lines of custom code eliminated
- **Industry standard**: Compatible with other tools
- **Reliable**: Well-tested library with community support
- **Fast**: JIT compilation via numba
- **Simple API**: Single function for all formats

## References

- [gsply](https://github.com/cgohlke/gsply) - PLY I/O library
- [PlayCanvas Compressed Format](https://github.com/playcanvas/splat-transform)
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
