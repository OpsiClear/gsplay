"""
Modular PLY I/O for Gaussian Splatting using gsply v0.2.5.

This package provides efficient loading and writing of PLY files for
3D Gaussian splatting, with support for:
- Standard PLY format (using gsply v0.2.5 GPU loading/writing interface)
- Compressed PLY format (PlayCanvas super-compressed, using gsply v0.2.5)
- Local filesystem and cloud storage (S3, GCS, Azure)

Public API:
-----------
**Loading**:
- load_ply()              - Unified loader (auto-detects format via gsply)

**Writing**:
- write_ply()             - Unified writer (format="ply" or "compressed")
- write_ply_bytes()       - Write to bytes (in-memory)

**Utilities**:
- sh2rgb(), rgb2sh()      - Spherical harmonics conversion

Architecture:
-------------
```
loader.py          - Unified loader using gsply (auto-detects format)
writer.py          - Unified writer using gsply (standard/compressed)
utils.py           - SH conversion utilities
```

Example Usage:
--------------
```python
from src.infrastructure.processing.ply import load_ply_as_gstensor, write_ply

# Load from local or cloud as GSTensor (auto-detects format)
gaussians = load_ply_as_gstensor(
    "s3://my-bucket/scene/frame_00000.ply",
    device="cuda"
)

# Write standard PLY (handles log/logit conversion automatically)
write_ply(
    "./output/result.ply",
    gaussians,
    format="ply"
)

# Write compressed PLY
write_ply(
    "gs://my-bucket/output/result.ply",
    gaussians,
    format="compressed"
)
```
"""

# Public API - Loaders
from src.infrastructure.processing.ply.loader import load_ply, load_ply_as_gstensor, load_ply_as_gsdata
from src.infrastructure.processing.ply.format_loader import (
    FormatAwarePlyLoader,
    PlyFrameEncoding,
)

# Public API - Writers
from src.infrastructure.processing.ply.writer import (
    write_ply,
    write_ply_bytes,
)

# Public API - Utilities
from src.infrastructure.processing.ply.utils import (
    sh2rgb,
    rgb2sh,
    sh2rgb_np,
    rgb2sh_np,
    SH_C0,
)

__all__ = [
    # Loaders
    "load_ply",
    "load_ply_as_gstensor",
    "load_ply_as_gsdata",
    "FormatAwarePlyLoader",
    "PlyFrameEncoding",
    # Writers
    "write_ply",
    "write_ply_bytes",
    # Utilities
    "sh2rgb",
    "rgb2sh",
    "sh2rgb_np",
    "rgb2sh_np",
    "SH_C0",
]
