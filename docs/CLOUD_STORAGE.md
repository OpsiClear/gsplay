# Cloud Storage Support

## Overview

Load and save PLY files from AWS S3, Google Cloud Storage (GCS), and Azure Blob Storage using the same API as local files.

Built on [fsspec](https://filesystem-spec.readthedocs.io/) with automatic protocol detection from path strings (`s3://`, `gs://`, `az://`).

## Installation

### Base (Local Only)

```bash
uv pip install -e .
```

### Cloud Providers

```bash
# AWS S3
uv pip install -e ".[s3]"

# Google Cloud Storage
uv pip install -e ".[gcs]"

# Azure Blob Storage
uv pip install -e ".[azure]"

# All providers
uv pip install -e ".[cloud-all]"
```

## Usage

### Basic API

`UniversalPath` works identically for local and cloud paths:

```python
from src.infrastructure.io.path_io import UniversalPath

# Local
path = UniversalPath("./export_with_edits")

# S3
path = UniversalPath("s3://bucket/gaussians/scene1")

# GCS
path = UniversalPath("gs://bucket/gaussians/scene1")

# Azure
path = UniversalPath("az://container/gaussians/scene1")

# Operations work the same
ply_files = path.glob("*.ply")
for ply in ply_files:
    data = ply.read_bytes()
```

### Loading PLY Files

```python
from src.infrastructure.ply import load_ply

# Auto-detects protocol from path
means, scales, quats, opacities, sh0, shN = load_ply(
    "s3://bucket/scene/frame_00000.ply",
    device="cuda"
)
```

### Saving PLY Files

```python
from src.infrastructure.ply import write_ply

# Standard PLY
write_ply(
    "s3://bucket/output/result.ply",
    means, scales, quats, opacities, sh0, shN,
    format="ply"
)

# Compressed PLY (14.5x smaller)
write_ply(
    "gs://bucket/output/result.ply",
    means, scales, quats, opacities, sh0, shN,
    format="compressed"
)
```

### Viewer Integration

```bash
# Load from cloud
viewer --config s3://bucket/configs/scene1.json
viewer --config gs://bucket/scene1

# Local still works
viewer --config ./export_with_edits
```

## Supported Providers

| Provider | Protocol | Library | Authentication |
|----------|----------|---------|----------------|
| AWS S3 | `s3://` | [s3fs](https://s3fs.readthedocs.io/) | AWS credentials, IAM roles |
| Google Cloud | `gs://` | [gcsfs](https://gcsfs.readthedocs.io/) | GCP credentials, service accounts |
| Azure Blob | `az://`, `abfs://` | [adlfs](https://github.com/fsspec/adlfs) | Azure credentials, CLI |
| HTTP/HTTPS | `http://`, `https://` | fsspec | None (public URLs, read-only) |

## Authentication

### AWS S3

```bash
# Configure AWS CLI
aws configure

# Or set environment variables
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
```

### Google Cloud Storage

```bash
# Service account key
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json

# Or use gcloud CLI
gcloud auth application-default login
```

### Azure Blob Storage

```bash
# Azure CLI
az login

# Or connection string
export AZURE_STORAGE_CONNECTION_STRING=your_connection_string
```

## Performance

### In-Memory Loading

Cloud files are loaded entirely into memory (single network request per file):

- **Typical PLY**: 128KB-11MB → loads in 100-500ms
- **With cache**: ~10-15ms (subsequent loads)
- **Prefetching**: Background loading for smooth playback

### Optimization Tips

```python
model = OptimizedPlyModel(
    ply_files=cloud_files,
    enable_concurrent_prefetch=True       # Background loading
)
```

Use compressed format to reduce network transfer:

```python
write_ply(..., format="compressed")  # 16 bytes/splat vs 232 bytes/splat
```

## Error Handling

### Missing Dependencies

```
ImportError: S3 support requires s3fs. Install with: pip install s3fs
```

**Solution**: `uv pip install -e ".[s3]"` or `uv pip install s3fs>=2024.1.0`

### Authentication Errors

```
PermissionError: Access denied to s3://bucket/data
```

**Solution**: Configure credentials (see Authentication section above)

### File Not Found

```
FileNotFoundError: Path does not exist: s3://bucket/missing/file.ply
```

**Solution**: Verify bucket name, path, and permissions

### Network Errors

```
ConnectionError: Failed to connect to s3://bucket
```

**Solution**: Check internet connectivity, bucket region, firewall/proxy settings

## API Reference

### UniversalPath Operations

| Method | Description |
|--------|-------------|
| `read_bytes()` | Read entire file as bytes |
| `write_bytes(data)` | Write bytes to file |
| `open(mode)` | Open file handle |
| `exists()` | Check if path exists |
| `is_dir()`, `is_file()` | Check path type |
| `glob(pattern)` | Find files matching pattern |
| `parent`, `name`, `/` | Path manipulation |

### Type Signatures

All path functions accept flexible types:

```python
str | Path | UniversalPath
```

This allows seamless migration:

```python
# Old code (still works)
load_ply("./local/file.ply")

# New code (cloud support)
load_ply("s3://bucket/file.ply")
```

## References

- [fsspec](https://filesystem-spec.readthedocs.io/) - Unified filesystem interface
- [s3fs](https://s3fs.readthedocs.io/) - AWS S3 support
- [gcsfs](https://gcsfs.readthedocs.io/) - Google Cloud Storage support
- [adlfs](https://github.com/fsspec/adlfs) - Azure Blob Storage support
