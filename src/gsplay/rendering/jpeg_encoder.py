"""GPU JPEG encoding using nvImageCodec.

Architecture
------------
    ┌─────────────────────────────────────────────────────────────────┐
    │                      Public API (Module Level)                   │
    │     encode_and_cache(), get_cached_jpeg(), encode()             │
    └─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                    JpegEncodingService                          │
    │  - Owns encoder instance (lazy-initialized)                     │
    │  - Caches JPEG bytes in thread-local storage                    │
    │  - Thread-safe encoding                                         │
    └─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                       GPUJpegBackend                            │
    │  - nvImageCodec (~0.5ms for 1080p)                              │
    │  - Zero-copy from GPU tensor                                    │
    └─────────────────────────────────────────────────────────────────┘

Requirements
------------
    pip install nvidia-nvimgcodec-cu12[nvjpeg]

Usage
-----
Simple encoding::

    from src.gsplay.rendering.jpeg_encoder import encode
    jpeg_bytes = encode(image_tensor, quality=85)

With caching for streaming (encode once, broadcast to multiple clients)::

    from src.gsplay.rendering.jpeg_encoder import encode_and_cache, get_cached_jpeg

    # In renderer - encode immediately while tensor is valid:
    encode_and_cache(gpu_frame_uint8)
    result = gpu_frame_uint8.cpu().numpy()

    # In broadcast - retrieve cached JPEG bytes:
    jpeg_bytes = get_cached_jpeg()
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


# =============================================================================
# Types
# =============================================================================


class ChromaSubsampling(str, Enum):
    """Chroma subsampling options for JPEG encoding.

    Controls the trade-off between file size and color fidelity:
    - CSS_420: Standard JPEG, best compression (~1.5MB for 1080p)
    - CSS_422: Horizontal subsampling (~2MB for 1080p)
    - CSS_444: No subsampling, best quality (~3.2MB for 1080p)
    """

    CSS_420 = "420"
    CSS_422 = "422"
    CSS_444 = "444"

    @classmethod
    def from_string(cls, value: str) -> "ChromaSubsampling":
        """Convert string to ChromaSubsampling enum."""
        mapping = {
            "420": cls.CSS_420, "4:2:0": cls.CSS_420,
            "422": cls.CSS_422, "4:2:2": cls.CSS_422,
            "444": cls.CSS_444, "4:4:4": cls.CSS_444,
        }
        if value in mapping:
            return mapping[value]
        raise ValueError(f"Unknown chroma subsampling: {value}. Use 420, 422, or 444.")


ChromaSubsamplingType = Literal["420", "422", "444"] | ChromaSubsampling


@dataclass
class EncodingConfig:
    """Configuration for JPEG encoding."""

    quality: int = 85
    chroma_subsampling: ChromaSubsamplingType = "420"
    device: str = "cuda:0"


# =============================================================================
# Backends
# =============================================================================


class GPUJpegBackend:
    """GPU JPEG encoding using NVIDIA nvImageCodec.

    Provides ~47x speedup over CPU encoding with zero-copy support
    for CUDA tensors.

    Thread Safety
    -------------
    nvImageCodec's Encoder is NOT thread-safe. This backend uses thread-local
    encoders to allow safe concurrent encoding from multiple renderer threads.

    Requirements:
        pip install nvidia-nvimgcodec-cu12[nvjpeg]
    """

    def __init__(self, device_id: int = 0):
        from nvidia import nvimgcodec

        self._nvimgcodec = nvimgcodec
        self._device_id = device_id
        # Thread-local storage for per-thread encoders
        self._thread_local = threading.local()

        self._SUBSAMPLING_MAP = {
            "420": nvimgcodec.ChromaSubsampling.CSS_420,
            "422": nvimgcodec.ChromaSubsampling.CSS_422,
            "444": nvimgcodec.ChromaSubsampling.CSS_444,
            ChromaSubsampling.CSS_420: nvimgcodec.ChromaSubsampling.CSS_420,
            ChromaSubsampling.CSS_422: nvimgcodec.ChromaSubsampling.CSS_422,
            ChromaSubsampling.CSS_444: nvimgcodec.ChromaSubsampling.CSS_444,
        }

    def _get_encoder(self):
        """Get thread-local encoder (created on first access per thread)."""
        if not hasattr(self._thread_local, "encoder"):
            self._thread_local.encoder = self._nvimgcodec.Encoder(
                device_id=self._device_id
            )
        return self._thread_local.encoder

    @property
    def name(self) -> str:
        return f"nvImageCodec (GPU:{self._device_id})"

    def encode(
        self,
        image: np.ndarray | "torch.Tensor",
        quality: int,
        chroma_subsampling: ChromaSubsamplingType,
    ) -> bytes:
        import torch

        # Prepare parameters
        css = self._SUBSAMPLING_MAP.get(
            chroma_subsampling,
            self._nvimgcodec.ChromaSubsampling.CSS_420
        )
        params = self._nvimgcodec.EncodeParams(
            quality_type=self._nvimgcodec.QualityType.QUALITY,
            quality_value=float(quality),
            color_spec=self._nvimgcodec.ColorSpec.RGB,
            chroma_subsampling=css,
        )

        # Prepare image
        if isinstance(image, torch.Tensor):
            image = self._prepare_tensor(image)
        else:
            image = self._prepare_numpy(image)

        # Encode using thread-local encoder
        # NOTE: We synchronize the CUDA device before encoding because:
        # 1. The input tensor may have been created on a different stream
        # 2. nvImageCodec uses its own CUDA context and needs stable memory
        # 3. Without sync, we can get "Unable to get context for stream" errors
        encoder = self._get_encoder()
        
        try:
            # Ensure all CUDA operations on this tensor are complete
            # This is critical when the tensor was created on a non-default stream
            import torch
            if isinstance(image, torch.Tensor) and image.is_cuda:
                torch.cuda.synchronize(image.device)
            
            nv_image = self._nvimgcodec.as_image(image)
            result = encoder.encode(nv_image, codec="jpeg", params=params)
            return result
        except Exception as e:
            # If nvImageCodec fails, attempt to recover CUDA state
            try:
                import torch
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            except Exception:
                pass
            logger.error(f"nvImageCodec encode failed: {e}")
            raise

    def _prepare_tensor(self, tensor: "torch.Tensor") -> "torch.Tensor":
        """Prepare tensor for encoding (GPU path, zero-copy).

        If tensor is already uint8 (pre-converted in renderer), skip conversion.
        """
        import torch

        # Only convert if not already uint8
        if tensor.dtype != torch.uint8:
            if tensor.dtype in (torch.float32, torch.float16):
                if tensor.max() <= 1.0:
                    tensor = (tensor * 255).clamp(0, 255).to(torch.uint8)
                else:
                    tensor = tensor.clamp(0, 255).to(torch.uint8)
            else:
                tensor = tensor.to(torch.uint8)

        # Ensure HWC contiguous layout
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        return tensor.contiguous()

    def _prepare_numpy(self, image: np.ndarray) -> np.ndarray:
        """Prepare numpy array for encoding (will upload to GPU)."""
        if image.dtype in (np.float32, np.float64):
            if image.max() <= 1.0:
                return (image * 255).clip(0, 255).astype(np.uint8)
            return image.clip(0, 255).astype(np.uint8)
        if image.dtype != np.uint8:
            return image.astype(np.uint8)
        return image


# =============================================================================
# Encoder Facade
# =============================================================================


class JpegEncoder:
    """GPU JPEG encoder using nvImageCodec.

    Lazily initializes the GPU backend. Raises ImportError if nvImageCodec
    is not available.
    """

    def __init__(self, device: str = "cuda:0"):
        self._device = device
        self._backend: GPUJpegBackend | None = None
        self._init_lock = threading.Lock()

    def _ensure_backend(self) -> GPUJpegBackend:
        """Lazy-initialize GPU backend (thread-safe)."""
        if self._backend is not None:
            return self._backend

        with self._init_lock:
            if self._backend is not None:
                return self._backend

            device_id = int(self._device.split(":")[-1]) if ":" in self._device else 0
            self._backend = GPUJpegBackend(device_id=device_id)
            logger.info(f"JPEG encoder: {self._backend.name}")
            return self._backend

    @property
    def backend_name(self) -> str:
        """Name of the active backend."""
        return self._ensure_backend().name

    def encode(
        self,
        image: np.ndarray | "torch.Tensor",
        quality: int = 85,
        chroma_subsampling: ChromaSubsamplingType = "420",
    ) -> bytes:
        """Encode image to JPEG bytes."""
        return self._ensure_backend().encode(image, quality, chroma_subsampling)


# =============================================================================
# Encoding Service (Main Entry Point)
# =============================================================================


# Thread-local storage for cached JPEG bytes
# Each renderer thread gets its own isolated cache
_thread_local = threading.local()


def _get_cached_jpeg() -> bytes | None:
    """Get the cached JPEG bytes for the current thread."""
    return getattr(_thread_local, "jpeg_bytes", None)


def _set_cached_jpeg(jpeg_bytes: bytes | None) -> None:
    """Set the cached JPEG bytes for the current thread."""
    _thread_local.jpeg_bytes = jpeg_bytes


@dataclass
class JpegEncodingService:
    """High-level service for JPEG encoding with JPEG bytes caching.

    This is the main entry point for JPEG encoding in the application.
    It provides:
    - JPEG bytes caching for efficient multi-client streaming
    - Thread-safe operation with per-thread cache isolation

    Thread Safety
    -------------
    Each renderer thread has its own isolated JPEG cache using
    thread-local storage. This prevents race conditions when multiple
    clients are rendering simultaneously.

    Example
    -------
    >>> service = JpegEncodingService(device="cuda:0")
    >>>
    >>> # Simple encoding
    >>> jpeg = service.encode(image, quality=85)
    >>>
    >>> # Encode and cache for streaming
    >>> jpeg = service.encode_and_cache(gpu_tensor)  # Encode + cache
    >>> jpeg = service.get_cached_jpeg()  # Retrieve cached bytes
    """

    device: str = "cuda:0"
    default_quality: int = 85
    default_chroma_subsampling: ChromaSubsamplingType = "420"

    # Private state
    _encoder: JpegEncoder = field(init=False, repr=False)

    def __post_init__(self):
        self._encoder = JpegEncoder(self.device)

    @property
    def backend_name(self) -> str:
        """Name of the active encoding backend."""
        return self._encoder.backend_name

    def encode(
        self,
        image: np.ndarray | "torch.Tensor",
        quality: int | None = None,
        chroma_subsampling: ChromaSubsamplingType | None = None,
    ) -> bytes:
        """Encode image to JPEG bytes.

        Parameters
        ----------
        image : np.ndarray | torch.Tensor
            Image in HWC format. Supports float32 [0,1] or uint8 [0,255].
        quality : int, optional
            JPEG quality 1-100. Uses default_quality if not specified.
        chroma_subsampling : str, optional
            "420", "422", or "444". Uses default if not specified.
        """
        q = quality if quality is not None else self.default_quality
        css = chroma_subsampling if chroma_subsampling is not None else self.default_chroma_subsampling
        return self._encoder.encode(image, q, css)

    def encode_and_cache(
        self,
        image: "torch.Tensor",
        quality: int | None = None,
        chroma_subsampling: ChromaSubsamplingType | None = None,
    ) -> bytes:
        """Encode image to JPEG and cache the result (thread-local).

        Encodes immediately while we have exclusive access to the tensor,
        then caches the JPEG bytes for later retrieval via get_cached_jpeg().

        Parameters
        ----------
        image : torch.Tensor
            GPU tensor in HWC format, uint8 [0,255].
        quality : int, optional
            JPEG quality 1-100.
        chroma_subsampling : str, optional
            "420", "422", or "444".

        Returns
        -------
        bytes
            JPEG bytes (also cached for later retrieval).
        """
        q = quality if quality is not None else self.default_quality
        css = chroma_subsampling if chroma_subsampling is not None else self.default_chroma_subsampling
        jpeg_bytes = self._encoder.encode(image, q, css)
        _set_cached_jpeg(jpeg_bytes)
        return jpeg_bytes

    def get_cached_jpeg(self) -> bytes | None:
        """Get cached JPEG bytes for current thread."""
        return _get_cached_jpeg()

    def clear_cached_jpeg(self) -> None:
        """Clear the cached JPEG bytes for current thread."""
        _set_cached_jpeg(None)


# =============================================================================
# Module-Level API (Default Service)
# =============================================================================

# Default service instance (lazy-initialized)
_default_service: JpegEncodingService | None = None
_service_lock = threading.Lock()


def get_service(device: str = "cuda:0") -> JpegEncodingService:
    """Get the default encoding service.

    Creates a singleton service instance on first call.
    """
    global _default_service

    if _default_service is None:
        with _service_lock:
            if _default_service is None:
                _default_service = JpegEncodingService(device=device)

    return _default_service


def encode_and_cache(
    image: "torch.Tensor",
    quality: int = 85,
    chroma_subsampling: ChromaSubsamplingType = "420",
) -> bytes:
    """Encode GPU tensor to JPEG and cache the result (thread-local).

    Encodes immediately while we have exclusive access to the tensor,
    avoiding the need to clone for memory safety.

    Parameters
    ----------
    image : torch.Tensor
        GPU tensor in HWC format, uint8 [0,255].
    quality : int
        JPEG quality 1-100 (default: 85).
    chroma_subsampling : str
        "420" (default), "422", or "444".

    Returns
    -------
    bytes
        JPEG bytes (also cached for get_cached_jpeg retrieval).
    """
    return get_service().encode_and_cache(image, quality, chroma_subsampling)


def get_cached_jpeg() -> bytes | None:
    """Get cached JPEG bytes for the current thread.

    Returns the JPEG bytes cached by the most recent encode_and_cache() call
    in the current thread.
    """
    return get_service().get_cached_jpeg()


def encode(
    image: np.ndarray | "torch.Tensor",
    quality: int = 85,
    chroma_subsampling: ChromaSubsamplingType = "420",
) -> bytes:
    """Encode image to JPEG bytes (no caching).

    Parameters
    ----------
    image : np.ndarray | torch.Tensor
        Image in HWC format.
    quality : int
        JPEG quality 1-100 (default: 85).
    chroma_subsampling : str
        "420" (default), "422", or "444".
    """
    return get_service().encode(image, quality, chroma_subsampling)


