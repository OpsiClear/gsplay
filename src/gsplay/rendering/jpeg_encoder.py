"""GPU JPEG encoding using torchvision (nvJPEG) or nvImageCodec.

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
    │                   TorchvisionJpegBackend (default)              │
    │  - Uses torchvision.io.encode_jpeg with nvJPEG                  │
    │  - GPU encoding on CUDA tensors                                 │
    │  - Reliable on consumer GPUs                                    │
    └─────────────────────────────────────────────────────────────────┘

Requirements
------------
    torchvision with CUDA support (nvJPEG)

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
import torch
from torchvision.io import encode_jpeg

if TYPE_CHECKING:
    pass

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


class TorchvisionJpegBackend:
    """GPU JPEG encoding using torchvision.io.encode_jpeg with nvJPEG.

    This backend uses PyTorch's native CUDA JPEG encoding which properly
    handles nvJPEG internally. It's more reliable than nvImageCodec for
    GPU encoding on consumer GPUs.

    Key features:
    - Native PyTorch CUDA tensor support
    - Proper nvJPEG integration via torchvision
    - No external dependencies beyond torchvision
    - Handles CHW format conversion automatically

    Thread Safety: Thread-safe (torchvision handles this internally)

    Requirements:
        torchvision with CUDA support (nvJPEG)
    """

    def __init__(self, device_id: int = 0):
        self._default_device_id = device_id
        self._logged = False

    @property
    def name(self) -> str:
        return f"torchvision (GPU:{self._default_device_id})"

    def encode(
        self,
        image: np.ndarray | torch.Tensor,
        quality: int,
        chroma_subsampling: ChromaSubsamplingType,
    ) -> bytes:
        # Log once
        if not self._logged:
            device_id = image.device.index if isinstance(image, torch.Tensor) and image.is_cuda else self._default_device_id
            logger.info(f"JPEG encoder: torchvision (GPU:{device_id}, nvJPEG)")
            self._logged = True

        # Prepare tensor
        if isinstance(image, torch.Tensor):
            tensor = self._prepare_tensor(image)
        else:
            # Convert numpy to tensor
            tensor = torch.from_numpy(image)
            if tensor.dim() == 3 and tensor.shape[-1] in (1, 3):
                tensor = tensor.permute(2, 0, 1)  # HWC -> CHW
            tensor = tensor.contiguous()

        # encode_jpeg expects CHW format, uint8
        # Returns a 1D tensor of JPEG bytes
        jpeg_tensor = encode_jpeg(tensor, quality=quality)

        # Convert to bytes (ensure CPU tensor for numpy conversion)
        return jpeg_tensor.cpu().numpy().tobytes()

    def _prepare_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Prepare tensor for encoding.

        Converts HWC to CHW format required by torchvision.encode_jpeg.
        """
        # Convert to uint8 if needed
        if tensor.dtype != torch.uint8:
            if tensor.dtype in (torch.float32, torch.float16):
                if tensor.max() <= 1.0:
                    tensor = (tensor * 255).clamp(0, 255).to(torch.uint8)
                else:
                    tensor = tensor.clamp(0, 255).to(torch.uint8)
            else:
                tensor = tensor.to(torch.uint8)

        # Handle batch dimension
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)

        # Convert HWC to CHW (torchvision requirement)
        if tensor.dim() == 3 and tensor.shape[-1] in (1, 3, 4):
            # Likely HWC format, convert to CHW
            tensor = tensor.permute(2, 0, 1)

        return tensor.contiguous()


class GPUJpegBackend(TorchvisionJpegBackend):
    """GPU JPEG encoding backend using torchvision.io.encode_jpeg (nvJPEG)."""
    pass


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
            # Logging is done per-device in GPUJpegBackend._get_encoder_for_device
            return self._backend

    @property
    def backend_name(self) -> str:
        """Name of the active backend."""
        return self._ensure_backend().name

    def encode(
        self,
        image: np.ndarray | torch.Tensor,
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


def _get_tensor_ref() -> torch.Tensor | None:
    """Get the cached tensor reference for the current thread."""
    return getattr(_thread_local, "tensor_ref", None)


def _set_tensor_ref(tensor: torch.Tensor | None) -> None:
    """Set the cached tensor reference for the current thread.

    This keeps the tensor alive to prevent PyTorch's memory allocator
    from reusing the memory while nvImageCodec may still be reading from it.
    """
    _thread_local.tensor_ref = tensor


@dataclass
class JpegEncodingService:
    """High-level service for JPEG encoding with JPEG bytes caching.

    This is the main entry point for JPEG encoding in the application.
    It provides:
    - JPEG bytes caching for efficient multi-client streaming
    - Thread-safe operation with per-thread cache isolation
    - Memory safety by holding tensor references until next encode

    Thread Safety
    -------------
    Each renderer thread has its own isolated JPEG cache using
    thread-local storage. This prevents race conditions when multiple
    clients are rendering simultaneously.

    Memory Safety
    -------------
    To prevent race conditions with nvImageCodec's internal async operations,
    we keep a reference to the last encoded tensor until the next encode starts.
    This prevents PyTorch's memory allocator from reusing that memory while
    nvImageCodec may still be reading from it.

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
    # Keep reference to last encoded tensor to prevent memory reuse race condition
    # This is stored in thread-local storage for thread safety
    _tensor_ref_lock: threading.Lock = field(init=False, repr=False)

    def __post_init__(self):
        self._encoder = JpegEncoder(self.device)
        self._tensor_ref_lock = threading.Lock()

    @property
    def backend_name(self) -> str:
        """Name of the active encoding backend."""
        return self._encoder.backend_name

    def encode(
        self,
        image: np.ndarray | torch.Tensor,
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
        image: torch.Tensor,
        quality: int | None = None,
        chroma_subsampling: ChromaSubsamplingType | None = None,
    ) -> bytes:
        """Encode image to JPEG and cache the result (thread-local).

        Encodes immediately while we have exclusive access to the tensor,
        then caches the JPEG bytes for later retrieval via get_cached_jpeg().

        IMPORTANT: This method also stores a reference to the tensor to prevent
        PyTorch's memory allocator from reusing the memory while nvImageCodec
        may still have internal async operations reading from it. This prevents
        the horizontal stripe corruption that occurs when memory is overwritten
        during JPEG encoding.

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

        # Encode first (while previous frame's tensor reference is still held)
        jpeg_bytes = self._encoder.encode(image, q, css)

        # CRITICAL: Store reference to the NEW tensor AFTER encoding completes.
        # This ensures the previous frame's tensor memory isn't released until
        # this frame's encoding is done. The sequence is:
        # 1. Previous frame's tensor is still referenced (memory protected)
        # 2. Current frame encodes (may have async internal operations)
        # 3. After encode() returns (with sync), update reference to current frame
        # 4. Previous frame's tensor is now released (safe - its encoding is long done)
        _set_tensor_ref(image)

        _set_cached_jpeg(jpeg_bytes)
        return jpeg_bytes

    def get_cached_jpeg(self) -> bytes | None:
        """Get cached JPEG bytes for current thread."""
        return _get_cached_jpeg()

    def clear_gpu_frame(self) -> None:
        """Clear cached GPU tensor reference for current thread.

        This releases the tensor reference, allowing PyTorch's memory allocator
        to reuse the memory. Should be called during error recovery to ensure
        potentially corrupted tensor references are released.
        """
        _set_tensor_ref(None)
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
    image: torch.Tensor,
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
    image: np.ndarray | torch.Tensor,
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


def clear_gpu_frame() -> None:
    """Clear cached GPU tensor reference for current thread (module-level API).

    This releases the tensor reference, allowing PyTorch's memory allocator
    to reuse the memory. Should be called during error recovery to ensure
    potentially corrupted tensor references are released.
    """
    return get_service().clear_gpu_frame()
