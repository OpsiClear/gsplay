"""GPU resource management for plugins.

Provides explicit tracking and cleanup of GPU tensors to prevent
memory leaks and enable controlled resource release.
"""

from __future__ import annotations

import logging
import threading
import weakref
from typing import Any, TypeVar


logger = logging.getLogger(__name__)

T = TypeVar("T")


class GPUResourceManager:
    """Manages GPU tensor lifecycle for plugins.

    Features:
    - Explicit tensor registration and cleanup
    - Weak reference tracking (doesn't prevent GC)
    - Bulk cleanup operations
    - Memory usage estimation

    Example
    -------
    >>> manager = GPUResourceManager("MyPlugin")
    >>> tensor = manager.register(torch.zeros(1000, device="cuda"))
    >>> # ... use tensor ...
    >>> manager.clear_all()  # Explicit cleanup
    """

    def __init__(self, name: str = "GPUResourceManager") -> None:
        """Initialize GPU resource manager.

        Parameters
        ----------
        name : str
            Name for logging and diagnostics
        """
        self.name = name
        self._tensors: dict[int, weakref.ref[Any]] = {}
        self._lock = threading.Lock()
        self._registration_count = 0

    def register(self, tensor: T) -> T:
        """Register a tensor for tracking.

        Parameters
        ----------
        tensor : T
            GPU tensor to track

        Returns
        -------
        T
            The same tensor (for chaining)
        """
        tensor_id = id(tensor)

        def _on_delete(ref: weakref.ref[Any]) -> None:
            with self._lock:
                self._tensors.pop(tensor_id, None)

        with self._lock:
            try:
                self._tensors[tensor_id] = weakref.ref(tensor, _on_delete)
                self._registration_count += 1
            except TypeError:
                # Some objects can't have weak references
                logger.debug("[%s] Could not create weak reference for tensor", self.name)

        return tensor

    def clear_all(self) -> int:
        """Clear all tracked tensors.

        This forces tensor deletion if they're still alive.

        Returns
        -------
        int
            Number of tensors cleared
        """
        cleared = 0

        with self._lock:
            tensor_refs = list(self._tensors.values())
            self._tensors.clear()

        for ref in tensor_refs:
            tensor = ref()
            if tensor is not None:
                # Try to delete the tensor
                try:
                    if hasattr(tensor, "data"):
                        # PyTorch tensor - clear data
                        del tensor.data
                    cleared += 1
                except Exception as e:
                    logger.debug("[%s] Error clearing tensor: %s", self.name, e)

        if cleared > 0:
            logger.debug("[%s] Cleared %d GPU tensors", self.name, cleared)

        # Request CUDA memory cleanup
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        return cleared

    def get_alive_count(self) -> int:
        """Get count of tensors still alive."""
        with self._lock:
            count = 0
            for ref in self._tensors.values():
                if ref() is not None:
                    count += 1
            return count

    def estimate_memory_bytes(self) -> int:
        """Estimate total GPU memory used by tracked tensors.

        Returns
        -------
        int
            Estimated memory in bytes (0 if unable to determine)
        """
        total_bytes = 0

        with self._lock:
            for ref in self._tensors.values():
                tensor = ref()
                if tensor is not None:
                    try:
                        # PyTorch tensor
                        if hasattr(tensor, "element_size") and hasattr(tensor, "numel"):
                            total_bytes += tensor.element_size() * tensor.numel()
                        # NumPy array
                        elif hasattr(tensor, "nbytes"):
                            total_bytes += tensor.nbytes
                    except Exception:
                        pass

        return total_bytes

    def get_diagnostics(self) -> dict[str, Any]:
        """Get diagnostic information."""
        alive = self.get_alive_count()
        memory = self.estimate_memory_bytes()

        return {
            "name": self.name,
            "tracked_tensors": len(self._tensors),
            "alive_tensors": alive,
            "total_registrations": self._registration_count,
            "estimated_memory_mb": memory / (1024 * 1024) if memory > 0 else 0,
        }

    def __enter__(self) -> GPUResourceManager:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.clear_all()


__all__ = ["GPUResourceManager"]
