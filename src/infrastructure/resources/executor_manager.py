"""Managed ThreadPoolExecutor with timeout-protected shutdown.

Provides a wrapper around ThreadPoolExecutor that ensures clean shutdown
even when tasks are running or blocked.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any, ParamSpec, TypeVar

from src.shared.exceptions import PluginResourceError, PluginTimeoutError


logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


class ManagedExecutor:
    """ThreadPoolExecutor wrapper with managed lifecycle.

    Features:
    - Timeout-protected shutdown (won't hang forever)
    - Task submission with automatic timeout
    - Graceful degradation on shutdown failure
    - Thread-safe state tracking

    Example
    -------
    >>> executor = ManagedExecutor(max_workers=4, name="FrameLoader")
    >>> future = executor.submit(load_frame, frame_path)
    >>> result = future.result(timeout=5.0)
    >>> executor.shutdown(timeout=2.0)
    """

    def __init__(
        self,
        max_workers: int = 4,
        name: str = "ManagedExecutor",
        *,
        thread_name_prefix: str | None = None,
    ) -> None:
        """Initialize managed executor.

        Parameters
        ----------
        max_workers : int
            Maximum number of worker threads
        name : str
            Name for logging and diagnostics
        thread_name_prefix : str | None
            Prefix for worker thread names (defaults to name)
        """
        self.name = name
        self.max_workers = max_workers
        self._prefix = thread_name_prefix or name

        self._executor: ThreadPoolExecutor | None = None
        self._lock = threading.Lock()
        self._is_shutdown = False
        self._pending_futures: set[Future[Any]] = set()

    def _ensure_executor(self) -> ThreadPoolExecutor:
        """Lazily create executor on first use."""
        if self._executor is None:
            with self._lock:
                if self._executor is None:
                    if self._is_shutdown:
                        raise PluginResourceError(
                            "Executor has been shutdown",
                            resource_type="executor",
                            recoverable=False,
                        )
                    self._executor = ThreadPoolExecutor(
                        max_workers=self.max_workers,
                        thread_name_prefix=self._prefix,
                    )
                    logger.debug(
                        "[%s] Created executor with %d workers",
                        self.name,
                        self.max_workers,
                    )
        return self._executor

    @property
    def is_shutdown(self) -> bool:
        """Whether the executor has been shutdown."""
        return self._is_shutdown

    def submit(
        self,
        fn: Callable[P, T],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Future[T]:
        """Submit a task for execution.

        Parameters
        ----------
        fn : Callable
            Function to execute
        *args : Any
            Positional arguments
        **kwargs : Any
            Keyword arguments

        Returns
        -------
        Future
            Future representing the pending result

        Raises
        ------
        PluginResourceError
            If executor is shutdown
        """
        executor = self._ensure_executor()
        future = executor.submit(fn, *args, **kwargs)

        with self._lock:
            self._pending_futures.add(future)

        # Remove from pending set when done
        def _cleanup(f: Future[Any]) -> None:
            with self._lock:
                self._pending_futures.discard(f)

        future.add_done_callback(_cleanup)
        return future

    def submit_with_timeout(
        self,
        fn: Callable[P, T],
        *args: P.args,
        timeout: float = 30.0,
        **kwargs: P.kwargs,
    ) -> T:
        """Submit task and wait for result with timeout.

        Parameters
        ----------
        fn : Callable
            Function to execute
        *args : Any
            Positional arguments
        timeout : float
            Maximum seconds to wait for result
        **kwargs : Any
            Keyword arguments

        Returns
        -------
        T
            Result of the function

        Raises
        ------
        PluginTimeoutError
            If timeout is exceeded
        """
        future = self.submit(fn, *args, **kwargs)
        try:
            return future.result(timeout=timeout)
        except FuturesTimeoutError:
            future.cancel()
            raise PluginTimeoutError(
                f"Task timed out after {timeout}s",
                timeout_seconds=timeout,
                operation=fn.__name__ if hasattr(fn, "__name__") else "unknown",
            )

    def shutdown(self, timeout: float = 5.0, *, cancel_pending: bool = True) -> bool:
        """Shutdown the executor with timeout protection.

        Parameters
        ----------
        timeout : float
            Maximum seconds to wait for shutdown
        cancel_pending : bool
            Whether to cancel pending futures before shutdown

        Returns
        -------
        bool
            True if shutdown completed cleanly, False if forced
        """
        with self._lock:
            if self._is_shutdown:
                return True
            self._is_shutdown = True
            executor = self._executor
            pending = list(self._pending_futures)

        if executor is None:
            return True

        # Cancel pending futures if requested
        if cancel_pending:
            for future in pending:
                future.cancel()

        # Attempt graceful shutdown
        logger.debug("[%s] Shutting down executor (timeout=%.1fs)", self.name, timeout)

        # Use a thread to perform shutdown with timeout
        shutdown_complete = threading.Event()

        def _do_shutdown() -> None:
            try:
                executor.shutdown(wait=True)
            except Exception as e:
                logger.warning("[%s] Error during shutdown: %s", self.name, e)
            finally:
                shutdown_complete.set()

        shutdown_thread = threading.Thread(target=_do_shutdown, daemon=True)
        shutdown_thread.start()

        if shutdown_complete.wait(timeout=timeout):
            logger.debug("[%s] Executor shutdown complete", self.name)
            return True
        else:
            logger.warning(
                "[%s] Executor shutdown timed out after %.1fs, forcing...",
                self.name,
                timeout,
            )
            # Force shutdown without waiting
            try:
                executor.shutdown(wait=False)
            except Exception:
                pass
            return False

    def get_diagnostics(self) -> dict[str, Any]:
        """Get diagnostic information about the executor."""
        with self._lock:
            return {
                "name": self.name,
                "max_workers": self.max_workers,
                "is_shutdown": self._is_shutdown,
                "pending_tasks": len(self._pending_futures),
                "executor_created": self._executor is not None,
            }

    def __enter__(self) -> ManagedExecutor:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.shutdown()


__all__ = ["ManagedExecutor"]
