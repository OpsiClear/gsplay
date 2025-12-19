"""Retry decorator with exponential backoff.

Provides configurable retry behavior for transient failures.
"""

from __future__ import annotations

import functools
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Callable, TypeVar, ParamSpec, Any

from src.shared.exceptions import PluginError

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


@dataclass
class RetryConfig:
    """Configuration for retry behavior.

    Attributes
    ----------
    max_attempts : int
        Maximum number of attempts (including initial)
    base_delay : float
        Initial delay in seconds
    max_delay : float
        Maximum delay between retries
    exponential_base : float
        Base for exponential backoff (delay = base_delay * base^attempt)
    jitter : bool
        Whether to add random jitter to delays
    retryable_exceptions : tuple[type[Exception], ...]
        Exceptions that trigger retry
    """

    max_attempts: int = 3
    base_delay: float = 0.1
    max_delay: float = 10.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: tuple[type[Exception], ...] = field(
        default_factory=lambda: (Exception,)
    )

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number (0-indexed).

        Parameters
        ----------
        attempt : int
            Attempt number (0 = first retry)

        Returns
        -------
        float
            Delay in seconds
        """
        delay = self.base_delay * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay)

        if self.jitter:
            # Add up to 25% jitter
            delay = delay * (0.75 + random.random() * 0.5)

        return delay


def retry(
    config: RetryConfig | None = None,
    *,
    max_attempts: int | None = None,
    base_delay: float | None = None,
    retryable_exceptions: tuple[type[Exception], ...] | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator to retry a function on failure.

    Can be used with a RetryConfig or individual parameters.

    Example
    -------
    >>> @retry(max_attempts=3, base_delay=0.5)
    ... def load_frame(path: str) -> Data:
    ...     # May fail transiently
    ...     return read_file(path)
    ...
    >>> # Or with config
    >>> @retry(RetryConfig(max_attempts=5, jitter=False))
    ... def load_frame(path: str) -> Data:
    ...     return read_file(path)

    Parameters
    ----------
    config : RetryConfig | None
        Retry configuration (overrides individual params)
    max_attempts : int | None
        Maximum retry attempts
    base_delay : float | None
        Initial delay between retries
    retryable_exceptions : tuple | None
        Exceptions that should trigger retry
    """
    # Build config from parameters if not provided
    if config is None:
        config = RetryConfig(
            max_attempts=max_attempts or 3,
            base_delay=base_delay or 0.1,
            retryable_exceptions=retryable_exceptions or (Exception,),
        )

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_exception: Exception | None = None

            for attempt in range(config.max_attempts):
                try:
                    return func(*args, **kwargs)
                except config.retryable_exceptions as e:
                    last_exception = e

                    # Check if it's a non-recoverable PluginError
                    if isinstance(e, PluginError) and not e.recoverable:
                        logger.debug(
                            "[retry] %s failed with non-recoverable error: %s",
                            func.__name__,
                            e,
                        )
                        raise

                    # Last attempt - don't sleep
                    if attempt == config.max_attempts - 1:
                        break

                    delay = config.get_delay(attempt)
                    logger.debug(
                        "[retry] %s failed (attempt %d/%d), retrying in %.2fs: %s",
                        func.__name__,
                        attempt + 1,
                        config.max_attempts,
                        delay,
                        e,
                    )
                    time.sleep(delay)

            # All attempts failed
            logger.warning(
                "[retry] %s failed after %d attempts",
                func.__name__,
                config.max_attempts,
            )
            raise last_exception  # type: ignore

        return wrapper

    return decorator


def retry_async(
    config: RetryConfig | None = None,
    **kwargs: Any,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Async version of retry decorator.

    Same parameters as retry(), but works with async functions.
    """
    import asyncio

    if config is None:
        config = RetryConfig(
            max_attempts=kwargs.get("max_attempts", 3),
            base_delay=kwargs.get("base_delay", 0.1),
            retryable_exceptions=kwargs.get("retryable_exceptions", (Exception,)),
        )

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_exception: Exception | None = None

            for attempt in range(config.max_attempts):
                try:
                    return await func(*args, **kwargs)  # type: ignore
                except config.retryable_exceptions as e:
                    last_exception = e

                    if isinstance(e, PluginError) and not e.recoverable:
                        raise

                    if attempt == config.max_attempts - 1:
                        break

                    delay = config.get_delay(attempt)
                    await asyncio.sleep(delay)

            raise last_exception  # type: ignore

        return wrapper  # type: ignore

    return decorator


__all__ = ["retry", "retry_async", "RetryConfig"]
