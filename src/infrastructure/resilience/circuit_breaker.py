"""Circuit breaker pattern for failure isolation.

Prevents cascade failures by temporarily blocking operations
after a threshold of failures is reached.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, TypeVar, ParamSpec, Any

from src.shared.exceptions import CircuitBreakerOpenError

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = auto()  # Normal operation, requests pass through
    OPEN = auto()  # Failing, requests are rejected immediately
    HALF_OPEN = auto()  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker.

    Attributes
    ----------
    failure_threshold : int
        Number of failures before opening circuit
    success_threshold : int
        Number of successes in HALF_OPEN to close circuit
    reset_timeout : float
        Seconds to wait before trying HALF_OPEN
    half_open_max_calls : int
        Max concurrent calls allowed in HALF_OPEN state
    """

    failure_threshold: int = 5
    success_threshold: int = 2
    reset_timeout: float = 30.0
    half_open_max_calls: int = 1


class CircuitBreaker:
    """Circuit breaker for plugin operations.

    Implements the circuit breaker pattern to prevent repeated
    calls to failing operations.

    Example
    -------
    >>> breaker = CircuitBreaker("FrameLoader")
    >>>
    >>> @breaker
    ... def load_frame(path: str) -> Data:
    ...     return read_file(path)
    ...
    >>> # Or explicit calls
    >>> breaker.call(load_frame, "path/to/file.ply")

    States
    ------
    CLOSED: Normal operation
        - Failures increment counter
        - When failures >= threshold, transition to OPEN

    OPEN: Rejecting all calls
        - Calls immediately raise CircuitBreakerOpenError
        - After reset_timeout, transition to HALF_OPEN

    HALF_OPEN: Testing recovery
        - Allow limited calls through
        - Success: increment success counter, if >= threshold -> CLOSED
        - Failure: immediately transition to OPEN
    """

    def __init__(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
    ) -> None:
        """Initialize circuit breaker.

        Parameters
        ----------
        name : str
            Name for logging and diagnostics
        config : CircuitBreakerConfig | None
            Configuration (uses defaults if None)
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float | None = None
        self._half_open_calls = 0
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        """Current circuit state."""
        with self._lock:
            self._check_state_transition()
            return self._state

    def _check_state_transition(self) -> None:
        """Check if state should transition (called with lock held)."""
        if self._state == CircuitState.OPEN:
            if self._last_failure_time is not None:
                elapsed = time.time() - self._last_failure_time
                if elapsed >= self.config.reset_timeout:
                    logger.info(
                        "[%s] Circuit transitioning OPEN -> HALF_OPEN after %.1fs",
                        self.name,
                        elapsed,
                    )
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
                    self._success_count = 0

    def _record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    logger.info(
                        "[%s] Circuit transitioning HALF_OPEN -> CLOSED after %d successes",
                        self.name,
                        self._success_count,
                    )
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success
                self._failure_count = 0

    def _record_failure(self, error: Exception) -> None:
        """Record a failed call."""
        with self._lock:
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                logger.warning(
                    "[%s] Circuit transitioning HALF_OPEN -> OPEN due to failure: %s",
                    self.name,
                    error,
                )
                self._state = CircuitState.OPEN
            elif self._state == CircuitState.CLOSED:
                self._failure_count += 1
                if self._failure_count >= self.config.failure_threshold:
                    logger.warning(
                        "[%s] Circuit transitioning CLOSED -> OPEN after %d failures",
                        self.name,
                        self._failure_count,
                    )
                    self._state = CircuitState.OPEN

    def _can_execute(self) -> bool:
        """Check if a call can be executed (called with lock held)."""
        self._check_state_transition()

        if self._state == CircuitState.CLOSED:
            return True
        elif self._state == CircuitState.HALF_OPEN:
            if self._half_open_calls < self.config.half_open_max_calls:
                self._half_open_calls += 1
                return True
            return False
        else:  # OPEN
            return False

    def call(self, fn: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
        """Execute a function through the circuit breaker.

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
        T
            Function result

        Raises
        ------
        CircuitBreakerOpenError
            If circuit is open
        """
        with self._lock:
            if not self._can_execute():
                time_until_reset = None
                if self._last_failure_time is not None:
                    elapsed = time.time() - self._last_failure_time
                    time_until_reset = max(0, self.config.reset_timeout - elapsed)

                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{self.name}' is open",
                    plugin_name=self.name,
                    failures_count=self._failure_count,
                    reset_timeout=time_until_reset,
                )

        try:
            result = fn(*args, **kwargs)
            self._record_success()
            return result
        except Exception as e:
            self._record_failure(e)
            raise

    def __call__(self, fn: Callable[P, T]) -> Callable[P, T]:
        """Use as a decorator."""
        import functools

        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            return self.call(fn, *args, **kwargs)

        return wrapper

    def reset(self) -> None:
        """Manually reset the circuit breaker to CLOSED state."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None
            self._half_open_calls = 0
            logger.info("[%s] Circuit breaker manually reset", self.name)

    def get_diagnostics(self) -> dict[str, Any]:
        """Get diagnostic information."""
        with self._lock:
            self._check_state_transition()
            time_in_state = None
            if self._last_failure_time is not None and self._state == CircuitState.OPEN:
                time_in_state = time.time() - self._last_failure_time

            return {
                "name": self.name,
                "state": self._state.name,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "time_in_open_state": time_in_state,
                "config": {
                    "failure_threshold": self.config.failure_threshold,
                    "success_threshold": self.config.success_threshold,
                    "reset_timeout": self.config.reset_timeout,
                },
            }


__all__ = ["CircuitBreaker", "CircuitBreakerConfig", "CircuitState"]
