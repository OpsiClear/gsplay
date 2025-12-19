"""Lifecycle management infrastructure for plugins.

This module provides:
- LifecycleMixin: Base class with default lifecycle implementations
- Context manager support for automatic resource cleanup
- State transition validation
"""

from __future__ import annotations

import logging
from typing import Any

from src.domain.interfaces import PluginState, HealthStatus, HealthCheckResult

logger = logging.getLogger(__name__)


class LifecycleError(Exception):
    """Raised when lifecycle operations fail."""

    def __init__(self, message: str, current_state: PluginState | None = None) -> None:
        self.current_state = current_state
        full_message = message
        if current_state is not None:
            full_message = f"{message} (current state: {current_state.name})"
        super().__init__(full_message)


class InvalidStateTransitionError(LifecycleError):
    """Raised when an invalid state transition is attempted."""

    def __init__(
        self,
        from_state: PluginState,
        to_state: PluginState,
        message: str | None = None,
    ) -> None:
        self.from_state = from_state
        self.to_state = to_state
        default_message = f"Invalid state transition: {from_state.name} -> {to_state.name}"
        super().__init__(message or default_message, from_state)


# Valid state transitions
_VALID_TRANSITIONS: dict[PluginState, set[PluginState]] = {
    PluginState.CREATED: {PluginState.INITIALIZING, PluginState.TERMINATED},
    PluginState.INITIALIZING: {PluginState.READY, PluginState.TERMINATED},
    PluginState.READY: {PluginState.SUSPENDED, PluginState.SHUTTING_DOWN},
    PluginState.SUSPENDED: {PluginState.READY, PluginState.SHUTTING_DOWN},
    PluginState.SHUTTING_DOWN: {PluginState.TERMINATED},
    PluginState.TERMINATED: set(),  # Terminal state
}


class LifecycleMixin:
    """Mixin providing default lifecycle management.

    Provides:
    - State tracking with transition validation
    - Default implementations for lifecycle hooks
    - Context manager support (__enter__/__exit__)
    - Health check baseline implementation

    Example
    -------
    >>> class MySource(LifecycleMixin, BaseGaussianSource):
    ...     def __init__(self, config: dict) -> None:
    ...         super().__init__()
    ...         self._config = config
    ...
    ...     def on_init(self) -> None:
    ...         super().on_init()  # Updates state
    ...         # Custom initialization here
    ...
    >>> with MySource(config) as source:
    ...     data = source.get_frame_at_time(0.5)
    """

    def __init__(self) -> None:
        self._state: PluginState = PluginState.CREATED
        self._initialization_error: Exception | None = None

    @property
    def state(self) -> PluginState:
        """Current lifecycle state."""
        return self._state

    def _transition_to(self, new_state: PluginState) -> None:
        """Transition to a new state with validation.

        Parameters
        ----------
        new_state : PluginState
            Target state to transition to

        Raises
        ------
        InvalidStateTransitionError
            If the transition is not allowed
        """
        valid_targets = _VALID_TRANSITIONS.get(self._state, set())
        if new_state not in valid_targets:
            raise InvalidStateTransitionError(self._state, new_state)

        old_state = self._state
        self._state = new_state
        logger.debug(
            "[%s] State transition: %s -> %s",
            self.__class__.__name__,
            old_state.name,
            new_state.name,
        )

    def on_init(self) -> None:
        """Called after construction for heavy initialization.

        Override in subclasses for custom initialization logic.
        Always call super().on_init() first.
        """
        if self._state != PluginState.CREATED:
            raise LifecycleError(
                "on_init() can only be called from CREATED state",
                self._state,
            )

        self._transition_to(PluginState.INITIALIZING)
        # Subclasses do their initialization here
        self._transition_to(PluginState.READY)

    def on_load(self) -> None:
        """Called when plugin becomes active/visible.

        Override in subclasses to load resources (e.g., first frame cache).
        """
        if self._state == PluginState.SUSPENDED:
            self._transition_to(PluginState.READY)
        elif self._state != PluginState.READY:
            raise LifecycleError(
                "on_load() requires READY or SUSPENDED state",
                self._state,
            )

    def on_unload(self) -> None:
        """Called when plugin becomes inactive.

        Override in subclasses to release temporary resources (e.g., GPU tensors).
        """
        if self._state != PluginState.READY:
            raise LifecycleError(
                "on_unload() requires READY state",
                self._state,
            )
        self._transition_to(PluginState.SUSPENDED)

    def on_shutdown(self, timeout: float = 5.0) -> None:
        """Called for final cleanup.

        Parameters
        ----------
        timeout : float
            Maximum seconds to wait for cleanup (unused in base implementation)

        Override in subclasses to cleanup resources (executors, file handles, etc).
        Always call super().on_shutdown() last.
        """
        if self._state in (PluginState.TERMINATED, PluginState.SHUTTING_DOWN):
            return  # Already shutting down or terminated

        if self._state not in (PluginState.READY, PluginState.SUSPENDED):
            # Allow shutdown from CREATED or INITIALIZING (failed init)
            if self._state not in (PluginState.CREATED, PluginState.INITIALIZING):
                raise LifecycleError(
                    "on_shutdown() called from invalid state",
                    self._state,
                )

        self._transition_to(PluginState.SHUTTING_DOWN)
        # Subclasses do their cleanup here
        self._transition_to(PluginState.TERMINATED)

    # Context manager support

    def __enter__(self) -> "LifecycleMixin":
        """Enter context - initialize plugin."""
        if self._state == PluginState.CREATED:
            try:
                self.on_init()
            except Exception as e:
                self._initialization_error = e
                # Ensure cleanup even on failed init
                try:
                    self.on_shutdown()
                except Exception:
                    pass
                raise
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context - shutdown plugin."""
        if self._state not in (PluginState.TERMINATED, PluginState.SHUTTING_DOWN):
            try:
                self.on_shutdown()
            except Exception as e:
                logger.warning(
                    "[%s] Error during context exit shutdown: %s",
                    self.__class__.__name__,
                    e,
                )

    # Health check baseline

    def health_check(self) -> HealthCheckResult:
        """Basic health check based on lifecycle state.

        Override in subclasses for more detailed health checks.
        """
        if self._state == PluginState.READY:
            return HealthCheckResult(
                status=HealthStatus.HEALTHY,
                message="Plugin is ready",
            )
        elif self._state == PluginState.SUSPENDED:
            return HealthCheckResult(
                status=HealthStatus.DEGRADED,
                message="Plugin is suspended",
            )
        elif self._state in (PluginState.TERMINATED, PluginState.SHUTTING_DOWN):
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Plugin is {self._state.name.lower()}",
            )
        else:
            return HealthCheckResult(
                status=HealthStatus.UNKNOWN,
                message=f"Plugin state: {self._state.name}",
            )

    def get_diagnostics(self) -> dict[str, Any]:
        """Get basic diagnostic information.

        Override in subclasses to add more detailed diagnostics.
        """
        return {
            "class": self.__class__.__name__,
            "state": self._state.name,
            "has_init_error": self._initialization_error is not None,
        }


__all__ = [
    "LifecycleError",
    "InvalidStateTransitionError",
    "LifecycleMixin",
]
