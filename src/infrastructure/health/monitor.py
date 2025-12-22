"""Plugin health monitoring.

Provides centralized health tracking for all plugins,
with background monitoring and diagnostic aggregation.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Protocol

from src.domain.interfaces import HealthCheckResult, HealthStatus


logger = logging.getLogger(__name__)


class HealthCheckable(Protocol):
    """Protocol for health-checkable components."""

    def health_check(self) -> HealthCheckResult:
        """Perform health check."""
        ...

    def get_diagnostics(self) -> dict[str, Any]:
        """Get diagnostic information."""
        ...


class PluginHealthMonitor:
    """Monitors health of registered plugins.

    Features:
    - Manual and automatic health checks
    - Background monitoring thread (optional)
    - Aggregated health status
    - History tracking

    Example
    -------
    >>> monitor = PluginHealthMonitor()
    >>> monitor.register("load-ply", ply_plugin)
    >>> monitor.register("composite", composite_plugin)
    >>>
    >>> # Manual check
    >>> status = monitor.check_all()
    >>>
    >>> # Or start background monitoring
    >>> monitor.start_background_checks(interval=30.0)
    >>> # ... later ...
    >>> monitor.stop_background_checks()
    """

    def __init__(self, max_history: int = 100) -> None:
        """Initialize health monitor.

        Parameters
        ----------
        max_history : int
            Maximum health check results to retain per plugin
        """
        self._plugins: dict[str, HealthCheckable] = {}
        self._history: dict[str, list[HealthCheckResult]] = {}
        self._max_history = max_history
        self._lock = threading.Lock()

        # Background monitoring
        self._monitor_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def register(self, name: str, plugin: HealthCheckable) -> None:
        """Register a plugin for health monitoring.

        Parameters
        ----------
        name : str
            Plugin identifier
        plugin : HealthCheckable
            Plugin implementing health_check()
        """
        with self._lock:
            self._plugins[name] = plugin
            self._history[name] = []
            logger.debug("[HealthMonitor] Registered plugin: %s", name)

    def unregister(self, name: str) -> None:
        """Unregister a plugin.

        Parameters
        ----------
        name : str
            Plugin identifier to remove
        """
        with self._lock:
            self._plugins.pop(name, None)
            self._history.pop(name, None)
            logger.debug("[HealthMonitor] Unregistered plugin: %s", name)

    def check(self, name: str) -> HealthCheckResult | None:
        """Perform health check on a specific plugin.

        Parameters
        ----------
        name : str
            Plugin identifier

        Returns
        -------
        HealthCheckResult | None
            Health check result, or None if plugin not found
        """
        with self._lock:
            plugin = self._plugins.get(name)
            if plugin is None:
                return None

        start = time.time()
        try:
            result = plugin.health_check()
            result.latency_ms = (time.time() - start) * 1000
        except Exception as e:
            result = HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {e}",
                latency_ms=(time.time() - start) * 1000,
            )

        # Store in history
        with self._lock:
            if name in self._history:
                self._history[name].append(result)
                # Trim history
                if len(self._history[name]) > self._max_history:
                    self._history[name] = self._history[name][-self._max_history :]

        return result

    def check_all(self) -> dict[str, HealthCheckResult]:
        """Perform health check on all registered plugins.

        Returns
        -------
        dict[str, HealthCheckResult]
            Map of plugin name to health check result
        """
        with self._lock:
            names = list(self._plugins.keys())

        results = {}
        for name in names:
            result = self.check(name)
            if result is not None:
                results[name] = result

        return results

    def get_aggregate_status(self) -> HealthStatus:
        """Get aggregate health status across all plugins.

        Returns
        -------
        HealthStatus
            Worst status across all plugins
        """
        results = self.check_all()

        if not results:
            return HealthStatus.UNKNOWN

        statuses = [r.status for r in results.values()]

        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        if HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        if HealthStatus.UNKNOWN in statuses:
            return HealthStatus.UNKNOWN
        return HealthStatus.HEALTHY

    def get_diagnostics(self, name: str | None = None) -> dict[str, Any]:
        """Get diagnostic information.

        Parameters
        ----------
        name : str | None
            Specific plugin name, or None for all plugins

        Returns
        -------
        dict[str, Any]
            Diagnostic information
        """
        if name is not None:
            with self._lock:
                plugin = self._plugins.get(name)
                history = list(self._history.get(name, []))

            if plugin is None:
                return {"error": f"Plugin '{name}' not found"}

            try:
                plugin_diag = plugin.get_diagnostics()
            except Exception as e:
                plugin_diag = {"error": str(e)}

            return {
                "plugin": plugin_diag,
                "health_history_count": len(history),
                "last_check": history[-1].timestamp.isoformat() if history else None,
            }

        # All plugins
        with self._lock:
            names = list(self._plugins.keys())

        return {
            "plugins": {n: self.get_diagnostics(n) for n in names},
            "aggregate_status": self.get_aggregate_status().name,
            "monitoring_active": self._monitor_thread is not None
            and self._monitor_thread.is_alive(),
        }

    def get_history(self, name: str, limit: int = 10) -> list[HealthCheckResult]:
        """Get health check history for a plugin.

        Parameters
        ----------
        name : str
            Plugin identifier
        limit : int
            Maximum results to return

        Returns
        -------
        list[HealthCheckResult]
            Recent health check results (newest last)
        """
        with self._lock:
            history = self._history.get(name, [])
            return list(history[-limit:])

    # Background monitoring

    def start_background_checks(self, interval: float = 30.0) -> None:
        """Start background health monitoring.

        Parameters
        ----------
        interval : float
            Seconds between health checks
        """
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            logger.warning("[HealthMonitor] Background monitoring already running")
            return

        self._stop_event.clear()

        def _monitor_loop() -> None:
            logger.info("[HealthMonitor] Background monitoring started (interval=%.1fs)", interval)
            while not self._stop_event.wait(timeout=interval):
                try:
                    results = self.check_all()
                    unhealthy = [
                        n for n, r in results.items() if r.status == HealthStatus.UNHEALTHY
                    ]
                    if unhealthy:
                        logger.warning("[HealthMonitor] Unhealthy plugins: %s", unhealthy)
                except Exception as e:
                    logger.error("[HealthMonitor] Error during health check: %s", e)
            logger.info("[HealthMonitor] Background monitoring stopped")

        self._monitor_thread = threading.Thread(target=_monitor_loop, daemon=True)
        self._monitor_thread.start()

    def stop_background_checks(self, timeout: float = 5.0) -> None:
        """Stop background health monitoring.

        Parameters
        ----------
        timeout : float
            Maximum seconds to wait for thread to stop
        """
        if self._monitor_thread is None:
            return

        self._stop_event.set()
        self._monitor_thread.join(timeout=timeout)

        if self._monitor_thread.is_alive():
            logger.warning("[HealthMonitor] Background thread did not stop within timeout")

        self._monitor_thread = None


# Global singleton (optional usage)
_global_monitor: PluginHealthMonitor | None = None


def get_health_monitor() -> PluginHealthMonitor:
    """Get the global health monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PluginHealthMonitor()
    return _global_monitor


__all__ = ["PluginHealthMonitor", "get_health_monitor"]
