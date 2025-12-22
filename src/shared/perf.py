"""
Lightweight performance instrumentation helpers.

These utilities keep runtime profiling concerns decoupled from the business
logic so we can plug in loggers, telemetry systems, or tests without touching
core modules.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class FrameLoadEvent:
    """Immutable event describing the result of loading a single frame."""

    frame_idx: int
    pipeline: str
    total_ms: float
    stage_timings: dict[str, float]
    prefetch_hit: bool


class FrameLoadObserver(Protocol):
    """Observer contract for frame load events."""

    def on_frame_loaded(self, event: FrameLoadEvent) -> None: ...


class FrameProfilingBroadcaster:
    """
    Simple pub/sub helper used by loaders that want to expose pluggable sinks.
    """

    def __init__(self) -> None:
        self._observers: list[FrameLoadObserver] = []

    def subscribe(self, observer: FrameLoadObserver) -> None:
        if observer not in self._observers:
            self._observers.append(observer)

    def notify(self, event: FrameLoadEvent) -> None:
        for observer in list(self._observers):
            observer.on_frame_loaded(event)


class FrameThroughputObserver(FrameLoadObserver):
    """
    Tracks real-time frame throughput (frames per second) based on load events.

    The observer keeps a sliding time window and logs periodically so operators
    can see actual frame ingestion rates instead of theoretical FPS.
    """

    def __init__(
        self,
        *,
        window_seconds: float = 3.0,
        log_interval_seconds: float = 5.0,
        logger: logging.Logger | None = None,
    ) -> None:
        self._window_seconds = max(window_seconds, 0.1)
        self._log_interval = max(log_interval_seconds, 0.5)
        self._timestamps: deque[float] = deque()
        self._logger = logger or logging.getLogger(__name__)
        self._last_log_time: float = 0.0
        self.latest_fps: float = 0.0

    def on_frame_loaded(self, event: FrameLoadEvent) -> None:
        now = time.perf_counter()
        self._timestamps.append(now)
        cutoff = now - self._window_seconds
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()

        count = len(self._timestamps)
        if count <= 1:
            self.latest_fps = 0.0
        else:
            duration = max(self._timestamps[-1] - self._timestamps[0], 1e-6)
            self.latest_fps = count / duration

        if now - self._last_log_time >= self._log_interval:
            self._last_log_time = now
            self._logger.info(
                "[Loader] Throughput %.1f FPS (window=%d frames over %.2fs)",
                self.latest_fps,
                count,
                max(self._timestamps[-1] - self._timestamps[0], self._window_seconds),
            )


class PerfMonitor:
    """
    Context-style helper for timing multi-stage work.
    """

    def __init__(self, label: str | None = None) -> None:
        self.label = label or ""
        self._start = time.perf_counter()
        self._timings: dict[str, float] = {}

    @contextmanager
    def track(self, stage: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self._timings[stage] = (time.perf_counter() - start) * 1000

    def record(self, stage: str, duration_ms: float) -> None:
        """Record a duration explicitly (useful for skipped stages)."""
        self._timings[stage] = duration_ms

    def stop(self) -> tuple[dict[str, float], float]:
        """
        Finish timing and return (stage_timings, total_ms).
        """
        total_ms = (time.perf_counter() - self._start) * 1000
        return self._timings.copy(), total_ms
