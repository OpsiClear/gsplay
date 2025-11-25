"""
Hybrid (memory + disk) frame cache for Gaussian data.

The cache keeps hot frames in RAM while persisting every processed frame to
disk so the PLY loader can avoid repeated decoding work between playback runs.
"""

from __future__ import annotations

import logging
import pickle
import threading
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from gsply import GSData

logger = logging.getLogger(__name__)


class BinaryFrameCache:
    """
    Sliding-window cache that spills old frames to disk.

    Memory is capped by ``max_memory_mb`` while disk persistence is optional
    (enabled when ``cache_dir`` is provided).
    """

    def __init__(
        self,
        *,
        cache_dir: str | Path | None = None,
        max_memory_mb: int = 2048,
        enable_prefetch: bool = True,
    ):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.enable_prefetch = enable_prefetch

        # In-memory payloads: GSData or tensor dictionaries
        self._memory_cache: dict[int, Any] = {}
        self._current_memory_usage = 0

        self._lock = threading.Lock()
        self._current_frame_idx: int | None = None
        self._playback_direction = 1  # 1 forward, -1 backward
        self.total_frames = 0  # Set externally by models

        logger.debug(
            "[FrameCache] Initialized (memory=%dMB, disk=%s)",
            max_memory_mb,
            self.cache_dir or "disabled",
        )

    # ------------------------------------------------------------------ #
    # Size estimation helpers
    # ------------------------------------------------------------------ #
    def _estimate_frame_size(self, data: Any) -> int:
        """Estimate payload size in bytes."""
        if isinstance(data, GSData):
            size = (
                data.means.nbytes
                + data.scales.nbytes
                + data.quats.nbytes
                + data.opacities.nbytes
                + data.sh0.nbytes
            )
            if data.shN is not None:
                size += data.shN.nbytes
            return size

        if isinstance(data, Mapping):
            return sum(self._estimate_field_size(value) for value in data.values())

        return self._estimate_field_size(data)

    def _estimate_field_size(self, value: Any) -> int:
        if isinstance(value, torch.Tensor):
            return value.element_size() * value.numel()
        if isinstance(value, np.ndarray):
            return value.nbytes
        if isinstance(value, (list, tuple)):
            return sum(self._estimate_field_size(v) for v in value)
        if hasattr(value, "nbytes"):
            try:
                return int(getattr(value, "nbytes"))
            except Exception:  # pragma: no cover - defensive
                return 0
        try:
            return len(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
        except Exception:  # pragma: no cover - defensive
            return 0

    # ------------------------------------------------------------------ #
    # Disk persistence helpers
    # ------------------------------------------------------------------ #
    def _get_cache_path(self, frame_idx: int) -> Path | None:
        if not self.cache_dir:
            return None
        return self.cache_dir / f"frame_{frame_idx:06d}.cache"

    def _write_to_disk(self, frame_idx: int, data: Any) -> None:
        path = self._get_cache_path(frame_idx)
        if path is None:
            return
        try:
            torch.save(data, path)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to persist frame %d: %s", frame_idx, exc)
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass

    def _load_from_disk(self, frame_idx: int) -> Any | None:
        path = self._get_cache_path(frame_idx)
        if path is None or not path.exists():
            return None
        try:
            return torch.load(path, map_location="cpu")
        except Exception:
            logger.warning("[FrameCache] Corrupted cache file %s; deleting.", path)
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
            return None

    def _disk_stats(self) -> tuple[int, float]:
        if not self.cache_dir:
            return 0, 0.0
        total_bytes = 0
        files = list(self.cache_dir.glob("frame_*.cache"))
        for file in files:
            try:
                total_bytes += file.stat().st_size
            except OSError:
                continue
        return len(files), total_bytes / 1024 / 1024

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def has_cache(self, frame_idx: int) -> bool:
        """Return True if the frame exists in memory or on disk."""
        if frame_idx in self._memory_cache:
            return True
        path = self._get_cache_path(frame_idx)
        return bool(path and path.exists())

    def get(self, frame_idx: int) -> Any | None:
        """
        Retrieve cached data, loading from disk if necessary.
        """
        with self._lock:
            self._current_frame_idx = frame_idx

            data = self._memory_cache.get(frame_idx)
            if data is not None:
                logger.debug(f"[FrameCache] Memory hit for frame {frame_idx}")
                return data

            disk_data = self._load_from_disk(frame_idx)
            if disk_data is None:
                logger.debug(f"[FrameCache] Miss for frame {frame_idx}")
                return None

            logger.debug(f"[FrameCache] Loaded frame {frame_idx} from disk")
            frame_size = self._estimate_frame_size(disk_data)
            self._add_to_memory_cache(frame_idx, disk_data, frame_size)
            return disk_data

    def put(self, frame_idx: int, payload: Any) -> None:
        """
        Store the payload in memory and persist it to disk.
        """
        frame_size = self._estimate_frame_size(payload)
        if self._current_frame_idx is not None:
            self._playback_direction = 1 if frame_idx > self._current_frame_idx else -1

        with self._lock:
            self._add_to_memory_cache(frame_idx, payload, frame_size)
            self._write_to_disk(frame_idx, payload)

    def _add_to_memory_cache(self, frame_idx: int, payload: Any, frame_size: int) -> None:
        while (self._current_memory_usage + frame_size > self.max_memory_bytes
               and self._memory_cache):
            evict_idx = self._find_eviction_candidate()
            if evict_idx is None:
                logger.warning("[FrameCache] Cannot evict to make room for %d", frame_idx)
                return

            evicted = self._memory_cache.pop(evict_idx)
            self._current_memory_usage -= self._estimate_frame_size(evicted)
            logger.debug(f"[FrameCache] Evicted frame {evict_idx}")

        if frame_size <= self.max_memory_bytes:
            self._memory_cache[frame_idx] = payload
            self._current_memory_usage += frame_size
            logger.debug(
                "[FrameCache] Added frame %d (%.1f/%.0f MB)",
                frame_idx,
                self._current_memory_usage / 1024 / 1024,
                self.max_memory_bytes / 1024 / 1024,
            )

    def _find_eviction_candidate(self) -> int | None:
        if not self._memory_cache:
            return None

        cached_frames = list(self._memory_cache.keys())
        if self._current_frame_idx is None:
            return min(cached_frames)

        if self._playback_direction == 1:
            frames_behind = [f for f in cached_frames if f < self._current_frame_idx]
            if frames_behind:
                return min(frames_behind)
        else:
            frames_ahead = [f for f in cached_frames if f > self._current_frame_idx]
            if frames_ahead:
                return max(frames_ahead)

        return max(cached_frames, key=lambda f: abs(f - self._current_frame_idx))

    def prefetch_next(self, current_frame: int, num_frames: int = 32) -> list[int]:
        if not self.enable_prefetch:
            return []

        frames_to_load: list[int] = []
        for i in range(1, num_frames + 1):
            next_frame = current_frame + (i * self._playback_direction)
            if next_frame < 0 or (self.total_frames > 0 and next_frame >= self.total_frames):
                continue
            if not self.has_cache(next_frame):
                frames_to_load.append(next_frame)
                logger.debug(f"[FrameCache] Prefetch hint for frame {next_frame}")
        return frames_to_load

    def set_playback_direction(self, direction: int) -> None:
        self._playback_direction = direction

    def clear(self) -> None:
        with self._lock:
            self._memory_cache.clear()
            self._current_memory_usage = 0
            if self.cache_dir:
                for cache_file in self.cache_dir.glob("frame_*.cache"):
                    try:
                        cache_file.unlink()
                    except OSError:
                        logger.debug("Failed to delete cache file %s", cache_file)

        logger.info("[FrameCache] Cleared all cached data")

    def get_stats(self) -> dict:
        with self._lock:
            disk_frames, disk_usage_mb = self._disk_stats()
            return {
                'memory_frames': len(self._memory_cache),
                'memory_usage_mb': self._current_memory_usage / 1024 / 1024,
                'memory_limit_mb': self.max_memory_bytes / 1024 / 1024,
                'disk_frames': disk_frames,
                'disk_usage_mb': disk_usage_mb,
                'playback_direction': 'forward' if self._playback_direction == 1 else 'backward',
            }
