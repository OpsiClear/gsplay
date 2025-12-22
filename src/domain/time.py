"""Time domain abstraction for continuous time support.

This module provides the TimeDomain abstraction that allows data sources
to express their native time representation and enables the viewer to
adapt its UI and playback accordingly.

Example
-------
>>> # Discrete frames (PLY sequence)
>>> td = TimeDomain.discrete(100)
>>> td.is_continuous  # False
>>> td.frame_count  # 100

>>> # Continuous seconds (neural model)
>>> td = TimeDomain.continuous(5.0)
>>> td.is_continuous  # True
>>> td.max_time  # 5.0

>>> # Interpolated keyframes (PLY with interpolation)
>>> td = TimeDomain.interpolated([0.0, 1.0, 2.0, 3.0])
>>> td.is_continuous  # True (can query at 0.5, 1.5, etc.)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TimeDomain:
    """Describes how a source represents time.

    Simplified time domain with just the essential fields:
    - Time range (min_time, max_time)
    - Whether continuous or discrete (is_continuous)
    - Source FPS for frame-to-seconds conversion (source_fps)
    - Keyframe times for interpolation (keyframe_times)

    Attributes
    ----------
    min_time : float
        Minimum time value (usually 0.0)
    max_time : float
        Maximum time value (frame index for discrete, seconds for continuous)
    is_continuous : bool
        True for continuous sources (neural networks, interpolated)
        False for discrete frame sources
    source_fps : float | None
        Original capture FPS. Used for frame-to-seconds conversion.
    keyframe_times : list[float] | None
        For interpolating sources, the discrete keyframe times.
        None for truly continuous sources or simple discrete sources.

    Examples
    --------
    >>> td = TimeDomain.discrete(100)
    >>> td.to_normalized(50)  # 0.505...
    >>> td.from_normalized(0.5)  # 49.5
    """

    min_time: float = 0.0
    max_time: float = 1.0
    is_continuous: bool = False
    source_fps: float | None = None
    keyframe_times: list[float] | None = None

    @property
    def is_discrete(self) -> bool:
        """Whether this domain represents discrete time points only."""
        return not self.is_continuous

    @property
    def total_duration(self) -> float:
        """Total time span in source units."""
        return self.max_time - self.min_time

    @property
    def frame_count(self) -> int:
        """Number of discrete frames.

        For continuous domains, returns an approximate count.
        """
        if self.keyframe_times is not None:
            return len(self.keyframe_times)
        if not self.is_continuous:
            # Discrete frames: max_time is last frame index
            return int(self.max_time - self.min_time) + 1
        # For truly continuous, estimate at 30fps
        return max(1, int(self.total_duration * 30))

    @property
    def duration_seconds(self) -> float | None:
        """Get total duration in seconds if calculable.

        Returns
        -------
        float | None
            Duration in seconds, or None if not calculable
        """
        if self.source_fps is not None and self.source_fps > 0:
            # Frame-based with known FPS
            return self.total_duration / self.source_fps
        elif self.is_continuous and self.keyframe_times is None:
            # Truly continuous (neural network) - assume max_time is in seconds
            return self.total_duration
        return None

    def to_normalized(self, source_time: float) -> float:
        """Convert source time to normalized [0, 1].

        Parameters
        ----------
        source_time : float
            Time in source-native units

        Returns
        -------
        float
            Normalized time in [0, 1]
        """
        if self.max_time == self.min_time:
            return 0.0
        return (source_time - self.min_time) / (self.max_time - self.min_time)

    def from_normalized(self, normalized_time: float) -> float:
        """Convert normalized [0, 1] to source time.

        Parameters
        ----------
        normalized_time : float
            Normalized time in [0, 1]

        Returns
        -------
        float
            Time in source-native units
        """
        return self.min_time + normalized_time * (self.max_time - self.min_time)

    def clamp(self, source_time: float) -> float:
        """Clamp source time to valid range.

        Parameters
        ----------
        source_time : float
            Time in source-native units

        Returns
        -------
        float
            Clamped time within [min_time, max_time]
        """
        return max(self.min_time, min(source_time, self.max_time))

    def snap_to_frame(self, source_time: float) -> int:
        """Snap source time to nearest frame index.

        Parameters
        ----------
        source_time : float
            Time in source-native units

        Returns
        -------
        int
            Nearest frame index
        """
        frame = round(source_time - self.min_time)
        return max(0, min(frame, self.frame_count - 1))

    def get_surrounding_keyframes(self, source_time: float) -> tuple[int, int, float] | None:
        """Get surrounding keyframe indices and interpolation factor.

        For interpolation sources, finds the two keyframes surrounding
        the given time and the interpolation factor between them.

        Parameters
        ----------
        source_time : float
            Time in source-native units

        Returns
        -------
        tuple[int, int, float] | None
            (lower_idx, upper_idx, t) where t is interpolation factor [0, 1],
            or None if no keyframes defined
        """
        if self.keyframe_times is None or len(self.keyframe_times) < 2:
            return None

        # Clamp to valid range
        source_time = self.clamp(source_time)

        # Find surrounding keyframes
        for i in range(len(self.keyframe_times) - 1):
            t0 = self.keyframe_times[i]
            t1 = self.keyframe_times[i + 1]
            if t0 <= source_time <= t1:
                if t1 == t0:
                    return (i, i, 0.0)
                t = (source_time - t0) / (t1 - t0)
                return (i, i + 1, t)

        # Edge case: at or beyond last keyframe
        return (len(self.keyframe_times) - 1, len(self.keyframe_times) - 1, 0.0)

    def source_time_to_nearest_keyframe(self, source_time: float) -> tuple[int, float]:
        """Find nearest keyframe to the given source time.

        For snap-to-keyframe export: converts a continuous time to the
        nearest discrete keyframe index, avoiding interpolation.

        Parameters
        ----------
        source_time : float
            Time in source-native units

        Returns
        -------
        tuple[int, float]
            (keyframe_index, keyframe_time) - the index and actual time
            of the nearest keyframe
        """
        import bisect

        # Handle None or empty keyframe_times - fall back to frame index rounding
        if self.keyframe_times is None or len(self.keyframe_times) == 0:
            # Discrete source without explicit keyframes - round to nearest frame
            idx = round(source_time)
            idx = max(0, min(idx, self.frame_count - 1))
            return (idx, float(idx))

        # Single keyframe case
        if len(self.keyframe_times) == 1:
            return (0, self.keyframe_times[0])

        # Binary search for position
        pos = bisect.bisect_left(self.keyframe_times, source_time)

        if pos == 0:
            return (0, self.keyframe_times[0])
        if pos >= len(self.keyframe_times):
            last_idx = len(self.keyframe_times) - 1
            return (last_idx, self.keyframe_times[last_idx])

        # Compare distances to keyframes at pos-1 and pos
        before_idx = pos - 1
        before_time = self.keyframe_times[before_idx]
        after_time = self.keyframe_times[pos]

        if abs(source_time - before_time) <= abs(after_time - source_time):
            return (before_idx, before_time)
        return (pos, after_time)

    # =========================================================================
    # Factory Methods
    # =========================================================================

    @classmethod
    def discrete(
        cls,
        total_frames: int,
        source_fps: float | None = None,
    ) -> TimeDomain:
        """Create a discrete frame-based time domain.

        Parameters
        ----------
        total_frames : int
            Total number of frames
        source_fps : float | None
            Original capture FPS (for frame-to-seconds conversion)

        Returns
        -------
        TimeDomain
            Discrete frame-based time domain
        """
        return cls(
            min_time=0.0,
            max_time=float(total_frames - 1) if total_frames > 1 else 0.0,
            is_continuous=False,
            source_fps=source_fps,
            keyframe_times=None,
        )

    @classmethod
    def continuous(
        cls,
        duration: float,
        start_time: float = 0.0,
    ) -> TimeDomain:
        """Create a continuous seconds-based time domain.

        For neural network models that can evaluate at any time.

        Parameters
        ----------
        duration : float
            Total duration in seconds
        start_time : float
            Start time in seconds (default 0.0)

        Returns
        -------
        TimeDomain
            Continuous time domain
        """
        return cls(
            min_time=start_time,
            max_time=start_time + duration,
            is_continuous=True,
            source_fps=None,
            keyframe_times=None,
        )

    @classmethod
    def interpolated(
        cls,
        keyframe_times: list[float],
        source_fps: float | None = None,
    ) -> TimeDomain:
        """Create a time domain for interpolated keyframes.

        The source has discrete keyframes but supports interpolation
        between them for arbitrary time queries.

        Parameters
        ----------
        keyframe_times : list[float]
            Times of each keyframe (usually frame indices: [0, 1, 2, ...])
        source_fps : float | None
            Original capture FPS (for frame-to-seconds conversion)

        Returns
        -------
        TimeDomain
            Interpolated keyframe time domain
        """
        if not keyframe_times:
            raise ValueError("keyframe_times cannot be empty")

        sorted_times = sorted(keyframe_times)

        return cls(
            min_time=sorted_times[0],
            max_time=sorted_times[-1],
            is_continuous=True,
            source_fps=source_fps,
            keyframe_times=sorted_times,
        )

    # =========================================================================
    # Backward Compatibility Aliases
    # =========================================================================

    @classmethod
    def discrete_frames(
        cls,
        total_frames: int,
        source_fps: float | None = None,
    ) -> TimeDomain:
        """Alias for discrete() - backward compatibility."""
        return cls.discrete(total_frames, source_fps)

    @classmethod
    def continuous_seconds(
        cls,
        duration: float,
        start_time: float = 0.0,
    ) -> TimeDomain:
        """Alias for continuous() - backward compatibility."""
        return cls.continuous(duration, start_time)

    @classmethod
    def interpolated_keyframes(
        cls,
        keyframe_times: list[float],
        source_fps: float | None = None,
        **kwargs,  # Ignore old parameters like unit, display_format
    ) -> TimeDomain:
        """Alias for interpolated() - backward compatibility."""
        return cls.interpolated(keyframe_times, source_fps)
