"""Synchronized playback: mapping a common clock to per-video frames.

The mapping mirrors the render-time composition semantics: every video
starts at its own start frame at t=0 and freezes on its end frame.
"""

from __future__ import annotations


class PlaybackClock:
    """Accumulates playback time, scaled by the current slow-motion speed.

    Speed changes only affect time accumulated after the change, so
    adjusting the preview speed mid-playback does not jump the position.
    """

    def __init__(self, speed: float = 1.0) -> None:
        self.speed = speed
        self._t = 0.0

    @property
    def t(self) -> float:
        return self._t

    def advance(self, dt_seconds: float) -> None:
        self._t += dt_seconds * self.speed

    def reset(self) -> None:
        self._t = 0.0


def session_duration(ranges: list[tuple[int, int, float]]) -> float:
    """Return the duration of the longest (start, end, fps) range, in seconds."""
    if not ranges:
        return 0.0
    return max((end - start) / fps for start, end, fps in ranges)


def frame_at(t_seconds: float, start_frame: int, end_frame: int, fps: float) -> int:
    """Return the frame a video should display at common clock time t.

    Time-based mapping keeps videos with different FPS synchronized:
    frame = start_frame + round(t * fps), clamped to [start_frame, end_frame].
    """
    frame = start_frame + round(t_seconds * fps)
    return max(start_frame, min(end_frame, frame))
