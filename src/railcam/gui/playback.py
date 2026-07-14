"""Synchronized playback: mapping a common clock to per-video frames.

The mapping mirrors the render-time composition semantics: every video
starts at its own start frame at t=0 and freezes on its end frame.
"""

from __future__ import annotations


def frame_at(t_seconds: float, start_frame: int, end_frame: int, fps: float) -> int:
    """Return the frame a video should display at common clock time t.

    Time-based mapping keeps videos with different FPS synchronized:
    frame = start_frame + round(t * fps), clamped to [start_frame, end_frame].
    """
    frame = start_frame + round(t_seconds * fps)
    return max(start_frame, min(end_frame, frame))
