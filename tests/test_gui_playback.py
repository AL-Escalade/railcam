"""Tests for synchronized playback time-to-frame mapping."""

from __future__ import annotations

from railcam.gui.playback import frame_at


def test_time_zero_maps_to_start_frame() -> None:
    assert frame_at(t_seconds=0.0, start_frame=100, end_frame=250, fps=30.0) == 100


def test_time_advances_by_video_fps() -> None:
    assert frame_at(t_seconds=1.0, start_frame=100, end_frame=250, fps=30.0) == 130
    assert frame_at(t_seconds=2.0, start_frame=100, end_frame=250, fps=30.0) == 160


def test_videos_with_different_fps_stay_time_synchronized() -> None:
    t = 1.0
    assert frame_at(t, start_frame=0, end_frame=500, fps=30.0) == 30
    assert frame_at(t, start_frame=0, end_frame=500, fps=25.0) == 25
    assert frame_at(t, start_frame=0, end_frame=500, fps=59.94) == 60


def test_fractional_time_rounds_to_nearest_frame() -> None:
    # 0.4 frames at 30 fps rounds down, 0.6 rounds up
    assert frame_at(t_seconds=0.4 / 30.0, start_frame=0, end_frame=100, fps=30.0) == 0
    assert frame_at(t_seconds=0.6 / 30.0, start_frame=0, end_frame=100, fps=30.0) == 1


def test_freezes_on_end_frame_when_time_exceeds_range() -> None:
    # 150 frames of range at 30 fps = 5 seconds; beyond that, freeze
    assert frame_at(t_seconds=10.0, start_frame=100, end_frame=250, fps=30.0) == 250


def test_negative_time_clamps_to_start_frame() -> None:
    assert frame_at(t_seconds=-1.0, start_frame=100, end_frame=250, fps=30.0) == 100
