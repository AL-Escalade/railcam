"""Tests for synchronized playback time-to-frame mapping and clock."""

from __future__ import annotations

import pytest

from railcam.gui.playback import PlaybackClock, frame_at, session_duration


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


def test_clock_accumulates_time_scaled_by_speed() -> None:
    clock = PlaybackClock(speed=0.25)

    clock.advance(2.0)

    assert clock.t == pytest.approx(0.5)


def test_clock_speed_change_only_affects_subsequent_time() -> None:
    clock = PlaybackClock(speed=1.0)

    clock.advance(1.0)
    clock.speed = 0.5
    clock.advance(1.0)

    assert clock.t == pytest.approx(1.5)


def test_clock_reset_returns_to_zero() -> None:
    clock = PlaybackClock(speed=1.0)
    clock.advance(3.0)

    clock.reset()

    assert clock.t == 0.0


def test_session_duration_is_longest_video_range() -> None:
    # (start, end, fps): 150 frames at 30 fps = 5 s; 120 frames at 20 fps = 6 s
    ranges = [(100, 250, 30.0), (60, 180, 20.0)]

    assert session_duration(ranges) == pytest.approx(6.0)


def test_session_duration_empty_is_zero() -> None:
    assert session_duration([]) == 0.0
