"""Tests for parsing railcam CLI progress output."""

from __future__ import annotations

import pytest

from railcam.gui.progress import (
    GlobalProgress,
    expected_stage_count,
    parse_progress,
    split_output_chunks,
)


def test_parses_detecting_progress_line() -> None:
    line = "  Detecting: [████████████░░░░░░░░░░░░░░░░░░] 45.5% (100/220)"

    assert parse_progress(line) == ("Detecting", 100, 220)


def test_parses_processing_progress_line() -> None:
    line = "  Processing: [██████████████████████████████] 100.0% (150/150)"

    assert parse_progress(line) == ("Processing", 150, 150)


def test_non_progress_lines_return_none() -> None:
    assert parse_progress("Analyzing: video.mp4") is None
    assert parse_progress("  Resolution: 1920x1080") is None
    assert parse_progress("") is None


def test_parses_writing_frames_progress_line() -> None:
    line = "  Writing frames: [███░░░] 45.5% (10/22)"

    assert parse_progress(line) == ("Writing frames", 10, 22)


def test_expected_stage_count_includes_output_stages() -> None:
    # Per video: Detecting + Processing; then Writing frames + Encoding
    assert expected_stage_count(1) == 4
    assert expected_stage_count(2) == 6


def test_full_render_sequence_reaches_one() -> None:
    tracker = GlobalProgress(total_stages=expected_stage_count(1))

    tracker.update("Detecting", 100, 100)
    tracker.update("Processing", 100, 100)
    assert tracker.update("Writing frames", 50, 100) < 1.0
    tracker.update("Writing frames", 100, 100)
    tracker.update("Encoding MP4", 100, 100)

    assert tracker.update("Complete", 100, 100) == pytest.approx(1.0)


def test_global_progress_single_video_two_stages() -> None:
    # One video: Detecting then Processing → each stage is half the total
    tracker = GlobalProgress(total_stages=2)

    assert tracker.update("Detecting", 50, 100) == pytest.approx(0.25)
    assert tracker.update("Detecting", 100, 100) == pytest.approx(0.5)
    assert tracker.update("Processing", 50, 100) == pytest.approx(0.75)
    assert tracker.update("Processing", 100, 100) == pytest.approx(1.0)


def test_global_progress_two_videos_advance_across_stage_repeats() -> None:
    # Two videos: Detecting/Processing twice; same stage name reappearing
    # with a lower counter means a new stage started
    tracker = GlobalProgress(total_stages=4)

    tracker.update("Detecting", 100, 100)  # video 1 detect done
    tracker.update("Processing", 100, 100)  # video 1 process done
    overall = tracker.update("Detecting", 50, 200)  # video 2 detect begins

    assert overall == pytest.approx(2 / 4 + (50 / 200) / 4)


def test_global_progress_never_exceeds_one() -> None:
    tracker = GlobalProgress(total_stages=2)

    tracker.update("Detecting", 100, 100)
    tracker.update("Processing", 100, 100)
    # An unexpected extra stage must not push the ratio beyond 100%
    assert tracker.update("Encoding", 50, 100) <= 1.0


def test_global_progress_handles_zero_total() -> None:
    tracker = GlobalProgress(total_stages=2)

    assert tracker.update("Detecting", 0, 0) == 0.0


def test_split_output_chunks_handles_carriage_returns() -> None:
    # The CLI redraws its bar with \r and ends stages with \n
    chunk = "  Detecting: [░] 1.0% (2/220)\r  Detecting: [█] 50.0% (110/220)\r\n  done\n"

    lines = split_output_chunks(chunk)

    assert "  Detecting: [░] 1.0% (2/220)" in lines
    assert "  Detecting: [█] 50.0% (110/220)" in lines
    assert "  done" in lines
