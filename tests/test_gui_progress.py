"""Tests for parsing railcam CLI progress output."""

from __future__ import annotations

from railcam.gui.progress import parse_progress, split_output_chunks


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


def test_split_output_chunks_handles_carriage_returns() -> None:
    # The CLI redraws its bar with \r and ends stages with \n
    chunk = "  Detecting: [░] 1.0% (2/220)\r  Detecting: [█] 50.0% (110/220)\r\n  done\n"

    lines = split_output_chunks(chunk)

    assert "  Detecting: [░] 1.0% (2/220)" in lines
    assert "  Detecting: [█] 50.0% (110/220)" in lines
    assert "  done" in lines
