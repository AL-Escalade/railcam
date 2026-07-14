"""Parsing of the railcam CLI progress output (see cli.print_progress)."""

from __future__ import annotations

import re

# Matches "  Stage: [██░░] 45.5% (100/220)" — tolerant to bar content
_PROGRESS_RE = re.compile(r"(\w[\w ]*?):\s*\[[^\]]*\]\s*[\d.]+%\s*\((\d+)/(\d+)\)")


def parse_progress(line: str) -> tuple[str, int, int] | None:
    """Extract (stage, current, total) from a CLI progress line, if any."""
    match = _PROGRESS_RE.search(line)
    if match is None:
        return None
    stage, current, total = match.groups()
    return stage.strip(), int(current), int(total)


def split_output_chunks(chunk: str) -> list[str]:
    """Split process output on both newlines and the \\r bar redraws."""
    return [line for line in re.split(r"[\r\n]+", chunk) if line.strip()]


def expected_stage_count(video_count: int) -> int:
    """Number of CLI progress-bar stages for a render.

    Per video: Detecting and Processing; then, once for the output:
    Writing frames and Encoding (see cli.main and output.generate_output).
    """
    return 2 * video_count + 2


class GlobalProgress:
    """Aggregates per-stage CLI progress into one overall ratio (0..1).

    The CLI runs a known number of bar stages (Detecting and Processing per
    video). A stage transition is a new stage name, or the same name whose
    counter went backwards (the next video started).
    """

    def __init__(self, total_stages: int) -> None:
        self._total_stages = max(total_stages, 1)
        self._completed = 0
        self._active_stage: str | None = None
        self._last_current = 0

    def update(self, stage: str, current: int, total: int) -> float:
        """Record a progress line and return the overall completion ratio."""
        is_new_stage = self._active_stage is not None and (
            stage != self._active_stage or current < self._last_current
        )
        if is_new_stage:
            self._completed = min(self._completed + 1, self._total_stages - 1)
        self._active_stage = stage
        self._last_current = current

        stage_fraction = current / total if total else 0.0
        return min((self._completed + stage_fraction) / self._total_stages, 1.0)
