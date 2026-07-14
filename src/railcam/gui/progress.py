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
