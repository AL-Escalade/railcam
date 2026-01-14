"""Multi-video input handling and parsing."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from railcam.pose import ClimberSelector


class InputParseError(Exception):
    """Error parsing input specification."""


@dataclass
class VideoInput:
    """Specification for a single video input."""

    path: Path
    start_frame: int
    end_frame: int
    climber_selector: ClimberSelector = ClimberSelector.AUTO

    def __post_init__(self) -> None:
        """Validate the input specification."""
        if self.start_frame < 0:
            raise InputParseError(f"Start frame must be >= 0, got {self.start_frame}")
        if self.end_frame <= self.start_frame:
            raise InputParseError(
                f"End frame ({self.end_frame}) must be > start frame ({self.start_frame})"
            )


def parse_climber_selector(selector_str: str) -> ClimberSelector:
    """Parse a climber selector string.

    Args:
        selector_str: Selector string ('left' or 'right').

    Returns:
        ClimberSelector enum value.

    Raises:
        InputParseError: If the selector is invalid.
    """
    selector_lower = selector_str.lower()
    if selector_lower == "left":
        return ClimberSelector.LEFT
    elif selector_lower == "right":
        return ClimberSelector.RIGHT
    else:
        raise InputParseError(
            f"Invalid climber selector: '{selector_str}'. Valid values are 'left' or 'right'."
        )


def parse_input_spec(spec: str) -> VideoInput:
    """Parse an input specification string.

    Format: path:start_frame:end_frame[:climber]
    Examples:
        video.mp4:100:250
        video.mp4:100:250:left
        video.mp4:100:250:right

    Args:
        spec: Input specification string.

    Returns:
        VideoInput with parsed values.

    Raises:
        InputParseError: If the specification is invalid.
    """
    # Match pattern: path:start:end or path:start:end:climber
    # Path can contain colons on Windows (C:\...), so we match from the end
    match_with_climber = re.match(r"^(.+):(\d+):(\d+):(left|right)$", spec, re.IGNORECASE)
    match_without_climber = re.match(r"^(.+):(\d+):(\d+)$", spec)

    if match_with_climber:
        path_str, start_str, end_str, climber_str = match_with_climber.groups()
        climber_selector = parse_climber_selector(climber_str)
    elif match_without_climber:
        path_str, start_str, end_str = match_without_climber.groups()
        climber_selector = ClimberSelector.AUTO
    else:
        raise InputParseError(
            f"Invalid input format: '{spec}'. "
            f"Expected format: path:start_frame:end_frame[:left|right] "
            f"(e.g., video.mp4:100:250 or video.mp4:100:250:left)"
        )

    try:
        start_frame = int(start_str)
        end_frame = int(end_str)
    except ValueError as e:
        raise InputParseError(f"Invalid frame number in '{spec}': {e}") from e

    path = Path(path_str)

    return VideoInput(
        path=path,
        start_frame=start_frame,
        end_frame=end_frame,
        climber_selector=climber_selector,
    )


def parse_multiple_inputs(specs: list[str]) -> list[VideoInput]:
    """Parse multiple input specifications.

    Args:
        specs: List of input specification strings.

    Returns:
        List of VideoInput objects.

    Raises:
        InputParseError: If any specification is invalid.
    """
    if not specs:
        raise InputParseError("At least one input is required")

    return [parse_input_spec(spec) for spec in specs]
