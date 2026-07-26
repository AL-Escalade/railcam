"""Tests for pose detection types.

Climber selection is now motion-based: see tests/test_tracking.py.
"""

from __future__ import annotations

from railcam.pose import ClimberSelector


class TestClimberSelector:
    """Tests for ClimberSelector enum."""

    def test_left_value(self) -> None:
        assert ClimberSelector.LEFT.value == "left"

    def test_right_value(self) -> None:
        assert ClimberSelector.RIGHT.value == "right"

    def test_auto_value(self) -> None:
        assert ClimberSelector.AUTO.value == "auto"
