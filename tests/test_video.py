"""Tests for video range validation."""

from __future__ import annotations

import pytest

from railcam.video import InvalidFrameRangeError, validate_frame_range


class TestValidateFrameRange:
    """Frame numbers are 0-indexed and the range is inclusive of both ends."""

    def test_last_frame_is_valid(self) -> None:
        validate_frame_range(0, 99, total_frames=100)

    def test_one_past_the_last_frame_is_rejected(self) -> None:
        # Frame 100 does not exist in a 100-frame video; accepting it made the
        # pipeline claim one more frame than it could ever decode
        with pytest.raises(InvalidFrameRangeError, match="last frame is 99"):
            validate_frame_range(0, 100, total_frames=100)

    def test_well_past_the_end_is_rejected(self) -> None:
        with pytest.raises(InvalidFrameRangeError, match="last frame is 99"):
            validate_frame_range(0, 500, total_frames=100)

    def test_negative_start_is_rejected(self) -> None:
        with pytest.raises(InvalidFrameRangeError, match="Start frame"):
            validate_frame_range(-1, 50, total_frames=100)

    def test_end_before_start_is_rejected(self) -> None:
        with pytest.raises(InvalidFrameRangeError, match="End frame"):
            validate_frame_range(50, 20, total_frames=100)

    def test_end_equal_to_start_is_rejected(self) -> None:
        with pytest.raises(InvalidFrameRangeError, match="End frame"):
            validate_frame_range(50, 50, total_frames=100)
