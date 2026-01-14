"""Tests for multi_video module."""

from pathlib import Path

import pytest

from railcam.multi_video import (
    InputParseError,
    VideoInput,
    parse_climber_selector,
    parse_input_spec,
    parse_multiple_inputs,
)
from railcam.pose import ClimberSelector


class TestVideoInput:
    def test_valid_input(self):
        vi = VideoInput(path=Path("video.mp4"), start_frame=100, end_frame=250)
        assert vi.path == Path("video.mp4")
        assert vi.start_frame == 100
        assert vi.end_frame == 250

    def test_default_climber_selector_is_auto(self):
        vi = VideoInput(path=Path("video.mp4"), start_frame=100, end_frame=250)
        assert vi.climber_selector == ClimberSelector.AUTO

    def test_with_climber_selector_left(self):
        vi = VideoInput(
            path=Path("video.mp4"),
            start_frame=100,
            end_frame=250,
            climber_selector=ClimberSelector.LEFT,
        )
        assert vi.climber_selector == ClimberSelector.LEFT

    def test_with_climber_selector_right(self):
        vi = VideoInput(
            path=Path("video.mp4"),
            start_frame=100,
            end_frame=250,
            climber_selector=ClimberSelector.RIGHT,
        )
        assert vi.climber_selector == ClimberSelector.RIGHT

    def test_negative_start_frame_raises(self):
        with pytest.raises(InputParseError, match="Start frame must be >= 0"):
            VideoInput(path=Path("video.mp4"), start_frame=-1, end_frame=100)

    def test_end_before_start_raises(self):
        with pytest.raises(InputParseError, match="End frame.*must be greater"):
            VideoInput(path=Path("video.mp4"), start_frame=100, end_frame=50)

    def test_end_equals_start_raises(self):
        with pytest.raises(InputParseError, match="End frame.*must be greater"):
            VideoInput(path=Path("video.mp4"), start_frame=100, end_frame=100)


class TestParseClimberSelector:
    def test_left_lowercase(self):
        assert parse_climber_selector("left") == ClimberSelector.LEFT

    def test_left_uppercase(self):
        assert parse_climber_selector("LEFT") == ClimberSelector.LEFT

    def test_left_mixed_case(self):
        assert parse_climber_selector("Left") == ClimberSelector.LEFT

    def test_right_lowercase(self):
        assert parse_climber_selector("right") == ClimberSelector.RIGHT

    def test_right_uppercase(self):
        assert parse_climber_selector("RIGHT") == ClimberSelector.RIGHT

    def test_invalid_selector_raises(self):
        with pytest.raises(InputParseError, match="Invalid climber selector"):
            parse_climber_selector("center")

    def test_auto_raises(self):
        # "auto" is not a valid user input, only "left" or "right"
        with pytest.raises(InputParseError, match="Invalid climber selector"):
            parse_climber_selector("auto")


class TestParseInputSpec:
    def test_valid_spec(self):
        vi = parse_input_spec("video.mp4:100:250")
        assert vi.path == Path("video.mp4")
        assert vi.start_frame == 100
        assert vi.end_frame == 250

    def test_default_selector_is_auto(self):
        vi = parse_input_spec("video.mp4:100:250")
        assert vi.climber_selector == ClimberSelector.AUTO

    def test_path_with_directory(self):
        vi = parse_input_spec("/path/to/video.mp4:50:200")
        assert vi.path == Path("/path/to/video.mp4")
        assert vi.start_frame == 50
        assert vi.end_frame == 200

    def test_relative_path(self):
        vi = parse_input_spec("../videos/climb.mov:0:100")
        assert vi.path == Path("../videos/climb.mov")
        assert vi.start_frame == 0
        assert vi.end_frame == 100

    def test_with_left_selector(self):
        vi = parse_input_spec("video.mp4:100:250:left")
        assert vi.path == Path("video.mp4")
        assert vi.start_frame == 100
        assert vi.end_frame == 250
        assert vi.climber_selector == ClimberSelector.LEFT

    def test_with_right_selector(self):
        vi = parse_input_spec("video.mp4:100:250:right")
        assert vi.climber_selector == ClimberSelector.RIGHT

    def test_with_selector_uppercase(self):
        vi = parse_input_spec("video.mp4:100:250:LEFT")
        assert vi.climber_selector == ClimberSelector.LEFT

    def test_with_selector_mixed_case(self):
        vi = parse_input_spec("video.mp4:100:250:Right")
        assert vi.climber_selector == ClimberSelector.RIGHT

    def test_path_with_directory_and_selector(self):
        vi = parse_input_spec("/path/to/video.mp4:50:200:left")
        assert vi.path == Path("/path/to/video.mp4")
        assert vi.start_frame == 50
        assert vi.end_frame == 200
        assert vi.climber_selector == ClimberSelector.LEFT

    def test_missing_frame_range_raises(self):
        with pytest.raises(InputParseError, match="Invalid input format"):
            parse_input_spec("video.mp4")

    def test_missing_end_frame_raises(self):
        with pytest.raises(InputParseError, match="Invalid input format"):
            parse_input_spec("video.mp4:100")

    def test_non_numeric_frame_raises(self):
        with pytest.raises(InputParseError, match="Invalid input format"):
            parse_input_spec("video.mp4:abc:250")

    def test_empty_path_raises(self):
        with pytest.raises(InputParseError, match="Invalid input format"):
            parse_input_spec(":100:250")

    def test_invalid_frame_order_raises(self):
        with pytest.raises(InputParseError, match="End frame.*must be greater"):
            parse_input_spec("video.mp4:250:100")

    def test_invalid_selector_raises(self):
        with pytest.raises(InputParseError, match="Invalid input format"):
            parse_input_spec("video.mp4:100:250:center")


class TestParseMultipleInputs:
    def test_single_input(self):
        inputs = parse_multiple_inputs(["video.mp4:100:250"])
        assert len(inputs) == 1
        assert inputs[0].path == Path("video.mp4")

    def test_multiple_inputs(self):
        inputs = parse_multiple_inputs(
            [
                "video1.mp4:100:250",
                "video2.mp4:50:200",
                "video3.mp4:0:150",
            ]
        )
        assert len(inputs) == 3
        assert inputs[0].path == Path("video1.mp4")
        assert inputs[1].path == Path("video2.mp4")
        assert inputs[2].path == Path("video3.mp4")

    def test_empty_list_raises(self):
        with pytest.raises(InputParseError, match="At least one input is required"):
            parse_multiple_inputs([])

    def test_invalid_spec_in_list_raises(self):
        with pytest.raises(InputParseError):
            parse_multiple_inputs(["video1.mp4:100:250", "invalid"])
