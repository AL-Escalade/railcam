"""Tests for video range validation and orientation handling."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from railcam.frame_source import FrameSource
from railcam.video import (
    InvalidFrameRangeError,
    extract_frames,
    get_video_metadata,
    open_capture,
    validate_frame_range,
)

FLAT_WIDTH, FLAT_HEIGHT = 64, 32
STRIPE_WIDTH = 16


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


@pytest.fixture(scope="module")
def rotated_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A landscape video carrying a 90 degree display rotation.

    Cameras store the orientation in the container instead of rotating the
    pixels; ffmpeg is the only tool here that can write such a file.
    """
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is required to write a rotated video")

    directory = tmp_path_factory.mktemp("rotated")
    flat = directory / "flat.mp4"
    writer = cv2.VideoWriter(
        str(flat), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (FLAT_WIDTH, FLAT_HEIGHT)
    )
    assert writer.isOpened(), "OpenCV could not create the test video"
    frame = np.zeros((FLAT_HEIGHT, FLAT_WIDTH, 3), dtype=np.uint8)
    frame[:, :STRIPE_WIDTH] = 255
    for _ in range(10):
        writer.write(frame)
    writer.release()

    rotated = directory / "rotated.mov"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-display_rotation", "90"]
        + ["-i", str(flat), "-c", "copy", str(rotated)],
        check=True,
    )
    return rotated


class TestOrientation:
    """Frames must arrive upright: the pipeline assumes a rising climber."""

    def test_capture_enables_auto_orientation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        applied: list[tuple[int, float]] = []

        class FakeCapture:
            def set(self, prop: int, value: float) -> bool:
                applied.append((prop, value))
                return True

        monkeypatch.setattr(cv2, "VideoCapture", lambda _: FakeCapture())

        open_capture(Path("video.mp4"))

        assert (cv2.CAP_PROP_ORIENTATION_AUTO, 1) in applied

    def test_metadata_reports_the_displayed_dimensions(self, rotated_video: Path) -> None:
        metadata = get_video_metadata(rotated_video)

        assert (metadata.width, metadata.height) == (FLAT_HEIGHT, FLAT_WIDTH)

    def test_extracted_frames_are_upright(self, rotated_video: Path) -> None:
        _, frame = next(extract_frames(rotated_video, 0, 0))

        assert frame.shape[:2] == (FLAT_WIDTH, FLAT_HEIGHT)
        # The stripe spanned the left columns before rotation, so it must now
        # span whole rows: a shape-only check would pass on unrotated pixels
        stripe_rows = np.where(frame.max(axis=(1, 2)) > 127)[0]
        assert len(stripe_rows) == STRIPE_WIDTH
        assert frame[stripe_rows].min() > 127

    def test_frame_source_serves_upright_frames(self, rotated_video: Path) -> None:
        source = FrameSource(rotated_video)
        try:
            assert source.get_frame(0).shape[:2] == (FLAT_WIDTH, FLAT_HEIGHT)
        finally:
            source.close()
