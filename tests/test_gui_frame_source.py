"""Tests for the GUI frame-exact video source."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from railcam.gui.frame_source import FrameSource
from railcam.video import VideoError

FRAME_COUNT = 30
FPS = 20.0
WIDTH, HEIGHT = 64, 48


def _encode_color(index: int) -> tuple[int, int, int]:
    """Encode a frame index as a BGR color robust to lossy compression.

    Blue holds the coarse part (steps of 64) and green the fine part
    (steps of 32): both steps are far larger than codec color error.
    """
    return ((index // 8) * 64, (index % 8) * 32, 0)


def decoded_index(frame: np.ndarray) -> int:
    """Recover the frame index from the center pixel's color."""
    blue = int(frame[HEIGHT // 2, WIDTH // 2, 0])
    green = int(frame[HEIGHT // 2, WIDTH // 2, 1])
    return round(blue / 64) * 8 + round(green / 32)


@pytest.fixture(scope="module")
def test_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A short synthetic video whose frames are identifiable by color."""
    path = tmp_path_factory.mktemp("videos") / "synthetic.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    assert writer.isOpened(), "OpenCV could not create the test video"
    for i in range(FRAME_COUNT):
        frame = np.full((HEIGHT, WIDTH, 3), _encode_color(i), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def test_exposes_metadata(test_video: Path) -> None:
    source = FrameSource(test_video)

    assert source.frame_count == FRAME_COUNT
    assert source.fps == pytest.approx(FPS)


def test_reads_exact_frame_by_index(test_video: Path) -> None:
    source = FrameSource(test_video)

    assert decoded_index(source.get_frame(0)) == 0
    assert decoded_index(source.get_frame(10)) == 10
    assert decoded_index(source.get_frame(FRAME_COUNT - 1)) == FRAME_COUNT - 1


def test_backward_access_after_forward(test_video: Path) -> None:
    source = FrameSource(test_video)

    assert decoded_index(source.get_frame(20)) == 20
    assert decoded_index(source.get_frame(5)) == 5
    assert decoded_index(source.get_frame(19)) == 19


def test_repeated_access_is_consistent(test_video: Path) -> None:
    source = FrameSource(test_video)

    first = source.get_frame(7)
    again = source.get_frame(7)

    assert np.array_equal(first, again)


def test_out_of_range_raises(test_video: Path) -> None:
    source = FrameSource(test_video)

    with pytest.raises(IndexError):
        source.get_frame(FRAME_COUNT)
    with pytest.raises(IndexError):
        source.get_frame(-1)


def test_unreadable_file_raises(tmp_path: Path) -> None:
    bogus = tmp_path / "not_a_video.mp4"
    bogus.write_bytes(b"this is not a video")

    with pytest.raises(VideoError):
        FrameSource(bogus)
