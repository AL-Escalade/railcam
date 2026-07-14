"""Tests for BGR frame to QImage conversion."""

from __future__ import annotations

import numpy as np

from railcam.gui.imaging import frame_to_qimage


def test_converts_dimensions_and_bgr_channel_order() -> None:
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    frame[:, :] = (255, 0, 0)  # pure blue in BGR

    image = frame_to_qimage(frame)

    assert image.width() == 64
    assert image.height() == 48
    color = image.pixelColor(10, 10)
    assert (color.red(), color.green(), color.blue()) == (0, 0, 255)


def test_image_survives_source_buffer_mutation() -> None:
    frame = np.full((8, 8, 3), (0, 255, 0), dtype=np.uint8)  # green

    image = frame_to_qimage(frame)
    frame[:, :] = 0  # mutate the source after conversion

    color = image.pixelColor(4, 4)
    assert color.green() == 255


def test_handles_non_contiguous_input() -> None:
    # Slicing with a step yields a non-contiguous array (stride != width * 3)
    frame = np.full((16, 16, 3), (0, 0, 255), dtype=np.uint8)[::2, ::2]

    image = frame_to_qimage(frame)

    assert image.width() == 8
    assert image.height() == 8
    assert image.pixelColor(2, 2).red() == 255
