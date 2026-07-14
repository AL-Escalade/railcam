"""Conversion between OpenCV BGR frames and Qt images."""

from __future__ import annotations

import numpy as np
from PySide6.QtGui import QImage


def frame_to_qimage(frame: np.ndarray) -> QImage:
    """Convert a BGR numpy frame to a QImage that owns its pixel data.

    The input array may be non-contiguous (e.g. a slice); the returned
    image is always an independent copy, safe to keep after the source
    buffer is reused or freed.
    """
    contiguous = np.ascontiguousarray(frame)
    height, width, _ = contiguous.shape
    image = QImage(
        contiguous.data,
        width,
        height,
        contiguous.strides[0],
        QImage.Format.Format_BGR888,
    )
    # Detach from the numpy buffer, which Qt does not keep alive
    return image.copy()
