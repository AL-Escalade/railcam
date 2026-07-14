"""Frame-exact video access with caching, for GUI scrubbing and playback.

Reuses railcam.video for metadata and error types. The cache keeps recently
displayed frames so scrubbing backward within a short window is instant
without re-seeking long-GOP codecs.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

from railcam.video import VideoError, VideoMetadata, get_video_metadata

# Full-resolution BGR frames are heavy (~6 MB at 1080p): keep the cache small.
DEFAULT_CACHE_SIZE = 32
# Decoding forward a few frames is cheaper than seeking on long-GOP codecs.
SEQUENTIAL_WINDOW = 25


class FrameSource:
    """Random access to a video's frames by exact frame index.

    Returned frames are BGR numpy arrays and must be treated as read-only:
    they may be served from the internal cache.
    """

    def __init__(self, path: Path, cache_size: int = DEFAULT_CACHE_SIZE) -> None:
        self._metadata = get_video_metadata(path)
        if self._metadata.total_frames <= 0:
            raise VideoError(f"Video has no readable frames: {path}")

        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            raise VideoError(f"Failed to open video: {path}")

        self._next_pos = 0
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._cache_size = cache_size

    @property
    def metadata(self) -> VideoMetadata:
        return self._metadata

    @property
    def path(self) -> Path:
        return self._metadata.path

    @property
    def fps(self) -> float:
        return self._metadata.fps

    @property
    def frame_count(self) -> int:
        return self._metadata.total_frames

    def get_frame(self, index: int) -> np.ndarray:
        """Return the exact frame at index, using the cache when possible.

        Raises:
            IndexError: If index is outside [0, frame_count).
            VideoError: If decoding fails.
        """
        if not 0 <= index < self.frame_count:
            raise IndexError(f"Frame {index} out of range [0, {self.frame_count})")

        cached = self._cache.get(index)
        if cached is not None:
            self._cache.move_to_end(index)
            return cached

        frame = self._read_frame(index)
        self._cache[index] = frame
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return frame

    def close(self) -> None:
        """Release the underlying video capture and drop cached frames."""
        self._cap.release()
        self._cache.clear()

    def _read_frame(self, index: int) -> np.ndarray:
        """Position the decoder on index and decode that frame."""
        if index != self._next_pos:
            if self._next_pos < index < self._next_pos + SEQUENTIAL_WINDOW:
                while self._next_pos < index:
                    if not self._cap.grab():
                        raise VideoError(f"Failed to read frame {index} from {self.path}")
                    self._next_pos += 1
            else:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                self._next_pos = index

        ok, frame = self._cap.read()
        if not ok:
            raise VideoError(f"Failed to read frame {index} from {self.path}")
        self._next_pos = index + 1
        return frame
