"""Video file handling and frame extraction."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


@dataclass
class VideoMetadata:
    """Metadata about a video file."""

    width: int
    height: int
    fps: float
    total_frames: int
    path: Path


class VideoError(Exception):
    """Base exception for video-related errors."""


class VideoNotFoundError(VideoError):
    """Video file not found."""


class UnsupportedFormatError(VideoError):
    """Video format not supported."""


class InvalidFrameRangeError(VideoError):
    """Frame range is invalid."""


def validate_video_path(path: Path) -> None:
    """Validate that the video file exists and has a supported format."""
    if not path.exists():
        raise VideoNotFoundError(f"Video file not found: {path}")

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise UnsupportedFormatError(
            f"Unsupported video format: {path.suffix}. Supported formats: {supported}"
        )


def get_video_metadata(path: Path) -> VideoMetadata:
    """Extract metadata from a video file."""
    validate_video_path(path)

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise VideoError(f"Failed to open video: {path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        return VideoMetadata(
            width=width,
            height=height,
            fps=fps,
            total_frames=total_frames,
            path=path,
        )
    finally:
        cap.release()


def validate_frame_range(start: int, end: int, total_frames: int) -> None:
    """Validate that the frame range is valid."""
    if start < 0:
        raise InvalidFrameRangeError(f"Start frame must be >= 0, got {start}")

    if end <= start:
        raise InvalidFrameRangeError(
            f"End frame ({end}) must be greater than start frame ({start})"
        )

    if end > total_frames:
        raise InvalidFrameRangeError(
            f"End frame ({end}) exceeds video length ({total_frames} frames)"
        )


def extract_frames(
    path: Path, start_frame: int, end_frame: int
) -> Iterator[tuple[int, np.ndarray]]:
    """Extract frames from a video file within the specified range.

    Yields tuples of (frame_number, frame_data) for each frame in the range.
    Frame numbers are 0-indexed and the range is inclusive of both start and end.
    """
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise VideoError(f"Failed to open video: {path}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for frame_num in range(start_frame, end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                break
            yield frame_num, frame
    finally:
        cap.release()
