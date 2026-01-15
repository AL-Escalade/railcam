"""Frame synchronization and side-by-side composition."""

from __future__ import annotations

import math

import cv2
import numpy as np


def calculate_output_fps(fps_list: list[float]) -> float:
    """Calculate optimal output FPS using LCM (Least Common Multiple).

    Using the LCM ensures that each source frame is displayed an exact integer
    number of times, avoiding judder caused by irregular frame duplication.
    This is especially important for slow-motion playback where timing
    irregularities become more visible.

    Args:
        fps_list: List of FPS values for each video.

    Returns:
        Output FPS (LCM of all input FPS values, rounded to integers).

    Raises:
        ValueError: If the list is empty or contains non-positive values.

    Example:
        >>> calculate_output_fps([60.0, 30.0, 25.0])
        300.0
        >>> calculate_output_fps([30.0, 24.0])
        120.0
    """
    if not fps_list:
        raise ValueError("At least one FPS value is required")

    # Round FPS to integers for LCM calculation
    # (handles cases like 29.97fps → 30fps)
    int_fps_list = [round(fps) for fps in fps_list]

    if any(fps <= 0 for fps in int_fps_list):
        raise ValueError("All FPS values must be positive")

    # Calculate LCM of all FPS values
    result = int_fps_list[0]
    for fps in int_fps_list[1:]:
        result = math.lcm(result, fps)

    return float(result)


def calculate_target_frame_count(frame_counts: list[int]) -> int:
    """Calculate the target frame count for synchronized output.

    Uses the maximum frame count across all videos.

    Args:
        frame_counts: List of frame counts for each video.

    Returns:
        Target frame count (maximum of all inputs).

    Raises:
        ValueError: If the list is empty.
    """
    if not frame_counts:
        raise ValueError("At least one frame count is required")
    return max(frame_counts)


def resample_frame_indices(source_count: int, target_count: int) -> list[int]:
    """Calculate which source frames to use for each target frame.

    Uses nearest-neighbor resampling (frame duplication/dropping).

    Args:
        source_count: Number of frames in the source video.
        target_count: Target number of frames.

    Returns:
        List of source frame indices, one for each target frame.
    """
    if source_count <= 0 or target_count <= 0:
        return []

    if source_count == target_count:
        return list(range(source_count))

    # Map each target index to the nearest source index
    indices = []
    for target_idx in range(target_count):
        # Calculate the corresponding position in the source
        source_pos = target_idx * (source_count - 1) / (target_count - 1) if target_count > 1 else 0
        source_idx = round(source_pos)
        source_idx = max(0, min(source_idx, source_count - 1))
        indices.append(source_idx)

    return indices


def calculate_duration(frame_count: int, fps: float) -> float:
    """Calculate duration in seconds from frame count and FPS.

    Args:
        frame_count: Number of frames.
        fps: Frames per second.

    Returns:
        Duration in seconds.
    """
    if fps <= 0:
        return 0.0
    return frame_count / fps


def calculate_max_duration(frame_counts: list[int], fps_list: list[float]) -> float:
    """Calculate the maximum duration across multiple videos.

    Args:
        frame_counts: List of frame counts for each video.
        fps_list: List of FPS values for each video.

    Returns:
        Maximum duration in seconds.

    Raises:
        ValueError: If lists have different lengths or are empty.
    """
    if not frame_counts or not fps_list:
        raise ValueError("At least one video is required")
    if len(frame_counts) != len(fps_list):
        raise ValueError("Frame counts and FPS lists must have the same length")

    durations = [calculate_duration(fc, fps) for fc, fps in zip(frame_counts, fps_list)]
    return max(durations)


def time_sync_frame_indices(
    source_count: int,
    source_fps: float,
    target_duration: float,
    output_fps: float,
) -> list[int]:
    """Calculate source frame indices for time-synchronized output.

    Maps each output frame to the corresponding source frame based on time.
    When the source video ends (in time), freezes on the last frame.

    Args:
        source_count: Number of frames in the source video.
        source_fps: FPS of the source video.
        target_duration: Target duration in seconds.
        output_fps: FPS of the output video.

    Returns:
        List of source frame indices for each output frame.
    """
    if source_count <= 0 or source_fps <= 0 or target_duration <= 0 or output_fps <= 0:
        return []

    target_frame_count = int(target_duration * output_fps)
    source_duration = source_count / source_fps

    indices = []
    for output_idx in range(target_frame_count):
        # Time of this output frame
        t = output_idx / output_fps

        if t >= source_duration:
            # Source video has ended, freeze on last frame
            indices.append(source_count - 1)
        else:
            # Map time to source frame
            source_idx = int(t * source_fps)
            source_idx = min(source_idx, source_count - 1)
            indices.append(source_idx)

    return indices


def time_sync_frames(
    frames: list[np.ndarray],
    source_fps: float,
    target_duration: float,
    output_fps: float,
) -> list[np.ndarray]:
    """Resample frames for time-synchronized output.

    Args:
        frames: List of source frames.
        source_fps: FPS of the source video.
        target_duration: Target duration in seconds.
        output_fps: FPS of the output video.

    Returns:
        List of frames synchronized to the target duration at output FPS.
    """
    if not frames:
        return []

    indices = time_sync_frame_indices(len(frames), source_fps, target_duration, output_fps)
    return [frames[idx] for idx in indices]


def extend_frame_indices(source_count: int, target_count: int) -> list[int]:
    """Calculate which source frames to use, freezing on last frame if shorter.

    Plays frames at normal speed (1:1), then repeats the last frame
    until target_count is reached.

    Args:
        source_count: Number of frames in the source video.
        target_count: Target number of frames.

    Returns:
        List of source frame indices, one for each target frame.
    """
    if source_count <= 0 or target_count <= 0:
        return []

    indices = []
    for target_idx in range(target_count):
        # Use source frame if available, otherwise freeze on last frame
        source_idx = min(target_idx, source_count - 1)
        indices.append(source_idx)

    return indices


def extend_frames(frames: list[np.ndarray], target_count: int) -> list[np.ndarray]:
    """Extend a list of frames to match a target count by freezing on last frame.

    Plays frames at normal speed, then repeats the last frame until
    target_count is reached.

    Args:
        frames: List of source frames.
        target_count: Target number of frames.

    Returns:
        List of frames with length equal to target_count.
    """
    if not frames:
        return []

    indices = extend_frame_indices(len(frames), target_count)
    return [frames[idx] for idx in indices]


def resample_frames(frames: list[np.ndarray], target_count: int) -> list[np.ndarray]:
    """Resample a list of frames to match a target count.

    Uses frame duplication/dropping (nearest-neighbor).

    Args:
        frames: List of source frames.
        target_count: Target number of frames.

    Returns:
        List of resampled frames with length equal to target_count.
    """
    if not frames:
        return []

    indices = resample_frame_indices(len(frames), target_count)
    return [frames[idx] for idx in indices]


def normalize_frame_height(frame: np.ndarray, target_height: int) -> np.ndarray:
    """Scale a frame to a target height while maintaining aspect ratio.

    Args:
        frame: Input frame.
        target_height: Target height in pixels.

    Returns:
        Scaled frame.
    """
    height, width = frame.shape[:2]
    if height == target_height:
        return frame

    scale = target_height / height
    new_width = int(width * scale)

    # Ensure even dimensions
    new_width = new_width - (new_width % 2)
    target_height = target_height - (target_height % 2)

    return cv2.resize(frame, (new_width, target_height), interpolation=cv2.INTER_LANCZOS4)


def compose_frames_horizontal(
    frame_lists: list[list[np.ndarray]],
    target_height: int | None = None,
) -> list[np.ndarray]:
    """Compose multiple frame lists side-by-side.

    All frame lists must have the same length. Frames are scaled to a common height
    and concatenated horizontally.

    Args:
        frame_lists: List of frame lists, one per video.
        target_height: Optional target height. If None, uses the maximum height.

    Returns:
        List of composed frames.

    Raises:
        ValueError: If frame lists have different lengths or are empty.
    """
    if not frame_lists:
        raise ValueError("At least one frame list is required")

    if not all(frame_lists):
        raise ValueError("All frame lists must be non-empty")

    # Verify all lists have the same length
    frame_count = len(frame_lists[0])
    if not all(len(fl) == frame_count for fl in frame_lists):
        raise ValueError("All frame lists must have the same length")

    # Determine target height
    if target_height is None:
        target_height = max(frames[0].shape[0] for frames in frame_lists if len(frames) > 0)

    # Ensure even height
    target_height = target_height - (target_height % 2)

    composed_frames = []
    for frame_idx in range(frame_count):
        # Normalize heights for this frame index
        normalized_frames = []
        for video_frames in frame_lists:
            frame = video_frames[frame_idx]
            normalized = normalize_frame_height(frame, target_height)
            normalized_frames.append(normalized)

        # Concatenate horizontally
        composed = np.concatenate(normalized_frames, axis=1)

        # Ensure even width
        if composed.shape[1] % 2 != 0:
            composed = composed[:, :-1]

        composed_frames.append(composed)

    return composed_frames


def calculate_output_dimensions(
    video_widths: list[int],
    video_heights: list[int],
    target_height: int | None = None,
) -> tuple[int, int]:
    """Calculate the dimensions of the composed output.

    Args:
        video_widths: List of video widths.
        video_heights: List of video heights.
        target_height: Optional target height. If None, uses the maximum.

    Returns:
        (total_width, height) of the composed output.
    """
    if not video_widths or not video_heights:
        raise ValueError("At least one video dimension is required")

    if len(video_widths) != len(video_heights):
        raise ValueError("Width and height lists must have the same length")

    # Determine target height
    if target_height is None:
        target_height = max(video_heights)

    # Ensure even height
    target_height = target_height - (target_height % 2)

    # Calculate scaled widths
    total_width = 0
    for w, h in zip(video_widths, video_heights):
        scale = target_height / h
        scaled_width = int(w * scale)
        scaled_width = scaled_width - (scaled_width % 2)
        total_width += scaled_width

    return total_width, target_height
