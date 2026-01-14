"""Position processing: interpolation and smoothing."""

from __future__ import annotations

from dataclasses import dataclass

from railcam.pose import DetectionResult

DEFAULT_SMOOTHING_ALPHA = 0.3


@dataclass
class ProcessedPosition:
    """A processed pelvis position with frame information."""

    frame_num: int
    x: float  # Normalized x coordinate (0-1)
    y: float  # Normalized y coordinate (0-1)
    interpolated: bool  # True if this position was interpolated


class NoValidDetectionsError(Exception):
    """No valid detections found in the frame range."""


def interpolate_positions(detections: list[DetectionResult]) -> list[ProcessedPosition]:
    """Fill in missing positions using linear interpolation.

    For gaps at the start, uses the first valid position.
    For gaps at the end, uses the last valid position.
    For gaps in the middle, linearly interpolates between valid positions.
    """
    if not detections:
        return []

    # Find valid detections
    valid_indices: list[int] = []
    for i, det in enumerate(detections):
        if det.position is not None:
            valid_indices.append(i)

    if not valid_indices:
        raise NoValidDetectionsError(
            "No valid pelvis detections found in the specified frame range. "
            "Ensure the climber is visible in the video."
        )

    results: list[ProcessedPosition] = []

    for i, det in enumerate(detections):
        if det.position is not None:
            # Valid detection - use it directly
            results.append(
                ProcessedPosition(
                    frame_num=det.frame_num,
                    x=det.position.x,
                    y=det.position.y,
                    interpolated=False,
                )
            )
        else:
            # Need to interpolate
            # Find nearest valid detections before and after
            prev_valid_idx = None
            next_valid_idx = None

            for vi in valid_indices:
                if vi < i:
                    prev_valid_idx = vi
                elif vi > i and next_valid_idx is None:
                    next_valid_idx = vi
                    break

            if prev_valid_idx is None and next_valid_idx is not None:
                # Gap at start - use first valid position
                pos = detections[next_valid_idx].position
                assert pos is not None
                results.append(
                    ProcessedPosition(
                        frame_num=det.frame_num,
                        x=pos.x,
                        y=pos.y,
                        interpolated=True,
                    )
                )
            elif prev_valid_idx is not None and next_valid_idx is None:
                # Gap at end - use last valid position
                pos = detections[prev_valid_idx].position
                assert pos is not None
                results.append(
                    ProcessedPosition(
                        frame_num=det.frame_num,
                        x=pos.x,
                        y=pos.y,
                        interpolated=True,
                    )
                )
            elif prev_valid_idx is not None and next_valid_idx is not None:
                # Gap in middle - linear interpolation
                prev_pos = detections[prev_valid_idx].position
                next_pos = detections[next_valid_idx].position
                assert prev_pos is not None and next_pos is not None

                # Calculate interpolation factor
                t = (i - prev_valid_idx) / (next_valid_idx - prev_valid_idx)

                x = prev_pos.x + t * (next_pos.x - prev_pos.x)
                y = prev_pos.y + t * (next_pos.y - prev_pos.y)

                results.append(
                    ProcessedPosition(
                        frame_num=det.frame_num,
                        x=x,
                        y=y,
                        interpolated=True,
                    )
                )

    return results


def smooth_positions(
    positions: list[ProcessedPosition], alpha: float = DEFAULT_SMOOTHING_ALPHA
) -> list[ProcessedPosition]:
    """Apply exponential moving average smoothing to positions.

    Args:
        positions: List of positions to smooth.
        alpha: Smoothing factor (0-1). Higher = more responsive, lower = smoother.
    """
    if not positions or alpha >= 1.0:
        return positions

    smoothed: list[ProcessedPosition] = []

    # Initialize with first position
    prev_x = positions[0].x
    prev_y = positions[0].y

    for pos in positions:
        # EMA: new_value = alpha * current + (1 - alpha) * previous
        smoothed_x = alpha * pos.x + (1 - alpha) * prev_x
        smoothed_y = alpha * pos.y + (1 - alpha) * prev_y

        smoothed.append(
            ProcessedPosition(
                frame_num=pos.frame_num,
                x=smoothed_x,
                y=smoothed_y,
                interpolated=pos.interpolated,
            )
        )

        prev_x = smoothed_x
        prev_y = smoothed_y

    return smoothed
