"""Frame cropping logic with 5:3 aspect ratio, boundary handling, and zoom normalization."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from railcam.processing import ProcessedPosition

# Vertical 5:3 ratio means width:height = 3:5
ASPECT_WIDTH = 3
ASPECT_HEIGHT = 5

# Zoom normalization: target torso height as fraction of output height
TORSO_HEIGHT_RATIO = 1 / 6

# Zoom limits to prevent extreme values
MIN_ZOOM_FACTOR = 0.5  # Allow zoom out (larger crop) if video dimensions permit
MAX_ZOOM_FACTOR = 3.0


@dataclass
class CropRegion:
    """Defines a crop region in pixel coordinates."""

    x: int  # Top-left x coordinate
    y: int  # Top-left y coordinate
    width: int
    height: int


def calculate_crop_dimensions(video_width: int, video_height: int) -> tuple[int, int]:
    """Calculate crop dimensions that fit within video while maintaining 5:3 vertical ratio.

    Returns (crop_width, crop_height) that maximizes usable area.
    """
    # Target ratio: width/height = 3/5 = 0.6
    target_ratio = ASPECT_WIDTH / ASPECT_HEIGHT

    # Current video ratio
    video_ratio = video_width / video_height

    if video_ratio > target_ratio:
        # Video is wider than target - height is the limiting factor
        crop_height = video_height
        crop_width = int(crop_height * target_ratio)
    else:
        # Video is taller than target - width is the limiting factor
        crop_width = video_width
        crop_height = int(crop_width / target_ratio)

    # Ensure dimensions are even (helps with video encoding)
    crop_width = crop_width - (crop_width % 2)
    crop_height = crop_height - (crop_height % 2)

    return crop_width, crop_height


def calculate_crop_region(
    position: ProcessedPosition,
    video_width: int,
    video_height: int,
    crop_width: int,
    crop_height: int,
) -> CropRegion:
    """Calculate crop region centered on pelvis position, clamped to video boundaries.

    The region is centered on the pelvis when possible. When the pelvis is near
    an edge, the region is shifted to stay within bounds (no black borders).
    """
    # Convert normalized position to pixel coordinates
    center_x = int(position.x * video_width)
    center_y = int(position.y * video_height)

    # Calculate ideal top-left corner (centered on pelvis)
    ideal_x = center_x - crop_width // 2
    ideal_y = center_y - crop_height // 2

    # Clamp to video boundaries
    x = max(0, min(ideal_x, video_width - crop_width))
    y = max(0, min(ideal_y, video_height - crop_height))

    return CropRegion(x=x, y=y, width=crop_width, height=crop_height)


def crop_frame(frame: np.ndarray, region: CropRegion) -> np.ndarray:
    """Extract the crop region from a frame."""
    return frame[region.y : region.y + region.height, region.x : region.x + region.width]


def crop_frame_with_padding(
    frame: np.ndarray,
    region: CropRegion,
    target_width: int,
    target_height: int,
) -> np.ndarray:
    """Extract crop region and add black padding to reach target dimensions.

    Used when "zooming out" beyond video bounds. The crop is centered
    in the padded output.

    Args:
        frame: Source frame.
        region: Crop region within the frame.
        target_width: Desired output width (may be larger than region).
        target_height: Desired output height (may be larger than region).

    Returns:
        Cropped frame with black padding if needed.
    """
    # First extract the crop region
    cropped = frame[region.y : region.y + region.height, region.x : region.x + region.width]

    # If no padding needed, return as-is
    if target_width <= region.width and target_height <= region.height:
        return cropped

    # Create black canvas at target size
    if len(frame.shape) == 3:
        padded = np.zeros((target_height, target_width, frame.shape[2]), dtype=frame.dtype)
    else:
        padded = np.zeros((target_height, target_width), dtype=frame.dtype)

    # Center the crop in the padded canvas
    offset_x = (target_width - region.width) // 2
    offset_y = (target_height - region.height) // 2

    padded[offset_y:offset_y + region.height, offset_x:offset_x + region.width] = cropped

    return padded


def scale_frame(frame: np.ndarray, target_width: int | None, target_height: int | None) -> np.ndarray:
    """Scale a frame to the target dimensions.

    If only one dimension is specified, the other is calculated to maintain aspect ratio.
    If neither is specified, returns the original frame.
    """
    if target_width is None and target_height is None:
        return frame

    height, width = frame.shape[:2]
    aspect_ratio = width / height

    if target_width is not None and target_height is None:
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    elif target_height is not None and target_width is None:
        new_height = target_height
        new_width = int(target_height * aspect_ratio)
    else:
        # Both specified - use as given (caller's responsibility to maintain ratio)
        new_width = target_width  # type: ignore
        new_height = target_height  # type: ignore

    # Ensure even dimensions
    new_width = new_width - (new_width % 2)
    new_height = new_height - (new_height % 2)

    return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)


def calculate_zoom_factor(
    avg_torso_height_normalized: float,
    target_ratio: float = TORSO_HEIGHT_RATIO,
) -> float:
    """Calculate the zoom factor to normalize torso height.

    Args:
        avg_torso_height_normalized: Average torso height as a fraction of frame height (0-1).
        target_ratio: Target torso height as a fraction of output height.

    Returns:
        Zoom factor clamped between MIN_ZOOM_FACTOR and MAX_ZOOM_FACTOR.
    """
    if avg_torso_height_normalized <= 0:
        return MIN_ZOOM_FACTOR

    # zoom_factor * avg_torso = target_ratio
    # So if avg_torso is smaller than target, we need zoom > 1
    zoom_factor = target_ratio / avg_torso_height_normalized

    # Clamp to limits
    return max(MIN_ZOOM_FACTOR, min(zoom_factor, MAX_ZOOM_FACTOR))


def calculate_zoomed_crop_dimensions(
    video_width: int,
    video_height: int,
    zoom_factor: float,
) -> tuple[int, int]:
    """Calculate crop dimensions with zoom applied, maintaining aspect ratio.

    A higher zoom factor means we crop a smaller region (more magnified).
    When zooming out (factor < 1), the crop is limited by video bounds
    while maintaining the 3:5 aspect ratio.

    Args:
        video_width: Original video width.
        video_height: Original video height.
        zoom_factor: Zoom multiplier (1.0 = no zoom, 2.0 = 2x magnification).

    Returns:
        (crop_width, crop_height) for the zoomed crop region.
    """
    # Target aspect ratio
    target_ratio = ASPECT_WIDTH / ASPECT_HEIGHT  # 3/5 = 0.6

    # Calculate base crop dimensions
    base_width, base_height = calculate_crop_dimensions(video_width, video_height)

    # Apply zoom: larger zoom = smaller crop, smaller zoom = larger crop
    ideal_width = base_width / zoom_factor
    ideal_height = base_height / zoom_factor

    # Check if we exceed video bounds and need to clamp
    if ideal_width > video_width or ideal_height > video_height:
        # Recalculate to fit within video while maintaining aspect ratio
        # Use the same logic as calculate_crop_dimensions but with full video
        video_ratio = video_width / video_height

        if video_ratio > target_ratio:
            # Video is wider than target - height is limiting
            zoomed_height = video_height
            zoomed_width = int(zoomed_height * target_ratio)
        else:
            # Video is taller than target - width is limiting
            zoomed_width = video_width
            zoomed_height = int(zoomed_width / target_ratio)
    else:
        zoomed_width = int(ideal_width)
        zoomed_height = int(ideal_height)

    # Ensure dimensions are even
    zoomed_width = zoomed_width - (zoomed_width % 2)
    zoomed_height = zoomed_height - (zoomed_height % 2)

    return zoomed_width, zoomed_height


def calculate_average_torso_height(torso_heights: list[float]) -> float:
    """Calculate the average torso height from a list of measurements.

    Args:
        torso_heights: List of normalized torso heights (0-1).

    Returns:
        Average torso height, or 0 if the list is empty.
    """
    valid_heights = [h for h in torso_heights if h > 0]
    if not valid_heights:
        return 0.0
    return sum(valid_heights) / len(valid_heights)


def calculate_effective_torso_ratio(
    avg_torso_height: float,
    video_width: int,
    video_height: int,
    zoom_factor: float,
) -> float:
    """Calculate the actual torso ratio in the output after zoom and clamping.

    Args:
        avg_torso_height: Average torso height normalized to source frame (0-1).
        video_width: Source video width.
        video_height: Source video height.
        zoom_factor: Calculated zoom factor.

    Returns:
        Effective torso ratio in the output (0-1).
    """
    if avg_torso_height <= 0:
        return 0.0

    # Get base crop dimensions
    base_width, base_height = calculate_crop_dimensions(video_width, video_height)

    # Calculate zoomed dimensions (may be clamped)
    zoomed_width, zoomed_height = calculate_zoomed_crop_dimensions(
        video_width, video_height, zoom_factor
    )

    # The torso height in pixels (approximate, based on source height)
    torso_px = avg_torso_height * video_height

    # Convert to crop coordinates: torso might span a portion of the crop
    # The crop is zoomed_height tall, and the torso is still torso_px
    # But we need to account for aspect ratio difference between source and crop
    # The torso spans vertically, so we use height ratio
    crop_to_source_ratio = zoomed_height / video_height

    # Effective torso ratio in the output
    effective_ratio = avg_torso_height / crop_to_source_ratio

    return effective_ratio
