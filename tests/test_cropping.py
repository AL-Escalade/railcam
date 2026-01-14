"""Tests for cropping module."""

import numpy as np

from railcam.cropping import (
    MAX_ZOOM_FACTOR,
    MIN_ZOOM_FACTOR,
    TORSO_HEIGHT_RATIO,
    calculate_average_torso_height,
    calculate_crop_dimensions,
    calculate_crop_region,
    calculate_zoom_factor,
    calculate_zoomed_crop_dimensions,
    crop_frame,
)
from railcam.processing import ProcessedPosition


class TestCalculateCropDimensions:
    def test_wider_video_uses_full_height(self):
        # 1920x1080 (16:9) video -> height limited
        width, height = calculate_crop_dimensions(1920, 1080)

        # Height should be 1080 (full height)
        assert height == 1080
        # Width should be height * 3/5 = 648
        assert width == 648
        # Check ratio
        assert abs(width / height - 3 / 5) < 0.01

    def test_taller_video_uses_full_width(self):
        # 1080x1920 (9:16) video -> width limited
        width, height = calculate_crop_dimensions(1080, 1920)

        # Width should be 1080 (full width)
        assert width == 1080
        # Height should be width * 5/3 = 1800
        assert height == 1800
        # Check ratio
        assert abs(width / height - 3 / 5) < 0.01

    def test_exact_ratio_video(self):
        # Video already at 3:5 ratio
        width, height = calculate_crop_dimensions(900, 1500)

        assert width == 900
        assert height == 1500

    def test_dimensions_are_even(self):
        # Odd input dimensions
        width, height = calculate_crop_dimensions(1921, 1081)

        assert width % 2 == 0
        assert height % 2 == 0


class TestCalculateCropRegion:
    def test_centered_pelvis(self):
        # Pelvis at center of 1920x1080 video
        # With 5:3 ratio: crop is 648x1080
        position = ProcessedPosition(0, 0.5, 0.5, False)
        region = calculate_crop_region(position, 1920, 1080, 648, 1080)

        # Should be centered horizontally
        expected_x = (1920 - 648) // 2
        assert region.x == expected_x
        assert region.y == 0
        assert region.width == 648
        assert region.height == 1080

    def test_pelvis_near_left_edge(self):
        # Pelvis very close to left edge
        position = ProcessedPosition(0, 0.1, 0.5, False)
        region = calculate_crop_region(position, 1920, 1080, 648, 1080)

        # Should be clamped to left edge
        assert region.x == 0
        assert region.width == 648

    def test_pelvis_near_right_edge(self):
        # Pelvis very close to right edge
        position = ProcessedPosition(0, 0.95, 0.5, False)
        region = calculate_crop_region(position, 1920, 1080, 648, 1080)

        # Should be clamped to right edge
        assert region.x == 1920 - 648
        assert region.width == 648

    def test_pelvis_near_top_edge(self):
        # Pelvis very close to top edge (in a tall video)
        # With 5:3 ratio: 1080x1920 -> crop is 1080x1800
        position = ProcessedPosition(0, 0.5, 0.05, False)
        region = calculate_crop_region(position, 1080, 1920, 1080, 1800)

        # Should be clamped to top edge
        assert region.y == 0
        assert region.height == 1800

    def test_pelvis_near_bottom_edge(self):
        # Pelvis very close to bottom edge
        position = ProcessedPosition(0, 0.5, 0.95, False)
        region = calculate_crop_region(position, 1080, 1920, 1080, 1800)

        # Should be clamped to bottom edge
        assert region.y == 1920 - 1800
        assert region.height == 1800

    def test_crop_region_always_within_bounds(self):
        # Test various positions to ensure crop is always within bounds
        positions = [
            (0.0, 0.0),  # Top-left corner
            (1.0, 0.0),  # Top-right corner
            (0.0, 1.0),  # Bottom-left corner
            (1.0, 1.0),  # Bottom-right corner
            (0.5, 0.5),  # Center
        ]

        for x, y in positions:
            position = ProcessedPosition(0, x, y, False)
            region = calculate_crop_region(position, 1920, 1080, 648, 1080)

            assert region.x >= 0
            assert region.y >= 0
            assert region.x + region.width <= 1920
            assert region.y + region.height <= 1080


class TestCropFrame:
    def test_crop_extracts_correct_region(self):
        # Create a simple test frame
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        # Mark a specific region
        frame[25:75, 50:150, 0] = 255  # Red region

        from railcam.cropping import CropRegion

        region = CropRegion(x=50, y=25, width=100, height=50)
        cropped = crop_frame(frame, region)

        assert cropped.shape == (50, 100, 3)
        # Should all be red
        assert np.all(cropped[:, :, 0] == 255)


class TestCalculateZoomFactor:
    def test_torso_at_target_returns_one(self):
        # If torso is already at target ratio, zoom should be 1.0
        zoom = calculate_zoom_factor(TORSO_HEIGHT_RATIO)
        assert abs(zoom - 1.0) < 0.01

    def test_small_torso_returns_zoom_in(self):
        # If torso is smaller than target, we need to zoom in (factor > 1)
        # Torso at half the target ratio -> zoom = 2.0
        zoom = calculate_zoom_factor(TORSO_HEIGHT_RATIO / 2)
        assert abs(zoom - 2.0) < 0.01

    def test_large_torso_clamps_to_min(self):
        # If torso is much larger than target, zoom should be clamped to MIN_ZOOM_FACTOR
        # With TORSO_HEIGHT_RATIO * 10, zoom would be 0.1, clamped to 0.5
        zoom = calculate_zoom_factor(TORSO_HEIGHT_RATIO * 10)
        assert zoom == MIN_ZOOM_FACTOR

    def test_very_small_torso_clamps_to_max(self):
        # If torso is very small, zoom should be clamped to MAX_ZOOM_FACTOR
        zoom = calculate_zoom_factor(TORSO_HEIGHT_RATIO / 10)
        assert zoom == MAX_ZOOM_FACTOR

    def test_zero_torso_returns_min(self):
        # Edge case: zero torso height
        zoom = calculate_zoom_factor(0.0)
        assert zoom == MIN_ZOOM_FACTOR

    def test_negative_torso_returns_min(self):
        # Edge case: negative torso height
        zoom = calculate_zoom_factor(-0.1)
        assert zoom == MIN_ZOOM_FACTOR


class TestCalculateZoomedCropDimensions:
    def test_no_zoom_same_as_base(self):
        # With zoom factor 1.0, dimensions should match calculate_crop_dimensions
        base_w, base_h = calculate_crop_dimensions(1920, 1080)
        zoomed_w, zoomed_h = calculate_zoomed_crop_dimensions(1920, 1080, 1.0)

        assert zoomed_w == base_w
        assert zoomed_h == base_h

    def test_zoom_2x_halves_dimensions(self):
        # With zoom factor 2.0, crop region should be half the base size
        base_w, base_h = calculate_crop_dimensions(1920, 1080)
        zoomed_w, zoomed_h = calculate_zoomed_crop_dimensions(1920, 1080, 2.0)

        # Allow for even-number rounding
        assert abs(zoomed_w - base_w / 2) <= 2
        assert abs(zoomed_h - base_h / 2) <= 2

    def test_zoom_maintains_aspect_ratio(self):
        # Zoomed dimensions should maintain 5:3 vertical ratio
        zoomed_w, zoomed_h = calculate_zoomed_crop_dimensions(1920, 1080, 1.5)
        assert abs(zoomed_w / zoomed_h - 3 / 5) < 0.02

    def test_dimensions_are_even(self):
        zoomed_w, zoomed_h = calculate_zoomed_crop_dimensions(1920, 1080, 1.7)
        assert zoomed_w % 2 == 0
        assert zoomed_h % 2 == 0


class TestCalculateAverageTorsoHeight:
    def test_average_of_valid_heights(self):
        heights = [0.3, 0.4, 0.5]
        avg = calculate_average_torso_height(heights)
        assert abs(avg - 0.4) < 0.001

    def test_ignores_zero_heights(self):
        heights = [0.3, 0.0, 0.5, 0.0]
        avg = calculate_average_torso_height(heights)
        assert abs(avg - 0.4) < 0.001

    def test_empty_list_returns_zero(self):
        avg = calculate_average_torso_height([])
        assert avg == 0.0

    def test_all_zeros_returns_zero(self):
        avg = calculate_average_torso_height([0.0, 0.0, 0.0])
        assert avg == 0.0
