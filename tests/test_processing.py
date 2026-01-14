"""Tests for position processing module."""

import pytest

from railcam.pose import DetectionResult, PelvisPosition
from railcam.processing import (
    NoValidDetectionsError,
    ProcessedPosition,
    interpolate_positions,
    smooth_positions,
)


class TestInterpolatePositions:
    def test_empty_list(self):
        result = interpolate_positions([])
        assert result == []

    def test_all_valid_detections(self):
        detections = [
            DetectionResult(0, PelvisPosition(0.5, 0.8, 0.9)),
            DetectionResult(1, PelvisPosition(0.5, 0.6, 0.9)),
            DetectionResult(2, PelvisPosition(0.5, 0.4, 0.9)),
        ]
        result = interpolate_positions(detections)

        assert len(result) == 3
        assert all(not p.interpolated for p in result)
        assert result[0].x == 0.5
        assert result[0].y == 0.8

    def test_no_valid_detections_raises(self):
        detections = [
            DetectionResult(0, None),
            DetectionResult(1, None),
            DetectionResult(2, None),
        ]
        with pytest.raises(NoValidDetectionsError):
            interpolate_positions(detections)

    def test_gap_at_start(self):
        detections = [
            DetectionResult(0, None),
            DetectionResult(1, None),
            DetectionResult(2, PelvisPosition(0.5, 0.5, 0.9)),
        ]
        result = interpolate_positions(detections)

        assert len(result) == 3
        # First two frames should use the first valid position
        assert result[0].x == 0.5
        assert result[0].y == 0.5
        assert result[0].interpolated is True
        assert result[1].interpolated is True
        assert result[2].interpolated is False

    def test_gap_at_end(self):
        detections = [
            DetectionResult(0, PelvisPosition(0.5, 0.5, 0.9)),
            DetectionResult(1, None),
            DetectionResult(2, None),
        ]
        result = interpolate_positions(detections)

        assert len(result) == 3
        # Last two frames should use the last valid position
        assert result[1].x == 0.5
        assert result[2].x == 0.5
        assert result[0].interpolated is False
        assert result[1].interpolated is True
        assert result[2].interpolated is True

    def test_gap_in_middle(self):
        detections = [
            DetectionResult(0, PelvisPosition(0.0, 0.0, 0.9)),
            DetectionResult(1, None),
            DetectionResult(2, None),
            DetectionResult(3, PelvisPosition(1.0, 1.0, 0.9)),
        ]
        result = interpolate_positions(detections)

        assert len(result) == 4
        # Linear interpolation for middle frames
        assert result[0].x == 0.0
        assert result[0].y == 0.0
        assert result[0].interpolated is False

        # Frame 1: t = 1/3
        assert abs(result[1].x - 1 / 3) < 0.001
        assert abs(result[1].y - 1 / 3) < 0.001
        assert result[1].interpolated is True

        # Frame 2: t = 2/3
        assert abs(result[2].x - 2 / 3) < 0.001
        assert abs(result[2].y - 2 / 3) < 0.001
        assert result[2].interpolated is True

        assert result[3].x == 1.0
        assert result[3].y == 1.0
        assert result[3].interpolated is False


class TestSmoothPositions:
    def test_empty_list(self):
        result = smooth_positions([])
        assert result == []

    def test_alpha_one_no_smoothing(self):
        positions = [
            ProcessedPosition(0, 0.0, 0.0, False),
            ProcessedPosition(1, 1.0, 1.0, False),
        ]
        result = smooth_positions(positions, alpha=1.0)

        assert result[0].x == 0.0
        assert result[1].x == 1.0

    def test_smoothing_reduces_jumps(self):
        # Sudden jump from 0 to 1
        positions = [
            ProcessedPosition(0, 0.0, 0.0, False),
            ProcessedPosition(1, 1.0, 1.0, False),
            ProcessedPosition(2, 1.0, 1.0, False),
        ]
        result = smooth_positions(positions, alpha=0.5)

        # First position unchanged
        assert result[0].x == 0.0

        # Second position should be smoothed (0.5 * 1.0 + 0.5 * 0.0 = 0.5)
        assert result[1].x == 0.5

        # Third position continues smoothing
        assert result[2].x == 0.75  # 0.5 * 1.0 + 0.5 * 0.5
