"""Tests for composition module."""

import numpy as np
import pytest

from railcam.composition import (
    calculate_duration,
    calculate_max_duration,
    calculate_output_dimensions,
    calculate_target_frame_count,
    compose_frames_horizontal,
    extend_frame_indices,
    extend_frames,
    normalize_frame_height,
    resample_frame_indices,
    resample_frames,
    time_sync_frame_indices,
    time_sync_frames,
)


class TestCalculateTargetFrameCount:
    def test_single_count(self):
        assert calculate_target_frame_count([100]) == 100

    def test_multiple_counts_returns_max(self):
        assert calculate_target_frame_count([100, 150, 120]) == 150

    def test_equal_counts(self):
        assert calculate_target_frame_count([100, 100, 100]) == 100

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="At least one frame count"):
            calculate_target_frame_count([])


class TestResampleFrameIndices:
    def test_same_count_returns_identity(self):
        indices = resample_frame_indices(100, 100)
        assert indices == list(range(100))

    def test_upsample_duplicates_frames(self):
        # 3 frames -> 5 frames
        indices = resample_frame_indices(3, 5)
        assert len(indices) == 5
        # Should be [0, 0 or 1, 1, 1 or 2, 2]
        assert indices[0] == 0
        assert indices[-1] == 2

    def test_downsample_drops_frames(self):
        # 5 frames -> 3 frames
        indices = resample_frame_indices(5, 3)
        assert len(indices) == 3
        assert indices[0] == 0
        assert indices[-1] == 4

    def test_empty_source_returns_empty(self):
        assert resample_frame_indices(0, 10) == []

    def test_empty_target_returns_empty(self):
        assert resample_frame_indices(10, 0) == []

    def test_single_frame_source(self):
        indices = resample_frame_indices(1, 5)
        assert indices == [0, 0, 0, 0, 0]

    def test_single_frame_target(self):
        indices = resample_frame_indices(5, 1)
        assert indices == [0]


class TestResampleFrames:
    def test_same_count_returns_copy(self):
        frames = [np.zeros((10, 10, 3)) for _ in range(5)]
        resampled = resample_frames(frames, 5)
        assert len(resampled) == 5

    def test_upsample(self):
        frames = [np.full((10, 10, 3), i) for i in range(3)]
        resampled = resample_frames(frames, 5)
        assert len(resampled) == 5

    def test_downsample(self):
        frames = [np.full((10, 10, 3), i) for i in range(10)]
        resampled = resample_frames(frames, 3)
        assert len(resampled) == 3

    def test_empty_list_returns_empty(self):
        assert resample_frames([], 10) == []


class TestExtendFrameIndices:
    """Tests for extend_frame_indices (freeze on last frame behavior)."""

    def test_same_count_returns_identity(self):
        indices = extend_frame_indices(100, 100)
        assert indices == list(range(100))

    def test_shorter_source_freezes_on_last_frame(self):
        # 3 frames -> 5 target frames
        indices = extend_frame_indices(3, 5)
        assert len(indices) == 5
        # Should be [0, 1, 2, 2, 2] - plays normally then freezes
        assert indices == [0, 1, 2, 2, 2]

    def test_longer_source_plays_all_frames(self):
        # 5 frames but only 3 target frames
        indices = extend_frame_indices(5, 3)
        assert len(indices) == 3
        assert indices == [0, 1, 2]

    def test_empty_source_returns_empty(self):
        assert extend_frame_indices(0, 10) == []

    def test_empty_target_returns_empty(self):
        assert extend_frame_indices(10, 0) == []

    def test_single_frame_source_repeats(self):
        indices = extend_frame_indices(1, 5)
        assert indices == [0, 0, 0, 0, 0]

    def test_single_frame_target(self):
        indices = extend_frame_indices(5, 1)
        assert indices == [0]


class TestExtendFrames:
    """Tests for extend_frames (freeze on last frame behavior)."""

    def test_same_count_returns_copy(self):
        frames = [np.full((10, 10, 3), i, dtype=np.uint8) for i in range(5)]
        extended = extend_frames(frames, 5)
        assert len(extended) == 5
        # Verify values are correct
        for i, frame in enumerate(extended):
            assert np.all(frame == i)

    def test_shorter_source_freezes_on_last_frame(self):
        frames = [np.full((10, 10, 3), i, dtype=np.uint8) for i in range(3)]
        extended = extend_frames(frames, 5)
        assert len(extended) == 5
        # First 3 frames should have values 0, 1, 2
        assert np.all(extended[0] == 0)
        assert np.all(extended[1] == 1)
        assert np.all(extended[2] == 2)
        # Last 2 frames should repeat value 2 (frozen)
        assert np.all(extended[3] == 2)
        assert np.all(extended[4] == 2)

    def test_longer_source_truncates(self):
        frames = [np.full((10, 10, 3), i, dtype=np.uint8) for i in range(10)]
        extended = extend_frames(frames, 3)
        assert len(extended) == 3
        # Should be first 3 frames
        assert np.all(extended[0] == 0)
        assert np.all(extended[1] == 1)
        assert np.all(extended[2] == 2)

    def test_empty_list_returns_empty(self):
        assert extend_frames([], 10) == []


class TestNormalizeFrameHeight:
    def test_same_height_returns_same(self):
        frame = np.zeros((100, 150, 3), dtype=np.uint8)
        result = normalize_frame_height(frame, 100)
        assert result.shape[0] == 100
        assert result.shape[1] == 150

    def test_scale_down(self):
        frame = np.zeros((200, 300, 3), dtype=np.uint8)
        result = normalize_frame_height(frame, 100)
        assert result.shape[0] == 100
        # Width should be approximately half (maintaining aspect ratio)
        assert abs(result.shape[1] - 150) <= 2

    def test_scale_up(self):
        frame = np.zeros((100, 150, 3), dtype=np.uint8)
        result = normalize_frame_height(frame, 200)
        assert result.shape[0] == 200
        # Width should be approximately double
        assert abs(result.shape[1] - 300) <= 2

    def test_ensures_even_dimensions(self):
        frame = np.zeros((100, 150, 3), dtype=np.uint8)
        result = normalize_frame_height(frame, 101)
        assert result.shape[0] % 2 == 0
        assert result.shape[1] % 2 == 0


class TestComposeFramesHorizontal:
    def test_two_videos_side_by_side(self):
        # Two videos, 3 frames each
        frames1 = [np.zeros((100, 150, 3), dtype=np.uint8) for _ in range(3)]
        frames2 = [np.zeros((100, 200, 3), dtype=np.uint8) for _ in range(3)]

        composed = compose_frames_horizontal([frames1, frames2])

        assert len(composed) == 3
        # Width should be sum of both video widths
        assert composed[0].shape[1] == 150 + 200
        assert composed[0].shape[0] == 100

    def test_three_videos(self):
        frames1 = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(2)]
        frames2 = [np.zeros((100, 150, 3), dtype=np.uint8) for _ in range(2)]
        frames3 = [np.zeros((100, 200, 3), dtype=np.uint8) for _ in range(2)]

        composed = compose_frames_horizontal([frames1, frames2, frames3])

        assert len(composed) == 2
        assert composed[0].shape[1] == 100 + 150 + 200

    def test_different_heights_normalized(self):
        # Different heights, should be normalized to max
        frames1 = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(2)]
        frames2 = [np.zeros((200, 100, 3), dtype=np.uint8) for _ in range(2)]

        composed = compose_frames_horizontal([frames1, frames2])

        assert composed[0].shape[0] == 200  # Max height

    def test_custom_target_height(self):
        frames1 = [np.zeros((100, 150, 3), dtype=np.uint8) for _ in range(2)]
        frames2 = [np.zeros((200, 200, 3), dtype=np.uint8) for _ in range(2)]

        composed = compose_frames_horizontal([frames1, frames2], target_height=150)

        assert composed[0].shape[0] == 150

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="At least one frame list"):
            compose_frames_horizontal([])

    def test_mismatched_lengths_raises(self):
        frames1 = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(3)]
        frames2 = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(5)]

        with pytest.raises(ValueError, match="same length"):
            compose_frames_horizontal([frames1, frames2])

    def test_ensures_even_dimensions(self):
        frames1 = [np.zeros((101, 151, 3), dtype=np.uint8) for _ in range(2)]

        composed = compose_frames_horizontal([frames1], target_height=101)

        assert composed[0].shape[0] % 2 == 0
        assert composed[0].shape[1] % 2 == 0


class TestCalculateOutputDimensions:
    def test_single_video(self):
        width, height = calculate_output_dimensions([100], [200])
        assert height == 200
        assert width == 100

    def test_multiple_videos(self):
        width, height = calculate_output_dimensions([100, 150, 200], [200, 200, 200])
        assert height == 200
        assert width == 100 + 150 + 200

    def test_different_heights_uses_max(self):
        width, height = calculate_output_dimensions([100, 100], [100, 200])
        assert height == 200

    def test_custom_target_height(self):
        width, height = calculate_output_dimensions([100, 100], [200, 200], target_height=100)
        assert height == 100

    def test_empty_lists_raise(self):
        with pytest.raises(ValueError):
            calculate_output_dimensions([], [])

    def test_mismatched_lists_raise(self):
        with pytest.raises(ValueError, match="same length"):
            calculate_output_dimensions([100, 200], [100])


class TestCalculateDuration:
    def test_basic_calculation(self):
        # 30 frames at 30 fps = 1 second
        assert calculate_duration(30, 30.0) == 1.0

    def test_fractional_duration(self):
        # 45 frames at 30 fps = 1.5 seconds
        assert calculate_duration(45, 30.0) == 1.5

    def test_zero_fps_returns_zero(self):
        assert calculate_duration(100, 0.0) == 0.0

    def test_negative_fps_returns_zero(self):
        assert calculate_duration(100, -30.0) == 0.0


class TestCalculateMaxDuration:
    def test_same_fps_different_counts(self):
        # 60 frames @ 30fps = 2s, 90 frames @ 30fps = 3s
        assert calculate_max_duration([60, 90], [30.0, 30.0]) == 3.0

    def test_different_fps_same_duration(self):
        # 30 frames @ 30fps = 1s, 60 frames @ 60fps = 1s
        assert calculate_max_duration([30, 60], [30.0, 60.0]) == 1.0

    def test_different_fps_different_duration(self):
        # 60 frames @ 30fps = 2s, 90 frames @ 60fps = 1.5s
        assert calculate_max_duration([60, 90], [30.0, 60.0]) == 2.0

    def test_empty_lists_raise(self):
        with pytest.raises(ValueError):
            calculate_max_duration([], [])

    def test_mismatched_lists_raise(self):
        with pytest.raises(ValueError):
            calculate_max_duration([60, 90], [30.0])


class TestTimeSyncFrameIndices:
    def test_same_fps_same_duration(self):
        # 30 frames @ 30fps = 1s, output @ 30fps for 1s = 30 frames
        indices = time_sync_frame_indices(30, 30.0, 1.0, 30.0)
        assert len(indices) == 30
        assert indices == list(range(30))

    def test_shorter_source_freezes(self):
        # 30 frames @ 30fps = 1s, but output for 2s @ 30fps = 60 frames
        indices = time_sync_frame_indices(30, 30.0, 2.0, 30.0)
        assert len(indices) == 60
        # First 30 frames map 1:1
        assert indices[:30] == list(range(30))
        # Remaining 30 frames freeze on last
        assert all(idx == 29 for idx in indices[30:])

    def test_different_fps_upsample(self):
        # 30 frames @ 30fps = 1s, output @ 60fps for 1s = 60 frames
        indices = time_sync_frame_indices(30, 30.0, 1.0, 60.0)
        assert len(indices) == 60
        # Each source frame should appear twice
        assert indices[0] == 0
        assert indices[1] == 0
        assert indices[2] == 1
        assert indices[3] == 1

    def test_different_fps_downsample(self):
        # 60 frames @ 60fps = 1s, output @ 30fps for 1s = 30 frames
        indices = time_sync_frame_indices(60, 60.0, 1.0, 30.0)
        assert len(indices) == 30
        # Every other source frame
        assert indices[0] == 0
        assert indices[1] == 2
        assert indices[2] == 4

    def test_empty_on_invalid_input(self):
        assert time_sync_frame_indices(0, 30.0, 1.0, 30.0) == []
        assert time_sync_frame_indices(30, 0.0, 1.0, 30.0) == []
        assert time_sync_frame_indices(30, 30.0, 0.0, 30.0) == []
        assert time_sync_frame_indices(30, 30.0, 1.0, 0.0) == []


class TestTimeSyncFrames:
    def test_same_fps_same_duration(self):
        frames = [np.full((10, 10, 3), i, dtype=np.uint8) for i in range(30)]
        synced = time_sync_frames(frames, 30.0, 1.0, 30.0)
        assert len(synced) == 30

    def test_shorter_source_freezes(self):
        frames = [np.full((10, 10, 3), i, dtype=np.uint8) for i in range(30)]
        synced = time_sync_frames(frames, 30.0, 2.0, 30.0)
        assert len(synced) == 60
        # Last frame should be repeated
        assert np.all(synced[29] == 29)
        assert np.all(synced[59] == 29)

    def test_empty_list_returns_empty(self):
        assert time_sync_frames([], 30.0, 1.0, 30.0) == []
