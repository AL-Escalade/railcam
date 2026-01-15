"""Tests for composition module."""

import numpy as np
import pytest

from railcam.composition import (
    calculate_duration,
    calculate_max_duration,
    calculate_output_dimensions,
    calculate_output_fps,
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


class TestCalculateOutputFps:
    """Tests for LCM-based output FPS calculation."""

    def test_single_fps(self):
        assert calculate_output_fps([60.0]) == 60.0

    def test_same_fps_multiple_videos(self):
        assert calculate_output_fps([30.0, 30.0, 30.0]) == 30.0

    def test_multiple_fps_60_30(self):
        # LCM(60, 30) = 60
        assert calculate_output_fps([60.0, 30.0]) == 60.0

    def test_multiple_fps_30_24(self):
        # LCM(30, 24) = 120
        assert calculate_output_fps([30.0, 24.0]) == 120.0

    def test_multiple_fps_60_30_25(self):
        # LCM(60, 30, 25) = 300 - the key use case for railcam
        assert calculate_output_fps([60.0, 30.0, 25.0]) == 300.0

    def test_multiple_fps_50_25(self):
        # LCM(50, 25) = 50 - PAL framerates
        assert calculate_output_fps([50.0, 25.0]) == 50.0

    def test_handles_float_fps_by_rounding(self):
        # 29.97fps (NTSC) should be rounded to 30
        # LCM(30, 24) = 120
        assert calculate_output_fps([29.97, 23.976]) == 120.0

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="At least one FPS"):
            calculate_output_fps([])

    def test_zero_fps_raises(self):
        with pytest.raises(ValueError, match="positive"):
            calculate_output_fps([60.0, 0.0])

    def test_negative_fps_raises(self):
        with pytest.raises(ValueError, match="positive"):
            calculate_output_fps([60.0, -30.0])


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

    def test_non_multiple_fps_25_to_60(self):
        """Test 25fps source synced to 60fps output - non-multiple framerates."""
        # 50 frames @ 25fps = 2s, output @ 60fps for 2s = 120 frames
        indices = time_sync_frame_indices(50, 25.0, 2.0, 60.0)
        assert len(indices) == 120

        # Verify time-based mapping at key points
        # t=0.0s (output #0): source frame 0
        assert indices[0] == 0
        # t=0.5s (output #30): source = int(0.5 * 25) = 12
        assert indices[30] == 12
        # t=1.0s (output #60): source = int(1.0 * 25) = 25 (halfway through)
        assert indices[60] == 25
        # t=1.5s (output #90): source = int(1.5 * 25) = 37
        assert indices[90] == 37
        # t=1.983s (output #119): source = int(1.983 * 25) = 49 (last frame)
        assert indices[119] == 49

    def test_non_multiple_fps_25_to_30(self):
        """Test 25fps source synced to 30fps output - close but non-multiple framerates."""
        # 50 frames @ 25fps = 2s, output @ 30fps for 2s = 60 frames
        indices = time_sync_frame_indices(50, 25.0, 2.0, 30.0)
        assert len(indices) == 60

        # t=0.0s (output #0): source frame 0
        assert indices[0] == 0
        # t=1.0s (output #30): source = int(1.0 * 25) = 25
        assert indices[30] == 25
        # t=1.967s (output #59): source = int(1.967 * 25) = 49
        assert indices[59] == 49


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


class TestMultiVideoTimeSyncIntegration:
    """Integration tests for multi-video time synchronization with different framerates."""

    def test_three_videos_60_30_25_fps_same_duration(self):
        """Test combining 3 videos at 60fps, 30fps, and 25fps - all 2 seconds long.

        This is a critical test for the railcam use case: combining climbing videos
        from different cameras with non-multiple framerates.

        Using LCM(60, 30, 25) = 300fps ensures each source frame appears an exact
        integer number of times (5x, 10x, 12x respectively), eliminating judder.
        """
        # Video A: 60fps, 2 seconds = 120 frames
        # Video B: 30fps, 2 seconds = 60 frames
        # Video C: 25fps, 2 seconds = 50 frames
        frame_counts = [120, 60, 50]
        fps_list = [60.0, 30.0, 25.0]

        # Calculate max duration (should be 2.0s for all)
        max_duration = calculate_max_duration(frame_counts, fps_list)
        assert max_duration == 2.0

        # Output at LCM FPS (300fps) for perfect sync
        output_fps = calculate_output_fps(fps_list)
        assert output_fps == 300.0

        target_frame_count = int(max_duration * output_fps)
        assert target_frame_count == 600

        # Create test frames with unique values to verify correct mapping
        # Each frame contains its frame number as pixel value
        video_a_frames = [np.full((10, 10, 3), i, dtype=np.uint8) for i in range(120)]
        video_b_frames = [np.full((10, 10, 3), i, dtype=np.uint8) for i in range(60)]
        video_c_frames = [np.full((10, 10, 3), i, dtype=np.uint8) for i in range(50)]

        # Sync all videos to common timeline
        synced_a = time_sync_frames(video_a_frames, 60.0, max_duration, output_fps)
        synced_b = time_sync_frames(video_b_frames, 30.0, max_duration, output_fps)
        synced_c = time_sync_frames(video_c_frames, 25.0, max_duration, output_fps)

        # All should have same length (600 frames for 2s at 300fps)
        assert len(synced_a) == 600
        assert len(synced_b) == 600
        assert len(synced_c) == 600

        # Verify time synchronization at key points
        # At t=0.0s (output frame 0): all videos at their frame 0
        assert np.all(synced_a[0] == 0)
        assert np.all(synced_b[0] == 0)
        assert np.all(synced_c[0] == 0)

        # At t=1.0s (output frame 300): halfway through each video
        # Video A @ 60fps: frame 60
        # Video B @ 30fps: frame 30
        # Video C @ 25fps: frame 25
        assert np.all(synced_a[300] == 60)
        assert np.all(synced_b[300] == 30)
        assert np.all(synced_c[300] == 25)

        # At t=0.5s (output frame 150): quarter through
        # Video A @ 60fps: frame 30
        # Video B @ 30fps: frame 15
        # Video C @ 25fps: frame 12
        assert np.all(synced_a[150] == 30)
        assert np.all(synced_b[150] == 15)
        assert np.all(synced_c[150] == 12)

        # At end t≈1.997s (output frame 599): near last frame of each
        # Video A @ 60fps: frame 119
        # Video B @ 30fps: frame 59
        # Video C @ 25fps: frame 49
        assert np.all(synced_a[599] == 119)
        assert np.all(synced_b[599] == 59)
        assert np.all(synced_c[599] == 49)

        # Verify uniform frame distribution (key benefit of LCM)
        # Each frame from 60fps source should appear exactly 5 times (300/60)
        # Each frame from 30fps source should appear exactly 10 times (300/30)
        # Each frame from 25fps source should appear exactly 12 times (300/25)
        for i in range(5):
            assert np.all(synced_a[i] == 0)  # First 5 output frames = source frame 0
        assert np.all(synced_a[5] == 1)  # Frame 5 = source frame 1

        for i in range(10):
            assert np.all(synced_b[i] == 0)  # First 10 output frames = source frame 0
        assert np.all(synced_b[10] == 1)  # Frame 10 = source frame 1

        for i in range(12):
            assert np.all(synced_c[i] == 0)  # First 12 output frames = source frame 0
        assert np.all(synced_c[12] == 1)  # Frame 12 = source frame 1

    def test_three_videos_different_durations_with_freeze(self):
        """Test combining videos with different durations - shorter ones freeze.

        Video A: 60fps, 3 seconds = 180 frames (longest)
        Video B: 30fps, 2 seconds = 60 frames (freezes at t=2s)
        Video C: 25fps, 1.48 seconds = 37 frames (freezes at t=1.48s)

        With LCM=300fps, output has 900 frames for 3 seconds.
        """
        frame_counts = [180, 60, 37]
        fps_list = [60.0, 30.0, 25.0]

        max_duration = calculate_max_duration(frame_counts, fps_list)
        assert max_duration == 3.0  # Video A is longest

        output_fps = calculate_output_fps(fps_list)
        assert output_fps == 300.0

        target_frame_count = int(max_duration * output_fps)
        assert target_frame_count == 900

        video_a_frames = [np.full((10, 10, 3), i, dtype=np.uint8) for i in range(180)]
        video_b_frames = [np.full((10, 10, 3), i, dtype=np.uint8) for i in range(60)]
        video_c_frames = [np.full((10, 10, 3), i, dtype=np.uint8) for i in range(37)]

        synced_a = time_sync_frames(video_a_frames, 60.0, max_duration, output_fps)
        synced_b = time_sync_frames(video_b_frames, 30.0, max_duration, output_fps)
        synced_c = time_sync_frames(video_c_frames, 25.0, max_duration, output_fps)

        assert len(synced_a) == 900
        assert len(synced_b) == 900
        assert len(synced_c) == 900

        # At t=1.0s (frame 300): all videos still playing
        assert np.all(synced_a[300] == 60)
        assert np.all(synced_b[300] == 30)
        assert np.all(synced_c[300] == 25)

        # Video C duration = 37/25 = 1.48s, freezes at frame 444 (1.48 * 300)
        # At t=1.48s (frame 444): Video C at last frame (36)
        assert np.all(synced_c[443] == 36)  # Last frame before freeze
        assert np.all(synced_c[450] == 36)  # Frozen

        # At t=2.0s (frame 600): Video B should be frozen on frame 59
        assert np.all(synced_b[599] == 59)  # Last frame before freeze
        assert np.all(synced_b[600] == 59)  # Frozen

        # At t=2.5s (frame 750): Both B and C frozen, A still playing
        assert np.all(synced_a[750] == 150)
        assert np.all(synced_b[750] == 59)  # Still frozen
        assert np.all(synced_c[750] == 36)  # Still frozen

        # At t≈3.0s (frame 899): A at end, B and C frozen
        assert np.all(synced_a[899] == 179)
        assert np.all(synced_b[899] == 59)
        assert np.all(synced_c[899] == 36)

    def test_full_composition_pipeline_mixed_fps(self):
        """Test the full pipeline: sync + horizontal composition with mixed FPS.

        With LCM=300fps, a 2-second video produces 600 output frames.
        """
        # Simulate 3 cropped video outputs with different FPS
        # Using small frames for test speed
        video_a = [np.full((100, 60, 3), i % 256, dtype=np.uint8) for i in range(120)]
        video_b = [np.full((100, 60, 3), i % 256, dtype=np.uint8) for i in range(60)]
        video_c = [np.full((100, 60, 3), i % 256, dtype=np.uint8) for i in range(50)]

        fps_list = [60.0, 30.0, 25.0]
        frame_counts = [len(video_a), len(video_b), len(video_c)]

        max_duration = calculate_max_duration(frame_counts, fps_list)
        output_fps = calculate_output_fps(fps_list)
        assert output_fps == 300.0

        # Sync all videos
        synced_a = time_sync_frames(video_a, 60.0, max_duration, output_fps)
        synced_b = time_sync_frames(video_b, 30.0, max_duration, output_fps)
        synced_c = time_sync_frames(video_c, 25.0, max_duration, output_fps)

        # Compose horizontally
        composed = compose_frames_horizontal([synced_a, synced_b, synced_c])

        # Verify output (2s at 300fps = 600 frames)
        assert len(composed) == 600
        assert composed[0].shape[0] == 100  # Height preserved
        assert composed[0].shape[1] == 60 * 3  # Width = 3 videos side by side
