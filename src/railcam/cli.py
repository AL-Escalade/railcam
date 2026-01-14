"""Command-line interface for railcam."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from railcam import __version__
from railcam.composition import (
    calculate_duration,
    calculate_max_duration,
    compose_frames_horizontal,
    time_sync_frames,
)
from railcam.cropping import (
    MAX_ZOOM_FACTOR,
    TORSO_HEIGHT_RATIO,
    CropRegion,
    calculate_average_torso_height,
    calculate_crop_dimensions,
    scale_frame,
)
from railcam.multi_video import (
    InputParseError,
    VideoInput,
    parse_input_spec,
)
from railcam.output import (
    FFmpegNotFoundError,
    OutputGenerationError,
    generate_output,
    get_output_path,
    parse_output_format,
)
from railcam.pose import (
    ClimberSelector,
    DetectionResult,
    PelvisPosition,
    PoseDetector,
    draw_pose_overlay_on_crop,
    person_to_detection_result,
    select_climber,
)
from railcam.processing import (
    NoValidDetectionsError,
    interpolate_positions,
    smooth_positions,
)
from railcam.video import (
    InvalidFrameRangeError,
    UnsupportedFormatError,
    VideoError,
    VideoNotFoundError,
    extract_frames,
    get_video_metadata,
    validate_frame_range,
)


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="railcam",
        description=(
            "Generate cropped videos from speed climbing footage, tracking the climber's pelvis."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single video mode (positional arguments)
  railcam video.mp4 100 250                  # Output MP4 (default)
  railcam video.mp4 100 250 --format gif     # Output GIF
  railcam video.mp4 100 250 --output climb.mp4
  railcam video.mp4 100 250 --width 480
  railcam video.mp4 100 250 --speed 0.5      # Slow motion (half speed)
  railcam video.mp4 100 250 --debug

  # Multi-climber video (track specific climber)
  railcam video.mp4 100 250 --climber left   # Track left climber
  railcam video.mp4 100 250 --climber right  # Track right climber

  # Multi-video mode (side-by-side comparison)
  railcam --input video1.mp4:100:250 --input video2.mp4:50:200
  railcam --input v1.mp4:0:100 --input v2.mp4:0:150 --input v3.mp4:0:120

  # Multi-video with climber selection
  railcam --input video.mp4:100:250:left --input video.mp4:100:250:right
        """,
    )

    # Positional arguments for single-video mode (optional when using --input)
    parser.add_argument(
        "video",
        type=Path,
        nargs="?",
        help="Path to the input video file (single-video mode)",
    )
    parser.add_argument(
        "start_frame",
        type=int,
        nargs="?",
        help="Start frame number (0-indexed, inclusive)",
    )
    parser.add_argument(
        "end_frame",
        type=int,
        nargs="?",
        help="End frame number (0-indexed, inclusive)",
    )

    # Multi-video mode
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        action="append",
        dest="inputs",
        metavar="PATH:START:END[:CLIMBER]",
        help="Video input specification (can be repeated for side-by-side). "
        "Format: path:start_frame:end_frame[:left|right] "
        "(e.g., video.mp4:100:250 or video.mp4:100:250:left)",
    )

    # Climber selection (for positional mode)
    parser.add_argument(
        "-c",
        "--climber",
        type=str,
        choices=["left", "right"],
        help="Which climber to track when video has multiple climbers (left or right). "
        "For --input mode, append :left or :right to the input spec instead.",
    )

    # Output options
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output file path (default: <video_name>.<format> in current directory)",
    )
    parser.add_argument(
        "-f",
        "--format",
        type=str,
        default="mp4",
        choices=["mp4", "gif"],
        help="Output format: mp4 (default) or gif",
    )
    parser.add_argument(
        "-w",
        "--width",
        type=int,
        help="Output width in pixels (height calculated from 5:3 ratio)",
    )
    parser.add_argument(
        "-H",
        "--height",
        type=int,
        help="Output height in pixels (width calculated from 5:3 ratio)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with pose overlay visualization",
    )
    parser.add_argument(
        "-s",
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed factor (0.5 = half speed/slow-mo, 2.0 = double speed). Default: 1.0",
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    return parser


def validate_args(args: argparse.Namespace) -> list[VideoInput]:
    """Validate and convert arguments to VideoInput list.

    Returns:
        List of VideoInput objects.

    Raises:
        SystemExit: If arguments are invalid.
    """
    has_positional = args.video is not None
    has_inputs = args.inputs is not None and len(args.inputs) > 0

    if has_positional and has_inputs:
        print(
            "Error: Cannot use both positional arguments and --input. Choose one mode.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not has_positional and not has_inputs:
        print(
            "Error: Must provide either positional arguments (video start_frame end_frame) "
            "or --input options.",
            file=sys.stderr,
        )
        sys.exit(1)

    if has_positional:
        # Validate that all positional args are provided
        if args.start_frame is None or args.end_frame is None:
            print(
                "Error: Must provide all three positional arguments: video start_frame end_frame",
                file=sys.stderr,
            )
            sys.exit(1)

        # Parse climber selector for positional mode
        climber_selector = ClimberSelector.AUTO
        if args.climber is not None:
            climber_selector = (
                ClimberSelector.LEFT if args.climber == "left" else ClimberSelector.RIGHT
            )

        return [
            VideoInput(
                path=args.video,
                start_frame=args.start_frame,
                end_frame=args.end_frame,
                climber_selector=climber_selector,
            )
        ]

    # Multi-video mode
    if args.climber is not None:
        print(
            "Error: --climber option is only for positional mode. "
            "For --input mode, append :left or :right to the input spec.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        return [parse_input_spec(spec) for spec in args.inputs]
    except InputParseError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def print_progress(current: int, total: int, stage: str) -> None:
    """Print progress information."""
    percent = (current / total) * 100 if total > 0 else 0
    bar_width = 30
    filled = int(bar_width * current / total) if total > 0 else 0
    bar = "█" * filled + "░" * (bar_width - filled)
    print(f"\r{stage}: [{bar}] {percent:5.1f}% ({current}/{total})", end="", flush=True)


@dataclass
class VideoAnalysisResult:
    """Result of analyzing a video (pose detection, no cropping yet)."""

    detections: list[DetectionResult]
    detections_by_frame: dict[int, DetectionResult]
    frames_cache: dict[int, Any]  # np.ndarray
    avg_torso_height: float
    fps: float
    video_width: int
    video_height: int
    frame_count: int


@dataclass
class VideoProcessingResult:
    """Result of processing a single video."""

    cropped_frames: list[Any]  # np.ndarray
    fps: float
    zoom_factor: float
    frame_count: int


@dataclass
class _PoseDetectionResult:
    """Internal result of pose detection loop."""

    detections: list[DetectionResult]
    detections_by_frame: dict[int, DetectionResult]
    frames_cache: dict[int, Any]
    torso_heights: list[float]


def _detect_poses_with_tracking(
    video_input: VideoInput,
    detector: PoseDetector,
    frame_count: int,
) -> _PoseDetectionResult:
    """Detect poses for all frames with proximity-based climber tracking.

    Args:
        video_input: Video input specification.
        detector: Pose detector instance.
        frame_count: Total number of frames to process.

    Returns:
        Detection results, frame cache, and torso heights.
    """
    selector = video_input.climber_selector
    detections: list[DetectionResult] = []
    detections_by_frame: dict[int, DetectionResult] = {}
    frames_cache: dict[int, Any] = {}
    torso_heights: list[float] = []

    # Track previous position for proximity-based tracking
    previous_position: PelvisPosition | None = None

    for i, (frame_num, frame) in enumerate(
        extract_frames(video_input.path, video_input.start_frame, video_input.end_frame)
    ):
        # Detect all persons with valid pelvis
        multi_result = detector.detect_all_persons(frame, frame_num)

        # Select the target climber
        selected_person = select_climber(
            multi_result.persons,
            selector,
            previous_position,
        )

        # Convert to DetectionResult format
        detection = person_to_detection_result(selected_person, frame_num)

        # Update previous position for next frame's proximity tracking
        if detection.position is not None:
            previous_position = detection.position

        detections.append(detection)
        detections_by_frame[frame_num] = detection
        frames_cache[frame_num] = frame

        # Collect torso heights for zoom calculation
        if detection.torso is not None:
            torso_heights.append(detection.torso.height)

        print_progress(i + 1, frame_count, "  Detecting")

    print()  # Newline after progress bar

    return _PoseDetectionResult(
        detections=detections,
        detections_by_frame=detections_by_frame,
        frames_cache=frames_cache,
        torso_heights=torso_heights,
    )


def analyze_video(
    video_input: VideoInput,
    detector: PoseDetector,
) -> VideoAnalysisResult:
    """Analyze a video: extract frames and detect poses.

    This is the first phase of processing - no cropping is done yet.
    Uses multi-person detection with climber selection based on video_input.climber_selector.

    Returns:
        VideoAnalysisResult with detections and metadata.
    """
    # Get video metadata
    print(f"\nAnalyzing: {video_input.path}")
    metadata = get_video_metadata(video_input.path)
    print(f"  Resolution: {metadata.width}x{metadata.height}")
    print(f"  FPS: {metadata.fps:.2f}")

    # Validate frame range
    validate_frame_range(video_input.start_frame, video_input.end_frame, metadata.total_frames)
    frame_count = video_input.end_frame - video_input.start_frame + 1
    print(f"  Frames {video_input.start_frame} to {video_input.end_frame} ({frame_count} frames)")

    # Show climber selector if not AUTO
    if video_input.climber_selector != ClimberSelector.AUTO:
        print(f"  Climber selection: {video_input.climber_selector.value}")

    # Pose detection with multi-person support
    print("  Detecting poses...")
    pose_result = _detect_poses_with_tracking(video_input, detector, frame_count)

    valid_count = sum(1 for d in pose_result.detections if d.position is not None)
    print(f"  Detected pelvis in {valid_count}/{frame_count} frames")

    avg_torso = calculate_average_torso_height(pose_result.torso_heights)
    print(f"  Avg torso height: {avg_torso:.3f} (normalized to source)")

    return VideoAnalysisResult(
        detections=pose_result.detections,
        detections_by_frame=pose_result.detections_by_frame,
        frames_cache=pose_result.frames_cache,
        avg_torso_height=avg_torso,
        fps=metadata.fps,
        video_width=metadata.width,
        video_height=metadata.height,
        frame_count=frame_count,
    )


def crop_video(
    analysis: VideoAnalysisResult,
    target_torso_ratio: float,
    debug: bool = False,
    target_width: int | None = None,
    target_height: int | None = None,
) -> VideoProcessingResult:
    """Crop a video based on analysis results and target torso ratio.

    Uses a scale-then-crop approach:
    1. Scale the entire frame so the torso reaches the target size
    2. Crop the region of interest centered on the pelvis
    3. Add padding if the crop extends beyond the scaled frame bounds

    Args:
        analysis: Result from analyze_video().
        target_torso_ratio: Target torso height as fraction of output (0-1).
        debug: Enable debug overlay.
        target_width: Optional output width.
        target_height: Optional output height.

    Returns:
        VideoProcessingResult with cropped frames.
    """
    # Define output dimensions (base crop with 3:5 ratio)
    output_width, output_height = calculate_crop_dimensions(
        analysis.video_width, analysis.video_height
    )

    # Calculate scale factor to achieve target torso ratio
    if analysis.avg_torso_height > 0:
        torso_px_source = analysis.avg_torso_height * analysis.video_height
        torso_px_target = target_torso_ratio * output_height
        scale_factor = torso_px_target / torso_px_source
        # Clamp to reasonable bounds
        scale_factor = max(0.1, min(scale_factor, MAX_ZOOM_FACTOR))
    else:
        scale_factor = 1.0

    print(f"  Source video: {analysis.video_width}x{analysis.video_height}")
    avg_torso = analysis.avg_torso_height
    print(f"  Avg torso in source: {avg_torso:.3f} ({avg_torso:.1%})")
    print(f"  Scale factor: {scale_factor:.2f}x (target torso: {target_torso_ratio:.1%})")
    print(f"  Output size: {output_width}x{output_height}")

    # Calculate scaled frame dimensions
    scaled_width = int(analysis.video_width * scale_factor)
    scaled_height = int(analysis.video_height * scale_factor)
    print(f"  Scaled frame: {scaled_width}x{scaled_height}")

    # Check if we need padding (scaled frame smaller than output)
    needs_padding = scaled_width < output_width or scaled_height < output_height
    if needs_padding:
        print("  Padding will be added (scaled frame smaller than output)")

    # Calculate effective torso ratio
    torso_px_scaled = analysis.avg_torso_height * analysis.video_height * scale_factor
    effective_ratio = torso_px_scaled / output_height
    print(f"  Torso in output: {effective_ratio:.1%} ({torso_px_scaled:.1f}px)")

    # Process positions
    positions = interpolate_positions(analysis.detections)
    positions = smooth_positions(positions)

    # Crop frames
    print("  Processing frames (scale then crop)...")
    cropped_frames = []

    for i, pos in enumerate(positions):
        frame = analysis.frames_cache[pos.frame_num]

        # 1. Scale the entire frame
        if abs(scale_factor - 1.0) > 0.01:
            scaled_frame = cv2.resize(
                frame,
                (scaled_width, scaled_height),
                interpolation=cv2.INTER_LANCZOS4 if scale_factor < 1 else cv2.INTER_LINEAR,
            )
        else:
            scaled_frame = frame

        # 2. Calculate pelvis position in scaled coordinates
        pelvis_x = int(pos.x * scaled_width)
        pelvis_y = int(pos.y * scaled_height)

        # 3. Calculate ideal crop region centered on pelvis
        ideal_x = pelvis_x - output_width // 2
        ideal_y = pelvis_y - output_height // 2

        # 4. Create output frame (may need padding)
        if needs_padding:
            # Create black canvas
            if len(scaled_frame.shape) == 3:
                output_frame = np.zeros(
                    (output_height, output_width, scaled_frame.shape[2]),
                    dtype=scaled_frame.dtype,
                )
            else:
                output_frame = np.zeros(
                    (output_height, output_width),
                    dtype=scaled_frame.dtype,
                )

            # Calculate source region (clamp to scaled frame bounds)
            src_x1 = max(0, ideal_x)
            src_y1 = max(0, ideal_y)
            src_x2 = min(scaled_width, ideal_x + output_width)
            src_y2 = min(scaled_height, ideal_y + output_height)

            # Calculate destination region in output
            dst_x1 = src_x1 - ideal_x
            dst_y1 = src_y1 - ideal_y
            dst_x2 = dst_x1 + (src_x2 - src_x1)
            dst_y2 = dst_y1 + (src_y2 - src_y1)

            # Copy the visible portion
            output_frame[dst_y1:dst_y2, dst_x1:dst_x2] = scaled_frame[src_y1:src_y2, src_x1:src_x2]
            cropped = output_frame
        else:
            # No padding needed - just clamp and crop
            x = max(0, min(ideal_x, scaled_width - output_width))
            y = max(0, min(ideal_y, scaled_height - output_height))
            cropped = scaled_frame[y : y + output_height, x : x + output_width]

        # Apply debug overlay with scaled coordinates
        if debug:
            original_detection = analysis.detections_by_frame[pos.frame_num]
            # Create a CropRegion in scaled space for coordinate transformation
            if needs_padding:
                # For padding case, use ideal coordinates (may be negative)
                crop_region = CropRegion(
                    x=ideal_x, y=ideal_y, width=output_width, height=output_height
                )
            else:
                crop_region = CropRegion(x=x, y=y, width=output_width, height=output_height)
            # Draw overlay using scaled dimensions as reference
            cropped = draw_pose_overlay_on_crop(
                cropped,
                original_detection,
                crop_region,
                scaled_width,
                scaled_height,
            )

        # Scale if requested
        if target_width is not None or target_height is not None:
            cropped = scale_frame(cropped, target_width, target_height)

        cropped_frames.append(cropped)
        print_progress(i + 1, len(positions), "  Processing")

    print()  # Newline after progress bar

    return VideoProcessingResult(
        cropped_frames=cropped_frames,
        fps=analysis.fps,
        zoom_factor=scale_factor,
        frame_count=analysis.frame_count,
    )


def process_single_video(
    video_input: VideoInput,
    detector: PoseDetector,
    debug: bool = False,
    target_width: int | None = None,
    target_height: int | None = None,
) -> VideoProcessingResult:
    """Process a single video: detect poses, calculate zoom, crop frames.

    For single video mode, uses TORSO_HEIGHT_RATIO as target.

    Returns:
        VideoProcessingResult with cropped frames and metadata.
    """
    analysis = analyze_video(video_input, detector)

    # For single video, target is TORSO_HEIGHT_RATIO (but can't zoom out)
    # Use the larger of TORSO_HEIGHT_RATIO and avg_torso (to avoid needing zoom out)
    target_torso = max(TORSO_HEIGHT_RATIO, analysis.avg_torso_height)

    return crop_video(
        analysis,
        target_torso,
        debug=debug,
        target_width=target_width,
        target_height=target_height,
    )


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)

    # Validate mutually exclusive options
    if args.width is not None and args.height is not None:
        print("Error: Cannot specify both --width and --height. Choose one.", file=sys.stderr)
        return 1

    # Validate speed
    if args.speed <= 0:
        print("Error: Speed must be a positive number.", file=sys.stderr)
        return 1

    # Parse and validate inputs
    video_inputs = validate_args(args)
    is_multi_video = len(video_inputs) > 1

    try:
        # Process all videos
        print(f"Processing {len(video_inputs)} video(s)...")
        results: list[VideoProcessingResult] = []

        with PoseDetector() as detector:
            if is_multi_video:
                # Multi-video mode: analyze all first, then crop with normalized zoom
                print("\n=== Phase 1: Analyzing all videos ===")
                analyses: list[VideoAnalysisResult] = []
                for video_input in video_inputs:
                    analysis = analyze_video(video_input, detector)
                    analyses.append(analysis)

                # Always target TORSO_HEIGHT_RATIO (1/6) for normalized output
                reference_torso = TORSO_HEIGHT_RATIO

                print("\n=== Phase 2: Cropping with normalized zoom ===")
                print(f"  Target torso ratio: {reference_torso:.1%}")

                # Info about videos needing padding (zoom out with black borders)
                for i, analysis in enumerate(analyses):
                    if analysis.avg_torso_height > reference_torso:
                        torso_pct = analysis.avg_torso_height
                        print(f"  Video {i + 1}: torso ({torso_pct:.1%}) > target - padding")

                for i, (video_input, analysis) in enumerate(zip(video_inputs, analyses)):
                    print(f"\nCropping video {i + 1}: {video_input.path.name}")
                    result = crop_video(
                        analysis,
                        reference_torso,
                        debug=args.debug,
                    )
                    results.append(result)
            else:
                # Single video mode
                for video_input in video_inputs:
                    result = process_single_video(
                        video_input,
                        detector,
                        debug=args.debug,
                        target_width=args.width,
                        target_height=args.height,
                    )
                    results.append(result)

        # Prepare output frames
        if is_multi_video:
            # Multi-video: synchronize by TIME and compose side-by-side
            print("\n=== Phase 3: Synchronizing and composing ===")

            # Calculate durations and find max
            frame_counts = [r.frame_count for r in results]
            fps_list = [r.fps for r in results]
            durations = [calculate_duration(fc, fps) for fc, fps in zip(frame_counts, fps_list)]
            max_duration = calculate_max_duration(frame_counts, fps_list)

            # Use max FPS for output (smoother result)
            output_fps = max(fps_list)
            target_frame_count = int(max_duration * output_fps)

            print(f"  Max duration: {max_duration:.2f}s")
            print(f"  Output FPS: {output_fps:.2f}")
            print(f"  Target frame count: {target_frame_count}")

            # Time-sync each video (freeze on last frame when source ends)
            synced_frame_lists = []
            for i, result in enumerate(results):
                synced = time_sync_frames(
                    result.cropped_frames,
                    result.fps,
                    max_duration,
                    output_fps,
                )
                synced_frame_lists.append(synced)
                duration = durations[i]
                # Show frame dimensions for debugging
                frame_h, frame_w = result.cropped_frames[0].shape[:2]
                fps = result.fps
                freeze = " (freezes)" if duration < max_duration else ""
                print(
                    f"  Video {i + 1}: {frame_w}x{frame_h}, {duration:.2f}s @ {fps:.1f}fps{freeze}"
                )

            # Compose side-by-side
            print("  Composing frames horizontally...")
            output_frames = compose_frames_horizontal(synced_frame_lists)

            # Scale if requested
            if args.width is not None or args.height is not None:
                print("  Scaling output...")
                output_frames = [scale_frame(f, args.width, args.height) for f in output_frames]

            # Use output FPS (max of all videos)
            avg_fps = output_fps
        else:
            # Single video: frames already processed
            output_frames = results[0].cropped_frames
            avg_fps = results[0].fps

        # Generate output
        output_format = parse_output_format(args.format)
        format_name = output_format.value.upper()
        print(f"\nGenerating {format_name}...")

        # Determine output path
        if args.output is not None:
            output_path = get_output_path(video_inputs[0].path, args.output, output_format)
        else:
            if is_multi_video:
                # Generate a combined name for multi-video output
                output_path = get_output_path(Path("combined"), None, output_format)
            else:
                output_path = get_output_path(video_inputs[0].path, None, output_format)

        # Calculate effective FPS based on speed factor
        effective_fps = avg_fps * args.speed
        if args.speed != 1.0:
            print(f"  Speed: {args.speed}x ({avg_fps:.2f} fps -> {effective_fps:.2f} fps)")

        generate_output(
            output_frames,
            output_path,
            effective_fps,
            output_format,
            on_progress=lambda c, t, s: print_progress(c, t, s),
        )
        print()  # Newline after progress bar

        # Report success
        file_size = output_path.stat().st_size
        size_str = (
            f"{file_size / 1024 / 1024:.2f} MB"
            if file_size > 1024 * 1024
            else f"{file_size / 1024:.1f} KB"
        )
        print(f"\nSuccess! {format_name} saved to: {output_path}")
        print(f"  Size: {size_str}")
        if is_multi_video:
            print(f"  Videos combined: {len(video_inputs)}")
            print(f"  Output frames: {len(output_frames)}")

        return 0

    except VideoNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except UnsupportedFormatError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except InvalidFrameRangeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except NoValidDetectionsError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except FFmpegNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except OutputGenerationError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except VideoError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nCancelled by user.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
