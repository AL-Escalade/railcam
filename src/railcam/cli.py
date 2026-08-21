"""Command-line interface for railcam."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator, Sequence
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from railcam import __version__
from railcam.composition import (
    FrameCursor,
    calculate_duration,
    calculate_max_duration,
    calculate_output_fps,
    compose_frame_row,
    time_sync_frame_indices,
)
from railcam.cropping import (
    MAX_ZOOM_FACTOR,
    TORSO_HEIGHT_RATIO,
    CropRegion,
    calculate_average_torso_height,
    calculate_crop_dimensions,
    max_zoom_keeping_body_in_frame,
    resize_interpolation,
    scale_frame,
)
from railcam.frame_source import FrameSource
from railcam.labeling import (
    LABEL_FONT_RATIO,
    SUBLABEL_FONT_RATIO,
    LabelLine,
    append_label_band,
    band_height,
    font_size,
)
from railcam.multi_video import (
    InputParseError,
    VideoInput,
    parse_input_spec,
)
from railcam.output import (
    FFmpegNotFoundError,
    OutputGenerationError,
    generate_output_stream,
    get_output_path,
    parse_output_format,
)
from railcam.pose import (
    CONFIDENCE_THRESHOLD,
    DEFAULT_MODEL_SIZE,
    MAX_IMGSZ,
    MIN_IMGSZ,
    MODEL_SIZES,
    ClimberSelector,
    DetectionResult,
    MultiPersonDetectionResult,
    PersonDetection,
    PoseDetector,
    draw_pose_overlay_on_crop,
)
from railcam.processing import (
    NoValidDetectionsError,
    ProcessedPosition,
    interpolate_positions,
    smooth_positions_zero_phase,
)
from railcam.tracking import (
    build_tracks,
    repair_track_gaps,
    select_track,
    track_to_detections,
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

  # Labels displayed under each video (paired with the inputs, in order)
  railcam video.mp4 100 250 --label "Dupont"
  railcam --input v1.mp4:0:100 --input v2.mp4:0:150 --label "Dupont" --label "Martin"
  railcam --input v1.mp4:0:100 --input v2.mp4:0:150 --label "Dupont"  # second one unlabeled
  railcam video.mp4 100 250 --label="-Dupont"  # attached form, for a label starting with -

  # Second line, drawn smaller under the label (paired with the inputs too)
  railcam video.mp4 100 250 --label "Dupont" --sublabel "4.704"
  railcam --input v1.mp4:0:100 --input v2.mp4:0:150 --sublabel "4.704" --sublabel "5.120"
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

    parser.add_argument(
        "--label",
        type=str,
        action="append",
        dest="labels",
        metavar="TEXT",
        help="Text displayed in a band under a video (can be repeated). "
        "Labels are paired with the inputs in the order they are given; "
        "inputs left without a label get an empty band. "
        "Use --label=TEXT when the text starts with a dash.",
    )

    parser.add_argument(
        "--sublabel",
        type=str,
        action="append",
        dest="sublabels",
        metavar="TEXT",
        help="Second line, drawn under the label in a smaller font (can be repeated). "
        "Paired with the inputs like --label, and usable without a label. "
        "Use --sublabel=TEXT when the text starts with a dash.",
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
        "--model",
        type=str,
        default=DEFAULT_MODEL_SIZE,
        choices=list(MODEL_SIZES),
        help="YOLOv8-pose model size, from n (fastest) to x (most accurate). "
        f"Default: {DEFAULT_MODEL_SIZE}. Larger models detect harder poses but are "
        "several times slower; missed frames are repaired at high resolution regardless.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        help="Detection resolution in pixels. Default: half the source's largest side, "
        f"clamped to [{MIN_IMGSZ}, {MAX_IMGSZ}]. Raise it when climbers are small in a "
        "wide shot, lower it for a faster run.",
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    return parser


def _text_for_single_video(values: list[str], option: str) -> str:
    """Return the single text given for a repeatable text option.

    Args:
        values: Texts in the order they were given.
        option: Option name, for the error message.

    Returns:
        The text, or an empty string when none was given.

    Raises:
        SystemExit: If more than one text was given.
    """
    if len(values) > 1:
        print(
            f"Error: {len(values)} {option} options given for a single video. "
            "Use --input to render several videos.",
            file=sys.stderr,
        )
        sys.exit(1)
    return values[0] if values else ""


def _texts_for_inputs(values: list[str], input_count: int, option: str) -> list[str]:
    """Pair the texts given for a repeatable text option with the inputs.

    Args:
        values: Texts in the order they were given.
        input_count: Number of inputs they are paired with.
        option: Option name, for the error message.

    Returns:
        One text per input, empty where none was given.

    Raises:
        SystemExit: If more texts than inputs were given.
    """
    if len(values) > input_count:
        print(
            f"Error: {len(values)} {option} option(s) given for {input_count} input(s). "
            f"{option} values are paired with the inputs in order.",
            file=sys.stderr,
        )
        sys.exit(1)
    return values + [""] * (input_count - len(values))


def validate_args(args: argparse.Namespace) -> list[VideoInput]:
    """Validate and convert arguments to VideoInput list.

    Returns:
        List of VideoInput objects.

    Raises:
        SystemExit: If arguments are invalid.
    """
    has_positional = args.video is not None
    has_inputs = args.inputs is not None and len(args.inputs) > 0
    labels: list[str] = list(args.labels or [])
    sublabels: list[str] = list(args.sublabels or [])

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

        label = _text_for_single_video(labels, "--label")
        sublabel = _text_for_single_video(sublabels, "--sublabel")

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
                label=label,
                sublabel=sublabel,
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

    labels = _texts_for_inputs(labels, len(args.inputs), "--label")
    sublabels = _texts_for_inputs(sublabels, len(args.inputs), "--sublabel")

    try:
        video_inputs = [parse_input_spec(spec) for spec in args.inputs]
    except InputParseError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    for video_input, text, subtext in zip(video_inputs, labels, sublabels):
        video_input.label = text
        video_input.sublabel = subtext

    return video_inputs


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
    avg_torso_height: float
    fps: float
    video_width: int
    video_height: int
    frame_count: int
    # Widest and longest reach from the pelvis over the clip, as fractions of
    # the video dimensions. The crop is centered on the pelvis, so this is what
    # has to fit on each side for the climber never to leave the frame.
    body_half_width: float = 0.0
    body_half_height: float = 0.0


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
    torso_heights: list[float]


def _detect_for_repair(
    detector: PoseDetector, source: FrameSource, frame_num: int
) -> list[PersonDetection]:
    """Detect persons on one frame at the repair resolution."""
    frame = source.get_frame(frame_num)
    return detector.detect_all_persons(frame, frame_num, imgsz=detector.repair_size(frame)).persons


def _detect_poses_with_tracking(
    video_input: VideoInput,
    detector: PoseDetector,
    frame_count: int,
) -> _PoseDetectionResult:
    """Detect poses for all frames, then select the climber by motion.

    All persons are detected first; detections are grouped into motion
    tracks and the climber is chosen among tracks that actually climb,
    so static bystanders are never selected (see railcam.tracking).

    Args:
        video_input: Video input specification.
        detector: Pose detector instance.
        frame_count: Total number of frames to process.

    Returns:
        Detection results and torso heights. No frame is retained.
    """
    frame_results: list[MultiPersonDetectionResult] = []

    # Frames are consumed one at a time and dropped: detection is the only
    # thing that needs them here, and cropping re-reads the range later.
    for i, (frame_num, frame) in enumerate(
        extract_frames(video_input.path, video_input.start_frame, video_input.end_frame)
    ):
        frame_results.append(detector.detect_all_persons(frame, frame_num))
        print_progress(i + 1, frame_count, "  Detecting")

    print()  # Newline after progress bar

    tracks = build_tracks(frame_results)
    selected = select_track(tracks, video_input.climber_selector)

    # Repair undetected frames (crouched start, occlusions) with targeted
    # high-resolution inference, bounded to the frames that need it
    if selected is not None:
        frame_nums = [r.frame_num for r in frame_results]
        missing = sum(1 for f in frame_nums if f not in selected.detections)
        if missing:
            print(f"  Repairing {missing} undetected frame(s) at high resolution...")
            # Re-open the source only for the frames that need repairing;
            # FrameSource seeks by index and decodes forward on long-GOP codecs
            with closing(FrameSource(video_input.path)) as source:
                repaired = repair_track_gaps(
                    selected,
                    frame_nums,
                    lambda f: _detect_for_repair(detector, source, f),
                    avoid=[track for track in tracks if track is not selected],
                )
            print(f"  Recovered {repaired}/{missing} frame(s)")

    detections = track_to_detections(selected, [r.frame_num for r in frame_results])

    return _PoseDetectionResult(
        detections=detections,
        detections_by_frame={d.frame_num: d for d in detections},
        torso_heights=[d.torso.height for d in detections if d.torso is not None],
    )


def _body_reach(detections: Sequence[DetectionResult]) -> tuple[float, float]:
    """Widest and longest reach from the pelvis over a clip.

    Args:
        detections: Per-frame detections of the selected climber.

    Returns:
        (half width, half height) as fractions of the video dimensions, both 0
        when nothing was detected confidently enough to measure.
    """
    half_width = half_height = 0.0
    for detection in detections:
        if detection.position is None:
            continue
        for x, y, confidence in detection.landmarks:
            if confidence < CONFIDENCE_THRESHOLD:
                continue
            half_width = max(half_width, abs(x - detection.position.x))
            half_height = max(half_height, abs(y - detection.position.y))
    return half_width, half_height


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
    probe = np.zeros((metadata.height, metadata.width, 3), dtype=np.uint8)
    print(f"  Detection resolution: {detector.inference_size(probe)}px")
    print("  Detecting poses...")
    pose_result = _detect_poses_with_tracking(video_input, detector, frame_count)

    valid_count = sum(1 for d in pose_result.detections if d.position is not None)
    print(f"  Detected pelvis in {valid_count}/{frame_count} frames")

    avg_torso = calculate_average_torso_height(pose_result.torso_heights)
    print(f"  Avg torso height: {avg_torso:.3f} (normalized to source)")

    half_width, half_height = _body_reach(pose_result.detections)

    return VideoAnalysisResult(
        detections=pose_result.detections,
        detections_by_frame=pose_result.detections_by_frame,
        avg_torso_height=avg_torso,
        fps=metadata.fps,
        video_width=metadata.width,
        video_height=metadata.height,
        frame_count=frame_count,
        body_half_width=half_width,
        body_half_height=half_height,
    )


@dataclass
class CropPlan:
    """Everything needed to crop a video, computed without touching a frame.

    Separating the geometry from the pixels is what lets cropping run as a
    stream, and it makes the arithmetic testable without fabricating frames.

    There is deliberately no frame count beside `positions`: the two once
    disagreed when a requested range ran past the end of a file, which skewed
    multi-video durations. The positions are the only count.

    `output_height` stays the height of the image alone; the band rows are kept
    beside it so it never has to be recovered from an already rounded total.
    """

    output_width: int
    output_height: int
    scaled_width: int
    scaled_height: int
    scale_factor: float
    needs_padding: bool
    positions: list[ProcessedPosition]
    fps: float
    debug: bool = False
    target_width: int | None = None
    target_height: int | None = None
    label_lines: tuple[LabelLine, ...] = ()

    @property
    def frame_height(self) -> int:
        """Height of an emitted frame before scaling: image plus label band."""
        return self.output_height + band_height(self.label_lines)

    @property
    def final_size(self) -> tuple[int, int]:
        """(width, height) of emitted frames, after any requested scaling.

        Measured by scaling a probe rather than recomputing the arithmetic, so
        it cannot drift from scale_frame's own even-dimension rounding.
        """
        if self.target_width is None and self.target_height is None:
            return self.output_width, self.frame_height
        probe = np.zeros((self.frame_height, self.output_width, 3), dtype=np.uint8)
        scaled = scale_frame(probe, self.target_width, self.target_height)
        return scaled.shape[1], scaled.shape[0]


def build_crop_plan(
    analysis: VideoAnalysisResult,
    target_torso_ratio: float,
    debug: bool = False,
    target_width: int | None = None,
    target_height: int | None = None,
    label_lines: Sequence[LabelLine] = (),
) -> CropPlan:
    """Compute the crop geometry from an analysis, without reading any frame.

    Uses a scale-then-crop approach: scale the whole frame so the torso reaches
    the target size, then crop around the pelvis, padding if the scaled frame
    is smaller than the crop.

    Args:
        analysis: Analysis of the video to plan.
        target_torso_ratio: Torso height wanted, as a fraction of the image height.
        debug: Whether cropped frames get the pose overlay.
        target_width: Requested output width, or None.
        target_height: Requested output height, or None.
        label_lines: Rows of the band drawn under the image, top to bottom;
            empty for no band.

    Returns:
        The crop plan for this video.
    """
    output_width, output_height = calculate_crop_dimensions(
        analysis.video_width, analysis.video_height
    )

    if analysis.avg_torso_height > 0:
        torso_px_source = analysis.avg_torso_height * analysis.video_height
        torso_px_target = target_torso_ratio * output_height
        scale_factor = max(0.1, min(torso_px_target / torso_px_source, MAX_ZOOM_FACTOR))
    else:
        scale_factor = 1.0

    scaled_width = int(analysis.video_width * scale_factor)
    scaled_height = int(analysis.video_height * scale_factor)

    positions = interpolate_positions(analysis.detections)
    positions = smooth_positions_zero_phase(positions)

    return CropPlan(
        output_width=output_width,
        output_height=output_height,
        scaled_width=scaled_width,
        scaled_height=scaled_height,
        scale_factor=scale_factor,
        needs_padding=scaled_width < output_width or scaled_height < output_height,
        positions=positions,
        fps=analysis.fps,
        debug=debug,
        target_width=target_width,
        target_height=target_height,
        label_lines=tuple(label_lines),
    )


def print_crop_plan(plan: CropPlan, analysis: VideoAnalysisResult, target_ratio: float) -> None:
    """Report the planned geometry, as the pipeline has always done."""
    print(f"  Source video: {analysis.video_width}x{analysis.video_height}")
    avg_torso = analysis.avg_torso_height
    print(f"  Avg torso in source: {avg_torso:.3f} ({avg_torso:.1%})")
    print(f"  Scale factor: {plan.scale_factor:.2f}x (target torso: {target_ratio:.1%})")
    print(f"  Output size: {plan.output_width}x{plan.output_height}")
    print(f"  Scaled frame: {plan.scaled_width}x{plan.scaled_height}")
    if plan.label_lines:
        band = plan.frame_height - plan.output_height
        print(
            f"  Label band: {band}px over {len(plan.label_lines)} line(s) "
            f"(frame {plan.output_width}x{plan.frame_height})"
        )
    if plan.needs_padding:
        print("  Padding will be added (scaled frame smaller than output)")
    torso_px = analysis.avg_torso_height * analysis.video_height * plan.scale_factor
    print(f"  Torso in output: {torso_px / plan.output_height:.1%} ({torso_px:.1f}px)")


def _crop_one_frame(
    plan: CropPlan,
    frame: np.ndarray,
    pos: ProcessedPosition,
    detections_by_frame: dict[int, DetectionResult] | None,
) -> np.ndarray:
    """Scale, crop and optionally annotate a single frame."""
    if abs(plan.scale_factor - 1.0) > 0.01:
        scaled_frame = cv2.resize(
            frame,
            (plan.scaled_width, plan.scaled_height),
            interpolation=resize_interpolation(plan.scale_factor),
        )
    else:
        scaled_frame = frame

    pelvis_x = int(pos.x * plan.scaled_width)
    pelvis_y = int(pos.y * plan.scaled_height)
    ideal_x = pelvis_x - plan.output_width // 2
    ideal_y = pelvis_y - plan.output_height // 2

    x, y = ideal_x, ideal_y
    if plan.needs_padding:
        if len(scaled_frame.shape) == 3:
            output_frame = np.zeros(
                (plan.output_height, plan.output_width, scaled_frame.shape[2]),
                dtype=scaled_frame.dtype,
            )
        else:
            output_frame = np.zeros(
                (plan.output_height, plan.output_width), dtype=scaled_frame.dtype
            )

        src_x1 = max(0, ideal_x)
        src_y1 = max(0, ideal_y)
        src_x2 = min(plan.scaled_width, ideal_x + plan.output_width)
        src_y2 = min(plan.scaled_height, ideal_y + plan.output_height)

        dst_x1 = src_x1 - ideal_x
        dst_y1 = src_y1 - ideal_y
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)

        output_frame[dst_y1:dst_y2, dst_x1:dst_x2] = scaled_frame[src_y1:src_y2, src_x1:src_x2]
        cropped = output_frame
    else:
        x = max(0, min(ideal_x, plan.scaled_width - plan.output_width))
        y = max(0, min(ideal_y, plan.scaled_height - plan.output_height))
        cropped = scaled_frame[y : y + plan.output_height, x : x + plan.output_width]

    if plan.debug and detections_by_frame is not None:
        crop_region = CropRegion(x=x, y=y, width=plan.output_width, height=plan.output_height)
        cropped = draw_pose_overlay_on_crop(
            cropped,
            detections_by_frame[pos.frame_num],
            crop_region,
            plan.scaled_width,
            plan.scaled_height,
        )

    # Before the output scaling, so the band keeps its proportion at any size
    cropped = append_label_band(cropped, plan.label_lines)

    if plan.target_width is not None or plan.target_height is not None:
        cropped = scale_frame(cropped, plan.target_width, plan.target_height)

    return cropped


def crop_frames(
    plan: CropPlan,
    video_input: VideoInput,
    detections_by_frame: dict[int, DetectionResult] | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> Iterator[np.ndarray]:
    """Yield cropped frames, re-reading the source video.

    This is the second decode pass. Cropping cannot happen during detection
    because it needs the smoothed position, which needs every detection first.

    The frame number of each decoded frame is checked against the position
    about to be applied. A mismatch means the two passes disagree about the
    range, which would otherwise mis-frame the whole output silently.

    Raises:
        VideoError: If a decoded frame does not match its planned position.
    """
    total = len(plan.positions)
    for index, (frame_num, frame) in enumerate(
        extract_frames(video_input.path, video_input.start_frame, video_input.end_frame)
    ):
        if index >= total:
            break
        pos = plan.positions[index]
        if pos.frame_num != frame_num:
            raise VideoError(
                f"Frame mismatch while cropping {video_input.path}: read frame {frame_num} "
                f"where the plan expects {pos.frame_num}. The source changed between passes."
            )
        yield _crop_one_frame(plan, frame, pos, detections_by_frame)
        if on_progress:
            on_progress(index + 1, total)


@dataclass
class VideoStream:
    """A planned video whose cropped frames can be produced on demand."""

    plan: CropPlan
    video_input: VideoInput
    detections_by_frame: dict[int, DetectionResult]

    def frames(self, on_progress: Callable[[int, int], None] | None = None) -> Iterator[np.ndarray]:
        """Produce the cropped frames, re-reading the source video."""
        return crop_frames(self.plan, self.video_input, self.detections_by_frame, on_progress)


def _band_row_ratios(video_inputs: Sequence[VideoInput]) -> list[float]:
    """Text sizes of the label band, as fractions of the image height.

    The layout belongs to the render, not to a single video: composition
    normalizes heights by scaling, so a video left without a row would have its
    image scaled by another factor and lose the torso normalization.

    Args:
        video_inputs: Every video of the render.

    Returns:
        One ratio per row, top to bottom; empty when no video has any text.
    """
    if any(v.sublabel for v in video_inputs):
        return [LABEL_FONT_RATIO, SUBLABEL_FONT_RATIO]
    if any(v.label for v in video_inputs):
        return [LABEL_FONT_RATIO]
    return []


def _label_lines(
    video_input: VideoInput, image_height: int, ratios: Sequence[float]
) -> tuple[LabelLine, ...]:
    """Build this video's band lines for the render's line layout.

    Args:
        video_input: Video whose texts fill the rows.
        image_height: Height of the cropped image the band sits under.
        ratios: Text size ratios shared by every video of the render.

    Returns:
        The lines, with an empty text where this video has none.
    """
    texts = (video_input.label, video_input.sublabel)
    return tuple(
        LabelLine(text, font_size(image_height, ratio)) for text, ratio in zip(texts, ratios)
    )


def _fitting_torso_ratio(analysis: VideoAnalysisResult) -> float:
    """Largest torso ratio that still keeps the whole climber inside the crop.

    Args:
        analysis: Analysis of the video, carrying the climber's reach.

    Returns:
        The torso height, as a fraction of the output height, that the zoom
        must not exceed. Unbounded when nothing was measured.
    """
    output_width, output_height = calculate_crop_dimensions(
        analysis.video_width, analysis.video_height
    )
    if analysis.avg_torso_height <= 0:
        return float("inf")

    zoom = max_zoom_keeping_body_in_frame(
        analysis.body_half_width,
        analysis.body_half_height,
        analysis.video_width,
        analysis.video_height,
        output_width,
        output_height,
    )
    return zoom * analysis.avg_torso_height * analysis.video_height / output_height


def _target_torso_ratio(analyses: Sequence[VideoAnalysisResult], is_multi_video: bool) -> float:
    """Torso height to zoom every video to, as a fraction of the output height.

    A single video cannot be zoomed out below its measured torso without
    padding the frame, so it never targets less than what it already shows.
    Every video then shares the most demanding framing limit: capping only the
    video that needs it would leave the climbers at different sizes, which is
    exactly what the zoom normalization exists to prevent.

    Args:
        analyses: Analysis of every video of the render.
        is_multi_video: Whether the videos are composed side by side.

    Returns:
        The shared target torso ratio.
    """
    measured = max(analysis.avg_torso_height for analysis in analyses)
    wanted = TORSO_HEIGHT_RATIO if is_multi_video else max(TORSO_HEIGHT_RATIO, measured)
    return min(wanted, *(_fitting_torso_ratio(analysis) for analysis in analyses))


def plan_videos(
    video_inputs: list[VideoInput],
    detector: PoseDetector,
    debug: bool = False,
    target_width: int | None = None,
    target_height: int | None = None,
) -> list[VideoStream]:
    """Analyze each video in turn and plan its crop, without cropping yet.

    Analysis retains no frames, so this holds only detections and geometry
    whatever the clip length. Every video is analyzed before any is planned:
    the zoom target is shared, so that a video whose climber reaches far does
    not end up framed at a different scale from the others.
    """
    is_multi_video = len(video_inputs) > 1
    row_ratios = _band_row_ratios(video_inputs)

    analyses = []
    for i, video_input in enumerate(video_inputs):
        if is_multi_video:
            print(f"\n--- Video {i + 1}/{len(video_inputs)}: {video_input.path.name} ---")
        analyses.append(analyze_video(video_input, detector))

    target_torso = _target_torso_ratio(analyses, is_multi_video)
    wanted = TORSO_HEIGHT_RATIO if is_multi_video else None
    if wanted is not None and target_torso < wanted:
        print(
            f"\nZoom capped to {target_torso:.1%} torso (from {wanted:.1%}) "
            "to keep every climber inside the frame"
        )

    streams: list[VideoStream] = []
    for video_input, analysis in zip(video_inputs, analyses):
        _, image_height = calculate_crop_dimensions(analysis.video_width, analysis.video_height)
        label_lines = _label_lines(video_input, image_height, row_ratios)

        if is_multi_video and analysis.avg_torso_height > target_torso:
            torso_pct = analysis.avg_torso_height
            print(f"  Torso ({torso_pct:.1%}) > target ({target_torso:.1%}) - padding")

        plan = build_crop_plan(
            analysis,
            target_torso,
            debug=debug,
            target_width=None if is_multi_video else target_width,
            target_height=None if is_multi_video else target_height,
            label_lines=label_lines,
        )

        print_crop_plan(plan, analysis, target_torso)
        streams.append(
            VideoStream(
                plan=plan,
                video_input=video_input,
                detections_by_frame=analysis.detections_by_frame,
            )
        )

    return streams


@dataclass
class OutputStream:
    """The frames to encode, with what the encoder needs to know up front."""

    frames: Iterator[np.ndarray]
    width: int
    height: int
    total_frames: int
    fps: float


def _measure(
    frame_producer: Callable[[list[np.ndarray]], np.ndarray], sizes: list[tuple[int, int]]
) -> tuple[int, int]:
    """Run the composition on blank frames to learn the output size.

    Measuring beats recomputing the arithmetic: the result cannot drift from
    the height normalization and even-dimension rounding actually applied.
    """
    probes = [np.zeros((h, w, 3), dtype=np.uint8) for w, h in sizes]
    composed = frame_producer(probes)
    return composed.shape[1], composed.shape[0]


def build_output_stream(
    streams: list[VideoStream],
    target_width: int | None = None,
    target_height: int | None = None,
) -> OutputStream:
    """Build the stream of output frames, cropping lazily as it is consumed.

    For a single video the cropped frames are the output. For several, each
    video's crop stream is pulled through a cursor driven by the time
    synchronization indices, and one composed row is emitted per output frame.
    """
    if len(streams) == 1:
        plan = streams[0].plan
        width, height = plan.final_size
        return OutputStream(
            frames=streams[0].frames(),
            width=width,
            height=height,
            total_frames=len(plan.positions),
            fps=plan.fps,
        )

    print("\n=== Synchronizing and composing ===")

    # Durations and the sync map both come from the frames actually decoded.
    # The requested range can still overstate it when container metadata is
    # wrong, and a duration longer than the frames that exist would stretch the
    # render by freezing on the last frame.
    decoded_counts = [len(s.plan.positions) for s in streams]
    fps_list = [s.plan.fps for s in streams]
    durations = [calculate_duration(fc, fps) for fc, fps in zip(decoded_counts, fps_list)]
    max_duration = calculate_max_duration(decoded_counts, fps_list)

    # LCM of all FPS gives each source frame an integer number of output
    # frames, which is what keeps slow motion free of judder
    output_fps = calculate_output_fps(fps_list)
    index_maps = [
        time_sync_frame_indices(count, fps, max_duration, output_fps)
        for count, fps in zip(decoded_counts, fps_list)
    ]
    target_frame_count = min(len(indices) for indices in index_maps)

    print(f"  Max duration: {max_duration:.2f}s")
    fps_str = ", ".join(f"{f:.0f}" for f in fps_list)
    print(f"  Output FPS: {output_fps:.0f} (LCM of {fps_str})")
    print(f"  Target frame count: {target_frame_count}")

    row_height = max(s.plan.frame_height for s in streams)
    row_height -= row_height % 2

    for i, stream in enumerate(streams):
        freeze = " (freezes)" if durations[i] < max_duration else ""
        print(
            f"  Video {i + 1}: {stream.plan.output_width}x{stream.plan.frame_height}, "
            f"{durations[i]:.2f}s @ {fps_list[i]:.1f}fps{freeze}"
        )

    def compose(parts: list[np.ndarray]) -> np.ndarray:
        row = compose_frame_row(parts, row_height)
        if target_width is not None or target_height is not None:
            row = scale_frame(row, target_width, target_height)
        return row

    width, height = _measure(compose, [(s.plan.output_width, s.plan.frame_height) for s in streams])

    def rows() -> Iterator[np.ndarray]:
        cursors = [FrameCursor(s.frames()) for s in streams]
        for output_index in range(target_frame_count):
            parts = [
                cursor.advance_to(indices[output_index])
                for cursor, indices in zip(cursors, index_maps)
            ]
            yield compose(parts)

    return OutputStream(
        frames=rows(),
        width=width,
        height=height,
        total_frames=target_frame_count,
        fps=output_fps,
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

    if args.imgsz is not None and args.imgsz < MIN_IMGSZ:
        print(f"Error: --imgsz must be at least {MIN_IMGSZ}.", file=sys.stderr)
        return 1

    # Parse and validate inputs
    video_inputs = validate_args(args)
    is_multi_video = len(video_inputs) > 1

    try:
        # Analyze and plan every video; no frames are retained
        print(f"Processing {len(video_inputs)} video(s)...")

        with PoseDetector(model_size=args.model, imgsz=args.imgsz) as detector:
            streams = plan_videos(
                video_inputs,
                detector,
                debug=args.debug,
                target_width=args.width,
                target_height=args.height,
            )

        output = build_output_stream(streams, args.width, args.height)
        avg_fps = output.fps

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

        # Cropping and composition happen lazily as FFmpeg consumes frames, so
        # this one progress bar covers the whole output phase
        generate_output_stream(
            output.frames,
            output_path,
            effective_fps,
            output.width,
            output.height,
            output.total_frames,
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
            print(f"  Output frames: {output.total_frames}")

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
