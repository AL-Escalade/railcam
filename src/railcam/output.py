"""Video output generation using FFmpeg (MP4 and GIF)."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from enum import Enum
from pathlib import Path
from typing import Callable

import cv2
import numpy as np


class OutputFormat(Enum):
    """Supported output formats."""

    MP4 = "mp4"
    GIF = "gif"


class FFmpegNotFoundError(Exception):
    """FFmpeg is not installed or not in PATH."""


class OutputGenerationError(Exception):
    """Error during output generation."""


# Keep old exception name for backward compatibility
GifGenerationError = OutputGenerationError


def check_ffmpeg() -> None:
    """Check if FFmpeg is available."""
    if shutil.which("ffmpeg") is None:
        raise FFmpegNotFoundError(
            "FFmpeg not found. Please install FFmpeg:\n"
            "  macOS: brew install ffmpeg\n"
            "  Ubuntu/Debian: sudo apt install ffmpeg\n"
            "  Windows: https://ffmpeg.org/download.html"
        )


def _write_frames_to_temp(
    frames: list[np.ndarray],
    temp_path: Path,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> Path:
    """Write frames to temporary PNG files.

    Returns the frame pattern path for FFmpeg.
    """
    frame_pattern = temp_path / "frame_%05d.png"
    total_frames = len(frames)

    for i, frame in enumerate(frames):
        frame_file = temp_path / f"frame_{i:05d}.png"
        # OpenCV uses BGR, write directly
        cv2.imwrite(str(frame_file), frame)

        if on_progress:
            on_progress(i + 1, total_frames, "Writing frames")

    return frame_pattern


def generate_gif(
    frames: list[np.ndarray],
    output_path: Path,
    fps: float,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> None:
    """Generate an optimized GIF from a list of frames.

    Uses FFmpeg's palette generation for high-quality output.
    """
    check_ffmpeg()

    if not frames:
        raise OutputGenerationError("No frames to process")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Write frames to temporary PNG files
        frame_pattern = _write_frames_to_temp(frames, temp_path, on_progress)
        total_frames = len(frames)

        # Generate palette for optimal colors
        palette_path = temp_path / "palette.png"
        palette_cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(frame_pattern),
            "-vf",
            "palettegen=stats_mode=diff",
            str(palette_path),
        ]

        result = subprocess.run(
            palette_cmd,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise OutputGenerationError(f"Palette generation failed: {result.stderr}")

        if on_progress:
            on_progress(total_frames, total_frames, "Generating palette")

        # Generate GIF using the palette
        gif_cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(frame_pattern),
            "-i",
            str(palette_path),
            "-lavfi",
            "paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle",
            str(output_path),
        ]

        result = subprocess.run(
            gif_cmd,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise OutputGenerationError(f"GIF generation failed: {result.stderr}")

        if on_progress:
            on_progress(total_frames, total_frames, "Complete")


def generate_mp4(
    frames: list[np.ndarray],
    output_path: Path,
    fps: float,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> None:
    """Generate an MP4 video from a list of frames.

    Uses H.264 encoding with yuv420p pixel format for broad compatibility.
    """
    check_ffmpeg()

    if not frames:
        raise OutputGenerationError("No frames to process")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Write frames to temporary PNG files
        frame_pattern = _write_frames_to_temp(frames, temp_path, on_progress)
        total_frames = len(frames)

        if on_progress:
            on_progress(total_frames, total_frames, "Encoding MP4")

        # Generate MP4 using H.264
        mp4_cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(frame_pattern),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]

        result = subprocess.run(
            mp4_cmd,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise OutputGenerationError(f"MP4 generation failed: {result.stderr}")

        if on_progress:
            on_progress(total_frames, total_frames, "Complete")


def generate_output(
    frames: list[np.ndarray],
    output_path: Path,
    fps: float,
    output_format: OutputFormat = OutputFormat.MP4,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> None:
    """Generate output in the specified format.

    Args:
        frames: List of frames to encode.
        output_path: Path to write the output file.
        fps: Frames per second for the output.
        output_format: Output format (MP4 or GIF).
        on_progress: Optional progress callback.
    """
    if output_format == OutputFormat.GIF:
        generate_gif(frames, output_path, fps, on_progress)
    else:
        generate_mp4(frames, output_path, fps, on_progress)


def get_output_path(
    input_path: Path,
    output: Path | None,
    output_format: OutputFormat = OutputFormat.MP4,
) -> Path:
    """Determine the output path with correct extension.

    If output is None, generates a default path based on input filename.
    If output has wrong extension, corrects it to match the format.
    """
    extension = f".{output_format.value}"

    if output is not None:
        # Correct extension if needed
        if output.suffix.lower() != extension:
            return output.with_suffix(extension)
        return output

    return Path.cwd() / f"{input_path.stem}{extension}"


def parse_output_format(format_str: str) -> OutputFormat:
    """Parse output format from string.

    Args:
        format_str: Format string ("mp4" or "gif").

    Returns:
        OutputFormat enum value.

    Raises:
        ValueError: If format is not supported.
    """
    try:
        return OutputFormat(format_str.lower())
    except ValueError as e:
        supported = ", ".join(f.value for f in OutputFormat)
        raise ValueError(f"Unsupported format '{format_str}'. Supported: {supported}") from e
