"""Video output generation using FFmpeg (MP4 and GIF)."""

from __future__ import annotations

import contextlib
import shutil
import subprocess
import tempfile
from enum import Enum
from pathlib import Path
from typing import IO, Callable

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


def _raw_input_args(fps: float, width: int, height: int) -> list[str]:
    """FFmpeg arguments reading BGR frames from standard input.

    `-nostats -loglevel error` keeps the captured error stream small; frames are
    OpenCV BGR arrays, which map directly onto `rawvideo` with `bgr24`.
    """
    return [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-nostats",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-framerate",
        str(fps),
        "-i",
        "pipe:0",
    ]


def build_mp4_command(output_path: Path, fps: float, width: int, height: int) -> list[str]:
    """Build the FFmpeg command encoding piped frames to H.264 MP4."""
    return [
        *_raw_input_args(fps, width, height),
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


def build_gif_command(output_path: Path, fps: float, width: int, height: int) -> list[str]:
    """Build the FFmpeg command encoding piped frames to a palette-optimized GIF.

    The two-command palette flow reads its input twice, which a pipe cannot do.
    Splitting the stream inside one invocation keeps the same palette settings
    while reading standard input a single time.
    """
    palette_chain = (
        "[0:v]split[a][b];"
        "[a]palettegen=stats_mode=diff[p];"
        "[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle"
    )
    return [
        *_raw_input_args(fps, width, height),
        "-filter_complex",
        palette_chain,
        str(output_path),
    ]


def _read_stderr(process: subprocess.Popen[bytes], handle: IO[bytes]) -> bytes:
    """Wait for the process and return what it wrote to its error stream."""
    process.wait()
    handle.seek(0)
    return handle.read()


def _encode_frames(
    frames: list[np.ndarray],
    command: list[str],
    stage: str,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> None:
    """Pipe frames into FFmpeg, raising with its error output on failure.

    The error stream goes to an anonymous temporary file rather than a pipe:
    a pipe FFmpeg fills while we are still writing frames would deadlock.
    """
    total_frames = len(frames)

    with tempfile.TemporaryFile() as error_log:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=error_log,
        )
        stdin = process.stdin
        assert stdin is not None

        try:
            for i, frame in enumerate(frames):
                if process.poll() is not None:
                    # FFmpeg gave up early; stop rather than block on the pipe
                    break
                stdin.write(np.ascontiguousarray(frame).tobytes())
                if on_progress:
                    on_progress(i + 1, total_frames, stage)
        except (BrokenPipeError, OSError):
            # FFmpeg closed the pipe; its error output explains why
            pass
        finally:
            with contextlib.suppress(BrokenPipeError, OSError):
                stdin.close()

        stderr = _read_stderr(process, error_log)

    if process.returncode != 0:
        message = stderr.decode("utf-8", errors="replace").strip()
        raise OutputGenerationError(f"FFmpeg failed ({process.returncode}): {message}")


def _frame_size(frames: list[np.ndarray]) -> tuple[int, int]:
    """Return (width, height) of the frames, taken from the first one."""
    height, width = frames[0].shape[:2]
    return width, height


def generate_gif(
    frames: list[np.ndarray],
    output_path: Path,
    fps: float,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> None:
    """Generate an optimized GIF from a list of frames.

    Uses FFmpeg's palette generation for high-quality output.
    """
    if not frames:
        raise OutputGenerationError("No frames to process")
    check_ffmpeg()

    width, height = _frame_size(frames)
    _encode_frames(
        frames,
        build_gif_command(output_path, fps, width, height),
        "Encoding GIF",
        on_progress,
    )

    if on_progress:
        on_progress(len(frames), len(frames), "Complete")


def generate_mp4(
    frames: list[np.ndarray],
    output_path: Path,
    fps: float,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> None:
    """Generate an MP4 video from a list of frames.

    Uses H.264 encoding with yuv420p pixel format for broad compatibility.
    """
    if not frames:
        raise OutputGenerationError("No frames to process")
    check_ffmpeg()

    width, height = _frame_size(frames)
    _encode_frames(
        frames,
        build_mp4_command(output_path, fps, width, height),
        "Encoding MP4",
        on_progress,
    )

    if on_progress:
        on_progress(len(frames), len(frames), "Complete")


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
