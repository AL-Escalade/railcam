"""Tests for FFmpeg output generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from railcam import output
from railcam.output import (
    OutputFormat,
    OutputGenerationError,
    build_gif_command,
    build_mp4_command,
    generate_output,
)


def _frames(count: int = 2, width: int = 4, height: int = 3) -> list[np.ndarray]:
    return [np.full((height, width, 3), i, dtype=np.uint8) for i in range(count)]


class TestCommandBuilding:
    def test_mp4_reads_raw_frames_from_stdin(self) -> None:
        command = build_mp4_command(Path("out.mp4"), fps=30.0, width=640, height=384)

        assert "-f" in command and "rawvideo" in command
        assert "bgr24" in command
        assert "640x384" in command
        assert "pipe:0" in command

    def test_mp4_keeps_h264_settings(self) -> None:
        command = build_mp4_command(Path("out.mp4"), fps=30.0, width=640, height=384)

        assert "libx264" in command
        assert "yuv420p" in command
        assert command[-1] == "out.mp4"

    def test_gif_is_a_single_invocation_with_palette(self) -> None:
        command = build_gif_command(Path("out.gif"), fps=30.0, width=640, height=384)
        chain = command[command.index("-filter_complex") + 1]

        assert "split" in chain
        assert "palettegen" in chain
        assert "paletteuse" in chain

    def test_gif_preserves_palette_quality_settings(self) -> None:
        command = build_gif_command(Path("out.gif"), fps=30.0, width=640, height=384)
        chain = command[command.index("-filter_complex") + 1]

        assert "stats_mode=diff" in chain
        assert "dither=bayer" in chain
        assert "diff_mode=rectangle" in chain

    def test_no_output_command_names_an_image_sequence(self) -> None:
        for command in (
            build_mp4_command(Path("out.mp4"), fps=30.0, width=8, height=8),
            build_gif_command(Path("out.gif"), fps=30.0, width=8, height=8),
        ):
            assert not any(".png" in part for part in command)


class _FakeProcess:
    """Stands in for a running FFmpeg, capturing what is piped into it."""

    def __init__(self, returncode: int = 0, stderr: bytes = b"") -> None:
        self.written = bytearray()
        self.returncode = returncode
        self._stderr = stderr
        self.stdin = self

    # stdin surface
    def write(self, data: bytes) -> int:
        self.written.extend(data)
        return len(data)

    def close(self) -> None:
        return None

    # process surface
    def poll(self) -> int | None:
        return None

    def wait(self) -> int:
        return self.returncode


def _patch_ffmpeg(monkeypatch, process: _FakeProcess) -> list[list[str]]:
    """Run without a real FFmpeg, recording the commands that would run."""
    commands: list[list[str]] = []

    def fake_popen(command, **kwargs):
        commands.append(command)
        return process

    monkeypatch.setattr(output, "check_ffmpeg", lambda: None)
    monkeypatch.setattr(output.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(output, "_read_stderr", lambda _process, _handle: process._stderr)
    return commands


class TestEncoding:
    def test_frames_are_piped_as_raw_bytes(self, monkeypatch, tmp_path) -> None:
        process = _FakeProcess()
        _patch_ffmpeg(monkeypatch, process)
        frames = _frames(count=2, width=4, height=3)

        generate_output(frames, tmp_path / "out.mp4", 30.0, OutputFormat.MP4)

        assert bytes(process.written) == frames[0].tobytes() + frames[1].tobytes()

    def test_frame_size_is_taken_from_the_frames(self, monkeypatch, tmp_path) -> None:
        commands = _patch_ffmpeg(monkeypatch, _FakeProcess())

        generate_output(_frames(count=1, width=4, height=3), tmp_path / "o.mp4", 30.0)

        assert "4x3" in commands[0]

    def test_progress_reported_per_frame(self, monkeypatch, tmp_path) -> None:
        _patch_ffmpeg(monkeypatch, _FakeProcess())
        seen: list[int] = []

        generate_output(
            _frames(count=3),
            tmp_path / "o.mp4",
            30.0,
            on_progress=lambda current, total, stage: seen.append(current),
        )

        assert seen[:3] == [1, 2, 3]

    def test_ffmpeg_failure_surfaces_its_error_output(self, monkeypatch, tmp_path) -> None:
        _patch_ffmpeg(monkeypatch, _FakeProcess(returncode=1, stderr=b"codec not found"))

        with pytest.raises(OutputGenerationError, match="codec not found"):
            generate_output(_frames(), tmp_path / "o.mp4", 30.0)

    def test_empty_frames_start_no_process(self, monkeypatch, tmp_path) -> None:
        commands = _patch_ffmpeg(monkeypatch, _FakeProcess())

        with pytest.raises(OutputGenerationError, match="No frames"):
            generate_output([], tmp_path / "o.mp4", 30.0)

        assert commands == []
