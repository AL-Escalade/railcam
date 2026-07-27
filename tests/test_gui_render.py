"""Tests for GUI render command building."""

from __future__ import annotations

from pathlib import Path

from railcam.gui.project import Project, RenderOptions, VideoEntry
from railcam.gui.render import build_cli_args, format_cli_command


def test_build_args_single_video_defaults() -> None:
    project = Project(
        videos=[VideoEntry(path=Path("a.mp4"), start_frame=100, end_frame=250, climber="auto")],
        render=RenderOptions(),
    )

    args = build_cli_args(project)

    assert args == ["-i", "a.mp4:100:250"]


def test_build_args_includes_climber_when_not_auto() -> None:
    project = Project(
        videos=[
            VideoEntry(path=Path("a.mp4"), start_frame=100, end_frame=250, climber="auto"),
            VideoEntry(path=Path("b.mp4"), start_frame=60, end_frame=210, climber="left"),
        ],
        render=RenderOptions(),
    )

    args = build_cli_args(project)

    assert args == ["-i", "a.mp4:100:250", "-i", "b.mp4:60:210:left"]


def test_build_args_includes_non_default_options() -> None:
    project = Project(
        videos=[VideoEntry(path=Path("a.mp4"), start_frame=0, end_frame=10, climber="auto")],
        render=RenderOptions(format="gif", height=720, speed=0.5, debug=True),
        output_path=Path("out.gif"),
    )

    args = build_cli_args(project)

    assert args == [
        "-i",
        "a.mp4:0:10",
        "-f",
        "gif",
        "-H",
        "720",
        "-s",
        "0.5",
        "--debug",
        "-o",
        "out.gif",
    ]


def test_format_command_prefixes_program_name() -> None:
    project = Project(
        videos=[VideoEntry(path=Path("a.mp4"), start_frame=0, end_frame=10, climber="auto")],
        render=RenderOptions(),
    )

    command = format_cli_command(project)

    assert command == "railcam -i a.mp4:0:10"


def test_format_command_quotes_paths_with_spaces() -> None:
    project = Project(
        videos=[
            VideoEntry(
                path=Path("my videos/run 1.mp4"), start_frame=0, end_frame=10, climber="auto"
            )
        ],
        render=RenderOptions(),
        output_path=Path("my output.mp4"),
    )

    command = format_cli_command(project)

    assert '"my videos/run 1.mp4:0:10"' in command
    assert '"my output.mp4"' in command


def test_build_args_omits_model_at_default() -> None:
    project = Project(
        videos=[VideoEntry(path=Path("a.mp4"), start_frame=0, end_frame=10)],
        render=RenderOptions(model="s"),
    )

    assert "--model" not in build_cli_args(project)


def test_build_args_includes_non_default_model() -> None:
    project = Project(
        videos=[VideoEntry(path=Path("a.mp4"), start_frame=0, end_frame=10)],
        render=RenderOptions(model="n"),
    )

    assert build_cli_args(project) == ["-i", "a.mp4:0:10", "--model", "n"]
