"""Tests for GUI render command building."""

from __future__ import annotations

import shlex
from pathlib import Path

import pytest

from railcam.cli import create_parser, validate_args
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

    assert "'my videos/run 1.mp4:0:10'" in command
    assert "'my output.mp4'" in command
    assert shlex.split(command) == ["railcam", *build_cli_args(project)]


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


def test_build_args_omits_label_when_no_video_is_labeled() -> None:
    project = Project(
        videos=[
            VideoEntry(path=Path("a.mp4"), start_frame=0, end_frame=10),
            VideoEntry(path=Path("b.mp4"), start_frame=0, end_frame=10),
        ],
        render=RenderOptions(),
    )

    assert "--label" not in build_cli_args(project)


def test_build_args_emits_one_label_per_video_when_any_is_labeled() -> None:
    project = Project(
        videos=[
            VideoEntry(path=Path("a.mp4"), start_frame=0, end_frame=10, label="Dupont"),
            VideoEntry(path=Path("b.mp4"), start_frame=0, end_frame=10),
        ],
        render=RenderOptions(),
    )

    args = build_cli_args(project)

    assert args == [
        "-i",
        "a.mp4:0:10",
        "--label=Dupont",
        "-i",
        "b.mp4:0:10",
        "--label=",
    ]


def test_format_command_keeps_an_empty_label() -> None:
    project = Project(
        videos=[
            VideoEntry(path=Path("a.mp4"), start_frame=0, end_frame=10, label="Jean Dupont"),
            VideoEntry(path=Path("b.mp4"), start_frame=0, end_frame=10),
        ],
        render=RenderOptions(),
    )

    command = format_cli_command(project)

    assert command == "railcam -i a.mp4:0:10 '--label=Jean Dupont' -i b.mp4:0:10 --label="
    assert shlex.split(command) == ["railcam", *build_cli_args(project)]


def _labeled_project(label: str) -> Project:
    return Project(
        videos=[VideoEntry(path=Path("a.mp4"), start_frame=0, end_frame=10, label=label)],
        render=RenderOptions(),
    )


@pytest.mark.parametrize(
    "label", ["-Dupont", "O'Brien", 'Jean "Jojo" Dupont', "Run 2: final", "a;rm -rf x"]
)
def test_displayed_command_survives_a_shell_round_trip(label: str) -> None:
    project = _labeled_project(label)

    command = format_cli_command(project)

    assert shlex.split(command) == ["railcam", *build_cli_args(project)]


@pytest.mark.parametrize("label", ["-Dupont", "--climber", "O'Brien", "Jean Dupont"])
def test_built_args_are_parsed_back_as_labels_by_the_cli(label: str) -> None:
    args = build_cli_args(_labeled_project(label))

    inputs = validate_args(create_parser().parse_args(args))

    assert [video.label for video in inputs] == [label]


class TestDetectionResolutionArgs:
    def test_omitted_when_auto(self) -> None:
        project = Project(videos=[], render=RenderOptions())

        assert "--imgsz" not in build_cli_args(project)

    def test_emitted_when_set(self) -> None:
        project = Project(videos=[], render=RenderOptions(imgsz=1280))
        args = build_cli_args(project)

        assert args[args.index("--imgsz") + 1] == "1280"
