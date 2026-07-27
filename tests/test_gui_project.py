"""Tests for GUI project file serialization."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from railcam.gui.project import (
    MODEL_LABELS,
    VALID_MODELS,
    Project,
    ProjectError,
    RenderOptions,
    VideoEntry,
)


def make_project(video_path: Path) -> Project:
    """Build a representative project for tests."""
    return Project(
        videos=[
            VideoEntry(path=video_path, start_frame=100, end_frame=250, climber="auto"),
            VideoEntry(path=video_path, start_frame=60, end_frame=210, climber="left"),
        ],
        render=RenderOptions(format="gif", height=720, speed=0.5, debug=True),
        output_path=Path("comparison.gif"),
    )


def test_round_trip_preserves_all_fields(tmp_path: Path) -> None:
    video = tmp_path / "a.mp4"
    project = make_project(video)
    project_file = tmp_path / "session.railcam.json"

    project.save(project_file)
    loaded = Project.load(project_file)

    assert loaded == project


def test_video_paths_stored_relative_to_project_file(tmp_path: Path) -> None:
    video = tmp_path / "videos" / "a.mp4"
    project = Project(
        videos=[VideoEntry(path=video, start_frame=0, end_frame=10, climber="auto")],
        render=RenderOptions(),
    )
    project_file = tmp_path / "session.railcam.json"

    project.save(project_file)

    raw = json.loads(project_file.read_text(encoding="utf-8"))
    stored = raw["videos"][0]["path"]
    assert not Path(stored).is_absolute()
    # Loading resolves back to the absolute path
    assert Project.load(project_file).videos[0].path == video


def test_video_path_on_other_drive_stored_absolute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Simulate os.path.relpath failing (e.g. different drive on Windows)
    import railcam.gui.project as project_module

    def raise_value_error(*args: object, **kwargs: object) -> str:
        raise ValueError("path is on mount 'D:', start on mount 'C:'")

    monkeypatch.setattr(project_module.os.path, "relpath", raise_value_error)

    video = tmp_path / "a.mp4"
    project = Project(
        videos=[VideoEntry(path=video, start_frame=0, end_frame=10, climber="auto")],
        render=RenderOptions(),
    )
    project_file = tmp_path / "session.railcam.json"

    project.save(project_file)

    raw = json.loads(project_file.read_text(encoding="utf-8"))
    assert Path(raw["videos"][0]["path"]).is_absolute()
    assert Project.load(project_file).videos[0].path == video


def test_render_options_defaults() -> None:
    options = RenderOptions()

    assert options.format == "mp4"
    assert options.height is None
    assert options.speed == 1.0
    assert options.debug is False


def test_load_rejects_corrupt_json(tmp_path: Path) -> None:
    project_file = tmp_path / "broken.railcam.json"
    project_file.write_text("{not json", encoding="utf-8")

    with pytest.raises(ProjectError, match="[Ii]nvalid"):
        Project.load(project_file)


def test_load_rejects_unsupported_version(tmp_path: Path) -> None:
    project_file = tmp_path / "future.railcam.json"
    project_file.write_text(json.dumps({"version": 999, "videos": []}), encoding="utf-8")

    with pytest.raises(ProjectError, match="[Vv]ersion"):
        Project.load(project_file)


def test_load_rejects_missing_fields(tmp_path: Path) -> None:
    project_file = tmp_path / "partial.railcam.json"
    project_file.write_text(json.dumps({"version": 1}), encoding="utf-8")

    with pytest.raises(ProjectError):
        Project.load(project_file)


def test_load_rejects_invalid_climber(tmp_path: Path) -> None:
    project_file = tmp_path / "bad-climber.railcam.json"
    content = {
        "version": 1,
        "videos": [{"path": "a.mp4", "start_frame": 0, "end_frame": 10, "climber": "middle"}],
        "render": {},
    }
    project_file.write_text(json.dumps(content), encoding="utf-8")

    with pytest.raises(ProjectError, match="climber"):
        Project.load(project_file)


def test_default_model_is_small() -> None:
    assert RenderOptions().model == "s"


def test_round_trip_preserves_model(tmp_path: Path) -> None:
    project = Project(
        videos=[VideoEntry(path=tmp_path / "a.mp4", start_frame=0, end_frame=10)],
        render=RenderOptions(model="m"),
    )
    project_file = tmp_path / "session.railcam.json"

    project.save(project_file)

    assert Project.load(project_file).render.model == "m"


def test_project_written_before_model_existed_loads_with_default(tmp_path: Path) -> None:
    project_file = tmp_path / "session.railcam.json"
    project_file.write_text(
        json.dumps(
            {
                "version": 1,
                "videos": [{"path": "a.mp4", "start_frame": 0, "end_frame": 10}],
                "render": {"format": "mp4", "height": None, "speed": 1.0, "debug": False},
                "output_path": None,
            }
        ),
        encoding="utf-8",
    )

    assert Project.load(project_file).render.model == "s"


def test_unknown_model_raises(tmp_path: Path) -> None:
    project_file = tmp_path / "session.railcam.json"
    project_file.write_text(
        json.dumps(
            {
                "version": 1,
                "videos": [{"path": "a.mp4", "start_frame": 0, "end_frame": 10}],
                "render": {"model": "huge"},
                "output_path": None,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ProjectError, match="Invalid model size"):
        Project.load(project_file)


def test_every_model_size_has_a_label() -> None:
    labelled = [value for value, _ in MODEL_LABELS]
    assert labelled == list(VALID_MODELS)


def test_labels_name_the_cli_value() -> None:
    # The panel shows the equivalent CLI command, so the flag value the user
    # would copy must be readable from the label they picked.
    for value, label in MODEL_LABELS:
        assert f"({value})" in label
