"""Tests for GUI project file serialization."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from railcam.gui.project import (
    MODEL_LABELS,
    SLOWMO_MAX,
    SLOWMO_MIN,
    VALID_MODELS,
    Project,
    ProjectError,
    RenderOptions,
    VideoEntry,
    clamp_range,
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


def test_round_trip_preserves_labels(tmp_path: Path) -> None:
    project = Project(
        videos=[
            VideoEntry(path=tmp_path / "a.mp4", start_frame=0, end_frame=10, label="Dupont"),
            VideoEntry(path=tmp_path / "b.mp4", start_frame=0, end_frame=10),
        ],
        render=RenderOptions(),
    )
    project_file = tmp_path / "session.railcam.json"

    project.save(project_file)
    loaded = Project.load(project_file)

    assert [video.label for video in loaded.videos] == ["Dupont", ""]


def test_project_written_before_labels_existed_loads_with_empty_labels(tmp_path: Path) -> None:
    project_file = tmp_path / "session.railcam.json"
    project_file.write_text(
        json.dumps(
            {
                "version": 1,
                "videos": [{"path": "a.mp4", "start_frame": 0, "end_frame": 10}],
                "render": {},
                "output_path": None,
            }
        ),
        encoding="utf-8",
    )

    assert Project.load(project_file).videos[0].label == ""


def test_load_rejects_non_string_label(tmp_path: Path) -> None:
    project_file = tmp_path / "bad-label.railcam.json"
    project_file.write_text(
        json.dumps(
            {
                "version": 1,
                "videos": [{"path": "a.mp4", "start_frame": 0, "end_frame": 10, "label": 42}],
                "render": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ProjectError, match="label"):
        Project.load(project_file)


class TestDetectionResolution:
    def test_round_trip(self, tmp_path) -> None:
        target = tmp_path / "session.railcam.json"
        Project(videos=[], render=RenderOptions(imgsz=1920)).save(target)

        assert Project.load(target).render.imgsz == 1920

    def test_absent_field_means_auto(self, tmp_path) -> None:
        target = tmp_path / "old.railcam.json"
        target.write_text(
            json.dumps({"version": 1, "videos": [], "render": {"model": "s"}}),
            encoding="utf-8",
        )

        assert Project.load(target).render.imgsz is None

    def test_below_the_minimum_is_rejected(self, tmp_path) -> None:
        target = tmp_path / "bad.railcam.json"
        target.write_text(
            json.dumps({"version": 1, "videos": [], "render": {"imgsz": 128}}),
            encoding="utf-8",
        )

        with pytest.raises(ProjectError):
            Project.load(target)


def test_round_trip_preserves_sublabels(tmp_path: Path) -> None:
    project = Project(
        videos=[
            VideoEntry(
                path=tmp_path / "a.mp4",
                start_frame=0,
                end_frame=10,
                label="Dupont",
                sublabel="4.704",
            ),
            VideoEntry(path=tmp_path / "b.mp4", start_frame=0, end_frame=10),
        ],
        render=RenderOptions(),
    )
    project_file = tmp_path / "session.railcam.json"

    project.save(project_file)
    loaded = Project.load(project_file)

    assert [video.sublabel for video in loaded.videos] == ["4.704", ""]
    assert loaded == project


def test_project_written_before_sublabels_existed_loads_with_empty_sublabels(
    tmp_path: Path,
) -> None:
    project_file = tmp_path / "session.railcam.json"
    project_file.write_text(
        json.dumps(
            {
                "version": 1,
                "videos": [{"path": "a.mp4", "start_frame": 0, "end_frame": 10, "label": "Dupont"}],
                "render": {},
                "output_path": None,
            }
        ),
        encoding="utf-8",
    )

    loaded = Project.load(project_file)

    assert loaded.videos[0].label == "Dupont"
    assert loaded.videos[0].sublabel == ""


def test_load_rejects_non_string_sublabel(tmp_path: Path) -> None:
    project_file = tmp_path / "bad-sublabel.railcam.json"
    project_file.write_text(
        json.dumps(
            {
                "version": 1,
                "videos": [{"path": "a.mp4", "start_frame": 0, "end_frame": 10, "sublabel": 4.7}],
                "render": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ProjectError, match="sublabel"):
        Project.load(project_file)


class TestClampRange:
    """Fitting a selected frame range into a replacement video."""

    def test_range_that_still_fits_is_kept(self) -> None:
        assert clamp_range(100, 250, 400) == (100, 250)

    def test_end_beyond_the_last_frame_is_pulled_back(self) -> None:
        assert clamp_range(100, 250, 200) == (100, 199)

    def test_range_entirely_past_the_end_falls_back_to_the_whole_video(self) -> None:
        assert clamp_range(300, 400, 200) == (0, 199)

    def test_single_frame_video_collapses_the_range(self) -> None:
        assert clamp_range(5, 20, 1) == (0, 0)


class TestSlowmo:
    """Per-video slow-motion factor stored in the project file."""

    @staticmethod
    def _write(target: Path, video: dict[str, object]) -> None:
        base = {"path": "a.mp4", "start_frame": 0, "end_frame": 10}
        target.write_text(
            json.dumps({"version": 1, "videos": [{**base, **video}], "render": {}}),
            encoding="utf-8",
        )

    def test_default_is_real_time(self, tmp_path: Path) -> None:
        assert VideoEntry(path=tmp_path / "a.mp4", start_frame=0, end_frame=10).slowmo == 1.0

    def test_round_trip(self, tmp_path: Path) -> None:
        project = Project(
            videos=[
                VideoEntry(path=tmp_path / "a.mp4", start_frame=0, end_frame=10, slowmo=2.5),
                VideoEntry(path=tmp_path / "b.mp4", start_frame=0, end_frame=10),
            ],
            render=RenderOptions(),
        )
        target = tmp_path / "session.railcam.json"

        project.save(target)
        loaded = Project.load(target)

        assert [video.slowmo for video in loaded.videos] == [2.5, 1.0]
        assert loaded == project

    def test_absent_field_means_real_time(self, tmp_path: Path) -> None:
        target = tmp_path / "old.railcam.json"
        self._write(target, {})

        assert Project.load(target).videos[0].slowmo == 1.0

    def test_integer_value_is_accepted(self, tmp_path: Path) -> None:
        target = tmp_path / "int.railcam.json"
        self._write(target, {"slowmo": 4})

        assert Project.load(target).videos[0].slowmo == 4.0

    def test_fine_value_survives_a_round_trip(self, tmp_path: Path) -> None:
        target = tmp_path / "fine.railcam.json"
        self._write(target, {"slowmo": 2.3333})

        assert Project.load(target).videos[0].slowmo == 2.3333

    @pytest.mark.parametrize("value", [SLOWMO_MIN, 32, SLOWMO_MAX])
    def test_values_within_the_editable_range_are_kept(self, tmp_path: Path, value: float) -> None:
        target = tmp_path / "range.railcam.json"
        self._write(target, {"slowmo": value})

        assert Project.load(target).videos[0].slowmo == value

    @pytest.mark.parametrize(
        "value", [0, -2, "2", None, True, 0.05, 100, float("inf"), float("nan")]
    )
    def test_invalid_value_is_rejected(self, tmp_path: Path, value: object) -> None:
        target = tmp_path / "bad.railcam.json"
        self._write(target, {"slowmo": value})

        with pytest.raises(ProjectError, match="slow-motion"):
            Project.load(target)
