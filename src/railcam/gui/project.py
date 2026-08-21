"""GUI session model and JSON project file (de)serialization.

This module has no Qt imports so it stays importable and unit-testable
without the GUI dependencies.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_VERSION = 1
VALID_CLIMBERS = ("auto", "left", "right")

# Mirrors railcam.pose.MODEL_SIZES / DEFAULT_MODEL_SIZE as plain strings, the
# same way VALID_CLIMBERS mirrors ClimberSelector: importing railcam.pose here
# would pull ultralytics and torch into the GUI's model layer.
VALID_MODELS = ("n", "s", "m", "l", "x")
DEFAULT_MODEL_SIZE = "s"

# Labels for the model selector, naming both the trade-off and the CLI value.
# The panel displays the equivalent CLI command, so a user who picks "Précis"
# must be able to recognise the `--model m` they end up copying.
MODEL_LABELS = (
    ("n", "Rapide (n)"),
    ("s", "Équilibré (s)"),
    ("m", "Précis (m)"),
    ("l", "Très précis (l)"),
    ("x", "Maximal (x)"),
)


class ProjectError(Exception):
    """Error loading or saving a project file."""


@dataclass
class VideoEntry:
    """One video in a session: source file, frame range, climber choice and label."""

    path: Path
    start_frame: int
    end_frame: int
    climber: str = "auto"
    label: str = ""


@dataclass
class RenderOptions:
    """Render options mirroring the railcam CLI defaults."""

    format: str = "mp4"
    height: int | None = None
    speed: float = 1.0
    debug: bool = False
    model: str = DEFAULT_MODEL_SIZE


@dataclass
class Project:
    """A full GUI session, serializable to a versioned JSON project file."""

    videos: list[VideoEntry] = field(default_factory=list)
    render: RenderOptions = field(default_factory=RenderOptions)
    output_path: Path | None = None

    def save(self, file_path: Path) -> None:
        """Write the project as JSON, storing video paths relative when possible."""
        project_dir = file_path.parent
        data = {
            "version": PROJECT_VERSION,
            "videos": [
                {
                    "path": _serialize_path(video.path, project_dir),
                    "start_frame": video.start_frame,
                    "end_frame": video.end_frame,
                    "climber": video.climber,
                    "label": video.label,
                }
                for video in self.videos
            ],
            "render": {
                "format": self.render.format,
                "height": self.render.height,
                "speed": self.render.speed,
                "debug": self.render.debug,
                "model": self.render.model,
            },
            "output_path": str(self.output_path) if self.output_path is not None else None,
        }
        file_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, file_path: Path) -> Project:
        """Read a project file, resolving relative video paths against its directory.

        Raises:
            ProjectError: If the file is not valid JSON, has an unsupported
                version, or is missing required fields.
        """
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            raise ProjectError(f"Invalid project file: {e}") from e

        if not isinstance(data, dict):
            raise ProjectError("Invalid project file: expected a JSON object")

        version = data.get("version")
        if version != PROJECT_VERSION:
            raise ProjectError(
                f"Unsupported project version: {version!r} (expected {PROJECT_VERSION})"
            )

        project_dir = file_path.parent
        videos_data = data.get("videos")
        render_data = data.get("render")
        if not isinstance(videos_data, list) or not isinstance(render_data, dict):
            raise ProjectError("Invalid project file: missing 'videos' or 'render'")

        videos = [_load_video_entry(entry, project_dir) for entry in videos_data]
        render = _load_render_options(render_data)
        output_raw = data.get("output_path")
        output_path = Path(output_raw) if output_raw else None

        return cls(videos=videos, render=render, output_path=output_path)


def _serialize_path(path: Path, project_dir: Path) -> str:
    """Return path relative to project_dir when possible, absolute otherwise."""
    try:
        return os.path.relpath(path, project_dir)
    except ValueError:
        # Different drive on Windows: no relative form exists
        return str(path)


def _load_video_entry(entry: Any, project_dir: Path) -> VideoEntry:
    if not isinstance(entry, dict):
        raise ProjectError("Invalid project file: video entry is not an object")
    try:
        raw_path = entry["path"]
        start_frame = int(entry["start_frame"])
        end_frame = int(entry["end_frame"])
    except (KeyError, TypeError, ValueError) as e:
        raise ProjectError(f"Invalid project file: bad video entry ({e})") from e

    climber = entry.get("climber", "auto")
    if climber not in VALID_CLIMBERS:
        raise ProjectError(
            f"Invalid climber selector: {climber!r}. Valid values are {VALID_CLIMBERS}."
        )

    # Absent in project files written before labels existed
    label = entry.get("label", "")
    if not isinstance(label, str):
        raise ProjectError(f"Invalid label: {label!r}. Expected a string.")

    path = Path(raw_path)
    if not path.is_absolute():
        path = project_dir / path
    return VideoEntry(
        path=path,
        start_frame=start_frame,
        end_frame=end_frame,
        climber=climber,
        label=label,
    )


def _load_render_options(data: dict[str, Any]) -> RenderOptions:
    defaults = RenderOptions()
    model = str(data.get("model", defaults.model))
    if model not in VALID_MODELS:
        raise ProjectError(f"Invalid model size: {model!r}. Valid values are {VALID_MODELS}.")
    try:
        height_raw = data.get("height", defaults.height)
        return RenderOptions(
            format=str(data.get("format", defaults.format)),
            height=int(height_raw) if height_raw is not None else None,
            speed=float(data.get("speed", defaults.speed)),
            debug=bool(data.get("debug", defaults.debug)),
            model=model,
        )
    except (TypeError, ValueError) as e:
        raise ProjectError(f"Invalid project file: bad render options ({e})") from e
