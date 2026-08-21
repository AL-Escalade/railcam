"""Build railcam CLI arguments from a GUI project.

Argument building is pure (no Qt) so it stays unit-testable; process
management lives in the Qt layer.
"""

from __future__ import annotations

import shlex
import shutil
import sys
from pathlib import Path

from railcam.gui.project import Project, RenderOptions, VideoEntry


def resolve_railcam_command() -> tuple[str, list[str]]:
    """Return (program, prefix_args) to run the railcam CLI.

    Prefers the railcam executable from the current environment; falls back
    to running the CLI through the GUI's own Python interpreter so both
    always share one installation.
    """
    # Prefer the railcam installed alongside the GUI's interpreter, so both
    # always come from the same environment (PATH may hold another install).
    scripts_dir = Path(sys.executable).parent
    for name in ("railcam.exe", "railcam"):
        candidate = scripts_dir / name
        if candidate.exists():
            return str(candidate), []
    executable = shutil.which("railcam")
    if executable:
        return executable, []
    return sys.executable, ["-c", "from railcam.cli import main; import sys; sys.exit(main())"]


def _input_spec(video: VideoEntry) -> str:
    """Format one video as a CLI input spec (path:start:end[:climber])."""
    spec = f"{video.path.as_posix()}:{video.start_frame}:{video.end_frame}"
    if video.climber != "auto":
        spec += f":{video.climber}"
    return spec


def build_cli_args(project: Project) -> list[str]:
    """Build the railcam CLI argument list, omitting options at their defaults."""
    defaults = RenderOptions()
    args: list[str] = []
    # Labels are paired with the inputs positionally, so as soon as one video
    # is labeled every video needs its own --label to keep the pairing right.
    any_label = any(video.label for video in project.videos)
    for video in project.videos:
        args.extend(["-i", _input_spec(video)])
        if any_label:
            # Single token: a label is free text and one starting with a dash
            # would be read as an option name in the two-token form.
            args.append(f"--label={video.label}")

    render = project.render
    if render.format != defaults.format:
        args.extend(["-f", render.format])
    if render.height is not None:
        args.extend(["-H", str(render.height)])
    if render.speed != defaults.speed:
        args.extend(["-s", str(render.speed)])
    if render.debug:
        args.append("--debug")
    if render.model != defaults.model:
        args.extend(["--model", render.model])
    if render.imgsz is not None:
        args.extend(["--imgsz", str(render.imgsz)])
    if project.output_path is not None:
        args.extend(["-o", project.output_path.as_posix()])
    return args


def format_cli_command(project: Project) -> str:
    """Return the copyable CLI command equivalent to the project configuration.

    Arguments are shell-quoted, so a command holding free text (a label with an
    apostrophe, a path with spaces) can be pasted as displayed.
    """
    return " ".join(["railcam", *(shlex.quote(arg) for arg in build_cli_args(project))])
