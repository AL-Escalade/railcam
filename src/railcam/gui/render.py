"""Build railcam CLI arguments from a GUI project.

Argument building is pure (no Qt) so it stays unit-testable; process
management lives in the Qt layer.
"""

from __future__ import annotations

from railcam.gui.project import Project, RenderOptions, VideoEntry


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
    for video in project.videos:
        args.extend(["-i", _input_spec(video)])

    render = project.render
    if render.format != defaults.format:
        args.extend(["-f", render.format])
    if render.height is not None:
        args.extend(["-H", str(render.height)])
    if render.speed != defaults.speed:
        args.extend(["-s", str(render.speed)])
    if render.debug:
        args.append("--debug")
    if project.output_path is not None:
        args.extend(["-o", project.output_path.as_posix()])
    return args


def _quote(arg: str) -> str:
    """Quote an argument for display if it contains spaces."""
    return f'"{arg}"' if " " in arg else arg


def format_cli_command(project: Project) -> str:
    """Return the copyable CLI command equivalent to the project configuration."""
    return " ".join(["railcam", *(_quote(arg) for arg in build_cli_args(project))])
