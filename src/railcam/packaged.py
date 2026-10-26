"""Facts about a Briefcase-packaged installation of railcam."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, distribution

RUN_CLI_ENV = "RAILCAM_RUN_CLI"
"""Set to make the packaged executable, which always starts the GUI, run the CLI instead."""


def is_packaged() -> bool:
    """Whether railcam runs from a Briefcase-built application."""
    try:
        installer = distribution("railcam").read_text("INSTALLER")
    except PackageNotFoundError:
        return False
    return (installer or "").strip() == "briefcase"
