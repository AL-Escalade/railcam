"""railcam-gui entry point."""

from __future__ import annotations

import sys


def main() -> int:
    """Launch the railcam GUI, with a clear message if Qt is unavailable."""
    try:
        import PySide6  # noqa: F401
    except ImportError:
        print(
            "railcam-gui requires the GUI dependencies.\n"
            'Install them with: pip install "railcam[gui]"',
            file=sys.stderr,
        )
        return 1

    from railcam.gui.main_window import run_app

    return run_app()
