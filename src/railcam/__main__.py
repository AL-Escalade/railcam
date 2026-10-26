"""Entry point of the packaged application, which can only launch this module."""

from __future__ import annotations

import os
import sys

from railcam.packaged import RUN_CLI_ENV


def _main() -> int:
    if os.environ.get(RUN_CLI_ENV):
        from railcam.cli import main as cli_main

        return cli_main()
    from railcam.gui.app import main as gui_main

    return gui_main()


sys.exit(_main())
