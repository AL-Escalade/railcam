# Change: Add cross-platform desktop GUI

## Why

Finding the right frame numbers, synchronizing multiple videos, and tuning railcam options currently requires trial-and-error round-trips through the CLI. A visual tool removes this friction: scrub to the exact start/end frames, preview synchronized slow-motion playback of multiple videos, then launch the render — all from one window.

## What Changes

- New optional subpackage `railcam.gui` (PySide6) with a `railcam-gui` entry point, installed via the `gui` extra (`pip install railcam[gui]`)
- Frame-accurate video navigation: per-video scrubbable timeline, keyboard frame stepping, "set start/end = current frame"
- Per-video climber selection (auto/left/right)
- Synchronized preview playback: all videos play side by side from their start frames on a common clock, at a selectable slow-motion speed, freezing on their end frame
- Render panel: remaining CLI options (format, height, output speed, debug), live copyable equivalent CLI command, render executed by running the `railcam` CLI as a subprocess with a progress bar
- Project files (JSON): save/load the full session (videos, frame ranges, climber choices, render options)
- No changes to the existing CLI or processing pipeline

## Impact

- Affected specs: new capability `gui-interface`
- Affected code: new `src/railcam/gui/` package, `pyproject.toml` (entry point + `gui` extra), new tests in `tests/`
- Existing pipeline modules are untouched; the GUI consumes the CLI as an external process and reuses `railcam.video` for metadata only
