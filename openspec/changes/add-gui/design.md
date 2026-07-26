# Design: Cross-platform desktop GUI

## Context

railcam is a Python CLI that crops climbing videos around the climber using YOLO pose detection. Users need a visual way to pick frame ranges, synchronize multiple videos, preview the result, and launch renders. Target platforms: macOS and Windows, fully local. The GUI must be practical and reasonably good-looking, launched first as a terminal command (`railcam-gui`), with the architecture leaving room for a packaged app later.

## Goals / Non-Goals

- Goals:
  - Frame-accurate visual selection of start/end frames per video
  - Synchronized side-by-side slow-motion preview of raw videos
  - Configure all relevant CLI options and run the render from the GUI
  - Save/load sessions as JSON project files
- Non-Goals:
  - Real-time preview of the final cropped output (YOLO is too slow without GPU); preview shows raw synchronized videos only
  - Packaged double-clickable app (.app/.exe) — deferred, but not precluded
  - Any change to the CLI or processing pipeline behavior

## Decisions

- **Decision: PySide6 (Qt for Python) with OpenCV-driven frame display.**
  Frame selection must be exact. HTML5 `<video>` seeks by timestamp, not frame, and codec support varies per-platform webview (HEVC notably). OpenCV (already a dependency) decodes deterministically frame by frame for any FFmpeg-supported format. Qt gives a native window, keyboard handling, and a themable UI in the same language as the pipeline.
  - Alternatives considered: local web app (FastAPI + browser) — prettier for free but frame-accuracy and codec issues; Tauri — best-looking shell but adds a Rust toolchain and has the same webview video limitations, while still requiring an external Python for the pipeline.

- **Decision: render runs the `railcam` CLI as a subprocess (QProcess), not an in-process import.**
  Keeps the GUI responsive while YOLO runs, guarantees the displayed equivalent command is exactly what executes, isolates crashes, and decouples the GUI from PyTorch startup (helps future packaging). Progress is parsed from the CLI's existing progress output (`stage: [bar] pct% (current/total)`, see `cli.py:print_progress`).

- **Decision: PySide6 is an optional dependency (`gui` extra) with a separate `railcam-gui` entry point.**
  CLI-only users don't download ~150 MB of Qt. Launching `railcam-gui` without PySide6 installed prints a clear install hint.

- **Decision: synchronization model = start frames aligned at t=0 on a common wall-clock.**
  Identical to the CLI/composition semantics. Per-video frame at time t: `frame = start_frame + round(t_seconds × fps_video)`, clamped to `end_frame` (freeze on last frame, matching `composition.py`). Handles heterogeneous FPS inputs. Preview speed (0.1×–1×) only scales the clock and is independent from the render `--speed` option.

- **Decision: project file is versioned JSON handled by a Qt-free `project.py` module.**
  Dataclasses + explicit `version: 1` field. Video paths stored relative to the project file when possible, absolute otherwise. Pure-Python module keeps serialization unit-testable.

## Module layout

```
src/railcam/gui/
├── app.py            # entry point: QApplication, theme, main window
├── main_window.py    # window shell, toolbar actions, player grid, render panel
├── frame_source.py   # OpenCV wrapper: frame-exact seek, sequential read, LRU cache, metadata
├── player_widget.py  # one video: frame view, timeline with range markers, stepping, climber selector
├── playback.py       # shared clock (QTimer) driving synchronized multi-video playback
├── project.py        # session model + JSON (de)serialization — no Qt imports
├── render.py         # GUI state → CLI argv, QProcess management, progress parsing — argv building has no Qt imports
└── style.qss         # dark theme
```

## Risks / Trade-offs

- Scrubbing backward on long-GOP codecs is slow with naive OpenCV seeks → LRU frame cache in `frame_source.py`; acceptable because speed-climbing clips are short (~10 s).
- Parsing CLI progress couples the GUI to the CLI's output format → the regex is tolerant (matches `(current/total)` anywhere in a line); worst case the bar degrades to indeterminate, render still works.
- PySide6 “looking good” requires styling effort → single dark QSS theme, no per-platform styling.
- Several videos decoded simultaneously during preview → at slow-motion speeds the frame rate is low (e.g. 0.25× of 30 fps = 7.5 fps per video), well within OpenCV decode budget.

## Migration Plan

Purely additive: new package, new entry point, new optional extra. No migration or rollback concerns; removing the `gui` extra restores the status quo.

## Open Questions

None — resolved during brainstorming (sync = start frames; preview = raw videos; distribution = terminal command first; project files = yes).
