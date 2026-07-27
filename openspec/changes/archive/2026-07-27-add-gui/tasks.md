# Tasks: add-gui

## 1. Foundations

- [x] 1.1 Add `gui` optional extra (PySide6) and `railcam-gui` entry point to `pyproject.toml`; graceful error message when PySide6 is missing
- [x] 1.2 Implement `gui/project.py`: session dataclasses, versioned JSON save/load, relative path handling + unit tests
- [x] 1.3 Implement `gui/render.py` argv building (GUI state → `railcam` CLI arguments) + unit tests
- [x] 1.4 Implement `gui/frame_source.py`: OpenCV frame-exact seek, sequential read, LRU cache, metadata + unit tests on a synthetic video

## 2. Player UI

- [x] 2.1 Implement `gui/player_widget.py`: frame display, scrubbable timeline with range highlight, frame counter
- [x] 2.2 Keyboard stepping (±1, Shift ±10), "set start/end = current frame" buttons, climber selector
- [x] 2.3 Implement `gui/main_window.py`: toolbar (open/save project, add video), side-by-side player grid, remove video
- [x] 2.4 Dark theme (`style.qss`) applied in `gui/app.py`

- [x] 2.5 View zoom and pan per player: pure viewport math + unit tests, wheel zoom at
      cursor, drag pan, double-click reset, persistent across frame navigation

## 3. Synchronized playback

- [x] 3.1 Implement `gui/playback.py`: common clock, time→frame mapping per video, freeze at end frame + unit tests (different FPS, freeze)
- [x] 3.2 Global transport bar: play/pause (space bar), stop, preview speed selector (0.1×–1×)

## 4. Render integration

- [x] 4.1 Render panel: format, height, output speed, debug, output path; live validation (disable render on invalid ranges)
- [x] 4.2 Live equivalent CLI command display with copy button
- [x] 4.3 QProcess execution: progress parsing from CLI output + unit tests, cancel button, error output panel, open-result action
- [x] 4.4 Project open with missing video: "missing" player state with relocate action

## 5. Polish and docs

- [x] 5.1a Automated end-to-end pass on Windows: load and scrub a video, read the
  render options, build the equivalent CLI command, save and reload a two-video
  project, and render both mp4 and gif through the real CLI path. Pose detection
  was stubbed, since the fixtures are test patterns with no climber in them
- [ ] 5.1b Manual pass on macOS, and visual/interactive verification on either
  platform (scrubbing feel, synchronized preview playback, progress display).
  Needs a human on the machine; not performed
- [x] 5.2 Update README with `railcam-gui` installation and usage
