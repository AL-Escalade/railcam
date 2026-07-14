# Tasks: add-gui

## 1. Foundations

- [ ] 1.1 Add `gui` optional extra (PySide6) and `railcam-gui` entry point to `pyproject.toml`; graceful error message when PySide6 is missing
- [ ] 1.2 Implement `gui/project.py`: session dataclasses, versioned JSON save/load, relative path handling + unit tests
- [ ] 1.3 Implement `gui/render.py` argv building (GUI state → `railcam` CLI arguments) + unit tests
- [ ] 1.4 Implement `gui/frame_source.py`: OpenCV frame-exact seek, sequential read, LRU cache, metadata + unit tests on a synthetic video

## 2. Player UI

- [ ] 2.1 Implement `gui/player_widget.py`: frame display, scrubbable timeline with range highlight, frame counter
- [ ] 2.2 Keyboard stepping (±1, Shift ±10), "set start/end = current frame" buttons, climber selector
- [ ] 2.3 Implement `gui/main_window.py`: toolbar (open/save project, add video), side-by-side player grid, remove video
- [ ] 2.4 Dark theme (`style.qss`) applied in `gui/app.py`

## 3. Synchronized playback

- [ ] 3.1 Implement `gui/playback.py`: common clock, time→frame mapping per video, freeze at end frame + unit tests (different FPS, freeze)
- [ ] 3.2 Global transport bar: play/pause (space bar), stop, preview speed selector (0.1×–1×)

## 4. Render integration

- [ ] 4.1 Render panel: format, height, output speed, debug, output path; live validation (disable render on invalid ranges)
- [ ] 4.2 Live equivalent CLI command display with copy button
- [ ] 4.3 QProcess execution: progress parsing from CLI output + unit tests, cancel button, error output panel, open-result action
- [ ] 4.4 Project open with missing video: "missing" player state with relocate action

## 5. Polish and docs

- [ ] 5.1 End-to-end manual pass on Windows and macOS (load, sync preview, render mp4 and gif)
- [ ] 5.2 Update README with `railcam-gui` installation and usage
