# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

<!-- OPENSPEC:START -->
# OpenSpec Instructions

These instructions are for AI assistants working in this project.

Always open `@/openspec/AGENTS.md` when the request:
- Mentions planning or proposals (words like proposal, spec, change, plan)
- Introduces new capabilities, breaking changes, architecture shifts, or big performance/security work
- Sounds ambiguous and you need the authoritative spec before coding

Use `@/openspec/AGENTS.md` to learn:
- How to create and apply change proposals
- Spec format and conventions
- Project structure and guidelines

Keep this managed block so 'openspec update' can refresh the instructions.

<!-- OPENSPEC:END -->

## Project Overview

**railcam** is a CLI tool that turns raw climbing footage into clean, analysis-ready videos by automatically tracking the climber using AI pose estimation. Designed for speed climbing analysis and side-by-side comparisons.

## Common Commands

```bash
# Install (development mode)
pip install -e ".[dev]"
pre-commit install

# Run the CLI
railcam video.mp4 100 250                        # Single video, frames 100-250
railcam video.mp4 100 250 --climber left         # Track left climber
railcam --input v1.mp4:0:100 --input v2.mp4:0:150  # Side-by-side comparison

# Run tests
pytest                    # Run all tests
pytest tests/test_processing.py  # Run specific test file
pytest -k "test_gap"     # Run tests matching pattern

# Linting and formatting
ruff check src tests     # Lint
ruff format src tests    # Format
ruff check --fix src     # Auto-fix lint issues

# Type checking
mypy src
```

## Architecture

```
src/railcam/
├── cli.py          # Entry point, argument parsing, orchestration
├── video.py        # Video file handling, frame extraction (OpenCV)
├── pose.py         # YOLOv8-pose detection, climber selection, torso measurement
├── processing.py   # Position interpolation and exponential smoothing
├── cropping.py     # Crop region calculation, zoom normalization
├── labeling.py     # Optional text label band appended under a cropped frame
├── composition.py  # Multi-video time sync and horizontal composition
├── multi_video.py  # Input spec parsing (path:start:end:climber)
└── output.py       # MP4/GIF generation via FFmpeg subprocess
```

### Processing Pipeline

Each video is decoded **twice**, and no frame is ever retained. Cropping cannot
happen during detection because it needs the smoothed position, which needs
every detection first — so the two passes are a consequence of the data
dependency, not a choice.

**Pass 1 — analyze** (`cli.py:analyze_video`), one video at a time:

1. **Frame extraction** (`video.py`) - Stream the frame range with OpenCV, releasing each frame after use
2. **Pose detection** (`pose.py`) - YOLOv8-pose detects all persons per frame
3. **Track selection** (`tracking.py`) - Group detections into motion tracks, pick the climber by upward movement
4. **Gap repair** (`frame_source.py`) - Re-read only the frames the selected track is missing, by index, and retry at 1280px
5. **Crop planning** (`cli.py:build_crop_plan`) - Interpolate and smooth positions, compute scale factor and output size. Pure: no pixels involved

**Pass 2 — render**, driven lazily by the encoder pulling frames:

6. **Cropping** (`cli.py:crop_frames`) - Re-decode the range and yield cropped frames one at a time: scale the frame, crop around the pelvis, pad if needed. Each decoded frame's number is checked against its planned position, since a silent desync would mis-frame everything. The optional label band (`labeling.py`) is appended under the image after the debug overlay and before the output scaling, so it stays a fixed fraction of the image at any output size
7. **Composition** (`composition.py`) - For multi-video: a `FrameCursor` per video pulls its crop stream forward along the time-sync indices (which never decrease), and `compose_frame_row` emits one composed frame at a time
8. **Output** (`output.py:generate_output_stream`) - Pipe raw BGR frames to FFmpeg stdin for MP4 (H.264) or GIF

Cropping has no progress stage of its own: it runs as FFmpeg consumes frames,
so the encoding bar covers it (`gui/progress.py:expected_stage_count`).

Planning never needs another video's analysis, since the multi-video zoom target
is the `TORSO_HEIGHT_RATIO` constant.

### Key Domain Concepts

- **Pelvis tracking**: Midpoint between hips used as crop center point
- **Torso height**: Shoulder-to-hip distance, normalized to 1/6 of output height for consistent scale
- **5:3 aspect ratio**: Portrait orientation optimized for vertical climbing videos
- **ClimberSelector**: `AUTO` (proximity-based tracking), `LEFT`, or `RIGHT` (for dual-lane walls)
- **LCM FPS sync**: Multi-video output uses LCM of all input FPS values to prevent judder

## Code Conventions

- Type hints on all functions
- `from __future__ import annotations` for Python 3.9 compatibility
- Line length: 100 characters
- Linter: ruff
- Tests: pytest in `tests/` directory
