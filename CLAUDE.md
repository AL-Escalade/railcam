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
├── composition.py  # Multi-video time sync and horizontal composition
├── multi_video.py  # Input spec parsing (path:start:end:climber)
└── output.py       # MP4/GIF generation via FFmpeg subprocess
```

### Processing Pipeline

1. **Frame extraction** (`video.py`) - Extract frame range from video using OpenCV
2. **Pose detection** (`pose.py`) - YOLOv8-pose detects all persons, selects target climber based on position (left/right/auto with proximity tracking)
3. **Position processing** (`processing.py`) - Interpolate gaps where pelvis not detected, apply exponential smoothing
4. **Zoom calculation** (`cropping.py`) - Compute scale factor so average torso height = 1/6 of output height
5. **Cropping** (`cli.py:crop_video`) - Scale-then-crop approach: scale entire frame, crop around pelvis, add padding if needed
6. **Composition** (`composition.py`) - For multi-video: time-synchronize (freeze last frame), compose horizontally
7. **Output** (`output.py`) - Pipe frames to FFmpeg for MP4 (H.264) or GIF

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
