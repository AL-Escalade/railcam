# Project Context

## Purpose
CLI tool to generate cropped videos (MP4 or GIF) from speed climbing footage, automatically tracking the climber's pelvis to keep them centered in frame. Supports side-by-side comparison of multiple climbers with normalized zoom.

## Tech Stack
- Python 3.9+
- OpenCV (video processing, frame extraction)
- YOLOv8-pose (ultralytics) - AI-based pose estimation
- FFmpeg (MP4/GIF generation)
- argparse (CLI argument parsing)

## Project Conventions

### Code Style
- Type hints on all functions
- Docstrings for public functions and classes
- `from __future__ import annotations` for Python 3.9 compatibility
- Line length: 100 characters
- Linting: ruff

### Architecture Patterns
- Modular design with separate concerns:
  - `video.py` - Video file handling
  - `pose.py` - Pose detection and torso measurement
  - `processing.py` - Position interpolation and smoothing
  - `cropping.py` - Frame cropping logic and zoom normalization
  - `output.py` - MP4/GIF generation
  - `multi_video.py` - Multi-video input parsing
  - `composition.py` - Frame synchronization and side-by-side composition
  - `cli.py` - Command-line interface

### Testing Strategy
- Unit tests for pure functions (processing, cropping)
- pytest as test runner
- Tests in `tests/` directory

### Git Workflow
- Feature branches
- Descriptive commit messages

## Domain Context
- Speed climbing: Athletes climb a standardized 15m wall as fast as possible (world record ~5 seconds)
- Pelvis tracking: The pelvis (midpoint between hips) is used as the center point for cropping
- 5:3 vertical ratio: Portrait aspect ratio optimized for climbing videos
- Torso normalization: Climber's torso height (shoulders to hips) is normalized to 1/3 of output height

## Important Constraints
- Must run locally (no cloud APIs)
- Cross-platform (macOS, Linux, Windows)
- Requires FFmpeg to be installed

## External Dependencies
- FFmpeg (must be installed separately)
- MediaPipe models (downloaded automatically on first run)
