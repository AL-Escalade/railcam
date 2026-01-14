# Change: Add Speed Climbing GIF CLI

## Why

Speed climbing videos capture athletes ascending a 15m wall in under 5 seconds. Creating focused GIFs that track the climber's movement is currently a manual, tedious process requiring frame-by-frame adjustments. This CLI tool automates the creation of cropped, centered GIFs that follow the climber's pelvis throughout the ascent.

## What Changes

- **NEW** CLI tool to generate GIFs from speed climbing videos
- **NEW** Pose estimation integration to detect climber's pelvis position
- **NEW** Smart cropping with 4:3 vertical aspect ratio
- **NEW** Automatic boundary handling (no black borders)
- **NEW** Motion smoothing for fluid camera movement
- **NEW** Interpolation for frames with failed detection

## Impact

- Affected specs:
  - `cli-interface` (new)
  - `pose-detection` (new)
  - `video-cropping` (new)
  - `gif-generation` (new)
- Affected code: New project - all code is new

## Scope

This is a greenfield project. The CLI will be implemented as a cross-platform command-line tool that:

1. Takes a video file, start frame, and end frame as input
2. Detects the climber's pelvis position on each frame using pose estimation
3. Calculates optimal crop regions following the pelvis with smooth motion
4. Handles edge cases (climber near video boundaries)
5. Outputs a high-quality GIF

## Success Criteria

- CLI runs on macOS, Linux, and Windows
- Accurate pelvis tracking throughout the climb
- Smooth, non-jarring crop motion
- No black borders in output GIF
- Framerate preserved from source video
