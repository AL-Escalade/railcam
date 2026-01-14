# Tasks: Speed Climbing GIF CLI

## 1. Project Setup
- [x] 1.1 Initialize Python project with pyproject.toml
- [x] 1.2 Configure development dependencies (pytest, ruff, mypy)
- [x] 1.3 Create project structure (src/speed_gif/, tests/)
- [x] 1.4 Set up CLI entry point with argparse or click

## 2. Video Processing Foundation
- [x] 2.1 Implement video file validation (existence, format support)
- [x] 2.2 Implement frame extraction using OpenCV
- [x] 2.3 Implement frame range validation
- [x] 2.4 Extract video metadata (fps, dimensions, total frames)

## 3. Pose Detection
- [x] 3.1 Integrate MediaPipe Pose
- [x] 3.2 Implement pelvis position calculation (midpoint of hips)
- [x] 3.3 Implement confidence thresholding (visibility >= 0.5)
- [x] 3.4 Handle single-hip detection fallback
- [ ] 3.5 Write unit tests for pose detection

## 4. Position Processing
- [x] 4.1 Implement linear interpolation for missing detections
- [x] 4.2 Handle edge cases (gaps at start/end of range)
- [x] 4.3 Implement exponential moving average smoothing
- [x] 4.4 Write unit tests for position processing

## 5. Cropping Logic
- [x] 5.1 Implement 4:3 crop dimension calculation
- [x] 5.2 Implement pelvis-centered crop positioning
- [x] 5.3 Implement boundary clamping (no black borders)
- [x] 5.4 Ensure consistent crop size across all frames
- [x] 5.5 Write unit tests for cropping logic

## 6. GIF Generation
- [x] 6.1 Implement frame cropping with OpenCV
- [x] 6.2 Implement optional resolution scaling
- [x] 6.3 Integrate FFmpeg for GIF generation with palette optimization
- [x] 6.4 Implement default output path generation
- [x] 6.5 Add progress feedback during processing
- [ ] 6.6 Write integration tests for GIF generation

## 7. CLI Polish
- [x] 7.1 Implement --debug mode with pose overlay visualization
- [x] 7.2 Add comprehensive error messages
- [x] 7.3 Implement --help and --version flags
- [x] 7.4 Add input validation for mutually exclusive options (--width/--height)

## 8. Documentation & Packaging
- [x] 8.1 Write README with usage examples
- [x] 8.2 Document installation instructions (pip, FFmpeg)
- [x] 8.3 Add type hints throughout codebase
- [x] 8.4 Configure package for pip installation

## 9. Testing & Validation
- [ ] 9.1 Create test fixtures (sample video clips)
- [ ] 9.2 Run end-to-end tests with real climbing videos
- [ ] 9.3 Validate cross-platform compatibility (macOS, Linux, Windows)
- [ ] 9.4 Performance testing with various video sizes
