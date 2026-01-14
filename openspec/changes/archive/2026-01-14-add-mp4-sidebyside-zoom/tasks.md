# Tasks: Add MP4 Output, Side-by-Side Videos, and Normalized Zoom

## 1. Update Aspect Ratio
- [x] 1.1 Update `ASPECT_WIDTH` and `ASPECT_HEIGHT` constants in `cropping.py` from 3:4 to 3:5
- [x] 1.2 Update CLI help text to reflect new aspect ratio
- [x] 1.3 Update existing tests for new aspect ratio

## 2. Extend Pose Detection for Torso Measurement
- [x] 2.1 Add shoulder landmark extraction constants (LEFT_SHOULDER=5, RIGHT_SHOULDER=6)
- [x] 2.2 Create `TorsoMeasurement` dataclass with shoulder and hip positions
- [x] 2.3 Add torso measurement to `detect_pelvis()` method in `PoseDetector`
- [x] 2.4 Handle partial landmark detection (single shoulder/hip fallback)

## 3. Implement Zoom Normalization
- [x] 3.1 Add `TORSO_HEIGHT_RATIO = 1/3` constant in `cropping.py`
- [x] 3.2 Create `calculate_zoom_factor()` function
- [x] 3.3 Implement zoom factor clamping (min=1.0, max=3.0)
- [x] 3.4 Create `calculate_zoomed_crop_dimensions()` function
- [x] 3.5 Create `calculate_average_torso_height()` function
- [x] 3.6 Write unit tests for zoom calculation

## 4. Implement MP4 Output
- [x] 4.1 Rename `gif.py` to `output.py`
- [x] 4.2 Update imports across the codebase
- [x] 4.3 Add `generate_mp4()` function with H.264 encoding
- [x] 4.4 Create `generate_output()` dispatcher function for format selection
- [x] 4.5 Add `OutputFormat` enum and `parse_output_format()` function
- [x] 4.6 Add `--format` CLI argument with mp4 default
- [x] 4.7 Implement automatic file extension correction in `get_output_path()`

## 5. Implement Multi-Video Input
- [x] 5.1 Create `multi_video.py` module
- [x] 5.2 Create `VideoInput` dataclass with path, start_frame, end_frame
- [x] 5.3 Add `--input` CLI argument with path:start:end parsing
- [x] 5.4 Implement mutual exclusivity validation with positional args
- [x] 5.5 Create `parse_input_spec()` function with error handling
- [x] 5.6 Write unit tests for input parsing

## 6. Implement Frame Synchronization
- [x] 6.1 Create `composition.py` module
- [x] 6.2 Create `calculate_target_frame_count()` function (max across inputs)
- [x] 6.3 Implement `resample_frame_indices()` function
- [x] 6.4 Implement `resample_frames()` function with frame duplication
- [x] 6.5 Write unit tests for frame resampling

## 7. Implement Side-by-Side Composition
- [x] 7.1 Implement `normalize_frame_height()` function
- [x] 7.2 Implement `compose_frames_horizontal()` function
- [x] 7.3 Handle height normalization across videos
- [x] 7.4 Ensure even dimensions in output
- [x] 7.5 Write unit tests for composition

## 8. Integrate Multi-Video Pipeline
- [x] 8.1 Refactor `cli.py` with `VideoProcessingResult` dataclass
- [x] 8.2 Create `process_single_video()` function
- [x] 8.3 Implement two-pass processing (detect + render)
- [x] 8.4 Integrate zoom normalization per video
- [x] 8.5 Add progress reporting for multi-video processing
- [x] 8.6 Update `main()` to support multi-video mode

## 9. Testing
- [x] 9.1 Unit tests for aspect ratio changes (test_cropping.py)
- [x] 9.2 Unit tests for zoom calculation (test_cropping.py)
- [x] 9.3 Unit tests for multi-video parsing (test_multi_video.py)
- [x] 9.4 Unit tests for frame composition (test_composition.py)
- [ ] 9.5 End-to-end tests with real videos (manual testing required)

## 10. Documentation
- [ ] 10.1 Update README with new CLI options
- [ ] 10.2 Add examples for multi-video usage
- [ ] 10.3 Document zoom normalization behavior
