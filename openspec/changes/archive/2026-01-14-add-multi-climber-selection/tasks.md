## 1. Core Pose Detection Changes

- [x] 1.1 Add `ClimberSelector` enum in `pose.py` (`LEFT`, `RIGHT`, `AUTO`)
- [x] 1.2 Add `PersonDetection` dataclass to represent a single detected person with pelvis
- [x] 1.3 Implement `detect_all_persons()` method that returns all persons with valid pelvis
- [x] 1.4 Add `select_climber()` function for initial selection based on selector
- [x] 1.5 Add `track_climber()` function for proximity-based tracking on subsequent frames
- [x] 1.6 Unit tests for multi-person detection and selection logic

## 2. Input Parsing Changes

- [x] 2.1 Add `climber_selector` field to `VideoInput` dataclass in `multi_video.py`
- [x] 2.2 Update `parse_input_spec()` to parse optional fourth component (left/right)
- [x] 2.3 Add validation for climber selector values
- [x] 2.4 Unit tests for extended input parsing

## 3. CLI Integration

- [x] 3.1 Add `--climber` argument for positional mode in `cli.py`
- [x] 3.2 Update help text with examples for climber selection
- [x] 3.3 Pass climber selector from `VideoInput` to pose detection in `analyze_video()`

## 4. Detection Loop Changes

- [x] 4.1 Modify `analyze_video()` to use multi-person detection
- [x] 4.2 Implement initial frame selection logic
- [x] 4.3 Implement proximity tracking for subsequent frames
- [x] 4.4 Handle edge case: no persons with valid pelvis detected

## 5. Validation

- [ ] 5.1 Test with dual-lane speed climbing video (left climber)
- [ ] 5.2 Test with dual-lane speed climbing video (right climber)
- [ ] 5.3 Test with video containing bystanders at bottom
- [ ] 5.4 Test backward compatibility with single-climber video (no selector)
