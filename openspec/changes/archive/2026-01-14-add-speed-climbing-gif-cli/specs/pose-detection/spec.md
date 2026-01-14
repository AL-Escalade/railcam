## ADDED Requirements

### Requirement: Pelvis Detection

The system SHALL detect the pelvis position of the climber in each frame using pose estimation.

#### Scenario: Successful pelvis detection
- **WHEN** the climber is visible in a frame with clear hip landmarks
- **THEN** the system SHALL compute the pelvis position as the midpoint between left and right hip landmarks

#### Scenario: Single hip visible
- **WHEN** only one hip landmark is detected with sufficient confidence
- **THEN** the system SHALL use the visible hip position as the pelvis estimate

#### Scenario: No hips detected
- **WHEN** neither hip landmark is detected with sufficient confidence
- **THEN** the system SHALL mark the frame as requiring interpolation

### Requirement: Detection Confidence

The system SHALL use a confidence threshold to determine detection validity.

#### Scenario: High confidence detection
- **WHEN** hip landmarks are detected with visibility score >= 0.5
- **THEN** the system SHALL use the detected positions

#### Scenario: Low confidence detection
- **WHEN** hip landmarks are detected with visibility score < 0.5
- **THEN** the system SHALL treat the frame as having no valid detection

### Requirement: Position Interpolation

The system SHALL interpolate pelvis positions for frames with failed detection.

#### Scenario: Gap between valid detections
- **WHEN** one or more consecutive frames have no valid pelvis detection
- **AND** valid detections exist before and after the gap
- **THEN** the system SHALL linearly interpolate positions for the gap frames

#### Scenario: Gap at start of range
- **WHEN** the first frames have no valid detection but later frames do
- **THEN** the system SHALL use the first valid detection position for the initial frames

#### Scenario: Gap at end of range
- **WHEN** the last frames have no valid detection but earlier frames do
- **THEN** the system SHALL use the last valid detection position for the final frames

#### Scenario: No valid detections in entire range
- **WHEN** no frames in the specified range have valid pelvis detection
- **THEN** the system SHALL exit with an error indicating the climber could not be detected

### Requirement: Position Smoothing

The system SHALL apply smoothing to pelvis positions for fluid motion.

#### Scenario: Smoothing applied to positions
- **WHEN** processing detected and interpolated positions
- **THEN** the system SHALL apply exponential moving average smoothing to produce fluid motion

#### Scenario: Smoothing preserves general trajectory
- **WHEN** the climber moves from bottom to top of frame
- **THEN** the smoothed positions SHALL follow the same general trajectory without significant lag
