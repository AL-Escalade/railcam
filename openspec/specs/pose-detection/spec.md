# pose-detection Specification

## Purpose
TBD - created by archiving change add-speed-climbing-gif-cli. Update Purpose after archive.
## Requirements
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

### Requirement: Multi-Person Detection
The system SHALL detect all persons in a frame and return their pelvis positions.

#### Scenario: Two climbers with visible pelvis
- **WHEN** a frame contains two persons with visible hip landmarks
- **THEN** both pelvis positions are returned with their X coordinates

#### Scenario: Climber and bystander
- **WHEN** a frame contains one climber with visible hips and one bystander with only upper body visible
- **THEN** only the climber's pelvis position is returned

### Requirement: Pelvis Visibility Filtering
The system SHALL only consider persons whose hip landmarks (left_hip and/or right_hip) are detected with confidence above the threshold.

#### Scenario: Person with hips below confidence threshold
- **WHEN** a person is detected but both hip landmarks have confidence below threshold
- **THEN** this person is excluded from the detection results

#### Scenario: Person with one hip visible
- **WHEN** a person is detected with only one hip landmark above threshold
- **THEN** this person is included with pelvis estimated from the visible hip

### Requirement: Climber Selection by Position
The system SHALL select a specific climber based on a selector value (left, right, or auto).

#### Scenario: Left selector with two climbers
- **WHEN** selector is "left" and two persons with valid pelvis are detected
- **THEN** the person with the smallest pelvis X coordinate is selected

#### Scenario: Right selector with two climbers
- **WHEN** selector is "right" and two persons with valid pelvis are detected
- **THEN** the person with the largest pelvis X coordinate is selected

#### Scenario: Auto selector with single climber
- **WHEN** selector is "auto" and only one person with valid pelvis is detected
- **THEN** that person is selected

#### Scenario: Auto selector with multiple climbers
- **WHEN** selector is "auto" and multiple persons with valid pelvis are detected
- **THEN** the person closest to center is selected

### Requirement: Proximity-Based Tracking
The system SHALL track the selected climber across frames using position proximity after initial selection.

#### Scenario: Consistent tracking across frames
- **WHEN** the initial frame selected the left climber
- **AND** subsequent frames contain multiple persons
- **THEN** the person whose pelvis is closest to the previous frame's pelvis position is selected

#### Scenario: Climber temporarily not detected
- **WHEN** the tracked climber's pelvis is not detected in a frame
- **AND** other persons are detected
- **THEN** the last known position is used for interpolation (existing behavior)

