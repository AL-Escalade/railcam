## ADDED Requirements

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
