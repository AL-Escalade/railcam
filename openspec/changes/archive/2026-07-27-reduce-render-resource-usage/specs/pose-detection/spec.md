## ADDED Requirements

### Requirement: Configurable Pose Model Size

The system SHALL allow the pose model size to be selected at startup, and SHALL
default to the `s` (small) variant.

#### Scenario: Default model size

- **WHEN** a pose detector is created without an explicit model size
- **THEN** the system SHALL load the `yolov8s-pose` weights

#### Scenario: Explicit model size

- **WHEN** a pose detector is created with model size `m`
- **THEN** the system SHALL load the `yolov8m-pose` weights

#### Scenario: Gap repair uses the selected model

- **WHEN** the selected climber's track has frames with no detection
- **THEN** the system SHALL run the high-resolution repair pass on those frames
- **AND** the repair SHALL use the same model size as the initial pass
