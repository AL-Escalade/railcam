## ADDED Requirements

### Requirement: Analysis Without Frame Retention

The system SHALL analyze a video without retaining its decoded source frames,
holding at most one frame at a time during the detection pass.

#### Scenario: Frames released during detection

- **WHEN** a video is analyzed
- **THEN** each decoded frame SHALL be released once it has been detected on
- **AND** the analysis result SHALL NOT carry decoded frames

#### Scenario: Gap repair reads only the frames it needs

- **WHEN** the selected climber's track has frames with no detection
- **THEN** the system SHALL read those frames by index from the source file
- **AND** SHALL NOT read the frames that already have a detection

#### Scenario: Nothing to repair

- **WHEN** the selected track has a detection in every frame
- **THEN** the system SHALL NOT re-open the source file for repair
