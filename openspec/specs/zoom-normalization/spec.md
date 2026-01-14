# zoom-normalization Specification

## Purpose
TBD - created by archiving change add-mp4-sidebyside-zoom. Update Purpose after archive.
## Requirements
### Requirement: Torso Height Measurement
The system SHALL measure the climber's torso height using pose landmarks.

#### Scenario: Both shoulders and hips detected
- **WHEN** left shoulder, right shoulder, left hip, and right hip landmarks are detected
- **THEN** torso height SHALL be calculated as the vertical distance between shoulder midpoint and hip midpoint

#### Scenario: Partial landmark detection
- **WHEN** only one shoulder or one hip is detected
- **THEN** the system SHALL use the available landmarks to estimate torso height

#### Scenario: No torso landmarks detected
- **WHEN** neither shoulders nor hips are detected in a frame
- **THEN** the frame SHALL be excluded from average torso height calculation

### Requirement: Normalized Zoom Calculation
The system SHALL calculate a zoom factor to normalize torso height across videos.

#### Scenario: Average torso calculation
- **WHEN** processing a video with multiple frames
- **THEN** the system SHALL compute the average torso height across all frames with valid detections

#### Scenario: Zoom factor determination
- **WHEN** the average torso height is computed
- **THEN** the zoom factor SHALL be calculated so that `average_torso_height * zoom_factor = TORSO_HEIGHT_RATIO * output_height`

#### Scenario: Zoom factor limits
- **WHEN** the calculated zoom factor exceeds 3.0
- **THEN** the zoom factor SHALL be clamped to 3.0
- **WHEN** the calculated zoom factor is less than 1.0
- **THEN** the zoom factor SHALL be clamped to 1.0

### Requirement: Torso Height Target Ratio
The system SHALL use a configurable target ratio for torso height relative to output height.

#### Scenario: Default ratio
- **WHEN** no custom ratio is specified
- **THEN** the target torso height SHALL be 1/3 (33.3%) of the output height

### Requirement: Zoom Application
The system SHALL apply the zoom factor to the crop dimensions.

#### Scenario: Zoom increases magnification
- **WHEN** a zoom factor of 1.5 is calculated
- **THEN** the crop region SHALL be 1/1.5 = 66.7% of the original size
- **AND** the climber SHALL appear 1.5x larger in the output

#### Scenario: Zoom respects boundaries
- **WHEN** the zoomed crop region would extend beyond video boundaries
- **THEN** the crop region SHALL be shifted to stay within bounds
- **AND** no black borders SHALL appear in the output

