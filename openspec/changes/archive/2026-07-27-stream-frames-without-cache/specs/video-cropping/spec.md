## ADDED Requirements

### Requirement: Crop Plan Independent Of Frames

The system SHALL compute the output dimensions, scale factor and smoothed
positions of a crop from the analysis alone, without access to any frame.

#### Scenario: Plan computed from an analysis

- **WHEN** a crop plan is built from an analysis and a target torso ratio
- **THEN** the plan SHALL carry the output dimensions, the scale factor and one
  position per frame
- **AND** building it SHALL NOT require any decoded frame

#### Scenario: Plan covers every frame of the range

- **WHEN** a plan is built for a range of N frames
- **THEN** the plan SHALL hold N positions, in frame order

### Requirement: Streamed Cropping

The system SHALL produce cropped frames one at a time by re-reading the source
video, holding at most one source frame and one cropped frame at a time.

#### Scenario: Cropped frames produced in order

- **WHEN** cropping is run against a plan
- **THEN** cropped frames SHALL be yielded in increasing frame order
- **AND** each SHALL match the output dimensions of the plan

#### Scenario: Second pass disagrees with the plan

- **WHEN** the frame read during cropping does not match the frame the next
  planned position belongs to
- **THEN** the system SHALL raise a video error identifying the mismatch
- **AND** SHALL NOT emit a frame cropped at the wrong position
