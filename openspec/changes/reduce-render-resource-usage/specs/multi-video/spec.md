## ADDED Requirements

### Requirement: Sequential Per-Video Processing

The system SHALL fully process each input video (analysis then cropping) before
starting the next one, and SHALL release a video's decoded source frames once
its frames have been cropped.

#### Scenario: Two videos processed in sequence

- **WHEN** the user specifies two `--input` arguments
- **THEN** the first video SHALL be analyzed and cropped before the second video
  is analyzed
- **AND** the decoded source frames of the first video SHALL be released before
  the second video is analyzed

#### Scenario: Cropped output is unchanged

- **WHEN** the same inputs are processed sequentially rather than in two phases
- **THEN** the cropped frames SHALL be identical to those produced by the
  previous two-phase processing

#### Scenario: Source frames released before composition

- **WHEN** all videos have been cropped
- **THEN** no decoded source frame SHALL be retained while frames are composed
  and encoded
