## ADDED Requirements

### Requirement: Direct Frame Encoding

The system SHALL send frames to FFmpeg over its standard input as raw video, and
SHALL NOT write intermediate image files to disk.

#### Scenario: MP4 encoding writes no temporary files

- **WHEN** the system generates an MP4 output
- **THEN** frames SHALL be written to the FFmpeg process standard input as
  `rawvideo` with `bgr24` pixel format
- **AND** no temporary image file SHALL be created

#### Scenario: GIF encoding writes no temporary files

- **WHEN** the system generates a GIF output
- **THEN** palette generation and application SHALL happen in a single FFmpeg
  invocation reading frames from standard input
- **AND** no temporary image or palette file SHALL be created

#### Scenario: FFmpeg fails during encoding

- **WHEN** the FFmpeg process exits with a non-zero status while frames are
  being written
- **THEN** the system SHALL raise an output generation error carrying FFmpeg's
  captured error output
- **AND** the system SHALL NOT block waiting to write further frames

#### Scenario: No frames to encode

- **WHEN** the frame list passed to output generation is empty
- **THEN** the system SHALL raise an output generation error
- **AND** the system SHALL NOT start an FFmpeg process
