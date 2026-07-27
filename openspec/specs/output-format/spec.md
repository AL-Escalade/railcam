# output-format Specification

## Purpose
TBD - created by archiving change add-mp4-sidebyside-zoom. Update Purpose after archive.
## Requirements
### Requirement: MP4 Output Generation
The system SHALL support generating MP4 video output as an alternative to GIF format.

#### Scenario: Generate MP4 with default settings
- **WHEN** the user runs the CLI with `--format mp4`
- **THEN** the output file SHALL be encoded using H.264 codec with yuv420p pixel format
- **AND** the output file SHALL have `.mp4` extension

#### Scenario: MP4 is the default format
- **WHEN** the user does not specify `--format`
- **THEN** the output SHALL default to MP4 format

#### Scenario: Generate GIF when explicitly requested
- **WHEN** the user runs the CLI with `--format gif`
- **THEN** the output SHALL be a GIF with palette optimization (unchanged from current behavior)

### Requirement: Output Format Selection
The system SHALL provide a `--format` CLI argument to select the output format.

#### Scenario: Valid format selection
- **WHEN** the user specifies `--format mp4` or `--format gif`
- **THEN** the system SHALL generate output in the requested format

#### Scenario: Invalid format rejection
- **WHEN** the user specifies an unsupported format (e.g., `--format avi`)
- **THEN** the system SHALL display an error message listing supported formats
- **AND** the system SHALL exit with a non-zero status code

### Requirement: Automatic Extension Handling
The system SHALL automatically adjust the output file extension based on the selected format.

#### Scenario: Extension correction for MP4
- **WHEN** the user specifies `--output climb.gif` with `--format mp4`
- **THEN** the output file SHALL be saved as `climb.mp4`

#### Scenario: Extension correction for GIF
- **WHEN** the user specifies `--output climb.mp4` with `--format gif`
- **THEN** the output file SHALL be saved as `climb.gif`

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

### Requirement: Streaming Frame Encoding

The system SHALL accept frames to encode as a stream, with the frame dimensions
and total count supplied separately, so the whole output need not exist in
memory before encoding starts.

#### Scenario: Frames encoded from a stream

- **WHEN** encoding is given an iterator of frames with dimensions and a count
- **THEN** each frame SHALL be written to FFmpeg as it is produced
- **AND** the encoder SHALL NOT require the frames to be materialized first

#### Scenario: Progress reported while streaming

- **WHEN** frames are encoded from a stream with a progress callback
- **THEN** progress SHALL be reported per frame against the supplied total

#### Scenario: Stream ends early

- **WHEN** the stream yields fewer frames than the supplied count
- **THEN** the system SHALL raise an output generation error naming the shortfall

#### Scenario: List-based encoding still supported

- **WHEN** encoding is given a list of frames
- **THEN** the output SHALL be identical to encoding the same frames as a stream

