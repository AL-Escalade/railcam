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

