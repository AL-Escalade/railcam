# cli-interface Specification

## Purpose
TBD - created by archiving change add-speed-climbing-gif-cli. Update Purpose after archive.
## Requirements
### Requirement: Video Input

The CLI SHALL accept a video file path as a required positional argument.

#### Scenario: Valid video file provided
- **WHEN** the user provides a path to an existing video file
- **THEN** the CLI SHALL process the video

#### Scenario: Invalid video file provided
- **WHEN** the user provides a path to a non-existent file
- **THEN** the CLI SHALL exit with an error message indicating the file was not found

#### Scenario: Unsupported video format
- **WHEN** the user provides a file that is not a supported video format
- **THEN** the CLI SHALL exit with an error message listing supported formats

### Requirement: Frame Range

The CLI SHALL accept start and end frame numbers as required arguments.

#### Scenario: Valid frame range provided
- **WHEN** the user provides valid start and end frame numbers where start < end
- **THEN** the CLI SHALL process only frames within that range (inclusive)

#### Scenario: Invalid frame range
- **WHEN** the user provides a start frame greater than or equal to the end frame
- **THEN** the CLI SHALL exit with an error message indicating invalid frame range

#### Scenario: Frame numbers out of bounds
- **WHEN** the user provides frame numbers exceeding the video's total frames
- **THEN** the CLI SHALL exit with an error message indicating the frame range exceeds video length

### Requirement: Output Path

The CLI SHALL accept an optional output file path for the generated GIF.

#### Scenario: Output path specified
- **WHEN** the user provides an output path via the --output flag
- **THEN** the CLI SHALL write the GIF to that path

#### Scenario: Output path not specified
- **WHEN** the user does not provide an output path
- **THEN** the CLI SHALL generate a default filename based on the input video name with .gif extension in the current directory

### Requirement: Output Resolution

The CLI SHALL accept an optional width or height parameter to control output resolution.

#### Scenario: Width specified
- **WHEN** the user provides a width via the --width flag
- **THEN** the CLI SHALL scale the output GIF to that width, calculating height to maintain 4:3 aspect ratio

#### Scenario: Height specified
- **WHEN** the user provides a height via the --height flag
- **THEN** the CLI SHALL scale the output GIF to that height, calculating width to maintain 4:3 aspect ratio

#### Scenario: Both width and height specified
- **WHEN** the user provides both --width and --height flags
- **THEN** the CLI SHALL exit with an error message indicating only one dimension can be specified

#### Scenario: No resolution specified
- **WHEN** the user does not specify width or height
- **THEN** the CLI SHALL use a resolution derived from the source video crop region

### Requirement: Debug Mode

The CLI SHALL provide a debug mode to visualize pose detection.

#### Scenario: Debug mode enabled
- **WHEN** the user provides the --debug flag
- **THEN** the CLI SHALL output additional frames showing detected pose landmarks overlaid on the video

### Requirement: Help Information

The CLI SHALL display usage information when requested.

#### Scenario: Help flag provided
- **WHEN** the user provides the --help flag
- **THEN** the CLI SHALL display usage information including all arguments and options

### Requirement: Version Information

The CLI SHALL display version information when requested.

#### Scenario: Version flag provided
- **WHEN** the user provides the --version flag
- **THEN** the CLI SHALL display the current version number

### Requirement: Extended Input Format with Climber Selector
The system SHALL accept an optional climber selector in the input specification format.

#### Scenario: Input with left climber selector
- **WHEN** user provides `--input video.mp4:100:250:left`
- **THEN** the system tracks the leftmost climber in the video

#### Scenario: Input with right climber selector
- **WHEN** user provides `--input video.mp4:100:250:right`
- **THEN** the system tracks the rightmost climber in the video

#### Scenario: Input without climber selector
- **WHEN** user provides `--input video.mp4:100:250`
- **THEN** the system uses auto selection (single climber or closest to center)

#### Scenario: Invalid climber selector
- **WHEN** user provides `--input video.mp4:100:250:center`
- **THEN** the system displays an error indicating valid selectors are "left" or "right"

### Requirement: Climber Selection in Positional Mode
The system SHALL accept a climber selector when using positional arguments.

#### Scenario: Positional with --climber flag
- **WHEN** user provides `railcam video.mp4 100 250 --climber left`
- **THEN** the system tracks the leftmost climber in the video

#### Scenario: Positional without --climber flag
- **WHEN** user provides `railcam video.mp4 100 250`
- **THEN** the system uses auto selection (existing behavior)

