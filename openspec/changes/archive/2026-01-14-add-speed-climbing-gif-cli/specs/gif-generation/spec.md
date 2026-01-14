## ADDED Requirements

### Requirement: GIF Output

The system SHALL generate an animated GIF from the cropped frames.

#### Scenario: GIF created successfully
- **WHEN** all frames are processed and cropped
- **THEN** the system SHALL output a single animated GIF file

### Requirement: Framerate Preservation

The system SHALL preserve the source video's framerate in the output GIF.

#### Scenario: Standard video framerate
- **WHEN** the source video has a standard framerate (e.g., 24, 30, 60 fps)
- **THEN** the output GIF SHALL play at the same framerate

#### Scenario: Variable framerate source
- **WHEN** the source video has variable framerate
- **THEN** the system SHALL use the average framerate for the GIF

### Requirement: Color Optimization

The system SHALL optimize the GIF color palette for quality.

#### Scenario: Palette generation
- **WHEN** generating the GIF
- **THEN** the system SHALL generate an optimized 256-color palette from the actual frames

### Requirement: Output Resolution Scaling

The system SHALL scale output frames when a resolution is specified.

#### Scenario: Width specified
- **WHEN** the user specifies output width
- **THEN** all frames SHALL be scaled to that width with height calculated from 4:3 ratio

#### Scenario: Height specified
- **WHEN** the user specifies output height
- **THEN** all frames SHALL be scaled to that height with width calculated from 4:3 ratio

#### Scenario: No scaling specified
- **WHEN** no output resolution is specified
- **THEN** frames SHALL retain the crop region's original resolution

### Requirement: File Writing

The system SHALL write the GIF to the specified or default output path.

#### Scenario: Output to specified path
- **WHEN** an output path is provided
- **THEN** the GIF SHALL be written to that exact path

#### Scenario: Output to default path
- **WHEN** no output path is provided
- **THEN** the GIF SHALL be written to the current directory with a name derived from the input file

#### Scenario: Output path not writable
- **WHEN** the output path is not writable
- **THEN** the system SHALL exit with an error message indicating the path is not writable

### Requirement: Progress Feedback

The system SHALL provide progress feedback during GIF generation.

#### Scenario: Processing progress
- **WHEN** processing frames
- **THEN** the system SHALL display progress information (e.g., percentage complete, current frame)

#### Scenario: Completion notification
- **WHEN** GIF generation completes successfully
- **THEN** the system SHALL display the output file path and size
