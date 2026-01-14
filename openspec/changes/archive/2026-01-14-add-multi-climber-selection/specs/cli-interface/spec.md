## ADDED Requirements

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
