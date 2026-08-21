## ADDED Requirements

### Requirement: Per-Video Second Label Field

The GUI SHALL expose a second text field per video for its second label line,
SHALL include it in the displayed CLI command, and SHALL persist it in project
files.

#### Scenario: Second line entered

- **WHEN** the user types a second label for a video
- **THEN** the displayed CLI command SHALL include a `--sublabel` argument for
  every video, in the same order as the inputs

#### Scenario: No second line entered

- **WHEN** no video has a second label
- **THEN** the displayed CLI command SHALL NOT include any `--sublabel` argument

#### Scenario: Second lines saved and restored

- **WHEN** a project is saved with second labels and reopened
- **THEN** each video SHALL show the second label it was saved with

#### Scenario: Project saved before this option existed

- **WHEN** a project file written without a second label field is opened
- **THEN** every video SHALL have an empty second label
- **AND** the project SHALL load without error
