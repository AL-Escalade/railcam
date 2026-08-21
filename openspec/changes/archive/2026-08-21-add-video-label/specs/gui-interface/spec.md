## ADDED Requirements

### Requirement: Per-Video Label Field

The GUI SHALL expose a text field per video for its label, SHALL include the
labels in the displayed CLI command, and SHALL persist them in project files.

#### Scenario: Label entered

- **WHEN** the user types a label in a video's label field
- **THEN** the displayed CLI command SHALL include a `--label` argument for every
  video, in the same order as the inputs
- **AND** unlabeled videos SHALL be represented by an empty label argument
- **AND** the command SHALL stay valid when a label starts with a dash or holds
  quotes, spaces or apostrophes

#### Scenario: No label entered

- **WHEN** no video has a label
- **THEN** the displayed CLI command SHALL NOT include any `--label` argument

#### Scenario: Labels saved and restored

- **WHEN** a project is saved with labels and reopened
- **THEN** each video SHALL show the label it was saved with

#### Scenario: Project saved before this option existed

- **WHEN** a project file written without a label field is opened
- **THEN** every video SHALL have an empty label
- **AND** the project SHALL load without error
