## ADDED Requirements

### Requirement: Detection Model Selection

The GUI SHALL expose the pose detection model size as a render option, SHALL
include it in the displayed CLI command, and SHALL persist it in project files.

#### Scenario: Model choice offered

- **WHEN** the render options are displayed
- **THEN** the GUI SHALL offer the five model sizes with labels naming both the
  trade-off and the CLI value, for example `Rapide (n)` and `Précis (m)`
- **AND** `Équilibré (s)` SHALL be selected by default

#### Scenario: Command reflects the model choice

- **WHEN** the user selects a model other than the default
- **THEN** the displayed CLI command SHALL include the matching `--model` argument

#### Scenario: Default model omitted from the command

- **WHEN** the selected model is the default
- **THEN** the displayed CLI command SHALL NOT include a `--model` argument

#### Scenario: Model saved and restored

- **WHEN** a project is saved with a selected model and reopened
- **THEN** the same model SHALL be selected in the render options

#### Scenario: Project saved before this option existed

- **WHEN** a project file written without a model field is opened
- **THEN** the GUI SHALL select the default model
- **AND** the project SHALL load without error
