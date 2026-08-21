## ADDED Requirements

### Requirement: Video Label Option

The CLI SHALL accept a repeatable `--label` argument giving the text displayed
under each video, paired with the inputs in the order they are given.

#### Scenario: Label for a single video

- **WHEN** the user provides `railcam video.mp4 100 250 --label "Dupont"`
- **THEN** the rendered video SHALL display `Dupont` under the image

#### Scenario: Labels paired with inputs

- **WHEN** the user provides two `--input` arguments and two `--label` arguments
- **THEN** the first label SHALL apply to the first input and the second label to
  the second input

#### Scenario: Fewer labels than inputs

- **WHEN** the user provides two `--input` arguments and one `--label`
- **THEN** the label SHALL apply to the first input
- **AND** the second input SHALL have an empty label

#### Scenario: More labels than inputs

- **WHEN** the user provides more `--label` arguments than inputs
- **THEN** the CLI SHALL display an error stating the counts do not match
- **AND** the CLI SHALL exit with a non-zero status code

#### Scenario: Several labels in positional mode

- **WHEN** the user provides positional arguments with more than one `--label`
- **THEN** the CLI SHALL display an error
- **AND** the CLI SHALL exit with a non-zero status code

#### Scenario: Label starting with a dash

- **WHEN** the user provides `--label=-Dupont`
- **THEN** the input SHALL have the label `-Dupont`

#### Scenario: No label given

- **WHEN** the user provides no `--label` argument
- **THEN** every input SHALL have an empty label
