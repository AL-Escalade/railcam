## ADDED Requirements

### Requirement: Second Label Option

The CLI SHALL accept a repeatable `--sublabel` argument giving the text drawn
under a video's label, paired with the inputs in the order they are given.

#### Scenario: Name and time on two lines

- **WHEN** the user provides `--label "Zhao" --sublabel "4.704"`
- **THEN** the rendered video SHALL display `Zhao` and, below it in a smaller
  font, `4.704`

#### Scenario: Sublabels paired with inputs

- **WHEN** the user provides two `--input` arguments and two `--sublabel`
  arguments
- **THEN** the first sublabel SHALL apply to the first input and the second to
  the second input

#### Scenario: Fewer sublabels than inputs

- **WHEN** the user provides two `--input` arguments and one `--sublabel`
- **THEN** the remaining input SHALL have an empty second line

#### Scenario: More sublabels than inputs

- **WHEN** the user provides more `--sublabel` arguments than inputs
- **THEN** the CLI SHALL display an error stating the counts do not match
- **AND** the CLI SHALL exit with a non-zero status code

#### Scenario: Several sublabels in positional mode

- **WHEN** the user provides positional arguments with more than one `--sublabel`
- **THEN** the CLI SHALL display an error
- **AND** the CLI SHALL exit with a non-zero status code
