## ADDED Requirements

### Requirement: Pose Model Selection

The CLI SHALL accept an optional `--model` argument selecting the size of the
YOLOv8-pose model used for detection.

#### Scenario: Model size specified

- **WHEN** the user provides `--model n`
- **THEN** the CLI SHALL load the `yolov8n-pose` model for detection

#### Scenario: No model specified

- **WHEN** the user does not provide `--model`
- **THEN** the CLI SHALL load the `yolov8s-pose` model

#### Scenario: Invalid model size

- **WHEN** the user provides a value outside `n`, `s`, `m`, `l`, `x`
- **THEN** the CLI SHALL display an error listing the accepted values
- **AND** the CLI SHALL exit with a non-zero status code

#### Scenario: Model applies to every input

- **WHEN** the user provides several `--input` arguments together with `--model n`
- **THEN** every video SHALL be analyzed with the same selected model
