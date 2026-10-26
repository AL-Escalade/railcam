## MODIFIED Requirements

### Requirement: Render Execution

The GUI SHALL execute renders by running the `railcam` CLI as a subprocess with the displayed arguments, SHALL remain responsive during the render, SHALL show progress parsed from the CLI output, and SHALL surface failures with the CLI's error output. In a packaged application, the subprocess SHALL be the application's own executable running the CLI.

#### Scenario: Successful render

- **WHEN** the user starts a render with a valid configuration
- **THEN** the GUI SHALL show a progress bar and, on success, offer to open the generated file

#### Scenario: Render failure

- **WHEN** the CLI subprocess exits with an error
- **THEN** the GUI SHALL display the CLI's error output and return to an editable state

#### Scenario: GUI stays responsive

- **WHEN** a render is in progress
- **THEN** the user SHALL still be able to interact with the GUI, including cancelling the render

#### Scenario: Render from a packaged application

- **WHEN** the user starts a render from an installed railcam-gui with no `railcam` command or Python available
- **THEN** the render SHALL run through the application's own executable and succeed
