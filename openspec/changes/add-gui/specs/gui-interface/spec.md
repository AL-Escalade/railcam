# gui-interface Specification (delta)

## ADDED Requirements

### Requirement: GUI Launch

The system SHALL provide a `railcam-gui` command that opens a local desktop window on macOS and Windows, installed through the optional `gui` extra (`pip install railcam[gui]`).

#### Scenario: Launch with GUI dependencies installed

- **WHEN** the user runs `railcam-gui` with the `gui` extra installed
- **THEN** a desktop window SHALL open without requiring any network access

#### Scenario: Launch without GUI dependencies

- **WHEN** the user runs `railcam-gui` without PySide6 installed
- **THEN** the command SHALL exit with a message explaining how to install the `gui` extra

### Requirement: Video Loading

The GUI SHALL allow the user to add one or more video files, each displayed as a side-by-side player with its metadata (resolution, FPS, frame count).

#### Scenario: Add a video

- **WHEN** the user adds a valid video file
- **THEN** a player SHALL appear showing the first frame, its timeline, and the video metadata

#### Scenario: Add an unreadable file

- **WHEN** the user adds a file that cannot be decoded as video
- **THEN** the GUI SHALL show an error message and SHALL NOT add a player

#### Scenario: Remove a video

- **WHEN** the user removes a loaded video
- **THEN** its player SHALL disappear and the equivalent CLI command SHALL update accordingly

### Requirement: Frame-Accurate Navigation

Each player SHALL provide frame-accurate navigation: a scrubbable timeline and keyboard stepping (±1 frame with arrow keys, ±10 frames with Shift+arrows), always displaying the exact requested frame.

#### Scenario: Scrub the timeline

- **WHEN** the user drags the timeline cursor
- **THEN** the displayed image SHALL update to the exact frame under the cursor and show its frame number

#### Scenario: Step by one frame

- **WHEN** the user presses an arrow key on a focused player
- **THEN** the displayed frame SHALL advance or rewind by exactly one frame, clamped to the video bounds

### Requirement: View Zoom and Pan

Each player SHALL allow zooming into the displayed frame (mouse wheel, centered on the
cursor) and panning the zoomed view (mouse drag), independently per video, so the user
can inspect a detail while stepping through frames. A double-click SHALL reset the view
to fit. The zoom and pan SHALL persist across frame navigation and preview playback.

#### Scenario: Zoom towards the cursor

- **WHEN** the user scrolls the mouse wheel over the frame display
- **THEN** the view SHALL zoom in or out keeping the point under the cursor stationary

#### Scenario: Pan while zoomed

- **WHEN** the view is zoomed in and the user drags the frame display
- **THEN** the visible region SHALL follow the drag, clamped to the frame bounds

#### Scenario: View persists across frames

- **WHEN** the user steps to another frame or plays synchronized preview while zoomed
- **THEN** the same zoomed region SHALL remain displayed for the new frames

#### Scenario: Reset the view

- **WHEN** the user double-clicks the frame display
- **THEN** the view SHALL return to fitting the whole frame

### Requirement: Frame Range Selection

Each player SHALL let the user set the start and end frames from the currently displayed frame, visually highlighting the selected range on the timeline.

#### Scenario: Set start frame

- **WHEN** the user clicks "set start" while a frame is displayed
- **THEN** that frame number SHALL become the video's start frame and the timeline SHALL highlight the range

#### Scenario: Invalid range

- **WHEN** the selected end frame is lower than or equal to the start frame
- **THEN** the GUI SHALL flag the player as invalid and disable rendering with an explanatory message

### Requirement: Climber Selection

Each player SHALL offer a climber selector with the values `auto`, `left`, and `right`, matching the CLI's per-input climber option.

#### Scenario: Select climber side

- **WHEN** the user selects `left` on a video
- **THEN** the equivalent CLI command SHALL include that video's input spec with the `left` climber selector

### Requirement: Synchronized Preview Playback

The GUI SHALL play all loaded videos simultaneously on a common clock, each starting at its own start frame, at a user-selectable slow-motion speed between 0.1× and 1×. The mapping SHALL be time-based (`frame = start_frame + round(t × fps)`) so videos with different FPS stay synchronized, and a video that reaches its end frame SHALL freeze on it while the others continue.

#### Scenario: Synchronized start

- **WHEN** the user starts synchronized playback
- **THEN** every video SHALL begin at its own start frame at the same instant

#### Scenario: Different frame rates

- **WHEN** loaded videos have different FPS values
- **THEN** playback SHALL remain time-synchronized using each video's own frame rate

#### Scenario: Video ends before the others

- **WHEN** a video reaches its end frame during synchronized playback
- **THEN** it SHALL freeze on that frame while remaining videos continue playing

#### Scenario: Play/pause shortcut

- **WHEN** the user presses the space bar
- **THEN** synchronized playback SHALL toggle between play and pause

### Requirement: Render Options and Equivalent Command

The GUI SHALL expose the remaining render options (output format mp4/gif, output height, output speed, debug overlay, output path) and SHALL display a live, copyable CLI command equivalent to the current configuration.

#### Scenario: Command reflects configuration

- **WHEN** the user changes any video range, climber selection, or render option
- **THEN** the displayed CLI command SHALL update immediately to reflect the exact configuration

#### Scenario: Copy the command

- **WHEN** the user clicks the copy button
- **THEN** the full CLI command SHALL be copied to the system clipboard

### Requirement: Render Execution

The GUI SHALL execute renders by running the `railcam` CLI as a subprocess with the displayed arguments, SHALL remain responsive during the render, SHALL show progress parsed from the CLI output, and SHALL surface failures with the CLI's error output.

#### Scenario: Successful render

- **WHEN** the user starts a render with a valid configuration
- **THEN** the GUI SHALL show a progress bar and, on success, offer to open the generated file

#### Scenario: Render failure

- **WHEN** the CLI subprocess exits with an error
- **THEN** the GUI SHALL display the CLI's error output and return to an editable state

#### Scenario: GUI stays responsive

- **WHEN** a render is in progress
- **THEN** the user SHALL still be able to interact with the GUI, including cancelling the render

### Requirement: Project Files

The GUI SHALL save and load sessions as versioned JSON project files containing the video list (path, start frame, end frame, climber) and render options. Video paths SHALL be stored relative to the project file when possible.

#### Scenario: Save and reload a project

- **WHEN** the user saves a project and later reopens it
- **THEN** all videos, frame ranges, climber selections, and render options SHALL be restored

#### Scenario: Missing video on load

- **WHEN** a project references a video file that no longer exists
- **THEN** the GUI SHALL mark that player as missing and offer to relocate the file without losing its settings

#### Scenario: Corrupt or unsupported project file

- **WHEN** the user opens a file that is not a valid project or has an unsupported version
- **THEN** the GUI SHALL show a clear error and SHALL NOT partially load the session
