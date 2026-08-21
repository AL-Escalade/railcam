# multi-video Specification

## Purpose
TBD - created by archiving change add-mp4-sidebyside-zoom. Update Purpose after archive.
## Requirements
### Requirement: Side-by-Side Video Composition
The system SHALL support combining multiple videos horizontally into a single output.

#### Scenario: Two videos side-by-side
- **WHEN** the user specifies `--input video1.mp4:100:250 --input video2.mp4:50:200`
- **THEN** the output SHALL display both videos side-by-side horizontally
- **AND** video1 SHALL be on the left and video2 on the right

#### Scenario: Three or more videos
- **WHEN** the user specifies three or more `--input` arguments
- **THEN** all videos SHALL be displayed side-by-side in the order specified

### Requirement: Input Specification Format
The system SHALL accept video inputs using the `--input path:start:end` format.

#### Scenario: Valid input specification
- **WHEN** the user specifies `--input /path/to/video.mp4:100:250`
- **THEN** the system SHALL extract frames 100 to 250 (inclusive) from the specified video

#### Scenario: Invalid input format
- **WHEN** the user specifies `--input video.mp4` without frame range
- **THEN** the system SHALL display an error message explaining the required format
- **AND** the system SHALL exit with a non-zero status code

#### Scenario: Mutual exclusivity with positional arguments
- **WHEN** the user specifies both `--input` and positional video/frame arguments
- **THEN** the system SHALL display an error about mutually exclusive options
- **AND** the system SHALL exit with a non-zero status code

### Requirement: Frame Count Synchronization
The system SHALL synchronize frame counts across all input videos.

#### Scenario: Videos with different frame counts
- **WHEN** video A has 100 frames and video B has 150 frames
- **THEN** the output SHALL have 150 frames
- **AND** video A frames SHALL be resampled (duplicated) to match 150 frames

#### Scenario: Videos with equal frame counts
- **WHEN** all input videos have the same frame count
- **THEN** frames SHALL be composed one-to-one without resampling

### Requirement: Uniform Output Height
The system SHALL ensure all composed videos have the same height in the output.

#### Scenario: Videos with different source resolutions
- **WHEN** composing videos with different original resolutions
- **THEN** all videos SHALL be scaled to the same output height
- **AND** the aspect ratio (5:3 vertical) SHALL be maintained for each video

### Requirement: Sequential Per-Video Processing

The system SHALL fully process each input video (analysis then cropping) before
starting the next one, and SHALL release a video's decoded source frames once
its frames have been cropped.

#### Scenario: Two videos processed in sequence

- **WHEN** the user specifies two `--input` arguments
- **THEN** the first video SHALL be analyzed and cropped before the second video
  is analyzed
- **AND** the decoded source frames of the first video SHALL be released before
  the second video is analyzed

#### Scenario: Cropped output is unchanged

- **WHEN** the same inputs are processed sequentially rather than in two phases
- **THEN** the cropped frames SHALL be identical to those produced by the
  previous two-phase processing

#### Scenario: Source frames released before composition

- **WHEN** all videos have been cropped
- **THEN** no decoded source frame SHALL be retained while frames are composed
  and encoded

### Requirement: Streamed Composition

The system SHALL compose multi-video output one frame at a time, drawing each
video's cropped frames from a stream rather than from a materialized list.

#### Scenario: One composed frame at a time

- **WHEN** several videos are composed
- **THEN** the system SHALL emit composed frames one at a time
- **AND** SHALL hold at most one cropped frame per video at any moment

#### Scenario: Slower video repeats its frame

- **WHEN** the time synchronization maps two consecutive output frames to the
  same source frame of a video
- **THEN** that video's frame SHALL be reused for both output frames
- **AND** the video's stream SHALL NOT be advanced between them

#### Scenario: Shorter video freezes on its last frame

- **WHEN** one video's range ends before the output duration
- **THEN** its last cropped frame SHALL be repeated for the remaining output
  frames

#### Scenario: Composed output matches the list-based result

- **WHEN** the same inputs are composed by streaming and by the list-based path
- **THEN** the composed frames SHALL be identical

### Requirement: Non-Decreasing Synchronization Indices

The system SHALL require the frame indices produced by time synchronization to
be non-decreasing, since streamed composition can only move forward.

#### Scenario: Indices move forward or stay

- **WHEN** synchronization indices are computed for any input
- **THEN** each index SHALL be greater than or equal to the previous one

#### Scenario: A backwards index is rejected

- **WHEN** a stream is asked for a frame index lower than the one it last served
- **THEN** the system SHALL raise an error rather than return a wrong frame

### Requirement: Labeled Video Composition

The composed output SHALL place each video's label under that video's image, so
that a side-by-side comparison names each climber.

#### Scenario: Labels follow their video

- **WHEN** two labeled videos are composed side by side
- **THEN** each label SHALL appear under the video it was given for
- **AND** each label SHALL be centered on the width of its own video, not on the
  composed frame

