## ADDED Requirements

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
