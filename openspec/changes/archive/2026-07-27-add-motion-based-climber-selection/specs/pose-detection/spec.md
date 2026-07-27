# pose-detection Specification (delta)

## ADDED Requirements

### Requirement: Motion Track Building

The system SHALL group per-frame person detections into motion tracks by associating
each detection with the nearest existing track within a maximum jump distance, one
track per person, creating new tracks for unmatched detections.

#### Scenario: Static and moving persons form separate tracks

- **WHEN** a video contains a static person and a climber moving vertically
- **THEN** their detections SHALL be grouped into two distinct tracks

#### Scenario: Detection gap does not split a track

- **WHEN** a tracked person is not detected for a few frames and reappears near their
  last known position
- **THEN** the reappearing detections SHALL continue the same track

## MODIFIED Requirements

### Requirement: Climber Selection by Position

The system SHALL select the climber among motion tracks, excluding tracks whose
vertical displacement is below the climbing threshold or below half of the largest
climbing track's displacement (detection fragments); the selector value (left, right,
or auto) SHALL apply to the remaining climbing tracks by their mean horizontal
position. If no track passes the climbing threshold, all tracks SHALL be considered.

#### Scenario: Static bystander is never selected

- **WHEN** a video contains a static person and one climbing person
- **THEN** the climbing person's track is selected regardless of the selector value
- **AND** regardless of which person is detected first

#### Scenario: Left selector with two climbers

- **WHEN** selector is "left" and two climbing tracks exist
- **THEN** the track with the smallest mean pelvis X coordinate is selected

#### Scenario: Right selector with two climbers

- **WHEN** selector is "right" and two climbing tracks exist
- **THEN** the track with the largest mean pelvis X coordinate is selected

#### Scenario: Left selector ignores a detection fragment further left

- **WHEN** selector is "left" and a short-span detection fragment sits left of a full
  climbing track
- **THEN** the full climbing track is selected

#### Scenario: Left selector ignores a static person further left

- **WHEN** selector is "left" and a static person stands left of the left-lane climber
- **THEN** the left-lane climber's track is selected

#### Scenario: Auto selector with a single climbing track

- **WHEN** selector is "auto" and exactly one climbing track exists
- **THEN** that track is selected

#### Scenario: Auto selector with multiple climbing tracks

- **WHEN** selector is "auto" and multiple climbing tracks exist
- **THEN** the track whose mean X coordinate is closest to the frame center is selected

#### Scenario: No climbing track at all

- **WHEN** no track reaches the climbing displacement threshold
- **THEN** the selector SHALL be applied to all tracks as a fallback

### Requirement: Proximity-Based Tracking

The system SHALL derive the selected climber's per-frame positions from their motion
track: frames where the track has a detection use that detection, and frames without
one are marked for interpolation.

#### Scenario: Consistent positions across frames

- **WHEN** a track is selected as the climber
- **THEN** every frame's position SHALL come from that same track, never from another
  person

#### Scenario: Climber temporarily not detected

- **WHEN** the selected track has no detection for a frame
- **AND** other persons are detected in that frame
- **THEN** the frame SHALL be marked for interpolation instead of using another person
