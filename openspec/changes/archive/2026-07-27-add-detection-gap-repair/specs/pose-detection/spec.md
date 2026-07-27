# pose-detection Specification (delta)

## ADDED Requirements

### Requirement: High-Resolution Gap Repair

The system SHALL re-analyze frames where the selected climber track has no detection
using higher-resolution inference, and SHALL attach a recovered detection to the track
only when it lies within the gap-scaled jump distance of the nearest known track
position, processing gap frames from the nearest known frames outward.

#### Scenario: Undetected start frames are recovered

- **WHEN** the selected track's first detection occurs after the start of the range
- **AND** high-resolution inference detects the climber near the track's first known
  position in the preceding frames
- **THEN** those detections SHALL be added to the track and used for cropping

#### Scenario: Distant person is not attached during repair

- **WHEN** high-resolution inference on a gap frame only detects persons beyond the
  allowed jump distance from the nearest known track position
- **THEN** the frame SHALL remain without detection and be handled by interpolation

#### Scenario: Repairs chain along the trajectory

- **WHEN** several consecutive frames before the first detection are repaired one by one
- **THEN** each repaired frame SHALL serve as the anchor for the next, following the
  climber's trajectory backwards

#### Scenario: Another track's person is never attached

- **WHEN** a recovered detection on a gap frame coincides with another track's
  detection on that same frame
- **THEN** it SHALL NOT be attached to the repaired track, even within the allowed
  jump distance

#### Scenario: Repair cost is bounded

- **WHEN** the selected track already has a detection for every frame
- **THEN** no high-resolution inference SHALL run
