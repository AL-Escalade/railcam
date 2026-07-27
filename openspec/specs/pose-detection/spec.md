# pose-detection Specification

## Purpose
TBD - created by archiving change add-speed-climbing-gif-cli. Update Purpose after archive.
## Requirements
### Requirement: Pelvis Detection

The system SHALL detect the pelvis position of the climber in each frame using pose estimation.

#### Scenario: Successful pelvis detection
- **WHEN** the climber is visible in a frame with clear hip landmarks
- **THEN** the system SHALL compute the pelvis position as the midpoint between left and right hip landmarks

#### Scenario: Single hip visible
- **WHEN** only one hip landmark is detected with sufficient confidence
- **THEN** the system SHALL use the visible hip position as the pelvis estimate

#### Scenario: No hips detected
- **WHEN** neither hip landmark is detected with sufficient confidence
- **THEN** the system SHALL mark the frame as requiring interpolation

### Requirement: Detection Confidence

The system SHALL use a confidence threshold to determine detection validity.

#### Scenario: High confidence detection
- **WHEN** hip landmarks are detected with visibility score >= 0.5
- **THEN** the system SHALL use the detected positions

#### Scenario: Low confidence detection
- **WHEN** hip landmarks are detected with visibility score < 0.5
- **THEN** the system SHALL treat the frame as having no valid detection

### Requirement: Position Interpolation

The system SHALL interpolate pelvis positions for frames with failed detection.

#### Scenario: Gap between valid detections
- **WHEN** one or more consecutive frames have no valid pelvis detection
- **AND** valid detections exist before and after the gap
- **THEN** the system SHALL linearly interpolate positions for the gap frames

#### Scenario: Gap at start of range
- **WHEN** the first frames have no valid detection but later frames do
- **THEN** the system SHALL use the first valid detection position for the initial frames

#### Scenario: Gap at end of range
- **WHEN** the last frames have no valid detection but earlier frames do
- **THEN** the system SHALL use the last valid detection position for the final frames

#### Scenario: No valid detections in entire range
- **WHEN** no frames in the specified range have valid pelvis detection
- **THEN** the system SHALL exit with an error indicating the climber could not be detected

### Requirement: Position Smoothing

The system SHALL apply smoothing to pelvis positions for fluid motion.

#### Scenario: Smoothing applied to positions
- **WHEN** processing detected and interpolated positions
- **THEN** the system SHALL apply exponential moving average smoothing to produce fluid motion

#### Scenario: Smoothing preserves general trajectory
- **WHEN** the climber moves from bottom to top of frame
- **THEN** the smoothed positions SHALL follow the same general trajectory without significant lag

### Requirement: Multi-Person Detection
The system SHALL detect all persons in a frame and return their pelvis positions.

#### Scenario: Two climbers with visible pelvis
- **WHEN** a frame contains two persons with visible hip landmarks
- **THEN** both pelvis positions are returned with their X coordinates

#### Scenario: Climber and bystander
- **WHEN** a frame contains one climber with visible hips and one bystander with only upper body visible
- **THEN** only the climber's pelvis position is returned

### Requirement: Pelvis Visibility Filtering
The system SHALL only consider persons whose hip landmarks (left_hip and/or right_hip) are detected with confidence above the threshold.

#### Scenario: Person with hips below confidence threshold
- **WHEN** a person is detected but both hip landmarks have confidence below threshold
- **THEN** this person is excluded from the detection results

#### Scenario: Person with one hip visible
- **WHEN** a person is detected with only one hip landmark above threshold
- **THEN** this person is included with pelvis estimated from the visible hip

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

### Requirement: Configurable Pose Model Size

The system SHALL allow the pose model size to be selected at startup, and SHALL
default to the `s` (small) variant.

#### Scenario: Default model size

- **WHEN** a pose detector is created without an explicit model size
- **THEN** the system SHALL load the `yolov8s-pose` weights

#### Scenario: Explicit model size

- **WHEN** a pose detector is created with model size `m`
- **THEN** the system SHALL load the `yolov8m-pose` weights

#### Scenario: Gap repair uses the selected model

- **WHEN** the selected climber's track has frames with no detection
- **THEN** the system SHALL run the high-resolution repair pass on those frames
- **AND** the repair SHALL use the same model size as the initial pass

### Requirement: Analysis Without Frame Retention

The system SHALL analyze a video without retaining its decoded source frames,
holding at most one frame at a time during the detection pass.

#### Scenario: Frames released during detection

- **WHEN** a video is analyzed
- **THEN** each decoded frame SHALL be released once it has been detected on
- **AND** the analysis result SHALL NOT carry decoded frames

#### Scenario: Gap repair reads only the frames it needs

- **WHEN** the selected climber's track has frames with no detection
- **THEN** the system SHALL read those frames by index from the source file
- **AND** SHALL NOT read the frames that already have a detection

#### Scenario: Nothing to repair

- **WHEN** the selected track has a detection in every frame
- **THEN** the system SHALL NOT re-open the source file for repair

