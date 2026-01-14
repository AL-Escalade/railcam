## Context

The current implementation uses YOLOv8-pose which already detects all persons in a frame. However, `detect_pelvis()` only returns the first (most confident) person's pelvis. For multi-climber videos, we need to:
1. Process all detected persons
2. Filter to those with visible pelvis
3. Select the correct one based on user preference

## Goals / Non-Goals

- Goals:
  - Support videos with 2 climbers on dual lanes
  - Filter out bystanders who don't have visible hips
  - Track the selected climber consistently across frames
  - Maintain backward compatibility (single climber videos work without changes)

- Non-Goals:
  - Tracking more than 2 climbers
  - Advanced person re-identification (using appearance features)
  - Handling climbers who cross paths mid-video

## Decisions

### Decision 1: Climber Selection Strategy

**What**: Use initial frame selection + proximity tracking.

**Why**:
- On the first valid frame, identify the climber by their pelvis X position (leftmost or rightmost)
- On subsequent frames, track by proximity (closest pelvis to previous position)
- This is simple, robust, and handles the common case where climbers stay in their lanes

**Alternatives considered**:
- Frame-by-frame position selection: Would cause jumpy tracking if climbers briefly swap positions
- Appearance-based tracking: Overkill for speed climbing where climbers stay in lanes

### Decision 2: Input Format Extension

**What**: Extend input spec to `path:start:end[:selector]` where selector is `left` or `right`.

**Why**:
- Follows existing pattern of colon-separated values
- Optional parameter maintains backward compatibility
- Intuitive naming (left/right matches visual position)

### Decision 3: Pelvis-Based Filtering

**What**: Only consider persons with detected hip landmarks (confidence >= threshold).

**Why**:
- Bystanders at bottom of frame typically only have upper body visible
- Hip detection is reliable indicator that we're tracking the full climber
- Reuses existing confidence threshold mechanism

## Implementation Approach

### Changes to pose.py

1. New `detect_all_persons()` method that returns all detected persons with valid pelvis
2. Modify `DetectionResult` or create new `MultiPersonDetectionResult` dataclass
3. Add `ClimberSelector` enum (`LEFT`, `RIGHT`, `AUTO`)
4. Add `select_climber()` function that picks the correct person based on selector and previous position

### Changes to multi_video.py

1. Extend `VideoInput` dataclass to include `climber_selector: ClimberSelector | None`
2. Update `parse_input_spec()` to parse the optional fourth component

### Changes to cli.py

1. Pass climber selector to `analyze_video()`
2. Use selector during pose detection loop

## Risks / Trade-offs

- **Risk**: Climbers crossing paths mid-video will cause incorrect tracking
  - Mitigation: This is extremely rare in speed climbing (separate lanes)
  - Future: Could add detection of path crossing and warning

- **Trade-off**: Proximity tracking may briefly lose the climber if they move very fast between frames
  - Mitigation: Speed climbing videos are typically 30+ fps, movement between frames is small

## Open Questions

None - design is straightforward for the speed climbing use case.
