# Change: Add Multi-Climber Selection

## Why

Currently, the tool assumes each input video contains only one climber. Real-world speed climbing videos often capture two climbers side-by-side (dual lanes) and may include bystanders at the bottom of the frame. Users need a way to specify which climber to track ("left climber" or "right climber") so the algorithm can focus on the correct person.

## What Changes

- **NEW** Climber selection parameter in input specification (e.g., `video.mp4:100:250:left`)
- **NEW** Multi-person detection: detect all persons with visible pelvis (hip landmarks)
- **NEW** Initial frame climber identification: select leftmost/rightmost person with detected pelvis on first frame
- **NEW** Proximity-based tracking: track the selected climber across subsequent frames based on position proximity
- **MODIFIED** Pelvis filtering: only consider detections where hip landmarks are visible (filters out bystanders whose upper body may be visible but hips are out of frame)

## Impact

- Affected specs:
  - `pose-detection` (modified) - Multi-person detection, climber selection, pelvis-based filtering
  - `cli-interface` (modified) - Extended input format with climber selector
- Affected code:
  - `pose.py` - Detect all persons, filter by pelvis visibility, select by position
  - `multi_video.py` - Parse extended input format with climber selector
  - `cli.py` - Pass climber selector to pose detection

## Scope

This change extends the existing pose detection to handle multi-person scenarios:

1. **Extended input format**: `path:start:end[:climber]` where `climber` is `left` or `right` (optional, default: track the only person or the closest to center)
2. **Pelvis-based filtering**: Only consider persons whose hip landmarks (left_hip and/or right_hip) are detected above confidence threshold
3. **Initial selection**: On the first frame with valid detections, select the leftmost/rightmost person based on pelvis X coordinate
4. **Proximity tracking**: On subsequent frames, select the person whose pelvis is closest to the previous frame's position

## Success Criteria

- Videos with 2 climbers can specify which one to track
- Bystanders at frame bottom (without visible pelvis) are automatically ignored
- Selected climber is tracked consistently throughout the video
- Single-climber videos continue to work without specifying a selector
