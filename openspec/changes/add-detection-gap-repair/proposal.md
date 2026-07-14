# Change: Repair detection gaps with targeted high-resolution inference

## Why

On 4K footage, default-resolution YOLO inference (640 px) misses climbers in their
crouched start position: the first ~13 frames of a real video have no detection at
all, so the crop falls back to the first detected position and the climber starts
off-center. High-resolution inference (1280 px) detects them from frame 0 but is
3.3× slower — too costly to apply to every frame.

## What Changes

- After track selection, frames where the selected track has no detection (start,
  interior gaps, end) are re-analyzed at high resolution (1280 px)
- Recovered detections are attached to the track only if they are within the
  gap-scaled jump distance of the nearest known track position (no teleporting),
  walking outward from known frames so repairs chain along the trajectory
- `PoseDetector.detect_all_persons` gains an optional inference-size parameter
- Cost is proportional to the number of missing frames only (~10 s on the reported
  video instead of +5 min for full high-resolution)

## Impact

- Affected specs: `pose-detection` (new requirement)
- Affected code: `src/railcam/tracking.py` (repair logic, pure with injected
  detector), `src/railcam/pose.py` (imgsz parameter), `src/railcam/cli.py`
  (repair pass after track selection)
