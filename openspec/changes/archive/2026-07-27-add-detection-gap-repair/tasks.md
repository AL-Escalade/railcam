# Tasks: add-detection-gap-repair

## 1. Implementation

- [x] 1.1 `tracking.repair_track_gaps` with injected detect function + unit tests
      (head-gap chaining, distant-person rejection, no-gap short-circuit)
- [x] 1.2 `PoseDetector.detect_all_persons` optional `imgsz` parameter
- [x] 1.3 CLI: repair pass after track selection using cached frames; log recovered count
- [x] 1.4 Verify on 6-36-CdM.MOV (right climber covered from frame 0); suite/ruff/mypy green
- [x] 1.5 `openspec validate add-detection-gap-repair --strict`
