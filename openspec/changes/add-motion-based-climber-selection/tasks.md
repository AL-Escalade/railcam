# Tasks: add-motion-based-climber-selection

## 1. Tracking module (pure, TDD)

- [x] 1.1 `tracking.py`: Track dataclass + `build_tracks` greedy nearest-neighbor
      association with max jump and gap tolerance + unit tests
- [x] 1.2 `select_track`: climbing filter on upward rise, left/right/auto on mean X,
      fallback to all tracks + unit tests (including the static-bystander scenarios)
- [x] 1.4 Rank tracks relative to the largest rise, with the climbing threshold demoted
      to a noise floor, so a short excerpt no longer drops every track at once + tests
- [x] 1.3 `track_to_detections`: chosen track → per-frame DetectionResult list + tests

## 2. Pipeline integration

- [x] 2.1 Rewire `cli._detect_poses_with_tracking` to collect detections then select
      via tracking; remove per-frame `select_climber` usage
- [x] 2.2 Verify on `6-36-CdM.MOV`: selected positions follow the climber, not the
      static person; full test suite, ruff, mypy green

## 3. Docs

- [x] 3.1 Validate change with `openspec validate --strict`
