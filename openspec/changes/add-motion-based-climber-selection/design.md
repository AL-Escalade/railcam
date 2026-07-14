# Design: Motion-based climber selection

## Context

Speed-climbing footage regularly contains non-climbers (belayers, judges, spectators)
who are stationary. The climber's unmistakable signature is large vertical displacement.
Diagnosis on `6-36-CdM.MOV`: static person at (x=0.16, y=0.92) in every frame; two
climbers sweep y from ~0.9 to ~0.15. On the first frame only the static person is
detected, so every selector currently locks onto them via proximity tracking.

## Goals / Non-Goals

- Goals: select the intended climber despite static persons; keep `left`/`right`/`auto`
  semantics; keep detection single-pass (no extra YOLO inference)
- Non-Goals: multi-camera identity, appearance-based re-identification, GUI seeding
  (possible later complement)

## Decisions

- **Decision: two-phase selection — collect per-frame detections, then pick a track.**
  The detection loop already runs `detect_all_persons` per frame; we keep all persons
  instead of selecting immediately. Cost is memory for pelvis positions only (a few
  floats per person per frame), no extra inference.

- **Decision: greedy nearest-neighbor track association.**
  For each frame, persons are matched to existing tracks by distance to the track's
  last pelvis position, closest pairs first, one-to-one, with a maximum jump of 0.15
  (normalized). Unmatched persons start new tracks. Simple, deterministic, and pure —
  crossing-track swaps are acceptable because lanes are horizontally separated.

- **Decision: climbing filter on vertical span.**
  A track is "climbing" when `max(y) - min(y) >= 0.10` (normalized). Selectors apply
  among climbing tracks ordered by mean x: `left` = smallest, `right` = largest,
  `auto` = closest to frame center. If no track passes the filter (short clips, static
  test footage), fall back to all tracks so existing single-person behavior is kept.

- **Decision: per-frame output unchanged.**
  The chosen track is converted back to one `DetectionResult` per frame (frames where
  the track has no detection yield position None), so interpolation, smoothing, torso
  averaging and cropping are untouched.

## Risks / Trade-offs

- Track fragmentation (detection gaps split a climber into two tracks) → association
  matches against the track's last known position with no expiry, tolerating gaps;
  span threshold is low enough that fragments usually still qualify.
- Thresholds are heuristic → constants documented in `tracking.py`, unit-tested on
  synthetic trajectories; fallback keeps degraded behavior no worse than today.
- Camera panning adds apparent motion to static persons → pan affects everyone
  equally; climbers still dominate vertical span.

## Migration Plan

Purely internal: same CLI surface. `select_climber` per-frame API is removed from the
analysis path; any external caller can use `tracking.select_track` equivalents.

## Open Questions

None.
