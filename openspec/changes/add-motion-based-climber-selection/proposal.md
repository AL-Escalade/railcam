# Change: Select the climber by motion instead of per-frame position

## Why

Static bystanders defeat the current per-frame climber selection: on a real video
(dual-lane wall with a person standing at the bottom), the bystander is the only
person detected on the first frame, gets selected regardless of the selector, and
proximity tracking then locks onto them for the whole video. Even `--climber left`
can pick a bystander standing further left than the left-lane climber.

## What Changes

- Pose analysis collects all detected persons per frame first, then builds motion
  tracks by frame-to-frame proximity association
- Tracks whose vertical displacement is below a threshold (static persons) are
  excluded; `left`/`right`/`auto` selectors apply among the remaining climbing
  tracks (with fallback to all tracks when nothing moves enough)
- The per-frame `select_climber` initial-pick + proximity logic is replaced by
  track building and track selection (**BREAKING** for `railcam.pose.select_climber`
  internal API; CLI flags and behavior contract are unchanged for normal footage)
- No GUI change: the GUI invokes the CLI and benefits automatically

## Impact

- Affected specs: `pose-detection` (climber selection and tracking requirements)
- Affected code: new `src/railcam/tracking.py` (pure, tested), `src/railcam/cli.py`
  detection loop, `src/railcam/pose.py` (select_climber retired or delegated)
